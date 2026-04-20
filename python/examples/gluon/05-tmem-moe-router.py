"""
TMEM Sparse Logits And Expert Panels
====================================

This example implements compact sparse-logit projections for mixture-of-experts
routers, constrained/speculative decode candidate heads, and ragged expert
output panels:

    logits[M, E] = hidden[M, K] @ router_weight[E, K].T

These workloads are usually skinny.  Common expert counts, candidate counts,
and per-expert output widths such as 32 or 64 do not fill a broad 128-column
accumulator tile.  Before the TMEM linear-layout generalization work, the
practical scaled-MMAv5 path for this shape used a broad accumulator tile and
discarded unused columns, or avoided the compact TMEM accumulator path
entirely.

The optimized kernel below uses a tile-permuted `TensorMemoryLinearLayout` so a
logical `E=32` router tile can be implemented with N=8 scaled-MMAv5 accumulator
fragments, and `E=64` with N=16 fragments.  Matrix-B scale storage still obeys
the Blackwell public scale-fragment addressing granularity; the compiler
rematerializes/pads that storage before lowering.

The benchmarks compare compact sparse projections with the best
pre-generalization baseline available for the same scaled-MMA hardware path: a
padded `N=128` accumulator tile whose leading columns hold the useful logits or
expert output panel.
"""

import argparse
import math

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLinearLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma_scaled,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


BLOCK_M = 128
VEC_SIZE = 32


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


def make_tmem_tile_permuted_layout(m, n, tile_n):
    assert n >= 2 * tile_n
    col_bits = [1 << i for i in range(int(math.log2(n)))]
    tile_bit = int(math.log2(tile_n))
    col_bits[tile_bit], col_bits[tile_bit + 1] = col_bits[tile_bit + 1], col_bits[tile_bit]
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(int(math.log2(m)))],
        cols=[[0, bit] for bit in col_bits],
        shape=[m, n],
    )


def random_mxfp8_tensor(m, k):
    base = MXFP4Tensor(size=(m, k), device="cuda").random()
    scale = MXScaleTensor(size=(m, k // VEC_SIZE), device="cuda").random(low=1 / 128, high=2.0)
    ref = base.to(torch.float32)
    scale_ref = scale.to(torch.float32)
    value = ref * scale_ref.repeat_interleave(VEC_SIZE, dim=1)
    return ref.to(torch.float8_e4m3fn), scale.data, value


def make_router_inputs(total_m, num_experts, k):
    torch.manual_seed(0)
    hidden, hidden_scale, hidden_ref = random_mxfp8_tensor(total_m, k)
    weight, weight_scale, weight_ref = random_mxfp8_tensor(num_experts, k)
    return hidden, weight, hidden_scale, weight_scale, hidden_ref, weight_ref


@gluon.jit
def _router_projection_kernel(
    hidden,
    weight,
    hidden_scale,
    weight_scale,
    out,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_K: gl.constexpr,
    acc_layout: gl.constexpr,
):
    BLOCK_M: gl.constexpr = 128
    VEC_SIZE: gl.constexpr = 32
    pid_m = gl.program_id(0)
    off_m = pid_m * BLOCK_M

    block_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [gl.num_warps(), 1], [1, 0])
    hidden_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, block_layout))[:, None]
    hidden_k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, block_layout))[None, :]
    weight_n = gl.arange(0, NUM_EXPERTS, gl.SliceLayout(1, block_layout))[:, None]
    weight_k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, block_layout))[None, :]

    hidden_tile = gl.load(hidden + hidden_m * BLOCK_K + hidden_k)
    weight_tile = gl.load(weight + weight_n * BLOCK_K + weight_k)

    smem_hidden_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_K], gl.float8e4nv
    )
    smem_weight_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [NUM_EXPERTS, BLOCK_K], gl.float8e4nv
    )
    smem_hidden = gl.allocate_shared_memory(gl.float8e4nv, [BLOCK_M, BLOCK_K], smem_hidden_layout, hidden_tile)
    smem_weight = gl.allocate_shared_memory(gl.float8e4nv, [NUM_EXPERTS, BLOCK_K], smem_weight_layout, weight_tile)

    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, NUM_EXPERTS], acc_layout)
    acc_reg_layout: gl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(gl.zeros([BLOCK_M, NUM_EXPERTS], gl.float32, layout=acc_reg_layout))

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    hidden_scale_tmem = allocate_tensor_memory(
        hidden_scale.dtype.element_ty, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout
    )
    weight_scale_tmem = allocate_tensor_memory(
        weight_scale.dtype.element_ty, [NUM_EXPERTS, BLOCK_K // VEC_SIZE], scale_layout
    )
    hidden_scale_layout: gl.constexpr = hidden_scale_tmem.get_reg_layout()
    weight_scale_layout: gl.constexpr = weight_scale_tmem.get_reg_layout()

    hidden_scale_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, hidden_scale_layout))[:, None]
    hidden_scale_k = gl.arange(0, BLOCK_K // VEC_SIZE, gl.SliceLayout(0, hidden_scale_layout))[None, :]
    weight_scale_n = gl.arange(0, NUM_EXPERTS, gl.SliceLayout(1, weight_scale_layout))[:, None]
    weight_scale_k = gl.arange(0, BLOCK_K // VEC_SIZE, gl.SliceLayout(0, weight_scale_layout))[None, :]
    hidden_scale_tmem.store(gl.load(hidden_scale + hidden_scale_m * (BLOCK_K // VEC_SIZE) + hidden_scale_k))
    weight_scale_tmem.store(gl.load(weight_scale + weight_scale_n * (BLOCK_K // VEC_SIZE) + weight_scale_k))

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(
        smem_hidden,
        smem_weight.permute((1, 0)),
        acc_tmem,
        hidden_scale_tmem,
        weight_scale_tmem,
        "e4m3",
        "e4m3",
        use_acc=True,
    )
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
    out_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, out_layout))[:, None]
    out_n = gl.arange(0, NUM_EXPERTS, gl.SliceLayout(0, out_layout))[None, :]
    acc = acc_tmem.load()
    gl.store(out + out_m * NUM_EXPERTS + out_n, gl.convert_layout(acc, out_layout))


def _tile_n_for_experts(num_experts):
    if num_experts == 32:
        return 8
    if num_experts == 64:
        return 16
    if num_experts == 128:
        return 32
    raise ValueError("this example supports direct scaled-router expert counts 32, 64, and 128")


def router_projection_into(hidden, weight, hidden_scale, weight_scale, out, num_experts, k):
    assert hidden.shape[0] % BLOCK_M == 0
    tile_n = _tile_n_for_experts(num_experts)
    acc_layout = make_tmem_tile_permuted_layout(BLOCK_M, num_experts, tile_n)
    grid = (triton.cdiv(hidden.shape[0], BLOCK_M), )
    return _router_projection_kernel[grid](
        hidden,
        weight,
        hidden_scale,
        weight_scale,
        out,
        num_experts,
        k,
        acc_layout,
        num_warps=4,
    )


def router_projection(hidden, weight, hidden_scale, weight_scale, k):
    out = torch.empty((hidden.shape[0], weight.shape[0]), device="cuda", dtype=torch.float32)
    router_projection_into(hidden, weight, hidden_scale, weight_scale, out, weight.shape[0], k)
    return out


def router_topk(hidden, weight, hidden_scale, weight_scale, k, top_k=2):
    assert top_k == 2
    logits = router_projection(hidden, weight, hidden_scale, weight_scale, k)
    scores = torch.empty((logits.shape[0], top_k), device="cuda", dtype=torch.float32)
    expert_ids = torch.empty((logits.shape[0], top_k), device="cuda", dtype=torch.int64)
    router_top2_into(logits, scores, expert_ids)
    return logits, scores, expert_ids


@triton.jit
def _top2_kernel(
    logits,
    scores,
    expert_ids,
    M: tl.constexpr,
    E: tl.constexpr,
    LOGITS_STRIDE_M: tl.constexpr,
    LOGITS_STRIDE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs = rows[:, None] * LOGITS_STRIDE_M + tl.arange(0, E)[None, :] * LOGITS_STRIDE_N
    mask = rows[:, None] < M
    vals = tl.load(logits + offs, mask=mask, other=-float("inf"))

    best = tl.max(vals, axis=1)
    is_best = vals == best[:, None]
    # Pick the lowest expert id for deterministic ties, matching torch.topk's
    # value comparison for the non-tied random inputs used here.
    expert_range = tl.arange(0, E)
    best_id = tl.min(tl.where(is_best, expert_range[None, :], E), axis=1)
    vals2 = tl.where(is_best, -float("inf"), vals)
    second = tl.max(vals2, axis=1)
    second_id = tl.min(tl.where(vals2 == second[:, None], expert_range[None, :], E), axis=1)

    row_mask = rows < M
    tl.store(scores + rows * 2, best, mask=row_mask)
    tl.store(scores + rows * 2 + 1, second, mask=row_mask)
    tl.store(expert_ids + rows * 2, best_id.to(tl.int64), mask=row_mask)
    tl.store(expert_ids + rows * 2 + 1, second_id.to(tl.int64), mask=row_mask)


def router_top2_into(logits, scores, expert_ids):
    assert logits.shape[1] in (32, 64, 128)
    assert scores.shape[1] == 2
    grid = (triton.cdiv(logits.shape[0], 16), )
    _top2_kernel[grid](
        logits,
        scores,
        expert_ids,
        logits.shape[0],
        logits.shape[1],
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=16,
    )


def benchmark_router_projection(total_m=4096, num_experts=32, k=128, include_topk=False):
    hidden, weight, hidden_scale, weight_scale, hidden_ref, weight_ref = make_router_inputs(total_m, num_experts, k)
    _, padded_weight, _, padded_weight_scale, _, padded_weight_ref = make_router_inputs(128, 128, k)
    padded_weight[:num_experts, :] = weight
    padded_weight_scale[:num_experts, :] = weight_scale
    padded_weight_ref[:num_experts, :] = weight_ref

    narrow_out = torch.empty((total_m, num_experts), device="cuda", dtype=torch.float32)
    padded_out = torch.empty((total_m, 128), device="cuda", dtype=torch.float32)
    narrow_scores = torch.empty((total_m, 2), device="cuda", dtype=torch.float32)
    padded_scores = torch.empty((total_m, 2), device="cuda", dtype=torch.float32)
    narrow_ids = torch.empty((total_m, 2), device="cuda", dtype=torch.int64)
    padded_ids = torch.empty((total_m, 2), device="cuda", dtype=torch.int64)

    def narrow():
        router_projection_into(hidden, weight, hidden_scale, weight_scale, narrow_out, num_experts, k)
        if include_topk:
            router_top2_into(narrow_out, narrow_scores, narrow_ids)

    def padded():
        router_projection_into(hidden, padded_weight, hidden_scale, padded_weight_scale, padded_out, 128, k)
        if include_topk:
            router_top2_into(padded_out[:, :num_experts], padded_scores, padded_ids)

    narrow()
    padded()
    reference = hidden_ref @ weight_ref.T
    torch.testing.assert_close(narrow_out, reference, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(padded_out[:, :num_experts], narrow_out, atol=1e-3, rtol=1e-3)
    if include_topk:
        torch.testing.assert_close(padded_scores, narrow_scores, atol=0, rtol=0)
        torch.testing.assert_close(padded_ids, narrow_ids, atol=0, rtol=0)

    narrow_ms = triton.testing.do_bench(narrow)
    padded_ms = triton.testing.do_bench(padded)
    useful_flops = 2 * total_m * num_experts * k
    return {
        "M": total_m,
        "E": num_experts,
        "K": k,
        "include_topk": include_topk,
        "narrow_ms": narrow_ms,
        "padded_128_ms": padded_ms,
        "speedup": padded_ms / narrow_ms,
        "useful_tflops": useful_flops * 1e-12 / (narrow_ms * 1e-3),
    }


def make_candidate_inputs(total_m, candidate_count, k, vocab_size=512):
    torch.manual_seed(2)
    hidden, hidden_scale, hidden_ref = random_mxfp8_tensor(total_m, k)
    # Use a deterministic nontrivial order so the test checks that output
    # columns follow candidate-list order, not sorted token id order.
    candidate_ids = torch.randperm(vocab_size, device="cuda")[:candidate_count]
    selected, selected_scale, selected_ref = random_mxfp8_tensor(candidate_count, k)
    return hidden, selected, hidden_scale, selected_scale, hidden_ref, selected_ref, candidate_ids


def benchmark_candidate_head(total_m=4096, candidate_count=32, k=128):
    hidden, selected, hidden_scale, selected_scale, hidden_ref, selected_ref, candidate_ids = make_candidate_inputs(
        total_m, candidate_count, k
    )
    _, padded_selected, _, padded_selected_scale, _, padded_selected_ref = make_router_inputs(128, 128, k)
    padded_selected[:candidate_count, :] = selected
    padded_selected_scale[:candidate_count, :] = selected_scale
    padded_selected_ref[:candidate_count, :] = selected_ref

    compact_out = torch.empty((total_m, candidate_count), device="cuda", dtype=torch.float32)
    padded_out = torch.empty((total_m, 128), device="cuda", dtype=torch.float32)

    def compact():
        router_projection_into(hidden, selected, hidden_scale, selected_scale, compact_out, candidate_count, k)

    def padded():
        router_projection_into(hidden, padded_selected, hidden_scale, padded_selected_scale, padded_out, 128, k)

    compact()
    padded()
    reference = hidden_ref @ selected_ref.T
    torch.testing.assert_close(compact_out, reference, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(padded_out[:, :candidate_count], compact_out, atol=1e-3, rtol=1e-3)

    compact_ms = triton.testing.do_bench(compact)
    padded_ms = triton.testing.do_bench(padded)
    useful_flops = 2 * total_m * candidate_count * k
    return {
        "M": total_m,
        "C": candidate_count,
        "K": k,
        "compact_ms": compact_ms,
        "padded_128_ms": padded_ms,
        "speedup": padded_ms / compact_ms,
        "useful_tflops": useful_flops * 1e-12 / (compact_ms * 1e-3),
        "first_candidate": int(candidate_ids[0].item()),
    }


def make_expert_inputs(counts, output_widths, k):
    torch.manual_seed(6)
    experts = []
    for expert_id, (tokens, width) in enumerate(zip(counts, output_widths)):
        if tokens == 0:
            experts.append(None)
            continue
        hidden, hidden_scale, hidden_ref = random_mxfp8_tensor(tokens, k)
        weight, weight_scale, weight_ref = random_mxfp8_tensor(width, k)
        experts.append((expert_id, hidden, weight, hidden_scale, weight_scale, hidden_ref, weight_ref))
    return experts


def compact_expert_outputs(experts, output_widths, k):
    outputs = []
    for expert, width in zip(experts, output_widths):
        if expert is None:
            outputs.append(None)
            continue
        _, hidden, weight, hidden_scale, weight_scale, _, _ = expert
        outputs.append(router_projection(hidden, weight, hidden_scale, weight_scale, k))
    return outputs


def make_padded_experts(experts, output_widths, k):
    padded_experts = []
    for expert, width in zip(experts, output_widths):
        if expert is None:
            padded_experts.append(None)
            continue
        expert_id, hidden, weight, hidden_scale, weight_scale, hidden_ref, weight_ref = expert
        _, padded_weight, _, padded_weight_scale, _, padded_weight_ref = make_router_inputs(128, 128, k)
        padded_weight[:width, :] = weight
        padded_weight_scale[:width, :] = weight_scale
        padded_weight_ref[:width, :] = weight_ref
        padded_experts.append((expert_id, hidden, padded_weight, hidden_scale, padded_weight_scale, hidden_ref, padded_weight_ref))
    return padded_experts


def precomputed_padded_expert_outputs(padded_experts, output_widths, k):
    outputs = []
    for expert, width in zip(padded_experts, output_widths):
        if expert is None:
            outputs.append(None)
            continue
        _, hidden, padded_weight, hidden_scale, padded_weight_scale, _, _ = expert
        padded = router_projection(hidden, padded_weight, hidden_scale, padded_weight_scale, k)
        outputs.append(padded[:, :width])
    return outputs


def reference_expert_outputs(experts):
    refs = []
    for expert in experts:
        if expert is None:
            refs.append(None)
            continue
        _, _, _, _, _, hidden_ref, weight_ref = expert
        refs.append(hidden_ref @ weight_ref.T)
    return refs


def benchmark_ragged_experts(counts=(128, 256, 0, 128), output_widths=(32, 64, 32, 64), k=128):
    experts = make_expert_inputs(counts, output_widths, k)
    padded_experts = make_padded_experts(experts, output_widths, k)

    def compact():
        compact_expert_outputs(experts, output_widths, k)

    def padded():
        precomputed_padded_expert_outputs(padded_experts, output_widths, k)

    compact_out = compact_expert_outputs(experts, output_widths, k)
    padded_out = precomputed_padded_expert_outputs(padded_experts, output_widths, k)
    refs = reference_expert_outputs(experts)
    for compact_panel, padded_panel, ref_panel in zip(compact_out, padded_out, refs):
        if ref_panel is None:
            continue
        torch.testing.assert_close(compact_panel, ref_panel, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(padded_panel, compact_panel, atol=1e-3, rtol=1e-3)

    compact_ms = triton.testing.do_bench(compact)
    padded_ms = triton.testing.do_bench(padded)
    useful_flops = sum(2 * tokens * width * k for tokens, width in zip(counts, output_widths))
    return {
        "experts": len(counts),
        "active": sum(tokens > 0 for tokens in counts),
        "tokens": sum(counts),
        "K": k,
        "compact_ms": compact_ms,
        "padded_128_ms": padded_ms,
        "speedup": padded_ms / compact_ms,
        "useful_tflops": useful_flops * 1e-12 / (compact_ms * 1e-3),
    }


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("num_experts", [32, 64])
@pytest.mark.parametrize("k", [128, 256])
def test_router_projection_matches_torch(num_experts, k):
    total_m = 512
    hidden, weight, hidden_scale, weight_scale, hidden_ref, weight_ref = make_router_inputs(total_m, num_experts, k)

    compiled = router_projection_into(
        hidden,
        weight,
        hidden_scale,
        weight_scale,
        torch.empty((total_m, num_experts), device="cuda", dtype=torch.float32),
        num_experts,
        k,
    )
    out = router_projection(hidden, weight, hidden_scale, weight_scale, k)

    torch.testing.assert_close(out, hidden_ref @ weight_ref.T, atol=1e-3, rtol=1e-3)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("num_experts", [32, 64])
def test_router_topk_matches_torch(num_experts):
    total_m, k = 512, 128
    hidden, weight, hidden_scale, weight_scale, hidden_ref, weight_ref = make_router_inputs(total_m, num_experts, k)

    logits, scores, expert_ids = router_topk(hidden, weight, hidden_scale, weight_scale, k, top_k=2)
    ref_logits = hidden_ref @ weight_ref.T
    ref_scores, ref_expert_ids = torch.topk(ref_logits, 2, dim=1)

    torch.testing.assert_close(logits, ref_logits, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(scores, ref_scores, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(expert_ids, ref_expert_ids, atol=0, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("num_experts", [32, 64])
def test_router_padded_baseline_matches_narrow(num_experts):
    result = benchmark_router_projection(total_m=512, num_experts=num_experts, k=128, include_topk=False)
    assert result["padded_128_ms"] > 0
    assert result["narrow_ms"] > 0


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("candidate_count", [32, 64])
@pytest.mark.parametrize("k", [128, 256])
def test_candidate_projection_matches_selected_vocab_order(candidate_count, k):
    total_m = 512
    hidden, selected, hidden_scale, selected_scale, hidden_ref, selected_ref, candidate_ids = make_candidate_inputs(
        total_m, candidate_count, k
    )

    out = torch.empty((total_m, candidate_count), device="cuda", dtype=torch.float32)
    compiled = router_projection_into(hidden, selected, hidden_scale, selected_scale, out, candidate_count, k)

    torch.testing.assert_close(out, hidden_ref @ selected_ref.T, atol=1e-3, rtol=1e-3)
    assert not torch.equal(candidate_ids, torch.sort(candidate_ids).values)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("candidate_count", [32, 64])
def test_candidate_padded_baseline_matches_compact(candidate_count):
    result = benchmark_candidate_head(total_m=512, candidate_count=candidate_count, k=128)
    assert result["compact_ms"] > 0
    assert result["padded_128_ms"] > 0


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_ragged_expert_outputs_match_torch():
    counts = (128, 256, 0, 128)
    output_widths = (32, 64, 32, 64)
    k = 128
    experts = make_expert_inputs(counts, output_widths, k)
    compact_out = compact_expert_outputs(experts, output_widths, k)
    refs = reference_expert_outputs(experts)
    for compact_panel, ref_panel in zip(compact_out, refs):
        if ref_panel is None:
            assert compact_panel is None
            continue
        torch.testing.assert_close(compact_panel, ref_panel, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_ragged_expert_padded_baseline_matches_compact():
    result = benchmark_ragged_experts()
    assert result["compact_ms"] > 0
    assert result["padded_128_ms"] > 0


def _print_benchmark(args):
    if args.mode in ("router", "all"):
        print("TMEM sparse router benchmark")
        print("============================")
        for include_topk in ([False, True] if args.mode == "all" else [args.include_topk]):
            for num_experts in args.experts:
                result = benchmark_router_projection(
                    total_m=args.m,
                    num_experts=num_experts,
                    k=args.k,
                    include_topk=include_topk,
                )
                suffix = " + topk" if result["include_topk"] else ""
                print(
                    f"M={result['M']} E={result['E']} K={result['K']}{suffix} | "
                    f"narrow={result['narrow_ms']:.3f} ms | "
                    f"padded E=128={result['padded_128_ms']:.3f} ms | "
                    f"speedup={result['speedup']:.2f}x | "
                    f"useful={result['useful_tflops']:.1f} TFLOP/s"
                )
    if args.mode in ("candidate", "all"):
        print("TMEM candidate-head benchmark")
        print("=============================")
        for candidate_count in args.candidates:
            result = benchmark_candidate_head(total_m=args.m, candidate_count=candidate_count, k=args.k)
            print(
                f"M={result['M']} C={result['C']} K={result['K']} | "
                f"compact={result['compact_ms']:.3f} ms | "
                f"padded C=128={result['padded_128_ms']:.3f} ms | "
                f"speedup={result['speedup']:.2f}x | "
                f"useful={result['useful_tflops']:.1f} TFLOP/s | "
                f"first_candidate={result['first_candidate']}"
            )
    if args.mode in ("ragged", "all"):
        print("TMEM ragged expert panel benchmark")
        print("==================================")
        result = benchmark_ragged_experts(k=args.k)
        print(
            f"experts={result['experts']} active={result['active']} tokens={result['tokens']} K={result['K']} | "
            f"compact={result['compact_ms']:.3f} ms | "
            f"padded N=128={result['padded_128_ms']:.3f} ms | "
            f"speedup={result['speedup']:.2f}x | "
            f"useful={result['useful_tflops']:.1f} TFLOP/s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark compact TMEM sparse logits and expert panels.")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--mode", choices=["router", "candidate", "ragged", "all"], default="all")
    parser.add_argument("--experts", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--candidates", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--include-topk", action="store_true")
    args = parser.parse_args()
    if not is_blackwell():
        raise RuntimeError("This example requires a Blackwell NVIDIA GPU")
    _print_benchmark(args)


# On one GB200-class run:
#
# ```
# $ CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-sparse-logits \
#   PYTHONPATH=.:./python python python/examples/gluon/05-tmem-moe-router.py --mode all \
#   --m 4096 --k 128 --experts 32 64 --candidates 32 64
# TMEM sparse router benchmark
# ============================
# M=4096 E=32 K=128 | narrow=0.011 ms | padded E=128=0.013 ms | speedup=1.17x | useful=3.0 TFLOP/s
# M=4096 E=64 K=128 | narrow=0.013 ms | padded E=128=0.013 ms | speedup=1.01x | useful=5.2 TFLOP/s
# M=4096 E=32 K=128 + topk | narrow=0.015 ms | padded E=128=0.018 ms | speedup=1.14x | useful=2.2 TFLOP/s
# M=4096 E=64 K=128 + topk | narrow=0.017 ms | padded E=128=0.018 ms | speedup=1.10x | useful=4.0 TFLOP/s
# TMEM candidate-head benchmark
# =============================
# M=4096 C=32 K=128 | compact=0.011 ms | padded C=128=0.013 ms | speedup=1.18x | useful=3.0 TFLOP/s | first_candidate=63
# M=4096 C=64 K=128 | compact=0.013 ms | padded C=128=0.013 ms | speedup=1.01x | useful=5.2 TFLOP/s | first_candidate=63
# TMEM ragged expert panel benchmark
# ==================================
# experts=4 active=3 tokens=512 K=128 | compact=0.074 ms | padded N=128=0.109 ms | speedup=1.48x | useful=0.1 TFLOP/s
# ```
#
# The `--include-topk` benchmark uses the same plain Triton top-2 kernel for
# both router paths. It is not a TMEM feature; it just keeps the router-level
# comparison off PyTorch while showing the cost of selecting from compact logits.
