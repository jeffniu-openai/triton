"""
TMEM Skinny Side Projections
============================

Many model blocks carry a small side projection next to a larger path.  Two
common examples are LoRA adapters and gated/control projections:

    tmp[M, R] = x[M, K] @ down[R, K].T
    y[M, N] = base[M, N] + alpha * tmp[M, R] @ up[N, R].T

    side[M, S] = x[M, K] @ side_w[S, K].T
    out[:, :S] = broad[:, :S] * sigmoid(side)

These side projections are skinny.  Before the TMEM linear-layout work, a
block-scaled TMEM implementation commonly had to form a broad rank/side-128
intermediate and slice the useful columns back out.  This example keeps the
side intermediate compact for `32` and `64` columns using a tile-permuted
`TensorMemoryLinearLayout` accumulator.  Plain Triton kernels implement the
non-TMEM update/gate stages in both benchmark paths, so the comparisons stay
off PyTorch while still checking the full output.
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


def make_lora_inputs(total_m, rank, k, n):
    torch.manual_seed(1)
    x, x_scale, x_ref = random_mxfp8_tensor(total_m, k)
    down, down_scale, down_ref = random_mxfp8_tensor(rank, k)
    up = torch.randn((n, rank), device="cuda", dtype=torch.float32) / math.sqrt(rank)
    base = torch.randn((total_m, n), device="cuda", dtype=torch.float32)
    return x, down, x_scale, down_scale, x_ref, down_ref, up, base


def make_gate_inputs(total_m, side_n, k, broad_n):
    torch.manual_seed(4)
    x, x_scale, x_ref = random_mxfp8_tensor(total_m, k)
    side_w, side_scale, side_ref = random_mxfp8_tensor(side_n, k)
    broad_w = torch.randn((broad_n, k), device="cuda", dtype=torch.float32) / (k**0.5)
    return x, side_w, x_scale, side_scale, x_ref, side_ref, broad_w


@gluon.jit
def _lora_down_projection_kernel(
    x,
    down,
    x_scale,
    down_scale,
    tmp,
    RANK: gl.constexpr,
    BLOCK_K: gl.constexpr,
    acc_layout: gl.constexpr,
):
    BLOCK_M: gl.constexpr = 128
    VEC_SIZE: gl.constexpr = 32
    pid_m = gl.program_id(0)
    off_m = pid_m * BLOCK_M

    block_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [gl.num_warps(), 1], [1, 0])
    x_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, block_layout))[:, None]
    x_k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, block_layout))[None, :]
    down_r = gl.arange(0, RANK, gl.SliceLayout(1, block_layout))[:, None]
    down_k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, block_layout))[None, :]

    x_tile = gl.load(x + x_m * BLOCK_K + x_k)
    down_tile = gl.load(down + down_r * BLOCK_K + down_k)
    smem_x_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float8e4nv)
    smem_down_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([RANK, BLOCK_K], gl.float8e4nv)
    smem_x = gl.allocate_shared_memory(gl.float8e4nv, [BLOCK_M, BLOCK_K], smem_x_layout, x_tile)
    smem_down = gl.allocate_shared_memory(gl.float8e4nv, [RANK, BLOCK_K], smem_down_layout, down_tile)

    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, RANK], acc_layout)
    acc_reg_layout: gl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(gl.zeros([BLOCK_M, RANK], gl.float32, layout=acc_reg_layout))

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    x_scale_tmem = allocate_tensor_memory(x_scale.dtype.element_ty, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout)
    down_scale_tmem = allocate_tensor_memory(down_scale.dtype.element_ty, [RANK, BLOCK_K // VEC_SIZE], scale_layout)
    x_scale_layout: gl.constexpr = x_scale_tmem.get_reg_layout()
    down_scale_layout: gl.constexpr = down_scale_tmem.get_reg_layout()

    x_scale_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, x_scale_layout))[:, None]
    x_scale_k = gl.arange(0, BLOCK_K // VEC_SIZE, gl.SliceLayout(0, x_scale_layout))[None, :]
    down_scale_r = gl.arange(0, RANK, gl.SliceLayout(1, down_scale_layout))[:, None]
    down_scale_k = gl.arange(0, BLOCK_K // VEC_SIZE, gl.SliceLayout(0, down_scale_layout))[None, :]
    x_scale_tmem.store(gl.load(x_scale + x_scale_m * (BLOCK_K // VEC_SIZE) + x_scale_k))
    down_scale_tmem.store(gl.load(down_scale + down_scale_r * (BLOCK_K // VEC_SIZE) + down_scale_k))

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(
        smem_x,
        smem_down.permute((1, 0)),
        acc_tmem,
        x_scale_tmem,
        down_scale_tmem,
        "e4m3",
        "e4m3",
        use_acc=True,
    )
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
    out_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, out_layout))[:, None]
    out_r = gl.arange(0, RANK, gl.SliceLayout(0, out_layout))[None, :]
    acc = acc_tmem.load()
    gl.store(tmp + out_m * RANK + out_r, gl.convert_layout(acc, out_layout))


def _tile_n_for_rank(rank):
    if rank == 32:
        return 8
    if rank == 64:
        return 16
    if rank == 128:
        return 32
    raise ValueError("this example supports direct scaled LoRA ranks 32, 64, and 128")


def lora_down_projection_into(x, down, x_scale, down_scale, tmp, rank, k):
    assert x.shape[0] % BLOCK_M == 0
    layout = make_tmem_tile_permuted_layout(BLOCK_M, rank, _tile_n_for_rank(rank))
    grid = (triton.cdiv(x.shape[0], BLOCK_M), )
    return _lora_down_projection_kernel[grid](
        x,
        down,
        x_scale,
        down_scale,
        tmp,
        rank,
        k,
        layout,
        num_warps=4,
    )


def lora_update(x, down, x_scale, down_scale, up, base, alpha, k):
    rank = down.shape[0]
    tmp = torch.empty((x.shape[0], rank), device="cuda", dtype=torch.float32)
    out = torch.empty_like(base)
    lora_down_projection_into(x, down, x_scale, down_scale, tmp, rank, k)
    lora_update_from_tmp(tmp, up, base, out, alpha)
    return out


@triton.jit
def _lora_update_kernel(
    tmp,
    up,
    base,
    out,
    M: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    TMP_STRIDE_M: tl.constexpr,
    TMP_STRIDE_R: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)
    a = tl.load(
        tmp + offs_m[:, None] * TMP_STRIDE_M + offs_r[None, :] * TMP_STRIDE_R,
        mask=(offs_m[:, None] < M) & (offs_r[None, :] < R),
        other=0.0,
    )
    b = tl.load(up + offs_n[None, :] * R + offs_r[:, None], mask=(offs_n[None, :] < N) & (offs_r[:, None] < R), other=0.0)
    acc = tl.dot(a, b, input_precision="ieee")
    base_vals = tl.load(base + offs_m[:, None] * N + offs_n[None, :], mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    tl.store(out + offs_m[:, None] * N + offs_n[None, :], base_vals + ALPHA * acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def lora_update_from_tmp(tmp, up, base, out, alpha):
    block_r = triton.next_power_of_2(tmp.shape[1])
    grid = (triton.cdiv(tmp.shape[0], 16), triton.cdiv(up.shape[0], 16))
    _lora_update_kernel[grid](
        tmp,
        up,
        base,
        out,
        tmp.shape[0],
        tmp.shape[1],
        up.shape[0],
        tmp.stride(0),
        tmp.stride(1),
        alpha,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_R=block_r,
    )


def side_projection_into(x, side_w, x_scale, side_scale, out, side_n, k):
    return lora_down_projection_into(x, side_w, x_scale, side_scale, out, side_n, k)


def side_projection(x, side_w, x_scale, side_scale, k):
    out = torch.empty((x.shape[0], side_w.shape[0]), device="cuda", dtype=torch.float32)
    side_projection_into(x, side_w, x_scale, side_scale, out, side_w.shape[0], k)
    return out


def side_gated_output(x, side_w, x_scale, side_scale, x_ref, broad_w, k):
    side = side_projection(x, side_w, x_scale, side_scale, k)
    out = torch.empty_like(side)
    side_gate_from_projection(x_ref, broad_w, side, out)
    return out


@triton.jit
def _side_gate_kernel(
    x_ref,
    broad_w,
    side,
    out,
    M: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    SIDE_STRIDE_M: tl.constexpr,
    SIDE_STRIDE_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_S), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kidx = k0 + offs_k
        a = tl.load(x_ref + offs_m[:, None] * K + kidx[None, :], mask=(offs_m[:, None] < M) & (kidx[None, :] < K), other=0.0)
        b = tl.load(broad_w + offs_s[None, :] * K + kidx[:, None], mask=(offs_s[None, :] < S) & (kidx[:, None] < K), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    side_vals = tl.load(
        side + offs_m[:, None] * SIDE_STRIDE_M + offs_s[None, :] * SIDE_STRIDE_S,
        mask=(offs_m[:, None] < M) & (offs_s[None, :] < S),
        other=0.0,
    )
    gate = 1.0 / (1.0 + tl.exp(-side_vals))
    tl.store(out + offs_m[:, None] * S + offs_s[None, :], acc * gate, mask=(offs_m[:, None] < M) & (offs_s[None, :] < S))


def side_gate_from_projection(x_ref, broad_w, side, out):
    grid = (triton.cdiv(side.shape[0], 16), triton.cdiv(side.shape[1], 16))
    _side_gate_kernel[grid](
        x_ref,
        broad_w,
        side,
        out,
        side.shape[0],
        x_ref.shape[1],
        side.shape[1],
        side.stride(0),
        side.stride(1),
        BLOCK_M=16,
        BLOCK_S=16,
        BLOCK_K=32,
    )


def benchmark_lora(total_m=4096, rank=32, k=128, n=128, alpha=0.25):
    x, down, x_scale, down_scale, x_ref, down_ref, up, base = make_lora_inputs(total_m, rank, k, n)
    _, padded_down, _, padded_down_scale, _, padded_down_ref, _, _ = make_lora_inputs(128, 128, k, n)
    padded_down[:rank, :] = down
    padded_down_scale[:rank, :] = down_scale
    padded_down_ref[:rank, :] = down_ref

    compact_tmp = torch.empty((total_m, rank), device="cuda", dtype=torch.float32)
    padded_tmp = torch.empty((total_m, 128), device="cuda", dtype=torch.float32)
    compact_out = torch.empty_like(base)
    padded_out = torch.empty_like(base)

    def compact():
        lora_down_projection_into(x, down, x_scale, down_scale, compact_tmp, rank, k)
        lora_update_from_tmp(compact_tmp, up, base, compact_out, alpha)

    def padded():
        lora_down_projection_into(x, padded_down, x_scale, padded_down_scale, padded_tmp, 128, k)
        lora_update_from_tmp(padded_tmp[:, :rank], up, base, padded_out, alpha)

    compact()
    padded()
    ref_tmp = x_ref @ down_ref.T
    torch.testing.assert_close(compact_tmp, ref_tmp, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(padded_tmp[:, :rank], compact_tmp, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(compact_out, base + alpha * (ref_tmp @ up.T), atol=1e-2, rtol=1e-2)

    compact_ms = triton.testing.do_bench(compact)
    padded_ms = triton.testing.do_bench(padded)
    useful_flops = 2 * total_m * rank * k + 2 * total_m * n * rank
    return {
        "M": total_m,
        "R": rank,
        "K": k,
        "N": n,
        "compact_ms": compact_ms,
        "padded_128_ms": padded_ms,
        "speedup": padded_ms / compact_ms,
        "useful_tflops": useful_flops * 1e-12 / (compact_ms * 1e-3),
    }


def benchmark_side_projection(total_m=4096, side_n=32, k=128, broad_n=128):
    x, side_w, x_scale, side_scale, x_ref, side_ref, broad_w = make_gate_inputs(total_m, side_n, k, broad_n)
    _, padded_side_w, _, padded_side_scale, _, padded_side_ref, _ = make_gate_inputs(128, 128, k, broad_n)
    padded_side_w[:side_n, :] = side_w
    padded_side_scale[:side_n, :] = side_scale
    padded_side_ref[:side_n, :] = side_ref

    compact_side = torch.empty((total_m, side_n), device="cuda", dtype=torch.float32)
    padded_side = torch.empty((total_m, 128), device="cuda", dtype=torch.float32)
    compact_out = torch.empty_like(compact_side)
    padded_out = torch.empty_like(compact_side)

    def compact():
        side_projection_into(x, side_w, x_scale, side_scale, compact_side, side_n, k)
        side_gate_from_projection(x_ref, broad_w, compact_side, compact_out)

    def padded():
        side_projection_into(x, padded_side_w, x_scale, padded_side_scale, padded_side, 128, k)
        side_gate_from_projection(x_ref, broad_w, padded_side[:, :side_n], padded_out)

    compact()
    padded()
    ref_side = x_ref @ side_ref.T
    torch.testing.assert_close(compact_side, ref_side, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(padded_side[:, :side_n], compact_side, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(compact_out, (x_ref @ broad_w[:side_n].T) * torch.sigmoid(ref_side), atol=1e-2, rtol=1e-2)

    compact_ms = triton.testing.do_bench(compact)
    padded_ms = triton.testing.do_bench(padded)
    useful_flops = 2 * total_m * side_n * k + 2 * total_m * broad_n * k
    return {
        "M": total_m,
        "S": side_n,
        "K": k,
        "N": broad_n,
        "compact_ms": compact_ms,
        "padded_128_ms": padded_ms,
        "speedup": padded_ms / compact_ms,
        "useful_tflops": useful_flops * 1e-12 / (compact_ms * 1e-3),
    }


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("rank", [32, 64])
@pytest.mark.parametrize("k", [128, 256])
def test_lora_down_projection_matches_torch(rank, k):
    total_m, n = 512, 128
    x, down, x_scale, down_scale, x_ref, down_ref, up, base = make_lora_inputs(total_m, rank, k, n)
    tmp = torch.empty((total_m, rank), device="cuda", dtype=torch.float32)

    compiled = lora_down_projection_into(x, down, x_scale, down_scale, tmp, rank, k)

    torch.testing.assert_close(tmp, x_ref @ down_ref.T, atol=1e-3, rtol=1e-3)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("rank", [32, 64])
def test_lora_update_matches_torch(rank):
    total_m, k, n, alpha = 512, 128, 128, 0.25
    x, down, x_scale, down_scale, x_ref, down_ref, up, base = make_lora_inputs(total_m, rank, k, n)

    out = lora_update(x, down, x_scale, down_scale, up, base, alpha, k)
    ref = base + alpha * ((x_ref @ down_ref.T) @ up.T)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("rank", [32, 64])
def test_lora_padded_baseline_matches_compact(rank):
    result = benchmark_lora(total_m=512, rank=rank, k=128, n=128)
    assert result["compact_ms"] > 0
    assert result["padded_128_ms"] > 0


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("side_n", [32, 64])
@pytest.mark.parametrize("k", [128, 256])
def test_side_projection_matches_torch(side_n, k):
    total_m, broad_n = 512, 128
    x, side_w, x_scale, side_scale, x_ref, side_ref, broad_w = make_gate_inputs(total_m, side_n, k, broad_n)
    out = torch.empty((total_m, side_n), device="cuda", dtype=torch.float32)

    compiled = side_projection_into(x, side_w, x_scale, side_scale, out, side_n, k)

    torch.testing.assert_close(out, x_ref @ side_ref.T, atol=1e-3, rtol=1e-3)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("side_n", [32, 64])
def test_side_gated_output_matches_torch(side_n):
    total_m, k, broad_n = 512, 128, 128
    x, side_w, x_scale, side_scale, x_ref, side_ref, broad_w = make_gate_inputs(total_m, side_n, k, broad_n)

    out = side_gated_output(x, side_w, x_scale, side_scale, x_ref, broad_w, k)
    ref = (x_ref @ broad_w[:side_n].T) * torch.sigmoid(x_ref @ side_ref.T)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("side_n", [32, 64])
def test_side_projection_padded_baseline_matches_compact(side_n):
    result = benchmark_side_projection(total_m=512, side_n=side_n, k=128, broad_n=128)
    assert result["compact_ms"] > 0
    assert result["padded_128_ms"] > 0


def _print_benchmark(args):
    if args.mode in ("lora", "all"):
        print("TMEM LoRA adapter benchmark")
        print("===========================")
        for rank in args.ranks:
            result = benchmark_lora(total_m=args.m, rank=rank, k=args.k, n=args.n, alpha=args.alpha)
            print(
                f"M={result['M']} R={result['R']} K={result['K']} N={result['N']} | "
                f"compact={result['compact_ms']:.3f} ms | "
                f"padded R=128={result['padded_128_ms']:.3f} ms | "
                f"speedup={result['speedup']:.2f}x | "
                f"useful={result['useful_tflops']:.1f} TFLOP/s"
            )
    if args.mode in ("gate", "all"):
        print("TMEM MLP side-projection benchmark")
        print("==================================")
        for side_n in args.side:
            result = benchmark_side_projection(total_m=args.m, side_n=side_n, k=args.k, broad_n=args.n)
            print(
                f"M={result['M']} S={result['S']} K={result['K']} N={result['N']} | "
                f"compact={result['compact_ms']:.3f} ms | "
                f"padded S=128={result['padded_128_ms']:.3f} ms | "
                f"speedup={result['speedup']:.2f}x | "
                f"useful={result['useful_tflops']:.1f} TFLOP/s"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark compact TMEM skinny side projections.")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--mode", choices=["lora", "gate", "all"], default="all")
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--side", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()
    if not is_blackwell():
        raise RuntimeError("This example requires a Blackwell NVIDIA GPU")
    _print_benchmark(args)


# On one GB200-class run:
#
# ```
# $ CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-side-projections \
#   PYTHONPATH=.:./python python python/examples/gluon/06-tmem-lora-fusion.py \
#   --m 4096 --k 128 --n 128 --mode all --ranks 32 64 --side 32 64
# TMEM LoRA adapter benchmark
# ===========================
# M=4096 R=32 K=128 N=128 | compact=0.024 ms | padded R=128=0.031 ms | speedup=1.27x | useful=2.8 TFLOP/s
# M=4096 R=64 K=128 N=128 | compact=0.033 ms | padded R=128=0.036 ms | speedup=1.08x | useful=4.0 TFLOP/s
# TMEM MLP side-projection benchmark
# ==================================
# M=4096 S=32 K=128 N=128 | compact=0.026 ms | padded S=128=0.032 ms | speedup=1.24x | useful=6.5 TFLOP/s
# M=4096 S=64 K=128 N=128 | compact=0.032 ms | padded S=128=0.034 ms | speedup=1.05x | useful=6.2 TFLOP/s
# ```
#
# Benchmark paths use the same plain Triton post-processing kernels after the
# compact or padded side projection, keeping the comparison off PyTorch while
# isolating the smaller TMEM intermediate.
