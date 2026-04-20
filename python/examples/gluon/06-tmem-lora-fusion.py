"""
TMEM LoRA Adapter Projection
============================

LoRA-style adapters compute a low-rank side path:

    tmp[M, R] = x[M, K] @ down[R, K].T
    y[M, N] = base[M, N] + alpha * tmp[M, R] @ up[N, R].T

The down projection is skinny.  Before the TMEM linear-layout work, a
block-scaled TMEM implementation commonly had to form a broad rank-128
intermediate and slice the useful rank columns back out.  This example keeps
the adapter intermediate compact for `R=32` and `R=64` using a tile-permuted
`TensorMemoryLinearLayout` accumulator.  The second projection is performed
with PyTorch so the example isolates the layout-sensitive TMEM win in the
adapter down projection while still checking the full LoRA update.
"""

import argparse
import math

import pytest
import torch
import triton
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
    lora_down_projection_into(x, down, x_scale, down_scale, tmp, rank, k)
    return base + alpha * (tmp @ up.T)


def benchmark_lora(total_m=4096, rank=32, k=128, n=128, alpha=0.25):
    x, down, x_scale, down_scale, x_ref, down_ref, up, base = make_lora_inputs(total_m, rank, k, n)
    _, padded_down, _, padded_down_scale, _, padded_down_ref, _, _ = make_lora_inputs(128, 128, k, n)
    padded_down[:rank, :] = down
    padded_down_scale[:rank, :] = down_scale
    padded_down_ref[:rank, :] = down_ref

    compact_tmp = torch.empty((total_m, rank), device="cuda", dtype=torch.float32)
    padded_tmp = torch.empty((total_m, 128), device="cuda", dtype=torch.float32)

    def compact():
        lora_down_projection_into(x, down, x_scale, down_scale, compact_tmp, rank, k)
        base + alpha * (compact_tmp @ up.T)

    def padded():
        lora_down_projection_into(x, padded_down, x_scale, padded_down_scale, padded_tmp, 128, k)
        base + alpha * (padded_tmp[:, :rank] @ up.T)

    compact()
    padded()
    ref_tmp = x_ref @ down_ref.T
    torch.testing.assert_close(compact_tmp, ref_tmp, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(padded_tmp[:, :rank], compact_tmp, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(base + alpha * (compact_tmp @ up.T), base + alpha * (ref_tmp @ up.T), atol=1e-2, rtol=1e-2)

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


def _print_benchmark(args):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the TMEM LoRA adapter example.")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()
    if not is_blackwell():
        raise RuntimeError("This example requires a Blackwell NVIDIA GPU")
    _print_benchmark(args)


# On one GB200-class run:
#
# ```
# $ CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-lora-fusion \
#   PYTHONPATH=.:./python python python/examples/gluon/06-tmem-lora-fusion.py \
#   --m 4096 --k 128 --n 128 --ranks 32 64
# TMEM LoRA adapter benchmark
# ===========================
# M=4096 R=32 K=128 N=128 | compact=0.125 ms | padded R=128=0.136 ms | speedup=1.09x | useful=0.5 TFLOP/s
# M=4096 R=64 K=128 N=128 | compact=0.125 ms | padded R=128=0.138 ms | speedup=1.11x | useful=1.1 TFLOP/s
# ```
#
# Both benchmark paths use PyTorch for the second `tmp @ up.T` projection.  The
# comparison isolates the compact TMEM down-projection and the smaller
# intermediate handed to the adapter update.
