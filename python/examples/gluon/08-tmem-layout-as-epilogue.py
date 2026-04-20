"""
TMEM Layout-As-Epilogue Store Ordering
=====================================

Some consumers want a producer's output in a blocked or grouped column order.
The old fallback is to store a canonical output and run a second reorder step.
With explicit TMEM linear layouts, the producer can keep an accumulator or
scratch tile in the same tile-permuted shape that the consumer wants and store
directly in consumer order.

This example uses a compact copy-style kernel rather than a full GEMM so the
layout effect is isolated.  The optimized path writes the consumer-ordered
output directly.  The baseline stores canonical order into an intermediate and
uses the same PyTorch reorder the consumer would otherwise need.
"""

import argparse
import math

import pytest
import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLinearLayout, allocate_tensor_memory


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


def consumer_permutation(n, tile_n, device="cuda"):
    cols = torch.arange(n, device=device)
    return cols ^ tile_n


@gluon.jit
def _direct_consumer_order_kernel(inp, out, layout: gl.constexpr, M: gl.constexpr, N: gl.constexpr, TILE_N: gl.constexpr):
    reg_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
    m = gl.arange(0, M, gl.SliceLayout(1, reg_layout))[:, None]
    n = gl.arange(0, N, gl.SliceLayout(0, reg_layout))[None, :]
    value = gl.load(inp + m * N + n)

    tmem = allocate_tensor_memory(inp.dtype.element_ty, [M, N], layout)
    tmem_layout: gl.constexpr = tmem.get_reg_layout()
    tmem.store(gl.convert_layout(value, tmem_layout))
    staged = gl.convert_layout(tmem.load(), reg_layout)

    consumer_n = n ^ TILE_N
    gl.store(out + m * N + consumer_n, staged)


@gluon.jit
def _canonical_store_kernel(inp, out, layout: gl.constexpr, M: gl.constexpr, N: gl.constexpr):
    reg_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
    m = gl.arange(0, M, gl.SliceLayout(1, reg_layout))[:, None]
    n = gl.arange(0, N, gl.SliceLayout(0, reg_layout))[None, :]
    value = gl.load(inp + m * N + n)

    tmem = allocate_tensor_memory(inp.dtype.element_ty, [M, N], layout)
    tmem_layout: gl.constexpr = tmem.get_reg_layout()
    tmem.store(gl.convert_layout(value, tmem_layout))
    staged = gl.convert_layout(tmem.load(), reg_layout)
    gl.store(out + m * N + n, staged)


def direct_consumer_order(inp, tile_n):
    m, n = inp.shape
    out = torch.empty_like(inp)
    layout = make_tmem_tile_permuted_layout(m, n, tile_n)
    _direct_consumer_order_kernel[(1, )](inp, out, layout, m, n, tile_n, num_warps=4)
    return out


def canonical_then_reorder(inp, tile_n):
    m, n = inp.shape
    tmp = torch.empty_like(inp)
    layout = make_tmem_tile_permuted_layout(m, n, tile_n)
    _canonical_store_kernel[(1, )](inp, tmp, layout, m, n, num_warps=4)
    return tmp[:, consumer_permutation(n, tile_n, inp.device)]


def benchmark_layout_epilogue(m=128, n=128, tile_n=32):
    torch.manual_seed(3)
    inp = torch.randn((m, n), device="cuda", dtype=torch.float32)
    perm = consumer_permutation(n, tile_n, inp.device)
    ref = inp[:, perm]

    direct_out = direct_consumer_order(inp, tile_n)
    baseline_out = canonical_then_reorder(inp, tile_n)
    torch.testing.assert_close(direct_out, ref, atol=0, rtol=0)
    torch.testing.assert_close(baseline_out, ref, atol=0, rtol=0)

    direct_ms = triton.testing.do_bench(lambda: direct_consumer_order(inp, tile_n))
    baseline_ms = triton.testing.do_bench(lambda: canonical_then_reorder(inp, tile_n))
    bytes_moved = inp.numel() * inp.element_size()
    return {
        "M": m,
        "N": n,
        "tile_n": tile_n,
        "direct_ms": direct_ms,
        "baseline_ms": baseline_ms,
        "speedup": baseline_ms / direct_ms,
        "bytes": bytes_moved,
    }


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n,tile_n", [(64, 16), (128, 32), (256, 64)])
def test_direct_consumer_order_matches_pytorch(n, tile_n):
    m = 128
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = direct_consumer_order(inp, tile_n)
    ref = inp[:, consumer_permutation(n, tile_n, inp.device)]
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n,tile_n", [(64, 16), (128, 32)])
def test_layout_epilogue_baseline_matches_direct(n, tile_n):
    result = benchmark_layout_epilogue(m=128, n=n, tile_n=tile_n)
    assert result["direct_ms"] > 0
    assert result["baseline_ms"] > 0


def _print_benchmark(args):
    print("TMEM layout-as-epilogue benchmark")
    print("=================================")
    for n, tile_n in [(64, 16), (128, 32), (256, 64)]:
        if n > args.n_max:
            continue
        result = benchmark_layout_epilogue(m=args.m, n=n, tile_n=tile_n)
        print(
            f"M={result['M']} N={result['N']} tile_n={result['tile_n']} | "
            f"direct={result['direct_ms']:.3f} ms | "
            f"canonical+reorder={result['baseline_ms']:.3f} ms | "
            f"speedup={result['speedup']:.2f}x | bytes={result['bytes']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark direct TMEM consumer-order stores.")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n-max", type=int, default=256)
    args = parser.parse_args()
    if not is_blackwell():
        raise RuntimeError("This example requires a Blackwell NVIDIA GPU")
    _print_benchmark(args)


# On one GB200-class run:
#
# ```
# $ CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-layout-epilogue \
#   PYTHONPATH=.:./python python python/examples/gluon/08-tmem-layout-as-epilogue.py \
#   --m 128 --n-max 256
# TMEM layout-as-epilogue benchmark
# =================================
# M=128 N=64 tile_n=16 | direct=0.011 ms | canonical+reorder=0.095 ms | speedup=8.56x | bytes=32768
# M=128 N=128 tile_n=32 | direct=0.017 ms | canonical+reorder=0.095 ms | speedup=5.61x | bytes=65536
# M=128 N=256 tile_n=64 | direct=0.027 ms | canonical+reorder=0.095 ms | speedup=3.53x | bytes=131072
# ```
