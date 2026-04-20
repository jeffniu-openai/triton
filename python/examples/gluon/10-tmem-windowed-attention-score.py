"""
TMEM Windowed Attention Score Reduction
=======================================

Attention kernels repeatedly reduce score tiles:

    row_max = max(scores + mask, axis=-1)

This example isolates that score-tile stage.  It stores a windowed score tile
in a tile-permuted `TensorMemoryLinearLayout` and uses `tmem.load_max` to
compute the row maximum.  The baseline uses a normal PyTorch reduction on the
same masked score tile, representing a path that materializes scores outside
the TMEM reduction flow.

The example is intentionally separate from the full attention example so the
TMEM reduction contract and benchmark are small and easy to inspect.
"""

import argparse
import math

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLinearLayout, allocate_tensor_memory


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


def is_blackwell_ultra():
    return is_blackwell() and torch.cuda.get_device_capability()[1] >= 3


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


@gluon.jit
def _score_row_max_kernel(
    scores,
    out_scores,
    row_max,
    layout: gl.constexpr,
    M: gl.constexpr,
    N: gl.constexpr,
    CAUSAL: gl.constexpr,
):
    reg_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    reg_layout_1d: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    m = gl.arange(0, M, gl.SliceLayout(1, reg_layout))[:, None]
    n = gl.arange(0, N, gl.SliceLayout(0, reg_layout))[None, :]
    value = gl.load(scores + m * N + n)
    if CAUSAL:
        value = gl.where(n <= (m % N), value, -float("inf"))

    tmem = allocate_tensor_memory(scores.dtype.element_ty, [M, N], layout)
    tmem_layout: gl.constexpr = tmem.get_reg_layout()
    tmem.store(gl.convert_layout(value, tmem_layout))

    restored, reduced = tmem.load_max(abs=False, propagate_nan=tl.PropagateNan.NONE)
    restored = gl.convert_layout(restored, reg_layout)
    gl.store(out_scores + m * N + n, restored)

    row = gl.arange(0, M, reg_layout_1d)
    reduced = gl.convert_layout(reduced, reg_layout_1d)
    gl.store(row_max + row, reduced)


def score_row_max(scores, causal=False, tile_n=32):
    m, n = scores.shape
    out_scores = torch.empty_like(scores)
    row_max = torch.empty((m,), device="cuda", dtype=scores.dtype)
    layout = make_tmem_tile_permuted_layout(m, n, tile_n)
    _score_row_max_kernel[(1, )](scores, out_scores, row_max, layout, m, n, causal, num_warps=4)
    return out_scores, row_max


def masked_scores(scores, causal=False):
    if not causal:
        return scores
    m, n = scores.shape
    rows = torch.arange(m, device=scores.device)[:, None]
    cols = torch.arange(n, device=scores.device)[None, :]
    return torch.where(cols <= (rows % n), scores, torch.full_like(scores, -float("inf")))


def benchmark_score_row_max(m=128, n=128, causal=False):
    torch.manual_seed(5)
    scores = torch.randn((m, n), device="cuda", dtype=torch.float32)
    tile_n = 16 if n == 64 else 32

    def direct():
        score_row_max(scores, causal=causal, tile_n=tile_n)

    def baseline():
        torch.max(masked_scores(scores, causal=causal), dim=1).values

    out_scores, direct_max = score_row_max(scores, causal=causal, tile_n=tile_n)
    ref_scores = masked_scores(scores, causal=causal)
    torch.testing.assert_close(out_scores, ref_scores, atol=0, rtol=0)
    torch.testing.assert_close(direct_max, torch.max(ref_scores, dim=1).values, atol=0, rtol=0)

    direct_ms = triton.testing.do_bench(direct)
    baseline_ms = triton.testing.do_bench(baseline)
    return {
        "M": m,
        "N": n,
        "causal": causal,
        "direct_ms": direct_ms,
        "baseline_ms": baseline_ms,
        "speedup": baseline_ms / direct_ms,
    }


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_score_row_max_matches_torch(n, causal):
    m = 128
    torch.manual_seed(5)
    scores = torch.randn((m, n), device="cuda", dtype=torch.float32)
    tile_n = 16 if n == 64 else 32

    out_scores, row_max = score_row_max(scores, causal=causal, tile_n=tile_n)
    ref_scores = masked_scores(scores, causal=causal)

    torch.testing.assert_close(out_scores, ref_scores, atol=0, rtol=0)
    torch.testing.assert_close(row_max, torch.max(ref_scores, dim=1).values, atol=0, rtol=0)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
def test_score_row_max_benchmark_smoke():
    result = benchmark_score_row_max(m=128, n=128, causal=True)
    assert result["direct_ms"] > 0
    assert result["baseline_ms"] > 0


def _print_benchmark(args):
    print("TMEM windowed attention score benchmark")
    print("=======================================")
    for n in args.n:
        for causal in args.causal:
            result = benchmark_score_row_max(m=args.m, n=n, causal=causal)
            suffix = "causal" if causal else "noncausal"
            print(
                f"M={result['M']} N={result['N']} {suffix} | "
                f"tmem_ldred={result['direct_ms']:.3f} ms | "
                f"torch_max={result['baseline_ms']:.3f} ms | "
                f"speedup={result['speedup']:.2f}x"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TMEM row-max reductions for score tiles.")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--causal", action="append", type=lambda x: x.lower() == "true", default=None)
    args = parser.parse_args()
    if args.causal is None:
        args.causal = [False, True]
    if not is_blackwell_ultra():
        raise RuntimeError("This example requires a Blackwell Ultra NVIDIA GPU")
    _print_benchmark(args)


# On one GB200-class run:
#
# ```
# $ CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-attn-score \
#   PYTHONPATH=.:./python python python/examples/gluon/10-tmem-windowed-attention-score.py \
#   --m 128 --n 64 128
# TMEM windowed attention score benchmark
# =======================================
# M=128 N=64 noncausal | tmem_ldred=0.009 ms | torch_max=0.008 ms | speedup=0.88x
# M=128 N=64 causal | tmem_ldred=0.008 ms | torch_max=0.069 ms | speedup=8.68x
# M=128 N=128 noncausal | tmem_ldred=0.010 ms | torch_max=0.008 ms | speedup=0.76x
# M=128 N=128 causal | tmem_ldred=0.009 ms | torch_max=0.071 ms | speedup=7.56x
# ```
#
# The causal baseline materializes the causal mask before `torch.max`.  In a
# full fused attention kernel the baseline would be a custom in-kernel
# reduction, so these numbers are an executable layout/reduction demonstration
# rather than a claim about all attention implementations.
