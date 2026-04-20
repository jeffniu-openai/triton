"""
TMEM MLP Side Projection
========================

Some MLP blocks carry a small side/control projection next to a broader
projection.  The side projection can be a gate, router hint, mask, or residual
control signal:

    broad[M, N] = x[M, K] @ W_broad[N, K].T
    side[M, S]  = x[M, K] @ W_side[S, K].T
    out[:, :S]  = broad[:, :S] * sigmoid(side)

This example keeps the side projection compact with a narrow TMEM
`TensorMemoryLinearLayout` accumulator for `S=32/64`.  The baseline computes the
same side projection through a padded `S=128` scaled-MMAv5 tile and slices back
to the useful side columns.  A plain Triton gate kernel computes the matching
broad side columns in both paths, so the benchmark isolates the compact
side-projection benefit without timing PyTorch.
"""

import argparse
import importlib.util
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl


_ROUTER_PATH = Path(__file__).with_name("05-tmem-moe-router.py")
_ROUTER_SPEC = importlib.util.spec_from_file_location("_tmem_moe_router_example", _ROUTER_PATH)
_router = importlib.util.module_from_spec(_ROUTER_SPEC)
assert _ROUTER_SPEC.loader is not None
_ROUTER_SPEC.loader.exec_module(_router)


def is_blackwell():
    return _router.is_blackwell()


def make_side_inputs(total_m, side_n, k, broad_n):
    torch.manual_seed(4)
    x, x_scale, x_ref = _router.random_mxfp8_tensor(total_m, k)
    side_w, side_scale, side_ref = _router.random_mxfp8_tensor(side_n, k)
    broad_w = torch.randn((broad_n, k), device="cuda", dtype=torch.float32) / (k**0.5)
    return x, side_w, x_scale, side_scale, x_ref, side_ref, broad_w


def side_projection_into(x, side_w, x_scale, side_scale, out, side_n, k):
    return _router.router_projection_into(x, side_w, x_scale, side_scale, out, side_n, k)


def side_projection(x, side_w, x_scale, side_scale, k):
    return _router.router_projection(x, side_w, x_scale, side_scale, k)


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


def benchmark_side_projection(total_m=4096, side_n=32, k=128, broad_n=128):
    x, side_w, x_scale, side_scale, x_ref, side_ref, broad_w = make_side_inputs(total_m, side_n, k, broad_n)
    _, padded_side_w, _, padded_side_scale, _, padded_side_ref, _ = make_side_inputs(128, 128, k, broad_n)
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
@pytest.mark.parametrize("side_n", [32, 64])
@pytest.mark.parametrize("k", [128, 256])
def test_side_projection_matches_torch(side_n, k):
    total_m, broad_n = 512, 128
    x, side_w, x_scale, side_scale, x_ref, side_ref, broad_w = make_side_inputs(total_m, side_n, k, broad_n)
    out = torch.empty((total_m, side_n), device="cuda", dtype=torch.float32)

    compiled = side_projection_into(x, side_w, x_scale, side_scale, out, side_n, k)

    torch.testing.assert_close(out, x_ref @ side_ref.T, atol=1e-3, rtol=1e-3)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("side_n", [32, 64])
def test_side_gated_output_matches_torch(side_n):
    total_m, k, broad_n = 512, 128, 128
    x, side_w, x_scale, side_scale, x_ref, side_ref, broad_w = make_side_inputs(total_m, side_n, k, broad_n)

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
    parser = argparse.ArgumentParser(description="Benchmark the TMEM MLP side-projection example.")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--side", type=int, nargs="+", default=[32, 64])
    args = parser.parse_args()
    if not is_blackwell():
        raise RuntimeError("This example requires a Blackwell NVIDIA GPU")
    _print_benchmark(args)


# On one GB200-class run:
#
# ```
# $ CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-mlp-side \
#   PYTHONPATH=.:./python python python/examples/gluon/09-tmem-mlp-side-projection.py \
#   --m 4096 --k 128 --n 128 --side 32 64
# TMEM MLP side-projection benchmark
# ==================================
# M=4096 S=32 K=128 N=128 | compact=0.026 ms | padded S=128=0.027 ms | speedup=1.07x | useful=6.6 TFLOP/s
# M=4096 S=64 K=128 N=128 | compact=0.033 ms | padded S=128=0.034 ms | speedup=1.02x | useful=6.0 TFLOP/s
# ```
