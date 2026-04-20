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
to the useful side columns.  The broad projection is computed with PyTorch in
both paths so the benchmark isolates the compact side-projection benefit.
"""

import argparse
import importlib.util
from pathlib import Path

import pytest
import torch
import triton


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
    broad = x_ref @ broad_w.T
    out = broad.clone()
    out[:, :side.shape[1]] = out[:, :side.shape[1]] * torch.sigmoid(side)
    return out


def benchmark_side_projection(total_m=4096, side_n=32, k=128, broad_n=128):
    x, side_w, x_scale, side_scale, x_ref, side_ref, broad_w = make_side_inputs(total_m, side_n, k, broad_n)
    _, padded_side_w, _, padded_side_scale, _, padded_side_ref, _ = make_side_inputs(128, 128, k, broad_n)
    padded_side_w[:side_n, :] = side_w
    padded_side_scale[:side_n, :] = side_scale
    padded_side_ref[:side_n, :] = side_ref

    compact_side = torch.empty((total_m, side_n), device="cuda", dtype=torch.float32)
    padded_side = torch.empty((total_m, 128), device="cuda", dtype=torch.float32)

    def compact():
        side_projection_into(x, side_w, x_scale, side_scale, compact_side, side_n, k)
        broad = x_ref @ broad_w.T
        broad[:, :side_n] * torch.sigmoid(compact_side)

    def padded():
        side_projection_into(x, padded_side_w, x_scale, padded_side_scale, padded_side, 128, k)
        broad = x_ref @ broad_w.T
        broad[:, :side_n] * torch.sigmoid(padded_side[:, :side_n])

    compact()
    padded()
    ref_side = x_ref @ side_ref.T
    torch.testing.assert_close(compact_side, ref_side, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(padded_side[:, :side_n], compact_side, atol=1e-3, rtol=1e-3)

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
    broad = x_ref @ broad_w.T
    ref = broad.clone()
    ref[:, :side_n] = ref[:, :side_n] * torch.sigmoid(x_ref @ side_ref.T)

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
# M=4096 S=32 K=128 N=128 | compact=0.140 ms | padded S=128=0.146 ms | speedup=1.04x | useful=1.2 TFLOP/s
# M=4096 S=64 K=128 N=128 | compact=0.141 ms | padded S=128=0.145 ms | speedup=1.03x | useful=1.4 TFLOP/s
# ```
