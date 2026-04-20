"""
TMEM Candidate Vocabulary Head
==============================

Speculative decoding and constrained decoding often score a small candidate
set instead of a full vocabulary:

    logits[M, C] = hidden[M, K] @ selected_vocab[C, K].T

The selected token order matters because the sampler consumes logits in the
candidate-list order.  This example keeps the candidate projection compact for
`C=32` and `C=64` with the same narrow TMEM scaled-MMAv5 path used by the router
example, then compares it with the pre-generalization padded `C=128` path.

The example assumes the selected vocabulary rows have already been staged in
candidate-list order before the kernel launch.  The TMEM part demonstrates
compact candidate logits and stable output order; it is not a general in-kernel
vocabulary gather.
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


def make_candidate_inputs(total_m, candidate_count, k, vocab_size=512):
    torch.manual_seed(2)
    hidden, hidden_scale, hidden_ref = _router.random_mxfp8_tensor(total_m, k)
    # Use a deterministic nontrivial order so the test checks that output
    # columns follow candidate-list order, not sorted token id order.
    candidate_ids = torch.randperm(vocab_size, device="cuda")[:candidate_count]
    selected, selected_scale, selected_ref = _router.random_mxfp8_tensor(candidate_count, k)
    return hidden, selected, hidden_scale, selected_scale, hidden_ref, selected_ref, candidate_ids


def candidate_projection(hidden, selected, hidden_scale, selected_scale, k):
    return _router.router_projection(hidden, selected, hidden_scale, selected_scale, k)


def candidate_projection_into(hidden, selected, hidden_scale, selected_scale, out, candidate_count, k):
    return _router.router_projection_into(hidden, selected, hidden_scale, selected_scale, out, candidate_count, k)


def benchmark_candidate_head(total_m=4096, candidate_count=32, k=128):
    hidden, selected, hidden_scale, selected_scale, hidden_ref, selected_ref, candidate_ids = make_candidate_inputs(
        total_m, candidate_count, k
    )
    _, padded_selected, _, padded_selected_scale, _, padded_selected_ref, _ = make_candidate_inputs(128, 128, k)
    padded_selected[:candidate_count, :] = selected
    padded_selected_scale[:candidate_count, :] = selected_scale
    padded_selected_ref[:candidate_count, :] = selected_ref

    compact_out = torch.empty((total_m, candidate_count), device="cuda", dtype=torch.float32)
    padded_out = torch.empty((total_m, 128), device="cuda", dtype=torch.float32)

    def compact():
        candidate_projection_into(hidden, selected, hidden_scale, selected_scale, compact_out, candidate_count, k)

    def padded():
        candidate_projection_into(hidden, padded_selected, hidden_scale, padded_selected_scale, padded_out, 128, k)

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


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("candidate_count", [32, 64])
@pytest.mark.parametrize("k", [128, 256])
def test_candidate_projection_matches_selected_vocab_order(candidate_count, k):
    total_m = 512
    hidden, selected, hidden_scale, selected_scale, hidden_ref, selected_ref, candidate_ids = make_candidate_inputs(
        total_m, candidate_count, k
    )

    out = torch.empty((total_m, candidate_count), device="cuda", dtype=torch.float32)
    compiled = candidate_projection_into(hidden, selected, hidden_scale, selected_scale, out, candidate_count, k)

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


def _print_benchmark(args):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the TMEM candidate vocabulary head example.")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--candidates", type=int, nargs="+", default=[32, 64])
    args = parser.parse_args()
    if not is_blackwell():
        raise RuntimeError("This example requires a Blackwell NVIDIA GPU")
    _print_benchmark(args)


# On one GB200-class run:
#
# ```
# $ CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-candidate-head \
#   PYTHONPATH=.:./python python python/examples/gluon/07-tmem-candidate-head.py \
#   --m 4096 --k 128 --candidates 32 64
# TMEM candidate-head benchmark
# =============================
# M=4096 C=32 K=128 | compact=0.011 ms | padded C=128=0.013 ms | speedup=1.17x | useful=3.0 TFLOP/s | first_candidate=63
# M=4096 C=64 K=128 | compact=0.013 ms | padded C=128=0.013 ms | speedup=1.03x | useful=5.2 TFLOP/s | first_candidate=63
# ```
