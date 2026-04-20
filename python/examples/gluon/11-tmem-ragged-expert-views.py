"""
TMEM Ragged Expert Output Panels
================================

Grouped MoE execution often sees a ragged list of expert token counts.  A
practical grouped kernel wants to compute only the compact output panel for
each nonempty expert instead of padding every expert to a broad 128-column
accumulator tile.

This example models that pattern with a small Python scheduler over expert
panels.  Each nonempty expert uses the narrow TMEM scaled-MMAv5 projection path
from the router example, and writes into its own compact output panel.  The
baseline computes each expert through a padded `N=128` panel and slices the
useful columns back out.

The example intentionally keeps scheduling in Python so the per-expert layout
and baseline are easy to inspect.  It is a user-facing demonstration of compact
ragged expert panels, not a full persistent grouped-MoE scheduler.
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


def make_expert_inputs(counts, output_widths, k):
    torch.manual_seed(6)
    experts = []
    for expert_id, (tokens, width) in enumerate(zip(counts, output_widths)):
        if tokens == 0:
            experts.append(None)
            continue
        hidden, hidden_scale, hidden_ref = _router.random_mxfp8_tensor(tokens, k)
        weight, weight_scale, weight_ref = _router.random_mxfp8_tensor(width, k)
        experts.append((expert_id, hidden, weight, hidden_scale, weight_scale, hidden_ref, weight_ref))
    return experts


def compact_expert_outputs(experts, output_widths, k):
    outputs = []
    for expert, width in zip(experts, output_widths):
        if expert is None:
            outputs.append(None)
            continue
        _, hidden, weight, hidden_scale, weight_scale, _, _ = expert
        outputs.append(_router.router_projection(hidden, weight, hidden_scale, weight_scale, k))
    return outputs


def padded_expert_outputs(experts, output_widths, k):
    outputs = []
    for expert, width in zip(experts, output_widths):
        if expert is None:
            outputs.append(None)
            continue
        _, hidden, weight, hidden_scale, weight_scale, _, weight_ref = expert
        _, padded_weight, _, padded_weight_scale, _, padded_weight_ref = _router.make_router_inputs(128, 128, k)
        padded_weight[:width, :] = weight
        padded_weight_scale[:width, :] = weight_scale
        padded_weight_ref[:width, :] = weight_ref
        padded = _router.router_projection(hidden, padded_weight, hidden_scale, padded_weight_scale, k)
        outputs.append(padded[:, :width])
    return outputs


def make_padded_experts(experts, output_widths, k):
    padded_experts = []
    for expert, width in zip(experts, output_widths):
        if expert is None:
            padded_experts.append(None)
            continue
        expert_id, hidden, weight, hidden_scale, weight_scale, hidden_ref, weight_ref = expert
        _, padded_weight, _, padded_weight_scale, _, padded_weight_ref = _router.make_router_inputs(128, 128, k)
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
        padded = _router.router_projection(hidden, padded_weight, hidden_scale, padded_weight_scale, k)
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
    parser = argparse.ArgumentParser(description="Benchmark compact ragged TMEM expert panels.")
    parser.add_argument("--k", type=int, default=128)
    args = parser.parse_args()
    if not is_blackwell():
        raise RuntimeError("This example requires a Blackwell NVIDIA GPU")
    _print_benchmark(args)


# On one GB200-class run:
#
# ```
# $ CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ragged-experts \
#   PYTHONPATH=.:./python python python/examples/gluon/11-tmem-ragged-expert-views.py --k 128
# TMEM ragged expert panel benchmark
# ==================================
# experts=4 active=3 tokens=512 K=128 | compact=0.089 ms | padded N=128=0.128 ms | speedup=1.43x | useful=0.1 TFLOP/s
# ```
