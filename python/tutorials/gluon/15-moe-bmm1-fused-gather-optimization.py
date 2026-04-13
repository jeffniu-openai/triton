"""
Optimizing a Fused-Gather MoE BMM1 Kernel
=========================================

This tutorial distills the practical workflow used to optimize the standalone
Gluon MoE BMM1 fused-gather kernel in:

- ``python/examples/gluon/05-moe-bmm1-fused-gather.py``

Unlike the earlier Gluon tutorials, this file is not about introducing one new
hardware primitive. It is about how to drive a real kernel-optimization effort
without getting lost in noise, folklore, or one-off benchmark artifacts.

The most important lessons from the project were:

- Measure on the same prepared inputs before trusting a change.
- Use correctness gates early and often.
- Treat Nsight Compute, PTX, and SASS as debugging tools for hypotheses, not as
  a substitute for benchmark evidence.
- Prefer one small, interpretable change over broad rewrites.
- Keep durable notes for both wins and dead ends.

The current kernel structure is:

.. code-block:: text

   activation loader ----.
                         |
   weight+scale loader --+--> MMA --> accumulator --> exact SwiGLU epilogue
                                                       |
                                                       v
                                              helper-store ring in SMEM
                                                       |
                                                       v
                                               store-helper partition
                                                       |
                                                       v
                                                  FP8 global output

What actually mattered in this project:

- packed ``f32x2`` epilogue math
- packed FP8 output conversion / store
- helper-store epilogue ownership
- low-batch ``BLOCK_M`` policy
- measurement discipline

What did not produce durable wins:

- broad schedule sweeps beyond the small promoted banded schedule
- deleting buffers to chase occupancy
- assuming split-K was the low-batch fix
- async TMA store as a drop-in replacement for the helper-store path
- source-level scalar cleanups without supporting PTX/SASS evidence

This tutorial shows how to:

1. load the live example safely
2. validate it against the reference implementation
3. benchmark the example and reference on the same prepared inputs
4. inspect the data that should drive your next hypothesis
5. continue the work without repeating already-exhausted directions

For the full historical record and deeper profiling notes, see:

- ``.codex/initiatives/artifacts/ws-matmul-performance-report.md``
- ``.codex/initiatives/artifacts/fp8-mxfp4-fused-gather-matmul.md``
"""

# %%
# Step 1: load the live example from the examples directory.
#
# The tutorial keeps the optimization story attached to the current executable
# kernel rather than copying logic into a second tutorial-only implementation.

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import pytest
import torch
import triton
from triton.testing import do_bench_cudagraph
from triton_kernels.testing import assert_close


EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "gluon" / "05-moe-bmm1-fused-gather.py"
SELECTED_BATCHES = [128, 512, 2048, 16384]


@lru_cache(maxsize=1)
def load_live_example():
    spec = importlib.util.spec_from_file_location("gluon_moe_bmm1_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    try:
        target = triton.runtime.driver.active.get_current_target()
    except RuntimeError:
        return False
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


# %%
# Step 2: keep one baseline preparation path and reuse it for both providers.
#
# This was one of the most important practical rules in the project: if the
# example and reference do not consume the exact same prepared inputs, the
# timing comparison is much harder to trust.


def prepare_pair(batch_size: int, device: str | None = None, seed: int = 0):
    example = load_live_example()
    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"
    prepared = example.prepare_case(batch_size=batch_size, device=device, seed=seed)
    return prepared


def run_provider(prepared, precision_config, provider: str):
    example = load_live_example()
    kernel = example.matmul if provider == "example" else example.reference_matmul
    out = example.make_output_buffer(prepared)
    return example.run_kernel(prepared, kernel, precision_config, out)


def benchmark_pair(batch_size: int, rep: int = 200, seed: int = 0):
    example = load_live_example()
    prepared = prepare_pair(batch_size, seed=seed)
    example_precision = example.make_precision_config(prepared)
    reference_precision = example.make_precision_config(prepared)

    example_out = example.make_output_buffer(prepared)
    reference_out = example.make_output_buffer(prepared)

    def run_example():
        return example.run_kernel(prepared, example.matmul, example_precision, example_out)

    def run_reference():
        return example.run_kernel(prepared, example.reference_matmul, reference_precision, reference_out)

    example_ms = do_bench_cudagraph(run_example, rep=rep)
    reference_ms = do_bench_cudagraph(run_reference, rep=rep)

    n_tokens = int(prepared.ragged_metadata.slice_sizes.sum().item())
    k, n = example.GPT_OSS_120B_MM1_SHAPE
    flops = 2 * n_tokens * k * n
    example_tflops = flops * 1e-12 / (example_ms * 1e-3)
    reference_tflops = flops * 1e-12 / (reference_ms * 1e-3)

    return {
        "batch_size": batch_size,
        "example_ms": example_ms,
        "reference_ms": reference_ms,
        "example_tflops": example_tflops,
        "reference_tflops": reference_tflops,
        "speedup_pct": 100.0 * (reference_ms - example_ms) / reference_ms,
    }


# %%
# Step 3: correctness comes before profiling.
#
# The project converged on comparing the example directly against the reference
# implementation and using the published tolerances from the example itself.


def validate_pair(batch_size: int, seed: int = 0):
    example = load_live_example()
    prepared = prepare_pair(batch_size, seed=seed)
    ref_precision = example.make_precision_config(prepared)
    cand_precision = example.make_precision_config(prepared)
    ref_y = run_provider(prepared, ref_precision, "reference")
    cand_y = run_provider(prepared, cand_precision, "example")

    assert_close(
        ref_y.to(torch.float32),
        cand_y.to(torch.float32),
        maxtol=example.OUTPUT_MAXTOL,
        rmstol=example.OUTPUT_RMSTOL,
        description=f"tutorial-bs{batch_size}:out",
        verbose=False,
    )

    ref_scale = ref_precision.flex_ctx.out_data.actual_scale
    cand_scale = cand_precision.flex_ctx.out_data.actual_scale
    if ref_scale is not None or cand_scale is not None:
        assert ref_scale is not None and cand_scale is not None
        assert_close(
            ref_scale.to(torch.float32),
            cand_scale.to(torch.float32),
            maxtol=example.SCALE_TOL,
            rmstol=example.SCALE_TOL,
            description=f"tutorial-bs{batch_size}:scale",
            verbose=False,
        )


# %%
# Step 4: benchmark a small representative sweep before choosing a hypothesis.
#
# In the actual project, most bad ideas died here. If a change did not survive a
# small, same-GPU, same-input sweep, it was not worth deeper profiling.


def print_selected_sweep(rep: int = 200):
    print("Gluon MoE BMM1 Selected Sweep")
    print("=============================")
    print(" batch | example (ms) | reference (ms) | example TFLOP/s | ref TFLOP/s | speedup")
    for batch_size in SELECTED_BATCHES:
        row = benchmark_pair(batch_size, rep=rep)
        print(
            f"{row['batch_size']:>6} | "
            f"{row['example_ms']:>12.5f} | "
            f"{row['reference_ms']:>14.5f} | "
            f"{row['example_tflops']:>15.2f} | "
            f"{row['reference_tflops']:>12.2f} | "
            f"{row['speedup_pct']:>7.2f}%"
        )


# %%
# Step 5: use profiling only to answer a concrete question.
#
# Good profiling questions from this project looked like:
#
# - Is the store helper winning because it frees the ring slot earlier?
# - Is async TMA store slower because the producer waits on global completion?
# - Is a low-batch loss really a split-K problem, or just a BLOCK_M problem?
#
# Useful command patterns:
#
# .. code-block:: bash
#
#    # Benchmark the live example
#    PYTHONPATH=python/triton_kernels:python \
#      python python/examples/gluon/05-moe-bmm1-fused-gather.py
#
#    # Validate the live example
#    PYTHONPATH=python/triton_kernels:python \
#      pytest -s --tb=short python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op
#
#    # Run a focused NCU profile on the example
#    USE_IR_LOC=ttgir ncu -o profile -f --import-source on --set full \
#      -k '::regex:.*matmul.*' \
#      python python/examples/gluon/05-moe-bmm1-fused-gather.py
#
# When reading SASS for this family, the most useful questions were usually:
#
# - where do `STS` instructions land relative to epilogue arithmetic?
# - where do `SYNCS.ARRIVE` or `mbarrier` operations actually issue?
# - are `STG.E` clusters or `UTMASTG` issuance on the same critical path as
#   the epilogue math?


@pytest.mark.skipif(not is_blackwell(), reason="This tutorial requires a Blackwell GPU")
def test_live_example_matches_reference():
    validate_pair(batch_size=128)


if __name__ == "__main__":
    if not is_blackwell():
        raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")
    print_selected_sweep(rep=100)
