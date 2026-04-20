# TMEM Example Implementation Plan

Last updated: 2026-04-20 22:45 UTC

This document tracks a follow-on project to turn the completed TMEM
linear-layout backend capabilities into user-facing Gluon examples under
`python/examples/gluon/`. The goal is not to add compiler tests. Each example
must be an executable kernel or small example module with correctness coverage,
shape coverage, benchmark output, and a baseline representing the best code path
available before the TMEM linear-layout generalization work.

## Success Criteria

- Every example lives in `python/examples/gluon/` and can be run as a pytest
  module and as a script or benchmark entry point.
- Each example contains a short algorithm/layout explanation near the top of
  the file, with the real hardware constraint and the new TMEM capability named
  explicitly.
- Each example has focused tests for correctness, shape coverage, edge masks,
  and at least one layout-sensitive assertion against TTGIR/LLIR/PTX when that
  assertion is stable.
- Each benchmark compares against the strongest credible pre-generalization
  baseline for the same workload, not against an intentionally weak reference.
- Baselines must be documented per example. Acceptable baseline categories are:
  padded broad-TMEM tile, shared/register fallback, two-kernel global-memory
  materialization, or existing `TensorMemoryLayout`-only spelling when that was
  the best old path.
- Benchmark output must be recorded with exact shapes, dtypes, device class,
  command, and pass/fail correctness status. Inline benchmark numbers in source
  comments should only be refreshed from real local runs.
- Work must preserve the current TMEM backend invariants: no new ad hoc layout
  rescue stacks, no fake support for true ISA/API boundaries, and no reliance
  on `TRITON_CACHE_DIR` isolation to hide compiler bugs.

## Shared Validation Recipe

Before running any tests, run:

```bash
CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8
```

For each example slice:

```bash
python -m py_compile python/examples/gluon/<example>.py
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-<example> PYTHONPATH=.:./python pytest -s --tb=short python/examples/gluon/<example>.py
git diff --check
```

For broad examples-lane validation, use four GPU shards with stable caches:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-examples-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 python/examples/gluon/
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-examples-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 python/examples/gluon/
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-examples-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 python/examples/gluon/
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-examples-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 python/examples/gluon/
```

## Example 1: MoE Router Skinny Scaled Projection With Top-K

- Status: projection/top-k wrapper implemented at 2026-04-20 21:34 UTC.
- Proposed file: `python/examples/gluon/05-tmem-moe-router.py`.
- New capability used:
  - narrow scaled-MMAv5 accumulator fragments through
    `TensorMemoryLinearLayout`;
  - B-scale storage rematerialization/padding for N=8/16 fragments;
  - optional TMEM load/reduction for row max or top-k prefiltering when the
    layout is reduction-friendly.
- Workload:
  - `hidden[M, K] @ router_weight[num_experts, K].T -> logits[M, num_experts]`;
  - primary shapes: `M in {512, 1024, 4096}`, `K in {128, 256}`,
    `num_experts in {8, 16, 32, 64}`.
- Algorithm:
  - compute router logits with block-scaled MMA into a narrow tile-permuted
    TMEM accumulator;
  - keep logits in TMEM for fused top-1/top-2 selection when practical;
  - store only selected expert ids and scores, plus optional logits for
    correctness/benchmark modes.
- Pre-generalization baseline:
  - padded `N=128` scaled-MMA accumulator followed by slicing/discarding unused
    logits;
  - for fused top-k, baseline may materialize padded logits then run a PyTorch
    or separate Triton/Gluon selection path if that was the strongest old
    executable route.
- Tests:
  - exact top-1/top-2 agreement with PyTorch for deterministic ties;
  - logits agreement for all tested expert counts;
  - masked tail rows and expert counts that are not a full 128-column tile;
  - TTGIR contains `tensor_memory_linear` and `ttng.tc_gen5_mma_scaled`.
- Benchmarks:
  - report projection-only and projection-plus-top-k timings;
  - record speedup over padded baseline and useful TFLOP/s for logits.
- Implementation notes:
  - implemented the narrow projection kernel, padded `N=128` baseline,
    PyTorch top-k wrapper for router-level comparison, correctness tests,
    K=128/256 shape coverage, TTGIR checks, and inline benchmark transcript;
  - the top-k path is not yet fused inside the Gluon kernel. It uses the same
    `torch.topk` call for both compact and padded logits, so the benchmark
    captures the smaller-logits benefit but not a custom in-kernel selector.

## Example 2: LoRA / Adapter Projection Fusion

- Status: compact down-projection/update wrapper implemented at
  2026-04-20 21:51 UTC.
- Proposed file: `python/examples/gluon/06-tmem-lora-fusion.py`.
- New capability used:
  - compact TMEM accumulator for the low-rank `X @ A.T` intermediate;
  - descriptor views or layout-compatible loads to feed the second projection;
  - optional fused epilogue to add the adapter output to a base output tile.
- Workload:
  - `tmp = X[M, K] @ A[r, K].T`;
  - `Y = base[M, N] + alpha * tmp[M, r] @ B[N, r].T`;
  - primary shapes: `M in {256, 1024, 4096}`, `K in {128, 256}`,
    `r in {8, 16, 32, 64}`, `N in {128, 256}`.
- Algorithm:
  - compute the low-rank projection into a narrow TMEM layout;
  - consume the compact TMEM intermediate in the second stage or materialize a
    minimal global-memory fallback only in the baseline;
  - fuse scaling and addition in the epilogue.
- Pre-generalization baseline:
  - two-kernel implementation that writes `tmp` to global memory;
  - or padded broad-TMEM `tmp` when that is faster for the tested shape.
- Tests:
  - compare against PyTorch `base + alpha * ((X @ A.T) @ B.T)`;
  - cover ranks `8/16/32/64`, nonzero `base`, multiple alpha values, and tail
    `M`;
  - assert the optimized path does not allocate/store a padded `N=128`
    low-rank intermediate for `r < 128`.
- Benchmarks:
  - optimized fused path versus best old two-kernel or padded intermediate;
  - report memory traffic avoided for the intermediate.
- Implementation notes:
  - implemented compact MXFP8 down projection, padded `R=128` baseline,
    full LoRA update wrapper using PyTorch for `tmp @ up.T`, shape coverage,
    TTGIR checks, and inline benchmark transcript;
  - the second projection is not fused into the Gluon kernel yet. The example
    isolates the TMEM layout win for the low-rank intermediate.

## Example 3: Small-Vocabulary / Speculative-Decode Candidate Head

- Status: implemented at 2026-04-20 22:04 UTC.
- Proposed file: `python/examples/gluon/07-tmem-candidate-head.py`.
- New capability used:
  - narrow scaled-MMAv5 accumulator fragments for candidate logits;
  - descriptor/view-compatible output order so selected candidate logits can be
    produced in sampler order without a separate reorder kernel.
- Workload:
  - `hidden[M, K] @ selected_vocab[V_selected, K].T`;
  - primary shapes: `M in {1, 4, 16, 128}`, `K in {128, 256}`,
    `V_selected in {16, 32, 64, 96, 128}`.
- Algorithm:
  - stage or gather candidate weights in selected-token order;
  - compute only selected candidate logits into compact TMEM;
  - optionally store logits directly in the order consumed by verification or
    sampling.
- Pre-generalization baseline:
  - padded candidate tile to `N=128`;
  - or two-step candidate projection plus reorder/materialization.
- Tests:
  - compare logits with PyTorch gather/projection;
  - cover candidate counts below and equal to 128, duplicate candidates if
    supported, and batch sizes from decode-style `M=1` to batched verification;
  - assert stable output ordering.
- Benchmarks:
  - candidate projection latency versus padded/reorder baseline;
  - include decode-like small `M` and batched speculative verification shapes.
- Implementation notes:
  - implemented compact selected-candidate projection for `C=32/64`, padded
    `C=128` baseline, candidate-order tests, TTGIR checks, and inline benchmark
    transcript;
  - the example assumes candidate rows have already been staged in candidate
    order. It does not implement an in-kernel vocabulary gather.

## Example 4: Ragged Grouped/MoE Expert Output Views

- Status: implemented at 2026-04-20 22:18 UTC.
- Proposed file: `python/examples/gluon/08-tmem-ragged-expert-views.py`.
- New capability used:
  - descriptor-view chains over a shared TMEM arena;
  - physical bitcast/view support for per-expert accumulator slices;
  - layout-compatible store/readback for compact expert panels.
- Workload:
  - grouped expert projections with per-expert token counts and compact output
    panels;
  - primary shapes: `num_experts in {2, 4, 8}`, per-expert `M_e` ragged over
    `{0, 16, 32, 64, 128}`, `K in {128, 256}`, output `N in {32, 64, 128}`.
- Algorithm:
  - allocate one TMEM arena for a batch of expert tiles;
  - form descriptor views for each expert output panel;
  - compute/store each expert panel without copying through shared memory solely
    to satisfy legacy layout spelling.
- Pre-generalization baseline:
  - one kernel per expert with independent broad accumulator tiles;
  - or a grouped kernel that materializes/reorders expert outputs through
    shared/global memory.
- Tests:
  - compare each expert panel against PyTorch;
  - cover empty experts, nonuniform token counts, and mixed output widths;
  - assert descriptor-view/physical-bitcast markers where stable.
- Benchmarks:
  - grouped compact TMEM arena versus per-expert padded baseline;
  - report total latency and effective useful TFLOP/s.
- Implementation notes:
  - this is likely the highest-complexity example and should come after the
    skinny projection and LoRA examples establish reusable helper patterns.

## Example 5: Windowed Attention Score Tile With TMEM Reductions

- Status: implemented at 2026-04-20 22:45 UTC.
- Proposed file: `python/examples/gluon/10-tmem-windowed-attention-score.py`.
- New capability used:
  - noncanonical score-tile TMEM layouts;
  - descriptor/subview-compatible `ld.red` for row max where supported;
  - fallback software reduction for layouts that are semantically valid but not
    directly `tcgen05.ld.red`-supported.
- Workload:
  - `scores = Q_window @ K_window.T` followed by masked row max/sum preparation;
  - primary shapes: head dim `D in {64, 128}`, window `W in {32, 64, 128}`,
    batch/head tiles chosen to match existing Gluon attention conventions.
- Algorithm:
  - compute attention scores into a TMEM layout matching the local window;
  - apply causal/window masking;
  - compute row max via TMEM reduction load when the layout is directly
    supported, otherwise use the branch's supported software reduction fallback.
- Pre-generalization baseline:
  - score tile staged through registers/shared memory for reduction;
  - or existing attention example path without the supported descriptor-view
    TMEM reduction.
- Tests:
  - compare row max and optional normalized probabilities against PyTorch;
  - cover causal and noncausal windows, shifted windows, and tail windows;
  - assert expected `tcgen05.ld.red` presence only for directly supported
    layouts.
- Benchmarks:
  - score-plus-row-max latency versus shared/register reduction baseline;
  - separate direct-ld.red and fallback-reduction shapes.
- Implementation notes:
  - implemented standalone score-tile row-max example with noncausal and
    causal masking, PyTorch max baseline, Blackwell Ultra skip guard, tests,
    and inline benchmark transcript;
  - kept separate from the full attention example.

## Example 6: Fused Quantized MLP Side Projection

- Status: planned.
- Proposed file: `python/examples/gluon/10-tmem-mlp-side-projection.py`.
- New capability used:
  - narrow scaled-MMAv5 side projection next to a broad projection;
  - compact TMEM storage for side activations;
  - fused epilogue consuming both broad and skinny accumulator tiles.
- Workload:
  - broad MLP projection plus small side/control projection;
  - primary shapes: `M in {256, 1024, 4096}`, `K in {128, 256}`,
    broad `N in {128, 256}`, side `N_side in {16, 32, 64}`.
- Algorithm:
  - compute broad output normally;
  - compute side output into compact narrow TMEM;
  - fuse side-dependent activation, mask, or gate before global store.
- Pre-generalization baseline:
  - separate side-projection kernel;
  - or broad padded side projection to `N=128`.
- Tests:
  - compare against PyTorch broad projection plus side gate/mask;
  - cover side widths, nonzero masks, and tail rows;
  - assert narrow scaled-MMAv5 path is used for side projection.
- Benchmarks:
  - fused broad+side kernel versus separate side kernel or padded side tile;
  - report latency, useful TFLOP/s for side projection, and avoided memory
    writes.
- Implementation notes:
  - implemented compact side projection plus PyTorch broad projection/gate
    wrapper, padded `S=128` side baseline, tests, TTGIR checks, and inline
    benchmark transcript.

## Example 7: Layout-As-Epilogue Store In Consumer Order

- Status: planned.
- Proposed file: `python/examples/gluon/11-tmem-layout-as-epilogue.py`.
- New capability used:
  - accumulator `TensorMemoryLinearLayout` chosen to match a downstream consumer
    layout;
  - descriptor-view/physical-bitcast store paths that avoid a separate epilogue
    transpose or regroup.
- Workload:
  - matrix multiply whose consumer wants transposed, blocked, grouped, or
    column-permuted output;
  - primary shapes: `M in {128, 256}`, `N in {64, 128, 256}`,
    `K in {128, 256}` with several consumer-order permutations.
- Algorithm:
  - choose the TMEM accumulator layout so the physical layout already matches
    the consumer-order store;
  - store directly from TMEM without an intermediate reorder kernel.
- Pre-generalization baseline:
  - canonical accumulator plus separate reorder/store kernel;
  - or shared-memory epilogue reorder after canonical TMEM load.
- Tests:
  - compare consumer-order output against PyTorch plus explicit permutation;
  - cover pure column permutation, tile permutation, and row/column grouping
    that are known supported;
  - assert unsupported true mask/full-footprint cases remain excluded rather
    than silently miscompiled.
- Benchmarks:
  - direct consumer-order store versus canonical-store-plus-reorder baseline;
  - report end-to-end latency and bytes moved by removed reorder.
- Implementation notes:
  - implemented direct consumer-order store versus canonical-store-plus-PyTorch
    reorder baseline, tests over `N=64/128/256`, and inline benchmark
    transcript.

## Execution Order

1. MoE router skinny projection, projection-only first and top-k second.
2. LoRA adapter fusion, compact intermediate first and full fusion second.
3. Candidate-head projection and output-ordering example.
4. Layout-as-epilogue store-order example.
5. Fused MLP side projection.
6. Windowed attention score/reduction example.
7. Ragged grouped/MoE expert views.

This order starts with the strongest narrow scaled-MMAv5 wins, then expands to
descriptor/view and reduction examples, and leaves the most coordination-heavy
grouped and attention examples until reusable helper patterns exist.

## Progress Log

- 2026-04-20 21:07 UTC: created the implementation plan from the seven
  candidate examples. No example files have been added yet. Next concrete task
  is Example 1 projection-only MoE router under `python/examples/gluon/` with a
  padded `N=128` baseline, correctness tests, shape sweep, and benchmark
  transcript.
- 2026-04-20 21:34 UTC: implemented Example 1 as
  `python/examples/gluon/05-tmem-moe-router.py`. Validation:
  `make -j8` had no work after the include-path workaround,
  `python -m py_compile` passed, focused pytest passed `8 passed in 12.21s`,
  projection benchmark printed `E=32` `1.21x` and `E=64` `1.02x`, and
  projection plus top-k wrapper benchmark printed `E=32` `2.11x` and
  `E=64` `2.03x` over the padded `E=128` baseline.
- 2026-04-20 21:51 UTC: implemented Example 2 as
  `python/examples/gluon/06-tmem-lora-fusion.py`. Validation:
  `python -m py_compile` passed, focused pytest passed `8 passed in 11.53s`,
  and the benchmark printed `R=32` `1.09x` and `R=64` `1.11x` over the padded
  `R=128` baseline.
- 2026-04-20 22:04 UTC: implemented Example 3 as
  `python/examples/gluon/07-tmem-candidate-head.py`. Validation:
  `python -m py_compile` passed, focused pytest passed `6 passed in 9.71s`,
  and the benchmark printed `C=32` `1.17x` and `C=64` `1.03x` over the padded
  `C=128` baseline.
- 2026-04-20 22:18 UTC: implemented Example 7, ahead of examples 4-6, as
  `python/examples/gluon/08-tmem-layout-as-epilogue.py`. Validation:
  `python -m py_compile` passed, focused pytest passed `5 passed in 7.07s`,
  and the benchmark printed `8.56x`, `5.61x`, and `3.53x` over the
  canonical-output plus reorder baseline for `N=64/128/256`.
- 2026-04-20 22:31 UTC: implemented Example 6 as
  `python/examples/gluon/09-tmem-mlp-side-projection.py`. Validation:
  `python -m py_compile` passed, focused pytest passed `8 passed in 10.41s`,
  and the benchmark printed `S=32` `1.04x` and `S=64` `1.03x` over the padded
  `S=128` baseline.
- 2026-04-20 22:45 UTC: implemented Example 5 as
  `python/examples/gluon/10-tmem-windowed-attention-score.py`. Validation:
  `python -m py_compile` passed, focused pytest passed `5 passed in 8.71s`,
  and the benchmark printed noncausal rows near parity/slower than `torch.max`
  and causal rows `8.68x`/`7.56x` faster than the executable PyTorch
  mask-plus-max baseline.
