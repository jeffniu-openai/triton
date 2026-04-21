# Round 10 Lane M: copy / scales / multicast / CGA adversarial fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only. No backend/compiler repairs were attempted.
- Temporary probe: `/tmp/tmem_copy_scales_cga_round10_probe.py`
- GPU/cache: `CUDA_VISIBLE_DEVICES=0`,
  `TRITON_CACHE_DIR=/tmp/triton-cache-round10-laneM`

## Scope

This lane probed the copy/scales frontier around:

- `tcgen05.cp` no-scales `warpx2::{01_23,02_13}` direct, indexed,
  subslice, and slice-index chains;
- `tcgen05.cp` scales `warpx4` direct, source-subslice, and descriptor-view
  paths;
- subword / `x1` no-scales `warpx2` rows;
- larger-CGA `num_ctas` requests around `{1,2,4,8,16}`;
- two-CTA layouts embedded in larger CGA contexts; and
- scaled-MMA copy / multicast setup rows that exercise the same copy planning
  machinery before and during `mma_scaled_tcgen05_copy`.

The probe ran each crashy or clean-boundary row in its own child Python
process. That kept the CUDA context alive even when a compile-time assertion
or parsing failure occurred.

## Commands

Required rebuild:

```bash
make -j8
```

Temporary probe:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-round10-laneM \
  PYTHONPATH=.:./python:./python/test/gluon python /tmp/tmem_copy_scales_cga_round10_probe.py
```

Exact positive confirmation for the harnessed two-CTA `warpx2::01_23` row:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-round10-laneM \
  PYTHONPATH=.:./python pytest -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_positive
```

## Probe Cases

Passes:

- `cp_scales_warpx4_direct`, shape `64x16`, direct scales copy;
- `cp_scales_warpx4_shared_subslice_row0`, source subslice `start_row=0`;
- `cp_scales_warpx4_shared_subslice_row64`, source subslice `start_row=64`;
- `cp_no_scales_warpx2_single_01_23`, shape `128x4`, `f32`;
- `cp_no_scales_warpx2_single_02_13`, shape `128x4`, `f32`;
- `cp_scales_mma_numctas1_false`, `cp_scales_mma_numctas1_true`,
  `cp_scales_mma_numctas2_false`, and `cp_scales_mma_numctas2_true`;
- exact runtime-matrix rerun of
  `test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_positive`
  passed `2/2` and invalidated the one temporary-wrapper false lead.

Clean unsupported / clean validation boundaries:

- `cp_scales_warpx4_descriptor_view`, shape `128x32`, destination-view copy
  over `TensorMemoryScalesLayout`;
- `cp_no_scales_warpx2_twocta_02_13_clean`,
  `cp_no_scales_warpx2_twocta_indexed_02_13_clean`, and
  `cp_no_scales_warpx2_twocta_slice_index_02_13_clean`;
- `cp_no_scales_warpx2_subword_single_f16` and
  `cp_no_scales_warpx2_subword_twocta_f16`;
- `cp_no_scales_twocta_context_4`, `cp_no_scales_twocta_context_8`, and
  `cp_no_scales_twocta_context_16`.

Assertion / guardrail row:

- `cp_scales_mma_numctas4_false`, `cp_scales_mma_numctas8_false`, and
  `cp_scales_mma_numctas16_false` all trip the same early CGA-shape assertion
  in `make_scales_descriptor`:

```text
AssertionError: Shape [1, 1, 1, 2, 256] is not divisible by CGA layout [[0, 1, 0, 0, 0]]
```

## Representative Diagnostics

No-scales `warpx2` subword boundary:

```text
direct tcgen05.copy.warpx2::01_23.64x128b requires enough TMEM column bases to cover the copy instruction width.
The destination-column footprint requirement exposes 2 column basis bits, but the 8-column copy instruction requires 3 logical column basis bits for 16-bit elements.
Support needs a packed-lane source/destination storage model...
This is reported as cleanly unsupported instead of falling through to late LLVM lowering.
```

Scales `warpx4` descriptor-view boundary:

```text
The source shared layout maps to tcgen05.copy.warpx4.32x128b, but Triton could not synthesize a compatible shared-memory descriptor plan for it.
...
The first differing physical-query field is physical layout for active view shape 128x32.
...
This is reported as cleanly unsupported instead of falling through to late LLVM lowering.
```

No-scales two-CTA `warpx2::02_13` boundary:

```text
tcgen05.copy.warpx2::02_13.64x128b descriptor message 0 has an unsupported source-row projection.
...
logical row bit 5 maps to a one-dword source offset rather than an affine 8-row source stride.
...
Decomposing this tensor-memory view into cta_group::1 copies is not valid because two-CTA TMEM allocation uses cta_group::2 granularity.
```

## Classification

No new `FZ-*` bucket was promoted from this lane.

What the lane shows:

- direct and source-subslice scales `warpx4` rows stay positive;
- the scales descriptor-view row stays a clean unsupported boundary;
- no-scales `warpx2` single-CTA positives remain positive for both families;
- no-scales `warpx2` two-CTA `02_13` rows remain clean unsupported;
- the two-CTA larger-CGA `num_ctas` contexts remain clean validation errors;
- the only assertion-like behavior was the scaled-MMA `num_ctas > 2` guardrail
  in `make_scales_descriptor`, which I left unpromoted because it is an early
  layout divisibility check rather than a backend miscompile or runtime crash.

The one temporary-wrapper false lead was the `cp_no_scales_warpx2_twocta_01_23`
row. I discarded that as a harness issue after the exact runtime-matrix
nodeid passed `2/2`.

Backend repair remains deferred until discovery stops finding new bugs or the
user explicitly pivots from discovery to repair.
