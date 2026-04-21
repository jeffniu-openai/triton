# Round 24 Lane BH: Mixed Scaled-MMAv5 Operand Fuzzer

- Date: 2026-04-21 13:45 UTC
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler fixes.
- Repo edit scope: this report only.
- Target: interaction between existing `FZ-20260421-0007`,
  `FZ-20260421-0013`, and `FZ-20260421-0015` by mixing dynamic selected
  accumulator views, scale descriptor views, selected direct B-scale
  descriptors, allocation-order variants, and side-channel `tmem_load`
  checks in the same kernels.

## Summary

No new independent `FZ-*` bucket is warranted from this lane.

The mixed probes separate the existing scaled-MMAv5 failures cleanly:

- Direct selected B-scale descriptors plus runtime-selected accumulator-parent
  views passed in the mixed kernels. Side-channel loads from the selected A
  and B scale descriptors and the selected accumulator initialization readback
  were correct.
- Every mixed kernel that fed scaled-MMAv5 through a
  reshape/permute/reshape scale descriptor view miscompiled, even when the
  same selected scale descriptor had `0` side-channel load mismatches.
- Adding runtime-selected accumulator views and selected direct B-scale
  descriptors on top of scale descriptor views did not change the failure
  shape: the result stayed in the existing `FZ-20260421-0013` family.
- The exact previous `FZ-20260421-0015` side-channel reproducer still fails on
  this HEAD, so the direct selected-B-scale allocation-order bug is still
  live. The mixed direct probes are negative contrasts, not evidence that
  `FZ-0015` is fixed. They show that `FZ-0015` is more structurally sensitive
  than "any direct runtime selected B-scale".

Current classification:

```text
FZ-20260421-0007 remains dynamic-accumulator-view-specific.
FZ-20260421-0013 remains scale-descriptor-view operand-consumption-specific.
FZ-20260421-0015 remains direct selected-B-scale/all-scale-before-real-accumulator
allocation-order-specific, but does not automatically compose with the mixed
parent-accumulator probes below.
```

## Commands

Required rebuild gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Primary mixed probe:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /root/tmp/tmem_scaled_mixed_operands_round24_probe.py

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /root/tmp/tmem_scaled_mixed_operands_round24_probe.py \
  2>&1 | tee /tmp/tmem_scaled_mixed_operands_round24_probe.log
```

Singleton-accumulator contrast probe:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /root/tmp/tmem_scaled_mixed_singleton_acc_round24_probe.py

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /root/tmp/tmem_scaled_mixed_singleton_acc_round24_probe.py \
  2>&1 | tee /tmp/tmem_scaled_mixed_singleton_acc_round24_probe.log
```

Exact previous `FZ-0015` sanity rerun:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_fz0015_side_channel_round19_probe.py::test_fz0015_round16 \
  --continue-on-collection-errors \
  2>&1 | tee /tmp/tmem_round24_exact_fz0015_rerun.log
```

## Primary Mixed Probe Results

All rows use `M=N=K=128`, `mxfp8`, one CTA, side-channel `tmem_load` probes for
the selected A and B scale descriptors, a side-channel selected-accumulator
zero readback where applicable, and scaled-MMAv5 opcode counting in PTX and
LLIR.

| Row | Operand mix | Side-channel mismatches | Scaled-MMAv5 result | Opcode count |
| --- | --- | --- | --- | --- |
| `direct_b_after_scales_sel1` | direct selected B, accumulator parent after scales | A `0`, B `0`, acc `0` | pass, `0/16384`, no NaN/Inf | PTX `4`, LLIR `4` |
| `direct_b_before_scales_sel1` | direct selected B, accumulator parent before scales | A `0`, B `0`, acc `0` | pass, `0/16384`, no NaN/Inf | PTX `4`, LLIR `4` |
| `direct_acc_runtime_after_scales_sel1` | runtime-selected accumulator parent, direct scales | A `0`, B `0`, acc `0` | pass, `0/16384`, no NaN/Inf | PTX `4`, LLIR `4` |
| `direct_acc_runtime_b_after_scales_sel1` | runtime-selected accumulator parent plus direct selected B | A `0`, B `0`, acc `0` | pass, `0/16384`, no NaN/Inf | PTX `4`, LLIR `4` |
| `view_static_after_scales_sel0` | static reshape/permute/reshape scale views | A `0`, B `0`, acc `0` | fail, `16374/16384`, max finite diff `783.6383` | PTX `4`, LLIR `4` |
| `view_b_after_scales_sel1` | scale views plus selected B | A `0`, B `0`, acc `0` | fail, `16376/16384`, max finite diff `924.7694` | PTX `4`, LLIR `4` |
| `view_acc_runtime_after_scales_sel1` | scale views plus runtime-selected accumulator parent | A `0`, B `0`, acc `0` | fail, `16375/16384`, max finite diff `916.7793` | PTX `4`, LLIR `4` |
| `view_acc_runtime_b_after_scales_sel1` | scale views plus selected B plus runtime-selected accumulator parent | A `0`, B `0`, acc `0` | fail, `16376/16384`, max finite diff `1062.4961` | PTX `4`, LLIR `4` |
| `view_b_before_scales_sel1` | scale views plus selected B, accumulator parent before scales | A `0`, B `0`, acc `0` | fail, `16370/16384`, max finite diff `854.0538` | PTX `4`, LLIR `4` |
| `view_acc_runtime_b_before_scales_sel1` | scale views plus selected B plus runtime-selected accumulator parent before scales | A `0`, B `0`, acc `0` | fail, `16372/16384`, max finite diff `730.4115` | PTX `4`, LLIR `4` |
| `view_b_after_scales_two_mma_sel1` | scale views plus selected B, two MMAs | A `0`, B `0`, acc `0` | fail, `16373/16384`, max finite diff `1641.2550` | PTX `8`, LLIR `8` |
| `view_both_after_scales_sel1` | scale views plus selected A and B | A `0`, B `0`, acc `0` | fail, `16371/16384`, max finite diff `865.2955` | PTX `4`, LLIR `4` |

No row produced a compiler crash, unexpected unsupported diagnostic, or opcode
count mismatch. The failing scale-view rows all emitted scaled-MMAv5
instructions and all side-channel descriptor loads were correct.

## Singleton-Accumulator Contrast Results

This second probe removed the runtime-selected accumulator parent and used a
single accumulator allocation to compare against the all-scale-before-real-acc
shape implicated in `FZ-0015`.

| Row | Operand mix | Side-channel mismatches | Scaled-MMAv5 result | Opcode count |
| --- | --- | --- | --- | --- |
| `direct_select_b_acc_after_b` | direct selected B, singleton acc after B scales | A `0`, B `0` | pass, `0/16384`, no NaN/Inf | PTX `4`, LLIR `4` |
| `direct_select_b_acc_before_b` | direct selected B, singleton acc before B scales | A `0`, B `0` | pass, `0/16384`, no NaN/Inf | PTX `4`, LLIR `4` |
| `view_static_acc_after_b` | scale views, singleton acc after B scales | A `0`, B `0` | fail, `16368/16384`, max finite diff `625.9279` | PTX `4`, LLIR `4` |
| `view_select_b_acc_after_b` | scale views plus selected B, singleton acc after B scales | A `0`, B `0` | fail, `16371/16384`, max finite diff `667.2988` | PTX `4`, LLIR `4` |
| `view_select_b_acc_before_b` | scale views plus selected B, singleton acc before B scales | A `0`, B `0` | fail, `16373/16384`, max finite diff `836.3613` | PTX `4`, LLIR `4` |

This contrast again points at `FZ-0013` for descriptor-view scale operands.
The direct singleton rows are useful negatives but do not replace the exact
`FZ-0015` reproducer.

## Exact `FZ-0015` Rerun

The previous side-channel reproducer stayed live:

| Row family | Representative result |
| --- | --- |
| `b_runtime_distinct`, selector `0` | side-channel `0/512`; scaled-MMAv5 `16381/16384` mismatches, `64` NaNs, `90` Infs |
| `b_runtime_distinct`, selector `1` | side-channel `0/512`; scaled-MMAv5 `16379/16384` mismatches, `119` Infs |
| `b_helper_branch`, selectors `0/1` | side-channel `0/512`; scaled-MMAv5 `16381/16384` and `16384/16384` mismatches |
| `b_loop_distinct`, selectors `0/1` | side-channel `0/512`; scaled-MMAv5 `16378/16384` and `16384/16384` mismatches |

The `direct_control`, `b_constexpr_distinct`, and `b_runtime_same_object`
controls still passed. The pytest file reports `9 passed` because those probe
rows encode the known miscompile as the expected classification; the printed
metrics above are the relevant evidence.

## Classification

The mixed rows did not expose a failure mode that is independent of the
existing buckets.

The strongest reading is:

- `FZ-0013` dominates whenever the scale operand reaching scaled-MMAv5 is a
  descriptor view. Runtime-selected accumulator views and selected B-scale
  values do not change the symptom: side-channel loads from the same selected
  descriptors are correct, but the scaled-MMAv5 result is wrong.
- `FZ-0015` remains narrower and more sensitive to the precise direct selected
  B-scale SSA/allocation/probe structure than the broad mixed probes. The
  exact old reproducer still fails, but nearby direct mixed kernels can pass.
- `FZ-0007` does not appear as a new composition in this lane because the
  dynamic accumulator-parent rows with direct scale descriptors passed, and
  the dynamic-accumulator plus scale-view rows are already explained by
  `FZ-0013`.

Future repair work should keep these buckets separate until fixes prove
otherwise: one repair for scaled-MMAv5 descriptor-view operand consumption,
one for direct selected B-scale allocation/address encoding, and one for
dynamic accumulator-view consumption.
