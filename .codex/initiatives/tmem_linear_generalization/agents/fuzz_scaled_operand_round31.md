# Round 31 Lane: scaled-MMAv5 TMEM descriptor operand fuzzing

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend/compiler code or checked-in tests
were edited.

## Scope

This lane focused on scaled-MMAv5 TMEM descriptor operands:

- scale descriptor views used as A-scale and B-scale operands;
- accumulator descriptor views and static accumulator subslices;
- mixed dynamic/static branches, helper returns, nested branches, and loop
  carried selected scale descriptors;
- reshape/permute/reshape descriptor chains and direct descriptor controls;
- 1CTA checked-in controls and 4/8/16 CTA high-CGA controls.

All findings classify into existing buckets. No new independent `FZ-*`
candidate was found.

## Required rebuild

```bash
make -j8
```

Result: `ninja: no work to do`.

## Temporary probes and artifacts

Temporary probes:

- `/tmp/tmem_scaled_dynamic_scales_round22_probe.py`
- `/tmp/tmem_scaled_multi_mma_round15_probe.py`
- `/tmp/tmem_high_cga_scaled_round26_probe.py`

Round-31 logs:

- `/tmp/tmem_scaled_operand_round31_checked_g0.log`
- `/tmp/tmem_scaled_operand_round31_checked_g1.log`
- `/tmp/tmem_scaled_operand_round31_checked_g2.log`
- `/tmp/tmem_scaled_operand_round31_checked_g3.log`
- `/tmp/tmem_scaled_operand_round31_dynamic_scales.log`
- `/tmp/tmem_scaled_operand_round31_dynamic_direct.log`
- `/tmp/tmem_scaled_operand_round31_views_multi.log`
- `/tmp/tmem_scaled_operand_round31_high_cga.log`

Py-compile:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile \
  /tmp/tmem_scaled_dynamic_scales_round22_probe.py \
  /tmp/tmem_scaled_multi_mma_round15_probe.py \
  /tmp/tmem_high_cga_scaled_round26_probe.py
```

Result: pass.

## Checked-in green scaled descriptor-view controls

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and descriptor_view and not reports'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and descriptor_view and not reports'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and descriptor_view and not reports'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and descriptor_view and not reports'
```

Result:

- group 1: `1 passed, 1614 deselected`
- group 2: `1 passed, 1614 deselected`
- group 3: `1 passed, 1614 deselected`
- group 4: `1 passed, 1614 deselected`
- aggregate selected result: `4 passed`

Classification: green control. This does not clear report-only
`FZ-20260421-0013`; it confirms the checked-in scaled descriptor-view rows
remain green.

## Dynamic scale-view and helper/control-flow probe

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_scaled_dynamic_scales_round22_probe.py
```

Result: `16` cases ran with `0` unexpected outcomes.

Counts:

- `10` direct selected scale descriptor rows passed, including branch,
  nested branch, helper-returned, loop-carried, A-scale selection, B-scale
  selection, both-scale selection, and two-MMA selection rows.
- `6` reshape/permute/reshape scale descriptor-view rows reproduced existing
  `FZ-20260421-0013`.

Representative passing rows:

| Case | Shape | Result |
| --- | --- | --- |
| `direct_b_if_acc_first` | direct B-scale runtime branch, accumulator allocated first | pass, `0/16384` mismatches, `4` PTX and LLIR scaled-MMA ops |
| `direct_b_helper_acc_last_sel1` | helper-returned direct B-scale descriptor | pass, `0/16384` mismatches |
| `direct_b_loop_acc_first` | loop-carried direct B-scale descriptor | pass, `0/16384` mismatches |
| `direct_b_if_two_mma` | direct B-scale selection, two scaled MMAs | pass, `0/16384` mismatches, `8` PTX and LLIR scaled-MMA ops |

Existing `FZ-20260421-0013` rows:

| Case | Descriptor chain | Mismatches | Classification |
| --- | --- | ---: | --- |
| `rtr_static_acc_first` | scale `reshape -> permute -> reshape` | `16374/16384` | existing `FZ-20260421-0013` |
| `rtr_static_acc_last` | same chain, accumulator allocated after scales | `16368/16384` | existing `FZ-20260421-0013` |
| `b_if_rtr_acc_first_sel0` | branch-selected B-scale descriptor view | `16373/16384` | existing `FZ-20260421-0013` |
| `b_if_rtr_acc_first` | branch-selected B-scale descriptor view | `16372/16384` | existing `FZ-20260421-0013` |
| `b_nested_rtr_acc_first` | nested branch B-scale descriptor view | `16373/16384` | existing `FZ-20260421-0013` |
| `both_if_rtr_acc_first` | branch-selected A-scale and B-scale descriptor views | `16371/16384` | existing `FZ-20260421-0013` |

In every `FZ-0013` row, side-channel loads of the selected scale descriptor
matched the expected payload (`a_probe_mismatch=0`, `b_probe_mismatch=0`),
and PTX/LLIR contained matching scaled-MMAv5 op counts. This keeps the
classification in scaled-MMAv5 scale descriptor operand consumption/lowering
rather than descriptor stores or loads.

## Direct dynamic scale descriptor and static controls

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_bscale_descriptor_selection \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_ascale_descriptor_selection \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_bscale_same_object_selection \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_branch_contained_bscale_mma \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_static_two_bscale_descriptors_use_first_control
```

Result: `2 failed, 7 passed`.

Failures:

| Case | Symptom | Classification |
| --- | --- | --- |
| `test_am_dynamic_bscale_descriptor_selection[0]` | `16381/16384` mismatches, NaN greatest difference | existing `FZ-20260421-0015` |
| `test_am_dynamic_bscale_descriptor_selection[1]` | `16380/16384` mismatches, NaN greatest difference | existing `FZ-20260421-0015` |

Passing controls:

- dynamic A-scale descriptor selection, selectors `0/1`;
- dynamic B-scale same-object selection, selectors `0/1`;
- branch-contained B-scale MMA, selectors `0/1`;
- static two-B-scale descriptors with one statically selected B-scale operand.

Classification: no new bucket. This keeps `FZ-20260421-0015` narrowed to a
merged runtime-selected distinct B-scale memdesc value consumed by
scaled-MMAv5. It is still distinct from `FZ-20260421-0013`, which needs a
descriptor-view chain, and from `FZ-20260421-0001`, because these rows compile
and execute.

## Scale descriptor-view and multi-MMA controls

Command:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_scale_descriptor_view_chains \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_acc_subslice_multi_mma \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_multi_mma_reused_scales
```

Result: `3 failed, 7 passed`.

Failures:

| Case | Mismatches | Classification |
| --- | ---: | --- |
| `test_am_scale_descriptor_view_chains[True-False]` | `16107/16384` | existing `FZ-20260421-0013` |
| `test_am_scale_descriptor_view_chains[False-True]` | `16116/16384` | existing `FZ-20260421-0013` |
| `test_am_scale_descriptor_view_chains[True-True]` | `16373/16384` | existing `FZ-20260421-0013` |

Passing controls:

- direct A/B scale descriptors;
- static accumulator subslice with two scaled-MMAv5 ops at slice starts `0`
  and `128`;
- reused direct scale descriptors across multi-MMA rows and format variations.

Classification: no new bucket. Accumulator subslices and scale reuse remain
green in this slice; only scale descriptor-view operands miscompile.

## High-CGA scaled controls

Command:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_high_cga_scaled_round26_probe.py
```

Result:

- `3` high-CGA scaled-MMAv5 controls passed for `num_ctas=4`, `8`, and `16`
  and emitted `tcgen05.mma.cta_group::2.kind::mxf8f6f4`.
- `24` mixed local TMEM rows reproduced existing `FZ-20260421-0010`.

`FZ-0010` matrix:

- high-CGA `4/8/16`;
- local TMEM layout CTA count `1/2`;
- local op `st`, `ld`, `ldred`, and `copy`.

Each row produced the known CTA-count diagnostic of the form
`<local> CTAs per CGA, but the context requires <num_ctas> CTAs per CGA`.
No high-CGA scaled-MMAv5 control failed independently.

## Classification

No new independent `FZ-*` bucket was found.

- `FZ-20260421-0013`: strengthened. The failing rows require scale
  descriptor-view chains consumed by scaled-MMAv5. Scale side-channel loads
  are correct, opcodes are present, and static/dynamic view variants all
  miscompile.
- `FZ-20260421-0015`: revalidated. The failing rows are direct distinct
  B-scale descriptors selected through a merged runtime memdesc value and then
  consumed by scaled-MMAv5. A-scale selection, same-object B-scale selection,
  branch-contained B-scale MMA, and static two-descriptor controls pass.
- `FZ-20260421-0010`: revalidated for high-CGA context mixing. High-CGA
  scaled-MMAv5 controls pass; adding otherwise-local 1CTA/2CTA TMEM
  descriptors in a 4/8/16 CTA context hits the known CTA-count gate.

No row reproduced `FZ-20260421-0001`: there was no late illegal
`ttg.memdesc_index` in this lane. No row produced a compiler crash,
unexpected unsupported diagnostic, or opcode mismatch outside the known
classifications above.

## Next scaled operand fuzzing slices

- Combine `FZ-0013` scale descriptor-view chains with the `FZ-0015`
  allocation-order/discriminator variants in one probe, while keeping direct
  branch-contained controls.
- Add 2CTA runtime probes for direct and descriptor-view scale operands using
  the same side-channel load checks.
- Add explicit accumulator-view dynamic selection rows next to the direct
  scale-selection controls to keep `FZ-0007` separate from `FZ-0013` and
  `FZ-0015`.
