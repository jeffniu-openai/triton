# Round 56 Lane B: scaled-MMAv5 dynamic scale and accumulator interactions

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `feeebd8de`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code was edited.

## Scope

This lane targeted scaled-MMAv5 interactions around dynamic scale operands and
accumulator views:

- dynamic/direct B-scale selection and adjacent A-scale/same-object controls;
- scale descriptor-view operands using `reshape -> permute/trans -> reshape`;
- accumulator subslice and index views with `use_acc=True`;
- linear and legacy accumulator layouts;
- 1CTA and 2CTA scaled-MMAv5 rows, plus high-CGA ownership controls;
- adjacent green controls for direct scales, static accumulator subslices,
  reused scales, and high-CGA scaled-MMAv5 without local TMEM mixing.

The relevant existing buckets for this lane were `FZ-20260421-0007`,
`FZ-20260421-0013`, `FZ-20260421-0015`, and `FZ-20260421-0010`.

## Required build

```bash
make -j8
```

Result: `ninja: no work to do`.

## Checked-in scaled interaction guardrail

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and (root_format_use_acc or indexed_acc_view_format_use_acc or acc_subslice_view_format_use_acc or twocta_acc_subslice_view_format_use_acc or bscale_descriptor_view or shared_scale_descriptor_view_auto_tmem_copy or acc_identity_narrow_format_use_acc or indexed_acc_identity_narrow_view_format_use_acc) and not reports'
```

Result: `55/1615` collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (root_format_use_acc or indexed_acc_view_format_use_acc or acc_subslice_view_format_use_acc or twocta_acc_subslice_view_format_use_acc or bscale_descriptor_view or shared_scale_descriptor_view_auto_tmem_copy or acc_identity_narrow_format_use_acc or indexed_acc_identity_narrow_view_format_use_acc) and not reports'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (root_format_use_acc or indexed_acc_view_format_use_acc or acc_subslice_view_format_use_acc or twocta_acc_subslice_view_format_use_acc or bscale_descriptor_view or shared_scale_descriptor_view_auto_tmem_copy or acc_identity_narrow_format_use_acc or indexed_acc_identity_narrow_view_format_use_acc) and not reports'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (root_format_use_acc or indexed_acc_view_format_use_acc or acc_subslice_view_format_use_acc or twocta_acc_subslice_view_format_use_acc or bscale_descriptor_view or shared_scale_descriptor_view_auto_tmem_copy or acc_identity_narrow_format_use_acc or indexed_acc_identity_narrow_view_format_use_acc) and not reports'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (root_format_use_acc or indexed_acc_view_format_use_acc or acc_subslice_view_format_use_acc or twocta_acc_subslice_view_format_use_acc or bscale_descriptor_view or shared_scale_descriptor_view_auto_tmem_copy or acc_identity_narrow_format_use_acc or indexed_acc_identity_narrow_view_format_use_acc) and not reports'
```

Result:

- group 1: `14 passed, 1601 deselected`
- group 2: `14 passed, 1601 deselected`
- group 3: `14 passed, 1601 deselected`
- group 4: `13 passed, 1602 deselected`
- aggregate: `55 passed`

Classification: green guardrail. The selected rows covered direct `use_acc`
scaled-MMAv5 roots, indexed accumulator views, accumulator subslices, 2CTA
accumulator subslice/copy paths, B-scale descriptor-view positives, shared
scale descriptor-view auto-copy, and narrow accumulator/index controls.

## Temporary probes

Py-compile:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile \
  /tmp/tmem_scaled_dynamic_scales_round22_probe.py \
  /tmp/tmem_scaled_multi_mma_round15_probe.py \
  /tmp/tmem_high_cga_scaled_round26_probe.py
```

Result: pass.

### Dynamic scale-view and helper/control-flow probe

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_scaled_dynamic_scales_round22_probe.py
```

Result: `16` cases, `0` unexpected.

Passing direct-scale controls:

- `direct_b_if_acc_first`, selector `0`, seed from probe corpus:
  `0/16384` mismatches, `4` PTX and LLIR scaled-MMA ops.
- `direct_b_if_acc_last_sel0`, selector `0`: `0/16384`, `4` ops.
- `direct_b_if_acc_last_sel1`, selector `1`: `0/16384`, `4` ops.
- `direct_b_helper_acc_last_sel1`, selector `1`: `0/16384`, `4` ops.
- `direct_b_nested_acc_first`, selector `1`: `0/16384`, `4` ops.
- `direct_b_helper_acc_first`, selector `1`: `0/16384`, `4` ops.
- `direct_b_loop_acc_first`, selector `1`: `0/16384`, `4` ops.
- `direct_a_if_acc_first`, selector `1`: `0/16384`, `4` ops.
- `direct_both_if_acc_first`, selector `1`: `0/16384`, `4` ops.
- `direct_b_if_two_mma`, selector `1`: `0/16384`, `8` ops.

Existing `FZ-20260421-0013` scale descriptor-view rows:

| Case | Selector | Mismatches | Notes |
| --- | ---: | ---: | --- |
| `rtr_static_acc_first` | 0 | `16374/16384` | scale descriptor view, `4` PTX/LLIR ops |
| `rtr_static_acc_last` | 0 | `16368/16384` | scale descriptor view, `4` PTX/LLIR ops |
| `b_if_rtr_acc_first_sel0` | 0 | `16373/16384` | branch-selected B-scale view |
| `b_if_rtr_acc_first` | 1 | `16372/16384` | branch-selected B-scale view |
| `b_nested_rtr_acc_first` | 1 | `16373/16384` | nested B-scale view |
| `both_if_rtr_acc_first` | 1 | `16371/16384` | A-scale and B-scale views |

All six `FZ-0013` rows had `a_probe_mismatch=0` and `b_probe_mismatch=0`, so
the side-channel scale loads remained correct while scaled-MMAv5 consumption
miscompiled.

### Direct dynamic B-scale selection and controls

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

Failures, both existing `FZ-20260421-0015`:

- `test_am_dynamic_bscale_descriptor_selection[0]`:
  `16381/16384` mismatches, NaN greatest difference.
- `test_am_dynamic_bscale_descriptor_selection[1]`:
  `16380/16384` mismatches, NaN greatest difference.

Passing controls:

- dynamic A-scale descriptor selection, selectors `0/1`;
- dynamic B-scale same-object selection, selectors `0/1`;
- branch-contained B-scale MMA, selectors `0/1`;
- static two-B-scale descriptors with one statically selected operand.

### Scale descriptor-view and multi-MMA controls

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

Failures, all existing `FZ-20260421-0013`:

- `test_am_scale_descriptor_view_chains[True-False]`:
  `16107/16384` mismatches.
- `test_am_scale_descriptor_view_chains[False-True]`:
  `16116/16384` mismatches.
- `test_am_scale_descriptor_view_chains[True-True]`:
  `16373/16384` mismatches.

Passing controls:

- direct A/B scale descriptors;
- accumulator subslice multi-MMA rows at slice starts `0` and `128`;
- reused direct scale descriptors across multi-MMA rows.

### High-CGA scaled controls and local TMEM ownership boundaries

Command:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_high_cga_scaled_round26_probe.py
```

Result:

- `3` high-CGA scaled-MMAv5 controls passed for `num_ctas=4`, `8`, and `16`.
- `24` mixed local TMEM rows reproduced existing `FZ-20260421-0010`.

`FZ-0010` matrix:

- high-CGA context: `4/8/16`;
- local TMEM layout CTA count: `1/2`;
- local op: `st`, `ld`, `ldred`, and `copy`.

Each failing row reported the known CTA-count diagnostic, such as
`1 CTAs per CGA, but the context requires 4 CTAs per CGA` or the corresponding
`2` versus `4/8/16` form. No high-CGA scaled-MMAv5 control failed on its own.

### Checked-in FZ-0007 accumulator-selection sentinel

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python \
  pytest -q -s --tb=short \
  python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_scaled_mma_acc_subslice_control_flow
```

Result: `1 xfailed`.

Classification: unchanged existing `FZ-20260421-0007`, the scaled-MMAv5
`use_acc` low-subslice accumulator selected through dynamic control flow.

## Classification

No new independent `FZ-*` bucket was found.

- `FZ-20260421-0007`: unchanged. The checked-in dynamic accumulator subslice
  `use_acc` sentinel still xfails.
- `FZ-20260421-0013`: revalidated. Scale descriptor-view operands consumed by
  scaled-MMAv5 still wrong-result even when side-channel scale loads are
  correct and PTX/LLIR scaled-MMA op counts match.
- `FZ-20260421-0015`: revalidated. Runtime-selected distinct direct B-scale
  descriptors still wrong-result. Adjacent A-scale selection, same-object
  B-scale selection, branch-contained B-scale MMA, and static direct B-scale
  controls pass.
- `FZ-20260421-0010`: revalidated. Legal high-CGA scaled-MMAv5 controls pass,
  while adding local 1CTA/2CTA TMEM operations in `4/8/16` CTA contexts hits
  the known CTA-count ownership diagnostic.

No row produced a compiler crash, verifier drift, opcode mismatch, false
unsupported diagnostic, clean-boundary drift, hang, or unclassified runtime
miscompile.
