# Round 46 Lane: scaled-MMAv5 descriptor and scale-operand composition fuzzing

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend/compiler code or checked-in tests
were edited.

## Scope

This lane adversarially exercised scaled-MMAv5 composition surfaces:

- accumulator descriptor views, indexed accumulator views, and subslice
  accumulator views;
- B-scale descriptor views, including extra-user/rematerialization and padded
  storage variants;
- LHS subslice descriptors and LHS tile-permuted descriptors;
- two-CTA accumulator-subslice scaled-copy/MMAv5 flows;
- format mixes across `mxfp8`, `mxfp4`, and `nvfp4`;
- all-scale-before-real-accumulator allocation order and direct selected
  B-scale descriptors;
- dynamic accumulator-parent selection adjacent to direct and descriptor-view
  scale operands.

All failures reproduced existing buckets. No new independent `FZ-*` bucket is
needed.

## Required rebuild

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: no work to do.
```

## Checked-in scaled-MMAv5 composition surface

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and (acc_subslice or indexed_acc or bscale_descriptor_view or tile_permuted or lhs_subslice or lhs_tile_permuted or shared_scale_descriptor_view) and not reports and not resource'
```

Result: `210/1615` tests collected.

Runtime split commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (acc_subslice or indexed_acc or bscale_descriptor_view or tile_permuted or lhs_subslice or lhs_tile_permuted or shared_scale_descriptor_view) and not reports and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (acc_subslice or indexed_acc or bscale_descriptor_view or tile_permuted or lhs_subslice or lhs_tile_permuted or shared_scale_descriptor_view) and not reports and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (acc_subslice or indexed_acc or bscale_descriptor_view or tile_permuted or lhs_subslice or lhs_tile_permuted or shared_scale_descriptor_view) and not reports and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (acc_subslice or indexed_acc or bscale_descriptor_view or tile_permuted or lhs_subslice or lhs_tile_permuted or shared_scale_descriptor_view) and not reports and not resource'
```

Results:

| Group | Result |
| --- | --- |
| 1 | `53 passed, 1562 deselected` |
| 2 | `53 passed, 1562 deselected` |
| 3 | `53 passed, 1562 deselected` |
| 4 | `51 passed, 1564 deselected` |

Aggregate: `210 passed`.

Coverage includes indexed and subslice accumulator views, two-CTA
accumulator-subslice scaled-copy/MMAv5 rows, shared-scale descriptor-view
auto-copy rows, LHS subslice and tile-permuted descriptors, B-scale
descriptor-view rows, tile-permuted accumulator layouts, narrow accumulator
layouts, `use_acc`, and mixed format pairs. No compiler crash, unexpected
unsupported diagnostic, opcode absence, runtime miscompile, or skip drift was
observed in this checked-in positive surface.

## Checked-in clean-boundary surface

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and (reports_clean_error or reports_clean_unsupported or clean_error or clean_unsupported) and not resource'
```

Result: `27/1615` tests collected.

Runtime split commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (reports_clean_error or reports_clean_unsupported or clean_error or clean_unsupported) and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (reports_clean_error or reports_clean_unsupported or clean_error or clean_unsupported) and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (reports_clean_error or reports_clean_unsupported or clean_error or clean_unsupported) and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (reports_clean_error or reports_clean_unsupported or clean_error or clean_unsupported) and not resource'
```

Results:

| Group | Result |
| --- | --- |
| 1 | `7 passed, 1608 deselected` |
| 2 | `7 passed, 1608 deselected` |
| 3 | `7 passed, 1608 deselected` |
| 4 | `6 passed, 1609 deselected` |

Aggregate: `27 passed`.

These rows cover clean hardware/API boundaries for mixed fp4 LHS operand-A in
TMEM, narrow B-scale descriptor views, and narrow tile-permuted accumulators.
The expected diagnostics remained clean: no `PassManager::run failed`, no
assertion text, and no crash surfaced.

## Temporary probe compile check

Command:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile \
  /tmp/tmem_scaled_dynamic_scales_round22_probe.py \
  /tmp/tmem_scaled_multi_mma_round15_probe.py \
  /tmp/tmem_fz0015_side_channel_round19_probe.py \
  /root/tmp/tmem_scaled_mixed_operands_round24_probe.py
```

Result: pass.

## Dynamic scale descriptor-view probe

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_scaled_dynamic_scales_round22_probe.py
```

Result: `ROUND22_SUMMARY {'cases': 16, 'unexpected': 0, 'unexpected_names': []}`.

Passing direct-scale controls:

| Case family | Rows | Result |
| --- | ---: | --- |
| Direct B-scale branch/helper/nested/loop selection | 7 | pass, `0/16384` mismatches, A/B side probes `0`, PTX/LLIR op counts match |
| Direct A-scale selection | 1 | pass, `0/16384` mismatches |
| Direct A+B scale selection | 1 | pass, `0/16384` mismatches |
| Direct B-scale selection with two scaled MMAs | 1 | pass, `0/16384` mismatches, `8` PTX and LLIR scaled-MMAv5 ops |

Known `FZ-20260421-0013` scale descriptor-view rows:

| Case | Mismatches | Side probes | Opcode evidence |
| --- | ---: | --- | --- |
| `rtr_static_acc_first` | `16374/16384` | A/B `0` | PTX/LLIR `4/4` |
| `rtr_static_acc_last` | `16368/16384` | A/B `0` | PTX/LLIR `4/4` |
| `b_if_rtr_acc_first_sel0` | `16373/16384` | A/B `0` | PTX/LLIR `4/4` |
| `b_if_rtr_acc_first` | `16372/16384` | A/B `0` | PTX/LLIR `4/4` |
| `b_nested_rtr_acc_first` | `16373/16384` | A/B `0` | PTX/LLIR `4/4` |
| `both_if_rtr_acc_first` | `16371/16384` | A/B `0` | PTX/LLIR `4/4` |

Classification: existing `FZ-20260421-0013`. The descriptor-view payloads load
correctly through side channels, and the expected scaled-MMAv5 opcodes are
present, so the failure remains in scaled-MMAv5 consumption/lowering of scale
descriptor-view operands rather than scale stores, scale loads, or opcode
selection.

## Multi-MMA and selected scale descriptor probe

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_scale_descriptor_view_chains \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_acc_subslice_multi_mma \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_multi_mma_reused_scales \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_bscale_descriptor_selection \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_ascale_descriptor_selection \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_bscale_same_object_selection \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_branch_contained_bscale_mma \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_static_two_bscale_descriptors_use_first_control
```

Result: `5 failed, 14 passed`.

Known failures:

| Test | Symptom | Classification |
| --- | --- | --- |
| `test_am_scale_descriptor_view_chains[True-False]` | `16107/16384` mismatches | existing `FZ-20260421-0013` |
| `test_am_scale_descriptor_view_chains[False-True]` | `16116/16384` mismatches | existing `FZ-20260421-0013` |
| `test_am_scale_descriptor_view_chains[True-True]` | `16373/16384` mismatches | existing `FZ-20260421-0013` |
| `test_am_dynamic_bscale_descriptor_selection[0]` | `16381/16384` mismatches, NaN greatest difference | existing `FZ-20260421-0015` |
| `test_am_dynamic_bscale_descriptor_selection[1]` | `16380/16384` mismatches, NaN greatest difference | existing `FZ-20260421-0015` |

Passing controls included direct A/B scale descriptors, direct A-scale
selection, same-object B-scale selection, branch-contained B-scale MMA, static
two-B-scale selection, static accumulator subslice multi-MMA, and reused direct
scales across multiple scaled MMAs.

Classification: no new bucket. `FZ-0013` still requires a scale
descriptor-view chain. `FZ-0015` remains specific to runtime-selected distinct
direct B-scale descriptors consumed by scaled-MMAv5.

## FZ-0015 side-channel replay

Command:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_fz0015_side_channel_round19_probe.py::test_fz0015_round16
```

Result: `9 passed`. The probe encodes the known wrong-output rows as expected
classification checks.

Representative rows:

| Row | Side-channel scale result | Runtime result |
| --- | --- | --- |
| `direct_control` | `0/512` mismatches | pass, `0/16384` mismatches |
| `b_constexpr_distinct` | `0/512` mismatches | pass, `0/16384` mismatches |
| `b_runtime_same_object` | `0/512` mismatches | pass, `0/16384` mismatches |
| `b_runtime_distinct`, selector `0` | `0/512` mismatches | `16381/16384`, `64` NaNs, `90` Infs |
| `b_runtime_distinct`, selector `1` | `0/512` mismatches | `16379/16384`, `119` Infs |
| `b_helper_branch`, selector `0` | `0/512` mismatches | `16381/16384`, `170` Infs |
| `b_helper_branch`, selector `1` | `0/512` mismatches | `16384/16384`, `32` NaNs, `52` Infs |
| `b_loop_distinct`, selector `0` | `0/512` mismatches | `16378/16384`, `32` NaNs, `60` Infs |
| `b_loop_distinct`, selector `1` | `0/512` mismatches | `16384/16384`, `32` NaNs, `52` Infs |

Classification: existing `FZ-20260421-0015`. The side-channel loads confirm
the selected B-scale descriptor payload is correct before scaled-MMAv5
consumption.

## Mixed operand replay

Command:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /root/tmp/tmem_scaled_mixed_operands_round24_probe.py
```

Result: matched prior expected classification.

Green direct rows:

| Row | Result |
| --- | --- |
| `direct_b_after_scales_sel1` | pass, A/B/acc side probes `0`, output `0/16384`, PTX/LLIR `4/4` |
| `direct_b_before_scales_sel1` | pass, A/B/acc side probes `0`, output `0/16384`, PTX/LLIR `4/4` |
| `direct_acc_runtime_after_scales_sel1` | pass, A/B/acc side probes `0`, output `0/16384`, PTX/LLIR `4/4` |
| `direct_acc_runtime_b_after_scales_sel1` | pass, A/B/acc side probes `0`, output `0/16384`, PTX/LLIR `4/4` |

Known `FZ-20260421-0013` rows:

| Row | Symptom |
| --- | --- |
| `view_static_after_scales_sel0` | `16374/16384` mismatches, side probes `0`, PTX/LLIR `4/4` |
| `view_b_after_scales_sel1` | `16376/16384` mismatches, side probes `0`, PTX/LLIR `4/4` |
| `view_acc_runtime_after_scales_sel1` | `16375/16384` mismatches, side probes `0`, PTX/LLIR `4/4` |
| `view_acc_runtime_b_after_scales_sel1` | `16376/16384` mismatches, side probes `0`, PTX/LLIR `4/4` |
| `view_b_before_scales_sel1` | `16370/16384` mismatches, side probes `0`, PTX/LLIR `4/4` |
| `view_acc_runtime_b_before_scales_sel1` | `16372/16384` mismatches, side probes `0`, PTX/LLIR `4/4` |
| `view_b_after_scales_two_mma_sel1` | `16373/16384` mismatches, side probes `0`, PTX/LLIR `8/8` |
| `view_both_after_scales_sel1` | `16371/16384` mismatches, side probes `0`, PTX/LLIR `4/4` |

Classification: existing `FZ-20260421-0013`. Dynamic accumulator-parent
selection with direct scale operands passed. Dynamic accumulator-parent
selection combined with scale descriptor views failed, but the scale
descriptor-view operand dominates the failure and keeps it in `FZ-0013`, not a
new `FZ-0007` composition.

## Structural dynamic accumulator sentinel

Exact sentinel command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python \
  pytest -q -s --tb=short \
  python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_scaled_mma_acc_subslice_control_flow
```

Result: `1 xfailed`.

Adjacent structural collect:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'scaled_mma_acc_subslice_control_flow or dynamic_index_load_only or generic_pass_dynamic_index or generic_pass_loop_carried or copy_scales'
```

Result: `5/33` tests collected.

Adjacent structural command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python \
  pytest -q -s --tb=short \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'scaled_mma_acc_subslice_control_flow or dynamic_index_load_only or generic_pass_dynamic_index or generic_pass_loop_carried or copy_scales'
```

Result: `2 passed, 28 deselected, 3 xfailed`.

The `xfailed` rows reproduced the known dynamic/generic descriptor lowering
sentinels, including `FZ-20260421-0001` late illegal `ttg.memdesc_index` and
the scaled-MMAv5 dynamic accumulator-subslice sentinel for
`FZ-20260421-0007`. No XPASS, new crash signature, or new failure mode
appeared.

## Classification

No new independent `FZ-*` bucket is warranted from this lane.

- `FZ-20260421-0007`: revalidated only by the checked-in structural dynamic
  scaled accumulator-subslice sentinel. Runtime rows that dynamically select
  the accumulator parent but use direct scale descriptors passed, so this lane
  did not broaden `FZ-0007`.
- `FZ-20260421-0013`: strengthened. Any failing row with scale
  descriptor-view operands had correct side-channel scale loads and expected
  scaled-MMAv5 opcode counts, but wrong output from scaled-MMAv5. This includes
  static, branch-selected, dynamic accumulator-adjacent, allocation-order, and
  repeated-MMA variants.
- `FZ-20260421-0015`: revalidated. Runtime-selected distinct direct B-scale
  descriptors miscompile even though side-channel scale loads are correct.
  Static distinct B-scale selection, same-object runtime selection,
  branch-contained use, and direct A-scale selection pass.
- `FZ-20260421-0016`: not reproduced. No unencoded tensor operand crash or
  unencoded TMEM path appeared in this lane.
- `FZ-20260421-0017`: not reproduced. No encoded 64-bit TMEM load/store
  lowering assertion appeared; this lane stayed on scaled-MMAv5 uint8 scale
  and fp8/fp4 operand surfaces.

No checked-in positive row failed, no clean-boundary row regressed into a crash
or generic pass failure, no unexpected unsupported diagnostic appeared, and no
new compiler/runtime miscompile outside the known buckets was found.

