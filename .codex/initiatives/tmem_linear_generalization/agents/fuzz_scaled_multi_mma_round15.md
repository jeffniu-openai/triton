# Round 15 Lane AM: scaled MMAv5 multi-op and scale/accumulator view fuzzing

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend/compiler code was changed.

## Summary

Lane AM focused on scaled MMAv5 composition cases:

- multiple scaled MMA operations in one kernel;
- reused A/B scale descriptors across multiple scaled MMA operations;
- static accumulator subslice views with multiple scaled MMA operations;
- A-scale and B-scale descriptor-view chains;
- dynamic B-scale TMEM descriptor selection through control flow;
- `mxfp8`, `mxfp4`, and `nvfp4` format variations;
- `use_acc=False` first-op plus `use_acc=True` accumulation, and
  `use_acc=True` with initialized accumulators.

No compiler crash was found. Seven rows are green. Three scale descriptor-view
rows reproduce existing `FZ-20260421-0013`. Two dynamic B-scale descriptor
selection rows are a new candidate bucket, tentatively
`FZ-20260421-0015`, because they are runtime wrong-results with a different
trigger from `FZ-0001`, `FZ-0007`, and `FZ-0013`.

## Required rebuild

Command:

```bash
make -j8
```

Result: no work to do.

## Temporary probe

Probe file:

```text
/tmp/tmem_scaled_multi_mma_round15_probe.py
```

Collect-only command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_scaled_multi_mma_round15_probe.py
```

Result: `12 tests collected`.

Full final command before the FZ-0015 discriminator row was added:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_scaled_multi_mma_round15_probe.py \
  2>&1 | tee /tmp/tmem_scaled_multi_mma_round15_full_final.log
```

Result: `7 passed, 5 failed`.

Additional FZ-0015 discriminator commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_static_two_bscale_descriptors_use_first_control \
  2>&1 | tee /tmp/tmem_scaled_multi_mma_round15_static_two_bscale.log
```

Result: `1 passed`.

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_bscale_descriptor_selection \
  2>&1 | tee /tmp/tmem_scaled_multi_mma_round15_dynamic_bscale_both_after_static.log
```

Result: `2 failed`.

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_ascale_descriptor_selection \
  2>&1 | tee /tmp/tmem_scaled_multi_mma_round15_dynamic_ascale.log
```

Result: `2 passed`.

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_branch_contained_bscale_mma \
  2>&1 | tee /tmp/tmem_scaled_multi_mma_round15_branch_contained_bscale_mma.log
```

Result: `2 passed`.

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_bscale_same_object_selection \
  2>&1 | tee /tmp/tmem_scaled_multi_mma_round15_dynamic_bscale_same_object.log
```

Result: `2 passed`.

Shape-broadening command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon:/tmp \
  python - <<'PY' 2>&1 | tee /tmp/tmem_scaled_multi_mma_round15_fz0015_shape_matrix.log
...
PY
```

Result: all tested dynamic distinct B-scale selector rows were wrong-result
rows.

## Green rows

The following rows passed runtime torch/reference comparison and PTX/LLIR
scaled-MMA opcode checks:

| Row | Focus | Result |
| --- | --- | --- |
| `multi-mma mxfp8/mxfp8 K=128 linear first_use_acc=False` | Two scaled MMA ops, same A/B scales reused, first op overwrites accumulator and second accumulates | passed |
| `multi-mma mxfp8/mxfp8 K=256 tile64 first_use_acc=True` | Two scaled MMA ops, initialized accumulator, reused scales, nontrivial accumulator layout | passed |
| `multi-mma mxfp4/mxfp4 K=128 linear first_use_acc=False` | FP4 scaled format with reused scale descriptors across two MMA ops | passed |
| `multi-mma nvfp4/nvfp4 K=128 linear first_use_acc=True` | NVFP4 scaled format with initialized accumulator and reused scale descriptors | passed |
| `scale descriptor view chains False/False` | Direct A-scale and direct B-scale controls in the descriptor-chain kernel | passed |
| `acc subslice multi-mma slice_start=0` | Static accumulator subslice, two scaled MMA ops, reused scales | passed |
| `acc subslice multi-mma slice_start=128` | High accumulator subslice, two scaled MMA ops, reused scales | passed |
| `static two B-scale descriptors, use first` | Allocates and populates two B-scale TMEM descriptors, but passes one statically to MMA | passed |
| `dynamic A-scale descriptor selection, selectors 0/1` | Runtime-selects between two identical A-scale TMEM descriptors | passed |
| `dynamic B-scale same-object selection, selectors 0/1` | Runtime branch assigns the same B-scale TMEM descriptor object from both arms | passed |
| `branch-contained B-scale MMA, selectors 0/1` | Calls scaled MMA inside the runtime branch with distinct B-scale descriptor operands, avoiding a merged B-scale memdesc value | passed |

These controls confirm that multiple scaled MMAv5 operations in one kernel,
scale descriptor reuse, static accumulator subslices, and `use_acc` sequencing
are not independently broken in the tested shapes.

## Existing `FZ-20260421-0013` broadening

Three rows fail as runtime wrong-results when either scale operand is supplied
through a TMEM descriptor-view chain:

| Row | Mismatches | Classification |
| --- | ---: | --- |
| `scale descriptor view chains True/False` | `16107 / 16384` | existing `FZ-20260421-0013` |
| `scale descriptor view chains False/True` | `16116 / 16384` | existing `FZ-20260421-0013` |
| `scale descriptor view chains True/True` | `16373 / 16384` | existing `FZ-20260421-0013` |

The descriptor chain used here is:

```text
reshape((MN / 2, 2, K / 32)) -> permute([1, 0, 2]) -> reshape((MN, K / 32))
```

This is consistent with the existing `FZ-0013` scaled-MMAv5 scale
descriptor-view wrong-result bucket. Lane AM broadens that bucket by showing
the same class is not limited to the previously minimized B-scale-only probe:
A-scale-only, B-scale-only, and both-scale view chains all miscompile.

## New candidate: `FZ-20260421-0015`

Two rows fail as runtime wrong-results when selecting between two B-scale TMEM
descriptors through a runtime branch. Both B-scale descriptors are populated
with identical scale data, so either branch should produce the same output.

| Row | Symptom | Opcode check | Classification |
| --- | --- | --- | --- |
| `dynamic_bscale_descriptor_selection[0]` | `16381 / 16384` mismatches; `96` NaNs in output | PTX/LLIR both contain four `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X` ops | new candidate `FZ-20260421-0015` |
| `dynamic_bscale_descriptor_selection[1]` | `16380 / 16384` mismatches; `96` NaNs in output | PTX/LLIR both contain four `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X` ops | new candidate `FZ-20260421-0015` |

Inspection command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon:/tmp \
  python /tmp/tmem_scaled_multi_mma_round15_inspect.py
```

Equivalent inline inspection was captured in:

```text
/tmp/tmem_scaled_multi_mma_round15_inspect.log
```

Observed inspection details:

- selector `0`: `96` output NaNs, `16287` finite mismatches, matching scaled
  MMA PTX/LLIR opcodes;
- selector `1`: `96` output NaNs, `16285` finite mismatches, matching scaled
  MMA PTX/LLIR opcodes.

Classification rationale:

- Not `FZ-20260421-0001`: this is not an illegal `ttg.memdesc_index` compiler
  failure at LLVM conversion; compilation succeeds and the kernel executes.
- Not `FZ-20260421-0007`: the dynamic value selects a B-scale descriptor, not
  an accumulator descriptor/view.
- Not `FZ-20260421-0013`: no reshape/permute descriptor-view chain is used on
  the selected B-scale operands. The direct scale and static descriptor-chain
  controls separate this trigger from pure scale descriptor-view lowering.

Recommended next discovery: minimize `FZ-0015` across same-object versus
distinct-object B-scale descriptors, A-scale versus B-scale dynamic selection,
branch form, selector constant folding, and single-use versus multi-use scale
descriptors before any backend repair work.

Additional discriminator after the initial report body:

- `test_am_static_two_bscale_descriptors_use_first_control` passed. Merely
  allocating and populating two B-scale TMEM descriptors is not sufficient.
- `test_am_dynamic_bscale_descriptor_selection` still failed for selectors `0`
  and `1` in the same fresh-process rerun. Runtime selection of the B-scale
  descriptor remains the sharp trigger.
- `test_am_dynamic_ascale_descriptor_selection` passed for selectors `0` and
  `1`, so this probe currently points at B-scale descriptor selection rather
  than arbitrary runtime-selected scale descriptors.
- `test_am_dynamic_bscale_same_object_selection` passed for selectors `0` and
  `1`; runtime assignment alone is not sufficient. The failing rows require
  selection between distinct B-scale TMEM descriptor objects.
- `test_am_branch_contained_bscale_mma` passed for selectors `0` and `1`;
  control flow around scaled MMA is not sufficient. The failing rows require a
  merged runtime B-scale descriptor value feeding one scaled MMA op.

Shape broadening for the failing merged B-scale descriptor trigger:

| N | K | Selector 0 | Selector 1 |
| ---: | ---: | --- | --- |
| 32 | 128 | `4096/4096` mismatches, `0` NaNs, 4 ops | `4096/4096` mismatches, `0` NaNs, 4 ops |
| 32 | 256 | `4096/4096` mismatches, `0` NaNs, 8 ops | `4096/4096` mismatches, `32` NaNs, 8 ops |
| 64 | 128 | `8191/8192` mismatches, `32` NaNs, 4 ops | `8192/8192` mismatches, `32` NaNs, 4 ops |
| 64 | 256 | `8192/8192` mismatches, `64` NaNs, 8 ops | `8192/8192` mismatches, `320` NaNs, 8 ops |
| 128 | 128 | `16383/16384` mismatches, `32` NaNs, 4 ops | `16383/16384` mismatches, `96` NaNs, 4 ops |
| 128 | 256 | `16384/16384` mismatches, `4256` NaNs, 8 ops | `16384/16384` mismatches, `736` NaNs, 8 ops |
| 256 | 128 | `32766/32768` mismatches, `96` NaNs, 4 ops | `32764/32768` mismatches, `32` NaNs, 4 ops |
| 256 | 256 | `32768/32768` mismatches, `3168` NaNs, 8 ops | `32768/32768` mismatches, `2624` NaNs, 8 ops |

This keeps `FZ-0015` broad across N/K while retaining the same trigger:
distinct B-scale TMEM descriptor values merged through runtime control flow.

Format sweep note:

- `/tmp/tmem_scaled_multi_mma_round15_fz0015_format_matrix.log` reran the
  dynamic distinct B-scale selector for `mxfp8`; both selectors reproduced
  wrong results (`16384/16384` and `16381/16384` mismatches).
- The attempted `mxfp4` and `nvfp4` rows in that quick sweep are discarded as
  probe-invalid because the temporary dynamic B-scale kernel was hardcoded for
  f8 shared-memory operands. They are not used for bucket decisions.

## Logs

Relevant temporary logs:

- `/tmp/tmem_scaled_multi_mma_round15_g0.log`
- `/tmp/tmem_scaled_multi_mma_round15_g1.log`
- `/tmp/tmem_scaled_multi_mma_round15_g2.log`
- `/tmp/tmem_scaled_multi_mma_round15_g3.log`
- `/tmp/tmem_scaled_multi_mma_round15_multi_rerun.log`
- `/tmp/tmem_scaled_multi_mma_round15_view_a.log`
- `/tmp/tmem_scaled_multi_mma_round15_view_b.log`
- `/tmp/tmem_scaled_multi_mma_round15_view_ab.log`
- `/tmp/tmem_scaled_multi_mma_round15_acc_subslice.log`
- `/tmp/tmem_scaled_multi_mma_round15_dyn_bscale0_rerun.log`
- `/tmp/tmem_scaled_multi_mma_round15_dyn_bscale1_rerun.log`
- `/tmp/tmem_scaled_multi_mma_round15_full_final.log`
- `/tmp/tmem_scaled_multi_mma_round15_inspect.log`
- `/tmp/tmem_scaled_multi_mma_round15_static_two_bscale.log`
- `/tmp/tmem_scaled_multi_mma_round15_dynamic_bscale_both_after_static.log`
- `/tmp/tmem_scaled_multi_mma_round15_dynamic_ascale.log`
- `/tmp/tmem_scaled_multi_mma_round15_dynamic_bscale_same_object.log`
- `/tmp/tmem_scaled_multi_mma_round15_branch_contained_bscale_mma.log`
- `/tmp/tmem_scaled_multi_mma_round15_fz0015_shape_matrix.log`
- `/tmp/tmem_scaled_multi_mma_round15_fz0015_format_matrix.log`
