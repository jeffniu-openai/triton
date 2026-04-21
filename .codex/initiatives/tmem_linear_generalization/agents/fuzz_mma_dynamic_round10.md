# Round 10 Lane N: MMAv5 and scaled-MMAv5 dynamic descriptor-selection fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- HEAD at report time: `20b07cad5`
- Mode: discovery-only. No backend/compiler fixes were attempted.
- Temporary harness: `/tmp/tmem_mma_dynamic_round10_probe.py`

## Scope

This lane stressed dynamic accumulator-descriptor selection across plain
MMAv5 and scaled-MMAv5, with a focus on:

- dynamic `if` / selector-based view choice;
- helper-returned views;
- loop-carried views;
- sibling views;
- runtime selector indices;
- indexed, subslice, and slice-index style chains;
- `use_acc` true and false where meaningful;
- 1CTA and 2CTA controls;
- `N` in `{16, 32, 64, 128, 256}` where supported; and
- `K` in `{32, 64, 128, 256}` where supported.

The first draft of the temp probe included custom direct-control rows, but the
exact runtime-matrix controls below showed those were not faithful baselines.
Those direct rows were discarded from the final classification.

## Validation

Rebuild:

```bash
make -j8
```

Probe syntax check:

```bash
python -m py_compile /tmp/tmem_mma_dynamic_round10_probe.py
```

Representative subprocess-isolated worker rows:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r10-dyn-test4 \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_mma_dynamic_round10_probe.py --worker-case '{"case_id":"plain-runtime-index-n64-k128-sel0-acc0","family":"plain","mode":"runtime_index","n":64,"k":128,"use_acc":false,"selector":0,"loop_count":2,"num_ctas":1,"fmt_a":"mxfp8","fmt_b":"mxfp8","chain_id":0,"parent_kind":"linear","slice_start":0}'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r10-dyn-test5 \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_mma_dynamic_round10_probe.py --worker-case '{"case_id":"scaled-dyn-if-mxfp8xmxfp4-n64-k128-sel0-acc0","family":"scaled","mode":"dynamic_if","n":64,"k":128,"use_acc":false,"selector":0,"loop_count":2,"num_ctas":1,"fmt_a":"mxfp8","fmt_b":"mxfp4","chain_id":0,"parent_kind":"linear","slice_start":0}'
```

Exact runtime-matrix controls:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r10-direct-check0 \
PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_plain_kinds_use_acc[f16-linear-128-128]

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r10-direct-check1 \
PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_plain_kinds_use_acc[f16-linear-128-128]

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r10-direct-check2 \
PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_identity_narrow_format_matrix[mxfp8-mxfp8-16-128]

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r10-direct-check2 \
PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_identity_narrow_format_use_acc

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r10-subslice-check \
PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_acc_subslice_view_plain_kinds[f16-64-64-0-True]

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r10-direct-check1 \
PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_indexed_acc_view[f16-linear-128-128-True]

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r10-subslice-check \
PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_subslice_view_format_use_acc[mxfp8-mxfp8-64-0-128]

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r10-direct-check1 \
PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_use_acc[mxfp8-mxfp8-64-0-128-False]
```

Results from the exact runtime-matrix controls:

- `test_tmem_runtime_matrix_mma_plain_kinds_use_acc[f16-linear-128-128]`: pass;
- `test_tmem_runtime_matrix_mma_twocta_plain_kinds_use_acc[f16-linear-128-128]`: pass;
- `test_tmem_runtime_matrix_mma_scaled_acc_identity_narrow_format_matrix[mxfp8-mxfp8-16-128]`: pass;
- `test_tmem_runtime_matrix_mma_scaled_acc_identity_narrow_format_use_acc`: pass;
- `test_tmem_runtime_matrix_mma_acc_subslice_view_plain_kinds[f16-64-64-0-True]`: pass;
- `test_tmem_runtime_matrix_mma_twocta_indexed_acc_view[f16-linear-128-128-True]`: pass;
- `test_tmem_runtime_matrix_mma_scaled_acc_subslice_view_format_use_acc[mxfp8-mxfp8-64-0-128]`: pass;
- `test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_use_acc[mxfp8-mxfp8-64-0-128-False]`: pass.

## Findings

### Plain MMAv5

The plain MMAv5 helper / loop / sibling / indexed / subslice controls stayed
green:

- `plain-dyn-if-n64-k128-sel1-acc1`: pass, `0` mismatches, opcode match true;
- `plain-helper-n64-k128-sel0-acc0`: pass, `0` mismatches, opcode match true;
- `plain-helper-n64-k128-sel1-acc1`: pass, `0` mismatches, opcode match true;
- `plain-loop-n64-k128-sel1-acc1`: pass, `0` mismatches, opcode match true;
- `plain-sibling-n64-k128-acc0`: pass, `0` mismatches, opcode match true;
- `plain-indexed-n64-k128-acc1`: pass, `0` mismatches, opcode match true;
- `plain-subslice` control via the exact runtime-matrix node above: pass.

The plain runtime-selector-index rows were not clean controls. They compiled
successfully, matched PTX/LLIR opcodes, and then miscompiled at runtime:

- `plain-runtime-index-n64-k128-sel0-acc0`: `8169 / 8192` mismatches;
- `plain-runtime-index-n64-k128-sel1-acc0`: `8169 / 8192` mismatches;
- `plain-runtime-index-n64-k128-sel1-acc1`: `8159 / 8192` mismatches;
- `plain-runtime-index-n128-k128-sel1-acc1`: `16314 / 16384` mismatches.

Observed symptoms were NaN-heavy output mismatches with opcode agreement
preserved. There was no `ttg.memdesc_index` illegal-op text in these rows, so
this does not overlap `FZ-20260421-0001`. It also does not overlap the scaled
`FZ-20260421-0007` surface. Treat this as a plain-MMAv5 dynamic
runtime-selector-index miscompile candidate, likely a new bucket.

### Scaled MMAv5

The scaled dynamic-select rows reproduced the known `FZ-20260421-0007`
surface:

- `scaled-dyn-if-mxfp8xmxfp4-n64-k128-sel0-acc0`: `8187 / 8192` mismatches;
- `scaled-dyn-if-mxfp8xmxfp4-n64-k128-sel1-acc1`: `8166 / 8192` mismatches;
- `scaled-dyn-if-nvfp4xnfp4-n16-k128-sel1-acc1`: `2048 / 2048` mismatches.

The scaled runtime-selector and indexed rows also stayed on `FZ-20260421-0007`:

- `scaled-runtime-index-mxfp8xmxfp4-n64-k128-acc1`: `8189 / 8192` mismatches;
- `scaled-indexed-mxfp8xmxfp8-n32-k128-acc1`: `4095 / 4096` mismatches.

In all of those rows, PTX and LLIR MMA opcode families still matched, which
points to accumulator-view selection / lowering rather than opcode selection.

The exact scaled direct and twoCTA controls stayed green, including the
`mxfp8xmxfp8` `N=16` direct control, the `use_acc` variant, the 1CTA
subslice control, and the 2CTA subslice control.

## Classification

- Plain MMAv5 runtime-selector-index rows: likely new plain-MMAv5 bucket,
  not `FZ-20260421-0001` and not `FZ-20260421-0007`.
- Scaled MMAv5 dynamic-select / runtime-index / indexed rows:
  `FZ-20260421-0007` expansion.
- Plain helper / loop / sibling / indexed / subslice rows: green controls.
- Exact 1CTA and 2CTA runtime-matrix controls: green controls.

## Notes

- The custom temp probe is intentionally subprocess-isolated per row, so the
  crashy or miscompiling cases do not poison the whole run.
- No backend/source/test files were edited for the lane.
