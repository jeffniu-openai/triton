# Round 28 Lane: Broad `ld.red` Runtime Slice

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler fixes.
- Repo edit scope: this report only.

## Summary

A broad `ld_red and not reports and not resource` runtime slice was rerun
across four GPUs. It reproduced the stable M64 row/column-permuted reduction
coverage gap already tracked as `FZ-20260421-0012`; no new independent bucket
was found.

Aggregate:

```text
237 passed
6 failed
```

All six failures are `ttng.tmem_load` reduction planner failures with:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
Failed to lower TMEM load/store: unsupported dst layout
where out dims are: [row (size 128), col (size 32)]
```

or the same diagnostic for `[row (size 128), col (size 128)]`.

## Commands

Four-GPU shard:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short --splits 4 --group 1 \
  -k 'ld_red and not reports and not resource' \
  python/test/gluon/test_tmem_runtime_matrix.py \
  2>&1 | tee /tmp/tmem_round28_ldred_g1.log

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short --splits 4 --group 2 \
  -k 'ld_red and not reports and not resource' \
  python/test/gluon/test_tmem_runtime_matrix.py \
  2>&1 | tee /tmp/tmem_round28_ldred_g2.log

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short --splits 4 --group 3 \
  -k 'ld_red and not reports and not resource' \
  python/test/gluon/test_tmem_runtime_matrix.py \
  2>&1 | tee /tmp/tmem_round28_ldred_g3.log

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short --splits 4 --group 4 \
  -k 'ld_red and not reports and not resource' \
  python/test/gluon/test_tmem_runtime_matrix.py \
  2>&1 | tee /tmp/tmem_round28_ldred_g4.log
```

Exact failing-node rerun:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]' \
  2>&1 | tee /tmp/tmem_round28_ldred_exact_failures.log
```

## Results

Shard results:

```text
group 1: 55 passed, 6 failed, 1554 deselected
group 2: 61 passed, 1554 deselected
group 3: 61 passed, 1554 deselected
group 4: 60 passed, 1555 deselected
```

Exact rerun:

```text
6 failed in 3.43s
```

Failing nodeids:

```text
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]
```

## Classification

Existing `FZ-20260421-0012`: M64 f32 `tcgen05.ld.red` destination-layout
planner gap for non-identity row basis. The rerun confirms the failure is
stable and limited to the already-known M64 row/column-permuted rows inside
this selector; the remaining selected `ld.red` rows passed.

No backend/compiler code was changed.
