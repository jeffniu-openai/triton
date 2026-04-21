# Round 13 Local: `ld.red` M64 Row-Permuted Layouts

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Repo edit scope: this report only.
- New candidate: `FZ-20260421-0012`

## Scope

This local lane ran checked-in `ld.red` runtime-matrix coverage for descriptor
chains, tile-permuted layouts, row/column permutations, N sweeps, and non-f32
software reductions. The selector intentionally excluded `reports_*` clean
diagnostic tests, so collected rows are expected to compile and execute unless
they are strict xfails elsewhere.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and (tile_permuted or permuted or descriptor_chain or n_sweep) and not reports'
```

Result: `115/1615` collected.

Split-4 runtime command:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_local_r13_ldred_descriptor_permuted_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and (tile_permuted or permuted or descriptor_chain or n_sweep) and not reports'
```

Results:

```text
group 1/GPU 0: 23 passed, 6 failed, 1586 deselected in 15.60s
group 2/GPU 1: 29 passed, 1586 deselected in 8.27s
group 3/GPU 2: 29 passed, 1586 deselected in 8.42s
group 4/GPU 3: 28 passed, 1587 deselected in 22.27s
aggregate:     109 passed, 6 failed
```

Exact fresh-process confirmations:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]'
```

All six failed again with the same diagnostic family.

Nearby green controls:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[col_reverse_n32-min]'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[col_reverse_n32]'
```

Both passed.

## Failing Rows

```text
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]
```

Representative diagnostic:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
requested layout direct-lowering details:
Failed to lower TMEM load/store: unsupported dst layout
where out dims are: [row (size 128), col (size 32)]
```

The `row_rotate_col_even_odd_n128` rows report the same unsupported destination
layout family for output dims `[row (size 128), col (size 128)]`.

## Classification

`FZ-20260421-0012`: M64 row-permuted `ld.red` false unsupported / over-strict
destination-layout lowering.

Why this is a new candidate:

- these are checked-in non-report runtime-matrix rows, so they are expected to
  execute;
- failures reproduce in fresh exact processes and are not shard/cache noise;
- neighboring M64 `col_reverse_n32` default and explicit split-N controls pass;
- the failure is not an opcode fallback, allocator assertion, runtime
  miscompile, high-CGA CTA-count gate, or 2CTA row/col/block optimizer abort;
  and
- the affected rows are direct M64 row-permuted reductions, not the existing
  transpose/slice descriptor-view false-unsupported bucket.

No backend repairs were attempted.
