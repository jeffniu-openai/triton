# Round 19 Local: ld.red runtime selector

- Date: 2026-04-21 12:20 UTC
- Branch: `codex/tmem`
- Commit at start: `509e8f369`
- Mode: discovery/validation only. No backend/compiler fixes were attempted.
- Duration cache:
  `/tmp/tmem_r19_ldred_durations.json`

## Scope

This local slice checks broad `ld.red` runtime coverage:

- identity and explicit compatible layouts;
- M64 split-N rows;
- descriptor-chain rows;
- compatible non-identity and permuted layouts;
- software-reduce fallback rows.

Selector:

```text
ld_red and not reports and not resource
```

## Required Rebuild

Command:

```bash
make -j8
```

Result from immediately preceding local work: no work to do.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource'
```

Result: `243/1615` selected.

## Runtime

Running split-4 with stable per-GPU caches:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_r19_ldred_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `6 failed, 55 passed, 1554 deselected in 5.39s` |
| 2 | 1 | `61 passed, 1554 deselected in 4.28s` |
| 3 | 2 | `61 passed, 1554 deselected in 4.25s` |
| 4 | 3 | `60 passed, 1555 deselected in 4.20s` |

Aggregate selected rows: `237 passed, 6 failed`.

## Failures

All six failures are in the M64 row/col-permuted `ld.red` split-N family:

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
```

The failing rows are current checked-in runtime tests. They match the existing
`FZ-20260421-0012` bucket: M64 `tcgen05.ld.red` destination-layout lowering
rejects effective non-identity/permuted row bases in split-N shapes. No new
independent bucket is needed.

## Classification

- New independent `FZ-*`: none.
- Existing bucket reproduced: `FZ-20260421-0012`.
- Runtime miscompile: none observed.
- Compiler crash: none observed; the failures are compile-time unsupported dst
  layout diagnostics.
- Unexpected status: yes, these checked-in runtime rows are red on current
  branch and should remain in the repair backlog once the fuzzing campaign
  exits discovery-only mode.

## Focused Follow-Up

Exact reruns after the split completed:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]'
```

Result: `1 failed` with the same `unsupported dst layout` diagnostic.

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[col_reverse_n32-min]'
```

Result: `1 passed`.

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[col_reverse_n32]'
```

Result: `1 passed`.

The focused contrast supports the `FZ-0012` diagnosis: the red surface is tied
to effective row-base/permuted-row handling in M64 split-N `ld.red`, not to all
M64 split-N or all col-permuted layouts.
