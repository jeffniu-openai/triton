# Round 26 Local: ld.red Runtime Slice

- Date: 2026-04-21 15:05 UTC
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler code was changed.

## Summary

The focused `ld.red` runtime slice collected `243` checked-in runtime-matrix
rows. The split-4 run produced `237` passing rows and `6` failures. All six
failures are the known M64 row-permuted destination-layout planner gap tracked
as `FZ-20260421-0012`.

No new compiler crash, runtime miscompile, opcode mismatch, false unsupported
bucket, or changed failure mode was found.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource'
```

Result:

```text
243/1615 tests collected (1372 deselected)
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <1..4> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource'
```

Observed split results:

```text
group 1: 55 passed, 6 failed
group 2: 61 passed
group 3: 61 passed
group 4: 60 passed
aggregate: 237 passed, 6 failed
```

Logs:

```text
/tmp/tmem_r26_ldred_g1.log
/tmp/tmem_r26_ldred_g2.log
/tmp/tmem_r26_ldred_g3.log
/tmp/tmem_r26_ldred_g4.log
```

## Failing Rows

The six failures are:

```text
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]
```

Each fails during Gluon parsing with:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
Failed to lower TMEM load/store: unsupported dst layout
```

This matches the existing `FZ-20260421-0012` classification: M64 f32
`tcgen05.ld.red` destination-layout planning does not support non-identity
effective row bases. Nearby controls in the same selector continued to pass,
including identity M64 split-N rows, M128 row/column permutations, column-only
M64 rows, descriptor-chain software-reduce rows, non-f32 software-reduce rows,
and compatible direct hardware-reduction rows.

## New Candidates

None.
