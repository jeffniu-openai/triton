# Round 42 ld.red Positive Layout Guardrail

Date: 2026-04-21 14:16 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Surface

This lane targeted checked-in positive `tcgen05.ld.red` f32 rows over identity,
explicit-compatible, tile-permuted, column-permuted, row-permuted, and
row/column-permuted layouts. It intentionally excluded report/resource rows,
non-f32 rows, descriptor-chain rows, and M64 rows because those are covered by
separate clean-boundary, subword, high-rank, and M64 oracle lanes.

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ld_red_identity_linear_layout or ld_red_tile_permuted_linear_layout or ld_red_col_permuted_linear_layout or ld_red_row_permuted_linear_layout or ld_red_rowcol_permuted_n_sweep or ld_red_explicit_compatible_layout_variants) and not reports and not resource and not non_f32 and not descriptor_chain and not m64'
```

Result: `60/1615` tests collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(ld_red_identity_linear_layout or ld_red_tile_permuted_linear_layout or ld_red_col_permuted_linear_layout or ld_red_row_permuted_linear_layout or ld_red_rowcol_permuted_n_sweep or ld_red_explicit_compatible_layout_variants) and not reports and not resource and not non_f32 and not descriptor_chain and not m64'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k '(ld_red_identity_linear_layout or ld_red_tile_permuted_linear_layout or ld_red_col_permuted_linear_layout or ld_red_row_permuted_linear_layout or ld_red_rowcol_permuted_n_sweep or ld_red_explicit_compatible_layout_variants) and not reports and not resource and not non_f32 and not descriptor_chain and not m64'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k '(ld_red_identity_linear_layout or ld_red_tile_permuted_linear_layout or ld_red_col_permuted_linear_layout or ld_red_row_permuted_linear_layout or ld_red_rowcol_permuted_n_sweep or ld_red_explicit_compatible_layout_variants) and not reports and not resource and not non_f32 and not descriptor_chain and not m64'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k '(ld_red_identity_linear_layout or ld_red_tile_permuted_linear_layout or ld_red_col_permuted_linear_layout or ld_red_row_permuted_linear_layout or ld_red_rowcol_permuted_n_sweep or ld_red_explicit_compatible_layout_variants) and not reports and not resource and not non_f32 and not descriptor_chain and not m64'
```

## Results

- Group 1: `15 passed, 1600 deselected`
- Group 2: `15 passed, 1600 deselected`
- Group 3: `15 passed, 1600 deselected`
- Group 4: `15 passed, 1600 deselected`

Aggregate: `60 passed`.

## Classification

No compiler crash, false unsupported diagnostic, opcode absence, runtime
miscompile, unexpected pass/fail transition, or new independent `FZ-*` bucket
was observed.
