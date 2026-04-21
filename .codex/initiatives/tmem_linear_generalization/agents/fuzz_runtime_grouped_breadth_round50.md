# Round 50 grouped runtime breadth sweep

Date: 2026-04-21
Branch: `codex/tmem`
Scope: broad but practical grouped runtime sweep over TMEM-bearing
`python/test/gluon/test_tmem_runtime_matrix.py` selectors. This round
cataloged and classified only; it did not modify backend code or checked-in
tests.

## Build

Command:

```bash
make -j8
```

Result: no-op rebuild succeeded.

## Partitioning

The sweep used three disjoint buckets:

- positive breadth: runtime cases expected to compile and execute correctly;
- known-red diagnostics: executable probes intentionally selecting currently
  known `ld.red` backend gaps;
- clean boundaries: cases expected to pass by reporting clean unsupported or
  clean resource diagnostics.

All pytest commands used stable per-GPU cache directories:
`/tmp/triton-cache-gpu0`, `/tmp/triton-cache-gpu1`,
`/tmp/triton-cache-gpu2`, and `/tmp/triton-cache-gpu3`.

## Positive Breadth

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q -k "(x1_subword or x1_f32_roundtrip or x1_f32_descriptor_chain_roundtrip or x1_i32_roundtrip or x1_i32_descriptor_chain_roundtrip or splitn_16bit or explicit_16x32bx2 or fixed_offset_patterns or ldst_scales_variant_sweep or ldst_scales_descriptor_view_cga_roundtrip or mma_plain_kinds_m64 or mma_acc_subslice_view_plain_kinds or mma_lhs_subslice_view_plain_kinds or mma_lhs_tile_permuted or mma_scaled_lhs_subslice_view_format_use_acc or mma_scaled_lhs_tile_permuted_format_use_acc or cp_no_scales_linear_tile_selector_permuted or cp_128x128_subword_exact_width) and not (unsupported_variants_report_clean_unsupported or reports_clean_error or reports_clean_unsupported)" python/test/gluon/test_tmem_runtime_matrix.py
```

Collection: `175/1615` tests collected, `1440` deselected.

Selected test functions:

```text
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_128x128_subword_exact_width
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_tile_selector_permuted
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_explicit_16x32bx2_matches_splitn
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_fixed_offset_patterns_128x256
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_scales_descriptor_view_cga_roundtrip
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_scales_variant_sweep
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_f32_descriptor_chain_roundtrip
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_f32_roundtrip
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_i32_descriptor_chain_roundtrip
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_i32_roundtrip
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_subword_roundtrip
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_subword_twocta_descriptor_chain_roundtrip
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_subword_twocta_roundtrip
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_acc_subslice_view_plain_kinds
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds_use_acc
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_lhs_tile_permuted
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_lhs_tile_permuted_use_acc
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_plain_kinds_m64
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view_format_use_acc
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_lhs_tile_permuted_format_use_acc
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_16bit_m64_auto_matches_explicit
```

Split commands used the same selector with:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> -k "<selector>" python/test/gluon/test_tmem_runtime_matrix.py
```

Split results:

- group 1 on GPU 0: `44 passed, 1571 deselected in 3.67s`
- group 2 on GPU 1: `44 passed, 1571 deselected in 4.03s`
- group 3 on GPU 2: `44 passed, 1571 deselected in 3.72s`
- group 4 on GPU 3: `43 passed, 1572 deselected in 4.44s`

Classification: positive breadth is green. No compiler crash, verifier drift,
wrong result, skip drift, or new `FZ-*` bucket.

## Known-Red Diagnostics

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q -k "(ld_red_m64_rowcol_permuted_default_layout or ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn or ld_red_descriptor_chain_n_sweep_explicit_variants or ld_red_explicit_n_sweep_variants)" python/test/gluon/test_tmem_runtime_matrix.py
```

Collection: `25/1615` tests collected, `1590` deselected.

Selected test functions:

```text
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_descriptor_chain_n_sweep_explicit_variants
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_explicit_n_sweep_variants
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn
```

Split results:

- group 1 on GPU 0: `5 failed, 2 passed, 1608 deselected in 5.21s`
- group 2 on GPU 1: `1 failed, 6 passed, 1608 deselected in 4.40s`
- group 3 on GPU 2: `7 passed, 1608 deselected in 4.00s`
- group 4 on GPU 3: `4 passed, 1611 deselected in 4.07s`

Failures:

```text
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]
```

Failure signature: `ttng.tmem_load` failed to compute TMEM encoding info for
reduction, with `Failed to lower TMEM load/store: unsupported dst layout`.

Classification: all six failures are existing `FZ-20260421-0012`, the M64 f32
`tcgen05.ld.red` destination-layout planner gap for non-identity row basis. No
new known-red signature appeared.

## Clean Boundaries

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q -k "(unsupported_variants_report_clean_unsupported or reports_clean_error or reports_clean_unsupported or block_descriptor_reports_clean_error or block_layout_reports_clean_error or scaled_acc_n16 or mixed_fp4a_reports_clean_unsupported or mma_i8_reports_clean_error or mma_m64_i8_reports_clean_error or twocta_i8_reports_clean_error or cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error or cp_no_scales_linear_exotic_reports_clean_unsupported)" python/test/gluon/test_tmem_runtime_matrix.py
```

Collection: `149/1615` tests collected, `1466` deselected.

Selected test functions:

```text
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_block_descriptor_reports_clean_error
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_blocked_layout_reports_clean_error
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_4x256b_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_exotic_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_rowcol_permuted_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_shared_subslice_bad_offset_reports_clean_error
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_transposed_shared_reports_clean_error
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_noncanonical_block_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_candidate_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_dense_shared_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_indexed_view_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_slice_index_view_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_subslice_view_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_row_permuted_destination_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_tmem_descriptor_view_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_4x256b_refresh_layout_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_4x256b_refresh_raw_bitcast_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_scales_variant_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_f32_unsupported_variants_report_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_i32_unsupported_variants_report_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_exotic_layout_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_i8_reports_clean_error
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_m64_i8_reports_clean_error
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_rowcol_permuted_layout_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_n16_bscale_descriptor_view_reports_clean_error
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_n16_tile_permuted_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view_mixed_fp4a_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_lhs_tile_permuted_mixed_fp4a_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_i8_reports_clean_error
```

Split results:

- group 1 on GPU 0: `38 passed, 1577 deselected in 9.59s`
- group 2 on GPU 1: `38 passed, 1577 deselected in 4.68s`
- group 3 on GPU 2: `38 passed, 1577 deselected in 4.28s`
- group 4 on GPU 3: `35 passed, 1580 deselected in 10.59s`

Classification: clean-boundary diagnostics are stable and green. No diagnostic
drift, unexpected pass/fail, resource-boundary drift, or new `FZ-*` bucket.

## Summary

- Total selected runtime-matrix coverage in final disjoint buckets:
  `349` collected cases.
- Final positive breadth: `175 passed`.
- Final clean-boundary diagnostics: `149 passed`.
- Known-red diagnostics: `19 passed`, `6 failed`, all existing
  `FZ-20260421-0012`.
- New `FZ-*`: none.
- Backend/code changes: none.
