# Round 52 Duration-Aware Runtime-Matrix Grouping

Date: 2026-04-21
Branch: `codex/tmem`
Scope: duration-noted grouped sweeps for TMEM-bearing rows in `python/test/gluon/test_tmem_runtime_matrix.py`.

This round did not modify backend code or checked-in tests. It built reusable `/tmp`
duration files for future local fuzz throughput and kept positives, clean
diagnostics, and known-red rows in separate lanes.

## Preflight

```bash
make -j8
```

Result: no-op rebuild succeeded.

## Selectors

Positive runtime selector:

```bash
python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (ldst_descriptor_rank5_small_roundtrip or ldst_descriptor_rank5_unit_parent_n256_roundtrip or ldst_x1 or ldst_scales_direct_roundtrip or ldst_scales_descriptor_view or cp_no_scales_warpx2_01_23 or cp_no_scales_linear_tile_permuted or cp_128x128 or cp_scales_warpx4_via_scaled_mma_geometry_sweep or mma_plain_kinds or mma_scaled_acc_tile_permuted_32_bscale_view_extra_user_rematerializes or mma_scaled_acc_identity_narrow_format_matrix)'
```

Collection:

```text
195/1615 tests collected (1420 deselected) in 3.31s
```

Clean-boundary runtime selector:

```bash
python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (reports_clean or report_clean or reports_backend_error or unsupported_variants_report_clean_unsupported or software_reduce or noncanonical_block_reports_clean_unsupported or layout_in_4cta_context_reports_clean_error or row_permuted_destination_reports_clean_unsupported)'
```

Collection:

```text
245/1615 tests collected (1370 deselected) in 3.51s
```

Known-red runtime selector:

```bash
python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (ld_red_m64_rowcol_permuted_default_layout or ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn)'
```

Collection:

```text
9/1615 tests collected (1606 deselected) in 3.51s
```

## Duration Files

Positive duration seed:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest -q -s --tb=short \
  --store-durations \
  --durations-path /tmp/tmem_round52_positive_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (ldst_descriptor_rank5_small_roundtrip or ldst_descriptor_rank5_unit_parent_n256_roundtrip or ldst_x1 or ldst_scales_direct_roundtrip or ldst_scales_descriptor_view or cp_no_scales_warpx2_01_23 or cp_no_scales_linear_tile_permuted or cp_128x128 or cp_scales_warpx4_via_scaled_mma_geometry_sweep or mma_plain_kinds or mma_scaled_acc_tile_permuted_32_bscale_view_extra_user_rematerializes or mma_scaled_acc_identity_narrow_format_matrix)'
```

Result:

```text
195 passed, 1420 deselected in 10.14s
real 0m12.735s
```

Stored:

```text
/tmp/tmem_round52_positive_durations.json
entries: 195
json size: 33833 bytes
recorded pytest-duration sum: 8.133s
```

Slowest positive entries:

```text
0.907s test_tmem_runtime_matrix_mma_scaled_acc_identity_narrow_format_matrix[mxfp8-mxfp4-16-256]
0.833s test_tmem_runtime_matrix_ldst_x1_subword_roundtrip[auto-f16-torch_dtype0-linear_packed-2-<lambda>-32x32b.x1.b32-32x32b.x1.b32]
0.658s test_tmem_runtime_matrix_mma_scaled_acc_identity_narrow_format_matrix[mxfp8-mxfp8-16-256]
0.569s test_tmem_runtime_matrix_mma_scaled_acc_identity_narrow_format_matrix[mxfp8-mxfp8-16-128]
0.562s test_tmem_runtime_matrix_mma_scaled_acc_identity_narrow_format_matrix[nvfp4-nvfp4-16-256]
```

Clean-boundary duration seed:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest -q -s --tb=short \
  --store-durations \
  --durations-path /tmp/tmem_round52_clean_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (reports_clean or report_clean or reports_backend_error or unsupported_variants_report_clean_unsupported or software_reduce or noncanonical_block_reports_clean_unsupported or layout_in_4cta_context_reports_clean_error or row_permuted_destination_reports_clean_unsupported)'
```

Result:

```text
245 passed, 1370 deselected in 48.17s
real 0m50.491s
```

Stored:

```text
/tmp/tmem_round52_clean_durations.json
entries: 245
json size: 42377 bytes
recorded pytest-duration sum: 46.165s
```

Slowest clean-boundary entries:

```text
2.664s test_tmem_runtime_matrix_ld_red_mixed_linear_layout_uses_software_reduce[128-256-4-True-propagate_nan1-max]
2.630s test_tmem_runtime_matrix_ld_red_mixed_linear_layout_uses_software_reduce[128-256-4-True-propagate_nan1-min]
2.562s test_tmem_runtime_matrix_ld_red_mixed_linear_layout_uses_software_reduce[128-128-4-True-propagate_nan1-min]
2.353s test_tmem_runtime_matrix_ld_red_mixed_linear_layout_uses_software_reduce[128-128-4-False-propagate_nan0-max]
2.191s test_tmem_runtime_matrix_ld_red_mixed_linear_layout_uses_software_reduce[128-128-4-True-propagate_nan1-max]
```

## Split Results

Positive least-duration split command shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
PYTHONPATH=.:./python \
pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --splitting-algorithm=least_duration \
  --durations-path /tmp/tmem_round52_positive_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (ldst_descriptor_rank5_small_roundtrip or ldst_descriptor_rank5_unit_parent_n256_roundtrip or ldst_x1 or ldst_scales_direct_roundtrip or ldst_scales_descriptor_view or cp_no_scales_warpx2_01_23 or cp_no_scales_linear_tile_permuted or cp_128x128 or cp_scales_warpx4_via_scaled_mma_geometry_sweep or mma_plain_kinds or mma_scaled_acc_tile_permuted_32_bscale_view_extra_user_rematerializes or mma_scaled_acc_identity_narrow_format_matrix)'
```

Positive split results:

```text
group 1: estimated 2.03s, 48 passed, 1567 deselected in 4.94s, real 0m7.109s
group 2: estimated 2.03s, 49 passed, 1566 deselected in 7.85s, real 0m10.173s
group 3: estimated 2.03s, 50 passed, 1565 deselected in 9.26s, real 0m12.658s
group 4: estimated 2.03s, 48 passed, 1567 deselected in 8.12s, real 0m10.423s
aggregate: 195 passed, 0 failed, 0 skipped
```

Clean-boundary count split before duration data:

```text
group 1: estimated 62.00s, 62 passed, 1553 deselected in 14.63s, real 0m16.852s
group 2: estimated 62.00s, 62 passed, 1553 deselected in 32.59s, real 0m35.613s
group 3: estimated 62.00s, 62 passed, 1553 deselected in 4.59s, real 0m6.731s
group 4: estimated 59.00s, 59 passed, 1556 deselected in 10.44s, real 0m12.658s
aggregate: 245 passed, 0 failed, 0 skipped
```

Clean-boundary least-duration split command shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
PYTHONPATH=.:./python \
pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --splitting-algorithm=least_duration \
  --durations-path /tmp/tmem_round52_clean_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (reports_clean or report_clean or reports_backend_error or unsupported_variants_report_clean_unsupported or software_reduce or noncanonical_block_reports_clean_unsupported or layout_in_4cta_context_reports_clean_error or row_permuted_destination_reports_clean_unsupported)'
```

Clean-boundary least-duration split results:

```text
group 1: estimated 11.54s, 62 passed, 1553 deselected in 8.61s, real 0m11.093s
group 2: estimated 11.54s, 62 passed, 1553 deselected in 8.90s, real 0m11.483s
group 3: estimated 11.54s, 61 passed, 1554 deselected in 8.89s, real 0m11.539s
group 4: estimated 11.54s, 60 passed, 1555 deselected in 9.05s, real 0m12.071s
aggregate: 245 passed, 0 failed, 0 skipped
```

Known-red run:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (ld_red_m64_rowcol_permuted_default_layout or ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn)'
```

Result:

```text
6 failed, 3 passed, 1606 deselected in 3.87s
real 0m6.101s
```

Failures:

```text
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]
```

Classification: all six are existing `FZ-20260421-0012` M64 f32
`tcgen05.ld.red` destination-layout planner gaps for non-identity row-basis
layouts. The three collected `col_reverse_n32` rows passed. No new `FZ-*`
bucket is needed.

## Throughput Notes

- Keep positives, clean-boundary diagnostics, and known-red diagnostics as
  separate lanes. Mixing known-red rows into positive shards obscures real
  throughput failures.
- Reuse `/tmp/tmem_round52_positive_durations.json` for this positive selector.
  The positive split is modestly imbalanced in wall time despite balanced
  duration estimates, probably because collection, compilation cache state, and
  per-GPU startup overhead dominate these short groups.
- Reuse `/tmp/tmem_round52_clean_durations.json` for clean diagnostics. The
  clean lane improved from a worst group of `35.613s` wall with count splitting
  to `12.071s` wall with least-duration splitting.
- The clean lane's slow rows are dominated by
  `ld_red_mixed_linear_layout_uses_software_reduce`; these should remain in the
  clean duration inventory because they are expected-pass diagnostics but have
  much higher runtime than most unsupported-case checks.
- Suggested future command default for broad runtime-matrix fuzzing:
  `--splitting-algorithm=least_duration --durations-path <lane-specific-json>`.

## Classification

No backend fixes were attempted.

No new compiler crash, unexpected unsupported case, verifier drift, clean
diagnostic regression, hang, or miscompile was found in this round.
