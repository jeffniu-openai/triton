# Round 60 Lane C: randomized runtime-matrix subset selection

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code
was edited.

## Scope

This lane sampled a bounded, seed-based subset from
`python/test/gluon/test_tmem_runtime_matrix.py` across TMEM runtime families.
The sampler excluded recent Round 58-59-heavy terms where feasible, including
scaled accumulator tile/narrow rows, plain accumulator indexed/subslice/tile
rows, higher-rank/rank5 descriptor rows, clean-diagnostic rows, TMA/proxy/
mbarrier rows, and scale-copy rows.

Sampling seed: `6042160`.

## Required build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Collection

Full runtime-matrix collection:

```bash
PYTHONPATH=./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `1615 tests collected`.

Seeded candidate generation:

- all nodeids: `1615`
- eligible after recent-heavy exclusions: `1116`
- selected exact nodeids: `64`

Selected family counts:

- `ld/st`: `20`
- no-scale copy: `12`
- `ld.red`: `14`
- plain MMAv5: `10`
- scaled MMAv5: `8`

Exact selected-nodeid collection:

```bash
env PYTHONPATH=./python xargs -a /tmp/round60_selected_nodeids.txt pytest --collect-only -q
```

Result: `64 tests collected`.

## Runtime split

```bash
env CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python xargs -a /tmp/round60_selected_group1.txt pytest -s --tb=short
env CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=./python xargs -a /tmp/round60_selected_group2.txt pytest -s --tb=short
env CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=./python xargs -a /tmp/round60_selected_group3.txt pytest -s --tb=short
env CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=./python xargs -a /tmp/round60_selected_group4.txt pytest -s --tb=short
```

Results:

- GPU 0 / group 1: `14 passed, 2 skipped`
- GPU 1 / group 2: `14 passed, 2 skipped`
- GPU 2 / group 3: `14 passed, 2 skipped`
- GPU 3 / group 4: `14 passed, 2 skipped`
- aggregate: `56 passed, 8 skipped`

Skip-reason check:

```bash
env CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python xargs -a /tmp/round60_selected_nodeids.txt pytest -q -rs --tb=short
```

Result: `56 passed, 8 skipped`. All skips were the existing lifted descriptor
roundtrip pre-skip for matrices exceeding the current Blackwell TMEM
allocation limit.

## Exact Nodeids

```text
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_permuted_layout_sweep[f32-torch_dtype2-even_odd-128-auto-32x32b.x128.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions_exotic_layouts[f32-torch_dtype5-scrambled_cols-256-16x128b-16x128b.x32.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_exotic_n32_linear_layout[f32-torch_dtype1-descriptor-scrambled_cols-auto-32x32b.x32.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_rowcol_permuted_layout_sweep[f32-torch_dtype3-identity-reverse-64-32x32b-32x32b.x64.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions_rowcol_permuted_layout_sweep[i32-torch_dtype5-rotate1-reverse-128-32x32b-32x32b.x128.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_i32_broad_linear_layouts[direct-block_two_ctas-128-auto-32x32b.x128.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_exotic_n32_linear_layout[f32-torch_dtype0-direct-block_single_cta-auto-32x32b.x32.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions_rowcol_permuted_layout_sweep[f32-torch_dtype0-identity-identity-128-auto-32x32b.x128.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_permuted_n32_linear_layout[f32-torch_dtype1-descriptor-reverse-auto-32x32b.x32.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions_permuted_layout_sweep[f32-torch_dtype2-even_odd-128-auto-32x32b.x128.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions_permuted_layout_sweep[f32-torch_dtype0-identity-128-auto-32x32b.x128.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_i32_broad_linear_layouts[descriptor-block_two_ctas-128-16x256b-16x256b.x16.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_roundtrip_sweeps[slice_index_multidim-1-7.0-required_ops1-mmav5_twocta-256-16x128b-16x128b.x32.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_roundtrip_rowcol_permuted_sweeps[even_odd-reverse-256-16x128b-16x128b.x32.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_roundtrip_sweeps[slice_index_deep_roundtrip-2-11.0-required_ops2-mmav5_twocta-128-auto-32x32b.x128.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_roundtrip_sweeps[slice_index_roundtrip-0-5.0-required_ops0-mixed-256-16x128b-16x128b.x32.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_roundtrip_rowcol_permuted_sweeps[reverse-identity-256-16x128b-16x128b.x32.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_roundtrip_sweeps[slice_index_deep_roundtrip-2-11.0-required_ops2-mixed-128-auto-32x32b.x128.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_roundtrip_sweeps[slice_index_multidim-1-7.0-required_ops1-mixed-256-16x128b-16x128b.x32.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_roundtrip_sweeps[slice_index_roundtrip-0-5.0-required_ops0-mmav5_twocta-64-16x128b-16x128b.x16.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive[i32-torch_dtype1]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_slice_index_view_positive[02_13-_make_tmem_copy_warpx2_tmem_layout_02_13-_expected_tmem_copy_warpx2_02_13_output-tcgen05.cp.cta_group::1.warpx2::02_13.64x128b-0-i32-torch_dtype1]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_indexed_view[128x128b_i32-torch_dtype1-128-4-smem_layout1-1-tcgen05.cp.cta_group::1.128x128b]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_128x128b_codegen[linear-f32-torch_dtype0]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_subslice_view_positive[4-i32-torch_dtype1]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_slice_index_view_positive[02_13-_make_tmem_copy_warpx2_tmem_layout_02_13-_expected_tmem_copy_warpx2_02_13_output-tcgen05.cp.cta_group::1.warpx2::02_13.64x128b-0-f32-torch_dtype0]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_subword_dtypes[f16-torch_dtype0-128-128-32-8]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_swizzles[256-128-64-32-i32-torch_dtype1]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_tile_selector_permuted[256-8-reverse-32-tcgen05.cp.cta_group::1.128x256b]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_subslice_view_positive[01_23-_make_tmem_copy_warpx2_parent_tmem_layout-_expected_tmem_copy_warpx2_01_23_output-tcgen05.cp.cta_group::1.warpx2::01_23.64x128b-4-i32-torch_dtype1]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_dense_shared_rematerializes[warpx2::02_13.64x128b-tmem_layout1-_expected_tmem_copy_warpx2_02_13_output-tcgen05.cp.cta_group::1.warpx2::02_13.64x128b-i32-torch_dtype1]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_tile_selector_permuted[128-8-reverse-16-tcgen05.cp.cta_group::1.128x256b]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_rowcol_permuted_n_sweep[reverse-reverse-32-32x32b.x32-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_descriptor_chain_n_sweep_explicit_variants[col_reverse_n64_32x32b-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_explicit_n_sharded_layout_uses_software_reduce[16x64b-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_mixed_linear_layout_uses_software_reduce[128-64-4-False-propagate_nan0-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_mixed_linear_layout_uses_software_reduce[128-256-4-True-propagate_nan1-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_explicit_n_sweep_variants[tile_permuted_n32_32x32b-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_expanded_row_permuted_linear_layout[reverse-32-32x32b.x32-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_descriptor_chain_n_sweep[rowcol_rotate_reverse_n64-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_tile_permuted_linear_layout[128-8-4-32x32b.x128-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_non_f32_descriptor_chain_uses_software_reduce[f16_plain_descriptor-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_pure_row_permuted_n_sweep[reverse-32-32x32b.x32-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_descriptor_chain[identity_32x32b-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_descriptor_chain[tile_permuted_32x32b-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_col_permuted_linear_layout[reverse-32-32x32b.x32-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_plain_kinds_use_acc[tf32-linear-128-64]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_m64_acc_subslice_view_plain_kinds[f8e4m3-128-64-0-False]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_m64_acc_subslice_view_plain_kinds[f8e5m2-128-64-0-False]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds_use_acc[f16-linear-64-64]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds[f16-linear-32-32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds_use_acc[f16-legacy-64-64]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_m64_acc_subslice_view_plain_kinds[f16-64-64-64-True]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_plain_kinds_m64[bf16-linear-128-64-False]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds_use_acc[f16-legacy-256-128]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_plain_kinds[f16-linear-64-64]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_indexed_acc_view_format_use_acc[mxfp8-mxfp8-128-256-linear_unit_parent]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_root_format_matrix[mxfp8-mxfp8-128-128-linear]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_indexed_acc_view_format_matrix[mxfp4-mxfp4-64-128-linear]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_indexed_acc_view_format_matrix[mxfp8-mxfp8-256-256-legacy_unit_parent]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_blockm64_reports_backend_error
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_matrix[mxfp8-mxfp8-128-0-128-True]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_root_format_matrix[mxfp4-mxfp8-128-128-linear]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_root_format_matrix[mxfp4-mxfp4-128-128-linear]
```

## Classification

No new independent `FZ-*` bucket was found. The selected `ld/st`, copy,
`ld.red`, plain MMAv5, and scaled-MMAv5 rows produced no compiler crash,
verifier drift, false unsupported diagnostic, runtime miscompile, hang, or
unexpected pass/fail transition. The only non-passing outcomes were the known
lifted descriptor roundtrip allocation-limit skips.

Existing buckets referenced for classification:

- lifted descriptor roundtrip allocation-limit skips: known resource boundary,
  no new `FZ-*`;
- no failures reproduced existing crash/miscompile buckets such as
  `FZ-20260421-0001`, `FZ-20260421-0003`, `FZ-20260421-0007`,
  `FZ-20260421-0012`, `FZ-20260421-0013`, `FZ-20260421-0015`,
  `FZ-20260421-0016`, `FZ-20260421-0017`, or `FZ-20260421-0023`.

Backend/compiler repairs remain deferred.
