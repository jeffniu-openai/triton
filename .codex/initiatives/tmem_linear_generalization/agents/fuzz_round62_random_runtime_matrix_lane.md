# Round 62 TMEM fuzz lane A: randomized runtime-matrix exact-nodeid sampler

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test
source files were edited.

## Build

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Collection and Selection

Full runtime-matrix collection:

```bash
PYTHONPATH=python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `1615 tests collected`.

Sampling seed: `6042162`.

Selection method:

- Parsed all collected `python/test/gluon/test_tmem_runtime_matrix.py::...`
  nodeids from the full collection.
- Built a Round 60-61 exact-avoid set from:
  `agents/fuzz_round60_random_runtime_matrix_lane.md`,
  `agents/fuzz_round61_runtime_adversarial_lane.md`, and fresh collection of
  the two Round 61 runtime selectors.
- Exact avoid set intersecting current collection: `200` nodeids.
- Eligible after exact avoid: `1415` nodeids.
- Sampled deterministic family quotas, then wrote 20 nodeids per shard:
  `ldst=12`, `cp_no_scales=12`, `cp_scales=10`, `ld_red=12`,
  `plain_mmav5=12`, `scaled_mmav5=12`, `clean_diagnostics=6`,
  `twocta_high_cga=4`.
- Selected exact nodeids: `80`.
- Exact overlap with Round 60-61 avoid set: `0`.

Feature coverage in the selected set, counting overlapping features:

- `ldst`: `18`
- `cp_no_scales`: `15`
- `cp_scales`: `10`
- `ld_red`: `12`
- `plain_mmav5`: `13`
- `scaled_mmav5`: `12`
- clean diagnostics/resource/unsupported/error rows: `6`
- 2CTA/high-CGA-like rows: `21`

Exact-nodeid collection check:

```bash
PYTHONPATH=python xargs -a /tmp/round62_selected_nodeids.txt pytest --collect-only -q
```

Result: `80 tests collected`.

## Runtime

Runtime commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=python xargs -a /tmp/round62_selected_group1.txt pytest -s --tb=short
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=python xargs -a /tmp/round62_selected_group2.txt pytest -s --tb=short
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=python xargs -a /tmp/round62_selected_group3.txt pytest -s --tb=short
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=python xargs -a /tmp/round62_selected_group4.txt pytest -s --tb=short
```

Shard results:

- GPU 0 / group 1: `18 passed, 2 skipped`
- GPU 1 / group 2: `18 passed, 2 skipped`
- GPU 2 / group 3: `20 passed`
- GPU 3 / group 4: `19 passed, 1 skipped`

Aggregate split result: `75 passed, 5 skipped, 0 failed`.

Skip-reason rerun:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=python xargs -a /tmp/round62_selected_nodeids.txt pytest -q -rs --tb=short
```

Result: `75 passed, 5 skipped`.

Skip reasons:

- `4` lifted descriptor roundtrip matrices exceed the current Blackwell TMEM
  allocation limit; this is the existing pre-skip boundary for known
  non-executable compiles.
- `1` no-scale copy row skipped because shared memory is out of resources for
  `M=256, N=256, BLOCK_N=128, swizzle=32`.

Failure signatures: none. No compiler crash, verifier drift, false unsupported
diagnostic, clean-boundary drift, opcode assertion failure, runtime miscompile,
hang, or process/device contamination was observed.

## Exact Nodeids

```text
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst[identity-128-auto-32x32b.x128.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_roundtrip_sweeps[slice_index_roundtrip-0-5.0-required_ops0-identity-128-16x128b-16x128b.x32.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_scales_direct_roundtrip[256-64-4-2-cga_layout3-expected_ops3]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_scales_variant_sweep[64-4-4-auto-expected_ops1]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_subword_descriptor_chain_roundtrip[i8-torch_dtype5-identity-128-16x256b-16x256b.x4.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_compositions[block_two_ctas-128-auto-32x32b.x128.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_multidim_slices[f32-torch_dtype3-block_two_ctas-128-16x64b]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_roundtrip_sweeps[slice_index_deep_roundtrip-2-11.0-required_ops2-block_two_ctas-128-16x64b-16x64b.x64.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_roundtrip_sweeps[slice_index_multidim-1-7.0-required_ops1-block_two_ctas-128-16x256b-16x256b.x16.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_roundtrip_sweeps[slice_index_multidim-1-7.0-required_ops1-block_two_ctas-256-32x32b-32x32b.x64.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_i32_descriptor_chain_roundtrip[32x32b-linear_onecta-128-1-<lambda>]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_subword_twocta_descriptor_chain_roundtrip[32x32b-f16-torch_dtype0-2]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales[256-256-128-32-f32-torch_dtype0]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales[256-64-64-32-f32-torch_dtype0]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_4x256b_refresh_layout_codegen
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear[256-32-128-8]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear[256-64-64-16]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_32bit_dtypes[f32-torch_dtype8-256-128-32-32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_subslice_view[f32-torch_dtype1-128-128-64-16-tcgen05.cp.cta_group::1.128x256b]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_swizzles[128-256-256-32-f32-torch_dtype0]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_swizzles[256-128-64-32-f32-torch_dtype0]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_codegen[legacy-i32-torch_dtype12-16-32-2]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_codegen[linear-i8-torch_dtype19-256-32-8]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_slice_index_view_positive[01_23-_make_tmem_copy_warpx2_tmem_layout-_expected_tmem_copy_warpx2_01_23_output-tcgen05.cp.cta_group::1.warpx2::01_23.64x128b-0-f32-torch_dtype0]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_layout_probe[warpx2_row32_after_low_rows-smem_layout4-PASS]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_layout_probe[warpx4-smem_layout0-PASS]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_shared_subslice_layout_rematerializes[0]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_shared_subslice_layout_rematerializes[64]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix[mxfp8-mxfp8-256-128-2-False-linear]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix[mxfp8-mxfp8-256-256-2-True-legacy]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix[nvfp4-nvfp4-256-256-2-True-linear]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix_use_acc[mxfp8-mxfp4-128-128-1-False-linear]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix_use_acc[mxfp8-mxfp8-128-128-1-False-linear]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_geometry_sweep[256-256-False-1-legacy]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_col_permuted_linear_layout[reverse-64-32x32b.x64-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_explicit_compatible_layout_variants[32x32b-True-propagate_nan1-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_explicit_n_sharded_layout_uses_software_reduce[16x256b-False-propagate_nan0-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_explicit_n_sweep_variants[row_reverse_n64_32x32b-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_identity_linear_layout[identity-128-32-4-32x32b.x32-True-propagate_nan1-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_splitn_linear_layout[m64_64x32-True-propagate_nan1-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_splitn_linear_layout[m64_64x64-True-propagate_nan1-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_mixed_linear_layout_uses_software_reduce[128-64-4-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_non_f32_contract_uses_software_reduce[f16_abs-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_non_f32_contract_uses_software_reduce[f16_plain-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_pure_row_permuted_n_sweep[even_odd-64-32x32b.x64-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_row_permuted_linear_layout[even_odd-identity-32x32b.x128-False-propagate_nan0-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_acc_subslice_view_plain_kinds[f8e5m2-128-64-0-False]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_indexed_acc_view[f16-linear-64-64-False]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds[f16-linear-256-128]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_lhs_tile_permuted_use_acc[f16-128-64-16]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_lhs_tile_permuted_use_acc[f8e4m3-128-64-16]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_lhs_tile_permuted_use_acc[tf32-128-64-16]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_plain_kinds_with_linear_acc[f16-linear-128-64]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta[f16-legacy-32-32-True]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta[f16-legacy-64-64-True]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta[f16-linear-128-128-True]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_plain_kinds[bf16-linear-128-64]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_tma_tf32_b_transposed_descriptor_use_acc[legacy-64-128]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_identity_narrow_format_matrix[mxfp4-mxfp4-16-128]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_identity_narrow_format_matrix[mxfp4-mxfp8-16-256]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_subslice_tile_permuted_format_matrix[256-mxfp4-mxfp4]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_subslice_tile_permuted_format_matrix[256-nvfp4-nvfp4]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_subslice_view_format_matrix[128-0-128-mxfp8-mxfp4]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_subslice_view_format_use_acc[mxfp8-mxfp8-64-0-128]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_format_matrix[mxfp8-mxfp8-256]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_narrow_format_matrix[mxfp4-mxfp8-64-16-128]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view_format_matrix[nvfp4-nvfp4-128-128-legacy]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view_format_matrix[nvfp4-nvfp4-128-128-linear]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_root_format_use_acc[mxfp4-mxfp8-128-128-linear]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_root_format_use_acc[mxfp8-mxfp4-128-128-linear]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_indexed_view_full_128x256_reports_tmem_oor
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_subword_dtypes_report_clean_error[single_01_23-1-tmem_layout3-i8-torch_dtype3]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_tmem_oor[identity-256-32x32b]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_dim0_slice_reports_tmem_oor[block_two_ctas-auto]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_half_rows_reports_tmem_oor[block_two_ctas-256-16x128b]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_tma_tf32_reports_clean_shared_transpose_error[linear-128-64]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_codegen[linear-i16-torch_dtype18-256-128-16]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_direct_half_rows_positive[mmav5_twocta-64-16x128b-16x128b.x16.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_half_rows_positive_lifted_layout[block_two_ctas-128-auto-32x32b.x128.b32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_index[f32-torch_dtype0-block_two_ctas-64-auto-32x32b.x64.b32-32x32b.x32.b32]
```

## FZ Classification

No new independent `FZ-*` bucket was found.

- Existing lifted descriptor roundtrip allocation-limit skips were reproduced
  only as expected skips, not failures.
- Existing shared-memory OOR no-scale copy behavior was reproduced only as the
  expected skip.
- Clean diagnostic rows stayed clean; no false unsupported diagnostic or
  assertion-text drift was observed.
- Existing TMEM crash/miscompile buckets such as `FZ-20260421-0001`,
  `FZ-20260421-0003`, `FZ-20260421-0007`, `FZ-20260421-0012`,
  `FZ-20260421-0013`, `FZ-20260421-0015`, `FZ-20260421-0016`,
  `FZ-20260421-0017`, `FZ-20260421-0018`, `FZ-20260421-0020`,
  `FZ-20260421-0022`, and `FZ-20260421-0023` were not reproduced as failures.

Backend/compiler repairs remain deferred.
