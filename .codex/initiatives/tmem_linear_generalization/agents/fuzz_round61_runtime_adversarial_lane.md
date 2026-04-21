# Round 61 TMEM fuzz lane A: runtime-matrix adversarial selector

Date: 2026-04-21 16:24 UTC
Branch: `codex/tmem`
HEAD before report commit: `c8db65cd9b26`
Scope: discovery/cataloging only. No backend, compiler, or test source files
were edited.

## Build

Required rebuild:

```bash
make -j8
```

Result: no-op, `ninja: no work to do`.

An initial collection without `PYTHONPATH=python` imported the wrong/no local
Triton package and failed with:

```text
ModuleNotFoundError: No module named 'triton.compiler.errors'
```

All collection and runtime commands below used `PYTHONPATH=python` so pytest
imports this checkout.

## Selector A: descriptor/ld.red/MMAv5/scales adversarial runtime matrix

Purpose: non-overlapping pressure after Rounds 58-60 on descriptor chains and
indexed views, two-CTA accumulator views, narrow N boundaries, scale descriptor
views, `ld.red`, and accumulator view/use-acc combinations. This intentionally
avoids broad Round 58 plain/scaled accumulator tile sweeps and Round 60 random
sampling, while reusing checked-in runtime rows with stronger adversarial
selection.

Collection command:

```bash
PYTHONPATH=python pytest --collect-only -q -k '((copy_no_scales and (twocta_linear_indexed_view or twocta_linear_subslice_view or warpx2_twocta_indexed_view or warpx2_twocta_slice_index_view or warpx2_twocta_subslice_view)) or (ld_red and descriptor_chain and not non_f32) or (mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds) or (mma_scaled_twocta_acc_subslice_view_format_use_acc or mma_scaled_twocta_acc_subslice_view_format_matrix or mma_scaled_shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc or mma_scaled_acc_n16_bscale_descriptor_view_reports_clean_error))' python/test/gluon/test_tmem_runtime_matrix.py
```

Collection result: `108/1615` collected.

Realized exact nodeid families:

- `test_tmem_runtime_matrix_ld_red_descriptor_chain[...]`: 10 nodeids covering
  `identity`, `tile_permuted`, `col_reverse`, `row_reverse`, and
  `rowcol_rotate_reverse` with `auto` and `32x32b`, all
  `False-propagate_nan0-min`.
- `test_tmem_runtime_matrix_ld_red_descriptor_chain_n_sweep[...]`: 12 nodeids
  covering `identity_n32`, `identity_n64`, `identity_n256`,
  `tile_permuted_n32`, `tile_permuted_n256`, `col_reverse_n64`,
  `col_reverse_n256`, `row_reverse_n64`, `row_reverse_n256`,
  `row_even_odd_n64`, `rowcol_rotate_reverse_n64`, and
  `rowcol_rotate_reverse_n256`, all `False-propagate_nan0-min`.
- `test_tmem_runtime_matrix_ld_red_descriptor_chain_n_sweep_explicit_variants[...]`:
  8 nodeids covering `identity_n32_32x32b`, `identity_n64_32x32b`,
  `identity_n256_32x32b_splitn`, `tile_permuted_n32_32x32b`,
  `col_reverse_n64_32x32b`, `row_reverse_n64_32x32b`,
  `row_even_odd_n64_32x32b`, and `rowcol_rotate_reverse_n64_32x32b`, all
  `False-propagate_nan0-min`.
- `test_tmem_runtime_matrix_mma_twocta_indexed_acc_view[...]`: 23 nodeids
  covering f16/tf32/bf16/f8 2CTA indexed accumulators, legacy and linear
  parents, `linear_unit_parent`, N32/N64/N128/N256, K32/K64/K128, and both
  `use_acc` states where checked in.
- `test_tmem_runtime_matrix_mma_twocta_acc_subslice_view_plain_kinds[...]`:
  21 nodeids covering f16/tf32/bf16/f8 2CTA accumulator subslices, N32/N64/
  N128/N256, slice starts 0/N, K32/K64/K128, and both `use_acc` states where
  checked in.
- `test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_matrix[...]`:
  14 nodeids covering `mxfp8`, `mxfp4`, mixed `mxfp8/mxfp4`,
  mixed `mxfp4/mxfp8`, `nvfp4`, N32/N64/N128, K128/K256, slice starts 0/N,
  and multicast false/true variants.
- `test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_use_acc[...]`:
  14 nodeids matching the above scaled two-CTA accumulator subslice matrix with
  `use_acc`.
- `test_tmem_runtime_matrix_mma_scaled_shared_scale_descriptor_view_auto_tmem_copy[...]`:
  2 nodeids: `[1-128-128-256-0-False]` and `[2-256-64-128-0-False]`.
- `test_tmem_runtime_matrix_mma_scaled_acc_n16_bscale_descriptor_view_reports_clean_error[...]`:
  2 nodeids: `[False]` and `[True]`.
- `test_tmem_runtime_matrix_mma_scaled_indexed_acc_identity_narrow_view_format_use_acc[...]`:
  2 nodeids: `[mxfp8-mxfp8-16-128-linear]` and
  `[mxfp8-mxfp8-16-128-linear_unit_parent]`.

4-GPU runtime commands:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short --splits 4 --group 1 -k '((copy_no_scales and (twocta_linear_indexed_view or twocta_linear_subslice_view or warpx2_twocta_indexed_view or warpx2_twocta_slice_index_view or warpx2_twocta_subslice_view)) or (ld_red and descriptor_chain and not non_f32) or (mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds) or (mma_scaled_twocta_acc_subslice_view_format_use_acc or mma_scaled_twocta_acc_subslice_view_format_matrix or mma_scaled_shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc or mma_scaled_acc_n16_bscale_descriptor_view_reports_clean_error))' python/test/gluon/test_tmem_runtime_matrix.py
PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 pytest -s --tb=short --splits 4 --group 2 -k '((copy_no_scales and (twocta_linear_indexed_view or twocta_linear_subslice_view or warpx2_twocta_indexed_view or warpx2_twocta_slice_index_view or warpx2_twocta_subslice_view)) or (ld_red and descriptor_chain and not non_f32) or (mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds) or (mma_scaled_twocta_acc_subslice_view_format_use_acc or mma_scaled_twocta_acc_subslice_view_format_matrix or mma_scaled_shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc or mma_scaled_acc_n16_bscale_descriptor_view_reports_clean_error))' python/test/gluon/test_tmem_runtime_matrix.py
PYTHONPATH=python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 pytest -s --tb=short --splits 4 --group 3 -k '((copy_no_scales and (twocta_linear_indexed_view or twocta_linear_subslice_view or warpx2_twocta_indexed_view or warpx2_twocta_slice_index_view or warpx2_twocta_subslice_view)) or (ld_red and descriptor_chain and not non_f32) or (mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds) or (mma_scaled_twocta_acc_subslice_view_format_use_acc or mma_scaled_twocta_acc_subslice_view_format_matrix or mma_scaled_shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc or mma_scaled_acc_n16_bscale_descriptor_view_reports_clean_error))' python/test/gluon/test_tmem_runtime_matrix.py
PYTHONPATH=python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 pytest -s --tb=short --splits 4 --group 4 -k '((copy_no_scales and (twocta_linear_indexed_view or twocta_linear_subslice_view or warpx2_twocta_indexed_view or warpx2_twocta_slice_index_view or warpx2_twocta_subslice_view)) or (ld_red and descriptor_chain and not non_f32) or (mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds) or (mma_scaled_twocta_acc_subslice_view_format_use_acc or mma_scaled_twocta_acc_subslice_view_format_matrix or mma_scaled_shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc or mma_scaled_acc_n16_bscale_descriptor_view_reports_clean_error))' python/test/gluon/test_tmem_runtime_matrix.py
```

Runtime result:

- Group 1 / GPU 0: `27 passed, 1588 deselected`.
- Group 2 / GPU 1: `27 passed, 1588 deselected`.
- Group 3 / GPU 2: `27 passed, 1588 deselected`.
- Group 4 / GPU 3: `27 passed, 1588 deselected`.

Aggregate selector A result: `108 passed, 0 failed, 0 skipped`.

## Selector B: copy/indexed/subslice descriptor-view add-on

Purpose: the `copy_no_scales` terms in selector A did not collect because the
checked-in tests use `cp_no_scales_*` names. This add-on explicitly covers the
requested copy descriptor-view pressure with two-CTA indexed/subslice views and
warpx2 indexed/slice-index positives plus clean unsupported `02_13` rows.

Collection command:

```bash
PYTHONPATH=python pytest --collect-only -q -k 'cp_no_scales and (twocta_linear_indexed_view or twocta_linear_subslice_view or warpx2_01_23_twocta_indexed_view_positive or warpx2_01_23_twocta_slice_index_view_positive or warpx2_02_13_twocta_indexed_view_reports_clean_unsupported or warpx2_02_13_twocta_slice_index_view_reports_clean_unsupported)' python/test/gluon/test_tmem_runtime_matrix.py
```

Collection result: `33/1615` collected.

Exact nodeids:

- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_indexed_view[f32-torch_dtype0-256-64-32-8-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_indexed_view[f32-torch_dtype1-256-128-128-16-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_indexed_view[i32-torch_dtype2-256-64-32-8-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_indexed_view[i32-torch_dtype3-256-128-128-16-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_indexed_view[f16-torch_dtype4-256-128-32-8-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_indexed_view[bf16-torch_dtype5-256-128-64-8-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_indexed_view[i16-torch_dtype6-256-128-128-8-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_indexed_view[i8-torch_dtype7-256-128-32-4-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_indexed_view_full_256x256_reports_tmem_oor`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_subslice_view[f32-torch_dtype0-256-128-32-16-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_subslice_view[f32-torch_dtype1-256-256-128-32-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_subslice_view[i32-torch_dtype2-256-128-32-16-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_subslice_view[i32-torch_dtype3-256-256-128-32-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_subslice_view[f16-torch_dtype4-256-128-32-8-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_subslice_view[bf16-torch_dtype5-256-256-64-16-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_subslice_view[i16-torch_dtype6-256-256-128-16-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_subslice_view[i8-torch_dtype7-256-256-32-8-tcgen05.cp.cta_group::2.128x256b]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_indexed_view_positive[0-f32-torch_dtype0]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_indexed_view_positive[0-i32-torch_dtype1]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_indexed_view_positive[1-f32-torch_dtype0]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_indexed_view_positive[1-i32-torch_dtype1]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_slice_index_view_positive[0-f32-torch_dtype0]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_slice_index_view_positive[0-i32-torch_dtype1]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_slice_index_view_positive[1-f32-torch_dtype0]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_slice_index_view_positive[1-i32-torch_dtype1]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_slice_index_view_reports_clean_unsupported[0-f32-torch_dtype0]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_slice_index_view_reports_clean_unsupported[0-i32-torch_dtype1]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_slice_index_view_reports_clean_unsupported[1-f32-torch_dtype0]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_slice_index_view_reports_clean_unsupported[1-i32-torch_dtype1]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_indexed_view_reports_clean_unsupported[0-f32-torch_dtype0]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_indexed_view_reports_clean_unsupported[0-i32-torch_dtype1]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_indexed_view_reports_clean_unsupported[1-f32-torch_dtype0]`
- `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_indexed_view_reports_clean_unsupported[1-i32-torch_dtype1]`

4-GPU runtime commands:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short --splits 4 --group 1 -k 'cp_no_scales and (twocta_linear_indexed_view or twocta_linear_subslice_view or warpx2_01_23_twocta_indexed_view_positive or warpx2_01_23_twocta_slice_index_view_positive or warpx2_02_13_twocta_indexed_view_reports_clean_unsupported or warpx2_02_13_twocta_slice_index_view_reports_clean_unsupported)' python/test/gluon/test_tmem_runtime_matrix.py
PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 pytest -s --tb=short --splits 4 --group 2 -k 'cp_no_scales and (twocta_linear_indexed_view or twocta_linear_subslice_view or warpx2_01_23_twocta_indexed_view_positive or warpx2_01_23_twocta_slice_index_view_positive or warpx2_02_13_twocta_indexed_view_reports_clean_unsupported or warpx2_02_13_twocta_slice_index_view_reports_clean_unsupported)' python/test/gluon/test_tmem_runtime_matrix.py
PYTHONPATH=python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 pytest -s --tb=short --splits 4 --group 3 -k 'cp_no_scales and (twocta_linear_indexed_view or twocta_linear_subslice_view or warpx2_01_23_twocta_indexed_view_positive or warpx2_01_23_twocta_slice_index_view_positive or warpx2_02_13_twocta_indexed_view_reports_clean_unsupported or warpx2_02_13_twocta_slice_index_view_reports_clean_unsupported)' python/test/gluon/test_tmem_runtime_matrix.py
PYTHONPATH=python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 pytest -s --tb=short --splits 4 --group 4 -k 'cp_no_scales and (twocta_linear_indexed_view or twocta_linear_subslice_view or warpx2_01_23_twocta_indexed_view_positive or warpx2_01_23_twocta_slice_index_view_positive or warpx2_02_13_twocta_indexed_view_reports_clean_unsupported or warpx2_02_13_twocta_slice_index_view_reports_clean_unsupported)' python/test/gluon/test_tmem_runtime_matrix.py
```

Runtime result:

- Group 1 / GPU 0: `9 passed, 1606 deselected`.
- Group 2 / GPU 1: `9 passed, 1606 deselected`.
- Group 3 / GPU 2: `9 passed, 1606 deselected`.
- Group 4 / GPU 3: `6 passed, 1609 deselected`.

Aggregate selector B result: `33 passed, 0 failed, 0 skipped`.

## Aggregate lane result

- Selector A: `108 passed, 0 failed, 0 skipped`.
- Selector B: `33 passed, 0 failed, 0 skipped`.
- Total runtime executions: `141 passed, 0 failed, 0 skipped`.

Failure signatures: none. No compiler crash, verifier drift, false unsupported
diagnostic, clean-boundary drift, opcode assertion failure, runtime miscompile,
hang, or process/device contamination was observed.

## FZ classification

- Existing `FZ-20260421-0003`: not reproduced. Descriptor/indexed view rows in
  this lane passed or stayed as clean unsupported boundaries.
- Existing `FZ-20260421-0012`: not reproduced. No M64 f32 `ld.red`
  unsupported-destination failure was selected.
- Existing `FZ-20260421-0018`: not reproduced. This lane stayed within
  checked-in `ld.red` descriptor-chain positive/runtime rows and did not hit
  the larger ptxas boundary probe.
- Existing `FZ-20260421-0020` / `FZ-20260421-0022`: not reproduced. No
  half-column or direct higher-rank `ld.red` abort rows were selected.
- Clean unsupported 2CTA `warpx2::02_13` copy descriptor-view rows behaved as
  expected when run; no false unsupported diagnostic or assertion drift.

No new independent issue was found; no candidate `FZ-20260421-00XX` assigned.

## Follow-up

If this exact lane is repeated, the selector-B group-4 command is:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 pytest -s --tb=short --splits 4 --group 4 -k 'cp_no_scales and (twocta_linear_indexed_view or twocta_linear_subslice_view or warpx2_01_23_twocta_indexed_view_positive or warpx2_01_23_twocta_slice_index_view_positive or warpx2_02_13_twocta_indexed_view_reports_clean_unsupported or warpx2_02_13_twocta_slice_index_view_reports_clean_unsupported)' python/test/gluon/test_tmem_runtime_matrix.py
```
