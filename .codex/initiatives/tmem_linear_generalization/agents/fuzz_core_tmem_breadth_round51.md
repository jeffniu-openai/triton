# Round 51 Core TMEM Breadth Fuzzing

Date: 2026-04-21

Scope: broad-but-practical `python/test/gluon/test_core.py` runtime control sweep for TMEM-bearing paths. This round intentionally stayed catalog-only: no backend fixes, no checked-in tests changed, and no source files were edited.

## Preconditions

- Branch: `codex/tmem`
- Rebuild before testing:

```bash
make -j8
```

Result: no-op rebuild succeeded.

Stable per-GPU caches were used:

- GPU 0: `TRITON_CACHE_DIR=/tmp/triton-cache-gpu0`
- GPU 1: `TRITON_CACHE_DIR=/tmp/triton-cache-gpu1`
- GPU 2: `TRITON_CACHE_DIR=/tmp/triton-cache-gpu2`
- GPU 3: `TRITON_CACHE_DIR=/tmp/triton-cache-gpu3`

## Selector Construction

The exact node list was collected from `test_core.py` with:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_core.py > /tmp/test_core_collect_round51.txt
rg "test_(tmem|tcgen05|tma|mma_scaled).*|test_.*(tmem|tcgen05|tma|mma_scaled|mbarrier).*" \
  /tmp/test_core_collect_round51.txt > /tmp/test_core_tmem_nodes_round51.txt
PYTHONPATH=.:./python xargs -d '\n' pytest --collect-only -q \
  < /tmp/test_core_tmem_nodes_round51.txt > /tmp/test_core_tmem_exact_collect_round51.txt
```

Collection result: `641 tests collected`.

Note: an initial broad `-k 'tmem or tcgen05 or tma or mma_scaled or mbarrier'` attempt was terminated because pytest keyword matching also selected adjacent non-TMEM `test_mma_shared_inputs[...]` cases. The final sweep used only the explicit 641-node list above.

## Exact Selector Coverage

The 641 executed node IDs came from these exact `test_core.py` test functions:

```text
216 test_tma_mma_shared_inputs
 96 test_mma_scaled_tcgen05_copy
 60 test_tmem_linear_roundtrip_variant_sweep
 40 test_tmem_reduction
 32 test_tmem_reduction_linear_layouts
 30 test_tmem_copy_no_scales_matrix
 26 test_tmem_descriptor_chain_matrix
 18 test_tmem_copy_no_scales
 18 test_tmem_linear_m64_roundtrip_direct_shapes
 14 test_tmem_linear_roundtrip_splitn_shapes
 10 test_tmem_linear_runtime_views
  8 test_mma_scaled_tcgen05_copy_linear_acc
  8 test_tmem_reduction_linear_legacy_block_equiv_layout
  7 test_tmem_linear_roundtrip_atom_shapes
  6 test_tcgen05_mma_multicast_commit
  4 test_tma_gather_scatter_multi_cta
  4 test_tma_im2col
  3 test_tcgen05_mma_plain_kind_i8_runtime_sm100
  3 test_tcgen05_mma_plain_kind_runtime
  3 test_tma_multicast_copy
  3 test_tmem_linear_f16_roundtrip
  3 test_tmem_linear_m64_roundtrip_32x32b_fallback
  2 test_tcgen05_mma_multicast_commit_twocta_linear_acc
  2 test_tmem_legacy_m64_subview_default_load_auto_selects_splitn
  2 test_tmem_physical_bitcast_preserves_subview_mapping
  2 test_tmem_reduction_linear_former_clean_errors_are_supported
  2 test_tmem_subslice_block_m_64
  2 test_tmem_subslice_block_m_64_parent_layout
  1 test_async_copy_mbarrier
  1 test_device_tma_load
  1 test_device_tma_store
  1 test_gluon_tma_round_f32_to_tf32
  1 test_tcgen05_mma_plain_kind_i8_reports_clean_error
  1 test_tcgen05_mma_scaled_direct_multicast_barrier
  1 test_tcgen05_mma_scaled_minimal
  1 test_tma
  1 test_tma_slice
  1 test_tmem_copy_2d
  1 test_tmem_copy_no_scales_shared_linear_128x128b
  1 test_tmem_linear_roundtrip_blocked_fallback
  1 test_tmem_linear_runtime_view_bitcast_reports_clean_error
  1 test_tmem_packed_f16_roundtrip_atom_shapes
  1 test_tmem_physical_bitcast_mma_lhs
  1 test_tmem_reduction_linear_mixed_layout_uses_software_reduce
  1 test_tmem_reduction_non_f32_uses_software_reduce
```

This covers allocation, linear TMEM load/store, descriptor chains, copy/no-scale copy, M64 layouts, `ld.red`, TMA, TMA+MMA shared-input paths, proxy/mbarrier controls, plain MMAv5, scaled MMAv5, scale copy, multicast/two-CTA controls, and clean-error diagnostics.

## Split Runtime Results

Command shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> PYTHONPATH=.:./python \
  xargs -d '\n' pytest -q -s --tb=short --splits 4 --group <group> \
  < /tmp/test_core_tmem_nodes_round51.txt | tee /tmp/round51_core_exact_group<group>.log
```

Results:

```text
group 1/4, GPU 0: 142 passed, 19 skipped, 480 deselected in 6.36s
group 2/4, GPU 1: 124 passed, 37 skipped, 480 deselected in 69.35s
group 3/4, GPU 2: 161 passed, 480 deselected in 104.64s
group 4/4, GPU 3: 158 passed, 483 deselected in 66.61s
```

Aggregate:

```text
641 selected
585 passed
56 skipped
0 failed
```

`pytest-split` reported no duration data for this exact selector, so the groups used even splitting rather than duration-aware balancing.

## Classification

No compiler crash, verifier drift, unsupported-case regression, clean-negative drift, or runtime miscompile was found in this exact `test_core.py` TMEM-bearing breadth sweep.

No failure needed classification against an existing `FZ-*` bucket, and no new `FZ-*` bucket is needed from Round 51.

The terminated over-broad `-k` attempt did expose adjacent `test_mma_shared_inputs[...]` wrong-result failures, but those node IDs were outside the explicit TMEM-bearing selector and were not used for the Round 51 classification.
