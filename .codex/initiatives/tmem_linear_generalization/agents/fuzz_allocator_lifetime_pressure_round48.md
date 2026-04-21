# Round 48 allocator, lifetime, and rematerialization pressure

Date: 2026-04-21

Scope: catalog-only adversarial fuzzing for TMEM allocator/lifetime pressure.
No backend code or checked-in tests were modified.  This lane focused on
checked-in Python runtime coverage in `python/test/gluon/test_tmem_runtime_matrix.py`
plus focused `test_core.py` controls.  Stable per-GPU caches were used throughout.

## Rebuild

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Runtime-matrix allocator/rematerialization pressure

Selector:

```text
(alloc_lifetime or allocation or source_initialization or rematerializes or use_acc or indexed_acc_view or acc_subslice_view or lhs_subslice_view or side_channel or side-channel or copy_mma) and not reports and not resource
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(alloc_lifetime or allocation or source_initialization or rematerializes or use_acc or indexed_acc_view or acc_subslice_view or lhs_subslice_view or side_channel or side-channel or copy_mma) and not reports and not resource'
```

Result:

```text
363/1615 tests collected (1252 deselected)
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(alloc_lifetime or allocation or source_initialization or rematerializes or use_acc or indexed_acc_view or acc_subslice_view or lhs_subslice_view or side_channel or side-channel or copy_mma) and not reports and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k '(alloc_lifetime or allocation or source_initialization or rematerializes or use_acc or indexed_acc_view or acc_subslice_view or lhs_subslice_view or side_channel or side-channel or copy_mma) and not reports and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k '(alloc_lifetime or allocation or source_initialization or rematerializes or use_acc or indexed_acc_view or acc_subslice_view or lhs_subslice_view or side_channel or side-channel or copy_mma) and not reports and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k '(alloc_lifetime or allocation or source_initialization or rematerializes or use_acc or indexed_acc_view or acc_subslice_view or lhs_subslice_view or side_channel or side-channel or copy_mma) and not reports and not resource'
```

Results:

```text
group 1: 91 passed, 1524 deselected
group 2: 91 passed, 1524 deselected
group 3: 91 passed, 1524 deselected
group 4: 90 passed, 1525 deselected
total:   363 passed
```

Coverage notes:

- `alloc_lifetime` and `source_initialization` rows exercise allocator lifetime
  contraction/extension and stores after intermediate TMEM users.
- `use_acc`, `indexed_acc_view`, and `acc_subslice_view` rows stress accumulator
  live-range reuse, accumulator descriptor views, and repeated MMA updates.
- `lhs_subslice_view` rows cover operand-side descriptor views that stay live
  across MMA consumers.
- `rematerializes` rows cover shared-source and scale-source rematerialization,
  including extra-user B-scale descriptor-view coverage that forces a
  side-channel `ttng.tmem_load` to remain live beside the MMA consumer.
- No compiler crash, verifier false-negative, wrong result, or new unsupported
  diagnostic appeared in this positive pressure sweep.

## Runtime-matrix diagnostic and resource boundaries

Selector:

```text
(allocator_crash or optimizer_crash or resource or reports_clean or clean_error or clean_unsupported or tmem_oor or clean_xfail) and (ld_red or alloc_lifetime or allocation or use_acc or ldst or mma or cp_no_scales or cp_scales)
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(allocator_crash or optimizer_crash or resource or reports_clean or clean_error or clean_unsupported or tmem_oor or clean_xfail) and (ld_red or alloc_lifetime or allocation or use_acc or ldst or mma or cp_no_scales or cp_scales)'
```

Result:

```text
184/1615 tests collected (1431 deselected)
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(allocator_crash or optimizer_crash or resource or reports_clean or clean_error or clean_unsupported or tmem_oor or clean_xfail) and (ld_red or alloc_lifetime or allocation or use_acc or ldst or mma or cp_no_scales or cp_scales)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k '(allocator_crash or optimizer_crash or resource or reports_clean or clean_error or clean_unsupported or tmem_oor or clean_xfail) and (ld_red or alloc_lifetime or allocation or use_acc or ldst or mma or cp_no_scales or cp_scales)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k '(allocator_crash or optimizer_crash or resource or reports_clean or clean_error or clean_unsupported or tmem_oor or clean_xfail) and (ld_red or alloc_lifetime or allocation or use_acc or ldst or mma or cp_no_scales or cp_scales)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k '(allocator_crash or optimizer_crash or resource or reports_clean or clean_error or clean_unsupported or tmem_oor or clean_xfail) and (ld_red or alloc_lifetime or allocation or use_acc or ldst or mma or cp_no_scales or cp_scales)'
```

Results:

```text
group 1: 46 passed, 1569 deselected
group 2: 46 passed, 1569 deselected
group 3: 46 passed, 1569 deselected
group 4: 46 passed, 1569 deselected
total:   184 passed
```

Classification:

- The `ld_red_identity_256_linear_layout_reports_resource_boundary` rows stayed
  clean report-only positives for the known large-geometry `ld.red` resource
  boundary tracked as `FZ-20260421-0018`.
- TMEM out-of-range and clean unsupported rows for `ld/st`, copy, MMAv5, scaled
  MMAv5, subword copy, exotic layouts, and high-CGA/two-CTA context checks all
  reported through expected clean diagnostics.
- No allocator-crash sentinel leaked into this runtime-matrix clean-boundary
  pass, and no positive case was reclassified as unsupported.

## Focused `test_core.py` lifetime and multi-allocation controls

Selector:

```text
(tmem_subslice_block_m_64 or tmem_physical_bitcast or tmem_copy_no_scales_matrix or tmem_copy_no_scales_shared_linear or tmem_descriptor_chain_matrix or tmem_legacy_m64_subview_default_load_auto_selects_splitn)
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_core.py \
  -k '(tmem_subslice_block_m_64 or tmem_physical_bitcast or tmem_copy_no_scales_matrix or tmem_copy_no_scales_shared_linear or tmem_descriptor_chain_matrix or tmem_legacy_m64_subview_default_load_auto_selects_splitn)'
```

Result:

```text
66/18114 tests collected (18048 deselected)
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_core.py -k '(tmem_subslice_block_m_64 or tmem_physical_bitcast or tmem_copy_no_scales_matrix or tmem_copy_no_scales_shared_linear or tmem_descriptor_chain_matrix or tmem_legacy_m64_subview_default_load_auto_selects_splitn)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_core.py -k '(tmem_subslice_block_m_64 or tmem_physical_bitcast or tmem_copy_no_scales_matrix or tmem_copy_no_scales_shared_linear or tmem_descriptor_chain_matrix or tmem_legacy_m64_subview_default_load_auto_selects_splitn)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_core.py -k '(tmem_subslice_block_m_64 or tmem_physical_bitcast or tmem_copy_no_scales_matrix or tmem_copy_no_scales_shared_linear or tmem_descriptor_chain_matrix or tmem_legacy_m64_subview_default_load_auto_selects_splitn)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_core.py -k '(tmem_subslice_block_m_64 or tmem_physical_bitcast or tmem_copy_no_scales_matrix or tmem_copy_no_scales_shared_linear or tmem_descriptor_chain_matrix or tmem_legacy_m64_subview_default_load_auto_selects_splitn)'
```

Results:

```text
group 1: 17 passed, 18097 deselected
group 2: 13 passed, 4 skipped, 18097 deselected
group 3: 16 passed, 1 skipped, 18097 deselected
group 4: 15 passed, 18099 deselected
total:   61 passed, 5 skipped
```

Coverage notes:

- `test_tmem_subslice_block_m_64` covers multiple TMEM allocations, subviews,
  physical bitcasts, repeated stores, and final loads through overlapping live
  views.  The `test_core.py` TTGIR assertions continue to see two independent
  64x128xf32 allocations.
- `test_tmem_physical_bitcast_mma_lhs` combines a rematerialized/bitcast TMEM
  LHS view with an MMAv5 consumer.
- `test_tmem_descriptor_chain_matrix` and M64 split-N default-load controls
  cover descriptor-view chains and auto-selected load variants outside the
  runtime-matrix parametrization.
- `test_tmem_copy_no_scales_matrix` and the shared-linear 128x128b control keep
  copy-path resource/lifetime behavior covered across 128-row and 256-row
  shapes.

## Structural allocator-crash sentinels

Selector:

```text
allocator_crash or ldst_256row_lifted_parent or ldred_1cta_direct_index
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'allocator_crash or ldst_256row_lifted_parent or ldred_1cta_direct_index'
```

Result:

```text
2/33 tests collected (31 deselected)
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'allocator_crash or ldst_256row_lifted_parent or ldred_1cta_direct_index'
```

Result:

```text
31 deselected, 2 xfailed
```

Classification:

- `test_tmem_structural_fuzzer_ldst_256row_lifted_parent_allocator_crash`
  remains the expected `FZ-20260421-0005` allocator-crash sentinel.
- `test_tmem_structural_fuzzer_ldred_1cta_direct_index_allocator_crash`
  remains the expected `FZ-20260421-0009` allocator-crash sentinel.
- Neither sentinel unexpectedly passed or changed failure class during this
  lane.

## Final classification

No new `FZ-*` bucket is needed from Round 48.

Current allocator/lifetime/resource classification after this lane:

- `FZ-20260421-0005`: still represented by the structural 256-row lifted-parent
  `ld/st` allocator xfail; no positive allocator/lifetime row regressed.
- `FZ-20260421-0009`: still represented by the structural 1CTA direct indexed
  `ld.red` allocator xfail; no new direct/indexed allocator assertion appeared.
- `FZ-20260421-0018`: large `ld.red` resource-boundary rows still report
  cleanly in the runtime matrix; no resource-boundary row leaked as an
  unexpected compiler crash.
- Allocation-order issues did not expand in this lane.  The broad `use_acc`,
  accumulator-view, scaled descriptor-view, and extra-user rematerialization
  positives passed.
- Clean resource boundaries stayed clean; there were no runtime wrong results,
  compiler crashes, false unsupported diagnostics, or verifier-tightening
  discoveries.
