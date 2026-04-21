# Round 38: Copy, Scales, and Descriptor Rematerialization Fuzzing

Date: 2026-04-21 13:54 UTC
Branch: `codex/tmem`
HEAD before report: `7327a561662b2f866056fb9c97e106a121e604ef`
Scope: discovery only; no backend repairs attempted.

## Objective

Adversarially fuzz/test `tcgen05.copy`, `tcgen05.cp` scales-copy, and
descriptor rematerialization boundaries. Focus areas:

- no-scale `warpx2` / `warpx4` copy, including descriptor/index/slice/subslice
  views and dense-shared rematerialization;
- `cp.scales`/`tcgen05.cp.warpx4.32x128b`, public-scale descriptors, and
  shared-scale rematerialization;
- tile-permuted shared/TMEM layouts and clean unsupported diagnostics;
- opcode absence, false unsupported diagnostics, compiler crashes, and runtime
  miscompiles beyond existing buckets.

## Build

Required rebuild before probes:

```bash
make -j8
```

Result: build was current (`ninja: no work to do`).

## Checked-In Runtime Matrix Selector

Collected a broad copy/scales/rematerialization selector:

```bash
PYTHONPATH=.:./python pytest --collect-only -q -s --tb=short \
  -k '(cp_no_scales and (warpx2 or warpx4 or dense_shared or tile_permuted or indexed_view or subslice_view or slice_index_view) and not resource) or (cp_scales and (warpx4 or layout_probe or shared_subslice or tmem_descriptor_view)) or (shared_scale_descriptor_view_auto_tmem_copy or bscale_descriptor_view or bscale_view_extra_user)' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `176/1615` rows collected.

Four-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 -k '(cp_no_scales and (warpx2 or warpx4 or dense_shared or tile_permuted or indexed_view or subslice_view or slice_index_view) and not resource) or (cp_scales and (warpx4 or layout_probe or shared_subslice or tmem_descriptor_view)) or (shared_scale_descriptor_view_auto_tmem_copy or bscale_descriptor_view or bscale_view_extra_user)' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 -k '(cp_no_scales and (warpx2 or warpx4 or dense_shared or tile_permuted or indexed_view or subslice_view or slice_index_view) and not resource) or (cp_scales and (warpx4 or layout_probe or shared_subslice or tmem_descriptor_view)) or (shared_scale_descriptor_view_auto_tmem_copy or bscale_descriptor_view or bscale_view_extra_user)' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 -k '(cp_no_scales and (warpx2 or warpx4 or dense_shared or tile_permuted or indexed_view or subslice_view or slice_index_view) and not resource) or (cp_scales and (warpx4 or layout_probe or shared_subslice or tmem_descriptor_view)) or (shared_scale_descriptor_view_auto_tmem_copy or bscale_descriptor_view or bscale_view_extra_user)' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 -k '(cp_no_scales and (warpx2 or warpx4 or dense_shared or tile_permuted or indexed_view or subslice_view or slice_index_view) and not resource) or (cp_scales and (warpx4 or layout_probe or shared_subslice or tmem_descriptor_view)) or (shared_scale_descriptor_view_auto_tmem_copy or bscale_descriptor_view or bscale_view_extra_user)' python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `176 passed` (`44/44/44/44` by split group).

Coverage matrix:

| Surface | Rows | Result |
| --- | ---: | --- |
| no-scale copy indexed/subslice/slice-index descriptor views | included | pass |
| no-scale `warpx2::{01_23,02_13}` direct/view/dense-shared rows | included | pass / expected clean unsupported |
| tile-permuted and tile-selector-permuted no-scale copy | included | pass / expected clean unsupported |
| `cp.scales` layout probes and shared subslice rematerialization | included | pass |
| TMEM scale descriptor-view boundary | included | expected clean unsupported |
| public shared-scale/B-scale descriptor rematerialization | included | pass / expected clean unsupported |

## Checked-In Structural Copy-Scales Fuzzer

Collected:

```bash
PYTHONPATH=.:./python pytest --collect-only -q -s --tb=short \
  python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales
```

Result: `2` rows collected:

- `copy-scales-warpx4-1cta`
- `copy-scales-warpx4-2cta`

Four-GPU split run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales
```

Result: groups 1 and 2 ran one row each and passed. Groups 3 and 4 had no
assigned rows and returned pytest exit code 5 with `2 deselected`; this is a
split artifact, not a test failure.

## Temporary Rematerialization Probe

Temporary path:

```text
/root/tmp/fuzz_copy_scale_remat_round38_probe.py
```

The probe imports checked-in runtime-matrix helpers and replays a compact
cross-product of high-risk rows:

- linear indexed and subslice no-scale copy;
- tile-permuted and tile-selector-permuted no-scale copy;
- `warpx2::02_13` dense shared rematerialization;
- `warpx2::01_23` two-CTA dense shared rematerialization;
- direct `cp.scales` warpx4 single/two-CTA copy;
- `cp.scales` noncanonical shared-layout rematerialization;
- public shared-scale descriptor-view auto-copy;
- B-scale tile-permuted descriptor-view extra-user and padded-view remat;
- clean unsupported TMEM scale descriptor-view, B-scale `N=16`, and
  two-CTA `warpx2::02_13` dense-shared boundaries.

Collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest --collect-only -q -s --tb=short /root/tmp/fuzz_copy_scale_remat_round38_probe.py
```

Result: `18` rows collected.

Four-GPU split run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 1 /root/tmp/fuzz_copy_scale_remat_round38_probe.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 2 /root/tmp/fuzz_copy_scale_remat_round38_probe.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 3 /root/tmp/fuzz_copy_scale_remat_round38_probe.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 4 /root/tmp/fuzz_copy_scale_remat_round38_probe.py
```

Result: `18 passed` (`5/5/5/3` by split group).

Excluded temp draft: the first collection attempt referenced a stale helper
name, `_make_scales_shared_layout_warpx2_column_candidate`, and failed during
pytest collection before any runtime probe executed. The temp file was corrected
to `_make_scales_shared_layout_warpx2_row32_after_columns`; no repo source was
changed.

## Checked-In `test_core` Scaled-Copy Matrix

Collected:

```bash
PYTHONPATH=.:./python pytest --collect-only -q -s --tb=short \
  python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy
```

Result: `96` rows collected.

Four-GPU split run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy
```

Result: `96 passed` (`24/24/24/24` by split group).

## Classification

No new independent `FZ-*` bucket proposed.

Observed results:

- no compiler crash;
- no runtime miscompile in checked-in or temp runtime rows;
- no opcode absence: checked-in assertions continued to match `tcgen05.cp`
  PTX/LLIR copy opcodes and scaled MMA opcodes where expected;
- no false unsupported diagnostic found;
- clean unsupported diagnostics for TMEM scale descriptor views, B-scale
  `N=16`, no-scale subword/tile/row-permuted boundaries, and two-CTA
  `warpx2::02_13` dense-shared boundaries remained clean.

Relationship to existing buckets:

- The two-CTA `warpx2::02_13` no-scale copy boundary remains the known clean
  unsupported surface, not a new crash or false negative.
- Public shared-scale/B-scale descriptor-view rematerialization remains green
  for the checked-in rows and the temp replay matrix.
- Tile-permuted no-scale copy rows remain green where the opcode family can
  represent the layout and cleanly unsupported where subinstruction or
  row/column-order constraints make the mapping impossible.

Backend repair remains deferred.
