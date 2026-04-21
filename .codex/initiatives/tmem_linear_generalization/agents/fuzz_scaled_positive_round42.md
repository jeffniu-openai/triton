# Round 42 Scaled-MMAv5 Positive Guardrail

Date: 2026-04-21 14:19 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Surface

This lane targeted checked-in positive scaled-MMAv5 runtime rows for root
format matrices, root format `use_acc`, accumulator subslice views, tile-
permuted 64-wide accumulator views, and narrow identity accumulator views. It
excluded report/resource/clean-boundary rows, leaving those to existing clean
diagnostic guardrails.

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_scaled_root_format_matrix or mma_scaled_root_format_use_acc or mma_scaled_acc_subslice_view_format_matrix or mma_scaled_acc_subslice_view_format_use_acc or mma_scaled_acc_tile_permuted_64_format_matrix or mma_scaled_acc_tile_permuted_64_format_use_acc or mma_scaled_acc_identity_narrow_format_matrix or mma_scaled_acc_identity_narrow_format_use_acc) and not reports and not resource and not clean'
```

Result: `94/1615` tests collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(mma_scaled_root_format_matrix or mma_scaled_root_format_use_acc or mma_scaled_acc_subslice_view_format_matrix or mma_scaled_acc_subslice_view_format_use_acc or mma_scaled_acc_tile_permuted_64_format_matrix or mma_scaled_acc_tile_permuted_64_format_use_acc or mma_scaled_acc_identity_narrow_format_matrix or mma_scaled_acc_identity_narrow_format_use_acc) and not reports and not resource and not clean'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k '(mma_scaled_root_format_matrix or mma_scaled_root_format_use_acc or mma_scaled_acc_subslice_view_format_matrix or mma_scaled_acc_subslice_view_format_use_acc or mma_scaled_acc_tile_permuted_64_format_matrix or mma_scaled_acc_tile_permuted_64_format_use_acc or mma_scaled_acc_identity_narrow_format_matrix or mma_scaled_acc_identity_narrow_format_use_acc) and not reports and not resource and not clean'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k '(mma_scaled_root_format_matrix or mma_scaled_root_format_use_acc or mma_scaled_acc_subslice_view_format_matrix or mma_scaled_acc_subslice_view_format_use_acc or mma_scaled_acc_tile_permuted_64_format_matrix or mma_scaled_acc_tile_permuted_64_format_use_acc or mma_scaled_acc_identity_narrow_format_matrix or mma_scaled_acc_identity_narrow_format_use_acc) and not reports and not resource and not clean'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k '(mma_scaled_root_format_matrix or mma_scaled_root_format_use_acc or mma_scaled_acc_subslice_view_format_matrix or mma_scaled_acc_subslice_view_format_use_acc or mma_scaled_acc_tile_permuted_64_format_matrix or mma_scaled_acc_tile_permuted_64_format_use_acc or mma_scaled_acc_identity_narrow_format_matrix or mma_scaled_acc_identity_narrow_format_use_acc) and not reports and not resource and not clean'
```

## Results

- Group 1: `24 passed, 1591 deselected`
- Group 2: `24 passed, 1591 deselected`
- Group 3: `24 passed, 1591 deselected`
- Group 4: `22 passed, 1593 deselected`

Aggregate: `94 passed`.

## Classification

No compiler crash, false unsupported diagnostic, opcode absence, runtime
miscompile, unexpected pass/fail transition, or new independent `FZ-*` bucket
was observed.
