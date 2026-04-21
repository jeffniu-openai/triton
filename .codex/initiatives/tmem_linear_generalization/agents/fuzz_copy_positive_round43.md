# Round 43 Copy Positive Guardrail

Date: 2026-04-21 14:25 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Surface

This lane targeted checked-in positive copy rows for no-scales linear copies,
indexed/subslice descriptor views, `warpx2::{01_23,02_13}` candidate positives,
tile-permuted linear copies, and scaled-copy geometry via scaled-MMAv5. It
excluded report/resource/subword/rematerialization rows, leaving those to
separate guardrails.

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales_linear or cp_no_scales_indexed_view_canonicalized or cp_no_scales_warpx2_01_23_candidate_positive or cp_no_scales_warpx2_02_13_candidate_positive or cp_scales_warpx4_via_scaled_mma_geometry_sweep) and not reports and not resource and not subword and not rematerializes'
```

Result: `77/1615` tests collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(cp_no_scales_linear or cp_no_scales_indexed_view_canonicalized or cp_no_scales_warpx2_01_23_candidate_positive or cp_no_scales_warpx2_02_13_candidate_positive or cp_scales_warpx4_via_scaled_mma_geometry_sweep) and not reports and not resource and not subword and not rematerializes'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k '(cp_no_scales_linear or cp_no_scales_indexed_view_canonicalized or cp_no_scales_warpx2_01_23_candidate_positive or cp_no_scales_warpx2_02_13_candidate_positive or cp_scales_warpx4_via_scaled_mma_geometry_sweep) and not reports and not resource and not subword and not rematerializes'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k '(cp_no_scales_linear or cp_no_scales_indexed_view_canonicalized or cp_no_scales_warpx2_01_23_candidate_positive or cp_no_scales_warpx2_02_13_candidate_positive or cp_scales_warpx4_via_scaled_mma_geometry_sweep) and not reports and not resource and not subword and not rematerializes'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k '(cp_no_scales_linear or cp_no_scales_indexed_view_canonicalized or cp_no_scales_warpx2_01_23_candidate_positive or cp_no_scales_warpx2_02_13_candidate_positive or cp_scales_warpx4_via_scaled_mma_geometry_sweep) and not reports and not resource and not subword and not rematerializes'
```

## Results

- Group 1: `20 passed, 1595 deselected`
- Group 2: `20 passed, 1595 deselected`
- Group 3: `20 passed, 1595 deselected`
- Group 4: `17 passed, 1598 deselected`

Aggregate: `77 passed`.

## Classification

No compiler crash, false unsupported diagnostic, opcode absence, runtime
miscompile, unexpected pass/fail transition, or new independent `FZ-*` bucket
was observed.
