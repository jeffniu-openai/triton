# Round 40 Clean-Boundary Runtime Guardrail

Date: 2026-04-21 14:10 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Surface

This lane targeted checked-in runtime rows that assert clean frontend/backend
diagnostics or resource boundaries across `ld/st`, `ld.red`, plain MMAv5, and
scaled-MMAv5. Copy and TMA-heavy clean rows were excluded because concurrent
Round 40 subagents own proxy/TMA sequencing and copy/dynamic descriptor
surfaces.

## Commands

Small initial sentinel:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(clean_unsupported or resource) and not reports'
```

Result: `4 passed, 1611 deselected`.

Broader collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(reports_clean_error or reports_clean_unsupported or reports_tmem_oor or reports_resource_boundary) and not cp_no_scales and not cp_scales and not mma_twocta_tma'
```

Result: `112/1615` tests collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(reports_clean_error or reports_clean_unsupported or reports_tmem_oor or reports_resource_boundary) and not cp_no_scales and not cp_scales and not mma_twocta_tma'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k '(reports_clean_error or reports_clean_unsupported or reports_tmem_oor or reports_resource_boundary) and not cp_no_scales and not cp_scales and not mma_twocta_tma'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k '(reports_clean_error or reports_clean_unsupported or reports_tmem_oor or reports_resource_boundary) and not cp_no_scales and not cp_scales and not mma_twocta_tma'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k '(reports_clean_error or reports_clean_unsupported or reports_tmem_oor or reports_resource_boundary) and not cp_no_scales and not cp_scales and not mma_twocta_tma'
```

## Results

- Group 1: `28 passed, 1587 deselected`
- Group 2: `28 passed, 1587 deselected`
- Group 3: `28 passed, 1587 deselected`
- Group 4: `28 passed, 1587 deselected`

Aggregate: `112 passed`.

## Classification

No compiler crash, diagnostic drift, unexpected pass/fail transition, runtime
miscompile, or new independent `FZ-*` bucket was observed. The clean-boundary
surface remains stable for the selected `ld/st`, `ld.red`, plain MMAv5, and
scaled-MMAv5 rows.
