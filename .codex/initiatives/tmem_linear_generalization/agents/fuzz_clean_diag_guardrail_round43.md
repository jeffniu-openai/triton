# Round 43 Clean Diagnostic Drift Guardrail

Date: 2026-04-21 14:23 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Surface

This lane targeted a compact clean-diagnostic selector across
`python/test/gluon/test_core.py` and `python/test/gluon/test_tmem_runtime_matrix.py`.
It covered clean errors, clean unsupported diagnostics, and backend diagnostic
rows while excluding copy-specific rows, generic MMA i8 rows, and resource rows
that have adjacent dedicated guardrails.

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_core.py python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(clean_error or clean_unsupported or reports_clean_error or reports_clean_unsupported or reports_backend_error) and not cp_no_scales and not cp_scales and not mma_i8 and not resource'
```

Result: `91/19729` tests collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_core.py python/test/gluon/test_tmem_runtime_matrix.py -k '(clean_error or clean_unsupported or reports_clean_error or reports_clean_unsupported or reports_backend_error) and not cp_no_scales and not cp_scales and not mma_i8 and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_core.py python/test/gluon/test_tmem_runtime_matrix.py -k '(clean_error or clean_unsupported or reports_clean_error or reports_clean_unsupported or reports_backend_error) and not cp_no_scales and not cp_scales and not mma_i8 and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_core.py python/test/gluon/test_tmem_runtime_matrix.py -k '(clean_error or clean_unsupported or reports_clean_error or reports_clean_unsupported or reports_backend_error) and not cp_no_scales and not cp_scales and not mma_i8 and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_core.py python/test/gluon/test_tmem_runtime_matrix.py -k '(clean_error or clean_unsupported or reports_clean_error or reports_clean_unsupported or reports_backend_error) and not cp_no_scales and not cp_scales and not mma_i8 and not resource'
```

## Results

- Group 1: `23 passed, 19706 deselected`
- Group 2: `23 passed, 19706 deselected`
- Group 3: `23 passed, 19706 deselected`
- Group 4: `22 passed, 19707 deselected`

Aggregate: `91 passed`.

## Classification

No compiler crash, diagnostic drift, unexpected pass/fail transition, runtime
miscompile, or new independent `FZ-*` bucket was observed.
