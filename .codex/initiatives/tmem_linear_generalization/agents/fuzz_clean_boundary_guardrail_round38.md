# Round 38 Clean-Boundary Guardrail

Date: 2026-04-21 14:30 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Revalidate a focused checked-in diagnostic slice for clean TMEM boundaries:
unsupported `.x1` load/store message shapes and subword `tcgen05.copy`
diagnostics for `warpx2` and legacy copy paths. This lane checks diagnostic
stability rather than positive execution.

## Commands

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(clean_unsupported or clean_error or reports_clean or tmem_oor) and not reports'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(clean_unsupported or clean_error or reports_clean or tmem_oor) and not reports'
```

## Results

- Collection: `22/1615`.
- Runtime result: `22 passed`.

## Classification

No new independent `FZ-*` bucket was found.

The selected clean diagnostic rows remained stable. There was no verifier
over-strictness drift, unexpected compiler crash, false unsupported diagnostic,
or unexpected pass/fail transition.

Backend repair remains deferred while discovery lanes continue.
