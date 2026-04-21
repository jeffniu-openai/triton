# Round 39 Allocation-Lifetime Guardrail

Date: 2026-04-21 14:55 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Run a focused checked-in runtime guardrail over TMEM allocation lifetime and
source-initialization rows. This lane checks allocator behavior for 1CTA and
2CTA load/store layouts at small, medium, and wide `N`, plus source-initialized
allocation lifetime handling.

## Commands

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(alloc_lifetime or allocation or source_initialization) and not reports and not resource'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(alloc_lifetime or allocation or source_initialization) and not reports and not resource'
```

## Results

- Collection: `7/1615`.
- Runtime result: `7 passed`.

## Classification

No new independent `FZ-*` bucket was found.

The selected allocation-lifetime rows stayed green. There was no allocator
assertion, resource drift, runtime miscompile, compiler crash, false
unsupported diagnostic, or unexpected pass/fail transition.

Backend repair remains deferred while discovery lanes continue.
