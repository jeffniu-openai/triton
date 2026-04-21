# Round 39 TMA/Multicast Guardrail

Date: 2026-04-21 15:02 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Run a focused `python/test/gluon/test_core.py` guardrail over TMA multicast
copy and `tcgen05.mma` multicast commit paths. This keeps synchronization and
CGA ownership controls sampled while the broader high-CGA and MMAv5 lanes
continue to be tracked separately.

## Commands

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_core.py \
  -k 'tma_multicast or multicast_commit'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_core.py \
  -k 'tma_multicast or multicast_commit'
```

## Results

- Collection: `11/18114`.
- Runtime result: `11 passed`.

## Classification

No new independent `FZ-*` bucket was found.

The selected TMA multicast and MMAv5 multicast commit controls stayed green.
There was no compiler crash, false unsupported diagnostic, runtime miscompile,
or unexpected pass/fail transition.

Backend repair remains deferred while discovery lanes continue.
