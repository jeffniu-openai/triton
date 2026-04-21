# Round 38 `test_core.py` TMEM/MMAv5 Guardrail

Date: 2026-04-21 14:35 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Run a checked-in `python/test/gluon/test_core.py` guardrail outside the TMEM
runtime matrix. The selected rows cover `tcgen05` MMAv5 multicast commit,
plain-kind runtime, scaled-MMAv5 minimal/copy paths, scaled copy linear
accumulators, and reduction-former clean-error support checks.

## Commands

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_core.py \
  -k 'tcgen05 and (tmem or mma or copy or multicast or mbarrier)'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_core.py \
  -k 'tcgen05 and (tmem or mma or copy or multicast or mbarrier)'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_core.py \
  -k 'tcgen05 and (tmem or mma or copy or multicast or mbarrier)'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_core.py \
  -k 'tcgen05 and (tmem or mma or copy or multicast or mbarrier)'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_core.py \
  -k 'tcgen05 and (tmem or mma or copy or multicast or mbarrier)'
```

## Results

- Collection: `123/18114`.
- Split-4 result:
  - group 1: `28 passed, 3 skipped`
  - group 2: `31 passed`
  - group 3: `31 passed`
  - group 4: `30 passed`
- Aggregate: `120 passed, 3 skipped`.

## Classification

No new independent `FZ-*` bucket was found.

The selected `test_core.py` TMEM/MMAv5 rows remained green or stably skipped.
There was no compiler crash, false unsupported diagnostic, opcode absence,
runtime miscompile, clean-boundary drift, or unexpected xfail/pass transition.

Backend repair remains deferred while discovery lanes continue.
