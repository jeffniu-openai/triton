# Round 37 Load/Store Positive Guardrail

Date: 2026-04-21 14:05 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Run a broad positive checked-in runtime guardrail over direct and descriptor
TMEM load/store rows while excluding report/resource/roundtrip/clean-diagnostic
rows. The selected surface covers narrow and broad layouts, row/column
permutations, exotic linear layouts, two-CTA layouts, allocation lifetime,
higher-rank views, fixed offset patterns, subword pack/unpack, and scale
layout variants.

## Commands

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst and not reports and not resource and not roundtrip and not clean'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst and not reports and not resource and not roundtrip and not clean'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst and not reports and not resource and not roundtrip and not clean'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst and not reports and not resource and not roundtrip and not clean'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst and not reports and not resource and not roundtrip and not clean'
```

## Results

- Collection: `201/1615`.
- Split-4 result:
  - group 1: `51 passed`
  - group 2: `51 passed`
  - group 3: `51 passed`
  - group 4: `48 passed`
- Aggregate: `201 passed`.

## Classification

No new independent `FZ-*` bucket was found.

There was no compiler crash, false unsupported diagnostic, opcode absence,
runtime miscompile, clean-boundary drift, or unexpected xfail/pass transition.
This is a green positive guardrail for the broad TMEM load/store surface after
the descriptor-equivalence lane sharpened existing descriptor-view wrong-result
buckets.

Backend repair remains deferred while discovery lanes continue.
