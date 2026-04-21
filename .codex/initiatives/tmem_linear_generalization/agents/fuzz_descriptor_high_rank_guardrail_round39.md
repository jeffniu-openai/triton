# Round 39 Descriptor/High-Rank Positive Guardrail

Date: 2026-04-21 15:30 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Run a checked-in runtime guardrail over positive descriptor composition and
higher-rank TMEM load/store rows. The selected surface covers descriptor
composition sweeps, permuted/exotic layouts, two-CTA descriptor compositions,
rank-5 roundtrips, small rank-5 parents, unit-parent `N=256` rows, multidim
slices, and half-row positives while excluding report/resource/clean-diagnostic
rows.

## Commands

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows) and not reports and not resource and not clean'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows) and not reports and not resource and not clean'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows) and not reports and not resource and not clean'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows) and not reports and not resource and not clean'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows) and not reports and not resource and not clean'
```

## Results

- Collection: `140/1615`.
- Split-4 result:
  - group 1: `35 passed`
  - group 2: `35 passed`
  - group 3: `27 passed, 8 skipped`
  - group 4: `23 passed, 12 skipped`
- Aggregate: `120 passed, 20 skipped`.

## Classification

No new independent `FZ-*` bucket was found.

The selected descriptor-composition and higher-rank positive rows stayed green
or stably skipped. There was no compiler crash, false unsupported diagnostic,
runtime miscompile, clean-boundary drift, unexpected xfail/pass transition, or
new opcode absence signal.

Backend repair remains deferred while adversarial high-rank chain-shape and
`ld.red` extremes subagent lanes continue.
