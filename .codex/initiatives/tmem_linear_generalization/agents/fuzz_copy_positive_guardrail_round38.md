# Round 38 Copy Positive Guardrail

Date: 2026-04-21 14:24 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Run a broad checked-in runtime guardrail over positive `tcgen05.copy` rows
while excluding report/resource/clean-diagnostic cases. The selected surface
covers no-scale copy, swizzles, descriptor/index/slice/subslice views, `warpx2`
and `warpx4` variants, tile-permuted layouts, scale-copy layout probes,
noncanonical/shared-subslice rematerialization, and scaled-MMAv5 scale-copy
paths.

## Commands

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales or cp_scales) and not reports and not resource and not clean'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales or cp_scales) and not reports and not resource and not clean'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales or cp_scales) and not reports and not resource and not clean'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales or cp_scales) and not reports and not resource and not clean'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales or cp_scales) and not reports and not resource and not clean'
```

## Results

- Collection: `246/1615`.
- Split-4 result:
  - group 1: `58 passed, 4 skipped`
  - group 2: `62 passed`
  - group 3: `62 passed`
  - group 4: `60 passed`
- Aggregate: `242 passed, 4 skipped`.

## Classification

No new independent `FZ-*` bucket was found.

The selected positive copy rows remained green or stably skipped. There was no
compiler crash, false unsupported diagnostic, opcode absence, runtime
miscompile, clean-boundary drift, or unexpected xfail/pass transition. The
skips are existing environment/resource-gated rows selected by the broad copy
keywords.

Backend repair remains deferred while deeper copy/scale rematerialization and
compiler-boundary subagent lanes continue.
