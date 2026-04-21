# Round 37 MMAv5 Format And Use-Acc Guardrail

Date: 2026-04-21 14:00 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Run a checked-in Python/Gluon runtime guardrail over MMAv5 and scaled-MMAv5
format/use-acc surfaces. The selected rows cover plain MMAv5 operand kinds,
2CTA plain MMAv5, root-format scaled MMAv5, public-scale copy paths,
accumulator reuse, accumulator subslices, LHS subslices, tile-permuted LHS
views, and narrow indexed-accumulator use-acc cases.

## Commands

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(root_format or use_acc or plain_kind) and not reports and not resource and not m64'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(root_format or use_acc or plain_kind) and not reports and not resource and not m64'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(root_format or use_acc or plain_kind) and not reports and not resource and not m64'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(root_format or use_acc or plain_kind) and not reports and not resource and not m64'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(root_format or use_acc or plain_kind) and not reports and not resource and not m64'
```

## Results

- Collection: `255/1615`.
- Split-4 result:
  - group 1: `64 passed`
  - group 2: `64 passed`
  - group 3: `64 passed`
  - group 4: `63 passed`
- Aggregate: `255 passed`.

## Classification

No new independent `FZ-*` bucket was found.

The selected MMAv5/scaled-MMAv5 rows remained green across format combinations
and accumulator reuse paths. There was no compiler crash, false unsupported
diagnostic, opcode absence, runtime miscompile, clean-boundary drift, or
unexpected xfail/pass transition.

Backend repair remains deferred while discovery lanes continue.
