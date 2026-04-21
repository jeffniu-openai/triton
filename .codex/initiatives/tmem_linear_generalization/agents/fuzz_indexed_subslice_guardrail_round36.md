# Round 36 Indexed-Accumulator And Subslice Guardrail

Date: 2026-04-21 13:50 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Run a broad checked-in Python/Gluon runtime guardrail across descriptor
subslice and indexed-accumulator surfaces that are adjacent to the new TMEM
layout machinery. The selected rows cover no-scale copy subslice views,
`warpx2` subslice/indexed/slice-index views, 2CTA copy view paths, scaled-MMAv5
accumulator subslices, indexed accumulator views, and LHS subslices.

## Commands

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(indexed_acc or subslice_view or lhs_subslice) and not reports and not resource and not m64'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(indexed_acc or subslice_view or lhs_subslice) and not reports and not resource and not m64'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(indexed_acc or subslice_view or lhs_subslice) and not reports and not resource and not m64'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(indexed_acc or subslice_view or lhs_subslice) and not reports and not resource and not m64'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(indexed_acc or subslice_view or lhs_subslice) and not reports and not resource and not m64'
```

## Results

- Collection: `269/1615`.
- Split-4 result:
  - group 1: `68 passed`
  - group 2: `68 passed`
  - group 3: `68 passed`
  - group 4: `65 passed`
- Aggregate: `269 passed`.

## Classification

No new independent `FZ-*` bucket was found.

This lane did not expose a compiler crash, false unsupported diagnostic, opcode
absence, runtime miscompile, clean-boundary drift, or unexpected xfail/pass
transition. It provides a green guardrail around subslice/indexed descriptor
composition after earlier merge-recovery work exposed layout coverage weakness
in narrow TMEM/MMAv5 paths.

Backend repair remains deferred while discovery lanes continue.
