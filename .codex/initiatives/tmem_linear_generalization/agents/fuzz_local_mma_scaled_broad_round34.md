# Round 34 Local: Broad scaled-MMAv5 Runtime Guardrail

Date: 2026-04-21
Branch: `codex/tmem`

## Scope

Reran the broad checked-in scaled-MMAv5 positive runtime surface, excluding
report-only, resource-boundary, and clean-diagnostic rows. This complements the
narrower Round 32/33 scaled layout, operand, and two-CTA guardrails.

Selector:

```bash
-k 'mma_scaled and not reports and not resource and not clean'
```

No backend or checked-in test source was edited.

## Commands

Required rebuild and collection:

```bash
make -j8
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and not reports and not resource and not clean'
```

Collection result: `243/1615`.

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and not reports and not resource and not clean'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and not reports and not resource and not clean'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and not reports and not resource and not clean'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and not reports and not resource and not clean'
```

## Result

- Group 1: `61 passed, 1554 deselected`
- Group 2: `61 passed, 1554 deselected`
- Group 3: `61 passed, 1554 deselected`
- Group 4: `60 passed, 1555 deselected`
- Aggregate selected result: `243 passed`

No compiler crash, false unsupported diagnostic, opcode mismatch, runtime
miscompile, clean-boundary drift, or new independent `FZ-*` bucket was found.

## Classification

This is a green guardrail. It keeps the broad scaled-MMAv5 positive runtime
surface stable next to the report-only scaled buckets (`FZ-0007`, `FZ-0013`,
and `FZ-0015`) and the high-CGA ownership bucket (`FZ-0010`).
