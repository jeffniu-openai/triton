# Round 34 Local: Two-CTA MMAv5/TMA Guardrail

Date: 2026-04-21

## Scope

Reran a checked-in two-CTA MMAv5 runtime slice focused on plain-kind rows,
indexed accumulator views, accumulator subslice views, and TMA B-transposed
descriptor rows. This complements the earlier plain-MMAv5 accumulator/view
guardrail by emphasizing the two-CTA and TMA-adjacent rows.

Selector:

```bash
-k '(mma_twocta and not reports and not resource and (tma or transposed or plain_kind or indexed_acc))'
```

The selector collected `103/1615` rows.

## Commands

Required rebuild and collection:

```bash
make -j8
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_twocta and not reports and not resource and (tma or transposed or plain_kind or indexed_acc))'
```

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_twocta and not reports and not resource and (tma or transposed or plain_kind or indexed_acc))'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_twocta and not reports and not resource and (tma or transposed or plain_kind or indexed_acc))'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_twocta and not reports and not resource and (tma or transposed or plain_kind or indexed_acc))'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_twocta and not reports and not resource and (tma or transposed or plain_kind or indexed_acc))'
```

## Result

- Group 1: `26 passed, 1589 deselected`
- Group 2: `26 passed, 1589 deselected`
- Group 3: `26 passed, 1589 deselected`
- Group 4: `25 passed, 1590 deselected`
- Aggregate selected result: `103 passed`

No compiler crash, false unsupported diagnostic, opcode mismatch, runtime
miscompile, clean-boundary drift, or new independent `FZ-*` bucket was found.
