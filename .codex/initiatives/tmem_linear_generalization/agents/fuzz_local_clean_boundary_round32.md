# Round 32 Local: Clean-Boundary Runtime Guardrail

Date: 2026-04-21

## Scope

This local slice reran the broad checked-in clean diagnostic surface after the
Round 31/32 fuzzing lanes.

Selector:

```bash
-k '(reports_clean or clean_unsupported or clean_error or tmem_oor or clean_xfail) and not resource'
```

The selector collected `184/1615` runtime rows.

## Commands

Required rebuild and collection:

```bash
make -j8
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(reports_clean or clean_unsupported or clean_error or tmem_oor or clean_xfail) and not resource'
```

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(reports_clean or clean_unsupported or clean_error or tmem_oor or clean_xfail) and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(reports_clean or clean_unsupported or clean_error or tmem_oor or clean_xfail) and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(reports_clean or clean_unsupported or clean_error or tmem_oor or clean_xfail) and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(reports_clean or clean_unsupported or clean_error or tmem_oor or clean_xfail) and not resource'
```

## Result

- Group 1: `46 passed, 1569 deselected`
- Group 2: `46 passed, 1569 deselected`
- Group 3: `46 passed, 1569 deselected`
- Group 4: `46 passed, 1569 deselected`
- Aggregate selected result: `184 passed`

No clean-diagnostic drift, compiler crash, false unsupported diagnostic,
unexpected xfail/pass transition, opcode mismatch, runtime miscompile, or new
independent `FZ-*` bucket was found.

