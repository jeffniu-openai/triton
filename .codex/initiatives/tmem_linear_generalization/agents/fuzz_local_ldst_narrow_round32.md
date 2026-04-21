# Round 32 Local: ld/st Narrow and Half-Row Runtime Guardrail

Date: 2026-04-21

## Scope

This local slice reran checked-in `ld/st` runtime rows around narrow `N=32`,
half-row, `i32` broad-layout, and subword-adjacent `16x128b`/`16x256b`
variants. It complements the Round 32 subword/narrow-shape subagent lane.

Selector:

```bash
-k '(ldst and (n32 or half_rows or x1 or i32_broad) and not reports and not resource and not clean and not roundtrip)'
```

The selector collected `141/1615` runtime rows.

## Commands

Required rebuild:

```bash
make -j8
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst and (n32 or half_rows or x1 or i32_broad) and not reports and not resource and not clean and not roundtrip)'
```

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst and (n32 or half_rows or x1 or i32_broad) and not reports and not resource and not clean and not roundtrip)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst and (n32 or half_rows or x1 or i32_broad) and not reports and not resource and not clean and not roundtrip)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst and (n32 or half_rows or x1 or i32_broad) and not reports and not resource and not clean and not roundtrip)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst and (n32 or half_rows or x1 or i32_broad) and not reports and not resource and not clean and not roundtrip)'
```

## Result

- Group 1: `36 passed, 1579 deselected`
- Group 2: `36 passed, 1579 deselected`
- Group 3: `36 passed, 1579 deselected`
- Group 4: `33 passed, 1582 deselected`
- Aggregate selected result: `141 passed`

Group 3 was slower (`104.45s`) but completed successfully. No compiler crash,
false unsupported diagnostic, opcode mismatch, runtime miscompile, or new
independent `FZ-*` bucket was found.

