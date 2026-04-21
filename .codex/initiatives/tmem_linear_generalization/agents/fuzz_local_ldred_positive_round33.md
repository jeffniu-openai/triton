# Round 33 Local: Positive `ld.red` Runtime Guardrail

Date: 2026-04-21

## Scope

Reran a broad positive `ld.red` runtime slice while excluding report-only,
resource-boundary, and explicitly named M64 rows. This keeps coverage on
descriptor chains, direct explicit layouts, non-f32 software reductions, scale
software reductions, row/column permutations, mixed layouts, and single-CTA
block layouts without re-recording the known `FZ-20260421-0012` M64 planner
failures.

Selector:

```bash
-k '(ld_red and not reports and not resource and not m64)'
```

The selector collected `204/1615` runtime rows.

## Commands

Required rebuild and collection:

```bash
make -j8
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ld_red and not reports and not resource and not m64)'
```

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ld_red and not reports and not resource and not m64)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ld_red and not reports and not resource and not m64)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ld_red and not reports and not resource and not m64)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ld_red and not reports and not resource and not m64)'
```

## Result

- Group 1: `51 passed, 1564 deselected`
- Group 2: `51 passed, 1564 deselected`
- Group 3: `51 passed, 1564 deselected`
- Group 4: `51 passed, 1564 deselected`
- Aggregate selected result: `204 passed`

No compiler crash, false unsupported diagnostic, opcode mismatch, runtime
miscompile, or new independent `FZ-*` bucket was found.
