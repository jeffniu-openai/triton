# Round 34 Local: No-Scale Copy Tile/Subword/2CTA Guardrail

Date: 2026-04-21

## Scope

Reran a focused checked-in no-scale `tcgen05.copy` runtime slice after the
Round 33 scale, `ld.red`, and scaled-MMAv5 guardrails. This selector covers
tile-permuted linear copies, subword-adjacent copies and clean subword
diagnostics, 2CTA codegen rows, 2CTA `128x128b` codegen rows, and dense-shared
`warpx2` rematerialization rows.

Selector:

```bash
-k '(cp_no_scales and not reports and not resource and (tile_permuted or subword or twocta_codegen or twocta_128x128b or dense_shared))'
```

The selector collected `68/1615` runtime rows.

## Commands

Required rebuild and collection:

```bash
make -j8
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and not reports and not resource and (tile_permuted or subword or twocta_codegen or twocta_128x128b or dense_shared))'
```

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and not reports and not resource and (tile_permuted or subword or twocta_codegen or twocta_128x128b or dense_shared))'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and not reports and not resource and (tile_permuted or subword or twocta_codegen or twocta_128x128b or dense_shared))'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and not reports and not resource and (tile_permuted or subword or twocta_codegen or twocta_128x128b or dense_shared))'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and not reports and not resource and (tile_permuted or subword or twocta_codegen or twocta_128x128b or dense_shared))'
```

## Result

- Group 1: `17 passed, 1598 deselected`
- Group 2: `17 passed, 1598 deselected`
- Group 3: `17 passed, 1598 deselected`
- Group 4: `17 passed, 1598 deselected`
- Aggregate selected result: `68 passed`

No compiler crash, false unsupported diagnostic, opcode mismatch, runtime
miscompile, clean-boundary drift, or new independent `FZ-*` bucket was found.
