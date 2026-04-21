# Round 33 Local: Scaled-MMAv5 Layout/Subslice Guardrail

Date: 2026-04-21

## Scope

Reran the checked-in scaled-MMAv5 runtime surface that stresses operand layout
planning and accumulator/LHS descriptor views. This slice covers:

- layout-format matrix rows,
- indexed accumulator views,
- accumulator subslice views,
- LHS subslice views,
- tile-permuted accumulator/LHS rows,
- valid 2CTA accumulator subslice rows, and
- narrow indexed accumulator identity views.

Selector:

```bash
-k '(mma_scaled and not reports and not resource and (layout_format or lhs_subslice or acc_subslice or tile_permuted or indexed_acc))'
```

The selector collected `208/1615` runtime rows.

## Commands

Required rebuild and collection:

```bash
make -j8
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_scaled and not reports and not resource and (layout_format or lhs_subslice or acc_subslice or tile_permuted or indexed_acc))'
```

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_scaled and not reports and not resource and (layout_format or lhs_subslice or acc_subslice or tile_permuted or indexed_acc))'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_scaled and not reports and not resource and (layout_format or lhs_subslice or acc_subslice or tile_permuted or indexed_acc))'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_scaled and not reports and not resource and (layout_format or lhs_subslice or acc_subslice or tile_permuted or indexed_acc))'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_scaled and not reports and not resource and (layout_format or lhs_subslice or acc_subslice or tile_permuted or indexed_acc))'
```

## Result

- Group 1: `52 passed, 1563 deselected`
- Group 2: `52 passed, 1563 deselected`
- Group 3: `52 passed, 1563 deselected`
- Group 4: `52 passed, 1563 deselected`
- Aggregate selected result: `208 passed`

No compiler crash, false unsupported diagnostic, opcode mismatch, runtime
miscompile, or new independent `FZ-*` bucket was found.
