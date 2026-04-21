# Round 32 Local: MMAv5 Accumulator/View Runtime Guardrail

Date: 2026-04-21

## Scope

This local slice reran checked-in plain-MMAv5 runtime rows around M64,
two-CTA, indexed accumulator views, and accumulator subslice views. It is a
separate guardrail from the scaled-MMAv5 operand fuzzing and from the `ld/st`
narrow/subword lanes.

Selector:

```bash
-k 'tmem_runtime_matrix_mma and not ldst and not mma_scaled and not reports and not resource and (m64 or twocta or indexed_acc_view or acc_subslice_view)'
```

The selector collected `185/1615` runtime rows.

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'tmem_runtime_matrix_mma and not ldst and not mma_scaled and not reports and not resource and (m64 or twocta or indexed_acc_view or acc_subslice_view)'
```

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'tmem_runtime_matrix_mma and not ldst and not mma_scaled and not reports and not resource and (m64 or twocta or indexed_acc_view or acc_subslice_view)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'tmem_runtime_matrix_mma and not ldst and not mma_scaled and not reports and not resource and (m64 or twocta or indexed_acc_view or acc_subslice_view)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'tmem_runtime_matrix_mma and not ldst and not mma_scaled and not reports and not resource and (m64 or twocta or indexed_acc_view or acc_subslice_view)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'tmem_runtime_matrix_mma and not ldst and not mma_scaled and not reports and not resource and (m64 or twocta or indexed_acc_view or acc_subslice_view)'
```

## Result

- Group 1: `47 passed, 1568 deselected`
- Group 2: `47 passed, 1568 deselected`
- Group 3: `47 passed, 1568 deselected`
- Group 4: `44 passed, 1571 deselected`
- Aggregate selected result: `185 passed`

No compiler crash, false unsupported diagnostic, opcode mismatch, runtime
miscompile, or new independent `FZ-*` bucket was found.

