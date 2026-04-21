# Round 34 Local: High-Rank Descriptor `ld/st` Guardrail

Date: 2026-04-21

## Scope

Reran the checked-in high-rank descriptor `ld/st` runtime surface. This selector
covers rank-5 parents, unit-parent round trips, multidimensional slice/index
chains, higher-rank dim0 slicing, higher-rank half-row positives, row/column
permuted round trips, and existing allocation/resource-gated roundtrip rows.

Selector:

```bash
-k '(ldst_descriptor and not reports and not resource and (rank5 or higher_rank or multidim or roundtrip))'
```

The selector collected `104/1615` runtime rows.

## Commands

Required rebuild and collection:

```bash
make -j8
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor and not reports and not resource and (rank5 or higher_rank or multidim or roundtrip))'
```

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor and not reports and not resource and (rank5 or higher_rank or multidim or roundtrip))'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor and not reports and not resource and (rank5 or higher_rank or multidim or roundtrip))'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor and not reports and not resource and (rank5 or higher_rank or multidim or roundtrip))'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor and not reports and not resource and (rank5 or higher_rank or multidim or roundtrip))'
```

## Result

- Group 1: `26 skipped, 1589 deselected`
- Group 2: `1 passed, 25 skipped, 1589 deselected`
- Group 3: `19 passed, 7 skipped, 1589 deselected`
- Group 4: `23 passed, 3 skipped, 1589 deselected`
- Aggregate selected result: `43 passed, 61 skipped`

The skipped rows are existing allocation/resource-gated descriptor roundtrip
cases in this selector. No compiler crash, false unsupported diagnostic, opcode
mismatch, runtime miscompile, clean-boundary drift, unexpected skip/pass
transition, or new independent `FZ-*` bucket was found.
