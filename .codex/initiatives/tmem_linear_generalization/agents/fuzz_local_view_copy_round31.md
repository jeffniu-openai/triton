# Round 31 Local: View-Chain and Copy `warpx2` Runtime Guardrail

Date: 2026-04-21

## Scope

This local slice mixed high-rank `ld/st` descriptor-view rows with
`tcgen05.copy` no-scales `warpx2` rows, including indexed, subslice, and
slice-index views. It is intended as a broad guardrail next to the Round 31
subagent lanes, not as a new backend repair.

Selector:

```bash
-k '(warpx2 or rank5 or multidim_slice or half_rows or slice_index or indexed_view) and not reports and not resource and not clean and not lifted'
```

The selector collected `197/1615` runtime-matrix rows. It intentionally
included some checked-in skipped descriptor roundtrip rows that remain out of
resource for the current allocation limits.

## Commands

Required rebuild:

```bash
make -j8
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(warpx2 or rank5 or multidim_slice or half_rows or slice_index or indexed_view) and not reports and not resource and not clean and not lifted'
```

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(warpx2 or rank5 or multidim_slice or half_rows or slice_index or indexed_view) and not reports and not resource and not clean and not lifted'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(warpx2 or rank5 or multidim_slice or half_rows or slice_index or indexed_view) and not reports and not resource and not clean and not lifted'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(warpx2 or rank5 or multidim_slice or half_rows or slice_index or indexed_view) and not reports and not resource and not clean and not lifted'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(warpx2 or rank5 or multidim_slice or half_rows or slice_index or indexed_view) and not reports and not resource and not clean and not lifted'
```

## Result

- Group 1: `50 skipped, 1565 deselected`
- Group 2: `30 passed, 20 skipped, 1565 deselected`
- Group 3: `46 passed, 4 skipped, 1565 deselected`
- Group 4: `47 passed, 1568 deselected`
- Aggregate selected result: `123 passed, 74 skipped`

No compiler crash, false unsupported diagnostic, opcode mismatch, or runtime
miscompile was found in the executed rows. No new `FZ-*` bucket is needed.

## Classification

This is a green guardrail for:

- high-rank `ld/st` descriptor-view rows, including rank-5 and multidimensional
  slice coverage;
- `tcgen05.copy` no-scales `warpx2::{01_23,02_13}` indexed/subslice/slice-index
  paths for 1CTA and 2CTA positive cases.

