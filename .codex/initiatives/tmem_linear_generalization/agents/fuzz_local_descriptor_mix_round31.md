# Round 31 Local: Descriptor-Mix Runtime Guardrail

Date: 2026-04-21

## Scope

This local slice reran a descriptor-heavy cross-family selector while Round 31
subagents fuzz scaled-MMAv5 operands, `ld.red` extremes, and compiler-only
TMEM verifier/index boundaries.

Selector:

```bash
-k '(mma_scaled or ld_red or cp_no_scales or copy) and descriptor and not reports and not resource and not clean'
```

The selector collected `54/1615` runtime-matrix rows:

- `ld.red` descriptor-chain rows, including `N=32/64/256`, row/col
  permutations, explicit variants, and non-f32 software-reduction descriptor
  rows.
- scaled-MMAv5 shared-scale descriptor-view rows and B-scale descriptor-view
  controls.

## Commands

Required rebuild:

```bash
make -j8
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_scaled or ld_red or cp_no_scales or copy) and descriptor and not reports and not resource and not clean'
```

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_scaled or ld_red or cp_no_scales or copy) and descriptor and not reports and not resource and not clean'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_scaled or ld_red or cp_no_scales or copy) and descriptor and not reports and not resource and not clean'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_scaled or ld_red or cp_no_scales or copy) and descriptor and not reports and not resource and not clean'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_scaled or ld_red or cp_no_scales or copy) and descriptor and not reports and not resource and not clean'
```

## Result

- Group 1: `14 passed, 1601 deselected`
- Group 2: `14 passed, 1601 deselected`
- Group 3: `14 passed, 1601 deselected`
- Group 4: `12 passed, 1603 deselected`
- Aggregate selected result: `54 passed`

No compiler crash, false unsupported diagnostic, opcode mismatch, or runtime
miscompile was found in this guardrail. No new `FZ-*` bucket is needed.

## Classification

This is a green control adjacent to:

- `FZ-20260421-0012` (`ld.red` M64 row-permuted destination-layout planner
  gap), which is excluded here by the selected descriptor rows.
- `FZ-20260421-0013` (scaled-MMAv5 scale descriptor-view miscompile), where
  the checked-in nearby descriptor-view controls remain green.

