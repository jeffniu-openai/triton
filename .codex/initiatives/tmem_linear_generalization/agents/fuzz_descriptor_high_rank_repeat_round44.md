# Round 44 Descriptor/High-Rank Positive Repeat

Date: 2026-04-21 14:27 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Surface

This lane repeated the checked-in descriptor/high-rank positive selector over
descriptor compositions, higher-rank views, rank-5 roots, multidim slices, and
half-row positives. It reused stable per-GPU caches after several prior fuzzing
rounds as a warm-cache positive guardrail.

## Commands

Required rebuild:

```bash
make -j8
```

Result: build-tree ninja reported `no work to do`.

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows) and not reports and not resource and not clean'
```

Result: `140/1615` tests collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows) and not reports and not resource and not clean'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k '(descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows) and not reports and not resource and not clean'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k '(descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows) and not reports and not resource and not clean'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k '(descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows) and not reports and not resource and not clean'
```

## Results

- Group 1: `35 passed, 1580 deselected`
- Group 2: `35 passed, 1580 deselected`
- Group 3: `27 passed, 8 skipped, 1580 deselected`
- Group 4: `23 passed, 12 skipped, 1580 deselected`

Aggregate: `120 passed, 20 skipped`.

## Classification

No compiler crash, false unsupported diagnostic, clean-boundary drift,
unexpected pass/fail transition, runtime miscompile, warm-cache sensitivity, or
new independent `FZ-*` bucket was observed.
