# Round 23 Local: ld/st descriptor composition guardrail

- Date: 2026-04-21 12:32 UTC
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler fixes attempted.
- Repo edit scope: this report only.

## Scope

This local slice exercised descriptor-based TMEM load/store paths outside the
scaled-MMAv5 accumulator lane:

- descriptor composition and roundtrip chains;
- row/column-permuted and exotic linear layouts;
- higher-rank and rank-5 descriptor views;
- direct half-row and lifted higher-rank layouts;
- `TensorMemoryScalesLayout` descriptor-view ld/st rows;
- 2CTA rank-5 descriptor rows.

The selector intentionally excluded scaled-MMAv5 rows and report-only clean
negative tests so unexpected failures would point at ordinary ld/st descriptor
lowering rather than the scaled-MMA operand buckets.

## Commands

Required build gate had already run for Round 23:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor or descriptor_view) and not reports and not mma_scaled'
```

Result:

```text
149/1615 tests collected (1466 deselected)
```

Split-4 runtime execution:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  --store-durations --durations-path /tmp/tmem_round23_ldst_desc_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor or descriptor_view) and not reports and not mma_scaled'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  --store-durations --durations-path /tmp/tmem_round23_ldst_desc_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor or descriptor_view) and not reports and not mma_scaled'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  --store-durations --durations-path /tmp/tmem_round23_ldst_desc_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor or descriptor_view) and not reports and not mma_scaled'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  --store-durations --durations-path /tmp/tmem_round23_ldst_desc_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor or descriptor_view) and not reports and not mma_scaled'
```

Result:

```text
group 1: 29 passed, 9 skipped, 1577 deselected
group 2: 38 skipped, 1577 deselected
group 3: 34 passed, 4 skipped, 1577 deselected
group 4: 25 passed, 10 skipped, 1580 deselected
aggregate: 88 passed, 61 skipped
```

## Classification

No new independent `FZ-*` bucket was found.

This is a green guardrail for ordinary descriptor-view load/store composition:
the currently cataloged dynamic generic descriptor failures remain sharper than
"descriptor views are broken generally." In particular, static descriptor
chains, rank-5 views, scales descriptor views, and nearby 2CTA descriptor ld/st
rows either executed correctly or skipped for existing environment/shape
guards.
