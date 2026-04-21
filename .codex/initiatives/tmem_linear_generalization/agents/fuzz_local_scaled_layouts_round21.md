# Round 21 Local: Scaled-MMAv5 Layout Baseline

Date: 2026-04-21 12:03 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

This local slice covered scaled-MMAv5 rows adjacent to known scaled-MMA
weaknesses while excluding already-classified report-only `FZ-20260421-0015`
selected-B-scale rows. The goal was to sanity-check that broader scaled-MMA
layout coverage still passes for:

- LHS subslice layouts;
- LHS tile-permuted layouts;
- accumulator tile-permuted M64 and M32 layouts; and
- narrow identity accumulator layouts.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Collect-only before execution selected `91/1615` rows:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and (lhs_subslice or lhs_tile_permuted or acc_tile_permuted_64 or acc_tile_permuted_32 or acc_identity_narrow) and not reports and not fz0015'
```

Split-4 runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  --store-durations --durations-path /tmp/tmem_r21_scaled_layouts_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and (lhs_subslice or lhs_tile_permuted or acc_tile_permuted_64 or acc_tile_permuted_32 or acc_identity_narrow) and not reports and not fz0015'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  --store-durations --durations-path /tmp/tmem_r21_scaled_layouts_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and (lhs_subslice or lhs_tile_permuted or acc_tile_permuted_64 or acc_tile_permuted_32 or acc_identity_narrow) and not reports and not fz0015'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  --store-durations --durations-path /tmp/tmem_r21_scaled_layouts_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and (lhs_subslice or lhs_tile_permuted or acc_tile_permuted_64 or acc_tile_permuted_32 or acc_identity_narrow) and not reports and not fz0015'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  --store-durations --durations-path /tmp/tmem_r21_scaled_layouts_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and (lhs_subslice or lhs_tile_permuted or acc_tile_permuted_64 or acc_identity_narrow) and not reports and not fz0015'
```

## Results

Aggregate result: `91 passed`.

Per-shard results:

- GPU 0 / group 1: `23 passed, 1592 deselected`.
- GPU 1 / group 2: `23 passed, 1592 deselected`.
- GPU 2 / group 3: `23 passed, 1592 deselected`.
- GPU 3 / group 4: `22 passed, 1593 deselected`.

Durations file:

```text
/tmp/tmem_r21_scaled_layouts_durations.json
```

## Classification

No runtime miscompile, compiler crash, false unsupported diagnostic, or new
independent `FZ-*` bucket was found.

This is a useful green baseline around scaled-MMAv5 layouts excluding the
known selected-B-scale `FZ-20260421-0015` rows. It suggests that the current
scaled-MMA failures remain more specific to selected/dynamic descriptor
operands and allocation/addressing interactions than to these static layout
families alone.

## Follow-up Rerun

The selector was rerun on 2026-04-21 after noticing that one ad hoc shard
command had omitted `acc_tile_permuted_32` and that concurrent
`--store-durations` writes can race on the shared duration file. The clean
follow-up used read-only duration splitting:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  --splitting-algorithm=least_duration \
  --durations-path /tmp/tmem_r21_scaled_layouts_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and (lhs_subslice or lhs_tile_permuted or acc_tile_permuted_64 or acc_tile_permuted_32 or acc_identity_narrow) and not reports and not fz0015'
```

Stable aggregate result: `91 passed`.

- GPU 0 / group 1: `11 passed, 1604 deselected`.
- GPU 1 / group 2: `58 passed, 1557 deselected`.
- GPU 2 / group 3: `11 passed, 1604 deselected`.
- GPU 3 / group 4: `11 passed, 1604 deselected`.
