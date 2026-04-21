# Round 13 Local: Scaled-MMAv5 FP4, Tile-Permuted, and Narrow Coverage

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Repo edit scope: this report only.

## Scope

This local lane stress-tested checked-in scaled-MMAv5 runtime coverage around
format combinations and accumulator layout shapes that have historically been
sensitive:

- MXFP4/NVFP4 and mixed FP8/FP4 formats;
- root and indexed accumulator views;
- accumulator subslice views;
- 1CTA and 2CTA scaled-MMAv5 rows;
- LHS subslice and LHS tile-permuted tensor-memory operands;
- tile-permuted accumulator layouts;
- narrow accumulator fragments; and
- clean unsupported mixed-precision FP4-LHS diagnostics.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and (tile_permuted or narrow or e2m1 or fp4)'
```

Result: `202/1615` collected.

Split-4 runtime command:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_local_r13_scaled_fp4_tile_narrow_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and (tile_permuted or narrow or e2m1 or fp4)'
```

Results:

```text
group 1/GPU 0: 51 passed, 1564 deselected in 6.12s
group 2/GPU 1: 51 passed, 1564 deselected in 9.76s
group 3/GPU 2: 51 passed, 1564 deselected in 10.66s
group 4/GPU 3: 49 passed, 1566 deselected in 5.04s
aggregate:     202 passed
```

Durations were stored at
`/tmp/tmem_local_r13_scaled_fp4_tile_narrow_durations.json`.

## Classification

No new `FZ-*` candidate was found.

The checked-in scaled-MMAv5 coverage remains green for the selected FP4,
tile-permuted, LHS-TMEM, indexed/subslice accumulator, 2CTA, narrow, and clean
unsupported diagnostic rows. This lane is a negative result only.
