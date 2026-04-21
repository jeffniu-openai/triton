# Round 23 Local: Plain MMAv5 Runtime Baseline

Date: 2026-04-21 13:05 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

This local slice covered broad plain MMAv5 TMEM runtime rows while excluding
scaled-MMAv5 rows, report/clean-diagnostic rows, and i8-only rows. It was run
in parallel with a separate Round 23 subagent lane focused on scaled-MMAv5
accumulator descriptor/view composition.

Covered families include:

- plain MMAv5 kind coverage with linear and legacy accumulator layouts;
- `use_acc` true/false rows;
- M64 accumulator and subslice views;
- 2CTA MMAv5 rows;
- 1CTA and 2CTA indexed accumulator views;
- tile-permuted accumulator layouts;
- LHS subslice plain MMAv5 rows; and
- TMA TF32 B-transposed descriptor rows.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Collect-only selected `255/1615` rows:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_twocta or plain_kinds or tma_tf32 or indexed_acc_view) and not mma_scaled and not reports and not clean and not i8'
```

Split-4 runtime sweep used stable per-GPU caches and stored durations in:

```text
/tmp/tmem_r23_plain_mma_durations.json
```

## Results

Aggregate result: `255 passed`.

Per-shard results:

- GPU 0 / group 1: `64 passed, 1551 deselected`.
- GPU 1 / group 2: `64 passed, 1551 deselected`.
- GPU 2 / group 3: `64 passed, 1551 deselected`.
- GPU 3 / group 4: `63 passed, 1552 deselected`.

## Classification

No runtime miscompile, compiler crash, false unsupported diagnostic, or new
independent `FZ-*` bucket was found.

This is an important green contrast for the active scaled-MMAv5 findings. The
plain MMAv5 paths still handle static accumulator layout families, indexed
accumulator views, subslice views, 2CTA rows, and TMA TF32 descriptor rows
across the broad checked-in matrix.
