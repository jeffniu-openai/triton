# Round 23 Local: Subword Runtime Baseline

Date: 2026-04-21 13:18 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

This local slice covered subword TMEM ld/st and copy runtime rows outside the
active Round 23 subagent lanes. It excluded `cp_no_scales_warpx2` rows because
those are covered by the copy/high-CGA fuzz lanes.

Covered families:

- subword ld/st pack and unpack;
- subword descriptor-chain ld/st roundtrips;
- x1 subword roundtrips;
- 2CTA x1 subword direct and descriptor-chain roundtrips;
- linear subword no-scale copy rows; and
- exact-width 128x128 subword no-scale copy rows.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Collect-only selected `42/1615` rows:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'subword and not reports and not cp_no_scales_warpx2'
```

Split-4 runtime sweep used stable per-GPU caches and stored durations in:

```text
/tmp/tmem_r23_subword_durations.json
```

## Results

Aggregate result: `42 passed`.

Per-shard results:

- GPU 0 / group 1: `11 passed, 1604 deselected`.
- GPU 1 / group 2: `11 passed, 1604 deselected`.
- GPU 2 / group 3: `11 passed, 1604 deselected`.
- GPU 3 / group 4: `9 passed, 1606 deselected`.

## Classification

No runtime miscompile, compiler crash, false unsupported diagnostic, or new
independent `FZ-*` bucket was found.

This remains a green baseline for subword ld/st and no-scale copy paths.
