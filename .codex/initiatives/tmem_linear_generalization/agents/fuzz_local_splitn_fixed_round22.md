# Round 22 Local: Split-N and Fixed-Offset Baseline

Date: 2026-04-21 12:03 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

This local slice covered non-`ld.red` split-N layout selection and fixed-offset
TMEM ld/st patterns. It deliberately excluded `ld_red` rows because Round 22
Lane BB is fuzzing that space independently.

Covered families:

- split-N immediate offset checks;
- split-N auto selection of `16x32bx2`;
- explicit `16x32bx2` versus split-N equivalence;
- row/column-permuted non-reduction split-N sweeps; and
- fixed-offset 128x256 ld/st patterns.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Collect-only selected `33/1615` rows:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(splitn or fixed_offset) and not ld_red'
```

Split-4 runtime sweep used stable per-GPU caches and stored durations in:

```text
/tmp/tmem_r22_splitn_fixed_durations.json
```

## Results

Aggregate result: `33 passed`.

Per-shard results:

- GPU 0 / group 1: `9 passed, 1606 deselected`.
- GPU 1 / group 2: `9 passed, 1606 deselected`.
- GPU 2 / group 3: `9 passed, 1606 deselected`.
- GPU 3 / group 4: `6 passed, 1609 deselected`.

## Classification

No runtime miscompile, compiler crash, false unsupported diagnostic, or new
independent `FZ-*` bucket was found.

This is a green baseline for non-reduction split-N and fixed-offset ld/st
coverage. It complements the active `ld.red` Round 22 subagent lane by keeping
the non-reduction split-N path separated from M64 hardware-reduction planner
failures.
