# Round 15 local higher-rank descriptor runtime sweep

Date: 2026-04-21 11:26 UTC
Branch: `codex/tmem`
HEAD before report: `1ec3f16e7`
Mode: discovery/cataloging only. No backend or compiler code was changed.

## Summary

This local sweep targeted checked-in higher-rank and multidimensional
descriptor-view runtime rows for TMEM `ld/st` and adjacent `ld.red` replay
coverage. The goal was to keep pressure on descriptor algebra, rank lifting,
dim0 slices, half-row views, rank-5 parents, and legal two-CTA variants after
the high-CGA ownership checkpoint.

No new `FZ-*` bucket was found. The selected rows stayed green or skipped
according to their checked-in expectations.

## Required rebuild

Command:

```bash
make -j8
```

Result: no work to do, inherited from the immediately preceding local slice.

## Selector

Collect-only command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(higher_rank or multidim_slice or half_rows or rank5) and not reports'
```

Result: `102/1615` selected.

Runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_local_r15_higher_rank_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(higher_rank or multidim_slice or half_rows or rank5) and not reports'
```

## Results

- GPU 0 / group 1: `26 passed, 1589 deselected in 18.12s`.
- GPU 1 / group 2: `26 passed, 1589 deselected in 16.09s`.
- GPU 2 / group 3: `7 passed, 19 skipped, 1589 deselected in 2.98s`.
- GPU 3 / group 4: `23 passed, 1 skipped, 1591 deselected in 2.55s`.
- Aggregate: `82 passed, 20 skipped`.

## Classification

- New independent `FZ-*`: none.
- Runtime miscompile: none observed.
- Compiler crash: none observed.
- False unsupported diagnostic: none observed in this checked-in selector.
- Stable positives: higher-rank dim0 slices, half-row direct and descriptor
  views, rank-5 roundtrips, unit-parent rank-5 rows, and legal two-CTA variants.

Recommended next discovery slice: move from checked-in higher-rank positives to
temporary adversarial probes that combine higher-rank descriptor views with
generic control-flow selected memdesc values, runtime indexing, or scaled-MMAv5
scale descriptors. Those combinations are being explored by the active Round 15
subagent lanes.
