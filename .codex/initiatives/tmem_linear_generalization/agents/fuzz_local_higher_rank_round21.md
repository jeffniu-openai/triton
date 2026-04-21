# Round 21 Local: higher-rank and multidim descriptor selector

- Date: 2026-04-21 12:47 UTC
- Branch: `codex/tmem`
- Commit at start: `9ca933c5c`
- Mode: discovery/validation only. No backend/compiler fixes were attempted.
- Duration cache:
  `/tmp/tmem_r21_higher_rank_durations.json`

## Scope

This local selector checked higher-rank, multidim-slice, half-row, and rank-5
descriptor runtime coverage while Round 21 subagents audited `FZ-0015` IR and
`FZ-0012` M64 `ld.red` minimization:

- single-CTA and 2CTA descriptor multidim slices;
- higher-rank index and lifted-layout rows;
- direct and descriptor half-row rows;
- direct higher-rank replay rows;
- rank-5 descriptor roundtrips, including small and unit-parent variants.

Selector:

```text
(higher_rank or multidim_slice or half_rows or rank5 or ldst_descriptor_higher_rank) and not reports
```

## Required Rebuild

Command:

```bash
make -j8
```

Result: no work to do.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(higher_rank or multidim_slice or half_rows or rank5 or ldst_descriptor_higher_rank) and not reports'
```

Result: `102/1615` selected.

## Runtime

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_r21_higher_rank_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(higher_rank or multidim_slice or half_rows or rank5 or ldst_descriptor_higher_rank) and not reports'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `26 passed, 1589 deselected in 4.52s` |
| 2 | 1 | `26 passed, 1589 deselected in 4.27s` |
| 3 | 2 | `7 passed, 19 skipped, 1589 deselected in 4.32s` |
| 4 | 3 | `23 passed, 1 skipped, 1591 deselected in 4.17s` |

Aggregate selected rows: `82 passed, 20 skipped`.

## Classification

No new failures were observed in this selector.

- New independent `FZ-*`: none.
- Runtime miscompile: none observed.
- Compiler crash: none observed.
- Unexpected unsupported diagnostic: none observed.

This is a current-head green/skipped baseline for higher-rank and multidim
descriptor runtime coverage.
