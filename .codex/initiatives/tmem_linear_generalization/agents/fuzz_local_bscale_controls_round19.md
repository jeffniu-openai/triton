# Round 19 Local: B-scale descriptor-view controls

- Date: 2026-04-21 12:14 UTC
- Branch: `codex/tmem`
- Commit at start: `ff839cba0`
- Mode: discovery/validation only. No backend/compiler fixes were attempted.

## Scope

This local slice checked current-head positive B-scale descriptor-view controls
adjacent to report-only `FZ-20260421-0015`:

- B-scale descriptor view consumed by scaled-MMAv5;
- B-scale descriptor view with an extra live `tmem_load` user;
- padded B-scale descriptor-view storage.

Selector:

```text
mma_scaled and bscale and not reports
```

## Required Rebuild

Command:

```bash
make -j8
```

Result from the immediately preceding local work: no work to do.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and bscale and not reports'
```

Result: `3/1615` selected.

## Runtime

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and bscale and not reports'
```

Result: `3 passed, 1612 deselected in 6.93s`.

## Classification

No new failures were observed in this selector.

- New independent `FZ-*`: none.
- Runtime miscompile: none observed.
- Compiler crash: none observed.
- Unexpected unsupported diagnostic: none observed.

These rows are green controls only. They do not clear `FZ-20260421-0015`,
which is specific to runtime-selected distinct direct B-scale descriptors in
the minimized Round 15/16 shape.
