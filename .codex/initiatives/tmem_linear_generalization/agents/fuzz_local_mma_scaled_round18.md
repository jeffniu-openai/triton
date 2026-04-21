# Round 18 Local: MMAv5 and scaled-MMAv5 runtime selector

- Date: 2026-04-21 11:54 UTC
- Branch: `codex/tmem`
- Commit at start: `7a388cf5e`
- Mode: discovery/validation only. No backend/compiler fixes were attempted.
- Duration cache:
  `/tmp/tmem_r18_mma_scaled_durations.json`

## Scope

This local selector checked broad green runtime coverage for MMAv5 and
scaled-MMAv5 rows while subagents probed narrower adversarial surfaces:

- 2CTA MMAv5 direct, indexed accumulator view, subslice accumulator view, and
  TMA-transposed descriptor rows;
- scaled-MMAv5 minimal, root-format, indexed-accumulator-view, accumulator
  subslice/tile-permuted, LHS subslice/tile-permuted, and narrow-accumulator
  rows;
- both `use_acc=False` and `use_acc=True` rows where represented by the
  checked-in matrix.

Selector:

```text
(mma_scaled and not reports and not fz0015 and not descriptor_view) or (mma_twocta and not reports_clean and not reports and not i8)
```

## Required Rebuild

Command:

```bash
make -j8
```

Result from the immediately preceding checkpoint: no work to do.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_scaled and not reports and not fz0015 and not descriptor_view) or (mma_twocta and not reports_clean and not reports and not i8)'
```

Result: `342/1615` selected.

## Runtime

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_r18_mma_scaled_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma_scaled and not reports and not fz0015 and not descriptor_view) or (mma_twocta and not reports_clean and not reports and not i8)'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `86 passed, 1529 deselected in 74.62s` |
| 2 | 1 | `86 passed, 1529 deselected in 19.96s` |
| 3 | 2 | `86 passed, 1529 deselected in 10.06s` |
| 4 | 3 | `84 passed, 1531 deselected in 25.76s` |

Aggregate selected rows: `342 passed`.

## Classification

No new failures were observed in this selector.

- New independent `FZ-*`: none.
- Runtime miscompile: none observed.
- Compiler crash: none observed.
- Unexpected unsupported diagnostic: none observed.

This is a current-head green baseline for a wide MMAv5/scaled-MMAv5 runtime
surface. It does not clear the report-only `FZ-20260421-0015` B-scale runtime
selection bucket because rows matching that class were intentionally excluded
from this green selector.
