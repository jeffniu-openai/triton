# Round 25 Local: Scaled-MMAv5 Descriptor-View Runtime Baseline

Date: 2026-04-21 14:35 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

This local slice reran the checked-in scaled-MMAv5 descriptor-view runtime
rows as a nearby control for the report-only `FZ-20260421-0013` scale-view
miscompile bucket.

Selector:

```text
mma_scaled and descriptor_view and not reports
```

## Command

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and descriptor_view and not reports'
```

## Result

```text
4 passed, 1611 deselected
```

## Classification

No runtime miscompile, compiler crash, unexpected unsupported diagnostic, or
new independent `FZ-*` bucket was found.

This stays a green checked-in nearby control. It does not clear
`FZ-20260421-0013`, whose report-only temporary probes cover legal selected
scale descriptor-view shapes that still miscompile while side-channel
scale loads are correct.
