# Round 29 Local: Broad Scaled-MMAv5 Runtime Guardrail

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler code was changed.

## Summary

The broad checked-in scaled-MMAv5 runtime selector completed cleanly across
four GPUs:

```text
243 passed
0 failed
```

Selector:

```text
mma_scaled and not reports and not resource
```

This selector includes root scaled-MMAv5 rows, accumulator indexed/subslice
views, two-CTA accumulator subslices, LHS descriptor views, tile-permuted and
narrow accumulator layouts, direct scale-copy rows selected by `mma_scaled`
markers, and nearby descriptor-view controls that are not marked as report-only
or resource-boundary rows.

No compiler crash, verifier failure, runtime miscompile, opcode mismatch, or
new independent `FZ-*` bucket was found.

## Commands

Collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and not reports and not resource'
```

Result:

```text
243/1615 tests collected
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group <1..4> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and not reports and not resource'
```

Shard results:

```text
group 1: 61 passed, 1554 deselected
group 2: 61 passed, 1554 deselected
group 3: 61 passed, 1554 deselected
group 4: 60 passed, 1555 deselected
aggregate: 243 passed
```

Logs:

```text
/tmp/tmem_round29_mma_scaled_g1.log
/tmp/tmem_round29_mma_scaled_g2.log
/tmp/tmem_round29_mma_scaled_g3.log
/tmp/tmem_round29_mma_scaled_g4.log
```

## Classification

No new bucket. This is a green guardrail for the checked-in scaled-MMAv5
runtime matrix outside the report-only and resource-boundary rows. It does not
clear report-only temporary probes for `FZ-20260421-0013` or
`FZ-20260421-0015`, but it confirms the committed positive surface remains
stable while discovery fuzzing continues.

