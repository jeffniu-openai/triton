# Round 30 Local: Descriptor ld/st Non-Roundtrip Guardrail

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler code was changed.

## Summary

The checked-in descriptor `ld/st` non-roundtrip selector stayed green across
four GPUs:

```text
53 passed
0 failed
```

Selector:

```text
ldst_descriptor and not reports and not roundtrip
```

This is a positive-surface guardrail for committed descriptor composition,
permuted layout, multidim slice, higher-rank slice, and half-row descriptor
`ld/st` rows outside report-only and roundtrip resource-boundary rows.

## Commands

Collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst_descriptor and not reports and not roundtrip'
```

Result:

```text
53/1615 tests collected
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <1..4> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst_descriptor and not reports and not roundtrip'
```

Shard results:

```text
group 1: 14 passed
group 2: 14 passed
group 3: 14 passed
group 4: 11 passed
aggregate: 53 passed
```

Logs:

```text
/tmp/tmem_r30_ldst_descriptor_g1.log
/tmp/tmem_r30_ldst_descriptor_g2.log
/tmp/tmem_r30_ldst_descriptor_g3.log
/tmp/tmem_r30_ldst_descriptor_g4.log
```

## Classification

No new bucket. The committed descriptor `ld/st` positive surface remains green
while Round 30 subagents fuzz adversarial descriptor-view chain shapes.
