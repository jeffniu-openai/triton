# Round 30 Local: Clean-Boundary Runtime Guardrail

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler code was changed.

## Summary

The checked-in clean-boundary runtime selector stayed green across four GPUs:

```text
184 passed
0 failed
```

Selector:

```text
reports_clean or clean_unsupported or clean_error or tmem_oor
```

This reruns the committed clean-negative/OOR surface while Round 30 subagents
continue adversarial descriptor-chain, 64-bit bitwidth, and copy-branch probes.

## Commands

Collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'reports_clean or clean_unsupported or clean_error or tmem_oor'
```

Result:

```text
184/1615 tests collected
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <1..4> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'reports_clean or clean_unsupported or clean_error or tmem_oor'
```

Shard results:

```text
group 1: 46 passed
group 2: 46 passed
group 3: 46 passed
group 4: 46 passed
aggregate: 184 passed
```

Logs:

```text
/tmp/tmem_r30_clean_boundary_g1.log
/tmp/tmem_r30_clean_boundary_g2.log
/tmp/tmem_r30_clean_boundary_g3.log
/tmp/tmem_r30_clean_boundary_g4.log
```

## Classification

No new bucket. The committed clean-boundary runtime surface remains stable.
