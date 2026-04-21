# Round 29 Local: No-Scale Copy Runtime Guardrail

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler code was changed.

## Summary

The checked-in no-scale `tcgen05.copy` selector completed cleanly across four
GPUs:

```text
197 passed
4 skipped
0 failed
```

Selector:

```text
cp_no_scales and not reports and not high and not cga and not selector and not subword
```

This is the same broad positive-surface copy selector used in earlier local
guardrails, excluding report-only rows, high-CGA ownership diagnostics,
selector-specific rows, and subword rows.

No compiler crash, verifier failure, runtime miscompile, opcode mismatch, or
new independent `FZ-*` bucket was found.

## Commands

Collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales and not reports and not high and not cga and not selector and not subword'
```

Result:

```text
201/1615 tests collected
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <1..4> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales and not reports and not high and not cga and not selector and not subword'
```

Shard results:

```text
group 1: 47 passed, 4 skipped
group 2: 51 passed
group 3: 51 passed
group 4: 48 passed
aggregate: 197 passed, 4 skipped
```

Logs:

```text
/tmp/tmem_r29_cp_no_scales_g1.log
/tmp/tmem_r29_cp_no_scales_g2.log
/tmp/tmem_r29_cp_no_scales_g3.log
/tmp/tmem_r29_cp_no_scales_g4.log
```

## Classification

No new bucket. The committed no-scale copy positive surface remains green while
Round 29 subagents fuzz dynamic descriptor SSA and verifier clean-boundary
surfaces.
