# Round 29 Local: Structural Fuzzer Smoke

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler code was changed.

## Summary

The checked-in structural fuzzer stayed stable on a single GPU:

```text
9 passed
24 xfailed
0 failed
0 xpassed
```

Known xfail diagnostics still include the `FZ-20260421-0001` illegal
`ttg.memdesc_index` LLVM-conversion repros.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py \
  2>&1 | tee /tmp/tmem_r29_structural_single_gpu.log
```

## Result

```text
9 passed, 24 xfailed in 13.84s
```

## Classification

No new bucket. The checked-in structural fuzzer remains stable while Round 29
subagents fuzz dynamic descriptor SSA and clean-boundary surfaces.
