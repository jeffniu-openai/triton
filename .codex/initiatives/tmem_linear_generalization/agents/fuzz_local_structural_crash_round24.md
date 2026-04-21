# Round 24 Local: Structural Fuzzer Crash Repro Guardrail

Date: 2026-04-21 14:00 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

Reran the three checked-in structural fuzzer crash repro rows in their existing
pytest child-process wrappers.  These rows intentionally remain xfailed while
the campaign is still in discovery mode.

Selector:

```text
allocator_crash or optimizer_crash
```

## Command

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'allocator_crash or optimizer_crash' \
  2>&1 | tee /tmp/tmem_r24_structural_crash_repros.log
```

Result:

```text
30 deselected, 3 xfailed
```

## Classification

No unexpected pass, unexpected failure, or changed crash mode was observed.
The rows continue to track existing crash buckets:

- `FZ-20260421-0005`: 256-row lifted parent allocator assertion;
- `FZ-20260421-0008`: 2CTA indexed `ld.red` row/col chain optimizer abort;
- `FZ-20260421-0009`: 1CTA direct indexed `ld.red` allocator assertion.
