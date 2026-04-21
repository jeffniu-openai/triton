# Round 24 Local: Clean-Boundary Runtime Sweep

Date: 2026-04-21 13:40 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

This local sweep reran the runtime-matrix clean diagnostic and out-of-resource
boundaries while Round 24 subagents fuzzed the new verifier crash and mixed
scaled-MMAv5 operand cases.

Selector:

```text
reports_clean or reports_tmem_oor or clean_unsupported or clean_error
```

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'reports_clean or reports_tmem_oor or clean_unsupported or clean_error'
```

Result:

```text
184/1615 tests collected
```

Split-4 runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <1..4> \
  --store-durations --durations-path /tmp/tmem_r24_clean_boundary_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'reports_clean or reports_tmem_oor or clean_unsupported or clean_error'
```

## Results

Aggregate result: `184 passed`.

Per-shard results:

- GPU 0 / group 1: `46 passed, 1569 deselected`.
- GPU 1 / group 2: `46 passed, 1569 deselected`.
- GPU 2 / group 3: `46 passed, 1569 deselected`.
- GPU 3 / group 4: `46 passed, 1569 deselected`.

Durations:

```text
/tmp/tmem_r24_clean_boundary_durations.json
```

## Classification

No unexpected failure, unexpected pass, compiler crash, or new independent
`FZ-*` bucket was found. Clean diagnostic boundaries remain stable.
