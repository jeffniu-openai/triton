# Round 14 Local Clean-Boundary Selector Sweep

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler fixes were attempted.

## Required rebuild

```bash
make -j8
```

Result: `ninja: no work to do`.

## Selector

Collect-only:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'reports_clean_unsupported or reports_clean_error or reports_tmem_oor'
```

Result: `157/1615` selected.

Runtime pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  --store-durations --durations-path /tmp/tmem_local_r14_clean_boundaries_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'reports_clean_unsupported or reports_clean_error or reports_tmem_oor'
```

Results:

- group 1/GPU 0: `40 passed, 1575 deselected`;
- group 2/GPU 1: `40 passed, 1575 deselected`;
- group 3/GPU 2: `40 passed, 1575 deselected`;
- group 4/GPU 3: `37 passed, 1578 deselected`.

Aggregate: `157 passed`.

## Classification

No new bucket. The clean unsupported/error/OOR diagnostic surface stayed
stable across ld/st higher-rank OOR rows, blocked layouts, scales descriptor
view boundaries, no-scales copy boundaries, high-CGA context diagnostics,
MMAv5 integer clean errors, row/column-permuted MMAv5/copy boundaries, and
scaled-MMAv5 clean diagnostics.

This sweep is a guardrail for the fuzzing campaign: none of the known clean
negative rows drifted into an unexpected compiler crash or runtime execution
path.
