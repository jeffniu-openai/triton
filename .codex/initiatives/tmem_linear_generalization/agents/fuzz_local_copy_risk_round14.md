# Round 14 Local Copy Risk Selector

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
  -k '(cp_no_scales_warpx2 or cp_no_scales_twocta or cp_scales_warpx4 or cp_scales) and not reports'
```

Result: `135/1615` selected.

Runtime pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  --store-durations --durations-path /tmp/tmem_local_r14_copy_risk_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales_warpx2 or cp_no_scales_twocta or cp_scales_warpx4 or cp_scales) and not reports'
```

Results:

- group 1/GPU 0: `34 passed, 1581 deselected`;
- group 2/GPU 1: `34 passed, 1581 deselected`;
- group 3/GPU 2: `34 passed, 1581 deselected`;
- group 4/GPU 3: `33 passed, 1582 deselected`.

Aggregate: `135 passed`.

## Classification

No new bucket. The checked-in copy surface around no-scales `warpx2`,
two-CTA no-scales copies, scales `warpx4`, and scaled-MMA scale-copy setup
remained green.

This is a baseline next to Lane AI's temporary multi-region mbarrier
composition fuzzing for `FZ-20260421-0014`: the existing single-region and
descriptor-chain copy matrix passes, while the temporary minimization points
at multiple independent cross-CTA no-scales copy regions in one kernel.
