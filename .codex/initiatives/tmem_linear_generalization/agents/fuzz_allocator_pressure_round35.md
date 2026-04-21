# Round 35 Local: Allocator-Pressure Runtime Guardrail

Date: 2026-04-21 14:05 UTC
Branch: `codex/tmem`
Scope: discovery-only guardrail run. No backend or checked-in test source was
modified.

## Objective

Exercise the audit-prioritized allocator-pressure surface covering co-live TMEM
allocations, LHS TMEM/subslice paths, scaled-MMAv5 accumulator views, shared
scale descriptor views, and narrow/tile-permuted accumulator layouts.

## Build And Collection

Required rebuild before this batch:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'alloc_lifetime or mma_scaled or lhs_subslice'
```

Result: `303/1615` rows collected.

## Runtime Sweep

Command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_allocator_pressure_round35_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'alloc_lifetime or mma_scaled or lhs_subslice'
```

Results:

- Group 1: `76 passed, 1539 deselected`
- Group 2: `76 passed, 1539 deselected`
- Group 3: `76 passed, 1539 deselected`
- Group 4: `75 passed, 1540 deselected`

Aggregate: `303 passed`.

## Classification

No compiler crash, false unsupported diagnostic, opcode mismatch, runtime
miscompile, clean-boundary drift, unexpected xfail/pass transition, or new
independent `FZ-*` bucket was found.

This keeps the checked-in allocator-pressure and scaled-MMAv5 positive runtime
surface stable before deeper temporary probes for dynamic B-scale descriptor
selection and co-live unrelated TMEM allocations.
