# Round 19 Local: 2CTA TMA and multicast sanity selectors

- Date: 2026-04-21 12:05 UTC
- Branch: `codex/tmem`
- Commit at start: `a80254d43`
- Mode: discovery/validation only. No backend/compiler fixes were attempted.

## Scope

This local slice checked two small non-overlapping runtime surfaces while
Round 19 subagents probed reproducer minimization and dynamic subslice/mixed
consumer paths:

- 2CTA TMA-transposed descriptor rows in `test_tmem_runtime_matrix.py`;
- `test_core.py` tcgen05 MMA multicast commit rows and scaled direct multicast
  barrier coverage.

## Required Rebuild

Command:

```bash
make -j8
```

Result: no work to do.

## Runtime Matrix 2CTA TMA Descriptor Rows

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma) and not reports_clean'
```

Result: `10/1615` selected.

Runtime:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma) and not reports_clean'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `3 passed, 1612 deselected in 5.88s` |
| 2 | 1 | `3 passed, 1612 deselected in 4.22s` |
| 3 | 2 | `3 passed, 1612 deselected in 4.20s` |
| 4 | 3 | `1 passed, 1614 deselected in 4.24s` |

Aggregate selected rows: `10 passed`.

## test_core Multicast Rows

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_core.py \
  -k 'multicast and (tcgen05 or tma_mma or mma_scaled)'
```

Result: `9/18114` selected.

Runtime:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  python/test/gluon/test_core.py \
  -k 'multicast and (tcgen05 or tma_mma or mma_scaled)'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `3 passed, 18111 deselected in 4.55s` |
| 2 | 1 | `3 passed, 18111 deselected in 4.49s` |
| 3 | 2 | `3 passed, 18111 deselected in 4.56s` |
| 4 | 3 | empty shard, `18114 deselected`, pytest exit code 5 |

Aggregate selected rows that ran: `9 passed`.

The group-4 exit code is an empty-shard artifact from splitting nine selected
rows into four groups, not a test failure.

## Classification

No new failures were observed in this slice.

- New independent `FZ-*`: none.
- Runtime miscompile: none observed.
- Compiler crash: none observed.
- Unexpected unsupported diagnostic: none observed.

This is a small current-head sanity baseline for 2CTA TMA descriptor and
multicast commit/barrier behavior while Round 19 focuses on deeper adversarial
descriptor and proxy-fence probes.
