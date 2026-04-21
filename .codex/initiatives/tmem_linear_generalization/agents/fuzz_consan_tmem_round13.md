# Round 13 Local: ConSan TMEM Instrumentation Sanity

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Repo edit scope: this report only.

## Scope

This local lane checked existing ConSan runtime coverage that touches TMEM
operations. It was chosen as a non-overlapping instrumentation contrast after
the FPSAN runtime-index MMAv5 lane.

Coverage:

- 1CTA, 2CTA, and 4CTA `tcgen05_mma` cases with TMEM load/store access kinds;
- ConSan pass/fail variants for TMEM load/store race detection;
- `tcgen05_copy` cases with TMEM load instrumentation;
- TMA plus `tcgen05_mma` multicast loops in 1CTA, 2CTA, and 4CTA launch
  contexts.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_consan.py -k 'tmem or tcgen05 or tensor_memory'
```

Result: `36/348` collected.

Split-4 runtime command:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  python/test/gluon/test_consan.py -k 'tmem or tcgen05 or tensor_memory'
```

Results:

```text
group 1/GPU 0: 5 passed, 4 skipped
group 2/GPU 1: 9 passed
group 3/GPU 2: 9 passed
group 4/GPU 3: 7 passed, 2 skipped
aggregate:     30 passed, 6 skipped
```

## Classification

No new `FZ-*` candidate was found.

The checked-in ConSan TMEM coverage remains green for the tested MMAv5, copy,
TMA, multicast, and TMEM load/store access patterns. This is a negative result
only and does not expand the FPSAN-specific `FZ-20260421-0011` bucket.
