# Round 25 Local: Scale-Copy Runtime Baseline

Date: 2026-04-21 14:18 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

This local slice covered checked-in `tcgen05.cp` scale-copy runtime rows while
Round 25 subagents fuzzed dynamic descriptor generator seeds, verifier
boundaries, and high-CGA mixed ownership.

Selector:

```text
cp_scales and not reports
```

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_scales and not reports'
```

Result:

```text
33/1615 tests collected
```

Split-4 runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <1..4> \
  --store-durations --durations-path /tmp/tmem_r25_cpscales_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_scales and not reports'
```

## Results

Aggregate result: `33 passed`.

Per-shard results:

- GPU 0 / group 1: `9 passed, 1606 deselected`.
- GPU 1 / group 2: `9 passed, 1606 deselected`.
- GPU 2 / group 3: `9 passed, 1606 deselected`.
- GPU 3 / group 4: `6 passed, 1609 deselected`.

Durations:

```text
/tmp/tmem_r25_cpscales_durations.json
```

## Classification

No runtime miscompile, compiler crash, unexpected unsupported diagnostic, or
new independent `FZ-*` bucket was found.
