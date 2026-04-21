# Round 25 Local: No-Scale Copy Runtime Baseline

Date: 2026-04-21 14:10 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

This local slice covered broad checked-in no-scale `tcgen05.cp` runtime rows
while Round 25 subagents fuzzed dynamic descriptor generation and high-CGA
mixed ownership. It excluded report-only diagnostics, high-CGA rows,
selector-permuted rows, and subword rows.

Selector:

```text
cp_no_scales and not reports and not high and not cga and not selector and not subword
```

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales and not reports and not high and not cga and not selector and not subword'
```

Result:

```text
201/1615 tests collected
```

Split-4 runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <1..4> \
  --store-durations --durations-path /tmp/tmem_r25_copy_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales and not reports and not high and not cga and not selector and not subword'
```

## Results

Aggregate result: `197 passed, 4 skipped`.

Per-shard results:

- GPU 0 / group 1: `47 passed, 4 skipped, 1564 deselected`.
- GPU 1 / group 2: `51 passed, 1564 deselected`.
- GPU 2 / group 3: `51 passed, 1564 deselected`.
- GPU 3 / group 4: `48 passed, 1567 deselected`.

Durations:

```text
/tmp/tmem_r25_copy_durations.json
```

## Classification

No runtime miscompile, compiler crash, unexpected unsupported diagnostic, or
new independent `FZ-*` bucket was found.

This is a green no-scale copy baseline, separate from the active Round 25
dynamic descriptor and high-CGA mixed ownership fuzz lanes.
