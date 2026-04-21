# Round 23 Local: multi-CTA TMEM runtime guardrail

- Date: 2026-04-21 12:42 UTC
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler fixes attempted.
- Repo edit scope: this report only.

## Scope

This local slice ran a broad multi-CTA-oriented runtime selector over
`test_tmem_runtime_matrix.py`. It covered 2CTA ld/st layouts, descriptor
composition, plain MMAv5 accumulator views, scaled-MMAv5 2CTA accumulator
subslice copy rows, multicast variants, scales descriptor views, and CGA/CTA
layout rows that are not report-only clean negatives.

## Commands

Required build gate had already run for Round 23:

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
  -k '(twocta or multicast or cta or cga) and tmem and not reports'
```

Result:

```text
430/1615 tests collected (1185 deselected)
```

Split-4 runtime execution:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  --store-durations --durations-path /tmp/tmem_round23_cta_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(twocta or multicast or cta or cga) and tmem and not reports'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  --store-durations --durations-path /tmp/tmem_round23_cta_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(twocta or multicast or cta or cga) and tmem and not reports'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  --store-durations --durations-path /tmp/tmem_round23_cta_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(twocta or multicast or cta or cga) and tmem and not reports'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  --store-durations --durations-path /tmp/tmem_round23_cta_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(twocta or multicast or cta or cga) and tmem and not reports'
```

Result:

```text
group 1: 81 passed, 27 skipped, 1507 deselected
group 2: 98 passed, 10 skipped, 1507 deselected
group 3: 108 passed, 1507 deselected
group 4: 106 passed, 1509 deselected
aggregate: 393 passed, 37 skipped
```

## Classification

No new independent `FZ-*` bucket was found.

The green result is useful contrast for the existing multi-CTA buckets:

- `FZ-20260421-0010` remains focused on clean CTA-count/layout-gate
  diagnostics for instruction-local 1CTA/2CTA layouts in larger-CGA contexts,
  not ordinary checked-in 2CTA runtime execution.
- `FZ-20260421-0007` remains focused on dynamic selected accumulator views to
  scaled-MMAv5; checked-in 2CTA scaled-MMAv5 accumulator-subslice copy rows
  stayed green.
- `FZ-20260421-0001` dynamic generic descriptor SSA failures do not generalize
  to static 2CTA descriptor composition rows in this selector.
