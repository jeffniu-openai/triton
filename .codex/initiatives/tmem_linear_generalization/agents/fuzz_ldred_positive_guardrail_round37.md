# Round 37 `ld.red` Positive Guardrail

Date: 2026-04-21 14:12 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Run a checked-in runtime guardrail over positive f32 `tcgen05.ld.red` surfaces
while excluding known M64/resource/report/non-f32 rows. The selected rows cover
identity layouts, explicit compatible variants, descriptor chains,
tile-permuted and row/column-permuted layouts, software-reduce fallback rows,
and single-CTA block layouts.

## Commands

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource and not m64 and not non_f32'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource and not m64 and not non_f32'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource and not m64 and not non_f32'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource and not m64 and not non_f32'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource and not m64 and not non_f32'
```

## Results

- Collection: `160/1615`.
- Split-4 result:
  - group 1: `40 passed`
  - group 2: `40 passed`
  - group 3: `40 passed`
  - group 4: `40 passed`
- Aggregate: `160 passed`.

## Classification

No new independent `FZ-*` bucket was found.

The selected positive f32 reduction rows remained green after excluding known
M64/resource and non-f32 boundary rows. There was no compiler crash, false
unsupported diagnostic, opcode absence, runtime miscompile, clean-boundary
drift, or unexpected xfail/pass transition.

Backend repair remains deferred while discovery lanes continue.
