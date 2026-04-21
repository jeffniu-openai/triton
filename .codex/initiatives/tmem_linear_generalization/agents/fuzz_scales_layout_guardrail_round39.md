# Round 39 Scale-Layout Guardrail

Date: 2026-04-21 15:24 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Run a focused checked-in runtime guardrail over TMEM scale layout load/store
variants and `cp.scales` layout probes. This samples scale-layout lowering
outside the larger scaled-MMAv5 matrix.

## Commands

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(scales_variant or scales_layout or cp_scales_layout_probe or scales_ldst) and not reports and not resource'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(scales_variant or scales_layout or cp_scales_layout_probe or scales_ldst) and not reports and not resource'
```

## Results

- Collection: `19/1615`.
- Runtime result: `19 passed`.

## Classification

No new independent `FZ-*` bucket was found.

The selected scale-layout `ld/st` and `cp.scales` layout-probe rows stayed
green. There was no opcode drift, false unsupported diagnostic, runtime
miscompile, compiler crash, or unexpected pass/fail transition.

Backend repair remains deferred while discovery lanes continue.
