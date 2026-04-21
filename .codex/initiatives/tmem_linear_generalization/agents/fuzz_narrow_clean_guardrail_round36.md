# Round 36 Narrow-Shape And Clean-Diagnostic Guardrail

Date: 2026-04-21 13:40 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Run a focused checked-in Python/Gluon runtime guardrail around narrow TMEM
shapes and clean unsupported diagnostics while excluding known M64/resource
planner failures. The goal was to detect compiler crashes, verifier
over-strictness, false unsupported diagnostics, opcode drift, unexpected
runtime miscompiles, or clean-boundary regressions in a surface that has
previously exposed TMEM layout coverage gaps.

## Commands

```bash
make -j8

PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(x1 or n16 or n32 or clean_unsupported) and not reports and not m64 and not resource'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(x1 or n16 or n32 or clean_unsupported) and not reports and not m64 and not resource'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(x1 or n16 or n32 or clean_unsupported) and not reports and not m64 and not resource'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(x1 or n16 or n32 or clean_unsupported) and not reports and not m64 and not resource'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(x1 or n16 or n32 or clean_unsupported) and not reports and not m64 and not resource'
```

## Results

- Required rebuild: `make -j8` was a no-op.
- Collection: `329/1615`.
- Split-4 result:
  - group 1: `83 passed`
  - group 2: `19 passed, 64 skipped`
  - group 3: `77 passed, 6 skipped`
  - group 4: `80 passed`
- Aggregate: `259 passed, 70 skipped`.

## Classification

No new independent `FZ-*` bucket was found.

The selected rows remained stable after excluding the already-cataloged
M64/resource planner lane. There was no compiler crash, false unsupported
diagnostic, opcode absence, runtime miscompile, clean-boundary drift, or
unexpected xfail/pass transition. The skipped rows are existing clean/resource
or harness-gated cases selected by the broad `x1`/`n16`/`n32`/clean-diagnostic
keywords, not newly regressed execution semantics.

Backend repair remains deferred while the structural fuzzing campaign continues
to launch new discovery lanes.
