# Round 43 Allocation/Rematerialization Guardrail

Date: 2026-04-21 14:21 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Surface

This lane targeted checked-in runtime rows around TMEM allocation lifetime,
source initialization lifetime, copy/scales source rematerialization, dense
shared rematerialization for `warpx2` copy, and scaled-MMAv5 B-scale extra-user
rematerialization.

## Commands

Required rebuild:

```bash
make -j8
```

Result: build-tree ninja reported `no work to do`.

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(alloc_lifetime or allocation or source_initialization or rematerializes) and not reports and not resource'
```

Result: `17/1615` tests collected.

Runtime:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(alloc_lifetime or allocation or source_initialization or rematerializes) and not reports and not resource'
```

Result: `17 passed, 1598 deselected`.

## Classification

No compiler crash, false unsupported diagnostic, opcode absence, runtime
miscompile, unexpected pass/fail transition, allocation-lifetime drift, or new
independent `FZ-*` bucket was observed.
