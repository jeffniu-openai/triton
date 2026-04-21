# Round 44 Proxy/TMA Core Guardrail

Date: 2026-04-21 14:30 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Surface

This lane targeted a compact checked-in `test_core.py` proxy/TMA/mbarrier
surface: TMA multicast copy, MMAv5 multicast commit, two-CTA linear
accumulator multicast commit, and async-copy mbarrier controls.

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py \
  -k '(tma_multicast or multicast_commit or tcgen05_commit or mbarrier or proxy) and not reports and not resource'
```

Result: `12/19729` tests collected.

Runtime:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py \
  -k '(tma_multicast or multicast_commit or tcgen05_commit or mbarrier or proxy) and not reports and not resource'
```

Result: `12 passed, 19717 deselected`.

## Classification

No compiler crash, false unsupported diagnostic, opcode absence, runtime
miscompile, proxy/mbarrier diagnostic drift, or new independent `FZ-*` bucket
was observed.
