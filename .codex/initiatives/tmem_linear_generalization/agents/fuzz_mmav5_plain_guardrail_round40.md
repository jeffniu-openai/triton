# Round 40 Plain MMAv5 Runtime Guardrail

Date: 2026-04-21 14:12 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Surface

This lane targeted checked-in runtime rows for plain MMAv5 format coverage and
use-accumulator variants, including one-CTA and two-CTA accumulator layouts:

- `test_tmem_runtime_matrix_mma_plain_kinds_with_linear_acc`
- `test_tmem_runtime_matrix_mma_plain_kinds_use_acc`
- `test_tmem_runtime_matrix_mma_twocta_plain_kinds`
- `test_tmem_runtime_matrix_mma_twocta_plain_kinds_use_acc`

It is intentionally separate from the concurrent Round 40 proxy/TMA/mbarrier
and copy/dynamic-descriptor subagent lanes.

## Commands

Required rebuild before this batch:

```bash
make -j8
```

Result: build-tree ninja reported `no work to do`.

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py \
  -k '(mma_plain_kinds_use_acc or mma_plain_kinds_with_linear_acc or mma_twocta_plain_kinds_use_acc or mma_twocta_plain_kinds) and not scaled and not reports and not resource'
```

Result: `52/19729` tests collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py -k '(mma_plain_kinds_use_acc or mma_plain_kinds_with_linear_acc or mma_twocta_plain_kinds_use_acc or mma_twocta_plain_kinds) and not scaled and not reports and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py -k '(mma_plain_kinds_use_acc or mma_plain_kinds_with_linear_acc or mma_twocta_plain_kinds_use_acc or mma_twocta_plain_kinds) and not scaled and not reports and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py -k '(mma_plain_kinds_use_acc or mma_plain_kinds_with_linear_acc or mma_twocta_plain_kinds_use_acc or mma_twocta_plain_kinds) and not scaled and not reports and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py -k '(mma_plain_kinds_use_acc or mma_plain_kinds_with_linear_acc or mma_twocta_plain_kinds_use_acc or mma_twocta_plain_kinds) and not scaled and not reports and not resource'
```

## Results

- Group 1: `13 passed, 19716 deselected`
- Group 2: `13 passed, 19716 deselected`
- Group 3: `13 passed, 19716 deselected`
- Group 4: `13 passed, 19716 deselected`

Aggregate: `52 passed`.

## Classification

No compiler crash, false unsupported diagnostic, opcode absence, runtime
miscompile, unexpected pass/fail transition, or new independent `FZ-*` bucket
was observed. This keeps the plain MMAv5 format and use-accumulator rows stable
while the fuzzing campaign continues probing descriptor/dynamic and proxy/TMA
surfaces in parallel.
