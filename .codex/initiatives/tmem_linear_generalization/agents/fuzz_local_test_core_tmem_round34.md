# Round 34 Local: `test_core.py` TMEM/Multicast Guardrail

Date: 2026-04-21

## Scope

Reran a focused `python/test/gluon/test_core.py` TMEM surface outside the
runtime-matrix file. This selector covers MMAv5 multicast commit, 2CTA linear
accumulator multicast, direct scaled-MMAv5 multicast barrier, plain-kind MMAv5,
I8 clean diagnostics, physical-bitcast TMEM MMA, scaled-MMAv5 minimal kernels,
and scaled-MMAv5 copy/shared-input rows.

Selector:

```bash
-k '(tcgen05 or tmem) and (multicast or shared_inputs or twocta or mma)'
```

The selector collected `122/18114` rows.

## Commands

Required rebuild and collection:

```bash
make -j8
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_core.py \
  -k '(tcgen05 or tmem) and (multicast or shared_inputs or twocta or mma)'
```

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_core.py \
  -k '(tcgen05 or tmem) and (multicast or shared_inputs or twocta or mma)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_core.py \
  -k '(tcgen05 or tmem) and (multicast or shared_inputs or twocta or mma)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_core.py \
  -k '(tcgen05 or tmem) and (multicast or shared_inputs or twocta or mma)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_core.py \
  -k '(tcgen05 or tmem) and (multicast or shared_inputs or twocta or mma)'
```

## Result

- Group 1: `28 passed, 3 skipped, 18083 deselected`
- Group 2: `31 passed, 18083 deselected`
- Group 3: `31 passed, 18083 deselected`
- Group 4: `29 passed, 18085 deselected`
- Aggregate selected result: `119 passed, 3 skipped`

No compiler crash, false unsupported diagnostic, opcode mismatch, runtime
miscompile, clean-boundary drift, unexpected skip/pass transition, or new
independent `FZ-*` bucket was found.
