# Round 34 Local: Copy warpx2/warpx4 Runtime Surface Guardrail

Date: 2026-04-21
Branch: `codex/tmem`

## Scope

Reran the checked-in no-scale `tcgen05.copy` `warpx2`/adjacent surface after
the Round 33 compiler-boundary and FZ-0018 PTX artifact work. This selector
keeps both positive and clean diagnostic rows in scope while excluding resource
OOR rows:

- single-CTA `warpx2::{01_23,02_13}` direct candidate rows;
- subslice, indexed, and slice-index descriptor-view rows;
- two-CTA `warpx2::01_23` positive rows;
- two-CTA `warpx2::02_13` clean unsupported rows;
- subword dtype clean diagnostics;
- dense-shared rematerialization rows;
- two-CTA `128x128b` codegen rows.

No backend or test source was edited.

## Commands

Required rebuild:

```bash
make -j8
```

Result: no-op rebuild.

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and (warpx2 or warpx4 or dense_shared or twocta_128x128b) and not resource)'
```

Result: `83/1615` rows collected.

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and (warpx2 or warpx4 or dense_shared or twocta_128x128b) and not resource)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and (warpx2 or warpx4 or dense_shared or twocta_128x128b) and not resource)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and (warpx2 or warpx4 or dense_shared or twocta_128x128b) and not resource)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and (warpx2 or warpx4 or dense_shared or twocta_128x128b) and not resource)'
```

## Result

- Group 1: `21 passed, 1594 deselected`
- Group 2: `21 passed, 1594 deselected`
- Group 3: `21 passed, 1594 deselected`
- Group 4: `20 passed, 1595 deselected`
- Aggregate selected result: `83 passed`

No compiler crash, false unsupported diagnostic, opcode mismatch, runtime
miscompile, clean-boundary drift, unexpected xfail/pass transition, or new
independent `FZ-*` bucket was found.

## Classification

This is a green guardrail. It confirms that the checked-in no-scale copy
`warpx2`/dense-shared/two-CTA surface is stable after the Round 33 fuzzing
checkpoint and that the clean unsupported `twocta_02_13` rows still diagnose
before runtime execution.
