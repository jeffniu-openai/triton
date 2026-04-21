# Round 42 Core TMA/MMAv5 Guardrail

Date: 2026-04-21 14:18 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Surface

This lane targeted checked-in `python/test/gluon/test_core.py` TMA-fed MMAv5
shared-input rows plus plain `tcgen05.mma` kind rows, while excluding scaled
and multicast rows. The selector was broader than a smoke test because
`tma_mma_shared_inputs` expands over transpose, gather/scatter, CGA, repetition,
and warp-count axes.

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_core.py \
  -k '(tcgen05_mma or tma_mma_shared_inputs) and not scaled and not multicast'
```

Result: `223/18114` tests collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_core.py -k '(tcgen05_mma or tma_mma_shared_inputs) and not scaled and not multicast'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_core.py -k '(tcgen05_mma or tma_mma_shared_inputs) and not scaled and not multicast'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_core.py -k '(tcgen05_mma or tma_mma_shared_inputs) and not scaled and not multicast'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_core.py -k '(tcgen05_mma or tma_mma_shared_inputs) and not scaled and not multicast'
```

## Results

- Group 1: `47 passed, 9 skipped, 18058 deselected`
- Group 2: `47 passed, 9 skipped, 18058 deselected`
- Group 3: `41 passed, 15 skipped, 18058 deselected`
- Group 4: `37 passed, 18 skipped, 18059 deselected`

Aggregate: `172 passed, 51 skipped`.

## Classification

No compiler crash, false unsupported diagnostic, opcode absence, runtime
miscompile, unexpected pass/fail transition, or new independent `FZ-*` bucket
was observed. The selected core TMA-fed MMAv5 and plain MMAv5 kind rows remain
stable under current runtime skip conditions.
