# Round 31 Local: Scales Copy and Clean-Boundary Guardrail

Date: 2026-04-21

## Scope

This local slice reran scaled-copy and clean-boundary runtime rows adjacent to
the Round 31 scaled operand and compiler-boundary fuzzing lanes.

Selector:

```bash
-k '(cp_scales or mbarrier or proxy or clean_error or clean_unsupported) and not reports and not resource'
```

The selector collected `55/1615` rows, including:

- `tcgen05.copy` scales layout probes;
- scales `warpx4` direct copy and scaled-MMAv5 copy-matrix rows;
- no-scales `warpx2` subword clean-error diagnostics;
- `ld/st` unsupported `x1` clean-unsupported diagnostics.

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_scales or mbarrier or proxy or clean_error or clean_unsupported) and not reports and not resource'
```

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_scales or mbarrier or proxy or clean_error or clean_unsupported) and not reports and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_scales or mbarrier or proxy or clean_error or clean_unsupported) and not reports and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_scales or mbarrier or proxy or clean_error or clean_unsupported) and not reports and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_scales or mbarrier or proxy or clean_error or clean_unsupported) and not reports and not resource'
```

## Result

- Group 1: `14 passed, 1601 deselected`
- Group 2: `14 passed, 1601 deselected`
- Group 3: `14 passed, 1601 deselected`
- Group 4: `13 passed, 1602 deselected`
- Aggregate selected result: `55 passed`

No compiler crash, false unsupported diagnostic, clean-diagnostic regression,
opcode mismatch, or runtime miscompile was found. No new `FZ-*` bucket is
needed.

