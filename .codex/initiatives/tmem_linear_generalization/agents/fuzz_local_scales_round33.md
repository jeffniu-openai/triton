# Round 33 Local: Scale-Bearing Runtime Guardrail

Date: 2026-04-21

## Scope

Reran the checked-in scale-bearing TMEM runtime surface after Round 32. This
slice intentionally excludes no-scale copy rows and clean/resource diagnostics
so it focuses on:

- scale `ld/st` direct and descriptor-view round trips,
- scale `ld.red` software-reduce fallbacks,
- `tcgen05.cp` scale-layout probes including `warpx2` candidates,
- `tcgen05.cp` scales `warpx4` direct copy, and
- scale-copy paths exercised through scaled-MMAv5 matrix cases.

Selector:

```bash
-k '((cp_scales or ldst_scales or ld_red_scales) and not cp_no_scales and not reports and not clean and not resource)'
```

The selector collected `66/1615` runtime rows.

## Commands

Required rebuild and collection:

```bash
make -j8
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '((cp_scales or ldst_scales or ld_red_scales) and not cp_no_scales and not reports and not clean and not resource)'
```

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '((cp_scales or ldst_scales or ld_red_scales) and not cp_no_scales and not reports and not clean and not resource)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '((cp_scales or ldst_scales or ld_red_scales) and not cp_no_scales and not reports and not clean and not resource)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '((cp_scales or ldst_scales or ld_red_scales) and not cp_no_scales and not reports and not clean and not resource)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '((cp_scales or ldst_scales or ld_red_scales) and not cp_no_scales and not reports and not clean and not resource)'
```

## Result

- Group 1: `17 passed, 1598 deselected`
- Group 2: `17 passed, 1598 deselected`
- Group 3: `17 passed, 1598 deselected`
- Group 4: `15 passed, 1600 deselected`
- Aggregate selected result: `66 passed`

No compiler crash, false unsupported diagnostic, opcode mismatch, runtime
miscompile, or new independent `FZ-*` bucket was found.
