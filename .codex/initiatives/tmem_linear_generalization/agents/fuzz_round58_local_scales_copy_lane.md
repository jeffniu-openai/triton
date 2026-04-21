# Round 58 Local: scales load/store and copy lane

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `8521122cf`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code was edited.

## Scope

This local lane covered scale TMEM load/store and `tcgen05.copy.warpx4`
scale-copy behavior in `python/test/gluon/test_tmem_runtime_matrix.py`,
separate from the active scaled-MMAv5 accumulator/tile worker lane. Coverage
included:

- scale direct load/store roundtrips;
- scale descriptor-view roundtrips;
- scale-copy shared-layout probes;
- noncanonical/rematerialized scale-copy layouts;
- shared-source subslice scale-copy rematerialization;
- TMEM descriptor-view scale-copy clean unsupported diagnostic;
- direct `warpx4` scale copy and two-CTA direct scale copy;
- scaled-MMAv5 helper paths that exercise scale copy setup, including
  `use_acc` and geometry sweeps.

## Required build

```bash
make -j8
```

Result: `ninja: no work to do`.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_scales or copy_scales or ldst_scales_direct_roundtrip or ldst_scales_descriptor_view_roundtrip'
```

Result: `41/1615` collected.

## Runtime split

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_scales or copy_scales or ldst_scales_direct_roundtrip or ldst_scales_descriptor_view_roundtrip'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_scales or copy_scales or ldst_scales_direct_roundtrip or ldst_scales_descriptor_view_roundtrip'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_scales or copy_scales or ldst_scales_direct_roundtrip or ldst_scales_descriptor_view_roundtrip'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_scales or copy_scales or ldst_scales_direct_roundtrip or ldst_scales_descriptor_view_roundtrip'
```

Result:

- group 1: `11 passed, 1604 deselected`
- group 2: `11 passed, 1604 deselected`
- group 3: `11 passed, 1604 deselected`
- group 4: `8 passed, 1607 deselected`
- aggregate: `41 passed`

## Classification

No compiler crash, verifier drift, false unsupported diagnostic,
clean-boundary drift, opcode absence, runtime miscompile, hang, or new
independent `FZ-*` bucket was found. Scale load/store, scale descriptor-view,
scale-copy `warpx4`, two-CTA scale-copy, rematerialization, and scaled-MMAv5
scale-copy helper paths stayed green or clean unsupported as expected.
