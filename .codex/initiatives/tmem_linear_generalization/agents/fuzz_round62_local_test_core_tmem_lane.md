# Round 62 Local: test_core.py TMEM slice

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `9a6a41a89`
Mode: discovery/cataloging only. No backend, compiler, runtime, or checked-in
test source was edited.

## Scope

This lane sampled `python/test/gluon/test_core.py` TMEM-bearing rows outside
the runtime-matrix and examples harnesses. The selector covered copy,
load/store, MMAv5, `ld.red`, and `tcgen05` names.

## Required build

The required `make -j8` had already run for the Round 62 local examples lane
immediately before this probe and reported `ninja: no work to do`.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_core.py \
  -k 'tmem and (copy or ld or load or store or mma or tcgen05)'
```

Result: `55/18114 tests collected`.

## Runtime split

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_core.py -k 'tmem and (copy or ld or load or store or mma or tcgen05)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_core.py -k 'tmem and (copy or ld or load or store or mma or tcgen05)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_core.py -k 'tmem and (copy or ld or load or store or mma or tcgen05)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_core.py -k 'tmem and (copy or ld or load or store or mma or tcgen05)'
```

Results:

- group 1: `14 passed, 18100 deselected`
- group 2: `14 passed, 18100 deselected`
- group 3: `14 passed, 18100 deselected`
- group 4: `8 passed, 5 skipped, 18101 deselected`
- aggregate: `50 passed, 5 skipped`

Skip reason rerun:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s -rs --tb=short --splits 4 --group 4 python/test/gluon/test_core.py -k 'tmem and (copy or ld or load or store or mma or tcgen05)'
```

All five skips were existing OOR TMEM copy matrix guards:

```text
Skipping OOR TMEM copy matrix case M=256, N=256, BLOCK_N=16, swizzle=32
Skipping OOR TMEM copy matrix case M=256, N=256, BLOCK_N=32, swizzle=32
Skipping OOR TMEM copy matrix case M=256, N=256, BLOCK_N=64, swizzle=32
Skipping OOR TMEM copy matrix case M=256, N=256, BLOCK_N=128, swizzle=32
Skipping OOR TMEM copy matrix case M=256, N=256, BLOCK_N=256, swizzle=32
```

## Classification

No compiler crash, verifier drift, false unsupported diagnostic, runtime
miscompile, hang, unexpected skip/fail transition, or new independent `FZ-*`
bucket was found. The only skips were existing OOR copy-matrix guards. No
repairs were attempted.
