# Round 13 Local: Warp-Specialized TMEM Partitioning

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Repo edit scope: this report only.

## Scope

This local lane checked warp-specialization and partitioning coverage that
mentions TMEM allocations, descriptor views, copy, stores, and MMAv5 async
reference flow.

Coverage:

- Python runtime rows for TMEM copy/store inside `gl.warp_specialize`;
- lit partition scheduling for TMEM load/store/MMA users;
- tensor-memory allocation through warp-specialized captures and subviews;
- TMEM interleave pass coverage; and
- NVWS async-reference insertion for TMEM/MMAv5 load/store flows.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Python collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_fpsan.py python/test/gluon/test_core.py \
  -k 'warp_specialize and tmem'
```

Result: `2/18218` collected.

Python runtime:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_fpsan.py -k 'warp_specialize and tmem'
```

Result: `2 passed, 102 deselected in 2.37s`.

Compiler-only lit coverage:

```bash
cd build/cmake.linux-aarch64-cpython-3.12 && \
ninja triton-opt && \
lit -v \
  test/TritonGPU/partition-scheduling.mlir \
  test/TritonNvidiaGPU/test_tensor_memory_allocation.mlir \
  test/TritonNvidiaGPU/interleave_tmem.mlir \
  test/NVWS/aref-tmem-insertion.mlir
```

Result: `4/4` passed.

## Classification

No new `FZ-*` candidate was found.

Warp-specialized TMEM copy/store runtime visibility, partition scheduling,
allocation capture/subview handling, interleave behavior, and NVWS async
reference insertion stayed green in the selected coverage.
