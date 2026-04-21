# Round 59 Lane C: TMA/TMEM proxy and mbarrier sequencing

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `8ee469bd0`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code was edited.

## Scope

This lane focused on checked-in TMA/TMEM integration and proxy/mbarrier
sequencing not covered by the high-CGA ownership lane. Coverage included:

- TMA gather/scatter with multicast and non-multicast variants;
- TMA-loaded shared inputs feeding `tcgen05_mma` / TMEM accumulator paths;
- multicast and non-multicast `tcgen05.commit` / mbarrier sequencing;
- scaled direct multicast barrier sequencing;
- device-side TMA load/store controls;
- runtime-matrix two-CTA MMAv5/TMA descriptor paths with indexed/subslice
  TMEM accumulator views;
- scaled-MMAv5 copy helper paths that assert commit opcode and multicast
  metadata behavior;
- lit coverage around TMA gather/scatter lowering, mbarrier init/fence
  insertion, invalid TMA diagnostics, pipelined proxy fences, and NVWS TMA/TMEM
  lowering.

## Required build

```bash
make -j8
```

Result: `ninja: no work to do`.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_core.py \
  -k 'tma_gather_scatter or tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tcgen05_mma_scaled_direct_multicast_barrier or device_tma_load or device_tma_store'
```

Result: `231/18114` collected.

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_scales_warpx4_via_scaled_mma or mma_scaled_twocta_acc_subslice or mma_twocta_tma_tf32_b_transposed or mma_twocta_indexed_acc or mma_twocta_acc_subslice'
```

Result: `104/1615` collected.

## Runtime split

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_core.py -k 'tma_gather_scatter or tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tcgen05_mma_scaled_direct_multicast_barrier or device_tma_load or device_tma_store'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_core.py -k 'tma_gather_scatter or tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tcgen05_mma_scaled_direct_multicast_barrier or device_tma_load or device_tma_store'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_core.py -k 'tma_gather_scatter or tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tcgen05_mma_scaled_direct_multicast_barrier or device_tma_load or device_tma_store'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_core.py -k 'tma_gather_scatter or tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tcgen05_mma_scaled_direct_multicast_barrier or device_tma_load or device_tma_store'
```

Result:

- group 1: `49 passed, 9 skipped, 18056 deselected`
- group 2: `49 passed, 9 skipped, 18056 deselected`
- group 3: `44 passed, 14 skipped, 18056 deselected`
- group 4: `41 passed, 16 skipped, 18057 deselected`
- aggregate: `183 passed, 48 skipped`

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_scales_warpx4_via_scaled_mma or mma_scaled_twocta_acc_subslice or mma_twocta_tma_tf32_b_transposed or mma_twocta_indexed_acc or mma_twocta_acc_subslice'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_scales_warpx4_via_scaled_mma or mma_scaled_twocta_acc_subslice or mma_twocta_tma_tf32_b_transposed or mma_twocta_indexed_acc or mma_twocta_acc_subslice'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_scales_warpx4_via_scaled_mma or mma_scaled_twocta_acc_subslice or mma_twocta_tma_tf32_b_transposed or mma_twocta_indexed_acc or mma_twocta_acc_subslice'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_scales_warpx4_via_scaled_mma or mma_scaled_twocta_acc_subslice or mma_twocta_tma_tf32_b_transposed or mma_twocta_indexed_acc or mma_twocta_acc_subslice'
```

Result:

- group 1: `26 passed, 1589 deselected`
- group 2: `26 passed, 1589 deselected`
- group 3: `26 passed, 1589 deselected`
- group 4: `26 passed, 1589 deselected`
- aggregate: `104 passed`

## Lit probes

```bash
cd build/cmake.linux-aarch64-cpython-3.12 && \
  ninja triton-opt && \
  lit -v \
    test/TritonNvidiaGPU/membar-cluster.mlir \
    test/TritonNvidiaGPU/invalid.mlir \
    test/Conversion/tma_to_llvm.mlir \
    test/TritonGPU/loop-pipeline-hopper.mlir \
    test/NVWS/lower_aref.mlir
```

Result: `5` tests discovered, `5 passed`; `ninja triton-opt` reported
`no work to do`.

## Classification

No compiler crash, verifier drift, false unsupported diagnostic, proxy-fence
insertion failure, opcode/commit mismatch, runtime miscompile, hang, or new
independent `FZ-*` bucket was found.

Existing `FZ-20260421-0014` / `FZ-0014` proxy-fence insertion behavior was not
reproduced by this checked-in TMA/TMEM lane. The passing rows act as green
contrasts for TMA gather/scatter, TMA-to-MMAv5 shared-input sequencing,
multicast and non-multicast commit/mbarrier paths, and two-CTA TMA descriptor
accumulator flows.
