# Round 60 Lane B: TMEM lit compiler breadth

Date: 2026-04-21 16:15 UTC
Branch: `codex/tmem`
HEAD at start: `b5d0cd781cd147badf58c178e627ddde0e4f5fe1`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code
was edited.

## Scope

This lane focused on compiler-only MLIR/lit coverage around TMEM allocation,
lowering, conversion, scheduling, warp-specialization, and invalid diagnostics
that was not included in the Round 59 clean-negative or TMA/proxy lanes.

Avoided exact Round 59 lit files:

- `test/Conversion/lower_tensor_memory_to_llvm.mlir`
- `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
- `test/Conversion/relayout_tritongpu.mlir`
- `test/TritonGPU/promote-lhs-to-tmem.mlir`
- `test/TritonGPU/hoist-tmem-alloc.mlir`
- `test/TritonGPU/memdesc-subview-split.mlir`
- `test/TritonNvidiaGPU/membar-cluster.mlir`
- `test/TritonNvidiaGPU/invalid.mlir`
- `test/Conversion/tma_to_llvm.mlir`
- `test/TritonGPU/loop-pipeline-hopper.mlir`
- `test/NVWS/lower_aref.mlir`

## Required build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## triton-opt build

The first lit sweep built `triton-opt` as requested:

```bash
cd build/cmake.linux-aarch64-cpython-3.12 && ninja triton-opt
```

Result: `ninja: no work to do`.

## Lit sweep 1: allocation, promotion, invalid diagnostics, NVWS

Command:

```bash
cd build/cmake.linux-aarch64-cpython-3.12 && ninja triton-opt && lit -v \
  test/TritonNvidiaGPU/test_tensor_memory_allocation.mlir \
  test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir \
  test/TritonNvidiaGPU/mma_lowering.mlir \
  test/TritonNvidiaGPU/canonicalize.mlir \
  test/TritonGPU/loop-pipeline-blackwell.mlir \
  test/TritonGPU/automatic-warp-specialization.mlir \
  test/TritonGPU/invalid.mlir \
  test/Conversion/tritonnvidiagpu_to_llvm.mlir \
  test/Conversion/warp_specialize_to_llvm.mlir \
  test/Conversion/nvgpu_invalid.mlir \
  test/NVWS/hoist_tmem_store.mlir
```

Result:

```text
Total Discovered Tests: 11
  Passed: 11 (100.00%)
```

Coverage notes:

- TMEM allocation offsets, reuse, scale allocation, partitioned allocation,
  and memdesc-indexed allocator cases passed.
- Promotion to tensor memory and MMAv5 lowering checks passed.
- Blackwell loop pipeline TMEM token removal, automatic warp specialization,
  and NVWS TMEM store hoisting passed.
- TritonGPU TMEM linear-layout verifier diagnostics and NVGPU invalid
  conversion diagnostics passed.
- Generic NvidiaGPU and warp-specialize conversion files passed.

## Lit sweep 2: TritonGPU scheduling and generic conversion breadth

Command:

```bash
cd build/cmake.linux-aarch64-cpython-3.12 && lit -v \
  test/TritonGPU/accelerate-matmul.mlir \
  test/TritonGPU/canonicalize.mlir \
  test/TritonGPU/fence-inserstion.mlir \
  test/TritonGPU/optimize-partition-warps.mlir \
  test/TritonGPU/partition-scheduling.mlir \
  test/TritonGPU/pipeline-assign-latencies.mlir \
  test/TritonGPU/pipeline-lower-loop.mlir \
  test/TritonGPU/pipeline-schedule-loop.mlir \
  test/Conversion/tritongpu_to_llvm.mlir \
  test/Conversion/tritongpu_to_llvm_hopper.mlir \
  test/Conversion/allocate_warp_groups.mlir \
  test/Conversion/nvgpu_to_llvm.mlir
```

Result:

```text
Total Discovered Tests: 12
  Passed: 12 (100.00%)
```

Coverage notes:

- TMEM-bearing matmul acceleration, canonicalization, partition scheduling,
  latency assignment, loop lowering, and loop scheduling lit checks passed.
- Conversion breadth around generic TritonGPU-to-LLVM, Hopper conversion,
  warp-group allocation, and NVGPU-to-LLVM passed.

## Aggregate result

- Required build: `make -j8` no-op.
- `triton-opt` build: no-op.
- Lit files run: `23`.
- Lit result: `23 passed, 0 failed`.

## Classification

No new independent `FZ-*` was found.

No existing failing bucket was reproduced in this lane. In particular:

- Existing `FZ-20260421-0016` from `Conversion/relayout_tritongpu.mlir` was
  not exercised because this lane avoided the exact Round 59 clean-negative
  lit file set.
- Existing TMA/proxy/mbarrier buckets were not exercised by the exact Round 59
  lane C lit files.
- No compiler crash, verifier drift, invalid-diagnostic drift, PassManager
  failure, TMEM allocation/lifetime lit failure, conversion failure, or NVWS
  TMEM hoisting failure was observed.

Backend/compiler repairs remain deferred.
