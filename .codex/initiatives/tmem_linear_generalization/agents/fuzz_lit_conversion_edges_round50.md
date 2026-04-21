# Round 50: Lit Conversion Edge Fuzzing

Date: 2026-04-21 14:51 UTC

Branch: `codex/tmem`

HEAD: `45968ab4584e`

Scope: compiler-only discovery/classification. No backend code, tests, or
central initiative documents were modified by this lane. This report is the
only checked-in file written for the lane.

## Objective

Focus on TMEM conversion and lowering edges:

- `Conversion` lit tests around TMEM, NVGPU, TMA, warp-specialization, and
  TritonNvidiaGPU lowering;
- `TritonNvidiaGPU` TMEM allocation/layout/interleave/membar lowering tests;
- verifier/assertion boundaries for unencoded tensors, relayout, memdesc
  subview/split, and `lower_tensor_memory_to_llvm`;
- saved MLIR reproducers for known `FZ-20260421-0014`,
  `FZ-20260421-0016`, and `FZ-20260421-0017`.

This lane catalogs and diagnoses only. It does not fix backend bugs.

## Required Build

Command:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

Tool build:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12 && ninja triton-opt
```

Result:

```text
ninja: no work to do.
```

## Green TMEM Conversion and Lowering Guardrail

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12 && \
  lit -v \
    test/Conversion/lower_tensor_memory_to_llvm.mlir \
    test/Conversion/tritonnvidiagpu_to_llvm.mlir \
    test/Conversion/nvgpu_to_llvm.mlir \
    test/TritonGPU/memdesc-subview-split.mlir \
    test/TritonNvidiaGPU/tmem_layouts.mlir \
    test/TritonNvidiaGPU/interleave_tmem.mlir \
    test/TritonNvidiaGPU/test_tensor_memory_allocation.mlir \
    test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir \
    test/TritonNvidiaGPU/membar-cluster.mlir \
    test/TritonNvidiaGPU/mma_lowering.mlir \
    test/TritonNvidiaGPU/tma_lowering.mlir
```

Result:

```text
Total Discovered Tests: 11
  Passed: 11 (100.00%)
```

Coverage notes:

- `lower_tensor_memory_to_llvm.mlir` stayed green.
- `memdesc-subview-split.mlir` stayed green.
- TMEM layout/interleave/allocation/promotion tests stayed green.
- TMA, MMA, and cluster membar TMEM-adjacent lowering tests stayed green.

No FileCheck drift, verifier drift, or new clean-negative gap was observed in
this group.

## Broader Conversion Lit Guardrail

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12 && \
  lit -v \
    test/Conversion/lower_tensor_memory_to_llvm.mlir \
    test/Conversion/tritongpu_to_llvm_blackwell.mlir \
    test/Conversion/tritongpu_to_llvm.mlir \
    test/Conversion/tritonnvidiagpu_to_llvm.mlir \
    test/Conversion/nvgpu_to_llvm.mlir \
    test/Conversion/tma_to_llvm.mlir \
    test/Conversion/tma_multicast_to_llvm.mlir \
    test/Conversion/warp_specialize_to_llvm.mlir
```

Result:

```text
Total Discovered Tests: 8
  Passed: 8 (100.00%)
```

This keeps the adjacent conversion surface green for Blackwell, NVGPU, TMA,
TMA multicast, warp-specialization, and TritonNvidiaGPU-to-LLVM lowering.

## Known-Red Relayout Boundary

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12 && \
  lit -v test/Conversion/relayout_tritongpu.mlir
```

Result:

```text
Total Discovered Tests: 1
  Failed: 1 (100.00%)
```

Failure signature:

```text
triton-opt: .../llvm/Support/Casting.h:650:
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
...
mlir::triton::gpu::TritonGPUDialect::toLinearLayout(...)
mlir::triton::nvidia_gpu::computeTMemLdStEncodingInfoImpl(...)
mlir::triton::nvidia_gpu::verifyTMEMOperand(...)
mlir::triton::nvidia_gpu::TMEMAllocOp::verify()
FileCheck error: '<stdin>' is empty.
```

Classification: existing `FZ-20260421-0016`. This is the known verifier/parser
boundary where an unencoded or otherwise non-distributed tensor layout reaches
TMEM operand verification and asserts through `toLinearEncoding` /
`toLinearLayout` instead of either accepting pre-relayout IR or emitting a typed
diagnostic. No new independent relayout bucket is needed.

## Saved Reproducer Rechecks

### `FZ-20260421-0014`: Proxy-Fence Insertion

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12 && \
  bin/triton-opt \
    /tmp/tmem_fz0014_plain_seq_round19_min_probe_branch_repro.mlir.make_llir.repro.mlir \
    --run-reproducer
```

Result: failed with the existing proxy-fence diagnostic.

```text
error: 'tt.func' op could not find an insertion point between cross-CTA
mbarrier.init ops and tracked mbarrier uses
```

Classification: existing `FZ-20260421-0014`. This is still the sequential
mbarrier/proxy-fence insertion gap, not a new conversion verifier issue.

### `FZ-20260421-0016`: Unencoded TMEM Operand

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12 && \
  bin/triton-opt \
    /tmp/tmem_compiler_boundaries_round38/unencoded_store_fz0016.mlir \
    -split-input-file
```

Result: aborted with the same `dyn_cast on a non-existent value` verifier
assertion as the relayout lit failure.

Classification: existing `FZ-20260421-0016`; this confirms the relayout lit
failure is representative of the broader unencoded-TMEM-operand verifier
boundary.

### `FZ-20260421-0017`: Encoded 64-bit TMEM Load/Store

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12 && \
  bin/triton-opt \
    /tmp/tmem_compiler_boundaries_round38/i64_chain_load_fz0017.mlir \
    -split-input-file \
    --convert-triton-gpu-to-llvm='target=sm_100'
```

Result: aborted with the existing `lowerTMemLdSt` bitwidth assertion.

```text
TensorMemoryUtils.cpp:7713:
Assertion `bitwidth == 32' failed.
```

Classification: existing `FZ-20260421-0017`. This remains isolated to encoded
64-bit non-reduction TMEM load/store lowering. It is distinct from `FZ-0016`
because the tensor layouts are encoded and reach the TMEM load/store lowering
planner before asserting on bitwidth.

## Classification

No new `FZ-*` bucket is needed from this lane.

Green surfaces:

- `lower_tensor_memory_to_llvm`;
- `tritonnvidiagpu_to_llvm`;
- `nvgpu_to_llvm`;
- TMA and TMA multicast conversion;
- warp-specialization conversion;
- `memdesc-subview-split`;
- TMEM layout/interleave/allocation/promotion;
- TMEM-adjacent MMA/TMA/membar lowering.

Known-red surfaces revalidated:

- `FZ-20260421-0014`: proxy-fence insertion cannot find an insertion point for
  sequential cross-CTA mbarrier intervals.
- `FZ-20260421-0016`: TMEM operand verification asserts on unencoded or
  non-distributed tensor layouts instead of accepting pre-relayout IR or
  producing a clean diagnostic.
- `FZ-20260421-0017`: encoded `i64`/`f64` non-reduction TMEM load/store reaches
  `lowerTMemLdSt` and asserts `bitwidth == 32`.

No new verifier gap, unsupported-case drift, FileCheck expectation drift, or
clean-negative mismatch was found outside those existing buckets.
