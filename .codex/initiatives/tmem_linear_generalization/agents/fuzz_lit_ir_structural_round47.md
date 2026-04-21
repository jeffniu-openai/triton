# Round 47: Lit, IR, and Structural Compiler-Only Fuzzing

Date: 2026-04-21 14:41 UTC

Branch: `codex/tmem`

HEAD: `f54e7c64c350`

Scope: discovery/classification only. No backend/compiler code or checked-in
tests were modified. This report is the only file written by this lane.

## Objective

Exercise the compiler-only and structural side of the TMEM backend without
adding runtime test coverage:

- TMEM-bearing lit tests in `TritonNvidiaGPU`, `TritonGPU`, `NVWS`, and
  `Conversion`;
- the checked-in structural fuzzer
  `python/test/gluon/test_tmem_structural_fuzzer.py`;
- verifier failures, unsupported diagnostics, IR/FileCheck expectation drift,
  XPASS/XFAIL drift, and missing clean-negative boundaries.

## Required Build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

Compiler tool build:

```bash
cd $(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())') && ninja triton-opt
```

Result:

```text
ninja: no work to do.
```

## Direct TMEM Lit Set

Command:

```bash
cd $(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())') && \
  lit -v \
    test/TritonNvidiaGPU/tmem_layouts.mlir \
    test/TritonNvidiaGPU/interleave_tmem.mlir \
    test/TritonGPU/hoist-tmem-alloc.mlir \
    test/TritonGPU/promote-lhs-to-tmem.mlir \
    test/TritonGPU/memdesc-subview-split.mlir \
    test/NVWS/hoist_tmem_store.mlir \
    test/NVWS/aref-tmem-insertion.mlir
```

Result: `7` discovered, `7 passed`.

No FileCheck drift, verifier drift, unsupported-diagnostic drift, or unexpected
lit failure was observed in the directly named TMEM lit files.

## Broader NVIDIA GPU Lit Guardrail

Command:

```bash
cd $(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())') && \
  lit -v test/TritonNvidiaGPU test/TritonGPU/nvidia-fpsan.mlir
```

Result: `16` discovered, `16 passed`.

This broadens the direct lit check to adjacent NVIDIA GPU files whose TMEM
coverage is not always obvious from the filename:

- `test/TritonNvidiaGPU/membar.mlir`;
- `test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir`;
- `test/TritonNvidiaGPU/test_tensor_memory_allocation.mlir`;
- `test/TritonNvidiaGPU/mma_lowering.mlir`;
- `test/TritonNvidiaGPU/invalid.mlir`;
- `test/TritonGPU/nvidia-fpsan.mlir`.

No IR/asm expectation drift was found in this group.

## Adjacent TMEM-Bearing Lit Expansion

Command:

```bash
cd $(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())') && \
  lit -v \
    test/Conversion/lower_tensor_memory_to_llvm.mlir \
    test/Conversion/relayout_tritongpu.mlir \
    test/TritonGPU/accelerate-matmul.mlir \
    test/TritonGPU/optimize-partition-warps.mlir \
    test/TritonGPU/pipeline-schedule-loop.mlir \
    test/TritonGPU/hoist-tmem-alloc.mlir \
    test/TritonGPU/promote-lhs-to-tmem.mlir \
    test/TritonGPU/memdesc-subview-split.mlir \
    test/NVWS/hoist_tmem_store.mlir \
    test/NVWS/aref-tmem-insertion.mlir
```

Result: `10` discovered, `9 passed`, `1 failed`.

Passing files:

- `Conversion/lower_tensor_memory_to_llvm.mlir`;
- `TritonGPU/accelerate-matmul.mlir`;
- `TritonGPU/optimize-partition-warps.mlir`;
- `TritonGPU/pipeline-schedule-loop.mlir`;
- `TritonGPU/hoist-tmem-alloc.mlir`;
- `TritonGPU/promote-lhs-to-tmem.mlir`;
- `TritonGPU/memdesc-subview-split.mlir`;
- `NVWS/hoist_tmem_store.mlir`;
- `NVWS/aref-tmem-insertion.mlir`.

Failing file:

- `Conversion/relayout_tritongpu.mlir`.

Failure signature:

```text
triton-opt: .../llvm/Support/Casting.h:650: decltype(auto) llvm::dyn_cast(From&) ...:
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
...
mlir::triton::gpu::TritonGPUDialect::toLinearLayout(...)
mlir::triton::nvidia_gpu::computeTMemLdStEncodingInfoImpl(...)
mlir::triton::nvidia_gpu::verifyTMEMOperand(...)
mlir::triton::nvidia_gpu::TMEMAllocOp::verify()
FileCheck error: '<stdin>' is empty.
```

Classification: existing `FZ-20260421-0016`. This is the known verifier/parser
path where a TMEM operand with an unencoded or otherwise non-distributed tensor
layout reaches `toLinearEncoding`/`toLinearLayout` and asserts instead of
emitting a clean diagnostic. The failure reproduces the earlier lit/relayout
boundary and does not require a new independent FZ bucket.

## Structural Fuzzer Collection

Command:

```bash
PYTHONPATH=.:./python pytest -q --collect-only \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `33` tests collected.

Attempting to collect with CUDA hidden:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  pytest -q -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: collection error before any structural case executed:

```text
RuntimeError: 0 active drivers ([]). There should only be one.
```

Classification: harness requirement, not a TMEM backend finding. The file
imports `triton._internal_testing`, which initializes the active driver at
collection time. The structural fuzzer therefore needs a visible CUDA driver
even when the checked cases are being used as structural compiler/assembly
coverage rather than as a broad runtime comparison sweep.

## Checked-In Structural Fuzzer

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `9 passed, 24 xfailed`.

The xfail rows remained strict expected failures. No XPASS transition,
unexpected compiler crash, new false unsupported diagnostic, or new
misclassification was observed.

Expected xfail diagnostics included the known `FZ-20260421-0001` late illegal
`ttg.memdesc_index` signature in dynamic/control-flow-carried TMEM descriptor
cases:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
Pipeline failed while executing [`ConvertTritonGPUToLLVM` on 'builtin.module' operation]
```

The checked-in structural fuzzer still covers the existing expected-failure
families only:

- `FZ-20260421-0001`: dynamic/generic TMEM memdesc_index reaches late lowering;
- `FZ-20260421-0002`: helper-returned or dynamic-if TMEM view wrong results;
- `FZ-20260421-0003`: descriptor-view `ld/st` packet-order wrong results;
- `FZ-20260421-0004`: descriptor-view or indexed `ld.red` opcode loss/software
  reduction;
- `FZ-20260421-0005`: lifted 256-row allocator assertion;
- `FZ-20260421-0006`: `ld.red` transpose/slice false unsupported;
- `FZ-20260421-0007`: scaled-MMAv5 dynamic accumulator subslice wrong results;
- `FZ-20260421-0008`: 2CTA indexed `ld.red` row/col optimizer abort;
- `FZ-20260421-0009`: 1CTA direct indexed `ld.red` allocator assertion.

## Classification

No new independent `FZ-*` bucket is proposed from this lane.

Current compiler-only status:

- Direct and adjacent TMEM lit coverage is green except the known
  `Conversion/relayout_tritongpu.mlir` crash.
- `Conversion/relayout_tritongpu.mlir` remains existing `FZ-20260421-0016`,
  not a new lit drift.
- The structural fuzzer is stable at `9 passed, 24 xfailed`.
- The strict xfail set did not show XPASS drift.
- No new unsupported-diagnostic false negative or missing clean-negative
  boundary was exposed by these compiler-only checks.

## Follow-Up Candidates

1. Add a lit-level clean-negative for the `FZ-20260421-0016` unencoded TMEM
   operand path once repair begins, so the verifier emits a diagnostic instead
   of asserting.
2. Consider a compiler-only structural sentinel for dynamic
   `ttng.tmem_copy`/linear descriptor selection from Round 45 after the
   discovery campaign pivots from cataloging to checked-in test promotion.
3. Keep running `Conversion/relayout_tritongpu.mlir` alongside the green TMEM
   lit set because it catches the verifier/assertion class that ordinary
   `TritonNvidiaGPU` lit coverage does not.

Backend repair remains deferred per the discovery-only campaign.
