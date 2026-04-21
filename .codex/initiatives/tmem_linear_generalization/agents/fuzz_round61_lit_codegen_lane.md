# Round 61 Lane C: TMEM lit/codegen compiler breadth

Date: 2026-04-21 16:23 UTC
Branch: `codex/tmem`
HEAD at start: `a35ca5bc67e022f01e191d9c49c8ad6088e44982`
Mode: discovery/cataloging only. No backend, compiler, runtime, or checked-in
test source files were edited.

## Scope

This lane focused on compiler-only lit/codegen breadth around TMEM lowering,
allocation, conversion, NVWS, warp specialization, partitioning, and 2CTA /
larger-than-2CTA ownership surfaces. It intentionally did not repeat Round 60's
exact lit file groups. Instead it covered files Round 60 skipped or left to
adjacent subsystems:

- TMEM hoist/promotion and tensor-memory LLVM lowering.
- Blackwell TritonGPU-to-LLVM codegen for TMEM load/store/copy/MMAv5/`ld.red`
  paths.
- Partition verifier and loop partitioning with three-stage partition tags.
- NVWS aref, stage/phase, warp-group, and invalid diagnostic coverage.
- NvidiaGPU TMEM layout, interleave, mbarrier, and cluster/more-than-2CTA
  ownership checks.

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

```bash
cd build/cmake.linux-aarch64-cpython-3.12 && ninja triton-opt
```

Result: `ninja: no work to do`.

## Lit sweep 1: TMEM hoist, partitioning, tensor-memory conversion

Command:

```bash
cd build/cmake.linux-aarch64-cpython-3.12 && lit -v \
  test/TritonGPU/hoist-tmem-alloc.mlir \
  test/TritonGPU/promote-lhs-to-tmem.mlir \
  test/TritonGPU/partition-verifier-locality.mlir \
  test/TritonGPU/partition-loops.mlir \
  test/Conversion/lower_tensor_memory_to_llvm.mlir \
  test/Conversion/tritongpu_to_llvm_blackwell.mlir
```

Result:

```text
Total Discovered Tests: 6
  Passed: 6 (100.00%)
```

Coverage notes:

- `hoist-tmem-alloc.mlir` passed both default and `hoist-out-of-if=true`
  RUN lines, covering loop-carried async token hoisting and conditional TMEM
  allocation motion.
- `promote-lhs-to-tmem.mlir` and `lower_tensor_memory_to_llvm.mlir` passed,
  including two-CTA automatic TMEM lifecycle lowering.
- `tritongpu_to_llvm_blackwell.mlir` passed, covering Blackwell TMEM
  store/load, `ld.red`, copy, scaled/MMAv5, and codegen opcode checks.
- Partition locality and loop partitioning passed, including multi-stage
  partition tags and warp-specialize partition metadata.

## Lit sweep 2: NVWS aref and invalid diagnostics

Command:

```bash
cd build/cmake.linux-aarch64-cpython-3.12 && lit -v \
  test/NVWS/lower_aref.mlir \
  test/NVWS/insert_aref.mlir \
  test/NVWS/ops.mlir \
  test/NVWS/invalid.mlir \
  test/NVWS/assign_stage_phase.mlir \
  test/NVWS/lower_warp_group.mlir \
  test/NVWS/aref-tmem-insertion.mlir
```

Result:

```text
Total Discovered Tests: 7
  Passed: 7 (100.00%)
```

Coverage notes:

- NVWS aref creation/lowering, invalid diagnostics, stage/phase assignment,
  warp-group lowering, and TMEM insertion all passed.
- Invalid diagnostics remained precise for aref dimension mismatches and async
  safety violations involving TMEM allocation users.

## Lit sweep 3: NvidiaGPU TMEM, mbarrier, and ownership

Command:

```bash
cd build/cmake.linux-aarch64-cpython-3.12 && lit -v \
  test/TritonNvidiaGPU/tmem_layouts.mlir \
  test/TritonNvidiaGPU/interleave_tmem.mlir \
  test/TritonNvidiaGPU/membar.mlir \
  test/TritonNvidiaGPU/membar-cluster.mlir \
  test/Conversion/allocate_shared_memory.mlir \
  test/Conversion/tritongpu_to_ptx_mmav3.mlir
```

Result:

```text
Total Discovered Tests: 6
  Passed: 6 (100.00%)
```

Coverage notes:

- TMEM linear layout and interleave checks passed.
- Mbarrier and cluster insertion checks passed, including two-CTA TMEM copy,
  two-CTA MMAv5 wait-barrier cases, and a four-CTA module containing two-CTA
  MMA ownership.
- Shared-memory allocation and MMAv3 PTX conversion controls passed.

## Direct triton-opt replays

These commands were run outside lit to stress the same pass-manager/codegen
surfaces directly and to confirm they exit cleanly without FileCheck masking a
crash:

```bash
build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  test/TritonGPU/hoist-tmem-alloc.mlir \
  -split-input-file -allow-unregistered-dialect \
  -tritongpu-hoist-tmem-alloc='hoist-out-of-if=true' -canonicalize \
  >/tmp/round61_hoist_tmem_alloc.out
```

Exit code: `0`; output: `261` lines.

```bash
build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  test/TritonNvidiaGPU/membar-cluster.mlir \
  -split-input-file --allocate-shared-memory -test-print-membar \
  >/tmp/round61_membar_cluster.out
```

Exit code: `0`; output: `657` lines.

```bash
build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  --split-input-file test/NVWS/invalid.mlir --verify-diagnostics \
  >/tmp/round61_nvws_invalid.out
```

Exit code: `0`; output: `3` lines.

```bash
build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  test/Conversion/tritongpu_to_llvm_blackwell.mlir \
  -split-input-file --convert-triton-gpu-to-llvm=compute-capability=100 -cse \
  >/tmp/round61_tritongpu_to_llvm_blackwell.out
```

Exit code: `0`; output: `30805` lines.

## Aggregate result

- Required build: `make -j8` no-op.
- `triton-opt` build: no-op.
- Lit files run: `19`.
- Lit result: `19 passed, 0 failed`.
- Direct `triton-opt` replays: `4` commands, all exit code `0`.

## Classification

No new independent `FZ-*` was found.

No existing failing bucket was reproduced in this lane:

- Existing `FZ-20260421-0016` was not reproduced; this lane did not run the
  known-failing `Conversion/relayout_tritongpu.mlir` test.
- Existing descriptor/view and runtime miscompile buckets were not exercised
  by runtime execution here; this lane stayed compiler-only as requested.
- Existing larger-CGA ownership bucket `FZ-20260421-0010` was not reproduced;
  `membar-cluster.mlir` four-CTA/two-CTA ownership checks passed.

No compiler crash, verifier drift, invalid-diagnostic drift, PassManager
failure, TMEM allocation/lifetime failure, conversion failure, NVWS lowering
failure, warp-specialization/partition failure, or 2CTA/more-than-2CTA
ownership failure was observed.

Backend/compiler repairs remain deferred.
