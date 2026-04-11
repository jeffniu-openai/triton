# TMEM Initiative Context Dump

Timestamp: 2026-04-11 02:55 UTC
Repo: `/root/code/triton-tmem-isolated`
Branch: `codex/tmem-isolated`
HEAD: `98f4fdbd27bf`

This file is a direct shell-written context/memory dump for session transfer because `apply_patch` is currently broken again. It is meant to capture the exact live state in one place so a fresh session can resume without depending on chat context.

## Executive Summary

The current dirty worktree contains a broad in-flight TMEM/MMAv5 refactor that already added:
- explicit TMEM physical-layout annotations on root allocs and selected stores/MMAv5 roots,
- memdesc-aware MMAv5 TMEM loader logic,
- row-plan projection helpers used across frontend, verifier, and LLVM,
- a pure rank-2 TMEM column-subview direct-lowering path in LLVM,
- several split-N and direct-load planner changes.

Despite earlier moments where some exacts looked green, the **current** exact reruns on this dirty worktree show three real live regressions:
1. `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-32-warps2-8-False-True-acc_dtype0]`
2. `python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-2-CGA_LAYOUT0-8-0-64-128-64]`
3. `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-layout9-64-128-32x32b_splitn-8-16x32bx2]`

The strongest current root-cause hypothesis is now:
- **over-broad root/backing row-plan preservation in `getTMemLdStRowPlanForQueryLayout(...)`**
- specifically, the explicit-physical-layout short-circuit is forcing root `32,64,128` row plans onto projected `64x32xf32` M64 direct loads that should instead use the projected `16,32,64` class.

That hypothesis is especially strong for `test_mma_shared_inputs[...]` and likely also explains the multicta regression.

## Tooling State

### `apply_patch`
- Broken again.
- Immediate failure mode:
  - `No such file or directory`
- The failure happened when trying to patch:
  - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
- I stopped immediately and made no fallback code edits after the failure.

### `apply_patch` runtime backup
- backup dir:
  - `/root/.codex/tooling_backups/apply_patch_runtime_20260410T210943Z`
- symlink:
  - `/root/.codex/tooling_backups/apply_patch_runtime_latest`

## Worktree State

No new commit was made in this slice.

### Dirty files at this checkpoint

#### Initiative docs
- `.codex/initiatives/tmem_linear_generalization/gb200_branch_recovery_plan.md`
- `.codex/initiatives/tmem_linear_generalization/gb200_current_branch_examples_multicta_failures.txt`
- `.codex/initiatives/tmem_linear_generalization/gb200_failure_manifest.md`
- `.codex/initiatives/tmem_linear_generalization/handoff_2026-04-09.md`
- `.codex/initiatives/tmem_linear_generalization/log.md`
- `.codex/initiatives/tmem_linear_generalization/memory.md`

#### Code
- `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h`
- `include/triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h`
- `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`
- `lib/Dialect/TritonGPU/Transforms/Pipeliner/MMAv5PipelineUtility.cpp`
- `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
- `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
- `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
- `lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp`
- `python/src/gluon_ir.cc`
- `python/test/gluon/test_core.py`
- `python/triton/experimental/gluon/language/nvidia/blackwell/__init__.py`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSCodePartition.cpp`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSDataPartition.cpp`
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/HoistTmemStore.cpp`
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertTmemAref.cpp`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAHelpers.h`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`

## Fresh Validation In This Slice

### Build
- Command:
  - `make -j8`
- Result:
  - `PASSED`

### Exact isolated current-head reruns

#### 1. `test_block_m_64_mma[linear]`
- Command:
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-blockm64-current PYTHONPATH=python:. pytest -s --tb=short -vv 'python/test/gluon/test_core.py::test_block_m_64_mma[linear]'`
- Result:
  - `PASSED`

#### 2. `test_tmem_linear_roundtrip_splitn_shapes[...]`
- Command:
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-roundtrip-current PYTHONPATH=python:. pytest -s --tb=short -vv 'python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]'`
- Result:
  - `PASSED`

#### 3. `test_mma_shared_inputs[...]`
- Command:
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-mma-shared-current PYTHONPATH=python:. pytest -s --tb=short -vv 'python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-32-warps2-8-False-True-acc_dtype0]'`
- Result:
  - `FAILED`
- Current symptom:
  - `1439 / 2048` mismatches (`70.3%`)
  - greatest absolute difference `35.35748291015625`

#### 4. `test_tmem_descriptor_chain_matrix[...]`
- Command:
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-descriptor-current PYTHONPATH=python:. pytest -s --tb=short -vv 'python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-layout9-64-128-32x32b_splitn-8-16x32bx2]'`
- Result:
  - `FAILED`
- Current symptom:
  - `4096 / 8192` mismatches (`50.0%`)

#### 5. multicta example exact old-mainline repro
- Command:
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-multicta-current PYTHONPATH=python:. pytest -s --tb=short -vv 'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-2-CGA_LAYOUT0-8-0-64-128-64]'`
- Result:
  - `FAILED`
- Current symptom:
  - `9600 / 20000` mismatches (`48.0%`)

### Merge-base confirmation

Representative merge-base check:
- repo:
  - `/root/code/triton-mergebase-ci`
- command:
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-mergebase-mma-shared PYTHONPATH=python:. pytest -s --tb=short -vv 'python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-32-warps2-8-False-True-acc_dtype0]'`
- result:
  - `PASSED`

Conclusion:
- `test_mma_shared_inputs[...]` is a real branch regression, not preexisting on merge-base.

## Current Artifact Paths

### `test_mma_shared_inputs[...]`
Current-head:
- `/tmp/triton-cache-mma-shared-current/HPMJ2P3ABPLICQPMP2N5U4YBZKKP3TTTPPW3SA5VDMUBAZBRM7GA/mma_kernel.ttgir`
- `/tmp/triton-cache-mma-shared-current/HPMJ2P3ABPLICQPMP2N5U4YBZKKP3TTTPPW3SA5VDMUBAZBRM7GA/mma_kernel.ptx`
- `/tmp/triton-cache-mma-shared-current/N562PGEQOPOG75A44H7VCLQWCEE4TWPEUGD26KZ7RRJTH3MMZ46A/mma_kernel.ttgir`
- `/tmp/triton-cache-mma-shared-current/N562PGEQOPOG75A44H7VCLQWCEE4TWPEUGD26KZ7RRJTH3MMZ46A/mma_kernel.ptx`

Merge-base:
- `/tmp/triton-cache-mergebase-mma-shared/R5QTWQAPWXYE4XOT7RTRU2KYDSXG24THC32FY7S6NCYWHLKR5V3A/mma_kernel.ttgir`
- `/tmp/triton-cache-mergebase-mma-shared/R5QTWQAPWXYE4XOT7RTRU2KYDSXG24THC32FY7S6NCYWHLKR5V3A/mma_kernel.ptx`

### multicta exact
- `/tmp/triton-cache-multicta-current/M2FHHTFUXFXCC6IDQPZL7QD5GLUJMOEABXHSDGL34FJ5YQ772VAA/_matmul_kernel.ttgir`
- `/tmp/triton-cache-multicta-current/M2FHHTFUXFXCC6IDQPZL7QD5GLUJMOEABXHSDGL34FJ5YQ772VAA/_matmul_kernel.ptx`
- `/tmp/triton-cache-multicta-current/M2FHHTFUXFXCC6IDQPZL7QD5GLUJMOEABXHSDGL34FJ5YQ772VAA/_matmul_kernel.source`
- `/tmp/triton-cache-multicta-current/V6VZXLSVGNZHSY437YGJM7ME3DRC5TNCUSPFAW3KC5LLUEQDWUPA/_matmul_kernel.ttgir`
- `/tmp/triton-cache-multicta-current/V6VZXLSVGNZHSY437YGJM7ME3DRC5TNCUSPFAW3KC5LLUEQDWUPA/_matmul_kernel.ptx`
- `/tmp/triton-cache-multicta-current/V6VZXLSVGNZHSY437YGJM7ME3DRC5TNCUSPFAW3KC5LLUEQDWUPA/_matmul_kernel.source`

### descriptor-chain exact
- `/tmp/triton-cache-descriptor-current/ZJCQZPEJIPFBGJWMD23YUAAXF4NJR4XOSTPUKZT35GJYFINKMU5Q/tmem_descriptor_chain_matrix_kernel.ttgir`
- `/tmp/triton-cache-descriptor-current/ZJCQZPEJIPFBGJWMD23YUAAXF4NJR4XOSTPUKZT35GJYFINKMU5Q/tmem_descriptor_chain_matrix_kernel.ptx`
- `/tmp/triton-cache-descriptor-current/ZJCQZPEJIPFBGJWMD23YUAAXF4NJR4XOSTPUKZT35GJYFINKMU5Q/tmem_descriptor_chain_matrix_kernel.source`

## Current-Vs-Merge-Base Codegen Facts

### `test_mma_shared_inputs[...]`
Merge-base TTGIR load result layout:
- `#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>`

Current-head TTGIR load result layout:
- `#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>`

Merge-base PTX family:
- `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`

Current-head PTX family:
- repeated scalar loads:
  - `tcgen05.ld.sync.aligned.32x32b.x1.b32`

Interpretation:
- the regression is already present in TTGIR/register-layout planning
- this is not just an LLVM emission bug

### multicta exact
Current TTGIR:
- `#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>`

Current PTX:
- two loads per slice:
  - `tcgen05.ld.sync.aligned.16x32bx2.x8.b32 ... [%r + 0], 8`
  - `tcgen05.ld.sync.aligned.16x32bx2.x8.b32 ... [%r + 16], 8`

Merge-base reference from earlier sessions:
- one packed family per slice:
  - `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`

Interpretation:
- this remains the same split-N / projected-view direct-load regression class

### descriptor-chain exact
Current TTGIR root alloc carries explicit attrs:
- `ttng.tmem_ldst_row_plan = array<i32: 32, 64, 128, 0>`
- `ttng.tmem_physical_layout = ...`

Current PTX still looks packed, not scalarized:
- `tcgen05.st.sync.aligned.16x32bx2.x16.b32`
- `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`

Interpretation:
- this bucket is still in the explicit-physical-layout / split-N recovery area
- but it has not been reduced as sharply as the `64x32xf32` direct-load regressions

## What Changed Earlier In The Dirty Worktree (Important Context)

These are the large conceptual changes already present in the dirty code and should not be rediscovered from scratch:

### Gluon / frontend
- `python/src/gluon_ir.cc`
  - now follows forwarding sources when finding backing TMEM allocs
  - annotates MMAv5 accumulator roots with explicit root row plans
  - annotates MMAv5 accumulator roots with explicit physical family layouts
  - annotates MMAv5 TMEM operand roots with copied/derived root contracts
  - annotates direct TMEM roots from stores using inferred raw query layout + projected row plan
  - has handle-aware candidate-layout ranking changes for TMEM reg-layout selection

### MMAv5 lowering
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAHelpers.h`
  - `DotOpMmaV5TmemLoader::build(...)` now takes the original memdesc value in addition to the lowered base
  - loader tries memdesc-aware query-layout inference first
  - sorted tile ordering can use `getTMemViewOffsetForLowering(memDescValue, ...)` instead of type-only offsets
  - dot conversion now threads original memdesc values into those helpers

### TensorMemory lowering
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
  - now uses `getTMemLdStRowPlanForQueryLayout(...)` in multiple raw/support query paths
  - has pure rank-2 TMEM column-subview handling intended to reuse the source family instead of degrading immediately to sliced query-type fallback
  - current dirty state still leaves multicta red, so whatever is there is not sufficient

### TensorMemory planner / query logic
- `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
- `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
  - many changes already exist around explicit physical layouts, support-query planning, projected row plans, canonical M64 split-N and contiguous M64 fast paths, and direct-layout generation
  - current live hypothesis focuses on row-plan selection in `getTMemLdStRowPlanForQueryLayout(...)`, not on re-adding more ad hoc rescue logic

## Best Current Diagnosis

### Narrowest likely core bug
File:
- `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`

Function:
- `getTMemLdStRowPlanForQueryLayout(Value memDesc, MemDescType queryTy, const TMemLdStQueryLayout &queryLayout)`

Current problematic logic:
- computes `queryPlan`
- computes `layoutPlan`
- computes `backingPlan`
- then has an early rule:
  - if `memDesc` has an explicit physical layout and that explicit layout equals `queryLayout.layout`, return `backingPlan`

Why this is suspicious:
- a producer-owned physical family contract is not the same thing as the row-anchor plan needed for the projected query
- for projected `64x32xf32` M64 loads, the direct ld/st planner likely needs the projected `layoutPlan` (`16,32,64` class)
- the current short-circuit appears to force the root `32,64,128` contract instead
- that lines up with both:
  - `test_mma_shared_inputs[...]` switching from merge-base `16x32bx2.x16` to current `32x32b.x1`
  - multicta switching from merge-base `16x32bx2.x16` to current `16x32bx2.x8 + x8`

### Most likely next patch
I was about to make exactly this change before `apply_patch` failed:
- remove the explicit-layout-equals-query-layout => `backingPlan` short-circuit entirely
- broaden the existing projected-M64 override from explicit-view-only to all projected M64 queries

Intended logic:
- if:
  - `queryTy.getRank() == 2`
  - `queryTy.getElementTypeBitWidth() == 32`
  - `queryTy.getShape()[0] == 64`
  - `layoutPlan->rowSpan == queryTy.getShape()[0]`
  - `layoutPlan->rowSpan < queryPlan->rowSpan`
- then:
  - return `layoutPlan`
- else:
  - return `queryPlan`

I did **not** apply it.

## Why Earlier Handoff Sections Are Now Stale

Earlier tail sections in the handoff and dirty initiative docs still describe a moment where:
- multicta exact passed
- multicta full file passed
- `test_block_m_64_mma[linear]` looked like the main remaining core bug

That is **not** the current exact state.

Current truth is:
- `test_block_m_64_mma[linear]` passes in isolation
- the real live exacts are:
  - `test_mma_shared_inputs[...]`
  - multicta exact
  - `test_tmem_descriptor_chain_matrix[...]`

Do not rely on the older prose without refreshing it after the next real fix.

## Recommended Next Steps For The Next Session

1. Permanently fix `apply_patch` first.
2. Patch `getTMemLdStRowPlanForQueryLayout(...)` as described above.
3. Rebuild:
   - `make -j8`
4. Rerun exacts in this order:
   - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-32-warps2-8-False-True-acc_dtype0]`
   - `python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-2-CGA_LAYOUT0-8-0-64-128-64]`
   - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-layout9-64-128-32x32b_splitn-8-16x32bx2]`
   - controls:
     - `python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
     - `python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]`
5. If `test_mma_shared_inputs[...]` flips back to packed `16x32bx2.x16` and multicta follows, update initiative docs immediately and commit.
6. If `test_mma_shared_inputs[...]` stays on `32x32b.x1`, inspect all callers of `getTMemLdStRowPlanForQueryLayout(...)` to confirm no later layer is re-injecting `backingPlan`:
   - `python/src/gluon_ir.cc`
   - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
   - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`

## Durable Docs To Refresh After The Next Real Fix
- `.codex/initiatives/tmem_linear_generalization/memory.md`
- `.codex/initiatives/tmem_linear_generalization/log.md`
- `.codex/initiatives/tmem_linear_generalization/handoff_2026-04-09.md`
- `.codex/initiatives/tmem_linear_generalization/gb200_failure_manifest.md`
- `.codex/initiatives/tmem_linear_generalization/gb200_branch_recovery_plan.md`

## One-Line Bottom Line

The current dirty branch appears to be over-preserving the root physical TMEM row-plan contract for projected M64 direct loads. The next real fix should be in row-plan selection for query layouts, not another layer of packet-level surgery.
