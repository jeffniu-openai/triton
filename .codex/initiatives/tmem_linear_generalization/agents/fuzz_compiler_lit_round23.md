# Round 23 Lane BF: Compiler-Only TMEM Lowering Audit

- Date: 2026-04-21 12:13 UTC
- Branch: `codex/tmem`
- HEAD: `faea54b71890629c501c598b374e2b3e650530f2`
- Mode: discovery/cataloging only; no backend/compiler fixes.
- Repo edit scope: this report plus initiative bookkeeping.

## Scope

This lane audited compiler-only TMEM lowering paths that are easy for Python
runtime fuzzing to miss:

- `TensorMemoryAllocation` and saved allocator repros;
- `TensorMemoryUtils` encoding-info verification and relayout;
- MMAv5/scaled-MMAv5 compiler-only lit;
- proxy fence insertion and saved mbarrier repros;
- TMEM copy repros;
- `ld.red` optimize/allocation repros;
- dynamic memdesc legality symptoms already cataloged by runtime fuzzing.

## Build Gate

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Compiler tool gate:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
```

Result:

```text
ninja: no work to do.
```

## Existing Lit Tests

All commands below were run from
`/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`.

Passing tests:

```bash
lit -v test/TritonNvidiaGPU/tmem_layouts.mlir \
  2>&1 | tee /tmp/tmem_round23_lit_tmem_layouts.log

lit -v test/TritonNvidiaGPU/mma_lowering.mlir \
  2>&1 | tee /tmp/tmem_round23_lit_mma_lowering.log

lit -v test/TritonGPU/proxy_fence_insertion.mlir \
  2>&1 | tee /tmp/tmem_round23_lit_proxy_fence.log

lit -v test/TritonNvidiaGPU/interleave_tmem.mlir \
  2>&1 | tee /tmp/tmem_round23_lit_interleave_tmem.log

lit -v test/NVWS/aref-tmem-insertion.mlir \
  2>&1 | tee /tmp/tmem_round23_lit_aref_tmem.log

lit -v test/NVWS/hoist_tmem_store.mlir \
  2>&1 | tee /tmp/tmem_round23_lit_hoist_tmem_store.log

lit -v test/TritonGPU/hoist-tmem-alloc.mlir \
  2>&1 | tee /tmp/tmem_round23_lit_hoist_tmem_alloc.log

lit -v test/TritonGPU/memdesc-subview-split.mlir \
  2>&1 | tee /tmp/tmem_round23_lit_memdesc_subview_split.log
```

Results:

```text
PASS TritonNvidiaGPU/tmem_layouts.mlir
PASS TritonNvidiaGPU/mma_lowering.mlir
PASS TritonGPU/proxy_fence_insertion.mlir
PASS TritonNvidiaGPU/interleave_tmem.mlir
PASS NVWS/aref-tmem-insertion.mlir
PASS NVWS/hoist_tmem_store.mlir
PASS TritonGPU/hoist-tmem-alloc.mlir
PASS TritonGPU/memdesc-subview-split.mlir
```

Failing test:

```bash
lit -v test/Conversion/relayout_tritongpu.mlir \
  2>&1 | tee /tmp/tmem_round23_lit_relayout_tritongpu.log
```

Result:

```text
FAIL: TRITON :: Conversion/relayout_tritongpu.mlir
```

The direct command also reproduces:

```bash
./build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  test/Conversion/relayout_tritongpu.mlir \
  -split-input-file \
  -convert-triton-to-tritongpu='target=cuda:100 num-warps=4 enable-source-remat=true' \
  -relayout-tritongpu \
  --mlir-disable-threading \
  >/tmp/tmem_round23_relayout_direct.out \
  2>/tmp/tmem_round23_relayout_direct.err
```

Exit:

```text
EXIT:134
```

Important stack:

```text
dyn_cast on a non-existent value
mlir::triton::gpu::TritonGPUDialect::toLinearLayout
mlir::triton::gpu::toLinearEncoding
mlir::triton::nvidia_gpu::computeTMemLdStEncodingInfoImpl
mlir::triton::nvidia_gpu::computeTMemLdStEncodingInfo
mlir::triton::nvidia_gpu::isDistributedLayoutTMemCompatible
mlir::triton::nvidia_gpu::verifyTMEMOperand
mlir::triton::nvidia_gpu::TMEMAllocOp::verify
```

The crash is triggered before pass execution, while parsing/verifying this
existing input:

```mlir
%cst = arith.constant dense<0> : tensor<128x128xi8>
%result = ttng.tmem_alloc %cst :
  (tensor<128x128xi8>) ->
  !ttg.memdesc<128x128xi8, #ttng.tensor_memory_scales_encoding<>,
               #ttng.tensor_memory>
```

As a contrast, the `origin/main` version of the same test file was extracted
to `/tmp/tmem_round23_origin_main_relayout_tritongpu.mlir` and run through the
current branch's `triton-opt` with the same pipeline. It produced the same
assertion. This is not proof that upstream main's own binary crashes, but it
does show the trigger is not a local edit unique to the current test file
contents.

Classification: new candidate `FZ-20260421-0016`.

`FZ-20260421-0016`: compiler-only TMEM verifier/encoding-info crash for
`ttng.tmem_alloc` with an unencoded tensor operand and a
`TensorMemoryScalesEncodingAttr` result. `verifyTMEMOperand` calls
`computeTMemLdStEncodingInfo`, which calls `toLinearEncoding` on a tensor type
whose encoding is absent, causing an assertion instead of either accepting the
pre-relayout input or emitting a typed verifier diagnostic. The likely repair
area after discovery ends is the TMEM operand verifier / compatibility path,
not FileCheck expectations.

## Saved Reproducer Replay

All commands below were run from `/root/code/triton`.

### Existing `FZ-20260421-0014`: Proxy Fence / Mbarrier

```bash
./build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir \
  --run-reproducer \
  >/tmp/tmem_round23_fz0014_runrep.out \
  2>/tmp/tmem_round23_fz0014_runrep.err
```

Exit:

```text
EXIT:1
```

Diagnostic:

```text
'tt.func' op could not find an insertion point between cross-CTA mbarrier.init
ops and tracked mbarrier uses
```

Classification: existing `FZ-20260421-0014`, already classified by Round 19
as a preexisting upstream proxy-fence limitation and not branch-specific.

The direct two-copy saved repro:

```bash
./build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_copy_mbarrier_round14_ai001_direct_two_copy_fail.mlir \
  --run-reproducer \
  >/tmp/tmem_round23_copy_mbarrier_direct_runrep.out \
  2>/tmp/tmem_round23_copy_mbarrier_direct_runrep.err
```

also exits `1` with the same insertion-point diagnostic, so it stays in the
same bucket.

### Existing `FZ-20260421-0008`: `ld.red` Optimizer Crash

```bash
./build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_ldred_crash_round6_min_256x2_evenodd.mlir \
  --run-reproducer \
  >/tmp/tmem_round23_fz0008_runrep.out \
  2>/tmp/tmem_round23_fz0008_runrep.err
```

Exit:

```text
EXIT:134
```

Diagnostic:

```text
LLVM ERROR: Dimensions must match, ignoring order, but they don't.
Got dims: ["row", "col"] and ["row", "col", "block"]
```

Important stack:

```text
mlir::triton::LinearLayout::compose
mlir::triton::nvidia_gpu::getExpandedRowDirectI32x32bLayout
mlir::triton::nvidia_gpu::getDistributedLayoutForTmemLdSt
mlir::triton::nvidia_gpu::getTmemLoadLayoutSplitLongM
TMemLoadReducePattern::matchAndRewrite
TritonNvidiaGPUOptimizeTMemLayoutsPass::runOnOperation
```

Classification: existing `FZ-20260421-0008`; this compiler-only replay
confirms the root remains a row/col/block dimensionality mismatch in the
`ld.red` optimizer layout path.

### Existing `FZ-20260421-0009`: `ld.red` Allocator Assertion

```bash
./build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_generator_ldst_ldred_round7_ldred256x32_alloc_assert.mlir \
  --run-reproducer \
  >/tmp/tmem_round23_fz0009_runrep.out \
  2>/tmp/tmem_round23_fz0009_runrep.err
```

Exit:

```text
EXIT:134
```

Diagnostic:

```text
TensorMemoryAllocation.cpp:65:
MemoryBitMap::findFirstFit(...): Assertion `kNumRows - numRows >= 0' failed.
```

Classification: existing `FZ-20260421-0009`; the saved compiler repro still
reaches `TensorMemoryAllocation.cpp::allocateTMem`.

## Summary

New candidate:

- `FZ-20260421-0016`: compiler-only parse/verify crash for
  `ttng.tmem_alloc` with an unencoded tensor operand and tensor-memory-scales
  result in `test/Conversion/relayout_tritongpu.mlir`.

Existing buckets re-confirmed:

- `FZ-20260421-0014`: proxy-fence insertion cannot place fences between
  sequential cross-CTA mbarrier regions; saved plain and TMEM-copy repros still
  fail the same way.
- `FZ-20260421-0008`: `ld.red` optimizer row/col/block dimensionality crash.
- `FZ-20260421-0009`: direct indexed `ld.red` allocator row-count assertion.

Green compiler-only coverage:

- TMEM layout lit;
- MMAv5 lowering lit;
- proxy-fence lit baseline;
- interleave/hoist/NVWS TMEM lit baselines;
- memdesc subview split lit baseline.

No backend/compiler code was changed.
