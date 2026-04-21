# Round 53 TMEM compiler/lit fuzzing lane B

Date: 2026-04-21
Branch: `codex/tmem`
HEAD at start: `060d6c136dc116a6a6fb76773fca938596549661`
Scope: compiler-only adversarial fuzzing for TMEM-bearing MLIR/lit surfaces.
Backend repairs were intentionally not attempted.

## Build

Command:

```bash
cd /root/code/triton
make -j8
```

Result: pass/no-op.

Output summary:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: no work to do.
```

Build dir:

```bash
PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'
```

Result:

```text
/root/code/triton/build/cmake.linux-aarch64-cpython-3.12
```

## Checked-in lit sweep

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
lit -v \
  test/Conversion/lower_tensor_memory_to_llvm.mlir \
  test/Conversion/tritongpu_to_llvm_blackwell.mlir \
  test/Conversion/tritonnvidiagpu_to_llvm.mlir \
  test/Conversion/relayout_tritongpu.mlir \
  test/TritonGPU/proxy_fence_insertion.mlir \
  test/TritonGPU/memdesc-subview-split.mlir \
  test/TritonGPU/hoist-tmem-alloc.mlir \
  test/TritonGPU/promote-lhs-to-tmem.mlir \
  test/TritonNvidiaGPU/tmem_layouts.mlir \
  test/TritonNvidiaGPU/interleave_tmem.mlir \
  test/TritonNvidiaGPU/membar.mlir \
  test/TritonNvidiaGPU/membar-cluster.mlir \
  test/TritonNvidiaGPU/test_tensor_memory_allocation.mlir \
  test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir
```

Result:

```text
Total Discovered Tests: 14
  Passed: 13
  Failed: 1
```

Failure:

- `TRITON :: Conversion/relayout_tritongpu.mlir`
- Signature: assertion abort in
  `TritonGPUDialect::toLinearLayout` / `computeTMemLdStEncodingInfoImpl`
  from `dyn_cast on a non-existent value`.
- Classification: existing `FZ-20260421-0016` unencoded/non-distributed TMEM
  operand verifier crash.
- No new candidate from this sweep.

## Mbarrier/proxy/invalid diagnostics sweep

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
lit -v \
  test/TritonNvidiaGPU/mma_lowering.mlir \
  test/TritonNvidiaGPU/invalid.mlir \
  test/TritonNvidiaGPU/ops.mlir \
  test/TritonNvidiaGPU/tma_lowering.mlir \
  test/TritonGPU/invalid.mlir \
  test/TritonGPU/fence-inserstion.mlir \
  test/Analysis/test-membar-ttng.mlir \
  test/Conversion/tma_to_llvm.mlir \
  test/Conversion/tma_multicast_to_llvm.mlir
```

Result:

```text
Total Discovered Tests: 9
  Passed: 9
```

Classification: proxy/mbarrier insertion, invalid diagnostic guardrails, TMA
conversion, and NvidiaGPU operation verifier surfaces stayed green. No strict
verifier drift or new `FZ-*` candidate.

## Direct conversion probes

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt /root/code/triton/test/Conversion/tritongpu_to_llvm_blackwell.mlir \
  -split-input-file --convert-triton-gpu-to-llvm=compute-capability=100 -cse \
  >/tmp/tmem_round53_blackwell.out 2>/tmp/tmem_round53_blackwell.err
```

Result:

```text
exit=0
30805 /tmp/tmem_round53_blackwell.out
0     /tmp/tmem_round53_blackwell.err
```

Coverage: checked-in Blackwell conversion rows including TMEM `tc_gen5_mma`,
scaled MMAv5, `ttng.tmem_copy`, wait/commit barrier, two-CTA, scales-copy,
descriptor-view/indexed copy, and linear layouts.

Classification: pass. No conversion crash, unsupported diagnostic drift, or
new `FZ-*`.

## Minimized known repros

### Unencoded/non-distributed operand

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt /tmp/tmem_round51_fz0016_min.mlir \
  -split-input-file \
  -convert-triton-to-tritongpu='target=cuda:100 num-warps=4 enable-source-remat=true' \
  -relayout-tritongpu \
  >/tmp/tmem_round53_fz0016.out 2>/tmp/tmem_round53_fz0016.err
```

Result:

```text
exit=134
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

Minimized repro path: `/tmp/tmem_round51_fz0016_min.mlir`.

Classification: existing `FZ-20260421-0016`. No new candidate.

### i64 bitwidth

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt /tmp/tmem_round51_fz0017_min.mlir \
  --convert-triton-gpu-to-llvm=compute-capability=100 -cse \
  >/tmp/tmem_round53_fz0017.out 2>/tmp/tmem_round53_fz0017.err
```

Result:

```text
exit=134
Assertion `bitwidth == 32' failed.
```

Minimized repro path: `/tmp/tmem_round51_fz0017_min.mlir`.

Classification: existing `FZ-20260421-0017`. No new candidate.

### f64 bitwidth contrast

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
sed 's/xi64/xf64/g; s/@i64_chain_load_fz0017/@f64_chain_load_round53/g' \
  /tmp/tmem_round51_fz0017_min.mlir > /tmp/tmem_round53_f64_chain_load.mlir
bin/triton-opt /tmp/tmem_round53_f64_chain_load.mlir \
  --convert-triton-gpu-to-llvm=compute-capability=100 -cse \
  >/tmp/tmem_round53_f64.out 2>/tmp/tmem_round53_f64.err
```

Result:

```text
exit=134
Assertion `bitwidth == 32' failed.
```

Minimized repro path: `/tmp/tmem_round53_f64_chain_load.mlir`.

Classification: same owner surface as existing `FZ-20260421-0017`; f64 hits
the same non-32-bit TMEM load/store lowering assertion. No new independent
candidate.

### f32 control for the bitwidth repro shape

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
sed 's/xi64/xf32/g; s/@i64_chain_load_fz0017/@f32_chain_load_round53_control/g' \
  /tmp/tmem_round51_fz0017_min.mlir > /tmp/tmem_round53_f32_chain_load_control.mlir
bin/triton-opt /tmp/tmem_round53_f32_chain_load_control.mlir \
  --convert-triton-gpu-to-llvm=compute-capability=100 -cse \
  >/tmp/tmem_round53_f32.out 2>/tmp/tmem_round53_f32.err
```

Result:

```text
exit=1
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

Minimized repro path: `/tmp/tmem_round53_f32_chain_load_control.mlir`.

Classification: clean compiler diagnostic for the same indexed TMEM view-chain
shape, consistent with existing parent-index/view-chain lowering coverage
(`FZ-20260421-0001` family). It does not reproduce the bitwidth assertion and
is not a new candidate.

## Summary

- Required build: passed/no-op.
- `ninja triton-opt`: passed/no-op.
- Checked-in focused lit sweep: `13 passed, 1 failed`; the failure is existing
  `FZ-20260421-0016`.
- Mbarrier/proxy/invalid diagnostics sweep: `9 passed`.
- Direct Blackwell conversion probe: exit `0`.
- Known minimized repros:
  - `FZ-20260421-0016`: reproduced, exit `134`.
  - `FZ-20260421-0017`: reproduced for i64, exit `134`.
  - f64 contrast: same `FZ-20260421-0017` signature, exit `134`.
  - f32 control: clean illegal `ttg.memdesc_index` diagnostic, exit `1`,
    existing parent-index/view-chain surface.

No new independent `FZ-*` candidate was found in Round 53 compiler/lit lane B.
