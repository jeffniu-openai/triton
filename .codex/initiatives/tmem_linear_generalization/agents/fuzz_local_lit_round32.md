# Round 32 Local: TMEM Lit / Compiler-Only Guardrail

Date: 2026-04-21

## Scope

This local compiler-only guardrail reran TMEM-adjacent lit tests while runtime
and subagent fuzzing continued.

## Command

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
lit -v test/TritonNvidiaGPU/tmem_layouts.mlir \
  test/TritonNvidiaGPU/invalid.mlir \
  test/Conversion/relayout_tritongpu.mlir
```

## Result

- `test/TritonNvidiaGPU/tmem_layouts.mlir`: pass.
- `test/TritonNvidiaGPU/invalid.mlir`: pass.
- `test/Conversion/relayout_tritongpu.mlir`: fail with the known
  `FZ-20260421-0016` verifier crash.

The failure signature is the existing unencoded-tensor verifier path:

```text
dyn_cast on a non-existent value
mlir::triton::gpu::TritonGPUDialect::toLinearLayout(...)
mlir::triton::nvidia_gpu::computeTMemLdStEncodingInfo(...)
mlir::triton::nvidia_gpu::verifyTMEMOperand(...)
mlir::triton::nvidia_gpu::TMEMAllocOp::verify()
```

## Classification

No new independent `FZ-*` bucket was found. This rerun reconfirms
`FZ-20260421-0016` in the existing conversion lit test while keeping the
TMEM NvidiaGPU layout/invalid lit baselines green.

