# Round 24 Local: Compiler Lit Baseline

Date: 2026-04-21 13:30 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

This quick local compiler baseline reran a compact set of TMEM-related lit
tests while Round 24 subagents fuzzed the new verifier crash and mixed scaled
MMAv5 operand cases.

## Commands

Build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Lit baseline:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
lit -v \
  test/TritonNvidiaGPU/tmem_layouts.mlir \
  test/TritonNvidiaGPU/mma_lowering.mlir \
  test/TritonNvidiaGPU/interleave_tmem.mlir \
  test/TritonGPU/memdesc-subview-split.mlir
```

Result:

```text
PASS: TRITON :: TritonGPU/memdesc-subview-split.mlir
PASS: TRITON :: TritonNvidiaGPU/mma_lowering.mlir
PASS: TRITON :: TritonNvidiaGPU/interleave_tmem.mlir
PASS: TRITON :: TritonNvidiaGPU/tmem_layouts.mlir

Total Discovered Tests: 4
  Passed: 4
```

## Classification

No compiler-only regression or new `FZ-*` bucket was found in this compact lit
baseline.
