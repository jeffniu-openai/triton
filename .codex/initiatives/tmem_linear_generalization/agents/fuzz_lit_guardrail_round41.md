# Round 41 TMEM Lit Guardrail

Date: 2026-04-21 14:14 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Command

Build directory:

```bash
PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'
```

Result:

```text
/root/code/triton/build/cmake.linux-aarch64-cpython-3.12
```

Lit guardrail:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
lit -v test/TritonNvidiaGPU/tmem_layouts.mlir \
  test/TritonNvidiaGPU/interleave_tmem.mlir \
  test/TritonGPU/memdesc-subview-split.mlir
```

## Result

`ninja triton-opt` reported `no work to do`.

```text
PASS: TRITON :: TritonGPU/memdesc-subview-split.mlir
PASS: TRITON :: TritonNvidiaGPU/interleave_tmem.mlir
PASS: TRITON :: TritonNvidiaGPU/tmem_layouts.mlir
```

Aggregate: `3 passed`.

## Classification

No unexpected IR/diagnostic drift or new independent `FZ-*` bucket was
observed in these compiler-only TMEM layout/subview guardrails.
