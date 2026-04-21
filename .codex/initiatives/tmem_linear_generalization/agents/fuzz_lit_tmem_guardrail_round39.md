# Round 39 TMEM Lit Guardrail

Date: 2026-04-21 15:18 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Run a quick compiler-only lit guardrail over core TMEM layout and memdesc
subview tests after several runtime fuzzing checkpoints. This verifies that
the build-tree lit expectations remain green while backend repairs are
deferred.

## Commands

```bash
BUILD_DIR=$(PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
cd "$BUILD_DIR"
ninja triton-opt
lit -v \
  test/TritonNvidiaGPU/tmem_layouts.mlir \
  test/TritonNvidiaGPU/interleave_tmem.mlir \
  test/TritonGPU/memdesc-subview-split.mlir
```

## Results

- `ninja triton-opt`: no-op.
- Lit result: `3 passed`.

## Classification

No new independent `FZ-*` bucket was found.

The selected compiler-only TMEM lit tests remained green. There was no
unexpected IR/diagnostic drift in this sanity slice.

Backend repair remains deferred while discovery lanes continue.
