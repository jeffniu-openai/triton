You are taking over the TMEM linear generalization effort in `/root/code/triton`.

Start by reading:
- `/root/code/triton/.codex/initiatives/tmem_linear_generalization/handoff_2026-03-26.md`
- `/root/code/triton/.codex/initiatives/tmem_linear_generalization/memory.md`
- `/root/code/triton/.codex/initiatives/tmem_linear_generalization/log.md`

Current repo state:
- branch: `main`
- HEAD: `a92c764c5`
- worktree: clean

What is already landed:
- TMEM memdesc view/type inference moved to op-level `InferTypeOpInterface` + dialect-interface dispatch.
- `gluon_ir.cc` is now a thin checked-builder wrapper.
- TMEM canonicalization is centralized via `getCanonicalTMemLinearEncoding(...)`.
- Plain TMEM `load` / `store` / initialized `alloc` reject incompatible explicit register layouts; no silent backend `convert_layout` fallback.
- Public TMEM slicing uses generic `ttg.memdesc_subslice`; `ttng.tmem_subslice` remains internal/backend-only for now.

Main open issue:
- The full TMEM runtime matrix currently gives `1557 passed, 117 skipped, 49 xfailed`.
- Those 49 xfails are intentional `BUG` markers for multidimensional TMEM slice/view miscompiles.
- The likely bug area is `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`, especially logical view to physical TMEM mapping across higher-rank `memdesc_subslice` / `memdesc_index` / reshape chains.

Other open issue:
- `tcgen05.copy.warpx2` is still intentionally unsupported. PTX probes show it is not an opcode-selection-only gap; do not enable it casually.

Important constraints from the user:
- Use subagents liberally for bounded tasks, including PTX probing and test work.
- Keep logging findings in the initiative directory.
- Before tests, rebuild with ccache:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
- Use pytest with `-s --tb=short`, and use `-n` plus `CUDA_VISIBLE_DEVICES` to spread load when running GPU tests.
- If a theoretically legal case fails, mark/log it as `BUG`.
- Parser and verifier must never crash.
- Do not reintroduce silent layout-repair for plain TMEM ld/st/alloc.
- Leave `RelayoutTritonGPU.cpp` alone.

Suggested next task:
1. Reproduce and fix the multidimensional TMEM slice/view bug bucket.
2. Convert the `BUG` xfails in `python/test/gluon/test_tmem_runtime_matrix.py` into real passing runtime checks.
3. Re-run the full TMEM runtime matrix.
4. Triage the current `test/TritonNvidiaGPU/ops.mlir` lit issue where raw `triton-opt` emits empty stdout.

Relevant files:
- `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
- `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
- `lib/Dialect/TritonGPU/IR/Ops.cpp`
- `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
- `python/test/gluon/test_tmem_runtime_matrix.py`
- `python/test/gluon/test_frontend.py`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`

Known BUG test markers:
- `python/test/gluon/test_tmem_runtime_matrix.py:2006`
- `python/test/gluon/test_tmem_runtime_matrix.py:2068`
- `python/test/gluon/test_tmem_runtime_matrix.py:2155`

Do not spend time re-deriving already-landed structural refactors unless the handoff summary says they are still open.
