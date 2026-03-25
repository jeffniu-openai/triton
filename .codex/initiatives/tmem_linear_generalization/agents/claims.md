# Agent Claims

- `tmem_gluon_impl`
  - mode: explorer
  - scope: `python/triton/experimental/gluon/`, `python/src/gluon_ir.cc`
  - output: concrete frontend/pybind integration notes only
- `tmem_test_worker`
  - mode: worker
  - scope: `test/TritonNvidiaGPU/`, `test/Conversion/`, `python/test/gluon/`
  - output: targeted TMEM linear/view/lowering tests
- `ptx_tmem_research`
  - mode: explorer
  - scope: PTX/TMEM semantics, `tcgen05` instruction behavior, missing verifier/lowering semantics
  - output: concrete findings and follow-up recommendations only
- `tmem_lowering_design`
  - mode: explorer
  - scope: TMEM verifier/lowering stack in `lib/` and `third_party/nvidia/lib/`
  - output: exact function-level patch recommendations only

## Active / Completed 2026-03-25

- `/root/tmem_gluon_impl_v2`
  - mode: worker
  - scope: `python/src/gluon_ir.cc`,
    `python/triton/experimental/gluon/language/_semantic.py`,
    `python/triton/experimental/gluon/language/nvidia/blackwell/__init__.py`
  - status: completed
- `/root/tmem_test_worker_v2`
  - mode: worker
  - scope: `python/test/gluon/test_frontend.py`,
    `test/TritonNvidiaGPU/ops.mlir`,
    `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
  - status: completed
- `/root/ptx_tmem_research_v2`
  - mode: explorer
  - scope: PTX/TMEM semantics, PTXAS behavior, missing TMEM semantics
  - status: completed
- `/root/tmem_lowering_design_v2`
  - mode: explorer
  - scope: TMEM verifier/lowering/relayout patch map
  - status: completed
- `/root/tmem_mmav5_passes_worker`
  - mode: worker
  - scope: `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`,
    `lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp`,
    `lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp`,
    `lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp`
  - status: completed
- `/root/tmem_relayout_worker`
  - mode: worker
  - scope: `lib/Conversion/TritonToTritonGPU/RelayoutTritonGPU.cpp`,
    `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
  - status: completed
