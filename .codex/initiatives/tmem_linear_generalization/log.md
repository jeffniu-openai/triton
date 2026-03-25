# TMEM Linear Generalization Log

## 2026-03-24
- Created initiative workspace.
- Captured baseline decisions from planning:
  - canonical TMEM-linear encoding
  - normalize legacy TMEM encodings before semantic use
  - shared-style TMEM descriptor API
  - remove `immHalfSplitOffset == 0` workaround
- Reverted unrelated local test edits per user direction before continuing implementation.
- Began canonical attr/type work:
  - added `TensorMemoryLinearEncodingAttr`
  - added canonical TMEM encoding helpers in the NVIDIA dialect
  - routed TMEM linear layout conversion through canonical TMEM-linear normalization
  - relaxed `MemDescType` TMEM verification to use canonical TMEM linear layouts
- Current implementation split:
  - main thread: C++ verifier/lowering integration, TMEM generic view semantics, incremental builds
  - `tmem_gluon_impl`: frontend/Gluon surface integration
  - `tmem_test_worker`: targeted lit/frontend tests
  - `ptx_tmem_research`: PTX/TMEM semantic validation and gap finding
  - `tmem_lowering_design`: verifier/lowering patch map
- Build instruction captured from `README.md`: set `TRITON_BUILD_WITH_CCACHE=true` for incremental rebuilds.

## 2026-03-25
- Verified the README/CMake/active build directory all agree that `ccache` is
  enabled:
  - `README.md` documents `TRITON_BUILD_WITH_CCACHE=true`
  - `CMakeLists.txt` wires `CMAKE_{C,CXX}_COMPILER_LAUNCHER=ccache`
  - active `CMakeCache.txt` records `/usr/bin/ccache`
  - observed compile commands were launched through `/usr/bin/ccache`
- Reproduced and fixed the canonical TMEM conversion recursion:
  - `canonicalizeLegacyTensorMemoryLayout(...)` now calls the direct legacy
    TMEM linear-layout converter instead of recursing through the public
    canonicalization path
- Landed canonical TMEM-linear attr/type/helper work and verifier integration:
  - added `TensorMemoryLinearEncodingAttr`
  - added canonical TMEM helper entry points in the NVIDIA dialect
  - normalized TMEM memdesc/type verification on canonical TMEM-linear layouts
  - generalized `getTmemAllocSizes(...)` and TMEM view-offset helpers
- Generalized TMEM memdesc views:
  - `ttg.memdesc_subslice`, `ttg.memdesc_index`, `ttg.memdesc_trans`,
    `ttg.memdesc_reshape`, and `ttg.memdesc_reinterpret` now accept TMEM
    descriptors in the intended cases
  - LLVM lowering treats TMEM `trans/reshape/reinterpret` as pointer no-ops
  - TMEM `subslice/index` lowering uses the TMEM pointer-offset helper
- Reworked `ttng.tmem_subslice` to preserve canonical physical TMEM encoding
  and the original allocation metadata instead of trying to re-encode the
  smaller logical result shape
- Removed the stale `immHalfSplitOffset == 0` workaround from TMEM ld/st
  lowering and updated the Blackwell conversion regression to expect literal
  `0`
- Relaxed plain TMEM verifier behavior:
  - `ttng.tmem_load`, `ttng.tmem_store`, and source-initialized
    `ttng.tmem_alloc` no longer reject non-default register layouts up front
  - layout feasibility is deferred to relayout/lowering
- Integrated subagent work:
  - `tmem_gluon_impl_v2`
    - made `TensorMemoryLinearLayout` first-class in Gluon/Python
    - canonicalized legacy `TensorMemoryLayout` through TMEM-linear attr
    - aligned TMEM descriptor API with shared memdesc slice/index/permute/
      reshape/reinterpret semantics
  - `tmem_test_worker_v2`
    - added lit/frontend coverage for TMEM-linear round-trips and generic
      TMEM views
  - `ptx_tmem_research_v2`
    - validated that `immHalfSplitOffset = 0` is safe with the in-tree
      Blackwell PTXAS assumptions
    - identified MMAv5/pass canonicalization and analysis/view-lowering gaps
  - `tmem_lowering_design_v2`
    - mapped the exact relayout/LLVM patch points for one-convert fallback
  - `tmem_mmav5_passes_worker`
    - normalized MMAv5 and TMEM transform code paths onto canonical TMEM
      helpers in `MMAv5.cpp`, `TensorMemoryAllocation.cpp`,
      `PromoteLHSToTMem.cpp`, and `OptimizeTMemLayouts.cpp`
  - `tmem_relayout_worker`
    - updated relayout to prefer direct TMEM-compatible layouts and use only a
      one-convert fallback when needed
    - widened NVIDIA TMEM memdesc-index lowering to accept canonical TMEM
      linear encodings
- Incremental validation performed:
  - `TRITON_BUILD_WITH_CCACHE=true make triton-opt`
  - `TRITON_BUILD_WITH_CCACHE=true make -j4`
    - still fails in unrelated GSan runtime build:
      `python/triton/experimental/gsan/src/GSanLibrary.cu` cannot find
      `<climits>` during CUDA device compilation
  - `python3 -m py_compile python/triton/experimental/gluon/language/_semantic.py python/triton/experimental/gluon/language/nvidia/blackwell/__init__.py python/test/gluon/test_frontend.py`
  - passing lit set:
    - `test/Conversion/relayout_tritongpu.mlir`
    - `test/Conversion/lower_tensor_memory_to_llvm.mlir`
    - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `test/TritonNvidiaGPU/ops.mlir`
    - `test/TritonNvidiaGPU/invalid.mlir`
    - `test/TritonNvidiaGPU/tmem_layouts.mlir`
    - `test/Analysis/test-buffer-region.mlir`
    - `test/TritonGPU/nvidia-fpsan.mlir`
- Frontend runtime validation remains blocked by the unrelated GSan-linked
  `libtriton.so` build dependency in this environment.

## 2026-03-25 (lit expansion checkpoint)
- Expanded TMEM lit coverage (tests only) for canonical-linear/weird-view
  combinations, invalid parser/verifier non-crash checks, and LLVM lowering
  pointer/asm checks:
  - `test/TritonNvidiaGPU/ops.mlir`
    - added a larger TMEM-linear view chain combining
      `memdesc_reshape/trans/subslice/reinterpret` with `tmem_store/load`
  - `test/TritonNvidiaGPU/tmem_layouts.mlir`
    - added optimize-pass regression for transpose/reshape TMEM-linear views,
      checking canonical simplification (`memdesc_trans` dropped) and direct
      `tmem_subslice + tmem_load`
  - `test/TritonNvidiaGPU/invalid.mlir`
    - added malformed TMEM-linear parser case (`row` missing)
    - added MMAv5-scaled rejection for canonical-linear layout with invalid
      CGA (2 CTAs per CGA in a 1-CTA context)
  - `test/TritonGPU/invalid.mlir`
    - added generic TMEM memdesc invalids:
      `memdesc_subslice` layout-encoding mismatch and
      `memdesc_reshape` element-count mismatch
  - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - added LLVM checks for TMEM-linear `memdesc_trans` pointer no-op
    - added LLVM checks for `index + trans + trans + tmem_subslice` exact
      pointer math (`*128 + 64`)
  - `test/Analysis/test-buffer-region.mlir`
    - added TMEM-linear transpose-chain buffer-region regression, validating
      exact tensor region `[64, 128]`
- Reconfirmed second-half offset behavior remains tested as literal zero:
  `tcgen05.st.sync.aligned.16x32bx2... , 0` (no legacy rewrite to `1`).
- Validation run:
  - `TRITON_BUILD_WITH_CCACHE=true make -j8`
  - targeted lit checks:
    - `test/TritonNvidiaGPU/ops.mlir`
    - `test/TritonNvidiaGPU/tmem_layouts.mlir`
    - `test/TritonNvidiaGPU/invalid.mlir`
    - `test/TritonGPU/invalid.mlir`
    - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `test/Analysis/test-buffer-region.mlir`

## 2026-03-25 (additional coverage pass)
- Added more TMEM-linear lit coverage in existing tests:
  - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - canonical `tensor_memory_linear` MMAv5 lowering test
      (`@tc_gen5_mma_tmem_linear`) with PTX-level checks for
      `tcgen05.mma...kind::f16` and `tcgen05.commit...`
  - `test/TritonNvidiaGPU/ops.mlir`
    - op-level canonical-linear MMAv5 acceptance test
      (`@tcgen5_tmem_linear_acc`)
  - `test/TritonNvidiaGPU/invalid.mlir`
    - negative verifier case for `twoCTAs` TMEM-linear load with
      incompatible 1-CTA result layout (`Result has an invalid layout`)
  - `test/TritonGPU/invalid.mlir`
    - negative generic TMEM view test for invalid `memdesc_trans` result
      type inference (expects both incompatibility and inference failure)
- Re-ran focused lit slice after edits and all targeted checks passed.

## 2026-03-25 (parser hardening + broader frontend matrix)
- Investigated a real parser/verifier crash on a two-CTA MMAv5 frontend
  repro:
  - Python reproducer:
    `run_parser(tcgen05_mma_linear_acc_kernel, ..., num_ctas=2)` with
    `_make_tmem_linear_layout_128_twoctas()` and operand-specific
    `_make_tcgen05_shared_layout(2, operand)`.
  - Native backtrace showed the fault in
    `triton::gpu::tensorMemoryToLinearLayout(...)` from
    `matchTensorMemoryLegacyEncoding(...)`, where the `blockM=64`,
    `twoCTAs=true`, `blockN=1` candidate path dereferenced an empty column
    basis.
- Hardened the legacy TMEM matcher/converter:
  - `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`
    - guard the `isM64TwoCTA` remap against empty `col` bases and return a
      structured error instead of crashing.
  - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
    - skip the impossible `blockM=64`, `twoCTAs`, `blockN=1` legacy candidate
      while matching canonical linear TMEM layouts back to legacy MMAv5
      families.
- Frontend test matrix updates:
  - `python/test/gluon/test_frontend.py`
    - added CGA-aware blocked register layouts for TMEM view tests
    - aligned block-based TMEM-linear variants with `num_ctas=2`
    - removed the invalid positive `reinterpret-block` case
    - split MMAv5 tests into:
      - positive canonical 1-CTA compile
      - negative 2-CTA context mismatch
      - negative unsupported MMAv5 families (`mixed`, `block`, `two_ctas`)
    - asserted detailed diagnostics via `capfd` where the Python wrapper only
      raises generic `RuntimeError("error encountered during parsing")`
  - `python/test/gluon/test_core.py`
  - `python/test/gluon/test_fpsan.py`
    - mirrored the CGA-aware TMEM view-layout helpers and removed the invalid
      positive `reinterpret-block` case
- Validation run after the crash fix:
  - `TRITON_BUILD_WITH_CCACHE=true ninja -C build/cmake.linux-aarch64-cpython-3.12 -j96 triton-opt triton`
  - focused frontend compile tests:
    - `python/test/gluon/test_frontend.py::test_tensor_memory_linear_views_layout_matrix`
    - `python/test/gluon/test_frontend.py::test_tensor_memory_descriptor_chain_ir`
    - `python/test/gluon/test_frontend.py::test_tensor_memory_descriptor_chain_variants`
    - `python/test/gluon/test_frontend.py::test_tensor_memory_descriptor_chain_reports_two_ctas_mismatch`
    - `python/test/gluon/test_frontend.py::test_tensor_memory_linear_layout_invalid_shape`
    - `python/test/gluon/test_frontend.py::test_tensor_memory_linear_mma_compile`
    - `python/test/gluon/test_frontend.py::test_tensor_memory_linear_mma_compile_reports_two_ctas_mismatch`
    - `python/test/gluon/test_frontend.py::test_tensor_memory_linear_mma_compile_reports_unsupported_layout`
    - `python/test/gluon/test_core.py::test_tmem_linear_view_ops_compile_ir`
    - `python/test/gluon/test_fpsan.py::test_tmem_linear_view_ops_compile_ir`
  - focused lit checks:
    - `test/TritonNvidiaGPU/ops.mlir`
    - `test/TritonNvidiaGPU/tmem_layouts.mlir`
    - `test/TritonNvidiaGPU/invalid.mlir`
    - `test/TritonNvidiaGPU/test_tensor_memory_allocation.mlir`
    - `test/TritonGPU/invalid.mlir`
    - `test/TritonGPU/nvidia-fpsan.mlir`
    - `test/Analysis/test-buffer-region.mlir`
    - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
