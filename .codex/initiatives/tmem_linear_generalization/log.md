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

## 2026-03-25 (GPU2 runtime matrix expansion)
- Expanded TMEM runtime execution coverage in
  `python/test/gluon/test_core.py` with generated matrices:
  - `test_tmem_copy_no_scales_matrix` over `(M, N, BLOCK_N)` combinations
    for no-scales `tcgen05.cp`, with strict PTX/LLIR opcode and count checks.
  - `test_tmem_descriptor_chain_matrix` over linear TMEM layout families and
    instruction variants, exercising composed descriptor ops
    (`slice/index/permute/reshape/_reinterpret/load/store`) end-to-end.
- Added reusable kernels for the above runtime matrices:
  - `tmem_copy_no_scales_matrix_kernel`
  - `tmem_descriptor_chain_matrix_kernel`
- Runtime findings on `CUDA_VISIBLE_DEVICES=2`:
  - All no-scales matrix cases that fit shared memory emit
    `tcgen05.cp...128x256b`; `M=256,N=256` cases exceed SMEM and are skipped.
  - Descriptor-chain matrix for linear TMEM layouts is stable across tested
    variants, including m64 fallback behavior where `32x32b` lowers through
    `16x32bx2` shapes.
  - The existing blocked-layout fallback test still reproduces the known
    `relayout-tritongpu` failure on this branch (`PassManager::run failed`).
- Focused validation (GPU 2):
  - `181 passed, 5 skipped` for the expanded TMEM runtime slice (excluding the
    known blocked-layout fallback failure case).

## 2026-03-25 (dedicated TMEM runtime matrix file, GPU0)
- Added a dedicated Blackwell runtime matrix file:
  - `python/test/gluon/test_tmem_runtime_matrix.py`
- Coverage added in that file:
  - generated end-to-end `tcgen05.ld/st` matrix over linear TMEM identity and
    mixed layouts across `N in {64,128,256}` and
    `instr_variant in {auto,32x32b,16x64b,16x128b,16x256b}`
  - descriptor-composition matrix over `slice/index/permute/reshape/reinterpret`
    chains with runtime correctness checks
  - m64 splitn immediate matrix (`32x32b_splitn`) with explicit offset/immediate
    assertions
  - generated no-scales `tcgen05.cp` matrix over `(M,N,BLOCK_N)` combinations,
    with expected OOR cases skipped
  - explicit `tcgen05.cp` coverage for `128x128b` and
    scales `warpx4.32x128b`
  - focused `tcgen05.mma` runtime matrix over legacy and canonical-linear
    accumulator layouts with `use_acc` on/off
- Validation commands/results on `CUDA_VISIBLE_DEVICES=0`:
  - `TRITON_BUILD_WITH_CCACHE=true ninja -C build/cmake.linux-aarch64-cpython-3.12 -j96 triton`
    - `ninja: no work to do`
  - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py`
    - `86 passed, 5 skipped in 28.06s`

## 2026-03-25 (development PTX/TMEM probe script, GPU0)
- Added a development-only probe driver:
## 2026-03-25 (fuzz-plan checkpoint)
- Wrote down the current TMEM instruction inventory and an exact fuzzing plan
  in `experiments/tmem_fuzz_plan.md`.
- The plan now explicitly covers:
  - allocation / lifetime instructions
  - plain `ld/st`
  - `ld.red`
  - `cp`
  - `commit`
  - plain `mma`
  - `mma_scaled`
- The plan also fixes the execution model for the next implementation pass:
  - use GPU sharding across the 4 local Blackwell devices
  - classify every generated case as `PASS`, `CLEAN_UNSUPPORTED`, or `BUG`
  - always check PTX and LLIR opcode parity
  - reduce every new failure to a smallest permanent reproducer before landing a
    regression

## 2026-03-25 (exact TMEM fuzz plan)
- Added `.codex/initiatives/tmem_linear_generalization/fuzz_plan.md` as the
  execution playbook for TMEM codegen fuzzing.
- Captured the backend-emitted TMEM opcode inventory directly from source:
  alloc / relinquish / dealloc / commit / wait, `ld/st`, `ld.red`, `cp`,
  `mma`, and `mma_scaled`.
- Broke the fuzz plan down per instruction family with:
  - exact positive and negative parameter axes
  - compile-time legality filters
  - runtime numeric oracles
  - PTX/LLIR opcode checks
  - four-GPU sharding
- Immediate priorities recorded in the plan:
  - saturate `ld/st` higher-rank positive descriptor chains
  - exhaust `ld.red` runtime coverage
  - prove or refute reachable `cp.warpx2` families
  - expand `mma` beyond `f16`
  - expand `mma_scaled` beyond the current minimal path
## 2026-03-25 (exact TMEM fuzz execution plan)
- Captured the current TMEM instruction inventory and an explicit per-family
  fuzzing playbook in `.codex/initiatives/tmem_linear_generalization/fuzz_plan.md`.
- The plan now spells out:
  - the exact instruction families in tree (`alloc`, `relinquish`, `dealloc`,
    `ld/st`, `ld.red`, `cp`, `mma`, `mma_scaled`, `commit`);
  - generator axes for TMEM-linear layouts and descriptor view chains;
  - compile-only vs runtime vs negative-oracle classification;
  - GPU assignment across the four Blackwell devices; and
  - the rollout order for broadening coverage and fixing lowering gaps.
- The immediate implementation priority remains:
  1. broaden positive `ld/st` runtime sweeps;
  2. keep unsupported higher-rank `ld/st` cases as clean no-crash regressions
     until lowering support expands;
  3. exhaust `cp` atom coverage, especially `warpx2`;
  4. then expand `mma` / `mma_scaled`.
  - `experiments/tmem_instruction_probes.py`

## 2026-03-25 (fuzz plan checkpoint)
- Wrote a detailed per-instruction TMEM fuzz plan in
  `fuzz_plan.md`, covering:
  - allocator/lifetime ops (`alloc`, `relinquish_alloc_permit`, `dealloc`,
    `commit`, `wait`);
  - plain `ld/st`;
  - `ld.red`;
  - `cp`;
  - `mma`;
  - `mma_scaled`.
- Captured a three-layer validation strategy:
  - compile-only no-crash fuzz,
  - runtime functional fuzz on Blackwell,
  - direct PTX microprobes for doc/legalization ambiguities.
- Locked the common classification contract:
  - `PASS`
  - `CLEAN_UNSUPPORTED`
  - `BUG`
- Recorded GPU sharding and immediate implementation backlog so new runtime and
  LLVMIR tests can be expanded systematically instead of one-off.
  - `.codex/initiatives/tmem_linear_generalization/experiments/tmem_instruction_probes.py`
  - covers representative ld/st atom discovery, split-N immediate validation,
    cp family discovery, and baseline MMA runtime sanity

## 2026-03-25 (runtime-fuzz planning + MMAv5 two-CTA fix, GPU0)
- Wrote the per-instruction TMEM runtime fuzz plan into `memory.md`, splitting
  the space into:
  - general TMEM ld/st-legal layouts
  - MMAv5-legal canonical TMEM-linear families
  - `tcgen05.cp`-legal TMEM/shared families
  - reduction, MMA, and scaled-MMA specific legal frontiers
- Revalidated the existing two-CTA ld/st runtime surface on GPU0:
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_block_layouts`
    - `12 passed in 6.81s`
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_compositions`
    - `9 passed in 6.36s`
- Found a real positive-test bug in
  `python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit_twocta_linear_acc`:
  - the TMEM-linear “positive” case used `_make_tmem_linear_layout_block(...)`,
    which is ld/st-legal but not a canonical MMAv5-compatible two-CTA layout
  - verifier correctly rejected it as not MMAv5-compatible
- Confirmed the canonical MMAv5-linear equivalent of the legacy two-CTA
  accumulator tile by parsing:
  - `TensorMemoryLayout(block=(128, 128), col_stride=1, two_ctas=True, cga_layout=...)`
    with `ttgl.to_linear_layout(..., [256, 128])`
  - observed canonical linear form:
    `rows=[[1,0],[2,0],[4,0],[8,0],[16,0],[32,0],[64,0]]`,
    `cols=[[0,1],[0,2],[0,4],[0,8],[0,16],[0,32],[0,64]]`,
    `block_bases=[[128,0]]`,
    `two_ctas=True`
- Fixed `python/test/gluon/test_core.py`:
  - added `_make_tmem_linear_layout_mmav5_twocta(m, n)`
  - switched the two-CTA MMAv5 runtime test to use the canonical MMAv5-linear
    layout instead of the general block-basis ld/st layout
- Validation after the fix:
  - `python3 -m py_compile python/test/gluon/test_core.py`
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit_twocta_linear_acc`
    - `2 passed in 3.67s`

## 2026-03-25 (runtime sweep expansion + conversion lit strengthening, GPU0)
- Re-ran and expanded the executable TMEM runtime surface on GPU0:
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy`
    - `48 passed in 18.13s`
- Integrated `tmem_fuzz_inventory` runtime-matrix expansion in
  `python/test/gluon/test_tmem_runtime_matrix.py`:
  - added executable no-scales `tcgen05.cp` canonical-linear coverage for
    `128x128b` and `128x256b`
  - widened ld/st descriptor sweeps to include `16x256b`
  - widened 2-CTA ld/st sweeps to exercise `auto`
  - added executable two-CTA MMA coverage for legacy and canonical-linear
    accumulators
  - added a minimal scaled-MMA runtime test with exact PTX/LLIR opcode-family
    matching
- Main-thread validation of the expanded runtime matrix:
  - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py`
    - `128 passed, 5 skipped in 43.04s`
- Extended `python/test/gluon/test_core.py` reduction coverage:
  - refactored the reduction runtime helper to accept arbitrary TMEM layouts
  - kept the legacy positive matrix intact
  - added a positive canonical-linear identity reduction sweep
  - added a negative mixed TMEM-linear reduction test that locks the clean
    diagnostic for N-sharded layouts
- Validated reduction coverage after the refactor:
  - `python3 -m py_compile python/test/gluon/test_core.py`
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_core.py::test_tmem_reduction python/test/gluon/test_core.py::test_tmem_reduction_linear_layouts python/test/gluon/test_core.py::test_tmem_reduction_linear_mixed_reports_clean_error`
    - `67 passed in 21.43s`
- Strengthened Blackwell conversion lit coverage in
  `test/Conversion/tritongpu_to_llvm_blackwell.mlir`:
  - strengthened the existing 2-CTA plain MMA block to assert the actual
    `ttg.warp_id`, `nvg.cluster_id`, lead-CTA predicate, `nvvm.elect.sync`,
    and multicast-commit structure
  - added explicit `cp.cta_group::1.128x128b` coverage in both barriered and
    barrierless forms using a one-message `128x4xi32` TMEM-linear destination
    and representative shared/TMEM pointer math checks
  - added canonical TMEM-linear 2-CTA MMA lowering coverage
- Conversion validation:
  - `build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt test/Conversion/tritongpu_to_llvm_blackwell.mlir -split-input-file --convert-triton-gpu-to-llvm=compute-capability=100 -cse | /root/.triton/llvm/llvm-7f77ca0d-ubuntu-arm64/bin/FileCheck test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - passed

## 2026-03-25 (cp warpx2 probe dead-end, GPU3)
- `cp_warpx2_runtime_worker` completed a development-only multicast search:
  - candidate shared-layout probes still fail `ttng.tmem_copy` legalization
    before reaching a `warpx2` opcode
  - inspected converted linear layouts show the row-32 / row-64 bases never
    become zero, so the current candidates do not trigger multicast bits
  - the search artifacts live in:
    - `experiments/probe_cp_multicast_layouts.py`
    - `experiments/warpx2_layout_manual.py`
    - `experiments/warpx2_layout_linear.py`
- Follow-up remains open:
  - broaden the shared-layout search space until a legal `warpx2::02_13` or
    `warpx2::01_23` case is found, or lock the still-unreachable families as
    clean negatives if backend legality makes them impossible in-tree.
  - `.codex/initiatives/tmem_linear_generalization/experiments/tmem_instruction_probes.py`
- The probe script reuses Gluon TMEM runtime kernels and runs targeted
  compile+runtime checks for:
  - all currently targetable `tcgen05.ld/st` atom families
    (`32x32b`, `16x64b`, `16x128b`, `16x256b`, `16x32bx2`)
  - `16x32bx2` immediate behavior for both `secondHalfOffset=0` and a nonzero
    immediate (`16`)
  - practical `tcgen05.cp` shape coverage:
    `128x256b`, `128x128b`, and `warpx4.32x128b`
  - practical `tcgen05.mma` runtime smoke checks with legacy and canonical
    linear accumulator layouts
- Commands run:
  - `python3 -m py_compile .codex/initiatives/tmem_linear_generalization/experiments/tmem_instruction_probes.py`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python3 .codex/initiatives/tmem_linear_generalization/experiments/tmem_instruction_probes.py`
- Probe outcomes:
  - observed `ld/st` families exactly:
    `16x128b`, `16x256b`, `16x32bx2`, `16x64b`, `32x32b`
  - splitn probes produced:
    - `n=2`: `tcgen05.{st,ld}.sync.aligned.16x32bx2.x2.b32 ... , 0`
    - `n=64`: `tcgen05.{st,ld}.sync.aligned.16x32bx2.x16.b32 ... , 16`
  - `cp` probes produced:
    - 32x `tcgen05.cp.cta_group::1.128x256b`
    - 1x `tcgen05.cp.cta_group::1.128x128b`
    - 2x `tcgen05.cp.cta_group::1.warpx4.32x128b`
  - `mma` probes executed successfully and emitted
    `tcgen05.mma.cta_group::1.kind::f16` in both PTX and LLIR for both tested
    accumulator layout modes

## 2026-03-25 (runtime/frontier sweep, relayout no-crash fix, GPU0)
- Revalidated the broad TMEM runtime matrix after the latest relayout work:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py::test_tmem_linear_roundtrip_blocked_fallback`
  - result: `91 passed, 5 skipped in 31.55s`
- Confirmed that the blocked-register fallback path is now genuinely working
  end to end:
  - it executes correctly on GPU
  - final PTX/LLIR contain the expected `tcgen05.st/ld.sync.aligned.32x32b`
    pair
  - the old `ttgir.count("ttg.convert_layout")` assertion was stale because
    the visible `ttgir` dump is pre-relayout
- Added execution coverage updates in
  `python/test/gluon/test_tmem_runtime_matrix.py`:
  - kept the positive matrix green
  - added a negative no-crash compiler test for unsupported block-bases TMEM
    descriptor views, asserting on the emitted diagnostic text rather than the
    generic `PassManager::run failed` exception wrapper
- Fixed a compiler no-crash bug in relayout:
  - file: `lib/Conversion/TritonToTritonGPU/RelayoutTritonGPU.cpp`
  - previous behavior: `getDefaultLayoutForTmemLdSt(...)` asserted when no
    compatible TMEM register layout existed for the descriptor view
  - new behavior: relayout consults `getTmemCompatibleLayouts(...)` and emits
    an op diagnostic with notes when the set is empty
- Newly confirmed unsupported frontier:
  - TMEM block-bases descriptor chains that reinterpret to a
    `64x32` TMEM-linear memdesc still have no TMEM-compatible register layout
    for plain `tmem_load/store`
  - the compiler now reports this cleanly:
    - `TMEM layout has no supported register layout`
    - note with the tensor type
    - note with the TMEM memdesc type
    - note suggesting reshape/permute or an explicit supported register layout

## 2026-03-25 (current TMEM fuzz axes)
- `tcgen05.ld/st`
  - execute: all current atom families
    - `32x32b`
    - `16x64b`
    - `16x128b`
    - `16x256b`
    - `16x32bx2`
  - sweep dimensions:
    - element path: `f32`, packed/unpacked `f16`, software-padded narrow types
      where codegen exists
    - TMEM linear family: identity, mixed, m64/splitn, legacy-equivalent
      layouts
    - logical view chain: direct, `slice`, `index`, `permute`, `reshape`,
      `_reinterpret`, blocked-register fallback
    - launch shape: `num_warps in {4, 8}` where legal
    - CGA/block axis: plain 1-CTA positive coverage, block-bases unsupported
      cases kept as negative diagnostic coverage until lowering is expanded
  - validate:
    - runtime equality on GPU
    - PTX opcode family/offsets/immediates exactly match LLIR
    - for fallback paths, confirm the shared-memory conversion path is present
- `tcgen05.cp`
  - execute:
    - no-scales `128x256b`
    - no-scales `128x128b`
    - scales `warpx4.32x128b`
    - scaled-MMA paths already exercise scale copies in larger end-to-end tests
  - sweep dimensions:
    - source element type / descriptor family
    - `(M, N, BLOCK_N)` tilings
    - CTA group `{1, 2}`
    - multicast family `{none, warpx2::02_13, warpx2::01_23, warpx4}`
    - barrier / no-barrier emission
    - subview/sliced shared descriptors
  - validate:
    - runtime equality
    - exact cp opcode multiset in PTX and LLIR
    - expected commit emission or omission
  - current gap:
    - direct runtime discovery/coverage for the two `warpx2` multicast forms
      still needs a dedicated shared-layout search/probe
- `tcgen05.mma`
  - execute:
    - existing end-to-end tests cover broad MMAv5 surfaces in
      `python/test/gluon/test_core.py`
    - dedicated runtime matrix covers canonical linear vs legacy accumulator
      layouts and `use_acc in {false, true}`
  - sweep dimensions:
    - accumulator TMEM family `{legacy, canonical linear}`
    - `use_acc`
    - multicast / barrier counts
    - 1-CTA vs 2-CTA cluster modes
    - operand swizzles / operand source memory
  - validate:
    - numeric equality against dense reference
    - exact `tcgen05.mma` emission in PTX and LLIR
- `tcgen05.mma_scaled`
  - execute:
    - large end-to-end scaled-MMA-copy matrix in `python/test/gluon/test_core.py`
      over format pairs, block sizes, `num_ctas in {1, 2}`, and multicast
  - sweep dimensions:
    - format pairs `{mxfp8/mxfp8, nvfp4/nvfp4, mxfp8/mxfp4}`
    - `BLOCK_N`, `BLOCK_K`
    - CTA group / multicast
    - scale-copy opcode family
  - validate:
    - numeric equality against dequantized reference
    - emitted scale-copy and MMA op families from compiled asm
- `tcgen05.cp.warpx2`
  - execute/record:
    - ran `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_multicast_layouts.py`
      on GPU3; each candidate layout failed during `tcgen05_copy` legalization because the
      shared descriptor could not be mapped to a `TensorMemoryScalesLayout`, and no `warpx2`
      opcode was emitted.
    - added `warpx2_layout_manual.py` to retry the subset of insertion positions that should
      target the `.warpx2::02_13` and `.warpx2::01_23` families; all runs still hit the same
      `tcgen05_copy` legalization failures, so no opcode families were captured.
    - added `warpx2_layout_linear.py` to introspect the converted linear layout directly;
      the computed row basis values at `log2(32)+{0,1}` stayed non-zero for every candidate,
      so `multicast = 0` even when the special row distortion was interleaved into the shared
      descriptor.
  - next steps:
    - widen the shared-layout search (e.g., allow additional column bases or block reorderings)
      so that the `tcgen05_copy` legalization can succeed, then re-run the probe to capture the
      actual `warpx2` opcodes for both CTA layouts.

2026-03-25 (warpx2 executable reachability refresh)
- GPU probes run on `CUDA_VISIBLE_DEVICES=0` with `TRITON_REPRODUCER_PATH=/tmp/tmem_repro`.
- Direct scales-copy search (`64x16 -> 128x32`) over all 1260 surjective interleavings of
  row bases `{1,2,4,8,16,32}` and col bases `{1,2,4,8}`:
  - only 2 layouts legalized and both emitted
    `tcgen05.cp.cta_group::1.warpx4.32x128b` (twice).
  - no `warpx2` opcodes observed.
- No-scales copy search (`128x4 i32`) over all surjective interleavings of row bases
  `{1,2,4,8,16,32,64}` and col bases `{1,2}`:
  - exactly 1 layout legalized, emitting `tcgen05.cp.cta_group::1.128x128b`.
  - no `warpx2` opcodes observed.
- Subslice search via new
  `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_warpx2_subslice.py`:
  - `parent_rows=64`: 2 legal layouts, all `warpx4`.
  - `parent_rows in {128,256}` with `start_row=0`: 0 legal layouts in the sampled sets.
  - non-zero starts also hit frontend slice constraints (`The split offset may not touch the tile`)
    for many layouts before lowering.
- Mixed-basis perturbations (e.g. replacing `[32,0]` by `[32,1|2|4|8]`) were accepted by
  frontend parsing but consistently failed in TMEM copy legalization with:
  `failed to find valid tcgen05.copy layout ...`.
- Conclusion:
  - no documented `tcgen05.cp.warpx2::{02_13,01_23}` family was executable in current tree
    under these probe spaces.
  - observed blockers are legalization/descriptorizability constraints in
    `copySharedToTmem`/`DotOpMmaSmemLoader`, not PTX emission itself.
