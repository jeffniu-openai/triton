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

## 2026-03-25 (cp runtime expansion + warpx2 probe refresh, GPU2)
- Added executable two-CTA `tcgen05.cp` runtime coverage in
  `python/test/gluon/test_tmem_runtime_matrix.py` by adding:
  - `test_tmem_runtime_matrix_cp_scales_warpx4_twocta_via_scaled_mma_copy`
    (uses proven scaled-MMA copy path, checks exact PTX/LLIR cp opcode family).
- Retained existing direct scales-copy runtime check:
  - `test_tmem_runtime_matrix_cp_scales_warpx4` (`cta_group::1.warpx4.32x128b`).
- Validation on `CUDA_VISIBLE_DEVICES=2` with private cache:
  - `python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4 python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_twocta_via_scaled_mma_copy`
    - `2 passed`
  - `python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k "cp_"`
    - `43 passed, 5 skipped`
- Re-ran a bounded warpx2 subslice search:
  - command:
    - `python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_warpx2_subslice.py --parent-rows 128 --max-layouts 96 --start-rows 0`
  - outcome:
    - `successful_compiles=0`
    - `no warpx2 opcodes observed`
    - failures are consistent with TMEM copy legalization failure at
      `ttng.tmem_copy` (`failed to find valid tcgen05.copy layout ...`).
- BUG tracking:
  - no parser/verifier/pass crash or hard assertion observed in this run.
  - failures observed in this checkpoint are classified as `CLEAN_UNSUPPORTED`.

## 2026-03-25 (BUG: NVGPU->LLVM TMEM lifecycle crash from lit)
- Focused lit run:
  - `lit -v build/cmake.linux-aarch64-cpython-3.12/test/Conversion/lower_tensor_memory_to_llvm.mlir`
- Reproducer command:
  - `build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt test/Conversion/lower_tensor_memory_to_llvm.mlir --convert-warp-specialize-to-llvm --convert-nv-gpu-to-llvm -allow-unregistered-dialect`
- Observed behavior:
  - compiler crashes (segfault) before producing output; `FileCheck` then reports
    empty stdin.
  - stack points to integer-attr cast in TMEM allocation lowering:
    - `third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp:607` (`initTensorMemory`)
    - `third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp:635` (`lowerTensorMemoryAlloc`)
    - `ConvertNVGPUToLLVM::runOnOperation`
- Classification:
  - `BUG` (compiler crash in pass pipeline).
- Notes:
  - this crash is independent of FileCheck expectations and reproduces with
    `triton-opt` alone.

## 2026-03-25 (BUG fixes: higher-rank TMEM no-crash + NVGPU alloc lowering)
- Fixed BUG: frontend/backend TMEM layout contract mismatch for higher-rank
  broadcasted subviews.
  - Reproducer before fix:
    - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_index[identity-256-16x64b-16x64b.x64.b32-16x64b.x64.b32]'`
  - Old behavior:
    - frontend `get_reg_layout(...)` returned a layout that later failed in
      `ConvertTritonGPUToLLVM` with `unsupported register broadcast pattern for
      direct lowering`.
  - Fix:
    - `python/src/gluon_ir.cc` now validates `compute_tmem_reg_layout` against
      `computeTMemLdStEncodingInfo(...)` before returning a TMEM register
      layout to Python.
    - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` now filters
      `getTmemCompatibleLayouts(...)` through the actual TMEM compatibility
      check so relayout/fallback paths do not pick lowering-invalid layouts.
  - New behavior:
    - these cases fail early and cleanly at `get_reg_layout(...)` with the
      actionable unsupported-layout diagnostic.
- Fixed BUG: higher-rank TMEM scratch-size analysis crash in
  `AllocateSharedMemoryNv`.
  - Reproducer before fix:
    - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_index[identity-128-32x32b-32x32b.x128.b32-32x32b.x64.b32]'`
  - Old behavior:
    - `LinearLayout::invertAndCompose` assertion in shared-memory scratch-size
      analysis during `AllocateSharedMemoryNv`.
  - Fix:
    - tightened `canInvertAndComposeLayouts(...)` guards in
      `lib/Analysis/Allocation.cpp` and
      `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp` so they
      match the real `invertAndCompose(...)` preconditions.
  - New behavior:
    - the higher-rank runtime matrix no longer crashes the pass pipeline.
- Fixed BUG: NVGPU->LLVM TMEM lifecycle lowering crashed when `ttg.shared` was
  absent.
  - Reproducer before fix:
    - `build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt test/Conversion/lower_tensor_memory_to_llvm.mlir --convert-warp-specialize-to-llvm --convert-nv-gpu-to-llvm -allow-unregistered-dialect`
  - Old behavior:
    - segfault in `third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp`
      from `cast<IntegerAttr>(mod->getAttr("ttg.shared"))`.
  - Fix:
    - default missing `ttg.shared` to zero in `initTensorMemory(...)`.
  - New behavior:
    - the reproducer runs successfully and emits the expected TMEM lifecycle
      opcodes.
- Validation after fixes:
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py`
    - `189 passed, 5 skipped`
  - `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_core.py::test_tmem_reduction_linear_layouts python/test/gluon/test_core.py::test_tmem_reduction_linear_reports_clean_error`
    - `35 passed`
  - `build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt test/Conversion/lower_tensor_memory_to_llvm.mlir --convert-warp-specialize-to-llvm --convert-nv-gpu-to-llvm -allow-unregistered-dialect`
    - succeeds and emits `alloc`, `relinquish_alloc_permit`, and `dealloc`
      instead of crashing.

## 2026-03-25 (runtime sweep expansion: higher-rank positives + clean negatives)
- Extended GPU-executed TMEM runtime matrix in
  `python/test/gluon/test_tmem_runtime_matrix.py`:
  - added two executable higher-rank descriptor compositions that are
    currently codegenable:
    - `reshape((2, M, N/2)) -> slice(dim=0) -> index`
    - `reshape((2, M/2, N)) -> slice(dim=0) -> index`
  - added broad sweeps for these compositions over:
    - 1-CTA identity TMEM-linear layouts (`N in {64,128,256}`)
    - 2-CTA block-basis TMEM-linear layouts (`N in {64,128,256}`)
    - ld/st variants `{32x32b,16x64b,16x128b,16x256b}`
  - each case executes on GPU and checks:
    - numerical result (`+7` or `+13`)
    - PTX/LLIR opcode parity and expected atom family
    - expected TMEM view ops in TTGIR.
- Added clean-negative runtime coverage:
  - MMAv5 2-CTA higher-rank descriptor composition now asserted as clean
    compile failure (`failed to infer tensor memory encoding for memdesc_index`)
    with no pass-manager/assert crash text.
  - unsupported linear-copy shape (`M=256, N=128`) now asserted as clean
    unsupported TMEM-layout diagnostic.
- Validation on `CUDA_VISIBLE_DEVICES=2`:
  - focused new cases:
    - `python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k "higher_rank_dim0_slice_positive or higher_rank_half_rows_positive or mmav5_descriptor_higher_rank_reports_clean_error or cp_no_scales_linear_unsupported_shape_reports_clean_error"`
    - `61 passed, 194 deselected`
  - full TMEM runtime matrix:
    - `python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py`
    - `250 passed, 5 skipped`

## 2026-03-25 (BUG: cp mixed-layout failure currently surfaces as parse RuntimeError)
- Reproducer:
  - `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 - <<'PY' ... tmem_copy_no_scales_linear_kernel(..., layout=_make_tmem_linear_layout_mixed(128,128), ...) ... PY`
- Observed behavior:
  - diagnostic is printed first:
    - `'ttng.tmem_copy' op Incorrect tmem layout.`
  - frontend then raises:
    - `RuntimeError: error encountered during parsing`
    - instead of a structured `CompilationError` carrying the same diagnostic.
- Classification:
  - `BUG` (error-reporting path quality regression; parser-path runtime error
    for an unsupported layout).
- Notes:
  - this is cleanly non-crashing, but it violates the desired contract for
    unsupported layouts to report a precise compile error.

## 2026-03-25 (higher-rank TMEM views: crash fixed, semantics still under probe)
- Fixed BUG: TMEM `memdesc_reshape(memdesc_subslice(...))` no longer crashes
  the compiler in linear-layout reshape.
  - Reproducer before fix:
    - higher-rank TMEM descriptor compositions that reshape a TMEM memdesc,
      slice a non-last logical dimension, then index.
  - Old behavior:
    - assertion in `LinearLayout::reshapeOuts` reached through TMEM view
      encoding inference.
  - Fix:
    - TMEM view inference now rebuilds view-local encodings through the refined
      TMEM helpers in:
      - `lib/Dialect/TritonGPU/IR/Ops.cpp`
      - `python/src/gluon_ir.cc`
      - `python/triton/experimental/gluon/language/_semantic.py`
  - New behavior:
    - the prior reproducer no longer crashes; unsupported layouts now fail with
      the standard actionable TMEM-layout diagnostic.
- Fixed BUG: shared memdesc slicing regression introduced while plumbing TMEM
  view-local `alloc_shape`.
  - Old behavior:
    - the builder started narrowing `alloc_shape` for all memdescs, which broke
      shared-mem descriptor slicing semantics.
  - Fix:
    - view-local `alloc_shape` narrowing is now TMEM-only in
      `python/src/gluon_ir.cc`.
- BUG / open semantics question: one-CTA higher-rank TMEM view compositions
  currently execute, but appear to alias the entire tensor instead of only the
  selected subview.
  - Reproducers in:
    - `.codex/initiatives/tmem_linear_generalization/experiments/probe_highrank_ldst.py`
  - Cases:
    - `reshape((2, M, N/2)) -> slice(dim=0) -> index(0)` with `+7`
    - `reshape((2, M/2, N)) -> slice(dim=0) -> index(0)` with `+13`
  - Observed runtime behavior on GPU:
    - `out - inp` is uniform over the full tensor
    - `torch.unique(diff)` is `{7}` or `{13}`
    - modified element count equals the full tensor size (`16384` for the
      128x128 probe)
  - Current status:
    - this is cleanly codegening and executing, but the semantics are not yet
      trusted as correct TMEM subview behavior.
    - leave these cases under active empirical validation before treating them
      as fully trusted positive coverage.
- Clean-negative behavior confirmed after the view-inference changes:
  - `reshape((M, N)).reshape((M, 2, N/2)).slice(dim=1)` style compositions that
    need unsupported column splitting now fail with:
    - `TMEM layout '16x128b' unsupported for shape [128, 64] and num_warps 4;
      try a different instr_variant, reshape or permute so TMEM columns stay
      contiguous, or use a supported TMEM register layout and insert
      convert_layout explicitly`
- Clean-negative behavior confirmed for the two-CTA MMAv5 higher-rank frontend
  repro:
  - emitted diagnostic contains:
    - `Result has an invalid layout`
    - `Layout has 1 CTAs per CGA, but the context requires 2 CTAs per CGA`
  - frontend still wraps this as:
    - `RuntimeError: error encountered during parsing`
  - Classification:
    - non-crashing, but still below the desired structured diagnostic quality.

## 2026-03-25 (BUG: cp shape coverage probe crashes in GluonResolveAutoEncodingsPass)
- Reproducer:
  - `CUDA_VISIBLE_DEVICES=1 python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_shape_coverage.py`
- Observed behavior:
  - building the grid tensor via `ttgl.arange` triggers `tt.make_range`
    failure during `GluonResolveAutoEncodingsPass`.
  - the pass manager aborts with `PassManager::run failed`, and the probe
    cannot record any `tcgen05.cp` opcode families for the 128xN shapes.
- Classification:
  - `BUG` (pass pipeline cannot handle the aggregated `tt.make_range` usage).
  - Follow-up: revisit once the auto-encoding resolver can handle these
    tensors or the probe is refactored to avoid the failing pattern.

## 2026-03-25 (runtime / lowering validation refresh)
- Added direct GPU coverage for canonical TMEM-linear `ld.red` in
  `python/test/gluon/test_tmem_runtime_matrix.py`:
  - `test_tmem_runtime_matrix_ld_red_identity_linear_layout`
  - matrix:
    - `red_op in {min, max}`
    - `abs in {false, true}`
    - `propagate_nan in {none, all}`
  - validation:
    - `CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_identity_linear_layout or ldst_descriptor_compositions or splitn_immediates or cp_no_scales_linear or higher_rank'`
    - `106 passed, 153 deselected`
- Refined fpsan MMA coverage to keep only proven-positive layouts in the
  positive runtime matrix and move unsupported linear 64x64 layouts into the
  clean-negative path.
  - validation:
    - `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_fpsan.py -k 'tcgen05_mma and not unsupported or tcgen05_mma_scaled and not unsupported'`
    - `9 passed, 63 deselected`
- Strengthened LLVMIR lit coverage and refreshed the brittle copy checks:
  - `test/Conversion/lower_tensor_memory_to_llvm.mlir`
  - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
  - validation:
    - `cd build/cmake.linux-aarch64-cpython-3.12 && lit -v test/Conversion/lower_tensor_memory_to_llvm.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - both tests pass.

## 2026-03-25 (GPU1 empirical TMEM/PTX probe sweep: cp families + ld/st gaps)
- Build baseline:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `ninja: no work to do.`
- Environment note:
  - probe scripts that import `python/test/gluon/test_tmem_runtime_matrix.py`
    must run with `PYTHONPATH=python:.`; otherwise Python can pick the
    site-packages Triton and fail to import `TensorMemoryLinearLayout`.
- Ran baseline instruction probe:
  - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/tmem_instruction_probes.py`
  - confirmed emitted families:
    - `ld/st`: `32x32b`, `16x64b`, `16x128b`, `16x256b`, `16x32bx2`
    - split-N immediates:
      - `n=2` -> immediate `0`
      - `n=64` -> immediate `16`
    - `cp`: `128x256b`, `128x128b`, `warpx4.32x128b`
    - `mma`: `tcgen05.mma.cta_group::1.kind::f16`

- Added development probe:
  - `.codex/initiatives/tmem_linear_generalization/experiments/probe_ldst_atom_gap_matrix.py`
  - purpose: broad runtime+codegen matrix for ld/st atom coverage with
    PASS/CLEAN_UNSUPPORTED/BUG classification.
  - run command:
    - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_ldst_atom_gap_matrix.py`
  - result summary:
    - `PASS=15`, `CLEAN_UNSUPPORTED=8`, `BUG=0`, `UNKNOWN_FAIL=0`
    - PASS includes all expected atom families and split-N forms:
      - `32x32b`, `16x64b`, `16x128b`, `16x256b`, `16x32bx2`
    - unsupported frontier in this matrix:
      - `M=64,N=64` across tested variants
      - `M=256,N=128` across tested variants
      - these fail cleanly with:
        `TMEM layout '<variant>' unsupported for shape [...]`
  - artifact:
    - `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_ldst_atom_gap_matrix_gpu1.json`

- Added development probe:
  - `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_shape_coverage_const.py`
  - purpose: confirm cp family emission without the known
    `ttgl.arange`/`tt.make_range` path.
  - run command:
    - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_shape_coverage_const.py`
  - result summary:
    - `cp_128x128b`: emitted `tcgen05.cp.cta_group::1.128x128b`
    - `cp_128x256b`: emitted `tcgen05.cp.cta_group::1.128x256b`
    - `cp_warpx4_32x128b`: emitted two
      `tcgen05.cp.cta_group::1.warpx4.32x128b`
  - artifact:
    - `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_shape_coverage_const_gpu1.txt`

## 2026-03-25 (BUG refresh: cp shape probe still crashes in auto-encoding)
- Reproducer:
  - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_shape_coverage.py`
- Observed behavior:
  - fails in `GluonResolveAutoEncodingsPass` with:
    - `'tt.make_range' op Failed to infer return type`
    - `RuntimeError: PassManager::run failed`
  - same failure class as previously logged, now with a fresh GPU1 artifact.
- Classification:
  - `BUG` (compiler pass failure on legal-looking probe construction).
- Artifact:
  - `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_shape_coverage_gpu1.log`

## 2026-03-25 (cp.warpx2 reachability refresh on GPU1)
- Re-ran bounded subslice search with saved logs:
  - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_warpx2_subslice.py --parent-rows 64 --start-rows 0 --max-layouts 128 --device cuda`
  - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_warpx2_subslice.py --parent-rows 128 --start-rows 0 --max-layouts 128 --device cuda`
  - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_warpx2_subslice.py --parent-rows 256 --start-rows 0 --max-layouts 128 --device cuda`
- Observed results:
  - `parent_rows=64`: `successful_compiles=2`, both emitted
    `tcgen05.cp.cta_group::1.warpx4.32x128b`; no `warpx2`.
  - `parent_rows=128`: `successful_compiles=0`; no `warpx2`.
  - `parent_rows=256`: `successful_compiles=0`; no `warpx2`.
  - failure text in logs is dominated by clean legalization rejects:
    - `'ttng.tmem_copy' op failed to find valid tcgen05.copy layout ...`
    - `failed to legalize operation 'ttng.tmem_copy' ...`
  - no segfault/assert/internal-compiler-error observed.
- Classification:
  - `CLEAN_UNSUPPORTED` for the probed `warpx2` search space in current tree.
  - interpretation: currently looks like frontend/legalization limitation, not
    ISA impossibility proven.
- Artifacts:
  - `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_warpx2_subslice_64_gpu1.log`
  - `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_warpx2_subslice_128_gpu1.log`
  - `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_warpx2_subslice_256_gpu1.log`

## 2026-03-25 (higher-rank TMEM descriptor runtime path fixed on GPU3)
- Added development probe:
  - `.codex/initiatives/tmem_linear_generalization/experiments/probe_fullrank_tmem_views.py`
  - purpose:
    - validate that higher-rank TMEM descriptors work when the descriptor rank
      is encoded directly in `TensorMemoryLinearLayout`, instead of relying on
      the legacy “rank+1 multibuffer” interpretation.
- Main fixes landed:
  - TMEM view-local encodings now strip dead physical `row`/`col` bits only in
    the view builders:
    - `lib/Dialect/TritonGPU/IR/Ops.cpp`
    - `python/src/gluon_ir.cc`
  - Builder-side TMEM `memdesc_subslice` now uses the view shape as
    `alloc_shape`:
    - `python/src/gluon_ir.cc`
  - `get_reg_layout()` now emits an actionable clean error for non-2D TMEM
    descriptor views:
    - `python/triton/experimental/gluon/language/_semantic.py`
- Probe results:
  - `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_fullrank_tmem_views.py --kind rank3 --m 128 --n 32`
    - PASS
    - emitted:
      - `tcgen05.st.sync.aligned.32x32b.x32.b32`
      - `tcgen05.ld.sync.aligned.32x32b.x32.b32`
  - `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_fullrank_tmem_views.py --kind rank4 --m 128 --n 32`
    - PASS
    - emitted:
      - `tcgen05.st.sync.aligned.32x32b.x32.b32`
      - `tcgen05.ld.sync.aligned.32x32b.x32.b32`
  - intermediate higher-rank TMEM views still fail cleanly when asked for a
    load/store register layout before being indexed down to 2D:
    - `TMEM load/store currently requires a 2D descriptor view ...`
- Two-CTA higher-rank full-rank layouts were also validated ad hoc on GPU3:
  - lifted `block_two_ctas` and `mmav5_twocta` layouts with prefix shape
    `[2, 2]` both executed correctly through the rank-4
    `slice -> index -> slice -> index` path.

## 2026-03-25 (regression fixed: split-N + legacy 64x64 MMAv5)
- Regression observed after the higher-rank TMEM view fix:
  - `python/test/gluon/test_tmem_runtime_matrix.py -k splitn_immediates`
    failed for all `N in {2,4,8,16,32,64,128}`
  - `python/test/gluon/test_fpsan.py -k 'test_tcgen05_mma and not unsupported or tcgen05_mma_scaled or twocta_asymmetric'`
    failed on the legacy `64x64` MMAv5 accumulator case
- Root cause:
  - global canonicalization of every `#ttng.tensor_memory_linear` was stripping
    zero `row`/`col` bases in:
    - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
  - that erased semantically meaningful TMEM “row-hole” structure used by:
    - split-N / `16x32bx2`-style `M=64` lowering
    - `matchTensorMemoryLegacyEncoding(...)` for legacy-compatible MMAv5
      layouts such as `64x64`
- Fix:
  - row/col zero-basis stripping is now limited to TMEM view builders only;
    generic TMEM-linear canonicalization no longer rewrites all layouts.
- Validation after the fix:
  - `CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'splitn_immediates'`
    - `7 passed`
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_fpsan.py -k 'test_tcgen05_mma and not unsupported or tcgen05_mma_scaled or twocta_asymmetric'`
    - `12 passed`
  - `CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'descriptor_roundtrip_sweeps or ldst_descriptor_compositions or ld_red_identity_linear_layout or splitn_immediates or cp_no_scales_linear'`
    - `138 passed, 48 skipped`
  - `cd build/cmake.linux-aarch64-cpython-3.12 && lit -v test/Conversion/lower_tensor_memory_to_llvm.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `PASS: 2/2`

## 2026-03-25 (higher-rank TMEM matrix revalidated on GPU2/GPU3)
- Rechecked the worker-reported higher-rank `memdesc_index` abort in fresh
  standalone Python processes after the TMEM subview representability fix:
  - the abort no longer reproduces;
  - lifted full-rank TMEM `index -> 2D load/store` paths execute correctly.
- Swept the higher-rank index matrix directly on GPU 2:
  - layouts:
    - 1-CTA: `identity`, `mixed`
    - 2-CTA: `block_two_ctas`, `mmav5_twocta`
  - `N in {64,128,256}`
  - variants `{32x32b,16x64b,16x128b,16x256b}`
  - result:
    - all 48 cases passed numerically (`out == inp + 5`) and produced matching
      PTX/LLIR TMEM opcode streams.
- Swept the higher-rank multidimensional slice matrix directly on GPU 2:
  - same layout / `N` / variant space as above
  - result:
    - all 48 cases now fail cleanly at compile time with
      `unsupported tensor memory memdesc_subslice view`
    - no `PassManager::run failed`, assertion, or hard abort observed.
- Updated `python/test/gluon/test_tmem_runtime_matrix.py` to match the real
  semantics:
  - removed stale `xfail` coverage for higher-rank index;
  - expanded higher-rank index runtime coverage to the full 48-case matrix with
    PTX/LLIR checks and TTGIR op presence checks;
  - converted multidimensional higher-rank slice cases into explicit clean
    negative tests;
  - tightened the MMAv5 2-CTA context-mismatch negative back to the intended
    invalid-layout / CGA-mismatch diagnostic;
  - updated blocked-descriptor negatives to assert the earlier
    `unsupported tensor memory memdesc_subslice view` failure mode.
- Validation after the test rewrite:
  - `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'higher_rank or splitn_auto_selects_16x32bx2 or ldst_descriptor_rank5_roundtrip or ldst_twocta_descriptor_rank5_roundtrip or ld_red_identity_linear_layout'`
    - `143 passed, 379 deselected`
  - `CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py`
    - `469 passed, 53 skipped`
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_fpsan.py -k 'tcgen05_mma'`
    - `26 passed, 56 deselected`
  - `cd build/cmake.linux-aarch64-cpython-3.12 && lit -v test/Conversion/lower_tensor_memory_to_llvm.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `PASS: 2/2`

## 2026-03-25 (regression fixed: `block_m_64` MMA after generic TMEM slice hardening)
- New regression found while widening `python/test/gluon/test_core.py` runtime
  coverage:
  - `test_block_m_64_mma[legacy]`
  - `test_block_m_64_mma[linear]`
  - both started failing with
    `unsupported tensor memory memdesc_subslice view`
    at the 2D `acc_tmem.slice(..., dim=1)` step.
- Root cause:
  - the new generic TMEM `memdesc_subslice` representability guard correctly
    rejects non-representable higher-rank views, but it also blocked a valid
    2D last-dimension split-N / M64 slice that needs to preserve the full
    physical TMEM encoding and allocation shape.
  - this is exactly the legacy `ttng.tmem_subslice` compatibility case.
- Fix:
  - `python/triton/experimental/gluon/language/_semantic.py`
    now routes 2D last-dimension TMEM slices (excluding scale layouts) through
    `builder.create_tmem_subslice(...)` instead of the generic
    `create_memdesc_subslice(...)` path.
  - this preserves the source physical TMEM encoding / `alloc_shape` and
    restores valid split-N / M64 MMAv5 compositions without reopening the
    higher-rank multidimensional slice cases.
- Validation after the fix:
  - `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_core.py -k 'test_block_m_64_mma or test_tcgen05_mma_plain_kind_runtime or test_tcgen05_mma_plain_kind_i8_reports_clean_error'`
    - `5 passed, 17944 deselected`
  - `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'higher_rank or block_descriptor_reports_clean_error'`
    - `110 passed, 412 deselected`
  - `CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py`
    - `469 passed, 53 skipped`
- Additional runtime coverage now present in `python/test/gluon/test_core.py`:
  - plain MMA runtime checks for:
    - `kind::tf32`
    - plain non-scaled `kind::f8f6f4` via `float8_e5m2`
  - clean-negative coverage for the current `kind::i8` PTXAS rejection on the
    in-tree Blackwell target:
    - `ptxas-blackwell ... error: Feature '.kind::i8' not supported on .target 'sm_103a'`
  - classification:
    - `CLEAN_UNSUPPORTED` for current `sm_103a` / PTXAS assumptions;
      no compiler crash or MLIR assertion observed.

## 2026-03-26 (BUG: unsupported scales copy layouts still raise generic RuntimeError path)
- Expanded TMEM runtime coverage for permuted/exotic linear layouts and widened
  scaled-mma copy format/layout sweeps in
  `python/test/gluon/test_tmem_runtime_matrix.py`.
- BUG classification added:
  - unsupported scales-copy shared layout candidates currently report the
    intended diagnostic notes to stderr, but the raised Python exception is a
    generic `RuntimeError("error encountered during parsing")` rather than a
    structured `CompilationError`.
  - Regression test:
    `test_tmem_runtime_matrix_bug_cp_scales_unsupported_layout_raises_runtimeerror_parse`.
  - This is a clean-negative diagnostics-surface bug (no crash, no
    `PassManager::run failed`, no assertion).

## 2026-03-25 (BUG classification from cp.warpx2 scales candidate sweep)
- Added a scales-copy layout probe in
  `python/test/gluon/test_tmem_runtime_matrix.py` with two explicit classes:
  - `PASS`: canonical `warpx4` shared layout
  - `BUG`: current `warpx2` candidate shared layout
- Repro result for the `warpx2` candidate:
  - compiler emits a specific legalization diagnostic
    (`failed to find valid tcgen05.copy layout`) but then still fails with
    `PassManager::run failed` / illegal-op legalization pipeline failure.
  - This is not a crash, but it is not a clean unsupported diagnostic path.
- Classification:
  - `BUG` (diagnostic quality / legalization error reporting gap).
- Kept this as an explicit runtime test expectation (`expected_status="BUG"`)
  so we can flip it to `CLEAN_UNSUPPORTED` or `PASS` once backend/legalization
  is fixed.

## 2026-03-26 (GPU1 copy-runtime matrix broadening: exact opcode sweeps)
- Expanded executable `tcgen05.copy` coverage in
  `python/test/gluon/test_tmem_runtime_matrix.py`:
  - added a shared helper `_assert_exact_cp_ptx_llir_match(...)` and switched
    copy tests to exact PTX/LLIR opcode-list equality checks (not suffix-only
    or presence-only checks);
  - widened no-scales positive coverage to lock both currently reachable
    families with exact counts:
    - `tcgen05.cp.cta_group::1.128x256b` (matrix sweeps)
    - `tcgen05.cp.cta_group::1.128x128b` (dedicated path);
  - widened scales positive runtime coverage via scaled-MMA copy to a matrix:
    - formats: `mxfp8/mxfp8`, `nvfp4/nvfp4`
    - CTA groups: `num_ctas in {1, 2}`
    - accumulator layouts: `legacy`, `linear`
    - with exact expected warpx4 opcode counts per vector size.
- Build + validation commands (GPU 1):
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `ninja: no work to do`
  - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 -m pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k "cp_"`
    - `49 passed, 5 skipped, 549 deselected in 2.90s`
- Newly encountered compiler-failure classification during sweep bring-up:
  - `CLEAN_UNSUPPORTED`: forcing `test_tmem_runtime_matrix_cp_128x128` to
    `M=256` triggers a clean frontend shape/layout diagnostic
    (`Mismatch in expected shape for dimension 0. Expected: 256, got: 128`);
    this kernel family is shape-locked to `M=128` today.
  - Resolved by constraining the positive sweep back to the actually legal
    shape, and keeping only executable codegen paths in the landed matrix.

## 2026-03-26 (BUG fix: explicit 16x32bx2 frontend path must not crash)
- Tried making `16x32bx2` a first-class backend atom in
  `getDistributedLayoutForTmemLdSt(...)`; that immediately exposed a real
  compiler abort:
  - requesting `tensor_memory_descriptor.get_reg_layout(instr_variant="16x32bx2")`
    on the existing legal `M=64` TMEM-linear layouts hit
    `LinearLayout::getBasis(...): Assertion 'pos < size()' failed`
    during frontend layout inference.
  - Classification: `BUG` (public API request triggered a compiler abort).
- Root cause:
  - the raw C++ `I16x32bx2` layout generator is still an internal helper, not a
    safe public entrypoint; exposing it directly violates the no-crash
    requirement for parser/validation/front-end inference.
- Fix implemented:
  - restored the C++ guard that keeps `I16x32bx2` implicit at the backend
    helper layer;
  - changed the public Gluon semantic path so
    `instr_variant="16x32bx2"` reuses the existing safe
    `32x32b_splitn` layout computation and basis-materialization logic;
  - this preserves the explicit user-facing variant while avoiding the unsafe
    internal helper.
- Validation:
  - direct per-shape repro on GPU3:
    - `n in {2,4,8,16,32,64,128}` now all return a valid
      `DistributedLinearLayout` instead of aborting;
  - focused runtime comparison:
    - `CUDA_VISIBLE_DEVICES=3 ... pytest ... -k 'splitn or explicit_16x32bx2'`
      -> `21 passed, 589 deselected`;
  - broad regression:
    - `CUDA_VISIBLE_DEVICES=3 ... pytest ... python/test/gluon/test_tmem_runtime_matrix.py`
      -> `557 passed, 53 skipped in 142.44s`;
  - LLVM/lit regression:
    - `lit -v test/Conversion/tritongpu_to_llvm_blackwell.mlir` -> pass.

## 2026-03-26 (TMEM copy frontier probe: warpx2 + 4x256b, GPU2)
- Rebuilt before probing:
  - `make`
    - `ninja: no work to do.`

- Q1 (`tcgen05.cp.warpx2::{02_13,01_23}.64x128b`) probing:
  - `BUG` (Gluon lowering path still not executable for warpx2 on this tree):
    - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_multicast_layouts.py > .codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_multicast_layouts_gpu2.log 2>&1`
      - result: no warpx2 layouts found; failing cases still end in
        `RuntimeError: PassManager::run failed` after
        `failed to find valid tcgen05.copy layout`.
    - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_warpx2_subslice.py --parent-rows 128 --max-layouts 384 --start-rows 0,32,64 > .codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_warpx2_subslice_128_gpu2.log 2>&1`
      - result: `successful_compiles=0`, `unknown=1152`, `no warpx2 opcodes observed`.
  - `PASS` (direct-PTX executable witness on this machine):
    - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_variants.py > .codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_variants_gpu2.log 2>&1`
      - result:
        - `warpx2_02_13: PASS launch_ok`
        - `warpx2_01_23: PASS launch_ok`
      - both run as executable cubins assembled by
        `third_party/nvidia/backend/bin/ptxas-blackwell`.
  - Shared/TMEM layout relationship required by the compiler matcher
    (`lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`):
    - `warpx2::02_13` iff multicast bits decode to `1`, i.e.
      `cvt.getBasis(row, log2(32), offset) == 0` and
      `cvt.getBasis(row, log2(64), offset) != 0`.
    - `warpx2::01_23` iff multicast bits decode to `2`, i.e.
      `cvt.getBasis(row, log2(32), offset) != 0` and
      `cvt.getBasis(row, log2(64), offset) == 0`.
    - current scales verifier still only accepts multicast `3` (warpx4) for
      compiler-lowered scales copies, so warpx2 scales stays blocked.

- Q2 (`tcgen05.cp.4x256b`) probing:
  - `PASS` (direct-PTX executable witness):
    - same command as above:
      `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_variants.py ...`
      - result: `cp_4x256b: PASS launch_ok`.
  - `CLEAN_UNSUPPORTED` (compiler-side lowering target not matched today):
    - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python3 - <<'PY' ... tmem_copy_no_scales_kernel[(1,)](..., 128, 256, 256, 32, ...) ... print(opcodes); print(has_4x256b) ... PY`
      - result: emits only `tcgen05.cp.cta_group::1.128x256b` (`has_4x256b False`).
    - `nl -ba lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp | sed -n '374,409p'`
      - result: `getTMemCopyAtom(...)` currently returns only
        `{128x128, 128x256, warpx2 64x128, warpx4 32x128}` atoms; no `4x256`.
  - Matcher/lowering info needed to lower `.4x256b` (inference from current
    code paths):
    - add a new `TMemCopyAtom` classification branch in `getTMemCopyAtom(...)`
      for `.4x256b` (currently absent);
    - extend verifier acceptance text in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
      so the family is recognized as legal where intended;
    - update the copy lowering descriptor synthesis in
      `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
      (`instrShape`, `reshapeIns`, and descriptor stepping logic) so row-4
      semantics are represented instead of the current row-32-centric path.

## 2026-03-26 (PTX/TMEM opcode catalog sweep, GPU2)
- Build pre-step:
  - `make`
    - `ninja: no work to do.`

- Main catalog command:
  - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_tmem_opcode_catalog.py > .codex/initiatives/tmem_linear_generalization/experiments/results/probe_tmem_opcode_catalog_gpu2.json 2> .codex/initiatives/tmem_linear_generalization/experiments/results/probe_tmem_opcode_catalog_gpu2.stderr`

- `PASS` families empirically compiled/executed:
  - `tcgen05.st/ld` atom families:
    - `tcgen05.{st,ld}.sync.aligned.32x32b.x128.b32` (`M=128,N=128`)
    - `tcgen05.{st,ld}.sync.aligned.16x64b.x64.b32` (`M=128,N=128`)
    - `tcgen05.{st,ld}.sync.aligned.16x128b.x32.b32` (`M=128,N=128`)
    - `tcgen05.{st,ld}.sync.aligned.16x256b.x16.b32` (`M=128,N=256`)
    - `tcgen05.{st,ld}.sync.aligned.16x32bx2.x16.b32` via both
      `instr_variant="32x32b_splitn"` and explicit `instr_variant="16x32bx2"`
      (`M=64,N=64`).
  - `tcgen05.ld.red` variants:
    - `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32`
    - `tcgen05.ld.red.sync.aligned.32x32b.x128.min.abs.NaN.f32`
    - `tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32`
    - `tcgen05.ld.red.sync.aligned.32x32b.x128.max.abs.NaN.f32`
  - `tcgen05.cp` families from compiler-lowered Gluon paths:
    - `tcgen05.cp.cta_group::1.128x256b`
    - `tcgen05.cp.cta_group::1.128x128b`
    - `tcgen05.cp.cta_group::1.warpx4.32x128b`
  - `tcgen05.cp` families from direct-PTX patch/assemble/run witness:
    - `tcgen05.cp.cta_group::1.warpx2::02_13.64x128b`
    - `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b`
    - `tcgen05.cp.cta_group::1.4x256b`
  - `tcgen05.mma` families:
    - plain:
      - `tcgen05.mma.cta_group::1.kind::tf32`
      - `tcgen05.mma.cta_group::1.kind::f8f6f4` (from both `f8e5m2` and
        `f8e4m3` frontend inputs)
    - scaled:
      - `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`
        (`acc_layout_kind=legacy`, CTA1), with scales-copy side opcode
        `tcgen05.cp.cta_group::1.warpx4.32x128b`
      - `tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X`
        (`acc_layout_kind=linear`, CTA2), with scales-copy side opcode
        `tcgen05.cp.cta_group::2.warpx4.32x128b`

- `CLEAN_UNSUPPORTED` findings:
  - `ld.red` on mixed TMEM-linear layout:
    - diagnostic class: `tmem_load reduction with N dimension sharded across
      threads is not supported`.
  - scales copy with warpx2 candidate shared layout:
    - diagnostic class: source shared layout does not lower to currently
      supported scales `tcgen05.copy.warpx4.32x128b` family.
  - plain MMA `kind::i8` on current target:
    - PTXAS reports `.kind::i8` is not supported on `.target sm_103a`.
  - MMA with mixed TMEM-linear accumulator layout:
    - verifier/parser rejects non-MMAv5-compatible accumulator layout.

- `BUG` findings in this sweep:
  - none observed in the catalog run (`probe_tmem_opcode_catalog_gpu2.json`).

- Recommended compiler follow-ups:
  - `cp` lowering:
    - add backend legalization/lowering path for scales-copy `warpx2`
      families (`02_13`, `01_23`) now that direct PTX witnesses execute;
    - add classifier + lowering support for `.4x256b` in
      `getTMemCopyAtom(...)` and NVGPU->LLVM descriptor synthesis.
  - MMA frontend/backend contract:
    - keep `kind::i8` explicitly gated as target-dependent
      `CLEAN_UNSUPPORTED` (no generic internal-error framing);
    - if mixed TMEM-linear MMAv5 accumulators are intended long-term, add a
      canonicalization/relayout path to supported MMAv5 accumulator tiles
      before `tc_gen5_mma`.

## 2026-03-26 (cp scales candidate hardened from BUG to clean unsupported)
- Former runtime `BUG` status for the probed scales-copy `warpx2_candidate`
  layout has been retired.
  - current behavior: early descriptive verifier failure, not
    `PassManager::run failed`.
  - exact runtime regression remains in
    `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_layout_probe`.
- Added compile-only no-crash coverage in
  `test/TritonNvidiaGPU/invalid.mlir` for the candidate shared layout:
  - shared bases:
    `[[32, 0], [0, 1], [1, 0], [0, 2], [0, 4], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]]`
  - expected diagnostic:
    `The source shared layout does not lower to Triton's currently supported tcgen05.copy.warpx4.32x128b descriptor family for tensor memory scales.`
- Important distinction preserved:
  - direct PTX still proves `tcgen05.cp.warpx2::{02_13,01_23}.64x128b`
    opcodes are executable on this machine;
  - current Triton gap is descriptor synthesis / legalization for this family,
    not ISA unavailability.

## 2026-03-26 (expanded permanent runtime/lit coverage validated)
- Rebuilt before validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `ninja: no work to do.`
- Focused compiler/lit validation after widening invalid + LLVM checks:
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && ninja triton-opt && lit -v test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - result: both tests pass.
- Focused runtime subset after widening TMEM matrices:
  - `CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_descriptor_roundtrip_sweeps or ldst_descriptor_rank5_roundtrip or cp_scales_layout_probe or mma_twocta or explicit_16x32bx2'`
    - result: `67 passed, 24 skipped, 519 deselected in 19.53s`.
- Full permanent TMEM runtime matrix after the new sweeps:
  - `CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py`
    - result: `674 passed, 53 skipped in 189.96s`.
- Permanent test surface expanded in this pass:
  - positive GPU sweeps for permuted TMEM-linear row/col basis orders
    (`identity`, `rotate1`, `even_odd`, `reverse`) across ld/st and
    descriptor-composition paths;
  - wider scaled-MMA copy matrix over format pairs
    `{mxfp8,mxfp4,nvfp4}` x CTA count x accumulator layout kind;
  - clean-negative GPU regressions for exotic non-canonical plain-copy and MMA
    layouts;
  - no-crash invalid IR coverage for malformed `tensor_memory_linear` syntax,
    mixed linear `tmem_copy`, and wrong-result-encoding TMEM subslices;
  - LLVM/lit coverage for non-zero TMEM alloc offsets plus explicit wait usage.

## 2026-03-26 (direct PTX semantics follow-up: warpx2 still needs descriptor work)
- Reused the direct-PTX semantics probe:
  - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_semantics.py > .codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_semantics_gpu2.log 2> .codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_semantics_gpu2.stderr`
  - result file:
    `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_semantics_gpu2.json`
- Key semantic findings from the patched baseline scales-copy kernel:
  - `tcgen05.cp.warpx2::02_13.64x128b` is deterministic and populates the
    expected warp-pair structure `(0,2)` / `(1,3)` with the current descriptor,
    but not the full warpx4 result.
  - `tcgen05.cp.warpx2::01_23.64x128b` is deterministic and populates the
    complementary warp-pair structure `(0,1)` / `(2,3)` with the current
    descriptor, but not the full warpx4 result.
  - `tcgen05.cp.4x256b` assembles and launches, but with the current descriptor
    mapping it is nondeterministic and does not correspond to the expected
    scales copy semantics.
- Additional hybrid check on GPU2:
  - direct one-off PTX patching that mixed `02_13` and `01_23` across the two
    existing copy sites in the baseline kernel also failed to recover the full
    expected warpx4 output.
- Engineering conclusion:
  - `warpx2` is a real, deterministic instruction family on this hardware, but
    supporting it in Triton requires descriptor/address synthesis work; simple
    opcode substitution is insufficient.
  - `.4x256b` should remain blocked until we have a descriptor model that makes
    its semantics reproducible.

## 2026-03-26 (exotic TMEM subview blocked-fallback BUG downgraded to clean unsupported)
- Repro:
  - development-only `/tmp/probe_tmem_blocked_view.py` on `CUDA_VISIBLE_DEVICES=0`
    built a rank-3 TMEM descriptor, reshaped/sliced it down to an exotic
    `128x32` view with layout
    `#ttng.tensor_memory_linear<{row = [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0]]}>`,
    then attempted `view.load(#blocked)` / `view.store(#blocked)`.
- Previous behavior:
  - `BUG`: `ttng.tmem_load` survived frontend parsing, then
    `RelayoutTritonGPU` failed with `PassManager::run failed`.
- Fix:
  - tightened `verifyTMEMOperand(...)` in
    `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` so TMEM load/store/alloc still
    defer feasible one-convert fallback cases, but now reject impossible cases
    early when `getTmemCompatibleLayouts(...)` is empty.
- New behavior:
  - early descriptive diagnostic at parse/verification time:
    - error: `result has no supported register layout`
    - note: `No TMEM-compatible register layout exists for this operand, so relayout cannot insert a fallback convert_layout.`
  - no backend pass crash, no `PassManager::run failed`.
- Regression coverage:
  - added compile-only invalid test in `test/TritonNvidiaGPU/invalid.mlir`
    for this exact exotic `ttng.tmem_load` case.

## 2026-03-26 (direct-PTX cp semantics characterization, GPU2)
- Goal: test semantic correctness (not just assembly) for
  `tcgen05.cp.cta_group::1.warpx2::{02_13,01_23}.64x128b` and
  `tcgen05.cp.cta_group::1.4x256b` using patched PTX.
- Command:
  - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_semantics.py > .codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_semantics_gpu2.log 2> .codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_semantics_gpu2.stderr`
- Probe method:
  - compiled known-good baseline PTX from
    `tmem_copy_scales_warpx4_kernel`;
  - replaced cp opcode family and swept 6 patch modes over descriptor
    base/offset usage:
    `both_orig`, `first_only`, `second_only`,
    `both_first_map`, `both_second_map`, `swap_maps`;
  - checked output against the known scales-copy expected layout used in
    `test_tmem_runtime_matrix` and recorded deterministic behavior and warp-pair
    invariants.
- Results (semantic witness search):
  - `warpx4_control`: multiple modes remained exact (`mismatch=0`);
    validates the harness.
  - `warpx2::01_23`:
    - best mode `both_orig`: deterministic, but not exact
      (`num_mismatch_expected=2040`);
    - observed pair invariant matches family (`chunk01==chunk23`), but only one
      pair matches the control tile map (`['B','B','other','other']`).
  - `warpx2::02_13`:
    - best mode `both_orig`: deterministic, but not exact
      (`num_mismatch_expected=2040`);
    - observed pair invariant matches family (`chunk02==chunk13`), but only one
      pair matches the control tile map (`['B','other','B','other']`).
  - `cp.4x256b`:
    - all tested modes launch but remain non-exact
      (`best mismatch=2074`);
    - outputs were non-deterministic across repeated launches (`deterministic=false`)
      and never matched control tiles, so this patched path is not semantically
      usable.
- Conclusion:
  - no semantically-correct direct PTX witness was found for either
    `warpx2` family or `4x256b` under the tested descriptor base/offset patch
    space;
  - `warpx2` shows structured but incomplete/incorrect data mapping for this
    kernel shape;
  - `4x256b` is unstable with these descriptor patterns.
- Follow-up implication for compiler work:
  - enabling these families in Triton needs dedicated descriptor synthesis and
    per-family pointer/offset stepping logic, not opcode substitution alone.

## 2026-03-26 (TMEM higher-rank subview active-footprint fix, lit + GPU)
- BUG:
  - rank-4 TMEM descriptor roundtrips with
    `slice(dim=0) -> index -> slice(dim=0) -> index -> 2D load/store`
    regressed after broadening TMEM generic views.
  - Symptoms:
    - verifier/parser crash was already downgraded earlier, but valid view
      chains still failed either with
      `unsupported tensor memory memdesc_subslice view` or with a stale
      `result tensor memory encoding must be ... but got ...` mismatch between
      frontend-inferred and verifier-inferred TMEM-linear encodings.
- Root cause:
  - both the MLIR verifier path in
    `lib/Dialect/TritonGPU/IR/Ops.cpp` and the frontend builder helper in
    `python/src/gluon_ir.cc` were inferring subslice pseudoinverses against the
    wrong physical footprint.
  - They needed to infer the active TMEM physical subspace touched by the
    sliced view, not reuse or heuristically re-infer the full source
    row/col footprint.
- Fix:
  - track active physical TMEM basis masks per output dim for the sliced view;
  - build `dstInv` with explicit active physical out sizes;
  - use `LinearLayout::tryCreate(...)` so malformed views fail cleanly;
  - keep the surjectivity gate before `pseudoinvert()` so invalid views remain
    verifier-clean rather than crashing.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `lit -v test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir test/TritonGPU/invalid.mlir`
    -> `3 passed`
  - targeted GPU row/col + descriptor view sweep on `CUDA_VISIBLE_DEVICES=0`:
    - `python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_rowcol_permuted_layout_sweep or ldst_descriptor_compositions_rowcol_permuted_layout_sweep or ldst_descriptor_roundtrip_rowcol_permuted_sweeps or splitn_rowcol_permuted_layout_sweep'`
    - result: `736 passed, 64 skipped`
  - targeted GPU copy/MMA sweep on `CUDA_VISIBLE_DEVICES=1`:
    - `python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales_linear or cp_scales_warpx4_via_scaled_mma_geometry_sweep or mma_plain_kinds_with_linear_acc or mma_i8_reports_clean_error or mma_rowcol_permuted_layout_reports_clean_unsupported'`
    - result: `96 passed`
- Outcome:
  - higher-rank TMEM descriptor roundtrips with arbitrary row/col-permuted
    TMEM-linear layouts are back to functional GPU execution;
  - the verifier now stays no-crash and agrees with frontend-inferred TMEM
    layouts on the exercised view chains.

## 2026-03-26 (BF16 plain MMAv5 confirmed, promoted to permanent coverage)
- Gap identified:
  - BF16 plain MMAv5 lowered as `kind::f16` in code, but did not yet have
    direct runtime coverage in the TMEM runtime matrix.
- Direct probe:
  - `CUDA_VISIBLE_DEVICES=2` temporary Python harness using
    `python.test.gluon.test_core.mma_kernel`
  - both legacy and canonical `TensorMemoryLinearLayout` accumulators passed
    numerically for BF16 inputs and emitted
    `tcgen05.mma.cta_group::1.kind::f16`.
- Permanent coverage added:
  - extended `python/test/gluon/test_tmem_runtime_matrix.py`
    `test_tmem_runtime_matrix_mma_plain_kinds_with_linear_acc` to cover
    `kind == "bf16"` for both legacy and linear accumulators.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96` -> no-op rebuild
  - `CUDA_VISIBLE_DEVICES=2 ... pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_plain_kinds_with_linear_acc and bf16'`
    -> `2 passed`

## 2026-03-26 (BUG: 2-CTA TF32 MMAv5 still fails lowering)
- Probe:
  - temporary `CUDA_VISIBLE_DEVICES=3` Python harness adapted from
    `tmem_mma_twocta_kernel` using TF32-rounded float32 inputs, both legacy and
    canonical TMEM-linear 2-CTA accumulators.
- Result:
  - clean lowering failure during `ConvertTritonGPUToLLVM` with diagnostic:
    - `tcgen05.mma does not support transposed float32 operands in shared memory`
  - no crash, but no executable path.
- Reproducer shape:
  - `block_m = 256`, `block_n = 128`, `block_k = 32`,
    `cta_group::2`, TMA-fed shared operands, multicast enabled.
- Current status:
  - classify as `BUG` / uncovered theoretically-plausible case pending further
    lowering work;
  - do not add as a permanent expected-failure test yet, because this should be
    revisited rather than locked in as unsupported ISA.

## 2026-03-26 (Descriptor multi-dim TMEM slice GPU coverage)
- Gap identified:
  - generic `tensor_memory_descriptor.slice(..., dim=...)` over TMEM views only
    had GPs / negative coverage; no positive run showed that slice along every
    dimension stays valid while still generating `ttg.memdesc_slice`.
- Work:
  - introduced `tmem_ldst_descriptor_multidim_slice_positive_kernel` that reshapes
    the TMEM to `[2, M/2, 2, N/2]`, slices dims `0..3`, and adds `11.0` to the
    extracted sub-block before materializing the result with a register layout.
  - added `test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_positive`
    covering `identity`, `mixed`, and `scrambled_cols` `TensorMemoryLinearLayout`
    inputs, asserting the block update, verifying PTX/LLIR `16x128b` opcodes, and
    checking `tensor_memory_linear` plus `ttg.memdesc_slice` in TTIR.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `CUDA_VISIBLE_DEVICES=2 TRITON_BUILD_WITH_CCACHE=true python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k descriptor_multidim_slice_positive` -> 3 tests passed

## 2026-03-26 (No-crash shared-layout validation + broader 2-CTA MMA coverage)
- Fixed a frontend no-crash bug in shared-layout validation:
  - `python/triton/experimental/gluon/language/_layouts.py`
    `SwizzledSharedLayout.__post_init__` now validates that every
    `cga_layout` basis rank matches `len(order)`.
  - Motivation:
    - a direct 2-CTA `tcgen05.copy` probe initially passed a rank-2 CGA basis
      into `mbarrier.MBarrierLayout` (a rank-1 swizzled shared layout), which
      previously fell through to a C++ `ArrayRef` assert in
      `buildCgaLayoutAttr(...)` during IR materialization.
  - Permanent regression:
    - `python/test/gluon/test_frontend.py`
      `test_mbarrier_invalid_cga_rank_reports_clean_error`
    - parser now raises a wrapped `CompilationError` with
      `all cga_layout bases must have rank 1` instead of crashing.
- Added permanent plain 2-CTA MMAv5 runtime coverage in
  `python/test/gluon/test_tmem_runtime_matrix.py`:
  - new `test_tmem_runtime_matrix_mma_twocta_plain_kinds`
  - matrix:
    - `kind in {tf32, bf16, f8e5m2, f8e4m3}`
    - `acc_layout_kind in {legacy, linear}`
  - all 8 cases execute correctly on GPU and assert exact PTX/LLIR opcode
    families:
    - `tf32 -> tcgen05.mma.cta_group::2.kind::tf32`
    - `bf16 -> tcgen05.mma.cta_group::2.kind::f16`
    - `f8e5m2/f8e4m3 -> tcgen05.mma.cta_group::2.kind::f8f6f4`
  - commit path is also checked:
    - `tcgen05.commit.cta_group::2...`
- Added LLVM lowering lit coverage for the newly-proven 2-CTA plain MMA kinds:
  - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `@tc_gen5_mma_2ctas_tf32`
    - `@tc_gen5_mma_2ctas_f8f6f4`
  - both check `nvg.cluster_id`, `nvvm.elect.sync`,
    `tcgen05.mma.cta_group::2.kind::*`, and
    `tcgen05.commit.cta_group::2...multicast::cluster...`.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96` -> no-op rebuild
  - `python3 -m py_compile python/triton/experimental/gluon/language/_layouts.py python/test/gluon/test_frontend.py python/test/gluon/test_tmem_runtime_matrix.py`
  - `PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_frontend.py -k 'mbarrier or invalid_cga_rank'`
    -> `6 passed`
  - `PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_twocta_plain_kinds'`
    -> `8 passed`
  - `lit -v test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    -> `1 passed`

## 2026-03-26 (BUG: direct no-scales 2-CTA tcgen05.copy still does not reach cta_group::2)
- Probe setup:
  - temporary `CUDA_VISIBLE_DEVICES=0` Gluon harness using
    `tcgen05_copy(smem, tmem)` with:
    - canonical 2-CTA MMAv5-equivalent TMEM-linear destination
      (`block = [[128, 0]], two_ctas=True`);
    - CGA-aware 2-CTA NVMMA shared source layout;
    - `num_ctas=2`.
- Observed behavior:
  - numerics are correct for `256x{64,128,256}` `f32`;
  - PTX/LLIR emit only `tcgen05.cp.cta_group::1.128x256b`;
  - no `tcgen05.commit.cta_group::2` is emitted.
- Interpretation:
  - this is not an ISA impossibility; it is a compiler/lowering gap.
  - `TensorMemoryToLLVM.cpp` chooses `cta_group::{1,2}` from
    `getModuleTwoCTAs(op)`, but the generic copy path is not propagating
    `ttng.two-ctas` from the two-CTA TMEM layout/module context the way MMAv5
    does today.
- BUG:
  - direct no-scales 2-CTA `tcgen05.copy` is still missing executable
    `cta_group::2.{128x128b,128x256b}` coverage/support in-tree.
- Related clean diagnostic:
  - using the generic `block_two_ctas` TMEM-linear destination with a 2-CTA
    NVMMA shared source fails cleanly in the verifier with
    `The source and destination must have the same cga layout`, which is the
    expected current matcher boundary rather than a crash.

## 2026-03-26 (follow-up: 2-CTA no-scales tcgen05.copy runtime bug resolved as async-proxy test contract issue)
- Reproduced the previously xfailed cold-start failure deterministically with
  repeated fresh-process runs of the existing `tmem_copy_no_scales_twocta`
  Gluon kernel.
  - failure signature:
    - wrong rows consistently came from the follower CTA half
      (`128:160`, sometimes `224:256`);
    - PTX/LLIR were already correct:
      `tcgen05.cp.cta_group::2.128x256b`,
      `tcgen05.commit.cta_group::2...multicast::cluster.b64`.
- Probed the same kernel with explicit pre-copy ordering in
  `.codex/initiatives/tmem_linear_generalization/experiments/probe_twocta_copy.py`.
  - `PRE_COPY_CLUSTER_FENCE=1`:
    - inserts `fence_async_shared(cluster=True)` before `tcgen05_copy`;
    - passed `20/20` fresh-process runs.
  - `PRE_COPY_CLUSTER_BARRIER=1`:
    - inserts `cluster.barrier()` before `tcgen05_copy`;
    - also passed `20/20` fresh-process runs.
- Interpretation:
  - the old xfail was not proving a legal TMEM codegen bug;
  - it was exercising an invalid async-proxy sequence that omitted the
    required pre-copy ordering from generic shared-memory writes to the async
    proxy reader.
- Landed follow-up changes:
  - `python/test/gluon/test_tmem_runtime_matrix.py`
    - added explicit `fence_async_shared()` before every `tcgen05_copy`;
    - upgraded 2-CTA no-scales copy from xfail to passing execution coverage;
    - added exact fence/commit PTX+LLIR assertions for the 2-CTA path.
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/BarrierOpToLLVM.cpp`
    - fixed unpredicated cross-CTA `wait_barrier` lowering to handle
      leader-predicate synthesis without dereferencing a null predicate.
  - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
    - `ttng.tmem_copy` now verifies its optional barrier with
      `verifyBarrierType(...)` and `verifyCompletionBarrierLayout(...)`.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_frontend.py -k 'test_fence_async_shared or test_tcgen05_commit'`
    -> `3 passed`
  - `PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales_twocta_codegen or cp_no_scales_twocta_execution'`
    -> `8 passed`
  - `PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales and not twocta and not unsupported'`
    -> `56 passed, 5 skipped`
  - `cd $(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())') && ninja triton-opt && lit -v test/Conversion/tritongpu_to_llvm_blackwell.mlir test/TritonNvidiaGPU/invalid.mlir`
    -> `2 passed`

## 2026-03-26 06:00 UTC: x1 ld/st and direct scales ld/st expansion

- Added executable GPU coverage in `python/test/gluon/test_tmem_runtime_matrix.py`
  for:
  - `128x2xf16` x1 TMEM load/store on:
    - canonical `TensorMemoryLinearLayout`,
    - legacy packed `TensorMemoryLayout(block=(128, 2), col_stride=1)`,
    - legacy unpacked `TensorMemoryLayout(block=(128, 2), col_stride=2)`;
  - direct `TensorMemoryScalesLayout` load/store on:
    - default path: `16x8xi8`, `num_warps=8`,
    - explicit variant sweep:
      - `16x4xi8`, `num_warps=4`, `instr_variant="16x32bx2"`,
      - `32x4xi8`, `num_warps=8`, `instr_variant="32x32b"`,
      - `64x8xi8`, `num_warps=4`, `instr_variant="32x32b"`,
      - `128x32xi8`, `num_warps=4`, `instr_variant="32x32b"`.
- Exact PASS boundaries observed:
  - x1 f16:
    - packed paths emit
      `tcgen05.st.sync.aligned.32x32b.x1.b32` +
      `tcgen05.ld.sync.aligned.32x32b.x1.b32`;
    - unpacked path emits
      `tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32` +
      `tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32`;
    - runtime outputs matched exactly.
  - direct scales:
    - `16x8`, default `get_reg_layout()`, `num_warps=8` emits
      `16x32bx2.x1.b32` store/load at offset `0`;
    - `16x4`, explicit `16x32bx2`, `num_warps=4` emits
      `16x32bx2.x1.b32` store/load at offset `0`;
    - `32x4`, `64x8`, `128x32` with explicit `32x32b` emit
      `32x32b.x1/x4/x32.b32` store/load respectively at offset `0`;
    - runtime outputs matched exactly.
- Added/strengthened LLVMIR checks:
  - canonical TMEM-linear x1 lowering now has an isolated conversion check;
  - direct scales x1 load/store check now asserts the load immediate is `0`
    and not `1`.
- Validation:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_x1_f16_roundtrip or ldst_scales_direct_roundtrip or ldst_scales_variant_sweep'`
    -> `8 passed`
  - `triton-opt /tmp/scales_load.mlir`
    -> `PASS` (direct `ttng.tmem_load` on `tensor_memory_scales_encoding`)
  - isolated LLVM check:
    - `triton-opt /tmp/tmem_x1_scales_checks.mlir -split-input-file --convert-triton-gpu-to-llvm=compute-capability=100 -cse | FileCheck /tmp/tmem_x1_scales_checks.mlir`
    -> `PASS`
- BUG notes:
  - `BUG`: direct scales ld/st with `shape=[16, 8]`, `num_warps=4`,
    `instr_variant="16x32bx2"` is currently rejected by
    `_compute_tmem_reg_layout(...)` even though adjacent narrow-tile scales
    cases pass. Need to determine whether this is a real ISA-impossible case or
    a reg-layout selection bug.
  - `BUG`: repo-wide `test/Conversion/tritongpu_to_llvm_blackwell.mlir` still
    has unrelated TMEM failures on this branch (MMAv5 col-stride expectation
    drift and indexed-view `tmem_copy` lowering/state drift), so the new x1 and
    scales LLVM checks were validated in isolation for now.
  - `BUG`: repo-wide `test/TritonNvidiaGPU/invalid.mlir` still has unrelated
    stale expectations on this branch (e.g. an M64 `tmem_copy` diagnostic text
    change from generic `Incorrect tmem layout.` to the more specific
    `blockM=128` failure).

## 2026-03-26 07:15 UTC: narrow scales split-layout fallback fix

- Fixed a real frontend/runtime gap in
  `python/triton/experimental/gluon/language/_semantic.py`:
  - explicit `instr_variant="16x32bx2"` on
    `TensorMemoryScalesLayout(shape=[16, 8], num_warps=8)` was failing during
    split-layout postprocessing even though the default `32x32b` path already
    lowered and executed as `tcgen05.{ld,st}.sync.aligned.16x32bx2.x1.b32`;
  - `_compute_tmem_reg_layout(...)` now falls back to the broadcasted
    `32x32b` register layout for this narrow scales corner instead of rejecting
    a codegenable case.
- Expanded permanent coverage:
  - `python/test/gluon/test_tmem_runtime_matrix.py`
    - added the explicit `16x32bx2` runtime regression for
      `shape=[16, 8], num_warps=8`, asserting exact PTX/LLIR opcode equality
      (`16x32bx2.x1.b32` at offset `0`);
    - added direct scales ld/st execution coverage for nontrivial CGA bases in
      2-CTA mode:
      - `shape=[128, 64]`, `num_ctas=2`, `CGALayout=[[1, 0]]`
      - `shape=[256, 32]`, `num_ctas=2`, `CGALayout=[[1, 0]]`
      - `shape=[256, 64]`, `num_ctas=2`, `CGALayout=[[1, 0]]`
      with exact PTX/LLIR checks for the resulting `32x32b.x32/x64.b32`
      message families and TTGIR assertions on the printed `CGALayout`;
  - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - added an LLVM regression for a wider scales case
      (`64x8`, 8 warps) where explicit `16x32bx2` register selection lowers to
      repeated `32x32b.x1.b32` messages at offsets `0` and `2`;
  - `test/TritonNvidiaGPU/invalid.mlir`
    - aligned the stale M64 `tmem_copy` negative expectation with the current
      more specific verifier diagnostic (`blockM=128`).
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `python3 -m py_compile python/triton/experimental/gluon/language/_semantic.py python/test/gluon/test_tmem_runtime_matrix.py`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_scales_direct_roundtrip or ldst_scales_variant_sweep or ldst_scales_variant_reports_clean_unsupported'`
    -> `38 passed, 1663 deselected`
  - `cd $(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())') && ninja triton-opt && lit -v test/Conversion/tritongpu_to_llvm_blackwell.mlir test/TritonNvidiaGPU/invalid.mlir`
    -> `2 passed`
- Updated PASS/CLEAN_UNSUPPORTED boundary:
  - `PASS`: direct scales ld/st with `shape=[16, 8]`, `num_warps=8`,
    `instr_variant="16x32bx2"` now compiles, runs, and emits the expected
    `16x32bx2.x1.b32` store/load pair.
  - `CLEAN_UNSUPPORTED`: the scales `warpx2` shared-layout candidate remains a
    deliberate verifier rejection, even though direct PTX probes continue to
    show `tcgen05.cp.warpx2::{02_13,01_23}.64x128b` exists at the ISA level.

## 2026-03-26 08:35 UTC: runtime matrix expansion for x1/scales/two-CTA copy

- Expanded executable TMEM coverage in
  `python/test/gluon/test_tmem_runtime_matrix.py`:
  - added `x1` `f32` positive runtime cases for:
    - canonical TMEM-linear one-CTA `128x1`
    - legacy-equivalent one-CTA `128x1`
    - canonical TMEM-linear two-CTA `256x1`
  - added `x1` `f32` descriptor-chain positives over those same shapes,
    checking exact two-round-trip `32x32b.x1.b32` opcode sequences;
  - added clean-negative `x1` `f32` coverage for impossible
    `16x64b` / `16x128b` / `16x256b` variants on `[M, 1]`;
  - widened the direct scales ld/st runtime matrix to cover exact PTX/LLIR
    opcode matches across:
    - `16x32bx2.x1`
    - `32x32b.x{1,2,4,8,16,32}`
    - both offset-bearing narrow cases and wider single-message cases;
  - locked representative clean-negative scales cases:
    - register-broadcasted `16x16`
    - unsupported `[32, 8]` / `[32, 16]` shapes for explicit
      `32x32b` / `16x32bx2`.
- Expanded runtime coverage for no-scales two-CTA `tcgen05.copy`:
  - now validates both legacy and canonical TMEM-linear destinations for:
    - `N=16` / `swizzle=32`
    - `N=32` / `swizzle in {32,64,128}`
    - `N=64,128,256` / `swizzle=128`
  - each case executes on GPU and asserts exact
    `tcgen05.cp.cta_group::2.128x256b` opcode counts plus cluster fence/barrier
    LLIR/PTX side effects.
- Expanded LLVM/lit checks:
  - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - added a wide direct-scales lowering regression for `128x32` ->
      `tcgen05.{ld,st}.sync.aligned.32x32b.x32.b32`;
  - `test/TritonNvidiaGPU/invalid.mlir`
    - aligned the stale M64 `tmem_copy` diagnostic expectation with the current
      specific verifier error (`Tmem layout must have blockM=128.`);
  - `test/TritonNvidiaGPU/membar-cluster.mlir`
    - kept green alongside the widened two-CTA copy/runtime surface.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_scales_direct_roundtrip or ldst_scales_variant_sweep or ldst_scales_variant_reports_clean_unsupported'`
    -> `35 passed, 1660 deselected`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_x1_f16_roundtrip or ldst_scales_direct_roundtrip or ldst_scales_variant_sweep or ldst_scales_variant_reports_clean_unsupported or cp_no_scales_twocta_codegen or cp_scales_warpx4 or cp_scales_warpx4_via_scaled_mma_copy_matrix or cp_scales_warpx4_via_scaled_mma_geometry_sweep or mma_twocta_plain_kinds'`
    -> `113 passed, 1582 deselected`
  - `cd $(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())') && ninja triton-opt && lit -v test/Conversion/tritongpu_to_llvm_blackwell.mlir test/TritonNvidiaGPU/invalid.mlir test/TritonNvidiaGPU/membar-cluster.mlir`
    -> `3 passed`
- PASS/CLEAN_UNSUPPORTED boundary updates:
  - `PASS`:
    - direct scales ld/st exact lowering across `16x32bx2.x1` and
      `32x32b.x{1,2,4,8,16,32}`;
    - `x1` `f32` one-CTA and two-CTA ld/st plus descriptor chains;
    - widened no-scales two-CTA `tcgen05.copy` swizzle/shape matrix.
  - `CLEAN_UNSUPPORTED`:
    - `x1` `f32` with `16x64b` / `16x128b` / `16x256b`;
    - scales `16x16` register-broadcasted cases and explicit unsupported
      `[32, 8]` / `[32, 16]` layouts;
    - no new parser/verifier/pass crash was observed in this pass.


## 2026-03-26 10:05 UTC: 2-CTA no-scales tcgen05.copy cold-start bug fixed and pinned

- Root-caused the earlier cold-start corruption on 2-CTA non-scales
  `tcgen05.copy` to missing cross-CTA synchronization between a distributed
  shared-memory producer and `ttng.tmem_copy`.
- Compiler changes already in tree from this pass now matter materially:
  - `ttng.tmem_copy` participates in memory-effect analysis;
  - `ClusterBarrierInsertion` treats 2-CTA `ttng.tmem_copy` as a cross-CTA
    consumer;
  - explicit-buffer alias filtering no longer hides distributed-shared hazards.
- Added a permanent membar regression in
  `test/TritonNvidiaGPU/membar-cluster.mlir` proving that a
  `ttg.local_store` into distributed shared memory is followed by
  `ttng.cluster_barrier` before `ttng.tmem_copy`.
- Updated the executable runtime matrix in
  `python/test/gluon/test_tmem_runtime_matrix.py` to assert the actual post-fix
  PTX/LLIR shape for 2-CTA no-scales copy:
  - exact `tcgen05.cp.cta_group::2.128x256b` counts;
  - exactly one `tcgen05.commit.cta_group::2`;
  - no `cta_group::1` copy/commit opcodes;
  - exactly one PTX `fence.proxy.async.shared::cluster`;
  - exactly two PTX `barrier.cluster.arrive.aligned` / `wait.aligned` pairs;
  - exact LLIR cluster intrinsic counts (`fence.proxy.async.shared_cluster=2`,
    `barrier.cluster.arrive.aligned=3`, `barrier.cluster.wait.aligned=3`), and
    no relaxed-cluster barrier path.
- Empirical validation:
  - 10 fresh-process runs for the former bad tuples both passed:
    - `(linear, N=256, swizzle=32)`
    - `(legacy, N=256, swizzle=64)`
  - full executable two-CTA no-scales copy matrix passed numerically with the
    new structural assertions.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
  - `cd $(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())') && ninja triton-opt && lit -v test/TritonNvidiaGPU/membar-cluster.mlir test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    -> `3 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales_twocta_codegen'`
    -> `28 passed, 1688 deselected`
- BUG status update:
  - the earlier `BUG` classification for 2-CTA no-scales `tcgen05.copy`
    miscompiles is closed on the current tree;
  - current frontier remains `warpx2` descriptor synthesis and other still-clean
    unsupported copy families, not this 2-CTA `128x256b` path.

## 2026-03-26 18:25 UTC: stabilize 2-CTA no-scales copy synchronization coverage

- Root-cause/fix trail:
  - the earlier cold-start 2-CTA no-scales `tcgen05.copy` corruption was caused
    by missing cross-CTA dependency tracking between distributed shared writes
    and `ttng.tmem_copy`.
  - landed compiler-side fixes:
    - `ttng.tmem_copy` now exposes explicit memory effects for analysis;
    - `ClusterBarrierInsertion` now treats `ttng.tmem_copy` as a tracked
      cross-CTA consumer;
    - the explicit-buffer alias filter no longer suppresses hazards when either
      side is a distributed shared memdesc slice.
- Permanent tests updated:
  - `python/test/gluon/test_tmem_runtime_matrix.py`
    - 2-CTA no-scales copy runtime assertions now pin the actual post-fix
      PTX/LLIR shape:
      - `fence.proxy.async.shared::cluster` before copy,
      - `barrier.cluster.arrive.aligned` / `wait.aligned` before the first
        `tcgen05.cp.cta_group::2.128x256b`,
      - no relaxed-arrive barrier on this path.
  - `test/TritonNvidiaGPU/membar-cluster.mlir`
    - added a dedicated regression proving `ttg.local_store` feeding 2-CTA
      `ttng.tmem_copy` gets a pre-copy `ttng.cluster_barrier`;
    - refreshed adjacent multi-CTA MMA/TMA expectations to the current pass
      behavior.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `cd $(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())') && ninja triton-opt && lit -v test/TritonNvidiaGPU/membar-cluster.mlir test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    -> `3 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales_twocta_codegen'`
    -> `28 passed, 1688 deselected`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_x1_f16_roundtrip or ldst_scales_direct_roundtrip or ldst_scales_variant_sweep or ldst_scales_variant_reports_clean_unsupported or cp_no_scales_twocta_codegen or cp_scales_warpx4 or cp_scales_warpx4_via_scaled_mma_copy_matrix or cp_scales_warpx4_via_scaled_mma_geometry_sweep or mma_twocta_plain_kinds'`
    -> `130 passed, 1586 deselected`
- BUG status:
  - close `BUG`: 2-CTA no-scales `tcgen05.copy` cold-start corruption on the
    `cta_group::2.128x256b` path.
  - open frontier remains `warpx2` descriptor synthesis / lowering, not this
    synchronized `128x256b` family.

## 2026-03-26 18:27 UTC: broaden post-fix runtime sweep and repair copy-test harness

- Broadened executable validation after the 2-CTA copy barrier fix with a fresh
  runtime slice over plain copies, scales copies, and 2-CTA MMA:
  - selector:
    `cp_no_scales_twocta_codegen or cp_scales_warpx4 or cp_scales_warpx4_via_scaled_mma_copy_matrix or cp_scales_warpx4_via_scaled_mma_geometry_sweep or cp_no_scales_linear_32bit_dtypes or mma_twocta or mma_twocta_plain_kinds`
  - result:
    `103 passed, 1613 deselected`
- During that sweep, hit a transient TEST BUG in
  `python/test/gluon/test_tmem_runtime_matrix.py`:
  - `tmem_copy_no_scales_kernel` and `tmem_copy_no_scales_linear_kernel` had
    lost their local `smem_layout = ttgl.NVMMASharedLayout(...)` definitions
    during the earlier `fence_async_shared` refactor.
  - failure mode was clean Python/AST compile failure, not a compiler backend
    bug:
    `NameError('smem_layout is not defined')`.
  - restored the missing `smem_layout` locals and reran the slice cleanly.
- Validation added on top of the earlier focused copy check:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales_twocta_codegen or cp_scales_warpx4 or cp_scales_warpx4_via_scaled_mma_copy_matrix or cp_scales_warpx4_via_scaled_mma_geometry_sweep or cp_no_scales_linear_32bit_dtypes or mma_twocta or mma_twocta_plain_kinds'`
    -> `103 passed, 1613 deselected`
- BUG ledger:
  - no new compiler BUG found in this slice after the barrier fix;
  - the only new failure was the repaired runtime-test harness bug above.

## 2026-03-26 18:32 UTC: current warpx2 compiler blocker audit

- Code-path audit of the current tree:
  - family classification already exists in
    `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp::getTMemCopyAtom(...)`
    for `warpx2::02_13.64x128b` and `warpx2::01_23.64x128b`.
  - PTX emission already exists in
    `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp::createTcgen05Cp(...)`.
  - the real blocker is shared-descriptor synthesis:
    - verifier side:
      `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp::TMEMCopyOp::verify()` gates all
      non-scales copy families through
      `canRepresentAsMMASmemDescriptor(cvtWarp, instrShape, bitwidth, 0, 5)`;
    - lowering side:
      `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp::copySharedToTmem(...)`
      rebuilds the same `cvtWarp` and requires
      `DotOpMmaSmemLoader::build(..., instrShape={32, bCol/bitwidth}, MNdim=0, mmaVersion=5)`.
    - both helpers ultimately use the generic MMAv5 shared-descriptor search in
      `DotOpToLLVM/MMAHelpers.h::{canRepresentAsMMASmemDescriptor,getDescriptor}`,
      which only models standard 32-row core-matrix descriptors plus LBO/SBO
      padding. There is no `warpx2`-specific descriptor/address synthesis path.
- Empirical cross-check:
  - direct PTX witnesses in the initiative logs still show
    `tcgen05.cp.warpx2::{02_13,01_23}.64x128b` are real and deterministic at the
    ISA level, but simple opcode substitution on the existing descriptor does
    not produce correct full-copy semantics.
  - bounded refresh on current tree:
    `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_warpx2_subslice.py --parent-rows 128 --max-layouts 32 --start-rows 0 --device cuda`
    now fails cleanly at verifier time with
    `The source shared layout does not lower to Triton's currently supported tcgen05.copy.warpx4.32x128b descriptor family for tensor memory scales.`
    No late `failed to legalize operation`, no crash.
- Current engineering conclusion:
  - `warpx2` support is blocked by missing family-specific descriptor/address
    synthesis, not by opcode emission or classifier recognition.
  - next enabling work needs a dedicated `warpx2` descriptor model in the
    MMAv5 SMEM loader / descriptor matcher path, plus matching pointer stepping
    semantics in `copySharedToTmem(...)`.

## 2026-03-26 18:33 UTC: refresh direct-PTX warpx2 / 4x256 semantics boundary

- Re-ran direct PTX probes on GPU 3 with the current toolchain:
  - `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_variants.py`
  - `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_semantics.py --output-json /tmp/probe_cp_direct_ptx_semantics_gpu3.json`
- Fresh empirical result matches the earlier initiative probes:
  - `tcgen05.cp.cta_group::1.warpx2::02_13.64x128b` assembles and launches, but deterministically populates only the `02` / `13` warp-pair pattern; it is not numerically equivalent to the working warpx4 path.
  - `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b` assembles and launches, but deterministically populates only the `01` / `23` warp-pair pattern; again not a drop-in replacement.
  - `tcgen05.cp.cta_group::1.4x256b` assembles and launches, but remains non-deterministic and does not match the working reference mapping.
- Compiler-side response:
  - tightened the clean-unsupported diagnostic in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` for descriptor-build failures on non-scales copy families so future reachable `warpx2` cases explain that Triton still needs family-specific shared-descriptor/address synthesis; this is not just a generic descriptor mismatch.
- Test maintenance:
  - synchronized `test/TritonNvidiaGPU/invalid.mlir` with the current canonical TMEM verifier diagnostics for `ttng.tmem_subslice` invalid cases.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `lit -v test/TritonNvidiaGPU/invalid.mlir`
    -> `1 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales_warpx2_candidate_reports_clean_error'`
    -> `1 passed, 1715 deselected`
- BUG frontier update:
  - keep `warpx2` open as a compiler lowering/design task, not an ISA availability question.
  - keep `4x256b` blocked behind further semantics work; current direct-PTX behavior is still not deterministic enough to expose in Triton.

## 2026-03-26 20:10 UTC: overlay warpx2 semantics probe and verifier-note refresh

- Added focused overlay probe infrastructure:
  - script:
    `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_overlay.py`
  - durable result captures:
    - `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_overlay_gpu3.json`
    - `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_overlay_gpu3_deltas.json`
- Probe design:
  - build one kernel with two sequential scales-style `tcgen05.copy` operations;
  - first copy is a known-good warpx4 seed from sentinel input `A`;
  - second copy is patched to a single active `warpx2` site sourcing
    diagnostic input `B = column_id`;
  - read back the `128x32` TMEM alias view and classify each `32x16` half as:
    `A`, `B`, or `other`.
- Stable hardware facts from the overlay probe:
  - second-copy PTX sites reuse the same base descriptor/address registers as
    the first copy:
    - site 0: `[ %r14 + 0 ], %rd3`
    - site 1: `[ %r12 + 0 ], %rd4`
  - single-site `warpx2` writes are structured but partial:
    - `warpx2::02_13` with `%rd3 @ %r14 + 0` writes the left half of chunks
      `0` and `2`;
    - `warpx2::02_13` with `%rd4 @ %r14 + 4` writes the right half of chunks
      `0` and `2`;
    - `warpx2::01_23` with `%rd3 @ %r14 + 0` writes the left half of chunks
      `0` and `1`;
    - `warpx2::01_23` with `%rd4 @ %r14 + 4` writes the right half of chunks
      `0` and `1`;
    - `site0` vs `site1` made no observable difference for these overwrite
      maps.
  - tested intermediate TMEM destination deltas `+1` and `+2` are not viable:
    they fault on hardware with `cuCtxSynchronize failed: misaligned address`
    for both `warpx2` families and both `%rd3` / `%rd4`.
- Two-message follow-up on the inherited base schedule:
  - `02_13` with `(%rd3 @ %r14, %rd4 @ %r12)` fully populates chunks `0` and
    `2`, but leaves chunks `1` and `3` wrong.
  - `01_23` with `(%rd3 @ %r14, %rd4 @ %r12)` fully populates chunks `0` and
    `1`, but leaves chunks `2` and `3` wrong.
  - mixed-family two-message schedules (`02_then_01`, `01_then_02`) are still
    not exact.
- Current engineering conclusion is now stronger than “needs more work”:
  - the inherited two-message warpx4 descriptor/address schedule is
    semantically insufficient for `warpx2`;
  - valid `warpx2` support requires family-specific descriptor/address
    synthesis and likely a different multi-message copy plan, not just opcode
    substitution or a small TMEM address tweak.
- Compiler response:
  - strengthened the clean-unsupported `warpx2` note in
    `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` so it now reports that:
    - the inherited warpx4 schedule only fills selected alias chunk pairs, and
    - intermediate `+1/+2` TMEM deltas fault with misaligned addresses.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && ninja triton-opt && lit -v test/TritonNvidiaGPU/invalid.mlir`
    -> `1 passed`
- BUG ledger:
  - no new Triton compiler BUG found in this pass;
  - the new failures are direct-PTX schedule candidates that fault on hardware
    (`misaligned address`), which should remain logged as invalid candidate
    schedules rather than compiler regressions.

## 2026-03-26 (strict explicit-layout policy for plain TMEM ld/st/alloc)
- Policy change requested by user:
  - if a user explicitly requests a TMEM register layout that is not directly
    TMEM-compatible, the compiler must report an error;
  - the compiler must not silently insert a fallback `convert_layout`.
- Compiler changes:
  - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
    - removed deferred layout feasibility for plain `ttng.tmem_load`,
      `ttng.tmem_store`, and initialized `ttng.tmem_alloc`;
    - these ops now fail in the verifier as soon as the explicit tensor layout
      is not directly compatible with the TMEM memdesc;
    - diagnostics now suggest either using one of the listed potential TMEM
      layouts or inserting `convert_layout` explicitly.
  - `third_party/nvidia/backend/compiler.py`
    - removed the late `passes.ttgpuir.add_relayout_tritongpu(pm)` call from
      `make_llir`;
    - this was the backend path that had been silently repairing plain TMEM
      ld/st/alloc by inserting `convert_layout` underneath the user.
- Test updates:
  - `python/test/gluon/test_tmem_runtime_matrix.py`
    - converted positive copy/scales kernels that previously relied on implicit
      blocked-layout fallback to use explicit `get_reg_layout(...)` plus
      explicit `convert_layout`;
    - converted the old blocked-fallback runtime regression into a clean error
      regression that accepts the current frontend failure boundary
      (`RuntimeError` during IR materialization or `CompilationError` later).
  - `test/TritonNvidiaGPU/invalid.mlir`
    - updated the impossible-layout note to the new verifier wording.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `python3 -m py_compile third_party/nvidia/backend/compiler.py python/test/gluon/test_tmem_runtime_matrix.py`
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && ninja triton-opt && lit -v test/TritonNvidiaGPU/invalid.mlir`
    -> `1 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'blocked_layout_reports_clean_error or cp_no_scales_indexed_view_canonicalized or cp_no_scales_linear_indexed_view or cp_128x128 or cp_scales_warpx4 or ldst_scales_direct_roundtrip or ldst_scales_variant_sweep'`
    -> `93 passed, 1623 deselected`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_descriptor_multidim_slice_identity_reports_clean_error or ldst_descriptor_multidim_slice_reports_clean_error or ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout or ldst_twocta_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout'`
    -> `27 passed, 1689 deselected`
  - full matrix rerun with xdist:
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -n 8 -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py`
      -> `1598 passed, 117 skipped, 1 xfailed in 60.41s`

## 2026-03-26 (checkpoint cleanup before TMEM inference refactor)
- Goal for this checkpoint:
  - make the dirty post-`cfd703808` tree coherent before starting the next
    TMEM inference/canonicalization refactor pass;
  - keep only validated changes and drop the failed higher-rank TMEM view
    experiment.
- Code changes kept:
  - `lib/Dialect/TritonGPU/IR/Dialect.cpp`
    - kept the shared-memory reshape fallback that rematerializes
      `SharedLinearEncodingAttr` from `reshapeLayout(...)`;
    - this fixed the `cp_scales_warpx4_via_scaled_mma_geometry_sweep`
      regression.
- Failed experiment dropped:
  - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
    - reverted the local attempt to stop stripping zero row/col bases in
      `tryMakeTMemViewEncoding(...)`;
    - that experiment correctly blocked one bad multidim-slice case, but it
      also over-rejected legitimate higher-rank indexed TMEM views.
- Test cleanups:
  - `test/TritonNvidiaGPU/ops.mlir`
    - converted unsupported positive TMEM ld/st-on-view examples into pure
      memdesc-view IR checks;
    - kept positive TMEM access coverage only for layouts the strict verifier
      actually accepts.
  - `test/TritonNvidiaGPU/invalid.mlir`
    - kept updated diagnostics for the new strict explicit-layout policy.
  - `test/Analysis/test-buffer-region.mlir`
    - synced expected TMEM region remarks to the current descriptor semantics.
  - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - updated stale `tc_gen5_mma_subslice_acc` checks to the current immediates;
    - removed dead TMEM conversion chunks that no longer emit LLVM in the
      current pipeline;
    - updated surviving `ttng.tmem_subslice` LLVM checks to the current
      lowering, which returns the base pointer unchanged for descriptor-only
      view helpers and carries the slice in the descriptor semantics.
  - `python/test/gluon/test_tmem_runtime_matrix.py`
    - relaxed a couple of TTGIR assertions so legacy TMEM sugar still checks
      for `tensor_memory_encoding` while linear layouts check for
      `tensor_memory_linear`.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && lit -v -j 32 test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir test/Analysis/test-buffer-region.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    -> `4 passed`
  - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. python3 -m pytest -n 4 -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_scales_warpx4_via_scaled_mma_geometry_sweep or ldst_x1_f32_roundtrip'`
    -> `35 passed in 7.60s`

- TMEM frontend/runtime sweep refresh (2026-03-27 01:10 UTC):
  - frontend parser alignment:
    - updated the TMEM subslice frontend parser coverage to use explicit
      `dim=1` for the legacy N-slice intent;
    - current IR now prints generic `ttg.memdesc_subslice` for that case, not
      legacy `ttng.tmem_subslice`.
  - runtime matrix updates:
    - fixed stale `cp` indexed-view expectations:
      - legacy indexed view stays in `tensor_memory_encoding` sugar and lowers
        to `tcgen05.cp.cta_group::1.128x128b`;
      - linear lifted indexed views now expect the narrower message/counts that
        current lowering actually emits (`128x128b` for the small case,
        `16 * 128x256b` for the 128x128 linear case).
    - fixed the broken linear-subslice-view copy kernel, which had been
      xfail-ing only because `smem_layout` was undefined; expanded it into a
      real dtype/swizzle sweep.
    - added a positive MMA runtime case where the accumulator is a TMEM
      `memdesc_index` view from a larger descriptor, covering both legacy TMEM
      sugar and lifted `tensor_memory_linear`.
  - BUG status recorded from the full GPU matrix:
    - multidimensional TMEM slice/view cases that use generic descriptor views
      are now often accepted by the compiler, but a large subset still writes
      incorrect values at runtime instead of failing cleanly;
    - these are now kept as explicit `pytest.xfail` BUG markers in
      `python/test/gluon/test_tmem_runtime_matrix.py`, not silent failures.
    - representative bug classes:
      - higher-rank `memdesc_subslice` / `memdesc_index` compositions on TMEM
        descriptors compile but update the wrong physical columns;
      - direct multidimensional slice views on `mixed` TMEM layouts compile but
        update the wrong region;
      - two-CTA MMAv5 higher-rank view cases still fail, but the current clean
        diagnostic is a CGA mismatch (`Layout has 1 CTAs per CGA, but the
        context requires 2 CTAs per CGA.`), not the older block-basis error.
  - Validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `PYTHONPATH=python:. python3 -m pytest -q -s --tb=short python/test/gluon/test_frontend.py::test_tmem_subslice_reg_layout_constexpr`
      -> `1 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -n 8 -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'block_descriptor_reports_clean_error or cp_no_scales_linear_indexed_view or cp_no_scales_indexed_view_canonicalized or cp_no_scales_linear_subslice_view or mma_indexed_acc_view'`
      -> `13 passed in 9.25s`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -n 8 -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'descriptor_multidim_slices_reports_clean_error or descriptor_multidim_slice_reports_clean_error or twocta_mmav5_descriptor_higher_rank_reports_clean_error'`
      -> `13 passed, 49 xfailed in 13.14s`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -n 8 -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py`
      -> `1557 passed, 117 skipped, 49 xfailed in 31.56s`

## 2026-03-27
- Fixed the multidimensional TMEM slice/view wrong-code bucket in
  `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp` by validating that a
  synthesized standalone TMEM-linear subslice encoding preserves the source
  logical-to-physical TMEM projection. If the projected view would densify
  sparse physical TMEM row/col bits, inference now rejects it as
  `unsupported tensor memory memdesc_subslice view` instead of silently
  re-encoding it to a different physical mapping.
- Updated the higher-rank multidimensional TMEM runtime-matrix clean-error
  assertions in `python/test/gluon/test_tmem_runtime_matrix.py` to also accept
  the clean `memdesc_index` inference failure that reports the sliced view is
  not representable as a standalone TMEM-linear layout.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - focused multidimensional TMEM slice buckets:
    `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -n 8 -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_ldst_descriptor_multidim_slices_reports_clean_error or test_tmem_runtime_matrix_ldst_twocta_descriptor_multidim_slices_reports_clean_error or test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_reports_clean_error'`
    -> `50 passed`
  - full canonical matrix:
    `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -n 8 -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py`
    -> `1606 passed, 117 skipped in 97.69s`
  - frontend parser/inference spot-check:
    `PYTHONPATH=python:. python3 -m pytest -s --tb=short -q python/test/gluon/test_frontend.py::test_tmem_subslice_reg_layout_constexpr`
    -> `1 passed`
- `test/TritonNvidiaGPU/ops.mlir` triage refresh:
  - `BUILD_DIR=/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`
  - raw `./bin/triton-opt /root/code/triton/test/TritonNvidiaGPU/ops.mlir`
    still exits `0` and emits `0` bytes on stdout/stderr, so `lit -v
    test/TritonNvidiaGPU/ops.mlir` still fails with `FileCheck error:
    '<stdin>' is empty.`
  - the issue is not driver-wide: `printf 'module {}\n' | ./bin/triton-opt`
    prints the trivial module normally, and `./bin/triton-opt --mlir-print-op-generic
    /root/code/triton/test/TritonNvidiaGPU/ops.mlir` is still empty.
  - post-link artifact note: `bin/triton-opt` lost its executable bit after the
    rebuild in this environment; adding `chmod +x` restored executability but
    did not change the empty-stdout behavior.

- Local confirmation on the current worktree after reverting the discarded projection experiment:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -n 8 -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py`
    -> `1606 passed, 117 skipped in 79.68s`
  - `PYTHONPATH=python:. python3 -m pytest -s --tb=short -q python/test/gluon/test_frontend.py::test_tmem_subslice_reg_layout_constexpr`
    -> `1 passed in 0.38s`
  - local `ops.mlir` repro in this worktree differs slightly from the earlier delegated run:
    - `build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt test/TritonNvidiaGPU/ops.mlir`
      exits `1` with both stdout/stderr empty,
    - `--mlir-print-op-generic` is also empty with exit `1`, and
    - `lit -v test/TritonNvidiaGPU/ops.mlir` still fails because `FileCheck` receives empty stdin.

## 2026-03-27 03:50 UTC
- Fixed the remaining `test/TritonNvidiaGPU/ops.mlir` verifier/roundtrip failure.
  - Root cause:
    - generic `ttg.memdesc_subslice` cases in `ops.mlir` still declared
      source-sugar TMEM-linear result types after slicing to `128x64`, while
      current generic TMEM subview inference expects the canonical smaller
      TMEM-linear result encoding;
    - verifier/equality checks also compared some semantically identical
      TMEM-linear encodings by raw attr identity instead of normalized
      TMEM-linear layout.
  - Fixes landed:
    - `test/TritonNvidiaGPU/ops.mlir`
      - canonicalized the generic TMEM view paths to use `#tmem_linear_small`
        on the `128x64` result types;
    - `lib/Dialect/TritonGPU/IR/Ops.cpp`
      - `memdesc_{reshape,reinterpret,index,subslice}` now emit an explicit
        inferred-vs-declared memdesc type diagnostic instead of silently
        returning `failure()`;
    - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
      - TMEM layout equality now compares normalized TMEM-linear layouts, not
        only raw canonical-attr identity;
    - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
      - `ttng.tmem_subslice` now routes canonical-TMEM equality through the
        same normalized layout comparison.
  - Validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `BUILD_DIR=/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`
    - `cd "$BUILD_DIR" && ./bin/triton-opt /root/code/triton/test/TritonNvidiaGPU/ops.mlir`
      -> `rc=0`, `stdout_bytes=19192`, `stderr_bytes=0`
    - `cd "$BUILD_DIR" && lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir test/Analysis/test-buffer-region.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
      -> `4 passed`
- Revalidated the TMEM runtime baseline after the verifier/layout-equality fixes.
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -n 8 -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py`
    -> `1606 passed, 117 skipped in 96.30s`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -s --tb=short -q python/test/gluon/test_frontend.py::test_tmem_subslice_reg_layout_constexpr`
    -> `1 passed in 0.37s`
- Remaining plain TMEM descriptor/load-store clean negatives were reclassified.
  - Identity quarter-tile `32x32` reshape/slice views, lifted-layout `dim0` /
    `half_rows` views, scrambled descriptor views, and the block-basis
    reinterpret frontier now all look like “no single direct TMEM register
    layout exists for this final view” cases, not residual `TensorMemoryUtils`
    typing bugs.
  - Under the current plain-TMEM contract (no silent layout repair for
    `tmem_load` / `tmem_store` / initialized `tmem_alloc`), these should be
    treated as direct-lowering / atom-family limits unless we later add a new
    composite multi-message reg-layout/lowering model.
- `tcgen05.copy.warpx2` remains the only confirmed supportable compiler gap.
  - Current evidence refresh:
    - verifier/classifier in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` already
      recognizes `warpx2::{02_13,01_23}.64x128b`;
    - LLVM lowering in
      `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
      already has opcode emission for those families;
    - direct PTX / initiative notes still show the inherited warpx4
      descriptor/address plan is semantically insufficient.
  - Current conclusion:
    - `warpx2` is not ISA-impossible;
    - enabling it needs family-specific shared-descriptor/address/message
      synthesis in the copy verifier/lowering path, not an opcode swap.

## 2026-03-27 04:44 UTC
- Added durable bounded PTX-probe helpers for the remaining `warpx2` gap.
  - new experiment scripts:
    - `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_overlay_both_bases.py`
    - `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_single_site_halves.py`
    - `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_4site.py`
  - quick validation:
    - `python3 -m py_compile .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_overlay_both_bases.py`
    - `python3 -m py_compile .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_single_site_halves.py`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_4site.py --limit 1 --output-json /tmp/probe_cp_direct_ptx_4site_smoke.json`
      -> `tested=1`, no exact match, exact halves only on the `02` family chunk
      pair (`0/2`) as expected from the earlier overlay work.
- Persisted the aligned both-base overlay sweep for the second-copy descriptor pair.
  - command:
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_overlay_both_bases.py --output-json .codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_overlay_both_bases_gpu0.json`
  - result:
    - `64 / 64` plans passed / launched;
    - `0` clean quadrant mappings;
    - only six unique overwrite signatures appear under aligned deltas;
    - `%r12 + 0` duplicates the productive `%r14 + 4` right-half signatures
      from the earlier overlay probe and does not unlock a new clean family.
  - strongest new conclusion:
    - within the current second-copy descriptor pair (`%rd3/%rd4`), no aligned
      one-message plan writes chunk `3`, left half as an exact expected tile.
- Persisted the aligned one-copy / live-descriptor sweep.
  - command:
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_single_site_halves.py --output-json .codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_single_site_halves_gpu0.json`
  - result:
    - `28 / 32` plans passed; `4` plans still faulted and remain invalid;
    - new clean one-message signatures exist on the live `%rd3` site:
      - `warpx2::02_13` with `%r18 + 4` or `%r7 + 0` gives `0B/00/0B/00`
        (clean exact right halves for chunks `0` and `2`);
      - `warpx2::01_23` with `%r18 + 4` or `%r7 + 0` gives `0B/0B/00/00`
        (clean exact right halves for chunks `0` and `1`).
  - combined with the both-base overlay sweep:
    - the aligned one-message coverage now reaches every exact half except
      chunk `3`, left half;
    - this is the current hard frontier for `warpx2`.
- Current engineering read after the new probes:
  - there is now strong evidence that the remaining blocker is not “choose a
    better aligned schedule over the current live descriptors”;
  - if the bounded four-site search over those same live descriptors fails, the
    next step should be descriptor synthesis / discovery for the missing `3L`
    quadrant rather than more schedule reshuffling over the current descriptor
    pair.

## 2026-03-27 04:46 UTC
- Ran the bounded four-site search over the current live descriptor/base family.
  - command:
    - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_4site.py --output-json .codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_4site_gpu1.json`
  - search space:
    - same input buffer fed through both copies of the two-copy overlay kernel;
    - all four emitted `tcgen05.cp` sites remain enabled;
    - each site keeps its live base register and live descriptor register;
    - per-site family `{warpx2::02_13, warpx2::01_23}`;
    - per-site aligned destination delta `{0,4}`.
  - result:
    - `256 / 256` plans launched cleanly;
    - `0` exact matches;
    - `0` launch failures;
    - the best bounded plan still stops at `2040` mismatched bytes.
- Stronger boundary after the bounded four-site result:
  - even four-message schedule reshuffling over the current live
    descriptor/base-register family is insufficient in the aligned bounded
    space;
  - combined with the earlier one-message probes, this moves the remaining
    `warpx2` frontier from “maybe there is a better schedule over the current
    live descriptors” to “we likely need descriptor synthesis / descriptor
    discovery for the missing quadrant”.

## 2026-03-27 04:59 UTC
- Ran the live-bit descriptor cube search for the missing `3L` quadrant.
  - command:
    - `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_descriptor_values.py --output-json .codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_descriptor_values_gpu3_fast.json`
  - search space:
    - one active `warpx2` site at a time on the one-copy kernel;
    - aligned address immediates `{0,4}`;
    - live descriptor bit subcube only, derived from the current `%rd2/%rd3`
      pair’s differing bits `{5,51}`:
      `0x0000400800000000`,
      `0x0000400800000020`,
      `0x0008400800000000`,
      `0x0008400800000020`.
  - result:
    - `112 / 128` plans passed; `16` still faulted;
    - `0` exact hits on target quadrant chunk `3`, left half;
    - `0` clean target hits;
    - `0` isolated target hits.
  - strongest read:
    - the missing `3L` quadrant is not hidden in the current live
      `{baseAddress, matrixBaseOffset}` cube;
    - the best near misses still only reproduce already-known clean right-half
      signatures such as `0B/0B/00/00`.
- Next bounded descriptor neighborhood after the live-bit miss:
  - the most plausible next fields are
    `leadDimensionBaseOffset` / `strideDimensionBaseOffset` around the clean
    `%rd3` seed, not more schedule reshuffling or more
    `{baseAddress, matrixBaseOffset}` permutations.

## 2026-03-27 05:02 UTC
- Ran the `%rd3` seed `leadDimensionBaseOffset` / `strideDimensionBaseOffset` search.
  - command:
    - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_descriptor_lbo_sbo.py --output-json .codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_descriptor_lbo_sbo_gpu1.json`
  - search space:
    - clean `%rd3` seed only (`0x0008400800000020`);
    - active site `1` only;
    - clean address seeds `%r18 + 4` and `%r7 + 0`;
    - `warpx2::{02_13,01_23}`;
    - `leadDimensionBaseOffset ∈ {0,4,8}`;
    - `strideDimensionBaseOffset ∈ {4,8,12,16}`.
  - result:
    - `36 / 48` plans passed; `12` faulted;
    - `0` exact hits on target quadrant chunk `3`, left half;
    - `0` clean target hits;
    - `0` isolated target hits.
- Current read after the field-aware miss:
  - bounded `lbo/sbo` perturbations around the clean `%rd3` seed are still not
    enough to expose the missing `3L` quadrant;
  - the next plausible PTX-side degree of freedom is now source-coordinate
    generation, i.e. descriptor values equivalent to changing `smemLoad(a, b)`
    around the clean `%rd3` seed, not more local field tweaking or more
    schedule reshuffling.

## 2026-03-27 06:18 UTC
- Reframed the remaining `warpx2` work against the public TMEM descriptor API
  rather than direct PTX reachability alone.
  - bounded public shared-layout classification results:
    - no-scales canonical destination (`128x4`, `f32`):
      - pure interleavings: `36` layouts checked, no `warpx2`;
      - one-mixed: `1134` tested / `252` surjective, no `warpx2`;
      - two-mixed: `254016` tested / `22680` surjective, no `warpx2`.
    - scales destination (`64x16`, `i8`):
      - pure interleavings: `210` layouts checked, all reachable families
        classify to `multicast=3` / `warpx4`;
      - one-mixed: `2400` tested / `480` surjective, no `warpx2`;
      - two-mixed: `1166400` tested / `90720` surjective, no `warpx2`.
  - bounded explicit TMEM-layout results:
    - raw `warpx2`-looking row-zero layouts exist for `[128,4]`, `[128,8]`,
      and `[128,16]`, but they are non-surjective:
      - `[[1,0],[2,0],[4,0],[8,0],[16,0],[0,0],[32,0]]`
        -> raw `multicast=1` / `warpx2::01_23`;
      - `[[1,0],[2,0],[4,0],[8,0],[16,0],[32,0],[0,0]]`
        -> raw `multicast=2` / `warpx2::02_13`.
    - the real frontend parser probe (`/tmp/probe_warpx2_explicit_tmem.py`)
      confirms these layouts fail at `allocate_tensor_memory` with
      `The layout must be surjective`; no `tcgen05.cp` opcode is emitted.
    - surjective explicit TMEM searches still found no `warpx2` hits:
      - `[128,4]`: `254016` tested / `22680` surjective;
      - `[128,8]`: `893025` tested / `68040` surjective;
      - `[128,16]`: `2371600` tested / `152460` surjective.
- Updated test coverage and refreshed stale negative expectations.
  - edited `python/test/gluon/test_frontend.py`:
    - added a parser regression for the two non-surjective row-zero explicit
      TMEM layouts that raw-classify as `warpx2`.
  - edited `python/test/gluon/test_tmem_runtime_matrix.py`:
    - refreshed the historical `warpx2` negative assertions to match the
      current clean verifier diagnostics (`descriptor plan` wording and
      family-agnostic scales message fragments).
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `PYTHONPATH=python:. python3 -m pytest -s --tb=short python/test/gluon/test_frontend.py -k 'warpx2_like_rows_report_non_surjective'`
    - `2 passed, 199 deselected`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_scales_layout_probe or cp_scales_unsupported_layout_raises_runtimeerror_parse or cp_no_scales_warpx2_candidate_reports_clean_error'`
    - `4 passed, 1719 deselected`
  - `CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=python:. python3 -m pytest -n 8 -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py`
    - `1606 passed, 117 skipped in 83.86s`
- Current conclusion:
  - direct PTX `tcgen05.cp.warpx2::{02_13,01_23}.64x128b` remains a real ISA
    frontier, but it is not a remaining blocker for “all public TMEM
    descriptor API code”;
  - under the current user-visible shared/TMEM layout constructors, bounded
    reachable searches found no surjective path that lowers to `warpx2`;
  - making `warpx2` user-reachable would require more than late opcode
    selection: at minimum a legal user-visible layout / descriptor
    representation that maps to `warpx2`, plus family-specific
    descriptor/address/message synthesis.
