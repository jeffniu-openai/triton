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
- Requested follow-up TODO:
  - evaluate an opt-in `warpx2` enablement path that permits non-surjective
    layouts only for the copy-direction use case, paired with the required
    family-specific `warpx2` codegen changes;
  - this would be a semantic/API extension, not just a late-lowering tweak,
    because current TMEM layout verification assumes tensor-storage
    surjectivity/bijectivity.
  - broaden the project scope from “all current public TMEM descriptor API
    code” to “all ISA/PTX-level `tcgen05` families reachable from some legal
    user input”, including:
    - `tcgen05.mma` and `tcgen05.mma_scaled`: support all PTX-level MMAv5
      instruction shapes/layouts that are ISA-legal, expand codegen to accept
      arbitrary user layouts that map to them, and fuzz at the user level;
    - `tcgen05.ld/st`: support all PTX-level ISA-legal atom/layout families
      reachable from arbitrary user TMEM layouts and descriptor-view chains,
      with user-level fuzzing;
    - `tcgen05.copy`: extend the same reachability/fuzzing model to the
      remaining PTX-level copy families.
  - broaden the initiative goal from the current runtime-matrix frontier to the
    full PTX-level tcgen05 surface:
    - `cp`: make every ISA/PTX-level family reachable from some legal user
      layout, then support arbitrary user layouts that map to those families;
    - `mma` / `mma_scaled`: support all PTX-level MMAv5 instruction
      shapes/layouts that are ISA-legal and expand user-layout lowering beyond
      the current MMAv5-compatible canonical subset;
    - `ld/st`: support all ISA-legal atom/layout combinations plus arbitrary
      user descriptor layouts that factor to those atoms without wrong-code;
    - add user-level fuzzing to saturate those expanded surfaces and keep the
      parser/verifier/lowering no-crash invariants.

2026-03-27 20:55Z - MMAv5 accumulator N-subslice support

- Landed a narrow MMAv5 accumulator-view widening that keeps the old generic
  TMEM matcher untouched.
  - added `matchMMAv5AccumulatorEncoding(gpu::MemDescType)` in
    `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` and declared it in
    `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h`;
  - helper behavior:
    - first try the existing logical-shape
      `matchTensorMemoryLegacyEncoding(...)`;
    - if that fails and the encoding is `tensor_memory_linear`, retry on the
      memdesc `allocShape`;
    - only allow the fallback when the view preserves the full `M` extent of
      the allocation and narrows only `N`.
- Routed accumulator-only users through the new helper:
  - plain `tc_gen5_mma` verifier;
  - `tc_gen5_mma_scaled` verifier;
  - MMAv5 LLVM lowering;
  - accumulator-derived row constraints in `TensorMemoryAllocation.cpp`;
  - accumulator-derived LHS promotion policy in `PromoteLHSToTMem.cpp`.
- Added coverage for public accumulator subviews from both legacy and
  TMEM-linear parents.
  - `test/TritonNvidiaGPU/ops.mlir`:
    - plain `ttg.memdesc_subslice` -> `ttng.tc_gen5_mma` verifier/roundtrip
      regression for a `128x64` TMEM-linear accumulator view with allocShape
      `128x128`.
  - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`:
    - plain MMAv5 lowering regression for a second-half `N` subslice from a
      TMEM-linear accumulator parent;
    - scaled MMAv5 lowering regression for the same second-half `N` subslice
      pattern.
  - `python/test/gluon/test_tmem_runtime_matrix.py`:
    - added end-to-end runtime coverage for plain `tcgen05_mma` with indexed
      and second-half `N`-subslice accumulators from legacy and TMEM-linear
      parents;
    - added end-to-end runtime coverage for `tcgen05_mma_scaled` with a
      second-half `N`-subslice accumulator from legacy and TMEM-linear parents.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && lit -v test/TritonNvidiaGPU/ops.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `2 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_indexed_acc_view or mma_subslice_acc_view or mma_scaled_minimal or mma_scaled_subslice_acc_view'`
    - `7 passed`
- Current boundary after this slice:
  - MMAv5 accumulator `D` now supports allocation-compatible `N`-subslice
    views from TMEM-linear parents;
  - this does **not** yet widen A-in-TMEM, non-legacy-equivalent accumulator
    families, or arbitrary shared-memory A/B layouts.

2026-03-27 08:25Z - MMAv5 accumulator view composition expansion; LHS views quarantined

- Kept the new accumulator-side normalization path and expanded its validated
  public-view surface without widening the unsafe LHS path.
  - accumulator `D` coverage now includes:
    - indexed views from lifted parents for plain `tcgen05_mma`;
    - indexed views from lifted parents for `tcgen05_mma_scaled`;
    - second-half `N` subslices from legacy and TMEM-linear parents for plain
      and scaled MMA;
    - composed `index(1).slice(64, 64, dim=1)` views from lifted parents for
      plain and scaled MMA.
- Converted the attempted LHS-TMEM subslice expansion into explicit clean
  negatives instead of leaving a wrong-code path open.
  - plain runtime-matrix coverage now asserts
    `LHS tensor memory views are currently unsupported for MMAv5...` and checks
    that we do not fall through to `PassManager::run failed` / `Assertion`;
  - `test/TritonNvidiaGPU/invalid.mlir` now covers the same verifier boundary
    for both `ttng.tc_gen5_mma` and `ttng.tc_gen5_mma_scaled`.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `3 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_indexed_acc_view or mma_subslice_acc_view or mma_lhs_subslice_view_reports_clean_error or mma_scaled_minimal or mma_scaled_subslice_acc_view'`
    - `9 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled_indexed_acc_view or mma_indexed_subslice_acc_view'`
    - `4 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled_indexed_subslice_acc_view'`
    - `2 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py`
    - `1618 passed, 117 skipped in 371.45s`
- Current boundary after this validation:
  - MMAv5 accumulator `D` supports indexed, `N`-subslice, and composed
    indexed-then-`N`-subslice public views from legacy / TMEM-linear parents
    when the physical allocation remains legacy-MMAv5-compatible;
  - LHS tensor-memory views still miscompile if accepted, so they remain
    verifier-rejected for both plain and scaled MMAv5;
  - this still does not widen arbitrary non-legacy-equivalent accumulator
    families or arbitrary shared-memory A/B layouts.

2026-03-27 - MMAv5 accumulator view-composition follow-up

- Expanded validated MMAv5 accumulator-view coverage without widening the
  underlying matcher again.
  - new runtime-matrix positives for `tcgen05_mma_scaled` with indexed
    accumulator `D` views from:
    - legacy lifted parents;
    - lifted `tensor_memory_linear` parents.
  - new runtime-matrix positives for composed public accumulator view chains:
    - plain `tcgen05_mma` with `acc.index(...).slice(..., dim=1)`;
    - scaled `tcgen05_mma_scaled` with the same
      `acc.index(...).slice(..., dim=1)` chain;
    - each validated for both legacy and lifted `tensor_memory_linear`
      parents.
- Locked down the newly discovered A-in-TMEM subview bug as a clean-negative
  boundary instead of allowing wrong-code.
  - verifier quarantine in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` now rejects
    LHS tensor-memory views for both:
    - `ttng.tc_gen5_mma`;
    - `ttng.tc_gen5_mma_scaled`.
  - added frontend/runtime clean-negative coverage in
    `python/test/gluon/test_tmem_runtime_matrix.py`;
  - added verifier regressions in `test/TritonNvidiaGPU/invalid.mlir`,
    including the scaled path.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `3 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_indexed_acc_view or mma_scaled_indexed_acc_view or mma_subslice_acc_view or mma_indexed_subslice_acc_view or mma_lhs_subslice_view_reports_clean_error or mma_scaled_minimal or mma_scaled_subslice_acc_view or mma_scaled_indexed_subslice_acc_view'`
    - `15 passed`
  - full matrix:
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py`
    - `1618 passed, 117 skipped in 331.68s`
- Current MMAv5 boundary after this follow-up:
  - accumulator `D` now has validated public-view coverage for:
    - index;
    - `N` subslice;
    - `index -> N-subslice` composition;
    - plain and scaled MMAv5;
    - legacy and TMEM-linear lifted parents.
  - LHS/A TMEM views remain explicitly unsupported pending a real lowering fix;
    they now fail cleanly instead of miscompiling.

2026-03-27 21:40Z - MMAv5 accumulator view-chain expansion and LHS TMEM-view quarantine

- Followed the next safe MMAv5 public-view increment after accumulator
  `N`-subslice support.
  - added positive runtime coverage for composed accumulator view chains
    `index(...).slice(..., dim=1)` on top of canonical MMAv5 parents for both
    `tcgen05_mma` and `tcgen05_mma_scaled`;
  - legacy parent coverage uses the existing `TensorMemoryLayout`;
  - TMEM-linear parent coverage uses lifted `tensor_memory_linear` layouts for
    rank-3 parents so the runtime path exercises `memdesc_index` plus
    `memdesc_subslice` composition before MMAv5 lowering.
- Added verifier/roundtrip coverage for the plain composed chain.
  - `test/TritonNvidiaGPU/ops.mlir` now has a
    `ttg.memdesc_index -> ttg.memdesc_subslice -> ttng.tc_gen5_mma`
    regression over a TMEM-linear parent accumulator.
- Confirmed a separate open BUG on A-in-TMEM public views instead of widening
  that path.
  - attempted positive LHS TMEM `subslice` support miscompiled at runtime for
    both legacy and TMEM-linear parents;
  - kept the compiler safe by rejecting MMAv5 LHS tensor-memory views whenever
    the memdesc logical shape differs from the allocation shape;
  - added direct verifier regressions for both plain and scaled MMAv5 LHS view
    rejection in `test/TritonNvidiaGPU/invalid.mlir`;
  - converted the Gluon runtime-matrix LHS subslice attempt into a clean
    negative regression that asserts the explicit diagnostic and no crash /
    `PassManager::run failed`.
- Source inspection after this slice still points to a larger frontier for
  arbitrary MMAv5 accumulator layouts.
  - the remaining row/col-mixed or low-bit-permuted negatives are not
    supportable by a bounded matcher tweak alone;
  - current MMAv5 lowering still normalizes through legacy-compatible families
    and emits accumulator addresses as `base + (row << 16 | col)` style tile
    offsets, with no instruction-level metadata for arbitrary accumulator
    swizzles;
  - so tile-preserving / canonical-family view composition is now stronger, but
    arbitrary PTX-level MMAv5 accumulator layouts still require a more general
    accumulator address model.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `3 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_indexed_acc_view or mma_subslice_acc_view or mma_indexed_subslice_acc_view or mma_lhs_subslice_view_reports_clean_error or mma_scaled_minimal or mma_scaled_subslice_acc_view or mma_scaled_indexed_subslice_acc_view'`
    - `13 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py`
    - `1616 passed, 117 skipped in 527.21s`
- Current boundary after this slice:
  - MMAv5 accumulator `D` now supports canonical public view chains that stay
    allocation-compatible along `M` and narrow only `N`, including
    `index(...).slice(...)` from lifted TMEM-linear parents;
  - MMAv5 LHS tensor-memory views remain an open BUG and are intentionally
    rejected cleanly for now;
  - arbitrary PTX-level MMAv5 accumulator layouts remain outside the current
    lowering model.

2026-03-27 22:20Z - MMAv5 canonical large-N accumulator normalization

- Landed a bounded MMAv5 accumulator-family fix for larger canonical
  accumulators along `N`.
  - refactored the legacy matcher in
    `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` through
    `matchTensorMemoryLegacyEncodingImpl(...)`, which can now search under an
    optional `maxBlockN` cap;
  - added accumulator-only `matchMMAv5AccumulatorFamily(...)` that selects the
    largest legacy family preserving the same physical TMEM packing while
    respecting the MMAv5 per-instruction `N <= 256` limit;
  - oversized explicit legacy layouts (for example
    `#ttng.tensor_memory_encoding<blockM = 128, blockN = 512, ...>`) now
    normalize through their canonical TMEM-linear form instead of hard-failing
    before the capped search runs.
- This unlocks canonical plain-MMAv5 accumulators that span multiple
  instructions along `N`.
  - direct GPU probes confirmed both explicit legacy and TMEM-linear
    `128x512xf32` accumulators now compile and run correctly for plain
    `tcgen05_mma`, each emitting four
    `tcgen05.mma.cta_group::1.kind::f16` instructions with correct numerics.
- Added permanent runtime coverage in
  `python/test/gluon/test_tmem_runtime_matrix.py`.
  - new `test_tmem_runtime_matrix_mma_large_n` covers legacy and TMEM-linear
    `128x512` accumulators and checks:
    - correct `A @ B + C` numerics;
    - exact PTX/LLIR opcode agreement;
    - four emitted `tcgen05.mma.cta_group::1.kind::f16` instructions.
- Important boundary after this fix:
  - the large-`N` widening is currently validated for plain MMAv5 only;
  - scaled `128x512` is still blocked separately by TMEM resource pressure
    (`Required: 532, Hardware limit: 512`) and was not widened by this
    matcher-only change;
  - non-canonical / tile-permuted accumulator layouts still need a more
    general address model than the current legacy-family matching.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `3 passed`
 - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_large_n or mma_indexed_acc_view or mma_subslice_acc_view or mma_indexed_subslice_acc_view or mma_lhs_subslice_view_reports_clean_error or mma_scaled_minimal or mma_scaled_subslice_acc_view or mma_scaled_indexed_subslice_acc_view'`
    - `15 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py`
    - `1620 passed, 117 skipped in 544.82s`

2026-03-28 01:35Z - MMAv5 one-CTA tile-permuted TMEM-linear accumulators

- Moved MMAv5 accumulator lowering past the old “exact legacy-equivalent only”
  boundary for a bounded new family: one-CTA TMEM-linear accumulators whose
  intra-instruction MMAv5 tile packing is canonical, but whose whole MMAv5
  tiles are permuted in TMEM.
  - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
    now gives `matchMMAv5AccumulatorFamily(...)` a second path for
    TMEM-linear accumulators that:
    - first strips exact legacy-equivalent matches under the existing
      `N <= 256` cap;
    - then accepts only tile-preserving one-CTA layouts against actual MMAv5
      instruction-width candidates `{64, 128, 256}` along `N`;
    - keeps low-bit/intra-tile packing exact while allowing only whole-tile
      movements from higher bits.
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
    now uses `DotOpMmaV5TmemLoader` for scaled-MMAv5 accumulator addressing
    too, instead of the old row-major block-id formula, so plain
    `tcgen05_mma` and `tcgen05.mma_scaled` share the same full-layout TMEM
    accumulator addressing model.
- Added new permanent coverage for a user-visible non-legacy TMEM-linear
  accumulator family.
  - `python/test/gluon/test_tmem_runtime_matrix.py` now covers:
    - plain `tcgen05_mma` with a `128x256` TMEM-linear accumulator whose
      64-wide MMAv5 tile columns are permuted (`mma_tile_permuted_layout`);
    - scaled `tcgen05.mma_scaled` with the same accumulator family
      (`mma_scaled_tile_permuted_layout`);
    - both assert correct numerics, exact PTX/LLIR agreement, and
      `tensor_memory_linear` staying present in TTGIR.
  - `test/TritonNvidiaGPU/ops.mlir` now has a direct verifier/round-trip
    regression for a non-legacy tile-permuted MMAv5 accumulator.
- Important tightening found during validation:
  - the first matcher draft was too permissive and allowed scrambled/permuted
    accumulator negatives to compile, which later faulted with GPU illegal
    instructions during the full matrix;
  - tightening the non-legacy search to real MMAv5 instruction widths
    `{64, 128, 256}` restored the old clean-negative behavior for
    mixed/scrambled/row-col-permuted layouts while keeping the intended
    tile-permuted positives.
- Current MMAv5 boundary after this slice:
  - supported:
    - canonical legacy / TMEM-linear accumulators;
    - canonical large-`N` accumulators (`128x512`);
    - public accumulator view chains that preserve full `M` and narrow `N`;
    - one-CTA TMEM-linear accumulators that preserve MMAv5 intra-tile packing
      and only permute whole tiles along `N`;
    - scaled MMAv5 shares that same one-CTA accumulator-address model.
  - still unsupported / open:
    - low-bit row/col-permuted or mixed/scrambled accumulator layouts;
    - two-CTA non-legacy tile permutations;
    - LHS/A TMEM public views.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `3 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_tile_permuted_layout or mma_scaled_tile_permuted_layout or mma_large_n or mma_indexed_acc_view or mma_subslice_acc_view or mma_indexed_subslice_acc_view or mma_scaled_minimal or mma_scaled_subslice_acc_view or mma_scaled_indexed_subslice_acc_view or mma_lhs_subslice_view_reports_clean_error'`
    - `17 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_exotic_layout_reports_clean_unsupported or mma_rowcol_permuted_layout_reports_clean_unsupported or mma_tile_permuted_layout or mma_scaled_tile_permuted_layout'`
    - `19 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py`
    - `1622 passed, 117 skipped in 538.84s`

2026-03-28 02:05Z - MMAv5 tile-permuted large-N coverage expansion

- Extended the new one-CTA tile-permuted MMAv5 coverage beyond the initial
  `128x256` case.
  - direct GPU probes confirmed plain `tcgen05_mma` also handles:
    - `128x512` accumulators with 64-wide tile-grid permutations;
    - `128x512` accumulators with 128-wide tile-grid permutations;
  - scaled MMAv5 remains bounded by TMEM capacity there:
    - `128x512` tile-permuted scaled accumulators still hit
      `OutOfResources out of resource: tensor memory, Required: 532, Hardware limit: 512`.
- Added permanent runtime coverage in
  `python/test/gluon/test_tmem_runtime_matrix.py`.
  - `MMA_TILE_PERMUTED_CASES` now covers:
    - `128x256`, tile width `64`;
    - `128x512`, tile width `64`;
    - `128x512`, tile width `128`;
  - opcode expectations now pin the observed instruction counts:
    - `8`, `16`, and `8` plain MMAv5 instructions respectively.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `ninja: no work to do.`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_tile_permuted_layout or mma_scaled_tile_permuted_layout'`
    - `4 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py`
    - `1624 passed, 117 skipped in 93.24s`
2026-03-28 03:45Z - Plain MMAv5 LHS TMEM views narrowed to whole-K direct subslices

- I tried the obvious verifier broadening first: remove the blanket `TCGen5MMAOp` / `TCGen5MMAScaledOp` ban on TMEM LHS views and flip the runtime-matrix `mma_lhs_subslice` BUG bucket positive.
- That was wrong. The new `128x32` plain-LHS and `128x64` scaled-LHS kernels compiled, but runtime numerics were bad (`NaN`-heavy for plain, materially wrong for scaled). So the broad ban was over-conservative, but the broad acceptance path was real wrong-code.
- The key clue was existing positive coverage in `python/test/gluon/test_core.py::test_block_m_64_mma`: TMEM operand-A direct `memdesc_subslice` tiles were already working there for the manually tiled `blockM=64` path. So the real boundary is not “all LHS views unsupported”, it is “only certain aligned whole-tile LHS views are supported today”.
- I replaced the blanket plain-MMA verifier ban in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` with a narrower rule:
  - plain `tc_gen5_mma` accepts TMEM LHS views only when they are direct `ttg.memdesc_subslice` values,
  - preserve full `M`,
  - slice whole `blockN` tiles along `K`,
  - and keep the subslice aligned to the underlying parent TMEM family.
- The historical runtime-matrix `128x32` / partial-`K` LHS cases stay clean-negative with the new diagnostic (“whole-K MMAv5 tiles”), while the pre-existing positive `test_block_m_64_mma` path continues to pass.
- I left `tc_gen5_mma_scaled` conservative for now: TMEM LHS views remain cleanly unsupported there because I do not yet have a validated positive scaled-LHS view family.
- Added verifier coverage:
  - `test/TritonNvidiaGPU/invalid.mlir` now checks the new plain-MMA partial-`K` rejection text.
  - `test/TritonNvidiaGPU/ops.mlir` now includes a positive plain-MMA whole-tile TMEM-LHS subslice case (`64x128 -> 64x64`).

Validation:
- `TRITON_BUILD_WITH_CCACHE=true make -j96`
- `lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir`
- `pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_lhs_subslice_view_reports_clean_error'`
- `pytest -s --tb=short -n 1 python/test/gluon/test_core.py -k 'test_block_m_64_mma'`
- full TMEM runtime matrix: `1624 passed, 117 skipped in 552.30s`

Open after this slice:
- scaled MMAv5 TMEM LHS views still need real positive support, not blanket rejection;
- plain MMAv5 TMEM LHS views are still not generalized beyond direct whole-`K` aligned subslices;
- the bigger MMAv5 work is still the non-legacy / arbitrary-layout accumulator-family expansion (two-CTA non-legacy, broader tile-grid permutations, etc.).

2026-03-28 05:20Z - ld/st clean-negative assertion normalization for frontend `auto` variant

- Folded the known full-matrix failure where ld/st clean-negative assertions
  still hardcoded `"TMEM layout '32x32b' unsupported..."` while frontend now
  defaults to `"auto"` in those paths.
- Updated `python/test/gluon/test_tmem_runtime_matrix.py` assertions to accept
  either spelling without changing the underlying unsupported semantics:
  - `test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_identity_reports_clean_error`;
  - `test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_reports_clean_unsupported`;
  - `test_tmem_runtime_matrix_cp_no_scales_linear_unsupported_shape_reports_clean_error`.
- Validation:
  - attempted rebuild (required before tests):
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - currently fails out of scope in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
      (`ceil<unsigned>(...)` parse error); no ld/st-owned files in that stack.
  - targeted cluster:
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_descriptor_multidim_slice_identity_reports_clean_error or ldst_descriptor_multidim_slice_reports_clean_unsupported or cp_no_scales_linear_unsupported_shape_reports_clean_error'`
    - `3 passed`
  - full matrix rerun:
    - `CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python:. pytest -s --tb=short -n 2 python/test/gluon/test_tmem_runtime_matrix.py`
    - no remaining failures from stale `'32x32b'` assertions.

2026-03-28 07:10Z - MMAv5 plain 32-wide tile-permuted accumulators; scaled fence

- Extended the non-legacy tile-preserving MMAv5 accumulator matcher in
  `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` to consider `blockN = 32`
  for plain MMAv5 accumulator families.
  - direct GPU probes confirmed plain `tcgen05_mma` handles one-CTA TMEM-linear
    tile-preserving `32`-wide permutations for:
    - `128x128`;
    - `128x256`;
    - `128x512`;
  - observed opcode counts:
    - `8`, `16`, and `32` respectively.
- The same widening is not safe for scaled MMAv5 yet.
  - direct probes showed non-legacy `blockN = 32` accumulators produce
    materially wrong results for `tcgen05.mma_scaled`;
  - added a verifier fence in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`:
    scaled MMAv5 now rejects non-legacy 32-wide accumulator permutations
    cleanly instead of compiling wrong-code;
  - this fence is intentionally narrow:
    canonical legacy-equivalent 32-wide layouts are not blanket-banned.
- Coverage updates:
  - `python/test/gluon/test_tmem_runtime_matrix.py`
    - positive plain-MMAv5 tile-permuted-32 cases for
      `128x128`, `128x256`, `128x512`;
    - clean-negative scaled-MMAv5 tile-permuted-32 cases for
      `128x128`, `128x256`;
  - `test/TritonNvidiaGPU/ops.mlir`
    - positive round-trip for a plain `128x128` tile-permuted-32 accumulator;
  - `test/TritonNvidiaGPU/invalid.mlir`
    - scaled non-legacy `blockN = 32` verifier regression.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `BUILD_DIR=/root/code/triton/build/cmake.linux-aarch64-cpython-3.12; cd "$BUILD_DIR" && lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `3 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_tile_permuted_layout or mma_scaled_tile_permuted_layout or mma_scaled_tile_permuted_32_reports_clean_unsupported or mma_exotic_layout_reports_clean_unsupported or mma_rowcol_permuted_layout_reports_clean_unsupported'`
    - `26 passed`
  - `CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python:. pytest -s --tb=short -n 2 python/test/gluon/test_tmem_runtime_matrix.py`
    - `1639 passed, 117 skipped in 307.17s`
- Current headline boundary after this slice:
  - plain MMAv5 supports one-CTA non-legacy tile-preserving accumulator
    permutations for tile widths `{32, 64, 128}`;
  - scaled MMAv5 supports the `{64}` family but still rejects the non-legacy
    `32`-wide family cleanly;
  - low-bit mixed/scrambled accumulator layouts and two-CTA non-legacy
    permutations are still open.

## 2026-03-27: ld/st frontend selection now uses TMEM view-analysis layout

- Scope (ld/st-only, disjoint from MMAv5 lowering edits):
  - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
  - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
  - `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h`
- Change:
  - added `getTMemViewAnalysisLinearLayout(shape, encoding, error)` API;
  - switched TMEM ld/st register-layout selection helpers to derive non-scales
    layout candidates from TMEM view-analysis layout instead of canonical-only
    extraction.
- Why:
  - canonical-only selection can drop valid view-preserved TMEM encodings in
    descriptor chains before `computeTMemLdStEncodingInfo(...)` runs.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_descriptor_higher_rank_dim0_slice_reports_clean_error_lifted_layout or ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout or ldst_descriptor_multidim_slice_positive or ldst_descriptor_multidim_slice_reports_clean_unsupported'`
    - `26 passed`
- Follow-up finding:
  - lifted higher-rank clean-negative buckets remain unchanged because frontend
    `compute_tmem_reg_layout(...)` still lacks enough view metadata when
    `shape` rank is lower than preserved encoding rank; this is a separate gap.

## 2026-03-27: removed implicit MMAv5 accumulator/LHS rematerialization

- User direction:
  - no silent compiler-inserted MMA rematerialization; user-specified layouts
    must either lower directly or fail with a clean diagnostic.
- Compiler-side outcome:
  - removed the hidden MMAv5 accumulator rematerialization path;
  - removed the hidden TMEM-LHS-view materialization path;
  - MMAv5 TMEM LHS views now fail cleanly instead of being rewritten behind the
    user’s back;
  - direct MMAv5 matching/lowering for supported layouts remains in place.
- Test updates:
  - restored plain MMA exotic / row-col permuted runtime buckets to clean
    unsupported checks;
  - converted MMAv5 TMEM-LHS-subslice runtime coverage to clean unsupported
    checks;
  - removed lit expectations that relied on hidden TMEM-LHS materialization;
  - added verifier coverage in `test/TritonNvidiaGPU/invalid.mlir` for plain
    and scaled MMAv5 TMEM-LHS view rejection.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `BUILD_DIR=/root/code/triton/build/cmake.linux-aarch64-cpython-3.12; cd "$BUILD_DIR" && lit -v test/TritonNvidiaGPU/invalid.mlir test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `4 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_exotic_layout_reports_clean_unsupported or mma_rowcol_permuted_layout_reports_clean_unsupported or mma_lhs_whole_tile_subslice_reports_clean_unsupported or mma_lhs_subslice_reports_clean_unsupported or mma_scaled_lhs_subslice_reports_clean_unsupported or mma_scaled_blockn32'`
    - `24 passed, 1 xfailed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_large_n or mma_tile_permuted_layout or mma_indexed_acc_view or mma_subslice_acc_view or mma_indexed_subslice_acc_view or mma_lhs_tmem_layout or mma_scaled_indexed_acc_view or mma_scaled_large_n or mma_scaled_tile_permuted_layout or mma_scaled_subslice_acc_view or mma_scaled_indexed_subslice_acc_view or mma_scaled_lhs_tmem_layout'`
    - `27 passed`
- New explicit BUG:
  - scaled MMAv5 direct linear `128x32` accumulator layout still compiles but
    produces incorrect numerics;
  - kept as explicit `xfail(strict=True)` runtime coverage in
    `test_tmem_runtime_matrix_mma_scaled_blockn32_linear_bug`.

## 2026-03-27: MMAv5 verifier/lowering moved off returned legacy TMEM attrs

- Goal of this slice:
  - start removing the “recover a legacy TMEM family, then inspect
    `blockM/blockN/colStride`” bottleneck from MMAv5 support paths while
    keeping current direct-support behavior unchanged.
- Compiler-side changes:
  - introduced `MMAv5TMemLayoutPlan` in
    `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h`;
  - replaced `matchMMAv5LhsEncoding(...)` /
    `matchMMAv5AccumulatorEncoding(...)` with
    `getMMAv5LhsLayoutPlan(...)` /
    `getMMAv5AccumulatorLayoutPlan(...)`;
  - updated MMAv5 verifier logic in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
    to consume the plan instead of a returned `TensorMemoryEncodingAttr`;
  - updated MMAv5 LLVM lowering in
    `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
    to derive instruction tile sizes from the plan instead of the legacy attr;
  - updated TMEM row-coupling allocation logic in
    `lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp`
    to use plan-derived instruction shape rather than legacy attr access;
  - updated `PromoteLHSToTMem.cpp` to derive accumulator blockM/twoCTAs from
    the MMAv5 plan.
- Important scope note:
  - this does not yet remove the internal finite-family search inside the MMAv5
    planner; it removes the returned legacy-attr dependency from verifier,
    lowering, and allocation call sites.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - direct `triton-opt` checks:
    - `test/TritonNvidiaGPU/invalid.mlir --split-input-file --verify-diagnostics`
      -> `OK`
    - `test/TritonNvidiaGPU/ops.mlir` -> `OK`
    - `test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir -split-input-file -tritongpu-promote-lhs-to-tmem`
      -> `OK`
    - `test/TritonGPU/promote-lhs-to-tmem.mlir -tritongpu-promote-lhs-to-tmem`
      -> `OK`
    - `test/Conversion/tritongpu_to_llvm_blackwell.mlir -split-input-file --convert-triton-gpu-to-llvm=compute-capability=100 -cse`
      -> `OK`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_exotic_layout_reports_clean_unsupported or mma_rowcol_permuted_layout_reports_clean_unsupported or mma_lhs_whole_tile_subslice_reports_clean_unsupported or mma_lhs_subslice_reports_clean_unsupported or mma_scaled_lhs_subslice_reports_clean_unsupported or mma_scaled_blockn32'`
    - `25 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_large_n or mma_tile_permuted_layout or mma_indexed_acc_view or mma_subslice_acc_view or mma_indexed_subslice_acc_view or mma_lhs_tmem_layout or mma_scaled_indexed_acc_view or mma_scaled_large_n or mma_scaled_tile_permuted_layout or mma_scaled_subslice_acc_view or mma_scaled_indexed_subslice_acc_view or mma_scaled_lhs_tmem_layout'`
    - `27 passed`
- Follow-up direction:
  - next MMAv5 step is to replace the planner’s remaining finite-family
    recovery logic with direct analysis over the normalized TMEM linear layout,
    then widen direct support beyond the current MMAv5-compatible family set.

## 2026-03-27: Gluon MMAv5 surface is async-only

- Changed the Gluon builders in `python/src/gluon_ir.cc` so
  `tcgen05_mma(...)` and `tcgen05_mma_scaled(...)` always set
  `is_async=true`.
- Updated the Gluon API docs in
  `python/triton/experimental/gluon/language/nvidia/blackwell/__init__.py`
  to remove the old “sync if no barrier” wording. The contract is now:
  MMAv5 ops are always async from Gluon; users must use
  `tcgen05_commit(...)` + `mbarrier.wait(...)` before consuming the
  accumulator unless they attach completion barriers directly.
- Updated Gluon parser expectations in
  `python/test/gluon/test_frontend.py` so no-barrier MMAv5 ops print
  `{is_async}`.
- Updated `tmem_mma_scaled_blockn32_kernel` in
  `python/test/gluon/test_tmem_runtime_matrix.py` to use explicit
  `tcgen05_commit(...)` + `mbarrier.wait(...)` before `acc_tmem.load()`.
- Important rationale:
  - the earlier `blockN=32` scaled wrong-data bucket was not a scale
    descriptor bug; the same emitted MMAv5 instruction sequence becomes
    numerically correct once the async completion contract is honored.
  - this avoids relying on the currently-missing sync-lowering path for
    Gluon-authored MMAv5 ops.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_frontend.py -k 'test_tcgen05_mma or test_tcgen05_mma_scaled'`
    -> `3 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_mma_scaled_blockn32_layout'`
    -> `2 passed`

## 2026-03-27: scaled MMAv5 32-wide tile-permuted bucket is a real BUG; higher-rank ld/st tests cleaned up

- Cleaned up the stale higher-rank ld/st runtime-matrix conversion from the
  delegated test pass.
  - one-CTA higher-rank `dim0_slice` / `half_rows` cases stay positive;
  - two-CTA higher-rank `dim0_slice` / `half_rows` cases are still clean
    negatives at the `get_reg_layout(...)` boundary, so the runtime matrix now
    reflects the current compiler instead of carrying a broken mixed state.
  - `python/test/gluon/test_tmem_runtime_matrix.py` imports cleanly again.
- Investigated scaled MMAv5 tile-permuted `tile_n=32` accumulators.
  - `128x128` and `128x256` tile-permuted-32 accumulators compile and emit the
    expected scaled MMAv5 PTX family/op counts (`8` and `16` ops
    respectively), but runtime numerics are wrong.
  - direct positive conversion of those cases is therefore incorrect; they are
    not stale negatives, they are a real wrong-code / fault bucket.
  - kept the working `256x64` scaled tile-permuted case positive.
  - added isolated `BUG` xfails for the `128x32` / `256x32` scaled
    tile-permuted cases using `@pytest.mark.forked` so the runtime matrix keeps
    the explicit BUG signal without poisoning the CUDA context for following
    tests.
- Tried a bounded lowering tweak in
  `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
  (loosening the `numColPerScaleBlockB >= 2` floor for `mmaSizeN == 32`).
  - result: wrong change; it turned the existing wrong-code into a device
    misaligned-address fault and also regressed the previously-good
    `256x64` scaled tile-permuted case.
  - reverted the tweak immediately; current tree is back to:
    - `256x64` scaled tile-permuted = positive;
    - `128x32` / `256x32` scaled tile-permuted = explicit `BUG` xfail.
- Repaired the `test/TritonNvidiaGPU/ops.mlir` regression James added for the
  higher-rank TMEM `reshape -> subslice -> index` chain.
  - the initial MLIR types were malformed because the intermediate TMEM view
    encodings / alloc-shapes change across the rank-3 reshape and slice.
  - `ops.mlir` now uses the exact inferred intermediate TMEM-linear types:
    - `#tmem_linear_rank3`
    - `#tmem_linear_rank3_small`
    - `#tmem_linear_half_rows`
  - also added a direct verifier/round-trip regression for
    `ttng.tc_gen5_mma_scaled` with a tile-permuted-32 accumulator.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `python -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_dim0_slice_reports_clean_error_lifted_layout or test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout'`
    -> `24 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_mma_scaled_tile_permuted_layout or test_tmem_runtime_matrix_mma_scaled_tile_permuted_32_bug or test_tmem_runtime_matrix_mma_scaled_indexed_acc_view'`
    -> `3 passed, 2 xfailed`
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd \"$BUILD_DIR\" && lit -v test/TritonNvidiaGPU/ops.mlir`
    -> `1 passed`

## 2026-03-27: canonical fallback added for view-derived layouts
- `compute_tmem_reg_layout` now analyses the user-supplied TMEM layout via `ttng::getTMemViewAnalysisLinearLayout` and `ttng::tryMakeTMemViewEncoding` before re-running `ttng::getTmemCompatibleLayouts`, so future higher-rank view slices can recycle the canonical TMEM encoding instead of being rejected outright.
- Higher-rank descriptor slice and half-rows kernels still hit the current `get_reg_layout` guard (`TMEM layout '<variant>' unsupported`), so the runtime-matrix clean-error cases stay in place while the new canonical fallback waits for the remaining layout-column-contiguity blockers to disappear.

## 2026-03-28: ld/st direct-lowering fallback + ld.red gap audit

- In `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`, updated
  `computeTMemLdStEncodingInfo(...)` to fall back to
  `getTMemViewAnalysisLinearLayout(shape, encoding, ...)` when
  `getCanonicalTMemLinearEncoding(...)` fails.
  - Prior behavior: immediate failure for non-standalone TMEM descriptor views.
  - New behavior: if the view-analysis projection is available, continue with
    direct `tcgen05.ld/st` lowering; reject only when both canonical and
    analysis layouts are unavailable.
  - This stays within the direct-support-only contract: no silent relayout and
    no hidden rematerialization.

- Validation run after rebuild:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_identity_reports_clean_error python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_reports_clean_unsupported python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_mixed_layout_reports_clean_unsupported`
    -> `14 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_ld_red or test_tmem_runtime_matrix_ldst_x1_f32_roundtrip or test_tmem_runtime_matrix_ldst_x1_f32_descriptor_chain_roundtrip or test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_positive'`
    -> `35 passed`
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd \"$BUILD_DIR\" && ninja triton-opt && lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir`
    -> `2 passed`

- `ld.red` status from this audit:
  - Supported: `f32` min/max row-wise reduction over `N`, with optional
    `.abs` and `.NaN` modifiers, for layouts where `N` is register-local for
    each row.
  - Clean-unsupported (intentional): mixed layouts where `N` is sharded across
    threads; tensor-memory scales reductions; unpacked reduction formats.
  - Remaining expansion path: support additional ISA-realizable cases where
    row-wise reduction can be proven despite nontrivial register distributions
    (requires explicit cross-thread combine semantics, not layout repair).

## 2026-03-28: ld/st + ld.red full owned-scope audit rerun (no new promotable gaps)

- Re-ran the broad owned-scope runtime surface after the
  `getTMemViewAnalysisLinearLayout(...)` fallback landed in
  `computeTMemLdStEncodingInfo(...)`.
- Result:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst or ld_red'`
    -> `1143 passed, 112 skipped in 403.56s`
- Re-ran the targeted clean-negative boundary slice:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_descriptor_higher_rank_dim0_slice_reports_clean_error_lifted_layout or ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout or ldst_descriptor_multidim_slice_identity_reports_clean_error or ldst_descriptor_multidim_slice_reports_clean_unsupported or ld_red_mixed_layout_reports_clean_unsupported'`
    -> `38 passed`
- Lit sanity in owned files:
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && ninja triton-opt && lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir`
    -> `2 passed`
- Conclusion for ld/st + ld.red in current owned scope:
  - no additional direct-lowering gaps were found that are safely promotable
    without changing semantics;
  - remaining clean negatives are still layout/semantic boundaries:
    - high-rank lifted descriptor views that cannot materialize a supported
      direct TMEM register layout for the requested explicit variant;
    - `ld.red` layouts where N is sharded across threads (would require
      explicit cross-thread combine semantics, not direct single-op lowering).

## 2026-03-28: MMAv5 transform linearization sweep + planner cleanup

- Removed one remaining internal legacy-TMEM construction site in
  `lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp`.
  - The pass now builds canonical TMEM-linear layouts directly via
    `getCanonicalTMemLinearEncoding(...)`, matching the earlier
    `AccelerateMatmul.cpp` sweep.
  - This keeps legacy TMEM encodings as user/IR sugar only and avoids
    reintroducing finite-family assumptions inside transforms.

- Updated `test/TritonGPU/accelerate-matmul.mlir` to stop depending on exact
  `#ttng.tensor_memory_encoding<...>` print forms where the canonical output is
  now `#ttng.tensor_memory_linear<...>`.
  - Also removed one stale warning expectation that no longer fires on the
    current transform path.

- Cleaned up the MMAv5 planner in
  `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`:
  - removed the unconditional "legacy attr in, legacy plan out" fast paths from
    the main `planMMAv5Family(...)`, `planMMAv5ExactFamily(...)`, and scaled
    accumulator-family flow;
  - kept a narrow explicit-legacy preservation path only for accumulator /
    scaled-accumulator cases where the user explicitly spelled a non-default
    `colStride` that canonical TMEM-linear form does not yet uniquely preserve;
  - kept LHS planning fully linear-first again.
  - practical effect: direct MMAv5 planning now canonicalizes and reasons from
    TMEM-linear layouts in more cases, while still preserving exact legacy
    `colStride=2/4` accumulator sugar where dropping it would regress valid
    direct codegen.

- Focused validation after this sweep:
  - transform lit sanity:
    - `lit -v test/TritonGPU/accelerate-matmul.mlir`
    - `lit -v test/TritonGPU/promote-lhs-to-tmem.mlir`
    - `lit -v test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir`
    -> all passed
  - MMAv5 frontend/runtime slices:
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_frontend.py -k 'test_tcgen05_mma or test_tcgen05_mma_mbar or test_tcgen05_mma_scaled or test_tensor_memory_linear_mma_compile_reports_unsupported_layout'`
      -> `6 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_mma_exotic_layout_reports_clean_unsupported or test_tmem_runtime_matrix_mma_rowcol_permuted_layout_reports_clean_unsupported or test_tmem_runtime_matrix_mma_scaled_acc_blockn32_direct_layout or test_tmem_runtime_matrix_mma_scaled_acc_blockn64_direct_layout or test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_reports_clean_unsupported'`
      -> `20 passed`
    - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && lit -v test/TritonNvidiaGPU/ops.mlir`
      -> `1 passed`

- Build note for this dirty tree:
  - a broad `TRITON_BUILD_WITH_CCACHE=true make -j96` succeeded earlier in the
    day, but a later full rebuild hit unrelated link failures in broad
    non-owned targets and one transient `bin/triton-opt` permission-bit glitch;
  - targeted relinks of `triton-opt` and `libtriton.so` with
    `ninja -C build/cmake.linux-aarch64-cpython-3.12 triton-opt triton`
    produced valid ELF artifacts and unblocked the focused TMEM/MMAv5 checks;
  - treat this as a workspace/build-system issue until reproduced in a clean
    tree; it is not currently tied to the MMAv5 planner changes above.

## 2026-03-28: one-CTA lifted higher-rank ld/st bucket split into clean negatives plus BUG xfails

- Re-audited the one-CTA lifted higher-rank `ld/st` runtime buckets after the
  recent view-analysis fallback work.
- Result:
  - the old blanket clean-negative expectation was too coarse;
  - however, these are **not** all safe positives.
- Precise outcome:
  - `dim0_slice` with explicit `32x32b` now compiles but miscomputes for
    `n = 64 / 128 / 256`.
    - converted to explicit strict `BUG` xfails in
      `python/test/gluon/test_tmem_runtime_matrix.py`.
  - `half_rows` now compiles but miscomputes for the full explicit variant set
    (`32x32b`, `16x64b`, `16x128b`, `16x256b`) at
    `n = 64 / 128 / 256`.
    - converted to explicit strict `BUG` xfails.
  - the remaining one-CTA lifted `dim0_slice` variants
    (`16x64b`, `16x128b`, `16x256b`) are still genuine clean negatives at the
    public `get_reg_layout(...)` boundary and stay as clean-error tests.
- Focused validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_ldst_descriptor_higher_rank_dim0_slice_reports_clean_error_lifted_layout or test_tmem_runtime_matrix_bug_ldst_descriptor_higher_rank_dim0_slice_lifted_layout or test_tmem_runtime_matrix_bug_ldst_descriptor_higher_rank_half_rows_lifted_layout'`
    -> `9 passed, 15 xfailed`

## 2026-03-28: copy-planner experiments and current `warpx2` / dense-copy status

- Bounded experiment: tried canonicalizing the internal copy descriptor layout
  by reordering TMEM row/col basis bits before descriptor synthesis.
  - reverted immediately.
  - reason: it widened support by changing semantics, producing one wrong-code
    no-scales copy case and one late lowering assertion. This violated the
    direct-support-only contract.
- Bounded follow-up: added a dense-family planner path that can try both the
  canonical `32x4` descriptor factorization and a `64x2` fallback before
  rejecting.
  - current public clean-negative copy buckets are unchanged after this:
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_cp_no_scales_linear_exotic_reports_clean_unsupported or test_tmem_runtime_matrix_cp_no_scales_linear_rowcol_permuted_reports_clean_unsupported or test_tmem_runtime_matrix_cp_no_scales_warpx2_candidate_reports_clean_error'`
      -> `19 passed`
    - `lit -v test/TritonNvidiaGPU/invalid.mlir`
      -> `1 passed`
  - so this is currently a safe planner broadening with no user-visible
    promotion yet.
- Side-audit conclusions:
  - the historical no-scales `warpx2` candidate still classifies as
    `tcgen05.copy.128x128b`, not `warpx2`;
  - the historical scales `warpx2` probe still classifies as `warpx4`, not
    `warpx2`;
  - real `warpx2` support still needs:
    - a user-visible representation path (likely scoped non-surjective TMEM for
      copy destinations or an equivalent copy-only representation),
    - family-specific `warpx2` message / descriptor / address planning,
    - lowering that consumes that plan directly instead of relying only on the
      generic MMA shared-descriptor builder.

## 2026-03-28: remaining MMAv5 legacy bottlenecks after planner cleanup

- The MMAv5 planner is more linear-first now, but the side audit identified
  remaining legacy-family bottlenecks:
  - `matchTensorMemoryLegacyEncoding(...)` still returns
    `TensorMemoryEncodingAttr` by searching finite legacy families;
  - `planMMAv5LegacyExactFamily(...)` is still used as an explicit fast path
    for accumulator / scaled-accumulator exact-family recovery;
  - verifier and lowering remain bounded by `getMMAv5*LayoutPlan(...)`, so any
    legal linear layout that does not collapse to one of the currently-plannable
    families is still rejected;
  - `impl::verifyMMAv5Op(...)` still uses a legacy-only interleaved
    `blockM=64` check keyed on `TensorMemoryEncodingAttr`;
  - allocator row-coupling still depends on MMAv5 family plans rather than a
    fully general linear TMEM physical-footprint query.

## 2026-03-28: MMAv5 planner cleanup follow-up

- Removed the explicit `planMMAv5LegacyExactFamily(...)` fast path from the
  MMAv5 accumulator / scaled-accumulator planners.
  - plain and scaled MMAv5 accumulator planning now share the same
    memdesc-based fallback: try the direct view shape first, then canonicalize
    the memdesc through `tryGetCanonicalTensorMemoryEncoding(memDescType, ...)`
    and retry on the backing allocation shape only for full-`M`, narrowed-`N`
    accumulator views.
  - this keeps legacy TMEM encodings as user sugar while routing internal MMAv5
    planning through canonical TMEM-linear analysis.
- Added direct scaled-accumulator subslice coverage from user code.
  - new Gluon runtime coverage uses a parent `128x128`
    `#ttng.tensor_memory_linear` accumulator and writes through a
    `ttg.memdesc_subslice` view for `N=32` and `N=64`.
  - new TTGIR regression added in `test/TritonNvidiaGPU/ops.mlir`.
- Fixed the remaining legacy-only interleaved `blockM=64` verifier check.
  - `impl::verifyMMAv5Op(...)` now keys the rejection off MMAv5 layout plans
    plus physical TMEM row footprint (`getTmemAllocSizes`), instead of only
    `TensorMemoryEncodingAttr`.
  - added an `invalid.mlir` regression that uses the explicit linear equivalent
    of the interleaved `blockM=64` family (`128x64` TMEM-linear), and it now
    rejects with the same diagnostic as the legacy sugar.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `lit -v test/TritonNvidiaGPU/ops.mlir` -> `1 passed`
  - `lit -v test/TritonNvidiaGPU/invalid.mlir` -> `1 passed`
  - focused scaled MMAv5 slice
    (`test_tmem_runtime_matrix_mma_scaled_minimal`,
    `..._blockn64_direct_layout`,
    `..._blockn32_direct_layout`,
    `..._acc_subslice_view`,
    `..._acc_tile_permuted_32_reports_clean_unsupported`)
    -> `6 passed`
  - broader MMA slice
    (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_mma'`)
    -> `52 passed`

## 2026-03-28: ld/st audit follow-up after raw-query repair

- Audited the remaining direct-lowering gaps on the ld/st track after the
  raw-query memdesc view path was repaired and the broader
  `ldst_descriptor_multidim_slice or ldst_descriptor_higher_rank` cluster went
  green.
- Concrete next unsupported-but-supportable buckets:
  - two-CTA higher-rank `half_rows` descriptor views:
    - current raw query layout for the failing `block_two_ctas` case is
      `row=[[2,0],[4,0],[8,0],[16,0],[32,0],[64,0]]`,
      `col=[[0,1],[0,2],[0,4],[0,8],[0,16],[0,32]]`,
      `block=[[1,0]]`, shape `[128,64]`;
    - the failure is no longer raw-view inference; it is ld/st planning on a
      view where the low logical row bit is carried in `block`;
    - likely fix area: linear-first row-footprint / row-anchor reasoning in
      `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
      (`getDistributedLayoutForTmemLdSt(...)`) together with the matching row
      anchor checks in
      `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
      (`computeTMemLdStEncodingInfoImpl(...)`).
  - narrow scales explicit `16x32bx2` (`M=32, N=16`) on
    `TensorMemoryScalesLayout`:
    - the raw-query reg-layout path can already synthesize a candidate
      `16x32bx2` register layout;
    - the remaining failure is in direct lowering / matcher selection, where
      `computeTMemLdStEncodingInfo(..., queryLayout.layout, ...)` falls through
      to `Failed to lower TMEM load/store: unsupported dst layout`;
    - likely fix area:
      `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
      (`lowerTMemLdSt(...)` / `I16x32bx2` quotient path).
- Larger future bucket still open:
  - `scrambled_cols` multidimensional descriptor slices still fail in the raw
    query-layout inference itself (`unsupported tensor memory memdesc_subslice
    view`), so they need a more general internal query representation or a
    multi-message decomposition rather than a small local ld/st planner tweak.
- Validation snapshot for the audit:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - targeted clean-negative slice
    (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout or test_tmem_runtime_matrix_ldst_scales_variant_reports_clean_unsupported or test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_reports_clean_unsupported'`)
    -> `23 passed`

## 2026-03-28: scales `16x32bx2` selector cleanup and frontend alignment

- Fixed a no-crash regression in the TMEM ld/st selector:
  `getDistributedLayoutForTmemLdSt(const LinearLayout&, ...)` was reading the
  row-16 basis unconditionally even when the TMEM row dimension was only 16
  high. That surfaced as a hard `LinearLayout::getBasis` assertion from the
  pure descriptor-type `get_reg_layout(instr_variant="16x32bx2")` path on
  tensor-memory scales shapes like `[16, 4]`.
- Kept the zero-base retry / row-plan consistency cleanup in
  `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`:
  - selector synthesis now retries after stripping zero row/col/block bases;
  - scales memtype selection now uses the same `32/64` row-anchor plan as
    direct ld/st validation instead of mixing that with the broadcast `0/0`
    row plan recovered from the raw scales linear layout.
- Aligned the pure type-based frontend helper in `python/src/gluon_ir.cc` with
  the handle-aware memdesc path:
  - explicit TMEM atoms now first filter the full compatible-layout set via
    `getTmemCompatibleLayouts(...)` instead of relying only on the narrower
    direct memtype synthesis path.
- User-visible result:
  - small explicit tensor-memory-scales `16x32bx2` descriptor-type queries are
    now exposed directly instead of failing/crashing:
    - `[16, 4]`, `num_warps=4` -> positive
    - `[16, 8]`, `num_warps=8` -> positive
  - larger shapes that still do not have a direct lowering remain clean
    negatives:
    - `[32, 16]`, `num_warps=4`
    - `[32, 4]`, `num_warps=8`
- Added frontend coverage in `python/test/gluon/test_frontend.py` for both:
  - the new positive small explicit `16x32bx2` descriptor-type path;
  - the larger clean-negative path, explicitly guarding against regressions
    back to crashes.
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_frontend.py -k 'test_tensor_memory_scales_explicit_16x32bx2_descriptor_type_small_tile or test_tensor_memory_scales_explicit_16x32bx2_descriptor_type_reports_clean_error'`
    -> `2 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_ldst_scales_variant_sweep or test_tmem_runtime_matrix_ldst_scales_variant_reports_clean_unsupported'`
    -> `34 passed`

## 2026-03-28: scaled MMAv5 tile-preserving planner widening

- Widened `planMMAv5ScaledAccumulatorFamily(...)` in
  `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` from exact-family matching to
  the same tile-preserving family search already used by plain MMAv5.
  - This is a real linear-first cleanup: scaled accumulator planning no longer
    rejects tile-preserving `#ttng.tensor_memory_linear` layouts solely because
    they are not byte-for-byte equal to a canonical legacy family.
- User-visible result:
  - direct scaled MMAv5 now supports a tile-permuted accumulator layout with
    shape `128x256` and `tile_n=64`;
  - the repeated-`N=32` tile-permuted case remains a clean negative, but now
    for the real public-scales-layout limit rather than the old
    “layout not directly supported” verifier gate.
- Added runtime coverage in
  `python/test/gluon/test_tmem_runtime_matrix.py`:
  - new positive:
    `test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_64_direct_layout`
  - tightened negative:
    `test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_repeated_n32_reports_clean_unsupported`
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - targeted scaled slice
    (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_mma_scaled_acc_blockn32_direct_layout or test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_64_direct_layout or test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_repeated_n32_reports_clean_unsupported'`)
    -> `3 passed`
  - broader scaled MMAv5 slice
    (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_mma_scaled'`)
    -> `7 passed`
  - full TMEM runtime matrix
    (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py > /tmp/tmem_runtime_matrix_full.log 2>&1`)
    -> `1588 passed, 118 skipped`

## 2026-03-28: reachable `warpx2::01_23` copy support plus `02_13` boundary

- Extended `getTMemCopyDescriptorLayouts(...)` in
  `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp` with bounded
  candidate generation for user-reachable `warpx2::01_23` layouts:
  - fold source warp-group bits into row/col;
  - rotate leading row-repeat bits out of the core descriptor prefix;
  - promote one extra low-order column bit into the row prefix when needed.
- User-visible result:
  - the public non-surjective TMEM-linear `warpx2::01_23` destination path now
    compiles, emits `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b`, and runs
    with the expected aliased logical result for that layout;
  - `warpx2::02_13` is still a clean negative with a specific
    descriptor-plan diagnostic.
- Runtime-matrix coverage was updated accordingly:
  - `test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive`
    is now a real positive with exact output checks;
  - `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_candidate_reports_clean_error`
    keeps the remaining unsupported family explicit.
- Focused validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - targeted `warpx2` slice
    (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'warpx2_01_23_candidate_positive or warpx2_02_13_candidate_reports_clean_error or warpx2_01_23_canonical_codegen'`)
    -> `3 passed`

## 2026-03-28: MMAv5 LHS `memdesc_subslice` direct support

- Extended `getMMAv5LhsLayoutPlan(...)` in
  `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` to treat full-`M`, narrower-`K`
  TMEM descriptor views the same way the accumulator path already treats
  full-`M`, narrower-`N` accumulator views.
  - Planning now retries on the backing allocation shape when the visible LHS
    view preserves `M` and only narrows `K`.
  - This is still direct lowering: the actual `ttg.memdesc_subslice` operand is
    preserved, and lowering uses the view value directly.
- New direct user-visible support:
  - plain `tcgen05_mma` with a TMEM LHS `memdesc_subslice`;
  - scaled `tcgen05_mma_scaled` with a TMEM LHS `memdesc_subslice`.
- Added runtime coverage in `python/test/gluon/test_tmem_runtime_matrix.py`:
  - `test_tmem_runtime_matrix_mma_lhs_subslice_view`
  - `test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view`
- Added positive IR coverage in `test/TritonNvidiaGPU/ops.mlir` and removed the
  stale rejection-only cases from `test/TritonNvidiaGPU/invalid.mlir`.
- Focused validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - plain LHS subslice runtime -> `1 passed`
  - scaled LHS subslice runtime -> `1 passed`
  - combined focused slice
    (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'warpx2_01_23_candidate_positive or warpx2_02_13_candidate_reports_clean_error or mma_lhs_subslice_view'`)
    -> `3 passed`
  - lit bundle
    (`lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir`)
    -> `2 passed`
  - full TMEM runtime matrix
    (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py > /tmp/tmem_runtime_matrix_full_20260328.log 2>&1`)
    -> `1592 passed, 118 skipped`

## 2026-03-28: ld/st legacy-hook cleanup plus broader ld.red coverage

- Removed the last legacy-only M64 ld/st layout-selection detour from
  `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`.
  - Deleted the `getLegacyEquivalentM64MemType(...)` recovery path and the now-dead
    `matchTensorMemoryLegacyEncoding(...)` helpers.
  - `getDistributedLayoutForTmemLdSt(...)` now stays fully on the linear-first
    selection path for the exercised M64 / split-N cases.
- Validation after the cleanup:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - focused M64 / split-N / f32 ld-st slice
    (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'splitn_immediates or splitn_auto_selects_16x32bx2 or explicit_16x32bx2_matches_splitn or ldst_x1_f32_roundtrip or ldst_x1_f32_descriptor_chain_roundtrip or ldst_x1_f32_unsupported_variants_report_clean_unsupported or mma_lhs_subslice_view or mma_scaled_lhs_subslice_view or mma_acc_tile_permuted'`)
    -> `40 passed`
- Probed the actual ld.red surface from user-written TMEM-linear layouts and
  found the current direct lowering is broader than the old runtime matrix
  covered.
  - Direct `tcgen05.ld.red` lowering already works for:
    - tile-permuted accumulators (`tile_n=32` / `64`) at `N=128/256`
    - all row/col-permuted `128x128` pure-linear layouts built from the current
      permutation sweep
  - Mixed TMEM layouts still fall back to the generic reduction path, which
    stays the correct negative boundary for now.
- Added new permanent runtime coverage in
  `python/test/gluon/test_tmem_runtime_matrix.py`:
  - `test_tmem_runtime_matrix_ld_red_tile_permuted_linear_layout`
  - `test_tmem_runtime_matrix_ld_red_rowcol_permuted_linear_layout`
  - kept `test_tmem_runtime_matrix_ld_red_mixed_linear_layout_falls_back`
    as the fallback boundary
- Validation for the new ld.red coverage:
  - `python -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
  - focused ld.red slice
    (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_identity_linear_layout or ld_red_tile_permuted_linear_layout or ld_red_rowcol_permuted_linear_layout or ld_red_mixed_linear_layout_falls_back'`)
    -> `100 passed`
  - broad ld/st + ld.red slice
    (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst or ld_red'`)
    -> `1215 passed, 113 skipped`

## 2026-03-28: bounded `warpx2::02_13` widening and handle-aware ld/st query follow-up

- Revisited the remaining user-visible `tcgen05.copy.warpx2::02_13.64x128b`
  gap in `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`.
  - Fixed the dead `multicast == 2` descriptor-layout widening branch in
    `getTMemCopyDescriptorLayouts(...)`.
  - Added a bounded single-row-basis move search on top of the existing
    row-rotation / row-column reassignment variants for `multicast == 2`.
  - Result: the historical non-surjective `warpx2::02_13` user path still
    cleanly rejects. The gap is no longer a trivial dead-branch bug; it needs
    deeper descriptor/message synthesis than the current bounded search.
- Extended the handle-aware TMEM reg-layout query path in
  `python/src/gluon_ir.cc`.
  - `compute_tmem_reg_layout_from_memdesc(...)` now augments the type-only
    candidate list with layouts synthesized using
    `getTMemLdStRowPlanForQuery(memDesc, queryTy)` and the new
    `getDistributedLayoutForTmemLdSt(memType, atom, numWarps, rowPlanOverride)`
    overload declared in `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h`.
  - This keeps descriptor-view reg-layout probing aligned with the backing
    memdesc value instead of only the erased query type.
- Probed two remaining ld/st clean-negative buckets with the new handle-aware
  path:
  - identity multidimensional `32x32` descriptor view:
    still clean-negative; the raw query and both canonical/surrogate query
    types fail to produce a legal direct ld/st register layout.
  - two-CTA higher-rank `half_rows` descriptor view:
    still clean-negative; raw-query and query-type probing now reach candidate
    blocked layouts, but `computeTMemLdStEncodingInfo(...)` still rejects them.
- Tried a 32-row TMEM row-plan extension to make the identity `32x32` slice
  reachable, but reverted it immediately.
  - It made the identity `32x32` slice compile, but the lowered code emitted a
    single `tcgen05.{ld,st}.sync.aligned.32x32b.x32.b32` pair at offset `0`
    and produced wrong data for the sliced view.
  - The revert keeps the safe clean-rejection boundary instead of introducing
    wrong-code.
- Focused validation after the safe revert:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir`
    -> `2 passed`
  - focused runtime slice
    (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'warpx2_01_23_candidate_positive or warpx2_02_13_candidate_reports_clean_error or test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_identity_reports_clean_error or test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout or test_tmem_runtime_matrix_mma_lhs_subslice_view or test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view'`)
    -> `17 passed`

## 2026-03-28: direct `warpx2::02_13` no-scales copy support plus scaled-MMAv5 repeated-`N=32` probe

- Finished the user-visible `tcgen05.copy.warpx2::02_13.64x128b` no-scales path.
  - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
    - relaxed `getDirectTMemCopySeedDescriptorImm(...)` so it matches the
      canonical `SharedLinear` offset pattern directly instead of requiring full
      linear-layout equality on optional empty dims
    - kept the existing direct seed descriptor immediate
      `0x0008400800000020` / `tmemDwordDelta=4` plan for
      `warpx2::02_13.64x128b`
    - extended `getTMemLdStRowPlanForType(...)` with a `baseOffset=4` plan for
      the reachable non-surjective `02_13` TMEM-linear layout
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
    - threaded the new TMEM ld/st `baseOffset` through lowering so the same
      non-surjective layout that `tcgen05.copy.warpx2::02_13` writes at
      `tmem_base + 4` also loads/stores from that anchor directly
  - `python/test/gluon/test_tmem_runtime_matrix.py`
    - converted
      `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_candidate_reports_clean_error`
      into the positive
      `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_candidate_positive`
    - added `_expected_tmem_copy_warpx2_02_13_output(...)`
- Validation for the new `02_13` path:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `CUDA_VISIBLE_DEVICES=0 TRITON_ALWAYS_COMPILE=1 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'warpx2'`
    -> `4 passed`
  - `PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_frontend.py -k 'non_surjective or warpx2_like_rows_parse'`
    -> `3 passed`
  - focused runtime slice after cache clear
    (`CUDA_VISIBLE_DEVICES=0 TRITON_ALWAYS_COMPILE=1 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'warpx2_01_23_candidate_positive or warpx2_02_13_candidate_positive or tile_permuted_32_repeated_n32'`)
    -> `3 passed`

- Bounded scaled-MMAv5 repeated-`N=32` probe:
  - temporarily lifted the clean rejection in
    `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` and
    `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
    for tile-permuted `blockN=32` accumulators
  - result:
    - with the original B-scale address model, the `128x128 tile_n=32` case
      compiled but hit a runtime `CUDA error: misaligned address`
    - instrumentation showed the second `N=32` instruction for matrix-B scales
      uses `origRaw=1` from `ttng::getTMemViewOffset(...)` at
      `(row,col)=(32,0)` under the public `TensorMemoryScalesLayout`
    - a bounded rewrite that folded the odd `N=32` half-tile into
      `(row,col)=(0,4)` removed the misaligned-address fault but still produced
      wrong numerics (`max_abs ~= 5e-2` versus the float32 reference), so it was
      reverted immediately
  - conclusion:
    - the current clean rejection is still required to satisfy the
      direct-support-or-clean-reject contract
    - supporting repeated scaled `N=32` needs a more complete matrix-B scale
      address model than the current `getTMemViewOffset(...)` + selector path

- Bounded higher-rank ld/st half-rows probe (`reshape((2, M/2, N)).slice(1,1,dim=0).index(0)`):
  - reproduced the only remaining full-matrix failures after the `warpx2`
    work:
    - `test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_positive_lifted_layout`
      for `layout_name="identity"` across `n in {64,128,256}` and all explicit
      variants
  - representative failure:
    - the subview is intended to update rows `64:128` of the backing
      `128xN` TMEM tile
    - runtime output still updates rows `0:64` instead
  - direct-lowering experiments:
    - tried preserving the local `64`-row row plan
    - tried forcing the larger `128`-row backing row plan
    - tried emitting backing warp anchors while keeping local register-layout
      selection
    - tried canceling the extra row-half TMEM base shift while keeping either
      local or backing anchors
  - outcome:
    - all bounded single-atom variants still hit the wrong half (or, when the
      planner was forced too hard toward the backing row span, turned into a
      clean compile-time rejection instead of a positive)
    - no simple `rowPlan` / `warpBaseOffset` / `baseOffset` tweak was enough
  - conclusion:
    - this bucket is not a small address-bookkeeping bug anymore
    - it likely needs a deeper direct decomposition for ld/st on row-half views
      carved from a larger TMEM backing tile, rather than another local anchor
      tweak
  - important cleanup:
    - all exploratory planner mutations were reverted after the probe, so the
      worktree is back on the last stable direct-lowering baseline for this
      bucket

- 2026-03-29: scales ld/st direct recovery, ld.red direct-or-reject cleanup,
  and clean-cache validation are now green
  - direct scales ld/st recovery landed in the linear-first path instead of a
    Python-only fallback:
    - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
      - restored scales-specific
        `getDistributedLayoutForTmemLdStLegacyAnchored(...)`
      - threaded scales handling through
        `getDistributedLayoutForTmemLdSt(...)` and
        `getTmemCompatibleLayouts(...)`
      - `getTmemLoadLayoutSplitLongM(...)` now returns `nullopt` for scales so
        the planner does not force invalid long-M reshapes
    - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
      - restored the scales-specific direct-validator branch in
        `computeTMemLdStEncodingInfoImpl(...)`
      - taught `lowerTMemLdSt(...)` to prefer split-N scales plans directly
    - `python/triton/experimental/gluon/language/_semantic.py`
      - widened the split-N fallback so explicit scales `16x32bx2` can reuse
        the `32x32b`-derived reg layout and then adjust the register bases
    - `python/triton/experimental/gluon/language/nvidia/blackwell/__init__.py`
      - added `_strip_zero_reg_bases_from_layout(...)`
      - applied it for scales descriptor `get_reg_layout(...)`
  - validation for the recovered scales direct path:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - focused scales ld/st slice:
      `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_scales_direct_roundtrip or ldst_scales_variant_sweep or ldst_scales_variant_reports_clean_unsupported'`
      -> `38 passed, 1 skipped`

- 2026-03-29: `ld.red` now follows the direct-support-or-clean-reject rule
  - removed the silent Gluon fallback from
    `python/triton/experimental/gluon/language/nvidia/blackwell/__init__.py`
    `_load_red(...)`
  - mixed / unsupported TMEM reduction layouts now raise a direct error instead
    of compiling `tmem_load + tt.reduce`
  - tests updated:
    - `python/test/gluon/test_core.py`
    - `python/test/gluon/test_tmem_runtime_matrix.py`
  - fresh-cache validation:
    - `rm -rf /root/.triton/cache/*`
    - `PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_core.py -k 'tmem_reduction_linear_mixed_layout_reports_clean_error'`
      -> `1 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_identity_linear_layout or ld_red_tile_permuted_linear_layout or ld_red_rowcol_permuted_linear_layout or ld_red_mixed_linear_layout_reports_clean_unsupported'`
      -> `100 passed`
  - important local-validation caveat:
    - Triton’s on-disk JIT cache will reuse an old binary even if the Python
      helper behavior changed without changing the kernel hash
    - clearing `/root/.triton/cache/*` was required to validate the fallback
      removal locally

- 2026-03-29: MMAv5 legacy sugar now re-enters the exact direct family path
  - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
    - `planMMAv5AccumulatorFamily(...)`
    - `planMMAv5LhsFamily(...)`
    - `planMMAv5ScaledAccumulatorFamily(...)`
  - all three now first try `planMMAv5ExactFamily(...)` when the input is
    legacy `#ttng.tensor_memory_encoding` sugar, then fall back to the broader
    linear-family matcher
  - result:
    - legacy sugar remains parseable/usable at the source and IR level
    - direct codegen stays on the same exact family path as the equivalent
      canonical linear layout instead of spuriously rejecting in conversion

- 2026-03-29: Blackwell conversion checks refreshed to the current direct codegen
  - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - updated the multi-tile MMA expectations to the current two-instruction
      direct emission
    - updated unpacked-f16 ld/st expectations to the current
      `st ... x64.b32` + `ld ... x64.pack::16b.b32` split
    - updated scales direct ld/st expectations to the current
      `16x32bx2` immediate form
    - relaxed the unpacked `x1` check so it no longer assumes the integer
      constants are materialized after the bitcast
  - validation:
    - `ninja -C build/cmake.linux-aarch64-cpython-3.12 triton-opt`
    - `cd build/cmake.linux-aarch64-cpython-3.12 && lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir test/TritonNvidiaGPU/tmem_layouts.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
      -> `4 passed`

- 2026-03-29: clean-cache full TMEM runtime matrix is green again on the current tree
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `rm -rf /root/.triton/cache/* && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=python:. pytest -s --tb=short -n 4 python/test/gluon/test_tmem_runtime_matrix.py`
      -> `1664 passed, 119 skipped in 149.43s`
  - this rerun covers the current direct-codegen state, including:
    - scales ld/st recovery
    - `ld.red` fallback removal
    - the MMAv5 legacy exact-family fast path

- 2026-03-29: read-only audits of the remaining semantic debt
  - `ld.red`
    - the direct lowering path is still narrower than generic ld/st
    - the default reduction-layout helper in
      `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` still hard-limits to
      `numWarps == 4`
    - missing targeted coverage remains for descriptor-view `ld.red`,
      scales rejection, unpacked rejection, and two-CTA reduction
  - MMAv5
    - the user-visible legacy-sugar rejection is fixed, but the semantic core
      still compresses arbitrary linear TMEM layouts through
      `MMAv5TMemLayoutPlan` / finite family recovery
    - the main bottlenecks remain the family planners in
      `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`, verifier consumers in
      `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`, allocator coupling in
      `lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp`, and
      final lowering in
      `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`

- 2026-03-29: `ld.red` source-layout legality is now enforced at the helper + verifier boundary
  - `python/src/gluon_ir.cc`
    - `compute_tmem_reduce_reg_layout_from_memdesc(...)` now immediately
      rejects TMEM descriptors whose source layout is not
      `isReductionFriendlyTmemSourceLayout(...)`
  - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
    - `TMEMLoadOp::verify()` now rejects `redOp` on source TMEM layouts that
      are not directly `tcgen05.ld.red`-compatible instead of silently
      accepting them and leaving runtime to miscompute
  - `python/test/gluon/test_core.py`
    - added a direct positive regression for the canonical linear form
      equivalent to legacy `TensorMemoryLayout(block=(128, 128), col_stride=1)`
      at shape `256x128`, `num_warps=8`
  - `python/test/gluon/test_tmem_runtime_matrix.py`
    - kept identity, tile-permuted, pure-column-permuted, and the canonical
      `256x128` legacy-equivalent linear case as positive `ld.red` runtime
      coverage
    - reclassified row-permuted and mixed TMEM-linear layouts to clean
      unsupported coverage
  - effective direct `ld.red` support boundary on the current tree:
    - positive:
      - canonical identity TMEM-linear layouts
      - pure-column permutations
      - tile-permuted layouts
      - the canonical linear form corresponding to legacy `256x128` `8-warp`
        reduction
    - clean unsupported:
      - row-permuted TMEM-linear layouts
      - mixed TMEM-linear layouts that mix row/column physical bits
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_core.py -k 'test_tmem_reduction_linear_legacy_block_equiv_layout or test_tmem_reduction and 256'`
      -> `32 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_identity_linear_layout or ld_red_tile_permuted_linear_layout or ld_red_col_permuted_linear_layout or ld_red_row_permuted_linear_layout_reports_clean_unsupported or ld_red_mixed_linear_layout_reports_clean_unsupported'`
      -> `104 passed`

- 2026-03-29: bounded sidecar probe on the remaining ld/st half-row/higher-rank bug
  - repro status stays negative for the direct row-half pattern:
    - `/tmp/test_direct_row_slice.py`
      - still `num_diff 4096`
      - wrong region pattern remains (first-half columns updated on both row
        halves instead of a clean bottom-half update)
    - `/tmp/test_candidate_layout.py`
      - projected full-view candidate layout also fails (`num_diff 4096`)
      - observed `x64` path still updates top-half rows only
  - key trace observation with
    `TRITON_TRACE_TMEM_QUERY_LOWERING_FILE=1`:
    - support-query lowering for the problematic 64x64 half-row view currently
      picks `I32x32b.x1` with:
      - `baseOffset=4194304` (`+64<<16`)
      - `warpBase0=2097152` (`+32<<16`)
      - `warpBase1=32` (column carry)
      - `reps` that only carry register-column steps
    - this indicates the remaining wrong-code is not a missing support origin
      carry anymore; it is a mapping/planning issue for the selected direct
      register-layout family.
  - bounded experiments and outcomes:
    - disabling the Gluon half-row special layout picker in
      `python/src/gluon_ir.cc` initially triggered a hard backend abort:
      `LLVM ERROR: Invalid basis 64 for in-dim 'warp' and out-dim 'row'`.
    - added a safety fix in
      `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` so
      `getTileLayout(..., warpBasis0, warpBasis1, ...)` grows the row out-dim
      extent based on warp basis coverage (rounded to power-of-two), mirroring
      the existing column-span handling.
      - this removes the invalid-basis abort for that path.
    - removing support-base adjustment in
      `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
      did not fix the half-row bug and was reverted.
  - current conclusion:
    - no clean direct-lowering fix landed for the half-row positive bucket in
      this probe.
    - the row/col support-origin plumbing is active, but the selected `x1`
      half-row mapping remains semantically wrong.
    - likely next step is a new direct decomposition/planner for row-half views
      carved from larger backing tiles rather than another base-offset tweak.
  - validation run during this probe:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `/tmp/test_direct_row_slice.py` (multiple clean-cache reruns)
    - `/tmp/test_candidate_layout.py` (clean-cache rerun)

- 2026-03-29: multidimensional `32x32` TMEM descriptor subviews are back to
  exact-final-view lowering only
  - root cause:
    - the old half-row fix only disabled support-query rescue for view-like
      `32x32` descriptors
    - bad multidimensional `reshape -> subslice -> reshape` views (identity and
      scrambled-cols) were still accepted through the canonical-surrogate query
      type path in `getTMemLdStQueryTypes(...)`
    - that let `get_reg_layout(auto)` recover a backing-tile `128-row` plan for
      a final `32x32` view whose standalone direct lowering is actually
      unsupported, producing wrong-code
  - fix:
    - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
      - `getTMemLdStQueryTypes(...)` now suppresses canonical-surrogate query
        types for explicit view-like `32x32` TMEM descriptors
    - `python/src/gluon_ir.cc`
      - the raw-query `32x32` view path no longer forces a backing row-plan
        override before probing direct reg layouts
    - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
      - verifier-side raw-query probing matches the same `32x32` exact-view
        rule
    - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
      - lowering-side raw-query probing matches the same `32x32` exact-view
        rule
  - result:
    - bad multidimensional `32x32` identity and scrambled-cols descriptor views
      are cleanly rejected again
    - the mixed positive stays directly supported
    - surrounding higher-rank ld/st descriptor buckets remain green
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'multidim_slice_identity_reports_clean_error or multidim_slice_reports_clean_unsupported or multidim_slice_positive'`
      -> `3 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'multidim_slice or higher_rank_half_rows or higher_rank_dim0_slice'`
      -> `99 passed, 1 skipped`
    - `lit -v test/TritonNvidiaGPU/invalid.mlir test/TritonNvidiaGPU/ops.mlir`
      -> `2 passed`

- 2026-03-29: full TMEM runtime matrix is green again after the
  multidimensional-ld/st and stale-expectation fixes
  - updated `python/test/gluon/test_tmem_runtime_matrix.py` for two expectation
    drifts:
    - small-shape scales `16x32bx2` direct sweeps now expect the shorter direct
      opcode sequences the current lowering actually emits:
      - `16x8, num_warps=4` -> one `st` + one `ld`
      - `16x16, num_warps=4` -> two `st` + two `ld`
      - `16x16, num_warps=8` -> one `st` + one `ld`
    - `ld.red` clean-unsupported tests now check the current verifier message:
      `tmem_load reduction source layout is not directly tcgen05.ld.red-compatible; use tmem.load(...)+tt.reduce(...) explicitly for software reduction`
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_scales_variant_sweep and 16x32bx2'`
      -> `13 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_row_permuted_linear_layout_reports_clean_unsupported or ld_red_mixed_linear_layout_reports_clean_unsupported'`
      -> `60 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py`
      -> `1668 passed, 119 skipped in 18.00s`

- 2026-03-29: MMAv5 accumulator-side consumers now use direct accumulator
  layout info instead of the recovered family-plan API
  - added a new public accumulator query in
    `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h` /
    `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`:
    - `getMMAv5AccumulatorLayoutInfo(...)`
    - `getMMAv5ScaledAccumulatorLayoutInfo(...)`
  - the new info bundles:
    - canonical TMEM linear layout
    - selected MMAv5 instruction sizes `mmaSizeM/N`
    - `colStride`
    - `twoCTAs`
    - `interleavedM64`
  - accumulator-side users switched from `MMAv5TMemLayoutPlan` to the new
    info:
    - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
      - plain/scaled MMA verifier D-side checks
    - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
      - `impl::verifyMMAv5Op`
    - `lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp`
      - row-anchor constraint logic for accumulator allocations
    - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
      - D-side instruction-size selection in lowering
  - the old `MMAv5TMemLayoutPlan` API still exists for the remaining LHS /
    promotion paths, but the D-side verifier/lowering/allocation stack is no
    longer directly wired to it
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `lit -v test/TritonNvidiaGPU/invalid.mlir test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/tmem_layouts.mlir test/TritonNvidiaGPU/canonicalize.mlir test/TritonGPU/accelerate-matmul.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
      -> `6 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_mma_acc_tile_permuted or test_tmem_runtime_matrix_mma_lhs_tile_permuted or test_tmem_runtime_matrix_mma_lhs_subslice_view or test_tmem_runtime_matrix_mma_scaled_acc_blockn32_direct_layout or test_tmem_runtime_matrix_mma_scaled_acc_subslice_view or test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view'`
      -> `8 passed`

- 2026-03-29: `ld.red` now directly supports pure row-permuted TMEM linear
  source layouts
  - root cause:
    - `isReductionFriendlyTmemSourceLayout(...)` accepted tile permutations and
      col permutations, but still required the row bases to appear in canonical
      order
    - that made all pure row-permuted source layouts fail cleanly even though
      the direct `ld.red` lowering path can handle them
  - fix:
    - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
      - collect row bases, sort them, and compare against the expected
        canonical set `{1, 2, 4, 8, 16, 32, 64}` instead of requiring
        canonical order
    - `python/test/gluon/test_tmem_runtime_matrix.py`
      - convert the row-permuted `ld.red` clean-negative bucket into positive
        runtime coverage
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `lit -v test/TritonNvidiaGPU/invalid.mlir test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/tmem_layouts.mlir test/TritonNvidiaGPU/canonicalize.mlir test/TritonGPU/accelerate-matmul.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
      -> `6 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_'`
      -> `104 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py`
      -> `1668 passed, 119 skipped in 519.91s`

- 2026-03-29: plain ld/st `32x32` multidim-slice identity still needs a
  dedicated support-query helper; raw-query row-plan override alone is not
  enough
  - attempted bounded fix:
    - add `getDirect32x32SubviewRawRowPlan(...)` in
      `TensorMemoryUtils.cpp/.h`
    - let frontend/verifier/lowering reuse the backing row plan on the raw
      query path for canonical contiguous view-like `32x32` f32 subviews
  - result:
    - the representative focused slice stayed unchanged:
      `pytest -k 'multidim_slice_identity_reports_clean_error or multidim_slice_reports_clean_unsupported or multidim_slice_positive'`
      -> `3 passed`
    - the identity case still clean-fails
  - trace result from `TRITON_TRACE_TMEM_REG_LAYOUT_FILE=1` on
    `test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_identity_reports_clean_error`:
    - the initial full-tile `16x128b` layout still succeeds through raw query
    - the final `view.get_reg_layout(auto)` never reaches the new raw-row-plan
      override because `inferStandaloneTMemLdStQueryLayout(...)` itself fails
      first with:
      `unsupported tensor memory memdesc_subslice view`
  - concrete implication:
    - the next patch should not keep pushing on the raw-query row-plan hook
    - it needs a dedicated `32x32` support-query helper that constructs a
      safe scalarizable query layout plus backing row plan for canonical
      contiguous view-like `32x32` subviews
    - `scrambled_cols` should stay negative in that patch

- 2026-03-29: lifted half-row ld/st (`reshape((2, M/2, N)).slice(1).index(0)`)
  is down to a direct TMEM packet/origin question, not a frontend
  reg-layout-discovery hole
  - representative bucket:
    - `python/test/gluon/test_tmem_runtime_matrix.py`
      `tmem_ldst_descriptor_higher_rank_half_rows_positive_kernel`
    - focused probe:
      - identity layout
      - `N=64`
      - `instr_variant='32x32b'`
      - `num_warps=4`
  - direct support status after bounded experiments:
    - removing the blanket
      `isUnsupportedDirectTMemLdStDescriptorView(...)` reject when a support
      query exists allows the part view to compile again
    - `ViewOpToLLVM.cpp` now zeroes the dim-0 subslice base for the lifted
      row-half `reshape -> subslice([1,0,0])` pattern, so the descriptor SSA
      value no longer carries the old extra `128<<16` row delta
    - but the runtime result is still wrong
  - concrete probe results:
    - widened `128x64` support-query path with `I32x32b.x64`:
      - PTX shape: direct `tcgen05.ld/st.sync.aligned.32x32b.x64.b32`
      - runtime: still updates rows `0:64` instead of `64:128`
      - representative TMEM base seen in PTX after the row-half subslice fix:
        `full_view`: `+64`
        `part_view`: `+4194368` (`0x400040`)
    - widened support-query path plus forced scalarization (`x1` packets):
      - PTX shape: 64x `tcgen05.ld/st.sync.aligned.32x32b.x1.b32`
      - runtime: still updates rows `0:64`
    - forcing the logical `64`-row warp anchors (`16/32`) onto the lifted
      support path:
      - result: `CUDA error: misaligned address`
      - so that anchor pair is not valid as a post-hoc override on the current
        support-tile packet plan
    - direct `64x64` query-layout-with-origin attempt:
      - reg-layout discovery does not stay on `I32x32b`
      - it falls back to `I16x32bx2` (`matchesDesiredAtom(...)` currently
        accepts that fallback for `32x32b` requests on `64x64` f32 views)
      - runtime remains wrong (`row_minmax 0..111`)
  - current best evidence:
    - the remaining hole is not “find any reg layout” anymore
    - it is whether a lifted upper-half `64x64` TMEM view has a physically
      correct direct packet decomposition under the real tcgen05 row-anchor
      rules
    - the two current candidate models are both bad:
      - support-tile `128x64` + lifted origin: wrong first half
      - logical-query `64x64` + lifted origin: degrades to `I16x32bx2`
        fallback and still wrong

- 2026-03-29: MMAv5 no longer exposes the old public family-plan API on the
  LHS side
  - cleanup:
    - added `MMAv5LhsLayoutInfo` in
      `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h`
    - moved `MMAv5TMemLayoutPlan` to an internal-only helper in
      `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
    - removed external users of:
      - `getMMAv5LhsLayoutPlan(...)`
      - `getMMAv5AccumulatorLayoutPlan(...)`
      - `getMMAv5ScaledAccumulatorLayoutPlan(...)`
  - updated users:
    - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
      - plain/scaled MMA verifier LHS checks now consume
        `getMMAv5LhsLayoutInfo(...)`
    - `lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp`
      - LHS/acc row-anchor coupling now uses `mmaSizeM` from layout info
    - `lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp`
      - canonical promoted-A layout now keys off accumulator layout info
    - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
      - TMEM-LHS detection now uses LHS layout info rather than the old plan
  - practical effect:
    - the remaining public MMAv5 semantic queries are now linear-layout info
      queries on both the A and D sides
    - the finite-family plan still exists internally inside `Dialect.cpp`, but
      verifier/lowering/allocation no longer consume it as a public semantic
      API
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `lit -v test/TritonNvidiaGPU/invalid.mlir test/TritonNvidiaGPU/ops.mlir test/TritonGPU/accelerate-matmul.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
      -> `4 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_mma_acc_tile_permuted or test_tmem_runtime_matrix_mma_lhs_tile_permuted or test_tmem_runtime_matrix_mma_lhs_subslice_view or test_tmem_runtime_matrix_mma_scaled_acc_blockn32_direct_layout or test_tmem_runtime_matrix_mma_scaled_acc_subslice_view or test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view'`
      -> `8 passed`

- 2026-03-29: bounded dense-copy permutation experiment stays rejected; forcing
  it through produces logical wrong-code
  - attempted bounded widening:
    - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
      - temporarily relaxed the dense-family `isDirectTMemCopyLayoutSupported`
        row/col basis-order gate to allow pure permutations of the canonical
        basis sets
      - temporarily widened `getTMemCopyDescriptorLayouts(...)` to try a
        sorted-basis descriptor candidate for dense families
  - focused runtime result:
    - representative case:
      - `python/test/gluon/test_tmem_runtime_matrix.py`
        `tmem_copy_no_scales_linear_kernel`
      - layout:
        `_make_tmem_linear_layout_permuted(128, 128, "identity", "rotate1")`
    - outcome with the widened descriptor search:
      - compile/lowering succeeded
      - PTX still emitted canonical
        `tcgen05.cp.cta_group::1.128x256b`
      - runtime was wrong:
        - `torch.equal(out, inp) == False`
        - `mismatch_count = 16128`
        - row 0 observed:
          - input:  `[0, 1, 2, 3, 4, 5, 6, 7, ...]`
          - output: `[0, 64, 1, 65, 2, 66, 3, 67, ...]`
  - conclusion:
    - this is not a harmless descriptor-ordering cleanup
    - forcing pure dense TMEM row/col permutations through the current
      `tcgen05.copy.128x{128,256}b` path changes logical semantics instead of
      preserving the user layout
    - the original clean-negative gate is therefore still the right boundary
      until we have a descriptor/address model that can faithfully realize
      those logical permutations
  - action taken:
    - reverted the experimental widening
    - restored the dense-family clean-negative expectations
  - validation after revert:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales_linear_rowcol_permuted_reports_clean_unsupported or cp_no_scales_linear_exotic_reports_clean_unsupported'`
      -> `18 passed`

- 2026-03-29: lifted row-half ld/st rescue is still not directly realizable;
  restored the clean-negative boundary
  - bucket:
    - `python/test/gluon/test_tmem_runtime_matrix.py`
      `test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout`
  - what I re-checked:
    - the multidim descriptor chain
      `full_view.reshape((2, M // 2, N)).slice(1, 1, dim=0).index(0)`
      over the lifted identity TMEM layout
    - raw direct lowering still hits the old sparse wrong-code footprint for
      `32x32b`:
      - modified rows: `0-15,48-79,112-127`
    - a `128x64` support-query rescue still over-covers the backing tile when
      scalarized to `tcgen05.ld/st.sync.aligned.32x32b.x1.b32`
    - a bounded `64x64` canonical support-query rescue falls back to
      `I16x32bx2`/local 64-row anchors and reproduces the same sparse wrong
      footprint instead of the logical upper-half view
  - conclusion:
    - this lifted row-half descriptor view is still not directly realizable by
      the current `tcgen05.ld/st` packet families without changing semantics
    - the right state today is a clean descriptor-view rejection, not a rescue
      that compiles to sparse or over-wide wrong-code
  - action taken:
    - removed the half-row support-query rescue from
      `getTMemLdStSupportQueryLayout(...)`
    - kept the existing verifier-side clean-negative wording for lifted
      row-half TMEM views
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout'`
      -> `12 passed`

- 2026-03-29: mixed `ld.red` layout with shared row/col bases stays a clean reject
  - Reviewed `TMEMLoadOp::verify` and the helper `isReductionFriendlyTmemSourceLayout` in
    `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` / `Dialect.cpp`; the mixed layout fails because the
    normalized TMEM linear layout strips zero bases, expects `row` to encode a 128-row block with
    pure row bases, and the mixed row dimension carries `[0,1]`/`[0,2]` column bits that break that
    canonical set.
  - The lowering path (`TensorMemoryToLLVM.cpp` + `lowerTMemLdStFromTypes`) never finds a valid
    register layout when row column carriers violate the 128-row anchor plan, so accepting it would
    silently emit wrong semantics rather than a clean reduction packet.
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_mixed_linear_layout_reports_clean_unsupported`
    still watches for the clean-negative diagnostic, so the current verifier message is the right
    boundary until a new decomposition strategy for those mixed row/col bases exists.

2026-03-29 12:40Z - Row/col-permuted MMAv5 accumulators remain unsupported

- Experiment: relaxed the MMAv5 accumulator layout matcher so row/col-permuted TMEM-linear accumulators would pass the verifier, then tried to let the runtime convert the permuted register layout back to the canonical AutoLayout before storing the result.
- Validation: `TRITON_BUILD_WITH_CCACHE=true make -j96` (pass) followed by `PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_rowcol_permuted_layout` (fails because the `tt.make_range` expansion in the converted kernel could not infer a return type after the AutoLayout conversion and ultimately emitted `PassManager::run failed`).
- Conclusion: even though the matcher could be widened, the kernel still miscompiles (and the emitted data layout cannot be trivially converted back to row-major), so the clean-negative behavior should remain in place.

- 2026-03-29: multidimensional `32x32` ld/st slice bucket is green again
  - remaining full-matrix failures were:
    - `test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_identity_reports_clean_error`
      compiled instead of rejecting
    - `test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_positive[mixed]`
      emitted wrong-code (`1536` mismatches)
  - root causes split into two separate branches:
    - identity:
      - the explicit descriptor view inferred an **exact dense row/col
        `32x32` TMEM-linear type**
      - the dedicated `32x32` support-plan rescue still admitted that exact
        canonical view, even though it comes from a larger backing tile and is
        not directly representable
    - mixed:
      - the direct `queryTy` scalar `tcgen05.ld/st.sync.aligned.32x32b.x1.b32`
        path was selected, but the recovery path for collapsed scalar reps was
        advancing the low-bit PTX immediate by `1` instead of `4`
      - that changed the packet sequence from the validated
        `0,4,8,...,28` col-immediates to `0,1,2,...,7`
  - fixes:
    - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
      - added an exact-literal `32x32` dense row/col view predicate for
        explicit TMEM descriptor views
      - exact dense `32x32` views from larger backing tiles now stay on the
        clean-negative path in `isUnsupportedDirectTMemLdStDescriptorView(...)`
      - the dedicated subview-support helper now also declines that exact
        canonical case
    - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
      - scalarized `32x32b.x1` recovery now advances PTX col-immediates in
        `.b32` byte units (`4` bytes per packet), restoring the intended
        `0,4,8,...,28` sequence
  - result:
    - identity is back to a clean compile-time descriptor-view rejection
    - mixed is back to a real positive with correct numerics and expected PTX
      packet offsets
    - scrambled-cols stays clean unsupported
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'multidim_slice_identity_reports_clean_error or multidim_slice_positive or multidim_slice_reports_clean_unsupported'`
      -> `3 passed`
- 2026-03-29: MMAv5 barriered Gluon kernels must not also issue explicit
  `tcgen05_commit(...)`
  - barriered `tcgen05_mma(..., mbarriers=[bar])` already lowers to
    completion-barrier signaling
  - the earlier extra `tcgen05_commit(bar)` in the runtime kernels emitted a
    second `tcgen05.commit...mbarrier::arrive::one...` in PTX/LLIR and wedged
    runtime execution
  - removing the redundant explicit commit restored the canonical plain MMA and
    indexed-acc-view positives
  - validation:
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma[legacy_no_acc-layout0-False]'`
      -> `1 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_mma_indexed_acc_view and legacy_parent'`
      -> `1 passed`
- 2026-03-29: MMAv5 direct-layout matcher now rejects in-tile row/col basis
  permutations again
  - root cause:
    - `planMMAv5Family(...)` was comparing the low-order TMEM row/col tile
      bases as a **sorted multiset**, which incorrectly treated arbitrary
      in-instruction basis permutations as MMAv5-compatible
  - fix:
    - require exact low-order basis order equality inside the candidate MMA
      tile and only allow freedom in the higher-order whole-tile repetition
      bases
  - effect:
    - scrambled / row-col permuted TMEM-linear accumulator layouts are back to
      clean verifier rejection
    - intended tile-permuted whole-tile layouts remain directly supported
    - scaled `tile_n=32` still rejects cleanly through the dedicated repeated
      `N=32` verifier path
  - tests updated:
    - runtime row/col-permuted MMA bucket now expects clean unsupported
    - `test/TritonNvidiaGPU/invalid.mlir` has a dedicated row/col-permuted
      MMAv5 accumulator rejection
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_exotic_layout_reports_clean_unsupported or mma_rowcol_permuted_layout or mma_acc_tile_permuted or mma_scaled_acc_tile_permuted_64_direct_layout or mma_scaled_acc_tile_permuted_32_repeated_n32_reports_clean_unsupported'`
      -> `21 passed`
    - `bin/triton-opt --split-input-file /root/code/triton/test/TritonNvidiaGPU/invalid.mlir --verify-diagnostics`
      -> success
- 2026-03-29: broad MMAv5 runtime slice is green after restoring the verifier
  boundary
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma'`
    -> `196 passed, 12 skipped`
  - this covers:
    - canonical plain MMA
    - indexed/subslice views
    - two-CTA cases
    - tile-permuted direct positives
    - scaled MMAv5 direct positives
    - the clean-negative exotic / row-col-permuted / repeated-`N=32` scaled
      buckets
- 2026-03-29: full TMEM runtime matrix is green again after the MMAv5 verifier
  boundary fix
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py`
    -> `1668 passed, 119 skipped`
- 2026-03-29: frontend/core validation is green on the TMEM surface
  - fixes after the full runtime matrix:
    - refreshed stale frontend expecttests for TMEM subview/index IR now that
      memdesc view/result types keep the precise inferred
      `#ttng.tensor_memory_linear` encoding
    - fixed `test_shared_memory_cast` to preserve the total number of bits in
      the shared-memory reinterpret (`16384xi8`, not `1024xi8`)
    - fixed Gluon `_check_tensor_memory_layout_ctas(...)` so `two_ctas=True`
      layouts honor the actual `cga_layout` CTA count when present instead of
      collapsing every two-CTA layout to `2` CTAs per CGA
    - updated stale `ld.red` clean-negative substrings in
      `python/test/gluon/test_core.py` to the current direct-lowering
      diagnostics
  - validation:
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_frontend.py`
      -> `205 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_core.py -k 'tmem_reduction or tcgen05_mma_multicast_commit'`
      -> `92 passed`
    - `build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt /root/code/triton/test/TritonNvidiaGPU/ops.mlir >/tmp/tmem_ops_out.mlir`
      -> non-empty output (`226` lines)
    - `build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt --split-input-file /root/code/triton/test/TritonNvidiaGPU/invalid.mlir --verify-diagnostics`
      -> success
- 2026-03-30: stale lit fallout cleared and current full lit is green again
  - compared stale CHECK-only files against `/tmp/triton-origin-main` output and
    updated expectations where the underlying pass/output intent stayed the same
  - fixed the real tensor-memory-scales / pipeline fallout and refreshed the
    stale `memdesc_subslice` / canonical TMEM-linear print expectations
  - current validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `cd build/cmake.linux-aarch64-cpython-3.12 && ninja check-triton-lit-tests`
      -> `248 passed, 2 unsupported`
- 2026-03-30: split-N permuted `64x2` ld/st direct support restored
  - root cause:
    - `getTMemLdStRowPlanForType(...)` started deriving the active `M64`
      split-N row footprint from the normalized TMEM-linear layout without
      first removing the explicit zero row basis
    - for `64x2` split-N layouts that left the active row-bit count widened, so
      `get_reg_layout(...)` stopped recovering the 64-row row plan and the
      direct `16x32bx2` / `32x32b_splitn` path fell off the supported set
  - fix:
    - restored the active-row query to use
      `normalizedLayout.removeZeroBasesAlongDim(kRow)` in
      `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
  - validation:
    - `PYTHONPATH=python:. python - <<'PY' ... tensor_memory_descriptor_type(float32, [64,2], layout, [64,2]).get_reg_layout(...) ... PY`
      -> `auto`, `32x32b`, `16x64b`, `16x32bx2`, and `32x32b_splitn`
         all return concrete layouts again for `_make_tmem_linear_layout_m64(2)`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_explicit_16x32bx2_matches_splitn or test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep'`
      -> `231 passed`
- 2026-03-30: copy-scales clean-unsupported diagnostics re-normalized in late lowering
  - remaining `warpx2`-candidate scales copies are still intentionally
    unsupported, but after shared-memory allocation they now fail in
    `TensorMemoryToLLVM.cpp`
  - updated late lowering to emit the same family-specific
    `maps to tcgen05.copy... could not synthesize a compatible shared-memory
    descriptor plan for tensor memory scales` guidance before the outer
    `PassManager::run failed`
  - updated the runtime-matrix tests to accept the current outer exception
    wrapper while still asserting the real unsupported reason
  - validation:
    - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. pytest -s --tb=short -x 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_layout_probe[warpx2_candidate-smem_layout1-CLEAN_UNSUPPORTED]'`
      -> `1 passed`
    - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. pytest -s --tb=short -x python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_bug_cp_scales_unsupported_layout_raises_runtimeerror_parse`
      -> `1 passed`
- 2026-03-30: broad Gluon/TMEM Python validation is green on the current tree
  - compile/tooling smoke:
    - `PYTHONPATH=python:. pytest -s --tb=short python/test/unit/language/test_compile_only.py`
      -> `7 passed`
    - `PYTHONPATH=python:. pytest -s --tb=short python/test/unit/language/test_frontend.py`
      -> `34 passed`
    - `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_frontend.py`
      -> `205 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_core.py -k 'tmem_reduction or tcgen05_mma_multicast_commit'`
      -> `92 passed`
    - `PYTHONPATH=python:. pytest -s --tb=short python/test/unit/tools/test_triton_to_gluon.py`
      -> `16 passed`
  - split GPU runtime batch:
    - ran `python/test/gluon/test_tmem_runtime_matrix.py` with
      `pytest -s --tb=short --splits 4 --group N` pinned to
      `CUDA_VISIBLE_DEVICES=0/1/2/3`
    - aggregate result:
      - group 1 -> `447 passed`
      - group 2 -> `367 passed, 80 skipped`
      - group 3 -> `414 passed, 33 skipped`
      - group 4 -> `440 passed, 6 skipped`
      - total -> `1668 passed, 119 skipped`
  - `triton_kernels` smoke:
    - exact nvfp4 Blackwell slice:
      - `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. pytest -s --tb=short -x 'python/triton_kernels/tests/test_matmul.py::test_op[None-True-False-False-False-None-16-16-256-256-ragged-nvfp4_e2m1-nvfp4_e2m1-bfloat16-10-1-False-True-None-False-False-False-True-None]'`
        -> `1 passed`
      - `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. pytest -s --tb=short -x 'python/triton_kernels/tests/test_matmul.py::test_op[None-True-False-False-False-None-16-1000-704-800-batched-nvfp4_e2m1-nvfp4_e2m1-bfloat16-10-1-True-True-None-False-False-False-True-None]' 'python/triton_kernels/tests/test_matmul.py::test_op[None-True-False-False-False-None-16-1024-1024-1024-batched-nvfp4_e2m1-nvfp4_e2m1-bfloat16-10-1-False-True-None-False-False-False-True-None]'`
        -> `1 passed, 1 skipped`

- 2026-04-01: fixed the GB200 `python/test/unit/language/test_core.py::test_dot` MMAv5 regression that showed up once the unit suite was split across GPUs
  - exact failing nodes before the fix:
    - `test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
    - `test_dot[1-64-64-64-4-False-False-chain-dot-ieee-bfloat16-float32-1-None]`
  - the regression had two parts:
    - canonical 4-warp M64 register-layout selection in `Dialect.cpp` had drifted to the split-N `x16` path (`laneSplitCol = n/4`) instead of the full `x32` path the pre-generalization code used for ordinary 64x64 MMAv5 accumulators
    - initialized `ttng.tmem_alloc %src` lowering in `TensorMemoryToLLVM.cpp` was still taking the type-only ld/st query path, which collapsed canonical M64 TMEM-linear allocations to a `64x64` standalone query (`warpBase0=2^20`, `warpBase1=2^21`) instead of the real backing-row support form (`128x64`, `warpBase0=2^21`, `warpBase1=2^22`)
  - symptoms and evidence:
    - origin/main passed those exact nodes, current tree failed with ~49.5% mismatches
    - PTX diff showed current code emitting `tcgen05.{st,ld}.16x32bx2.x16.b32` for the accumulator path until the canonical-M64 fix restored `x32`
    - TMEM query trace then showed the remaining mismatch: plain `tmem_store` / `tmem_load` on the memdesc value used the raw `128x64` support query, while initialized allocs still used the narrower `queryType 64x64` path
  - fix:
    - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`: restore the canonical 4-warp M64 `x32` register-layout selection (`laneSplitCol = n/2`)
    - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`: pass `op.getResult()` into `lowerTMemLdStFromTypes(...)` for initialized allocs so the lowering can reuse the real memdesc query/support path
    - kept the MMAv5/TMEM legacy-equivalent bridge in `MMAv5.cpp` and the matching canonical-M64 direct-ld/st bridge in `TensorMemoryUtils.cpp`, since they keep explicit TMEM-linear M64 layouts on the same physical addressing model as the old legacy sugar
  - validation:
    - fresh-cache exact nodes:
      - `...tf32x3...` -> `1 passed`
      - `...chain-dot-ieee-bfloat16...` -> `1 passed`

- 2026-04-01: fixed the GB200 `python/test/unit/language/test_matmul.py::test_simple_matmul` epilogue-subtile regressions for explicit TMEM-linear accumulators
  - exact failing nodes before the fix:
    - `test_simple_matmul[True-True-4-1-128-128-16-4-float16-float16]`
    - `test_simple_matmul[True-True-4-1-128-128-16-4-float32-tensorfloat32]`
  - failure pattern:
    - `EPILOGUE_SUBTILE=True` materializes two `ttng.tmem_subslice` views (`N=0` / `N=64`) and loads each half separately
    - TF32 loaded the left half correctly but duplicated it into the right half
    - packed f16 needed the old packed-aware query path; using the same direct-origin remap path as TF32 broke the right half again
  - root cause:
    - raw `ttng.tmem_subslice` ld/st query lowering was conflating two cases
    - 32-bit views need the direct translated-origin remap so the raw-query `baseOffset` survives into lowering for the second half-tile
    - packed subword views still need the packed-aware fallback query construction so the raw query preserves the packed TMEM column structure before lowering
  - fix:
    - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
      - keep the new direct-origin remap for `ttng.tmem_subslice`, but only on 32-bit-and-up element types
      - route packed subword `ttng.tmem_subslice` views through the packed-aware resized-layout fallback
      - keep `ttng.tmem_subslice` raw-query ordering preferring the standalone type before the raw memdesc type for subword layouts
    - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
      - preserve raw-query `baseOffset` for `ttng.tmem_subslice` so the translated right-half origin is not dropped during direct ld/st lowering
  - evidence:
    - TF32 PTX before the fix emitted the second `ttng.tmem_load` from the same effective base as the first half, so the right output half was an exact duplicate of the left
    - after preserving the raw-query base offset, TF32 passed, but packed f16 regressed until the fast direct-origin remap was restricted to 32-bit-and-up layouts
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. TRITON_DISABLE_LINE_INFO=1 PROTON_SKIP_PC_SAMPLING_TEST=1 python3 -m pytest -s --tb=short -x 'python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-128-128-16-4-float32-tensorfloat32]'`
      -> `1 passed`
    - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. TRITON_DISABLE_LINE_INFO=1 PROTON_SKIP_PC_SAMPLING_TEST=1 python3 -m pytest -s --tb=short -x 'python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-128-128-16-4-float16-float16]'`
      -> `1 passed`

- 2026-04-01: root-TMEM direct ld/st raw-layout fast path for GB200 `16x256b` epilogue subtiles
  - symptom:
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-64-128-32-4-float16-float16]` failed on current HEAD `69c7822a8` with `TRITON_PREFER_TMEM_16x256_LAYOUT=1` because PTX fell back to `16x128b` instead of the expected `16x256b`
    - `/tmp/triton-origin-main` still passed the same node and emitted `tcgen05.st/ld.sync.aligned.16x256b.x8.{unpack,pack}::16b.b32`
  - root cause:
    - the newer direct ld/st helpers were routing even whole-root TMEM descriptors (`shape == allocShape`) through the generalized TMEM-view analysis layout path
    - that normalized whole `64x128 f16` M64 accumulator descriptor no longer matched the legacy-equivalent `16x256b` whole-tile direct path, so selection fell back to `16x128b`
    - origin/main still uses the raw storage layout (`toLinearLayout(shape, encoding)`) for whole descriptors, which preserves the working `16x256b.x8.unpack/pack` lowering
  - fix:
    - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
      - in `getDistributedLayoutForTmemLdSt(memType, atom, numWarps, rowPlanOverride)`, restore a raw-storage-layout fast path for non-scales whole descriptors when `shape == allocShape` and no row-plan override is active
    - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
      - in `computeTMemLdStEncodingInfo(regTy, memTy, ...)`, restore the same raw-storage-layout fast path for non-scales whole descriptors before falling back to the generalized TMEM-view analysis layout path
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. python3 -m pytest -s --tb=short -x 'python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-64-128-32-4-float16-float16]'`
      -> `1 passed`
    - direct repro now emits:
      - `tcgen05.st.sync.aligned.16x256b.x8.unpack::16b.b32`
      - `tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32`

- 2026-04-01: revalidated the committed GB200 TMEM checkpoint on clean HEAD `69c7822a8`
  - discarded the dead-end opcode-override experiments; no source changes beyond the committed checkpoint remain in the worktree
  - confirmed the exact GB200 matmul node still expected from the initiative note is green on the clean tree:
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-64-128-32-4-float16-float16]` -> `1 passed`
    - paired control `python/test/unit/language/test_matmul.py::test_simple_matmul[False-True-4-1-64-128-32-4-float16-float16]` -> `1 passed`
  - targeted adjacent smoke:
    - `lit -v test/Conversion/tritongpu_to_llvm_blackwell.mlir` -> `1 passed`
    - `python/test/gluon/test_core.py -k 'linear_m64_16x256b_64x128 and test_tmem_linear_m64_roundtrip_direct_shapes'` -> `1 passed`
    - `python/test/gluon/test_core.py::test_tmem_packed_f16_roundtrip_atom_shapes` -> `1 passed`

- 2026-04-01: GB200 unit matmul `256x128x32` f16 epilogue-subtile regression fixed on top of `69c7822a8`
  - failing node:
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-256-128-32-4-float16-float16]`
  - symptom before fix:
    - numerics wrong (`261331 / 524288` mismatches)
    - PTX showed the accumulator MMAv5 writes stepping by `+256`, while `/tmp/triton-origin-main` stepped by `+128`
    - the final load was still a single `tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32`; the numerical bug tracked the MMAv5 accumulator address model, not the load opcode family
  - root cause:
    - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
    - `DotOpMmaV5TmemLoader::build(...)` was re-normalizing TMEM-linear accumulator/LHS memdescs back through legacy-family matching before building the address model
    - for this whole-tile linear accumulator case that legacy recovery changed the physical accumulator stepping and over-advanced the second MMAv5 slice along N/M
  - fix:
    - remove the legacy-family normalization from `DotOpMmaV5TmemLoader::build(...)`
    - keep the loader on the raw `toLinearLayout(memTy)` physical TMEM mapping instead
    - this is also the correct long-term direction for the initiative: loader/codegen should consume the full linear form directly, not recover legacy shorthands internally
  - related cleanup:
    - `lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp`
    - broaden `TMemSplitLoadPattern` so its M-preservation check can look through a `ttg.memdesc_reinterpret` of the backing TMEM allocation when matching the reshape->trans->split load pattern
    - this did not change the PTX for the exact `256x128` GB200 node, but it keeps the split-load rewrite aligned with the newer TMEM-linear accumulator forms
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `make test-lit` -> `248 passed, 2 unsupported`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-64-128-32-4-float16-float16]` -> `1 passed`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-256-128-32-4-float16-float16]` -> `1 passed`
    - resumed GB200-style `python/test/unit` shards:
      - groups `13..16` across GPUs `0..3` -> all passed

- 2026-04-01: fixed GB200 tf32 matmul `64x512x32` direct accumulator-init regression on top of `69c7822a8`
  - failing node:
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-1-64-512-32-2-float32-tensorfloat32]`
  - symptom before fix:
    - `510572 / 524288` mismatches
    - current PTX scalarized the initial `ttng.tmem_store` into many `tcgen05.st.sync.aligned.32x32b.x1.b32` messages, while `/tmp/triton-origin-main` kept the wide `32x32b.x64.b32` path
  - root cause:
    - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
    - the legacy-anchored M64 direct ld/st rescue path was still trying to re-identify explicit TMEM-linear accumulator layouts from the raw encoding form; for the `64x512` linear accumulator that raw form exposed only the compact physical column span and never matched the actual normalized analysis/support layout used by the raw-query planner
    - because of that mismatch, the whole-root direct `tmem_store` path fell back to the generic `I32x32b.x1` selection even though the descriptor is exactly legacy-equivalent and should lower directly
  - fix:
    - keep the legacy-anchored M64 fast path, but match against the normalized direct-ld/st analysis layout already being lowered (`normalizeTensorMemoryLinearLayoutForAnalysis(memLayout)`) instead of reconstructing the family from the raw encoding bits
    - allow the fast path for the standard `warpRow0=32`, `warpRow1=64`, `rowSpan=128`, `baseOffset=0` row-plan override used by whole-root M64 descriptors
    - restrict the rescue to whole-root descriptors (`shape == allocShape`), so generalized descriptor-view lowering still goes through the view-analysis path
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `make test-lit` -> `248 passed, 2 unsupported`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-1-64-512-32-2-float32-tensorfloat32]` -> `1 passed`
    - paired controls stay green:
      - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-256-128-32-4-float16-float16]`
      - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-64-128-32-4-float16-float16]`

- 2026-04-01: fixed GB200 f16 matmul `64x512x32` whole-root M64 ld/st subword regression on top of `0fb4fd519`
  - failing node:
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-1-64-512-32-2-float16-float16]`
  - symptom before fix:
    - `515903 / 524288` mismatches after the first M64 rescue change
    - PTX had recovered the wide `tcgen05.st.sync.aligned.32x32b.x128` / `tcgen05.ld.sync.aligned.32x32b.x128` path, but it emitted the bare 32-bit form instead of the required `unpack::16b` / `pack::16b` suffixes used by `/tmp/triton-origin-main`
  - root cause:
    - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
    - the whole-root legacy-anchored M64 rescue was using a stripped TMEM-linear layout (with zero `col` bases removed) both to recognize the exact legacy-equivalent family and to build the direct ld/st lowering transform
    - that is fine for family equivalence matching, but wrong for lowering subword TMEM layouts: the stripped layout hides the raw zero-column-basis spacing that `lowerTMemLdSt(...)` uses to select `pack::16b` / `unpack::16b` semantics
    - the helper also inherited a 32-bit-only post-check (`getExpectedTMemLoadValueCount`) that rejects all subword cases by construction
  - fix:
    - split the helper into two layout views:
      - raw whole-root storage layout from `toLinearLayout(memTy.getShape(), memTy.getEncoding())` for the actual direct lowering transform
      - stripped family-equivalence layout only for exact legacy-family recognition
    - keep the `getExpectedTMemLoadValueCount(...)` sanity gate only for `bitwidth == 32`
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-1-64-512-32-2-float16-float16]` -> `1 passed`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-1-64-512-32-2-float32-tensorfloat32]` -> `1 passed`


- 2026-04-01: fixed persistent ragged `triton_kernels` MMAv5 accumulator N-half corruption for TMEM subslices
  - failing node:
    - `python/triton_kernels/tests/test_matmul.py::test_op[None-True-False-False-False-None-128-768-512-1024-ragged-float16-float16-None-10-1-False-False-None-False-False-False-True-None]`
  - symptom before fix:
    - structured corruption only in the second 128-column half of each 256-column MMAv5 accumulator tile (`columns 128:256` and `384:512` wrong; neighbors correct)
    - dumped PTX for `_p_matmul` showed the two `ttng.tmem_load` halves issuing `tcgen05.ld.sync.aligned.32x32b.x64.b32` from the same physical TMEM base after `ttng.tmem_subslice {N = 0}` / `{N = 128}`
  - root cause:
    - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
    - `TMEMSubSliceOpConversion` was computing the physical base advance from the narrowed result memdesc type instead of the source tile
    - for N-half views like `128x256 -> 128x128`, that erases the high-order column basis needed to distinguish the second half and collapses distinct subslices onto the same base address
    - `lib/Analysis/BufferRegion.cpp` had the same source-vs-result-type bug in its TMEM subslice offset accounting, so analysis/tests were also encoding the stale zero-offset assumption
    - the current worktree also carries a direct-ld/st query fix in `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp` so multibuffer `memdesc_index` support queries preserve only the already-lowered TMEM base translation instead of trying to re-encode the selected buffer as extra query origin
  - fix:
    - compute `ttng.tmem_subslice` physical offsets from the source memdesc type in both LLVM lowering and buffer-region analysis
    - update lit expectations for the now-correct nonzero TMEM subslice offsets (`test-buffer-region.mlir`, `consan.mlir`, `tritongpu_to_llvm_blackwell.mlir`)
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - exact repro -> `1 passed`
      - `python/triton_kernels/tests/test_matmul.py::test_op[None-True-False-False-False-None-128-768-512-1024-ragged-float16-float16-None-10-1-False-False-None-False-False-False-True-None]`
    - focused lit bundle -> `3 passed`
      - `test/Analysis/test-buffer-region.mlir`
      - `test/TritonGPU/consan.mlir`
      - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - note: `make test-lit` immediately after an incremental rebuild hit transient `triton-opt: Text file busy` fanout; rerunning lit separately from the already-built tree avoids the harness issue


- 2026-04-01: fixed GB200 Blackwell MMAv5 16x256 epilogue-subtile unit failures by disabling the buggy sub-32-bit split-load replay rewrite
  - failing GB200 shard node after the TMEM subslice/base-offset fixes:
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-128-128-16-4-float16-float16]`
  - control node that already passed with the direct ld/st base-offset fix:
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-128-128-16-4-float32-tensorfloat32]`
  - symptom before fix:
    - only the epilogue-subtile path failed, and only for sub-32-bit MMAv5 accumulators with `TRITON_PREFER_TMEM_16x256_LAYOUT=1`
    - the rewritten TTGIR replayed `reshape -> trans -> split` as two `ttng.tmem_subslice` / `ttng.tmem_load` pairs on `128x64xf16`
    - runtime corruption matched the second 64-column half of each 128-column tile (`48.9%` mismatches), while the original full-load path was otherwise healthy
  - root cause:
    - `lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp`
    - the split-load replay optimization is currently sound for 32-bit TMEM tiles, but not for the sub-32-bit `pack::16b` MMAv5 accumulator path
    - the transformed F16 kernel stayed numerically wrong even after fixing TMEM subslice physical base offsets and raw-query double-application of TMEM origins, which isolates the remaining bug to the replayed split-load rewrite itself rather than the base lowering
  - fix:
    - keep the exact user-visible semantics by falling back to the original full `ttng.tmem_load` epilogue path for sub-32-bit replay candidates instead of applying the buggy split-load rewrite
    - keep the earlier TMEM lowering fixes in place (`TMEMSubSliceOpConversion` source-type base offsets and raw-query baseOffset zeroing for already-adjusted `ttng.tmem_subslice` / `ttg.memdesc_index` views)
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - exact F16 repro -> `1 passed`
      - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-128-128-16-4-float16-float16]`
    - exact TF32 control -> `1 passed`
      - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-128-128-16-4-float32-tensorfloat32]`
    - persistent ragged `triton_kernels` control -> `1 passed`
      - `python/triton_kernels/tests/test_matmul.py::test_op[None-True-False-False-False-None-128-768-512-1024-ragged-float16-float16-None-10-1-False-False-None-False-False-False-True-None]`
    - full lit -> `248 passed, 2 unsupported`

- 2026-04-01: started the post-commit GB200 CI-equivalent sweep from `f70f9aeb6`
  - intent:
    - rerun the GB200 CI surface in split GPU microbatches (`--splits/--group`) across `CUDA_VISIBLE_DEVICES=0..3` so failures stop early and logs stay localized
  - validated so far:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `ninja -C build/cmake.linux-aarch64-cpython-3.12 check-triton-unit-tests` -> `240 passed`
    - `ninja -C build/cmake.linux-aarch64-cpython-3.12 check-triton-lit-tests` -> `248 passed, 2 unsupported`
    - `python/test/unit` main shard (split 32-way, excluding `plugins/*` and `test_debug.py`) -> all `32/32` groups passed
    - `python/test/unit/test_debug.py` -> `95 passed`
    - plugin/custom-op unit slices:
      - `python/test/unit/plugins/test_plugin.py` -> `1 passed`
      - `python/test/unit/plugins/test_dialect_plugin.py` -> `1 passed`
      - `python/test/unit/plugins/custom_ops.py` -> `1 passed`
  - in flight:
    - `python/triton_kernels/tests` under `/tmp/run_split_batches.sh gb200-triton-kernels 32 python/triton_kernels/tests`
    - first wave (`groups 1-4`) passed; later waves are still running

- 2026-04-01: GB200 CI-equivalent sweep checkpoint after completing `triton_kernels` and the remainder of `test-unit`
  - completed since the earlier checkpoint:
    - `python/triton_kernels/tests` split `32` ways across `CUDA_VISIBLE_DEVICES=0..3` -> all `32/32` groups green
      - aggregated group-log totals: `2013 passed, 3444 skipped`
    - `python/tutorials/06-fused-attention.py` split `16` ways -> all groups green (`384 skipped` total on this node)
    - `python/test/unit/instrumentation/test_gpuhello.py` -> `1 passed`
  - currently in flight:
    - `python/test/gluon/` + `python/tutorials/gluon/` split `32` ways across the `4` GPUs

- 2026-04-01: fixed the first GB200 `python/test/gluon` blocker before resuming the split sweep
  - failure:
    - `python/test/gluon/test_consan.py::test_aliasing_tensor_visibility_outstanding_read[1ctas-True]` failed in two stages during the staged `gb200-gluon` run
    - first, TMEM `slice(start, length)` no longer preserved the historical trailing-dimension shorthand and compiled the aliasing view as `128x256` instead of `128x128`
    - after restoring that shorthand, `warp_specialize` still rebuilt TMEM descriptor arguments through generic Python types and generated `tt.call` signatures that no longer matched the exact `ttg.memdesc_subslice` handle types
    - once that was fixed, the concurrency sanitizer still asserted in `BufferRegion.cpp` because generic `ttg.memdesc_subslice` on TMEM was routed through a shared-memory-only byte-offset inversion path
  - fix:
    - restored the public TMEM shorthand so `tensor_memory_descriptor.slice(start, length)` defaults to slicing the trailing dimension while explicit `dim=` keeps generic `memdesc_subslice` behavior
    - preserved exact TMEM memdesc IR types through Gluon semantic view construction (`slice/index/trans/reshape/reinterpret`) and through `tensor_memory_descriptor_type._unflatten_ir`, so `warp_specialize` block arguments and outlined helper calls keep the exact handle type instead of rebuilding an equivalent-but-distinct memdesc type
    - taught `BufferRegionAnalysis` to treat generic TMEM `ttg.memdesc_subslice` like TMEM views: use `ttng::getTMemViewOffset(...)` for exact offsets, and conservatively inherit the parent region when a generic shared-memory subslice cannot be inverted exactly instead of asserting
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - direct repro (expected device-side assert) -> reproduced the original outstanding-read failure again
      - `CUDA_VISIBLE_DEVICES=0 DISABLE_SUBPROCESS=1 PYTHONPATH=python:. python3 -m pytest -s --tb=short -x 'python/test/gluon/test_consan.py::test_aliasing_tensor_visibility_outstanding_read[1ctas-True]'`
    - wrapped test -> `1 passed`
      - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -s --tb=short -x 'python/test/gluon/test_consan.py::test_aliasing_tensor_visibility_outstanding_read[1ctas-True]'`
    - stale frontend expectation updated for the restored TMEM slice shorthand
      - `EXPECTTEST_ACCEPT=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -s --tb=short -x python/test/gluon/test_frontend.py::test_tensor_memory`


- 2026-04-01: fixed the GB200 `test_tmem_subslice_block_m_64` TMEM column-subview regression
  - failure:
    - `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[{legacy,linear}]` was still broken after the earlier GB200 checkpoint
    - `s_tmem.slice(...)._reinterpret(...).store(...)` on `64xN` TMEM tiles either rejected the full-tile `get_reg_layout()` layout or miscompiled the column-subview packets
    - the bad support-rescue path treated `64`-row TMEM column subviews like contiguous leading columns and shifted the `64x2` follow-up stores onto the wrong TMEM half
  - fix:
    - keep the new TMEM reinterpret query builder in `TensorMemoryUtils.cpp`, but disable the old column-subview support-rescue path for `64`-row TMEM column subviews so they fall back to the raw TMEM view query instead of the lossy reshaped-support approximation
    - tighten `gluon_ir.cc` auto layout selection for root `64xN` TMEM descriptors so `get_reg_layout()` no longer gets pinned to the old legacy-only shortcut before the direct TMEM-compatible layout search runs
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -s --tb=short -x 'python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[legacy]'` -> `1 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -s --tb=short -x 'python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[linear]'` -> `1 passed`
    - paired rerun -> `2 passed`


- 2026-04-01: fixed the remaining GB200 stale-test fallout and TMEM descriptor-chain alias bug before the full suite rerun
  - failures/fallout:
    - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[...]` still miscompiled the double-buffer alias chain after the earlier GB200 checkpoint because standalone ld/st query-origin preservation copied logical prefix offsets instead of remapping through the physical TMEM coordinates of the translated support view
    - `MemDescReinterpretOp` still validated non-TMEM reinterpret bitcounts against backing `allocShape`, which rejected valid contiguous slice reinterprets in `test_slice_reinterpret`
    - several `test_core.py` / runtime-matrix / FPSAN checks were stale after the tighter TMEM fallback planning and broader verifier diagnostics
  - fix:
    - added a physical-coordinate remap helper in `TensorMemoryUtils.cpp` and switched translated support-query origin preservation to remap through the destination layout's left inverse instead of copying logical origins dimension-by-dimension
    - changed non-TMEM `ttg.memdesc_reinterpret` bitcount validation in `Ops.cpp` to compare the visible source/destination view shapes rather than backing allocShapes
    - updated Gluon TMEM descriptor-chain tests to use explicit `dim=0` when slicing the outer multibuffer dimension
    - refreshed the M64 fallback opcode expectations to the exact `16x32bx2` packets emitted by the current direct planner, removed the last blocked-layout test's dependence on implicit repair, and widened stale FPSAN error-string checks to the current clean verifier diagnostics
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. python3 -m pytest -s --tb=short -x python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_identity_32x32b_4w-layout10-128-128-32x32b-4-32x32b]` -> `1 passed`
    - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. python3 -m pytest -s --tb=short -x python/test/gluon/test_core.py::test_slice_reinterpret python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix` -> `27 passed`
    - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. python3 -m pytest -s --tb=short -x python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes python/test/gluon/test_core.py::test_tmem_linear_m64_roundtrip_32x32b_fallback` -> `17 passed`
    - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. python3 -m pytest -s --tb=short -x python/test/gluon/test_core.py::test_tmem_linear_roundtrip_blocked_fallback` -> `1 passed`
    - targeted FPSAN reruns for the updated unsupported-layout diagnostics -> all passing


- 2026-04-01: fixed the remaining TMEM split-N descriptor-view store/load mismatch before the GB200 suite rerun
  - failure:
    - `python/test/gluon/test_fpsan.py::test_tmem_index_subslice` still failed after the row-plan/query fix even though the offending `64x32` TMEM subview was finally using `rowPlan=64`
    - traced lowering showed the same `ttg.memdesc_subslice` view lowering to different packet families on the two sides:
      - store: `tcgen05.st.sync.aligned.32x32b.x32.b32`
      - load: `tcgen05.ld.sync.aligned.16x32bx2.x8.b32` twice
    - the user-level explicit `sub.get_reg_layout("32x32b_splitn")` was being stripped on the store side by the generic `tmem_store(convert_layout) -> tmem_store` canonicalization, so the store silently fell back to the blocked source layout while the load kept the split-N linear layout
  - fix:
    - tighten `CanonicalizeConvertFromTMEMStore` in `lib/Dialect/TritonGPU/IR/Ops.cpp` so it does not erase explicit `convert_layout` ops when the TMEM destination is a view-like memdesc (`memdesc_subslice/index/reshape/reinterpret/trans` or internal `ttng.tmem_subslice`)
    - this preserves the explicit direct TMEM register layout on descriptor views and keeps store/load on the same packet family for the split-N subview case
    - separately updated `test/Conversion/relayout_tritongpu.mlir` to the new preferred 64x64 TMEM store layout after comparing current output against `/tmp/triton-origin-main`; the test intent stayed the same (direct TMEM-compatible relayout), but the register/lane split moved to the new leaf-64 preference
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. python3 -m pytest -s --tb=short -x python/test/gluon/test_fpsan.py::test_tmem_index_subslice` -> `1 passed`
    - full lit: `248 passed, 2 unsupported`
    - focused FPSAN smoke: `python/test/gluon/test_fpsan.py -k 'tmem_index_subslice or tmem_reduction'` -> `1 passed, 85 deselected`

- 2026-04-01: fixed the GB200 MMAv5 shared-input `64x32xf32` accumulator readback regression
  - failure:
    - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-64-128-warps2-16-False-True-acc_dtype4]` regressed relative to `/tmp/triton-origin-main`
    - origin lowered the final `ttng.tmem_load %acc_tmem` on the `64x32xf32` legacy accumulator through the canonical M64 split readback path (`tcgen05.ld.sync.aligned.16x32bx2.x16.b32`)
    - current head instead typed the same load as a scalar `32x32b.x1` layout and emitted 32 separate `tcgen05.ld.sync.aligned.32x32b.x1.b32` packets, producing large numerical errors in the MMA shared-input kernel
  - root cause:
    - the type-only TMEM layout search was already selecting the canonical M64 layout, but the handle-aware `compute_tmem_reg_layout_from_memdesc(...)` path revalidated candidates with `getTMemLdStRowPlanForQuery(...)`
    - for plain legacy `blockM=64` leaves, `getTMemLdStRowPlanForType(...)` still classified the raw zero-row-basis legacy form from its 7 raw row bits and returned the widened 128-row plan
    - that row plan made the handle-aware path prefer the raw generic `32x32b.x1` candidate over the canonical stripped M64 layout even though the descriptor type itself reported the correct `64x32` register layout
  - fix:
    - in `TensorMemoryUtils.cpp`, derive the TMEM ld/st row plan from the active row bases for all logical `M=64` layouts with `activeRowBits == 6`, not only the split-N special case
    - in `Dialect.cpp`, keep the stripped canonical M64 selection ahead of the raw generic planner and validate stripped canonical candidates against the stripped query layout overload of `computeTMemLdStEncodingInfo(...)`
    - this makes the handle-aware reg-layout query agree with the type-only TMEM-compatible layout search again for plain `64x32` / `64x64` MMA accumulators
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. python3 -m pytest -s --tb=short -x 'python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-64-128-warps2-16-False-True-acc_dtype4]'` -> `1 passed`

- 2026-04-01: fixed the remaining GB200 Blackwell `TRITON_PREFER_TMEM_16x256_LAYOUT=1` matmul regressions in the physical-support ld/st path
  - failures:
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-512-64-32-2-float16-float16]`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-512-64-32-2-float16-float8e5]`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-8-1-256-128-32-4-float32-tensorfloat32]`
    - all three passed on `/tmp/triton-origin-main` and failed locally only because PTX missed the expected native `16x256b` ld/st path
  - root cause:
    - the current TMEM optimizer was legitimately lowering these tall/narrow accumulators through `getTMemLdStPhysicalSupportPlan(...)`, i.e. `512x64 -> reinterpret 128x256` and `256x128 -> reinterpret 128x256`
    - inside that support-plan chooser, atom probing still ran in `I32x32b, I16x256b, I16x128b, I16x64b` order, so the support path succeeded on `32x32b.x128` before it ever considered `I16x256b`
    - as a result, `TRITON_PREFER_TMEM_16x256_LAYOUT=1` had no effect once a store/load went through the physical-support rewrite
  - fix:
    - keep the corrected legacy `M64/I16x256b` basis in `Dialect.cpp`
    - in `TensorMemoryUtils.cpp`, honor `TRITON_PREFER_TMEM_16x256_LAYOUT` inside `getTMemLdStPhysicalSupportPlan(...)` by probing support atoms in `I16x256b, I32x32b, I16x128b, I16x64b` order
    - this preserves the generalized physical-support rewrite while restoring the native `16x256b` PTX family for exact Blackwell matmul cases that want it
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:python/test/unit/language:. TRITON_PREFER_TMEM_16x256_LAYOUT=1 python3 -m pytest -s --tb=short -x \
      'python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-512-64-32-2-float16-float16]' \
      'python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-512-64-32-2-float16-float8e5]' \
      'python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-8-1-256-128-32-4-float32-tensorfloat32]'`
      -> `3 passed`

- 2026-04-02: restored the GB200 preferred-`16x256b` exact-family `64x128xf16` path and updated the stale compile-only PTX check
  - failures:
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-64-128-32-4-float16-float16]` failed locally with `TRITON_PREFER_TMEM_16x256_LAYOUT=1` because PTX no longer contained `16x256b`
    - the compile-only smoke `python/test/unit/language/test_compile_only.py::test_compile_only_dot` also failed, but only because its PTX regex still required the older exact `x16|x32` packet counts
  - before/after comparison:
    - on `/tmp/triton-origin-main`, the `64x128xf16` node still lowered the accumulator through legacy `#ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 2>` and emitted `tcgen05.{st,ld}.sync.aligned.16x256b...`
    - on the current tree before the fix, the same node selected canonical `#ttng.tensor_memory_linear`, produced a different register layout, and missed the `16x256b` path entirely
  - root cause:
    - in `Dialect.cpp`, `getDefaultLayoutForTmemLdSt(...)` still probed the normalized canonical M64 view before the exact-family legacy-anchored selector, so full exact tiles no longer matched the origin/main register layout under `TRITON_PREFER_TMEM_16x256_LAYOUT`
    - inside `getDistributedLayoutForTmemLdStLegacyAnchored(...)`, `layout16Rows` had been widened to depend on the logical row size (`<= 16`) instead of the actual zero row basis at `row=16`; for legacy `M64` leaves this added a dead register basis and turned `ttng.tmem_load` into an unsupported broadcasted TMEM view
    - the compile-only PTX check was simply stale: both origin/main and current emit the same ordered tcgen alloc/store/mma/commit/wait/load sequence, but packet counts may now be `x64`
  - fixes:
    - in `Dialect.cpp`, try the exact-family legacy-anchored preferred selector before the normalized canonical M64 fallback for full-shape TMEM leaves
    - in `getDistributedLayoutForTmemLdStLegacyAnchored(...)`, restore the legacy `layout16Rows` detection based on the zero basis at `row=16` (while still allowing true `<=16` row tiles)
    - in `python/test/unit/language/test_compile_only.py`, replace the brittle single PTX regex with ordered sequential checks and accept `x16|x32|x64` on the direct ld/st packets
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=0 python3 -m pytest -s --tb=short -x 'python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-64-128-32-4-float16-float16]'` -> `1 passed`
    - `CUDA_VISIBLE_DEVICES=0 python3 -m pytest -s --tb=short -x 'python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-64-128-32-4-float32-tensorfloat32]'` -> `1 passed`
    - `CUDA_VISIBLE_DEVICES=0 python3 -m pytest -s --tb=short -x 'python/test/unit/language/test_compile_only.py::test_compile_only_dot'` -> `1 passed`

- 2026-04-01: fixed the GB200 persistent-attention warp-specialization TMEM OOR regression
  - failing node:
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_persistent_forward[False-8-False-2-128-128-8192-8192]`
    - current tree required `262336` bytes of shared memory and failed launch metadata init with `OutOfResources`; `/tmp/triton-origin-main` required `230076` bytes and passed
  - root cause:
    - the first divergence from origin/main appeared in `triton-nvidia-optimize-tmem-layouts`, not in LLVM lowering
    - `TMemLoadReducePattern` retuned the loop-carried `ttng.tmem_load` accumulator tile from `tensor<128x128xf32, #linear>` to `tensor<128x128xf32, #linear1>` because it saw a reduction along `N`
    - that load value is later written back to TMEM in the persistent attention loop, so the reduction-friendly relayout was not a pure win; it forced the softmax/update path onto `#linear1`, introduced `#linear1 -> #linear` store conversions, and ultimately lowered through the larger shared-memory scratch path
    - a follow-on canonicalizer fold (`tmem_store(cvt) -> tmem_store`) was masking part of the path, but the real regression started earlier in `OptimizeTMemLayouts`
  - fixes:
    - in `OptimizeTMemLayouts.cpp`, make `TMemLoadReducePattern` bail out when the forward slice from the `ttng.tmem_load` feeds a later `ttng.tmem_store`; keep the reduction-friendly relayout only for true reduction-only consumers
    - in `Dialect.cpp`, make `getDefaultLayoutForTmemLdSt(...)` prefer the exact-family legacy-anchored full-tile layout for exact MMAv5 TMEM leaves even without the `TRITON_PREFER_TMEM_16x256_LAYOUT` env knob
    - in `TritonGPU/IR/Ops.cpp`, keep an explicit `convert_layout` feeding `ttng.tmem_store` when it targets the preferred TMEM store layout instead of erasing it solely because the pre-convert layout is also compatible
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - warmup repro: shared memory dropped from `262336` to `230592` bytes, and the bad `#linear1` accumulator load/store path disappeared from TTGIR
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -s --tb=short -x 'python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_persistent_forward[False-8-False-2-128-128-8192-8192]'` -> `1 passed`


- 2026-04-06: fixed the packed sparse `I16x32bx2` half-row TMEM subslice bug on the direct ld/st support-query path
  - failing repros:
    - reduced repro `/tmp/repro_zero_only.py` (`64x128xf32` linear TMEM tile, zeroing only the sliced half-row view) mismatched `2048` elements
    - `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[linear]` failed
  - narrowed root cause:
    - the reinterpret-based packed sparse support helper in `TensorMemoryUtils.cpp` recovered the right `I16x32bx2` packet family, but its external warp anchors were still wrong for views whose packed TMEM support layout carries a hidden zero `row=16` basis
    - using the stripped support layout for anchor selection produced `(16, 32)` row anchors and broke the `warp1/warp3` halves; using the raw unstripped support layout produced `(0, 16)` and broke the back half of the tile instead
    - PTX/repro mapping showed the packed support path itself was otherwise correct: after the anchor-only experiments, the remaining mismatches were confined to the row blocks selected by `warpBaseOffset0/1`
  - fix:
    - keep anchor discovery on the raw packed support layout so the hidden zero row basis is visible
    - when the first recovered anchor basis is zero but the next one is non-zero, skip that hidden half-row basis for the second external anchor and lift it to the next non-zero logical row bit (`warpRow1 * 2`)
    - this yields the correct packed sparse anchor pair `(0, 32)` rows for the `64x64` support-query `I16x32bx2` path
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `PYTHONPATH=python:. TRITON_CACHE_DIR=/tmp/triton_cache_packed_sparse_i16_anchorbasis2 python3 /tmp/repro_zero_only_map.py` -> `mismatch 0`
    - `PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[linear]` -> `1 passed`

- 2026-04-06: Bounded the remaining `block_m_64` reinterpret/store wrong-code bucket further. Disabling the `getDistributedLayoutForTmemLdSt(...)` split-N fast path for the 16->32 unpacked recursion did not change either the explicit `p_tmem.get_reg_layout(instr_variant="32x32b")` rejection or the runtime `mismatch 2048` pattern in `/tmp/repro_zero_only_map_p_layout.py`. Fresh debug shows the stronger root cause: the auto-layout path picks a valid public reg layout `#ttg.linear<{register=[[0,1],[0,2],[0,4],[0,8],[0,16],[0,32]], lane=[[1,0],[2,0],[4,0],[8,0],[0,64]], warp=[[16,0],[32,0]], block=[]}>`, but `computeTMemLdStEncodingInfo(...)` still classifies that layout as atom=4 (`I16x32bx2`) for the sparse reinterpret support view instead of `I32x32b`. So the remaining bug is in `TensorMemoryUtils.cpp` atom selection / packed-support lowering, not just in `Dialect.cpp` family construction.

- 2026-04-06: aligned legacy `blockM=64` / expanded-`N` TMEM sugar with the row-zero-lift support form used by direct ld/st reinterpret lowering
  - symptom:
    - legacy `TensorMemoryLayout((64, 64), col_stride=1)` allocs for shape `64x128` still misaligned/faulted in the `p_tmem` reinterpret/store bucket even after the packed sparse support fixes
    - minimal repro `/tmp/repro_block_m64_p_zero_only_legacy.py` faulted with `CUDA misaligned address`, while the equivalent explicit linear layout passed
  - root cause:
    - the legacy canonicalizer in `Dialect.cpp` still expanded `blockM=64` / `shape[1] > blockN` through the old interleaved basis `row=16 -> (0, blockN)`
    - direct ld/st reinterpret support rescue in `TensorMemoryUtils.cpp` already normalizes the same user-visible view to the row-zero-lift form (`row=16 -> (0, 0)` with the extra `col` bit)
    - allocator/runtime therefore reserved/anchored the backing TMEM as `nRow=128, nCol=64` (`tcgen05.alloc ... 64`) while the actual direct store lowering addressed it as the row-zero-lift `64x128` support form
  - fix:
    - changed `buildCanonicalLegacyLikeTMemLinearLayout(...)` so non-twoCTA `blockM=64` legacy sugar keeps `N` expansion in the column dimension instead of reusing the hidden `row=16` basis
    - this makes legacy sugar canonicalize to the same full linear form already used by the direct ld/st support-query rescue path
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=python:. python3 /tmp/repro_block_m64_p_zero_only_legacy.py` -> `ok`
    - `CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[legacy]` -> `1 passed`
    - `make test-lit` -> `248 passed, 2 unsupported`
    - `make test-cpp` -> `240/240 passed`

- 2026-04-06: fixed the GB200 warp-specialization TMEM OOR regression after the linear-generalization sweep
  - failing repro:
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. python3 -m pytest -s --tb=short --maxfail=1 "python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-False-2-64-128-8192-8192]"` failed with `OutOfResources: tensor memory, Required: 640, Hardware limit: 512`
    - the same exact nodeid passed on `origin/main`
  - root cause:
    - `getTmemAllocSizes(...)` in `Dialect.cpp` was overriding the physical TMEM column count for canonical linear encodings with the widest MMAv5 family planner result (`instrShapeN`)
    - that planner result is correct for instruction-family matching but incorrect for allocation-size accounting; equivalent legacy-sugar and canonical-linear TMEM allocs therefore got different `ttg.tensor_memory_size` values
    - the exact attention-forward TTGIR matched `origin/main` structurally (`128x64xf32` and `128x128xf16/f32` allocs) and differed only in legacy sugar vs canonical `#ttng.tensor_memory_linear`, confirming the regression was accounting-only
  - fix:
    - `getTmemAllocSizes(...)` now uses the physical linear-layout row/column extents directly, matching the old legacy-sugar behavior for storage accounting
    - kept the widened MMAv5 family planning for verifier/codegen selection, but removed it from TMEM allocation metadata sizing
    - also restored `AccelerateMatmul.cpp` to use `getDefaultLayoutForTmemLdSt(...)` for default MMAv5 accumulator layouts so GB200 unit matmul coverage follows the same direct-layout policy as `origin/main`
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - exact repro above -> `1 passed`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-8-2-64-128-32-4-float32-tensorfloat32]` and `...[True-False-4-2-128-128-16-4-float16-float16]` no longer regress on the current tree

- 2026-04-07: fixed the remaining `linear_m64_32x32b_8w` TMEM descriptor-chain wrong-code on the packed `16x32bx2.x32` direct path
  - failing repro before the fix:
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short --maxfail=1 'python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_8w-layout5-64-128-32x32b-8-16x32bx2]'`
    - first failed as `misaligned address`; after recovering the raw packed query path it still failed numerically with `4096/8192` mismatches, then `2048/8192` after removing a lowering-side warp-group col squash
  - root cause:
    - the MLIR-side direct `I16x32bx2.x32` plan for the row-zero `64x128xf32` reinterpret already carried the required recovered warp-group TMEM band bit (`warp=4 -> col 32`)
    - `lowerTMemLdSt(...)` in `TensorMemoryToLLVM.cpp` was explicitly zeroing that recovered higher-warp col contribution for the packed path, and the later `secondHalfOffset: 64 -> 32` special cases in `TensorMemoryUtils.cpp` were compensating for that older broken lowering
    - once the warp-group col contribution is preserved, the natural packed `secondHalfOffset=64` is the correct direct lowering again
  - fix:
    - removed the lowering-side overcorrection that zeroed the recovered `warp=4 -> col 32` contribution in `lowerTMemLdSt(...)`
    - removed the `>4 warps` row-zero M64 packed-support `secondHalfOffset` halving / `64 -> 32` rewrites in `TensorMemoryUtils.cpp`
    - kept the raw query-layout recovery and raw `rowPlan=64` selection for the reinterpret path; that remains necessary to stay on the packed `16x32bx2.x32` direct path instead of regressing to scalar `32x32b.x1`
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - exact isolated repro above -> `1 passed`
    - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -k 'tmem_descriptor_chain_matrix' python/test/gluon/test_core.py` -> `26 passed`

- 2026-04-07: narrowed the remaining Blackwell MMAv5 root-accumulator wrong-code fix to raw lowering only
  - failures after the stale-lit / GB200 rerun were no longer in generic user-visible TMEM descriptor ld/st; the remaining repros were internal MMAv5 accumulator leaves in
    - `python/test/unit/cuda/test_tma_store_gemm.py::test_tma_load_store[64-128-32-1-4-False-True-False]`
    - `python/test/unit/language/test_block_pointer.py::test_block_ptr_matmul_no_scf[shape3-8]`
    - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-64-128-warps2-16-False-True-acc_dtype4]`
  - fresh `TRITON_DEBUG_TMEM_QUERY=1` / `TRITON_DEBUG_TMEM_HALFROWS=1` traces showed those kernels all lower raw root `ttng.tmem_alloc` row-zero `M=64` TMEM accumulators through `atom=I16x32bx2` with the active-row `rowPlan=64` (`warpRow0=16`, `warpRow1=32`)
  - that active-row plan is still correct for user-facing reg-layout selection and descriptor-view ld/st, but it does not match the physical TMEM row anchoring MMAv5 uses for the backing accumulator tile
  - origin/current LLIR comparison on the TMA repro confirmed the divergence:
    - origin: store/load TMEM base uses the widened row anchors (`shl 21`, mask `6291456`)
    - current failing path: store/load TMEM base used the narrower active-row anchors (`shl 20`, mask `3145728`)
  - broadening the backing row-plan helper to always use the widened storage view fixed the GB200 unit repros but regressed the plain `linear_m64_*` descriptor-chain matrix, so that approach was reverted
  - final fix:
    - keep `getTMemLdStRowPlanForType(...)` and the generic query/backing helpers unchanged
    - in `TensorMemoryToLLVM.cpp`, detect only the raw root `ttng.tmem_alloc` row-zero `64xNxf32` accumulator leaves (`hasZeroBasisAlong(row) && !hasZeroBasisAlong(col)`) and override the raw lowering row plan to `{warpRow0=32, warpRow1=64, rowSpan=128}`
    - leave descriptor-view/query selection on the active-row `64`-row path
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `python/test/unit/cuda/test_tma_store_gemm.py::test_tma_load_store[64-128-32-1-4-False-True-False]` -> `1 passed`
    - `python/test/unit/language/test_block_pointer.py::test_block_ptr_matmul_no_scf[shape3-8]` -> `1 passed`
    - `python/test/gluon/test_core.py -k 'tmem_descriptor_chain_matrix'` -> `26 passed`
    - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-64-128-warps2-16-False-True-acc_dtype4]` -> `1 passed`


- 2026-04-07: finalized the root `64xNxf32` MMAv5 accumulator ld/st fix for the Blackwell `16x256b` preference path
  - after the broader GB200-equivalent `python/test/unit` rerun, the remaining reproducible wrong-code bucket collapsed to `python/test/unit/language/test_matmul.py::test_simple_matmul[...]` cases with
    - `M=64`
    - `out_dtype=float32`
    - Blackwell MMAv5 lowering
    - `TRITON_PREFER_TMEM_16x256_LAYOUT=1` or a nearby root-accumulator layout family
  - root cause:
    - the remaining bad cases were not MMAv5 instruction selection failures; they were generic `ttng.tmem_store` / `ttng.tmem_load` around a root `ttng.tmem_alloc` accumulator after MMA
    - the preferred `16x256b` root accumulator layout is a lifted sparse / zero-row-basis `64xNxf32` TMEM view
    - raw lowering must still use the widened lifted anchor family for that backing tile, but the stable widened anchors are `warpRow0=16`, `warpRow1=32`, `rowSpan=128`, not the earlier over-broad `32/64 @ 128` experiment
    - `TensorMemoryToLLVM.cpp` therefore needs to widen only raw root `ttng.tmem_alloc` accumulators that are actually used by `tcgen05_mma` / `tcgen05.mma_scaled`, and `TensorMemoryUtils.cpp` must explicitly accept that lifted `16/32 @ 128` override for the zero-row-basis `64xNxf32` accumulator family
    - the `I32x32b` row-zero M64 base-offset halving special-case also has to stay scoped to the real `I32x32b` path, otherwise the recovered packed `16x256b` / `16x32bx2` families are overcorrected again
  - fix:
    - in `lowerTMemLdStFromTypes(...)`, replace the older row-zero-root heuristic with a narrower MMAv5-accumulator check:
      - raw root `ttng.tmem_alloc`
      - rank-2
      - element bitwidth `32`
      - logical `M=64`
      - actual `TCGen5MMAOp` / `TCGen5MMAScaledOp` users
    - thread the widened raw/query row-plan override as `{warpRow0=16, warpRow1=32, rowSpan=128}`
    - in `computeTMemLdStEncodingInfoImpl(...)`, allow exactly that lifted override for `64xNxf32` zero-row-basis accumulator layouts even though the nominal input row span is still `64`
    - keep the `isI32RowZeroM64DirectView` warp-base halving rewrite gated on `atom == I32x32b`
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-2-64-512-32-2-float32-tensorfloat32]` -> `1 passed`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-2-64-512-32-2-float32-tensorfloat32]` -> `1 passed`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-2-64-128-32-4-float32-tensorfloat32]` -> `1 passed`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-2-64-128-32-4-float32-tensorfloat32]` -> `1 passed`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-2-64-512-32-2-float16-float16]` -> `1 passed`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-2-64-512-32-2-float16-float16]` -> `1 passed`


- 2026-04-07: split the raw root `M=64` MMAv5 accumulator override by actual memdesc family
  - the first committed `16/32 @ 128` root-accumulator fix was still too coarse: it fixed the `python/test/unit/language/test_matmul.py::test_simple_matmul[...]` bucket, but a full `python/test/unit` split rerun immediately re-broke
    - `python/test/unit/cuda/test_tma_store_gemm.py::test_tma_load_store[...]`
    - `python/test/unit/language/test_block_pointer.py::test_block_ptr_matmul_no_scf[...]`
  - fresh `TRITON_DEBUG_TMEM_QUERY=1` traces showed the missing distinction:
    - TMA / block-pointer repros lower root MMAv5 accumulators as explicit zero-row-basis linear memdescs like `!ttg.memdesc<64x64xf32, #ttng.tensor_memory_linear<...>>`
    - the `test_simple_matmul` repro lowers its root accumulator through legacy `#ttng.tensor_memory_encoding<blockM=64, blockN=256, ...>` sugar (or the `16x256b`-preferred variant), not the explicit zero-row-basis linear family
    - both still hit raw `atom=4` / packed root ld/st, so atom selection alone is not enough to choose the right row anchors
  - stable split:
    - legacy `TensorMemoryEncodingAttr` root MMAv5 accumulators keep the `16/32 @ 128` widened raw row plan
    - explicit zero-row-basis linear root MMAv5 accumulators use the older `32/64 @ 128` widened raw row plan
    - `TensorMemoryUtils.cpp` now accepts both widened override families for the lifted zero-row-basis `64xNxf32` accumulator case
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `python/test/unit/cuda/test_tma_store_gemm.py::test_tma_load_store[64-64-16-1-4-False-True-False]` -> `1 passed`
    - `python/test/unit/language/test_block_pointer.py::test_block_ptr_matmul_no_scf[shape3-8]` -> `1 passed`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-2-64-512-32-2-float32-tensorfloat32]` -> `1 passed`
    - `TRITON_PREFER_TMEM_16x256_LAYOUT=1 python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-2-64-512-32-2-float32-tensorfloat32]` -> `1 passed`


- 2026-04-07: fix TMEM scales `mxfp` regression without re-breaking descriptor ld/st
  - GB200-equivalent `python/test/unit` split shard 18 exposed a fresh regression in
    - `python/test/unit/language/test_matmul.py::test_mxfp[0-4-1-128-16-256]`
    - `python/test/unit/language/test_matmul.py::test_mxfp[0-4-3-128-16-256]`
  - symptom:
    - `ConvertTritonGPUToLLVM` aborted in `ColumnAction::apply(ValueRange)`
    - reproducer stack pinned the crash to `TensorMemoryToLLVM.cpp:707` while lowering `ttng.tmem_alloc` for `#ttng.tensor_memory_scales_encoding<>`
  - root cause:
    - the broad fallback fix kept the correct raw linear register layout for TMEM ld/st, but `computeTMemLdStEncodingInfoImpl(...)` still eagerly stripped broadcasted register bits from `regLayout`
    - that is safe for some direct ld/st queries, but it is wrong for TMEM scales alloc lowering because `lowerTMemLdSt(...)` needs to see the original broadcasted register structure and record it in `info.broadcast`
    - once the broadcasted bits were erased too early, `info.perm` still expected the reduced register domain while `TMEMAllocOpConversion` passed the full unpacked source register list, tripping the `ColumnAction::apply` size assert
  - fix:
    - keep the raw linear-register query switch in `computeTMemLdStEncodingInfoImpl(...)`
    - remove the local `actionRemoveBroadcastedRegs(regLayout).apply(regLayout)` pre-pass there
    - let `lowerTMemLdSt(...)` own broadcast handling again via `info.broadcast`
  - validation:
    - `TRITON_BUILD_WITH_CCACHE=true make -j96`
    - `/tmp/fpsan_store_only_probe.py` -> `ok`
    - `/tmp/fpsan_load_only_probe.py` -> `ok`
    - `python/test/gluon/test_fpsan.py::test_tmem_index_subslice` -> `1 passed`
    - `python/test/gluon/test_frontend.py::test_tensor_memory_linear_view_load_reports_clean_error` -> `1 passed`
    - `python/test/unit/language/test_matmul.py::test_mxfp[0-4-1-128-16-256]` -> `1 passed`
    - `python/test/unit/language/test_matmul.py::test_mxfp[0-4-3-128-16-256]` -> `1 passed`


## 2026-04-07 GB200 CI-equivalent sweep blocker
- Commit under test: `b639073a7` (`Fix TMEM query regressions from GB200 sweep`).
- Validated before GPU-driver failure: `check-triton-lit-tests` green (`248 passed, 2 unsupported`), `check-triton-unit-tests` green (`240/240 passed`), `test-unit` equivalent green including `python/test/unit`, `test_debug.py`, `python/triton_kernels/tests/`, fused attention tutorial, instrumentation, and plugin tests.
- `test-gluon` split sweep progressed cleanly through groups `1-16/32`.
- First failure at `gluon-main` group `17/32`: `python/test/gluon/test_core.py::test_mma_shared_inputs[...]` failed during `torch.randn(..., device='cuda')` setup with `RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable`.
- Isolated rerun of group `17/32` on `CUDA_VISIBLE_DEVICES=0` reproduced mass `device busy or unavailable` failures before Triton kernel execution.
- Isolated single-test rerun of representative `test_mma_shared_inputs[...]` on `CUDA_VISIBLE_DEVICES=1` entered uninterruptible sleep (`STAT=D`) instead of failing at the same point.
- Concurrent minimal Torch CUDA probe also hung.
- `dmesg` at the same time shows repeated NVRM/NVLink driver errors: `knvlinkUpdatePostRxDetectLinkMask_IMPL: Failed to update Rx Detect Link mask!` and `knvlinkDiscoverPostRxDetLinks_GH100: Getting peer0's postRxDetLinkMask failed!`.
- Conclusion: current blocker is machine/driver state, not a new confirmed Triton regression in TMEM codegen. Resume remaining GB200-equivalent GPU targets only after GPU/NVLink health is restored.

- After killing the stray probe and wedged pytest processes, `timeout 20s nvidia-smi --query-gpu=...` still timed out. GPU control path remained unhealthy, so the GB200-equivalent GPU sweep could not be completed further in this session.

## 2026-04-09
- Landed the first bounded cleanup slice for the `M=64` MMAv5 root-accumulator
  hack and checkpointed it in modular remote-pushed commits.
- Repo/operator guidance updates:
  - added `AGENTS.md` guidance for unstable-machine recovery, detailed
    handoff-grade commit messages, and pushing each TMEM commit to
    `jeffniu-openai/codex/tmem`
  - documented the standard 4-GPU pytest sweep workflow there
  - added `pytest-split` to `python/test-requirements.txt`
- Compiler contract cleanup:
  - replaced the lowering-side
    `preferredRootRowPlanForM64AccumulatorAlloc` heuristic in
    `TensorMemoryToLLVM.cpp`
  - added `ttng.tmem_ldst_row_plan` helpers in `TensorMemoryUtils`
  - `getBackingTMemLdStRowPlan(...)` now honors an explicit root row-plan
    contract before type-derived inference
  - `AccelerateMatmul` attaches the explicit contract to MMAv5 accumulator root
    allocs
  - MMAv5 software pipelining and Hopper warp-specialization rematerialized
    allocs now copy the explicit contract
  - manual Gluon `tcgen05_mma` / `tcgen05_mma_scaled` builders now walk
    accumulator subviews back to the backing `ttng.tmem_alloc` and attach the
    same explicit contract there
  - shared helper `getMMAv5AccumulatorRootRowPlan(...)` now owns the actual
    `16/32 @ 128` vs `32/64 @ 128` family split
- Build environment note:
  - full `make` initially reproduced the unrelated GSan runtime failure
    (`GSanLibrary.cu`: `fatal error: 'climits' file not found`)
  - rerunning with
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13`
    unblocked the GSan step and allowed `make -j8` to complete on this node
- Validation:
  - `TRITON_HOME=/tmp CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
  - manual FileCheck-equivalent runs:
    - `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt /root/code/triton/test/TritonGPU/loop-pipeline-blackwell.mlir -split-input-file -tritongpu-hoist-tmem-alloc -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline -triton-nvidia-gpu-remove-tmem-tokens -canonicalize | /tmp/.triton/llvm/llvm-7f77ca0d-ubuntu-arm64/bin/FileCheck /root/code/triton/test/TritonGPU/loop-pipeline-blackwell.mlir --check-prefixes=CHECK`
    - `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt /root/code/triton/test/TritonNvidiaGPU/mma_lowering.mlir -split-input-file --triton-nvidia-mma-lowering | /tmp/.triton/llvm/llvm-7f77ca0d-ubuntu-arm64/bin/FileCheck /root/code/triton/test/TritonNvidiaGPU/mma_lowering.mlir`
  - targeted GPU pytest slice on `CUDA_VISIBLE_DEVICES=0` with isolated cache:
    - `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`
    - `python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[True-4-64-128-32]`
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-128-32]`
    - all four passed after the Gluon/manual-path backing-alloc annotation fix
- Remaining hack backlog after this slice:
  - direct `32x32` support rescue / scalarization paths
  - `64x128xf32` reinterpret rescue
  - packed row-zero-lifted `TMemLdStEncodingInfo` fixups
  - `warpx2` family-specific planner cleanup

## 2026-04-09 managed-session handoff
- Resumed the older Codex session in the managed `~/code/triton` checkout only to
  extract handoff context; no further implementation should continue there because
  `brix-agent` can reset that checkout back to the provisioned baseline branch.
- That interrupted session reported that its in-progress isolated-worktree slice
  was specifically aimed at collapsing bespoke TMEM support hacks into one
  support-query contract and then validating the cleanup in
  `/root/code/triton-tmem-isolated`.
- Hack backlog it was still targeting relative to the post-merge-base cleanup:
  - direct `32x32` support rescue / scalarization
  - `64x128xf32` reinterpret rescue
  - packed row-zero-lifted `TMemLdStEncodingInfo` fixups
  - `warpx2` family-specific planner cleanup
- The managed-session handoff also said the uncommitted isolated-worktree patch
  was incomplete at the moment of interruption: stale Gluon-side support-plan
  plumbing remained and a malformed debug-string block made that snapshot
  non-buildable until repaired.
- Validation/build state reported by that session before interruption:
  - already-green bounded checkpoint before this slice:
    - `make` with `TRITON_HOME=/tmp` and
      `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13`
    - targeted `triton-opt` checks for
      `test/TritonGPU/loop-pipeline-blackwell.mlir` and
      `test/TritonNvidiaGPU/mma_lowering.mlir`
    - GPU pytests:
      - `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`
      - `python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
      - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[True-4-64-128-32]`
      - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-128-32]`
  - the newer isolated-worktree support-query cleanup itself had not yet been
    built or tested in that session and should have been assumed non-buildable
    until the stale Gluon callsite and malformed debug strings were fixed.

## 2026-04-09 resumed managed-session refresh
- Re-resumed the paused managed-checkout Codex session and asked only for a
  status handoff; then stopped interacting with it again.
- The managed session's remembered hack inventory still matches the current
  initiative backlog:
  - direct `32x32` support rescue / scalarization
  - `64x128xf32` reinterpret rescue
  - packed row-zero-lifted `TMemLdStEncodingInfo` fixups
  - `warpx2` family-specific planner cleanup
- That session confirmed its last known validated checkpoint was still the
  already-landed `ttng.tmem_ldst_row_plan` cleanup slice and its associated
  targeted build / lit / GPU pytest coverage.
- The isolated worktree has since moved beyond the specific malformed-debug-
  string snapshot remembered by that session. The active local work now needs
  to finish reconciling the support-query / subview cleanup with the broader
  initiative goal: keep all physically codegen-legal TMEM layouts positive,
  use clean negatives only for impossible ISA cases, and choose among multiple
  legal codegen paths with performance-oriented heuristics rather than
  hard-coded exclusions.

## 2026-04-09 half-row ld/st probe refresh: still a planner/decomposition BUG, not a new negative boundary
- Re-ran the remaining lifted half-row ld/st wrong-code bucket from the
  isolated worktree after repairing the earlier malformed-debug-string state
  and keeping the broader support-query cleanup changes buildable again.
- Reconfirmed that the bug is not limited to the lifted higher-rank
  `reshape((2, M/2, N)).slice(1, 1, dim=0).index(0)` chain:
  - a direct `128x64 -> slice(M/2, M/2, dim=0)` TMEM row-half view is also
    wrong under the current direct ld/st planner
  - a standalone root `64x64` TMEM tile with the same `16x128b` family still
    executes correctly
- The new concrete evidence is that the current direct packet families map the
  row-half view to alternating 16-row bands instead of the contiguous lower
  half:
  - `32x32b`, `16x64b`, `16x256b`, and the current auto-selected `16x32bx2`
    path all hit rows `0-15,32-47,64-79,96-111`
  - `16x128b` hits rows `16-31,48-63,80-95,112-127`
  - expected semantics remain the contiguous lower half `64-127`
- Interpretation:
  - this bucket still looks physically codegen-legal in principle because the
    root `64x64` family itself is valid; the current lowering is simply
    choosing the wrong TMEM packet decomposition for views carved from a larger
    backing tile
  - therefore this is still an open BUG in planner/decomposition semantics, not
    evidence for introducing a new permanent clean-negative verifier boundary
- Bounded experiments retried and reverted in this session because they did not
  restore correct semantics:
  - raw-query rebasing with direct-view layouts plus lifted origins
  - widened backing-tile support queries
  - local-vs-backing row anchors
  - packet-band / base-offset reinterpretations
  - half-row base-shift cancel / no-cancel variants
- Validation kept from the probe:
  - `TRITON_BUILD_WITH_CCACHE=true TRITON_HOME=/tmp CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
- Next-action consequence:
  - keep the half-row bucket classified as a remaining positive-correctness
    problem that needs a real linear-layout packet decomposition
  - pivot implementation effort back to the tractable support-query cleanup
    backlog (`32x32` rescue/scalarization, `64x128xf32` reinterpret rescue,
    packed row-zero-lifted fixups, `warpx2` family cleanup) instead of landing
    another local half-row workaround

## 2026-04-09 support-query cleanup checkpoint: direct 32x32 / reinterpret rescue now use one contract
- Re-audited the dirty isolated-worktree support-query cleanup after the
  half-row probe and found it was already a coherent landing slice once built
  and revalidated.
- Main cleanup in this slice:
  - replaced the split `SubviewSupportPlan` / ad hoc support-query fallback
    layering with one `getTMemLdStSupportQueryPlan(...)` contract shared by
    verifier diagnostics, Gluon direct-layout search, and LLVM lowering
  - moved TMEM subview offset quirks into one
    `getTMemSubviewOffsetForLowering(...)` helper so `ViewOpToLLVM.cpp` and
    `TensorMemoryToLLVM.cpp` agree on the canonical contiguous `32x32` special
    case and the other lowering-only subview adjustments
  - removed the old `32x32` descriptor-view support-rescue quarantine and
    instead made query-layout lowering explicitly scalarize vectorized
    `32x32b` messages to `x1` packets when translated query origins would
    otherwise over-cover the logical TMEM view
  - preserved non-canonical TMEM view layouts when the direct-view path asks
    for them, and let support-query planning cover reinterpret / column-subview
    descriptor views through the same linear-layout query/origin contract
- Effect on the previous hack inventory:
  - direct `32x32` support rescue / scalarization is now handled by the shared
    query-layout decomposition instead of a hard negative gate
  - `64x128xf32` reinterpret rescue now lowers through the same support-query /
    raw-query machinery instead of a one-off rescue split
  - the `block_m_64` packed row-zero-lifted subslice / reinterpret behavior is
    green again under the unified plan
  - the remaining post-merge-base cleanup backlog is now the separate
    `warpx2` family-specific planner cleanup
- Validation for this checkpoint:
  - `TRITON_BUILD_WITH_CCACHE=true TRITON_HOME=/tmp CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_core.py -k 'test_tmem_linear_runtime_views or test_tmem_linear_m64_roundtrip_32x32b_fallback'`
    - `14 passed`
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_core.py -k test_tmem_descriptor_chain_matrix`
    - `26 passed`
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_core.py -k 'test_tmem_subslice_block_m_64 or test_tmem_subslice_block_m_64_parent_layout_reports_clean_error'`
    - `4 passed`
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_positive or test_tmem_runtime_matrix_ldst_x1_f32_descriptor_chain_roundtrip or test_tmem_runtime_matrix_ldst_descriptor_rank5_roundtrip or test_tmem_runtime_matrix_ldst_twocta_descriptor_rank5_roundtrip'`
    - `4 passed, 12 skipped`
  - manual `triton-opt` checks from the build dir:
    - `bin/triton-opt test/TritonNvidiaGPU/ops.mlir | FileCheck`
    - `bin/triton-opt --split-input-file --verify-diagnostics test/TritonNvidiaGPU/invalid.mlir`
    - `bin/triton-opt test/TritonNvidiaGPU/tmem_layouts.mlir -split-input-file --triton-nvidia-optimize-tmem-layouts --allow-unregistered-dialect | FileCheck`
- Still open after this slice:
  - lifted/direct half-row ld/st remains a separate packet-decomposition BUG;
    do not treat this support-query cleanup as resolving that bucket
  - the GB200-equivalent full sweep is still blocked on node/GPU health, so
    this checkpoint only has focused local validation

## 2026-04-09 warpx2 planner cleanup: shared descriptor-layout search factored out
- Followed the support-query landing by cleaning up the remaining
  `tcgen05.copy.warpx2` planner residue in
  `TensorMemoryUtils.cpp`.
- What changed:
  - kept the real ISA distinction between `warpx2::01_23` and
    `warpx2::02_13` in the small plan-spec selection logic inside
    `getTMemCopyPlans(...)`
  - collapsed the duplicated descriptor-layout mutation search in
    `getTMemCopyDescriptorLayouts(...)` into one bounded
    linear-layout-driven helper shared by both warpx2 families
  - removed the redundant `02_13` seed-layout reinsertion and the stale
    no-op `atom->multicast == 3 ? 32u : 32u` generic plan ternary while
    preserving plan ordering
- Why this matters:
  - the old code implied that `01_23` and `02_13` needed separate descriptor
    mutation logic, but once the family-specific core descriptor shape is
    chosen the remaining layout search is the same linear-layout problem for
    both families
  - this keeps the family-specific behavior where it belongs
    (plan specification / opcode choice) and removes duplicated search code
    from the planner
- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true TRITON_HOME=/tmp CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k warpx2`
    - `4 passed`
  - `build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt --split-input-file --verify-diagnostics test/TritonNvidiaGPU/invalid.mlir`
- Remaining initiative work after this cleanup:
  - the post-merge-base hack inventory from the managed-session handoff is now
    closed
  - still-open initiative work is the separate half-row ld/st
    packet-decomposition correctness bug plus the blocked GB200-equivalent
    validation rerun once node health is restored

## 2026-04-09 GB200-equivalent validation resumed: shard 23/32 clean after cache isolation and boundary updates
- GPU health is restored enough to resume the blocked `gluon-main` sweep.
- First rerun of `gluon-main` group `23/32` exposed one false regression and one real test-boundary drift:
  - `python/test/gluon/test_fpsan.py::test_tmem_index_subslice` was a stale-compiler-cache false failure, not a live TMEM codegen bug.
    - Symptom: the exact nodeid reused a cached scalar `tcgen05.ld.sync.aligned.32x32b` kernel and then faulted with `CUDA error: misaligned address`.
    - Root cause: `fresh_knobs` deletes `TRITON_CACHE_DIR`, so command-line cache-dir isolation is ignored unless the test sets `knobs.cache.dir` again after the fixture reset.
    - Fix landed in test coverage: `test_tmem_index_subslice` now pins `fresh_knobs.cache.dir = fresh_triton_cache`, and the exact nodeid is green again.
  - `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64_parent_layout[...]` had drifted because the old negative expectation no longer matched all layouts.
    - `legacy` is now truly positive: the parent-layout direct store into the packed reinterpret subview executes correctly and lowers without an extra `ttg.convert_layout`.
    - `linear` remains clean-negative for now: the row-zero-lifted `_make_tmem_linear_layout_m64(128)` reinterpret still requires the packed `32x32b.unpack::16b` direct path, so a parent TMEM register layout is not directly codegenable there.
    - Added per-test fresh-cache isolation as well because the `legacy` and `linear` parametrizations can alias through the compile cache inside one pytest process otherwise.
- Validation for this checkpoint:
  - `make -j8`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_core.py::test_tmem_subslice_block_m_64_parent_layout`
    - `2 passed`
  - `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_fpsan.py::test_tmem_index_subslice`
    - `1 passed`
  - `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2-m64splitn3 PYTHONPATH=python:. pytest -s --tb=short -k tmem_legacy_m64_subview_default_load_auto_selects_splitn python/test/gluon/test_core.py`
    - `2 passed`
  - `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. pytest -s --tb=short --splits 32 --group 23 python/test/gluon python/tutorials/gluon`
    - `734 passed, 72 skipped`
- Additional observation from the shard:
  - a `.kind::i8` PTXAS rejection for `sm_103a` still prints under `-s` from an earlier `mma_kernel` test path, but it is not a current failing nodeid in group `23/32`; treat it as separate follow-up noise unless it starts failing a test directly.

## 2026-04-09 higher-rank boundary refresh: OOR and CTA/CGA expectations corrected, remaining failures triaged
- Rebuilt on the now-healthy local GB300 node and reran the higher-rank / exotic TMEM runtime-matrix buckets that were still ambiguous after the resumed 32-way sweep.
- Test-matrix updates in `python/test/gluon/test_tmem_runtime_matrix.py`:
  - `block_single_cta` exotic descriptor-view negatives now assert the real clean diagnostic (`Layout has 2 CTAs per CGA, but the context requires 1 CTAs per CGA.`) instead of the older unsupported-view wording.
  - lifted higher-rank `index`, `multidim_slice`, and `dim0_slice` positives now stop at `n in {64, 128}`; new `n=256` tests assert the real TMEM capacity boundary (`out of resource: tensor memory`, `Required: 1024`, `Hardware limit: 512`) for one-CTA layouts and `block_two_ctas`.
  - two-CTA `mmav5_twocta` lifted `dim0_slice` stays in the existing clean CTA/CGA mismatch bucket; it is not part of the OOR frontier because compilation fails earlier with `Layout has 1 CTAs per CGA, but the context requires 2 CTAs per CGA.`
- Remaining live failures after the test-only refresh are real implementation bugs, not stale expectations:
  - `test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout`
    - for `n in {64, 128}` the kernel now compiles and runs instead of failing cleanly, but the output is wrong.
    - direct probe on `identity`, `n=128`, `variant=32x32b` reproduced `max_abs = 13.0` with `8192` wrong elements; the lowering emits `16x32bx2.x32.b32` packets at offsets `0` and `64`, which does not implement the logical lower-half view semantics.
    - `n=256` is a separate real OOR boundary (`Required: 1024`, `Hardware limit: 512`).
  - `test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_reports_clean_unsupported[scrambled_cols-<lambda>]`
    - the compiler now admits the view, but a direct positive probe is wrong-code (`max_abs = 11.0`, `5120` wrong elements).
    - the emitted sequence decomposes the view into many `32x32b.x1.b32` packets at scalar offsets `0..31`, which does not preserve the scrambled-column physical mapping.
  - `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[rotate1-identity-2-32x32b_splitn]`
    - still fails at launch with `Triton Error [CUDA]: misaligned address` under `CUDA_LAUNCH_BLOCKING=1`; this remains a separate runtime/lowering bug.
- Recommended next implementation order:
  - treat the current commit as a bounded test/triage checkpoint.
  - next, tighten or repair the view-like fallback path in `compute_tmem_reg_layout_from_memdesc(...)` so surrogate query-type fallback does not over-admit wrong-code descriptor views (`scrambled_cols` and the generic lifted half-row bucket).
  - after that, return to the true half-row packet-decomposition fix rather than papering over it with new permanent negatives.
  - keep the split-N permuted `M=64` misaligned-address issue as a separate lowering/reg-layout bug after the view-like descriptor work.
- Validation for this checkpoint:
  - `make -j8`
  - `HOME=/tmp/triton-home-checkpoint-g0 TRITON_CACHE_DIR=/tmp/triton-cache-checkpoint-g0 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_exotic_layouts_report_clean_unsupported python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_index_reports_tmem_oor python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_multidim_slices_report_tmem_oor python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_dim0_slice_positive_lifted_layout python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_dim0_slice_reports_tmem_oor`
    - `44 passed`
  - `HOME=/tmp/triton-home-checkpoint-g2 TRITON_CACHE_DIR=/tmp/triton-cache-checkpoint-g2 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_index_reports_tmem_oor python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_multidim_slices_report_tmem_oor python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_dim0_slice_positive_lifted_layout python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_dim0_slice_reports_tmem_oor python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_mmav5_descriptor_higher_rank_reports_clean_error`
    - `40 passed`
  - exact triage repros kept as evidence for the remaining live bugs:
    - `HOME=/tmp/triton-home-iso-halfrows-g1 TRITON_CACHE_DIR=/tmp/triton-cache-iso-halfrows-g1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout`
      - `12 failed` (`8` wrong-code admissions at `n in {64,128}`, `4` real OORs at `n=256`)
    - `HOME=/tmp/triton-home-iso-scrambled-g3 TRITON_CACHE_DIR=/tmp/triton-cache-iso-scrambled-g3 CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_reports_clean_unsupported[scrambled_cols-<lambda>]`
      - `1 failed` (`did not raise`, later confirmed wrong-code by direct probe)
    - `HOME=/tmp/triton-home-iso-splitn-20260409 TRITON_CACHE_DIR=/tmp/triton-cache-iso-splitn-20260409 CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[rotate1-identity-2-32x32b_splitn]`
      - `1 failed` (`Triton Error [CUDA]: misaligned address`)

## 2026-04-09 lifted column-half descriptor views restored; M64 permuted row-anchor bug isolated
- The dirty descriptor-view fallback work was converted into a real TMEM-utils checkpoint instead of another Gluon-only rescue.
- Implementation outcome:
  - `getColumnSubviewTMemLdStSupportQueryPlan(...)` now recognizes lifted `reshape -> dim0 subslice -> index` column-half views and exposes the backing 2D support frame directly.
  - `isUnsupportedDirectTMemLdStDescriptorView(...)` now rejects true direct-view impossibilities via row-anchor representability instead of relying on the previous blanket surrogate-query skip.
  - Gluon direct reg-layout search no longer suppresses surrogate query types after raw-query failure; valid descriptor views can still fall through to the canonical/surrogate query type once support-query and raw-query paths are exhausted.
- Validated effect on the previously live descriptor-view buckets:
  - lifted one-CTA `dim0_slice` positives are green again for `n in {64, 128}` across all explicit ld/st variants.
  - lifted higher-rank half-row clean negatives stay green.
  - `scrambled_cols` multidim slice stays clean-negative.
  - two-CTA `block_two_ctas` lifted `dim0_slice` positive/OOR slices stay green.
- Focused validation for this checkpoint:
  - `make -j8`
  - `HOME=/tmp/triton-home-targeted-20260409b TRITON_CACHE_DIR=/tmp/triton-cache-targeted-20260409b CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_dim0_slice_positive_lifted_layout python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_reports_clean_unsupported[scrambled_cols-<lambda>]'
    - `21 passed`
  - `HOME=/tmp/triton-home-1cta-20260409 TRITON_CACHE_DIR=/tmp/triton-cache-1cta-20260409 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_index python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_dim0_slice_positive_lifted_layout python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_dim0_slice_reports_tmem_oor python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_identity_reports_clean_error python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_positive python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_reports_clean_unsupported python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_positive_lifted_layout python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout`
    - `47 passed, 1 skipped`
  - `HOME=/tmp/triton-home-2cta-20260409 TRITON_CACHE_DIR=/tmp/triton-cache-2cta-20260409 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_dim0_slice_positive_lifted_layout python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_dim0_slice_reports_tmem_oor`
    - `12 passed`
  - exact split-N repro kept as live evidence:
    - `HOME=/tmp/triton-home-iso-splitn-20260409 TRITON_CACHE_DIR=/tmp/triton-cache-iso-splitn-20260409 CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[rotate1-identity-2-32x32b_splitn]`
      - `1 failed` (`Triton Error [CUDA]: misaligned address`)
- Broad validation notes:
  - the attempted 4-GPU `python/test/gluon/test_core.py + python/test/gluon/test_tmem_runtime_matrix.py` sweep is not a trustworthy TMEM signal on this node yet because GPU 0 hit many unrelated `python/test/gluon/test_core.py::test_mma_shared_inputs[...]` NaN failures.
  - a matrix-only 4-GPU shard run confirmed that the first real TMEM failure is still the `splitn_rowcol_permuted_layout_sweep` bucket; once that misaligned-address launch fires, later tests on the same shard fail as follow-on CUDA-context poison and should not be counted as independent regressions.
- Remaining live TMEM implementation bug after this checkpoint:
  - `M=64` row/col-permuted direct ld/st layouts still admit wrong warp anchor pairs for the 64-row family.
  - direct warmup on `row_perm=rotate1`, `col_perm=identity`, `n=2` shows the generated `#linear` warp bases as `[[32, 0], [1, 0]]` and the TMEM row bases as `[[2, 0], [4, 0], [8, 0], [16, 0], [0, 0], [32, 0], [1, 0]]`; both explicit `16x32bx2` and auto-selected `32x32b_splitn` emit the expected `16x32bx2.x2.b32` packet family at immediate `0`, then fault at launch.
  - this points at a row-anchor selection/validation bug: the current direct-ld/st path is effectively deriving warp anchors from permuted row-basis position instead of solving for the logical 64-row anchor family (`16,32`) in linear-layout space.
- Recommended next step:
  - fix the 64-row direct ld/st anchor selection/validation path so it chooses (or rejects against) logical row anchors in linear-layout coordinates rather than basis-order position, then rerun the exact `splitn_rowcol_permuted_layout_sweep` bucket and the interrupted matrix-only shard.

## 2026-04-09 MMAv5 `f16` root-accumulator row-plan fix; split-N permuted ld/st bug remains separate
- The `test_mma_shared_inputs` failures on the healthy local GPUs reduced to a producer-side row-plan omission, not a new generic tcgen05 arithmetic bug.
- Exact failing repros before the fix:
  - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-0-warps1-16-False-False-acc_dtype1]`
  - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-0-warps2-16-False-False-acc_dtype1]`
- Root cause:
  - `python/src/gluon_ir.cc` already calls `annotateMMAv5AccumulatorRootRowPlan(acc)` before `ttng.tc_gen5_mma`, but `getMMAv5AccumulatorRootRowPlan(...)` in `TensorMemoryUtils.cpp` only returned a plan for `bitwidth == 32`.
  - As a result, raw `64x{64,32}xf16` MMA accumulators never received the explicit `ttng.tmem_ldst_row_plan` contract, fell back to the active-row `16/32 @ 64` family, and lowered through the narrow `shl 20` / mask `3145728` TMEM base path.
  - Exact failing TTGIR/PTX shape before the fix:
    - no `ttng.tmem_ldst_row_plan` attribute on the `ttng.tmem_alloc`
    - `tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32`
    - runtime numerics zeroed the `row 16-31` and `48-63` bands (`2001 / 4096` mismatches on the `64x64xf16` repro)
- Fix:
  - broaden `getMMAv5AccumulatorRootRowPlan(...)` to all rank-2 `M=64` MMAv5 accumulators instead of gating it to 32-bit element types.
  - This restores the producer-owned explicit row-plan metadata for the `f16` accumulator family as well.
- Post-fix evidence:
  - fresh TTGIR now shows `ttng.tmem_alloc {ttng.tmem_ldst_row_plan = array<i32: 32, 64, 128, 0>}` for both `64x64xf16` and `64x32xf16` root accumulators.
  - fresh PTX for the repaired `64x64xf16` repro now uses the widened `shl 21` / mask `6291456` TMEM base math and `tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32`.
- Validation:
  - `make -j8`
  - `CUDA_VISIBLE_DEVICES=0 ... pytest -s --tb=short -vv python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-0-warps1-16-False-False-acc_dtype1]`
    - `1 passed`
  - `CUDA_VISIBLE_DEVICES=1 ... pytest -s --tb=short -vv python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-0-warps2-16-False-False-acc_dtype1]`
    - `1 passed`
  - `CUDA_VISIBLE_DEVICES=2 ... pytest -s --tb=short -vv python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-0-warps1-16-False-False-acc_dtype2]`
    - `1 passed`
  - `CUDA_VISIBLE_DEVICES=0 ... pytest -s --tb=short -vv python/test/unit/cuda/test_tma_store_gemm.py::test_tma_load_store[64-128-32-1-4-False-True-False]`
    - `1 passed`
- Remaining separate live bug after this checkpoint:
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[rotate1-identity-2-32x32b_splitn]`
    - still fails at launch with `Triton Error [CUDA]: misaligned address`
    - unchanged by the row-plan producer fix, so keep it classified as the separate `M=64` permuted direct-ld/st anchor-selection bug.

## 2026-04-09 centralize logical row-anchor matching; close the split-N permuted `M=64` bucket
- The remaining live local TMEM failure after the MMAv5 `f16` checkpoint was the `M=64` row/col-permuted direct ld/st anchor-selection bug.
- Root cause:
  - direct ld/st support and lowering were still treating the row-anchor family as if row-basis position matched `log2(logicalRow)`.
  - that assumption is false once the row bases are permuted or interleaved with explicit zero-lift bases; on the `rotate1/identity/n=2` repro, the logical `16,32` family was being read back as `32,1`.
- Fix shape:
  - factor `getLogicalRowAnchorBasis(...)` in `TensorMemoryUtils.cpp` and use it for:
    - direct packed-support query planning
    - main `computeTMemLdStEncodingInfoImpl(...)` row-anchor matching
    - unsupported descriptor-view row-anchor diagnostics
  - the helper matches row anchors by the row-coordinate contribution of each TMEM row basis, not by basis index.
  - important implementation note: a follow-on attempt to recover the row coordinate from `LinearLayout` output-dimension names was wrong and immediately regressed valid `test_mma_shared_inputs` / split-N / descriptor-view positives; the stable invariant here is that TMEM row bases are emitted as `[row, col]` output vectors, so the row coordinate is still `basis.front()`.
- Durable outcome:
  - positive lowering and clean-negative descriptor-view diagnostics now use the same row-anchor materialization rule.
  - the split-N row/col-permuted `M=64` bucket is green again without reintroducing Gluon-only rescue logic or new permanent negatives.
- Final validation for this checkpoint:
  - `make -j8`
  - `HOME=/tmp/triton-home-mma-postrefactor2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-postrefactor2 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-0-warps1-16-False-False-acc_dtype1]`
    - `1 passed`
  - `HOME=/tmp/triton-home-splitn-postrefactor2 TRITON_CACHE_DIR=/tmp/triton-cache-splitn-postrefactor2 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[rotate1-identity-2-32x32b_splitn]`
    - `1 passed`
  - `HOME=/tmp/triton-home-desc-rowcol-postrefactor2 TRITON_CACHE_DIR=/tmp/triton-cache-desc-rowcol-postrefactor2 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions_rowcol_permuted_layout_sweep[rotate1-identity-64-32x32b-32x32b.x64.b32]`
    - `1 passed`
  - `HOME=/tmp/triton-home-multidim-neg TRITON_CACHE_DIR=/tmp/triton-cache-multidim-neg CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. pytest -s --tb=short -vv 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_reports_clean_unsupported[scrambled_cols-<lambda>]'`
    - `1 passed`
  - `HOME=/tmp/triton-home-mma-minimal-final TRITON_CACHE_DIR=/tmp/triton-cache-mma-minimal-final CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -vv <30 exact nodeids from python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-0-...]`
    - `30 passed`
  - `HOME=/tmp/triton-home-splitn-rowcol-final TRITON_CACHE_DIR=/tmp/triton-cache-splitn-rowcol-final CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py -k test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep`
    - `224 passed, 1566 deselected`
- Remaining initiative work after this checkpoint:
  - the separate half-row ld/st packet-decomposition correctness bucket
  - resuming the broader GB300-equivalent validation on the now-healthy machine

## 2026-04-09 full-image row-anchor representability closes the widened 2CTA MMAv5 descriptor-view bucket
- Resuming the healthy-node broad validation surfaced a real new TMEM regression in `python/test/gluon/test_core.py::test_mma_shared_inputs[...]`, not node noise:
  - exact failing compiler repros before the fix:
    - `python/test/gluon/test_core.py::test_mma_shared_inputs[True-ctas_per_cga1-1-1-1-64-0-0-warps0-8-False-True-acc_dtype0]`
    - `python/test/gluon/test_core.py::test_mma_shared_inputs[True-ctas_per_cga2-1-1-1-64-64-0-warps0-16-True-False-acc_dtype6]`
  - exact failure shape:
    - `TMEM layout 'auto' unsupported for descriptor view tensor_memory_descriptor<...>`
    - debug reason from `TRITON_DEBUG_TMEM_REG_LAYOUT=1`:
      - `unsupported tensor memory descriptor view for direct tcgen05.ld/st: required row anchors 32,64 are not directly representable in the descriptor view.`
- Root cause:
  - the centralized `getLogicalRowAnchorBasis(...)` helper from the previous checkpoint was still too narrow: it only accepted required row anchors if they appeared in the TMEM `row` basis family.
  - that is not a stable invariant for arbitrary linear layouts. In the failing widened two-CTA MMAv5 accumulator descriptor view, the required logical `64` anchor is real but carried by the full linear-layout image through a `col` basis rather than a `row` basis.
  - direct ld/st legality here is a question about whether the output-space anchor vector `[logical_row, 0]` is representable at all, not which input-dimension family materializes it.
- Concrete evidence from the failing layout:
  - `cga_layout_c = ((1, 0),)`
  - `TensorMemoryLinearLayout(rows=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 32]], cols=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [64, 0]], shape=[256, 64], block_bases=[[128, 0]], two_ctas=True)`
  - the `64` row anchor is visibly materialized by the `cols` family as `[64, 0]`, so rejecting the descriptor view was a compiler bug.
- Fix:
  - widen `getLogicalRowAnchorBasis(...)` so it checks full-image representability of the desired TMEM output-space anchor instead of scanning only `layout.getBases().lookup(kRow)`.
  - implementation shape in `TensorMemoryUtils.cpp`:
    - build the expected TMEM output vector `[logical_row, 0, ...]`
    - use `layout.pseudoinvert()` to synthesize candidate input coordinates
    - reapply the original layout and accept the anchor only if the realized output vector exactly matches the expected one
  - keep the same shared helper wiring for packed support-query planning, the main `computeTMemLdStEncodingInfoImpl(...)` path, and the clean-negative unsupported-descriptor diagnostic.
- Durable outcome:
  - direct tcgen05 ld/st support is now keyed off the full TMEM linear-layout image, which is the right abstraction for arbitrary linear layouts.
  - valid widened descriptor views are no longer rejected just because the matching anchor is carried by a `col` or block basis family.
- Focused validation after the fix:
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
  - `HOME=/tmp/triton-home-twocta-repro1 TRITON_CACHE_DIR=/tmp/triton-cache-twocta-repro1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_core.py::test_mma_shared_inputs[True-ctas_per_cga1-1-1-1-64-0-0-warps0-8-False-True-acc_dtype0]`
    - `1 passed`
  - `HOME=/tmp/triton-home-twocta-repro2 TRITON_CACHE_DIR=/tmp/triton-cache-twocta-repro2 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_core.py::test_mma_shared_inputs[True-ctas_per_cga2-1-1-1-64-64-0-warps0-16-True-False-acc_dtype6]`
    - `1 passed`
  - one-CTA control:
    - `HOME=/tmp/triton-home-twocta-control TRITON_CACHE_DIR=/tmp/triton-cache-twocta-control CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga1-1-1-1-64-0-0-warps0-8-False-True-acc_dtype0]`
      - `1 passed`
  - split-N guard:
    - `HOME=/tmp/triton-home-splitn-guard4 TRITON_CACHE_DIR=/tmp/triton-cache-splitn-guard4 CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[rotate1-identity-2-32x32b_splitn]`
      - `1 passed`
  - widened two-CTA grouped slices:
    - `HOME=/tmp/triton-home-twocta-slice1 TRITON_CACHE_DIR=/tmp/triton-cache-twocta-slice1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -vv <30 exact nodeids matching test_mma_shared_inputs[True-ctas_per_cga1-1-1-1-64-0-0-...]>`
      - `30 passed`
    - `HOME=/tmp/triton-home-twocta-slice2 TRITON_CACHE_DIR=/tmp/triton-cache-twocta-slice2 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. pytest -s --tb=short -vv <30 exact nodeids matching test_mma_shared_inputs[True-ctas_per_cga2-1-1-1-64-64-0-...]>`
      - `30 passed`
    - `HOME=/tmp/triton-home-twocta-slice3 TRITON_CACHE_DIR=/tmp/triton-cache-twocta-slice3 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. pytest -s --tb=short -vv <30 exact nodeids matching test_mma_shared_inputs[True-ctas_per_cga2-1-1-1-64-128-32-...]>`
      - `30 passed`
- Next step after this checkpoint:
  - resume the interrupted 4-GPU `python/test/gluon/test_core.py` + `python/test/gluon/test_tmem_runtime_matrix.py` sweep and stop on the next clean failure bucket, if any.

## 2026-04-09 row anchors are the analyzed layout image of logical row coordinates, not fixed output vectors
- The previous checkpoint's "full-image output-space anchor `[logical_row, 0]`" rule was still too strong.
- Broad reruns exposed a real false-negative on the direct split-N root path:
  - `python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[splitn_64x128-layout6-64-128-expected_offset_imms6]`
  - failure shape before the fix:
    - `ttng.tmem_store op source has no supported register layout`
    - diagnostic: `required row anchors 32,64 are not directly representable in the descriptor view`
- Half-rows debug showed why the output-vector rule was wrong for legal M64 direct paths:
  - analyzed mem layout for the split-N `64x128` root case:
    - `row=16 -> (0, 64)`
    - `row=32 -> (16, 0)`
    - `row=64 -> (32, 0)`
  - row plans are expressed in the analyzed layout's logical `row` input space, and the physical warp anchor is the image of that logical row coordinate under the analyzed layout.
  - in other words, the direct path does not require literal output vectors `[32, 0]` and `[64, 0]`; it requires whatever output-space vectors `row=32` and `row=64` map to in the analyzed layout.
- Corrected helper invariant:
  - `getLogicalRowAnchorBasis(...)` must build the sparse logical input point `{row = logicalRow}` and apply the analyzed `LinearLayout` to it.
  - this naturally handles row/col/block entanglement without assuming basis-family position or assuming a fixed output-space row axis.
- Important cross-check:
  - the widened two-CTA MMAv5 case that motivated the previous checkpoint still stays green under this corrected rule.
  - half-rows debug there shows the same phenomenon: `row=64` maps to `(0, 32)` in the analyzed mem layout, while the lowering still succeeds.
- Focused validation after the correction:
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
  - `HOME=/tmp/triton-home-splitn64x128-fixcheck TRITON_CACHE_DIR=/tmp/triton-cache-splitn64x128-fixcheck CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[splitn_64x128-layout6-64-128-expected_offset_imms6]`
    - `1 passed`
  - `HOME=/tmp/triton-home-twocta-fixcheck TRITON_CACHE_DIR=/tmp/triton-cache-twocta-fixcheck CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_core.py::test_mma_shared_inputs[True-ctas_per_cga2-1-1-1-64-64-0-warps0-16-True-False-acc_dtype6]`
    - `1 passed`
  - `HOME=/tmp/triton-home-splitn-guard-fixcheck TRITON_CACHE_DIR=/tmp/triton-cache-splitn-guard-fixcheck CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[rotate1-identity-2-32x32b_splitn]`
    - `1 passed`
- New remaining live bucket after this correction:
  - `python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
  - exact current failure:
    - `ttng.tmem_store op source has no supported register layout`
    - diagnostic: `row-zero lifted TMEM reinterpret views require the packed 32x32b.unpack::16b direct path`
  - TMEM query debug now shows this is not an arbitrary rejection:
    - packed support analysis reaches a valid `atom=4` / `regsPerMsg=1` candidate and computes the packed conversion
    - the current hard stop is the explicit guard `packed16 support skip: row-zero lifted views require 32x32b.unpack direct lowering`
  - recommendation:
    - implement the packed `32x32b.unpack::16b` direct lowering for row-zero-lifted reinterpret views instead of rejecting them at support-query time.

## 2026-04-09: `block_m_64` linear is fixed; the remaining failures were stale tests, not a live TMEM bug

- The original `block_m_64` diagnosis was only half right.
  - there was a real compiler bug, but it was not "missing packed unpack direct lowering" anymore after the row-zero-lifted support-query cleanup.
  - after that cleanup, the remaining wrong-code came from producer planning: the manual `tcgen05_mma` builder annotated the accumulator TMEM root with an explicit `ttng.tmem_ldst_row_plan`, but left the TMEM operand roots implicit.
- Root cause:
  - `test_block_m_64_mma[linear]` stores `al` / `ar` through row-zero-lifted reinterpret views and then consumes those TMEM allocations through `tcgen05_mma`.
  - with only `%acc_tmem` carrying `ttng.tmem_ldst_row_plan = array<i32: 32, 64, 128, 0>`, TTGIR/LLIR still chose the expected `16x32bx2` direct op mix, but the TMEM source roots and accumulator root no longer agreed on the explicit row-anchor contract.
  - that mismatch produced wrong numerics even though the emitted opcode family looked plausible.
- Fix shape:
  - in `TensorMemoryUtils.cpp`, keep row-zero-lifted `M=64` reinterpret recognition keyed off the original reinterpret-view semantics, not only the remapped support-query mem layout.
  - still reject the packed-support shortcut for row-zero-lifted non-`I32x32b` candidates so the compiler does not silently fall back to the wrong packed path.
  - in `python/src/gluon_ir.cc`, add `annotateMMAv5TMemOperandRootRowPlan(...)` and copy the accumulator's explicit row plan onto TMEM-backed MMA operands in both `create_tcgen05_mma(...)` and `create_tcgen05_mma_scaled(...)`.
- Concrete evidence after the fix:
  - TTGIR now carries `ttng.tmem_ldst_row_plan = array<i32: 32, 64, 128, 0>` on `%al_tmem`, `%ar_tmem`, and `%acc_tmem`.
  - the focused `block_m_64` probe is numerically clean:
    - `close True`
    - `max_abs 0.015384674072265625`
  - the intended direct-path opcode mix is preserved:
    - `tcgen05.st.sync.aligned.16x32bx2.x16.b32` -> `4`
    - `tcgen05.st.sync.aligned.16x32bx2.x32.b32` -> `2`
    - `tcgen05.ld.sync.aligned.16x32bx2.x32.b32` -> `2`
    - `tcgen05.mma.cta_group::1.kind::f16` -> `16`
- Test fallout:
  - `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64_parent_layout[linear-False]` was a stale negative. The linear parent-layout reinterpret path is now a valid direct positive, so the test should compile/run and assert no `ttg.convert_layout`.
  - `python/test/gluon/test_core.py::test_block_m_64_mma[linear]` failed only because its TTGIR/LLIR shape assertions lagged the fix.
  - one attempted test update overfit the exact TMEM alloc word count (`128` vs `256`); that is not the right contract here, so the final test keeps the direct-path opcode and correctness assertions and only checks that the alloc exists.
- Focused validation after the fix and test refresh:
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
  - `HOME=/tmp/triton-home-blockm64-parent-final3 TRITON_CACHE_DIR=/tmp/triton-cache-blockm64-parent-final3 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_core.py::test_tmem_subslice_block_m_64_parent_layout`
    - `2 passed`
  - `HOME=/tmp/triton-home-blockm64-final3 TRITON_CACHE_DIR=/tmp/triton-cache-blockm64-final3 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_core.py::test_block_m_64_mma`
    - `2 passed`
  - `HOME=/tmp/triton-home-mma-twocta-final3 TRITON_CACHE_DIR=/tmp/triton-cache-mma-twocta-final3 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_core.py::test_mma_shared_inputs[True-ctas_per_cga2-1-1-1-64-64-0-warps0-16-True-False-acc_dtype6]`
    - `1 passed`
  - `HOME=/tmp/triton-home-splitn-guard-final3 TRITON_CACHE_DIR=/tmp/triton-cache-splitn-guard-final3 CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[rotate1-identity-2-32x32b_splitn]`
    - `1 passed`
- Current status after closing `block_m_64`:
  - the `block_m_64` linear bucket is no longer a live issue.
  - the immediate next step is to resume the broader healthy-node sweep and stop on the next real TMEM failure surface, if any.

## 2026-04-09: broad healthy-node sweep narrows to an `even_odd` M64 direct-path misaligned-address bug

- I resumed the blocked 4-GPU `python/test/gluon/test_core.py` + `python/test/gluon/test_tmem_runtime_matrix.py` sweep on the healthy node using per-shard log files so expected clean-negative PTXAS output would not masquerade as a live regression.
- Sweep status:
  - group 1: `4327 passed, 612 skipped, 14814 deselected`
  - group 2: `2596 passed, 2343 skipped, 14814 deselected`
  - group 3: `2388 passed, 2551 skipped, 14814 deselected`
  - group 4: `505 failed, 2810 passed, 1621 skipped, 14817 deselected`
- Important non-bug classification from group 4:
  - the early `.kind::i8` PTXAS dump is expected output from `python/test/gluon/test_core.py::test_tcgen05_mma_plain_kind_i8_reports_clean_error`.
  - exact isolate:
    - `CUDA_VISIBLE_DEVICES=3 ... pytest -s --tb=short -vv python/test/gluon/test_core.py::test_tcgen05_mma_plain_kind_i8_reports_clean_error`
    - result: `1 passed`
  - do not treat the `.kind::i8` PTXAS text in the shard log as a TMEM initiative regression.
- Group-4 failures reduce to two coherent buckets:
  - stale clean-negative expectation:
    - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_block_descriptor_reports_clean_error[block_two_ctas-layout1-reinterpret_layout1-expected_fragments1]`
    - exact isolate still fails only because the expected text is stale.
    - current compiler diagnostic is more specific:
      - `unsupported tensor memory descriptor view for direct tcgen05.ld/st: required row anchors 16,32 are not directly representable in the descriptor view. Access the full backing tile or reshape/copy so the TMEM row anchors stay materializable.`
    - old expected fragment `No TMEM-compatible register layout exists for this operand.` should be updated.
  - real runtime bug:
    - exact first failing shard nodeid:
      - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[even_odd-identity-2-32x32b_splitn]`
    - fresh isolate with `CUDA_LAUNCH_BLOCKING=1` still fails at launch with:
      - `RuntimeError: Triton Error [CUDA]: misaligned address`
    - explicit direct-path control also fails fresh:
      - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[even_odd-identity-2-16x32bx2]`
      - same `misaligned address`
- Root-cause classification for the real bucket:
  - this is not a split-N auto-selection heuristic issue; the explicit `16x32bx2` path fails too.
  - it is also not a clean legality-screening issue in the current support query:
    - debug shows `rawQuery -> ok atom=4`
    - chosen row plan is still `warpRow0=16 warpRow1=32 rowSpan=64 baseOffset=0`
    - chosen packet shape is still `secondHalfOffset=0`
  - the likely bug is the common M64 direct `16x32bx2` half-row packet decomposition / anchor realization for certain row-basis permutations.
  - evidence from a passing control:
    - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[rotate1-identity-2-32x32b_splitn]`
    - still passes with the same high-level row plan and same `atom=4`
  - key difference is the analyzed TMEM row-basis ordering:
    - failing raw memTy rows:
      - `[[1, 0], [4, 0], [16, 0], [2, 0], [0, 0], [8, 0], [32, 0]]`
    - passing raw memTy rows:
      - `[[2, 0], [4, 0], [8, 0], [16, 0], [0, 0], [32, 0], [1, 0]]`
  - inference:
    - the direct path is being admitted for both cases, but the emitted access decomposition only stays valid for some row-basis permutations.
    - this matches the initiative's remaining "half-row ld/st packet-decomposition correctness" bucket much more than a new verifier/support-query bucket.
- Important sweep-reading guidance:
  - after the first `misaligned address`, many later tests in the same group-4 process fail immediately on unrelated CUDA calls or on stale clean-negative expectations.
  - do not count the full `505 failed` summary as 505 independent regressions.
  - the next real implementation target is the exact `even_odd` split-N/direct `16x32bx2` misaligned-address repro above.
- Recommended next step:
  - fix the common M64 direct `16x32bx2` row-permuted half-row planner in `TensorMemoryUtils.cpp`
  - rerun:
    - the exact `even_odd-identity-2` `32x32b_splitn` repro
    - the matching explicit `16x32bx2` repro
    - the stale block-descriptor clean-negative test after updating its expected fragments
    - group 4 of the broad sweep

## 2026-04-09: root `M=64x32xf32` TMEM loads were selecting the wrong atom family, and the structural fix is validated

- The remaining one-CTA tcgen05 failure after the `even_odd` cleanup was:
  - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-32-warps2-8-False-True-acc_dtype0]`
- I stopped treating that as another packet-offset bug and traced the whole root-load selection path instead:
  - the public descriptor `get_reg_layout(...)` for the same `64x32xf32` root descriptor already returned the canonical split-N register layout;
  - the actual TTGIR `acc_tmem.load()` result layout still came out as a generic linear layout that lowered to thirty-two scalar `tcgen05.ld.sync.aligned.32x32b.x1.b32` messages; and
  - Gluon trace showed the handle-aware memdesc path was satisfying a split-N request with an `I32x32b` candidate before the real `I16x32bx2` family was considered.
- Root cause:
  - `python/src/gluon_ir.cc` treated `I32x32b` and `I16x32bx2` as interchangeable for rank-2 `M=64` `f32` descriptors via `matchesDesiredAtom(...)`; and
  - the handle-aware candidate list prefixed row-plan-specific direct layouts ahead of the generic `getTmemCompatibleLayouts(...)` order, so the search could short-circuit on the wrong family.
- Durable fix shape:
  - `python/src/gluon_ir.cc`
    - make `matchesDesiredAtom(...)` strict again for the TMEM descriptor API and handle-aware memdesc path;
    - align handle-aware candidate ranking with `getTmemCompatibleLayouts(queryTy, numWarps)` before appending memdesc-specific rescue layouts; and
    - when the public descriptor API is asked for `M=64`, 4-warp `I16x32bx2`/`I32x32b` layouts, try the canonical TMEM-linear type first so the same split-N family is visible through the public type path.
  - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
    - factor a reusable canonical `M=64` split-N linear layout builder; and
    - inject that layout into both direct-layout construction and `getTmemCompatibleLayouts(...)` so the compiler has a stable canonical split-N candidate family.
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
    - keep the earlier query-type-before-raw-query ordering for rank-2 `M=64` root loads so LLVM lowering follows the same family choice as Gluon.
- Concrete evidence after the fix:
  - TTGIR for the exact failing kernel now uses the split-N linear register layout:
    - `register = [[0,1],[0,2],[0,4],[0,16]]`
    - `lane = [[1,0],[2,0],[4,0],[8,0],[0,8]]`
    - `warp = [[16,0],[32,0]]`
  - PTX for the root load now uses the correct two-message family instead of 32 scalar loads:
    - `tcgen05.ld.sync.aligned.16x32bx2.x8.b32`
- Focused validation on the cleaned tree:
  - `make -j8`
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-mma-exact-fix-clean PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-32-warps2-8-False-True-acc_dtype0]`
    - `1 passed`
  - `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-blockm64-fix-clean PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
    - `1 passed`
  - `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-roundtrip-fix-clean PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]`
    - `1 passed`
  - `CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-tmem-rowcol-fix PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_auto_selects_16x32bx2 python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_rowcol_permuted_layout_sweep python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions_rowcol_permuted_layout_sweep`
    - `615 passed in 378.72s`
- Validation gap on this node:
  - compiler lit checks are still unavailable here because neither `lit` nor `python3 -m lit` is installed and there is no local `llvm-lit` in the build tree.
- Status after this checkpoint:
  - the live `M=64x32xf32` tcgen05 root-load bug is fixed;
  - the `even_odd` row/col TMEM runtime slice is green again; and
  - the next initiative step is to resume the broader healthy-node 4-GPU `test_core.py` + `test_tmem_runtime_matrix.py` sweep from this cleaner tip and classify the next real failure surface, if any.

## 2026-04-10: audited merge-base diff for TMEM patchwork debt and turned it into an explicit cleanup backlog

- I reviewed the compiler-side diff against merge-base `7f61ac734edc657b737fb159a1b9d50cb47944e6` (`master`), excluding `.codex`, to separate real generalization work from temporary rescues that should now be folded back into cleaner abstractions.
- The recurring TMEM debt families are:
  - duplicated ld/st planning across Gluon / verifier / LLVM lowering;
  - M64-family-specific layout-selection fast paths;
  - reinterpret/support-query packet surgery in `TensorMemoryUtils.cpp`;
  - legacy-encoding rescue paths that still diverge from TMEM-linear lowering;
  - scattered row-plan override acquisition; and
  - tests that still rely on `_reinterpret` as an implicit physical-layout escape hatch.
- I correlated that audit with the current live `block_m_64` failures and the previously recorded PTX analysis:
  - `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[legacy]`
    - fresh exact pytest still fails at `4096 / 8192`
    - fresh probe still shows wrong bands only in `64:96` and `96:128`
    - PTX still emits a legacy-style reinterpret zero-store through `tcgen05.st.sync.aligned.16x32bx2.x16.b32`
    - passing TMEM-linear control emits repeated `tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32`
    - conclusion: legacy lowering is still diverging from the correct packed-support physical family
  - `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64_parent_layout[linear]`
    - fresh exact pytest still fails at `2048 / 8192`
    - fresh probe shows mismatches only in `32:64` and `96:128`
    - PTX is already on the packed `tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32` family and still omits extra `ttg.convert_layout`
    - conclusion: atom-family choice is already good; the remaining bug is packet decomposition, specifically failing to separate support-band selection from ordinary repetition
- This reinforces the initiative's working principle:
  - fix the shared linear-layout / quotient / planner abstraction, not the symptom with more M64-specific offset surgery.
- Explicit cleanup order recorded for the next implementation passes:
  1. build one shared TMEM ld/st planner consumed by Gluon, verifier, and LLVM lowering;
  2. canonicalize legacy TMEM physical families onto the same planner as TMEM-linear layouts;
  3. derive reinterpret packet offsets / repetition / support-band selection from quotient factorization in one physical coordinate frame;
  4. after the core logic is stable, rewrite the reinterpret-heavy `block_m_64` tests to use guaranteed descriptor/view APIs (`slice/index/reshape/permute`) so they encode user intent rather than old compiler accidents.
- Validation performed for this audit pass:
  - `make -j8`
  - `CUDA_VISIBLE_DEVICES=0 ... pytest -s --tb=short -vv python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[legacy]`
    - `1 failed` (`4096 / 8192`)
  - `CUDA_VISIBLE_DEVICES=1 ... pytest -s --tb=short -vv python/test/gluon/test_core.py::test_tmem_subslice_block_m_64_parent_layout[linear]`
    - `1 failed` (`2048 / 8192`)
  - throwaway PTX/LLIR probe matching those kernels:
    - confirmed the opcode-family split described above between broken legacy direct and working TMEM-linear direct
    - confirmed the parent-layout bucket is already on the packed opcode family and is now blocked specifically on packet decomposition semantics

## 2026-04-10: recorded the long-term TMEM completion plan so future sessions align on the full mission

- I added a durable `Long-Term Mission And Completion Plan` section to
  `.codex/initiatives/tmem_linear_generalization/memory.md`.
- The purpose is to keep future context windows aligned on the actual project
  finish line rather than stopping at the current bug buckets.
- The recorded plan states that after the current planner/cleanup work lands,
  the initiative still needs:
  - shared-planner unification across frontend/verifier/LLVM;
  - legacy/TMEM-linear unification;
  - quotient-driven reinterpret packet decomposition;
  - reinterpret-heavy test rewrites toward guaranteed descriptor/view APIs;
  - broader `tcgen05` surface completion across `ld/st`, `ld.red`, `copy`,
    `mma`, and `mma_scaled`;
  - performance-oriented family-selection heuristics; and
  - staged broad validation up through TMEM runtime, MMA/matmul,
    `triton_kernels`, broader suite, and grouped multi-GPU sweeps.
- I also added a shorter future-session anchor to
  `.codex/initiatives/tmem_linear_generalization/handoff_2026-04-09.md` so
  sessions that begin from the handoff tail still see the long-term plan and
  current phase ordering immediately.

## 2026-04-10: full documentation sweep refreshed the current source-of-truth docs without rewriting history

- I read the full initiative doc set:
  - `memory.md`
  - `log.md`
  - `handoff_2026-04-09.md`
  - `fuzz_plan.md`
  - `handoff_2026-03-26.md`
  - `next_agent_prompt_2026-03-26.md`
- I left the older dated handoffs/prompts alone on purpose; they are archival
  provenance, not current status.
- Documentation classification after the sweep:
  - `memory.md`
    - durable source of truth for mission, current decisions, long-term plan,
      and current undated priorities
  - `log.md`
    - append-only chronology
  - `handoff_2026-04-09.md`
    - active current-state handoff
  - `fuzz_plan.md`
    - operational post-cleanup saturation playbook
  - `handoff_2026-03-26.md` and `next_agent_prompt_2026-03-26.md`
    - archival
- Current-doc refreshes made:
  - refreshed `memory.md`'s undated `Next Execution Plan` and
    `Active Fuzz Backlog` so they match the current planner-cleanup-first
    reality rather than the earlier broad-fuzz-first phase
  - added a `How To Use This File` section to `fuzz_plan.md`
  - updated `fuzz_plan.md`'s immediate sections so they clearly defer broad
    saturation until the shared-planner / legacy-unification / reinterpret
    packet-decomposition work is green

## 2026-04-10: moved the initiative doc index into a dedicated README

- Added `.codex/initiatives/tmem_linear_generalization/README.md` as the entry
  point for the initiative folder.
- The README now:
  - classifies each document by role;
  - tells future sessions to start there, then read `memory.md` as the
    initiative's durable memory/source of truth;
  - points to the active handoff and `fuzz_plan.md`; and
  - explicitly says to keep the README up to date when document roles or
    source-of-truth entry points change.
- Removed the duplicated documentation index from `memory.md` so that
  `memory.md` stays focused on project memory, decisions, and plans rather than
  folder navigation.

## 2026-04-10: added a GB200-only NVIDIA CI inventory and cleared the stale block-descriptor diagnostic expectation

- I added `.codex/initiatives/tmem_linear_generalization/gb200_nvidia_ci_inventory.md`
  as the current CI-grounded baseline for this Blackwell devbox phase.
- Scope for this stabilization pass is intentionally narrowed to the GB200 lane
  of `.github/workflows/integration-tests-nvidia.yml`; AMD, macOS, and the
  H100-only interpreter lane are out of scope for the current red-list pass.
- The new inventory records:
  - the GB200 workflow commands;
  - a repo-level test-surface snapshot (`ctest -N`, lit file count, Python test
    file counts, and the current `test_core.py` + `test_tmem_runtime_matrix.py`
    collect-only size); and
  - the currently confirmed red list on this devbox.
- Exact-nodeid confirmation before the test expectation update:
  - `CUDA_VISIBLE_DEVICES=0 ... pytest -s --tb=short -vv python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[legacy]`
    - `1 failed` (`4096 / 8192`)
  - `CUDA_VISIBLE_DEVICES=1 ... pytest -s --tb=short -vv python/test/gluon/test_core.py::test_tmem_subslice_block_m_64_parent_layout[linear]`
    - `1 failed` (`2048 / 8192`)
  - `CUDA_VISIBLE_DEVICES=2 ... pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_block_descriptor_reports_clean_error[block_two_ctas-layout1-reinterpret_layout1-expected_fragments1]`
    - failed only because the expected diagnostic text still referenced the old
      generic register-layout message
- I updated the stale block-descriptor expectation to the current row-anchor
  materialization diagnostic fragments:
  - `source has no supported register layout`
  - `unsupported tensor memory descriptor view for direct tcgen05.ld/st`
  - `required row anchors 16,32 are not directly representable in the descriptor view`
- Validation after the update:
  - `make -j8`
  - exact rerun of the same block-descriptor nodeid
    - `1 passed`
- Status after this checkpoint:
  - the stale runtime-matrix expectation is no longer part of the GB200 red
    list; and
  - the only currently confirmed GB200/NVIDIA red tests are the two
    reinterpret-heavy `block_m_64` tests, which are now explicitly tracked as
    rewrite candidates under the newer TMEM descriptor/view contract rather
    than proven compiler bugs.

## 2026-04-10: first explicit-view rewrite attempts for the remaining `block_m_64` GB200 reds narrowed the problem

- After the stale expectation fix, I tried to rewrite the two remaining
  `block_m_64` GB200 red tests to explicit descriptor/view APIs instead of
  implicit `_reinterpret(...)` behavior.
- First attempt: direct 2D slices only
  - kernel idea:
    - replace the old half-band borrow with `slice(0, 32, dim=1)` /
      `slice(64, 32, dim=1)` stores
    - replace the old `64x2` reinterpret fragments with nested 2-column slices
      inside `slice(32, 32, dim=1)`
  - validation:
    - `make -j8`
    - `CUDA_VISIBLE_DEVICES=0 ... pytest -s --tb=short -vv python/test/gluon/test_core.py::test_tmem_subslice_block_m_64`
    - `CUDA_VISIBLE_DEVICES=1 ... pytest -s --tb=short -vv python/test/gluon/test_core.py::test_tmem_subslice_block_m_64_parent_layout`
  - result:
    - both tests compiled and still failed with the same old mismatch counts:
      - `4096 / 8192`
      - `2048 / 8192`
  - conclusion:
    - plain logical slices do not restate the intended borrowed-physical
      contract for M64
- Second attempt: explicit quarter-band reorder
  - candidate view:
    - `reshape((64, 2, 2, 32)).permute((0, 2, 1, 3)).reshape((64, 4, 32))`
  - rationale:
    - this reorders quarter bands as `[0, 2, 1, 3]`, matching the old expected
      physical-half-band behavior much better than direct slices
  - throwaway probes:
    - `/tmp/probe_blockm64_reorder.py`
    - `/tmp/probe_blockm64_reorder_splitbands.py`
  - result:
    - direct stores to the reordered views fail cleanly with
      `required row anchors 32,64 are not directly representable in the descriptor view`
    - one broader rank-3 variant also exposed a `dims.size() == 2` assertion in
      `getDistributedLayoutForTmemLdSt(...)` when asking `get_reg_layout()` on
      the intermediate parent view
  - conclusion:
    - the reordered quarter-band view is the most plausible explicit-contract
      direction found so far, but it is not directly lowerable today
- I reverted the experimental test edits after the probes so the worktree
  returned to the clean committed baseline.
- Next useful step:
  - determine whether one more explicit view projection can keep the reordered
    quarter-band contract inside the currently supported direct-view frontier,
    or whether supporting that reordered M64 view requires a compiler fix.
- Follow-up row-split probes after that checkpoint:
  - `/tmp/probe_blockm64_reorder_rowsplit.py`
    - tried to split the M64 tile into `2 x 32` rows before the quarter-band
      reorder and then iterate over the halves
    - failed in Gluon AST lowering because Python list iteration inside the JIT
      kernel is not supported there (`'List' object has no attribute 'func'`)
  - `/tmp/probe_blockm64_reorder_rowsplit2.py`
    - unrolled the two halves explicitly to remove the loop limitation
    - failed at higher-rank TMEM view inference when indexing the reordered
      `2 x 32 x 4 x 32` view:
      - `failed to infer memdesc_index result type`
      - `unsupported tensor memory memdesc_subslice view; preserved tensor memory view encoding also failed: rank must be less than or equal to the memdesc rank for tensor memory`
- Net new conclusion:
  - the explicit rewrite candidate is now blocked at two layers:
    - direct reordered-view ld/st still wants unsupported `32,64` row anchors;
      and
    - the row-split workaround currently hits higher-rank TMEM view/index
      inference limits before lowering.
- Record this as initiative debt, not just failed experimentation:
  - the current TMEM API/compiler/backend surface is still missing coverage for
    the explicit-contract replacement of the old `block_m_64`
    reinterpret-dependent tests;
  - the concrete missing pieces currently exposed are:
    - reordered M64 quarter-band direct views; and
    - higher-rank row-split TMEM reshape/index/subslice inference.
- Future cleanup work should treat those as remembered backlog items when
  resuming the TMEM bug/hack refactor rather than rediscovering them through
  another round of test rewrites.

## 2026-04-10: executed the first broad GB200/NVIDIA CI census and found new non-reinterpret red buckets

- I continued the initiative by moving from the old exact-nodeid-only TMEM
  baseline into the broader GB200 Blackwell lane inventory.
- Local environment status:
  - `make test-lit`
    - broken locally because `LLVM_EXTERNAL_LIT` is empty in `CMakeCache.txt`,
      which makes the generated rule try to run `/llvm-lit`
    - manual fallback `cd $BUILD_DIR && python3 -m lit.main -sv test` passes
  - `make test-proton`
    - blocked locally by missing `llnl-hatchet`
- Confirmed green lanes:
  - `make test-cpp`
    - `240 / 240` passed
  - `make test-gsan`
    - `20 / 20` passed
  - `make test-microbenchmark`
    - passed
- Confirmed new GB200 red lane outside TMEM reinterpret:
  - `make test-regression`
    - `234 failed, 856 passed, 216 skipped`
    - all observed failures were in `python/test/regression/test_cast_matmul.py`
- Additional GB200 red lane from the `test-gluon` examples half:
  - `CUDA_VISIBLE_DEVICES=1 ... pytest -s --tb=short -n 2 python/examples/gluon/`
    - `62 failed, 759 passed, 74 skipped`
  - two visible buckets:
    - `python/examples/gluon/02-convolution.py`
      - repeated `OutOfResources`
      - required shared memory `262208`, hardware limit `232448`
    - `python/examples/gluon/03-matmul-multicta.py`
      - repeated wrong-code around `48%` mismatches
- I then ran the main Gluon/tutorial surface with `pytest --splits 4 --group`
  across four GPUs, excluding only the two documented rewrite-blocked
  `block_m_64` subslice tests.
- Shard results observed during this checkpoint:
  - shard `2 / 4`
    - green
    - `2663 passed, 3784 skipped, 19344 deselected`
  - shard `4 / 4`
    - red
    - `687 failed, 4910 passed, 849 skipped, 19345 deselected`
  - shard `3 / 4`
    - red
    - `1160 failed, 3251 passed, 2036 skipped, 19344 deselected`
  - shard `1 / 4`
    - still running when I wrote this entry
    - no exact failure had appeared yet
- Most important clean exact reruns after the broad shards:
  - stale-negative / broadened-support candidate:
    - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout[identity-64-32x32b-32x32b.x64.b32]`
    - still fails, but only because it no longer raises `CompilationError`
  - non-reproducible in isolation:
    - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout[identity-64-16x64b-16x64b.x32.b32]`
    - passes clean on an isolated rerun
  - non-reproducible in isolation:
    - `python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
    - passes clean on an isolated rerun
  - real split-N TMEM regressions confirmed clean:
    - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-layout9-64-128-32x32b_splitn-8-16x32bx2]`
      - `4096 / 8192` mismatches
    - `python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]`
      - `2048 / 2048` mismatches
- I also captured a separate exact toolchain/codegen bucket from the red Gluon
  shard, but it turned out not to be a new bug:
  - `python/test/gluon/test_core.py::test_tcgen05_mma_plain_kind_i8_reports_clean_error`
    passes clean on an isolated rerun
  - the PTXAS message
    `Feature '.kind::i8' not supported on .target 'sm_103a'`
    is the expected behavior for that negative test
  - the broad shard only looked alarming because `-s` printed the PTXAS
    reproducer
- Net new conclusion:
  - the initiative is no longer just about the old reinterpret-contract tests
  - the current branch tip has a real `M=64` split-N TMEM regression again
  - the higher-rank half-row "clean error" family now mixes stale negatives and
    contaminated shard fallout, so it needs per-nodeid reclassification
  - and there are broader GB200 non-TMEM buckets too:
    - `test_cast_matmul.py` in the regression suite;
    - shared-memory-limit portability failures in the Gluon convolution
      examples; and
    - Gluon multicta matmul wrong-code in the examples lane

## 2026-04-10: fixed the local lit and proton environment blockers and completed the last Gluon shard

- I fixed the local `make test-lit` wrapper by reconfiguring the existing build
  with:
  - `cmake -S . -B $BUILD_DIR -DLLVM_EXTERNAL_LIT=$(which lit)`
- Validation:
  - `make test-lit`
    - now passes
    - `248 passed, 2 unsupported`
- I installed the missing Proton viewer dependency:
  - `python3 -m pip install llnl-hatchet`
- Validation after the install:
  - `CUDA_VISIBLE_DEVICES=1 make test-proton`
    - no longer fails on import
    - now reaches real test failures:
      - `10 failed, 114 passed, 1 skipped`
      - all observed failures in `third_party/proton/test/test_profile.py`
      - failures are concentrated in cudagraph-profile structure / frame-name /
        periodic-flush assertions
- The long-running last Gluon shard also finished:
  - sharded `python/test/gluon/ python/tutorials/gluon/` group `1 / 4`
    - `5433 passed, 1014 skipped, 19344 deselected`
    - no failures
- This completes the broad sharded `test-gluon` census picture for this pass:
  - groups `1 / 4` and `2 / 4` green
  - groups `3 / 4` and `4 / 4` red
- Net new conclusion:
  - the old local environment blockers are no longer hiding those lanes on this
    devbox
  - the remaining Proton failures are real test failures, not a missing-package
    issue
  - and the GB200 `test-gluon` picture is now complete enough to move back to
    the cleanly reproduced split-N TMEM regressions as the next TMEM-local
    debugging target

## 2026-04-10: expanded the GB200 census through test-unit, isolated new standalone buckets, and started merge-base baseline bring-up

- I expanded `make test-unit` into diagnosable pieces:
  - sharded `python/test/unit` across 4 GPUs with
    `pytest --splits 4 --group <n> -n 2 ... --ignore-glob='plugins/*' --ignore=test_debug.py`
  - then ran:
    - `python/test/unit/test_debug.py`
    - `python/triton_kernels/tests/`
    - `python/tutorials/06-fused-attention.py`
    - `python/test/unit/instrumentation/test_gpuhello.py`
    - plugin tests under `python/test/unit/plugins/`
- Results recorded so far:
  - shard `1 / 4`: green (`5048 passed, 114 skipped`)
  - shard `2 / 4`: red (`8 failed, 4507 passed, 647 skipped`)
    - exact failures are the tf32/tf32x3 `python/test/unit/language/test_core.py::test_dot[...]` family
  - shard `3 / 4`: red (`23 failed, 729 passed, 4410 skipped`)
    - concentrated in `python/test/unit/language/test_matmul.py`
  - shard `4 / 4`: red (`131 failed, 4707 passed, 321 skipped`)
    - concentrated in `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[...]`
      and large `python/test/unit/language/test_warp_specialization.py` forward/persistent-forward surfaces
  - `python/test/unit/test_debug.py`: red in both xdist and serial forms (`20 failed, 75 passed`)
  - `python/tutorials/06-fused-attention.py`: green (`192 passed, 192 skipped`)
  - `python/test/unit/instrumentation/test_gpuhello.py`: green (`1 passed`)
  - plugin tests: green
  - `python/triton_kernels/tests/`: still running while I wrote this entry
- Most important isolated reruns:
  - confirmed real standalone wrong-code:
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
      - `49.4%` mismatches
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-1-64-512-32-2-float32-tensorfloat32]`
      - `49.9%` mismatches
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
      - `49.7%` mismatches
  - confirmed real standalone compile/pipeline failures:
    - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float32]`
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-8-64-128-32]`
    - both hit the same `ttng.tmem_store` unsupported-row-anchors `32,64`
      diagnostic chain before `PassManager::run failed`
  - debug-harness classification:
    - `python/test/unit/test_debug.py` is not just outer xdist noise;
    - serial rerun still fails because the test path itself forks and then
      touches CUDA (`py._process.forkedfunc`)
- Branch-surface classification:
  - modified GB200-relevant test files on this branch are only:
    - `python/test/gluon/test_core.py`
    - `python/test/gluon/test_tmem_runtime_matrix.py`
    - `python/test/unit/language/test_compile_only.py`
    - `python/test/unit/language/test_matmul.py`
  - the new `test_dot`, tensor-descriptor, warp-specialization, debug,
    regression, example, and Proton failures are therefore not explained by
    test-file edits alone
- I fetched `origin/main` and confirmed merge-base:
  - `7f61ac734edc657b737fb159a1b9d50cb47944e6`
- I created a detached baseline worktree:
  - `/root/code/triton-mergebase-ci`
- Baseline bring-up status:
  - direct `make` / raw CMake on the old tree hit build-system drift around
    missing `NVWS` tablegen products;
  - I applied local-only baseline shims in the merge-base worktree to skip
    example plugin builds and the legacy GSan runtime; and
  - `python3 setup.py build_ext` now compiles substantially farther, making the
    branch-vs-main comparison path viable once it completes
- Net new conclusion:
  - the GB200 red surface is now clearly larger than the original TMEM
    reinterpret rewrite candidates;
  - there are confirmed new standalone unit/runtime/compiler buckets to
    classify against main;
  - and the baseline-comparison infrastructure is finally moving, but it still
    depends on local-only build shims because the exact merge-base tree does
    not build cleanly against the current host toolchain out of the box

## 2026-04-10: merge-base classification is live, `triton_kernels` is green, and the unit census reduced further

- I finished the remaining easy-firm pieces of the GB200 unit census:
  - `python/triton_kernels/tests/` is green on the current branch
    (`2377 passed, 3444 skipped`)
  - `python/test/unit/language/test_tensor_descriptor.py` is fully reduced to:
    - `test_tensor_descriptor_reshape_matmul[float16]`
    - `test_tensor_descriptor_reshape_matmul[bfloat16]`
    - `test_tensor_descriptor_reshape_matmul[float32]`
    - all three fail with the same unsupported-row-anchors `32,64`
      `ttng.tmem_store` diagnostic chain before `PassManager::run failed`
  - `python/test/unit/test_debug.py` is fully reduced to `20` exact failures:
    - `9` `test_sanitize_int_add_overflow[...]`
    - `6` `test_sanitize_int_mul_overflow[...]`
    - `5` `test_sanitize_int_sub_overflow[...]`
    - all fail because the sanitizer path forks and then re-initializes CUDA
- Merge-base baseline comparison is no longer blocked:
  - the local-only bring-up shims are sufficient to run exact nodeids in
    `/root/code/triton-mergebase-ci`
  - the following exact current-branch failures all pass on merge-base:
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-1-64-512-32-2-float32-tensorfloat32]`
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-8-64-128-32]`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
    - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float32]`
    - `python/test/unit/test_debug.py::test_sanitize_int_add_overflow[-2147483648--1-int32-int32-False-False]`
  - this is now enough to state that the branch has introduced real new-on-top
    GB200 failures outside the old reinterpret-test discussion
- The branch-added TMEM coverage split is also explicit now:
  - exact split-N TMEM nodeids such as
    `test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-...]` and
    `test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-...]`
    do not exist on merge-base, so they should be tracked as branch-added
    coverage that is currently red, not as regressions against old mainline CI
- I left the longer exact inventories running:
  - `python/test/unit/language/test_matmul.py`
  - `python/test/unit/language/test_warp_specialization.py`
  - `python/test/regression/test_cast_matmul.py`
  - `python/examples/gluon/`
- Partial pattern correlation from those in-flight logs already points at one
  broader Blackwell/TMEM regression surface:
  - `test_matmul.py` failures cluster in `BLOCK_M=64`
  - `test_warp_specialization.py` failures cluster in `N=64`
  - `test_cast_matmul.py` failures cluster heavily in `GN=64`

## 2026-04-10: finished most exact branch-vs-main classifications and ruled Proton out of the branch backlog

- I finished the remaining exact reruns for most non-Gluon targets:
  - `python/test/unit/language/test_matmul.py`
    - `23 failed, 738 passed, 4780 skipped`
    - exact split:
      - `18` `test_simple_matmul[...]`
      - `4` `test_simple_persistent_matmul[...]`
      - `1` `test_lhs_in_tmem[...]`
  - `python/test/regression/test_cast_matmul.py`
    - `234` exact failing nodeids
  - `python/examples/gluon/`
    - `62` exact failing nodeids
    - `48` in `02-convolution.py`
    - `14` in `03-matmul-multicta.py`
  - `third_party/proton/test/test_profile.py`
    - `11 failed, 31 passed, 2 skipped`
- Merge-base exact classification is now much sharper:
  - new-on-branch:
    - `test_lhs_in_tmem[float32-False-64-128-32]` passes on merge-base
    - two distant `test_cast_matmul[...]` representatives pass on merge-base
    - a second `test_warp_specialize_attention_forward[...]` representative in
      the `128x64` shape family passes on merge-base
    - representative failures from both `02-convolution.py` and
      `03-matmul-multicta.py` pass on merge-base
  - pre-existing-on-merge-base:
    - the full `third_party/proton/test/test_profile.py` file fails with the
      same `11` nodeids on merge-base after fixing the local-only baseline
      `libproton.so` symlink
- I also ran a shared-cache sanity probe for the suspected cache-collision
  issue:
  - run the known-bad split-N TMEM descriptor-chain test
  - then run the known-good `test_block_m_64_mma[linear]`
  - keep `TRITON_CACHE_DIR` identical but use fresh processes
  - result:
    - the failing split-N test still fails
    - `test_block_m_64_mma[linear]` still passes immediately afterward
  - conclusion:
    - this does not reproduce simple on-disk cache poisoning
    - the earlier shard-only anomalies are more likely process/device
      contamination after a bad kernel than a trivial cache-key collision
- Remaining current-branch exact inventories still running when I wrote this:
  - `python/test/gluon/ python/tutorials/gluon/` shard `3 / 4`
  - `python/test/gluon/ python/tutorials/gluon/` shard `4 / 4`

## 2026-04-10: completed `test-gluon` shard 4 and tied the new red surface back to the row-plan / canonical-M64 planner

- I finished the previously in-flight `python/test/gluon/ python/tutorials/gluon/`
  shard `4 / 4`:
  - `686 failed, 4911 passed, 849 skipped, 19345 deselected`
  - every failure is in the branch-added file
    `python/test/gluon/test_tmem_runtime_matrix.py`
- The dominant runtime-matrix families in the completed shard are:
  - `224` `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[...]`
  - `48` `test_tmem_runtime_matrix_ld_red_row_permuted_linear_layout[...]`
  - `33` `test_tmem_runtime_matrix_ldst_scales_variant_sweep[...]`
  - `32` `test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_geometry_sweep[...]`
  - `30` `test_tmem_runtime_matrix_cp_no_scales[...]`
- That means the branch-added TMEM runtime-matrix surface is broadly red right
  now; it is not just a stale-negative or clean-error expectation cleanup
  story anymore.
- I also recorded the strongest current branch-vs-merge-base PTX clue so far:
  - for representative `BLOCK_M=64` tf32 matmul wrong-code, the forced
    `16x256` variant still chooses the same family on both sides, but the
    current branch collapses the second support band into contiguous packet
    offsets (`+0,+128,+256,+384`) while merge-base uses the lifted offsets
    (`+0,+128,+1048576,+1048704`);
  - and the non-forced sibling shifts family selection from merge-base
    `32x32b.x64` to current-branch `16x32bx2.x64`.
- Best current interpretation:
  - the new row-plan-aware canonical-`M=64` family selection / packet
    decomposition is collapsing a support-band / lifted-image quotient bit into
    ordinary packet repetition
  - this is the most plausible common root cause across the split-N TMEM
    regressions, the new runtime-matrix failures, and the broader
    tf32/tf32x3 matmul/dot wrong-code buckets

## 2026-04-10: finished the current-branch GB200 census and converted more of the backlog from samples into full branch-vs-main facts

- The current-branch GB200/NVIDIA CI census is now complete:
  - `make NUM_PROCS=24 test-unit`
    - `162 failed, 14991 passed, 5492 skipped`
    - exact-family split from the completed xdist log:
      - `64` `test_warp_specialize_attention_forward[...]`
      - `64` `test_warp_specialize_attention_persistent_forward[...]`
      - `18` `test_simple_matmul[...]`
      - `8` `test_dot[...]`
      - `4` `test_simple_persistent_matmul[...]`
      - `3` `test_tensor_descriptor_reshape_matmul[...]`
      - `1` `test_lhs_in_tmem[...]`
  - `test-gluon` shard `3 / 4`
    - `1160 failed, 3251 passed, 2036 skipped, 19344 deselected`
    - file split:
      - `793` `python/test/gluon/test_lowerings.py`
      - `293` `python/test/gluon/test_core.py`
      - `72` `python/test/gluon/test_fpsan.py`
      - `1` `python/test/gluon/test_frontend.py`
      - `1` `python/test/gluon/test_layout_format_view.py`
    - dominant families:
      - `603` `test_reduce_layouts[...]`
      - `176` `test_scan_layouts[...]`
      - `64` `test_mma_scaled_tcgen05_copy[...]`
      - `40` `test_tmem_reduction[...]`
      - `32` `test_tmem_reduction_linear_layouts[...]`
- Merge-base full-file comparisons got materially sharper:
  - `python/test/unit/language/test_tensor_descriptor.py`
    - merge-base is fully green (`2604 passed, 110 skipped`)
    - therefore all three current-branch
      `test_tensor_descriptor_reshape_matmul[...]` failures are new-on-branch
  - `python/test/regression/test_cast_matmul.py`
    - merge-base is fully green (`1080 passed, 216 skipped`)
    - therefore the current-branch `234`-failure regression bucket is entirely
      new-on-branch
- I also split the completed `test-gluon` shard-3 failures into
  merge-base-existing vs branch-added/renamed names without running any extra
  GPU work:
  - `python/test/gluon/test_core.py`
    - `203` failures are on functions that already exist on merge-base
    - `90` failures are branch-added or renamed coverage
  - `python/test/gluon/test_fpsan.py`
    - `60` failures are on functions that already exist on merge-base
    - `12` failures are branch-added or renamed coverage
  - `python/test/gluon/test_frontend.py::test_tensor_memory_linear_layout_non_surjective_reg_layout_parses`
    - missing on merge-base, so branch-added
- This changes the recovery shape:
  - the branch-caused backlog is no longer just TMEM runtime-matrix and a few
    unit tests
  - it now includes broad new-on-branch matmul / warp-specialization /
    lowering / reduction surfaces that still need exact merge-base reduction
    before fix ordering is frozen

## 2026-04-10: materialized durable exact-failure manifests in the initiative folder

- I generated durable exact nodeid lists from the saved current-branch logs:
  - `gb200_current_branch_test_unit_failures.txt` (`182`)
  - `gb200_current_branch_test_gluon_group3_failures.txt` (`1160`)
  - `gb200_current_branch_test_gluon_group4_failures.txt` (`686`)
  - `gb200_current_branch_test_regression_failures.txt` (`234`)
  - `gb200_current_branch_examples_gluon_failures.txt` (`62`)
  - `gb200_current_branch_test_proton_failures.txt` (`11`)
- I also materialized the shard-3 merge-base split lists:
  - `gb200_mergebase_existing_test_gluon_core_group3_failures.txt` (`203`)
  - `gb200_branch_added_or_renamed_test_gluon_core_group3_failures.txt` (`90`)
  - `gb200_mergebase_existing_test_gluon_fpsan_group3_failures.txt` (`60`)
  - `gb200_branch_added_or_renamed_test_gluon_fpsan_group3_failures.txt` (`12`)
  - `gb200_mergebase_existing_test_gluon_layout_format_view_group3_failures.txt` (`1`)
  - `gb200_branch_added_or_renamed_test_gluon_frontend_group3_failures.txt` (`1`)
- I added `gb200_failure_manifest.md` as the human-readable index explaining
  what each list is, how to rerun one nodeid, and how to rerun a whole
  generated exact list on either the current branch or the merge-base worktree.
- This means future sessions no longer depend on transient `/tmp` logs just to
  recover the exact failing nodeids from the completed GB200 census.

## 2026-04-10: closed almost all remaining branch-vs-main ambiguity

- I finished the remaining merge-base exact reductions that matter for the GB200
  backlog:
  - `python/test/unit/language/test_matmul.py`
    - fully green on merge-base (`761 passed, 4780 skipped`)
  - `python/test/unit/language/test_warp_specialization.py`
    - fully green on merge-base (`1599 passed, 202 skipped`)
  - `python/test/unit/test_debug.py`
    - same `20` exact nodeids still fail on merge-base
  - `python/test/gluon/test_lowerings.py`
    - fully green on merge-base (`4937 passed, 512 skipped`)
  - exact `test_dot[...]` subset from the current-branch unit manifest:
    - `8 / 8` pass on merge-base
  - exact current-branch `examples/gluon` failure manifest:
    - `62 / 62` pass on merge-base
  - exact merge-base-present shard-3 subsets:
    - `185 / 185` `test_core.py` nodeids pass
    - `38 / 38` `test_fpsan.py` nodeids pass
    - `1 / 1` `test_layout_format_view.py` nodeid passes
- I also refined the shard-3 split beyond function-name existence using
  merge-base `--collect-only`:
  - `test_core.py`
    - `203` function-name-existing nodeids split into:
      - `185` exact nodeids present on merge-base
      - `18` exact nodeids missing on merge-base because the branch changed the
        parametrization surface inside existing functions
  - `test_fpsan.py`
    - `60` function-name-existing nodeids split into:
      - `38` exact nodeids present on merge-base
      - `22` exact nodeids missing on merge-base for the same reason
- Net result:
  - the vast majority of the finished GB200 red surface is now either
    confirmed `REAL_NEW_ON_BRANCH_REGRESSION`,
    confirmed `PREEXISTING_ON_MERGE_BASE`, or
    clearly `BRANCH_ADDED_OR_CHANGED_COVERAGE`
  - the inventory/classification phase is effectively done; the next phase is
    fix ordering and implementation against the now-frozen branch backlog
