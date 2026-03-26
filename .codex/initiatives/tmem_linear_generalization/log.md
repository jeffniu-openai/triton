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
