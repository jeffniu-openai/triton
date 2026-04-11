# TMEM Linear Generalization Initiative

This directory is the durable coordination space for the TMEM linear-layout
generalization project.

Keep this README up to date when the role of any document changes, when a new
current-state handoff supersedes an older one, or when the source-of-truth
entry points change.

## How To Use This Folder

When resuming the initiative:
- start here;
- then read `memory.md` as the initiative's durable memory/source of truth;
- use `gb200_nvidia_ci_inventory.md` for the current GB200/NVIDIA CI baseline
  and confirmed red list;
- use `gb200_failure_manifest.md` plus the adjacent generated `.txt` files
  when you need the exact nodeid inventories for the current branch or the
  merge-base-existing shard-3 subsets;
- use `gb200_branch_recovery_plan.md` when you need the prioritized
  branch-caused recovery backlog and the exact representative repros to drive
  fixes;
- then read the tail of the active handoff for the latest live buckets and next
  concrete steps;
- use `log.md` for chronological provenance and validation history; and
- use `fuzz_plan.md` once the current planner/cleanup blockers are green and
  broad saturation work is the next task.

## Current Project Invariant

- `_reinterpret` is not an escape hatch for relying on whichever physical TMEM
  lowering the compiler happens to choose.
- Production kernels, examples, and tests that currently depend on private
  `_reinterpret` behavior or compiler-specific physical-layout knowledge are
  migration targets.
- The intended replacement is a supported descriptor operation sequence:
  offset to the right TMEM region, slice/subview to the desired physical bits,
  then bitcast to the desired dtype/shape/layout only when total bit size and
  the exact physical TMEM mapping are preserved.
- The bitcast step must not change which physical TMEM memory the input
  descriptor maps to; if the desired view is not equal-size and
  physical-mapping equivalent, use a different supported API that matches the
  real kernel intent.
- Lowering-side fixes are still appropriate for supported APIs that miscompile,
  but do not add ad-hoc selectors just to preserve old `_reinterpret`
  accidents.

## Current Checkpoint

- As of 2026-04-11 16:30 UTC, the latest pushed source/test checkpoint is
  `c83d358e4` on `origin/codex/tmem`.
- The latest full four-way `python/test/gluon` sweep remains the green sweep
  recorded at `77c43f696` / source `be3cba0cd`; the newer `57a06c29b` and
  copy / `ld.red` / scaled-MMA coverage slices were validated with focused
  direct-i8 MMA tests, adjacent positive `tcgen05_mma` kind tests, the
  Blackwell conversion lit test, focused copy and `ld.red` runtime exacts, and
  the focused scaled-MMA opcode matrix.
- A fresh four-way `python/test/gluon` sweep from `be3cba0cd` is green:
  - group 1:
    `5448 passed, 1002 skipped, 19348 deselected`
  - group 2:
    `2666 passed, 3784 skipped, 19348 deselected`
  - group 3:
    `4408 passed, 2041 skipped, 19348 deselected, 1 xfailed`
  - group 4:
    `5582 passed, 866 skipped, 19350 deselected`
  - the current `gb200_current_branch_group{1,2,3,4}_latest_failures.txt`
    manifests are refreshed to `0` nodeids.
- The broad-sweep `.kind::i8` PTX assembler diagnostics are now addressed for
  the direct Gluon API path by `57a06c29b`: on `sm_103a+`,
  direct `tcgen05_mma` i8/u8 inputs with an int32 accumulator fail before IR
  lowering with a clean frontend diagnostic, while the generic
  `compute-capability=100` `ttng.tc_gen5_mma` i8 conversion coverage remains
  intact.
- The direct canonical TMEM-linear `128x128b` copy root coverage gap is closed
  by `c5bdb6d5c`:
  - `test_tmem_runtime_matrix_cp_128x128` now runs both the legacy
    `TensorMemoryLayout` destination and canonical `_make_tmem_linear_layout`
    destination;
  - both variants assert the exact `tcgen05.cp.cta_group::1.128x128b` PTX/LLIR
    family;
  - the linear variant also checks the TTGIR contains `tensor_memory_linear`.
- The standalone scaled `warpx4.32x128b` copy validation no longer depends on
  `_reinterpret` after `bf3dd781b`:
  - `test_tmem_copy_2d`, `test_tmem_runtime_matrix_cp_scales_warpx4`, and the
    scales layout probe validate the copied data through supported logical
    `TensorMemoryScalesLayout` loads;
  - the tests still assert exact `tcgen05.cp.cta_group::1.warpx4.32x128b`
    PTX/LLIR selection;
  - they now assert no `ttg.memdesc_reinterpret` appears in TTGIR for those
    kernels.
- Do not treat scales physical-inspection aliases as ordinary equal-size
  bitcasts: a direct `bitcast(...)` replacement exposed that the old dense
  inspection view was larger than the logical scales descriptor. Use logical
  scales loads for this test intent, and reserve physical bitcasts for the
  explicit offset/slice/subview cases where total size and physical mapping are
  equivalent.
- `128x256b` indexed-view copy coverage is broadened by `7075f31fc`:
  - the fit positive view shape `[128, 128]` now covers f32 and i32 payloads
    across 32, 64, and 128-byte shared swizzles;
  - the full `[2, 128, 256]` indexed parent is recorded as a tensor-memory OOR
    boundary (`Required: 1024, Hardware limit: 512`) instead of a positive
    target.
- `ld.red` modifier saturation is expanded by `eac11c719`:
  - positive identity, tile-permuted, column-permuted, and row-permuted linear
    layouts now cover all four legal modifier pairs:
    plain, propagate-NaN, abs, and abs+propagate-NaN;
  - mixed unsupported layouts check the same modifier matrix and continue to
    report the dedicated `tmem_load reduction source layout is not directly
    tcgen05.ld.red-compatible` diagnostic.
- Scaled-MMAv5 copy-matrix instruction coverage is tightened by `3374bbf12`:
  - the runtime-matrix scaled copy tests still assert exact
    `tcgen05.cp.cta_group::{1,2}.warpx4.32x128b` PTX/LLIR streams;
  - the same format/CTA/geometry/accumulator-layout matrix now also asserts
    exact `tcgen05.mma.cta_group::{1,2}.kind::{mxf8f6f4,mxf4,mxf4nvf4}.block_scale.scale_vec::*`
    opcode selection for PTX and LLIR.
- Direct scaled-MMAv5 TMEM-view tests now pin exact `mxf8f6f4` scaled-MMA
  opcodes after `5e3b2ae87`:
  - this covers the existing minimal, block-N direct-layout, accumulator
    subview, LHS subview, and tile-permuted accumulator runtime tests;
  - the original e5m2/e5m2 direct-view kernels remain in place.
- Direct scaled-MMAv5 accumulator-subview format coverage is broadened by
  `602fd9b44`:
  - the new direct subview matrix covers `mxfp8/mxfp8`, `mxfp4/mxfp4`,
    `mxfp8/mxfp4`, `mxfp4/mxfp8`, and `nvfp4/nvfp4`;
  - it validates numeric output, exact PTX/LLIR
    `mxf8f6f4` / `mxf4` / `mxf4nvf4` scaled-MMA opcodes, and the
    `ttg.memdesc_subslice` + `tensor_memory_linear` accumulator path.
- Plain MMAv5 kind saturation now includes f16 in the explicit 1-CTA and
  2-CTA kind matrices after `7766be003`:
  - `MMA_PLAIN_KINDS` covers `f16`, `tf32`, `bf16`, `f8e5m2`, and `f8e4m3`;
  - both legacy and canonical linear accumulator layouts are covered for
    `cta_group::1` and `cta_group::2`;
  - the tests continue to assert exact PTX/LLIR opcode agreement.
- Staged `ld/st` fuzzing has started with descriptor-chain `auto` coverage at
  `ebb23b697`:
  - the basic descriptor-composition matrix now covers `auto` instruction
    selection in addition to explicit `32x32b`, `16x64b`, `16x128b`, and
    `16x256b`;
  - coverage spans identity and mixed TMEM-linear layouts at `N = 64, 128,
    256`.
- Two-CTA descriptor-chain `auto` coverage is added by `ca9f94760`:
  - `LDST_TWOCTA_DESCRIPTOR_CASES` now uses the same `LDST_VARIANTS` matrix;
  - coverage reaches both `block_two_ctas` and MMAv5-like two-CTA layouts;
  - the validation also reran the single-CTA descriptor roundtrip sweep now
    reached by `LDST_DESCRIPTOR_CASES`.
- Higher-rank `ld/st` index descriptor views now include `auto` instruction
  selection after `7df887318`:
  - auto subview shapes are pinned to the expected `32x32b` family;
  - single-CTA identity/mixed and two-CTA block/MMAv5-like layouts are covered
    for `N = 64, 128`.
- Higher-rank multidimensional-slice `ld/st` descriptor views now include
  `auto` instruction selection after `81b4e5a9d`:
  - single-CTA identity/mixed and two-CTA block/MMAv5-like layouts are covered
    for `N = 64, 128`;
  - the tests continue to check functional output, PTX/LLIR agreement, and the
    expected reshape/transpose/split/join IR markers.
- Higher-rank clean OOR guards now include `auto` after `e2d793726`:
  - single-CTA and two-CTA higher-rank index, multidimensional-slice, and
    dim0-slice resource-boundary tests all use `LDST_VARIANTS`;
  - validation confirms these remain clean tensor-memory OOR boundaries.
- Higher-rank half-row clean errors now include `auto` after `7c1a6f63b`:
  - single-CTA and two-CTA lifted half-row clean-negative matrices now use
    `LDST_VARIANTS`;
  - explicit variants keep the existing unsupported-layout `CompilationError`
    contract, while `auto` may report the direct descriptor parser
    `RuntimeError` about translated row-half TMEM origins;
  - the tests still require a descriptor-view diagnostic and guard against
    PassManager/assertion crashes.
- Higher-rank dim0-slice positive `ld/st` views now include `auto` after
  `572fb25b2`:
  - single-CTA identity and two-CTA block lifted-layout dim0-slice matrices now
    use `LDST_VARIANTS`;
  - the tests validate functional output plus full-tile and narrowed-subview
    PTX/LLIR opcode agreement.
- The two-CTA MMAv5-like higher-rank `ld/st` clean-negative matrix now includes
  `auto` after `8f622b938`:
  - the unsupported matrix uses `LDST_VARIANTS`;
  - validation confirms auto still fails through the clean CTA-per-CGA mismatch
    path rather than PassManager/assertion crashes.
- Same-row/column permuted `ld/st` layouts now include `auto` after
  `c83d358e4`:
  - `LDST_PERMUTED_CASES` uses `LDST_VARIANTS`;
  - this reaches both direct ld/st and descriptor-chain composition runtime
    tests;
  - validation covers numerical behavior, `tensor_memory_linear` IR, and exact
    PTX/LLIR opcode agreement.
- The supported M64 subview/physical-bitcast slice is now checkpointed:
  - `47a07a37d` added normalized source-query inversion for physical bitcast
    views whose source subview keeps inactive zero support bases;
  - `85d8dbbf4` preserved out-of-range physical row/col origins for narrowed
    subviews, migrated the M64 `_reinterpret` tests to supported
    `slice/subview -> bitcast` where the mapping is equivalent, and made
    parameterized kernels pass layouts as explicit constexpr arguments so
    Gluon cache keys distinguish legacy and linear variants.
- The wider grouped `python/test/gluon` group-3 rerun that started before
  these two checkpoints is now superseded by the green group-3 rerun above. It
  had reduced the stale current-head red set to:
  - `test_tmem_subslice_block_m_64_parent_layout[linear]`;
  - `test_block_m_64_mma[legacy]`;
  - `test_block_m_64_mma[linear]`;
  - result: `3 failed, 4406 passed, 2041 skipped, 19348 deselected`.
- Focused current-head reruns after `85d8dbbf4` close that exact M64 set:
  - subview/physical-bitcast exacts:
    `4 passed`;
  - `test_block_m_64_mma[legacy]` / `[linear]`:
    `1 passed, 1 xfailed`;
  - the xfail is deliberate design debt for legacy M64 `64x64` layout sugar
    lacking producer-visible physical-family semantics for MMAv5 consumers;
    the linear supported layout passes in the same process.
- The supported descriptor bitcast API remains on the branch, but the
  attention example migration has been intentionally reverted for now:
  - `python/examples/gluon/01-attention-forward.py` is back to its pre-bitcast
    `_reinterpret` form;
  - the representative attention exact is red again with
    `LLVM ERROR: Invalid basis 32 for in-dim 'col' and out-dim 'dim1'`;
  - the old examples/Gluon green aggregate at `4263ae61` / `49f1a0fd` is
    stale for current `HEAD`;
  - future attention work must use a synchronization-aware supported
    `offset/slice/subview -> bitcast` sequence, not private `_reinterpret`
    behavior or lowering selectors.
- The stale `test_mma_shared_inputs` two-CTA TTGIR spelling bucket is closed:
  - current TMEM encoding text may spell the type field as `twoCTAs`, while
    physical-layout and op attributes still use `two_ctas`;
  - the assertion now accepts either spelling;
  - patched four-way `test_mma_shared_inputs` refresh:
    `3830 passed, 490 skipped`, `2954 passed, 1366 skipped`,
    `1206 passed, 3114 skipped`, and `2584 passed, 1736 skipped`.
- The focused persistent `python/triton_kernels/tests/test_matmul.py`
  shared-memory OOR bucket is fixed, and the full directory now has a green
  current-head rerun:
  - root cause was ordinary replayed `ttng.tmem_subslice` loads/stores using
    the exact preserved physical query and therefore selecting a root-width
    register family for a shape-local replay slice;
  - replayed subslices now use the subview-local descriptor type for register
    layout selection;
  - only explicit `tmem_physical_bitcast` roots use the preserved physical
    query, keeping the supported physical bitcast contract intact.
- Validation for the latest code checkpoint:
  - build: `PASSED`
  - lit `tmem_layouts.mlir` and `tritongpu_to_llvm_blackwell.mlir`:
    `2 passed`
  - representative persistent matmul OOR repro:
    `1 passed`, metadata back to `shared=214120`
  - focused persistent fp8/mxfp4 matmul slice:
    `16 passed, 6 skipped`
  - full 4-way `python/triton_kernels/tests` refresh:
    `2377 passed, 3444 skipped`
- The M64 row/col-permuted split-N direct ld/st bucket remains closed as a
  positive hardware surface:
  - exact row-permuted `16x32bx2` warp anchors that would lower to misaligned
    one-row PTX addresses are rejected;
  - simple row/col-permuted `64xN` f32 TMEM-linear roots select the canonical
    aligned M64 split-N register layout;
  - backend default-layout selection has the same fallback, so conversions do
    not hit the empty-compatible-layout assertion.
- The full `python/test/gluon/test_tmem_runtime_matrix.py` file was refreshed
  from this branch:
  - full-file rerun before the final assertion refresh:
    `1437 passed, 354 skipped, 1 failed in 1226.69s`;
  - the only failure was a stale TTGIR text assertion in
    `test_tmem_runtime_matrix_cp_no_scales_indexed_view_canonicalized[128]`;
  - that test now checks the stable `ttng.tmem_physical_layout` marker on the
    legacy `tensor_memory_encoding` view path, and the exact nodeid rerun
    passes.
- The old `413`-nodeid branch-added runtime-matrix manifest is now refreshed
  to empty:
  - `gb200_current_branch_test_tmem_runtime_matrix_focus_e70a3aa09_failures.txt`
- The old `332`-nodeid focused `test_core.py` TMEM/MMA manifest is also now
  refreshed to empty:
  - the full focused manifest rerun passed:
    - `332 passed in 125.78s`
  - the only intermediate issues were stale expectations for:
    - the row-encoded `1048576` PTX offset in the legacy `64x128` split-N
      roundtrip;
    - the current clean unsupported descriptor-view diagnostic for a block
      basis `ld.red` negative.
  - `gb200_current_branch_test_core_group3_focus_e70a3aa09_failures.txt`
- The supported `_reinterpret` migration invariant remains:
  - offset to the right part of TMEM;
  - slice/subview it to the desired physical bits;
  - bitcast to the desired dtype/shape/layout only when equal-size and
    physical-mapping equivalent to the input descriptor.
- Next required durable step: continue the remaining recovery queue with the
  intentionally reverted attention example tracked separately from supported
  bitcast API validation and the legacy M64 MMAv5 xfail visible as design debt.
  Continue broader MMAv5/`mma_scaled`, staged `ld/st` fuzzing, stale-negative
  cleanup, and heuristic phases, with any remaining copy work limited to
  shapes that fit the TMEM allocation budget.

## Document Roles

- `memory.md`
  - durable memory for the initiative
  - source of truth for mission, current decisions, long-term plan, and current
    undated priorities
  - future sessions should update this when the stable plan or current
    priorities change

- `log.md`
  - append-only chronology of checkpoints, bug classification, validation, and
    conclusions
  - do not rewrite old entries except for obvious factual corrections

- `handoff_2026-04-09.md`
  - current active handoff
  - read its tail first for the latest live buckets, disproved experiments, and
    next implementation order
  - keep it updated at meaningful checkpoints and before likely context
    rollover

- `gb200_nvidia_ci_inventory.md`
  - current GB200/NVIDIA CI baseline for this Blackwell devbox phase
  - records the workflow coverage, confirmed red list, and classification of
    each current failure as stale expectation, rewrite-candidate, or real bug
  - update it before broadening beyond exact nodeids or changing the current
    CI-grounded failure story

- `gb200_failure_manifest.md`
  - index for the generated exact current-branch failure lists and the
    merge-base-existing shard-3 subsets
  - use the adjacent `.txt` files when you need exact nodeids rather than
    prose summaries
  - refresh the manifests whenever the current-branch failure surface changes
    materially

- `gb200_branch_recovery_plan.md`
  - prioritized recovery backlog for restoring GB200 CI parity with main
  - separates real branch regressions from preexisting failures, stale
    negatives, and reinterpret-contract rewrite candidates
  - use it to choose the next fix slice once the GB200 census is current

- `fuzz_plan.md`
  - operational saturation/fuzzing playbook for the broader `tcgen05` surface
  - not the immediate source of truth for current bug-fix priority while the
    shared-planner / legacy-unification / reinterpret-decomposition work is
    still active

- `handoff_2026-03-26.md`
  - historical handoff from an earlier phase of the initiative
  - archival provenance, not current status

- `next_agent_prompt_2026-03-26.md`
  - archived takeover prompt from an earlier managed-session workflow
  - archival only

## Current Session Discipline

- Keep `memory.md`, the active handoff, and this README aligned.
- Treat `memory.md` as memory: if the stable understanding of the project or
  the current undated execution priorities changes, update `memory.md` rather
  than leaving that knowledge only in chat context.
- Keep dated historical analyses in `log.md` and the dated handoff files
  instead of rewriting old reports in place.
