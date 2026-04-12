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
- Operationally, the offset/subview step selects the actual physical TMEM bits
  first; the bitcast only changes the descriptor's dtype/shape/layout view over
  that already-selected physical image.
- This means the bitcast may express the desired dtype/shape/layout only after
  the input descriptor already denotes exactly the same physical TMEM image; it
  must not perform physical selection itself.
- In the concrete attention-style pattern, the kernel first offsets to the
  right part of TMEM, then takes the subview that denotes the exact desired
  bits, then performs a size-and-physical-mapping-equivalent bitcast to the
  desired dtype, shape, and layout.
- Equivalently: offset to the right part of TMEM, slice/subview to the desired
  bit image, then bitcast only to a dtype/shape/layout view whose total size
  and physical mapping are equivalent to that sliced descriptor.
- The bitcast step must not change which physical TMEM memory the input
  descriptor maps to; if the desired view is not equal-size and
  physical-mapping equivalent, use a different supported API that matches the
  real kernel intent.
- Said another way, this is an offset/subview over the existing physical TMEM
  allocation followed by a size-and-physical-mapping-equivalent type/view
  reinterpretation, not a remapping or relocation operation.
- The attention example's scratch-borrow path has a focused supported
  migration: helpers now offset/slice the f32 scratch TMEM before bitcasting,
  and the exp2 partition stores slice the exact physical subregion before
  bitcasting each bf16 view. Preserve this synchronization-aware pattern if
  broader attention coverage exposes more aliasing cases.
- Lowering-side fixes are still appropriate for supported APIs that miscompile,
  but do not add ad-hoc selectors just to preserve old `_reinterpret`
  accidents.

## Current Checkpoint

- Full GB200 `integration-tests-nvidia` inventory at `cb76c31a0` is not green:
  - branch-new actionable failures are `9` lit files, `162` `python/test/unit`
    nodeids, and `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`;
  - those exacts pass on merge-base `11ee1144a737006921231bbd3386c187812c38e1`;
  - Proton's `11` cudagraph / periodic flushing failures reproduce on
    merge-base and are ignored as preexisting GB200 lane noise.
- The latest current-head `python/examples/gluon` aggregate passed all four
  split groups, so `python/examples/gluon/01-attention-forward.py` is now
  covered by the examples aggregate as well as its focused SDPA-backed matrix.

- The supported `slice/subview -> bitcast` contract now has an explicit
  root-aligned and right-half regression in
  `test_tmem_physical_bitcast_preserves_subview_mapping`: the right-half case
  first selects columns `64:128` of the original f32 TMEM tile, then bitcasts
  that already-selected physical image to an f16 view, and verifies only the
  selected physical half is overwritten.
- Direct subword `ld/st` instruction-selection coverage now spans `f16`,
  `bf16`, `i16`, and `i8` over identity `128x{64,128,256}` layouts and every
  public variant in `{auto,32x32b,16x64b,16x128b,16x256b}`.
- Narrow x1 subword `ld/st` coverage now spans `f16`, `bf16`, and `i16`
  packed/unpacked cases plus packed and padded `i8` x1 cases for `auto` and
  `32x32b`.
- As of the latest focused coverage checkpoint, the current source/test slices
  add runtime-matrix allocator-lifetime anchors for `tcgen05.alloc`,
  `relinquish_alloc_permit`, `dealloc`, and `wait`, including pow2 alloc-size
  immediates for both CTA groups, plus exact `tcgen05.commit` opcode anchors.
- The latest full `python/test/gluon/test_tmem_runtime_matrix.py` file
  validation checkpoint remains green:
  - `1757 passed, 442 skipped in 1640.11s (0:27:20)`
- The preferred four-way heavy Gluon sweep over
  `python/test/gluon/test_core.py` and
  `python/test/gluon/test_tmem_runtime_matrix.py` is green at `be14fedc5`:
  - group 1:
    `4407 passed, 634 skipped, 15123 deselected`
  - group 2:
    `2528 passed, 2513 skipped, 15123 deselected`
  - group 3:
    `2694 passed, 2347 skipped, 15123 deselected`
  - group 4:
    `3314 passed, 1726 skipped, 15123 deselected, 1 xfailed`
- The preferred four-way `python/triton_kernels/tests` sweep is green at
  `2ad0ccf5e`:
  - group 1:
    `674 passed, 782 skipped, 4365 deselected`
  - group 2:
    `533 passed, 923 skipped, 4365 deselected`
  - group 3:
    `394 passed, 1062 skipped, 4365 deselected`
  - group 4:
    `776 passed, 677 skipped, 4368 deselected`
  - aggregate:
    `2377 passed, 3444 skipped, 17463 deselected`
- The clean Gluon examples subset remains green at `24bec4ecf`, and the
  attention file now has SDPA-backed benchmark-parameter unit coverage at
  current head:
  - `python/examples/gluon/01-attention-forward.py`:
    `64 passed in 102.66s (0:01:42)` over the benchmark grid except the
    SDPA-OOM context lengths (`N_CTX in {1024,2048,4096,8192}` for tests;
    benchmark remains `2**10..2**16`);
  - covered test parameters: `Z=4`, `H=32`, `HEAD_DIM in {64,128}`,
    `causal in {False,True}`, providers `triton-fp16` and `triton-fp8`, and
    `use_tmem_red in {False,True}` on this Blackwell Ultra box;
  - `python/examples/gluon/02-convolution.py`:
    `48 passed`
  - `python/examples/gluon/03-matmul-multicta.py`:
    `82 passed, 14 skipped`
  - `python/examples/gluon/04-2cta-block-scale-matmul.py`:
    `690 passed, 60 skipped`
  - full `python/examples/gluon` was rerun in the 2026-04-12 GB200 sweep at
    `cb76c31a0` and passed all four split groups.
- The current-head runtime-matrix saturation slices are green:
  - broad `ld/st`:
    `1181 passed, 441 skipped, 1027 deselected`
  - broad `tcgen05.cp`:
    `162 passed, 5 skipped, 2488 deselected`
  - broad `tcgen05.ld.red`:
    `487 passed, 2183 deselected`
  - true `tcgen05.mma` / direct `mma_scaled`:
    `226 passed, 50 skipped, 2407 deselected`
  - scaled-MMA copy-helper matrix:
    `52 passed, 2147 deselected` with exact copy, MMA, and commit opcode checks
- Allocator/lifetime coverage now has explicit runtime anchors:
  - single-CTA and two-CTA ld/st kernels assert exact PTX/LLIR
    `tcgen05.alloc`, `tcgen05.relinquish_alloc_permit`, `tcgen05.dealloc`,
    and `tcgen05.wait::{st,ld}` emission;
  - both CTA groups cover alloc/dealloc size immediates
    `32, 64, 128, 256, 512`;
  - the two-CTA case asserts cluster arrive/wait before dealloc;
  - a source-initialized `allocate_tensor_memory(..., value=...)` kernel
    round-trips the initialized values through a TMEM load.
- Tensor-memory allocation rounding coverage now pins raw live TMEM totals
  `96`, `192`, and `384` as rounded boundaries to supported module allocation
  sizes `128`, `256`, and `512`; literal non-pow2 `tcgen05.alloc` immediates
  are not the compiler contract.
- Commit-opcode coverage now has exact PTX/LLIR anchors:
  - single-CTA MMA pins
    `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
  - two-CTA copy and MMA pin
    `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`.
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
- No-scales `warpx2` copy exact coverage now includes:
  - single-CTA `warpx2::{01_23,02_13}.64x128b` candidate positives with exact
    `cta_group::1` copy and commit opcode checks plus runtime oracles;
  - two-CTA `warpx2::01_23.64x128b` executable coverage through public
    `SharedLinearLayout` + `TensorMemoryLinearLayout` objects, with exact
    `tcgen05.cp.cta_group::2.warpx2::01_23.64x128b` and multicast commit
    opcode checks;
  - dense shared-layout `warpx2` forms are now clean unsupported for both
    single-CTA and two-CTA cases because runtime probes showed the old
    codegen-only path emitted `warpx2` opcodes while copying the wrong data;
  - the two-CTA `warpx2::02_13` candidate shared layout is now pinned as a
    clean unsupported descriptor-synthesis frontier under the canonical public
    shared layout; it remains a layout-surface / direct-PTX frontier, not a
    proven ISA-impossible negative;
  - a 2026-04-11 temporary direct-seed relaxation for the canonical two-CTA
    `warpx2::02_13` source emitted
    `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b` but produced all-zero
    output, so the current clean unsupported boundary is protecting a real
    wrong-code path and should not be removed without a better descriptor/address
    model.
  - a follow-up direct-PTX patch sweep is recorded in
    `experiments/probe_cp_warpx2_02_13_twocta_direct_ptx.py` with results in
    `experiments/results/probe_cp_warpx2_02_13_twocta_direct_ptx_current.log`:
    opcode-only, source-row-plus-16, single-CTA seed, destination `+4`, and two
    message variants all assemble and launch through the cluster-aware Triton
    launcher, but none match the extended single-CTA `02_13` oracle; the
    descriptor/address mutations either duplicate source-column pairs or copy
    the wrong row/column mix, so they still do not recover the missing 4-byte
    source-column bit.
- The historical scales `warpx2` probe candidate is now pinned more precisely:
  under public `TensorMemoryScalesLayout` it classifies as
  `tcgen05.copy.warpx4.32x128b` and then hits the tensor-memory-scales
  descriptor-plan diagnostic, now from `TMEMCopyOp` verification rather than
  falling through to late LLVM lowering. `CP_SCALES_LAYOUT_PROBE_CASES` now also
  pins four nearby shared-linear basis-order variants as clean unsupported; all
  of them still map to `warpx4.32x128b` descriptor-plan failures rather than
  public `warpx2` scales lowering. A refreshed FD-capturing subslice probe over
  384 parent-row shared-linear basis interleavings and starts `[0, 32, 64]`
  likewise found `1152` clean unsupported outcomes, zero unknown or bug-like
  failures, and no `warpx2` opcode. The `warpx4` control remains the positive
  copy path for this public scales surface.
- No-scales `cta_group::2` dense copy coverage now includes both dense
  families:
  - `tcgen05.cp.cta_group::2.128x128b` for the fixed `256x4` two-CTA
    i32/f32-bitwidth shape through explicit `SharedLinearLayout`;
  - `tcgen05.cp.cta_group::2.128x256b` for the existing wider N matrix.
- `ld.red` modifier saturation is expanded by `eac11c719`:
  - positive identity, tile-permuted, column-permuted, and row-permuted linear
    layouts now cover all four legal modifier pairs:
    plain, propagate-NaN, abs, and abs+propagate-NaN;
  - mixed unsupported layouts check the same modifier matrix and continue to
    report the dedicated `tmem_load reduction source layout is not directly
    tcgen05.ld.red-compatible` diagnostic.
- `ld.red` row-256 source-boundary coverage is expanded:
  - reduction-friendly `256x{32,64,128}` TMEM-linear layouts with the extra
    row bit carried in the column/query frame are positive and still emit
    `tcgen05.ld.red.sync.aligned.32x32b`;
  - plain identity `256x{32,64,128,256}` TMEM-linear source layouts are pinned
    as clean unsupported cases with the same actionable software-reduction
    diagnostic.
- `ld.red` explicit reduction-load layout coverage is expanded:
  - compatible explicit variants `auto`, `32x32b`, `16x32bx2`, and
    `32x32b_splitn` all execute correctly and still canonicalize to the exact
    `tcgen05.ld.red.sync.aligned.32x32b.x128.{min,max}.f32` opcodes;
  - explicit N-sharded variants `16x64b`, `16x128b`, and `16x256b` are pinned
    as clean unsupported cases with the dedicated `N dimension sharded across
    threads` diagnostic and register-layout note;
  - focused min/max explicit-layout validation: `11 passed in 8.57s`;
  - broad `ld_red` validation: `487 passed, 2183 deselected in 482.92s
    (0:08:02)`.
- Scaled-MMAv5 copy-matrix instruction coverage is tightened by `3374bbf12`:
  - the runtime-matrix scaled copy tests still assert exact
    `tcgen05.cp.cta_group::{1,2}.warpx4.32x128b` PTX/LLIR streams;
  - the same format/CTA/geometry/accumulator-layout matrix now also asserts
    exact `tcgen05.mma.cta_group::{1,2}.kind::{mxf8f6f4,mxf4,mxf4nvf4}.block_scale.scale_vec::*`
    opcode selection for PTX and LLIR.
- Tile-permuted MMAv5 accumulator coverage now spans all plain supported
  operand kinds:
  - `f16`, `tf32`, `bf16`, `f8e5m2`, and `f8e4m3`;
  - each kind is covered for `128x128` / `tile_n=32` and `128x256` /
    `tile_n=64` accumulator layouts;
  - each path checks PTX/LLIR opcode equality and the expected
    `tcgen05.mma` kind.
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
- That direct scaled-MMAv5 accumulator-subview format matrix now covers both
  `slice_start=0` and `slice_start=64`, so the same format/opcode assertions
  exercise root-aligned and offset accumulator subviews.
- Direct scaled-MMAv5 TMEM-LHS subview format coverage now includes the
  fp8-A reachable subset:
  - `test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view_fp8a_format_matrix`
    covers `mxfp8/mxfp8` and `mxfp8/mxfp4` for both legacy and canonical
    TMEM-linear accumulator layouts;
  - the test slices a packed-storage TMEM-linear operand-A parent, feeds that
    subview directly to `tcgen05_mma_scaled`, and pins exact PTX/LLIR scaled
    MMA and commit opcodes;
  - A-side fp4 scaled TMEM-LHS subviews remain a known wrong-code frontier from
    the 2026-04-11 probe and are not promoted to positive coverage; the durable
    reproduction is
    `experiments/probe_mma_scaled_lhs_subslice_formats.py` with current results
    in `experiments/results/probe_mma_scaled_lhs_subslice_formats_current.log`.
- Two-CTA direct scaled-MMAv5 accumulator-subview coverage is now present:
  - `test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_matrix`
    reuses the cga-aware TMA/scales-copy topology, allocates a larger
    TMEM-linear accumulator parent, slices it to the `cta_group::2` MMA result
    tile, and covers both `slice_start=0` and `slice_start=128`;
  - it validates numeric output, exact PTX/LLIR scaled-MMA opcodes for the same
    five format pairs, exact `tcgen05.cp.cta_group::2.warpx4.32x128b` scale
    copies, exact multicast `tcgen05.commit.cta_group::2`, and
    `ttg.memdesc_subslice` + `tensor_memory_linear` + `two_ctas` TTGIR.
- Plain MMAv5 kind saturation now includes f16 in the explicit 1-CTA and
  2-CTA kind matrices after `7766be003`, and the current two-CTA matrix has
  been widened to `N=128` and `N=256`:
  - `MMA_PLAIN_KINDS` covers `f16`, `tf32`, `bf16`, `f8e5m2`, and `f8e4m3`;
  - both legacy and canonical linear accumulator layouts are covered for
    `cta_group::1` and `cta_group::2`;
  - the `cta_group::2` kind matrix covers both `256x128` and `256x256`
    accumulator shapes.
  - the tests continue to assert exact PTX/LLIR opcode agreement and now pin
    exact root op counts: `f16=2`, `bf16=2`, `tf32=4`, and
    `f8e5m2/f8e4m3=1`.
- Plain MMAv5 `use_acc=True` coverage now spans all `MMA_PLAIN_KINDS`
  across both 1-CTA and 2-CTA paths, for both legacy and canonical
  TMEM-linear accumulator layouts, validating runtime accumulator addition
  plus exact PTX/LLIR opcode agreement and the same exact root op counts.
- Plain MMAv5 tile-permuted accumulator coverage now also pins its expected
  expanded instruction counts: four times the root kind count for accumulator
  tile permutations, and `16` f16 ops for the wider-K tile-permuted TMEM-LHS
  path.
- Plain MMAv5 TMEM-LHS subview coverage now spans all `MMA_PLAIN_KINDS` for
  both legacy and canonical TMEM-linear accumulator layouts:
  - `test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds` slices the
    right half of a TMEM-linear operand-A parent and feeds that subview directly
    to `tcgen05_mma`;
  - the matrix pins exact PTX/LLIR opcodes, expected instruction counts
    (`f16=2`, `bf16=2`, `tf32=4`, `f8e5m2/f8e4m3=1`), and the single-CTA
    commit opcode for every supported plain kind.
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
- Direct higher-rank `ld/st` access is now pinned as a clean unsupported
  boundary:
  - direct rank-3 `get_reg_layout`, explicit `load`, and explicit `store`
    attempts fail in the Gluon descriptor API with a 2D-only diagnostic telling
    users to index/slice/reshape higher-rank TMEM descriptors to a 2D view;
  - the C++ `getDistributedLayoutForTmemLdSt` planners return no layout for
    non-2D physical TMEM layouts instead of reaching the old `dims.size() == 2`
    assertion;
  - focused validation: `4 passed in 3.14s`, plus adjacent higher-rank indexed
    and dim0-slice positives `30 passed in 25.88s`;
  - broader `ldst and higher_rank` validation: `144 passed, 1 skipped,
    2521 deselected in 73.43s (0:01:13)`.
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
- Exotic `ld/st` layout matrices now include `auto` after `20f2c8db8`:
  - direct exotic, descriptor-chain exotic, and clean unsupported block-layout
    matrices use `LDST_VARIANTS`;
  - validation covers positive opcode agreement and clean negative diagnostics.
- Row/column cross-product permuted `ld/st` layouts now include `auto` after
  `6d44718e7`:
  - `LDST_ROWCOL_PERMUTED_CASES` uses `LDST_VARIANTS`;
  - this reaches direct ld/st and descriptor-chain composition sweeps;
  - focused validation ran the newly added auto parametrizations across the
    row/column permutation cross-product.
- Row/column descriptor-roundtrip `ld/st` boundary coverage now includes
  `auto` after `00139772b`:
  - `LDST_DESCRIPTOR_ROUNDTRIP_ROWCOL_CASES` uses `LDST_VARIANTS`;
  - focused validation shows the new auto parametrizations all hit the existing
    clean tensor-memory OOR skip boundary for lifted roundtrip shapes.
- The stale unused `LDST_EXPLICIT_VARIANTS` / higher-rank positive case list is
  removed after `a487942ac`, so scans no longer report a false explicit-only
  active `ld/st` matrix.
- Rank-5 descriptor-roundtrip `ld/st` boundary coverage now includes `auto`
  after `be193b6c6`:
  - one-CTA rank-5 cases use `LDST_VARIANTS`;
  - two-CTA rank-5 cases add `auto` to the existing explicit subset;
  - focused validation shows all new auto cases hit the existing clean
    tensor-memory OOR skip boundary.
- F16 subword ld/st pack/unpack coverage now includes `auto` after
  `4fe7a44de`:
  - `F16_LDST_CASES` uses `LDST_VARIANTS`;
  - the `auto` shape expectations match the explicit `32x32b` f16 packet
    family.
- X1 f32 direct and descriptor-chain ld/st roundtrips now include `auto` after
  `c475eecc0`:
  - `X1_F32_LDST_VARIANTS` covers `auto` and explicit `32x32b`;
  - validation covers one-CTA linear, one-CTA legacy, and two-CTA linear
    layouts with exact `32x32b.x1.b32` opcode expectations.
- X1 f16 ld/st roundtrips now include `auto` after `a950338f7`:
  - `X1_F16_LDST_VARIANTS` covers `auto` and explicit `32x32b`;
  - validation covers linear packed, legacy packed, and legacy unpacked
    layouts with the existing pack/unpack opcode expectations.
- Focused `ld/st` fuzz/stale-coverage cleanup now includes the remaining
  fixed-offset and rank-5 variant tails:
  - `test_tmem_runtime_matrix_ldst_fixed_offset_patterns_128x256` includes an
    `auto` companion and confirms it selects the canonical
    `32x32b.x64.b32` offset pattern;
  - two-CTA rank-5 descriptor roundtrip coverage uses the full
    `LDST_VARIANTS` set and confirms the newly added large explicit families
    hit the clean tensor-memory OOR boundary.
- Scales `ld/st` instruction-selection coverage now includes `auto`:
  - auto does not simply mirror explicit `32x32b`; for several shapes it
    selects wider `16x64b`, `16x128b`, or `16x256b` packet streams with the
    expected high-half `1048576` offsets;
  - `test_tmem_runtime_matrix_ldst_scales_variant_sweep` pins those exact
    PTX/LLIR streams across the existing scales variant matrix.
- Combined current-head `ld/st` runtime-matrix validation is green after the
  staged auto expansions, allocator-lifetime anchors, and subword coverage
  expansions:
  - command:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-broad-after-subword-refresh PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k ldst`;
  - result:
    `1181 passed, 441 skipped, 1027 deselected in 1340.05s (0:22:20)`;
  - skips are expected tensor-memory OOR / clean-boundary cases in lifted
    descriptor roundtrip and rank-5 families.
- Current-head `tcgen05.cp` runtime-matrix validation is green:
  - command:
    `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-cp-scales-verifier-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k cp`;
  - result:
    `162 passed, 5 skipped, 2488 deselected in 40.99s`;
  - this covers the current no-scales `warpx2` positives with exact commit
    opcodes, including the two-CTA `warpx2::01_23` path, the new two-CTA
    `warpx2::02_13` canonical-shared clean unsupported boundary, dense-shared
    `warpx2` clean negatives, dense copy positives, 2-CTA `128x128b` /
    `128x256b` copy, scaled `warpx4` copy paths, exact scaled-copy commit
    opcodes, and clean unsupported copy boundaries; a follow-up direct-seed
    probe found that aligned `tmemDwordDelta=0` moves data for two-CTA
    `02_13` but duplicates each source-column pair, `tmemDwordDelta=4` is the
    known all-zero path, unaligned deltas trap, and the known seed fields do not
    recover the missing source-column bit; it also covers the historical scales
    `warpx2` candidate and four nearby basis-order variants as clean unsupported
    verifier-side `warpx4.32x128b` descriptor-plan failures.
- Current-head `tcgen05.ld.red` runtime-matrix validation is green:
  - command:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldred-wait-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k ld_red`;
  - result:
    `487 passed, 2183 deselected in 482.92s (0:08:02)`;
  - this covers identity, tile-permuted `128x{64,128,256}` including the
    minimal `N=64, tile_n=16` case, pure column-permuted
    `128x{64,128,256}`, pure row-permuted `128x{64,128,256}`, row/column
    cross-product permutations at `128x{64,128,256}`, all legal modifier pairs,
    reduction-friendly row-256 positives, clean unsupported identity row-256
    layouts, exact offset immediates (`[0]` for `N <= 128` and
    `[0, 64, 128, 192]` for `N=256`), exact `tcgen05.wait::{st,ld}`
    emission and ordering around store/reduction-load consumption, and clean
    unsupported mixed-layout negatives.
- Current-head direct `tcgen05.mma` / `mma_scaled` runtime-matrix validation is
  green:
  - command:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-lhs-scaled-fp8a-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma and not cp'`;
  - result:
    `226 passed, 50 skipped, 2407 deselected in 131.68s (0:02:11)`;
  - this covers canonical, indexed, subview, tile-permuted, 1-CTA and 2-CTA
    direct MMA surfaces plus direct scaled-MMA view cases, including root-aligned
    and offset one-CTA accumulator subview format-matrix cases, the two-CTA
    cga-aware accumulator-subview matrix, tile-permuted plain-kind accumulators
    at `128x128/tile_n=32` and `128x256/tile_n=64`, the all-plain-kind
    TMEM-LHS subview matrix for both legacy/canonical accumulator layouts, the
    fp8-A scaled TMEM-LHS subview matrix, and two-CTA plain-kind accumulators at
    `256x128` and `256x256`; it also covers both 1-CTA and 2-CTA `use_acc=True`
    plain-kind accumulator addition for all supported plain kinds and both
    legacy/canonical accumulator layouts, plus TMA-fed two-CTA TF32 when matrix
    B is loaded through a non-transposed `[N, K]` TMA
    descriptor and passed to MMAv5 as a shared-memory `permute((1, 0))` view.
    Scaled-MMA copy-helper coverage remains tracked separately.
- Tile-permuted scaled-MMAv5 accumulator-subview clean-negative coverage now
  spans the same format pairs as the positive direct subview matrix:
  - `mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`,
    `mxfp4/mxfp8`, and `nvfp4/nvfp4`;
  - each case confirms the repeated `N=32` tile-permuted layout fails with the
    dedicated `matrix-B scale fragments at 64-column alignment` diagnostic and
    does not fall through to a PassManager/assertion crash.
- The recorded 2-CTA TF32 TMA-fed shared-transpose issue is now pinned as a
  clean verifier negative instead of a late LLVM pass failure:
  - `test_tmem_runtime_matrix_mma_twocta_tma_tf32_reports_clean_shared_transpose_error`
    covers both legacy and canonical TMEM-linear accumulators;
  - legal non-transposed TMA descriptors still cannot feed this TF32 MMAv5
    shape because TMA descriptors cannot be transposed and MMAv5 rejects the
    resulting transposed shared-memory operand;
  - the verifier now reports
    `tcgen05.mma does not support transposed float32 operands in shared memory`
    without `PassManager::run failed` or assertions;
  - direct non-TMA 2-CTA TF32 remains covered by the green direct-MMA slice, so
    this remains a TMA-to-shared layout materialization/API frontier rather
    than a direct accumulator-layout regression.
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
    the linear supported layout passes in the same process. This is now an
    explicit follow-up fix on the docket, not just a status note, but it should
    stay behind current CI-lane freshness and aggregate examples validation.
- The supported descriptor bitcast API remains on the branch, and the attention
  scratch-alias path is migrated across the SDPA-safe benchmark-shaped unit
  matrix:
  - `python/examples/gluon/01-attention-forward.py` no longer uses
    `_reinterpret` for the scratch-borrow helpers;
  - P scratch slicing is dtype-aware: fp16/bf16 use 64 f32 columns for the P
    image, while fp8 uses 32, and alpha/epilogue scratch is placed after the
    actual P physical region;
  - exp2 partitions still slice the original f32 scratch subregion before
    bitcasting each exact physical image;
  - causal `use_tmem_red` now computes the diagonal-stage row max after causal
    masking instead of using a pre-mask `ld.red` max;
  - the SDPA-backed attention matrix is green:
    `64 passed in 102.66s (0:01:42)`;
  - full `python/examples/gluon` still needs a fresh aggregate rerun before
    marking the broader examples lane current.
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
- The latest recorded full GB200 `make NUM_PROCS=24 test-unit` lane was
  recovered at an older checkpoint and must be rerun at current HEAD before a
  full fresh CI stamp:
  - checkpoint `78196b4e4` reran the full wrapper green;
  - main `python/test/unit`: `15153 passed, 5492 skipped`;
  - `python/test/unit/test_debug.py`: `95 passed`;
  - unit-tail sublanes were green: `python/triton_kernels/tests`,
    `python/tutorials/06-fused-attention.py`, instrumentation, and plugin tests;
  - current unit manifests are empty:
    `gb200_current_branch_test_unit_failures.txt`,
    `gb200_current_branch_test_unit_matmul_refresh_failures.txt`,
    `gb200_current_branch_test_unit_tensor_descriptor_refresh_failures.txt`,
    `gb200_current_branch_test_unit_warp_specialization_refresh_failures.txt`,
    and `gb200_current_branch_test_unit_rowanchor_refresh_failures.txt`;
  - this unit green point predates later focused compiler/test coverage commits,
    so do not treat it as a current-HEAD full-wrapper rerun without refreshing.
- The supported `_reinterpret` migration invariant remains:
  - offset to the right part of TMEM;
  - slice/subview it to the desired physical bits;
  - bitcast to the desired dtype/shape/layout only when equal-size and
    physical-mapping equivalent to the input descriptor.
- Next required durable step: continue the remaining recovery queue with the
  intentionally reverted attention example tracked separately from supported
  bitcast API validation. Keep the legacy M64 MMAv5 xfail as an explicit
  follow-up fix item, then continue broader MMAv5/`mma_scaled`, staged `ld/st`
  fuzzing, stale-negative cleanup, and heuristic phases, with any remaining copy
  work limited to shapes that fit the TMEM allocation budget.

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
