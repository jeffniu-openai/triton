# TMEM Linear Generalization

## Goal
- Canonicalize TMEM layouts on `tensor_memory_linear`.
- Normalize legacy TMEM layouts before semantic consumption.
- Generalize TMEM descriptors to the shared memdesc API and semantics.
- Broaden TMEM lowering for linear layouts while keeping MMAv5 and `tmem_copy`
  on explicit hardware-family matchers.

## Current Decisions
- Backward compatibility is by early normalization, not by maintaining dual
  semantic paths.
- `tensor_memory_descriptor.slice(...)` will match
  `shared_memory_descriptor.slice(...)`, including `dim=0` default and
  arbitrary-dimension slicing.
- `immHalfSplitOffset == 0` is a legal `16x32bx2` lowering case; the old PTXAS
  workaround must be removed.
- Development-time PTX and hardware validation is expected for ambiguous
  `tcgen05` behavior, but only targeted regressions land in-tree.
- Plain `ttng.tmem_load`, `ttng.tmem_store`, and source-initialized
  `ttng.tmem_alloc` now defer layout feasibility to relayout/lowering instead
  of rejecting non-default compatible register layouts in the verifier.
- `ttg.memdesc_subslice` on TMEM preserves the canonical physical TMEM
  encoding; it is a pointer/view operation over the same backing allocation.
- `ttng.tmem_subslice` is treated as compatibility sugar over the same physical
  TMEM allocation and no longer re-encodes the result shape.
- `ttg.memdesc_index` remains permissive for the extra leading TMEM
  multibuffer dimension; an alloc-shape equality requirement there was tried
  and reverted because it broke the established TMEM multibuffer view model.
- Buffer-region accounting for TMEM subviews follows physical TMEM allocation
  size, not only the smaller logical result shape.
- Relayout now prefers direct TMEM ld/st-compatible layouts and only inserts a
  single `convert_layout` fallback when direct lowering is not possible.
- A non-empty TMEM `block` basis implies multi-CTA CGA semantics regardless of
  `twoCTAs`; `twoCTAs` is extra MMAv5/protocol intent metadata, not the sole
  source of CTA tiling.
- Unsupported TMEM/MMAv5 combinations must fail with diagnostics, never with
  parser or verifier crashes; legacy-layout matching paths need defensive
  handling for impossible candidate families.
- TMEM block-bases descriptor views that eventually reinterpret to a
  `64x32` TMEM-linear memdesc currently have no TMEM-compatible register
  layout for plain `tmem_load/store`; they are a tracked unsupported frontier
  and must fail with a clean relayout diagnostic instead of asserting.
- Higher-rank TMEM descriptor compositions that repartition a tile into
  broadcasted subviews currently fail at `get_reg_layout(...)` with a clean
  diagnostic (`TMEM layout '<atom>' unsupported ... reshape or permute so TMEM
  columns stay contiguous ...`) rather than reaching a pass-time assert or
  late lowering failure.
- TMEM load-reduction on canonical linear layouts is only legal when the
  N dimension is not sharded across threads. Canonical identity TMEM-linear
  layouts satisfy this and execute correctly; the current mixed TMEM-linear
  family does not and should fail with the dedicated
  `tmem_load reduction with N dimension sharded across threads is not supported`
  diagnostic.
- The runtime sweep must distinguish between:
  - general TMEM ld/st-legal linear layouts, which may be far broader than
    MMAv5/copy hardware families;
  - MMAv5-legal canonical TMEM-linear families, which are exactly the
    TMEM-linear layouts equivalent to supported legacy
    `#ttng.tensor_memory_encoding` tiles; and
  - `tcgen05.cp`-legal TMEM destinations, which are constrained by the finite
    `cp` atom family and shared-memory multicast patterns.
- A positive MMAv5 runtime test must use the canonical TMEM-linear equivalent
  of a supported legacy MMAv5 accumulator/LHS tile. A generic block-basis
  TMEM-linear layout that is fine for ld/st is not necessarily MMAv5-legal.
- For tensor-memory scales direct ld/st, `instr_variant` is a hint, not a
  guarantee of the final atom family:
  - narrow tiles requested as `32x32b` may still lower through
    `16x32bx2.x1`, and
  - some `16x32bx2`-requested cases canonicalize to wider `32x32b.x{1,2,4,8}`
    atoms when the register layout permits it.
- The current direct scales ld/st positive frontier is now runtime-validated
  for exact PTX/LLIR opcode matches across:
  - `16x32bx2.x1`
  - `32x32b.x{1,2,4,8,16,32}`
  - including representative offset-bearing cases (`+2`, `+4`, `+8`, `+16`)
    and both direct/default and explicit-variant `get_reg_layout(...)` paths.
- The current x1 direct ld/st frontier is runtime-validated for:
  - `f16` packed/unpacked one-CTA TMEM
  - `f32` one-CTA and two-CTA TMEM-linear/legacy-equivalent layouts
  - descriptor-chain compositions over those x1 shapes
  - while `16x64b`, `16x128b`, and `16x256b` remain clean negatives for
    `[M, 1]` x1 shapes.
- The current no-scales two-CTA `tcgen05.copy` frontier is runtime-validated
  for both legacy and canonical TMEM-linear destinations over:
  - `N=16` with `swizzle=32`
  - `N=32` with `swizzle in {32,64,128}`
  - `N in {64,128,256}` with `swizzle=128`
  - all lowering to `tcgen05.cp.cta_group::2.128x256b` with exact counts.

## Runtime Fuzz Strategy

- The exact execution matrix, opcode inventory, GPU sharding, and per-family
  fuzz axes now live in `fuzz_plan.md`. This file keeps the durable policy and
  current-state summary; `fuzz_plan.md` is the operational playbook.

### Outcome classes
- `PASS`: kernel compiles, executes on GPU, numerics are correct, and PTX/LLIR
  contain the expected `tcgen05` opcode family and offsets/immediates.
- `CLEAN_UNSUPPORTED`: compile fails with a diagnostic that explains why the
  layout/op is impossible to lower with the ISA and suggests a viable fix
  (`get_reg_layout`, `convert_layout`, reshape/permute so columns stay
  contiguous, or use a supported MMAv5/copy tile).
- `BUG`: parser/verifier/pass/lowering crashes, malformed IR cannot be printed,
  valid IR crashes a pass, or GPU output/PTX mismatches expectations.

### Shared test oracles
- Numerical correctness on real Blackwell hardware is mandatory for positive
  cases.
- The extracted `tcgen05` opcode stream from PTX and LLIR must match exactly.
- For generic descriptor compositions, TTGIR must contain the generic memdesc
  view ops (`memdesc_subslice/index/trans/reshape/reinterpret`) rather than a
  bespoke TMEM-only path, except where `ttng.tmem_subslice` is explicitly
  being regression-tested as compatibility sugar.
- Impossible cases must never assert/crash and must emit actionable notes.

### `tcgen05.ld/st` fuzz plan
- Sweep TMEM-linear families by legality class:
  - canonical identity (`128x{64,128,256}`);
  - canonical mixed/interleaved row/col (`128x{64,128,256}`);
  - `block`-basis general layouts for `256x{64,128,256}`;
  - `block + two_ctas` general layouts for `256x{64,128,256}`;
  - canonical MMAv5-like two-CTA layouts derived from
    `ttgl.to_linear_layout(TensorMemoryLayout(...), shape)` for the narrower
    MMAv5 overlap cases;
  - `blockM=64`/split-N families that must lower through `16x32bx2`.
- For each legal family, sweep instruction selection:
  - `auto`
  - `32x32b`
  - `16x64b`
  - `16x128b`
  - `16x256b`
  - `32x32b_splitn` where applicable
- For each case, sweep descriptor compositions over legal views:
  - multibuffer `slice(..., dim=0)` then `index`
  - `slice(..., dim=0/1/2)` on reshaped 3D views
  - paired `permute`/inverse-`permute`
  - `reshape` chains that preserve element count
  - `_reinterpret` back to a legal TMEM-linear layout
- Functional oracle:
  - round-trip equality for plain load/store
  - exact offset/immediate checks for `16x32bx2`
  - exact atom family checks for all other atoms
- Negative frontier:
  - layouts with non-contiguous TMEM columns or no supported TMEM register
    layout should fail with the relayout diagnostic, not assert.

### `tcgen05.ld.red` fuzz plan
- Sweep `red_op in {min, max}`.
- Sweep modifiers:
  - `abs in {false, true}`
  - `propagate_nan in {none, all}` where legal
- Sweep shapes/warp partitions that are theoretically reducible:
  - `M in {128, 256}`
  - `N in {32, 64, 128, 256}`
  - `num_warps` chosen so the N dimension is not sharded across threads when
    hardware reduction is required.
- Functional oracle:
  - TMEM round-trip path still returns the original tensor
  - reduced output matches the corresponding PyTorch reduction with correct
    NaN/abs semantics
  - PTX/LLIR use `tcgen05.ld.red.sync.aligned.32x32b.x*.{min|max}[.abs][.NaN].f32`
- Negative frontier:
  - unpacked formats, integer types, or N-sharded register layouts must fail
    with the dedicated reduction diagnostic.

### `tcgen05.cp` fuzz plan
- Enumerate the currently implemented atom families:
  - `cta_group::1.128x256b`
  - `cta_group::1.128x128b`
  - `cta_group::{1,2}.warpx2::02_13.64x128b` if supported in lowering
  - `cta_group::{1,2}.warpx2::01_23.64x128b` if supported in lowering
  - `cta_group::{1,2}.warpx4.32x128b`
- For no-scales copies, sweep `M in {128, 256}`, `N in {16,32,64,128,256}`,
  `BLOCK_N` divisors, and relevant swizzles, skipping only true SMEM
  out-of-resource cases.
- For scales and multicast variants, synthesize exact shared layouts that drive
  the documented multicast bits and alias the destination back through a legal
  TMEM view to validate the payload.
- Functional oracle:
  - copied output equals input (or equals the documented aliasing/permutation
    for scales layouts)
  - PTX/LLIR opcode family and count exactly match the expected atomization
- Negative frontier:
  - unsupported shared multicast/layout combinations must fail cleanly, not
    silently change atom family or crash.

- Progress note:
  - `probe_cp_multicast_layouts.py` and the companion `cp_warpx2_probe_test.py` both exhaust the
    simple interleavings seen above; the current runs on GPU3 keep hitting the
    `tcgen05_copy` legalization error before emitting any `warpx2` opcode.
  - `warpx2_layout_manual.py` tries the same insertion patterns with quieter
    logging but still fails to produce `warpx2`, while `warpx2_layout_linear.py`
    inspects the converted linear layout and shows that the two 32-byte-row
    basis entries never become `[0, 0]`, so multicast stays zero.
  - Need to broaden the layout search (more column strides, extra block
    reorderings, two-CTA-specific layouts) until `tcgen05_copy` legalizes one of
    the documented `warpx2` families.
  - 2026-03-25 refresh:
    - Exhaustive surjective one-hot interleavings for direct scales copies
      (`64x16 -> 128x32`) produced only 2 legal layouts, both emitting
      `warpx4.32x128b` (no `warpx2`).
    - Exhaustive one-hot interleavings for no-scales (`128x4 i32`) produced
      only `128x128b` (no `warpx2`).
    - Subslice probes from larger shared allocations (`parent_rows=128/256`)
      yielded no legal `tcgen05_copy` cases in sampled layout sets; when slice
      start was non-zero, frontend constraints (`split offset may not touch the
      tile`) rejected many candidates.
    - Mixed-basis shared-linear mutations (e.g. `[32, k]`) reached lowering but
      still failed descriptor legalization; still no executable `warpx2`.
    - Runtime coverage now includes executable `cta_group::2.warpx4.32x128b`
      in `python/test/gluon/test_tmem_runtime_matrix.py` via the scaled-MMA
      copy path; direct standalone scales-copy kernels still have not produced
      executable `warpx2`.

### `tcgen05.mma` fuzz plan
- Separate sweeps by hardware family:
  - `cta_group::1`
  - `cta_group::2`
- Sweep accumulator layouts:
  - legacy `TensorMemoryLayout`
  - canonical TMEM-linear equivalents of those legacy layouts
  - TMEM-general but non-MMAv5 layouts as negative cases
- Sweep accumulator usage:
  - `use_acc in {false, true}`
  - multicast/commit path where applicable
- Sweep supported tile families:
  - canonical `blockM=128`
  - canonical `blockM=64` split-N/interleaved families where legal
  - LHS-in-shared and LHS-in-TMEM where supported
- Functional oracle:
  - result matches `torch.matmul` (plus accumulator add when `use_acc=true`)
  - PTX/LLIR contain the exact `tcgen05.mma.cta_group::{1,2}.kind::*` opcode
    family and expected commit/barrier opcodes
- Negative frontier:
  - mixed/block-basis TMEM-general layouts that are not legacy-equivalent must
    fail with the MMAv5 compatibility diagnostic.

### `tcgen05.mma_scaled` fuzz plan
- Sweep format combinations already supported in tree (`mxfp8`, `mxf4`,
  `mxf4nvf4`, etc.) and confirm the emitted `scale_vec::{1X,2X,4X}` suffix.
- Sweep `cta_group::{1,2}` where supported.
- Sweep accumulator/scales layouts:
  - legacy encodings
  - canonical TMEM-linear equivalents
  - two-CTA canonical families
- Functional oracle:
  - numerical output matches the reference within the established tolerance
  - PTX/LLIR `tcgen05.mma...block_scale.scale_vec::*` opcode streams match
  - associated `cp`/barrier ops appear as expected for the chosen pipeline
- Negative frontier:
  - unsupported TMEM-linear accumulator families, scale layouts, or CGA
    mismatches must fail cleanly with MMAv5/scaled-MMA diagnostics.

## Work Split
- Main rollout owns core attr/type/helper/lowering integration.
- Subagents may be used for bounded exploration, PTX validation snippets, and
  test drafting. Claims belong in `agents/claims.md`.

## Next Execution Plan
- The detailed per-instruction fuzz plan lives in
  `experiments/tmem_fuzz_plan.md`.
- Immediate priority order:
  - close the clean-diagnostic vs. runtime-pass boundary for higher-rank
    broadcasted TMEM views;
  - broaden positive runtime coverage until every currently documented and
    backend-reachable TMEM opcode family is seen in both PTX and LLIR on real
    hardware;
  - then use the fuzz results to decide which remaining clean failures are true
    ISA limits and which are lowering gaps that should be implemented.

## Active Fuzz Backlog
- Convert the current higher-rank TMEM `ld/st` broadcasted-subview failure into
  a permanent clean-diagnostic regression, then expand positive higher-rank
  runtime coverage only on compositions with proven legal TMEM register layouts.
- Status update (2026-03-25):
  - positive higher-rank descriptor `index -> 2D load/store` compositions are
    now covered for `identity`, `mixed`, `block_two_ctas`, and
    `mmav5_twocta` over `N in {64,128,256}` and variants
    `{32x32b,16x64b,16x128b,16x256b}` with GPU execution and PTX/LLIR checks.
  - non-representable higher-rank multidimensional TMEM slices are now locked
    as clean negatives with `unsupported tensor memory memdesc_subslice view`.
  - MMAv5 two-CTA context-mismatch higher-rank compositions remain clean
    negatives with the intended invalid-layout / CGA-mismatch diagnostic.
- Ensure every currently emitted TMEM instruction family has:
  - at least one passing runtime test with exact PTX/LLIR opcode checks;
  - at least one lit lowering test that validates the generated LLVMIR; and
  - at least one clean-negative regression if the family has known impossible
    frontiers.
- Treat undocumented or ambiguous documented families, especially `cp.warpx2`,
  as direct-PTX probe targets until the backend/legalizer boundary is clear.
- Latest coverage refresh (2026-03-25):
  - canonical TMEM-linear `ld.red` now has direct GPU coverage in the runtime
    matrix for `min/max x {abs,no-abs} x {NaN,non-NaN}`;
  - TMEM LLVMIR lit coverage is green after tightening lifecycle, `cp`,
    `mma_scaled`, and pointer-math checks;
  - `cp.warpx2` still has no executable compiler path in the current tree from
    the bounded subslice search, so it remains a live probe target rather than
    a passing family.

## Current Higher-Rank TMEM Status
- Chosen semantic direction is now explicit:
  - higher-rank TMEM descriptors should use a full-rank
    `TensorMemoryLinearLayout` that embeds descriptor prefix dims into TMEM
    `col` high bits;
  - generic TMEM `slice/index/permute/reshape/reinterpret` then operate on the
    full-rank encoding;
  - actual TMEM access ops (`load/store/mma/cp`) remain 2D-only for now, so
    higher-rank descriptors must be sliced/indexed/reshaped down to a final 2D
    TMEM view before access.
- Runtime status after the latest GPU validation:
  - full-rank rank-3 TMEM descriptors are working end-to-end for
    `index -> 2D load/store`;
  - full-rank rank-4 TMEM descriptors are working end-to-end for
    `slice -> index -> slice -> index -> 2D load/store`;
  - the same rank-4 path now passes broad row/col-permuted TMEM-linear sweeps
    (`736 passed, 64 skipped`) across explicit ld/st variants plus split-N;
  - the same rank-4 path is working for 2-CTA `block_two_ctas` and
    `mmav5_twocta` layouts as well.
  - descriptor multi-dim slice semantics now have positive GPU proof via the
    new `tmem_ldst_descriptor_multidim_slice_positive_kernel`; the test
    exercises `identity`, `mixed`, and `scrambled_cols` linear layouts,
    verifies the block update and `ttg.memdesc_slice` in TTIR, and checks the
    `16x128b` PTX/LLIR match.
  - 2D last-dimension TMEM slices that need to preserve split-N / M64 physical
    layout now route through the `ttng.tmem_subslice` compatibility path from
    the frontend, restoring valid `block_m_64` MMAv5 compositions.
- Plain MMAv5 runtime coverage status:
  - `f16`, `tf32`, `f8f6f4`, and now `bf16 -> kind::f16` are all exercised
    with GPU execution and exact PTX/LLIR opcode checks for canonical
    accumulator layouts.
  - `kind::i8` remains a clean PTXAS-level unsupported case on `sm_103a`.
- Active uncovered-but-plausible MMA frontier:
  - 2-CTA TF32 MMAv5 currently fails lowering cleanly with
    `tcgen05.mma does not support transposed float32 operands in shared memory`;
    treat this as a compiler BUG / follow-up target, not a settled unsupported
    ISA boundary.
- New 2-CTA plain MMA coverage status (2026-03-26):
  - plain `tcgen05.mma.cta_group::2` is now GPU-proven for:
    - `tf32`
    - `bf16 -> kind::f16`
    - `f8e5m2 -> kind::f8f6f4`
    - `f8e4m3 -> kind::f8f6f4`
  - both legacy and canonical TMEM-linear 2-CTA accumulator layouts are now
    covered in the runtime matrix with exact PTX/LLIR opcode equality.
- New no-crash frontend guard:
  - `SwizzledSharedLayout` now validates `cga_layout` basis rank against
    `order` rank in Python before IR materialization.
  - Invalid 1D shared layouts such as `mbarrier.MBarrierLayout(cga_layout=[[1,
    0]])` now fail with a wrapped `CompilationError` rather than tripping a C++
    assert during `buildCgaLayoutAttr(...)`.
- Active 2-CTA copy BUG frontier:
  - the earlier cold-start failure on 2-CTA no-scales `tcgen05.copy` was not a
    legal-codegen bug once the sequence was validated empirically;
  - correct 2-CTA copy usage requires explicit pre-copy async-proxy ordering,
    i.e. `fence_async_shared(cluster=True)` before `tcgen05_copy` when the
    source lives in distributed shared memory;
  - with that fence in place, the runtime matrix now passes for both legacy and
    canonical TMEM-linear 2-CTA destinations, and PTX/LLIR show the expected
    `tcgen05.cp.cta_group::2.128x256b` plus
    `tcgen05.commit.cta_group::2...multicast::cluster.b64`.
- New verifier/lowering fixes landed:
  - `ttng.tmem_copy` now validates its optional completion barrier through
    `verifyBarrierType(...)` and `verifyCompletionBarrierLayout(...)`, so bad
    barrier layouts fail early and cleanly.
  - unpredicated cross-CTA `ttng.wait_barrier` lowering no longer crashes when
    the leader predicate must be synthesized in LLVM lowering.
- High-value TMEM test gaps from the latest audit:
  - executable linear `128x128b` copy coverage exists only for legacy TMEM,
    not canonical TMEM-linear;
  - TMEM view destinations for `128x256b` and scales `warpx4.32x128b` copy are
    still under-tested;
  - direct scaled MMAv5 via TMEM views remains a good next runtime target.
- New PASS coverage:
  - executable `32x32b.x1` load/store is now covered for:
    - canonical `TensorMemoryLinearLayout`,
    - legacy packed TMEM,
    - legacy unpacked TMEM;
  - direct `TensorMemoryScalesLayout` ld/st is now covered end-to-end for:
    - default descriptor API on `16x8xi8` with `num_warps=8`,
    - explicit-variant cases on `16x4xi8`, `32x4xi8`, `64x8xi8`, and
      `128x32xi8`;
  - isolated LLVMIR checks for canonical x1 TMEM-linear lowering and direct
    scales x1 load/store are green.
- New BUG frontier from the scales sweep:
  - The explicit narrow-scales gap is now closed:
    - `TensorMemoryScalesLayout(shape=[16, 8], num_warps=8,
      instr_variant="16x32bx2")` now reuses the broadcasted `32x32b`
      register layout and lowers/executed correctly as
      `tcgen05.{ld,st}.sync.aligned.16x32bx2.x1.b32`.
  - Direct scales ld/st with nontrivial CGA bases is now GPU-proven for:
    - `shape=[128, 64]`, `num_ctas=2`, `CGALayout=[[1, 0]]`
    - `shape=[256, 32]`, `num_ctas=2`, `CGALayout=[[1, 0]]`
    - `shape=[256, 64]`, `num_ctas=2`, `CGALayout=[[1, 0]]`
    with exact PTX/LLIR checks for the emitted `32x32b` families.
  - Remaining copy-family frontier:
    - direct PTX probes still show `tcgen05.cp.warpx2::{02_13,01_23}.64x128b`
      is real, but Triton's scales-copy verifier/lowering intentionally keeps
      those layouts `CLEAN_UNSUPPORTED` because the required shared-memory
      descriptor synthesis is not implemented yet.
  - Current targeted lit slice is green:
    - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `test/TritonNvidiaGPU/invalid.mlir`
- The key implementation boundary is now cleaner:
  - row/col zero-basis stripping is only valid for TMEM view-local encodings;
    doing it globally breaks semantically meaningful M64 / split-N / legacy
    MMAv5 families.
- Current remaining sharp edge:
  - intermediate higher-rank TMEM views still fail cleanly at
    `get_reg_layout()/load()/store()` with an actionable 2D-only diagnostic.
  - This is acceptable for the current landing, but direct higher-rank TMEM
    access remains future work.


- 2-CTA no-scales `tcgen05.copy` status (2026-03-26 10:05 UTC):
  - the earlier cold-start corruption is fixed on the current tree;
  - required compiler invariants are now pinned by tests:
    - `ttng.tmem_copy` is a tracked cross-CTA consumer for barrier insertion,
    - distributed shared -> `ttng.tmem_copy` gets a pre-copy
      `ttng.cluster_barrier`,
    - runtime PTX/LLIR for the 2-CTA `128x256b` family uses aligned cluster
      barriers, not the old relaxed-barrier shape.
  - executable coverage is now strong for this path:
    - 28/28 `cp_no_scales_twocta_codegen` runtime cases pass on GPU,
    - both former cold-start tuples also pass 10 fresh-process runs each.
  - remove this from the active BUG frontier; the next copy-family frontier is
    `warpx2` shared-descriptor synthesis.

- 2-CTA no-scales copy compiler contract (2026-03-26 18:25 UTC):
  - `ttng.tmem_copy` must participate in memory-effect analysis and cross-CTA
    barrier insertion just like MMA/TMA consumers.
  - The explicit-buffer alias fast-path is only sound for non-distributed
    shared accesses; distributed slices must still participate in hazard
    tracking.
  - The executable/runtime contract for the working `cta_group::2.128x256b`
    path is now pinned as:
    - `fence_async_shared(cluster=True)` before the copy,
    - aligned cluster barrier pair before the first copy message,
    - exact `tcgen05.cp.cta_group::2.128x256b` counts,
    - exact `tcgen05.commit.cta_group::2...multicast::cluster.b64` commit.
  - Current validation status:
    - former cold-start repro tuples pass 10 fresh-process runs each;
    - full `cp_no_scales_twocta_codegen` matrix is green;
    - surrounding copy/scales/plain-MMA runtime slice is green (`130 passed`).

- Runtime validation checkpoint (2026-03-26 18:27 UTC):
  - the repaired 2-CTA copy path still holds under a broader executable slice;
  - post-fix runtime sweep status:
    - `cp_no_scales_twocta_codegen`
    - `cp_scales_warpx4`
    - `cp_scales_warpx4_via_scaled_mma_copy_matrix`
    - `cp_scales_warpx4_via_scaled_mma_geometry_sweep`
    - `cp_no_scales_linear_32bit_dtypes`
    - `mma_twocta`
    - `mma_twocta_plain_kinds`
    all green together (`103 passed`).
  - no new compiler/runtime correctness frontier opened by this broader slice.
  - note for future work: the transient `smem_layout is not defined` failure was
    only a test-harness regression in `test_tmem_runtime_matrix.py`, not a TMEM
    compiler bug.

- warpx2 frontier audit (2026-03-26 18:32 UTC):
  - current compiler status:
    - classifier recognizes `warpx2::{02_13,01_23}.64x128b`;
    - PTX builder can emit the opcode suffixes;
    - verifier/lowering still rely on the generic MMAv5 shared-descriptor
      builder, which has no `warpx2`-specific descriptor/address model.
  - practical implication:
    - enabling `warpx2` is not a small opcode toggle; it requires dedicated
      descriptor synthesis and copy-address stepping.
  - bounded current-tree probe of the old scales `warpx2` search now fails
    cleanly at verifier time rather than late LLVM legalization.

- Copy-family frontier update (2026-03-26 18:33 UTC):
  - fresh direct-PTX probes reconfirm:
    - `warpx2::02_13` and `warpx2::01_23` are real opcodes with deterministic warp-pair semantics;
    - they are not drop-in substitutes for the current Triton warpx4 descriptor/message schedule;
    - `4x256b` still launches but is not deterministic enough for compiler support.
  - practical implication:
    - the next `warpx2` support step is family-specific descriptor/address/message synthesis, not just classifier widening or opcode substitution.
  - until then, keep these families on the clean-unsupported side with explicit notes rather than attempting partial lowering.

- warpx2 overlay semantics checkpoint (2026-03-26 20:10 UTC):
  - durable experiment artifacts now live in:
    - `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_overlay.py`
    - `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_overlay_gpu3.json`
    - `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_overlay_gpu3_deltas.json`
  - hardware-proven semantics for the inherited descriptor/address family:
    - base copy sites are `[ %r14 + 0 ], %rd3` and `[ %r12 + 0 ], %rd4`;
    - with one active `warpx2` message:
      - `02_13 + (%rd3 @ %r14 + 0)` writes the left half of chunks `0/2`;
      - `02_13 + (%rd4 @ %r14 + 4)` writes the right half of chunks `0/2`;
      - `01_23 + (%rd3 @ %r14 + 0)` writes the left half of chunks `0/1`;
      - `01_23 + (%rd4 @ %r14 + 4)` writes the right half of chunks `0/1`;
      - no tested one-message schedule produced the missing complementary left
        halves for chunks `1/3` or `2/3`.
    - intermediate TMEM destination deltas `+1` and `+2` are invalid on
      hardware for both families and both descriptors:
      `cuCtxSynchronize failed: misaligned address`.
  - two-message inference:
    - `02_13` with the inherited descriptor pair `(%rd3 @ %r14, %rd4 @ %r12)`
      fully covers chunks `0/2` only;
    - `01_23` with that same pair fully covers chunks `0/1` only;
    - mixed-family two-message schedules are still not exact.
  - current design consequence:
    - `warpx2` cannot be enabled from the current two-message warpx4 plan by
      changing only opcode suffixes or small TMEM address deltas;
    - the likely enabling path is a dedicated copy-family planner that can
      synthesize different descriptors and, if necessary, emit more than the
      inherited two copy messages.
  - until that exists, keep `warpx2` cleanly unsupported and preserve the new
    explicit verifier note about partial chunk-pair fills and invalid `+1/+2`
    deltas.

- Plain TMEM ld/st/alloc explicit-layout policy (2026-03-26 21:35 UTC):
  - explicit user-requested TMEM register layouts are now strict:
    - if the requested tensor layout is directly TMEM-compatible, compile it;
    - otherwise, fail with a diagnostic and possible suggested TMEM layouts.
  - the compiler no longer silently inserts fallback `convert_layout` for plain
    `ttng.tmem_load`, `ttng.tmem_store`, or initialized `ttng.tmem_alloc`.
  - the late `make_llir` scheduling of `relayout-tritongpu` was removed for
    this reason; warp-specialization still retains its own relayout path for
    partition re-inference after changing `numWarps`.
  - implication for tests/examples:
    - positive TMEM tests should use `get_reg_layout(...)` when they want a
      directly codegenable register layout;
    - if a test wants a blocked or other non-direct TMEM layout, it must use
      `convert_layout` explicitly around the TMEM access rather than relying on
      hidden compiler repair.

- TMEM lit / LLVM cleanup status (2026-03-26 23:40 UTC):
  - keep the shared-memory reshape fallback in
    `lib/Dialect/TritonGPU/IR/Dialect.cpp`; it is required for the
    `cp_scales_warpx4_via_scaled_mma_geometry_sweep` path.
  - do not keep the local `tryMakeTMemViewEncoding(...)` experiment that stops
    stripping zero row/col bases; it over-restricts legitimate higher-rank TMEM
    indexed views and is not the right fix for the multidim-slice miscompile.
  - some older TMEM conversion lit chunks no longer emit LLVM in the current
    pipeline at all; keep only chunks that survive lowering and check them
    meaningfully.
  - current surviving `ttng.tmem_subslice` descriptor-only LLVM lowering often
    returns the base pointer unchanged and relies on descriptor semantics rather
    than a materialized pointer delta. Tests should check the current emitted
    LLVM, not the pre-linearization offset assumptions.
