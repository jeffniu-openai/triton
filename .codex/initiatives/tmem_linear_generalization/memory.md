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
- This devbox can be unstable. Land TMEM work in frequent, modular commits
  with commit messages that explain context, motivation, implementation, and
  validation, then push each commit to `jeffniu-openai/codex/tmem` so the
  remote branch remains a recoverable checkpoint.
- `M=64` MMAv5 accumulator ld/st row plans are an explicit producer-owned
  contract on the backing `ttng.tmem_alloc` via
  `ttng.tmem_ldst_row_plan`; producer-side TTGIR construction (currently
  `AccelerateMatmul` and the manual Gluon `tcgen05_*` builders) is
  responsible for attaching it, and lowering/query helpers must consume that
  contract instead of inferring it from alloc users.
- On this arm64 devbox, full `make` currently needs
  `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13`
  to unblock the unrelated GSan runtime build.
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
- TMEM `memdesc_subslice` inference must preserve the source logical-to-physical
  projection when it synthesizes a standalone TMEM-linear result. If projecting
  away sliced dimensions would densify sparse physical TMEM row/col bits, the
  view is unsupported and must fail cleanly instead of being re-encoded to a
  different physical mapping.
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

- Current TMEM runtime-matrix state (2026-03-27 01:10 UTC):
  - parser/frontend:
    - TMEM slice semantics in Gluon now follow shared descriptors by default;
      basic parser coverage should use explicit `dim=1` when it wants the old
      N-slice behavior.
    - the corresponding IR now uses generic `ttg.memdesc_subslice` in the
      frontend test, which is expected.
  - positive runtime coverage newly added:
    - TMEM linear subslice-view `tcgen05.copy` now has real runtime coverage
      for `f32`/`i32` across swizzles `32/64/128`;
    - MMA now has a positive runtime case where the accumulator is obtained via
      `memdesc_index` from a larger TMEM descriptor, for both legacy TMEM sugar
      and lifted `tensor_memory_linear`.
  - BUG bucket kept explicit in tests:
    - there is still a coherent cluster of multidimensional TMEM descriptor
      slice/view cases that compile but produce wrong values at runtime.
    - these cases are intentionally tracked as `pytest.xfail` with `BUG`
      reasons, not promoted to positive coverage and not hidden.
    - representative bug symptoms:
      - higher-rank TMEM slice/view chains write the wrong physical columns;
      - direct multidimensional slice views on `mixed` TMEM layouts update the
        wrong region;
      - two-CTA MMAv5 higher-rank view chains still fail at inference time with
        a CGA mismatch, which is now the expected clean diagnostic.
  - supported boundary remains:
    - plain ld/st, copy, MMA, scaled MMA, and many higher-rank TMEM
      `index`/`reshape`/`transpose` compositions are runtime-covered and pass;
    - `warpx2` copy remains clean unsupported;
    - multidimensional TMEM slice semantics are partially implemented, but not
      yet trustworthy enough to remove the BUG xfails.

- Current TMEM runtime-matrix state (2026-03-27 03:50 UTC):
  - the multidimensional TMEM slice/view BUG bucket is now closed in-tree:
    - the full runtime matrix is `1606 passed, 117 skipped`;
    - there are no remaining TMEM BUG/xfail markers in
      `python/test/gluon/test_tmem_runtime_matrix.py`.
  - `test/TritonNvidiaGPU/ops.mlir` is fixed again:
    - raw `triton-opt test/TritonNvidiaGPU/ops.mlir` now exits `0` and emits
      normal stdout;
    - the focused lit bundle over
      `test/TritonNvidiaGPU/ops.mlir`,
      `test/TritonNvidiaGPU/invalid.mlir`,
      `test/Analysis/test-buffer-region.mlir`, and
      `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
      is green.
  - generic TMEM view typing/parsing boundary:
    - public `ttg.memdesc_subslice` now expects the canonical smaller
      TMEM-linear result encoding when the sliced view is materializable as a
      standalone TMEM-linear memdesc;
    - `ttng.tmem_subslice` remains compatibility sugar, but verifier/equality
      checks now compare normalized TMEM-linear layouts so semantically
      identical canonical encodings do not fail parse/verify because of hidden
      layout-name differences.
  - remaining plain TMEM descriptor/load-store clean negatives are no longer
    pointing at `TensorMemoryUtils.cpp` typing bugs:
    - identity quarter-tile `32x32` views from reshape/slice chains;
    - lifted-layout `dim0` / `half_rows` descriptor views;
    - scrambled/permuted descriptor views;
    - the block-basis reinterpret frontier.
    These all currently mean “no single direct TMEM register layout exists for
    the final view”. Under the current plain-TMEM contract (no silent layout
    repair for ld/st/alloc), they should be treated as direct-lowering/atom
    limits rather than small local compiler bugs.
  - direct-PTX `tcgen05.copy.warpx2` remains a real ISA frontier, but it is no
    longer the remaining blocker for the public TMEM descriptor API:
    - PTX families `warpx2::{02_13,01_23}.64x128b` are real and deterministic;
    - Triton already recognizes/classifies them and can emit the opcode names;
    - current verifier/lowering still rely on the inherited warpx4 shared
      descriptor/address schedule, which is semantically insufficient for a
      future user-reachable `warpx2` path;
    - however, bounded searches across the current public shared-layout and
      explicit TMEM-layout constructors did not find a surjective user-visible
      layout that lowers to `warpx2`;
    - the only raw `warpx2`-looking explicit TMEM layouts found so far require
      row-zero bases, are non-surjective, and are rejected at
      `allocate_tensor_memory` before codegen;
    - under the current public API surface, `warpx2` is therefore a
      direct-PTX / future-API frontier rather than a blocker for “all code the
      user can currently write”.

- public TMEM descriptor API reachability audit (2026-03-27 06:18 UTC):
  - bounded shared-layout classification searches:
    - no-scales canonical destination (`128x4`, `f32`):
      - pure interleavings: `36` layouts checked, no `warpx2`;
      - one-mixed: `1134` tested / `252` surjective, no `warpx2`;
      - two-mixed: `254016` tested / `22680` surjective, no `warpx2`.
    - scales destination (`64x16`, `i8`):
      - pure interleavings: `210` layouts checked, all reachable families
        classify to `multicast=3` / `warpx4`;
      - one-mixed: `2400` tested / `480` surjective, no `warpx2`;
      - two-mixed: `1166400` tested / `90720` surjective, no `warpx2`.
  - bounded explicit TMEM-layout search:
    - raw `warpx2`-looking row-zero layouts exist for `[128,4]`, `[128,8]`,
      and `[128,16]`, but they are non-surjective:
      - `[[1,0],[2,0],[4,0],[8,0],[16,0],[0,0],[32,0]]`
        -> raw `multicast=1` / `warpx2::01_23`;
      - `[[1,0],[2,0],[4,0],[8,0],[16,0],[32,0],[0,0]]`
        -> raw `multicast=2` / `warpx2::02_13`.
    - real parser probe confirms those layouts fail at
      `allocate_tensor_memory` with `The layout must be surjective`; no
      `tcgen05.cp` opcode is emitted.
    - surjective explicit TMEM searches still found no `warpx2` hits:
      - `[128,4]`: `254016` tested / `22680` surjective;
      - `[128,8]`: `893025` tested / `68040` surjective;
      - `[128,16]`: `2371600` tested / `152460` surjective.
  - test coverage added / refreshed:
    - `python/test/gluon/test_frontend.py` now directly checks that the
      non-surjective row-zero explicit TMEM layouts fail cleanly.
    - `python/test/gluon/test_tmem_runtime_matrix.py` now matches the current
      clean verifier diagnostics for the historical `warpx2` candidates.
  - practical conclusion:
    - direct PTX `warpx2` support still needs family-specific
      descriptor/address/message synthesis if Triton ever grows a public,
      surjective route to it;
    - that work is not required to support all code currently expressible
      through the TMEM descriptor API.

- warpx2 probe frontier refresh (2026-03-27 04:44 UTC):
  - added durable bounded probe helpers:
    - `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_overlay_both_bases.py`
    - `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_single_site_halves.py`
    - `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_4site.py`
  - aligned second-copy overlay sweep across both live base registers:
    - result:
      `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_overlay_both_bases_gpu0.json`
    - `64 / 64` plans launched and returned structured output;
    - `0` plans produced a clean quadrant mapping;
    - only six unique overwrite signatures appear under aligned `{0,4,8,12}`
      deltas;
    - `%r12 + 0` is not a new semantic family: it duplicates the productive
      `%r14 + 4` right-half signatures from the earlier overlay probe;
    - within the current second-copy descriptor pair (`%rd3/%rd4`), no aligned
      one-message plan writes chunk `3`, left half as an exact expected tile.
  - aligned one-copy / live-descriptor sweep:
    - result:
      `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_single_site_halves_gpu0.json`
    - `28 / 32` plans launched cleanly; `4` plans faulted and remain invalid;
    - this probe exposes four clean quadrant mappings that do not corrupt any
      other half:
      - `tcgen05.cp.warpx2::02_13` via the live `%rd3` site with `%r18 + 4`
        or `%r7 + 0` gives `0B/00/0B/00` (clean exact right halves for chunks
        `0` and `2`);
      - `tcgen05.cp.warpx2::01_23` via the live `%rd3` site with `%r18 + 4`
        or `%r7 + 0` gives `0B/0B/00/00` (clean exact right halves for chunks
        `0` and `1`).
    - despite those new clean signatures, the combined aligned one-message
      coverage from both probe families still reaches every exact half except
      chunk `3`, left half.
  - current inference from the durable probes:
    - current live descriptors plus aligned `{0,4}` destination deltas are not
      sufficient to cover the full tile, even before considering verifier or
      lowering structure;
    - the missing `3L` half is the present hard frontier;
    - if the bounded four-site search over the current live descriptors also
      fails, the next warpx2 step is no longer “find a better schedule over the
      existing descriptors”, but “synthesize or discover a new descriptor
      variant that can write the missing quadrant”.
  - bounded four-site search over the current live descriptors:
    - result:
      `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_4site_gpu1.json`
    - search space:
      - four emitted copy sites from the two-copy overlay kernel;
      - same input buffer fed through both copies;
      - each site kept its live base register and live descriptor register;
      - per-site family `{warpx2::02_13, warpx2::01_23}`;
      - per-site aligned destination delta `{0,4}`.
    - outcome:
      - `256 / 256` plans launched cleanly;
      - `0` exact matches;
      - the best bounded plans still stop at `2040` mismatched bytes.
    - current implication:
      - schedule reshuffling over the current live descriptor/base-register
        family is exhausted for the bounded aligned space;
      - the remaining `warpx2` frontier is descriptor synthesis / descriptor
        discovery for the missing quadrant, not a better assignment of the
        current four live sites.
  - live-bit descriptor cube search:
    - result:
      `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_descriptor_values_gpu3_fast.json`
    - search space:
      - one active `warpx2` site at a time on the one-copy kernel;
      - aligned address immediates `{0,4}`;
      - current live descriptor cube implied by the differing live bits
        `{5,51}` only:
        `0x0000400800000000`,
        `0x0000400800000020`,
        `0x0008400800000000`,
        `0x0008400800000020`;
      - target quadrant remains chunk `3`, left half.
    - outcome:
      - `112 / 128` plans launched cleanly; `16` still faulted;
      - `0` exact hits on the target `3L` quadrant;
      - `0` clean or isolated target hits;
      - the best near misses still only reproduce already-known clean right-half
        signatures such as `0B/0B/00/00`.
    - current implication:
      - the missing `3L` quadrant is not hidden in the current live
        `{baseAddress, matrixBaseOffset}` bit cube;
      - the next bounded descriptor neighborhood should move to
        `leadDimensionBaseOffset` / `strideDimensionBaseOffset` around the
        clean `%rd3` seed rather than widening more schedule or
        `{baseAddress, matrixBaseOffset}` permutations.
  - `%rd3` seed `leadDimensionBaseOffset` / `strideDimensionBaseOffset` search:
    - result:
      `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_descriptor_lbo_sbo_gpu1.json`
    - search space:
      - clean `%rd3` seed only (`0x0008400800000020`);
      - active site `1` only;
      - address seeds `%r18 + 4` and `%r7 + 0`;
      - `warpx2::{02_13,01_23}`;
      - `leadDimensionBaseOffset ∈ {0,4,8}`;
      - `strideDimensionBaseOffset ∈ {4,8,12,16}`.
    - outcome:
      - `36 / 48` plans launched cleanly; `12` faulted;
      - `0` exact hits on target quadrant chunk `3`, left half;
      - `0` clean or isolated target hits.
    - current implication:
      - bounded `leadDimensionBaseOffset` / `strideDimensionBaseOffset`
        perturbations around the clean `%rd3` seed are still insufficient;
      - the next plausible PTX-side degree of freedom is source-coordinate
        generation (descriptor values equivalent to changing `smemLoad(a, b)`
        around the clean `%rd3` seed), not more local field tweaks.

MMAv5 accumulator N-subslice support (2026-03-27)

- `tcgen05_mma` and `tcgen05_mma_scaled` now accept accumulator `D` views from
  TMEM-linear parents when the physical allocation is still legacy-MMAv5
  compatible and the view preserves the full `M` extent while narrowing `N`.
- Implementation boundary:
  - new helper `matchMMAv5AccumulatorEncoding(...)` tries the old logical-shape
    legacy matcher first, then retries on `allocShape` for
    `tensor_memory_linear` accumulators only when `shape[-2] == allocShape[-2]`;
  - the helper is used only for accumulator-side verifier/lowering/allocator
    consumers, not for generic TMEM matching.
- New coverage:
  - plain MMAv5:
    - verifier/roundtrip regression for public `memdesc_subslice` accumulator
      views in `test/TritonNvidiaGPU/ops.mlir`;
    - LLVM lowering regression for a second-half `N` subslice from a
      TMEM-linear accumulator parent in
      `test/Conversion/tritongpu_to_llvm_blackwell.mlir`;
    - runtime-matrix positives for indexed and second-half `N`-subslice
      accumulator views from legacy and TMEM-linear parents.
  - scaled MMAv5:
    - LLVM lowering regression for a second-half `N` subslice from a
      TMEM-linear accumulator parent;
    - runtime-matrix positives for second-half `N`-subslice accumulators from
      legacy and TMEM-linear parents.
- Important non-goals:
  - no widening of A-in-TMEM layout acceptance;
  - no support yet for row-sliced accumulator views (`M` changed);
  - no support yet for arbitrary non-legacy-equivalent MMAv5 accumulator
    layouts.

MMAv5 accumulator indexed / composed view coverage (2026-03-27)

- The accumulator-side MMAv5 widening is now validated beyond simple
  second-half `N` subslices.
- Supported public `D` view shapes now include:
  - `acc_parent.index(i)` from lifted legacy and TMEM-linear parents for plain
    `tcgen05_mma` and `tcgen05_mma_scaled`;
  - `acc_parent.slice(64, 64, dim=1)` from legacy and TMEM-linear parents for
    plain and scaled MMAv5;
  - `acc_parent.index(1).slice(64, 64, dim=1)` from lifted legacy and
    TMEM-linear parents for plain and scaled MMAv5.
- The implementation model is still the same:
  - `matchMMAv5AccumulatorEncoding(...)` first matches the current logical
    shape against legacy MMAv5 families;
  - if that fails and the encoding is `tensor_memory_linear`, it retries on
    `allocShape`;
  - it only accepts views that preserve the full `M` extent and narrow `N`.
- LHS tensor-memory views are still deliberately rejected.
  - attempted `A.slice(..., dim=1)` acceptance compiled but produced wrong
    numerics;
  - the verifier now rejects that path cleanly for both plain and scaled MMAv5,
    and the runtime/lit coverage checks that this remains a clean diagnostic
    rather than a crash or late-lowering failure.
- Current validated MMAv5 view boundary:
  - `D`: indexed, `N`-subslice, and composed indexed+`N`-subslice views are
    supported when the underlying allocation still maps to a legacy MMAv5
    family;
  - `A`: standalone legacy-compatible TMEM allocations remain supported, but
    TMEM subviews are quarantined until the lowering bug is fixed;
  - row-sliced `D` views (`M` changed) and arbitrary non-legacy-equivalent
    accumulator families remain unsupported.

MMAv5 accumulator view composition and LHS-view quarantine (2026-03-27)

- Accumulator `D` coverage now extends beyond the initial single-op slices.
  - plain `tcgen05_mma`:
    - indexed accumulator views from lifted parents;
    - `N`-subslice accumulator views from canonical parents;
    - composed `index(...).slice(..., dim=1)` accumulator views from lifted
      parents.
  - scaled `tcgen05_mma_scaled`:
    - indexed accumulator views from lifted parents;
    - `N`-subslice accumulator views from canonical parents;
    - composed `index(...).slice(..., dim=1)` accumulator views from lifted
      parents.
  - each of the new runtime positives was validated for both:
    - legacy-compatible accumulator parents;
    - lifted `tensor_memory_linear` accumulator parents.
- The accumulator matcher boundary is still intentionally narrow.
  - support is limited to views that remain legacy-MMAv5 compatible after
    normalization;
  - the helper still does not generalize to arbitrary row-changing or
    non-legacy-equivalent accumulator layouts.
- A-in-TMEM subviews remain a real open bug, not a supported feature.
  - public TMEM views on the LHS/A operand currently miscompile when allowed
    through;
  - the verifier now rejects them cleanly for both plain and scaled MMAv5 with
    an explicit diagnostic instead of permitting wrong-code;
  - dedicated verifier/runtime regressions now pin that boundary down.
- Current validated MMAv5 view boundary:
  - supported:
    - accumulator `D` index;
    - accumulator `D` `N` subslice;
    - accumulator `D` `index -> N-subslice` composition;
    - plain and scaled MMAv5;
    - legacy and TMEM-linear lifted parents.
  - unsupported:
    - LHS/A TMEM subviews;
    - row-sliced accumulator views (`M` changed);
    - arbitrary non-legacy-equivalent accumulator families.

MMAv5 accumulator view-chain expansion and LHS TMEM-view quarantine (2026-03-27)

- Supported accumulator view family is now broader inside the existing
  MMAv5-compatible/canonical boundary.
  - plain `tcgen05_mma` and `tcgen05_mma_scaled` now have positive runtime
    coverage for `D.index(i).slice(..., dim=1)` when the underlying allocation
    remains a legacy-MMAv5-compatible `128x128` accumulator and the final view
    preserves full `M` while narrowing `N`;
  - TMEM-linear coverage uses lifted rank-3 parents, so the composed
    `memdesc_index -> memdesc_subslice` path is exercised end-to-end before
    MMAv5 lowering.
- Implementation model did not need another matcher widening beyond the
  accumulator-side `matchMMAv5AccumulatorEncoding(...)` helper.
  - the helper already keys off the allocation shape for accumulator views, so
    composed `index(...).slice(...)` chains work as long as the trailing
    rank-2 allocation remains MMAv5-compatible and the final view keeps full
    `M`.
- A-in-TMEM public views are still an open correctness bug.
  - positive LHS TMEM `subslice` attempts miscompiled at runtime for both
    legacy and TMEM-linear parents, so the compiler now deliberately rejects
    MMAv5 LHS tensor-memory views whenever the logical shape differs from the
    allocation shape;
  - this is covered by direct verifier regressions for plain and scaled MMAv5
    plus a Gluon runtime-matrix clean-negative regression that checks the
    explicit diagnostic and no verifier/pass-manager crash.
- Current evidence for the remaining MMAv5 frontier:
  - the currently rejected row/col-mixed and low-bit-permuted accumulator
    layouts are not supportable by a small verifier/matcher tweak alone;
  - current MMAv5 lowering still assumes legacy-compatible accumulator tiles
    and materializes only base-plus-row/col offsets per instruction, so
    arbitrary PTX-level accumulator swizzles would require a more general
    accumulator address model.
- Validation after this slice:
  - lit bundle:
    `test/TritonNvidiaGPU/ops.mlir`,
    `test/TritonNvidiaGPU/invalid.mlir`,
    `test/Conversion/tritongpu_to_llvm_blackwell.mlir` -> `3 passed`;
  - focused runtime MMA accumulator/view cluster -> `13 passed`;
  - full TMEM runtime matrix -> `1616 passed, 117 skipped`.

MMAv5 canonical large-N accumulator normalization (2026-03-27)

- Plain MMAv5 now supports larger canonical accumulators along `N` when they
  span multiple instructions but still use canonical TMEM packing.
  - the accumulator matcher now picks the largest legacy family consistent
    with the same physical TMEM layout and the MMAv5 per-instruction
    `N <= 256` limit, rather than hard-failing on oversized canonical
    `blockN` values;
  - oversized explicit legacy layouts normalize through their equivalent
    canonical TMEM-linear form before that capped family search, so explicit
    `TensorMemoryLayout((128, 512), col_stride=1)` and its linear equivalent
    both reach the same supported MMAv5 family.
- New user-visible support:
  - plain `tcgen05_mma` with legacy and TMEM-linear `128x512xf32`
    accumulators now compiles and runs correctly, emitting four
    `tcgen05.mma.cta_group::1.kind::f16` instructions and producing correct
    `A @ B + C` numerics.
- Permanent coverage:
  - `python/test/gluon/test_tmem_runtime_matrix.py` now contains
    `test_tmem_runtime_matrix_mma_large_n` covering both legacy and
    TMEM-linear `128x512` accumulators and asserting the four-op PTX/LLIR
    shape.
- Important boundary:
  - this widening is currently validated for plain MMAv5 only;
  - scaled `128x512` still fails separately because the overall TMEM resource
    footprint exceeds hardware limits (`Required: 532, Hardware limit: 512`),
    so that path was not widened by the accumulator matcher change;
  - arbitrary non-canonical / tile-permuted accumulator layouts are still not
    representable by the current legacy-family matcher plus base-offset MMAv5
    lowering model.
- Validation after this slice:
  - lit bundle remains `3 passed`;
  - focused runtime MMA accumulator/view cluster including `mma_large_n` ->
    `15 passed`;
  - full TMEM runtime matrix -> `1620 passed, 117 skipped`.

MMAv5 one-CTA tile-permuted TMEM-linear accumulators (2026-03-28)

- MMAv5 accumulator lowering now supports a bounded new non-legacy family:
  one-CTA TMEM-linear accumulators whose intra-instruction MMAv5 tile packing
  is canonical, but whose whole MMAv5 tiles are permuted in TMEM.
  - the accumulator matcher in
    `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` now has a second path after
    exact legacy-equivalent matching:
    - it checks TMEM-linear accumulators against real MMAv5 instruction-width
      families `{64, 128, 256}` along `N`;
    - it requires the low bits that define the MMAv5 tile packing to match the
      canonical family exactly;
    - it allows higher bits only when they move by whole MMAv5 tiles, so the
      layout can permute tile-grid order without changing the per-instruction
      tile image.
- Scaled MMAv5 now shares the same accumulator-address model as plain MMAv5.
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
    switched `convertScaledDot(...)` from the old row-major block-id
    accumulator addressing to `DotOpMmaV5TmemLoader`, so both plain and scaled
    MMAv5 now address accumulators through the full TMEM linear layout.
- New user-visible support:
  - plain `tcgen05_mma` now accepts and runs a `128x256xf32`
    `TensorMemoryLinearLayout` whose 64-wide MMAv5 tiles are permuted along
    `N`;
  - scaled `tcgen05.mma_scaled` accepts the same accumulator family and
    produces correct numerics.
- Permanent coverage:
  - `python/test/gluon/test_tmem_runtime_matrix.py` now contains:
    - `test_tmem_runtime_matrix_mma_tile_permuted_layout`;
    - `test_tmem_runtime_matrix_mma_scaled_tile_permuted_layout`;
    - both assert numerics, exact PTX/LLIR opcode agreement, and
      `tensor_memory_linear` staying present in TTGIR;
  - `test/TritonNvidiaGPU/ops.mlir` now contains a direct verifier/round-trip
    positive for a tile-permuted MMAv5 accumulator.
- Important safety note:
  - the first matcher draft was too permissive and briefly let scrambled /
    row-col-permuted accumulator negatives through, which later faulted with
    GPU illegal instructions during the full runtime matrix;
  - restricting the new matcher path to actual MMAv5 instruction widths
    `{64, 128, 256}` restored clean-negative behavior for the old mixed /
    scrambled / row-col-permuted tests while keeping the intended tile-grid
    permutation positives.
- Current MMAv5 boundary after this slice:
  - supported:
    - canonical legacy and TMEM-linear accumulators;
    - canonical large-`N` accumulators (`128x512`);
    - public accumulator view chains that preserve full `M` and narrow `N`;
    - one-CTA tile-permuted TMEM-linear accumulators that keep MMAv5
      intra-tile packing canonical;
    - scaled MMAv5 with that same one-CTA accumulator family.
  - unsupported:
    - low-bit row/col-permuted or mixed/scrambled accumulator layouts;
    - two-CTA non-legacy tile permutations;
    - LHS/A TMEM public views.
- Validation after this slice:
  - lit bundle -> `3 passed`;
  - focused MMA positive cluster including tile-permuted + view cases ->
    `17 passed`;
  - focused negative/positive boundary cluster for exotic/permuted layouts ->
    `19 passed`;
  - full TMEM runtime matrix -> `1622 passed, 117 skipped`.

MMAv5 tile-permuted large-`N` follow-up (2026-03-28)

- The new one-CTA tile-permuted MMAv5 accumulator support also covers larger
  `N` tiles beyond the initial `128x256` case.
  - plain `tcgen05_mma` now has direct runtime coverage for:
    - `128x512` accumulators with 64-wide tile-grid permutations;
    - `128x512` accumulators with 128-wide tile-grid permutations.
- The observed instruction counts are now pinned in the runtime matrix.
  - `128x256` / tile `64` -> `8` MMAv5 instructions;
  - `128x512` / tile `64` -> `16` MMAv5 instructions;
  - `128x512` / tile `128` -> `8` MMAv5 instructions.
- Scaled MMAv5 remains resource-limited for the `128x512` tile-permuted cases.
  - the same TMEM-capacity limit still applies:
    `OutOfResources ... Required: 532, Hardware limit: 512`.
- Permanent coverage:
  - `python/test/gluon/test_tmem_runtime_matrix.py`
    `MMA_TILE_PERMUTED_CASES` now includes the two `128x512` tile-permuted
    plain-MMAv5 layouts.
- Validation after this follow-up:
  - focused tile-permuted sweep -> `4 passed`;
  - full TMEM runtime matrix -> `1624 passed, 117 skipped`.
## Plain MMAv5 LHS TMEM view boundary (2026-03-28)

- I tested the tempting broadening first: remove the explicit MMAv5 TMEM-LHS-view verifier ban and treat the runtime-matrix `mma_lhs_subslice` bucket as positive.
- That path is not safe. The partial-`K` cases (`128x64 -> 128x32` for plain, `128x128 -> 128x64` for scaled) compile but produce wrong results; plain MMAv5 was heavily `NaN`-corrupted and scaled MMAv5 was numerically wrong.
- Existing Gluon core coverage showed the real boundary: `python/test/gluon/test_core.py::test_block_m_64_mma` already uses TMEM operand-A `memdesc_subslice` views successfully for the manually tiled `blockM=64` path. So the blanket verifier ban was too broad, but “all TMEM-LHS views are fine” was also wrong.

Current rule implemented in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`:
- plain `tc_gen5_mma` accepts TMEM LHS views only for direct `ttg.memdesc_subslice` tiles;
- the view must preserve full `M`;
- the `K` extent must be a whole `blockN` tile of the parent TMEM family;
- the `K` offset must be aligned to that parent `blockN`.

Implications:
- partial-`K` LHS views stay cleanly unsupported and are covered in the runtime matrix;
- the known-good blockM=64 whole-tile TMEM-LHS slice path stays supported;
- `tc_gen5_mma_scaled` is still conservative for TMEM LHS views (no validated positive scaled-LHS view family yet).

Coverage added/updated:
- `test/TritonNvidiaGPU/invalid.mlir` checks the new plain-MMA partial-`K` rejection text;
- `test/TritonNvidiaGPU/ops.mlir` has a positive plain-MMA whole-tile TMEM-LHS subslice example;
- `python/test/gluon/test_tmem_runtime_matrix.py` keeps the partial-`K` runtime probes as clean negatives;
- `python/test/gluon/test_core.py::test_block_m_64_mma` remains the positive runtime proof for TMEM operand-A tile views.

TMEM ld/st clean-negative diagnostic normalization (2026-03-28)

- The frontend default TMEM register-layout variant for several clean-negative
  paths is now `"auto"` rather than the historical explicit `"32x32b"`.
- Runtime-matrix clean-negative assertions were normalized to accept either
  variant spelling where the underlying unsupported reason is unchanged.
  - updated checks:
    - `ldst_descriptor_multidim_slice_identity_reports_clean_error`;
    - `ldst_descriptor_multidim_slice_reports_clean_unsupported`;
    - `cp_no_scales_linear_unsupported_shape_reports_clean_error`.
- This is test-expectation hygiene only; it does not widen ld/st lowering.
- Validation status after the normalization:
  - targeted assertion cluster (`2` ld/st + `1` cp clean-negative tests):
    `3 passed`;
  - full runtime matrix no longer fails on stale `'32x32b'` text.

MMAv5 32-wide tile-preserving accumulator split (2026-03-28)

- Plain MMAv5 now supports another bounded non-legacy TMEM-linear accumulator
  family: one-CTA layouts that preserve canonical intra-instruction packing for
  `blockN = 32` and only permute whole 32-wide MMAv5 tiles along `N`.
  - direct runtime probes validated:
    - `128x128`, tile width `32`;
    - `128x256`, tile width `32`;
    - `128x512`, tile width `32`;
  - the observed plain-MMAv5 instruction counts are now pinned:
    - `128x128` / tile `32` -> `8`;
    - `128x256` / tile `32` -> `16`;
    - `128x512` / tile `32` -> `32`.
- The widened matcher lives in
  `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`.
  - `matchMMAv5AccumulatorFamily(...)` now considers `blockN = 32` in the
    non-legacy tile-preserving path for plain MMAv5 accumulators.
- Scaled MMAv5 does **not** share this new family yet.
  - the same non-legacy `blockN = 32` accumulator layouts compile and run
    correctly for plain `tcgen05_mma`, but produced numerically wrong results
    for `tcgen05.mma_scaled`;
  - the scaled verifier in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` now
    rejects non-legacy 32-wide accumulator layouts cleanly with an explicit
    diagnostic instead of allowing wrong-code;
  - canonical legacy-equivalent 32-wide layouts are not blanket-banned by that
    verifier fence.
- Permanent coverage added:
  - `python/test/gluon/test_tmem_runtime_matrix.py`
    - positive plain-MMAv5 tile-permuted `32`-wide cases for
      `128x128`, `128x256`, and `128x512`;
    - clean-negative scaled-MMAv5 regressions for
      `128x128` and `128x256` non-legacy `32`-wide tile permutations;
  - `test/TritonNvidiaGPU/ops.mlir`
    - direct verifier / round-trip positive for the plain `128x128`
      tile-permuted-32 accumulator;
  - `test/TritonNvidiaGPU/invalid.mlir`
    - direct verifier regression for the scaled non-legacy `blockN = 32`
      rejection.
- Current MMAv5 boundary after this slice:
  - supported:
    - canonical legacy / TMEM-linear accumulators;
    - canonical large-`N` accumulators (`128x512`);
    - one-CTA tile-permuted TMEM-linear accumulators with tile widths
      `{32, 64, 128}` for plain MMAv5;
    - one-CTA tile-permuted TMEM-linear accumulators with tile widths
      `{64}` for scaled MMAv5;
    - public accumulator view chains that preserve full `M` and narrow `N`;
    - whole-`K` aligned plain-MMAv5 TMEM LHS subslices.
  - still unsupported / open:
    - scaled non-legacy `blockN = 32` accumulator permutations;
    - low-bit row/col-permuted or mixed/scrambled accumulator layouts;
    - two-CTA non-legacy tile permutations;
    - arbitrary TMEM LHS public views beyond the current aligned-whole-tile
      plain-MMAv5 subset.
- Validation after this slice:
  - lit bundle:
    `test/TritonNvidiaGPU/ops.mlir`,
    `test/TritonNvidiaGPU/invalid.mlir`,
    `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    -> `3 passed`;
  - focused runtime boundary cluster:
    `mma_tile_permuted_layout`,
    `mma_scaled_tile_permuted_layout`,
    `mma_scaled_tile_permuted_32_reports_clean_unsupported`,
    `mma_exotic_layout_reports_clean_unsupported`,
    `mma_rowcol_permuted_layout_reports_clean_unsupported`
    -> `26 passed`;
  - full TMEM runtime matrix:
    `1639 passed, 117 skipped in 307.17s`.

## 2026-03-27: ld/st selection uses TMEM view-analysis layout

- Added API `getTMemViewAnalysisLinearLayout(shape, encoding, error)` and used
  it in ld/st selection paths:
  - `getDistributedLayoutForTmemLdSt(memType, atom, numWarps)`
  - `getTmemCompatibleLayouts(op, tensorType, memType)`
- This changes non-scales candidate generation from canonical-only extraction
  to view-analysis layout extraction, while keeping `computeTMemLdStEncodingInfo`
  as the final legality gate.
- Focused validation remained green on ld/st descriptor-view clusters:
  - `ldst_descriptor_multidim_slice_positive`
  - `ldst_descriptor_multidim_slice_reports_clean_unsupported`
  - lifted higher-rank clean negatives (still clean-negatives).
- Remaining open gap unchanged:
  - frontend `compute_tmem_reg_layout(...)` still cannot infer a register
    layout when descriptor `shape` rank is lower than preserved encoding rank
    (lifted views), because it does not have full view offset/index history.

## 2026-03-27: explicit MMAv5-only contract restored for user layouts

- Removed the hidden MMAv5 accumulator rematerialization path and the hidden
  TMEM-LHS-view materialization path. Supported MMAv5 layouts still lower
  directly; unsupported user-visible layouts now fail cleanly instead of being
  rewritten behind the user’s back.
- Plain/scaled MMAv5 TMEM LHS views are currently clean-unsupported.
  The exact runtime failure site is:
  - plain/scaled aligned whole-tile LHS subslices can fail early during
    `get_reg_layout(auto)` / TMEM register-layout selection with a clean
    “insert convert_layout explicitly” style diagnostic;
  - other TMEM-LHS view cases fail at the MMAv5 verifier with the explicit
    “LHS tensor memory views are not directly supported” diagnostic.
- Runtime/test state after the revert:
  - lit bundle:
    `test/TritonNvidiaGPU/invalid.mlir`,
    `test/TritonNvidiaGPU/ops.mlir`,
    `test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir`,
    `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    -> `4 passed`;
  - focused clean-negative runtime bucket:
    exotic MMA layouts, row/col permuted MMA layouts, plain/scaled TMEM-LHS
    view rejection, scaled `blockN=32` rejection
    -> `24 passed, 1 xfailed`;
  - focused direct-positive MMA bucket:
    large-`N`, tile-permuted direct layouts, accumulator index/subslice views,
    direct TMEM-LHS layouts, scaled direct-positive variants
    -> `27 passed`.
- New explicit BUG after removing the hidden rewrite:
  - scaled MMAv5 direct linear `128x32` accumulator layout compiles and emits
    code, but runtime numerics are wrong;
  - preserved as explicit `xfail(strict=True)` coverage in
    `test_tmem_runtime_matrix_mma_scaled_blockn32_linear_bug`.

## 2026-03-27: first MMAv5 linear-first planner slice

- MMAv5 verifier/lowering/allocation no longer consume a returned legacy
  `TensorMemoryEncodingAttr` from the TMEM matcher path.
- New API surface:
  - `MMAv5TMemLayoutPlan`
  - `getMMAv5LhsLayoutPlan(...)`
  - `getMMAv5AccumulatorLayoutPlan(...)`
- Current behavior after the refactor is intentionally stable:
  - supported direct MMAv5 layouts still lower directly;
  - unsupported TMEM-LHS views remain clean-negative;
  - focused direct-positive MMA runtime buckets remain green.
- The planner still internally recovers a supported finite MMAv5 family to
  derive instruction tile sizes and packing facts; the important change is that
  verifier/lowering/allocation are no longer coupled to the legacy attr type
  itself.
- Validation after this slice:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - raw `triton-opt` sanity passes for:
    - `test/TritonNvidiaGPU/invalid.mlir`
    - `test/TritonNvidiaGPU/ops.mlir`
    - `test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir`
    - `test/TritonGPU/promote-lhs-to-tmem.mlir`
    - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
  - focused clean-negative MMA runtime slice -> `25 passed`
  - focused direct-positive MMA runtime slice -> `27 passed`
- Remaining MMAv5 linear-generalization gap:
  - the planner still uses finite-family recovery internally, so truly arbitrary
    ISA-realizable linear layouts are not all supported yet;
  - next work should replace that remaining family recovery logic with direct
    analysis over the normalized TMEM linear layout and then widen supported
    cases.

## 2026-03-27: Gluon MMAv5 API contract

- From Gluon, `tcgen05_mma(...)` and `tcgen05_mma_scaled(...)` are now
  explicitly async-only.
- The C++ builders in `python/src/gluon_ir.cc` always set `is_async=true`
  for both ops, even when no completion barriers are attached.
- Gluon docs now state that omitting `mbarriers=` does **not** make the op
  synchronous; callers must explicitly use `tcgen05_commit(...)` +
  `mbarrier.wait(...)` before reading the accumulator.
- Immediate consequence for tests:
  - any Gluon kernel that issues MMAv5 and then directly loads the accumulator
    must either attach completion barriers on the op or do an explicit
    `tcgen05_commit(...)` / `mbarrier.wait(...)` sequence.
- Concrete regression fixed under this contract:
  - `tmem_mma_scaled_blockn32_kernel` now uses explicit commit/wait before
    `acc_tmem.load()`, and the focused runtime bucket is green.
- Validation for this contract change:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - focused frontend parser slice -> `3 passed`
  - focused scaled `blockN=32` runtime slice -> `2 passed`

## 2026-03-27: scaled MMAv5 tile-permuted-32 remains a BUG; higher-rank ld/st tests corrected

- The delegated higher-rank ld/st test conversion was only partly valid.
  - one-CTA higher-rank `dim0_slice` / `half_rows` buckets remain positive.
  - two-CTA higher-rank `dim0_slice` / `half_rows` still fail cleanly at the
    public `get_reg_layout(...)` query boundary, so the runtime matrix should
    keep them as explicit clean negatives.
  - the immediate cleanup was to restore the correct two-CTA negative
    expectations and remove the broken undefined parametrization name so
    `python/test/gluon/test_tmem_runtime_matrix.py` imports again.
- Scaled MMAv5 tile-permuted `tile_n=32` accumulators are **not** supported yet.
  - `128x128` and `128x256` tile-permuted-32 accumulators compile and emit the
    expected scaled MMAv5 PTX op counts (`8` and `16` ops), but runtime
    numerics are wrong; one bounded lowering experiment even escalated the
    bucket to a device misaligned-address fault.
  - keep these cases as explicit `BUG` xfails, isolated with
    `@pytest.mark.forked`, until there is a real lowering fix.
  - do **not** treat them as clean unsupported negatives anymore; compilation
    succeeds.
  - do **not** keep the attempted `numColPerScaleBlockB` tweak in
    `MMAv5.cpp`; it regresses the known-good `256x64` scaled tile-permuted
    path and must stay reverted.
- `test/TritonNvidiaGPU/ops.mlir` now contains a correctly-typed higher-rank
  TMEM `reshape -> subslice -> index` regression using the exact inferred TMEM
  linear encodings for the intermediate rank-3 views:
  - `#tmem_linear_rank3`
  - `#tmem_linear_rank3_small`
  - `#tmem_linear_half_rows`
- Also added a direct `ttng.tc_gen5_mma_scaled` verifier/round-trip regression
  for a tile-permuted-32 accumulator in `ops.mlir`.
- Validation after this slice:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `python -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
  - two-CTA higher-rank ld/st clean-negative slice -> `24 passed`
  - focused scaled tile-permuted / indexed-acc slice -> `3 passed, 2 xfailed`
  - `lit -v test/TritonNvidiaGPU/ops.mlir` -> `1 passed`
- `python/src/gluon_ir.cc`: the `compute_tmem_reg_layout` helper now tries to canonicalize general TensorMemory layouts with `ttng::getTMemViewAnalysisLinearLayout` + `ttng::tryMakeTMemViewEncoding` before querying `ttng::getTmemCompatibleLayouts`, so high-rank descriptor views can be reinterpreted with their canonical TMEM-linear encodings once the remaining row/column contiguity blockers are resolved.
- higher-rank descriptor slice/half-rows runtimes still hit the `TMEM layout '<variant>' unsupported` guard, so the runtime-matrix clean-error cases remain in place until the new canonical fallback stops returning empty layouts.

## 2026-03-28: ld/st fallback and ld.red support boundary

- `computeTMemLdStEncodingInfo(...)` in
  `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp` now attempts a
  view-analysis fallback for TMEM memdesc types:
  - first try standalone canonical TMEM-linear encoding via
    `getCanonicalTMemLinearEncoding(...)`;
  - if that fails, try `getTMemViewAnalysisLinearLayout(shape, encoding, ...)`;
  - only fail when both are unavailable.
- This preserves the direct-support-only contract:
  - no implicit `convert_layout`,
  - no hidden temporary/rematerialization path,
  - still cleanly rejects non-lowerable views.
- `ld.red` boundary (current):
  - positive: `f32` row-wise min/max over `N` with optional `abs` and `NaN`
    propagation when `N` is register-local per row;
  - clean-negative: layouts with `N` sharded across threads, tensor-memory
    scales reductions, and unpacked reduction format.

## 2026-03-28: owned-scope ld/st + ld.red audit rerun

- Full runtime slice in owned scope:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst or ld_red'`
    -> `1143 passed, 112 skipped in 403.56s`.
- Targeted clean-negative boundary slice:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_descriptor_higher_rank_dim0_slice_reports_clean_error_lifted_layout or ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout or ldst_descriptor_multidim_slice_identity_reports_clean_error or ldst_descriptor_multidim_slice_reports_clean_unsupported or ld_red_mixed_layout_reports_clean_unsupported'`
    -> `38 passed`.
- Lit sanity:
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && ninja triton-opt && lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir`
    -> `2 passed`.
- Current owned-scope conclusion:
  - no additional direct-lowering promotions were identified without changing
    semantics;
  - remaining negatives are still legitimate boundaries:
    - unsupported high-rank lifted descriptor views for explicit TMEM ld/st
      variant requests;
    - `ld.red` cases with N sharded across threads.

## 2026-03-28: MMAv5 planner cleanup after transform linearization

- `PromoteLHSToTMem.cpp` now builds canonical TMEM-linear layouts directly
  instead of reconstructing legacy `#ttng.tensor_memory_encoding` attrs
  internally.
- `accelerate-matmul.mlir` checks were updated to match canonical
  `#ttng.tensor_memory_linear<...>` output instead of exact legacy print forms.
- MMAv5 planning in `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` is now more
  linear-first:
  - removed the unconditional legacy-attr fast paths from the generic MMAv5
    family planners;
  - kept only a narrow explicit-legacy preservation path for accumulator /
    scaled-accumulator cases with non-default `colStride`, because current
    canonical TMEM-linear form does not yet uniquely preserve that distinction;
  - kept LHS TMEM planning on the linear-first path.
- Practical rule after this slice:
  - legacy TMEM encodings remain accepted surface syntax / IR sugar;
  - MMAv5 verifier and lowering should prefer canonical TMEM-linear analysis;
  - only preserve explicit legacy family information when needed to avoid
    regressing direct codegen for exact non-default accumulator `colStride`.
- Validation snapshot:
  - transform lit slice passed (`accelerate-matmul`, `promote-lhs-to-tmem`,
    `test_promotion_to_tensor_memory`);
  - focused MMAv5 frontend/runtime slices passed (`6 passed`, then `20 passed`);
  - `test/TritonNvidiaGPU/ops.mlir` lit passed.
- Build note:
  - broad `make -j96` became unreliable in the dirty tree because of unrelated
    link failures in broader targets plus one transient `triton-opt`
    permission-bit glitch;
  - targeted relinks of `triton-opt` and `libtriton.so` were sufficient for
    focused MMAv5/TMEM validation.

## 2026-03-28: one-CTA lifted higher-rank ld/st boundary refined

- The one-CTA lifted higher-rank `ld/st` clean-negative bucket is no longer a
  single class.
- Current precise split:
  - `dim0_slice`, explicit `32x32b`, `n = 64 / 128 / 256`:
    compiles but miscomputes -> strict `BUG` xfail.
  - `half_rows`, explicit `32x32b / 16x64b / 16x128b / 16x256b`,
    `n = 64 / 128 / 256`:
    compiles but miscomputes -> strict `BUG` xfail.
  - remaining one-CTA lifted `dim0_slice` explicit variants
    (`16x64b`, `16x128b`, `16x256b`) still fail cleanly at
    `get_reg_layout(...)`.
- Validation snapshot after reclassification:
  - focused slice -> `9 passed, 15 xfailed`.
- Practical rule:
  - keep the compiling-but-wrong higher-rank view cases explicit as `BUG`,
    not as positives and not as clean unsupported negatives.

## 2026-03-28: copy-planner exploration status

- Attempted one internal copy-descriptor canonicalization experiment by
  reordering TMEM row/col basis bits before descriptor synthesis.
  - reverted immediately because it introduced wrong-code and a late lowering
    assertion.
  - this was the wrong layer: it changed user-visible TMEM logical mapping
    instead of only broadening the descriptor factorization search.
- Current safe copy change:
  - dense non-scales copy planning can now try both the canonical `32x4`
    descriptor factorization and a `64x2` fallback before rejecting.
  - current public negative buckets remain unchanged (`19 passed` on the
    focused no-scales copy clean-negative slice), so this is only a safe
    planner broadening for now, not a user-visible promotion yet.
- `warpx2` status remains unchanged:
  - the historical no-scales candidate still maps to dense `128x128b`;
  - the historical scales candidate still maps to `warpx4`;
  - real `warpx2` still needs both a user-visible representation path and a
    family-specific copy planner/lowering path.

## 2026-03-28: remaining MMAv5 legacy bottlenecks

- Remaining known legacy-family dependencies after the planner cleanup:
  - `matchTensorMemoryLegacyEncoding(...)` still returns legacy family attrs;
  - `planMMAv5LegacyExactFamily(...)` still preserves explicit legacy exact
    families for some accumulator/scaled-accumulator paths;
  - verifier/lowering are still gated by `getMMAv5*LayoutPlan(...)`;
  - `impl::verifyMMAv5Op(...)` still performs its interleaved `blockM=64`
    guard through a legacy-only `TensorMemoryEncodingAttr` check;
  - allocator row-coupling still keys off MMAv5 family plans rather than a
    fully general linear TMEM physical-footprint query.

## 2026-03-28: MMAv5 linear-first cleanup progress

- MMAv5 accumulator and scaled-accumulator planning no longer use a
  `planMMAv5LegacyExactFamily(...)` fast path.
  - Both now share a memdesc-based fallback:
    1. plan the visible view shape directly;
    2. if that fails, canonicalize the memdesc through
       `tryGetCanonicalTensorMemoryEncoding(memDescType, ...)`;
    3. only for full-`M`, narrowed-`N` accumulator views, retry planning on the
       backing allocation shape.
  - This preserves legacy TMEM attrs as parse/source sugar, but moves the
    planner itself further toward linear-first semantics.
- Direct scaled MMAv5 accumulator subslices are now permanently runtime-covered.
  - New user-level coverage allocates a parent `128x128`
    `#ttng.tensor_memory_linear` accumulator, takes a `ttg.memdesc_subslice`
    view, and runs `tcgen05_mma_scaled` directly into that view for `N=32` and
    `N=64`.
  - The focused scaled slice is green: `6 passed`.
- The interleaved `blockM=64` MMAv5 rejection is now linear-first too.
  - `impl::verifyMMAv5Op(...)` now rejects based on the MMAv5 layout plan plus
    physical TMEM row footprint, not only on `TensorMemoryEncodingAttr`.
  - The explicit linear equivalent of the interleaved `blockM=64` family
    (`128x64` with row bases `[1, 2, 4, 8, 64, 16, 32]` and contiguous `N=64`)
    now hits the same clean verifier rejection as the legacy sugar.
- Current MMAv5 validation snapshot after these changes:
  - `lit -v test/TritonNvidiaGPU/ops.mlir` -> `1 passed`
  - `lit -v test/TritonNvidiaGPU/invalid.mlir` -> `1 passed`
  - broad MMA slice -> `52 passed`

## 2026-03-28: scales explicit `16x32bx2` frontend alignment

- The pure descriptor-type TMEM reg-layout helper and the handle-aware memdesc
  helper had drifted apart for tensor-memory scales:
  - the handle-aware path could expose the small explicit `16x32bx2` layouts
    used by the runtime matrix;
  - the pure type-based helper still took the narrower direct memtype path and
    either returned a clean negative or, after selector cleanup, could trip a
    `LinearLayout::getBasis` assertion for `[16, 4]`.
- The selector crash root cause was an unconditional read of the row-16 basis
  in `getDistributedLayoutForTmemLdSt(const LinearLayout&, ...)` even when the
  TMEM row dimension was already only 16 high.
- The current state after the fix:
  - selector no longer crashes on small explicit scales `16x32bx2`;
  - the pure descriptor-type helper now filters the full compatible-layout set
    first, matching the handle-aware path more closely;
  - small explicit `16x32bx2` descriptor-type queries are positive:
    - `[16, 4]`, `num_warps=4`
    - `[16, 8]`, `num_warps=8`
  - larger shapes that still do not have a direct lowering remain clean
    negatives:
    - `[32, 16]`, `num_warps=4`
    - `[32, 4]`, `num_warps=8`
- Focused validation:
  - `python/test/gluon/test_frontend.py` targeted slice -> `2 passed`
  - scales ld/st runtime slice -> `34 passed`

## 2026-03-28: scaled MMAv5 tile-preserving support boundary

- Scaled MMAv5 accumulator planning now shares the same tile-preserving family
  search as plain MMAv5 instead of requiring exact canonical-family equality.
  - Implementation site:
    `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
    (`planMMAv5ScaledAccumulatorFamily(...)` now uses
    `planMMAv5Family(...)`).
- This unlocks a real new direct user-visible layout:
  - `#ttng.tensor_memory_linear` accumulator with shape `128x256` and
    a `tile_n=64` column-tile permutation now lowers and runs correctly with
    `tcgen05.mma_scaled`.
- The repeated-`N=32` tile-permuted scaled case is still intentionally
  unsupported, but the blocker is now the public scales layout contract, not
  the old exact-family matcher.
  - The public `tensor_memory_scales_encoding` exposes matrix-B scale
    fragments only at `64`-column alignment.
  - Repeated scaled `N=32` instructions along `N` still fail cleanly with:
    - `direct block-scaled MMAv5 does not support repeated N=32 instructions along N`
    - `The public tensor-memory scales layout only exposes matrix-B scale fragments at 64-column alignment`
- Current support boundary to remember:
  - direct scaled `blockN=32` canonical layout: supported
  - scaled tile-permuted `tile_n=64` (`128x256`): supported
  - scaled tile-permuted repeated `tile_n=32`: clean negative unless/until the
    public scales layout semantics/API expand
- Validation snapshot:
  - focused new scaled slice -> `3 passed`
  - broader scaled MMAv5 slice -> `7 passed`
  - full TMEM runtime matrix -> `1588 passed, 118 skipped`

## 2026-03-28: reachable non-surjective `warpx2::01_23` copy path

- The public non-surjective TMEM-linear `warpx2::01_23` destination path is
  now live.
  - Implementation site:
    `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
    (`getTMemCopyDescriptorLayouts(...)`).
  - The bounded fix was to widen descriptor candidate generation for
    `warpx2::01_23`:
    - fold source warp-group bits into row/col;
    - rotate leading row-repeat bits out of the core descriptor prefix;
    - promote one extra low-order column bit into the row prefix when needed.
- The user-visible runtime result is intentionally not an identity roundtrip.
  - The supported `warpx2::01_23` layout is non-surjective, so the observable
    logical result is the aliased TMEM view defined by that layout.
  - The runtime matrix now checks the exact aliased output pattern instead of
    only looking for a clean error.
- Current family boundary:
  - `warpx2::01_23.64x128b`: direct positive from user code
  - `warpx2::02_13.64x128b`: still clean unsupported with a descriptor-plan
    diagnostic
- Focused validation:
  - targeted `warpx2` slice -> `3 passed`

## 2026-03-28: MMAv5 LHS subslice support

- Full-`M`, narrower-`K` TMEM LHS descriptor views are now directly supported
  for both plain and scaled MMAv5.
  - Implementation site:
    `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
    (`getMMAv5LhsLayoutPlan(...)` now retries on the backing allocation shape
    when the visible LHS view preserves `M` and only narrows `K`).
  - This is still direct lowering: the actual `ttg.memdesc_subslice` operand
    stays visible in TTGIR and lowering uses the view value directly.
- New permanent user-level coverage:
  - plain `tcgen05_mma` with a TMEM LHS `memdesc_subslice`
  - scaled `tcgen05_mma_scaled` with a TMEM LHS `memdesc_subslice`
- IR coverage:
  - positive verifier/roundtrip coverage for both plain and scaled LHS
    subslices now lives in `test/TritonNvidiaGPU/ops.mlir`
  - the old stale rejection-only invalid cases were removed from
    `test/TritonNvidiaGPU/invalid.mlir`
- Focused validation:
  - plain LHS subslice runtime -> `1 passed`
  - scaled LHS subslice runtime -> `1 passed`
  - `lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir`
    -> `2 passed`
  - full TMEM runtime matrix -> `1592 passed, 118 skipped`

## 2026-03-28: ld/st legacy cleanup and ld.red support matrix update

- The remaining legacy-only M64 ld/st layout-selection hook is gone.
  - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` no longer recovers a legacy
    `TensorMemoryEncodingAttr` just to pick M64 / split-N register layouts.
  - Deleted the dead `matchTensorMemoryLegacyEncoding(...)` helpers after
    confirming the focused M64 / split-N slice stays green on the pure
    linear-first path.
- Validation for that cleanup:
  - focused M64 / split-N / f32 ld-st slice -> `40 passed`
- `tcgen05.ld.red` direct lowering is broader than the older runtime matrix
  covered.
  - Direct hardware-reduction lowering is now permanently covered for:
    - tile-permuted TMEM-linear layouts:
      - `128x128`, `tile_n=32`
      - `128x256`, `tile_n=32`
      - `128x256`, `tile_n=64`
    - all pure row/col-permuted `128x128` TMEM-linear layouts generated from
      `PERMUTED_ROW_COL_LAYOUT_KINDS`
  - Those layouts all still emit `tcgen05.ld.red.sync.aligned.32x32b...f32`
    directly, with exact PTX/LLIR opcode agreement.
  - Mixed TMEM-linear layouts remain on the fallback path with no
    `tcgen05.ld.red` opcodes, which stays the current correct negative
    boundary.
- New runtime coverage in `python/test/gluon/test_tmem_runtime_matrix.py`:
  - `test_tmem_runtime_matrix_ld_red_tile_permuted_linear_layout`
  - `test_tmem_runtime_matrix_ld_red_rowcol_permuted_linear_layout`
  - retained `test_tmem_runtime_matrix_ld_red_mixed_linear_layout_falls_back`
    for the fallback boundary
- Validation snapshot:
  - focused ld.red slice -> `100 passed`
  - broad ld/st + ld.red slice -> `1215 passed, 113 skipped`

## 2026-03-28: `warpx2::02_13` remains a real planner gap; handle-aware ld/st probing is stable

- `tcgen05.copy.warpx2::02_13.64x128b` is still not reachable from the public
  descriptor API through the current direct-lowering path, even after the
  bounded `multicast == 2` descriptor-layout search was widened.
  - The old dead-branch bug in
    `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
    (`getTMemCopyDescriptorLayouts(...)`) is fixed.
  - The current bounded search now tries:
    - row/col fold repartitions of the source warp-group bit
    - row-basis rotations
    - single moved-row permutations
    - bounded row/col reassignment variants
  - Result: the historical non-surjective `warpx2::02_13` test case still
    cleanly rejects, so the remaining gap is deeper descriptor/message
    synthesis, not another trivial candidate-enumeration bug.
- The handle-aware ld/st reg-layout query path in `python/src/gluon_ir.cc` now
  supplements the old type-only candidate list with layouts synthesized using
  the actual memdesc value’s row plan.
  - This uses the new public overload
    `getDistributedLayoutForTmemLdSt(memType, atom, numWarps, rowPlanOverride)`
    from `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h`.
  - It does not silently widen support; it just lets descriptor-view probing
    see the same backing-row-plan information that lowering already uses.
- Two remaining ld/st negatives were re-probed with that handle-aware path:
  - identity multidimensional `32x32` descriptor slice:
    still clean-negative
  - two-CTA higher-rank `half_rows` descriptor slice:
    still clean-negative
- Important failed experiment:
  - adding a 32-row TMEM row-plan made the identity `32x32` slice compile, but
    it miscompiled: the lowered code collapsed the slice to a single
    `tcgen05.{ld,st}.sync.aligned.32x32b.x32.b32` pair at offset `0`, so that
    experiment was reverted immediately.
  - The safe boundary remains clean rejection until the physical TMEM mapping
    for those 32-row/gapped views is modeled correctly.
- Focused validation after the revert:
  - `lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir`
    -> `2 passed`
  - focused runtime slice covering `warpx2`, the remaining ld/st negatives, and
    the direct MMAv5 LHS-subslice positives -> `17 passed`

## 2026-03-28: direct `warpx2::02_13` copy support is live; scaled repeated-`N=32` still needs a new B-scale address model

- `tcgen05.copy.warpx2::02_13.64x128b` is now reachable and correct from the
  public non-surjective TMEM-linear descriptor API.
  - The key fix was not more shared-descriptor enumeration. The direct seed
    descriptor path was already present; the blocker was that
    `getDirectTMemCopySeedDescriptorImm(...)` in
    `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp` was matching the
    full shared-memory linear layout too strictly.
  - After relaxing that matcher to the canonical shared offset bases and
    threading a TMEM ld/st `baseOffset=4` through
    `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`,
    the same `02_13` non-surjective layout now:
    - compiles
    - emits `tcgen05.cp.cta_group::1.warpx2::02_13.64x128b`
    - round-trips correctly through direct `tmem_load`
  - The user-visible runtime expectation is now covered by
    `_expected_tmem_copy_warpx2_02_13_output(...)` and
    `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_candidate_positive`
    in `python/test/gluon/test_tmem_runtime_matrix.py`.

- Important MMAv5 scaled follow-up:
  - The clean rejection for repeated scaled `N=32` instructions along `N`
    remains necessary for now.
  - Bounded experiments showed:
    - removing the verifier/lowering guard makes the
      `tile_n=32`, `128x128` scaled case compile
    - with the current address model, it faults at runtime with
      `CUDA error: misaligned address`
    - the second `N=32` matrix-B scale fragment currently lowers through
      `ttng::getTMemViewOffset(...)` as `origRaw=1` from `(row,col)=(32,0)`
      under the public `TensorMemoryScalesLayout`
    - a bounded rewrite to fold that half-tile into `(row,col)=(0,4)` avoids
      the misaligned-address fault but still gives wrong numerics
      (`max_abs ~= 5e-2`)
  - So the remaining gap is not a stale guard. Repeated scaled `N=32` needs a
    more complete matrix-B scale address model before the clean rejection can
    be lifted safely.

## 2026-03-29: higher-rank ld/st half-row views are still a real BUG and not fixable by a small row-anchor/base-offset tweak

- The remaining `ld/st` runtime failures are the `identity` higher-rank
  half-row positives in
  `test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_positive_lifted_layout`
  from `python/test/gluon/test_tmem_runtime_matrix.py`.
  - Shape pattern:
    - allocate `2x128xN` TMEM
    - `full_view = tmem.index(1)`
    - `view = full_view.reshape((2, 64, N)).slice(1, 1, dim=0).index(0)`
  - Expected behavior:
    - update rows `64:128` of `full_view`
  - Actual behavior:
    - rows `0:64` are updated instead

- The important probe result is negative:
  - I tried all bounded single-atom fixes that still preserve the current
    direct `tcgen05.ld/st` lowering model:
    - local `64`-row anchors
    - backing `128`-row anchors
    - mixed local-selection / backing-emission anchors
    - with and without canceling the extra TMEM row-half base shift
  - None of those variants made the half-row subview hit the correct rows.

- Takeaway:
  - this no longer looks like a small `rowPlan` / `warpBaseOffset` /
    `baseOffset` bookkeeping bug
  - the half-row bucket likely needs a deeper direct decomposition for
    row-half views carved from a larger TMEM backing tile, not another local
  row-anchor tweak
  - after the bounded probe, all exploratory planner mutations were reverted,
    so the tree returns to the last stable baseline for this bug

## 2026-03-29: current stable validation baseline after the scales/`ld.red` cleanup

- The current tree is green on a clean JIT cache:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `rm -rf /root/.triton/cache/* && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=python:. pytest -s --tb=short -n 4 python/test/gluon/test_tmem_runtime_matrix.py`
    -> `1664 passed, 119 skipped in 149.43s`
  - lit bundle:
    - `ops.mlir`
    - `invalid.mlir`
    - `tmem_layouts.mlir`
    - `tritongpu_to_llvm_blackwell.mlir`
    -> `4 passed`

- Important TMEM state on this baseline:
  - scales ld/st direct recovery is live again through the C++ planner path
    rather than a Python-only fallback
  - `ld.red` now obeys the direct-support-or-clean-reject rule; the old Gluon
    fallback to `tmem_load + tt.reduce` is removed
  - MMAv5 legacy `#ttng.tensor_memory_encoding` sugar now re-enters the exact
    direct family path instead of spuriously rejecting in conversion

- Local validation caveat:
  - helper-only Python behavior changes can leave Triton’s on-disk JIT cache
    stale even when the kernel hash does not change
  - clearing `/root/.triton/cache/*` is required when validating changes like
    the `ld.red` fallback removal

## 2026-03-29: remaining semantic debt after the green matrix

- `ld.red`
  - direct lowering is still narrower than generic ld/st
  - targeted coverage is still missing for:
    - descriptor-view `ld.red`
    - scales rejection
    - unpacked rejection
    - two-CTA reduction

- MMAv5
  - the visible legacy-sugar rejection is fixed, but the semantic core still
    compresses arbitrary linear TMEM layouts through the finite
    `MMAv5TMemLayoutPlan` family model
  - the main bottlenecks remain:
    - family planners in `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
    - verifier consumers in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
    - allocator coupling in
      `lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp`
    - final lowering in
      `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`

## 2026-03-29: `ld.red` legality boundary after the 8-warp + linear-layout sweep

- `getTmemLoadReductionLayout(...)` no longer hard-gates to `numWarps == 4`;
  the canonical legacy `256x128`, `num_warps=8` reduction path is supportable,
  and the equivalent canonical TMEM-linear source layout is now permanently
  covered.

- The real `ld.red` direct-support boundary is narrower than generic `ld`:
  - supported:
    - canonical identity TMEM-linear layouts
    - pure-column permutations
    - tile-permuted TMEM-linear layouts
    - the canonical linear form equivalent to legacy
      `TensorMemoryLayout(block=(128, 128), col_stride=1)` at shape `256x128`
  - not directly supported:
    - row-permuted TMEM-linear layouts
    - mixed TMEM-linear layouts that mix row and column physical bits

- Important implementation detail:
  - the reduction source-layout legality predicate already existed as
    `isReductionFriendlyTmemSourceLayout(...)` in
    `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
  - the missing piece was enforcing that predicate at the user-facing helper
    and verifier boundary:
    - `python/src/gluon_ir.cc` now refuses to infer a default reduction layout
      for non-`ld.red`-compatible TMEM source layouts
    - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` now rejects explicit reduction
      ops on those layouts instead of allowing runtime wrong-code
  - the mixed layout still fails because `normalizeTensorMemoryLinearLayoutForAnalysis(...)`
    strips zero bases and requires the `row` dim to present a 128-row block
    composed of pure row bases; the `[0,1]`/`[0,2]` column bits in the mixed
    rows take it out of the canonical set, so `TensorMemoryToLLVM::lowerTMemLdStFromTypes`
    never sees a valid register layout and would emit wrong semantics if the
    predicate were lifted.
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_mixed_linear_layout_reports_clean_unsupported`
    still checks for the "not tcgen05.ld.red-compatible" diagnostic and
    `tt.reduce` suggestion that `TMEMLoadOp::verify` emits, so the clean-negative
    boundary remains the right guard until a new packet decomposition is found.

- Local validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_core.py -k 'test_tmem_reduction_linear_legacy_block_equiv_layout or test_tmem_reduction and 256'`
    -> `32 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_identity_linear_layout or ld_red_tile_permuted_linear_layout or ld_red_col_permuted_linear_layout or ld_red_row_permuted_linear_layout_reports_clean_unsupported or ld_red_mixed_linear_layout_reports_clean_unsupported'`
    -> `104 passed`

## 2026-03-29: half-row ld/st sidecar probe (higher-rank descriptor bug)

- The remaining half-row bug is still open after bounded direct-lowering probes.
  - Repro scripts:
    - `/tmp/test_direct_row_slice.py`
    - `/tmp/test_candidate_layout.py`
  - Both still report `num_diff 4096`.

- Trace state (`TRITON_TRACE_TMEM_QUERY_LOWERING_FILE=1`):
  - support-query lowering is active for the half-row pattern
  - support plan currently resolves to `I32x32b.x1` with:
    - `baseOffset=4194304`
    - `warpBase0=2097152`
    - `warpBase1=32`
  - so the failure is no longer "missing +64 support-origin carry"; the
    remaining issue is the selected mapping family for this direct path.

- One safety fix did land during this probe:
  - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`:
    `getTileLayout(..., warpBasis0, warpBasis1, ...)` now expands row out-dim
    extent from warp basis coverage (power-of-two rounded), analogous to the
    existing column-span logic.
  - This removes an invalid-basis abort seen when probing fallback layouts:
    `Invalid basis 64 for in-dim 'warp' and out-dim 'row'`.

- Reverted experiment:
  - dropping support-base adjustment in
    `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
    did not solve the bug and was reverted.

- Takeaway:
  - no clean direct-lowering fix for the half-row positive bucket yet
  - likely needs a new planner/decomposition for row-half views carved from
    larger backing tiles, not another local base-offset tweak.

## 2026-03-29: multidimensional `32x32` descriptor views should not use
canonical-surrogate query types

- The bad multidimensional `reshape -> subslice -> reshape` bucket regressed
  after the half-row quarantine because the view-like `32x32` descriptors were
  still allowed to reach `get_reg_layout(auto)` through
  `getTMemLdStQueryTypes(...)` canonical-surrogate query types.

- Important distinction:
  - support-query rescue and physical-support rescue were already disabled for
    view-like `32x32` descriptors
  - that was not sufficient
  - the remaining leak was the surrogate query type built from the larger
    backing tile, which let verifier/frontend/lowering recover a `128-row`
    plan for a final `32x32` view that is not directly materializable

- Correct rule for this family:
  - explicit view-like `32x32` TMEM descriptors must be judged on their exact
    final view
  - if the final view is directly supported, keep it
  - if only a surrogate/backing-tile type is supported, reject it instead of
    silently widening the physical plan

- Concrete implementation now in tree:
  - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
    - `getTMemLdStQueryTypes(...)` suppresses canonical-surrogate query types
      for explicit view-like `32x32` descriptors
  - `python/src/gluon_ir.cc`
  - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
    - raw-query probing for that same family no longer injects a backing
      row-plan override

- Observed effect:
  - identity and scrambled-cols multidimensional `32x32` subviews are back to
    clean compile-time rejection
  - mixed multidimensional `32x32` subviews remain directly supported
  - nearby higher-rank ld/st descriptor buckets remain green

## 2026-03-29: current tree reality on copy `warpx2` and the remaining copy gap

- Non-scales `warpx2` is already user-reachable and directly lowered in the
  current tree.
  - Positive runtime coverage now exists for both
    `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b` and
    `tcgen05.cp.cta_group::1.warpx2::02_13.64x128b` from user-written
    `TensorMemoryLinearLayout` values in
    `python/test/gluon/test_tmem_runtime_matrix.py`.

- The copy frontier has therefore moved:
  - the remaining open copy cases are no longer “make non-scales `warpx2`
    reachable”
  - the likely next direct-support slice is dense row/col-permuted linear
    layouts that still fail cleanly because copy-family recognition or
    descriptor-plan synthesis does not accept them yet
  - the scales-side historical `warpx2_candidate` path remains clean
    unsupported

- Full TMEM runtime matrix status after the latest ld/st + expectation fixes:
  - `1668 passed, 119 skipped`

## 2026-03-29: MMAv5 accumulator-side verifier/lowering now uses direct
accumulator layout info

- The D-side MMAv5 consumer stack no longer has to thread the recovered
  `MMAv5TMemLayoutPlan` through verifier, allocation, and lowering just to
  recover `instrShapeM/N`, `colStride`, and `twoCTAs`.

- New public query now in tree:
  - `getMMAv5AccumulatorLayoutInfo(...)`
  - `getMMAv5ScaledAccumulatorLayoutInfo(...)`

- The new info bundles:
  - canonical TMEM linear layout
  - selected MMAv5 instruction sizes `mmaSizeM/N`
  - `colStride`
  - `twoCTAs`
  - `interleavedM64`

- Current state after this slice:
  - D-side plain/scaled MMAv5 verifier checks use the new accumulator info
  - D-side MMAv5 lowering uses the new accumulator info
  - TMEM allocation row-anchor constraints use the new accumulator info
  - the old `MMAv5TMemLayoutPlan` still remains on the LHS / promotion side,
    so the full MMAv5 stack is not fully free of the recovered family-plan API
    yet

## 2026-03-29: pure row-permuted `ld.red` source layouts are directly
supported

- The old `ld.red` source-layout recognizer already tolerated:
  - tile permutations
  - pure col permutations
  - pure row-carry permutations in the col dimension

- But it still hard-required the row bases themselves to appear in canonical
  order. That was stricter than the direct lowering actually needed.

- The fix is intentionally small:
  - in `isReductionFriendlyTmemSourceLayout(...)`, collect the row bases,
    sort them, and compare against the expected canonical basis set
    `{1, 2, 4, 8, 16, 32, 64}`
  - keep all the other `ld.red` direct-support constraints unchanged

- Result:
  - pure row-permuted TMEM linear source layouts are now positive runtime
    cases for `ld.red`
  - mixed layouts remain clean negatives
  - col-permuted and tile-permuted positives stay green

- Focused validation:
  - `pytest -k 'ld_red_'` -> `104 passed`

- Full TMEM runtime matrix on top of the MMAv5 accumulator-info cleanup plus
  row-permuted `ld.red` support is green again:
  - `1668 passed, 119 skipped in 519.91s`

## 2026-03-29: the remaining plain ld/st `32x32` identity subview hole is
not blocked on row-plan reuse anymore; it is blocked earlier in raw-query
construction

- I tried the smallest raw-query-side extension first:
  - new helper `getDirect32x32SubviewRawRowPlan(...)`
  - canonical contiguous view-like `32x32` f32 subviews can borrow the backing
    row plan on the raw-query path in frontend/verifier/lowering

- That was not sufficient.
  - the focused multidim-slice slice stayed:
    - identity negative
    - mixed positive
    - scrambled-cols negative

- The decisive trace finding:
  - the final `view.get_reg_layout(auto)` for the identity case never reaches
    the raw-query override
  - `inferStandaloneTMemLdStQueryLayout(...)` itself fails first with
    `unsupported tensor memory memdesc_subslice view`

- So the next direct-support slice should be:
  - a dedicated support-query helper for canonical contiguous view-like
    `32x32` subviews
  - it must construct a safe scalarizable query layout plus backing row plan
  - it should remain narrow enough that `scrambled_cols` still stays clean
    unsupported

## 2026-03-29: lifted half-row ld/st is now specifically a TMEM
packet-decomposition problem

- Representative user view:
  - start from a lifted public TMEM descriptor
  - `full_view.reshape((2, M // 2, N)).slice(1, 1, dim=0).index(0)`
  - with the identity `64`-column TMEM layout and `instr_variant='32x32b'`

- What is no longer the blocker:
  - reg-layout discovery itself
  - after letting support-query-capable descriptor views past the old blanket
    reject, the part view can again reach direct-planning code
  - `ViewOpToLLVM.cpp` also now zeroes the row-half dim-0 subslice base, so
    the descriptor SSA value no longer carries the old extra `128<<16` row
    delta from the `reshape -> subslice([1,0,0])` step

- What the current experiments say:
  1. Widened `128x64` support-tile query + lifted `row=64` origin:
     - selects direct `I32x32b`
     - can also be scalarized to `x1`
     - both variants still update rows `0:64` instead of `64:128`
  2. Forcing the logical `64`-row warp anchors (`16/32`) onto that widened
     support path:
     - produces `CUDA error: misaligned address`
  3. Recasting the same view as a direct logical `64x64` query with lifted
     origin:
     - reg-layout selection no longer stays on `I32x32b`
     - it falls back to `I16x32bx2`
     - runtime is still wrong

- So the remaining question is narrower now:
  - does the hardware ISA admit a correct direct packet plan for the upper-half
    `64x64` view at all, under the real row-anchor/alignment rules?
  - the two obvious models are currently both bad:
    - widened support tile + lifted origin
    - direct logical query + lifted origin

- Practical implication for the next slice:
  - either find a genuinely correct packet decomposition for that upper-half
    `64x64` view
  - or turn this specific bucket back into a justified clean negative with a
    precise reason grounded in the tcgen05 row-anchor rules

## 2026-03-29: MMAv5 semantic queries are now linear-layout info queries on
both A and D sides

- The old public `MMAv5TMemLayoutPlan` API is no longer used outside
  `Dialect.cpp`.

- New public shape/layout query split:
  - `getMMAv5LhsLayoutInfo(...)`
  - `getMMAv5AccumulatorLayoutInfo(...)`
  - `getMMAv5ScaledAccumulatorLayoutInfo(...)`

- These info objects now cover the external MMAv5 semantic consumers:
  - plain/scaled MMA verifier
  - TMEM allocation row-anchor coupling
  - MMAv5 LLVM lowering
  - LHS promotion to TMEM

- This does **not** mean the finite-family planner is gone.
  - The current instruction-family search still exists internally in
    `Dialect.cpp`
  - But the rest of the compiler no longer consumes that planner as a public
    semantic API or as a recovered legacy layout answer

- Current remaining MMAv5 legacy/family bottleneck after this cleanup:
  - the internal planner still recognizes only layouts that preserve the
    tile-internal canonical MMAv5 mapping and only vary by whole-tile
    multiples/permutations
  - so tile-permuted layouts stay positive, while mixed / scrambled /
    row-col-permuted layouts still clean-reject

## 2026-03-29: dense no-scales copy permutations are still a real semantic
boundary, not just an over-strict verifier gate

- I tried the smallest dense-family widening for
  `tcgen05.copy.128x{128,256}b`:
  - relax the TMEM row/col basis-order gate in
    `isDirectTMemCopyLayoutSupported(...)`
  - try an extra sorted-basis candidate in
    `getTMemCopyDescriptorLayouts(...)`

- That was not a sound support extension.
  - The representative col-permuted case
    `_make_tmem_linear_layout_permuted(128, 128, "identity", "rotate1")`
    compiled and still emitted canonical
    `tcgen05.cp.cta_group::1.128x256b`, but runtime produced logically rotated
    data instead of preserving the user layout:
    - row 0 became `[0, 64, 1, 65, 2, 66, ...]`
    - `mismatch_count = 16128`

- Interpretation:
  - the current dense copy path preserves the canonical physical TMEM order
  - simply accepting or re-sorting a permuted TMEM linear basis order does not
    make the hardware copy realize the user’s logical row/col permutation
  - so the clean-negative dense-permutation gate remains justified until we
    have a descriptor/address model that can faithfully implement those logical
    permutations

- Action taken:
  - reverted the widening and kept the dense row/col-permuted copy buckets as
    clean unsupported

## 2026-03-29: dense `tcgen05.copy.128x{128,256}b` row/col permutations are
not safely supportable by a descriptor-ordering-only widening

- I ran a bounded experiment on the remaining dense no-scales copy
  clean-negative bucket:
  - temporarily let pure permutations of the canonical TMEM row/col basis
    sets past `isDirectTMemCopyLayoutSupported(...)`
  - temporarily let dense families try a sorted-basis descriptor candidate in
    `getTMemCopyDescriptorLayouts(...)`

- Result on the representative col-permuted layout:
  - kernel:
    `tmem_copy_no_scales_linear_kernel`
  - layout:
    `_make_tmem_linear_layout_permuted(128, 128, "identity", "rotate1")`
  - compile succeeded and still emitted canonical
    `tcgen05.cp.cta_group::1.128x256b`
  - but the logical roundtrip was wrong:
    - `out[0, :16] = [0, 64, 1, 65, 2, 66, 3, 67, ...]`
    - not the original logical tensor order

- Interpretation:
  - the dense copy hardware path is preserving canonical physical TMEM order
    for this descriptor/message family
  - simply reordering descriptor basis candidates does **not** make a logical
    TMEM row/col permutation real
  - so the old clean-negative gate on non-ascending dense TMEM row/col basis
    order is not just conservative; it is preventing wrong-code

- Practical boundary after this experiment:
  - dense pure row/col-permuted TMEM layouts should stay rejected for direct
    `tcgen05.copy.128x{128,256}b` unless we add a descriptor/address model
    that can faithfully realize the logical permutation
  - mixed / scrambled dense-copy negatives remain in the same bucket

## 2026-03-29: lifted row-half ld/st descriptor views still need clean
rejection, not rescue

- Representative user view:
  - `full_view.reshape((2, M // 2, N)).slice(1, 1, dim=0).index(0)` on a
    lifted `[2, M, N]` TMEM descriptor

- Re-checked rescue attempts:
  - raw direct lowering for the `32x32b` identity case still produces the old
    sparse wrong-code footprint:
    - modified rows: `0-15,48-79,112-127`
  - a `128x64` support-query rescue can scalarize to
    `tcgen05.ld/st.sync.aligned.32x32b.x1.b32`, but that over-covers the
    backing tile instead of the logical upper-half view
  - a bounded `64x64` canonical support-query rescue falls back to the
    `I16x32bx2` family with local `16/32` row anchors and reproduces the same
    sparse wrong footprint as the raw direct path

- Interpretation:
  - with the currently reachable `tcgen05.ld/st` packet families, this lifted
    row-half descriptor view still does not have a direct semantics-preserving
    lowering
  - leaving the rescue path enabled only turns the clean-negative verifier
    boundary into wrong-code

- Action:
  - removed the half-row support-query rescue path from
    `getTMemLdStSupportQueryLayout(...)`
  - kept the existing clean-negative verifier wording for lifted row-half TMEM
    descriptor views

- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout'`
    -> `12 passed`

- 2026-03-29: row/col-permuted MMAv5 TMEM-linear accumulator layouts still behave as a clean negative because an attempted run-time conversion back to AutoLayout (`ttgl.convert_layout(out, ttgl.AutoLayout())`) failed to compile (`tt.make_range` inference error + `PassManager::run failed`) after a successful `TRITON_BUILD_WITH_CCACHE=true make -j96`, so the existing verifier rejection is safer than forcing the wrong code path.

- 2026-03-29: barriered Gluon MMAv5 kernels should not also emit explicit
  `tcgen05_commit(...)`
  - `tcgen05_mma(..., mbarriers=[bar])` already performs completion-barrier
    signaling
  - an added explicit `tcgen05_commit(bar)` produced duplicate
    `tcgen05.commit...mbarrier::arrive::one...` instructions in PTX/LLIR and
    caused the runtime kernel to wedge
  - removing the redundant commit restored the canonical plain MMA and indexed
    accumulator-view runtime positives

- 2026-03-29: MMAv5 family matching should preserve exact in-instruction basis
  order
  - the direct-layout matcher was too permissive when it compared low-order
    row/col bases as a sorted multiset
  - that admitted scrambled and row/col-permuted TMEM-linear accumulator
    layouts even though the lowering only supports canonical in-tile placement
    plus whole-tile repetition/permutation
  - tightening the matcher to require exact low-order basis order restores the
    desired boundary:
    - row/col-permuted and scrambled layouts -> clean verifier rejection
    - tile-permuted whole-tile layouts -> still directly supported
    - scaled repeated-`N=32` tile-permuted layouts -> still clean verifier
      rejection via the dedicated block-scaled rule

- 2026-03-29: Gluon frontend CTA checking for two-CTA TMEM layouts must honor
  the real CGA shape
  - the old `_check_tensor_memory_layout_ctas(...)` helper treated every
    `two_ctas=True` layout as exactly `2` CTAs per CGA, even when the provided
    `cga_layout` represented a larger CTA grid
  - this broke the Gluon multicast-MMA core tests once TMEM layouts started
    flowing through larger two-CTA CGA configurations such as `8` and `16`
    CTAs per CGA
  - the helper should count CTAs from the explicit `cga_layout` whenever one
    is present, and only fall back to `2` for legacy two-CTA layouts without
    an explicit CGA description

## 2026-03-29: multidimensional `32x32` ld/st identity/mixed split restored

- Remaining regression before the latest rebuild:
  - full TMEM runtime matrix had exactly two failures:
    - the identity multidim `32x32` slice compiled instead of rejecting
    - the mixed multidim `32x32` slice compiled but miscomputed

- Root causes:
  - identity:
    - the explicit descriptor-view type was an **exact dense row/col `32x32`
      TMEM-linear layout**, and the dedicated subview-support rescue still
      admitted it even though it came from a larger backing tile
  - mixed:
    - the direct scalar `32x32b.x1` path was still valid, but the scalar
      packet recovery code advanced PTX `.b32` col-immediates in units of `1`
      instead of `4` bytes, so the emitted sequence became `0..7` instead of
      `0,4,8,...,28`

- Fixes:
  - `TensorMemoryUtils.cpp`
    - exact dense row/col `32x32` descriptor views from explicit backing-tile
      view chains now stay on the clean-negative path
    - the dedicated subview-support rescue declines that exact-canonical case
  - `TensorMemoryToLLVM.cpp`
    - scalarized `32x32b.x1` recovery now steps `.b32` immediates by `4`
      bytes per packet

- Result:
  - identity -> clean compile-time rejection again
  - mixed -> correct numerics and the expected `tcgen05.ld/st.sync.aligned.32x32b.x1.b32`
    offset sequence
  - scrambled-cols -> still clean unsupported

- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'multidim_slice_identity_reports_clean_error or multidim_slice_positive or multidim_slice_reports_clean_unsupported'`
    -> `3 passed`

## 2026-03-30: follow-up validation and direct-support fixes

- Full lit is green again on the current post-fix tree:
  - `cd build/cmake.linux-aarch64-cpython-3.12 && ninja check-triton-lit-tests`
    -> `248 passed, 2 unsupported`

- Direct non-scales ld/st regained support for permuted `64x2` split-N TMEM
  layouts:
  - root cause was a premature non-scales lane/warp broadcast rejection in
    `computeTMemLdStEncodingInfoImpl(...)`
  - `lowerTMemLdSt(...)` already knows how to remove broadcasted registers, so
    rejecting those layouts before it ran was unnecessarily cutting off the
    valid `16x32bx2` / `32x32b_splitn` direct path
  - after removing that precheck, the full
    `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep` bucket is
    back to green (`224 passed`)

- Tensor-memory-scales `warpx2` candidate copies remain intentionally
  unsupported, but their late-lowering diagnostics now re-emit the family-level
  `tcgen05.copy` guidance:
  - lowering still fails after shared-memory allocation because no compatible
    descriptor plan exists
  - `TensorMemoryToLLVM.cpp` now emits the
    `maps to tcgen05.copy... could not synthesize a compatible shared-memory
    descriptor plan for tensor memory scales` diagnostic before the outer
    `PassManager::run failed`
  - runtime-matrix tests now accept the outer exception wrapper while checking
    the real unsupported reason

- Current Python validation checkpoint:
  - `python/test/unit/language/test_compile_only.py` -> `7 passed`
  - `python/test/gluon/test_core.py -k 'tmem_reduction or tcgen05_mma_multicast_commit'`
    -> `92 passed`
  - `python/test/unit/tools/test_triton_to_gluon.py` -> `16 passed`
  - split 4-way run over
    `python/test/gluon/test_frontend.py` +
    `python/test/gluon/test_tmem_runtime_matrix.py`
    -> aggregate `1873 passed, 119 skipped`
  - focused `triton_kernels` nvfp4 persistent slice
    -> `1 passed, 1 skipped`


## 2026-04-01: GB200 unit-suite MMAv5 `test_dot` regression

- The first full `python/test/unit` GB200 microbatch uncovered a real compiler regression outside the TMEM runtime matrix:
  - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
  - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-chain-dot-ieee-bfloat16-float32-1-None]`
- Origin/main passed those nodes. The current tree failed with ~49.5% mismatches.
- Root cause was not the MMA opcode count itself; it was the interaction between canonical explicit M64 TMEM-linear layouts and direct TMEM lowering:
  - canonical M64 register-layout selection had drifted to the narrower split-N `x16` path for ordinary 4-warp 64x64 accumulators
  - after restoring `x32`, initialized `ttng.tmem_alloc %src` still used the type-only ld/st query path, which collapsed the explicit TMEM-linear M64 allocation to a standalone `64x64` query instead of the real `128x64` backing/support form
- The TMEM query trace made the second issue explicit:
  - good path: raw query/support form `row size 128`, `warpBase0=2097152`, `warpBase1=4194304`
  - bad path: standalone `queryType 64x64`, `warpBase0=1048576`, `warpBase1=2097152`
- Fixes:
  - `Dialect.cpp`: keep canonical 4-warp M64 on the full `x32` path
  - `TensorMemoryToLLVM.cpp`: initialized allocs must pass the actual memdesc SSA value to `lowerTMemLdStFromTypes(...)`, not `Value()`, so direct lowering can use the same backing/support query logic as `tmem_store` / `tmem_load`
- Fresh-cache validation after the fix:
  - exact `tf32x3` node -> pass
  - exact `chain-dot-ieee-bfloat16` node -> pass

## 2026-04-01: GB200 unit-suite `test_simple_matmul` epilogue-subtile regressions

- The next GB200 split/group unit batch uncovered another real regression in direct TMEM lowering outside the TMEM runtime matrix:
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-128-128-16-4-float16-float16]`
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-128-128-16-4-float32-tensorfloat32]`
- Both cases use `EPILOGUE_SUBTILE=True`, which lowers the accumulator epilogue through two `ttng.tmem_subslice` views (`N=0` and `N=64`) followed by two direct `ttng.tmem_load`s.
- Failure signatures:
  - TF32: left half matched, right half was an exact duplicate of the left
  - packed f16: after the TF32 fix alone, the right half no longer duplicated the left but still read the wrong packed window
- Root cause split:
  - 32-bit `ttng.tmem_subslice` raw queries need the translated `baseOffset` preserved into lowering; zeroing it drops the `N=64` shift and duplicates the left half
  - packed subword `ttng.tmem_subslice` views cannot reuse the same direct-origin remap path, because they still need the packed-aware fallback query construction to preserve packed column semantics before lowering
- Current rule:
  - preserve raw-query `baseOffset` for `ttng.tmem_subslice` during direct ld/st lowering
  - use the new direct translated-origin remap only for 32-bit-and-up `ttng.tmem_subslice` element types
  - keep subword `ttng.tmem_subslice` on the packed-aware resized-layout fallback and on the standalone-before-raw query ordering
- Validation checkpoint:
  - exact GB200 TF32 node -> pass
  - exact GB200 f16 node -> pass


## 2026-04-01: whole-root TMEM descriptors must bypass view-analysis normalization for direct ld/st selection

- Another GB200 unit-matmul regression surfaced outside the TMEM runtime matrix:
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-64-128-32-4-float16-float16]`
- With `TRITON_PREFER_TMEM_16x256_LAYOUT=1`, current HEAD `69c7822a8` was selecting `16x128b` for the `EPILOGUE_SUBTILE=True` accumulator load/store path, while `/tmp/triton-origin-main` still selected the correct `16x256b.x8.unpack/pack` path.
- The key behavioral difference was not in TTGIR result layout or in PTX opcode spelling. It was earlier:
  - current direct ld/st selection/planning was routing even whole-root TMEM descriptors (`shape == allocShape`) through `getTMemViewAnalysisLinearLayout(...)` + `normalizeTensorMemoryLinearLayoutForAnalysis(...)`
  - origin/main still used the raw storage mapping via `toLinearLayout(shape, encoding)` for whole descriptors
- For true descriptor views and higher-rank slices, the analysis-layout path is still required. But for whole descriptors it is too lossy for legacy-equivalent M64 `f16` whole-tile selection and rejects the working `16x256b` direct path.
- Current rule:
  - whole non-scales descriptors (`shape == allocShape`) use the raw storage layout for direct ld/st layout selection and encoding-info planning
  - descriptor views continue to use the generalized TMEM-view analysis layout path
- Validation checkpoint:
  - exact GB200 `64x128x32` f16 epilogue-subtile node passes again
  - direct PTX repro shows `16x256b.x8.unpack::16b` / `16x256b.x8.pack::16b` restored

- Revalidation note (2026-04-01):
  - the exact GB200 `64x128x32` `f16` epilogue-subtile matmul node is green on clean committed HEAD `69c7822a8`
  - the temporary opcode-shape experiments were a dead end and were discarded; the current tree is back at the committed checkpoint before the full GB200 CI-equivalent sweep

## 2026-04-01: MMAv5 lowering must use raw TMEM-linear accumulator mappings, not recovered legacy families

- A second GB200 unit-matmul regression surfaced after the whole-root ld/st fix:
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-256-128-32-4-float16-float16]`
- The direct PTX symptom isolated the bug:
  - stores were still `tcgen05.st.sync.aligned.16x256b.x16.unpack::16b.b32` as expected
  - the final load could stay a single `tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32` and still be numerically correct
  - the real divergence was the MMAv5 accumulator write addresses: current tree used `[%r437 + 256]`, while `/tmp/triton-origin-main` used `[%r437 + 128]`
- Root cause was in the LLVM lowering loader, not the ld/st selector:
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
  - `DotOpMmaV5TmemLoader::build(...)` was taking a TMEM-linear memdesc, recovering a legacy-like MMAv5 family, then rebuilding the loader address model from that recovered legacy family
  - for whole-tile linear accumulators, that internal legacy normalization can change the physical stepping used by the generated MMAv5 address operands even though the user-visible memdesc is already precise
- Current rule:
  - MMAv5 loader/address generation must use the raw `toLinearLayout(memTy)` mapping for TMEM-linear accumulators/LHS descriptors
  - legacy/TMEM shorthands may still exist in IR/source as sugar, but lowering should not depend on recovering them when a precise linear mapping is already available
- Validation checkpoint:
  - `64x128x32` and `256x128x32` GB200 `f16` epilogue-subtile unit matmuls both pass again
  - PTX for the `256x128` case now restores the `+128` MMAv5 slice offsets
  - `make test-lit` is green again after updating the one stale Blackwell conversion CHECK from `+64` to `+32`


## 2026-04-01: whole-root explicit TMEM-linear M64 accumulator stores must match the analysis/support layout, not the raw encoding form

- Another GB200 unit-matmul regression surfaced during the broader unit-test sweep:
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-1-64-512-32-2-float32-tensorfloat32]`
- The regression was not in MMAv5 proper. It was in the initial whole-root `ttng.tmem_store` that zeroes the accumulator memdesc before the async MMAv5 loop.
- PTX symptom:
  - current tree scalarized the init into many `tcgen05.st.sync.aligned.32x32b.x1.b32`
  - `/tmp/triton-origin-main` kept the direct wide `32x32b.x64.b32` path
- Root cause:
  - `computeTMemLdStEncodingInfoImpl(...)` still had a legacy-anchored M64 rescue path for explicit TMEM-linear descriptors, but that rescue was trying to rediscover the family from the raw encoding form.
  - For the `64x512` explicit linear accumulator, the raw encoding exposes the compact physical span (`256` physical TMEM columns with a row bit spilling into the second output dim), while the direct ld/st planner is already working from the normalized analysis/support layout.
  - That meant the rescue path never recognized the exact legacy-equivalent family and the direct whole-root store fell back to scalar `I32x32b.x1` selection.
- Current rule:
  - whole-root non-view M64 rescue should compare against the normalized analysis/support layout that direct ld/st is actually lowering, not against the raw explicit encoding bits
  - the rescue remains limited to whole-root descriptors (`shape == allocShape`) and only for the standard `32/64` within `128` row-plan override, so descriptor views still go through the generalized view-analysis path
- Validation checkpoint:
  - the tf32 `64x512x32` GB200 unit-matmul node is green again
  - the earlier f16 GB200 control nodes remain green
  - `make test-lit` stayed green after the fix

## 2026-04-01: whole-root explicit TMEM-linear M64 subword ld/st rescue must preserve the raw zero-col-basis layout for lowering

- Another GB200 unit-matmul regression surfaced while resuming the sharded `python/test/unit` sweep:
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-1-64-512-32-2-float16-float16]`
- The first M64 rescue change fixed the scalarization problem but still produced wrong code for the `f16` case.
- PTX symptom after the first fix:
  - current tree recovered the wide `32x32b.x128` direct TMEM store/load again
  - but it emitted the bare 32-bit form (`tcgen05.{st,ld}.sync.aligned.32x32b.x128.b32`) instead of the correct subword form from `/tmp/triton-origin-main`:
    - store: `unpack::16b`
    - load: `pack::16b`
- Root cause:
  - the legacy-anchored M64 helper in `computeTMemLdStEncodingInfoImpl(...)` was using the same stripped whole-root layout for two different purposes:
    - exact legacy-family equivalence matching
    - actual ld/st lowering
  - for subword TMEM layouts those are not the same thing
  - removing zero `col` bases is useful for recognizing that an explicit TMEM-linear layout is equivalent to a legacy `blockM=64` family, but it destroys the spacing information that `lowerTMemLdSt(...)` relies on to select `pack::16b` / `unpack::16b`
  - the helper also copied a `getExpectedTMemLoadValueCount(...)` sanity check that is intentionally defined only for 32-bit ld/st paths, so it will always fail on subword cases
- Current rule:
  - for whole-root explicit TMEM-linear M64 rescue paths, use the stripped layout only to answer “is this exactly legacy-equivalent?”
  - use the raw storage layout to build the direct lowering transform
  - keep the `getExpectedTMemLoadValueCount(...)` post-check restricted to `bitwidth == 32`
- Validation checkpoint:
  - the GB200 `f16` `64x512x32` unit matmul passes again
  - the earlier tf32 `64x512x32` control also stays green
  - PTX now restores the origin-style wide subword forms with `pack::16b` / `unpack::16b`


## 2026-04-01: TMEM subslice physical offsets must be computed in the source tile address space

- A GB200 `triton_kernels` persistent ragged matmul regression remained after the earlier TMEM direct-lowering sweep:
  - `python/triton_kernels/tests/test_matmul.py::test_op[None-True-False-False-False-None-128-768-512-1024-ragged-float16-float16-None-10-1-False-False-None-False-False-False-True-None]`
- Symptom pattern:
  - only the second 128-column half of each 256-column MMAv5 accumulator tile was corrupted
  - the dumped `_p_matmul` TTGIR split the accumulator into `ttng.tmem_subslice {N = 0}` and `{N = 128}` followed by two `ttng.tmem_load`s
  - the dumped PTX then showed both loads coming from the same physical TMEM base, which explains why the `128:256` and `384:512` column bands were wrong
- Root cause:
  - `TMEMSubSliceOpConversion` in `TensorMemoryToLLVM.cpp` was calling `getTMemSubSliceOffset(...)` on the narrowed result memdesc type
  - physical TMEM base arithmetic is defined in the source tile's address space, not the result view's address space
  - for N-half views like `128x256 -> 128x128`, the result type drops the high-order column basis that differentiates the second half, so `{N = 128}` collapsed onto `{N = 0}`
  - `BufferRegion.cpp` had the same stale assumption in its TMEM-subslice region accounting
- Current rule:
  - compute TMEM subslice physical offsets from the source memdesc type in lowering and region analysis
  - the result memdesc type still describes the logical view shape/encoding, but not the address space used to advance the raw TMEM base
  - multibuffer direct-ld/st query plumbing should preserve the already-lowered base translation instead of re-encoding the selected leading buffer as an extra query-origin offset
- Validation checkpoint:
  - the exact persistent ragged `triton_kernels` repro is green again
  - focused conversion/analysis lit checks now encode the correct nonzero subslice offsets


## 2026-04-01: sub-32-bit Blackwell MMAv5 epilogue subtile replay must stay on the original full-load path

- After the TMEM subslice/base-offset fixes, one GB200 unit bucket was still failing:
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-128-128-16-4-float16-float16]`
- Symptom pattern:
  - only the Blackwell `LAYOUT_16x256=True` + `EPILOGUE_SUBTILE=True` path failed for sub-32-bit accumulators
  - the TTGIR replayed the epilogue `reshape -> trans -> split` into two `ttng.tmem_subslice` / `ttng.tmem_load` pairs on `128x64xf16`
  - numerically, the second 64-column half of each 128-column tile was corrupted while the matching TF32 control passed
- Root cause:
  - the remaining bug is in the replay optimization itself (`OptimizeTMemLayouts.cpp`), not in raw TMEM address arithmetic
  - once the earlier source-type TMEM subslice offset fix and raw-query baseOffset fix were in place, the F16 replayed split-load kernel still failed unchanged, which means the sub-32-bit `pack::16b` replayed split-load lowering is not yet semantics-preserving
- Current rule:
  - for sub-32-bit TMEM tiles, do not apply the `reshape -> trans -> split` replay optimization into `ttng.tmem_subslice` / `ttng.tmem_load`
  - keep the original full `ttng.tmem_load` epilogue path until a direct sub-32-bit replay decomposition is proven correct
  - 32-bit replayed split-loads stay enabled
- Validation checkpoint:
  - exact F16 GB200 node is green again
  - matching TF32 control stays green
  - the previously fixed persistent ragged `triton_kernels` MMAv5 repro still passes
  - full lit remains `248 passed, 2 unsupported`

## 2026-04-01: GB200 CI-equivalent sweep checkpoint after `f70f9aeb6`

- Running the post-commit GB200 validation as split GPU microbatches over `CUDA_VISIBLE_DEVICES=0..3` using `/tmp/run_split_batches.sh`.
- Completed so far:
  - rebuild: `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - C++ unit tests: `240 passed`
  - full lit: `248 passed, 2 unsupported`
  - `python/test/unit` main shard: all `32` split groups passed
  - `python/test/unit/test_debug.py`: `95 passed`
  - plugin/custom-op unit slices: all green
- Current in-flight target:
  - `python/triton_kernels/tests` split `32` ways across the `4` GPUs; first wave passed and later waves are still running.

## 2026-04-01: GB200 sweep progress after `triton_kernels` and the rest of `test-unit`

- Additional completed targets:
  - `python/triton_kernels/tests`: all `32` split groups green; aggregated totals from the per-group logs are `2013 passed, 3444 skipped`
  - `python/tutorials/06-fused-attention.py`: all `16` split groups green; selected cases are skipped on this node
  - `python/test/unit/instrumentation/test_gpuhello.py`: `1 passed`
- Current in-flight target:
  - `python/test/gluon/` + `python/tutorials/gluon/` split `32` ways across the `4` GPUs.

## 2026-04-01: GB200 `python/test/gluon` blocker fix

- Restored the historical TMEM source shorthand: `tensor_memory_descriptor.slice(start, length)` now defaults to slicing the trailing dimension again. Explicit `dim=` still uses generic `ttg.memdesc_subslice` semantics.
- Exact TMEM memdesc IR types now survive Gluon view reconstruction and warp-specialize block-arg deserialization:
  - semantic view builders (`slice/index/trans/reshape/reinterpret`) cache `handle.get_type()` on TMEM descriptor types
  - `tensor_memory_descriptor_type._unflatten_ir` now restores the exact handle type instead of rebuilding a nominally equivalent TMEM type
- This fixes the `tt.call` operand mismatch that appeared in the GB200 `test_consan` aliasing kernel after the public slicing path moved to generic `ttg.memdesc_subslice`.
- `BufferRegionAnalysis` now handles generic TMEM `ttg.memdesc_subslice` correctly:
  - use `ttng::getTMemViewOffset(...)` for tensor-memory subslices
  - if a generic shared-memory subslice cannot be inverted exactly, conservatively keep the parent region instead of asserting
- Validation checkpoint:
  - the direct `DISABLE_SUBPROCESS=1` repro is back to the original expected device-side assert (`Buffer being accessed has outstanding reads`)
  - the wrapped `test_aliasing_tensor_visibility_outstanding_read[1ctas-True]` passes again
  - the stale `test_tensor_memory` frontend expectation was updated to the restored trailing-dimension TMEM shorthand


## 2026-04-01: GB200 `test_tmem_subslice_block_m_64` fix

- The remaining M64 TMEM column-subview bug was not the half-tile zero store anymore; after the root `64xN` TMEM layout selection change, the `64x128 -> 64x64 -> reinterpret<f16, 64x128>` zeroing path was correct and only the three `64x2xf32` follow-up stores were still wrong.
- The bad behavior came from the old column-subview support-rescue path in `TensorMemoryUtils.cpp`:
  - it approximated `64`-row TMEM column subviews with a reshaped support tile that treated them like contiguous leading columns
  - that shifted `64x2` packets onto the wrong TMEM half (the LLIR base addresses moved to `+64/+68/+72` instead of the expected `+32/+36/+40` family)
- The fix is to stop using that support-rescue path for `64`-row TMEM column subviews and let them use the raw TMEM view query instead.
- `gluon_ir.cc` also now avoids the old legacy-only auto-layout shortcut for root `64xN` TMEM descriptors before running the direct TMEM-compatible layout search, so `s_tmem.get_reg_layout()` can be reused by those M64 subviews again.
- Validation checkpoint:
  - `test_tmem_subslice_block_m_64[legacy]` passes
  - `test_tmem_subslice_block_m_64[linear]` passes
  - paired rerun passes


## 2026-04-01: descriptor-chain physical-origin remap and GB200 stale-test cleanup

- The remaining GB200 descriptor-chain wrong-code was in translated standalone ld/st support queries, not in the TMEM packet selection itself.
- Root cause:
  - `inferStandaloneTMemLdStQueryLayoutImpl(... preserveViewOrigin ...)` reused the translated support layout but carried origins over by logical dim name.
  - For reshaped / permuted / reinterpreted TMEM chains that preserve the same physical storage while changing the logical basis, that origin copy can point the direct ld/st plan at the wrong physical tile even when the support layout itself is correct.
- Fix:
  - add `remapTMemLdStQueryOriginThroughPhysicalCoords(...)` in `TensorMemoryUtils.cpp`
  - compute the source physical TMEM coordinates from the original query origin, then apply the destination support layout's left inverse to recover the correct origin in the translated support basis
  - use that remapped origin whenever a translated support analysis succeeds
- This restores the intended aliasing behavior for the double-buffer descriptor-chain matrix kernel and keeps the direct ld/st path exact instead of falling back to implicit repair.
- Separate cleanup in `MemDescReinterpretOp::inferReturnType`:
  - non-TMEM reinterprets now validate the visible source/destination view bits (`shape`) instead of the backing `allocShape`
  - this preserves valid reinterprets of contiguous slices with larger retained backing allocations
- Additional GB200 cleanup:
  - explicit `dim=0` in multi-buffer TMEM slice chains where the test intent was always the outer buffer dimension
  - refreshed exact M64 fallback packet expectations to the current `16x32bx2` direct lowering
  - widened stale negative-diagnostic assertions in `test_fpsan.py` to the current clean verifier/lowering messages


## 2026-04-01: explicit split-N TMEM store layout must survive descriptor-view canonicalization

- After the `rowPlan=64` subview query fix, the remaining `test_tmem_index_subslice` wrong-code was no longer a TMEM base-offset/row-anchor issue.
- Root cause:
  - the `64x32` TMEM descriptor view itself lowered with the correct direct query plan, but the generic `tmem_store(convert_layout) -> tmem_store` canonicalization in `lib/Dialect/TritonGPU/IR/Ops.cpp` erased the user-requested split-N register layout before lowering.
  - that left the store on the blocked source layout while the load still used the split-N linear layout, producing a packet-family mismatch on the same TMEM subview (`32x32b` store vs `16x32bx2` load).
- Fix:
  - keep that canonicalization for plain/root TMEM stores, but disable it for view-like TMEM destinations (`memdesc_subslice`, `memdesc_index`, `memdesc_reshape`, `memdesc_reinterpret`, `memdesc_trans`, and internal `ttng.tmem_subslice`).
  - this preserves explicit user-visible TMEM register layouts on descriptor views, which is required by the project rule that direct user layouts must either lower faithfully or be rejected, not silently rewritten.
- Validation checkpoint:
  - `python/test/gluon/test_fpsan.py::test_tmem_index_subslice` passes again
  - full lit is back to `248 passed, 2 unsupported`
- Related stale test update:
  - `test/Conversion/relayout_tritongpu.mlir` now expects the current preferred 64x64 TMEM store relayout after comparing its output against the `origin/main` checkout in `/tmp/triton-origin-main`; the file still checks that relayout produces a direct TMEM-compatible store layout, only the register/lane split changed.

## 2026-04-01: GB200 MMAv5 shared-input `64x32xf32` accumulator readback regression

- The remaining GB200 MMAv5 failure was not in shared A/B descriptor synthesis or in the MMAv5 opcode family itself.
- Root cause was a split between the type-only and handle-aware TMEM reg-layout queries for plain legacy `blockM=64, blockN=32, colStride=1` accumulators:
  - type-only `tensor_memory_descriptor_type.get_reg_layout(num_warps=4, instr_variant="auto")` already returned the canonical M64 `64x32` layout
  - handle-aware `tensor_memory_descriptor.get_reg_layout(...)` still revalidated candidates with `getTMemLdStRowPlanForQuery(...)`, which classified the raw zero-row-basis legacy leaf as a widened 128-row family and therefore accepted the raw scalar `32x32b.x1` readback layout first
- Symptom:
  - fresh TTGIR/PTX for the failing `test_mma_shared_inputs` node showed
    - bad current path: `#linear<{register=[[0,1],[0,2],[0,4],[0,8],[0,16]], lane=[[1,0],[2,0],[4,0],[8,0],[16,0]], warp=[[32,0],[64,0]]}>` and 32 scalar `tcgen05.ld.sync.aligned.32x32b.x1.b32` loads
    - passing origin path: canonical split M64 layout and `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`
- Fix:
  - broaden `getTMemLdStRowPlanForType(...)` so logical `M=64` leaves derive their row plan from the active row bases (`activeRowBits == 6`) instead of the raw 7-bit legacy encoding
  - when `Dialect.cpp` validates stripped canonical M64 candidates, use the stripped-query overload of `computeTMemLdStEncodingInfo(...)` so the validator sees the same `64`-row TMEM view as the selector
- Result:
  - handle-aware and type-only TMEM load layout selection agree again on plain `64x32xf32` / `64x64xf32` MMA accumulators
  - the GB200 `test_mma_shared_inputs[...]acc_dtype4` repro passes again

## 2026-04-01: GB200 Blackwell preferred `16x256b` matmul support through physical-support ld/st rewrites

- The remaining GB200 Blackwell matmul regressions with `TRITON_PREFER_TMEM_16x256_LAYOUT=1` were not a pure `getDefaultLayoutForTmemLdSt(...)` problem.
- TTGIR showed that the affected cases were going through `OptimizeTMemLayouts` physical-support rewriting:
  - current failing path: `ttng.tmem_alloc !ttg.memdesc<512x64xf16, #tmem_linear>` then `ttg.memdesc_reinterpret -> !ttg.memdesc<128x256xf16, ...>` and `ttng.tmem_store/load` on the support view
  - origin passing path: direct `ttng.tmem_store/load` on the exact family and PTX `tcgen05.{st,ld}.sync.aligned.16x256b...`
- The support rewrite itself is acceptable; the bug was atom selection inside `getTMemLdStPhysicalSupportPlan(...)`.
- Root cause:
  - support-plan probing still preferred `I32x32b` over `I16x256b`
  - therefore any shape that reached the support rewrite ignored `TRITON_PREFER_TMEM_16x256_LAYOUT`, even when the support view was exactly `128x256`
- Fix:
  - preserve the corrected legacy `M64/I16x256b` selector basis in `Dialect.cpp`
  - reorder support-plan atom probing under `TRITON_PREFER_TMEM_16x256_LAYOUT` to try `I16x256b` before `I32x32b`
- Result:
  - tall/narrow exact Blackwell matmul accumulator/store-load cases that rewrite to a `128x256` physical support tile now emit native `16x256b` PTX again instead of stopping at `32x32b.x128`

## 2026-04-02: GB200 preferred `16x256b` exact-family `64x128xf16` fix and compile-only stale PTX check

- The remaining GB200 unit failure after the earlier physical-support `16x256b` fixes was the exact-family `64x128xf16` matmul node with `TRITON_PREFER_TMEM_16x256_LAYOUT=1`.
- Real before/after comparison against `/tmp/triton-origin-main` showed:
  - origin/main still lowered the accumulator through legacy `#ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 2>` and emitted `tcgen05.{st,ld}.sync.aligned.16x256b...`
  - the current tree instead selected canonical `#ttng.tensor_memory_linear`, produced a different register layout, and missed the `16x256b` PTX family
- Root cause was twofold in `Dialect.cpp`:
  - `getDefaultLayoutForTmemLdSt(...)` still tried the normalized canonical M64 preference before the exact-family legacy-anchored selector, so full exact leaves diverged from origin/main under `TRITON_PREFER_TMEM_16x256_LAYOUT`
  - `getDistributedLayoutForTmemLdStLegacyAnchored(...)` had changed `layout16Rows` from the old zero-row-basis test (`basis(row,16) == 0`) to a pure logical-size test (`rows <= 16`); for legacy `M64` leaves that incorrectly added a dead register basis, and direct `ttng.tmem_load` then failed with `unsupported broadcasted TMEM lowering for this view`
- Fix:
  - prefer the exact-family legacy-anchored selector before the normalized canonical M64 selector for full-shape TMEM leaves
  - restore the legacy `layout16Rows` zero-basis detection while still allowing true `<=16` row tiles
- Result:
  - the `64x128xf16` exact-family preferred path is back to the origin/main register layout and `16x256b` PTX family
  - the earlier repaired `64x128xtf32` preferred path still passes
- Related stale test cleanup:
  - `python/test/unit/language/test_compile_only.py::test_compile_only_dot` was not a real regression; current and origin emit the same ordered tcgen alloc/store/mma/commit/wait/load sequence, but current direct ld/st packet counts can be `x64`
  - the test now checks the PTX sequence incrementally and accepts `x16|x32|x64`

## 2026-04-01: GB200 persistent-attention warp-specialization TMEM OOR regression

- Symptom:
  - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_persistent_forward[False-8-False-2-128-128-8192-8192]` failed only on the current tree with `OutOfResources: shared memory, Required: 262336, Hardware limit: 232448`
  - `/tmp/triton-origin-main` compiled the same node with `230076` shared bytes and passed
- TTGIR / pass-dump comparison:
  - before `triton-nvidia-optimize-tmem-layouts`, current still matched origin on the relevant accumulator loop: `ttng.tmem_load ... -> tensor<128x128xf32, #linear>` and `ttng.tmem_store ... tensor<128x128xf16, #linear>`
  - after `triton-nvidia-optimize-tmem-layouts`, current retuned that loop-carried accumulator load to `tensor<128x128xf32, #linear1>` and propagated `#linear1` through the softmax/update path
  - origin/main never took that retune and stayed on the `#linear` path
- Root cause:
  - `TMemLoadReducePattern` in `OptimizeTMemLayouts.cpp` was designed to pick a reduction-friendly layout for 8-warp TMEM loads consumed by reductions along `N`
  - in persistent attention, that same TMEM load is also written back to TMEM after the reduction/elementwise update
  - the relayout was therefore not profitable: it forced the value onto the `#linear1` family, then required later `#linear1 -> #linear` conversions for the TMEM store path, and those conversions/lowerings inflated shared memory enough to exceed GB200's limit
- Fix:
  - make `TMemLoadReducePattern` bail out when the forward slice from the load hits a later `ttng.tmem_store`
  - keep the exact-family full-tile default TMEM ld/st selector on the legacy-anchored layout for exact MMAv5 leaves (`getDefaultLayoutForTmemLdSt(...)`) so the accumulator path stays aligned with origin/main expectations
  - preserve explicit `convert_layout` on `ttng.tmem_store` when it converts into the preferred TMEM store layout instead of folding it away just because the source layout is also direct-compatible
- Result:
  - persistent attention returns to the origin-like `#linear` accumulator path
  - warmup shared memory is back down to `230592` bytes
  - the GB200 persistent warp-specialization repro passes again


## 2026-04-06: packed sparse `I16x32bx2` TMEM half-row support-query anchor fix

- Symptom:
  - reduced repro `/tmp/repro_zero_only.py` and `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[linear]` still failed after the earlier packed sparse support recovery work
  - the direct path had converged to the correct `I16x32bx2` family, but runtime still wrote the wrong row blocks
- Debugging result:
  - the remaining bug was not packet count or second-half offset anymore; it was the external warp anchor pair used for the recovered packed support query
  - for the packed `64x64` support view, the raw layout still contains a hidden zero `row=16` basis
  - stripped-layout anchor selection gave `(16, 32)` rows and broke the `warp1/warp3` halves
  - raw unstripped anchor selection gave `(0, 16)` rows and broke the back half of the tile
  - the correct external anchors are `(0, 32)`: warp parity already consumes the hidden zero `row=16` half internally, so the second external anchor must jump to the next non-zero row bit rather than the immediately following logical bit
- Fix:
  - in the packed sparse `I16x32bx2` helper in `TensorMemoryUtils.cpp`, keep anchor basis selection on the raw packed support layout
  - if the first recovered anchor basis is all zero and the next recovered basis is non-zero, retarget the second external anchor to `warpRow1 * 2`
- Result:
  - reduced repro mismatch drops from `2048` to `0`
  - `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[linear]` passes again

- 2026-04-06: Bounded the remaining `block_m_64` reinterpret/store wrong-code bucket further. Disabling the `getDistributedLayoutForTmemLdSt(...)` split-N fast path for the 16->32 unpacked recursion did not change either the explicit `p_tmem.get_reg_layout(instr_variant="32x32b")` rejection or the runtime `mismatch 2048` pattern in `/tmp/repro_zero_only_map_p_layout.py`. Fresh debug shows the stronger root cause: the auto-layout path picks a valid public reg layout `#ttg.linear<{register=[[0,1],[0,2],[0,4],[0,8],[0,16],[0,32]], lane=[[1,0],[2,0],[4,0],[8,0],[0,64]], warp=[[16,0],[32,0]], block=[]}>`, but `computeTMemLdStEncodingInfo(...)` still classifies that layout as atom=4 (`I16x32bx2`) for the sparse reinterpret support view instead of `I32x32b`. So the remaining bug is in `TensorMemoryUtils.cpp` atom selection / packed-support lowering, not just in `Dialect.cpp` family construction.

## 2026-04-06: legacy M64 expanded-N TMEM sugar now canonicalizes to the row-zero-lift support form

- Problem:
  - legacy `TensorMemoryLayout((64, 64), col_stride=1)` on shape `64x128` still lowered through the old interleaved M64 canonical form (`row=16 -> (0, 64)`)
  - direct ld/st reinterpret support rescue for the same user-visible view already used the row-zero-lift form (`row=16 -> (0, 0)`, extra `col=64` basis)
  - that mismatch was enough to make the minimal legacy `p_tmem` reinterpret/store kernel fault even though the later store PTX matched the explicit-linear case
- Diagnostic proof:
  - with temporary allocator tracing, the legacy alloc reported `nRow=128, nCol=64`, `tmem_size=64`, and `tcgen05.alloc ... 64`
  - the equivalent explicit linear alloc reported `nRow=64, nCol=128`, `tmem_size=128`, and `tcgen05.alloc ... 128`
  - `TRITON_DEBUG_TMEM_QUERY=1` showed the reinterpret support path was already rescuing the legacy subslice to the row-zero-lift `64x64 -> 64x128` support layouts, so allocation and lowering were using different physical decompositions for the same legacy sugar
- Resolution:
  - canonical legacy M64 layouts with `shape[0] == 64` and expanded `N` now keep the hidden `row=16` basis zero and let the extra `N` growth stay in the column dimension
  - this aligns legacy sugar with the direct-support linear analysis/lowering model and fixes the legacy `block_m_64` reinterpret/store bucket without adding hidden rematerialization or layout repair


## 2026-04-06: GB200 warp-specialization TMEM allocation metadata regression

- Symptom:
  - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-False-2-64-128-8192-8192]` failed on the generalized tree with `OutOfResources: tensor memory, Required: 640, Hardware limit: 512`
  - the exact same nodeid passed on `origin/main`
- Diagnostic proof:
  - current and `origin/main` TTGIR matched structurally for the failing attention kernel; the only meaningful difference was canonical `#ttng.tensor_memory_linear` attrs on current versus legacy `#ttng.tensor_memory_encoding` sugar on `origin/main`
  - the allocs themselves were the same logical TMEM tiles (`128x64xf32` accumulator plus `128x128xf32` / `128x128xf16` temporaries)
  - therefore the regression had to be in TMEM size accounting, not in the generated MMAv5 or ld/st IR
- Root cause:
  - `getTmemAllocSizes(...)` had been widened to use `planMMAv5AccumulatorFamily(...)` / `planMMAv5ScaledAccumulatorFamily(...)` for canonical linear encodings and to replace the physical TMEM column count with `instrShapeN / preferredColStride`
  - that widened planner is appropriate when selecting a supported MMAv5 instruction family, but it is not the same thing as the backing TMEM storage footprint
  - for equivalent legacy-sugar and canonical-linear TMEM allocs, this inflated `ttg.tensor_memory_size` even though the physical allocs were unchanged
- Fix:
  - TMEM allocation metadata sizing now comes from the physical linear-layout extents directly (`ll.getInDimSize(row/col)`) for both legacy-sugar and canonical-linear encodings
  - MMAv5 family planners remain in place for verifier/codegen matching, but no longer participate in `getTmemAllocSizes(...)`
- Result:
  - the exact GB200 attention-forward repro drops back to the origin-like TMEM budget and passes again
  - canonical linear TMEM encodings no longer pay a larger allocator budget than the equivalent legacy sugar

## 2026-04-07: row-zero M64 packed `16x32bx2.x32` descriptor-chain fix

- Symptom:
  - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_8w-layout5-64-128-32x32b-8-16x32bx2]` was the last remaining wrong-code bucket in the packed direct ld/st path
  - after the packed raw-query recovery landed, the crash was gone but the output still duplicated/misplaced 32-column bands
- Diagnostic proof:
  - the recovered MLIR-side direct plan reported `atom=I16x32bx2`, `rowPlan=64`, and `reps` with exactly one higher-warp recovered band bit: `warp=4 -> (row 0, col 32)`
  - generated LLIR showed the packed PTX path still using one `16x32bx2.x32` message, but the lowering-side address assembly had been suppressing that recovered higher-warp col contribution and later forcing `secondHalfOffset` from `64` down to `32`
  - removing only the lowering-side col squash reduced the mismatch bucket from `4096/8192` to `2048/8192`, proving the remaining bug was the stale `64 -> 32` special-case rather than the raw packed plan itself
- Root cause:
  - two compensating hacks had accumulated while recovering the packed path:
    - LLVM lowering zeroed the recovered higher-warp col band contribution for the packed row-zero M64 path
    - TMEM ld/st analysis then rewrote `secondHalfOffset=64` to `32` for the same `>4 warp` row-zero M64 packed-support family
  - once the warp-group band contribution is preserved in lowering, those `64 -> 32` rewrites are wrong and skip the logical `cols 32:64` band
- Fix:
  - preserve the recovered higher-warp col contribution in `lowerTMemLdSt(...)`
  - keep `secondHalfOffset=64` for the `>4 warp` row-zero M64 packed-support direct path
  - keep the raw packed query-layout / raw `rowPlan=64` recovery intact so the path still selects `tcgen05.{ld,st}.sync.aligned.16x32bx2.x32.b32`
- Result:
  - the isolated `linear_m64_32x32b_8w` descriptor-chain repro now passes directly
  - the whole `tmem_descriptor_chain_matrix` family is green again (`26 passed`)

## 2026-04-07: raw root row-zero `M=64` MMAv5 accumulators need widened ld/st anchors

- The remaining post-GB200 rerun wrong-code bucket was not a generic TMEM descriptor problem anymore. User-visible `linear_m64_*` descriptor-chain ld/st still passed, but internal MMAv5 accumulator kernels failed at exactly the first half-row boundary (`row 16`).
- Repros:
  - `python/test/unit/cuda/test_tma_store_gemm.py::test_tma_load_store[64-128-32-1-4-False-True-False]`
  - `python/test/unit/language/test_block_pointer.py::test_block_ptr_matmul_no_scf[shape3-8]`
  - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-64-128-warps2-16-False-True-acc_dtype4]`
- Debug traces showed a common lowering shape:
  - raw root `ttng.tmem_alloc`
  - row-zero `#ttng.tensor_memory_linear` / legacy-equivalent `blockM=64` accumulator encodings
  - `computeTMemLdStEncodingInfo(...)` choosing `atom=I16x32bx2`
  - active-row `rowPlan=64` (`warpRow0=16`, `warpRow1=32`)
- That active-row plan is still right for query/reg-layout selection, which is why user-facing descriptor ld/st continued to pass. But it is too small for the physical backing accumulator tile MMAv5 writes into.
- Origin/current LLIR comparison on the TMA repro made the mismatch concrete:
  - origin ld/st around MMA base the TMEM address on the widened `32/64` anchors
  - current failing path used `16/32` anchors instead
- A broad storage-row-plan change fixed the unit repros but broke plain `linear_m64_*` descriptor-chain tests, so the durable fix is narrower: only raw lowering of root row-zero `64xNxf32` `ttng.tmem_alloc` accumulators is widened back to the `128`-row anchor family. Query planners stay unchanged.


## 2026-04-07: final stable form of the Blackwell root `64xNxf32` MMAv5 accumulator fix

- The earlier widened-root experiment identified the right class of failure but used the wrong anchor family. The stable fix for the preferred sparse/root `M=64` accumulator layout is **not** `32/64 @ 128`; it is the lifted anchor family `warpRow0=16`, `warpRow1=32`, `rowSpan=128`.
- Why that works:
  - the root accumulator still has logical `M=64`, so query/reg-layout selection for user-visible descriptor ld/st should continue to look like the active-row `64`-row family
  - but the physical backing accumulator tile used by MMAv5 spans the lifted `128`-row support form, and the sparse/root row basis means raw lowering must be allowed to keep that lifted support tile even though the nominal input row span is `64`
- The durable implementation split is therefore:
  - `TensorMemoryToLLVM.cpp`: only widen raw root `ttng.tmem_alloc` accumulators that are actually consumed by `tcgen05_mma` / `tcgen05.mma_scaled`
  - `TensorMemoryUtils.cpp`: explicitly accept the lifted `16/32 @ 128` override for `64xNxf32` zero-row-basis accumulator layouts and keep the old M64 `I32x32b` base-offset halving rewrite gated to the real `I32x32b` path
- Effect:
  - the forced `16x256b` Blackwell root-accumulator path now lowers correct `tcgen05.{st,ld}` around MMA
  - the user-facing packed `linear_m64_*` descriptor-chain families stay green because the override does not leak into ordinary query/reg-layout selection


## 2026-04-07: final root `M=64` MMAv5 accumulator split is by memdesc family, not by MMA use alone

- The durable rule is now:
  - legacy/root MMAv5 accumulator memdescs that still use `TensorMemoryEncodingAttr` sugar keep the lifted raw row anchors `16/32 @ 128`
  - explicit zero-row-basis linear root MMAv5 accumulator memdescs use the lifted raw row anchors `32/64 @ 128`
- Why the earlier single-family fix failed:
  - both families can still lower through the same packed `atom=4` direct ld/st family, so atom choice alone does not identify the right physical TMEM anchor family
  - the important distinction is the root memdesc itself: explicit zero-row-basis linear accumulators still need the backing `32/64` anchor family that the older TMA/block-pointer kernels expected, while legacy-sugar root accumulators want the `16/32` lifted family that fixes `test_simple_matmul`
- The implementation therefore chooses the raw widening override in `TensorMemoryToLLVM.cpp` from the root accumulator memdesc family and lets `TensorMemoryUtils.cpp` accept either lifted `128`-row override for the zero-row-basis `64xNxf32` accumulator case.


## 2026-04-07: TMEM scales alloc lowering must keep broadcasted register structure until `lowerTMemLdSt(...)`

- The GB200-equivalent `mxfp` failure was not another generic TMEM ld/st issue and not a regression from the global `LayoutUtils.cpp` helper anymore.
- The actual failing path was `ttng.tmem_alloc` into `#ttng.tensor_memory_scales_encoding<>` during `tcgen05.mma_scaled` lowering.
- `TMEMAllocOpConversion` passes the full unpacked register value list into `lowerTMemLdStFromTypes(...)`; if `computeTMemLdStEncodingInfoImpl(...)` strips broadcasted register bits from `regLayout` before calling `lowerTMemLdSt(...)`, the returned `info.perm` is defined on the reduced register domain, but the lowering still sees the full source value list and aborts in `ColumnAction::apply(ValueRange)`.
- The right ownership boundary is:
  - `computeTMemLdStEncodingInfoImpl(...)` should keep the raw linear register layout for the query
  - `lowerTMemLdSt(...)` should detect and record broadcasted-register structure in `info.broadcast`
  - `TensorMemoryToLLVM.cpp` should then remove/broadcast values through the existing `info.broadcast` handling in `lowerTMemLdStFromInfo(...)`
- With that split restored, the earlier TMEM ld/st raw-linear fix remains valid and the `mxfp` scaled-matmul kernels lower cleanly again.


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

## 2026-04-09: half-row ld/st still needs a real linear-layout packet decomposition

- Revalidated the remaining lifted row-half ld/st wrong-code bucket after
  repairing the interrupted support-query cleanup state.
- Important refinement:
  - the bug is not only the lifted higher-rank
    `reshape((2, M/2, N)).slice(1, 1, dim=0).index(0)` descriptor chain
  - the direct `128x64 -> slice(M/2, M/2, dim=0)` TMEM row-half view is also
    wrong, while a root `64x64` tile with the same `16x128b` family still
    works
- The current direct ld/st families do not describe the row-half view as a
  simple contiguous lower-half rebasing:
  - `32x32b`, `16x64b`, `16x256b`, and auto-selected `16x32bx2` touch the even
    16-row bands `0-15,32-47,64-79,96-111`
  - `16x128b` touches the odd 16-row bands `16-31,48-63,80-95,112-127`
  - expected semantics remain the contiguous lower half `64-127`
- This preserves the earlier initiative conclusion:
  - the row-half bucket still looks physically codegenable in principle
  - the current lowering is selecting the wrong packet decomposition for a view
    carved from a larger backing tile
  - do not convert this into a new permanent clean-negative boundary; fix it
    through a cleaner linear-layout planner/decomposition model
- Session outcome:
  - retried bounded support-query/base-offset/anchor experiments
  - reverted them because none restored correct semantics
  - kept only the green rebuilt baseline and recorded the evidence
