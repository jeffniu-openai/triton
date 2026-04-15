# TMEM Backend Completion Plan

Created: 2026-04-15 06:15 UTC

## Purpose

The runtime matrix is now broad enough to serve as the correctness anchor for
backend work. The next phase is not more coverage-first expansion. The next
phase is to make TMEM lowering complete over the Blackwell TMEM ISA surface
that Triton can legally expose.

The central conclusion to preserve is:

- The remaining hard failures are evidence that the backend is not yet
  linear-layout complete.
- Scales issues and no-scales two-CTA `warpx2::02_13` are not random
  layout-specific bugs; they expose places where generic-looking APIs still
  lower through family-specific recognizers, type-only fallbacks, or bounded
  descriptor searches.
- `tcgen05.cp ... 4x256b` is a separate missing ISA-coverage item, but it will
  be easier and safer to add once the copy backend has a layout-complete
  planner rather than another isolated recognizer.

## Correctness Anchor

- Use the expanded `python/test/gluon/test_tmem_runtime_matrix.py` matrix as the
  main executable oracle.
- Treat current green state, as a diff against `origin/main`, as the baseline.
- Preserve exact PTX/LLIR opcode checks and runtime oracles.
- Keep clean negatives only when the final planner can explain why no supported
  ISA schedule realizes the physical projection.

## Design Invariants

- TMEM physical layout is the source of truth.
- Frontend type spellings must not be used as substitutes for exact physical
  layout, origin, support, broadcast, CTA ownership, and descriptor-view state.
- Scales are TMEM physical layouts plus block-scale semantics. They are not a
  separate lowering universe.
- Copy, load/store, reduction, MMA, and scaled MMA must share the same physical
  query model where possible.
- ISA legality is separate from physical mapping. A layout may be well-defined
  but still not atomizable by a finite instruction schedule.
- Zero bases represent equivalence/broadcast semantics and must be preserved
  until an operation proves the equivalence class is valid for that instruction.
- Error messages should report the failed planner layer: unsupported physical
  query, unsupported ISA atom, unsupported descriptor synthesis, unsupported
  CTA ownership, unsupported element width, or hardware resource boundary.

## Phase 0: Freeze The Map And Carve Abstractions

Artifact targets:
- This plan.
- Updated `memory.md`, `README.md`, `log.md`, and handoff state.
- A behavior-preserving copy-planner cleanup commit.

Tasks:
- Record the backend-completion strategy and the preserved conclusion above.
- Inventory current TMEM planner surfaces and identify the first abstraction
  seams to pull out of operation verifiers.
- Start by centralizing repeated copy descriptor-plan representability logic in
  `TensorMemoryUtils`.

Exit criteria:
- Current tests still pass for affected compiler-only copy diagnostics.
- The copy verifier has less duplicated inline descriptor-plan logic.

## Phase 1: Define A Complete Physical Query Model

Artifact targets:
- C++ structs and helpers for a shared physical TMEM query description.
- Compiler-only tests for query extraction from linear, scales, and descriptor
  view chains.

Progress:
- 2026-04-15 06:34 UTC: added the first `TMemPhysicalQuery` carrier plus
  `inferStandaloneTMemPhysicalQuery(...)`, and routed `ttng.tmem_copy`
  destination analysis through it. This preserves existing standalone
  view-type behavior while creating the API seam for exact physical-query
  expansion.
- 2026-04-15 06:37 UTC: enriched `TMemPhysicalQuery` with active shape,
  allocation shape, and element bitwidth so future planner layers can consume
  resource and atom-legality facts directly from the query object.
- 2026-04-15 06:39 UTC: added `inferExactTMemPhysicalQuery(...)`, which builds
  `TMemPhysicalQuery` from the existing descriptor-view query algebra and can
  carry exact origin/layout facts separately from standalone type fallback.
- 2026-04-15 06:42 UTC: routed direct copy destination layout support through a
  `TMemPhysicalQuery` overload while delegating to the existing memdesc-type
  implementation to preserve behavior.
- 2026-04-15 06:44 UTC: added side-effect-free physical-query comparison
  helpers covering shape, allocation shape, element bitwidth, physical layout,
  CTA ownership, origin, and scales classification.
- 2026-04-15 06:48 UTC: wired the comparison helpers into
  `ttng.tmem_copy` verification as `TRITON_DEBUG_TMEM_QUERY`-gated
  exact-vs-standalone destination-query reporting. Normal verifier behavior
  and diagnostics remain unchanged, but copy debugging can now show the first
  differing physical-query field plus the standalone and exact layouts/origins.
- 2026-04-15 06:51 UTC: introduced `TMemCopySupportResult` and
  `TMemCopySupportFailureLayer`, added structured direct-destination and
  shared-layout support APIs, kept the legacy bool wrappers as compatibility
  shims, and routed `ttng.tmem_copy` verification through the structured
  results. Direct destination support now consumes the `TMemPhysicalQuery`
  layout instead of re-deriving support from the memdesc type at the copy
  verifier call site.
- 2026-04-15 06:53 UTC: added `getTMemCopyPlanSupport(...)`, the first
  per-plan structured support helper for copy. It combines destination
  physical-query support, shared-layout/runtime support, and
  shared-descriptor synthesis into one result while preserving existing
  diagnostics.
- 2026-04-15 06:55 UTC: aligned `tcgen05.copy` lowering with verification by
  using `inferStandaloneTMemPhysicalQuery(...)` and `getTMemCopyPlanSupport(...)`
  in `TensorMemoryToLLVM.cpp`. The lowering path now consumes the same
  physical-query carrier and per-plan support helper as verification for
  no-scales copy plans.
- 2026-04-15 06:57 UTC: added the first guarded exact-query migration for copy.
  Verification and lowering now compute the exact destination query and prefer
  it only when it compares equal to the standalone projection; divergent or
  failed exact queries preserve the current standalone behavior.

Tasks:
- Introduce a first-class physical view/query object carrying:
  - physical `LinearLayout`;
  - logical shape and element bitwidth;
  - origin/base offsets;
  - CTA group and CGA ownership;
  - support/broadcast bases;
  - allocation shape versus active view shape;
  - scale semantic metadata when applicable.
- Route standalone query inference for linear and scales layouts through this
  model.
- Replace type-only descriptor-view fallbacks with explicit query results or
  precise clean negatives.

Exit criteria:
- Existing `ld/st`, copy, and scales descriptor-view positives stay green.
- Existing clean negatives retain or improve their diagnostics.
- There is one obvious API to ask, "what physical TMEM image does this memdesc
  denote?"

## Phase 2: Rebuild `tcgen05.cp` Around Atomized Layout Planning

Artifact targets:
- A copy planner that takes source shared layout plus destination TMEM physical
  query and returns an ordered message schedule or structured failure.
- Runtime-matrix expansions for newly positive copy schedules.

Tasks:
- Replace current finite family recognition with layered planning:
  1. compute source-to-destination projection;
  2. enumerate candidate copy atoms;
  3. derive descriptor rows/cols, source warp groups, source offsets, TMEM
     dword deltas, and message tiling;
  4. prove MMAShared descriptor representability;
  5. prove CTA ownership and barrier/commit legality.
- Preserve existing dense, `warpx2`, and `warpx4` positives.
- Solve or conclusively reject no-scales two-CTA `warpx2::02_13` by deriving a
  real `cta_group::2` descriptor/address schedule that preserves the high
  source-column bit.
- Add `tcgen05.cp ... 4x256b` only after confirming the exact atom semantics and
  fitting it into the atomized schedule model.
- Move scales copy planning through the same projection/schedule path.

Progress:
- 2026-04-15 07:08 UTC: added `tcgen05.cp.4x256b` as a normal copy family
  with descriptor synthesis and conversion coverage. Superseded by the
  2026-04-15 10:29 safety checkpoint below after runtime probing showed the
  emitted schedule was semantically wrong.
- 2026-04-15 07:12 UTC: added a copy-specific physical-query projection
  comparator so exact descriptor-view queries can be used when active copy
  layout facts match even if origin/allocation metadata differs.
- 2026-04-15 07:15 UTC: introduced `TMemCopyPlanSelection` and routed
  no-scales verification/lowering through one plan selector.
- 2026-04-15 07:25 UTC: added explicit copy plan support modes so scales and
  no-scales use one selector while preserving their distinct legality
  contracts.
- 2026-04-15 07:28 UTC: factored descriptor-layout/MN-orientation selection
  into `selectTMemCopyDescriptorLayout(...)` and consumed it from both support
  checking and lowering.
- 2026-04-15 07:36 UTC: plan selection now preserves all failed support
  results. Negative probes showed that simply broadening descriptor
  permutations does not fix scales warpx2-like layouts, and copying the
  single-CTA `02_13` direct-seed path to cta-group::2 emits the opcode but
  produces zeroed output.
- 2026-04-15 07:39 UTC: added conversion coverage for
  `tcgen05.cp.cta_group::2.4x256b`, confirming the 4x256b copy-family support
  reaches both cta-group ISA variants. Superseded by the 2026-04-15 10:29
  safety checkpoint below; opcode emission was not a correctness proof.
- 2026-04-15 09:02 UTC: `selectTMemCopyPlan(...)` now returns a realized
  executable schedule, including per-message descriptor-layout selection or
  direct-seed descriptor immediates. Lowering consumes that schedule directly,
  reducing support/lowering drift before adding row-offset/source-format
  schedule dimensions.
- 2026-04-15 09:05 UTC: scales exact descriptor-view failures now carry an
  explicit schedule-level diagnostic explaining that the physical row
  permutation cannot be realized by source descriptor selection alone and needs
  a destination-row / source-message schedule.
- 2026-04-15 09:08 UTC: copy executable schedules now have lowering-level
  fields for TMEM destination row deltas and the two currently probed
  source-format suffixes (`.b8x16.b6x16_p32`, `.b8x16.b4x16_p64`). Current
  planners leave both at defaults; this is scaffolding for the exact
  row/message schedule derivation, not a support promotion.
- 2026-04-15 09:31 UTC: direct dense no-scales copy support now reasons about
  the destination `LinearLayout` at instruction-tile granularity instead of
  requiring globally canonical column basis order. The planner admits
  low-descriptor-macro column tile permutations (`tile_n=8`, `tile_n=16`) when
  every logical instruction-width tile maps to aligned contiguous physical
  columns. Superseded by the 2026-04-15 10:38 scheduler slice for higher
  macro-selector reorders.
- 2026-04-15 09:59 UTC: copy message plans can now carry an optional
  descriptor projection separate from the full copy conversion. Existing
  schedules leave it unset, but this creates the planner slot needed for
  atomized schedules where the MMAv5 shared descriptor is a legal projection
  and destination/source offsets complete the full TMEM view.
- 2026-04-15 10:13 UTC: descriptor-candidate generation now consistently
  consumes the per-message descriptor projection for folded descriptor extents
  and direct shared-seed candidates. This keeps the new projection slot
  internally coherent before any planner starts populating it.
- 2026-04-15 10:15 UTC: descriptor-synthesis failures now carry a generic
  copy-instruction column projection note. The scales descriptor-view row
  currently fails because source column bit 2 must map to shared offset 256
  inside a 16-column copy atom; current scheduling cannot split that
  sub-instruction source-column bit, so descriptor representability alone is
  not a support proof.
- 2026-04-15 10:29 UTC: `tcgen05.cp.4x256b` is recognized but disabled as a
  clean unsupported family until the atomized planner derives a validated
  descriptor/address schedule. The previous four-row descriptor candidate
  emitted `tcgen05.cp.cta_group::1.4x256b` but placed source row values into
  one destination row at runtime, so conversion-only coverage was removed and
  replaced with verifier plus runtime clean-negative coverage.
- 2026-04-15 11:10 UTC: follow-up probes refined the `4x256b` gap from
  "wrong four-row copy" to the exact missing schedule: the instruction behaves
  as a TMEM refresh primitive. `smemColOffset` selects source columns, each
  message writes four source rows into destination dwords across lanes
  separated by 32, and destination dword deltas must be 128-bit aligned. The
  current clean negative now records those semantics so future work adds an
  explicit refresh schedule instead of retrying descriptor-orientation tweaks.
- 2026-04-15 10:38 UTC: dense no-scales copy scheduling now has an exact
  destination-tile offset helper. The direct-copy support proof no longer
  rejects whole 128-byte column macro-tile permutations by global basis order;
  instead it proves each logical instruction-width tile maps to one aligned,
  contiguous, unique physical TMEM tile, and lowering uses a per-tile
  destination offset only for macro-crossing permutations. This promotes
  `tile_n=32` dense linear copy to positive runtime coverage while preserving
  the existing `tile_n=8/16` positives and clean negatives for mixed or
  bit-scrambled layouts.
- 2026-04-15 10:41 UTC: the dense macro-tile scheduler now also detects
  non-ascending high macro-selector bases, promoting the `N=256/tile_n=64`
  dense copy row after a direct probe showed the first macro-tile schedule only
  handled high-low crossings. This keeps the support predicate general over
  instruction-tile physical destinations rather than tied to one tile width.
- 2026-04-15 10:57 UTC: fixed a descriptor-view base-offset lowering hole
  exposed by no-scales `warpx2` indexed and slice-index views. Zero trailing
  logical coordinates no longer force `getTMemViewOffsetImpl` to pseudoinvert
  a non-surjective 2-D view layout; the prefix offset is still applied
  separately. Copy verifier/lowering now also require selected exact physical
  queries to be safely composable with the shared-memory layout before
  planning. This is another example of replacing an accidental layout-family
  assumption with explicit linear-layout algebra and clean diagnostics.
- 2026-04-15 11:21 UTC: dense no-scales copy support now reaches canonical
  subword element types (`f16`, `bf16`, `i16`, `i8`) by removing the
  verifier-level 32-bit source guard and making the runtime kernels advertise
  the actual source element bitwidth in `NVMMASharedLayout`. This was a
  backend generality bug rather than an ISA limitation for dense
  `128x256b`. `warpx2` subword copies remain clean unsupported with a
  descriptor-plan diagnostic because the current `warpx2` schedule still
  requires 32-bit shared elements. Copy plan construction can now build a plan
  from an explicitly supplied dense atom, while dense-only fallbacks stay
  isolated from multicast/scales families.
- 2026-04-15 11:30 UTC: two-CTA no-scales `warpx2::02_13` remains a
  schedule gap, and the verifier diagnostic now records why the direct-seed
  path is not a viable support route. cta-group::2 direct-seed probes emit the
  opcode but duplicate the low source-column pair; the aligned dword deltas
  that complete the single-CTA direct-seed schedule read zeros under
  cta-group::2. The next support attempt must derive a real cta-group::2
  descriptor/address schedule preserving the high source-column bit rather
  than reusing the single-CTA schedule.
- 2026-04-15 15:47 UTC: a bounded follow-up direct-seed sweep reinforced the
  11:30 boundary. Representative B128 source bases and aligned destination
  dword deltas produced either shifted duplicate low-column pairs (`delta=0`)
  or zeros (`delta>0`). All probe code was removed. Keep this frontier parked
  on deriving a real descriptor/address schedule, not seed-parameter tuning.
- 2026-04-15 11:39 UTC: scales descriptor-view copy now has a sharper
  row-order proof. A temporary descriptor-basis exchange can make the
  `warpx4.32x128b` shared descriptor representable, but runtime output follows
  parent/root row order (`row r -> source row (r % 64) * 2 + r / 64`) instead
  of the logical descriptor-view order. This confirms the remaining gap is a
  row-interleaving destination/source schedule layered after descriptor
  projection, not simply expanding descriptor candidates.
- 2026-04-15 11:46 UTC: a sparse four-row view over a `128x8` parent was
  checked as a possible `4x256b` refresh-shaped layout. The exact copy
  conversion drops the row dimension and exposes only shared-offset column
  bases `4` and `8`, so it does not classify as `4x256b`; the canonical
  four-row parent slice still classifies and is blocked by the refresh
  diagnostic. This keeps the `4x256b` task scoped to explicit refresh schedule
  modeling, not descriptor-view reshaping.
- 2026-04-15 11:52 UTC: legacy packed subword dense copies now report the
  actual missing abstraction. The planner sees zero low column bases for
  16-bit and 8-bit legacy `TensorMemoryLayout` copies, representing packed
  lanes outside the `LinearLayout` offset dimension. Canonical unpacked
  `TensorMemoryLinearLayout` subword copies remain positive. The remaining
  legacy-packed support task is to carry packed-lane semantics in the physical
  query / copy schedule, or prove that `tcgen05.copy` cannot realize that
  packed destination image.
- 2026-04-15 11:56 UTC: dense no-scales `M=256,N=128` now has a row-group
  diagnostic that names the missing atomized schedule. A temporary classifier
  guard lift proved that treating a 256-row projection as a single dense atom
  only reaches a descriptor-realization abort; the planner needs to split the
  projection into multiple 128-row messages and carry the high row-group
  selector through descriptor projection plus source/destination row offsets.
- 2026-04-15 12:10 UTC: the reachable subset of dense `M=256` linear copies is
  now positive. For `N in {32,64,128}`, the exact physical query exposes a
  legal `128x2N` MMAv5 family image, so copy planning can use the exact query
  and the allocator can reserve the proven 128-row physical image. This
  promotes nine runtime rows and leaves `256x16` as the remaining clean atom
  classification negative.
- 2026-04-15 15:31 UTC: supersedes the final sentence of the 12:10 checkpoint
  for coverage purposes. A fresh probe showed `256x16` already lowers through
  the same exact-query discipline: row bit 128 is folded into the column stream,
  the conversion selects `tcgen05.cp.128x256b`, and swizzle 32/64 runtime
  oracles pass with four messages. The stale unsupported-shape test was removed
  and the two valid swizzles were promoted to positive coverage; swizzle 128 is
  rejected by shared-memory layout preconditions for a 16-column f32 tile.
- 2026-04-15 15:53 UTC: dense no-scales row/column permutations were re-probed
  under a temporary guard lift and remain real schedule gaps. Row-basis
  permutations violate the current lowering row-stride assumptions when forced
  through; column-basis permutations either misalign or produce wrong data with
  the existing per-tile address schedule. Keep these clean negatives parked on
  a future row/packet scheduler rather than treating them as stale guards.
- 2026-04-15 12:25 UTC: scales descriptor-view row interleaving was re-probed
  with temporary source-only experiments. Existing message fields do not
  provide the needed even/odd row partition: `smemRow=64` is not an independent
  row pass, `tmemRowDelta=1` faults, `tmemRowDelta=64` preserves the same wrong
  output, and the only two representable descriptor selections are both wrong.
  Keep the row as a clean schedule gap until the planner has a new row
  partition abstraction or ISA evidence for one.
- 2026-04-15 12:35 UTC: the physical `4x256b` refresh schedule is now known:
  two messages with source column offsets 0 and 4 and destination dword offsets
  0 and 4 cover an 8-column tile. The support blocker moved to representing a
  refresh-shaped destination view and its load/store contract; the current
  public API/view inference rejects the needed 4x8 slice over a 128x8 physical
  parent as an unsupported `memdesc_subslice` view.
- 2026-04-15 12:52 UTC: `4x256b` copy support is now positive for the exact
  refresh-shaped active `TensorMemoryLinearLayout`. The planner recognizes the
  proved physical image, uses a per-message descriptor projection, and emits
  two `tcgen05.cp.cta_group::1.4x256b` messages. Verifier and lowering share
  physical-query selection, so noncanonical exact linear roots no longer need
  a standalone canonical fallback to reach copy planning. Ordinary contiguous
  four-row copies stay negative because they are not the refresh image the ISA
  realizes. Remaining `4x256b` work is the load/store/register-layout contract
  for the refresh layout, not the copy opcode schedule.
- 2026-04-15 13:03 UTC: the same refresh-shaped `4x256b` copy support now
  covers two-CTA `cta_group::2`. The accepted layout is the single-CTA refresh
  image lifted with `block_bases=[[4,0]]`, `shape=[8,8]`, and
  `two_ctas=True`; the recognizer ignores implicit size-1 block dimensions so
  single-CTA exact queries still match. Runtime and conversion coverage now
  assert two `tcgen05.cp.cta_group::2.4x256b` messages, while ordinary
  contiguous two-CTA 4x256 remains a clean negative. Remaining `4x256b` work is
  still the direct load/store/register-layout contract for refresh-shaped
  active layouts.
- 2026-04-15 13:07 UTC: the direct load/store/register-layout contract for
  refresh-shaped 4x256 active layouts is now a clean negative with a specific
  reason. The planner cannot make the required physical row anchors into
  `tcgen05.ld/st` warp bases because the refresh view stores logical row bits
  in TMEM columns and low logical column bits in rows 32/64. Copy remains
  positive for the proved refresh images; direct `get_reg_layout`, `load`,
  reduction load, and `store` now fail early with that explanation.

Exit criteria:
- Copy support decisions can be explained by a planner trace instead of by a
  hard-coded recognizer branch.
- True scales `warpx2` and no-scales two-CTA `warpx2::02_13` are either
  positive or rejected with proof-level diagnostics.
- `4x256b`, if ISA-valid for the targeted backend, is represented as a normal
  copy atom with a row-coded runtime proof rather than a conversion-only opcode
  emission case.

## Phase 3: Make Scales A Full Physical Layout Citizen

Artifact targets:
- Scales layout representation or lowering metadata that preserves exact
  physical bases.
- Positive or clean-negative runtime rows for scales descriptor views under
  non-default CGA and multibuffering.

Tasks:
- Stop treating `TensorMemoryScalesLayout` as only a marker plus CGA layout at
  the points where exact physical mapping is required.
- Represent scale packed minima, `scale_vec`, format constraints, and padded
  physical images as operation semantics layered on top of the shared physical
  query.
- Remove scales-only fallbacks once the shared query model can prove the same
  facts.

Progress:
- 2026-04-15 08:35 UTC: added scales-root recovery through descriptor-view
  chains and routed copy verification/lowering through
  `TMemPhysicalQuery::isScales`. Scales descriptor-view copies now report a
  clean tensor-memory-scales copy-family miss instead of falling into the
  non-scales 32-bit-source diagnostic. Exact query origins are now available to
  copy lowering through `getTMemPhysicalQueryOriginBaseOffset(...)`; the next
  planner gap is atomizing the exact permuted scales descriptor-view projection.
- 2026-04-15 08:42 UTC: scales descriptor-view copies now use the exact
  physical query for planning when shape/bitwidth/CTA/scales facts match the
  standalone query. The first scale-backed permuted copy row now reaches
  `warpx4.32x128b` classification and fails at descriptor synthesis, proving
  the remaining gap is descriptor-layout derivation rather than family
  recognition or scales classification.

Exit criteria:
- Scales descriptor-view decisions no longer depend on type-only
  canonicalization.
- Scales copy and scaled MMA use the same physical query concepts as no-scales
  TMEM.

## Phase 4: Unify Load, Store, And Reduction Planning

Artifact targets:
- Shared query/support layout machinery for `ld`, `st`, and `ld.red`.
- Removal or quarantine of row/column rewrite rescue paths.

Tasks:
- Re-express support layout, row anchors, packet offsets, split-N, and
  broadcast handling in terms of the Phase 1 physical query model.
- Preserve the existing broad `ld/st` and `ld.red` matrix while replacing
  special-case layout fixes with algebraic derivation.
- Keep non-f32 `ld.red` and subword boundary diagnostics precise.

Progress:
- 2026-04-15 13:30 UTC: promoted ordinary identity `M=256` `ld.red`
  sources for `N in {32,64,128}`. The reduction-friendly source predicate no
  longer requires the high row selector to be encoded as a column carry basis;
  direct raw-query lowering can use the pure 256-row identity layout and still
  satisfy the reduction register-layout contract with 8 warps. The old
  identity `256x256` clean negative is now a shared-memory resource boundary.
- 2026-04-15 13:43 UTC: promoted the row-basis-permuted subset of
  expanded-row `M=256` `ld.red` sources. Allocation sizing now derives the
  compact 128-row physical TMEM image from separable power-of-two linear
  layout facts instead of depending on the canonical MMAv5 row-basis order.
  Runtime coverage pins representative `reverse`, `rotate1`, and `even_odd`
  row permutations. Expanded-row column permutations remain clean unsupported
  because probes showed current direct `ld/st` packet scheduling can miscopy
  them after allocation succeeds.
- 2026-04-15 13:59 UTC: moved the expanded-row column-permutation boundary
  into a shared direct `ld/st` classifier and diagnostic. The backend now
  detects pure 256-row linear images whose column bases are not in canonical
  packet order and rejects direct `tcgen05.ld/st`/source initialization before
  allocation or lowering can assert or expose wrong-code behavior. This is a
  superseded clean negative: the 14:26 folded-query support checkpoint below
  replaces it with positive direct `ld/st` coverage.
- 2026-04-15 14:26 UTC: promoted the expanded-row column-permuted direct
  `ld/st` subset using exact folded query arithmetic. The new helper derives
  a physical 128-row query from the original separable linear layout: low row
  bases remain in the TMEM row coordinate, the original column bases remain in
  their exact user order, and the high row selector is carried as the next
  TMEM column bit. This removes the temporary frontend/op-verifier guards and
  avoids both wrong raw row-128 schedules and canonical fallback queries that
  would erase column order. `ld.red` still has an independent source-layout
  boundary for expanded-row column permutations.
- 2026-04-15 14:32 UTC: promoted representative expanded-row
  column-permuted `ld.red` sources using the folded direct-query semantics.
  The source predicate now proves the 256-row separable column bases as a set
  instead of requiring canonical order, and direct encoding validation remains
  responsible for rejecting layouts the folded query cannot realize. Positive
  runtime rows now cover `identity/reverse,N=64` and
  `even_odd/even_odd,N=128` across min/max, abs, and NaN propagation.
- 2026-04-15 14:58 UTC: promoted the first two-CTA scales descriptor-view
  direct `ld/st` row. The backend now recognizes the exact `128x64`
  reshape/permute scales view raw query and returns a proved 4-warp
  `I32x32b` register layout that validates against the descriptor-view query
  with row anchors `16,32`. This keeps the broader two-CTA int8 guard and
  type-only fallback ban intact while letting the exact supported family
  through. The `256x32` and `256x64` scales CGA views remain clean unsupported
  pending separate exact-query/layout proofs.
- 2026-04-15 15:04 UTC: completed the current two-CTA scales descriptor-view
  direct `ld/st` CGA bucket. The exact recognizer now accepts the raw `256`
  view form where zero row-tail bases are stripped, row anchors are `0,0`,
  row-carry bits `16,32` are in the column stream, and block ownership is
  `[64,0]`. The planner constructs the register layout from those raw facts
  and validates it, promoting `256x32` and `256x64` beside the earlier
  `128x64` row.
- 2026-04-15 15:24 UTC: M64 split-N `ld.red` was probed and kept clean
  unsupported. The ordinary split-N `ld/st` schedule needs a lane-carried N
  selector and multiple messages; a direct `ld.red.16x32bx2` row-plan lift
  aliases rows and leaves half the tile uncovered. Treat future support here
  as a cross-lane/thread partial-reduction design, not a row-plan guard lift.

Exit criteria:
- `ld/st` and `ld.red` lower through shared physical-query facts, not separate
  handwritten interpretations of descriptor chains.
- Existing expanded matrix remains green.

## Phase 5: Unify MMA And Scaled MMA Descriptor Semantics

Artifact targets:
- Shared accumulator and scale descriptor query paths.
- Runtime-matrix rows for newly reachable scaled-MMAv5 descriptor views.

Tasks:
- Separate accumulator physical layout legality from MMA instruction shape
  legality.
- Route accumulator descriptor views through the same physical query model.
- Route scale descriptors through physical query plus explicit scale semantics.
- Preserve packed 128-row scale descriptor minima and padded scale-copy behavior.

Progress:
- 2026-04-15 17:28 UTC: promoted scaled-MMAv5 two-CTA accumulator subslice
  `block_n=32` rows. Scaled root/view/two-CTA N32 coverage is now positive
  where each operation maps to supported scale fragments.
- 2026-04-15 17:26 UTC: promoted supported scaled-MMAv5 `N=32`
  root/index/subslice/LHS matrices. This narrows the remaining scale issue:
  single-tile/root/view `N=32` is valid, while repeated/tile-permuted `N=32`
  remains a scale-B fragment/addressing gap.
- 2026-04-15 17:12 UTC: saturated the remaining supported plain-MMAv5 `N=32`
  M64 and TMEM-LHS matrices. M64 linear/root and subslice rows plus LHS
  tile/subslice rows are positive at `N=32`; the M64 legacy root remains on
  its 64-column accumulator-tile contract.
- 2026-04-15 16:58 UTC: extended the `N=32` plain-MMAv5 accumulator coverage
  through adjacent single-CTA root/index/subslice matrices and
  TensorDescriptor-fed two-CTA matrices. Direct i8 unsupported diagnostics now
  cover `N=32` as well. This keeps the N32 cleanup as matrix saturation over
  the now-general accumulator family instead of a new backend branch.
- 2026-04-15 16:43 UTC: promoted plain MMAv5 two-CTA accumulator
  `block_n=32` coverage for root layouts, indexed accumulator descriptor
  views, and accumulator subslice views. The backend already supports the
  narrow cta-group::2 physical image; this removes a stale runtime-matrix
  floor while keeping the `N=256` linear-parent hardware resource boundary.
- 2026-04-15 16:29 UTC: promoted plain MMAv5 full-shape TMEM-LHS
  tile-permuted rows for `K=32` and `K=64`. The LHS planner now admits
  8-column storage families, and the positive LHS matrix spans
  `K in {32,64,128,256}` except for the existing tf32 shared-memory resource
  boundary.
- 2026-04-15 16:23 UTC: re-probed scaled narrow tile-permuted accumulator rows
  with a temporary `blockN=8/16` scaled planner lift. The rows compile but are
  numerically wrong across sampled scaled format pairs, confirming they remain
  scale-fragment/addressing gaps and not stale family-floor negatives.
- 2026-04-15 16:18 UTC: promoted plain MMAv5 narrow tile-permuted accumulator
  rows. The plain accumulator planner now admits 8- and 16-column N families,
  moving `(N,tile_n)=(32,8)` and `(64,16)` into the positive tile-permuted
  matrices. Scaled MMAv5 keeps the separate wider-alignment planner because
  its narrow rows remain scale-fragment clean negatives.
- 2026-04-15 16:09 UTC: promoted full-shape tile-permuted scaled-MMAv5 TMEM-LHS
  fp4/nvfp4 `K=128` storage rows. The LHS planner now admits 16-column physical
  storage families, which is correct for packed fp4 operands because LHS TMEM
  planning is in byte storage columns. The stale clean-negative table was
  removed and the positive/use-acc matrices now cover these rows.
- 2026-04-15 15:58 UTC: repeated `N=32` direct block-scaled MMAv5 was re-probed
  with a temporary verifier/lowering guard lift. The accumulator physical
  layout can be forced to compile, but representative scaled format pairs all
  produce wrong output. Keep this as a true matrix-B scale-fragment addressing
  gap until the scale descriptor model can express sub-64-column B fragments.

Exit criteria:
- Plain and scaled MMAv5 descriptor-view support are explained by the same
  physical-query framework plus operation-specific legality checks.

## Phase 6: Fuzz, Saturate, And Prove Clean Negatives

Artifact targets:
- Generated or table-driven fuzz matrix rows.
- Updated manifests distinguishing pass, clean unsupported, and true bug.

Tasks:
- Use `fuzz_plan.md` to generate bounded layout/view-chain cases across copy,
  `ld/st`, `ld.red`, MMA, and scaled MMA.
- Require exact opcode streams and runtime oracles for positives.
- Require targeted diagnostics for negatives.
- Use duration-aware split-4 pytest runs for broad sweeps.

Exit criteria:
- The backend accepts every ISA-realizable generated case in the bounded matrix.
- Remaining unsupported rows are proven ISA/resource/API boundaries.

## Phase 7: Delete Compatibility Scaffolding

Artifact targets:
- Cleanup commits removing obsolete branch-only attributes, rescue paths, and
  fallback canonicalization.
- Final handoff and validation manifest.

Tasks:
- Remove stale physical-layout/root markers and ad-hoc compatibility logic made
  obsolete by the shared planner.
- Keep public migration paths for descriptor view APIs.
- Re-run broad compiler and runtime validation.

Exit criteria:
- TMEM backend decisions flow through the shared physical-query/planner stack.
- Compatibility code that hides layout bugs is gone or explicitly quarantined.
