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
- 2026-04-16 profiling note: do not respond to slow runtime-matrix sweeps by
  trimming rows. The profiled `ld.red` representative is now within target
  (`2.252s` cold compile, sub-millisecond warm execution); if broad sweeps are
  still too slow, fix shard balance, duration-cache use, pytest
  collection/import overhead, or new compiler hot spots.
- 2026-04-16 follow-up: the x1 two-CTA subword descriptor-chain hotspot is
  also within target after verifier duplicate-work cleanup (`3.614s` cold
  compile). Avoid context-unsafe static caches for TMEM encoding info; prefer
  reducing repeated proofs or adding context-owned caches if future hotspots
  require memoization.
- 2026-04-16 test-surface note: the runtime matrix is now representative
  rather than exhaustive over every independent Cartesian axis. Collection was
  reduced from `11133` to `5019` tests by keeping each instruction family,
  hard-frontier diagnostic, data kind, layout class, and edge geometry
  represented while avoiding duplicate modifier/use-acc/view products. When
  changing a specific planner family, temporarily expand or target that family
  as needed, then keep the default matrix lean for iteration.

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
  instead of a contiguous sub-instruction offset that a single copy message
  can carry inside a 16-column copy atom; current scheduling cannot split that
  sub-instruction source-column bit, so descriptor representability alone is
  not a support proof.
- 2026-04-15 18:39 UTC: moved the two-CTA no-scales
  `warpx2::02_13` boundary into the shared copy-plan realization layer. The
  planner now returns the known cta-group::2 descriptor/address schedule gap
  directly, and diagnostic attachment suppresses duplicate failure notes. This
  keeps verifier and lowering aligned and avoids treating bounded descriptor
  enumeration as a possible realization for a schedule already disproven by
  direct-seed probes.
- 2026-04-16 10:50 UTC: re-probed that two-CTA `warpx2::02_13` direct-seed
  boundary with configurable source offsets and destination dword deltas. The
  only non-faulting opcode-emitting variants still either duplicate one source
  column pair into both destination pairs (`dword_delta=0`) or write zeros
  (aligned nonzero deltas), so this remains a true descriptor/address schedule
  gap rather than a parameter choice in the existing direct-seed path.
- 2026-04-16 12:04 UTC: rehydrated the copy frontier after the M64 backend
  cleanup. The current two-CTA `warpx2::02_13` exact conversion still maps
  row bit 32 to source offset `1`, confirming the missing fact is a
  source-column selector, not a row-stride variant. The current scales
  descriptor-view exact conversion still maps source column bit 2 to descriptor
  row `+32` for 4-column destination runs every 8 columns inside one
  `warpx4.32x128b` instruction. Source-format suffixes are packed source data
  formats and do not provide a destination-column mask. Treat both as
  source-message / destination-footprint modeling tasks, not frontend fallback
  cleanup.
- 2026-04-16 21:52 UTC: added the source-row counterpart to the typed schedule
  requirements. `TMemCopySourceRowSplitRequirement` now derives row-selected
  source-offset facts from non-affine source-row projection failures:
  selected row run, row selection period, actual shared offset, expected
  affine offset, source row stride, and instruction footprint. The two-CTA
  no-scales `warpx2::02_13` gap is now reported as this generic
  instruction-schedule requirement plus the existing direct-seed probe
  evidence, so future support work can consume structured schedule data
  instead of a known-gap string.
- 2026-04-16 21:58 UTC: added `TMemCopyDestinationMaskRequirement` as the
  shared scheduler consequence of those split requirements. Descriptor-row
  splits now derive a column-mask requirement, and source-row splits derive a
  row-mask requirement. This keeps the scales `warpx4` descriptor-view gap and
  no-scales `warpx2` row-selected source-offset gap on one planner concept:
  the current public atom writes a full destination footprint while the split
  only owns selected rows or columns.
- 2026-04-16 23:52 UTC: promoted that mask requirement into a shared
  schedule-gap helper. Both descriptor-row splits and source-row selected-offset
  splits now report the no-mask proof through the same axis-aware code path:
  selected runs smaller than the instruction footprint require a destination
  row/column mask, narrower atom/source format, or a proved non-overwriting
  multi-message schedule. This is still a clean unsupported boundary, but it
  removes duplicated family-shaped no-mask reasoning from the copy planner.
- 2026-04-15 18:43 UTC: raised the copy instruction-column projection check
  into a preflight support layer. The planner now rejects sub-instruction
  source-column permutations before descriptor enumeration, so an expanded
  MMAShared descriptor search cannot accidentally turn a copy atom into a
  false support claim when the per-instruction source columns are not
  contiguous.
- 2026-04-15 18:55 UTC: introduced `TMemCopyDestinationTile` and
  `getTMemCopyDestinationTilePlan(...)`. Lowering now consumes an explicit
  destination tile plan from the shared planner utility instead of recomputing
  destination offsets inline. Current behavior is preserved, but this creates
  the carrier needed for future non-uniform destination/source message
  schedules.
- 2026-04-15 18:59 UTC: added
  `getTMemCopySourceRowProjectionSupport(...)` and replaced dense-copy
  lowering assertions with shared source-row projection checks. Unsupported
  source row projections now fail during plan realization with diagnostics
  instead of relying on lowering-time asserts.
- 2026-04-15 19:02 UTC: promoted the source-row projection proof into a
  structured executable-plan carrier. `TMemCopySourceRowProjection` now records
  the 8-row source stride and affine row-bit steps on each scheduled copy
  message during plan realization, and lowering no longer repeats the support
  check after the plan is selected.
- 2026-04-15 19:08 UTC: promoted the instruction-column projection proof into
  a second executable-plan carrier. `TMemCopyInstructionColumnProjection`
  records the instruction width, unit source offset, and supported contiguous
  low column-bit steps on each scheduled message. A temporary bypass probe
  confirmed the scales descriptor-view copy still fails descriptor synthesis
  after that preflight is removed, so support needs real source-column/message
  splitting rather than broader descriptor enumeration.
- 2026-04-16 09:43 UTC: repeated the scales-copy source-column probe on the
  current backend after the later descriptor-view work. The result is still
  negative: relaxing non-contiguous source-column preflight either exposes a
  later descriptor-row split/destination-mask requirement or no representable
  MMAv5 descriptor, and allowing transposed descriptors for `warpx4.32x128b`
  does not change the boundary. Keep this as evidence that the next copy slice
  must add a real source-message/source-format/destination-mask schedule rather
  than widening descriptor enumeration.
- 2026-04-15 22:02 UTC: sharpened the instruction-column projection diagnostic
  to identify column bits that select descriptor-row-stride multiples. This
  keeps the current clean negatives intact while making the next atomization
  target explicit: a schedule needs to represent column-selected row deltas
  without overwriting unrelated columns, or prove that the public
  `tcgen05.copy` ISA cannot mask/partition that projection.
- 2026-04-15 22:07 UTC: sharpened the dense row-projection diagnostic in the
  same direction. Row-permuted dense copies are now described as needing a
  destination-row mask, row-partitioned atom, or smaller copy footprint because
  current dense copy atoms write the full physical row footprint in basis
  order.
- 2026-04-15 22:23 UTC: sharpened the two-CTA no-scales
  `warpx2::02_13` known schedule gap. The descriptor path cannot represent the
  exact view because logical row bit 5 is a one-dword source offset rather
  than an affine 8-row source stride. The direct-seed `cta_group::2` path with
  source offset `32` proves opcode emission and the low destination columns,
  but duplicates that low source-column pair into the high destination
  columns; subaligned destination dword deltas fault and aligned deltas read
  zeros. Keep this as a real Phase 2 schedule gap until a `cta_group::2`
  descriptor/address plan preserves the high source-column bit.
- 2026-04-16 01:41 UTC: removed the blanket early `warpx2::02_13` two-CTA
  known-gap return from shared descriptor plan realization. The diagnostic now
  derives from the actual source-row projection planner failure, and the
  special schedule-gap evidence is attached only when the conversion proves
  row bit 5 is the one-dword offset case. This keeps the gap classified at the
  correct instruction-schedule layer without letting descriptor enumeration
  masquerade as a proof of support.
- 2026-04-15 22:29 UTC: rechecked the scales shared row-subslice copy row.
  It is the same Phase 2 atomization class as the scales descriptor-view copy:
  source column bit 2 maps to shared offset `1024`, or `32`
  descriptor-row strides, inside one `warpx4.32x128b` instruction. This
  confirms the next support-bearing copy edit must introduce a real
  sub-instruction source-column/message split or destination-column mask rather
  than another descriptor basis enumeration.
- 2026-04-15 22:33 UTC: converted instruction-column projection misses from
  inline string construction into `TMemCopyInstructionColumnProjectionFailure`.
  The planner now classifies descriptor-row-stride selection separately from
  packed-lane state, non-contiguous offsets, and non-offset components. This
  preserves current behavior while making the scales descriptor-view /
  shared-subslice split requirement a typed scheduler input instead of a
  diagnostic-only fact.
- 2026-04-15 22:42 UTC: re-probed legacy no-scales subword copy as a packed
  lane support path. Relaxing only the instruction-column predicate let
  zero-offset low lane bits pass, but descriptor synthesis still had no MMAv5
  shared-memory descriptor for the packed TMEM projection and tile planning
  still saw lane-expanded logical columns. Phase 2 therefore needs a
  lane-aware physical query and copy schedule for packed sub-dword lanes,
  not just a looser column-contiguity check.
- 2026-04-15 22:45 UTC: carried those two schedule requirements into the
  structured failure object. `TMemCopyInstructionColumnProjectionFailure` now
  records `packedLaneBits` for packed-lane failures and `descriptorRowDelta`
  for descriptor-row-stride selections, so the next scheduler implementation
  can consume the required lane or row-delta fact without reparsing diagnostic
  text.
- 2026-04-15 22:50 UTC: a temporary packed-lane descriptor projection that
  dropped zero-offset lane bases and scheduled physical dword columns compiled
  but produced wrong runtime output (`f16/i16` copied every second source
  column, `i8` every fourth). This rules out treating lane removal as a
  descriptor projection; correct support needs lane bits preserved as explicit
  schedule state.
- 2026-04-15 22:56 UTC: descriptor-row-stride instruction-column failures now
  record whether the source row delta spans a full copy-instruction row
  footprint. The scales descriptor-view row reports
  `descriptorRowDelta=32 spansInstructionRows=1`, and the diagnostic now names
  the public `tcgen05.copy` operand boundary: one tensor-memory address and
  one shared descriptor per instruction, with no per-column destination mask.
  This pushes the next support attempt toward physical-footprint scheduling
  instead of another descriptor enumeration.
- 2026-04-16 01:49 UTC: descriptor-row-stride failures now derive an explicit
  `TMemCopyDescriptorRowSplitRequirement`, not just scalar row-delta fields.
  The requirement spells out the column mask shape the scheduler would need:
  selected destination-column run, selection period, descriptor-row delta, and
  whether the row delta is a full instruction-row footprint. The scales
  descriptor-view case is now pinned as descriptor row `+32` for 4-column runs
  every 8 columns inside a 16-column instruction, spanning a full 32-row
  source footprint. This is the right input to a split/mask schedule proof;
  the public atom still remains clean unsupported until a non-overlapping
  destination schedule or different ISA format is proven.
- 2026-04-16 01:52 UTC: exposed
  `getTMemCopyDescriptorRowSplitRequirement(...)` as shared planner API and
  added query-debug output for the derived selected-column run and period.
  Future split scheduling should consume this helper directly.
- 2026-04-16 01:55 UTC: the shared descriptor-row split requirement now feeds
  a schedule-proof support check. When the selected destination-column run is
  narrower than the instruction footprint, the planner reports that separate
  descriptor-row messages would overwrite the complementary columns because
  the public copy atom writes the full destination footprint for each
  descriptor row. This classifies the scales descriptor-view/subslice copy
  gap as a masked/narrow-atom/source-format ISA boundary under current public
  `warpx4.32x128b`, not a descriptor synthesis gap.
- 2026-04-16 01:59 UTC: packed-lane source-storage failures now feed a
  `TMemCopyPackedLaneRequirement` and a dedicated schedule-proof diagnostic.
  The planner preserves the physical dword-column projection but rejects
  schedules that drop the low lane bases, because that aliases sub-dword
  lanes inside each 32-bit shared-memory word. Support now requires an
  explicit packed source-storage model through descriptor synthesis, source
  footprint planning, and instruction scheduling; the dense unpacked
  `TensorMemoryLinearLayout` subword path remains the supported layout form.
- 2026-04-15 23:08 UTC: factored copy destination-to-source conversion into
  `getTMemCopySourceConversion(...)` and routed both `TMEMCopyOp::verify()`
  and `TensorMemoryToLLVM.cpp` through it. This does not change support, but
  it gives Phase 2 a single exact-`LinearLayout` seam for future physical
  instruction-footprint scheduling and keeps verifier/lowering conversion
  algebra locked together.
- 2026-04-15 23:22 UTC: generalized scheduled copy tiles to carry
  `TMemCopyDestinationFootprint`: logical destination coordinate, physical
  row/column, instruction footprint rows/columns, and encoded TMEM offset.
  Lowering now consumes the footprint instead of a raw tile offset. Current
  schedules are unchanged, but future copy-planner work can reason about
  physical write footprints and overwrite/mask legality without re-deriving
  offsets in lowering.
- 2026-04-15 23:27 UTC: probed the `warpx2` dense-shared clean negatives by
  temporarily removing the canonical shared-linear offset-basis preflight.
  `warpx2::01_23` dense shared rows compiled and emitted the expected opcode,
  but runtime was wrong (`maxdiff=127` for both single-CTA and two-CTA, `f32`
  and `i32`); `warpx2::02_13` still failed deeper. Keep the preflight until
  the planner has an explicit source shared-layout schedule/projection proof.
- 2026-04-16 03:45 UTC: re-probed `warpx2` noncanonical shared layouts after
  the copy planner gained source-footprint proofs. The conclusion sharpened:
  descriptor representability is still not enough. Dense `01_23` now fails
  descriptor-loader footprint bounds, while other near-canonical `01_23`
  layouts and dense single-CTA `02_13` can emit opcodes but copy wrong logical
  source rows/columns. The direct-seed plan path is now guarded so a missing
  seed immediate cannot fall through to descriptor-loader lowering with stale
  direct-seed source semantics. The canonical shared-layout preflight remains
  until `warpx2` source rematerialization is modeled explicitly.
- 2026-04-15 23:40 UTC: added `TMemCopySourceFootprint` to scheduled copy
  instructions. The planner-owned emitted instruction stream now pairs each
  destination tile/footprint with the source row/column footprint that lowering
  will read, including direct-seed descriptor source offsets. Current schedules
  are unchanged, but source/destination footprint legality can now be checked
  in the planner instead of being reconstructed in lowering.
- 2026-04-15 23:43 UTC: added the message-adjusted destination footprint to
  scheduled copy instructions. `tmemRowDelta` and `tmemDwordDelta` are now
  applied during instruction scheduling, with lowering consuming
  `instruction.destination.offset` directly. Each emitted copy instruction now
  owns both final source and final destination footprints.
- 2026-04-15 23:47 UTC: added the first destination-footprint legality proof
  on top of those scheduled instructions. Ordinary copy-family plans are now
  rejected if the final physical destination footprints overlap; the diagnostic
  names this as an instruction-schedule failure requiring a non-overlapping
  plan or an ISA atom with an explicit destination mask. `Dense4x256b` is
  intentionally excluded until the refresh primitive has its own footprint
  model.
- 2026-04-15 23:51 UTC: removed that `Dense4x256b` exclusion by modeling the
  refresh primitive's effective instruction width. Each `4x256b` scheduled
  message now reports a 4-column source/destination footprint, matching the
  low/high half-message schedule at destination dword offsets `0` and `4`, so
  refresh copy participates in the same overlap legality proof.
- 2026-04-15 23:57 UTC: added the matching source-footprint bounds proof for
  dense descriptor-loaded copy schedules. The first all-family attempt exposed
  an important non-dense distinction: scales `warpx4` loader coordinates are
  descriptor-space coordinates, not plain logical source-tile coordinates, so
  non-dense families need a descriptor-space source proof rather than a tensor
  bounds check.
- 2026-04-15 23:59 UTC: made that distinction explicit in the plan carrier.
  `TMemCopySourceFootprint` now records logical shared-tile,
  descriptor-loader, or direct-seed-immediate coordinate space, and the dense
  bounds proof consumes that tag. This is the next seam for descriptor-space
  source proofs and packed-lane state.
- 2026-04-16 00:02 UTC: completed the descriptor-space source proof for
  selected descriptor-loader coordinates. Source-footprint support now checks
  descriptor-loader footprints against the selected descriptor layout bounds,
  so scales `warpx4` has a real source legality proof instead of an exemption
  from source-bounds checking.
- 2026-04-16 00:04 UTC: refreshed compiler-only invalid diagnostics for the
  current copy planner wording, keeping the lit/direct-verifier oracle aligned
  with the source legality and no-mask diagnostics.
- 2026-04-16 00:14 UTC: corrected the source-footprint coordinate model after
  broad copy validation exposed the dense half of the model was still assuming
  logical source columns. Descriptor-backed dense copies can also schedule in
  descriptor-loader coordinates when exact TMEM queries fold high logical row
  selectors into the copy column dimension. Source-footprint bounds now check
  every non-direct-seed copy message against its selected descriptor layout,
  while direct-seed immediates remain skipped.
- 2026-04-16 00:22 UTC: deleted the now-stale logical shared-tile coordinate
  mode from `TMemCopySourceFootprint`. The copy scheduler has two realized
  source coordinate spaces: selected descriptor-loader coordinates and
  direct-seed immediates. This keeps the source-footprint carrier aligned with
  the actual public `tcgen05.copy` operand model before the next scheduler
  work.
- 2026-04-16 02:46 UTC: dense copy destination scheduling was first moved to
  exact physical tile coordinates for every dense copy tile. This removed a
  macro-tile-only offset heuristic that let 4-column tile permutations compile
  with wrong logical tile addresses. The all-dense form was refined at
  03:05 UTC after broad copy validation exposed M256 folded-row regressions.
- 2026-04-16 03:05 UTC: dense copy destination scheduling now uses exact
  physical tile coordinates only when pure column tile selectors are permuted
  at or above the selected copy instruction width. Row-touching column bases
  are treated as folded-row source/descriptor selectors and keep logical
  destination column offsets. This preserves the promoted `tile_n=4`
  no-scales positives while restoring the broad M256 dense copy matrix.
- 2026-04-16 03:34 UTC: added runtime coverage for non-adjacent pure
  column tile-selector permutations. The planner now has regression coverage
  that proves it is instruction-footprint general for selector permutations,
  choosing `128x128b` only when the wider footprint would cross a selector
  boundary and otherwise staying on `128x256b`.
- 2026-04-16 00:25 UTC: added explicit source byte-range legality for
  direct-seed copy messages. Direct-seed messages still do not have a selected
  descriptor layout, but they now prove the immediate source offset and
  scheduled footprint stay inside the shared source tile instead of bypassing
  source bounds.
- 2026-04-16 00:32 UTC: packed-lane instruction-column failures now carry a
  `TMemCopyPackedLaneProjection` when the hidden sub-dword lane bits are
  followed by a contiguous physical dword-column stream. The support status is
  unchanged, but the planner now preserves the exact lane-aware schedule input
  (`laneBits`, `lanesPerDword`, `physicalInstructionColumns`, and physical
  offset steps) instead of reducing legacy subword copy to a diagnostic-only
  zero-offset column bit.
- 2026-04-16 00:39 UTC: a temporary raw-dword descriptor support attempt was
  removed. The candidate used the packed-lane carrier to drop hidden lane
  bases, schedule physical source columns, and synthesize a 32-bit MMAShared
  descriptor. It built, but the actual f16 shared layout had no representable
  MMAv5 descriptor for those raw-dword shapes. This rules out the simple
  "drop lanes + raw descriptor" path; support needs a packed source-storage
  descriptor model or must remain a clean ISA/layout boundary.
- 2026-04-15 20:44 UTC: direct `ld/st` verification now has a backend-level
  clean diagnostic for the `tcgen05.copy.4x256b` refresh-shaped layout. This
  does not promote direct `ld/st` support; it prevents bypassed frontend paths
  from falling through to generic register-layout enumeration and records the
  row-anchor schedule requirement in C++.
- 2026-04-17 05:18 UTC: the remaining type-only Python guard for that refresh
  layout and its raw physical bitcast was deleted. Type-only Gluon
  register-layout queries now call the backend
  `getUnsupportedDirectTMemLdStReason(MemDescType)` utility, and descriptor
  views reuse the same utility. This keeps the true packet-footprint boundary
  in C++ rather than split between frontend shape matching and backend
  verifier diagnostics.
- 2026-04-17 05:27 UTC: ordinary contiguous `tcgen05.copy.4x256b` is now
  represented as a typed refresh-image requirement instead of duplicated
  string guards. `TMemCopy4x256RefreshImageRequirement` records the source
  column split and destination refresh image facts, and both direct layout
  support and descriptor-plan realization format the same requirement. Support
  is unchanged: the 05:00 probe remains decisive evidence that opcode emission
  writes the refresh physical image into the wrong ordinary logical view.
  Future support must make that refresh image explicit in the planner and
  define a valid load/store/readback contract, or leave the ordinary layout as
  a clean negative.
- 2026-04-15 20:51 UTC: scales descriptor-view copy projection probes ruled
  out descriptor-basis reassignment as a standalone support path. The
  representable projection preserved columns but rotated rows; the row-correct
  projection was not representable as an MMAv5 shared descriptor. Treat the
  duplicate source offset `8` row/column role as a scheduler/ISA atomization
  problem, not another descriptor enumeration problem.
- 2026-04-15 19:23 UTC: moved destination tile planning into
  `TMemCopyExecutablePlan`. Plan realization now computes and stores
  destination tiles after descriptor/message scheduling, while lowering only
  consumes the selected executable plan. A temporary standalone-query probe for
  scales descriptor-view copy failed copy-family classification, confirming the
  remaining path must preserve the exact descriptor-view query and add real
  split scheduling.
- 2026-04-15 19:36 UTC: generalized the destination tile carrier into
  `TMemCopyScheduledTile`, which records both the logical destination column
  and the source shared-memory column for each emitted copy tile. Current
  schedules keep `sourceCol == logicalCol`, but lowering now consumes the
  selected plan's source coordinate instead of recomputing it from the
  destination tile loop. This is the behavior-preserving carrier needed before
  planner-generated source-column/message splits can be expressed cleanly.
- 2026-04-15 19:42 UTC: a temporary scales descriptor-view copy probe ruled
  out descriptor-enumeration as the next support path. Enabling the existing
  descriptor repartition search for `warpx4` and bypassing the
  instruction-column preflight produced `254` candidate `32x16` descriptor
  layouts for the exact view, but none were MMAShared-representable. The
  remaining gap needs an ISA-grounded split schedule for sub-instruction
  source columns or row partitions, not another descriptor basis shuffle.
- 2026-04-15 19:55 UTC: copy executable plans now materialize the actual
  emitted instruction stream as `TMemCopyScheduledInstruction` entries pairing
  a selected message index with a scheduled tile. The current schedule is the
  same tile-major/message-minor Cartesian product as before, so behavior is
  preserved, but lowering no longer owns that product. This gives future
  source-column/message split schedules one planner-owned representation
  instead of another lowering-local nested loop.
- 2026-04-15 20:07 UTC: factored the current Cartesian instruction-stream
  construction into `getTMemCopyInstructionSchedule(...)`. The helper has a
  structured failure path for invalid message schedules and is now the single
  utility that maps scheduled messages and tiles into emitted copy
  instructions. This is still behavior-preserving, but it narrows the next
  support-bearing edit to planner data rather than lowering control flow.
- 2026-04-15 20:11 UTC: bounded probes clarified the next Phase 2 frontier.
  Re-running the scales descriptor-view copy with query debug confirmed the
  exact physical view still maps logical column bit 2 to shared offset `256`
  inside a 16-column `warpx4` instruction. Temporarily bypassing the Python
  4x256b refresh ld/st guard reached the C++ register-layout generator, which
  rejected every candidate with `compInput cannot compose`; remove no guard or
  test expectation until a real row-anchor/register-layout representation is
  derived.
- 2026-04-15 20:14 UTC: added an explicit copy support failure layer for
  `instruction schedule`. Source-row projection, instruction-column
  projection, invalid emitted-instruction stream construction, and known
  cta-group::2 schedule gaps now report through the scheduler layer instead of
  descriptor synthesis. This keeps the planner semantics layered while
  preserving existing diagnostics.
- 2026-04-15 20:16 UTC: added
  `getTMemCopySourceFormatSupport(...)` and invoke it during copy plan
  realization. The helper validates non-default source-format suffixes as
  8-bit-source ISA variants before row/instruction scheduling or descriptor
  synthesis. Current plans still use `None`, so this is behavior-preserving
  scaffolding for source-format support attempts.
- 2026-04-15 20:18 UTC: a temporary source-format probe forced
  `.b8x16.b6x16_p32` and `.b8x16.b4x16_p64` onto the already-supported int8
  scales `warpx4` copy. Both variants compiled but produced wrong runtime
  output, so source-format suffix availability alone is not a valid support
  promotion for descriptor-view scales copy. The hook was removed; keep
  looking for a row-partition or sub-instruction schedule.
- 2026-04-15 20:36 UTC: scheduled copy tiles now carry explicit
  `logicalRow` and `sourceRow` coordinates. Current schedules still use zero
  row coordinates, but lowering consumes the planner-owned source row when
  loading the shared descriptor. This keeps row partitioning in the executable
  schedule model instead of forcing the next support attempt to rewrite
  lowering again.
- 2026-04-15 20:39 UTC: pure dense-copy row-order and row-repetition ordering
  failures now use the `instruction schedule` support layer. Mixed row/column
  bases remain physical-query failures, while row permutations are classified
  as missing explicit source-row schedules.
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
- 2026-04-15 18:32 UTC: re-probed dense row-permuted copies by temporarily
  lifting the row-order support guard. The inverse source projection reaches
  descriptor selection, but dense lowering then fails the `128x256b` affine
  row-stride invariant. This confirms descriptor projection alone is not the
  missing layer; the copy planner needs an explicit source-row schedule that
  can materialize non-affine physical row order, or it must keep the clean
  negative as an ISA atom boundary.
- 2026-04-15 18:35 UTC: factored the dense direct-copy row projection proof
  into `getDenseTMemCopyRowProjectionSupport(...)`. This keeps behavior
  unchanged while separating row/source projection legality from column tile
  offset legality, giving the future scheduler a clean place to replace the
  strict ascending-row proof with a real source-row schedule.
- 2026-04-16 03:18 UTC: rechecked dense row-permuted copy after the folded-row
  offset repair. Both the full `rotate1` row permutation and a high-row-tile
  scratch layout still stop at the row-order proof, confirming this frontier
  needs a destination-row mask, row-partitioned atom, or smaller copy footprint
  schedule rather than more destination column-offset logic.
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
- 2026-04-15 17:36 UTC: the dense row-permutation gap was sharpened with a
  second temporary probe. Bypassing both the physical-query row-order guard and
  the lowering row-stride guard lets row-only permutations compile, but they
  copy rows in physical row-basis order (`reverse` exposes bit-reversed source
  rows; `rotate1` exposes alternating low/high physical groups). The selected
  shared descriptor does not invert the destination row-basis permutation.
  Column-only permutations still fail at packet-contiguity. The real Phase 2
  task is now explicit: derive a row/source projection schedule or row-group
  atomization that proves logical row order, rather than relaxing dense guards.
- 2026-04-15 18:08 UTC: the dense row-permutation diagnostic now names that
  Phase 2 task directly. Non-identity row permutations assert the
  `source-row projection schedule` diagnostic in focused runtime-matrix
  coverage. This is a contract/diagnostic checkpoint; support remains blocked
  on a real row/source projection or row-group atomization plan.
- 2026-04-16 06:24 UTC: no-scales `warpx2` subword copy was moved past the old
  shared-element-bitwidth preflight and into the shared planner. The added
  subword `warpx2::01_23` 64-row descriptor candidate does not promote support:
  it proves the current boundary is representable packed/subword source
  storage and descriptor scheduling, while `warpx2::02_13` still fails the
  source-row projection proof for logical row bit 5. Keep this as a Phase 2
  support frontier rather than restoring the family-level guard.
- 2026-04-16 06:30 UTC: a temporary footprint-aware descriptor-selection probe
  showed that subword `warpx2::01_23` cannot be promoted by descriptor bounds
  alone. The probe compiled f16/bf16/i16 rows but selected a descriptor whose
  row unit offset skipped every other required subword source row, producing
  wrong output. Phase 2 needs descriptor semantic-equivalence checking against
  each copy atom's internal row/column mapping before descriptor variants can
  be treated as executable schedules.
- 2026-04-16 06:37 UTC: the subword source-storage / semantic-equivalence
  boundary is now part of live descriptor-synthesis diagnostics. When an
  instruction footprint spans more logical element columns than the source
  layout exposes, failures state that the extra sub-dword lanes need packed
  source-storage or descriptor semantic-equivalence modeling before lowering.
- 2026-04-16 06:45 UTC: a temporary scales source-format probe ruled out the
  simplest use of the dormant `.b8x16.b6x16_p32` and `.b8x16.b4x16_p64`
  suffixes for the scales row-split frontier. Adding those source formats as
  candidate `warpx4.32x128b` messages and bypassing the instruction-column
  projection preflight still failed descriptor synthesis: the exact source
  image has no representable MMAv5 shared descriptor for the descriptor-row
  `+32` split. Keep this bucket on real descriptor/source-format semantics
  plus a non-overwriting destination schedule, not a suffix-only promotion.
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
- 2026-04-15 17:59 UTC: re-probed the 4x256 refresh direct-ld/st boundary by
  temporarily bypassing the frontend guard. The backend raw-query planner also
  found no legal layout: it selected the 128-row row plan for the exact
  refresh query, but every atom returned no layout and the only fallback
  candidates had zero warp bases while carrying the needed 32/64 row anchors
  through lanes or logical columns. This confirms the remaining 4x256 work is
  not a stale guard; direct ld/st would need a new row-anchor rematerialization
  model, while copy opcode support is already positive for the ISA refresh
  image.
- 2026-04-16 10:43 UTC: descriptor-level direct `ld/st` rejection for the
  4x256b refresh image is now centralized in the shared backend
  `isUnsupportedDirectTMemLdStDescriptorView(...)` helper. Python descriptor
  methods no longer duplicate that physical-image recognizer; the type-only
  Python guard remains only because no memdesc handle is available there.

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
- 2026-04-15 21:09 UTC: completed that M64 split-N cross-lane reduction
  design for the ISA-realizable lane-16 case. The backend now models the
  reduction layout as either all N bits in registers or exactly one missing N
  basis carried by lane bit 4. Lowering combines per-message partial
  reductions, waits for `tcgen05.ld.red`, and then combines the lane-16 pair
  with `shfl.xor 16` and the requested min/max/NaN semantics. M64
  `N in {32,64,128,256}` is positive for default reductions and explicit
  `auto`, `32x32b`, `16x32bx2`, and `32x32b_splitn`; broader N sharding
  remains a software-reduction boundary.
- 2026-04-15 21:23 UTC: closed the adjacent M64 row-permutation default
  reduction gap. Probes showed row-permuted M64 layouts are hardware
  reducible through the existing split-N `16x32bx2` schedule; only the default
  frontend request for a `32x32b` register layout scalarized into illegal
  `.ld.red.x1` packets. Default reduction layout selection now chooses the
  handle-aware split-N layout for noncanonical M64 row layouts, and verifier
  plus lowering reject any selected reduction message shape below the PTX
  `.x2` minimum. This is still a Phase 4 cleanup target: the next step is to
  make reduction layout selection a backend message-planner decision instead
  of relying on frontend variant spelling.
- 2026-04-15 21:45 UTC: promoted canonical single-CTA block-row layouts for
  `ld/st` and `ld.red`. The `block` input dimension in these layouts was only
  carrying the low row basis (`[1,0]`) before the ordinary `row` bases
  (`[2,0]`, `[4,0]`, ...), so for `two_ctas=False` it is algebraically the
  same TMEM image as pure row bases. The frontend now folds exactly this
  canonical form before IR construction, and the backend applies the same
  conservative fold before view analysis, row-anchor planning, physical support
  checks, and reduction friendliness. The former `block_single_cta` clean
  negatives are now positive direct/descriptor `ld/st` rows plus M128
  `ld.red` rows for `N in {64,128,256}`; true two-CTA block layouts remain
  untouched.
- 2026-04-16 00:57 UTC: cleaned up multidimensional descriptor-view reshape
  inference for active subviews that are smaller than their backing allocation
  image. The backend now strips leading unit dimensions by comparing descriptor
  rank to layout rank. This removes a false `memdesc_reshape` type-inference
  failure for the identity `[1,32,1,32] -> [32,32]` slice chain. It does not
  promote that row:
  after exact view inference succeeds, the planner correctly reports the real
  direct `tcgen05.ld/st` row-anchor boundary for the canonical `32x32` subview.
- 2026-04-16 01:04 UTC: a temporary 32-row `ld/st` query-plan lift for the
  same exact `32x32` descriptor-view row was removed. It exposed a deeper
  active-subview type-inference mismatch in intermediate `memdesc_subslice`
  verification instead of proving support. Keep this row as a clean
  row-anchor/rematerialization boundary until active-view inference and packet
  scheduling are both modeled algebraically.
- 2026-04-16 07:32 UTC: promoted explicit two-CTA tensor-memory-scales
  descriptor-view `ld/st` for `16x64b`, `16x128b`, and `16x256b`. The backend
  now derives those register layouts by constructing the requested atom's
  pre-packed physical packet basis and lifting each physical basis through the
  exact descriptor-view query layout, rather than adding another shape-family
  rewrite. The exact encoding-info validator remains the support proof. The
  expanded runtime matrix covers all realizable power-of-two `N=4..128` rows
  for `M in {128,256}` and pins the explicit two-offset packet schedule.
- 2026-04-16 07:51 UTC: pinned the adjacent explicit two-CTA scales
  descriptor-view `16x32bx2` row as an atom-semantic clean negative. The exact
  physical query is valid and remains realizable by `32x32b` and by wider
  n-sharded scale atoms, but `16x32bx2` specifically requires its half-tile
  split to be a lane-selected second-half offset. In this descriptor view that
  split is represented by register/message repetition, so footprint or shape
  promotion would be a false support claim. The frontend split-N fallback now
  preserves the backend requested-variant reason for this row.
- 2026-04-16 07:56 UTC: promoted the row-permuted M64 explicit-`32x32b`
  `ld.red` row by making reduction layout selection reuse the existing
  handle-aware split-N planner when the provided direct `32x32b` layout would
  produce scalar `.x1` reduction packets. This keeps load/store layout queries
  stable while eliminating a frontend spelling gap for an already
  ISA-realizable M64 physical query. Long-term cleanup is still to move this
  decision into a backend reduction message planner.
- 2026-04-16 07:58 UTC: tightened the previous M64 explicit-`32x32b`
  reduction promotion so `_load_red` first asks the backend
  `compute_tmem_reduce_reg_layout_from_memdesc(...)` helper for the
  reduction-compatible layout. The frontend split-N helper remains only a
  fallback. Positive coverage now spans the existing M64 row/column-permuted
  default-case matrix.
- 2026-04-16 08:46 UTC: tightened the backend reduction helper itself so it
  validates actual hardware-reduction message legality (`packed`, at least
  `.x2` reduction repeats) and can consider `16x32bx2` after `32x32b`.
  A deliberately broad default-routing probe was rejected: message legality
  alone caused non-M64 column-permuted packet-order changes and a row-permuted
  runtime mismatch. Keep the helper scoped to the M64 scalarization boundary
  until the backend reduction planner proves exact physical-query equivalence,
  including packet order, row origins, and row/column permutation effects.
- 2026-04-16 09:07 UTC: factored that `.ld.red` message predicate into shared
  `TensorMemoryUtils` helpers and routed the reduction layout helper,
  verifier, and Python bridge through the same check. This is a cleanup
  checkpoint for Phase 4 rather than a support promotion. A forced-helper
  probe showed the unsafe broad default path can emit legal
  `32x32b.x128` packets while returning row-permuted output, so the next
  support-bearing reduction step needs an equivalence proof over physical row
  order and packet origins, not only message legality.
- 2026-04-16 09:18 UTC: default reductions now use the backend reduction
  helper as a safe first query for f32 non-scales TMEM. The helper's contract
  is intentionally conservative: it returns no override whenever direct
  `32x32b` already satisfies the reduction predicate, preserving the existing
  direct path's exact packet order for non-M64 row/column permutations. It
  still returns the split-N rescue layout for M64 scalarized direct
  reductions, so the special case is backend-routed without reopening the
  rejected broad-route failures.
- 2026-04-16 09:59 UTC: moved that direct-compatible `32x32b` no-override
  rule into C++ `getTmemLoadReductionLayout(...)`, so the backend helper owns
  the safe selector contract and the Python binding no longer duplicates the
  packet-order guard.
- 2026-04-16 10:08 UTC: removed the now-redundant Python default-path M64
  split-N fallback from `_load_red(layout=None)`. Default reductions now rely
  on the backend bridge for M64 split-N selection after the helper's
  no-override contract, while the explicit-`32x32b` helper fallback remains as
  a distinct user-provided-layout escape path.
- 2026-04-16 10:12 UTC: re-probed deleting that explicit-`32x32b` M64 helper
  fallback and rejected the deletion. Noncanonical M64 row/column permutations
  still need the frontend handle-aware split-N fallback when the backend bridge
  returns no layout; otherwise they select scalar `.x1` reduction packets.
  This pins the next Phase 4 cleanup target: teach the backend reduction
  helper to derive those noncanonical M64 split-N layouts directly.
- 2026-04-16 10:38 UTC: completed that Phase 4 cleanup target for the current
  M64 row/column-permuted reduction matrix. The C++ memdesc reduction bridge
  now recognizes exact raw rank-2 M64 f32 non-scales queries with
  noncanonical row bases, derives the canonical M64 split-N layout, and
  validates it before returning it to `_load_red`. The explicit Python helper
  no longer falls back to the frontend split-N planner. A canonical-row guard
  preserves the existing column-only M64 path and its packet offsets.
- 2026-04-16 11:07 UTC: re-probed deleting the broader handle-aware M64
  split-N Python fallback and rejected the deletion. Backend-only
  `32x32b_splitn` selection still fails row/column-permuted M64 auto opcode
  checks and M64 reduction output for noncanonical views. The reduction bridge
  cleanup above is valid, but generic M64 split-N auto selection still needs a
  backend physical-query/packet-equivalence proof before the Python fallback
  can be removed.
- 2026-04-16 11:54 UTC: completed that generic M64 split-N auto-selection
  cleanup for simple noncanonical rank-2 M64 raw queries. The C++ memdesc
  register-layout bridge now proves the split-N support image by checking one
  zero row basis plus permuted power-of-two row/column bases and returns the
  canonical split-N register layout for `32x32b_splitn`. The handle-aware
  Python auto helper now trusts the backend query directly. A probe showed the
  separate explicit-`16x32bx2` requested-variant path still needs its Python
  canonical fallback; keep that as the next cleanup target rather than
  treating all M64 split-N fallback deletion as complete.
- 2026-04-16 12:00 UTC: completed the adjacent explicit-`16x32bx2` cleanup.
  The same C++ raw-query recognizer now handles both `32x32b_splitn` and
  `16x32bx2`, and the Python canonical M64 split-N constructor/fallback has
  been removed. The M64 row/column-permuted split-N register-layout family is
  now backend-owned for auto/default and explicit split-N requests.
- 2026-04-17 04:53 UTC: removed the remaining Python explicit-`32x32b` M64
  `ld.red` helper stack. The C++ `create_tmem_load` bridge now receives
  `numWarps`, recognizes simple noncanonical M64 f32 non-scales raw queries,
  and selects the canonical split-N reduction layout before creating
  `ttng.tmem_load`. The M64 split-N reduction family is now backend-owned for
  default, explicit split-N, and explicit logical `32x32b` reduction requests.
- 2026-04-17 05:03 UTC: removed the Python M64 split-N auto-layout shim. The
  C++ raw-query recognizer now handles `instr_variant="auto"` before generic
  raw-query selection for rank-2 M64 f32 non-scales descriptor values, so
  `load(layout=None)` and `get_reg_layout(auto)` no longer need to pre-query
  `32x32b_splitn` from Python.
- 2026-04-17 05:11 UTC: removed the descriptor-handle Python type-only
  shortcut for rank-2 M64 f16/bf16 split-N layout requests. The same C++
  simple M64 raw-query recognizer now covers 16-bit descriptor values, returns
  the canonical split-N user layout for `auto`, and keeps `32x32b_splitn` on
  the backend-owned M64 `16x32bx2` route. Focused coverage now compares
  `auto`, `32x32b_splitn`, and explicit `16x32bx2` for representative 16-bit
  M64 rows.
- 2026-04-17 05:13 UTC: removed the adjacent Python `store()` equality guard
  for rank-2 M64 f16/bf16 descriptors. Store source-layout legality now uses
  `TMEMStoreOp::verify()` and the shared backend `computeTMemLdStEncodingInfo`
  proof instead of a frontend shape-family check.
- 2026-04-17 05:53 UTC: moved the simple M64 split-N raw-query proof itself
  out of `python/src/gluon_ir.cc` and into `TensorMemoryUtils`. Pybind still
  marshals the result to Gluon layouts, but the row/column-basis recognizer and
  canonical split-N layout derivation are now shared backend utilities used by
  both handle-aware `get_reg_layout()` and M64 `ld.red` canonicalization.
- 2026-04-17 06:01 UTC: moved the broader value-aware reduction
  register-layout selector out of `python/src/gluon_ir.cc` and into
  `TensorMemoryUtils` as `getTMemLoadReductionLayoutForMemDesc(...)`. That
  backend API now owns raw-query reduction compatibility, descriptor-view
  exact-query refusal, row-plan selection, and the M64 split-N rescue path.
  Support is unchanged, but reduction layout selection is now another step
  closer to the shared backend planner instead of pybind-local layout logic.
- 2026-04-17 05:00 UTC: re-probed ordinary contiguous `tcgen05.copy.4x256b` by
  bypassing the refresh-only support and lowering guards. The ISA opcode
  emitted, but the output copied the refresh physical pattern into the ordinary
  parent view (`[0, 8, 16, 24, 0, 8, 16, 24]` in row 0 for input `0..31`),
  confirming the guard is a real representation boundary. Covering ordinary
  4x256b requires a first-class refresh-image remapping layer, not an
  atom-enable patch.
- 2026-04-16 09:37 UTC: re-probed two-CTA tensor-memory-scales descriptor-view
  direct `ld/st` at `M=64`. The exact-query shape is close to the promoted
  `M in {128,256}` forms, but support is not a stale shape gate: one required
  32-row warp anchor is represented as broadcast/support state, so the current
  scales lowering cannot materialize the 32/64 row-anchor pair. The backend
  diagnostic now names this M64 row-anchor rematerialization /
  packet-footprint boundary directly, and the runtime matrix pins it as a
  clean negative.
- 2026-04-16 01:29 UTC: removed the reshape zero-basis cardinality trim from
  the 00:57 cleanup after it regressed scales descriptor-view `ld/st`.
  Preserving zero row/column/block bases is the correct Phase 4 invariant:
  these bases are support/broadcast axes, and operation planners must prove
  or reject their equivalence classes instead of reshape inference deleting
  them. The scales descriptor-view positives and the identity descriptor-view
  clean-negative are green together with this simpler rule.
- 2026-04-16 02:35 UTC: fixed chained `memdesc_subslice` folding so combined
  offsets are re-inferred against the original source memdesc and the folded
  result type is updated accordingly. This keeps active tensor-memory view
  types consistent after canonicalization. The exact identity `32x32`
  descriptor-view `ld/st` support probe that exposed the mismatch remains
  unsupported for ISA reasons: without a load predicate/rematerialization
  model, the public packet schedule updates a wider row window than the
  logical view.
- 2026-04-16 05:42 UTC: re-probed lifted row-half direct `ld/st` after the
  diagnostic checkpoint. Dimension-aware already-adjusted-base subtraction can
  preserve the outer leading-buffer column offset and make the selected half
  visible through the right full-view tile, but the public packet footprint
  still updates the complementary row half. Forcing a logical 64-row row plan
  into the 128-row support image does not avoid that over-update. This keeps
  lifted half-row support as a true packet/warp rematerialization or predicate
  model task, not a base-offset or row-plan override cleanup.
- 2026-04-16 06:11 UTC: re-probed the exact identity `32x32`
  descriptor-view high quadrant after active-view folding was fixed. The
  standalone query algebra is correct and carries `origin=(64,64)`, but naive
  support promotions are wrong-code: final/root support layouts select wider
  `16x32bx2.x32.b32` row bands, and a forced 32-row row plan selects scalar
  packets that over-cover all rows for the target columns. This keeps the row
  as a Phase 4 packet-footprint/rematerialization task, not a type fallback or
  row-plan lookup task.
- 2026-04-16 06:15 UTC: frontend descriptor-view register-layout failures now
  surface the backend's direct-`ld/st` unsupported reason. The row-anchor
  diagnostic explicitly points to the Phase 4 packet-footprint abstraction
  that is still missing, so the expanded runtime matrix no longer collapses
  the identity high-quadrant gap into a generic frontend "auto unsupported"
  message.
- 2026-04-16 07:15 UTC: promoted the stale two-CTA scales descriptor-view
  shape gate for direct `ld/st`. The exact-query recognizer now covers
  `M in {128,256}` with power-of-two `N=4..128`, and the returned layout is
  still selected only after exact-query `computeTMemLdStEncodingInfo(...)`
  validation. Runtime coverage now spans the full promoted CGA range with
  `16x32bx2` root packets and `32x32b` descriptor-view packets. `M=64`
  remains a row-anchor/packet-footprint boundary.
- 2026-04-16 23:56 UTC: direct `ld/st` packet-footprint clean negatives now
  share a typed requirement layer. Missing descriptor-view row anchors,
  translated row-half origins, raw 4x256b refresh physical bitcasts, and M64
  two-CTA scales broadcast row anchors all route through
  `TMemLdStPacketFootprintRequirement`. This is not a support promotion; it
  aligns the current clean negatives under the same missing abstraction:
  packet-base and row-anchor rematerialization plus per-message offsets, or an
  explicit read/modify/write or mask footprint model.
- 2026-04-17 00:05 UTC: extended that same packet-footprint requirement layer
  to explicit n-sharded atom column footprints for 32-bit direct `ld/st` views.
  The x1 `f32/i32` explicit `16x128b` rows now report a backend-derived
  variant-specific reason: the public atom covers 4 dword columns but the view
  materializes only 1. This confirms these rows are true footprint/mask
  boundaries, not stale descriptor enumeration gaps, while preserving the
  positive `32x32b.x1` path.
- 2026-04-17 05:46 UTC: explicit `16x32bx2` two-CTA scales descriptor-view
  `ld/st` now gets its atom-specific diagnostic from `TensorMemoryUtils`
  instead of the Python binding. The backend checks that wider supported atoms
  can realize the exact view and explains that `16x32bx2` needs a
  lane-selected second-half offset, while this descriptor view carries the
  split through register/message repetition. A 16-row-per-CTA support probe
  did not promote the row and was reverted.
- 2026-04-17 00:12 UTC: promoted block-backed two-CTA higher-rank index
  descriptor views. The old row-anchor diagnostic assumed anchors must appear
  as pure row bases; the successful schedules prove the anchors can be carried
  through a nontrivial two-CTA `block` dimension and then validated by concrete
  register-layout selection plus `TMEMLoad/Store` verification. The runtime
  matrix now makes the four representative two-CTA index rows positive while
  preserving the lifted half-row and single-CTA multidim row-anchor negatives.
- 2026-04-17 06:10 UTC: re-probed the remaining single-CTA identity `32x32`
  multidim-slice descriptor-view `ld/st` boundary with a base-relative support
  query. The support image can make the compiler choose a scalar `32x32b.x1`
  atom, but the resulting packet schedule still updates the wrong physical
  footprint because the high-row/high-column subview origin is not represented
  as a proven packet-base plus per-message offset schedule. Keep this as a
  Phase 4 packet-rematerialization or mask/read-modify-write task; do not
  retry it as another support-layout fallback.
- 2026-04-17 06:16 UTC: removed the pybind-local TMEM atom request
  interpretation. Access-atom spelling, split-N alias mapping for descriptor
  handles, the M64 split-N descriptor-type predicate, and M64
  requested-vs-realized atom compatibility now live in `TensorMemoryUtils`.
  This is a layering cleanup toward one backend planner contract: support did
  not change, but frontend code no longer carries a separate copy of the
  split-N/M64 atom semantics.
- 2026-04-17 06:19 UTC: moved the remaining pybind-local M64 reduction-load
  result-type canonicalization into `TensorMemoryUtils`. The new backend API
  keeps the same exact-query guard and reduction-friendly proof while removing
  another frontend copy of the M64 split-N raw-query rewrite.
- 2026-04-17 06:22 UTC: moved the M64 split-N raw-query fallback request
  predicate into `TensorMemoryUtils`. The backend now decides when the
  `auto`/`32x32b_splitn`/`16x32bx2` fallback may return the canonical split-N
  layout; pybind no longer owns that request/desired-atom/M64-type filter.

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
- 2026-04-15 22:10 UTC: unsupported plain-MMAv5 tensor-memory operand
  diagnostics now carry a physical-tile note explaining that current public
  `tcgen05.mma` atoms require canonical row/column basis order within each
  instruction tile. This keeps exotic and row/column-permuted accumulator
  layouts as clean negatives while identifying the real support frontier as a
  permutation, masked writeback, or tile-splitting schedule rather than a
  generic "layout not compatible" bucket.
- 2026-04-15 17:28 UTC: promoted scaled-MMAv5 two-CTA accumulator subslice
  `block_n=32` rows. Scaled root/view/two-CTA N32 coverage is now positive
  where each operation maps to supported scale fragments.
- 2026-04-15 17:26 UTC: promoted supported scaled-MMAv5 `N=32`
  root/index/subslice/LHS matrices. This narrows the remaining scale issue:
  single-tile/root/view `N=32` is valid, while repeated/tile-permuted `N=32`
  remains a scale-B fragment/addressing gap.
- 2026-04-15 17:55 UTC: verifier and lowering now share one helper for the
  repeated-`N=32` scaled-MMAv5 boundary. Probes ruled out the easy fixes:
  fixed B-scale selector remaps stay wrong, and removing the forced
  64-column B-scale address stride reaches a misaligned scale address. The
  remaining support task is a real B-scale fragment representation below the
  current public 64-column scale layout, not another guard or selector tweak.
- 2026-04-15 18:17 UTC: a broader temporary B-scale address/SFB-ID sweep for
  repeated `N=32` confirmed the same boundary. Address/sub-ID formulas
  `wordaddr_nid` and `wordaddr_xor` compiled but stayed wrong; `kaddr_nid`,
  `kaddr_xor`, and `packed` faulted. The default guard lift again left only
  the first 32-column tile correct. A support path needs a fragment planner
  that models B-scale N/K sub-fragments explicitly.
- 2026-04-15 18:20 UTC: factored scaled-MMAv5 scale operand addressing into
  `MMAv5ScaleFactorFragment`, a behavior-preserving helper that returns a
  TMEM column offset plus SFA/SFB sub-column ID for each MMA repetition. This
  keeps all current semantics and diagnostics intact while replacing inline
  arithmetic with the abstraction point needed for repeated-N32 B-scale
  fragment modeling.
- 2026-04-15 18:28 UTC: probed the scale-B `scale_vec::1X` XOR schedule
  implied by the PTX B-scale sub-column figure. The derived schedule reaches
  odd B-scale TMEM word columns for some repeated-N32 N/K fragments, and the
  hardware rejects those addresses as misaligned. This makes the remaining
  support task larger than B-scale address arithmetic: either the B-scale
  storage contract must provide an aligned fragment view for repeated N32, or
  the clean negative remains a true layout/ISA boundary for the current public
  scales layout.
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
- 2026-04-16 09:49 UTC: the repeated-`N=32` probe was repeated on the current
  backend by bypassing both live guard sites. The mxfp8/mxfp8 representative
  still compiled but returned wrong results (`12274 / 16384` mismatches),
  while the restored clean-negative selector passed (`10 passed`). This
  boundary remains current after the copy/reduction cleanups.
- 2026-04-16 23:59 UTC: the repeated-`N=32` scaled-MMAv5 boundary is now a
  typed `MMAv5ScaledRepeatedN32ScaleFragmentRequirement` instead of only a
  diagnostic string. Verifier and lowering still consume the existing error
  wrapper, but future support work can use the structured accumulator encoding,
  selected N instruction size, CTA N columns, and repeated-N instruction count
  when building a real B-scale fragment planner.
- 2026-04-17 05:23 UTC: verifier and lowering now consume a shared
  `MMAv5ScaledAccumulatorSupport` object. The object carries the selected
  scaled accumulator layout proof plus typed repeated-N32 and narrow-N
  requirements, tightening the abstraction boundary for future B-scale
  fragment planner work without changing current support.
- 2026-04-17 05:37 UTC: removed the frontend `tcgen05_mma_scaled` blockM=64
  assertion and added a focused clean negative showing the backend verifier
  owns that ISA rule (`only supports instruction shape blockM=128`). Repeated
  N32 was also re-probed: simply lifting the guard produces wrong output, and
  a one-column B-scale fragment variant faults/fails fresh processes. The
  remaining work is therefore an actual B-scale fragment storage/alignment
  model, not another frontend guard or address tweak.
- 2026-04-15 22:29 UTC: temporarily bypassed the mixed fp4A TMEM-LHS verifier
  guard and found that representative tile and subslice TMEM-LHS cases compile
  but are numerically wrong (`max ~= 1084`, `mean ~= 69.8`) for both legacy
  and linear accumulator layouts. Keep mixed fp4A TMEM-LHS as a padded
  operand-A storage contract boundary until the TMEM storage model can express
  the same padded physical image as shared memory.

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
