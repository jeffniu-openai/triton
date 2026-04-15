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
  with descriptor synthesis and conversion coverage.
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
  reaches both cta-group ISA variants.

Exit criteria:
- Copy support decisions can be explained by a planner trace instead of by a
  hard-coded recognizer branch.
- True scales `warpx2` and no-scales two-CTA `warpx2::02_13` are either
  positive or rejected with proof-level diagnostics.
- `4x256b`, if ISA-valid for the targeted backend, is represented as a normal
  copy atom rather than a special case.

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
