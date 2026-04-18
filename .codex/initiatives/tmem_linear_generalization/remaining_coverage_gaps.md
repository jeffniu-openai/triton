# TMEM Linear-Layout Remaining Coverage Gaps

Last updated: 2026-04-18 23:03 UTC

This is the stable reference list for remaining TMEM support gaps whose
coverage depends on linear-layout generalization. The list was reconsolidated
on 2026-04-18 to remove old `tcgen05.cp` placeholder gaps and to move
non-linear ISA/API follow-ups, such as i8 MMAv5 signedness and saturation, out
to `isa_api_followups.md`. New linear-layout discoveries should be appended as
new gap numbers unless the user explicitly asks to reconsolidate again.

Historical note: the former i8 MMAv5 Gap #1 is closed for this project. Signed
i8 direct MMAv5 is supported for the current project scope; extra
instruction-descriptor attributes such as independent A/B signedness and
integer saturation controls are out of scope here.

Each gap should be examined with the same decision standard:
- Is this impossible under the public PTX/ISA and current hardware behavior?
- Is it possible with compiler-only schedule/layout work?
- Is it possible only after an explicit frontend/API/storage contract change?
- What compile-only, FileCheck, PTX, PTXAS, or runtime evidence is needed to
  close the question?

## Gap #1: `tcgen05.cp` Complete Support Umbrella

- Status: closed for the TMEM linear-layout generalization project at
  2026-04-18 19:50 UTC.
- Area: all remaining `tcgen05.cp` support and coverage questions after
  normalization to `LinearLayout`, including partial-footprint work,
  `4x256b` refresh-view work, packed/subword copy work, two-CTA
  `warpx2::02_13` work, and copy-source/frontend contract work.
- Current state: the formal audit is recorded in
  `tcgen05_cp_gap2_audit_20260418.md`. The planner/verifier path normalizes
  through `LinearLayout`: legacy `TensorMemoryLayout` encodings are frontend
  compatibility syntax, not a separate backend capability. The 2026-04-18
  19:32 UTC implementation checkpoint additionally removed the copy planner's
  scales-versus-non-scales support mode, so copy coverage should be judged by
  normalized layout equivalence classes and public instruction-family
  footprints rather than by source/destination encoding spelling.
- Supported baseline: no broad new `tcgen05.cp` implementation gap was found.
  Supported public families are dense `128x128b`, dense `128x256b`, explicit
  refresh-shaped `4x256b`, 32-bit no-scales `warpx2::01_23`, 32-bit
  no-scales single-CTA `warpx2::02_13`, and scales `warpx4.32x128b`.
- Resolved sub-buckets:
  - `1A`: closed at 2026-04-18 19:46 UTC. Dense subword exact-width
    `128x128b` evidence now has positive f16 `128x8` and i8 `128x16` rows
    using explicit `TensorMemoryLinearLayout`.
  - `1B`: closed as classified. Full public-copy atoms and tile-selector
    permutations outside the instruction footprint are supported; remaining
    sub-instruction permutations, row/column masks, mixed physical bases, and
    noncanonical two-CTA ownership need unavailable masks/smaller atoms or
    remain clean unsupported.
  - `1C`: closed as classified. Explicit refresh-shaped `4x256b` copy layouts
    are positive for one-CTA and two-CTA; ordinary contiguous `4x8` exposure and
    direct `ld/st` readback are deferred behind a first-class refresh-image
    view/remap/load-store contract.
  - `1D`: closed as classified. Direct subword no-scales `warpx2` copy needs
    packed-lane storage semantics that public `tcgen05.cp` does not expose.
    Future staged support belongs to a packed-storage/staged-copy project.
  - `1E`: closed as an ISA/schedule boundary. Current descriptor plans need
    row-selected source offsets inside one full 64-row atom, and the direct-seed
    sweep emitted `cta_group::2.warpx2::02_13` only with low-column duplication,
    zero reads, launch failures, or illegal/faulting deltas.
  - `1F`: closed as classified. Safe source rematerialization is supported;
    remaining descriptor-view/copy-source cases need explicit frontend/API
    contracts or remain rejected when implicit behavior is ambiguous or too
    expensive.
- Closure artifact: `tcgen05_cp_gap1_closure_20260418.md`.
- Current classification: no active `tcgen05.cp` linear-layout gap remains.
  Future work on masks, packed lanes, refresh-image readback, or frontend
  copy-source APIs should be tracked as separate ISA/API/storage follow-ups.

## Gap #2: Narrow Scaled-MMAv5 `N=8/16`

- Status: closed at 2026-04-18 23:03 UTC.
- Area: block-scaled MMAv5 accumulator layouts that require narrow N
  instruction fragments, especially `N=8` and `N=16`.
- Current state: positive supported. Scaled accumulator planning now admits the
  same narrow `N=8/16` linear-layout families as plain MMAv5 when the
  accumulator tile itself is instruction-family compatible.
- Implementation: the tensor-memory allocation pass rematerializes matrix-B
  scale storage for narrow fragments by padding each logical N-fragment group
  to the public 64-column scale-fragment addressing granularity. Lowering then
  uses the normal scaled-MMAv5 path with `N=8` or `N=16` instruction
  descriptors and the rematerialized scale storage.
- Coverage: `test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_narrow_format_matrix`
  is positive across `mxfp8`, `mxfp4`, mixed `mxfp8/mxfp4`, mixed
  `mxfp4/mxfp8`, and `nvfp4` for `N=32/tile_n=8` and `N=64/tile_n=16`,
  with `K=128/256`.
- Validation: `make -j8`; full narrow-N runtime matrix `20 passed`; broader
  non-twoCTA scaled-accumulator tile-permuted selector `52 passed`.

## Gap #3: Mixed fp4 TMEM LHS

- Status: closed as a direct-TMEM unsupported boundary at 2026-04-18 23:03 UTC.
- Area: block-scaled MMAv5 with mixed fp4 operand A in tensor memory, such as
  fp4 A with non-fp4 B under `mxf8f6f4`-style semantics.
- Current state: current TMEM LHS lowering treats A storage as raw packed
  columns. Mixed fp4 A needs the padded operand-A storage model represented by
  `fp4_padded` shared memory, where only part of each group carries real
  packed fp4 values and the rest are padding aliases.
- Closure evidence: a speculative direct-TMEM fp4-padded rematerialization
  path was tried and rejected because it compiled but produced wrong runtime
  numerics. The zero-basis padding-only representation missed the
  shared-memory `fp4_padded` row-dependent 128-byte swizzle; attempting to
  encode that row swizzle in the TMEM/register layout was not compatible with
  the current direct TMEM store layout machinery.
- Classification: direct raw TMEM LHS storage cannot model the
  `fp4_padded` operand-A contract. Supporting mixed fp4A from tensor memory
  would require a new explicit storage/API contract or staged shared-memory
  materialization path, not another verifier guard lift in the general
  linear-layout backend.
- Coverage: mixed fp4A TMEM-LHS tile-permuted and subslice-view rows remain
  clean structured negatives that tell users to keep operand A in shared
  memory or use a homogeneous fp4 scaled-MMA kind.
- Validation: `make -j8`; full mixed-fp4A TMEM-LHS clean-negative matrix
  `24 passed`.
