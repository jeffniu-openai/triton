# TMEM Linear-Layout Remaining Coverage Gaps

Last updated: 2026-04-18 07:13 UTC

This is the stable reference list for remaining TMEM support gaps whose
coverage depends on linear-layout generalization. The list was reconsolidated
on 2026-04-18 to remove old `tcgen05.cp` placeholder gaps and to move
non-linear ISA/API follow-ups, such as i8 MMAv5 signedness and saturation, out
to `isa_api_followups.md`. New linear-layout discoveries should be appended as
new gap numbers unless the user explicitly asks to reconsolidate again.

Each gap should be examined with the same decision standard:
- Is this impossible under the public PTX/ISA and current hardware behavior?
- Is it possible with compiler-only schedule/layout work?
- Is it possible only after an explicit frontend/API/storage contract change?
- What compile-only, FileCheck, PTX, PTXAS, or runtime evidence is needed to
  close the question?

## Gap #1: `tcgen05.cp` Complete Support Umbrella

- Area: all remaining `tcgen05.cp` support and coverage questions after
  normalization to `LinearLayout`, including partial-footprint work,
  `4x256b` refresh-view work, packed/subword copy work, two-CTA
  `warpx2::02_13` work, and copy-source/frontend contract work.
- Current state: the formal audit is recorded in
  `tcgen05_cp_gap2_audit_20260418.md`. The planner/verifier path normalizes
  through `LinearLayout`: legacy `TensorMemoryLayout` encodings are frontend
  compatibility syntax, not a separate backend capability. Coverage should be
  judged by normalized layout equivalence classes.
- Supported baseline: no broad new `tcgen05.cp` implementation gap was found.
  Supported public families are dense `128x128b`, dense `128x256b`, explicit
  refresh-shaped `4x256b`, 32-bit no-scales `warpx2::01_23`, 32-bit
  no-scales single-CTA `warpx2::02_13`, and scales `warpx4.32x128b`.
- Active sub-buckets:
  - `1A`: optional evidence polish for dense subword exact-width `128x128b`
    rows, such as f16 `128x8` or i8 `128x16`; existing dense subword
    `128x256b` rows exercise the same packed dense path at wider N.
  - `1B`: partial copy footprints, including row/column permutations,
    sub-instruction tile permutations, descriptor-view column slices, and any
    copy that wants only part of a public copy atom footprint. Decide whether
    each negative row decomposes into non-overlapping full atoms or needs an
    unavailable mask/smaller footprint/source format.
  - `1C`: `tcgen05.cp.4x256b` refresh views. Explicit refresh-shaped copies
    are positive for one-CTA and two-CTA, while ordinary contiguous `4x8`
    exposure and direct `ld/st` readback need a first-class refresh
    view/remap/load-store contract or should remain rejected.
  - `1D`: packed-lane `tcgen05.cp`, especially f16/bf16/i16/i8 no-scales
    `warpx2`. Direct packed copy appears ISA-limited because `tcgen05.cp`
    lacks the `ld/st` pack/unpack modifiers; staged compiler support may be
    possible through shared/register/TMEM `ld/st.pack` paths.
  - `1E`: no-scales `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b`.
    Single-CTA `02_13` and two-CTA `01_23` are positive. Two-CTA `02_13`
    still needs proof that a legal descriptor/address/source schedule can
    preserve the high source-column bit, or a final hardware/ISA limitation
    classification.
  - `1F`: frontend descriptor and copy-source contracts for
    `tcgen05_copy`, including explicit descriptor-view APIs and
    transposed/padded/noncanonical shared sources. Decide which cases should
    become explicit API contracts, which should rematerialize automatically,
    and which should stay rejected because semantics would be ambiguous or too
    expensive implicitly.
- Current classification: one active umbrella gap. Close Gap #1 only after the
  sub-buckets above are individually classified as supported, impossible, or
  deferred.

## Gap #2: Narrow Scaled-MMAv5 `N=8/16`

- Area: block-scaled MMAv5 accumulator layouts that require narrow N
  instruction fragments, especially `N=8` and `N=16`.
- Current state: current lowering reports a structured narrow-N scale-fragment
  requirement. The issue is not merely selecting an opcode; it requires a
  correct accumulator permutation plus matrix-B scale-fragment
  padding/rematerialization. Writes also need a safe physical accumulator
  footprint.
- Open question: can the backend physically pad/remap accumulator and scale
  storage to a supported `N>=32` tile while exposing a logical `N=8/16` view,
  or are packed adjacent accumulator layouts impossible without masks?
- Next evidence: design the padded physical accumulator/scale-fragment
  contract and prove whether writeback can avoid clobbering adjacent logical
  data.
- Initial classification: implementable with storage/schedule redesign for
  padded views; impossible for arbitrary tightly packed adjacent layouts
  without masks.

## Gap #3: Mixed fp4 TMEM LHS

- Area: block-scaled MMAv5 with mixed fp4 operand A in tensor memory, such as
  fp4 A with non-fp4 B under `mxf8f6f4`-style semantics.
- Current state: current TMEM LHS lowering treats A storage as raw packed
  columns. Mixed fp4 A needs the padded operand-A storage model represented by
  `fp4_padded` shared memory, where only part of each group carries real
  packed fp4 values and the rest are padding aliases.
- Open question: can TMEM LHS layouts gain a real `fp4_padded` storage
  contract equivalent to the shared-memory operand-A contract, including
  descriptor addressing, tile ordering, and view composition?
- Next evidence: define the TMEM `fp4_padded` layout semantics, then add
  compiler-only and runtime rows for tile-permuted and descriptor-view LHS
  cases.
- Initial classification: implementable backend/storage-model gap, not a
  known ISA impossibility.
