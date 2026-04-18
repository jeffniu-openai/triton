# TMEM Remaining Coverage Gaps

Last updated: 2026-04-18 06:54 UTC

This is the stable reference list for the remaining TMEM coverage gaps that
need deeper discussion. Keep the numbering stable. If a gap is resolved or
merged, classify it in place as `supported`, `impossible`, `deferred`, or
`merged`; do not renumber later gaps. New discoveries should be appended as
new gap numbers.

Each gap should be examined with the same decision standard:
- Is this impossible under the public PTX/ISA and current hardware behavior?
- Is it possible with compiler-only schedule/layout work?
- Is it possible only after an explicit frontend/API/storage contract change?
- What compile-only, FileCheck, PTX, PTXAS, or runtime evidence is needed to
  close the question?

## Gap #1: GB200 i8 MMAv5 Compile-Only Coverage

- Area: plain MMAv5 direct `tcgen05.mma.kind::i8`.
- Current state: signed i8 direct MMAv5 is validated on GB200 for the current
  compiler path. Local `sm_100` compilation emits `.target sm_100a`,
  `tcgen05.mma.cta_group::1.kind::i8`, descriptor immediate `136316064`, and a
  ptxas cubin (`97072` bytes). A `caas-gpu10`/`cudaberry-arm` container
  confirmed `NVIDIA GB200` compute capability `10.0`, loaded the generated
  cubin through the tiny torch/CUDA Driver shim, and produced an exact int32
  matmul match.
- Evidence:
  - four-GPU `nvidia-smi` probe showed four `NVIDIA GB200` devices with
    compute capability `10.0`;
  - one-GPU execution probe showed torch `device_count=1`, device
    `NVIDIA GB200`, capability `(10, 0)`;
  - `python3 run_i8_ptx_torch.py --launcher cpp` reported
    `PASS signed i8 tcgen05.mma sm100 PTX matches torch int32 matmul`.
- Boundary note: remote driver PTX JIT rejected the `.version 9.1` PTX with
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, so the runner loads the local ptxas
  cubin by default. The PTX remains the human/codegen inspection artifact.
  This is a deployment/toolkit compatibility boundary, not evidence that the
  compiler emits invalid i8 MMAv5 PTX.
- Current classification: signed i8 MMAv5 is supported on GB200 for the
  current frontend/lowering path. Runtime pytest coverage now includes a tiny
  sm100-gated signed-i8 shape set: `64x128x32`, `128x128x32`, and
  `128x256x64`, with exact int32 matmul checks and PTX/LLIR opcode checks.
  Unsigned/per-operand signedness and saturation still need an explicit
  IR/frontend exposure decision.

## Gap #2: `tcgen05.cp` Complete Support Umbrella

- Area: all remaining `tcgen05.cp` support and coverage questions after
  normalization to `LinearLayout`, including the formerly separate Gap #3
  partial-footprint work, Gap #4 `4x256b` refresh-view work, Gap #5
  packed/subword copy work, Gap #6 two-CTA `warpx2::02_13` work, and Gap #9
  copy-source/frontend contract work.
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
  - `2A`: optional evidence polish for dense subword exact-width `128x128b`
    rows, such as f16 `128x8` or i8 `128x16`; existing dense subword
    `128x256b` rows exercise the same packed dense path at wider N.
  - `2B`: partial copy footprints, including row/column permutations,
    sub-instruction tile permutations, descriptor-view column slices, and any
    copy that wants only part of a public copy atom footprint. Decide whether
    each negative row decomposes into non-overlapping full atoms or needs an
    unavailable mask/smaller footprint/source format.
  - `2C`: `tcgen05.cp.4x256b` refresh views. Explicit refresh-shaped copies
    are positive for one-CTA and two-CTA, while ordinary contiguous `4x8`
    exposure and direct `ld/st` readback need a first-class refresh
    view/remap/load-store contract or should remain rejected.
  - `2D`: packed-lane `tcgen05.cp`, especially f16/bf16/i16/i8 no-scales
    `warpx2`. Direct packed copy appears ISA-limited because `tcgen05.cp`
    lacks the `ld/st` pack/unpack modifiers; staged compiler support may be
    possible through shared/register/TMEM `ld/st.pack` paths.
  - `2E`: no-scales `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b`.
    Single-CTA `02_13` and two-CTA `01_23` are positive. Two-CTA `02_13`
    still needs proof that a legal descriptor/address/source schedule can
    preserve the high source-column bit, or a final hardware/ISA limitation
    classification.
  - `2F`: frontend descriptor and copy-source contracts for
    `tcgen05_copy`, including explicit descriptor-view APIs and
    transposed/padded/noncanonical shared sources. Decide which cases should
    become explicit API contracts, which should rematerialize automatically,
    and which should stay rejected because semantics would be ambiguous or too
    expensive implicitly.
- Current classification: one active umbrella gap. Close Gap #2 only after the
  sub-buckets above are individually classified as supported, impossible, or
  deferred.

## Gap #3: Merged Into Gap #2

- Former area: partial `tcgen05.cp` footprints.
- Current classification: merged. Use Gap #2 sub-bucket `2B`.

## Gap #4: Merged Into Gap #2

- Former area: `tcgen05.cp.4x256b` refresh views.
- Current classification: merged. Use Gap #2 sub-bucket `2C`.

## Gap #5: Merged Into Gap #2

- Former area: packed `tcgen05.cp` and subword `warpx2`.
- Current classification: merged. Use Gap #2 sub-bucket `2D`.

## Gap #6: Merged Into Gap #2

- Former area: two-CTA no-scales `tcgen05.cp.warpx2::02_13`.
- Current classification: merged. Use Gap #2 sub-bucket `2E`.

## Gap #7: Narrow Scaled-MMAv5 `N=8/16`

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

## Gap #8: Mixed fp4 TMEM LHS

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

## Gap #9: Merged Into Gap #2 For `tcgen05.cp`

- Former area: frontend descriptor and copy-source contracts, including
  `tcgen05_copy` source contracts for transposed/padded or otherwise
  noncanonical shared sources.
- Current classification: merged for all `tcgen05.cp`-related work. Use Gap #2
  sub-bucket `2F`.
- Boundary note: if a future non-copy frontend descriptor/API gap is identified,
  append a new gap with explicit non-`tcgen05.cp` scope instead of reusing
  Gap #9 ambiguously.
