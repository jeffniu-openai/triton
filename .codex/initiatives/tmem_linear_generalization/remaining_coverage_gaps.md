# TMEM Remaining Coverage Gaps

Last updated: 2026-04-18 01:28 UTC

This is the stable reference list for the remaining TMEM coverage gaps that
need deeper discussion. Keep the numbering stable. If a gap is resolved,
classify it in place as `supported`, `impossible`, or `deferred`; do not
renumber later gaps. New discoveries should be appended as new gap numbers.

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
  current frontend/lowering path. Unsigned/per-operand signedness and
  saturation still need an explicit IR/frontend exposure decision.

## Gap #2: `tcgen05.cp` Supported-Layout Completeness

- Area: complete coverage of `tcgen05.cp` layouts modulo public instruction
  families and hardware restrictions.
- Current state: the planner recognizes the public copy families currently in
  use: dense `128x128b`, dense `128x256b`, refresh `4x256b`,
  `warpx2::01_23.64x128b`, `warpx2::02_13.64x128b`, and
  `warpx4.32x128b`. The runtime matrix covers many positive descriptor-view,
  indexed-view, subslice-view, one-CTA, two-CTA, dense-source, and
  rematerialized-source cases.
- Open question: do we have a complete enough compiler/test enumeration of all
  layout families that are realizable by public `tcgen05.cp`, or are there
  supported layouts not represented by the current matrix?
- Next evidence: build a coverage audit over recognized copy families,
  shape/CTA/source-layout categories, and current clean negatives; add
  compiler-only or runtime rows only where an actually realizable family is
  missing.
- Initial classification: likely mostly covered, but needs a formal coverage
  audit before closing.

## Gap #3: Partial `tcgen05.cp` Footprints

- Area: row/column permutations, sub-instruction tile permutations,
  descriptor-view column slices, and any copy that wants only part of a public
  copy atom footprint.
- Current state: current diagnostics report destination-mask,
  descriptor-row-split, source-row-split, column-permutation, or packed-lane
  requirements. Public `tcgen05.cp` writes full atom footprints, and there is
  no current destination row/column mask operand.
- Open question: are any currently negative partial-copy layouts decomposable
  into non-overlapping full public atoms, or do they fundamentally require
  unavailable masks/smaller footprints/source formats?
- Next evidence: for each representative partial row, prove either an exact
  non-overlapping atom decomposition or an unavoidable write-overlap/mask
  requirement.
- Initial classification: partially implementable for exact decompositions;
  otherwise likely public-ISA limited.

## Gap #4: `tcgen05.cp.4x256b` Refresh Views

- Area: `tcgen05.cp.4x256b` ordinary contiguous views, refresh-shaped layouts,
  and direct `ld/st` readback of refresh images.
- Current state: refresh-shaped `4x256b` copy is positive for one-CTA and
  two-CTA layouts. Ordinary contiguous `4x8` exposure and direct `ld/st`
  readback remain clean unsupported because the physical refresh image stores
  logical row/column bits in a non-ordinary layout.
- Open question: can the compiler expose a first-class refresh view/remap
  contract that makes ordinary user-facing copies and readback safe, or should
  ordinary contiguous `4x256b` remain impossible for direct copy/load/store?
- Next evidence: specify the logical-to-physical refresh image contract and
  determine whether readback can be implemented through supported packets or
  only through explicit rematerialization/staging.
- Initial classification: copy is supported only for explicit refresh layouts;
  broader support needs API/storage contract work.

## Gap #5: Packed `tcgen05.cp` and Subword `warpx2`

- Area: packed-lane `tcgen05.cp`, especially f16/bf16/i16/i8 `warpx2`
  no-scales copies whose logical N is narrower than the 128-bit public atom
  footprint.
- Current state: `tcgen05.ld/st` have pack/unpack support, but `tcgen05.cp`
  does not expose an analogous pack/unpack modifier. Current subword `warpx2`
  negatives report packed-lane source/destination storage requirements and
  insufficient logical column basis coverage.
- Open question: can the compiler support any of these through explicit
  packed-lane storage/rematerialization while preserving semantics, or is
  direct packed `tcgen05.cp` impossible under the public instruction surface?
- Next evidence: separate single-instruction `tcgen05.cp` impossibility from
  possible multi-step staging strategies through shared/register/TMEM
  `ld/st.pack` paths, including correctness and cost.
- Initial classification: direct packed `tcgen05.cp` is likely ISA-limited;
  staged compiler support may be possible for selected cases.

## Gap #6: Two-CTA `warpx2::02_13`

- Area: no-scales `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b`.
- Current state: single-CTA `02_13` is positive. Two-CTA `01_23` is positive.
  Two-CTA `02_13` currently reports a high source-column preservation
  requirement: known descriptor/direct-seed schedules either duplicate low
  source columns, fault, or read zeros.
- Open question: is there any legal `cta_group::2` descriptor/address/source
  schedule that preserves the high source-column bit for `02_13`, or should
  this be closed as a hardware/ISA schedule limitation?
- Next evidence: perform a focused schedule search and, where possible, use
  compile-only PTX checks plus GB200 runtime/probe evidence to validate or
  eliminate candidates.
- Initial classification: open research gap with strong current evidence for
  a hardware schedule limitation.

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

## Gap #9: Frontend Descriptor and Copy-Source Contracts

- Area: frontend block-layout tensor-memory descriptors, explicit descriptor
  view APIs, and `tcgen05_copy` source contracts for transposed/padded or
  otherwise noncanonical shared sources.
- Current state: `tcgen05_copy` requires a shared-memory descriptor source and
  tensor-memory descriptor destination, with only selected scales and `warpx2`
  source rematerialization paths. Block-layout TMEM descriptors and some copy
  source contracts still fail before they become backend support questions.
- Open question: which failures should become explicit API contracts, which
  should rematerialize through canonical shared/TMEM descriptors, and which
  should remain rejected because the semantics would be ambiguous or too
  expensive implicitly?
- Next evidence: design the frontend descriptor/view contract and decide where
  rematerialization is automatic versus opt-in.
- Initial classification: implementable API/lowering design gap.
