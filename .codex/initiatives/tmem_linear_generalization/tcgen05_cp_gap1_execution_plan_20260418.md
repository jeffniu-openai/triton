# Gap #1 `tcgen05.cp` Execution Plan

Last updated: 2026-04-18 19:46 UTC

## Goal

Resolve the current Gap #1 `tcgen05.cp` umbrella after the generic
linear-layout planner refactor. A sub-bucket is complete only when it is
positive with focused coverage, proved impossible under the public
instruction/storage model, or explicitly deferred behind a frontend/API/storage
contract with clean diagnostics.

The key invariant is that none of this work should reintroduce legacy-vs-linear
or scales-vs-no-scales backend modes. Support and negatives are classified by
normalized `LinearLayout`, public copy-family footprint, CTA ownership, packed
lane/storage semantics, and explicit frontend contracts.

## Phase 1: Evidence Closure For `1A`

- Status: complete.
- Scope: add one tiny positive dense subword exact-width `128x128b` runtime
  row, preferably explicit `TensorMemoryLinearLayout` f16 `128x8`.
- Artifact: focused pytest row proving `tcgen05.cp.cta_group::1.128x128b` on a
  sub-32-bit exact-width dense layout.
- Closure criterion: runtime output matches input and PTX/LLIR contain exactly
  one `128x128b` copy.
- Result: added f16 `128x8` and i8 `128x16` exact-width rows through
  `test_tmem_runtime_matrix_cp_128x128_subword_exact_width`; both pass and
  check one `tcgen05.cp.cta_group::1.128x128b` in PTX/LLIR.

## Phase 2: Bounded Probe For `1E`

- Status: pending.
- Scope: determine whether
  `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b` can preserve the high
  source-column bit with any legal descriptor seed/source-offset schedule.
- Artifact: a small probe report in this folder, backed by either generated
  compiler probes or hand-written PTX/PTXAS/runtime evidence.
- Closure criterion:
  - supported if a schedule produces correct output for candidate, indexed, and
    subslice forms without `cta_group::1` fallbacks; or
  - impossible if exhaustive practical seed/source/destination offset probes
    show only low-column duplication, zero reads, or illegal/faulting
    subaligned deltas.

## Phase 3: `1B` Partial-Footprint Decomposition Audit

- Status: pending.
- Scope: classify row/column permutations, sub-instruction tile permutations,
  scales descriptor-view row schedules, noncanonical two-CTA block ownership,
  and mixed row/column basis negatives.
- Artifact: update the Gap #1 audit with a table of each representative
  negative class and one of:
  - decomposable into non-overlapping full public copy atoms;
  - impossible without masks or smaller public copy footprints;
  - deferred behind source-message/destination-row schedule design.
- Closure criterion: no residual "generic layout unsupported" bucket remains.

## Phase 4: `1C` Refresh-Image Contract Decision

- Status: pending.
- Scope: decide whether ordinary contiguous `4x8` views and direct `ld/st`
  readback of `tcgen05.cp.4x256b` refresh images should be implemented through
  an explicit refresh image view/remap/load-store contract or remain clean
  unsupported.
- Artifact: update the audit and gap register with the selected contract.
- Closure criterion: either a concrete implementation plan exists with API
  shape and validation rows, or the current clean negatives are marked final
  because ordinary views misrepresent the sparse refresh image.

## Phase 5: `1D` Packed-Lane Storage Design

- Status: pending.
- Scope: decide whether subword no-scales `warpx2` copy can be supported by
  staged compiler movement through shared/register/TMEM `ld/st.pack` paths, or
  whether direct `tcgen05.cp` remains impossible because copy lacks pack/unpack
  modifiers.
- Artifact: packed-lane design note or impossibility note, with at least one
  representative probe or PTX/manual ISA argument.
- Closure criterion: either implement one safe staged positive slice, or mark
  direct support impossible and staged support deferred behind a storage-model
  design.

## Phase 6: `1F` Frontend/API Contract Split

- Status: pending.
- Scope: split frontend descriptor/copy-source rows into:
  - explicit API contracts to add;
  - automatic rematerialization paths that are semantically safe;
  - permanent clean errors for ambiguous or too-expensive implicit behavior.
- Artifact: updated audit table plus targeted tests or stable clean-error
  diagnostics for each class.
- Closure criterion: no frontend/source-contract row remains unclassified.

## Validation Strategy

- Run `make` before tests.
- Use focused pytest nodeids while developing each sub-bucket.
- For broad copy regressions, use four independent GPU shards with stable
  per-GPU `TRITON_CACHE_DIR` values.
- Lit/FileCheck are unavailable in the current shell; use direct `triton-opt`
  pass-pipeline checks unless the environment is repaired.

## Current Execution Cursor

- Start from pushed commit `4f7dd8671`.
- Execute Phase 2 next.
- After every completed phase, update this file, `remaining_coverage_gaps.md`,
  `tcgen05_cp_gap2_audit_20260418.md`, `memory.md`, `log.md`, and the handoff
  tail as needed, then checkpoint commit and push.
