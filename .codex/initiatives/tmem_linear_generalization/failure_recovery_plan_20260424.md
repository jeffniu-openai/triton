# TMEM Broad Failure Recovery Plan - 2026-04-24

## Validation Baseline

Current head when this plan was written: `6d16268f8`. Broad validation is not green: `184 failed, 12771 passed, 6913 skipped` across the 4-GPU split over `test_core.py`, `test_tmem_runtime_matrix.py`, and `test_tmem_structural_fuzzer.py`.

## Root Design Weakness

The common weakness across most failures is that several TMEM surfaces still mix two different concepts:

- **exact current memdesc layout/address image**: the physical row/element-column mapping represented by the current memdesc type plus current SSA `taddr`; this is the only valid source for codegen addresses and user-visible semantics.
- **support/family/planning layout**: a canonical shape used to decide whether an ISA atom family can cover a layout; this is useful for planning but is not necessarily the physical layout of the current descriptor.

The MMAv5 accumulator bug fixed in `6a50032c3` was one instance of this. The remaining failures show the same issue in ld/st, ld.red, copy, scales, and TMEM-LHS MMA. A second weakness is that the frontend `get_reg_layout()` uses the same planning helpers as lowering, so missing type-local planning support surfaces as generic frontend unsupported errors before lowering can make a more precise decision.

## Failure Buckets and Planned Fixes

### 1. Frontend/type-local ld/st layout selection

Status: tensor-memory-scales sub-bucket closed on 2026-04-25; remaining non-scales descriptor-view frontend failures are tracked in buckets 2, 3, and 6 according to their owning backend surface.

Symptoms:
- descriptor multidim replay rows fail in `get_reg_layout(auto)` for rank-2 views.
- tensor-memory scales descriptor-view rows fail in `get_reg_layout(32x32b)`.
- clean-unsupported rows report generic unsupported text instead of the intended structural reason.

Likely root cause:
- `compute_tmem_reg_layout_from_memdesc` does not provide enough type-local candidate layouts for scales and descriptor views, and the unsupported-reason hooks are placeholders returning `None`.

Fix executed for scales:
- `32x32b` requests now accept the packed `16x32bx2` realization for scales roots and type-local scales descriptor views when the current type/layout plans that atom.
- Frontend unsupported reasons now report variant-specific scales packet-footprint requirements and the current M=64 two-CTA clean boundary.
- Tests document the verified current-layout lowering where roots use packed stores/loads and descriptor views may use direct `32x32b`; runtime correctness is checked before opcode assertions.

Remaining plan:
- Address non-scales descriptor-view `get_reg_layout` failures in the owning ld/st and MMAv5 buckets.
- Keep frontend selection type/local; no producer-chain walking.

### 2. Subword ld/st dynamic and loop-carried views

Status: closed on 2026-04-25 for current runtime-matrix subword ld/st rows.

Symptoms:
- wide-opcode expectations see scalar `x1` RMW sequences.
- dynamic f16 subword descriptor views miscompare.

Likely root cause:
- subword phase/alignment for selected memdesc SSA values is conservative or wrong: static type/layout proves candidate atom coverage, but dynamic selected `taddr` can carry subword phase. Lowering scalarizes or packs/unpacks using an incomplete current-address model.

Fix executed:
- Immediate `MemDescIndexOp` SSA values now contribute to subword phase proof, so dynamic/static indexed-away dimensions that can alter the runtime `taddr` element-column phase force the unknown-phase path instead of incorrectly proving wide-op alignment from result shape alone.
- Unknown-phase layouts with zero column bases use elementwise 32-bit RMW through the exact type-local query layout. Contiguous packed unknown-phase layouts continue to use the wide realignment path. Non-contiguous/reversed packed phase layouts report a clean structural diagnostic.
- Opcode expectations now require scalar `32x32b.x1` RMW when phase is unknown; runtime correctness remains the primary assertion.

### 3. Multidim descriptor ld/st and half-row views

Status: closed on 2026-04-25 for current non-red ld/st runtime-matrix coverage; subword dynamic ld/st is closed separately in bucket 2.

Symptoms:
- stale `tt.reshape` string expectations.
- half-row direct/lifted descriptor views had 50% miscompares.

Root cause found:
- `memdesc_subslice` preserved allocation context, but `memdesc_index` erased it when indexing a leading TMEM dimension that had already been narrowed by a subview. The final rank-2 descriptor looked like a plain origin-zero view, so type-local ld/st planning could not know that the current SSA `taddr` selected the second row half.

Fix executed:
- `memdesc_index` now preserves omitted leading allocation extent in the result row or column dimension according to the current source type/layout physical offset. Row-moving narrowed views remain clean unsupported when the result type cannot encode a direct row footprint; column-moving narrowed views stay runtime-positive.
- Direct row-slice ld/st without a layout-encoded row support bit is a clean unsupported case from the current memdesc type; frontend, verifier/planner, and late lowering share this predicate.
- Stale IR string assertions were removed or refreshed while keeping runtime correctness and TMEM operation checks.
- Frontend clean-negative reasons now cover atom footprint/exposed-column mismatches and `tcgen05.copy.4x256b` refresh-image ld/st boundaries.

Validation:
- `make -j8`
- `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python pytest -q -s --tb=short -k 'half_rows' python/test/gluon/test_tmem_runtime_matrix.py` -> `20 passed, 1691 deselected`.

### 4. `ld.red` permuted/tile-permuted layouts

Symptoms:
- compiler lowering failures for unsupported destination layouts.
- opcode-count mismatches.
- 50% output miscompares for tile-permuted and descriptor-chain red-load rows.

Likely root cause:
- `ld.red` planning chooses a valid reduction atom from a support layout but writes/loads through the canonical atom order instead of the exact current memdesc physical order. Some M64 row plans also choose too many split-N messages after family normalization.

Fix plan:
- Split reduction atom selection from exact address/tile ordering, mirroring the MMAv5 accumulator fix.
- Ensure red-load output register layout and reduced-value layout are derived from the exact current layout plus selected atom, not from a canonical family layout alone.

### 5. `tcgen05.copy` direct-root and view planning

Status: closed on 2026-04-25 for the current runtime-matrix copy bucket.

Symptoms:
- dense 256-row direct-root positives now clean-unsupported.
- warpx2 descriptor/subslice/indexed views and tile-permuted linear layouts miscompare.

Root causes found:
- Direct-root expanded-row layouts needed the same type-derived folded physical query that ld/st already use; trying to cover a larger logical row extent by looping copy instructions is wrong because copy atom row semantics include folded/broadcast row bits.
- Some direct-root tile-permuted layouts needed a source-support query with canonicalized column bases to synthesize the shared descriptor, but destination offsets still had to come from the exact current destination layout.
- Dense/tile-permuted destination offset planning normalized away exact layout ordering, so physical tile offsets could be computed in the wrong column order.
- The single-CTA `warpx2::02_13` direct-seed plan added an extra TMEM dword destination delta; the source seed offset was already carrying the required source-column shift.
- The attempted logical-row scheduler treated `atom.nRow` as a loop stride over the whole logical matrix. That broke `4x256b` refresh and warpx2 because their descriptor source coordinate space is atom-local and logical row bits may be folded/broadcast by the layout.
- The late subword `tcgen05.copy` destination-origin rejection was a stale compile-time alignment check. Runtime-value alignment belongs to iisan, not verifier/lowering.

Fix executed:
- Direct roots may choose a folded root physical query when type/layout algebra proves the copy conversion, and direct roots may separately choose a canonical source-support query while preserving exact destination addressing.
- `selectTMemCopyPlan` and lowering now use exact destination queries for executable schedule and base-offset codegen, while descriptor conversion can use the support query.
- Dense destination tile coordinates use exact layout arithmetic instead of normalization.
- `warpx2::02_13` single-CTA direct-seed destination dword delta is zero.
- Removed the logical-row copy scheduler; expanded rows are represented through the selected type-local physical query or rejected cleanly.
- Removed the late subword destination-origin compile rejection and updated tests to require non-iisan compilation plus iisan assertions for static-subview and loop-carried unaligned destinations.

Validation:
- `make -j8`
- Focused refresh/warpx2 selector -> `14 passed, 1697 deselected`.
- Full warpx2 selector -> `83 passed, 1628 deselected`.
- Representative dense 256-row direct-root row -> `1 passed`.
- Subword/iisan copy selector -> `6 passed, 1706 deselected`.
- Full `cp_no_scales` selector -> `301 passed, 4 skipped, 1407 deselected`.

### 6. MMAv5 / scaled-MMAv5 TMEM-LHS and descriptor-subview paths

Symptoms:
- TMEM-LHS tile-permuted plain and scaled rows miscompare.
- scaled accumulator subslice tile-permuted descriptor views fail in `get_reg_layout(auto)`.

Likely root cause:
- The MMAv5 accumulator address path was fixed, but TMEM-LHS and scaled subview frontend planning still use family layout information as if it were exact physical order.

Fix plan:
- Apply the same exact-layout/address split to TMEM-LHS load operands.
- Extend frontend `get_reg_layout` for MMAv5-compatible descriptor subviews using exact current layout.

## Execution Order

1. [done] Restore direct-root copy support-query fallback and validate dense direct-root copy rows.
2. [done for scales] Fix/frontend type-local layout selection for tensor-memory-scales roots/views; continue non-scales descriptor-view failures in their owning buckets.
3. [done] Fix ld/st exact row/address planning: half-row row-origin, subword dynamic views, and remaining non-red descriptor-view rows are closed for current coverage.
4. Fix ld.red exact-layout reduction planning.
5. [done] Fix copy tile-permuted/warpx2 exact destination offsets.
6. Fix TMEM-LHS MMAv5/scaled-MMAv5 exact address/layout handling.
7. Re-run focused bucket selectors after each slice, then the 4-GPU broad split.
