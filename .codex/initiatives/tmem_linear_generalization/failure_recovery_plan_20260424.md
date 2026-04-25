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

Symptoms:
- descriptor multidim replay rows fail in `get_reg_layout(auto)` for rank-2 views.
- tensor-memory scales descriptor-view rows fail in `get_reg_layout(32x32b)`.
- clean-unsupported rows report generic unsupported text instead of the intended structural reason.

Likely root cause:
- `compute_tmem_reg_layout_from_memdesc` does not provide enough type-local candidate layouts for scales and descriptor views, and the unsupported-reason hooks are placeholders returning `None`.

Fix plan:
- Teach the frontend bridge to report real unsupported reasons from type-local ld/st query/planning failures.
- Ensure scales descriptor roots/views route through the scales-specific query/layout path before generic fallback.
- Keep frontend selection type/local; no producer-chain walking.

### 2. Subword ld/st dynamic and loop-carried views

Symptoms:
- wide-opcode expectations see scalar `x1` RMW sequences.
- dynamic f16 subword descriptor views miscompare.

Likely root cause:
- subword phase/alignment for selected memdesc SSA values is conservative or wrong: static type/layout proves candidate atom coverage, but dynamic selected `taddr` can carry subword phase. Lowering scalarizes or packs/unpacks using an incomplete current-address model.

Fix plan:
- Recheck the current `taddr` plus element-column/subword phase model for dynamic selected descriptors.
- For phases not representable by a wide atom, either emit correct scalar/subword RMW and update opcode expectations, or reject cleanly if correct lowering is impossible. Runtime miscompares must be fixed first.

### 3. Multidim descriptor ld/st and half-row views

Status: half-row row-origin sub-bucket closed on 2026-04-25; remaining subword dynamic ld/st is tracked separately in bucket 2.

Symptoms:
- stale `tt.reshape` string expectations.
- half-row direct/lifted descriptor views had 50% miscompares.

Root cause found:
- `memdesc_subslice` preserved allocation context, but `memdesc_index` erased it when indexing a leading TMEM dimension that had already been narrowed by a subview. The final rank-2 descriptor looked like a plain origin-zero view, so type-local ld/st planning could not know that the current SSA `taddr` selected the second row half.

Fix executed:
- `memdesc_index` now folds the omitted leading allocation extent into the result row allocation for narrowed TMEM views.
- Direct row-slice ld/st without a layout-encoded row support bit is a clean unsupported case from the current memdesc type; frontend, verifier/planner, and late lowering share this predicate.
- Stale `tt.reshape` string assertions were removed while keeping runtime correctness and IR feature checks.

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

Symptoms:
- dense 256-row direct-root positives now clean-unsupported.
- warpx2 descriptor/subslice/indexed views and tile-permuted linear layouts miscompare.

Likely root cause:
- direct-root support-query fallback was lost or over-tightened; the type-local exact query is correct for views but insufficient for roots that require a wider support query. For tile-permuted/warpx2 views, copy planning again uses canonical copy family order for destination offsets.

Fix plan:
- Restore support-query fallback for direct roots only.
- Keep active descriptor views on exact type-local queries.
- For tile-permuted/warpx2 views, compute destination offsets from the exact current descriptor layout while using canonical family layouts only to select atom family/schedule.

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

1. Restore direct-root copy support-query fallback and validate dense direct-root copy rows.
2. Fix/frontend type-local layout selection for scales and descriptor views.
3. Fix ld/st exact row/address planning: half-row row-origin closed; subword dynamic views remain open under bucket 2.
4. Fix ld.red exact-layout reduction planning.
5. Fix copy tile-permuted/warpx2 exact destination offsets.
6. Fix TMEM-LHS MMAv5/scaled-MMAv5 exact address/layout handling.
7. Re-run focused bucket selectors after each slice, then the 4-GPU broad split.
