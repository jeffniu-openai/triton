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

Status: closed on 2026-04-25 for current runtime-matrix `ld_red` coverage.

Symptoms:
- compiler lowering failures for unsupported destination layouts.
- opcode-count mismatches.
- 50% output miscompares for tile-permuted and descriptor-chain red-load rows.

Root causes found:
- `tcgen05.ld.red` data payload ordering is not identical to normal `tcgen05.ld` for all atom families. The I32 red payload is already in the requested per-message register order, but multi-message exact-layout lowering still has to assemble chunks by exact physical packet offset. The M64 `16x32bx2` split-N path still needs the normal column action because its second-half operand interleaves logical columns inside each message. Treating all red payloads like normal loads corrupted tile-permuted I32 rows; treating all red payloads as already ordered corrupted M64 col-reverse rows.
- Gluon's builder performed a late `canonicalizeTMemLoadReductionType` rewrite for every red load result type. That canonicalized explicit/current M64 row-permuted layouts after frontend layout selection, reintroducing support-layout semantics into codegen and causing `ConvertTritonGPUToLLVM` failures.
- Several opcode expectations encoded the old canonical support-layout packet schedule rather than the exact-layout packet schedule required for correct lowering. Runtime correctness was preserved after refreshing them.

Fix executed:
- Red-load lowering now assembles data chunks separately from reduction scalars. I32 red chunks are sorted by exact packet offset before packing the result, while M64 `16x32bx2` chunks retain planner order and then apply the column action needed for second-half interleaving.
- Removed the late Gluon builder result-type canonicalization; red-load result layouts now remain the layout selected by the frontend/type-local planner or explicitly supplied by the caller.
- Refreshed M64 and tile-permuted opcode expectations to the exact physical packet schedule after runtime assertions passed.

Validation:
- `make -j8`
- `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python pytest -q -s --tb=short -k 'ld_red' python/test/gluon/test_tmem_runtime_matrix.py` -> `264 passed, 1448 deselected`.

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

Status: closed on 2026-04-25 for the current MMAv5/scaled runtime-matrix selector, with two documented clean unsupported boundaries.

Symptoms:
- TMEM-LHS tile-permuted plain and scaled rows miscompared.
- scaled accumulator subslice tile-permuted descriptor views failed in `get_reg_layout(auto)`.

Root causes found:
- The TMEM-LHS verifier accepted layouts that preserved canonical column order only for a sub-tile smaller than the hardware instruction K tile. `tcgen05.mma` has no TMEM-A descriptor stride or permutation operand: it reads one fixed physical K tile from the TMEM-A base. When the logical K tile is permuted inside that physical tile, A is consumed in physical order while B remains logical, so direct codegen would miscompile. This is a true ISA/codegen boundary unless the program explicitly repacks/copies A into a compatible layout.
- `memdesc_subslice` result-type inference for pure 2D column subviews over tile-permuted layouts had a narrowed fast path that forced the physical output column span to the logical view width. A half-tile accumulator view such as `128x64` from a `128x128` tile-permuted parent actually maps logical columns into a wider physical image. The old type was not self-contained enough for type-local planning.
- After the type inference fix, direct `ld/st` of that sparse physical accumulator subview is still unsupported by the current register tensor abstraction. The sliced descriptor can be a valid MMA D address, but a `128x64` register tensor cannot directly name physical columns such as `{0..31,64..95}` relative to its current `taddr`. That is recorded as a clean frontend unsupported case instead of replaying the parent view chain.

Fix executed:
- Plain and scaled MMAv5 verification now rejects TMEM-LHS layouts whose preserved canonical K span is smaller than the instruction's required storage K tile, with a diagnostic explaining that in-tile column permutations cannot be represented by the TMEM-A address operand.
- The `memdesc_subslice` query/type path now distinguishes simple column slices whose retained bases fit a narrowed physical span from sparse/tile-permuted column slices that need a wider self-contained physical output image. The latter falls through to exact inverse/projection arithmetic instead of losing physical columns.
- Runtime-matrix rows were split into positives and clean negatives. Valid larger K tiles and mxfp8 scaled TMEM-LHS rows still execute and check numerical correctness/opcodes. Plain f16/f8 and scaled mxfp4/nvfp4 in-tile K permutations now assert clean unsupported diagnostics. Sparse scaled-accumulator subview `ld/st` rows assert the current direct-layout unsupported diagnostic rather than a compiler crash or silent replay.

Validation:
- `make -j8`
- Focused plain TMEM-LHS selector -> `8 passed, 1700 deselected`.
- Focused scaled TMEM-LHS selector -> `18 passed, 1690 deselected`.
- Focused scaled accumulator sparse-subview selector -> `10 passed, 1698 deselected`.
- Full MMAv5/scaled selector `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python pytest -q -s --tb=short -k 'mma and (tile_permuted or scaled)' python/test/gluon/test_tmem_runtime_matrix.py` -> `343 passed, 1365 deselected`.

## Execution Order

1. [done] Restore direct-root copy support-query fallback and validate dense direct-root copy rows.
2. [done for scales] Fix/frontend type-local layout selection for tensor-memory-scales roots/views; continue non-scales descriptor-view failures in their owning buckets.
3. [done] Fix ld/st exact row/address planning: half-row row-origin, subword dynamic views, and remaining non-red descriptor-view rows are closed for current coverage.
4. [done] Fix ld.red exact-layout reduction planning.
5. [done] Fix copy tile-permuted/warpx2 exact destination offsets.
6. [done] Fix TMEM-LHS MMAv5/scaled-MMAv5 exact address/layout handling, including clean unsupported boundaries for in-tile K permutations and sparse physical accumulator subview `ld/st`.
7. Re-run focused bucket selectors after each slice, then the 4-GPU broad split. Focused recovery buckets are now closed; the next validation step is expectation refresh for structural-fuzzer sentinels, then broad 4-GPU validation.
