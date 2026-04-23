# TMEM Completion Execution Tracker

Last updated: 2026-04-23 03:45 UTC

Active phase: newer TMEM memdesc model implementation, first vertical slices.

Active implementation checklist:

- [x] Rehydrate the initiative and record the newer memdesc-model plan in the
  durable design note.
- [x] Audit chain-dependent TMEM planning/lowering helpers and classify each as
  semantic-to-rewrite, optimizer-only, or allocation/effect-only.
- [x] Prototype physical-element-column `taddr` lowering and prove unnecessary
  `subword_index` arithmetic is removed for f32/aligned cases.
- [x] Add type-local planner scaffolding that can compare against the current
  chain-dependent planner without changing codegen.
- [ ] Fix view result type/layout computation and source-type-only `taddr`
  updates for origin-changing views. First completed slice: TMEM
  `memdesc_subslice`/`ttng.tmem_subslice` now prefer active
  descriptor-relative result encodings before preserving parent encodings.
- [ ] Migrate ld/st, subword packing, `ld.red`, copy, and MMAv5/scales in
  vertical slices with focused tests after each slice. First ld/st-facing
  completed slice: active self-contained subview layouts now try their
  standalone canonical load/store surrogate before raw parent-allocation query
  types, using a type-local shape-vs-alloc/layout predicate. First ld/st
  row-plan slice: active self-contained subviews now use the current
  descriptor's type-local row plan instead of backing parent row plans. First
  ld/st raw-query slice: active self-contained subviews now dispatch through
  `inferTypeLocalTMemLdStQueryLayout(MemDescType)`. First ld/st query-type
  slice: active self-contained subviews now return query types from
  `getTypeLocalTMemLdStQueryTypes(MemDescType)`. First ld/st support-query
  slice: active self-contained subviews now return support-query plans from
  `getTypeLocalTMemLdStSupportQueryPlan(MemDescType)`. First fallback-tightening
  slice: active self-contained raw-query, direct-support, and reduction row-plan
  paths no longer borrow backing row plans when a type-local row plan is absent.
  First LLVM lowering slice: active self-contained descriptors no longer use
  standalone type reconstruction, source-column support rescue, or
  already-adjusted base-offset correction in `lowerTMemLdStFromTypes`, and
  type-local copy queries no longer subtract chain-derived view offsets.
  First verifier-locality slice: generic ld/st verification skips standalone
  physical-support rescue for active self-contained descriptors, and ld.red
  verification no longer uses backing-row fallback for active support/raw
  row-plan checks. First MMAv5/scales-facing active physical-support slice:
  pure 2D column subview inference now avoids compacting a view when its
  selected physical support image is wider than the logical slice, preventing
  tile-permuted accumulator subviews from collapsing their high column basis to
  zero. MMAv5 active subview address and tile-order lowering use current
  descriptor type/layout facts for self-contained descriptors. Compact active
  layouts still use the current-`taddr` planning type with `allocShape` reset
  to the logical shape; wider exact support layouts keep their support shape.
  First MMAv5 family-address slice: narrowed MMAv5-family descriptors now
  derive address layouts and tile-order offsets from current `MemDescType`
  family facts before legacy producer-chain fallback. First MMAv5 helper-split
  slice: the type-local address and tile-order computations are now explicit
  APIs, with value-taking wrappers retaining legacy fallback. First B-scale
  descriptor-view storage slice: generated padded and unpadded B-scale storage
  layouts can be classified from the current `MemDescType`, so scaled-MMAv5
  legality no longer requires walking B-scale descriptor-view producers when a
  same-typed descriptor is selected dynamically. First B-scale
  rematerialization slice: selected unpadded B-scale descriptor views are
  rematerialized branch-wise to padded scales storage before scaled MMA, with
  cleanup when the original select is single-use. First general scale
  descriptor-view slice: generated A/B scale descriptor-view storage is
  classified from current type/layout facts, and selected scale descriptor
  views rematerialize branch-wise for scaled MMA. First scales ld/st
  descriptor-view slice: generated A/B scales descriptor views now use their
  type-local recovered `tensor_memory_scales` storage type for load/store
  query-type rescue and LLVM planning, preserving the current selected runtime
  `taddr` and avoiding unsafe canonical dense-linear surrogate lowering.
  First copy-planning slice:
  `selectTMemCopyPhysicalQuery` now selects the
  type-local destination physical query for active self-contained subviews,
  keeping legacy standalone/exact selection for direct roots and older
  parent-encoding views. First clean-negative copy slice: too-narrow current
  descriptors such as `128x1xf32` and `128x2xf32` now report an explicit
  hardware copy-atom boundary.
- [ ] Split helper APIs so semantic lowering/verifiers use type-local helpers
  and producer-chain matchers are optimizer-only.
- [ ] Run staged lit, focused pytest, 4-GPU runtime matrix, structural fuzzer,
  and example performance checks before considering the migration complete.

Current invariant for this phase: user-facing TMEM APIs do not change. The only
planned semantic tightening is clean rejection of `tcgen05.copy` descriptors
that are too small for any legal ISA atom, such as dense `128x1xf32` or
`128x2xf32`, rather than borrowing hidden parent footprint.

Current implementation checkpoint: active subviews now have two type-local
planning modes. Compact active layouts, whose current layout materializes the
logical shape exactly, plan with `allocShape == shape` because the lowered
base is already the current `taddr`. Exact physical-support layouts, whose
current layout needs a wider support image than the logical slice, keep that
support shape and avoid canonical-surrogate query-type invention. The
tile-permuted scaled-MMAv5 accumulator subslice crash is fixed. MMAv5 address
layout and tile-order offset lowering now derive the family layout from the
current narrowed descriptor type when the current type still carries the
MMAv5-family allocation shape. This is deliberately narrower than direct ld/st:
a raw tile-permuted current layout is not a valid direct support query by
itself, so ld/st support planning still uses the canonical family support
layout. Remaining work continues with broader type-local MMAv5/scales cleanup,
`ld.red` cleanup, and helper API separation.

Current API-separation checkpoint: MMAv5 address and tile-order planning now
has explicit type-local entry points:
`getTypeLocalMMAv5TMemAddressLayout(MemDescType)` and
`getTypeLocalMMAv5TMemViewOffsetForLowering(MemDescType, offsets)`. The old
value-taking helpers still preserve physical-bitcast behavior and legacy
producer-chain fallback, but the semantic type-local path is now callable and
auditable independently.

Current B-scale checkpoint: scaled-MMAv5 B-scale storage classification now has
a type-local entry point, `getMMAv5ScaledBScaleStorageType(MemDescType)`.
Generated B-scale descriptor views that are represented as
`tensor_memory_linear` can be recognized from their current layout alone, which
keeps dynamic same-typed selected views legal without relying on producer-chain
root recovery. Legacy value-taking chain recovery remains as fallback for
descriptor classes not migrated yet.

Current B-scale rematerialization checkpoint: the B-scale fragment
rematerializer now splits through an `arith.select` result. Each selected
unpadded branch is rematerialized from its store/view chain into padded
`tensor_memory_scales` storage and a new selected padded descriptor is created
for the MMA. When the original select is single-use, the obsolete unpadded
select/store/view chains are cleaned up once the MMA operand is rewritten;
multi-use originals remain available to unrelated consumers.

Current general scale descriptor-view checkpoint:
`getMMAv5ScaleStorageType(MemDescType)` recognizes generated unpadded A/B scale
descriptor-view storage from the current memdesc type/layout. The general scale
descriptor-view rematerializer uses that type-local classification before
legacy root walking and can split selected descriptors branch-wise, creating a
rematerialized selected scales descriptor for scaled MMA while preserving
multi-use originals for unrelated consumers.

Current scales ld/st descriptor-view checkpoint:
dynamic selected scales descriptor views now lower from type-local facts.
`getTMemLdStQueryTypes(Value)` recognizes generated scales descriptor-view
storage from the current `MemDescType` and returns the recovered
`tensor_memory_scales` storage type before the original linear view type, rather
than inventing a canonical dense-linear surrogate. LLVM ld/st lowering also
uses the recovered storage type as the semantic planning type for raw/support
attempts. The selected descriptor's runtime `taddr` still carries the dynamic
base; only the static storage semantics change. This fixed the 1CTA
row-permuted selected scales view miscompile and kept the 2CTA row green.

Completed twelfth implementation slice: MMAv5 family address/tile-order
lowering is now type-local for narrowed MMAv5-family descriptors.
`getMMAv5TMemFamilyAddressLayout(MemDescType)` no longer rejects
`shape != allocShape`, allowing the existing MMAv5 family-info helpers to use
the current descriptor's allocation shape for row-preserving N-narrowed
subviews. `getMMAv5TMemAddressLayout` and
`getMMAv5TMemViewOffsetForLowering` consult that type-local family layout
before legacy producer-chain inference. Validation: required `make -j8`; full
scaled tile-permuted accumulator subslice function `10 passed`; exact two-CTA
scaled subslice row `1 passed`; 4-GPU positive MMAv5 selector passed as group1
`133 passed, 14 skipped`, group2 `147 passed`, group3 `147 passed`, group4
`147 passed`; 4-GPU combined ld/st+ld.red+copy selector passed as group1
`120 passed, 28 skipped`, group2 `98 passed, 50 skipped`, group3
`128 passed, 20 skipped`, group4 `146 passed`; targeted lit set passed `6/6`;
`git diff --check` passed.

Completed thirteenth implementation slice: MMAv5 address/tile-order helper
surface split. `getExactTypeTMemAddressLayout` centralizes exact current-type
layout extraction. `getTypeLocalMMAv5TMemAddressLayout` returns the local plan
for active self-contained descriptors, MMAv5-family descriptors, and direct
full-shape descriptors. `getTypeLocalMMAv5TMemViewOffsetForLowering` computes
the matching local tile-order offset. Validation: required `make -j8`; full
scaled tile-permuted accumulator subslice function `10 passed`; focused
B-scale descriptor-view rows `3 passed`; 4-GPU positive MMAv5 selector passed
as group1 `133 passed, 14 skipped`, group2 `147 passed`, group3 `147 passed`,
group4 `147 passed`; targeted lit set passed `6/6`; `git diff --check`
passed.

Completed fourteenth implementation slice: MMAv5 B-scale descriptor-view
storage classification is type-local. `getMMAv5ScaledBScaleStorageType` accepts
current `tensor_memory_scales` descriptors and conservatively recognizes the
generated padded and unpadded B-scale descriptor-view storage layouts from a
rank-2 8-bit `tensor_memory_linear` `MemDescType`. The existing
`getMMAv5ScaledBScaleStorageTypeThroughViews` wrapper now calls the type-local
helper before the legacy reshape/trans/index root-walk. New runtime coverage
selects between two same-typed padded B-scale descriptor views through dynamic
control flow before scaled MMA. Validation: required `make -j8`; focused
`bscale_descriptor_view` selector `5 passed, 1621 deselected`; targeted lit set
previously passed `6/6`; 4-GPU positive MMAv5 selector previously passed as
group1 `134 passed, 14 skipped`, group2 `148 passed`, group3 `148 passed`,
group4 `145 passed`; `git diff --check` passed. Remaining boundary:
unpadded dynamic selected B-scale views still need an allocation/rematerialized
storage transform repair before they are covered as runtime positives.

Completed fifteenth implementation slice: unpadded dynamic selected B-scale
descriptor views now rematerialize correctly. `RematerializeScaledMmaBScaleFragments`
factors direct branch rematerialization from top-level MMA rewriting and handles
an `arith.select` by rematerializing true and false branches to the required
padded scales shape and creating a new select over the rematerialized
descriptors for scaled MMA. Single-use originals are erased and cleaned up;
multi-use originals remain available to unrelated consumers. The dynamic
B-scale descriptor-view test now parameterizes both padded and unpadded
storage. Validation: required `make -j8`; exact unpadded dynamic
selected row `1 passed`; padded+unpadded dynamic rows `2 passed`; focused
`bscale_descriptor_view` selector `6 passed, 1621 deselected`; targeted lit set
`6/6`; 4-GPU positive MMAv5 selector passed as group1
`134 passed, 14 skipped`, group2 `148 passed`, group3 `148 passed`, group4
`146 passed`; `git diff --check` passed.

Completed sixteenth implementation slice: general A/B scale descriptor-view
storage and selected rematerialization are type-local. Added
`getMMAv5ScaleStorageType(MemDescType)` and factored the common unpadded
interleaved scale descriptor-view layout classifier out of the B-scale-specific
classifier. `getMMAv5ScaleStorageTypeThroughViews(Value)` now tries the
type-local helper before legacy producer-chain fallback. The general scale
descriptor-view rematerializer now handles selected descriptors by
rematerializing each branch to a scales allocation and selecting between the
rematerialized descriptors for the MMA. New dynamic A/B scale descriptor-view
runtime rows cover both scale operands. Validation: required `make -j8`;
dynamic A/B scale descriptor-view rows `2 passed`; combined
`scale_descriptor_view or bscale_descriptor_view` selector
`12 passed, 1617 deselected`; targeted lit set `6/6`; 4-GPU positive MMAv5
selector passed as group1 `134 passed, 14 skipped`, group2 `148 passed`,
group3 `148 passed`, group4 `148 passed`; `git diff --check` passed.

Completed seventeenth implementation slice: selected scales descriptor-view
load/store lowering is type-local. The old path identified scale roots by
walking descriptor producers; after `arith.select`, exact lowering failed and
query-type rescue chose a dense-linear surrogate that did not preserve the
row-permuted physical mapping. The fix treats generated scales descriptor-view
linear layouts as local scales storage for ld/st query selection and LLVM
planning. Generic scale descriptor-view storage classification now accepts
two-CTA layouts; the B-scale-specific scaled-MMA padded-storage classifier still
rejects two-CTA as before. New runtime coverage selects between two same-typed
row-permuted scales descriptor views and asserts correctness plus exact tcgen05
opcode sequences for 1CTA and 2CTA. Validation: required `make -j8`; exact
dynamic selected scales ld/st rows `2 passed`; focused `ldst_scales` selector
`35 passed, 1596 deselected`; combined
`scale_descriptor_view or bscale_descriptor_view` selector
`12 passed, 1619 deselected`; targeted lit set `6/6`; 4-GPU
`ldst_scales or scale_descriptor_view or bscale_descriptor_view` split passed
as group1 `12 passed`, group2 `12 passed`, group3 `12 passed`, group4
`11 passed`; `git diff --check` passed.

First migration-slice finding: a focused `cp_no_scales` subslice selector with
`TRITON_DEBUG_TMEM_QUERY=1` passed runtime correctness (`63 passed`), but
reported many type-local/exact divergences. The exact repro
`test_tmem_runtime_matrix_cp_no_scales_linear_subslice_view[f32-...-128-128-32-16-tcgen05.cp.cta_group::1.128x256b]`
showed the result descriptor type for a column subview still carried the
parent-width TMEM-linear encoding (`128x256` physical layout) while the exact
producer-chain query recovered the active `128x128` view with origin
`col=128`. This confirms the first required semantic fix: `memdesc_subslice`
result type inference must prefer the active type-local layout relative to the
new `taddr`; preserving the parent encoding is only a fallback for genuinely
unrepresentable views.

Completed first implementation slice: `inferTMemSubsliceOpEncoding` now tries
the inferred active TMEM-linear layout before falling back to exact parent
encoding preservation. Handwritten lit tests were updated so subslice results
spell active-width encodings (`128x64` uses six column bases, `128x32` uses
five, `64x256` drops the parent `512` column bit). This is a semantic type
contract update, not a runtime correctness change. Validation: required
`make -j8`; lit `TritonNvidiaGPU/ops.mlir`,
`TritonNvidiaGPU/tmem_layouts.mlir`,
`Conversion/tritongpu_to_llvm_blackwell.mlir`,
`Analysis/test-buffer-region.mlir`, `TritonNvidiaGPU/invalid.mlir`, and
`TritonGPU/invalid.mlir` passed; exact runtime repro
`test_tmem_runtime_matrix_cp_no_scales_linear_subslice_view[f32-...-128-128-32-16-tcgen05.cp.cta_group::1.128x256b]`
passed; focused copy selector
`cp_no_scales and (indexed_view or linear_subslice_view or warpx2_candidate or 128x128)`
passed as `63 passed, 1560 deselected`.

Completed second implementation slice: active TMEM subviews with
self-contained result layouts now avoid borrowing parent load/store support
images and try the canonical standalone load/store surrogate before raw
parent-allocation query types. This fixed the `warpx2::01_23` two-CTA
subslice copy/load path, where copy planning was already selecting the same
physical destination family as the direct allocation but `tmem.get_reg_layout()`
picked the raw narrowed copy layout instead of the canonical direct
load/store register layout. The ordering predicate is type-local: the current
memdesc shape differs from alloc shape, and the current tensor-memory layout
matches the active shape. Validation after this slice: required `make -j8`;
exact `warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused
selector
`cp_no_scales and (indexed_view or linear_subslice_view or warpx2_candidate or 128x128)`
`63 passed, 1560 deselected`; 4-GPU
`cp_no_scales and not reports` split passed as group1 `54 passed, 4 skipped`,
group2 `58 passed`, group3 `58 passed`, group4 `57 passed`; targeted lit set
`TritonNvidiaGPU/ops.mlir`, `TritonNvidiaGPU/tmem_layouts.mlir`,
`Conversion/tritongpu_to_llvm_blackwell.mlir`,
`Analysis/test-buffer-region.mlir`, `TritonNvidiaGPU/invalid.mlir`, and
`TritonGPU/invalid.mlir` passed `6/6`.

Completed third implementation slice: active self-contained TMEM subview copy
destinations now select the type-local physical query instead of the legacy
exact producer-chain query. This makes the selected copy plan relative to the
current destination `taddr` for the same active subview class fixed by the
result-type and ld/st slices, while avoiding the earlier overbroad experiment
that broke M=256 row-group, tile-selector, and raw-root rows. Validation:
required `make -j8`; exact
`test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_subslice_view_positive`
`4 passed`; focused selector
`cp_no_scales and (indexed_view or linear_subslice_view or warpx2_candidate or 128x128)`
`63 passed, 1560 deselected`; 4-GPU
`cp_no_scales and not reports` split passed as group1 `54 passed, 4 skipped`,
group2 `58 passed`, group3 `58 passed`, group4 `57 passed`; targeted lit set
passed `6/6`; `git diff --check` passed.

Completed fourth implementation slice: too-small copy destinations now have an
explicit clean-negative diagnostic. `getTMemCopyAtomFailureMessage` describes
the current descriptor's logical column bits and the minimum legal hardware
copy atom width, and both `TMEMCopyOp::verify` and late LLVM copy lowering
attach the note if copy family classification fails. New runtime coverage uses
linear shared layouts for `128x1xf32` and `128x2xf32` so the failure reaches
`ttng.tmem_copy` rather than shared-memory layout construction. Validation:
required `make -j8`; new exact test `2 passed`; adjacent exact-width positive
`2 passed`; focused selector including the new negatives `65 passed,
1560 deselected`; 4-GPU `cp_no_scales and not reports` split passed as group1
`54 passed, 4 skipped`, group2 `58 passed`, group3 `58 passed`, group4
`57 passed`; targeted lit set passed `6/6`; `git diff --check` passed.

Completed fifth implementation slice: active self-contained TMEM subviews now
use local ld/st row planning. The row-plan helpers no longer fall through to
`getBackingTMemLdStRowPlan` for this descriptor class, and
`getTMemLdStQueryTypes` uses `getTMemLdStRowPlanForType(memTy)` for canonical
surrogate generation instead of parent-chain backing plans. This keeps legacy
views on existing rescue paths while removing one semantic parent-chain
dependency from active-layout subviews. Validation: required `make -j8`; exact
`warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused ld/st
selector `78 passed, 1547 deselected`; 4-GPU
`ldst and not reports and not scales` split passed as group1 `88 passed`,
group2 `32 passed, 56 skipped`, group3 `66 passed, 22 skipped`, group4
`67 passed, 20 skipped`; targeted lit set passed `6/6`; `git diff --check`
passed.

Completed sixth implementation slice: active self-contained TMEM subviews now
have a type-local ld/st raw-query builder. `inferTypeLocalTMemLdStQueryLayout`
computes the zero-origin query layout from `MemDescType` and its TMEM encoding,
then canonicalizes out-dim names to match the previous query contract.
`inferStandaloneTMemLdStQueryLayout(Value, ...)` dispatches to the type-local
builder for this descriptor class while preserving chain reconstruction for
legacy views. Validation: required `make -j8`; exact
`warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused ld/st
selector `78 passed, 1547 deselected`; 4-GPU
`ldst and not reports and not scales` split passed as group1 `88 passed`,
group2 `32 passed, 56 skipped`, group3 `66 passed, 22 skipped`, group4
`67 passed, 20 skipped`; targeted lit set passed `6/6`; `git diff --check`
passed.

Completed seventh implementation slice: active self-contained TMEM subviews now
have a type-local ld/st query-type list. `getTypeLocalTMemLdStQueryTypes`
builds the canonical surrogate/current-type candidate list using
`getTMemLdStRowPlanForType(memTy)`, and `getTMemLdStQueryTypes(Value)` returns
that list immediately for this descriptor class. This avoids producer-aware
standalone type reconstruction and parent backing-plan selection for active
layouts. Validation: required `make -j8`; exact
`warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused ld/st
selector `78 passed, 1547 deselected`; 4-GPU
`ldst and not reports and not scales` split passed as group1 `88 passed`,
group2 `32 passed, 56 skipped`, group3 `66 passed, 22 skipped`, group4
`67 passed, 20 skipped`; targeted lit set passed `6/6`; `git diff --check`
passed.

Completed eighth implementation slice: active self-contained TMEM subviews now
have type-local ld/st support-query planning. `getTypeLocalTMemLdStSupportQueryPlan`
builds the query layout from the current descriptor type and derives the row
plan from the current type or query layout. `getTMemLdStSupportQueryPlan(Value)`
returns that plan immediately for active self-contained descriptors, leaving
legacy descriptor views on the existing support-image reconstruction stack.
Validation: required `make -j8`; exact
`warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused ld/st
selector `78 passed, 1547 deselected`; focused `ld_red and not reports and not
scales` selector `239 passed, 1386 deselected`; 4-GPU
`(ldst or ld_red) and not reports and not scales` split passed as group1
`120 passed, 28 skipped`, group2 `98 passed, 50 skipped`, group3
`128 passed, 20 skipped`, group4 `146 passed`; targeted lit set passed `6/6`.

Completed ninth implementation slice: active self-contained TMEM subviews no
longer borrow hidden parent row plans in fallback paths after type-local query
planning. `getTMemLdStRowPlanForRawQuery`,
`getTMemLdStDirectSupportTensorType`, and
`getTMemLoadReductionLayoutForMemDesc` now use the current type's row plan for
active self-contained descriptors and only call `getBackingTMemLdStRowPlan` for
legacy descriptors. Validation: required `make -j8`; exact
`warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused ld/st
selector `78 passed, 1547 deselected`; focused `ld_red and not reports and not
scales` selector `239 passed, 1386 deselected`; 4-GPU
`(ldst or ld_red) and not reports and not scales` split passed as group1
`120 passed, 28 skipped`, group2 `98 passed, 50 skipped`, group3
`128 passed, 20 skipped`, group4 `146 passed`; targeted lit set passed `6/6`.

Completed tenth implementation slice: active self-contained TMEM subviews now
keep ld/st and copy LLVM lowering relative to the current `taddr`.
`hasSelfContainedTMemSubviewLayout(MemDescType)` is public so conversion code
can share the active-layout predicate. `lowerTMemLdStFromTypes` now skips
standalone reg-layout query-type reconstruction, backing-row fallback,
source-column support rescue, and already-adjusted base-offset subtraction for
active self-contained descriptors. `copySharedToTmem` skips
`getTMemSubviewRelativeBaseOffset` when the selected destination query is
type-local. Validation: required `make -j8`; exact
`warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused ld/st
selector `78 passed`; focused `ld_red and not reports and not scales` selector
`239 passed`; focused copy selector `57 passed`; 4-GPU combined
`(ldst or ld_red or cp_no_scales) and not reports and not scales` split passed
as group1 `120 passed, 28 skipped`, group2 `98 passed, 50 skipped`, group3
`128 passed, 20 skipped`, group4 `146 passed`; dedicated 4-GPU
`cp_no_scales and not reports` split passed as group1 `54 passed, 4 skipped`,
group2 `58 passed`, group3 `58 passed`, group4 `57 passed`; targeted lit set
passed `6/6`.

Completed eleventh implementation slice: verifier behavior now matches the
active-layout lowering contract. `verifyTMEMOperand` skips standalone
view-type physical-support rescue for active self-contained descriptors.
`TMEMLoadOp::verify` skips `getBackingTMemLdStRowPlan` fallback in support and
raw reduction checks for the same descriptor class. Validation: required
`make -j8`; exact `warpx2_01_23_twocta_subslice_view_positive` `4 passed`;
focused ld/st selector `78 passed`; focused `ld_red and not reports and not
scales` selector `239 passed`; 4-GPU
`(ldst or ld_red) and not reports and not scales` split passed as group1
`120 passed, 28 skipped`, group2 `98 passed, 50 skipped`, group3
`128 passed, 20 skipped`, group4 `146 passed`; targeted lit set passed `6/6`.

Current prototype evidence: hand-written LLVM IR passed through
`opt -S -O2` shows unused or statically zero subword-phase arithmetic is removed
for f32, unpacked f16, and unpacked i8 cases. Packed f16/i8 examples retain
only the phase arithmetic when the synthetic pack/unpack consumer uses it.

Latest design checkpoint: 2026-04-22 20:16 UTC refined the TMEM memdesc runtime
abstraction around colStride/packing and legal-lowering derivation. Basic
memdesc identity remains `MemDescType` state; CTA/CGA ownership belongs in the
tensor-memory layout; `LinearLayout` `col` is the physical element-slot
coordinate; packed and unpacked f16/f8/i8 storage are represented by layout
bases/column stride; support footprints must be derived from the current
descriptor layout plus allocation/storage shape relative to current `taddr`;
and row-plan/copy/reduction/MMAv5 validity must be derived from type/layout and
operation semantics. Producer-chain peepholes may rank or select among
already-valid lowerings for performance, but must not define legality.
Too-small `tcgen05.copy` memdescs, such as dense `128x1xf32` or `128x2xf32`,
are clean negatives unless the current descriptor layout itself represents a
legal copy family; lowering must not borrow hidden parent columns to make an
illegal shape codegenable.
Documentation-only checkpoint; no build or runtime tests were run.

Latest validation checkpoint: 2026-04-22 03:13 UTC upstream-merge and
post-merge refactor pass completed. `codex/tmem` is merged with upstream main
`37c9a4b569a0`; merge checkpoint `b96231ec8` was pushed to
`origin/codex/tmem`. The post-merge code cleanup kept TMEM packed-address
logic centralized by adding shared 32-bit word-column scaling and replacing
remaining branch-local `col * bitwidth / 32`, `(row << 16)`, and direct packed
offset expressions in the IR planner, copy planner, and MMAv5 lowering.
Validation: required `make -j8`; lit `TritonNvidiaGPU/tmem_layouts.mlir`,
`TritonNvidiaGPU/interleave_tmem.mlir`, and
`Conversion/lower_tensor_memory_to_llvm.mlir` passed `3/3`; full runtime
matrix split-4 passed as group1 `308 passed, 98 skipped`, group2 `402 passed,
4 skipped`, group3 `406 passed`, group4 `405 passed`; structural fuzzer
split-4 passed `9 + 9 + 9 + 9 = 36`. Current next slice: with the merge and
refactor green, continue only with current-head cleanup/probes or the broader
completion queue if a live gap appears.

Previous validation checkpoint: 2026-04-22 02:28 UTC branch-diff refactor pass
completed over the TMEM backend surface in `origin/main...HEAD`. The full
branch delta is intentionally broad, so this pass avoided unrelated
upstream/AMD/proton/test churn and focused on reusable TMEM backend
generalization. Source cleanup: shared packed TMEM row/column offset helpers
now live in `TensorMemoryUtils.h`; direct ld/st planning, copy destination
planning, and LLVM lowering use them for row-base construction, row/column
packing/extraction, row-limit checks, and row scaling. Query-origin base-offset
construction for physical queries and ld/st queries now shares one helper.
Validation: required `make -j8`; exact prior `ld.red` descriptor-chain crash
repro `1 passed`; focused affected runtime selector `83 passed`; structural
fuzzer split-4 `9 + 9 + 9 + 9 = 36 passed`; targeted lit
`TritonNvidiaGPU/tmem_layouts.mlir` and `interleave_tmem.mlir` `2 passed`;
runtime/structural `py_compile` passed; `git diff --check` passed. Current
next slice: from this green baseline, continue with current-head probes or
the broader completion queue only if a live gap appears.

Previous validation checkpoint: 2026-04-22 02:09 UTC cleanup/generalization
pass completed over the recent TMEM repair code. Source cleanup:
`OptimizeTMemLayouts` now has a shared `isPlainSingleResultTMemLoad` guard
used by split-load, physical-support, full/half/leading replay,
reduction-rewrite, fused reduction, and TMEM-to-shared patterns; generic
one-result rewrites no longer duplicate ad hoc `redOp`/token checks or risk
rewriting reduction/token TMEM loads. `TensorMemoryToLLVM` now names the
packed row/column packet-offset split and reuses
`getTMemLdStReductionRepeats` for `ld.red` packet legality. Validation:
required `make -j8`; exact prior `ld.red` descriptor-chain crash repro
`1 passed`; focused affected runtime selector `83 passed`; structural fuzzer
split-4 `9 + 9 + 9 + 9 = 36 passed`; targeted lit
`TritonNvidiaGPU/tmem_layouts.mlir` and `interleave_tmem.mlir` `2 passed`;
runtime/structural `py_compile` passed; `git diff --check` passed. Current
fuzz status: no checked-in structural xfails; full runtime matrix remains
green from the 00:30 checkpoint; do not spend implementation time on stale
temporary `FZ-*` buckets unless they reproduce on current head. Current next
slice: cleanup/validation/current-head probes only, then broader completion
queue items if a live gap appears.

Previous validation checkpoint: 2026-04-22 00:38 UTC external TMEM coverage
outside the runtime-matrix file is green after the full-matrix fallout repair.
No source changes were needed. `test_core.py` selector
`tmem and (copy or ld or load or store or mma or tcgen05)` passed split-4 as
group1 `14 passed`, group2 `14 passed`, group3 `14 passed`, group4
`8 passed, 5 skipped`. TMEM-focused Gluon examples
`05-moe-bmm1-fused-gather.py`, `05-tmem-moe-router.py`,
`06-tmem-lora-fusion.py`, and `08-tmem-layout-as-epilogue.py` collected
`85` tests with `PYTHONPATH=.:./python:./python/triton_kernels` and passed
split-4 as group1 `22 passed`, group2 `22 passed`, group3 `22 passed`, group4
`19 passed`. Attention-forward fp16/N1024 smoke passed split-4 as `2 + 2 + 2
+ 2`. Remaining checked-in structural xfails: none. Current next slice:
because the runtime matrix, structural fuzzer, `test_core.py` TMEM selector,
and TMEM-focused examples are green, continue with cleanup/validation only
unless a new current-head probe reproduces a cataloged boundary as a real
compiler/runtime failure.

Previous validation checkpoint: 2026-04-22 00:30 UTC full runtime-matrix fallout
from the rank-5 selected-parent replay checkpoint is repaired. The first
full-file split after commit `f2c2233db` found one real optimizer crash and
stale expectation rows. Root cause for the crash: generic full-view replay
patterns matched `ttng.tmem_load` with `redOp`/token results and tried to
replace the two-result reduction load with one replay value. Implementation:
`OptimizeTMemLayouts` now skips those generic replay patterns for reduction or
token TMEM loads, leaving reduction-specific lowering/fusion paths in charge.
Expectation cleanup: runtime matrix opcode assertions now follow the new
lowering contract where row displacement advances the TMEM base register and
the `tcgen05` bracket immediate records only column displacement. Old row-only
`1048576` offsets become `0`, and row+column offsets retain only the column
component. Marker checks were relaxed only for no-op descriptor shape views
that canonicalization can now remove; runtime correctness, exact opcode
families/counts, software/hardware reduction checks, `tensor_memory_linear`,
descriptor indexing, and `twoCTAs` checks remain. Validation: required
`make -j8`; exact reduction crash repro `1 passed`; affected selector
`83 passed`; full runtime matrix split-4 passed as group1
`308 passed, 98 skipped`, group2 `402 passed, 4 skipped`, group3
`406 passed`, group4 `405 passed`; structural fuzzer split-4 `36 passed`;
targeted lit `TritonNvidiaGPU/tmem_layouts.mlir` and `interleave_tmem.mlir`
`2 passed`; runtime/structural `py_compile` passed; `git diff --check`
passed. Remaining checked-in structural xfails: none. Next frontier: continue
the remaining completion queue from a green full runtime-matrix baseline:
broader MMAv5 reachable-family support, heuristic cleanup, copy/scales if new
current-head probes reproduce, and staged broad validation.

Previous repair checkpoint: 2026-04-21 23:56 UTC rank-5 selected-parent `ld/st`
backend gap is repaired. Current-head rank-5 replay showed six small rows still
failed after marker cleanup, and one earlier unit-parent row had exposed a
wrong-result risk from vectorized row-advancing `32x32b.x64` packet repetition.
Root causes: full-view replay could try to direct-load/store an
`index(subslice(parent, [1, 0, 0]), 0)` descriptor instead of the equivalent
`index(parent, 1)` descriptor, support selection could choose a logical
alias-layout packetization for a physical storeback, and lowering packed row
bits into the bracket offset confused row displacement with the column
immediate. Implementation: full-view replay now canonicalizes leading unit
subslice plus index bases to the parent index only for that case, tracks
equivalent replay bases through constant index chains, uses physical base
support for canonicalized storeback while preserving existing replay behavior
elsewhere, rejects vectorized `32x32b` packet repetition through TMEM rows, and
splits row/column packet offsets in LLVM lowering. Runtime-matrix rank-5
markers now allow the canonicalized subslice/index to disappear while keeping
runtime correctness and exact `tcgen05` opcode counts. Validation: required
`make -j8`; exact small repro `1 passed`; full rank-5 ld/st descriptor selector
`23 passed, 10 skipped`; full structural fuzzer split-4 `36 passed`; lit
`TritonNvidiaGPU/tmem_layouts.mlir` and `interleave_tmem.mlir` `2 passed`;
runtime and structural `py_compile` passed; broad MMAv5 selector split-4
`133 passed, 14 skipped`, `147 passed`, `147 passed`, `147 passed`;
`git diff --check` passed. Remaining checked-in structural xfails: none. Next
frontier: continue staged broad validation and then move to heuristic cleanup
or remaining non-structural completion phases.

Previous validation checkpoint: 2026-04-21 23:04 UTC broad positive MMAv5
frontier is green on current head after stale rank-5 descriptor-view marker
cleanup. Selector `mma and not reports and not clean and not unsupported`
collected `588/1623` rows. Initial split-4 run had `3` failures, all two-CTA
MMAv5 rank-5 `ld/st` rows with correct runtime outputs and exact opcode
counts but stale requirements that no-op reshape/transpose/full-slice
descriptor ops survive in TTGIR. Tests now still assert `tensor_memory_linear`,
descriptor indexing/subslice where nontrivial, `twoCTAs = true`, exact
`tcgen05` opcode counts, and runtime correctness; they no longer require
canonicalized no-op shape views to remain visible for exact lifted two-CTA
layouts. Validation: required `make -j8`; exact affected rows `5 passed`;
runtime-matrix `py_compile` passed; broad MMAv5 split-4 rerun passed as
`133 passed, 14 skipped`, `147 passed`, `147 passed`, and `147 passed`
(`574 passed, 14 skipped`). Next frontier: staged broad validation and
current-head reruns of remaining temporary-probe buckets.

Previous repair checkpoint: 2026-04-21 22:50 UTC `FZ-20260421-0016` and
`FZ-20260421-0023` are repaired on current head. `FZ-0016` was a verifier abort
in `Conversion/relayout_tritongpu.mlir`: pre-conversion unencoded tensors were
accepted by TMEM verifier preconditions, but the verifier then called the
ld/st compatibility planner and crashed while deriving a `LinearLayout` from a
missing tensor encoding. `verifyTMEMOperand` now defers compatibility planning
for unencoded tensors, and the tensor-memory-scales broadcast and reduction
load verifier paths guard their later layout-dependent checks the same way.
`FZ-0023` was frontend inline expected-text drift; the expected TMEM subslice
register layout now includes the current `[0,32]`, `[0,64]`, and `[0,128]`
register bases. Current copy `warpx2`/scales frontier replay collected
`97/1623` and passed split-4 as `25 + 25 + 25 + 22`, so no live copy/scales
implementation failure was found in this slice. Validation: required
`make -j8`; lit `Conversion/relayout_tritongpu.mlir`,
`TritonNvidiaGPU/tmem_layouts.mlir`, and `TritonNvidiaGPU/interleave_tmem.mlir`
passed `3/3`; exact frontend drift test `1 passed`; frontend
`tensor_memory or tmem_` selector `29 passed`; full structural fuzzer split-4
`36 passed`; frontend `py_compile` passed. Remaining checked-in structural
xfails: none. Next frontier: broader MMAv5 reachable-family support, heuristic
cleanup, staged broad validation, and any still-live temporary-probe buckets
that reproduce on current head.

Previous repair checkpoint: 2026-04-21 22:39 UTC `FZ-20260421-0015` is repaired
for valid selected direct B-scale scaled-MMAv5 rows. Current-head replay
showed that direct branch/helper/pass-through selected B-scale rows were
already stale-green, but the loop-carried `scf.for` B-scale row still
miscompiled. Root cause: allocation liveness followed some alias values, but
it did not map a tensor-memory memdesc operand consumed by scaled MMA back to
all root TMEM allocations when that operand was produced by loop-carried
control flow. The accumulator was allocated over still-live B-scale columns.
Implementation: `TensorMemoryAllocation` now traces every tensor-memory
memdesc operand through `getAlloc` to attach the consuming op as an extra live
user of each root allocation, records direct alias users explicitly, and
follows `scf.for` init operands to region iter args/results. New positive
runtime coverage `test_tmem_runtime_matrix_mma_scaled_dynamic_bscale_direct`
covers branch and loop selected direct B-scale descriptors, both selectors,
`N=64`, and `K=256` with distinct scale payloads. Validation: required
`make -j8`; exact new test `6 passed`; temporary Round 17 audit rows for
direct, constexpr, same-object, branch-selected, loop-carried, and A-scale
dynamic controls all show `0` mismatches; focused scaled-MMAv5 selector
split-4 `113 passed`; structural fuzzer split-4 `36 passed`; targeted lit
`2 passed`; broader positive scaled-MMAv5 split-4 `251 passed`; `py_compile`
and `git diff --check` passed. Remaining checked-in structural xfails: none.
The old parent-slice row in the temporary minimizer
remains a separate frontend unsupported-shape boundary, not live `FZ-0015`.
Next frontier: copy `warpx2`/scales boundaries, broader MMAv5 reachable-family
support, heuristic cleanup, and staged broad validation.

Previous repair checkpoint: 2026-04-21 22:19 UTC `FZ-20260421-0013` is repaired
for valid scaled-MMAv5 scale descriptor-view operands. `TensorMemoryAllocation`
now canonicalizes A-scale and B-scale descriptor views before MMA by finding
the unique producer store through the scale alias chain, replaying supported
reshape/transpose descriptor-view transforms on the stored tensor, and storing
the logical view tensor into a fresh `TensorMemoryScalesLayout` allocation. The
B-scale fragment rematerializer uses the same alias/store helper, and cleanup
preserves the original store when another operand on the same MMA still uses
the alias chain. Reclassification: the temporary padded full-width linear
fuzzer rows are invalid positives without a narrow/repeated-N scale-fragment
requirement; padding a full `N=128/256` linear scale descriptor duplicates only
part of the scale rows before MMA consumes the first full tile. The checked-in
padded N32 tile-permuted B-scale control remains valid. Validation: required
`make -j8`; new linear A/B scale descriptor-view runtime test `2 passed`;
existing B-scale descriptor-view, extra-user, and padded controls `3 passed`;
temporary `FZ-0013` B-view/A-view/NVFP4 B-view probes `3 passed`;
scaled-MMAv5 descriptor/accumulator selector split-4 `107 passed`; structural
fuzzer split-4 `36 passed`; M64 `ld_red_m64 and not reports` split-4
`39 passed`, so `FZ-20260421-0012` is stale on current head; targeted lit
`2 passed`; `py_compile` and `git diff --check` passed. Remaining checked-in
structural xfails: none. Next repair-plan frontier: copy `warpx2`/scales
boundaries, broader MMAv5 reachable-family support, heuristic cleanup, and
staged broad validation.

Previous repair checkpoint: 2026-04-21 21:56 UTC checked-in `FZ-0007`
(`mma-scaled-fz20260421-0007-subslice-if-n64-selector0`) is repaired and the
checked-in structural fuzzer has no remaining xfails. The core bug was TMEM
allocation liveness, not scaled MMA lowering: selected/yielded memdesc aliases
did not extend the parent allocation lifetime, letting later scale allocations
reuse the low accumulator subslice before the selected descriptor reached MMA
and load. `TensorMemoryAllocation` now follows `arith.select`, descriptor
views, `TMEMSubSliceOp`, and `scf.yield` aliases for `scf.if`/`scf.for`.
Validation exposed and repaired a related scaled-MMAv5 B-scale
descriptor-view gap: rematerialization now replays reshape/transpose
descriptor-view transforms on the producer store tensor before padding row
fragments, so it repeats logical B-scale rows rather than upstream parent
storage rows. Validation: required `make -j8`; exact promoted FZ-0007 row
`1 passed`; exact B-scale descriptor-view controls `3 passed`; full structural
fuzzer split-4 `36 passed`; scaled-MMAv5 use-acc/B-scale selector split-4
`55 passed`; targeted lit `2 passed`; `py_compile` and `git diff --check`
passed. Remaining checked-in structural xfails: none. Next repair-plan
frontier: broader cataloged backend gaps and long-term completion phases
(`FZ-20260421-0012` M64 `ld.red`, copy `warpx2`/scales, broader MMAv5,
heuristic cleanup, staged broad validation).

Previous repair checkpoint: 2026-04-21 21:39 UTC checked-in
`FZ-20260421-0006` is repaired as an execution blocker. Static no-op
full-shape descriptor slices are now canonicalized away in Gluon, so the
rotate/transpose/full-slice chain reaches backend replay instead of failing
frontend row-anchor inference. The promoted row validates correct runtime
results as a software-reduce fallback (`.ld.` plus `tt.reduce`) and explicitly
does not assert hardware `.ld.red.`; the remaining hardware opcode gap belongs
to existing `FZ-20260421-0012` M64 non-identity-row destination planning.
Validation: required `make -j8`; exact promoted row `1 passed`; full
structural fuzzer split-4 `35 passed, 1 xfailed`; targeted lit `2 passed`;
`py_compile` and `git diff --check` passed. Remaining repair-plan frontier:
the final checked-in structural xfail is `FZ-0007`.

Previous repair checkpoint: 2026-04-21 21:29 UTC checked-in `R5-C`
(`generic-pass-loop-carried-memdesc-view-chain0`) is repaired.
`OptimizeTMemLayouts` now rewrites eligible loop-carried full-view memdesc
loads by carrying the replayed tensor through `scf.for`: the loop init and
yield descriptors are materialized from their original full-view chains,
nested `scf.if` yields are rewritten recursively, and pass-through carried
values reuse the tensor iter arg. Validation: required `make -j8`; exact
promoted row `1 passed`; nearby control-flow replay guard `2 passed`; full
structural fuzzer split-4 `34 passed, 2 xfailed`; targeted lit `2 passed`;
`py_compile` and `git diff --check` passed. Remaining repair-plan frontier:
the two checked-in structural xfails are `FZ-0006` and `FZ-0007`.

Previous repair checkpoint: 2026-04-21 21:18 UTC checked-in
`FZ-20260421-0005`, `FZ-20260421-0008`, and `FZ-20260421-0009` are repaired.
Expanded separable lifted-rank allocations now use the compact 128-row TMEM
physical image and do not double-count prefix selectors. Layout composition
now rejects incompatible dimension sets instead of aborting. Full-view replay
now treats descriptor-order loads, axis-sensitive reduction loads, external
stores, and TMEM-load-sourced roundtrips separately. Promoted positives: the
256-row lifted-parent `ld/st` sentinel, the 2CTA row/col indexed `ld.red`
optimizer-crash sentinel, and the 1CTA 256x32 direct-index `ld.red` allocator
sentinel. The two `ld.red` crash sentinels are currently correct software
fallbacks (`.ld.` plus `tt.reduce`). Validation: required `make -j8`; exact
promoted rows `3 passed`; full structural fuzzer split-4
`33 passed, 3 xfailed`; targeted lit `2 passed`; `py_compile` and
`git diff --check` passed. Remaining repair-plan frontier: the three checked-in
structural xfails are `FZ-0006`, `R5-C`, and `FZ-0007`.

Previous repair checkpoint: 2026-04-21 20:37 UTC checked-in
`FZ-20260421-0004` `ld.red` opcode-loss rows are repaired. The reduction-load
planner now uses support-query-aware descriptor planning, Gluon only emits a
direct `ttng.tmem_load {redOp}` when the specific memdesc/result contract is
backend-legal, and `OptimizeTMemLayouts` fuses replayable plain-load plus
min/max reductions back into hardware `tcgen05.ld.red` after layout replay.
M64 row planning now preserves canonical 128-row backing layouts while using
active 64-row row plans for noncanonical split-N descriptor views. Promoted
positives: the five `ldred-fz20260421-0004-*` structural sentinels. Validation:
required `make -j8`; exact promoted rows `5 passed`; full structural fuzzer
split-4 `30 passed, 6 xfailed`; broad `ld_red` runtime selector `145 passed`;
direct M64 permuted controls `6 passed`; targeted lit `2 passed`;
`py_compile` and `git diff --check` passed. Remaining repair-plan frontier:
inspect the remaining `6` structural xfails and pick the highest-impact
backend bucket that is still a true support gap.

Latest repair checkpoint: 2026-04-21 19:39 UTC checked-in
`FZ-20260421-0002` generic-pass/control-flow descriptor SSA rows are repaired.
`OptimizeTMemLayouts` now sinks replayable full-view TMEM loads through
`scf.if` results when both branches yield replayable full-view descriptors,
so replay uses each branch's original descriptor-view chain instead of the
merged transformed memdesc. Promoted positives:
`generic-pass-dynamic-if-chain0-true`,
`generic-pass-dynamic-if-chain0-false-16x128b`,
`generic-pass-dynamic-if-chain0-inline`,
`generic-pass-mixed-captures-chain0`, and
`generic-pass-tuple-mixed-captures-chain0`. Validation: required `make -j8`;
exact generic-pass memdesc-control `7 passed`; full structural fuzzer split-4
`25 passed, 11 xfailed`; targeted lit `2 passed`; structural fuzzer
`py_compile` passed. Remaining repair-plan frontier: the non-`FZ-0002`
checked-in structural xfails and older cataloged runtime/compiler buckets.

Latest repair checkpoint: 2026-04-21 18:56 UTC direct `ld/st` full-view
descriptor replay repair landed locally for checked-in `FZ-20260421-0003`
coverage. Code changes preserve TMEM descriptor-view provenance through
`memdesc_trans`, replay full-view loads/stores through a requested-packet-aware
support layout that is validated against the base descriptor, handle M64
non-family raw row plans for non-split packets, normalize rank-5 support-query
out-dim names, and let scaled-MMAv5 B-scale rematerialization find stores
through descriptor-view alias chains. Promoted positives:
`ldst-fz20260421-0003-chain1-64x32-32x32b`,
`ldst-fz20260421-0003-chain2-col-reverse-64x32-16x64b`,
`ldst-fz20260421-0003-f16-chain2-identity-64x32-16x64b`,
`generic-pass-dynamic-index-chain0`,
`generic-pass-layout-conversion-pressure-chain0`, and
`generic-pass-layout-conversion-pressure-chain0-16x128b`. Validation:
required `make -j8`; exact scales CGA `9 passed`; exact rank-5 unit-parent
`9 passed`; descriptor/higher-rank runtime split groups green as
`56 passed, 32 skipped`, `42 passed, 46 skipped`,
`68 passed, 20 skipped`, and `85 passed`; full structural fuzzer split-4
`20 passed, 16 xfailed`. Remaining repair-plan frontier:
`FZ-20260421-0002` dynamic/control-flow/capture-carried descriptor SSA wrong
results, plus the older cataloged non-`FZ-0003` buckets.

Latest fuzzing checkpoint: 2026-04-21 16:15 UTC Round 60 lit compiler breadth
lane B completed. Report:
`agents/fuzz_round60_lit_compiler_breadth_lane.md`. Required `make -j8` and
build-tree `ninja triton-opt` were no-ops. Non-overlapping compiler-only lit
coverage ran over TMEM allocation, promotion, MMAv5 lowering, Blackwell
pipeline lowering, automatic warp specialization, TritonGPU/NVGPU invalid
diagnostics, NVWS TMEM store hoisting, TritonGPU scheduling, and generic
conversion as `23 passed, 0 failed`. No compiler crash, verifier drift,
invalid-diagnostic drift, PassManager failure, TMEM allocation/lifetime
failure, conversion failure, NVWS hoisting failure, reproduced existing
`FZ-*`, or new independent `FZ-*`.

Latest fuzzing checkpoint: 2026-04-21 16:09 UTC Round 59 clean-negative
diagnostics lane B completed. Report:
`agents/fuzz_round59_clean_negative_diagnostics_lane.md`. Required `make -j8`
was a no-op. Checked-in pytest clean-negative selector over invalid TMEM
layouts, copy, `ld.red`, plain MMAv5, and scaled-MMAv5 collected `200/1615`
and passed split-4 as `200 passed` (`50`, `50`, `50`, `50`). Non-Round-57 lit
conversion/pass-manager sweep discovered `6` tests and ran as
`5 passed, 1 failed`; the failure is existing `FZ-20260421-0016` in
`Conversion/relayout_tritongpu.mlir`. No false unsupported diagnostic,
assertion text drift outside the known bucket, PassManager failure,
vague/incorrect diagnostic drift, runtime miscompile, or new independent
`FZ-*`.

Latest fuzzing checkpoint: 2026-04-21 16:09 UTC Round 59 higher-rank
descriptor-chain lane A completed. Report:
`agents/fuzz_round59_higher_rank_descriptor_lane.md`. Required `make -j8` was
a no-op. Checked-in descriptor-chain and direct two-CTA higher-rank selector
collected `39/1615` and passed split-4 as `39 passed`; frontend clean
diagnostic controls collected `3/225` and passed. Disposable contrasts
reproduced existing `FZ-20260421-0021` for unit-rank half-column descriptor
views, while adjacent rank-5 unit-prefix full views passed and did not
reproduce `FZ-20260421-0019`. No compiler crash, verifier drift, false
unsupported diagnostic, runtime miscompile, hang, or new independent `FZ-*`.

Previous fuzzing checkpoint: 2026-04-21 Round 58 local scales load/store and
copy lane completed. Report:
`agents/fuzz_round58_local_scales_copy_lane.md`. Required `make -j8` was a
no-op. Selector over scale `ld/st`, scale descriptor-view roundtrips,
scale-copy layout probes/rematerialization, TMEM descriptor-view clean
unsupported diagnostics, direct and two-CTA `warpx4` scale copy, and
scaled-MMAv5 helper copy paths collected `41/1615` and passed split-4 as
`41 passed`. No compiler crash, verifier drift, false unsupported diagnostic,
clean-boundary drift, opcode absence, runtime miscompile, hang, or new
independent `FZ-*`.

Previous fuzzing checkpoint: 2026-04-21 16:01 UTC Round 58 scaled-MMAv5
accumulator tile/narrow boundary lane A completed. Report:
`agents/fuzz_round58_scaled_tile_boundary_lane.md`. Required `make -j8` was a
no-op. Checked-in runtime selector collected `83/1615` rows over
accumulator tile-permuted N32/N64, narrow tile/identity/use-acc, B-scale
descriptor-view positive and N16 clean-boundary rows, and mixed fp4A clean
unsupported rows. Split-4 runtime passed as `83 passed` (`21`, `21`, `21`,
`20`). No temporary probe was needed. Existing `FZ-20260421-0007`,
`FZ-20260421-0013`, and `FZ-20260421-0015` classifications were unchanged.
No compiler crash, verifier drift, false unsupported diagnostic,
clean-boundary drift, runtime miscompile, opcode-count mismatch, hang, or new
independent `FZ-*`.

Previous fuzzing checkpoint: 2026-04-21 15:34 UTC Round 55 lit negative-boundary
lane C completed. Report:
`agents/fuzz_round55_lit_negative_boundary_lane.md`. Required `make -j8` was a
no-op. Checked-in lit over TMEM/proxy/mbarrier/relayout/allocation/conversion
surfaces discovered `16` tests and ran as `15 passed, 1 failed`; the failure
is existing `FZ-20260421-0016`. Extra proxy/mbarrier/TMA/invalid lit
diagnostics discovered `7` tests and passed. Direct Blackwell conversion
exited `0`; Round 38 compiler-boundary corpus replay stayed at `18 pass`,
`24` clean diagnostics, `5` existing late illegal-op rows, and `9` existing
abort/crash rows; Round 24 parse-only `FZ-0016` corpus replay stayed at `15`
existing abort/crash rows and `40` clean diagnostics. Minimized repros
revalidated existing `FZ-20260421-0016` and `FZ-20260421-0017`. No new
independent `FZ-*`.

Previous fuzzing checkpoint: 2026-04-21 Round 55 FZ-0003 boundary lane B
completed. Report: `agents/fuzz_round55_fz0003_boundary_lane.md`. Required
`make -j8` was a no-op. A temporary static descriptor-view probe completed as
`23 passed, 1 failed`; the failure was a clean unsupported minimal 2CTA
`128x32` chain-0 diagnostic. Wrong-result rows sharpen existing
`FZ-20260421-0003`: static no-control-flow chain-0 load-only rows miscompile
across 1CTA/2CTA, f32/i32/f16, and row/column variants, while matching
identity/row-neighbor chain-1 rows pass. Pure column-reverse `64x32` is a
harsher boundary where both chain-0 and chain-1 wrong-result. Checked-in exact
sentinels stayed `3 xfailed`. No new independent `FZ-*`; no backend repair.

Previous fuzzing checkpoint: 2026-04-21 15:24 UTC Round 54 resource/shape
extremes lane B completed. Report:
`agents/fuzz_round54_resource_shape_extremes_lane.md`. Required `make -j8`
was a no-op. Combined checked-in selector over resource, M64/N extremes,
high-CGA, `.x1`, subword, non-f32, 4x256b, `warpx2`, and N16/N32 rows
collected `515/1615` and ran split-4 as `439 passed, 70 skipped, 6 failed`;
the failures are exactly the existing `FZ-20260421-0012` M64 f32 `ld.red`
unsupported-destination rows. Clean/resource selector collected `66/1648` and
passed as `66 passed`; high-CGA core controls collected `16/18114` and passed
as `16 passed`; large `ld.red` ptxas probe classified `60` rows as `16`
existing `FZ-20260421-0018`, `12` pass, and `32` clean boundaries. `i64`/`f64`
compiler contrast reproduced existing `FZ-20260421-0017`; non-f32 `ld.red`
and 64-bit copy controls stayed clean/green. No new independent `FZ-*`.

Earlier fuzzing checkpoint: 2026-04-21 15:21 UTC Round 54 opcode consistency
lane C completed. Report: `agents/fuzz_round54_opcode_consistency_lane.md`.
Required `make -j8` was a no-op. Focused runtime-matrix opcode selector
collected `172/1615` rows and passed split-4 as `172 passed`; focused
proxy/mbarrier/multicast selector collected `39/19729` rows and passed as
`39 passed`; representative forced-compile dump replay passed `9` nodeids and
captured `27` TTGIR/LLIR/PTX artifacts under `/tmp/tmem_round54_opcode_dump`.
Coverage included `ld/st`, hardware f32 `ld.red`, intentional software-reduce
fallbacks, copy including `warpx2` and two-CTA `128x256b`, scaled-copy, plain
MMAv5, scaled MMAv5, wait/commit/proxy/mbarrier/multicast streams. No
PTX-vs-LLIR mismatch, missing hardware opcode, unexpected fallback, verifier
drift, compiler crash, runtime miscompile, or new independent `FZ-*`.

This is the active execution tracker for finishing the TMEM linear-layout
generalization project. It turns `backend_completion_plan.md` into a concrete
progress board so future sessions can resume without relying on chat context.

## Completion Definition

The project is complete when:
- TMEM copy, direct load/store, reduction load, plain MMAv5, and scaled-MMAv5
  all consume shared backend physical-query/planner objects instead of
  frontend or lowering-local layout policy;
- every bounded matrix case that the Blackwell TMEM ISA can realize is
  positive with runtime or opcode coverage;
- every remaining unsupported row is a true ISA/resource/API boundary with a
  typed backend diagnostic and probe evidence;
- compatibility scaffolding, stale frontend guards, and family-specific rescue
  paths are removed or explicitly quarantined;
- the TMEM-focused runtime matrix remains practical for iteration, with
  representative cold compile near the 3-4 second target and split-4 runtime
  validation kept duration-aware.

## Execution Contract

- Keep this file as the live progress board for the completion plan. Update it
  before and after meaningful implementation slices so the plan remains usable
  after an outage, context rollover, or branch handoff.
- 2026-04-21 structural fuzzing campaign override: the active workstream is
  continuous TMEM backend fuzzing for crashes, false unsupported diagnostics,
  verifier over-strictness, and runtime miscompiles. Do not end the turn while
  unblocked fuzzing work remains unless the user interrupts or a concrete
  external blocker is reached. Keep launching subagent rounds and local
  deterministic runtime fuzz sweeps. Catalog and minimize bugs, but do not
  begin backend fixes until fuzzing stops finding new failures or the user
  explicitly pivots to repair.
- Execute the highest-priority unblocked support-bearing slice first. Use
  cleanup-only slices when they remove family-specific policy or sharpen a true
  ISA boundary for the next support slice.
- Do not treat a green focused selector as completion. A slice is complete only
  after the relevant initiative docs record the new state, validation evidence,
  residual boundary, commit SHA, and pushed remote state.
- Continue execution against this board until all phases below are complete,
  unless the user interrupts or a concrete external blocker prevents further
  progress.

## Active Phase Board

- Phase R, systematic repair-plan execution: active as of 2026-04-21 16:59
  UTC, superseding the Round 62 stop condition and the catalog-only fuzzing
  campaign. Completed slices: dynamic encoded `memdesc_index` lowering
  (`FZ-20260421-0001`), broader dynamic copy/`ld.red` consumer positives, and
  direct full-view `ld/st` descriptor replay (`FZ-20260421-0003`), and
  generic-pass/control-flow full-view descriptor SSA replay
  (`FZ-20260421-0002`), and support-query-aware `ld.red`
  descriptor-chain/indexed opcode selection (`FZ-20260421-0004`), and
  compact lifted-rank allocation plus full-view replay directionality
  (`FZ-20260421-0005`, `FZ-20260421-0008`, `FZ-20260421-0009`), and
  loop-carried full-view replay (`R5-C`), and no-op full-slice descriptor
  canonicalization for `FZ-20260421-0006` as a correct software-reduce
  fallback, and selected/yielded memdesc alias liveness plus B-scale
  descriptor-view row-order rematerialization for `FZ-20260421-0007`.
  Current state after checkpoint: no checked-in structural fuzzer xfails
  remain. Current next unblocked slice: resume the broader completion queue
  from the remaining cataloged backend gaps, prioritizing high-value items such
  as `FZ-20260421-0012` M64 `ld.red` destination planning, copy
  `warpx2`/scales boundaries, broader MMAv5 reachable-family support,
  heuristic cleanup, and staged broad validation.

- Phase Z, 24-hour structural fuzzing campaign: active as of 2026-04-21
  08:18 UTC. Build a systematic deterministic Python/Gluon runtime fuzzer plus
  parallel subagent audit lanes. Primary outputs are:
  - `python/test/gluon/test_tmem_structural_fuzzer.py` for checked-in
    deterministic runtime fuzz cases and future regression promotion;
  - `.codex/initiatives/tmem_linear_generalization/tmem_structural_fuzzing_20260421.md`
    for the active case schema, round log, failure catalog, and repro queue;
  - subagent round reports split by TMEM family (`ld/st/ld.red`, `copy`,
    `MMAv5/scaled-MMAv5`, warp-specialization/membar, and generic pass
    interactions);
  - periodic checkpoint commits pushed to `origin/codex/tmem`.
  Backend fixes are intentionally deferred during this campaign while new
  failures are still appearing.
  2026-04-21 14:33 UTC Round 45 mixed exact smoke guardrail completed.
  Artifact: `agents/fuzz_mixed_exact_smoke_round45.md`. Required `make -j8`
  reported `no work to do`. Exact nodeids for scaled-MMAv5 B-scale extra-user,
  rank-5 descriptor-view roundtrip, and two-CTA `warpx2::01_23` dense-shared
  copy rematerialization passed on GPU0 as `3 passed`. No compiler crash,
  false unsupported diagnostic, runtime miscompile, or cache/process-order
  drift; backend repair remains deferred.
  2026-04-21 14:30 UTC Round 44 structural expansion audit integrated.
  Artifact: `agents/fuzz_structural_expansion_round44.md`; subagent commit
  `b63e7216`. Required `make -j8` reported `no work to do`. Structural
  overlap selector collected `28/33` and ran split-4 as `5 passed,
  23 xfailed`. Dynamic-copy temporary probe produced `4 passed, 2` existing
  `FZ-20260421-0001` failures; B-scale dynamic descriptor-view probe
  `5 passed`; focused high-rank/scale runtime controls `4 passed`; same-cache
  process-order control `2 passed, 1 xfailed`. No new independent `FZ-*`.
  Future checked-in candidate: dynamic descriptor plus `ttng.tmem_copy` as an
  expected `FZ-0001` copy-specific structural sentinel. Backend repair remains
  deferred.
  2026-04-21 14:30 UTC Round 44 `ld.red` M64 boundary lane integrated.
  Artifact: `agents/fuzz_ldred_m64_boundaries_round44.md`. Required
  `make -j8` reported `no work to do`. `ld_red_m64` selector collected
  `39/1615` and ran split-4 as `33 passed, 6 failed`; failures are the exact
  existing `FZ-20260421-0012` M64 row-basis unsupported-layout rows. Clean/
  resource `ld.red` selector `4 passed`; structural descriptor/`ld.red`
  sentinels `3 passed, 11 xfailed`; Round 41 M64 descriptor-chain oracle
  refresh reproduced existing `FZ-20260421-0003` and direct M64 `ld.red`
  control passed. No new independent `FZ-*`; backend repair remains deferred.
  2026-04-21 14:30 UTC Round 44 proxy/TMA core guardrail completed. Artifact:
  `agents/fuzz_proxy_tma_core_guardrail_round44.md`. Selector
  `(tma_multicast or multicast_commit or tcgen05_commit or mbarrier or proxy)
  and not reports and not resource` collected `12/19729` rows and passed on
  GPU0 as `12 passed`. No compiler crash, false unsupported diagnostic, opcode
  absence, runtime miscompile, proxy/mbarrier diagnostic drift, or new
  independent `FZ-*`; backend repair remains deferred.
  2026-04-21 14:27 UTC Round 44 descriptor/high-rank positive repeat
  completed. Artifact: `agents/fuzz_descriptor_high_rank_repeat_round44.md`.
  Required `make -j8` reported `no work to do`. Selector
  `(descriptor_compositions or higher_rank or rank5 or multidim_slice or
  half_rows) and not reports and not resource and not clean` collected
  `140/1615` rows and passed/stably skipped split-4 as
  `120 passed, 20 skipped` (`35`, `35`, `27/8`, `23/12`). No compiler crash,
  false unsupported diagnostic, clean-boundary drift, unexpected pass/fail
  transition, runtime miscompile, warm-cache sensitivity, or new independent
  `FZ-*`; backend repair remains deferred.
  2026-04-21 14:26 UTC Round 43 mixed MMA sequence subagent lane integrated.
  Artifact: `agents/fuzz_mma_sequence_mix_round43.md`. Required `make -j8`
  reported `no work to do`. Checked-in selector over plain/scaled MMAv5,
  `use_acc`, indexed/subslice accumulators, B-scale, and scale surfaces
  collected `648/19729` and passed/stably skipped split-4 as
  `645 passed, 3 skipped`. Replayed Round 24 mixed-scaled probe and Round 15
  multi-MMA probe; failures were known `FZ-20260421-0013` scale
  descriptor-view wrong-results and known `FZ-20260421-0015` selected distinct
  B-scale wrong-results. Exact `FZ-0015` replay remained known. New temporary
  plain-plus-scaled sequence probe passed all five rows, showing direct-scale
  plain-MMAv5/scaled-MMAv5 sequencing is green for the tested shapes. No new
  `FZ-*`; backend repair remains deferred.
  2026-04-21 14:25 UTC Round 43 copy/ldst lifetime subagent lane integrated.
  Artifact: `agents/fuzz_copy_ldst_lifetime_round43.md`; subagent commit
  `8b268af06`. Required `make -j8` reported `no work to do`. Broad checked-in
  selector collected `516/1648` and ran split-4 as
  `414 passed, 98 skipped, 4 xfailed`; xfails were expected existing
  `FZ-20260421-0003` structural `ld/st` descriptor-view rows. Temporary
  interleaving probe passed 4/4 rows covering copy-then-ldst, ldst-then-copy,
  descriptor reuse, and lifetime pressure. Report-inclusive copy/ldst clean
  diagnostic selector collected `85/19729` and passed as `85 passed`. No
  compiler crash, runtime miscompile, opcode absence, false unsupported
  diagnostic, clean-boundary drift, unexpected xfail/pass transition, or new
  independent `FZ-*`; backend repair remains deferred.
  2026-04-21 14:25 UTC Round 43 copy positive guardrail completed. Artifact:
  `agents/fuzz_copy_positive_round43.md`. Selector covering no-scales linear,
  indexed/subslice descriptor-view, `warpx2::{01_23,02_13}` candidate, tile-
  permuted linear, and scaled-copy geometry positives collected `77/1615` and
  passed split-4 as `77 passed` (`20/20/20/17`). No compiler crash, false
  unsupported diagnostic, opcode absence, runtime miscompile, unexpected
  pass/fail transition, or new independent `FZ-*`; backend repair remains
  deferred.
  2026-04-21 14:23 UTC Round 43 clean diagnostic drift guardrail completed.
  Artifact: `agents/fuzz_clean_diag_guardrail_round43.md`. Selector
  `(clean_error or clean_unsupported or reports_clean_error or
  reports_clean_unsupported or reports_backend_error) and not cp_no_scales and
  not cp_scales and not mma_i8 and not resource` collected `91/19729` rows
  across `test_core.py` and `test_tmem_runtime_matrix.py`, then passed split-4
  as `91 passed` (`23/23/23/22`). No compiler crash, diagnostic drift,
  unexpected pass/fail transition, runtime miscompile, or new independent
  `FZ-*`; backend repair remains deferred.
  2026-04-21 14:21 UTC Round 43 allocation/rematerialization guardrail
  completed. Artifact: `agents/fuzz_alloc_remat_guardrail_round43.md`.
  Required `make -j8` reported `no work to do`. Selector
  `(alloc_lifetime or allocation or source_initialization or rematerializes)
  and not reports and not resource` collected `17/1615` rows and passed on
  GPU0 as `17 passed`. No compiler crash, false unsupported diagnostic, opcode
  absence, runtime miscompile, unexpected pass/fail transition,
  allocation-lifetime drift, or new independent `FZ-*`; backend repair remains
  deferred.
  2026-04-21 14:20 UTC Round 42 high-CGA/cache ownership lane integrated.
  Artifact: `agents/fuzz_high_cga_cache_round42.md`. Required `make -j8`
  reported `no work to do`. Focused high-CGA/multicast selector collected
  `28/19762` and passed split-4 as `28 passed`. Broader adjacent selector
  collected `317/19729` and ran as `311 passed, 6 failed`; the six failures
  are existing `FZ-20260421-0012` M64 row/row-column-permuted `ld.red`
  unsupported-layout rows, not high-CGA ownership. Repeated the
  `FZ-20260421-0010` 2CTA-layout-in-4CTA-context clean diagnostic through
  stable warm caches on GPU0 and GPU1 as `1 passed` each run. No cache/process
  sensitivity, high-CGA drift, or new independent `FZ-*`; backend repair
  remains deferred.
  2026-04-21 14:19 UTC Round 42 scaled-MMAv5 positive guardrail completed.
  Artifact: `agents/fuzz_scaled_positive_round42.md`. Selector over
  scaled-MMAv5 root format, `use_acc`, accumulator subslice, tile-permuted
  64-wide accumulator, and narrow identity accumulator positives collected
  `94/1615` and passed split-4 as `94 passed` (`24/24/24/22`). No compiler
  crash, false unsupported diagnostic, opcode absence, runtime miscompile,
  unexpected pass/fail transition, or new independent `FZ-*`; backend repair
  remains deferred.
  2026-04-21 14:19 UTC Round 42 cache/state reuse subagent lane integrated.
  Artifact: `agents/fuzz_cache_state_reuse_round42.md`; subagent commit
  `38ec3f39`. Required `make -j8` reported `no work to do`. Mixed in-process
  GPU lanes intentionally ran positives, clean diagnostics, and expected
  structural FZ failures in one pytest process, then further positives; all
  post-failure positives passed. Warm-cache process-boundary controls matched
  prior guardrails: descriptor/high-rank `120 passed, 20 skipped`, clean
  diagnostics `22 passed`, and structural known-failure controls
  `3 passed, 23 xfailed`. No cache/state bug, process-boundary sensitivity,
  in-process contamination, or candidate new `FZ-*`; backend repair remains
  deferred.
  2026-04-21 14:18 UTC Round 42 core TMA/MMAv5 guardrail completed. Artifact:
  `agents/fuzz_core_tma_mma_guardrail_round42.md`. Selector
  `(tcgen05_mma or tma_mma_shared_inputs) and not scaled and not multicast`
  collected `223/18114` rows from `python/test/gluon/test_core.py` and
  passed/stably skipped split-4 as `172 passed, 51 skipped`
  (`47/9`, `47/9`, `41/15`, `37/18`). No compiler crash, false unsupported
  diagnostic, opcode absence, runtime miscompile, unexpected pass/fail
  transition, or new independent `FZ-*`; backend repair remains deferred.
  2026-04-21 14:16 UTC Round 42 `ld.red` positive layout guardrail completed.
  Artifact: `agents/fuzz_ldred_positive_layout_round42.md`. Selector
  `(ld_red_identity_linear_layout or ld_red_tile_permuted_linear_layout or
  ld_red_col_permuted_linear_layout or ld_red_row_permuted_linear_layout or
  ld_red_rowcol_permuted_n_sweep or
  ld_red_explicit_compatible_layout_variants) and not reports and not resource
  and not non_f32 and not descriptor_chain and not m64` collected `60/1615`
  rows and passed split-4 as `60 passed` (`15/15/15/15`). No compiler crash,
  false unsupported diagnostic, opcode absence, runtime miscompile,
  unexpected pass/fail transition, or new independent `FZ-*`; backend repair
  remains deferred.
  2026-04-21 14:16 UTC Round 41 subword/non-f32 descriptor-chain lane
  integrated. Artifact: `agents/fuzz_subword_descriptor_chains_round41.md`.
  Required `make -j8` reported `no work to do`. Checked-in selector over
  subword `ld/st`, x1/narrow clean boundaries, subword copy, non-f32 `ld.red`,
  f16/f8/i8 MMAv5/scaled-MMAv5, core descriptor-chain, and packed f16 rows
  collected `388/19762` and passed/stably skipped split-4 as
  `385 passed, 3 skipped`. Compiler-only probes reproduced only existing
  `FZ-20260421-0017` for encoded `i64`/`f64` descriptor-view non-reduction
  load/store; non-f32 `ld.red` remained a clean f32-only diagnostic, and
  `i64`/`f64` copy passed. No compiler crash, false unsupported diagnostic,
  opcode absence, clean-boundary drift, runtime miscompile, or new independent
  `FZ-*`; backend repair remains deferred.
  2026-04-21 14:15 UTC Round 41 M64 descriptor-chain oracle lane integrated.
  Artifact: `agents/fuzz_m64_descriptor_oracle_round41.md`. Required
  `make -j8` reported `no work to do`. The Round 39 unpromoted
  `same_chain_m64_n256_m64_row_reverse_w4` watch item was confirmed as real
  wrong-output evidence, not a harness artifact, but classified as additional
  `FZ-20260421-0003` evidence rather than a new bucket. Dedicated temporary
  oracle minimized the issue to `same_load`, M64, `N=32`, with a no-op
  reshape/slice descriptor chain zeroing rows `16..31` and `48..63`
  (`1024/2048` mismatches). `same_ldred` rows fail similarly across
  `N=32/64/128/256`, `min`/`max`, and explicit variants while still emitting
  hardware `.ld.red`; direct M64 `ld.red` passes. Not `FZ-0002`, not
  `FZ-0012`, and not `FZ-0004`; backend repair remains deferred.
  2026-04-21 14:14 UTC Round 41 TMEM lit guardrail completed. Artifact:
  `agents/fuzz_lit_guardrail_round41.md`. Build-tree `ninja triton-opt`
  reported `no work to do`. Lit passed `tmem_layouts.mlir`,
  `interleave_tmem.mlir`, and `memdesc-subview-split.mlir` as `3 passed`. No
  unexpected IR/diagnostic drift or new independent `FZ-*`; backend repair
  remains deferred.
  2026-04-21 14:13 UTC Round 41 `ld/st` descriptor positive guardrail
  completed. Artifact: `agents/fuzz_ldst_descriptor_positive_round41.md`.
  Selector
  `(ldst_descriptor_roundtrip or ldst_descriptor_compositions or
  ldst_twocta_descriptor_compositions or ldst_descriptor_multidim_slice_positive
  or ldst_descriptor_rank5_roundtrip or ldst_twocta_descriptor_rank5_roundtrip)
  and not reports and not resource and not x1 and not subword and not scales`
  collected `28/1615` rows and passed/stably skipped split-4 as
  `6 passed, 22 skipped`. No compiler crash, false unsupported diagnostic,
  clean-boundary drift, unexpected pass/fail transition, runtime miscompile, or
  new independent `FZ-*`; backend repair remains deferred.
  2026-04-21 14:12 UTC Round 41 full structural fuzzer guardrail completed.
  Artifact: `agents/fuzz_structural_full_guardrail_round41.md`. Required
  `make -j8` reported `no work to do`. Full
  `python/test/gluon/test_tmem_structural_fuzzer.py` passed as
  `9 passed, 24 xfailed`; visible xfail output matched expected
  `FZ-20260421-0001` late illegal `ttg.memdesc_index` diagnostics. No XPASS,
  unexpected failure, compiler crash outside expected xfail rows, runtime
  miscompile, or new independent `FZ-*`; backend repair remains deferred.
  2026-04-21 14:10 UTC Round 39 high-rank descriptor-chain subagent lane
  integrated. Artifact: `agents/fuzz_high_rank_chain_shapes_round39.md`;
  subagent commit `a82c17ce3`. Required `make -j8` reported `no work to do`.
  Checked-in selector `(higher_rank or rank5 or half_rows or ld_red and
  descriptor_chain) and not reports and not resource` passed/stably skipped as
  `92 passed, 20 skipped`. Temporary subprocess matrix covered `ld/st`,
  `load_min`, and `load_max` over direct, rank-4/rank-5 unit-prefix,
  inner-unit-rank, rank-5 half-row, and unit-rank half-column views with
  identity, row-reverse, column-reverse, and even/odd-plus-rotated bases.
  Aggregate: `80 pass`, `16` existing `FZ-20260421-0004` opcode-loss rows,
  `24` clean TMEM resource boundaries, and `24` existing
  `FZ-20260421-0021` half-column/unit-rank aborts. No new independent `FZ-*`;
  backend repair remains deferred.
  2026-04-21 14:10 UTC Round 40 copy/dynamic descriptor subagent lane
  integrated. Artifact: `agents/fuzz_copy_dynamic_descriptor_round40.md`.
  Required `make -j8` reported `no work to do`. Checked-in runtime selector
  covering `cp_no_scales`, `cp_scales`, descriptor-view B-scale/scales copy,
  source initialization, and rematerialization rows collected `270/1615` and
  passed/stably skipped split-4 as `266 passed, 4 skipped`. Checked-in
  structural selector collected `16/33` and stayed stable as `2 passed,
  14 xfailed`. Temporary branch-selected `warpx2::{01_23,02_13}` copy
  descriptor rows all passed with expected `tcgen05.cp` opcodes and correct
  output. Temporary branch-selected linear `128x128b` copy descriptors
  reproduced existing `FZ-20260421-0001` for dynamic/control-flow-carried
  memdesc values consumed by `ttng.tmem_copy`. No new independent `FZ-*`;
  backend repair remains deferred.
  2026-04-21 14:10 UTC Round 40 proxy/TMA sequencing subagent lane integrated.
  Artifact: `agents/fuzz_proxy_tma_sequences_round40.md`. Required `make -j8`
  reported `no work to do`. Checked-in `test_core.py` TMA/multicast/commit/
  mbarrier selector passed `12` rows; runtime-matrix proxy/mbarrier/multicast/
  two-CTA selector collected `322/1615` and passed/stably skipped as
  `285 passed, 37 skipped`; structural selector stayed stable as `1 passed,
  16 xfailed`; TMA-fed MMAv5 shared-input/core selector collected `217/18114`
  and passed/stably skipped as `169 passed, 48 skipped`. Temporary replays
  reproduced existing `FZ-20260421-0014` in three proxy-fence insertion rows,
  existing `FZ-20260421-0001` in six dynamic proxy-view rows, and two existing
  clean descriptor-view boundaries. No new independent `FZ-*`; backend repair
  remains deferred.
  2026-04-21 14:10 UTC Round 40 clean-boundary runtime guardrail completed.
  Artifact: `agents/fuzz_clean_boundary_guardrail_round40.md`. Initial sentinel
  `(clean_unsupported or resource) and not reports` passed `4` rows. Broader
  selector
  `(reports_clean_error or reports_clean_unsupported or reports_tmem_oor or
  reports_resource_boundary) and not cp_no_scales and not cp_scales and not
  mma_twocta_tma` collected `112/1615` rows and passed split-4 as
  `112 passed` (`28/28/28/28`). No compiler crash, diagnostic drift,
  unexpected pass/fail transition, runtime miscompile, or new independent
  `FZ-*`; backend repair remains deferred.
  2026-04-21 14:08 UTC Round 39 `ld.red` extremes subagent lane integrated.
  Artifact: `agents/fuzz_ldred_extremes_round39.md`; subagent commit
  `88e3a31d4`. Required `make -j8` reported `no work to do`. Checked-in
  positive selector `ld_red and not reports and not resource and not m64 and
  not non_f32` collected `160/1615` rows and passed split-4 as `160 passed`.
  Temporary resource/extreme rows reproduced existing `FZ-20260421-0018`,
  clean shared-memory/tensor-memory resource boundaries, and passing hardware
  controls. Half-view rows reproduced existing `FZ-20260421-0020` and
  `FZ-20260421-0022`, plus clean descriptor-view/scalar-message diagnostics
  and passes. No new independent `FZ-*` was proposed. One
  `same_chain_m64_n256_m64_row_reverse_w4` wrong-output signal was explicitly
  left unpromoted pending a smaller descriptor-chain oracle/minimizer.
  2026-04-21 14:12 UTC Round 40 plain MMAv5 runtime guardrail completed.
  Artifact: `agents/fuzz_mmav5_plain_guardrail_round40.md`. Required
  `make -j8` reported `no work to do`. Selector
  `(mma_plain_kinds_use_acc or mma_plain_kinds_with_linear_acc or
  mma_twocta_plain_kinds_use_acc or mma_twocta_plain_kinds) and not scaled and
  not reports and not resource` collected `52/19729` rows and passed split-4
  as `52 passed` (`13/13/13/13`). No compiler crash, false unsupported
  diagnostic, opcode absence, runtime miscompile, unexpected pass/fail
  transition, or new independent `FZ-*`; backend repair remains deferred.
  2026-04-21 14:05 UTC Round 39 MMAv5/TMA runtime guardrail completed.
  Artifact: `agents/fuzz_mmav5_tma_guardrail_round39.md`. Selector
  `mma_twocta_tma_tf32_b_transposed_descriptor or mma_twocta_indexed_acc_view or
  mma_twocta_acc_subslice_view_plain_kinds` collected `54/1615` rows and passed
  split-4 as `54 passed` (`14/14/14/12`). No compiler crash, false unsupported
  diagnostic, opcode absence, runtime miscompile, unexpected pass/fail
  transition, or new independent `FZ-*`; backend repair remains deferred.
  2026-04-21 15:30 UTC Round 39 descriptor/high-rank positive guardrail
  completed. Artifact: `agents/fuzz_descriptor_high_rank_guardrail_round39.md`.
  Selector
  `(descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows) and not reports and not resource and not clean`
  collected `140/1615` and passed/stably skipped split-4 as
  `120 passed, 20 skipped`. No compiler crash, false unsupported diagnostic,
  runtime miscompile, clean-boundary drift, unexpected xfail/pass transition,
  opcode absence signal, or new independent `FZ-*`; backend repair remains
  deferred.
  2026-04-21 15:24 UTC Round 39 scale-layout guardrail completed. Artifact:
  `agents/fuzz_scales_layout_guardrail_round39.md`. Selector
  `(scales_variant or scales_layout or cp_scales_layout_probe or scales_ldst) and not reports and not resource`
  collected `19/1615` and passed as `19 passed`. No opcode drift, false
  unsupported diagnostic, runtime miscompile, compiler crash, unexpected
  pass/fail transition, or new independent `FZ-*`; backend repair remains
  deferred.
  2026-04-21 15:18 UTC Round 39 TMEM lit guardrail completed. Artifact:
  `agents/fuzz_lit_tmem_guardrail_round39.md`. Build-tree `ninja triton-opt`
  was a no-op. Lit passed `tmem_layouts.mlir`, `interleave_tmem.mlir`, and
  `memdesc-subview-split.mlir`. No unexpected IR/diagnostic drift or new
  independent `FZ-*`; backend repair remains deferred.
  2026-04-21 15:12 UTC Round 39 MMAv5 dynamic descriptor/view subagent lane
  integrated. Artifact: `agents/fuzz_mmav5_dynamic_views_round39.md`.
  Checked-in runtime selector collected `190/19729` and passed split-4 as
  `190 passed`; structural selector completed as `2 passed, 3 xfailed`; temp
  dynamic-scale, B-scale dynamic-view, scaled multi-MMA, and high-CGA scaled
  probes found no new independent `FZ-*`. Findings remain under existing
  `FZ-20260421-0001`, `FZ-20260421-0007`, `FZ-20260421-0013`,
  `FZ-20260421-0015`, and `FZ-20260421-0010`; backend repair remains deferred.
  2026-04-21 15:08 UTC Round 39 structural xfail guardrail completed.
  Artifact: `agents/fuzz_structural_xfail_guardrail_round39.md`. Selector
  `generic_pass or descriptor_view or ldred or scaled_mma` over
  `python/test/gluon/test_tmem_structural_fuzzer.py` collected `26/33` and
  completed split-4 as `3 passed, 23 xfailed`. No XPASS, unexpected failure,
  changed clean-boundary behavior, new crash outside expected xfails, or new
  independent `FZ-*`; backend repair remains deferred.
  2026-04-21 15:02 UTC Round 39 TMA/multicast guardrail completed. Artifact:
  `agents/fuzz_tma_multicast_guardrail_round39.md`. Selector
  `tma_multicast or multicast_commit` over `python/test/gluon/test_core.py`
  collected `11/18114` and passed as `11 passed`. No compiler crash, false
  unsupported diagnostic, runtime miscompile, unexpected pass/fail transition,
  or new independent `FZ-*`; backend repair remains deferred.
  2026-04-21 14:55 UTC Round 39 allocation-lifetime guardrail completed.
  Artifact: `agents/fuzz_alloc_lifetime_guardrail_round39.md`. Selector
  `(alloc_lifetime or allocation or source_initialization) and not reports and not resource`
  collected `7/1615` and passed as `7 passed`. No allocator assertion,
  resource drift, runtime miscompile, compiler crash, false unsupported
  diagnostic, unexpected pass/fail transition, or new independent `FZ-*`;
  backend repair remains deferred.
  2026-04-21 14:45 UTC Round 38 copy/scale rematerialization subagent lane
  integrated. Artifact: `agents/fuzz_copy_scale_remat_round38.md`.
  Runtime-matrix copy/scales/remat selector collected `176/1615` and passed
  split-4 as `176 passed`; structural copy-scales assigned rows passed;
  temporary replay probe collected `18` rows and passed split-4 as `18 passed`;
  `test_core.py::test_mma_scaled_tcgen05_copy` collected `96` and passed
  split-4 as `96 passed`. No compiler crash, opcode absence, false unsupported
  diagnostic, runtime miscompile, clean-boundary drift, or new independent
  `FZ-*`; backend repair remains deferred.
  2026-04-21 14:40 UTC Round 38 compiler-boundary subagent lane integrated.
  Artifact: `agents/fuzz_compiler_boundaries_round38.md`. Build-tree lit
  controls `tmem_layouts.mlir`, `interleave_tmem.mlir`, and
  `memdesc-subview-split.mlir` passed. Generated `14` MLIR probes under
  `/tmp/tmem_compiler_boundaries_round38`; matrix over four compiler modes was
  `18` pass, `24` clean diagnostics, `5` late illegal-op failures, and `9`
  abort/crash classifications. No new independent `FZ-*`; existing
  `FZ-20260421-0001`, `FZ-20260421-0016`, and `FZ-20260421-0017` were
  broadened or revalidated; backend repair remains deferred.
  2026-04-21 14:35 UTC Round 38 `test_core.py` TMEM/MMAv5 guardrail
  completed. Artifact: `agents/fuzz_test_core_tmem_guardrail_round38.md`.
  Selector `tcgen05 and (tmem or mma or copy or multicast or mbarrier)` over
  `python/test/gluon/test_core.py` collected `123/18114` and passed/stably
  skipped split-4 as `120 passed, 3 skipped`. No compiler crash, false
  unsupported diagnostic, opcode absence, runtime miscompile, clean-boundary
  drift, unexpected xfail/pass transition, or new independent `FZ-*`; backend
  repair remains deferred.
  2026-04-21 14:30 UTC Round 38 clean-boundary guardrail completed. Artifact:
  `agents/fuzz_clean_boundary_guardrail_round38.md`. Selector
  `(clean_unsupported or clean_error or reports_clean or tmem_oor) and not reports`
  collected `22/1615` and passed as `22 passed`. No verifier
  over-strictness drift, unexpected compiler crash, false unsupported
  diagnostic, unexpected pass/fail transition, or new independent `FZ-*`;
  backend repair remains deferred.
  2026-04-21 14:24 UTC Round 38 copy positive guardrail completed. Artifact:
  `agents/fuzz_copy_positive_guardrail_round38.md`. Selector
  `(cp_no_scales or cp_scales) and not reports and not resource and not clean`
  collected `246/1615` and passed/stably skipped split-4 as
  `242 passed, 4 skipped`. No compiler crash, false unsupported diagnostic,
  opcode absence, runtime miscompile, clean-boundary drift, unexpected
  xfail/pass transition, or new independent `FZ-*`; backend repair remains
  deferred.
  2026-04-21 14:16 UTC Round 37 subword and 64-bit boundary subagent lane
  integrated. Artifact: `agents/fuzz_subword_i64_boundaries_round37.md`.
  Checked-in selector collected `123/1648` and passed split-4 as `123 passed`.
  Temporary subprocess matrix expanded existing `FZ-20260421-0017` to
  unit-rank 1CTA/2CTA `i64`/`f64` descriptor-view non-reduction load/store
  rows; matching `i32` and `f16` controls passed. Non-f32 `ld.red` remains a
  clean f32-only diagnostic and pure `i64`/`f64` shared-to-TMEM copy lowers
  through LLVM. No new independent `FZ-*`; backend repair remains deferred.
  2026-04-21 14:12 UTC Round 37 `ld.red` positive guardrail completed.
  Artifact: `agents/fuzz_ldred_positive_guardrail_round37.md`. Selector
  `ld_red and not reports and not resource and not m64 and not non_f32`
  collected `160/1615` and passed split-4 as `160 passed`. No compiler crash,
  false unsupported diagnostic, opcode absence, runtime miscompile,
  clean-boundary drift, unexpected xfail/pass transition, or new independent
  `FZ-*`; backend repair remains deferred.
  2026-04-21 14:08 UTC Round 37 high-CGA ownership subagent lane integrated.
  Artifact: `agents/fuzz_high_cga_ownership_round37.md`. Checked-in
  high-CGA/multicast selector collected `28` rows and passed split-4 as
  `28 passed`; build-tree `membar-cluster.mlir` lit passed; focused MMAv5,
  scaled-MMAv5, TMA, mbarrier, scale-copy, and shared-scale descriptor
  controls passed. Temporary ownership probe classified `15/15` expected
  diagnostics as existing `FZ-20260421-0010`; no missed `getModuleTwoCTAs`
  propagation bug or new independent `FZ-*`; backend repair remains deferred.
  2026-04-21 14:05 UTC Round 37 load/store positive guardrail completed.
  Artifact: `agents/fuzz_ldst_positive_guardrail_round37.md`. Selector
  `ldst and not reports and not resource and not roundtrip and not clean`
  collected `201/1615` and passed split-4 as `201 passed`. No compiler crash,
  false unsupported diagnostic, opcode absence, runtime miscompile,
  clean-boundary drift, unexpected xfail/pass transition, or new independent
  `FZ-*`; backend repair remains deferred.
  2026-04-21 14:00 UTC Round 37 MMAv5 format/use-acc guardrail completed.
  Artifact: `agents/fuzz_mmav5_format_useacc_guardrail_round37.md`. Selector
  `(root_format or use_acc or plain_kind) and not reports and not resource and not m64`
  collected `255/1615` and passed split-4 as `255 passed`. No compiler crash,
  false unsupported diagnostic, opcode absence, runtime miscompile,
  clean-boundary drift, unexpected xfail/pass transition, or new independent
  `FZ-*`; backend repair remains deferred.
  2026-04-21 13:50 UTC Round 36 indexed-accumulator and subslice guardrail
  completed. Artifact: `agents/fuzz_indexed_subslice_guardrail_round36.md`.
  Selector
  `(indexed_acc or subslice_view or lhs_subslice) and not reports and not resource and not m64`
  collected `269/1615` and passed split-4 as `269 passed`. No compiler crash,
  false unsupported diagnostic, opcode absence, runtime miscompile,
  clean-boundary drift, unexpected xfail/pass transition, or new independent
  `FZ-*`.
  2026-04-21 13:50 UTC Round 36 subagent dynamic/equivalence reports
  integrated. Artifacts: `agents/fuzz_dynamic_proxy_views_round36.md` and
  `agents/fuzz_descriptor_equivalence_round36.md`. Dynamic proxy/mbarrier
  probe found no new proxy-specific bucket (`6` existing `FZ-20260421-0001`,
  `2` clean copy descriptor-view boundaries). Descriptor equivalence oracle
  ran `108` rows and sharpened existing `FZ-20260421-0002` /
  `FZ-20260421-0003`, `FZ-20260421-0004`, `FZ-20260421-0020`, and
  `FZ-20260421-0022`; copy guardrail passed `87` rows. No new independent
  `FZ-*`; backend repair remains deferred.
  2026-04-21 13:45 UTC Round 36 tile-permuted and structural fuzzer guardrail
  completed. Artifact: `agents/fuzz_tile_structural_guardrail_round36.md`.
  Checked-in structural fuzzer stayed stable as `9 passed, 24 xfailed`.
  Selector `tile_permuted and not reports and not resource and not m64`
  collected `151/1615` and passed split-4 as `151 passed`. No compiler crash,
  false unsupported diagnostic, opcode absence, runtime miscompile, xfail
  drift, clean-boundary regression, or new independent `FZ-*`; backend repair
  remains deferred.
  2026-04-21 13:40 UTC Round 36 narrow-shape and clean-diagnostic guardrail
  completed. Artifact: `agents/fuzz_narrow_clean_guardrail_round36.md`.
  Required `make -j8` was a no-op. Selector
  `(x1 or n16 or n32 or clean_unsupported) and not reports and not m64 and not resource`
  collected `329/1615` and passed split-4 as `259 passed, 70 skipped`.
  No compiler crash, false unsupported diagnostic, opcode absence, runtime
  miscompile, clean-boundary drift, unexpected xfail/pass transition, or new
  independent `FZ-*`; backend repair remains deferred.
  2026-04-21 13:33 UTC Round 36 B-scale dynamic descriptor-view runtime lane
  completed. Artifact: `agents/fuzz_bscale_dynamic_views_round36.md`.
  Required `make -j8` was a no-op. Checked-in B-scale/shared-scale selector
  collected `7/1615` and passed split-4 as `7 passed`. Temporary probe
  `/tmp/tmem_bscale_dynamic_views_round36_probe.py` collected `5` valid
  direct/branch/loop/view-chain/extra-user B-scale descriptor rows feeding
  `tcgen05_mma_scaled` and passed split-4 as `5 passed`. No new independent
  `FZ-*`; excluded drafts are clean frontend/API boundaries for scale
  multibuffering and repeated N=32 public-scale MMAv5 rematerialization.
  2026-04-21 Round 36 local high-rank `ld.red` compile/execute classifier
  completed. Artifact: `agents/fuzz_high_rank_ldred_compile_round36.md`.
  Checked-in selector `ld_red and (higher_rank or rank5 or multidim)`
  collected no tests. Temporary subprocess probe
  `/tmp/tmem_high_rank_ldred_round35_probe.py` ran `72` rank-5
  descriptor-view `view.load_min/load_max` consumers and all passed
  compile+execute classification. This is not a runtime-correctness oracle;
  no new independent `FZ-*`.
  2026-04-21 13:35 UTC Round 35 `ld.red` descriptor-view chain lane
  completed. Artifact: `agents/fuzz_ldred_descriptor_views_round35.md`.
  Required `make -j8` was a no-op. Temporary subprocess probe
  `/root/tmp/tmem_ldred_descriptor_views_round35_probe.py` covered direct,
  rank-4 identity, rank-5 identity, half-row, and half-column
  `view.load_min/load_max` consumers across identity, row-reversed,
  column-reversed, and row+column-reversed layouts for `N=64/128`.
  Direct/rank-identity rows passed as `40 pass`; corrected half-view rerun
  classified `8 pass`, `8` clean unsupported descriptor-view diagnostics, `4`
  clean scalar `.x1` diagnostics, `8` existing `FZ-20260421-0022`, and `4`
  existing `FZ-20260421-0020`. Checked-in selector
  `ld_red and descriptor_chain and not non_f32` collected `30/1615` and
  passed split-4 as `30 passed` (`8/8/8/6`). No runtime wrong-result
  miscompile, opcode mismatch, process abort, false unsupported diagnostic, or
  new independent `FZ-*`; backend repair remains deferred.
  2026-04-21 13:10 UTC Round 32 FZ-0018 `ld.red` minimization completed.
  Artifact: `agents/fuzz_ldred_fz0018_min_round32.md`. Required `make -j8`
  was a no-op. Temporary subprocess-isolated runtime probe
  `/tmp/tmem_ldred_fz0018_min_round32.py` classified `60` rows as `16`
  `FZ-20260421-0018`, `12` pass, `6` clean shared-memory boundaries, `6`
  clean tensor-memory boundaries, and `20` clean power-of-two shape
  boundaries. Classification: FZ-0018 is not direct-only because
  same-footprint descriptor-view chains reproduce it; not a generic high-`N`
  impossibility because `M64xN512` and `M128xN512` column-reversed hardware
  `.ld.red` pass; and it generalizes to `M256xN256,w4`, while matching `w8`
  rows cleanly report shared-memory OOR. Backend repair remains deferred.
  2026-04-21 12:49 UTC Round 31 scaled-MMAv5 descriptor operand lane
  completed. Artifact: `agents/fuzz_scaled_operand_round31.md`. No new
  independent `FZ-*`; the lane revalidated `FZ-0013` scale descriptor-view
  miscompiles, `FZ-0015` runtime-selected distinct B-scale miscompiles, and
  `FZ-0010` high-CGA local-layout CTA-count diagnostics. Green controls:
  checked-in scaled descriptor-view selector `4 passed`, dynamic direct
  scale-selection/helper/loop rows `10 passed`, accumulator subslice and
  multi-MMA scale-reuse controls passed, high-CGA scaled-only controls passed
  for `4/8/16` CTAs.
  2026-04-21 Round 31 local guardrails completed. Artifacts:
  `agents/fuzz_local_descriptor_mix_round31.md` and
  `agents/fuzz_local_view_copy_round31.md`. Results: descriptor mix
  `54 passed`; view-chain/copy-`warpx2` `123 passed, 74 skipped`. No new
  bucket.
  2026-04-21 Round 31 Python descriptor-view frontend probing completed.
  Artifact: `agents/fuzz_python_descriptor_views_round31.md`. No new bucket;
  expands existing `FZ-20260421-0017` to valid Python/Gluon descriptor-sliced
  64-bit TMEM load/store kernels. Result: `48` `ASSERT_BITWIDTH_32`, `40`
  passes, and `20` frontend compile diagnostics. Covered legacy/linear
  parents, offsets `0/64/128`, roundtrip/store/load modes, and in-bounds 2CTA
  rows for `torch.float64`/`torch.int64`; matching 1CTA `float32`/`int32`
  controls passed with roundtrip correctness. Backend fixes remain deferred
  while fuzzing continues.
  2026-04-21 Round 31 compiler-boundary fuzzing completed. Artifact:
  `agents/fuzz_compiler_boundaries_round31.md`. No new bucket; expands
  existing `FZ-0001`, `FZ-0016`, and `FZ-0017` across `18` MLIR probes and
  three pass modes. Results: `18` passes, `23` clean diagnostics, `12`
  assertion/stack-dump aborts, and `1` late illegal-op failure.
  2026-04-21 12:57 UTC Round 32 subword/narrow-shape Python runtime TMEM
  fuzzing completed. Artifacts: `agents/fuzz_subword_narrow_round32.md` and
  `agents/fuzz_sub32_python_round32.md`. Required `make -j8` was a no-op.
  Results: subword/copy/diagnostic split-4 sweep `79 passed`; supported
  f32/i32 narrow/control split-4 sweep `60 passed`; sibling temporary
  Python/Gluon dtype roundtrip probe `80 passed`. No new independent `FZ-*`;
  this is a green guardrail around `FZ-0017`, which remains isolated to
  encoded 64-bit non-reduction TMEM load/store lowering under this coverage.
  2026-04-21 Round 32 local `ld/st` narrow and half-row guardrail completed.
  Artifact: `agents/fuzz_local_ldst_narrow_round32.md`. Selector
  `ldst and (n32 or half_rows or x1 or i32_broad) and not reports and not resource and not clean and not roundtrip`
  collected `141/1615` rows and passed split-4 as `141 passed`; no new
  independent `FZ-*`.
  2026-04-21 Round 32 scaled-MMAv5 2CTA runtime guardrail completed.
  Artifact: `agents/fuzz_scaled_twocta_round32.md`. Selector
  `mma_scaled and twocta and not reports and not resource` passed split-4 as
  `28 passed`; no new bucket.
  2026-04-21 Round 31 scales-copy and clean-boundary guardrail completed.
  Artifact: `agents/fuzz_local_scales_copy_clean_round31.md`. Selector
  `(cp_scales or mbarrier or proxy or clean_error or clean_unsupported) and not reports and not resource`
  passed split-4 as `55 passed`; no new bucket.
  2026-04-21 Round 31b compiler-boundary reduction/scale follow-up completed.
  Artifact: `agents/fuzz_compiler_boundaries_round31b.md`. No new bucket;
  dynamic indexed reduction/scale loads expand `FZ-0001`, and unencoded
  reduction load results expand `FZ-0016`.
  2026-04-21 13:18 UTC Round 33 compiler-boundary fuzz lane completed.
  Artifact: `agents/fuzz_compiler_boundaries_round33.md`. No new independent
  `FZ-*`; generated `16` MLIR probes under
  `/tmp/tmem_compiler_boundaries_round33` and ran verify, optimize, and
  allocation+LLVM modes. Result: `10` pass, `18` clean diagnostics, `16`
  assertion/stack-dump aborts, and `4` late illegal-op failures. Existing
  `FZ-0001` dynamic descriptor-index lowering, `FZ-0016` unencoded tensor
  handling, and `FZ-0017` encoded 64-bit TMEM load/store/alloc boundaries were
  revalidated or sharpened; backend repair remains deferred.
  2026-04-21 13:13 UTC Round 33 FZ-0018 PTX artifact capture completed.
  Artifact: `agents/fuzz_ldred_fz0018_ptx_round33.md`.
  `TRITON_REPRODUCER_PATH` produced no MLIR reproducer because the failure is
  at ptxas/cubin time. `TRITON_KERNEL_DUMP=1` preserved failing
  `direct_kernel.{ttgir,llir,ptx}` under `/tmp/tmem_fz0018_dump/...`, and
  standalone `ptxas-blackwell` replay on the dumped PTX reproduces exit `255`.
  Passing dumped controls: `M128xN512` column-reversed hardware `.ld.red`
  assembles with `96` registers/eight `.ld.red` PTX occurrences, and
  `M64xN512` assembles with `255` registers/four `.ld.red` PTX occurrences.
  The bucket remains a large 4-warp hardware-`ld.red`
  resource-planning/ptxas boundary, not a Python harness-only failure.
  2026-04-21 13:11 UTC Round 33 dynamic descriptor SSA/control-flow runtime
  lane completed. Artifact:
  `agents/fuzz_dynamic_descriptor_round33.md`. Required `make -j8` was a
  no-op. Temporary probe `/tmp/tmem_dynamic_descriptor_round33_probe.py`
  collected `100` rows and classified `46` pass, `20` existing
  `FZ-20260421-0001` late illegal `ttg.memdesc_index`, `24` existing
  `FZ-20260421-0002`/`FZ-20260421-0003` descriptor-view wrong-result rows, and
  `10` clean unsupported chain0 copy diagnostics. Plain MMAv5 indexed
  accumulator controls passed as `2 passed`; scaled-MMAv5 low-subslice dynamic
  accumulator sentinel stayed `1 xfailed` under existing `FZ-0007`. No new
  independent `FZ-*`; backend repair remains deferred.
  2026-04-21 Round 34 local no-scale copy tile/subword/2CTA guardrail
  completed. Artifact: `agents/fuzz_local_copy_tile_subword_round34.md`.
  Selector
  `(cp_no_scales and not reports and not resource and (tile_permuted or subword or twocta_codegen or twocta_128x128b or dense_shared))`
  collected `68/1615` and passed split-4 as `68 passed`; no compiler crash,
  false unsupported diagnostic, opcode mismatch, runtime miscompile,
  clean-boundary drift, or new independent `FZ-*`.
  2026-04-21 Round 34 local high-rank descriptor `ld/st` guardrail completed.
  Artifact: `agents/fuzz_local_high_rank_ldst_round34.md`. Selector
  `(ldst_descriptor and not reports and not resource and (rank5 or higher_rank or multidim or roundtrip))`
  collected `104/1615` and completed split-4 as `43 passed, 61 skipped`; no
  compiler crash, false unsupported diagnostic, opcode mismatch, runtime
  miscompile, clean-boundary drift, unexpected skip/pass transition, or new
  independent `FZ-*`.
  2026-04-21 Round 34 copy `warpx2`/`warpx4` runtime guardrail completed.
  Artifact: `agents/fuzz_local_copy_warpx2_round34.md`. Selector
  `(cp_no_scales and (warpx2 or warpx4 or dense_shared or twocta_128x128b) and not resource)`
  collected `83/1615` and passed split-4 as `83 passed`; no compiler crash,
  false unsupported diagnostic, opcode mismatch, runtime miscompile,
  clean-boundary drift, unexpected xfail/pass transition, or new independent
  `FZ-*`.
  2026-04-21 Round 34 copy `warpx2` compiler-probe attempt completed.
  Artifact: `agents/fuzz_copy_compiler_probe_round34.md`. Positive two-CTA
  `warpx2::01_23` runtime row passed and dumped TTGIR/LLIR/PTX/cubin/SASS;
  PTX contains `tcgen05.cp.cta_group::2.warpx2::01_23.64x128b`. The two-CTA
  `02_13` clean-unsupported row passed its checked-in diagnostic assertion.
  Hand-built standalone `triton-opt` allocation+LLVM lowering over the dumped
  positive TTGIR segfaulted in `TensorMemoryAllocOpConversion`, but the JIT
  path compiles and runs the same row. Classified as a compiler-probe harness
  limitation, not a new independent `FZ-*`, unless a faithful JIT-equivalent
  pipeline reproduces it.
  2026-04-21 Round 34 broad scaled-MMAv5 runtime guardrail completed.
  Artifact: `agents/fuzz_local_mma_scaled_broad_round34.md`. Selector
  `mma_scaled and not reports and not resource and not clean` collected
  `243/1615` and passed split-4 as `243 passed`; no compiler crash, false
  unsupported diagnostic, opcode mismatch, runtime miscompile,
  clean-boundary drift, or new independent `FZ-*`.
  2026-04-21 Round 34 local `test_core.py` TMEM/multicast guardrail completed.
  Artifact: `agents/fuzz_local_test_core_tmem_round34.md`. Selector
  `(tcgen05 or tmem) and (multicast or shared_inputs or twocta or mma)`
  collected `122/18114` and completed split-4 as `119 passed, 3 skipped`; no
  compiler crash, false unsupported diagnostic, opcode mismatch, runtime
  miscompile, clean-boundary drift, unexpected skip/pass transition, or new
  independent `FZ-*`.
  2026-04-21 Round 34 local two-CTA MMAv5/TMA guardrail completed. Artifact:
  `agents/fuzz_local_mma_twocta_tma_round34.md`. Selector
  `(mma_twocta and not reports and not resource and (tma or transposed or plain_kind or indexed_acc))`
  collected `103/1615` and passed split-4 as `103 passed`; no compiler crash,
  false unsupported diagnostic, opcode mismatch, runtime miscompile,
  clean-boundary drift, or new independent `FZ-*`.
  2026-04-21 11:21 UTC Lane AI completed copy/mbarrier composition fuzzing
  without backend repairs. Report:
  `agents/fuzz_copy_mbarrier_composition_round14.md`. Checked-in copy baseline
  `cp_no_scales and (warpx2 or twocta) and not reports` passed split-4 as
  `103 passed`; temporary fresh-subprocess rows classified as `4` green
  controls, `2` clean `FZ-20260421-0010` context diagnostics, and `2`
  `FZ-20260421-0014` proxy-fence crashes. `FZ-0014` is broadened to multiple
  independent legal 2CTA no-scales copy/mbarrier regions, direct or
  descriptor-chain, without scales copy. No runtime miscompile and no new
  independent `FZ-*` bucket was found.
  2026-04-21 11:23 UTC Lane AJ completed high-CGA mixed ownership fuzzing
  without backend repairs. Report:
  `agents/fuzz_high_cga_mixed_ownership_round14.md`. Required `make -j8` was
  a no-op. Local 1CTA/2CTA `ld/st`, `ld.red`, and copy rows in 4/8/16 CTA
  launch contexts stayed under existing `FZ-20260421-0010`; the legal 2CTA
  mixed copy/scales row and full saved reproducer stayed under existing
  `FZ-20260421-0014`. Adjacent high-CGA MMAv5/TMA-MMA/scaled-MMAv5 controls
  passed `9/9`; `triton-nvidia-check-matmul-two-cta` controls propagated
  `"ttng.two-ctas" = true` for 8/16 CTA consistent modules. No new
  independent `FZ-*` bucket was assigned, and backend repair remains deferred.
  2026-04-21 11:24 UTC local Round 14 twoCTA/high-CGA runtime sweep stayed
  green. Report: `agents/fuzz_local_twocta_highcga_round14.md`.
  Runtime-matrix selector
  `(twocta or cga_roundtrip or layout_in_4cta_context) and not reports`
  collected `331/1615` rows and passed split-4 as
  `294 passed, 37 skipped`. Adjacent high-CGA `test_core.py` controls reported
  `3 passed`. No runtime miscompile, compiler crash, false unsupported
  diagnostic, or new independent `FZ-*` bucket was found.
  2026-04-21 11:26 UTC local Round 15 higher-rank descriptor runtime sweep
  stayed green. Report:
  `agents/fuzz_local_higher_rank_descriptor_round15.md`. Runtime-matrix
  selector `(higher_rank or multidim_slice or half_rows or rank5) and not reports`
  collected `102/1615` rows and passed split-4 as `82 passed, 20 skipped`. No
  runtime miscompile, compiler crash, false unsupported diagnostic, or new
  independent `FZ-*` bucket was found.
  2026-04-21 11:28 UTC Round 15 descriptor/control and `ld.red` semantics
  checkpoint stayed green or mapped to existing buckets. Reports:
  `agents/fuzz_local_descriptor_control_round15.md` and
  `agents/fuzz_ldred_semantics_round15.md`. The descriptor/control selector
  collected `329/1615` rows and passed split-4 as `251 passed, 78 skipped`.
  Lane AL's temporary `ld.red` semantic probe passed `16/16`; checked-in
  `ld.red` matrix failures remained known `FZ-20260421-0012`, and structural
  sentinels stayed under existing `FZ-20260421-0004/0006/0008/0009`. No new
  independent `FZ-*` bucket was found.
  2026-04-21 08:30 UTC Lane E2 promoted FZ-20260421-0001 and
  FZ-20260421-0002 into checked-in strict xfail runtime coverage in
  `python/test/gluon/test_tmem_structural_fuzzer.py`, with exact fresh-process
  validation recorded in `agents/fuzz_generic_pass_round2.md`. Backend repair
  remains deferred.
  2026-04-21 09:10 UTC Lane A2 promoted FZ-20260421-0003 through
  FZ-20260421-0006 into checked-in strict xfail coverage in the same
  structural fuzzer: ld/st descriptor-chain miscompile, ld.red plain-load
  opcode fallback, 256-row lifted-parent allocator assertion via subprocess,
  and transpose/slice ld.red false-unsupported candidate. Exact nodeids
  reported `4 xfailed`; the full structural fuzzer reported
  `9 passed, 9 xfailed`. Backend repair remains deferred.
  2026-04-21 09:20 UTC promoted the Round 5 generic-pass loop-carried
  memdesc-view crash into checked-in strict xfail coverage:
  `generic-pass-loop-carried-memdesc-view-chain0`. This covers the
  `GluonResolveAutoEncodingsPass` / auto-layout inference failure while
  keeping discovery mode repair-deferred. Validation: required `make -j8`,
  py-compile, `30` collected structural-fuzzer nodeids, exact nodeid
  `1 xfailed`, and full fuzzer `9 passed, 21 xfailed`.
  2026-04-21 Round 6 Lane D prototyped a deterministic structural descriptor
  generator under `/tmp` and kept it out of the repo pending schema cleanup.
  The inventory generated `25245` normalized descriptors across `ldst`,
  `ldred`, `copy`, `mma`, and `mma_scaled`, with `0` exact normalized-id
  matches against the current `30` checked-in structural-fuzzer cases. Next
  generator slice should add a repo-local data-only `StructuralCaseDescriptor`
  plus `legacy_case_id` bridge before promoting runnable family adapters.
  Validation: required `make -j8` no-op, collect-only `30` nodeids, and small
  runtime sample `3 passed, 1 xfailed`.
  2026-04-21 Round 6 Lane B found `FZ-20260421-0007`, a report-only
  scaled-MMAv5 `use_acc` miscompile for a low-column accumulator subslice
  selected through dynamic `if`. Focused existing scaled-MMAv5 and plain-MMAv5
  matrix selectors remained green, so the next promotion slice should add a
  self-contained strict xfail to `test_tmem_structural_fuzzer.py` for the
  dynamic-if low-subslice boundary before repair work begins.
  2026-04-21 Round 6 Lane C isolated `FZ-20260421-0008`, a report-only
  two-CTA indexed `ld.red` chain1 row/col optimizer crash in
  `TritonNvidiaGPUOptimizeTMemLayoutsPass`. The minimized crash row is parent
  `[2,256,2]`, selected view `[256,2]`, row `even_odd`, col `identity`, chain
  `index(1).reshape((128,2,2)).permute([1,0,2]).reshape((256,2))`. It
  reproduces in a subprocess Python child and as an extracted MLIR
  `triton-opt --run-reproducer` abort; plain `ld` fallback and clean
  unsupported controls are cataloged separately. No backend/compiler repair
  was attempted.
  2026-04-21 09:08 UTC promoted `FZ-20260421-0007` and
  `FZ-20260421-0008` into checked-in structural-fuzzer strict xfails. The
  scaled-MMAv5 dynamic-if low-subslice row is an in-process expected
  miscompile sentinel; the `ld.red` row/col optimizer abort is isolated in a
  subprocess so the parent pytest process survives. Validation: required
  `make -j8`, py-compile, `32` collected structural-fuzzer nodeids, exact
  sentinels `2 xfailed`, and full fuzzer `9 passed, 23 xfailed`.
  2026-04-21 09:11 UTC Round 7 Lane D completed generic-pass / analysis
  interaction discovery without backend repairs. Tuple-like memdesc+tensor
  returns, nested helper-selected memdesc values, loop-carried memdesc+tensor
  iter args, chain2 loop-carried slice/slice views, and multi-live sibling
  views with non-TMEM tensor side inputs all classify under the existing R5-C
  `GluonResolveAutoEncodingsPass` auto-layout crash. A separate
  `16x128b` layout-conversion/non-TMEM tensor row extends
  `FZ-20260421-0002` with a fresh `8063 / 8192` runtime mismatch. No new
  independent FZ id was assigned. Report:
  `agents/fuzz_generic_analysis_round7.md`.
  2026-04-21 09:12 UTC Round 7 Lane B completed generator-backed `ld/st` and
  `ld.red` discovery without backend repairs. The Round 6 `/tmp` descriptor
  generator selected eight compact rows. New `FZ-20260421-0009`: 1CTA direct
  indexed `ld.red` over parent `[2,256,32]` aborts in
  `TritonTensorMemoryAllocationPass` with the allocator assertion
  `kNumRows - numRows >= 0`; extracted MLIR replay exits `134`. Positive and
  boundary rows confirmed `128x64` indexed `.ld.red.`, existing
  `FZ-20260421-0004` 2CTA `256x64` opcode fallback, already-covered
  `FZ-20260421-0003` f16 subword read-only miscompile, and clean unsupported
  descriptor-view diagnostics. Report:
  `agents/fuzz_generator_ldst_ldred_round7.md`.
  2026-04-21 09:13 UTC Round 7 Lane C completed copy/readback discovery
  without backend repairs. `/tmp/tmem_copy_readback_round7_probe.py` covered
  no-scales `warpx2` descriptor-view copy/readback positives,
  `warpx2::02_13` clean unsupported diagnostics, packed/subword clean
  diagnostics, descriptor-chain `ld.red` readback, scales `warpx4`, and the
  two-CTA-layout/four-CTA-context clean error. No stable new
  backend/compiler failure was found and no `FZ-*` id was assigned. Report:
  `agents/fuzz_copy_readback_round7.md`. Validation: required `make -j8`
  no-op, probe py-compile, `/tmp` launcher six cases passed, and four-GPU
  selector sweep passed `28` selected tests per group.
  2026-04-21 Round 7 Lane A completed scaled-MMAv5 accumulator control-flow
  expansion without backend repairs. The temporary harness
  `/tmp/tmem_scaled_mma_controlflow_round7_probe.py` collected `410` nodeids
  and classified a broad `FZ-20260421-0007` expansion: direct low/high and
  high-selector dynamic rows pass, while low-selector dynamic `if`, helper,
  loop, and indexed rows miscompile across the probed `N`, `K`, and feasible
  scaled-format cells. `indexed_helper` exceptions overlap
  `FZ-20260421-0001`; no new independent `FZ-*` id was assigned. Report:
  `agents/fuzz_scaled_mma_controlflow_round7.md`.
  2026-04-21 promoted Round 7 Lane B `FZ-20260421-0009` into checked-in
  subprocess-isolated strict xfail coverage:
  `test_tmem_structural_fuzzer_ldred_1cta_direct_index_allocator_crash`.
  This covers the 1CTA direct indexed `ld.red` parent `[2,256,32]`
  `TritonTensorMemoryAllocationPass` allocator assertion while keeping the
  parent pytest process alive. Validation: required `make -j8`, py-compile,
  `33` collected structural-fuzzer nodeids, exact sentinel `1 xfailed`, and
  full fuzzer `9 passed, 24 xfailed`. Backend repair remains deferred.
  2026-04-21 Round 8 Lane C completed copy/readback generator-adapter
  discovery without backend repairs. Positive no-scales/scales copy/readback
  and descriptor-chain `ld.red` readback rows stayed positive; two-CTA
  `warpx2::02_13`, packed/subword, larger-CGA two-CTA-layout, and scales
  descriptor-view copy rows stayed clean diagnostics. No new `FZ-*` id.
  Validation: required `make -j8`, py-compile, `18` collected nodeids, and
  full probe `18 passed`. Report:
  `agents/fuzz_copy_readback_round8.md`.
  2026-04-21 Round 8 Lane D completed generic-pass / analysis interaction
  discovery without backend repairs. No new independent `FZ-*` id. Direct
  chain0 descriptor-view mismatches dominate several generic-looking helper,
  tuple, sibling, static-loop, and same-base dynamic-if rows, so they extend
  `FZ-20260421-0003`; runtime direct `memdesc_index` still extends
  `FZ-20260421-0001`. Chain1/chain2 generic controls reached runtime and
  passed. Validation: required `make -j8`, py-compile, `14` collected
  nodeids, split-4 classified sweep `7 failed, 7 passed`, and fresh exact
  confirmations. Report: `agents/fuzz_generic_pass_round8.md`.
  2026-04-21 Round 8 Lane A completed `ld.red` allocator/opcode discovery
  without backend repairs. No new independent `FZ-*` id. The lane broadened
  `FZ-20260421-0004` with 18 two-CTA indexed opcode fallback rows,
  `FZ-20260421-0008` with row-chain optimizer aborts across
  `N={2,16,32,64}`, and `FZ-20260421-0005/0009` with direct indexed
  256-row allocator failures for both `ld.red` and `ld/st` dtype variants.
  Positive 128-row indexed/chained controls stayed green. Validation:
  required `make -j8`, py-compile, 55 subprocess-isolated rows, exact fresh
  confirmations, and `git diff --check`. Report:
  `agents/fuzz_ldred_allocator_opcode_round8.md`.
  2026-04-21 Round 9 Lane E completed normalized generator inventory and
  follow-up `ld/st` inventory-hole probing without backend repairs. The
  `/tmp` generator emitted `44,043` descriptors and linked all `33`
  checked-in structural-fuzzer legacy ids; `18 / 46` generated axis buckets
  have legacy coverage. The `ld/st` hole probe found no new owner: chain1
  rows extend `FZ-20260421-0003`, one small transpose/slice row is a clean
  row-anchor diagnostic, and positive controls pass. Reports:
  `agents/fuzz_generator_inventory_round9.md` and
  `agents/fuzz_ldst_inventory_holes_round9.md`.
  2026-04-21 Round 9 Lane F completed clean-diagnostic adversarial fuzzing
  without backend repairs. No new independent `FZ-*` id. Generated rows
  classified as eight clean diagnostics, one positive control, one
  `FZ-20260421-0001` late illegal-op row, and four allocator assertions
  extending `FZ-20260421-0005/0009`; exact copy-boundary runtime-matrix rows
  passed `4/4`. Report: `agents/fuzz_clean_diagnostics_round9.md`.
  2026-04-21 Round 8 Lane B completed scaled-MMAv5 accumulator
  descriptor-view control-flow fuzzing without backend repairs. No new
  independent bucket. `FZ-20260421-0007` expands to `nvfp4xnvfp4`, `N=16`,
  `K=256`, selector-1 dynamic slice/indexed accumulator views, including
  `use_acc=False`; direct controls pass and MMA opcodes agree. Helper-indexed
  rows overlap `FZ-20260421-0001`. Report:
  `agents/fuzz_scaled_mma_round8.md`.
  2026-04-21 Round 9 Lane H completed validation logistics without backend
  repairs. No new `FZ-*` id. Use the structural fuzzer as the frequent smoke
  gate (`33` nodeids, current expected `9 passed, 24 xfailed`). Prefer
  focused runtime slices for repeated lanes: scaled MMA `use_acc` (`28`),
  copy `warpx2` (`28`), LD.RED descriptor (`30`), and clean diagnostics
  (`96`), with stored durations and least-duration split-4 reruns. Report:
  `agents/fuzz_validation_logistics_round9.md`.
  2026-04-21 09:44 UTC local Round 10 structural-fuzzer smoke gate remains
  green while Lane I and Lane J fuzzing are active. Required `make -j8` no-op;
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py`
  reported `9 passed, 24 xfailed in 24.16s`.
  2026-04-21 local Round 10 clean-diagnostic runtime slice stayed green:
  `reports_clean_unsupported` ran split-4 across GPUs 0-3 with stable caches,
  each shard selected `23` tests, and the aggregate result was `92 passed`.
  Durations were stored at
  `/tmp/tmem_local_r10_clean_diagnostics_durations.json`.
  2026-04-21 local Round 10 copy `warpx2` runtime slice stayed green:
  `cp_no_scales and warpx2` ran split-4 across GPUs 0-3 with stable caches,
  selected `20/20/20/19` tests, and passed all `79`. Durations were stored at
  `/tmp/tmem_local_r10_copy_warpx2_durations.json`.
  2026-04-21 local Round 10 scaled-MMA `use_acc` runtime slice stayed green:
  collect-only selected `83/1615`, split-4 across GPUs 0-3 selected
  `21/21/21/20` tests, and all `83` passed. Durations were stored at
  `/tmp/tmem_local_r10_scaled_use_acc_durations.json`.
  2026-04-21 local Round 10 higher-rank/rank-5 descriptor runtime slice stayed
  green: collect-only selected `82/1615`, split-4 across GPUs 0-3 selected
  `21/21/21/19` tests, and the aggregate result was
  `62 passed, 20 skipped`. Durations were stored at
  `/tmp/tmem_local_r10_rank5_higher_rank_durations.json`.
  2026-04-21 local Round 10 `ld/st` row/column descriptor runtime slice stayed
  green: collect-only selected `32/1615`, split-4 across GPUs 0-3 selected
  `8/8/8/8` tests, and the aggregate result was `8 passed, 24 skipped`.
  Durations were stored at
  `/tmp/tmem_local_r10_ldst_rowcol_descriptor_durations.json`.
  2026-04-21 Round 10 Lane J completed descriptor-view composition-depth
  fuzzing without backend repairs. No new independent `FZ-*` id. The temporary
  probe found `7` passes and `3` clean diagnostics: rank-4/rank-5 `ld/st`
  unsupported direct-layout materialization and rank-5 sibling-view `ld.red`
  tensor-memory OOR (`2048` required, hardware limit `512`). Report:
  `agents/fuzz_descriptor_depth_round10.md`.
  2026-04-21 Round 10 Lane I completed `ld.red` modifier/NaN edge fuzzing
  without backend repairs. No new independent `FZ-*` id. The `266`-row
  subprocess sweep reclassified to `140` pass, `70` opcode fallback rows
  broadening `FZ-20260421-0004`, `28` optimizer abort rows broadening
  `FZ-20260421-0008`, and `28` clean OutOfResources diagnostics. No runtime
  miscompile or allocator assertion. Report:
  `agents/fuzz_ldred_modifiers_round10.md`.
  2026-04-21 local Round 10 broad non-rowcol `ld/st` descriptor slice stayed
  green: collect-only selected `162/1615`, split-4 across GPUs 0-3 selected
  `41/41/41/39` tests, and the aggregate result was `108 passed, 54 skipped`.
  Durations were stored at
  `/tmp/tmem_local_r10_ldst_descriptor_broad_durations.json`.
  2026-04-21 Round 10 Lane K completed allocator/resource-boundary fuzzing
  without backend repairs. No new independent `FZ-*` id. Fresh probes mapped
  to existing allocator/compiler-crash buckets `FZ-20260421-0005/0009`, f16
  descriptor-chain miscompile `FZ-20260421-0003`, 2CTA indexed `ld.red`
  fallback `FZ-20260421-0004`, and clean copy/scaled-MMA boundaries. Report:
  `agents/fuzz_allocator_resource_round10.md`.
  2026-04-21 local Round 10 `cp_scales` runtime slice stayed green:
  collect-only selected `34/1615`, split-4 across GPUs 0-3 selected
  `9/9/9/7` tests, and all `34` passed. Durations were stored at
  `/tmp/tmem_local_r10_cp_scales_durations.json`.
  2026-04-21 09:58 UTC post-integration structural-fuzzer smoke gate remains
  green after Lane I/J/K integration: required `make -j8` no-op, and
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py`
  reported `9 passed, 24 xfailed in 8.16s`.
  2026-04-21 local Round 10 plain-MMAv5 descriptor slice stayed green:
  collect-only selected `109/1615`, split-4 across GPUs 0-3 selected
  `28/28/28/25` tests, and all `109` passed. Durations were stored at
  `/tmp/tmem_local_r10_plain_mma_desc_durations.json`.
  2026-04-21 local Round 10 lit sanity stayed green: `ninja triton-opt` no-op
  and `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
  passed `3/3`.
  2026-04-21 Round 10 Lane L completed generic pass/control-flow descriptor
  fuzzing without backend repairs. No new independent `FZ-*` id. The exact
  four-GPU checked-in-case rerun produced `7` passes, `1` clean unsupported
  diagnostic, and `20` failures owned by existing buckets
  `FZ-20260421-0001`, `FZ-20260421-0002`, `FZ-20260421-0003`,
  `FZ-20260421-0004`, `FZ-20260421-0006`, `FZ-20260421-0007`, and `R5-C`.
  Report: `agents/fuzz_generic_controlflow_round10.md`.
  2026-04-21 Round 10 Lane M completed copy/scales/CGA fuzzing without
  backend repairs. No new `FZ-*` id. Positive scales `warpx4`, no-scales
  `warpx2`, and scaled-MMA copy setup rows stayed positive; descriptor-view,
  two-CTA `warpx2::02_13`, subword, and larger-CGA rows stayed clean
  boundaries. The `num_ctas > 2` scaled-MMA shape assertion is an early
  guardrail. Report: `agents/fuzz_copy_scales_cga_round10.md`.
  2026-04-21 local Round 10 clean-unsupported selector stayed green with
  duration-aware splitting: collect-only selected `92/1615`, split-4
  least-duration across GPUs 0-3 selected `19/25/24/24` tests, and all `92`
  passed using `/tmp/tmem_local_r10_clean_diagnostics_durations.json`.
  2026-04-21 local Round 10 checked-in `ld.red` NaN/modifier selector stayed
  green: collect-only selected `176/1615`, split-4 across GPUs 0-3 selected
  `44/44/44/44` tests, and all `176` passed. Durations were stored at
  `/tmp/tmem_local_r10_ldred_nan_durations.json`.
  2026-04-21 Round 10 Lane O completed opcode/IR consistency fuzzing without
  backend repairs. No new independent `FZ-*` id and no PTX-vs-LLIR opcode
  disagreement were found across sampled positive `ld`, `st`, `ld.red`, `cp`,
  plain MMA, and scaled-MMA rows. Matching expected opcodes included direct
  and descriptor-chain `ld/st`, direct `ld.red` min/max.abs.NaN, no-scales
  and scaled-copy `cp`, plain MMAv5, and 1CTA/2CTA scaled-MMAv5. The only bug
  overlap was the known indexed 256x32 `ld.red` plain-load fallback under
  `FZ-20260421-0004`; the exotic copy layout remained a clean unsupported
  boundary. Report: `agents/fuzz_opcode_consistency_round10.md`.
  2026-04-21 local Round 11 no-scales copy non-`warpx2` runtime slice stayed
  green. Required `make -j8` no-op. Collect-only selected `169/1615` for
  `cp_no_scales and not warpx2 and not reports`; split-4 across GPUs 0-3 with
  stable caches selected `43/43/43/40` tests and reported aggregate
  `165 passed, 4 skipped`. Durations were stored at
  `/tmp/tmem_local_r11_cp_no_scales_nonwarpx2_durations.json`.
  2026-04-21 Round 10 Lane N completed MMAv5/scaled-MMAv5 dynamic descriptor
  selection fuzzing without backend repairs. New report-only candidate
  `FZ-20260421-0011`: plain MMAv5 runtime-selector-index accumulator rows
  compile, keep PTX/LLIR MMA opcode agreement, and miscompile at runtime with
  NaN-heavy mismatches (`~8150/8192` for `N=64,K=128`). The temporary harness
  did not show the `FZ-20260421-0001` illegal `ttg.memdesc_index` signature,
  but a first checked-in minimization attempt did hit `FZ-0001` and was not
  committed. Scaled rows expand `FZ-20260421-0007`; exact controls stayed
  green. Report: `agents/fuzz_mma_dynamic_round10.md`.
  2026-04-21 Round 10 Lane P completed clean-boundary adversarial fuzzing
  without backend repairs. No new independent `FZ-*` id. Most perturbations
  stayed pass or clean-diagnostic across copy OOR, explicit subword, `warpx2`,
  scales descriptor-view, scaled-MMA accumulator-layout, CTA/CGA mismatch, and
  ld.red transpose/slice boundaries. `ld_red_identity_n512` overlaps the
  existing `FZ-20260421-0005/0009` ld.red resource/crash family. Report:
  `agents/fuzz_clean_boundary_round10.md`.
  2026-04-21 Round 11 Lane Q completed multi-CTA/CGA structural fuzzing
  without backend repairs. New independent candidate `FZ-20260421-0010`:
  1CTA/2CTA TMEM linear/scales layouts are rejected in 4/8/16 CTA launch
  contexts with `Layout has X CTAs per CGA, but the context requires Y CTAs
  per CGA`, even though the rows are local 1CTA/2CTA TMEM operations and
  checked-in high-CGA MMA/TMA-MMA controls passed as `136 passed, 12 skipped,
  74 deselected`. This points at an over-strict layout CTA-count gate that
  conflates kernel CGA shape with instruction-local `cta_group`. Report:
  `agents/fuzz_multicta_cga_round11.md`.
  2026-04-21 local post-report structural-fuzzer smoke gate stayed green:
  required `make -j8` no-op, and
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py`
  reported `9 passed, 24 xfailed in 8.70s`.
  2026-04-21 local Round 12 non-f32 `ld.red` descriptor-chain slice stayed
  green. Collect-only selected `20/1615` for
  `ld_red and descriptor and not propagate_nan and not reports`; split-4
  across GPUs 0-3 selected `5/5/5/5` tests and reported aggregate
  `20 passed`. Durations were stored at
  `/tmp/tmem_local_r12_ldred_nonf32_descriptor_durations.json`.
  2026-04-21 local Round 12 scaled-MMA accumulator-subslice selector stayed
  green. Collect-only selected `74/1615` for
  `mma_scaled and acc_subslice_view and not reports`; split-4 across GPUs 0-3
  selected `19/19/19/17` tests and reported aggregate `74 passed`. Durations
  were stored at `/tmp/tmem_local_r12_scaled_acc_subslice_durations.json`.
  2026-04-21 local Round 12 lit sanity stayed green: `ninja triton-opt` no-op
  and `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
  passed `3/3`.
  2026-04-21 Round 12 Lane T completed cache/process stability fuzzing without
  backend repairs. No new cache/process/replay instability candidate. Stable
  same-process, immediate repeat, fresh subprocess, and fresh diagnostic-cache
  contrasts all matched `7 passed, 5 xfailed`; bad rows stayed in known
  buckets and green controls stayed green. Report:
  `agents/fuzz_cache_process_round12.md`.
  2026-04-21 Round 12 Lane R completed high-CGA gate minimization without
  backend repairs. `FZ-20260421-0010` is reproduced across `18` minimized
  rows: 1CTA/2CTA linear `ld/st`, direct `ld.red`, no-scales copy, and
  scales-copy layout-construction diagnostics under 4/8/16 CTA launch
  contexts. Passing high-CGA MMA controls stayed green as `2 passed in
  3.18s`. Report: `agents/fuzz_high_cga_gate_round12.md`.
  2026-04-21 Round 12 Lane S completed plain-MMAv5 runtime-index reduction
  without backend repairs. `FZ-20260421-0011` is FPSAN-specific: the
  `N=32,K=128,selector=1,use_acc=False` row repeated `3/3` fresh subprocesses
  with stable `4081/4096` mismatches under FPSAN, while the non-FPSAN
  lifted-linear runtime-index form falls back to known `FZ-20260421-0001`
  illegal `ttg.memdesc_index` lowering. Report:
  `agents/fuzz_plain_mma_runtime_index_round12.md`.
  2026-04-21 local high-CGA MMA contrast controls stayed green after required
  `make -j8` no-op: `test_tcgen05_mma_multicast_commit[False-ctas_per_cga1]`
  and `[True-ctas_per_cga2]` reported `2 passed in 3.05s`.
  2026-04-21 local Round 12 `cp_scales` non-report selector stayed green.
  Collect-only selected `33/1615` for `cp_scales and not reports`; split-4
  across GPUs 0-3 selected `9/9/9/6` tests and reported aggregate
  `33 passed`. Durations were stored at
  `/tmp/tmem_local_r12_cp_scales_nonreports_durations.json`.
  2026-04-21 Round 12 Lane V completed copy descriptor/addressing fuzzing
  without backend repairs. No new independent `FZ-*`. Checked-in copy/scales
  runtime slice selected `128/1615` and passed `128/128`; fresh-process probe
  classified `6` pass, `3` `FZ-20260421-0010` high-CGA no-scales copy rows,
  and one harness capture limitation later confirmed clean. Report:
  `agents/fuzz_copy_descriptor_round12.md`.
  2026-04-21 local exact structural xfail sample stayed stable:
  `FZ-20260421-0004`, `FZ-20260421-0009`, and `FZ-20260421-0007` exact nodeids
  reported `3 xfailed in 4.59s`.
  2026-04-21 local clean-boundary selector replay stayed green:
  `reports_clean_unsupported` passed split-4 across GPUs 0-3 with stable
  caches as `19/25/24/24`, aggregate `92 passed`; durations reused from
  `/tmp/tmem_local_r10_clean_diagnostics_durations.json`.
  2026-04-21 Round 12 Lane U completed ld.red row/column/opcode fuzzing
  without backend repairs. No new independent `FZ-*` and no runtime
  miscompile. Probe total: `94` rows classified as `42` pass, `19`
  `FZ-20260421-0004`, `14` `FZ-20260421-0008`, `13`
  `FZ-20260421-0005/0009`, and `6` clean TMEM OOR boundaries. Report:
  `agents/fuzz_ldred_rowcol_round12.md`.
  2026-04-21 Round 12 Lane W completed generic descriptor-view pass fuzzing
  without backend repairs. No new independent `FZ-*`. Checked-in generic-pass
  rows stayed as `11 xfailed`; the temporary probe classified as `7` pass and
  `2` known failures. Runtime `memdesc_index` remains `FZ-20260421-0001`;
  chain0 generic-pass/control-flow wrong results remain `FZ-20260421-0002`.
  The loop-carried checked-in row currently presents as an `8064/8192`
  runtime mismatch, not the historical `R5-C` auto-layout crash. Report:
  `agents/fuzz_generic_views_round12.md`.
  2026-04-21 post Lane U/W structural fuzzer smoke gate stayed stable after
  required `make -j8` no-op:
  `python/test/gluon/test_tmem_structural_fuzzer.py` reported
  `9 passed, 24 xfailed in 8.69s`.
  2026-04-21 local allocation/lifetime selector collected `7/1615` and passed
  `7 passed, 1608 deselected` after required `make -j8` no-op.
  2026-04-21 local two-CTA/multicast-adjacent selector collected `370/1615`
  and passed split-4 across GPUs 0-3 with stable caches as `333 passed,
  37 skipped`; durations stored at
  `/tmp/tmem_local_r13_twocta_multicast_durations.json`.
  2026-04-21 Round 13 Lane X completed allocation/lifetime/commit/barrier
  fuzzing without backend repairs. No new independent `FZ-*` and no runtime
  miscompile. Probe total: `28` rows classified as `18` pass, `4`
  `FZ-20260421-0010`, `2` clean TMEM OOR, `2` clean copy packed-lane
  unsupported, and `2` harness/shared-layout setup limitations. Report:
  `agents/fuzz_alloc_lifetime_round13.md`.
  2026-04-21 local rank5/higher-rank descriptor selector collected `82/1615`
  and passed split-4 across GPUs 0-3 with stable caches as `62 passed,
  20 skipped`; durations stored at
  `/tmp/tmem_local_r13_rank5_higher_rank_durations.json`.
  2026-04-21 local lifetime/mbarrier sanity report found no new `FZ-*`: core
  commit/mbarrier rows passed `13/13`, runtime-matrix allocation lifetime rows
  passed `7/7`, and descriptor-chain/physical-bitcast rows passed `29/29`.
  Report: `agents/fuzz_lifetime_mbarrier_round13.md`.
  2026-04-21 local ConSan TMEM instrumentation report found no new `FZ-*`:
  `python/test/gluon/test_consan.py -k 'tmem or tcgen05 or tensor_memory'`
  collected `36/348` and passed split-4 as `30 passed, 6 skipped`. Report:
  `agents/fuzz_consan_tmem_round13.md`.
  2026-04-21 checked-in FPSAN MMAv5/scaled-MMAv5 controls collected `37/104`
  and passed split-4 as `32 passed, 5 skipped`; durations stored at
  `/tmp/tmem_local_r13_fpsan_controls_durations.json`.
  2026-04-21 Round 13 Lane Y completed FPSAN MMAv5 runtime descriptor
  selection fuzzing without backend repairs. No new independent `FZ-*`.
  FPSAN runtime `parent.index(ttgl.load(selector_ptr))` feeding plain MMAv5
  remains report-only `FZ-20260421-0011`; adjacent checked-in plain/scaled
  FPSAN controls, constexpr parent-index, and dynamic-slice controls stayed
  green. Report: `agents/fuzz_fpsan_mma_round13.md`.
  2026-04-21 local scaled-MMAv5 FP4/tile/narrow controls found no new
  `FZ-*`: selector
  `mma_scaled and (tile_permuted or narrow or e2m1 or fp4)` collected
  `202/1615` and passed split-4 as `202 passed`. Report:
  `agents/fuzz_scaled_fp4_tile_narrow_round13.md`.
  2026-04-21 local warp-specialized TMEM partitioning controls found no new
  `FZ-*`: Python selector `warp_specialize and tmem` collected `2/18218` and
  passed `2/2`; lit partition/allocation/interleave/NVWS files passed `4/4`.
  Report: `agents/fuzz_warpspec_partition_round13.md`.
  2026-04-21 local copy/subword selector collected `106/1615` and passed
  split-4 across GPUs 0-3 with stable caches as `106 passed`; durations stored
  at `/tmp/tmem_local_r13_copy_subword_durations.json`.
  2026-04-21 Round 13 Lane Z completed non-overlapping copy/subword edge
  fuzzing without backend repairs. No new independent `FZ-*`; checked-in copy
  selector collected `69/1615` and passed split-4 as `69 passed`,
  representative PTX/LLIR opcodes matched for `128x128b`, `4x256b`, and
  exact-width subword copy, and 4/8/16 CTA high-CGA contrasts mapped to
  existing `FZ-20260421-0010`. Report:
  `agents/fuzz_copy_subword_round13.md`.
  2026-04-21 local `ld.red` M64 row-permuted fuzzing found new candidate
  `FZ-20260421-0012`: checked-in selector
  `ld_red and (tile_permuted or permuted or descriptor_chain or n_sweep) and not reports`
  collected `115/1615` and split-4 ran as `109 passed, 6 failed`. The six
  failures reproduce as over-strict unsupported destination-layout lowering for
  M64 row-permuted reductions while nearby column-permuted controls pass.
  Report: `agents/fuzz_ldred_m64_permuted_round13.md`.
  2026-04-21 local `ld/st` descriptor-selection runtime sweep found no new
  `FZ-*`: selector
  `ldst and (descriptor_roundtrip or descriptor_compositions or descriptor_chain or runtime_selector or rowcol_permuted or tile_selector) and not reports`
  collected `142/1615` and split-4 ran as `64 passed, 78 skipped`.
  Descriptor roundtrips/compositions, row/column-permuted descriptor parents,
  two-CTA descriptor parents, and subword descriptor chains did not reproduce
  the FPSAN MMAv5 runtime-index mismatch or the M64 `ld.red` unsupported-dst
  diagnostic. Report: `agents/fuzz_ldst_descriptor_round13.md`.
  Follow-up selector `ldst and (descriptor_chain or x1 or subword or replay)`
  collected `261/1615` and split-4 ran as `191 passed, 70 skipped`.
  2026-04-21 local MMAv5 descriptor/view runtime sweep found no new `FZ-*`:
  selector
  `mma_twocta_indexed_acc_view or mma_twocta_tma_tf32_b_transposed_descriptor or mma_scaled_shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc`
  collected `37/1615` and split-4 passed as `37 passed`. Non-FPSAN indexed
  accumulator views, lifted `linear_unit_parent` parents, `use_acc`
  true/false rows, TMA transposed-B descriptor rows, and one scaled-MMAv5
  indexed narrow control did not reproduce `FZ-20260421-0011`. Report:
  `agents/fuzz_mma_descriptor_view_round13.md`.
  2026-04-21 local ld.red non-f32 descriptor sanity selected `20/1615` and
  passed split-4 as `20 passed`; durations stored at
  `/tmp/tmem_local_r14_ldred_nonf32_descriptor_durations.json`.
  2026-04-21 local high-CGA/scales sanity selected `17/1615` and passed
  split-4 as `17 passed`; durations stored at
  `/tmp/tmem_local_r14_high_cga_scales_durations.json`.
  2026-04-21 local scaled descriptor sanity selected `32/1615` and passed
  split-4 as `32 passed`; durations stored at
  `/tmp/tmem_local_r14_scaled_descriptor_durations.json`.
  2026-04-21 structural fuzzer smoke gate stayed stable after required
  `make -j8` no-op: `9 passed, 24 xfailed in 8.15s`.
  2026-04-21 Round 14 Lane AA broadened `FZ-20260421-0012` without backend
  repairs. Checked-in `ld_red_m64 and not reports` collected `39/1615` and
  split-4 ran as `33 passed, 6 failed`; temporary `133`-row grid classified
  as `38` pass, `86` `FZ-20260421-0012`, `5` `FZ-20260421-0010`, and `4`
  descriptor-permute harness-limited rows. The bucket now covers effective
  non-identity M64 row bases across `N`, column permutations, modifiers,
  explicit variants, and descriptor-preserving chains. Report:
  `agents/fuzz_ldred_fz0012_round14.md`.
  2026-04-21 Round 14 Lane AB found new candidate `FZ-20260421-0013`
  without backend repairs. Checked-in scaled selector outside the prior
  FP4/tile/narrow slice collected `69/1615` and passed `69/69`; temporary
  shared-scale/copy probe classified `8` pass and `2` harness setup
  limitations; temporary B-scale descriptor-view probe classified `1` pass,
  `3` runtime miscompile candidates, and `3` clean diagnostic boundaries.
  `FZ-0013` is a local 1CTA non-FPSAN scaled-MMAv5 B-scale descriptor-view
  wrong-result candidate with matching PTX/LLIR scaled-MMA opcodes and
  retained descriptor-view chains. Report:
  `agents/fuzz_scaled_descriptor_round14.md`.
  2026-04-21 10:53 UTC local Round 14 generic-pass structural selector
  stayed stable after required `make -j8` no-op. Collect-only selected
  `11/33` rows from
  `python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'`.
  Split-4 execution across GPUs 0-3 with stable caches reported aggregate
  `11 xfailed`. Runtime dynamic `ttg.memdesc_index` illegal lowering remains
  `FZ-20260421-0001`; chain0/control-flow/layout-pressure runtime
  mismatches remain `FZ-20260421-0002`. No new bucket and no backend repair.
  Lane AC remains active on broader generic runtime-index fuzzing and should
  write `agents/fuzz_generic_runtime_index_round14.md`.
  2026-04-21 11:02 UTC Round 14 Lane AC completed and wrote
  `agents/fuzz_generic_runtime_index_round14.md`. No new independent `FZ-*`;
  `FZ-20260421-0001` broadens to copy/MMAv5 consumers and dynamic
  branch-selected memdesc SSA values, while `FZ-20260421-0002` remains the
  chain0 generic wrong-result owner. Temporary probes classified `27`
  generic rows (`10` pass, `6` `FZ-0001`, `11` `FZ-0002`) and `12` copy/MMA
  rows (`2` pass, `8` `FZ-0001`, `2` clean diagnostics). Local adjacent
  two-CTA copy/scales selector collected `53/1615` and passed split-4 as
  `53 passed`.
  2026-04-21 local descriptor and `ld.red` selector sweep wrote
  `agents/fuzz_local_descriptor_ldred_round14.md`. The descriptor algebra
  selector collected `116/1615` and passed split-4 as `38 passed, 78 skipped`.
  The `ld_red and (m64 or descriptor or rowcol_permuted) and not reports`
  selector collected `100/1615` and ran as `94 passed, 6 failed`; the six
  failures remain known `FZ-20260421-0012`.
  2026-04-21 local clean-boundary selector wrote
  `agents/fuzz_local_clean_boundaries_round14.md`. The
  `reports_clean_unsupported or reports_clean_error or reports_tmem_oor`
  selector collected `157/1615` and passed split-4 as `157 passed`; no clean
  diagnostic row drifted into an unexpected crash or runtime path.
  2026-04-21 local scaled-MMAv5 descriptor/accumulator selector wrote
  `agents/fuzz_local_scaled_descriptor_acc_round14.md`. The
  `mma_scaled and (descriptor or scale_descriptor or bscale or acc_subslice_view or indexed_acc) and not reports`
  selector collected `105/1615` and passed split-4 as `105 passed`; no new
  bucket from checked-in scaled descriptor coverage.
- Phase A, rebaseline and classify: done for this branch. The current
  clean-negative/error surface is stable at `145/1615`; unsupported-only
  collect-only is `92/1615`. Every bucket below is classified as positive
  support, typed clean boundary, or frontend/API contract error.
- Phase B, complete shared physical-query model: done for known TMEM policy
  surfaces in this branch. Remaining frontend/lowering code is orchestration
  around backend helpers, not an unowned support predicate with known red
  coverage.
- Phase C, finish `tcgen05.copy` atomized planner: closed for the TMEM
  linear-layout generalization project at 2026-04-18 19:50 UTC. The copy
  planner uses one generic family/footprint path for scales and non-scales,
  with no `isScales` physical-query dimension or support-kind mode. Residual
  rows are classified as public atom mask/full-footprint boundaries,
  refresh-image API work, packed-lane storage/staged-copy work,
  cta-group::2 `warpx2::02_13` ISA/schedule behavior, or explicit frontend
  copy-source contracts. See `tcgen05_cp_gap1_closure_20260418.md`.
- Phase D, finish `ld/st` and `ld.red` packet/replay planning:
  boundary-complete for the current matrix. Supported descriptor-view/replay
  rows are positive; residual rows are true atom-footprint or refresh-row-anchor
  boundaries with structured diagnostics.
- Phase E, finish MMAv5 and scaled-MMAv5 descriptor/storage semantics:
  boundary-complete for the current public ISA and storage contracts. Gap #2
  narrow scaled-MMAv5 `N=8/16` is now positive through B-scale storage
  rematerialization/padding. Gap #3 mixed fp4A TMEM-LHS is closed as a typed
  direct-TMEM storage/API boundary because raw TMEM LHS cannot model the
  shared-memory `fp4_padded` operand-A contract. Other residual rows are
  external `.kind::i8` PTXAS/ISA rejections, instruction-tile order
  boundaries, or explicit frontend/API contracts. Gap #1 signed i8 direct
  MMAv5 is now validated on GB200 through the generated `sm_100a` PTX/cubin
  artifact under
  `experiments/mmav5_i8_remote/`, and has checked-in sm100-gated runtime pytest
  coverage for three signed-i8 shapes; remaining i8 work is IR/frontend
  exposure for unsigned/per-operand signedness and saturation, not the signed
  compiler path.
- Phase F, cleanup/redesign deletion: active cleanup/support checkpoint in
  progress. Recent checkpoints moved MMAv5 address selection, direct `ld/st`
  support/replay predicates, physical-bitcast/view predicates, query-type
  lowering precedence, and now shared Gluon/backend `ld/st` register-layout
  arithmetic into backend helpers, deleted stale copy address dead code, and
  removed obsolete hardcoded trace-file/debug-disable shims from the touched
  Gluon and LLVM TMEM paths. The 2026-04-19 branch cleanup audit also pruned
  superseded raw experiment result logs while preserving compact current
  summaries and reusable evidence artifacts.
  2026-04-20 follow-ups rebaselined the current clean-negative inventory,
  saved exact compact current logs under `experiments/results/`, and clarified
  `legacy` runtime-matrix vocabulary as `TensorMemoryLayout` frontend
  compatibility rather than backend policy. Debug-helper consolidation remains
  deferred until profiling shows the gated debug checks are material.
  2026-04-20 19:03 UTC added
  `python/tutorials/gluon/15-tmem-linear-layout-generalization.py` as a
  presentation/tutorial cleanup summarizing the completed branch features and
  true residual boundaries with runnable examples. This did not open a new
  support-bearing implementation slice.
  2026-04-20 20:10 UTC rewrote that tutorial to be standalone and
  hardware-focused, removed compiler-text assertions, added numeric runtime
  examples, and added a skinny MXFP8 block-scaled projection benchmark using
  narrow TMEM accumulator fragments.
  2026-04-21 06:23 UTC adversarial point-test audit added coverage for
  scaled-MMAv5 narrow view/use-acc paths, two-CTA direct higher-rank
  load/store/reduction boundaries, CTA-count layout contracts, and generic
  TMEM partition relayout behavior. It also fixed `InterleaveTMem` indexed
  view alias ranges and made `OptimizePartitionWarps` preserve TMEM partition
  warp counts until TMEM-aware relayout is implemented.
  2026-04-21 06:44 UTC warp-specialization partition audit fixed
  same-allocation TMEM partition discovery through memdesc view chains and SCF
  value forwarding, with lit coverage for two `ttng.tmem_subslice` views of
  one allocation in a warp-specialized loop.
  2026-04-21 06:47 UTC copy/ld/st/ld.red audit fixed multicast
  `tcgen05.copy` ownership classification for 128-row two-CTA scales
  destinations. The pair-local block selector is now derived from half the
  destination row extent and capped at `128`, so smaller pair-local scales
  tiles use row `64` while 4/8/16 CTA `warpx2` outer ownership remains
  supported.
  2026-04-21 07:00 UTC Round 4 generic backend audit found no new concrete
  fixable gap in `AxisInfo`, `Coalesce`, `RemoveLayoutConversions`,
  `OptimizeThreadLocality`, `LayoutPropagationUtility`, or linear-layout
  conversion interactions with TMEM memdescs. The audited paths already keep
  memdesc SSA values out of AxisInfo/layout-conversion tensor rewrites, or
  restrict themselves to ranked pointer/tensor payloads. Validation covered
  the generic pass lit files plus TMEM layout/verifier/conversion lit.
  2026-04-21 07:03 UTC Round 3 warp-specialization / partitioning /
  barrier audit found no new concrete scheduling or membar implementation bug.
  Added `membar-cluster.mlir` coverage for a valid 4-CTA CGA using two-CTA
  `tc_gen5_mma` with an async completion mbarrier, so the larger-CGA
  pair/mbarrier path is pinned alongside existing 2-CTA MMA and 4/16 CTA
  `ttng.tmem_copy` barrier cases. The memdesc edge-size heuristic remains
  noted as a scheduling-cost risk without a reproduced bad partition.
  2026-04-21 07:03 UTC Round 2 copy/ld/st/ld.red follow-up fixed an
  additional ld.red descriptor-view false negative: `ttng.tmem_subslice`
  views can now use the canonical standalone query type for reduction
  verification/layout inference instead of being rejected because the raw
  subview type still carries the backing column basis. Added conversion lit
  coverage for a 128x64 ld.red from a 128x256 backing TMEM subview; focused
  ld.red descriptor-chain plus recent copy runtime slice passed `31/31`
  across four pytest-split GPU groups.
  2026-04-21 07:10 UTC Round 5 adversarial runtime-matrix probing found no
  production backend bug, but corrected stale coverage: two-CTA higher-rank
  direct `ld.red` through a lifted descriptor-view chain is positive when the
  reduced output uses a matching two-CTA register layout. The runtime matrix
  now validates the positive result and
  `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32` instead of carrying a
  masked clean-negative row. The transposed shared-copy diagnostic assertion
  is aligned to the current clean `ttng.tmem_copy` verifier boundary.
- Phase G, saturation/performance/final validation: done for local branch
  validation. The corrected full runtime-matrix runner passed at 2026-04-17
  21:45 UTC with `1490 passed, 102 skipped` across all `1592` cases.
- Phase H, user-facing performance examples: consolidating into the three
  retained, non-overlapping example files requested by the user. The source of
  truth is `tmem_example_implementation_plan_20260420.md`. Active files:
  `05-tmem-moe-router.py`, `06-tmem-lora-fusion.py`, and
  `08-tmem-layout-as-epilogue.py`. The standalone candidate-head, MLP side
  projection, and ragged expert files were merged into `05`/`06`; the former
  attention score example remains removed because it did not beat the optimized
  plain Triton baseline.
- Historical Phase H checkpoint: complete for the six retained
  example files. The source of truth is
  `tmem_example_implementation_plan_20260420.md`. Implemented examples:
  `05-tmem-moe-router.py`, `06-tmem-lora-fusion.py`,
  `07-tmem-candidate-head.py`, `08-tmem-layout-as-epilogue.py`,
  `09-tmem-mlp-side-projection.py`, and `11-tmem-ragged-expert-views.py`.
  The former `10-tmem-windowed-attention-score.py` was removed after a fair
  Triton mask-plus-row-max baseline beat the TMEM `load_max` path on the
  measured standalone shapes, and row-blocked TMEM variants hit true
  descriptor-view/row-anchor support boundaries. The retained examples use
  plain Triton non-TMEM kernels for post-processing comparisons where needed:
  router top-2, LoRA update, layout reorder, and MLP side gate. Combined
  validation over the retained six files passed `37 passed in 23.52s`.

## Current Clean-Negative Inventory

For the stable discussion order of support/coverage questions, use
`remaining_coverage_gaps.md`. Gap #1 `tcgen05.cp`, Gap #2 narrow
scaled-MMAv5 `N=8/16`, and Gap #3 mixed fp4 TMEM LHS are now closed for this
project. Do not use old pre-reconsolidation copy gap numbers unless explicitly
discussing history.

Collected at 2026-04-17 18:59 UTC after `make -j8`:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q --collect-only python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'reports_clean_unsupported'
```

Result after the 19:25 two-CTA `warpx2::01_23` source-rematerialization
slice: `115/1592` tests collected (1477 deselected) in 3.14s.

Result after the 20:48 integer `ld.red` NaN no-op promotion:
`111/1592` tests collected (1481 deselected) in 3.13s.

Additional combined clean-negative/clean-error rebaseline:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q --collect-only python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'reports_clean_unsupported or reports_clean_error'
```

Result after the 19:25 two-CTA `warpx2::01_23` source-rematerialization
slice: `165/1592` tests collected (1427 deselected) in 3.20s.

Result after the 20:48 integer `ld.red` NaN no-op promotion:
`161/1592` tests collected (1431 deselected) in 3.13s.

Result after the 21:37 backend-policy cleanup checkpoints:
`161/1592` tests collected (1431 deselected) in 2.97s.

Result after the 2026-04-18 22:54 Gap #2/#3 checkpoint:
`141/1594` tests collected (1453 deselected) in 3.07s. The total matrix grew
by two rows from intervening coverage, and the former 20 narrow scaled-MMAv5
clean-negative rows are now positive runtime coverage.

Result after the 2026-04-20 cleanup follow-up rebaseline:
`91/1594` tests collected (1503 deselected) in 3.83s. Exact nodeids are stored
in `experiments/results/clean_unsupported_inventory_current.log`.

Combined clean-negative/clean-error result after the same follow-up:
`141/1594` tests collected (1453 deselected) in 3.83s. Exact nodeids are stored
in `experiments/results/clean_unsupported_or_error_inventory_current.log`.

Round 5 adversarial audit rebaseline at 2026-04-21 07:10 UTC:
`reports_clean_unsupported` collects `92/1615`; combined
`reports_clean_unsupported or reports_clean_error` collects `145/1615`. The
two-CTA higher-rank direct `ld.red` row is no longer in the clean-negative
inventory because the valid two-CTA reduced-output register layout reaches a
positive `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32` runtime path.

Current buckets:
- `ld/st` scales variant atom-footprint boundaries:
  too-narrow n-sharded scale atoms. The n-sharded rows now report a structured
  tensor-memory-scales packet-footprint requirement with required/exposed
  scale-element counts.
- `ld.red` non-f32 reductions: integer `propagate_nan=ALL` is now positive
  through the software reduction path because NaN propagation is a no-op for
  integer element types. Remaining non-f32 rows are software positives or true
  dtype/storage boundaries outside this bucket.
- `tcgen05.copy` scales descriptor-view rows: ordinary noncanonical 64x16
  shared-linear sources and 64x16 shared subslices now rematerialize into the
  canonical warpx4 shared source before copy. The remaining descriptor-view row
  is a destination-view boundary: source column bit 2 selects descriptor row
  `+32` inside a `warpx4` instruction, requiring a source-message/destination-
  column split, narrower atom, valid source format, or destination mask.
  Current probes show the public `warpx4.32x128b` atom writes the full
  16-column destination footprint, so the requested 4-of-8 column split is a
  true mask/source-format schedule boundary.
- `tcgen05.copy` no-scales ordinary contiguous `4x256b`: copy support is
  positive for refresh-shaped layouts only; ordinary view exposure needs a
  first-class refresh remap/readback contract or stays negative.
- Direct `ld/st` of `4x256b` refresh images: row anchors are not materializable
  as public load/store warp bases without a row-anchor rematerialization model;
  this boundary is now reported from structured refresh-image facts shared with
  the copy diagnostic.
- No-scales two-CTA `warpx2::02_13`: current public `cta_group::2`
  direct-seed schedules either duplicate low source columns or read zeros; a
  valid schedule must preserve the high source-column bit. This is now reported
  through a typed source-column preservation requirement derived from the
  source-row split requirement.
- `warpx2` dense/noncanonical shared-source layouts and subword copies:
  single-CTA dense/noncanonical 128x4 shared sources now rematerialize into
  the canonical `warpx2` shared source and are positive for both `01_23` and
  `02_13`; two-CTA dense/noncanonical 256x4 shared sources now rematerialize
  with the canonical shared block basis and are positive for `01_23`.
  Remaining two-CTA `02_13` dense/source-view rows are constrained by the high
  source-column preservation boundary. Subword rows still report a structured
  destination-column footprint requirement with required logical column bits
  and packed-lane facts; they need packed-lane storage and descriptor
  semantic-equivalence proofs.
- Copy row/column permutation and sub-instruction tile permutation rows:
  now have representative probe evidence that descriptor representability is
  not enough. The public copy atoms update full row/column footprints; the
  failing rows need row/column partitioning, smaller footprints, masks, or a
  different source format before they can be positive.
- Plain MMAv5 direct i8 clean-error rows: a guard-lift probe emitted
  `tcgen05.mma.cta_group::1.kind::i8`, but `ptxas-blackwell` rejects
  `.kind::i8` on `sm_103a`. Keep the frontend guard and classify this as an
  external PTXAS/ISA boundary, not a linear-layout backend gap.
- Frontend/API contract clean-error rows: block-layout TMEM descriptors and
  copy source-contract failures still require explicit API redesign before
  they can become backend support work.
- Plain MMAv5 exotic/row-column-permuted accumulators: now reported through a
  typed instruction-tile order requirement. Public atoms require canonical
  row/column basis order within each 64x8-or-larger instruction tile unless a
  tile-splitting or masked writeback schedule is designed.
- Scaled-MMAv5 mixed fp4A TMEM-LHS: closed as a typed direct-TMEM storage/API
  boundary. A speculative fp4-padded direct-TMEM path compiled but produced
  wrong output; the missing semantics are the shared-memory `fp4_padded`
  operand-A row-dependent 128-byte swizzle and padding aliases.
- Scaled-MMAv5 narrow accumulator `N=8/16`: closed as positive support.
  Narrow accumulator families now lower with B-scale storage
  rematerialization/padding, and the former clean-negative matrix is now
  runtime-positive. The 2026-04-21 follow-up hardened the single-fragment
  narrow-N predicate so direct support still requires tensor-memory-scales
  storage with enough addressable rows, added identity `N=16` runtime coverage
  across scaled format pairs, and added a clean-negative `N=16` tile-permuted
  row for the public MMAv5 in-tile basis-order boundary.

## Current Runtime Matrix Validation

Latest checkpoint at 2026-04-17 21:45 UTC after the Phase F backend-policy
cleanup commits:

- `python3 .codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py`
  passed all corrected buckets with complete `1592/1592` coverage.
- Bucket evidence:
  - `cp`: `312 passed, 4 skipped`;
  - `mma`: `601 passed`;
  - `splitn`: `35 passed`;
  - `ld_red`: `247 passed`;
  - `ldst`: `295 passed, 98 skipped`;
  - aggregate: `1490 passed, 102 skipped` across all `1592` cases.
- Logs:
  - `.codex/initiatives/tmem_linear_generalization/experiments/results/tmem_runtime_matrix_sweep_20260417_213804/`

Previous checkpoint at 2026-04-17 21:26 UTC after the Phase F leading-slice
replay policy cleanup and MMAv5 family address-layout cleanup:

- `python3 .codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py`
  passed all corrected buckets with complete `1592/1592` coverage.
- Bucket evidence:
  - `cp`: `312 passed, 4 skipped`;
  - `mma`: `601 passed`;
  - `splitn`: `35 passed`;
  - `ld_red`: `247 passed`;
  - `ldst`: `295 passed, 98 skipped`;
  - aggregate: `1490 passed, 102 skipped` across all `1592` cases.
- Logs:
  - `.codex/initiatives/tmem_linear_generalization/experiments/results/tmem_runtime_matrix_sweep_20260417_211829/`

Baseline checkpoint at 2026-04-17 20:41 UTC after the M64 physical-subview fix:

- Fixed `run_tmem_runtime_matrix_sweep.py` so every shard uses the current
  checkout's `PYTHONPATH` (`repo`, `repo/python`, and
  `repo/python/test/gluon`) instead of accidentally importing the installed
  `triton` wheel.
- Fixed the runner's `splitn` bucket to include
  `test_tmem_runtime_matrix_splitn_16bit_m64_auto_matches_explicit`; a
  collection audit now shows the bucket union covers all `1592/1592` nodeids
  with zero missing and zero extra cases.
- Removed stale `ld.red` split-offset expectations for reverse-column layouts:
  those rows produce correct runtime output and use the default opcode offset
  order, while `tile_permuted` remains the only currently verified N=256
  descriptor-chain row needing `(0, 128, 64, 192)`.
- Bucket evidence for the corrected full runtime matrix:
  - `cp`: `312 passed, 4 skipped`;
  - `mma`: `601 passed`;
  - `splitn`: `35 passed`;
  - `ld_red`: `247 passed`;
  - `ldst`: `295 passed, 98 skipped`;
  - aggregate: `1490 passed, 102 skipped` across all `1592` collected cases.
- Hygiene:
  - `python -m py_compile
    .codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py
    python/test/gluon/test_tmem_runtime_matrix.py`;
  - `git diff --check`.

## Latest Support Promotion

2026-04-17 20:48 UTC:

- Promoted direct and descriptor-chain i32 `ld.red` rows with
  `propagate_nan=ALL` from clean unsupported to positive software reductions.
- Semantics: NaN propagation is treated as `NONE` for non-floating element
  types before selecting the software reduction combiner; floating non-f32
  behavior is unchanged.
- Validation:
  - `make -j8`;
  - exact promoted rows passed `4/4`;
  - `-k 'ld_red_non_f32'` passed `44/44`;
  - full runner `ld_red` bucket passed `247/247`;
  - clean-negative inventory is now `111/1592`;
  - combined clean-negative/error inventory is now `161/1592`;
  - Python byte-compile for the changed Python files;
  - `git diff --check`.

## Latest Cleanup Checkpoint

2026-04-17 21:48 UTC:

- Moved direct `ld/st` query-type-lowering precedence into backend helper
  `shouldPreferTMemLdStQueryTypeLoweringBeforeRawQuery`.
- The helper owns the M64 direct-root exception and derives the warp count from
  the register layout before delegating to
  `shouldPreferTMemLdStQueryTypeLayoutsBeforeRawQuery`.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory changed.
- Validation:
  - `make -j8`;
  - focused replay/direct-support runtime selector split across four GPUs
    passed `6/6`, `6/6`, `6/6`, and `4/4`.

2026-04-17 21:37 UTC:

- Exposed backend `isExplicitTMemLdStViewProducer` and made the Gluon
  register-layout bridge use it for type-only fallback/view-like decisions.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory changed.
- Validation:
  - `make -j8`;
  - focused replay/direct-support runtime selector split across four GPUs
    passed `6/6`, `6/6`, `6/6`, and `4/4`;
  - `git diff --check`.

2026-04-17 21:34 UTC:

- Deleted unused `TensorMemoryToLLVM.cpp` copy address-layout/view-offset
  helpers that still carried family-specific layout sorting outside the shared
  copy planner.
- This is deletion-only cleanup; no support rows were promoted and no
  clean-negative inventory changed.
- Validation:
  - `make -j8`;
  - `git diff --check`.

2026-04-17 21:32 UTC:

- Moved MMAv5 TMEM address-layout and tile-order offset selection out of
  `DotOpMmaV5TmemLoader` and into backend `TensorMemoryUtils` helpers:
  `isTMemPhysicalBitcast`, `getMMAv5TMemAddressLayout`, and
  `getMMAv5TMemViewOffsetForLowering`.
- This keeps physical-bitcast coordinate-frame selection and typed/family/query
  fallback ordering in the backend helper layer instead of in LLVM lowering.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory was changed.
- Validation:
  - `make -j8`;
  - focused MMAv5 runtime selector split across four GPUs passed `26/26` on
    each group;
  - `test_tmem_physical_bitcast_mma_lhs` plus
    `test_tmem_physical_bitcast_preserves_subview_mapping` passed `3/3`.

2026-04-17 21:01 UTC:

- Removed a verifier-local duplicate of the row-zero lifted reinterpret
  query-type rescue predicate.
- `verifyTMEMOperand` now calls the backend-owned
  `disallowTMemLdStQueryTypeRescue` helper, so this policy lives in the shared
  `TensorMemoryUtils` layer.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory was changed.
- Validation:
  - `make -j8`;
  - block descriptor clean-error runtime rows passed `2/2`;
  - blocked-layout clean-error runtime rows passed `2/2`;
  - `git diff --check`.
- Validation note:
  - `lit` is not installed in this shell as either `lit` or `python3 -m lit`.

2026-04-17 21:05 UTC:

- Moved the direct `ld/st` support register-layout search used by
  `OptimizeTMemLayouts` replay/leading-slice rewrites into backend
  `TensorMemoryUtils` as `getTMemLdStDirectSupportTensorType`.
- This removes a transform-local copy of raw-query/support-query/query-type
  search policy and keeps replay lowering as a consumer of backend-owned
  support helpers.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory was changed.
- Validation:
  - `make -j8`;
  - focused replay/direct-support runtime selector split across four GPUs
    passed `6/6`, `6/6`, `6/6`, and `4/4`;
  - `git diff --check`.

2026-04-17 21:08 UTC:

- Exposed the half-slice replay predicate as backend utility
  `getTMemLdStReplayableHalfSliceDim`.
- `OptimizeTMemLayouts` now consumes that helper and computes only the
  rewrite-local selected-half bit.
- Audit result: remaining replay structs/step builders are transform-local
  rewrite construction, not duplicated backend support policy.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory was changed.
- Validation:
  - `make -j8`;
  - focused half-slice replay selector split across four GPUs passed `5/5`,
    `5/5`, `5/5`, and `3/3`;
  - `git diff --check`.

2026-04-17 21:12 UTC:

- Moved the leading-slice direct-vs-replay preservation policy out of
  `OptimizeTMemLayouts` and into backend utility
  `shouldPreserveDirectTMemLdStLeadingSliceView`.
- The helper owns the gapped-column-basis check that decides when direct
  physical support would alias logical halves and the transform should replay
  the descriptor view instead.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory was changed.
- Validation:
  - `make -j8`;
  - focused replay/direct-support runtime selector split across four GPUs
    passed `6/6`, `6/6`, `6/6`, and `4/4`.

2026-04-17 21:17 UTC:

- Moved MMAv5 family address-layout selection from the LLVM lowering loader
  into backend API `getMMAv5TMemFamilyAddressLayout`.
- The helper returns the normalized accumulator, scaled-accumulator, or TMEM
  LHS family layout for full-allocation MMAv5 descriptors, keeping address
  lowering aligned with verifier/planner layout info.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory was changed.
- Validation:
  - `make -j8`;
  - focused MMAv5 tile-permutation selector split across four GPUs passed
    `26/26` on each group.

## Final Execution State

1. No unblocked support-bearing runtime-matrix row remains in the current
   branch. The broad copy, load/store, reduction, MMAv5, and scaled-MMAv5
   matrix is green.
2. Remaining unsupported rows are classified boundaries, not unexplained
   generic linear-layout gaps.
3. Cleanup-only work should now require a newly identified duplicated policy
   surface or a regression. Do not churn bridge/lowering orchestration that
   already consumes backend helpers.
4. If a future ISA revision, PTXAS change, or frontend API adds masks,
   packed-lane storage, refresh remap/readback, or scaled-MMAv5 fragment
   support, reopen the corresponding boundary bucket as a new initiative slice.
5. Keep final validation evidence and this boundary inventory with the branch
   until PR/CI integration is complete.

## Remaining Work Plan

Status legend: `done`, `active`, `pending`, `blocked`, `boundary`.

### Phase A: Rebaseline And Classify

- `done`: collect the current clean-negative surface after a green build and
  record the bucket inventory.
- `done`: classify each bucket as stale guard, missing planner schedule,
  missing storage representation, missing ISA-family coverage, or true
  hardware/API boundary.
- `done`: keep compile/runtime-duration expectations attached to validation
  commands so broad matrix work remains practical on four GPUs.

### Phase B: Shared Physical Query And Requirement Model

- `done`: move identified frontend/lowering-only TMEM decisions into backend
  physical-query helpers or typed requirement structs.
- `done`: make diagnostics consume structured requirement data rather than
  ad hoc strings, especially for clean negatives that remain hardware
  boundaries.
- `done`: remove identified compatibility shims once copy, load/store,
  reduction, MMA, and scaled-MMA consume the shared query/support APIs
  directly.

### Phase C: `tcgen05.copy` Completion

- `boundary`: finish direct no-scales dense/multicast planner boundaries:
  row-order, mixed-basis, sub-instruction column masks, and destination/source
  ownership.
- `done`: decide and encode the correct `4x256b` contract: refresh-image
  positive support, ordinary-view remapping, direct readback limitations, and
  clean typed diagnostics for nonmaterializable refresh images.
- `boundary`: complete scales copy support or prove boundaries for
  descriptor-row split/mask schedules, source-message formats, and
  destination-column partitions.
- `boundary`: complete or prove the no-scales two-CTA `warpx2::02_13`
  schedule, including preservation of the high source-column bit.
- `done/boundary`: handle dense/noncanonical `warpx2` shared-source layouts and
  subword/packed-lane copies through source rematerialization or typed
  non-support proofs.
- `done`: delete copy-specific rescue stacks that duplicate the shared
  planner once their behavior is represented by query/support objects.

### Phase D: Direct `ld/st` And `ld.red`

- `done`: keep packet-footprint limitations represented as structured
  atom-footprint requirements rather than layout-name failures.
- `done/boundary`: continue scales direct `ld/st` descriptor-view support where the
  ISA can realize the requested atom. The two-CTA `16x32bx2` row is now
  positive; remaining clean negatives should be only true packet-footprint
  boundaries or future missing schedules proved by exact layout arithmetic.
- `boundary`: decide whether `4x256b` refresh images can be read back through a
  rematerialized public load/store view; otherwise keep a precise row-anchor
  diagnostic.
- `done`: finish `ld.red` non-f32/NaN semantics either with a correct
  software fallback or a typed semantic-boundary diagnostic.

### Phase E: Plain And Scaled MMAv5

- `done`: keep plain MMAv5 accumulator layout failures tied to a structured
  instruction-tile row/column ordering requirement while preserving existing
  whole-tile permutation positives.
- `done`: prove and structure the scaled-MMAv5 mixed fp4A TMEM-LHS storage
  representation boundary.
- `done`: support scaled-MMAv5 narrow-N through accumulator family planning and
  B-scale fragment storage rematerialization.
- `boundary`: expand opcode/runtime positives only when a new backend schedule
  is real, not when a frontend spelling happens to compile.

### Phase F: Cleanup And Redesign Deletion

- `done`: delete stale frontend guards and lowering-local type-only fallback
  policy as backend helpers subsume them.
- `done`: consolidate repeated layout arithmetic into shared helpers with
  exact `LinearLayout` compose/invert/pseudoinvert proofs.
- `done`: keep only durable abstractions; quarantine or remove temporary
  row/column rewrite smells after each family is represented in the planner.

### Phase G: Saturation, Performance, And Final Validation

- `done`: run staged validation: `make`, targeted lit/compiler checks,
  focused four-GPU runtime selectors, then duration-aware broad sweeps.
- `done`: profile any representative compile above the 3-4 second target or
  execution above the 1-2 second target before accepting the regression.
- `not required for this local branch checkpoint`: refresh GB200/NVIDIA
  manifests after broad validation and mark stale pre-fix counts explicitly.
  Do this only when consuming fresh external CI results.
- `done`: final state requires no unexplained clean negatives, no stale
  family-specific policy, practical split-4 runtime-matrix iteration, and an
  updated handoff summarizing every residual true ISA boundary.

## Next Concrete Slice

- 2026-04-18 22:54 UTC: closed Gap #2 narrow scaled-MMAv5 `N=8/16` as
  positive support and Gap #3 mixed fp4A TMEM-LHS as a direct-TMEM storage/API
  boundary. Validation: `make -j8`; narrow-N runtime matrix `20 passed`;
  mixed-fp4A TMEM-LHS clean-negative matrix `24 passed`; non-twoCTA
  scaled-accumulator tile-permuted selector `52 passed`.

No unblocked local implementation slice remains in the current TMEM
generalization plan. The next action is PR/CI integration and external
signal handling:

- if CI finds a branch-only correctness failure, classify it against this
  boundary inventory and fix the regression;
- if CI exposes a new support-bearing layout that is not covered here, reopen
  the relevant phase with a new targeted repro;
- if a future ISA/API exposes masks, packed-lane storage, refresh remapping,
  direct i8 MMAv5, or scaled-MMAv5 narrow-fragment support, create a new
  support slice from the corresponding boundary bucket.

- 2026-04-20 19:03 UTC: user-requested presentation cleanup added a Gluon
  tutorial for the branch feature suite. Validation was limited to py-compile
  because required `make -j8` failed immediately on an empty local CMake build
  directory with no `build.ninja`; direct import also failed on the unbuilt
  local `triton._C.libtriton.getenv` symbol. Tutorial checkpoint commit
  `4e96b6d40` was pushed to `origin/codex/tmem`.

- 2026-04-20 20:10 UTC: tutorial rewrite is runtime-validated. Required
  `make -j8` passed with explicit host libstdc++ include paths in
  `CPLUS_INCLUDE_PATH`; focused tutorial pytest passed `6/6`; script benchmark
  reported `N=32` narrow `0.011 ms` vs padded `0.013 ms` and `N=64` narrow
  `0.013 ms` vs padded `0.013 ms`. Tutorial rewrite checkpoint `021688bc2`
  was pushed to `origin/codex/tmem`.

- 2026-04-20 20:21 UTC: tutorial polish added inline benchmark output and a
  pre-branch capability note. Merge-base inspection found that the old
  scaled-MMA tutorial used `TensorMemoryLayout` accumulator allocations and did
  not expose explicit tile-permuted linear narrow fragments. Focused tutorial
  pytest passed `6/6`; script benchmark reported `N=32` `1.18x` and `N=64`
  `1.05x`.

## Progress

- 2026-04-17 21:50 UTC: updated the tracker to final local boundary status.
  The active plan has no unblocked local support-bearing slice remaining after
  the 21:45 full runtime-matrix sweep and the stable `161/1592`
  clean-negative/error collect-only inventory. Remaining buckets require
  external ISA/PTXAS support, frontend/API contract changes, destination
  masks, packed-lane storage, refresh remap/readback, or scaled-MMAv5 fragment
  representation work beyond the current public model.

- 2026-04-17 21:48 UTC: moved direct `ld/st` query-type-lowering precedence
  from `TensorMemoryToLLVM` into
  `shouldPreferTMemLdStQueryTypeLoweringBeforeRawQuery`. Validation:
  `make -j8`; focused split-4 replay/direct-support selector passed `6/6`,
  `6/6`, `6/6`, and `4/4`.

- 2026-04-17 21:45 UTC: reran the corrected full runtime-matrix runner after
  the latest backend-policy cleanup commits. All buckets passed with complete
  `1592/1592` coverage: `cp` `312 passed, 4 skipped`; `mma` `601 passed`;
  `splitn` `35 passed`; `ld_red` `247 passed`; `ldst`
  `295 passed, 98 skipped`; aggregate `1490 passed, 102 skipped`. Per-shard
  logs are in
  `.codex/initiatives/tmem_linear_generalization/experiments/results/tmem_runtime_matrix_sweep_20260417_213804/`.
  Combined clean-negative/error collect-only remains `161/1592`.

- 2026-04-17 21:37 UTC: exposed backend
  `isExplicitTMemLdStViewProducer` and replaced the Gluon bridge's local
  explicit-view op list with that helper. Validation: `make -j8`; focused
  split-4 replay/direct-support selector passed `6/6`, `6/6`, `6/6`, and
  `4/4`; `git diff --check`.

- 2026-04-17 21:34 UTC: deleted unused copy address-layout/view-offset helper
  code from `TensorMemoryToLLVM.cpp`. The removed code was not referenced by
  lowering and preserved stale family-specific layout sorting outside the
  backend copy planner. Validation: `make -j8`; `git diff --check`.

- 2026-04-17 21:32 UTC: moved MMAv5 TMEM address-layout and tile-order offset
  selection from LLVM lowering into backend `TensorMemoryUtils` helpers. This
  includes sharing `isTMemPhysicalBitcast` with the direct `ld/st` support
  planner. Validation: `make -j8`; focused MMAv5 split-4 runtime selector
  passed `26/26` on all groups; physical-bitcast core smoke passed `3/3`.

- 2026-04-17 21:26 UTC: reran the corrected full runtime-matrix runner after
  the leading-slice replay policy cleanup and MMAv5 family address-layout
  helper cleanup. All buckets passed with complete `1592/1592` coverage:
  `cp` `312 passed, 4 skipped`; `mma` `601 passed`; `splitn` `35 passed`;
  `ld_red` `247 passed`; `ldst` `295 passed, 98 skipped`; aggregate `1490
  passed, 102 skipped`. Per-shard logs are in
  `.codex/initiatives/tmem_linear_generalization/experiments/results/tmem_runtime_matrix_sweep_20260417_211829/`.

- 2026-04-17 21:12 UTC: moved the leading-slice direct-vs-replay preservation
  policy into backend utility `shouldPreserveDirectTMemLdStLeadingSliceView`.
  `OptimizeTMemLayouts` now consumes the helper instead of owning the
  gapped-column query check locally. Validation: `make -j8`; split-4 focused
  replay/direct-support runtime selector passed `6/6`, `6/6`, `6/6`, and
  `4/4`.

- 2026-04-17 21:17 UTC: moved MMAv5 family address-layout selection into
  backend API `getMMAv5TMemFamilyAddressLayout`. The LLVM MMAv5 loader now
  consumes the helper instead of carrying its own full-allocation
  accumulator/scaled/LHS family-layout lambda. Validation: `make -j8`;
  split-4 focused MMAv5 tile-permutation selector passed `26/26` on each
  group.

- 2026-04-17 20:57 UTC: ran the corrected full runtime-matrix runner. All
  buckets passed with complete `1592/1592` coverage: `cp` `312 passed, 4
  skipped`; `mma` `601 passed`; `splitn` `35 passed`; `ld_red` `247 passed`;
  `ldst` `295 passed, 98 skipped`; aggregate `1490 passed, 102 skipped`.
  Per-shard logs are in
  `.codex/initiatives/tmem_linear_generalization/experiments/results/tmem_runtime_matrix_sweep_20260417_205257/`.

- 2026-04-17 20:52 UTC: completed the copy-planner boundary audit for the
  remaining representative clean negatives. Validation command:
  `make -j8 && CUDA_VISIBLE_DEVICES=0
  TRITON_CACHE_DIR=/tmp/triton-cache-copy-boundary-audit
  PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short -k
  'cp_scales_tmem_descriptor_view_reports_clean_unsupported or
  cp_no_scales_4x256b_reports_clean_unsupported or
  cp_no_scales_warpx2_02_13_twocta or
  cp_no_scales_warpx2_row_permuted_destination_reports_clean_unsupported or
  cp_no_scales_twocta_noncanonical_block_reports_clean_unsupported or
  cp_no_scales_linear_exotic_reports_clean_unsupported or
  cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported
  or cp_no_scales_linear_rowcol_permuted_reports_clean_unsupported'
  python/test/gluon/test_tmem_runtime_matrix.py`, which passed `44/44`. No
  support rows were promoted because the diagnostics still point to true
  packed-lane, source-column, row/column-mask, refresh-remap, scales-mask, or
  CTA-ownership requirements.

- 2026-04-17 20:24 UTC: fixed the legacy M64 split-N physical-subview
  mismatch exposed by `test_core.py::test_block_m_64_mma[legacy]`. The failure
  was not a scales- or M64-only opcode issue: pure TMEM column subviews were
  computing relative bases by subtracting query origins as row/column
  coordinates. For source queries whose exact `LinearLayout` is surjective but
  folds a logical column basis into a physical row basis, that arithmetic
  lowers high-N descriptor views to a plain column offset. The base lowering
  now first pseudoinverts the source query's exact surjective layout for the
  requested subview offset, then falls back to the older origin-delta path for
  projected/non-surjective queries. This aligns MMAv5 `memdesc_subslice`
  addresses with direct root `ld/st` packet offsets for legacy M64 split-N
  high bands. Validation: `make -j8`; `test_core.py::test_block_m_64_mma`
  passed `2`; focused legacy M64 runtime rows passed `4`; ld.red
  descriptor-chain tile-permuted N=256 passed `1`; Python byte-compile for
  `test_tmem_runtime_matrix.py`; `git diff --check`.

- 2026-04-17 19:27 UTC: probed the remaining `warpx2` subword/packed-lane
  bucket after source rematerialization. Representative f16 single-CTA
  `01_23`, single-CTA `02_13`, and two-CTA `01_23` runs all fail through the
  structured destination-column footprint requirement: the N=4 destination
  exposes two column basis bits, while a public 128-bit f16 `warpx2` copy
  instruction needs three logical column bits and writes four physical dword
  columns with two packed lanes per word. This is not a stale shared-source
  layout failure. Support needs a first-class packed-lane source/destination
  storage model through descriptor synthesis, source-footprint planning, and
  instruction scheduling; otherwise the rows remain true storage/ISA
  boundaries.

- 2026-04-17 19:25 UTC: promoted two-CTA no-scales `warpx2::01_23`
  dense-source rematerialization. The same `tcgen05_copy` rematerialization
  helper now handles 256x4 TensorMemoryLinear `warpx2` destinations by
  rematerializing noncanonical shared sources into the canonical `warpx2`
  source layout with shared block basis `[[128, 0]]` and issuing a cluster
  async-shared fence. The former `01_23` two-CTA dense-source clean negatives
  now check runtime output and opcode coverage. Dense `02_13` two-CTA rows now
  rematerialize the source far enough to report the existing high
  source-column preservation boundary instead of the older source-layout
  mismatch. Validation: `make -j8`; focused two-CTA dense selector passed
  `4`; full `cp_no_scales_warpx2` passed `79`; split-4
  `cp_no_scales_warpx2` groups passed `20/20/20/19`; Python byte-compile;
  collect-only rebaseline is `115/1592` clean negatives and `165/1592` clean
  negatives/errors.

- 2026-04-17 19:21 UTC: promoted single-CTA no-scales `warpx2` dense-source
  rematerialization. `tcgen05_copy` now recognizes 128x4 TensorMemoryLinear
  destinations that map to public `warpx2::01_23` or `warpx2::02_13` copy
  families, reloads noncanonical `SharedLinearLayout` sources through the
  logical source coordinates, stores them into the canonical 128x4 `warpx2`
  shared-linear source layout, issues a fresh async-shared fence, and emits
  the existing backend copy op. The former single-CTA dense-source clean
  negatives now check runtime output and opcode coverage; two-CTA dense-source
  rows and the `02_13` source-column boundary remain negative. Validation:
  `make -j8`; exact promoted selector passed `4`; neighboring negative plus
  `cp_scales` selector passed `41`; full `cp_no_scales_warpx2` passed `79`;
  split-4 `cp_no_scales_warpx2` groups passed `20/20/20/19`; Python
  byte-compile; collect-only rebaseline is `117/1592` clean negatives and
  `167/1592` clean negatives/errors.

- 2026-04-17 19:14 UTC: promoted TensorMemoryScales 64x16 source
  rematerialization for `tcgen05.copy`. The Gluon `tcgen05_copy` builtin now
  rematerializes noncanonical 64x16 `SharedLinearLayout` scales sources into
  the canonical warpx4 shared-linear source before emitting the backend copy
  op. This makes the historical warpx2-shaped scales layout probes and
  shared-subslice source probes positive while preserving the scaled-MMA scale
  descriptor path and keeping the 128x32 descriptor-view destination boundary
  negative. Validation: `make -j8`; focused selector
  `cp_scales_layout_probe or cp_scales_noncanonical_layout_rematerializes or
  cp_scales_shared_subslice_layout_rematerializes or
  cp_scales_tmem_descriptor_view_reports_clean_unsupported` passed `10`;
  broader `-k 'cp_scales'` passed `34`; split-4 `cp_scales` groups passed
  `9/9/9/7`; Python byte-compile; `git diff --check`. Rebaseline:
  `reports_clean_unsupported` is `121/1592`; combined clean-negative/error is
  `171/1592`.

- 2026-04-17 19:07 UTC: classified two remaining buckets and refreshed the
  execution contract. Representative direct Python probes with
  `TRITON_DEBUG_TMEM_QUERY=1` show sub-instruction copy tile permutations and
  row/column permutations need partial row/column updates inside public
  full-footprint copy atoms; this is a mask/smaller-atom/source-format
  boundary, not a stale descriptor recognizer. A temporary direct-MMAv5 i8
  guard lift reached PTX emission, but `ptxas-blackwell` rejected
  `.kind::i8` for `sm_103a`; the guard was restored and the i8 bucket is now
  classified as an external PTXAS/ISA boundary. `AGENTS.md` now explicitly
  says the initiative docs, not chat context, are the active checklist for the
  no-stop execution rule. Validation/probes: `make -j8` before runtime probes;
  combined clean-negative/clean-error collect-only is `174/1592`.

- 2026-04-17 18:59 UTC: promoted the two-CTA scales descriptor-view
  `16x32bx2` direct `ld/st` row from clean-negative to positive support. The
  backend now synthesizes an exact lifted view layout for
  `tcgen05.ld/st.16x32bx2` by keeping the half-tile split on lane=16 and not
  duplicating the lifted row basis as register repetition. Generic lowering
  now recognizes this descriptor-view candidate even though the memdesc arrives
  as `TensorMemoryLinear`, preserving the requested split-N atom instead of
  silently selecting `32x32b`. Validation: `make -j8`; exact promoted row
  passed; focused CGA descriptor-view selector passed as `9 passed,
  1583 deselected`; Python byte-compile; `git diff --check`. Clean-negative
  rebaseline: `reports_clean_unsupported` is now `123/1592`, and
  `reports_clean_unsupported or reports_clean_error` is now `174/1592`.

- 2026-04-17 18:41 UTC: structured the explicit n-sharded scales `ld/st`
  packet-footprint boundary. `getUnsupportedDirectTMemLdStAtomFootprintReason`
  now also classifies tensor-memory-scales views for 16x64b/16x128b/16x256b
  requests whose exposed element count is smaller than the public atom
  footprint, and the clean-negative tests assert required versus exposed scale
  elements. Behavior is unchanged: these rows remain unsupported unless the
  user selects a narrower atom/auto layout or reshapes/copies to cover the
  requested packet footprint. Validation: `make -j8`; built `triton-opt
  test/TritonNvidiaGPU/invalid.mlir --split-input-file
  --verify-diagnostics`; focused scales unsupported selector passed
  `5 passed`; split-4 selected groups passed `2/2/1` with group 4 empty;
  Python byte-compile for `test_tmem_runtime_matrix.py`; `git diff --check`.

- 2026-04-17 18:36 UTC: structured the under-wide subword `warpx2`
  destination footprint boundary. The multicast destination-layout check now
  formats a `TMemCopyColumnFootprintRequirement` with element bitwidth,
  logical instruction columns, physical 32-bit dword columns, packed lanes per
  word, exposed column basis bits, and required column basis bits. Behavior is
  unchanged: `N=4` subword `warpx2` rows remain clean unsupported because the
  public 128-bit copy atom needs packed-lane storage semantics across
  descriptor synthesis, source footprint planning, and instruction scheduling.
  Validation: `make -j8`; built `triton-opt
  test/TritonNvidiaGPU/invalid.mlir --split-input-file
  --verify-diagnostics`; split-4 focused subword selector passed `4/4/4/2`;
  Python byte-compile for `test_tmem_runtime_matrix.py`; `git diff --check`.

- 2026-04-17 18:32 UTC: structured the `warpx2` dense/noncanonical
  shared-source boundary. `TMemCopyWarpx2SharedSourceRequirement` now carries
  source shape and optional offset-basis mismatch data, and the formatter
  reports the first shared offset-basis mismatch plus the source
  rematerialization boundary explicitly. Behavior is unchanged: the dense
  shared-source rows still reject cleanly because the public `warpx2` source
  message schedule assigns fixed meanings to shared offset bases, so descriptor
  representability alone is not enough. Validation: `make -j8`; built
  `triton-opt test/TritonNvidiaGPU/invalid.mlir --split-input-file
  --verify-diagnostics`; split-4 focused dense shared-source selector passed
  `2/2/2/2`; Python byte-compile for `test_tmem_runtime_matrix.py`;
  `git diff --check`.

- 2026-04-17 18:27 UTC: structured the scaled-MMAv5 narrow-N accumulator
  boundary. `MMAv5ScaledNarrowNScaleFragmentRequirement` now records logical
  shape, CTA shape, the plain MMAv5-compatible instruction shape, fragment
  count along N, the public scaled `N>=32` floor, and the B-scale
  padding/rematerialization factor implied by 64-column tensor-memory scale
  storage. Behavior is unchanged: `N=8/16` tile-permuted accumulator layouts
  remain clean unsupported until the backend can synthesize both a correct
  accumulator permutation and B-scale fragment rematerialization. Validation:
  `make -j8`; built `triton-opt test/TritonNvidiaGPU/invalid.mlir
  --split-input-file --verify-diagnostics`; split-4 focused narrow-N selector
  passed `5/5/5/5`; Python byte-compile for `test_tmem_runtime_matrix.py`;
  `git diff --check`.

- 2026-04-17 18:23 UTC: promoted the mixed fp4A TMEM-LHS guard into a
  structured `MMAv5ScaledMixedFp4ATMemRequirement`. The requirement records
  LHS storage shape, CTA shape, raw storage K columns, logical K, A/B bitwidths,
  and the `fp4_padded` shared-memory storage group facts. A temporary
  guard-lift probe for the `n=128, linear` row emitted
  `tcgen05.mma.kind::mxf8f6f4` but failed numerical comparison for essentially
  the whole tile (`16376/16384` mismatches), proving this is a real storage
  representation gap rather than a stale verifier guard. Validation: `make
  -j8`; built `triton-opt test/TritonNvidiaGPU/invalid.mlir
  --split-input-file --verify-diagnostics`; split-4 focused mixed-fp4A
  selector passed `6/6/6/6`; Python byte-compile for
  `test_tmem_runtime_matrix.py`; `git diff --check`.

- 2026-04-17 18:16 UTC: promoted the plain MMAv5 exotic and row/column
  permuted accumulator clean negatives into a structured
  `MMAv5TMemInstructionTileRequirement`. The requirement records logical
  shape, CTA shape, element bitwidth, minimum public instruction tile, and the
  first noncanonical in-tile basis when one is available. Behavior is
  unchanged: whole-tile accumulator permutations remain positive, while
  permutations inside the public `64x8`-or-larger tile remain a true
  tile-order / masked-writeback boundary. Validation: `make -j8`; built
  `triton-opt test/TritonNvidiaGPU/invalid.mlir --split-input-file
  --verify-diagnostics`; split-4 focused MMAv5 negative selector passed as
  `5/5/5/2`; Python byte-compile for `test_tmem_runtime_matrix.py`;
  `git diff --check`.

- 2026-04-17 18:10 UTC: promoted the no-scales two-CTA
  `warpx2::02_13` probe evidence into a typed
  `TMemCopyWarpx2TwoCTASourceColumnRequirement`. The diagnostic still reports
  the same ISA facts, but it now formats from the source-row split requirement:
  logical row bit 5 would need to select a one-dword source offset for
  32-of-64 destination rows, while `cta_group::2` direct-seed probes either
  duplicate the low source-column pair or read zeros when trying to complete
  the single-CTA schedule. Support is unchanged and remains a true source
  column / destination-row mask boundary. Validation: `make -j8`; split-4
  focused `warpx2::02_13`/neighbor selector passed as `8/8/8/6`; built
  `triton-opt test/TritonNvidiaGPU/invalid.mlir --split-input-file
  --verify-diagnostics`; Python byte-compile for `test_tmem_runtime_matrix.py`;
  `git diff --check`.

- 2026-04-17 18:04 UTC: classified the scales `tcgen05.copy`
  descriptor-view/shared-subslice split as a true full-footprint schedule
  boundary and structured the exact-view note. `TMemCopyExactViewScheduleNote`
  now derives from `TMemCopyExactViewScheduleRequirement`, recording the first
  differing physical-query field and active shape before reporting that a
  destination-row/source-message schedule is required. Debug probes showed the
  concrete gap: source column bit 2 selects descriptor row `+32` for
  4-column runs every 8 columns, but `warpx4.32x128b` writes the full
  16-column destination footprint. Validation: `make -j8`; split-4 focused
  scales-copy selector passed as `3/3/3/2`; built `triton-opt
  test/TritonNvidiaGPU/invalid.mlir --split-input-file --verify-diagnostics`;
  Python byte-compile for `test_tmem_runtime_matrix.py`; `git diff --check`.

- 2026-04-17 17:59 UTC: represented the direct `ld/st` `4x256b`
  refresh-image boundary with structured refresh-image data instead of a
  static string. `getTMemCopy4x256RefreshLdStUnsupportedMessage(...)` now
  formats the logical tile, low-column row anchors, high-column dword offset,
  and source-column split from `TMemCopy4x256RefreshImageRequirement`; the raw
  physical-bitcast packet-footprint diagnostic also carries the same facts.
  Behavior is unchanged: copy refresh layouts remain opcode-positive, ordinary
  contiguous `4x256b` views and direct refresh-image readback remain clean
  unsupported. Validation: `make -j8`; focused runtime selector
  `cp_no_scales_4x256b or ldst_4x256b_refresh` passed as `5 passed,
  1587 deselected`; split-4 selector groups with selected rows passed as
  `2/2/1` while group 4 was empty; `triton-opt
  test/TritonNvidiaGPU/invalid.mlir --split-input-file --verify-diagnostics`;
  Python byte-compile for `test_tmem_runtime_matrix.py`; `git diff --check`.

- 2026-04-17 17:55 UTC: strengthened the execution contract in this tracker
  and in `AGENTS.md`. The remaining work is now tracked as a phase-by-phase
  board from rebaseline through final validation, with explicit instructions
  to keep this file current and continue executing until the tracked plan is
  complete unless interrupted or blocked.

- 2026-04-17 17:53 UTC: made the scaled-MMAv5 narrow-N accumulator
  requirement carry `minimumAddressableBScaleFragmentN` explicitly instead of
  hardcoding the 64-column matrix-B scale-fragment alignment in the formatter.
  Support is unchanged: N=8 and N=16 accumulator tiles remain clean
  unsupported until a real scale-fragment rematerialization/storage model can
  preserve the public scaled-MMAv5 fragment semantics. Validation: `make -j8`;
  split-4 `mma_scaled_acc_tile_permuted_narrow_reports_clean_unsupported`
  selector passed `5/5/5/5`; `git diff --check`.

- 2026-04-17 17:50 UTC: added a typed
  `TMemCopyMixedBasisRequirement` for direct copy physical-query failures where
  a TMEM row or column basis mixes physical row and column contributions. The
  `mixed` no-scales copy negative now reports the exact offending basis bit and
  physical TMEM delta, rather than stopping at a generic mixed-basis string.
  Probe evidence from `TRITON_DEBUG_TMEM_QUERY=1` also classified the
  sub-instruction column-permutation and scales descriptor-row split negatives
  as full-footprint/mask gaps: the public atoms would overwrite complementary
  rows/columns without a narrower atom, source format, or destination mask.
  Validation: `make -j8`; split-4 selector for copy exotic, row/column
  permuted, and `warpx2` row-permuted clean negatives passed `5/5/5/4`;
  `git diff --check`.

- 2026-04-17 17:45 UTC: promoted copy destination row-order rejection into a
  shared typed `TMemCopyDestinationRowOrderRequirement` in
  `TensorMemoryUtils.cpp`. Dense and multicast row-permuted copy layouts now
  carry the offending row-basis sequence, instruction footprint, and derived
  destination-row mask/schedule gap instead of formatting the boundary from two
  separate ad hoc string builders. Support is intentionally unchanged: these
  rows still require a future row-selected source projection, row-partitioned
  atom, or destination-row mask before becoming positive. Validation:
  `make -j8`; split-4 focused selector for
  `cp_no_scales_warpx2_row_permuted_destination_reports_clean_unsupported` and
  `cp_no_scales_linear_rowcol_permuted_reports_clean_unsupported` passed
  `4/4/4/4`; `git diff --check`.

- 2026-04-17 17:40 UTC: converted the `warpx2` shared-source runtime preflight
  in `TensorMemoryUtils.cpp` from inline string checks into a typed internal
  `TMemCopyWarpx2SharedSourceRequirement`. Behavior is intentionally unchanged:
  noncanonical `warpx2` dense/shared-source rows still fail cleanly with the
  same user-facing diagnostics. This advances Phase B/F cleanup by moving one
  more copy-family boundary onto requirement data instead of ad hoc verifier
  text. Validation: `make -j8`; split-4 focused selector for
  `cp_no_scales_warpx2_dense_shared_reports_clean_unsupported`,
  `cp_no_scales_warpx2_twocta_dense_shared_reports_clean_unsupported`, and
  `cp_no_scales_warpx2_02_13_twocta_candidate_reports_clean_unsupported`
  passed as `3/3/3/1`; `git diff --check`.

- 2026-04-20 21:07 UTC: opened Phase H as a tracked follow-on examples
  workstream and added `tmem_example_implementation_plan_20260420.md`. No
  example implementation has started. Next concrete implementation slice is
  the projection-only MoE router example with a padded `N=128` baseline,
  correctness tests, shape coverage, and benchmark transcript.

- 2026-04-20 21:34 UTC: implemented Phase H Example 1 in
  `python/examples/gluon/05-tmem-moe-router.py`. The example covers narrow
  MXFP8 router projection for `E=32/64`, `K=128/256`, a padded `E=128`
  pre-generalization baseline, PyTorch top-k wrapper validation, TTGIR
  checks for `tensor_memory_linear` and `ttng.tc_gen5_mma_scaled`, and inline
  benchmark transcripts. Focused validation passed `8/8`. Next concrete slice:
  Example 2 LoRA / adapter projection fusion.

- 2026-04-20 21:51 UTC: implemented Phase H Example 2 in
  `python/examples/gluon/06-tmem-lora-fusion.py`. The example covers compact
  MXFP8 LoRA down projection for `R=32/64`, `K=128/256`, a padded `R=128`
  pre-generalization baseline, a full LoRA update wrapper, TTGIR checks, and
  inline benchmark transcript. Focused validation passed `8/8`. Next concrete
  slice: Example 3 candidate-head projection.

- 2026-04-20 22:04 UTC: implemented Phase H Example 3 in
  `python/examples/gluon/07-tmem-candidate-head.py`. The example covers compact
  selected-candidate MXFP8 logits for `C=32/64`, `K=128/256`, padded `C=128`
  baseline, candidate-order validation, TTGIR checks, and inline benchmark
  transcript. Focused validation passed `6/6`. Next concrete slice: Example 4
  layout-as-epilogue store ordering.

- 2026-04-20 22:18 UTC: implemented Phase H layout-as-epilogue example in
  `python/examples/gluon/08-tmem-layout-as-epilogue.py`. The example covers
  direct consumer-order stores for `N=64/128/256`, a canonical-store plus
  PyTorch reorder baseline, and inline benchmark transcript. Focused
  validation passed `5/5`. Next concrete slice: Example 5 fused MLP
  side-projection.

- 2026-04-20 22:31 UTC: implemented Phase H MLP side-projection example in
  `python/examples/gluon/09-tmem-mlp-side-projection.py`. The example covers
  compact side projections `S=32/64`, `K=128/256`, a padded `S=128` baseline,
  a broad-projection/gate wrapper, TTGIR checks, and inline benchmark
  transcript. Focused validation passed `8/8`. Next concrete slice: windowed
  attention score reductions.

- 2026-04-20 22:45 UTC: implemented Phase H windowed attention score/reduction
  example in `python/examples/gluon/10-tmem-windowed-attention-score.py`. The
  example covers TMEM row-max reductions for noncausal and causal score tiles,
  a PyTorch mask/max baseline, Blackwell Ultra gating, and inline benchmark
  transcript. Focused validation passed `5/5`. Next concrete slice: ragged
  expert views.

- 2026-04-20 23:01 UTC: implemented Phase H ragged expert panel example in
  `python/examples/gluon/11-tmem-ragged-expert-views.py`. The example covers a
  Python-scheduled ragged expert list with one empty expert, compact per-expert
  panels, a precomputed padded `N=128` per-expert baseline, correctness checks,
  and inline benchmark transcript. Focused validation passed `2/2`. All seven
  planned Phase H example files now exist; next concrete slice is broad
  examples validation and final plan refresh.

- 2026-04-20 23:14 UTC: broad Phase H validation passed. Py-compile passed for
  `python/examples/gluon/05-tmem-moe-router.py` through
  `python/examples/gluon/11-tmem-ragged-expert-views.py`; combined focused
  pytest over the seven files passed `42 passed in 26.83s`. Phase H is now
  complete for the requested example suite, with caveats recorded in the
  individual example files and plan.

- 2026-04-20 23:25 UTC: Phase H timed comparisons were rewritten to remove
  PyTorch post-processing from benchmark paths. Router top-k, LoRA update,
  layout reorder, MLP side gate, and attention mask-plus-row-max now use plain
  Triton kernels without the new TMEM features. PyTorch remains only for
  correctness references and data setup. Py-compile passed for all seven files,
  combined pytest passed `42 passed in 6.53s`, and benchmark transcripts were
  refreshed from local script runs.

- 2026-04-21 00:05 UTC: removed
  `python/examples/gluon/10-tmem-windowed-attention-score.py`. Rebenchmarking
  against the optimized plain Triton mask-plus-row-max baseline showed the
  standalone TMEM `load_max` example remained slower. Attempts to improve
  parallelism by row-blocking the TMEM tile at `BLOCK_M=32` and `BLOCK_M=64`
  failed at compile time on real TMEM descriptor-view support boundaries, so
  the example did not meet the performance bar for a user-facing example.
  Required `make -j8` was no-op, py-compile passed for the retained six files,
  and combined pytest passed `37 passed in 23.52s`.

- 2026-04-21 00:30 UTC: started three-example consolidation requested by the
  user. Merged candidate-head and ragged expert implementations into
  `05-tmem-moe-router.py`, merged MLP side projection into
  `06-tmem-lora-fusion.py`, and deleted the standalone `07`, `09`, and `11`
  files. Required `make -j8` was no-op, py-compile passed for the three retained
  files, focused pytest passed `37 passed in 23.47s`, and the three script
  benchmark transcripts were refreshed from local runs.

- 2026-04-20 23:34 UTC: active phase is upstream-main merge recovery. Merged
  `upstream/main` at `2c7ce4925d37802dd84dfde1f6458cae19485617` into
  `codex/tmem` from pre-merge `c9166449eeaad20516a3f831c0899c56d91f9fc8`.
  Completed build and conflicted lit recovery: `make -j8` passed with the host
  C++ include path, and the six conflicted lit files passed. Open validation
  blockers are runtime-only: two-CTA scale-copy tests now hit the clean
  unsupported broadcast `warpx4` copy boundary, and scaled-MMA FPSAN payload
  checks fail for both legacy and linear accumulators. Next concrete
  implementation slice is to make the scale-copy tests respect the hardware
  boundary or extend the planner/API, then root-cause the FPSAN payload drift.

- 2026-04-20 23:55 UTC: post-merge runtime recovery completed for the focused
  scaled-MMA/copy surface. The FPSAN payload drift was an oracle merge bug and
  is fixed by carrying separate A/B element types through
  `_mm_scaled_payload_u32`. Two-CTA scaled-copy and 2x2 direct multicast
  accumulator readback cases are now explicit xfails for documented
  hardware/API boundaries. Validation passed for the focused runtime slices,
  retained examples/tutorial, and a split 4-GPU selector over the affected
  Gluon tests. Next concrete slice is final `make`/lit sanity, commit the
  post-merge runtime follow-up, push, and then decide whether a broader
  non-selector Gluon sweep is warranted.

- 2026-04-21 01:13 UTC: superseded the previous scaled-copy/direct-multicast
  xfail classification per user direction. Runtime tests green on main are not
  branch-xfail candidates. Implemented the backend fixes for larger-CGA
  scaled `tcgen05.copy` and 2x2 direct multicast accumulator load selection,
  removed the xfails, restored upstream lit coverage, and audited MLIR test
  symbols against `origin/main` merge-base (`TOTAL_MISSING_FUNCS 0`). Current
  active slice is ready to commit/push after final status review; next broad
  work can resume GB200/Gluon sweeps from a coverage-preserving branch.

- 2026-04-21 02:30 UTC: completed a static multi-CTA CGA backend audit and
  recorded it in
  `.codex/initiatives/tmem_linear_generalization/multi_cta_cga_backend_audit_20260421.md`.
  Active follow-up checklist:
  1. Add mixed-kernel lit coverage for TMA load/gather so unrelated TMA ops do
     not inherit `cta_group::2` merely because the module has two-CTA TMEM.
  2. Add return-side cluster-barrier coverage for `num_ctas > 2` kernels that
     mix two-CTA TMEM with other distributed shared-memory/TMA dependencies.
  3. Add PTX coverage proving 4-CTA TMEM copy/commit emits from each even CTA
     pair leader, not only CTA 0.
  4. Add instrumentation/barrier-recipient coverage for TMEM copy with
     outer-CGA barrier ownership.
  5. Keep module-wide mixed single-CTA/two-CTA TMEM as an explicit design
     restriction until all consumers derive instruction group from per-op or
     per-type state.

- 2026-04-21 02:35 UTC: user clarified module-wide two-CTA mode is a hardware
  restriction for all two-CTA-capable instructions, so TMA load/gather
  `getModuleTwoCTAs` use is not a gap. Implemented the real gap exposed by the
  follow-up 4-CTA probe: `tcgen05.copy.warpx2` shared-source validation now
  accepts canonical outer CTA-pair block bases for `128 * num_ctas` by 4
  source tiles. Added conversion and membar lit coverage for 4-CTA pair-leader
  lowering and pre-copy cluster-barrier insertion. Validation passed: required
  `make -j8`, then focused lit for
  `test/Conversion/tritongpu_to_llvm_blackwell.mlir` and
  `test/TritonNvidiaGPU/membar-cluster.mlir`.

- 2026-04-21 02:51 UTC: extended the multi-CTA coverage to the maximum
  practical CTA-per-CGA point used by this backend path: 16 CTAs. Conversion
  lit now covers 4/8/16 CTA `tcgen05.copy.warpx2` canonical outer block bases,
  and membar lit covers 16-CTA pre-copy cluster-barrier insertion. Validation
  passed: required `make -j8` no-op and focused lit for the two touched files
  passed `2/2`.

- 2026-04-21 04:16 UTC: completed correctness-first recovery for the requested
  Gluon example benchmark targets. Fixed the attention non-causal joined-N row
  max layout mismatch, the `OptimizePartitionWarps` memdesc-capture AxisInfo
  assertion, and the single-fragment narrow-N scaled-MMAv5 B-scale legality
  false negative hit by the example-5 reference persistent matmul. Validation
  passed: required `make -j8`; saved `triton-opt --run-reproducer` for the
  original partition-warps crash; exact attention and MoE repro nodeids; and
  four-GPU split sweep over examples 01 and 05 with `112/112` selected tests
  passing. Next concrete step is performance comparison/benchmarking for the
  now-green examples, unless the user asks for broader correctness sweeps first.

- 2026-04-21 06:46 UTC: completed another adversarial scaled-MMAv5 audit slice.
  Fixed the B-scale rematerialization gap where verifier accepted an
  unpadded/repeated-N32 scaled MMA because padded rematerialization was
  possible, but the rewrite refused any original B-scale descriptor with an
  additional live user. The rewrite now rematerializes a separate padded
  descriptor for the MMA and preserves the original descriptor for other users.
  Validation: required `make -j8`; exact new runtime nodeid passed; neighboring
  scaled selector passed `22 passed, 1593 deselected`; py-compile and
  `git diff --check` passed. Concurrent `TensorMemoryUtils.cpp` edits were not
  staged.

- 2026-04-21 06:47 UTC: completed the requested copy/ld/st/ld.red-focused
  adversarial audit slice. Found and fixed a real multicast-copy ownership
  false negative: 128-row two-CTA `TensorMemoryScalesLayout` destinations
  expose `block=1 -> (64, 0)`, but the copy support checker required
  `[[128, 0]]` for every multicast family. The checker now uses
  `min(row_extent / 2, 128)` for the pair-local selector. Validation passed:
  required `make -j8`; exact scales two-CTA copy nodeid; paired single/two-CTA
  scales copy nodeids; copy adversarial selector `31/31`; descriptor-chain
  ld/st selector `128 passed, 61 skipped`; ld.red selector `52/52`; lit
  `3/3`; `git diff --check`.

- 2026-04-21 08:33 UTC: completed TMEM structural fuzzing round 2, Lane
  Expansion, in report-only discovery mode. Required `make -j8` was no-op.
  Temporary `/tmp` Python/Gluon harnesses expanded around first-round failures
  and found stable adjacent variants in existing buckets: runtime
  `memdesc_index` chain2/chain3 crashes, helper chain0 false-branch and
  `16x64b` miscompiles, ld/st chain2 col-reverse `16x64b` miscompile, broader
  ld.red descriptor-chain plain-load fallback, and resource-valid 2CTA ld.red
  plain-load fallback. Current next discovery slice: promote/minimize the new
  adjacent repros as strict xfails where useful, and compare the 2CTA ld.red
  row against the checked-in two-CTA direct higher-rank positive before any
  repair work.

- 2026-04-21 08:43 UTC: completed TMEM structural fuzzing round 3 and promoted
  the highest-value new sentinels while preserving discovery-only mode. Four
  subagent lanes produced reports for dynamic `memdesc_index`, ld.red opcode
  loss, MMA/copy clean surfaces, and ld/st read-only descriptor-view
  minimization. New checked-in strict xfails:
  `generic-pass-dynamic-index-load-only-128x32`,
  `ldst-fz20260421-0003-chain2-col-reverse-64x32-16x64b`, and
  `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min`. Required
  `make -j8` was no-op; py-compile passed; collect-only found `21` structural
  fuzzer nodeids; exact new sentinels each xfailed; full structural fuzzer
  passed as `9 passed, 12 xfailed`. Existing clean-surface sanity sweeps passed
  for ld.red descriptor matrix (`50/50`), copy warpx2/scales (`103/103`), and
  scaled-MMA descriptor/narrow/tile selectors (`139/139`). Current active
  phase remains continuous fuzzing/cataloging with no backend fixes until
  findings stop or the user pivots.

- 2026-04-21 08:50 UTC: completed TMEM structural fuzzing round 4 sentinel
  promotion. Added eight strict xfails for non-overlapping round-four findings:
  helper/control-flow false branch `16x128b`, inline chain0, tuple-like mixed
  capture, layout-pressure `16x128b`, f16 ld/st subword chain2 identity, and
  2CTA indexed ld.red max/abs/NaN opcode-loss rows. Validation passed:
  py-compile, collect-only `29` nodeids, exact new sentinels `8 xfailed`, and
  full structural fuzzer `9 passed, 20 xfailed`. Report-only follow-ups:
  dynamic `memdesc_index` lit candidate policy, R4-D row/col chain1 optimizer
  crash with a crash-safe harness, and continued clean-negative vs opcode-loss
  separation.

- 2026-04-21 11:03 UTC: completed Round 14 Lane AE `ld/st` descriptor-algebra
  fuzzing in discovery-only mode. Report:
  `agents/fuzz_ldst_descriptor_algebra_round14.md`. Required `make -j8` was a
  no-op. Temporary rank-6/runtime-selector probe classified `12` rows as `1`
  pass, `2` `FZ-20260421-0001`, `2` `FZ-20260421-0003`, `5` clean
  direct-`ld/st` row-anchor diagnostics, and `2` clean rank-view diagnostics.
  Checked-in `ld/st` descriptor/rank/subword selector passed split-4 as
  `134 passed, 98 skipped`; adjacent copy/MMAv5 controls passed as
  `243 passed`. No new independent bucket was assigned. Active phase remains
  continuous discovery/cataloging with backend repairs deferred.

- 2026-04-21 11:11 UTC: completed Round 14 Lane AG
  `FZ-20260421-0013` minimization in discovery-only mode. Report:
  `agents/fuzz_scaled_fz0013_min_round14.md`. Required `make -j8` was a
  no-op. Temporary scaled-MMAv5 scale descriptor-view probe classified `33`
  fresh-subprocess rows as `19` runtime miscompiles, `5` passes, `2` clean
  unsupported diagnostics, and `7` diagnostic/probe-limited non-runtime rows.
  Smallest repeated row is local 1CTA non-FPSAN `N=128,K=128`, linear
  accumulator, B-scale `reshape -> trans -> reshape`, unpadded storage, no
  extra user; `3/3` repeats produced `16109/16384` mismatches with matching
  PTX/LLIR scaled-MMA opcodes. `FZ-0013` is broadened to cover A-scale and
  B-scale descriptor views; no new independent bucket was assigned. Active
  phase remains continuous discovery/cataloging with backend repairs deferred.

- 2026-04-21 11:15 UTC: completed Round 14 Lane AH compiler-only/lit/verifier
  fuzzing in discovery-only mode. Report:
  `agents/fuzz_lit_verifier_round14.md`. Required `make -j8` and
  `ninja triton-opt` were no-ops. Anchor lit files passed `5/5`; temporary
  `triton-opt` probes classified as `7` pass/clean expected diagnostics and
  `4` known-bucket failures covering `FZ-20260421-0001`,
  `FZ-20260421-0012`, and `FZ-20260421-0014`. 4/8/16 CTA verifier probes and
  matmul two-CTA consistency probes did not expose a new bucket. Active phase
  remains continuous discovery/cataloging with backend repairs deferred.

- 2026-04-21: local broad MMAv5 descriptor selector wrote
  `agents/fuzz_local_mma_descriptor_broad_round14.md`. The
  `(mma and (descriptor or runtime_selector or indexed_acc or acc_subslice_view or scale_descriptor)) and not reports`
  selector collected `253/1615` and passed split-4 as `239 passed,
  14 skipped`; no new bucket.

- 2026-04-21: local copy risk selector wrote
  `agents/fuzz_local_copy_risk_round14.md`. The
  `(cp_no_scales_warpx2 or cp_no_scales_twocta or cp_scales_warpx4 or cp_scales) and not reports`
  selector collected `135/1615` and passed split-4 as `135 passed`; no new
  bucket.

- 2026-04-21: local structural-fuzzer smoke gate wrote
  `agents/fuzz_local_structural_smoke_round14b.md`. Full
  `python/test/gluon/test_tmem_structural_fuzzer.py` reported
  `9 passed, 24 xfailed`; no new bucket.

- 2026-04-21 11:30 UTC: integrated Round 15 Lane AK generic
  memdesc/control-flow fuzzing in discovery-only mode. Report:
  `agents/fuzz_generic_memdesc_control_round15.md`. Required `make -j8` was a
  no-op. Temporary probes classified branch/helper-selected memdesc values
  feeding `ld/st`, copy, and plain-MMAv5 as existing
  `FZ-20260421-0001`, loop-carried helper-returned chain0 `ld/st` views as
  existing `FZ-20260421-0002`, and branch-selected scaled-MMAv5
  descriptor-view scale rows as existing `FZ-20260421-0013`. Chain1/chain2
  loop-carried `ld/st` controls passed and chain0 copy/MMAv5 controls produced
  clean unsupported diagnostics. No new independent bucket was assigned.

- 2026-04-21 11:30 UTC: integrated Round 15 Lane AM scaled multi-MMA
  composition fuzzing. Report:
  `agents/fuzz_scaled_multi_mma_round15.md`. Temporary probe collected `12`
  rows and final run reported `7 passed, 5 failed`. Green controls covered
  multiple scaled MMA ops, scale descriptor reuse, static accumulator
  subslices, and format variation. Three scale descriptor-view rows stayed
  under existing `FZ-20260421-0013`. New report-only candidate
  `FZ-20260421-0015` covers runtime branch selection between two direct
  B-scale TMEM descriptors feeding scaled-MMAv5: compilation succeeds,
  PTX/LLIR scaled-MMA opcodes match, but selector `0/1` produce
  `16381/16384` and `16380/16384` mismatches with `96` NaNs.

- 2026-04-21 11:30 UTC: local twoCTA/high-CGA runtime selector wrote
  `agents/fuzz_local_twocta_highcga_round15.md`. Selector
  `(twocta or cga_roundtrip or layout_in_4cta_context) and not reports`
  collected `331/1615` and passed split-4 as `294 passed, 37 skipped`; no new
  `FZ-20260421-0010` or `FZ-20260421-0014` exposure.

- Active next concrete slice: minimize `FZ-20260421-0015` in discovery mode
  across same-object versus distinct-object scale descriptors, A-scale versus
  B-scale runtime selection, branch versus loop/pass forms, selector
  constant-folding, and single-use versus multi-use scale descriptors. Do not
  start backend repair until this candidate is minimized and the active fuzzing
  lanes stop finding new bugs or the user pivots.

- 2026-04-21 11:31 UTC: completed first `FZ-20260421-0015` minimization slice
  in discovery-only mode. Report:
  `agents/fuzz_scaled_fz0015_min_round15.md`. Temporary probe collected `10`
  rows and ran split-4 as `5 passed, 5 failed`. Green controls: direct
  B-scale, constexpr distinct B-scale selection, same-object dynamic B-scale
  selection, and A-scale dynamic selection. Failing rows require distinct
  B-scale descriptors selected by runtime control flow before scaled-MMAv5;
  branch and loop forms fail, and an extra selected-scale user does not avoid
  the issue. Next minimization: parent-index versus independent allocations,
  `N/K/format` sweep, and TTGIR selected-scale SSA inspection.

- 2026-04-21 11:34 UTC: completed local `FZ-20260421-0015` TTGIR
  discriminator. Report: `agents/fuzz_scaled_fz0015_ttgir_round15.md`.
  Parent-view rows are probe-limited because tensor-memory `memdesc_subslice`
  forms fail before lowering even for direct controls. TTGIR inspection shows
  direct/constexpr controls pass concrete B-scale memdescs into
  `ttng.tc_gen5_mma_scaled`, while branch and loop failures pass merged
  B-scale memdesc SSA values (`arith.select` or `scf.for` iter_arg results).
  Next slice: generate a small IR reproducer around the selected B-scale
  operand and inspect the scaled-MMAv5 lowering/planner path, still without
  backend repair.

- 2026-04-21 11:35 UTC: integrated Round 16 Lane AN `FZ-20260421-0015`
  minimization. Report: `agents/fuzz_fz0015_min_round16.md`. The strongest
  current hypothesis is scaled-MMAv5 lowering/codegen mishandles a
  runtime-selected distinct B-scale TMEM descriptor value. Selected-descriptor
  loads pass `4/4`, so the descriptor contents are readable outside scaled
  MMA. The issue reproduces across branch, loop, helper, pass-through,
  extra-user, two-MMA, `use_acc=False`, `N/K`, `mxfp4`, and `nvfp4` variants.
  Continue discovery by extracting a small IR reproducer and auditing the
  B-scale operand path before backend repair.

- 2026-04-21 11:38 UTC: local Round 17 copy/ldst selector stayed green.
  Report: `agents/fuzz_local_copy_ldst_round17.md`. Required `make -j8` was a
  no-op. Checked-in structural fuzzer reported `9 passed, 24 xfailed`.
  Runtime-matrix selector
  `(cp_no_scales or cp_scales or ldst_descriptor_compositions or ldst_descriptor_roundtrip) and not reports`
  collected `344/1615` rows and passed split-4 as
  `289 passed, 55 skipped`. Continue active Round 17 subagent lanes.

- 2026-04-21 11:39 UTC: integrated Round 17 Lane AO `FZ-20260421-0015`
  lowering audit. Report:
  `agents/fuzz_fz0015_lowering_audit_round17.md`. The saved TTGIR is
  verifier-clean, selected B-scale loads pass, and LLVM/PTX preserve the
  runtime-selected B-scale memdesc as a scalar `select` / `selp.b32` feeding
  the SFB address operand. Likely site is scaled MMAv5 LLVM lowering around
  `convertScaledDot` / `createScaledGen5MMA` in `MMAv5.cpp`, or an unmodeled
  SFB-address operand constraint. Keep backend repair deferred until active
  fuzz lanes stop yielding new evidence or the user pivots.

- 2026-04-21 11:41 UTC: integrated Round 17 Lane AP copy/ldst mixed fuzzing
  and local clean-boundary selector. Reports:
  `agents/fuzz_copy_ldst_mixed_round17.md`,
  `agents/fuzz_clean_boundaries_round17.md`, and
  `agents/fuzz_local_clean_boundaries_round17.md`. AP temporary probe passed
  `7/7`; adjacent copy/ldst checked-in selector passed as
  `179 passed, 61 skipped`; local clean-boundary selector collected `199/1615`
  and passed as `199 passed`. No new bucket. AP-007 reproduces existing
  `FZ-0014`, while AP-004/AP-006 are negative contrasts: two independent 2CTA
  copy regions can pass under different wait/readback ordering, so the
  proxy-fence failure depends on exact mbarrier use interval/ordering rather
  than simply region count. AQ clean-boundary inventory selected `157/1615`
  checked-in rows and passed split-4, then its focused temporary probe passed
  descriptor-view, high-CGA, subword, non-f32 `ld.red`, scale-shape, and
  parent-view subslice boundaries without late illegal ops.

- 2026-04-21 11:48 UTC: local Round 18 selector
  `(ldst_descriptor_compositions or ldred_descriptor or cp_scales) and not reports`
  collected `62/1615` rows and passed split-4 as `62 passed`. Report:
  `agents/fuzz_local_ldst_cpscales_round18.md`. No new bucket; this is a green
  baseline for descriptor `ld/st` composition and scale-copy runtime rows while
  subagents probe proxy-fence intervals and dynamic descriptor clean-boundary
  combinations.

- 2026-04-21 11:54 UTC: local Round 18 selector
  `(mma_scaled and not reports and not fz0015 and not descriptor_view) or (mma_twocta and not reports_clean and not reports and not i8)`
  collected `342/1615` rows and passed split-4 as `342 passed`. Report:
  `agents/fuzz_local_mma_scaled_round18.md`. No new bucket; this is a broad
  green MMAv5/scaled-MMAv5 runtime baseline excluding report-only
  `FZ-20260421-0015` selected-B-scale rows.

- 2026-04-21 11:56 UTC: integrated Round 18 Lane AR and Lane AS reports:
  `agents/fuzz_proxy_fence_intervals_round18.md` and
  `agents/fuzz_dynamic_clean_boundaries_round18.md`. No new independent
  `FZ-*`. AR narrowed `FZ-0014`: sequential wait-before-next-init 2CTA copy
  regions reproduce without readback, while init-all-before-copy/wait shapes
  pass with two and three regions. AS strengthened `FZ-0001` with a load-only
  dynamic index illegal-op row, `FZ-0002` with tuple-like dynamic descriptor
  capture wrong results, `FZ-0010` with a clean high-CGA diagnostic row, and
  `FZ-0015` with runtime-selected direct B-scale scaled-MMAv5 mismatch. AS
  checked-in selector passed as `159 passed, 11 xfailed`; AR adjacent
  checked-in selector passed as `63 passed`.

- 2026-04-21 11:57 UTC: integrated Round 18 proxy-fence plain-mbarrier
  follow-up report:
  `agents/fuzz_proxy_fence_plain_mbarrier_round18.md`. No new bucket. This
  sharpens `FZ-0014`: one legal 2CTA no-scales copy-tracked mbarrier plus one
  independent plain mbarrier interval reproduces in both copy-then-plain and
  plain-then-copy orderings, so two copy-tracked regions are not required.

- 2026-04-21 11:58 UTC: integrated Round 18 proxy-fence plain-only follow-up
  report: `agents/fuzz_proxy_fence_plain_only_round18.md`. No new bucket. This
  materially revises `FZ-0014`: the proxy-fence insertion failure is not
  TMEM-copy-specific. Two sequential plain init/arrive/wait mbarrier intervals
  with no TMEM operations reproduce; initializing both mbarriers before either
  arrive/wait use passes.

- 2026-04-21 12:05 UTC: local Round 19 sanity selectors passed. Report:
  `agents/fuzz_local_multicast_round19.md`. Runtime-matrix 2CTA TMA descriptor
  selector collected `10/1615` and passed as `10 passed`; `test_core.py`
  multicast selector collected `9/18114` and passed as `9 passed` with one
  empty split shard. No new bucket.

- 2026-04-21 12:09 UTC: integrated Lane AV dynamic subslice/mixed-consumer
  report and local structural xfail sentinel report:
  `agents/fuzz_dynamic_subslice_mixed_consumers_round19.md` and
  `agents/fuzz_local_structural_xfail_round19.md`. No new bucket. AV
  strengthens `FZ-0007`: selected `memdesc_subslice` load is correct, plain
  MMAv5 is correct, but scaled-MMAv5 low-subslice accumulator use miscompiles.
  AV's late B-scale mixed-consumer extension passed branch/helper/loop selected
  direct B-scale descriptors for both selectors, a negative contrast for
  `FZ-0015`. Local structural selector collected `11/1648` and stayed
  `11 xfailed`.

- 2026-04-21 12:14 UTC: local B-scale positive controls passed. Report:
  `agents/fuzz_local_bscale_controls_round19.md`. Selector
  `mma_scaled and bscale and not reports` collected `3/1615` rows and passed
  as `3 passed`; no new bucket.

- 2026-04-21 12:18 UTC: integrated Lane AU plain-only proxy-fence reproducer
  report: `agents/fuzz_proxy_fence_plain_reproducer_round19.md`. Saved
  reproducer
  `/tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir`.
  Branch, upstream main `dea2e9d7324309fdc9198144669621f57f3704a2`, and
  merge-base `2c7ce4925d37802dd84dfde1f6458cae19485617` all fail with the same
  proxy-fence insertion diagnostic. Classification: `FZ-0014` is preexisting
  upstream and not branch-specific; TMEM copy/mbarrier tests newly expose it.

- 2026-04-21 12:20 UTC: local `ld.red` selector started. Report:
  `agents/fuzz_local_ldred_round19.md`. Selector
  `ld_red and not reports and not resource` collected `243/1615`; split-4
  runtime completed as `237 passed, 6 failed`. Failures reproduce existing
  `FZ-20260421-0012` M64 row/col-permuted split-N destination-layout coverage
  gap; no new bucket. Focused exact reruns show `row_reverse_n32-min` fails
  while `col_reverse_n32-min` and explicit `col_reverse_n32` split-N controls
  pass.

- 2026-04-21 12:23 UTC: integrated Lane AW `FZ-0015` side-channel report:
  `agents/fuzz_fz0015_side_channel_round19.md`. No new bucket. Exact selected
  distinct B-scale branch/helper/loop rows still miscompile with selected-scale
  load side channel, but every side-channel load has `0/512` byte mismatches.
  Same-global-source rows still fail; accumulator-first allocation-order
  variant passes all six selected B-scale rows. Current strongest hypothesis is
  allocation-order-sensitive B-scale address/SFB operand encoding or descriptor
  base rematerialization.

- 2026-04-21 12:30 UTC: local Round 20 descriptor roundtrip and copy/scales
  selector passed/skipped. Report: `agents/fuzz_local_roundtrip_copy_round20.md`.
  Selector
  `(ldst_descriptor_roundtrip or cp_scales or cp_no_scales_warpx2_01_23 or cp_no_scales_warpx4) and not reports and not tmem_oor`
  collected `102/1615` and completed as `51 passed, 51 skipped`; no new bucket.

- 2026-04-21 12:35 UTC: local Round 20 2CTA MMAv5/scaled accumulator-subslice
  selector passed. Report: `agents/fuzz_local_mma_twocta_round20.md`.
  Selector collected `90/1615` and completed as `90 passed`; no new bucket.

- 2026-04-21 12:40 UTC: integrated Lane AX allocation-order report:
  `agents/fuzz_fz0015_allocation_order_round20.md`. No new bucket. `FZ-0015`
  follows whether the real accumulator is allocated after all scale descriptors
  versus before/between scale groups. All-scale-before-real-acc fails even with
  dummy scale/acc allocations between scales and real acc; accumulator before B
  scales or between A/B scales passes. Side-channel B-scale loads are clean in
  every row.

- 2026-04-21 12:40 UTC: local clean-negative/xfail selector passed/xfailed as
  expected. Report: `agents/fuzz_local_clean_xfail_round20.md`. Selector
  collected `208/1648` and completed as `197 passed, 11 xfailed`; no unexpected
  pass/fail and no new bucket.

- 2026-04-21 12:47 UTC: local Round 21 higher-rank/multidim descriptor selector
  completed. Report: `agents/fuzz_local_higher_rank_round21.md`. Selector
  collected `102/1615` and completed as `82 passed, 20 skipped`; no new bucket.

- 2026-04-21 12:03 UTC: Lane AY report
  `agents/fuzz_fz0015_ir_compare_round21.md` integrated. It sharpens
  `FZ-20260421-0015`: selected B-scale SSA and side-channel loads are correct;
  the failing scaled-MMAv5 allocation-order row lowers acc/A-scale/B-scale
  operands into the same low-offset neighborhood, unlike the passing controls.

- 2026-04-21 12:03 UTC: Lane AZ report
  `agents/fuzz_ldred_m64_min_round21.md` integrated. It sharpens
  `FZ-20260421-0012`: M64 f32 row-permuted `ld.red` fails in destination
  layout planning, column-only permutations pass, and explicit load variants
  do not rescue row-permuted rows.

- 2026-04-21 12:03 UTC: local scaled-MMAv5 static-layout baseline completed.
  Report: `agents/fuzz_local_scaled_layouts_round21.md`. Selector collected
  `91/1615` and passed split-4 as `91 passed`; no new bucket.

- 2026-04-21 12:03 UTC: local split-N/fixed-offset baseline completed.
  Report: `agents/fuzz_local_splitn_fixed_round22.md`. Selector collected
  `33/1615` and passed split-4 as `33 passed`; no new bucket.

- 2026-04-21 12:03 UTC: local structural fuzzer guardrail completed.
  Report: `agents/fuzz_local_structural_round22.md`. Selector collected
  `30/33` and passed split-4 as `9 passed, 21 xfailed`; no unexpected
  pass/fail and no new bucket.

- 2026-04-21 12:22 UTC: Lane BB `ld.red` layout fuzzing completed. Report:
  `agents/fuzz_ldred_layouts_round22.md`. No new bucket; broad selector
  completed as `237 passed, 6 failed`, all existing `FZ-0012`, with additional
  fallback, descriptor-chain, M64 modifier, and M=128 contrast controls passing.

- 2026-04-21 12:58 UTC: Lane BA scaled dynamic scale descriptor fuzzing
  completed. Report: `agents/fuzz_scaled_dynamic_scales_round22.md`. No new
  bucket; `10` direct selected-scale controls passed and `6` descriptor-view
  scale rows sharpened existing `FZ-0013`, with clean side-channel scale loads.

- 2026-04-21 12:55 UTC: Lane BC copy/ld-st generic descriptor fuzzing
  completed. Report: `agents/fuzz_copy_ldst_generic_round22.md`. No new
  bucket; temporary probe sharpened `FZ-0001` branch-yielded copy descriptors
  and `FZ-0002` multi-consumer branch-selected ld/st descriptors. Checked-in
  selector ran as `254 passed, 61 skipped, 14 xfailed`; scale-copy/high-CGA
  selector passed as `36 passed`.

- 2026-04-21 13:05 UTC: local plain MMAv5 baseline completed. Report:
  `agents/fuzz_local_plain_mma_round23.md`. Selector collected `255/1615` and
  passed split-4 as `255 passed`; no new bucket.

- 2026-04-21 13:18 UTC: local subword baseline completed. Report:
  `agents/fuzz_local_subword_round23.md`. Selector collected `42/1615` and
  passed split-4 as `42 passed`; no new bucket.

- 2026-04-21 12:20 UTC: Lane BE scaled accumulator fuzzing completed. Report:
  `agents/fuzz_scaled_accumulator_round23.md`. No new bucket; checked-in
  accumulator selector passed `206` rows, broader scaled-MMAv5 selector passed
  `243` rows, and dynamic selected accumulator mixed-consumer rows sharpened
  existing `FZ-0007`.

- 2026-04-21 12:32 UTC: local ld/st descriptor guardrail completed. Report:
  `agents/fuzz_local_ldst_descriptor_round23.md`. Selector collected
  `149/1615` and completed as `88 passed, 61 skipped`; no new bucket.

- 2026-04-21 12:13 UTC: Round 23 Lane BF compiler-only lit audit completed.
  Report: `agents/fuzz_compiler_lit_round23.md`. Required `make -j8` and
  `ninja triton-opt` were no-ops. Selected lit tests passed for TMEM layouts,
  MMAv5 lowering, proxy fence insertion, interleave/hoist/NVWS TMEM, and
  memdesc subview splitting. New candidate `FZ-20260421-0016`:
  `test/Conversion/relayout_tritongpu.mlir` parse-time verifier crash for
  `ttng.tmem_alloc` with an unencoded tensor source and tensor-memory-scales
  result. Saved repro replays stayed in existing `FZ-0014`, `FZ-0008`, and
  `FZ-0009`.

- 2026-04-21 13:20 UTC: Lane BD high-CGA fuzzing completed. Report:
  `agents/fuzz_high_cga_round23.md`. No new bucket; it reconfirmed `FZ-0010`
  across ld/st, `ld.red`, no-scales copy, and scales-copy local 1CTA/2CTA
  layouts in 4/8/16 CTA contexts while high-CGA controls passed.

- 2026-04-21 12:42 UTC: local multi-CTA guardrail completed. Report:
  `agents/fuzz_local_multicta_round23.md`. Selector collected `430/1615` and
  completed as `393 passed, 37 skipped`; no new bucket.

- 2026-04-21 13:30 UTC: local compiler lit baseline completed. Report:
  `agents/fuzz_local_lit_baseline_round24.md`. Four TMEM lit tests passed; no
  new bucket.

- 2026-04-21 13:40 UTC: local clean-boundary sweep completed. Report:
  `agents/fuzz_local_clean_boundary_round24.md`. Selector collected `184/1615`
  and passed split-4 as `184 passed`; no new bucket.

- 2026-04-21 13:35 UTC: Lane BI structural generator prototype completed.
  Report: `agents/fuzz_structural_generator_round24.md`. Temporary generator
  ran 12 cases: `7` green, `3` clean unsupported, `2` existing `FZ-0012`; no
  new bucket.
- 2026-04-21 Round 24 Lane BG `FZ-0016` minimization completed. Report:
  `agents/fuzz_fz0016_min_round24.md`. No backend/compiler repair was
  attempted. The crash is parse/verifier-time, not relayout-pass-time:
  `/tmp/tmem_fz0016_round24/minimal_unencoded_scales_i8.mlir` aborts under
  parse-only `triton-opt -split-input-file --mlir-disable-threading` through
  `TMEMAllocOp::verify -> verifyTMEMOperand -> computeTMemLdStEncodingInfo ->
  toLinearEncoding`. Neighboring matrix: unencoded `i8` operands abort for
  standard/linear/scales TMEM results; unencoded `f16`/`f32` abort for
  standard/linear; invalid `f16`/`f32` scales rows keep the clean bitwidth
  diagnostic; encoded operands produce ordinary compatibility diagnostics.
  Continue discovery-only fuzzing and defer the verifier fix until the fuzzing
  campaign stops finding new failures or the user pivots.
- 2026-04-21 Round 24 Lane BH mixed scaled-MMAv5 operand fuzzing completed.
  Report: `agents/fuzz_scaled_mixed_operands_round24.md`. No new independent
  bucket. Mixed direct selected B-scale plus runtime-selected accumulator-parent
  rows passed; mixed scale descriptor-view rows miscompiled with clean
  side-channel loads and stay under `FZ-0013`; the exact previous `FZ-0015`
  side-channel reproducer still fails. Keep `FZ-0007`, `FZ-0013`, and
  `FZ-0015` separate for repair planning.
- 2026-04-21 local structural crash guardrail completed. Report:
  `agents/fuzz_local_structural_crash_round24.md`. Selector
  `allocator_crash or optimizer_crash` remained `3 xfailed`; existing
  `FZ-0005`, `FZ-0008`, and `FZ-0009` crash sentinels did not unexpectedly
  pass or change mode.

- 2026-04-21 13:40 UTC: Lane BG `FZ-0016` minimization completed. Report:
  `agents/fuzz_fz0016_min_round24.md`. Minimal parse-only reproducer saved at
  `/tmp/tmem_fz0016_round24/minimal_unencoded_scales_i8.mlir`; no new bucket
  beyond sharpened `FZ-0016`.

- 2026-04-21 13:50 UTC: local selector-permuted copy guardrail completed.
  Report: `agents/fuzz_local_selector_copy_round24.md`. Selector collected
  `4/1615` and passed as `4 passed`; no new bucket.

- 2026-04-21 13:55 UTC: Lane BH mixed scaled-MMAv5 operands completed.
  Report: `agents/fuzz_scaled_mixed_operands_round24.md`. No new bucket:
  direct selected-B/runtime-selected-accumulator-parent rows passed; all
  scale descriptor-view rows miscompiled with clean side-channel loads and
  classify as existing `FZ-0013`; exact `FZ-0015` side-channel reproducer
  remains live and distinct.

- 2026-04-21 13:55 UTC: local structural guardrail completed. Report:
  `agents/fuzz_local_structural_round24.md`. Selector collected `30/33` and
  completed as `9 passed, 21 xfailed`; no new bucket.

- 2026-04-21 13:45 UTC: Lane BH mixed scaled-MMAv5 operand fuzzing completed.
  Report: `agents/fuzz_scaled_mixed_operands_round24.md`. No new bucket;
  scale-view rows stayed `FZ-0013`, exact `FZ-0015` still fails, and nearby
  direct mixed selected-B-scale/accumulator rows passed.

- 2026-04-21 14:00 UTC: local structural crash guardrail completed. Report:
  `agents/fuzz_local_structural_crash_round24.md`. Selector completed as
  `3 xfailed`; no changed crash mode.

- 2026-04-21 14:10 UTC: local no-scale copy baseline completed. Report:
  `agents/fuzz_local_copy_round25.md`. Selector collected `201/1615` and
  completed as `197 passed, 4 skipped`; no new bucket.

- 2026-04-21 14:18 UTC: local scale-copy baseline completed. Report:
  `agents/fuzz_local_cpscales_round25.md`. Selector collected `33/1615` and
  passed split-4 as `33 passed`; no new bucket.

- 2026-04-21 14:22 UTC: Lane BK `FZ-0016` verifier code audit completed.
  Report: `agents/fuzz_fz0016_code_audit_round25.md`. No new bucket; broadened
  `FZ-0016` to shared alloc/load/store verifier crashes for unencoded register
  tensors.

- 2026-04-21 14:25 UTC: local `relayout_tritongpu.mlir` baseline completed.
  Report: `agents/fuzz_local_fz0016_relayout_round25.md`. Reproduces existing
  `FZ-0016`; no new bucket.

- 2026-04-21 14:24 UTC: local descriptor ld/st non-roundtrip baseline
  completed. Report: `agents/fuzz_local_ldst_descriptor_nonroundtrip_round25.md`.
  Executing selector collected `53/1615` and passed split-4 as `53 passed`;
  no new bucket.

- 2026-04-21 14:35 UTC: local scaled-MMAv5 descriptor-view baseline
  completed. Report: `agents/fuzz_local_scaled_descriptor_view_round25.md`.
  Selector passed as `4 passed, 1611 deselected`; no new bucket.

- 2026-04-21 14:28 UTC: Lane BJ dynamic structural generator completed.
  Report: `agents/fuzz_structural_generator_dynamic_round25.md`. No new
  bucket; generated seeds sharpened `FZ-0001`/`FZ-0002` and produced a
  promotion-ready seed table.

- 2026-04-21 14:35 UTC: local dynamic structural guardrail completed. Report:
  `agents/fuzz_local_dynamic_structural_round25.md`. Selector collected
  `12/33` and completed as `12 xfailed`; no new bucket.

- 2026-04-21 14:35 UTC: local scaled descriptor-view control completed.
  Report: `agents/fuzz_local_scaled_descriptor_view_round25.md`. Selector
  passed as `4 passed`; no new bucket.

- 2026-04-21 14:28 UTC: Lane BJ dynamic descriptor structural generator
  completed. Report: `agents/fuzz_structural_generator_dynamic_round25.md`.
  No new bucket; sharpens `FZ-0001` and `FZ-0002` across dynamic descriptor
  SSA axes and ld/st, copy, load-only, and `ld.red` consumers.

- 2026-04-21 14:30 UTC: local descriptor load/store non-roundtrip baseline
  completed. Report:
  `agents/fuzz_local_ldst_descriptor_nonroundtrip_round25.md`. Selector
  collected `53/1615` and passed split-4 as `53 passed`; no new bucket.
  Roundtrip selector collected `51/1615` but all selected rows pre-skipped at
  the current TMEM allocation limit.

- 2026-04-21 14:00 UTC: local structural crash repro guardrail completed.
  Report: `agents/fuzz_local_structural_crash_round24.md`. Selector
  `allocator_crash or optimizer_crash` completed as `3 xfailed`; no changed
  failure mode and no new bucket.

- 2026-04-21 12:25 UTC: Round 25 Lane BL high-CGA mixed ownership completed.
  Report: `agents/fuzz_high_cga_mixed_round25.md`. Discovery-only, no backend
  repair. Same-kernel TMA multicast plus local copy and same-kernel 2CTA MMAv5
  multicast/commit plus local copy rows produced `10` existing
  `FZ-20260421-0010` CTA-count diagnostics and no new crash/miscompile. The
  scaled-MMAv5/local-view attempt was harness-limited before the local check.
  Local scaled-layout selector rerun stayed green as `91 passed`.

- 2026-04-21 15:05 UTC: local Round 26 `ld.red` runtime slice completed.
  Report: `agents/fuzz_local_ldred_round26.md`. Required `make -j8` was a
  no-op. Selector `ld_red and not reports and not resource` collected
  `243/1615` and ran split-4 as `237 passed, 6 failed`. The six failures are
  existing `FZ-20260421-0012` M64 row-permuted `unsupported dst layout` rows;
  no new bucket or changed failure mode was found. Continue Round 26 subagent
  lanes and local non-overlapping runtime slices.

- 2026-04-21 15:20 UTC: local Round 26 MMAv5-adjacent guardrail completed.
  Report: `agents/fuzz_local_mma_guard_round26.md`. Required `make -j8` was a
  no-op. Selector `mma and not mma_scaled and not reports and not clean and not
  resource` collected `337/1615` and ran split-4 as
  `323 passed, 14 skipped`. No new bucket was found. Continue Round 26
  subagent lanes and local non-overlapping runtime slices.

- 2026-04-21: Round 26 verifier robustness lane completed. Report:
  `agents/fuzz_verifier_round26.md`. No new bucket; broadened existing
  `FZ-20260421-0016` to shared `verifyTMEMOperand` crashes for unencoded
  tensor operands/results across `ttng.tmem_alloc`, `ttng.tmem_load`, and
  `ttng.tmem_store`. Clean copy, malformed memdesc, wrong-shape, wrong-space,
  rank, and encoded-unsupported diagnostics remained typed diagnostics.

- 2026-04-21: Round 27 structural-fuzzer rerun completed. Report:
  `agents/fuzz_structural_rerun_round27.md`. Full
  `python/test/gluon/test_tmem_structural_fuzzer.py` split across four GPUs
  stayed stable as `9 passed, 24 xfailed`; no XPASS or unexpected failure.
  Continue the active Round 26 dynamic SSA and high-CGA scaled lanes.

- 2026-04-21: Round 26 high-CGA scaled-MMAv5 mixed-ownership lane completed.
  Report: `agents/fuzz_high_cga_scaled_round26.md`. Required `make -j8` was a
  no-op. The repaired harness first proved legal high-CGA scaled-MMAv5 controls
  for `num_ctas=4`, `8`, and `16` with the expected scaled opcode and torch
  reference match. Adding local 1CTA/2CTA descriptor-view `st`, `ld`,
  `ld.red`, or `tcgen05.copy` in the same kernels produced `24` clean
  `FZ-20260421-0010` CTA-count diagnostics. No new crash, miscompile,
  assembler failure, or scaled-MMAv5-specific bucket was found.

- 2026-04-21: Round 28 broad `ld.red` runtime slice completed. Report:
  `agents/fuzz_ldred_round28.md`. Split-4 selector
  `ld_red and not reports and not resource` completed as
  `237 passed, 6 failed`; exact rerun confirmed the same six known
  `FZ-20260421-0012` M64 row/column-permuted planner rows. No new independent
  bucket or changed failure mode was found.

- 2026-04-21: Round 26 dynamic descriptor SSA/control-flow lane completed.
  Report: `agents/fuzz_dynamic_ssa_round26.md`. Seven temporary runtime rows
  covered nested helpers, tuple-like returns, loop-carried descriptors,
  runtime indices, `ld/st`, `ld.red`, copy, mixed consumers, and plain-MMAv5
  accumulator consumers. No new bucket: copy/runtime-index rows broaden
  existing `FZ-20260421-0001`, wrong-output `ld/st`/`ld.red` rows broaden
  existing `FZ-20260421-0002`, and the plain-MMAv5 branch-selected accumulator
  control passed. Follow-up direct descriptor controls passed for branch
  selected `ld/st`, same-object branch `ld/st`, and branch-selected `ld.red`,
  narrowing `FZ-0002` toward descriptor-view composition crossing SSA/control
  flow.

- 2026-04-21: Round 29 broad scaled-MMAv5 runtime guardrail completed. Report:
  `agents/fuzz_local_mma_scaled_round29.md`. Selector
  `mma_scaled and not reports and not resource` collected `243/1615` and
  passed split-4 as `243 passed`. No compiler crash, verifier failure, runtime
  miscompile, opcode mismatch, or new bucket was found.

- 2026-04-21: Round 29 no-scale copy runtime guardrail completed. Report:
  `agents/fuzz_local_copy_round29.md`. Selector
  `cp_no_scales and not reports and not high and not cga and not selector and
  not subword` collected `201/1615` and passed split-4 as
  `197 passed, 4 skipped`. No new bucket was found.

- 2026-04-21: Round 29 dynamic arm-symmetry lane completed. Report:
  `agents/fuzz_dynamic_arm_symmetry_round29.md`. No new bucket. Selector 0 and
  selector 1 both fail for descriptor-view `ld/st` and `ld.red` consumers;
  same-object view branches and loop-carried no-switch rows still miscompile;
  direct offset-only/same-object descriptors pass across branch/SSA structures.
  This further narrows `FZ-0002` to descriptor-view chain materialization
  across SSA/control flow. Copy dynamic/branch rows remain `FZ-0001`.
  Follow-up chain-shape probes found direct `ld/st` chain0 wrong-code at
  `M=128` without branch, loop, or parent indexing; alternate identity chains
  pass and both bad/green controls emit the same public packet shape. Treat
  this as accepted-but-misplanned descriptor-view layout mapping adjacent to
  `FZ-0003`, with SSA/control-flow rows as amplifications.

- 2026-04-21: Round 29 structural smoke completed. Report:
  `agents/fuzz_local_structural_round29.md`. Checked-in structural fuzzer on
  GPU 0 stayed stable as `9 passed, 24 xfailed`; no XPASS or unexpected
  failure.

- 2026-04-21: Round 29 clean-boundary verifier lane completed. Report:
  `agents/fuzz_clean_boundary_round29.md`. New candidate
  `FZ-20260421-0017`: encoded `i64` and `f64` TMEM load/store operands reach
  `-triton-tensor-memory-allocation` and abort in `lowerTMemLdSt` on
  `Assertion 'bitwidth == 32' failed`. Runtime clean-boundary selector passed
  as `184 passed`; `test/TritonNvidiaGPU/invalid.mlir` lit baseline passed;
  other malformed shape, memory-space, CTA-count, and layout probes produced
  typed diagnostics. Python frontend probes for `torch.float64` and
  `torch.int64` TMEM round trips abort during JIT at the same assertion; dead
  uninitialized 64-bit alloc is eliminated, but initialized alloc, standalone
  store, and standalone load crash.

- 2026-04-21: Round 29 memdesc-index expansion lane completed. Report:
  `agents/fuzz_memdesc_index_round29.md`. No new bucket. Runtime
  `parent.index(ttgl.load(selector))` leaves illegal `ttg.memdesc_index` for
  every direct consumer tested: load, store, copy, `ld.red`, and mixed
  load/store/`ld.red`. Direct distinct-branch `tcgen05.copy` also fails late
  with illegal `ttg.memdesc_index`, while direct branch load/store/`ld.red`/
  mixed controls pass. Descriptor-chain read/reduction miscompiles remain
  existing descriptor-view semantic buckets; descriptor-chain copy rows are
  clean unsupported diagnostics.

- 2026-04-21: Round 29 Python runtime bitwidth follow-up completed. Report:
  `agents/fuzz_bitwidth_runtime_round29.md`. No new bucket beyond
  `FZ-20260421-0017`. Subprocess-isolated Gluon probes show `float64` and
  `int64` roundtrip, store-only, and load-only TMEM access all abort with
  `ASSERT_BITWIDTH_32`, so both store and load sides need a clean 64-bit
  policy.

- 2026-04-21: Round 30 local descriptor `ld/st` guardrail completed. Report:
  `agents/fuzz_local_ldst_descriptor_round30.md`. Selector
  `ldst_descriptor and not reports and not roundtrip` collected `53/1615` and
  passed split-4 as `53 passed`. No new bucket was found.

- 2026-04-21: Round 30 local clean-boundary runtime guardrail completed.
  Report: `agents/fuzz_local_clean_boundary_round30.md`. Selector
  `reports_clean or clean_unsupported or clean_error or tmem_oor` collected
  `184/1615` and passed split-4 as `184 passed`. No new bucket was found.

- 2026-04-21: Round 30 local TMEM lit baseline completed. Report:
  `agents/fuzz_local_lit_round30.md`. `tmem_layouts.mlir` and `invalid.mlir`
  both passed after `ninja triton-opt`. No new bucket was found.

- 2026-04-21: Round 30 FZ-0017 expansion completed. Report:
  `agents/fuzz_fz0017_round30.md`. No new bucket; `FZ-20260421-0017` now
  covers legacy/linear layouts, 1CTA/2CTA, initialized alloc, store-only,
  load-only, roundtrip, descriptor-view load/store at LLVM conversion, and
  Python/Gluon frontend kernels for `i64`/`f64`. Clean contrasts: 64-bit
  `ld.red` rejects with the f32-only diagnostic, 64-bit `ttng.tmem_copy`
  passes generated allocation+LLVM probes, and high-CGA rows reject under
  existing CTA-count diagnostics.

- 2026-04-21: Round 30 direct branch copy minimization completed. Report:
  `agents/fuzz_copy_branch_round30.md`. No new bucket; sharpens
  `FZ-20260421-0001`. Same-parent, branch-selected distinct indices feeding
  `ttng.tmem_copy` fail for selector `0/1`, readback on/off, and 1CTA/2CTA.
  Same-parent same-index branch selection, distinct unindexed TMEM objects,
  and same-index different-parent controls pass at `N=32`.

- 2026-04-21: Round 26 dynamic descriptor SSA lane completed. Report:
  `agents/fuzz_dynamic_ssa_round26.md`. No new bucket. Runtime-index `ld.red`
  and branch-yielded copy descriptors sharpen existing `FZ-20260421-0001`;
  nested-helper/loop-carried `ld/st` and branch/mixed-consumer `ld.red`
  wrong-output rows sharpen existing `FZ-20260421-0002`. Plain-MMAv5
  branch-selected accumulator descriptor passed.

- 2026-04-21: Round 26 high-CGA scaled-MMAv5 mixed ownership lane completed.
  Report: `agents/fuzz_high_cga_scaled_round26.md`. No new bucket. The
  Round 25 scaled harness limitation is resolved: high-CGA scaled-MMAv5
  controls pass for 4/8/16 CTA launches, and mixed local 1CTA/2CTA
  descriptor-view store/load/`ld.red`/copy rows fail cleanly under existing
  `FZ-20260421-0010`.

- 2026-04-21: Round 28 broad `ld.red` rerun completed. Report:
  `agents/fuzz_ldred_round28.md`. No new bucket. The broad selector and exact
  failing-node rerun reconfirmed the same six existing
  `FZ-20260421-0012` M64 row-permuted `unsupported dst layout` failures.

- 2026-04-21 12:50 UTC: Round 31 local guardrails completed. Reports:
  `agents/fuzz_local_descriptor_mix_round31.md`,
  `agents/fuzz_local_view_copy_round31.md`, and
  `agents/fuzz_structural_rerun_round31.md`. Descriptor-heavy `ld.red` plus
  scaled-MMAv5 rows passed split-4 as `54 passed`; high-rank `ld/st`
  descriptor plus `tcgen05.copy` `warpx2` indexed/subslice/slice-index rows
  completed as `123 passed, 74 skipped`; checked-in structural fuzzer stayed
  stable as `9 passed, 24 xfailed`. No new independent `FZ-*` bucket. Round
  31 subagent lanes remain active for scaled-MMAv5 operands, `ld.red`
  extremes, compiler-boundary fuzzing, and descriptor-chain shapes.

- 2026-04-21 12:51 UTC: Round 31 compiler boundary lane completed. Report:
  `agents/fuzz_compiler_boundaries_round31.md`. Ran `18` generated MLIR cases
  through verifier, optimize, and lower modes after `make -j8`. Results:
  `18` passes, `23` clean diagnostics, `12` assertion aborts, and `1` late
  illegal `ttg.memdesc_index` compiler failure. No new bucket; classifications
  sharpen `FZ-0001`, `FZ-0016`, and `FZ-0017`. Next compiler-only slice should
  extend the same matrix to reduction loads, scale TMEM layouts, and
  descriptor-view chains while keeping TMEM consumers live.

- 2026-04-21 13:00 UTC: Round 31b compiler boundary follow-up completed.
  Report: `agents/fuzz_compiler_boundaries_round31b.md`. Ran `8` generated
  reduction/load and scale-layout MLIR cases through verifier, optimize, and
  lower modes after `make -j8`. Results after manual classification: `6`
  passes, `15` clean diagnostics or known late illegal-op failures, and `3`
  assertion aborts. No new bucket; classifications sharpen `FZ-0001` and
  `FZ-0016`, and confirm clean `i64 ld.red` / scale-copy diagnostics.

- 2026-04-21 12:54 UTC: Round 31 ld.red descriptor/layout extremes completed.
  Report: `agents/fuzz_ldred_extremes_round31.md`. Temporary subprocess
  runtime probe covered `23` M/N extremes, row/col permutations, direct and
  indexed roots, descriptor-view chains, explicit variants, invalid 2CTA
  ownership, and resource boundaries. Result: `15` pass, `3` existing
  `FZ-0012`, `1` existing `FZ-0010`, `2` clean resource boundaries, and `2`
  new `FZ-20260421-0018` candidate rows. `FZ-0018` is direct `M128xN512`
  hardware `ld.red` reaching ptxas register allocation failure; `M64xN512`
  passes and `M128xN512` with `num_warps=8` cleanly reports shared-memory
  OOR. Checked-in ld.red selector stayed stable as `237 passed, 6 failed`,
  all known `FZ-0012`.

- 2026-04-21: Round 32 local ld/st narrow and half-row guardrail completed.
  Report: `agents/fuzz_local_ldst_narrow_round32.md`. Selector
  `(ldst and (n32 or half_rows or x1 or i32_broad) and not reports and not resource and not clean and not roundtrip)`
  collected `141/1615` and passed split-4 as `141 passed`; group 3 was slow
  (`104.45s`) but completed. No new independent `FZ-*` bucket.

- 2026-04-21: Round 32 local plain-MMAv5 accumulator/view guardrail completed.
  Report: `agents/fuzz_local_mma_accumulator_round32.md`. Selector
  `tmem_runtime_matrix_mma and not ldst and not mma_scaled and not reports and not resource and (m64 or twocta or indexed_acc_view or acc_subslice_view)`
  collected `185/1615` and passed split-4 as `185 passed`. No new independent
  `FZ-*` bucket.

- 2026-04-21: Round 32 local TMEM lit guardrail completed. Report:
  `agents/fuzz_local_lit_round32.md`. `tmem_layouts.mlir` and `invalid.mlir`
  passed; `relayout_tritongpu.mlir` failed with existing `FZ-20260421-0016`.
  No new independent `FZ-*` bucket.

- 2026-04-21: Round 32 local clean-boundary runtime guardrail completed.
  Report: `agents/fuzz_local_clean_boundary_round32.md`. Selector
  `(reports_clean or clean_unsupported or clean_error or tmem_oor or clean_xfail) and not resource`
  collected `184/1615` and passed split-4 as `184 passed`. No new independent
  `FZ-*` bucket.

- 2026-04-21: Round 32 checked-in structural fuzzer rerun completed. Report:
  `agents/fuzz_structural_rerun_round32.md`. Required `make -j8` was a no-op.
  Command:
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py`.
  Result: `9 passed, 24 xfailed in 9.42s`. No XPASS, unexpected failure,
  compiler crash, false unsupported diagnostic, opcode mismatch, runtime
  miscompile, or new independent `FZ-*`.

- 2026-04-21: Round 33 local scale-bearing runtime guardrail completed.
  Report: `agents/fuzz_local_scales_round33.md`. Selector
  `((cp_scales or ldst_scales or ld_red_scales) and not cp_no_scales and not reports and not clean and not resource)`
  collected `66/1615` and passed split-4 as `66 passed`. No new independent
  `FZ-*` bucket.

- 2026-04-21: Round 33 local scaled-MMAv5 layout/subslice guardrail completed.
  Report: `agents/fuzz_local_scaled_layouts_round33.md`. Selector
  `(mma_scaled and not reports and not resource and (layout_format or lhs_subslice or acc_subslice or tile_permuted or indexed_acc))`
  collected `208/1615` and passed split-4 as `208 passed`. No new independent
  `FZ-*` bucket.

- 2026-04-21: Round 33 local positive `ld.red` runtime guardrail completed.
  Report: `agents/fuzz_local_ldred_positive_round33.md`. Selector
  `(ld_red and not reports and not resource and not m64)` collected `204/1615`
  and passed split-4 as `204 passed`. No new independent `FZ-*` bucket.

- 2026-04-21: Round 32 FZ-0018 `ld.red` minimization completed. Report:
  `agents/fuzz_ldred_fz0018_min_round32.md`. Temporary subprocess harness ran
  `60` rows: `16` reproduced `FZ-20260421-0018`, `12` passed, `6` were clean
  shared-memory boundaries, `6` were clean tensor-memory boundaries, and `20`
  were clean power-of-two shape boundaries. `FZ-0018` remains an open backend
  resource-policy/planning bug; repair is intentionally deferred during
discovery.

- 2026-04-21 13:17 UTC: Round 34 mbarrier/proxy-fence composition lane
  completed. Report: `agents/fuzz_mbarrier_composition_round34.md`. Required
  `make -j8` was a no-op. Temporary probe
  `/tmp/tmem_mbarrier_composition_round34_probe.py` collected `4` rows and
  passed split-4 as harness tests (`1/1/1/1`), manually classified as `3`
  existing `FZ-20260421-0014` proxy-fence insertion reproductions and `1`
  green init-all contrast. Adjacent checked-in
  `cp_scales`/mbarrier/proxy/clean-boundary selector collected `55/1615` and
  passed split-4 as `55 passed`. No new independent `FZ-*`; no
  `FZ-20260421-0010` ownership diagnostic in the legal `num_ctas=2` temporary
  rows.

- 2026-04-21 13:20 UTC: Round 34 high-rank descriptor-view runtime lane
  completed. Artifact: `agents/fuzz_high_rank_views_round34.md`. Required
  `make -j8` was a no-op. Checked-in high-rank descriptor selector
  `(higher_rank or multidim_slice or half_rows or rank5) and not reports`
  collected `102/1615` and stayed stable as `82 passed, 20 skipped`; focused
  rank-5 small/unit-parent controls passed as `23 passed`. Temporary probe
  `/tmp/tmem_high_rank_views_round34_probe.py` classified `9` rows as `2`
  pass, `1` clean TMEM OOR, `1` existing `FZ-20260421-0002`/
  `FZ-20260421-0003` descriptor-view wrong result, `1` clean unsupported
  row-half `.ld.red` descriptor view, and `4` existing-candidate
  `FZ-20260421-0019` rank-5 `ld/st` unit-dimension `get_reg_layout` aborts.
  No additional independent `FZ-*` id was proposed; repair remains deferred
  under the discovery-only campaign contract.

- 2026-04-21: Round 30 descriptor-view chain-shape expansion completed and
  late-integrated. Report: `agents/fuzz_chain_shapes_round30.md`. Required
  `make -j8` was a no-op. Temporary subprocess probes covered `ld/st`,
  rank-4 unit-prefix roots, and actual `view.load_max(...)` `ld.red`
  consumers. Results: `ld/st` `132 pass, 48 clean unsupported, 42 clean OOR,
  36 optimizer exceptions, 18 process aborts`; unit-prefix rank-4 `8/8`
  process aborts; actual `ld.red` `32 pass, 12 clean unsupported, 4 parse
  failures`. Existing `FZ-20260421-0019` now also covers rank-4 unit-prefix
  dimension aborts; new candidates are `FZ-20260421-0020`,
  `FZ-20260421-0021`, and `FZ-20260421-0022`. No runtime wrong-result
  miscompile was confirmed; repair remains deferred during discovery.

- 2026-04-21: Round 35 copy/scales descriptor-view guardrail completed.
  Report: `agents/fuzz_copy_scales_descriptor_views_round35.md`. Required
  `make -j8` was a no-op. Runtime sweeps covered `53` checked-in row
  executions over `31` scale-copy descriptor-view/layout-probe rows plus `19`
  adjacent descriptor-view copy rows; all passed. No compiler crash, false
  unsupported diagnostic, opcode mismatch, runtime miscompile, clean-boundary
  drift, unexpected xfail/pass transition, or new independent `FZ-*`.

- 2026-04-21: Round 35 `ld.red`/descriptor positive runtime sweep completed.
  Reports: `agents/fuzz_ldred_ldst_positive_round35.md` and
  `agents/fuzz_audit_priorities_round35.md`. Required `make -j8` was a no-op.
  Selector
  `(ld_red or ldst_descriptor) and not reports and not resource and not clean`
  collected `380/1615` rows and completed split-4 as `313 passed`,
  `61 skipped`, `6 failed`. All six failures reproduced by exact nodeid in
  fresh pytest processes. Classification: existing `FZ-20260421-0012`
  expands to M64 f32 `tcgen05.ld.red` row-basis destination-layout planner
  failures for `row_reverse_n32` and `row_rotate_col_even_odd_n128`, in both
  default and explicit `32x32b` split-N paths. No runtime wrong-result
  miscompile or hard process abort in this lane.

- 2026-04-21: Round 35 two-CTA proxy/commit and subword guardrails completed.
  Report: `agents/fuzz_twocta_subword_guardrails_round35.md`. Scale/proxy
  selector collected `34/1615` and passed as `34 passed`; two-CTA
  commit/proxy-heavy selector collected `325/1615` and completed as
  `288 passed, 37 skipped`; subword and non-f32 `ld.red` selector collected
  `100/1615` and passed as `100 passed`. No compiler crash, false unsupported
  diagnostic, opcode mismatch, runtime miscompile, clean-boundary drift,
  unexpected xfail/pass transition, or new independent `FZ-*`.

- 2026-04-21: Round 35 allocator-pressure runtime guardrail completed.
  Report: `agents/fuzz_allocator_pressure_round35.md`. Selector
  `alloc_lifetime or mma_scaled or lhs_subslice` collected `303/1615` and
  passed split-4 as `303 passed` (`76/76/76/75`). No compiler crash, false
  unsupported diagnostic, opcode mismatch, runtime miscompile, clean-boundary
  drift, unexpected xfail/pass transition, or new independent `FZ-*`.

- 2026-04-21: Round 35 structural descriptor/`ld.red` fuzzer slice completed.
  Report: `agents/fuzz_structural_descriptor_ldred_round35.md`. Selector
  `descriptor_view or ldst_view or ldred or ld_red` collected `18/33` and ran
  as `7 passed, 11 xfailed`. No XPASS, unexpected failure, compiler crash,
  false unsupported diagnostic, opcode mismatch, runtime miscompile, or new
  independent `FZ-*`.

- 2026-04-21 14:35 UTC: Round 45 dynamic copy sentinel lane completed.
  Report: `agents/fuzz_dynamic_copy_sentinel_round45.md`. Required
  `make -j8` was a no-op. The minimized branch-selected linear-copy dynamic
  descriptor row (`f32`, `M=128`, `N=4`, lifted
  `_make_tmem_linear_layout(128, 4)`) reproduces existing
  `FZ-20260421-0001` as a late illegal `ttg.memdesc_index` feeding
  `ttng.tmem_copy`/`ttng.tmem_load`. Dynamic branch-selected
  `warpx2::{01_23,02_13}` copy descriptors, including a
  `slice(...).index(0)` chain, passed as positive controls. No new
  independent `FZ-*`.

- 2026-04-21 14:38 UTC: Round 45 runtime-matrix breadth lane completed.
  Report: `agents/fuzz_runtime_matrix_breadth_round45.md`. Required
  `make -j8` was a no-op. Positive descriptor/scaled/copy selector collected
  `452/1615` and completed split-4 as `428 passed, 24 skipped`; isolated
  diagnostic/resource boundary selector collected `103/1615` and passed as
  `103 passed`; compact `test_core.py` TMA/MMAv5/scaled-copy selector
  collected `20/18114` and passed as `20 passed`. No compiler crash, runtime
  wrong-result, false unsupported diagnostic, boundary drift, or new
  independent `FZ-*`.

- 2026-04-21 14:41 UTC: Round 45 high-CGA/M64 breadth probe completed.
  Report: `agents/fuzz_high_cga_m64_breadth_round45.md`. Required `make -j8`
  was a no-op. Mixed selector over `ld_red_m64`, `mma_m64`, two-CTA TMA
  transposed descriptors, high-CGA clean-boundary, and scale descriptor-view
  CGA rows produced `86 passed, 6 failed, 1523 deselected`. All six failures
  match existing `FZ-20260421-0012` M64 f32 `tcgen05.ld.red`
  destination-layout planner gaps; adjacent high-CGA/two-CTA positives stayed
  green. No new independent `FZ-*`.

- 2026-04-21 14:38 UTC: Round 46 subword/`.x1` runtime guardrail completed.
  Report: `agents/fuzz_subword_x1_round46.md`. Required `make -j8` was a
  no-op. Selector
  `(x1 or subword or i64 or f64 or non_f32 or descriptor_chain_roundtrip) and not reports and not resource and not clean`
  produced `301 passed, 70 skipped, 1244 deselected`. No compiler crash,
  false unsupported diagnostic, runtime miscompile, unexpected skip/pass
  drift, or new independent `FZ-*`.

- 2026-04-21: Round 45 descriptor-chain shape adversary completed. Report:
  `agents/fuzz_descriptor_chain_shape_adversary_round45.md`. Required
  `make -j8` was a no-op. Broad descriptor-chain runtime sweep collected
  `123/1615` and completed split-4 as `62 passed, 61 skipped`; M64 reduction
  and split-N shape sweep produced `35 passed, 6 failed` with all six failures
  matching existing `FZ-20260421-0012`; structural descriptor subset stayed
  `7 passed, 11 xfailed`; unit/high-rank parent-style chains passed as
  `26 passed`; descriptor-composition perturbations passed as `36 passed`.
  No new independent `FZ-*`.

- 2026-04-21 14:38 UTC: Round 46 multi-CTA/CGA boundary fuzzing completed.
  Report: `agents/fuzz_multicta_cga_boundaries_round46.md`. Required
  `make -j8` was a no-op. Runtime-matrix multi-CTA/two-CTA breadth selector
  collected `350/1615` and completed split-4 as `313 passed, 37 skipped`;
  high-CGA/clean-boundary selector collected `56/1615` and passed as
  `56 passed`; focused `test_core.py` TMA/multicast/MMAv5 controls collected
  `16/18114` and passed as `16 passed`, including
  `test_tma_multicast_copy[ctas_per_cga2]` with `[4, 4]`/16 CTAs per CGA.
  No compiler crash, false unsupported case, runtime miscompile, or new
  independent `FZ-*`.

- 2026-04-21: Round 46 scaled-MMAv5 descriptor/scale-operand composition
  completed. Report:
  `agents/fuzz_scaled_mmav5_descriptor_composition_round46.md`. Required
  `make -j8` was a no-op. Checked-in scaled-MMAv5 composition positives
  collected `210/1615` and passed as `210 passed`; checked-in clean-boundary
  rows collected `27/1615` and passed as `27 passed`; structural dynamic
  accumulator sentinel remained `1 xfailed`, and adjacent structural rows
  completed as `2 passed, 3 xfailed`. Temporary probes reproduced only
  existing `FZ-20260421-0007`, `FZ-20260421-0013`, and
  `FZ-20260421-0015`. No new independent `FZ-*`.

- 2026-04-21 14:41 UTC: Round 47 non-M64 `ld.red` positive guardrail
  completed. Report: `agents/fuzz_ldred_positive_round47.md`. Required
  `make -j8` was a no-op. Selector
  `(ld_red or load_red) and not m64 and not reports and not resource and not clean and not descriptor_chain`
  produced `156 passed, 1459 deselected`. No compiler crash, false
  unsupported diagnostic, runtime miscompile, opcode absence, or new
  independent `FZ-*`.

- 2026-04-21 14:41 UTC: Round 47 copy descriptor layout fuzzing completed.
  Report: `agents/fuzz_copy_descriptor_layout_variants_round47.md`.
  Required `make -j8` was a no-op. Runtime-matrix positive copy variants
  collected `246/1615` and completed split-4 as `242 passed, 4 skipped`;
  clean copy boundaries collected `67/1615` and passed as `67 passed`;
  structural copy scales passed as `2 passed`; older `test_core.py` copy
  controls collected `50/18114` and completed as `45 passed, 5 skipped`.
  Static copy descriptor views, `warpx2`, two-CTA copy, dense shared
  rematerialization, scales copy, and clean unsupported boundaries stayed
  stable. No new independent `FZ-*`.

- 2026-04-21 14:41 UTC: Round 47 lit/IR structural compiler-only fuzzing
  completed. Report: `agents/fuzz_lit_ir_structural_round47.md`. Required
  `make -j8` was a no-op and `ninja triton-opt` was up to date. Direct TMEM
  lit set passed `7/7`; broader NVIDIA GPU lit guardrail passed `16/16`;
  adjacent TMEM-bearing lit expansion passed `9/10`, with the sole failure
  `Conversion/relayout_tritongpu.mlir` matching existing `FZ-20260421-0016`
  unencoded/non-distributed TMEM operand verifier assertion. Structural fuzzer
  collected `33` and ran as `9 passed, 24 xfailed`. No new independent
  `FZ-*`.

- 2026-04-21: Round 47 proxy/mbarrier ordering fuzzing completed. Report:
  `agents/fuzz_proxy_mbarrier_ordering_round47.md`. Required `make -j8` was
  a no-op. Core proxy/TMA/MMAv5/mbarrier selector collected `232/18114` and
  completed split-4 as `184 passed, 48 skipped`; runtime-matrix
  proxy-adjacent selector collected `213/1615` and passed as `213 passed`;
  core scaled-copy linear-accumulator controls passed as `9 passed`; checked-in
  proxy-fence lit guardrail passed `1/1`. Disposable Round 14 and saved
  sequential-mbarrier reproducers reproduced only existing
  `FZ-20260421-0014`. No new independent `FZ-*`.

- 2026-04-21 14:44 UTC: Round 48 clean-boundary runtime guardrail completed.
  Report: `agents/fuzz_clean_boundary_round48.md`. Required `make -j8` was a
  no-op. Selector
  `(reports_clean or clean_unsupported or clean_error or tmem_oor or resource) and not m64`
  produced `176 passed, 1439 deselected`. No diagnostic drift, assertion
  crash, generic pass failure, runtime miscompile, or new independent `FZ-*`.

- 2026-04-21 14:45 UTC: Round 48 TMA/indexed/subslice positive mix
  completed. Report: `agents/fuzz_tma_indexed_subslice_round48.md`. Required
  `make -j8` was a no-op. Selector
  `(tma_b_transposed_descriptor or indexed_acc_view or acc_subslice_view_plain_kinds or indexed_acc_identity_narrow_view_format_use_acc or acc_subslice_view_format_use_acc) and not reports and not clean and not resource and not m64`
  produced `142 passed, 1473 deselected`. No compiler crash, false
  unsupported diagnostic, runtime miscompile, opcode drift, or new independent
  `FZ-*`.

- 2026-04-21: Round 48 allocator/lifetime/rematerialization pressure
  completed. Report: `agents/fuzz_allocator_lifetime_pressure_round48.md`.
  Required `make -j8` was a no-op. Runtime-matrix allocator/rematerialization
  pressure selector collected `363/1615` and passed as `363 passed`;
  diagnostic/resource boundary selector collected `184/1615` and passed as
  `184 passed`; focused `test_core.py` lifetime/multi-allocation controls
  collected `66/18114` and completed as `61 passed, 5 skipped`; structural
  allocator-crash sentinels remained `2 xfailed`. Existing `FZ-20260421-0005`,
  `FZ-20260421-0009`, and `FZ-20260421-0018` stayed stable. No new
  independent `FZ-*`.

- 2026-04-21: Round 48 high-rank and half-view TMEM fuzzing completed. Report:
  `agents/fuzz_high_rank_half_views_round48.md`. Required `make -j8` was a
  no-op. High-rank/half-view `ld/st` selector collected `112/1615` and
  completed as `92 passed, 20 skipped`; adjacent `ld.red`
  descriptor/M64/row-permutation selector collected `88/1615` and completed as
  `82 passed, 6 failed`, with all six failures matching existing
  `FZ-20260421-0012`; structural `ld/st`/`ld.red` subset collected `15/33`
  and ran as `7 passed, 8 xfailed`. Disposable probes reproduced existing
  `FZ-20260421-0021` and `FZ-20260421-0004`. No new independent `FZ-*`.

- 2026-04-21 14:45 UTC: Round 48 cache/process-order stability completed.
  Report: `agents/fuzz_cache_process_order_round48.md`. Required `make -j8`
  was a no-op. Mixed same-process positives plus xfail completed as
  `4 passed, 1 xfailed`; dynamic copy ordering kept linear failures as
  existing `FZ-20260421-0001` while warpx2 positives passed before, between,
  and after failures; warm-cache runtime positives repeated as `3 passed` then
  `3 passed`; M64 failure ordering kept row-basis failure as existing
  `FZ-20260421-0012` while adjacent positives passed; structural warm-cache
  repeats stayed `2 passed, 2 xfailed` twice and `2 passed, 1 xfailed` twice.
  No cache-key, process-contamination, global-state, XPASS, or new `FZ-*`
  issue was found.

- 2026-04-21 14:47 UTC: Round 49 scaled/MMAv5 and scales-copy positive union
  completed. Report: `agents/fuzz_scaled_copy_positive_round49.md`. Required
  `make -j8` was a no-op. Selector
  `((mma_scaled and not reports and not clean and not resource and not m64) or (cp_scales and not reports and not clean and not resource))`
  produced `276 passed, 1339 deselected`. No compiler crash, false
  unsupported diagnostic, runtime miscompile, opcode drift, or new independent
  `FZ-*`.

- 2026-04-21 14:49 UTC: Round 49 structural exact smoke completed. Report:
  `agents/fuzz_structural_exact_smoke_round49.md`. Required `make -j8` was a
  no-op. Exact structural smoke over two positives and two expected failures
  completed as `2 passed, 2 xfailed`; xfail signatures remained existing
  `FZ-20260421-0001` and `FZ-20260421-0004`. No XPASS drift, unexpected
  compiler crash, false unsupported diagnostic, positive-row failure, or new
  independent `FZ-*`.

- 2026-04-21 14:48 UTC: Round 49 checked-in test promotion candidates
  completed. Report: `agents/fuzz_promotion_candidates_round49.md`. Required
  `make -j8` was a no-op. Reconfirmed candidate evidence: dynamic linear
  `ttng.tmem_copy` expected `FZ-20260421-0001` as `1 failed, 1 passed`;
  sequential mbarrier expected `FZ-20260421-0014` via saved reproducer plus
  copy probe `2 failed, 6 passed`; relayout verifier expected
  `FZ-20260421-0016` lit `1 failed`; unit-rank half-column expected
  `FZ-20260421-0021` process abort; positives `3 passed`, proxy-fence lit
  `1 passed`, and rank-5 half-row worker passed. No new independent `FZ-*`.

- 2026-04-21: Round 49 TMEM numeric/layout shape extremes completed. Report:
  `agents/fuzz_shape_extremes_round49.md`. Required `make -j8` was a no-op.
  Broad shape-extreme runtime selector collected `316/1615` and completed as
  `286 passed, 24 skipped, 6 failed`, with all six failures matching existing
  `FZ-20260421-0012`; clean diagnostic/resource selector collected `83/1615`
  and passed as `83 passed`; structural subset collected `19/33` and ran as
  `7 passed, 12 xfailed`. Compiler-only 64-bit contrast reproduced existing
  `FZ-20260421-0017`; 64-bit copy still lowered successfully. No new
  independent `FZ-*`.

- 2026-04-21: Round 49 TMA descriptor/TMEM consumer interactions completed.
  Report: `agents/fuzz_tma_descriptor_interactions_round49.md`. Required
  `make -j8` was a no-op. Core TMA/shared-input selector collected
  `232/18114` and completed split-4 as `184 passed, 48 skipped`;
  runtime-matrix TMA descriptor interactions collected `17/1615` and passed as
  `17 passed`; descriptor-consumer expansion collected `61/1615` and passed as
  `61 passed`; basic TMA descriptor/mbarrier and scaled-copy controls passed
  as `8 passed`. No new evidence for `FZ-20260421-0010`,
  `FZ-20260421-0014`, or descriptor-consumer buckets; no new independent
  `FZ-*`.

- 2026-04-21 14:51 UTC: Round 50 mixed exact runtime smoke completed. Report:
  `agents/fuzz_mixed_exact_runtime_round50.md`. Required `make -j8` was a
  no-op. After correcting a stale two-CTA TMA parametrized nodeid, exact smoke
  over static `warpx2` copy, scaled B-scale descriptor-view rematerialization,
  rank-5 unit-parent `ld/st`, and two-CTA TMA B-transposed TF32 descriptor
  path passed as `4 passed`. No compiler crash, runtime miscompile, nodeid
  drift after correction, or new independent `FZ-*`.

- 2026-04-21 14:51 UTC: Round 50 lit conversion edge fuzzing completed.
  Report: `agents/fuzz_lit_conversion_edges_round50.md`. Required
  `make -j8` and `ninja triton-opt` were no-ops. Focused TMEM
  conversion/lowering lit guardrail passed `11/11`; broader Conversion
  lowering guardrail passed `8/8`; known-red `relayout_tritongpu.mlir`
  remained existing `FZ-20260421-0016`; saved reproducers revalidated existing
  `FZ-20260421-0014`, `FZ-20260421-0016`, and `FZ-20260421-0017`. No new
  verifier gap, clean-negative gap, FileCheck drift, or independent `FZ-*`.

- 2026-04-21 14:53 UTC: Round 50 `ld/st` descriptor positive sweep completed.
  Report: `agents/fuzz_ldst_descriptor_positive_round50.md`. Required
  `make -j8` was a no-op. Selector
  `(ldst_descriptor or ldst_twocta or ldst_scales) and not reports and not clean and not resource and not m64 and not non_f32`
  produced `166 passed, 98 skipped, 1351 deselected`; skips matched existing
  checked-in allocation/resource skips. No compiler crash, false unsupported
  diagnostic, runtime miscompile, skip drift beyond known skips, or new
  independent `FZ-*`.

- 2026-04-21: Round 50 grouped runtime breadth sweep completed. Report:
  `agents/fuzz_runtime_grouped_breadth_round50.md`. Required `make -j8` was a
  no-op. Positive breadth selector collected `175/1615` and passed as
  `175 passed`; clean-boundary selector collected `149/1615` and passed as
  `149 passed`; known-red selector collected `25/1615` and completed as
  `19 passed, 6 failed`, with all six failures matching existing
  `FZ-20260421-0012`. No compiler crash, verifier drift, wrong result,
  diagnostic drift, or new independent `FZ-*`.

- 2026-04-21 14:55 UTC: Round 50 dynamic copy exact probe completed. Report:
  `agents/fuzz_dynamic_copy_exact_round50.md`. Required `make -j8` was a
  no-op. Exact `/tmp` dynamic copy probe completed as `1 failed, 2 passed`;
  the failed row is existing `FZ-20260421-0001` minimized branch-selected
  linear `ttng.tmem_copy` with live illegal `ttg.memdesc_index`, while
  branch-selected `warpx2` positives passed before and after it in the same
  pytest process. No new independent `FZ-*`.

- 2026-04-21: Round 50 dynamic descriptor/control-flow adversary completed.
  Report: `agents/fuzz_dynamic_descriptor_controlflow_round50.md`. Required
  `make -j8` was a no-op. Broad dynamic descriptor rerun classified `46`
  passes, `20` existing `FZ-20260421-0001`, `24` existing
  `FZ-20260421-0002`/`FZ-20260421-0003`, and `10` clean unsupported copy
  boundaries. New disposable distinct-object probe passed `24/24` across
  branch/helper/loop selection between separate concrete TMEM objects for
  load, hardware `ld.red`, and `ttng.tmem_copy`. Linear dynamic-copy sentinel
  stayed `2 failed` as existing `FZ-20260421-0001`; dynamic `warpx2` copy
  controls passed `4 passed`; structural dynamic rows stayed `3 xfailed`;
  MMAv5 controls passed `2 passed, 1 xfailed` with the xfail existing
  `FZ-20260421-0007`. No new independent `FZ-*`.

- 2026-04-21 14:56 UTC: Round 51 local lit smoke completed. Report:
  `agents/fuzz_lit_smoke_round51.md`. Required `make -j8` and
  `ninja triton-opt` were no-ops. Lit smoke over `tmem_layouts.mlir`,
  `memdesc-subview-split.mlir`, and `proxy_fence_insertion.mlir` passed
  `3/3`. No FileCheck drift, verifier drift, proxy-fence lit drift, or new
  independent `FZ-*`.

- 2026-04-21 14:57 UTC: Round 51 local core control smoke completed. Report:
  `agents/fuzz_core_control_smoke_round51.md`. Required `make -j8` was a
  no-op. `test_core.py` selector over TMEM copy, descriptor-chain, M64 split-N
  default-load, and two-CTA MMAv5 multicast commit linear-accumulator controls
  passed as `31 passed, 18083 deselected`. No compiler crash, runtime
  miscompare, skip drift, or new independent `FZ-*`.

- 2026-04-21 14:57 UTC: Round 51 reproducer minimization completed. Report:
  `agents/fuzz_reproducer_minimization_round51.md`. Required `make -j8` was a
  no-op. `FZ-20260421-0014` was minimized to a 27-line MLIR repro with no
  function arg or trailing store; `FZ-20260421-0016` was minimized to an
  8-line compiler-only MLIR repro; `FZ-20260421-0017` was minimized to a
  10-line compiler-only MLIR repro. `FZ-20260421-0001` dynamic copy and
  `FZ-20260421-0021` half-column signatures were revalidated through existing
  Python `/tmp` harnesses. No new independent `FZ-*`.

- 2026-04-21 14:59 UTC: Round 51 local minimized repro confirmation
  completed. Report: `agents/fuzz_minimized_repro_local_round51.md`. Direct
  reruns of `/tmp/tmem_round51_fz0014_no_store.mlir`,
  `/tmp/tmem_round51_fz0016_min.mlir`, and
  `/tmp/tmem_round51_fz0017_min.mlir` reproduced existing `FZ-20260421-0014`
  with exit code `1` and `FZ-20260421-0016`/`FZ-20260421-0017` with exit code
  `134` assertion aborts. No new independent `FZ-*`.

- 2026-04-21: Round 51 example/runtime sanity completed. Report:
  `agents/fuzz_examples_runtime_sanity_round51.md`. Required `make -j8` was a
  no-op. Compact TMEM examples `05-tmem-moe-router.py`,
  `06-tmem-lora-fusion.py`, and `08-tmem-layout-as-epilogue.py` passed as
  `37 passed`; fused-gather sampled runtime passed as `3 passed` after adding
  `./python/triton_kernels` to `PYTHONPATH`; older TMEM-bearing attention,
  convolution, multi-CTA matmul, and 2CTA scaled-matmul example nodeids passed
  as `8 passed`. Lightweight benchmark entrypoints for router/candidate/ragged,
  LoRA/side projection, and layout-as-epilogue executed. No backend `FZ-*`;
  only harness issue is bare collection of `05-moe-bmm1-fused-gather.py`
  needing `./python/triton_kernels` on `PYTHONPATH`.

- 2026-04-21 15:00 UTC: Round 51 local two-CTA exact probe completed. Report:
  `agents/fuzz_twocta_exact_round51.md`. Required `make -j8` was a no-op.
  Exact probe over clean two-CTA-in-four-CTA diagnostic, two-CTA `warpx2`
  dense shared rematerialization, two-CTA indexed-accumulator MMAv5, and
  two-CTA scales copy passed as `4 passed`. No clean-diagnostic drift,
  compiler crash, runtime miscompile, or new independent `FZ-*`.

- 2026-04-21 15:01 UTC: Round 51 local proxy pair check completed. Report:
  `agents/fuzz_proxy_pair_round51.md`. `ninja triton-opt` was a no-op.
  Checked-in `proxy_fence_insertion.mlir` passed `1/1`; minimized
  `FZ-20260421-0014` repro exited `1` with the expected proxy-fence insertion
  diagnostic. No checked-in proxy-fence lit drift, `FZ-0014` signature drift,
  or new independent `FZ-*`.

- 2026-04-21: Round 51 core TMEM breadth fuzzing completed. Report:
  `agents/fuzz_core_tmem_breadth_round51.md`. Required `make -j8` was a
  no-op. Explicit `test_core.py` TMEM-bearing node list collected `641` tests
  and completed split-4 as `585 passed, 56 skipped, 0 failed`. Coverage
  included allocation, linear load/store, descriptor chains, copy, M64,
  `ld.red`, TMA, TMA+MMA shared-input paths, proxy/mbarrier controls, plain
  MMAv5, scaled MMAv5, scale copy, multicast/two-CTA controls, and clean-error
  diagnostics. No compiler crash, verifier drift, unsupported-case regression,
  clean-negative drift, runtime miscompile, or new independent `FZ-*`.

- 2026-04-21 15:03 UTC: Round 52 local known-red/green runtime mix completed.
  Report: `agents/fuzz_known_red_green_mix_round52.md`. Required `make -j8`
  was a no-op. Same-process exact mix produced `1 failed, 2 passed`; the
  failed row is existing `FZ-20260421-0012` M64 f32 `ld.red`
  destination-layout planner gap, and adjacent split-N/warpx2 positives passed
  afterward. No new independent `FZ-*` and no process-contamination signal.

- 2026-04-21 15:04 UTC: Round 52 local MoE example harness path check
  completed. Report: `agents/fuzz_moe_harness_path_round52.md`. Bare
  collection of `python/examples/gluon/05-moe-bmm1-fused-gather.py` with
  `PYTHONPATH=.:./python` failed with exit code `2` and
  `ModuleNotFoundError: No module named 'triton_kernels.distributed'`;
  adding `./python/triton_kernels` collected `48` tests with exit code `0`.
  Classification: harness/import-path issue, not a TMEM backend `FZ-*`.

- 2026-04-21: Round 52 example harness path sharpening completed. Report:
  `agents/fuzz_example_harness_paths_round52.md`. Required `make -j8` was a
  no-op. Per-example collection showed only
  `python/examples/gluon/05-moe-bmm1-fused-gather.py` needs
  `./python/triton_kernels` on `PYTHONPATH`; the other seven Gluon examples
  collect with `PYTHONPATH=.:./python`. Corrected aggregate collection with
  `PYTHONPATH=.:./python:./python/triton_kernels` collected `1043` tests, and
  corrected smoke passed as `6 passed`. Classification:
  harness/documentation invocation issue, not backend; no new `FZ-*`.

- 2026-04-21: Round 52 duration-aware runtime grouping completed. Report:
  `agents/fuzz_duration_grouped_runtime_round52.md`. Required `make -j8` was
  a no-op. Positive runtime matrix selector passed as `195 passed`; clean
  boundary selector passed as `245 passed`; known-red selector produced
  `6 failed, 3 passed`, with all failures classified as existing
  `FZ-20260421-0012`. Duration files `/tmp/tmem_round52_positive_durations.json`
  and `/tmp/tmem_round52_clean_durations.json` were created for future local
  throughput. Clean-boundary least-duration splitting reduced the worst group
  from `35.613s` wall to `12.071s` wall. No compiler crash, false unsupported
  diagnostic, verifier drift, runtime miscompile, hang, or new independent
  `FZ-*`.

- 2026-04-21: Round 53 local structural guardrail completed. Report:
  `agents/fuzz_round53_local_structural_guardrail.md`. Required `make -j8`
  was a no-op. The checked-in structural TMEM fuzzer passed as
  `9 passed, 24 xfailed`, with strict xfails remaining in existing
  `FZ-20260421-0001` through `FZ-20260421-0009` buckets and existing `R5-C`.
  No XPASS, signature drift, runtime miscompile, process contamination, or new
  independent `FZ-*`.

- 2026-04-21: Round 53 higher-rank descriptor runtime slice completed.
  Report: `agents/fuzz_round53_higher_rank_descriptor_runtime.md`. Required
  `make -j8` was a no-op. Selector over rank-5 descriptors, higher-rank
  slices, multidimensional slices, half-row views, direct higher-rank replay,
  `load_red` replay, and 2CTA variants collected `112` rows and completed
  split-4 as `92 passed, 20 skipped, 0 failed`. No compiler crash,
  unexpected unsupported diagnostic, verifier drift, runtime miscompile, hang,
  or new independent `FZ-*`.

- 2026-04-21: Round 53 compiler/lit lane completed. Report:
  `agents/fuzz_round53_compiler_lit_lane.md`. Required `make -j8` and
  `ninja triton-opt` were no-ops. Focused TMEM lit sweep completed as
  `13 passed, 1 failed`, with the sole failure classified as existing
  `FZ-20260421-0016`. Proxy/mbarrier/invalid diagnostics passed as
  `9 passed`. Direct Blackwell conversion exited `0`. Minimized repros for
  `FZ-20260421-0016` and `FZ-20260421-0017` were revalidated; f64 hit the
  same `FZ-0017` bitwidth assertion, and f32 control stayed on the existing
  indexed TMEM view-chain diagnostic surface. No new independent `FZ-*`.

- 2026-04-21: Round 53 generator gap lane completed. Report:
  `agents/fuzz_round53_generator_gap_lane.md`. Required `make -j8` was a
  no-op, and the temporary Round 25 dynamic generator plus Round 43
  plain/scaled sequence probe py-compiled. Dynamic descriptor/control-flow
  generator ran `22` rows; plain/scaled MMAv5 sequence probe ran `5` rows;
  high-CGA dtype/layout pytest slice passed as `17 passed`; scaled
  descriptor-view slice passed as `4 passed`. Aggregate classification:
  `48` rows total, with `30` pass/green, `8` existing `FZ-20260421-0001`,
  `7` existing `FZ-20260421-0002`, `3` clean boundaries, and `0` new
  `FZ-*` candidates. No checked-in tests or backend code changed.

- 2026-04-21: Round 53 structural/runtime lane completed. Report:
  `agents/fuzz_round53_structural_runtime_lane.md`. Required `make -j8` was a
  no-op. Structural fuzzer split-4 aggregate stayed at
  `9 passed, 24 xfailed`; higher-rank descriptor runtime selector collected
  `79` rows and completed as `59 passed, 20 skipped`; scaled/narrow
  accumulator descriptor selector collected `29` rows and passed as
  `29 passed`; disposable structural permutation probe ran five additional
  row/column permutation rows and passed as `5 passed`. No XPASS, compiler
  crash, verifier drift, false unsupported diagnostic, runtime miscompile, or
  new independent `FZ-*`.

- 2026-04-21: Round 54 local TMA/TMEM runtime lane completed. Report:
  `agents/fuzz_round54_local_tma_tmem_lane.md`. Required `make -j8` was a
  no-op. Two-CTA TMA/TF32 MMAv5 selector collected `15` rows and passed as
  `15 passed`, covering shared-transpose clean diagnostics and B-transposed
  descriptor positives with and without `use_acc`. No compiler crash,
  verifier drift, clean-boundary drift, runtime miscompile, or new `FZ-*`.

- 2026-04-21: Round 54 local split-N/x1/subword lane completed. Report:
  `agents/fuzz_round54_local_splitn_x1_subword_lane.md`. Required `make -j8`
  was a no-op. Broad selector collected `101` rows; filtered positive lane
  excluded known-red M64 row/column `ld.red` rows and completed split-4 as
  `98 passed, 0 failed`. Coverage included split-N immediates, auto
  `16x32bx2`, x1 `f32`/`i32`, f16/i8 subword, 1CTA/2CTA descriptor chains,
  and clean unsupported x1 variants. No new independent `FZ-*`.

- 2026-04-21: Round 54 local `test_core.py` TMEM smoke completed. Report:
  `agents/fuzz_round54_local_test_core_smoke.md`. Required `make -j8` was a
  no-op. A collection probe over core TMEM families found `64` relevant rows;
  exact smoke over multicast commit, two-CTA linear accumulator MMAv5,
  no-scale copy, descriptor-view runtime, `ld.red`, and scaled copy passed as
  `6 passed`. No compiler crash, verifier drift, opcode absence, runtime
  miscompile, or new independent `FZ-*`.

- 2026-04-21: Round 54 opcode consistency lane completed. Report:
  `agents/fuzz_round54_opcode_consistency_lane.md`. Required `make -j8` was a
  no-op. Focused runtime-matrix opcode selector collected `172` rows and
  passed split-4 as `172 passed`, covering allocation/wait, `ld/st`, hardware
  and software `ld.red`, copy, scaled-copy, plain MMAv5, scaled MMAv5, and
  commit. Focused proxy/mbarrier/multicast selector over runtime matrix plus
  `test_core.py` collected `39/19729` and passed. Representative forced-dump
  replay passed `9` nodeids and captured `27` TTGIR/LLIR/PTX artifacts under
  `/tmp/tmem_round54_opcode_dump`; inspected streams matched for expected
  opcodes including hardware `ld.red`, software non-f32 reductions, warpx2,
  two-CTA copy, plain/scaled MMAv5, waits, commit, proxy fence, and cluster
  barriers. No PTX-vs-LLIR mismatch, missing hardware opcode, unexpected
  software fallback, verifier over-strictness, crash, miscompile, or new
  independent `FZ-*`.

- 2026-04-21 15:21 UTC: Round 54 dynamic 2CTA descriptor SSA lane completed.
  Report: `agents/fuzz_round54_dynamic_2cta_lane.md`. Required `make -j8`
  was a no-op. Temporary `/tmp/tmem_round54_dynamic_2cta_probe.py` crossed
  branch-selected, mixed-capture, and loop-carried descriptor SSA with legal
  lifted two-CTA `ld/st` layouts. Dynamic rows produced `3 passed, 3 failed`;
  three static controls for the failing shapes also failed. Aggregate:
  `3` pass and `6` existing `FZ-20260421-0003` descriptor-view chain wrong
  results, with `0` new independent `FZ-*`. No compiler crash, verifier
  strictness issue, false unsupported diagnostic, opcode mismatch, or process
  contamination was observed.

- 2026-04-21: Round 54 local scaled-MMAv5 LHS/tile/narrow lane completed.
  Report: `agents/fuzz_round54_local_scaled_lhs_tile_lane.md`. Selector over
  LHS subslice, LHS tile-permuted, accumulator tile-permuted, and narrow
  accumulator identity/tile rows collected `120/1615` and passed split-4 as
  `120 passed`. Coverage included legacy and linear scale operands,
  `mxfp8`, `mxfp4`, mixed `mxfp8/mxfp4`, `nvfp4`, `use_acc`, mixed-fp4A
  clean unsupported diagnostics, accumulator tile-permuted 64-wide rows, and
  narrow accumulator rows. No compiler crash, verifier drift, false
  unsupported diagnostic, clean-boundary drift, runtime miscompile, hang, or
  new independent `FZ-*`.

- 2026-04-21: Round 55 local clean-boundary runtime lane completed. Report:
  `agents/fuzz_round55_local_clean_boundary_lane.md`. Required `make -j8` was
  a no-op. Selector over block descriptors/layouts, x1 unsupported variants,
  no-scale copy clean boundaries, warpx2/4x256b diagnostics, row/column copy
  unsupported cases, and scaled LHS clean unsupported rows collected
  `96/1615` and passed split-4 as `96 passed`. No compiler crash, verifier
  drift, diagnostic regression, false unsupported diagnostic, runtime
  miscompile, hang, or new independent `FZ-*`.

- 2026-04-21: Round 55 local copy/MMA positive runtime lane completed. Report:
  `agents/fuzz_round55_local_copy_mma_positive_lane.md`. Selector over
  positive no-scale copy, scaled-copy, plain MMAv5, and scaled-MMAv5 root rows
  collected `88/1615` and passed split-4 as `88 passed`. Coverage included
  `warpx2` positives, dense-shared rematerialization, two-CTA copy codegen,
  tile-permuted copy atoms, scales `warpx4`, plain MMAv5 root kinds, and
  scaled-MMAv5 root formats. No compiler crash, verifier drift, opcode
  absence, runtime miscompile, hang, or new independent `FZ-*`.

- 2026-04-21: Round 55 lit negative-boundary lane C completed. Report:
  `agents/fuzz_round55_lit_negative_boundary_lane.md`. Required `make -j8`
  and `ninja triton-opt` were no-ops. Checked-in TMEM/proxy/mbarrier/relayout/
  allocation/conversion lit sweep completed as `15 passed, 1 failed`, with the
  failure classified as existing `FZ-20260421-0016`; extra proxy/mbarrier/TMA/
  invalid lit sweep passed `7` tests plus one no-tests path warning. Direct
  Blackwell conversion exited `0`. Round 38 compiler-boundary corpus replay
  stayed at `18 pass`, `24` clean diagnostics, `5` existing late illegal-op
  rows, and `9` existing abort/crash rows. Round 24 parse-only `FZ-0016`
  corpus replay stayed at `15` existing abort/crash rows and `40` clean
  diagnostics. Minimized repros revalidated existing `FZ-20260421-0016` and
  `FZ-20260421-0017`. No new independent `FZ-*`.

- 2026-04-21: Round 56 scaled dynamic lane B completed. Report:
  `agents/fuzz_round56_scaled_dynamic_lane.md`. Required `make -j8` was a
  no-op. Checked-in scaled interaction selector collected `55/1615` and
  passed split-4 as `55 passed`. Temporary probes revalidated existing
  `FZ-20260421-0007` (`1 xfailed` accumulator subslice dynamic `use_acc`),
  `FZ-20260421-0013` (`6` dynamic scale-view plus `3` multi-MMA scale-view
  wrong-result rows), `FZ-20260421-0015` (`2` selected direct B-scale
  wrong-result rows), and `FZ-20260421-0010` (`24` high-CGA mixed local TMEM
  CTA-count diagnostics). Adjacent direct scale, accumulator subslice/index,
  legacy/linear, 1CTA/2CTA, reused-scale, and high-CGA scaled controls passed.
  No new independent `FZ-*`; no backend repairs attempted.

- 2026-04-21: Round 56 local `ld.red` mixed hardware/software lane completed.
  Report: `agents/fuzz_round56_local_ldred_mixed_lane.md`. Required
  `make -j8` was a no-op. Selector over compatible f32 hardware reductions,
  descriptor-chain N-sweep variants, non-identity layouts that canonicalize to
  `32x32b`, explicit N-sharded software-reduce layouts, and non-f32
  software-reduce contracts collected `60/1615` and passed split-4 as
  `60 passed`. No compiler crash, verifier drift, false unsupported
  diagnostic, opcode absence, hardware/software classification drift, runtime
  miscompile, hang, or new independent `FZ-*`.

- 2026-04-21: Round 56 local descriptor composition positive lane completed.
  Report: `agents/fuzz_round56_local_descriptor_positive_lane.md`. Required
  `make -j8` was a no-op. Selector over positive `ld/st` descriptor
  composition, roundtrip, deep roundtrip, multidimensional positive slice,
  direct half-row, and rank-5 rows collected `19/1615`; split-4 runtime
  executed as `5 passed, 14 skipped` under local capability guards. No
  selected executed row failed; no compiler crash, verifier drift, false
  unsupported diagnostic, runtime miscompile, hang, or new independent `FZ-*`.

- 2026-04-21: Round 56 `test_core.py` TMEM breadth lane C completed. Report:
  `agents/fuzz_round56_test_core_breadth_lane.md`. Required `make -j8` was a
  no-op. Main TMEM breadth selector collected `305/18114` and passed split-4
  as `297 passed, 8 skipped`; high-CGA scaled-copy selector collected
  `48/18114` and passed as `48 passed`; TMA-to-TMEM shared-input selector
  collected `63/18114` and passed as `57 passed, 6 skipped`. No compiler
  crash, verifier drift, false unsupported diagnostic, opcode absence, runtime
  miscompile, hang, unexpected skip/fail transition, or new independent
  `FZ-*`.

- 2026-04-21: Round 56 copy dynamic descriptor lane A completed. Report:
  `agents/fuzz_round56_copy_dynamic_descriptor_lane.md`. Required `make -j8`
  was a no-op. Temporary copy probe ran `23` generated rows over direct index,
  runtime index, branch-selected concrete TMEM views, TMEM column subslices,
  shared-source subslices, `128x128b`, `128x256b`, `warpx2::01_23`,
  `warpx2::02_13`, 1CTA, and 2CTA: `15` green, `7` existing
  `FZ-20260421-0001`, `1` clean unsupported, and `0` new independent `FZ-*`.
  Checked-in copy selector collected `127/1615` and passed as `127 passed`.
  No backend repairs were attempted.

- 2026-04-21: Round 57 local structural-fuzzer/lit bridge lane completed.
  Report: `agents/fuzz_round57_local_lit_runtime_bridge_lane.md`. Required
  `make -j8` was a no-op. Checked-in structural-fuzzer selector collected
  `15/33` and split-4 ran as `15 xfailed`, revalidating existing
  `FZ-20260421-0001`, `FZ-20260421-0002`, `FZ-20260421-0007`,
  `FZ-20260421-0008`, and `FZ-20260421-0009`. Lit bridge after
  `ninja triton-opt` passed `invalid.mlir`, `ops.mlir`, and
  `tmem_layouts.mlir` as `3/3`. No unexpected xfail transition, lit
  verifier/conversion drift, or new independent `FZ-*`.

- 2026-04-21: Round 57 copy view boundary lane D completed. Report:
  `agents/fuzz_round57_copy_view_boundaries_lane.md`. Required `make -j8`
  was a no-op. Primary no-scale copy non-overlap selector collected `7/1615`
  and passed split-4 as `7 passed`; broad no-scale copy guardrail collected
  `121/1615` and passed as `121 passed`; structural control-flow contrast
  collected `10/33` and ran as `2 passed, 8 xfailed` under existing
  `FZ-20260421-0001` and `FZ-20260421-0002`. No new independent `FZ-*`.

- 2026-04-21: Round 57 local direct higher-rank lane completed. Report:
  `agents/fuzz_round57_local_direct_higher_rank_lane.md`. Required
  `make -j8` was a no-op. Selector `ldst_direct_higher_rank and not reports`
  collected `4/1615` and passed split-4 as `4 passed`. Direct higher-rank
  `get_reg_layout`, load/store replay, and `load_red` replay positives stayed
  green with no compiler crash, verifier drift, false unsupported diagnostic,
  runtime miscompile, hang, or new independent `FZ-*`.

- 2026-04-21: Round 57 `ld.red` boundary/opcode lane E completed. Report:
  `agents/fuzz_round57_ldred_boundary_opcode_lane.md`. Required `make -j8`
  was a no-op. M64/resource-ish selector collected `39/1615` and ran as
  `33 passed, 6 failed`, all existing `FZ-20260421-0012`; non-M64
  non-identity hardware `ld.red` selector collected `66/1615` and passed as
  `66 passed`; software-vs-hardware classification selector collected
  `48/1615` and passed as `48 passed`; half-view replay ran `32` rows as
  `8` pass, `12` clean diagnostics, `8` existing `FZ-20260421-0022`, and
  `4` existing `FZ-20260421-0020`. No compiler crash, verifier drift, false
  unsupported diagnostic, opcode-selection regression, software/hardware
  classification drift, runtime miscompile, or new independent `FZ-*`.

- 2026-04-21: Round 58 local x1/subword lane completed. Report:
  `agents/fuzz_round58_local_x1_subword_lane.md`. Required `make -j8` was a
  no-op. Selector `ldst_x1 or subword_descriptor_chain or ldst_subword`
  collected `48/1615` and passed split-4 as `48 passed`. Coverage included
  subword pack/unpack, subword descriptor-chain roundtrips, x1 subword
  roundtrips, two-CTA x1 subword and descriptor-chain roundtrips, x1
  `f32`/`i32` roundtrips and descriptor-chain roundtrips, and clean
  unsupported `16x128b` x1 variants. No compiler crash, verifier drift, false
  unsupported diagnostic, clean-boundary drift, runtime miscompile, hang, or
  new independent `FZ-*`.

- 2026-04-21: Round 58 plain MMAv5 accumulator-view lane C completed. Report:
  `agents/fuzz_round58_plain_mma_acc_views_lane.md`. Required `make -j8` was
  a no-op. Focused runtime-matrix selector over plain MMAv5 indexed
  accumulator views, accumulator subslices, tile-permuted accumulator layouts,
  `use_acc`, and 2CTA variants collected `142/1615` and passed split-4 as
  `142 passed`. Exact `test_core.py` plain-MMAv5 controls collected `5/18114`
  and passed. No compiler crash, verifier drift, false unsupported diagnostic,
  opcode/commit mismatch, runtime miscompile, hang, unexpected skip/fail
  transition, or new independent `FZ-*`; existing `FZ-20260421-0003` and
  `FZ-20260421-0011` were not reproduced by this lane.

- 2026-04-21: Round 58 local scales load/store and copy lane completed.
  Report: `agents/fuzz_round58_local_scales_copy_lane.md`. Required
  `make -j8` was a no-op. Selector
  `cp_scales or copy_scales or ldst_scales_direct_roundtrip or ldst_scales_descriptor_view_roundtrip`
  collected `41/1615` and passed split-4 as `41 passed`. Coverage included
  scale `ld/st`, scale descriptor-view roundtrips, scale-copy layout
  probes/rematerialization, TMEM descriptor-view clean unsupported
  diagnostics, direct and two-CTA `warpx4` scale copy, and scaled-MMAv5 helper
  copy paths. No compiler crash, verifier drift, false unsupported
  diagnostic, clean-boundary drift, opcode absence, runtime miscompile, hang,
  or new independent `FZ-*`.

- 2026-04-21: Round 58 high-CGA ownership lane B completed. Report:
  `agents/fuzz_round58_high_cga_ownership_lane.md`. Required `make -j8` was a
  no-op. Checked-in high-CGA selector over runtime-matrix and `test_core.py`
  rows collected `93/19729` and ran as `87 passed, 6 skipped`; lit
  `membar-cluster.mlir` passed `1/1`. Temporary ownership probe ran `12`
  rows: `6` direct local-TMEM-in-larger-CGA rows and `6` parser-wrapped
  scales-copy rows all classified as existing `FZ-20260421-0010`. No missed
  `getModuleTwoCTAs` propagation bug, compiler crash, verifier drift, false
  unsupported diagnostic outside the known CTA-count gate, runtime miscompile,
  hang, proxy/TMA/mbarrier ownership regression, or new independent `FZ-*`.

- 2026-04-21: Round 59 local codegen/opcode lane completed. Report:
  `agents/fuzz_round59_local_codegen_opcode_lane.md`. Required `make -j8` was
  a no-op. Selector `(cp_no_scales and codegen) or mma_scaled_minimal`
  collected `27/1615` and passed split-4 as `27 passed`. Coverage included
  no-scale `4x256b` refresh codegen, two-CTA no-scale copy codegen across
  linear/legacy and multiple dtypes, two-CTA `128x128b` no-scale copy codegen,
  and minimal scaled-MMAv5. No compiler crash, verifier drift, opcode/codegen
  assertion drift, false unsupported diagnostic, runtime miscompile, hang, or
  new independent `FZ-*`.

- 2026-04-21: Round 60 local frontend tensor-memory lane completed. Report:
  `agents/fuzz_round60_local_frontend_tmem_lane.md`. Required `make -j8` was
  a no-op. Focused frontend selector `tensor_memory or tmem_` collected
  `29/225` and ran as `28 passed, 1 failed`; exact rerun of
  `test_tmem_subslice_reg_layout_constexpr` reproduced the same inline
  expecttest mismatch. Candidate `FZ-20260421-0023`: frontend TMEM subslice
  register-layout expectation drift. No backend repair or expectation update
  attempted.

- 2026-04-21: Round 60 examples breadth lane A completed. Report:
  `agents/fuzz_round60_examples_breadth_lane.md`. Required `make -j8` was a
  no-op. Broad collection over five `python/examples/gluon` files discovered
  `974` tests; focused selector collected `12` and ran split-4 as
  `11 passed, 1 skipped`. The skip is the existing fp4 packed descriptor
  `K` multiple-of-128 example guard. No new independent `FZ-*`.

- 2026-04-21: Round 60 random runtime-matrix lane C completed. Report:
  `agents/fuzz_round60_random_runtime_matrix_lane.md`. Required `make -j8`
  was a no-op. Seed `6042160` selected `64` exact nodeids from `1615`
  collected rows across `ld/st`, copy, `ld.red`, plain MMAv5, and scaled
  MMAv5. Split-4 runtime result: `56 passed, 8 skipped`; skips are known
  lifted descriptor roundtrip Blackwell allocation-limit guards. No new
  independent `FZ-*`.

- 2026-04-21: Round 61 local examples edge-parameter lane completed. Report:
  `agents/fuzz_round61_local_examples_edge_lane.md`. Required `make -j8` was
  a no-op. Six example files collected `979` tests; focused exact selector
  collected `16` and ran split-4 as `15 passed, 1 skipped`. The skip is the
  existing `03-matmul-multicta.py` `Out of resources` example guard. No new
  independent `FZ-*`.

- 2026-04-22: TMEM memdesc runtime abstraction audit completed. Durable note:
  `tmem_memdesc_runtime_abstraction_20260422.md`. Current design target:
  producer-chain walking is not part of valid TTGIR lowering/codegen. Runtime
  memdesc lowering remains one packed TMEM `taddr` carrying current physical
  origin; static descriptor layout, storage footprint, row-plan, copy,
  reduction, and MMAv5 legality facts must be derivable from the current
  `MemDescType` and tensor-memory layout. Disallowed current chain-dependent
  sites are catalogued for
  `TensorMemoryUtils.cpp`, `TensorMemoryToLLVM.cpp`, `MMAv5.cpp`, `Ops.cpp`,
  and `python/src/gluon_ir.cc`; allowed transform-time uses are catalogued for
  `OptimizeTMemLayouts.cpp`, `TensorMemoryAllocation.cpp`, fence insertion, and
  scale materialization. No code or validation changes in this analysis slice.
  First underlying issue to fix: TMEM view ops must compute exact result
  `MemDescType` state and exact lowered `taddr` updates. The result type must
  carry the view's shape/storage/layout relative to its own current `taddr`;
  origin-changing views must advance `taddr` from source type/layout and
  explicit/dynamic offsets only; layout-only views must leave `taddr` unchanged.
  Consumer lowering should then never recover root descriptors, parent view
  chains, accumulated offsets, or non-zero query origins.
  Column-coordinate decision: `LinearLayout` `col` should be physical
  element-slot, not raw hardware word-column. Packed and unpacked sub-32-bit
  storage are represented by layout bases/column stride; codegen converts to
  hardware word columns and emits any required subword packing/unpacking at the
  instruction boundary, so a separate `addressMode` type field is not part of
  the target abstraction.
  Lowering may pessimistically emit a local `subword_index` SSA value for
  sub-32-bit dtypes; if the selected instruction path does not use it, or
  dtype/alignment proves it irrelevant, it should be removed by ordinary
  MLIR/LLVM DCE or constant folding. It must remain a local pack/unpack or
  diagnostic input, not a second provenance channel.
  Required performance gate before implementation: prototype this lowering by
  hand-writing MLIR/LLVM IR for f32, aligned f16/i8, and unaligned f16/i8
  physical-element-column origins, then run through the rest of the compiler
  and inspect reduced IR/PTX to verify unused or provably-zero `subword_index`
  code is removed before wiring the production lowering.

- 2026-04-21: Round 61 frontend plus structural adversarial lane B completed.
  Report: `agents/fuzz_round61_frontend_structural_lane.md`. Required
  `make -j8` was a no-op. Frontend selector `tensor_memory or tmem_`
  collected `29/225` and ran as `28 passed, 1 failed`, reproducing existing
  `FZ-20260421-0023`. Full structural fuzzer collected `33` and ran split-4
  as `9 passed, 24 xfailed`. No new independent `FZ-*`.

- 2026-04-21: Round 61 lit/codegen compiler breadth lane C completed. Report:
  `agents/fuzz_round61_lit_codegen_lane.md`. Required `make -j8` and
  `ninja triton-opt` were no-ops. Lit breadth ran `19 passed`; direct
  `triton-opt` replays ran `4/4` with exit code `0`. No compiler crash,
  diagnostic drift, conversion failure, NVWS failure, partition failure,
  ownership failure, or new independent `FZ-*`.

- 2026-04-21: Round 61 runtime-matrix adversarial lane A completed. Report:
  `agents/fuzz_round61_runtime_adversarial_lane.md`. Required `make -j8` was
  a no-op. Selector A collected `108/1615` and passed split-4 as
  `108 passed`; selector B collected `33/1615` and passed split-4 as
  `33 passed`. Aggregate: `141 passed, 0 failed, 0 skipped`. No new
  independent `FZ-*`.

- 2026-04-21: Round 62 local convolution/router examples lane completed.
  Report: `agents/fuzz_round62_local_examples_convolution_router_lane.md`.
  Required `make -j8` was a no-op. Broad collection found `64` tests; focused
  selector collected `12` and ran split-4 as `12 passed, 0 failed, 0 skipped`.
  No new independent `FZ-*`.

- 2026-04-21: Round 62 local `test_core.py` TMEM slice completed. Report:
  `agents/fuzz_round62_local_test_core_tmem_lane.md`. Selector
  `tmem and (copy or ld or load or store or mma or tcgen05)` collected
  `55/18114` and ran split-4 as `50 passed, 5 skipped`. Skips are existing
  OOR TMEM copy matrix guards; no new independent `FZ-*`.

- 2026-04-21: Round 62 backend audit lane C completed. Report:
  `agents/fuzz_round62_backend_audit_lane.md`. Required `make -j8` was a
  no-op and focused runtime pair ran as `2 passed`. Audit produced future
  probe candidates only; no observed new independent `FZ-*`.

- 2026-04-21: Round 62 random runtime-matrix lane A completed. Report:
  `agents/fuzz_round62_random_runtime_matrix_lane.md`. Seed `6042162`
  selected `80` exact nodeids with zero overlap against the Round 60-61 avoid
  set. Split-4 runtime result: `75 passed, 5 skipped, 0 failed`; skips were
  known lifted descriptor allocation-limit and shared-memory OOR boundaries.
  No new independent `FZ-*`.

- 2026-04-21: Round 62 generated contrast lane B completed. Report:
  `agents/fuzz_round62_generated_contrast_lane.md`. Probes reproduced only
  existing buckets `FZ-0001`, `FZ-0003`, `FZ-0007`, `FZ-0012`, `FZ-0013`,
  `FZ-0015`, `FZ-0020`, and `FZ-0022`; clean unsupported boundaries stayed
  clean. No new independent `FZ-*`.

- 2026-04-21 16:59 UTC: active phase changed from fuzzing/cataloging to
  systematic repair-plan execution. Slice 1 completed dynamic encoded TMEM
  `memdesc_index` lowering for the checked-in `FZ-20260421-0001` structural
  sentinels. Implementation: `ViewOpToLLVM.cpp` now lowers non-constant
  encoded tensor-memory leading indices by bit-decomposing the runtime index,
  using `getTMemViewOffset` for each basis bit, XOR-composing the encoded TMEM
  row/column offset, and advancing the base pointer. Tests: direct
  `generic-pass-dynamic-index-load-only-128x32` and
  `generic-pass-dynamic-index-chain1` are now positives; chain0 remains xfailed
  as `FZ-20260421-0003` descriptor-view mapping. Validation: required
  `make -j8`; exact rows `1 passed`, `1 passed`, `1 xfailed`; full structural
  fuzzer split-4 `11 passed, 22 xfailed`. Remaining inventory for this phase:
  broaden `FZ-0001` over dynamic/control-flow-carried copy, `ld.red`,
  MMAv5/scales, and branch/helper-yielded descriptor consumers; then repair
  `FZ-0003` descriptor-view packet/layout mapping exposed by chain0.

- 2026-04-21 17:05 UTC: dynamic `FZ-0001` broad-consumer validation promoted
  to checked-in tests. Temporary replays showed runtime-index `ld.red`,
  branch-yielded copy, and runtime-index copy now pass; compiler-only
  `ld.red`/scales dynamic-index repros no longer fail on illegal
  `ttg.memdesc_index`; the Round 56 dense copy matrix has five former
  dense-index `FZ-0001` rows green across 1CTA/2CTA and branch/runtime index.
  Added structural positives for dynamic branch copy, dynamic runtime copy, and
  runtime-index `ld.red`. Validation: required `make -j8`; exact new nodeids
  `3 passed`; full structural fuzzer split-4 `14 passed, 22 xfailed`.
  Remaining clean-negative/bug inventory: branch-subslice copy belongs with
  descriptor-view/copy-planning, not late dynamic-index lowering; the next core
  repair target is `FZ-20260421-0003` descriptor-view packet/layout mapping.
