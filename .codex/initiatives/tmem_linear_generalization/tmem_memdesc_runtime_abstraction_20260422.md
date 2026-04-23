# TMEM Memdesc Runtime Abstraction

Date: 2026-04-22

Branch: `codex/tmem`

HEAD at analysis time: `2602983e8af22d7aa6befd40db42df7c94b6685b`

Merge-base used for branch-diff orientation: `11ee1144a737006921231bbd3386c187812c38e1`

## Problem Statement

TMEM lowering and codegen must not recover semantic state by walking the
producer chain of a memdesc SSA value. A valid TTGIR program must remain valid
if a tensor-memory descriptor is forwarded through a block argument, selected
with `arith.select`, returned from `scf.if`, passed through a helper function, or
otherwise loses a visible local chain of `memdesc_subslice` / `memdesc_index` /
`memdesc_reshape` / `memdesc_trans` / `memdesc_reinterpret` operations.

Optimization passes may still walk producer chains to recognize peephole
patterns and rewrite IR. After such rewrites, however, the resulting TTGIR must
be self-contained: lowering to LLVM/PTX must use only the memdesc SSA value's
runtime contents, the memdesc result type, and explicit operation attributes.

## Current Lowering Contract

Current LLVM lowering already treats a tensor-memory memdesc runtime value as a
single tensor-memory address:

- `ttng.tmem_alloc` lowers to an address-space-3 pointer whose integer payload is
  the hardware TMEM base address plus the allocation's packed row/column offset.
- TMEM `memdesc_subslice`, `ttng.tmem_subslice`, and `memdesc_index` lower by
  adding a packed physical offset to that pointer.
- TMEM `memdesc_reshape`, `memdesc_trans`, and `memdesc_reinterpret` lower as
  runtime no-ops.
- TMEM `ttng.tmem_load`, `ttng.tmem_store`, `ttng.tmem_copy`, and MMAv5 lowering
  consume the pointer by `ptrtoint` and issue `tcgen05.*` instructions from the
  resulting address.

The runtime payload is therefore effectively:

```text
taddr : i32
  high bits: physical TMEM row base
  low bits:  TMEM column base
```

This payload is enough for dynamic origin selection. For example, two
same-typed descriptors that differ only by origin can be selected dynamically:

```mlir
%view = arith.select %pred, %origin, %offset
%x = ttng.tmem_load %view : !ttg.memdesc<64x32xf32, ...> -> tensor<64x32xf32, ...>
```

The selected runtime value carries the current row/physical-element-column base.
The memdesc type carries the static access pattern.

## Proposed Invariant

A tensor-memory memdesc SSA value is a pair of:

1. Runtime state:
   - one `taddr` value, lowered as `ptr addrspace(3)` or `i32`;
   - this is the current row/physical-element-column origin of the descriptor.
2. Static state:
   - the `!ttg.memdesc` result type;
   - its tensor-memory encoding describes the complete relative physical access
     layout of the descriptor from that current origin;
   - explicit operation attributes describe requested instruction variants,
     reduction modifiers, copy barriers, MMA shape, etc.

No lowering or verifier code may require the producer chain to discover:

- the root allocation;
- prior view offsets;
- whether the value came from a subslice, index, reshape, trans, or reinterpret;
- whether a reinterpret was a special physical bitcast;
- a parent row plan or wider support image;
- a query origin that is still relative to a root descriptor.

If any of those facts are needed for correct lowering, they must be represented
in the current memdesc type or in an explicit operation attribute. If they are
only useful for optimization, a pass may use producer-chain analysis before
lowering and then rewrite to self-contained IR.

## Is Raw `taddr` Enough?

One packed `taddr` is enough as the runtime representation of the current
row/physical-element-column origin. It is not enough as the whole descriptor
abstraction.

Correct codegen needs both:

```text
runtime:  current row/physical-element-column base taddr
static:   current MemDescType + tensor-memory layout
```

The static part must answer these locally from the current memdesc type and its
tensor-memory layout:

- logical result shape, element type/bitwidth, memory space, mutability, and
  allocation/storage shape; these are already `MemDescType` responsibilities;
- exact relative mapping from logical coordinates to TMEM row and physical
  element slot;
- `cta_group` / two-CTA ownership from the tensor-memory encoding/layout;
- storage footprint or support-image information from the current descriptor's
  allocation/storage shape and layout, relative to the current `taddr`;
- the complete set of semantically valid instruction lowerings derivable from
  the current descriptor layout and the operation being lowered.

Instruction-selection heuristics may inspect producer/view chains as peephole
optimization context, but only to choose among lowerings already proven valid
from the current type/layout. The set of valid lowerings must not depend on the
IR chain.

The runtime state should not grow a parent pointer, a root base, a chain id, or a
side table entry. Those would still make the codegen result depend on dynamic
provenance rather than the SSA value and its type.

## Column Coordinate Convention

The tensor-memory `LinearLayout` `col` dimension should represent physical
element slots, not raw 32-bit word columns.

That is the more natural descriptor model:

- memdesc shapes are expressed in elements;
- view algebra composes in element-slot coordinates;
- layout bases encode the storage stride from logical elements to physical
  element slots;
- f16/i8/f8 layouts do not need a second "address mode" convention;
- codegen is responsible for translating element slots into the hardware word
  address and for packing/unpacking sub-32-bit elements.

With this convention, a separate static `addressMode` field is not part of the
target abstraction. The element type and exact element-slot layout are the
static facts. A consumer that needs a PTX hardware address computes:

```text
elements_per_word = 32 / element_bitwidth
word_col          = floor(physical_element_col / elements_per_word)
subword_phase     = physical_element_col % elements_per_word
```

The hardware `tcgen05` address uses `word_col`; register packing, unpacking,
lane shifts, widened packets, or read/modify/write handling use
`subword_phase` when the ISA path supports it.

Packing is represented by the layout's column stride:

```text
packed f16:    logical col +1 -> physical element col +1
unpacked f16:  logical col +1 -> physical element col +2
packed f8/i8:  logical col +1 -> physical element col +1
unpacked f8/i8 logical col +1 -> physical element col +4
```

In the unpacked f16/f8/i8 cases, every logical element starts at
`subword_phase == 0` even though the dtype is sub-32-bit. The pack/unpack logic
then degenerates to extension/truncation around the 32-bit hardware slot rather
than bit shifting and masking/oring within a packed slot.

This also clarifies the runtime `taddr` contract. If the lowered memdesc value
were always the raw hardware row/word-column address, it could not represent an
odd packed-f16 column or a non-4-aligned packed-i8 column without losing the
subword phase.
The preferred single-payload model is therefore:

```text
compiler memdesc taddr = packed row + physical-element-column origin
PTX tcgen05 address    = packed row + hardware word-column origin
```

The conversion from compiler `taddr` to PTX address happens at the instruction
emission boundary. This still keeps the runtime memdesc to one scalar value and
preserves dynamic control-flow joins for descriptors with the same type. If a
dynamic subword phase cannot be realized for a particular instruction family,
that is a local ISA/codegen limitation to diagnose from the current type and
operation semantics, not from the producer chain.

The lowering helper may compute the subword phase pessimistically for
sub-32-bit element types:

```text
subword_index = physical_element_col % elements_per_word
```

That value should be ordinary SSA. When the selected instruction path only needs
the hardware word address, or when dtype/alignment makes the phase statically
irrelevant, MLIR/LLVM DCE and constant folding can remove the computation. The
important constraint is that uses of `subword_index` are local to pack/unpack,
lane-selection, read/modify/write, or diagnostic codegen. It should not become a
second provenance channel for recovering parent view state.

Before implementing this in the TMEM lowering, prototype the SSA shape directly
in MLIR/LLVM IR and run it through the remaining compiler pipeline. The
prototype must prove the performance-sensitive cases:

- 32-bit element types do not leave a live `subword_index` computation.
- Sub-32-bit element types with statically hardware-column-aligned accesses fold
  `subword_index` to zero and remove unused phase logic.
- Sub-32-bit unaligned accesses keep only the phase computation actually
  consumed by pack/unpack or diagnostics.

Do not rely on this cleanup by assumption. Capture the reduced IR/PTX evidence
before wiring the pattern into production lowering.

## View Operation Semantics

Each TMEM view operation should update exactly one side of the abstraction:

- `memdesc_subslice`, `ttng.tmem_subslice`, and encoded `memdesc_index`:
  - update the runtime `taddr` by adding the view's
    row/physical-element-column origin delta;
  - produce a result type whose layout is relative to the new current origin.
- Dynamic `memdesc_index`:
  - computes a dynamic packed row/physical-element-column delta from the source
    type's relative layout;
  - advances the runtime `taddr`;
  - the result type still describes the post-index relative layout.
- `memdesc_reshape`, `memdesc_trans`, and layout-preserving
  `memdesc_reinterpret`:
  - do not change runtime `taddr`;
  - produce a result type whose relative layout has been composed with the view.
- Bitwidth-changing or physical reinterpret:
  - still has no extra runtime payload unless it changes origin;
  - any special raw-word consumer behavior must be derived from the physical
    element-slot layout and element type, or be explicit on the consumer op, not
    discovered by checking that the defining op was `memdesc_reinterpret`.

After this rule, use-site lowering should never need a `TMemLdStQueryLayout`
with non-zero `origin`. Non-zero origins are an intermediate type-inference
device only; before LLVM lowering, the origin delta should have been folded into
the runtime `taddr`, and the static layout should be expressed relative to
zero.

## Static Descriptor Shape

The branch already has most of the raw material:

- `TensorMemoryLinearEncodingAttr` and `TensorMemoryScalesEncodingAttr` carry
  tensor-memory layout state.
- `MemDescType` carries shape, element type, memory space, mutability, and
  allocation shape.
- `LinearLayout` can describe non-canonical physical mappings.
- `TMemLdStRowPlan`, `TMemLdStQueryLayout`, and `TMemPhysicalQuery` describe the
  missing codegen concepts, but today they are mostly reconstructed from a
  `Value` by producer-chain analysis.

The intended destination is a type-local descriptor analysis, for example:

```c++
struct TMemDescriptorPlan {
  gpu::MemDescType memTy;
  LinearLayout relativeLayout;
  unsigned elementBitWidth;
  bool twoCTAs;
  std::optional<TMemLdStRowPlan> rowPlan;
  SmallVector<TMemInstructionPlan> legalInstructionPlans;
};
```

This structure should be constructible from the current `MemDescType`, its
layout, and the semantic operation kind being lowered. It should not accept a
`Value` unless that value is used only to fetch its type. Operation attributes
may describe the operation itself, such as reduction kind or barrier operands,
but they must not be required to prove descriptor-layout legality.

If a descriptor is too small for a hardware instruction footprint, lowering
should reject it cleanly. For example, `tcgen05.copy` has no dense `128x32b` or
`128x64b` destination atom; a `128x1xf32` or `128x2xf32` destination memdesc is
therefore not codegenable as a dense copy unless the current descriptor layout
itself represents one of the legal copy families. The compiler should not copy
extra columns just because a hidden parent allocation happens to contain them.

If a view needs a different descriptor image to use a legal instruction, there
are only two acceptable representations:

- the current descriptor type's layout explicitly describes that legal image
  relative to the current `taddr`; or
- an optimization pass rewrites the IR to an explicit supported descriptor and
  consumer before lowering.

There should not be a third path where LLVM lowering walks from the consumer's
operand back to a parent descriptor to borrow a hidden support image. A useful
counterexample is a dynamic join between two same-shaped views at different
offsets in a root allocation: if a widened support footprint is legal for one
origin but not the other, the selected descriptor cannot rely on root provenance
to decide legality. Either the current result type represents the legal
instruction image directly, or the lowering is not semantically valid for that
descriptor type.

### Implementation Checkpoint: Active Physical-Support Views

As of 2026-04-23 01:54 UTC, the first MMAv5/scales-facing physical-support
slice is implemented. It exposed an important distinction in the "current type
is self-contained" rule:

- compact active views have a current layout whose physical image is the same
  size as the logical active shape; these can plan with `allocShape == shape`
  because the runtime `taddr` already names the active origin;
- exact physical-support views have a current layout whose logical shape is
  narrower than the physical support image required to describe the selected
  columns or rows; these must keep that support image in the type/layout and
  must not be compacted to the logical shape during instruction planning.

The motivating repro is a `128x64xf32` accumulator subview taken from a
tile-permuted `128x128` parent at logical column 64. The runtime `taddr`
advances to the current physical origin, but the active logical column bit
`32` still maps to a physical column displacement outside the compact
`64`-column window. Compacting the result encoding to `out = [128,64]`
therefore turned that high column basis into zero, making the layout
non-injective and causing MMAv5 lowering to abort while pseudoinverting it.
The fixed inference lets exact subview algebra or exact-view fallback retain a
wider support image instead of forcing compact output dimensions.

This checkpoint does not make every exact physical-support view fully
type-local yet. Some preserved parent-encoding cases still route through
producer-aware support planning. The invariant remains unchanged: lowering must
eventually derive legality and address generation from the current `taddr`,
current `MemDescType`/layout, and operation semantics only.

### Implementation Checkpoint: Type-Local MMAv5 Family Address

As of 2026-04-23 02:19 UTC, MMAv5 address layout and tile-order offset lowering
derive the family address layout from the current descriptor type for
row-preserving narrowed MMAv5-family descriptors. The key change is that
`getMMAv5TMemFamilyAddressLayout(MemDescType)` no longer rejects
`shape != allocShape`: the current descriptor's allocation shape is still the
static family footprint, and the existing MMAv5 accumulator/scaled-info helpers
can use it to recover the family address layout without walking the producer
chain.

This is an address-generation slice, not a blanket "raw current layout is
always a support query" rule. A rejected overbroad experiment made direct ld/st
use the raw current tile-permuted narrowed encoding as a support query and
failed runtime correctness. The valid direct-support query for those
descriptors is still the canonical MMAv5 family support layout, derivable from
the current type's family facts. Operation-specific type-local planning remains
necessary: MMAv5 address/tile-order can use the family address layout, while
direct ld/st support planning must select the canonical support family instead
of blindly using the current tile-permuted encoding.

As of 2026-04-23 02:32 UTC, that type-local computation is also split into
explicit APIs: `getTypeLocalMMAv5TMemAddressLayout(MemDescType)` and
`getTypeLocalMMAv5TMemViewOffsetForLowering(MemDescType, offsets)`. The
value-taking MMAv5 helpers remain as wrappers for physical-bitcast behavior and
legacy producer-chain fallback, but the semantic local plan is now separated
from the compatibility path.

## Current Disallowed Chain-Walking Sites

These sites currently use parent operations to decide semantic verifier,
lowering, or codegen behavior. They should be migrated to type-local analysis or
to pre-lowering canonicalization.

### `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`

- `getTMemScalesRootEncoding(Value)` follows view and forwarding chains to find
  a scales root encoding.
  - This should become a type-local property. A descriptor that is semantically
    a scales descriptor should have scales encoding on its current type.
- `inferStandaloneTMemLdStQueryLayoutImpl(Value, ...)` recursively walks
  `memdesc_subslice`, `ttng.tmem_subslice`, `memdesc_index`, `memdesc_reshape`,
  `memdesc_trans`, and `memdesc_reinterpret`.
  - This is the central violation. Its output is used by verifiers, Gluon
    frontend helpers, layout optimization, LLVM lowering, copy planning, and
    MMAv5 lowering.
  - The lowering-facing replacement should compute a zero-origin relative query
    from `MemDescType` only.
- `isTMemPhysicalBitcast(Value)` checks whether the defining op is a
  `memdesc_reinterpret` with `tmem_physical_bitcast`.
  - If this affects codegen, it must be represented in result type/encoding or
    explicit consumer attributes.
- `getMMAv5TMemAddressLayout(MemDescType, Value)` and
  `getMMAv5TMemViewOffsetForLowering(Value, MemDescType, offsets)` use the value
  chain to choose MMAv5 address layout and tile order.
  - Partially migrated: narrowed MMAv5-family descriptors now derive the family
    address layout from current `MemDescType` before legacy producer-chain
    fallback, and the type-local address/offset computation is now exposed as
    separate helpers. Remaining fallbacks and physical-bitcast checks still need
    to be split into type-local semantics versus optimizer-only rewrites.
- `getBackingTMemLdStRowPlan(Value)` walks through forwarding sources and view
  ops to borrow a wider parent row plan.
  - Any required row plan must be derivable from the current descriptor layout or
    encoded as static descriptor metadata.
- `shouldPreferTMemLdStQueryTypeLayoutsBeforeRawQuery`,
  `shouldPreferTMemLdStQueryTypeLoweringBeforeRawQuery`,
  `shouldDeferTMemLdStCanonicalM64SplitNCompatibleLayout`,
  `shouldUseExactTMemLdStViewLayoutForM64DirectView`,
  `disallowTMemLdStRawQueryRowPlanOverride`, and
  `disallowTMemLdStQueryTypeRescue` make codegen choices based on producer
  shape.
  - These should collapse into type-local layout and row-plan predicates.
- `getTMemLoadReductionLayoutForMemDesc(Value, ...)` and
  `canonicalizeTMemLoadReductionType(..., Value, ...)` select reduction layouts
  using raw/support query reconstruction from the chain.
  - Reduction legality and layout should be determined from the descriptor type
    and requested reduction operation.
- `getTMemLdStQueryTypes(Value)` uses the producer chain and standalone view
  inference to build fallback query types.
  - Fallback query types should either be type-local derived plans or an
    optimizer rewrite before lowering.
- `isPureOuterTMemIndexView`, `isHigherRankHalfRowsSubview`,
  `isDirectHalfRowsSubview`, and related half-row predicates match specific
  producer chains.
  - These may remain optimizer matchers, but must not gate verifier/lowering
    legality.
- `disallowTMemLdStTypeOnlyFallback(Value, ...)` uses the chain to reject
  fallbacks.
  - A true hardware-negative should be expressible from current type/layout and
    requested atom.
- `getTMemViewOffsetForLowering(Value, offsets)` and
  `getTMemSubviewOffsetForLowering(memdesc_subslice)` compute origin deltas by
  reconstructing source and destination queries from the chain.
  - View lowering should compute the delta from the source type/layout and the
    view op's explicit offsets. It should not ask how the source was produced.
- `getAlreadyAdjustedTMemSubviewBaseOffset(Value)` and
  `getTMemSubviewRelativeBaseOffset(Value, baseOffset)` subtract already-lowered
  view offsets recovered from the chain.
  - These functions are symptoms of split state. In the target model,
    already-applied origin deltas live only in runtime `taddr`, and use-site
    static base offsets are relative to that `taddr`.
- `inferStandaloneTMemViewTypeImpl(Value, ...)`,
  `inferStandaloneTMemRegLayoutQueryType(Value, ...)`,
  `inferStandaloneTMemViewType(Value, ...)`,
  `inferStandaloneTMemPhysicalQuery(Value, ...)`, and
  `inferExactTMemPhysicalQuery(Value, ...)` reconstruct type/query state from a
  producer chain.
  - The type-inference logic belongs on the view ops themselves or in an
    optimizer/canonicalizer that materializes a self-contained result type.
- `getTMemLdStSupportQueryPlan(Value, ...)` and helpers such as
  `getHalfRowsTMemLdStSupportQueryLayout`,
  `getDirectHalfRowsTMemLdStSupportQueryPlan`,
  `getColumnSubviewTMemLdStSupportQueryPlan`,
  `getOuterIndexTMemLdStSupportQueryPlan`, and
  `getGenericTMemLdStReshapedSupportQueryPlan` derive hidden support images from
  the chain.
  - Support images must either be encoded in the current descriptor's static
    layout/plan or produced by an optimizer rewrite.
- `isUnsupportedDirectTMemLdStDescriptorView(Value, ...)`,
  `getUnsupportedDirectTMemLdStVariantReason(Value, ...)`, and helper checks
  decide true hardware legality using chain-derived queries.
  - Hardware negatives should be local: current type/layout + requested atom.
- `selectTMemCopyPhysicalQuery(Value, ...)` chooses between standalone and exact
  physical queries by walking the destination chain.
  - `ttng.tmem_copy` lowering should consume one explicit destination physical
    query derived from the destination type, or an optimizer should rewrite the
    copy into a supported explicit schedule.

### `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`

- `lowerTMemLdStFromTypes(...)` accepts both `MemDescType` and `Value
  memDescValue`, then calls the chain-dependent query, support, row-plan, and
  base-offset helpers.
  - The target signature should not need `memDescValue` except for diagnostics.
    It should accept `MemDescType`, current `taddr`, the register type, and a
    type-local instruction plan.
- `TensorMemoryLoadOpConversion`, `TensorMemoryStoreOpConversion`, and initialized
  `TensorMemoryAllocOpConversion` pass the original memdesc operand/result value
  into `lowerTMemLdStFromTypes`.
  - This should become type-only planning plus runtime pointer use.
- `copySharedToTmem(...)` calls `selectTMemCopyPhysicalQuery(op.getDst(), ...)`
  and then subtracts `getTMemSubviewRelativeBaseOffset(op.getDst(), ...)`.
  - This should use one type-local copy destination plan relative to the current
    destination `taddr`.
- The Nvidia-local `MemDescIndexOpConversion` duplicates the generic
  tensor-memory index lowering and rejects some non-leading dynamic index cases.
  - This does not itself walk parents, but it should be unified with the generic
    source-type-only view lowering so dynamic origin updates are consistent.

### `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`

- `DotOpMmaV5TmemLoader::build(...)` calls
  `getMMAv5TMemAddressLayout(memTy, memDescValue)`.
- `getSortedTMemTileOrder(...)` calls
  `getMMAv5TMemViewOffsetForLowering(memDescValue, memTy, offsets)`.

Both should become type-local. MMAv5 lowering should use current `taddr` plus
the current descriptor's static address layout.

### `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`

- `verifyTMEMOperandPreconditions` and `verifyTMEMOperand` call
  `isUnsupportedDirectTMemLdStDescriptorView`,
  `getTMemLdStQueryTypes`, `inferStandaloneTMemLdStQueryLayout`,
  `getTMemLdStSupportQueryPlan`, `inferStandaloneTMemViewType`, and row-plan
  helpers.
- `TMEMLoadOp::verify` repeats the same pattern for `ld.red`.

Verifiers are not LLVM lowering, but they define which TTGIR is valid. Validity
must not depend on local producer-chain visibility. These checks should use the
same type-local descriptor plan as LLVM lowering.

### `python/src/gluon_ir.cc`

The Gluon bindings call the same chain-dependent helpers for auto-layout and
legality:

- `canonicalizeTMemLoadReductionType`
- `inferStandaloneTMemRegLayoutQueryType`
- `getTMemLdStSupportQueryPlan`
- `inferStandaloneTMemLdStQueryLayout`
- `inferStandaloneTMemViewType`
- `isUnsupportedDirectTMemLdStDescriptorView`
- `getTMemLdStQueryTypes`
- `getTMemLoadReductionLayoutForMemDesc`

The frontend may still run optimization-time peepholes, but the legality and
layout-selection contract exposed to TTGIR should be type-local. Otherwise a
Gluon helper boundary or control-flow join can change whether the backend can
compile the same semantic descriptor.

### Public Headers

`include/triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h` exposes many
`Value`-taking planning APIs. The API surface should be split:

- lowering/verifier APIs: type-local, no producer-chain walk;
- optimizer analysis APIs: explicitly named as view-chain matchers and used only
  from transforms.

## Allowed Chain-Walking Sites

These sites may walk producers because their role is optimization, liveness, or
IR rewriting. The constraint is that their output must be self-contained before
LLVM lowering.

### `lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp`

Allowed uses include:

- matching split-load, half-slice, leading-slice, and full-view replay patterns;
- recognizing reshape/trans/subslice chains and replacing them with explicit
  TMEM loads/stores/subslices;
- choosing profitable load-reduction rewrites;
- fusing load+reduce forms when a local peephole proves the rewrite.

This pass may call current chain-dependent helpers while it is still a
pre-lowering optimization. The migration goal is that any plan it relies on is
materialized in the rewritten IR or in result types before LLVM conversion.

### `lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp`

Allowed uses include:

- tracing aliases through memdesc view ops, `arith.select`, `scf.if`, `scf.for`,
  `scf.while`, and warp-specialization captures to find root `ttng.tmem_alloc`
  ops;
- conservative liveness extension for aliasing tensor-memory descriptors;
- rematerializing shared MMA scales into TMEM storage;
- cleaning up dead scale alias chains.

This is allocation and liveness analysis. It may reason about roots and aliases,
but it must not be the only place where semantic view offsets or instruction
layouts are represented.

### `lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp`

Walking users through `MemDescViewTrait` for fence placement is an effect and
alias analysis concern. It is allowed as long as it does not compute PTX address
operands or instruction legality.

### Scale Materialization Helpers

`getMMAv5ScaledBScaleStorageTypeThroughViews` in
`lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` and the similar helper in
`TensorMemoryAllocation.cpp` walk view chains to discover scale storage during a
materialization transform. That is acceptable as a transform-time convenience.
Final scaled-MMAv5 codegen should still see a self-contained scale memdesc type.

## First Underlying Issue To Fix

The first repair target is the TMEM view operations themselves.

For every tensor-memory memdesc view op, fix both sides of the memdesc state:

1. Result type computation.
   - The result `MemDescType` must have the right shape, allocation/storage
     shape, element type, tensor-memory encoding, and exact relative
     `LinearLayout` for the view.
   - The result layout must describe TMEM row/physical-element-column
     displacements
     relative to the result descriptor's own current `taddr`, not relative to a
     root allocation or parent descriptor.
   - A view op should preserve the source encoding only when that encoding is
     exactly correct for the result view. Otherwise it must compose/project the
     layout algebraically or reject the view before lowering.

2. Runtime `taddr` lowering.
   - View ops that change physical origin, such as `memdesc_subslice`,
     `ttng.tmem_subslice`, and encoded `memdesc_index`, must update the lowered
     `taddr` using only the source `MemDescType`, the explicit view offsets, and
     any dynamic index value.
   - View ops that only reinterpret the relative layout, such as reshape,
     transpose, and layout-only reinterpret, must leave the lowered `taddr`
     unchanged.
   - After a view op is lowered, no later consumer should need to subtract an
     "already adjusted" offset or recover a non-zero query origin from the parent
     chain.

This is the root issue behind the current producer-chain dependence. Once view
ops maintain this invariant, a consumer can lower from:

```text
current taddr + current MemDescType/layout + operation semantics
```

without reconstructing:

```text
root descriptor + parent view chain + accumulated offsets
```

## Migration Plan

### Execution Checklist Started 2026-04-22 22:05 UTC

Active branch: `codex/tmem`

Active HEAD at checklist start:
`2602983e8af22d7aa6befd40db42df7c94b6685b`

Execution rule for this migration: do not change user-facing TMEM APIs. The
only planned semantic tightening is that `tcgen05.copy` destinations that are
too small for any ISA atom, such as dense `128x1xf32` or `128x2xf32`, should
be rejected cleanly instead of borrowing hidden parent columns.

Checklist state:

- [x] Record the memdesc model and migration plan in durable initiative docs.
- [ ] Audit every current TMEM `Value`-taking planning/lowering helper and
  classify it as semantic-to-rewrite, optimizer-only, or allocation/effect-only.
- [ ] Add prototype evidence for physical-element-column `taddr` lowering and
  `subword_index` DCE in aligned/common cases before wiring production lowering.
- [ ] Introduce type-local descriptor-planning helpers beside the existing
  chain-dependent helpers.
- [ ] Fix TMEM view result type computation and source-type-only `taddr`
  lowering for origin-changing views.
- [ ] Migrate ld/st planning to type-local analysis.
  First row-plan slice complete: active self-contained subviews use the current
  descriptor's row plan instead of parent backing-row plans. First raw-query
  slice complete: active self-contained subviews dispatch through
  `inferTypeLocalTMemLdStQueryLayout(MemDescType)`.
- [ ] Migrate subword pack/unpack handling for packed and unpacked sub-32-bit
  layouts.
- [ ] Migrate `ld.red` legality/layout selection to the type-local planner.
- [ ] Migrate `tcgen05.copy` planning to destination-type-only analysis and add
  too-small-copy clean negatives. First partial slice complete: active
  self-contained subview destinations select a type-local physical query;
  direct roots and legacy parent-encoding views still use the old
  standalone/exact selection. First clean-negative slice complete:
  `128x1xf32` and `128x2xf32` current descriptors report an explicit hardware
  copy-atom boundary.
- [ ] Migrate MMAv5/scales address planning to current type/layout plus runtime
  `taddr`. First partial slice complete: narrowed MMAv5-family descriptors now
  derive address layout and tile-order offsets from current `MemDescType`
  family facts before legacy producer-chain fallback. First helper-split slice
  complete: type-local MMAv5 address and tile-order computations now have
  explicit APIs separate from value-taking fallback wrappers.
- [ ] Split public helper APIs into lowering-facing type-local helpers and
  optimizer-only producer-chain matchers.
- [ ] Delete or quarantine obsolete support-query, backing-row, and
  already-adjusted-offset rescue paths after coverage is green.
- [ ] Run staged lit, focused pytest, full 4-GPU runtime matrix, structural
  fuzzer, and example performance validation.

### Audit Classification Started 2026-04-22 22:05 UTC

Semantic paths that must be rewritten to type-local planning:

- `lowerTMemLdStFromTypes` in Nvidia LLVM lowering. It currently receives
  `Value memDescValue` and uses descriptor provenance to choose query types,
  raw/support query layouts, row plans, base offsets, and rescue ordering.
- `TMEMCopyOp::verify` and `copySharedToTmem`. They currently call
  `selectTMemCopyPhysicalQuery(Value, ...)`, which compares standalone and exact
  physical queries reconstructed from the destination chain, then subtracts
  already-lowered view offsets.
- MMAv5 TMEM address planning. `getMMAv5TMemAddressLayout(MemDescType, Value)`
  and `getMMAv5TMemViewOffsetForLowering(Value, ...)` still use
  `isTMemPhysicalBitcast(Value)` and raw query reconstruction.
- TMEM op verifiers in `Ops.cpp`. `verifyTMEMOperand` and `TMEMLoadOp::verify`
  use `getTMemLdStQueryTypes(Value)`, support queries, raw queries, backing row
  plans, and optimizer-replayability to decide whether TTGIR is valid.
- Gluon layout selection in `python/src/gluon_ir.cc`. The frontend uses the same
  chain-dependent helpers to pick register layouts and reduction layouts.

Producer-chain utilities that may survive only as optimizer/allocation helpers:

- Replay recognizers in `OptimizeTMemLayouts.cpp`, including half/full/leading
  descriptor-view matchers and reduction peepholes. Their output must be
  rewritten IR with self-contained result types before LLVM lowering.
- Alias/liveness walks in `TensorMemoryAllocation.cpp`, including memdesc alias
  tracing through `arith.select`, `scf.if`, `scf.for`, and scale
  rematerialization. These may reason about roots but must not be required for
  codegen legality.
- Fence insertion user walks and scale-materialization discovery helpers. These
  are effect/transform analyses, not instruction planners.

High-priority hacks and debt to remove after replacement coverage exists:

- `inferStandaloneTMemLdStQueryLayoutImpl(Value, ...)` as a semantic source of
  truth. It is the central producer-chain reconstruction stack.
- `getBackingTMemLdStRowPlan(Value)` and support-query row borrowing. These
  borrow hidden parent row footprint and conflict with dynamic descriptor joins.
- `getAlreadyAdjustedTMemSubviewBaseOffset` and
  `getTMemSubviewRelativeBaseOffset`. These exist because origin state is split
  between lowered `taddr` and reconstructed query origins.
- `isTMemPhysicalBitcast(Value)`. If physical reinterpret behavior affects
  codegen, the result type/layout or explicit consumer semantics must carry it.
- Query rescue/preference helpers such as
  `shouldPreferTMemLdStQueryTypeLoweringBeforeRawQuery`,
  `disallowTMemLdStRawQueryRowPlanOverride`, and
  `disallowTMemLdStQueryTypeRescue`. These should collapse into type-local
  predicates plus optimizer-only performance rewrites.
- Exact-copy scheduling notes that depend on differences between standalone and
  exact chain-derived physical queries. Under the target model, copy legality
  comes from the current destination descriptor only.

### Phased Implementation Plan

1. Prototype and derisk before broad rewrites.
   - Hand-write small MLIR/LLVM probes for f32, packed/unpacked fp16, and
     packed/unpacked fp8/i8 physical-element-column origins.
   - Run them through the remaining lowering pipeline and inspect reduced
     IR/PTX to prove that f32 and statically aligned sub-32-bit cases do not
     leave live `subword_index` arithmetic.
   - Establish branch-local PTX/SASS and benchmark baselines for examples
     `01-attention-forward` and `05-fused-gather-bmm1`, plus representative
     ld/st, copy, `ld.red`, MMAv5, and scales rows.
   - Build a temporary type-local planner mirror and compare its decisions with
     the current planner across runtime-matrix shapes before switching codegen.

2. Fix TMEM view type inference and view lowering together.
   - Audit `memdesc_subslice`, `ttng.tmem_subslice`, `memdesc_index`,
     `memdesc_reshape`, `memdesc_trans`, and `memdesc_reinterpret`.
   - Ensure each result type's shape/layout/storage facts exactly describe the
     resulting descriptor relative to its own current `taddr`.
   - Ensure origin-changing views advance the lowered `taddr` from the source
     type/layout and explicit/dynamic offsets only.
   - Remove the need for `TMemLdStQueryLayout.origin` in use-site lowering.

3. Introduce a type-local TMEM descriptor planner.
   - Inputs: `MemDescType`, tensor-memory layout, operand/result types, and the
     semantic operation kind.
   - Outputs: relative physical-element-column layout, row plan, the set of
     legal atom/packet plans, packet offsets, physical element-column stride,
     subword packing/unpacking requirements, and clean hardware-negative
     diagnostics.
   - Performance heuristics may choose among the legal plans and may use
     producer-chain peepholes for that choice, but they may not expand the legal
     set.
   - No `Value` input except optional debug naming.

4. Make view lowering source-type-only.
   - `memdesc_subslice` and `memdesc_index` lowering should compute the packed
     origin delta from the source `MemDescType` and explicit/dynamic offsets.
   - Remove `getTMemSubviewRelativeBaseOffset` and
     `getAlreadyAdjustedTMemSubviewBaseOffset`.

5. Replace ld/st lowering planning.
   - Remove `Value memDescValue` from `lowerTMemLdStFromTypes`.
   - Use the type-local descriptor planner for direct, reduction, and support
     variants.
   - If a support path is not derivable from the current descriptor layout,
     require an optimizer rewrite before lowering.

6. Replace copy and MMAv5 planning.
   - `ttng.tmem_copy` should use a destination physical query from the destination
     type only.
   - MMAv5 accumulator/LHS/scale address layouts should be derived from operand
     types and layouts only.

7. Split helper APIs.
   - Rename or move producer-chain matchers into optimizer-only utilities.
   - Keep `TensorMemoryUtils.h` lowering-facing APIs type-local.
   - Add assertions in LLVM conversion that no lowering planner calls a
     chain-walking API.

8. Add compositional tests.
   - `arith.select` between two same-typed TMEM views with different origins,
     followed by `ttng.tmem_load`, `ttng.tmem_store`, `ttng.tmem_copy`, `ld.red`,
     and MMAv5 where legal.
   - `scf.if` and `scf.for` carrying same-typed TMEM descriptors through block
     arguments.
   - Helper-function returned TMEM descriptors with no local producer chain at
     the consumer.
   - Negative tests where two branches have different descriptor layouts and
     therefore cannot join at the same memdesc type.

## Review Conclusions

- The runtime memdesc payload should remain one packed scalar address. Adding
  parent pointers, root descriptors, or side metadata is not necessary for the
  current model and would make control-flow joins harder.
- The layout column convention should be physical element slots. Codegen
  converts to hardware word columns and handles subword packing/unpacking at the
  instruction boundary.
- Raw `taddr` alone is not sufficient for codegen. The current descriptor type
  and layout must carry or derive all static layout and instruction-planning
  facts; no additional address-mode field is part of the target model.
- Valid lowering sets must be derived directly from the current type/layout and
  operation semantics. Producer-chain analysis is optimizer-only context for
  ranking or rewriting among already-valid choices.
- The branch currently still has substantial semantic dependence on producer
  chains for ld/st, ld.red, copy, MMAv5, verifier, and Gluon layout selection.
- The right direction is not to make parent-chain walking more robust. It is to
  make view ops update runtime `taddr` and result types exactly, then make all
  lowering-facing planners type-local.

## Implementation Notes

### 2026-04-22 First Subslice Type Slice

- Added a type-local physical-query scaffold:
  `inferTypeLocalTMemPhysicalQuery(MemDescType)`. It is currently used only for
  debug comparison under `TRITON_DEBUG_TMEM_QUERY=1`; default copy lowering is
  still unchanged.
- Added explicit physical element-column helpers:
  `getTMemElementsPerWord`, `getTMemSubwordIndex`, and
  `getTMemAddressColumns`. The DCE prototype showed unused/static-zero
  subword-index arithmetic is removed for f32 and aligned/unpacked cases.
- Changed `inferTMemSubsliceOpEncoding` to prefer active inferred
  descriptor-relative TMEM-linear layouts before preserving parent encodings.
  This fixes the immediate bug where a `128x256 -> 128x128` column subview's
  result type still described the parent-width physical layout.
- Updated lit expectations so handwritten subslice result types spell the
  active layout. Existing lowering already advances `taddr` with a source-type
  offset; the active result layout makes the SSA value self-contained for the
  common column-subview case.
- Remaining gap: copy selection still prefers the old exact producer-chain
  query in some cases, so debug output can show a non-zero exact origin even
  when the current type-local query composes. The next copy slice should switch
  legal planning to the type-local query where available and keep exact
  producer-chain logic as an optimizer/debug fallback only until retired.

### 2026-04-22 Active Subview Load/Store Query Slice

- The first active-layout subslice fix exposed a load-side weakness in the
  `warpx2::01_23` two-CTA copy rows. Copy planning selected the same physical
  destination query and `tcgen05.cp` family as the direct allocation, but
  `tmem.get_reg_layout()` selected a raw narrowed copy-layout register mapping
  for the subview. Direct allocation instead uses the canonical standalone
  load/store surrogate layout.
- Active subviews with self-contained result layouts now do two things:
  - `getColumnSubviewTMemLdStSupportQueryPlan` refuses to borrow the source
    support image when the result encoding already differs from the source
    encoding. That source borrowing is now reserved for legacy views that still
    preserve parent encodings.
  - `getTMemLdStQueryTypes` tries the standalone canonical load/store surrogate
    before raw parent-allocation query types when the current memdesc type is
    self-contained. The predicate is type-local: trailing shape differs from
    trailing alloc shape, and `getCanonicalTMemLinearEncoding(memTy)` proves the
    current layout matches the active shape.
- This is still an incremental bridge, not the final planner. The use site
  still accepts `Value` and still has chain-dependent fallbacks, but the fixed
  correctness path no longer needs to inspect the source descriptor to decide
  that the active subview has an active layout or a canonical load/store
  surrogate.
- Validation after this slice: required `make -j8`; exact
  `warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused selector
  `cp_no_scales and (indexed_view or linear_subslice_view or warpx2_candidate or 128x128)`
  `63 passed, 1560 deselected`; 4-GPU
  `cp_no_scales and not reports` split passed as group1 `54 passed, 4 skipped`,
  group2 `58 passed`, group3 `58 passed`, group4 `57 passed`; targeted lit set
  `TritonNvidiaGPU/ops.mlir`, `TritonNvidiaGPU/tmem_layouts.mlir`,
  `Conversion/tritongpu_to_llvm_blackwell.mlir`,
  `Analysis/test-buffer-region.mlir`, `TritonNvidiaGPU/invalid.mlir`, and
  `TritonGPU/invalid.mlir` passed `6/6`.

### 2026-04-22 Active Subview Copy Query Slice

- `selectTMemCopyPhysicalQuery` now keeps the type-local destination physical
  query in `TMemCopyPhysicalQuerySelection`, along with debug diagnostics and a
  `usedTypeLocal` flag. `TMEMCopyOp::verify` reports this under
  `TRITON_DEBUG_TMEM_QUERY=1`.
- Copy planning now selects the type-local destination query for active
  self-contained TMEM subviews only. The type-local predicate is the same one
  used by the load/store ordering bridge: trailing shape differs from trailing
  alloc shape, and `getCanonicalTMemLinearEncoding(memTy)` proves the current
  layout matches the active shape. For this descriptor class, the runtime
  `taddr` already carries the subview origin, so the selected copy query should
  be origin-zero and relative to the current descriptor.
- The selection is deliberately narrow. Direct roots, raw parent-allocation
  layouts, tile-selector rows, and legacy views whose result type still
  preserves parent encoding continue to use the existing standalone/exact
  selection. This preserves the broad `cp_no_scales` matrix while migrating the
  active-layout correctness path away from producer-chain exact queries.
- Validation after this slice: required `make -j8`; exact
  `warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused selector
  `cp_no_scales and (indexed_view or linear_subslice_view or warpx2_candidate or 128x128)`
  `63 passed, 1560 deselected`; 4-GPU
  `cp_no_scales and not reports` split passed as group1 `54 passed, 4 skipped`,
  group2 `58 passed`, group3 `58 passed`, group4 `57 passed`; targeted lit set
  `TritonNvidiaGPU/ops.mlir`, `TritonNvidiaGPU/tmem_layouts.mlir`,
  `Conversion/tritongpu_to_llvm_blackwell.mlir`,
  `Analysis/test-buffer-region.mlir`, `TritonNvidiaGPU/invalid.mlir`, and
  `TritonGPU/invalid.mlir` passed `6/6`; `git diff --check` passed.

### 2026-04-22 Too-Small Copy Clean-Negative Slice

- Added `getTMemCopyAtomFailureMessage(const LinearLayout &, int)` as the
  diagnostic counterpart to `getTMemCopyAtom`. It does not expand or change the
  recognized atom set. It only explains a true hardware boundary when the
  current source-to-destination projection has four rows with fewer than
  `256` logical column bits, or 128 rows with fewer than `128` logical column
  bits.
- `TMEMCopyOp::verify` and late LLVM copy lowering both attach this note when
  no copy family can be classified. The note reports the actual
  `columns x element-bitwidth` width and states that lowering cannot borrow
  hidden parent columns for a narrower current descriptor.
- New runtime coverage:
  `test_tmem_runtime_matrix_cp_no_scales_too_small_destination_reports_clean_error`
  covers `128x1xf32` and `128x2xf32` with a linear shared layout. This avoids
  the unrelated `NVMMASharedLayout` minimum-contiguous-width failure and proves
  the clean negative happens at `ttng.tmem_copy`.
- Validation after this slice: required `make -j8`; new exact clean-negative
  test `2 passed`; adjacent exact-width copy positive `2 passed`; focused copy
  selector including the new negatives `65 passed, 1560 deselected`; 4-GPU
  `cp_no_scales and not reports` split passed as group1
  `54 passed, 4 skipped`, group2 `58 passed`, group3 `58 passed`, group4
  `57 passed`; targeted lit set passed `6/6`; `git diff --check` passed.

### 2026-04-22 Active Subview Load/Store Row-Plan Slice

- Active self-contained TMEM subviews now use type-local row planning in the
  ld/st row-plan helpers. `getTMemLdStRowPlanForQuery`,
  `getTMemLdStRowPlanForQueryLayout`, and
  `getTMemLdStRowPlanForSupportQuery` return the current descriptor's row plan
  for this descriptor class and do not consult a parent backing plan.
- `getTMemLdStQueryTypes` uses `getTMemLdStRowPlanForType(memTy)` when building
  canonical surrogate types for active self-contained subviews. Legacy views
  still use `getBackingTMemLdStRowPlan` until their result types are made
  self-contained or an optimizer rewrite materializes an explicit supported
  descriptor.
- This is a row-planning slice, not the final ld/st planner. Verifier and LLVM
  lowering still call value-taking query helpers; the next step is to introduce
  an explicit type-local ld/st query-plan API and route one active descriptor
  consumer path through it.
- Validation after this slice: required `make -j8`; exact
  `warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused ld/st
  selector `78 passed, 1547 deselected`; 4-GPU
  `ldst and not reports and not scales` split passed as group1 `88 passed`,
  group2 `32 passed, 56 skipped`, group3 `66 passed, 22 skipped`, group4
  `67 passed, 20 skipped`; targeted lit set passed `6/6`; `git diff --check`
  passed.

### 2026-04-23 Type-Local B-Scale Descriptor-View Storage Slice

- Generated MMAv5 B-scale descriptor views are not currently represented as
  `tensor_memory_scales` result types after reshape/trans/index composition;
  they are `tensor_memory_linear` descriptors whose layout still encodes the
  B-scale storage pattern.
- A value selected through `arith.select`, yielded from a block, or otherwise
  forwarded dynamically cannot rely on a visible reshape/trans/index producer
  chain to recover the root scales encoding. This is exactly the semantic gap
  the new memdesc model is removing.
- Added `getMMAv5ScaledBScaleStorageType(MemDescType)`. It first accepts
  current `tensor_memory_scales` descriptors, then conservatively recognizes
  generated padded and unpadded B-scale storage layouts from a rank-2 8-bit
  `tensor_memory_linear` type by checking exact row/column basis sequences
  after normalized TMEM-view layout analysis.
- `getMMAv5ScaledBScaleStorageTypeThroughViews(Value)` now tries this type-local
  helper before the legacy producer-chain walk. The chain walk remains a
  migration fallback for descriptor classes whose result types are not yet
  self-contained.
- Added runtime coverage that stores B-scale data into two padded descriptor
  views, dynamically selects one same-typed view, and consumes it in scaled MMA.
  The test asserts runtime correctness, exact `tcgen05.mma` opcode count,
  dynamic-control-flow/select presence, and that TTGIR still contains the
  linear descriptor-view type.
- Boundary: an unpadded dynamically selected B-scale descriptor view still
  exposes a separate allocation/rematerialization transform gap. The semantic
  lowering legality can now be classified from the current type, but the
  transform stack must still learn how to rematerialize or preserve unpadded
  B-scale storage through dynamic forwarding before that variant becomes a
  runtime positive.
- Validation after this slice: required `make -j8`; focused
  `bscale_descriptor_view` selector `5 passed, 1621 deselected`; targeted lit
  set previously passed `6/6`; 4-GPU positive MMAv5 selector previously passed
  as group1 `134 passed, 14 skipped`, group2 `148 passed`, group3 `148 passed`,
  group4 `145 passed`; `git diff --check` passed.

### 2026-04-23 Selected B-Scale Rematerialization Slice

- The type-local B-scale storage classifier made selected unpadded descriptor
  views semantically classifiable, but the allocation transform still only knew
  how to rematerialize a single direct store/view chain. A selected value had
  multiple root allocations, so the direct rematerialization path declined and
  LLVM conversion later rejected the unsupported unpadded storage.
- `RematerializeScaledMmaBScaleFragments` now separates direct branch
  rematerialization from top-level MMA operand rewriting. For an `arith.select`
  B-scale result, it rematerializes the true and false operands independently
  into the required padded `tensor_memory_scales` shape, creates a new selected
  padded descriptor, and rewrites the MMA operand to that descriptor.
- If the original select is single-use, the pass erases the obsolete select and
  then runs the normal store/view cleanup for the original unpadded branch
  chains. If it is multi-use, the original selected descriptor is left intact
  for unrelated consumers and only the MMA receives the new padded selected
  descriptor. Broader control-flow rematerialization can build on the same
  branch-local helper once there is a concrete `scf.if`/loop-carried
  reproducer.
- The runtime matrix now parameterizes the dynamic B-scale descriptor-view test
  over padded and unpadded storage. The unpadded row would previously compile
  to an unsupported selected `tensor_memory_linear` B-scale view at LLVM
  conversion; it now passes runtime correctness and exact opcode checks.
- Validation after this slice: required `make -j8`; exact unpadded dynamic
  selected row `1 passed`; padded+unpadded dynamic rows `2 passed`; focused
  `bscale_descriptor_view` selector `6 passed, 1621 deselected`; targeted lit
  set `6/6`; 4-GPU positive MMAv5 selector passed as group1
  `134 passed, 14 skipped`, group2 `148 passed`, group3 `148 passed`, group4
  `146 passed`; `git diff --check` passed.

### 2026-04-23 General Selected Scale Descriptor-View Slice

- General A/B scaled-MMAv5 scale descriptor views had the same semantic
  producer-chain dependency as B-scale storage: the current value was a
  `tensor_memory_linear` memdesc whose layout described generated scales
  storage, while the allocation transform recovered the root
  `tensor_memory_scales` encoding by walking reshape/trans producers.
- Added `getMMAv5ScaleStorageType(MemDescType)`, a type-local classifier for
  generated unpadded interleaved scale descriptor-view storage. The B-scale
  classifier now reuses the same unpadded-layout predicate and keeps its
  additional padded-storage predicate for B-scale fragment requirements.
- `getMMAv5ScaleStorageTypeThroughViews(Value)` now tries the type-local helper
  before falling back to legacy root walking, so semantic rematerialization does
  not require a visible root view chain for the covered layout family.
- The general `RematerializeScaledMmaScaleDescriptorViews` pattern now mirrors
  the selected B-scale flow: it rematerializes selected true/false scale
  descriptors branch-wise into direct scales allocations, creates a selected
  rematerialized descriptor for the MMA, cleans up single-use originals, and
  leaves multi-use originals intact for unrelated consumers.
- Added dynamic A/B scale descriptor-view runtime rows. Both rows select between
  same-typed linear descriptor views and consume the selected scale in scaled
  MMA; both now pass runtime correctness and exact opcode checks.
- Validation after this slice: required `make -j8`; dynamic A/B scale
  descriptor-view rows `2 passed`; combined
  `scale_descriptor_view or bscale_descriptor_view` selector
  `12 passed, 1617 deselected`; targeted lit set `6/6`; 4-GPU positive MMAv5
  selector passed as group1 `134 passed, 14 skipped`, group2 `148 passed`,
  group3 `148 passed`, group4 `148 passed`; `git diff --check` passed.

### 2026-04-23 Selected Scales Descriptor-View Load/Store Slice

- The same current-type-only rule applies to scales load/store, not only scaled
  MMA allocation rewrites. A dynamically selected row-permuted scales descriptor
  view has a `tensor_memory_linear` current type, but its layout is enough to
  classify generated scales storage. The previous ld/st path failed exact
  support/raw lowering and then fell back to a canonical dense-linear query
  type, which changed the physical access pattern for the selected value.
- Added type-local scales encoding helpers in `TensorMemoryUtils.cpp` for
  ld/st diagnostics and support planning. Generic generated scale
  descriptor-view classification now accepts two-CTA layouts; the
  B-scale-specific scaled-MMA padded-storage classifier keeps the narrower
  one-CTA rule.
- `getTMemLdStQueryTypes(Value)` now returns the recovered
  `tensor_memory_scales` storage type for type-local scales descriptor views
  before the original linear view type, and avoids inventing the unsafe
  dense-linear surrogate for that descriptor class. LLVM ld/st lowering uses
  the recovered storage type as the planning memdesc for raw/support attempts
  while keeping the selected descriptor's runtime `taddr`.
- Added dynamic selected scales descriptor-view load/store runtime coverage for
  1CTA and 2CTA. The test selects between two same-typed row-permuted views,
  stores through the selected view, loads the selected root, and asserts exact
  tcgen05 opcode sequences.
- Validation after this slice: required `make -j8`; exact dynamic selected
  scales ld/st rows `2 passed`; focused `ldst_scales` selector
  `35 passed, 1596 deselected`; combined
  `scale_descriptor_view or bscale_descriptor_view` selector
  `12 passed, 1619 deselected`; targeted lit set `6/6`; 4-GPU
  `ldst_scales or scale_descriptor_view or bscale_descriptor_view` split
  passed as group1 `12 passed`, group2 `12 passed`, group3 `12 passed`, group4
  `11 passed`; `git diff --check` passed.

### 2026-04-23 Type-Local MMAv5 Family Address Slice

- `getMMAv5TMemFamilyAddressLayout(MemDescType)` no longer rejects narrowed
  `shape != allocShape` descriptors. The current descriptor type's allocation
  shape is enough for the existing MMAv5 family-info helpers to recover the
  address layout for row-preserving N-narrowed accumulator and scaled
  accumulator subviews.
- `getMMAv5TMemAddressLayout` consults that type-local family address layout
  before entering legacy producer-chain support-layout inference.
  `getMMAv5TMemViewOffsetForLowering` computes tile-order offsets in the same
  family coordinate frame before falling back to chain-derived view offsets.
- A deliberately overbroad local experiment failed correctness when direct
  ld/st used the raw tile-permuted narrowed current layout as the support query.
  The durable rule is operation-specific: MMAv5 address/tile-order may use the
  family address layout, while direct ld/st support must still choose the
  canonical support family derivable from current descriptor family facts.
- Validation after this slice: required `make -j8`; full scaled tile-permuted
  accumulator subslice function `10 passed`; exact two-CTA scaled subslice row
  `1 passed`; 4-GPU positive MMAv5 selector passed as group1
  `133 passed, 14 skipped`, group2 `147 passed`, group3 `147 passed`, group4
  `147 passed`; 4-GPU combined ld/st+ld.red+copy selector passed as group1
  `120 passed, 28 skipped`, group2 `98 passed, 50 skipped`, group3
  `128 passed, 20 skipped`, group4 `146 passed`; targeted lit set passed
  `6/6`; `git diff --check` passed.

### 2026-04-23 MMAv5 Address Helper Split

- Added `getTypeLocalMMAv5TMemAddressLayout(MemDescType)` for the semantic
  type-local MMAv5 address plan. It covers active self-contained descriptors,
  MMAv5-family descriptors, and direct full-shape descriptors without accepting
  a `Value`.
- Added `getTypeLocalMMAv5TMemViewOffsetForLowering(MemDescType, offsets)` for
  the matching type-local tile-order offset computation.
- The existing value-taking helpers now act as compatibility wrappers:
  physical-bitcast behavior is preserved first, then the type-local helper is
  used, and only then does legacy producer-chain inference run.
- Validation after this slice: required `make -j8`; full scaled tile-permuted
  accumulator subslice function `10 passed`; focused B-scale descriptor-view
  rows `3 passed`; 4-GPU positive MMAv5 selector passed as group1
  `133 passed, 14 skipped`, group2 `147 passed`, group3 `147 passed`, group4
  `147 passed`; targeted lit set passed `6/6`; `git diff --check` passed.

### 2026-04-22 Active Subview Load/Store Query-Type Slice

- Added `getTypeLocalTMemLdStQueryTypes(MemDescType)`. It derives candidate
  ld/st query types directly from the current descriptor type: the canonical
  surrogate from the current type-local row plan, followed by the current
  descriptor type. Duplicate types are filtered while preserving that order.
- `getTMemLdStQueryTypes(Value)` now returns the type-local candidate list
  immediately for active self-contained TMEM subviews. This removes another
  semantic lowering dependency on parent/view-chain reconstruction for the
  descriptor class being migrated to the new memdesc model.
- Legacy descriptors still use the old value-chain path. That is intentional
  during the incremental migration: existing behavior remains stable until
  their view ops compute fully self-contained result types or an optimizer
  rewrite materializes an explicit supported descriptor.
- Validation after this slice: required `make -j8`; exact
  `warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused ld/st
  selector `78 passed, 1547 deselected`; 4-GPU
  `ldst and not reports and not scales` split passed as group1 `88 passed`,
  group2 `32 passed, 56 skipped`, group3 `66 passed, 22 skipped`, group4
  `67 passed, 20 skipped`; targeted lit set passed `6/6`.

### 2026-04-23 Active Subview Load/Store Support-Query Slice

- Added `getTypeLocalTMemLdStSupportQueryPlan(MemDescType)`. For an active
  self-contained descriptor, support-query facts are now the same local facts
  that valid lowering is allowed to use: the current descriptor's type-local
  query layout and a row plan derived from that type/query layout.
- `getTMemLdStSupportQueryPlan(Value)` dispatches to the type-local helper
  before the legacy support-image reconstruction stack. This keeps support-query
  consumers, including normal ld/st verification and load-reduction planning,
  from making active-descriptor legality depend on producer-chain context.
- Legacy descriptor views still use the existing direct half-row, higher-rank
  half-row, outer-index, column-subview, and generic support-query helpers until
  their result types are made self-contained or are rewritten by an optimizer.
- Validation after this slice: required `make -j8`; exact
  `warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused ld/st
  selector `78 passed, 1547 deselected`; focused
  `ld_red and not reports and not scales` selector `239 passed, 1386 deselected`;
  4-GPU `(ldst or ld_red) and not reports and not scales` split passed as
  group1 `120 passed, 28 skipped`, group2 `98 passed, 50 skipped`, group3
  `128 passed, 20 skipped`, group4 `146 passed`; targeted lit set passed `6/6`.

### 2026-04-23 Active Subview LLVM Lowering Locality

- Exported `hasSelfContainedTMemSubviewLayout(MemDescType)` so LLVM conversion
  can share the same active-layout predicate as IR planning.
- `lowerTMemLdStFromTypes` now treats active self-contained descriptors as
  already relative to their current runtime `taddr`: raw-query lowering keeps
  the current memdesc type, raw/support row-plan fallback does not borrow
  backing rows, source-column support rescue is skipped, and base offsets are
  not passed through `getTMemSubviewRelativeBaseOffset`.
- `copySharedToTmem` now skips `getTMemSubviewRelativeBaseOffset` when
  `selectTMemCopyPhysicalQuery` selected the type-local destination query.
- Validation after this slice: required `make -j8`; exact
  `warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused ld/st
  selector `78 passed`; focused `ld_red and not reports and not scales`
  selector `239 passed`; focused copy selector `57 passed`; 4-GPU combined
  `(ldst or ld_red or cp_no_scales) and not reports and not scales` split
  passed as group1 `120 passed, 28 skipped`, group2 `98 passed, 50 skipped`,
  group3 `128 passed, 20 skipped`, group4 `146 passed`; dedicated 4-GPU
  `cp_no_scales and not reports` split passed as group1
  `54 passed, 4 skipped`, group2 `58 passed`, group3 `58 passed`, group4
  `57 passed`; targeted lit set passed `6/6`.

### 2026-04-23 Active Subview Verifier Locality

- `verifyTMEMOperand` skips standalone view-type physical-support rescue for
  active self-contained descriptors. That rescue can still exist for legacy
  descriptor views, but it no longer defines validity for active layouts whose
  current type should be sufficient.
- `TMEMLoadOp::verify` uses the active-layout predicate to avoid
  `getBackingTMemLdStRowPlan` fallback in support/raw ld.red row-plan checks.
- Validation after this slice: required `make -j8`; exact
  `warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused ld/st
  selector `78 passed`; focused `ld_red and not reports and not scales`
  selector `239 passed`; 4-GPU
  `(ldst or ld_red) and not reports and not scales` split passed as group1
  `120 passed, 28 skipped`, group2 `98 passed, 50 skipped`, group3
  `128 passed, 20 skipped`, group4 `146 passed`; targeted lit set passed `6/6`.

### 2026-04-23 Active Subview Row-Plan Fallback Tightening

- Active self-contained descriptors now reject missing type-local row-plan facts
  locally instead of borrowing a wider hidden parent row plan from
  `getBackingTMemLdStRowPlan`.
- `getTMemLdStRowPlanForRawQuery` skips backing-row fallback for active
  self-contained descriptors and falls back only to the query layout itself.
  `getTMemLdStDirectSupportTensorType` uses the current descriptor type's row
  plan as its fallback plan for this class. `getTMemLoadReductionLayoutForMemDesc`
  applies the same rule in raw-query, support-query, and query-type reduction
  paths.
- Legacy descriptor views continue to use backing-row fallback during the
  incremental migration.
- Validation after this slice: required `make -j8`; exact
  `warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused ld/st
  selector `78 passed, 1547 deselected`; focused
  `ld_red and not reports and not scales` selector `239 passed, 1386 deselected`;
  4-GPU `(ldst or ld_red) and not reports and not scales` split passed as
  group1 `120 passed, 28 skipped`, group2 `98 passed, 50 skipped`, group3
  `128 passed, 20 skipped`, group4 `146 passed`; targeted lit set passed `6/6`.

### 2026-04-22 Active Subview Load/Store Raw-Query Slice

- Added `inferTypeLocalTMemLdStQueryLayout(MemDescType)` as the ld/st analogue
  of the type-local physical-query scaffold. It derives the query layout from
  the current descriptor type and tensor-memory encoding, sets the query origin
  to zero, and canonicalizes out-dim names to preserve the existing
  `TMemLdStQueryLayout` contract.
- `inferStandaloneTMemLdStQueryLayout(Value, ...)` now dispatches to the
  type-local helper for active self-contained subviews. This keeps call sites
  stable while removing producer-chain reconstruction for that descriptor
  class. Legacy views continue through the old wrapper path until their result
  types are self-contained or an optimizer rewrite materializes a supported
  descriptor.
- Validation after this slice: required `make -j8`; exact
  `warpx2_01_23_twocta_subslice_view_positive` `4 passed`; focused ld/st
  selector `78 passed, 1547 deselected`; 4-GPU
  `ldst and not reports and not scales` split passed as group1 `88 passed`,
  group2 `32 passed, 56 skipped`, group3 `66 passed, 22 skipped`, group4
  `67 passed, 20 skipped`; targeted lit set passed `6/6`; `git diff --check`
  passed.
