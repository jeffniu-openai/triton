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


### 2026-04-24 Audit Classification: Semantic vs Optimizer Chain Walking

Audit point: branch `codex/tmem` at `bd3ebcbb1f1942058d1c79eac67c61427f216e17`, compared against upstream merge-base `37c9a4b569a0f1719bdd77ca5f666096e10b7227`. The agreed separation is:

- High-risk semantic/codegen uses are disallowed. These are any verifier, frontend layout-selection, or LLVM/PTX lowering paths where legality, query family, row plan, instruction atom, base offset, subword phase path, reduction eligibility, or copy schedule changes because a producer chain is visible.
- Optimizer-only uses are allowed when they run before lowering and rewrite IR, compute allocation/liveness/rematerialization facts, or choose among lowerings already proven legal from the current memdesc type/layout and immediate SSA value. They must not expand the semantic legal set.

High-risk open surfaces found in the audit:

1. `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` TMEM load/store verification still calls `isUnsupportedDirectTMemLdStDescriptorView(Value)`, `getTMemLdStQueryTypes(Value)`, `inferStandaloneTMemLdStQueryLayout(Value)`, `getTMemLdStSupportQueryPlan(Value)`, and `inferStandaloneTMemViewType(Value)` to accept or reject register layouts. This is semantic legality.
2. `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp` TMEM load/store lowering still selects raw/support/query lowering, row plans, phase-aware codegen, and adjusted base offsets from value-shaped producer-chain helpers. This directly changes emitted PTX/LLVM behavior.
3. `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp` subword phase and alignment helpers still inspect forwarding, CFG predecessors, `arith.select`, `scf.if`, `scf.for`, and view producers. These helpers feed `ld/st`, `ld.red`, `tcgen05.copy`, verifier, and Gluon gating. In the target model, phase/alignment facts must come from current memdesc SSA contents or be unknown and rejected where required.
4. `tcgen05.copy` verification/lowering still goes through `selectTMemCopyPhysicalQuery(Value, ...)`, which can choose type-local, standalone, or exact physical queries by reconstructing visible view history. Copy legality, family selection, and destination base offset must be type-local or rejected.
5. `python/src/gluon_ir.cc` still chooses frontend-visible TMEM register layouts and reduction eligibility with value-shaped standalone/support/raw query helpers. This is codegen behavior because it changes inserted register layouts before verifier/lowering.
6. Relative-base-offset helpers such as `getTMemViewOffsetForLowering(Value, ...)`, `getTMemSubviewOffsetForLowering(...)`, `getAlreadyAdjustedTMemSubviewBaseOffset(Value)`, and `getTMemSubviewRelativeBaseOffset(Value, ...)` are symptoms of split state. View ops must update the runtime memdesc value; use-site lowering must not subtract producer-chain-derived offsets.
7. Public value-shaped compatibility helpers are still reachable from semantic paths. They should be renamed/quarantined into optimizer/debug namespaces or removed from verifier/lowering/frontend call graphs once type-local replacements exist.

Optimizer-only or allowed-with-quarantine surfaces:

- `lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp` may inspect view chains for allocation aliasing, B-scale/A-scale rematerialization, branch splitting through `arith.select`, and cleanup of pre-lowering IR. This is allowed because it rewrites IR before lowering rather than defining final ISA legality.
- `lib/Dialect/TritonNvidiaGPU/Transforms/InterleaveTMem.cpp` may inspect view chains for local access-range/alias analysis. This is allocation/scheduling analysis, not TMEM instruction legality.
- `lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp` may use producer-chain peepholes for split/join/half-slice rewrites when the resulting IR is independently legal. Its physical-support load/store rewrite that calls `inferStandaloneTMemViewType(Value)` remains suspicious and should either be proved to rewrite only already-local-fact-legal accesses or be removed/converted to a type-local rewrite.

Closure rule from this audit: future fixes should first disconnect semantic paths from value-shaped helpers, then leave producer-chain inspection only in clearly named optimizer/rematerialization/debug helpers. Clean negatives are preferred over recovering hidden facts from a producer chain.

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
  lowering for origin-changing views. Pointwise view-offset inversion is now
  type/local for static offsets: lowering solves the requested logical offset
  against the current `MemDescType`/layout instead of requiring a global
  layout pseudoinverse.
- [ ] Migrate ld/st planning to type-local analysis.
  First row-plan slice complete: active self-contained subviews use the current
  descriptor's row plan instead of parent backing-row plans. First raw-query
  slice complete: active self-contained subviews dispatch through
  `inferTypeLocalTMemLdStQueryLayout(MemDescType)`.
- [ ] Migrate subword pack/unpack handling for packed and unpacked sub-32-bit
  layouts.
- [ ] Migrate `ld.red` legality/layout selection to the type-local planner.
- [x] Migrate `tcgen05.copy` planning to destination-type-only analysis and add
  too-small-copy clean negatives. First partial slice complete: active
  self-contained subview destinations select a type-local physical query.
  Matched-projection slice complete: other descriptors now also select the
  type-local destination query when it matches the physical projection chosen by
  the legacy standalone/exact selector. First clean-negative slice complete:
  `128x1xf32` and `128x2xf32` current descriptors report an explicit hardware
  copy-atom boundary. Closeout slice complete: active self-contained subviews
  and scales descriptor views now use a copy-specific type-local physical query
  before legacy standalone/exact comparison; direct roots keep the support-query
  path because raw type-local direct-root planning regressed 256-row dense and
  tile-selector layouts during validation.
- [ ] Migrate MMAv5/scales address planning to current type/layout plus runtime
  `taddr`. First partial slice complete: narrowed MMAv5-family descriptors now
  derive address layout and tile-order offsets from current `MemDescType`
  family facts before legacy producer-chain fallback. First helper-split slice
  complete: type-local MMAv5 address and tile-order computations now have
  explicit APIs separate from value-taking fallback wrappers. Physical-bitcast
  MMAv5 address and tile-order lowering now no longer inspect the defining
  reinterpret op; they rely on the current type/layout and rescaled runtime
  `taddr`.
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

### 2026-04-23 Dynamic Selected Descriptor-Chain LD.Red Coverage

- Added a runtime positive for a selected descriptor-chain ld.red consumer.
  The kernel initializes two TMEM descriptor chains with distinct f32 payloads,
  selects between same-typed descriptor views with dynamic control flow, and
  performs `load_max` through the selected descriptor.
- This is an important sentinel for the memdesc model because the selected SSA
  value has no single producer chain that lowering can legally inspect. The
  current `MemDescType` plus selected runtime `taddr` must be sufficient.
- The representative row already passed with the current type-local ld/st and
  ld.red planning. No backend change was required; only the test helper was
  generalized so rows with two producer stores can expect two store waits while
  preserving the existing default of one wait.
- Validation after this slice: required `make -j8`; exact new row `1 passed`;
  4-GPU `ld_red and not reports and not scales` split passed as group1
  `60 passed`, group2 `60 passed`, group3 `60 passed`, group4 `60 passed`;
  `git diff --check` passed.

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

### 2026-04-23 Dynamic Selected Copy Sentinel

- Added a runtime sentinel for `tcgen05.copy` over a selected same-parent column
  subview. The kernel selects between two same-typed `128x128xf32` views of a
  `128x256xf32` TMEM allocation, copies shared memory into the selected
  descriptor, and then loads through the selected descriptor.
- This validates the intended split for copy lowering: the legal copy family is
  derived from the current self-contained descriptor type/layout, while the
  emitted destination address uses the selected memdesc SSA value's runtime
  `taddr`. The selector-1 row would fail if lowering recovered a fixed
  producer-chain origin instead of using the selected value.
- No backend change was required for this sentinel. The existing type-local
  copy physical-query path already handles this representative selected-copy
  case.
- Validation after this slice: required `make -j8`; exact new row `2 passed`;
  adjacent copy subview/indexed selector `23 passed, 1611 deselected`; 4-GPU
  `cp_no_scales and not reports` split passed as group1
  `55 passed, 4 skipped`, group2 `59 passed`, group3 `59 passed`, group4
  `56 passed`.

### 2026-04-23 Dynamic Selected Warpx2 Copy Sentinel

- Added runtime sentinels for selected same-parent `warpx2` copy column
  subviews. The kernel selects between two same-typed `128x4xf32` views of a
  `128x8xf32` parent, copies shared memory into the selected descriptor, and
  then loads through the selected descriptor.
- The test covers both non-dense copy schedules,
  `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b` and
  `tcgen05.cp.cta_group::1.warpx2::02_13.64x128b`, including their
  schedule-specific output transforms. This checks the selected runtime-`taddr`
  contract on copy families whose legal instruction schedule is more involved
  than dense `128x256b`.
- No backend change was required. The existing type-local copy physical-query
  path already handles these representative non-dense selected-copy rows.
- Validation after this slice: required `make -j8`; exact new rows `4 passed`;
  adjacent `warpx2` subview/indexed selector `40 passed, 1598 deselected`;
  4-GPU `cp_no_scales and not reports` split passed as group1
  `56 passed, 4 skipped`, group2 `60 passed`, group3 `60 passed`, group4
  `57 passed`.

### 2026-04-23 Dynamic Selected 2CTA Copy Sentinel

- Added a runtime sentinel for selected same-parent two-CTA dense copy column
  subviews. The kernel selects between two same-typed `256x128xf32` views of a
  `256x256xf32` two-CTA parent, copies shared memory into the selected
  descriptor, and then loads through the selected descriptor.
- The test checks the `tcgen05.cp.cta_group::2.128x256b` path, including
  cluster fence/barrier sequencing and multicast commit. This extends the
  selected runtime-`taddr` contract to hardware CTA ownership cases.
- No backend change was required. The existing type-local copy physical-query
  path already handles this representative 2CTA selected-copy row.
- Validation after this slice: required `make -j8`; exact new rows `2 passed`;
  adjacent 2CTA copy subview/indexed selector `23 passed, 1617 deselected`;
  4-GPU `cp_no_scales and not reports` split passed as group1
  `56 passed, 4 skipped`, group2 `60 passed`, group3 `60 passed`, group4
  `59 passed`.

### 2026-04-23 Active Copy Helper Locality

- `selectTMemCopyPhysicalQuery` now computes the type-local destination
  physical query first. For active self-contained TMEM subviews, it chooses that
  query or returns its local copy-conversion error before attempting
  standalone/exact producer-chain reconstruction.
- This turns the migrated copy path into a true type-local semantic path: the
  dense, `warpx2`, and 2CTA selected-copy sentinels no longer rely on producer
  provenance even as an intermediate planner input. Legacy descriptor classes
  still use the existing standalone/exact fallback ordering after this gate.
- Validation after this slice: required `make -j8`; combined selected-copy rows
  `8 passed`; 4-GPU `cp_no_scales and not reports` split passed as group1
  `56 passed, 4 skipped`, group2 `60 passed`, group3 `60 passed`, group4
  `59 passed`; targeted lit set passed `6/6`; `git diff --check` passed.

### 2026-04-23 Dynamic Selected Normal Ld/St Sentinel

- Added runtime coverage for normal `ttng.tmem_store`/`ttng.tmem_load` over a
  selected active column subview. The kernel initializes two same-typed
  `128x128xf32` views of a `128x256xf32` parent with distinct values, stores
  through the selected descriptor, and then loads both explicit candidate
  views.
- This is intentionally stronger than loading only through the selected
  descriptor: if selected store and selected load both ignored the selected
  runtime `taddr`, a simple roundtrip could pass. Checking both candidates
  proves the selected store writes the selected runtime base while the explicit
  view loads read their own bases.
- No backend change was required. Existing type-local ld/st query, support,
  row-plan, and LLVM lowering paths already handle this representative selected
  active subview.
- Validation after this slice: required `make -j8`; exact new rows `2 passed`;
  adjacent ld/st selector `40 passed, 1602 deselected`; 4-GPU
  `ldst and not reports and not scales` split passed as group1 `89 passed`,
  group2 `33 passed, 56 skipped`, group3 `67 passed, 22 skipped`, group4
  `66 passed, 20 skipped`.

### 2026-04-23 Dynamic Selected Active-Subview Ld.Red Sentinel

- Added runtime coverage for hardware row reduction through a selected active
  column subview. The kernel initializes two same-typed `128x128xf32` views of
  one `128x256xf32` parent allocation with distinct values, dynamically
  selects one view, and performs `load_max` through the selected descriptor.
- This extends the selected-value contract from normal ld/st and copy to
  `tcgen05.ld.red`. The selector-1 row catches lowering that ignores the
  selected memdesc SSA value's current `taddr` and instead recovers the first
  visible producer-chain subview origin.
- No backend change was required. Existing type-local ld/st/ld.red planning
  already handles this representative selected active subview.
- Validation after this slice: required `make -j8`; exact new rows `2 passed`;
  adjacent ld.red selector `53 passed, 1591 deselected`; 4-GPU
  `ld_red and not reports and not scales` split passed as group1 `61 passed`,
  group2 `61 passed`, group3 `61 passed`, group4 `59 passed`.

### 2026-04-23 Loop-Carried Active-Subview Ld/St Sentinel

- Added runtime coverage for normal ld/st after a selected active column
  subview flows through an `scf.for` result. The kernel initializes two
  same-typed `128x128xf32` views of one `128x256xf32` parent, carries the
  selected descriptor through a loop, then calls `get_reg_layout()`, loads from
  the selected descriptor, stores through it, and reads back both explicit
  candidate views.
- This covers the no-local-producer use shape required by the new memdesc
  model. The consumer only sees the current memdesc SSA value and type/layout;
  it cannot rely on walking directly to a visible subslice producer.
- No backend change was required. Existing active-subview type-local ld/st
  planning handles the loop-carried memdesc result.
- Validation after this slice: required `make -j8`; exact new rows `2 passed`;
  adjacent ld/st selector `42 passed, 1604 deselected`; 4-GPU
  `ldst and not reports and not scales` split passed as group1 `89 passed`,
  group2 `35 passed, 54 skipped`, group3 `65 passed, 24 skipped`, group4
  `68 passed, 20 skipped`.

### 2026-04-23 Loop-Carried Active-Subview Copy Sentinel

- Added runtime coverage for `tcgen05.copy` after a selected active column
  subview flows through an `scf.for` result. The kernel initializes both
  candidate `128x128xf32` subviews of one `128x256xf32` parent with distinct
  values, copies shared memory into the loop-carried selected descriptor, and
  reads back both explicit candidates.
- This is stronger than selected-copy loadback alone. If both copy and selected
  load ignored the selected runtime `taddr`, a selected roundtrip could pass;
  explicit candidate readback catches that class of mistake.
- No backend change was required. The active self-contained copy physical-query
  gate handles this no-local-producer selected value from current
  `MemDescType`/layout facts.
- Validation after this slice: required `make -j8`; exact new rows `2 passed`;
  adjacent copy selector `42 passed, 1606 deselected`; 4-GPU
  `cp_no_scales and not reports` split passed as group1
  `57 passed, 4 skipped`, group2 `61 passed`, group3 `61 passed`, group4
  `58 passed`.

### 2026-04-23 Active-Subview Row-Plan Helper Separation

- `getTMemLdStRowPlanForRawQuery` and
  `getTMemLdStRowPlanForSupportQuery` now detect active self-contained
  subviews first and return a row plan from the current descriptor type or
  support query layout before consulting value-taking override/backing-row
  helpers.
- This is a cleanup/generalization step rather than a behavior fix. The
  selected active-subview sentinels already passed, but this makes the intended
  semantic boundary explicit: migrated active subviews do not need
  producer-chain row-plan fallback.
- Validation after this slice: required `make -j8`; selected active-subview
  exact rows `10 passed`; focused active-subview selector
  `6 passed, 1642 deselected`; targeted lit set `6/6`.

### 2026-04-23 Query-Rescue Helper Separation

- `disallowTMemLdStQueryTypeRescue` now has a `MemDescType` overload. The
  verifier and LLVM lowering use that overload directly; the Value-taking
  wrapper remains only as a compatibility shim.
- The row-zero lifted rescue decision is therefore based on current descriptor
  type/layout facts instead of a visible `memdesc_reinterpret` producer. This
  matches the target rule that legality cannot depend on producer-chain
  visibility.
- Validation after this slice: required `make -j8`; targeted lit set `6/6`;
  focused reinterpret/subword/selected ld/st selector
  `10 passed, 1638 deselected`; selected ld.red/copy exact rows `4 passed`;
  4-GPU `(ldst or ld_red) and not reports and not scales` split passed as
  group1 `124 passed, 26 skipped`, group2 `98 passed, 52 skipped`, group3
  `130 passed, 20 skipped`, group4 `147 passed`.

### 2026-04-23 M64 Query-Ordering Helper Separation

- `shouldPreferTMemLdStQueryTypeLoweringBeforeRawQuery` now has a
  `MemDescType`/register-layout overload for active self-contained
  descriptors. The shared predicate accepts an already-derived type-local
  raw-query layout, so migrated active subviews do not need a producer-chain
  raw-query probe to decide M64 split-N query-type-vs-raw-query order.
- Legacy descriptor classes still use the Value wrapper and existing
  standalone raw-query inference. This keeps the change scoped to descriptors
  whose current type/layout already encode the semantic facts required by the
  new memdesc model.
- Validation after this slice: required `make -j8`; exact selected
  active-subview ld/st rows `4 passed`; M64 split-N ld/st cluster `20 passed`;
  targeted lit set `6/6`; `git diff --check` passed.

### 2026-04-23 Loop-Carried Active-Subview Ld.Red Sentinel

- Added runtime coverage for `tcgen05.ld.red` after a selected active column
  subview flows through an `scf.for` result. The kernel writes distinct values
  to both same-parent subviews, carries the selected descriptor through the
  loop, then computes `get_reg_layout()` and `load_max` from the loop-carried
  value.
- No backend change was required. This covers the reduction path for the same
  no-local-producer selected-value contract already covered by normal ld/st and
  copy.
- Validation after this slice: required `make -j8`; exact new rows `2 passed`;
  adjacent `ld_red and linear_subslice_view` selector
  `4 passed, 1646 deselected`.

### 2026-04-23 Gluon M64 Ordering Helper Separation

- `shouldPreferTMemLdStQueryTypeLayoutsBeforeRawQuery` now has a
  `MemDescType` overload. The Value overload delegates to it for active
  self-contained descriptors, and the Gluon register-layout picker calls the
  type-local overload directly for that descriptor class.
- This removes another frontend layout-selection dependency on visible
  producer-chain raw-query inference for migrated active subviews while keeping
  legacy view-chain descriptors on the compatibility path.
- Validation after this slice: required `make -j8`; selected active-subview
  ld/st rows `4 passed`; M64 split-N ld/st cluster `20 passed`; targeted lit
  set `6/6`.

### 2026-04-23 Ld.Red Active-Subview Helper Locality

- `getTMemLoadReductionLayoutForMemDesc` now treats active self-contained
  descriptors as view-like based on the current descriptor type/layout. A
  selected or loop-carried active subview therefore does not need a visible
  `memdesc_subslice` producer to enter the reduction-support path.
- The Gluon reduction support predicate no longer falls back to backing-row
  plans for active self-contained descriptors. That keeps the support decision
  local to the current descriptor class while preserving legacy compatibility
  for older descriptor-view chains.
- Validation after this slice: required `make -j8`; selected active-subview
  ld.red rows `4 passed, 1646 deselected`; descriptor-chain ld.red N-sweep
  `12 passed`; targeted lit set `6/6`.

### 2026-04-23 Unsupported-Direct Diagnostic Locality

- Direct ld/st unsupported-descriptor-view diagnostics now use
  `inferStandaloneTMemLdStQueryLayout`, not the raw producer-chain
  implementation, when building row-anchor layouts. For active
  self-contained descriptors this dispatches to the type-local query layout.
- Variant diagnostics and row-anchor fallback no longer borrow backing row
  plans for active self-contained descriptors. The legacy compatibility path
  remains for older descriptor views.
- Validation after this slice: required `make -j8`; scales ld/st
  descriptor-view and variant-report selector `18 passed`; unsupported ld/st
  selector `4 passed`; targeted lit set `6/6`.

### 2026-04-23 Broader Active/Scales Validation

- After the ld.red and unsupported-direct helper-locality slices, ran the
  active/scales runtime selector
  `(linear_subslice_view or ldst_scales or dynamic_scale_descriptor_view or
  dynamic_bscale_descriptor_view) and not reports` as four GPU split commands
  with separate caches.
- Result: group1 `18 passed`, group2 `18 passed`, group3 `18 passed`, group4
  `15 passed`.

### 2026-04-23 Physical-Bitcast Type Inference Locality

- `inferTMemBitcastType` now calls the public
  `inferStandaloneTMemLdStQueryLayout` wrapper. Active self-contained
  descriptors therefore use type-local query facts for physical bitcast result
  type inference instead of the raw producer-chain implementation.
- Added a runtime sentinel where a selected active column subview is physically
  bitcast from `f32[128,64]` to `f16[128,128]` before storing through the
  selected descriptor. Selector `0/1` rows verify that the selected runtime
  `taddr` controls which half of the parent TMEM allocation is zeroed.
- Full-slice index/reshape runtime-view bitcast cases now allow the frontend
  to fold no-op `memdesc_subslice` markers. This is an IR expectation update
  only; runtime correctness, physical-bitcast markers, and opcode checks remain.
- Validation after this slice: required `make -j8`; exact selected/preserved
  physical-bitcast rows `4 passed`; broad
  `tmem_linear_runtime_views or physical_bitcast` selector `15 passed`;
  targeted lit set `6/6`.

### 2026-04-23 Standalone View/Reg-Layout Query Type Locality

- `inferStandaloneTMemViewTypeImpl` now returns the active self-contained
  planning type for migrated active descriptors before checking for a visible
  descriptor-view producer. This prevents selected or loop-carried compact
  active subviews from falling through to the old shape-vs-alloc error.
- The change also covers `inferStandaloneTMemRegLayoutQueryType` and
  standalone physical-query fallback because those helpers share the same
  view-type inference.
- Validation after this slice: required `make -j8`; selected active-subview
  ld/st and ld.red rows `6 passed`; selected/broad physical-bitcast selector
  `17 passed`; targeted lit set `6/6`.

### 2026-04-23 Broad Non-Scale Runtime Validation

- Ran the non-scale ld/st, `ld.red`, and copy runtime selector
  `(ldst or ld_red or cp_no_scales) and not reports and not scales` as four GPU
  split commands with distinct caches.
- Result: group1 `124 passed, 26 skipped`, group2 `98 passed, 52 skipped`,
  group3 `130 passed, 20 skipped`, group4 `149 passed`.

### 2026-04-23 Broad MMAv5/Scales Runtime Validation

- Ran the MMAv5/scales runtime selector
  `mma and not reports and not clean and not unsupported` as four GPU split
  commands with distinct caches.
- Result: group1 `134 passed, 14 skipped`, group2 `148 passed`, group3
  `148 passed`, group4 `148 passed`.

### 2026-04-23 Scalar Query-Type Refinement Locality

- `refineTMemLdStQueryTypeEncodingInfo` now treats active self-contained
  descriptors as view-like based on the current `MemDescType`, rather than only
  when the SSA value has a visible view producer. This is a lowering-facing
  semantic cleanup: the selected current `taddr` remains the only runtime base,
  and the current descriptor layout determines whether the `32x32` query-type
  path should be refined to scalar packets.
- This closes the last active-subview use found in the final helper audit where
  "view-like" still meant "visible producer operation". Legacy descriptor views
  still use the producer-chain compatibility paths until their result types are
  made self-contained or an optimizer rewrite materializes a supported view.
- Validation after this slice: required `make -j8`; active ld/st
  `linear_subslice_view` selector `4 passed`; loop-carried active
  ld/st/ld.red/copy selector `6 passed`; `test_core.py`
  `physical_bitcast_selected_subview or tmem_linear_runtime_views` selector
  `12 passed`; lit `tmem_layouts.mlir` `1 passed` (the other requested tmem
  lit paths contained no tests in this checkout); broad `test_core.py` TMEM
  split passed as group1 `14 passed`, group2 `14 passed`, group3 `14 passed`,
  group4 `8 passed, 5 skipped`.

### 2026-04-23 Final Active-Semantics Audit Classification

- Lowering/verifier paths for active self-contained descriptors now dispatch to
  type-local facts for ld/st raw-query layout, query-type lists, row plans,
  support-query plans, copy physical-query selection, `ld.red` layout
  inference, physical bitcast type inference, M64 query ordering, row-zero
  query-rescue gating, unsupported-direct diagnostics, and scalar query-type
  refinement.
- Remaining producer-chain walkers in `TensorMemoryUtils.cpp` are classified as
  either legacy compatibility for older descriptor views that still preserve
  parent encodings, source/replay helpers for known clean-boundary diagnostic
  paths, or optimizer/allocation/effect analysis. They are not used to define
  active self-contained legality before the type-local path has had the chance
  to choose or reject the lowering.
- The next architectural cleanup remains splitting those compatibility helpers
  into clearly named optimizer-only or legacy-only utilities. That cleanup
  should be mechanical after the active semantic path is covered, but it should
  keep the same validation cadence because many legacy descriptor-view tests
  still exercise those fallbacks intentionally.

### 2026-04-23 Scalar Refinement API Split

- Added a `MemDescType` overload of
  `refineTMemLdStQueryTypeEncodingInfo`. Active self-contained lowering now
  calls this overload directly, so the scalar refinement path no longer needs a
  `Value` parameter for the migrated descriptor class.
- The Value-taking overload remains for legacy descriptor-view compatibility,
  where "view-like" still intentionally means a visible producer-chain view
  until those descriptors either compute self-contained result types or are
  rewritten by an optimizer.
- Validation after this slice: required `make -j8`; active ld/st
  `linear_subslice_view` selector `4 passed`; loop-carried active
  ld/st/ld.red/copy selector `6 passed`; lit `tmem_layouts.mlir` `1 passed`;
  unsplit `test_core.py -k 'tmem and (copy or ld or load or store or mma or
  tcgen05)'` `50 passed, 5 skipped`.

### 2026-04-23 Subword Active Selected-View Runtime Coverage

- The active selected-subview ld/st test kernels now allocate TMEM with the
  input element type and use typed initializer constants, so the same sentinel
  exercises f32, f16, and i8 descriptor values without changing the user-facing
  API.
- Added f16/i8 dynamic-select and loop-carried active column-subview runtime
  positives for ld/st. These rows verify both candidate views after storing
  through the selected descriptor and assert the expected subword
  `tcgen05.ld/st` packet shapes (`32x32b.x64.b32` for f16,
  `32x32b.x32.b32` for i8).
- Added f16/i8 dynamic-select and loop-carried active column-subview positives
  for dense `tcgen05.copy`. The checked dynamic copy sentinel initializes both
  candidate views, copies through the selected descriptor, and reads both
  candidates back; this prevents a selected-base lowering bug from being hidden
  by loading through the same selected SSA value.
- Validation after this slice: required `make -j8`; an initial pytest attempt
  without `PYTHONPATH=./python` failed during collection against the wrong
  installed Triton package and ran no tests. Rerun with `PYTHONPATH=./python`:
  `linear_subslice_view_subword` split passed as group1 `4 passed`, group2
  `4 passed`, group3 `4 passed`, group4 `4 passed`; adjacent non-subword
  active ld/st/copy selector passed as group1 `8 passed`, group2 `8 passed`,
  group3 `8 passed`, group4 `6 passed`; `git diff --check` passed.

### 2026-04-23 Unaligned Subword View Safety Guard

- A scratch packed-f16 active-view probe using `slice(1, 128)` showed the
  current backend still projects physical element columns to hardware word
  columns too early: physical element column 1 was lowered as hardware word
  column 0, so the offset view aliased the base view and produced wrong output.
- Added `getTMemViewPhysicalRowElementCol(MemDescType, offsets)` as a small
  bridge toward the intended abstraction. Existing `getTMemViewOffset` still
  returns the legacy hardware word-column packed offset, but its implementation
  now computes exact row/element-column coordinates first and projects to word
  columns at the final boundary.
- Until the runtime memdesc carries element-column phase and the ld/st/copy
  lowering consumes subword-index information, static subword
  `memdesc_subslice` inference and generic/NVIDIA lowering-only subview/index
  paths reject non-32-bit-column-aligned physical element origins cleanly. This
  prevents the silent aliasing miscompile while keeping aligned f16/i8 active
  view positives green.
- This is not a final clean-negative boundary. The design target remains
  element-column `taddr` plus subword-index lowering, with software
  pack/unpack/RMW where required and worthwhile. Future work should replace the
  guard for cases the ISA/software sequence can correctly realize.
- Validation after this slice: required `make -j8`; exact clean-negative row
  `1 passed`; split selector `linear_subslice_view_subword or
  ldst_unaligned_subword_linear_subslice_view` passed as group1 `5 passed`,
  group2 `5 passed`, group3 `5 passed`, group4 `2 passed`; adjacent
  non-subword active ld/st/copy selector passed as group1 `8 passed`, group2
  `8 passed`, group3 `8 passed`, group4 `6 passed`; direct subword ld/st/copy
  selector passed as group1 `5 passed`, group2 `5 passed`, group3 `5 passed`,
  group4 `5 passed`; lit `tmem_layouts.mlir` `1 passed`; `git diff --check`
  passed.

### 2026-04-23 Aligned Element-Column Runtime Slice

- Implemented the first runtime slice of the planned memdesc abstraction for
  aligned subword views. The lowered memdesc SSA value now carries a physical
  element-column `taddr`: `tcgen05.alloc` word-column results are converted to
  element columns, and generic/NVIDIA view/index/subslice lowering advances by
  element-column offsets computed from the current `MemDescType` layout.
- ISA emission is the projection boundary. Shared NVIDIA LLVM helpers convert
  element-column memdesc bases back to hardware word-column addresses for
  `tcgen05.ld/st`, `tcgen05.copy`, MMAv5 tmem operands/accumulators, and
  scaled-MMAv5 scale operands. This keeps semantic view lowering independent
  of producer chains while preserving the hardware address form required by
  PTX.
- Two validation findings tightened the abstraction:
  packed f16/i8 ld/st lowering cannot use the packed LLVM packet type to decide
  TMEM address projection, because software packs subword elements into b32
  registers before emission. `lowerTMemLdSt` now carries the memdesc element
  bitwidth separately from the LLVM packet type. Scaled-MMAv5 scale operands
  also needed the same element-to-word projection as other TMEM ISA operands.
- This slice does not remove the unaligned subword guard. Non-32-bit-column
  aligned physical element origins still require a runtime `subword_index` plus
  software pack/unpack/RMW path before they can be accepted as positives.
- Validation after this slice: required `make -j8`; split selector
  `linear_subslice_view_subword or
  ldst_unaligned_subword_linear_subslice_view` passed as group1 `5 passed`,
  group2 `5 passed`, group3 `5 passed`, group4 `2 passed`; direct subword
  ld/st/copy selector passed as group1 `5 passed`, group2 `5 passed`, group3
  `5 passed`, group4 `5 passed`; tmem-backed MMAv5 smoke rows `4 passed`; lit
  `tmem_layouts.mlir` `1 passed`; `git diff --check` passed.

### 2026-04-23 First Unaligned Packed-Subword Ld/St Slice

- Removed the temporary static odd-column rejection from
  `memdesc_subslice` type inference and from generic/NVIDIA TMEM view/subslice
  lowering. These operations now continue to update the runtime memdesc by
  element-column offsets, including origins that start inside a 32-bit hardware
  word.
- Added a transition helper, `TMemSubwordPhaseStatus`, that identifies visible
  TMEM values whose current `taddr` may carry a nonzero subword phase. It
  handles roots, static subviews/indexes, reshapes/transposes, same-bitwidth
  reinterprets, simple forwarding block arguments, and `arith.select` joins.
  This is not the final semantic contract; it is used to keep the first
  phase-aware lowering slice targeted while later consumers are migrated.
- Implemented the first executable `subword_index` consumer for packed
  contiguous `32x32b` ld/st plans. The lowering derives
  `phase = element_col % elements_per_word` from the current runtime `taddr`,
  projects the base to hardware word columns for PTX, loads one tail word for
  unaligned reads, and realigns packed b32 registers with dynamic shifts and
  ors. Stores read the first and tail hardware words, wait for those loads, and
  write shifted packed words plus a predicated tail word so lanes outside the
  logical view are preserved.
- `tcgen05.copy` now rejects known-nonzero subword destination origins locally.
  The copy ISA path has no RMW sequence in this slice, so accepting those
  descriptors would reintroduce floor-to-word aliasing.
- Added f16/i8 `slice(1, 128)` runtime positives that initialize neighboring
  aligned views, stores through the odd-column view, loads the odd-column view
  back, checks the left boundary through the aligned first view, and checks the
  right boundary through a second odd-column view starting at `N-1`. This
  covers both two-lane and four-lane packed word phases. Added a copy
  diagnostic row for the same odd-column destination.
- Remaining boundaries: non-packed or non-contiguous subword ld/st layouts,
  `ld.red`, copy, MMAv5, and unknown non-local subword phases still need
  follow-up support or local rejection. The intended end state remains local
  lowering from the current `taddr` plus current type/layout, not producer-chain
  dependence for validity.
- Validation after this slice: required `make -j8`; exact unaligned f16/i8
  ld/st runtime positives `2 passed`; exact unaligned f16 copy diagnostic row
  `1 passed`; focused adjacent subword ld/st/copy selector `32 passed,
  1636 deselected`; lit `tmem_layouts.mlir` `1 passed`; `git diff --check`
  passed.

### 2026-04-23 Subword Phase Control-Flow Widening

- Generalized phase-status value forwarding from Triton-only calls/functions to
  MLIR `FunctionOpInterface` and `CallOpInterface`, so phase facts survive
  helper-function boundaries introduced during lowering.
- Added semantic classification for `scf.if` results, `scf.for` results and
  iter args, and lowered CFG block arguments using `BranchOpInterface`
  successor operands. Cyclic carried values are treated as zero contribution
  unless an incoming value or recognized view transform introduces nonzero
  phase; unknown operations still classify as unknown.
- Ld/st now uses the packed contiguous phase-aware `32x32b` lowering whenever a
  subword descriptor is not proven phase-zero. This prevents unknown carried
  values from taking the old hardware-word floor path. Known-aligned selected
  and loop-carried values still lower to the original non-RMW messages.
- `tcgen05.copy` now rejects any subword destination whose phase is not proven
  zero, not just statically visible odd-column views. This keeps copy on the
  clean-negative side until a valid copy lowering exists for odd subword
  origins.
- Added f16/i8 dynamic-select and loop-carried unaligned ld/st runtime
  positives and a loop-carried unaligned copy diagnostic. Validation: required
  `make -j8`; selected unaligned f16/i8 ld/st/copy rows `18 passed`; broader
  adjacent subword ld/st/copy selector `28 passed, 1650 deselected`; focused
  `cp_no_scales and subword` sweep `36 passed, 1642 deselected`; lit
  `tmem_layouts.mlir` `1 passed`.

### 2026-04-23 Residue-Precise Alignment Slice

- The transition phase helper now uses residue-set analysis internally. The
  public subword query still returns `KnownZero`, `MayBeNonZero`, or `Unknown`,
  but recognized roots, view/index offsets, selects, loops, calls, and CFG
  joins are combined modulo the queried granularity instead of only recording
  whether any static offset was nonzero.
- Added `getTMemElementOffsetModuloStatus(Value, modulus)` for consumers that
  need local alignment facts other than subword phase. This remains a value
  classifier, not a producer-chain semantic contract: it classifies the current
  SSA value through local CFG/call forwarding while the runtime descriptor
  still carries the physical element-column `taddr`.
- Copy lowering now asks two independent questions before emitting
  `tcgen05.copy`:
  - is the current subword phase proven zero for sub-32-bit destinations;
  - is the current destination origin proven aligned to a 128-bit hardware copy
    address.
- The second gate is necessary because word alignment is not sufficient for the
  copy ISA. A packed f16 descriptor at element column 2 projects to hardware
  word column 1; that is a valid 32-bit word address but caused a misaligned
  `tcgen05.copy` runtime failure. The backend now rejects it cleanly. Nested
  packed f16/i8 slices whose static offsets cancel to a 128-bit boundary are
  runtime positives.
- Validation after this slice: required `make -j8`; nested copy positives and
  the word-aligned/copy-misaligned clean negative `3 passed, 1678 deselected`;
  adjacent subword copy/ldst selector `49 passed, 1632 deselected`; adjacent
  non-subword active copy selector `26 passed, 1655 deselected`; lit
  `tmem_layouts.mlir` `1 passed`.

### 2026-04-23 Phase-Aware Fallback Crash Guard

- A reversed-column packed f16 subview demonstrated a remaining weakness in the
  transition phase-aware ld/st lowering. The current implementation only
  supports contiguous packed `32x32b` RMW plans; that part correctly emitted an
  unsupported-layout diagnostic, but the enclosing lowering continued into
  source/raw fallback paths that were designed for aligned word-column support
  layouts and could assert with `Invalid basis`.
- `lowerTMemLdStFromTypes` now distinguishes between query-computation failure
  and actual phase-aware codegen failure. Query failures may still fall through
  to later candidate plans. Once phase-aware codegen has emitted its
  unsupported-layout diagnostic, lowering stops for that op so no incompatible
  fallback can reinterpret the same current memdesc value through a stale
  support layout.
- Added runtime-matrix coverage for the reversed-column packed f16 subview.
  This is a temporary clean negative for the current narrow RMW implementation,
  not proof that the layout is ISA-impossible. Future non-contiguous subword
  support should revisit it with a valid local lowering from the current
  element-column `taddr` and current `MemDescType`/layout.
- Validation after this slice: required `make -j8`; focused unaligned subword
  ld/st selector including the new reversed-column row `11 passed, 1671
  deselected`; adjacent subword copy/ldst selector including the new row
  `50 passed, 1632 deselected`; lit `tmem_layouts.mlir` `1 passed`;
  `git diff --check` passed.

### 2026-04-23 Subword Software-Reduction Sentinel

- Added f16/i8 odd-column active-view coverage through `load_max`. These
  element types do not lower to hardware `tcgen05.ld.red`; the frontend uses
  normal TMEM loads plus software reduction. That makes the row a useful
  reduction-facing sentinel for the current phase-aware ld/st RMW path without
  conflating it with the separate hardware f32 `redOp` lowering question.
- The sentinel stores through `slice(1, N)`, runs `load_max` through the same
  odd descriptor, checks row maxima, and reloads neighboring aligned/tail views
  to prove packed-lane preservation across the store and load sequence. It
  also asserts that no hardware `ld.red` opcodes are emitted and that the
  scalar tail load/store opcodes needed by phase-aware RMW are present.
- Validation after this slice: required `make -j8`; exact new rows `2 passed,
  1682 deselected`; adjacent ld.red/software-reduce selector `50 passed,
  1634 deselected`.

### 2026-04-23 Hardware Ld.Red Address Alignment

- A f32 active view at element-column offset 1 showed that hardware
  `tcgen05.ld.red` has an address-alignment requirement that normal TMEM loads
  do not share. The old support query considered only the current
  `MemDescType`/register layout and allowed the hardware reduction, which then
  faulted at runtime with a misaligned address.
- Added `isTMemLoadReductionAddressAligned(Value)`. It asks the same
  element-column residue engine used for subword and copy alignment whether
  the current memdesc SSA value is proven aligned to a 128-bit hardware
  address. For f32 this is element-column modulo 4.
- The check is now applied at every place that can select or accept hardware
  `ld.red`: the Gluon frontend support query, the `OptimizeTMemLayouts`
  load+reduce fusion pass, the `TMEMLoadOp` verifier, and LLVM lowering. This
  keeps legality local to the current memdesc value/type/layout and prevents
  later rewrites from reintroducing a bad hardware red op.
- Misaligned and unknown origins remain semantically valid through normal
  TMEM load plus software reduction. Runtime coverage checks f32 offset 1
  takes that path and f32 offset 4 still emits `tcgen05.ld.red`, with explicit
  neighboring-view readback to prove the selected current `taddr` is honored.
- Validation after this slice: required `make -j8`; exact subword/software and
  f32 offset-column reduction rows `4 passed, 1682 deselected`; adjacent
  ld.red/software-reduce selector `52 passed, 1634 deselected`; lit
  `tmem_layouts.mlir` `1 passed`; `git diff --check` passed.

### 2026-04-23 Subword Storage-Layout Positives

- Added odd logical column ld/st runtime positives for storage classes that
  exercise different parts of the memdesc model:
  tile-permuted packed f16 `tensor_memory_linear`, legacy unpacked f16
  (`col_stride=2`), and legacy padded i8 (`col_stride=4`).
- The tile-permuted row is still a packed subword RMW case, but it proves the
  valid lowering does not require an identity parent layout; support-query
  planning can recover a contiguous packed `32x32b` plan from the current
  descriptor facts.
- The unpacked/padded rows are deliberately different: logical column offset 1
  maps to a 32-bit-word-aligned current origin because the storage layout
  spaces f16/i8 elements across whole dword slots. These rows document that
  "odd logical column" and "nonzero packed subword phase" are not equivalent;
  legality must come from physical element-column residue in the current
  memdesc value and layout.
- Validation after this slice: required `make -j8`; focused unaligned subword
  ld/st selector `14 passed, 1675 deselected`; adjacent subword copy/ldst
  selector `53 passed, 1636 deselected`; `git diff --check` passed.

### 2026-04-23 Pointwise View-Offset Inversion

- A `warpx2` copy-subview shard showed that globally non-surjective layouts
  can still have valid preimages for the concrete logical offsets used by a
  view. The old `ll.pseudoinvert().apply(offsets)` path required the whole
  image to be invertible enough for a layout-wide pseudoinverse and asserted in
  `LinearLayout::lstsq` before lowering could use the representable point.
- `getTMemViewPhysicalRowElementCol` now solves one augmented GF(2) system for
  the requested logical offset and returns the physical row/element-column
  point when it exists. The public optional variant lets analysis callers
  report unknown for non-representable points; the non-optional helper keeps
  the existing hard-failure behavior for code paths that require a valid
  offset.
- Because the point solve is now strong enough, generic TMEM
  `memdesc_subslice` lowering no longer needs a chain-query element-offset
  workaround. It uses `getTMemViewElementOffset(srcTy, offsets)`, so the
  lowered address update is derived from the current source `MemDescType` and
  explicit offsets only.
- The load+reduce fusion pattern now calls the hardware `ld.red` address
  alignment helper only after it has found a reduction user. This avoids
  executing phase/alignment analysis on plain loads that cannot be fused and
  whose descriptor layouts may be difficult but irrelevant to reduction
  legality.
- Validation after this slice: required `make -j8`; saved warpx2 compiler
  reproducer passed with `triton-opt --run-reproducer`; exact warpx2 runtime
  row `2 passed, 1687 deselected`; focused subword ld/st selector `12 passed,
  1677 deselected`; combined non-scale TMEM runtime selector passed as group1
  `145 passed, 11 skipped`, group2 `89 passed, 67 skipped`, group3
  `136 passed, 20 skipped`, group4 `156 passed`.

### 2026-04-23 Static Subslice Phase Fallback Removal

- With the pointwise view-offset query in place, static
  `memdesc_subslice` phase analysis can ask the current source
  `MemDescType`/layout directly. The old fallback reconstructed standalone
  source/result query origins from the producer chain and used their column
  delta when the type-local physical query failed.
- That fallback is removed. Non-representable type/local points now become
  unknown phase, which keeps copy and `ld.red` conservative without allowing a
  visible producer chain to define legality.
- Validation after this slice: required `make -j8`; focused subword ld/st
  selector `12 passed, 1677 deselected`; subword copy selector `39 passed,
  1650 deselected`; ld.red subword/linear-subslice selector `8 passed, 1681
  deselected`; `git diff --check` passed.

### 2026-04-23 MMAv5 Physical-Bitcast Locality

- After runtime `taddr` rescaling for physical bitcasts, MMAv5 lowering no
  longer needs to ask whether the current memdesc SSA value was directly
  defined by `tmem_physical_bitcast`. The runtime base is already in the result
  element-column coordinate frame, and the current `MemDescType`/layout carries
  the static address family.
- Removed the physical-bitcast special case from
  `getMMAv5TMemAddressLayout` and `getMMAv5TMemViewOffsetForLowering`. Both
  helpers now try type-local planning first, then use legacy value fallback only
  for descriptor classes not yet migrated.
- Added a selected physical-bitcast MMA lhs test: two f32 column subviews are
  bitcast to same-typed bf16 TMEM views, populated with different constants,
  selected dynamically, and consumed by `tcgen05_mma`. This ensures valid MMAv5
  lowering does not depend on a visible reinterpret producer.
- Validation after this slice: required `make -j8`; direct plus selected
  physical-bitcast MMA rows `3 passed`; `test_core.py -k physical_bitcast`
  `7 passed`; runtime-matrix `mma and not reports` selector passed across four
  GPUs as group1 `135 passed, 14 skipped`, group2 `149 passed`, group3
  `149 passed`, group4 `147 passed`; lit `tmem_layouts.mlir` `1 passed`.

### 2026-04-23 Matched Type-Local Copy Query Selection

- Broadened `selectTMemCopyPhysicalQuery` beyond the active-subview-only case.
  The selector still computes the legacy standalone/exact candidates, but now
  chooses the current-type physical query when that type-local query matches the
  physical projection the legacy algorithm would have selected and can compose
  with the shared-memory source layout.
- This is intentionally conservative: descriptor classes whose type-local query
  does not match the selected legacy projection keep the old standalone/exact
  fallback. For matched cases, lowering records `usedTypeLocal=true` and keeps
  the current runtime `taddr` instead of applying a producer-chain-derived
  relative destination offset.
- Validation after this slice: required `make -j8`; no-scale copy runtime
  matrix passed across four GPUs as group1 `59 passed, 4 skipped`, group2
  `63 passed`, group3 `63 passed`, group4 `62 passed`; scales copy selector
  `34 passed`; lit `tmem_layouts.mlir` `1 passed`; `git diff --check` passed.

### 2026-04-23 Type-Local Copy Closeout

- Added a copy-specific type-local predicate for descriptor classes whose copy
  destination semantics are self-contained in the current `MemDescType`: active
  TMEM subviews and type-local scales descriptor views.
- `selectTMemCopyPhysicalQuery` now chooses that type-local query before
  constructing legacy standalone/exact producer-chain candidates. For these
  classes, copy lowering keeps the current runtime `taddr` and clean-negative
  diagnostics come from the current destination layout.
- Direct roots are deliberately not forced through the raw type-local query. A
  broad no-scale copy probe showed that 256-row dense roots and
  tile-selector-permuted layouts still need the existing support-query planner;
  using the raw 256-row direct query either rejected legal rows or miscompiled a
  selector-permuted copy. This remains a support-planner boundary, not a reason
  to reintroduce producer-chain semantics for active descriptor views.
- Packed-lane/too-small `tcgen05.copy` rows remain clean negatives for this
  phase unless the current descriptor layout itself represents a legal copy
  family. The compiler must not borrow hidden parent columns to satisfy a copy
  atom footprint.
- Validation after this slice: required `make -j8`; focused reports `18
  passed`; focused linear/direct probe `100 passed`; no-scale copy selector
  `302 passed, 4 skipped`; scales copy selector `34 passed`; four-GPU
  runtime-matrix `cp_no_scales or cp_scales` passed as group1 `81 passed, 4
  skipped`, group2 `85 passed`, group3 `85 passed`, group4 `85 passed`;
  `test_core.py -k 'tmem_copy or mma_scaled_tcgen05_copy'` `149 passed, 5
  skipped`; lit `TritonNvidiaGPU/tmem_layouts.mlir` `1 passed`.

### 2026-04-23 MMAv5/Scaled Semantic Closeout

- Removed the remaining lowering-facing value fallback from MMAv5 address and
  tile-order planning. `getMMAv5TMemAddressLayout` and
  `getMMAv5TMemViewOffsetForLowering` now take only the current `MemDescType`,
  and `DotOpMmaV5TmemLoader` no longer receives the memdesc SSA value for
  producer-chain query recovery.
- Scaled MMA verifier and LLVM lowering now classify B-scale padded/unpadded
  descriptor-view storage with `getMMAv5ScaledBScaleStorageType(MemDescType)`.
  The public `getMMAv5ScaledBScaleStorageTypeThroughViews(Value)` API was
  removed; tensor-memory allocation keeps a private through-view helper only to
  rematerialize scale storage before lowering.
- This closes the MMAv5/MMAv5-scaled semantic lowering surface for the current
  memdesc model: legality and PTX address/tile-order facts come from operation
  semantics, current `MemDescType`/layout, and the runtime `taddr`, not the
  producer chain. Optimizers may still inspect producer chains to rewrite IR
  before lowering.
- Validation after this slice: required `make -j8`; focused dynamic selected
  physical-bitcast/scales rows `10 passed`; runtime-matrix `mma and not
  reports` passed across four GPUs as group1 `135 passed, 14 skipped`, group2
  `149 passed`, group3 `149 passed`, group4 `147 passed`; focused
  `test_core.py` MMA selector `17 passed, 3 skipped`; lit
  `TritonNvidiaGPU/tmem_layouts.mlir` `1 passed`; `git diff --check` passed.
