# TMEM iisan Policy Audit - 2026-04-24

## Policy Decision

TMEM backend diagnostics must separate two different classes of hardware
constraints:

1. **Runtime trap preconditions.** If Triton can directly emit the intended PTX
   instruction, but the instruction traps or raises an illegal instruction when
   the final runtime tensor-memory address or runtime operand value violates an
   undocumented hardware precondition, the compiler should not reject the kernel.
   These facts belong in illegal-instruction sanitizer (`iisan`) instrumentation.
   With iisan enabled, Triton should emit a device-side assertion that explains
   the precondition before the hardware trap. With iisan disabled, Triton should
   emit the PTX and let the hardware behavior match raw PTX.

2. **PTX/codegen impossibility.** If Triton cannot synthesize a correct PTX
   instruction sequence for the requested operation because no supported atom,
   message shape, operand descriptor, register layout, or schedule exists, the
   compiler must continue to report a clean unsupported diagnostic. These errors
   are not iisan checks because there is no correct direct PTX lowering to guard.

This mirrors the existing TMA policy: gather/scatter/load/store runtime
alignment and nonnegative-index requirements are modeled as iisan device asserts
in Gluon, while structural descriptor and layout requirements remain verifier or
lowering diagnostics.

A static type/layout proof is still useful, but only as an optimization:
if the current memdesc type/layout and immediate SSA value prove a runtime trap
precondition, iisan may elide the device assertion. Failure to prove the
precondition must not be used as a compiler rejection when a PTX instruction can
otherwise be emitted.

The producer-chain invariant remains unchanged: verifier and codegen legality
must be decided from the current memdesc SSA value, current `MemDescType`/layout,
and operation semantics. iisan checks may use the current runtime memdesc address
value and operation-local offsets. They must not recover legality by walking
view/producer chains.

## Probed TMEM Runtime Trap Preconditions

The raw PTX sweep in `tmem_ptx_alignment_probe_20260424.md` establishes the
following iisan candidates. All passing rows checked output correctness; failing
rows assembled and then trapped at runtime.

| Instruction surface | Observed runtime rule | Compiler policy |
|---|---|---|
| `tcgen05.ld.sync.aligned.32x32b.x64.b32` f32 | offsets `0..15` passed | no iisan alignment check currently needed |
| `tcgen05.st.sync.aligned.32x32b.x64.b32` f32 | offsets `0..15` passed | no iisan alignment check currently needed |
| `tcgen05.ld/st.sync.aligned.16x64b.x32.b32` f32 roundtrip | final TMEM address even, i.e. 64-bit | iisan if this atom is selected |
| `tcgen05.ld/st.sync.aligned.16x128b.x32.b32` f32 roundtrip | final TMEM address even, i.e. 64-bit | iisan if this atom is selected |
| packed f16 `16x32bx2.x16.b32` ld/st | offsets `0..15` passed | no hardware-word-address iisan check currently needed |
| unpacked f16 `16x32bx2.x16.{pack,unpack}::16b.b32` ld/st | final TMEM address even, i.e. 64-bit | iisan if this path is selected |
| f32 `tcgen05.ld.red.sync.aligned.32x32b.x{2,4,8,16,32,64}` | final TMEM address even, i.e. 64-bit | iisan; remove verifier/lowering rejection |
| `tcgen05.cp.cta_group::1.128x256b` f32 destination | final destination address multiple of 4 f32 columns, i.e. 128-bit | iisan; do not reject merely because type proof is unknown |
| plain MMAv5 f16 accumulator D | final D address even, i.e. 64-bit | iisan if D is read/written by MMA |
| plain MMAv5 f16 A-in-TMEM operand | final A address multiple of 4 hardware columns, i.e. 128-bit | iisan when A operand is TMEM |
| scaled MMAv5 accumulator D | final D address even, i.e. 64-bit | iisan |
| scaled MMAv5 A/B scale address path | final scale address even, i.e. 64-bit | iisan for scale operands |

The `.32x32b` f32 `ld.red` rule is especially important for
`python/examples/gluon/01-attention-forward.py`: the branch currently rejects
`s_tmem.slice(...).load_max()` because it cannot prove a 128-bit origin, but the
empirical hardware rule is 64-bit final-address alignment and should be reported
by iisan rather than the verifier.

## Current Branch Audit

### Checks That Should Move To iisan

- `TMEMLoadOp::verify` rejects red-loads through
  `isTMemLoadReductionAddressAligned(getSrc())` with the diagnostic
  `tmem_load reduction requires a 128-bit-aligned tensor memory origin for
  tcgen05.ld.red` in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`. This is a runtime
  address trap precondition, and the probed rule is 64-bit for f32 `.32x32b`, not
  128-bit. The verifier should keep structural red-load checks, but this address
  rejection should be removed or converted into an iisan-only assertion.

- `lowerTMemLdStFromTypes` repeats the same `ld.red` address rejection in
  `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`. Lowering
  should emit `tcgen05.ld.red` when structural layout planning succeeds. iisan
  should assert final red-load address alignment when enabled.

- `isTMemLoadReductionAddressAligned` and
  `getTMemElementOffsetModuloStatus(..., 128 / bitwidth)` remain useful only as
  static proof helpers for iisan assertion elision. They should no longer define
  red-load legality.

- `tcgen05.copy` lowering rejects destination origins whose type/layout does not
  prove 128-bit alignment in `TensorMemoryToLLVM.cpp`. The PTX sweep shows this
  is a real hardware address trap precondition for the probed f32
  `128x256b` copy, not a PTX-shape impossibility. If the copy planner has an
  executable plan, unknown or unproven final destination alignment should be an
  iisan assertion rather than a compiler error.

- Copy-planner failures that are purely "instruction base would be unaligned"
  need to be split out from genuine schedule/coverage failures. For example,
  diagnostics such as `direct tcgen05.copy requires each logical column tile to
  start at a physical column aligned to the copy instruction width` describe an
  address precondition if the atom schedule is otherwise complete. Those should
  become iisan facts. Checks for duplicate tile offsets, missing coverage, or
  impossible source/destination projections should remain compiler errors.

- There are currently no iisan checks for selected plain ld/st atoms with probed
  runtime alignment requirements (`16x64b`, `16x128b`, unpacked
  `16x32bx2`), MMAv5 D/A-in-TMEM address alignment, or scaled-MMA scale address
  alignment. These are missing iisan coverage, not verifier responsibilities.

### Checks That Should Remain Compiler Unsupported Diagnostics

- `tmem.load_min/max` structural checks remain verifier diagnostics:
  f32 element type, no tensor-memory scales, packed lowering, `.x2` or wider
  `ld.red` message shape, and directly supported reduction register layout.
  These determine whether Triton can emit a direct `tcgen05.ld.red` sequence.

- Generic TMEM ld/st layout failures remain compiler diagnostics: unsupported
  destination/register layouts, unsupported row footprints, missing row anchors,
  unsupported register broadcast patterns, non-self-contained descriptor-view
  types, and `16x32bx2` warp-anchor row-tile incompatibilities. These are direct
  lowering/planner gaps, not runtime address values.

- `tcgen05.copy` family and schedule failures remain compiler diagnostics when
  no executable copy plan exists: too-small current descriptors relative to ISA
  atom footprints, no recognized copy family, source/destination conversion
  failures, shared descriptor synthesis failures, CTA-ownership restrictions,
  source row/column projection gaps, subword packed-lane footprint gaps, and
  resource-boundary failures. The explicit too-small atom message is the model
  case: the compiler cannot borrow hidden parent columns for a narrower current
  descriptor.

- The sub-32-bit `tcgen05.copy` destination subword-phase check should remain a
  compiler unsupported diagnostic for now. The current direct copy lowering has
  only a hardware word address and no way to represent "start in the high half"
  of a 32-bit destination word without a separate packed-lane copy model. If raw
  PTX probes later show a trap-only rule for a specific subword copy opcode, that
  specific fact can move to iisan, but the current correctness issue is
  representability rather than merely runtime alignment.

- `tcgen05.copy` direct-seed source offset alignment to 128-bit descriptor units
  should remain a compiler/schedule diagnostic unless a different source
  descriptor/address schedule is synthesized. This is an encoding/synthesis
  limitation of the selected shared-memory descriptor path, not just a TMEM
  runtime destination-address assertion.

- MMAv5 and scaled-MMAv5 structural verifier/lowering failures remain compiler
  diagnostics: unsupported dtypes, rank/encoding mismatches, unsupported
  transposed f32 shared operands, required TMEM col strides, accumulator or
  LHS tile layouts that cannot normalize to an instruction tile, two-CTA
  repetition restrictions, block-basis mismatches, and scaled-MMA scale-fragment
  storage/rematerialization gaps. These determine whether the compiler can emit
  a valid MMA instruction sequence.

- Module-wide CTA-group consistency (`getModuleTwoCTAs`) remains a static
  compiler contract. It is not a runtime value-dependent precondition: once any
  instruction in the kernel requires cta_group::2, all compatible instructions
  in the kernel must be emitted consistently.

## Implementation Consequences

- Add TMEM iisan helpers alongside the existing TMA iisan pattern. The helper
  should assert on the current runtime memdesc address plus operation-local
  packet/tile offsets, and should be predicated consistently with the guarded
  instruction where possible.

- Keep static type/layout proofs as optional iisan-elision proofs. A proven
  aligned address may skip the assertion; an unknown proof emits the assertion
  under iisan and still compiles under normal mode.

- Remove verifier/lowering address-alignment rejections for instructions whose
  PTX can be emitted directly. Start with `ld.red` because it is blocking the
  attention red path and the current 128-bit rule is empirically too strict.

- Split copy diagnostics before deleting broad checks: isolate true atom/schedule
  impossibility from final-address alignment facts so unsupported messages remain
  rigorous.

- Add iisan tests that intentionally pass misaligned TMEM addresses in a fresh
  process and check for the new device-assert message instead of a raw CUDA
  illegal-instruction error. Keep non-iisan runtime trap tests process-isolated
  if they are needed as raw hardware sentinels.


## Implementation Update - 2026-04-24 18:39 UTC

Implemented the first policy slice in the branch:

- Plumbed `instrumentation_mode="iisan"` into `convert-triton-gpu-to-llvm` as `enable-illegal-instruction-sanitizer`, so TMEM instruction lowering can emit sanitizer checks at the point where final instruction addresses are known.
- Removed `ld.red` final-address alignment as a verifier/lowering legality condition. Structural `ld.red` support checks remain unchanged: f32 element type, non-scales storage, packed direct lowering, directly supported register layout, and `.x2` or wider message shape.
- Removed executable-plan `tcgen05.copy` destination 128-bit alignment as a compiler rejection and replaced it with an iisan address assertion. True copy impossibility still rejects at compile time, including too-small atom shapes, unsupported copy-family schedules, descriptor synthesis failures, and sub-32-bit destination subword phase.
- Added iisan assertions derived only from the current lowered TMEM address plus operation-local offsets for `ld.red`, selected direct ld/st atoms (`16x64b`, `16x128b`, unpacked `16x32bx2`), `tcgen05.copy`, MMAv5 accumulator/TMEM-A operands, and scaled-MMAv5 scale addresses. No producer/view-chain recovery was added.
- Added runtime iisan tests in `python/test/gluon/test_tmem_runtime_matrix.py` that run in fresh child processes and check the device-assert messages for misaligned `ld.red` and `tcgen05.copy` addresses. Refreshed stale clean-diagnostic expectations in the structural fuzzer and lit verifier file.

Validation for this implementation slice:

- `make`
- `pytest -s --tb=short` exact iisan/aligned runtime rows in `test_tmem_runtime_matrix.py`: `3 passed`
- `pytest -s --tb=short` structural-fuzzer ld.red clean-negative rows: `4 passed`
- `TRITON_INSTRUMENTATION_MODE=iisan pytest -s --tb=short` MMAv5 positive rows in `test_core.py`: `2 passed`
- `TRITON_INSTRUMENTATION_MODE=iisan pytest -s --tb=short` selected direct ld/st rows in `test_tmem_runtime_matrix.py`: `2 passed`
- `lit -v test/TritonNvidiaGPU/invalid.mlir test/TritonNvidiaGPU/tmem_layouts.mlir`: `2 passed`
- `python/examples/gluon/01-attention-forward.py` representative `use_tmem_red=True` rows: `4 passed`

## Implementation Update - 2026-04-24 19:01 UTC

Follow-up policy adjustment after user review: TMEM iisan checks that are useful for Gluon should be emitted by the Gluon frontend, matching the existing TMA iisan implementation. Triton/non-Gluon paths are expected to have middle-end legality and should not rely on NVIDIA GPU-to-LLVM sanitizer insertion.

Changes from the previous 18:39 backend implementation:

- Removed the `convert-triton-gpu-to-llvm` `enable-illegal-instruction-sanitizer` pass option and the TMEM iisan assertion helpers from NVIDIA GPU-to-LLVM.
- Added `ttng.tmem_address` so Gluon frontend checks can inspect the current memdesc SSA value's packed runtime taddr without walking producer/view chains. NVIDIA GPU-to-LLVM only lowers this helper op to the already-lowered memdesc address.
- Moved Gluon-visible TMEM runtime address checks to `python/triton/experimental/gluon/language/nvidia/blackwell/__init__.py` for explicit reduction loads, `tcgen05_copy`, plain MMAv5, and scaled MMAv5. The checks use the memdesc dtype to convert hardware word-column alignment requirements into element-column moduli.
- Direct ld/st selected-atom iisan checks from the backend implementation were removed rather than preserved in LLVM lowering. If needed later, they should be implemented through a frontend query that exposes the exact chosen atom and packet offsets to Gluon; otherwise the middle-end must avoid generating illegal selected ld/st instructions.

Validation for the frontend migration:

- `make`
- exact red/copy iisan runtime rows: `2 passed`
- aligned red subslice runtime row: `1 passed`
- structural-fuzzer ld.red clean-negative rows: `2 passed`
- `TRITON_INSTRUMENTATION_MODE=iisan` MMAv5/scaled positive rows: `2 passed`
- lit `invalid.mlir` + `tmem_layouts.mlir`: `2 passed`
- representative attention red-path rows: `4 passed`
