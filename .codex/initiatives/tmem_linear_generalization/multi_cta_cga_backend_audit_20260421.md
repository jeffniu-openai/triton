# Multi-CTA CGA TMEM Backend Audit - 2026-04-21

## Scope

This audit follows the scaled-copy ownership regression fixed in
`22222eb38`. The specific question is whether the TMEM backend still has
places that accidentally treat `two_ctas` as the total number of CTAs in the
CGA, instead of treating it as the tcgen05 instruction group size inside a
possibly larger CGA.

The invariant for future work is:

- `two_ctas` selects `tcgen05.*.cta_group::2` behavior and within-pair
  ownership.
- Total CGA ownership comes from the operand, result, barrier, descriptor, or
  shared-memory layout.
- For `num_ctas > 2`, a correct path must reason about both the pair bit and
  the outer CTA bits.

## Findings

### A. TMA lowering uses the module-wide two-CTA bit

Code:

- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp:1223`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp:1649`

`AsyncTMACopyGlobalToLocalOp` and `AsyncTMAGatherOp` append
`cta_group::2` whenever `getModuleTwoCTAs(op)` is true. This means a module
with one two-CTA TMEM operation changes all TMA load/gather instruction forms,
even if the TMA op's own multicast and barrier layout are not two-CTA-pair
operations.

Resolution:

- This is not a backend gap. The hardware restriction is module-wide: if any
  two-CTA-capable instruction in a kernel uses two-CTA mode, every
  two-CTA-capable instruction in that kernel must use two-CTA mode.
- TMA load/gather `cta_group::2` selection from `getModuleTwoCTAs` is therefore
  the correct source of truth for instruction form. Ownership and multicast
  recipients still come from the TMA result/barrier layouts.

Recommended next step:

- Keep module-wide selection. Do not refactor this to per-op TMA mode unless
  the hardware restriction changes.

Status: hardware restriction, not a bug.

### B. End-of-kernel cluster barrier suppression is module-wide

Code:

- `lib/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.cpp:347`
- `lib/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.cpp:358`
- `third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp:558`

Cluster barrier insertion suppresses the normal return-side cluster barrier
for any kernel where `getModuleTwoCTAs(funcOp)` is true, relying on the TMEM
deallocation path to emit a cluster arrive/wait before `tcgen05.dealloc`.

Risk:

- The deallocation sync is whole-cluster, so it is probably sufficient for a
  simple single-return kernel with all unresolved distributed dependencies
  before TMEM deallocation.
- The current test in `ClusterBarrierInsertion` still conflates "module has a
  two-CTA tcgen05 instruction group" with "all outstanding cross-CTA DSM/TMA
  dependencies are covered by the TMEM deallocation sync." Mixed kernels and
  control-flow-heavy kernels need targeted coverage.

Recommended next step:

- Add lit tests with `num_ctas = 4` where a two-CTA TMEM op and a separate
  multicast TMA/DSM dependency both reach return.
- Check that every return path still has a cluster-level synchronization point
  after the final distributed shared-memory use.
- If a path can return before the deallocation sync covers the dependency,
  split the decision into "TMEM dealloc provides return sync" versus
  "module has two_ctas".

Status: coverage gap. The follow-up test added in this slice pins the
pre-copy cluster barrier for a 4-CTA CGA with an outer CTA bit; return-side
mixed-control-flow coverage remains a separate broad validation item.

### C. TMEM copy and commit predicate is pair-leader only

Code:

- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp:1199`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.cpp:186`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp:1035`

For two-CTA TMEM copy and commit, the predicate is restricted to
`cluster_id & 1 == 0`. This is pair-leader semantics, not whole-cluster leader
semantics, so it is the right shape for multiple CTA pairs. The recent
scaled-copy runtime fix also exercised `(4, 1)`, `(4, 2)`, and `(4, 4)` CGAs.

Risk:

- `TMEMCopyOp` has no completion descriptors, so `tcgen05.commit` only uses
  the two-CTA pair broadcast implicit in `cta_group::2`; any outer-CGA
  completion semantics must come from the barrier layout and cluster pointer.
- There is no explicit lit test proving that pair leaders for CTA pairs
  `(0,1)` and `(2,3)` both emit the expected copy/commit in a 4-CTA CGA.

Recommended next step:

- Add a PTX/lit test that checks the predicate and emitted `tcgen05.cp` /
  `tcgen05.commit` pattern for a 4-CTA CGA scale-copy case with an outer
  row-only block basis.

Status: current implementation is correct under the module-wide two-CTA
hardware rule. The follow-up test added in this slice pins the pair-leader
predicate for a 4-CTA CGA.

Follow-up fix:

- The 4-CTA lit probe exposed a real source-layout gap for
  `tcgen05.copy.warpx2`: source validation accepted only a 128x4 single-CTA
  tile or a 256x4 two-CTA tile. It now accepts `128 * num_ctas` by 4 tiles
  when the shared-linear block bases are the canonical row-only sequence
  `[[128, 0], [256, 0], ...]`.

### D. Barrier recipient masks default to the pair bit when descriptors are absent

Code:

- `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:2114`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp:330`
- `lib/Dialect/TritonInstrument/Transforms/ConcurrencySanitizer.cpp:251`
- `lib/Dialect/TritonInstrument/Transforms/ConcurrencySanitizer.cpp:277`

`getCTABroadcastMasks(twoCTAs, descs)` ORs bit 0 into descriptor-derived
broadcast masks when descriptors exist, and returns only bit 0 when
descriptors are absent. This is correct for pure two-CTA pair behavior, but
operations with outer-CGA completion semantics need descriptors or an
operation-specific ownership source.

Risk:

- MMAv5 commit paths pass completion descriptors, so outer CTA masks can be
  represented there.
- `TMEMCopyOp` currently has no descriptors. Its sanitizer mem-effect
  recipient path also calls `getCTABroadcastMasks(twoCTAs, {})`, so it models
  only pair recipients.
- If future TMEM copies signal barriers whose ownership includes outer CTA
  bits, both lowering and instrumentation need an explicit source for those
  recipient masks.

Recommended next step:

- Add instrumentation and backend tests for TMEM copy with `num_ctas = 4`
  where barrier layout has outer broadcast bits.
- Decide whether TMEM copy should grow completion-desc-like metadata or derive
  recipient masks from the barrier/destination memdesc.

Status: no current bug found. Descriptor-less TMEM copy completion is pairwise
under `cta_group::2`; future copy forms with additional outer-CGA completion
semantics should add explicit recipient metadata rather than overloading the
module-wide mode bit.

### E. The module-level consistency pass forbids mixed single-CTA and two-CTA TMEM

Code:

- `lib/Dialect/TritonNvidiaGPU/Transforms/CheckMatmulTwoCTAs.cpp:31`
- `lib/Dialect/TritonNvidiaGPU/Transforms/CheckMatmulTwoCTAs.cpp:77`
- `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:897`

The pass records the first explicit two-CTA user and requires every MMAv5 op
and TMEM type in the module to match. MMAv5 verification also requires the LHS
and accumulator TMEM encodings to match the op's `two_ctas` attribute.

Resolution:

- This is the compiler representation of the hardware restriction: if any
  two-CTA-capable instruction is two-CTA, all two-CTA-capable instructions in
  the kernel must be two-CTA.
- The consistency pass is therefore required, not an expressiveness limitation
  to remove.

Recommended next step:

- Keep the pass and keep `getModuleTwoCTAs` consumers module-wide for
  instruction-form selection. Continue to derive ownership, offsets, and
  recipient masks from layouts.

Status: hardware restriction, not a bug.

### F. Structured TMEM verifier requires the first CGA block basis to be the pair bit

Code:

- `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp:4632`
- `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp:4707`
- `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:858`

Structured tensor-memory encodings require the first CGA block basis to be the
within-pair split (`[1, 0]` for accumulator-like TMEM and `[0, 1]` for RHS
shared layouts). Linear TMEM encodings are more permissive: they require a
non-empty, non-zero block basis for `twoCTAs` but do not force the first block
basis.

Risk:

- This looks like a real hardware/layout requirement for structured MMAv5
  families: the pair bit must be adjacent to the instruction's two-CTA split.
- For arbitrary linear layouts, the permissive verifier is aligned with the
  project goal and should be preserved.

Recommended next step:

- Treat structured-first-basis checks as hardware-family requirements unless
  a concrete ISA-realizable counterexample appears.
- Ensure future failures from these checks are explained as pair-bit placement
  requirements, not as project scope limitations.

Status: likely hardware boundary for structured families, not a bug.

## Priority Checklist

1. Return-side cluster-barrier tests for mixed two-CTA TMEM plus other
   distributed dependencies in `num_ctas > 2`.
2. Concurrency-sanitizer and barrier-recipient coverage for TMEM copy with
   outer-CGA barrier ownership.
3. Keep adding regression tests that distinguish module-wide two-CTA
   instruction form from per-layout CGA ownership.

## Validation Evidence

2026-04-21 follow-up:

- Added conversion coverage for a 4-CTA `ttng.tmem_copy` lowering that checks
  `cluster_id & 1 == 0`, proving the predicate is per-pair leader selection
  rather than whole-CGA CTA 0 selection.
- Added membar coverage for a 4-CTA distributed shared producer followed by a
  two-CTA-capable `ttng.tmem_copy`, proving the pre-copy cluster barrier is
  still inserted when an outer CTA bit exists.
- Fixed `tcgen05.copy.warpx2` shared-source validation so the same canonical
  pair-local source layout works in larger CGAs with outer CTA-pair bases.

The original audit was static and did not change compiler behavior.

Relevant prior validation at current HEAD `22222eb38`:

- Required `make -j8` passed before the previous backend fix.
- The affected scaled-copy/direct-multicast selector passed across four GPUs:
  `27`, `27`, `27`, and `24` selected tests.
- The affected lit suite passed `11/11`.
- Restored-upstream MLIR symbol audit reported `TOTAL_MISSING_FUNCS 0`.
