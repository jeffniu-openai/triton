# Round 62 Lane C: Backend Source Audit for Missing Runtime Probes

Date: 2026-04-21
Branch: `codex/tmem`
Scope: source-code audit only, with one focused existing runtime sanity pair.
Ownership boundary: this report only; no backend/compiler/test source edits.

## Commands

Required rebuild:

```bash
make -j8
```

Result: no-op, `ninja: no work to do`.

Focused existing runtime probe:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  pytest -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_positive[f32-torch_dtype0]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_descriptor_chain_n_sweep[identity_n64-False-propagate_nan0-min]'
```

Result: `2 passed in 3.05s`.

Collection note: an earlier attempt without `PYTHONPATH=.:./python` failed
before collection with `ModuleNotFoundError: No module named
'triton.compiler.errors'`. A second attempt used stale guessed parametrized
nodeids and collected zero tests. The collect-only selector then identified the
exact checked-in nodeids above.

## Audited Backend Surfaces

- `lib/Conversion/TritonGPUToLLVM/ViewOpToLLVM.cpp`
  - TMEM `memdesc_index` lowering uses exact linear-layout offsets for constant
    encoded-view indexes, but only permits dynamic TMEM indexes for unencoded
    leading buffer dimensions (`MemDescIndexOpConversion`, lines 504-535).
  - TMEM `memdesc_subslice` lowers by `getTMemSubviewOffsetForLowering` (lines
    598-603).
  - TMEM `memdesc_reinterpret` is currently a raw base-pointer no-op in LLVM
    conversion (lines 641-643), so semantic coverage must come from earlier
    planner/query/view arithmetic.
- `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
  - Scale-root discovery walks forwarding, index, subslice, TMEM subslice,
    reshape, trans, and reinterpret chains (`getTMemScalesRootEncoding`, lines
    82-124).
  - `getTMemLdStDirectSupportTensorType` combines raw query, support query,
    candidate atom search, generic-compatible layouts, scale register-layout
    normalization, and row-plan overrides (lines 348-465).
  - `getTMemLoadReductionLayoutForMemDesc` has separate raw-query, M64 split-N,
    view-like, and row-plan paths for `ld.red` (lines 3511-3605).
  - `getTMemCopyPlans` and `getTMemCopySharedDescriptorPlanRealization` have
    distinct dense, `warpx2::01_23`, `warpx2::02_13`, `warpx4`, 4x256 refresh,
    source-row projection, instruction-column projection, and descriptor
    synthesis branches (lines 11903-12019 and 12881-13058).
- `lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp`
  - Replay lowering has dedicated leading-slice, replayable half-slice, and
    full-view matching, but intentionally excludes scales from these replay
    paths (lines 140-220 and 332-337).
  - The leading-slice store pattern rematerializes support loads, applies view
    transforms, splits, joins, and stores back through a support layout (lines
    1032-1090).
- `lib/Dialect/TritonNvidiaGPU/Transforms/CheckMatmulTwoCTAs.cpp`
  - `getModuleTwoCTAs` is derived once from explicit MMAv5 attributes and TMEM
    operand/result types; mismatches fail the pass, and the module attribute is
    then consumed by later ownership/barrier/lowering code (lines 52-78).

## Gap Inventory

| Gap | Current coverage found | Proposed minimal future probe |
| --- | --- | --- |
| Dynamic TMEM `memdesc_index` for encoded descriptor views remains a known hard boundary. LLVM conversion only supports dynamic index for unencoded leading buffers. | Covered as existing `FZ-20260421-0001` by Round 38/39/61 structural lanes; Round 61 structural reran `9 passed, 24 xfailed`. | Keep current xfail sentinels. Future non-repair probe should isolate dynamic encoded index after `reinterpret` plus after `slice` to verify the diagnostic remains the same bucket, not a silent fallback. |
| TMEM `memdesc_reinterpret` is a no-op at LLVM conversion, so correctness depends entirely on exact query/view arithmetic before lowering. | Descriptor-chain and rank-5 positives were covered by Round 59; FZ-0003 boundary by Round 55; lit/compiler breadth by Round 60/61. | Add a future runtime row that performs `bitcast/reinterpret -> reshape -> permute -> slice -> load`, then independently reads the original base with a different access atom. This targets pointer-no-op masking and exact pre-lowering query arithmetic. |
| Scale descriptor-view replay is intentionally excluded from `OptimizeTMemLayouts` replay paths; scale views instead rely on scale-root discovery and ld/st support planning. | Round 58 scale copy/ldst passed 41 rows; Round 59 TMA/TMEM lane covered scaled-copy commits and two-CTA MMAv5/TMA descriptor accumulator paths; Round 61 runtime covered shared-scale descriptor-view auto-copy. | Future adversarial row: two-CTA `TensorMemoryScalesLayout` view through `reinterpret` and higher-rank prefix, then one `ld/st` user plus one `tcgen05_mma_scaled` user. This maps to `getTMemScalesRootEncoding` across forwarding/view chains. |
| `ld.red` raw-query handling and M64 split-N canonicalization are covered broadly, but reinterpret-plus-ld.red is less direct than descriptor-chain coverage. | Round 54/57/59 covered M64 resource/diagnostic/opcode boundaries; Round 61 covered descriptor-chain `ld.red`; this lane reran one checked-in descriptor-chain `ld.red` row green. | Add future row: M64 `f32` descriptor chain with `bitcast`/reinterpret before `ld.red`, explicit `32x32b_splitn`, and a base load comparison to catch wrong row-plan selection. |
| Two-CTA ownership is well covered for plain/scaled MMAv5, copy, TMA/proxy, and high-CGA barriers, but mixed explicit two-CTA users in one module are mostly diagnostic/lit style rather than runtime. | Round 58 high-CGA ownership, Round 59 TMA/proxy/mbarrier, Round 61 runtime adversarial, and `CheckMatmulTwoCTAs` diagnostics cover the main paths. | Future lit-only negative: module with one explicit two-CTA scale TMEM operand and one one-CTA direct TMEM copy to ensure the pass reports the first/current source notes stably. Runtime is not needed for this pure consistency check. |
| `warpx2::02_13` two-CTA remains a clean unsupported frontier with source-column preservation requirements; `01_23` positives are covered. | Round 34/38/54/57/61 cover `warpx2` positives and clean unsupported rows; this lane reran the two-CTA `01_23` f32 row green. | Future probe should keep `02_13` as a clean diagnostic sentinel and add no repair. If discovery continues, vary only shared descriptor layout with a stable per-GPU cache to ensure no stale descriptor-plan cache changes the diagnostic. |
| 4x256 refresh-image copy path has planner branches distinct from ordinary dense copy. | Round 57 copy-view boundaries and Round 60/61 lit/codegen cover some 4x256 paths; runtime matrix has direct 4x256 refresh rows. | Future minimal row: 4x256 refresh with two-CTA parent view and a `memdesc_subslice` before copy, checking exact two-message opcode count and output. |
| Memdesc and AxisInfo interaction appears indirect for TMEM: TMEM view lowering advances opaque tensor-memory bases, while AxisInfo heavily influences pointer/load/store paths elsewhere. | No focused Round 55-61 runtime row found that intentionally perturbs pointer `tt.contiguity`/`tt.divisibility` metadata around TMEM source/global loads while also exercising TMEM view lowering. Existing tests incidentally use normal loads/stores. | Future runtime probe: load global input with deliberately different `max_contiguous`/`multiple_of` metadata, store through a TMEM descriptor view, then compare with a base-view readback. This should catch accidental dependence on pointer AxisInfo in TMEM operand preparation without changing backend code. |

## Coverage Overlap With Rounds 55-61

- Descriptor-view composition: Round 55 sharpened existing `FZ-20260421-0003`;
  Round 59 covered higher-rank/rank-5 descriptor chains; Round 61 covered
  frontend/structural xfails. Remaining gap is specifically reinterpret/view
  composition where LLVM lowering erases reinterpret.
- LinearLayout algebra: heavily covered indirectly by runtime matrix descriptor,
  scale, copy, and `ld.red` rows, plus Round 60/61 lit/codegen. Remaining gap
  is cross-surface composition rather than a missing single primitive.
- `getModuleTwoCTAs` / CTA ownership: Round 58 high-CGA, Round 59
  TMA/proxy/mbarrier, and Round 61 runtime cover the active ownership paths.
  Remaining gap is a small lit diagnostic for inconsistent mixed explicit users.
- Scale layouts: Round 58 and Round 61 cover positive scale ldst/copy/MMAv5
  paths. Remaining gap is two-CTA scale root discovery through reinterpret and
  higher-rank forwarding chains.
- `ld.red`: Round 54/57/61 cover hardware, software-reduce, descriptor-chain,
  M64, and clean-boundary rows. Remaining gap is `ld.red` after reinterpret.
- Copy `warpx2`/`warpx4`: Round 38/54/57/58/61 cover positive and clean
  unsupported rows. Remaining gap is 4x256 refresh through a two-CTA subview.
- Memdesc/AxisInfo: no direct TMEM-specific adversarial runtime row found in
  Rounds 55-61; propose one future probe that varies pointer metadata feeding a
  TMEM descriptor-view store/load.

## Classification

No new independent `FZ-*` bucket is assigned from this audit. The only concrete
untested surfaces found are future probe candidates, not observed crashes,
miscompiles, false unsupported diagnostics, or clean-boundary drift.

Backend repair remains deferred under the current fuzzing campaign rules.
