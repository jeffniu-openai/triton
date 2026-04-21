# Round 49: checked-in test promotion candidates

Date: 2026-04-21 14:48 UTC

Branch/HEAD: `codex/tmem` at `c839da07a91c`

Scope: cataloging only. No backend/compiler code and no checked-in tests were
modified. This pass reviewed recent fuzz reports, reran exact disposable
reproducers and compact checked-in controls, and ranked future test-promotion
candidates.

## Build

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Reports reviewed

- `fuzz_dynamic_copy_sentinel_round45.md`
- `fuzz_proxy_mbarrier_ordering_round47.md`
- `fuzz_lit_ir_structural_round47.md`
- `fuzz_high_rank_half_views_round48.md`
- `fuzz_cache_process_order_round48.md`

Relevant classification summary:

- `FZ-20260421-0001`: dynamic/control-flow-yielded TMEM memdesc values reaching
  `ttng.tmem_copy` still leave `ttg.memdesc_index` live into LLVM conversion.
- `FZ-20260421-0014`: sequential independent cross-CTA mbarrier intervals still
  fail proxy-fence insertion.
- `FZ-20260421-0016`: unencoded or non-distributed tensor operands in TMEM
  verifier paths still assert instead of producing clean diagnostics.
- `FZ-20260421-0021`: unit-rank half-column `ld/st` descriptor views still hit a
  memdesc-shape / `MemDescType` invariant abort.
- Static copy, warpx2 copy, proxy-fence lit, and high-rank/half-row `ld/st`
  positives remain stable.

## Confirmation commands and results

Dynamic copy expected failure plus adjacent positive:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
    '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor[0]' \
    '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[01_23-0-index]'
```

Result: `1 failed, 1 passed in 4.04s`.

Failure signature:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
Pipeline failed while executing [`ConvertTritonGPUToLLVM` on 'builtin.module' operation]
RuntimeError: PassManager::run failed
```

The passing row is the branch-selected `warpx2::01_23` `ttng.tmem_copy`
positive, executed in the same pytest process after the dynamic linear-copy
failure.

Saved plain sequential-mbarrier reproducer:

```bash
/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_fz0014_plain_seq_round19_min_probe_branch_repro.mlir.make_llir.repro.mlir \
  --run-reproducer
```

Result: failed with the existing `FZ-20260421-0014` diagnostic:

```text
'tt.func' op could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
```

Copy/mbarrier disposable probe:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_copy_mbarrier_composition_round14_probe.py
```

Result: `2 failed, 6 passed in 12.68s`.

Failed rows:

- `test_ai_001_two_independent_twocta_no_scales_direct_regions`
- `test_ai_002_two_independent_twocta_no_scales_descriptor_chain_regions`

Both failed with the same `FZ-20260421-0014` proxy-fence insertion diagnostic.
The six passing rows keep useful copy/proxy ordering contrasts green.

Relayout verifier crash:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12 && \
  ninja triton-opt && \
  lit -v test/Conversion/relayout_tritongpu.mlir
```

Result: `1 failed`.

Failure signature:

```text
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
mlir::triton::gpu::TritonGPUDialect::toLinearLayout(...)
mlir::triton::nvidia_gpu::computeTMemLdStEncodingInfoImpl(...)
mlir::triton::nvidia_gpu::verifyTMEMOperand(...)
mlir::triton::nvidia_gpu::TMEMAllocOp::verify()
FileCheck error: '<stdin>' is empty.
```

Unit-rank half-column `ld/st` descriptor view:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
ROUND39_CASE='{"case_id":"ldst-unit_rank_half_col_upper-identity_identity-auto","family":"ldst","m":128,"n":128,"row_kind":"identity","col_kind":"identity","view_kind":"unit_rank_half_col_upper","variant":"auto","red_op":"min","seed":121}' \
  python /tmp/tmem_high_rank_chain_shapes_round39/worker.py
```

Result: process aborted with existing `FZ-20260421-0021` signature:

```text
TMEM layout shape must be bounded by the memdesc shape and allocShape.
shape = 128, 1, 128, allocShape = 128, 1, 128, layoutShape = 128, 128
Assertion `succeeded( ConcreteT::verifyInvariants(...))' failed.
Fatal Python error: Aborted
```

Checked-in runtime positives:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short \
    'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_indexed_view[128x128b_f32-torch_dtype0-128-4-smem_layout0-1-tcgen05.cp.cta_group::1.128x128b]' \
    'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive[f32-torch_dtype0]' \
    'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_rank5_unit_parent_n256_roundtrip[f32-torch_dtype0-single_identity-single-identity-128-1-auto-32x32b.x64.b32]'
```

Result: `3 passed in 2.64s`.

Checked-in proxy-fence lit positive:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12 && \
  ninja triton-opt && \
  lit -v test/TritonGPU/proxy_fence_insertion.mlir
```

Result: `1 passed`.

Disposable rank-5 half-row `ld/st` positive:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
ROUND39_CASE='{"case_id":"ldst-rank5_half_row_lower-identity_identity-auto","family":"ldst","m":128,"n":128,"row_kind":"identity","col_kind":"identity","view_kind":"rank5_half_row_lower","variant":"auto","red_op":"min","seed":122}' \
  python /tmp/tmem_high_rank_chain_shapes_round39/worker.py
```

Result:

```json
{"status":"pass","ptx_ld":["tcgen05.ld.sync.aligned.32x32b.x128.b32","tcgen05.ld.sync.aligned.32x32b.x128.b32","tcgen05.ld.sync.aligned.32x32b.x128.b32"]}
```

## Ranked promotion candidates

1. **Dynamic linear `ttng.tmem_copy` descriptor xfail (`FZ-20260421-0001`).**
   Promote a structural or runtime xfail based on the branch-selected linear
   copy row from `/tmp/tmem_copy_dynamic_descriptor_round40_probe.py`. This is
   the highest-value candidate because the checked-in structural fuzzer already
   covers dynamic memdesc load-only behavior, but not a copy-specific dynamic
   descriptor consumer. Pair it with a branch-selected `warpx2` copy positive
   when practical, because the same-process contrast stayed green.

2. **Sequential mbarrier proxy-fence xfail (`FZ-20260421-0014`).** Promote the
   saved plain sequential-mbarrier reproducer as the smallest compiler-only
   sentinel, and later add a copy variant if test cost permits. The plain
   reproducer isolates the proxy-fence interval planner without TMEM copy
   noise; the copy/mbarrier probe remains useful as broader evidence.

3. **Unencoded TMEM operand verifier clean-negative (`FZ-20260421-0016`).**
   Promote a lit-level verifier test once the expected behavior is a diagnostic
   rather than an assertion. `Conversion/relayout_tritongpu.mlir` is currently a
   good red guardrail, but a smaller dedicated clean-negative would better
   protect `verifyTMEMOperand` / `toLinearLayout` from crashing on absent or
   non-distributed encodings.

4. **Unit-rank half-column `ld/st` descriptor-view xfail
   (`FZ-20260421-0021`).** Promote a minimized runtime or compile-only sentinel
   around `unit_rank_half_col_upper` after turning the worker case into a stable
   checked-in helper. This captures a real encoded descriptor-view shape that
   reaches a `MemDescType` invariant abort, distinct from the unencoded verifier
   crash in `FZ-0016`.

5. **Stable checked-in positives around the red sentinels.** Keep or add compact
   positives for static linear copy, static warpx2 copy, rank-5 descriptor
   `ld/st`, proxy-fence lit, and rank-5 half-row `ld/st`. These should be run
   near future xfails so the suite distinguishes true unsupported/failing cases
   from broad regressions in static copy, high-rank views, or proxy-fence
   ordinary behavior.

## Promotion notes

- Do not promote the `/tmp` worker files verbatim. Convert the selected cases
  into stable checked-in helpers or lit inputs with exact expected diagnostics.
- Expected-failure tests should be strict. The useful signal is a specific
  known diagnostic, not simply "any compile failed".
- For `FZ-0014`, the plain mbarrier-only lit/reproducer candidate is the cleanest
  first promotion; the two-region copy probe is a second-tier broader sentinel.
- For `FZ-0021`, keep the positive rank-5 half-row contrast nearby because it
  proves high-rank descriptor-view `ld/st` is not generally broken.
- Backend repair remains deferred per the active fuzzing campaign.
