# GB200 Failure Classification - 2026-04-12

This classifies the branch-new failures from the full GB200-only
`integration-tests-nvidia` sweep recorded at source checkpoint `cb76c31a0`.

Artifacts:
- current sweep: `/tmp/gb200-ci-current-20260412-032422`
- merge-base checkout: `/root/code/triton-mergebase-ci`
- merge-base SHA: `11ee1144a737006921231bbd3386c187812c38e1`

## Summary

| Bucket | Count | Classification |
| --- | ---: | --- |
| `python/test/unit` runtime failures | 162 nodeids | Actual compiler/runtime bugs |
| `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]` | 1 nodeid | Actual compiler/runtime bug |
| lit tests with now-invalid TMEM copy/MMAv5/view contracts | 6 files | Tests require API/contract update |
| lit tests with stale FileCheck / expected-error text | 3 files | Stale tests |
| Proton cudagraph / periodic flushing | 11 nodeids | Preexisting on merge-base, ignore for branch recovery |

## Actual Bugs

These fail by wrong numerical output on current branch and pass on merge-base.
They do not appear to be `_reinterpret`-contract tests or stale text checks.

### `python/test/unit` - 162 nodeids

Manifest:
- `gb200_branch_new_20260412_unit_main_failures.txt`

Breakdown:
- `python/test/unit/language/test_matmul.py`: 27 nodeids
  - `test_simple_matmul`: 24
  - `test_simple_persistent_matmul`: 3
- `python/test/unit/language/test_tensor_descriptor.py`: 7 nodeids
  - `test_make_tensor_descriptor_matmul`: 2
  - `test_host_tensor_descriptor_matmul`: 2
  - `test_tensor_descriptor_reshape_matmul`: 3
- `python/test/unit/language/test_warp_specialization.py`: 128 nodeids
  - `test_warp_specialize_attention_forward`: 64
  - `test_warp_specialize_attention_persistent_forward`: 64

Observed symptoms:
- matmul / tensor descriptor failures are broad wrong-output mismatches,
  commonly around 50% of elements;
- warp-specialization attention failures are exact equality failures against
  the non-warp-specialized reference path;
- all 162 selected nodeids pass on merge-base.

Classification:
- actual compiler/runtime regression;
- likely a shared MMAv5 / descriptor / `block_m=64` planning issue rather than
  162 independent test problems;
- root-cause the shared lowering or planner regression before editing tests.

### `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`

Manifest:
- `gb200_branch_new_20260412_gluon_failures.txt`

Observed symptoms:
- current branch:
  - `legacy` fails with wrong output;
  - `linear` passes;
- merge-base legacy test passes.

Classification:
- actual compiler/runtime bug in legacy M64 MMAv5 producer-family handling;
- this is not a stale test and should not be re-xfailed;
- fix by making legacy M64 layout sugar normalize into the same effective
  producer-visible physical family as the working TMEM-linear case.

## Tests Requiring API / Contract Updates

These tests feed IR or descriptor types that are no longer valid under the
supported TMEM copy/MMAv5/view contracts. They are not evidence that the new
clean verifier diagnostics are wrong; the tests need to be rewritten to a
supported spelling or intentionally converted to negative tests.

### Scales-copy descriptor-plan contract

Files:
- `Analysis/test-membar-ttng.mlir`
- `TritonGPU/proxy_fence_insertion.mlir`
- `TritonNvidiaGPU/mma_lowering.mlir`

Observed diagnostic:
- `ttng.tmem_copy` reports that the source shared layout maps to
  `tcgen05.copy.warpx4.32x128b`, but Triton cannot synthesize a compatible
  shared-memory descriptor plan for tensor memory scales.

Classification:
- API/contract update required;
- these pass tests were relying on a now-rejected scales-copy setup to reach
  downstream membar / proxy-fence / MMA-lowering checks;
- update the source shared layout or descriptor construction to one of the
  supported scales-copy forms, or move the case into a clean-negative test if
  the unsupported shape is the point.

### MMAv5 transposed-f32 shared-memory contract

Files:
- `NVWS/assign_stage_phase.mlir`
- `NVWS/aref-tmem-insertion.mlir`

Observed diagnostic:
- `ttng.tc_gen5_mma` reports that `tcgen05.mma` does not support transposed
  `float32` operands in shared memory.

Classification:
- API/contract update required;
- these NVWS tests use now-invalid f32 shared-memory operand layouts while
  trying to test scheduling / aref behavior;
- update the test IR to use supported operand dtype/layouts, or make the
  invalid f32 case an expected diagnostic if that is the intended coverage.

### `memdesc_subslice` result-type contract

File:
- `TritonNvidiaGPU/ops.mlir`

Observed diagnostic:
- `ttg.memdesc_subslice` result memdesc types no longer match the verifier's
  inferred type; the written result types use the older reduced
  `tensor_memory_linear` encoding instead of the current physical-layout view
  contract.

Classification:
- API/dialect contract update required;
- update the handwritten IR to the current inferred descriptor-view type
  spelling, preserving the parent physical mapping where required.

## Stale Tests

These appear to be test-text drift rather than actual compiler regressions or
API misuse.

### `TritonNvidiaGPU/invalid.mlir`

Observed diagnostic:
- `tmem_copy_scales_descriptor_family_clean_unsupported` now emits the clean
  unsupported scales-copy diagnostic, but the test lacks the matching
  `expected-error` / `expected-note` annotations.

Classification:
- stale invalid-test annotation;
- add the expected diagnostic lines. The behavior is the intended clean
  negative.

### `TritonGPU/pipeline-loop-nest.mlir`

Observed diagnostic:
- FileCheck expects a bare `ttng.tmem_alloc`, but the output contains the same
  allocation with current TMEM physical-layout attributes.

Classification:
- stale FileCheck;
- loosen/update checks to accept the current `ttng.tmem_physical_layout`
  attribute spelling.

### `TritonGPU/pipeline-lower-loop.mlir`

Observed diagnostic:
- FileCheck expects bare `ttng.tmem_alloc` text, while the output includes
  current attributes such as `ttng.tmem_mmav5_accumulator_root`.

Classification:
- stale FileCheck;
- update checks to allow the current alloc attributes rather than changing
  lowering behavior.

## Preexisting / Ignored

Manifest:
- `gb200_preexisting_20260412_proton_main_failures.txt`

Classification:
- `PREEXISTING_ON_MERGE_BASE`;
- all 11 Proton cudagraph / periodic flushing failures reproduce on
  merge-base with the same `-n 8` CI command;
- leave out of branch recovery unless a future run shows a nodeid that does
  not fail on merge-base.

## Recommended Recovery Order

1. Fix stale lit tests first; this should reduce `test-lit` noise quickly.
2. Rewrite the API/contract-update lit files to supported TMEM copy/MMAv5/view
   spellings or convert them to explicit negatives where appropriate.
3. Debug the actual runtime bug cluster, starting with the smaller matmul /
   tensor-descriptor set before the 128 warp-specialization attention nodeids.
4. Fix `test_block_m_64_mma[legacy]` without reintroducing an xfail.
