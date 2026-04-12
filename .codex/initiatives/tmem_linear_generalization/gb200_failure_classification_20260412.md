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
| lit tests with now-invalid TMEM copy/MMAv5 contracts | 5 files | Fixed at focused-lit scope on 2026-04-12 08:21 UTC |
| lit tests with stale FileCheck / expected-error / branch-local view text | 4 files | Stale tests |
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

Status, 2026-04-12 08:21 UTC: fixed at focused-lit scope. These tests fed
IR or descriptor types that are no longer valid under the supported TMEM
copy/MMAv5 contracts. They were not evidence that the new
clean verifier diagnostics were wrong; the tests needed to be rewritten to a
supported spelling or intentionally converted to negative tests.

Main-validity check, 2026-04-12:
- merge-base `triton-opt` accepts the original shallow lit pipelines for these
  files;
- pushing representative post-pass IR through the merge-base backend does not
  generate valid code;
- scales-copy cases fail or crash in backend lowering around unsupported
  `ttng.tmem_copy` to tensor-memory scales;
- the MMAv5 f32 shape from the NVWS tests fails merge-base backend lowering
  with `failed to find valid tcgen05.mma layout for operand A`;
- therefore these are main-invalid IR patterns that the current branch reports
  earlier and more cleanly, not branch regressions.

### Scales-copy descriptor-plan contract

Files:
- `Analysis/test-membar-ttng.mlir`
- `TritonGPU/proxy_fence_insertion.mlir`
- `TritonNvidiaGPU/mma_lowering.mlir`

Observed diagnostic:
- `ttng.tmem_copy` reports that the source shared layout maps to
  `tcgen05.copy.warpx4.32x128b`, but Triton cannot synthesize a compatible
  shared-memory descriptor plan for tensor memory scales.

Resolution:
- fixed by rewriting the pass tests to supported scales-copy forms;
- `Analysis/test-membar-ttng.mlir` and
  `TritonGPU/proxy_fence_insertion.mlir` now use
  descriptor-plan-compatible `shared_linear` source layouts;
- `TritonNvidiaGPU/mma_lowering.mlir` keeps shared-scale lowering coverage
  on supported `128x8` A/B scale tiles with `blockN=128`;
- focused lit validation passed for all three files.

### MMAv5 transposed-f32 shared-memory contract

Files:
- `NVWS/assign_stage_phase.mlir`
- `NVWS/aref-tmem-insertion.mlir`

Observed diagnostic:
- `ttng.tc_gen5_mma` reports that `tcgen05.mma` does not support transposed
  `float32` operands in shared memory.

Resolution:
- fixed by changing the scheduling-only IR to supported operand dtype/layouts;
- unscaled NVWS MMAv5 tests now use f16 shared operands with f32 accumulators;
- scaled e4m3 NVWS tests now use f8E4M3FN shared operands and a compatible
  f8 shared layout;
- focused lit validation passed for both NVWS files.

## Stale Tests

These appear to be test-text drift rather than actual compiler regressions or
API misuse.

### `TritonNvidiaGPU/ops.mlir`

Observed diagnostic:
- branch-local linear-TMEM `ttg.memdesc_subslice` result memdesc types no
  longer match the verifier's inferred type; the written result types use the
  older reduced `tensor_memory_linear` encoding instead of the current
  physical-layout-preserving view contract.

Main-validity check, 2026-04-12:
- merge-base cannot parse this exact IR because `#ttng.tensor_memory_linear`
  is branch-added;
- the relevant branch conversion tests were already updated by `ab8ff6e64` to
  keep the subview result's physical mapping equivalent to the parent/root
  descriptor;
- this is stale branch-local lit IR, not proof of a main-loose-verifier case.

Classification:
- stale branch test / dialect-contract text;
- update the handwritten IR to the current inferred descriptor-view type
  spelling, preserving the parent physical mapping where required.

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

1. Fix the four remaining stale lit tests.
2. Rerun full `make test-lit` after stale lit refresh; the five
   API/contract lit files already pass focused lit after their
   supported-spelling rewrite.
3. Debug the actual runtime bug cluster, starting with the smaller matmul /
   tensor-descriptor set before the 128 warp-specialization attention nodeids.
4. Fix `test_block_m_64_mma[legacy]` without reintroducing an xfail.
