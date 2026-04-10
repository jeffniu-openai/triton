# GB200 Failure Manifest

This file is the durable index for the exact GB200 current-branch failure
lists generated during the 2026-04-10 census. The `.txt` files beside it are
meant to be machine-usable artifacts, not hand-maintained prose.

Use this file together with:
- `gb200_nvidia_ci_inventory.md` for the executed lane state and classification
- `gb200_branch_recovery_plan.md` for the prioritized branch-caused backlog

## How To Use

- Rerun a single exact failure:
  - `PYTHONPATH=python:. python3 -m pytest -q -rf --tb=no <nodeid>`
- Rerun an entire generated list on the current branch:

```bash
PYTHONPATH=python:. python3 - <<'PY'
from pathlib import Path
import pytest
nodeids = [line.strip() for line in Path('<manifest.txt>').read_text().splitlines() if line.strip()]
raise SystemExit(pytest.main(['-q', '-rf', '--tb=no', *nodeids]))
PY
```

- Rerun a generated list on merge-base:
  - run the same command from `/root/code/triton-mergebase-ci`
  - keep a distinct `TRITON_CACHE_DIR`
  - set `CUDA_VISIBLE_DEVICES=<gpu>` explicitly

## Current-Branch Exact Failure Lists

### Refreshed Current-Head Unit Reduction

- [gb200_current_branch_test_unit_matmul_refresh_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_unit_matmul_refresh_failures.txt)
  - `0` nodeids
  - source:
    - refreshed after the full green rerun at `78196b4e4`
  - current interpretation:
    - the current branch has no remaining GB200 `test-unit`
      `test_matmul.py` failures
- [gb200_current_branch_test_unit_tensor_descriptor_refresh_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_unit_tensor_descriptor_refresh_failures.txt)
  - `0` nodeids
  - source:
    - refreshed after the full green rerun at `78196b4e4`
  - current interpretation:
    - the current branch has no remaining GB200 `test-unit`
      `test_tensor_descriptor.py` failures
- [gb200_current_branch_test_unit_warp_specialization_refresh_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_unit_warp_specialization_refresh_failures.txt)
  - `0` nodeids
  - source:
    - refreshed current-head
      `python/test/unit/language/test_warp_specialization.py`
    - rerun result:
      - `1599 passed, 202 skipped`
  - current interpretation:
    - the old populated warp-specialization manifest is now historical only
- [gb200_current_branch_test_unit_rowanchor_refresh_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_unit_rowanchor_refresh_failures.txt)
  - `0` nodeids
  - source:
    - concatenation of the now-empty refreshed unit manifests above
  - current interpretation:
    - the current branch has no remaining GB200 `test-unit` exact failures

- [gb200_current_branch_test_unit_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_unit_failures.txt)
  - `0` nodeids
  - source:
    - fresh full `make NUM_PROCS=24 test-unit` rerun at `78196b4e4`
    - green sub-lanes:
      - main `python/test/unit`
      - `test_debug.py`
      - `python/triton_kernels/tests`
      - `python/tutorials/06-fused-attention.py`
      - instrumentation test
      - plugin tests
- [gb200_current_branch_test_unit_xml_failures_2026-04-10.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_unit_xml_failures_2026-04-10.txt)
  - `2098` nodeids
  - source:
    - near-complete broad `python/test/unit` rerun with `--junitxml`
      written to `/tmp/test-unit-stage1-current.xml`
  - current interpretation:
    - historical only
    - do not use this as the current branch unit red list anymore
    - all `2098` exact nodeids are present on merge-base and the exact
      merge-base rerun is fully green/skip:
      - `2083 passed, 15 skipped`
    - do **not** read this as `2098` independent fresh root-cause buckets on
      the current branch; representative fresh reruns from the tail files
      pass, so much of this list is branch-local xdist/process fallout after a
      smaller number of primary MMAv5/TMEM failures
- [gb200_current_branch_test_gluon_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_gluon_group3_failures.txt)
  - `1160` nodeids
  - source:
    - `python3 -m pytest -q -rf --tb=no --splits 4 --group 3 -k 'not test_tmem_subslice_block_m_64 and not test_tmem_subslice_block_m_64_parent_layout' python/test/gluon/ python/tutorials/gluon/`
- [gb200_current_branch_test_gluon_group4_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_gluon_group4_failures.txt)
  - `686` nodeids
  - source:
    - `python3 -m pytest -q -rf --tb=no --splits 4 --group 4 -k 'not test_tmem_subslice_block_m_64 and not test_tmem_subslice_block_m_64_parent_layout' python/test/gluon/ python/tutorials/gluon/`
- [gb200_current_branch_test_core_branch_added_descriptor_chain_refresh_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_core_branch_added_descriptor_chain_refresh_failures.txt)
  - `12` nodeids
  - source:
    - isolated rerun of the `26` descriptor-chain exacts extracted from the
      branch-added / branch-changed focused `test_core.py` shard reduction
  - current interpretation:
    - this is the current independent descriptor-chain runtime bug bucket
    - fresh rerun at `c6990c1e9`:
      - `12 failed in 3.73s`
    - all `12` exacts fail cleanly with
      `RuntimeError: CUDA error: misaligned address`
    - the other `14` descriptor-chain exacts from the contaminated shard pass
      in isolation and should not stay in the live red list
- [gb200_current_branch_test_core_branch_added_splitn_expectation_refresh_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_core_branch_added_splitn_expectation_refresh_failures.txt)
  - `0` nodeids
  - source:
    - refreshed after the canonical `M=64` split-N lowering fix
  - current interpretation:
    - this stale/opinionated split-N expectation exact is now green too
- [gb200_current_branch_test_tmem_runtime_matrix_splitn_rowcol_refresh_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_tmem_runtime_matrix_splitn_rowcol_refresh_failures.txt)
  - `0` nodeids
  - source:
    - refreshed after the canonical `M=64` split-N lowering fix
  - current interpretation:
    - the old dominant split-N runtime bucket is now fully green
    - keep the old count only in the log/history, not in the live red set
- [gb200_current_branch_test_tmem_runtime_matrix_warpx2_candidate_refresh_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_tmem_runtime_matrix_warpx2_candidate_refresh_failures.txt)
  - `0` nodeids
  - source:
    - refreshed after the non-surjective `[128, 4]` raw-query/direct-view
      preservation fix
  - current interpretation:
    - this manifest is now empty
    - the paired frontend parse exact is green again too
    - do not treat the older two-nodeid content as a live bucket anymore
- [gb200_current_branch_test_regression_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_regression_failures.txt)
  - `234` nodeids
  - source:
    - `python3 -m pytest -q -rf --tb=no python/test/regression/test_cast_matmul.py`
- [gb200_current_branch_examples_gluon_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_examples_gluon_failures.txt)
  - `62` nodeids
  - source:
    - `python3 -m pytest -q -rf --tb=no python/examples/gluon/`
- [gb200_current_branch_examples_convolution_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_examples_convolution_failures.txt)
  - `48` nodeids
  - source:
    - isolated current-head `python/examples/gluon/02-convolution.py`
  - current interpretation:
    - preferred exact manifest for the convolution half of the examples lane
    - fresh current-head rerun at `c6990c1e9`:
      - `48 failed in 8.38s`
    - fresh merge-base rerun at `7f61ac734`:
      - `48 passed in 7.30s`
- [gb200_current_branch_examples_multicta_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_examples_multicta_failures.txt)
  - `14` nodeids
  - source:
    - isolated current-head `python/examples/gluon/03-matmul-multicta.py`
  - current interpretation:
    - preferred exact manifest for the multicta matmul half of the examples
      lane
    - fresh current-head rerun at `c6990c1e9`:
      - `14 failed, 68 passed, 14 skipped in 65.24s`
    - fresh merge-base rerun at `7f61ac734`:
      - `82 passed, 14 skipped in 37.96s`
- [gb200_current_branch_test_proton_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_proton_failures.txt)
  - `11` nodeids
  - source:
    - `python3 -m pytest -q -rf --tb=no third_party/proton/test/test_profile.py`

## Classified Recovery Split Lists

- [gb200_preexisting_test_unit_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_preexisting_test_unit_failures.txt)
  - `20` nodeids
  - meaning:
    - the `python/test/unit/test_debug.py` exact failures that are also red on
      merge-base
- [gb200_branch_recovery_test_unit_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_branch_recovery_test_unit_failures.txt)
  - `0` nodeids
  - meaning:
    - the current branch has no remaining GB200 `test-unit` exacts in the
      branch-recovery backlog
- [gb200_branch_recovery_test_regression_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_branch_recovery_test_regression_failures.txt)
  - `234` nodeids
  - meaning:
    - the exact `test_cast_matmul.py` failures; all exist and pass on
      merge-base, so the whole manifest is branch-caused recovery work
- [gb200_branch_changed_examples_gluon_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_branch_changed_examples_gluon_failures.txt)
  - `62` nodeids
  - meaning:
    - the exact current-branch `python/examples/gluon/` failures
  - note:
    - these nodeids do not exist on merge-base as exact parametrized tests, so
      this bucket is tracked as branch-changed coverage rather than an
      identical old-mainline failure set

## Current Underlying-Bucket Refinement

- The live current-branch manifests above are no longer interpreted as a flat
  list of unrelated tests:
  - `gb200_current_branch_test_tmem_runtime_matrix_splitn_rowcol_refresh_failures.txt`
    - now empty after the canonical `M=64` split-N lowering symmetry fix
  - `gb200_current_branch_test_tmem_runtime_matrix_warpx2_candidate_refresh_failures.txt`
    - now empty after the non-surjective `[128, 4]` raw-query/direct-view fix
  - `gb200_current_branch_test_core_branch_added_descriptor_chain_refresh_failures.txt`
    - at least two subfamilies:
      - packed `linear_m64_*` launch-fault cases
      - mixed-layout scalar-family access-mapping cases
- Use the manifests as machine-readable rerun inputs, but use
  `gb200_nvidia_ci_inventory.md` and `gb200_branch_recovery_plan.md` for the
  current trace-backed recovery ordering. The live TMEM/compiler recovery
  bucket is now the `12`-nodeid descriptor-chain manifest above.

## Shard-3 Merge-Base Reduction Lists

These lists were derived from the current-branch shard-3 exact nodeids by
checking whether the failing function name exists on the merge-base tree.

- [gb200_mergebase_existing_test_gluon_core_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_mergebase_existing_test_gluon_core_group3_failures.txt)
  - `203` nodeids
  - meaning:
    - current-branch shard-3 `test_core.py` failures whose function already
      exists on merge-base
- [gb200_branch_added_or_renamed_test_gluon_core_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_branch_added_or_renamed_test_gluon_core_group3_failures.txt)
  - `90` nodeids
  - meaning:
    - current-branch shard-3 `test_core.py` failures in branch-added or renamed
      coverage
- [gb200_mergebase_existing_test_gluon_fpsan_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_mergebase_existing_test_gluon_fpsan_group3_failures.txt)
  - `60` nodeids
- [gb200_branch_added_or_renamed_test_gluon_fpsan_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_branch_added_or_renamed_test_gluon_fpsan_group3_failures.txt)
  - `12` nodeids
- [gb200_mergebase_existing_test_gluon_layout_format_view_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_mergebase_existing_test_gluon_layout_format_view_group3_failures.txt)
  - `1` nodeid
- [gb200_branch_added_or_renamed_test_gluon_frontend_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_branch_added_or_renamed_test_gluon_frontend_group3_failures.txt)
  - `1` nodeid

After that first split, the merge-base-existing lists were reduced once more by
collecting the merge-base test files and comparing exact nodeids:

- [gb200_mergebase_present_test_gluon_core_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_mergebase_present_test_gluon_core_group3_failures.txt)
  - `185` nodeids
  - exact current-branch shard-3 `test_core.py` failures that also exist as
    exact nodeids on merge-base
- [gb200_mergebase_missing_test_gluon_core_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_mergebase_missing_test_gluon_core_group3_failures.txt)
  - `18` nodeids
  - meaning:
    - function exists on merge-base, but the exact current-branch parametrized
      nodeid does not
- [gb200_mergebase_present_test_gluon_fpsan_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_mergebase_present_test_gluon_fpsan_group3_failures.txt)
  - `38` nodeids
- [gb200_mergebase_missing_test_gluon_fpsan_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_mergebase_missing_test_gluon_fpsan_group3_failures.txt)
  - `22` nodeids
- [gb200_mergebase_present_test_gluon_layout_format_view_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_mergebase_present_test_gluon_layout_format_view_group3_failures.txt)
  - `1` nodeid
- [gb200_mergebase_present_test_gluon_lowerings_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_mergebase_present_test_gluon_lowerings_group3_failures.txt)
  - `793` nodeids
  - meaning:
    - every current-branch shard-3 `test_lowerings.py` failure also exists as
      the same exact nodeid on merge-base
- [gb200_mergebase_missing_test_gluon_lowerings_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_mergebase_missing_test_gluon_lowerings_group3_failures.txt)
  - `0` nodeids

## Notes

- These manifests reflect the current branch tip at the time of generation.
- The refreshed `3 + 3 + 128 = 134` unit manifests above are now the primary
  current-head unit inventory for the latest dirty MMAv5 row-plan propagation
  checkpoint.
- The focused branch-added `test_core.py` tail is now reduced too:
  - the `21` `test_tmem_linear_roundtrip_variant_sweep[...]` exacts from the
    contaminated shard all pass in clean isolation; and
  - the independent live surface is the `12`-nodeid descriptor-chain manifest
    plus the single split-N PTX-expectation node above.
- The focused branch-added runtime-matrix tail is now reduced too:
  - the dominant live bucket is the `208`-nodeid split-N row/col-permuted
    manifest above;
  - the rest of the earlier `413`-nodeid runtime-matrix reduction is almost
    entirely green now; and
  - only two warpx2 candidate-positive exacts remain outside the split-N
    bucket.
- The older `182`-nodeid unit manifest is still useful as the first reduced
  branch-recovery slice, but the newer `2098`-nodeid XML manifest is the
  current broad CI-like unit inventory.
- The unit XML and regression manifests are now historical with respect to the
  current dirty MMAv5 root-row-plan fix:
  - they still prove branch-vs-merge-base classification; but
  - they no longer reflect current-head counts for
    `python/test/regression/test_cast_matmul.py`; and
  - they overcount the current unit surface until a post-fix rerun refreshes
    them.
- Fresh isolated current-branch reruns of representative exact nodeids from
  the XML-only tail files all pass:
  - `python/test/unit/language/test_standard.py::test_maximum_minium[maximum-int32]`
  - `python/test/unit/language/test_random.py::test_randint[10-0-int32-False]`
  - `python/test/unit/runtime/test_cache.py::test_reuse`
  - `python/test/unit/runtime/test_blaslt.py::test_blaslt[float8_e4m3fn-16-16-16]`
  - `python/test/unit/runtime/test_autotuner.py::test_kwargs[False]`
  - `python/test/unit/runtime/test_launch.py::test_metadata`
  - `python/test/unit/runtime/test_bindings.py::test_python_func_in_visit_call`
  - `python/test/unit/tools/test_triton_to_gluon.py::test_simple_matmul[4-128-128-64-1-float32-float16]`
  - use that evidence together with the merge-base exact-green result to treat
    the XML tail mainly as branch-local fallout, not as the primary fix order
- Post-census status update:
  - after the later `TensorMemoryUtils.cpp` direct-planning fix, the
    representative exact split-N TMEM nodeids that motivated the original
    group-4 TMEM story now pass cleanly again:
    - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-layout9-64-128-32x32b_splitn-8-16x32bx2]`
    - `python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]`
    - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[identity-identity-2-32x32b_splitn]`
  - do not treat the current shard/group manifest counts as fully current
    until those manifests are regenerated from a broader rerun.
- The current `.txt` manifests normalize `pytest -q -rf` summary lines back to
  real nodeids by stripping any appended ` - AssertionError...` /
  ` - RuntimeError...` suffixes.
- Branch-vs-main classification is now complete for most of the exact GB200
  surface; the remaining work is using those classifications to choose and fix
  the first core compiler buckets.
- The examples bucket is the main place where the distinction between
  exact-old-mainline and branch-changed coverage matters:
  - the exact current-branch example nodeids are absent on merge-base; but
  - the merge-base full-file rerun is green, so the branch-changed example
    coverage is still real branch recovery work.
- When a manifest corresponds to a preexisting-on-main bucket
  (`test_debug.py`, Proton), keep it for census completeness but do not treat
  it as branch recovery work.
- Proton is included here for census completeness even though it is already
  known to be preexisting on merge-base.

## 2026-04-10 Partial Overnight Refresh

- Refreshed current-head shard manifests at `e70a3aa09`:
  - `gb200_current_branch_test_gluon_group3_e70a3aa09_failures.txt`
    - `1252` exact nodeids
  - `gb200_current_branch_test_gluon_group4_e70a3aa09_failures.txt`
    - `686` exact nodeids
- Exact merge-base split for the refreshed shard `3 / 4` manifest:
  - `gb200_mergebase_present_test_gluon_group3_e70a3aa09_failures.txt`
    - `967` exact old-mainline nodeids
  - `gb200_mergebase_missing_test_gluon_group3_e70a3aa09_failures.txt`
    - `285` exact branch-added / branch-changed nodeids
- Important current reading:
  - shard `4 / 4` is unchanged from the older manifest and remains entirely
    branch-added `python/test/gluon/test_tmem_runtime_matrix.py` coverage
  - shard `3 / 4` now has a durable branch-vs-main split, but it still needs
    isolated reruns before the raw `1252` nodeids are treated as independent
    bugs

## 2026-04-10 Additional Isolated Reductions

- Fresh merge-base full-file proofs:
  - `python/test/unit/language/test_matmul.py`
    - `761 passed, 4780 skipped`
  - `python/test/unit/language/test_warp_specialization.py`
    - `1599 passed, 202 skipped`
- New isolated current-head exact manifests:
  - `gb200_current_branch_test_core_group3_focus_e70a3aa09_failures.txt`
    - `332` exact nodeids from the dominant shard-3 `test_core.py` TMEM/MMA
      slice
  - `gb200_mergebase_present_test_core_group3_focus_e70a3aa09_failures.txt`
    - `146` exact old-mainline nodeids inside that focused core slice
  - `gb200_mergebase_missing_test_core_group3_focus_e70a3aa09_failures.txt`
    - `186` branch-added / branch-changed nodeids inside that focused core
      slice
  - `gb200_current_branch_test_fpsan_isolated_e70a3aa09_failures.txt`
    - `1` exact nodeid
  - `gb200_current_branch_test_layout_frontend_isolated_e70a3aa09_failures.txt`
    - `1` exact nodeid
  - `gb200_current_branch_test_tmem_runtime_matrix_focus_e70a3aa09_failures.txt`
    - `413` exact nodeids from the dominant branch-added runtime-matrix slice
- Important current reading:
  - the focused shard-3 core slice fails independently, so that bucket is
    real and no longer just a poisoned-context suspicion
  - the isolated FPSAN tail is down to one branch-changed exact:
    - `python/test/gluon/test_fpsan.py::test_tcgen05_mma_scaled[linear_identity-acc_layout1-e4m3-e2m1]`
  - `python/test/gluon/test_layout_format_view.py` is green in isolation; the
    remaining isolated non-core shard-3 tail is the single branch-added
    frontend exact
  - the focused runtime-matrix slice fails independently with `413` exact
    failures, so the dominant branch-added group-4 families are also real

## 2026-04-10 Inventory Closure

- Final isolated current-head completions:
  - `python/triton_kernels/tests`
    - `2377 passed, 3444 skipped`
  - `python/test/gluon/test_lowerings.py`
    - `4937 passed, 512 skipped`
- Implication:
  - the raw shard-3 `test_lowerings.py` failures are fully explained as
    fallout and should not be treated as a live independent bucket
  - the remaining independently red `test-gluon` surface is now:
    - focused `test_core.py` exact manifest
    - focused `test_tmem_runtime_matrix.py` exact manifest
    - one `test_fpsan.py` exact
    - one `test_frontend.py` exact
    - the split example manifests
- Final example classification for this pass:
  - `python/examples/gluon/02-convolution.py`
    - current branch:
      - `48` exact failures
      - representative symptom:
        `OutOfResources` (`262208` requested vs `232448` hardware limit)
    - merge-base full-file rerun:
      - `48 passed`
  - `python/examples/gluon/03-matmul-multicta.py`
    - current branch:
      - `14` exact failures
      - representative symptom:
        `9600 / 20000` mismatches (`48.0%`)
    - merge-base full-file rerun:
      - `82 passed, 14 skipped`
