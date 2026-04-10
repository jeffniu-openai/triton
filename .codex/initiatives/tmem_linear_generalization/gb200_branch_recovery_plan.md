# GB200 Branch Recovery Plan

This file turns the completed GB200/NVIDIA CI census into the actionable
branch recovery backlog.

Scope is intentionally narrow:
- restore GB200 CI parity with the last merge-base against `origin/main`;
- keep reinterpret-contract rewrites and branch-added TMEM coverage visible,
  but separate them from old-mainline regressions; and
- track performance-only concerns for the end of the initiative rather than
  mixing them into the current correctness/debugging loop.

## Recovery Rules

- A failure belongs in the branch recovery backlog only if it is one of:
  - `REAL_NEW_ON_BRANCH_REGRESSION`
  - `BRANCH_ADDED_COVERAGE_RED`
  - `BRANCH_CHANGED_COVERAGE_RED`
- A failure does **not** belong in the branch recovery backlog if it is:
  - `PREEXISTING_ON_MERGE_BASE`
  - `STALE_NEGATIVE_OR_SUPPORT_BROADENED`
  - `REINTERPRET_CONTRACT_TEST_TO_REWRITE`
- If a failure looks cache-sensitive, reproduce it across a fresh process
  boundary before blaming `TRITON_CACHE_DIR`.
- Keep performance regressions and heuristic-quality issues in notes, but do
  not let them preempt correctness unless they also break functional CI.

## Exact Repro Conventions

- Single-nodeid rerun:
  - `CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/<cache> PYTHONPATH=python:. python3 -m pytest -s --tb=short -vv <nodeid>`
- Rerun a whole current-branch manifest:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/<cache> PYTHONPATH=python:. python3 - <<'PY'
from pathlib import Path
import pytest
nodeids = [line.strip() for line in Path('<manifest.txt>').read_text().splitlines() if line.strip()]
raise SystemExit(pytest.main(['-s', '--tb=short', '-vv', *nodeids]))
PY
```

- Merge-base check:
  - run the same command from `/root/code/triton-mergebase-ci` with a distinct
    cache directory

## Current Classification Summary

### Post-Census Repair Checkpoint

- The first representative TMEM exact bucket from the census is now repaired
  on the current worktree:
  - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-layout9-64-128-32x32b_splitn-8-16x32bx2]`
  - `python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]`
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[identity-identity-2-32x32b_splitn]`
- The fix was localized to `TensorMemoryUtils.cpp`: direct planning now uses
  the active physical TMEM layout when a zero-row-basis analyzed view already
  collapses to the logical shape, and the old row-zero direct-view repair
  shims are disabled in that path.
- The nearby one-CTA MMAv5 controls remain green:
  - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-32-warps2-8-False-True-acc_dtype0]`
  - `python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
- The remaining live representative branch-regression bucket is now narrower
  and appears producer-side:
  - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-1-64-512-32-2-float32-tensorfloat32]`
  - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
- The exact manifest `.txt` files still reflect the pre-fix census, so they
  remain useful for backlog shape but not as current pass/fail counts until a
  broader rerun refreshes them.

### Excluded From The Branch Recovery Backlog

- `python/test/unit/test_debug.py`
  - `PREEXISTING_ON_MERGE_BASE`
  - same exact `20` nodeids fail on merge-base
- `third_party/proton/test/test_profile.py`
  - `PREEXISTING_ON_MERGE_BASE`
  - same exact `11` nodeids fail on merge-base
- `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[legacy]`
  - `REINTERPRET_CONTRACT_TEST_TO_REWRITE`
- `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64_parent_layout[linear]`
  - `REINTERPRET_CONTRACT_TEST_TO_REWRITE`
- `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout[...]`
  - `STALE_NEGATIVE_OR_SUPPORT_BROADENED`

### Included In The Branch Recovery Backlog

1. Common M64 TMEM planner / packet-decomposition regression
- Why it comes first:
  - it matches the strongest cross-bucket PTX clue;
  - it hits both old-mainline tests and branch-added TMEM coverage; and
  - it is the best current common-cause theory for the branch-wide Blackwell
    wrong-code surface.
- Representative exact nodeids:
  - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-layout9-64-128-32x32b_splitn-8-16x32bx2]`
  - `python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]`
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[identity-identity-2-32x32b_splitn]`
  - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-1-64-512-32-2-float32-tensorfloat32]`
  - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
- Broader affected manifests / files:
  - `gb200_current_branch_test_gluon_group4_failures.txt`
  - `python/test/regression/test_cast_matmul.py`
  - `python/examples/gluon/03-matmul-multicta.py`
  - `python/test/gluon/test_lowerings.py`
  - merge-base-present shard-3 `test_core.py` / `test_fpsan.py` failure lists
- Current root-cause hypothesis:
  - the current row-plan-aware canonical-`M=64` family-selection /
    packet-decomposition path is collapsing a lifted support-band quotient bit
    into ordinary packet repetition
  - status update:
    - the direct split-N ld/st slice of this bucket is now fixed on exact
      reruns;
    - the remaining exact failures are the MMAv5 producer-side dot / matmul /
      warp-specialization family, which should now be debugged without
      reopening the recovered direct-planning path

2. Unsupported-row-anchor compile bucket on the new planner
- Why it is probably adjacent:
  - these failures hit `ttng.tmem_store` row anchors `32,64` and may be the
    compile-time sibling of the same structural planner mistake
- Representative exact nodeids:
  - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-8-64-128-32]`
  - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float32]`
- Broader affected coverage:
  - all three `test_tensor_descriptor_reshape_matmul[...]`
  - the remaining `test_simple_persistent_matmul[...]` failures

3. Branch-added TMEM runtime-matrix coverage
- Status:
  - whole file is branch-added
  - not every failure should be treated as a regression against main
- Why it still matters now:
  - it is the closest new coverage to the TMEM generalization work and is the
    fastest way to tell whether a core TMEM fix is actually general
- Primary current families:
  - `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[...]`
  - `test_tmem_runtime_matrix_ld_red_row_permuted_linear_layout[...]`
  - `test_tmem_runtime_matrix_ldst_scales_variant_sweep[...]`
  - `test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_geometry_sweep[...]`
  - `test_tmem_runtime_matrix_cp_no_scales[...]`

4. Broad Gluon lowering/backend regression shard
- Source:
  - `gb200_current_branch_test_gluon_group3_failures.txt`
- Main old-mainline proof:
  - merge-base full-file `python/test/gluon/test_lowerings.py` is green
  - merge-base-present shard-3 `test_core.py` / `test_fpsan.py` / layout-view
    exact lists are green
- Dominant families:
  - `test_reduce_layouts[...]`
  - `test_scan_layouts[...]`
  - `test_mma_scaled_tcgen05_copy[...]`
  - `test_tmem_reduction[...]`
  - `test_tmem_reduction_linear_layouts[...]`
- Current reading:
  - this likely contains downstream fallout from the same planner bug plus
    additional branch-added coverage changes

5. Example portability / launch-shape bucket
- Source:
  - `python/examples/gluon/02-convolution.py`
  - `python/examples/gluon/03-matmul-multicta.py`
- Classification:
  - `BRANCH_CHANGED_COVERAGE_RED`, but not the first correctness target
- Failure mode:
  - `OutOfResources` due to `262208` requested shared memory vs `232448`
    hardware limit
  - wrong-code in the multicta matmul example
- Merge-base status:
  - the exact current-branch example nodeids do not exist on merge-base
  - the merge-base full-file rerun is green (`130 passed, 14 skipped`)
- Priority:
  - after the core TMEM / matmul correctness surface is green

## Current Execution Order

1. Preserve the repaired direct split-N TMEM path and attack the remaining
   MMAv5 producer-side M64 bucket.
- Validation ladder:
  - keep the repaired direct split-N controls green:
    - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-layout9-64-128-32x32b_splitn-8-16x32bx2]`
    - `python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]`
    - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[identity-identity-2-32x32b_splitn]`
    - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-32-warps2-8-False-True-acc_dtype0]`
    - `python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
  - then debug the remaining producer-side exacts:
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-1-64-512-32-2-float32-tensorfloat32]`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
  - then the `8` exact `test_dot[...]` nodeids
  - then `python/test/unit/language/test_matmul.py`
  - then `python/test/unit/language/test_warp_specialization.py`
  - then `python/test/regression/test_cast_matmul.py`
  - then the nearest runtime-matrix families

2. Re-check the compile-time row-anchor bucket.
- If the first fix clears it, keep it folded into bucket 1.
- If not, treat it as the next planner legality / representability task.

3. Re-broaden into the branch-added runtime-matrix and shard-3 Gluon surfaces.
- Use `gb200_current_branch_test_gluon_group4_failures.txt`
- Then use the merge-base-present shard-3 manifests before the branch-added /
  changed-coverage shard-3 lists

4. Revisit reinterpret-contract tests only after the core backend is stable.
- Rewrite intent tests to explicit TMEM view APIs once the necessary reordered
  view coverage exists.

5. Leave performance tuning and heuristic cleanup to the end.
- Keep notes on codegen-family changes or slower/faster choices, but do not
  block correctness recovery on them unless the CI lane explicitly tests them.

## Cache / Flake Notes

- The current evidence does **not** support a simple on-disk cache-collision
  theory:
  - a known-bad TMEM split-N test followed by `test_block_m_64_mma[linear]`
    passes in the same pytest process;
  - the same two tests also behave the same across separate fresh processes
    while sharing a cache directory.
- Current interpretation:
  - some shard-only anomalies are more likely process/device contamination
    after bad kernels, or another missing invalidation/input in the runtime
    path, than a trivial `TRITON_CACHE_DIR` key collision.
- Root-cause plan for future work:
  1. preserve the exact failing order when a shard-only anomaly appears;
  2. replay it in one process, then across fresh processes with the same cache,
     then across fresh caches;
  3. only blame cache keys if the symptom survives the process boundary and
     remains tied to cache reuse;
  4. otherwise investigate bad-kernel aftermath, stream/device state leakage,
     or compiler global-state reuse.

## Performance Notes To Revisit Later

- `make test-microbenchmark` is green, so there is no immediate microbenchmark
  gate failure.
- The current branch recovery phase is functionality/correctness-first.
- Keep an eye on codegen-family selection changes while fixing TMEM, because
  they may later explain performance deltas even when correctness is restored.
