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

### Latest Full GB200 NVIDIA CI Sweep (2026-04-12 05:24 UTC)

- Validated checkpoint:
  - `cb76c31a0`
- Branch-caused recovery backlog from this sweep:
  - `test-lit`: `9` branch-new lit failures, all green on merge-base;
  - `python/test/unit`: `162` branch-new runtime correctness failures:
    - `27` matmul / persistent matmul exacts;
    - `7` tensor descriptor matmul / reshape exacts;
    - `128` warp-specialization attention exacts;
  - `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`:
    branch-new visible failure after removing the temporary xfail.
- Detailed classification is in `gb200_failure_classification_20260412.md`:
  - actual bugs: the `162` `python/test/unit` wrong-output nodeids and the
    legacy M64 Gluon wrong-output nodeid;
  - stale tests: `TritonNvidiaGPU/invalid.mlir`,
    `TritonGPU/pipeline-loop-nest.mlir`,
    `TritonGPU/pipeline-lower-loop.mlir`, and branch-local
    `TritonNvidiaGPU/ops.mlir` `memdesc_subslice` result-type text;
  - API/contract updates verified main-invalid through merge-base backend
    lowering: `Analysis/test-membar-ttng.mlir`,
    `TritonGPU/proxy_fence_insertion.mlir`,
    `TritonNvidiaGPU/mma_lowering.mlir`, `NVWS/assign_stage_phase.mlir`,
    and `NVWS/aref-tmem-insertion.mlir`.
- Not recovery blockers:
  - Proton main `11` cudagraph / periodic flushing failures reproduce on
    merge-base and remain `PREEXISTING_ON_MERGE_BASE`;
  - `test_debug.py`, instrumentation/plugin tails, and Proton split-only extra
    failures were harness artifacts and cleared in exact CI-form reruns.
- Recovery order:
  1. Refresh the four stale lit tests first; this removes FileCheck /
     expected-diagnostic / branch-local view text drift from the red list
     without touching lowering.
  2. Rewrite the five API/contract-update lit files to supported scales-copy
     or MMAv5 spellings, or convert each to a deliberate negative when the
     unsupported case is the point.
  3. Root-cause the `162` branch-new `python/test/unit` correctness failures
     as shared compiler/runtime regressions before chasing individual nodeids.
     They all pass on merge-base and cluster in matmul, tensor descriptor, and
     warp-specialization attention surfaces.
  4. Fix the exposed legacy M64 MMA producer-family issue without re-adding
     the xfail.
  5. Leave Proton out of the branch recovery queue unless a future rerun shows
     a nodeid that is not also failing on merge-base.

### Latest Whole-`python/test/gluon` Refresh (2026-04-11 13:05 UTC)

- Validated checkpoint:
  - `be3cba0cd`
- Result:
  - group 1:
    - `5448 passed, 1002 skipped, 19348 deselected`
  - group 2:
    - `2666 passed, 3784 skipped, 19348 deselected`
  - group 3:
    - `4408 passed, 2041 skipped, 19348 deselected, 1 xfailed`
  - group 4:
    - `5582 passed, 866 skipped, 19350 deselected`
- Recovery-plan consequence:
  - the full current-branch Gluon group manifests are refreshed to
    `0` nodeids;
  - the stale M64 group-3 failures are closed at shard scope;
  - the remaining local recovery queue should no longer prioritize old full
    Gluon shard failure lists.
- Notes:
  - groups 3 and 4 printed `.kind::i8` PTX assembler diagnostics in the
    pre-`57a06c29b` sweep but finished with green pytest summaries;
  - `57a06c29b` now rejects the direct Gluon `tcgen05_mma` i8 path before IR
    lowering on `sm_103a+`, while preserving `compute-capability=100`
    conversion-lit coverage;
  - keep any future `.kind::i8` diagnostics in the noise/watch bucket unless a
    fresh current-head focused exact turns one into a failed nodeid.

### Latest M64 Physical-Bitcast Refresh (2026-04-11 12:10 UTC)

- Source checkpoint:
  - `85d8dbbf4`
- Closed at focused current-head scope:
  - `test_tmem_subslice_block_m_64_parent_layout[linear]`
  - `test_block_m_64_mma[linear]`
- How:
  - physical bitcast query inference now handles normalized M64 subview layouts
    and out-of-range physical origins;
  - M64 tests use supported `slice/subview -> bitcast` where the physical
    mapping is equivalent;
  - same-shape f32 2-column writes use direct supported subviews;
  - Gluon kernels pass layout objects as explicit `ttgl.constexpr` args so
    legacy/linear parameterization does not reuse the wrong cached kernel.
- Current legacy status:
  - `test_block_m_64_mma[legacy]` is xfailed, not fixed;
  - reason:
    - legacy M64 `64x64` layout sugar still needs producer-visible
      physical-family semantics for MMAv5 consumers.
- Recovery-plan consequence:
  - do not keep prioritizing stale group-3 M64 failures as bitcast bugs;
  - keep the legacy MMAv5 producer-family problem in the longer design queue;
  - rerun wider grouped `python/test/gluon` from `85d8dbbf4` before the next
    GB200 prioritization call.

### Latest Attention Revert And Two-CTA Assertion Refresh (2026-04-11 10:33 UTC)

- Current pushed `HEAD`:
  - `13930b1ff`
- The attention example migration to the supported bitcast API was intentionally
  backed out of `python/examples/gluon/01-attention-forward.py` at the user's
  request while keeping the API/lowering support on the branch.
- Current exact status:
  - `python/examples/gluon/01-attention-forward.py::test_op[False-dtype0-True-128-1024-48-4]`
  - `FAILED` in `gluon_to_ttgir`
  - symptom:
    - `LLVM ERROR: Invalid basis 32 for in-dim 'col' and out-dim 'dim1'. Basis must be less than the out-dim size.`
- Classification:
  - current branch-changed examples red;
  - deliberately deferred supported-view rewrite, not a request to preserve
    private `_reinterpret` behavior in lowering.
- Recovery consequence:
  - the examples lane is no longer green at current `HEAD`;
  - keep the attention exact separate from the supported bitcast API tests;
  - next attention work should express the reuse as
    `offset/slice/subview -> bitcast` while preserving the kernel's
    synchronization intent.
- The stale two-CTA spelling assertion in `test_mma_shared_inputs` is closed:
  - group 1: `3830 passed, 490 skipped`
  - group 2: `2954 passed, 1366 skipped`
  - group 3: `1206 passed, 3114 skipped`
  - group 4: `2584 passed, 1736 skipped`
- Recovery consequence:
  - the MMAv5 shared-input function surface remains refreshed to `0` exact
    failures;
  - rerun wider grouped `python/test/gluon` from current `HEAD` before using
    any old full-shard counts for prioritization.

### Latest `triton_kernels` Persistent Matmul OOR Refresh (2026-04-11 09:44 UTC)

- The persistent `python/triton_kernels/tests/test_matmul.py` shared-memory OOR
  bucket is fixed at the full-directory level.
- Before the fix:
  - full `python/triton_kernels/tests` current-head rerun:
    - `8 failed, 1270 passed, 1840 skipped`
  - representative failure:
    - `OutOfResources` with shared metadata such as `278632` bytes against the
      `232448` byte Blackwell limit.
- Classification:
  - real branch regression in layout-selection policy for replayed TMEM
    subslice loads/stores;
  - not an ISA boundary and not a case for preserving private `_reinterpret`
    behavior.
- Fix:
  - ordinary replayed `ttng.tmem_subslice` descriptors use the shape-local
    subview type for register-layout selection;
  - explicit `tmem_physical_bitcast` descriptor chains keep the exact
    preserved physical query.
- Validation:
  - code/test checkpoint:
    - `ab8ff6e6444b1cf8521a06941d6cf1b8a25e3217`
  - representative exact:
    - `1 passed in 5.38s`
    - metadata returned to `shared=214120`
  - focused persistent fp8/mxfp4 matmul slice:
    - `16 passed, 6 skipped in 31.85s`
  - full 4-way `python/triton_kernels/tests` refresh:
    - `2377 passed, 3444 skipped`
    - `0 failed`
- Recovery consequence:
  - `gb200_current_branch_triton_kernels_matmul_oor_refresh_failures.txt`
    is `0` nodeids;
  - `python/triton_kernels/tests` is no longer a current blocker.

### Latest Focused `test_core.py` Manifest Refresh (2026-04-11 07:25 UTC)

- The old focused `test_core.py` TMEM/MMA current-branch manifest is now
  locally reduced to zero live exact failures at the current checkpoint.
- Old manifest rerun:
  - `332` nodeids
  - `330 passed, 2 failed in 132.02s`
- The two failures were stale expectations, not recovery blockers:
  - legacy `64x128` split-N roundtrip expected packet offset `64`; current
    correct expectation is the row-encoded `1048576`
  - block-basis `ld.red` negative expected the older CTA-per-CGA diagnostic;
    current correct expectation is
    `TMEM layout '32x32b' unsupported for descriptor view`
- Exact post-refresh rerun:
  - `2 passed in 3.46s`
- Full focused manifest rerun with the refreshed diagnostic nodeid:
  - `332 passed in 125.78s`
- Recovery consequence:
  - `gb200_current_branch_test_core_group3_focus_e70a3aa09_failures.txt`
    is refreshed from the stale `332` nodeids to `0` nodeids;
  - this focused branch-added/changed `test_core.py` bucket is no longer the
    next recovery target;
  - continue staged validation with `triton_kernels` and wider grouped Gluon
    sweeps from current head before moving to the long-term saturation phases.

### Latest Runtime-Matrix File Refresh (2026-04-11 07:15 UTC)

- The branch-added `python/test/gluon/test_tmem_runtime_matrix.py` file is now
  locally reduced to zero live exact failures at the current checkpoint.
- Full-file run from the M64 split-N checkpoint:
  - `1437 passed, 354 skipped, 1 failed in 1226.69s`
- The single failure was stale expectation coverage, not a compiler/runtime
  correctness bug:
  - `test_tmem_runtime_matrix_cp_no_scales_indexed_view_canonicalized[128]`
  - runtime output and exact `tcgen05.cp.cta_group::1.128x128b` opcode were
    correct
  - TTGIR now prints the canonical physical mapping as
    `ttng.tmem_physical_layout` on the legacy `tensor_memory_encoding` root
    for this indexed-view path
- Exact post-fix rerun:
  - `test_tmem_runtime_matrix_cp_no_scales_indexed_view_canonicalized`
  - `1 passed in 3.31s`
- Recovery consequence:
  - `gb200_current_branch_test_tmem_runtime_matrix_focus_e70a3aa09_failures.txt`
    is refreshed from the stale `413` nodeids to `0` nodeids;
  - the branch-added runtime-matrix focus slice is no longer the next recovery
    target;
  - continue staged validation with MMA/matmul and `triton_kernels`, while the
    broader whole-`test-gluon` shard counts remain stale until rerun from
    current head.

### Latest M64 Row/Col Split-N And Half-Row Refresh (2026-04-11 06:51 UTC)

- The branch-added M64 row/col-permuted split-N runtime-matrix bucket is now
  locally green:
  - full explicit sweep plus auto guards:
    - `226 passed in 35.56s`
- The branch-added half-row stale-negative exacts that remained in the old
  manifest are now green under the new clean unsupported-descriptor contract:
  - split-N controls plus half-row and exotic clean-negative guards:
    - `45 passed in 5.32s`
- Recovery consequence:
  - `gb200_current_branch_test_tmem_runtime_matrix_splitn_rowcol_refresh_failures.txt`
    stays at `0` nodeids;
  - `gb200_current_branch_test_core_branch_added_splitn_expectation_refresh_failures.txt`
    stays at `0` nodeids;
  - `gb200_current_branch_test_gluon_halfrow_stale_negative_failures.txt`
    is refreshed to `0` nodeids;
  - older full Gluon shard counts remain stale until rerun from this new head.

### Historical Attention Bitcast Migration Checkpoint (2026-04-11 06:07 UTC, Superseded)

- The examples/Gluon refresh had exposed a current-branch regression in:
  - `python/examples/gluon/01-attention-forward.py::test_op[False-dtype0-True-128-1024-48-4]`
- Classification:
  - branch regression caused by an unsupported `_reinterpret` contract in the
    attention example;
  - fixed by migrating to a supported `slice/subview -> bitcast` TMEM
    descriptor API, not by preserving the old private reinterpret fallback.
- Historical exact status:
  - the attention exact is green on committed/pushed checkpoint
    `4263ae61b80e4f20e5c372a3c2cf5a7538d67620`.
- Broader examples status:
  - `python/examples/gluon/` is green:
    - `821 passed, 74 skipped in 134.89s`
- Recovery consequence:
  - historical only: the attention source was reverted at `13930b1ff`, so the
    examples lane has the attention exact red again at current `HEAD`;
  - do not treat this superseded green aggregate as the live current-head
    examples status.

### Latest MMAv5 Direct-Load Family Fix Refresh (2026-04-10 19:30 UTC)

- The merge-base-present `python/test/gluon/test_core.py::test_mma_shared_inputs[...]`
  bucket is now closed at the exact-function level on the current branch.
- Root cause:
  - the current branch had widened the 16-bit unpacked recursion in
    `getDistributedLayoutForTmemLdSt(...)` so far that ordinary direct MMAv5
    accumulator loads were falling into the generic `I32x32b` builder.
  - that changed their TTGIR physical layout and PTX family away from the
    merge-base direct-load path.
- Fix:
  - keep that recursion only on the actual reinterpret/query-style
    `I32x32b` path and leave direct loads on the standard bitwidth-packing
    path.
- Validation:
  - `make -j8`
  - representative current-fail / merge-base-pass exacts now pass
  - fresh four-way current-head function rerun:
    - group `1`
      - `3830 passed, 490 skipped`
    - group `2`
      - `2954 passed, 1366 skipped`
    - group `3`
      - `1206 passed, 3114 skipped`
    - group `4`
      - `2584 passed, 1736 skipped`
- Recovery consequence:
  - `gb200_current_branch_test_gluon_mma_shared_inputs_failures.txt` is now
    refreshed to `0` nodeids.
  - the older `4685`-nodeid MMAv5 manifest and the pre-fix full-group counts
    are now stale historical artifacts for that function.
  - the old-mainline queue moved to the examples lane after this checkpoint
  - keep the branch-added PTX-expectation and stale-negative TMEM tails
    outside the core recovery queue until the example regressions are reduced.

### Latest `02-convolution.py` Recovery Checkpoint (2026-04-10 19:46 UTC)

- The merge-base-present convolution example regression is now closed on the
  current branch.
- Root cause:
  - the final failure was not another TMEM row-plan or MMAv5 family bug
  - it was a branch-added full-tile shared-memory scratch fallback in
    `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp`
    for `ttg.convert_layout` when either side used `LinearEncodingAttr`
  - for the convolution epilogue
    `tensor<256x256xf16, #linear> -> tensor<256x256xf16, #blocked>`, that
    fallback over-allocated the scratch buffer to the full `128 KB` tile and
    pushed the kernel from merge-base `147516` shared bytes to current-head
    `262208`
- Supporting cleanup found during the same reduction:
  - preserve exact TMEM encoding across outer-dimension `memdesc_index` views
    when the leaf descriptor is still valid
  - restore barrier-driven `tcgen05.mma` / `tcgen05.mma_scaled` async
    selection in the Gluon builder instead of forcing `is_async=true`
- Validation:
  - `make -j8`
  - representative exact:
    - `python/examples/gluon/02-convolution.py::test_op[0-1-3-3-384-384-64-64-1]`
    - `PASSED`
  - full file:
    - `python/examples/gluon/02-convolution.py`
    - `48 passed in 7.96s`
  - nearby controls:
    - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-0-warps0-16-False-True-acc_dtype3]`
    - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_4w-layout0-64-128-32x32b-4-16x32bx2]`
    - both `PASSED`
- Recovery consequence:
  - the live old-mainline branch-recovery queue is now:
    1. `REAL_NEW_ON_BRANCH_REGRESSION`
       - `python/examples/gluon/03-matmul-multicta.py`
  - `gb200_current_branch_examples_convolution_failures.txt` should now be
    treated as refreshed empty
  - `gb200_current_branch_examples_gluon_failures.txt` is now stale until the
    full examples directory is rerun on top of this fix

### Latest Full GB200 Census Refresh

- The whole GB200 census has now been rerun on the current branch, so the
  recovery queue no longer stops at the earlier descriptor-chain checkpoint.
- Current lane state:
  - green:
    - `make test-lit`
    - `make test-cpp`
    - `make NUM_PROCS=24 test-unit`
    - `make test-regression`
    - `make NUM_PROCS=24 test-gsan`
    - `make test-microbenchmark`
  - red:
    - `make test-proton`
      - same `11` exact failures on merge-base
    - current-head `python/test/gluon/ + python/tutorials/gluon/` four-way
      sweep
    - `python/examples/gluon/`
- Current branch-recovery backlog after merge-base classification:
  1. `REAL_NEW_ON_BRANCH_REGRESSION`
     - `python/test/gluon/test_core.py::test_mma_shared_inputs[...]`
     - semantic manifest:
       - `gb200_current_branch_test_gluon_mma_shared_inputs_failures.txt`
       - `4685` latest exact nodeids
     - merge-base-present and independently proven branch-caused with exact
       reruns:
       - `False-ctas_per_cga0-1-1-1-64-0-0-warps0-16-False-True-acc_dtype3`
       - `False-ctas_per_cga2-1-1-1-64-64-128-warps0-16-False-True-acc_dtype3`
       - `False-ctas_per_cga2-2-4-1-64-32-32-warps2-16-True-True-acc_dtype7`
       - `True-ctas_per_cga1-2-4-1-64-128-128-warps0-16-False-True-acc_dtype3`
     - representative current-head symptom:
       - wrong-code / `torch.testing.assert_close(...)` mismatches
  2. `REAL_NEW_ON_BRANCH_REGRESSION`
     - `python/examples/gluon/02-convolution.py`
       - `48` exacts
       - current symptom:
         - `OutOfResources: shared memory, Required: 262208, Hardware limit: 232448`
       - merge-base rerun:
         - `48 passed in 7.30s`
  3. `REAL_NEW_ON_BRANCH_REGRESSION`
     - `python/examples/gluon/03-matmul-multicta.py`
       - `14` exacts
       - current symptom:
         - wrong-code / `AssertionError: Tensor-likes are not close!`
       - merge-base rerun:
         - `82 passed, 14 skipped in 37.96s`
- Current branch-added tails that stay visible but outside the core recovery
  queue until the old-mainline regressions are under control:
  - `BRANCH_CHANGED_COVERAGE_RED`
    - `gb200_current_branch_test_gluon_splitn_expectation_tail_failures.txt`
    - `1` exact
    - absent on merge-base
    - currently only an opinionated PTX-offset expectation mismatch
  - `STALE_NEGATIVE_OR_SUPPORT_BROADENED`
    - `gb200_current_branch_test_gluon_halfrow_stale_negative_failures.txt`
    - `3` exacts
    - absent on merge-base
    - current failure is `Failed: DID NOT RAISE CompilationError`
- Contamination / cache-sensitive interpretation:
  - the large MMAv5 shard bucket is not explained by a trivial
    `TRITON_CACHE_DIR` collision, because the refreshed four-way sweep used
    isolated per-GPU caches and still reproduced it
  - but raw shard counts overstate independent root causes:
    - at least one shard-failing exact
      `False-ctas_per_cga2-1-1-1-64-0-32-warps2-16-False-True-acc_dtype3`
      passes in a fresh isolated current-head process
  - current working hypothesis:
    - a real MMAv5 correctness bug exists on the current branch, and later
      shard failures are partly inflated by process/device contamination after
      earlier bad kernels or by missing cache-key / global-state inputs
- Immediate execution order from this refreshed state:
  1. isolate one representative current-fail / merge-base-pass
     `test_mma_shared_inputs[...]` exact from each visible subfamily
  2. compare TTGIR / PTX / planner-side query traces on current branch vs
     merge-base for those representatives
  3. reduce the broad MMAv5 bucket to the smallest structural root causes
     before touching the examples lane
  4. fix the examples only after the MMAv5/compiler bucket is under control
  5. revisit the branch-added PTX-expectation and stale-negative TMEM tails
     afterward
- Dedicated contamination-root-cause plan to keep separate from the compiler
  fix loop:
  1. build a two-nodeid same-process reproducer where the first exact poisons a
     later one that otherwise passes in isolation
  2. check whether the failure survives:
     - a fresh Python process
     - a fresh `TRITON_CACHE_DIR`
     - a different GPU
     - `CUDA_LAUNCH_BLOCKING=1`
  3. if the failure only appears after a prior bad kernel, trace device/runtime
     state reuse before blaming cache files
  4. if the failure survives a fresh process with the same compiled artifact,
     inspect compiler cache-key inputs and runtime-global state next
  5. only add an environmental workaround if the actual missing invalidation or
     state-reuse bug is understood and recorded

### Latest Validated State After Descriptor-Chain Recovery

- The branch-only TMEM/compiler bucket is now clear:
  - `gb200_current_branch_test_core_branch_added_descriptor_chain_refresh_failures.txt`
    - refreshed to `0` nodeids
  - nearby validation:
    - full `test_tmem_descriptor_chain_matrix`
      - `26 passed in 15.94s`
- Important reduction learning:
  - the old `12`-nodeid descriptor-chain manifest was not `12` independent
    bugs
  - a fresh-process isolation matrix showed only:
    - `linear_m64_32x32b_4w`
    - `linear_m64_32x32b_8w`
    were independently red
  - the other `10` nodeids were fallout after those first CUDA launch faults
- The next recovery queue is now just the old-mainline example regressions:
  1. `python/examples/gluon/02-convolution.py`
  2. `python/examples/gluon/03-matmul-multicta.py`

### Historical Verified Recovery Buckets After `c6990c1e9`

- Fresh isolated current-head reruns now reduce the actionable GB200 backlog
  to three concrete surfaces:
  1. branch-added TMEM/compiler exacts:
     - `gb200_current_branch_test_core_branch_added_descriptor_chain_refresh_failures.txt`
     - current rerun:
       - `12 failed in 3.73s`
     - merge-base status:
       - absent from merge-base coverage
  2. old-mainline convolution example regression:
     - `python/examples/gluon/02-convolution.py`
     - current rerun:
       - `48 failed in 8.38s`
     - merge-base rerun:
       - `48 passed in 7.30s`
  3. old-mainline multicta example regression:
     - `python/examples/gluon/03-matmul-multicta.py`
     - current rerun:
       - `14 failed, 68 passed, 14 skipped in 65.24s`
     - merge-base rerun:
       - `82 passed, 14 skipped in 37.96s`
- Updated practical consequence:
  - the next compiler/debugging slice should stay on the `12` descriptor-chain
    exacts
  - once that branch-only TMEM bucket is reduced or explained, move to the two
    old-mainline example regressions
  - the broader shard/group manifests are now historical context, not the live
    recovery queue

### Current Clean-HEAD Checkpoint

- HEAD:
  - `181e369bd`
- Worktree:
  - clean
- Current branch-topline state:
  - `make test-lit`
    - green
    - `248 passed, 2 unsupported`
  - `make NUM_PROCS=24 test-unit`
    - green
  - the next active Gluon recovery slice is no longer the old
    merge-base-present focused `test_core.py` bucket:
    - `146 / 146` merge-base-present exacts now pass on the current branch
- The remaining focused-core branch-added / branch-changed tail now reduces to:
  - `12` exact descriptor-chain runtime failures:
    - use
      `gb200_current_branch_test_core_branch_added_descriptor_chain_refresh_failures.txt`
    - shared symptom:
      - `RuntimeError: CUDA error: misaligned address`
  - `1` split-N PTX-expectation exact:
    - use
      `gb200_current_branch_test_core_branch_added_splitn_expectation_refresh_failures.txt`
    - current classification:
      - runtime-correct but stale/opinionated codegen expectation
  - `21` `test_tmem_linear_roundtrip_variant_sweep[...]` exacts from the same
    earlier shard now all pass in clean isolation
- Practical consequence:
  - the next compiler fix should be driven by the descriptor-chain manifest,
    not by the older contaminated shard counts.
- Additional isolated-reduction updates after that checkpoint:
  - the branch-added / branch-changed `test_fpsan.py` manifest now reruns
    clean (`11 passed, 1 skipped`) and drops out of the live red list
  - the focused runtime-matrix bucket is now reduced to:
    - `208`
      `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[...]`
      exacts, all failing with `CUDA error: misaligned address`
    - `2` warpx2 candidate-positive exacts failing at `tmem.get_reg_layout()`
      on the same `tensor_memory_descriptor<fp32, ['128', '4'], ...>` shape
      that still fails in the isolated frontend exact
- Updated practical consequence:
  - the active branch-added recovery work is now split between:
    - the `12`-nodeid descriptor-chain focused-core runtime bucket
    - the `208`-nodeid runtime-matrix split-N runtime bucket
    - the smaller `auto`-reg-layout descriptor-view frontend/warpx2 bucket
  - the examples lane remains separate:
    - it is still red on current head (`48` convolution OOR exacts and `14`
      multicta wrong-code exacts)
    - but unlike the branch-added TMEM tests above, those failing example
      functions already exist on merge-base, so they should be treated as
      branch-caused regressions on old upstream coverage
- Latest trace-backed refinement:
  - the `208` split-N runtime-matrix exacts now have the strongest failing vs
    passing contrast and should move to the front of the queue:
    - passing control keeps a packed `16x32bx2.x2` direct path
    - failing permuted cases log `packed16 support precondition fail` /
      `no packed mem layout` and degrade to scalar `32x32b.x1`
  - the `2` warpx2 positives plus the isolated frontend exact are one shared
    representability gap on non-surjective `[128, 4]` descriptor views, not
    independent Python/frontend issues
  - the `12` descriptor-chain exacts are likely at least two sub-buckets:
    - packed `linear_m64_*` launch faults
    - mixed-layout scalar-family accesses that currently look identical to a
      passing identity control at the PTX level
- Updated execution order:
  1. split-N row/col-permuted packed-family discovery
  2. shared non-surjective `[128, 4]` direct-view representability
  3. descriptor-chain cleanup, split by `linear_m64_*` vs `linear_mixed_*` if
     needed
  4. examples lane after the TMEM/compiler buckets are under control
- Post-fix update:
  - step `1` is now complete
  - the `208`-nodeid split-N runtime-matrix manifest reruns fully green
  - the stale split-N PTX-expectation exact is also green now
  - the live TMEM/compiler backlog is therefore reduced to:
    - the shared `[128, 4]` representability gap (`2 + 1` exacts)
    - the `12` descriptor-chain exacts
    - then the examples lane
- Immediate next execution order:
  1. shared non-surjective `[128, 4]` direct-view representability
  2. descriptor-chain cleanup, still split by `linear_m64_*` vs
     `linear_mixed_*`
  3. examples lane after the TMEM/compiler buckets are under control
- Post-fix update:
  - step `1` is now complete too
  - the two warpx2 candidate-positive runtime-matrix exacts are green
  - the isolated frontend `get_reg_layout()` exact is green
  - the live TMEM/compiler backlog is therefore reduced to:
    - the `12` descriptor-chain exacts
    - then the separate examples lane
- New immediate execution order:
  1. descriptor-chain cleanup, still split by `linear_m64_*` vs
     `linear_mixed_*`
  2. examples lane after the TMEM/compiler bucket is under control

### Latest Dirty MMAv5 Row-Plan Propagation Checkpoint

- The current dirty follow-up propagates explicit MMAv5 root row plans through
  the remaining alloc-cloning / higher-rank producer paths instead of only the
  original root producers:
  - new helper:
    - `setExplicitMMAv5RootRowPlanIfNeeded(TMEMAllocOp)`
  - cloned or recreated allocs now preserve or reinitialize the contract in:
    - `MMAv5PipelineUtility.cpp`
    - `WSCodePartition.cpp`
    - `WSDataPartition.cpp`
    - `HoistTmemStore.cpp`
    - `InsertTmemAref.cpp`
- Broad current-head impact after `make -j8`:
  - `make NUM_PROCS=24 test-unit`
    - `134 failed, 15019 passed, 5492 skipped`
- Refreshed current-head exact manifests now reduce the live unit surface to:
  - `3` `test_matmul.py` nodeids
  - `3` `test_tensor_descriptor.py` nodeids
  - `128` `test_warp_specialization.py` nodeids
  - combined:
    - `gb200_current_branch_test_unit_rowanchor_refresh_failures.txt`
      (`134` exact nodeids)
- Fresh merge-base confirmation completed so far:
  - `python/test/unit/language/test_tensor_descriptor.py`
    - `2604 passed, 110 skipped`
  - the `3` refreshed tensor-descriptor exact failures are therefore
    definitely branch-local
- Current root-cause hypothesis:
  - the remaining red surface is no longer the old broad MMAv5 producer bug;
  - the planner is likely collapsing a root-backed `128`-row MMAv5 contract to
    the `64`-row active descriptor view too early for root-preserving
    descriptor views;
  - that breaks direct row-anchor representability for `memdesc_index` /
    higher-rank views that still semantically depend on the backing root row
    contract.
- Practical consequence:
  - the next structural fix should be in the planner / direct-view
    representability logic, not another producer annotation sweep.
- Still pending in the census:
  - refreshed merge-base full-file reruns of `test_matmul.py` and
    `test_warp_specialization.py`
  - refreshed current-head `python/triton_kernels/tests`
  - until those complete, the older merge-base-green file reruns remain the
    provisional branch-vs-main evidence for those files

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

### Latest Dirty Worktree Checkpoint

- The current dirty follow-up has moved from the older LLVM raw-first
  experiment to a producer-side MMAv5 root-row-plan fix:
  - `getMMAv5RootRowPlan(...)` is now the shared helper for all `M=64` MMAv5
    root TMEM families;
  - `PromoteLHSToTMem` now annotates the source-initialized TMEM roots it
    creates with the same explicit row-plan contract; and
  - the helper now accepts higher-rank roots whose trailing TMEM row dimension
    is `64`.
- Exact impact on the current worktree:
  - repaired:
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
    - `python/test/unit/language/test_matmul.py::test_lhs_in_tmem[float32-False-64-128-32]`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-1-64-512-32-2-float32-tensorfloat32]`
    - `python/test/regression/test_cast_matmul.py::test_cast_matmul[768-768-1024-16-64-16-bfloat16-float16-float16]`
    - `python/test/regression/test_cast_matmul.py`
      - `1080 passed, 216 skipped`
  - still green:
    - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-layout9-64-128-32x32b_splitn-8-16x32bx2]`
    - `python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]`
    - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[identity-identity-2-32x32b_splitn]`
    - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-32-warps2-8-False-True-acc_dtype0]`
    - `python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
  - still failing fresh:
    - warp-specialization higher-rank roots:
      - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
        - `CUDA error: misaligned address`
    - `50.0%` wrong-code:
      - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_persistent_forward[False-8-True-2-128-64-1024-1024]`
    - compile-time row-anchor gap:
      - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-128-32]`
      - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-16-16]`
      - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-8-64-128-32]`
      - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float32]`
- The broad unit xdist XML surface is now fully classified against merge-base:
  - `2098` current-branch exact nodeids exist on merge-base; and
  - the exact merge-base rerun is fully green/skip:
    - `2083 passed, 15 skipped`
- Fresh isolated current-branch reruns from the XML-only tail files all pass:
  - `test_standard.py`, `test_random.py`, `test_cache.py`, `test_blaslt.py`,
    `test_autotuner.py`, `test_launch.py`, `test_bindings.py`, and
    `test_triton_to_gluon.py`
- Current interpretation:
  - the broad unit XML surface is branch-local, but it is now even more
    clearly stale as a current-head count:
    - the source-root MMAv5 slice is repaired;
    - the remaining primary fixes are the higher-rank warp-specialization
      producer path and the descriptor-view row-anchor materialization gap; and
    - the rest of the unit XML tail is still most likely fallout after bad
      kernels until a post-fix rerun refreshes the counts.

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

1. Remaining higher-rank MMAv5 root producer bucket
- Why it comes first:
  - it is the largest remaining old-mainline correctness bucket after the
    source-root fix;
  - it still hits real standalone runtime failures on the current tree; and
  - the current PTX / TTGIR evidence says higher-rank warp-specialization root
    allocs are still not carrying the explicit MMAv5 row-plan contract.
- Representative exact nodeids:
  - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
  - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_persistent_forward[False-8-True-2-128-64-1024-1024]`
- Broader affected manifests / files:
  - `python/test/unit/language/test_warp_specialization.py`
  - likely NVWS TMEM-root producer helpers:
    - `third_party/nvidia/lib/Dialect/NVWS/Transforms/Utilities.cpp`
    - `third_party/nvidia/lib/Dialect/NVWS/Transforms/HoistTmemStore.cpp`
- Current root-cause hypothesis:
  - higher-rank TMEM root allocs created or rewritten inside warp
    specialization still bypass the explicit MMAv5 root row-plan annotation,
    so they lower with scalar `32x32b.x1`-style accesses instead of the
    packed `16x32bx2` family used by the repaired source-root path

2. Unsupported-row-anchor compile bucket on the new planner
- Why it is probably adjacent:
  - these failures still hit `ttng.tmem_store` row anchors `32,64`, but they
    now look like a separate direct-ld/st descriptor-view representability gap
    after the source-root MMAv5 fix
- Representative exact nodeids:
  - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-128-32]`
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
   MMAv5 producer-side higher-rank root bucket.
- Validation ladder:
  - keep the repaired direct split-N controls green:
    - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-layout9-64-128-32x32b_splitn-8-16x32bx2]`
    - `python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]`
    - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[identity-identity-2-32x32b_splitn]`
    - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-32-warps2-8-False-True-acc_dtype0]`
    - `python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
  - then debug the remaining producer-side exacts:
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_persistent_forward[False-8-True-2-128-64-1024-1024]`
  - then rerun `python/test/unit/language/test_warp_specialization.py`
  - only after that bucket is reduced, rerun the full `2098`-nodeid unit XML
    manifest on the current branch to see how much of the xdist fallout clears
  - then rerun `python/test/unit/language/test_matmul.py` and
    `python/test/unit/language/test_tensor_descriptor.py` to measure what is
    left of the compile-time row-anchor bucket
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
- Additional concrete harness discrepancy to investigate:
  - a simple standalone process obeys `CUDA_VISIBLE_DEVICES` correctly on this
    machine;
  - but some pytest/xdist workers in the broad sweeps either lost
    `CUDA_VISIBLE_DEVICES` or appeared on physical GPU `0` even when their
    environment reported `CUDA_VISIBLE_DEVICES=3`.
- Root-cause plan for future work:
  1. preserve the exact failing order when a shard-only anomaly appears;
  2. replay it in one process, then across fresh processes with the same cache,
     then across fresh caches;
  3. instrument worker env + physical GPU selection before changing cache
     logic;
  4. only blame cache keys if the symptom survives the process boundary and
     remains tied to cache reuse;
  5. otherwise investigate bad-kernel aftermath, stream/device state leakage,
     xdist worker isolation, or compiler global-state reuse.

## Performance Notes To Revisit Later

- `make test-microbenchmark` is green, so there is no immediate microbenchmark
  gate failure.
- The current branch recovery phase is functionality/correctness-first.
- Keep an eye on codegen-family selection changes while fixing TMEM, because
  they may later explain performance deltas even when correctness is restored.

## 2026-04-10 Overnight Classification Checkpoint

- Refreshed current-head shard manifests now make the branch-vs-main split
  explicit without claiming the raw shard totals are independent bugs:
  - shard `3 / 4`:
    - `1252` exact nodeids total
    - `967` exact old-mainline nodeids
    - `285` exact branch-added / branch-changed nodeids
  - shard `4 / 4`:
    - `686` exact nodeids
    - entirely branch-added runtime-matrix coverage
    - unchanged from the older group-4 manifest
- Practical consequence:
  - do not pick fixes from the raw shard-3 total yet
  - first reduce shard `3 / 4` through isolated current-head reruns of:
    1. `python/test/gluon/test_lowerings.py`
    2. `python/test/gluon/test_fpsan.py`
    3. focused `python/test/gluon/test_core.py` TMEM/MMA slice
- Fresh status still pending while this checkpoint was written:
  - current-head shard `1 / 4`
  - current-head `python/triton_kernels/tests`
  - fresh merge-base full-file reruns of `test_matmul.py` and
    `test_warp_specialization.py`
- Until those finish, the live recovery ordering remains:
  1. old-mainline unit/lit descriptor-view row-anchor bucket
  2. old-mainline warp-specialization higher-rank MMAv5-root bucket
  3. branch-added TMEM runtime-matrix bucket
  4. remaining isolated shard-3 exacts after the queued reductions land

## 2026-04-10 Additional Isolated Reductions

- The focused shard-3 reductions materially changed the backlog shape:
  - isolated `test_core.py` TMEM/MMA slice:
    - `332` exact independent failures
    - `146` exact old-mainline nodeids
    - `186` branch-added / branch-changed nodeids
  - isolated `test_fpsan.py`:
    - `1` exact branch-changed failure
  - isolated `test_layout_format_view.py`:
    - green
  - isolated frontend tail:
    - `1` exact branch-added failure
- The focused branch-added runtime-matrix slice also fails independently:
  - `413` exact failures across the dominant split-N / `ld.red` /
    `ldst_scales` / `cp` families
- Practical change in recovery order:
  1. old-mainline unit/lit descriptor-view row-anchor bucket
  2. old-mainline warp-specialization higher-rank MMAv5-root bucket
  3. old-mainline shard-3 core slice:
     - start with `test_mma_scaled_tcgen05_copy[...]`,
       `test_tmem_reduction[...]`, `test_padded_shared_layout_subslice[...]`,
       and the merge-base-present subset of the focused core manifest
  4. branch-added runtime-matrix focus slice
  5. remaining branch-added / branch-changed shard-3 exacts
  6. only then return to the still-pending isolated `test_lowerings.py`
     result if it uncovers another independent bucket

## 2026-04-10 Inventory Closure And Updated Fix Order

- The last pending GB200 census jobs are now closed:
  - current-head `python/triton_kernels/tests`
    - `2377 passed, 3444 skipped`
  - isolated current-head `python/test/gluon/test_lowerings.py`
    - `4937 passed, 512 skipped`
  - consequence:
    - `triton_kernels` is not a current blocker
    - the raw shard-3 `test_lowerings.py` failures were fallout only
- The completed GB200 lane now splits cleanly into:
  - real current branch red buckets:
    - `test/TritonGPU/pipeline-lower-loop.mlir`
    - `134` exact current-head unit nodeids
    - focused `python/test/gluon/test_core.py`
      - `332` exact independent nodeids
      - `146` merge-base-present
      - `186` branch-added / branch-changed
    - focused `python/test/gluon/test_tmem_runtime_matrix.py`
      - `413` exact branch-added nodeids
    - isolated branch-changed tails:
      - `python/test/gluon/test_fpsan.py::test_tcgen05_mma_scaled[linear_identity-acc_layout1-e4m3-e2m1]`
      - `python/test/gluon/test_frontend.py::test_tensor_memory_linear_layout_non_surjective_reg_layout_parses`
    - branch-changed examples:
      - `48` exact `python/examples/gluon/02-convolution.py` nodeids
      - `14` exact `python/examples/gluon/03-matmul-multicta.py` nodeids
  - preexisting or excluded buckets:
    - Proton `11` exact nodeids
    - the two reinterpret-contract rewrite candidates
    - stale-negative / broadened-support expectations
- Updated recovery order:
  1. direct-view row-anchor representability
     - exact entry points:
       - `test/TritonGPU/pipeline-lower-loop.mlir`
       - `python/test/unit/language/test_matmul.py`
       - `python/test/unit/language/test_tensor_descriptor.py`
     - why first:
       - smallest old-mainline compile bucket
       - likely same root cause across lit + unit
       - most direct path to shrinking `make test-lit` and part of
         `make test-unit`
     - exact lit repro:
       - `cd build/cmake.linux-aarch64-cpython-3.12 && lit -v /root/code/triton-tmem-isolated/test/TritonGPU/pipeline-lower-loop.mlir`
  2. higher-rank MMAv5-root warp-specialization bucket
     - `python/test/unit/language/test_warp_specialization.py`
     - still the dominant remaining old-mainline runtime unit surface
  3. merge-base-present focused `python/test/gluon/test_core.py` bucket
     - start with `test_mma_scaled_tcgen05_copy[...]`,
       `test_tmem_reduction[...]`, `test_padded_shared_layout_subslice[...]`,
       and `test_gather/scatter_padded[...]`
  4. branch-added / branch-changed Gluon coverage
     - focused runtime-matrix bucket
     - focused core tail (`186`)
     - FPSAN exact
     - frontend exact
  5. examples lane
     - `02-convolution.py` shared-memory portability/resource bucket
     - `03-matmul-multicta.py` wrong-code bucket
  6. reinterpret-contract rewrites plus the missing explicit descriptor/view
     support needed to express their intent cleanly
- Cache / xdist investigation is still required, but it no longer blocks the
  fix order above:
  - current evidence is stronger for worker/process/device contamination after
    bad kernels than for a simple on-disk `TRITON_CACHE_DIR` collision
  - concrete harness plan:
    1. reproduce a known shard-only failure in fresh serial processes
    2. reproduce it again with xdist but distinct cache directories per worker
    3. log device identity, cache key inputs, and compile artifact paths
    4. only change cache-key semantics if the failure survives the fresh
       process boundary and still tracks cache reuse

## 2026-04-10 direct-view row-anchor checkpoint: step 1 is effectively landed, step 2 is now the live blocker

- Implemented recovery for the step-1 bucket:
  - `getTMemLdStRowPlanForQuery(...)` now uses the query row plan for pure
    outer `ttg.memdesc_index` TMEM views instead of blindly inheriting the
    larger backing row plan.
- Current validation:
  - `make -j8`
    - green
  - `make test-lit`
    - green
    - `248 passed, 2 unsupported`
  - focused exacts:
    - `test_simple_persistent_matmul[False-4-64-128-32]`
      - green
    - `test_tensor_descriptor_reshape_matmul[float32]`
      - green
    - direct `pipeline-lower-loop.mlir` repro
      - green
- Broad `test-unit` consequence:
  - the run no longer surfaced any independent `test_matmul.py` or
    `test_tensor_descriptor.py` failures before reaching
    `test_warp_specialization.py`;
  - this is strong evidence that the old direct-view lit + unit
    representability bucket is fixed for the right structural reason.
- Live remaining old-mainline unit blocker:
  - first real exact:
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-False-2-64-64-1024-1024]`
  - isolated current-head repro on GPU 0 with `CUDA_LAUNCH_BLOCKING=1`:
    - still fails at kernel launch with `Triton Error [CUDA]: misaligned address`
  - interrupted broad-unit log:
    - `10` true warp-specialization kernel failures
    - `109` later `torch.manual_seed` contamination failures on poisoned xdist
      workers
- Separate harness symptom worth remembering:
  - after the warp failures, the last remaining xdist worker can stall in
    `python/test/unit/runtime/test_cache.py::test_async_compile_mock`;
  - `py-spy` shows it waiting in `AsyncCompileMode.__exit__` on unfinished
    futures; and
  - this should be investigated as part of the cache/async-compile harness work,
    not conflated with the TMEM row-plan bug.
- Updated recovery order after this checkpoint:
  1. higher-rank MMAv5-root warp-specialization bucket
  2. merge-base-present focused `python/test/gluon/test_core.py` bucket
  3. branch-added / branch-changed Gluon coverage
  4. example lane
  5. reinterpret-contract rewrites and missing explicit view support

## 2026-04-10 warp-specialization capture-forwarding checkpoint: step 2 is likely fixed, but needs a fresh file-level/unit census

- Structural recovery landed for the step-2 bucket:
  - explicit `ttng.tmem_ldst_row_plan` contracts now survive
    `ttg.warp_specialize` partition captures because
    `getExplicitTMemLdStRowPlan(...)` follows block arguments back through
    `gpu::WarpSpecializePartitionsOp::getExplicitCaptures()`
- Why this is the right fix:
  - the bug was contract loss across partition captures, not a missing root
    annotation and not an MMAv5 family-selection heuristic problem
  - fixing the forwarding edge keeps the planner consistent instead of adding a
    new warp-specialization-specific rescue stack
- Current evidence:
  - compile-only TMEM traces now keep `ttg.memdesc_index<64x64xf32>` raw
    queries on `rawRowPlan=128` / `atom=4`
  - PTX for the first exact warp repro is back on packed
    `16x32bx2.x32` at source line `491`
  - five representative `test_warp_specialize_attention_forward[...]` exacts
    from the earlier interrupted `test-unit` log now pass in isolation
- Updated recovery order from here:
  1. rerun `python/test/unit/language/test_warp_specialization.py`
  2. rerun `make NUM_PROCS=24 test-unit`
  3. refresh the exact manifests and the GB200 CI inventory from that result
  4. if the old-mainline unit bucket is closed or materially smaller, continue
     with the merge-base-present focused `python/test/gluon/test_core.py`
     bucket
  5. keep the cache/xdist investigation as a separate harness track unless a
     fresh serial repro proves a deeper cache-key bug

## 2026-04-10 unit-lane closure checkpoint: recovery step 2 is closed

- Fresh validation closed the old-mainline unit bucket completely:
  - `python/test/unit/language/test_warp_specialization.py`
    - `1599 passed, 202 skipped`
    - `0 failed`
  - full `make NUM_PROCS=24 test-unit`
    - green end to end
    - included:
      - main `python/test/unit`
      - `test_debug.py`
      - `python/triton_kernels/tests`
      - `python/tutorials/06-fused-attention.py`
      - instrumentation test
      - plugin tests
- Consequence:
  - the previous unit manifests are now stale historical artifacts
  - the live GB200 branch-recovery backlog no longer includes a unit-lane
    bucket
- Updated recovery order:
  1. merge-base-present focused `python/test/gluon/test_core.py`
     - `146` exact nodeids
  2. branch-added / branch-changed Gluon coverage
     - focused core tail
     - focused runtime-matrix bucket
     - isolated FPSAN and frontend exacts
  3. examples lane
  4. reinterpret-contract rewrites plus any missing explicit descriptor/view
     support still needed to express those tests cleanly
- Harness note:
  - the earlier async-compile / `test_cache.py` hang did not reproduce in the
    fresh green unit rerun, so it stays on the side as a failure-induced
    contamination investigation rather than a primary blocker

## 2026-04-10 examples-lane refinement: `03-matmul-multicta.py` is a TMEM column-slice planner bug

- `02-convolution.py` is no longer the live examples blocker.
- The active old-mainline examples bucket is now:
  1. `python/examples/gluon/03-matmul-multicta.py`
- Updated interpretation of that bucket:
  - not a reinterpret-contract test
  - not generic MMA semantics
  - not yet a performance-only issue
  - it is a real TMEM `64x128 -> 64x32` column-slice direct-ld/st planner bug
- Immediate recovery order from here:
  1. fix the column-slice raw/support query construction so the direct planner
     sees the full parent row-zero structure instead of the collapsed partial
     `64x32` support image
  2. rerun the exact multicta repro until support/raw direct planning succeeds
     or TTGIR/PTX returns to merge-base-class `x16`
  3. rerun the full `python/examples/gluon/03-matmul-multicta.py` file
  4. refresh `gb200_current_branch_examples_multicta_failures.txt`
  5. rerun the full `python/examples/gluon/` directory so the aggregate
     examples manifest becomes current again

## 2026-04-10 22:41 UTC: the old-mainline multicta example regression is closed; the next live TMEM issue is outside the merge-base recovery queue

- `python/examples/gluon/03-matmul-multicta.py` is now green on the current
  branch:
  - representative exact:
    - `python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-2-CGA_LAYOUT0-8-0-64-128-64]`
    - `PASSED`
  - full file:
    - `82 passed, 14 skipped`
- Root cause / fix:
  - the branch regression was a real TMEM `64x128 -> 64x32` pure column-subview
    direct-ld/st bug;
  - the fix was to let the lowering reuse the source support/raw query plan
    for pure rank-2 TMEM column subviews instead of degrading immediately to
    the sliced `64x32` query type, plus the associated whole-tile warp-anchor
    guard and split-N frontend cleanup.
- Recovery consequence:
  - the old-mainline examples bucket is now closed at the exact-file level
  - `gb200_current_branch_examples_multicta_failures.txt` is refreshed empty
  - the aggregate `gb200_current_branch_examples_gluon_failures.txt` remains
    stale until the full examples directory is rerun on top of this fix
- The next live TMEM engineering problem is *not* part of the merge-base
  parity queue:
  - `python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
  - current reading:
    - branch-added / branch-changed TMEM coverage
    - still real wrong-code after relaxing the PTX-immediate assertion
    - points to a missing producer-owned physical TMEM family contract, because
      later TMEM slices / MMAv5 consumers still reason from the logical memdesc
      type instead of the family actually selected by the producer store
- Updated recovery order from here:
  1. rerun the full `python/examples/gluon/` directory so the stale aggregate
     examples manifest is replaced by a current post-multicta one
  2. keep the merge-base parity queue focused on any remaining old-mainline
     red surfaces from that rerun
  3. handle `test_block_m_64_mma[linear]` as the next core TMEM design fix,
     not as an examples-lane recovery item


## 2026-04-12 docket update: legacy M64 MMAv5 xfail must be fixed later

- Latest recorded unit-lane reading:
  - checkpoint `78196b4e4` reran the full `make NUM_PROCS=24 test-unit` wrapper
    green;
  - the current unit failure manifests are empty;
  - the large XML unit manifest remains historical only;
  - because this predates later focused compiler/test coverage commits, rerun
    `make NUM_PROCS=24 test-unit` at current HEAD for a fresh full-wrapper stamp.
- The current full-Gluon xfail is:
  - `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`
- Reason:
  - legacy M64 `64x64` layout sugar still needs producer-visible
    physical-family semantics for MMAv5 consumers.
- Classification:
  - not a current GB200 failure nodeid;
  - not a reason to block the immediate examples aggregate / CI freshness pass;
  - now an explicit follow-up fix item rather than a permanent tolerated
    design-debt note.
