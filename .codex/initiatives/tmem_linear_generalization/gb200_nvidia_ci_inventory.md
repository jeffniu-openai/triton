# GB200 NVIDIA CI Inventory

This file is the current CI-grounded validation baseline for the TMEM linear
generalization initiative.

For the current stabilization phase, scope is intentionally limited to the
NVIDIA GB200 lane from `.github/workflows/integration-tests-nvidia.yml`. This
GB300 devbox is treated as representative for Blackwell TMEM behavior in that
lane.

The current execution order follows the plan recorded in `memory.md`:
- inventory the GB200 lane first;
- classify failures into real bug vs stale expectation vs rewrite candidate vs
  local environment blocker; and then
- feed only the real compiler/backend gaps back into the TMEM cleanup/refactor
  work.

The exact branch-caused recovery order that sits on top of this inventory now
lives in `gb200_branch_recovery_plan.md`.

## Latest M64 Physical-Bitcast Refresh (2026-04-11 12:10 UTC)

- Latest pushed source checkpoint:
  - `85d8dbbf4` on `origin/codex/tmem`
- Scope:
  - M64 subview physical bitcast;
  - stale group-3 M64 behavior failures;
  - Gluon JIT cache-key hygiene for parameterized M64 layout tests.
- Grouped validation context:
  - a `python/test/gluon` group-3 sweep started before `47a07a37d` /
    `85d8dbbf4` finished with:
    - `3 failed, 4406 passed, 2041 skipped, 19348 deselected`
  - exact failures:
    - `test_tmem_subslice_block_m_64_parent_layout[linear]`
    - `test_block_m_64_mma[legacy]`
    - `test_block_m_64_mma[linear]`
  - this supersedes the old `316`-nodeid group-3 manifest for M64
    prioritization, but is itself stale for those exacts after `85d8dbbf4`.
- Fix summary:
  - physical bitcast query inference now normalizes inactive zero support bases
    before inversion when the normalized source view still exactly covers the
    source descriptor shape;
  - narrowed subview origins that are outside the reduced layout's in-dim size
    are preserved as physical row/col origins rather than being applied through
    the reduced layout;
  - dtype-changing physical bitcast scales the physical column origin by
    source/destination bitwidth under that fallback;
  - tests migrated from private `_reinterpret` to supported
    `slice/subview -> bitcast` only for physical-mapping-equivalent view
    changes.
- Focused current-head validation after `85d8dbbf4`:
  - build:
    - `PASSED`
  - lit:
    - `2 passed`
  - existing physical-bitcast controls:
    - `2 passed`
  - M64 subview/physical-bitcast exacts:
    - `4 passed`
  - block-M=64 MMA exacts:
    - `1 passed, 1 xfailed`
  - temp M64 bitcast repro:
    - `PASSED`
  - hygiene:
    - `git diff --check`
    - `PASSED`
- Inventory consequences:
  - the M64 exact failures from the latest group-3 sweep are closed at
    focused current-head scope;
  - the legacy M64 MMAv5 parameter is now an explicit xfail for remaining
    producer-visible physical-family design debt;
  - rerun grouped `python/test/gluon` from `85d8dbbf4` before replacing the
    broader group manifests with a new whole-shard green/red count.

## Latest Attention Revert And Two-CTA Assertion Refresh (2026-04-11 10:33 UTC)

- Latest pushed source checkpoint:
  - `13930b1ff` on `origin/codex/tmem`
- The supported descriptor bitcast API remains on the branch, but the
  attention example source has been restored to its pre-bitcast `_reinterpret`
  state at the user's request.
- Attention exact status at current `HEAD`:
  - `python/examples/gluon/01-attention-forward.py::test_op[False-dtype0-True-128-1024-48-4]`
  - `FAILED` in `gluon_to_ttgir`
  - symptom:
    - `LLVM ERROR: Invalid basis 32 for in-dim 'col' and out-dim 'dim1'. Basis must be less than the out-dim size.`
- Inventory consequence:
  - the examples/Gluon green aggregate from `4263ae61` / `49f1a0fd` is stale
    for current `HEAD`;
  - `gb200_current_branch_examples_gluon_failures.txt` and
    `gb200_branch_changed_examples_gluon_failures.txt` contain the attention
    exact again;
  - this is an intentional deferred migration target, not permission to add
    ad-hoc lowering selectors for private `_reinterpret` behavior.
- The stale two-CTA assertion bucket in
  `python/test/gluon/test_core.py::test_mma_shared_inputs` is closed:
  - current TTGIR type text may spell the field as `twoCTAs`;
  - the assertion now accepts both `two_ctas` and `twoCTAs`;
  - patched four-way function refresh:
    - group 1: `3830 passed, 490 skipped`
    - group 2: `2954 passed, 1366 skipped`
    - group 3: `1206 passed, 3114 skipped`
    - group 4: `2584 passed, 1736 skipped`
- Inventory consequence:
  - `gb200_current_branch_test_gluon_mma_shared_inputs_failures.txt` remains
    refreshed to `0` nodeids for the function-level surface;
  - the interrupted full `python/test/gluon` shards that still saw this
    assertion were pre-patch imports and are stale.
- Next validation frontier:
  - rerun wider grouped `python/test/gluon` from current `HEAD`, keeping the
    known attention exact separate from the supported bitcast API tests.

## Latest `triton_kernels` Persistent Matmul OOR Refresh (2026-04-11 09:44 UTC)

- Latest pushed code/test checkpoint before this docs-only refresh:
  - `ab8ff6e6444b1cf8521a06941d6cf1b8a25e3217`
- Focused current-head remeasurement:
  - before the fix, a full `python/triton_kernels/tests` rerun reported
    `8 failed, 1270 passed, 1840 skipped`;
  - failures were persistent Blackwell matmul variants with shared-memory
    metadata above the `232448` byte hardware limit;
  - the representative bad compile artifact used `shared=278632`, while the
    good/fixed artifact uses `shared=214120`.
- Root cause:
  - ordinary split-load/store replay constructed shape-local
    `ttng.tmem_subslice` descriptors;
  - layout selection nevertheless used the preserved full/root physical query;
  - the selected register family forced a `ttg.convert_layout` scratch before
    the persistent matmul epilogue.
- Fix summary:
  - ordinary replayed subslices select register layouts from the subview-local
    descriptor type;
  - explicit `tmem_physical_bitcast` roots still use the preserved physical
    query, matching the supported physical bitcast API contract.
- Validation:
  - build:
    - `PASSED`
  - lit:
    - `2 passed`
  - representative OOR repro:
    - `1 passed in 5.38s`
  - focused persistent fp8/mxfp4 matmul slice:
    - `16 passed, 6 skipped in 31.85s`
  - full 4-way `python/triton_kernels/tests` refresh:
    - group 1:
      - `674 passed, 782 skipped, 4365 deselected in 2296.49s`
    - group 2:
      - `533 passed, 923 skipped, 4365 deselected in 1863.14s`
    - group 3:
      - `394 passed, 1062 skipped, 4365 deselected in 1192.94s`
    - group 4:
      - `776 passed, 677 skipped, 4368 deselected in 555.61s`
    - aggregate:
      - `2377 passed, 3444 skipped`
      - `0 failed`
- Inventory consequences:
  - `gb200_current_branch_triton_kernels_matmul_oor_refresh_failures.txt`
    records this bucket as `0` nodeids;
  - older whole-directory `python/triton_kernels/tests` counts are stale;
  - the next validation frontier is wider grouped Gluon sweeps.

## Latest Focused `test_core.py` Manifest Refresh (2026-04-11 07:25 UTC)

- Latest pushed code/test checkpoint before this docs-only refresh:
  - `42f62fb143630512aff9a45b7b3846a02d9935fa`
- Focused current-head remeasurement:
  - old manifest:
    - `gb200_current_branch_test_core_group3_focus_e70a3aa09_failures.txt`
    - `332` nodeids before refresh
  - initial rerun:
    - `330 passed, 2 failed in 132.02s`
- Both failures were stale expectations:
  - legacy `64x128` split-N roundtrip now expects the row-encoded
    `1048576` packet offset instead of the old immediate-like `64` value;
  - the block-basis `ld.red` negative now expects the clean unsupported
    descriptor-view diagnostic:
    - `TMEM layout '32x32b' unsupported for descriptor view`
- Exact post-refresh rerun:
  - `2 passed in 3.46s`
- Full focused manifest rerun with the refreshed diagnostic nodeid:
  - `332 passed in 125.78s`
- Inventory consequences:
  - `gb200_current_branch_test_core_group3_focus_e70a3aa09_failures.txt`
    is refreshed from the stale `332` nodeids to `0` nodeids;
  - the old merge-base-present / merge-base-missing split files for this focus
    list are historical reduction artifacts and should not be used as the
    live branch-red list without a fresh rerun;
  - the next validation frontier remains `triton_kernels` and wider grouped
    Gluon sweeps from current head.

## Latest Runtime-Matrix File Refresh (2026-04-11 07:15 UTC)

- Latest pushed code/test checkpoint before this docs-only refresh:
  - `5b619c9e3dfd5ca3479f644dada7813bd198e34a`
- Full-file current-head remeasurement:
  - `python/test/gluon/test_tmem_runtime_matrix.py`
  - command:
    - `CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 TRITON_CACHE_DIR=/tmp/triton-cache-tmem-runtime-full-<timestamp> PYTHONPATH=python:. pytest -s --tb=short -vv python/test/gluon/test_tmem_runtime_matrix.py`
  - result before the final assertion refresh:
    - `1437 passed, 354 skipped, 1 failed in 1226.69s`
- The lone failure was a stale TTGIR spelling assertion:
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_indexed_view_canonicalized[128]`
  - current codegen/runtime was already correct:
    - exact `tcgen05.cp.cta_group::1.128x128b` opcode
    - correct runtime output
  - the stable invariant is now `ttng.tmem_physical_layout` on the legacy
    `tensor_memory_encoding` root for this indexed-view canonicalization path.
- Exact post-fix rerun:
  - `test_tmem_runtime_matrix_cp_no_scales_indexed_view_canonicalized`
  - `1 passed in 3.31s`
- Inventory consequences:
  - `gb200_current_branch_test_tmem_runtime_matrix_focus_e70a3aa09_failures.txt`
    is refreshed from the stale `413` nodeids to `0` nodeids;
  - the branch-added TMEM runtime-matrix bucket no longer has a live red list
    at this checkpoint;
  - older whole-`test-gluon` shard counts still remain stale until the grouped
    Gluon lane is rerun from current head.

## Latest M64 Row/Col Split-N And Half-Row Refresh (2026-04-11 06:51 UTC)

- Current local checkpoint on top of:
  - `49f1a0fd2e1b2e8ed7d144c470bc7838b2b5ff4a`
- Fix summary:
  - M64 row/col-permuted split-N direct ld/st remains a positive surface;
  - misaligned exact `16x32bx2` warp anchors are rejected before PTX lowering;
  - simple row/col-permuted `64xN` f32 TMEM-linear roots select the canonical
    aligned M64 split-N register layout;
  - direct half-row and higher-rank row-half descriptor views now report clean
    unsupported diagnostics.
- Validation:
  - build:
    - `make -j8`
    - `PASSED`
  - full row/col-permuted M64 split-N sweep plus auto guards:
    - `226 passed in 35.56s`
  - split-N controls plus half-row and exotic clean-negative guards:
    - `45 passed in 5.32s`
- Inventory consequence:
  - `gb200_current_branch_test_tmem_runtime_matrix_splitn_rowcol_refresh_failures.txt`
    remains refreshed to `0` nodeids;
  - `gb200_current_branch_test_core_branch_added_splitn_expectation_refresh_failures.txt`
    remains refreshed to `0` nodeids;
  - `gb200_current_branch_test_gluon_halfrow_stale_negative_failures.txt`
    is now refreshed to `0` nodeids;
  - the older full Gluon shard aggregate counts are still stale for
    prioritization until rerun from this new head.

## Historical Attention Bitcast Migration Checkpoint (2026-04-11 06:07 UTC, Superseded)

- A later examples/Gluon refresh found one current-branch attention exact red:
  - `python/examples/gluon/01-attention-forward.py::test_op[False-dtype0-True-128-1024-48-4]`
- Current committed checkpoint:
  - `4263ae61b80e4f20e5c372a3c2cf5a7538d67620`
  - pushed to `origin/codex/tmem`
  - this exact is green after the supported TMEM descriptor bitcast API and
    attention migration.
- Validation:
  - `python/examples/gluon/01-attention-forward.py::test_op[False-dtype0-True-128-1024-48-4]`
    - `PASSED`
  - full `python/examples/gluon/` aggregate:
    - `821 passed, 74 skipped in 134.89s`
- Inventory consequence:
  - historical only: this examples-lane green result was superseded when the
    attention source was reverted at `13930b1ff`;
  - `gb200_current_branch_examples_gluon_failures.txt`,
    `gb200_branch_changed_examples_gluon_failures.txt`,
    `gb200_current_branch_examples_convolution_failures.txt`, and
    `gb200_current_branch_examples_multicta_failures.txt` were refreshed to
    empty at that historical checkpoint, but the first two now contain the
    reverted attention exact again;
  - continue with staged broad validation while keeping the attention migration
    as an explicit deferred supported-view rewrite.

## Latest MMAv5 Direct-Load Family Fix Refresh (2026-04-10 19:30 UTC)

- The previous full-lane GB200 census remains the last whole-suite Gluon
  snapshot, but the largest merge-base-present compiler bucket from that
  census is now remeasured after a targeted direct-load fix.
- Root cause:
  - `getDistributedLayoutForTmemLdSt(...)` in
    `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` was letting the 16-bit
    unpacked reinterpret rescue recurse into the generic `I32x32b` builder
    even for ordinary direct MMAv5 accumulator loads.
  - that drifted current-head direct loads away from the legacy-equivalent
    packed bitwidth-packing path and changed the emitted TTGIR/PTX family for
    representative old-mainline `test_mma_shared_inputs[...]` cases.
- Fix:
  - restrict that recursion to the actual reinterpret/query-style
    `I32x32b` path:
    - `bitwidth == 16`
    - `atom == TMemAccessAtom::I32x32b`
    - `!allowSplitNFastPath`
  - ordinary direct MMAv5 accumulator loads now stay on the standard
    bitwidth-packing path.
- Validation:
  - `make -j8`
  - representative current-fail / merge-base-pass exacts now pass again:
    - `False-ctas_per_cga0-1-1-1-64-0-0-warps0-16-False-True-acc_dtype3`
    - `False-ctas_per_cga2-1-1-1-64-64-128-warps0-16-False-True-acc_dtype3`
    - `False-ctas_per_cga2-2-4-1-64-32-32-warps2-16-True-True-acc_dtype7`
    - `True-ctas_per_cga1-2-4-1-64-128-128-warps0-16-False-True-acc_dtype3`
  - fresh four-way function rerun:
    - `pytest --splits 4 --group 1 -k 'test_mma_shared_inputs' python/test/gluon/test_core.py`
      - `3830 passed, 490 skipped`
    - `pytest --splits 4 --group 2 -k 'test_mma_shared_inputs' python/test/gluon/test_core.py`
      - `2954 passed, 1366 skipped`
    - `pytest --splits 4 --group 3 -k 'test_mma_shared_inputs' python/test/gluon/test_core.py`
      - `1206 passed, 3114 skipped`
    - `pytest --splits 4 --group 4 -k 'test_mma_shared_inputs' python/test/gluon/test_core.py`
      - `2584 passed, 1736 skipped`
- Consequence:
  - `gb200_current_branch_test_gluon_mma_shared_inputs_failures.txt` is now
    refreshed to `0` exact nodeids.
  - the old `4685`-nodeid MMAv5 manifest and the pre-fix full-group counts are
    now historical reduction artifacts for this function and must not be used
    as the live backlog without rerunning the broader Gluon shards.
  - confirmed current branch-recovery work now starts with the two example
    regressions, while the branch-added PTX-expectation / stale-negative TMEM
    tails remain separate.

## Latest Full GB200 Census Refresh (2026-04-10 18:48 UTC)

- Current branch / baseline:
  - `HEAD`:
    - `1f0b0d01c`
  - this refresh supersedes the older intermediate story that the live GB200
    backlog was only descriptor-chain plus examples
- Full GB200 lane state on the current branch:
  - `make test-lit`
    - green
    - `248 passed, 2 unsupported`
  - `make test-cpp`
    - green
    - `240 / 240 passed`
  - `make NUM_PROCS=24 test-unit`
    - green
    - `15153 passed, 5492 skipped, 101 warnings`
    - important sublane:
      - `python/triton_kernels/tests`
        - `2377 passed, 3444 skipped`
  - `make test-regression`
    - green
    - `1090 passed, 216 skipped`
  - `make NUM_PROCS=24 test-gsan`
    - green
    - `20 / 20 passed`
  - `make test-microbenchmark`
    - green
  - `make test-proton`
    - red
    - `11 failed, 114 passed`
    - merge-base rerun in `/root/code/triton-mergebase-ci` shows the same
      `11` exact failures, so keep this bucket classified as preexisting
- Full `python/test/gluon/ + python/tutorials/gluon/` current-head refresh
  with isolated per-GPU caches:
  - group `1 / 4`
    - `4105 failed, 1329 passed, 1014 skipped, 19343 deselected`
  - group `2 / 4`
    - `265 failed, 2399 passed, 3784 skipped, 19343 deselected`
  - group `3 / 4`
    - `316 failed, 4091 passed, 2041 skipped, 19343 deselected`
  - group `4 / 4`
    - `3 failed, 5578 passed, 866 skipped, 19344 deselected`
- Refreshed current-head Gluon/tutorial classification:
  - real current-head merge-base-present regression bucket:
    - `python/test/gluon/test_core.py::test_mma_shared_inputs[...]`
    - semantic manifest:
      - `gb200_current_branch_test_gluon_mma_shared_inputs_failures.txt`
      - `4685` exact nodeids from the latest four-way sweep
    - merge-base proof:
      - representative exact reruns pass on merge-base and fail on the current
        branch:
        - `False-ctas_per_cga0-1-1-1-64-0-0-warps0-16-False-True-acc_dtype3`
        - `False-ctas_per_cga2-1-1-1-64-64-128-warps0-16-False-True-acc_dtype3`
        - `False-ctas_per_cga2-2-4-1-64-32-32-warps2-16-True-True-acc_dtype7`
        - `True-ctas_per_cga1-2-4-1-64-128-128-warps0-16-False-True-acc_dtype3`
    - representative current-head symptom:
      - wrong-code / `torch.testing.assert_close(...)` mismatch, not a clean
        compile-time rejection
  - branch-added opinionated PTX-expectation tail:
    - `gb200_current_branch_test_gluon_splitn_expectation_tail_failures.txt`
    - `1` exact
    - absent on merge-base
    - current symptom:
      - PTX immediate-offset expectation `64` vs actual `1048576`
  - branch-added stale-negative / support-broadened tail:
    - `gb200_current_branch_test_gluon_halfrow_stale_negative_failures.txt`
    - `3` exacts
    - absent on merge-base
    - current symptom:
      - `Failed: DID NOT RAISE CompilationError`
- Examples lane remains independently red on the current branch:
  - `python/examples/gluon/02-convolution.py`
    - `48 failed in 8.38s`
    - stable symptom:
      - `OutOfResources: shared memory, Required: 262208, Hardware limit: 232448`
    - merge-base rerun:
      - `48 passed in 7.30s`
  - `python/examples/gluon/03-matmul-multicta.py`
    - `14 failed, 68 passed, 14 skipped in 65.24s`
    - stable symptom:
      - wrong-code / `AssertionError: Tensor-likes are not close!`
    - merge-base rerun:
      - `82 passed, 14 skipped in 37.96s`
- Cache / contamination note:
  - the fresh four-way Gluon refresh used isolated per-GPU `TRITON_CACHE_DIR`
    values and still reproduced the large `test_mma_shared_inputs[...]`
    bucket, so this is not explained by a trivial on-disk cache collision
  - however, raw shard counts still overstate independently failing exacts:
    - one shard-failing exact
      `False-ctas_per_cga2-1-1-1-64-0-32-warps2-16-False-True-acc_dtype3`
      passes in a fresh isolated current-head process
  - treat that as process/device contamination after bad kernels, or some
    other missing cache-key / global-state reuse bug, until a dedicated
    harness pass proves otherwise
- Practical consequence:
  - the live GB200 branch-recovery queue is now:
    1. the merge-base-present MMAv5 `test_mma_shared_inputs[...]` bucket
    2. the two merge-base-present example regressions
    3. the branch-added PTX-expectation and stale-negative tails
  - keep the separate reinterpret-contract rewrite candidates and the
    preexisting Proton failures out of the branch-recovery queue while this
    MMAv5 work is active

## Latest Descriptor-Chain Recovery Checkpoint (2026-04-10 20:35 UTC)

- The branch-only TMEM/compiler bucket is now green again:
  - `gb200_current_branch_test_core_branch_added_descriptor_chain_refresh_failures.txt`
    - now `0` nodeids
  - fresh reruns:
    - isolated real failures:
      - `linear_m64_32x32b_4w`
      - `linear_m64_32x32b_8w`
      - both now pass
    - full `test_tmem_descriptor_chain_matrix` surface:
      - `26 passed in 15.94s`
    - refreshed old `12`-nodeid manifest:
      - `12 passed in 9.84s`
- Root cause / fix summary:
  - the memdesc-aware TMEM reg-layout selector still treated a user request for
    `instr_variant="32x32b"` as a hard `I32x32b` atom requirement on rank-2
    `M=64`, `f32` descriptor views
  - that forced the plain `linear_m64 32x32b` views onto scalar `32x32b.x1`
    raw-query lowering, while the passing split-N sibling already used the
    packed `16x32bx2` direct family
  - the fix centralizes requested-atom matching so those `M=64`, `f32`
    descriptor-view `32x32b` requests may be satisfied by the packed
    `I16x32bx2` direct family
- Important contamination learning:
  - a fresh-process isolation matrix showed that only the two plain
    `linear_m64_32x32b_{4w,8w}` exacts were independently red
  - the other `10` nodeids from the old descriptor-chain manifest already
    passed in fresh processes and were fallout after the first CUDA fault
- Practical consequence:
  - the live GB200 branch backlog is now only the two example regressions:
    - `python/examples/gluon/02-convolution.py`
    - `python/examples/gluon/03-matmul-multicta.py`

## Historical Verified Red List After `c6990c1e9` (2026-04-10 20:05 UTC)

- Fresh isolated current-head reruns confirm that the live GB200 branch backlog
  is now:
  - `12` descriptor-chain exacts
    - `gb200_current_branch_test_core_branch_added_descriptor_chain_refresh_failures.txt`
    - rerun result:
      - `12 failed in 3.73s`
  - `48` convolution example exacts
    - `python/examples/gluon/02-convolution.py`
    - rerun result:
      - `48 failed in 8.38s`
  - `14` multicta example exacts
    - `python/examples/gluon/03-matmul-multicta.py`
    - rerun result:
      - `14 failed, 68 passed, 14 skipped in 65.24s`
- Fresh merge-base checks revalidate the old-mainline boundary:
  - the TMEM descriptor-chain / split-N / warpx2 / non-surjective parser exacts
    do not exist anywhere under merge-base `python/test/gluon/`
  - `python/examples/gluon/02-convolution.py`
    - merge-base rerun:
      - `48 passed in 7.30s`
  - `python/examples/gluon/03-matmul-multicta.py`
    - merge-base rerun:
      - `82 passed, 14 skipped in 37.96s`
- Practical consequence:
  - the branch-only TMEM/compiler recovery work is now just the
    `12`-nodeid descriptor-chain manifest
  - the remaining old-mainline regression lane is the two example files above
  - the earlier larger shard counts remain useful only as historical reduction
    artifacts and cache/process-fallout evidence, not as the live red list

## Latest Non-Surjective `[128,4]` Recovery Checkpoint (2026-04-10 18:45 UTC)

- The shared `[128,4]` direct-view tail is now gone:
  - `gb200_current_branch_test_tmem_runtime_matrix_warpx2_candidate_refresh_failures.txt`
    - now `0` nodeids
  - exact reruns now green:
    - `test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive`
    - `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_candidate_positive`
    - `test_tensor_memory_linear_layout_non_surjective_reg_layout_parses`
    - nearby parser control
      `test_tensor_memory_linear_layout_warpx2_like_rows_parse[...]`
      stays green
- Root cause / fix summary:
  - the raw-query helper already had a preserve-non-canonical-leaf path for
    raw `tensor_memory_linear` descriptors
  - the direct-view row-anchor diagnostic and Gluon raw-query wrapper were not
    actually using that same preserved layout
  - the fix makes both consume the same raw-query model, so the preserved
    zero-row basis survives through the unsupported-view gate and the Gluon
    raw-query path
- Practical consequence:
  - the live TMEM/compiler backlog is now reduced to the `12`-nodeid
    descriptor-chain manifest, plus the separate examples lane.

## Latest Split-N Recovery Checkpoint (2026-04-10 18:20 UTC)

- The dominant TMEM runtime bucket is now gone:
  - `gb200_current_branch_test_tmem_runtime_matrix_splitn_rowcol_refresh_failures.txt`
    - now `0` nodeids
  - isolated rerun result after the canonical split-N lowering fix:
    - `208 passed in 21.52s`
- The adjacent split-N PTX-expectation exact is also gone:
  - `gb200_current_branch_test_core_branch_added_splitn_expectation_refresh_failures.txt`
    - now `0` nodeids
  - `test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]`
    - now passes
- Root cause / fix summary:
  - Gluon/layout synthesis already knew the canonical `M=64` split-N register
    family
  - lowering did not preserve that symmetry and could reinterpret the same
    family as scalar `I32x32b`
  - exposing `getCanonicalM64SplitNLayout(...)` to lowering and preferring the
    packed `I16x32bx2` family when the direct register layout already matches
    that canonical split-N family fixed the runtime bucket
- What remains live after the split-N fix:
  - descriptor-chain exacts:
    - still `12 / 12` failing
  - shared `[128, 4]` representability tail:
    - `2` warpx2 candidate-positive runtime-matrix exacts still fail
    - `1` isolated frontend exact still fails
- Practical consequence:
  - the next compiler slice should move to the shared non-surjective
    `[128, 4]` direct-view representability gap, not back to split-N.

## Latest Trace-Driven Bucket Split (2026-04-10 17:25 UTC)

- The remaining live Gluon compiler buckets now have representative passing vs
  failing TTGIR/PTX/query traces, which changes the recovery ordering.
- Descriptor-chain exacts (`12` live nodeids) split into at least two
  structural subfamilies:
  - `linear_m64_*` / split-N descriptor-chain cases:
    - raw query stays packed (`atom=4`)
    - PTX uses `tcgen05.{st,ld}.sync.aligned.16x32bx2.x32.b32`
    - still faults at launch with `RuntimeError: CUDA error: misaligned address`
  - `linear_mixed_*` descriptor-chain cases:
    - raw query stays scalar (`atom=0`)
    - PTX uses the same `32x32b.x128` family and the same visible address
      sequence as the passing identity control
    - implication:
      - mixed-layout basis information is likely being erased or ignored before
        scalar-family access mapping is formed
- The dominant runtime-matrix bucket (`208` split-N row/col-permuted exacts)
  now has a clean planner-side signature:
  - passing control:
    - packed raw query on raw memdesc `64x2`
    - emits `16x32bx2.x2.b32`
  - failing permuted case:
    - logs `packed16 support precondition fail`
    - logs `packed16 support skip: no packed mem layout`
    - degrades to scalar `atom=0` `32x32b.x1`
  - implication:
    - the active bug is packed-family / packed-mem-layout discovery for
      permuted `M=64` direct views, not a generic late packet-offset tweak
- The smaller frontend-style tail is also unified:
  - the two `warpx2` candidate-positive runtime-matrix exacts and the isolated
    frontend exact fail for the same direct-view representability reason on a
    non-surjective `[128, 4]` descriptor view:
    - required row anchors `32,64` are not directly representable
- Practical consequence:
  - the next compiler fix should start with the split-N packed-family discovery
    gap, then the shared `[128, 4]` representability gap, and only then the
    descriptor-chain mixed-layout bucket.

## Latest Runtime-Matrix / Frontend Refresh (2026-04-10 16:55 UTC)

- The old `413`-nodeid focused `test_tmem_runtime_matrix.py` bucket is no
  longer opaque after isolated family reruns:
  - `208 / 208`
    `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[...]` exacts
    still fail cleanly with `RuntimeError: CUDA error: misaligned address`
    - manifest:
      - `gb200_current_branch_test_tmem_runtime_matrix_splitn_rowcol_refresh_failures.txt`
  - the next four largest families are now clean:
    - `48 / 48`
      `test_tmem_runtime_matrix_ld_red_row_permuted_linear_layout[...]` pass
    - `33 / 33` `test_tmem_runtime_matrix_ldst_scales_variant_sweep[...]` pass
    - `32 / 32`
      `test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_geometry_sweep[...]`
      pass
    - `25 passed, 5 skipped`
      `test_tmem_runtime_matrix_cp_no_scales[...]`
  - the remaining `62` exacts reduce to only `2` live failures:
    - `test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive`
    - `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_candidate_positive`
    - both fail at `tmem.get_reg_layout()` with
      `TMEM layout 'auto' unsupported for descriptor view tensor_memory_descriptor<fp32, ['128', '4'], ...>`
    - manifest:
      - `gb200_current_branch_test_tmem_runtime_matrix_warpx2_candidate_refresh_failures.txt`
- The previously isolated `test_fpsan` tail is now gone:
  - rerunning the full branch-added/renamed `test_fpsan.py` exact manifest now
    gives `11 passed, 1 skipped`
  - implication:
    - treat the earlier isolated `test_fpsan` failure file as historical shard
      fallout, not as a live current-head bucket
- The remaining isolated `test_frontend.py` exact stays red:
  - `python/test/gluon/test_frontend.py::test_tensor_memory_linear_layout_non_surjective_reg_layout_parses`
  - current symptom matches the two live warpx2 candidate positives:
    - `TMEM layout 'auto' unsupported for descriptor view tensor_memory_descriptor<fp32, ['128', '4'], ...>`
- Current interpretation:
  - the runtime-matrix lane is now driven primarily by one real split-N
    runtime bucket (`208` exacts) plus one smaller frontend/get-reg-layout
    surface (`2` warpx2 positives + `1` frontend exact), not by the old mixed
    `413`-nodeid census.

## Latest Examples / Merge-Base Classification Refresh (2026-04-10 17:00 UTC)

- The examples lane was refreshed on the current head rather than relying only
  on the earlier broad census:
  - `python/examples/gluon/02-convolution.py`
    - `48 / 48` exacts still fail
    - stable symptom:
      - `OutOfResources: shared memory, Required: 262208, Hardware limit: 232448`
  - `python/examples/gluon/03-matmul-multicta.py`
    - `14 / 14` exacts still fail
    - stable symptom:
      - wrong-code / `AssertionError: Tensor-likes are not close!`
- Merge-base boundary is now explicit for the remaining Gluon buckets:
  - the active TMEM-focused red functions are branch-added coverage and do not
    exist on merge-base:
    - `test_tmem_descriptor_chain_matrix`
    - `test_tmem_linear_roundtrip_splitn_shapes`
    - `test_tmem_linear_roundtrip_variant_sweep`
    - `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep`
    - `test_tmem_runtime_matrix_cp_no_scales_warpx2_{01_23,02_13}_candidate_positive`
    - `test_tensor_memory_linear_layout_non_surjective_reg_layout_parses`
  - the examples red functions already exist on merge-base:
    - `python/examples/gluon/02-convolution.py::test_op`
    - `python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch`
  - those merge-base example files are already known green from the earlier
    full-file reruns, so the examples lane remains branch-caused recovery work
    on old upstream functions rather than brand-new branch-only coverage.

## Latest Focused-Core Refresh (2026-04-10 16:45 UTC)

- HEAD:
  - `460c12c63`
- Worktree:
  - clean
- Refreshed local baseline after `make -j8`:
  - `make test-lit`
    - green
    - `248 passed, 2 unsupported`
- The older split `test-gluon` shard-3 focused-core story is now materially
  narrower after fresh exact reruns:
  - merge-base-present focused `test_core.py` exacts:
    - `146 / 146` pass on the current branch
    - implication:
      - the previously prioritized merge-base-present focused core bucket is no
        longer a live current-head red surface
  - branch-added / branch-changed focused `test_core.py` exacts:
    - initial shard reduction:
      - `48` raw failing exacts
    - exact isolated reduction:
      - `21 / 21` `test_tmem_linear_roundtrip_variant_sweep[...]` exacts pass
        in isolation
      - `12 / 26` `test_tmem_descriptor_chain_matrix[...]` exacts still fail
        cleanly with `RuntimeError: CUDA error: misaligned address`
      - the remaining `14 / 26`
        `test_tmem_descriptor_chain_matrix[...]` exacts pass in isolation
      - `1 / 1` `test_tmem_linear_roundtrip_splitn_shapes[...]` exact still
        fails only because the PTX offset-immediate expectation is stale while
        runtime correctness is already green
- New current exact manifests for the independently failing focused-core tail:
  - `gb200_current_branch_test_core_branch_added_descriptor_chain_refresh_failures.txt`
    - `12` exact nodeids
  - `gb200_current_branch_test_core_branch_added_splitn_expectation_refresh_failures.txt`
    - `1` exact nodeid
- Current interpretation:
  - the branch-added focused-core surface is no longer “descriptor-chain `26`
    plus variant sweep `21`”;
  - the real independent bug bucket is the `12`-nodeid descriptor-chain
    subfamily, not the whole contaminated shard reduction; and
  - the variant-sweep tail is now firmly classified as shard fallout after the
    earlier CUDA fault, not as another independent TMEM family.
- Current next action:
  - drive the next compiler fix from the `12`-nodeid descriptor-chain manifest
    and keep the split-N offset-expectation node in the stale/opinionated
    bucket unless a broader heuristic change makes it worth updating.

## Latest Current-Branch Reduction (2026-04-10)

### MMAv5 Row-Plan Propagation Checkpoint

- The current dirty worktree extends the earlier MMAv5 source-root fix into the
  remaining alloc-cloning / higher-rank producer paths:
  - added `setExplicitMMAv5RootRowPlanIfNeeded(TMEMAllocOp)` in
    `TensorMemoryUtils.{h,cpp}`;
  - replaced duplicated annotation ladders in
    `AccelerateMatmul.cpp`, `PromoteLHSToTMem.cpp`, and `python/src/gluon_ir.cc`;
  - propagated or reinitialized explicit MMAv5 root row plans through cloned
    allocs in:
    - `MMAv5PipelineUtility.cpp`
    - `WSCodePartition.cpp`
    - `WSDataPartition.cpp`
    - `HoistTmemStore.cpp`
    - `InsertTmemAref.cpp`
- Fresh broad current-branch rerun after `make -j8`:
  - `make NUM_PROCS=24 test-unit`
    - `134 failed, 15019 passed, 5492 skipped in 317.52s`
- The broad unit surface is now materially smaller and cleaner than the older
  `162`/`182`/`2098` manifests:
  - exact failing files:
    - `python/test/unit/language/test_matmul.py`
      - `3` exact nodeids
    - `python/test/unit/language/test_tensor_descriptor.py`
      - `3` exact nodeids
    - `python/test/unit/language/test_warp_specialization.py`
      - `128` exact nodeids
  - exact current-head manifests:
    - `gb200_current_branch_test_unit_matmul_refresh_failures.txt`
    - `gb200_current_branch_test_unit_tensor_descriptor_refresh_failures.txt`
    - `gb200_current_branch_test_unit_warp_specialization_refresh_failures.txt`
    - `gb200_current_branch_test_unit_rowanchor_refresh_failures.txt`
- Fresh isolated current-branch reruns completed so far:
  - `python/test/unit/language/test_tensor_descriptor.py`
    - `3 failed, 2601 passed, 110 skipped`
  - `python/test/unit/language/test_warp_specialization.py`
    - `128 failed, 1471 passed, 202 skipped`
  - `python/test/unit/language/test_matmul.py`
    - exact current-head failing nodeids already reduced to:
      - `test_simple_persistent_matmul[False-4-64-128-32]`
      - `test_simple_persistent_matmul[False-4-64-16-16]`
      - `test_simple_persistent_matmul[False-8-64-128-32]`
- Fresh merge-base confirmation completed so far:
  - `python/test/unit/language/test_tensor_descriptor.py`
    - `2604 passed, 110 skipped`
  - this reconfirms that the whole current-head tensor-descriptor bucket is
    branch-local
- Current interpretation:
  - the explicit MMAv5 row-plan propagation patch is a real improvement and
    should be treated as the new baseline;
  - the older larger unit manifests remain useful as branch-vs-main evidence,
    but they are historical with respect to the current dirty worktree; and
  - the remaining unit red surface now looks like a much narrower
    root-preserving descriptor-view row-anchor problem.
- Current root-cause hypothesis from the latest reproducer work:
  - the planner is collapsing a root-backed `128`-row MMAv5 contract into the
    `64`-row active descriptor view too early for root-preserving views;
  - that makes direct `tcgen05.ld/st` representability checks reason about the
    wrong physical row-anchor space for `memdesc_index` / higher-rank views;
  - the strongest clue from the saved reproducer is that
    `activePhysicalRows=64` shows up even while the backing explicit row plan
    still wants anchors `32,64,128`.
- Still pending from this checkpoint:
  - refreshed merge-base full-file reruns of:
    - `python/test/unit/language/test_matmul.py`
    - `python/test/unit/language/test_warp_specialization.py`
  - the current-branch `python/triton_kernels/tests` inventory run
  - use the older merge-base-green file reruns in this folder as provisional
    branch-vs-main evidence until those refreshes complete

- The current dirty worktree has now moved past the older raw-first LLVM probe.
  The live fix is producer-side:
  - `getMMAv5RootRowPlan(...)` is now the shared row-plan helper for all
    `M=64` MMAv5 root TMEM families, including mutable accumulators and
    source-initialized operand roots;
  - `PromoteLHSToTMem` now sets that explicit row-plan contract on the
    non-mutable TMEM roots it creates; and
  - the helper now accepts higher-rank roots whose trailing TMEM row dimension
    is `64`, because those roots still use the full `128`-row backing tile.
- Fresh exact reruns after `make -j8` show that the earlier old-mainline
  root-load and regression slice is repaired on the current dirty tree:
  - passing exacts:
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
    - `python/test/unit/language/test_matmul.py::test_lhs_in_tmem[float32-False-64-128-32]`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-1-64-512-32-2-float32-tensorfloat32]`
    - `python/test/regression/test_cast_matmul.py::test_cast_matmul[768-768-1024-16-64-16-bfloat16-float16-float16]`
  - still green controls:
    - `test_tmem_descriptor_chain_matrix[...]`
    - `test_tmem_linear_roundtrip_splitn_shapes[...]`
    - `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[...]`
    - `test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-32-warps2-8-False-True-acc_dtype0]`
    - `test_block_m_64_mma[linear]`
  - file-level reruns:
    - `python/test/regression/test_cast_matmul.py`
      - `1080 passed, 216 skipped`
    - `python/test/unit/language/test_matmul.py`
      - `3 failed, 758 passed, 4780 skipped`
- The remaining fresh exact current-head failures are now concentrated in two
  structural buckets:
  - descriptor-view row-anchor materialization:
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-128-32]`
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-16-16]`
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-8-64-128-32]`
    - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float32]`
    - representative failure:
      `unsupported tensor memory descriptor view for direct tcgen05.ld/st: required row anchors 32,64 are not directly representable in the descriptor view`
  - warp-specialization higher-rank MMAv5 roots:
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
      - `CUDA error: misaligned address`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_persistent_forward[False-8-True-2-128-64-1024-1024]`
      - `65535 / 131072` mismatches (`50.0%`)
- The older broad unit XML and regression manifests are now explicitly
  historical for the current dirty tree:
  - the `2098`-nodeid XML manifest still proves the branch-local nature of the
    broad unit surface against merge-base; but
  - its counts, and the old `234`-nodeid regression manifest, must be
    refreshed before they are used as current-head failure totals.

## Workflow Coverage

The GB200/NVIDIA workflow currently runs:

- `make test-lit`
- `make NUM_PROCS=24 test-unit`
- `make NUM_PROCS=24 test-gluon`
- `make NUM_PROCS=24 test-gsan`
- `make test-regression`
- `make test-microbenchmark`
- `make test-cpp`
- `make test-proton`

Out of scope for this GB200-only triage pass:

- AMD CI
- macOS CI
- the H100-only `make test-interpret` lane

## Repo Test Surface Snapshot

Current local inventory relevant to the GB200 lane:

- `ctest -N` reports `240` C++ unit tests.
- `test/` currently contains about `250` `.mlir` / `.ll` lit inputs.
- Python test-file counts:
  - `python/test/gluon`: `7`
  - `python/test/unit`: `55`
  - `python/test/regression`: `2`
  - `python/test/gsan`: `4`
  - `python/triton_kernels/tests`: `16`
- `python/test/gluon/test_core.py` +
  `python/test/gluon/test_tmem_runtime_matrix.py` currently collect
  `19753` tests together.

## Current Confirmed GB200 Red List

As of 2026-04-10, after `make -j8`, the initial exact-nodeid reruns, and the
first executed GB200 CI census on this devbox, the currently confirmed red list
is:

1. Reinterpret-contract rewrite candidate
- Nodeid:
  - `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[legacy]`
- Current result:
  - `4096 / 8192` mismatches
- Current intent:
  - use `_reinterpret(...)` plus knowledge of the old TMEM physical mapping to
    zero one physical half-band and then overwrite three small physical
    subfragments
- Classification:
  - `REINTERPRET_CONTRACT_TEST_TO_REWRITE`
- Current policy:
  - do not treat this as a proven compiler bug until the test is restated with
    explicit TMEM descriptor/view APIs (`slice/index/reshape/permute`) or
    otherwise narrowed to reinterpret's actual contract
  - current status after rewrite exploration:
    - direct logical-slice rewrites do not encode the intended borrowed
      physical fragment behavior; and
    - the more faithful explicit quarter-band reorder is currently blocked by
      missing TMEM surface coverage in the compiler/backend
  - missing coverage now observed:
    - reordered `M=64` quarter-band views fail direct ld/st with unsupported
      row anchors `32,64`; and
    - row-split higher-rank variants fail TMEM view/index inference before
      lowering

2. Reinterpret-contract rewrite candidate
- Nodeid:
  - `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64_parent_layout[linear]`
- Current result:
  - `2048 / 8192` mismatches
- Current intent:
  - write the same physical half-band update through the parent TMEM register
    layout rather than the subview's own direct layout
- Classification:
  - `REINTERPRET_CONTRACT_TEST_TO_REWRITE`
- Current policy:
  - same as above; this test currently encodes implicit physical-mapping
    assumptions rather than the newer explicit TMEM view contract
  - current status after rewrite exploration:
    - same blocker as the legacy case: the explicit descriptor/view rewrite is
      not landable yet because the necessary reordered M64 views are not fully
      supported by the current TMEM API/backend surface

3. Recovered direct split-N TMEM bucket
- Exact nodeids rerun clean on an isolated GPU after the latest
  `TensorMemoryUtils.cpp` planner fix:
  - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-layout9-64-128-32x32b_splitn-8-16x32bx2]`
  - `python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]`
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[identity-identity-2-32x32b_splitn]`
- Current result:
  - all three exact nodeids pass clean
- Classification:
  - `FIXED_ON_CURRENT_WORKTREE`
- Current interpretation:
  - the direct `M=64` split-N ld/st bug was in planner arithmetic, not in the
    remaining MMAv5 producer-side buckets;
  - the fix was to solve the direct planning query on the active physical TMEM
    layout when the analyzed view carries a zero row basis but the active
    layout already matches the logical shape; and
  - the broad `test-gluon` shard manifests in this folder still reflect the
    pre-fix census and must be regenerated after the next wider rerun before
    their counts are treated as current state.

4. Stale-negative / broadened-support bucket
- Nodeid confirmed clean on an isolated GPU:
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout[identity-64-32x32b-32x32b.x64.b32]`
- Current result:
  - no longer raises `CompilationError`
- Classification:
  - `STALE_NEGATIVE_OR_SUPPORT_BROADENED`
- Current interpretation:
  - this is the kind of expected churn discussed in the initiative: support has
    broadened, so an old clean-negative is no longer a valid correctness oracle
  - do not "fix" the compiler to restore the old rejection
  - re-evaluate the whole higher-rank half-row clean-error family one exact
    nodeid at a time before deciding which expectations should become positive

5. Branch-added TMEM runtime-matrix bucket
- Source:
  - `python/test/gluon/test_tmem_runtime_matrix.py`
- Current result from `test-gluon` shard `4 / 4`:
  - `686 failed, 4911 passed, 849 skipped, 19345 deselected`
- Classification:
  - `BRANCH_ADDED_COVERAGE_RED`
- Why this is branch-local:
  - `python/test/gluon/test_tmem_runtime_matrix.py` does not exist on the
    merge-base, so every failure in this file is new coverage introduced by
    this branch
- Current interpretation:
  - this is not one small stale-expectation pocket; the shard is dominated by
    real runtime-matrix breakage in the new TMEM coverage
  - the top failing families in the completed shard are:
    - `224` `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[...]`
    - `48` `test_tmem_runtime_matrix_ld_red_row_permuted_linear_layout[...]`
    - `33` `test_tmem_runtime_matrix_ldst_scales_variant_sweep[...]`
    - `32` `test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_geometry_sweep[...]`
    - `30` `test_tmem_runtime_matrix_cp_no_scales[...]`
  - there is still some stale-negative churn mixed in, but the dominant signal
    is wrong-code / unsupported-layout fallout in branch-added TMEM coverage

6. Recovered old-mainline MMAv5 root-load / regression bucket
- Fresh exact passes on the current dirty tree:
  - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
  - `python/test/unit/language/test_matmul.py::test_lhs_in_tmem[float32-False-64-128-32]`
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-1-64-512-32-2-float32-tensorfloat32]`
  - `python/test/regression/test_cast_matmul.py::test_cast_matmul[768-768-1024-16-64-16-bfloat16-float16-float16]`
- File-level rerun:
  - `python/test/regression/test_cast_matmul.py`
    - `1080 passed, 216 skipped`
- Classification:
  - `FIXED_ON_CURRENT_WORKTREE`
- Current interpretation:
  - the missing explicit MMAv5 root row-plan contract on source-initialized
    operand roots was a real branch bug; this fix repairs that slice without
    reopening the already-fixed split-N TMEM controls

7. Real current-head descriptor-view compile bucket
- Representative exact nodeids:
  - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-128-32]`
  - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-16-16]`
  - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-8-64-128-32]`
  - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float32]`
- Current result:
  - representative exact reruns still fail at compile time with
    `required row anchors 32,64 are not directly representable in the descriptor view`
- Classification:
  - `REAL_NEW_ON_BRANCH_REGRESSION`
- Current interpretation:
  - this is now the clean compile-time sibling bucket after the source-root
    row-plan fix; it is not explained by the older cast-matmul failures

8. Real current-head warp-specialization bucket
- Representative exact nodeids:
  - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
  - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_persistent_forward[False-8-True-2-128-64-1024-1024]`
- Current result:
  - forward still hits `CUDA error: misaligned address`
  - persistent-forward still shows `65535 / 131072` mismatches (`50.0%`)
- Classification:
  - `REAL_NEW_ON_BRANCH_REGRESSION`
- Current interpretation:
  - higher-rank MMAv5 root producers used by warp-specialization still look
    under-annotated or differently rewritten compared with the now-fixed
    source-root path

## Branch-vs-Main Classification Snapshot

The GB200 branch-vs-main classification is now strong enough to drive fixes:

- `PREEXISTING_ON_MERGE_BASE`
  - `python/test/unit/test_debug.py`
  - `third_party/proton/test/test_profile.py`
- `REINTERPRET_CONTRACT_TEST_TO_REWRITE`
  - `test_tmem_subslice_block_m_64[legacy]`
  - `test_tmem_subslice_block_m_64_parent_layout[linear]`
- `STALE_NEGATIVE_OR_SUPPORT_BROADENED`
  - the higher-rank half-row clean-negative family that no longer raises
- `REAL_NEW_ON_BRANCH_REGRESSION`
  - unit dot / matmul / warp-specialization / tensor-descriptor reshape
  - `python/test/regression/test_cast_matmul.py`
  - merge-base-present shard-3 Gluon subsets
- `BRANCH_ADDED_COVERAGE_RED`
  - `python/test/gluon/test_tmem_runtime_matrix.py`
  - branch-added shard-3 Gluon subsets
- `BRANCH_CHANGED_COVERAGE_RED`
  - `python/examples/gluon/`
  - branch-changed shard-3 Gluon subsets

Use `gb200_failure_manifest.md` for the exact `.txt` lists and
`gb200_branch_recovery_plan.md` for the prioritized fix order.

## Executed GB200 Census Status

### Green Or Healthy

- `make test-lit`
  - passed after reconfiguring the build cache to point `LLVM_EXTERNAL_LIT` at
    the installed `lit` script
  - `248 passed, 2 unsupported`
- `make test-cpp`
  - passed
  - `240 / 240` tests passed
- manual lit fallback
  - still works as a fallback:
    - `cd $BUILD_DIR && python3 -m lit.main -sv test`
- `make test-gsan`
  - passed
  - `20 passed`
- `make test-microbenchmark`
  - passed
  - completed both Tensor and TensorDescriptor launch-overhead runs
- `test-gluon` shard `1 / 4`
  - `5433 passed, 1014 skipped, 19344 deselected`
  - no failures observed
- `test-gluon` shard `2 / 4`
  - `2663 passed, 3784 skipped, 19344 deselected`
  - no new exact failures

### Local Environment Blockers

- no remaining local environment blocker is currently preventing the recorded
  GB200 inventory from running
- local env fixes applied during this census:
  - reconfigured the existing build with
    `cmake -S . -B $BUILD_DIR -DLLVM_EXTERNAL_LIT=$(which lit)` so
    `make test-lit` uses the installed `lit` script instead of `/llvm-lit`
  - installed `llnl-hatchet` into the current Python environment so Proton
    tests could run past the viewer import

### Red But Not Yet Fully Reduced

- `make test-regression`
  - red
  - `234 failed, 856 passed, 216 skipped`
  - concentrated in `python/test/regression/test_cast_matmul.py`
- `python/examples/gluon/`
  - red
  - `62 failed, 759 passed, 74 skipped`
  - failure concentration:
    - `02-convolution.py`
      - repeated `OutOfResources` on Blackwell shared-memory limit
      - required shared memory `262208`, hardware limit `232448`
    - `03-matmul-multicta.py`
      - repeated wrong-code at about `48%` mismatches in the failing cases
  - current interpretation:
    - the convolution failures look like a launch-configuration / autotuning /
      portability issue rather than TMEM reinterpret fallout
    - the multicta matmul failures likely belong near the broader GB200 matmul
      regression surface and should be triaged alongside the split-N / cast
      matmul buckets
- `test-gluon` shard `3 / 4`
  - red
  - `1160 failed, 3251 passed, 2036 skipped, 19344 deselected`
  - file split:
    - `793` in `python/test/gluon/test_lowerings.py`
    - `293` in `python/test/gluon/test_core.py`
    - `72` in `python/test/gluon/test_fpsan.py`
    - `1` in `python/test/gluon/test_frontend.py`
    - `1` in `python/test/gluon/test_layout_format_view.py`
  - dominant exact-family counts:
    - `603` `test_reduce_layouts[...]`
    - `176` `test_scan_layouts[...]`
    - `64` `test_mma_scaled_tcgen05_copy[...]`
    - `40` `test_tmem_reduction[...]`
    - `32` `test_tmem_reduction_linear_layouts[...]`
    - `27` `test_padded_shared_layout_subslice[...]`
  - interpretation:
    - this shard is a broad Gluon lowering/backend regression surface, not
      just residual TMEM-runtime contamination
- `test-gluon` shard `4 / 4`
  - red
  - `686 failed, 4911 passed, 849 skipped, 19345 deselected`
  - every failure in the completed shard is in the branch-added file
    `python/test/gluon/test_tmem_runtime_matrix.py`
  - top exact-family counts:
    - `224` `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[...]`
    - `48` `test_tmem_runtime_matrix_ld_red_row_permuted_linear_layout[...]`
    - `33` `test_tmem_runtime_matrix_ldst_scales_variant_sweep[...]`
    - `32` `test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_geometry_sweep[...]`
    - `30` `test_tmem_runtime_matrix_cp_no_scales[...]`
  - stale-negative fallout still exists inside the shard, but the finished
    result is much broader than the original half-row clean-error family
- `make test-proton`
  - red after the `llnl-hatchet` install removed the old import blocker
  - `11 failed, 31 passed, 2 skipped` on the exact-file rerun
  - all observed failures are in `third_party/proton/test/test_profile.py`
  - current interpretation:
    - local environment is no longer the blocker, but this is also not a new
      branch-caused bucket
    - the same `11` nodeids fail on merge-base after fixing the local-only
      baseline `libproton.so` symlink
    - keep it in the GB200 census, but do not keep it in the TMEM branch
      recovery backlog

## Cache / Flake Status

- Current evidence still does not support a simple on-disk cache-collision
  theory.
- Reconfirmed probe:
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[identity-identity-2-32x32b_splitn]`
  - followed by
    `python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
- Results:
  - same pytest process: bad test fails, `test_block_m_64_mma[linear]` still
    passes
  - separate fresh processes with the same `TRITON_CACHE_DIR`: same outcome
- Working interpretation:
  - shard-only anomalies are more likely process/device contamination after bad
    kernels, or another missing runtime invalidation/input, than a trivial
    cache-key collision
- Additional harness anomaly to remember before redesigning cache keys:
  - on this devbox, a simple standalone process respects
    `CUDA_VISIBLE_DEVICES=3` and lands on physical GPU `3`;
  - but some pytest/xdist workers in the broader sweeps either had no
    `CUDA_VISIBLE_DEVICES` in `/proc/<pid>/environ`, or reported
    `CUDA_VISIBLE_DEVICES=3` while `nvidia-smi` attributed them to physical
    GPU `0`;
  - so there is a real worker/device-isolation discrepancy in the harness, and
    it must be ruled out before blaming on-disk cache reuse.
- Actionable rule:
  - if a future anomaly looks cache-sensitive, preserve the exact order and
    replay it across fresh processes before designing around the cache

## Cross-Bucket PTX / TTGIR Clue

The current branch-vs-merge-base PTX comparison is now strong enough to guide
the recovery plan, even before the last broad sweeps finish:

- Representative wrong-code kernel:
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-1-64-512-32-2-float32-tensorfloat32]`
- Forced `16x256` variant:
  - current branch still chooses the `16x256b` family, but the final
    `tcgen05.ld.sync.aligned.16x256b.x16.b32` packet offsets are contiguous:
    `+0`, `+128`, `+256`, `+384`
  - merge-base uses the same family but places the second support band at the
    large lifted offsets:
    `+0`, `+128`, `+1048576`, `+1048704`
- Non-forced sibling:
  - current branch chooses `16x32bx2.x64`
  - merge-base chooses `32x32b.x64`
- Current interpretation:
  - the new row-plan-aware canonical-`M=64` family selection / packet
    decomposition is collapsing a support-band / lifted-image bit into ordinary
    packet repetition
  - this is a better fit for the observed branch-wide wrong-code than any
    theory based on isolated cache poisoning or unrelated LLVM noise

### `test-unit` Sweep Update (2026-04-10)

To finish the GB200 census, `make test-unit` was expanded into its component
commands and run in a more diagnosable shape:

- broad unit lane:
  - `CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-unit-gpu<gpu> PYTHONPATH=python:. python3 -m pytest -s --tb=short --splits 4 --group <group> -n 2 python/test/unit --ignore-glob='python/test/unit/plugins/*' --ignore=python/test/unit/test_debug.py`
- tail commands:
  - `python/test/unit/test_debug.py`
  - `python/triton_kernels/tests/`
  - `python/tutorials/06-fused-attention.py`
  - `python/test/unit/instrumentation/test_gpuhello.py`
  - plugin tests under `python/test/unit/plugins/`

Current results:

- direct CI-target mirror:
  - `make NUM_PROCS=24 test-unit`
  - result:
    - `162 failed, 14991 passed, 5492 skipped`
  - exact-family counts from the completed xdist log:
    - `64` `test_warp_specialize_attention_forward[...]`
    - `64` `test_warp_specialize_attention_persistent_forward[...]`
    - `18` `test_simple_matmul[...]`
    - `8` `test_dot[...]`
    - `4` `test_simple_persistent_matmul[...]`
    - `3` `test_tensor_descriptor_reshape_matmul[...]`
    - `1` `test_lhs_in_tmem[...]`
  - this confirms the earlier shard-based read, with one important refinement:
    - the warp-specialization bucket is larger than first recorded because the
      persistent-forward half is red too, not just the forward half

- unit shard `1 / 4`
  - green
  - `5048 passed, 114 skipped`
- unit shard `2 / 4`
  - red
  - `8 failed, 4507 passed, 647 skipped`
  - exact failing nodeids:
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-trans-tf32x3-float32-float32-1-None]`
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-add-matrix-tf32x3-float32-float32-1-None]`
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-add-rows-tf32x3-float32-float32-1-None]`
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-add-cols-tf32x3-float32-float32-1-None]`
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-softmax-tf32x3-float32-float32-1-None]`
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-chain-dot-tf32-float32-float32-1-None]`
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-chain-dot-tf32x3-float32-float32-1-None]`
  - isolated rerun confirms at least the leading nodeid is a real standalone
    wrong-code failure (`49.4%` mismatches), not shard contamination
- unit shard `3 / 4`
  - red
  - `23 failed, 729 passed, 4410 skipped`
  - dedicated full-file inventory is still running, but the exact partial list
    already covers:
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[...]`
      - `18` exact nodeids so far
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[...]`
      - `4` exact nodeids so far
  - isolated reruns already confirm both sub-buckets:
    - simple matmul wrong-code:
      - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-1-64-512-32-2-float32-tensorfloat32]`
      - `49.9%` mismatches
    - persistent matmul compile/pipeline failure:
      - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-8-64-128-32]`
      - `ttng.tmem_store` unsupported register layout / row anchors `32,64`
        followed by `PassManager::run failed`
  - current pattern:
    - the exact failures cluster in `BLOCK_M=64` cases
    - the branch-side `test_matmul.py` edits only relaxed PTX-shape
      expectations and added a TTGIR row-plan assertion, so these wrong-code
      and compile failures are not explained by the test edits themselves
- unit shard `4 / 4`
  - red
  - `131 failed, 4707 passed, 321 skipped`
  - now reduced further:
    - `python/test/unit/language/test_tensor_descriptor.py`
      - fully materialized
      - exact red list:
        - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float16]`
        - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[bfloat16]`
        - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float32]`
      - result:
        - `3 failed, 2601 passed, 110 skipped`
      - failure mode:
        - `ttng.tmem_store` unsupported register layout / row anchors `32,64`
          followed by `PassManager::run failed`
      - merge-base classification:
        - the full file passes on merge-base
        - `2604 passed, 110 skipped`
        - therefore this is a real new-on-branch compile bucket
    - `python/test/unit/language/test_warp_specialization.py`
      - the completed direct `make test-unit` log refines the earlier shard
        read
      - current exact family split is:
        - `64` `test_warp_specialize_attention_forward[...]`
        - `64` `test_warp_specialize_attention_persistent_forward[...]`
      - representative standalone failure:
        - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
        - `49.7%` mismatches
      - current pattern:
        - the failures cluster in attention cases with `N=64`
        and again line up with the broader `M=64` / Blackwell TMEM
        regression surface rather than looking like random shard noise

Tail-command status:

- `python/test/unit/test_debug.py`
  - red
  - xdist-shaped run: `20 failed, 75 passed`
  - serial rerun: same `20 failed, 75 passed`
  - classification:
    - real harness/runtime problem, not just outer xdist noise
    - failures all show `Cannot re-initialize CUDA in forked subprocess`
      originating from `py._process.forkedfunc`
  - exact failure families:
    - `python/test/unit/test_debug.py::test_sanitize_int_add_overflow[...]`
      - `9` failing nodeids
    - `python/test/unit/test_debug.py::test_sanitize_int_mul_overflow[...]`
      - `6` failing nodeids
    - `python/test/unit/test_debug.py::test_sanitize_int_sub_overflow[...]`
      - `5` failing nodeids
- `python/tutorials/06-fused-attention.py`
  - green
  - `192 passed, 192 skipped`
- `python/test/unit/instrumentation/test_gpuhello.py`
  - green
  - `1 passed`
- plugin tests
  - green
  - `python/test/unit/plugins/test_plugin.py`
  - `python/test/unit/plugins/test_dialect_plugin.py`
  - `python/test/unit/plugins/custom_ops.py`
- `python/triton_kernels/tests/`
  - green
  - `2377 passed, 3444 skipped`

Interim interpretation of the unit lane:

- The new red surface is broader than the TMEM reinterpret rewrite candidates.
- Confirmed standalone buckets now include:
  - tf32/tf32x3 dot wrong-code
  - matmul wrong-code
  - warp-specialization attention wrong-code
  - tensor-descriptor reshape-to-matmul compile/pipeline failure
  - persistent matmul compile/pipeline failure
  - debug harness CUDA-after-fork failure
- Branch-surface check:
  - among GB200-relevant Python tests, this branch modified:
    - `python/test/gluon/test_core.py`
    - `python/test/gluon/test_tmem_runtime_matrix.py`
    - `python/test/unit/language/test_compile_only.py`
    - `python/test/unit/language/test_matmul.py`
  - it did **not** modify:
    - `python/test/unit/language/test_core.py`
    - `python/test/unit/language/test_tensor_descriptor.py`
    - `python/test/unit/language/test_warp_specialization.py`
    - `python/test/unit/test_debug.py`
    - `python/test/regression/test_cast_matmul.py`
    - `python/examples/gluon/*`
    - `third_party/proton/test/test_profile.py`
  - this means those buckets are not explained by test-file edits and must be
    classified later as pre-existing-on-main, real branch regressions, or
    flake/harness/cache issues.

### Merge-Base Comparison Status (2026-04-10)

- `origin/main` is now fetched locally.
- Merge-base for current branch vs main:
  - `7f61ac734edc657b737fb159a1b9d50cb47944e6`
- A detached worktree exists at:
  - `/root/code/triton-mergebase-ci`
- The baseline worktree is usable after local-only bring-up shims:
  - skip example plugin builds in the merge-base tree; and
  - skip the legacy GSan runtime target that does not build cleanly against the
    current host toolchain.
- The branch-vs-main classification is now mostly complete for the finished
  GB200 red surface:
  - fully green on merge-base, so the current-branch failures are entirely new:
    - `python/test/unit/language/test_matmul.py`
      - `761 passed, 4780 skipped`
    - `python/test/unit/language/test_warp_specialization.py`
      - `1599 passed, 202 skipped`
    - `python/test/unit/language/test_tensor_descriptor.py`
      - `2604 passed, 110 skipped`
    - `python/test/regression/test_cast_matmul.py`
      - `1080 passed, 216 skipped`
    - `python/test/gluon/test_lowerings.py`
      - `4937 passed, 512 skipped`
  - exact current-branch subsets that are green on merge-base:
    - the `162` non-debug nodeids from the current-branch unit failure
      manifest
    - the `8` failing `test_dot[...]` nodeids from the current-branch unit
      manifest
    - the `185` merge-base-present shard-3 `test_core.py` nodeids
    - the `38` merge-base-present shard-3 `test_fpsan.py` nodeids
    - the `1` merge-base-present `test_layout_format_view.py` nodeid
    - the `793` merge-base-present `test_lowerings.py` shard-3 nodeids
  - preexisting on merge-base:
    - `python/test/unit/test_debug.py`
      - same `20` nodeids fail on merge-base
    - `third_party/proton/test/test_profile.py`
      - same `11` nodeids fail on merge-base
  - branch-added or branch-changed coverage rather than old mainline exact
    nodeids:
    - the entire branch-added `python/test/gluon/test_tmem_runtime_matrix.py`
      file
    - the branch-added TMEM-core/fpsan/frontend buckets recorded in the
      failure-manifest split files
    - the `18` shard-3 `test_core.py` nodeids whose function exists on
      merge-base but whose exact parametrized nodeid does not
    - the `22` shard-3 `test_fpsan.py` nodeids in the same category
    - the `62` failing `python/examples/gluon/` nodeids, whose exact
      parametrizations do not exist on merge-base even though the merge-base
      full-file rerun is green
- This means the remaining work is no longer “figure out what is new.” It is:
  - fix the clearly new compiler/backend regressions; and
  - separately decide how to treat branch-added or branch-changed coverage that
    is red because the branch expanded the tested surface.

### Follow-Up Exact Classification (2026-04-10)

- `python/test/unit/language/test_core.py::test_dot[...]`
  - exact current-branch failing set:
    - `8` nodeids
  - merge-base exact rerun of those `8` nodeids:
    - `8 / 8` pass
  - current classification:
    - `REAL_NEW_ON_BRANCH_REGRESSION`

- `python/test/unit/language/test_matmul.py`
  - full exact rerun is now complete:
    - `23 failed, 738 passed, 4780 skipped`
  - exact family split:
    - `18` `test_simple_matmul[...]`
    - `4` `test_simple_persistent_matmul[...]`
    - `1` `test_lhs_in_tmem[float32-False-64-128-32]`
  - merge-base exact rerun:
    - all `23` current-branch failing nodeids are present inside the
      merge-base unit manifest reduction
    - the full `182`-nodeid merge-base unit rerun reduces to only the
      preexisting `20` `test_debug.py` failures, so the `23 / 23` `test_matmul`
      nodeids pass there
  - current classification:
    - `REAL_NEW_ON_BRANCH_REGRESSION`

- `python/test/unit/language/test_warp_specialization.py`
  - current-branch CI-target mirror exposes:
    - `64` `test_warp_specialize_attention_forward[...]`
    - `64` `test_warp_specialize_attention_persistent_forward[...]`
  - merge-base exact rerun:
    - the full file is green
    - `1599 passed, 202 skipped`
    - the `128` current-branch failing nodeids are also covered by the
      `182`-nodeid merge-base unit rerun and all pass there
  - current classification:
    - `REAL_NEW_ON_BRANCH_REGRESSION`

- `python/test/unit/test_debug.py`
  - current-branch exact rerun:
    - `20 failed, 75 passed`
  - merge-base exact rerun:
    - the same `20` nodeids fail
  - current classification:
    - `PREEXISTING_ON_MERGE_BASE`

- `python/test/regression/test_cast_matmul.py`
  - full exact rerun is complete:
    - `234` exact failing nodeids
  - failure pattern:
    - failures span the `GN=64` family across all observed `GM in {16,32,64}`
      and `GK in {16,64,128}` combinations
  - merge-base exact rerun:
    - the whole file is green on merge-base
    - `1080 passed, 216 skipped`
  - current classification:
    - `REAL_NEW_ON_BRANCH_REGRESSION`

- `python/examples/gluon/`
  - full exact rerun is complete:
    - `62` exact failing nodeids
  - exact family split:
    - `48` failures in `python/examples/gluon/02-convolution.py`
      - repeated `OutOfResources` on shared memory
    - `14` failures in `python/examples/gluon/03-matmul-multicta.py`
      - wrong-code around `35%` to `48%` mismatches
  - merge-base collect-only reduction:
    - exact current-branch example nodeids present on merge-base: `0`
    - exact current-branch example nodeids missing on merge-base: `62`
  - merge-base full-file rerun:
    - `130 passed, 14 skipped`
  - current classification:
    - `BRANCH_CHANGED_COVERAGE_RED`

- `python/test/gluon/test_lowerings.py`
  - shard-3 current-branch failures:
    - `793` exact nodeids
    - dominated by `603` `test_reduce_layouts[...]` and `176`
      `test_scan_layouts[...]`
  - merge-base collect-only reduction:
    - exact current-branch shard-3 `test_lowerings.py` nodeids present on
      merge-base: `793`
    - exact current-branch shard-3 `test_lowerings.py` nodeids missing on
      merge-base: `0`
  - merge-base exact rerun:
    - the full file is green
    - `4937 passed, 512 skipped`
  - current classification:
    - `REAL_NEW_ON_BRANCH_REGRESSION`

- merge-base-existing shard-3 exact subsets
  - `python/test/gluon/test_core.py`
    - `203` nodeids had matching function names on merge-base
    - collect-only exact-nodeid split:
      - `185` exact nodeids present on merge-base
      - `18` exact nodeids absent on merge-base despite the function existing
    - merge-base exact rerun of the `185` present nodeids:
      - `185 / 185` pass
    - current classification:
      - `185` `REAL_NEW_ON_BRANCH_REGRESSION`
      - `18` `BRANCH_CHANGED_PARAMETRIZATION_SURFACE`
  - `python/test/gluon/test_fpsan.py`
    - `60` nodeids had matching function names on merge-base
    - collect-only exact-nodeid split:
      - `38` exact nodeids present on merge-base
      - `22` exact nodeids absent on merge-base despite the function existing
    - merge-base exact rerun of the `38` present nodeids:
      - `38 / 38` pass
    - current classification:
      - `38` `REAL_NEW_ON_BRANCH_REGRESSION`
      - `22` `BRANCH_CHANGED_PARAMETRIZATION_SURFACE`
  - `python/test/gluon/test_layout_format_view.py`
    - `1` current-branch failing nodeid exists on merge-base
    - merge-base exact rerun:
      - `1 / 1` pass
    - current classification:
      - `REAL_NEW_ON_BRANCH_REGRESSION`
  - `python/test/gluon/test_frontend.py::test_tensor_memory_linear_layout_non_surjective_reg_layout_parses`
    - exact nodeid missing on merge-base
    - current classification:
      - `BRANCH_ADDED_COVERAGE`

- `third_party/proton/test/test_profile.py`
  - current-branch exact rerun:
    - `11 failed, 31 passed, 2 skipped`
  - merge-base exact rerun:
    - the same `11` nodeids fail on merge-base
  - current classification:
    - `PREEXISTING_ON_MERGE_BASE`
  - implication:
    - Proton should be removed from the branch-caused TMEM recovery backlog

- shared-cache sanity probe
  - probe shape:
    - first run the known-bad split-N TMEM descriptor-chain test
    - then run the known-good `test_block_m_64_mma[linear]`
    - keep `CUDA_VISIBLE_DEVICES=0` and `TRITON_CACHE_DIR` identical but use a
      fresh Python process for each run
  - result:
    - the split-N test still fails (`4096 / 8192`)
    - `test_block_m_64_mma[linear]` still passes immediately afterward
  - current interpretation:
    - this does not reproduce simple on-disk JIT cache poisoning
    - the earlier shard-only anomalies are more likely process / device
      contamination after a bad kernel than a trivial cache-key collision
    - keep the cache-collision hypothesis open only for future repros that
      actually survive a fresh process boundary

### Noisy But Expected During `-s` Shard Runs

- `python/test/gluon/test_core.py::test_tcgen05_mma_plain_kind_i8_reports_clean_error`
  - clean isolated rerun passes
  - the PTXAS rejection
    `Feature '.kind::i8' not supported on .target 'sm_103a'`
    is the expected behavior for this negative test
  - the shard log is noisy because `-s` prints the PTXAS reproducer even though
    the test itself passes

## Recently Cleared During This Checkpoint

- `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_block_descriptor_reports_clean_error[block_two_ctas-layout1-reinterpret_layout1-expected_fragments1]`
  - classification: `STALE_EXPECTATION`
  - action: updated the expected fragments to match the current row-anchor
    materialization diagnostic
  - validation: exact-nodeid rerun now passes

## Current Known Green Controls

The most relevant previously validated positive controls for this phase are:

- `python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
- representative `python/test/gluon/test_core.py::test_mma_shared_inputs[...]`
  M64 split-N roots
- descriptor-chain M64 direct-load cases
- split-N row/col-permuted TMEM runtime slices

These stay in the nearest validation loop while the red list is being reduced.

## Triage Policy

- Keep this file current whenever the confirmed GB200 red list changes.
- Fix `STALE_EXPECTATION` failures first so the remaining red list reflects
  actual behavior changes.
- For reinterpret-heavy TMEM tests, first restate the user intent under the new
  explicit descriptor/view API contract.
- When a big sharded run fails after the first device/runtime error, rerun the
  earliest exact nodeids on a clean isolated GPU before treating the whole
  shard summary as multiple independent bugs.
- If the explicit rewrite itself is blocked by missing TMEM view/index or
  direct-lowering surface coverage, record that as initiative debt and move the
  work back to the core compiler/API backlog instead of forcing the old
  reinterpret behavior to remain the contract.

## 2026-04-10 Overnight Sweep Checkpoint (Partial)

- Current branch / commit:
  - `e70a3aa09`
- Newly frozen `test-gluon` exact manifests:
  - shard `3 / 4`:
    - `1252` exact nodeids
    - exact merge-base split:
      - `967` old-mainline nodeids
      - `285` branch-added / branch-changed nodeids
  - shard `4 / 4`:
    - `686` exact nodeids
    - identical nodeid set to the older group-4 manifest
- Stable current-head shard `4 / 4` result:
  - `686 failed, 4911 passed, 849 skipped, 19345 deselected`
  - all failures are in branch-added
    `python/test/gluon/test_tmem_runtime_matrix.py`
  - dominant families still:
    - split-N row/col-permuted
    - `ld.red` row-permuted linear-layout
    - `ldst_scales`
    - scaled-MMA copy `warpx4`
    - `cp_no_scales`
- Current-head shard `3 / 4` status:
  - the merge-base split is now durable, but the raw shard count still
    overstates the number of independent failures
  - earlier standalone reruns already showed bad-kernel fallout after the
    first primary TMEM failures
- Still running when this checkpoint was written:
  - current-head shard `1 / 4`
    - no failure markers emitted yet
  - current-head `python/triton_kernels/tests`
    - no failure markers emitted yet
    - caveat: the run currently shares GPU `3` with an older non-CI
      `python/test/unit` shard, so any red result must be rerun in isolation
  - fresh merge-base full-file reruns:
    - `python/test/unit/language/test_matmul.py`
    - `python/test/unit/language/test_warp_specialization.py`
    - both clean so far
- Queued next reductions after those GPUs free:
  - isolated current-head `python/test/gluon/test_lowerings.py`
  - isolated current-head `python/test/gluon/test_fpsan.py`
  - isolated current-head focused `python/test/gluon/test_core.py` slice for
    the dominant shard-3 TMEM/MMA families

## 2026-04-10 Overnight Exact Reductions (Further)

- Fresh merge-base file-level confirmation is now complete for the live unit
  bucket:
  - `python/test/unit/language/test_matmul.py`
    - `761 passed, 4780 skipped`
  - `python/test/unit/language/test_warp_specialization.py`
    - `1599 passed, 202 skipped`
  - implication:
    - the current `134`-nodeid unit red surface is now backed by fresh
      file-level merge-base proof, not just earlier spot checks
- Current-head shard `1 / 4` is now green:
  - `5433 passed, 1014 skipped, 19344 deselected`
- Isolated shard-3 reductions now separate the real exact core from the shard
  noise:
  - focused `python/test/gluon/test_core.py` TMEM/MMA slice:
    - `332 failed, 12 passed, 17619 deselected`
    - exact split:
      - `146` old-mainline exact nodeids
      - `186` branch-added / branch-changed nodeids
    - dominant independent families:
      - `64` `test_mma_scaled_tcgen05_copy[...]`
      - `60` `test_tmem_linear_roundtrip_variant_sweep[...]`
      - `40` `test_tmem_reduction[...]`
      - `32` `test_tmem_reduction_linear_layouts[...]`
      - `26` `test_tmem_descriptor_chain_matrix[...]`
  - isolated `python/test/gluon/test_fpsan.py`:
    - `1 failed, 66 passed, 19 skipped`
    - lone exact failure:
      - `python/test/gluon/test_fpsan.py::test_tcgen05_mma_scaled[linear_identity-acc_layout1-e4m3-e2m1]`
    - exact nodeid absent on merge-base, so this is branch-changed coverage
  - isolated `python/test/gluon/test_layout_format_view.py` +
    frontend tail:
    - `python/test/gluon/test_layout_format_view.py` is green in isolation
    - the remaining isolated tail is the single branch-added frontend exact:
      - `python/test/gluon/test_frontend.py::test_tensor_memory_linear_layout_non_surjective_reg_layout_parses`
- Focused branch-added runtime-matrix reduction is also independently red:
  - focused `python/test/gluon/test_tmem_runtime_matrix.py` dominant-family
    slice:
    - `413 failed, 17 passed, 1360 deselected`
  - dominant independent families:
    - `208` split-N row/col-permuted
    - `48` `ld.red` row-permuted linear-layout
    - `33` `ldst_scales`
    - `32` scaled-MMA copy `warpx4` geometry
    - `30` `cp_no_scales`
- Still running when this section was written:
  - current-head `python/triton_kernels/tests`
    - still clean so far, but still sharing GPU `3` with an older non-CI
      `python/test/unit` shard
  - isolated current-head `python/test/gluon/test_lowerings.py`
    - queued behind the now-finished shard `1 / 4`; waiting for the queue
      shell to hand off

## 2026-04-10 Inventory Closure

- The last pending isolated completions are now done:
  - current-head `python/triton_kernels/tests`
    - `2377 passed, 3444 skipped`
  - isolated current-head `python/test/gluon/test_lowerings.py`
    - `4937 passed, 512 skipped`
- Final GB200 lane status for this pass:
  - `make test-lit`
    - red
    - current branch:
      - `247 passed, 1 failed, 2 unsupported`
    - lone exact:
      - `test/TritonGPU/pipeline-lower-loop.mlir`
    - merge-base:
      - `248 passed, 2 unsupported`
  - `make test-cpp`
    - green
    - `240 / 240 passed`
  - `make NUM_PROCS=24 test-unit`
    - red
    - `134 failed, 15019 passed, 5492 skipped`
    - exact current-head red surface:
      - `3` `test_matmul.py`
      - `3` `test_tensor_descriptor.py`
      - `128` `test_warp_specialization.py`
  - `make NUM_PROCS=24 test-gluon`
    - red
    - independent current-head buckets now reduced to:
      - focused `python/test/gluon/test_core.py`
        - `332` exact nodeids
        - `146` merge-base-present
        - `186` branch-added / branch-changed
      - focused `python/test/gluon/test_tmem_runtime_matrix.py`
        - `413` exact branch-added nodeids
      - isolated `python/test/gluon/test_fpsan.py`
        - `1` exact branch-changed nodeid
      - isolated `python/test/gluon/test_frontend.py`
        - `1` exact branch-added nodeid
      - `python/examples/gluon/02-convolution.py`
        - `48` exact branch-changed nodeids
        - representative failure:
          `OutOfResources` (`262208` shared-memory bytes requested vs
          `232448` hardware limit)
        - merge-base full-file rerun is green (`48 passed`)
      - `python/examples/gluon/03-matmul-multicta.py`
        - `14` exact branch-changed nodeids
        - representative failure:
          `9600 / 20000` mismatches (`48.0%`)
        - merge-base full-file rerun is green (`82 passed, 14 skipped`)
    - isolated current-head green proofs:
      - `python/test/gluon/test_layout_format_view.py`
      - `python/test/gluon/test_lowerings.py`
    - implication:
      - the raw shard-3 lowerings/layout failures were fallout
  - `make NUM_PROCS=24 test-gsan`
    - green
    - `20 / 20 passed`
  - `make test-regression`
    - green
    - `1090 passed, 216 skipped`
  - `make test-microbenchmark`
    - green
  - `make test-proton`
    - red but preexisting
    - `11 failed, 114 passed, 1 skipped`
    - same exact nodeids are red on merge-base
- The exact independent red list now drives the fix order directly:
  1. `test/TritonGPU/pipeline-lower-loop.mlir`
  2. the `134`-nodeid `test-unit` bucket
  3. the merge-base-present focused `test_core.py` bucket
  4. the branch-added / branch-changed Gluon tails
  5. the examples lane
- Harness caveat after the closure:
  - the census no longer needs the older raw shard totals to choose fix order
  - keep the cache/xdist investigation separate:
    - current evidence still favors worker/process/device contamination over a
      proven on-disk cache-key collision

## 2026-04-10 post-closure recovery checkpoint: pure outer memdesc_index row-plan selection shrinks the active lit/unit surface

- This section is a current dirty-worktree recovery note layered on top of the
  closed GB200 census above. The closure numbers remain the durable baseline
  for `9150a3d5e`; the notes below describe the next local recovery slice.
- Code change under validation:
  - `getTMemLdStRowPlanForQuery(...)` now prefers the query row plan for pure
    outer `ttg.memdesc_index` TMEM views that only peel non-layout prefix
    dimensions and preserve the trailing 2D TMEM tile.
- Current validation on that dirty worktree:
  - `make -j8`
    - green
  - `make test-lit`
    - green
    - `248 passed, 2 unsupported`
  - exact focused unit repros:
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-128-32]`
      - passes
    - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float32]`
      - passes
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-False-2-64-64-1024-1024]`
      - still fails with `Triton Error [CUDA]: misaligned address`
  - direct lit repro:
    - `test/TritonGPU/pipeline-lower-loop.mlir`
      - passes
- Interrupted broad `make NUM_PROCS=24 test-unit` rerun:
  - no `test_matmul.py` or `test_tensor_descriptor.py` failures were observed
    before the run entered `test_warp_specialization.py`;
  - the first real remaining exact is:
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-False-2-64-64-1024-1024]`
  - the interrupted log currently contains:
    - `10` true kernel failures (`assert_close` / launch-site exacts); and
    - `109` later `torch.manual_seed` contamination failures on poisoned xdist
      workers.
- Harness symptom discovered during that rerun:
  - after the warp failures, the last xdist worker stalled in
    `python/test/unit/runtime/test_cache.py::test_async_compile_mock`;
  - `py-spy` shows the worker blocked in
    `triton.runtime._async_compile.AsyncCompileMode.__exit__`, waiting on
    unfinished futures; and
  - this should be tracked under the separate cache/async-compile harness
    investigation, not as a TMEM correctness regression by itself.
- Practical implication for the GB200 inventory:
  - the old baseline closure remains valid for the last committed `HEAD`; but
  - the active recovery surface after this local row-plan slice is now:
    1. higher-rank MMAv5-root `test_warp_specialization.py`
    2. merge-base-present focused `test_core.py`
    3. branch-added runtime-matrix and other Gluon tails

## 2026-04-10 post-recovery checkpoint: warp-specialization row-plan forwarding likely collapses the remaining old-mainline unit bucket

- This section updates the active recovery story above without replacing the
  last committed GB200 closure numbers until a fresh broad rerun is finished.
- New structural fix under validation:
  - explicit TMEM row-plan recovery now follows
    `gpu::WarpSpecializePartitionsOp` block arguments back through
    `getExplicitCaptures()`
  - this preserves the original `ttng.tmem_ldst_row_plan` contract inside the
    partition regions used by `python/test/unit/language/test_warp_specialization.py`
- Compile-only evidence:
  - TMEM debug query traces for the first representative exact now show only:
    - `rawRowPlan=128`
    - `rawQuery -> ok atom=4`
  - the earlier `rawRowPlan=64` / `atom=0` split on captured
    `ttg.memdesc_index<64x64xf32>` views is gone
  - PTX for the same kernel is back on packed
    `tcgen05.ld/st.sync.aligned.16x32bx2.x32.b32` at source line `491`
- Focused runtime evidence:
  - these previously failing representative exacts now pass:
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-False-2-64-64-1024-1024]`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-True-3-128-64-1024-1024]`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-8-False-3-128-64-8192-8192]`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-True-2-64-64-8192-8192]`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-2-128-64-8192-8192]`
- Inventory consequence:
  - the old `128`-nodeid `test_warp_specialization.py` manifest is now likely
    stale
  - do not keep treating that exact count as current-head truth once the fresh
    file rerun lands
- Next measurement required before changing the top-line GB200 counts:
  1. rerun `python/test/unit/language/test_warp_specialization.py`
  2. rerun `make NUM_PROCS=24 test-unit`
  3. refresh the unit exact manifests from the new result

## 2026-04-10 unit-lane closure: `make test-unit` is green on the current branch

- This supersedes the earlier unit-red status in the closed census above.
- Fresh current-head remeasurement:
  - `python/test/unit/language/test_warp_specialization.py`
    - rerun as a 4-GPU `pytest --splits 4 --group {1..4}` sweep
    - result:
      - `1599 passed, 202 skipped`
      - `0 failed`
  - `make NUM_PROCS=24 test-unit`
    - artifact:
      - `/tmp/test-unit-gb200-after-ws.log`
    - green sub-lanes:
      - main `python/test/unit`
        - `15153 passed, 5492 skipped, 101 warnings`
      - `python/test/unit/test_debug.py`
        - `95 passed`
      - `python/triton_kernels/tests`
        - `2377 passed, 3444 skipped`
      - `python/tutorials/06-fused-attention.py`
        - `192 passed, 192 skipped, 1 warning`
      - `python/test/unit/instrumentation/test_gpuhello.py`
        - `1 passed`
      - `python/test/unit/plugins/test_plugin.py`
        - `1 passed`
      - `python/test/unit/plugins/test_dialect_plugin.py`
        - `1 passed`
      - `python/test/unit/plugins/custom_ops.py`
        - `1 passed`
- Inventory consequences:
  - the current-branch GB200 unit lane is no longer red
  - current unit exact manifests are now empty:
    - `gb200_current_branch_test_unit_failures.txt`
    - `gb200_current_branch_test_unit_matmul_refresh_failures.txt`
    - `gb200_current_branch_test_unit_tensor_descriptor_refresh_failures.txt`
    - `gb200_current_branch_test_unit_warp_specialization_refresh_failures.txt`
    - `gb200_current_branch_test_unit_rowanchor_refresh_failures.txt`
    - `gb200_branch_recovery_test_unit_failures.txt`
  - the older populated unit manifests remain historical only
- Separate harness conclusion:
  - the earlier async-compile / `test_cache.py` hang did not reproduce in the
    fresh green rerun
  - the current evidence still points more strongly at failure-induced worker
    or device contamination than at an independent always-on cache-key issue
- Updated remaining GB200 red lanes after this closure:
  - `make test-gluon`
    - focused `python/test/gluon/test_core.py`
      - `146` merge-base-present exacts still prioritized
      - `186` branch-added / branch-changed focused core exacts still pending
    - focused `python/test/gluon/test_tmem_runtime_matrix.py`
      - `413` branch-added exacts
    - isolated `test_fpsan.py`
      - `1` branch-changed exact
    - isolated `test_frontend.py`
      - `1` branch-added exact
    - examples:
      - `48` `02-convolution.py`
      - `14` `03-matmul-multicta.py`
  - `make test-proton`
    - still preexisting on merge-base

## 2026-04-10 `02-convolution.py` closure: the shared-memory OOR was a linear-convert scratch over-allocation

- The merge-base-present convolution example regression is now closed on the
  current branch.
- Current-head remeasurement:
  - representative exact:
    - `python/examples/gluon/02-convolution.py::test_op[0-1-3-3-384-384-64-64-1]`
    - `PASSED`
  - full file:
    - `python/examples/gluon/02-convolution.py`
    - `48 passed in 7.96s`
  - representative compile artifact:
    - shared memory dropped from `262208` to `147520`
    - merge-base reference was `147516`
    - the remaining `+4` bytes are the deliberate
      `kTensorMemoryAllocSharedBytes` padding used to materialize the
      `tcgen05.alloc` base pointer
- Root cause:
  - the final OOR was not another TTGIR / TMEM planner mismatch
  - branch code in
    `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp`
    had added a pessimistic early return that treated any
    `ttg.convert_layout` touching `LinearEncodingAttr` as needing a full-tile
    scratch buffer
  - the convolution epilogue’s
    `tensor<256x256xf16, #linear> -> tensor<256x256xf16, #blocked>`
    convert then grabbed an unnecessary `128 KB` scratch allocation and pushed
    the kernel over the GB200/GB300 shared-memory limit
- Related structural cleanups that stayed in the final fix slice:
  - preserve exact TMEM encoding across outer-dimension `memdesc_index` views
    when the leaf descriptor type is still valid
  - restore barrier-driven `tcgen05.mma` / `tcgen05.mma_scaled` async
    selection in `python/src/gluon_ir.cc` instead of forcing async without
    barriers
- Inventory consequence:
  - `gb200_current_branch_examples_convolution_failures.txt` should now be
    considered refreshed empty
  - the aggregate `gb200_current_branch_examples_gluon_failures.txt` file is
    stale until the whole examples directory is rerun on top of the fix
  - the next confirmed old-mainline examples bucket is now
    `python/examples/gluon/03-matmul-multicta.py`
