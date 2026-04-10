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

## Latest Current-Branch Reduction (2026-04-10)

- The broad GB200 unit lane is now better understood than the earlier
  `162`-failure reduced manifest suggested:
  - the direct CI-style run still finished red at
    `2058 failed, 13095 passed, 5492 skipped`;
  - a near-complete `--junitxml` rerun captured `2098` unique failing nodeids
    across `12` unit files in
    `gb200_current_branch_test_unit_xml_failures_2026-04-10.txt`; and
  - the exact same `2098` nodeids are fully green/skip on merge-base:
    `2083 passed, 15 skipped`.
- That closes the "already failing on main?" question for the broad unit
  surface: it is branch-local.
- It does **not** mean there are `2098` independent fresh branch bugs:
  - isolated current-branch reruns of representative exact nodeids from the
    XML-only tail files (`test_standard.py`, `test_random.py`,
    `test_cache.py`, `test_blaslt.py`, `test_autotuner.py`,
    `test_launch.py`, `test_bindings.py`, and `test_triton_to_gluon.py`) all
    pass cleanly; so
  - the current read is "few primary MMAv5/TMEM buckets plus substantial
    branch-local xdist/process fallout after bad kernels", not twelve separate
    root-cause families.
- The current dirty worktree also has one new positive fix candidate in
  `TensorMemoryToLLVM.cpp`:
  - root TMEM ld/st lowering now stays on the exact raw-query path instead of
    preferring the same-atom query-type rescue first;
  - this restores the representative wide-N root-load unit matmul exact
    `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-1-64-512-32-2-float32-tensorfloat32]`;
  - and it keeps the repaired split-N TMEM controls green:
    - `test_tmem_descriptor_chain_matrix[...]`
    - `test_tmem_linear_roundtrip_splitn_shapes[...]`
    - `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[...]`
    - `test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-32-warps2-8-False-True-acc_dtype0]`
    - `test_block_m_64_mma[linear]`.
- The remaining fresh exact branch-local failures on the current dirty tree are
  now concentrated in a short list:
  - launch-time misaligned address:
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
    - `python/test/unit/language/test_matmul.py::test_lhs_in_tmem[float32-False-64-128-32]`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
    - `python/test/regression/test_cast_matmul.py::test_cast_matmul[768-768-1024-16-64-16-bfloat16-float16-float16]`
  - stable wrong-code:
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_persistent_forward[False-8-True-2-128-64-1024-1024]`
      with `65535 / 131072` mismatches (`50.0%`)
  - compile-time row-anchor materialization gap:
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-128-32]`
    - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float32]`
      both still fail with
      `unsupported tensor memory descriptor view for direct tcgen05.ld/st: required row anchors 32,64 are not directly representable in the descriptor view`
      before the software pipeliner asserts.

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

6. Real GB200 regression-suite bucket outside the TMEM reinterpret tests
- Command:
  - `make test-regression`
- Current result:
  - `234 failed, 856 passed, 216 skipped`
- Failure concentration:
  - every observed failure was in `python/test/regression/test_cast_matmul.py`
- Classification:
  - `REAL_BUG`
- Current interpretation:
  - this is not part of the old TMEM reinterpret-contract debate
  - it is a broader Blackwell regression-suite wrong-code bucket and needs its
    own follow-up after the TMEM census is recorded

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
