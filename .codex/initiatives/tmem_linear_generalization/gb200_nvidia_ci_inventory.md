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

3. Real TMEM split-N regression
- Nodeids confirmed clean on an isolated GPU:
  - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-layout9-64-128-32x32b_splitn-8-16x32bx2]`
  - `python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]`
- Current result:
  - descriptor-chain case: `4096 / 8192` mismatches
  - split-N roundtrip case: `2048 / 2048` mismatches
- Classification:
  - `REAL_BUG`
- Current interpretation:
  - this is not shard contamination; both cases still fail on a clean isolated
    rerun
  - the current branch tip has regressed the previously green `M=64` split-N
    TMEM family again
  - nearby shard failures such as `test_block_m_64_mma[linear]` are not
    reliable by themselves until rerun clean, because at least that one passes
    in isolation after the shard failure

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

5. Real GB200 regression-suite bucket outside the TMEM reinterpret tests
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

## Executed GB200 Census Status

### Green Or Healthy

- `make test-cpp`
  - passed
  - `240 / 240` tests passed
- manual lit fallback
  - `cd $BUILD_DIR && python3 -m lit.main -sv test`
  - passed
- `make test-gsan`
  - passed
  - `20 passed`
- `make test-microbenchmark`
  - passed
  - completed both Tensor and TensorDescriptor launch-overhead runs
- `test-gluon` shard `2 / 4`
  - `2663 passed, 3784 skipped, 19344 deselected`
  - no new exact failures

### Local Environment Blockers

- `make test-lit`
  - local wrapper is broken on this devbox because `LLVM_EXTERNAL_LIT` is empty
    in `CMakeCache.txt`, which makes the generated command try to execute
    `/llvm-lit`
  - this is why the manual `python3 -m lit.main -sv test` fallback is used for
    local inventory here
- `make test-proton`
  - blocked locally by missing Python dependency `llnl-hatchet`
  - current import error:
    - `Failed to import hatchet. pip install llnl-hatchet to get the correct version.`
  - treat this as a local environment blocker until the devbox matches the CI
    proton environment

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
  - first useful failures:
    - real `M=64` split-N TMEM wrong-code
  - later `misaligned address` waves in the same shard are likely contamination
    until rerun clean
- `test-gluon` shard `4 / 4`
  - red
  - `687 failed, 4910 passed, 849 skipped, 19345 deselected`
  - first useful failure:
    - higher-rank half-row "reports clean error" case that no longer raises
  - at least one sibling case in that family passes clean in isolation, so much
    of the later shard output should be treated as stale-negative fallout and
    contamination until rerun clean
- `test-gluon` shard `1 / 4`
  - still running when this checkpoint entry was written
  - no exact failure had appeared yet

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
