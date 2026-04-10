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
- `make test-proton`
  - red after the `llnl-hatchet` install removed the old import blocker
  - `10 failed, 114 passed, 1 skipped`
  - all observed failures were in `third_party/proton/test/test_profile.py`
  - current interpretation:
    - this is now a real Proton/cudagraph-profile bucket rather than an
      environment issue
    - failures are concentrated in expected tree-shape / frame-name /
      periodic-flush assertions for cudagraph profiling behavior

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
  - failure concentration:
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[...]`
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[...]`
    - `python/test/unit/language/test_matmul.py::test_lhs_in_tmem[...]`
  - isolated reruns already confirm both sub-buckets:
    - simple matmul wrong-code:
      - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-1-64-512-32-2-float32-tensorfloat32]`
      - `49.9%` mismatches
    - persistent matmul compile/pipeline failure:
      - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-8-64-128-32]`
      - `ttng.tmem_store` unsupported register layout / row anchors `32,64`
        followed by `PassManager::run failed`
- unit shard `4 / 4`
  - red
  - `131 failed, 4707 passed, 321 skipped`
  - failure concentration:
    - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[{float16,bfloat16,float32}]`
    - large systematic `python/test/unit/language/test_warp_specialization.py`
      forward and persistent-forward failure surface
  - isolated reruns already confirm representative standalone failures:
    - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float32]`
      - `ttng.tmem_store` unsupported register layout / row anchors `32,64`
        followed by `PassManager::run failed`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
      - `49.7%` mismatches
  - remaining work:
    - materialize the full exact failing nodeid list for the large
      warp-specialization bucket in a dedicated follow-up rerun so merge-base
      comparison can be done nodeid-by-nodeid

Tail-command status:

- `python/test/unit/test_debug.py`
  - red
  - xdist-shaped run: `20 failed, 75 passed`
  - serial rerun: same `20 failed, 75 passed`
  - classification:
    - real harness/runtime problem, not just outer xdist noise
    - failures all show `Cannot re-initialize CUDA in forked subprocess`
      originating from `py._process.forkedfunc`
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
  - still running at the time of this update
  - inventory is not complete until this command finishes and its exact red
    surface is recorded

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

### Merge-Base Comparison Setup Status (2026-04-10)

- `origin/main` is now fetched locally.
- Merge-base for current branch vs main:
  - `7f61ac734edc657b737fb159a1b9d50cb47944e6`
- A detached worktree exists at:
  - `/root/code/triton-mergebase-ci`
- Baseline comparison is currently blocked by merge-base build-system drift:
  - raw `make` / direct CMake path on that older tree does not emit the
    `NVWS` tablegen products expected by the modern local toolchain;
  - a local-only shim was added in the merge-base worktree to skip example
    plugin builds so the baseline can configure farther; and
  - `setup.py build_ext` is currently being tried as a more faithful old-tree
    build path because it wires more compatibility arguments than the direct
    `make` path.
- Until that baseline build is usable, the branch-vs-main classification is
  incomplete. The current doc intentionally distinguishes:
  - exact local red buckets already confirmed on this branch, and
  - baseline comparison work still in progress.

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
