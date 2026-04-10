# GB200 NVIDIA CI Inventory

This file is the current CI-grounded validation baseline for the TMEM linear
generalization initiative.

For the current stabilization phase, scope is intentionally limited to the
NVIDIA GB200 lane from `.github/workflows/integration-tests-nvidia.yml`. This
GB300 devbox is treated as representative for Blackwell TMEM behavior in that
lane.

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

As of 2026-04-10, after `make -j8` and exact-nodeid reruns on this devbox, the
confirmed GB200/NVIDIA red list is:

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
- If the explicit rewrite itself is blocked by missing TMEM view/index or
  direct-lowering surface coverage, record that as initiative debt and move the
  work back to the core compiler/API backlog instead of forcing the old
  reinterpret behavior to remain the contract.
