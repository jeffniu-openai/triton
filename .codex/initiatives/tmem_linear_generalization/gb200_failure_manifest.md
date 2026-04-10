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

- [gb200_current_branch_test_unit_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_unit_failures.txt)
  - `182` nodeids
  - source:
    - `make NUM_PROCS=24 test-unit`
    - plus the exact `python/test/unit/test_debug.py` rerun
- [gb200_current_branch_test_gluon_group3_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_gluon_group3_failures.txt)
  - `1160` nodeids
  - source:
    - `python3 -m pytest -q -rf --tb=no --splits 4 --group 3 -k 'not test_tmem_subslice_block_m_64 and not test_tmem_subslice_block_m_64_parent_layout' python/test/gluon/ python/tutorials/gluon/`
- [gb200_current_branch_test_gluon_group4_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_gluon_group4_failures.txt)
  - `686` nodeids
  - source:
    - `python3 -m pytest -q -rf --tb=no --splits 4 --group 4 -k 'not test_tmem_subslice_block_m_64 and not test_tmem_subslice_block_m_64_parent_layout' python/test/gluon/ python/tutorials/gluon/`
- [gb200_current_branch_test_regression_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_test_regression_failures.txt)
  - `234` nodeids
  - source:
    - `python3 -m pytest -q -rf --tb=no python/test/regression/test_cast_matmul.py`
- [gb200_current_branch_examples_gluon_failures.txt](/root/code/triton-tmem-isolated/.codex/initiatives/tmem_linear_generalization/gb200_current_branch_examples_gluon_failures.txt)
  - `62` nodeids
  - source:
    - `python3 -m pytest -q -rf --tb=no python/examples/gluon/`
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
  - `162` nodeids
  - meaning:
    - the current-branch `test-unit` failures that survive the preexisting
      `test_debug.py` carve-out and therefore belong in the branch recovery
      backlog
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
