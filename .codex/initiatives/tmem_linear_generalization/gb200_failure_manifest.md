# GB200 Failure Manifest

This file is the durable index for the exact GB200 current-branch failure
lists generated during the 2026-04-10 census. The `.txt` files beside it are
meant to be machine-usable artifacts, not hand-maintained prose.

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

## Notes

- These manifests reflect the current branch tip at the time of generation.
- Branch-vs-main classification is still in progress for some merge-base-
  existing lists.
- Proton is included here for census completeness even though it is already
  known to be preexisting on merge-base.
