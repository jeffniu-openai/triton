# Round 51 Local Minimized Repro Confirmation

Date: 2026-04-21 14:59 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Independently rerun the new Round 51 minimized MLIR reproducers for
`FZ-20260421-0014`, `FZ-20260421-0016`, and `FZ-20260421-0017`.

## Command

```bash
cd $(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
bin/triton-opt /tmp/tmem_round51_fz0014_no_store.mlir --run-reproducer
bin/triton-opt /tmp/tmem_round51_fz0016_min.mlir -split-input-file
bin/triton-opt /tmp/tmem_round51_fz0017_min.mlir -split-input-file --convert-triton-gpu-to-llvm='target=sm_100'
```

## Results

- `FZ-20260421-0014`: exit code `1`; diagnostic:
  `could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses`.
- `FZ-20260421-0016`: exit code `134`; assertion:
  `dyn_cast on a non-existent value`.
- `FZ-20260421-0017`: exit code `134`; assertion:
  `bitwidth == 32`.

## Classification

The minimized reproducers are stable and still map to existing buckets. No new
independent `FZ-*` bucket was observed.

Backend repair remains deferred per the discovery-only campaign.
