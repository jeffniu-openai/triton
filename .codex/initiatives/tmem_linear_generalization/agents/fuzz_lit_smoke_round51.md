# Round 51 Local Lit Smoke

Date: 2026-04-21 14:56 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Run a compact lit smoke over TMEM layout, memdesc subview/split, and
proxy-fence insertion tests while Round 51 runtime and reproducer agents run.

## Commands

```bash
make -j8
cd $(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
ninja triton-opt
lit -v test/TritonNvidiaGPU/tmem_layouts.mlir test/TritonGPU/memdesc-subview-split.mlir test/TritonGPU/proxy_fence_insertion.mlir
```

Result:

```text
Total Discovered Tests: 3
  Passed: 3
```

## Classification

No FileCheck drift, verifier drift, proxy-fence lit drift, or new independent
`FZ-*` bucket was observed.

Backend repair remains deferred per the discovery-only campaign.
