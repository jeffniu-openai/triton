# Round 51 Local Proxy Pair Check

Date: 2026-04-21 15:01 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Pair the green checked-in proxy-fence lit test with the minimized Round 51
`FZ-20260421-0014` proxy-fence reproducer.

## Commands

```bash
cd $(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
ninja triton-opt
lit -v test/TritonGPU/proxy_fence_insertion.mlir
bin/triton-opt /tmp/tmem_round51_fz0014_no_store.mlir --run-reproducer
```

## Results

- Checked-in lit: `1 passed`.
- Minimized `FZ-20260421-0014`: exit code `1`, expected diagnostic:
  `could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses`.

## Classification

No checked-in proxy-fence lit drift and no `FZ-0014` signature drift were
observed. No new independent `FZ-*` bucket.

Backend repair remains deferred per the discovery-only campaign.
