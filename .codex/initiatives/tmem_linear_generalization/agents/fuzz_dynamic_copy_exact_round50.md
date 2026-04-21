# Round 50 Dynamic Copy Exact Probe

Date: 2026-04-21 14:55 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Run a compact exact `/tmp` probe around dynamic/control-flow-yielded TMEM copy
descriptors while the broader dynamic descriptor Round 50 agent runs. The probe
checks whether branch-selected `warpx2` copy positives still pass before and
after the minimized branch-selected linear-copy expected failure.

## Required Build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Runtime Command

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[01_23-0-index]' \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor_128x128b[0]' \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[02_13-1-slice_index]'
```

Result:

```text
1 failed, 2 passed
```

The failed row is the expected existing `FZ-20260421-0001` minimized dynamic
linear-copy sentinel. It failed at `ConvertTritonGPUToLLVM` with a live illegal
`ttg.memdesc_index` feeding `ttng.tmem_copy` and `ttng.tmem_load`.

The two branch-selected `warpx2` copy rows passed in the same pytest process,
before and after the failing linear row.

## Classification

No new independent `FZ-*` bucket is proposed. This is stable existing
`FZ-20260421-0001` evidence with adjacent dynamic `warpx2` positives.

Backend repair remains deferred per the discovery-only campaign.
