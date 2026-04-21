# Round 51 Local Core Control Smoke

Date: 2026-04-21 14:57 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Run a compact `test_core.py` smoke over TMEM copy, descriptor-chain, M64
split-N default-load, and two-CTA MMAv5 multicast commit linear-accumulator
controls.

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
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short \
  -k '(tmem_copy_no_scales_shared_linear_128x128b or tmem_descriptor_chain_matrix or tmem_legacy_m64_subview_default_load_auto_selects_splitn or tcgen05_mma_multicast_commit_twocta_linear_acc)' \
  python/test/gluon/test_core.py
```

Result:

```text
31 passed, 18083 deselected
```

## Classification

No compiler crash, runtime miscompare, skip drift, or new independent `FZ-*`
bucket was observed.

Backend repair remains deferred per the discovery-only campaign.
