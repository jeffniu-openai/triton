# Round 49 Scaled/MMAv5 and Scales-Copy Positive Union

Date: 2026-04-21 14:47 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Run a broad positive union over scaled-MMAv5 rows and scales-copy rows while
excluding clean diagnostics, resource boundaries, reports, and M64 cases. This
lane intentionally overlaps real execution paths across scaled MMA and copy
setup rather than testing explicit clean-boundary assertions.

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
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short \
  -k '((mma_scaled and not reports and not clean and not resource and not m64) or (cp_scales and not reports and not clean and not resource))' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result:

```text
276 passed, 1339 deselected
```

## Classification

No compiler crash, false unsupported diagnostic, runtime miscompile, opcode
drift, or new independent `FZ-*` bucket was observed.

Backend repair remains deferred per the discovery-only campaign.
