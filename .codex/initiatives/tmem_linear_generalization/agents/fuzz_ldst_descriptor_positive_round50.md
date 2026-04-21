# Round 50 LD/ST Descriptor Positive Sweep

Date: 2026-04-21 14:53 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Run a positive runtime selector over TMEM `ld/st` descriptor-view surfaces,
including two-CTA and scales descriptor rows, while excluding clean diagnostics,
resource boundaries, M64, and non-f32 rows.

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
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short \
  -k '(ldst_descriptor or ldst_twocta or ldst_scales) and not reports and not clean and not resource and not m64 and not non_f32' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result:

```text
166 passed, 98 skipped, 1351 deselected
```

The skips match existing allocation/resource skips in the checked-in matrix;
no selected positive row failed.

## Classification

No compiler crash, false unsupported diagnostic, runtime miscompile, skip drift
beyond known checked-in skips, or new independent `FZ-*` bucket was observed.

Backend repair remains deferred per the discovery-only campaign.
