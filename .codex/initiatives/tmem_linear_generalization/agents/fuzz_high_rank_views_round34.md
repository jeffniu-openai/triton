# Round 34 High-Rank Descriptor Views Runtime Lane

Date: 2026-04-21 13:20 UTC

Branch: `codex/tmem`
HEAD: `1777866d4`
Mode: discovery/cataloging only. No backend or compiler source was modified.

## Scope

This lane targeted adversarial high-rank descriptor-view shape algebra around
TMEM `ld/st`, `tcgen05.copy`, and hardware `.ld.red`:

- rank-4/rank-5 parents;
- unit dimensions adjacent to a non-unit prefix dimension;
- slice/index/reshape/permute chains;
- higher-rank dim0 slicing and row-half boundaries;
- roundtrip correctness, TMEM resource boundaries, and descriptor-view
  wrong-result classification against known `FZ-0002`/`FZ-0003`.

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

## Checked-In Baseline

Selector:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(higher_rank or multidim_slice or half_rows or rank5) and not reports'
```

The split runs used:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_high_rank_views_round34_checked_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(higher_rank or multidim_slice or half_rows or rank5) and not reports'
```

Logs:

- `/tmp/tmem_high_rank_views_round34_checked_g1.log`
- `/tmp/tmem_high_rank_views_round34_checked_g2.log`
- `/tmp/tmem_high_rank_views_round34_checked_g3.log`
- `/tmp/tmem_high_rank_views_round34_checked_g4.log`

Result: `82 passed, 20 skipped` across `102/1615` selected rows.

Per shard:

- group 1: `26 passed, 1589 deselected`
- group 2: `26 passed, 1589 deselected`
- group 3: `7 passed, 19 skipped, 1589 deselected`
- group 4: `23 passed, 1 skipped, 1591 deselected`

Focused rank-5 contrast:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'rank5_small_roundtrip or rank5_unit_parent_n256_roundtrip'
```

Result: `23 passed, 1592 deselected`.

## Temporary Probe

Probe: `/tmp/tmem_high_rank_views_round34_probe.py`

Validation:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_high_rank_views_round34_probe.py
PYTHONPATH=.:./python:./python/test/gluon pytest -q --collect-only /tmp/tmem_high_rank_views_round34_probe.py
```

Collection: `9 tests collected`.

Split command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  --splits 4 --group <group> \
  /tmp/tmem_high_rank_views_round34_probe.py
```

Logs:

- `/tmp/tmem_high_rank_views_round34_probe_g1.log`
- `/tmp/tmem_high_rank_views_round34_probe_g2.log`
- `/tmp/tmem_high_rank_views_round34_probe_g3.log`
- `/tmp/tmem_high_rank_views_round34_probe_g4.log`
- exact rerun log: `/tmp/tmem_high_rank_views_round34_probe_exact.log`

Initial split result:

- groups 1 and 2 aborted on rank-5 `ld/st` rows with:
  `LLVM ERROR: Dimensions must match, ignoring order, but they don't. Got dims:
  ["dim0", "dim1"] and ["dim1", "dim2"]`
- group 3: `3 failed, 6 deselected`
- group 4: `9 deselected`

Exact fresh-process classification:

| Classification | Count | Rows |
| --- | ---: | --- |
| Pass | 2 | `copy-rank4-index0-128x128`, `copy-rank4-index1-128x128` |
| Clean TMEM OOR | 1 | `copy-rank4-index1-128x256`, `Required: 1024, Hardware limit: 512` |
| Existing `FZ-20260421-0002`/`FZ-20260421-0003` wrong-result surface | 1 | `ldred-rank4-chain0-128x128`, `16128/16384` output mismatches |
| Clean unsupported row-half descriptor view | 1 | `ldred-rank4-dim0-halfrows-128x128`, clean diagnostic says row-half views need row-origin decomposition |
| New candidate `FZ-20260421-0019` | 4 | all four rank-5 `ld/st` rows abort during `get_reg_layout` |

## New Candidate Bucket

`FZ-20260421-0019`: rank-5 `ld/st` descriptor views with unit dimensions around
a non-unit prefix dimension abort during Gluon `get_reg_layout` with an LLVM
dimension-set mismatch.

Seed rows:

- `ldst-rank5-unit-chain0-128x128`
- `ldst-rank5-unit-multidim-128x128`
- `ldst-rank5-dim0-halfrows-128x128`
- `ldst-rank5-unit-chain0-128x256`

Repro command for a single seed:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  '/tmp/tmem_high_rank_views_round34_probe.py::test_high_rank_views[ldst-rank5-unit-chain0-128x128]'
```

Failure mode: hard compiler/runtime abort, not a clean diagnostic and not a
runtime miscompare.

Likely owner surface: high-rank descriptor-view layout algebra in the
Gluon/TMEM `get_reg_layout` path, specifically rank-name/dimension matching
after slicing/indexing `[1, 2, 1, M, N]` parents down to 2-D views and then
applying reshape/permute view chains.

Contrast: checked-in rank-5 small and unit-parent controls passed as
`23 passed`, and the broader checked-in high-rank selector stayed stable as
`82 passed, 20 skipped`, so this is not a blanket rank-5 failure.

## Classification

No backend repair was attempted.

Known bucket coverage:

- `FZ-20260421-0001`: no dynamic `memdesc_index` lane was exercised here, and
  no late illegal `ttg.memdesc_index` appeared.
- `FZ-20260421-0002`/`FZ-20260421-0003`: `ldred-rank4-chain0-128x128` matches
  existing descriptor-view wrong-result evidence.
- Clean boundaries: rank-4 copy at `128x256` cleanly hit the TMEM hardware
  allocation limit, and row-half `.ld.red` descriptor view produced the
  existing clean unsupported row-origin diagnostic.
- New candidate: `FZ-20260421-0019` for the rank-5 unit-dimension `ld/st`
  `get_reg_layout` abort.
