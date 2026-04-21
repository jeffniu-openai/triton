# Round 58 TMEM fuzz lane C: plain MMAv5 accumulator views

Date: 2026-04-21 16:01 UTC

Branch: `codex/tmem`

HEAD before report commit: `8521122cf`

Scope: discovery/cataloging only. No backend/compiler repairs attempted.

## Objective

Focus plain MMAv5 accumulator view and layout coverage, especially checked-in
runtime rows for indexed accumulator views, accumulator subslices,
tile-permuted accumulator layouts, `use_acc`, and 2CTA variants. Avoid broad
coverage already exercised by the recent `test_core.py` breadth lane except
for a tiny exact control replay.

## Build

Command:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Checked-In Runtime Matrix Selector

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k '(mma_indexed_acc_view or mma_acc_subslice_view_plain_kinds or mma_plain_kinds_tile_permuted_acc or mma_twocta_plain_kinds or mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds or mma_twocta_tma_tf32_b_transposed_descriptor) and not scaled and not m64 and not reports_clean and not clean_error and not exotic and not rowcol'
```

Collection result:

```text
142/1615 tests collected
```

Coverage included:

- 1CTA indexed accumulator views with legacy, linear, and linear unit-parent layouts.
- 1CTA accumulator subslice views with `slice_start` at zero and nonzero offsets.
- 1CTA tile-permuted accumulator layouts, with and without `use_acc`.
- 2CTA plain MMAv5 root accumulator layouts, with and without `use_acc`.
- 2CTA indexed accumulator views with legacy, linear, and linear unit-parent layouts.
- 2CTA accumulator subslice views, including nonzero subslice offsets and `use_acc`.
- 2CTA TF32 transposed-B descriptor rows, with and without `use_acc`.

Shard commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(mma_indexed_acc_view or mma_acc_subslice_view_plain_kinds or mma_plain_kinds_tile_permuted_acc or mma_twocta_plain_kinds or mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds or mma_twocta_tma_tf32_b_transposed_descriptor) and not scaled and not m64 and not reports_clean and not clean_error and not exotic and not rowcol'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k '(mma_indexed_acc_view or mma_acc_subslice_view_plain_kinds or mma_plain_kinds_tile_permuted_acc or mma_twocta_plain_kinds or mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds or mma_twocta_tma_tf32_b_transposed_descriptor) and not scaled and not m64 and not reports_clean and not clean_error and not exotic and not rowcol'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k '(mma_indexed_acc_view or mma_acc_subslice_view_plain_kinds or mma_plain_kinds_tile_permuted_acc or mma_twocta_plain_kinds or mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds or mma_twocta_tma_tf32_b_transposed_descriptor) and not scaled and not m64 and not reports_clean and not clean_error and not exotic and not rowcol'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k '(mma_indexed_acc_view or mma_acc_subslice_view_plain_kinds or mma_plain_kinds_tile_permuted_acc or mma_twocta_plain_kinds or mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds or mma_twocta_tma_tf32_b_transposed_descriptor) and not scaled and not m64 and not reports_clean and not clean_error and not exotic and not rowcol'
```

Shard results:

```text
group 1: 36 passed, 1579 deselected
group 2: 36 passed, 1579 deselected
group 3: 36 passed, 1579 deselected
group 4: 34 passed, 1581 deselected
aggregate: 142 passed
```

## Exact `test_core.py` Controls

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_core.py -k 'test_tcgen05_mma_plain_kind_runtime or test_tcgen05_mma_multicast_commit_twocta_linear_acc'
```

Collection result:

```text
5/18114 tests collected
```

Runtime command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_core.py -k 'test_tcgen05_mma_plain_kind_runtime or test_tcgen05_mma_multicast_commit_twocta_linear_acc'
```

Runtime result:

```text
5 passed, 18109 deselected
```

## Classification

No failures were observed.

No compiler crash, verifier drift, false unsupported diagnostic, opcode
absence, commit opcode mismatch, runtime miscompile, hang, unexpected
skip/fail transition, or new independent `FZ-*` was found.

Existing buckets such as `FZ-20260421-0003` and `FZ-20260421-0011` were not
reproduced by this lane. The selected plain MMAv5 accumulator-view rows stayed
green across 1CTA/2CTA, indexed/subslice/tile-permuted/use-acc coverage.

## Artifacts

Logs were kept locally under:

- `/tmp/tmem_round58_lane_c_runtime_g0.log`
- `/tmp/tmem_round58_lane_c_runtime_g1.log`
- `/tmp/tmem_round58_lane_c_runtime_g2.log`
- `/tmp/tmem_round58_lane_c_runtime_g3.log`
- `/tmp/tmem_round58_lane_c_test_core.log`
