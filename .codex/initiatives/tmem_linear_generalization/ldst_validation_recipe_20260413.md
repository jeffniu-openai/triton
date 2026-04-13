# `ld/st` Runtime-Matrix Validation Recipe - 2026-04-13

## Scope

Checkpoint commit: `77d3edea1` (`codex/tmem`).

Selector:

```bash
PYTHONPATH=python:. pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k ldst
```

Collection result: `1642/2758` tests selected.

The old coarse four-way selector is not a good local recipe. A previous split-4
attempt left individual shards running far beyond the project owner's GB200 lane
timing reference. Static split-16 improves the shape, but still leaves dense
composition, roundtrip, and higher-rank buckets heavy enough to hit local timeout
wrappers on cold caches.

## Durable Duration Cache

Merged pytest-split duration cache:

```bash
.codex/initiatives/tmem_linear_generalization/experiments/results/ldst_pytest_durations_20260413.json
```

It contains timings for all `1642` selected `ldst` nodeids, merged from the
bucketed runs below. A collect-only check with this cache and `least_duration`
produces balanced split-16 groups around `780s` estimated each:

```bash
PYTHONPATH=python:. pytest --collect-only -q \
  --durations-path .codex/initiatives/tmem_linear_generalization/experiments/results/ldst_pytest_durations_20260413.json \
  --splitting-algorithm=least_duration --splits 16 --group <1..16> \
  python/test/gluon/test_tmem_runtime_matrix.py -k ldst
```

That is useful for avoiding pathological static partitions, but it also shows
this selector is intrinsically expensive locally: roughly `13m` estimated per
split-16 group. With four GPUs, a full local least-duration refresh is still a
multi-wave run. Prefer focused selectors for normal development, and reserve the
full selector for phase boundaries.

## Accepted Evidence

Static split-16 groups that completed cleanly:

- groups `1..4`: `412 passed` total (`103` each); group times `6:26`, `9:13`,
  `9:49`, `13:23`.
- groups `14..16`: `282 passed, 21 skipped` total; group times `7:33`, `2:03`,
  `4:19`.

Static split-16 groups not accepted as broad evidence:

- groups `5..8` timed out under a `900s` wrapper while traversing descriptor
  composition / roundtrip buckets.
- group `13` timed out under a `900s` wrapper while traversing higher-rank
  descriptor-view cases.

These timeout groups were replaced by exact function buckets:

- `test_tmem_runtime_matrix_ldst_descriptor_compositions_rowcol_permuted_layout_sweep`
  - split-8 exact bucket; result `240 passed`.
  - cold split-8 groups with many 256-row cases can still exceed `600s`; warmed
    reruns closed those groups.
  - slowest observed cases were around `36s`.
- `test_tmem_runtime_matrix_ldst_descriptor_roundtrip_rowcol_permuted_sweeps`
  - split-8 exact bucket; result `240 skipped`, no failures.
  - slowest clean skip/OOR paths were around `25s` for group 1 and around `18s`
    for the remaining groups.
- Remaining descriptor composition / roundtrip / two-CTA descriptor buckets:
  - exact split-8 over:
    - `test_tmem_runtime_matrix_ldst_descriptor_compositions`
    - `test_tmem_runtime_matrix_ldst_descriptor_compositions_permuted_layout_sweep`
    - `test_tmem_runtime_matrix_ldst_descriptor_compositions_exotic_layouts`
    - `test_tmem_runtime_matrix_ldst_descriptor_roundtrip_sweeps`
    - `test_tmem_runtime_matrix_ldst_twocta_block_layouts`
    - `test_tmem_runtime_matrix_ldst_twocta_descriptor_compositions`
    - `test_tmem_runtime_matrix_ldst_twocta_descriptor_roundtrip_sweeps`
  - result `180 passed, 180 skipped`, no failures.
- Higher-rank index / multidimensional descriptor buckets:
  - exact split-8 over:
    - `test_tmem_runtime_matrix_ldst_descriptor_higher_rank_index`
    - `test_tmem_runtime_matrix_ldst_descriptor_higher_rank_index_reports_tmem_oor`
    - `test_tmem_runtime_matrix_ldst_descriptor_multidim_slices`
    - `test_tmem_runtime_matrix_ldst_descriptor_multidim_slices_report_tmem_oor`
    - `test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_index`
    - `test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_index_reports_tmem_oor`
    - `test_tmem_runtime_matrix_ldst_twocta_descriptor_multidim_slices`
    - `test_tmem_runtime_matrix_ldst_twocta_descriptor_multidim_slices_report_tmem_oor`
  - result `120 passed`, no failures.

Together, this covers every function in the `-k ldst` collection with either a
completed static split group or an exact replacement bucket. No deterministic
`ld/st` runtime-matrix failure was found at `77d3edea1`.

## Recommended Commands

Use focused nodeids or function buckets first. For a broad local refresh, start
from the duration cache:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
TRITON_CACHE_DIR=/tmp/triton-cache-ldst-gpu<gpu> \
PYTHONPATH=python:. \
pytest -s --tb=short \
  --durations-path .codex/initiatives/tmem_linear_generalization/experiments/results/ldst_pytest_durations_20260413.json \
  --splitting-algorithm=least_duration \
  --splits 16 --group <group> \
  -q python/test/gluon/test_tmem_runtime_matrix.py -k ldst
```

If wall time matters more than a single broad command shape, prefer the bucketed
recipe from this file: static split-16 for the non-heavy front/tail groups, then
exact split-8 buckets for the composition, roundtrip, and higher-rank families.

## Open Validation Work

- The duration cache should be refreshed after material changes to the heavy
  `ld/st` runtime matrix.
- The broad local selector remains slower than the project owner's full GB200 CI
  timing reference. This is a validation throughput issue, not a product failure.
  Future work should consider reducing redundant heavy cases, using selected
  representative phase-boundary buckets, or carefully trying small xdist values
  for compile-heavy skip/OOR paths while watching for false OOMs.
