# Round 17 Local: clean-boundary runtime selector

- Date: 2026-04-21 11:41 UTC
- Branch: `codex/tmem`
- Mode: discovery/validation only. No backend/compiler fixes were attempted.
- Duration cache:
  `/tmp/tmem_local_r17_clean_boundaries_durations.json`

## Scope

This local selector checked that known clean unsupported, clean error,
out-of-resource, subword-copy, non-f32 `ld.red`, MMAv5, scaled-MMAv5, and
descriptor-view boundaries still fail or pass according to their checked-in
contracts rather than drifting into compiler crashes, late illegal operations,
or runtime miscompiles.

Selector:

```text
(reports_clean_unsupported or reports_clean_error or reports_tmem_oor or subword_dtypes_report_clean_error or non_f32_contract)
```

## Required rebuild

Command:

```bash
make -j8
```

Result: no work to do.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(reports_clean_unsupported or reports_clean_error or reports_tmem_oor or subword_dtypes_report_clean_error or non_f32_contract)'
```

Result: `199/1615` selected.

## Runtime

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_local_r17_clean_boundaries_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(reports_clean_unsupported or reports_clean_error or reports_tmem_oor or subword_dtypes_report_clean_error or non_f32_contract)'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `50 passed, 1565 deselected in 22.26s` |
| 2 | 1 | `50 passed, 1565 deselected in 11.34s` |
| 3 | 2 | `50 passed, 1565 deselected in 4.57s` |
| 4 | 3 | `49 passed, 1566 deselected in 11.16s` |

Aggregate selected rows: `199 passed`.

## Classification

No new failures were observed in this selector.

- New independent `FZ-*`: none.
- Runtime miscompile: none observed.
- Compiler crash: none observed.
- Late illegal-op failure: none observed.
- False unsupported diagnostic: none observed in this checked-in selector.

This is a current-head clean-boundary baseline while Round 17 subagent lanes
probe adjacent unsupported/verifier surfaces.
