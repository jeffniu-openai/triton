# Round 20 Local: clean-negative and structural xfail selector

- Date: 2026-04-21 12:40 UTC
- Branch: `codex/tmem`
- Commit at start: `e9046f362`
- Mode: discovery/validation only. No backend/compiler fixes were attempted.
- Duration cache:
  `/tmp/tmem_r20_clean_xfail_durations.json`

## Scope

This local selector checked broad clean-negative runtime rows together with
known structural fuzzer dynamic-memdesc xfail sentinels:

- clean unsupported/error/out-of-resource runtime matrix rows;
- non-f32 `ld.red` software-reduce contract rows;
- subword copy clean-error rows;
- structural fuzzer generic-pass and dynamic-index xfail rows.

Selector:

```text
(reports_clean_unsupported or reports_clean_error or reports_tmem_oor or non_f32_contract or subword_dtypes_report_clean_error or generic_pass or dynamic_index) and not block_descriptor
```

Files:

```text
python/test/gluon/test_tmem_runtime_matrix.py
python/test/gluon/test_tmem_structural_fuzzer.py
```

## Required Rebuild

Command:

```bash
make -j8
```

Result: no work to do.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k '(reports_clean_unsupported or reports_clean_error or reports_tmem_oor or non_f32_contract or subword_dtypes_report_clean_error or generic_pass or dynamic_index) and not block_descriptor'
```

Result: `208/1648` selected.

## Runtime

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_r20_clean_xfail_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k '(reports_clean_unsupported or reports_clean_error or reports_tmem_oor or non_f32_contract or subword_dtypes_report_clean_error or generic_pass or dynamic_index) and not block_descriptor'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `52 passed, 1596 deselected in 6.38s` |
| 2 | 1 | `52 passed, 1596 deselected in 11.06s` |
| 3 | 2 | `52 passed, 1596 deselected in 4.77s` |
| 4 | 3 | `41 passed, 1596 deselected, 11 xfailed in 12.42s` |

Aggregate selected rows: `197 passed, 11 xfailed`.

## Classification

No new failures were observed in this selector.

- New independent `FZ-*`: none.
- Unexpected pass: none observed.
- Unexpected failure: none observed.
- Runtime miscompile outside known rows: none observed.
- Compiler crash outside known xfail rows: none observed.

The xfail diagnostics remained the expected illegal `ttg.memdesc_index`
conversion failures for known dynamic/generic memdesc report-only rows.
