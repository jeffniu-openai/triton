# Round 19 Local: structural fuzzer xfail sentinel selector

- Date: 2026-04-21 12:09 UTC
- Branch: `codex/tmem`
- Commit at start: `210e51ac3`
- Mode: discovery/validation only. No backend/compiler fixes were attempted.

## Scope

This local selector checked that known structural fuzzer report-only sentinels
still classify as expected xfails rather than drifting into unexpected passes,
unexpected failures, hangs, or different diagnostics.

Selector:

```text
(fz0015 or generic_pass or dynamic_index or proxy_fence or reports_fz or reports_bug) and not performance
```

Files:

```text
python/test/gluon/test_tmem_structural_fuzzer.py
python/test/gluon/test_tmem_runtime_matrix.py
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
  python/test/gluon/test_tmem_structural_fuzzer.py \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(fz0015 or generic_pass or dynamic_index or proxy_fence or reports_fz or reports_bug) and not performance'
```

Result: `11/1648` selected.

## Runtime

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(fz0015 or generic_pass or dynamic_index or proxy_fence or reports_fz or reports_bug) and not performance'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `3 xfailed, 1645 deselected in 5.06s` |
| 2 | 1 | `3 xfailed, 1645 deselected in 4.43s` |
| 3 | 2 | `3 xfailed, 1645 deselected in 4.51s` |
| 4 | 3 | `2 xfailed, 1646 deselected in 4.45s` |

Aggregate selected rows: `11 xfailed`.

## Classification

No new failures were observed in this selector.

- New independent `FZ-*`: none.
- Unexpected pass: none observed.
- Unexpected failure: none observed.
- Runtime miscompile outside known xfail rows: none observed.
- Compiler crash outside known xfail rows: none observed.

The visible diagnostics remained the expected illegal `ttg.memdesc_index`
conversion failures for known generic-pass/dynamic-index report-only rows.
