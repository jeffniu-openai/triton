# Round 18 Local: descriptor ld/st plus cp_scales runtime selector

- Date: 2026-04-21 11:48 UTC
- Branch: `codex/tmem`
- Commit at start: `71d678d05`
- Mode: discovery/validation only. No backend/compiler fixes were attempted.
- Duration cache:
  `/tmp/tmem_r18_ldst_cpscales_durations.json`

## Scope

This local selector checked green runtime coverage adjacent to the active
Round 18 proxy-fence and dynamic-clean-boundary subagent lanes:

- descriptor `ld/st` composition runtime rows, including permuted and exotic
  linear layouts;
- `cp_scales` layout probes;
- `cp_scales` rematerialization through noncanonical shared layouts and shared
  subslices;
- `warpx4` scale-copy rows, including scaled-MMAv5 copy-matrix consumers.

Selector:

```text
(ldst_descriptor_compositions or ldred_descriptor or cp_scales) and not reports
```

`ldred_descriptor` did not add rows under this selector on current HEAD, but
the spelling is recorded because it was intentionally included while searching
for non-overlapping TMEM-bearing coverage.

## Required Rebuild

Command:

```bash
make -j8
```

Result from the immediately preceding checkpoint: no work to do.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor_compositions or ldred_descriptor or cp_scales) and not reports'
```

Result: `62/1615` selected.

## Runtime

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_r18_ldst_cpscales_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor_compositions or ldred_descriptor or cp_scales) and not reports'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `16 passed, 1599 deselected in 4.34s` |
| 2 | 1 | `16 passed, 1599 deselected in 29.19s` |
| 3 | 2 | `16 passed, 1599 deselected in 7.49s` |
| 4 | 3 | `14 passed, 1601 deselected in 4.76s` |

Aggregate selected rows: `62 passed`.

## Classification

No new failures were observed in this selector.

- New independent `FZ-*`: none.
- Runtime miscompile: none observed.
- Compiler crash: none observed.
- Unexpected unsupported diagnostic: none observed.

This is a green current-head baseline for descriptor `ld/st` composition and
scale-copy runtime rows while Round 18 subagents probe more adversarial dynamic
descriptor and proxy-fence interval shapes.
