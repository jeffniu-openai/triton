# Round 17 Local: copy and ld/st descriptor runtime selector

- Date: 2026-04-21 11:38 UTC
- Branch: `codex/tmem`
- Mode: discovery/validation only. No backend/compiler fixes were attempted.
- Duration cache:
  `/tmp/tmem_local_r17_copy_ldst_durations.json`

## Scope

This local selector exercised a mixed checked-in surface around TMEM copy and
`ld/st` descriptor algebra while Round 17 subagent lanes explored FZ-0015
lowering and copy/ldst composition:

```text
(cp_no_scales or cp_scales or ldst_descriptor_compositions or ldst_descriptor_roundtrip) and not reports
```

Coverage includes descriptor composition and roundtrip rows, row/column
permuted descriptor roundtrips, no-scales copy shapes, warpx2 copy variants,
subword clean-boundary copy rows, two-CTA copy rows, and scales-copy rows used
by scaled-MMAv5 setup.

## Required rebuild

Command:

```bash
make -j8
```

Result: no work to do. The rebuild was run immediately before the structural
fuzzer smoke gate in this same local slice.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales or cp_scales or ldst_descriptor_compositions or ldst_descriptor_roundtrip) and not reports'
```

Result: `344/1615` selected.

## Runtime

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_local_r17_copy_ldst_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales or cp_scales or ldst_descriptor_compositions or ldst_descriptor_roundtrip) and not reports'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `35 passed, 51 skipped, 1529 deselected in 4.31s` |
| 2 | 1 | `82 passed, 4 skipped, 1529 deselected in 25.95s` |
| 3 | 2 | `86 passed, 1529 deselected in 26.49s` |
| 4 | 3 | `86 passed, 1529 deselected in 6.56s` |

Aggregate selected rows: `289 passed`, `55 skipped`.

## Adjacent sentinel smoke

The checked-in structural fuzzer was also rerun locally:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `9 passed, 24 xfailed in 8.20s`.

## Classification

No new failures were observed in this selector.

- New independent `FZ-*`: none.
- Runtime miscompile: none observed.
- Compiler crash: none observed.
- False unsupported diagnostic: none observed in this checked-in selector.
- Existing sentinels: structural-fuzzer expected failures remain stable.

This is a green current-head baseline for copy and `ld/st` descriptor rows next
to the active Round 17 copy/ldst mixed subagent lane.
