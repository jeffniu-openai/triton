# Round 20 Local: descriptor roundtrip and copy/scales selector

- Date: 2026-04-21 12:30 UTC
- Branch: `codex/tmem`
- Commit at start: `576e0bb71`
- Mode: discovery/validation only. No backend/compiler fixes were attempted.
- Duration cache:
  `/tmp/tmem_r20_roundtrip_copy_durations.json`

## Scope

This local selector checked green runtime coverage while Lane AX probed
`FZ-20260421-0015` allocation-order sensitivity:

- descriptor `ld/st` roundtrip sweeps, including row/col-permuted variants;
- `cp_scales` layout probes and scaled-MMAv5 copy-matrix consumers;
- positive `cp_no_scales warpx2::01_23` direct, 2CTA, subslice, indexed, and
  dense-shared rows.

Selector:

```text
(ldst_descriptor_roundtrip or cp_scales or cp_no_scales_warpx2_01_23 or cp_no_scales_warpx4) and not reports and not tmem_oor
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
  -k '(ldst_descriptor_roundtrip or cp_scales or cp_no_scales_warpx2_01_23 or cp_no_scales_warpx4) and not reports and not tmem_oor'
```

Result: `102/1615` selected.

## Runtime

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_r20_roundtrip_copy_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor_roundtrip or cp_scales or cp_no_scales_warpx2_01_23 or cp_no_scales_warpx4) and not reports and not tmem_oor'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `26 skipped, 1589 deselected in 3.09s` |
| 2 | 1 | `1 passed, 25 skipped, 1589 deselected in 3.89s` |
| 3 | 2 | `26 passed, 1589 deselected in 4.29s` |
| 4 | 3 | `24 passed, 1591 deselected in 4.76s` |

Aggregate selected rows: `51 passed, 51 skipped`.

## Classification

No new failures were observed in this selector.

- New independent `FZ-*`: none.
- Runtime miscompile: none observed.
- Compiler crash: none observed.
- Unexpected unsupported diagnostic: none observed.

This is a current-head green/skipped baseline for descriptor roundtrip and
copy/scales surfaces adjacent to the active allocation-order fuzzing lane.
