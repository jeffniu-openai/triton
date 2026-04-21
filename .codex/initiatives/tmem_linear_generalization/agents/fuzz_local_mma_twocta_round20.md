# Round 20 Local: 2CTA MMAv5 and scaled accumulator-subslice selector

- Date: 2026-04-21 12:35 UTC
- Branch: `codex/tmem`
- Commit at start: `e4f14e5a0`
- Mode: discovery/validation only. No backend/compiler fixes were attempted.
- Duration cache:
  `/tmp/tmem_r20_mma_twocta_durations.json`

## Scope

This local selector checked 2CTA MMAv5 and scaled-MMAv5 accumulator-subslice
runtime coverage adjacent to, but not overlapping, Lane AX's selected B-scale
allocation-order probes:

- MMAv5 `use_acc=True` plain-kind rows;
- MMAv5 indexed accumulator view rows;
- MMAv5 accumulator subslice view rows;
- MMAv5 TMA-transposed descriptor `use_acc=True` rows;
- scaled-MMAv5 2CTA accumulator subslice format rows with and without
  `use_acc`.

Selector:

```text
((mma_twocta and (indexed_acc_view or acc_subslice_view_plain_kinds or plain_kinds_use_acc or tma_tf32_b_transposed_descriptor_use_acc)) or (mma_scaled and twocta_acc_subslice)) and not reports and not clean
```

## Required Rebuild

Command:

```bash
make -j8
```

Result from the immediately preceding local work: no work to do.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '((mma_twocta and (indexed_acc_view or acc_subslice_view_plain_kinds or plain_kinds_use_acc or tma_tf32_b_transposed_descriptor_use_acc)) or (mma_scaled and twocta_acc_subslice)) and not reports and not clean'
```

Result: `90/1615` selected.

## Runtime

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_r20_mma_twocta_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '((mma_twocta and (indexed_acc_view or acc_subslice_view_plain_kinds or plain_kinds_use_acc or tma_tf32_b_transposed_descriptor_use_acc)) or (mma_scaled and twocta_acc_subslice)) and not reports and not clean'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `23 passed, 1592 deselected in 4.24s` |
| 2 | 1 | `23 passed, 1592 deselected in 4.26s` |
| 3 | 2 | `23 passed, 1592 deselected in 4.59s` |
| 4 | 3 | `21 passed, 1594 deselected in 4.53s` |

Aggregate selected rows: `90 passed`.

## Classification

No new failures were observed in this selector.

- New independent `FZ-*`: none.
- Runtime miscompile: none observed.
- Compiler crash: none observed.
- Unexpected unsupported diagnostic: none observed.

This is a current-head green baseline for the selected 2CTA MMAv5 and
scaled-MMAv5 accumulator-subslice runtime surface.
