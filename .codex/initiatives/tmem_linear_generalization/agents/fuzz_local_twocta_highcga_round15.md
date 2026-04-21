# Round 15 Local: twoCTA and High-CGA Runtime Selector

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/validation only. No backend/compiler fixes were attempted.
- Logs:
  - `/tmp/tmem_local_r15_twocta_highcga_g1.log`
  - `/tmp/tmem_local_r15_twocta_highcga_g2.log`
  - `/tmp/tmem_local_r15_twocta_highcga_g3.log`
  - `/tmp/tmem_local_r15_twocta_highcga_g4.log`
  - `/tmp/tmem_local_r15_twocta_highcga_durations.json`

## Scope

This local selector followed up the Round 14 high-CGA and twoCTA ownership
audits. It exercised checked-in runtime-matrix rows selected by:

```text
(twocta or cga_roundtrip or layout_in_4cta_context) and not reports
```

The goal was to sanity-check the current branch after the Round 14 proxy-fence
and high-CGA findings without adding new tests or attempting repairs.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(twocta or cga_roundtrip or layout_in_4cta_context) and not reports'
```

Result: `331/1615` selected.

## Runtime

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_local_r15_twocta_highcga_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(twocta or cga_roundtrip or layout_in_4cta_context) and not reports'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `56 passed, 27 skipped, 1532 deselected in 4.29s` |
| 2 | 1 | `73 passed, 10 skipped, 1532 deselected in 4.84s` |
| 3 | 2 | `83 passed, 1532 deselected in 5.51s` |
| 4 | 3 | `82 passed, 1533 deselected in 5.20s` |

Aggregate selected rows: `294 passed`, `37 skipped`.

## Classification

No new failures were observed in this selector.

The pass result is a useful current-head control for the Round 14 multi-CTA
findings:

- checked-in twoCTA load/store, copy, MMAv5, scaled-MMAv5, and CGA descriptor
  roundtrip rows remain green;
- no new `FZ-20260421-0010` high-CGA gate regression was exposed by this
  selector; and
- no new `FZ-20260421-0014` proxy-fence insertion crash was exposed by the
  checked-in positive twoCTA rows.
