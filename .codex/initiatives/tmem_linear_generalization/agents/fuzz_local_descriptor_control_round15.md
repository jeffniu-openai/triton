# Round 15 local descriptor/control runtime sweep

Date: 2026-04-21 11:28 UTC
Branch: `codex/tmem`
HEAD before report: `1fbbdf299`
Mode: discovery/cataloging only. No backend or compiler code was changed.

## Summary

This local sweep covered a broader checked-in descriptor/control surface after
the higher-rank descriptor checkpoint:

- `ld/st` descriptor compositions and descriptor roundtrips;
- row/column-permuted descriptor roundtrips;
- plain MMAv5 indexed and accumulator-subslice views;
- scaled-MMAv5 indexed and accumulator-subslice views;
- scaled-MMAv5 shared scale descriptor-view copy rows.

No new `FZ-*` bucket was found. The selected rows stayed green or skipped
according to their checked-in expectations.

## Required rebuild

Command:

```bash
make -j8
```

Result: no work to do, inherited from the preceding local slice.

## Selector

Collect-only command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(runtime_selector or descriptor_roundtrip or descriptor_compositions or scale_descriptor or acc_subslice_view or indexed_acc) and not reports'
```

Result: `329/1615` selected.

Runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_local_r15_descriptor_control_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(runtime_selector or descriptor_roundtrip or descriptor_compositions or scale_descriptor or acc_subslice_view or indexed_acc) and not reports'
```

## Results

- GPU 0 / group 1: `38 passed, 45 skipped, 1532 deselected in 3.84s`.
- GPU 1 / group 2: `50 passed, 33 skipped, 1532 deselected in 4.39s`.
- GPU 2 / group 3: `83 passed, 1532 deselected in 6.01s`.
- GPU 3 / group 4: `80 passed, 1535 deselected in 10.56s`.
- Aggregate: `251 passed, 78 skipped`.

## Classification

- New independent `FZ-*`: none.
- Runtime miscompile: none observed.
- Compiler crash: none observed.
- False unsupported diagnostic: none observed in this checked-in selector.
- Stable positives: descriptor roundtrip chains, row/column-permuted
  descriptor roundtrips, plain MMAv5 indexed and accumulator-subslice rows,
  and scaled-MMAv5 indexed/accumulator/scale-descriptor rows.

Recommended next discovery slice: continue temporary adversarial probes that
combine these otherwise-green descriptor/control rows with dynamic memdesc
selection and multiple MMAv5 operations in one kernel.
