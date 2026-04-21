# Round 14 Local Scaled-MMAv5 Descriptor/Accumulator Selector

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler fixes were attempted.

## Required rebuild

```bash
make -j8
```

Result: `ninja: no work to do`.

## Selector

Collect-only:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and (descriptor or scale_descriptor or bscale or acc_subslice_view or indexed_acc) and not reports'
```

Result: `105/1615` selected.

Runtime pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  --store-durations --durations-path /tmp/tmem_local_r14_scaled_descriptor_acc_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and (descriptor or scale_descriptor or bscale or acc_subslice_view or indexed_acc) and not reports'
```

Results:

- group 1/GPU 0: `27 passed, 1588 deselected`;
- group 2/GPU 1: `27 passed, 1588 deselected`;
- group 3/GPU 2: `27 passed, 1588 deselected`;
- group 4/GPU 3: `24 passed, 1591 deselected`.

Aggregate: `105 passed`.

## Classification

No new bucket from the checked-in selector. Indexed accumulator views,
accumulator subslice views, use-acc variants, two-CTA scaled accumulator
subslice rows, shared scale descriptor-view auto-copy rows, and the checked-in
B-scale descriptor-view/rematerialization controls remained green.

This is a baseline next to Lane AG's temporary `FZ-20260421-0013`
minimization: the existing checked-in scaled-MMAv5 descriptor and accumulator
coverage does not currently expose the report-only B-scale descriptor-view
miscompile found by the temporary Round 14 probes.
