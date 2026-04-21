# Round 14 Local Broad MMAv5 Descriptor Selector

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
  -k '(mma and (descriptor or runtime_selector or indexed_acc or acc_subslice_view or scale_descriptor)) and not reports'
```

Result: `253/1615` selected.

Runtime pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  --store-durations --durations-path /tmp/tmem_local_r14_mma_descriptor_broad_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mma and (descriptor or runtime_selector or indexed_acc or acc_subslice_view or scale_descriptor)) and not reports'
```

Results:

- group 1/GPU 0: `50 passed, 14 skipped, 1551 deselected`;
- group 2/GPU 1: `64 passed, 1551 deselected`;
- group 3/GPU 2: `64 passed, 1551 deselected`;
- group 4/GPU 3: `61 passed, 1554 deselected`.

Aggregate: `239 passed, 14 skipped`.

## Classification

No new bucket. The broad checked-in MMAv5 descriptor surface remained green,
including direct `ld/st` two-CTA MMAv5-layout descriptor controls selected by
the keyword expression, M64 and two-CTA accumulator subslice views, indexed
accumulator views, TMA transposed-B descriptor rows, and the scaled-MMAv5
descriptor/accumulator subset.

This baseline complements Lane AG's temporary `FZ-20260421-0013`
minimization: checked-in descriptor/indexed-accumulator rows pass, while the
temporary scale descriptor-view probe exposes the report-only scale-fragment
mapping wrong-result bucket.
