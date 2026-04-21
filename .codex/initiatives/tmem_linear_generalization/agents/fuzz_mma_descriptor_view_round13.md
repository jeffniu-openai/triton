# Round 13 Local: MMAv5 Descriptor/View Runtime Sweep

## Scope

This lane targeted checked-in MMAv5 runtime-matrix rows adjacent to the
FPSAN-specific `FZ-20260421-0011` issue without enabling FPSAN. It covered
plain MMAv5 accumulator descriptor views, lifted `linear_unit_parent`
accumulator parents, `use_acc` true/false rows, TMA-backed transposed-B
descriptor inputs, and a scaled-MMAv5 indexed narrow accumulator-view control.

This lane did not edit backend/compiler code.

## Commands

Required rebuild:

```bash
make -j8
```

Result: no-op success.

Collect-only:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_twocta_indexed_acc_view or mma_twocta_tma_tf32_b_transposed_descriptor or mma_scaled_shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc'
```

Result: `37/1615` rows collected.

Runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 --store-durations --durations-path /tmp/tmem_local_r13_mma_descriptor_view_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_twocta_indexed_acc_view or mma_twocta_tma_tf32_b_transposed_descriptor or mma_scaled_shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 --store-durations --durations-path /tmp/tmem_local_r13_mma_descriptor_view_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_twocta_indexed_acc_view or mma_twocta_tma_tf32_b_transposed_descriptor or mma_scaled_shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 --store-durations --durations-path /tmp/tmem_local_r13_mma_descriptor_view_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_twocta_indexed_acc_view or mma_twocta_tma_tf32_b_transposed_descriptor or mma_scaled_shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 --store-durations --durations-path /tmp/tmem_local_r13_mma_descriptor_view_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_twocta_indexed_acc_view or mma_twocta_tma_tf32_b_transposed_descriptor or mma_scaled_shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc'
```

## Results

- Group 1/GPU 0: `10 passed, 1605 deselected in 4.41s`
- Group 2/GPU 1: `10 passed, 1605 deselected in 4.38s`
- Group 3/GPU 2: `10 passed, 1605 deselected in 7.40s`
- Group 4/GPU 3: `7 passed, 1608 deselected in 4.65s`

Aggregate: `37 passed`, no failures.

## Classification

No new `FZ-*` candidate.

This sweep strengthens the `FZ-20260421-0011` isolation:

- non-FPSAN MMAv5 indexed accumulator views pass for legacy, linear, and
  lifted `linear_unit_parent` parents;
- `use_acc=True` and `use_acc=False` controls pass in the checked-in matrix;
- TMA-backed transposed-B descriptor rows pass for both linear and legacy
  accumulator layouts;
- the scaled-MMAv5 indexed narrow accumulator-view control passes, so the
  FPSAN plain-MMAv5 runtime-selector mismatch does not generalize to this
  scaled checked-in neighbor.

`FZ-20260421-0011` remains FPSAN-specific runtime outer descriptor selection
feeding plain MMAv5. This lane does not warrant a new bucket.
