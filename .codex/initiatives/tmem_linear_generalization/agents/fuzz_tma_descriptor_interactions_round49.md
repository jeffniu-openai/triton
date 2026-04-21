# Round 49: TMA descriptor and TMEM consumer interactions

Date: 2026-04-21
Branch: `codex/tmem`

This lane adversarially tested interactions between TMA descriptors, shared
memory inputs, mbarrier waits, multicast, gather/scatter, and TMEM consumers.
The focus was discovery and classification only. No backend fixes and no
checked-in test changes were made.

## Summary

No new `FZ-*` bucket is needed.

Checked-in coverage stayed green across the requested surfaces:

- TMA-fed MMAv5, shared-input multicast, gather/shared multicast, and mbarrier
  waits in `test_core.py`: `184 passed, 48 skipped`.
- Runtime-matrix two-CTA TMA descriptor consumers, B-transposed descriptors,
  descriptor-view accumulator/use-acc rows, and clean shared-transpose
  diagnostics: `17 passed`.
- Runtime-matrix scaled descriptor-view consumers, including b-scale descriptor
  views, padded descriptor views, narrow clean diagnostics, and shared-scale
  descriptor-view auto-copy rows: `61 passed`.
- Basic TMA descriptor/mbarrier and scaled-copy controls: `8 passed`.

Classification against existing buckets:

- `FZ-20260421-0010` was not reproduced. High-CGA/multicast ownership controls
  in the checked-in shared-input and TMA/gather tests passed or skipped for
  existing environment/resource reasons.
- `FZ-20260421-0014` was not reproduced by checked-in proxy/mbarrier coverage.
  The tested mbarrier wait sequences, multicast TMA loads, gather/scatter TMA
  paths, and direct scaled-MMAv5 multicast barrier path stayed green.
- Descriptor-consumer buckets such as `FZ-20260421-0007`,
  `FZ-20260421-0013`, and `FZ-20260421-0015` did not show new leakage in this
  checked-in slice. Adjacent descriptor-view positive and clean-negative rows
  kept their expected behavior.

## Commands And Results

Rebuild:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

Core TMA/shared-input selector collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tma_gather_scatter_multi_cta or tma_multicast_copy or mma_scaled_direct_multicast_barrier'
```

Result:

```text
232/18114 tests collected (17882 deselected)
```

Core TMA/shared-input execution, split across all four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tma_gather_scatter_multi_cta or tma_multicast_copy or mma_scaled_direct_multicast_barrier'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tma_gather_scatter_multi_cta or tma_multicast_copy or mma_scaled_direct_multicast_barrier'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tma_gather_scatter_multi_cta or tma_multicast_copy or mma_scaled_direct_multicast_barrier'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tma_gather_scatter_multi_cta or tma_multicast_copy or mma_scaled_direct_multicast_barrier'
```

Results:

```text
group 1: 49 passed, 9 skipped, 18056 deselected
group 2: 49 passed, 9 skipped, 18056 deselected
group 3: 44 passed, 14 skipped, 18056 deselected
group 4: 42 passed, 16 skipped, 18056 deselected
total:   184 passed, 48 skipped
```

Runtime-matrix TMA descriptor interaction selector collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'twocta_tma or b_transposed_descriptor or shared_transpose or shared_scale_descriptor_view_auto_tmem_copy or acc_subslice_k'
```

Result:

```text
17/1615 tests collected (1598 deselected)
```

Runtime-matrix TMA descriptor interaction execution:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'twocta_tma or b_transposed_descriptor or shared_transpose or shared_scale_descriptor_view_auto_tmem_copy or acc_subslice_k'
```

Result:

```text
17 passed, 1598 deselected
```

Runtime-matrix descriptor-consumer expansion collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'bscale_descriptor_view or acc_tile_permuted or acc_n16_bscale or acc_subslice_k or shared_scale_descriptor_view'
```

Result:

```text
61/1615 tests collected (1554 deselected)
```

Runtime-matrix descriptor-consumer expansion execution:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'bscale_descriptor_view or acc_tile_permuted or acc_n16_bscale or acc_subslice_k or shared_scale_descriptor_view'
```

Result:

```text
61 passed, 1554 deselected
```

Basic TMA descriptor/mbarrier and scaled-copy controls:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short \
  python/test/gluon/test_core.py::test_tma \
  'python/test/gluon/test_core.py::test_tma_im2col[32-32-32]' \
  python/test/gluon/test_core.py::test_gluon_tma_round_f32_to_tf32 \
  python/test/gluon/test_core.py::test_async_copy_mbarrier \
  python/test/gluon/test_core.py::test_device_tma_load \
  python/test/gluon/test_core.py::test_device_tma_store \
  'python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy_linear_acc[2-mxfp8-mxfp8]' \
  'python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy_linear_acc[2-mxfp8-mxfp4]'
```

Result:

```text
8 passed
```

## Notes

The strongest checked-in adversarial coverage remains
`test_tma_mma_shared_inputs`: it composes TMA descriptor loads, optional
Blackwell gather, shared-input multicast, repeated mbarrier phases, MMAv5 TMEM
accumulators, and optional scatter of accumulator results back through shared
memory. The full selected matrix stayed green, so this round did not expose a
new TMA-to-TMEM consumer gap.

