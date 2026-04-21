# Round 54 Local `test_core.py` TMEM Smoke

Date: 2026-04-21
Branch: `codex/tmem`
Checkpoint base: `ac0791c80 Record Round 54 local splitn x1 subword lane`

This local lane ran a small exact-nodeid smoke slice from
`python/test/gluon/test_core.py`, independent of the runtime-matrix helper
surface. The selected rows covered multicast commit, two-CTA linear accumulator
MMAv5, no-scale copy, descriptor-view linear runtime, `ld.red`, and scaled
copy with linear accumulator layout.

No backend fixes were attempted.

## Preflight

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Collection Probe

```bash
PYTHONPATH=.:./python \
pytest --collect-only -q \
  python/test/gluon/test_core.py \
  -k 'tcgen05_mma_multicast_commit or tmem_copy_no_scales_matrix or tmem_linear_runtime_views or tmem_reduction_linear_legacy_block_equiv_layout or mma_scaled_tcgen05_copy_linear_acc'
```

Result:

```text
64/18114 tests collected (18050 deselected) in 3.74s
```

## Exact Smoke

```bash
CUDA_VISIBLE_DEVICES=1 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
PYTHONPATH=.:./python \
pytest -q -s --tb=short \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[False-ctas_per_cga1]' \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit_twocta_linear_acc[linear]' \
  'python/test/gluon/test_core.py::test_tmem_copy_no_scales_matrix[128-128-32-32]' \
  'python/test/gluon/test_core.py::test_tmem_linear_runtime_views[index_reshape_bitcast_128_mixed_32x32b-tmem_linear_runtime_view_kernel_b-layout7-reinterpret_layout7-32x32b-4]' \
  'python/test/gluon/test_core.py::test_tmem_reduction_linear_legacy_block_equiv_layout[propagate_nan0-False-min]' \
  'python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy_linear_acc[1-mxfp8-mxfp8]'
```

Result:

```text
6 passed in 5.08s
```

## Classification

No compiler crash, verifier drift, false unsupported diagnostic, opcode
absence, runtime miscompile, hang, or new independent `FZ-*` bucket was found
in this smoke lane.
