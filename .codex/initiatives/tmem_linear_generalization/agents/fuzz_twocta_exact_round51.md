# Round 51 Local Two-CTA Exact Probe

Date: 2026-04-21 15:00 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Run a compact exact runtime probe over two-CTA and multi-CTA related rows:

- clean two-CTA layout in a four-CTA context diagnostic;
- two-CTA `warpx2::01_23` dense shared rematerialization;
- two-CTA indexed-accumulator MMAv5;
- two-CTA scales copy.

## Required Build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Runtime Command

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_dense_shared_rematerializes[f32-torch_dtype0]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_indexed_acc_view[f16-linear-32-32-True]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_twocta_direct_copy'
```

Result:

```text
4 passed
```

## Classification

No clean-diagnostic drift, compiler crash, runtime miscompile, or new
independent `FZ-*` bucket was observed.

Backend repair remains deferred per the discovery-only campaign.
