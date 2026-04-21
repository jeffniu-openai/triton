# Round 45 Mixed Exact Smoke Guardrail

Date: 2026-04-21 14:33 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Surface

This lane ran a compact exact-nodeid smoke over three recently active positive
surfaces:

- scaled-MMAv5 B-scale descriptor-view extra-user rematerialization;
- rank-5 descriptor-view `ld/st` roundtrip;
- two-CTA `warpx2::01_23` dense-shared copy rematerialization.

## Commands

Required rebuild:

```bash
make -j8
```

Result: build-tree ninja reported `no work to do`.

Runtime:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_bscale_view_extra_user_rematerializes' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_rank5_unit_parent_n256_roundtrip[f32-torch_dtype0-single_identity-single-identity-128-1-auto-32x32b.x64.b32]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_dense_shared_rematerializes[f32-torch_dtype0]'
```

Result: `3 passed`.

## Classification

No compiler crash, false unsupported diagnostic, runtime miscompile, or
cache/process-order drift was observed in this exact mixed smoke.
