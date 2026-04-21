# Round 50 Mixed Exact Runtime Smoke

Date: 2026-04-21 14:51 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Run a compact exact-node runtime smoke over four recently important positive
surfaces:

- `warpx2::01_23` static copy;
- scaled-MMAv5 B-scale descriptor-view extra-user rematerialization;
- rank-5 unit-parent descriptor-view `ld/st`;
- two-CTA TMA B-transposed TF32 MMAv5 descriptor path.

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

## Nodeid Correction

The first attempt used a stale nonexistent parametrized nodeid:

```text
test_tmem_runtime_matrix_mma_twocta_tma_tf32_b_transposed_descriptor[legacy-32-32]
```

Pytest reported `no tests ran`. A collection query showed the valid nearby
nodeids, including:

```text
test_tmem_runtime_matrix_mma_twocta_tma_tf32_b_transposed_descriptor[linear-32-32]
```

## Runtime Command

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive[f32-torch_dtype0]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_bscale_view_extra_user_rematerializes' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_rank5_unit_parent_n256_roundtrip[f32-torch_dtype0-single_identity-single-identity-128-1-auto-32x32b.x64.b32]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_tma_tf32_b_transposed_descriptor[linear-32-32]'
```

Result:

```text
4 passed
```

## Classification

No compiler crash, runtime miscompile, nodeid drift after correction, or new
independent `FZ-*` bucket was observed.

Backend repair remains deferred per the discovery-only campaign.
