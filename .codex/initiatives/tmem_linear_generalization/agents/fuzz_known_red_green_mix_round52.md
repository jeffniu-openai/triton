# Round 52 Local Known-Red/Green Runtime Mix

Date: 2026-04-21 15:03 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Run a compact same-process runtime mix with one expected known-red M64
`ld.red` row followed by adjacent green rows. This verifies the known failure
signature and checks that it does not contaminate later positive compiles in
the same pytest process.

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
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_splitn_linear_layout[m64_64x32-False-propagate_nan0-min]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive[f32-torch_dtype0]'
```

Result:

```text
1 failed, 2 passed
```

The failed row is existing `FZ-20260421-0012`: M64 f32 `tcgen05.ld.red`
destination-layout planner gap for non-identity row basis. The two adjacent
positive rows passed after that failure in the same pytest process.

## Classification

No new independent `FZ-*` bucket and no process-contamination signal was
observed.

Backend repair remains deferred per the discovery-only campaign.
