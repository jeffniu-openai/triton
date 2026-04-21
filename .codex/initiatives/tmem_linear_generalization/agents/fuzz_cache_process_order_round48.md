# Round 48: Cache and Process-Order Stability

Date: 2026-04-21 14:45 UTC

Branch/HEAD: `codex/tmem` at `ac874ace1459`

Scope: stress TMEM-bearing tests for cache-key omissions, pytest-process
ordering effects, device/process contamination after expected compiler failures,
async compile/global-state reuse, and known `FZ-*` signature drift. This round
cataloged behavior only; it did not modify backend code or checked-in tests.

## Rebuild

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Row Selection

The probe intentionally mixed:

- checked-in runtime positives:
  - `cp_no_scales_linear_indexed_view[128x128b_f32...]`
  - `cp_no_scales_warpx2_01_23_candidate_positive[f32]`
  - `mma_scaled_acc_tile_permuted_32_bscale_view_extra_user_rematerializes`
  - rank-5 descriptor load/store and two-CTA dense-shared copy positives
- checked-in structural expected failures/positives:
  - generic dynamic-index load-only expected `FZ-20260421-0001`
  - `ld.red` structural positives and xfail sentinels
- disposable Round 45 dynamic copy sentinel:
  - `/tmp/tmem_copy_dynamic_descriptor_round40_probe.py`
  - linear branch-selected descriptors expected `FZ-20260421-0001`
  - warpx2 branch-selected descriptors expected positive
- M64 `ld.red` row-basis failure:
  - `ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]`
  - classified as existing `FZ-20260421-0012`

Collection checks:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_m64_splitn_linear_layout or ld_red_m64_rowcol_permuted_default_layout or cp_no_scales_linear_indexed_view or cp_no_scales_warpx2_01_23_candidate_positive or mma_scaled_acc_tile_permuted_32_bscale_view_extra_user_rematerializes'
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_structural_fuzzer.py -k 'copy_scales or generic_pass_dynamic_index_load_only or ldst_view_roundtrip or ldred'
PYTHONPATH=.:./python:./python/test/gluon pytest --collect-only -q /tmp/tmem_copy_dynamic_descriptor_round40_probe.py
```

Results:

```text
runtime matrix selector: 34/1615 tests collected
structural selector: 18/33 tests collected
dynamic copy disposable probe: 8 tests collected
```

## Same-Process Mixed Guardrail

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_indexed_view[128x128b_f32-torch_dtype0-128-4-smem_layout0-1-tcgen05.cp.cta_group::1.128x128b]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive[f32-torch_dtype0]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_bscale_view_extra_user_rematerializes' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-1cta]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_dynamic_index_load_only[generic-pass-dynamic-index-load-only-128x32]'
```

Result:

```text
4 passed, 1 xfailed in 3.38s
```

The xfail emitted the expected `FZ-20260421-0001` signature:
`ttg.memdesc_index` survived until `ConvertTritonGPUToLLVM` and was rejected as
an illegal op. The preceding runtime/copy/scaled positives passed in the same
pytest process.

## Dynamic Copy Failure Mixed With Positives

Same process, fail/pass/fail/pass ordering:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor[0]' \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[01_23-0-index]' \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor[1]' \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[02_13-1-slice_index]'
```

Result:

```text
2 failed, 2 passed in 5.14s
```

Both failures had the expected `FZ-20260421-0001` signature: branch-selected
linear `ttng.tmem_copy` descriptors leave a `ttg.memdesc_index` illegal op for
LLVM conversion. The warpx2 rows passed between and after those failures.

Fresh pytest process repeats with the same stable cache:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor[0]'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[01_23-0-index]'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor[1]'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[02_13-1-slice_index]'
```

Results:

```text
linear[0]: 1 failed in 3.89s
warpx2[01_23-0-index]: 1 passed in 3.09s
linear[1]: 1 failed in 3.25s
warpx2[02_13-1-slice_index]: 1 passed in 4.63s
```

Reverse same-process ordering, pass/fail/pass:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[01_23-0-index]' \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor[0]' \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[02_13-1-slice_index]'
```

Result:

```text
1 failed, 2 passed in 5.65s
```

Classification: stable existing `FZ-20260421-0001`. No evidence that the failed
linear compile poisons later warpx2 copy compiles, and no evidence that a
successful warpx2 compile masks the linear failure.

## Warm-Cache Runtime Positive Repeats

Command:

```bash
for i in 1 2; do
  CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short \
    'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive[f32-torch_dtype0]' \
    'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_rank5_unit_parent_n256_roundtrip[f32-torch_dtype0-single_identity-single-identity-128-1-auto-32x32b.x64.b32]' \
    'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_dense_shared_rematerializes[f32-torch_dtype0]'
done
```

Result:

```text
3 passed in 2.71s
3 passed in 3.99s
```

No warm-cache drift appeared for copy, descriptor-chain load/store, or two-CTA
dense-shared rematerialization positives.

## M64 `ld.red` Failure Ordering

Failure followed by positives in one process:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_splitn_linear_layout[m64_64x32-False-propagate_nan0-min]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[col_reverse_n32-max]'
```

Result:

```text
1 failed, 2 passed in 4.33s
```

The failure matched existing `FZ-20260421-0012`: `ttng.tmem_load` failed to
compute TMEM encoding info for reduction, with `unsupported dst layout` details
for a row-basis M64 destination layout. The following split-N and column-only
M64 rows passed in the same process.

Fresh-process pass/fail/pass repeat:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_splitn_linear_layout[m64_64x32-False-propagate_nan0-min]'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_splitn_linear_layout[m64_64x32-False-propagate_nan0-min]'
```

Results:

```text
split-N before: 1 passed in 3.87s
row_reverse_n32-min: 1 failed in 3.30s
split-N after: 1 passed in 2.70s
```

Classification: stable existing `FZ-20260421-0012`. No process-order or
warm-cache signature drift was observed.

## Structural Warm-Cache Repeats

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-view-128x64]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-chain1-64x32-min]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred_twocta_rowcol_optimizer_crash'
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-view-128x64]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-chain1-64x32-min]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred_twocta_rowcol_optimizer_crash'
```

Result:

```text
2 passed, 2 xfailed in 5.05s
2 passed, 2 xfailed in 4.63s
```

Additional structural pass/xfail repeat:

```bash
for i in 1 2; do
  CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short \
    'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_dynamic_index_load_only[generic-pass-dynamic-index-load-only-128x32]' \
    'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-direct-128x64]' \
    'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-2cta]'
done
```

Result:

```text
2 passed, 1 xfailed in 3.94s
2 passed, 1 xfailed in 3.00s
```

The xfail signatures remained the existing illegal `ttg.memdesc_index` and
known `ld.red` structural buckets; no XPASS drift or unexpected failure appeared.

## Classification

No new `FZ-*` bucket is needed from this cache/process-order round.

Observed failures were stable existing buckets:

- `FZ-20260421-0001`: dynamic/control-flow-carried TMEM memdesc values reach
  LLVM conversion as illegal `ttg.memdesc_index`, including the dynamic linear
  `ttng.tmem_copy` sentinel and structural generic-pass dynamic-index sentinel.
- `FZ-20260421-0012`: M64 f32 `tcgen05.ld.red` destination-layout planner gap
  for row-basis layouts, reported as `unsupported dst layout`.

No evidence found for:

- missing cache-key inputs;
- a failed compile contaminating later positive compiles in the same pytest
  process;
- a positive compile masking a later known failure;
- device/process contamination across stable per-GPU cache directories;
- async compile/global-state reuse drift;
- XPASS drift in structural expected failures.

