# Round 52: Negative Boundary Inventory

Date: 2026-04-21

Branch: `codex/tmem`

Scope: catalog-only audit of checked-in TMEM clean-negative and known-red
boundaries across the runtime matrix, structural fuzzer, and lit. No backend
code or checked-in tests were modified.

## Objective

Separate true hardware/API clean boundaries from backend gaps, then look for:

- clean diagnostics that drifted into generic crashes;
- expected-failure rows that unexpectedly pass;
- unsupported/verifier checks that are too strict for cases the backend should
  support;
- checked-in lit failures that map to existing or new `FZ-*` buckets.

## Required Build

Command:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Runtime Matrix: Clean Boundaries

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k "(unsupported_variants_report_clean_unsupported or reports_clean_error or reports_clean_unsupported or block_descriptor_reports_clean_error or block_layout_reports_clean_error or scaled_acc_n16 or mixed_fp4a_reports_clean_unsupported or mma_i8_reports_clean_error or mma_m64_i8_reports_clean_error or twocta_i8_reports_clean_error or cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error or cp_no_scales_linear_exotic_reports_clean_unsupported or reports_backend_error or resource)" \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Collection: `154/1615` tests collected, `1461` deselected.

Split run pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short -rxXs \
  --splits 4 --group <group> \
  -k "(unsupported_variants_report_clean_unsupported or reports_clean_error or reports_clean_unsupported or block_descriptor_reports_clean_error or block_layout_reports_clean_error or scaled_acc_n16 or mixed_fp4a_reports_clean_unsupported or mma_i8_reports_clean_error or mma_m64_i8_reports_clean_error or twocta_i8_reports_clean_error or cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error or cp_no_scales_linear_exotic_reports_clean_unsupported or reports_backend_error or resource)" \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Split results:

- group 1 on GPU0: `39 passed, 1576 deselected in 9.78s`
- group 2 on GPU1: `39 passed, 1576 deselected in 4.73s`
- group 3 on GPU2: `39 passed, 1576 deselected in 4.30s`
- group 4 on GPU3: `37 passed, 1578 deselected in 11.10s`

Aggregate: `154 passed`.

Classification: clean hardware/API/resource boundaries are stable. This bucket
includes subword and x1 unsupported variants, block and blocked-layout errors,
TMEM copy descriptor-plan boundaries, 2CTA-layout-in-4CTA-context mismatch,
linear exotic layouts, i8 MMAv5 unsupported rows, mixed FP4A scaled-MMAv5
unsupported rows, M64 i8 unsupported rows, scaled accumulator `N=16`
diagnostics, the scaled accumulator `M=64` backend diagnostic row, and large
`ld.red` resource-boundary rows. No generic crash, unexpected failure, XPASS,
or new clean-negative gap was observed.

## Runtime Matrix: Known Backend Gaps

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k "(ld_red_m64_rowcol_permuted_default_layout or ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn or ld_red_descriptor_chain_n_sweep_explicit_variants or ld_red_explicit_n_sweep_variants or reports_backend_error)" \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Collection: `26/1615` tests collected, `1589` deselected.

Split run pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short -rxXs \
  --splits 4 --group <group> \
  -k "(ld_red_m64_rowcol_permuted_default_layout or ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn or ld_red_descriptor_chain_n_sweep_explicit_variants or ld_red_explicit_n_sweep_variants or reports_backend_error)" \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Split results:

- group 1 on GPU0: `5 failed, 2 passed, 1608 deselected in 4.93s`
- group 2 on GPU1: `1 failed, 6 passed, 1608 deselected in 4.18s`
- group 3 on GPU2: `7 passed, 1608 deselected in 3.94s`
- group 4 on GPU3: `5 passed, 1610 deselected in 4.05s`

Aggregate: `20 passed, 6 failed`.

Failed nodeids:

```text
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]
```

Failure signature:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
Failed to lower TMEM load/store: unsupported dst layout
```

Classification: all six failures are existing `FZ-20260421-0012`, the M64
f32 `tcgen05.ld.red` destination-layout planner gap for non-identity row
basis. The added backend-error diagnostic row passed, so no new known-red
signature or clean-boundary drift appeared.

## Structural Fuzzer

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Collection: `33 tests collected`.

Run command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short -rxXs \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result:

```text
9 passed, 24 xfailed in 9.69s
```

Expected-failure classifications:

- `FZ-20260421-0001`: dynamic/generic TMEM `memdesc_index` reaches late LLVM
  conversion as an illegal op.
- `FZ-20260421-0002`: helper-returned or dynamic-if TMEM view wrong results.
- `FZ-20260421-0003`: descriptor-view `ld/st` packet-order wrong results.
- `FZ-20260421-0004`: descriptor-view or indexed `ld.red` opcode loss /
  software reduction.
- `FZ-20260421-0005`: lifted 256-row parent allocator assertion instead of a
  clean diagnostic.
- `FZ-20260421-0006`: `ld.red` transpose/slice false unsupported.
- `FZ-20260421-0007`: scaled-MMAv5 dynamic accumulator subslice wrong result.
- `FZ-20260421-0008`: 2CTA indexed `ld.red` row/column optimizer abort.
- `FZ-20260421-0009`: 1CTA direct indexed `ld.red` allocator assertion.
- Legacy `R5-C`: loop-carried TMEM view auto-layout inference failure. This is
  still an expected structural sentinel but is not currently assigned an
  `FZ-*` bucket.

Classification: no XPASS drift. The xfail rows are still backend gaps, not
true clean hardware/API boundaries. No new structural fuzzer failure or
unexpected clean-boundary change was found.

## Lit: Focused TMEM Boundary Set

Command:

```bash
BUILD_DIR=$(PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
cd "$BUILD_DIR" && ninja triton-opt && lit -v \
  test/TritonNvidiaGPU/tmem_layouts.mlir \
  test/TritonNvidiaGPU/interleave_tmem.mlir \
  test/TritonGPU/memdesc-subview-split.mlir \
  test/TritonGPU/proxy_fence_insertion.mlir \
  test/TritonGPU/promote-lhs-to-tmem.mlir \
  test/TritonGPU/hoist-tmem-alloc.mlir \
  test/TritonGPU/nvidia-fpsan.mlir \
  test/Conversion/relayout_tritongpu.mlir
```

Result:

```text
Total Discovered Tests: 8
  Passed: 7 (87.50%)
  Failed: 1 (12.50%)
```

Failing file: `Conversion/relayout_tritongpu.mlir`.

Failure signature:

```text
dyn_cast on a non-existent value
mlir::triton::gpu::TritonGPUDialect::toLinearLayout(...)
mlir::triton::nvidia_gpu::computeTMemLdStEncodingInfoImpl(...)
mlir::triton::nvidia_gpu::verifyTMEMOperand(...)
```

Classification: existing `FZ-20260421-0016`, the unencoded or
non-distributed TMEM operand verifier/assertion boundary.

## Lit: Broad TMEM-Bearing Inventory

File discovery command:

```bash
rg -l "ttng\\.tmem|tensor_memory|TMEM|tmem" \
  test/Conversion test/TritonNvidiaGPU test/TritonGPU | rg -v '\.nyi$' | sort
```

Run command:

```bash
BUILD_DIR=$(PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
mapfile -t LIT_FILES < <(rg -l "ttng\\.tmem|tensor_memory|TMEM|tmem" \
  test/Conversion test/TritonNvidiaGPU test/TritonGPU | rg -v '\.nyi$' | sort)
cd "$BUILD_DIR" && lit -v "${LIT_FILES[@]}"
```

Result:

```text
Total Discovered Tests: 36
  Passed: 34 (94.44%)
  Failed:  2 (5.56%)
```

Failed files:

- `Conversion/relayout_tritongpu.mlir`
- `TritonGPU/consan.mlir`

`Conversion/relayout_tritongpu.mlir` matches the focused-lit
`FZ-20260421-0016` signature.

`TritonGPU/consan.mlir` was rerun in isolation:

```bash
BUILD_DIR=$(PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
cd "$BUILD_DIR" && lit -v test/TritonGPU/consan.mlir
```

Result:

```text
Total Discovered Tests: 1
  Failed: 1 (100.00%)
```

Failure signature:

```text
dyn_cast on a non-existent value
mlir::triton::gpu::TritonGPUDialect::toLinearLayout(...)
mlir::triton::nvidia_gpu::computeTMemLdStEncodingInfoImpl(...)
mlir::triton::nvidia_gpu::verifyTMEMOperand(...)
mlir::triton::nvidia_gpu::TMEMLoadOp::verify()
FileCheck error: '<stdin>' is empty.
```

Classification: additional checked-in lit evidence for existing
`FZ-20260421-0016`. The relevant `consan.mlir` chunk has a TMEM load returning
`tensor<128x128xf32>` without a distributed encoding, so TMEM operand
verification reaches `toLinearEncoding`/`toLinearLayout` and asserts. This is
not a true hardware/API clean boundary; it should eventually become either an
accepted pre-relayout shape or a typed verifier diagnostic. It does not need a
new `FZ-*` bucket because the stack and abstraction mismatch are the same as
the existing relayout/unencoded-TMEM-operand bucket.

## Classification Summary

True clean hardware/API/resource boundaries:

- Runtime-matrix clean diagnostics and clean unsupported rows: `154 passed`.
- Includes copy layout/API constraints, 2CTA/4CTA ownership mismatch,
  unsupported i8/M64/mixed-FP4A MMAv5 variants, scaled accumulator narrow-N
  diagnostics, block/blocked layout errors, and large resource-boundary
  reduction rows.

Known backend gaps revalidated:

- `FZ-20260421-0001` through `FZ-20260421-0009`: structural fuzzer xfail rows,
  all stable with no XPASS.
- `FZ-20260421-0012`: six runtime M64 `ld.red` non-identity row-basis planner
  failures.
- `FZ-20260421-0016`: `Conversion/relayout_tritongpu.mlir` and
  `TritonGPU/consan.mlir` unencoded/non-distributed TMEM operand verifier
  assertion.
- Legacy `R5-C`: loop-carried TMEM view auto-layout inference xfail remains
  stable and unbucketed.

No new `FZ-*` bucket is proposed.

New inventory evidence:

- `TritonGPU/consan.mlir` is a checked-in lit failure under existing
  `FZ-20260421-0016`. It should remain on the repair queue when the campaign
  pivots from cataloging to backend fixes.

No new clean-negative gap was found. The only crashes observed are already
classified backend gaps, not true hardware/API boundaries.
