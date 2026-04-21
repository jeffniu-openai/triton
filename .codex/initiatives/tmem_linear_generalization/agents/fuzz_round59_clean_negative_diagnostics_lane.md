# Round 59 Lane B: TMEM clean-negative diagnostics

Date: 2026-04-21
Branch: `codex/tmem`
HEAD at start: `8ee469bd0`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code
was edited.

## Scope

This lane focused on clean negative diagnostics around invalid TMEM layouts,
copy, `ld.red`, plain MMAv5, and scaled-MMAv5 rows not already covered by the
Round 57 lit bridge (`TritonNvidiaGPU/invalid.mlir`,
`TritonNvidiaGPU/ops.mlir`, and `TritonNvidiaGPU/tmem_layouts.mlir`).

Targets:

- checked-in pytest `clean_error`, `reports_clean`, `clean_unsupported`, and
  `unsupported_variants` selectors in `test_tmem_runtime_matrix.py`;
- invalid copy layout diagnostics, including `4x256b`, `warpx2::02_13`,
  subword dtype, exotic linear, row/column-permuted, and CTA-count mismatch
  rows;
- `ld.red` non-f32/scales rows that should use software reduction rather than
  producing an unsupported or crash diagnostic;
- plain-MMAv5 i8/TMA/exotic-layout clean errors;
- scaled-MMAv5 mixed-fp4A and N16 accumulator clean unsupported/error rows;
- non-Round-57 lit conversion/pass-manager coverage for TMEM lowering and
  allocation paths.

## Required build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Runtime-matrix clean-negative selector

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(clean_error or reports_clean or clean_unsupported or unsupported_variants or block_descriptor_reports_clean_error or block_layout_reports_clean_error or mma_i8_reports_clean_error or twocta_i8_reports_clean_error or cp_no_scales_linear_exotic_reports_clean_unsupported or cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error or cp_scales_tmem_descriptor_view_reports_clean_unsupported or ld_red_non_f32_contract or ld_red_scales_uses_software_reduce)'
```

Result: `200/1615` collected.

Split-4 runtime commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(clean_error or reports_clean or clean_unsupported or unsupported_variants or block_descriptor_reports_clean_error or block_layout_reports_clean_error or mma_i8_reports_clean_error or twocta_i8_reports_clean_error or cp_no_scales_linear_exotic_reports_clean_unsupported or cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error or cp_scales_tmem_descriptor_view_reports_clean_unsupported or ld_red_non_f32_contract or ld_red_scales_uses_software_reduce)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k '(clean_error or reports_clean or clean_unsupported or unsupported_variants or block_descriptor_reports_clean_error or block_layout_reports_clean_error or mma_i8_reports_clean_error or twocta_i8_reports_clean_error or cp_no_scales_linear_exotic_reports_clean_unsupported or cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error or cp_scales_tmem_descriptor_view_reports_clean_unsupported or ld_red_non_f32_contract or ld_red_scales_uses_software_reduce)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k '(clean_error or reports_clean or clean_unsupported or unsupported_variants or block_descriptor_reports_clean_error or block_layout_reports_clean_error or mma_i8_reports_clean_error or twocta_i8_reports_clean_error or cp_no_scales_linear_exotic_reports_clean_unsupported or cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error or cp_scales_tmem_descriptor_view_reports_clean_unsupported or ld_red_non_f32_contract or ld_red_scales_uses_software_reduce)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k '(clean_error or reports_clean or clean_unsupported or unsupported_variants or block_descriptor_reports_clean_error or block_layout_reports_clean_error or mma_i8_reports_clean_error or twocta_i8_reports_clean_error or cp_no_scales_linear_exotic_reports_clean_unsupported or cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error or cp_scales_tmem_descriptor_view_reports_clean_unsupported or ld_red_non_f32_contract or ld_red_scales_uses_software_reduce)'
```

Results:

- group 1: `50 passed, 1565 deselected`
- group 2: `50 passed, 1565 deselected`
- group 3: `50 passed, 1565 deselected`
- group 4: `50 passed, 1565 deselected`
- aggregate: `200 passed`

Classification:

- No false unsupported diagnostic, assertion text regression, pass-manager
  failure, vague/incorrect diagnostic drift, unexpected failure, or new
  independent `FZ-*`.
- Copy clean negatives stayed clean across direct copy, scales-copy descriptor
  views, subword dtype restrictions, `4x256b`, `warpx2::02_13`, exotic linear
  layouts, row/column-permuted layouts, and CTA-count mismatch rows.
- `ld.red` non-f32 and scale rows stayed green through software-reduce
  fallback; they did not produce spurious clean unsupported errors.
- Plain MMAv5 i8/TMA/exotic-layout diagnostics and scaled-MMAv5 mixed-fp4A/N16
  diagnostics stayed clean and specific.

## Structural-fuzzer overlap collection

Collection only, used to confirm this lane was not re-running the Round 57
structural/lit bridge xfail set as the main signal:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'ldred or scaled_mma or descriptor_view or generic_pass'
```

Result: `26/33` collected. These are the known structural sentinels already
tracked by `FZ-20260421-0001`, `FZ-20260421-0002`,
`FZ-20260421-0003`, `FZ-20260421-0004`, `FZ-20260421-0006`,
`FZ-20260421-0007`, `FZ-20260421-0008`, `FZ-20260421-0009`, and legacy
`R5-C`. They were not rerun as this lane's primary evidence because Round 57
already bridged the central xfail/lit subset.

## Non-bridge lit sweep

Command:

```bash
BUILD_DIR=$(PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
cd "$BUILD_DIR" && ninja triton-opt && lit -v \
  test/Conversion/lower_tensor_memory_to_llvm.mlir \
  test/Conversion/tritongpu_to_llvm_blackwell.mlir \
  test/Conversion/relayout_tritongpu.mlir \
  test/TritonGPU/promote-lhs-to-tmem.mlir \
  test/TritonGPU/hoist-tmem-alloc.mlir \
  test/TritonGPU/memdesc-subview-split.mlir
```

Result:

```text
Total Discovered Tests: 6
  Passed: 5 (83.33%)
  Failed: 1 (16.67%)
```

Failing file: `TRITON :: Conversion/relayout_tritongpu.mlir`.

Failure signature:

```text
dyn_cast on a non-existent value
mlir::triton::gpu::TritonGPUDialect::toLinearLayout(...)
mlir::triton::nvidia_gpu::computeTMemLdStEncodingInfoImpl(...)
mlir::triton::nvidia_gpu::verifyTMEMOperand(...)
```

Classification: existing `FZ-20260421-0016`, the unencoded or
non-distributed TMEM operand verifier/assertion boundary. This is the same
known failure classified in Round 52 and Round 55, not a new independent
bucket. The other five lit files passed.

## Final classification

No new independent `FZ-*` was found. Checked-in pytest clean-negative coverage
for invalid TMEM layouts, copy, `ld.red`, plain MMAv5, and scaled-MMAv5 stayed
stable as `200 passed`. Non-bridge lit coverage reproduced only existing
`FZ-20260421-0016`; no additional PassManager failure, assertion text drift,
too-vague diagnostic, false unsupported diagnostic, runtime miscompile, or
compiler crash was observed. Backend/compiler repairs remain deferred.
