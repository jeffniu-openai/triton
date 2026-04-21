# Round 37 Subword and 64-Bit TMEM Boundary Fuzzing

Date: 2026-04-21

Mode: discovery/cataloging only. No backend, compiler, or checked-in test code
was modified.

Required first step:

```bash
make -j8
```

Result: no-op rebuild through
`/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`.

## Scope

This lane targeted adversarial subword and 64-bit TMEM boundaries around
non-reduction `ld/st`, descriptor/rank/unit views, copy paths, initialized
allocation, roundtrip, 1CTA/2CTA, and clean diagnostics. It intentionally
classified against existing `FZ-20260421-0017` and subword/copy guardrails
rather than proposing overlapping buckets.

## Checked-In Selector

Collection:

```bash
PYTHONPATH=.:./python:./python/test/gluon pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k '(ldst_subword or x1_subword or cp_128x128_subword_exact_width or cp_no_scales_linear_subword_dtypes or cp_no_scales_twocta_linear_indexed_view or cp_no_scales_twocta_linear_subslice_view or cp_no_scales_warpx2_subword_dtypes_report_clean_error or cp_no_scales_legacy_subword_dtypes_report_clean_error or cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported or ld_red_non_f32 or subword) and not resource'
```

Result: `123/1648` tests collected.

Four-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 1 --store-durations --durations-path /tmp/tmem_round37_checked_durations.json python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_tmem_structural_fuzzer.py -k '(ldst_subword or x1_subword or cp_128x128_subword_exact_width or cp_no_scales_linear_subword_dtypes or cp_no_scales_twocta_linear_indexed_view or cp_no_scales_twocta_linear_subslice_view or cp_no_scales_warpx2_subword_dtypes_report_clean_error or cp_no_scales_legacy_subword_dtypes_report_clean_error or cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported or ld_red_non_f32 or subword) and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 2 --store-durations --durations-path /tmp/tmem_round37_checked_durations.json python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_tmem_structural_fuzzer.py -k '(ldst_subword or x1_subword or cp_128x128_subword_exact_width or cp_no_scales_linear_subword_dtypes or cp_no_scales_twocta_linear_indexed_view or cp_no_scales_twocta_linear_subslice_view or cp_no_scales_warpx2_subword_dtypes_report_clean_error or cp_no_scales_legacy_subword_dtypes_report_clean_error or cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported or ld_red_non_f32 or subword) and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 3 --store-durations --durations-path /tmp/tmem_round37_checked_durations.json python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_tmem_structural_fuzzer.py -k '(ldst_subword or x1_subword or cp_128x128_subword_exact_width or cp_no_scales_linear_subword_dtypes or cp_no_scales_twocta_linear_indexed_view or cp_no_scales_twocta_linear_subslice_view or cp_no_scales_warpx2_subword_dtypes_report_clean_error or cp_no_scales_legacy_subword_dtypes_report_clean_error or cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported or ld_red_non_f32 or subword) and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 4 --store-durations --durations-path /tmp/tmem_round37_checked_durations.json python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_tmem_structural_fuzzer.py -k '(ldst_subword or x1_subword or cp_128x128_subword_exact_width or cp_no_scales_linear_subword_dtypes or cp_no_scales_twocta_linear_indexed_view or cp_no_scales_twocta_linear_subslice_view or cp_no_scales_warpx2_subword_dtypes_report_clean_error or cp_no_scales_legacy_subword_dtypes_report_clean_error or cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported or ld_red_non_f32 or subword) and not resource'
```

Result:

| Group | Result |
| --- | --- |
| 1 | `31 passed, 1617 deselected` |
| 2 | `31 passed, 1617 deselected` |
| 3 | `31 passed, 1617 deselected` |
| 4 | `30 passed, 1618 deselected` |

Aggregate: `123 passed`. This reconfirms subword `ld/st`, x1 subword
roundtrip, 1CTA/2CTA subword descriptor views, linear/two-CTA no-scales copy
subword positives, `warpx2`/legacy/tile-permuted clean diagnostics, and
non-f32 `ld.red` software/diagnostic contracts.

Logs:

- `/tmp/tmem_round37_checked_g1.log`
- `/tmp/tmem_round37_checked_g2.log`
- `/tmp/tmem_round37_checked_g3.log`
- `/tmp/tmem_round37_checked_g4.log`
- `/tmp/tmem_round37_checked_durations.json`

## Temporary Probe

Probe/controller:

- `/tmp/tmem_round37_probe.py`
- `/tmp/tmem_round37_probe_controller.log`
- `/tmp/tmem_round37_subword_i64/child.py`
- `/tmp/tmem_round37_subword_i64/summary.json`
- `/tmp/tmem_round37_subword_i64/*.log`

Command:

```bash
rm -rf /tmp/tmem_round37_subword_i64
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_round37_probe.py
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_round37_probe.py 2>&1 | tee /tmp/tmem_round37_probe_controller.log
```

The controller launches each child case in a fresh process and assigns
`CUDA_VISIBLE_DEVICES=<case_id % 4>` with stable
`TRITON_CACHE_DIR=/tmp/triton-cache-gpu<N>`.

Matrix:

| Case | Result | Classification |
| --- | --- | --- |
| `i64_direct_1cta_128x64` | aborts at `lowerTMemLdSt` `bitwidth == 32` | Existing `FZ-20260421-0017` |
| `i64_direct_2cta_256x64` | same assertion | Existing `FZ-20260421-0017` |
| `f64_direct_1cta_128x64` | same assertion | Existing `FZ-20260421-0017` |
| `i64_init_alloc_1cta_128x128` | same assertion after source-initialized allocation | Existing `FZ-20260421-0017` |
| `i64_rank_unit_view_1cta_128x64` | same assertion through `[1,1,1,M,N]` unit-rank descriptor view | Existing `FZ-20260421-0017` expansion |
| `i64_rank_unit_view_2cta_256x64` | same assertion through 2CTA unit-rank descriptor view | Existing `FZ-20260421-0017` expansion |
| `i32_rank_unit_control_1cta_128x64` | pass, exact roundtrip | Green control |
| `f16_rank_unit_subword_control_1cta_128x64` | pass, exact roundtrip | Green subword control |
| `i64_copy_indexed_2cta_256x4_no_readback` | aborts in `get_reg_layout` before copy isolation | Existing `FZ-20260421-0017`, not copy-specific |
| `i64_copy_indexed_2cta_256x4_with_readback` | same assertion | Existing `FZ-20260421-0017`, not copy-specific |
| `i32_copy_indexed_2cta_control_256x8` | frontend/layout harness diagnostic: source and destination CGA layouts differ | Probe harness limitation; checked-in 2CTA copy controls cover this surface |

The rank/unit rows are the useful new coverage from this lane: 64-bit
non-reduction load/store still fails with the same bitwidth assertion even
after descriptor-view prefix dimensions are all unit-sized, while matching
`i32` and `f16` unit-rank views pass. This separates the false compiler
assertion from rank/unit descriptor arithmetic and from subword support.

## Compiler-Only Copy And Reduction Contrasts

`ld.red` clean diagnostic:

```bash
BUILD_DIR=$(PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
$BUILD_DIR/bin/triton-opt /tmp/tmem_fz0017_round30/i64_128x128_linear_ld_red_min_1cta.mlir \
  -allow-unregistered-dialect -triton-tensor-memory-allocation \
  2>&1 | tee /tmp/tmem_round37_i64_ldred_alloc.log
```

Result: clean verifier error, not an assertion:

```text
'ttng.tmem_load' op tmem_load reduction currently requires f32 element type
```

Pure 64-bit copy:

```bash
BUILD_DIR=$(PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
$BUILD_DIR/bin/triton-opt /tmp/tmem_clean_boundary_round29/bitwidth_copy_red/i64_copy_shared_to_tmem.mlir \
  -allow-unregistered-dialect \
  --triton-tensor-memory-allocation \
  --allocate-shared-memory-nv=compute-capability=100 \
  --convert-triton-gpu-to-llvm=compute-capability=100 \
  2>&1 | tee /tmp/tmem_round37_i64_copy_llvm.log
$BUILD_DIR/bin/triton-opt /tmp/tmem_clean_boundary_round29/bitwidth_copy_red/f64_copy_shared_to_tmem.mlir \
  -allow-unregistered-dialect \
  --triton-tensor-memory-allocation \
  --allocate-shared-memory-nv=compute-capability=100 \
  --convert-triton-gpu-to-llvm=compute-capability=100 \
  2>&1 | tee /tmp/tmem_round37_f64_copy_llvm.log
```

Result: both commands pass and lower to `tcgen05.cp.cta_group::1.128x256b`.
This keeps `FZ-20260421-0017` scoped to 64-bit non-reduction
`ttng.tmem_load`/`ttng.tmem_store` planning and `get_reg_layout` paths, not
pure `ttng.tmem_copy`.

## Classification

No new independent `FZ-*` bucket is proposed.

Existing bucket expanded:

- `FZ-20260421-0017`: encoded `i64`/`f64` non-reduction TMEM load/store
  lowering assertion now explicitly includes unit-rank descriptor views and a
  same-day 1CTA/2CTA Python runtime repro matrix. The assertion is still:
  `TensorMemoryUtils.cpp:7713: lowerTMemLdSt(...): Assertion 'bitwidth == 32' failed`.

Clean/API/ISA boundaries:

- Non-f32 `ld.red` remains a clean diagnostic requiring f32 reduction loads.
- Subword `warpx2`, legacy subword copy, tile-permuted copy, and non-f32
  reduction guardrails remain clean under checked-in tests.
- Pure `i64`/`f64` shared-to-TMEM copy lowers through LLVM, so copy itself is
  not the 64-bit assertion owner.

Green controls:

- `123` checked-in subword/copy/diagnostic rows passed.
- Temporary `i32` and `f16` unit-rank descriptor-view roundtrips passed with
  exact output equality.
