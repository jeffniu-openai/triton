# Round 32 Subword/Narrow-Shape Python Runtime TMEM Fuzzing

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

This lane focused on the narrow/subword neighborhood around
`FZ-20260421-0017` without using unsupported 64-bit TMEM element types as the
main stimulus. It used Python/Gluon runtime execution and checked-in opcode and
diagnostic assertions to cover:

- `f16`, `bf16`, `i16`, `i8`, `f32`, and `i32` TMEM load/store/copy paths;
- narrow N and packet-width boundaries, including x1 packed load/store forms
  and `tcgen05.copy` `128x128b` / `128x256b` forms;
- direct, descriptor-chain, indexed-view, and subslice-view operands;
- 1CTA and 2CTA tensor-memory layouts;
- packed/subword clean diagnostics for `warpx2`, legacy subword copy, and
  subinstruction tile-permuted layouts.

The lane also incorporates the sibling temporary Python/Gluon roundtrip probe
from `agents/fuzz_sub32_python_round32.md`, which independently executed
`float32`, `float16`, `bfloat16`, `int32`, `int16`, `int8`, `uint8`, and `bool`
TMEM load/store roundtrips through direct and descriptor-view operands.

## Commands

Runtime subword/copy/diagnostic sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_subword or x1_subword or cp_128x128_subword_exact_width or cp_no_scales_linear_subword_dtypes or cp_no_scales_twocta_linear_indexed_view or cp_no_scales_twocta_linear_subslice_view or cp_no_scales_warpx2_subword_dtypes_report_clean_error or cp_no_scales_legacy_subword_dtypes_report_clean_error or cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported) and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_subword or x1_subword or cp_128x128_subword_exact_width or cp_no_scales_linear_subword_dtypes or cp_no_scales_twocta_linear_indexed_view or cp_no_scales_twocta_linear_subslice_view or cp_no_scales_warpx2_subword_dtypes_report_clean_error or cp_no_scales_legacy_subword_dtypes_report_clean_error or cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported) and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_subword or x1_subword or cp_128x128_subword_exact_width or cp_no_scales_linear_subword_dtypes or cp_no_scales_twocta_linear_indexed_view or cp_no_scales_twocta_linear_subslice_view or cp_no_scales_warpx2_subword_dtypes_report_clean_error or cp_no_scales_legacy_subword_dtypes_report_clean_error or cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported) and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_subword or x1_subword or cp_128x128_subword_exact_width or cp_no_scales_linear_subword_dtypes or cp_no_scales_twocta_linear_indexed_view or cp_no_scales_twocta_linear_subslice_view or cp_no_scales_warpx2_subword_dtypes_report_clean_error or cp_no_scales_legacy_subword_dtypes_report_clean_error or cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported) and not resource'
```

Supported 32-bit narrow/control sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales_linear_32bit_dtypes or cp_no_scales_twocta_codegen or cp_no_scales_linear_indexed_view or cp_no_scales_linear_subslice_view) and not reports and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales_linear_32bit_dtypes or cp_no_scales_twocta_codegen or cp_no_scales_linear_indexed_view or cp_no_scales_linear_subslice_view) and not reports and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales_linear_32bit_dtypes or cp_no_scales_twocta_codegen or cp_no_scales_linear_indexed_view or cp_no_scales_linear_subslice_view) and not reports and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales_linear_32bit_dtypes or cp_no_scales_twocta_codegen or cp_no_scales_linear_indexed_view or cp_no_scales_linear_subslice_view) and not reports and not resource'
```

Sibling temporary Python/Gluon probe commands and artifacts are recorded in
`agents/fuzz_sub32_python_round32.md`. Its artifacts are under
`/tmp/tmem_sub32_round32/`.

## Results

Runtime subword/copy/diagnostic sweep:

```text
group 1: 20 passed, 1595 deselected
group 2: 20 passed, 1595 deselected
group 3: 20 passed, 1595 deselected
group 4: 19 passed, 1596 deselected
aggregate: 79 passed
```

Supported 32-bit narrow/control sweep:

```text
group 1: 15 passed, 1600 deselected
group 2: 15 passed, 1600 deselected
group 3: 15 passed, 1600 deselected
group 4: 15 passed, 1600 deselected
aggregate: 60 passed
```

Sibling temporary Python/Gluon sub-32/32-bit roundtrip probe:

```text
80 passed
```

Combined runtime evidence for this lane: `219` passing rows and no unexpected
compiler crash, unsupported diagnostic, opcode mismatch, or runtime
miscompile.

## Classification

No new independent `FZ-*` bucket was found.

The lane is a green guardrail around `FZ-20260421-0017`: the current
bitwidth-related crash remains isolated to encoded `i64`/`f64` non-reduction
TMEM load/store lowering. Supported 32-bit controls and sub-32-bit load/store
and copy paths, including descriptor views and 2CTA variants, executed with
correct outputs or produced the expected clean diagnostics for hardware- or
layout-impossible packed copy forms.
