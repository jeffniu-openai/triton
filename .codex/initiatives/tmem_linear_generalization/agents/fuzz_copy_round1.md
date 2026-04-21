# Lane B: tcgen05.copy Structural Fuzzing Round 1

- Time: 2026-04-21 08:24 UTC
- Branch/HEAD: `codex/tmem` at `87c50a24211a`
- Lane: B, `tcgen05.copy` Python/Gluon runtime fuzzing
- Mode: discovery only; no backend fixes attempted

## Summary

Round 1 did not find a new compiler crash, runtime miscompile, or unexpected
verifier failure in the checked copy-family probes.

One preexisting strict xfail was observed in the untracked structural fuzzer:
`copy-scales-warpx4-2cta` (`seed=0x302`) executes correctly but the structural
test expects `cta_group::2` while current direct scales-copy runtime-matrix
coverage expects `tcgen05.cp.cta_group::1.warpx4.32x128b`. Treat this as an
existing opcode-contract/coverage disagreement for follow-up classification,
not as a new lane-B backend failure from this round.

## Build

```bash
CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8
```

Result: passed, `ninja: no work to do`.

## Broad Runtime Probe

Purpose: deterministic copy-family sweep covering scales/non-scales,
`warpx4`, `warpx2::01_23`, `warpx2::02_13`, indexed and subslice descriptor
views, shared subslices, two-CTA cases, >2 CTA clean diagnostics where present,
and packed/subword clean diagnostics.

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_ and (scales_warpx4 or no_scales_warpx2 or twocta or 4cta or descriptor_view or indexed_view or subslice_view or shared_subslice or subword or packed or reports_clean)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_ and (scales_warpx4 or no_scales_warpx2 or twocta or 4cta or descriptor_view or indexed_view or subslice_view or shared_subslice or subword or packed or reports_clean)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_ and (scales_warpx4 or no_scales_warpx2 or twocta or 4cta or descriptor_view or indexed_view or subslice_view or shared_subslice or subword or packed or reports_clean)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_ and (scales_warpx4 or no_scales_warpx2 or twocta or 4cta or descriptor_view or indexed_view or subslice_view or shared_subslice or subword or packed or reports_clean)'
```

Results:

- Group 1: `54 passed, 1563 deselected`
- Group 2: `54 passed, 1563 deselected`
- Group 3: `54 passed, 1563 deselected`
- Group 4: `51 passed, 1566 deselected`

Note: the `-k 'cp_ ...'` selector excludes the structural fuzzer copy nodeids
because their names use `copy`, so those were rerun exactly below.

## Exact Runtime Probes

Scales `warpx4`, including the structural fuzzer case:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales
```

Result: `1 passed, 1 xfailed`. The xfail is
`copy-scales-warpx4-2cta`, `seed=0x302`, reason: executes correctly but emits
`cta_group::1` instead of the structural test's expected `cta_group::2`.

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4 python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_twocta_direct_copy
```

Result: `2 passed`. This pins the current runtime-matrix contract:
single-CTA and two-CTA direct scales copy both use
`tcgen05.cp.cta_group::1.warpx4.32x128b`.

Two-CTA layout in a 4-CTA context clean diagnostic:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error
```

Result: `1 passed`.

Two-CTA `warpx2::02_13` clean unsupported boundary:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_candidate_reports_clean_unsupported
```

Result: `2 passed`.

Xfail confirmation:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short -rx 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-2cta]'
```

Result: `1 xfailed`; failure class for cataloging is `opcode_mismatch` if the
campaign decides the structural fuzzer expectation should be authoritative.

## Collected Descriptor/Subslice/Diagnostic Inventory

Collect-only command:

```bash
PYTHONPATH=.:./python:./python/test/gluon pytest -q --collect-only python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales and (linear_indexed_view or linear_subslice_view or warpx2_dense_shared_rematerializes or warpx2_01_23_twocta_dense_shared_rematerializes or warpx2_02_13_twocta_dense_shared_reports_clean_unsupported or warpx2_02_13_twocta_subslice_view_reports_clean_unsupported or warpx2_02_13_twocta_indexed_view_reports_clean_unsupported or warpx2_subword_dtypes_report_clean_error or twocta_layout_in_4cta_context_reports_clean_error or shared_subslice)'
```

Result: `70/1615` nodeids collected. The broad runtime probe above covered
these buckets through `cp_` selectors:

- single-CTA linear indexed descriptor views: `128x128b`, `128x256b`, f32/i32
- two-CTA linear indexed descriptor views: f32/i32/f16/bf16/i16/i8
- single-CTA linear TMEM subslice views: f32/i32, N=128/256, swizzle 32/64/128
- two-CTA linear TMEM subslice views: f32/i32/f16/bf16/i16/i8
- shared subslice bad-offset clean error
- single-CTA `warpx2::01_23` and `warpx2::02_13` dense shared rematerialization positives
- two-CTA `warpx2::01_23` dense shared rematerialization positives
- two-CTA `warpx2::02_13` dense/indexed/subslice clean unsupported diagnostics
- `warpx2` packed/subword clean errors for single-CTA and two-CTA variants
- two-CTA layout in a 4-CTA context clean error

## Failure Catalog

No new lane-B failure was found.

Preexisting/staged item:

- `case_id`: `copy-scales-warpx4-2cta`
- `seed`: `0x302`
- `family`: `copy`
- `shape`: scales copy, M=128, N=16, int8
- `layout_kind`: `TensorMemoryScalesLayout(cga_layout=[[1, 0]])`
- `view_chain`: none
- `num_warps`: 4
- `num_ctas`: 2
- `expected result`: structural fuzzer expects pass with `cta_group::2`
- `observed result`: strict xfail; output is correct but opcode uses `cta_group::1`
- `failure class`: `opcode_mismatch` if this expectation is accepted; otherwise coverage-contract mismatch between structural fuzzer and runtime matrix
- `exact command`: see xfail confirmation command above
- `repro/minimization status`: already minimal to one structural fuzzer param
- `likely owner surface`: `tcgen05.copy` scales CTA-group opcode contract / structural fuzzer expected opcode

## Suggested Next Cases

- Add a report-only or checked-in deterministic probe that removes ambiguity
  between two-CTA scale-copy execution semantics and opcode naming:
  compare `TensorMemoryScalesLayout(cga_layout=[[1,0]])` direct copy against
  the scaled-MMA copy path and record whether `cta_group::1` is intentionally
  the ISA atom for scale copy under a two-CTA CGA.
- Extend copy ad-hoc probes to `num_ctas=4` and `num_ctas=8` contexts for
  positive single-pair operations where the TMEM/shared layouts are explicitly
  one- or two-CTA scoped, confirming clean context diagnostics instead of
  verifier crashes.
- Probe shared subslice starts beyond current bad-offset coverage for both
  no-scale dense copies and scales-layout copies, especially starts that are
  aligned to the copy atom but not to the parent descriptor's natural tile.
- Add exact runtime probes for two-CTA `warpx2::01_23` descriptor chains using
  combined `slice(...).index(...)` views, not only separate indexed/subslice
  rows.
- Continue packed/subword negative probing around boundary widths that expose
  exactly enough physical dword columns but insufficient packed-lane semantic
  information.

## Continued Slice: Descriptor/View Follow-Ups

Time: 2026-04-21 08:25 UTC.

After the first report checkpoint, lane B continued with exact nodeid probes
for the suggested next descriptor/view cases. No new failure was found.

Scales shared subslices plus scales TMEM descriptor-view clean unsupported:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_shared_subslice_layout_rematerializes python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_tmem_descriptor_view_reports_clean_unsupported
```

Result: `3 passed`.

Single-CTA no-scale `warpx2` combined `slice(...).index(...)` descriptor
views for both `01_23` and `02_13`:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_slice_index_view_positive
```

Result: `8 passed`.

Two-CTA no-scale `warpx2::01_23` combined `slice(...).index(...)` descriptor
views:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_slice_index_view_positive
```

Result: `4 passed`.

Two-CTA no-scale `warpx2::02_13` combined `slice(...).index(...)` clean
unsupported plus no-scale shared-subslice clean error:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_slice_index_view_reports_clean_unsupported python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_shared_subslice_bad_offset_reports_clean_error
```

Result: `5 passed`.

Additional classification from this slice:

- scales shared subslice starts currently covered by runtime matrix (`0`, `64`)
  remain positive through rematerialization and emit
  `tcgen05.cp.cta_group::1.warpx4.32x128b`;
- scales TMEM descriptor-view copy remains a clean unsupported diagnostic,
  not a crash;
- combined slice/index `warpx2::01_23` is positive for single-CTA and two-CTA
  no-scale copies;
- combined slice/index `warpx2::02_13` remains positive for single-CTA and
  clean unsupported for two-CTA, matching the dense/indexed/subslice
  classification from the broad sweep;
- no packed/subword or shared-subslice diagnostic regressed during the exact
  reruns.
