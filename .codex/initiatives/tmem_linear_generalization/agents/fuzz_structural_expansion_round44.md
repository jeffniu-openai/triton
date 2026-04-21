# Round 44: Structural Fuzzer Expansion Audit

Date: 2026-04-21 14:29 UTC

Branch: `codex/tmem`

HEAD: `3503f3a0a747`

Scope: discovery/classification only. No backend/compiler code or checked-in
tests were modified. This report is the only repo file written by this lane.

## Objective

Audit the checked-in structural fuzzer and recent reports for under-covered
structural combinations that could become future checked-in cases:

- dynamic descriptor plus copy;
- high-rank `ld/st` wrong-output coverage;
- scale descriptor-view variants; and
- cache/process order controls.

Temporary probes were reused from `/tmp` when they already existed from recent
rounds. No new temporary harness was required.

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

## Checked-In Structural Fuzzer Audit

The checked-in file `python/test/gluon/test_tmem_structural_fuzzer.py` has 33
total tests. The overlap selector below selected 28 rows:

```bash
PYTHONPATH=./python pytest --collect-only -q \
  -k 'copy_scales or generic_pass or dynamic_index or descriptor_view or ldred or scaled_mma' \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `28/33` collected.

Four-GPU split run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python pytest -q -s --tb=short --splits 4 --group 1 -k 'copy_scales or generic_pass or dynamic_index or descriptor_view or ldred or scaled_mma' python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=./python pytest -q -s --tb=short --splits 4 --group 2 -k 'copy_scales or generic_pass or dynamic_index or descriptor_view or ldred or scaled_mma' python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=./python pytest -q -s --tb=short --splits 4 --group 3 -k 'copy_scales or generic_pass or dynamic_index or descriptor_view or ldred or scaled_mma' python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=./python pytest -q -s --tb=short --splits 4 --group 4 -k 'copy_scales or generic_pass or dynamic_index or descriptor_view or ldred or scaled_mma' python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `5 passed, 23 xfailed`.

Split details:

- group 1: `3 passed, 4 xfailed`;
- group 2: `7 xfailed`;
- group 3: `2 passed, 5 xfailed`;
- group 4: `7 xfailed`.

Observed expected xfail diagnostics included existing `FZ-20260421-0001`
runtime `ttg.memdesc_index` reaching LLVM conversion as an illegal op. No XPASS
transition, unexpected compiler crash, false clean diagnostic, or new
independent failure appeared.

What is already covered structurally:

- static `ld/st` view roundtrip positives;
- checked-in `ld/st` descriptor-view wrong-output xfails for existing
  `FZ-20260421-0003`;
- direct, descriptor-view, and 2CTA `ld.red` positives/xfails;
- direct copy-scales warpx4 1CTA/2CTA positives;
- dynamic descriptor generic-pass `load/store`, dynamic-if, mixed-capture,
  tuple-capture, layout-pressure, loop-carried, and load-only xfails;
- one scaled-MMAv5 accumulator subslice dynamic-if xfail.

What remains under-covered structurally:

- dynamic descriptor values consumed by `ttng.tmem_copy` are not checked in;
  they currently live only in temporary round-40 evidence;
- high-rank `ld/st` coverage is mostly in runtime-matrix positives, not in the
  structural fuzzer, and there is no current high-rank `ld/st` wrong-output
  xfail sentinel;
- scale descriptor-view coverage is mostly runtime-matrix positive/clean
  guardrails, not structural rows;
- cache/process-order controls are documented by round-42/round-43 reports but
  not represented as a dedicated checked-in process-order test.

## Runtime Coverage Inventory

Broad runtime overlap collection:

```bash
PYTHONPATH=./python pytest --collect-only -q \
  -k '(cp_no_scales or cp_scales or shared_scale_descriptor_view_auto_tmem_copy or bscale_descriptor_view or bscale_view_extra_user or ldst_descriptor_rank5 or higher_rank or scales_descriptor_view or rank5) and not reports and not resource' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `353/1615` collected.

This selector shows the runtime matrix already has broad checked-in coverage
for copy no-scales, copy scales, warpx2 copy descriptor/index/slice views,
rank-5 and higher-rank `ld/st`, scale `ld/st` descriptor views, shared-scale
descriptor-view auto-copy, and B-scale descriptor-view extra-user cases.

Focused exact runtime controls:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_rank5_unit_parent_n256_roundtrip[f32-torch_dtype0-single_identity-single-identity-128-1-auto-32x32b.x64.b32]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_positive_lifted_layout[identity-128-auto-32x32b.x128.b32]'
```

Result: `2 passed`.

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_scales_descriptor_view_cga_roundtrip[128-32-4-2-cga_layout6-16x256b-expected_ops6]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_bscale_descriptor_view'
```

Result: `2 passed`.

## Temporary Probe Reuse

Dynamic descriptor plus copy probe:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_copy_dynamic_descriptor_round40_probe.py
```

Result: `4 passed, 2 failed`.

Classification:

- the four branch-selected `warpx2::{01_23,02_13}` copy descriptor rows passed;
- the two branch-selected linear copy descriptor rows failed with the existing
  `FZ-20260421-0001` signature: an `scf.if`-yielded TMEM memdesc consumed by
  `ttng.tmem_copy` leaves `ttg.memdesc_index` illegal at LLVM conversion;
- this remains useful candidate evidence for a future checked-in xfail row,
  but it is not a new independent `FZ-*`.

B-scale dynamic descriptor-view probe:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_bscale_dynamic_views_round36_probe.py
```

Result: `5 passed`.

Classification: direct, branch-selected, loop-carried, descriptor-view chain,
and extra-user B-scale descriptor rows remain positive. No new opcode absence,
wrong result, compiler crash, or clean-boundary drift appeared.

## Cache And Process-Order Control

After the dynamic-copy probe reproduced expected `FZ-20260421-0001` failures
using `/tmp/triton-cache-gpu0`, a same-cache pytest process ran an expected
dynamic-index xfail followed by two positives:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python pytest -q -s --tb=short -rxX \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_dynamic_index_load_only[generic-pass-dynamic-index-load-only-128x32]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_bscale_view_extra_user_rematerializes' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_rank5_unit_parent_n256_roundtrip[f32-torch_dtype0-single_identity-single-identity-128-1-auto-32x32b.x64.b32]'
```

Result: `2 passed, 1 xfailed`.

Classification: no process-order or stable-cache contamination was observed.
The checked-in expected `FZ-20260421-0001` xfail did not poison later B-scale
descriptor-view or rank-5 high-rank positives in the same pytest process.

## Coverage Conclusions

No new independent `FZ-*` is proposed from this lane.

Candidate future checked-in rows:

1. Dynamic descriptor plus copy xfail candidate: promote the temporary
   `branch_selected_linear_copy_descriptor[0/1]` shape from
   `/tmp/tmem_copy_dynamic_descriptor_round40_probe.py` into the structural
   fuzzer as an expected `FZ-20260421-0001` copy-specific sentinel. It exercises
   `ttng.tmem_copy` as the dynamic memdesc consumer, which is not currently
   checked in.
2. Dynamic descriptor plus `warpx2` copy positives: the four temporary
   branch-selected `warpx2` rows stayed green and could become positive
   guardrails if the structural fuzzer is expanded beyond known-failure
   sentinels.
3. Process-order mini-sentinel: a small mixed process containing one expected
   dynamic descriptor xfail followed by B-scale descriptor-view and rank-5
   positives would encode the round-42/round-44 cache-state finding, but this
   is a heavier test shape and should only be checked in if process-order drift
   recurs.

Still under-covered after this audit:

- high-rank `ld/st` wrong-output has no new failing candidate; runtime matrix
  positives and round-39 temporary probes did not find a high-rank `ld/st`
  wrong-result bucket;
- scale descriptor-view variants are broad enough as positives, but scale
  descriptor-view plus dynamic descriptor selection is only partially covered by
  temporary B-scale probes and existing scaled-MMAv5 dynamic-view reports;
- dynamic descriptor plus copy has clear known-bucket evidence but no checked-in
  structural copy-specific row yet.

Backend repair remains deferred per the discovery-only campaign.
