# Round 49: TMEM numeric and layout shape extremes

Date: 2026-04-21
Branch/HEAD: `codex/tmem` at `c839da07a`
Scope: report-only adversarial fuzzing of TMEM numeric and layout shape
extremes. No backend, test, or initiative source files were modified.

## Summary

No new independent `FZ-*` bucket is needed.

Checked-in runtime and structural coverage stayed stable across narrow `N`,
M64/M128/M256-like boundaries, split-N, subword, row/column permutations, and
clean shape/resource diagnostics:

- Broad shape-extreme runtime selector: `316` selected,
  `286 passed, 24 skipped, 6 failed`.
- Clean diagnostic/resource selector: `83` selected, `83 passed`.
- Structural descriptor/`ld.red`/allocator subset: `19` selected,
  `7 passed, 12 xfailed`.

The six runtime failures are exactly the existing `FZ-20260421-0012` M64 f32
`tcgen05.ld.red` destination-layout planner gap for non-identity row basis
layouts. The skips are stable resource/environment skips from the checked-in
matrix. There were no new compiler crashes, unexpected unsupported cases,
miscompiles, XPASS transitions, or clean-diagnostic drift.

I also reran the existing compiler-only 64-bit contrast for `FZ-20260421-0017`.
The owner split remains unchanged: encoded `i64`/`f64` descriptor-view
load/store hits the known `bitwidth == 32` assertion, non-`f32` `ld.red`
reports the clean f32-only diagnostic, and pure 64-bit `ttng.tmem_copy` lowers
successfully to `tcgen05.cp`.

## Commands

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

Broad shape-extreme collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k 'narrow or n16 or n32 or splitn or subword or ld_red_m64 or rowcol_permuted or pure_row_permuted or tile_permuted_32 or tile_permuted_64' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `316/1615 tests collected`.

Broad shape-extreme split-4 runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 -k 'narrow or n16 or n32 or splitn or subword or ld_red_m64 or rowcol_permuted or pure_row_permuted or tile_permuted_32 or tile_permuted_64' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 -k 'narrow or n16 or n32 or splitn or subword or ld_red_m64 or rowcol_permuted or pure_row_permuted or tile_permuted_32 or tile_permuted_64' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 -k 'narrow or n16 or n32 or splitn or subword or ld_red_m64 or rowcol_permuted or pure_row_permuted or tile_permuted_32 or tile_permuted_64' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 -k 'narrow or n16 or n32 or splitn or subword or ld_red_m64 or rowcol_permuted or pure_row_permuted or tile_permuted_32 or tile_permuted_64' python/test/gluon/test_tmem_runtime_matrix.py
```

Results:

```text
group 1/GPU0: 55 passed, 24 skipped, 1536 deselected in 5.04s
group 2/GPU1: 73 passed, 6 failed, 1536 deselected in 13.84s
group 3/GPU2: 79 passed, 1536 deselected in 7.93s
group 4/GPU3: 79 passed, 1536 deselected in 5.83s
aggregate: 286 passed, 24 skipped, 6 failed
```

Failed nodeids, all existing `FZ-20260421-0012`:

```text
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]
```

Representative diagnostic:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
Failed to lower TMEM load/store: unsupported dst layout
where out dims are: [row (size 128), col (size 32)]
```

The `row_rotate_col_even_odd_n128` rows report the same unsupported-destination
diagnostic for `[row (size 128), col (size 128)]`.

Clean diagnostic/resource collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k '(reports_clean or report_clean or reports or resource or unsupported) and (subword or narrow or n16 or n32 or m64 or rowcol or row_permuted or tile_permuted or ld_red)' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `83/1615 tests collected`.

Clean diagnostic/resource split-4 sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 -k '(reports_clean or report_clean or reports or resource or unsupported) and (subword or narrow or n16 or n32 or m64 or rowcol or row_permuted or tile_permuted or ld_red)' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 -k '(reports_clean or report_clean or reports or resource or unsupported) and (subword or narrow or n16 or n32 or m64 or rowcol or row_permuted or tile_permuted or ld_red)' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 -k '(reports_clean or report_clean or reports or resource or unsupported) and (subword or narrow or n16 or n32 or m64 or rowcol or row_permuted or tile_permuted or ld_red)' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 -k '(reports_clean or report_clean or reports or resource or unsupported) and (subword or narrow or n16 or n32 or m64 or rowcol or row_permuted or tile_permuted or ld_red)' python/test/gluon/test_tmem_runtime_matrix.py
```

Results:

```text
group 1/GPU0: 21 passed, 1594 deselected in 6.11s
group 2/GPU1: 21 passed, 1594 deselected in 5.18s
group 3/GPU2: 21 passed, 1594 deselected in 4.42s
group 4/GPU3: 20 passed, 1595 deselected in 7.52s
aggregate: 83 passed
```

This selector covers clean resource/shape boundaries for 256-row `ld.red`,
subword copy, legacy subword copy, row-permuted copy destinations, linear
row/column-permuted copy layouts, M64 i8 MMA clean errors, MMA row/column
permutation clean unsupported diagnostics, scaled block-M64 backend diagnostic,
mixed scaled-lhs format clean unsupported diagnostics, and N16 scaled
accumulator clean boundaries.

Structural collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k 'ldred or descriptor_view or subword or splitn or ldst_view or allocator_crash' \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `19/33 tests collected`.

Structural execution:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short \
  -k 'ldred or descriptor_view or subword or splitn or ldst_view or allocator_crash' \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result:

```text
7 passed, 14 deselected, 12 xfailed in 7.38s
```

The strict xfails cover existing descriptor-view wrong-output, `ld.red`
opcode-loss/software-reduction, row/col optimizer, and allocator/resource
sentinels. There was no XPASS drift.

## Compiler-Only 64-Bit Contrast

Command pattern:

```bash
BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
TRITON_OPT="$BUILD_DIR/bin/triton-opt"
timeout 60 "$TRITON_OPT" <case>.mlir \
  -allow-unregistered-dialect \
  --triton-tensor-memory-allocation \
  --allocate-shared-memory-nv=compute-capability=100 \
  --convert-triton-gpu-to-llvm=compute-capability=100
```

Cases and results:

| Case | Result | Classification |
| --- | --- | --- |
| `/tmp/tmem_fz0017_round30/descriptor_valid/i64_descriptor_valid_load.mlir` | `rc=134`, `bitwidth == 32` assertion | Existing `FZ-20260421-0017` |
| `/tmp/tmem_fz0017_round30/descriptor_valid/i64_descriptor_valid_store.mlir` | `rc=134`, `bitwidth == 32` assertion | Existing `FZ-20260421-0017` |
| `/tmp/tmem_fz0017_round30/descriptor_valid/f64_descriptor_valid_load.mlir` | `rc=134`, `bitwidth == 32` assertion | Existing `FZ-20260421-0017` |
| `/tmp/tmem_fz0017_round30/descriptor_valid/f64_descriptor_valid_store.mlir` | `rc=134`, `bitwidth == 32` assertion | Existing `FZ-20260421-0017` |
| `/tmp/tmem_clean_boundary_round29/bitwidth_copy_red/i64_ld_red_min.mlir` | `rc=1`, clean `tmem_load reduction currently requires f32 element type` diagnostic | Clean non-`f32` `ld.red` hardware/API boundary |
| `/tmp/tmem_clean_boundary_round29/bitwidth_copy_red/f64_ld_red_min.mlir` | `rc=1`, clean `tmem_load reduction currently requires f32 element type` diagnostic | Clean non-`f32` `ld.red` hardware/API boundary |
| `/tmp/tmem_clean_boundary_round29/bitwidth_copy_red/i64_copy_shared_to_tmem.mlir` | `rc=0`, lowers to `tcgen05.cp.cta_group::1.128x256b` | Green 64-bit copy contrast |
| `/tmp/tmem_clean_boundary_round29/bitwidth_copy_red/f64_copy_shared_to_tmem.mlir` | `rc=0`, lowers to `tcgen05.cp.cta_group::1.128x256b` | Green 64-bit copy contrast |

## Classification

- `FZ-20260421-0012`: freshly reproduced by the six M64 f32 `ld.red`
  row-basis failures. No adjacent `ld/st`, split-N, or row/column-permuted
  positive leaked into this bucket.
- `FZ-20260421-0017`: freshly reproduced by the four compiler-only encoded
  `i64`/`f64` descriptor-view load/store rows. The 64-bit copy contrast remains
  green, and non-`f32` `ld.red` remains a clean unsupported diagnostic.
- `FZ-20260421-0018`: no new large-footprint `ld.red` resource-planning
  failure appeared. Checked-in resource diagnostics passed cleanly.
- `FZ-20260421-0020`, `FZ-20260421-0021`, `FZ-20260421-0022`: no new
  half-row/half-column drift was found in this shape-extreme slice. Structural
  xfails stayed stable and checked-in clean diagnostics did not regress.
- Clean hardware/resource boundaries: all 83 selected diagnostic/resource rows
  passed and preserved their expected clean error surfaces.

## Conclusion

Round 49 did not find a new TMEM backend bug. The useful evidence is negative:
the narrow `N`/split-N/subword/tile-permuted positives are stable, explicit
resource and unsupported-shape diagnostics remain clean, and the only failures
are already-cataloged `FZ-20260421-0012` and `FZ-20260421-0017` boundaries.
