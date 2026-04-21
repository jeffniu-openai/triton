# Round 54 Lane B: TMEM resource and shape extremes

- Date: 2026-04-21 15:24 UTC
- Branch: `codex/tmem`
- Starting HEAD: `2912b49bb Integrate Round 53 fuzz evidence`
- Scope: discovery/cataloging only. No backend/compiler repairs and no
  checked-in tests were modified.

## Objective

Stress resource and shape extremes for TMEM runtime surfaces:

- M/N extremes and split-N/M64 rows;
- high-CGA and 4/8/16 CTA multicast contexts;
- 4x256b, `warpx2`, `128x128b`, and large-copy boundaries;
- subword, `.x1`, `i64`, and `f64` boundaries;
- known ptxas/resource failures around large `ld.red`.

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

## Checked-in runtime matrix: resource/shape positive lane

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k '(resource or m64 or n512 or n384 or high_cga or x1 or subword or i64 or f64 or non_f32 or ld_red_m64 or 4x256b or warpx2 or n16 or n32) and not reports' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `515/1615 tests collected`.

Split-4 run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 -k '(resource or m64 or n512 or n384 or high_cga or x1 or subword or i64 or f64 or non_f32 or ld_red_m64 or 4x256b or warpx2 or n16 or n32) and not reports' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 -k '(resource or m64 or n512 or n384 or high_cga or x1 or subword or i64 or f64 or non_f32 or ld_red_m64 or 4x256b or warpx2 or n16 or n32) and not reports' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 -k '(resource or m64 or n512 or n384 or high_cga or x1 or subword or i64 or f64 or non_f32 or ld_red_m64 or 4x256b or warpx2 or n16 or n32) and not reports' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 -k '(resource or m64 or n512 or n384 or high_cga or x1 or subword or i64 or f64 or non_f32 or ld_red_m64 or 4x256b or warpx2 or n16 or n32) and not reports' python/test/gluon/test_tmem_runtime_matrix.py
```

Results:

```text
group 1/GPU0: 102 passed, 27 skipped, 1486 deselected in 4.69s
group 2/GPU1: 88 passed, 41 skipped, 1486 deselected in 6.66s
group 3/GPU2: 121 passed, 2 skipped, 6 failed, 1486 deselected in 5.71s
group 4/GPU3: 128 passed, 1487 deselected in 16.44s
aggregate: 439 passed, 70 skipped, 6 failed
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

## Checked-in clean/resource boundary lane

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k '(reports or reports_clean or resource or clean or unsupported or high_cga) and (m64 or n512 or n384 or high_cga or x1 or subword or i64 or f64 or non_f32 or ld_red or 4x256b or warpx2 or n16 or n32 or copy)' \
  python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `66/1648 tests collected`.

Split-4 run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 -k '(reports or reports_clean or resource or clean or unsupported or high_cga) and (m64 or n512 or n384 or high_cga or x1 or subword or i64 or f64 or non_f32 or ld_red or 4x256b or warpx2 or n16 or n32 or copy)' python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 -k '(reports or reports_clean or resource or clean or unsupported or high_cga) and (m64 or n512 or n384 or high_cga or x1 or subword or i64 or f64 or non_f32 or ld_red or 4x256b or warpx2 or n16 or n32 or copy)' python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 -k '(reports or reports_clean or resource or clean or unsupported or high_cga) and (m64 or n512 or n384 or high_cga or x1 or subword or i64 or f64 or non_f32 or ld_red or 4x256b or warpx2 or n16 or n32 or copy)' python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 -k '(reports or reports_clean or resource or clean or unsupported or high_cga) and (m64 or n512 or n384 or high_cga or x1 or subword or i64 or f64 or non_f32 or ld_red or 4x256b or warpx2 or n16 or n32 or copy)' python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_tmem_structural_fuzzer.py
```

Results:

```text
group 1/GPU0: 17 passed, 1631 deselected in 4.98s
group 2/GPU1: 17 passed, 1631 deselected in 8.24s
group 3/GPU2: 17 passed, 1631 deselected in 6.76s
group 4/GPU3: 15 passed, 1633 deselected in 5.18s
aggregate: 66 passed
```

Coverage included `256x256` `ld.red` shared-memory resource diagnostics,
rank/higher-rank TMEM OOR diagnostics, `.x1` unsupported variants, 4x256b
copy/load-store refresh diagnostics, two-CTA `warpx2::02_13` clean
unsupported rows, warpx2 subword clean errors, M64 i8 MMAv5 clean errors,
scaled block-M64 backend diagnostics, and N16 scaled accumulator clean
boundaries.

## High-CGA core controls

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k '(test_tma_multicast_copy or test_tma_gather_scatter_multi_cta or test_tcgen05_mma_multicast_commit or test_tcgen05_mma_multicast_commit_twocta_linear_acc or test_tcgen05_mma_scaled_direct_multicast_barrier or test_tma_mma_uses_tma_shared_inputs_multicast or test_tma_mma_uses_tma_gather_shared_inputs_multicast)' \
  python/test/gluon/test_core.py
```

Result: `16/18114 tests collected`.

Split-4 run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 -k '(test_tma_multicast_copy or test_tma_gather_scatter_multi_cta or test_tcgen05_mma_multicast_commit or test_tcgen05_mma_multicast_commit_twocta_linear_acc or test_tcgen05_mma_scaled_direct_multicast_barrier or test_tma_mma_uses_tma_shared_inputs_multicast or test_tma_mma_uses_tma_gather_shared_inputs_multicast)' python/test/gluon/test_core.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 -k '(test_tma_multicast_copy or test_tma_gather_scatter_multi_cta or test_tcgen05_mma_multicast_commit or test_tcgen05_mma_multicast_commit_twocta_linear_acc or test_tcgen05_mma_scaled_direct_multicast_barrier or test_tma_mma_uses_tma_shared_inputs_multicast or test_tma_mma_uses_tma_gather_shared_inputs_multicast)' python/test/gluon/test_core.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 -k '(test_tma_multicast_copy or test_tma_gather_scatter_multi_cta or test_tcgen05_mma_multicast_commit or test_tcgen05_mma_multicast_commit_twocta_linear_acc or test_tcgen05_mma_scaled_direct_multicast_barrier or test_tma_mma_uses_tma_shared_inputs_multicast or test_tma_mma_uses_tma_gather_shared_inputs_multicast)' python/test/gluon/test_core.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 -k '(test_tma_multicast_copy or test_tma_gather_scatter_multi_cta or test_tcgen05_mma_multicast_commit or test_tcgen05_mma_multicast_commit_twocta_linear_acc or test_tcgen05_mma_scaled_direct_multicast_barrier or test_tma_mma_uses_tma_shared_inputs_multicast or test_tma_mma_uses_tma_gather_shared_inputs_multicast)' python/test/gluon/test_core.py
```

Result: `16 passed` (`4/4/4/4`). This keeps the 4/8/16 CTA multicast and
CGA rows green, including `test_tma_multicast_copy[ctas_per_cga2]` for the
16-CTA CGA case.

## Large `ld.red` ptxas/resource probe

Existing subprocess probe:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_fz0018_min_round32.py
```

Result:

```json
{
  "counts": {
    "FZ-20260421-0018": 16,
    "clean-shape-power-of-two-boundary": 20,
    "clean-shared-memory-boundary": 6,
    "clean-tensor-memory-boundary": 6,
    "pass": 12
  },
  "total": 60
}
```

The rerun reproduced the known late `ptxas-blackwell` register-allocation
failure for direct and same-footprint descriptor-view `M128xN512` `ld.red`
rows, plus adjacent `M256xN256`/legacy-equivalent rows. Passing controls
included `M64xN512` and column-reversed `M128xN512`; non-power-of-two
`M/N={96,160,384}` stayed clean shape boundaries, and larger live footprints
stayed clean shared-memory or tensor-memory resource boundaries. Classification:
existing `FZ-20260421-0018`, not new.

## i64/f64 compiler-only contrast

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

Results:

| Case | Result | Classification |
| --- | --- | --- |
| `/tmp/tmem_fz0017_round30/descriptor_valid/i64_descriptor_valid_load.mlir` | `rc=134`, abort/core dump | Existing `FZ-20260421-0017` |
| `/tmp/tmem_fz0017_round30/descriptor_valid/i64_descriptor_valid_store.mlir` | `rc=134`, abort/core dump | Existing `FZ-20260421-0017` |
| `/tmp/tmem_fz0017_round30/descriptor_valid/f64_descriptor_valid_load.mlir` | `rc=134`, abort/core dump | Existing `FZ-20260421-0017` |
| `/tmp/tmem_fz0017_round30/descriptor_valid/f64_descriptor_valid_store.mlir` | `rc=134`, abort/core dump | Existing `FZ-20260421-0017` |
| `/tmp/tmem_clean_boundary_round29/bitwidth_copy_red/i64_ld_red_min.mlir` | `rc=1`, clean `tmem_load reduction currently requires f32 element type` diagnostic | Clean non-f32 `ld.red` boundary |
| `/tmp/tmem_clean_boundary_round29/bitwidth_copy_red/f64_ld_red_min.mlir` | `rc=1`, clean `tmem_load reduction currently requires f32 element type` diagnostic | Clean non-f32 `ld.red` boundary |
| `/tmp/tmem_clean_boundary_round29/bitwidth_copy_red/i64_copy_shared_to_tmem.mlir` | `rc=0`, emits `tcgen05.cp.cta_group::1.128x256b` | Green 64-bit copy contrast |
| `/tmp/tmem_clean_boundary_round29/bitwidth_copy_red/f64_copy_shared_to_tmem.mlir` | `rc=0`, emits `tcgen05.cp.cta_group::1.128x256b` | Green 64-bit copy contrast |

## Classification

- Existing `FZ-20260421-0012`: reproduced by the six checked-in M64 f32
  `ld.red` row-basis failures. No adjacent `ld/st`, copy, subword, high-CGA,
  or clean diagnostic row drifted.
- Existing `FZ-20260421-0017`: reproduced by encoded `i64`/`f64`
  descriptor-view load/store MLIR aborts. Non-f32 `ld.red` still rejects
  cleanly, and `i64`/`f64` copy still lowers.
- Existing `FZ-20260421-0018`: reproduced by the large-footprint `ld.red`
  ptxas/resource probe. Adjacent shape/resource boundaries and passing controls
  stayed stable.
- Clean boundaries: all `66` checked-in clean/resource rows passed, and the
  large `ld.red` probe kept clean shape, shared-memory, and tensor-memory
  boundaries stable.
- No new independent `FZ-*` bucket was found. No backend repair was attempted.
