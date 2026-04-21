# TMEM structural fuzzing lane D: warp specialization / membar round 1

- Time: 2026-04-21 08:23 UTC
- Branch: `codex/tmem`
- HEAD: `1a3c13b937a17d191e46c640e5d2e07ab3b5ebe5`
- Lane: warp-specialization, partition scheduling, membar, and >2 CTA
- Mode: discovery only; no backend code edits

## Setup

- Read `AGENTS.md`.
- Read `.codex/initiatives/tmem_linear_generalization/tmem_structural_fuzzing_20260421.md`.
- Also re-read the required initiative entry points:
  `.codex/initiatives/tmem_linear_generalization/README.md`,
  `.codex/initiatives/tmem_linear_generalization/memory.md`, and the tail of
  `.codex/initiatives/tmem_linear_generalization/handoff_2026-04-09.md`.
- Required rebuild:
  `make -j8`
  - Result: passed; Ninja reported no work to do.
- Environment note: an initial collect-only command without `PYTHONPATH=.:./python`
  picked up `/tmp/triton-upstream-main-check` and failed import collection before
  any test execution. All real probe commands below used `PYTHONPATH=.:./python`.

## Probe Commands

### Runtime structural fuzzer

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py
```

Result:

- `8 passed, 1 failed`
- The failing case is minimized below.

### Larger-CGA tcgen05 commit / async completion barrier probes

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_core.py -k 'tcgen05_mma_multicast_commit or tcgen05_mma_scaled_direct_multicast_barrier'
```

Result:

- `9 passed, 18105 deselected`
- Covered 2, 4, and 16 CTA multicast/commit paths, two-CTA accumulator layouts,
  scaled direct multicast barrier, `tcgen05.commit`, and runtime output compares.

### Warp-specialized TMA/MMA shared-input probes

Command:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_core.py::test_tma_mma_shared_inputs[True-True-True-ctas_per_cga2-reps0-warps2]' 'python/test/gluon/test_core.py::test_tma_mma_shared_inputs[True-True-True-ctas_per_cga1-reps0-warps2]' 'python/test/gluon/test_core.py::test_tma_mma_shared_inputs[True-False-True-ctas_per_cga2-reps0-warps2]' 'python/test/gluon/test_core.py::test_tma_mma_shared_inputs[False-True-False-ctas_per_cga2-reps0-warps2]'
```

Result:

- `4 passed`
- Covered gather/scatter, multicast, two-CTA TMEM, 2-CTA and 16-CTA CGA cases,
  mbarrier phase toggling, and runtime output compares.

### Descriptor-view / memdesc-capture runtime matrix probes

Command:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_indexed_acc_view[f16-linear-128-128-True]' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_indexed_acc_view[f16-linear_unit_parent-256-128-True]' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_acc_subslice_view_plain_kinds[f16-128-128-128-True]' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_acc_subslice_view_plain_kinds[f16-256-128-256-True]' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_shared_scale_descriptor_view_auto_tmem_copy[2-256-64-128-0-False]'
```

Result:

- `5 passed`
- Covered `ttg.memdesc_index`, `ttg.memdesc_subslice`, lifted linear parent
  layouts, `use_acc=True`, two-CTA MMA commits, and shared scale descriptor
  views through the auto TMEM copy path.

### Compiler-side partition / membar sanity

Command:

```bash
BUILD_DIR=$(PYTHONPATH='./python' python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && ninja triton-opt && lit -v test/TritonGPU/partition-scheduling.mlir test/TritonGPU/optimize-partition-warps.mlir test/TritonGPU/partition-verifier-locality.mlir test/TritonNvidiaGPU/membar-cluster.mlir
```

Result:

- `4 passed`
- Covered partition scheduling, partition warp optimization, partition verifier
  locality, and cluster membar insertion.

## Minimized Failure

### laneD-copy-scales-warpx4-2cta-opcode-expectation

- `case_id`: `copy-scales-warpx4-2cta`
- `seed`: `0x302`
- `family`: `membar` / `copy`
- `shape`: `128x16xi8`
- `dtype`: `int8`
- `layout_kind`: `TensorMemoryScalesLayout(cga_layout=[[1, 0]])`
- `view_chain`: none
- `num_warps`: `4`
- `num_ctas`: `2`
- expected result in structural fuzzer: pass with `tcgen05.cp.cta_group::2.warpx4.32x128b`
- observed result: runtime data compare passes, but PTX/LLIR opcodes are
  `tcgen05.cp.cta_group::1.warpx4.32x128b`
- failure class: `opcode_mismatch`
- likely owner surface: structural fuzzer expectation, not backend lowering
- classification: stale/adversarial harness expectation; current runtime-matrix
  coverage explicitly expects the same direct two-CTA scales-copy geometry to
  lower as `cta_group::1` and pass.

Exact failing repro:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-2cta]'
```

Observed failure excerpt:

```text
AssertionError: ['tcgen05.cp.cta_group::1.warpx4.32x128b', 'tcgen05.cp.cta_group::1.warpx4.32x128b']
```

Confirming existing runtime-matrix contract:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_twocta_direct_copy'
```

Result:

- `1 passed`
- This test asserts `["tcgen05.cp.cta_group::1.warpx4.32x128b"] * 2` plus the
  corresponding `cta_group::1` commit and exact runtime output.

## Findings

- No backend compiler crash, verifier false unsupported diagnostic, clean-error
  regression, or runtime miscompile was found in this lane-D round.
- One deterministic structural-fuzzer failure was found, minimized, and
  classified as a stale opcode expectation in the untracked fuzzer artifact.
- Existing runtime-matrix coverage supports the current backend behavior for
  the minimized failing shape.

## Suggested Next Cases

- Add true warp-specialized TMEM descriptor-view probes once a Python/Gluon
  surface can force `ttg.warp_specialize` with TMEM memdesc values crossing
  partitions, rather than relying on lit-only partition coverage.
- Extend the structural fuzzer's copy family to separate direct scales-copy
  two-CTA ownership from scaled-MMA scale-copy ownership, since they can expect
  different `cta_group` opcodes.
- Add >2 CTA runtime probes for descriptor views through conditional selection
  and loop-carried memdesc values, then compare against current lit coverage
  for partition scheduling root discovery.
