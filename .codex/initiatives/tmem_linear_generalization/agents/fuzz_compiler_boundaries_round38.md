# Round 38 Compiler Boundary Fuzzing

Date: 2026-04-21 13:53 UTC
Branch: `codex/tmem`
HEAD: `7327a5616`
Lane: compiler-only adversarial TMEM boundary fuzzing with generated MLIR.

## Scope

This lane generated focused MLIR probes under
`/tmp/tmem_compiler_boundaries_round38` for:

- verifier strictness for malformed `ttng.tensor_memory_linear` layouts;
- dynamic `ttg.memdesc_index` feeding TMEM `load`, `store`, `copy`,
  `tc_gen5_mma`, and `tc_gen5_mma_scaled`;
- high-rank descriptor-view algebra;
- unencoded tensor operands/results;
- encoded 64-bit TMEM load;
- `ld.red` non-f32 and unencoded-reduction boundaries; and
- copy wrong-memory-space diagnostics.

No backend/compiler/test source was edited.

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

Build-tree controls:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
lit -v test/TritonNvidiaGPU/tmem_layouts.mlir
lit -v test/TritonNvidiaGPU/interleave_tmem.mlir
lit -v test/TritonGPU/memdesc-subview-split.mlir
```

Results:

```text
ninja: no work to do.
PASS: TRITON :: TritonNvidiaGPU/tmem_layouts.mlir
PASS: TRITON :: TritonNvidiaGPU/interleave_tmem.mlir
PASS: TRITON :: TritonGPU/memdesc-subview-split.mlir
```

Generated corpus and matrix:

```bash
ROOT=/tmp/tmem_compiler_boundaries_round38
OPT=/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt

# Generated 14 MLIR probes under $ROOT, then ran:
"$OPT" "$f" -split-input-file --mlir-disable-threading
"$OPT" "$f" -split-input-file --mlir-disable-threading \
  --triton-nvidia-optimize-tmem-layouts --allow-unregistered-dialect
"$OPT" "$f" -split-input-file --mlir-disable-threading \
  --triton-tensor-memory-allocation --allow-unregistered-dialect
"$OPT" "$f" -split-input-file --mlir-disable-threading \
  --triton-nvidia-optimize-tmem-layouts \
  --allocate-shared-memory-nv='compute-capability=100 ptx-version=87' \
  --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' \
  --convert-nv-gpu-to-llvm --allow-unregistered-dialect
```

Artifacts:

- `/tmp/tmem_compiler_boundaries_round38/results.tsv`
- `/tmp/tmem_compiler_boundaries_round38/*.mlir`
- `/tmp/tmem_compiler_boundaries_round38/*.{verify,optimize,tmem_alloc,lower}.log`

## Pass Matrix

Across 14 MLIR cases and four compiler modes per case:

```text
PASS: 18
CLEAN_DIAG: 24
ILLEGAL_OP: 5
ABORT_OR_CRASH: 9
```

Pipeline rows:

```text
bad_linear_dup_clean                   clean diag in all 4 modes
bad_linear_zero_clean                  clean diag in all 4 modes
copy_dynamic_dst_warpx2_fz0001         pass verify/optimize/tmem_alloc; illegal op in lower
copy_tmem_src_wrong_space_clean        clean diag in all 4 modes
dynamic_idx_acc_mma_fz0001             pass verify/optimize/tmem_alloc; illegal op in lower
dynamic_idx_load_live_fz0001           pass verify/optimize/tmem_alloc; illegal op in lower
dynamic_idx_scales_mma_scaled_fz0001   pass verify/optimize/tmem_alloc; illegal op in lower
dynamic_idx_store_live_fz0001          pass verify/optimize/tmem_alloc; illegal op in lower
high_rank_const_index_chain_pass       clean unsupported diagnostic in all 4 modes
high_rank_dynamic_second_index_fz0001  clean unsupported diagnostic in all 4 modes
i64_chain_load_fz0017                  aborts in all 4 modes
ldred_non_f32_clean                    clean diag in all 4 modes
ldred_reduction_unencoded_fz0016       pass verify/optimize/tmem_alloc; crashes in lower
unencoded_store_fz0016                 aborts in all 4 modes
```

## Classification

No new independent `FZ-*` bucket was found.

### Existing `FZ-20260421-0001`: Dynamic Descriptor Index Lowering

Five rows still pass verifier, `OptimizeTMemLayouts`, and explicit
`--triton-tensor-memory-allocation`, then fail during LLVM conversion with:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

Repros:

- `/tmp/tmem_compiler_boundaries_round38/dynamic_idx_load_live_fz0001.mlir`
- `/tmp/tmem_compiler_boundaries_round38/dynamic_idx_store_live_fz0001.mlir`
- `/tmp/tmem_compiler_boundaries_round38/copy_dynamic_dst_warpx2_fz0001.mlir`
- `/tmp/tmem_compiler_boundaries_round38/dynamic_idx_acc_mma_fz0001.mlir`
- `/tmp/tmem_compiler_boundaries_round38/dynamic_idx_scales_mma_scaled_fz0001.mlir`

The new evidence broadens the same lowering gap to dynamic accumulator
descriptors for MMAv5 and dynamic scale descriptors for scaled-MMAv5. It is
not independent from `FZ-0001`: the live illegal operation and failure point
are still `ttg.memdesc_index` after earlier passes.

### Existing `FZ-20260421-0016`: Unencoded Tensor Boundaries

Unencoded operands/results remain assertion/crash paths instead of clean
diagnostics:

- `/tmp/tmem_compiler_boundaries_round38/unencoded_store_fz0016.mlir`
  aborts in verifier-level paths with `dyn_cast on a non-existent value`.
- `/tmp/tmem_compiler_boundaries_round38/ldred_reduction_unencoded_fz0016.mlir`
  passes verifier, `OptimizeTMemLayouts`, and `--triton-tensor-memory-allocation`,
  then crashes during LLVM conversion.

The latter keeps sharpening the known verifier-missed reduction-result arm of
`FZ-0016`; it is not a new bucket because the root boundary is still unencoded
tensor values reaching TMEM lowering.

### Existing `FZ-20260421-0017`: 64-Bit TMEM Load

Encoded `i64` descriptor-chain TMEM load still aborts immediately:

- `/tmp/tmem_compiler_boundaries_round38/i64_chain_load_fz0017.mlir`

Failure signature:

```text
TensorMemoryUtils.cpp:7713: Assertion `bitwidth == 32' failed.
```

This is the same encoded 64-bit non-reduction TMEM load/store boundary as
`FZ-0017`.

## Clean Boundaries

Clean diagnostics stayed clean for:

- malformed duplicate and zero-basis linear TMEM layouts:
  `Layout is expected to be surjective`;
- TMEM copy with TMEM source:
  `The source must be a shared memory buffer`;
- non-f32 `ld.red`:
  `tmem_load reduction currently requires f32 element type`;
- rank-4 descriptor-view attempts over preserved TMEM view encodings:
  `unsupported tensor memory memdesc_subslice view; preserved tensor memory view encoding also failed`.

The high-rank descriptor-view rows did not reproduce a crash or false
unsupported diagnostic in this compiler-only MLIR lane; they stop at the
current clean unsupported verifier boundary before dynamic-index lowering.

## Temp Paths

Primary directory:

```text
/tmp/tmem_compiler_boundaries_round38
```

Important logs:

```text
/tmp/tmem_compiler_boundaries_round38/results.tsv
/tmp/tmem_compiler_boundaries_round38/dynamic_idx_acc_mma_fz0001.lower.log
/tmp/tmem_compiler_boundaries_round38/dynamic_idx_scales_mma_scaled_fz0001.lower.log
/tmp/tmem_compiler_boundaries_round38/i64_chain_load_fz0017.verify.log
/tmp/tmem_compiler_boundaries_round38/ldred_reduction_unencoded_fz0016.lower.log
/tmp/tmem_compiler_boundaries_round38/unencoded_store_fz0016.verify.log
```

## Next Leads

- Keep the dynamic MMAv5 and scaled-MMAv5 descriptor-index rows as
  compiler-only `FZ-0001` sentinels if a checked-in structural MLIR fuzzer is
  added.
- For high-rank descriptor-view algebra, generate rows from already accepted
  Python/Gluon rank-5 patterns instead of hand-authored rank-4 linear aliases;
  the hand-authored forms stopped at a clean unsupported preserved-view
  diagnostic before stressing later lowering.
- Preserve the unencoded `ld.red` reduction-result row because it distinguishes
  verifier-missed unencoded results from verifier-caught unencoded operands.
