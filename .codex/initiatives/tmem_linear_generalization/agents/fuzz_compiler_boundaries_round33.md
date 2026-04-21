# Round 33 Compiler Boundary Fuzzing

Date: 2026-04-21 13:18 UTC
Branch: `codex/tmem`
HEAD: `184e76196`
Lane: compiler-only adversarial TMEM verifier/allocation/lowering boundaries.

## Scope

This lane generated focused MLIR probes under
`/tmp/tmem_compiler_boundaries_round33` for:

- descriptor-view chains and dynamic `ttg.memdesc_index` consumers;
- unencoded tensor operands and reduction results;
- encoded 64-bit TMEM alloc/load/store operands;
- malformed but near-valid `ttng.tensor_memory_linear` layouts;
- `tcgen05.copy` `warpx2`/`warpx4` candidate boundaries;
- reduction-load and scale-layout operands.

No backend/compiler/test source was edited.

## Commands

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Generated probe artifacts:

```bash
python3 - <<'PY'
# Generated 16 focused MLIR probes under:
# /tmp/tmem_compiler_boundaries_round33
PY
```

Compiler matrix:

```bash
BUILD=/root/code/triton/build/cmake.linux-aarch64-cpython-3.12
OPT=$BUILD/bin/triton-opt

for f in /tmp/tmem_compiler_boundaries_round33/*.mlir; do
  timeout 20s "$OPT" "$f"
  timeout 20s "$OPT" "$f" \
    --triton-nvidia-optimize-tmem-layouts \
    --allow-unregistered-dialect
  timeout 20s "$OPT" "$f" \
    --triton-nvidia-optimize-tmem-layouts \
    --allocate-shared-memory-nv='compute-capability=100 ptx-version=87' \
    --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' \
    --convert-nv-gpu-to-llvm \
    --allow-unregistered-dialect
done
```

Artifacts:

- `/tmp/tmem_compiler_boundaries_round33/results.tsv`
- `/tmp/tmem_compiler_boundaries_round33/<case>.<verify|optimize|lower>.log`

## Counts

Across 16 MLIR cases and three compiler modes per case:

- `10` passed.
- `18` produced clean diagnostics.
- `16` aborted with assertions, crashes, or stack dumps.
- `4` produced late illegal-op failures:
  - `3` are existing `FZ-20260421-0001` dynamic `ttg.memdesc_index` lowerings;
  - `1` is a harness-only tensor-returning `tt.func` LLVM-conversion
    limitation on the constant-index positive control.

Cases:

```text
bad_linear_dup_load_boundary
bad_linear_zero_basis_store_boundary
chain_const_index_load_pass
chain_dynamic_index_load_fz0001
chain_dynamic_store_fz0001
copy_dynamic_dst_warpx2_fz0001
copy_warpx2_shared_to_tmem_pass_or_clean
copy_warpx4_shared_to_tmem_pass_or_clean
f64_descriptor_chain_store_fz0017
i64_descriptor_chain_load_fz0017
i64_tmem_alloc_init_fz0017
ldred_reduction_unencoded_fz0016
ldred_value_unencoded_fz0016
scales_dynamic_index_copy_fz0001
scales_i64_load_clean_or_fz0017
unencoded_alloc_init_fz0016
```

## Classification

No new independent `FZ-*` bucket was found.

### `FZ-20260421-0001`: Dynamic Descriptor Index Lowering

Live dynamic `ttg.memdesc_index` still fails during lowering when feeding
load/store/copy consumers:

- `/tmp/tmem_compiler_boundaries_round33/chain_dynamic_index_load_fz0001.mlir`
- `/tmp/tmem_compiler_boundaries_round33/chain_dynamic_store_fz0001.mlir`
- `/tmp/tmem_compiler_boundaries_round33/copy_dynamic_dst_warpx2_fz0001.mlir`

Failure:

```text
error: failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

The constant-index descriptor-chain load verifies and optimizes cleanly. Its
full LLVM conversion fails at `tt.return` because the probe returns a tensor;
that is a probe-shape limitation, not a TMEM backend finding.

### `FZ-20260421-0016`: Unencoded Tensor Boundaries

Unencoded TMEM operands/results still crash instead of diagnosing cleanly:

- `/tmp/tmem_compiler_boundaries_round33/unencoded_alloc_init_fz0016.mlir`
  crashes at verifier time through `verifyTMEMOperand` /
  `TMEMAllocOp::verify`.
- `/tmp/tmem_compiler_boundaries_round33/ldred_value_unencoded_fz0016.mlir`
  crashes at verifier time when the value result is unencoded.
- `/tmp/tmem_compiler_boundaries_round33/ldred_reduction_unencoded_fz0016.mlir`
  verifies and optimizes when only the reduction result is unencoded, then
  crashes during LLVM conversion in `TritonGPUToLLVMTypeConverter` while
  converting an unencoded tensor result.

The single-result reduction case sharpens `FZ-0016`: the shared verifier
does not catch every unencoded reduction result, allowing a later lowering
crash. It is not a new independent bucket because the root boundary remains
unencoded tensor values reaching TMEM lowering paths without a clean
diagnostic.

### `FZ-20260421-0017`: Encoded 64-Bit TMEM Load/Store/Alloc

Encoded `i64`/`f64` non-reduction TMEM load/store and initialized alloc remain
under the existing bitwidth assertion:

- `/tmp/tmem_compiler_boundaries_round33/i64_descriptor_chain_load_fz0017.mlir`
- `/tmp/tmem_compiler_boundaries_round33/f64_descriptor_chain_store_fz0017.mlir`
- `/tmp/tmem_compiler_boundaries_round33/i64_tmem_alloc_init_fz0017.mlir`

Failure:

```text
TensorMemoryUtils.cpp:7713: Assertion `bitwidth == 32' failed.
```

The descriptor-chain rows show dynamic descriptor views do not need to reach
lowering to reproduce `FZ-0017`; verification itself asks for load/store
encoding info and aborts first.

## Clean Diagnostic Boundaries

Clean diagnostics were observed for:

- malformed duplicate/zero-basis linear TMEM layouts:
  `Layout is expected to be surjective`;
- 2CTA/4CTA copy candidates with a 1CTA shared layout:
  `Layout has 1 CTAs per CGA, but the context requires ...`;
- scale-copy candidate with a noncanonical shared layout:
  `The source shared layout does not match any recognized tcgen05.copy family`;
- `tensor_memory_scales_encoding<>` with `i64`:
  `bitwidth must be 8`.

These are clean verifier boundaries, not assertion aborts or false unsupported
diagnostics under this lane.

## Next Leads

- Generate copy `warpx2`/`warpx4` probes with CGA-compatible shared layouts so
  they reach the copy planner rather than context-layout verification.
- Keep the `ldred_reduction_unencoded_fz0016` row for a future regression
  sentinel: it distinguishes verifier-missed unencoded reduction results from
  verifier-caught unencoded value results.
