# Round 31b Compiler Boundary Follow-Up

Date: 2026-04-21 13:00 UTC
Branch: `codex/tmem`
Base checkpoint: `0dee35c2b`
Lane: compiler-only follow-up for TMEM reduction loads and scale layouts.

## Scope

This follow-up extended `fuzz_compiler_boundaries_round31.md` to:

- `ttng.tmem_load` reduction forms with live return values.
- `ttng.tensor_memory_scales_encoding` load/copy boundaries.
- Runtime `ttg.memdesc_index` into reduction and scale consumers.
- Clean diagnostic controls for unsupported 64-bit reductions and scale copy
  descriptor planning.

No backend or checked-in tests were edited.

## Commands

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Generated artifacts:

```text
/tmp/tmem_compiler_boundaries_round31b/*.mlir
/tmp/tmem_compiler_boundaries_round31b/*.log
/tmp/tmem_compiler_boundaries_round31b/results.tsv
/tmp/tmem_compiler_boundaries_round31b/results_round2.tsv
```

Compiler modes:

```bash
BUILD=/root/code/triton/build/cmake.linux-aarch64-cpython-3.12
OPT=$BUILD/bin/triton-opt

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
```

## Counts

Across 8 MLIR cases and three compiler modes per case:

- `6` passed.
- `15` produced clean diagnostics or known late illegal-op failures.
- `3` aborted with the existing unencoded-tensor verifier assertion.

The quick classifier initially counted the `ldred_i64_clean_or_crash` clean
diagnostic as crash-class because the filename contained `crash`; manual log
inspection corrected that classification.

## Cases

```text
ldred_dynamic_index_return_fz0001
ldred_i64_clean_or_crash
ldred_unencoded_result_fz0016
scales_copy_shared_to_tmem_boundary
scales_copy_wrong_src_space
scales_dynamic_index_load
scales_dynamic_index_load_linear_child
scales_load_boundary
```

## Classification

No new independent `FZ-*` bucket was found.

### `FZ-20260421-0001`: Dynamic Index Still Fails for Live Consumers

Reduction load and scale load consumers reproduce the same late illegal
`ttg.memdesc_index` failure when the indexed descriptor is live:

- `/tmp/tmem_compiler_boundaries_round31b/ldred_dynamic_index_return_fz0001.mlir`
- `/tmp/tmem_compiler_boundaries_round31b/scales_dynamic_index_load_linear_child.mlir`

Failure:

```text
error: failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

The first scale dynamic-index attempt used `tensor_memory_scales_encoding<>` as
the indexed child type; the verifier correctly inferred a linear child layout
and rejected the wrong expected result type. The corrected linear-child case
then reached the existing `FZ-0001` lowering boundary.

### `FZ-20260421-0016`: Unencoded Reduction Result Crash

The reduction form of `ttng.tmem_load` crashes in the same unencoded tensor
verifier path when both result tensors are unencoded:

- `/tmp/tmem_compiler_boundaries_round31b/ldred_unencoded_result_fz0016.mlir`

Signature:

```text
llvm::dyn_cast<mlir::triton::gpu::DistributedEncodingTrait>(mlir::Attribute)
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

This expands `FZ-0016` to mixed-result reduction loads.

### Clean Diagnostics and Harness Boundaries

Clean diagnostics:

- `ldred_i64_clean_or_crash` rejects with:
  `tmem_load reduction currently requires f32 element type`.
- `scales_copy_wrong_src_space` rejects TMEM source for `ttng.tmem_copy` with:
  `The source must be a shared memory buffer`.
- `scales_copy_shared_to_tmem_boundary` rejects the noncanonical scale shared
  layout with the existing clean unsupported tcgen05.copy descriptor-planning
  diagnostic.
- The first `scales_dynamic_index_load` rejects a wrong expected child layout:
  `result memdesc type does not match inferred type`.

Harness-only boundary:

- `scales_load_boundary` verifies and optimizes cleanly, but the full
  allocation+LLVM conversion pipeline reports illegal `tt.return` for the
  tensor-returning `tt.func`. This is a probe-shape limitation, not a TMEM
  backend finding.

## Next Leads

- Generate verifier-only reduction probes where only one of the two reduction
  results is unencoded, to localize which result drives `FZ-0016`.
- Keep scale dynamic-index child types linear in future probes; the verifier
  does not infer `tensor_memory_scales_encoding<>` for indexed children of
  arbitrary linear scale parents.
