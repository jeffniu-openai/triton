# Round 31 Compiler Boundary Fuzzing

Date: 2026-04-21 12:51 UTC
Branch: `codex/tmem`
HEAD: `8a0f6468e`
Lane: compiler-only verifier/index boundary fuzzing for TMEM.

## Scope

This lane built small MLIR probes under `/tmp` to stress verifier and lowering
boundaries around `ttg.memdesc_index`, `ttg.memdesc_subslice`,
`ttg.memdesc_reshape`, `ttg.memdesc_trans`, `ttng.tmem_alloc/load/store/copy`,
encoded versus unencoded tensors, wrong memory-space operands, and 64-bit
element types.

No backend or test source was edited for this lane.

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
# Generated 18 focused MLIR probes under:
# /tmp/tmem_compiler_boundaries_round31
PY
```

Compiler matrix:

```bash
BUILD=/root/code/triton/build/cmake.linux-aarch64-cpython-3.12
OPT=$BUILD/bin/triton-opt

for f in /tmp/tmem_compiler_boundaries_round31/*.mlir; do
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

- `/tmp/tmem_compiler_boundaries_round31/results.tsv`
- `/tmp/tmem_compiler_boundaries_round31/results_round2.tsv`
- `/tmp/tmem_compiler_boundaries_round31/<case>.<verify|optimize|lower>.log`

## Counts

Across 18 MLIR cases and three compiler modes per case:

- `18` passed.
- `23` produced clean verifier or lowering diagnostics.
- `12` aborted with assertions or stack dumps.
- `1` was a late illegal-op compiler failure initially matched by the harness
  as crash-class because it reported `failed to legalize operation`.

Cases:

```text
f64_store_fz0017
i64_load_fz0017
i8_load_boundary
idx_const_load_pass
idx_const_store_return_pass
idx_dynamic_copy_dst_fz0001
idx_dynamic_copy_src_wrong_space_boundary
idx_dynamic_load_lower_fz0001
idx_dynamic_load_return_fz0001
idx_select_store_lower_fz0001
reshape_bad_element_count
subslice_source_rank_too_few
subslice_static_oob
tmem_copy_tmem_src_wrong_space
transpose_bad_order_dup
unencoded_load_fz0016
unencoded_store_fz0016
wrong_space_tmem_load_shared
```

## Classification

No new independent `FZ-*` bucket was found.

### `FZ-20260421-0001`: Dynamic `ttg.memdesc_index`

Runtime `ttg.memdesc_index` still lacks complete lowering when the indexed
descriptor feeds live TMEM consumers.

Minimized compiler-only repros:

- `/tmp/tmem_compiler_boundaries_round31/idx_dynamic_load_return_fz0001.mlir`
- `/tmp/tmem_compiler_boundaries_round31/idx_select_store_lower_fz0001.mlir`
- `/tmp/tmem_compiler_boundaries_round31/idx_dynamic_copy_dst_fz0001.mlir`

Failure:

```text
error: failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

Positive controls:

- `idx_const_load_pass` passes verify, optimize, and lower.
- `idx_const_store_return_pass` passes verify, optimize, and lower.
- `idx_dynamic_load_lower_fz0001` passes lower when the load result is dead and
  DCE removes the consumer, so future minimizers must keep loaded values live.

### `FZ-20260421-0016`: Unencoded Tensor Verifier Crash

Unencoded tensor operands still abort in verifier-level paths for both load and
store.

Repros:

- `/tmp/tmem_compiler_boundaries_round31/unencoded_load_fz0016.mlir`
- `/tmp/tmem_compiler_boundaries_round31/unencoded_store_fz0016.mlir`

Failure signature:

```text
llvm::dyn_cast<mlir::triton::gpu::DistributedEncodingTrait>(mlir::Attribute)
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

The abort happens on plain `triton-opt <file>`, before any explicit optimize or
LLVM conversion pass.

### `FZ-20260421-0017`: 64-Bit TMEM Load/Store Verifier Crash

Encoded `i64` and `f64` TMEM non-reduction load/store now reproduce as
verifier-time process aborts, not only allocation or LLVM-conversion aborts.

Repros:

- `/tmp/tmem_compiler_boundaries_round31/i64_load_fz0017.mlir`
- `/tmp/tmem_compiler_boundaries_round31/f64_store_fz0017.mlir`

Failure signature:

```text
TensorMemoryUtils.cpp:7713: Assertion `bitwidth == 32' failed.
```

The `i8_load_boundary` control passes all three compiler modes.

## Clean Diagnostic Boundaries

These cases produced clean diagnostics rather than crashes:

- Static subslice out of bounds: `subslice must stay within the source shape`.
- Subslice rank mismatch: `offsets must have the same rank as input`.
- Duplicate transpose order: `order must be a permutation of [0, ..., rank - 1]`.
- Bad reshape target using legacy TMEM sugar:
  `cannot be canonicalized for shape [127, 128]`.
- `ttng.tmem_load` on shared memory:
  `source must be a tensor memory buffer`.
- `ttng.tmem_copy` with TMEM source:
  `The source must be a shared memory buffer`.
- Dynamic-index TMEM source to `ttng.tmem_copy` diagnoses the wrong source
  memory space before reaching the dynamic-index lowering gap.

## Next Leads

- Generate the same verifier-only matrix for `ttng.tmem_load` reduction forms
  and scale TMEM layouts, keeping return values live to avoid DCE hiding
  `memdesc_index`.
- Expand `FZ-0016` probes to every op that calls the shared TMEM operand
  verifier, especially mixed-result `ttng.tmem_load` with `redOp`.
- Expand `FZ-0017` to narrower 64-bit shapes and descriptor-view chains at
  verifier-only parse time.
