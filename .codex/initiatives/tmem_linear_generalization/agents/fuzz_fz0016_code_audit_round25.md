# Round 25 Lane BK: FZ-0016 Verifier Boundary Code Audit

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler repairs.
- Repo edit scope: this report only.

## Summary

`FZ-20260421-0016` is a shared TMEM operand-verifier boundary bug, not just an
alloc-initializer issue.

The audited code path deliberately treats unencoded register tensors as passing
the first precondition check, but the caller immediately continues into TMEM
layout compatibility planning. That path eventually calls
`ttg::toLinearEncoding(RankedTensorType)` on the unencoded tensor type and
asserts in LLVM casting with `dyn_cast on a non-existent value`.

Confirmed affected operations:

- `ttng.tmem_alloc` with an unencoded tensor initializer.
- `ttng.tmem_load` with an unencoded tensor result.
- `ttng.tmem_store` with an unencoded tensor source.

`ttng.tmem_copy` does not have an analogous unencoded register tensor operand:
its verifier consumes shared-memory and tensor-memory `memdesc` types. The
audited copy path uses encoded memdesc layouts (`toLinearLayout(srcTy)` and
destination TMEM query selection), so this specific unencoded-register crash
does not directly apply to copy operands.

## Build Gates

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Compiler tool gate:

```bash
BUILD_DIR=$(PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
ninja -C "$BUILD_DIR" triton-opt
```

Result:

```text
ninja: no work to do.
```

## Code Audit

Primary verifier path:

- `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:1388`
  `verifyTMEMOperandPreconditions` checks rank and explicitly returns success
  for `!type.getEncoding()` at lines `1391-1394`.
- `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:1417`
  `verifyTMEMOperand` calls the precondition helper, then immediately calls
  `isDistributedLayoutTMemCompatible(op, type, memdesc)` at lines `1423-1424`
  even when the tensor has no encoding.
- `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp:4620`
  `isDistributedLayoutTMemCompatible` calls
  `computeTMemLdStEncodingInfo(tensorType, memType, maxnreg)`.
- `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp:9229`
  `computeTMemLdStEncodingInfo` derives a TMEM memory layout and then calls
  `computeTMemLdStEncodingInfoImpl` at lines `9296-9297`.
- `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp:7987`
  `computeTMemLdStEncodingInfoImpl` constructs the register layout with
  `squeezeTrivialBlock(toLinearEncoding(regTy).getLinearLayout())`.
- `lib/Dialect/TritonGPU/IR/Dialect.cpp:65`
  `toLinearEncoding(RankedTensorType)` forwards `type.getEncoding()` into
  `TritonGPUDialect::toLinearEncoding`; for an unencoded tensor this is null,
  which triggers the assertion.

Operation-specific ownership:

- `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:1779`
  `TMEMAllocOp::verify` calls `verifyTMEMOperand` for `getSrc()` at
  lines `1782-1785`, so unencoded alloc initializers hit the bug.
- `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:1581`
  `TMEMLoadOp::verify` calls `verifyTMEMOperand` for non-reduction load
  results at lines `1601-1602`, so unencoded load results hit the bug.
  Reduction loads only call the precondition helper at lines `1597-1600`, but
  later reduction-specific checks also call `toLinearLayout(regTy)` and should
  be treated as part of the same boundary when fuzzing unencoded reduction
  results.
- `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:1558`
  `TMEMStoreOp::verify` calls `verifyTMEMOperand` for the source tensor at
  lines `1564-1566`, so unencoded store sources hit the bug.
- `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:1821`
  `TMEMCopyOp::verify` does not call `verifyTMEMOperand`; it casts the source
  and destination to `MemDescType`, checks shared/tensor-memory encodings, then
  calls `toLinearLayout(srcTy)` and `selectTMemCopyPhysicalQuery(getDst(), ...)`
  at lines `1847-1856`. This is a memdesc-layout verifier, not an unencoded
  register-tensor verifier.

## MLIR Probes

Probe directories:

```text
/tmp/tmem_fz0016_round25/
/tmp/tmem_fz0016_round25_mod/
```

The first directory contains parse-only smoke probes. The second repeats
load/store probes inside a module with:

```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32}
```

Commands:

```bash
BUILD_DIR=$(PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
for f in /tmp/tmem_fz0016_round25/*.mlir; do
  name=$(basename "$f" .mlir)
  "$BUILD_DIR/bin/triton-opt" "$f" -split-input-file --mlir-disable-threading \
    >"/tmp/tmem_fz0016_round25/${name}.out" \
    2>"/tmp/tmem_fz0016_round25/${name}.err"
  echo "CASE $name EXIT:$?"
done

for f in /tmp/tmem_fz0016_round25_mod/*.mlir; do
  name=$(basename "$f" .mlir)
  "$BUILD_DIR/bin/triton-opt" "$f" -split-input-file --mlir-disable-threading \
    >"/tmp/tmem_fz0016_round25_mod/${name}.out" \
    2>"/tmp/tmem_fz0016_round25_mod/${name}.err"
  echo "CASE $name EXIT:$?"
done
```

Results:

| Probe | Result | Classification |
| --- | --- | --- |
| `alloc_standard_unencoded.mlir` | exit `134` | aborts in `TMEMAllocOp::verify` through `verifyTMEMOperand` |
| `alloc_scales_unencoded.mlir` | exit `134` | aborts in `TMEMAllocOp::verify` through `verifyTMEMOperand` |
| `load_unencoded.mlir` | exit `134` | aborts in `TMEMLoadOp::verify` through `verifyTMEMOperand` |
| `store_unencoded.mlir` | exit `134` | aborts in `TMEMStoreOp::verify` through `verifyTMEMOperand` |
| `load_encoded.mlir` | exit `1` | clean diagnostic for missing module `ttg.num-warps` context |
| `store_encoded.mlir` | exit `1` | clean diagnostic for missing module `ttg.num-warps` context |
| `load_encoded_with_attrs.mlir` | exit `0` | encoded control parses cleanly |
| `store_encoded_with_attrs.mlir` | exit `0` | encoded control parses cleanly |
| `load_unencoded_with_attrs.mlir` | exit `134` | abort remains after adding module attributes |
| `store_unencoded_with_attrs.mlir` | exit `134` | abort remains after adding module attributes |

Representative stack excerpts:

```text
toLinearEncoding(RankedTensorType) /root/code/triton/lib/Dialect/TritonGPU/IR/Dialect.cpp:69
computeTMemLdStEncodingInfoImpl(...) /root/code/triton/lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp:7988
computeTMemLdStEncodingInfo(...) /root/code/triton/lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp:9297
isDistributedLayoutTMemCompatible(...) /root/code/triton/lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp:4624
verifyTMEMOperand(...) /root/code/triton/lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:1423
```

Operation-specific terminal frames:

```text
TMEMAllocOp::verify() /root/code/triton/lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:1782
TMEMLoadOp::verify()  /root/code/triton/lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:1601
TMEMStoreOp::verify() /root/code/triton/lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp:1564
```

## Classification

No new independent `FZ-*` bucket is needed, but `FZ-20260421-0016` should be
broadened:

```text
FZ-20260421-0016: TMEM load/store/alloc verifier crashes when asked to verify
unencoded register tensors. The shared `verifyTMEMOperand` path lets unencoded
tensors pass preconditions, then runs direct TMEM compatibility planning that
requires a distributed register encoding and asserts in `toLinearEncoding`.
```

The verifier ownership boundary should be decided explicitly before repair:

- If unencoded pre-relayout Triton IR is legal at this stage, then
  `verifyTMEMOperand` should not call direct TMEM compatibility planning until
  after a distributed layout has been assigned.
- If encoded operands are required for `ttng.tmem_*` ops at parse time, the
  verifier should emit a typed diagnostic before reaching
  `isDistributedLayoutTMemCompatible`.

The existing `test/Conversion/relayout_tritongpu.mlir` expectations imply the
first interpretation for conversion tests: relayout/conversion is expected to
own layout assignment for unencoded `tmem_alloc`, `tmem_load`, and
`tmem_store` operands. In either interpretation, the current assertion is a
verifier robustness gap.

Backend/compiler code was not changed.
