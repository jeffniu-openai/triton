# Round 24 Lane BG: FZ-0016 TMEM Alloc Verifier Minimization

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler repairs.
- Repo edit scope: this report only.

## Summary

`FZ-20260421-0016` is not specific to the full
`test/Conversion/relayout_tritongpu.mlir` file and does not require
`-relayout-tritongpu` or `-convert-triton-to-tritongpu`.

The smallest stable trigger is a `ttng.tmem_alloc` whose operand tensor has no
layout encoding. During parse-time verification, `TMEMAllocOp::verify` calls
`verifyTMEMOperand`, which calls `computeTMemLdStEncodingInfo` and eventually
`ttg.toLinearEncoding` on the unencoded operand tensor type. That path asserts
with `dyn_cast on a non-existent value` instead of accepting pre-relayout IR or
emitting a typed verifier diagnostic.

This lane broadens the bucket beyond the original tensor-memory-scales row:

- unencoded `i8` operands abort for standard TMEM, linear TMEM, and scales
  TMEM results;
- unencoded `f16`/`f32` operands abort for standard and linear TMEM results;
- invalid `f16`/`f32` scales rows correctly produce the existing clean
  `bitwidth must be 8` diagnostic;
- encoded operand tensors do not hit this assertion in this probe and instead
  produce ordinary operand-compatibility diagnostics for the generated layouts;
- the same classification holds with parse-only, `-relayout-tritongpu`, and
  `-convert-triton-to-tritongpu ... -relayout-tritongpu` pipelines, confirming
  this is parse/verifier-time rather than a relayout pass crash.

No backend/compiler code was changed.

## Build Gate

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Compiler tool gate:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
```

Result:

```text
ninja: no work to do.
```

## Minimal Reproducer

Path:

```text
/tmp/tmem_fz0016_round24/minimal_unencoded_scales_i8.mlir
```

Contents:

```mlir
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

tt.func @minimal() {
  %cst = arith.constant dense<0> : tensor<128x128xi8>
  %result = ttng.tmem_alloc %cst : (tensor<128x128xi8>) -> !ttg.memdesc<128x128xi8, #tmem_scales, #ttng.tensor_memory>
  tt.return
}
```

Parse-only reproducer:

```bash
/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_fz0016_round24/minimal_unencoded_scales_i8.mlir \
  -split-input-file --mlir-disable-threading \
  >/tmp/tmem_fz0016_round24_minimal.out \
  2>/tmp/tmem_fz0016_round24_minimal.err
```

Result:

```text
EXIT:134
dyn_cast on a non-existent value
```

The stack consistently includes:

```text
TritonGPUDialect::toLinearLayout
TritonGPUDialect::toLinearEncoding
toLinearEncoding(RankedTensorType)
computeTMemLdStEncodingInfoImpl
computeTMemLdStEncodingInfo
isDistributedLayoutTMemCompatible
verifyTMEMOperand
TMEMAllocOp::verify
parseAsmSourceFile
```

The original relayout-style pipeline fails the same way:

```bash
/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_fz0016_round24/minimal_unencoded_scales_i8.mlir \
  -split-input-file \
  -convert-triton-to-tritongpu='target=cuda:100 num-warps=4 enable-source-remat=true' \
  -relayout-tritongpu \
  --mlir-disable-threading \
  >/tmp/tmem_fz0016_round24_minimal_convert.out \
  2>/tmp/tmem_fz0016_round24_minimal_convert.err
```

Result:

```text
EXIT:134
dyn_cast on a non-existent value
```

## Fuzz Matrix

The generated probe directory is:

```text
/tmp/tmem_fz0016_round24/
```

Full raw results:

```text
/tmp/tmem_fz0016_round24/results.json
```

Pipelines:

- `parse_only`: `triton-opt <file> -split-input-file --mlir-disable-threading`
- `relayout_only`: `triton-opt <file> -split-input-file -relayout-tritongpu --mlir-disable-threading`
- `convert_relayout`: `triton-opt <file> -split-input-file -convert-triton-to-tritongpu='target=cuda:100 num-warps=4 enable-source-remat=true' -relayout-tritongpu --mlir-disable-threading`

Each row was tested for immutable and mutable memdesc results. The table below
compresses both mutability variants because they classified identically.

| Operand element | Operand encoding | TMEM result encoding | Result |
| --- | --- | --- | --- |
| `i8` | unencoded | standard sugar | `ABORT_DYN_CAST` in all three pipelines |
| `i8` | unencoded | tensor-memory-linear | `ABORT_DYN_CAST` in all three pipelines |
| `i8` | unencoded | tensor-memory-scales | `ABORT_DYN_CAST` in all three pipelines |
| `f16` | unencoded | standard sugar | `ABORT_DYN_CAST` in all three pipelines |
| `f16` | unencoded | tensor-memory-linear | `ABORT_DYN_CAST` in all three pipelines |
| `f16` | unencoded | tensor-memory-scales | clean `bitwidth must be 8` diagnostic |
| `f32` | unencoded | standard sugar | `ABORT_DYN_CAST` in all three pipelines |
| `f32` | unencoded | tensor-memory-linear | `ABORT_DYN_CAST` in all three pipelines |
| `f32` | unencoded | tensor-memory-scales | clean `bitwidth must be 8` diagnostic |
| `i8`/`f16`/`f32` | encoded `#ttg.linear` | standard/linear/scales | clean operand-compatibility or bitwidth diagnostics |
| `i8`/`f16`/`f32` | encoded `#ttg.blocked` | standard/linear/scales | clean operand-compatibility or bitwidth diagnostics |

## Shape Minimization

Additional shape fuzzing was written under:

```text
/tmp/tmem_fz0016_round24_shapes/
/tmp/tmem_fz0016_round24_shapes/shape_results.json
```

Important shape outcomes:

| Shape | Standard TMEM result with unencoded operand | Scales TMEM result with unencoded `i8` operand |
| --- | --- | --- |
| `1x1` | clean canonicalization diagnostic: shape smaller than block | `ABORT_DYN_CAST` |
| `1x8` | clean canonicalization diagnostic | `ABORT_DYN_CAST` |
| `8x8` | clean canonicalization diagnostic | `ABORT_DYN_CAST` |
| `16x16` | clean canonicalization diagnostic | `ABORT_DYN_CAST` |
| `64x64` | clean canonicalization diagnostic | `ABORT_DYN_CAST` |
| `128x8` | `ABORT_DYN_CAST` | `ABORT_DYN_CAST` |
| `128x128` | `ABORT_DYN_CAST` | `ABORT_DYN_CAST` |

The smallest observed scales crash is therefore `tensor<1x1xi8>` into
`#ttng.tensor_memory_scales_encoding<>`, but the existing
`relayout_tritongpu.mlir` row remains a better representative because it uses
the shape already present in upstream lit coverage.

## Classification

`FZ-20260421-0016`: parse/verifier-time TMEM alloc operand-layout crash.

This is a verifier robustness gap in the TMEM operand-compatibility path. The
code assumes the operand tensor already has a distributed layout encoding when
checking whether it is compatible with the destination TMEM memdesc. That
assumption is false for pre-relayout Triton IR and for existing
`relayout_tritongpu.mlir` coverage, where unencoded tensor operands are meant
to be converted before or during relayout.

Expected eventual repair direction after discovery stops:

- guard `verifyTMEMOperand` / `isDistributedLayoutTMemCompatible` against
  unencoded operand tensors before calling `toLinearEncoding`;
- either allow unencoded pre-relayout operands through verification, or emit a
  typed verifier diagnostic explaining that encoded operands are required at
  that IR stage;
- keep existing clean diagnostics such as invalid scales bitwidth and
  impossible standard TMEM sugar shapes.

Backend/compiler repair remains deferred by the active fuzzing campaign.
