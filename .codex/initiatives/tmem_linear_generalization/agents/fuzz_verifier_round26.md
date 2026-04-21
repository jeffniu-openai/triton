# Round 26 Lane: TMEM Verifier Robustness

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler fixes.
- Repo edit scope: this report only.

## Summary

Round 26 focused on `FZ-20260421-0016` and adjacent verifier robustness for
`ttng.tmem_alloc`, `ttng.tmem_load`, `ttng.tmem_store`, and
`ttng.tmem_copy`.

The fuzzing did not find a new independent FZ bucket. It did broaden the known
`FZ-20260421-0016` diagnosis: the same unencoded tensor layout assertion is
reachable through all operand/result verifier paths that call
`verifyTMEMOperand`, not just `ttng.tmem_alloc`.

Observed crash class:

```text
dyn_cast on a non-existent value
TritonGPUDialect::toLinearLayout
TritonGPUDialect::toLinearEncoding
toLinearEncoding(RankedTensorType)
computeTMemLdStEncodingInfoImpl
computeTMemLdStEncodingInfo
isDistributedLayoutTMemCompatible
verifyTMEMOperand
TMEM{Alloc,Load,Store}Op::verify
parseAsmSourceFile
```

Clean boundary checks stayed clean for:

- encoded-but-unsupported register layouts;
- wrong load/store shapes;
- `ttng.tmem_copy` wrong source/destination memory spaces;
- `ttng.tmem_copy` shape mismatch;
- malformed/missing memdesc encoding syntax;
- tensor-memory memdescs using non-TMEM encodings;
- rank-1/rank-3 forms rejected before the unencoded-layout compatibility path.

## Build Gate

Required first command:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Artifacts

Primary generated corpus:

```text
/tmp/tmem_verifier_round26/
/tmp/tmem_verifier_round26/results.json
```

Follow-up generated corpus:

```text
/tmp/tmem_verifier_round26_extra/
/tmp/tmem_verifier_round26_extra/results.json
```

Representative minimal repros:

```text
/tmp/tmem_verifier_round26/alloc_unencoded_f32_standard.mlir
/tmp/tmem_verifier_round26/load_unencoded_f32_standard.mlir
/tmp/tmem_verifier_round26/store_unencoded_f32_standard.mlir
/tmp/tmem_verifier_round26_extra/load_unencoded_i8_scales.mlir
/tmp/tmem_verifier_round26_extra/store_unencoded_i8_scales.mlir
/tmp/tmem_verifier_round26_extra/load_unencoded_f32_linear.mlir
/tmp/tmem_verifier_round26_extra/store_unencoded_f32_linear.mlir
```

Representative command:

```bash
/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_verifier_round26/load_unencoded_f32_standard.mlir \
  -split-input-file --mlir-disable-threading
```

Result:

```text
EXIT:-6
dyn_cast on a non-existent value
```

## Matrix Results

Primary corpus result counts:

```text
ABORT_DYN_CAST: 6
CLEAN_DIAG: 12
PASS: 2
```

Follow-up corpus result counts:

```text
ABORT_DYN_CAST: 6
CLEAN_DIAG: 6
```

Important rows:

| Case | Result | Classification |
| --- | --- | --- |
| `ttng.tmem_alloc` from unencoded `tensor<128x128xf32>` to standard TMEM | abort | Existing `FZ-20260421-0016` |
| `ttng.tmem_alloc` from unencoded `tensor<1x1xi8>` to scales TMEM | abort | Existing `FZ-20260421-0016` |
| `ttng.tmem_alloc` from unencoded `tensor<128x128xi1>` to standard TMEM | abort | Existing `FZ-20260421-0016`; dtype broadening |
| `ttng.tmem_load` to unencoded `tensor<128x128xf32>` | abort | Existing `FZ-20260421-0016`; load verifier path |
| `ttng.tmem_load` to unencoded `tensor<128x128xf16>` | abort | Existing `FZ-20260421-0016`; dtype broadening |
| `ttng.tmem_load` to unencoded `tensor<128x128xi32>` | abort | Existing `FZ-20260421-0016`; dtype broadening |
| `ttng.tmem_load` from scales TMEM to unencoded `tensor<128x128xi8>` | abort | Existing `FZ-20260421-0016`; scales load path |
| `ttng.tmem_load` from linear TMEM to unencoded `tensor<128x128xf32>` | abort | Existing `FZ-20260421-0016`; linear load path |
| `ttng.tmem_store` from unencoded `tensor<128x128xf32>` | abort | Existing `FZ-20260421-0016`; store verifier path |
| `ttng.tmem_store` from unencoded `tensor<128x128xf16>` | abort | Existing `FZ-20260421-0016`; dtype broadening |
| `ttng.tmem_store` from unencoded `tensor<128x128xi8>` to scales TMEM | abort | Existing `FZ-20260421-0016`; scales store path |
| `ttng.tmem_store` from unencoded `tensor<128x128xf32>` to linear TMEM | abort | Existing `FZ-20260421-0016`; linear store path |
| Encoded standard load/store controls | pass | Expected verifier acceptance |
| Encoded scales load/store with generated blocked layout | clean unsupported-register-layout diagnostic | Clean boundary |
| Encoded bad-layout alloc control | clean unsupported-register-layout diagnostic | Clean boundary |
| Load/store wrong shape | clean shape mismatch diagnostic | Clean boundary |
| `ttng.tmem_copy` source TMEM instead of shared | clean source-must-be-shared diagnostic | Clean boundary |
| `ttng.tmem_copy` destination shared instead of TMEM | clean expected-TMEM-descriptor diagnostic | Clean boundary |
| `ttng.tmem_copy` source/destination shape mismatch | clean shape mismatch diagnostic | Clean boundary |
| Tensor-memory memdesc with `#ttg.blocked` encoding | clean memdesc layout diagnostic | Clean boundary |
| Tensor-memory memdesc with shared encoding | clean memory-space mismatch diagnostic | Clean boundary |
| Missing memdesc encoding syntax | clean parser diagnostic | Clean boundary |
| Rank-1 standard TMEM load/store attempts | clean rank diagnostic | Clean boundary |
| Rank-3 linear TMEM load/store attempts | clean 2D tensor diagnostic | Clean boundary |

## Classification

No new independent FZ bucket.

`FZ-20260421-0016` should be tracked as a shared verifier robustness issue in
the `verifyTMEMOperand` / `isDistributedLayoutTMemCompatible` path for
unencoded tensor operands/results. The bug affects `ttng.tmem_alloc`,
`ttng.tmem_load`, and `ttng.tmem_store`; `ttng.tmem_copy` was not affected by
this exact path in the Round 26 probes because it verifies memdesc operands and
hits clean memory-space/shape/encoding diagnostics instead.

Expected eventual repair direction after the active fuzzing campaign stops:

- guard the compatibility path before calling `ttg.toLinearEncoding` on a
  tensor type without an encoding;
- either allow pre-relayout unencoded tensors through the verifier stage where
  that is semantically valid, or emit a typed diagnostic that encoded tensors
  are required at that IR stage;
- preserve the clean diagnostics cataloged above.

Backend/compiler repair remains deferred by the active fuzzing campaign.
