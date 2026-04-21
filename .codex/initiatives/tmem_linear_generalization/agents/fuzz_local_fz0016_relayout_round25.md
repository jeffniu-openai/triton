# Round 25 Local: FZ-0016 Relayout Baseline

Date: 2026-04-21 14:25 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

This local compiler-only baseline reran the original
`test/Conversion/relayout_tritongpu.mlir` command after Round 24/25 narrowed
`FZ-20260421-0016`, to confirm the current red test maps to that bucket and
not to a new compiler failure.

## Command

```bash
BUILD_DIR=$(PYTHONPATH='./python' python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
"$BUILD_DIR/bin/triton-opt" \
  test/Conversion/relayout_tritongpu.mlir \
  -split-input-file \
  -convert-triton-to-tritongpu='target=cuda:100 num-warps=4 enable-source-remat=true' \
  -relayout-tritongpu \
  --mlir-disable-threading \
  >/tmp/tmem_r25_relayout_expected_fz0016.out \
  2>/tmp/tmem_r25_relayout_expected_fz0016.err
```

Result:

```text
EXIT:134
dyn_cast on a non-existent value
```

The stack includes:

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

Artifacts:

```text
/tmp/tmem_r25_relayout_expected_fz0016.out
/tmp/tmem_r25_relayout_expected_fz0016.err
```

## Classification

This is existing `FZ-20260421-0016`: parse/verifier-time TMEM operand-layout
assertion on an unencoded tensor operand flowing into
`verifyTMEMOperand`/`toLinearEncoding`.

No new independent `FZ-*` bucket was found.
