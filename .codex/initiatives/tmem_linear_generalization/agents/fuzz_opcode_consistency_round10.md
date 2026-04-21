# Round 10 Lane O: TMEM Opcode/IR Consistency Fuzzing

Probe command:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_opcode_consistency_round10_probe.py
```

Scope for this lane:
- `tcgen05.ld`, `tcgen05.st`
- `tcgen05.ld.red`
- `tcgen05.cp`
- `tcgen05.mma`
- `tcgen05.mma_scaled`
- descriptor views, 1CTA/2CTA, row/column permutations, explicit variants, reduction modifiers, and `use_acc`

## Summary

I did not find a new independent `FZ-*` bucket in this lane.

What I did find:
- positive rows where PTX and LLIR opcode extraction matched exactly;
- one known `FZ-20260421-0004` fallback row where `ld.red` lowers to plain `ld` opcodes instead of a hardware reduction opcode; and
- one clean unsupported `tcgen05.copy` boundary for an incompatible permuted shared layout.

No PTX-vs-LLIR opcode disagreement showed up on the sampled positive rows.

## Representative Positive Rows

### `tcgen05.ld` / `tcgen05.st`

Direct 1CTA identity:
- case: `direct_128x64_auto`
- runtime: pass
- PTX and LLIR both emitted:

```text
tcgen05.st.sync.aligned.32x32b.x64.b32
tcgen05.ld.sync.aligned.32x32b.x64.b32
```

Explicit variant on a permuted layout:
- case: `direct_128x64_explicit_rowreverse_colevens`
- runtime: pass
- PTX and LLIR both emitted:

```text
tcgen05.st.sync.aligned.16x64b.x32.b32
tcgen05.ld.sync.aligned.16x64b.x32.b32
```

Descriptor-chain row:
- case: `descriptor_chain_128x32`
- runtime: pass
- PTX and LLIR both emitted:

```text
tcgen05.st.sync.aligned.32x32b.x32.b32
tcgen05.ld.sync.aligned.32x32b.x32.b32
```

### `tcgen05.ld.red`

Direct reduction rows:
- case: `direct_128x64_min`
- runtime: pass
- PTX and LLIR both emitted:

```text
tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32
```

- case: `direct_128x64_max_abs_nan`
- runtime: pass
- PTX and LLIR both emitted:

```text
tcgen05.ld.red.sync.aligned.32x32b.x64.max.abs.NaN.f32
```

### `tcgen05.cp`

No-scales direct row:
- case: `no_scales_1cta_128x128_swizzle32`
- runtime: pass
- PTX and LLIR both emitted repeated `tcgen05.cp.cta_group::1.128x256b`

Scaled-copy 2CTA row:
- case: `scaled_copy_2cta_via_helper`
- runtime: pass
- PTX and LLIR both emitted repeated `tcgen05.cp.cta_group::2.warpx4.32x128b`

### `tcgen05.mma`

Plain MMA row:
- case: `f16_1cta_helper`
- runtime: pass
- PTX and LLIR both emitted repeated `tcgen05.mma.cta_group::1.kind::f16`

### `tcgen05.mma_scaled`

1CTA scaled-MMA row:
- case: `mxfp8_root_acc_init0`
- runtime: pass
- PTX and LLIR both emitted repeated:

```text
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
```

2CTA use-acc scaled-MMA row:
- case: `nvfp4_twocta_acc_init1` in the temporary probe
- runtime: pass
- PTX and LLIR both emitted repeated:

```text
tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X
tcgen05.cp.cta_group::2.warpx4.32x128b
```

## Known Overlap / Boundary Classification

### Existing `FZ-20260421-0004`

The 1CTA indexed `ld.red` 256x32 parent/view row was still the known opcode fallback boundary:

- case: `twocta_indexed_256x32_known_fz0004`
- PTX and LLIR both contained plain load opcodes:

```text
tcgen05.ld.sync.aligned.32x32b.x64.b32
```

- no `tcgen05.ld.red` opcode was present in either stream
- classification: existing `FZ-20260421-0004`, not a new bug

### Clean Unsupported Boundary

The odd permuted copy boundary:

- case: `exotic_layout_boundary`
- result: clean diagnostic
- compiler note: the shared layout could not synthesize a compatible `tcgen05.copy.128x256b` plan
- classification: clean negative, not a backend regression

## Bottom Line

- New opcode-consistency issue found: none
- PTX/LLIR opcode disagreement found: none on the sampled positive rows
- Expected hardware opcode missing in a new bucket: none
- Existing bucket overlap: `FZ-20260421-0004` only
- Clean boundaries: supported and preserved

