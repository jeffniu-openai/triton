# FP4 padded shared-memory compression exploration - 2026-04-28

## Purpose

Investigate whether Blackwell mixed-precision fp8/fp4 `tcgen05.mma` kernels can avoid the shared-memory waste imposed by `fp4_padded` operand layouts. The concrete target is to feed fp4 operands to mixed fp8/fp4 MMA while using materially less shared memory than the current padded shared layout, without staging through registers.

This is part of the broader TMEM linear-layout/generalization initiative because the most plausible non-register path uses TMEM as an operand buffer and `tcgen05.cp` as a shared-to-TMEM transform.

## User-level requirements

- Exhaust all plausible paths, or prove why each path is not viable.
- Check actual output correctness for PTX probes, not only whether instructions trap.
- Prefer hardware/ISA facts and measured behavior over assumptions from the current compiler implementation.
- Record findings durably in the TMEM initiative docs.

## Existing context to preserve

The five incremental main-slice branches are set aside for now and should be remembered later:

1. `codex/tmem-main-01-alloc-rounding-lit`
2. `codex/tmem-main-02-alloc-contract-lit`
3. `codex/tmem-main-03-commit-contract-lit`
4. `codex/tmem-main-04-barrier-commit-verify`
5. `codex/tmem-main-05-i8-mma-clean-error`

## ISA facts and hypotheses

### Documented facts

- `tcgen05` shared-memory matrix descriptors encode matrix start address, leading-dimension byte offset, and stride-dimension byte offset with `matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4`; the PTX ISA document states that all three must be 16-byte aligned.
- For `.kind::mxf8f6f4`, 4-bit and 6-bit operands in shared memory are contiguously packed with padding. For `.kind::mxf4`/`.kind::mxf4nvf4`, shared-memory fp4 packs two elements per byte without padding; this exploration is about the `.kind::mxf8f6f4` mixed case.
- `tcgen05.cp` supports optional decompression syntax `tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc` with `.dst_fmt = .b8x16` and `.src_fmt = .b4x16_p64` or `.b6x16_p32`.
- The PTX tensor-map sub-byte type `.b4x16_p64` represents sixteen packed 4-bit elements followed by 64 bits of padding in shared memory; the padded region is documented as uninitialized for tensor copy expansion.

### Hardware facts already measured by the initiative

- Prior TMEM address-alignment probes on GB300 showed that several TMEM address restrictions are runtime hardware facts rather than always-static compiler facts.
- For TMEM address operands, prior probes found: plain `tcgen05.ld/st 32x32b.x64.b32` tolerated offsets 0..15, `ld.red 32x32b` required even offsets, `tcgen05.cp.cta_group::1.128x256b` destination required offsets multiple of 4, MMA accumulator D tolerated even offsets, and MMA A-in-TMEM f16 operands required offsets multiple of 4.
- Those probes did not answer whether a shared-memory matrix descriptor for fp4-padded MMA accepts a base address offset by 8 bytes.

## Candidate paths

### Path A: shared descriptor +8 interleaving

Exploit the unused half of each 16-byte fp4-padded shared chunk by placing a second tile in the bytes at offset +8 and issuing a second MMA using a shared-memory descriptor whose matrix start address is offset by 8 bytes.

Open questions:

- Does `tcgen05.mma.kind::mxf8f6f4` accept an 8-byte-shifted shared matrix descriptor in hardware, despite the PTX descriptor alignment statement and the descriptor encoding dropping the low 4 address bits?
- If it accepts the descriptor, do the MMA data-path semantics actually read the shifted 8-byte half as the first packed fp4 payload, or does descriptor encoding make +8 indistinguishable from +0?
- Can a dense global/TMA layout feed an interleaved shared layout efficiently without using `.b4x16_p64` expansion that leaves the padded half uninitialized?

Likely implementation route if viable:

1. Convert global fp4 bytes into an interleaved layout where tile0 occupies bytes 0..7 and tile1 occupies bytes 8..15 in every 16-byte group.
2. Use a dense shared-memory load/copy into ordinary shared storage, not tensor-map `.b4x16_p64` expansion.
3. Build/patch two shared descriptors pointing at base +0 and base +8 for two MMA uses.

### Path B: `tcgen05.cp` dense-to-TMEM transform

Use `tcgen05.cp` to transform a packed/padded shared fp4 source into the TMEM operand format expected by mixed-precision MMA, avoiding register unpack/repack.

Open questions:

- Does the documented `.b8x16.b4x16_p64` decompression produce the same TMEM byte/container layout expected by `tcgen05.mma.kind::mxf8f6f4` when A is supplied from TMEM?
- Can the source shared layout be fed by a dense/coalesced TMA path that does not waste shared memory, or does the cp source format itself require the same 16-byte padded source chunks?
- Are the available cp shapes (`128x256b`, `4x256b`, `128x128b`, `64x128b`, `32x128b`) sufficient to cover useful fp4 MMA operand tiles without overcopying or requiring missing destination masks?
- Does the existing branch clean-negative for mixed fp4 A-in-TMEM reflect only compiler modeling limitations, or a real ISA/data-path mismatch?

Likely implementation route if viable:

1. TMA/copy dense or pre-swizzled fp4 bytes into shared memory using a layout chosen to keep global accesses coalesced.
2. Issue `tcgen05.cp...b8x16.b4x16_p64` into a TMEM staging operand.
3. Feed that TMEM operand into `tcgen05.mma.kind::mxf8f6f4.block_scale` as A, with B still in shared memory.

### Path C: global layout only

If Path A or B works, define the global memory layout that makes the shared/TMEM staging path efficient. The two obvious candidates are:

- regular dense fp4 bytes with a lightweight shared/TMEM transform;
- Blackwell-preshuffled fp4 bytes that land directly in the target shared/TMEM access order under coalesced TMA.

This path is not independently useful unless A or B proves a legal MMA input path.

## Probe plan

1. Rebuild the branch (`make`) before running probes/tests.
2. Generate and inspect PTX for a minimal working Gluon `mxf8f6f4` scaled MMA using shared fp4-padded operands.
3. Identify where the fp4 shared descriptor base address is constructed and whether a +8 byte shift is representable in the emitted descriptor bits.
4. Build an isolated PTX/runtime probe for shared-descriptor offsets that verifies numerical output for at least base +0 and base +8.
5. Build an isolated PTX/runtime probe for `tcgen05.cp.b8x16.b4x16_p64` into TMEM, then read back or consume with MMA to validate the transformed data.
6. If either path works, characterize global/shared layout requirements and estimate shared-memory savings/performance implications.


## Running results

- 2026-04-28: Exploration started on branch `codex/tmem` at `fdf9318d2`.
- 2026-04-28: `make -j8` completed successfully before running probes.
- Probe GPU: NVIDIA GB300, compute capability 10.3 (`CUDA_VISIBLE_DEVICES=0`).

### Baseline mixed fp8/fp4 shared MMA

Generated PTX for `tmem_mma_scaled_layout_format_kernel` with `M=N=K=128`, `a_format=mxfp8`, `b_format=mxfp4`, and a linear accumulator layout. The generated kernel passed the existing torch reference check.

```text
torch_correct [-41.179779052734375, 0.5989990234375, 12.82421875,
               -23.2210693359375, -47.32763671875, 63.09716796875,
               30.0, 0.876953125] mma_count 4
```

The emitted PTX builds shared descriptors by shifting the shared byte address right by 4:

```ptx
bfe.u32  %r59, %r4, 4, 14;
cvt.u64.u32  %rd220, %r59;
add.s32  %r60, %r4, 16384;
bfe.u32  %r61, %r60, 4, 14;
cvt.u64.u32  %rd221, %r61;
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [ %r192 + 0 ], %rd212, %rd213, ...;
```

This is the core Path A issue: descriptor byte-address bits 0..3 are not represented in the shared-memory descriptor field.

### Shared descriptor +8 probes

Patched the correct `mxfp8 x mxfp4` PTX so operand B's descriptor source address changed from `base + 16384` to `base + 16392` before `bfe ..., 4, 14`. The patched kernel was launched with valid MX inputs and compared against the unpatched kernel output.

```json
{
  "b_plus8_preencode": {"exact_vs_base": true, "finite": true, "max_abs_diff": 0.0},
  "b_plus16_preencode": {"exact_vs_base": false, "finite": true, "max_abs_diff": 888.2763671875},
  "b_descfield_plus8_on_second_mma": {"exact_vs_base": false, "finite": true, "max_abs_diff": 779.0}
}
```

Repeated the same check with `a_format=mxfp4`, `b_format=mxfp8` so operand A was the fp4 shared operand.

```json
{
  "a_plus8_preencode": {"exact_vs_base": true, "finite": true, "max_abs_diff": 0.0},
  "a_plus16_preencode": {"exact_vs_base": false, "finite": true, "max_abs_diff": 899.6376953125}
}
```

Conclusion: a `+8` byte shared-memory matrix start is not a hidden legal address mode for `tcgen05.mma`; it is not representable. The descriptor encoder drops that bit, so `base + 8` is identical to `base + 0`. The next representable source start is `base + 16`.

### `tcgen05.cp.b8x16.b4x16_p64` decompression probe

Built a raw-byte `tcgen05.cp.cta_group::1.128x128b` kernel using an `int32[128,4]` shared source and `int32[128,4]` TMEM destination, then patched the opcode suffix to `.b8x16.b4x16_p64`. Input row bytes were `00 01 02 ... 0f`; output was read back as raw bytes from TMEM.

```json
{
  "base": {
    "row0_bytes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  },
  "b4decomp": {
    "row0_bytes": [0, 0, 4, 0, 8, 0, 12, 0, 16, 0, 20, 0, 24, 0, 28, 0]
  }
}
```

This proves that `.b8x16.b4x16_p64` consumes the first 8 bytes of each 16-byte shared source group as 16 fp4 nibbles, ignores the 8-byte padding half, and writes the fp4 values into 8-bit TMEM containers as `nibble << 2` in low/high nibble order per source byte. That matches the `.kind::mxf8f6f4` tensor-memory contract where 4-bit elements are placed inside 8-bit containers.

### `tcgen05.cp` source descriptor +8 probe

Patched the `.b8x16.b4x16_p64` cp probe so the shared source address was shifted before descriptor encoding.

```json
{
  "base": {"same_as_base": true, "row0": [0, 0, 4, 0, 8, 0, 12, 0, 16, 0, 20, 0, 24, 0, 28, 0], "maxdiff": 0},
  "src_plus8_preencode": {"same_as_base": true, "row0": [0, 0, 4, 0, 8, 0, 12, 0, 16, 0, 20, 0, 24, 0, 28, 0], "maxdiff": 0},
  "src_plus16_preencode": {"same_as_base": false, "row0": [0, 0, 4, 0, 8, 0, 12, 0, 16, 0, 20, 0, 24, 0, 28, 0], "maxdiff": 60}
}
```

Conclusion: `tcgen05.cp` source descriptors have the same +8 non-representability as MMA shared descriptors. They can read a padded `.b4x16_p64` source group and decompress it into TMEM, but they cannot address a second fp4 tile placed in the padding half at base +8.

## Current conclusion

No no-register path found so far can double-buffer two fp4 operands in one `.b4x16_p64` shared-memory footprint:

- Path A is blocked by descriptor representation: shared matrix descriptors cannot encode byte offsets 8..15.
- Path B can use `tcgen05.cp.b8x16.b4x16_p64` to produce the right TMEM fp4-in-8-bit-container format, but the source format itself still requires a 16-byte group containing 8 bytes of payload and 8 bytes of padding. The cp source descriptor also cannot address the padding half as a second source tile.
- A dense `.b4x16` TMA/shared layout is incompatible with the `.kind::mxf8f6f4` shared-memory operand contract. It would have to be unpacked through registers or through the documented cp decompression path, and that cp path requires `.b4x16_p64` source groups.

Potentially viable but outside the requested no-register/no-padding goal: dense fp4 shared storage can be manually unpacked through registers and written into TMEM 8-bit containers before MMA. That saves shared memory but spends register bandwidth/instructions and is not a `tcgen05.cp` no-register transform.
