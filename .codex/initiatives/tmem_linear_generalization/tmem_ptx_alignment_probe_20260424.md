# TMEM PTX Alignment Probe - 2026-04-24

## Method

Hardware: NVIDIA GB300, `CUDA_VISIBLE_DEVICES=0`.

The probes generated known-good Gluon PTX, patched tensor-memory address immediates/register offsets directly in PTX, assembled each variant with `/usr/local/cuda/bin/ptxas -arch=sm_103a`, launched the cubin through PyCUDA, and checked actual output values for every passing point. Faulting points were run in separate processes because CUDA `misaligned address` poisons the context.

For `ld.red` and some verifier-blocked templates, the base PTX was generated from upstream main `a9ced8362` so the branch verifier could not pre-decide legality. For other templates, PTX was generated from the current branch. All recorded failures below are runtime/hardware failures unless noted; `ptxas` accepted the patched instructions.

Offset values are PTX tensor-memory address offsets, i.e. hardware TMEM address columns as used in the instruction bracket operand. For sub-32-bit logical element layouts, this is not always the same thing as a logical element-column phase.

## Findings

| Instruction surface | Probe | Passing offsets in `0..15` | Failing offsets in `0..15` | Observed rule |
|---|---|---:|---:|---|
| Plain `tcgen05.ld.sync.aligned.32x32b.x64.b32` | Isolated shifted load from a wider initialized f32 parent; output compared to shifted slice | all `0..15` | none | No final-address alignment requirement observed over this range. |
| Plain `tcgen05.st.sync.aligned.32x32b.x64.b32` | Isolated shifted store into a wider f32 parent; parent readback compared at shifted slice | all `0..15` | none | No final-address alignment requirement observed over this range. |
| `tcgen05.ld/st.sync.aligned.16x64b.x32.b32` | Direct store+load roundtrip at same shifted f32 address; exact output compare | even offsets | odd offsets | Combined roundtrip requires even final address, i.e. 2 hardware columns / 64 bits. |
| `tcgen05.ld/st.sync.aligned.16x128b.x32.b32` | Direct store+load roundtrip at same shifted f32 address; exact output compare | even offsets | odd offsets | Combined roundtrip requires even final address, i.e. 2 hardware columns / 64 bits. |
| `tcgen05.ld/st.sync.aligned.16x32bx2.x16.b32` packed f16 | Direct packed f16 store+load roundtrip at same shifted address; exact output compare | all `0..15` | none | No hardware-word-address alignment requirement observed for the packed PTX address. Logical f16 subword phase is represented outside this PTX address offset. |
| `tcgen05.ld/st.sync.aligned.16x32bx2.x16.{pack,unpack}::16b.b32` unpacked f16 | Direct unpacked f16 store+load roundtrip at same shifted address; exact output compare | even offsets | odd offsets | Combined roundtrip requires even final address, i.e. 2 hardware columns / 64 bits. |
| `tcgen05.ld.red.sync.aligned.32x32b.x{2,4,8,16,32,64}.max.f32` | Wider initialized parent, shifted red-load, exact loaded values and row max compare | even offsets | odd offsets | Red-load requires even final address, i.e. 2 f32 columns / 64 bits. Not 128 bits. |
| `tcgen05.ld.red.sync.aligned.32x32b.x64.{min,max}.f32` | Wider initialized parent, shifted red-load, exact loaded values and row min/max compare | even offsets | odd offsets | Modifier does not change the observed address rule: 2 f32 columns / 64 bits. |
| `tcgen05.cp.cta_group::1.128x256b` f32 destination | Patched all copy destinations and same-address readback; exact copied output compare. Readback used `32x32b` load, which independently accepts all offsets. | multiples of 4 | all non-multiples of 4 | Copy destination requires 4 f32 columns / 128 bits. |
| `tcgen05.mma.cta_group::1.kind::f16` accumulator D | Shifted f32 accumulator parent, patched MMA D address, parent readback compared at shifted slice | even offsets | odd offsets | MMA accumulator D requires even final address, i.e. 2 f32 columns / 64 bits. |
| `tcgen05.mma.cta_group::1.kind::f16` A-in-TMEM operand | Uniform f16 A parent, patched A operand address, exact uniform output compare | multiples of 4 | all non-multiples of 4 | MMA A-in-TMEM operand requires address multiple of 4 hardware columns. |
| `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X` accumulator D | Shifted f32 accumulator parent, patched scaled-MMA D address, parent readback compared at shifted slice | even offsets | odd offsets | Scaled MMA accumulator D matches plain MMA D: 2 f32 columns / 64 bits. |
| Scaled-MMA A scale address plus corresponding scale stores | Patched A-scale TMEM stores and scaled-MMA A-scale operand together; exact output compare to baseline | even offsets | odd offsets | Combined scale-store plus scaled-MMA scale operand path requires even final address. |
| Scaled-MMA B scale address plus corresponding scale stores | Patched B-scale TMEM stores and scaled-MMA B-scale operand together; exact output compare to baseline | even offsets | odd offsets | Combined scale-store plus scaled-MMA scale operand path requires even final address. |

## Excluded / Superseded Probes

- Parent-view `16x64b` and `16x128b` isolated `ld`/`st` templates produced mismatches even at offset 0 because the test layout was not a valid correctness oracle for those atoms. They did show odd-offset traps, but the correctness-invalid rows are excluded from the table above. The direct roundtrip rows replaced them.
- Direct scaled-MMA D probes without a wider accumulator parent produced wrong values at even shifted offsets because shifted D was not a valid initialized/output window for that kernel shape. The wider-parent scaled-MMA D probe supersedes those rows.

## Compiler Implications

- The current branch rule that treats `.32x32b` f32 `ld.red` as requiring 128-bit alignment is too strict. Empirically, `.32x32b` f32 `ld.red` requires only an even final TMEM address, i.e. 64-bit alignment in f32 element-column terms.
- `tcgen05.cp.128x256b` still supports the existing 128-bit destination-address requirement.
- Plain `.32x32b` f32 `ld`/`st` should not inherit the `ld.red` alignment rule; shifted addresses `1..15` all returned exact data.
- MMA accumulator-D alignment appears to be 64-bit, while A-in-TMEM operand alignment is 128-bit in the probed f16 shape.
- Scale-address support should be treated separately from accumulator-D: the combined scale-store/scaled-MMA-scale path passed at even offsets and trapped at odd offsets.

## Artifacts

Raw logs and temporary scripts:

- `/tmp/tmem_align_sweep_logs/`
- `/tmp/tmem_align_direct_logs/`
- `/tmp/tmem_align_scaled_logs/`
- `/tmp/tmem_align_scaled_parent_d_logs/`
- `/tmp/tmem_ldred_num_ptx_probe_logs/`
- `/tmp/tmem_align_run_case.py`
- `/tmp/tmem_align_run_scaled.py`
- `/tmp/tmem_align_run_scaled_parent_d.py`
- `/tmp/tmem_align2_ptx/`
