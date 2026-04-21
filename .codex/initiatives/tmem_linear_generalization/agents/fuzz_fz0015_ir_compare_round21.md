# Round 21 Lane AY: FZ-0015 IR Compare

Date: 2026-04-21 12:05 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes, no commit, no push.

## Scope

This round compared exact `FZ-20260421-0015` allocation-order contrast rows at
TTGIR, LLVM IR, and PTX:

- failing all-scales-before-real-accumulator row;
- passing accumulator-before-scales row;
- passing A-scales-before-accumulator-before-B-scales row.

The goal was to check whether the selected B-scale memdesc SSA form, SFB
address operands, descriptor-base rematerialization, allocation indices, and
selected-scale `tmem_load` side channel diverge between failing and passing
rows.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Fresh Round 21 probe:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_fz0015_ir_compare_round21_probe.py

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_fz0015_ir_compare_round21_probe.py \
  2>&1 | tee /tmp/tmem_fz0015_ir_compare_round21_probe.log
```

The probe imports the existing exact-file minimizers and writes fresh artifacts
under `/tmp/tmem_fz0015_ir_compare_round21/`.

## Runtime Results

All three rows use selector `0`, `M=N=K=128`, `mxfp8`, direct
`TensorMemoryScalesLayout` B-scale descriptors, one scaled-MMAv5 op, and a
selected B-scale side-channel `tmem_load`.

| Row | Source probe | Allocation order | Selected B-scale load | Scaled-MMAv5 result |
| --- | --- | --- | --- | --- |
| `scale_first_fail` | `/tmp/tmem_fz0015_side_channel_round19_probe.py` | `a0,a1,b0,b1,acc` | `0/512` byte mismatches | `16381/16384` mismatches, `64` NaNs, `90` Infs |
| `acc_first_pass` | `/tmp/tmem_fz0015_acc_first_round19_probe.py` | `acc,a0,a1,b0,b1` | `0/512` byte mismatches | pass, `0/16384` mismatches |
| `a_before_acc_b_pass` | `/tmp/tmem_fz0015_a_before_acc_b_round20_probe.py` | `a0,a1,acc,b0,b1` | `0/512` byte mismatches | pass, `0/16384` mismatches |

## Artifacts

Probe and log:

```text
/tmp/tmem_fz0015_ir_compare_round21_probe.py
/tmp/tmem_fz0015_ir_compare_round21_probe.log
```

Fresh IR/assembly artifacts:

```text
/tmp/tmem_fz0015_ir_compare_round21/scale_first_fail.ttgir
/tmp/tmem_fz0015_ir_compare_round21/scale_first_fail.ll
/tmp/tmem_fz0015_ir_compare_round21/scale_first_fail.ptx
/tmp/tmem_fz0015_ir_compare_round21/acc_first_pass.ttgir
/tmp/tmem_fz0015_ir_compare_round21/acc_first_pass.ll
/tmp/tmem_fz0015_ir_compare_round21/acc_first_pass.ptx
/tmp/tmem_fz0015_ir_compare_round21/a_before_acc_b_pass.ttgir
/tmp/tmem_fz0015_ir_compare_round21/a_before_acc_b_pass.ll
/tmp/tmem_fz0015_ir_compare_round21/a_before_acc_b_pass.ptx
/tmp/tmem_fz0015_ir_compare_round21/metrics.json
/tmp/tmem_fz0015_ir_compare_round21/ttgir_key.txt
/tmp/tmem_fz0015_ir_compare_round21/ll_key.txt
/tmp/tmem_fz0015_ir_compare_round21/ptx_key.txt
```

## TTGIR Observations

All three rows preserve the same important selected B-scale SSA shape:

```mlir
%24 = arith.select %23, %b_sel_17, %b_sel
%selected_probe = ttng.tmem_load %24
%30 = ttng.tc_gen5_mma_scaled ..., %acc[], %a0, %24, ...
```

The selected descriptor is therefore not dropped or rewritten away before
scaled-MMAv5. The side-channel load uses the same selected memdesc value that
feeds the B-scale operand, and the side-channel is correct in all rows.

The TTGIR discriminator is allocation order only:

```text
scale_first_fail:    a0, a1, b_sel, b_sel_17, acc
acc_first_pass:      acc, a0, a1, b_sel, b_sel_17
a_before_acc_b_pass: a0, a1, acc, b_sel, b_sel_17
```

## LLVM IR Observations

The failing row lowers to a suspicious low-offset operand pattern:

```llvm
%1309 = add i32 %14, 4
%1310 = add i32 %14, 8
%1311 = add i32 %14, 12
%.v = select i1 %.not, i32 %1310, i32 %1311
tail call void asm sideeffect "... tcgen05.mma ...",
  ... (i32 %1309, ..., i32 %14, i32 %.v, ...)
```

Interpreting the inline-asm operands by the
`createScaledGen5MMA(..., acc, ..., scaleA, scaleB, ...)` call order, the
failing row passes:

- accumulator destination address: `%14 + 4`;
- A-scale address: `%14`;
- selected B-scale address: `%14 + 8` or `%14 + 12`.

The passing accumulator-first row uses a separated accumulator base and moves
scale operands above it:

```llvm
%1313 = add i32 %14, 128
%1314 = add i32 %14, 136
%1315 = add i32 %14, 140
%.v = select i1 %.not, i32 %1314, i32 %1315
tail call void asm sideeffect "... tcgen05.mma ...",
  ... (i32 %14, ..., i32 %1313, i32 %.v, ...)
```

The passing A-before-acc-B row also separates the selected B scales from the
accumulator:

```llvm
%1309 = add i32 %14, 8
%1310 = add i32 %14, 136
%1311 = add i32 %14, 140
%.v = select i1 %.not, i32 %1310, i32 %1311
tail call void asm sideeffect "... tcgen05.mma ...",
  ... (i32 %1309, ..., i32 %14, i32 %.v, ...)
```

This makes the failure look less like an `arith.select` lowering problem by
itself and more like a TMEM allocation/address-space representation problem for
scaled-MMAv5 operands when the real accumulator is allocated after the scale
descriptors.

## PTX Observations

PTX preserves the same distinction:

```ptx
// scale_first_fail
selp.b32 %r46, 8, 12, %p4;
add.s32 %r48, %r188, 4;
@%p9 tcgen05.mma... [ %r48 + 0 ], ..., [ %r188 + 0 ], [ %r50 + 0 ], %p8;

// acc_first_pass
selp.b32 %r47, 136, 140, %p4;
add.s32 %r50, %r189, 128;
@%p9 tcgen05.mma... [ %r189 + 0 ], ..., [ %r50 + 0 ], [ %r51 + 0 ], %p8;

// a_before_acc_b_pass
selp.b32 %r47, 136, 140, %p4;
add.s32 %r49, %r189, 8;
@%p9 tcgen05.mma... [ %r49 + 0 ], ..., [ %r189 + 0 ], [ %r51 + 0 ], %p8;
```

All rows still emit four
`tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`
instructions. The bug is not opcode selection.

The failing row is the only one where the scaled-MMA accumulator destination,
A-scale, and selected B-scale operands all live in the same low-address scale
slot neighborhood (`0`, `4`, `8`, `12` relative to the same base register).
The two passing rows put the accumulator and selected B-scale operands on
opposite sides of a larger `128`-column separation.

## Repair-Site Hypothesis

No new `FZ-*` bucket is needed. This is a sharper `FZ-20260421-0015`.

The most likely repair area is the interaction between:

- `lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp`
  `allocateTMem`, which assigns one linear `tensor_memory_col_offset` namespace
  across direct scale and accumulator `ttng.tmem_alloc` ops; and
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
  `convertScaledDot` / `createScaledGen5MMA`, which passes
  `ptrtoint(adaptor.getD())`, `ptrtoint(adaptor.getAScale())`, and
  `ptrtoint(adaptor.getBScale())` directly as the accumulator and SFB address
  operands.

The observed bad row suggests that the current direct pointer/integer address
representation lets scaled-MMAv5 see scale allocations and accumulator
allocations in an incompatible shared low-offset coordinate system when scales
are allocated before the accumulator. A normal selected-scale `tmem_load` still
works, so the descriptor value is valid for ordinary TMEM ld/st lowering. The
failure appears specific to the scaled-MMAv5 SFB/address operand contract, or
to the way accumulator versus scales offsets are encoded before being handed to
the inline `tcgen05.mma...block_scale` instruction.

Before a backend fix, the next diagnostic should inspect post-allocation TTGIR
or add a compiler-only probe that asserts the selected B-scale MMA operand does
not alias the accumulator-address encoding for scale-first order. Discovery
mode defers that fix.
