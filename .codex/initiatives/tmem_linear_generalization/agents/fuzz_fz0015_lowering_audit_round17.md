# Round 17 Lane AO: FZ-0015 Lowering Audit

- Date: 2026-04-21
- Branch: `codex/tmem`
- HEAD at audit start: `bfa1aa54a`
- Mode: discovery/cataloging only. No backend/compiler code was changed. No
  commit or push was made by this lane.
- Primary probe: `/tmp/tmem_fz0015_lowering_audit_round17_probe.py`
- Primary log: `/tmp/tmem_fz0015_lowering_audit_round17_probe.log`
- Extracted IR/assembly directory:
  `/tmp/tmem_fz0015_lowering_audit_round17/`

## Summary

`FZ-20260421-0015` remains distinct from `FZ-20260421-0013`.

The repro uses direct, non-view `TensorMemoryScalesLayout` B-scale TMEM
allocations. No descriptor-view chain is required. Runtime selection between
two distinct direct B-scale memdesc SSA values still produces wrong results,
while direct, constexpr-selected B-scale, same-object dynamic B-scale, and
runtime-selected A-scale controls pass.

The selected B-scale descriptor value is not generally corrupt: the Round 16
selected-load discriminator read the runtime-selected B-scale descriptor through
normal TMEM load with `0/512` byte mismatches for both selectors. The failure
appears only when that selected descriptor feeds `ttng.tc_gen5_mma_scaled` as
the B-scale operand.

The strongest current hypothesis is a B-scale-specific lowering/codegen
contract gap around `ttng.tc_gen5_mma_scaled` when the SFB TMEM address is a
runtime-selected register value. The LLVM/PTX lowering preserves the dynamic
selection as a `select` / `selp.b32`; the wrong result is not caused by the
selected memdesc being dropped before LLVM.

## Commands

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Probe generation and execution:

```bash
cat > /tmp/tmem_fz0015_lowering_audit_round17_probe.py <<'PY'
...
PY

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon:/tmp \
  python /tmp/tmem_fz0015_lowering_audit_round17_probe.py \
  2>&1 | tee /tmp/tmem_fz0015_lowering_audit_round17_probe.log
```

The probe imported the existing Round 16 minimizer from
`/tmp/tmem_fz0015_min_round16_probe.py`, reran focused cases, and wrote
`ttgir`, `llir`, `ptx`, and `cubin` for each case under
`/tmp/tmem_fz0015_lowering_audit_round17/`.

TTGIR verifier sanity:

```bash
BUILD_DIR=$(PYTHONPATH=./python python3 -c \
  'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
$BUILD_DIR/bin/triton-opt \
  /tmp/tmem_fz0015_lowering_audit_round17/b_runtime_distinct_sel1_const0.ttgir \
  -verify-diagnostics \
  >/tmp/tmem_fz0015_lowering_audit_round17_triton_opt_verify.log 2>&1
```

Result: `triton-opt` exited `0`. The saved TTGIR is parser/verifier-clean.

SASS probe:

```bash
cuobjdump --dump-sass \
  /tmp/tmem_fz0015_lowering_audit_round17/b_runtime_distinct_sel1_const0.cubin \
  > /tmp/tmem_fz0015_lowering_audit_round17_sass_excerpt.log
```

`cuobjdump` ran, but the textual SASS dump did not expose a useful
`TCGEN05.MMA` mnemonic excerpt in the grep slice. PTX/LLIR were more useful for
this audit.

## Runtime Results

The focused audit probe produced:

| Case | Result |
| --- | --- |
| `direct_control_sel0_const0` | `0/16384` mismatches, no NaNs/Infs |
| `b_constexpr_distinct_sel1_const1` | `0/16384` mismatches, no NaNs/Infs |
| `b_runtime_same_object_sel1_const0` | `0/16384` mismatches, no NaNs/Infs |
| `b_runtime_distinct_sel1_const0` | `16379/16384` mismatches, `119` Infs |
| `b_loop_distinct_sel1_const0` | `16384/16384` mismatches, `32` NaNs, `52` Infs |
| `a_runtime_distinct_sel1_const0` | `0/16384` mismatches, no NaNs/Infs |

Every row emitted four matching scaled-MMAv5 opcodes in LLIR/PTX:

```text
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
```

## TTGIR Discriminator

The failing runtime-distinct B-scale row has direct scale allocations and a
plain `arith.select` feeding only the B-scale operand:

```mlir
%b_sel = ttng.tmem_alloc : () -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
%b_sel_17 = ttng.tmem_alloc : () -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
ttng.tmem_store %16, %b_sel, %true : tensor<128x4xi8, #linear> -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
ttng.tmem_store %21, %b_sel_17, %true : tensor<128x4xi8, #linear> -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
%24 = arith.select %23, %b_sel_17, %b_sel : !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
%26 = ttng.tc_gen5_mma_scaled %a_smem_11, %25, %acc[], %a0, %24, %true, %true lhs = e4m3 rhs = e4m3
```

The loop-carried row fails similarly, with an `scf.for` result feeding the
B-scale operand:

```mlir
%b_sel_20 = scf.for ... iter_args(%b_sel_23 = %b_sel)
%23 = ttng.tc_gen5_mma_scaled %a_smem_11, %22, %acc[], %a0, %b_sel_20, %true, %true lhs = e4m3 rhs = e4m3
```

The A-scale dynamic control uses the same kind of direct dynamic memdesc value,
but as the A-scale operand, and passes:

```mlir
%24 = arith.select %23, %a_sel_17, %a_sel : !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
%26 = ttng.tc_gen5_mma_scaled %a_smem_11, %25, %acc[], %24, %b0, %true, %true lhs = e4m3 rhs = e4m3
```

## LLVM/PTX Evidence

The TTGIR dynamic B-scale memdesc survives into LLVM as a scalar `i32` select.
In the failing runtime-distinct row:

```llvm
%1308 = add i32 %13, 4
%1309 = add i32 %13, 8
%1310 = add i32 %13, 12
%.v = select i1 %.not, i32 %1309, i32 %1310
tail call void asm sideeffect
  "@$7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
   [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;",
  "r,l,l,r,r,r,b,b"(i32 %1308, ..., i32 %13, i32 %.v, ...)
```

The corresponding PTX has the same shape:

```ptx
selp.b32 %r46, %r3, %r4, %p8;
@%p7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
  [ %r44 + 0 ], %rd286, %rd287, %r45, [ %r184 + 0 ], [ %r46 + 0 ], %p6;
```

The failing loop-carried row is analogous:

```ptx
selp.b32 %r48, %r4, %r51, %p8;
@%p7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
  [ %r45 + 0 ], %rd286, %rd287, %r46, [ %r186 + 0 ], [ %r48 + 0 ], %p6;
```

The passing dynamic A-scale control proves a similar runtime-selected scale
address register is not universally invalid:

```llvm
%.v = select i1 %.not, i32 %13, i32 %1308
tail call void asm sideeffect
  ... (i32 %1310, ..., i32 %.v, i32 %1309, ...)
```

PTX:

```ptx
selp.b32 %r45, %r184, %r3, %p8;
@%p7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
  [ %r43 + 0 ], %rd286, %rd287, %r44, [ %r45 + 0 ], [ %r46 + 0 ], %p6;
```

The passing constexpr B-scale row feeds a direct scalar address, not a
`select`:

```llvm
tail call void asm sideeffect ... (i32 %1359, ..., i32 %13, i32 %1308, ...)
```

## Compiler-Side Audit

The most suspicious code is scaled MMAv5 LLVM lowering in
`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`.

Relevant slice:

```cpp
Value baseScaleA = tb.ptrtoint(i32_ty, adaptor.getAScale());
Value baseScaleB = tb.ptrtoint(i32_ty, adaptor.getBScale());
...
Value scaleA =
    tb.add(baseScaleA, tb.i32_val(scaleAFragment.tmemColumnOffset));
Value scaleB =
    tb.add(baseScaleB, tb.i32_val(scaleBFragment.tmemColumnOffset));
...
createScaledGen5MMA(..., scaleA, scaleB, ...);
```

The dynamic selected B-scale value reaches `baseScaleB`, and the emitted PTX
passes the selected register as the SFB address operand. For this shape,
`scaleBFragment.tmemColumnOffset` is `0` for all four MMAs and the subcolumn is
encoded in the instruction descriptor, so the dynamic-vs-direct difference is
the selected SFB base register itself.

`RematerializeScaledMmaBScaleFragments` in
`lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp` does not
explain this repro:

- this shape has no narrow/repeated-N32 fragment requirement, so the pass exits
  before rematerialization;
- the repro uses direct non-view scale allocations;
- normal selected descriptor loads pass.

The verifier and planner paths accept this shape. That is useful evidence:
this is not a too-strict verifier or unsupported-case diagnostic. It is a
runtime miscompile after legal TTGIR/LLIR/PTX generation.

## Classification

Keep `FZ-20260421-0015` separate.

- Not `FZ-0001`: compilation succeeds; no illegal `ttg.memdesc_index` reaches
  LLVM conversion.
- Not `FZ-0007`: the dynamic value is the B-scale operand, not an accumulator
  descriptor/view.
- Not `FZ-0013`: there is no B-scale descriptor-view chain in the minimized
  repro. The values are direct `ttng.tmem_alloc` results.
- Not generic dynamic memdesc selection: selected descriptor loads pass, and
  runtime-selected A-scale direct descriptors pass.

Current likely backend site: `convertScaledDot` / `createScaledGen5MMA` in
`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`, or an
unmodeled ISA/codegen constraint immediately around the SFB address operand of
`tcgen05.mma...block_scale`.

## Artifacts

- `/tmp/tmem_fz0015_lowering_audit_round17_probe.py`
- `/tmp/tmem_fz0015_lowering_audit_round17_probe.log`
- `/tmp/tmem_fz0015_lowering_audit_round17/direct_control_sel0_const0.ttgir`
- `/tmp/tmem_fz0015_lowering_audit_round17/direct_control_sel0_const0.ll`
- `/tmp/tmem_fz0015_lowering_audit_round17/direct_control_sel0_const0.ptx`
- `/tmp/tmem_fz0015_lowering_audit_round17/b_constexpr_distinct_sel1_const1.ttgir`
- `/tmp/tmem_fz0015_lowering_audit_round17/b_constexpr_distinct_sel1_const1.ll`
- `/tmp/tmem_fz0015_lowering_audit_round17/b_constexpr_distinct_sel1_const1.ptx`
- `/tmp/tmem_fz0015_lowering_audit_round17/b_runtime_distinct_sel1_const0.ttgir`
- `/tmp/tmem_fz0015_lowering_audit_round17/b_runtime_distinct_sel1_const0.ll`
- `/tmp/tmem_fz0015_lowering_audit_round17/b_runtime_distinct_sel1_const0.ptx`
- `/tmp/tmem_fz0015_lowering_audit_round17/b_loop_distinct_sel1_const0.ttgir`
- `/tmp/tmem_fz0015_lowering_audit_round17/b_loop_distinct_sel1_const0.ll`
- `/tmp/tmem_fz0015_lowering_audit_round17/b_loop_distinct_sel1_const0.ptx`
- `/tmp/tmem_fz0015_lowering_audit_round17/a_runtime_distinct_sel1_const0.ttgir`
- `/tmp/tmem_fz0015_lowering_audit_round17/a_runtime_distinct_sel1_const0.ll`
- `/tmp/tmem_fz0015_lowering_audit_round17/a_runtime_distinct_sel1_const0.ptx`
- `/tmp/tmem_fz0015_lowering_audit_round17_triton_opt_verify.log`
- `/tmp/tmem_fz0015_lowering_audit_round17_sass_excerpt.log`

