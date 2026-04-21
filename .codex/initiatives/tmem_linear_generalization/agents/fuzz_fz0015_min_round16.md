# Round 16 Lane AN: FZ-0015 Dynamic B-Scale Descriptor Selection Minimization

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler fixes were attempted.
- Probe: `/tmp/tmem_fz0015_min_round16_probe.py`
- Logs:
  - `/tmp/tmem_fz0015_min_round16_g1.log`
  - `/tmp/tmem_fz0015_min_round16_g2.log`
  - `/tmp/tmem_fz0015_min_round16_g3.log`
  - `/tmp/tmem_fz0015_min_round16_g4.log`
  - `/tmp/tmem_fz0015_min_round16_no_parent.log`
  - `/tmp/tmem_fz0015_min_round16_parent_index.log`
  - `/tmp/tmem_fz0015_min_round16_inspect.log`
  - `/tmp/tmem_fz0015_select_load_round16.log`

## Summary

This round keeps `FZ-20260421-0015` distinct from `FZ-0001`,
`FZ-0007`, and `FZ-0013`.

Smallest stable reproducer:

- direct, non-view `TensorMemoryScalesLayout` B-scale descriptors;
- two distinct B-scale TMEM allocations populated with identical payloads;
- runtime control-flow merge of those B-scale descriptor SSA values;
- merged descriptor used as the B-scale operand of `ttng.tc_gen5_mma_scaled`;
- `M=N=K=128`, `mxfp8`, `use_acc=True`, one scaled MMA op.

The smallest row compiles and executes, emits matching scaled-MMAv5 PTX/LLIR
opcodes, and returns NaN/Inf-heavy wrong results.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Probe syntax and collection:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_fz0015_min_round16_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_fz0015_min_round16_probe.py
```

Result: `18 tests collected`.

Primary split run:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group <group> \
  /tmp/tmem_fz0015_min_round16_probe.py
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `5 passed, 13 deselected` |
| 2 | 1 | `5 passed, 13 deselected` |
| 3 | 2 | `4 passed, 1 failed, 13 deselected` |
| 4 | 3 | `3 passed, 15 deselected` |

The one failed row was the parent-storage probe, which did not reach runtime
and is discussed separately below.

Stable no-parent matrix:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_fz0015_min_round16_probe.py \
  -k 'not parent_slice'
```

Result: `17 passed, 1 deselected`.

Inspection command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon:/tmp \
  python - <<'PY' 2>&1 | tee /tmp/tmem_fz0015_min_round16_inspect.log
...
PY
```

## Case Table

| Case | Result | Classification |
| --- | --- | --- |
| direct B-scale control | `0/16384` mismatches, 4 scaled-MMA ops | Green control. |
| constexpr distinct B-scale selection | `0/16384` mismatches, 4 scaled-MMA ops | Green control; compile-time choice is safe. |
| constexpr-folded runtime arm disabled | `0/16384` mismatches, 4 scaled-MMA ops | Green control; selector load alone is not enough. |
| runtime same-object B-scale selection | `0/16384` mismatches, 4 scaled-MMA ops | Green control; branch merge is safe when both arms carry the same descriptor object. |
| runtime A-scale distinct selection | `0/16384` mismatches, 4 scaled-MMA ops | Green control; tested shape is B-scale-specific. |
| runtime distinct B-scale, selector `0` | `16381/16384` mismatches, `64` NaNs, `90` Infs, 4 scaled-MMA ops | `FZ-0015`. |
| runtime distinct B-scale, selector `1` | `16379/16384` mismatches, `119` Infs, 4 scaled-MMA ops | `FZ-0015`. |
| loop-carried distinct B-scale | `16384/16384` mismatches, `32` NaNs, `52` Infs, 4 scaled-MMA ops | `FZ-0015`; not specific to structured `if`. |
| helper-returned distinct B-scale | `16384/16384` mismatches, `32` NaNs, `52` Infs, 4 scaled-MMA ops | `FZ-0015`; helper boundary does not avoid it. |
| pass-through after branch | `16378/16384` mismatches, `64` NaNs, `96` Infs, 4 scaled-MMA ops | `FZ-0015`; identity helper after merge does not avoid it. |
| selected B-scale extra user | `16382/16384` mismatches, `160` NaNs, `207` Infs, 4 scaled-MMA ops | `FZ-0015`; extra load user does not rematerialize safely. |
| two scaled MMA ops | `16384/16384` mismatches, `1024` NaNs, `666` Infs, 8 scaled-MMA ops | `FZ-0015`; multiple ops amplify but are not required. |
| `use_acc=False` first op | `16366/16384` mismatches, `32` NaNs, `48` Infs, 4 scaled-MMA ops | `FZ-0015`; accumulator initialization is not required. |
| `N=64,K=128,mxfp8` | `8189/8192` mismatches, `51` Infs, 4 scaled-MMA ops | `FZ-0015`; reproduces at narrow N. |
| `N=128,K=256,mxfp8` | `16384/16384` mismatches, `640` NaNs, `389` Infs, 8 scaled-MMA ops | `FZ-0015`; reproduces at larger K. |
| `mxfp4` | `16378/16384` mismatches, `32` NaNs, `114` Infs, 2 scaled-MMA ops | `FZ-0015`; not limited to MXFP8. |
| `nvfp4` | `16383/16384` mismatches, no NaNs/Infs, 2 scaled-MMA ops | `FZ-0015`; FP4 NV scale format also wrong-results. |

## TTGIR/Opcode Evidence

The direct, same-object, distinct-runtime, and constexpr-distinct rows all
emitted matching PTX/LLIR scaled-MMAv5 opcodes. For the smallest failing
runtime-distinct row, PTX and LLIR both contained four:

```text
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
```

TTGIR inspection shows the failing row carries an `arith.select` of two
different `!ttg.memdesc<128x4xi8, #tmem_scales, ...>` values into the B-scale
operand of `ttng.tc_gen5_mma_scaled`. The corresponding same-object control
has no distinct-object `arith.select` and passes. The constexpr-distinct
control passes the selected descriptor directly and also passes.

## Selected-Descriptor Load Discriminator

I also checked whether the runtime-selected B-scale descriptor value is already
corrupt before scaled-MMAv5 consumes it.

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_fz0015_select_load_round16_probe.py
```

Result: `4 passed`.

Rows:

| Case | Result |
| --- | --- |
| runtime distinct B-scale, selector `0` | `0/512` selected-scale byte mismatches |
| runtime distinct B-scale, selector `1` | `0/512` selected-scale byte mismatches |
| runtime same-object B-scale | `0/512` selected-scale byte mismatches |
| constexpr selected B-scale | `0/512` selected-scale byte mismatches |

This narrows the issue further: the selected B-scale TMEM descriptor can be
loaded correctly as a normal TMEM load. The wrong result appears when that same
runtime-selected distinct descriptor is consumed by scaled-MMAv5.

## Parent-Storage Probe

The requested single-parent storage variant did not reach runtime. Both direct
slice and reshape/index attempts failed during Gluon parsing/type inference.

Final command:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  '/tmp/tmem_fz0015_min_round16_probe.py::test_fz0015_round16[b_runtime_distinct-1-0-parent_slice-False-False-True-mxfp8-128-128-False]'
```

Result:

```text
failed to infer memdesc_index result type
unsupported tensor memory memdesc_subslice view; preserved tensor memory view encoding also failed:
rank must be less than or equal to the memdesc rank for tensor memory
```

This row is not evidence for or against `FZ-0015` because it never reaches
scaled-MMAv5 lowering or execution. It may be useful for a later descriptor-view
support audit, but I did not assign a new `FZ-*` bucket in this lane.

## Classification

`FZ-20260421-0015` should stay separate:

- Not `FZ-0001`: compilation succeeds and the runtime wrong result is produced;
  there is no illegal `ttg.memdesc_index` surviving to LLVM conversion.
- Not `FZ-0007`: the dynamic descriptor is the B-scale operand, not an
  accumulator descriptor/view.
- Not `FZ-0013`: the minimized reproducer uses direct non-view B-scale
  descriptors. Descriptor-view chain lowering is not required.

The strongest current hypothesis is that scaled-MMAv5 lowering/codegen mishandles
a runtime-selected distinct B-scale TMEM descriptor value. The issue appears
specific to the B-scale operand path in this shape because runtime A-scale
descriptor selection passes.
