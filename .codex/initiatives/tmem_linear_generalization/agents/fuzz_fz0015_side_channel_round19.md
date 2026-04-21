# Round 19 Lane AW: FZ-0015 Selected B-Scale Side-Channel Reprobe

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes, no commit, no push.

## Scope

Re-run the minimized Round 15/16 `FZ-20260421-0015` shape, but add the Lane AV
style side channel: after runtime selection of the direct B-scale TMEM
descriptor and before `tcgen05_mma_scaled`, load the selected B-scale descriptor
through `ttng.tmem_load` and write the raw scale bytes to an output buffer.

This tests whether the exact failing selected B-scale descriptor is already
wrong for ordinary TMEM loads, or only wrong when consumed by scaled-MMAv5's
B-scale operand path.

## Probe

Temporary probe:

```text
/tmp/tmem_fz0015_side_channel_round19_probe.py
```

The probe was derived from `/tmp/tmem_fz0015_min_round16_probe.py` and preserves
the minimized shape:

- `M=N=K=128`;
- format `mxfp8`;
- one scaled-MMAv5 op;
- `use_acc=True`;
- linear `TensorMemoryLinearLayout([M, N])` accumulator;
- direct, non-view `TensorMemoryScalesLayout` B-scale descriptors;
- two distinct B-scale TMEM allocations populated with identical payloads;
- runtime branch/helper/loop selection of the B-scale descriptor;
- selected B-scale descriptor consumed first by `ttng.tmem_load`, then by
  `ttng.tc_gen5_mma_scaled`.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Syntax and collection:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_fz0015_side_channel_round19_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_fz0015_side_channel_round19_probe.py
```

Result: `9 tests collected`.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group <group> \
  /tmp/tmem_fz0015_side_channel_round19_probe.py \
  2>&1 | tee /tmp/tmem_fz0015_side_channel_round19_g<group>.log
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `3 passed, 6 deselected in 6.30s` |
| 2 | 1 | `3 passed, 6 deselected in 5.37s` |
| 3 | 2 | `3 passed, 6 deselected in 5.44s` |
| 4 | 3 | `9 deselected in 1.21s` |

Aggregate: `9 passed`.

The failing-miscompile rows are encoded as expected-failing classifications in
the probe; each row asserts that the selected-scale side channel has zero byte
mismatches.

## Row Table

| Row | Selection | Selector | Selected B-scale load | Scaled-MMAv5 result | Ops | Classification |
| --- | --- | ---: | --- | --- | ---: | --- |
| AW-001 | direct control | 0 | `0/512` byte mismatches | pass, `0/16384` mismatches | 4 | Green control |
| AW-002 | constexpr distinct | 1 | `0/512` byte mismatches | pass, `0/16384` mismatches | 4 | Green control |
| AW-003 | runtime same-object | 1 | `0/512` byte mismatches | pass, `0/16384` mismatches | 4 | Green control |
| AW-004 | runtime distinct branch | 0 | `0/512` byte mismatches | `16381/16384` mismatches, `64` NaNs, `90` Infs | 4 | Existing `FZ-20260421-0015` |
| AW-005 | runtime distinct branch | 1 | `0/512` byte mismatches | `16379/16384` mismatches, `119` Infs | 4 | Existing `FZ-20260421-0015` |
| AW-006 | helper-returned distinct | 0 | `0/512` byte mismatches | `16381/16384` mismatches, `170` Infs | 4 | Existing `FZ-20260421-0015` |
| AW-007 | helper-returned distinct | 1 | `0/512` byte mismatches | `16384/16384` mismatches, `32` NaNs, `52` Infs | 4 | Existing `FZ-20260421-0015` |
| AW-008 | loop-carried distinct | 0 | `0/512` byte mismatches | `16378/16384` mismatches, `32` NaNs, `60` Infs | 4 | Existing `FZ-20260421-0015` |
| AW-009 | loop-carried distinct | 1 | `0/512` byte mismatches | `16384/16384` mismatches, `32` NaNs, `52` Infs | 4 | Existing `FZ-20260421-0015` |

## Difference From Lane AV

Lane AV rows `AV-009` through `AV-011` also selected between direct B-scale
descriptors and used the selected descriptor for both a scale load and scaled
MMA, but they were green. The important shape difference is the operand storage
orientation around B:

- Round 15/16 and this AW repro use Round 16's scaled root-format shape:
  `b` has logical/storage shape `[N, K]`, shared memory layout is built for
  `[N, K]`, and the MMA consumes `b_smem.permute((1, 0))`.
- Lane AV's B-scale contrast uses a B operand with shape `[N, K]` too, but its
  helper is part of a mixed-consumer accumulator-subview probe and did not carry
  the exact Round 16 minimizer's mode matrix, random seed, and cloned
  `a_scale0/a_scale1` plus `b_scale0/b_scale1` descriptor-allocation structure.
- Both AV and AW allocate two distinct direct `TensorMemoryScalesLayout`
  B-scale descriptors and populate them with identical bytes. AW proves the
  selected descriptor bytes are still readable immediately before MMA, so the
  green AV rows do not invalidate `FZ-0015`; they are a narrower green contrast.

I also checked whether the host/global scale source explains the contrast.
Round 16 originally populated `b0` and `b1` from two cloned CUDA scale tensors,
whereas Lane AV populated both TMEM descriptors from one global `b_scale`
pointer. A focused same-global-source rerun preserved distinct TMEM B-scale
descriptors but passed the same CUDA `b_scale0` pointer as both source
arguments:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon:/tmp \
  python - <<'PY' 2>&1 | tee /tmp/tmem_fz0015_same_global_source_round19.log
...
PY
```

Result:

| Selection | Selector | Selected B-scale load | Scaled-MMAv5 result | Ops |
| --- | ---: | --- | --- | ---: |
| runtime distinct branch | 0 | `0/512` byte mismatches | `16383/16384` mismatches, `96` NaNs, `96` Infs | 4 |
| runtime distinct branch | 1 | `0/512` byte mismatches | `16380/16384` mismatches, `114` Infs | 4 |
| helper-returned distinct | 0 | `0/512` byte mismatches | `16383/16384` mismatches, `111` Infs | 4 |
| helper-returned distinct | 1 | `0/512` byte mismatches | `16382/16384` mismatches, `32` NaNs, `77` Infs | 4 |
| loop-carried distinct | 0 | `0/512` byte mismatches | `16380/16384` mismatches, `49` Infs | 4 |
| loop-carried distinct | 1 | `0/512` byte mismatches | `16382/16384` mismatches, `32` NaNs, `77` Infs | 4 |

So the AV contrast is not explained by distinct global source tensors.

The next discriminator did explain the contrast. Round 16 allocates the scale
TMEM descriptors before the accumulator, while Lane AV allocates the accumulator
before the scale descriptors. A scratch variant moved only the accumulator
allocation and zero initialization ahead of the scale allocations, preserving
the selected distinct B-scale descriptor SSA forms and the side-channel load.

Command:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_fz0015_acc_first_round19_probe.py

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_fz0015_acc_first_round19_probe.py \
  -k 'b_runtime_distinct or b_helper_branch or b_loop_distinct' \
  2>&1 | tee /tmp/tmem_fz0015_acc_first_round19.log
```

Result: the six rows were reported as pytest failures only because the inherited
probe still expected them to be miscompile rows. Every actual metric was green:

| Selection | Selector | Selected B-scale load | Scaled-MMAv5 result | Ops |
| --- | ---: | --- | --- | ---: |
| runtime distinct branch | 0 | `0/512` byte mismatches | pass, `0/16384` mismatches | 4 |
| runtime distinct branch | 1 | `0/512` byte mismatches | pass, `0/16384` mismatches | 4 |
| helper-returned distinct | 0 | `0/512` byte mismatches | pass, `0/16384` mismatches | 4 |
| helper-returned distinct | 1 | `0/512` byte mismatches | pass, `0/16384` mismatches | 4 |
| loop-carried distinct | 0 | `0/512` byte mismatches | pass, `0/16384` mismatches | 4 |
| loop-carried distinct | 1 | `0/512` byte mismatches | pass, `0/16384` mismatches | 4 |

This makes allocation order the strongest current discriminator:

- scale descriptors allocated before the accumulator reproduce `FZ-0015`;
- the same scale descriptors selected through the same SSA forms pass when the
  accumulator is allocated before them;
- Lane AV's green rows match the accumulator-first order.

IR snapshots were saved for the minimal branch selector-0 contrast:

```text
/tmp/tmem_fz0015_scale_first_round19.ttgir
/tmp/tmem_fz0015_scale_first_round19.llir
/tmp/tmem_fz0015_scale_first_round19.ptx
/tmp/tmem_fz0015_acc_first_round19.ttgir
/tmp/tmem_fz0015_acc_first_round19.llir
/tmp/tmem_fz0015_acc_first_round19.ptx
```

Both TTGIR variants have the same critical selected B-scale operand form:

```text
%24 = arith.select %23, %b_sel_17, %b_sel
%selected_probe = ttng.tmem_load %24
%30 = ttng.tc_gen5_mma_scaled ..., %a0, %24, ...
```

The distinguishing TTGIR fact is allocation order:

| Variant | Runtime result | TMEM allocation order around scales/acc |
| --- | --- | --- |
| scale-first | `16381/16384` mismatches, `64` NaNs, `90` Infs | `%a0`, `%a1`, `%b_sel`, `%b_sel_17`, then `%acc` |
| accumulator-first | pass, `0/16384` mismatches | `%acc`, then `%a0`, `%a1`, `%b_sel`, `%b_sel_17` |

PTX/LLIR still emit four matching
`tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X` ops in both
variants. The allocation-order contrast is therefore not an opcode-selection
failure.

## Classification

No new independent `FZ-*` bucket is needed. This is existing
`FZ-20260421-0015`.

The side channel strengthens the current diagnosis:

- the runtime-selected direct B-scale descriptor SSA value is correct for
  ordinary `ttng.tmem_load`;
- the same selected descriptor produces NaN/Inf-heavy wrong results only when
  consumed by `ttng.tc_gen5_mma_scaled` as the B-scale operand;
- direct, constexpr-selected, and runtime same-object controls remain green;
- branch, helper-returned, and loop-carried distinct B-scale descriptor merges
  all reproduce the wrong result.

Current hypothesis: scaled-MMAv5 lowering/planning mishandles merged SSA values
for the B-scale operand when the selected scale descriptors occupy the earlier
TMEM allocation slots than the accumulator. The issue now looks more like an
allocation-order-sensitive B-scale address/SFB operand encoding or descriptor
base rematerialization bug than generic TMEM descriptor selection or descriptor
payload storage.
