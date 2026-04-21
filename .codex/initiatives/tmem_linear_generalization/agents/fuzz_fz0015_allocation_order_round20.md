# Round 20 Lane AX: FZ-0015 Allocation-Order Sensitivity

Date: 2026-04-21 11:59 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes, no commit, no push.

## Scope

Sharpen `FZ-20260421-0015` by varying only the TMEM allocation order around
selected distinct direct B-scale descriptors and the accumulator:

- scale-before-accumulator;
- accumulator-before-scale;
- A scales before accumulator before B scales;
- B scales before accumulator before A scales;
- dummy TMEM scale allocation between all scales and the accumulator;
- dummy TMEM accumulator allocation between all scales and the real
  accumulator;
- accumulator then dummy scale before real scales;
- same-object and constexpr controls from the Round 19 side-channel probe.

Every row kept the selected B-scale `tmem_load` side channel so descriptor
contents could be separated from scaled-MMAv5 consumption.

## Temporary Probes

Primary exact-file probes:

```text
/tmp/tmem_fz0015_side_channel_round19_probe.py
/tmp/tmem_fz0015_acc_first_round19_probe.py
/tmp/tmem_fz0015_a_before_acc_b_round20_probe.py
/tmp/tmem_fz0015_b_before_acc_a_round20_probe.py
/tmp/tmem_fz0015_dummy_scale_between_round20_probe.py
/tmp/tmem_fz0015_dummy_acc_between_round20_probe.py
/tmp/tmem_fz0015_acc_dummy_scale_round20_probe.py
```

I also built a parametrized allocation-order probe:

```text
/tmp/tmem_fz0015_allocation_order_round20_probe.py
```

That probe is useful as a compiler-shape warning but not as the main evidence:
adding a large `ALLOC_ORDER` constexpr branch chain made the original
`scale_before_acc` rows pass. The exact-file probes below are the cleaner
evidence because each kernel starts from the Round 19 minimizer and changes
only the allocation block for one order.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Syntax and collection for the parametrized probe:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_fz0015_allocation_order_round20_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_fz0015_allocation_order_round20_probe.py
```

Result: `21 tests collected`.

Parametrized split run:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group <group> \
  /tmp/tmem_fz0015_allocation_order_round20_probe.py \
  2>&1 | tee /tmp/tmem_fz0015_allocation_order_round20_g<group>.log
```

Result: expected-failure assertions tripped because several rows were green.
This is recorded as probe-shape sensitivity, not as the main classification.

Exact-file syntax:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile \
  /tmp/tmem_fz0015_a_before_acc_b_round20_probe.py \
  /tmp/tmem_fz0015_b_before_acc_a_round20_probe.py \
  /tmp/tmem_fz0015_dummy_scale_between_round20_probe.py \
  /tmp/tmem_fz0015_dummy_acc_between_round20_probe.py \
  /tmp/tmem_fz0015_acc_dummy_scale_round20_probe.py
```

Focused exact-file runtime commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_fz0015_side_channel_round19_probe.py \
  /tmp/tmem_fz0015_a_before_acc_b_round20_probe.py \
  -k 'b_runtime_distinct and not helper and not loop' \
  2>&1 | tee /tmp/tmem_fz0015_alloc_order_exact_g0.log

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_fz0015_acc_first_round19_probe.py \
  /tmp/tmem_fz0015_b_before_acc_a_round20_probe.py \
  -k 'b_runtime_distinct and not helper and not loop' \
  2>&1 | tee /tmp/tmem_fz0015_alloc_order_exact_g1.log

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_fz0015_dummy_scale_between_round20_probe.py \
  /tmp/tmem_fz0015_dummy_acc_between_round20_probe.py \
  -k 'b_runtime_distinct and not helper and not loop' \
  2>&1 | tee /tmp/tmem_fz0015_alloc_order_exact_g2.log

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_fz0015_acc_dummy_scale_round20_probe.py \
  -k 'b_runtime_distinct and not helper and not loop' \
  2>&1 | tee /tmp/tmem_fz0015_alloc_order_exact_g3.log
```

## Exact-File Row Table

All rows used `M=N=K=128`, `mxfp8`, `use_acc=True`, one scaled-MMAv5 op, two
distinct direct B-scale descriptors, and identical B-scale payload bytes.

| Row | Probe | Allocation order | Selector | Selected B-scale load | Scaled-MMAv5 result | Classification |
| --- | --- | --- | ---: | --- | --- | --- |
| AX-001 | Round 19 side-channel | `a0,a1,b0,b1,acc` | 0 | `0/512` byte mismatches | `16381/16384` mismatches, `64` NaNs, `90` Infs | Existing `FZ-0015` |
| AX-002 | Round 19 side-channel | `a0,a1,b0,b1,acc` | 1 | `0/512` byte mismatches | `16379/16384` mismatches, `119` Infs | Existing `FZ-0015` |
| AX-003 | A-before-acc-B | `a0,a1,acc,b0,b1` | 0 | `0/512` byte mismatches | pass, `0/16384` mismatches | Green contrast |
| AX-004 | A-before-acc-B | `a0,a1,acc,b0,b1` | 1 | `0/512` byte mismatches | pass, `0/16384` mismatches | Green contrast |
| AX-005 | Acc-first | `acc,a0,a1,b0,b1` | 0 | `0/512` byte mismatches | pass, `0/16384` mismatches | Green contrast |
| AX-006 | Acc-first | `acc,a0,a1,b0,b1` | 1 | `0/512` byte mismatches | pass, `0/16384` mismatches | Green contrast |
| AX-007 | B-before-acc-A | `b0,b1,acc,a0,a1` | 0 | `0/512` byte mismatches | pass, `0/16384` mismatches | Green contrast |
| AX-008 | B-before-acc-A | `b0,b1,acc,a0,a1` | 1 | `0/512` byte mismatches | pass, `0/16384` mismatches | Green contrast |
| AX-009 | Dummy scale between | `a0,a1,b0,b1,dummy_scale,acc` | 0 | `0/512` byte mismatches | `16381/16384` mismatches, `64` NaNs, `90` Infs | Existing `FZ-0015` |
| AX-010 | Dummy scale between | `a0,a1,b0,b1,dummy_scale,acc` | 1 | `0/512` byte mismatches | `16379/16384` mismatches, `119` Infs | Existing `FZ-0015` |
| AX-011 | Dummy acc between | `a0,a1,b0,b1,dummy_acc,acc` | 0 | `0/512` byte mismatches | `16381/16384` mismatches, `64` NaNs, `90` Infs | Existing `FZ-0015` |
| AX-012 | Dummy acc between | `a0,a1,b0,b1,dummy_acc,acc` | 1 | `0/512` byte mismatches | `16379/16384` mismatches, `119` Infs | Existing `FZ-0015` |
| AX-013 | Acc dummy scale | `acc,dummy_scale,a0,a1,b0,b1` | 0 | `0/512` byte mismatches | pass, `0/16384` mismatches | Green contrast |
| AX-014 | Acc dummy scale | `acc,dummy_scale,a0,a1,b0,b1` | 1 | `0/512` byte mismatches | pass, `0/16384` mismatches | Green contrast |

## Controls and Probe-Shape Notes

The parametrized probe rechecked same-object and constexpr controls:

| Row | Selection | Allocation order | Selected B-scale load | Result |
| --- | --- | --- | --- | --- |
| AX-P1 | direct B-scale control | `scale_before_acc` | `0/512` byte mismatches | pass |
| AX-P2 | constexpr distinct B-scale | `scale_before_acc` | `0/512` byte mismatches | pass |
| AX-P3 | runtime same-object B-scale | `scale_before_acc` | `0/512` byte mismatches | pass |

The same parametrized probe unexpectedly made runtime-distinct
`scale_before_acc`, helper, and loop rows pass. I am not using those rows to
weaken `FZ-0015`; the untouched Round 19 exact-file probe still reproduces the
failure in the same cache/process generation, while exact-file allocation
variants give stable contrasts.

## Diagnosis

The bug does not follow an absolute B-scale allocation slot by itself:

- `b0,b1,acc,a0,a1` passes even though the B scales are allocated first;
- `a0,a1,acc,b0,b1` passes even though the B scales are allocated after A
  scales;
- `acc,dummy_scale,a0,a1,b0,b1` passes even with extra scale-slot pressure.

The bug follows whether the selected B-scale descriptors and accumulator are
separated by an already-allocated accumulator before the MMA consumes the
selected B-scale memdesc:

- all-scale-before-real-accumulator fails;
- inserting a dummy scale or dummy accumulator between the scales and the real
  accumulator still fails;
- putting the real accumulator before the B scales, or between A and B scales,
  passes.

The A-scale allocation order is not sufficient to trigger the failure:

- `a0,a1,acc,b0,b1` passes;
- `b0,b1,acc,a0,a1` passes.

The selected-scale side channel is clean in every row, so the selected B-scale
descriptor contents are readable through `tmem_load`; the failure remains in
scaled-MMAv5's consumption/lowering of the selected B-scale memdesc.

Current hypothesis: `FZ-0015` is sensitive to the real accumulator's TMEM
allocation position relative to the selected B-scale descriptor operands, not
to the B-scale payload or generic memdesc selection itself. The likely repair
area remains the scaled-MMAv5 B-scale/SFB operand lowering or descriptor-base
rematerialization path; discovery mode defers backend fixes.
