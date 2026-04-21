# Round 22 Lane BA: Scaled-MMAv5 Dynamic Scale Descriptor Fuzzing

- Date: 2026-04-21 12:58 UTC
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler fixes.
- Repo edit scope: this report only.
- Probe: `/tmp/tmem_scaled_dynamic_scales_round22_probe.py`
- Log: `/tmp/tmem_scaled_dynamic_scales_round22_probe.log`

## Scope

This lane focused on scaled-MMAv5 scale descriptors, especially combinations
that are adjacent to but not identical to the known selected direct B-scale
allocation-order failure:

- runtime-selected direct A-scale and B-scale descriptors;
- nested `if`, helper-returned, and loop-carried selected scale descriptors;
- mixed consumers where the same selected scale descriptor feeds both
  `ttng.tmem_load` side-channel probes and `ttng.tc_gen5_mma_scaled`;
- accumulator-first and accumulator-last allocation-order contrasts;
- scale descriptor-view chains using legal reshape/permute/reshape views.

The goal was to classify against:

- `FZ-20260421-0001`: dynamic/generic memdesc illegal op at LLVM conversion;
- `FZ-20260421-0007`: scaled dynamic accumulator-subslice miscompile;
- `FZ-20260421-0013`: scaled-MMAv5 scale descriptor-view wrong results;
- `FZ-20260421-0015`: selected distinct direct B-scale allocation/order
  miscompile.

## Commands

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Temporary probe validation:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_scaled_dynamic_scales_round22_probe.py

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_scaled_dynamic_scales_round22_probe.py \
  2>&1 | tee /tmp/tmem_scaled_dynamic_scales_round22_probe.log
```

Nearby checked-in descriptor-view baseline:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and descriptor_view and not reports' \
  2>&1 | tee /tmp/tmem_round22_checked_descriptor_view.log
```

Result:

```text
4 passed, 1611 deselected
```

## Probe Results

The final probe ran `16` cases. All cases matched expected classification:

- `10` direct selected-scale controls passed.
- `6` scale descriptor-view cases miscompiled as expected under existing
  `FZ-20260421-0013`.
- No compiler crash, late illegal op, too-strict unsupported diagnostic, or
  new independent `FZ-*` was found.

Direct dynamic selected-scale controls:

| Case | Selection shape | Acc order | Selector | Result |
| --- | --- | --- | --- | --- |
| `direct_b_if_acc_first` | B-scale runtime `if` | acc first | 0 | pass |
| `direct_b_if_acc_last_sel0` | B-scale runtime `if` | acc last | 0 | pass |
| `direct_b_if_acc_last_sel1` | B-scale runtime `if` | acc last | 1 | pass |
| `direct_b_helper_acc_last_sel1` | helper-returned B-scale | acc last | 1 | pass |
| `direct_b_nested_acc_first` | nested B-scale `if` | acc first | 1 | pass |
| `direct_b_helper_acc_first` | helper-returned B-scale | acc first | 1 | pass |
| `direct_b_loop_acc_first` | loop-carried B-scale | acc first | 1 | pass |
| `direct_a_if_acc_first` | A-scale runtime `if` | acc first | 1 | pass |
| `direct_both_if_acc_first` | A-scale and B-scale runtime `if` | acc first | 1 | pass |
| `direct_b_if_two_mma` | B-scale runtime `if`, two MMAs | acc first | 1 | pass |

All direct selected-scale controls had:

- `a_probe_mismatch = 0`;
- `b_probe_mismatch = 0`;
- matching PTX/LLIR scaled-MMAv5 opcode counts (`4`, or `8` for the two-MMA
  case);
- zero output mismatches.

Descriptor-view scale cases:

| Case | Selection shape | Acc order | Selector | Output mismatches | Side-channel probes |
| --- | --- | --- | --- | --- | --- |
| `rtr_static_acc_first` | static reshape/permute/reshape view | acc first | 0 | `16374/16384` | clean |
| `rtr_static_acc_last` | static reshape/permute/reshape view | acc last | 0 | `16368/16384` | clean |
| `b_if_rtr_acc_first_sel0` | B-scale runtime `if` over views | acc first | 0 | `16373/16384` | clean |
| `b_if_rtr_acc_first` | B-scale runtime `if` over views | acc first | 1 | `16372/16384` | clean |
| `b_nested_rtr_acc_first` | nested B-scale `if` over views | acc first | 1 | `16373/16384` | clean |
| `both_if_rtr_acc_first` | A/B-scale runtime `if` over views | acc first | 1 | `16371/16384` | clean |

The side-channel scale loads were correct in every descriptor-view row
(`a_probe_mismatch = 0`, `b_probe_mismatch = 0`). The miscompile is therefore
not due to bad payload writes or reads through the view descriptor. It remains
specific to scaled-MMAv5 consuming scale descriptors with view-chain layouts.

## Classification

No new independent `FZ-*` bucket is warranted.

This lane sharpens `FZ-20260421-0013`:

```text
Scaled-MMAv5 miscompiles when A-scale and/or B-scale operands are legal
reshape/permute/reshape TensorMemoryScalesLayout descriptor views. Static,
runtime-selected, nested-selected, and mixed A/B selected view forms all show
large wrong-result mismatches while side-channel tmem_load from the same
selected scale descriptors is correct.
```

This lane also adds a negative contrast for `FZ-20260421-0015`:

```text
Direct runtime-selected B-scale descriptors did not reproduce the
allocation-order-sensitive FZ-0015 failure in this simplified same-payload
probe, even when the accumulator was allocated after A/B scale descriptors.
The FZ-0015 discriminator remains more specific than "any direct selected
B-scale consumed by scaled-MMAv5"; the exact all-scale-before-real-accumulator
shape from the prior probes remains the reproducer.
```

It does not overlap `FZ-20260421-0001`, because no `ttg.memdesc_index` escaped
to LLVM conversion and no dynamic/generic memdesc illegal-op failure occurred.
It does not overlap `FZ-20260421-0007`, because the accumulator descriptor was
not dynamically selected or subsliced in this lane.

## Harness Notes

An initial version of the probe intentionally tried more aggressive scale view
forms. Those were discarded as harness-invalid rather than cataloged:

- direct `.slice(...)` on TensorMemoryScalesLayout descriptors produced a
  parsing-time shared-memory encoding diagnostic;
- `reshape(...).index(...)` over a lifted scale parent produced a clean
  frontend descriptor-view inference diagnostic;
- dynamic branches selecting differently shaped `rtr` view layouts produced
  type mismatch assertions in Gluon control-flow typing.

The final reported probe uses only same-layout direct descriptors and
same-layout reshape/permute/reshape view descriptors, so the remaining
wrong-result rows are backend-relevant and classify under existing
`FZ-20260421-0013`.

## Validation

```bash
git diff --check -- .codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_dynamic_scales_round22.md
```

Result:

```text
clean
```
