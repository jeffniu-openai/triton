# Round 53 Lane C: Generator Gap Audit

Date: 2026-04-21
Branch: `codex/tmem`
HEAD: `059a5fa39`
Mode: discovery/cataloging only. No backend/compiler repairs and no checked-in
runtime tests were modified.

## Objective

Audit the existing structural fuzzer and run a compact systematic probe over
combinations that are not all represented by the checked-in
`test_tmem_structural_fuzzer.py` table:

- descriptor view chains crossed with control-flow/SSA selection;
- op-family consumers: `ld/st`, `ld.red`, `tcgen05.copy`, plain MMAv5, and
  scaled MMAv5;
- 1CTA, 2CTA, and high-CGA/4CTA-boundary contexts;
- dtype/layout variants around scales, descriptor views, and accumulator
  layout forms.

The probe reused existing temporary generators where they already encode the
desired axes, then added focused pytest slices for high-CGA dtype/layout and
scaled descriptor-view coverage. This lane is classification-only: failures
are mapped to existing buckets unless a stable non-overlapping symptom appears.

## Preflight

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: no work to do.
```

Syntax check for temporary probes:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile \
  /tmp/tmem_structural_generator_dynamic_round25.py \
  /tmp/tmem_plain_scaled_sequence_round43_probe.py
```

Result: passed.

## Generator Schema

The effective generated case schema for this lane is:

```text
case_id / seed_id
seed range or seed source
consumer_family in {ldst, ldred, copy, mma_plain, mma_scaled, scales_ldst}
ssa_axis in {static, dynamic_index, branch, inline_branch, helper,
             mixed_captures, tuple_capture, layout_pressure, loop_carried}
view_chain in {none, index, subslice, slice_index, reshape_perm_reshape,
               reshape_pair, direct_slice, accumulator_index,
               accumulator_subslice, scale_descriptor_view}
dtype in {f32, i32, int8 scales payload, fp8 scaled-MMA operands}
layout_variant in {identity, row-permuted, column/row reshaped,
                   TensorMemoryScalesLayout, two-CTA CGA, high-CGA boundary}
num_ctas_context in {1, 2, 4-boundary/high-CGA selector rows}
expected_owner in {pass, clean_boundary, FZ-20260421-0001,
                   FZ-20260421-0002, existing scaled-MMAv5 buckets}
observed_result
opcode/memdesc feature bits when available
```

Seed sources:

- Dynamic descriptor generator: `/tmp/tmem_structural_generator_dynamic_round25.py`
  with deterministic `BJ-*` seed IDs, 22 generated rows.
- Plain/scaled sequence probe: `/tmp/tmem_plain_scaled_sequence_round43_probe.py`
  with five deterministic `ROUND43_RESULT` rows.
- Focused pytest selectors over checked-in runtime rows for high-CGA
  dtype/layout and scaled descriptor-view rows.

## Commands and Results

### 1. Dynamic descriptor/control-flow generator

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_structural_generator_dynamic_round25.py \
  > /tmp/tmem_round53_dynamic_descriptor_generator.log 2>&1
```

Generated rows: `22`.

Raw summary emitted by the probe:

```text
SUMMARY {"FZ-20260421-0002": 5, "error-unclassified": 11, "green": 4, "miscompile-new-or-related": 2}
```

Post-classification from stderr and prior minimized Round 25 detail:

| Class | Count | Notes |
| --- | ---: | --- |
| `green` | 4 | Passing `ld/st`, `copy`, and `ld.red` contrasts with expected `tcgen05` ops and generic memdesc features. |
| existing `FZ-20260421-0001` | 8 | Dynamic/helper-selected TMEM `ttg.memdesc_index` reaches LLVM conversion as an illegal op. |
| existing `FZ-20260421-0002` | 7 | Five direct ld/st control-flow rows plus two ld.red chain0 rows whose full-view readback was previously shown wrong before reduction-specific ownership is needed. |
| clean copy unsupported boundary | 3 | Branch/helper/loop-carried copy chain0 rows hit the current row-permuted destination copy-planner diagnostic. |

No new `FZ-*` candidate: every compiler failure, clean diagnostic, or wrong
result overlaps an existing owner.

### 2. Plain-MMAv5 plus scaled-MMAv5 sequence probe

Command:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_plain_scaled_sequence_round43_probe.py \
  2>&1 | tee /tmp/tmem_round53_plain_scaled_sequence.log
```

Generated rows: `5`.

Result:

| Case | Plain ops | Scaled ops | Result |
| --- | ---: | ---: | --- |
| `plain_first` | 2 | 4 | pass, `0/16384` mismatches for both outputs |
| `scaled_first` | 2 | 4 | pass, `0/16384` mismatches |
| `scaled_twice` | 2 | 8 | pass, `0/16384` mismatches |
| `plain_subslice` | 2 | 4 | pass, descriptor subslice present |
| `plain_indexed` | 2 | 4 | pass, `ttg.memdesc_index` present |

No new `FZ-*` candidate. Mixed plain/scaled sequencing remains a green contrast
when scaled scale descriptors are direct.

### 3. High-CGA / CTA-context dtype-layout slice

Collect command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'high_cga or num_ctas or four or 4cta or twocta_layout_in_4cta or cga'
```

Collection: `17/1615` rows.

Runtime command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  -k 'high_cga or num_ctas or four or 4cta or twocta_layout_in_4cta or cga' \
  python/test/gluon/test_tmem_runtime_matrix.py \
  2>&1 | tee /tmp/tmem_round53_high_cga_dtype_layout.log
```

Result:

```text
17 passed, 1598 deselected
```

Coverage includes tensor-memory scales direct/descriptor rows across cga-aware
layouts, 1CTA/2CTA context rows, and the two-CTA-layout-in-4CTA-context clean
diagnostic. No high-CGA ownership drift or new `FZ-20260421-0010` evidence was
found.

### 4. Scaled-MMAv5 descriptor-view slice

Collect command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and descriptor_view and not reports'
```

Collection: `4/1615` rows.

Runtime command:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  -k 'mma_scaled and descriptor_view and not reports' \
  python/test/gluon/test_tmem_runtime_matrix.py \
  2>&1 | tee /tmp/tmem_round53_scaled_descriptor_view.log
```

Result:

```text
4 passed, 1611 deselected
```

No new scaled-MMAv5 descriptor-view drift was observed in this small slice.

## Aggregate Classification

Total generated/probed rows: `48`.

| Classification | Count |
| --- | ---: |
| pass / green | 30 |
| existing `FZ-20260421-0001` | 8 |
| existing `FZ-20260421-0002` | 7 |
| clean unsupported/error boundary | 3 |
| new `FZ-*` candidate | 0 |

## Gap Audit Notes

- The checked-in structural fuzzer already owns strict xfails for dynamic
  memdesc indexing, control-flow descriptor miscompiles, ld/st descriptor
  read miscompiles, ld.red opcode fallback, scaled accumulator selection, and
  allocator/assertion boundaries.
- The current temporary generator still covers control-flow descriptor chains
  mostly in 1CTA contexts. The high-CGA pytest slice covers 2CTA and the
  4CTA-context clean boundary, but a unified generator that directly crosses
  dynamic descriptor SSA with legal 2CTA positive layouts remains a useful next
  gap.
- Plain/scaled MMA sequencing with direct scale descriptors is green; scale
  descriptor-view and dynamically distinct B-scale failures remain owned by
  existing scaled-MMAv5 buckets from earlier rounds, not by this lane.
- Copy plus control-flow descriptor chains split cleanly between existing
  `FZ-0001` dynamic memdesc-index lowering and the current clean copy-planner
  row-permutation boundary.

## Follow-Up

No backend repairs should start from this lane. If this generator is promoted
later, the useful next slice is a first-class checked-in seed table that adds
`num_ctas_context` to the Round 25 dynamic descriptor generator and separates
expected compiler-failure rows into subprocess tests so stderr diagnostics can
be matched precisely.
