# Round 43 Lane: mixed plain-MMAv5 and scaled-MMAv5 sequence fuzzing

- Date: 2026-04-21 14:24 UTC
- Branch: `codex/tmem`
- Scope: discovery/cataloging only; no backend/compiler code modified.
- Owned write path: this report only.

## Summary

No new independent `FZ-*` bucket is warranted from this lane.

Checked-in runtime selectors stayed green across the broad MMAv5/scaled-MMAv5
surface, and temporary probes reproduced only known scaled-MMAv5 buckets:

- Plain MMAv5 and scaled MMAv5 can coexist in one runtime kernel with separate
  accumulators, either ordering, repeated scaled execution, a plain accumulator
  subslice view, and a runtime indexed plain accumulator view.
- Direct selected B-scale descriptors, runtime accumulator selection, and
  `use_acc` sequencing still pass in the broad mixed-scaled probe.
- Scale descriptor-view operands still produce runtime wrong results with
  correct side-channel scale loads and correct scaled-MMA opcode counts. This
  overlaps existing `FZ-20260421-0013`.
- Runtime-selected distinct direct B-scale descriptors still produce runtime
  wrong results with correct side-channel scale loads. This overlaps existing
  `FZ-20260421-0015`.
- Dynamic descriptor/index forms that reach late illegal lowering remain
  covered by existing `FZ-20260421-0001`; no new illegal-lowering shape was
  found here.
- The lane did not find a new `FZ-20260421-0007` composition: dynamic/runtime
  accumulator selection with direct scales passed, and dynamic accumulator
  rows combined with scale descriptor views are still explained by
  `FZ-20260421-0013`.

## Required rebuild

Command:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: no work to do.
```

## Environment note

An initial collect-only command without `PYTHONPATH=./python` imported a
different editable Triton checkout from `/tmp/triton-upstream-main-check` and
failed during collection with missing `triton.compiler.errors` /
`triton.runtime.jit`. All successful commands below used the workspace
`PYTHONPATH` explicitly.

## Checked-in selector sweep

Collect command:

```bash
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 \
  TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_core.py \
  -k "(mma_scaled or tcgen05_mma or use_acc or indexed_acc or subslice or bscale or b_scale) and not reports and not resource"
```

Result: `648/19729` collected.

Split-4 runtime commands used stable per-GPU cache directories:

```bash
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py -k "(mma_scaled or tcgen05_mma or use_acc or indexed_acc or subslice or bscale or b_scale) and not reports and not resource"
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py -k "(mma_scaled or tcgen05_mma or use_acc or indexed_acc or subslice or bscale or b_scale) and not reports and not resource"
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py -k "(mma_scaled or tcgen05_mma or use_acc or indexed_acc or subslice or bscale or b_scale) and not reports and not resource"
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py -k "(mma_scaled or tcgen05_mma or use_acc or indexed_acc or subslice or bscale or b_scale) and not reports and not resource"
```

Results:

| Group | Result |
| --- | --- |
| 1 | `162 passed, 19567 deselected` |
| 2 | `162 passed, 19567 deselected` |
| 3 | `159 passed, 3 skipped, 19567 deselected` |
| 4 | `162 passed, 19567 deselected` |

Aggregate: `645 passed, 3 skipped`.

Coverage included checked-in plain MMAv5 `use_acc`, root/indexed/subslice
accumulator views, scaled-MMAv5 root/indexed/subslice/tile-permuted
accumulator rows, B-scale descriptor-view/extra-user rows, direct scaled
copy/MMA loops, and multicast/TMA-adjacent MMAv5 rows. No compiler crash,
unexpected unsupported diagnostic, opcode absence, runtime miscompile, or
unexpected pass/fail transition was observed in this checked-in slice.

## Temporary mixed-scaled replay

Command:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  python /root/tmp/tmem_scaled_mixed_operands_round24_probe.py
```

Result: the replay matched the prior Round 24 classification.

Green direct rows:

| Row | Result |
| --- | --- |
| `direct_b_after_scales_sel1` | pass, A/B/acc side probes `0` mismatches, scaled output `0/16384` mismatches, PTX/LLIR scaled ops `4/4` |
| `direct_b_before_scales_sel1` | pass, probes `0`, output `0/16384`, ops `4/4` |
| `direct_acc_runtime_after_scales_sel1` | pass, probes `0`, output `0/16384`, ops `4/4` |
| `direct_acc_runtime_b_after_scales_sel1` | pass, probes `0`, output `0/16384`, ops `4/4` |

Known `FZ-20260421-0013` rows:

| Row | Symptom |
| --- | --- |
| `view_static_after_scales_sel0` | `16374/16384` mismatches, side probes `0`, ops `4/4` |
| `view_b_after_scales_sel1` | `16376/16384` mismatches, side probes `0`, ops `4/4` |
| `view_acc_runtime_after_scales_sel1` | `16375/16384` mismatches, side probes `0`, ops `4/4` |
| `view_acc_runtime_b_after_scales_sel1` | `16376/16384` mismatches, side probes `0`, ops `4/4` |
| `view_b_before_scales_sel1` | `16370/16384` mismatches, side probes `0`, ops `4/4` |
| `view_acc_runtime_b_before_scales_sel1` | `16372/16384` mismatches, side probes `0`, ops `4/4` |
| `view_b_after_scales_two_mma_sel1` | `16373/16384` mismatches, side probes `0`, ops `8/8` |
| `view_both_after_scales_sel1` | `16371/16384` mismatches, side probes `0`, ops `4/4` |

Classification: no new bucket. Scale descriptor-view operands dominate these
failures and remain `FZ-20260421-0013`, even with B-scale selection,
runtime accumulator selection, allocation-order changes, and repeated scaled
MMA execution.

## Exact `FZ-0015` replay

Command:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  pytest -q -s --tb=short \
  /tmp/tmem_fz0015_side_channel_round19_probe.py::test_fz0015_round16
```

Result: `9 passed`, where the probe encodes the known miscompile as expected
classification. Representative printed metrics:

| Row | Side-channel | Runtime result |
| --- | --- | --- |
| `direct_control` | `0/512` scale mismatches | pass, `0/16384` mismatches |
| `b_constexpr_distinct` | `0/512` | pass, `0/16384` |
| `b_runtime_same_object` | `0/512` | pass, `0/16384` |
| `b_runtime_distinct`, selector `0` | `0/512` | `16381/16384` mismatches, `64` NaNs, `90` Infs |
| `b_runtime_distinct`, selector `1` | `0/512` | `16379/16384` mismatches, `119` Infs |
| `b_helper_branch`, selector `0` | `0/512` | `16381/16384` mismatches, `170` Infs |
| `b_helper_branch`, selector `1` | `0/512` | `16384/16384` mismatches, `32` NaNs, `52` Infs |
| `b_loop_distinct`, selector `0` | `0/512` | `16378/16384` mismatches, `32` NaNs, `60` Infs |
| `b_loop_distinct`, selector `1` | `0/512` | `16384/16384` mismatches, `32` NaNs, `52` Infs |

Classification: existing `FZ-20260421-0015`; no new bucket. The failure is
still direct selected distinct B-scale specific and remains separated from
scale descriptor-view `FZ-20260421-0013` and dynamic accumulator-view
`FZ-20260421-0007`.

## Round 15 multi-MMA replay

Command:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  pytest -q -s --tb=short /tmp/tmem_scaled_multi_mma_round15_probe.py
```

Result: `14 passed, 5 failed`.

Failures were expected known-bucket rows:

| Test | Classification |
| --- | --- |
| `test_am_scale_descriptor_view_chains[True-False]` | existing `FZ-20260421-0013`, `16107/16384` mismatches |
| `test_am_scale_descriptor_view_chains[False-True]` | existing `FZ-20260421-0013`, `16116/16384` mismatches |
| `test_am_scale_descriptor_view_chains[True-True]` | existing `FZ-20260421-0013`, `16373/16384` mismatches |
| `test_am_dynamic_bscale_descriptor_selection[0]` | existing `FZ-20260421-0015`, `16381/16384` mismatches with NaNs |
| `test_am_dynamic_bscale_descriptor_selection[1]` | existing `FZ-20260421-0015`, `16380/16384` mismatches with NaNs |

The `14` passing rows preserve the green controls for multiple scaled MMAs,
scale reuse, static accumulator subslices, branch-contained B-scale MMAs,
same-object B-scale selection, and dynamic A-scale selection.

## New temporary plain-plus-scaled sequence probe

Probe:

```text
/tmp/tmem_plain_scaled_sequence_round43_probe.py
```

Compile and run command:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_plain_scaled_sequence_round43_probe.py && \
PYTHONPATH=.:./python:./python/test/gluon \
  CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  python /tmp/tmem_plain_scaled_sequence_round43_probe.py
```

Rows:

| Case | Plain-MMAv5 result | Scaled-MMAv5 result | Opcode counts | Notes |
| --- | --- | --- | --- | --- |
| `plain_first` | `0/16384` mismatches | `0/16384` mismatches | plain `2`, scaled `4` | Plain `use_acc=True` then scaled MMA |
| `scaled_first` | `0/16384` | `0/16384` | plain `2`, scaled `4` | Scaled MMA before plain accumulator user |
| `scaled_twice` | `0/16384` | `0/16384` | plain `2`, scaled `8` | Repeated scaled MMA execution in same kernel |
| `plain_subslice` | `0/16384` | `0/16384` | plain `2`, scaled `4` | Plain accumulator via `slice(...).index(0)` |
| `plain_indexed` | `0/16384` | `0/16384` | plain `2`, scaled `4` | Plain accumulator via runtime `index(load(selector))` |

The probe intentionally keeps scaled scale descriptors direct, so it is a
positive control for mixed plain/scaled sequencing rather than another
`FZ-0013` reproducer. The runtime-indexed plain row did not reproduce
`FZ-20260421-0001` because this shape compiles and executes correctly; the
TTGIR still contains `ttg.memdesc_index`, but it does not reach an illegal
LLVM-conversion failure in this live-consumer sequence.

## Classification

No new `FZ-*` id proposed.

Existing overlaps:

- `FZ-20260421-0001`: still covers dynamic TMEM descriptor/index forms that
  leave illegal `ttg.memdesc_index` at LLVM conversion. This lane found no new
  illegal-lowering reproducer.
- `FZ-20260421-0007`: not reproduced by the direct-scale dynamic accumulator
  rows. Mixed scale-view plus accumulator rows are better classified as
  `FZ-20260421-0013`.
- `FZ-20260421-0013`: all scale descriptor-view wrong-result rows here,
  including repeated scaled MMA and mixed B-scale/accumulator compositions.
- `FZ-20260421-0015`: exact direct selected distinct B-scale reproducer and
  Round 15 dynamic B-scale descriptor selection rows.

The new positive mixed plain-plus-scaled sequence probe adds a green contrast:
plain-MMAv5 and scaled-MMAv5 sequencing itself is not independently broken for
the tested direct-scale one-CTA shapes.
