# Round 23 Lane BE: scaled-MMAv5 accumulator descriptor/view composition

- Date: 2026-04-21 12:20 UTC
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler fixes attempted.
- Repo edit scope: this report only.

## Scope

This lane focused on accumulator operands to `tcgen05_mma_scaled`:

- direct accumulator TMEM layouts;
- static accumulator subslices;
- indexed accumulator views;
- narrow and tile-permuted accumulator layouts;
- `use_acc=True` and zero-initialized accumulator rows;
- 1CTA and 2CTA checked-in coverage;
- dynamic selected accumulator subslices with a same-descriptor `tmem_load`
  side channel.

The intent was to classify any failures against existing
`FZ-20260421-0007`, `FZ-20260421-0013`, and `FZ-20260421-0015` rather than
start backend repairs during the fuzzing campaign.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Checked-in accumulator coverage collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and acc and not reports and not fz0015'
```

Result:

```text
206/1615 tests collected (1409 deselected)
```

Split-4 runtime execution:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  --store-durations --durations-path /tmp/tmem_round23_scaled_acc_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and acc and not reports and not fz0015'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  --store-durations --durations-path /tmp/tmem_round23_scaled_acc_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and acc and not reports and not fz0015'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  --store-durations --durations-path /tmp/tmem_round23_scaled_acc_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and acc and not reports and not fz0015'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  --store-durations --durations-path /tmp/tmem_round23_scaled_acc_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and acc and not reports and not fz0015'
```

Result:

```text
group 1: 52 passed, 1563 deselected
group 2: 52 passed, 1563 deselected
group 3: 52 passed, 1563 deselected
group 4: 50 passed, 1565 deselected
aggregate: 206 passed
```

Dynamic selected-accumulator mixed-consumer probe:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py
```

Result:

```text
3 failed, 11 passed

FAILED test_scaled_selected_subslice_load_and_mma[0-branch]
  Mismatched elements: 4413 / 16384 (26.9%)
  Greatest absolute difference: nan at index (0, 48)

FAILED test_scaled_selected_subslice_load_and_mma[0-helper]
  Mismatched elements: 4413 / 16384 (26.9%)
  Greatest absolute difference: nan at index (0, 48)

FAILED test_scaled_selected_subslice_load_and_mma[0-loop]
  Mismatched elements: 4413 / 16384 (26.9%)
  Greatest absolute difference: nan at index (0, 48)
```

Existing structural xfail sentinel:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_scaled_mma_acc_subslice_control_flow[mma-scaled-fz20260421-0007-subslice-if-n64-selector0]'
```

Result:

```text
1 xfailed
```

Round 8 exact high-selector/dynamic-indexed contrasts rerun:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_scaled_mma_round8_probe.py

rm -f /tmp/tmem_scaled_acc_round23_exact.jsonl
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  TMEM_ROUND8_RESULTS=/tmp/tmem_scaled_acc_round23_exact.jsonl \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_scaled_mma_round8_probe.py \
  -k 'direct_high-nvfp4xnvfp4-n16-k256-sel0-acc1 or direct_low-nvfp4xnvfp4-n16-k256-sel0-acc1 or slice_if-nvfp4xnvfp4-n16-k256-sel1-acc1 or indexed_if-nvfp4xnvfp4-n16-k256-sel1-acc1 or slice_if-nvfp4xnvfp4-n16-k256-sel0-acc1'
```

Pytest result:

```text
5 passed, 1115 deselected
```

The tests pass because the harness records expected bug classifications to
JSONL. The recorded outcomes were:

| Case | Status | Mismatches | NaNs | PTX/LLIR opcodes |
| --- | --- | ---: | ---: | --- |
| `direct_low-nvfp4xnvfp4-n16-k256-sel0-acc1` | pass | 0 | 0 | 4 matching `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X` |
| `direct_high-nvfp4xnvfp4-n16-k256-sel0-acc1` | pass | 0 | 0 | 4 matching `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X` |
| `slice_if-nvfp4xnvfp4-n16-k256-sel0-acc1` | miscompile | 2048 | 320 | 4 matching expected ops |
| `slice_if-nvfp4xnvfp4-n16-k256-sel1-acc1` | miscompile | 2048 | 736 | 4 matching expected ops |
| `indexed_if-nvfp4xnvfp4-n16-k256-sel1-acc1` | miscompile | 2048 | 256 | 4 matching expected ops |

## Coverage Summary

The 206 checked-in green rows cover these accumulator families:

- direct `TensorMemoryLayout` and `TensorMemoryLinearLayout` accumulator
  layouts for scaled MMAv5;
- static `memdesc_index` accumulator views, including unit-parent views;
- static `memdesc_subslice` accumulator views for `N in {32,64,128}`;
- `mxfp8`, `mxfp4`, mixed `mxfp8/mxfp4`, and `nvfp4` format combinations;
- `K=128` and `K=256`;
- `use_acc=True` root, indexed, subslice, tile-permuted, and narrow rows;
- 2CTA scaled-MMAv5 accumulator-subslice copy rows.

No unexpected compiler crash, unexpected unsupported diagnostic, or runtime
miscompile appeared in those checked-in direct/static accumulator cases.

After the accumulator-focused run, a broader adjacent scaled-MMAv5 selector was
also run:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and not reports and not fz0015'
```

Result:

```text
243/1615 tests collected (1372 deselected)
```

Split-4 runtime result:

```text
group 1: 61 passed, 1554 deselected
group 2: 61 passed, 1554 deselected
group 3: 61 passed, 1554 deselected
group 4: 60 passed, 1555 deselected
aggregate: 243 passed
```

This broader guardrail includes root scaled-MMAv5 layouts, LHS subslice/tile
permutation rows, accumulator rows, 2CTA accumulator-subslice copy rows, and
known descriptor-view green controls while excluding the explicit `fz0015`
rows and report-only clean-negative cases.

## Classification

No new independent `FZ-*` bucket is warranted.

`FZ-20260421-0007` remains the owner for dynamic selected accumulator-view
miscompiles:

- direct low/high accumulator views pass;
- static subslice/indexed/tile-permuted accumulator rows pass in the checked-in
  suite;
- dynamic selected subslice rows miscompile while PTX/LLIR opcode selection is
  still correct;
- the mixed-consumer probe shows the exact selected descriptor also feeds a
  `tmem_load` side channel, and that side-channel observes the correct
  selected initialization. The dynamic descriptor is therefore not globally
  wrong for ordinary TMEM load consumers.

The Round 23 evidence also keeps `FZ-20260421-0007` distinct from the scaled
scale-operand bugs:

- it is not `FZ-20260421-0013`, because the dynamic value is the accumulator
  descriptor, not a `TensorMemoryScalesLayout` descriptor view;
- it is not `FZ-20260421-0015`, because the selected direct B-scale
  mixed-consumer rows from the refreshed Round 19 probe still pass and the
  accumulator-focused failure reproduces with clean scale operands.

## Repair-Site Hypothesis For Later

The failing rows point at scaled-MMAv5 accumulator address/view
materialization after dynamic descriptor selection. The ordinary `tmem_load`
consumer can use the same selected descriptor correctly, and direct/static
scaled-MMAv5 accumulator views are green, so the likely repair area is the
scaled-MMAv5 lowering path that rematerializes or encodes accumulator TMEM
addresses from dynamic `memdesc_subslice` / `memdesc_index` SSA values.

Backend repair remains deferred until the fuzzing campaign stops finding new
bugs or the user explicitly pivots to repair mode.
