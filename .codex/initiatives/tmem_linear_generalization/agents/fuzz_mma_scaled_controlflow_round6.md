# Round 6 Lane B: MMAv5 / Scaled-MMAv5 TMEM Descriptor Views Under Control Flow

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery only; no backend/compiler repair attempted.

## Scope

Lane B focused on MMAv5 and scaled-MMAv5 TMEM descriptor views under control
flow:

- indexed and subslice accumulator views;
- `use_acc` paths;
- two-CTA and larger-CGA contexts;
- helper-returned descriptors;
- `if`- and loop-carried accumulator descriptors;
- narrow `N` and scale layouts.

The probe work preferred runtime executions that compare numeric output and
inspect generated `tcgen05` opcodes where the kernel reaches codegen.

## Build Prerequisite

Required pre-test build:

```bash
make -j8
```

Result: `ninja: no work to do`.

## Existing Matrix Sweeps

Scaled-MMAv5 descriptor/narrow/use-acc sweep:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group <1..4> \
python/test/gluon/test_tmem_runtime_matrix.py \
-k 'mma_scaled and (indexed_acc or acc_subslice or lhs_subslice or tile_permuted or narrow or bscale_descriptor_view or use_acc)'
```

Result:

- group 1: `64 passed, 1551 deselected`
- group 2: `64 passed, 1551 deselected`
- group 3: `64 passed, 1551 deselected`
- group 4: `62 passed, 1553 deselected`

Plain MMAv5 indexed/subslice/two-CTA/tile/lhs sweep:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group <1..4> \
python/test/gluon/test_tmem_runtime_matrix.py \
-k 'mma and not mma_scaled and (indexed_acc or acc_subslice or twocta_indexed or twocta_acc_subslice or tile_permuted or lhs_subslice)'
```

Result:

- group 1: `44 passed, 1571 deselected`
- group 2: `44 passed, 1571 deselected`
- group 3: `44 passed, 1571 deselected`
- group 4: `43 passed, 1572 deselected`

Larger-CGA / two-CTA-layout diagnostic selector:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python pytest -s --tb=short \
python/test/gluon/test_tmem_runtime_matrix.py \
-k 'larger_cga or four_cta or num_ctas or twocta_layout'
```

Result: `2 passed, 1613 deselected`.

## Temporary Control-Flow Harness

Temporary harness:

```text
/tmp/tmem_mma_scaled_controlflow_round6_probe.py
```

The harness was copied from `python/test/gluon/test_tmem_runtime_matrix.py`
and extended locally with scaled-MMAv5 accumulator-control-flow kernels. The
first collection attempt failed because `/tmp` was the pytest root and did not
see `python/test/gluon/tmem_test_utils.py`; reruns used
`PYTHONPATH=.:./python:./python/test/gluon`.

Control rows:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short \
/tmp/tmem_mma_scaled_controlflow_round6_probe.py \
-k 'round6_scaled_mma_acc_controlflow'
```

Result after harness correction: `1 failed, 3 passed` for the original four
rows.

Focused subslice boundary rerun:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short \
/tmp/tmem_mma_scaled_controlflow_round6_probe.py -k 'r6-scaled-subslice'
```

Result: `1 failed, 3 passed, 1617 deselected`.

## New Finding

### FZ-20260421-0007: scaled-MMAv5 `use_acc` subslice accumulator miscompile through dynamic `if`

- Case id: `r6-scaled-subslice-if-n64`
- Seed: `0x6B00` implicit harness seed via `torch.manual_seed(0)`
- Family: `mma_scaled`
- Shape: `M=128`, `N=64`, `K=128`
- Formats: `mxfp8` x `mxfp8`
- Accumulator parent: `TensorMemoryLinearLayout` for `[128, 128]`
- View chain: `acc_parent.slice(0, 64, dim=1)` versus
  `acc_parent.slice(64, 64, dim=1)`
- Control flow: dynamic `if ttgl.load(selector_ptr) != 0`
- Selector: `0`, selecting the low-column subslice
- `use_acc`: true, with preloaded accumulator values
- Expected: `a_ref @ b_ref.T + 3.0`
- Observed: runtime numeric miscompile after successful compile and launch

Fresh exact repro:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short \
'/tmp/tmem_mma_scaled_controlflow_round6_probe.py::test_round6_scaled_mma_acc_controlflow[r6-scaled-subslice-if-n64-subslice-if-64-0-2]'
```

Fresh result:

```text
Mismatched elements: 3693 / 8192 (45.1%)
Greatest absolute difference: nan at index (0, 15)
```

Controls that passed in the same harness:

- `r6-scaled-indexed-if-narrow-n32`
- `r6-scaled-indexed-loop-narrow-n32`
- `r6-scaled-subslice-if-n64-selector1`
- `r6-scaled-subslice-direct-n64-selector1`
- `r6-scaled-subslice-loop-n64`

Existing direct matrix controls also passed:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python pytest -s --tb=short \
python/test/gluon/test_tmem_runtime_matrix.py \
-k 'larger_cga or four_cta or num_ctas or twocta_layout'
```

The direct scaled-MMAv5 accumulator-subslice `use_acc` rows for
`N=64, slice_start=0` were included in the green scaled-MMAv5 split sweep.

Classification: stable runtime miscompile. Likely owner surface is
scaled-MMAv5 accumulator descriptor-view selection after dynamic `if`, with a
low-subslice/slice-origin interaction. This is not a backend repair request;
repair is deferred until the discovery campaign pivots.

## Non-Findings

- Existing scaled-MMAv5 indexed accumulator views, accumulator subslices,
  LHS subslices, tile-permuted/narrow accumulator layouts, B-scale descriptor
  views, and `use_acc` rows remained green in the focused matrix sweep.
- Existing plain MMAv5 indexed accumulator, accumulator subslice, two-CTA,
  tile-permuted, and LHS-subslice rows remained green.
- The temporary control-flow harness did not reproduce a failure for indexed
  scaled-MMAv5 accumulator descriptors under dynamic `if` or loop-carried
  selection.
- The same temporary harness did not reproduce a failure for the high
  accumulator subslice selected through dynamic `if`, direct selection, or
  loop-carried selection.

## Promotion Recommendation

Promote `FZ-20260421-0007` into
`python/test/gluon/test_tmem_structural_fuzzer.py` as a strict xfail when the
next promotion lane is open. The checked-in sentinel should be minimized from
the temporary harness to avoid importing from another test file and should
record the exact `selector=0` low-subslice dynamic-`if` boundary.
