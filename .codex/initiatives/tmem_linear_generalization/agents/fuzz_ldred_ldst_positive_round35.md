# Round 35 Local: `ld.red` + Descriptor Positive Runtime Sweep

Date: 2026-04-21 13:35 UTC
Branch: `codex/tmem`
HEAD at start: `6623c94d9`
Scope: discovery only. No backend or checked-in test source was modified.

## Objective

Rerun the broad checked-in positive runtime surface for `tcgen05.ld.red` and
TMEM descriptor `ld/st` rows outside report-only, resource-boundary, and clean
diagnostic cases. The goal was to find compiler crashes, false unsupported
diagnostics, opcode drift, or runtime miscompiles while leaving backend repair
deferred.

## Build And Collection

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ld_red or ldst_descriptor) and not reports and not resource and not clean'
```

Result: `380/1615` rows collected.

## Four-GPU Sweep

Command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_ldred_ldst_positive_round35_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ld_red or ldst_descriptor) and not reports and not resource and not clean'
```

Results:

- Group 1: `44 passed, 51 skipped, 1520 deselected`
- Group 2: `81 passed, 10 skipped, 4 failed, 1520 deselected`
- Group 3: `93 passed, 2 failed, 1520 deselected`
- Group 4: `95 passed, 1520 deselected`

Aggregate selected result: `313 passed`, `61 skipped`, `6 failed`.

## Fresh Exact Reruns

All six failures reproduced in fresh pytest processes with stable per-GPU cache
directories:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]'
```

Each exact rerun failed during parsing with:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
Failed to lower TMEM load/store: unsupported dst layout
RuntimeError: error encountered during parsing
```

## Classification

No runtime wrong-result miscompile or hard process abort was observed in this
lane.

The six fresh-process failures expand existing `FZ-20260421-0012`: M64 f32
hardware `tcgen05.ld.red` destination-layout planning still rejects
non-identity row-basis layouts that are represented as checked-in positive
rows. The expansion covers both default layout selection and explicit
`32x32b` split-N requests:

- `row_reverse_n32` min/max default layout;
- `row_rotate_col_even_odd_n128` min/max default layout;
- `row_reverse_n32` explicit `32x32b` split-N;
- `row_rotate_col_even_odd_n128` explicit `32x32b` split-N.

Backend repair remains deferred under the discovery-only fuzzing contract.
