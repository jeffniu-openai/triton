# Round 42: high-CGA ownership and stable-cache fuzzing

Date: 2026-04-21 14:20 UTC
Branch: `codex/tmem`
Scope: discovery/classification only. No backend/compiler files modified.
Owned artifact: `.codex/initiatives/tmem_linear_generalization/agents/fuzz_high_cga_cache_round42.md`.

## Objective

Adversarially recheck TMEM high-CGA ownership behavior under stable per-GPU
caches, with emphasis on:

- kernel `num_ctas > 2` contexts;
- local 1CTA/2CTA TMEM layouts in larger-CGA launches;
- adjacent TMA multicast, MMAv5 multicast/commit, scaled-MMAv5/copy, and
  `ld.red` rows;
- repeated process-boundary runs using stable `TRITON_CACHE_DIR` values;
- whether clean diagnostics drift, positives become cache/process sensitive, or
  `FZ-20260421-0010` needs sharper evidence.

## Environment and Setup

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: no work to do.
```

The ambient shell initially resolved `triton` from
`/tmp/triton-upstream-main-check/python`, causing collection-only import
errors. All actual classification runs below used `PYTHONPATH=./python` so they
tested this checkout and the rebuilt tree.

Stable caches used throughout:

- GPU 0: `TRITON_CACHE_DIR=/tmp/triton-cache-gpu0`
- GPU 1: `TRITON_CACHE_DIR=/tmp/triton-cache-gpu1`
- GPU 2: `TRITON_CACHE_DIR=/tmp/triton-cache-gpu2`
- GPU 3: `TRITON_CACHE_DIR=/tmp/triton-cache-gpu3`

No fresh temporary cache was used to mask failures.

## Collections

Focused high-CGA/multicast selector:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python \
  pytest --collect-only -q -s --tb=short \
  -k "layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma" \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_core.py
```

Result: `28/19762` collected. The selected set included:

- `test_tmem_runtime_matrix_cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error`;
- two-CTA TMA/MMAv5 descriptor rows in `test_tmem_runtime_matrix.py`;
- `test_tma_multicast_copy`;
- `test_tcgen05_mma_multicast_commit`;
- `test_tcgen05_mma_multicast_commit_twocta_linear_acc`;
- `test_tcgen05_mma_scaled_direct_multicast_barrier`.

Broader adjacent selector:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=./python \
  pytest --collect-only -q -s --tb=short \
  -k "(layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma or scaled_direct_multicast or cp_scales_warpx4 or mma_scaled_twocta_acc_subslice or ld_red) and not reports and not resource" \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_core.py
```

Result: `317/19729` collected. This intentionally pulled in broad `ld.red`,
scaled-copy/MMAv5, two-CTA TMA, and multicast positives adjacent to the
high-CGA ownership rows.

## Focused High-CGA Split Run

Command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> PYTHONPATH=./python \
  pytest -s --tb=short --splits 4 --group <group> \
  -k "layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma" \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_core.py
```

Results:

- group 1, GPU 0: `7 passed`
- group 2, GPU 1: `7 passed`
- group 3, GPU 2: `7 passed`
- group 4, GPU 3: `7 passed`

Aggregate: `28 passed`.

Classification:

- Legal TMA multicast, MMAv5 multicast/commit, two-CTA TMA/MMAv5 descriptor,
  and scaled direct multicast controls stayed green.
- The checked-in 2CTA layout in 4CTA launch-context diagnostic remained a
  clean expected diagnostic and did not surface as a crash, process abort, or
  wrong result.
- No high-CGA positive became process/cache sensitive in this focused pass.
- No new independent `FZ-*`.

## Broader Adjacent Split Run

Command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> PYTHONPATH=./python \
  pytest -s --tb=short --splits 4 --group <group> \
  -k "(layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma or scaled_direct_multicast or cp_scales_warpx4 or mma_scaled_twocta_acc_subslice or ld_red) and not reports and not resource" \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_core.py
```

Results:

- group 1, GPU 0: `74 passed, 6 failed`
- group 2, GPU 1: `80 passed`
- group 3, GPU 2: `80 passed`
- group 4, GPU 3: `77 passed`

Aggregate: `311 passed, 6 failed`.

The six failures are existing `FZ-20260421-0012` M64 row/row-column permuted
`ld.red` rows, not high-CGA ownership failures:

- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]`

Failure signature:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
Failed to lower TMEM load/store: unsupported dst layout
RuntimeError: error encountered during parsing
```

Classification:

- Existing `FZ-20260421-0012`, already documented in prior M64 `ld.red`
  row-permuted reports.
- Not `FZ-20260421-0010`: no local 1CTA/2CTA layout in a larger high-CGA
  context, and no layout/context CTA-count diagnostic.
- No crash, no runtime wrong result, and no new candidate bucket.
- The broader selector is intentionally noisy because it includes live known-red
  non-report `ld.red` rows; it should not be treated as a green guardrail.

## Stable-Cache Rechecks

Repeated `FZ-20260421-0010` diagnostic row after the broader sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python \
  pytest -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error
```

Result: `1 passed`.

Immediate same-cache rerun on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python \
  pytest -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error
```

Result: `1 passed`.

Cross-cache/process rerun on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=./python \
  pytest -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error
```

Result: `1 passed`.

Known-red `FZ-20260421-0012` contrast rerun:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=./python \
  pytest -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]'
```

Result: `1 failed` with the same `unsupported dst layout` diagnostic. This
confirms the broader-selector red row is stable known evidence, not a
cache-sensitive high-CGA drift.

## Classification Matrix

| Surface | Result | Classification |
| --- | --- | --- |
| Focused high-CGA/multicast selector | `28 passed` split-4 | Green guardrail |
| 2CTA layout in 4CTA context clean diagnostic | `1 passed` after warm broader run; repeated GPU0/GPU1 passes | Existing `FZ-20260421-0010`, stable |
| TMA multicast controls | Passed inside focused selector | Green control |
| MMAv5 multicast/commit controls | Passed inside focused selector | Green control |
| Scaled direct multicast control | Passed inside focused selector | Green control |
| Broader adjacent selector | `311 passed, 6 failed` | Six existing `FZ-20260421-0012`; no new high-CGA signal |
| M64 row-permuted `ld.red` contrast | Reproduced same `unsupported dst layout` | Existing `FZ-20260421-0012`, stable |

## FZ-20260421-0010 Evidence Update

No sharpening beyond prior Round 37/39 evidence is required from this lane.
The strongest current statement remains:

- valid high-CGA kernels exist and stay green for TMA multicast,
  MMAv5 multicast/commit, and scaled-MMAv5 controls;
- local 1CTA/2CTA TMEM layouts embedded in larger `num_ctas > 2` contexts are
  still rejected by the layout/context CTA-count gate;
- the checked-in 2CTA-in-4CTA clean diagnostic remained stable across process
  boundaries and warm stable caches;
- no evidence here suggests the diagnostic is cache-sensitive, process-order
  sensitive, or masking a later backend crash.

## Candidate New Buckets

None.

The only red rows seen in the expanded selector are existing
`FZ-20260421-0012` M64 row/row-column-permuted `ld.red` clean unsupported-layout
failures. No high-CGA ownership, multicast/TMA/MMAv5/scaled-copy, or stable-cache
row produced a new crash, false unsupported diagnostic, runtime miscompile, or
unexpected pass/fail transition.

