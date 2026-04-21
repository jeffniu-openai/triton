# Round 45 High-CGA/M64 Breadth Probe

Date: 2026-04-21 14:41 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Run a mixed checked-in runtime selector over recently sensitive high-CGA,
two-CTA/TMA, scale-descriptor, and M64 surfaces. The goal was to check whether
the adjacent high-CGA positives still pass while the known M64 `ld.red`
destination-layout failures remain isolated to existing `FZ-20260421-0012`
rows.

## Required Build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Runtime Command

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short \
  -k 'ld_red_m64 or mma_m64 or twocta_tma_tf32_b_transposed_descriptor or cp_no_scales_twocta_layout_in_4cta_context or ldst_scales_descriptor_view_cga_roundtrip' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result:

```text
6 failed, 86 passed, 1523 deselected
```

## Failed Rows

All failures matched existing `FZ-20260421-0012` M64 f32 `tcgen05.ld.red`
destination-layout planner gaps for non-identity row/rowcol bases:

- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]`

Representative diagnostic:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
Failed to lower TMEM load/store: unsupported dst layout
```

## Passing Surface

The remaining `86` selected rows passed, including the adjacent M64 MMA,
two-CTA TMA transposed-descriptor, high-CGA clean-boundary, and scale
descriptor-view CGA rows selected by the probe.

## Classification

No new independent `FZ-*` bucket is proposed. The six failures are existing
`FZ-20260421-0012`; the high-CGA/two-CTA adjacent rows stayed green.

Backend repair remains deferred per the discovery-only campaign.
