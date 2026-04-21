# Round 14 Local Descriptor and ld.red Selector Sweep

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler fixes were attempted.

## Required rebuild

```bash
make -j8
```

Result: `ninja: no work to do`.

## ld/st descriptor algebra selector

Collect-only:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'runtime_selector or descriptor_roundtrip or descriptor_compositions'
```

Result: `116/1615` selected.

Runtime pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  --store-durations --durations-path /tmp/tmem_local_r14_descriptor_selector_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'runtime_selector or descriptor_roundtrip or descriptor_compositions'
```

Results:

- group 1/GPU 0: `29 passed, 1586 deselected`;
- group 2/GPU 1: `9 passed, 20 skipped, 1586 deselected`;
- group 3/GPU 2: `29 skipped, 1586 deselected`;
- group 4/GPU 3: `29 skipped, 1586 deselected`.

Aggregate: `38 passed, 78 skipped`.

Classification: no new bucket. Checked-in descriptor roundtrips,
compositions, two-CTA variants, runtime-selector-adjacent rows, and row/column
permuted descriptor sweeps remained stable.

## ld.red M64 / descriptor / rowcol selector

Collect-only:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and (m64 or descriptor or rowcol_permuted) and not reports'
```

Result: `100/1615` selected.

Runtime pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  --store-durations --durations-path /tmp/tmem_local_r14_ldred_m64_descriptor_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and (m64 or descriptor or rowcol_permuted) and not reports'
```

Results:

- group 1/GPU 0: `21 passed, 4 failed, 1590 deselected`;
- group 2/GPU 1: `23 passed, 2 failed, 1590 deselected`;
- group 3/GPU 2: `25 passed, 1590 deselected`;
- group 4/GPU 3: `25 passed, 1590 deselected`.

Aggregate: `94 passed, 6 failed`.

Classification: no new bucket. The six failures are the known
`FZ-20260421-0012` M64 `tcgen05.ld.red` destination-layout lowering gap for
effective non-identity row bases:

- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]`

Representative diagnostic:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
requested layout direct-lowering details:
Failed to lower TMEM load/store: unsupported dst layout
```

The surrounding M64 split-N rows, descriptor-chain reductions, non-f32
software reductions, and row/column-permuted non-M64 controls in this selector
remained green.
