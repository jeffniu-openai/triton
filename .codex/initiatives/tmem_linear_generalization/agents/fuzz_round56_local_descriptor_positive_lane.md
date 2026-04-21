# Round 56 Local: descriptor composition and positive roundtrip lane

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `57262fc4a`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code was edited.

## Scope

This local lane sampled positive TMEM load/store descriptor composition
coverage outside the known `ld.red`, scale, x1, and 2CTA surfaces:

- basic descriptor composition rows;
- slice/index roundtrip chains;
- deep roundtrip chains;
- multidimensional positive slices;
- direct half-row positive rows;
- rank-5 descriptor roundtrip rows.

The lane intentionally excluded `reports`, `twocta`, `ld_red`, `x1`, and
`scales` selectors so it would not overlap the active Round 56 copy, scaled, or
mixed `ld.red` lanes.

## Required build

```bash
make -j8
```

Result: `ninja: no work to do`.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst_descriptor and (compositions or roundtrip_sweeps or rank5_roundtrip or multidim_slice_positive or higher_rank_dim0_slice_positive or higher_rank_half_rows_positive or direct_half_rows_positive) and not reports and not twocta and not ld_red and not x1 and not scales'
```

Result: `19/1615` collected.

## Runtime split

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_descriptor and (compositions or roundtrip_sweeps or rank5_roundtrip or multidim_slice_positive or higher_rank_dim0_slice_positive or higher_rank_half_rows_positive or direct_half_rows_positive) and not reports and not twocta and not ld_red and not x1 and not scales'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_descriptor and (compositions or roundtrip_sweeps or rank5_roundtrip or multidim_slice_positive or higher_rank_dim0_slice_positive or higher_rank_half_rows_positive or direct_half_rows_positive) and not reports and not twocta and not ld_red and not x1 and not scales'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_descriptor and (compositions or roundtrip_sweeps or rank5_roundtrip or multidim_slice_positive or higher_rank_dim0_slice_positive or higher_rank_half_rows_positive or direct_half_rows_positive) and not reports and not twocta and not ld_red and not x1 and not scales'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_descriptor and (compositions or roundtrip_sweeps or rank5_roundtrip or multidim_slice_positive or higher_rank_dim0_slice_positive or higher_rank_half_rows_positive or direct_half_rows_positive) and not reports and not twocta and not ld_red and not x1 and not scales'
```

Result:

- group 1: `3 passed, 2 skipped, 1610 deselected`
- group 2: `1 passed, 4 skipped, 1610 deselected`
- group 3: `1 passed, 4 skipped, 1610 deselected`
- group 4: `4 skipped, 1611 deselected`
- aggregate: `5 passed, 14 skipped`

The skipped rows stayed under existing local capability guards for the selected
positive descriptor variants. No selected executed row failed.

## Classification

No new independent `FZ-*` bucket was found. This is a green/skip-bounded
positive descriptor-composition guardrail rather than a new runtime failure
surface.
