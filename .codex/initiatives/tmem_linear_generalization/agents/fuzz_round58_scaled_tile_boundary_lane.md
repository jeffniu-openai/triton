# Round 58 Lane A: scaled-MMAv5 accumulator tile and narrow boundary

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `7c499fb4d`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code was edited.

## Scope

This lane focused on scaled-MMAv5 accumulator/tile/narrow boundaries that
were not the dynamic scaled descriptor lane from Round 56 and were more
targeted than the broad Round 54 scaled LHS/tile/narrow sweep.

Checked-in coverage came from
`python/test/gluon/test_tmem_runtime_matrix.py` and included:

- accumulator tile-permuted N32 and N64 rows;
- narrow accumulator tile-permuted N32/N64 and identity/use-acc rows;
- B-scale descriptor-view positives and N16 clean unsupported/error rows;
- mixed fp4A TMEM-LHS clean unsupported rows for subslice and tile-permuted
  LHS layouts;
- `use_acc` variants for tile-permuted N64 and narrow identity/indexed
  accumulator views.

No temporary Python probe was needed because the checked-in selector completed
without an unexpected failure.

## Required build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and mma_scaled and (acc_tile_permuted_32 or acc_tile_permuted_64 or acc_tile_permuted_narrow or acc_n16_bscale_descriptor_view_reports_clean_error or acc_n16_tile_permuted_reports_clean_unsupported or lhs_tile_permuted_mixed_fp4a_reports_clean_unsupported or lhs_subslice_view_mixed_fp4a_reports_clean_unsupported or acc_identity_narrow_format_use_acc or indexed_acc_identity_narrow_view_format_use_acc)'
```

Result:

```text
83/1615 tests collected (1532 deselected) in 1.97s
```

## Runtime Split

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix and mma_scaled and (acc_tile_permuted_32 or acc_tile_permuted_64 or acc_tile_permuted_narrow or acc_n16_bscale_descriptor_view_reports_clean_error or acc_n16_tile_permuted_reports_clean_unsupported or lhs_tile_permuted_mixed_fp4a_reports_clean_unsupported or lhs_subslice_view_mixed_fp4a_reports_clean_unsupported or acc_identity_narrow_format_use_acc or indexed_acc_identity_narrow_view_format_use_acc)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix and mma_scaled and (acc_tile_permuted_32 or acc_tile_permuted_64 or acc_tile_permuted_narrow or acc_n16_bscale_descriptor_view_reports_clean_error or acc_n16_tile_permuted_reports_clean_unsupported or lhs_tile_permuted_mixed_fp4a_reports_clean_unsupported or lhs_subslice_view_mixed_fp4a_reports_clean_unsupported or acc_identity_narrow_format_use_acc or indexed_acc_identity_narrow_view_format_use_acc)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix and mma_scaled and (acc_tile_permuted_32 or acc_tile_permuted_64 or acc_tile_permuted_narrow or acc_n16_bscale_descriptor_view_reports_clean_error or acc_n16_tile_permuted_reports_clean_unsupported or lhs_tile_permuted_mixed_fp4a_reports_clean_unsupported or lhs_subslice_view_mixed_fp4a_reports_clean_unsupported or acc_identity_narrow_format_use_acc or indexed_acc_identity_narrow_view_format_use_acc)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix and mma_scaled and (acc_tile_permuted_32 or acc_tile_permuted_64 or acc_tile_permuted_narrow or acc_n16_bscale_descriptor_view_reports_clean_error or acc_n16_tile_permuted_reports_clean_unsupported or lhs_tile_permuted_mixed_fp4a_reports_clean_unsupported or lhs_subslice_view_mixed_fp4a_reports_clean_unsupported or acc_identity_narrow_format_use_acc or indexed_acc_identity_narrow_view_format_use_acc)'
```

Result:

- group 1: `21 passed, 1594 deselected in 9.59s`
- group 2: `21 passed, 1594 deselected in 5.60s`
- group 3: `21 passed, 1594 deselected in 4.52s`
- group 4: `20 passed, 1595 deselected in 4.73s`
- aggregate: `83 passed`

## Classification

No new independent `FZ-*` bucket was found.

- `FZ-20260421-0007`: no new dynamic accumulator-selection signal. This lane
  used checked-in static accumulator tile/narrow rows and `use_acc` controls;
  all selected rows passed or produced their expected clean diagnostics.
- `FZ-20260421-0013`: no new B-scale descriptor-view wrong-result signal.
  Direct N32 B-scale descriptor-view positives and extra-user/rematerialized
  rows passed.
- `FZ-20260421-0015`: no new runtime-selected distinct B-scale signal. This
  lane did not enter dynamic B-scale selection; direct B-scale descriptor-view
  boundaries stayed green or clean unsupported as expected.

The clean unsupported/error rows stayed clean:

- mixed fp4A TMEM-LHS rows reported the expected `fp4_padded` operand-A
  diagnostic without `PassManager::run failed` or assertion text;
- N16 B-scale descriptor-view rows reported the expected descriptor-view
  `auto` layout diagnostic;
- N16 tile-permuted accumulator reported the expected public scaled-MMAv5
  instruction-tile diagnostic.

No compiler crash, verifier drift, false unsupported diagnostic,
clean-boundary drift, runtime miscompile, opcode-count mismatch, or hang was
observed.
