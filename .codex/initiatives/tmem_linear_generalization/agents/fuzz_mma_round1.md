# Lane C Round 1: Plain and Scaled MMAv5 Runtime Fuzzing

- Time: 2026-04-21 08:25 UTC
- Branch: `codex/tmem`
- HEAD: `87c50a242`
- Mode: discovery only. No backend code was changed.
- Scope: plain MMAv5 and scaled-MMAv5 Python/Gluon runtime probes for accumulator descriptor views, `use_acc`, indexed/subslice accumulators, narrow N, B-scale descriptor views and rematerialization, two-CTA paths, K=128/256, and scaled format mixes.

## Result

No compiler crash, verifier false unsupported diagnostic, opcode mismatch, or runtime miscompile was found in this lane C round.

## Build

- Command: `make`
- Result: passed, `ninja: no work to do`.

## Probe Set C1: Broad MMA / Scaled-MMA Runtime Selector

- Collected: `295/1719` selected.
- Command:
  `PYTHONPATH=.:./python pytest --collect-only -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_fpsan.py -k '(mma_plain_kinds_tile_permuted_acc or mma_lhs_tile_permuted or mma_lhs_subslice_view_plain_kinds or mma_scaled_root_format or mma_scaled_indexed_acc_view or mma_scaled_acc_subslice or mma_scaled_acc_tile_permuted or mma_scaled_acc_identity_narrow or mma_scaled_indexed_acc_identity_narrow or mma_scaled_acc_n16 or mma_scaled_lhs_subslice_view_format or mma_scaled_lhs_tile_permuted_format or tcgen05_mma_twocta or tcgen05_mma_scaled)'`

Runtime commands:

- `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_fpsan.py -k '(mma_plain_kinds_tile_permuted_acc or mma_lhs_tile_permuted or mma_lhs_subslice_view_plain_kinds or mma_scaled_root_format or mma_scaled_indexed_acc_view or mma_scaled_acc_subslice or mma_scaled_acc_tile_permuted or mma_scaled_acc_identity_narrow or mma_scaled_indexed_acc_identity_narrow or mma_scaled_acc_n16 or mma_scaled_lhs_subslice_view_format or mma_scaled_lhs_tile_permuted_format or tcgen05_mma_twocta or tcgen05_mma_scaled)'`
  - Result: `74 passed, 1645 deselected in 86.31s`.
- Same command with `CUDA_VISIBLE_DEVICES=1`, `TRITON_CACHE_DIR=/tmp/triton-cache-gpu1`, `--group 2`.
  - Result: `74 passed, 1645 deselected in 53.24s`.
- Same command with `CUDA_VISIBLE_DEVICES=2`, `TRITON_CACHE_DIR=/tmp/triton-cache-gpu2`, `--group 3`.
  - Result: `74 passed, 1645 deselected in 97.76s`.
- Same command with `CUDA_VISIBLE_DEVICES=3`, `TRITON_CACHE_DIR=/tmp/triton-cache-gpu3`, `--group 4`.
  - Result: `72 passed, 1 skipped, 1646 deselected in 110.61s`.

Coverage notes:

- Plain MMAv5: tile-permuted accumulator layouts, LHS tile-permuted TMEM, LHS subslice descriptor views, `use_acc`, f16/tf32/bf16/f8 variants, N=32/64/128/256, K=32/64/128/256 where covered by existing matrix rows.
- Scaled-MMAv5: root/indexed/subslice accumulator views, `use_acc`, narrow N=16/32/64, K=128/256, mxfp8/mxfp4/nvfp4 and mixed mxfp8/mxfp4 format pairs.
- B-scale descriptor views: tile-permuted N=32 descriptor-view, extra-user rematerialization, padded descriptor-view, and clean N=16 descriptor-view diagnostics.
- Clean-negative checks: N=16 tile-permuted scaled accumulator and FPSAN asymmetric/unsupported layout paths remained clean, without `PassManager::run failed` or assertion-style crashes.

## Probe Set C2: FPSAN Plain MMAv5 Payload Oracle

- Collected: `16/104` selected.
- Command:
  `PYTHONPATH=.:./python pytest --collect-only -q -s --tb=short python/test/gluon/test_fpsan.py -k 'tcgen05_mma and not scaled and not twocta'`

Runtime commands:

- `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_fpsan.py -k 'tcgen05_mma and not scaled and not twocta'`
  - Result: `4 passed, 100 deselected in 7.86s`.
- Same command with `CUDA_VISIBLE_DEVICES=1`, `TRITON_CACHE_DIR=/tmp/triton-cache-gpu1`, `--group 2`.
  - Result: `2 passed, 2 skipped, 100 deselected in 7.85s`.
- Same command with `CUDA_VISIBLE_DEVICES=2`, `TRITON_CACHE_DIR=/tmp/triton-cache-gpu2`, `--group 3`.
  - Result: `2 passed, 2 skipped, 100 deselected in 30.45s`.
- Same command with `CUDA_VISIBLE_DEVICES=3`, `TRITON_CACHE_DIR=/tmp/triton-cache-gpu3`, `--group 4`.
  - Result: `4 passed, 100 deselected in 4.20s`.

Coverage notes:

- Plain MMAv5 FPSAN payload checks covered accumulator descriptor views, `use_acc`, legacy and linear accumulator layouts, expected clean unsupported layouts, and bit-level payload equality.
- The skipped cases are existing clean skips for lifted legacy accumulator views.

## Probe Set C3: Scaled-MMAv5 Two-CTA Accumulator Subslice Views

- Collected: `28/1615` selected.
- Command:
  `PYTHONPATH=.:./python pytest --collect-only -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (twocta or tcgen05_copy or acc_subslice_copy or cp_scales_warpx4_twocta or direct_multicast)'`

Runtime commands:

- `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and twocta_acc_subslice_view_format'`
  - Result: `7 passed, 1608 deselected in 6.16s`.
- Same command with `CUDA_VISIBLE_DEVICES=1`, `TRITON_CACHE_DIR=/tmp/triton-cache-gpu1`, `--group 2`.
  - Result: `7 passed, 1608 deselected in 6.25s`.
- Same command with `CUDA_VISIBLE_DEVICES=2`, `TRITON_CACHE_DIR=/tmp/triton-cache-gpu2`, `--group 3`.
  - Result: `7 passed, 1608 deselected in 7.36s`.
- Same command with `CUDA_VISIBLE_DEVICES=3`, `TRITON_CACHE_DIR=/tmp/triton-cache-gpu3`, `--group 4`.
  - Result: `7 passed, 1608 deselected in 3.88s`.

Coverage notes:

- Scaled two-CTA accumulator subslice views covered direct and `use_acc` paths, N=32/64/128, K=128/256, slice starts 0/32/64/128, multicast true/false, and mxfp8/mxfp4/nvfp4 format mixes.

## Probe Set C4: Plain MMAv5 Indexed and Two-CTA Accumulator Views

- Collected: `67/1615` selected.
- Command:
  `PYTHONPATH=.:./python pytest --collect-only -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k '(mma_indexed_acc_view or mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds)'`

Runtime commands:

- `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(mma_indexed_acc_view or mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds)'`
  - Result: `17 passed, 1598 deselected in 20.39s`.
- Same command with `CUDA_VISIBLE_DEVICES=1`, `TRITON_CACHE_DIR=/tmp/triton-cache-gpu1`, `--group 2`.
  - Result: `17 passed, 1598 deselected in 18.33s`.
- Same command with `CUDA_VISIBLE_DEVICES=2`, `TRITON_CACHE_DIR=/tmp/triton-cache-gpu2`, `--group 3`.
  - Result: `17 passed, 1598 deselected in 17.61s`.
- Same command with `CUDA_VISIBLE_DEVICES=3`, `TRITON_CACHE_DIR=/tmp/triton-cache-gpu3`, `--group 4`.
  - Result: `16 passed, 1599 deselected in 11.14s`.

Coverage notes:

- Plain MMAv5 indexed accumulator views and two-CTA indexed/subslice accumulator views covered direct and `use_acc` paths, legacy and linear parent layouts, unit-parent linear views, N=32/64/128/256, K=32/64/128, and f16/tf32/bf16/f8 variants where covered by existing rows.

## Suggested Next Lane C Cases

- Add an ad-hoc B-scale descriptor-view/rematerialization variant with K=256, because current B-scale descriptor-view rows are strongest at K=128 while other scaled-MMA rows cover K=256.
- Add a two-CTA plain-MMAv5 runtime probe with K=128/256 and accumulator view chains; the current FPSAN two-CTA plain path uses smaller K while one-CTA runtime-matrix rows cover broader K.
- Add scaled-MMAv5 B-scale descriptor-view cases combined with two-CTA accumulator subslice views, if the frontend can express the scale descriptor view without turning the test into a copy-family probe.
