# Round 9 Lane H: TMEM Fuzzing Validation Logistics

Date: 2026-04-21 09:35 UTC

Scope: discovery-only validation logistics. No backend/compiler fixes were attempted. Repo edits are limited to this report.

## Summary

The current checked-in structural fuzzer is small enough to run as a frequent smoke gate, but it is duration-skewed by a few crash/allocator/2CTA rows. The runtime matrix remains too broad for casual discovery with unqualified selectors: the full file collects 1,615 nodeids, and broad `mma_scaled`, `twocta`, `ldst and descriptor`, and `ld_red` selectors each collect hundreds of nodeids. Practical next sweeps should use family-specific slices around 25-60 nodeids, store durations from one representative run, then use `pytest-split` with `--splitting-algorithm=least_duration` for 4-GPU reruns.

No new compiler/backend `FZ-*` bucket was found during this logistics pass.

## Branch State

- Branch was clean before report creation: `## codex/tmem...origin/codex/tmem`.
- Required pre-test build command:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Collection Inventory

Exact commands and observed collection sizes:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `33 tests collected in 3.28s`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `1615 tests collected in 3.35s`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and acc_subslice and use_acc'
```

Result: `28/1615 tests collected`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red or ldred'
```

Result: `247/1615 tests collected`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst and descriptor'
```

Result: `273/1615 tests collected`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'copy'
```

Result: `47/1615 tests collected`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled'
```

Result: `271/1615 tests collected`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'twocta'
```

Result: `370/1615 tests collected`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst and not descriptor and not lifetime'
```

Result: `117/1615 tests collected`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red and descriptor'
```

Result: `50/1615 tests collected`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'reports_clean_unsupported or resource_boundary'
```

Result: `96/1615 tests collected`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_structural_fuzzer.py -k 'ldred or ld_red'
```

Result: `11/33 tests collected`.

Useful compact runtime-matrix slices:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_descriptor_roundtrip_sweeps and not twocta'
```

Result: `27/1615 tests collected`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_descriptor_chain and not non_f32'
```

Result: `30/1615 tests collected`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'copy and warpx2'
```

Result: `28/1615 tests collected`.

## Representative Runtime Probes

Structural fuzzer:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-round9-lane-h-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --durations=10 --store-durations --durations-path /tmp/tmem_round9_lane_h_structural_durations.json python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `9 passed, 24 xfailed in 27.10s`.

Slowest rows:

- `test_tmem_structural_fuzzer_ldred[ldred-twocta-lifted-256x64]`: `4.32s`
- `test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]`: `1.47s`
- `test_tmem_structural_fuzzer_ldred_twocta_rowcol_optimizer_crash`: `1.32s`
- `test_tmem_structural_fuzzer_ldred_1cta_direct_index_allocator_crash`: `1.29s`
- `test_tmem_structural_fuzzer_ldst_256row_lifted_parent_allocator_crash`: `1.28s`

Scaled MMA accumulator subslice `use_acc`:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-round9-lane-h-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --durations=10 --store-durations --durations-path /tmp/tmem_round9_lane_h_scaled_use_acc_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and acc_subslice and use_acc'
```

Result: `28 passed, 1587 deselected in 19.21s`.

Slowest rows:

- `test_tmem_runtime_matrix_mma_scaled_acc_subslice_view_format_use_acc[mxfp8-mxfp8-64-0-128]`: `1.90s`
- `test_tmem_runtime_matrix_mma_scaled_acc_subslice_view_format_use_acc[mxfp8-mxfp8-128-128-256]`: `0.98s`
- `test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_use_acc[mxfp8-mxfp4-64-64-128-True]`: `0.70s`

Copy `warpx2`:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-round9-lane-h-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --durations=10 --store-durations --durations-path /tmp/tmem_round9_lane_h_copy_warpx2_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'copy and warpx2'
```

Result: `28 passed, 1587 deselected in 24.08s`.

Slowest rows:

- `test_tmem_runtime_matrix_cp_no_scales_warpx2_subslice_view_positive[01_23-...]`: `1.39s`
- Four indexed-view positive rows around `01_23`: `1.09s-1.10s`
- Four indexed-view positive rows around `02_13`: `0.87s-0.88s`

LD/ST descriptor roundtrip slice:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-round9-lane-h-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --durations=10 --store-durations --durations-path /tmp/tmem_round9_lane_h_ldst_desc_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_descriptor_roundtrip_sweeps and not twocta'
```

Result: `27 skipped, 1588 deselected in 3.41s`.

This selector is useful for collection sizing, but it did not provide runtime timing on this host/configuration. Do not use it as a duration calibration source unless the target environment actually runs those rows.

## Duration-Aware Split Checks

Structural fuzzer split with stored durations:

```bash
PYTHONPATH=.:./python pytest --collect-only -q --splits 4 --group 1 --splitting-algorithm=least_duration --durations-path /tmp/tmem_round9_lane_h_structural_durations.json python/test/gluon/test_tmem_structural_fuzzer.py
PYTHONPATH=.:./python pytest --collect-only -q --splits 4 --group 2 --splitting-algorithm=least_duration --durations-path /tmp/tmem_round9_lane_h_structural_durations.json python/test/gluon/test_tmem_structural_fuzzer.py
PYTHONPATH=.:./python pytest --collect-only -q --splits 4 --group 3 --splitting-algorithm=least_duration --durations-path /tmp/tmem_round9_lane_h_structural_durations.json python/test/gluon/test_tmem_structural_fuzzer.py
PYTHONPATH=.:./python pytest --collect-only -q --splits 4 --group 4 --splitting-algorithm=least_duration --durations-path /tmp/tmem_round9_lane_h_structural_durations.json python/test/gluon/test_tmem_structural_fuzzer.py
```

Collected groups: `6`, `9`, `9`, `9`. The uneven node count is expected because the one slow 4.32s row is isolated into a smaller group.

Scaled MMA `use_acc` split with stored durations:

```bash
PYTHONPATH=.:./python pytest --collect-only -q --splits 4 --group 1 --splitting-algorithm=least_duration --durations-path /tmp/tmem_round9_lane_h_scaled_use_acc_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and acc_subslice and use_acc'
PYTHONPATH=.:./python pytest --collect-only -q --splits 4 --group 2 --splitting-algorithm=least_duration --durations-path /tmp/tmem_round9_lane_h_scaled_use_acc_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and acc_subslice and use_acc'
PYTHONPATH=.:./python pytest --collect-only -q --splits 4 --group 3 --splitting-algorithm=least_duration --durations-path /tmp/tmem_round9_lane_h_scaled_use_acc_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and acc_subslice and use_acc'
PYTHONPATH=.:./python pytest --collect-only -q --splits 4 --group 4 --splitting-algorithm=least_duration --durations-path /tmp/tmem_round9_lane_h_scaled_use_acc_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and acc_subslice and use_acc'
```

Collected groups: `5`, `7`, `8`, `8`. This is a good candidate for duration-aware 4-GPU reruns because the slow first row otherwise dominates a count-balanced group.

## Recommended Next Sweep Slices

Use these as practical next subagent/runtime lanes:

1. Frequent smoke gate:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py
```

Expected current result: `9 passed, 24 xfailed`. Store durations after structural fuzzer changes, then run 4-way least-duration splitting when this file grows beyond roughly 50-60 nodeids.

2. Scaled MMA control-flow/accumulator-view lane:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group <1..4> --splitting-algorithm=least_duration --durations-path /tmp/tmem_round9_lane_h_scaled_use_acc_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and acc_subslice and use_acc'
```

This is the best focused runtime-matrix slice for `FZ-20260421-0007` follow-up: 28 nodeids, all runtime-reaching here, and currently green.

3. Copy/readback lane:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'copy and warpx2'
```

This is a compact 28-nodeid positive slice that exercises indexed/subslice copy paths and takes about 24s on one GPU with a fresh cache.

4. LD.RED descriptor lane:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_descriptor_chain and not non_f32'
```

This collects 30 nodeids and is a better next duration-calibration target than broad `ld_red`.

5. Clean-negative diagnostic lane:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k 'reports_clean_unsupported or resource_boundary'
```

This collects 96 nodeids. Split it across 4 GPUs after a stored-duration calibration; do not run it as an incidental smoke unless the goal is specifically diagnostic-boundary coverage.

## Selectors To Avoid For Casual Sweeps

Avoid these unqualified selectors unless deliberately running a larger lane:

- `python/test/gluon/test_tmem_runtime_matrix.py`: 1,615 nodeids. This is a full-file sweep, not a logistics or smoke command.
- `-k 'twocta'`: 370 nodeids and crosses LD/ST, copy, MMA, scaled MMA, and diagnostic contracts. It is too heterogeneous for quick triage.
- `-k 'ldst and descriptor'`: 273 nodeids and includes many families; split by exact test function or by descriptor/readback intent.
- `-k 'mma_scaled'`: 271 nodeids; split into accumulator-view, LHS-view, tile-permuted, narrow, and clean-unsupported lanes.
- `-k 'ld_red or ldred'`: 247 nodeids; split into descriptor-chain, row/col permutation, non-f32 software-reduce, and resource-boundary lanes.

## Logistics Recommendations

- Always run `make -j8` first, even for collection-only passes, to preserve the project rule and avoid stale extension state.
- For subagent discovery, require a collection count in every report before runtime execution.
- Keep normal lanes to roughly 25-60 selected runtime nodeids per one-GPU representative run.
- Store durations for any slice that will be rerun more than once:

```bash
--store-durations --durations-path /tmp/<stable-slice-name>.json
```

- Use duration-aware 4-GPU splitting for skewed slices:

```bash
--splits 4 --group <1..4> --splitting-algorithm=least_duration --durations-path /tmp/<stable-slice-name>.json
```

- Keep structural fuzzer crash sentinels subprocess-isolated. They are useful smoke coverage, but they should remain strict xfails until fuzzing stops and backend fixes begin.
- Do not use selectors that only collect skipped rows as timing calibration sources. The `ldst_descriptor_roundtrip_sweeps and not twocta` probe collected 27 rows but skipped all 27 here.
