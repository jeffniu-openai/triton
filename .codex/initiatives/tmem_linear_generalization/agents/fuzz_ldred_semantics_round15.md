# Round 15 Lane AL: ld.red semantics fuzzing

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend/compiler code was changed. No commit
or push was made by this lane.

## Summary

Lane AL focused on `tcgen05.ld.red` semantics and lowering boundaries:

- f32 hardware `ld.red` modifiers (`min`/`max`, `abs`, NaN propagation);
- descriptor-chain versus direct layouts;
- integer/f16/bf16 software-reduce contract boundaries;
- M64, M128, M256 direct and descriptor-chain shapes;
- 1CTA versus 2CTA structural sentinels, including indexed parent rows;
- opcode inspection for `.ld.red` versus fallback plain `ld`.

No new independent `FZ-*` candidate was found. The broad checked-in runtime
matrix reproduces the already-known M64 permuted row-base unsupported bucket
(`FZ-20260421-0012`). Checked-in structural sentinels continue to classify the
2CTA indexed/fallback, allocator, and row/col crash surfaces as existing
`FZ-20260421-0004`, `FZ-20260421-0005`, `FZ-20260421-0006`,
`FZ-20260421-0008`, and `FZ-20260421-0009`.

The temporary semantic probe added coverage that is not fully represented by
the checked-in representative selectors:

- descriptor-chain f32 reductions with both `min` and `max`, `abs=True`, and
  `PropagateNan.ALL`;
- descriptor-chain integer `int8`/`int16`/`int32` reductions with `abs=True`
  using software reduce and no `.ld.red` opcode;
- M256 descriptor-chain identity smoke for N=64 and N=128;
- M64 identity controls with `max.abs.NaN`.

All temporary probe rows passed.

## Required rebuild

Command:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Checked-in ld.red runtime matrix

Collect-only command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports'
```

Result: `243/1615` selected.

Runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  --store-durations --durations-path /tmp/tmem_lane_al_round15_ldred_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `6 failed, 55 passed, 1554 deselected in 8.80s` |
| 2 | 1 | `61 passed, 1554 deselected in 20.34s` |
| 3 | 2 | `61 passed, 1554 deselected in 24.81s` |
| 4 | 3 | `60 passed, 1555 deselected in 16.11s` |

Aggregate: `237 passed`, `6 failed`.

The six failures are the existing M64 permuted row-base unsupported cases:

```text
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]
```

Failure signature:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
requested layout direct-lowering details:
Failed to lower TMEM load/store: unsupported dst layout
```

Classification: existing `FZ-20260421-0012`. This lane avoided duplicating that
bucket except as a baseline control.

Logs:

- `/tmp/tmem_lane_al_round15_ldred_g1.log`
- `/tmp/tmem_lane_al_round15_ldred_g2.log`
- `/tmp/tmem_lane_al_round15_ldred_g3.log`
- `/tmp/tmem_lane_al_round15_ldred_g4.log`

## Temporary semantic probe

Temporary file:

```text
/tmp/tmem_ldred_semantics_round15_probe.py
```

Collect-only command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_ldred_semantics_round15_probe.py
```

Result: `16 tests collected`.

Runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group <group> \
  /tmp/tmem_ldred_semantics_round15_probe.py
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `4 passed, 12 deselected in 8.32s` |
| 2 | 1 | `4 passed, 12 deselected in 8.60s` |
| 3 | 2 | `4 passed, 12 deselected in 7.22s` |
| 4 | 3 | `4 passed, 12 deselected in 5.68s` |

Aggregate: `16 passed`.

Case coverage:

| Case family | Rows | Result | Classification |
| --- | ---: | --- | --- |
| f32 descriptor-chain full modifiers: identity, row-reverse, row/col rotate+reverse with `min/max`, `abs=True`, `PropagateNan.ALL` | 6 | passed with `.ld.red` in PTX/LLIR | green |
| descriptor-chain integer `int8`/`int16`/`int32` with `min/max`, `abs=True` | 6 | passed with software reduce; no `.ld.red` opcodes | green contract boundary |
| M256 descriptor-chain identity, N=64 and N=128 | 2 | passed with expected `.ld.red` shapes | green |
| M64 identity controls, N=32 and N=64, `max.abs.NaN` | 2 | passed with expected split-N `.ld.red` shapes | green |

Logs:

- `/tmp/tmem_lane_al_round15_probe_g1.log`
- `/tmp/tmem_lane_al_round15_probe_g2.log`
- `/tmp/tmem_lane_al_round15_probe_g3.log`
- `/tmp/tmem_lane_al_round15_probe_g4.log`

## Checked-in structural ld.red sentinels

Collect-only command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'ldred and not allocator and not optimizer and not direct_index'
```

Result: `9/33` selected.

Runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'ldred and not allocator and not optimizer and not direct_index'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `3 passed, 30 deselected in 4.22s` |
| 2 | 1 | `30 deselected, 3 xfailed in 6.92s` |
| 3 | 2 | `30 deselected, 3 xfailed in 5.21s` |
| 4 | 3 | `33 deselected in 2.96s` |

Aggregate selected rows: `3 passed`, `6 xfailed`.

Classification:

- green direct/view/lifted 2CTA controls remain green;
- xfailed 2CTA indexed and descriptor-chain rows remain expected existing
  `FZ-20260421-0004` / `FZ-20260421-0006` coverage.

Logs:

- `/tmp/tmem_lane_al_round15_struct_ldred_g1.log`
- `/tmp/tmem_lane_al_round15_struct_ldred_g2.log`
- `/tmp/tmem_lane_al_round15_struct_ldred_g3.log`
- `/tmp/tmem_lane_al_round15_struct_ldred_g4.log`

## Isolated allocator/assertion sentinels

Collect-only command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'ldred and (allocator or optimizer or direct_index)'
```

Result: `2/33` selected.

Runtime commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred_twocta_rowcol_optimizer_crash'
```

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred_1cta_direct_index_allocator_crash'
```

Results:

- optimizer crash sentinel: `1 xfailed in 3.75s`;
- direct-index allocator sentinel: `1 xfailed in 3.75s`.

Classification:

- optimizer crash sentinel remains existing `FZ-20260421-0008`;
- direct-index allocator sentinel remains existing `FZ-20260421-0009`.

Logs:

- `/tmp/tmem_lane_al_round15_ldred_optimizer_crash.log`
- `/tmp/tmem_lane_al_round15_ldred_direct_index_allocator.log`

## Final classification

- New independent compiler crashes: `0`.
- New runtime wrong-results: `0`.
- New false unsupported/verifier-too-strict candidates: `0`.
- Existing failures reproduced: `FZ-20260421-0012` in the checked-in ld.red
  runtime matrix.
- Existing xfail sentinels confirmed: `FZ-20260421-0004`,
  `FZ-20260421-0006`, `FZ-20260421-0008`, `FZ-20260421-0009`.
- Green semantic additions: 16 temporary probe rows covering descriptor-chain
  full modifiers, integer abs software-reduce boundaries, M256 descriptor-chain
  identity, and M64 identity controls.

Recommended next discovery slice: extend the temporary semantic probe into a
generated descriptor-chain fuzzer that varies descriptor-chain depth, row-base
anchor, and `load_variant` independently for M128/M256, while keeping the known
M64 permuted row-base `FZ-0012` cases as controls rather than counting them as
new findings.
