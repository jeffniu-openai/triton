# TMEM Runtime Matrix Validation Recipe - 2026-04-13

This recipe is the current local way to run the full `python/test/gluon/test_tmem_runtime_matrix.py` surface without reducing the matrix or dropping coverage. It fixes the timeout-prone workflow by changing scheduling, cache reuse, and selectors only.

## Current Diagnosis

The runtime-matrix timeout is not behaving like a deadlock. The slow runs keep printing progress, exact slow nodeids pass when isolated, and immediate warm reruns are much faster. The bottleneck is cold compilation plus poor static partitioning of a few dense families.

Full collection with `PYTHONPATH` unset reports `6075` tests after the 2026-04-14 staged CP, `ld.red`, `ld/st`, MMAv5, and scaled-MMAv5 coverage expansions. The coverage-preserving bucket split is:

| Bucket | Selector | Cases | Scheduling |
| --- | --- | ---: | --- |
| `cp` | `-k cp` | 381 | split 4, one process per GPU |
| `mma` | `-k test_tmem_runtime_matrix_mma` | 1358 | split 4, one process per GPU |
| `splitn` / misc | exact function nodeids | 499 | split 4, one process per GPU |
| `ld_red` | `-k ld_red` | 968 | split 16, four waves, `pytest-xdist -n 4` inside each GPU shard |
| `ldst` | `-k ldst` | 2869 | split 16, least-duration split using the stored `ldst` durations, `pytest-xdist -n 4` inside each GPU shard |

The buckets sum to all `6075` collected tests. The `splitn` bucket must use exact nodeids; plain `-k splitn` also matches parameter IDs such as `32x32b_splitn` inside `ld_red` and `ld/st`, which pollutes the timing profile.

## Canonical Command

Always rebuild first:

```bash
make -j8
```

Then run the full matrix through the runner:

```bash
python3 .codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py
```

Useful variants:

```bash
# Show the exact commands without running them.
python3 .codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py --dry-run

# Run only the heavy families when validating the timeout fix.
python3 .codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py --categories ld_red ldst

# Rerun exact split groups after a shard failure while preserving canonical GPU/cache mapping.
python3 .codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py --categories ldst --groups 5 8

# Temporarily experiment with inner xdist for one bucket without changing defaults.
python3 .codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py --categories ldst --groups 5 8 --xdist-override ldst=8

# Keep caches separate for an experiment while still stable across waves.
python3 .codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py --cache-prefix /tmp/triton-cache-tmem-runtime-matrix-experiment

# Refresh duration data without concurrent shards clobbering one JSON file.
# The runner writes per-group duration files and merges them after each bucket passes.
python3 .codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py --categories ldst --store-durations
```

The runner removes inherited `PYTHONPATH`, sets a stable per-GPU `TRITON_CACHE_DIR`, and writes per-shard logs under `.codex/initiatives/tmem_linear_generalization/experiments/results/tmem_runtime_matrix_sweep_<timestamp>/`. It does not delete caches by default.

## Measured Profile

Small buckets are not the timeout source:

- `cp`: after the 2026-04-14 CP 128x128b, supported `warpx2` dtype expansions, broad two-CTA no-scales `128x256b` f32+i32 parity expansion, and `warpx2` dense-shared clean-negative dtype parity, `376 passed, 5 skipped` across four groups. The latest warm-cache CP bucket pass took `4.97s`, `7.20s`, `6.62s`, and `5.66s` pytest time; the prior cold/warm-mixed broad two-CTA validation took `72.22s`, `35.08s`, `81.12s`, and `79.59s`.
- `mma`: after the 2026-04-14 scaled-MMAv5 narrow tile-permuted clean-negative expansion, the tight MMA bucket collects `1358` cases. The latest focused `scaled_acc_tile_permuted_narrow` selector passed `20` cases across four groups (`5` per group) in `5.82s`, `5.90s`, `5.69s`, and `6.08s`; the prior focused `tile_permuted_narrow` selector passed `40` cases across four groups (`10` per group) in `5.76s`, `5.98s`, `5.90s`, and `5.76s`; the prior focused LHS use-acc selector passed `84` cases across four groups (`21` per group) in `44.69s`, `41.20s`, `59.75s`, and `67.78s`; the adjacent changed-callsite selector passed `198` cases across four groups (`50`, `50`, `50`, and `48`) in `100.77s`, `146.15s`, `59.66s`, and `9.47s`.
- true exact-nodeid `splitn` / misc bucket: after the 2026-04-14 M64 row/column split-N f32+i32 parity expansion, the bucket contains `499` cases. The latest focused M64 row/column split-N selector passed `452` cases across split-8 (`57`, `57`, `57`, `57`, `57`, `57`, `57`, and `53`) in `7.44s`, `14.24s`, `14.32s`, `13.94s`, `8.52s`, `14.91s`, `14.67s`, and `13.53s`; the prior identity M64 split-N selector passed `42` cases across four groups (`11`, `11`, `11`, and `9`) before this expansion.

Heavy buckets need finer scheduling:

- After the 2026-04-14 unsupported-layout shape sweep expansion, the focused selector `ld_red_additional_unsupported_layouts_report_clean_unsupported` passed `56` cases across four split groups (`14` each; shard times `4.42s`, `4.42s`, `6.79s`, and `8.01s`). The full `ld_red` bucket now collects `920` cases.
- After the 2026-04-14 non-f32 min/max contract expansion, the focused selector `ld_red_non_f32_contract_reports_clean_unsupported` passed `8` cases across four split groups (`2` each; shard times about `4s`). The full `ld_red` bucket now collects `880` cases.
- After the 2026-04-14 explicit N-sharded clean-negative modifier-matrix expansion, the focused selector `ld_red_explicit_n_sharded_layout_reports_clean_unsupported` passed `24` cases across four split groups (`6` each; shard times about `5s`). The full `ld_red` bucket now collects `880` cases.
- After the 2026-04-14 identity-256 clean-negative modifier-matrix expansion, the focused selector `ld_red_identity_256_linear_layout_reports_clean_unsupported` passed `32` cases across four split groups (`8` each; shard times about `5s`). The full `ld_red` bucket now collects `880` cases; refresh it with the runner after shared lowering changes or before a phase boundary.
- Serial split-4 `ld_red` was green but too slow and imbalanced: groups took about `13:37`, `22:21`, `25:15`, and `20:37`.
- Runner `ld_red` split-16 with `-n 4` passed the full bucket before the N=32 expansion: `643 passed` across 16 groups, with pytest shard times from `13.58s` to `95.91s`. After the 2026-04-14 N=32 non-identity expansion, the same runner passed `771` cases across 16 groups, with pytest shard times from `27.66s` to `110.05s`.
- `ld/st` is the largest bucket. Initial runner `ldst` split-16 with the stored duration cache, least-duration splitting, and `-n 4` passed the full bucket: `1201 passed, 441 skipped` across 16 groups, with pytest shard times from `271.68s` to `350.27s`.
- Follow-up speedup on 2026-04-13: the five lifted descriptor roundtrip matrices were proven to be skip-only on current Blackwell hardware (`440` cases that compiled until `OutOfResources` and then called `pytest.skip`). Marking those matrices as known pre-execution skips preserves instruction/op coverage because they produced no op coverage before. After this change, the exact skip-only functions skip `440` cases in `2.40s`, and the full `ldst` bucket still reports `1201 passed, 441 skipped` with shard times reduced to `194.71s` to `273.96s`.
- Duration-cache refresh on 2026-04-13: after those `440` cases became pre-execution skips, their stored `ldst_pytest_durations_20260413.json` entries were set to `0.001s` so least-duration splitting no longer treats them as cold OOR compiles. The full `ldst` runner still reports `1201 passed, 441 skipped`; group times on the validation run were `208.7s` to `263.8s`.
- Rejected kernel-reuse idea: runtime `variant_id` selector kernels work only with `@gluon.jit(do_not_specialize=["variant_id"])`, but they were worse under the real split-16/xdist workflow because each distinct layout compiles a larger multi-branch kernel. Do not replace the current per-variant ld/st tests with selector kernels without first changing grouping to batch all variants for the same layout and proving a full-bucket timing win.

- 2026-04-14 selected-shard rerun result: runner `--categories ldst --groups 5 8` preserved canonical GPU/cache mapping and passed group 5 in `11.3s` wall (`77 passed, 27 skipped in 9.57s`) and group 8 in `11.1s` wall (`74 passed, 28 skipped in 9.39s`). This is the preferred exact-rerun path after a shard failure/timeout and is not a reduced full-matrix replacement.
- `pytest-xdist -n 8` was tested on warm `ldst` shards and was slower than the retained `-n 4` default; keep default xdist unchanged unless a full-bucket timing proves a real win.

- After the 2026-04-14 higher-rank descriptor `ld/st` i32-parity expansion, the focused positive descriptor selector passed `200` cases across split-8 on four GPUs (`25` per group) in `384.32s`, `339.11s`, `402.75s`, `430.09s`, `418.65s`, `382.93s`, `131.40s`, and `278.68s`. The full `ldst` bucket now collects `2712` cases; refresh the full `ldst` runner after shared lowering changes or before a phase boundary.

- After the 2026-04-14 subword descriptor-chain expansion, the focused selector passed `60` cases across split-4 on four GPUs (`15` per group) in `151.24s`, `153.02s`, `150.40s`, and `152.90s`. The full `ldst` bucket then collected `2772` cases; refresh the full `ldst` runner after shared lowering changes or before a phase boundary.
- After the 2026-04-14 x1 i32 expansion, the focused selector passed `21` cases across split-4 on four GPUs (`6`, `6`, `6`, and `3` selected) in `7.75s`, `41.66s`, `4.65s`, and `4.64s`. The full `ldst` bucket then collected `2793` cases; this is a test-only dtype-parity expansion over the existing one-column 32-bit lowering.
- After the 2026-04-14 scales explicit N-sharded variant expansion, the focused `ldst_scales_variant` selector passed `91` cases across split-4 on four GPUs (`23`, `23`, `23`, and `22` selected) in `5.25s`, `5.62s`, `5.23s`, and `5.49s`. The full `ldst` bucket then collected `2829` cases; this is test-only variant coverage over already-supported scales `ld/st` lowering plus clean below-threshold unsupported diagnostics.
- After the 2026-04-14 rank-5 descriptor positive and dtype-parity expansions, the focused `rank5_small` selector passed `40` cases across split-4 on four GPUs (`10` per group) in `127.35s`, `92.63s`, `127.20s`, and `92.18s`. The full `ldst` bucket now collects `2869` cases; this is test-only positive descriptor-view coverage over a resource-safe rank-5 allocation for `f32` and `i32`, while the old `[2,2,2]` rank-5 matrices remain pre-execution OOR skips.
- After the 2026-04-14 `ld.red` descriptor-view and explicit-variant expansions, the focused `ld_red_descriptor_chain` selector passed `48` cases across split-4 on four GPUs (`12` per group) in `44.84s`, `48.10s`, `58.35s`, and `190.20s`. The full `ld_red` bucket now collects `968` cases; this is f32 hardware-reduction coverage through `slice`/`index`/reshape descriptor views over identity, tile-permuted, and row/column-permuted source layouts, with identity also covering every compatible explicit reduction register-layout request.

Aggregating the current per-bucket evidence gives full matrix coverage: `5629 passed, 446 skipped` across all `6075` collected cases. This is bucketed evidence from focused/bucket reruns, not a reduced matrix claim; refresh the full runner after shared lowering or major scheduling changes.

Representative compile evidence:

- `test_tmem_runtime_matrix_ldst_descriptor_compositions_rowcol_permuted_layout_sweep[reverse-reverse-256-16x64b-16x64b.x64.b32]` took about `31s` cold and about `3s` warm.
- `test_tmem_runtime_matrix_ld_red_col_permuted_linear_layout[even_odd-256-32x32b.x64-False-propagate_nan1-min]` took about `10s` cold and about `3s` warm.

## Rules For Future Sweeps

- Do not reduce the matrix to make timeout symptoms disappear. Change scheduling first.
- Do not overwrite or recreate `TRITON_CACHE_DIR` inside test helpers or per test. Stable per-GPU cache reuse is required for useful local velocity.
- Use `pytest-split` for outer GPU sharding and only use `pytest-xdist` inside the compile-heavy single-GPU buckets (`ld_red`, `ld/st`) unless a focused experiment shows it is safe elsewhere.
- Treat xdist OOMs as potentially false negatives. Rerun exact failing nodeids on an isolated GPU before classifying them as product failures.
- If a future runner shard times out while still printing progress, inspect the shard log and rerun the slow exact nodeids with the same cache before raising the timeout.
- Use runner `--store-durations` after material test-status or timing changes instead of running pytest-split `--store-durations` directly across concurrent shards; the raw plugin writes one shared JSON path and can race.
