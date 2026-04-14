# TMEM Runtime Matrix Validation Recipe - 2026-04-13

This recipe is the current local way to run the full `python/test/gluon/test_tmem_runtime_matrix.py` surface without reducing the matrix or dropping coverage. It fixes the timeout-prone workflow by changing scheduling, cache reuse, and selectors only.

## Current Diagnosis

The runtime-matrix timeout is not behaving like a deadlock. The slow runs keep printing progress, exact slow nodeids pass when isolated, and immediate warm reruns are much faster. The bottleneck is cold compilation plus poor static partitioning of a few dense families.

Full collection with `PYTHONPATH` unset reports `4977` tests after the 2026-04-14 staged CP, `ld.red`, `ld/st`, MMAv5, and scaled-MMAv5 coverage expansions. The coverage-preserving bucket split is:

| Bucket | Selector | Cases | Scheduling |
| --- | --- | ---: | --- |
| `cp` | `-k cp` | 381 | split 4, one process per GPU |
| `mma` | `-k test_tmem_runtime_matrix_mma` | 852 | split 4, one process per GPU |
| `splitn` / misc | exact function nodeids | 252 | split 4, one process per GPU |
| `ld_red` | `-k ld_red` | 880 | split 16, four waves, `pytest-xdist -n 4` inside each GPU shard |
| `ldst` | `-k ldst` | 2612 | split 16, least-duration split using the stored `ldst` durations, `pytest-xdist -n 4` inside each GPU shard |

The buckets sum to all `4977` collected tests. The `splitn` bucket must use exact nodeids; plain `-k splitn` also matches parameter IDs such as `32x32b_splitn` inside `ld_red` and `ld/st`, which pollutes the timing profile.

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
- `mma`: after the 2026-04-14 scaled-MMAv5 mixed-fp4A clean-negative shape-parity expansion, the tight MMA bucket collects `852` cases. The latest focused mixed-fp4A clean-negative selector passed `18` cases across four groups (`5`, `5`, `5`, and `3`) in `5.21s`, `5.20s`, `4.95s`, and `4.71s`; the immediately prior focused repeated-N32 clean-negative selector passed `20` cases across four groups (`5` per group) in `4.98s`, `4.97s`, `5.13s`, and `5.46s`.
- true exact-nodeid `splitn` / misc bucket: runner smoke passed `252` tests across four groups (`63` per group) in `8.95s`, `16.06s`, `15.93s`, and `14.22s` pytest time.

Heavy buckets need finer scheduling:

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

Aggregating the current per-bucket evidence gives full matrix coverage: `4531 passed, 446 skipped` across all `4977` collected cases. This is bucketed evidence from focused/bucket reruns, not a reduced matrix claim; refresh the full runner after shared lowering or major scheduling changes.

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
