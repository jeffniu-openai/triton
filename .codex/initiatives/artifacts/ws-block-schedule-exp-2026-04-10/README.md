# WS Block Scheduling Follow-Up Experiments

Updated: 2026-04-10T02:32:00Z

## Scope

This directory contains a no-repo-edit follow-up to the earlier block-scheduling pass. The live repo code under `python/` was left untouched. Instead, the current WS example was copied here and modified in place for schedule-only experiments:

- `moe_bmm1_fused_gather_exp.py`
- `sweep_block_schedule_exp.py`

The benchmark target stayed fixed:

- case: `bs16384_E256_es8_B5120x10240`
- `local_rank=0`
- exact validation enabled

## Experiment Families

### Wider XCD-style swizzles

Added:

- `row_major_swizzle_8`
- `row_major_swizzle_19`
- `row_major_swizzle_38`

Result: all were decisively worse than the current row-major traversal.

GPU 2 screen (`rep=200`):

- `row_major = 0.3423 ms`
- `row_major_swizzle_8 = 0.4273 ms`
- `row_major_swizzle_19 = 0.4386 ms`
- `row_major_swizzle_38 = 0.4557 ms`

Conclusion: larger swizzle factors over-scatter the traversal badly on this bucket.

### Wave-aware N-banded grouped schedules

Added:

- `band_n_8_grouped_m_19`
- `band_n_4_grouped_m_38`
- `band_n_2_grouped_m_76`

These were designed so a contiguous chunk of work lines up with the persistent launch size (`152`) and keeps the same narrow N band live across many CTA iterations.

GPU 1 screen (`rep=200`):

- `row_major = 0.3452 ms`
- `band_n_8_grouped_m_19 = 0.3483 ms`
- `band_n_4_grouped_m_38 = 0.3542 ms`
- `band_n_2_grouped_m_76 = 0.3552 ms`

Conclusion: exact wave alignment alone does not help here. The grouped inner traversal is too costly.

### Row-major N-banded schedules

Added:

- `band_n_2_row_major`
- `band_n_4_row_major`
- `band_n_5_row_major`
- `band_n_8_row_major`
- `band_n_10_row_major`
- `band_n_20_row_major`

This keeps the inner traversal row-major, but narrows the live N footprint.

GPU 0 follow-up screen (`rep=200`):

- `row_major = 0.33934377 ms`
- `band_n_5_row_major = 0.34458243 ms`
- `band_n_8_row_major = 0.34047384 ms`
- `band_n_10_row_major = 0.33920375 ms`
- `band_n_20_row_major = 0.33803356 ms`
- `band_n_2_row_major = 0.34300021 ms`

Conclusion: the promising signal is not "smaller N is always better"; it is specifically that a half-width N band (`20` columns) may improve locality while preserving the row-major inner order.

### 20-column band local follow-up

Added:

- `band_n_20_grouped_m_4`
- `band_n_20_grouped_m_8`
- `band_n_20_grouped_m_16`

GPU 0 screen (`rep=200`):

- `row_major = 0.3391 ms`
- `band_n_20_row_major = 0.3373 ms`
- `band_n_20_grouped_m_4 = 0.3395 ms`
- `band_n_20_grouped_m_8 = 0.3404 ms`
- `band_n_20_grouped_m_16 = 0.3436 ms`

Conclusion: once the schedule is restricted to a 20-column N band, the best inner order is still plain row-major.

## Long Same-GPU Reruns

Final direct compare on GPU 0 (`warmup=30`, `rep=1000`):

- `row_major = 0.34115855 ms`
- `band_n_20_row_major = 0.34022407 ms`
- `band_n_20_grouped_m_4 = 0.34222745 ms`

Head-to-head rerun on GPU 0 (`warmup=30`, `rep=2000`):

- `row_major = 0.34173582 ms`
- `band_n_20_row_major = 0.34050464 ms`

Observed edge for `band_n_20_row_major`:

- about `0.29%` on the `rep=1000` direct compare
- about `0.36%` on the `rep=2000` head-to-head rerun

So the only surviving nontrivial candidate from this follow-up is:

- `band_n_20_row_major`

## NCU Note

A quick `rep=1` NCU comparison against `row_major` did **not** reinforce the timing edge:

- profiled `row_major` duration: `~479.0 us`
- profiled `band_n_20_row_major` duration: `~480.8 us`
- profiled L2 throughput also came back lower for the banded variant

Given that the timing edge is only about `0.3%`, the current interpretation is:

- the schedule difference is real enough to show up in repeated cudagraph timing
- but it is small enough that single-shot NCU profiling is not yet a trustworthy tiebreaker

If this candidate is promoted later, it should be re-profiled with a more careful NCU setup.

The averaged metrics extracted from those two NCU runs are stored in:

- `ncu_summary.csv`

The raw `.ncu-rep` files were kept only transiently during this experiment and moved out of the repo workspace afterward so this artifact directory stays lightweight.

## Recommended Next Step

If block scheduling is revisited again, the most justified follow-up is:

1. upstream `band_n_20_row_major` into the real schedule switch
2. compare it directly against the current repo `row_major` path in the normal benchmark harness
3. only if that still wins, repeat NCU with a tighter apples-to-apples profile setup

No other schedule family in this follow-up produced a comparable signal.
