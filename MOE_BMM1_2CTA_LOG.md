# MoE BMM1 2CTA Optimization Log

## Current Objective

Continuously optimize `python/examples/gluon/05-moe-bmm1-fused-gather.py` so the
2CTA path beats 1CTA for every batch size, can be enabled for every batch size,
and keeps improving, with special focus on batch sizes `<= 2048`.

## Durable Rules

- Follow `large-project-methodology`: update this log with hypotheses,
  experiments, validation, and remaining work.
- Use subagents liberally for code analysis, related-codebase research,
  profiling analysis, review, autotuning ideas, and hypothesis testing when
  the user has authorized parallel work.
- The machine has 4 GB300 GPUs. Use them to parallelize independent sweeps and
  compile warmups, but reserve quiet GPUs for final comparisons.
- Prefer paired measurements on the same prepared inputs. Confirm apparent
  small regressions with longer repetitions before changing selector policy.

## 2026-04-18 State

- Branch: `codex/2cta-bmm1`.
- Current selector enables 2CTA for `slice=16`, `36..72`, and `>=80`.
- Current unresolved low-batch gap: `slice=20..32`, corresponding to batch
  sizes `640`, `768`, `896`, and `1024` for GPT-OSS-120B.
- Existing useful artifacts:
  - `/tmp/moe_compare_after_merge_prod.csv`
  - `/tmp/moe_compare_after_merge_uniform.csv`
  - `/tmp/moe_prod_vs_last_saved.csv`
  - `/tmp/moe_bmm1_bench_final.txt`
- All four GPUs were idle when checked with `nvidia-smi` on 2026-04-18.

## Active Slice

Find a stable 2CTA configuration for `slice=20..32` without regressing existing
2CTA wins. First experiments should compare forced 1CTA baseline against forced
2CTA candidates on shared prepared cases for batch sizes `640`, `768`, `896`,
and `1024`, under both simulated production routing and uniform routing.

## 2026-04-18 Experiments

- Built `/tmp/moe_bmm1_pair_tune.py`, a paired logical-case benchmark harness
  that materializes p-specific weight layouts from shared routing, input,
  logical MXFP weights, scales, and bias.
- Initial `32x256` 2CTA variants for `slice=20..32` mostly lost to the
  current `32x128` 1CTA baseline. `W_NUM_BUFS=6` was consistently bad.
- `SWIGLU_SUBTILE_FACTOR=2` improves the `32x256` 2CTA family and brought
  simulated-prod batch `640` close to parity, but uniform routing remained
  below 1CTA for `640`, `768`, and `896`.
- `BLOCK_M=16, BLOCK_N=256, NUM_CTAS=2` compiled and validated, but was much
  slower on uniform batch `640`.
- `BLOCK_N=384` is not legal for this path because descriptor layout creation
  requires a power-of-two shape element. `BLOCK_N=512` compiled and validated,
  but was much slower on uniform batch `640`.
- Increasing launch `OCCUPANCY` to `3` or `4` was consistently worse.
- Lower/mid register variants around `m32/sub2` moved results slightly but did
  not solve the uniform-routing gap.
- Loader prologue-fill skip was implemented in the source: activation and
  weight loaders now skip empty-barrier waits during initial ring fill. Focused
  correctness passed for batch sizes `640`, `1024`, `2048`, and `2560`.
- Attaching the activation empty barrier directly to `tcgen05_mma_scaled`
  validated but was a net performance loss on prod and did not fix uniform, so
  that change was reverted.

## Current Best Low-Gap Candidates

- For prod-like routing, the best config family seen is
  `BLOCK_M=32, BLOCK_N=256, NUM_CTAS=2, X_NUM_BUFS=5, W_NUM_BUFS=5,
  SWIGLU_SUBTILE_FACTOR=2`, with `BAND_N=22` or `26` depending on noise and
  batch size.
- For uniform routing, no tested 2CTA candidate beats 1CTA for `640`, `768`,
  or `896`; `1024` can be slightly positive with the `64x256` family.
- Next diagnostic target: run `ncu` on uniform batch `640`, comparing forced
  `1CTA 32x128` against the best `2CTA 32x256/sub2` candidate.

## 2026-04-18 Follow-up Experiments

- NCU on uniform batch `640` showed `2CTA 32x256/sub2` has lower shared-memory
  conflict severity than `1CTA 32x128`, but executes more instructions and runs
  for more active cycles. The gap is therefore not explained by global-memory
  bandwidth alone; fixed instruction/control/shared overhead is material.
- Explicit `mbarrier.wait(..., deps=[...])` on x/w/store consumer waits compiled
  and passed focused correctness, but did not improve the low-batch gap, so it
  was reverted.
- Added an optional precomputed full-tile schedule path. The first version used
  expected `grid_m` and faulted on sparse prod-like routing; the fixed version
  uses the actual `grid_m` from `x_block_offs[NUM_SLICES]` and avoids host
  `.item()` during cudagraph capture by doing it only on schedule-cache misses.
- Full-tile schedule results:
  - Prod-like routing improved for several candidates, for example best
    `batch=640` reached `1.002x` with `m32/sub2/regs48/fullsched`, and
    `batch=896` reached `1.001x` with `m32/sub2/b26/fullsched`.
  - Uniform routing did not close the critical gap: best observed was about
    `0.961x` at `640`, `0.965x` at `768`, `0.916x` at `896`, and `1.023x` at
    `1024`.
- Buffer-count, accumulator-buffer, epilogue-depth, and band-width micro-sweep:
  - `x6/w5` and `BAND_N=32` can improve prod-like routing, with prod `896`
    reaching `1.032x` for `m32/sub2/x6w5/b22`.
  - Uniform `640/768/896` still topped out around `0.969x/0.964x/0.897x`.
  - Lowering to 4 total warps regressed; 6 total warps is illegal because
    `num_warps` must be a power of two.
- `64x256` and `64x512` sweeps for uniform `896/1024` did not fix `896`.
  `64x256` remains useful only around `1024`; `64x512` and `subtile=2` were
  substantially slower.

## Current Interpretation

- The optional full-schedule path is correct and useful as a tuning axis but is
  not sufficient for uniform low-batch 2CTA.
- Uniform `896` is the hardest case. The best `2CTA` shapes still spend too
  much fixed overhead per expert tile relative to `1CTA`, even when scheduling,
  buffer counts, and launch occupancy are adjusted.
- Next experiments should target source-level overhead reduction in the
  epilogue/store path or reduce duplicated per-partition work, not just broader
  parameter sweeps.

## 2026-04-18 Direct-Store Follow-up

- Implemented optional direct epilogue store path and fixed an accumulator
  lifetime race by releasing `acc_empty_bar` only after loading accumulator
  registers. Direct-store variants now validate in focused sweeps.
- Removed direct-path dummy store buffer/barrier allocation by reusing existing
  descriptors for unused `PartitionArgs` fields. This is the clearest source
  win in the low-batch direct path.
- Best high-repetition uniform run after direct cleanup:
  - `640`: best direct `0.992x`, `415.7 TFLOP/s`, `5.55 TB/s`.
  - `768`: best direct `0.992x`, `489.0 TFLOP/s`, `5.54 TB/s`.
  - `896`: best direct `0.948x`, `524.9 TFLOP/s`, `5.22 TB/s`.
  - `1024`: `m64/b32/fullsched` `1.003x`, `583.2 TFLOP/s`, `5.08 TB/s`.
- Simulated-prod routing is much better for direct 2CTA:
  - `640`: `1.042x`, `103.9 TFLOP/s`, `6.04 TB/s`.
  - `768`: still slightly below parity at `0.989x`.
  - `896`: `1.050x`, `105.3 TFLOP/s`, `5.23 TB/s`.
  - `1024`: `1.029x`, `133.5 TFLOP/s`, `5.19 TB/s`.
  - `1280..2048`: direct 2CTA wins by roughly `1.11x` to `1.39x` in the sampled seed.
- NCU on uniform `896` comparing `1CTA 32x128` to best direct `2CTA 32x256`:
  - 1CTA duration `38.62 us`, memory throughput `3.89 TB/s`, executed
    instructions `9.57M`, achieved occupancy `48.0%`, theoretical occupancy `50%`.
  - 2CTA duration `40.67 us`, memory throughput `3.75 TB/s`, executed
    instructions `7.48M`, achieved occupancy `36.3%`, theoretical occupancy `37.5%`.
  - Interpretation: 2CTA executes fewer instructions but loses latency hiding
    because shared memory limits occupancy. Reducing staging below x5/w5 loses
    more pipeline depth than the occupancy relief is worth.
- Dead ends after the direct cleanup:
  - Program-striped full-tile schedule regressed helper-store paths and
    deadlocked direct+fullschedule, so it was removed.
  - `BLOCK_N=128` 2CTA does not compile in this kernel; `warp_specialize`
    rejects the shape.
  - `BLOCK_M=24` fails layout construction because a shape element must be a
    power of two.
  - `BLOCK_M=16`, `NUM_WARPS=4`, `OCCUPANCY=1/3/4`, `W_NUM_BUFS=6`,
    `X/W` staging below x5/w5, helper `subtile=4`, and `ACC_NUM_BUFS=2`
    all regressed uniform low-batch performance.
  - Skipping the first accumulator empty-barrier wait was correctness-clean but
    performance-inconsistent due to extra per-iteration predicate overhead, so
    it was reverted.

## Next Hypotheses

- The remaining uniform `896` gap is probably structural: `2CTA 32x256` pays
  cluster/shared-memory occupancy cost that `1CTA 32x128` avoids. Search for a
  way to reduce per-cluster shared memory without dropping below x5/w5 staging,
  or find a legal smaller-N 2CTA tensor-memory layout.
- Investigate whether a source-local direct epilogue layout can use fewer
  shared-memory operations or fewer participating epilogue warps without
  changing `NUM_WARPS`.
- If selector work proceeds before uniform is solved, use routing-aware
  evidence carefully: prod-like routing strongly favors direct 2CTA, while
  uniform `640/768/896` still does not.
