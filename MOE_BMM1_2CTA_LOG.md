# MoE BMM1 2CTA Optimization Log

## Current Objective

Continuously optimize `python/examples/gluon/05-moe-bmm1-fused-gather.py` so the
2CTA path beats 1CTA for every batch size, can be enabled for every batch size,
and keeps improving, with special focus on batch sizes `<= 2048`.

## Durable Rules

- Follow `large-project-methodology`: update this log with hypotheses,
  experiments, validation, and remaining work.
- Never voluntarily stop or end an optimization turn while there is an
  untested hypothesis, unmeasured regression, plausible tuning path, running
  validation/profiler, or uncommitted durable state. Keep moving through
  experiments, measurements, source edits, validation, notes, commits, and
  pushes. If an external interruption forces a stop, update this log with the
  branch, artifacts, validation, current hypothesis, and exact next unblocked
  command first.
- Use subagents liberally for code analysis, related-codebase research,
  profiling analysis, review, autotuning ideas, and hypothesis testing when
  the user has authorized parallel work.
- The machine has 4 GB300 GPUs. Use them to parallelize independent sweeps and
  compile warmups, but reserve quiet GPUs for final comparisons.
- Prefer paired measurements on the same prepared inputs. Confirm apparent
  small regressions with longer repetitions before changing selector policy.

## 2026-04-18 State

- Branch: `codex/2cta-bmm1`.
- Current selector enables 2CTA for `slice=16`, `20`, `24`, `32`,
  `36..72`, and `>=80`.
- Current unresolved low-batch selector gap: `slice=28`, corresponding to
  batch size `896` for GPT-OSS-120B. This slice remains on 1CTA because
  uniform routing still shows a stable 2CTA regression.
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

## 2026-04-18 Current Pass

- Current uncommitted source state:
  - Direct epilogue path prefetches bias before `acc_ready_bar` wait.
  - Direct epilogue path hoists `shape_m`, `off_m`, and `out_off_n` before
    `apply_bias_and_scale`, overlapping the `shape_m` load with accumulator
    readiness.
  - Direct accumulator handoff ordering is `wait -> advance -> acc load ->
    arrive`. The alternate `wait -> acc load -> arrive -> advance` was
    rechecked and not kept because the earlier ordering is better on the
    sensitive low-batch runs.
- Validation:
  - `make` completed with no rebuild work.
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/code/triton/python:/root/code/triton/python/triton_kernels pytest -s --tb=short python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op`
    passed: `48 passed in 48.90s`.
- Current focused benchmarks:
  - `/tmp/moe_bmm1_direct_advance_before_load_uniform640_896_rep1000.csv`
    confirms uniform `640` can be positive with direct 2CTA, best
    `m32_bn256_sub1_direct_x6w5_act2w1m1_regs52_epin1_b32` at `1.010x`,
    `423.3 TFLOP/s`, `5.65 TB/s`.
  - Same run confirms uniform `896` remains negative, best
    `m32_bn256_sub1_direct_x6w5_nomcscale_act2w1m1_epin1_b32` at `0.935x`,
    `525.6 TFLOP/s`, `5.22 TB/s`.
  - `/tmp/moe_bmm1_direct_advance_before_load_prod_rep500.csv` shows
    prod-like routing remains positive for `640/768/896/1024`, with best
    speedups `1.053x`, `1.047x`, `1.044x`, and `1.018x` respectively.
- Current NCU on uniform `896`:
  - Reports:
    `/tmp/ncu_moe_896_uniform_1cta_advbefore.ncu-rep` and
    `/tmp/ncu_moe_896_uniform_2cta_advbefore_x6nomcscale.ncu-rep`.
  - 1CTA: duration `37.76 us`, memory throughput `3.97 TB/s`, executed
    instructions `9.57M`, theoretical occupancy `50%`, achieved occupancy
    `47.83%`, eligible warps/scheduler `0.78`.
  - 2CTA: duration `40.67 us`, memory throughput `3.77 TB/s`, executed
    instructions `7.46M`, theoretical occupancy `37.5%`, achieved occupancy
    `36.14%`, eligible warps/scheduler `0.51`.
  - Interpretation is unchanged: the best 2CTA path does less work but loses
    on latency hiding because cluster/shared-memory occupancy is too low.
- Additional dead ends in this pass:
  - Program-striped full-tile schedule was reimplemented as a host-only knob
    and tested with direct+reuse; it regressed uniform `896` to about
    `0.88x`, so the source knob was removed.
  - Planar snake variants in the temporary harness regressed uniform `896`.
  - `BLOCK_N=512` direct 2CTA compiled but was much slower; `BLOCK_N=384`
    fails descriptor layout creation because a shape element must be a power
    of two.
  - `BLOCK_M=28` would match uniform `896` slice size, but layout construction
    rejects it because the M shape element must be a power of two.
  - `BLOCK_K=64` is invalid for this MX scale layout:
    `block_shape[0]=2 must be divisible by 4`.
  - Activation warp counts `1`, `3`, and `4` were worse or failed to compile;
    the current `LOAD_ACTIVATION_WARPS=2` direct family remains best.
  - Direct epilogue N-splitting (`FORCE_EPILOGUE_WARPS_N1=False`) regressed
    `640/768/896/1024`; keep `FORCE_EPILOGUE_WARPS_N1=True`.
  - Rechecked `ACC_NUM_BUFS=2` under the current source; it still does not fix
    uniform `896`.
  - Missing no-multicast `BAND_N` variants were filled in the temporary
    harness. They help select per-batch candidates for `640/768/1024`, but
    not `896`.

## 2026-04-18 Selector and Current Frontier

- Source changes added in this pass:
  - Weight loader now computes MX scale K offsets as `ki * scale_k_stride`
    instead of `ki * BLOCK_K // (...)`, avoiding a hot-loop constexpr division
    pattern in TTGIR.
  - Store paths now pass packed output-column offsets into `store_packed_out`,
    avoiding repeated output-N divisions in the direct and helper epilogues.
  - Selector now uses direct 2CTA `32x256` four-warp configs for slice sizes
    `16`, `20`, `24`, and `32`; slice `28` intentionally remains 1CTA because
    uniform batch `896` still regresses under every validated 2CTA candidate.
  - Low-slice direct selector configs keep `BAND_N=32`; an intermediate
    selector run showed that overwriting this to `22` regressed the tuned
    direct path.
- Validation:
  - `make` completed with no rebuild work.
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/code/triton/python:/root/code/triton/python/triton_kernels pytest -s --tb=short python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op`
    passed after the final selector adjustment: `48 passed in 4.75s`.
- Stable low-batch selector evidence:
  - `/tmp/moe_bmm1_slice16_direct_uniform512_rep1500.csv`: direct four-warp
    `x5w5` 2CTA reached `1.022x`, `328.7 TFLOP/s`, `5.66 TB/s`; the old
    non-direct selected 2CTA path regressed to `0.965x`.
  - `/tmp/moe_bmm1_selector_low_uniform_rep3000.csv`: best uniform `640`,
    `768`, and `1024` direct candidates reached about `1.011x`, `1.011x`,
    and `1.029x` respectively.
  - `/tmp/moe_bmm1_selected_vs_named_640_768_rep3000.csv` confirmed the
    selected slice-24 config is a small win (`1.006x`) and showed slice-20 is
    noisy between x5 and x6; selector currently uses x6 based on the broader
    set of recent rechecks.

## 2026-04-18 Slice-16 Multicast Selector Probe

- Rebuilt with `make` before running benchmarks; ninja reported no work.
- Focused same-prepared repeated checks for `batch=512`, `slice=16` showed
  that `x5/w5` direct 2CTA remains the best uniform family, but allocator and
  prepared-weight address effects are large enough to change apparent winners
  between identical configs:
  - Uniform `512`: selected x5 direct stayed near `1.018x`; the x5 no-multicast
    variant was effectively tied, while x6 staging was lower at about
    `1.003x..1.009x`.
  - Prod-like `512`: x5 no-multicast was consistently better than the current
    selected all-multicast variant, reaching about `1.073x..1.078x` in the
    same-prepared repeated run.
- A source selector edit that changed slice `16` to no-multicast was tested and
  then reverted: when the same config was exercised through the `selected`
  candidate after rematerializing prepared tensors, prod-like `512` regressed
  to `0.989x` in `/tmp/moe_bmm1_slice16_selected_after_nomc_prod_rep4000.csv`.
  The side-by-side run
  `/tmp/moe_bmm1_slice16_prod_nomc_after_side_by_side_rep5000.csv` showed the
  identical selected/no-multicast configs ranging from `0.986x` to `1.035x`
  depending on allocation order, so this is not safe selector evidence.
- CSV evidence:
  - `/tmp/moe_bmm1_slice16_uniform_nomc_focus_rep4000.csv`
  - `/tmp/moe_bmm1_slice16_prod_nomc_focus_rep4000.csv`
  - `/tmp/moe_bmm1_slice16_selected_after_nomc_uniform_rep4000.csv`
  - `/tmp/moe_bmm1_slice16_selected_after_nomc_prod_rep4000.csv`
  - `/tmp/moe_bmm1_slice16_prod_nomc_after_side_by_side_rep5000.csv`
  - `/tmp/moe_bmm1_selector_recheck_uniform_512_1024_rep1800.csv`
  - `/tmp/moe_bmm1_selector_recheck_prod_512_1024_rep1200.csv`
- Decision: keep the existing slice-16 selector for now. Future selector
  changes need a harness that controls prepared allocation order or reuses the
  same prepared tensors when comparing identical configs.
- Current uniform `896` status:
  - Best high-repetition candidates remain below parity. Recent `rep=3000`
    checks:
    `/tmp/moe_bmm1_warps4_x6_regs60_uniform896_rep3000.csv` and
    `/tmp/moe_bmm1_warps4_x6_nomc_uniform896_rep3000.csv` both topped out
    around `0.936x`.
  - `BAND_N=24` did not survive focused recheck; because `GRID_N=23`, it
    shares the same fast scheduling path as `BAND_N=32` and the apparent win
    was noise.
- Current NCU on uniform `896`:
  - Reports:
    `/tmp/ncu_moe_896_uniform_1cta_current_ws.ncu-rep` and
    `/tmp/ncu_moe_896_uniform_2cta_warps4_x6_current_ws.ncu-rep`.
  - 1CTA: duration `37.95 us`, memory throughput `3.95 TB/s`, executed
    instructions `9.24M`, theoretical occupancy `50%`, achieved occupancy
    `48.23%`, eligible warps/scheduler `0.73`.
  - 2CTA four-warp direct: duration `41.22 us`, memory throughput
    `3.70 TB/s`, executed instructions `6.66M`, theoretical occupancy `25%`,
    achieved occupancy `24.14%`, eligible warps/scheduler `0.37`.
  - Interpretation: four-warp 2CTA reduces instructions and shared memory
    versus the earlier direct path, but the resident/eligible warp count is
    even worse. The remaining `896` gap is still latency hiding, not
    instruction count.
- Additional dead ends in this pass:
  - Direct-store N-CGA layout failed `warp_specialize` lowering.
  - `M16/BN512` direct 2CTA was `0.47x-0.60x` on uniform `896/1024`.
  - `M64/BN256` direct/helper/wide variants were severe regressions on
    uniform `896/1024`.
  - `X_NUM_BUFS=2/3` with `W_NUM_BUFS=5` was much slower; reducing X staging
    does not cross a useful occupancy threshold.
  - `OCCUPANCY=3` remains much slower for the direct low-batch family.

## 2026-04-18 Continuous Slice-28 Probe

- Rebuilt with `make` first; ninja reported no work.
- Strengthened the never-stop optimization invariant in `AGENTS.md` and this
  log, committed as `25cb47e4d`, and pushed to
  `jeffniu-openai/codex/2cta-bmm1`.
- Focused uniform `batch=896` / `slice=28` measurements continued on quiet
  GB300 GPUs:
  - `BLOCK_M=28` direct 2CTA variants are illegal because descriptor/layout
    shape elements must be powers of two.
  - Four-, eight-, and sixteen-warp direct families were rechecked. Sixteen
    warps are severe regressions (`~0.63x`); the best legal result was still
    four-warp `M32/BN256/sub1/direct/x5w5` at about `0.936x`.
  - Inline MMA input release regressed the best four-warp family (`~0.929x`)
    and did not improve the eight-warp inline candidates.
  - Extra accumulator buffering (`ACC_NUM_BUFS=2/3`) regressed both four- and
    eight-warp direct paths.
  - Ring-depth threshold checks showed `x4` and `w4` are not usable for this
    slice: `x4/w5` was only `~0.915x`, and any `w4` variant was about
    `0.86x`. `x7/w5` also regressed.
  - `OCCUPANCY=1` and `OCCUPANCY=3` are large regressions for both four- and
    eight-warp direct variants; keep `OCCUPANCY=2`.
  - Gather-index reuse produced one noisy `~0.937x` result for eight-warp
    `x6/w5`, but a focused sweep across loader/MMA warp ownership, multicast,
    band, ring, and register settings did not reproduce a stable improvement.
- New CSV artifacts:
  - `/tmp/moe_bmm1_m28_uniform896_rep1200.csv`
  - `/tmp/moe_bmm1_warp_family_uniform896_rep1800.csv`
  - `/tmp/moe_bmm1_inline_release_uniform896_rep1800.csv`
  - `/tmp/moe_bmm1_accbuf_uniform896_rep1800.csv`
  - `/tmp/moe_bmm1_ring_depth_uniform896_rep1500.csv`
  - `/tmp/moe_bmm1_occupancy_uniform896_rep1200.csv`
  - `/tmp/moe_bmm1_reuse_uniform896_rep1500.csv`
  - `/tmp/moe_bmm1_reuse_x6_sweep_uniform896_rep1400.csv`
- New NCU report:
  - `/tmp/ncu_moe_896_uniform_2cta_warps4_x5_current_ws.ncu-rep`
  - Four-warp `x5/w5` direct: duration `~40.5-40.7 us`, memory throughput
    `~3.75-3.77 TB/s`, executed instructions `~6.65M`, theoretical occupancy
    `25%`, achieved occupancy `~24.3-25.2%`, eligible warps/scheduler
    `0.37-0.38`.
  - Launch/occupancy detail: dynamic shared memory is `~96.99 KB/block`, with
    shared memory limiting the kernel to two blocks per SM. Register, barrier,
    and warp limits are looser than shared memory for this shape.
  - Dominant long-scoreboard source waits remain activation empty-barrier
    wait around line 556, weight empty-barrier wait around line 594, and
    accumulator-ready wait around line 898.
- Additional quick falsifications after the main probe:
  - `BLOCK_K=64` direct candidates fail current scale descriptor layout
    assertions (`block_shape[0]=2 must be divisible by 4`); the `BK256`
    candidate also fails assertions. Artifact:
    `/tmp/moe_bmm1_bk64_uniform896_rep1200.csv`.
  - `M16/BN256` compile/autotest path was interrupted after several minutes
    with only the 1CTA and current `M32` baseline emitted. Treat this path as
    impractical until a narrower M16 candidate is justified. Partial artifact:
    `/tmp/moe_bmm1_m16bn256_uniform896_rep1200.csv`.
- Current interpretation: `slice=28` is not blocked by instruction count or a
  simple launch/warp/ring selector. The next source-level path should target a
  structural reduction in 2CTA latency exposure, likely by changing producer
  and consumer overlap around activation/weight ring waits or by reducing
  direct epilogue accumulator wait time without increasing tensor-memory or
  shared-memory residency.

## 2026-04-18 Selector Measurement Recheck

- Raw focused selector sweeps for `slice=16` and `slice=20` reproduced the
  earlier warning: `selected` and an explicit identical candidate could measure
  several percent apart when materialized and benchmarked separately. Treat
  artifacts
  `/tmp/moe_bmm1_slice16_20_selector_uniform_rep2500_s012.csv` and
  `/tmp/moe_bmm1_slice16_20_selector_prod_rep2500_s012.csv` as useful noise
  evidence, not selector evidence.
- Built a temporary cached harness,
  `/tmp/moe_bmm1_pair_tune_cached.py`, that aliases identical configs and
  reuses prepared tensors for configs with the same `BLOCK_K/BLOCK_N/NUM_WARPS`
  layout. Cached artifacts:
  - `/tmp/moe_bmm1_slice16_20_selector_cached_uniform_rep2500_s012.csv`
  - `/tmp/moe_bmm1_slice16_20_selector_cached_prod_rep2500_s012.csv`
- Cached selector outcome:
  - Uniform `batch=512` / `slice=16`: current selected `x5/w5` remains best on
    geometric speedup, `1.017x` versus 1CTA. No multicast variant produced a
    stable win.
  - Uniform `batch=640` / `slice=20`: all x5/x6 multicast variants are within
    about `0.03%` average of selected; this is below the promotion threshold.
  - Prod `batch=512`: x6 variants average about `0.1%` faster than selected,
    but uniform does not agree.
  - Prod `batch=640`: selected x6/w5 remains effectively tied with the best
    no-multicast variants (`<=0.02%` average gap).
- Decision: no selector source change. The current low-batch selector is still
  the safest policy for `slice=16` and `slice=20`; any future sub-percent
  selector change needs cached/aliased measurement or an even stricter
  alternating same-graph harness.

## 2026-04-18 Low-Footprint Inline Probe

- Added temporary scratch-harness candidates for four-warp direct `slice=28`
  variants with lower X/W staging and inline MMA input release. The goal was to
  see whether inline release could make a smaller shared-memory footprint
  viable without losing too much producer/consumer overlap.
- Artifacts:
  - `/tmp/moe_bmm1_slice28_inline_lowfootprint_uniform_rep1800_v2.csv`
  - `/tmp/moe_bmm1_slice28_inline_lowfootprint_prod_rep1200_v2.csv`
- Uniform `batch=896` remained the hard failure:
  - Current best four-warp `x5/w5` direct candidate: `0.929x` in this run.
  - `x5/w4` variants fell to `0.86x-0.87x`.
  - `x4/w5` variants fell to `0.907x-0.917x`.
  - `x4/w4` variants fell to `0.858x-0.866x`.
  - Inline release did not recover the lost staging depth; it usually made the
    lower-footprint variants worse.
- Prod `batch=896` showed a small positive result for
  `x4/w5/inline` (`1.006x`), but uniform is the selector gate and strongly
  rejects the path.
- Decision: do not pursue reduced X/W ring footprint via inline release for
  `slice=28`. The x5/w5 shared-memory footprint still appears necessary for
  uniform routing, despite its low occupancy.

## 2026-04-18 Route-Local Slice-28 Probes

- Narrow `BAND_N` sweep on uniform `batch=896`:
  - Initial four-seed artifact set:
    `/tmp/moe_bmm1_slice28_band_uniform_seed{0,1,2,3}_rep1600.csv`.
    `BAND_N=20/22` helped some seeds but did not fix seed 0.
  - Cached recheck artifacts:
    `/tmp/moe_bmm1_slice28_band_cached_uniform_seed{0,1,2,3}_rep3000.csv`.
    Current `BAND_N=32` remained best geometrically (`0.989x`), with
    `BAND_N=22` close but still below (`0.989x`) and worse on the hardest
    seed. No band-only selector change is justified.
- Extended `/tmp/moe_bmm1_pair_tune_cached.py` to support fixed
  `--local-ranks` and fixed a scratch-harness key bug where `NUM_WARPS` was
  not part of the config key. Any artifact from the cached harness before this
  fix must not be used to compare four- versus eight-warp variants.
- Fixed-local-rank uniform `batch=896` artifacts:
  - `/tmp/moe_bmm1_slice28_localrank_uniform_r0_1_rep1400.csv`
  - `/tmp/moe_bmm1_slice28_localrank_uniform_r2_3_rep1400.csv`
  - `/tmp/moe_bmm1_slice28_localrank_uniform_r4_5_rep1400.csv`
  - `/tmp/moe_bmm1_slice28_localrank_uniform_r6_7_rep1400.csv`
- Local-rank result:
  - 2CTA wins or ties on ranks `0`, `1`, `2`, and `6`.
  - 2CTA loses badly on ranks `3`, `4`, `5`, and `7`.
  - Losing ranks usually have four local experts above 32 rows, but total local
    tokens alone is not predictive. This points at route/block distribution
    sensitivity rather than a single global batch-size effect.
- Shape/schedule checks on the losing local ranks:
  - `M64/BN256` helper and direct candidates stayed below the current
    four-warp `M32/BN256` direct path. Artifacts:
    `/tmp/moe_bmm1_slice28_m64_uniform_rank{3,4,5,7}_rep1200.csv`.
  - `M16/BN256` direct on rank 4 was much worse (`~0.75x-0.76x`) even with a
    bounded compile/benchmark. Artifact:
    `/tmp/moe_bmm1_slice28_m16_uniform_rank4_timeout240_rep600.csv`.
  - Corrected eight-warp direct recheck on losing ranks gave only small local
    improvements and stayed below parity. Artifacts:
    `/tmp/moe_bmm1_slice28_8warp_uniform_rank{3,4,5,7}_rep1200_v2.csv`.
  - Full-tile schedule plus optional gather-index reuse regressed the losing
    ranks. Artifacts:
    `/tmp/moe_bmm1_slice28_fullsched_uniform_rank{3,4,5,7}_rep1200.csv`.
  - `BN512` direct variants were severe regressions on losing ranks, with
    two-warp variants around `0.75x-0.78x` and three-warp variants around
    `0.60x`. Artifacts:
    `/tmp/moe_bmm1_slice28_bn512_uniform_rank{3,4,5,7}_rep1000.csv`.
  - Eight-warp partition variants were rechecked after fixing the cached
    harness key. Most active-warp / weight-warp splits fail `warp_specialize`
    lowering; the only legal `act4/w1/mma1` register-56 variant regressed the
    losing ranks versus the four-warp direct path. Artifacts:
    `/tmp/moe_bmm1_slice28_8warp_partition_rank{3,4,5,7}_rep1200.csv`.
- Fixed-rank NCU reports for the hard rank-4 route:
  - `/tmp/ncu_moe_896_uniform_rank4_1cta_target.ncu-rep`
  - `/tmp/ncu_moe_896_uniform_rank4_2cta_warps4_x5w5.ncu-rep`
  - 1CTA: duration `37.888 us`, DRAM bandwidth `3.953 TB/s`, instructions
    `9.24M`, active warps `49.31%`, eligible warps/scheduler `0.741`, issue
    active `0.38`, registers/thread `64`, dynamic shared memory
    `114.432 KB/block`, and occupancy limited to two blocks by both shared
    memory and registers.
  - 2CTA four-warp direct: duration `39.968 us`, DRAM bandwidth `3.822 TB/s`,
    instructions `6.65M`, active warps `24.37%`, eligible warps/scheduler
    `0.369`, issue active `0.29`, registers/thread `52`, dynamic shared
    memory `96.988 KB/block`, and occupancy limited to two blocks by shared
    memory.
  - Interpretation: the hard route confirms the global `slice=28` diagnosis.
    2CTA executes fewer instructions, but lower active/eligible warp count and
    shared-memory-limited residency leave it slower than 1CTA.
- Current interpretation: `slice=28` is now known to be route-local. The
  unsolved case is not all uniform routes, but specific local-rank
  distributions with several 33+ row experts. M16, M64, BN512, eight-warps,
  full-tile scheduling, 8-warp partitioning, and band-only tuning do not solve
  those distributions.

## 2026-04-18 Slice-28 Mixed-Route Structural Probes

- Rebuilt with `make` first; ninja reported no work.
- Four-warp planar-snake schedule variants were tested on the hard uniform
  `batch=896` ranks. Artifact set:
  `/tmp/moe_bmm1_slice28_warps4_snake_rank{3,4,5,7}_rep1200.csv`.
  - Rank 3 had a small local improvement with `minor_dim=1,width=8`
    (`0.923x` versus current `0.918x` in that run).
  - Ranks 4, 5, and 7 all regressed versus the current four-warp direct path.
  - Decision: no planar-snake selector change.
- Four-warp multicast and `x6/w5` staging variants were tested on the same
  hard ranks. Artifact set:
  `/tmp/moe_bmm1_slice28_warps4_mc_rank{3,4,5,7}_rep1200.csv`.
  - `x6/w5` with both X gather and W-scale multicast disabled helped rank 4
    locally (`0.934x`) but regressed rank 5 and did not fix ranks 3 or 7.
  - Decision: no multicast or `x6/w5` selector change for `slice=28`.
- Four-warp register-cap variants were tested for both `x5/w5` and `x6/w5`.
  Artifact set:
  `/tmp/moe_bmm1_slice28_warps4_regs_rank{3,4,5,7}_rep1200.csv`.
  - Lower `MAXNREG=48` helped ranks 3, 4, and 5 slightly, while rank 7 stayed
    best at the current/higher cap. All variants remained below parity.
  - Decision: register caps are sub-percent tuning only; they do not solve the
    route-local gap.
- A temporary split diagnostic measured first-block-only and spill-only
  subroutes with the same prepared inputs. Artifacts:
  - `/tmp/moe_bmm1_slice28_hybrid_rank4_rep800.csv`
  - `/tmp/moe_bmm1_slice28_split_diag_rank{0,3,4,5,7}_rep800.csv`
  - Two-launch hybrid `first rows via 2CTA + spill rows via 1CTA` was correct
    but much slower (`~0.68x-0.73x`) because launch overhead dominates.
  - The diagnostic is still useful: on losing ranks, first-only 2CTA was near
    parity (`~0.983x-1.003x`) and spill-only 2CTA beat spill-only 1CTA, while
    the combined route stayed far below parity (`~0.904x-0.958x`). The gap is
    therefore not simply spill-row arithmetic; mixed full/spill block behavior
    inside the persistent loop is the problem.
- An opt-in layered block schedule was prototyped and then removed from source
  after measurement. Artifact set:
  `/tmp/moe_bmm1_slice28_layered_rank{3,4,5,7}_rep1200.csv`.
  - Scheduling all first blocks across experts before all spill blocks
    regressed every hard rank. The current slice-major block schedule remains
    better.
- An opt-in full-M activation gather descriptor was prototyped and removed
  after validation:
  - No-CGA X layout failed lowering because the shared-memory layout had one
    CTA per CGA inside a two-CTA context.
  - Zero-basis two-CTA X layout compiled after matching the gather-offset
    layout, but failed correctness with row-16 mismatches. Artifact:
    `/tmp/moe_bmm1_slice28_xfull_rank4_rep600_v3.csv`.
  - Decision: the MMA/descriptor mapping depends on the current row-CGA
    split; full-M gather is not a safe structural path without a deeper layout
    redesign.
- Controlled synthetic-route sweeps isolated the bad route pattern. Artifacts:
  - `/tmp/moe_bmm1_slice28_custom_routes_rank4_rep600.csv`
  - `/tmp/moe_bmm1_slice28_custom_routes_rank4_order_rep600.csv`
  - With the same `475` local tokens and `9` spill rows, no-spill routes were
    near parity (`~1.002x`), one-spill and two-spill routes were wins
    (`~1.007x-1.009x`), three moderate spill experts were near parity
    (`~0.996x`), and four moderate spill experts were the hard loss
    (`~0.91x-0.94x` depending on expert order). Nine tiny one-row spills were
    a win because 1CTA slowed more than 2CTA.
  - Conclusion: the failure is not total tokens, max rows, or spill rows alone.
    It is the interaction of several moderate `>32` experts with persistent
    loop scheduling and load/compute overlap.
- Scratch metadata-schedule probes changed only `block_schedule_data` for the
  precomputed 32-row schedule, without source changes. Artifacts:
  - `/tmp/moe_bmm1_slice28_schedule_order_rank4_actual_rep800.csv`
  - `/tmp/moe_bmm1_slice28_schedule_order_rank4_front4_rep800.csv`
  - `/tmp/moe_bmm1_slice28_schedule_order_rank{3,5,7}_actual_rep800.csv`
  - Schedule order can reduce synthetic front-clustered losses
    (`spill_experts_last` reached `0.950x` versus current `0.902x`), and
    `slice_major_reverse` helped rank 3 (`0.942x`), but no tested order reached
    parity on the actual hard ranks. Current slice-major remains best for
    ranks 4 and 7.
- W-ring depth was the most useful axis in this pass. Artifacts:
  - `/tmp/moe_bmm1_slice28_warps4_w6_rank{0,1,2,3,4,5,6,7}_rep1200.csv`
  - `/tmp/moe_bmm1_slice28_warps4_wdepth_rank{3,4,5,7}_rep900.csv`
  - `x5/w6` improved every fixed local rank. It preserved and improved the
    previous winning ranks (`~1.028x-1.042x` on ranks 0, 1, 2, 6) and lifted
    the hard ranks to about rank 3 `0.948x`, rank 4 `0.961x`, rank 5
    `0.948x`, rank 7 `0.974x`.
  - `x6/w6`, `x5/w7`, and `x5/w8` crossed a resource cliff and regressed to
    about `0.70x-0.72x`.
  - NCU report `/tmp/ncu_moe_896_uniform_rank4_2cta_warps4_x5w6.ncu-rep`
    showed why W6 helped: duration improved from `39.968 us` (W5) to
    `38.400 us`, DRAM bandwidth rose from `3.822 TB/s` to `3.984 TB/s`,
    eligible warps/scheduler rose from `0.369` to `0.389`, and dynamic shared
    memory rose from `96.988 KB/block` to `113.884 KB/block`.
- W6 combination checks:
  - Schedule-order plus W6/regs52 artifacts:
    `/tmp/moe_bmm1_slice28_schedule_order_w6_rank{3,4,5,7}_actual_rep800.csv`.
  - Schedule-order plus W6/regs48 artifacts:
    `/tmp/moe_bmm1_slice28_schedule_order_w6r48_rank{3,4,5,7}_actual_rep800.csv`.
  - W6 register sweep artifacts:
    `/tmp/moe_bmm1_slice28_w6_regs_rank{3,4,5,7}_rep1000.csv`.
  - W6 band artifacts:
    `/tmp/moe_bmm1_slice28_w6_band_rank{3,4,5,7}_rep900.csv`.
  - W6 multicast artifacts:
    `/tmp/moe_bmm1_slice28_w6_mc_rank{3,4,5,7}_rep900.csv`.
  - W6 inline artifacts:
    `/tmp/moe_bmm1_slice28_w6_inline_rank{3,4,5,7}_rep900.csv`.
  - W6 occupancy artifacts:
    `/tmp/moe_bmm1_slice28_w6_occ_rank{4,5}_rep900.csv`.
  - Best stacked W6 results stayed below parity: rank 3 reached about
    `0.979x` with `BAND_N=20`, rank 4 about `0.966x` with no multicast, rank
    5 about `0.960x` with `spill_first_reverse` schedule plus W6/regs48, and
    rank 7 about `0.976x` with W6/regs48. Inline release, W7/W8, eight-warps,
    and `OCCUPANCY=1/3` regressed.
- `BN384` direct candidates are illegal in this descriptor path. Artifacts:
  `/tmp/moe_bmm1_slice28_bn384_rank{3,4,5,7}_rep1000.csv`. The descriptor
  rejects the scale shape because a shape element must be a power of two.
- The cached scratch harness was corrected after discovering that the key
  aliased candidates with different warp ownership or `ACC_NUM_BUFS`. The
  scratch-only key now includes `NUM_WARPS`, activation/weight/MMA/store warp
  counts, and accumulator buffer depth before comparing variants.
- Additional W6/regs48 scratch probes after the key fix remained below parity.
  Artifacts:
  - `/tmp/moe_bmm1_slice28_w6_new_rank{37,45}_rep1200.csv`
  - `/tmp/moe_bmm1_slice28_w6_warpownership_keyfix_rank{37,45}_rep1200.csv`
  - `act1w2m1` regressed every hard rank (`~0.93x-0.97x`), `sub2` direct
    regressed (`~0.94x-0.98x`), `ACC_NUM_BUFS=2` was at best a small local
    lift on ranks 5/7 but still lost to 1CTA, and `x4/w6` regressed. Gather
    reuse and full-tile schedule variants did not close the gap.
  - Narrower band checks (`BAND_N=18`) and wider edge checks (`28/30`) did not
    beat the best earlier band choices. Rank 5 had a near-miss with
    no-multicast W6/regs48 (`0.997x`), and rank 7 reached `~0.992x` with
    no-multicast/no-scale-multicast variants, but ranks 3 and 4 remained well
    below parity.
- Smaller-M alternatives are not a viable slice-28 escape hatch. Artifact:
  `/tmp/moe_bmm1_slice28_small_m_uniform_rank3457_rep1200.csv`.
  - `BLOCK_M=28` and `BLOCK_M=24` direct candidates are illegal because the
    descriptor shape element must be a power of two.
  - Legal `BLOCK_M=16` direct variants were much slower (`~0.75x-0.77x`) on
    the hard ranks.
- A W6/regs48 front-spill schedule-order probe confirmed schedule sensitivity
  but no source-ready schedule win. Artifact:
  `/tmp/moe_bmm1_sched_w6_regs48_rank4_front4_rep2000.csv`.
  - `spill_experts_last` was best on the synthetic front-clustered rank-4
    shape at `0.976x`, followed by `slice_major_reverse` at `0.970x`; default
    slice-major was `0.947x`. Even the best reordered schedule stayed below
    1CTA.
- A temporary default-off source probe moved direct-epilogue bias loading after
  the accumulator wait, then was reverted after measurement. Artifact:
  `/tmp/moe_bmm1_slice28_w6_biaswait_rank3457_rep1200.csv`.
  - Bias-after-wait regressed every hard rank versus W6/regs48, e.g. rank 4
    dropped from `0.980x` to `0.960x` in that run and rank 5 from `0.965x` to
    `0.949x`. Keeping the existing bias prefetch before `acc_ready` is better.
- A temporary source probe split W data and W-scale staging into independent
  buffer/barrier rings, then was reverted after measurement. Artifacts:
  - `/tmp/moe_bmm1_slice28_wscale_legality_rank4_rep200.csv`
  - `/tmp/moe_bmm1_slice28_wscale_depth_rank4_rep800.csv`
  - The path compiled and validated, but separate scale barriers were too
    expensive. Rank 4 fell from `0.963x` for the original combined W6/regs48
    ring to `0.935x` even with equal scale depth (`ws6`), and shallower scale
    rings regressed further (`ws5` `0.900x`, `ws4` `0.840x`, `ws3` `0.731x`,
    `ws2` `0.560x`). The current combined W+scale barrier is better.
- NCU source/scheduler extraction on uniform `896`, local rank `4`, confirmed
  that W6 improves the best W5 2CTA family but does not solve the structural
  eligibility gap. Artifacts:
  - `/tmp/ncu_moe_896_uniform_rank4_1cta_target.ncu-rep`
  - `/tmp/ncu_moe_896_uniform_rank4_2cta_warps4_x5w5.ncu-rep`
  - `/tmp/ncu_moe_896_uniform_rank4_2cta_warps4_x5w6.ncu-rep`
  - 1CTA: issue active `0.38`, active warps/scheduler `7.73`, eligible
    warps/scheduler `0.74`, long-scoreboard `12.11` inst/issue, barrier
    `2.18`, wait `1.61`, registers/thread `64`, dynamic shared memory
    `114.432 KB`.
  - 2CTA W5: issue active `0.29`, active warps/scheduler `3.83`, eligible
    warps/scheduler `0.37`, long-scoreboard `7.48` inst/issue, barrier
    `1.72`, wait `1.40`, registers/thread `52`, dynamic shared memory
    `96.988 KB`.
  - 2CTA W6: issue active `0.30`, active warps/scheduler `3.87`, eligible
    warps/scheduler `0.39`, long-scoreboard `7.22` inst/issue, barrier
    `1.73`, wait `1.40`, registers/thread `52`, dynamic shared memory
    `113.884 KB`.
  - The source/SASS view attributes the dominant 2CTA W6 long-scoreboard
    samples to `SYNCS.PHASECHK.TRANS64.TRYWAIT` loops and nearby branch waits
    around TMA gather/W staging, with additional samples on
    `UTMALDG.2D.GATHER4.2CTA`. This makes loader/wait-loop overlap the next
    structural target; epilogue bias placement and split scale barriers are
    already measured dead ends.
- W6 no-multicast schedule-order probes on actual hard local ranks did not
  produce a host-ordering selector candidate. Artifacts:
  - `/tmp/moe_bmm1_sched_w6_nomc_rank3_actual_rep1200.csv`
  - `/tmp/moe_bmm1_sched_w6_nomc_rank4_actual_rep1200.csv`
  - `/tmp/moe_bmm1_sched_w6_nomc_rank5_actual_rep1200.csv`
  - `/tmp/moe_bmm1_sched_w6_nomc_rank7_actual_rep1200.csv`
  - Best observed speedups were rank 3 `0.964x` (`slice_major_reverse`),
    rank 4 `0.962x` (`slice_major_reverse` / `spill_first_reverse`), rank 5
    `0.952x` (`spill_first_reverse`), and rank 7 `0.966x` (`slice_major`).
- `BLOCK_K=64` scratch variants are illegal in the current scale descriptor
  layout. Artifact: `/tmp/moe_bmm1_slice28_bk64_warps4_rank4_rep800.csv`.
  Every BK64 candidate failed with
  `AssertionError('block_shape[0]=2 must be divisible by 4')`.
- X/W-scale multicast isolation on W6/regs48 did not yield a stable
  selector-ready configuration. Artifact:
  `/tmp/moe_bmm1_slice28_w6_xmc_iso_rank3457_rep1200.csv`.
  - Rank 3 still lost (`0.954x`) and rank 5 still lost (`0.973x`).
  - Rank 4 and rank 7 showed local >1.0 readings (`1.002x` for no-X
    multicast on rank 4 and `1.008x` for no-scale multicast on rank 7), but
    the same family has previously measured below parity on nearby repeated
    hard-route runs, so these are not sufficient for promotion.
- W6/regs48 planar-snake variants are a dead end. Artifact:
  `/tmp/moe_bmm1_slice28_w6_snake_rank3457_rep1200.csv`.
  - The default W6/regs48 schedule beat every snake variant on all four hard
    ranks. Best speedups remained rank 3 `0.952x`, rank 4 `1.002x`, rank 5
    `0.973x`, and rank 7 `1.002x`; ranks 3 and 5 still block promotion.
- Route-matched NCU for the current W6/regs48 near-miss confirms that the
  remaining gap is still low eligibility under shared-memory-limited
  residency, not instruction count. Artifacts:
  - `/tmp/ncu_moe_896_uniform_rank3_1cta_target.ncu-rep`
  - `/tmp/ncu_moe_896_uniform_rank3_2cta_warps4_x5w6.ncu-rep`
  - `/tmp/ncu_moe_896_uniform_rank4_2cta_warps4_x5w6_regs48.ncu-rep`
  - Rank 3 1CTA: duration `38.112 us`, DRAM bandwidth `3.949 TB/s`,
    instructions `14.13M`, issue active `0.38`, active warps/scheduler
    `7.72`, eligible warps/scheduler `0.741`, registers/thread `64`,
    dynamic shared memory `114.432 KB`, shared-memory occupancy limit
    `2` blocks.
  - Rank 3 W6/regs48 2CTA: duration `39.072 us`, DRAM bandwidth
    `3.946 TB/s`, instructions `8.35M`, issue active `0.29`, active
    warps/scheduler `3.88`, eligible warps/scheduler `0.364`,
    registers/thread `48`, dynamic shared memory `113.884 KB`,
    shared-memory occupancy limit `2` blocks.
  - Rank 4 W6/regs48 2CTA was similar on eligibility (`0.381`) and remained
    shared-memory limited to `2` blocks. Lowering the register cap removes the
    register occupancy limit, but not the actual shared-memory residency limit.
    Further occupancy hints alone are unlikely to help unless the W/X/scale
    footprint drops or the producer/consumer wait timing changes.
- A temporary source probe changed the MMA partition to wait for X before W,
  then was reverted after measurement. Artifact:
  `/tmp/moe_bmm1_slice28_w6_waitx_rank3457_rep1200.csv`.
  - The probe slightly lifted rank 3 in one run (`0.952x` vs `0.950x` for
    default W6/regs48), but still lost to 1CTA and regressed ranks 4, 5, and 7
    versus the default W-before-X order. Keeping the existing W wait, scale
    copy, then X wait order is better because it can overlap scale movement
    with X readiness.
- W6 banding interactions still do not close slice 28. Artifacts:
  - `/tmp/moe_bmm1_slice28_w6_band_mc_combo_rank3457_rep1200.csv`
  - `/tmp/moe_bmm1_slice28_w6_band_acc2_rank3457_rep1200.csv`
  - `/tmp/moe_bmm1_slice28_w6_b20_regs_rank3_rep1400.csv`
  - `BAND_N=20` remains the best rank-3 local result at about `0.978x`, and
    disabling scale or X multicast makes that rank worse.
  - `BAND_N=24 + ACC_NUM_BUFS=2` can lift rank 4 locally (`1.004x` in this
    run), but rank 5 remains below parity (`0.973x`) and rank 7 is only about
    parity. It does not solve the hard-rank set.
  - A rank-3 `BAND_N=20` register sweep (`MAXNREG=52/56/60`) also stayed below
    parity; best was `regs56` at `0.977x`, so the rank-3 near-miss is not a
    simple register-cap artifact.
- Direct `BLOCK_K=256` variants are illegal too. Artifact:
  `/tmp/moe_bmm1_slice28_bk256_direct_rank3_rep400.csv`.
  - Direct four-warp `BK256` variants with `x3/w3`, `x4/w3`, and `x3/w4` all
    failed with `AssertionError()`, matching earlier non-direct `BK256`
    failures. Larger-K is not a legal way to reduce K-loop wait frequency in
    this descriptor path.
- NCU and schedule-order follow-up on the rank-3 `BAND_N=20` near-miss did not
  expose a promotion path. Artifacts:
  - `/tmp/ncu_moe_896_uniform_rank3_2cta_warps4_x5w6_b20_regs48.ncu-rep`
  - `/tmp/moe_bmm1_sched_w6_b20_rank3_actual_rep1200.csv`
  - The B20 NCU report still shows the same shared-memory residency limit
    (`2` blocks) and low eligible warps/scheduler (`0.378`), only a small
    improvement over default W6/regs48 (`0.364`) and still far below 1CTA
    (`0.741`).
  - Custom block orders did not improve beyond the B20 near-miss:
    `full_first_reverse` reached `0.978x`; other orders regressed.
- A temporary source probe swapped the direct-path warp-specialization order
  so the weight loader was listed before the activation loader, then was
  reverted. Artifact: `/tmp/moe_bmm1_slice28_w6_weightfirst_rank3457_rep1200.csv`.
  - Weight-first ordering improved rank 5 and rank 7 locally in one run, but
    regressed the rank-3 `BAND_N=20` near-miss from `~0.978x` to `0.972x`.
    It is not selector-safe.
- Inspected the direct-store shared-memory alias suggested by the profiler
  follow-up. In the direct path, `store_bufs = x_bufs` is a dummy alias to fill
  `PartitionArgs`; `epilogue_direct_store` writes directly to global memory
  and does not use a store ring. Replacing it with a dedicated packed-output
  ring would add shared memory rather than reduce the current residency limit.
- A gather-index reuse check on the best B20 family regressed the remaining
  blockers. Artifact: `/tmp/moe_bmm1_slice28_w6_b20_reuse_rank35_rep1200.csv`.
  - Rank 3 fell from `0.972x` to `0.965x` in the paired run, and rank 5 also
    regressed. This makes a more invasive shared gather-index fanout between
    CTAs unlikely to pay for the extra shared state and synchronization.
- A temporary source probe moved `mma_partition` first in the direct-path
  `gl.warp_specialize` ordering, then was reverted after compile failure.
  Artifact: `/tmp/moe_bmm1_slice28_w6_mmafirst_rank35_rep500.csv`.
  - The paired 1CTA baseline row compiled and ran, but every direct 2CTA
    candidate failed in `gl.warp_specialize` during compilation. This makes
    MMA-first primary partition ownership illegal for the current direct-store
    warp-specialized shape rather than a viable low-eligibility fix.
- W6 MMA-warp ownership and warp-count follow-ups did not improve the hard
  slice-28 ranks. Artifacts:
  - `/tmp/moe_bmm1_slice28_w6_mmawarps_rank3457_rep800.csv`
  - `/tmp/moe_bmm1_slice28_warpcount_rank35_rep400.csv`
  - Moving from `act2/w1/m1` to `act1/w1/m2` was legal but slower on every
    blocking rank. Rank 3 fell to `0.931x` or worse, rank 4 stayed below the
    existing B24/ACC2 comparator, rank 5 fell to about `0.962x`, and rank 7
    lost the existing positive margin.
  - `NUM_WARPS=6` is illegal (`num_warps must be a power of 2`), and
    16-warps direct variants were severe regressions (`~0.62x-0.65x`) on
    ranks 3 and 5. Extra warp ownership is not the missing slice-28 lever.
- A temporary source probe delayed the MMA partition's W-scale
  `tcgen05_copy` until after the X-ready wait, then was reverted. Artifact:
  `/tmp/moe_bmm1_slice28_w6_scale_after_x_rank35_rep900.csv`.
  - The rank-3 B20 near-miss stayed around the existing result (`0.979x`),
    while rank 5 remained below parity and the B24/ACC2 comparator regressed
    to `0.988x`. Keeping the scale copy between the W-ready wait and X-ready
    wait is better because it overlaps scale movement with X readiness.
- Temporary one-sided inline input-release source probes timed out and were
  reverted. Artifacts:
  - `/tmp/moe_bmm1_slice28_w6_inline_xonly_timeout.log`
  - `/tmp/moe_bmm1_slice28_w6_inline_wonly_timeout.log`
  - Both probes printed the paired 1CTA baseline row and then timed out on the
    first inline-release W6 candidate (`timeout_status=124`). Releasing only
    one MMA input barrier is hang-prone in this warp-specialized protocol.
    The already-tested all-input inline release is legal but slower, so this
    release axis is closed for the current direct W6 family.
- Route-matched NCU for the rank-5 B24/ACC2 near-miss shows the same structural
  limiter as ranks 3 and 4. Artifacts:
  - `/tmp/ncu_moe_896_uniform_rank5_1cta_target.ncu-rep`
  - `/tmp/ncu_moe_896_uniform_rank5_2cta_warps4_x5w6_b24_acc2.ncu-rep`
  - Rank 5 1CTA: duration `37.376 us`, DRAM bandwidth `4.017 TB/s`,
    instructions `14.14M`, active warps/scheduler `7.72`, eligible
    warps/scheduler `0.754`, registers/thread `64`, dynamic shared memory
    `114.432 KB`, shared-memory occupancy limit `2` blocks.
  - Rank 5 B24/ACC2 2CTA: duration `40.384 us`, DRAM bandwidth `3.787 TB/s`,
    instructions `8.35M`, active warps/scheduler `3.85`, eligible
    warps/scheduler `0.371`, registers/thread `48`, dynamic shared memory
    `113.892 KB`, shared-memory occupancy limit `2` blocks.
  - The 2CTA path again wins on instruction count but loses on resident and
    eligible warp count. The remaining rank-5 gap is latency hiding, not raw
    instruction volume.
- Deeper accumulator buffering for the B24 W6 family does not solve the hard
  ranks. Artifact:
  `/tmp/moe_bmm1_slice28_w6_b24_accdepth_rank3457_rep800.csv`.
  - `ACC_NUM_BUFS=3` slightly lifted rank 3 within B24 (`0.959x`) but still
    lost badly to the B20 near-miss, while ranks 4, 5, and 7 were best with
    ACC1 or ACC2 and remained below parity. `ACC_NUM_BUFS=4` was worse.
- Deeper accumulator buffering for the B20 W6 family also does not solve the
  hard ranks. Artifact:
  `/tmp/moe_bmm1_slice28_w6_b20_accdepth_rank3457_rep800.csv`.
  - ACC2/ACC3/ACC4 did not preserve the best B20 rank-3 near-miss and stayed
    well below parity on ranks 4, 5, and 7. The B20 and B24 evidence together
    makes additional accumulator ring depth unattractive for this family.
- Helper-store W6 variants are still severe regressions. Artifact:
  `/tmp/moe_bmm1_slice28_w6_helper_rank35_rep500.csv`.
  - B20/B22/B24 helper-store variants with W6 measured only about
    `0.64x-0.66x` on ranks 3 and 5. The low-batch W6 family needs direct
    epilogue stores; returning to helper-store overlap adds too much cost and
    shared-memory pressure.
- W6 `OCCUPANCY=4` launch-grid overdecomposition is a dead end. Artifact:
  `/tmp/moe_bmm1_slice28_w6_occ4_rank35_rep600.csv`.
  - Default, B20, and B24 OCC4 variants regressed to about `0.85x-0.87x` on
    ranks 3 and 5. Prior OCC1/OCC3 checks were also bad, so the persistent
    launch-grid density axis is closed for the current W6 direct family.
- Odd W6 band factors expose rank-specific optima but not a selector-safe
  policy. Artifact:
  `/tmp/moe_bmm1_slice28_w6_oddbands_rank3457_rep800.csv`.
  - In the same-input rank sweep, B20 remained best for rank 3 (`0.979x`),
    B25 was best for rank 4 (`0.997x`), B19 crossed parity on rank 5
    (`1.006x`), and B24 crossed parity on rank 7 (`1.015x`). The axis is a
    useful routing-rank specialization clue, but no single band beats 1CTA on
    all hard fixed ranks.
- Odd-band multicast/ACC follow-up confirms the same route-specific split, not
  a selector-safe policy. Artifacts:
  - `/tmp/moe_bmm1_slice28_w6_oddcombo_rank34_rep700.csv`
  - `/tmp/moe_bmm1_slice28_w6_oddcombo_rank57_rep700.csv`
  - `/tmp/moe_bmm1_slice28_w6_oddcombo_confirm_rank3457_rep1200.csv`
  - The quiet single-GPU confirm found rank 3 best with B20 and W-scale
    multicast disabled (`0.983x`), rank 4 best with B23 (`0.997x`), rank 5
    best with B19 (`1.009x`), and rank 7 best with B24 (`1.015x`). Disabling
    W-scale multicast can help a local rank, but the winning combination still
    varies by route and does not fix ranks 3 or 4.
- A temporary source probe reduced 2CTA activation shared-memory allocation
  from `BLOCK_M` rows to `BLOCK_M_PER_CTA` rows, then was reverted after
  compile failure. Artifacts:
  - `/tmp/moe_bmm1_slice28_xlocal_legality_rank3_rep200.csv`
  - `/tmp/moe_bmm1_slice28_xlocal_legality_rank3_retry_rep200.csv`
  - The first attempt used the wrong local constexpr name. The corrected
    allocation failed because `ttng.async_tma_gather` still requires 32 index
    rows while the memory descriptor had 16 rows, and `tc_gen5_mma_scaled`
    then saw an incompatible operand/output shape. Simple CTA-local X storage
    is not legal without also changing the gather/MMA/accumulator layout
    contract.
- Temporary source probes reduced the 2CTA W shared-memory tile, and then the
  W+scale shared-memory tiles, to CTA-local N extents. Both were reverted after
  compile failure. Artifacts:
  - `/tmp/moe_bmm1_slice28_wlocal_legality_rank3_rep200.csv`
  - `/tmp/moe_bmm1_slice28_wlocal_wonly_legality_rank3_rep200.csv`
  - Both probes failed during `gl.warp_specialize` parsing for the current
    direct W6 layout. Like X storage, simple CTA-local W storage cannot be
    dropped into the existing descriptor/MMA/accumulator contract; a W
    footprint reduction needs a deeper layout rewrite rather than only
    shrinking the shared-memory allocation shape.
- Full-tile scheduling does not combine profitably with the odd-band W6
  near-misses. Artifact:
  `/tmp/moe_bmm1_slice28_w6_odd_fullsched_rank3457_rep900.csv`.
  - The fullsched variants failed to improve the blocking ranks and usually
    regressed versus their non-fullsched comparators. Rank 3 topped out at
    `0.968x`, rank 4 at `0.980x`, and rank 5 at `0.980x` in this run; rank 7
    still had positive non-fullsched rows, but fullsched was lower. Host
    full-tile ordering remains closed for the current W6 odd-band family.
- Register caps do not rescue the odd-band multicast near-misses. Artifact:
  `/tmp/moe_bmm1_slice28_w6_odd_regs_mc_rank34_rep1000.csv`.
  - On the rank-3 B20/no-scale-multicast path, `MAXNREG=52` was best at
    `0.979x`; higher caps were worse. On rank 4, B23/regs48 remained best at
    `0.997x`, with B23 regs56 and B25/no-multicast regs48/56 also below
    parity. The rank-3 and rank-4 gaps are not fixed by pairing odd bands,
    multicast toggles, and register caps.
- A temporary source probe issued the W-scale TMA before the W-data TMA in
  `load_weights`, then was reverted after measurement. Artifacts:
  - `/tmp/moe_bmm1_slice28_scale_tma_first_legality_rank3_rep250.csv`
  - `/tmp/moe_bmm1_slice28_scale_tma_first_rank4_rep400.csv`
  - The path was legal and correctness-gated, but it regressed the rank-3
    B20/no-scale/regs52 row to `0.965x` and the rank-4 B23/B25 rows to about
    `0.965x-0.966x`. Keeping W data before W scale in the shared ready
    barrier is better for the current loader/MMA overlap.
- NCU on the best current rank-3 odd-band near-miss shows the same structural
  limiter as earlier W6 profiles. Artifact:
  `/tmp/ncu_moe_896_uniform_rank3_2cta_warps4_x5w6_b20_nomcscale_regs52.ncu-rep`.
  - Profile target: uniform batch `896`, local rank `3`,
    `m32_bn256_sub1_direct_warps4_x5w6_b20_nomcscale_act2w1m1_regs52_epin1_b32`.
  - Duration was `39.78 us`, executed instructions `6.85M`, active
    warps/scheduler `3.86`, eligible warps/scheduler `0.39`,
    registers/thread `52`, dynamic shared memory `113.88 KB`, and
    shared-memory occupancy limit `2` blocks.
  - The Nsight rule output again identifies low issue-slot utilization from
    too few eligible warps, shared-memory-limited theoretical occupancy, and
    long-scoreboard stalls. Disabling W-scale multicast and raising the
    register cap shifts local timing but does not change the core limiter.
- A read-only W-reuse feasibility pass found that a `load_weights`-local cache
  is not legal under the current barrier protocol. W addressing ignores
  `pid_m`, so reuse is mathematically available, but each W ring stage has one
  ready/empty lifecycle and `mma_partition` releases `w_empty_bar` after one
  MMA use. Reusing W for a second `pid_m` block would require pair-aware
  scheduling, two live accumulators or interleaved K loops, and delayed W
  release; a small cache-only source patch risks overwrite or deadlock.
- X-depth interactions around the odd-band W6 near-misses do not improve the
  hard ranks. Artifact:
  `/tmp/moe_bmm1_slice28_w6_odd_xdepth_rank34_rep900.csv`.
  - For B20/no-scale/regs52, B23, and B25/no-multicast variants, `x4/w6`
    regressed and `x6/w6` hit a severe resource cliff (`~0.69x-0.72x` on
    rank 4). The existing `x5/w6` pairing remained best, at `0.979x` on rank
    3 and `0.997x` on rank 4.
- W-depth interactions around the odd-band near-misses also keep `x5/w6` as
  the best local depth pairing. Artifact:
  `/tmp/moe_bmm1_slice28_w6_odd_wdepth_rank34_rep900.csv`.
  - Retesting `x5/w5` and `x6/w5` against `x5/w6` for B20/no-scale/regs52,
    B23, and B25/no-multicast showed W5 materially worse on ranks 3 and 4.
    Best rows stayed `x5/w6`, at `0.976x` for rank 3 and `0.997x` for rank 4.
- A contiguous B16-B31 sweep and focused B21/B28 confirmation did not uncover
  a hidden selector-safe band. Artifacts:
  - `/tmp/moe_bmm1_slice28_w6_band16_31_rank34_rep800.csv`
  - `/tmp/moe_bmm1_slice28_w6_band21_28_confirm_rank34_rep1200.csv`
  - The broad sweep found local bests at B21 for rank 3 (`0.982x`) and B28 for
    rank 4 (`0.997x`). The longer confirm kept B21/plain as rank 3's best
    (`0.982x`) but still below parity, and B28/plain was only `0.995x` on
    rank 4. B21/B28 multicast and register interactions regressed.
- Added a repo-local structural exploration harness:
  `python/examples/gluon/06-moe-bmm1-structural-explore.py`. It defines new
  Gluon kernel entry points that reuse the same descriptors, reference checks,
  and prepared inputs but change producer/MMA ownership:
  - `combined`: one fused input producer issues both gathered X and W/scale TMA.
  - `mmaw`: activation producer stays separate; MMA owns W/scale TMA.
  - `mmax`: weight producer stays separate; MMA owns gathered X TMA.
  Artifacts:
  - `/tmp/moe_bmm1_structural_explore_selfcontained_smoke.csv`
  - `/tmp/moe_bmm1_structural_explore_rank34_selfcontained_rep180.csv`
  - The first attempted `combined@l3m1` shape failed compile because Gluon
    blocked layouts require power-of-two worker-warps; the script now rejects
    non-power-of-two worker counts before compilation.
  - All legal structural ownership merges were correctness-gated but much
    slower than the split-producer baseline. On uniform batch `896`, rank 3:
    split B21 was `0.975x`, `combined@l2m1` was `0.886x`, `mmaw@l2m1` was
    `0.380x`, and `mmax@w1m1` was `0.527x` versus 1CTA. On rank 4: split B21
    was `1.004x`, `combined@l2m1` was `0.905x`, `mmaw@l2m1` was `0.383x`, and
    `mmax@w1m1` was `0.530x`.
  - Decision: do not pursue simple producer-collapse structures further. The
    existing split activation/weight producer ownership is essential for TMA
    overlap; future structural work should preserve independent producers and
    instead target epilogue ownership, shared-memory footprint, or pair-aware
    scheduling with delayed W release.
- Aggressively tuned the structural exploration harness instead of stopping at
  the first structural pass.
  - Harness changes:
    `python/examples/gluon/06-moe-bmm1-structural-explore.py` now supports
    `--candidates-file`, parses 4- and 8-warp base names, and accepts
    per-candidate overrides for X/W/ACC buffer depth, band, warp count,
    occupancy, SwiGLU subtile factor, epilogue buffer depth, epilogue N-warp
    forcing, multicast toggles, worker splits, `MAXNREG`, and explicit
    activation/weight/MMA partition register budgets.
  - Aggressive rank-3 family sweeps:
    `/tmp/moe_bmm1_struct_combined_rank3_aggressive_rep30.csv`,
    `/tmp/moe_bmm1_struct_mmax_rank3_aggressive_rep30.csv`, and
    `/tmp/moe_bmm1_struct_mmaw_rank3_aggressive_rep30.csv`.
    Best `combined` improved from the first-pass `~0.886x` to `0.909x`
    versus 1CTA, while `mmax` topped out at `0.547x` and `mmaw` at `0.407x`.
    The useful ridge was `combined@l1m1,x6,w5,b21,regs52,nomcscale`; W6 in
    the combined structure hit a severe resource cliff, and 8-warp variants
    were not competitive.
  - Wider route-rank ridge sweep:
    `/tmp/moe_bmm1_struct_combined_ridge_rank{3,4,5,7}_rep20.csv`.
    Short-rep local bests were rank 3 `0.928x` at B16, rank 4 `0.911x`,
    rank 5 `0.918x` at B19/B22, and rank 7 `0.935x`. Route-specific banding
    was the dominant axis; no single combined row approached the split
    producer baseline across hard ranks.
- Added two more scratch structural kernels that change input-producer issue
  order:
  - `combinedx:` issues gathered X as soon as its ring slot is free, then
    waits/issues W+scale.
  - `combinedw:` issues W+scale first, then waits/issues gathered X.
  Artifacts:
  - Smoke: `/tmp/moe_bmm1_structural_order_smoke.csv`.
  - Aggressive order sweep:
    `/tmp/moe_bmm1_struct_combined_order_rank{3,4,5,7}_rep20.csv`.
  - Same-GPU long confirm:
    `/tmp/moe_bmm1_struct_order_confirm_rank3457_rep500.csv`.
  - Long-confirm best structural rows versus 1CTA were rank 3 `0.922x`
    (`combinedw@x6,w5,b16,regs56,nomcscale`), rank 4 `0.935x`
    (`combinedx@x6,w5,b32,regs48,nomcscale`), rank 5 `0.931x`
    (`combinedw@x6,w5,b19,regs60,nomcscale`), and rank 7 `0.951x`
    (`combinedw@x6,w5,b16,regs56,nomcscale`). These are improvements over the
    first structural pass but still below the split-producer baselines on all
    hard ranks; the short-rep `0.937x` rank-3 result did not survive the
    `rep=500` confirmation.
  - Decision: keep the order modes in the scratch harness as diagnostic tools,
    but do not port them into `05-moe-bmm1-fused-gather.py` or selector policy.
    Producer-collapse and producer-order rewrites reduce overlap too much; the
    next promising work should preserve separate producers and target
    shared-memory footprint, epilogue ownership, or delayed-release pair-aware
    scheduling.
- Pair-aware W reuse has limited immediate surface at the current M32 shape,
  and the M16 family is too slow to use as an easy pair-reuse base. Artifacts:
  - Route diagnostic for uniform batch `896`, seed `0`, fixed ranks `3/4/5/7`:
    at `BLOCK_M=32`, each rank had only `4` extra M blocks that could reuse W
    for a second tile; at `BLOCK_M=16`, ranks had `19-20` extra pairable
    blocks.
  - `/tmp/moe_bmm1_slice28_m16_pair_surface_rank34_rep180.csv`
  - On ranks 3 and 4, the tested M16 direct candidates were only
    `0.754x-0.768x` versus 1CTA, while the M32 W6 split baseline stayed near
    `0.976x` and `0.998x`. A delayed-release pair kernel would need to start
    from the M32 family or radically improve M16; do not spend a large rewrite
    on the existing M16 shape as-is.
- Decision: do not promote a slice-28 selector change yet. W6/regs48 is the
  new best near-miss family and should be the baseline for future slice-28
  work, but it still loses to 1CTA on hard uniform fixed-rank routes.

## 2026-04-20 20% Criterion And Paired W-Reuse Scratch Kernels

- Updated `AGENTS.md` to record the stronger success criterion: 2CTA must be
  at least `1.20x` faster than the 1CTA comparator for every batch size before
  the workstream is considered finished, and authorized subagent usage should
  be used aggressively for research, tuning, benchmarking, profiling, and
  verification.
- Subagent rehydration found that the current selector still leaves generated
  sweep slices `4,5,6,7,8,10,12,14,28` on 1CTA. It also highlighted that
  already-enabled low slices are only small wins under current evidence
  (`~1.00x-1.03x` for batches `512..1024`, depending on route), so the new
  `1.20x` target requires structural improvement rather than only enabling
  the slice-28 near-miss family.
- Existing W6/direct hard-rank recheck:
  `/tmp/moe_bmm1_20pct_slice28_hard_probe1.csv`
  - Command: `make`, then
    `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/code/triton/python:/root/code/triton/python/triton_kernels python /tmp/moe_bmm1_pair_tune_cached.py --batches 896 --routing uniform --seeds 0 --local-ranks 3 4 5 7 --candidates 1cta selected ...W6/B20..B26... --rep 400 --validate --shuffle`.
  - Best results remained far below the `1.20x` target: hard ranks 3/4/5
    still preferred 1CTA or were below parity; rank 7 reached only `1.006x`
    for `x5w6/B24/acc2/regs48`.
- Added paired W-reuse scratch modes to
  `python/examples/gluon/06-moe-bmm1-structural-explore.py`:
  - `pair:` uses one combined producer that issues W once per K tile and
    immediately applies it to two X tiles when the launch-grid-aware schedule
    can place same-`(slice,pid_n)` blocks in consecutive iterations of the same
    persistent program.
  - `pairsplit:` preserves separate activation and weight producers while using
    the same paired schedule and a paired MMA consumer.
  - Added minimal M16 direct bases for paired experiments.
- Paired W-reuse legality and tuning artifacts:
  - `/tmp/moe_bmm1_pair_reuse_compile_rank4.csv`: first `pair:` compile and
    correctness gate passed, but rank 4 was `0.860x`.
  - `/tmp/moe_bmm1_pair_reuse_split_compile_rank4.csv`: `pairsplit:` improved
    the paired structure to `0.933x`, still slower than the original split W6
    candidate.
  - `/tmp/moe_bmm1_pairsplit_tune_rank4_rep120.csv`: M32 paired split tuning
    across `x`, `w`, `acc`, multicast, warp, and register options peaked around
    `0.934x`; larger X rings and 8-warp variants were much slower.
  - `/tmp/moe_bmm1_pairsplit_m16_rank4_rep100.csv`: M16 paired split improved
    the old M16 direct base from `0.747x` to `0.856x`, but remained far below
    the M32 split baseline and 1CTA.
  - `/tmp/moe_bmm1_pairsplit_hard_ranks_rep200.csv`: best paired split
    candidates across hard ranks were still regressions (`~0.925x-0.958x`).
  - Decision: paired W reuse is legal but not competitive in this form. At
    M32/batch-896 the paired schedule covers only about `20%` of blocks, so the
    upside is too small; M16 increases pairable work but starts from a much
    slower tile.
- Disabled-small-batch probe:
  `/tmp/moe_bmm1_small_disabled_probe_rank4_rep200.csv`
  - Uniform rank 4, batches `128..448`, same-input cached harness, `rep=200`.
  - `x5w6/regs48` produced mild wins for batches `128..384`
    (`1.02x-1.05x`) but batch `448` still regressed (`0.966x`). This is useful
    evidence for future selector work, but still far below the `1.20x` target.
- Occupancy and wider-N checks:
  - `/tmp/moe_bmm1_w6_occupancy_recheck_rep300.csv`: explicit W6
    `occ3/occ4` variants were much slower on hard ranks; occupancy alone is
    not the fix.
  - `/tmp/moe_bmm1_warps8_w6_rank4_rep160.csv`: 8-warp W6 split variants were
    `~0.69x`, much worse than four-warps.
  - `/tmp/moe_bmm1_wide_n_rank4_rep200.csv`: `BN384` candidates are illegal
    with the current scale descriptor (`Shape element 4 must be a power of 2`),
    and `BN512` candidates are consistently slower.
- NCU sample, GPU 0, uniform batch `896`, seed `0`, rank `4`:
  - 2CTA W6/B21 command:
    `CUDA_VISIBLE_DEVICES=0 ... ncu --target-processes all --kernel-name-base function --kernel-name regex:.*ws_matmul_kernel.* --launch-count 1 --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active python /tmp/moe_bmm1_pair_tune.py --batches 896 --routing uniform --seeds 0 --local-ranks 4 --candidates m32_bn256_sub1_direct_warps4_x5w6_b21_act2w1m1_regs48_epin1_b32 --rep 1`.
  - 2CTA W6/B21 sample: `40.64 us`, SM throughput `51.70%`, DRAM
    throughput `47.48%`, L2 throughput `35.87%`, active warps `24.09%`.
  - 1CTA comparator sample: `37.31 us`, SM throughput `55.62%`, DRAM
    throughput `50.71%`, L2 throughput `38.27%`, active warps `48.27%`.
  - Conclusion: the hard-rank 2CTA W6 path is not only memory limited; it has
    materially lower active-warps and lower SM throughput than 1CTA. Future
    source work should change CTA/layout/epilogue ownership or in-kernel work
    decomposition, not just add occupancy, warps, wider N, or paired scheduling
    on top of the current M32 shape.

## 2026-04-20 Structural Scratch Follow-Up

- Added additional scratch modes and parser controls to
  `python/examples/gluon/06-moe-bmm1-structural-explore.py`:
  - `mmamc:` passes `multicast=p.USE_2CTA` to the scaled MMA.
  - `epiload:`/`epiload1:`/`epiload2:`/`epiload3:` test the legal explicit
    TMEM load layouts reported by lowering for the transposed `[BLOCK_N,
    BLOCK_M]` accumulator.
  - `splitws:` is a scratch proxy for the production direct split pipeline.
  - `mmacount:` tests descriptor-aware `tcgen05_commit(..., descs=[...])`
    and barrier counts following the multicta matmul examples.
  - `split:` candidates can now carry tuning options (`bm`, `bn`, `bk`,
    `ctas`, helper/wide store, store-helper warps/regs, inline release, and
    existing buffer/register/band knobs) so production-kernel structural
    variants can be benchmarked without adding one-off candidate names.
- Subagent findings:
  - Hume recommended epilogue TMEM load ownership, MMA multicast/barrier
    counts, accumulator buffering, and helper/wide-handoff epilogues as the
    next concrete experiments; descriptor CGA flips were called out as higher
    risk.
  - Schrodinger ran a GPU 2 hard-rank sweep in
    `/tmp/moe_bmm1_agent_structural_sweep.csv`; all `80/80` rows validated.
    Best non-1CTA speedups were still below parity: rank 3 `0.9756x`
    (`mmamc`), rank 4 `0.9676x` (split W6/B21), rank 5 `0.9477x`
    (`x4w6/B26`), rank 7 `0.9829x` (split W6/B21).
- Scratch and production-variant classification artifacts:
  - `/tmp/moe_bmm1_mmamc_rank4_smoke.csv`: `mmamc` compiled and validated but
    remained a regression (`0.9667x` in the first smoke).
  - `/tmp/moe_bmm1_epiload_linear_variants_rank4.csv`: the first legal
    explicit TMEM load layout was neutral/slightly worse (`0.9657x`), while
    the other legal layouts were much slower (`~0.697x-0.703x`).
  - `/tmp/moe_bmm1_splitws_inline_rank4.csv`: inline MMA input release was
    slower (`0.952x`); `acc3+inline` recovered only to `0.9655x`.
  - `/tmp/moe_bmm1_mmacount_rank4.csv`: descriptor-aware counted commits were
    slower (`0.9536x`).
  - `/tmp/moe_bmm1_helper_store_rank4.csv`: helper-store and wide-handoff M32
    2CTA epilogues were all far slower (`~0.656x-0.665x`).
  - `/tmp/moe_bmm1_shape_flex_rank4.csv`: flexible production shape probes did
    not find a better structure; M64 helper was the best alternate shape at
    only `0.9008x`, M64 direct was `~0.77x-0.81x`, M16 direct was `0.745x`,
    and M128 helper was `0.403x`.
  - `/tmp/moe_bmm1_rank4_param_sweep.csv`: a 191-candidate M32/BN256 direct
    sweep around bands, X/W rings, registers, and `acc3` peaked at `0.9676x`
    (`x5w6/B26/regs52`), essentially tied with the previous W6 near-miss and
    still far below the `1.20x` target.
- Decision: none of the follow-up synchronization, epilogue ownership, helper
  store, shape, or parameter sweeps are promotable. Keep the scratch modes and
  flexible parser as diagnostics, but the next winning attempt needs a more
  fundamental work decomposition than the current M32/BN256 2CTA split.

## 2026-04-20 Natural And Fake-N-Split Scratch Kernels

- Classified a natural `[BLOCK_M, BLOCK_N]` accumulator rewrite:
  - BM64/BM128 natural 2CTA is illegal because scaled MMAv5 rejects the
    resulting per-CTA M instruction shape (`32` or `64` rows). A subagent
    confirmed that a separate natural-only kernel would not fix this; legal
    natural true-2CTA needs `BLOCK_M / CTA == 128`.
  - BM256 natural compiled only after using a compatible W-scale TMEM layout,
    but the 2CTA result failed correctness with large, structured N-block
    mismatches. The same natural path with `NUM_CTAS=1` validated but was very
    slow (`0.185x` versus the 1CTA comparator), so natural 2CTA is now guarded
    with a static assertion in the scratch harness rather than left as an
    unsafe benchmark mode.
  - Artifacts:
    `/tmp/moe_bmm1_natural_bm256_smoke.csv`,
    `/tmp/moe_bmm1_natural_bm256_smoke2.csv`,
    `/tmp/moe_bmm1_natural_bm256_smoke3.csv`,
    `/tmp/moe_bmm1_natural_bm256_sub1_smoke.csv`, and
    `/tmp/moe_bmm1_natural_1cta_smoke.csv`.
- Added a correct `fakens:` scratch mode:
  - `NUM_CTAS=2`, transposed accumulator `[BLOCK_N, BLOCK_M]`, CGA-split N,
    but `TensorMemoryLayout(two_ctas=False)` so each CTA issues regular
    one-CTA MMA for its local N shard. X uses a degenerate `((0, 0),)` CGA
    layout so both CTAs see the full M tile, while W/W-scale remain N-local.
  - Correctness passed on uniform batch `896`, rank `4`.
  - First point was slow (`0.640x`), but tuning W5 recovered to only
    `~0.900x`; W6 remained around `0.62x-0.65x`, and 8-warp fake-N-split was
    not competitive.
  - Artifacts:
    `/tmp/moe_bmm1_fakens_smoke.csv`,
    `/tmp/moe_bmm1_fakens_smoke2.csv`,
    `/tmp/moe_bmm1_fakens_smoke3.csv`, and
    `/tmp/moe_bmm1_fakens_rank4_sweep.csv`. A post-guard smoke in
    `/tmp/moe_bmm1_fakens_postguard_smoke.csv` passed correctness and measured
    the best W5 fake-N-split point at `0.905x`.
- Rechecked several structural axes around the current near-miss:
  - `/tmp/moe_bmm1_splitws_rank4_recheck.csv`: split producer and `splitws`
    controls remained below parity; best was still around `0.966x`.
  - `/tmp/moe_bmm1_bk256_rank4_smoke.csv`: `BLOCK_K=256` is illegal for this
    gather operand because the layout would require a 256-byte swizzle.
  - `/tmp/moe_bmm1_warps2_rank4.csv`: two-warp variants are illegal because
    Gluon requires `num_warps >= 4`.
  - `/tmp/moe_bmm1_reg_ext_rank4.csv` and
    `/tmp/moe_bmm1_regs68_hard_ranks.csv`: wider register caps found small
    route-specific improvements (`MAXNREG=68`, `acc2/B24` helped ranks 4/5/7)
    but still did not reach parity, much less the `1.20x` target.
- Subagent W-local investigation concluded the existing true-2CTA W path
  already uses CGA shape-per-CTA shared storage for logical `BLOCK_N=256`.
  Physically changing the W descriptor/block shape to 128 would become a
  half-width logical MMA tile, not a true 256-column 2CTA kernel.

## 2026-04-20 Phase-Pair And Late-Stage Retuning

- Added a `phasepair:` scratch mode to
  `python/examples/gluon/06-moe-bmm1-structural-explore.py`. The intent was to
  exploit the earlier finding that full-only and spill-only subroutes are much
  closer to parity than the combined route by pairing adjacent full/spill
  transitions inside the persistent loop while preserving the proven W-reuse
  producer protocol.
- Correctness passed for uniform batch `896`, seed `0`, local rank `4`, but
  the hard-rank sweep showed the structure is a regression:
  `/tmp/moe_bmm1_phasepair_hard_ranks.csv`.
  Best `phasepair:` speedups were rank 3 `0.91569x`, rank 4 `0.95414x`,
  rank 5 `0.94106x`, and rank 7 `0.95437x`. The ordinary split baseline in
  the same run remained better, with best split speedups rank 3 `0.97521x`,
  rank 4 `1.00056x`, rank 5 `0.95167x`, and rank 7 `0.98931x`.
- Rechecked wider register caps and accumulator buffering around the current
  near-miss:
  - `/tmp/moe_bmm1_reg_ext_rank4.csv`: `MAXNREG=68` with
    `la48,lw40,mma40` gave a small rank-4 local improvement (`0.97416x`) but
    still did not reach parity.
  - `/tmp/moe_bmm1_regs68_hard_ranks.csv`: the best fixed-rank hard-route
    points stayed below parity: rank 3 `0.97647x`, rank 4 `0.97372x`, rank 5
    `0.96741x`, and rank 7 `0.98509x`.
  - `/tmp/moe_bmm1_regs68_reuse_fullsched_hard_ranks.csv`: `reuse` and
    `fullsched` did not convert the register-cap gain into a broad win. The
    best rank-7 point reached only `0.98981x`; ranks 3-5 remained below
    parity.
- Rechecked M64 alternatives with the same hard routes in
  `/tmp/moe_bmm1_m64_regs68_hard_ranks.csv`. The best rows were still the M32
  split baseline; tested M64 rows remained materially slower and are not a
  promising route to the `1.20x` target.
- Additional structural constraints confirmed during this slice:
  - `BLOCK_K=256` remains illegal for the gather operand because the implied
    shared-memory swizzle would need an unsupported 256-byte swizzle.
  - Two-warp variants remain illegal because Gluon requires
    `num_warps >= 4`.
- Decision: `phasepair:` is useful as a scratch diagnostic but not promotable.
  It did not solve the transition-overlap hypothesis and should not be used as
  selector policy. The next candidate needs to change the amount or ownership
  of resident work more fundamentally than transition pairing, late register
  retuning, `reuse`, `fullsched`, M64, BK256, or two-warp variants.

## 2026-04-20 X-Reuse, Scale-Prefetch, And Shape-Escape Probes

- Added scratch modes to isolate additional structural ideas in
  `python/examples/gluon/06-moe-bmm1-structural-explore.py`:
  - `xphaseprefetch:` uses the transition-aware X producer without the paired
    accumulator/MMA path from `phasepair:`.
  - `xpair:` creates an adjacent-N schedule for same `(slice_idx, pid_m)` work
    and attempts to reuse one gathered X tile for two N tiles. It needs a
    second W-scale TMEM descriptor because scale TMEM does not support batched
    multibuffering.
  - `xpairsched:` uses the same adjacent-N schedule but keeps normal per-block
    X loads and MMAs, isolating schedule effects from X-buffer reuse.
  - `scaleprefetch:` uses a second W-scale TMEM descriptor to pre-copy the next
    K tile's scale before waiting on the current X tile.
- `xpair:` is not currently a valid performance candidate. The first batched
  scale-TMEM version failed compilation with `Scales don't currently support
  multibuffering`; the separate-descriptor version and a post-wait local
  barrier variant both timed out under `timeout 180s-240s` on uniform batch
  `896`, rank `4`. Treat same-X/two-N MMAv5 reuse as unresolved/hazardous
  until a lower-level mbarrier/SASS investigation proves the protocol.
- Transition-only X prefetch validated but regressed:
  `/tmp/moe_bmm1_xphaseprefetch_hard_ranks.csv`. Best `xphaseprefetch:` rows
  were rank 3 `0.95619x`, rank 4 `0.95851x`, rank 5 `0.94794x`, and rank 7
  `0.97127x`, all below the ordinary split baseline in the same run.
- Adjacent-N schedule-only ordering validated but regressed:
  `/tmp/moe_bmm1_xpairsched_hard_ranks.csv`. Best `xpairsched:` rows were rank
  3 `0.92950x`, rank 4 `0.95234x`, rank 5 `0.92222x`, and rank 7 `0.93252x`.
- W-scale prefetching validated but regressed:
  `/tmp/moe_bmm1_scaleprefetch_hard_ranks.csv`. Best `scaleprefetch:` rows were
  rank 3 `0.94863x`, rank 4 `0.93510x`, rank 5 `0.93880x`, and rank 7
  `0.97041x`. Waiting on the next W tile before the current X wait likely
  reduces overlap instead of increasing it.
- Subagent multicast/off-axis sweep on GPU 1:
  `/tmp/moe_bmm1_2cta_uniform_bs896_seed0_ranks3457_multicast_offaxis_rep180.csv`.
  All `52` rows validated; disabling X multicast, W-scale multicast, or both
  did not reach parity. Best rows were rank 3 `0.97876x`, rank 4 `0.97856x`,
  rank 5 `0.97142x`, and rank 7 `0.98710x`.
- Subagent shape-escape sweep on GPU 1:
  `/tmp/moe_bmm1_shape_escape_bs896_seed0_r3_4_5_7_rep150.csv`. All ok rows
  passed validation; `24` rows errored consistently. M64 helped rank 5
  locally (`0.96248x`) but did not beat 1CTA and did not beat M32 on ranks 3,
  4, or 7. M128 and `BN512` probes were much slower; `BM48`, `BM80`, and
  `BM96` errored because shape element 0 must be a power of two, while one
  `BM64/BN512` variant exceeded shared memory.
- Decision: X transition prefetch, adjacent-N schedule-only ordering, same-X
  adjacent-N reuse, W-scale prefetch, multicast disabling, and another
  shape-escape sweep are not promotable. The hard slice-28 route still needs a
  different ownership model or a lower-level fix to increase eligible work
  without introducing the pair/schedule stalls measured here.

## 2026-04-20 Epilogue Ownership And Late Banded Retuning

- Added an `mmaepi:` scratch attempt that fuses the direct epilogue into the
  MMA partition to remove the separate accumulator handoff. The initial
  three-partition form and a four-partition no-op lead form were both rejected
  by `gl.warp_specialize` lowering in the rank-4 smoke
  `/tmp/moe_bmm1_mmaepi_smoke2.csv`. Treat this as a lowering/API constraint,
  not a measured performance result.
- Ran a direct epilogue/subtile sweep on the hard uniform batch `896` ranks in
  `/tmp/moe_bmm1_epilogue_subtile_hard_ranks.csv`:
  - rank 3 best: `0.97496x`
    (`B21/sub2`)
  - rank 4 best: `1.00004x`
    (`B24/acc2/regs68/epin0`)
  - rank 5 best: `0.98241x`
    (`B24/acc2/epin0`)
  - rank 7 best: `1.00160x`
    (`B24/acc2/epin0`)
- Expanded around `epin0`, bands, register caps, `acc2`, and X/W depth on
  separate rank groups:
  - `/tmp/moe_bmm1_epin0_band_regs_rank35.csv`: rank 3 still preferred the
    plain W6/B21 near-miss at `0.97529x`, while rank 5 reached a local small
    win (`1.00413x`) with `B22/acc2/regs68/epin0`.
  - `/tmp/moe_bmm1_epin0_band_regs_rank47.csv`: rank 4 best was only
    `0.97444x` and rank 7 best was `0.99675x` in that run. The earlier
    near-parity rank-4/7 points are not stable enough to drive selector policy.
- Decision: epilogue ownership and band/register retuning can move individual
  hard routes by a few percent, but they do not solve the hard rank-3 route and
  remain far below the `1.20x` requirement. `X_NUM_BUFS=6` is consistently
  disastrous in this family (`~0.69x-0.73x` on hard ranks); W5 variants also
  regress.

## Next Frontier

- Uniform slice `28` / batch `896` needs a structural change that increases
  resident or eligible warps without giving up W5 staging. Parameter sweeps
  that only reduce staging, change banding, or change 32-vs-64 row shape have
  not worked.
- Do not treat this frontier as closed. Continue iterating through harness
  improvements, structural occupancy/shared-memory hypotheses, source changes,
  correctness, benchmarks, notes, commits, and pushes until 2CTA beats 1CTA for
  every batch and further improvements are exhausted.
- Promising next directions:
  - Use the split diagnostic result to focus on mixed full/spill persistent
    loop behavior. First-only and spill-only subroutes are near parity or
    positive; the combined route loses. Layered host block ordering did not
    help, so the next attempt should target in-kernel overlap/ownership around
    the transition between full and spill blocks rather than simply reordering
    the host block schedule.
  - Continue from the four-warp `x5/w6/regs48` direct family for slice 28.
    W6 is the only tested knob that improved every fixed local rank; deeper W
    rings, W6+inline, W6+8warps, W6+occupancy changes, and simple schedule
    reorderings do not close the remaining gap.
  - Find a legal way to reduce 2CTA shared-memory footprint while preserving
    W5 depth, possibly by changing scale staging or descriptor/layout
    ownership rather than X/W buffer counts.
  - Preserve separate X and W producer ownership. Scratch kernels that fused
    both producers, moved W TMA into MMA, or moved gathered X TMA into MMA all
    passed correctness but regressed badly.
  - Treat pair-aware W reuse as a targeted M32 rewrite, not an M16 shortcut.
    M32 has only a few pairable spill blocks in the hard routes; M16 has more
    pairable work but is far below parity before any reuse rewrite.
  - Avoid separate W-scale barrier rings for this kernel. The split W-scale
    probe was legal but slower even at equal depth, so future shared-memory
    reductions need to preserve the single combined W+scale ready/empty
    protocol or change the descriptor/storage layout more fundamentally.
  - Inspect SASS/source counters for the dominant long-scoreboard locations in
    the direct 2CTA epilogue and loaders; NCU shows lower instruction count but
    much lower eligibility.
  - Investigate whether 2CTA can use a different epilogue ownership model that
    raises active warps without reintroducing the helper-store overhead.

## 2026-04-20 Structural Ownership And Direct-Scale Scratch

- Added more scratch-only structural modes and infrastructure in
  `python/examples/gluon/06-moe-bmm1-structural-explore.py`:
  - `ctampair:` mode `40` tries CTA-M pairing, where rank 0/1 own adjacent
    M blocks for the same N tile with local accumulator/direct stores.
  - `privsched:` mode `41` consumes a full-then-spill host schedule to isolate
    route class ordering from kernel codegen.
  - `mmascale:` mode `42` removes W-scale staging from the W producer and tries
    direct scale TMA from the MMA partition.
  - `dualmma:` mode `43` is a fresh two-lane interleave: paired X/W producers
    issue two adjacent logical blocks per K step and the MMA partition keeps
    two accumulators live.
- `dualmma:` validated on hard uniform batch `896`, ranks `3/4/5/7`, but
  regressed strongly in `/tmp/moe_bmm1_dualmma_uniform896_smoke.csv`. Explicit
  best speedups were rank 3 `0.90731x`, rank 4 `0.92271x`, rank 5 `0.90290x`,
  and rank 7 `0.94992x`. Larger `x6/w6` buffering was worse (`~0.67x-0.68x`).
  Deterministic two-lane interleave does not replace a true nonblocking
  wait-steal primitive.
- MMA wait/dependency/barrier variants did not help the same hard route:
  `/tmp/moe_bmm1_mma_barrier_family_uniform896_rep320.csv`. Explicit bests
  were rank 3 `0.97850x`, rank 4 `0.98376x`, rank 5 `0.96840x`, and rank 7
  `0.99279x`.
- `mmascale:` was partially fixed at compile time by allocating the direct
  scale-ready barrier with `two_ctas=mma_two_ctas` and using the standard
  W-scale multicast flag. It still launch-fails under
  `CUDA_LAUNCH_BLOCKING=1` in
  `/tmp/moe_bmm1_mmascale_multicast_blocking_smoke.csv`, so scale TMA from the
  MMA partition is classified unsafe until the protocol is redesigned.
- `ctampair:` remains invalid. The rank-3 smoke
  `/tmp/moe_bmm1_ctampair_recheck_smoke.csv` failed in the `gl.warp_specialize`
  lowering branch before a usable benchmark row. Earlier long `l4m1` attempts
  could hang, so do not spend broad GPU time on this path without an IR-level
  fix for CTA-rank/local-store control.
- Persistent launch-grid oversubscription was tested in
  `/tmp/moe_bmm1_launch_occ_uniform896_rep500.csv` and
  `/tmp/moe_bmm1_launch_occ_prod_r5_rep500.csv`. The inherited `occ2` remains
  best; `occ1`, `occ3`, and `occ4` all regress on hard uniform ranks and prod
  rank-5 rows.
- Direct BM64/BN256 without the helper-store path was tested in
  `/tmp/moe_bmm1_bm64_direct_uniform896_rep240.csv`; explicit bests stayed
  around `0.915x-0.918x` on ranks `3/4/5/7`. The earlier BM64 helper sweeps
  were killed after more than 35 minutes with no rows or output files.
- A selected-family large-batch prod sweep used `split:selected@...` so tuning
  started from the production BN512/helper configuration instead of the M32
  scratch base: `/tmp/moe_bmm1_selected_large_tune_prod_rep360.csv`. Best rows
  improved some points but still missed the `1.20x` bar:
  - rank 3: bs1536 `1.14704x`, bs2048 `1.17655x`, bs3072 `1.15245x`,
    bs4096 `1.12192x`.
  - rank 4: bs1536 `1.11188x`, bs2048 `1.10446x`, bs3072 `1.12488x`,
    bs4096 `1.18558x`.
  - `w6` exceeded shared memory on large rows, and explicit `regs72` or
    over-constrained register partitions frequently regressed.
- Current decision: the 20% criterion is still not met. The hard uniform
  slice-28 route is not responsive to ordinary staging, banding, launch-grid,
  M64, deterministic dual-block interleave, or wait-order tuning. The next
  useful frontier should either change CTA ownership with a correct rank/local
  control model, or reduce the true 2CTA shared-memory/scale footprint without
  moving TMA into the MMA partition.

## 2026-04-20 CTA-Pair Frontend Diagnosis And Scale-Ring Scratch

- Added parent `NUM_WARPS` plumbing through `PartitionArgs` so scratch
  descriptors can be configured consistently when parser candidates override
  `num_warps`.
- Subagent and local traceback analysis resolved the generic CTA-pair
  `gl.warp_specialize` frontend error: implicit accumulator TMEM loads call
  `blackwell.tensor_memory_descriptor.get_reg_layout()`, which requires a
  worker-local `num_warps >= 4`. The earlier `ctapair:`/`ctampair:` smoke used
  `l2m1`, so the accumulator-load owner had only one warp and failed before
  lowering.
- Reverted one experimental CTA-N pair layout to worker-local
  `gl.num_warps()` and fixed `load_inputs_cta_npair_partition` to use the same
  unsplit local row CGA layout as its descriptor. A 4-warp TMEM-owner N-pair
  retry then passed the immediate frontend/layout verifier but timed out with
  GPU 0 at full utilization:
  `/tmp/moe_bmm1_ctapair_npair_4warp_layoutfix_smoke.csv` was not flushed.
  Treat current CTA-pair modes `36/37/40/44` as unsafe without a deeper
  rank-local barrier/layout redesign.
- Independent subagent warps8 sweep on GPU 2:
  `/tmp/moe_bmm1_warps8_uniform896_gpu2_20260420T0710Z.csv`. All 13 candidates
  validated, none exceeded parity. The best row was only `0.69588x`
  (`mmamc @ l4m1, regs64, la64, lw48, mma48`), so broad 8-warp retuning is not
  a promising low-batch path.
- Added scratch mode `wscaleind:` (`STRUCTURAL_MODE=45`) with an independent
  W-scale shared-memory ring. The first version launch-failed because it
  recycled the scale buffer with a plain `mbarrier.arrive()` immediately after
  `tcgen05_copy`; changing the release to `blackwell.tcgen05_commit()` made the
  protocol validate.
- The independent W-scale ring is correct but too slow on hard uniform batch
  `896`, local rank `4`: `/tmp/moe_bmm1_wscaleind_tune_uniform896.csv`.
  `ws1/ws2/ws3/ws5` produced `0.40443x`, `0.65646x`, `0.79964x`, and
  `0.93028x`; the normal split row in the same run was `0.97669x`. Decoupling
  W-scale staging adds synchronization cost and does not solve occupancy.
- Existing pipeline and split-scale variants are also not promotable:
  `/tmp/moe_bmm1_pipe_modes_uniform896.csv`. `mmawpipe` was `0.40138x`
  (`l2m1`) and `0.23712x` (8-warp), `mmaxpipe` was `0.73786x` (`w1m1`) and
  `0.48122x` (8-warp), while `wscaleready`/`wscalefirst` were about
  `0.95x`.
- Current decision: no change here satisfies the `1.20x` criterion. Do not
  promote CTA-pair, independent scale rings, 8-warp near-parity retuning, or
  pipeline-owner variants into production selection. The next useful frontier
  should either change persistent work ownership more substantially or split
  route classes into separately tuned launches despite the launch-overhead
  risk.

## 2026-04-20 Route-Split And Remaining Parameter Escape Hatches

- Added scratch route-class scheduling support in
  `python/examples/gluon/06-moe-bmm1-structural-explore.py`: helper schedules
  can now isolate full M tiles from spill/tail M tiles, and `routesplit:`
  launches the same 2CTA config once for full tiles and once for spill tiles.
- `routesplit:` validated against the reference on hard uniform batch `896`,
  rank `4`, but `do_bench_cudagraph` cannot capture the two-launch wrapper
  (`operation failed due to a previous error during capture`). A non-graph
  fallback was added for this scratch mode only, and the measured walltime was
  not competitive: `/tmp/moe_bmm1_routesplit_wall_uniform896.csv` reported
  `0.30960 ms`, only `0.09019x` versus the normal 1CTA graph baseline. Treat
  Python-level two-launch route splitting as non-promotable unless it can be
  fused into one launch or captured reliably.
- BN512 direct low-batch escape remains poor:
  `/tmp/moe_bmm1_bn512_direct_uniform896.csv`. Best tested BN512 row was
  `0.78819x` (`x4,w3,regs52`); other BN512 points were `0.63x-0.67x`.
- M16 pair-reuse does recover some of the weak M16 baseline but remains far
  below parity: `/tmp/moe_bmm1_m16_pair_uniform896.csv`. Best tested M16 row
  was `0.84682x` (`pairsplit:x8w5`), so a from-scratch M16x2 rewrite would
  need a very large structural improvement before it could matter.
- Aggressive low-register retuning did not improve hard rank 4:
  `/tmp/moe_bmm1_lowregs_uniform896.csv`. `regs44` was essentially tied with
  the current split row (`0.97491x` vs `0.97530x` in that run), while
  `regs40/36` and explicit `la32,lw24,mma24` variants regressed.
- Deeper W staging with shallower X is not viable:
  `/tmp/moe_bmm1_wdeep_uniform896.csv`. `x4/w7`, `x3/w7`, `x4/w8`, and
  `x3/w8` landed at `0.58565x-0.66862x`; `x4/w6,regs52` and `x3/w6,regs52`
  were also below the current split row.
- Current decision: ordinary parameter escape hatches are exhausted for this
  slice. The remaining plausible single-launch frontier is a CLC-style
  persistent scheduler or a deeper rewrite that changes how multiple logical
  blocks are owned inside one CTA cluster without adding an uncaptured launch.

## 2026-04-20 CLC Combined Scratch And Fused-Epilogue Recheck

- Added scratch `clccombined:` mode (`STRUCTURAL_MODE=48`) in
  `python/examples/gluon/06-moe-bmm1-structural-explore.py`. It uses one CLC
  partition plus the combined input loader, MMA partition, and direct epilogue
  partition, keeping the CLC consumed barrier count at `3` workers.
- Initial CLC versions failed until two protocol issues were fixed:
  the CLC handoff now mirrors the working tutorial pattern by publishing a
  scalar canceled block id through shared memory, and the full-grid launch now
  uses the actual block-schedule count from `x_block_offs[-1]` instead of the
  host `n_blocks()` estimate. The estimate can be much larger than the compact
  MoE block schedule on small ragged slices, which over-launched clusters and
  caused out-of-range gather-index loads under Compute Sanitizer.
- `clccombined:` validates on small ragged uniform batch `128` and hard
  uniform batch `896`, local rank `4`. It is not graph-capturable in the
  current harness, so `bench_ms` uses the same non-graph fallback as
  `routesplit:` for this scratch mode.
- Hard-row timing is not competitive:
  `/tmp/moe_bmm1_clccombined_wall_uniform896_v2.csv`. The paired run reported
  `1cta = 0.02786 ms`, normal `splitws = 0.02960 ms` (`0.94126x`), and
  `clccombined = 0.18675 ms` (`0.14920x`). Treat this CLC shape as a
  diagnostic scheduler experiment, not a promotable 2CTA path.
- Rechecked the existing fused-MMA-epilogue scratch (`mmaepi:`). With the
  default one-warp MMA owner it fails frontend lowering because TMEM
  `acc_buf.load()` requires a worker-local power-of-two `num_warps >= 4`.
  Retrying with `warps8,l2m4` reaches PTXAS but fails register allocation:
  `/tmp/moe_bmm1_mmaepi_warps8_uniform896.csv` needs `80` registers with
  `maxnreg=52`; `/tmp/moe_bmm1_mmaepi_regs80_uniform896.csv` then needs `136`;
  `/tmp/moe_bmm1_mmaepi_regs160_uniform896.csv` then needs `255`. The current
  fused epilogue implementation is not a viable quick rescue without a much
  smaller epilogue/MMA live-range rewrite.
- Current decision: CLC and existing `mmaepi:` are both non-promotable. The
  next structural work should focus on a new ownership rewrite such as true
  N-shard/X-reuse or a tighter two-output fused body, not another scheduler-only
  wrapper.

## 2026-04-20 Shared-X Adjacent-N Scratch

- Added scratch `x2n_fused:` mode (`STRUCTURAL_MODE=49`) in
  `python/examples/gluon/06-moe-bmm1-structural-explore.py`. The structure
  pairs adjacent N tiles in one 2CTA cluster, gathers the activation tile once
  from rank 0 with TMA multicast, and lets each CTA own one local N tile while
  reusing the shared X tile.
- Barrier/layout findings: using two-CTA CGA layout on the X empty barrier
  fails verifier lowering for `tcgen05_commit` (`completion barrier cga_layout`
  expected local layout). Using two-CTA CGA layout on the X ready barrier fails
  `async_tma_gather` verifier lowering (`TMA barrier cga_layout` expected local
  layout). The standard local X ready barrier plus an X empty arrival count of
  `2` reaches PTXAS.
- Odd `GRID_N` tails are handled by inactive ranks that only consume/release
  X-ready stages, so the mode no longer requires an even N-grid. This is needed
  for the GPT-OSS-120B shape `N=5760`, `BLOCK_N=256`, where `GRID_N=23`.
- The fused epilogue version is not viable in its current form: a one-warp
  owner fails frontend lowering because TMEM accumulator loads require
  worker-local `num_warps >= 4`; the four-warp retry reaches PTXAS but follows
  the same register cliff as `mmaepi:`.
- The split-epilogue retry is also non-promotable. On hard uniform batch `896`,
  local rank `4`, candidate
  `x2n_fused:m32_bn256_sub1_direct_warps4_x5w6_b20_nomcscale_act2w1m1_regs52_epin1_b32@l1m1`
  needed `80` registers with `maxnreg=52`; raising to `regs80` needed `136`;
  raising to `regs160` needed `255`. No timing was collected because the mode
  does not compile at a practical register target.
- Current decision: shared-X adjacent-N reuse is structurally interesting but
  not promotable with the existing epilogue/MMA live ranges. The next attempt
  should either make the epilogue much narrower so shared-X can compile, or
  pursue a different local ownership split that avoids the extra live X/N-tail
  state.

## 2026-04-20 Shared-X Fragment-Epilogue And BN128 Probes

- Added scratch `x2n_mfrag:` mode (`STRUCTURAL_MODE=51`) in
  `python/examples/gluon/06-moe-bmm1-structural-explore.py`. It keeps the
  shared-X adjacent-N loader and compute path, but tries to load accumulator
  TMEM in M fragments (`sub4`) before bias/SwiGLU/store so the full
  accumulator tile is not live in the epilogue.
- The first inferred fragment load reached the TMEM verifier and printed legal
  fragment layouts, confirming the slice shape was recognized. Switching to
  the first printed layout still failed the `ttng.tmem_load` result verifier.
  The likely blocker is the local 2CTA accumulator CGA layout: slicing the last
  TMEM dimension produces a degenerate local block basis (`block = [[0, 0]]`)
  that the load verifier does not accept for this path.
- A variant that allocated `x2n_mfrag` accumulator TMEM with no CGA layout was
  rejected earlier in TTIR verification because the 2CTA warp-specialize
  context requires a 2-CTA-per-CGA memory layout. That rules out simply
  dropping the local CGA layout to make M-fragment slices legal.
- A full-tile explicit TMEM-load scratch was also tested internally and then
  left unexposed as a parser candidate: its first local layout aborts in layout
  construction because the full `[256, 32]` layout is not surjective. Do not use
  full-tile explicit TMEM load as the next path.
- Subagent-proposed `BN128` shared-X adjacent-N was tested by running
  `x2n_fused:` with `@bn128`. It fails before PTXAS:
  `tcgen05_mma_scaled does not support blockM=64`. The current scaled-MMA
  operand/layout path therefore cannot use `BLOCK_N=128` as the quick way to
  halve the per-CTA epilogue live range.
- Smaller-M shared-X did not rescue compilation. With `BLOCK_M=16`, `regs80`
  still needed `136` registers and `regs128` escalated to `232`, so M-size-only
  tuning does not avoid the epilogue register cliff.
- Current decision: M-fragment epilogue is still the most promising way to
  attack the register cliff, but it needs a layout-level fix for local 2CTA
  TMEM slicing or a different accumulator orientation. `BN128` and simple M16
  retuning are non-promotable with the current scaled-MMA path.

## 2026-04-20 Selector Refresh And Additional X2N Structures

- Refreshed the hard uniform selector surface in
  `/tmp/moe_bmm1_selector_refresh_uniform_hardranks.csv` for ranks `3/4/5/7`,
  batches `128..2048`, rep `120`. Best available candidates were still far
  short of the requested `1.20x` target: best min/max/avg speedup was
  `0.99698 / 1.11351 / 1.03483`; selected min/max/avg was
  `0.99620 / 1.09991 / 1.01755`. Worst hard rows remain around uniform rank
  `4/5` at batch `896`.
- Refreshed prod-routing ranks `0/1/2/3` in
  `/tmp/moe_bmm1_selector_refresh_prod_ranks0123.csv`. Prod routing has more
  exploitable skew: best min/max/avg was `0.99439 / 1.27139 / 1.08910`, with
  several `>=1.15x` rows at batches `512/768/896/1536/2048`, but selector
  regressions remain and uniform hard rows still dominate the all-batch
  criterion.
- Focused slice-28 W-buffer/band/register retunes for uniform ranks `3/4/5/7`
  in `/tmp/moe_bmm1_slice28_tune_rank34.csv` and
  `/tmp/moe_bmm1_slice28_tune_rank57.csv` did not find a parameter-only rescue.
  Best explicit rank-4 rows stayed below parity; rank-7 had one small
  `1.00584x` explicit split row. Current conclusion: the hard uniform rows
  need a structural change, not another narrow selector tweak.
- Added scratch natural-adjacent-N mode `x2n_natfrag:` (`STRUCTURAL_MODE=52`)
  to test a local natural `[M, N]` accumulator and N-fragment epilogue. It is
  blocked by Blackwell scaled-MMA shape constraints for this workload:
  `BLOCK_M=32` fails because natural scaled MMA requires `blockM` at least
  `64/128`, and `BLOCK_M=64` is also rejected on the current frontend path.
  Do not spend more time on natural local `[M, N]` unless the MMA operand shape
  can be made genuinely `128` rows without wasting the small-M workload.
- Added scratch helper-handoff mode `x2n_helper:` (`STRUCTURAL_MODE=53`) to
  decouple SwiGLU/packing from global stores. The first failure was a
  worker-local warp-count issue: `acc_buf.load()` requires the handoff worker
  to run with at least four warps, so candidates need `w4m1` or an explicit
  handoff-warp knob. With four handoff warps, the full-tile helper follows the
  same register cliff as fused epilogues: `regs52 -> needs 80`,
  `regs80 -> needs 136`, `regs136 -> needs 248`.
- Retried `x2n_helper:` with M-fragment handoff so only one row fragment is
  live before passing it to the store worker. This reaches the TMEM subslice
  verifier but fails on the local 2CTA accumulator layout: the inferred load
  has a degenerate local block basis (`block = [[0, 0]]`). Allocating the
  accumulator without CGA layout is rejected because the two-CTA
  warp-specialize context requires two CTAs per CGA. This confirms the earlier
  `x2n_mfrag:` blocker applies to helper handoff too.
- Fixed the local adjacent-N TMEM block calculation in the scratch harness so
  x2n modes use `BLOCK_N` rather than `BLOCK_N // num_ctas` for their local
  accumulator instruction tile. This opens `BN128` past the previous
  `blockM=64` assertion, but the direct BN128 x2n epilogue still fails register
  allocation: `regs52 -> needs 80`, `regs80 -> needs 136`,
  `regs136 -> needs 248`, and even `regs255` asks for `255+`.
- Added scratch fused-MMA-epilogue adjacent-N mode `x2n_mmaepi:`
  (`STRUCTURAL_MODE=54`) to run the direct epilogue inside the x2n MMA worker
  with four MMA warps. It required host-side local-N descriptor/schedule
  classification like the other x2n modes; after that fix it reaches PTXAS but
  has the same `80 -> 136` register escalation. `M16/BN128` variants did not
  finish compilation under the smoke timeouts, so they are not promotable for
  the fast loop.
- Current decision: the adjacent-N/X-reuse family is still attractive at the
  algorithmic level, but every version that needs a full accumulator load is
  non-promotable due register pressure, while M-fragment loads are blocked by
  local 2CTA TMEM subslice verification. Next structural directions should
  avoid local 2CTA TMEM subslices and full-tile epilogue liveness: e.g. route
  two independent single-CTA tiles inside one 2CTA launch with a cheaper
  scheduler, or find a legal true-2CTA accumulator orientation whose last
  dimension is the epilogue fragment dimension.

## 2026-04-20 TMEM View-Chain Patch And Aggressive Structural Probes

- Updated the compiler scratch support for TMEM descriptor view chains because
  the adjacent-N fragment experiments need `TMEMLoadOp`/`TMEMStoreOp` operands
  that are no longer direct `ttng.tmem_alloc` results. `PartitionScheduling`
  now walks through `ttg.memdesc_index`, `ttg.memdesc_reinterpret`,
  `ttg.memdesc_reshape`/`trans`, and `ttng.tmem_subslice` to recover the
  underlying TMEM allocation. `InterleaveTMem` treats `memdesc_reinterpret` as
  a pure view, and `TensorMemoryAllocation` extends liveness through
  `ttng.tmem_subslice`.
- Added lit coverage for the view-chain fixes:
  `test/TritonGPU/partition-scheduling.mlir` has a
  `memdesc_index -> memdesc_reinterpret -> tmem_subslice -> tmem_load` path
  inside `ttg.warp_specialize`; `test/TritonNvidiaGPU/interleave_tmem.mlir`
  keeps interleaving through `memdesc_reinterpret`; and
  `test/TritonNvidiaGPU/test_tensor_memory_allocation.mlir` checks that a
  store through a reinterpreted subslice keeps the base allocation live so a
  later allocation cannot reuse the same TMEM columns.
- Validation so far:
  - `make`
  - `python3 -m py_compile python/examples/gluon/06-moe-bmm1-structural-explore.py`
  - `git diff --check`
  - `cd build/cmake.linux-aarch64-cpython-3.12 && ninja triton-opt && lit -v test/TritonNvidiaGPU/test_tensor_memory_allocation.mlir`
  - `cd build/cmake.linux-aarch64-cpython-3.12 && ninja triton-opt && lit -v test/TritonNvidiaGPU/interleave_tmem.mlir`
  - `cd build/cmake.linux-aarch64-cpython-3.12 && ninja triton-opt && lit -v test/TritonGPU/partition-scheduling.mlir`
- The compiler patch makes the permuted-fragment TMEM view chain legal enough
  to reach runtime, but it does not make the current `x2n_permfrag:` kernel
  promotable. Hard-row isolated probes still timeout or hang before producing
  a valid CSV row:
  `/tmp/moe_bmm1_structural_hardrow_x2n_permfrag_20260420.log`.
- Retried `x2n_fused:` with `@bn128,l1m1,regs80,sr80` after the local-N
  descriptor fix. The inline validation command on GPU 0 did not complete
  within the 240s timeout, so the BN128 shared-X path remains a non-promotable
  compile/hang candidate until it can validate quickly.
- `ctalinear:` remains a useful control for "two independent 1CTA-like lanes
  inside one 2CTA cluster", but the current direct M32 attempts hit a register
  target cliff (`regs52` asks for `72/80+` depending options) and the BN128
  retest is running under the GPU1 benchmark agent. Do not promote it without
  a same-input hard-row timing win.
- Fresh subagent review prioritized the next structural work as:
  pre-fragmented adjacent-N accumulators that avoid `tmem_subslice` of a local
  2CTA full accumulator, then a lane-private two-block 2CTA launch, then
  single-launch route-class work stealing, then transition-only W reuse. The
  shared theme is to avoid another broad buffer/register sweep; the remaining
  hard row needs changed ownership or changed accumulator storage.
- Same-input hard-row benchmark refreshes after the view-chain patch still do
  not produce a promotable 2CTA path:
  - GPU1 isolated artifact
    `/tmp/moe_bmm1_structural_20260420_gpu1_091430_2441867_isolated.log`:
    `1cta = 0.02782461 ms`, `selected = 0.02783320 ms`, explicit split
    `b21 = 0.02896984 ms` (`0.96047x`), explicit split `b24_acc2 =
    0.02905904 ms` (`0.95752x`). `ctalinear @bn128,l1m1,regs80,sr80`,
    `ctalinear @bn128,l2m1,regs80,sr80`, and `x2n_fused
    @bn128,l1m1,regs80,sr80` all hung in `torch.cuda.synchronize()` during
    validation and were killed with exit status `143`.
  - GPU2 transition-mode artifact
    `/tmp/moe_bmm1_transition_modes_hardrow_20260420.log`:
    `1cta = 0.02805212 ms`, explicit split `b21 = 0.02891449 ms`
    (`0.97018x`), `phasepair @acc2 = 0.03070218 ms` (`0.91369x`), and
    `xphaseprefetch = 0.02963246 ms` (`0.94667x`). The following `xpair`
    validation hung in `torch.cuda.synchronize()` and the process was killed.
  - GPU3 cta-linear tune artifact
    `/tmp/moe_bmm1_ctalinear_tune_hardrow_20260420.log`: baseline
    `1cta = 0.02796241 ms`; the first `ctalinear @bn128,warps8,l1m1`
    candidate hung in `torch.cuda.synchronize()` and the process was killed.
- Current decision: keep the compiler view-chain support because it is a real
  correctness gap for TMEM views, but do not spend more fast-loop time on the
  current `ctalinear` or BN128 adjacent-N shapes until their synchronization
  hang is diagnosed. The next structural attempt should change accumulator
  storage or scheduling more deeply rather than raising register targets.

## 2026-04-20 Continuous Tuning Plan And Latest Hard-Row Evidence

- Active success criterion: every batch size must have a 2CTA Gluon path at
  least `1.20x` faster than the same-input 1CTA comparator before 2CTA can be
  treated as solved or enabled for all sizes. Any row below `1.20x`, especially
  uniform hard rows at batch `896`, remains unfinished optimization work.
- Durable execution rule: do not voluntarily stop the turn while there are
  plausible 2CTA tuning paths, uncommitted scratch-harness improvements,
  benchmark runs in flight, or unrecorded benchmark/profiler results. If an
  external limit forces handoff, record the branch, artifacts, active command,
  current best rows, and exact next experiment here first.
- Stable GPU2 same-input sweep artifact:
  `/tmp/moe_bmm1_stable_sweep_gpu2_20260420T0925Z.csv`.
  Best speedups were still far below target:
  `256=1.01779x`, `512=1.06618x`, `896=1.00266x`,
  `1024=1.07205x`, `1536=1.01493x`, `2048=1.08765x`.
- Broad structural small-batch artifact:
  `/tmp/moe_bmm1_structural_small_gpu2_20260420.csv`.
  Best rows were `256=1.02926x`, `512=1.06192x`, and
  `1024=1.07574x`, all from the explicit split `b24_acc2` family.
- Broad structural hard-row artifact:
  `/tmp/moe_bmm1_structural_896_gpu0_20260420.csv`.
  The best measured row was `selected=1.00186x`; split, combined, multicast,
  counted-barrier, and prefetch structural variants were all parity or slower.
  This is the current bottleneck and needs a structural change or a much wider
  schedule/shape discovery than the previous narrow sweeps.
- Broad structural large-batch artifact:
  `/tmp/moe_bmm1_structural_large_gpu3_20260420.csv`.
  Best rows were `1536=1.04055x` from `splitws:selected` and
  `2048=1.11257x` from `mmaxfirst:selected`. Large batches have useful 2CTA
  headroom but still do not meet the `1.20x` criterion.
- `ctapair`/`ctampair` diagnosis: fused-epilogue variants that run
  `apply_bias_and_scale()` inside the MMA worker cannot use `MMA_WARPS=1`
  because `acc_buf.load()` needs a power-of-two worker warp count `>=4`.
  Split-epilogue `ctapair2`/`ctampair2` variants need
  `STORE_HELPER_WARPS>=4` for the same reason. The scratch harness now has
  local static asserts and parser defaults for those legality constraints.
  Legal `@warps8` shapes currently hit PTXAS register cliffs (`136` registers
  or worse), so this family needs register-lifetime reduction before promotion.
- `x2n` ready-barrier experiment: setting the shared-X ready mbarrier to
  `two_ctas=True` is invalid for TMA multicast. The compiler rejects
  `ttng.async_tma_gather` because TMA expects a CGA barrier layout `[[1]]`;
  `two_ctas=True` produces `[[0]]`. The hang in current BN128 x2n probes is
  therefore not fixed by changing the ready-barrier allocation.
- Active tuning plan:
  1. Run standard-family hard-row sweeps for batch `896` over split,
     splitws, counted barriers, multicast/control variants, `BLOCK_M`
     alternatives, band values, buffer counts, and register targets while
     avoiding known hang-prone x2n/ctalinear shapes.
  2. Run larger-batch sweeps for `1536/2048` around `splitws:selected`,
     `combinedw:selected`, `mmaxfirst:selected`, `mmamc:selected`,
     `mmacount:selected`, and barrier-order variants because those are the
     only families showing nontrivial headroom so far.
  3. Keep `ctapair`/`ctampair` as structural probes only after the legal
     warp-count patch; next useful work there is reducing full-epilogue
     accumulator liveness, not simply raising `MAXNREG`.
  4. Revisit adjacent-N shared-X only through a design that avoids full-tile
     TMEM epilogue loads or fixes the local 2CTA TMEM fragment layout; do not
     retry the invalid `two_ctas=True` ready-barrier path.
  5. Promote a candidate only after same-input validation, paired timing
     against 1CTA, and a broader batch sweep. Rows under `1.20x` stay in the
     active queue.

## 2026-04-20 Follow-Up Forced-2CTA Sweeps

- GPU1 subagent artifact `/tmp/moe_bmm1_bs896_20260420T093204Z.csv` found a
  best `1.03668x` row from
  `mmaxfirst:...@bm32,b21,x5,w6,acc1,regs48,direct`, but that run used uniform
  `local_rank=0`. Treat it as directional only; it is not the known hard
  uniform rank-4 row.
- GPU2 hard-rank forced-2CTA artifact
  `/tmp/moe_bmm1_gpu2_hard896_forced2cta_20260420T093246Z.csv` checked the
  same region on uniform `local_rank=4`. The best explicit 2CTA row was still
  slower than 1CTA: `mmacount:m32_bn256...b21` at `0.97361x`, followed by
  `mmabarw=0.97308x`, `mmabarx=0.97291x`, `mmamc=0.97176x`,
  and `mmaxfirst=0.96923x`. Standard forced 2CTA remains non-promotable for
  batch `896`.
- The helper-store epilogue proposal was tested through the original split
  kernel on the hard row. `sub4,epi2,sh1,sr32` reached only `0.93990x`; the
  `sub2` helper path was much worse at `0.66477x`. Helper ownership alone does
  not rescue slice `28`.
- GPU0 `BN512` hard-rank artifact
  `/tmp/moe_bmm1_gpu0_hard896_bn512_20260420T093357Z.csv` ruled out the wider
  N tile for this row. Normal 4-warp `BN512` variants were about `0.68x`, and
  the best 8-warp epilogue variant was only `0.76423x`. The extra N width
  reduces tile count but creates too much per-tile work/live state for the
  small-M hard row.
- GPU3 larger-batch artifact `/tmp/moe_bmm1_focused_20260420T093329Z.csv`
  improved the current large-batch frontier but did not reach target:
  `1536` best was `mmaxfirst:selected@regs60` at `1.04268x`;
  `2048` best was `splitws:selected@sh1,sr32` at `1.12241x`.
- Scratch harness updates in progress:
  - CSV rows are now flushed after every candidate so timeout-prone structural
    probes preserve completed measurements.
  - `ctapair`/`ctampair` mode branches now assert that the worker loading TMEM
    has at least four warps, and parser defaults for those families use legal
    8-warp/4-helper-warp shapes.
  - A post-ready-wait `gl.barrier()` convergence experiment was temporarily
    tested in the lane-private `ctalinear` compute path to check the mbarrier
    stale-epoch deadlock hypothesis. The first BN128 `ctalinear` smoke still
    stalled after the 1CTA row, so the scratch barrier change was reverted from
    active code and kept only as a dead-end note.

## 2026-04-20 Hard-Row Profiling And Dead-End Structural Checks

- NCU reports:
  `/tmp/moe_bmm1_ncu_hard896_1cta_20260420T093822Z.ncu-rep` and
  `/tmp/moe_bmm1_ncu_hard896_2cta_mmacount_20260420T093837Z.ncu-rep`.
  The profiler timings are inflated by collection overhead, but the relative
  counters are useful:
  - 1CTA hard row: duration `38.37 us`, compute throughput `54.58%`,
    memory throughput `49.34%`, issue slots busy `33.09%`, achieved occupancy
    `48.35%`, eligible warps/scheduler `0.75`, active warps/scheduler `7.72`,
    registers/thread `64`, dynamic shared memory/block `114.43 KiB`.
  - best forced 2CTA (`mmacount b21`): duration `39.84 us`, compute throughput
    `50.89%`, memory throughput `48.51%`, issue slots busy `22.70%`, achieved
    occupancy `24.11%`, eligible warps/scheduler `0.40`, active
    warps/scheduler `3.84`, registers/thread `48`, dynamic shared
    memory/block `113.88 KiB`.
  - Interpretation: the hard-row 2CTA path does less instruction work but gives
    up about half the active/eligible warp pool under the two-CTA cluster
    launch. More bandwidth is not the missing ingredient; the current 2CTA
    shape is scheduler/occupancy limited and does not create enough useful
    work per cluster to pay for that loss.
- GPU0 structural artifacts after the NCU pass:
  - `/tmp/moe_bmm1_gpu0_hard896_structural2_20260420T093736Z.csv`:
    `dualmma` was slower (`best 0.91498x`) and `mmascale` triggered an
    unspecified launch failure. Do not include `mmascale` in broad sweeps until
    isolated under `CUDA_LAUNCH_BLOCKING=1`.
  - `/tmp/moe_bmm1_gpu0_hard896_structural3_20260420T093753Z.csv`:
    `privsched` peaked at `0.94990x`, `pairsplit` at `0.94871x`,
    `xphaseprefetch` at `0.95239x`, and fake-N variants were either a lowering
    failure or very slow (`fakenspair=0.40067x`, `fakensmmaepi=0.24258x`).
  - `/tmp/moe_bmm1_gpu0_hard896_warps8_20260420T093927Z.csv`:
    8-warp `BN256` forced-2CTA variants were much worse (`~0.68x`). The
    scheduler issue is not fixed by simply adding more warps to each CTA.
  - `/tmp/moe_bmm1_gpu0_hard896_occ_20260420T093951Z.csv`:
    persistent oversubscription above the current `occ2` regressed. `occ1`
    underfilled badly (`~0.75x`); `occ3/occ4/occ5` were also slower.
  - `/tmp/moe_bmm1_gpu0_hard896_bk_20260420T094013Z.csv`:
    `BK64` violates the MX scale block-shape constraint, and `BK256` fails
    layout assertions. `BLOCK_K` is not a viable quick lever in the current
    descriptor path.
  - `/tmp/moe_bmm1_gpu0_hard896_ctapair2_20260420T094028Z.csv`:
    after the legal warp-count parser defaults, the first split-epilogue
    `ctapair2` smoke still stalled during validation after the 1CTA row. This
    confirms the pair-family blocker is a synchronization/register-ownership
    problem beyond the original low-warp frontend error.
- Updated active hypothesis: to beat 1CTA by `1.20x` at batch `896`, a 2CTA
  kernel probably needs either a legal local-2CTA fragment epilogue that keeps
  occupancy near 1CTA while reducing epilogue live state, or a fundamentally
  different work mapping that creates more useful independent work per
  two-CTA cluster. Wider tiles, more warps, more persistent programs, helper
  store handoff, and existing route/pair/dual-block modes do not solve the
  hard row.

## 2026-04-20 Structural x2n4 Helper And Hard-Row Tuning

- Subagent reviews converged on the same diagnosis: the current hard row is
  scheduler/occupancy limited, not bandwidth limited. The recommended high-risk
  structural branch was an `x2n4` shared-X, four-N-tile supergroup that amortizes
  the gathered activation tile across two adjacent N tiles per CTA rank, with a
  helper-store fallback to avoid the direct epilogue's 200+ register pressure.
- Implemented scratch `STRUCTURAL_MODE == 57` in
  `python/examples/gluon/06-moe-bmm1-structural-explore.py`:
  - `load_inputs_x2n4_partition` schedules `ceil(GRID_N / 4)` N groups,
    multicasts X from rank 0, and loads up to two W/W-scale tiles per CTA rank.
  - `mma_x2n4_compute_partition` computes up to two accumulators per persistent
    block using the shared X tile and alternate W-scale TMEM for the second N
    tile.
  - `epilogue_x2n4_partition` keeps the direct-store path available for compile
    pressure experiments.
  - `epilogue_x2n4_handoff_partition` and
    `epilogue_x2n4_store_partition` add a helper-store path selected with
    `x2n4:...@helper`.
- x2n4 validation outcomes:
  - Direct `x2n4` is not currently viable: the direct epilogue still triggers
    ptxas register allocation failures around `200-255` registers in earlier
    smokes.
  - Non-wide helper `sub4` with a 4-warp handoff reaches TMEM lowering but
    fails the verifier at the sliced accumulator load:
    `/tmp/moe_bmm1_x2n4_helper_sub4_w4_20260420T1015Z.csv`.
  - Wide helper `sub2` initially exceeded shared memory by `233736 > 232448`
    bytes for `x5w6`: `/tmp/moe_bmm1_x2n4_helper_wide_w4_20260420T1015Z.csv`.
  - Reducing to `x4w5` or even `x4w4` fits resources and launches, but hangs
    until timeout during validation:
    `/tmp/moe_bmm1_x2n4_helper_wide_x4w5_20260420T1021Z.csv` and
    `/tmp/moe_bmm1_x2n4_helper_wide_x4w4_20260420T1028Z.csv`.
    Increasing helper depth to `epi4` also hangs:
    `/tmp/moe_bmm1_x2n4_helper_wide_x4w4_epi4_20260420T1038Z.csv`.
  - Conclusion: x2n4 now has useful scaffolding, but the helper path has a real
    synchronization/order bug and the non-wide path is blocked by TMEM
    sub-slice layout verification. Do not promote x2n4 until a single
    validation pass completes.
- Permuted-fragment epilogue follow-up:
  - Replaced the explicit `acc_sub.get_reg_layout(instr_variant="32x32b")`
    call inside the static fragment loop with implicit `acc_sub.load()` to avoid
    frontend layout extraction from an indexed reinterpret view.
  - The plain-load probe still timed out before writing a row:
    `/tmp/moe_bmm1_permfrag_plainload_20260420T1048Z.log`. Treat
    `x2n_permfrag` as compile/lowering blocked.
- Hard-row tuning results against `batch=896`, uniform routing, `local_rank=3`,
  seed `0`:
  - Direct fragmented split epilogue is the closest compiling family but still
    below parity. `/tmp/moe_bmm1_hard896_subfrag_direct_20260420T1030Z.csv`
    measured `split@sub4,regs56` at `0.97960x` versus 1CTA.
  - Expanded sub4/band/staging sweep
    `/tmp/moe_bmm1_hard896_split_sub4_tune_20260420T1033Z.csv` found
    `fullsched` best at only `0.97587x`; `b16/b20/b21/b24/b32` and
    `x4w5/x5w5/x6w5` all stayed below parity.
  - `combined` inline/nomc variants in
    `/tmp/moe_bmm1_hard896_occ_tune2_20260420T1016Z.csv` were about
    `0.912-0.913x`. Raising x2n_fused register caps made ptxas demand
    `120-136` registers, so that branch is not a low-reg rescue.
  - `BM16` worsened the hard row (`~0.76x` split, `~0.73x` combined) in
    `/tmp/moe_bmm1_hard896_m16_20260420T1051Z.csv`.
  - `BN128` is rejected by the existing 2CTA width assertion, `BN384` is not a
    legal descriptor shape, and earlier `BN512` measurements were much slower.
  - `ctalinear` wants at least `88` registers and then balloons to `168` when
    allowed more cap; do not prioritize it without a lower-register rewrite.
  - `ctampair` with the intended `w1m4` split hangs after the 1CTA row,
    matching the earlier M-pair synchronization blocker.
  - `routesplit` is dominated by two-kernel overhead (`~0.07x`) in
    `/tmp/moe_bmm1_hard896_routesplit_20260420T1058Z.csv`.
- Current frontier: no candidate in this slice met parity, let alone the
  `1.20x` success gate. The least-bad measured hard-row candidate remains a
  direct split-epilogue `sub4`/`fullsched` shape around `0.976-0.980x`. The next
  unblocked work is either fixing x2n4 helper synchronization or building a new
  low-register M-pair/fragment epilogue that avoids sliced-TMEM verifier
  failures.

## 2026-04-20 Follow-Up x2n4 And M-Pair Fragment Attempts

- Followed up on the x2n4 helper hang:
  - Removed a `gl.barrier()` inside `mma_x2n4_compute_partition`; a CTA-wide
    barrier inside only the MMA warp-specialized partition is unsafe.
  - Added descriptor-aware X-empty commits for x2n4 and a second X-empty commit
    when the second local N tile is active, because each CTA rank issues two
    MMAs against the same multicast X buffer.
  - These changes did not resolve the hang. The resource-fitting wide helper
    `x4w4` still timed out:
    `/tmp/moe_bmm1_x2n4_helper_wide_x4w4_nobar_20260420T1105Z.csv`,
    `/tmp/moe_bmm1_x2n4_helper_wide_x4w4_desc_20260420T1127Z.csv`, and
    `/tmp/moe_bmm1_x2n4_helper_wide_x4w4_doublex_20260420T1134Z.csv`.
  - Current interpretation: x2n4 has at least one remaining producer/consumer
    mismatch beyond the obvious CTA barrier and X-empty arrival count. Do not
    spend more benchmark time on x2n4 until the synchronization protocol is
    simplified or instrumented.
- Probed the separated M-pair family:
  - Existing `ctampair2` with `sub4` does not immediately hang, but ptxas wants
    `72` registers at a `48` cap:
    `/tmp/moe_bmm1_ctampair2_sub4_smoke_20260420T1144Z.csv`.
  - Raising the cap to `72` made ptxas request `120` registers, matching the
    pattern seen in other full-tile epilogues:
    `/tmp/moe_bmm1_ctampair2_sub4_regs72_20260420T1146Z.csv`.
- Added scratch `ctampair_permfrag` (`STRUCTURAL_MODE == 58`) to combine
  M-pair load/compute with a direct N-fragment epilogue modeled on
  `x2n_permfrag`. This avoids M-sliced TMEM descriptors and processes
  `BLOCK_N / SWIGLU_SUBTILE_FACTOR` columns at a time.
  - First smoke exposed a launch-side descriptor classification miss for mode
    58; fixed by treating it as an M-pair mode in the Python launch wrapper.
  - After that fix, `ctampair_permfrag` compiled and launched but hung until
    timeout:
    `/tmp/moe_bmm1_ctampair_permfrag_smoke2_20260420T1201Z.csv`.
  - Conclusion: M-pair scheduling still has a runtime synchronization issue
    once it reaches executable code. A useful next M-pair attempt should first
    reduce to a minimal no-epilogue or single-K smoke to isolate load/compute
    barriers before adding fragment epilogues.
- Existing x2n fragment paths were rechecked with legal epilogue warp counts:
  - `x2n_mfrag` with `sh4/sub4/fullsched` fails the sliced-TMEM verifier at
    `acc_sub.load().permute((1, 0))`; no-catch output confirms the same
    `ttng.tmem_load` layout class seen in other M-fragment attempts.
  - `x2n_permfrag` with `sh4/sub4/fullsched` hangs during validation before
    writing a row. Artifact:
    `/tmp/moe_bmm1_x2n_frag_existing_sh4_20260420T1215Z.csv`.
  - Conclusion: neither existing fragment mode is ready to tune; the next
    fragment attempt needs either a compiler/layout fix or a new epilogue that
    avoids TMEM subviews entirely.

## 2026-04-20 M-Pair Drain Isolation

- Added scratch `ctampair_drain` (`STRUCTURAL_MODE == 59`) to isolate M-pair
  synchronization. It uses `load_inputs_cta_mpair_partition` and
  `mma_cta_mpair_compute_partition`, then an epilogue worker only waits on
  `acc_ready` and arrives `acc_empty` without reading TMEM or writing output.
- Smoke command intentionally omitted `--validate` because the mode writes no
  output:
  `/tmp/moe_bmm1_ctampair_drain_smoke_20260420T1225Z.csv`.
- Result: the drain mode still hung until timeout. This means the M-pair
  blocker is in load/compute/accumulator synchronization, not in the SwiGLU
  epilogue, global store, or fragment layout. Next M-pair work should reduce
  further, for example one K tile, one buffer, or explicit debug counters around
  X/W ready-empty phases.
