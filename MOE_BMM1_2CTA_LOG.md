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
- Decision: do not promote a slice-28 selector change yet. W6/regs48 is the
  new best near-miss family and should be the baseline for future slice-28
  work, but it still loses to 1CTA on hard uniform fixed-rank routes.

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
  - Avoid separate W-scale barrier rings for this kernel. The split W-scale
    probe was legal but slower even at equal depth, so future shared-memory
    reductions need to preserve the single combined W+scale ready/empty
    protocol or change the descriptor/storage layout more fundamentally.
  - Inspect SASS/source counters for the dominant long-scoreboard locations in
    the direct 2CTA epilogue and loaders; NCU shows lower instruction count but
    much lower eligibility.
  - Investigate whether 2CTA can use a different epilogue ownership model that
    raises active warps without reintroducing the helper-store overhead.
