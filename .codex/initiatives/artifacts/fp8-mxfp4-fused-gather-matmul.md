---
owner: root@codex-kernel-devbox-0.brix.jeffniu.svc.cluster.local
created: 2026-04-06T23:18:36Z
updated: 2026-04-09T04:30:39Z
---

# FP8 x MXFP4 Fused-Gather Matmul Optimization

## Rationale

Optimize `python/triton_kernels/triton_kernels/matmul.py` for the GB300 target workload that uses fp8 activations, mxfp4 weights, and fused gather. This workstream needs durable notes because the tuning loop will span multiple sessions, require direct measurement on this machine, and be handed back and forth between the user's laptop and this checkout.

## Invariants

- Keep the public `triton_kernels.matmul` behavior stable unless we explicitly agree to an API change.
- Preserve correctness for the target fp8 x mxfp4 fused-gather cases before trading for speed.
- Record benchmark conditions, observations, and performance deltas in persistent memory outside the repo so multi-week tuning stays reproducible.
- Preserve handoff clarity: the user syncs work by pushing into and fetching from this machine's repo; there is no access to the user's laptop.

## Current Understanding

The current dispatch path is `python/triton_kernels/triton_kernels/matmul.py`, which canonicalizes inputs, decides whether TMA/persistent execution is legal, computes `opt_flags`, and launches either `_matmul` or `_p_matmul`. For the target workload on Blackwell-class GPUs, the persistent kernel path is the likely focus because it already supports TMA-based operand handling and the heuristics bias toward persistent execution for low-precision LHS inputs.

The exact workload shape family is now grounded by `/root/code/openai/lib/ki/tests/spo/test_matmul_parrot_gather.py`, which the user identified as a direct description of the target `triton_kernels.matmul` inputs. The primary non-parrot cases use:

- `n_expts_act = 4`
- `b_shape = (5120, 10240)` meaning `K = 5120`, `N = 10240`
- `batch_size` sweep from `1` up to `33792`
- `n_expts_tot` / `n_expts_shards` combinations: `(256, 8)`, `(256, 16)`, `(256, 32)`, `(272, 8)`, `(288, 8)`
- `is_parrot_gather = False`

The smaller parrot-gather case uses:

- `n_expts_act = 4`
- `b_shape = (1280, 2560)` meaning `K = 1280`, `N = 2560`
- `batch_size` sweep from `1` to `128`
- `n_expts_tot = 64`, `n_expts_shards = 1`
- `is_parrot_gather = True`

The KI test also fixes the exact datatype/config pattern we should optimize first:

- Activations: plain `torch.float8_e4m3fn`, not microscaled activations
- Weights: local-expert MXFP4 packed values plus MX scales in the default mxfp4 weight/value layouts
- Bias: `float32`
- Output: `torch.float8_e4m3fn`
- Fused activation: `swiglu` with `reduction_n = 2`
- Gathered/ragged dispatch: output rows scale with `batch_size * n_expts_act`
- Precision config: flexpoint output path with `flexpoint_saturate_inf=True`, per-run `x_scale`, expected `y_scale`, and `b_mx_scale`

The persistent kernel in `python/triton_kernels/triton_kernels/matmul_details/_p_matmul.py` already contains a fused-gather path:

- `HAS_GATHER = GatherIndx is not None`
- `USE_GATHER_TMA = HAS_GATHER and X_TMA_MODE == "dense"`
- `x = X.gather(offs_x_m, off_k_x // block_div)` when gather uses dense TMA

The non-persistent kernel in `python/triton_kernels/triton_kernels/matmul_details/_matmul.py` handles gather by loading row indices directly from `GatherIndx`, so it is a fallback/reference path but not the obvious Blackwell optimization target.

The test surface in `python/triton_kernels/tests/test_matmul.py` already exercises gather/scatter and fp8/mxfp4 cases together, but it also documents current guardrails that matter for planning:

- Gather is not supported in batched mode.
- Gathered activations currently do not combine with x-scale swizzling (`a_hbm_swizzling`) in tests.
- Blackwell/native MXFP heuristics in `opt_flags.py` and `opt_flags_nvidia.py` push many MXFP cases toward persistent/TMA execution and cap `block_k` to conserve shared memory.

### Systems Involved

- System/component: `python/triton_kernels/triton_kernels/matmul.py`
- Why it matters: Host-side legality checks, layout handling, TMA descriptor construction, kernel launch arguments, and `opt_flags` selection all live here.

- System/component: `python/triton_kernels/triton_kernels/matmul_details/_p_matmul.py`
- Why it matters: Primary Blackwell persistent kernel with the existing fused-gather path and likely tuning target.

- System/component: `python/triton_kernels/triton_kernels/matmul_details/_matmul.py`
- Why it matters: Non-persistent fallback path and reference for gather behavior outside TMA.

- System/component: `python/triton_kernels/triton_kernels/matmul_details/opt_flags.py`
- Why it matters: Chooses persistent vs non-persistent execution and the block/stage/warp configuration that will likely dominate performance.

- System/component: `python/triton_kernels/triton_kernels/matmul_details/opt_flags_details/opt_flags_nvidia.py`
- Why it matters: Blackwell/NVIDIA-specific heuristics for `block_n`, `block_k`, warps, and stages.

- System/component: `python/triton_kernels/tests/test_matmul.py`
- Why it matters: Existing correctness coverage for gather, fp8, mxfp4, persistence, and layout constraints.

### Expected Change Surface

- Files/directories likely to change:
  - `python/triton_kernels/triton_kernels/matmul.py`
  - `python/triton_kernels/triton_kernels/matmul_details/_p_matmul.py`
  - `python/triton_kernels/triton_kernels/matmul_details/opt_flags.py`
  - `python/triton_kernels/triton_kernels/matmul_details/opt_flags_details/opt_flags_nvidia.py`
  - `python/perf/test_matmul_parrot_gather.py`
- Data models/APIs affected:
  - `PrecisionConfig`
  - launch-time `opt_flags`
  - gather/scatter input contract for `matmul`
- Operational or rollout touchpoints:
  - GB300 local benchmarking
  - direct handoff between this checkout and the user's laptop
  - persistent memory notes in `/root/.codex/memories/triton-fp8-mxfp4-fused-gather/`

## Assumptions and Risks

| Type | Item | Confidence (Low/Med/High) | Validation Plan |
|---|---|---|---|
| Assumption | The target workload remains on the non-batched gather/ragged path, because the KI harness uses gather with ragged metadata rather than the batched path rejected by `matmul`. | High | Benchmark the direct reproduction of the KI harness inputs and confirm the chosen kernel path. |
| Assumption | GB300 tuning will primarily land on the persistent `_p_matmul` path. | High | Capture the chosen kernel path and launch flags during baseline measurement. |
| Assumption | Weight layout and scale layout will matter materially for mxfp4 performance on Blackwell. | High | Benchmark swizzled vs strided layouts and record the selected launch flags. |
| Assumption | The highest-value first target is the large non-parrot family at `K=5120, N=10240`; the smaller parrot case is a secondary check. | Med | Start with the large family unless the user redirects prioritization. |
| Risk | Current `opt_flags` heuristics may choose suboptimal `block_m/block_n/block_k`, `num_warps`, or `num_stages` for fused gather. | High | Log chosen flags for the target shapes, then force controlled overrides and compare. |
| Risk | Gather may underutilize TMA or create row-index irregularity that hides the benefit of existing Blackwell MXFP handling. | High | Measure gathered vs ungathered baselines with the same dtype/layout settings. |
| Risk | Gathered activations and x-scale swizzling are currently disallowed together in tests, which narrows the safe tuning space. | High | Keep initial tuning away from x-scale swizzle unless we decide to extend support deliberately. |
| Risk | Editable Triton builds on this ARM64 host fail unless the GCC 13 C++ include paths are exported explicitly for bundled clang CUDA compilation. | High | Use the canonical `CPLUS_INCLUDE_PATH=... TRITON_BUILD_WITH_CCACHE=true pip install -v -e . --no-build-isolation` command from `AGENTS.md`. |
| Risk | Multi-session handoff can lose context if notes only live in chat or local shell history. | High | Keep canonical status in this initiative file and append measurements to persistent memory every session. |

## Plan

### Phase 1: Baseline and Scaffolding

- [x] Create durable planning and memory scaffolding for the workstream.
  - Artifact: Initiative doc, persistent memory directory, and global AGENTS note
  - Dependencies: None
  - Notes: Done on 2026-04-06 after codepath and machine discovery.
- [x] Define the representative benchmark matrix for fp8 activations + mxfp4 weights + fused gather on GB300.
  - Artifact: KI workload summary plus seeded perf log entries
  - Dependencies: User-provided target file
  - Notes: Derived from `/root/code/openai/lib/ki/tests/spo/test_matmul_parrot_gather.py`; includes large non-parrot and smaller parrot-gather families.
- [x] Add a local benchmark driver that reproduces the reference workload shapes and routing pattern.
  - Artifact: `python/perf/bench_matmul_parrot_gather.py`
  - Dependencies: Local editable installs for `triton` and `triton_kernels`
  - Notes: Uses only local `triton_kernels` code plus proton-matching flops/bytes accounting, emits `% peak FP8` and `% peak HBM bandwidth` with overridable GB300 roofline constants, times with `do_bench_cudagraph`, and can compare the baseline kernel against `python/perf/matmul_gluon.py::matmul_ogs` on the same prepared inputs.
- [ ] Capture current baseline performance and launch choices for the target cases.
  - Artifact: Baseline report in persistent memory
  - Dependencies: Benchmark matrix
  - Notes: Partial baseline captured: completed the full GB300 `E256/es8` non-parrot batch sweep (`bs=1..33792`) with `warmup=25` and `rep=100`; explicit launch-flag capture and the remaining family sweeps are still pending.

### Phase 2: Kernel Tuning

- [ ] Identify the dominant bottleneck in the persistent fused-gather path.
  - Artifact: Investigation note with candidate bottlenecks and evidence
  - Dependencies: Baseline report
  - Notes: Focus on gather load path, scale handling, shared-memory pressure, and scheduling. Partial result on `bs=16384, E256/es8`: `ws` and `gluon_optimized` launch the same `512`-thread / `152`-CTA shape with `128` registers per thread, but `ws` trails on tensor/memory utilization and shows materially higher `mio_throttle` and `long_scoreboard` stalls, pointing at producer/consumer scheduling inside the separate activation-vs-weight partitions rather than a simple occupancy shortfall.
- [ ] Implement the first round of host-side or kernel-side tuning changes.
  - Artifact: Code change plus targeted correctness coverage
  - Dependencies: Bottleneck analysis
  - Notes: Likely touch `_p_matmul.py`, launch config, or Blackwell opt-flag heuristics.
- [ ] Iterate on Blackwell-specific tuning knobs and compare variants.
  - Artifact: Benchmark comparison table with before/after deltas
  - Dependencies: First tuning patch
  - Notes: Include `block_m`, `block_n`, `block_k`, warps, stages, and layout interactions.

### Phase 3: Validation and Handoff

- [ ] Validate the tuned path against relevant correctness cases and nearby regressions.
  - Artifact: Test and benchmark log
  - Dependencies: Candidate tuned implementation
  - Notes: Keep scope tight to the touched path first, then expand if needed.
- [ ] Prepare handoff notes for the user's laptop sync.
  - Artifact: Summary of commits, measured gains, remaining risks, and follow-ups
  - Dependencies: Validated tuned state
  - Notes: Include exact repo state and any local-only artifacts that should not be fetched.

## Execution Log

- `2026-04-06` Completed: Initiative bootstrap, codepath discovery, and durable note setup
  - Artifact: `.codex/initiatives/artifacts/fp8-mxfp4-fused-gather-matmul.md`, `/root/.codex/memories/triton-fp8-mxfp4-fused-gather/`, `/root/AGENTS.md`
  - Validation: Confirmed 4x NVIDIA GB300 GPUs with `nvidia-smi`; inspected `matmul.py`, `_p_matmul.py`, `_matmul.py`, `opt_flags.py`, `opt_flags_nvidia.py`, and `test_matmul.py`
  - Learnings: The persistent kernel already exposes a dense-TMA gather path; gather is unsupported in batched mode; Blackwell/native MXFP heuristics favor persistent execution; gathered activations are currently incompatible with x-scale swizzling in tests
  - Plan updates: Start with baseline measurement on the persistent fused-gather path and postpone x-scale-swizzle combinations unless the target workload requires them
- `2026-04-06` Completed: Grounded the exact target workload from the KI harness and installed local editable packages
  - Artifact: `/root/code/openai/lib/ki/tests/spo/test_matmul_parrot_gather.py`, `/root/code/triton/AGENTS.md`
  - Validation: Verified imports resolve to `/root/code/triton/python/triton/__init__.py` and `/root/code/triton/python/triton_kernels/triton_kernels/__init__.py`; confirmed `ccache 4.9.1`; completed editable installs for `triton` and `triton_kernels`
  - Learnings: The priority workload uses plain fp8 activations with mxfp4 weights, ragged gather, swiglu fusion, and fp8 output; ARM64 Triton builds on this host require `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13` in addition to `TRITON_BUILD_WITH_CCACHE=true`
  - Plan updates: Benchmark the exact KI configuration directly against the local editable install and record launch flags/perf before changing code
- `2026-04-07` Completed: Added a local-only benchmark driver for the exact workload sweep
  - Artifact: `python/perf/bench_matmul_parrot_gather.py`
  - Validation: `python -m py_compile python/perf/bench_matmul_parrot_gather.py`; smoke-ran one parrot case and one large non-parrot case directly through `triton_kernels.matmul`
  - Learnings: The benchmark can stay entirely inside this repo while still reproducing the reference workload shapes and routing pattern; Triton-kernels flex dataclasses require scale tensors to be at least 1D, so scalar scales need a compatibility reshape
  - Plan updates: Use the local-only script for the first serious baseline sweep, then capture launch flags and identify opt-flag/kernel bottlenecks from the measured cases
- `2026-04-07` Completed: Collected the first full GB300 baseline sweep for the large non-parrot family (`E256/es8`)
  - Artifact: `/root/.codex/memories/triton-fp8-mxfp4-fused-gather/raw/es8_E256_batch_sweep_2026-04-07.csv`
  - Validation: `python python/perf/bench_matmul_parrot_gather.py --case-family non-parrot --limit 45 --csv-out /tmp/triton_perf/es8_E256_batch_sweep.csv`
  - Learnings: With `warmup=25` and `rep=100`, runtime stayed near `0.28-0.30 ms` through `bs=14336`; peak bandwidth reached `3.06 TB/s` at `bs=3328`; a clear latency cliff appeared at `bs=15360` and `bs=16384`; TFLOP/s kept rising and peaked at `2830.66` for `bs=32768`; bandwidth fell below `2 TB/s` starting at `bs=24576`
  - Plan updates: Instrument launch flags around the `bs=14336 -> 16384` transition and complete the remaining non-parrot baseline families before changing kernel code
- `2026-04-07` Completed: Added roofline normalization and re-collected the canonical `E256/es8` baseline sweep
  - Artifact: `python/perf/bench_matmul_parrot_gather.py`, `/root/.codex/memories/triton-fp8-mxfp4-fused-gather/raw/es8_E256_batch_sweep_roofline_2026-04-07.csv`
  - Validation: `python -m py_compile python/perf/bench_matmul_parrot_gather.py`; `python python/perf/bench_matmul_parrot_gather.py --case-family non-parrot --limit 45 --csv-out /root/.codex/memories/triton-fp8-mxfp4-fused-gather/raw/es8_E256_batch_sweep_roofline_2026-04-07.csv`
  - Learnings: The latest canonical sweep peaks at `56.16%` of the assumed dense FP8 roofline (`2808.21 TFLOP/s` at `bs=32768`) and `40.13%` of the assumed HBM roofline (`3.21 TB/s` at `bs=8192`); `bs=14336` reaches `50.17%` FP8 roofline while still sustaining `38.22%` HBM roofline; the efficiency cliff remains at `bs=15360 -> 16384`
  - Plan updates: Use the roofline-enabled CSV as the baseline source of truth and capture launch flags on both sides of the `bs=14336 -> 16384` transition
- `2026-04-07` Completed: Switched the benchmark to `do_bench_cudagraph` and re-ran the full `E256/es8` sweep
  - Artifact: `python/perf/bench_matmul_parrot_gather.py`, `/root/.codex/memories/triton-fp8-mxfp4-fused-gather/raw/es8_E256_batch_sweep_cudagraph_roofline_2026-04-07.csv`
  - Validation: `python -m py_compile python/perf/bench_matmul_parrot_gather.py`; `python python/perf/bench_matmul_parrot_gather.py --case-family non-parrot --limit 45 --rep 100 --csv-out /root/.codex/memories/triton-fp8-mxfp4-fused-gather/raw/es8_E256_batch_sweep_cudagraph_roofline_2026-04-07.csv`
  - Learnings: Removing eager host-launch overhead changes the shape of the curve materially at small and medium batch sizes; `bs=1` improves by about `20.9x` (`0.2989 -> 0.0143 ms`) and `bs=1024` by about `1.84x`, but the large-batch regime is nearly unchanged to slightly worse; peak HBM utilization shifts to `70.72%` at `bs=128` (`5.66 TB/s`) while peak FP8 utilization is `54.21%` at `bs=32768` (`2710.50 TFLOP/s`); the device-side cliff still appears at `bs=15360 -> 16384`
  - Plan updates: Treat the cudagraph sweep as the preferred baseline when comparing specialized kernel device-side performance, and keep the eager sweep only as a reference for end-to-end launch overhead

- `2026-04-07` Completed: Hooked the specialized Gluon kernel into the benchmark and added a correctness harness
  - Artifact: `python/perf/bench_matmul_parrot_gather.py`, `python/perf/matmul_gluon.py`, `python/perf/test_matmul_parrot_gather.py`
  - Validation: `make`; `python -m py_compile python/perf/matmul_gluon.py python/perf/bench_matmul_parrot_gather.py python/perf/test_matmul_parrot_gather.py`; `python python/perf/bench_matmul_parrot_gather.py --case-family non-parrot --limit 1 --kernel both --rep 20`; `pytest -s --tb=short python/perf/test_matmul_parrot_gather.py::test_matmul_ogs_matches_matmul[bs1_E256_es8_B5120x10240] python/perf/test_matmul_parrot_gather.py::test_matmul_ogs_matches_matmul[bs1_E64_es1_B1280x2560_parrot]`; `pytest -s --tb=short python/perf/test_matmul_parrot_gather.py::test_matmul_ogs_matches_matmul[bs14336_E256_es8_B5120x10240]`
  - Learnings: The benchmark can now emit side-by-side timing for the baseline and specialized kernels using identical prepared inputs and one CSV row per kernel; on this checkout the pushed Gluon kernel needed a small API update from `blackwell.get_tmem_reg_layout(...)` to descriptor `.get_reg_layout()` to compile against the local Triton version; the specialized kernel is already faster than baseline on the smallest `E256/es8` smoke case (`1.48x` at `bs=1`); the specialized-kernel correctness harness lives in `python/perf` so the perf workspace stays self-contained
  - Plan updates: Use the comparison mode for future performance sweeps while keeping the new perf-side pytest harness, which compares outputs in float32 with the existing fp8 x mxfp4 tolerance (`maxtol=3e-2`) and uses `assert_close` for output-scale checks, as the correctness gate for specialized-kernel edits
- `2026-04-07` Completed: Refactored the perf-side correctness harness to mirror benchmark case execution
  - Artifact: `python/perf/test_matmul_parrot_gather.py`
  - Validation: `make`; `python -m py_compile python/perf/test_matmul_parrot_gather.py`; `pytest -s --tb=short python/perf/test_matmul_parrot_gather.py`
  - Learnings: The test now reuses the benchmark module's case list, kernel iteration, and input preparation directly, so the perf-side correctness path stays aligned with the benchmarked workload instead of growing its own test scaffolding
  - Plan updates: Keep future perf-side correctness changes expressed in terms of benchmark helpers first, then add only the minimal comparison logic needed by pytest
- `2026-04-07` Completed: Replaced perf-side dynamic module loading with normal relative imports
  - Artifact: `python/perf/__init__.py`, `python/perf/bench_matmul_parrot_gather.py`, `python/perf/test_matmul_parrot_gather.py`
  - Validation: `python -m py_compile python/perf/__init__.py python/perf/bench_matmul_parrot_gather.py python/perf/test_matmul_parrot_gather.py`; `python -m python.perf.bench_matmul_parrot_gather --case-family non-parrot --limit 1 --kernel both --rep 20`; `pytest -s --tb=short python/perf/test_matmul_parrot_gather.py::test_matmul_ogs_matches_matmul[bs1_E256_es8_B5120x10240]`
  - Learnings: The perf benchmark and test modules now import each other with plain relative imports and a local `python/perf` package marker instead of `importlib` path loading; the repo root `python` directory works as a namespace package here, so a top-level `python/__init__.py` is unnecessary; benchmark execution should use module mode (`python -m python.perf.bench_matmul_parrot_gather ...`) so the relative imports resolve cleanly
  - Plan updates: Keep import-only changes on narrow validation by default: module importability, one benchmark smoke, and one targeted pytest case are enough unless the packaging change actually touches execution semantics
- `2026-04-07` Completed: Added benchmark-only `ws` kernel selection plus inline and validate-only correctness checks
  - Artifact: `python/perf/bench_matmul_parrot_gather.py`
  - Validation: `python -m py_compile python/perf/bench_matmul_parrot_gather.py python/perf/matmul_ws.py`; `python -m python.perf.bench_matmul_parrot_gather --case-family non-parrot --limit 1 --kernel both --rep 20`; `python -m python.perf.bench_matmul_parrot_gather --case-family non-parrot --limit 1 --kernel both --validate-only`
  - Learnings: The benchmark now treats baseline `triton_kernels.matmul` as the reference and can validate specialized kernels inline during timed runs; `--validate-only` runs the same correctness path without paying the cudagraph timing loop; `both` stays `original+gluon` so the perf pytest surface remains stable, while `all` adds the incomplete `ws` entrypoint for benchmark-side experimentation only
  - Plan updates: Keep `ws` benchmark-only until its entrypoint is complete enough for real timing, and use the new validate-only mode for fast correctness checks while iterating on specialized kernels
- `2026-04-07` Completed: Tightened the benchmark kernel contract to require strict `matmul`-compatible APIs
  - Artifact: `python/perf/bench_matmul_parrot_gather.py`
  - Validation: `python -m py_compile python/perf/bench_matmul_parrot_gather.py`; `python -m python.perf.bench_matmul_parrot_gather --case-family non-parrot --limit 1 --kernel both --rep 20`; `python -m python.perf.bench_matmul_parrot_gather --case-family non-parrot --limit 1 --kernel both --validate-only`
  - Learnings: The benchmark no longer carries a `None`-return fallback or a per-kernel call adapter; all specialized kernels are now expected to accept the same kwargs as `triton_kernels.matmul` and return the output tensor directly, which keeps the benchmark path simpler and avoids masking API drift
  - Plan updates: Keep future specialized kernel entrypoints aligned to the baseline `matmul` call contract so benchmark integration stays declarative rather than per-kernel special-cased
- `2026-04-09` Completed: Synced the pushed WS sources, added a benchmark-isolated `ws_optimized` fork, and ran the first bounded tuning loop on the 16K `E256/es8` case
  - Artifact: `python/perf/matmul_ws.py`, `python/perf/matmul_ws_optimized.py`, `python/perf/bench_matmul_parrot_gather.py`, `/tmp/ws_vs_gluon_opt_16384.ncu-rep`
  - Validation: `make`; `python -m py_compile python/perf/matmul_ws_optimized.py python/perf/bench_matmul_parrot_gather.py`; `python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel original,gluon,gluon_optimized,ws --limit 1`; `USE_IR_LOC=ttgir ncu -o /tmp/ws_vs_gluon_opt_16384 -f --import-source on --set full -k '::regex:.*matmul.*' python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws,gluon_optimized --limit 1 --validate-only`; `python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws,ws_optimized,gluon_optimized --limit 1`
  - Learnings: On the refreshed branch, `gluon_optimized` runs at about `0.3390 ms` while `ws` is about `0.3496 ms` on the target bucket. The best bounded `ws_optimized` variant so far keeps the baseline launch/scheduling shape and only raises the weight-loader register budget, reaching about `0.3480 ms`; attempts to add another weight-loader warp, cap the persistent grid at `128`, enable `XCD_SWIZZLE=2`, switch to `N_MAJOR`, or rebalance staging buffers all regressed or overflowed shared memory. The NCU capture confirms the gap is still dominated by higher `mio_throttle` and `long_scoreboard` stalls inside the WS producer pipeline rather than by register spill or CTA under-occupancy.
  - Plan updates: Keep `ws_optimized` as the sandbox for further tactical experiments, but treat the remaining `~2.7%` gap to `gluon_optimized` as evidence that a larger partition/layout change may be required if small heuristic nudges stop moving the result.
- `2026-04-09` Completed: Exhausted the remaining bounded WS heuristic sweep on the 16K `E256/es8` bucket
  - Artifact: `python/perf/matmul_ws_optimized.py`, `/tmp/ws_opt_vs_gluon_opt_16384.ncu-rep`
  - Validation: `python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws,ws_optimized,gluon_optimized --limit 1`; `USE_IR_LOC=ttgir ncu -o /tmp/ws_opt_vs_gluon_opt_16384 -f --import-source on --set full -k '::regex:.*matmul.*' python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1 --validate-only`; temporary direct-module sweeps over tile geometry, worker register budgets, worker warp allocations, and staging depths on the fixed `bs=16384, E256/es8` prepared case
  - Learnings: A Gluon-style epilogue transplant into `ws_optimized` keeps the kernel correct and slightly improves the direct fixed-case benchmark (`~0.3441 -> ~0.3437 ms` on one prepared case), but it does not materially change the harness result (`~0.3481 ms`) or close the gap to `gluon_optimized` (`~0.337-0.339 ms`). Geometry changes (`BLOCK_M=64`, `BLOCK_N=128`) were decisively worse, weight-heavy register reallocations regressed, `x_num_bufs=3` regressed badly, and `w_num_bufs=5` overflowed shared memory. The refreshed NCU profile for `ws_optimized` still shows the same signature as plain `ws`: lower DRAM/L2 throughput and tensor utilization than `gluon_optimized`, plus higher `long_scoreboard` (`6.57` vs `5.30`) and `mio_throttle` (`0.74` vs `0.30`) stall ratios.
  - Plan updates: Treat the current WS architecture as heuristically exhausted for the target bucket. The next promising step is a larger producer-pipeline rewrite, most likely collapsing the separate activation and weight+scale loader partitions into a combined load partition or otherwise changing how the WS pipeline is partitioned.
- `2026-04-09` Completed: Investigated ragged TMA output stores for the WS epilogue, including direct descriptor-store support in Gluon
  - Artifact: `python/perf/matmul_ws_optimized.py`, `python/triton/experimental/gluon/language/nvidia/hopper/tma.py`, `/tmp/ws_opt_tma_store_vs_gluon_opt_16384.ncu-rep`, `/tmp/ws_opt_direct_ragged_store_vs_gluon_opt_16384.ncu-rep`
  - Validation: temporary standalone Gluon probe using `create_ragged_descriptor(...)` plus `async_copy_shared_to_global(...)` to confirm ragged output descriptors work in this frontend; `python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized --limit 1 --validate-only`; `python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws,ws_optimized,gluon_optimized --limit 1`; `USE_IR_LOC=ttgir ncu -o /tmp/ws_opt_tma_store_vs_gluon_opt_16384 -f --import-source on --set full -k '::regex:.*matmul.*' python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1 --validate-only`; `USE_IR_LOC=ttgir ncu -o /tmp/ws_opt_direct_ragged_store_vs_gluon_opt_16384 -f --import-source on --set full -k '::regex:.*matmul.*' python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1 --validate-only`
  - Learnings: A staged ragged TMA-store epilogue with a `64x128` output subtile fits in the remaining shared-memory headroom, but it regressed badly in profile: CTA shared memory rose from `218.2 KiB` to `226.5 KiB`, DRAM/L2 throughput fell further, and kernel duration moved to about `501.5 us`. I then added a small Gluon builtin to issue direct tensor-descriptor stores from registers so `ws_optimized` could approximate Triton's `store_ragged(...)` path without the extra SMEM staging buffer. That direct ragged-store version validates, but the target harness result remains effectively unchanged (`~0.3481 ms`), and a full-tile `128x128` ragged store regressed to about `0.3508 ms`. Under NCU, the direct ragged-store kernel still profiles at about `501.9 us` with lower memory/tensor utilization than `gluon_optimized`, `0` explicit global stores, and a large jump in shared-store instructions (`~2.55M`), so the epilogue write path is still not the primary lever.
  - Plan updates: Keep the direct descriptor-store support as a useful Gluon capability, but treat ragged TMA output stores as another exhausted bounded direction for this WS design. The remaining gap still points back to the separate producer partitions rather than to the scalar epilogue stores.
- `2026-04-09` Completed: Reinvested epilogue shared-memory savings into WS stage-count experiments
  - Artifact: `python/perf/matmul_ws_optimized.py`, `/tmp/wsopt_split4.ncu-rep`, `/tmp/wsopt_5x4.ncu-rep`
  - Validation: `make`; `PYTHONPATH=python:python/triton_kernels TRITON_WS_X_NUM_BUFS=4 TRITON_WS_W_NUM_BUFS=4 TRITON_WS_EPILOGUE_N_ELEMS=4 python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1`; `PYTHONPATH=python:python/triton_kernels TRITON_WS_X_NUM_BUFS=5 TRITON_WS_W_NUM_BUFS=4 TRITON_WS_EPILOGUE_N_ELEMS=4 python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1`; `PYTHONPATH=python:python/triton_kernels TRITON_WS_X_NUM_BUFS=3 TRITON_WS_W_NUM_BUFS=5 TRITON_WS_EPILOGUE_N_ELEMS=4 python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1`; `PYTHONPATH=python:python/triton_kernels TRITON_WS_X_NUM_BUFS=5 TRITON_WS_W_NUM_BUFS=4 TRITON_WS_EPILOGUE_N_ELEMS=4 USE_IR_LOC=ttgir ncu -o /tmp/wsopt_5x4 -f --import-source on --set full -k '::regex:.*matmul.*' python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1 --validate-only`
  - Learnings: Changing the epilogue target layout from blocked `[1,2]` to `[1,4]` cuts the WS epilogue relayout scratch from `16 KiB` to `4 KiB`, which frees enough SMEM to add a fifth activation stage while staying under the `232,448 B` CTA cap. On the target `bs=16384, E256, es8` case, the `[1,4]` control (`x_num_bufs=4, w_num_bufs=4`) runs at about `0.3477 ms`, the extra activation-stage variant (`5/4`) improves materially to about `0.3422 ms`, and the tradeoff variant that funds a fifth weight stage by dropping activation stages (`3/5`) regresses badly to about `0.3937 ms`. NCU confirms the promoted `5/4` variant keeps the kernel under budget at about `222.4 KB` shared memory per block and improves the WS kernel duration itself (`~489.3 us -> ~484.3 us`), but it still trails `gluon_optimized` (`~0.3389 ms`) with the same broad stall signature.
  - Plan updates: Keep the `[1,4]` epilogue plus `x_num_bufs=5` as the new bounded WS default. If more stage depth is needed, the next bytes should go toward preserving the fifth activation stage while searching for additional savings elsewhere, because funding a fifth weight stage by dropping activation buffering was decisively harmful on the target bucket.

## Next Up

- [ ] Capture launch flags around the `E256/es8` device-side efficiency cliff under cudagraph benchmarking and run the remaining non-parrot baseline families (`E256/es16`, `E256/es32`, `E272/es8`, `E288/es8`)
- [ ] Keep searching for bounded ways to preserve the `x_num_bufs=5` gain while cutting enough additional SMEM to test `w_num_bufs=5` without dropping activation buffering; if that stalls again, revisit a larger WS partition rewrite with the user

## Open Questions

- Question: If time is limited, should optimization start with the large non-parrot family only, or should the smaller parrot-gather case be treated as equally blocking?
  - Owner: `root@codex-kernel-devbox-0.brix.jeffniu.svc.cluster.local`
  - Resolution path: Ask the user for prioritization; otherwise start with the large `K=5120, N=10240` family and validate the parrot case after.

## Deferred / Out of Scope

- Batched gather support
  - Reason: `matmul` explicitly rejects gather in batched mode today.
  - Follow-up initiative (if any): None yet.
- Gather plus x-scale swizzle enablement
  - Reason: Test coverage currently skips this combination; it should be treated as a separate extension unless the target workload requires it.
  - Follow-up initiative (if any): None yet.
