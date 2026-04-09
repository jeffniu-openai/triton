---
owner: root@codex-kernel-devbox-0.brix.jeffniu.svc.cluster.local
created: 2026-04-06T23:18:36Z
updated: 2026-04-09T09:05:04Z
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
- [x] Implement the first round of host-side or kernel-side tuning changes.
  - Artifact: Code change plus targeted correctness coverage
  - Dependencies: Bottleneck analysis
  - Notes: Done for the `python/perf/matmul_ws_optimized.py` sandbox on the target `bs=16384, E256, es8` case. The winning patch is a bounded epilogue rewrite: packed `f32x2` accumulator scale+bias, packed output scaling, and a packed FP8 store path that keeps the final `e4m3x2` conversion in halfword form instead of exploding back to byte stores.
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
- `2026-04-09` Completed: Moved active tuning to a fresh worktree and re-checked whether the TMEM load layout itself can satisfy the SwiGLU `reshape(..., 2) -> split` constraint
  - Artifact: `/root/code/triton-ws-opt`, `python/perf/matmul_ws_optimized.py`
  - Validation: dumped the actual f32 TMEM load layout bases with `gluon_ir.compute_tmem_reg_layout(...)` for the `128x256` accumulator tile and mapped them through the `128x256 -> 128x128x2` reshape used by `swiglu`
  - Learnings: For the f32 accumulator case, the relevant TMEM-compatible load layout is the `32x32b` linear layout. Its register bases are `[0,1], [0,2], ..., [0,64]`, so after the `N=256 -> (128, 2)` reshape the low-bit basis becomes `[0,0,1]`, which is exactly the register-local last-dimension basis that `SplitOp`'s generic linear-layout path requires. The 8-warp `split-long-M` TMEM variant produced by `getTmemLoadLayoutSplitLongM(...)` preserves the same register bases and therefore also satisfies the split requirement. The important conclusion is that the current epilogue relayout is not needed for `gl.split` itself; it is still being driven by the downstream bias/store layout choices.
  - Plan updates: Treat split-compatibility as solved for the TMEM-side linear layout and focus the next epilogue experiments on keeping bias broadcast and final stores legal and fast without converting into the current blocked store-oriented layout.
- `2026-04-09` Completed: Added a benchmarkable `ws_split_sf` fork with a separate weight-scale producer partition and swept its stage counts on the target bucket
  - Artifact: `python/perf/matmul_ws_split_sf.py`, `python/perf/bench_matmul_parrot_gather.py`
  - Validation: `make` (fails in `/root/code/triton-ws-opt` because `build/cmake.linux-aarch64-cpython-3.12/build.ninja` is absent in this worktree); `python -m py_compile python/perf/matmul_ws.py python/perf/matmul_ws_optimized.py python/perf/matmul_ws_split_sf.py python/perf/bench_matmul_parrot_gather.py`; `PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_split_sf,ws_optimized,gluon_optimized --limit 1 --validate-only`; `PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws,ws_optimized,ws_split_sf,gluon_optimized --limit 1`; `TRITON_WS_SPLIT_SF_X_NUM_BUFS=5 TRITON_WS_SPLIT_SF_W_NUM_BUFS=4 TRITON_WS_SPLIT_SF_SCALE_NUM_BUFS={1,2,3,4,5} PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_split_sf --limit 1`; `TRITON_WS_SPLIT_SF_X_NUM_BUFS=6 TRITON_WS_SPLIT_SF_W_NUM_BUFS=4 TRITON_WS_SPLIT_SF_SCALE_NUM_BUFS=1 ...`; `TRITON_WS_SPLIT_SF_X_NUM_BUFS=5 TRITON_WS_SPLIT_SF_W_NUM_BUFS=5 TRITON_WS_SPLIT_SF_SCALE_NUM_BUFS=1 ...`
  - Learnings: The split-scale variant is correct but slower than the existing WS pipeline on the target `bs=16384, E256, es8` case. With the best measured bounded config (`x_num_bufs=5`, `w_num_bufs=4`, `w_scale_num_bufs=4`), `ws_split_sf` runs at about `0.3723 ms` versus `ws_optimized` at `0.3431 ms`, `ws` at `0.3494 ms`, and `gluon_optimized` at `0.3390 ms`. Lowering the scale-ring depth regressed sharply (`scale_num_bufs=1 -> 0.7886 ms`, `2 -> 0.4656 ms`, `3 -> 0.3785 ms`, `4 -> 0.3727 ms`, `5 -> 0.3728 ms`), so the new partition only stops hurting once the scale ring climbs back to roughly the same depth as the weight ring. The hoped-for shared-memory trade never materialized: `x_num_bufs=6, w_num_bufs=4, scale_num_bufs=1` overflows shared memory at `234,720 B`, and `x_num_bufs=5, w_num_bufs=5, scale_num_bufs=1` overflows at `251,136 B`, both above the `232,448 B` hardware limit.
  - Plan updates: Keep `ws_split_sf` as an experimental branch of the WS design, but treat the separate scale-loader partition as another bounded direction that does not buy useful stage depth on the target bucket. The next meaningful work should stay focused on the epilogue relayout and larger producer-pipeline changes instead of on additional scale-ring micro-tuning.
- `2026-04-09` Completed: Added a reusable `ws_split_sf` sweep harness and ran a broad legality/performance sweep over buffer-count trades
  - Artifact: `python/perf/sweep_matmul_ws_split_sf.py`, `/root/.codex/memories/triton-fp8-mxfp4-fused-gather/raw/ws_split_sf_buffer_sweep_2026-04-09.csv`
  - Validation: `make` in `/root/code/triton-ws-opt` (fails because the fresh worktree still lacks `build/cmake.linux-aarch64-cpython-3.12/build.ninja`); `make` in `/root/code/triton` (`ninja: no work to do`, which is the editable Triton runtime these perf scripts are currently using); `python -m py_compile python/perf/sweep_matmul_ws_split_sf.py`; `PYTHONPATH=python/triton_kernels python -m python.perf.sweep_matmul_ws_split_sf --x-range 5 --w-range 4 --scale-range 4 --rep 5 --rerun-rep 12 --top 1`; `PYTHONPATH=python/triton_kernels python -m python.perf.sweep_matmul_ws_split_sf --x-range 2:6 --w-range 2:5 --scale-range 1:8 --rep 6 --rerun-rep 40 --top 20 --csv-out /root/.codex/memories/triton-fp8-mxfp4-fused-gather/raw/ws_split_sf_buffer_sweep_2026-04-09.csv`
  - Learnings: The broad sweep covered `160` configs on the target `bs=16384, E256, es8` bucket. The rerun baselines in the same harness were `ws=0.3448 ms`, `ws_optimized=0.3385 ms`, and `gluon_optimized=0.3328 ms`. The best rerun `ws_split_sf` config was still far behind at `x_num_bufs=5, w_num_bufs=4, scale_num_bufs=4 -> 0.3672 ms`; the next best cluster was `x=4, w=4, scale=4..6 -> 0.3682-0.3688 ms`. The strongest pattern is that extra scale buffering helps only until about `4-6` scale buffers and then flattens or regresses. `scale_num_bufs=1` is catastrophically bad across the grid (`~0.77-0.79 ms`), `2` is still poor (`~0.47-0.55 ms` on the plausible regions), and the first useful regime starts at `3-4`. Sacrificing weight or activation buffers to buy more scale depth never wins: the best legal configs keep `w_num_bufs=4` and `x_num_bufs=4-5`, and even those remain slower than plain `ws`. The legality map is also now clear: every extra scale buffer costs `1024 B` of shared memory, `x=4, w=5` is illegal for all swept scale counts (`234,720 B` and above), and `x=6, w=4` is also illegal for all swept scale counts (`234,720 B` and above).
  - Plan updates: Treat `ws_split_sf` buffer-count tuning as exhausted on the target bucket. The sweep harness is still useful for future architecture experiments, but no additional bounded `x/w/scale` sweeps are likely to change the conclusion unless another kernel change first creates materially more shared-memory headroom or removes the extra partition cost.
- `2026-04-09` Completed: Added durable PTX 9.2 research notes and rewrote the winning WS epilogue around packed float2 math plus packed FP8 stores
  - Artifact: `.codex/initiatives/artifacts/ptx-isa-9.2-blackwell-float2-research.md`, `python/perf/matmul_ws_optimized.py`, `/tmp/wsopt_packed_store_vs_gluon.ncu-rep`
  - Validation: `make` in `/root/code/triton`; `python -m py_compile python/perf/matmul_ws_optimized.py`; `CUDA_VISIBLE_DEVICES=0/1/2/3 PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather ...` across the bounded epilogue sweeps; `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python/triton_kernels TRITON_WS_USE_EXP2_SIGMOID=0 TRITON_WS_USE_PACKED_FINAL_FMA=0 TRITON_WS_USE_PACKED_FP8_STORE=1 TRITON_WS_USE_PACKED_OUT_SCALE=1 TRITON_WS_EPILOGUE_N_ELEMS=4 python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws,ws_optimized,gluon_optimized --limit 1 --warmup 30 --rep 100`; `CUDA_VISIBLE_DEVICES=3 USE_IR_LOC=ttgir PYTHONPATH=python/triton_kernels TRITON_WS_USE_EXP2_SIGMOID=0 TRITON_WS_USE_PACKED_FINAL_FMA=0 TRITON_WS_USE_PACKED_FP8_STORE=1 TRITON_WS_USE_PACKED_OUT_SCALE=1 TRITON_WS_EPILOGUE_N_ELEMS=4 ncu -o /tmp/wsopt_packed_store_vs_gluon -f --import-source on --set full -k '::regex:.*matmul.*' python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1 --validate-only`
  - Learnings: The shell on this host still cannot mirror `docs.nvidia.com` directly (`wget`, `curl`, `requests`, and `urllib` all fail with the same TLS EOF), so the durable PTX notes were written from the accessible official docs via the browsing/subagent path instead of from a local mirror. On the kernel side, the useful bounded epilogue changes were: `fma.rn.f32x2` for accumulator scale+bias, `mul.f32x2` for output scaling, and a custom packed `cvt.rn.satfinite.e4m3x2.f32` + `st.global.b16` store path that preserves FP8 pairs through the final write. The rejected variants are also now explicit: the `exp2` sigmoid path failed validation on the target bucket, scalar output scaling regressed, packed final SwiGLU FMA regressed, and the float2 path only compiled cleanly with `EPILOGUE_N_ELEMS=4` (`2/8/16` hit layout-mismatch failures in `float2.fma`). The winning measured result on the target bucket is now `ws=0.3536 ms`, `ws_optimized=0.3420 ms`, `gluon_optimized=0.3443 ms` with `warmup=30`, `rep=100`, so `ws_optimized` is about `3.3%` faster than `ws` and about `0.7%` faster than `gluon_optimized`.
  - Plan updates: Promote the packed-store epilogue as the default WS-optimized target-bucket configuration. The next useful work is to validate whether the same epilogue still wins on the remaining non-parrot families and to decide whether further gains should come from store-layout generalization or from a larger producer-pipeline rewrite.
- `2026-04-09` Completed: Probed epilogue fragment scheduling and a blocked packed-store relayout without changing the default winning WS path
  - Artifact: `python/perf/matmul_ws_optimized.py`
  - Validation: `make` in `/root/code/triton`; `python -m py_compile python/perf/matmul_ws_optimized.py`; `CUDA_VISIBLE_DEVICES=0/1/2 PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized --limit 1 --validate-only`; `CUDA_VISIBLE_DEVICES=0/1/2/3 PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws,ws_optimized,gluon_optimized --limit 1 --warmup 15 --rep 50`; `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1 --warmup 20 --rep 60`; `CUDA_VISIBLE_DEVICES=1 TRITON_WS_USE_BLOCKED_PACKED_STORE=1 PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1 --warmup 20 --rep 60`
  - Learnings: I tried three bounded fragment-scheduling directions for the epilogue: splitting the TMEM-loaded accumulator after a full register load, splitting after the blocked relayout, and splitting after the packed `f32x2` accumulator-scale+bias step. All three run into the same current Gluon/Triton limitation when `EPILOGUE_SUBTILE_N` is reduced below `256`: `SplitOp` / `Fp4ToFpOp` requires at least two elements per thread in the last dimension for the layouts produced here, so the smaller `128`/`64` fragment schedules do not compile cleanly without a larger rewrite or compiler-side support. I also tried a default-off blocked relayout of the final packed FP8 halfwords before `gl.store`; it validates, but the backend still lowers the final write to `st.global.b16`, not a wider vector store, and the same-GPU measurement regressed slightly (`ws_optimized ~0.3383 ms -> ~0.3410 ms` on GPU 1, while `gluon_optimized` stayed around `0.3415-0.3421 ms`). The current default path therefore remains the best bounded epilogue configuration on this branch, still landing around `ws ~0.3491 ms`, `ws_optimized ~0.3371-0.3393 ms`, and `gluon_optimized ~0.3396-0.3429 ms` across the confirmation runs here.
  - Plan updates: Keep the fragment scheduler and blocked packed-store relayout as documented rejected directions for this bounded tuning pass. If the user still wants more epilogue-only gain after this point, the next meaningful step is likely a larger manual fragment rewrite or compiler-aware layout work rather than another small flag sweep.
- `2026-04-09` Completed: Retried epilogue fragmenting on the arithmetic/store side and checked the remaining bounded epilogue flag space on a same-GPU basis
  - Artifact: `python/perf/matmul_ws_optimized.py`
  - Validation: `make` in `/root/code/triton`; `python -m py_compile python/perf/matmul_ws_optimized.py`; `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1 --warmup 30 --rep 100`; `CUDA_VISIBLE_DEVICES=0/1/2/3 PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1 --warmup 15 --rep 50` with bounded env sweeps; `CUDA_VISIBLE_DEVICES=1 TRITON_WS_EPILOGUE_ROW_SUBTILE_FACTOR=2 PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized --limit 1 --validate-only`; `CUDA_VISIBLE_DEVICES=2 TRITON_WS_EPILOGUE_ROW_SUBTILE_FACTOR=4 PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized --limit 1 --validate-only`
  - Learnings: The packed-output N-fragment idea is still blocked by the compiler even when moved later in the epilogue: splitting the packed `out` tile ran into the same `SplitOp` / layout-compatibility failures as the earlier accumulator-side attempts, and a manual `i32` repack store path hit the same limitation when trying to pair adjacent packed-FP8 halfwords. I removed those dead-end knobs from the sandbox code after confirming they were just another form of the same compiler fight. The useful surviving experiment is a row-fragment epilogue path (`TRITON_WS_EPILOGUE_ROW_SUBTILE_FACTOR=2/4`) that splits `gelu`/`linear` by rows, computes SwiGLU per fragment, and stores each row fragment with fragment-sized pointer math. It validates cleanly, but the same-GPU comparison says it is neutral to slightly worse on the target bucket: with `warmup=30`, `rep=100` on GPU 0, the current default is `ws_optimized=0.3426 ms` versus `0.3433 ms` for row-2, `0.3428 ms` for row-4, and `0.3434 ms` for row-4 plus packed-final-FMA, while `gluon_optimized` stays around `0.3454-0.3459 ms`. I also rechecked the existing bounded epilogue flags on the same GPU: `packed_final_fma` (`0.3434 ms`), scalar output scaling (`0.3438 ms`), and the combination (`0.3437 ms`) all lost to the default `0.3426 ms`. The main lesson is that cross-GPU spot checks can make some epilogue variants look faster, but once the comparison is held on one GPU the current default path is still the best bounded WS epilogue on this branch.
  - Plan updates: Keep the row-fragment epilogue as an optional exploration hook in `matmul_ws_optimized.py`, but do not promote it as the default. Treat the packed-output N-fragment and manual wider-store repack directions as rejected bounded ideas for now. If more epilogue-only gain is still required, the next real step is a larger manual fragment rewrite or compiler-aware layout/store work rather than another small flag sweep.
- `2026-04-09` Completed: Implemented a second round of packed row-fragment epilogue scheduling and reran it with same-GPU confirmation
  - Artifact: `python/perf/matmul_ws_optimized.py`, `/tmp/ws_epilogue_fragment_sweep.json`, `/tmp/ws_epilogue_same_gpu_candidates.json`
  - Validation: `make` in `/root/code/triton`; `python -m py_compile python/perf/matmul_ws_optimized.py`; `CUDA_VISIBLE_DEVICES=0/1/2/3 PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized --limit 1 --validate-only` across `row_subtile_factor={1,4,8}`, `schedule={direct,wavefront}`, and `packed_final_fma={0,1}` spot checks; mixed-GPU screening sweep saved to `/tmp/ws_epilogue_fragment_sweep.json`; same-GPU reruns on GPU 0 with `--kernel ws,ws_optimized,gluon_optimized --warmup 30 --rep 100` for the best direct and wavefront candidates
  - Learnings: The new code path now splits rows directly on the packed accumulator tile before unpacking the SwiGLU pairs, which shortens the nonlinear/store live range without introducing extra shared memory. I implemented both a direct row-fragment schedule and a wavefront schedule that pipelines `prepare -> nonlinear -> store` across fragments, and I extended the row factor sweep to `2/4/8`. The mixed-GPU screen made several candidates look promising relative to same-command `gluon_optimized` (`~0.7-1.2%` wins depending on GPU), but the longer same-GPU confirmation collapsed the gain into noise: the best medium-run candidate was `row=2, direct` at about `ws=0.3473 ms`, `ws_optimized=0.3399 ms`, `gluon_optimized=0.3428 ms` on GPU 0 with `warmup=20`, `rep=60`, while the longer rerun put both the current default and `row=2, direct` at about `ws_optimized=0.3441 ms` against `gluon_optimized=0.3457-0.3458 ms`. Direct row-fragment plus packed final FMA (`row=2/4/8`) and the fragment-local scale+bias materialization probe both regressed and were removed.
  - Plan updates: Keep the packed row-fragment scheduling support in `matmul_ws_optimized.py` as an internal exploration hook, but do not change the default configuration on the basis of this round. The current packed-float2 + packed-FP8-store epilogue remains the promoted default; if more epilogue-only gain is required, the next step should be a larger manual fragment/dataflow rewrite rather than more bounded row/schedule sweeps.
- `2026-04-09` Completed: Added a reusable 4-GPU `ws_optimized` sweep harness, exposed worker-allocation knobs, and cleaned the experimental surface before the next larger epilogue rewrite
  - Artifact: `python/perf/sweep_matmul_ws_optimized.py`, `python/perf/matmul_ws_optimized.py`, `/root/.codex/memories/triton-fp8-mxfp4-fused-gather/raw/ws_optimized_epilogue_sweep_2026-04-09a.csv`, `/root/.codex/memories/triton-fp8-mxfp4-fused-gather/raw/ws_optimized_epilogue_sweep_2026-04-09b.csv`, `/root/.codex/memories/triton-fp8-mxfp4-fused-gather/raw/ws_optimized_epilogue_sweep_2026-04-09c.csv`, `/tmp/wsopt_clean_reconfirm.csv`
  - Validation: `make` in `/root/code/triton`; `python -m py_compile python/perf/matmul_ws_optimized.py python/perf/sweep_matmul_ws_optimized.py`; broad 4-GPU screening sweeps over `row_factor`, helper/direct schedule, worker warps, and worker register budgets using `python python/perf/sweep_matmul_ws_optimized.py ...`; same-GPU cleanup reconfirmation on GPU 0 with `python python/perf/sweep_matmul_ws_optimized.py --gpus 0 --batch-size 16384 --n-expts-tot 256 --n-expts-shards 8 --rep 40 --csv-out /tmp/wsopt_clean_reconfirm.csv --x-num-bufs 5 --w-num-bufs 4 --row-factors 8 --packed-final-fma 1 --store-helper 1 --schedules direct,wavefront --load-activation-warps 4 --load-weight-warps 1 --mma-warps 1 --store-helper-warps 1,2 --load-activation-regs 96,104 --load-weight-regs 48,56 --mma-regs 24 --store-helper-regs 16,24`
  - Learnings: The useful infrastructure from this round is the new 4-GPU sweep harness and the env-controlled worker-allocation tuning surface in `matmul_ws_optimized.py` (`load_activation_warps`, `load_weight_warps`, `mma_warps`, `store_helper_warps`, and their register budgets). The best basin remains a row-8 helper-store epilogue with a direct schedule and `aw=4`, `ww=1`, `mw=1`, `srw=2`; on the post-cleanup same-GPU rerun, the current default `ws_optimized` baseline was `0.3388 ms`, `gluon_optimized` was `0.3352 ms`, and the best helper-store configs landed at `0.3352-0.3355 ms`, effectively tying `gluon_optimized` while beating the default by about `0.0033-0.0036 ms`. The invalid or unstable options from the exploratory branch are now explicitly removed from the promoted surface: `num_warps=12` is rejected by the backend because Triton CUDA requires a power-of-two warp count, and the `STORE_HELPER_DEPTH=1` path deadlocked in practice, so the live helper pipeline is back to a fixed depth-2 ring.
  - Plan updates: Keep the 4-GPU harness and the worker-allocation knobs as the reusable infrastructure for future `ws_optimized` work. Do not spend more time on helper depth-1 or non-power-of-two warp counts. The next experiments should be larger exact-math epilogue rewrites that use this harness for fast cross-GPU screening and same-GPU promotion checks.
- `2026-04-09` Completed: Revalidated the active branch, found that the `6e50a0a5ae` helper-cleanup commit regressed default exactness, and restored the last valid `ws_optimized` epilogue file before continuing
  - Artifact: `python/perf/matmul_ws_optimized.py`, `/tmp/triton-016ec`, `/tmp/triton-7995`
  - Validation: `make` in `/root/code/triton`; `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1 --validate-only`; `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1 --warmup 20 --rep 60`; detached-worktree validation of `016ec044c4` and `7995b3cabd` with the same `--validate-only` command
  - Learnings: The current `HEAD` after `6e50a0a5ae` was not actually a safe promotion point: `ws_optimized` missed `original` by 6 deterministic output elements on the target bucket even with all obvious epilogue toggles disabled. Detached worktree checks showed both `016ec044c4` and `7995b3cabd` still validate cleanly, so the regression boundary is exactly the helper-cleanup commit. Restoring `python/perf/matmul_ws_optimized.py` to the `016ec044c4` version in the active branch brings the kernel back to a correct beating-gluon baseline on GPU 0 (`ws_optimized=0.3404 ms`, `gluon_optimized=0.3448 ms`, both `ok(original)`).
  - Plan updates: Treat the sweep harness from `6e50a0a5ae` as still useful infrastructure, but do not trust the widened worker-allocation/helper-tuning surface from that commit until it is reintroduced on top of a validating baseline. Resume exact-math epilogue work from the restored `016ec044c4` file state, then add new tuning controls incrementally with immediate `--validate-only` checks.

## Next Up

- [ ] Capture launch flags around the `E256/es8` device-side efficiency cliff under cudagraph benchmarking and run the remaining non-parrot baseline families (`E256/es16`, `E256/es32`, `E272/es8`, `E288/es8`)
- [ ] Validate the packed-store `ws_optimized` epilogue on the remaining non-parrot families and the smaller parrot-gather family to see whether the target-bucket win generalizes or is tightly bucket-specific
- [ ] Do not spend more bounded tuning effort on `ws_split_sf` or more small row-fragment schedule sweeps; if more target-bucket gain is required beyond the packed-store win, the next move is either a larger manual epilogue fragment/dataflow rewrite, compiler-aware layout work, or a larger WS producer-pipeline rewrite with the user
- [ ] Use the new 4-GPU sweep harness plus same-GPU confirmation to test larger exact-math `ws_optimized` epilogue rewrites, starting with linear-layout epilogues and more aggressive helper-partition work sharing

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
