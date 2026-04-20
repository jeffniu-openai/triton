# Working on Triton

## Build and Testing Guidelines
- Before running any tests, run `make` in the triton directory to rebuild triton.
- For compiler changes, add tests in `python/test/` (pytest) or test (lit). Keep GPU-only tests in `python/test/unit/` or `python/test/gluon/`, name them `test_<feature>_<condition>`, and avoid creating new test files unless requested.
- Run pytest with `-s --tb=short`. Run a single test with `pytest file.py::test_name`.
- The build dir is given by `BUILD_DIR := $(shell PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')`
- Run lit from the build dir:  `cd BUILD_DIR; ninja triton-opt; lit -v test/<path>.mlir` (example: `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir`).
- Lit tests can be run locally (no GPU required).
- Compiler crashes sometimes print an MLIR reproducer (external_resources / mlir_reproducer). Save the full MLIR + {-# ... #-} metadata to `/tmp/<file>.mlir`, then run `triton-opt /tmp/<file>.mlir --run-reproducer` to reproduce locally.

## 2CTA MoE BMM1 Optimization Workstream
- Apply the `large-project-methodology` skill to this workstream: keep durable notes, validation evidence, benchmark artifacts, and handoff state updated as the work evolves.
- Never voluntarily stop, pause, or end a turn while actively optimizing this workstream if any untested hypothesis, unmeasured regression, plausible 2CTA tuning path, unfinished validation, running benchmark/profiler, or uncommitted durable state remains. If an external limit forces a handoff, first update durable notes with the current branch, artifacts, validation, active hypothesis, and exact next command.
- The current success criterion is stronger than parity: create, tune, or replace kernels until `python/examples/gluon/05-moe-bmm1-fused-gather.py` has a 2CTA Gluon path that is at least `20%` faster than the 1CTA comparator for every batch size, then enable 2CTA for every batch size. Treat any batch with less than `1.20x` 2CTA-vs-1CTA speedup as unfinished work.
- Continue optimizing `python/examples/gluon/05-moe-bmm1-fused-gather.py` without converting open optimization work into a plan-only response. Keep experimenting, measuring, editing, validating, documenting, committing, and pushing coherent slices. The target is to make the 2CTA kernel at least `20%` faster than 1CTA on every batch size, enable 2CTA for every batch size, then keep improving performance forever, especially for batch sizes `<= 2048`.
- Treat 2CTA performance optimization as an ongoing performance workstream. Keep searching for better schedules, tile shapes, stage counts, buffer counts, register allocations, descriptor layouts, epilogue overlap, and routing-dependent behavior.
- When the user authorizes parallel work, generously and liberally leverage subagents instead of doing all thinking serially. Use them for research, current-codebase analysis, related-codebase analysis, review of code and performance ideas, detailed `ncu` analysis, hypothesis testing, experiment design, autotuning, and verification. Useful comparison targets include TensorRT-LLM, CUTLASS, vLLM, SGLang, Triton tutorials, and other high-performance MoE or grouped GEMM kernels.
- Keep subagents coordinated with explicit ownership. Do not let benchmark subagents compete on the same GPU as final measurements; assign independent GPUs or read-only analysis tasks, and record their findings in durable notes before relying on them.
- The machine has 4 GPUs. Use them to parallelize independent compile-cache warmups, correctness checks, autotuning sweeps, benchmark sweeps, and profiler runs. When measuring final performance or comparing close variants, first make sure the selected GPU is quiet and avoid running competing workloads on that GPU.
- Prefer paired benchmarks using the same prepared inputs when comparing 1CTA versus 2CTA or old versus new selectors. Use longer repetitions for small or noisy deltas, and confirm any apparent regression before changing selector policy.
- Use `ncu` for detailed performance analysis when benchmark results are surprising or when a candidate needs deeper diagnosis. Capture the command, GPU, batch size, routing distribution, key metrics, and conclusion in durable notes.
