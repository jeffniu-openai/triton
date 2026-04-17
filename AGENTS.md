# Working on Triton

## GitHub Account and Remote
- For this Triton checkout and ongoing TMEM work on `codex/tmem`, use the `jeffniu-openai` GitHub account and credentials for all GitHub operations, including pushes, PR updates, comments, and related `gh` actions.
- Push checkpoint commits to `origin` at `github.com/jeffniu-openai/triton` unless explicitly instructed otherwise.

## Build and Testing Guidelines
- Before running any tests, run `make` in the triton directory to rebuild triton.
- For compiler changes, add tests in `python/test/` (pytest) or test (lit). Keep GPU-only tests in `python/test/unit/` or `python/test/gluon/`, name them `test_<feature>_<condition>`, and avoid creating new test files unless requested.
- Run pytest with `-s --tb=short`. Run a single test with `pytest file.py::test_name`.
- The build dir is given by `BUILD_DIR := $(shell PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')`
- Run lit from the build dir:  `cd BUILD_DIR; ninja triton-opt; lit -v test/<path>.mlir` (example: `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir`).
- Lit tests can be run locally (no GPU required).
- Compiler crashes sometimes print an MLIR reproducer (external_resources / mlir_reproducer). Save the full MLIR + {-# ... #-} metadata to `/tmp/<file>.mlir`, then run `triton-opt /tmp/<file>.mlir --run-reproducer` to reproduce locally.

## Execution Safety
- This machine can be unstable. Make frequent, modular, incremental commits instead of carrying large uncommitted diffs.
- Commit messages must be detailed enough to serve as a durable handoff: record context, why the change was needed, what changed, and any important validation or remaining boundaries.
- After each commit for ongoing TMEM work, push the current `HEAD` to `jeffniu-openai/codex/tmem` so the remote branch is always recoverable if the node dies mid-session.
- Always push checkpoint commits to the `github.com/jeffniu-openai/triton` remote using the `jeffniu-openai` GitHub credentials. For long checkpoints or changesets, make occasional recoverability commits with the subject `WIP checkpoint: <very brief tag>` and push them promptly so interrupted work is not lost.
- Keep commits scoped so they can be understood and reverted independently.
- Before resuming the TMEM linear-layout generalization initiative, start with `.codex/initiatives/tmem_linear_generalization/README.md`, then re-read `.codex/initiatives/tmem_linear_generalization/memory.md` as the initiative's durable memory/source of truth, and then read the latest tail of `.codex/initiatives/tmem_linear_generalization/handoff_2026-04-09.md`. Do not optimize only for the current red tests; keep the work aligned with the full mission:
  - support arbitrary linear TMEM layouts and descriptor-view chains whenever the ISA can realize them correctly;
  - keep clean negatives only for true ISA-impossible cases;
  - prefer one shared planner and exact linear-layout arithmetic over family-specific rescue stacks; and
  - after the current bug buckets are green, continue through the recorded long-term phases (`ld.red`, `copy` `warpx2`, broader MMAv5 reachable-family support, heuristic cleanup, and staged broad validation) instead of treating the initiative as done.
- For TMEM initiative work, keep `.codex/initiatives/tmem_linear_generalization/README.md`, `.codex/initiatives/tmem_linear_generalization/memory.md`, `.codex/initiatives/tmem_linear_generalization/log.md`, and `.codex/initiatives/tmem_linear_generalization/handoff_2026-04-09.md` aligned at meaningful checkpoints. Do not let critical state live only in chat context. Commit messages should remain detailed enough that a reader of `git log` can recover the relevant context, motivation, validation, and remaining boundaries without needing the lost session transcript.
- For the TMEM linear-layout generalization project, also keep `.codex/initiatives/tmem_linear_generalization/completion_execution_tracker.md` current. Record the active phase, remaining clean-negative inventory, completed plan items, validation evidence, and the next concrete implementation slice there; do not stop execution or end the turn while unblocked work remains unless explicitly interrupted by the user or a real external blocker.
- When executing the TMEM generalization plan, write or refresh the plan in the initiative documents before relying on it, use those documents as the active checklist at session start and after each meaningful slice, track progress steadily against that plan, and continue executing without ending the turn until the tracked plan is complete unless the user interrupts or a concrete external blocker is reached.
- Treat the initiative documents, not chat context, as the active checklist for that no-stop execution rule. Whenever the plan changes or a bucket is proved to be a true boundary, update the initiative docs before continuing so the next resumed session can follow the same checklist.
- When debugging compiler or codegen bugs, identify and fix the core linear-layout, planner, or lowering issue rather than layering patchwork or ad-hoc special cases. Use targeted probes to find the real abstraction mismatch first, then implement the general fix and update tests to match correct behavior.
- For TMEM reinterpret lowering in particular, prefer exact linear-layout arithmetic end-to-end:
  - derive support layouts, origins, row anchors, packet offsets, and query families from `LinearLayout` compose/invert/pseudoinvert algebra;
  - assume there is a sane algebraic lowering for reinterpret/view composition and keep pushing toward it, because once the arithmetic is right the implementation tends to stop producing whole classes of bugs;
  - expect the arithmetic to be tricky, but treat that as a reason to solve the math cleanly rather than to accumulate one-off rewrites;
  - treat special-case row/col rewrites as a temporary smell to remove, not a durable design; and
  - keep clean negatives only for cases the ISA cannot realize after the layout arithmetic is done correctly.
- For multi-session initiatives, keep a dated handoff document in the initiative folder and update it at every meaningful checkpoint and before any likely context rollover. Record the current branch/HEAD, dirty files, exact repros run, what passed or failed, the current root-cause hypothesis, and the next concrete steps so the next session is not dependent on context compaction alone.
- During the GB200/NVIDIA CI stabilization phase, keep `.codex/initiatives/tmem_linear_generalization/gb200_nvidia_ci_inventory.md`, `.codex/initiatives/tmem_linear_generalization/gb200_failure_manifest.md`, and `.codex/initiatives/tmem_linear_generalization/gb200_branch_recovery_plan.md` aligned with the latest branch-vs-merge-base classification:
  - record exact repro commands for each bucket;
  - mark whether a failure is preexisting on merge-base, branch-added coverage, branch-changed coverage, a real new regression, or flake/cache-sensitive noise;
  - keep reinterpret-contract tests separate from proven compiler regressions until they are rewritten to explicit TMEM view APIs where appropriate; and
  - track performance-only deltas separately so correctness recovery stays first.
- After each meaningful GB200 branch-recovery fix, explicitly mark which older manifests and aggregate counts are now stale, refresh the nearest exact/file-level reruns, and record the new current-head failing set before using any pre-fix counts for prioritization.
- If a test looks cache-sensitive, verify whether the symptom survives a fresh process boundary before blaming `TRITON_CACHE_DIR`. Prefer root-causing missing cache-key inputs, process/device contamination after bad kernels, or compiler global-state reuse over relying on environmental workarounds.
- Do not import helpers from other test files, especially via `from python.test...`. Move reusable test kernels, descriptor builders, and assertions into a non-test helper module next to the tests, and import that helper without `sys.path` edits or child-process `PYTHONPATH` injection.
- Do not work around runtime-matrix issues by assigning a fresh temporary `TRITON_CACHE_DIR` inside tests or subprocess wrappers. Compilation dominates these sweeps; preserve stable per-GPU cache directories across reruns unless a test is explicitly validating cache isolation. If a cache-sensitive symptom appears, identify the missing cache-key input, stale runtime metadata, process/device contamination, or async-compile state bug and fix that root cause.

## Python Sweep Best Practices
- Always leverage all 4 GPUs for pytest work when possible. Even focused `-k` slices and small runtime buckets should normally run as four `pytest-split` groups with one outer pytest process per GPU and a distinct `TRITON_CACHE_DIR` per process.
- Install and use `pytest-split` for outer sharding and keep `pytest-xdist` available for lighter CPU-bound cases.
- Calibrate local runtime against GB200 CI: the full GB200 NVIDIA lane on a comparable 4-GPU machine is expected to take roughly 35 minutes including clean build and LLVM download, with actual test time around 20 minutes. Treat single pytest shards that run far beyond that as a partitioning, xdist, cache, or hang problem first.
- For broad selectors with hundreds of parametrized cases, collect/store durations and use duration-aware splitting before raising timeouts: add `--store-durations --durations-path <path>` on representative runs, then reuse that file with `--splitting-algorithm=least_duration` or finer split counts.
- Use `pytest-xdist` inside a single GPU shard when it improves CPU-bound collection/compile throughput, but keep inner parallelism conservative (`-n 1` or `-n 2` first). Higher xdist fanout can create rare false-negative OOM failures by oversubscribing GPU memory or process-local compiler/runtime state.
- Treat validation in stages:
  - `make`
  - targeted lit checks for compiler-only changes
  - focused pytest nodeids or `-k` slices split across the 4 GPUs
  - 4-GPU grouped sweeps only after the focused slice is green
- Keep compile-only and heavy GPU runtime sweeps separate. Do not mix large runtime files with broad compile-only files in the same shard.
- For heavy GPU runtime files, run one outer pytest process per GPU:
  - set `CUDA_VISIBLE_DEVICES=<gpu>`
  - set a distinct `TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu>`
  - use the same file list on all shards with `pytest -s --tb=short --splits 4 --group <group>`
- Preferred 4-GPU runtime sweep pattern:
  - GPU 0: `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_core.py python/test/gluon/test_tmem_runtime_matrix.py`
  - GPU 1: `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_core.py python/test/gluon/test_tmem_runtime_matrix.py`
  - GPU 2: `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_core.py python/test/gluon/test_tmem_runtime_matrix.py`
  - GPU 3: `CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_core.py python/test/gluon/test_tmem_runtime_matrix.py`
- Do not combine `--splits 4` with `-n auto` for heavy TMEM runtime sweeps. If inner parallelism is needed for lighter sweeps, keep it small (`-n 1` or `-n 2`) and do not oversubscribe the GPU.
- Use `pytest --collect-only -q <files>` before large sweeps when changing file lists or selecting exact nodeids.
- Rerun failures by exact nodeid on an isolated GPU before rerunning an entire shard.
