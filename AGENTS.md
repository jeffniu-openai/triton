# Working on Triton

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
- Keep commits scoped so they can be understood and reverted independently.
- Before resuming the TMEM linear-layout generalization initiative, start with `.codex/initiatives/tmem_linear_generalization/README.md`, then re-read `.codex/initiatives/tmem_linear_generalization/memory.md` as the initiative's durable memory/source of truth, and then read the latest tail of `.codex/initiatives/tmem_linear_generalization/handoff_2026-04-09.md`. Do not optimize only for the current red tests; keep the work aligned with the full mission:
  - support arbitrary linear TMEM layouts and descriptor-view chains whenever the ISA can realize them correctly;
  - keep clean negatives only for true ISA-impossible cases;
  - prefer one shared planner and exact linear-layout arithmetic over family-specific rescue stacks; and
  - after the current bug buckets are green, continue through the recorded long-term phases (`ld.red`, `copy` `warpx2`, broader MMAv5 reachable-family support, heuristic cleanup, and staged broad validation) instead of treating the initiative as done.
- For TMEM initiative work, keep `.codex/initiatives/tmem_linear_generalization/README.md`, `.codex/initiatives/tmem_linear_generalization/memory.md`, `.codex/initiatives/tmem_linear_generalization/log.md`, and `.codex/initiatives/tmem_linear_generalization/handoff_2026-04-09.md` aligned at meaningful checkpoints. Do not let critical state live only in chat context. Commit messages should remain detailed enough that a reader of `git log` can recover the relevant context, motivation, validation, and remaining boundaries without needing the lost session transcript.
- When debugging compiler or codegen bugs, identify and fix the core linear-layout, planner, or lowering issue rather than layering patchwork or ad-hoc special cases. Use targeted probes to find the real abstraction mismatch first, then implement the general fix and update tests to match correct behavior.
- For TMEM reinterpret lowering in particular, prefer exact linear-layout arithmetic end-to-end:
  - derive support layouts, origins, row anchors, packet offsets, and query families from `LinearLayout` compose/invert/pseudoinvert algebra;
  - assume there is a sane algebraic lowering for reinterpret/view composition and keep pushing toward it, because once the arithmetic is right the implementation tends to stop producing whole classes of bugs;
  - expect the arithmetic to be tricky, but treat that as a reason to solve the math cleanly rather than to accumulate one-off rewrites;
  - treat special-case row/col rewrites as a temporary smell to remove, not a durable design; and
  - keep clean negatives only for cases the ISA cannot realize after the layout arithmetic is done correctly.
- For multi-session initiatives, keep a dated handoff document in the initiative folder and update it at every meaningful checkpoint and before any likely context rollover. Record the current branch/HEAD, dirty files, exact repros run, what passed or failed, the current root-cause hypothesis, and the next concrete steps so the next session is not dependent on context compaction alone.

## Python Sweep Best Practices
- Install and use `pytest-split` for outer sharding and keep `pytest-xdist` available for lighter CPU-bound cases.
- Treat validation in stages:
  - `make`
  - targeted lit checks for compiler-only changes
  - focused pytest nodeids or `-k` slices on one GPU
  - 4-GPU grouped sweeps only after the focused slice is green
- Keep compile-only and heavy GPU runtime sweeps separate. Do not mix large runtime files with broad compile-only files in the same shard.
- For heavy GPU runtime files, run one outer pytest process per GPU:
  - set `CUDA_VISIBLE_DEVICES=<gpu>`
  - set a distinct `TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu>`
  - use the same file list on all shards with `pytest -s --tb=short --splits 4 --group <group>`
- Preferred 4-GPU runtime sweep pattern:
  - GPU 0: `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=python:. pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_core.py python/test/gluon/test_tmem_runtime_matrix.py`
  - GPU 1: `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=python:. pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_core.py python/test/gluon/test_tmem_runtime_matrix.py`
  - GPU 2: `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=python:. pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_core.py python/test/gluon/test_tmem_runtime_matrix.py`
  - GPU 3: `CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=python:. pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_core.py python/test/gluon/test_tmem_runtime_matrix.py`
- Do not combine `--splits 4` with `-n auto` for heavy TMEM runtime sweeps. If inner parallelism is needed for lighter sweeps, keep it small (`-n 1` or `-n 2`) and do not oversubscribe the GPU.
- Use `pytest --collect-only -q <files>` before large sweeps when changing file lists or selecting exact nodeids.
- Rerun failures by exact nodeid on an isolated GPU before rerunning an entire shard.
