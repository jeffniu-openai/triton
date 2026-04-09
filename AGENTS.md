# Working on Triton

## Repo Workflow
- Keep the `jeffniu-openai` remote configured to `https://github.com/jeffniu-openai/triton.git`.
- This machine is unstable, so make frequent incremental and modular commits while working.
- Write detailed commit messages and bodies. Include rationale, scope, and any relevant benchmark or profiling logs so the git history itself serves as initiative documentation.
- When you make a commit, push the current branch to the matching branch on `jeffniu-openai` and keep that remote branch up to date.
- This machine has 4 GPUs available. For performance-sensitive sweeps, prefer parallelizing work across all 4 GPUs when practical instead of serializing everything onto one device.

## Build and Testing Guidelines
- Before running any tests, run `make` in the triton directory to rebuild triton.
- For compiler changes, add tests in `python/test/` (pytest) or test (lit). Keep GPU-only tests in `python/test/unit/` or `python/test/gluon/`, name them `test_<feature>_<condition>`, and avoid creating new test files unless requested.
- Run pytest with `-s --tb=short`. Run a single test with `pytest file.py::test_name`.
- The build dir is given by `BUILD_DIR := $(shell PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')`
- Run lit from the build dir:  `cd BUILD_DIR; ninja triton-opt; lit -v test/<path>.mlir` (example: `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir`).
- Lit tests can be run locally (no GPU required).
- Compiler crashes sometimes print an MLIR reproducer (external_resources / mlir_reproducer). Save the full MLIR + {-# ... #-} metadata to `/tmp/<file>.mlir`, then run `triton-opt /tmp/<file>.mlir --run-reproducer` to reproduce locally.
