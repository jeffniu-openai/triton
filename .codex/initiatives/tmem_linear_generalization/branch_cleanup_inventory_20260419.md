# Branch Cleanup Inventory - 2026-04-19

Status: executed cleanup checkpoint; validation passed.

Production cleanup checkpoint commit:
`e1e4d6336934706c3d91bf5f268c7f54fb49351e`, pushed to
`origin/codex/tmem`.

Branch: `codex/tmem`

Merge-base against `origin/main`: `11ee1144a737006921231bbd3386c187812c38e1`

Starting head: `5ef66b3bb87e7c5991d92b5b1d35953b7aca4fbd`

## Raw Branch Inventory

The branch diff from the merge-base is intentionally broad: `623` files and
roughly `594k` added lines. Most of the line count is not production compiler
logic. It comes from durable initiative state and raw experiment outputs:

- Production TMEM/compiler surfaces:
  - `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h`;
  - `include/triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h`;
  - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`;
  - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`;
  - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`;
  - `lib/Dialect/TritonNvidiaGPU/Transforms/*`;
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/*`;
  - `python/src/gluon_ir.cc`.
- Validation/test surfaces:
  - `python/test/gluon/test_tmem_runtime_matrix.py`;
  - `python/test/gluon/tmem_test_utils.py`;
  - lit tests under `test/TritonNvidiaGPU/` and
    `test/Conversion/lower_tensor_memory_to_llvm.mlir`.
- Durable initiative records:
  - README, memory, log, handoff, gap registers, GB200 manifests, validation
    recipes, and closure reports under
    `.codex/initiatives/tmem_linear_generalization/`.
- Raw experiment/probe outputs:
  - `298` files under `.codex/initiatives/tmem_linear_generalization/experiments`;
  - `256` tracked result files under `experiments/results`;
  - about `19 MB` of raw failure/result logs in result files alone.

## Keep

- Keep the production TMEM backend abstractions and tests. They are the actual
  support surface and coverage anchor.
- Keep the initiative source-of-truth docs required by `AGENTS.md`.
- Keep reusable scripts and small durable artifacts that are explicitly used by
  later validation:
  - `run_tmem_runtime_matrix_sweep.py`;
  - `tmem_runtime_matrix_validation_recipe_20260413.md`;
  - `ldst_pytest_durations_20260413.json`;
  - `experiments/mmav5_i8_remote/`, including the PTX/cubin pair, because the
    user explicitly asked for reusable GB200 validation artifacts.
- Keep compact probe summaries when they are the only durable machine-readable
  evidence for a boundary.

## Cleanup Plan

1. Remove temporary trace-file hooks and debug-disable shims from production
   code.
   - Remove `TRITON_TRACE_TMEM_QUERY_LOWERING_FILE` and
     `TRITON_TRACE_TMEM_REG_LAYOUT_FILE`, both of which append to hardcoded
     `/tmp` paths and were only useful during investigation.
   - Remove `TRITON_DISABLE_TMEM_SUPPORT_QUERY_LOWERING` and
     `TRITON_DISABLE_TYPE_ONLY_TMEM_REG_LAYOUT_FALLBACK`, which are old
     toggles for comparing against obsolete fallback behavior.
   - Keep normal structured `TRITON_DEBUG_TMEM_*` diagnostics for now; they are
     gated and still useful for future backend audits.
2. Prune superseded raw experiment outputs.
   - Remove bulky raw probe/sweep logs and per-offset JSONL dumps now
     summarized in README, log, memory, handoff, and closure reports.
   - Keep `experiments/results/README.md`, duration data used by the sweep
     runner, and compact current summary/result files that are still useful as
     evidence anchors.
   - Remove ignored `__pycache__` from the initiative experiments directory.
3. Validate narrowly.
   - Run `make -j8`.
   - Run `git diff --check`.
   - Run focused TMEM selectors affected by the removed shims:
     - Gap #2 narrow scaled-MMAv5 runtime matrix;
     - direct ld/st descriptor/query smoke;
     - Gluon TMEM register layout smoke if collection stays cheap.
4. Commit and push the cleanup checkpoint to `origin/codex/tmem`.

## Executed Cleanup

- Removed investigation-only trace-file hooks and debug-disable toggles from
  the production TMEM paths:
  - `TRITON_TRACE_TMEM_QUERY_LOWERING_FILE`;
  - `TRITON_TRACE_TMEM_REG_LAYOUT_FILE`;
  - `TRITON_DISABLE_TMEM_SUPPORT_QUERY_LOWERING`;
  - `TRITON_DISABLE_TYPE_ONLY_TMEM_REG_LAYOUT_FALLBACK`.
- Kept normal gated `TRITON_DEBUG_TMEM_*` diagnostics.
- Pruned superseded raw result logs and per-offset probe dumps under
  `experiments/results/`, while keeping:
  - `README.md`;
  - `ldst_pytest_durations_20260413.json`;
  - compact `*_summary.json`, `*_current.jsonl`, and `*_current.log` files;
  - reusable opcode/shape/gap evidence artifacts.
- Added `.gitignore` rules so future raw result files stay local unless they
  are explicitly promoted as compact durable evidence.
- Removed ignored Python `__pycache__` directories from the initiative folder.

## Validation

- `make -j8`;
- `git diff --check`;
- split-4 focused runtime-matrix selector covering direct `ld/st`
  descriptor/query paths plus narrow scaled-MMAv5 support:
  `13/13`, `13/13`, `13/13`, and `13/13` passed;
- split-4 `test_core.py::test_tmem_descriptor_chain_matrix` Gluon smoke:
  `7/7`, `7/7`, `7/7`, and `5/5` passed.

## Later Cleanup Candidates

- Consolidate the remaining `TRITON_DEBUG_TMEM_*` printing behind one helper if
  compile-time profiling shows the `getenv` checks matter. Reviewed on
  2026-04-20; no source consolidation was made because this pass had no
  profiling evidence that the remaining gated checks materially affect compile
  time.
- Review the names of Python runtime matrix `legacy` parameters. Executed on
  2026-04-20 by adding explicit `TensorMemoryLayout` compatibility constants
  and helper usage in representative runtime-matrix cases while preserving the
  historical `legacy` nodeid spelling used by manifests and logs.
- Split or squash historical initiative docs before PR presentation only if
  the user wants a smaller public review surface. The current repo instructions
  still require durable initiative state to stay in-tree.
