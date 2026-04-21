# TMEM Structural Fuzzing Campaign, 2026-04-21

## Operating Contract

This is the active catalog for the 24-hour structural TMEM backend fuzzing
campaign requested on 2026-04-21.

- Continue fuzzing until explicitly interrupted by the user or blocked by a
  concrete external failure.
- Keep using subagents in independent lanes and local four-GPU runtime sweeps.
- Prefer Python/Gluon runtime tests that compile, execute, compare outputs, and
  inspect opcodes.
- Use lit/IR repros to minimize compiler crashes and verifier false negatives.
- Catalog and diagnose findings first. Do not begin backend fixes while new
  failures are still being discovered unless the user explicitly pivots to
  repair.
- Make frequent checkpoint commits and push `origin/codex/tmem`.

## Active Artifacts

- Runtime fuzzer pytest: `python/test/gluon/test_tmem_structural_fuzzer.py`
- Existing broad matrix: `python/test/gluon/test_tmem_runtime_matrix.py`
- Fuzzing playbook: `fuzz_plan.md`
- Completion tracker: `completion_execution_tracker.md`
- Handoff log: `handoff_2026-04-09.md`

## Case Schema

Every structural fuzz case records:

- `case_id`
- `seed`
- `family`: `ldst`, `ldred`, `copy`, `mma`, `mma_scaled`, `membar`, or
  `generic_pass`
- `shape`
- `dtype`
- `layout_kind`
- `view_chain`
- `num_warps`
- `num_ctas`
- expected result: `pass`, `clean_unsupported`, or `clean_error`
- observed result
- failure class: `compiler_crash`, `false_unsupported`, `miscompile`,
  `opcode_mismatch`, `diagnostic_regression`, or `flake`
- exact command
- repro/minimization status

## Round Log

### Round 0, Campaign Setup

- Time: 2026-04-21 08:18 UTC
- Branch: `codex/tmem`
- Starting HEAD: `1a3c13b93`
- User direction: run a continuous 24-hour fuzzing campaign, use subagents,
  primarily write Python runtime tests, catalog and diagnose bugs, and do not
  start fixing until fuzzing stops finding new bugs.
- Setup tasks:
  - recorded no-stop fuzzing contract in `AGENTS.md`;
  - promoted `fuzz_plan.md` to the active campaign playbook;
  - added this campaign catalog;
  - creating deterministic Python structural fuzzer and launching subagent
    lanes.
- Initial harness validation found setup issues, not classified backend bugs:
  - `ldst-view-identity-32x32b` compared a reshaped/permuted view directly
    against the base logical tensor instead of round-tripping through the base
    descriptor;
  - copy opcode extraction was too broad for stable `tcgen05.cp` assertions;
  - pytest-split group 4 had zero tests with only nine initial cases, so local
    harness validation should use whole-file or split-3 until the fuzzer grows.
- Validation after harness cleanup:
  - `make -j8`: no work to do;
  - `PYTHONPATH=.:./python python -m py_compile python/test/gluon/test_tmem_structural_fuzzer.py`;
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py`
    passed as `8 passed, 1 xfailed`.

## Failure Catalog

### FZ-20260421-0001: constexpr-branch two-CTA scales copy emits cta_group::1

- Status: open, discovery-only, not fixed.
- First seen: Round 0 local fuzzer validation.
- Case id: `copy-scales-warpx4-2cta`
- Seed: `0x302`
- Test: `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-2cta]`
- Command:
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-2cta]`
- Observed behavior: runtime output matches input, but PTX/LLIR opcode check
  sees two `tcgen05.cp.cta_group::1.warpx4.32x128b` instructions.
- Expected behavior: two-CTA copy should use `cta_group::2`, matching the
  existing non-constexpr-branch runtime-matrix anchor
  `test_tmem_runtime_matrix_cp_scales_warpx4_twocta_direct_copy`, which still
  passes.
- Failure class: opcode mismatch / possible false single-CTA instruction-form
  selection in a structurally generated kernel.
- Current minimization hypothesis: the fuzzer kernel expresses one-CTA and
  two-CTA copy forms behind a `TWO_CTAS` constexpr branch. The existing
  dedicated two-CTA kernel emits the correct opcode. This points at a
  structural/codegen specialization or module two-CTA discovery gap rather
  than a simple copy-planner failure.
- Current test disposition: checked-in fuzzer marks the case `xfail(strict)`,
  so it remains visible without making every fuzz sweep red during the
  discovery phase.

## Repro Queue

- FZ-20260421-0001: minimize `TWO_CTAS` constexpr branch interaction and
  compare TTGIR attributes against the dedicated runtime-matrix two-CTA copy
  kernel. Do not fix until the active fuzzing campaign stops finding new
  failures or the user pivots to repair.
