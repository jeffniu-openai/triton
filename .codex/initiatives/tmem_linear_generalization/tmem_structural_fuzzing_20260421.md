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
    passed as `8 passed, 1 xfailed` before the stale direct scales-copy
    expectation was corrected.
  - After Lane D reclassified direct two-CTA scales-copy opcode expectations,
    the `ldst or ldred` subset passed `7 passed`, and the full fuzzer should
    be rerun as a green `9/9` check.
  - Full fuzzer rerun after correction:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py`
    passed as `9 passed`.

### Lane D Round 1, Warp-Specialization / Membar / >2 CTA

- Time: 2026-04-21 08:23 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_warpspec_membar_round1.md`
- Scope: runtime and lit probes for TMEM interactions with warp-specialized
  shared-input TMA/MMA, partition scheduling, cluster membar insertion, async
  completion barriers, descriptor-view memdesc chains, pair-leader behavior,
  and 2/4/16 CTA CGA cases.
- Result: no backend compiler crash, verifier false unsupported diagnostic,
  clean-error regression, or runtime miscompile found.
- Non-backend finding: rerunning `copy-scales-warpx4-2cta` on current HEAD
  still emits `tcgen05.cp.cta_group::1.warpx4.32x128b`, but the existing
  runtime-matrix contract for the same direct two-CTA scales-copy geometry also
  expects `cta_group::1` and passes. Lane D therefore classifies this as a
  stale structural-fuzzer opcode expectation, not a backend failure.

### Lane B Round 1, Copy

- Time: 2026-04-21 08:24 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_copy_round1.md`
- Scope: deterministic copy-family runtime probes covering scales/non-scales,
  `warpx4`, `warpx2::{01_23,02_13}`, indexed and subslice descriptor views,
  shared subslices, two-CTA cases, >2 CTA clean diagnostics where present, and
  packed/subword clean diagnostics.
- Result: no new copy-family compiler crash, runtime miscompile, or unexpected
  verifier failure.
- Validation summary:
  - broad copy selector split-4 passed as `54/54/54/51`;
  - exact structural copy fuzzer originally passed `1` and xfailed the stale
    opcode expectation, now corrected locally;
  - direct single/two-CTA scales copy anchors passed `2/2`;
  - two-CTA layout in 4-CTA context clean diagnostic passed;
  - two-CTA `warpx2::02_13` clean unsupported boundary passed `2/2`;
  - copy descriptor/subslice/diagnostic collect-only inventory selected
    `70/1615` nodeids.

### Lane C Round 1, Plain MMAv5 / Scaled-MMAv5

- Time: 2026-04-21 08:25 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_mma_round1.md`
- Result: no compiler crashes, false unsupported diagnostics, opcode
  mismatches, or runtime miscompiles found.
- Validation summary:
  - broad MMA/scaled-MMA selector: `294 passed, 1 skipped`;
  - plain MMAv5 FPSAN payload selector: `12 passed, 4 skipped`;
  - scaled-MMAv5 two-CTA accumulator-subslice selector: `28 passed`;
  - plain MMAv5 indexed/two-CTA accumulator-view selector: `67 passed`.

### Lane E Round 1, Generic Pass / Layout Analysis

- Time: 2026-04-21 08:25 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_generic_pass_round1.md`
- Result: found five real failures: two compiler crashes and three
  miscompiles around runtime/dynamic memdesc values, helper-returned TMEM view
  chains, mixed tensor+memdesc captures, and layout-conversion pressure.
- Exact repro commands and logs are recorded in the lane report.

### Lane A Round 1, ld/st and ld.red

- Time: 2026-04-21 08:25 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldst_ldred_round1.md`
- Result: found stable ld/st and ld.red failures plus one false-unsupported
  candidate. One initial baseline mismatch was classified as a flake after
  fresh-process reruns passed.
- Exact repro commands and minimization notes are recorded in the lane report.

### Lane A2 Round 2, ld/st and ld.red Promotion

- Time: 2026-04-21 09:10 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldst_ldred_round2.md`
- Scope: promote/minimize lane A findings `FZ-20260421-0003` through
  `FZ-20260421-0006` into checked-in Python runtime xfail coverage without
  backend repairs.
- Result: added strict xfail coverage in
  `python/test/gluon/test_tmem_structural_fuzzer.py` for the ld/st chain1
  miscompile, ld.red plain-load opcode mismatch, 256-row allocator assertion,
  and transpose/slice ld.red false-unsupported candidate.
- Validation:
  - required `make -j8` completed with ninja reporting no work to do;
  - exact promoted nodeids passed as `4 xfailed`;
  - full structural fuzzer passed as `9 passed, 9 xfailed`;
  - Python byte-compile passed.

### Lane E2 Round 2, Generic Pass Promotion

- Time: 2026-04-21 08:30 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_generic_pass_round2.md`
- Scope: promote/minimize FZ-20260421-0001 and FZ-20260421-0002 from the
  temporary Lane E probe into checked-in Python runtime xfail tests.
- Result: added five strict xfail nodeids to
  `python/test/gluon/test_tmem_structural_fuzzer.py` covering the dynamic
  TMEM `memdesc_index` compiler crash and the helper/control-flow/layout
  pressure miscompiles. No backend/compiler code was changed.
- Validation summary:
  - required `make`: `ninja: no work to do`;
  - generic-pass collect-only selected `5/18` structural-fuzzer nodeids;
  - all five promoted exact nodeids were run in fresh pytest processes and
    reported `1 xfailed` each.

## Failure Catalog

### FZ-20260421-0001: dynamic TMEM memdesc_index reaches LLVM conversion

- Source: Lane E findings `LANE-E-GP-001` and `LANE-E-GP-002`.
- Failure class: `compiler_crash`.
- Family: `generic_pass` / dynamic memdesc indexing.
- Shape: parent `[2, 128, 64]`, selected view `[128, 64]`.
- View chains:
  - chain0: `parent.index(runtime_i32).reshape((64,2,64)).permute([1,0,2]).reshape((128,64))`;
  - chain1: `parent.index(runtime_i32).reshape((64,2,32,2)).permute(...).reshape((128,64))`.
- Observed: `failed to legalize operation 'ttg.memdesc_index' that was
  explicitly marked illegal` in `ConvertTritonGPUToLLVM`.
- Expected: either compile/runtime pass, or a clean unsupported diagnostic if
  runtime TMEM indexing is intentionally unsupported.
- Repro: see `fuzz_generic_pass_round1.md` exact commands and
  `/tmp/lane_e_dynamic_index_chain{0,1}.log`.
- Promotion status: checked in as strict xfail runtime nodeids:
  - `test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-index-chain0]`;
  - `test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-index-chain1]`.

### FZ-20260421-0002: helper-returned chain0 TMEM view miscompiles through control flow

- Source: Lane E findings `LANE-E-GP-003`, `LANE-E-GP-004`, and
  `LANE-E-GP-005`.
- Failure class: `miscompile`.
- Family: `generic_pass` / memdesc `scf.if` results and layout-conversion
  pressure.
- Shape: parent `[2, 128, 64]`, selected view `[128, 64]`.
- View chain: helper-returned
  `reshape((64,2,64)).permute([1,0,2]).reshape((128,64))`.
- Observed:
  - dynamic `if` view case: output mismatches `input + 30.0`;
  - mixed tensor+memdesc capture case: output mismatches `input + input + 3.0`;
  - layout-conversion pressure case: output mismatches `input`.
- Control: inline structural fuzzer chain0 baseline passes, so the current
  evidence points at helper/control-flow/layout-pass interaction.
- Repro: see `fuzz_generic_pass_round1.md` exact commands and logs.
- Promotion status: checked in as strict xfail runtime nodeids:
  - `test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-if-chain0-true]`;
  - `test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-mixed-captures-chain0]`;
  - `test_tmem_structural_fuzzer_generic_pass_layout_conversion_pressure[generic-pass-layout-conversion-pressure-chain0]`.

### FZ-20260421-0003: ld/st descriptor-view chain1 miscompiles

- Source: Lane A finding `A1`.
- Failure class: `miscompile`.
- Family: `ldst`.
- Minimal shape: parent `[2, 64, 32]`, indexed view `[64, 32]`.
- Dtypes: stable for `f32`; also reproduced for `i32`; some `f16` compiling
  variants mismatch.
- View chain:
  `index(1).reshape((M//2,2,N)).permute([1,0,2]).reshape((M,N))`.
- Instruction variants: `32x32b`, `16x64b`, and `16x128b`.
- Observed: runtime output mismatches PyTorch/reference after successful
  compile.
- Controls: direct chain and other view chains pass for neighbor cases.
- Repro: see `fuzz_ldst_ldred_round1.md` exact inline Python command.
- Checked-in xfail:
  `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_descriptor_view_read[ldst-fz20260421-0003-chain1-64x32-32x32b]`.

### FZ-20260421-0004: ld.red descriptor chains fall back to plain ld plus software reduce

- Source: Lane A finding `A2`.
- Failure class: `opcode_mismatch`.
- Family: `ldred`.
- Minimal shape: parent `[2, 64, 32]`, indexed view `[64, 32]`.
- View chain:
  `index(1).reshape((M//2,2,N)).permute([1,0,2]).reshape((M,N))`.
- Observed: output and reduced tensor match, but PTX contains plain
  `tcgen05.ld.sync.aligned...`, not `.ld.red.`, so hardware reduction
  selection is lost after descriptor-view chains.
- Repro: see `fuzz_ldst_ldred_round1.md` exact command.
- Checked-in xfail:
  `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-chain1-64x32-min]`.

### FZ-20260421-0005: 256-row lifted parent asserts in TensorMemoryAllocation

- Source: Lane A finding `A3`.
- Failure class: `compiler_crash`.
- Family: `ldst` and `ldred`.
- Minimal shape: parent `[2, 256, 32]`, indexed view `[256, 32]`.
- View chain: direct `index(1)`.
- Observed: assertion from `TensorMemoryAllocation.cpp:65`:
  `MemoryBitMap::findFirstFit(...): Assertion 'kNumRows - numRows >= 0' failed.`
- Expected: clean resource diagnostic or supported lowering, not assertion.
- Repro: see `fuzz_ldst_ldred_round1.md` exact command.
- Checked-in xfail:
  `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_256row_lifted_parent_allocator_crash`.
  This case runs the crashing compile in a subprocess so the parent pytest
  process survives the current C++ assertion.

### FZ-20260421-0006: ld.red transpose/slice view may be false unsupported

- Source: Lane A finding `A4`.
- Failure class: `false_unsupported` candidate.
- Family: `ldred`.
- Shape: parent `[2, 64, 128]`, indexed view `[64, 128]`.
- Layout: row `rotate1`, col `identity`.
- View chain:
  `index(1).permute([1,0]).permute([1,0]).slice(0,M,dim=0).slice(0,N,dim=1)`.
- Observed: `view.get_reg_layout()` rejects with row-anchor diagnostic:
  required row anchors `32,64` are not directly representable.
- Status: keep as candidate until planner proves ISA-impossible.
- Checked-in xfail:
  `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0006-rotate1-transpose-slice-max]`.

### Retired Non-Bug: direct two-CTA scales copy cta_group expectation

- Initial local fuzzer expected `copy-scales-warpx4-2cta` to emit
  `cta_group::2`.
- Lane D verified the current runtime-matrix anchor expects and passes with
  `tcgen05.cp.cta_group::1.warpx4.32x128b` for this direct scales-copy shape.
- Classification: stale fuzzer expectation, corrected in the fuzzer.

## Repro Queue

- FZ-20260421-0001 is now covered by checked-in Python runtime xfail repros;
  optional next minimization is a lit `ttg.memdesc_index` crash/clean-error
  contract if repair work needs a compiler-only reproducer.
- FZ-20260421-0002 is now covered by checked-in Python runtime xfail repros
  for dynamic `if`, mixed tensor+memdesc capture, and layout-conversion
  pressure variants.
- FZ-20260421-0003 and FZ-20260421-0004 are now covered by checked-in Python
  runtime xfail repros.
- FZ-20260421-0005 is now covered by a checked-in subprocess xfail. Optional
  next minimization remains capturing the MLIR reproducer and rerunning with
  `triton-opt --run-reproducer`.
- FZ-20260421-0006 is now covered by a checked-in Python xfail. Expand around
  adjacent row/col permutations before classifying as a true boundary.
