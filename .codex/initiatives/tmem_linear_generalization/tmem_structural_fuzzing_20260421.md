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

Round 6 Lane D schema recommendation: split the durable descriptor from
family-specific runnable kernels. A normalized descriptor should carry
`family`, `shape`, `dtype`, `two_cta`, `num_ctas`, `view_chain`, `row_kind`,
`col_kind`, `reg_variant`, `expected_class`, stable `seed`, and optional
`legacy_case_id`/`source_case_id`. Keep descriptor enumeration CPU-only so
coverage inventory can run without importing Gluon; runnable adapters can
remain family-specific and consume a bounded subset of the inventory.

## Round Log

### Round 32 Local, FZ-0018 ld.red minimization

- Time: 2026-04-21 13:10 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_fz0018_min_round32.md`
- Scope: subprocess-isolated Python/Gluon runtime minimization for
  `FZ-20260421-0018`, sweeping `M={96,128,160,256}`,
  `N={256,384,512}`, `num_warps={4,8}`, direct/indexed/chain roots,
  same-footprint descriptor-view chains, explicit `32x32b`, row/column/tile
  layout perturbations, and `min`/`max`.
- Result: no backend/test source edits. `60` rows classified as `16`
  `FZ-20260421-0018`, `12` pass, `6` clean shared-memory boundaries, `6`
  clean tensor-memory boundaries, and `20` clean power-of-two shape
  boundaries. `FZ-0018` is not direct-only: same-footprint descriptor-view
  chains reproduce it. It is not a generic high-`N` impossibility:
  `M64xN512` and `M128xN512` column-reversed hardware `.ld.red` pass. It also
  generalizes to `M256xN256,w4`; matching `w8` rows diagnose clean
  shared-memory OOR.

### Round 33 Local, FZ-0018 PTX artifact capture

- Time: 2026-04-21 13:13 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_fz0018_ptx_round33.md`
- Scope: preserve standalone IR/PTX artifacts for the failing direct
  `M128xN512,w4` FZ-0018 row and replay ptxas outside the Python runtime
  harness.
- Result: `TRITON_REPRODUCER_PATH` produced no MLIR reproducer because the
  failure is ptxas/cubin-stage. `TRITON_KERNEL_DUMP=1` preserved
  `direct_kernel.ttgir`, `direct_kernel.llir`, and `direct_kernel.ptx` under
  `/tmp/tmem_fz0018_dump/...`. Standalone `ptxas-blackwell` replay on the
  dumped PTX exits `255` with the FZ-0018 register-allocation failure. Passing
  dumped controls show `M128xN512` column-reversed hardware `.ld.red` assembles
  with `96` registers/eight `.ld.red` PTX occurrences and `M64xN512` assembles
  with `255` registers/four `.ld.red` occurrences, so the issue is not simply
  "all 255-register kernels fail."

### Round 34 Local, copy warpx2/warpx4 runtime guardrail

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_copy_warpx2_round34.md`
- Scope: checked-in no-scale `tcgen05.copy` `warpx2`/adjacent runtime rows
  selected by
  `(cp_no_scales and (warpx2 or warpx4 or dense_shared or twocta_128x128b) and not resource)`.
- Result: `83/1615` rows collected and passed split-4 as `83 passed`
  (`21/21/21/20`). No compiler crash, false unsupported diagnostic, opcode
  mismatch, runtime miscompile, clean-boundary drift, unexpected
  xfail/pass transition, or new independent `FZ-*`.

### Round 32 Lane, Subword/Narrow-Shape Python Runtime TMEM

- Time: 2026-04-21 12:57 UTC
- Reports:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_subword_narrow_round32.md`
  and
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_sub32_python_round32.md`
- Scope: Python/Gluon runtime fuzzing around `f16`, `bf16`, `i16`, `i8`,
  `f32`, and `i32` TMEM load/store/copy paths, narrow N and x1 packed forms,
  descriptor chains, indexed/subslice views, 1CTA/2CTA layouts, and clean
  diagnostics for packed/subword hardware or layout boundaries.
- Result: no new independent `FZ-*` bucket. Split-4 subword/copy/diagnostic
  sweep passed as `79 passed`; supported f32/i32 narrow/control sweep passed
  as `60 passed`; sibling temporary Python/Gluon dtype roundtrip probe passed
  as `80 passed`. Combined evidence: `219` rows, with no compiler crash,
  false unsupported diagnostic, opcode mismatch, or runtime miscompile. The
  bitwidth crash cataloged by `FZ-20260421-0017` remains isolated to encoded
  `i64`/`f64` non-reduction TMEM load/store lowering under the covered
  neighborhood.

### Round 32 Local, ld/st narrow and half-row runtime guardrail

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_ldst_narrow_round32.md`
- Scope: checked-in runtime rows around `ld/st` narrow `N=32`, half-row,
  `i32` broad-layout, and subword-adjacent `16x128b` / `16x256b` variants.
- Result: no new independent `FZ-*` bucket. Selector
  `ldst and (n32 or half_rows or x1 or i32_broad) and not reports and not resource and not clean and not roundtrip`
  collected `141/1615` rows and passed split-4 as `141 passed`.

### Round 32 Local, plain-MMAv5 accumulator/view guardrail

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_mma_accumulator_round32.md`.
- Scope: checked-in runtime rows selected by
  `tmem_runtime_matrix_mma and not ldst and not mma_scaled and not reports and not resource and (m64 or twocta or indexed_acc_view or acc_subslice_view)`.
- Result: no new independent `FZ-*` bucket. Selector collected `185/1615`
  rows and passed split-4 as `185 passed`.

### Round 32 Local, TMEM Lit / Compiler-Only Guardrail

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_lit_round32.md`.
- Scope: `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir
  test/TritonNvidiaGPU/invalid.mlir test/Conversion/relayout_tritongpu.mlir`
  after `ninja triton-opt`.
- Result: `tmem_layouts.mlir` and `invalid.mlir` passed;
  `relayout_tritongpu.mlir` failed with existing `FZ-20260421-0016`.
- Classification: no new independent `FZ-*` bucket.

### Round 32 Local, Clean-Boundary Runtime Guardrail

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_clean_boundary_round32.md`.
- Scope: checked-in runtime rows selected by
  `(reports_clean or clean_unsupported or clean_error or tmem_oor or clean_xfail) and not resource`.
- Result: no new independent `FZ-*` bucket. Selector collected `184/1615`
  rows and passed split-4 as `184 passed`.

### Round 31 Lane, ld.red descriptor/layout extremes

- Time: 2026-04-21 12:54 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_extremes_round31.md`
- Scope: temporary runtime probe for f32 `ld.red` M/N extremes, row/column
  basis permutations, explicit instruction variants, indexed descriptor roots,
  descriptor-view chains, invalid 2CTA ownership, clean resource boundaries,
  and opcode consistency.
- Result: new candidate `FZ-20260421-0018`. The temporary matrix produced
  `15` passes, `3` existing `FZ-20260421-0012`, `1` existing
  `FZ-20260421-0010`, `2` clean resource boundaries, and `2`
  `FZ-20260421-0018` rows. Checked-in `ld_red and not reports and not resource`
  stayed stable as `237 passed, 6 failed`, all known `FZ-0012`.

### Round 32 Local, scaled-MMAv5 2CTA runtime guardrail

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_twocta_round32.md`
- Scope: checked-in runtime-matrix selector
  `mma_scaled and twocta and not reports and not resource`.
- Result: no new independent `FZ-*` bucket. The selector passed split-4 as
  `28 passed`, keeping legal 2CTA scaled-MMAv5 rows green next to existing
  report-only `FZ-20260421-0007`, `FZ-20260421-0013`, and
  `FZ-20260421-0015`.

### Round 31 Local, scales-copy and clean-boundary guardrail

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_scales_copy_clean_round31.md`
- Scope: selector
  `(cp_scales or mbarrier or proxy or clean_error or clean_unsupported) and not reports and not resource`.
- Result: no new independent `FZ-*` bucket. The selector collected `55/1615`
  and passed split-4 as `55 passed`, covering scales copy rows and clean
  unsupported/error diagnostics adjacent to Round 31 scaled operand and
  compiler-boundary fuzzing.

### Round 31b Compiler Boundary Follow-Up, reduction and scale layouts

- Time: 2026-04-21 13:00 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_compiler_boundaries_round31b.md`
- Scope: compiler-only MLIR probes for live `ttng.tmem_load` reductions,
  tensor-memory scale layouts, dynamic `ttg.memdesc_index` into reduction and
  scale consumers, 64-bit reduction diagnostics, and scale copy boundaries.
- Result: no new independent `FZ-*` bucket. Across `8` MLIR cases and three
  compiler modes, the matrix produced `6` passes, `15` clean diagnostics or
  known late illegal-op failures, and `3` existing unencoded-tensor verifier
  aborts. Dynamic indexed reduction/scale loads expand `FZ-20260421-0001`;
  unencoded reduction load results expand `FZ-20260421-0016`; non-f32
  reduction and scale-copy wrong-source/layout rows stayed clean diagnostics.

### Round 31 Lane, scaled-MMAv5 TMEM descriptor operands

- Time: 2026-04-21 12:49 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_operand_round31.md`
- Scope: temporary Python/Gluon probes for scaled-MMAv5 A/B scale operands,
  scale descriptor-view chains, dynamic direct B-scale selection, A-scale
  selection controls, same-object B-scale controls, branch-contained B-scale
  MMA controls, accumulator subslice/multi-MMA controls, and 4/8/16 CTA
  high-CGA scaled controls.
- Result: no new independent `FZ-*` bucket. Checked-in
  `mma_scaled and descriptor_view and not reports` controls passed split-4 as
  `4 passed`. The dynamic scale-view probe ran `16` rows with `0` unexpected
  outcomes: `10` direct selected scale rows passed, and `6` descriptor-view
  rows reproduced `FZ-20260421-0013`. The direct dynamic-scale pytest slice
  reported `2 failed, 7 passed`; both failures are existing
  `FZ-20260421-0015` runtime-selected distinct B-scale rows. The
  descriptor-view/multi-MMA slice reported `3 failed, 7 passed`; all failures
  are existing `FZ-20260421-0013`. High-CGA scaled controls passed for
  `num_ctas=4/8/16`; adding local 1CTA/2CTA `st`, `ld`, `ld.red`, or `copy`
  reproduced existing `FZ-20260421-0010` in `24` rows.

### Round 31 Local, Descriptor Mix / View Copy / Structural Rerun

- Time: 2026-04-21 12:50 UTC
- Reports:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_descriptor_mix_round31.md`,
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_view_copy_round31.md`,
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_python_descriptor_views_round31.md`,
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_compiler_boundaries_round31.md`,
  and
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_structural_rerun_round31.md`.
- Scope:
  - descriptor-heavy `ld.red` plus scaled-MMAv5 rows selected by
    `(mma_scaled or ld_red or cp_no_scales or copy) and descriptor and not reports and not resource and not clean`;
  - high-rank `ld/st` descriptor and `tcgen05.copy` `warpx2`
    indexed/subslice/slice-index rows selected by
    `(warpx2 or rank5 or multidim_slice or half_rows or slice_index or indexed_view) and not reports and not resource and not clean and not lifted`;
  - Python/Gluon descriptor-sliced 64-bit TMEM load/store frontend
    reachability for `FZ-20260421-0017`;
  - compiler-only verifier/index boundary MLIR probes for dynamic
    `ttg.memdesc_index`, unencoded tensors, wrong memory spaces, descriptor
    view errors, and 64-bit load/store operands;
  - checked-in structural fuzzer strict xfail catalog.
- Results:
  descriptor mix `54 passed`; view/copy `123 passed, 74 skipped`;
  Python descriptor views `48` `ASSERT_BITWIDTH_32`, `40` pass, and `20`
  Python/compile diagnostics; compiler-boundary matrix `18` passes, `23`
  clean diagnostics, `12` assertion/stack-dump aborts, and `1` late illegal-op
  failure across `18` MLIR cases and three pass modes;
  structural fuzzer `9 passed, 24 xfailed`.
- Classification: no new independent `FZ-*` bucket. These are green guardrails
  plus expansions of existing `FZ-20260421-0001`, `FZ-20260421-0016`, and
  `FZ-20260421-0017`, while Round 31 subagents continue adversarial
  `ld.red` extreme and descriptor-chain-shape lanes.

### Round 26 Local, ld.red Runtime Slice

- Time: 2026-04-21 15:05 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_ldred_round26.md`
- Scope: checked-in runtime-matrix `ld.red` rows selected by
  `ld_red and not reports and not resource`.
- Result: no new independent `FZ-*` bucket. The selector collected `243` rows;
  split-4 execution produced `237` passing rows and `6` failures. All failures
  are the known `FZ-20260421-0012` M64 f32 row-permuted destination-layout
  planner gap (`unsupported dst layout`).
- Note: this run intentionally stayed in discovery/cataloging mode; no backend
  repair was attempted.

### Round 26 Local, MMAv5-Adjacent Runtime Guardrail

- Time: 2026-04-21 15:20 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_mma_guard_round26.md`
- Scope: checked-in runtime-matrix rows selected by
  `mma and not mma_scaled and not reports and not clean and not resource`.
- Result: no new independent `FZ-*` bucket. The selector collected `337` rows
  and completed split-4 as `323 passed, 14 skipped`.
- Note: this is a green guardrail for plain-MMAv5 and MMAv5-adjacent descriptor
  and copy setup rows while Round 26 subagents fuzz more adversarial dynamic
  descriptor, verifier, and high-CGA mixed-ownership surfaces.

### Round 26 Lane, TMEM Verifier Robustness

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_verifier_round26.md`
- Scope: `ttng.tmem_alloc`, `ttng.tmem_load`, `ttng.tmem_store`, and
  `ttng.tmem_copy` verifier behavior for unencoded tensors, malformed memdesc
  encodings, wrong memory spaces, wrong shapes, and clean diagnostic
  boundaries.
- Result: no new independent `FZ-*` bucket. Existing `FZ-20260421-0016` now
  covers the shared `verifyTMEMOperand` / compatibility path for unencoded
  tensor operands/results through alloc, load, and store. `ttng.tmem_copy`
  stayed on clean memdesc diagnostics for the probed wrong-space/shape cases.
- Counts: primary corpus `6` dyn-cast aborts, `12` clean diagnostics, `2`
  passes; follow-up corpus `6` dyn-cast aborts and `6` clean diagnostics.

### Round 27 Lane, Checked-In Structural Fuzzer Rerun

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_structural_rerun_round27.md`
- Scope: full checked-in `python/test/gluon/test_tmem_structural_fuzzer.py`
  rerun across four GPUs with stable per-GPU caches.
- Result: stable expected catalog, `9 passed, 24 xfailed`, no failures and no
  XPASS. Known xfail output still includes the `FZ-20260421-0001` illegal
  `ttg.memdesc_index` LLVM-conversion repros.

### Round 26 Lane, Dynamic Descriptor SSA / Control Flow

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_dynamic_ssa_round26.md`
- Scope: temporary Python/Gluon probes for nested-helper, branch-yielded,
  tuple-like, loop-carried, dynamic-index, mixed-consumer, copy, `ld/st`,
  `ld.red`, and plain-MMAv5 accumulator descriptor SSA paths.
- Result: no new independent `FZ-*` bucket. Runtime-index `ld.red` and
  branch-yielded copy rows broaden existing `FZ-20260421-0001` illegal
  `ttg.memdesc_index` lowering. Nested-helper/loop-carried `ld/st` and
  branch/mixed-consumer `ld.red` rows broaden existing `FZ-20260421-0002`
  descriptor SSA/control-flow wrong results. The `ld.red` rows emitted
  `.ld.red`, so they are not opcode-fallback `FZ-20260421-0004` cases.
- Positive control: branch-selected plain-MMAv5 accumulator descriptor passed.

### Round 26 Lane, High-CGA Scaled-MMAv5 Mixed Ownership

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_high_cga_scaled_round26.md`
- Scope: legal high-CGA scaled-MMAv5 controls mixed with local 1CTA/2CTA TMEM
  descriptor-view `st`, `ld`, `ld.red`, and `tcgen05.copy` operations in
  4/8/16 CTA launch contexts.
- Result: no new independent `FZ-*` bucket. The Round 25 scaled-MMAv5 harness
  limitation was removed; high-CGA scaled-MMAv5 controls passed for 4/8/16 CTA
  launches with expected scaled opcodes and torch reference matches. Every
  mixed local 1CTA/2CTA operation failed at the known `FZ-20260421-0010`
  layout/context CTA-count ownership gate.

### Round 28 Lane, Broad ld.red Runtime Rerun

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_round28.md`
- Scope: rerun of `ld_red and not reports and not resource` plus exact
  failing-node rerun.
- Result: no new independent `FZ-*` bucket. Aggregate stayed
  `237 passed, 6 failed`; exact failing-node rerun reproduced the same six
  `FZ-20260421-0012` M64 row-permuted `unsupported dst layout` rows.

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

### Lane J Round 10, Descriptor-View Composition Depth

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_descriptor_depth_round10.md`
- Scope: subprocess-isolated `/tmp` fuzzing for deeper `ld/st` and `ld.red`
  descriptor-view chains: `reshape/permute/reshape`, double transpose,
  slice/slice mixes, rank-4/rank-5 parents, sibling views, and immediate
  readback.
- Result: no new independent `FZ-*` bucket. The final sweep had `7` passes
  and `3` clean diagnostics. Rank-4 and rank-5 `ld/st` rows hit the existing
  unsupported direct-layout register-materialization boundary; rank-5 sibling
  `ld.red` hit clean tensor-memory OOR (`2048` required, hardware limit
  `512`).
- Follow-up: keep using these clean boundaries as descriptor-depth guardrails;
  no backend repairs are warranted from this lane during discovery mode.

### Lane I Round 10, LD.RED Modifier and NaN Edges

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_modifiers_round10.md`
- Scope: subprocess-isolated `/tmp` fuzzing of `tcgen05.ld.red` min/max,
  `abs`, NaN propagation, first/last/multi-payload NaN placement, 1CTA and
  2CTA direct/indexed/chained descriptor reductions, and opcode suffix checks
  for `.min`, `.max`, `.abs`, `.NaN`, and `.f32`.
- Result: no new independent `FZ-*` bucket. The `266` rows reclassified as
  `140` pass, `70` opcode fallback rows broadening `FZ-20260421-0004`,
  `28` optimizer abort rows broadening `FZ-20260421-0008`, and `28` clean
  OutOfResources diagnostics. No runtime miscompile, allocator assertion, or
  clean `.x1` diagnostic was found.
- Follow-up: keep `FZ-20260421-0004` and `FZ-20260421-0008` as the owners for
  these modifier/NaN variants; no backend repairs during discovery mode.

### Lane K Round 10, Allocator and Resource Boundaries

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_allocator_resource_round10.md`
- Scope: subprocess-isolated probes around live parent vs selected child size,
  direct index/slice/subslice chains, `M` around `64/128/256/512`, `N` around
  `32/64/128/256`, `ld/st`, `ld.red`, copy, and scaled-MMAv5 accumulator
  descriptor paths.
- Result: no new independent `FZ-*` bucket. Fresh probes map to existing
  buckets: allocator/compiler failures for `ld/st` and `ld.red` 256-row and
  larger parents under `FZ-20260421-0005/0009`, f16 chain2 `ld/st` descriptor
  miscompile under `FZ-20260421-0003`, and 2CTA indexed `ld.red` opcode
  fallback under `FZ-20260421-0004`.
- Clean boundaries: 2CTA copy `256x256` OOR and scaled-MMA tile-permuted
  `N=16` unsupported diagnostic.

### Lane L Round 10, Generic Pass and Control Flow

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_generic_controlflow_round10.md`
- Scope: exact four-GPU replay of checked-in structural-fuzzer cases through a
  `/tmp` runner, stressing nested helpers, tuple-like returns, sibling views,
  loops with memdesc/tensor iter args, dynamic `if`/select, mixed TMEM and
  non-TMEM tensors, layout conversions, runtime scalar indices, `ld/st`,
  `ld.red`, and one scaled-MMA accumulator descriptor row.
- Result: no new independent `FZ-*` bucket. The exact rerun produced `7`
  passes, `1` clean unsupported diagnostic, and `20` failures owned by
  existing buckets: `FZ-20260421-0001`, `FZ-20260421-0002`,
  `FZ-20260421-0003`, `FZ-20260421-0004`, `FZ-20260421-0006`,
  `FZ-20260421-0007`, and `R5-C`.
- Follow-up: keep using this exact checked-in-case replay as a compact generic
  control-flow bucket guard; no backend repairs during discovery mode.

### Lane M Round 10, Copy / Scales / Multicast / CGA

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_copy_scales_cga_round10.md`
- Scope: temporary `/tmp` subprocess fuzzing of no-scales `warpx2::{01_23,02_13}`
  copy, scales `warpx4` copy, source/destination descriptor views,
  indexed/subslice/slice-index chains, subword rows, two-CTA layouts in larger
  CGA contexts, and scaled-MMA copy/multicast setup rows.
- Result: no new `FZ-*` bucket. Direct/source-subslice scales `warpx4`,
  no-scales `warpx2` single-CTA, and scaled-MMA copy setup rows stayed
  positive. Scales descriptor-view, no-scales two-CTA `warpx2::02_13`,
  subword, and larger-CGA two-CTA-context rows stayed clean boundaries.
- Follow-up: keep the scaled-MMA `num_ctas > 2` `make_scales_descriptor`
  shape-divisibility assertion classified as an early guardrail for now; exact
  runtime-matrix rerun ruled out the temporary-wrapper two-CTA `01_23` false
  lead (`2/2` passed).

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

### Lane Expansion Round 2, Adjacent Structural Fuzzing

- Time: 2026-04-21 08:33 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_expansion_round2.md`
- Scope: expand around first-round failures with temporary Python/Gluon runtime
  harnesses under `/tmp`, focused on helper-returned view chains, runtime
  memdesc values, row/col permutations around A1/A4, resource-valid 2CTA
  ld/st and ld.red, and ld.red opcode selection under descriptor chains.
- Result: found adjacent stable variants for existing catalog buckets:
  runtime `memdesc_index` crashes now reach chain2/chain3, helper-returned
  chain0 miscompiles include false-branch and `16x64b` rows, ld/st chain2
  col-reverse `16x64b` miscompiles, and ld.red plain-load fallback reproduces
  across broader descriptor-chain and 2CTA rows.
- Validation:
  - required `make -j8` completed with ninja reporting no work to do;
  - control-flow/helper sweep ran as four pytest-split groups;
  - ld/st and ld.red adjacency sweep ran as four pytest-split groups;
  - representative failures were rerun as fresh single-nodeid pytest
    processes and remained stable.

### Lane R3-B Round 3, ld.red Opcode Loss Expansion

- Time: 2026-04-21 08:39 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_opcode_round3.md`
- Scope: expand `FZ-20260421-0004` around ld.red descriptor-chain plain-load
  fallback, including resource-valid 2CTA rows and nearby descriptor-view /
  layout shapes.
- Result: no backend repair attempted; found a tighter boundary and one
  recommended additional strict xfail sentinel.
- Key findings:
  - 1CTA `parent.index(1)` chain0 still emits hardware `ld.red`, while chain1,
    chain2, chain4, abs, NaN-propagating, and adjacent row/col variants over
    smaller packet families emit plain `tcgen05.ld...` despite correct output;
  - 1CTA `[128,64]` chain1 remains a positive and emits `ld.red`, so the
    failure is shape/layout packet-family dependent rather than all descriptor
    chains;
  - 2CTA full-parent reduction emits `ld.red`, but resource-valid
    `parent.index(1)` emits plain `tcgen05.ld...` even at chain0 for
    `[2,256,32]` and `[2,256,64]`, isolating the 2CTA failure to indexed
    descriptor-view provenance rather than global 2CTA reduction support.

### Lane R3-A Round 3, Dynamic `memdesc_index` Minimization

- Time: 2026-04-21 08:39 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_memdesc_index_round3.md`
- Scope: minimize `FZ-20260421-0001` and determine whether descriptor-view
  chains are required for the illegal dynamic `ttg.memdesc_index`.
- Result: no backend repair attempted; minimized to direct runtime
  `parent.index(tt.load(selector))` on parent `[2,128,32]`, selected view
  `[128,32]`, with no helper return, control-flow merge, pre-store, reshape,
  or permute.
- Validation:
  - required `make -j8` reported no work to do;
  - temporary Python/Gluon probes showed static/constexpr direct index rows
    pass while runtime direct index fails in `ConvertTritonGPUToLLVM`;
  - the compiler-emitted load-only MLIR reproduces with
    `triton-opt --run-reproducer`.
- Promotion: added checked-in strict xfail
  `generic-pass-dynamic-index-load-only-128x32`.

### Lane R3-C Round 3, MMA / Copy / Membar Clean-Surface Fuzzing

- Time: 2026-04-21 08:42 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_mma_copy_round3.md`
- Scope: adversarially probe copy, plain MMA, scaled MMA, larger-CGA clean
  diagnostics, and warp-specialized/membar surfaces outside the currently
  failing ld/st and ld.red buckets.
- Result: no new backend failures.
- Validation:
  - required `make -j8` reported no work to do;
  - runtime-matrix selector over copy `warpx2`, plain two-CTA MMA
    accumulator views, and scaled-MMA descriptor/narrow/tile-permuted surfaces
    passed as `49 + 49 + 49 + 47`;
  - structural-fuzzer anchors passed as `3 passed, 5 xfailed`;
  - temporary larger-CGA probes for `num_ctas=4/8/16` produced clean
    diagnostics for copy, plain MMA, and scaled MMA;
  - warp-specialized/membar probes passed `2`.

### Lane R3-D Round 3, ld/st Descriptor-View Miscompile Minimization

- Time: 2026-04-21 08:43 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldst_round3.md`
- Scope: minimize and expand `FZ-20260421-0003` with read-only descriptor-view
  loads, separating them from roundtrip patterns that can mask address
  arithmetic bugs.
- Result: no backend repair attempted; smallest runtime miscompile remains
  parent `[2,64,32]`, indexed view `[64,32]`, `f32`, chain1, identity layout,
  `32x32b`. Smaller `16x32` and `32x32` rows cleanly reject before launch.
- Additional stable same-bucket evidence:
  - chain2 and chain4 col-reverse `16x64b` rows miscompile with matching
    packet-order samples/opcodes;
  - direct chain0 read-only controls pass;
  - same-view roundtrip controls can pass because the bad mapping cancels.
- Promotion: added checked-in strict xfail
  `ldst-fz20260421-0003-chain2-col-reverse-64x32-16x64b`.

### Round 3 Local Promotion / Validation

- Time: 2026-04-21 08:43 UTC
- Promoted three round-three sentinels into
  `python/test/gluon/test_tmem_structural_fuzzer.py` without backend repairs:
  - minimized direct dynamic-index load-only `FZ-20260421-0001`;
  - ld/st chain2 col-reverse `16x64b` `FZ-20260421-0003`;
  - 2CTA resource-valid indexed-view ld.red opcode loss
    `FZ-20260421-0004`.
- Validation:
  - required `make -j8` reported no work to do;
  - `PYTHONPATH=.:./python python -m py_compile python/test/gluon/test_tmem_structural_fuzzer.py`;
  - collect-only found `21` structural-fuzzer nodeids;
  - the three new exact nodeids each reported `1 xfailed`;
  - full structural fuzzer reported `9 passed, 12 xfailed`.

### Lane R4-B Round 4, Dynamic `memdesc_index` Lit Candidate

- Time: 2026-04-21 08:47 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_memdesc_index_lit_round4.md`
- Scope: produce a compact MLIR/lit minimization candidate for
  `FZ-20260421-0001` direct runtime TMEM `memdesc_index`, without backend
  repair.
- Result: no backend repair attempted; minimized the compiler-only reproducer
  to a live dynamic scalar index into a `[2,128,32]` TMEM descriptor plus a
  `tmem_load`/store use. An unused `ttg.memdesc_index` is eliminated and does
  not reproduce.
- Boundary:
  - dynamic scalar-argument and selector-load forms fail under
    `triton-opt --run-reproducer` with illegal `ttg.memdesc_index`;
  - static `%c1` lowers successfully through the same reproducer pipeline;
  - small dynamic `[64,32]` still fails earlier with the clean row-anchor
    unsupported diagnostic;
  - `--verify-diagnostics` treats the current illegal-op as an unexpected
    compiler error.
- Recommendation: keep the checked-in coverage as Python strict xfail for
  now. The 26-line `/tmp/tmem_memdesc_index_r4_lit_candidate.mlir` is ready as
  a GPU-free repair target, but should only be checked in as lit once the
  intended contract is explicit: clean negative diagnostic if unsupported, or
  temporary `XFAIL`/future positive lowering if dynamic indexing is meant to
  be supported.

### Lane R4-A Round 4, Helper/Control-Flow Expansion

- Time: 2026-04-21 09:00 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_helper_cf_round4.md`
- Scope: expand and minimize `FZ-20260421-0002` through false-branch control
  flow, mixed tensor+memdesc captures, tuple-like captures, 16x64b/16x128b
  layout pressure, chain1/chain2 boundaries, nested helper returns, and
  no-helper inline comparisons.
- Result: no backend repair attempted; found additional stable same-bucket
  chain0 miscompiles and a tighter boundary.
- Key findings:
  - false-branch chain0 reaches `16x128b` and `[128,128]` with stable
    runtime mismatches;
  - tuple-like `(memdesc_view, tensor_bias)` capture reproduces the chain0
    mixed-capture miscompile;
  - layout-conversion pressure reaches `16x128b`;
  - helper boundaries are not required: helper, nested helper, dispatcher-body,
    and true no-helper inline chain0 all fail with the same `8064/8192`
    mismatch pattern;
  - chain1 and chain2 rows pass across the same control-flow, tuple-capture,
    `16x64b`, `16x128b`, and layout-pressure probes.
- Validation:
  - required `make -j8` reported no work to do;
  - temporary harness collected `24` nodeids;
  - four-GPU sweep reported `11 failed, 13 passed`;
  - representative failing and passing nodeids were rerun in fresh pytest
    processes.
- Recommended additional strict xfail sentinels:
  `generic-pass-dynamic-if-chain0-false-16x128b`,
  `generic-pass-dynamic-if-chain0-inline`,
  `generic-pass-tuple-mixed-captures-chain0`, and
  `generic-pass-layout-conversion-pressure-chain0-16x128b`.

### Lane R4-D Round 4, 2CTA Indexed ld.red Opcode Selection

- Time: 2026-04-21 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_2cta_round4.md`
- Scope: expand `FZ-20260421-0004` around resource-valid 2CTA indexed-view
  provenance, non-min reductions, abs/NaN modifiers, row/col permutations,
  descriptor chain0/1/2, and resource/clean diagnostic boundaries.
- Result: no backend repair attempted; all 37 rows that reached runtime and
  opcode inspection passed the runtime oracle but emitted plain
  `tcgen05.ld.sync...` instead of hardware `tcgen05.ld.red...`.
- Key findings:
  - direct 2CTA indexed `max`, `abs`, and NaN-propagating reductions lose
    `ld.red` just like the existing checked-in `min` sentinel;
  - the plain-load fallback spans `N=32/64/128`, chain0/1/2, and row/col
    layout permutations that are runtime-correct;
  - row/col permutation chain1 variants found an adjacent compiler crash in
    `TritonNvidiaGPUOptimizeTMemLayoutsPass` with a row/col vs row/col/block
    dimension mismatch;
  - `M=512,N=32` produced a clean TMEM encoding/CGA-layout diagnostic boundary.
- Validation:
  - required `make -j8` reported no work to do;
  - temporary probe collected `42` nodeids;
  - four-GPU sweep reported `11 + 11 + 11 + 9` discovery failures;
  - representative opcode mismatches, compiler crash, and diagnostic boundary
    were rerun in fresh pytest processes;
  - existing checked-in 2CTA indexed sentinel reported `1 xfailed`, and the
    full-parent 2CTA positive reported `1 passed`.
- Recommended additional strict xfail sentinels:
  `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-max`,
  `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min-abs`, and
  `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min-nan`.

### Lane R4-C Round 4, ld/st Read-Only Descriptor-View Fuzzing

- Time: 2026-04-21 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldst_readonly_round4.md`
- Scope: expand `FZ-20260421-0003` around read-only descriptor-view loads
  across dtype/subword variants, packet families, M/N boundaries, two-CTA
  lifted views, and roundtrip masking controls.
- Result: no backend repair attempted; found one non-overlapping subword
  sentinel and several same-root stability rows.
- Key findings:
  - `f32`/`i32` chain1 rows reproduce the existing row-stripe failure;
  - `f16` chain2 identity `64x32 16x64b` miscompiles even though f32/i32
    chain2 identity was green, making it a distinct subword read candidate;
  - f8 legal smoke rows passed;
  - direct two-CTA indexed-view controls passed, while chained two-CTA
    descriptor views rejected cleanly;
  - same-view roundtrip controls still mask the read-only address bug.
- Recommended additional strict xfail sentinel:
  `ldst-fz20260421-0003-f16-chain2-identity-64x32-16x64b`.

### Round 4 Local Promotion / Validation

- Time: 2026-04-21 08:50 UTC
- Promoted eight round-four sentinels into
  `python/test/gluon/test_tmem_structural_fuzzer.py` without backend repairs:
  - `generic-pass-dynamic-if-chain0-false-16x128b`;
  - `generic-pass-dynamic-if-chain0-inline`;
  - `generic-pass-tuple-mixed-captures-chain0`;
  - `generic-pass-layout-conversion-pressure-chain0-16x128b`;
  - `ldst-fz20260421-0003-f16-chain2-identity-64x32-16x64b`;
  - `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-max`;
  - `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min-abs`;
  - `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min-nan`.
- Validation:
  - `PYTHONPATH=.:./python python -m py_compile python/test/gluon/test_tmem_structural_fuzzer.py`;
  - collect-only found `29` structural-fuzzer nodeids;
  - the eight new exact nodeids reported `8 xfailed` across four GPUs;
  - full structural fuzzer reported `9 passed, 20 xfailed`;
  - `git diff --check` passed.

### Lane R5-B Round 5, Copy / ld.red Descriptor-View Interactions

- Time: 2026-04-21 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_copy_ldred_interactions_round5.md`
- Scope: combined copy plus readback probes not already covered by standalone
  copy or standalone ld/st and ld.red lanes. The lane copied into
  descriptor-indexed and subslice TMEM parents, then forced TMEM `ld/st` or
  `ld.red` readback; it also covered single-CTA and two-CTA `warpx2`
  descriptor views, two-CTA `warpx2::02_13` clean diagnostics, and larger-CGA
  two-CTA-layout diagnostics.
- Result: no new backend compiler crash, false unsupported diagnostic, opcode
  mismatch, or runtime miscompile found.
- Validation:
  - required `make -j8` reported no work to do;
  - temporary probe collected `9` nodeids;
  - four-GPU split sweep passed as `3 passed`, `3 passed`, `3 passed`, and
    `0 selected`;
  - opcode summary confirmed `copy -> indexed view -> ld/st` emits
    `tcgen05.cp.cta_group::1.128x256b` plus TMEM `ld/st`;
  - `copy -> subslice view -> ld.red` emits
    `tcgen05.ld.red.sync.aligned.32x32b.x128.min.f32`;
  - `warpx2::01_23` and `warpx2::02_13` single-CTA indexed descriptor-view
    readback passed, two-CTA `warpx2::01_23` indexed readback passed, and
    two-CTA `warpx2::02_13` indexed view remained a clean unsupported
    diagnostic;
  - `num_ctas=4/8/16` larger-CGA contexts for a two-CTA indexed copy parent
    reported clean layout-context diagnostics without assertions.

### Lane R5-C Round 5, Generic-Pass Loop-Carried Memdesc Views

- Time: 2026-04-21 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_generic_pass_round5.md`
- Scope: generic-pass stress on TMEM descriptor-view values carried through
  loops, nested conditionals, multiple live sibling views, and helper-returned
  chains.
- Result: found a distinct compiler-crash bucket in
  `GluonResolveAutoEncodingsPass`. Loop-carried memdesc views and related
  generic-pass forms fail auto-layout inference for `tt.make_range` before
  they can reach runtime execution.
- Checked-in sentinel:
  `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_loop_carried[generic-pass-loop-carried-memdesc-view-chain0]`.
- Classification: compiler crash / false failure for resource-valid TMEM
  descriptor-view control flow. Backend repair is deferred during the active
  discovery campaign.
- Validation:
  - required `make -j8` reported no work to do;
  - py-compile passed for `test_tmem_structural_fuzzer.py`;
  - collect-only found `30` structural-fuzzer nodeids;
  - the new exact sentinel reported `1 xfailed`;
  - full structural fuzzer reported `9 passed, 21 xfailed`.

### Lane B Round 6, MMAv5 / Scaled-MMAv5 Control-Flow Descriptor Views

- Time: 2026-04-21 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_mma_scaled_controlflow_round6.md`
- Scope: MMAv5 and scaled-MMAv5 TMEM descriptor views under control flow,
  including indexed/subslice accumulator views, `use_acc`, two-CTA and
  larger-CGA contexts, helper-returned descriptors, loop/if-carried
  accumulator descriptors, narrow `N`, and scale layouts.
- Result: found one stable scaled-MMAv5 runtime miscompile, cataloged as
  `FZ-20260421-0007`. No backend/compiler repair attempted.
- Key finding:
  - scaled-MMAv5 `use_acc` over a low-column accumulator subslice selected by
    dynamic `if` miscompiles for `M=128,N=64,K=128` and `mxfp8 x mxfp8`;
  - the same temporary harness passed indexed accumulator `if` and loop rows,
    high-subslice `if`, high-subslice direct, and high-subslice loop rows;
  - existing direct matrix controls for scaled-MMAv5 accumulator subslice
    `use_acc` remained green, so the current evidence points at the dynamic
    `if` descriptor selection plus low-subslice origin interaction.
- Validation:
  - required `make -j8` reported no work to do;
  - scaled-MMAv5 focused matrix sweep passed as `64/64/64/62`;
  - plain MMAv5 focused matrix sweep passed as `44/44/44/43`;
  - larger-CGA / two-CTA-layout diagnostic selector passed `2`;
  - temporary control-flow harness reproduced the exact new failure in a fresh
    process and passed the adjacent controls.

### Lane A Round 7, scaled-MMAv5 accumulator control-flow expansion

- Time: 2026-04-21 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_mma_controlflow_round7.md`
- Scope: expanded `FZ-20260421-0007` across scaled-MMAv5 `use_acc`
  accumulator subslices selected through dynamic `if`, loop, and
  helper-returned paths; compared low vs high subslices, direct vs indexed vs
  subslice selection, `N in {32,64,128}`, `K in {128,256}`, selector values,
  and feasible `mxfp8`/`mxfp4`/`nvfp4` format mixes.
- Result: no new independent `FZ-*` id. The stable runtime miscompiles all
  extend the existing `FZ-20260421-0007` owner surface. Dynamic
  helper-returned indexed-view compiler failures overlap
  `FZ-20260421-0001`.
- Key finding:
  - direct low/high accumulator subslices pass;
  - high-selector dynamic `if`, helper, and indexed rows pass;
  - low-selector dynamic `if`, helper, loop, and indexed rows miscompile
    across the probed `N`, `K`, and scaled-format cells that reach runtime;
  - `indexed_helper` rows fail earlier in lowering through the existing
    dynamic `ttg.memdesc_index` illegal-lowering bucket.
- Validation:
  - required `make -j8` reported no work to do;
  - `/tmp/tmem_scaled_mma_controlflow_round7_probe.py` py-compiled;
  - collect-only found `410` nodeids;
  - four-GPU split sweep completed as `103/103/103/101` classified tests;
  - fresh exact reruns confirmed selector-sensitive miscompile/pass contrasts
    for `subslice_if`, `indexed_if`, `subslice_helper`, and `subslice_loop`.

### Lane D Round 6, Deterministic Structural Generator Prototype

- Time: 2026-04-21 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_generator_round6.md`
- Scope: build a `/tmp` prototype deterministic descriptor generator that
  enumerates compact TMEM case descriptors across family, shape, dtype,
  `two_cta`, `num_ctas`, descriptor-view chain, row/col layout permutation,
  register-layout variant, and expected class. Use it to inventory checked-in
  structural-fuzzer coverage and run a small runtime sample. No backend repair
  was attempted.
- Prototype artifacts:
  - `/tmp/tmem_structural_case_generator_round6.py`;
  - `/tmp/tmem_structural_case_generator_round6.json`.
- Inventory result:
  - generated descriptor total: `25245`;
  - checked-in structural-fuzzer nodeids/case ids: `30`;
  - exact normalized-id overlap: `0`, because the checked-in fuzzer currently
    uses historical repro ids rather than normalized descriptor ids.
- Runtime sample:
  - required `make -j8` completed with Ninja reporting no work to do;
  - collect-only found `30` structural-fuzzer nodeids;
  - representative runtime sample over current `ldst`, `ldred`, copy, and
    generic-pass loop-carried sentinel rows reported `3 passed, 1 xfailed`.
- Coverage/schema recommendations:
  - introduce a repo-local normalized `StructuralCaseDescriptor` before
    promoting the full generator;
  - preserve existing repro pytest ids initially through `legacy_case_id`, but
    add normalized descriptor ids for mechanical inventory diffs;
  - add generator-backed runnable slices in the order `ldred` indexed/direct
    opcode rows, `ldst` read-only descriptor-view rows, copy descriptor-view
    and `warpx2` readback rows, then compact MMA/scaled-MMA descriptor-view
    control-flow rows;
  - treat `num_ctas > 2`, copy non-scales/packed/subword boundaries, and
    MMA/scaled-MMA descriptor-view control-flow contracts as first-class
    inventory gaps, even when their runnable coverage remains in
    `test_tmem_runtime_matrix.py` for now.

### Lane C Round 6, ld.red Optimizer Crash Isolation

- Time: 2026-04-21 09:03 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_crash_round6.md`
- Scope: isolate the report-only R5-A row/col chain1 two-CTA indexed
  `ld.red` optimizer crash family from the already checked-in plain `ld`
  opcode fallback rows and from clean unsupported descriptor-view boundaries.
- Result: confirmed a stable subprocess-isolated optimizer crash and extracted
  an MLIR `triton-opt --run-reproducer` candidate. No backend/compiler repair
  attempted.
- Key findings:
  - `256x2`, chain1, row `even_odd`, col `identity` crashes in
    `TritonNvidiaGPUOptimizeTMemLayoutsPass` for `min`, `max`, `min(abs=True)`,
    and NaN-propagating `min`;
  - `256x64`, chain1, row `identity`, col `reverse` still crashes in the same
    optimizer pass;
  - `256x32`, chain1, row `identity`, col `identity` remains a runtime-correct
    plain `tcgen05.ld...` fallback and belongs to `FZ-20260421-0004`, not this
    crash bucket;
  - `256x2` chain2/chain3 `even_odd` controls also lower as plain `ld`, so
    the optimizer crash is chain1-specific in this probe;
  - `128x32`, chain1, row `even_odd` remains a clean unsupported descriptor
    view boundary.
- Validation:
  - required `make -j8` reported no work to do;
  - temporary probe collected `9` nodeids;
  - parent pytest passed as `9 passed` because each candidate ran in a child
    Python process;
  - direct minimized child command exited `1` with the row/col vs
    row/col/block dimension mismatch;
  - extracted MLIR replay with `triton-opt --run-reproducer` aborted with exit
    `134` in the same optimizer pass.

### Lane D Round 7, Generic-Pass / Analysis Interaction Fuzz

- Time: 2026-04-21 09:11 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_generic_analysis_round7.md`
- Scope: generic-pass and analysis interactions after the new scaled-MMAv5 and
  `ld.red` promotions: memdesc values through nested `scf.if`, `scf.for`,
  tuple-like returns, helper chains, layout conversions, and non-TMEM tensor
  interactions.
- Result: no backend/compiler repair attempted. No new independent FZ id was
  assigned; the stable failures split cleanly into existing buckets:
  - R5-C auto-layout compiler crash extends to tuple-like `(memdesc, tensor)`
    results, nested helper-selected memdesc values, loop-carried memdesc plus
    tensor iter args, chain2 slice/slice loop-carried views, and multiple live
    sibling views with a non-TMEM tensor side input;
  - `FZ-20260421-0002` runtime miscompile survives an additional
    layout-conversion/non-TMEM tensor pressure row, compiling successfully but
    mismatching `8063 / 8192` elements.
- Boundary separation:
  - no `FZ-20260421-0008`-style `TritonNvidiaGPUOptimizeTMemLayoutsPass`
    dimension mismatch appeared in this generic-pass lane;
  - no new `FZ-20260421-0001` dynamic `memdesc_index` illegal-lowering row was
    found;
  - the direct-control row in the temporary harness was contaminated by a
    helper returning a non-TMEM tensor with `#gluon.auto_encoding`, so the
    checked-in `ldst-view-identity-32x32b` pass remains the direct control.
- Validation:
  - required `make -j8` reported no work to do;
  - `/tmp/tmem_generic_analysis_round7.py` py-compiled and collected `10`
    nodeids;
  - four-GPU split sweep reported `3 failed`, `3 failed`, `3 failed`, and
    `1 failed`;
  - fresh exact tuple and loop confirmations reproduced the R5-C
    `GluonResolveAutoEncodingsPass` / `tt.make_range` diagnostic;
  - fresh exact layout-conversion/non-TMEM row reproduced the FZ-0002-style
    runtime mismatch;
  - checked-in `ldst-view-identity-32x32b` control passed.

### Lane C Round 8, Copy/Readback Generator Adapters

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_copy_readback_round8.md`
- Scope: deterministic temporary copy/readback probe with its own case table,
  input generation, opcode checks, and clean-diagnostic classification. It
  covered no-scales `warpx2::01_23` and `warpx2::02_13`, direct/indexed/
  subslice/slice-index TMEM descriptor views, two-CTA `warpx2::01_23`
  positives, two-CTA `warpx2::02_13` clean diagnostics, packed/subword clean
  diagnostics, scales `warpx4` direct/two-CTA positives, scales descriptor
  view clean unsupported boundary, descriptor-chain `ld.red` readback, and
  two-CTA layouts used from `num_ctas` `{4,8,16}` contexts.
- Result: no stable new backend/compiler failure and no new `FZ-*` id.
- Boundary separation:
  - no-scales `warpx2` direct and view copy/readback stayed positive;
  - two-CTA no-scales `warpx2::01_23` direct/view stayed positive;
  - scales `warpx4` direct and two-CTA direct copy/readback stayed positive;
  - descriptor-chain `ld.red` readback emitted
    `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32`;
  - two-CTA no-scales `warpx2::02_13`, f16 packed/subword copy, larger-CGA
    two-CTA-layout contexts, and scales descriptor-view copy all reported
    clean diagnostics rather than assertions or late lowering failures.
- Validation:
  - required `make -j8` reported no work to do;
  - `/tmp/tmem_copy_readback_round8_probe.py` py-compiled;
  - collect-only found `18` nodeids;
  - full temporary probe passed `18 passed`.

### Lane D Round 8, Generic-Pass / Analysis Interaction Fuzz

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_generic_pass_round8.md`
- Scope: temporary runtime probes around TMEM memdesc values through helper
  returns, tuple-like returns, direct controls, sibling views, static indexes,
  `ttgl.static_range`, dynamic `scf.if`, runtime `memdesc_index`, layout
  conversions, and non-TMEM tensor side inputs, while avoiding rows already
  masked by the known R5-C auto-layout crash.
- Result: no new independent `FZ-*` id.
- Key classification:
  - direct chain0 view, static helper tuple, sibling view, static loop, and
    same-base dynamic-if rows all mismatch at roughly `8061-8064 / 8192`
    elements and extend `FZ-20260421-0003`;
  - direct base load, direct chain1 view, static tuple chain1, sibling
    chain1, static-loop chain1, and dynamic-if chain1/chain2 controls pass;
  - runtime direct `parent.index(ttgl.load(selector))` still reaches
    `ConvertTritonGPUToLLVM` as illegal `ttg.memdesc_index`, extending
    `FZ-20260421-0001`;
  - generic control flow over memdesc values is not uniformly broken once the
    probe avoids the direct chain0 descriptor-view packet mismatch and R5-C.
- Validation:
  - required `make -j8` reported no work to do;
  - `/tmp/tmem_generic_pass_round8.py` py-compiled;
  - collect-only found `14` nodeids;
  - split-4 sweep classified as `7 failed, 7 passed`;
  - fresh exact confirmations reproduced the FZ-0003 and FZ-0001 overlaps
    and the green chain1/chain2 controls.

### Lane A Round 8, ld.red Allocator/Opcode Structural Fuzzing

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_allocator_opcode_round8.md`
- Scope: 55 subprocess-isolated Python/Gluon rows around `ld.red` and
  `ld/st` allocation/opcode boundaries, covering direct indexed parents,
  row/column descriptor chains, 1CTA/2CTA parent shapes `[2,M,N]`,
  `M in {128,256,512}`, `N in {1,2,16,32,64,128}`, row/column permutations,
  `min`/`max`/`min(abs=True)`/NaN-propagating reductions, and `ld/st`
  read-only `f16`/`i32` variants.
- Result: no new non-overlapping `FZ-*` id.
- Key classification:
  - 18 rows extend `FZ-20260421-0004`: two-CTA indexed reductions produce
    correct values but emit plain `tcgen05.ld` instead of `.ld.red.` across
    direct shapes and reduction modifiers;
  - four rows extend `FZ-20260421-0008`: two-CTA row-chain
    `256x{2,16,32,64}` abort in `TritonNvidiaGPUOptimizeTMemLayoutsPass` with
    row/col versus row/col/block dimensions;
  - 14 rows extend the allocator family `FZ-20260421-0005/0009`: 1CTA direct
    indexed `ld.red` for `256x{16,32,64,128}` and `512x{16,32,64,128}`, plus
    direct indexed read-only `ld/st` `f16`/`i32` for `256x{16,32,64}`, fail
    in `TritonTensorMemoryAllocationPass` or reproduce the
    `MemoryBitMap::findFirstFit` assertion;
  - 18 rows passed, including 1CTA indexed `ld.red` for
    `128x{16,32,64,128}`, row-permuted 128-row indexed `ld.red`,
    128-row column-chain `load_max`, and 128-row `ld/st` read-only `f16`/`i32`
    descriptor-chain controls;
  - `256x1` two-CTA row-chain reports a clean `.x1` `ld.red` minimum-message
    diagnostic.
- Validation:
  - required `make -j8` reported no work to do;
  - `/tmp/tmem_ldred_allocator_opcode_round8_probe.py` py-compiled;
  - case inventory was `55`;
  - full subprocess-isolated sweep classified `18` passes, `18` opcode
    fallbacks, `14` allocator failures/asserts, `4` optimizer aborts, and
    `1` clean diagnostic;
  - exact confirmations on distinct GPUs/caches reproduced a positive
    `.ld.red.`, a 2CTA opcode fallback, a row/col optimizer abort, and an
    allocator assertion;
  - `git diff --check` passed.

### Lane E Round 9, Normalized Generator Inventory

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_generator_inventory_round9.md`
- Scope: `/tmp`-only normalized structural case generator inventory. The
  prototype tracks `normalized_id`, `legacy_case_ids`, `expected_class`,
  shape, dtype, `two_cta`, `num_ctas`, view chain, row/col kind, and
  register/instruction variant across `ldst`, `ldred`, `copy`, `mma`, and
  `mma_scaled`.
- Result: no new `FZ-*` id; this was discovery-support inventory work.
- Inventory:
  - generated descriptors: `44,043`;
  - checked-in structural-fuzzer legacy ids: `33`;
  - checked-in legacy ids with Round 9 normalized linkage: `33`;
  - generated descriptors with legacy links: `460`;
  - generated `(family, view_chain, expected_class)` buckets with legacy
    coverage: `18 / 46`.
- Sample validation:
  - required `make -j8` reported no work to do;
  - prototype py-compiled;
  - structural-fuzzer collect-only found `33` nodeids;
  - runnable sample reported `3 passed, 2 xfailed`, with the xfails being
    existing `FZ-20260421-0007` and `FZ-20260421-0009` sentinels.

### Lane E Follow-Up Round 9, ld/st Inventory-Hole Probe

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldst_inventory_holes_round9.md`
- Scope: compact runnable `ld/st` descriptor-view rows selected from the
  Round 9 normalized inventory holes, using the checked-in structural-fuzzer
  `ld/st` descriptor-view kernel.
- Result: no new `FZ-*` id.
- Classification:
  - chain1 `128x64` f32 identity/evenrow and `128x32` f16 identity rows
    extend `FZ-20260421-0003`;
  - `64x32` f32 transpose/slice reports the expected clean row-anchor
    diagnostic;
  - direct control, chain2 `128x64` f32 revcol, chain3 `128x64` f32
    transpose/slice, and chain2 `128x32` f16 revcol pass.
- Validation:
  - probe py-compiled;
  - collect-only found `8` nodeids;
  - full probe classified `4 failed, 4 passed`.

### Lane F Round 9, Clean-Diagnostic Adversarial Fuzzing

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_clean_diagnostics_round9.md`
- Scope: subprocess-isolated generated rows targeting expected clean
  unsupported/resource/API boundaries: `.x1 ld.red`, dynamic
  `memdesc_index`, two-CTA layouts in 4/8/16 CTA contexts, scales descriptor
  views, 128/256/512-row resource boundaries for direct indexed `ld/st` and
  `ld.red`, plus exact packed/subword and impossible `warpx2` copy schedule
  runtime-matrix nodeids.
- Result: no new non-overlapping `FZ-*` id.
- Classification:
  - eight generated rows reported clean diagnostics: `.x1 ld.red` minimum
    message shape, >2-CTA context mismatch, scales multibuffering, and
    `128x256` tensor-memory OOR;
  - one direct scales-copy control passed;
  - dynamic `memdesc_index` reproduced the known
    `FZ-20260421-0001` late illegal `ttg.memdesc_index`;
  - four generated `ldst/ldred` `256x32` and `512x32` rows reproduced
    allocator assertions in the `FZ-20260421-0005/0009` family;
  - exact packed/subword and impossible copy schedule nodeids passed `4/4`.
- Validation:
  - required `make -j8` reported no work to do;
  - `/tmp/tmem_clean_diagnostics_round9_probe.py` py-compiled;
  - generated case inventory was `14`;
  - subprocess-isolated sweep summary was `8` clean diagnostics, `1` pass,
    `1` late illegal op, and `4` allocator assertions;
  - exact copy-boundary pytest rows passed `4 passed`.

### Lane B Round 8, Scaled-MMAv5 Accumulator Control-Flow Fuzzing

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_mma_round8.md`
- Scope: 1CTA temporary scaled-MMAv5 harness over accumulator views
  `direct_low`, `direct_high`, dynamic slice `if`/loop/helper, dynamic
  indexed `if`/loop/helper, `use_acc` `False/True`, `N in {16,32,64,128}`,
  `K in {128,256}`, formats `mxfp8xmxfp8`, `mxfp8xmxfp4`,
  `mxfp4xmxfp8`, `mxfp4xmxfp4`, and `nvfp4xnvfp4`, and selectors `0/1`.
- Result: no new independent bucket; expands `FZ-20260421-0007` and adds
  scaled-MMAv5 evidence for `FZ-20260421-0001`.
- Counts:
  - JSONL rows: `1120`;
  - `628` pass, `332` miscompile, `160` exception;
  - all pass/miscompile rows had PTX/LLIR opcode agreement and expected
    scaled-MMAv5 opcode families.
- Key finding:
  - direct low/high controls pass across the probed surface;
  - dynamic slice/indexed paths still reproduce the broad known
    `FZ-20260421-0007` miscompile;
  - new important expansion: `nvfp4xnvfp4`, `N=16`, `K=256`, selector `1`
    dynamic slice/indexed accumulator views miscompile for both
    `use_acc=False` and `use_acc=True`, while direct high/low controls pass;
  - helper-returned indexed accumulator rows fail through dynamic
    `ttg.memdesc_index` illegal lowering and overlap `FZ-20260421-0001`;
  - existing 4CTA and 16CTA scaled-MMA copy sanity controls passed.
- Validation:
  - required `make -j8` reported no work to do;
  - `/tmp/tmem_scaled_mma_round8_probe.py` py-compiled;
  - collect-only found `1120` nodeids;
  - four split groups each classified `280` rows;
  - exact `nvfp4` high-selector contrasts and larger-CGA sanity controls
    passed as classified tests.

### Lane H Round 9, Validation Logistics

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_validation_logistics_round9.md`
- Scope: collection sizing, representative runtime timing, duration-aware
  splitting, and selector triage for TMEM-heavy structural-fuzzer and
  runtime-matrix sweeps. No backend/compiler repair attempted.
- Result: no new `FZ-*` id.
- Key inventory:
  - structural fuzzer: `33` collected; representative runtime
    `9 passed, 24 xfailed in 27.10s`;
  - full runtime matrix: `1615` collected;
  - compact recommended slices: scaled MMA `use_acc` (`28`), copy `warpx2`
    (`28`), LD.RED descriptor (`30`), clean diagnostics (`96`);
  - broad/noisy selectors to avoid for casual sweeps: full runtime matrix,
    `twocta`, broad `mma_scaled`, broad `ldst and descriptor`, and broad
    `ld_red or ldred`.
- Logistics recommendation: store durations for repeat slices and use
  `--splits 4 --group <n> --splitting-algorithm=least_duration` for skewed
  4-GPU reruns; do not use selectors that only collect skipped rows as timing
  calibration sources.

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
- Round 2 expansion:
  - runtime `memdesc_index` also reaches LLVM conversion as an illegal op
    through chain2 and chain3 helper-returned descriptor-view chains;
  - representative repro:
    `/tmp/tmem_expansion_round2_cf.py::test_cf_helper_returned_views[index-128-64-identity-identity-3-0-32x32b]`;
  - log: `/tmp/tmem_expansion_round2_confirm_cf_index_chain3.log`.
- Round 3 minimization:
  - direct runtime `parent.index(tt.load(selector))` on `[2,128,32]` reaches
    the same illegal `ttg.memdesc_index` in `ConvertTritonGPUToLLVM` with no
    helper, control flow, pre-store, reshape, or permute;
  - static/constexpr direct index rows pass;
  - the emitted load-only MLIR reproduces with `triton-opt --run-reproducer`;
  - checked-in strict xfail:
    `test_tmem_structural_fuzzer_generic_pass_dynamic_index_load_only[generic-pass-dynamic-index-load-only-128x32]`.

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
- Round 2 expansion:
  - false branch selector `0` also miscompiles for the dynamic `if` helper
    case;
  - `16x64b` also miscompiles for the same false-branch chain0 shape;
  - layout-pressure chain0 repro remained stable in a fresh process;
  - representative logs:
    `/tmp/tmem_expansion_round2_confirm_cf_if_false.log`,
    `/tmp/tmem_expansion_round2_confirm_cf_if_false_16x64.log`, and
    `/tmp/tmem_expansion_round2_confirm_cf_layout_pressure.log`.
- Round 4 expansion:
  - false-branch chain0 also miscompiles under `16x128b` for `[128,64]` and
    `[128,128]`;
  - tuple-like `(memdesc_view, tensor_bias)` capture reproduces the chain0
    mixed-capture failure;
  - layout-pressure chain0 also miscompiles under `16x128b`;
  - helper boundaries are not required: true no-helper inline chain0 fails
    identically to helper and nested-helper forms;
  - chain1 and chain2 variants passed across the same 16x64b/16x128b,
    tuple-capture, and layout-pressure probes, so the active boundary is
    chain0-specific in this harness;
  - report:
    `.codex/initiatives/tmem_linear_generalization/agents/fuzz_helper_cf_round4.md`.
- Promotion status: added checked-in strict xfails:
  - `test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-if-chain0-false-16x128b]`;
  - `test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-if-chain0-inline]`;
  - `test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-tuple-mixed-captures-chain0]`;
  - `test_tmem_structural_fuzzer_generic_pass_layout_conversion_pressure[generic-pass-layout-conversion-pressure-chain0-16x128b]`.
- Round 7 Lane D expansion:
  - layout-conversion pressure with a separate non-TMEM tensor input still
    compiles and miscompiles for chain0 under `16x128b`;
  - fresh confirmation reported `8063 / 8192` mismatched elements, matching
    the existing FZ-0002 failure scale;
  - tuple-like and loop-carried variants in the same lane were blocked earlier
    by the R5-C auto-layout compiler crash, so rerun them after the R5-C
    sentinel is fixed before assigning any new runtime-miscompile bucket.
  - report:
    `.codex/initiatives/tmem_linear_generalization/agents/fuzz_generic_analysis_round7.md`.
- Round 8 Lane D classification:
  - direct chain0 descriptor-view load mismatches even with no helper,
    tuple-return, side tensor, loop, or meaningful branch merge, while direct
    base and chain1 controls pass;
  - helper/tuple/sibling/static-loop/same-base dynamic-if chain0 rows are
    therefore repair-validation inventory for `FZ-20260421-0003`, not a new
    generic-pass root;
  - report:
    `.codex/initiatives/tmem_linear_generalization/agents/fuzz_generic_pass_round8.md`.

### FZ-20260421-0003: ld/st descriptor-view chains miscompile

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
- Round 2 expansion:
  - chain2 double-transpose/slice descriptor view with row `identity`, col
    `reverse`, and instruction `16x64b` also miscompiles;
  - representative repro:
    `/tmp/tmem_expansion_round2_ldst_ldred.py::test_ldst_adjacent_row_col_permutations[64-32-identity-reverse-2-16x64b]`;
  - fresh result mismatched `1024 / 2048` elements;
  - log: `/tmp/tmem_expansion_round2_confirm_ldst_chain2_colrev.log`.
- Round 3 minimization:
  - read-only `64x32` chain1 identity `32x32b` remains the smallest stable
    miscompile; smaller `16x32` and `32x32` rows cleanly reject;
  - chain2 and chain4 col-reverse `16x64b` share mismatch samples/opcodes and
    are treated as same-root packet-ordering evidence;
  - roundtrip tests can hide the bug because load and store use the same bad
    view mapping;
  - added checked-in strict xfail:
    `test_tmem_structural_fuzzer_ldst_descriptor_view_read[ldst-fz20260421-0003-chain2-col-reverse-64x32-16x64b]`.
- Round 4 expansion:
  - `f16` chain2 identity `64x32 16x64b` miscompiles with subword packet
    opcodes while f32/i32 chain2 identity is green in the R4-C harness;
  - f8 smoke rows passed;
  - chained two-CTA descriptor views reject cleanly while direct two-CTA
    indexed views pass;
  - added checked-in strict xfail:
    `test_tmem_structural_fuzzer_ldst_descriptor_view_read[ldst-fz20260421-0003-f16-chain2-identity-64x32-16x64b]`.

### FZ-20260421-0007: scaled-MMAv5 `use_acc` subslice accumulator miscompiles through dynamic `if`

- Source: Lane B Round 6.
- Failure class: `miscompile`.
- Family: `mma_scaled`.
- Shape: `M=128`, `N=64`, `K=128`.
- Formats: `mxfp8 x mxfp8`.
- Accumulator layout: `TensorMemoryLinearLayout` parent `[128,128]`.
- View chain: low-column accumulator subslice
  `acc_parent.slice(0, 64, dim=1)` selected against high-column sibling
  `acc_parent.slice(64, 64, dim=1)`.
- Control flow: dynamic `if ttgl.load(selector_ptr) != 0`, with selector `0`.
- `use_acc`: true, with preloaded accumulator values.
- Observed: successful compile and launch, but runtime output mismatches
  `a_ref @ b_ref.T + 3.0`; fresh exact repro reported
  `3693 / 8192` mismatched elements and NaN greatest-difference samples.
- Controls:
  - indexed accumulator `if` and loop rows passed;
  - high-subslice `if`, direct, and loop rows passed;
  - existing direct scaled-MMAv5 accumulator-subslice `use_acc` matrix rows
    remained green.
- Round 7 Lane A expansion:
  - broad corrected harness
    `/tmp/tmem_scaled_mma_controlflow_round7_probe.py` collected `410`
    nodeids and classified `140` passes, `210` runtime miscompiles, and
    `60` overlapping compiler exceptions;
  - direct low/high controls still pass, and high-selector `subslice_if`,
    `subslice_helper`, and `indexed_if` rows pass;
  - low-selector `subslice_if`, `subslice_helper`, `subslice_loop`,
    `indexed_if`, and `indexed_loop` rows miscompile across the probed
    `N in {32,64,128}`, `K in {128,256}`, and feasible
    `mxfp8`/`mxfp4`/`nvfp4` format cells that reach runtime;
  - fresh confirmations include `subslice_if` `mxfp8xmxfp8`
    `N=32,K=128,selector=0` with `1520 / 4096` mismatches versus selector
    `1` and direct controls green; `subslice_if` `N=128,K=256,selector=0`
    with `11669 / 16384` mismatches; and `indexed_if`
    `N=64,K=128,selector=0` with `2441 / 8192` mismatches while selector
    `1` is green;
  - `indexed_helper` exceptions are dynamic `ttg.memdesc_index`
    illegal-lowering overlap with `FZ-20260421-0001`, not a new id.
- Round 8 Lane B expansion:
  - `/tmp/tmem_scaled_mma_round8_probe.py` classified `1120` rows and found
    the dynamic accumulator view-selection bug is not limited to low-column
    selection or `use_acc=True`;
  - `nvfp4xnvfp4`, `N=16`, `K=256`, selector `1` dynamic slice/indexed
    accumulator views miscompile for both `use_acc=False` and `use_acc=True`;
  - direct high/low controls pass and PTX/LLIR MMA opcodes agree, so the
    current evidence still points at descriptor-view/control-flow selection,
    not opcode selection.
- Exact repro:
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short '/tmp/tmem_mma_scaled_controlflow_round6_probe.py::test_round6_scaled_mma_acc_controlflow[r6-scaled-subslice-if-n64-subslice-if-64-0-2]'`.
- Promotion status: checked-in strict xfail as of 2026-04-21 09:08 UTC:
  `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_scaled_mma_acc_subslice_control_flow[mma-scaled-fz20260421-0007-subslice-if-n64-selector0]`.
  The sentinel is self-contained in the structural fuzzer and imports only the
  non-test `tmem_test_utils.random_quantized_tensor` helper.

### FZ-20260421-0008: two-CTA indexed ld.red chain1 row/col optimizer crash

- Source: Lane C Round 6, extending the R4-D/R5-A report-only optimizer-crash
  family.
- Failure class: `compiler_crash`.
- Family: `ldred`.
- Likely owner surface:
  `TritonNvidiaGPUOptimizeTMemLayoutsPass`.
- Minimal stable shape on current `HEAD`: parent `[2,256,2]`, indexed view
  `[256,2]`.
- Layout: two-CTA `TensorMemoryLinearLayout`, row `even_odd`, col
  `identity`, lifted through prefix `[2]`.
- View chain:
  `parent.index(1).reshape((128,2,2)).permute([1,0,2]).reshape((256,2))`.
- Operation: store full tile, then `view.load_min()`.
- Observed: compiler reaches `TritonNvidiaGPUOptimizeTMemLayoutsPass` and
  fails before runtime/opcode inspection with
  `Dimensions must match, ignoring order, but they don't.  Got dims:
  ["row", "col"] and ["row", "col", "block"]`.
- Exact subprocess repro:
  `/tmp/tmem_ldred_crash_round6_child.py` with
  `CASE_M=256 CASE_N=2 CASE_CHAIN=1 CASE_ROW=even_odd CASE_COL=identity CASE_OP=min`.
- MLIR candidate:
  `/tmp/tmem_ldred_crash_round6_min_256x2_evenodd.mlir` reproduces with
  `triton-opt --run-reproducer` and aborts with exit `134`.
- Round 6 expansion:
  - `max`, `min(abs=True)`, and NaN-propagating `min` on the same shape all
    fail in the same optimizer pass;
  - `256x64`, chain1, row `identity`, col `reverse` reproduces the same
    optimizer crash;
  - `256x32`, chain1, row `identity`, col `identity` is a plain `ld` fallback
    and remains part of `FZ-20260421-0004`;
  - `256x2` chain2/chain3 `even_odd` controls are plain `ld` fallback rows,
    not optimizer crashes;
  - `128x32` chain1 `even_odd` remains a clean unsupported boundary.
- Promotion status: checked-in subprocess-isolated strict xfail as of
  2026-04-21 09:08 UTC:
  `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred_twocta_rowcol_optimizer_crash`.
  The subprocess isolation is required because the current failure aborts in
  `TritonNvidiaGPUOptimizeTMemLayoutsPass`.
- Round 8 Lane A expansion:
  - row-chain `256x{2,16,32,64}` variants with row `even_odd` reproduce the
    same `TritonNvidiaGPUOptimizeTMemLayoutsPass` row/col versus
    row/col/block optimizer abort;
  - `256x1` reports a clean `.x1` `ld.red` minimum-message diagnostic and is
    not part of the crash bucket;
  - report:
    `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_allocator_opcode_round8.md`.

### FZ-20260421-0009: 1CTA indexed ld.red 256-row parent allocator assertion

- Source: Round 7 Lane B generator-backed `ld/st` and `ld.red` probe.
- Failure class: `compiler_crash`.
- Family: `ldred`.
- Likely owner surface: `TritonTensorMemoryAllocationPass`.
- Seed/case id:
  `ldred-f32-1cta-ncta1-256x32-index-identity-identity-32x32b-min`,
  seed `0x46708516`.
- Shape: parent `[2,256,32]`, indexed view `[256,32]`.
- Layout: 1CTA `TensorMemoryLinearLayout`, row `identity`, col `identity`,
  lifted through prefix `[2]`.
- View chain: direct `parent.index(1)`.
- Operation: store full tile, then `view.load_min()`.
- Observed: no runtime/opcode stage is reached. The compiler emits an MLIR
  reproducer and fails while executing `TritonTensorMemoryAllocationPass`.
- Assertion:
  `TensorMemoryAllocation.cpp:65: MemoryBitMap::findFirstFit(...): Assertion
  'kNumRows - numRows >= 0' failed.`
- Exact repro:
  ```bash
  CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r7b-confirm-ldred256x32 PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short '/tmp/tmem_generator_ldst_ldred_round7_probe.py::test_generator_ldred_round7[ldred-f32-1cta-ncta1-256x32-index-identity-identity-32x32b-min]'
  ```
- MLIR replay:
  `/tmp/tmem_generator_ldst_ldred_round7_ldred256x32_alloc_assert.mlir`
  aborts with exit `134` under `triton-opt --run-reproducer`.
- Boundary evidence from the same lane:
  - `ldred-f32-1cta-ncta1-128x64-index-even_odd-identity-32x32b-min`
    passed and emitted hardware `.ld.red.`;
  - `ldred-f32-2cta-ncta2-256x64-index-identity-identity-32x32b-min`
    remains an existing `FZ-20260421-0004` opcode-loss row, runtime-correct
    but plain `tcgen05.ld`;
  - f16 read-only `ld/st` subword chain2 reproduced already-covered
    `FZ-20260421-0003`, while same-view roundtrip and i32 transpose/slice rows
    stopped at clean unsupported descriptor-view diagnostics.
- Promotion status: checked-in subprocess-isolated strict xfail as of
  2026-04-21:
  `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred_1cta_direct_index_allocator_crash`.
  The subprocess isolation is required because the current failure aborts in
  `TritonTensorMemoryAllocationPass`.
- Main-session Round 8 adjacency probe:
  - `M=128,N=32` and `M=128,N=64` direct indexed 1CTA `ld.red` rows passed
    and emitted `tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32` and
    `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32`;
  - `M=256,N=16`, `M=256,N=32`, `M=256,N=64`, and `M=512,N=32` all
    reproduced the same allocator assertion in subprocess children;
  - current classification: the crash family is not isolated to `N=32`, and
    the observed boundary is tied to indexed child reductions at `M >= 256`
    in this probe.

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
- Round 2 expansion:
  - opcode loss reproduced across chain1/chain2 rows, `N` in
    `{32, 64, 128}`, min/max reductions, NaN propagation variants, and
    adjacent row/col permutations when runtime output remained correct;
  - representative chain1 identity repro emits
    `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`;
  - representative chain2 rotate1 repro emits repeated
    `tcgen05.ld.sync.aligned.32x32b.x1.b32`;
  - resource-valid 2CTA ld.red direct indexed parent `[2, 256, 32]` also
    emitted plain `tcgen05.ld.sync.aligned.16x32bx2.x16.b32` instead of
    `.ld.red.`;
  - logs:
    `/tmp/tmem_expansion_round2_confirm_ldred_chain1_opcode.log`,
    `/tmp/tmem_expansion_round2_confirm_ldred_chain2_rot_opcode.log`, and
    `/tmp/tmem_expansion_round2_confirm_twocta_ldred.log`.
- Round 3 expansion:
  - confirmed stable 1CTA boundary: direct `parent.index(1)` chain0 emits
    `tcgen05.ld.red.sync.aligned.16x32bx2.x16.min.f32`, while chain1 emits
    `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`;
  - confirmed stable 2CTA boundary: full-parent `[2,256,64]` emits
    `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32`, but resource-valid
    `parent.index(1)` `[256,32]` chain0 emits
    `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`;
  - recommended additional checked-in strict xfail:
    `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min`;
  - logs:
    `/tmp/tmem_ldred_opcode_round3_confirm_onecta_chain1.log`,
    `/tmp/tmem_ldred_opcode_round3_confirm_twocta_index_chain0.log`, and
    `/tmp/tmem_ldred_opcode_round3_confirm_twocta_full_parent_positive.log`.
- Promotion status: the 2CTA indexed sentinel is now checked in as strict
  xfail:
  `test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min]`.
- Round 4 expansion:
  - 2CTA indexed `load_max`, `load_min(abs=True)`, and NaN-propagating
    `load_min` all pass runtime correctness but emit plain `tcgen05.ld` rather
    than `.ld.red.`;
  - opcode loss also spans `N=32/64/128`, chain0/1/2, and additional row/col
    layouts;
  - row/col chain1 variants found a separate optimizer crash, kept report-only
    until a crash-safe checked-in harness is added;
  - added checked-in strict xfails:
    `test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-twocta-indexed-256x32-chain0-max]`,
    `test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min-abs]`, and
    `test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min-nan]`.
- Round 8 Lane A expansion:
  - 18 additional two-CTA indexed rows produce correct runtime values but
    emit plain `tcgen05.ld` instead of `.ld.red.`;
  - representative rows include `128x{2,32,64}` and `256x{2,32,64}` direct
    indexed `min`, plus `256x{2,16,32,64}` `max`, `min(abs=True)`, and
    NaN-propagating `min`;
  - modifiers are therefore not the root; the owner remains the 2CTA indexed
    opcode-selection path.

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
- Round 8 Lane A expansion:
  - direct indexed read-only `ld/st` `f16` and `i32` for `256x{16,32,64}`
    reproduce allocator failures/assertions;
  - larger `512x{16,32,64,128}` direct indexed `ld.red` rows also fail in the
    allocator family and should become clean resource diagnostics if they are
    beyond physical TMEM capacity;
  - report:
    `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_allocator_opcode_round8.md`.

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

### Lane R5-A Round 5, ld.red row/col chain1 optimizer crash

- Time: 2026-04-21 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_optimizer_crash_round5.md`
- Scope: minimize the R4-D report-only 2CTA indexed `ld.red` row/col chain1
  optimizer crash and keep it separate from opcode fallback and clean
  diagnostics.
- Result: minimized the crash to a crash-safe subprocess candidate:
  `ldred-fz20260421-crash-twocta-indexed-256x2-chain1-even_odd-min`.
- Smallest stable repro:
  parent `[2,256,2]`, selected `[256,2]`,
  `parent.index(1).reshape((128,2,2)).permute([1,0,2]).reshape((256,2))`,
  row `even_odd`, col `identity`, `load_min`.
- Classification: compiler crash/optimizer abort in
  `TritonNvidiaGPUOptimizeTMemLayoutsPass`, printing the dimensions mismatch
  `["row","col"]` vs `["row","col","block"]`.
- Boundary separation:
  smaller `M=128/64` variants stop at clean unsupported descriptor-view
  diagnostics; `N=1` stops at the clean `.x1` `ld.red` minimum-message
  diagnostic; identity chain1 remains a plain-`ld` opcode fallback, not this
  crash.
- Recommendation: keep report-only unless checked-in coverage can use a
  subprocess-isolated strict xfail. Do not add an in-process strict xfail for
  this row.

### Round 7 Lane C, copy and copy-readback generator gaps

- Time: 2026-04-21 09:13 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_copy_readback_round7.md`
- Scope: deterministic `/tmp` launcher for copy descriptor-view chains plus
  immediate readback behavior, covering no-scales `warpx2` positives,
  no-scales `warpx2::02_13` clean diagnostics, packed/subword boundaries,
  copy-adjacent `ld.red` readback, scales `warpx4`, and two-CTA layout in a
  larger-CGA context.
- Result: no stable new backend/compiler failure was found. Positive
  descriptor-view copy/readback rows and clean diagnostic rows stayed green.
- Validation:
  - required `make -j8` reported no work to do;
  - `/tmp/tmem_copy_readback_round7_probe.py` py-compiled;
  - `/tmp` launcher passed six deterministic launch cases;
  - four-GPU selector sweep passed as `28 passed, 1587 deselected` on each
    group.
- Classification note: early custom combined-kernel attempts failed at
  harness construction, not at a backend surface, and were not assigned
  `FZ-*` ids.

### FZ-20260421-0018: M128xN512 f32 ld.red reaches late ptxas register allocation failure

- Source: Round 31 `ld.red` descriptor/layout extremes; expanded by Round 32
  FZ-0018 minimization.
- Failure class: `compiler_crash` / late toolchain resource failure without a
  clean frontend/backend diagnostic.
- Family: `ldred`.
- Shapes: direct or same-footprint descriptor-view `[128,512]` f32 TMEM tile;
  direct/legacy-equivalent `[256,256]` f32 TMEM tile.
- Layout: 1CTA direct `TensorMemoryLinearLayout` identity, row-reversed, and
  tile-swapped row/column bases. Column-reversed `M128xN512` is a passing
  hardware `.ld.red` control.
- Operation: store full tile, then `load_min()` with hardware `.ld.red`
  expected; Round 32 also reproduced with `load_max()`.
- Observed: compilation reaches `ptxas-blackwell`, then fails with register
  allocation count `255` instead of an earlier clean resource or unsupported
  diagnostic.
- Rows:
  - Round 31: `direct_m128_n512_identity` and
    `direct_m128_n512_identity_32x32b_splitn`;
  - Round 32: `direct` and same-footprint descriptor-chain `M128xN512,w4`
    identity/row-reverse/tile-swap rows with `auto` or explicit `32x32b`;
  - Round 32: direct and legacy-equivalent `M256xN256,w4`;
  - Round 32: `M128xN512,w4,max`.
- Independence evidence:
  - not `FZ-20260421-0012`: no `unsupported dst layout`; lowering reaches
    ptxas;
  - not `FZ-20260421-0010`: no CTA ownership mismatch;
  - not a generic `N=512` impossibility: `direct_m64_n512_identity_splitn`
    executes correctly and emits four `16x32bx2.x64` `.ld.red` instructions;
  - not a generic large-footprint impossibility: `M128xN512` column-reversed
    direct and same-footprint descriptor-chain rows execute correctly with
    eight hardware `.ld.red` instructions;
  - matching `M128xN512` and `M256xN256` shapes with `num_warps=8` report
    clean shared-memory OOR.
- Artifacts:
  `/tmp/tmem_ldred_extremes_round31.py`,
  `/tmp/tmem_ldred_extremes_round31/summary.json`, and per-case stdout/stderr
  files under `/tmp/tmem_ldred_extremes_round31/`; Round 32 minimizer
  `/tmp/tmem_ldred_fz0018_min_round32.py`,
  `/tmp/tmem_ldred_fz0018_min_round32/summary.json`, and per-case logs under
  `/tmp/tmem_ldred_fz0018_min_round32/`; Round 33 PTX capture
  `/tmp/tmem_fz0018_dump/.../direct_kernel.{ttgir,llir,ptx}` and passing
  control dumps under `/tmp/tmem_fz0018_dump_compare_pass/`.
- Repair guidance: add a compiler-only minimizer and decide whether the correct
  outcome is supported codegen with a better resource plan or an earlier clean
  resource diagnostic. Backend repair is intentionally deferred during the
  active fuzzing campaign.

## Repro Queue

- FZ-20260421-0001 is now covered by checked-in Python runtime xfail repros;
  optional next minimization is a lit `ttg.memdesc_index` crash/clean-error
  contract if repair work needs a compiler-only reproducer.
- FZ-20260421-0002 is now covered by checked-in Python runtime xfail repros
  for dynamic `if`, false-branch 16x128b, inline chain0, mixed and tuple-like
  tensor+memdesc capture, and layout-conversion pressure variants.
- FZ-20260421-0003 and FZ-20260421-0004 are now covered by checked-in Python
  runtime xfail repros, including the round-three chain2 col-reverse ld/st
  packet sentinel, the round-four f16 subword ld/st sentinel, and 2CTA indexed
  ld.red min/max/abs/NaN opcode sentinels.
- FZ-20260421-0005 is now covered by a checked-in subprocess xfail. Optional
  next minimization remains capturing the MLIR reproducer and rerunning with
  `triton-opt --run-reproducer`.
- FZ-20260421-0006 is now covered by a checked-in Python xfail. Expand around
  adjacent row/col permutations before classifying as a true boundary.
- Round 5 discovery queue:
  - continue structural fuzzing without backend repairs until new findings
    stop or the user pivots;
  - keep the R5-A minimized 2CTA indexed ld.red row/col chain1 optimizer crash
    report-only unless a subprocess-isolated checked-in xfail is desired;
  - keep dynamic `memdesc_index` lit candidate report-only until the intended
    contract is decided;
  - continue probing clean-negative boundaries separately from opcode fallback
    and runtime miscompile buckets;
  - use future subagents on surfaces not yet stressed by the structural
    fuzzer, especially copy/ld.red interactions and descriptor-view chains
    through additional generic passes.
- Round 7 Lane C found no new copy/readback failure to add to the repro queue.
  Future copy generator work should promote a stable repo-local runnable case
  adapter before replacing the current runtime-matrix-backed launcher.
- FZ-20260421-0018 now has a subprocess-isolated runtime minimizer covering
  direct and same-footprint descriptor-chain rows. Optional next work is a
  compiler-only minimizer that preserves the failing `.ld.red` resource plan
  without needing runtime execution; keep it report-only until the minimized
  repro is ready for checked-in subprocess or lit coverage.
- Round 8 Lane C found no new copy/readback failure to add to the repro queue.
  It validated 18 temporary generator-adapter rows and kept scales
  descriptor-view copy as a clean unsupported boundary.
- Round 8 Lane D found no new generic-pass failure to add to the repro queue.
  Keep its chain0 helper/tuple/sibling/static-loop/same-base dynamic-if rows
  as `FZ-20260421-0003` repair-validation inventory.
- Main-session Round 8 adjacency around `FZ-20260421-0009` broadened the
  allocator-crash evidence to direct indexed 1CTA `ld.red` rows at
  `M=256,N={16,32,64}` and `M=512,N=32`, with `M=128,N={32,64}` as positive
  `.ld.red.` controls. Use this as repair-boundary inventory; the checked-in
  sentinel remains the compact `256x32` subprocess xfail.
- Round 9 generator inventory found no new repro to add, but provides the
  current normalized coverage inventory and migration recommendation.
- Round 9 `ld/st` inventory-hole probing found no new repro owner; keep its
  chain1 rows as `FZ-20260421-0003` repair-validation inventory and the small
  transpose/slice row as clean row-anchor evidence.
- Round 9 clean-diagnostic fuzzing found no new repro owner; keep the
  allocator rows as `FZ-20260421-0005/0009`, the dynamic-index row as
  `FZ-20260421-0001`, and the other generated/copy rows as clean-boundary
  evidence.
- Round 8 scaled-MMAv5 fuzzing recommends adding a future strict runtime
  xfail under `FZ-20260421-0007` for the `nvfp4xnvfp4`, `N=16`, `K=256`,
  selector-1 dynamic accumulator-view miscompile, with direct high/low
  controls documented as passing.
- Round 9 validation logistics added practical next sweep slices and selector
  guidance; use those before launching broad runtime-matrix sweeps.

### Round 10 Lane O, opcode/IR consistency

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_opcode_consistency_round10.md`
- Probe command:
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon python /tmp/tmem_opcode_consistency_round10_probe.py`
- Scope: PTX-vs-LLIR opcode extraction consistency for `tcgen05.ld`,
  `tcgen05.st`, `tcgen05.ld.red`, `tcgen05.cp`, `tcgen05.mma`, and
  `tcgen05.mma_scaled` across descriptor views, 1CTA/2CTA, row/column
  permutations, explicit variants, reduction modifiers, and `use_acc`.
- Result: no new independent `FZ-*` bucket and no PTX/LLIR opcode disagreement
  on sampled positive rows.
- Positive controls emitted matching expected opcodes for direct and
  descriptor-chain `ld/st`, direct `ld.red` min and max.abs.NaN, no-scales and
  scaled-copy `cp`, plain MMAv5, and 1CTA/2CTA scaled-MMAv5.
- Known-overlap row: the indexed 256x32 `ld.red` case still emitted plain
  `tcgen05.ld.sync.aligned.32x32b.x64.b32` in both PTX and LLIR rather than
  `.ld.red.`, so it remains `FZ-20260421-0004` evidence rather than a new
  bucket.
- Clean boundary: an exotic permuted copy layout reported the expected clean
  unsupported `tcgen05.copy.128x256b` planning diagnostic.

### Round 11 local no-scales copy non-warpx2 slice

- Time: 2026-04-21
- Required build: `make -j8` no-op.
- Collect-only:
  `PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales and not warpx2 and not reports'`
  selected `169 / 1615` tests.
- Runtime: split-4 across GPUs 0-3 with stable
  `/tmp/triton-cache-gpu{0,1,2,3}` and durations stored at
  `/tmp/tmem_local_r11_cp_no_scales_nonwarpx2_durations.json`.
- Result: group counts `39 passed, 4 skipped`; `43 passed`; `43 passed`;
  `40 passed`. Aggregate `165 passed, 4 skipped`.
- Classification: green contrast coverage for no-scales copy indexed,
  subslice, two-CTA, subword, and tile-permuted rows outside the already-run
  `warpx2` selector. No new bucket.

### Round 10 Lane N, MMAv5/scaled-MMAv5 dynamic descriptor selection

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_mma_dynamic_round10.md`
- Probe: `/tmp/tmem_mma_dynamic_round10_probe.py`, subprocess-isolated worker
  rows after required `make -j8` and py-compile.
- Scope: dynamic `if`, helper-returned views, loop-carried views, sibling
  views, runtime selector indices, indexed/subslice/slice-index chains,
  `use_acc`, 1CTA/2CTA controls, `N in {16,32,64,128,256}`, and
  `K in {32,64,128,256}` where supported.
- New report-only bucket candidate: `FZ-20260421-0011`, plain MMAv5
  runtime-selector-index accumulator selection miscompiles. Representative
  rows compiled, had matching PTX/LLIR MMA opcodes, and then produced
  NaN-heavy runtime mismatches:
  - `plain-runtime-index-n64-k128-sel0-acc0`: around `8150/8192` mismatches;
  - `plain-runtime-index-n64-k128-sel1-acc0`: `8169/8192` mismatches;
  - `plain-runtime-index-n64-k128-sel1-acc1`: `8159/8192` mismatches;
  - `plain-runtime-index-n128-k128-sel1-acc1`: `16314/16384` mismatches.
- Classification note: the temporary harness did not show the illegal
  `ttg.memdesc_index` lowering signature of `FZ-20260421-0001`. A first
  checked-in minimization attempt in `test_tmem_structural_fuzzer.py` did hit
  that known `FZ-0001` path instead, so no checked-in sentinel was committed.
  Keep `FZ-0011` report-only until the exact runtime-miscompile shape is
  minimized without changing its failure mode.
- Existing-bucket expansion: scaled dynamic-select/runtime-index/indexed rows
  expand `FZ-20260421-0007`. Plain helper/loop/sibling/indexed/subslice rows
  and exact 1CTA/2CTA runtime-matrix controls stayed green.

### Round 10 Lane P, clean-boundary adversarial sweep

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_clean_boundary_round10.md`
- Probe: `/tmp/fuzz_clean_boundary_round10_probe.py`, split over GPUs 0-3.
- Result: no new independent `FZ-*` bucket.
- Pass/clean-diagnostic coverage stayed stable for copy/TMEM OOR boundaries,
  explicit `.x1` subword rows, legacy subword parse boundaries,
  `warpx2::02_13`, scales descriptor-view boundaries, scaled-MMA
  tile-permuted accumulator layouts, >2 CTA/CGA mismatches, and ld.red
  transpose/slice boundaries.
- Only compiler-failure row: `ld_red_identity_n512`, classified as overlap
  with the existing `FZ-20260421-0005/0009` ld.red resource/crash family.

### Round 11 Lane Q, multi-CTA/CGA structural fuzzing

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_multicta_cga_round11.md`
- Required build: `make -j8` no-op.
- Checked-in high-CGA controls:
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit python/test/gluon/test_core.py::test_tma_mma_shared_inputs -k 'ctas_per_cga2 or ctas_per_cga1'`
  reported `136 passed, 12 skipped, 74 deselected in 139.31s`.
- Temporary probe:
  `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon python /tmp/tmem_multicta_cga_round11_probe.py`.
- New independent candidate: `FZ-20260421-0010`, over-strict layout CTA-count
  gate in larger CGA contexts. Local 1CTA/2CTA TMEM linear/scales operations
  inside 4/8/16 CTA launch contexts reject before lowering with messages such
  as:
  - `Layout has 1 CTAs per CGA, but the context requires 4 CTAs per CGA.`
  - `Layout has 1 CTAs per CGA, but the context requires 8 CTAs per CGA.`
  - `Layout has 1 CTAs per CGA, but the context requires 16 CTAs per CGA.`
  - `Layout has 2 CTAs per CGA, but the context requires 4 CTAs per CGA.`
  - `Layout has 2 CTAs per CGA, but the context requires 8 CTAs per CGA.`
  - `Layout has 2 CTAs per CGA, but the context requires 16 CTAs per CGA.`
- Representative affected surfaces: 1CTA `ld/st` descriptor-view chains,
  1CTA and 2CTA tensor-memory-scales copy, and 2CTA indexed `ld.red` layouts
  launched in 4/8/16 CTA contexts.
- Classification note: this is distinct from the earlier 2CTA uniformity
  hardware rule. The rows are not requesting a 4/8/16-CTA TCGEN05 instruction;
  the passing high-CGA controls show legal high-CGA kernels exist when full CGA
  metadata is available. The likely schema/backend gap is distinguishing
  kernel CGA shape from instruction-local `cta_group` for linear/scales TMEM
  layouts.

### Round 12 Lane R, high-CGA gate minimization

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_high_cga_gate_round12.md`
- Required build: `make -j8` no-op.
- Probe:
  `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon python /tmp/tmem_high_cga_gate_round12_probe.py`.
- Result: `FZ-20260421-0010` reproduced across `18` minimized rows:
  - 1CTA/2CTA linear `ld/st` descriptor-view chain in 4/8/16 CTA contexts;
  - 1CTA/2CTA direct `ld.red` in 4/8/16 CTA contexts;
  - 1CTA/2CTA no-scales copy in 4/8/16 CTA contexts;
  - scales-copy layout construction diagnostics for 1CTA/2CTA high-CGA rows.
- Representative diagnostics remain the exact CTA-count equality gate:
  `Layout has 1 CTAs per CGA, but the context requires 4 CTAs per CGA` and
  `Layout has 2 CTAs per CGA, but the context requires 16 CTAs per CGA`.
- Passing contrast:
  `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[False-ctas_per_cga1]' 'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[True-ctas_per_cga2]'`
  reported `2 passed in 3.18s`.
- Promotion recommendation: add xfails for 1CTA and 2CTA linear `ld/st`,
  direct `ld.red`, and no-scales copy in 4/8/16 CTA contexts, plus a smaller
  scales-layout diagnostic test. Keep the high-CGA MMA controls as green
  contrast evidence.

### Round 12 Lane S, plain-MMAv5 runtime-index reduction

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_plain_mma_runtime_index_round12.md`
- Reducer:
  `/tmp/tmem_plain_mma_runtime_index_round12_probe.py`.
- Scope: selector `0/1`, `use_acc` false/true, `N in {32,64,128,256}`,
  `K in {32,64,128,256}`, lifted rank-3 linear parents, legacy/direct parent
  variants, explicit reshape, preinit, shared/load-layout variants,
  dynamic-slice controls, constexpr-index controls, and runtime-matrix import
  vs standalone reducer.
- Result: `FZ-20260421-0011` is stable only when FPSAN instrumentation is
  enabled. The smallest stable row is `N=32,K=128,selector=1,use_acc=False`
  under FPSAN; it repeated `3/3` fresh subprocesses with `4081/4096`
  mismatches, `16` NaNs, and finite max absolute difference around
  `3.303e38`.
- Non-FPSAN classification: lifted-linear runtime-index rows change failure
  mode to known `FZ-20260421-0001` illegal `ttg.memdesc_index` lowering.
  Importing runtime-matrix helpers is not the preserving detail.
- Controls: constexpr index on lifted linear parent, dynamic slice selection
  between `[M,2N]` sibling views, default-pipeline runtime index on a legacy
  parent, and default-pipeline runtime index with a direct 2D parent layout
  passed. Under FPSAN, legacy-parent and direct-2D-parent runtime-index rows
  also miscompiled.
- Promotion recommendation: only add a strict xfail if the test explicitly
  pins FPSAN instrumentation; include dynamic-slice and constexpr-index green
  controls next to it.

### Round 12 Lane V, copy descriptor/addressing fuzzing

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_copy_descriptor_round12.md`
- Required build: `make -j8` no-op.
- Checked-in copy/scales selector:
  `python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales_warpx2 or cp_scales or cp_128x128 or cp_no_scales_linear_subword or cp_no_scales_twocta_layout_in_4cta_context'`
  selected `128/1615` tests and passed split-4 as `32/32/32/32`.
- Fresh-process probe:
  `/tmp/tmem_copy_descriptor_round12_probe.py` reported `6` pass,
  `3` `FZ-20260421-0010`, and one harness capture limitation that exact
  pytest later confirmed as the known scales descriptor-view clean boundary.
- Positive opcode checks:
  - no-scales 1CTA `warpx2::01_23` f32 emitted matching
    `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b`;
  - no-scales indexed `warpx2::02_13` i32 emitted matching
    `tcgen05.cp.cta_group::1.warpx2::02_13.64x128b`;
  - no-scales 2CTA `warpx2::01_23` slice-index emitted matching
    `tcgen05.cp.cta_group::2.warpx2::01_23.64x128b`;
  - linear subword i8 emitted `tcgen05.cp.cta_group::1.128x256b`;
  - scales `warpx4` 1CTA/2CTA direct rows emitted matching
    `tcgen05.cp.cta_group::1.warpx4.32x128b`.
- `FZ-20260421-0010` copy evidence: no-scales 2CTA copy in 4/8/16 CTA launch
  contexts still rejects with `Layout has 2 CTAs per CGA, but the context
  requires Y CTAs per CGA`.
- Clean boundaries stayed clean for scales descriptor-view row-order/footprint
  requirements and no-scales `warpx2` subword packed-lane storage-model
  requirements. No new independent copy-family bucket.

### Round 12 Lane U, ld.red row/column/opcode fuzzing

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_rowcol_round12.md`
- Required build: `make -j8` no-op.
- Temporary probe:
  `/tmp/tmem_ldred_rowcol_round12_probe.py`.
- Scope: 94 subprocess-isolated `tcgen05.ld.red` rows over 1CTA direct,
  1CTA indexed, 1CTA descriptor chains, 2CTA direct controls, 2CTA indexed,
  2CTA row chains, 2CTA column chains, row/column permutations, `N` from `2`
  through `512`, and `min`/`max.abs.NaN` modifier variants where supported.
- Result: no new independent `FZ-*` bucket and no runtime miscompile.
- Final classification:
  - `42` pass;
  - `19` `FZ-20260421-0004` opcode-fallback rows where runtime values were
    correct but PTX/LLIR emitted plain `tcgen05.ld` instead of `.ld.red`;
  - `14` `FZ-20260421-0008` optimizer abort rows in
    `TritonNvidiaGPUOptimizeTMemLayoutsPass` with the known row/col/block
    dimension mismatch;
  - `13` `FZ-20260421-0005/0009` allocator/resource failures for 1CTA
    `M=256` indexed and chained reductions;
  - `6` clean TMEM OutOfResources boundaries.
- Bucket expansion: broadens `FZ-20260421-0004` across 2CTA indexed and
  column-chain `min` and `max.abs.NaN` plus one row-chain permutation,
  broadens `FZ-20260421-0008` through `N=128` and multiple row permutations,
  and broadens `FZ-20260421-0005/0009` across narrow and wide `N`.

### Round 12 Lane W, generic descriptor-view pass fuzzing

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_generic_views_round12.md`
- Required build: `make -j8` no-op.
- Checked-in generic-pass slice:
  `python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'`
  collected `11/33` rows and passed as `11 xfailed` split across GPUs 0-3.
- Temporary probe:
  `/tmp/tmem_generic_views_round12_probe.py` collected `9` rows and, after
  excluding one harness-call bug and rerunning the corrected direct-control
  row, classified as `7` pass and `2` known failures.
- Result: no new independent `FZ-*` bucket.
- Classification:
  - runtime `memdesc_index` through chain2 load/store remains
    `FZ-20260421-0001`;
  - chain0 generic-pass/control-flow/layout-pressure wrong results remain
    `FZ-20260421-0002`;
  - the checked-in loop-carried memdesc-view row currently reaches runtime and
    mismatches `8064/8192`, so on this head it should be treated as
    `FZ-20260421-0002` rather than the historical `R5-C` auto-layout crash;
  - direct/static/constexpr and chain1/chain2 controls passed.
- Promotion guidance: keep the checked-in strict xfail set; update the
  loop-carried expected reason in the later repair/test-refresh phase only if
  repeated reruns keep showing the runtime mismatch instead of the historical
  crash.

### Round 13 Lane X, allocation/lifetime/commit/barrier fuzzing

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_alloc_lifetime_round13.md`
- Required build: `make -j8` no-op.
- Temporary probe:
  `/tmp/tmem_alloc_lifetime_round13_probe.py`.
- Scope: 28 subprocess-isolated rows covering two independent live TMEM
  allocations, sibling descriptor views with overlapping live ranges,
  copy-plus-ldst with one commit/wait sequence, MMA accumulator views sharing
  a parent allocation with sibling views, 1/2/4 CTA launch contexts, and
  allocator pressure around the 512-column hardware limit.
- Result: no new independent `FZ-*` bucket and no runtime miscompile.
- Final classification:
  - `18` pass;
  - `4` `FZ-20260421-0010` high-CGA CTA-count gate rows;
  - `2` clean TMEM OutOfResources boundaries;
  - `2` clean copy packed-lane unsupported boundaries;
  - `2` harness/shared-layout setup limitations.
- Positive coverage candidates after discovery: multiple independent live
  TMEM allocations, overlapping sibling descriptor views, 1CTA MMA accumulator
  view plus sibling load/store view, and allocator-pressure rows that
  distinguish legal reuse from clean 512-column rejection.

### Round 13 local lifetime/mbarrier sanity

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_lifetime_mbarrier_round13.md`
- Required build: `make -j8` no-op before each runtime slice.
- Scope: existing runtime coverage around TMEM allocation lifetime
  bookkeeping, commit opcode selection, multicast mbarrier plumbing,
  descriptor-chain subviews, and physical-bitcast view mapping.
- Result: no new independent `FZ-*` bucket.
- Evidence:
  - core commit/mbarrier selector over TMA multicast, 1CTA/2CTA MMAv5
    multicast commit, scaled-MMAv5 multicast barrier, and async-copy mbarrier
    rows passed `13/13`;
  - runtime-matrix allocation lifetime selector passed `7/7`;
  - descriptor-chain and physical-bitcast selector passed `29/29`.

### Round 13 local ConSan and FPSAN instrumentation controls

- Time: 2026-04-21
- ConSan report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_consan_tmem_round13.md`
- Required build: `make -j8` no-op.
- ConSan scope: existing ConSan runtime coverage touching TMEM operations:
  1CTA/2CTA/4CTA `tcgen05_mma`, TMEM load/store race checks,
  `tcgen05_copy`, TMA, multicast, and TMEM load/store access patterns.
- ConSan result: `36/348` collected, split-4 passed as `30 passed,
  6 skipped`; no new `FZ-*` bucket.
- FPSAN checked-in control scope:
  `python/test/gluon/test_fpsan.py -k 'tcgen05_mma or tcgen05_mma_scaled'`
  selected `37/104` rows.
- FPSAN checked-in control result: split-4 passed as `32 passed, 5 skipped`;
  no new `FZ-*` bucket. This is independent green-control evidence around
  the still-active FPSAN runtime-index reducer lane.

### Round 13 Lane Y, FPSAN MMAv5 Runtime Descriptor Selection

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_fpsan_mma_round13.md`
- Scope: FPSAN on/off contrasts for plain MMAv5 runtime accumulator
  descriptor selection across lifted-linear, legacy, and direct-linear parent
  layouts, selector values, `use_acc`, neighboring `N/K` variants,
  constexpr-index controls, dynamic-slice controls, reshape/preinit
  perturbations, imported-helper contrast, and checked-in plain/scaled FPSAN
  controls.
- Result: no new independent `FZ-*` bucket. Temporary contrast grid classified
  as `8` `FZ-20260421-0011`, `6` pass, and `2` `FZ-20260421-0001`.
- Finding: `FZ-20260421-0011` is isolated to FPSAN runtime outer descriptor
  selection by `parent.index(ttgl.load(selector_ptr))` feeding plain MMAv5.
  Checked-in plain/scaled FPSAN controls, constexpr parent-index controls, and
  dynamic slice controls are green.

### Round 13 local scaled-MMAv5 FP4/tile/narrow controls

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_fp4_tile_narrow_round13.md`
- Required build: `make -j8` no-op.
- Scope: checked-in scaled-MMAv5 runtime coverage around MXFP4/NVFP4 and
  mixed FP8/FP4 formats, root/indexed/subslice accumulator views, 1CTA/2CTA
  rows, LHS subslice and tile-permuted TMEM operands, tile-permuted
  accumulator layouts, narrow accumulator fragments, and clean unsupported
  mixed-precision FP4-LHS diagnostics.
- Result: `202/1615` collected for
  `mma_scaled and (tile_permuted or narrow or e2m1 or fp4)`, split-4 passed
  as `202 passed`; no new `FZ-*` bucket.

### Round 13 local warp-specialized TMEM partitioning controls

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_warpspec_partition_round13.md`
- Required build: `make -j8` no-op.
- Scope: runtime rows for TMEM copy/store inside `gl.warp_specialize`, lit
  partition scheduling for TMEM load/store/MMA users, tensor-memory allocation
  through warp-specialized captures and subviews, TMEM interleave coverage, and
  NVWS async-reference insertion for TMEM/MMAv5 flows.
- Result:
  - Python selector `warp_specialize and tmem` collected `2/18218` and passed
    `2/2`;
  - lit files `partition-scheduling.mlir`,
    `test_tensor_memory_allocation.mlir`, `interleave_tmem.mlir`, and
    `aref-tmem-insertion.mlir` passed `4/4`.
- No new `FZ-*` bucket.

### Round 13 Lane Z, Copy/Subword Edge Fuzzing

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_copy_subword_round13.md`
- Scope: copy-family clean-boundary and neighboring-positive coverage beyond
  Lane V: tile-permuted and tile-selector-permuted no-scales destinations,
  row/column-permuted clean diagnostics, mixed/exotic linear-layout clean
  diagnostics, `4x256b` refresh-shaped destinations, exact-width subword
  `128x128b` positives, legacy/unpacked subword diagnostics, and 4/8/16 CTA
  high-CGA contrasts for local 2CTA copy.
- Result: no new independent `FZ-*` bucket. The checked-in selector collected
  `69/1615` and split-4 passed as `69 passed`; representative PTX/LLIR
  opcodes matched for `tcgen05.cp.cta_group::1.128x128b`,
  `tcgen05.cp.cta_group::1.4x256b`, and exact-width subword
  `128x128b`. The high-CGA contrasts map to existing `FZ-20260421-0010`.

### Round 13 Local, LD.RED M64 Row-Permuted Layouts

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_m64_permuted_round13.md`
- Scope: checked-in `ld.red` runtime-matrix coverage for descriptor chains,
  tile-permuted layouts, row/column permutations, N sweeps, and non-f32
  software reductions, excluding `reports_*` clean-diagnostic rows.
- Result: new candidate `FZ-20260421-0012`. The selector collected
  `115/1615` and split-4 ran as `109 passed, 6 failed`; exact fresh reruns
  reproduced the six M64 row-permuted failures. Representative diagnostic:
  `ttng.tmem_load` could not compute TMEM encoding info for reduction because
  destination-layout lowering reported unsupported dst layout. Nearby
  `col_reverse_n32` controls passed, so this is currently classified as an
  over-strict destination-layout lowering gap rather than cache noise or an
  existing opcode/allocator/high-CGA bucket.

### Round 13 Local, LD/ST Descriptor Selection

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldst_descriptor_round13.md`
- Scope: checked-in `ld/st` descriptor roundtrips, descriptor-composition
  chains, row/column-permuted descriptor parents, two-CTA descriptor parents,
  and subword descriptor chains, excluding `reports_*` clean-diagnostic rows.
- Result: no new `FZ-*`. The selector collected `142/1615` and split-4 ran as
  `64 passed, 78 skipped`. This is a green-control lane for runtime descriptor
  selection outside FPSAN/plain-MMAv5 and outside `ld.red`; it did not
  reproduce `FZ-20260421-0011` or the new `FZ-20260421-0012` unsupported-dst
  diagnostic.
- Follow-up: broader selector
  `ldst and (descriptor_chain or x1 or subword or replay)` collected
  `261/1615` and split-4 passed as `191 passed, 70 skipped`, covering x1
  f32/i32/subword rows, two-CTA x1 descriptor chains, subword pack/unpack,
  descriptor-chain roundtrips, replay rows, and rank5/higher-rank descriptor
  rows.

### Round 13 Local, MMAv5 Descriptor/View Selection

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_mma_descriptor_view_round13.md`
- Scope: checked-in MMAv5 indexed accumulator views, lifted
  `linear_unit_parent` parents, `use_acc` true/false rows, TMA-backed
  transposed-B descriptor inputs, shared scale descriptor-view rows, and a
  scaled-MMAv5 indexed narrow accumulator-view control.
- Result: no new `FZ-*`. The selector collected `37/1615` and split-4 passed
  as `37 passed`, strengthening the classification that
  `FZ-20260421-0011` is FPSAN-specific runtime outer descriptor selection
  feeding plain MMAv5 rather than a general MMAv5 descriptor/view issue.

### Round 14 local ld.red non-f32 descriptor sanity

- Time: 2026-04-21
- Required build: `make -j8` no-op.
- Scope: checked-in non-f32 `ld.red` descriptor-chain software-reduction rows
  over i32, bf16, f16, i16, and i8 min/max variants.
- Result: selector `ld_red and non_f32 and descriptor` collected `20/1615`
  and split-4 passed as `20 passed`; no new `FZ-*` bucket.

### Round 14 local high-CGA scales sanity

- Time: 2026-04-21
- Required build: `make -j8` no-op.
- Scope: checked-in high-CGA/scales rows selected by
  `4cta_context or cga or high_cga`, including scales direct roundtrips,
  scales descriptor-view CGA roundtrips, and the clean 4CTA copy diagnostic.
- Result: `17/1615` collected and split-4 passed as `17 passed`; no new
  `FZ-*` bucket.

### Round 14 local scaled descriptor sanity

- Time: 2026-04-21
- Required build: `make -j8` no-op.
- Scope: checked-in scaled-MMAv5 indexed accumulator views, shared scale
  descriptor-view auto-copy rows, B-scale descriptor views, and narrow
  indexed accumulator controls selected by
  `mma_scaled and (descriptor or scale_descriptor or indexed_acc)`.
- Result: `32/1615` collected and split-4 passed as `32 passed`; no new
  `FZ-*` bucket.

### Round 14 structural fuzzer smoke gate

- Time: 2026-04-21
- Required build: `make -j8` no-op.
- Scope: full checked-in structural fuzzer.
- Result:
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py`
  reported `9 passed, 24 xfailed in 8.15s`.

### Round 14 Lane AA, FZ-0012 ld.red expansion

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_fz0012_round14.md`
- Required build: `make -j8` no-op.
- Scope: broadened and minimized `FZ-20260421-0012` across direct M64
  `ld.red`, explicit load variants, modifiers, descriptor-preserving chains,
  attempted two-CTA rows, and nearby row-identity column-permuted and M128
  row-permuted controls.
- Result: no additional independent `FZ-*` bucket, but `FZ-20260421-0012`
  broadens substantially.
- Checked-in M64 selector `ld_red_m64 and not reports` collected `39/1615`
  and split-4 ran as `33 passed, 6 failed`.
- Temporary subprocess-isolated grid ran `133` rows:
  - `38` pass;
  - `86` `FZ-20260421-0012`;
  - `5` `FZ-20260421-0010`;
  - `4` descriptor-permute harness-limited rows.
- Updated classification:
  `FZ-20260421-0012` covers M64 `tcgen05.ld.red` destination-layout lowering
  rejecting effective non-identity row bases across `N in {32,64,128,256}`,
  column permutations, `min`/`max`, `abs`/NaN modifiers, explicit variants
  `32x32b`, `auto`, `16x32bx2`, `32x32b_splitn`, and descriptor-preserving
  `slice`/`reshape` chains.
- Green controls: row-identity M64 column permutations passed, explicit M64
  row-identity column controls passed, and M128 direct row-permuted controls
  passed. Attempted two-CTA M64-style rows mapped to existing `FZ-0010`.

### Round 14 Lane AB, scaled-MMAv5 descriptor fuzzing

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_descriptor_round14.md`
- Required build: `make -j8` no-op.
- Scope: scaled-MMAv5 descriptor/view and scale-copy interactions outside the
  prior FP4/tile/narrow green selector: shared scale descriptor views, auto
  `tcgen05.cp` into TMEM scales, direct shared scales vs explicit TMEM scales,
  indexed/subslice/narrow accumulator views, 1CTA/2CTA variants, `use_acc`,
  and B-scale descriptor-view chains.
- Result: new candidate `FZ-20260421-0013`.
- Checked-in selector
  `mma_scaled and not (tile_permuted or narrow or e2m1 or fp4)` collected
  `69/1615` and split-4 passed as `69 passed`.
- Temporary shared-scale/copy probe classified `8` pass and `2`
  harness/setup limitations, with matching `tcgen05.cp` and scaled-MMA opcode
  evidence for 1CTA/2CTA shared-scale rows.
- Temporary corrected B-scale descriptor-view probe classified `1` pass,
  `3` runtime miscompile candidates, and `3` clean unsupported / diagnostic
  boundaries.
- `FZ-20260421-0013`: scaled-MMAv5 B-scale descriptor-view miscompile.
  Repro rows are local 1CTA, non-FPSAN wrong results with matching PTX/LLIR
  `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`
  opcodes and retained TTGIR descriptor-view chains. Current minimized forms
  include `N=128` linear accumulator layouts with B-scale
  `reshape -> trans -> reshape` descriptor views and `N=256` tile-N64
  accumulator layout with unpadded B-scale storage.
- Not mapped to existing buckets: it is not dynamic accumulator selection
  `FZ-0007`, not high-CGA `FZ-0010`, and not FPSAN runtime descriptor
  selection `FZ-0011`.

### Round 14 local generic-pass structural selector

- Time: 2026-04-21 10:53 UTC
- Required build: `make -j8` no-op.
- Scope: checked-in generic-pass structural fuzzer rows while Lane AC broadened
  the same family independently.
- Collect-only:
  `PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'`
  selected `11/33` rows.
- Runtime pattern:
  `CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> --store-durations --durations-path /tmp/tmem_local_r14_generic_pass_durations.json python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'`
- Result:
  - group 1/GPU 0: `3 xfailed, 30 deselected`;
  - group 2/GPU 1: `3 xfailed, 30 deselected`;
  - group 3/GPU 2: `3 xfailed, 30 deselected`;
  - group 4/GPU 3: `2 xfailed, 31 deselected`.
- Aggregate: `11 xfailed`. No new independent `FZ-*` bucket.
- Classification: runtime dynamic `ttg.memdesc_index` illegal lowering remains
  `FZ-20260421-0001`; chain0/control-flow/layout-pressure generic
  descriptor-view wrong results remain `FZ-20260421-0002`. Backend repair
  remains deferred until discovery stops finding new bugs or the user pivots.
- Active subagent lane: Lane AC is still fuzzing generic/control-flow/runtime
  index TMEM descriptor use beyond the checked-in selector and is expected to
  write `agents/fuzz_generic_runtime_index_round14.md`.

### Round 14 Lane AC, generic runtime-index/control-flow consumers

- Time: 2026-04-21 11:02 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_generic_runtime_index_round14.md`
- Required build: `make -j8` no-op.
- Scope: runtime `memdesc_index`, dynamic `if`, loop-carried values,
  tuple/mixed captures, helper-returned descriptor chains, and copy/MMAv5
  consumers beyond the checked-in `generic_pass` selector.
- Result: no new independent `FZ-*` bucket.
- Temporary generic probe classified `27` rows:
  - `10` pass;
  - `6` `FZ-20260421-0001`;
  - `11` `FZ-20260421-0002`.
- Temporary copy/MMA consumer probe classified `12` rows:
  - `2` pass;
  - `8` `FZ-20260421-0001`;
  - `2` clean hardware/layout diagnostics.
- Classification update: `FZ-20260421-0001` is a general memdesc SSA
  control-flow/index lowering gap covering copy/MMAv5 consumers and
  branch-selected static `memdesc_index` values, not only direct dynamic
  integer indexing. `FZ-20260421-0002` remains the chain0 generic
  control-flow/helper/mixed-capture/layout-pressure wrong-result family.
- Local adjacent baseline:
  `python/test/gluon/test_tmem_runtime_matrix.py -k '(cp_no_scales_twocta or cp_scales_warpx4_twocta or cga_roundtrip or layout_in_4cta_context) and not via_scaled_mma'`
  collected `53/1615` rows and passed split-4 as `53 passed`.

### Round 14 local descriptor and ld.red selectors

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_descriptor_ldred_round14.md`
- Required build: `make -j8` no-op.
- ld/st descriptor algebra selector:
  `runtime_selector or descriptor_roundtrip or descriptor_compositions`
  collected `116/1615` rows and passed split-4 as `38 passed, 78 skipped`.
- `ld.red` M64/descriptor/rowcol selector:
  `ld_red and (m64 or descriptor or rowcol_permuted) and not reports`
  collected `100/1615` rows and ran split-4 as `94 passed, 6 failed`.
- Classification: no new bucket. The six failures are exactly the known
  `FZ-20260421-0012` M64 destination-layout lowering failures for effective
  non-identity row bases.

### Round 14 local clean-boundary selector

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_clean_boundaries_round14.md`
- Required build: `make -j8` no-op.
- Selector:
  `reports_clean_unsupported or reports_clean_error or reports_tmem_oor`
  collected `157/1615` rows.
- Split-4 result:
  `157 passed` (`40`, `40`, `40`, `37` by shard).
- Classification: no new bucket. The clean unsupported/error/OOR surface
  remained stable and did not drift into unexpected compiler crashes or runtime
  execution paths.

### Round 14 local scaled-MMAv5 descriptor/accumulator selector

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_scaled_descriptor_acc_round14.md`
- Required build: `make -j8` no-op.
- Selector:
  `mma_scaled and (descriptor or scale_descriptor or bscale or acc_subslice_view or indexed_acc) and not reports`
  collected `105/1615` rows.
- Split-4 result:
  `105 passed` (`27`, `27`, `27`, `24` by shard).
- Classification: no new bucket. This is a checked-in green baseline next to
  Lane AG's temporary `FZ-20260421-0013` minimization.

### Round 14 Lane AG, FZ-0013 scaled-MMAv5 scale descriptor minimization

- Time: 2026-04-21 11:11 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_fz0013_min_round14.md`
- Required build: `make -j8` no-op.
- Scope: minimized and broadened the Round 14 Lane AB
  `FZ-20260421-0013` scaled-MMAv5 B-scale descriptor-view wrong-result
  candidate across `N=128/256`, padded/unpadded scale storage, extra scale
  user, accumulator layout family, `use_acc`, scale format combinations,
  A-scale versus B-scale, descriptor chain order/depth, and nearby
  unsupported boundaries.
- Temporary probe:
  `/tmp/tmem_scaled_fz0013_min_round14_probe.py`.
- Result: no new independent `FZ-*`; `FZ-20260421-0013` is broadened.
- Fresh-subprocess matrix classification over `33` rows:
  - `19` runtime miscompiles;
  - `5` passes;
  - `2` clean unsupported diagnostics;
  - `1` explicit narrow-N unsupported diagnostic followed by
    `PassManager::run failed`;
  - `6` parser/setup/unsupported-view limitations.
- Smallest stable row:
  `min-n128-b-rtr-pad0-linear`, local 1CTA non-FPSAN `N=128,K=128`, linear
  accumulator, B-scale `reshape -> trans -> reshape`, unpadded storage, no
  extra user. It reproduced `3/3` fresh subprocesses with `16109/16384`
  mismatches and matching PTX/LLIR
  `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`.
- Classification update: `FZ-0013` should be treated as a scaled-MMAv5 scale
  descriptor-view wrong-result bucket covering both A-scale and B-scale
  descriptor views. Direct no-view controls and unpadded same-shape
  `slice -> index` A/B scale views passed, so the trigger remains tied to
  descriptor-view scale-fragment mapping/rematerialized scale storage rather
  than generic scaled-MMAv5 opcode selection.

### Round 14 local broad MMAv5 descriptor selector

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_mma_descriptor_broad_round14.md`
- Required build: `make -j8` no-op.
- Selector:
  `(mma and (descriptor or runtime_selector or indexed_acc or acc_subslice_view or scale_descriptor)) and not reports`
  collected `253/1615` rows.
- Split-4 result:
  `239 passed, 14 skipped` (`50/14`, `64`, `64`, `61` by shard).
- Classification: no new bucket. Checked-in plain/scaled MMAv5 descriptor and
  indexed-accumulator coverage remains green.

### Round 14 local copy risk selector

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_copy_risk_round14.md`
- Required build: `make -j8` no-op.
- Selector:
  `(cp_no_scales_warpx2 or cp_no_scales_twocta or cp_scales_warpx4 or cp_scales) and not reports`
  collected `135/1615` rows.
- Split-4 result:
  `135 passed` (`34`, `34`, `34`, `33` by shard).
- Classification: no new bucket. Checked-in copy rows remain green next to
  Lane AI's temporary multi-region mbarrier composition fuzzing.

### Round 14 local structural-fuzzer smoke gate

- Time: 2026-04-21
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_structural_smoke_round14b.md`
- Required build: `make -j8` no-op.
- Command:
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py`
- Result: `9 passed, 24 xfailed in 10.00s`.
- Classification: no new bucket. The checked-in structural-fuzzer sentinels
  remain stable.

### Round 15 Lane AK generic memdesc/control-flow fuzzing

- Time: 2026-04-21 11:30 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_generic_memdesc_control_round15.md`
- Required build: `make -j8` no-op.
- Temporary probes:
  `/tmp/tmem_generic_memdesc_control_round15_probe.py` and
  `/tmp/tmem_r15_scaled_probe.py`.
- Checked-in control:
  `python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'`
  passed as `11 xfailed`.
- Result: no new independent `FZ-*` bucket.
- Classification:
  - `FZ-20260421-0001` broadened to branch/helper-selected memdesc values
    feeding `ld/st`, `tcgen05.copy`, and plain MMAv5;
  - `FZ-20260421-0002` broadened to loop-carried helper-returned chain0
    `ld/st` views with `8064/8192` mismatches;
  - `FZ-20260421-0013` broadened to scaled-MMAv5 scale descriptor-view rows
    selected through runtime branch control flow.

### Round 15 Lane AM scaled-MMAv5 multi-op and dynamic scale fuzzing

- Time: 2026-04-21 11:30 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_multi_mma_round15.md`
- Required build: `make -j8` no-op.
- Temporary probe:
  `/tmp/tmem_scaled_multi_mma_round15_probe.py`.
- Result: `7 passed, 5 failed`.
- Green controls: multiple scaled MMA operations in one kernel, reused scale
  descriptors, static accumulator subslices, `use_acc` sequencing, and
  `mxfp8`/`mxfp4`/`nvfp4` format variation.
- Existing bucket broadening:
  - three A/B scale descriptor-view rows stayed under `FZ-20260421-0013`
    with `16107/16384`, `16116/16384`, and `16373/16384` mismatches.
- New report-only candidate:
  - `FZ-20260421-0015`: direct B-scale TMEM descriptors selected by runtime
    branch feed `tcgen05_mma_scaled`, compile and execute, and emit matching
    PTX/LLIR scaled-MMA opcodes, but produce NaN-heavy wrong results
    (`16380-16381/16384` mismatches and `96` NaNs). Minimize this before
    repair; do not fold it into `FZ-0013` unless a smaller repro shows the
    same scale-fragment/view rematerialization trigger.

### Round 15 local twoCTA/high-CGA selector

- Time: 2026-04-21 11:30 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_twocta_highcga_round15.md`
- Selector:
  `(twocta or cga_roundtrip or layout_in_4cta_context) and not reports`
  collected `331/1615`.
- Split-4 result:
  `294 passed, 37 skipped`.
- Classification: no new bucket; checked-in twoCTA/high-CGA runtime surface
  stayed green next to existing report-only `FZ-20260421-0010` and
  `FZ-20260421-0014`.

### Round 15 FZ-0015 scaled B-scale dynamic-selection minimization

- Time: 2026-04-21 11:31 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_fz0015_min_round15.md`
- Temporary probe:
  `/tmp/tmem_fz0015_min_round15_probe.py`.
- Result: `10` collected; split-4 ran as `5 passed, 5 failed`.
- Pass boundaries:
  direct B-scale control, constexpr distinct B-scale selection, same-object
  dynamic B-scale branch, and A-scale dynamic selection.
- Failing rows:
  distinct B-scale descriptors selected by runtime branch or loop before
  scaled-MMAv5, with `16381-16383/16384` mismatches; extra selected-scale user
  still fails.
- Classification: `FZ-20260421-0015` remains distinct. Current smallest
  trigger is two distinct direct B-scale `TensorMemoryScalesLayout`
  descriptors with identical payloads selected by runtime control flow before
  `tcgen05_mma_scaled`.

### Round 15 FZ-0015 TTGIR discriminator

- Time: 2026-04-21 11:34 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_fz0015_ttgir_round15.md`
- Parent-view discriminator is probe-limited: tensor-memory
  `memdesc_subslice` forms fail before lowering for all rows, including direct
  controls.
- TTGIR inspection:
  - direct and constexpr controls feed concrete B-scale memdescs (`%b0` or
    `%b1`) into `ttng.tc_gen5_mma_scaled`;
  - branch failures feed an `arith.select` result between B-scale memdescs into
    scaled MMAv5;
  - loop failures feed an `scf.for` memdesc iter_arg result into scaled MMAv5.
- Classification: `FZ-20260421-0015` is currently best described as a
  scaled-MMAv5 lowering/planner gap for merged B-scale memdesc SSA values.

### Round 16 Lane AN FZ-0015 minimization

- Time: 2026-04-21 11:35 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_fz0015_min_round16.md`
- Probe:
  `/tmp/tmem_fz0015_min_round16_probe.py`.
- Smallest stable trigger: two distinct direct B-scale
  `TensorMemoryScalesLayout` descriptors with identical payloads selected by
  runtime control flow and consumed as the B-scale operand of
  `ttng.tc_gen5_mma_scaled`.
- Green controls: direct B-scale, constexpr distinct B-scale selection,
  constexpr-folded runtime arm disabled, same-object runtime B-scale
  selection, runtime A-scale selection, and selected-descriptor TMEM load.
- Bucket breadth: branch, loop, helper, pass-through, extra selected-scale
  user, two scaled MMA ops, `use_acc=False`, `N=64`, `K=256`, `mxfp4`, and
  `nvfp4` all reproduce or preserve the `FZ-0015` wrong-result family.
- Selected-descriptor load discriminator passed `4/4`, so the selected B-scale
  TMEM descriptor can be read correctly outside scaled MMAv5.
- Classification: `FZ-20260421-0015` stays distinct from `FZ-0001`,
  `FZ-0007`, and `FZ-0013`; current owner hypothesis is the scaled-MMAv5
  B-scale operand lowering/codegen path for merged memdesc SSA values.

### Round 17 local copy/ldst selector

- Time: 2026-04-21 11:38 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_copy_ldst_round17.md`
- Checked-in structural-fuzzer smoke: `9 passed, 24 xfailed`.
- Runtime-matrix selector:
  `(cp_no_scales or cp_scales or ldst_descriptor_compositions or ldst_descriptor_roundtrip) and not reports`
  collected `344/1615`.
- Split-4 result:
  `289 passed, 55 skipped`.
- Classification: no new independent `FZ-*` bucket and no runtime miscompile.

### Round 17 Lane AO FZ-0015 lowering audit

- Time: 2026-04-21 11:39 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_fz0015_lowering_audit_round17.md`
- Artifacts:
  `/tmp/tmem_fz0015_lowering_audit_round17/`.
- Result: no new independent bucket; `FZ-20260421-0015` remains distinct.
- Evidence:
  - selected B-scale loads pass, so the selected descriptor is readable outside
    scaled MMAv5;
  - saved TTGIR for the failing row is verifier-clean;
  - runtime-selected distinct B-scale fails, but direct, constexpr,
    same-object B-scale, and runtime-selected A-scale controls pass;
  - LLVM/PTX preserve the dynamic B-scale memdesc as a scalar
    `select` / `selp.b32` SFB address operand.
- Likely owner: scaled MMAv5 LLVM lowering around `convertScaledDot` /
  `createScaledGen5MMA` in `MMAv5.cpp`, or an unmodeled SFB-address operand
  constraint.

### Round 17 Lane AP copy/ldst mixed fuzzing

- Time: 2026-04-21 11:41 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_copy_ldst_mixed_round17.md`
- Temporary probe:
  `/tmp/tmem_copy_ldst_mixed_round17_probe.py`.
- Result: final probe passed as `7 passed`; `AP-007` classified an existing
  `FZ-20260421-0014` proxy-fence reproduction.
- Adjacent checked-in selector:
  `(cp_no_scales and (warpx2 or twocta) and not reports) or (ldst_descriptor and not reports)`
  collected `240/1615` and passed split-4 as `179 passed, 61 skipped`.
- Classification: no new independent `FZ-*` bucket and no runtime miscompile.
  AP-004/AP-006 are negative contrasts for `FZ-20260421-0014`: two independent
  2CTA copy regions passed under different wait/readback placement than the
  reproducing AP-007 row.

### Round 17 local clean-boundary selector

- Time: 2026-04-21 11:41 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_clean_boundaries_round17.md`
- Selector:
  `(reports_clean_unsupported or reports_clean_error or reports_tmem_oor or subword_dtypes_report_clean_error or non_f32_contract)`
  collected `199/1615`.
- Split-4 result:
  `199 passed`.
- Classification: no clean-boundary drift into late illegal ops, crashes,
  false unsupported diagnostics, runtime miscompiles, or new `FZ-*` buckets.

### Round 17 Lane AQ clean-boundary diagnostics

- Time: 2026-04-21 11:42 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_clean_boundaries_round17.md`
- Checked-in selector:
  `reports_clean_unsupported or reports_clean_error or reports_tmem_oor`
  collected `157/1615` and passed split-4.
- Focused temporary probe:
  `/tmp/tmem_clean_boundaries_round17_probe.py`.
- Result: descriptor-view, high-CGA, subword, non-f32 `ld.red`, scale-shape,
  and parent-view subslice clean-boundary probes passed without late illegal
  op signatures.
- Classification: no new independent `FZ-*` bucket. Existing dynamic/generic
  memdesc late failures remain `FZ-20260421-0001`; high-CGA clean diagnostics
  remain associated with `FZ-20260421-0010` when otherwise-legal
  instruction-local layouts are launched in incompatible CGA contexts.

### Round 18 local descriptor ld/st plus cp_scales selector

- Time: 2026-04-21 11:48 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_ldst_cpscales_round18.md`
- Selector:
  `(ldst_descriptor_compositions or ldred_descriptor or cp_scales) and not reports`
  collected `62/1615`.
- Split-4 result:
  `62 passed`.
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*` bucket.

### Round 22 local split-N and fixed-offset baseline

- Time: 2026-04-21 12:03 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_splitn_fixed_round22.md`
- Selector:
  `(splitn or fixed_offset) and not ld_red` collected `33/1615`.
- Split-4 result:
  `33 passed` (`9/9/9/6`).
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*` bucket.

### Round 22 local structural fuzzer guardrail

- Time: 2026-04-21 12:03 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_structural_round22.md`
- Selector:
  `structural_fuzzer and not allocator_crash and not optimizer_crash`
  collected `30/33`.
- Split-4 result:
  `9 passed, 21 xfailed`.
- Classification: no unexpected pass, unexpected failure, runtime miscompile
  outside known sentinels, compiler crash outside known xfail rows, or new
  independent `FZ-*` bucket.

### Round 22 Lane BB ld.red layout fuzzing

- Time: 2026-04-21 12:22 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_layouts_round22.md`
- Broad selector:
  `ld_red and not reports and not resource` completed as
  `237 passed, 6 failed`; all six failures are existing `FZ-20260421-0012`
  M64 row-basis rows.
- Additional controls:
  structural `ldred` `3 passed, 8 xfailed`; fallback dtype/scales/n-sharded
  `60 passed`; M64 split-N/modifier controls `38 passed`; descriptor-chain
  reductions `30 passed`; temporary M64/M128 cross-product probe produced
  `8` M64 row-basis failures and `16` adjacent controls passed.
- Classification: no new independent `FZ-*`. The lane sharpens `FZ-0012` as
  specific to the M64 split-N destination-layout planner: M64 row-basis rows
  fail across min/max/abs/NaN/default/explicit split-N, while M64 column-only
  and M=128,N=32 row-permuted hardware reductions pass and emit `.ld.red.`
  opcodes.

### Round 22 Lane BA scaled dynamic scale descriptors

- Time: 2026-04-21 12:58 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_dynamic_scales_round22.md`
- Temporary probe:
  `/tmp/tmem_scaled_dynamic_scales_round22_probe.py`.
- Result:
  `10` direct selected-scale controls passed; `6` legal
  reshape/permute/reshape scale descriptor-view rows miscompiled under
  existing `FZ-20260421-0013`. Nearby checked-in descriptor-view baseline
  passed as `4 passed`.
- Classification: no new independent `FZ-*`. The lane sharpens `FZ-0013`:
  scaled-MMAv5 consuming scale descriptor views miscompiles while side-channel
  scale loads are correct. It also provides a negative contrast for `FZ-0015`:
  simplified direct runtime-selected B-scale descriptors did not reproduce the
  allocation-order-sensitive failure.

### Round 22 Lane BC copy/ld-st generic descriptors

- Time: 2026-04-21 12:55 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_copy_ldst_generic_round22.md`
- Temporary probe:
  `/tmp/tmem_round22_copy_generic_probe.py`.
- Checked-in selector:
  `(cp_no_scales and (indexed_view or subslice_view or warpx2 or twocta or linear) and not reports) or (generic_pass and not mma) or (ldst_descriptor and not reports)`
  collected `329/1648` and completed as
  `254 passed, 61 skipped, 14 xfailed`.
- Scale-copy/high-CGA diagnostic selector:
  `36 passed`.
- Classification: no new independent `FZ-*`. The lane sharpens `FZ-0001`
  with branch-yielded generic copy descriptors leaving illegal
  `ttg.memdesc_index`, and sharpens `FZ-0002` with multi-consumer
  branch-selected ld/st descriptor miscompiles. Loop-carried copy descriptors
  and `warpx2::01_23` branch-selected copy descriptors passed as contrasts.

### Round 23 local plain MMAv5 baseline

- Time: 2026-04-21 13:05 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_plain_mma_round23.md`
- Selector:
  `(mma_twocta or plain_kinds or tma_tf32 or indexed_acc_view) and not mma_scaled and not reports and not clean and not i8`
  collected `255/1615`.
- Split-4 result:
  `255 passed` (`64/64/64/63`).
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*` bucket.

### Round 23 local subword baseline

- Time: 2026-04-21 13:18 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_subword_round23.md`
- Selector:
  `subword and not reports and not cp_no_scales_warpx2` collected `42/1615`.
- Split-4 result:
  `42 passed` (`11/11/11/9`).
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*` bucket.

### Round 23 Lane BE scaled accumulator descriptors

- Time: 2026-04-21 12:20 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_accumulator_round23.md`
- Checked-in accumulator selector:
  `mma_scaled and acc and not reports and not fz0015` collected `206/1615`
  and passed as `206 passed`.
- Broader adjacent selector:
  `mma_scaled and not reports and not fz0015` collected `243/1615` and passed
  as `243 passed`.
- Dynamic selected accumulator probe:
  branch/helper/loop selector-0 rows failed with `4413/16384` mismatches and
  NaNs while the same selected descriptor's `tmem_load` side channel stayed
  correct.
- Classification: no new independent `FZ-*`; sharpens `FZ-20260421-0007` as
  a scaled-MMAv5 dynamic selected accumulator-view materialization issue,
  distinct from scale descriptor-view `FZ-0013` and selected B-scale
  `FZ-0015`.

### Round 23 local ld/st descriptor guardrail

- Time: 2026-04-21 12:32 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_ldst_descriptor_round23.md`
- Selector:
  `(ldst_descriptor or descriptor_view) and not reports and not mma_scaled`
  collected `149/1615`.
- Split-4 result:
  `88 passed, 61 skipped`.
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*` bucket.

### Round 23 Lane BD high-CGA ownership

- Time: 2026-04-21 13:20 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_high_cga_round23.md`
- Checked-in selector `cga or cta_per_cga` passed as `16 passed`.
- High-CGA controls passed: TMA multicast `2`, MMAv5 multicast/commit `4`,
  scaled-MMAv5 copy+MMA `3`, and TMA+MMA `2`.
- Temporary probes classified local 1CTA/2CTA ld/st, `ld.red`, no-scales copy,
  and scales-copy rows inside 4/8/16 CTA contexts as existing `FZ-0010`
  CTA-count diagnostics.
- Classification: no new independent `FZ-*`.

### Round 23 Lane BF compiler-only TMEM audit

- Time: 2026-04-21 12:13 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_compiler_lit_round23.md`
- Green lit baselines passed: TMEM layouts, MMAv5 lowering, proxy fence,
  interleave/hoist/NVWS TMEM, and memdesc subview split.
- New candidate `FZ-20260421-0016`: `relayout_tritongpu.mlir` crashes while
  verifying `ttng.tmem_alloc` with an unencoded tensor operand and
  TensorMemoryScales result, via `verifyTMEMOperand` ->
  `computeTMemLdStEncodingInfo` -> `toLinearEncoding`.
- Existing saved repros reconfirmed `FZ-0014`, `FZ-0008`, and `FZ-0009`.

### Round 23 local multi-CTA guardrail

- Time: 2026-04-21 12:42 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_multicta_round23.md`
- Selector:
  `(twocta or multicast or cta or cga) and tmem and not reports` collected
  `430/1615`.
- Split-4 result:
  `393 passed, 37 skipped`.
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*` bucket.

### Round 24 local compiler lit baseline

- Time: 2026-04-21 13:30 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_lit_baseline_round24.md`
- Result:
  `4` lit tests passed: TMEM layouts, MMAv5 lowering, interleave TMEM, and
  memdesc subview split.
- Classification: no compiler-only regression or new `FZ-*` bucket.

### Round 24 local clean-boundary sweep

- Time: 2026-04-21 13:40 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_clean_boundary_round24.md`
- Selector:
  `reports_clean or reports_tmem_oor or clean_unsupported or clean_error`
  collected `184/1615`.
- Split-4 result:
  `184 passed` (`46/46/46/46`).
- Classification: no unexpected failure, unexpected pass, compiler crash, or
  new independent `FZ-*` bucket.

### Round 24 Lane BI structural generator prototype

- Time: 2026-04-21 13:35 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_structural_generator_round24.md`
- Prototype:
  `/tmp/tmem_structural_generator_round24.py`.
- Rerun result:
  `12` generated cases: `7` green, `3` clean unsupported, `2` existing
  `FZ-20260421-0012`.
- Classification: no new independent `FZ-*`; generated M64 row-basis
  `ld.red` rows stayed in `FZ-0012`, lifted rank-4 descriptor roundtrips stayed
  clean unsupported, and the report records a promotion plan for deterministic
  checked-in generator seeds.

### Round 24 Lane BG FZ-0016 minimization

- Time: 2026-04-21 13:40 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_fz0016_min_round24.md`
- Minimal reproducer:
  `/tmp/tmem_fz0016_round24/minimal_unencoded_scales_i8.mlir`.
- Result:
  parse-only `triton-opt` aborts in `TMEMAllocOp::verify` via
  `toLinearEncoding` on an unencoded operand tensor. The crash is not specific
  to relayout or conversion pipelines.
- Classification: sharpens `FZ-20260421-0016`; no new independent `FZ-*`.

### Round 24 local selector-permuted copy rows

- Time: 2026-04-21 13:50 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_selector_copy_round24.md`
- Selector:
  `dynamic or generic or branch or selector` collected `4/1615` current
  selector-permuted no-scales copy rows.
- Runtime result:
  `4 passed`.
- Classification: no new independent `FZ-*`; narrow green guardrail only.

### Round 24 local structural guardrail

- Time: 2026-04-21 13:55 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_structural_round24.md`
- Selector:
  `not allocator_crash and not optimizer_crash` collected `30/33`.
- Split-4 result:
  `9 passed, 21 xfailed`.
- Classification: no unexpected pass/fail, additional compiler crash, runtime
  miscompile outside known sentinels, or new independent `FZ-*`.

### Round 24 Lane BH mixed scaled-MMAv5 operands

- Time: 2026-04-21 13:45 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_mixed_operands_round24.md`
- Result:
  no new independent `FZ-*`. Direct selected B-scale plus runtime-selected
  accumulator-parent mixed rows passed. Any mixed row consuming scale
  reshape/permute/reshape descriptor views miscompiled with clean side-channel
  loads and matching opcode counts, staying in `FZ-0013`. Exact `FZ-0015`
  old reproducer still fails.
- Classification: keeps `FZ-0007`, `FZ-0013`, and `FZ-0015` separate.

### Round 24 local structural crash guardrail

- Time: 2026-04-21 14:00 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_structural_crash_round24.md`
- Selector:
  `allocator_crash or optimizer_crash`.
- Result:
  `3 xfailed`.
- Classification: no changed crash mode; rows remain existing `FZ-0005`,
  `FZ-0008`, and `FZ-0009`.

### Round 25 local no-scale copy baseline

- Time: 2026-04-21 14:10 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_copy_round25.md`
- Selector:
  `cp_no_scales and not reports and not high and not cga and not selector and not subword`
  collected `201/1615`.
- Split-4 result:
  `197 passed, 4 skipped`.
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*`.

### Round 25 local scale-copy baseline

- Time: 2026-04-21 14:18 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_cpscales_round25.md`
- Selector:
  `cp_scales and not reports` collected `33/1615`.
- Split-4 result:
  `33 passed` (`9/9/9/6`).
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*`.

### Round 25 Lane BK FZ-0016 verifier code audit

- Time: 2026-04-21 14:22 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_fz0016_code_audit_round25.md`
- Result:
  broadened `FZ-20260421-0016` to shared `ttng.tmem_alloc`,
  `ttng.tmem_load`, and `ttng.tmem_store` verifier crashes for unencoded
  register tensors. Encoded controls parse cleanly or diagnose cleanly;
  `ttng.tmem_copy` does not use this exact unencoded-register path.
- Classification: no new independent `FZ-*`.

### Round 25 local FZ-0016 relayout baseline

- Time: 2026-04-21 14:25 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_fz0016_relayout_round25.md`
- Result:
  original `test/Conversion/relayout_tritongpu.mlir` pipeline exits `134`
  with the known `toLinearEncoding` verifier abort.
- Classification: existing `FZ-20260421-0016`; no new independent `FZ-*`.

### Round 25 local descriptor ld/st non-roundtrip baseline

- Time: 2026-04-21 14:24 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_ldst_descriptor_nonroundtrip_round25.md`
- Selector:
  `ldst_descriptor and not reports and not roundtrip` collected `53/1615`.
- Result:
  `53 passed` (`14/14/14/11`).
- Classification: no new independent `FZ-*`.

### Round 25 Lane BJ dynamic structural generator

- Time: 2026-04-21 14:28 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_structural_generator_dynamic_round25.md`
- Prototype:
  `/tmp/tmem_structural_generator_dynamic_round25.py`.
- Result:
  `22` generated seeds. Post-classification: `8` existing `FZ-0001`, `5`
  direct ld/st `FZ-0002` plus `2` likely ld.red consumer extensions, `3`
  clean copy unsupported, `4` green.
- Classification: no new independent `FZ-*`; promotion-ready dynamic
  descriptor generator seed table produced.

### Round 25 local dynamic structural guardrail

- Time: 2026-04-21 14:35 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_dynamic_structural_round25.md`
- Selector:
  `generic_pass or dynamic_index or loop_carried or scaled_mma_acc_subslice`
  collected `12/33`.
- Result:
  `12 xfailed`.
- Classification: no unexpected pass, changed crash mode, runtime miscompile
  outside existing xfail set, or new independent `FZ-*`.

### Round 25 local scaled descriptor-view control

- Time: 2026-04-21 14:35 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_scaled_descriptor_view_round25.md`
- Selector:
  `mma_scaled and descriptor_view and not reports`.
- Result:
  `4 passed`.
- Classification: no new independent `FZ-*`; checked-in green control for
  report-only `FZ-20260421-0013`.

### Round 25 Lane BJ dynamic descriptor structural generator

- Time: 2026-04-21 14:28 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_structural_generator_dynamic_round25.md`
- Temporary prototype:
  `/tmp/tmem_structural_generator_dynamic_round25.py`
- Result:
  rerun executed `22` generated dynamic descriptor SSA seeds across ld/st,
  copy/readback, and f32 `ld.red` consumers.
- Classification: no new independent `FZ-*`. Seeds sharpen existing
  `FZ-20260421-0001` and `FZ-20260421-0002`; direct-slice loop-carried copy
  and branch/loop-carried direct-slice `ld.red` are green contrasts.

### Round 25 local descriptor load/store non-roundtrip baseline

- Time: 2026-04-21 14:30 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_ldst_descriptor_nonroundtrip_round25.md`
- Selector:
  `ldst_descriptor and not reports and not roundtrip` collected `53/1615`.
- Split-4 result:
  `53 passed` (`14/14/14/11`).
- Skipped contrast:
  `ldst_descriptor_roundtrip and not reports` collected `51/1615`, but all
  rows pre-skipped at the current Blackwell TMEM allocation limit.
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*`.

### Round 22 Lane BA scaled-MMAv5 dynamic scale descriptors

- Time: 2026-04-21 12:58 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_dynamic_scales_round22.md`
- Temporary probe:
  `/tmp/tmem_scaled_dynamic_scales_round22_probe.py`
- Result:
  final probe ran `16` cases with expected classifications: `10` direct
  dynamic selected-scale controls passed, and `6` reshape/permute/reshape
  scale-view rows miscompiled under existing `FZ-20260421-0013`.
- Checked-in nearby selector:
  `mma_scaled and descriptor_view and not reports` passed as
  `4 passed, 1611 deselected`.
- Classification: no new independent `FZ-*`. Direct runtime-selected
  A/B-scale descriptors, nested/helper/loop-carried selected direct B-scale
  descriptors, mixed selected-scale side-channel loads, two-MMA direct
  selected-scale, and accumulator-first/last direct controls stayed green in
  this simplified same-payload probe. Legal same-layout scale descriptor views
  still produce large wrong-result mismatches while side-channel loads from the
  same selected scale descriptors are correct, sharpening `FZ-0013` and
  providing a negative contrast for the narrower direct selected-B-scale
  allocation-order `FZ-0015`.

### Round 22 Lane BC copy/ld-st generic descriptors

- Time: 2026-04-21 12:55 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_copy_ldst_generic_round22.md`
- Temporary probe:
  `/tmp/tmem_round22_copy_generic_probe.py`
- Checked-in selector:
  `(cp_no_scales and (indexed_view or subslice_view or warpx2 or twocta or linear) and not reports) or (generic_pass and not mma) or (ldst_descriptor and not reports)`
  collected `329/1648` and ran split-4 as
  `254 passed, 61 skipped, 14 xfailed`.
- Additional scale-copy/high-CGA selector:
  `36 passed`.
- Classification: no new independent `FZ-*`. The temporary branch-yielded
  linear-copy descriptor row hits existing `FZ-20260421-0001` with illegal
  `ttg.memdesc_index` left for LLVM conversion; branch-selected multi-consumer
  ld/st descriptors hit existing `FZ-20260421-0002` wrong results. Loop-carried
  copy descriptors and 1CTA/2CTA `warpx2::01_23` branch-selected copy
  descriptors passed, narrowing the failure to generic descriptor SSA lowering
  rather than the copy planner or hardware-copy schedule.

### Round 18 local MMAv5 and scaled-MMAv5 selector

- Time: 2026-04-21 11:54 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_mma_scaled_round18.md`
- Selector:
  `(mma_scaled and not reports and not fz0015 and not descriptor_view) or (mma_twocta and not reports_clean and not reports and not i8)`
  collected `342/1615`.
- Split-4 result:
  `342 passed`.
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*` bucket. Report-only
  `FZ-20260421-0015` rows were intentionally excluded.

### Round 18 Lane AR proxy-fence intervals

- Time: 2026-04-21 11:56 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_proxy_fence_intervals_round18.md`
- Temporary probe:
  `/tmp/tmem_proxy_fence_intervals_round18_probe.py`.
- Result: `13 passed`, with existing proxy-fence compiler failures caught and
  classified as `FZ-20260421-0014`.
- Adjacent checked-in selector:
  `cp_no_scales and twocta and not reports` collected `63/1615` and passed as
  `63 passed`.
- Classification: no new independent `FZ-*`. `FZ-0014` now appears tied to
  sequential cross-CTA mbarrier interval construction: wait region 0 before
  initializing/starting region 1 reproduces even without TMEM readback, while
  initializing all mbarriers before the copy/commit/wait phase passes for two
  and three regions.

### Round 18 Lane AS dynamic clean boundaries

- Time: 2026-04-21 11:56 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_dynamic_clean_boundaries_round18.md`
- Temporary probe:
  `/tmp/tmem_dynamic_clean_boundaries_round18_probe.py`.
- Checked-in selector across structural fuzzer plus runtime matrix collected
  `170/1648` and passed split-4 as `159 passed, 11 xfailed`.
- Classification: no new independent `FZ-*`. The lane strengthened existing
  `FZ-0001` with a load-only dynamic-index illegal-op row, `FZ-0002` with
  tuple-like dynamic descriptor capture wrong results, `FZ-0010` with a clean
  high-CGA diagnostic, and `FZ-0015` with the selected direct B-scale
  scaled-MMAv5 mismatch.

### Round 18 proxy-fence plain-mbarrier follow-up

- Time: 2026-04-21 11:57 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_proxy_fence_plain_mbarrier_round18.md`
- Temporary probe:
  `/tmp/tmem_proxy_fence_intervals_round18_probe.py`.
- Result: follow-up selected `2/15` rows and passed as `2 passed`, with both
  rows classified as existing `FZ-20260421-0014`.
- Classification: no new independent `FZ-*`. One legal 2CTA no-scales
  copy-tracked mbarrier plus one independent plain mbarrier interval is enough
  to reproduce proxy-fence insertion failure in either order, so the bug does
  not require two copy-tracked regions.

### Round 18 proxy-fence plain-only follow-up

- Time: 2026-04-21 11:58 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_proxy_fence_plain_only_round18.md`
- Temporary probe:
  `/tmp/tmem_proxy_fence_intervals_round18_probe.py`.
- Result: follow-up selected `2/17` rows and passed as `2 passed`, with the
  sequential row classified as existing `FZ-20260421-0014`.
- Classification: no new independent `FZ-*`. `FZ-0014` is not
  TMEM-copy-specific: two sequential plain init/arrive/wait mbarrier intervals
  with no TMEM operations reproduce the same proxy-fence insertion failure,
  while initializing both mbarriers before either arrive/wait use passes.

### Round 19 local multicast sanity selectors

- Time: 2026-04-21 12:05 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_multicast_round19.md`
- Runtime-matrix selector:
  `(layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma) and not reports_clean`
  collected `10/1615` and passed as `10 passed`.
- `test_core.py` selector:
  `multicast and (tcgen05 or tma_mma or mma_scaled)` collected `9/18114` and
  passed as `9 passed`; one split shard was empty.
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*` bucket.

### Round 19 Lane AV dynamic subslice mixed consumers

- Time: 2026-04-21 12:09 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_dynamic_subslice_mixed_consumers_round19.md`
- Temporary probe:
  `/tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py`.
- Result: no new independent `FZ-*`. Scaled-MMAv5 selected low-subslice
  accumulator use miscompiled with `4413/16384` mismatches and NaNs, while the
  same selected descriptor's `tmem_load` was correct and plain MMAv5 passed for
  both selected subslices.
- Classification: strengthens `FZ-20260421-0007`.
- Late extension: branch/helper/loop selected direct B-scale descriptors with
  both a scale-load side channel and scaled-MMAv5 B-scale operand passed for
  both selectors. This is a negative contrast for `FZ-20260421-0015` and shows
  that not every selected direct B-scale descriptor plus extra load user
  reproduces the minimized `FZ-0015` wrong result.

### Round 19 local structural xfail sentinels

- Time: 2026-04-21 12:09 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_structural_xfail_round19.md`
- Selector:
  `(fz0015 or generic_pass or dynamic_index or proxy_fence or reports_fz or reports_bug) and not performance`
  collected `11/1648`.
- Split-4 result:
  `11 xfailed`.
- Classification: no unexpected pass, unexpected failure, runtime miscompile
  outside known xfails, or new independent `FZ-*`.

### Round 19 local B-scale controls

- Time: 2026-04-21 12:14 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_bscale_controls_round19.md`
- Selector:
  `mma_scaled and bscale and not reports` collected `3/1615`.
- Result:
  `3 passed`.
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*`; green controls adjacent to
  `FZ-20260421-0015`.

### Round 19 Lane AU plain proxy-fence reproducer

- Time: 2026-04-21 12:18 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_proxy_fence_plain_reproducer_round19.md`
- Saved reproducer:
  `/tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir`.
- Result: branch, upstream main
  `dea2e9d7324309fdc9198144669621f57f3704a2`, and merge-base
  `2c7ce4925d37802dd84dfde1f6458cae19485617` all fail
  `triton-opt --run-reproducer` with the same proxy-fence insertion
  diagnostic.
- Classification: `FZ-20260421-0014` plain-only sequential mbarrier failure is
  preexisting upstream, not branch-specific. The minimized reproducer has
  `ttg.tensor_memory_size = 0` and no TMEM/tcgen05 ops.

### Round 19 local ld.red selector

- Time: 2026-04-21 12:20 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_ldred_round19.md`
- Selector:
  `ld_red and not reports and not resource` collected `243/1615`.
- Runtime:
  `237 passed, 6 failed`.
- Classification: existing `FZ-20260421-0012`; no new independent `FZ-*`.
  The red rows are checked-in M64 row/col-permuted split-N `ld.red` runtime
  tests that reject effective non-identity row bases with `unsupported dst
  layout`. Focused contrast: `row_reverse_n32-min` fails, while
  `col_reverse_n32-min` and explicit `col_reverse_n32` split-N controls pass.

### Round 19 Lane AW FZ-0015 side channel

- Time: 2026-04-21 12:23 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_fz0015_side_channel_round19.md`
- Temporary probe:
  `/tmp/tmem_fz0015_side_channel_round19_probe.py`.
- Result: `9 passed`, with six expected wrong-result rows classified as
  existing `FZ-20260421-0015`.
- Classification: no new independent `FZ-*`. The exact selected distinct
  B-scale shape still miscompiles while the selected-scale side-channel load is
  correct (`0/512` byte mismatches). Same-global-source controls still fail.
  Moving accumulator allocation before scale allocations makes all selected
  B-scale branch/helper/loop rows pass, making allocation order the strongest
  current discriminator.

### Round 20 local descriptor roundtrip and copy/scales selector

- Time: 2026-04-21 12:30 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_roundtrip_copy_round20.md`
- Selector:
  `(ldst_descriptor_roundtrip or cp_scales or cp_no_scales_warpx2_01_23 or cp_no_scales_warpx4) and not reports and not tmem_oor`
  collected `102/1615`.
- Split-4 result:
  `51 passed, 51 skipped`.
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*` bucket.

### Round 20 local 2CTA MMAv5 and scaled accumulator-subslice selector

- Time: 2026-04-21 12:35 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_mma_twocta_round20.md`
- Selector:
  `((mma_twocta and (indexed_acc_view or acc_subslice_view_plain_kinds or plain_kinds_use_acc or tma_tf32_b_transposed_descriptor_use_acc)) or (mma_scaled and twocta_acc_subslice)) and not reports and not clean`
  collected `90/1615`.
- Split-4 result:
  `90 passed`.
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*` bucket.

### Round 20 Lane AX FZ-0015 allocation order

- Time: 2026-04-21 12:40 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_fz0015_allocation_order_round20.md`
- Result: no new independent `FZ-*`. All scale descriptors before the real
  accumulator fail; dummy scale/acc allocations between scale descriptors and
  the real accumulator still fail; the real accumulator before B scales or
  between A/B scales passes.
- Classification: strengthens `FZ-20260421-0015`; the selected B-scale load
  side channel is clean in every row, so the issue remains scaled-MMAv5 B-scale
  operand consumption/lowering and is sensitive to real accumulator allocation
  order relative to scale descriptors.

### Round 20 local clean-negative and structural xfail selector

- Time: 2026-04-21 12:40 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_clean_xfail_round20.md`
- Selector:
  `(reports_clean_unsupported or reports_clean_error or reports_tmem_oor or non_f32_contract or subword_dtypes_report_clean_error or generic_pass or dynamic_index) and not block_descriptor`
  collected `208/1648`.
- Split-4 result:
  `197 passed, 11 xfailed`.
- Classification: no unexpected pass, unexpected failure, runtime miscompile
  outside known rows, compiler crash outside known xfails, or new independent
  `FZ-*`.

### Round 21 local higher-rank and multidim descriptor selector

- Time: 2026-04-21 12:47 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_higher_rank_round21.md`
- Selector:
  `(higher_rank or multidim_slice or half_rows or rank5 or ldst_descriptor_higher_rank) and not reports`
  collected `102/1615`.
- Split-4 result:
  `82 passed, 20 skipped`.
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*` bucket.

- Round 10 Lane N recommends a future strict runtime xfail under the
  report-only `FZ-20260421-0011` once the plain-MMAv5 runtime-selector-index
  miscompile can be minimized without changing failure mode. Round 12 Lane S
  narrowed that to an FPSAN-pinned sentinel candidate; do not promote a
  non-FPSAN variant because it degenerates into `FZ-20260421-0001`.

### Round 21 Lane AY FZ-0015 IR compare

- Time: 2026-04-21 12:03 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_fz0015_ir_compare_round21.md`
- Artifacts:
  `/tmp/tmem_fz0015_ir_compare_round21_probe.py`,
  `/tmp/tmem_fz0015_ir_compare_round21_probe.log`, and
  `/tmp/tmem_fz0015_ir_compare_round21/`.
- Classification: no new independent `FZ-*`; this sharpens
  `FZ-20260421-0015`. The selected B-scale SSA is preserved into
  `ttng.tmem_load` and `ttng.tc_gen5_mma_scaled`, and the side-channel load is
  correct in fail/pass rows. The failing scale-first row lowers scaled-MMAv5
  operands to `base+0/4/8/12`, while passing controls separate accumulator and
  B-scale operands by an accumulator-sized offset.

### Round 21 Lane AZ M64 ld.red minimization

- Time: 2026-04-21 12:03 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_m64_min_round21.md`
- Classification: no new independent `FZ-*`; this sharpens
  `FZ-20260421-0012`. M64 f32 `ld.red` rows with non-identity effective row
  bases fail before runtime with `unsupported dst layout`. Column-only
  permutations pass and emit hardware reductions. Explicit `auto`, `32x32b`,
  `16x32bx2`, and `32x32b_splitn` requests do not rescue row-permuted M64
  rows.

### Round 21 local scaled-MMAv5 layout baseline

- Time: 2026-04-21 12:03 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_scaled_layouts_round21.md`
- Selector:
  `mma_scaled and (lhs_subslice or lhs_tile_permuted or acc_tile_permuted_64 or acc_tile_permuted_32 or acc_identity_narrow) and not reports and not fz0015`
  collected `91/1615`.
- Split-4 result:
  `91 passed` (`23/23/23/22`).
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*` bucket.

### Round 23 Lane BF compiler-only lit audit

- Time: 2026-04-21 12:13 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_compiler_lit_round23.md`
- Commands:
  - `make -j8`
  - build-dir `ninja triton-opt`
  - selected `lit -v` tests for TMEM layouts, MMAv5 lowering, proxy fence,
    interleave/hoist/NVWS TMEM, memdesc subview splitting, and relayout
  - saved `triton-opt --run-reproducer` replays for proxy fence, copy/mbarrier,
    `ld.red` optimizer, and `ld.red` allocator repros
- New candidate `FZ-20260421-0016`: compiler-only verifier/encoding-info crash
  in `test/Conversion/relayout_tritongpu.mlir` for `ttng.tmem_alloc` with an
  unencoded tensor source and `TensorMemoryScalesEncodingAttr` result.
- Existing buckets re-confirmed: `FZ-20260421-0014`, `FZ-20260421-0008`, and
  `FZ-20260421-0009`.
- Green compiler-only baselines: all selected lit tests except
  `Conversion/relayout_tritongpu.mlir`.

### Round 23 local multi-CTA TMEM guardrail

- Time: 2026-04-21 12:42 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_multicta_round23.md`
- Selector:
  `(twocta or multicast or cta or cga) and tmem and not reports` collected
  `430/1615`.
- Split-4 result:
  `393 passed, 37 skipped`.
- Classification: no runtime miscompile, compiler crash, unexpected
  unsupported diagnostic, or new independent `FZ-*` bucket.

### Round 24 Lane BH mixed scaled-MMAv5 operands

- Time: 2026-04-21 13:55 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_mixed_operands_round24.md`
- Temporary probes:
  `/root/tmp/tmem_scaled_mixed_operands_round24_probe.py` and
  `/root/tmp/tmem_scaled_mixed_singleton_acc_round24_probe.py`.
- Result:
  direct selected-B plus runtime-selected accumulator-parent rows passed with
  clean side-channel probes; every reshape/permute/reshape scale descriptor-
  view row miscompiled with clean selected-scale loads.
- Classification: no new independent `FZ-*`. Mixed scale-view rows sharpen
  `FZ-20260421-0013`; dynamic accumulator-view composition did not reveal a
  new class beyond `FZ-0007`; exact previous `FZ-0015` side-channel rows still
  fail and remain distinct from the nearby direct mixed negative controls.

### Round 24 Lane BG FZ-0016 verifier minimization

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_fz0016_min_round24.md`
- Minimal representative:
  `/tmp/tmem_fz0016_round24/minimal_unencoded_scales_i8.mlir`
- Classification: existing candidate `FZ-20260421-0016`, sharpened. This is a
  parse/verifier-time `ttng.tmem_alloc` operand-layout crash. The minimal
  repro aborts with `dyn_cast on a non-existent value` even under parse-only
  `triton-opt -split-input-file --mlir-disable-threading`; relayout passes are
  not required.
- Neighboring fuzzing:
  unencoded `i8` operands abort for standard, linear, and scales TMEM results;
  unencoded `f16`/`f32` operands abort for standard and linear TMEM results;
  invalid `f16`/`f32` scales rows keep the clean `bitwidth must be 8`
  diagnostic; encoded operand rows produce ordinary compatibility diagnostics.
- No backend/compiler repair was attempted. The eventual repair target is the
  TMEM operand verifier / encoding-info path before calling
  `toLinearEncoding` on unencoded tensor types.

### Round 24 Lane BH mixed scaled-MMAv5 operands

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_scaled_mixed_operands_round24.md`
- Classification: no new independent `FZ-*` bucket.
- Direct selected B-scale descriptors plus runtime-selected accumulator-parent
  views passed in mixed kernels with correct side-channel loads.
- Mixed kernels that feed scaled-MMAv5 through scale descriptor views still
  miscompile with correct side-channel loads and remain in
  `FZ-20260421-0013`.
- The exact previous `FZ-20260421-0015` side-channel reproducer still fails,
  so `FZ-0015` remains live but structurally narrower than broad direct
  selected-B-scale use.
- Keep `FZ-20260421-0007`, `FZ-20260421-0013`, and
  `FZ-20260421-0015` separate until backend repairs prove a shared root cause.

### Round 24 local structural crash repro guardrail

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_structural_crash_round24.md`
- Selector:
  `allocator_crash or optimizer_crash`
- Result:
  `30 deselected, 3 xfailed`
- Classification: no unexpected pass/failure and no changed crash mode. The
  checked-in child-process crash sentinels continue to cover existing
  `FZ-20260421-0005`, `FZ-20260421-0008`, and `FZ-20260421-0009`.

### Round 24 local selector and structural guardrails

- Reports:
  - `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_selector_copy_round24.md`
  - `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_structural_round24.md`
  - `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_structural_crash_round24.md`
- Selector-permuted no-scales copy rows collected `4/1615` and passed as
  `4 passed`.
- Checked-in structural fuzzer non-crash rows collected `30/33` and completed
  split-4 as `9 passed, 21 xfailed`.
- Structural crash repro wrappers completed as `3 xfailed`.
- Classification: no unexpected pass/fail, changed crash mode, or new
  independent `FZ-*` bucket.

### Round 25 Lane BL high-CGA mixed ownership

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_high_cga_mixed_round25.md`
- Temporary probe:
  `/tmp/tmem_high_cga_mixed_round25_probe.py`; logs at
  `/tmp/tmem_high_cga_mixed_round25_probe.log` and
  `/tmp/tmem_high_cga_mixed_round25_probe_verbose.log`.
- Result:
  same-kernel TMA multicast plus local copy rows and same-kernel 2CTA MMAv5
  multicast/commit plus local copy rows all failed at the local TMEM allocation
  with the existing CTA-count diagnostic. Covered local 1CTA/2CTA copies in
  4/8/16 CTA contexts for TMA and in 4/8 CTA contexts for MMAv5.
- Classification: no new independent `FZ-*`. This sharpens
  `FZ-20260421-0010` as a local-layout/context ownership gate even when a
  legal high-CGA operation is present earlier in the same kernel. The attempted
  scaled-MMAv5 plus local indexed-view row hit a temporary probe setup
  limitation before reaching the local ownership check and is not counted as
  new backend evidence.

### Round 21 local scaled-MMAv5 layout rerun

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_scaled_layouts_round21.md`
- Follow-up rerun used read-only `least_duration` splitting after concurrent
  `--store-durations` writes changed shard boundaries during an ad hoc rerun.
- Stable aggregate result:
  `91 passed` (`11`, `58`, `11`, `11` across groups 1-4).
- Classification: still green; no new bucket outside excluded `FZ-0015` rows.

### Round 25 local dynamic descriptor guardrail

- Time: 2026-04-21 14:38 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_dynamic_structural_round25.md`
- Selector:
  `generic_pass or dynamic_index or loop_carried or scaled_mma_acc_subslice`
  collected `12/33`.
- Split-4 result:
  `12 xfailed`.
- Classification: no unexpected pass, changed crash mode, runtime miscompile
  outside the existing xfail set, or new independent `FZ-*` bucket.

### Round 25 local scaled descriptor-view baseline

- Time: 2026-04-21 14:35 UTC
- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_scaled_descriptor_view_round25.md`
- Selector:
  `mma_scaled and descriptor_view and not reports`
- Result:
  `4 passed, 1611 deselected`.
- Classification: no new independent `FZ-*`. This remains a green checked-in
  nearby control for report-only `FZ-20260421-0013` scale descriptor-view
  miscompile probes.

### Round 26 high-CGA scaled-MMAv5 mixed ownership

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_high_cga_scaled_round26.md`
- Temporary probe:
  `/tmp/tmem_high_cga_scaled_round26_probe.py`; final log at
  `/tmp/tmem_high_cga_scaled_round26_probe_final.log`.
- Controls:
  scaled-MMAv5 high-CGA kernels passed for `num_ctas=4`, `8`, and `16`, emitted
  the expected scaled opcode, and matched torch matmul reference.
- Mixed rows:
  local 1CTA/2CTA descriptor-view `st`, `ld`, `ld.red`, and `tcgen05.copy`
  inside those 4/8/16 CTA kernels all failed at the existing CTA-count
  ownership diagnostic.
- Classification:
  no new independent `FZ-*`; all mixed rows are existing
  `FZ-20260421-0010`.

### Round 28 broad `ld.red` runtime slice

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_ldred_round28.md`
- Selector:
  `ld_red and not reports and not resource`
- Split-4 result:
  `237 passed, 6 failed`.
- Classification:
  no new independent `FZ-*`; the six failures are existing
  `FZ-20260421-0012` M64 row/column-permuted destination-layout planner gaps.

### Round 26 dynamic descriptor SSA/control-flow

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_dynamic_ssa_round26.md`
- Temporary probe:
  `/tmp/tmem_dynamic_ssa_round26_probe.py`
- Result:
  seven rows collected. Copy/runtime-index rows reproduced existing
  `FZ-20260421-0001`; nested-helper, loop-carried, branch-yielded `ld.red`,
  and mixed `ld/st`+`ld.red` rows reproduced existing `FZ-20260421-0002`;
  plain-MMAv5 branch-selected accumulator view passed.
- Follow-up:
  direct descriptor controls passed for branch-selected direct `ld/st`,
  same-object direct branch `ld/st`, and branch-selected direct `ld.red`,
  narrowing `FZ-0002` toward descriptor-view composition crossing SSA/control
  flow rather than plain branch-carried memdescs.
- Classification:
  no new independent `FZ-*`.

### Round 29 broad scaled-MMAv5 runtime guardrail

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_mma_scaled_round29.md`
- Selector:
  `mma_scaled and not reports and not resource`
- Split-4 result:
  `243 passed`.
- Classification:
  no new independent `FZ-*`; committed scaled-MMAv5 positive surface remains
  green outside report-only and resource-boundary rows.

### Round 29 no-scale copy runtime guardrail

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_copy_round29.md`
- Selector:
  `cp_no_scales and not reports and not high and not cga and not selector and
  not subword`
- Split-4 result:
  `197 passed, 4 skipped`
- Classification:
  no new independent `FZ-*`; committed no-scale copy positive surface remains
  green outside report-only and high-CGA boundary rows.

### Round 29 dynamic descriptor arm symmetry

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_dynamic_arm_symmetry_round29.md`
- Temporary probe:
  `/tmp/tmem_dynamic_arm_symmetry_round29_probe.py`
- Exact result:
  `17 failed, 8 passed` across `25` new selector/control rows.
- Classification:
  no new independent `FZ-*`. `FZ-0002` is selector-symmetric and requires
  descriptor-view composition across SSA/control flow; direct descriptor branch
  controls pass. Copy dynamic/branch rows stay in `FZ-0001`.
- Follow-up:
  chain-shape probes found direct `ld/st` chain0 wrong-code at `M=128` without
  branch, loop, or parent indexing. Alternate identity chains pass and bad/
  green controls emit the same public `tcgen05.st/ld.32x32b.x32.b32` packet
  shape, pointing at accepted-but-misplanned descriptor-view layout mapping
  adjacent to `FZ-0003`.

### Round 29 structural fuzzer smoke

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_structural_round29.md`
- Result:
  `9 passed, 24 xfailed`
- Classification:
  no new independent `FZ-*`; checked-in structural-fuzzer expectations remain
  stable.

### Round 29 clean-boundary verifier fuzzing

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_clean_boundary_round29.md`
- Scope:
  runtime clean-boundary matrix, `invalid.mlir` lit verifier baseline, and
  temporary MLIR probes for edge shapes, memory-space mismatches, shape
  mismatches, CTA-count requirements, malformed linear-layout attributes, and
  unsupported bitwidth rows.
- Result:
  new candidate `FZ-20260421-0017`: encoded `i64` and `f64` TMEM load/store
  operands reach `-triton-tensor-memory-allocation` and abort in
  `lowerTMemLdSt` on `bitwidth == 32`. Runtime clean-boundary selector passed
  as `184 passed`; `test/TritonNvidiaGPU/invalid.mlir` passed; neighboring
  boundary probes produced typed diagnostics. Python frontend probes for
  `torch.float64`/`torch.int64` TMEM round trips reach the same JIT-time
  assertion; initialized alloc, standalone store, and standalone load crash,
  while dead uninitialized alloc is eliminated.

### Round 29 memdesc-index expansion

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_memdesc_index_round29.md`
- Scope:
  `40` temporary runtime cases across direct and descriptor-chain forms,
  constant index, same-object branch, distinct branch, runtime index, and
  load/store/copy/`ld.red`/mixed consumers.
- Result:
  no new independent `FZ-*`. Runtime index leaves illegal `ttg.memdesc_index`
  for every direct consumer tested, broadening `FZ-0001`. Direct
  distinct-branch `tcgen05.copy` also fails late with illegal
  `ttg.memdesc_index`, while direct branch load/store/`ld.red` controls pass.
  Descriptor-chain read/reduction miscompiles remain existing descriptor-view
  semantic buckets; descriptor-chain copy rows are clean unsupported.

### Round 29 Python runtime bitwidth follow-up

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_bitwidth_runtime_round29.md`
- Result:
  no new independent `FZ-*` beyond `FZ-0017`; subprocess-isolated Gluon
  `float64` and `int64` roundtrip, store-only, and load-only TMEM probes all
  abort with `ASSERT_BITWIDTH_32`.

### Round 30 local descriptor ld/st guardrail

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_ldst_descriptor_round30.md`
- Selector:
  `ldst_descriptor and not reports and not roundtrip`
- Split-4 result:
  `53 passed`
- Classification:
  no new independent `FZ-*`; committed descriptor `ld/st` positive surface
  remains green outside report-only and roundtrip resource-boundary rows.

### Round 30 local clean-boundary runtime guardrail

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_clean_boundary_round30.md`
- Selector:
  `reports_clean or clean_unsupported or clean_error or tmem_oor`
- Split-4 result:
  `184 passed`
- Classification:
  no new independent `FZ-*`; committed clean-boundary runtime surface remains
  green.

### Round 30 local TMEM lit baseline

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_local_lit_round30.md`
- Result:
  `test/TritonNvidiaGPU/tmem_layouts.mlir` and
  `test/TritonNvidiaGPU/invalid.mlir` passed after `ninja triton-opt`.
- Classification:
  no new independent `FZ-*`; compiler-only TMEM lit baselines remain green.

### Round 30 FZ-0017 64-bit expansion

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_fz0017_round30.md`
- Result:
  no new independent `FZ-*`; expanded `FZ-0017` to legacy/linear layouts,
  1CTA/2CTA, initialized alloc, store-only, load-only, roundtrip,
  descriptor-view load/store at LLVM conversion, and Python/Gluon frontend
  kernels for `i64`/`f64`.
- Clean contrasts:
  64-bit `ld.red` rejects with the f32-only diagnostic, 64-bit `ttng.tmem_copy`
  passes generated allocation+LLVM probes, and high-CGA rows reject under
  existing CTA-count diagnostics.

### Round 30 direct branch copy minimization

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_copy_branch_round30.md`
- Scope:
  same-parent/same-index/distinct-object branch-selected descriptors feeding
  `ttng.tmem_copy`, selector `0/1`, readback on/off, and 1CTA/2CTA.
- Result:
  no new independent `FZ-*`; same-parent branch-selected distinct indices are
  the failing `FZ-0001` shape. Same-index branch, distinct unindexed objects,
  and same-index different-parent controls pass at `N=32`.

### Round 31 Python descriptor-view FZ-0017 follow-up

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_python_descriptor_views_round31.md`
- Scope:
  valid Python/Gluon descriptor-view kernels using a larger TMEM parent and an
  N-sliced view, with input/output tensors already shaped to the view. Covered
  `float64`, `int64`, `float32`, and `int32`; legacy and linear parents;
  roundtrip, store-only, and load-only modes; 1CTA offsets `0/64/128`; and
  2CTA in-bounds/out-of-bounds offsets.
- Result:
  no new independent `FZ-*`; expanded `FZ-0017`. The lane ran `108`
  subprocess cases and produced `48` `ASSERT_BITWIDTH_32`, `40` passes, and
  `20` Python compile diagnostics. Valid 1CTA `float64`/`int64` descriptor
  views assert for all layout/mode/offset combinations, while matching 1CTA
  `float32`/`int32` rows pass and roundtrip correctly. Valid 2CTA 64-bit
  in-bounds offsets also assert. Out-of-bounds 2CTA rows reject cleanly, and
  remaining 2CTA 32-bit diagnostics are the existing global-store frontend or
  harness limitation.

### Round 32 structural fuzzer rerun

- Report:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_structural_rerun_round32.md`
- Command:
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py`
- Result:
  `9 passed, 24 xfailed in 9.42s`
- Classification:
  no XPASS, unexpected failure, compiler crash, false unsupported diagnostic,
  opcode mismatch, runtime miscompile, or new independent `FZ-*`; the
  checked-in structural sentinel catalog remains stable before the next
  discovery lane.
