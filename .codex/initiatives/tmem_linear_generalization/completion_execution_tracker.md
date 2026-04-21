# TMEM Completion Execution Tracker

Last updated: 2026-04-21

This is the active execution tracker for finishing the TMEM linear-layout
generalization project. It turns `backend_completion_plan.md` into a concrete
progress board so future sessions can resume without relying on chat context.

## Completion Definition

The project is complete when:
- TMEM copy, direct load/store, reduction load, plain MMAv5, and scaled-MMAv5
  all consume shared backend physical-query/planner objects instead of
  frontend or lowering-local layout policy;
- every bounded matrix case that the Blackwell TMEM ISA can realize is
  positive with runtime or opcode coverage;
- every remaining unsupported row is a true ISA/resource/API boundary with a
  typed backend diagnostic and probe evidence;
- compatibility scaffolding, stale frontend guards, and family-specific rescue
  paths are removed or explicitly quarantined;
- the TMEM-focused runtime matrix remains practical for iteration, with
  representative cold compile near the 3-4 second target and split-4 runtime
  validation kept duration-aware.

## Execution Contract

- Keep this file as the live progress board for the completion plan. Update it
  before and after meaningful implementation slices so the plan remains usable
  after an outage, context rollover, or branch handoff.
- 2026-04-21 structural fuzzing campaign override: the active workstream is
  continuous TMEM backend fuzzing for crashes, false unsupported diagnostics,
  verifier over-strictness, and runtime miscompiles. Do not end the turn while
  unblocked fuzzing work remains unless the user interrupts or a concrete
  external blocker is reached. Keep launching subagent rounds and local
  deterministic runtime fuzz sweeps. Catalog and minimize bugs, but do not
  begin backend fixes until fuzzing stops finding new failures or the user
  explicitly pivots to repair.
- Execute the highest-priority unblocked support-bearing slice first. Use
  cleanup-only slices when they remove family-specific policy or sharpen a true
  ISA boundary for the next support slice.
- Do not treat a green focused selector as completion. A slice is complete only
  after the relevant initiative docs record the new state, validation evidence,
  residual boundary, commit SHA, and pushed remote state.
- Continue execution against this board until all phases below are complete,
  unless the user interrupts or a concrete external blocker prevents further
  progress.

## Active Phase Board

- Phase Z, 24-hour structural fuzzing campaign: active as of 2026-04-21
  08:18 UTC. Build a systematic deterministic Python/Gluon runtime fuzzer plus
  parallel subagent audit lanes. Primary outputs are:
  - `python/test/gluon/test_tmem_structural_fuzzer.py` for checked-in
    deterministic runtime fuzz cases and future regression promotion;
  - `.codex/initiatives/tmem_linear_generalization/tmem_structural_fuzzing_20260421.md`
    for the active case schema, round log, failure catalog, and repro queue;
  - subagent round reports split by TMEM family (`ld/st/ld.red`, `copy`,
    `MMAv5/scaled-MMAv5`, warp-specialization/membar, and generic pass
    interactions);
  - periodic checkpoint commits pushed to `origin/codex/tmem`.
  Backend fixes are intentionally deferred during this campaign while new
  failures are still appearing.
  2026-04-21 08:30 UTC Lane E2 promoted FZ-20260421-0001 and
  FZ-20260421-0002 into checked-in strict xfail runtime coverage in
  `python/test/gluon/test_tmem_structural_fuzzer.py`, with exact fresh-process
  validation recorded in `agents/fuzz_generic_pass_round2.md`. Backend repair
  remains deferred.
  2026-04-21 09:10 UTC Lane A2 promoted FZ-20260421-0003 through
  FZ-20260421-0006 into checked-in strict xfail coverage in the same
  structural fuzzer: ld/st descriptor-chain miscompile, ld.red plain-load
  opcode fallback, 256-row lifted-parent allocator assertion via subprocess,
  and transpose/slice ld.red false-unsupported candidate. Exact nodeids
  reported `4 xfailed`; the full structural fuzzer reported
  `9 passed, 9 xfailed`. Backend repair remains deferred.
  2026-04-21 09:20 UTC promoted the Round 5 generic-pass loop-carried
  memdesc-view crash into checked-in strict xfail coverage:
  `generic-pass-loop-carried-memdesc-view-chain0`. This covers the
  `GluonResolveAutoEncodingsPass` / auto-layout inference failure while
  keeping discovery mode repair-deferred. Validation: required `make -j8`,
  py-compile, `30` collected structural-fuzzer nodeids, exact nodeid
  `1 xfailed`, and full fuzzer `9 passed, 21 xfailed`.
  2026-04-21 Round 6 Lane D prototyped a deterministic structural descriptor
  generator under `/tmp` and kept it out of the repo pending schema cleanup.
  The inventory generated `25245` normalized descriptors across `ldst`,
  `ldred`, `copy`, `mma`, and `mma_scaled`, with `0` exact normalized-id
  matches against the current `30` checked-in structural-fuzzer cases. Next
  generator slice should add a repo-local data-only `StructuralCaseDescriptor`
  plus `legacy_case_id` bridge before promoting runnable family adapters.
  Validation: required `make -j8` no-op, collect-only `30` nodeids, and small
  runtime sample `3 passed, 1 xfailed`.
  2026-04-21 Round 6 Lane B found `FZ-20260421-0007`, a report-only
  scaled-MMAv5 `use_acc` miscompile for a low-column accumulator subslice
  selected through dynamic `if`. Focused existing scaled-MMAv5 and plain-MMAv5
  matrix selectors remained green, so the next promotion slice should add a
  self-contained strict xfail to `test_tmem_structural_fuzzer.py` for the
  dynamic-if low-subslice boundary before repair work begins.
  2026-04-21 Round 6 Lane C isolated `FZ-20260421-0008`, a report-only
  two-CTA indexed `ld.red` chain1 row/col optimizer crash in
  `TritonNvidiaGPUOptimizeTMemLayoutsPass`. The minimized crash row is parent
  `[2,256,2]`, selected view `[256,2]`, row `even_odd`, col `identity`, chain
  `index(1).reshape((128,2,2)).permute([1,0,2]).reshape((256,2))`. It
  reproduces in a subprocess Python child and as an extracted MLIR
  `triton-opt --run-reproducer` abort; plain `ld` fallback and clean
  unsupported controls are cataloged separately. No backend/compiler repair
  was attempted.
  2026-04-21 09:08 UTC promoted `FZ-20260421-0007` and
  `FZ-20260421-0008` into checked-in structural-fuzzer strict xfails. The
  scaled-MMAv5 dynamic-if low-subslice row is an in-process expected
  miscompile sentinel; the `ld.red` row/col optimizer abort is isolated in a
  subprocess so the parent pytest process survives. Validation: required
  `make -j8`, py-compile, `32` collected structural-fuzzer nodeids, exact
  sentinels `2 xfailed`, and full fuzzer `9 passed, 23 xfailed`.
  2026-04-21 09:11 UTC Round 7 Lane D completed generic-pass / analysis
  interaction discovery without backend repairs. Tuple-like memdesc+tensor
  returns, nested helper-selected memdesc values, loop-carried memdesc+tensor
  iter args, chain2 loop-carried slice/slice views, and multi-live sibling
  views with non-TMEM tensor side inputs all classify under the existing R5-C
  `GluonResolveAutoEncodingsPass` auto-layout crash. A separate
  `16x128b` layout-conversion/non-TMEM tensor row extends
  `FZ-20260421-0002` with a fresh `8063 / 8192` runtime mismatch. No new
  independent FZ id was assigned. Report:
  `agents/fuzz_generic_analysis_round7.md`.
  2026-04-21 09:12 UTC Round 7 Lane B completed generator-backed `ld/st` and
  `ld.red` discovery without backend repairs. The Round 6 `/tmp` descriptor
  generator selected eight compact rows. New `FZ-20260421-0009`: 1CTA direct
  indexed `ld.red` over parent `[2,256,32]` aborts in
  `TritonTensorMemoryAllocationPass` with the allocator assertion
  `kNumRows - numRows >= 0`; extracted MLIR replay exits `134`. Positive and
  boundary rows confirmed `128x64` indexed `.ld.red.`, existing
  `FZ-20260421-0004` 2CTA `256x64` opcode fallback, already-covered
  `FZ-20260421-0003` f16 subword read-only miscompile, and clean unsupported
  descriptor-view diagnostics. Report:
  `agents/fuzz_generator_ldst_ldred_round7.md`.
  2026-04-21 09:13 UTC Round 7 Lane C completed copy/readback discovery
  without backend repairs. `/tmp/tmem_copy_readback_round7_probe.py` covered
  no-scales `warpx2` descriptor-view copy/readback positives,
  `warpx2::02_13` clean unsupported diagnostics, packed/subword clean
  diagnostics, descriptor-chain `ld.red` readback, scales `warpx4`, and the
  two-CTA-layout/four-CTA-context clean error. No stable new
  backend/compiler failure was found and no `FZ-*` id was assigned. Report:
  `agents/fuzz_copy_readback_round7.md`. Validation: required `make -j8`
  no-op, probe py-compile, `/tmp` launcher six cases passed, and four-GPU
  selector sweep passed `28` selected tests per group.
  2026-04-21 Round 7 Lane A completed scaled-MMAv5 accumulator control-flow
  expansion without backend repairs. The temporary harness
  `/tmp/tmem_scaled_mma_controlflow_round7_probe.py` collected `410` nodeids
  and classified a broad `FZ-20260421-0007` expansion: direct low/high and
  high-selector dynamic rows pass, while low-selector dynamic `if`, helper,
  loop, and indexed rows miscompile across the probed `N`, `K`, and feasible
  scaled-format cells. `indexed_helper` exceptions overlap
  `FZ-20260421-0001`; no new independent `FZ-*` id was assigned. Report:
  `agents/fuzz_scaled_mma_controlflow_round7.md`.
  2026-04-21 promoted Round 7 Lane B `FZ-20260421-0009` into checked-in
  subprocess-isolated strict xfail coverage:
  `test_tmem_structural_fuzzer_ldred_1cta_direct_index_allocator_crash`.
  This covers the 1CTA direct indexed `ld.red` parent `[2,256,32]`
  `TritonTensorMemoryAllocationPass` allocator assertion while keeping the
  parent pytest process alive. Validation: required `make -j8`, py-compile,
  `33` collected structural-fuzzer nodeids, exact sentinel `1 xfailed`, and
  full fuzzer `9 passed, 24 xfailed`. Backend repair remains deferred.
  2026-04-21 Round 8 Lane C completed copy/readback generator-adapter
  discovery without backend repairs. Positive no-scales/scales copy/readback
  and descriptor-chain `ld.red` readback rows stayed positive; two-CTA
  `warpx2::02_13`, packed/subword, larger-CGA two-CTA-layout, and scales
  descriptor-view copy rows stayed clean diagnostics. No new `FZ-*` id.
  Validation: required `make -j8`, py-compile, `18` collected nodeids, and
  full probe `18 passed`. Report:
  `agents/fuzz_copy_readback_round8.md`.
  2026-04-21 Round 8 Lane D completed generic-pass / analysis interaction
  discovery without backend repairs. No new independent `FZ-*` id. Direct
  chain0 descriptor-view mismatches dominate several generic-looking helper,
  tuple, sibling, static-loop, and same-base dynamic-if rows, so they extend
  `FZ-20260421-0003`; runtime direct `memdesc_index` still extends
  `FZ-20260421-0001`. Chain1/chain2 generic controls reached runtime and
  passed. Validation: required `make -j8`, py-compile, `14` collected
  nodeids, split-4 classified sweep `7 failed, 7 passed`, and fresh exact
  confirmations. Report: `agents/fuzz_generic_pass_round8.md`.
  2026-04-21 Round 8 Lane A completed `ld.red` allocator/opcode discovery
  without backend repairs. No new independent `FZ-*` id. The lane broadened
  `FZ-20260421-0004` with 18 two-CTA indexed opcode fallback rows,
  `FZ-20260421-0008` with row-chain optimizer aborts across
  `N={2,16,32,64}`, and `FZ-20260421-0005/0009` with direct indexed
  256-row allocator failures for both `ld.red` and `ld/st` dtype variants.
  Positive 128-row indexed/chained controls stayed green. Validation:
  required `make -j8`, py-compile, 55 subprocess-isolated rows, exact fresh
  confirmations, and `git diff --check`. Report:
  `agents/fuzz_ldred_allocator_opcode_round8.md`.
  2026-04-21 Round 9 Lane E completed normalized generator inventory and
  follow-up `ld/st` inventory-hole probing without backend repairs. The
  `/tmp` generator emitted `44,043` descriptors and linked all `33`
  checked-in structural-fuzzer legacy ids; `18 / 46` generated axis buckets
  have legacy coverage. The `ld/st` hole probe found no new owner: chain1
  rows extend `FZ-20260421-0003`, one small transpose/slice row is a clean
  row-anchor diagnostic, and positive controls pass. Reports:
  `agents/fuzz_generator_inventory_round9.md` and
  `agents/fuzz_ldst_inventory_holes_round9.md`.
  2026-04-21 Round 9 Lane F completed clean-diagnostic adversarial fuzzing
  without backend repairs. No new independent `FZ-*` id. Generated rows
  classified as eight clean diagnostics, one positive control, one
  `FZ-20260421-0001` late illegal-op row, and four allocator assertions
  extending `FZ-20260421-0005/0009`; exact copy-boundary runtime-matrix rows
  passed `4/4`. Report: `agents/fuzz_clean_diagnostics_round9.md`.
  2026-04-21 Round 8 Lane B completed scaled-MMAv5 accumulator
  descriptor-view control-flow fuzzing without backend repairs. No new
  independent bucket. `FZ-20260421-0007` expands to `nvfp4xnvfp4`, `N=16`,
  `K=256`, selector-1 dynamic slice/indexed accumulator views, including
  `use_acc=False`; direct controls pass and MMA opcodes agree. Helper-indexed
  rows overlap `FZ-20260421-0001`. Report:
  `agents/fuzz_scaled_mma_round8.md`.
  2026-04-21 Round 9 Lane H completed validation logistics without backend
  repairs. No new `FZ-*` id. Use the structural fuzzer as the frequent smoke
  gate (`33` nodeids, current expected `9 passed, 24 xfailed`). Prefer
  focused runtime slices for repeated lanes: scaled MMA `use_acc` (`28`),
  copy `warpx2` (`28`), LD.RED descriptor (`30`), and clean diagnostics
  (`96`), with stored durations and least-duration split-4 reruns. Report:
  `agents/fuzz_validation_logistics_round9.md`.
  2026-04-21 09:44 UTC local Round 10 structural-fuzzer smoke gate remains
  green while Lane I and Lane J fuzzing are active. Required `make -j8` no-op;
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py`
  reported `9 passed, 24 xfailed in 24.16s`.
  2026-04-21 local Round 10 clean-diagnostic runtime slice stayed green:
  `reports_clean_unsupported` ran split-4 across GPUs 0-3 with stable caches,
  each shard selected `23` tests, and the aggregate result was `92 passed`.
  Durations were stored at
  `/tmp/tmem_local_r10_clean_diagnostics_durations.json`.
  2026-04-21 local Round 10 copy `warpx2` runtime slice stayed green:
  `cp_no_scales and warpx2` ran split-4 across GPUs 0-3 with stable caches,
  selected `20/20/20/19` tests, and passed all `79`. Durations were stored at
  `/tmp/tmem_local_r10_copy_warpx2_durations.json`.
  2026-04-21 local Round 10 scaled-MMA `use_acc` runtime slice stayed green:
  collect-only selected `83/1615`, split-4 across GPUs 0-3 selected
  `21/21/21/20` tests, and all `83` passed. Durations were stored at
  `/tmp/tmem_local_r10_scaled_use_acc_durations.json`.
  2026-04-21 local Round 10 higher-rank/rank-5 descriptor runtime slice stayed
  green: collect-only selected `82/1615`, split-4 across GPUs 0-3 selected
  `21/21/21/19` tests, and the aggregate result was
  `62 passed, 20 skipped`. Durations were stored at
  `/tmp/tmem_local_r10_rank5_higher_rank_durations.json`.
  2026-04-21 local Round 10 `ld/st` row/column descriptor runtime slice stayed
  green: collect-only selected `32/1615`, split-4 across GPUs 0-3 selected
  `8/8/8/8` tests, and the aggregate result was `8 passed, 24 skipped`.
  Durations were stored at
  `/tmp/tmem_local_r10_ldst_rowcol_descriptor_durations.json`.
  2026-04-21 Round 10 Lane J completed descriptor-view composition-depth
  fuzzing without backend repairs. No new independent `FZ-*` id. The temporary
  probe found `7` passes and `3` clean diagnostics: rank-4/rank-5 `ld/st`
  unsupported direct-layout materialization and rank-5 sibling-view `ld.red`
  tensor-memory OOR (`2048` required, hardware limit `512`). Report:
  `agents/fuzz_descriptor_depth_round10.md`.
  2026-04-21 Round 10 Lane I completed `ld.red` modifier/NaN edge fuzzing
  without backend repairs. No new independent `FZ-*` id. The `266`-row
  subprocess sweep reclassified to `140` pass, `70` opcode fallback rows
  broadening `FZ-20260421-0004`, `28` optimizer abort rows broadening
  `FZ-20260421-0008`, and `28` clean OutOfResources diagnostics. No runtime
  miscompile or allocator assertion. Report:
  `agents/fuzz_ldred_modifiers_round10.md`.
  2026-04-21 local Round 10 broad non-rowcol `ld/st` descriptor slice stayed
  green: collect-only selected `162/1615`, split-4 across GPUs 0-3 selected
  `41/41/41/39` tests, and the aggregate result was `108 passed, 54 skipped`.
  Durations were stored at
  `/tmp/tmem_local_r10_ldst_descriptor_broad_durations.json`.
  2026-04-21 Round 10 Lane K completed allocator/resource-boundary fuzzing
  without backend repairs. No new independent `FZ-*` id. Fresh probes mapped
  to existing allocator/compiler-crash buckets `FZ-20260421-0005/0009`, f16
  descriptor-chain miscompile `FZ-20260421-0003`, 2CTA indexed `ld.red`
  fallback `FZ-20260421-0004`, and clean copy/scaled-MMA boundaries. Report:
  `agents/fuzz_allocator_resource_round10.md`.
  2026-04-21 local Round 10 `cp_scales` runtime slice stayed green:
  collect-only selected `34/1615`, split-4 across GPUs 0-3 selected
  `9/9/9/7` tests, and all `34` passed. Durations were stored at
  `/tmp/tmem_local_r10_cp_scales_durations.json`.
  2026-04-21 09:58 UTC post-integration structural-fuzzer smoke gate remains
  green after Lane I/J/K integration: required `make -j8` no-op, and
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py`
  reported `9 passed, 24 xfailed in 8.16s`.
  2026-04-21 local Round 10 plain-MMAv5 descriptor slice stayed green:
  collect-only selected `109/1615`, split-4 across GPUs 0-3 selected
  `28/28/28/25` tests, and all `109` passed. Durations were stored at
  `/tmp/tmem_local_r10_plain_mma_desc_durations.json`.
  2026-04-21 local Round 10 lit sanity stayed green: `ninja triton-opt` no-op
  and `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
  passed `3/3`.
  2026-04-21 Round 10 Lane L completed generic pass/control-flow descriptor
  fuzzing without backend repairs. No new independent `FZ-*` id. The exact
  four-GPU checked-in-case rerun produced `7` passes, `1` clean unsupported
  diagnostic, and `20` failures owned by existing buckets
  `FZ-20260421-0001`, `FZ-20260421-0002`, `FZ-20260421-0003`,
  `FZ-20260421-0004`, `FZ-20260421-0006`, `FZ-20260421-0007`, and `R5-C`.
  Report: `agents/fuzz_generic_controlflow_round10.md`.
  2026-04-21 Round 10 Lane M completed copy/scales/CGA fuzzing without
  backend repairs. No new `FZ-*` id. Positive scales `warpx4`, no-scales
  `warpx2`, and scaled-MMA copy setup rows stayed positive; descriptor-view,
  two-CTA `warpx2::02_13`, subword, and larger-CGA rows stayed clean
  boundaries. The `num_ctas > 2` scaled-MMA shape assertion is an early
  guardrail. Report: `agents/fuzz_copy_scales_cga_round10.md`.
  2026-04-21 local Round 10 clean-unsupported selector stayed green with
  duration-aware splitting: collect-only selected `92/1615`, split-4
  least-duration across GPUs 0-3 selected `19/25/24/24` tests, and all `92`
  passed using `/tmp/tmem_local_r10_clean_diagnostics_durations.json`.
  2026-04-21 local Round 10 checked-in `ld.red` NaN/modifier selector stayed
  green: collect-only selected `176/1615`, split-4 across GPUs 0-3 selected
  `44/44/44/44` tests, and all `176` passed. Durations were stored at
  `/tmp/tmem_local_r10_ldred_nan_durations.json`.
  2026-04-21 Round 10 Lane O completed opcode/IR consistency fuzzing without
  backend repairs. No new independent `FZ-*` id and no PTX-vs-LLIR opcode
  disagreement were found across sampled positive `ld`, `st`, `ld.red`, `cp`,
  plain MMA, and scaled-MMA rows. Matching expected opcodes included direct
  and descriptor-chain `ld/st`, direct `ld.red` min/max.abs.NaN, no-scales
  and scaled-copy `cp`, plain MMAv5, and 1CTA/2CTA scaled-MMAv5. The only bug
  overlap was the known indexed 256x32 `ld.red` plain-load fallback under
  `FZ-20260421-0004`; the exotic copy layout remained a clean unsupported
  boundary. Report: `agents/fuzz_opcode_consistency_round10.md`.
  2026-04-21 local Round 11 no-scales copy non-`warpx2` runtime slice stayed
  green. Required `make -j8` no-op. Collect-only selected `169/1615` for
  `cp_no_scales and not warpx2 and not reports`; split-4 across GPUs 0-3 with
  stable caches selected `43/43/43/40` tests and reported aggregate
  `165 passed, 4 skipped`. Durations were stored at
  `/tmp/tmem_local_r11_cp_no_scales_nonwarpx2_durations.json`.
  2026-04-21 Round 10 Lane N completed MMAv5/scaled-MMAv5 dynamic descriptor
  selection fuzzing without backend repairs. New report-only candidate
  `FZ-20260421-0011`: plain MMAv5 runtime-selector-index accumulator rows
  compile, keep PTX/LLIR MMA opcode agreement, and miscompile at runtime with
  NaN-heavy mismatches (`~8150/8192` for `N=64,K=128`). The temporary harness
  did not show the `FZ-20260421-0001` illegal `ttg.memdesc_index` signature,
  but a first checked-in minimization attempt did hit `FZ-0001` and was not
  committed. Scaled rows expand `FZ-20260421-0007`; exact controls stayed
  green. Report: `agents/fuzz_mma_dynamic_round10.md`.
  2026-04-21 Round 10 Lane P completed clean-boundary adversarial fuzzing
  without backend repairs. No new independent `FZ-*` id. Most perturbations
  stayed pass or clean-diagnostic across copy OOR, explicit subword, `warpx2`,
  scales descriptor-view, scaled-MMA accumulator-layout, CTA/CGA mismatch, and
  ld.red transpose/slice boundaries. `ld_red_identity_n512` overlaps the
  existing `FZ-20260421-0005/0009` ld.red resource/crash family. Report:
  `agents/fuzz_clean_boundary_round10.md`.
  2026-04-21 Round 11 Lane Q completed multi-CTA/CGA structural fuzzing
  without backend repairs. New independent candidate `FZ-20260421-0010`:
  1CTA/2CTA TMEM linear/scales layouts are rejected in 4/8/16 CTA launch
  contexts with `Layout has X CTAs per CGA, but the context requires Y CTAs
  per CGA`, even though the rows are local 1CTA/2CTA TMEM operations and
  checked-in high-CGA MMA/TMA-MMA controls passed as `136 passed, 12 skipped,
  74 deselected`. This points at an over-strict layout CTA-count gate that
  conflates kernel CGA shape with instruction-local `cta_group`. Report:
  `agents/fuzz_multicta_cga_round11.md`.
  2026-04-21 local post-report structural-fuzzer smoke gate stayed green:
  required `make -j8` no-op, and
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py`
  reported `9 passed, 24 xfailed in 8.70s`.
  2026-04-21 local Round 12 non-f32 `ld.red` descriptor-chain slice stayed
  green. Collect-only selected `20/1615` for
  `ld_red and descriptor and not propagate_nan and not reports`; split-4
  across GPUs 0-3 selected `5/5/5/5` tests and reported aggregate
  `20 passed`. Durations were stored at
  `/tmp/tmem_local_r12_ldred_nonf32_descriptor_durations.json`.
  2026-04-21 local Round 12 scaled-MMA accumulator-subslice selector stayed
  green. Collect-only selected `74/1615` for
  `mma_scaled and acc_subslice_view and not reports`; split-4 across GPUs 0-3
  selected `19/19/19/17` tests and reported aggregate `74 passed`. Durations
  were stored at `/tmp/tmem_local_r12_scaled_acc_subslice_durations.json`.
  2026-04-21 local Round 12 lit sanity stayed green: `ninja triton-opt` no-op
  and `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir test/TritonNvidiaGPU/invalid.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
  passed `3/3`.
  2026-04-21 Round 12 Lane T completed cache/process stability fuzzing without
  backend repairs. No new cache/process/replay instability candidate. Stable
  same-process, immediate repeat, fresh subprocess, and fresh diagnostic-cache
  contrasts all matched `7 passed, 5 xfailed`; bad rows stayed in known
  buckets and green controls stayed green. Report:
  `agents/fuzz_cache_process_round12.md`.
  2026-04-21 Round 12 Lane R completed high-CGA gate minimization without
  backend repairs. `FZ-20260421-0010` is reproduced across `18` minimized
  rows: 1CTA/2CTA linear `ld/st`, direct `ld.red`, no-scales copy, and
  scales-copy layout-construction diagnostics under 4/8/16 CTA launch
  contexts. Passing high-CGA MMA controls stayed green as `2 passed in
  3.18s`. Report: `agents/fuzz_high_cga_gate_round12.md`.
  2026-04-21 Round 12 Lane S completed plain-MMAv5 runtime-index reduction
  without backend repairs. `FZ-20260421-0011` is FPSAN-specific: the
  `N=32,K=128,selector=1,use_acc=False` row repeated `3/3` fresh subprocesses
  with stable `4081/4096` mismatches under FPSAN, while the non-FPSAN
  lifted-linear runtime-index form falls back to known `FZ-20260421-0001`
  illegal `ttg.memdesc_index` lowering. Report:
  `agents/fuzz_plain_mma_runtime_index_round12.md`.
  2026-04-21 local high-CGA MMA contrast controls stayed green after required
  `make -j8` no-op: `test_tcgen05_mma_multicast_commit[False-ctas_per_cga1]`
  and `[True-ctas_per_cga2]` reported `2 passed in 3.05s`.
  2026-04-21 local Round 12 `cp_scales` non-report selector stayed green.
  Collect-only selected `33/1615` for `cp_scales and not reports`; split-4
  across GPUs 0-3 selected `9/9/9/6` tests and reported aggregate
  `33 passed`. Durations were stored at
  `/tmp/tmem_local_r12_cp_scales_nonreports_durations.json`.
  2026-04-21 Round 12 Lane V completed copy descriptor/addressing fuzzing
  without backend repairs. No new independent `FZ-*`. Checked-in copy/scales
  runtime slice selected `128/1615` and passed `128/128`; fresh-process probe
  classified `6` pass, `3` `FZ-20260421-0010` high-CGA no-scales copy rows,
  and one harness capture limitation later confirmed clean. Report:
  `agents/fuzz_copy_descriptor_round12.md`.
  2026-04-21 local exact structural xfail sample stayed stable:
  `FZ-20260421-0004`, `FZ-20260421-0009`, and `FZ-20260421-0007` exact nodeids
  reported `3 xfailed in 4.59s`.
  2026-04-21 local clean-boundary selector replay stayed green:
  `reports_clean_unsupported` passed split-4 across GPUs 0-3 with stable
  caches as `19/25/24/24`, aggregate `92 passed`; durations reused from
  `/tmp/tmem_local_r10_clean_diagnostics_durations.json`.
  2026-04-21 Round 12 Lane U completed ld.red row/column/opcode fuzzing
  without backend repairs. No new independent `FZ-*` and no runtime
  miscompile. Probe total: `94` rows classified as `42` pass, `19`
  `FZ-20260421-0004`, `14` `FZ-20260421-0008`, `13`
  `FZ-20260421-0005/0009`, and `6` clean TMEM OOR boundaries. Report:
  `agents/fuzz_ldred_rowcol_round12.md`.
  2026-04-21 Round 12 Lane W completed generic descriptor-view pass fuzzing
  without backend repairs. No new independent `FZ-*`. Checked-in generic-pass
  rows stayed as `11 xfailed`; the temporary probe classified as `7` pass and
  `2` known failures. Runtime `memdesc_index` remains `FZ-20260421-0001`;
  chain0 generic-pass/control-flow wrong results remain `FZ-20260421-0002`.
  The loop-carried checked-in row currently presents as an `8064/8192`
  runtime mismatch, not the historical `R5-C` auto-layout crash. Report:
  `agents/fuzz_generic_views_round12.md`.
  2026-04-21 post Lane U/W structural fuzzer smoke gate stayed stable after
  required `make -j8` no-op:
  `python/test/gluon/test_tmem_structural_fuzzer.py` reported
  `9 passed, 24 xfailed in 8.69s`.
  2026-04-21 local allocation/lifetime selector collected `7/1615` and passed
  `7 passed, 1608 deselected` after required `make -j8` no-op.
  2026-04-21 local two-CTA/multicast-adjacent selector collected `370/1615`
  and passed split-4 across GPUs 0-3 with stable caches as `333 passed,
  37 skipped`; durations stored at
  `/tmp/tmem_local_r13_twocta_multicast_durations.json`.
  2026-04-21 Round 13 Lane X completed allocation/lifetime/commit/barrier
  fuzzing without backend repairs. No new independent `FZ-*` and no runtime
  miscompile. Probe total: `28` rows classified as `18` pass, `4`
  `FZ-20260421-0010`, `2` clean TMEM OOR, `2` clean copy packed-lane
  unsupported, and `2` harness/shared-layout setup limitations. Report:
  `agents/fuzz_alloc_lifetime_round13.md`.
  2026-04-21 local rank5/higher-rank descriptor selector collected `82/1615`
  and passed split-4 across GPUs 0-3 with stable caches as `62 passed,
  20 skipped`; durations stored at
  `/tmp/tmem_local_r13_rank5_higher_rank_durations.json`.
  2026-04-21 local lifetime/mbarrier sanity report found no new `FZ-*`: core
  commit/mbarrier rows passed `13/13`, runtime-matrix allocation lifetime rows
  passed `7/7`, and descriptor-chain/physical-bitcast rows passed `29/29`.
  Report: `agents/fuzz_lifetime_mbarrier_round13.md`.
  2026-04-21 local ConSan TMEM instrumentation report found no new `FZ-*`:
  `python/test/gluon/test_consan.py -k 'tmem or tcgen05 or tensor_memory'`
  collected `36/348` and passed split-4 as `30 passed, 6 skipped`. Report:
  `agents/fuzz_consan_tmem_round13.md`.
  2026-04-21 checked-in FPSAN MMAv5/scaled-MMAv5 controls collected `37/104`
  and passed split-4 as `32 passed, 5 skipped`; durations stored at
  `/tmp/tmem_local_r13_fpsan_controls_durations.json`.
  2026-04-21 Round 13 Lane Y completed FPSAN MMAv5 runtime descriptor
  selection fuzzing without backend repairs. No new independent `FZ-*`.
  FPSAN runtime `parent.index(ttgl.load(selector_ptr))` feeding plain MMAv5
  remains report-only `FZ-20260421-0011`; adjacent checked-in plain/scaled
  FPSAN controls, constexpr parent-index, and dynamic-slice controls stayed
  green. Report: `agents/fuzz_fpsan_mma_round13.md`.
  2026-04-21 local scaled-MMAv5 FP4/tile/narrow controls found no new
  `FZ-*`: selector
  `mma_scaled and (tile_permuted or narrow or e2m1 or fp4)` collected
  `202/1615` and passed split-4 as `202 passed`. Report:
  `agents/fuzz_scaled_fp4_tile_narrow_round13.md`.
  2026-04-21 local warp-specialized TMEM partitioning controls found no new
  `FZ-*`: Python selector `warp_specialize and tmem` collected `2/18218` and
  passed `2/2`; lit partition/allocation/interleave/NVWS files passed `4/4`.
  Report: `agents/fuzz_warpspec_partition_round13.md`.
  2026-04-21 local copy/subword selector collected `106/1615` and passed
  split-4 across GPUs 0-3 with stable caches as `106 passed`; durations stored
  at `/tmp/tmem_local_r13_copy_subword_durations.json`.
  2026-04-21 Round 13 Lane Z completed non-overlapping copy/subword edge
  fuzzing without backend repairs. No new independent `FZ-*`; checked-in copy
  selector collected `69/1615` and passed split-4 as `69 passed`,
  representative PTX/LLIR opcodes matched for `128x128b`, `4x256b`, and
  exact-width subword copy, and 4/8/16 CTA high-CGA contrasts mapped to
  existing `FZ-20260421-0010`. Report:
  `agents/fuzz_copy_subword_round13.md`.
  2026-04-21 local `ld.red` M64 row-permuted fuzzing found new candidate
  `FZ-20260421-0012`: checked-in selector
  `ld_red and (tile_permuted or permuted or descriptor_chain or n_sweep) and not reports`
  collected `115/1615` and split-4 ran as `109 passed, 6 failed`. The six
  failures reproduce as over-strict unsupported destination-layout lowering for
  M64 row-permuted reductions while nearby column-permuted controls pass.
  Report: `agents/fuzz_ldred_m64_permuted_round13.md`.
  2026-04-21 local `ld/st` descriptor-selection runtime sweep found no new
  `FZ-*`: selector
  `ldst and (descriptor_roundtrip or descriptor_compositions or descriptor_chain or runtime_selector or rowcol_permuted or tile_selector) and not reports`
  collected `142/1615` and split-4 ran as `64 passed, 78 skipped`.
  Descriptor roundtrips/compositions, row/column-permuted descriptor parents,
  two-CTA descriptor parents, and subword descriptor chains did not reproduce
  the FPSAN MMAv5 runtime-index mismatch or the M64 `ld.red` unsupported-dst
  diagnostic. Report: `agents/fuzz_ldst_descriptor_round13.md`.
  Follow-up selector `ldst and (descriptor_chain or x1 or subword or replay)`
  collected `261/1615` and split-4 ran as `191 passed, 70 skipped`.
  2026-04-21 local MMAv5 descriptor/view runtime sweep found no new `FZ-*`:
  selector
  `mma_twocta_indexed_acc_view or mma_twocta_tma_tf32_b_transposed_descriptor or mma_scaled_shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc`
  collected `37/1615` and split-4 passed as `37 passed`. Non-FPSAN indexed
  accumulator views, lifted `linear_unit_parent` parents, `use_acc`
  true/false rows, TMA transposed-B descriptor rows, and one scaled-MMAv5
  indexed narrow control did not reproduce `FZ-20260421-0011`. Report:
  `agents/fuzz_mma_descriptor_view_round13.md`.
  2026-04-21 local ld.red non-f32 descriptor sanity selected `20/1615` and
  passed split-4 as `20 passed`; durations stored at
  `/tmp/tmem_local_r14_ldred_nonf32_descriptor_durations.json`.
  2026-04-21 local high-CGA/scales sanity selected `17/1615` and passed
  split-4 as `17 passed`; durations stored at
  `/tmp/tmem_local_r14_high_cga_scales_durations.json`.
  2026-04-21 local scaled descriptor sanity selected `32/1615` and passed
  split-4 as `32 passed`; durations stored at
  `/tmp/tmem_local_r14_scaled_descriptor_durations.json`.
  2026-04-21 structural fuzzer smoke gate stayed stable after required
  `make -j8` no-op: `9 passed, 24 xfailed in 8.15s`.
  2026-04-21 Round 14 Lane AA broadened `FZ-20260421-0012` without backend
  repairs. Checked-in `ld_red_m64 and not reports` collected `39/1615` and
  split-4 ran as `33 passed, 6 failed`; temporary `133`-row grid classified
  as `38` pass, `86` `FZ-20260421-0012`, `5` `FZ-20260421-0010`, and `4`
  descriptor-permute harness-limited rows. The bucket now covers effective
  non-identity M64 row bases across `N`, column permutations, modifiers,
  explicit variants, and descriptor-preserving chains. Report:
  `agents/fuzz_ldred_fz0012_round14.md`.
  2026-04-21 Round 14 Lane AB found new candidate `FZ-20260421-0013`
  without backend repairs. Checked-in scaled selector outside the prior
  FP4/tile/narrow slice collected `69/1615` and passed `69/69`; temporary
  shared-scale/copy probe classified `8` pass and `2` harness setup
  limitations; temporary B-scale descriptor-view probe classified `1` pass,
  `3` runtime miscompile candidates, and `3` clean diagnostic boundaries.
  `FZ-0013` is a local 1CTA non-FPSAN scaled-MMAv5 B-scale descriptor-view
  wrong-result candidate with matching PTX/LLIR scaled-MMA opcodes and
  retained descriptor-view chains. Report:
  `agents/fuzz_scaled_descriptor_round14.md`.
  2026-04-21 10:53 UTC local Round 14 generic-pass structural selector
  stayed stable after required `make -j8` no-op. Collect-only selected
  `11/33` rows from
  `python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'`.
  Split-4 execution across GPUs 0-3 with stable caches reported aggregate
  `11 xfailed`. Runtime dynamic `ttg.memdesc_index` illegal lowering remains
  `FZ-20260421-0001`; chain0/control-flow/layout-pressure runtime
  mismatches remain `FZ-20260421-0002`. No new bucket and no backend repair.
  Lane AC remains active on broader generic runtime-index fuzzing and should
  write `agents/fuzz_generic_runtime_index_round14.md`.
  2026-04-21 11:02 UTC Round 14 Lane AC completed and wrote
  `agents/fuzz_generic_runtime_index_round14.md`. No new independent `FZ-*`;
  `FZ-20260421-0001` broadens to copy/MMAv5 consumers and dynamic
  branch-selected memdesc SSA values, while `FZ-20260421-0002` remains the
  chain0 generic wrong-result owner. Temporary probes classified `27`
  generic rows (`10` pass, `6` `FZ-0001`, `11` `FZ-0002`) and `12` copy/MMA
  rows (`2` pass, `8` `FZ-0001`, `2` clean diagnostics). Local adjacent
  two-CTA copy/scales selector collected `53/1615` and passed split-4 as
  `53 passed`.
  2026-04-21 local descriptor and `ld.red` selector sweep wrote
  `agents/fuzz_local_descriptor_ldred_round14.md`. The descriptor algebra
  selector collected `116/1615` and passed split-4 as `38 passed, 78 skipped`.
  The `ld_red and (m64 or descriptor or rowcol_permuted) and not reports`
  selector collected `100/1615` and ran as `94 passed, 6 failed`; the six
  failures remain known `FZ-20260421-0012`.
  2026-04-21 local clean-boundary selector wrote
  `agents/fuzz_local_clean_boundaries_round14.md`. The
  `reports_clean_unsupported or reports_clean_error or reports_tmem_oor`
  selector collected `157/1615` and passed split-4 as `157 passed`; no clean
  diagnostic row drifted into an unexpected crash or runtime path.
  2026-04-21 local scaled-MMAv5 descriptor/accumulator selector wrote
  `agents/fuzz_local_scaled_descriptor_acc_round14.md`. The
  `mma_scaled and (descriptor or scale_descriptor or bscale or acc_subslice_view or indexed_acc) and not reports`
  selector collected `105/1615` and passed split-4 as `105 passed`; no new
  bucket from checked-in scaled descriptor coverage.
- Phase A, rebaseline and classify: done for this branch. The current
  clean-negative/error surface is stable at `145/1615`; unsupported-only
  collect-only is `92/1615`. Every bucket below is classified as positive
  support, typed clean boundary, or frontend/API contract error.
- Phase B, complete shared physical-query model: done for known TMEM policy
  surfaces in this branch. Remaining frontend/lowering code is orchestration
  around backend helpers, not an unowned support predicate with known red
  coverage.
- Phase C, finish `tcgen05.copy` atomized planner: closed for the TMEM
  linear-layout generalization project at 2026-04-18 19:50 UTC. The copy
  planner uses one generic family/footprint path for scales and non-scales,
  with no `isScales` physical-query dimension or support-kind mode. Residual
  rows are classified as public atom mask/full-footprint boundaries,
  refresh-image API work, packed-lane storage/staged-copy work,
  cta-group::2 `warpx2::02_13` ISA/schedule behavior, or explicit frontend
  copy-source contracts. See `tcgen05_cp_gap1_closure_20260418.md`.
- Phase D, finish `ld/st` and `ld.red` packet/replay planning:
  boundary-complete for the current matrix. Supported descriptor-view/replay
  rows are positive; residual rows are true atom-footprint or refresh-row-anchor
  boundaries with structured diagnostics.
- Phase E, finish MMAv5 and scaled-MMAv5 descriptor/storage semantics:
  boundary-complete for the current public ISA and storage contracts. Gap #2
  narrow scaled-MMAv5 `N=8/16` is now positive through B-scale storage
  rematerialization/padding. Gap #3 mixed fp4A TMEM-LHS is closed as a typed
  direct-TMEM storage/API boundary because raw TMEM LHS cannot model the
  shared-memory `fp4_padded` operand-A contract. Other residual rows are
  external `.kind::i8` PTXAS/ISA rejections, instruction-tile order
  boundaries, or explicit frontend/API contracts. Gap #1 signed i8 direct
  MMAv5 is now validated on GB200 through the generated `sm_100a` PTX/cubin
  artifact under
  `experiments/mmav5_i8_remote/`, and has checked-in sm100-gated runtime pytest
  coverage for three signed-i8 shapes; remaining i8 work is IR/frontend
  exposure for unsigned/per-operand signedness and saturation, not the signed
  compiler path.
- Phase F, cleanup/redesign deletion: active cleanup/support checkpoint in
  progress. Recent checkpoints moved MMAv5 address selection, direct `ld/st`
  support/replay predicates, physical-bitcast/view predicates, query-type
  lowering precedence, and now shared Gluon/backend `ld/st` register-layout
  arithmetic into backend helpers, deleted stale copy address dead code, and
  removed obsolete hardcoded trace-file/debug-disable shims from the touched
  Gluon and LLVM TMEM paths. The 2026-04-19 branch cleanup audit also pruned
  superseded raw experiment result logs while preserving compact current
  summaries and reusable evidence artifacts.
  2026-04-20 follow-ups rebaselined the current clean-negative inventory,
  saved exact compact current logs under `experiments/results/`, and clarified
  `legacy` runtime-matrix vocabulary as `TensorMemoryLayout` frontend
  compatibility rather than backend policy. Debug-helper consolidation remains
  deferred until profiling shows the gated debug checks are material.
  2026-04-20 19:03 UTC added
  `python/tutorials/gluon/15-tmem-linear-layout-generalization.py` as a
  presentation/tutorial cleanup summarizing the completed branch features and
  true residual boundaries with runnable examples. This did not open a new
  support-bearing implementation slice.
  2026-04-20 20:10 UTC rewrote that tutorial to be standalone and
  hardware-focused, removed compiler-text assertions, added numeric runtime
  examples, and added a skinny MXFP8 block-scaled projection benchmark using
  narrow TMEM accumulator fragments.
  2026-04-21 06:23 UTC adversarial point-test audit added coverage for
  scaled-MMAv5 narrow view/use-acc paths, two-CTA direct higher-rank
  load/store/reduction boundaries, CTA-count layout contracts, and generic
  TMEM partition relayout behavior. It also fixed `InterleaveTMem` indexed
  view alias ranges and made `OptimizePartitionWarps` preserve TMEM partition
  warp counts until TMEM-aware relayout is implemented.
  2026-04-21 06:44 UTC warp-specialization partition audit fixed
  same-allocation TMEM partition discovery through memdesc view chains and SCF
  value forwarding, with lit coverage for two `ttng.tmem_subslice` views of
  one allocation in a warp-specialized loop.
  2026-04-21 06:47 UTC copy/ld/st/ld.red audit fixed multicast
  `tcgen05.copy` ownership classification for 128-row two-CTA scales
  destinations. The pair-local block selector is now derived from half the
  destination row extent and capped at `128`, so smaller pair-local scales
  tiles use row `64` while 4/8/16 CTA `warpx2` outer ownership remains
  supported.
  2026-04-21 07:00 UTC Round 4 generic backend audit found no new concrete
  fixable gap in `AxisInfo`, `Coalesce`, `RemoveLayoutConversions`,
  `OptimizeThreadLocality`, `LayoutPropagationUtility`, or linear-layout
  conversion interactions with TMEM memdescs. The audited paths already keep
  memdesc SSA values out of AxisInfo/layout-conversion tensor rewrites, or
  restrict themselves to ranked pointer/tensor payloads. Validation covered
  the generic pass lit files plus TMEM layout/verifier/conversion lit.
  2026-04-21 07:03 UTC Round 3 warp-specialization / partitioning /
  barrier audit found no new concrete scheduling or membar implementation bug.
  Added `membar-cluster.mlir` coverage for a valid 4-CTA CGA using two-CTA
  `tc_gen5_mma` with an async completion mbarrier, so the larger-CGA
  pair/mbarrier path is pinned alongside existing 2-CTA MMA and 4/16 CTA
  `ttng.tmem_copy` barrier cases. The memdesc edge-size heuristic remains
  noted as a scheduling-cost risk without a reproduced bad partition.
  2026-04-21 07:03 UTC Round 2 copy/ld/st/ld.red follow-up fixed an
  additional ld.red descriptor-view false negative: `ttng.tmem_subslice`
  views can now use the canonical standalone query type for reduction
  verification/layout inference instead of being rejected because the raw
  subview type still carries the backing column basis. Added conversion lit
  coverage for a 128x64 ld.red from a 128x256 backing TMEM subview; focused
  ld.red descriptor-chain plus recent copy runtime slice passed `31/31`
  across four pytest-split GPU groups.
  2026-04-21 07:10 UTC Round 5 adversarial runtime-matrix probing found no
  production backend bug, but corrected stale coverage: two-CTA higher-rank
  direct `ld.red` through a lifted descriptor-view chain is positive when the
  reduced output uses a matching two-CTA register layout. The runtime matrix
  now validates the positive result and
  `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32` instead of carrying a
  masked clean-negative row. The transposed shared-copy diagnostic assertion
  is aligned to the current clean `ttng.tmem_copy` verifier boundary.
- Phase G, saturation/performance/final validation: done for local branch
  validation. The corrected full runtime-matrix runner passed at 2026-04-17
  21:45 UTC with `1490 passed, 102 skipped` across all `1592` cases.
- Phase H, user-facing performance examples: consolidating into the three
  retained, non-overlapping example files requested by the user. The source of
  truth is `tmem_example_implementation_plan_20260420.md`. Active files:
  `05-tmem-moe-router.py`, `06-tmem-lora-fusion.py`, and
  `08-tmem-layout-as-epilogue.py`. The standalone candidate-head, MLP side
  projection, and ragged expert files were merged into `05`/`06`; the former
  attention score example remains removed because it did not beat the optimized
  plain Triton baseline.
- Historical Phase H checkpoint: complete for the six retained
  example files. The source of truth is
  `tmem_example_implementation_plan_20260420.md`. Implemented examples:
  `05-tmem-moe-router.py`, `06-tmem-lora-fusion.py`,
  `07-tmem-candidate-head.py`, `08-tmem-layout-as-epilogue.py`,
  `09-tmem-mlp-side-projection.py`, and `11-tmem-ragged-expert-views.py`.
  The former `10-tmem-windowed-attention-score.py` was removed after a fair
  Triton mask-plus-row-max baseline beat the TMEM `load_max` path on the
  measured standalone shapes, and row-blocked TMEM variants hit true
  descriptor-view/row-anchor support boundaries. The retained examples use
  plain Triton non-TMEM kernels for post-processing comparisons where needed:
  router top-2, LoRA update, layout reorder, and MLP side gate. Combined
  validation over the retained six files passed `37 passed in 23.52s`.

## Current Clean-Negative Inventory

For the stable discussion order of support/coverage questions, use
`remaining_coverage_gaps.md`. Gap #1 `tcgen05.cp`, Gap #2 narrow
scaled-MMAv5 `N=8/16`, and Gap #3 mixed fp4 TMEM LHS are now closed for this
project. Do not use old pre-reconsolidation copy gap numbers unless explicitly
discussing history.

Collected at 2026-04-17 18:59 UTC after `make -j8`:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q --collect-only python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'reports_clean_unsupported'
```

Result after the 19:25 two-CTA `warpx2::01_23` source-rematerialization
slice: `115/1592` tests collected (1477 deselected) in 3.14s.

Result after the 20:48 integer `ld.red` NaN no-op promotion:
`111/1592` tests collected (1481 deselected) in 3.13s.

Additional combined clean-negative/clean-error rebaseline:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q --collect-only python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'reports_clean_unsupported or reports_clean_error'
```

Result after the 19:25 two-CTA `warpx2::01_23` source-rematerialization
slice: `165/1592` tests collected (1427 deselected) in 3.20s.

Result after the 20:48 integer `ld.red` NaN no-op promotion:
`161/1592` tests collected (1431 deselected) in 3.13s.

Result after the 21:37 backend-policy cleanup checkpoints:
`161/1592` tests collected (1431 deselected) in 2.97s.

Result after the 2026-04-18 22:54 Gap #2/#3 checkpoint:
`141/1594` tests collected (1453 deselected) in 3.07s. The total matrix grew
by two rows from intervening coverage, and the former 20 narrow scaled-MMAv5
clean-negative rows are now positive runtime coverage.

Result after the 2026-04-20 cleanup follow-up rebaseline:
`91/1594` tests collected (1503 deselected) in 3.83s. Exact nodeids are stored
in `experiments/results/clean_unsupported_inventory_current.log`.

Combined clean-negative/clean-error result after the same follow-up:
`141/1594` tests collected (1453 deselected) in 3.83s. Exact nodeids are stored
in `experiments/results/clean_unsupported_or_error_inventory_current.log`.

Round 5 adversarial audit rebaseline at 2026-04-21 07:10 UTC:
`reports_clean_unsupported` collects `92/1615`; combined
`reports_clean_unsupported or reports_clean_error` collects `145/1615`. The
two-CTA higher-rank direct `ld.red` row is no longer in the clean-negative
inventory because the valid two-CTA reduced-output register layout reaches a
positive `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32` runtime path.

Current buckets:
- `ld/st` scales variant atom-footprint boundaries:
  too-narrow n-sharded scale atoms. The n-sharded rows now report a structured
  tensor-memory-scales packet-footprint requirement with required/exposed
  scale-element counts.
- `ld.red` non-f32 reductions: integer `propagate_nan=ALL` is now positive
  through the software reduction path because NaN propagation is a no-op for
  integer element types. Remaining non-f32 rows are software positives or true
  dtype/storage boundaries outside this bucket.
- `tcgen05.copy` scales descriptor-view rows: ordinary noncanonical 64x16
  shared-linear sources and 64x16 shared subslices now rematerialize into the
  canonical warpx4 shared source before copy. The remaining descriptor-view row
  is a destination-view boundary: source column bit 2 selects descriptor row
  `+32` inside a `warpx4` instruction, requiring a source-message/destination-
  column split, narrower atom, valid source format, or destination mask.
  Current probes show the public `warpx4.32x128b` atom writes the full
  16-column destination footprint, so the requested 4-of-8 column split is a
  true mask/source-format schedule boundary.
- `tcgen05.copy` no-scales ordinary contiguous `4x256b`: copy support is
  positive for refresh-shaped layouts only; ordinary view exposure needs a
  first-class refresh remap/readback contract or stays negative.
- Direct `ld/st` of `4x256b` refresh images: row anchors are not materializable
  as public load/store warp bases without a row-anchor rematerialization model;
  this boundary is now reported from structured refresh-image facts shared with
  the copy diagnostic.
- No-scales two-CTA `warpx2::02_13`: current public `cta_group::2`
  direct-seed schedules either duplicate low source columns or read zeros; a
  valid schedule must preserve the high source-column bit. This is now reported
  through a typed source-column preservation requirement derived from the
  source-row split requirement.
- `warpx2` dense/noncanonical shared-source layouts and subword copies:
  single-CTA dense/noncanonical 128x4 shared sources now rematerialize into
  the canonical `warpx2` shared source and are positive for both `01_23` and
  `02_13`; two-CTA dense/noncanonical 256x4 shared sources now rematerialize
  with the canonical shared block basis and are positive for `01_23`.
  Remaining two-CTA `02_13` dense/source-view rows are constrained by the high
  source-column preservation boundary. Subword rows still report a structured
  destination-column footprint requirement with required logical column bits
  and packed-lane facts; they need packed-lane storage and descriptor
  semantic-equivalence proofs.
- Copy row/column permutation and sub-instruction tile permutation rows:
  now have representative probe evidence that descriptor representability is
  not enough. The public copy atoms update full row/column footprints; the
  failing rows need row/column partitioning, smaller footprints, masks, or a
  different source format before they can be positive.
- Plain MMAv5 direct i8 clean-error rows: a guard-lift probe emitted
  `tcgen05.mma.cta_group::1.kind::i8`, but `ptxas-blackwell` rejects
  `.kind::i8` on `sm_103a`. Keep the frontend guard and classify this as an
  external PTXAS/ISA boundary, not a linear-layout backend gap.
- Frontend/API contract clean-error rows: block-layout TMEM descriptors and
  copy source-contract failures still require explicit API redesign before
  they can become backend support work.
- Plain MMAv5 exotic/row-column-permuted accumulators: now reported through a
  typed instruction-tile order requirement. Public atoms require canonical
  row/column basis order within each 64x8-or-larger instruction tile unless a
  tile-splitting or masked writeback schedule is designed.
- Scaled-MMAv5 mixed fp4A TMEM-LHS: closed as a typed direct-TMEM storage/API
  boundary. A speculative fp4-padded direct-TMEM path compiled but produced
  wrong output; the missing semantics are the shared-memory `fp4_padded`
  operand-A row-dependent 128-byte swizzle and padding aliases.
- Scaled-MMAv5 narrow accumulator `N=8/16`: closed as positive support.
  Narrow accumulator families now lower with B-scale storage
  rematerialization/padding, and the former clean-negative matrix is now
  runtime-positive. The 2026-04-21 follow-up hardened the single-fragment
  narrow-N predicate so direct support still requires tensor-memory-scales
  storage with enough addressable rows, added identity `N=16` runtime coverage
  across scaled format pairs, and added a clean-negative `N=16` tile-permuted
  row for the public MMAv5 in-tile basis-order boundary.

## Current Runtime Matrix Validation

Latest checkpoint at 2026-04-17 21:45 UTC after the Phase F backend-policy
cleanup commits:

- `python3 .codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py`
  passed all corrected buckets with complete `1592/1592` coverage.
- Bucket evidence:
  - `cp`: `312 passed, 4 skipped`;
  - `mma`: `601 passed`;
  - `splitn`: `35 passed`;
  - `ld_red`: `247 passed`;
  - `ldst`: `295 passed, 98 skipped`;
  - aggregate: `1490 passed, 102 skipped` across all `1592` cases.
- Logs:
  - `.codex/initiatives/tmem_linear_generalization/experiments/results/tmem_runtime_matrix_sweep_20260417_213804/`

Previous checkpoint at 2026-04-17 21:26 UTC after the Phase F leading-slice
replay policy cleanup and MMAv5 family address-layout cleanup:

- `python3 .codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py`
  passed all corrected buckets with complete `1592/1592` coverage.
- Bucket evidence:
  - `cp`: `312 passed, 4 skipped`;
  - `mma`: `601 passed`;
  - `splitn`: `35 passed`;
  - `ld_red`: `247 passed`;
  - `ldst`: `295 passed, 98 skipped`;
  - aggregate: `1490 passed, 102 skipped` across all `1592` cases.
- Logs:
  - `.codex/initiatives/tmem_linear_generalization/experiments/results/tmem_runtime_matrix_sweep_20260417_211829/`

Baseline checkpoint at 2026-04-17 20:41 UTC after the M64 physical-subview fix:

- Fixed `run_tmem_runtime_matrix_sweep.py` so every shard uses the current
  checkout's `PYTHONPATH` (`repo`, `repo/python`, and
  `repo/python/test/gluon`) instead of accidentally importing the installed
  `triton` wheel.
- Fixed the runner's `splitn` bucket to include
  `test_tmem_runtime_matrix_splitn_16bit_m64_auto_matches_explicit`; a
  collection audit now shows the bucket union covers all `1592/1592` nodeids
  with zero missing and zero extra cases.
- Removed stale `ld.red` split-offset expectations for reverse-column layouts:
  those rows produce correct runtime output and use the default opcode offset
  order, while `tile_permuted` remains the only currently verified N=256
  descriptor-chain row needing `(0, 128, 64, 192)`.
- Bucket evidence for the corrected full runtime matrix:
  - `cp`: `312 passed, 4 skipped`;
  - `mma`: `601 passed`;
  - `splitn`: `35 passed`;
  - `ld_red`: `247 passed`;
  - `ldst`: `295 passed, 98 skipped`;
  - aggregate: `1490 passed, 102 skipped` across all `1592` collected cases.
- Hygiene:
  - `python -m py_compile
    .codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py
    python/test/gluon/test_tmem_runtime_matrix.py`;
  - `git diff --check`.

## Latest Support Promotion

2026-04-17 20:48 UTC:

- Promoted direct and descriptor-chain i32 `ld.red` rows with
  `propagate_nan=ALL` from clean unsupported to positive software reductions.
- Semantics: NaN propagation is treated as `NONE` for non-floating element
  types before selecting the software reduction combiner; floating non-f32
  behavior is unchanged.
- Validation:
  - `make -j8`;
  - exact promoted rows passed `4/4`;
  - `-k 'ld_red_non_f32'` passed `44/44`;
  - full runner `ld_red` bucket passed `247/247`;
  - clean-negative inventory is now `111/1592`;
  - combined clean-negative/error inventory is now `161/1592`;
  - Python byte-compile for the changed Python files;
  - `git diff --check`.

## Latest Cleanup Checkpoint

2026-04-17 21:48 UTC:

- Moved direct `ld/st` query-type-lowering precedence into backend helper
  `shouldPreferTMemLdStQueryTypeLoweringBeforeRawQuery`.
- The helper owns the M64 direct-root exception and derives the warp count from
  the register layout before delegating to
  `shouldPreferTMemLdStQueryTypeLayoutsBeforeRawQuery`.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory changed.
- Validation:
  - `make -j8`;
  - focused replay/direct-support runtime selector split across four GPUs
    passed `6/6`, `6/6`, `6/6`, and `4/4`.

2026-04-17 21:37 UTC:

- Exposed backend `isExplicitTMemLdStViewProducer` and made the Gluon
  register-layout bridge use it for type-only fallback/view-like decisions.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory changed.
- Validation:
  - `make -j8`;
  - focused replay/direct-support runtime selector split across four GPUs
    passed `6/6`, `6/6`, `6/6`, and `4/4`;
  - `git diff --check`.

2026-04-17 21:34 UTC:

- Deleted unused `TensorMemoryToLLVM.cpp` copy address-layout/view-offset
  helpers that still carried family-specific layout sorting outside the shared
  copy planner.
- This is deletion-only cleanup; no support rows were promoted and no
  clean-negative inventory changed.
- Validation:
  - `make -j8`;
  - `git diff --check`.

2026-04-17 21:32 UTC:

- Moved MMAv5 TMEM address-layout and tile-order offset selection out of
  `DotOpMmaV5TmemLoader` and into backend `TensorMemoryUtils` helpers:
  `isTMemPhysicalBitcast`, `getMMAv5TMemAddressLayout`, and
  `getMMAv5TMemViewOffsetForLowering`.
- This keeps physical-bitcast coordinate-frame selection and typed/family/query
  fallback ordering in the backend helper layer instead of in LLVM lowering.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory was changed.
- Validation:
  - `make -j8`;
  - focused MMAv5 runtime selector split across four GPUs passed `26/26` on
    each group;
  - `test_tmem_physical_bitcast_mma_lhs` plus
    `test_tmem_physical_bitcast_preserves_subview_mapping` passed `3/3`.

2026-04-17 21:01 UTC:

- Removed a verifier-local duplicate of the row-zero lifted reinterpret
  query-type rescue predicate.
- `verifyTMEMOperand` now calls the backend-owned
  `disallowTMemLdStQueryTypeRescue` helper, so this policy lives in the shared
  `TensorMemoryUtils` layer.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory was changed.
- Validation:
  - `make -j8`;
  - block descriptor clean-error runtime rows passed `2/2`;
  - blocked-layout clean-error runtime rows passed `2/2`;
  - `git diff --check`.
- Validation note:
  - `lit` is not installed in this shell as either `lit` or `python3 -m lit`.

2026-04-17 21:05 UTC:

- Moved the direct `ld/st` support register-layout search used by
  `OptimizeTMemLayouts` replay/leading-slice rewrites into backend
  `TensorMemoryUtils` as `getTMemLdStDirectSupportTensorType`.
- This removes a transform-local copy of raw-query/support-query/query-type
  search policy and keeps replay lowering as a consumer of backend-owned
  support helpers.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory was changed.
- Validation:
  - `make -j8`;
  - focused replay/direct-support runtime selector split across four GPUs
    passed `6/6`, `6/6`, `6/6`, and `4/4`;
  - `git diff --check`.

2026-04-17 21:08 UTC:

- Exposed the half-slice replay predicate as backend utility
  `getTMemLdStReplayableHalfSliceDim`.
- `OptimizeTMemLayouts` now consumes that helper and computes only the
  rewrite-local selected-half bit.
- Audit result: remaining replay structs/step builders are transform-local
  rewrite construction, not duplicated backend support policy.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory was changed.
- Validation:
  - `make -j8`;
  - focused half-slice replay selector split across four GPUs passed `5/5`,
    `5/5`, `5/5`, and `3/3`;
  - `git diff --check`.

2026-04-17 21:12 UTC:

- Moved the leading-slice direct-vs-replay preservation policy out of
  `OptimizeTMemLayouts` and into backend utility
  `shouldPreserveDirectTMemLdStLeadingSliceView`.
- The helper owns the gapped-column-basis check that decides when direct
  physical support would alias logical halves and the transform should replay
  the descriptor view instead.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory was changed.
- Validation:
  - `make -j8`;
  - focused replay/direct-support runtime selector split across four GPUs
    passed `6/6`, `6/6`, `6/6`, and `4/4`.

2026-04-17 21:17 UTC:

- Moved MMAv5 family address-layout selection from the LLVM lowering loader
  into backend API `getMMAv5TMemFamilyAddressLayout`.
- The helper returns the normalized accumulator, scaled-accumulator, or TMEM
  LHS family layout for full-allocation MMAv5 descriptors, keeping address
  lowering aligned with verifier/planner layout info.
- This is cleanup only; no support rows were promoted and no clean-negative
  inventory was changed.
- Validation:
  - `make -j8`;
  - focused MMAv5 tile-permutation selector split across four GPUs passed
    `26/26` on each group.

## Final Execution State

1. No unblocked support-bearing runtime-matrix row remains in the current
   branch. The broad copy, load/store, reduction, MMAv5, and scaled-MMAv5
   matrix is green.
2. Remaining unsupported rows are classified boundaries, not unexplained
   generic linear-layout gaps.
3. Cleanup-only work should now require a newly identified duplicated policy
   surface or a regression. Do not churn bridge/lowering orchestration that
   already consumes backend helpers.
4. If a future ISA revision, PTXAS change, or frontend API adds masks,
   packed-lane storage, refresh remap/readback, or scaled-MMAv5 fragment
   support, reopen the corresponding boundary bucket as a new initiative slice.
5. Keep final validation evidence and this boundary inventory with the branch
   until PR/CI integration is complete.

## Remaining Work Plan

Status legend: `done`, `active`, `pending`, `blocked`, `boundary`.

### Phase A: Rebaseline And Classify

- `done`: collect the current clean-negative surface after a green build and
  record the bucket inventory.
- `done`: classify each bucket as stale guard, missing planner schedule,
  missing storage representation, missing ISA-family coverage, or true
  hardware/API boundary.
- `done`: keep compile/runtime-duration expectations attached to validation
  commands so broad matrix work remains practical on four GPUs.

### Phase B: Shared Physical Query And Requirement Model

- `done`: move identified frontend/lowering-only TMEM decisions into backend
  physical-query helpers or typed requirement structs.
- `done`: make diagnostics consume structured requirement data rather than
  ad hoc strings, especially for clean negatives that remain hardware
  boundaries.
- `done`: remove identified compatibility shims once copy, load/store,
  reduction, MMA, and scaled-MMA consume the shared query/support APIs
  directly.

### Phase C: `tcgen05.copy` Completion

- `boundary`: finish direct no-scales dense/multicast planner boundaries:
  row-order, mixed-basis, sub-instruction column masks, and destination/source
  ownership.
- `done`: decide and encode the correct `4x256b` contract: refresh-image
  positive support, ordinary-view remapping, direct readback limitations, and
  clean typed diagnostics for nonmaterializable refresh images.
- `boundary`: complete scales copy support or prove boundaries for
  descriptor-row split/mask schedules, source-message formats, and
  destination-column partitions.
- `boundary`: complete or prove the no-scales two-CTA `warpx2::02_13`
  schedule, including preservation of the high source-column bit.
- `done/boundary`: handle dense/noncanonical `warpx2` shared-source layouts and
  subword/packed-lane copies through source rematerialization or typed
  non-support proofs.
- `done`: delete copy-specific rescue stacks that duplicate the shared
  planner once their behavior is represented by query/support objects.

### Phase D: Direct `ld/st` And `ld.red`

- `done`: keep packet-footprint limitations represented as structured
  atom-footprint requirements rather than layout-name failures.
- `done/boundary`: continue scales direct `ld/st` descriptor-view support where the
  ISA can realize the requested atom. The two-CTA `16x32bx2` row is now
  positive; remaining clean negatives should be only true packet-footprint
  boundaries or future missing schedules proved by exact layout arithmetic.
- `boundary`: decide whether `4x256b` refresh images can be read back through a
  rematerialized public load/store view; otherwise keep a precise row-anchor
  diagnostic.
- `done`: finish `ld.red` non-f32/NaN semantics either with a correct
  software fallback or a typed semantic-boundary diagnostic.

### Phase E: Plain And Scaled MMAv5

- `done`: keep plain MMAv5 accumulator layout failures tied to a structured
  instruction-tile row/column ordering requirement while preserving existing
  whole-tile permutation positives.
- `done`: prove and structure the scaled-MMAv5 mixed fp4A TMEM-LHS storage
  representation boundary.
- `done`: support scaled-MMAv5 narrow-N through accumulator family planning and
  B-scale fragment storage rematerialization.
- `boundary`: expand opcode/runtime positives only when a new backend schedule
  is real, not when a frontend spelling happens to compile.

### Phase F: Cleanup And Redesign Deletion

- `done`: delete stale frontend guards and lowering-local type-only fallback
  policy as backend helpers subsume them.
- `done`: consolidate repeated layout arithmetic into shared helpers with
  exact `LinearLayout` compose/invert/pseudoinvert proofs.
- `done`: keep only durable abstractions; quarantine or remove temporary
  row/column rewrite smells after each family is represented in the planner.

### Phase G: Saturation, Performance, And Final Validation

- `done`: run staged validation: `make`, targeted lit/compiler checks,
  focused four-GPU runtime selectors, then duration-aware broad sweeps.
- `done`: profile any representative compile above the 3-4 second target or
  execution above the 1-2 second target before accepting the regression.
- `not required for this local branch checkpoint`: refresh GB200/NVIDIA
  manifests after broad validation and mark stale pre-fix counts explicitly.
  Do this only when consuming fresh external CI results.
- `done`: final state requires no unexplained clean negatives, no stale
  family-specific policy, practical split-4 runtime-matrix iteration, and an
  updated handoff summarizing every residual true ISA boundary.

## Next Concrete Slice

- 2026-04-18 22:54 UTC: closed Gap #2 narrow scaled-MMAv5 `N=8/16` as
  positive support and Gap #3 mixed fp4A TMEM-LHS as a direct-TMEM storage/API
  boundary. Validation: `make -j8`; narrow-N runtime matrix `20 passed`;
  mixed-fp4A TMEM-LHS clean-negative matrix `24 passed`; non-twoCTA
  scaled-accumulator tile-permuted selector `52 passed`.

No unblocked local implementation slice remains in the current TMEM
generalization plan. The next action is PR/CI integration and external
signal handling:

- if CI finds a branch-only correctness failure, classify it against this
  boundary inventory and fix the regression;
- if CI exposes a new support-bearing layout that is not covered here, reopen
  the relevant phase with a new targeted repro;
- if a future ISA/API exposes masks, packed-lane storage, refresh remapping,
  direct i8 MMAv5, or scaled-MMAv5 narrow-fragment support, create a new
  support slice from the corresponding boundary bucket.

- 2026-04-20 19:03 UTC: user-requested presentation cleanup added a Gluon
  tutorial for the branch feature suite. Validation was limited to py-compile
  because required `make -j8` failed immediately on an empty local CMake build
  directory with no `build.ninja`; direct import also failed on the unbuilt
  local `triton._C.libtriton.getenv` symbol. Tutorial checkpoint commit
  `4e96b6d40` was pushed to `origin/codex/tmem`.

- 2026-04-20 20:10 UTC: tutorial rewrite is runtime-validated. Required
  `make -j8` passed with explicit host libstdc++ include paths in
  `CPLUS_INCLUDE_PATH`; focused tutorial pytest passed `6/6`; script benchmark
  reported `N=32` narrow `0.011 ms` vs padded `0.013 ms` and `N=64` narrow
  `0.013 ms` vs padded `0.013 ms`. Tutorial rewrite checkpoint `021688bc2`
  was pushed to `origin/codex/tmem`.

- 2026-04-20 20:21 UTC: tutorial polish added inline benchmark output and a
  pre-branch capability note. Merge-base inspection found that the old
  scaled-MMA tutorial used `TensorMemoryLayout` accumulator allocations and did
  not expose explicit tile-permuted linear narrow fragments. Focused tutorial
  pytest passed `6/6`; script benchmark reported `N=32` `1.18x` and `N=64`
  `1.05x`.

## Progress

- 2026-04-17 21:50 UTC: updated the tracker to final local boundary status.
  The active plan has no unblocked local support-bearing slice remaining after
  the 21:45 full runtime-matrix sweep and the stable `161/1592`
  clean-negative/error collect-only inventory. Remaining buckets require
  external ISA/PTXAS support, frontend/API contract changes, destination
  masks, packed-lane storage, refresh remap/readback, or scaled-MMAv5 fragment
  representation work beyond the current public model.

- 2026-04-17 21:48 UTC: moved direct `ld/st` query-type-lowering precedence
  from `TensorMemoryToLLVM` into
  `shouldPreferTMemLdStQueryTypeLoweringBeforeRawQuery`. Validation:
  `make -j8`; focused split-4 replay/direct-support selector passed `6/6`,
  `6/6`, `6/6`, and `4/4`.

- 2026-04-17 21:45 UTC: reran the corrected full runtime-matrix runner after
  the latest backend-policy cleanup commits. All buckets passed with complete
  `1592/1592` coverage: `cp` `312 passed, 4 skipped`; `mma` `601 passed`;
  `splitn` `35 passed`; `ld_red` `247 passed`; `ldst`
  `295 passed, 98 skipped`; aggregate `1490 passed, 102 skipped`. Per-shard
  logs are in
  `.codex/initiatives/tmem_linear_generalization/experiments/results/tmem_runtime_matrix_sweep_20260417_213804/`.
  Combined clean-negative/error collect-only remains `161/1592`.

- 2026-04-17 21:37 UTC: exposed backend
  `isExplicitTMemLdStViewProducer` and replaced the Gluon bridge's local
  explicit-view op list with that helper. Validation: `make -j8`; focused
  split-4 replay/direct-support selector passed `6/6`, `6/6`, `6/6`, and
  `4/4`; `git diff --check`.

- 2026-04-17 21:34 UTC: deleted unused copy address-layout/view-offset helper
  code from `TensorMemoryToLLVM.cpp`. The removed code was not referenced by
  lowering and preserved stale family-specific layout sorting outside the
  backend copy planner. Validation: `make -j8`; `git diff --check`.

- 2026-04-17 21:32 UTC: moved MMAv5 TMEM address-layout and tile-order offset
  selection from LLVM lowering into backend `TensorMemoryUtils` helpers. This
  includes sharing `isTMemPhysicalBitcast` with the direct `ld/st` support
  planner. Validation: `make -j8`; focused MMAv5 split-4 runtime selector
  passed `26/26` on all groups; physical-bitcast core smoke passed `3/3`.

- 2026-04-17 21:26 UTC: reran the corrected full runtime-matrix runner after
  the leading-slice replay policy cleanup and MMAv5 family address-layout
  helper cleanup. All buckets passed with complete `1592/1592` coverage:
  `cp` `312 passed, 4 skipped`; `mma` `601 passed`; `splitn` `35 passed`;
  `ld_red` `247 passed`; `ldst` `295 passed, 98 skipped`; aggregate `1490
  passed, 102 skipped`. Per-shard logs are in
  `.codex/initiatives/tmem_linear_generalization/experiments/results/tmem_runtime_matrix_sweep_20260417_211829/`.

- 2026-04-17 21:12 UTC: moved the leading-slice direct-vs-replay preservation
  policy into backend utility `shouldPreserveDirectTMemLdStLeadingSliceView`.
  `OptimizeTMemLayouts` now consumes the helper instead of owning the
  gapped-column query check locally. Validation: `make -j8`; split-4 focused
  replay/direct-support runtime selector passed `6/6`, `6/6`, `6/6`, and
  `4/4`.

- 2026-04-17 21:17 UTC: moved MMAv5 family address-layout selection into
  backend API `getMMAv5TMemFamilyAddressLayout`. The LLVM MMAv5 loader now
  consumes the helper instead of carrying its own full-allocation
  accumulator/scaled/LHS family-layout lambda. Validation: `make -j8`;
  split-4 focused MMAv5 tile-permutation selector passed `26/26` on each
  group.

- 2026-04-17 20:57 UTC: ran the corrected full runtime-matrix runner. All
  buckets passed with complete `1592/1592` coverage: `cp` `312 passed, 4
  skipped`; `mma` `601 passed`; `splitn` `35 passed`; `ld_red` `247 passed`;
  `ldst` `295 passed, 98 skipped`; aggregate `1490 passed, 102 skipped`.
  Per-shard logs are in
  `.codex/initiatives/tmem_linear_generalization/experiments/results/tmem_runtime_matrix_sweep_20260417_205257/`.

- 2026-04-17 20:52 UTC: completed the copy-planner boundary audit for the
  remaining representative clean negatives. Validation command:
  `make -j8 && CUDA_VISIBLE_DEVICES=0
  TRITON_CACHE_DIR=/tmp/triton-cache-copy-boundary-audit
  PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short -k
  'cp_scales_tmem_descriptor_view_reports_clean_unsupported or
  cp_no_scales_4x256b_reports_clean_unsupported or
  cp_no_scales_warpx2_02_13_twocta or
  cp_no_scales_warpx2_row_permuted_destination_reports_clean_unsupported or
  cp_no_scales_twocta_noncanonical_block_reports_clean_unsupported or
  cp_no_scales_linear_exotic_reports_clean_unsupported or
  cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported
  or cp_no_scales_linear_rowcol_permuted_reports_clean_unsupported'
  python/test/gluon/test_tmem_runtime_matrix.py`, which passed `44/44`. No
  support rows were promoted because the diagnostics still point to true
  packed-lane, source-column, row/column-mask, refresh-remap, scales-mask, or
  CTA-ownership requirements.

- 2026-04-17 20:24 UTC: fixed the legacy M64 split-N physical-subview
  mismatch exposed by `test_core.py::test_block_m_64_mma[legacy]`. The failure
  was not a scales- or M64-only opcode issue: pure TMEM column subviews were
  computing relative bases by subtracting query origins as row/column
  coordinates. For source queries whose exact `LinearLayout` is surjective but
  folds a logical column basis into a physical row basis, that arithmetic
  lowers high-N descriptor views to a plain column offset. The base lowering
  now first pseudoinverts the source query's exact surjective layout for the
  requested subview offset, then falls back to the older origin-delta path for
  projected/non-surjective queries. This aligns MMAv5 `memdesc_subslice`
  addresses with direct root `ld/st` packet offsets for legacy M64 split-N
  high bands. Validation: `make -j8`; `test_core.py::test_block_m_64_mma`
  passed `2`; focused legacy M64 runtime rows passed `4`; ld.red
  descriptor-chain tile-permuted N=256 passed `1`; Python byte-compile for
  `test_tmem_runtime_matrix.py`; `git diff --check`.

- 2026-04-17 19:27 UTC: probed the remaining `warpx2` subword/packed-lane
  bucket after source rematerialization. Representative f16 single-CTA
  `01_23`, single-CTA `02_13`, and two-CTA `01_23` runs all fail through the
  structured destination-column footprint requirement: the N=4 destination
  exposes two column basis bits, while a public 128-bit f16 `warpx2` copy
  instruction needs three logical column bits and writes four physical dword
  columns with two packed lanes per word. This is not a stale shared-source
  layout failure. Support needs a first-class packed-lane source/destination
  storage model through descriptor synthesis, source-footprint planning, and
  instruction scheduling; otherwise the rows remain true storage/ISA
  boundaries.

- 2026-04-17 19:25 UTC: promoted two-CTA no-scales `warpx2::01_23`
  dense-source rematerialization. The same `tcgen05_copy` rematerialization
  helper now handles 256x4 TensorMemoryLinear `warpx2` destinations by
  rematerializing noncanonical shared sources into the canonical `warpx2`
  source layout with shared block basis `[[128, 0]]` and issuing a cluster
  async-shared fence. The former `01_23` two-CTA dense-source clean negatives
  now check runtime output and opcode coverage. Dense `02_13` two-CTA rows now
  rematerialize the source far enough to report the existing high
  source-column preservation boundary instead of the older source-layout
  mismatch. Validation: `make -j8`; focused two-CTA dense selector passed
  `4`; full `cp_no_scales_warpx2` passed `79`; split-4
  `cp_no_scales_warpx2` groups passed `20/20/20/19`; Python byte-compile;
  collect-only rebaseline is `115/1592` clean negatives and `165/1592` clean
  negatives/errors.

- 2026-04-17 19:21 UTC: promoted single-CTA no-scales `warpx2` dense-source
  rematerialization. `tcgen05_copy` now recognizes 128x4 TensorMemoryLinear
  destinations that map to public `warpx2::01_23` or `warpx2::02_13` copy
  families, reloads noncanonical `SharedLinearLayout` sources through the
  logical source coordinates, stores them into the canonical 128x4 `warpx2`
  shared-linear source layout, issues a fresh async-shared fence, and emits
  the existing backend copy op. The former single-CTA dense-source clean
  negatives now check runtime output and opcode coverage; two-CTA dense-source
  rows and the `02_13` source-column boundary remain negative. Validation:
  `make -j8`; exact promoted selector passed `4`; neighboring negative plus
  `cp_scales` selector passed `41`; full `cp_no_scales_warpx2` passed `79`;
  split-4 `cp_no_scales_warpx2` groups passed `20/20/20/19`; Python
  byte-compile; collect-only rebaseline is `117/1592` clean negatives and
  `167/1592` clean negatives/errors.

- 2026-04-17 19:14 UTC: promoted TensorMemoryScales 64x16 source
  rematerialization for `tcgen05.copy`. The Gluon `tcgen05_copy` builtin now
  rematerializes noncanonical 64x16 `SharedLinearLayout` scales sources into
  the canonical warpx4 shared-linear source before emitting the backend copy
  op. This makes the historical warpx2-shaped scales layout probes and
  shared-subslice source probes positive while preserving the scaled-MMA scale
  descriptor path and keeping the 128x32 descriptor-view destination boundary
  negative. Validation: `make -j8`; focused selector
  `cp_scales_layout_probe or cp_scales_noncanonical_layout_rematerializes or
  cp_scales_shared_subslice_layout_rematerializes or
  cp_scales_tmem_descriptor_view_reports_clean_unsupported` passed `10`;
  broader `-k 'cp_scales'` passed `34`; split-4 `cp_scales` groups passed
  `9/9/9/7`; Python byte-compile; `git diff --check`. Rebaseline:
  `reports_clean_unsupported` is `121/1592`; combined clean-negative/error is
  `171/1592`.

- 2026-04-17 19:07 UTC: classified two remaining buckets and refreshed the
  execution contract. Representative direct Python probes with
  `TRITON_DEBUG_TMEM_QUERY=1` show sub-instruction copy tile permutations and
  row/column permutations need partial row/column updates inside public
  full-footprint copy atoms; this is a mask/smaller-atom/source-format
  boundary, not a stale descriptor recognizer. A temporary direct-MMAv5 i8
  guard lift reached PTX emission, but `ptxas-blackwell` rejected
  `.kind::i8` for `sm_103a`; the guard was restored and the i8 bucket is now
  classified as an external PTXAS/ISA boundary. `AGENTS.md` now explicitly
  says the initiative docs, not chat context, are the active checklist for the
  no-stop execution rule. Validation/probes: `make -j8` before runtime probes;
  combined clean-negative/clean-error collect-only is `174/1592`.

- 2026-04-17 18:59 UTC: promoted the two-CTA scales descriptor-view
  `16x32bx2` direct `ld/st` row from clean-negative to positive support. The
  backend now synthesizes an exact lifted view layout for
  `tcgen05.ld/st.16x32bx2` by keeping the half-tile split on lane=16 and not
  duplicating the lifted row basis as register repetition. Generic lowering
  now recognizes this descriptor-view candidate even though the memdesc arrives
  as `TensorMemoryLinear`, preserving the requested split-N atom instead of
  silently selecting `32x32b`. Validation: `make -j8`; exact promoted row
  passed; focused CGA descriptor-view selector passed as `9 passed,
  1583 deselected`; Python byte-compile; `git diff --check`. Clean-negative
  rebaseline: `reports_clean_unsupported` is now `123/1592`, and
  `reports_clean_unsupported or reports_clean_error` is now `174/1592`.

- 2026-04-17 18:41 UTC: structured the explicit n-sharded scales `ld/st`
  packet-footprint boundary. `getUnsupportedDirectTMemLdStAtomFootprintReason`
  now also classifies tensor-memory-scales views for 16x64b/16x128b/16x256b
  requests whose exposed element count is smaller than the public atom
  footprint, and the clean-negative tests assert required versus exposed scale
  elements. Behavior is unchanged: these rows remain unsupported unless the
  user selects a narrower atom/auto layout or reshapes/copies to cover the
  requested packet footprint. Validation: `make -j8`; built `triton-opt
  test/TritonNvidiaGPU/invalid.mlir --split-input-file
  --verify-diagnostics`; focused scales unsupported selector passed
  `5 passed`; split-4 selected groups passed `2/2/1` with group 4 empty;
  Python byte-compile for `test_tmem_runtime_matrix.py`; `git diff --check`.

- 2026-04-17 18:36 UTC: structured the under-wide subword `warpx2`
  destination footprint boundary. The multicast destination-layout check now
  formats a `TMemCopyColumnFootprintRequirement` with element bitwidth,
  logical instruction columns, physical 32-bit dword columns, packed lanes per
  word, exposed column basis bits, and required column basis bits. Behavior is
  unchanged: `N=4` subword `warpx2` rows remain clean unsupported because the
  public 128-bit copy atom needs packed-lane storage semantics across
  descriptor synthesis, source footprint planning, and instruction scheduling.
  Validation: `make -j8`; built `triton-opt
  test/TritonNvidiaGPU/invalid.mlir --split-input-file
  --verify-diagnostics`; split-4 focused subword selector passed `4/4/4/2`;
  Python byte-compile for `test_tmem_runtime_matrix.py`; `git diff --check`.

- 2026-04-17 18:32 UTC: structured the `warpx2` dense/noncanonical
  shared-source boundary. `TMemCopyWarpx2SharedSourceRequirement` now carries
  source shape and optional offset-basis mismatch data, and the formatter
  reports the first shared offset-basis mismatch plus the source
  rematerialization boundary explicitly. Behavior is unchanged: the dense
  shared-source rows still reject cleanly because the public `warpx2` source
  message schedule assigns fixed meanings to shared offset bases, so descriptor
  representability alone is not enough. Validation: `make -j8`; built
  `triton-opt test/TritonNvidiaGPU/invalid.mlir --split-input-file
  --verify-diagnostics`; split-4 focused dense shared-source selector passed
  `2/2/2/2`; Python byte-compile for `test_tmem_runtime_matrix.py`;
  `git diff --check`.

- 2026-04-17 18:27 UTC: structured the scaled-MMAv5 narrow-N accumulator
  boundary. `MMAv5ScaledNarrowNScaleFragmentRequirement` now records logical
  shape, CTA shape, the plain MMAv5-compatible instruction shape, fragment
  count along N, the public scaled `N>=32` floor, and the B-scale
  padding/rematerialization factor implied by 64-column tensor-memory scale
  storage. Behavior is unchanged: `N=8/16` tile-permuted accumulator layouts
  remain clean unsupported until the backend can synthesize both a correct
  accumulator permutation and B-scale fragment rematerialization. Validation:
  `make -j8`; built `triton-opt test/TritonNvidiaGPU/invalid.mlir
  --split-input-file --verify-diagnostics`; split-4 focused narrow-N selector
  passed `5/5/5/5`; Python byte-compile for `test_tmem_runtime_matrix.py`;
  `git diff --check`.

- 2026-04-17 18:23 UTC: promoted the mixed fp4A TMEM-LHS guard into a
  structured `MMAv5ScaledMixedFp4ATMemRequirement`. The requirement records
  LHS storage shape, CTA shape, raw storage K columns, logical K, A/B bitwidths,
  and the `fp4_padded` shared-memory storage group facts. A temporary
  guard-lift probe for the `n=128, linear` row emitted
  `tcgen05.mma.kind::mxf8f6f4` but failed numerical comparison for essentially
  the whole tile (`16376/16384` mismatches), proving this is a real storage
  representation gap rather than a stale verifier guard. Validation: `make
  -j8`; built `triton-opt test/TritonNvidiaGPU/invalid.mlir
  --split-input-file --verify-diagnostics`; split-4 focused mixed-fp4A
  selector passed `6/6/6/6`; Python byte-compile for
  `test_tmem_runtime_matrix.py`; `git diff --check`.

- 2026-04-17 18:16 UTC: promoted the plain MMAv5 exotic and row/column
  permuted accumulator clean negatives into a structured
  `MMAv5TMemInstructionTileRequirement`. The requirement records logical
  shape, CTA shape, element bitwidth, minimum public instruction tile, and the
  first noncanonical in-tile basis when one is available. Behavior is
  unchanged: whole-tile accumulator permutations remain positive, while
  permutations inside the public `64x8`-or-larger tile remain a true
  tile-order / masked-writeback boundary. Validation: `make -j8`; built
  `triton-opt test/TritonNvidiaGPU/invalid.mlir --split-input-file
  --verify-diagnostics`; split-4 focused MMAv5 negative selector passed as
  `5/5/5/2`; Python byte-compile for `test_tmem_runtime_matrix.py`;
  `git diff --check`.

- 2026-04-17 18:10 UTC: promoted the no-scales two-CTA
  `warpx2::02_13` probe evidence into a typed
  `TMemCopyWarpx2TwoCTASourceColumnRequirement`. The diagnostic still reports
  the same ISA facts, but it now formats from the source-row split requirement:
  logical row bit 5 would need to select a one-dword source offset for
  32-of-64 destination rows, while `cta_group::2` direct-seed probes either
  duplicate the low source-column pair or read zeros when trying to complete
  the single-CTA schedule. Support is unchanged and remains a true source
  column / destination-row mask boundary. Validation: `make -j8`; split-4
  focused `warpx2::02_13`/neighbor selector passed as `8/8/8/6`; built
  `triton-opt test/TritonNvidiaGPU/invalid.mlir --split-input-file
  --verify-diagnostics`; Python byte-compile for `test_tmem_runtime_matrix.py`;
  `git diff --check`.

- 2026-04-17 18:04 UTC: classified the scales `tcgen05.copy`
  descriptor-view/shared-subslice split as a true full-footprint schedule
  boundary and structured the exact-view note. `TMemCopyExactViewScheduleNote`
  now derives from `TMemCopyExactViewScheduleRequirement`, recording the first
  differing physical-query field and active shape before reporting that a
  destination-row/source-message schedule is required. Debug probes showed the
  concrete gap: source column bit 2 selects descriptor row `+32` for
  4-column runs every 8 columns, but `warpx4.32x128b` writes the full
  16-column destination footprint. Validation: `make -j8`; split-4 focused
  scales-copy selector passed as `3/3/3/2`; built `triton-opt
  test/TritonNvidiaGPU/invalid.mlir --split-input-file --verify-diagnostics`;
  Python byte-compile for `test_tmem_runtime_matrix.py`; `git diff --check`.

- 2026-04-17 17:59 UTC: represented the direct `ld/st` `4x256b`
  refresh-image boundary with structured refresh-image data instead of a
  static string. `getTMemCopy4x256RefreshLdStUnsupportedMessage(...)` now
  formats the logical tile, low-column row anchors, high-column dword offset,
  and source-column split from `TMemCopy4x256RefreshImageRequirement`; the raw
  physical-bitcast packet-footprint diagnostic also carries the same facts.
  Behavior is unchanged: copy refresh layouts remain opcode-positive, ordinary
  contiguous `4x256b` views and direct refresh-image readback remain clean
  unsupported. Validation: `make -j8`; focused runtime selector
  `cp_no_scales_4x256b or ldst_4x256b_refresh` passed as `5 passed,
  1587 deselected`; split-4 selector groups with selected rows passed as
  `2/2/1` while group 4 was empty; `triton-opt
  test/TritonNvidiaGPU/invalid.mlir --split-input-file --verify-diagnostics`;
  Python byte-compile for `test_tmem_runtime_matrix.py`; `git diff --check`.

- 2026-04-17 17:55 UTC: strengthened the execution contract in this tracker
  and in `AGENTS.md`. The remaining work is now tracked as a phase-by-phase
  board from rebaseline through final validation, with explicit instructions
  to keep this file current and continue executing until the tracked plan is
  complete unless interrupted or blocked.

- 2026-04-17 17:53 UTC: made the scaled-MMAv5 narrow-N accumulator
  requirement carry `minimumAddressableBScaleFragmentN` explicitly instead of
  hardcoding the 64-column matrix-B scale-fragment alignment in the formatter.
  Support is unchanged: N=8 and N=16 accumulator tiles remain clean
  unsupported until a real scale-fragment rematerialization/storage model can
  preserve the public scaled-MMAv5 fragment semantics. Validation: `make -j8`;
  split-4 `mma_scaled_acc_tile_permuted_narrow_reports_clean_unsupported`
  selector passed `5/5/5/5`; `git diff --check`.

- 2026-04-17 17:50 UTC: added a typed
  `TMemCopyMixedBasisRequirement` for direct copy physical-query failures where
  a TMEM row or column basis mixes physical row and column contributions. The
  `mixed` no-scales copy negative now reports the exact offending basis bit and
  physical TMEM delta, rather than stopping at a generic mixed-basis string.
  Probe evidence from `TRITON_DEBUG_TMEM_QUERY=1` also classified the
  sub-instruction column-permutation and scales descriptor-row split negatives
  as full-footprint/mask gaps: the public atoms would overwrite complementary
  rows/columns without a narrower atom, source format, or destination mask.
  Validation: `make -j8`; split-4 selector for copy exotic, row/column
  permuted, and `warpx2` row-permuted clean negatives passed `5/5/5/4`;
  `git diff --check`.

- 2026-04-17 17:45 UTC: promoted copy destination row-order rejection into a
  shared typed `TMemCopyDestinationRowOrderRequirement` in
  `TensorMemoryUtils.cpp`. Dense and multicast row-permuted copy layouts now
  carry the offending row-basis sequence, instruction footprint, and derived
  destination-row mask/schedule gap instead of formatting the boundary from two
  separate ad hoc string builders. Support is intentionally unchanged: these
  rows still require a future row-selected source projection, row-partitioned
  atom, or destination-row mask before becoming positive. Validation:
  `make -j8`; split-4 focused selector for
  `cp_no_scales_warpx2_row_permuted_destination_reports_clean_unsupported` and
  `cp_no_scales_linear_rowcol_permuted_reports_clean_unsupported` passed
  `4/4/4/4`; `git diff --check`.

- 2026-04-17 17:40 UTC: converted the `warpx2` shared-source runtime preflight
  in `TensorMemoryUtils.cpp` from inline string checks into a typed internal
  `TMemCopyWarpx2SharedSourceRequirement`. Behavior is intentionally unchanged:
  noncanonical `warpx2` dense/shared-source rows still fail cleanly with the
  same user-facing diagnostics. This advances Phase B/F cleanup by moving one
  more copy-family boundary onto requirement data instead of ad hoc verifier
  text. Validation: `make -j8`; split-4 focused selector for
  `cp_no_scales_warpx2_dense_shared_reports_clean_unsupported`,
  `cp_no_scales_warpx2_twocta_dense_shared_reports_clean_unsupported`, and
  `cp_no_scales_warpx2_02_13_twocta_candidate_reports_clean_unsupported`
  passed as `3/3/3/1`; `git diff --check`.

- 2026-04-20 21:07 UTC: opened Phase H as a tracked follow-on examples
  workstream and added `tmem_example_implementation_plan_20260420.md`. No
  example implementation has started. Next concrete implementation slice is
  the projection-only MoE router example with a padded `N=128` baseline,
  correctness tests, shape coverage, and benchmark transcript.

- 2026-04-20 21:34 UTC: implemented Phase H Example 1 in
  `python/examples/gluon/05-tmem-moe-router.py`. The example covers narrow
  MXFP8 router projection for `E=32/64`, `K=128/256`, a padded `E=128`
  pre-generalization baseline, PyTorch top-k wrapper validation, TTGIR
  checks for `tensor_memory_linear` and `ttng.tc_gen5_mma_scaled`, and inline
  benchmark transcripts. Focused validation passed `8/8`. Next concrete slice:
  Example 2 LoRA / adapter projection fusion.

- 2026-04-20 21:51 UTC: implemented Phase H Example 2 in
  `python/examples/gluon/06-tmem-lora-fusion.py`. The example covers compact
  MXFP8 LoRA down projection for `R=32/64`, `K=128/256`, a padded `R=128`
  pre-generalization baseline, a full LoRA update wrapper, TTGIR checks, and
  inline benchmark transcript. Focused validation passed `8/8`. Next concrete
  slice: Example 3 candidate-head projection.

- 2026-04-20 22:04 UTC: implemented Phase H Example 3 in
  `python/examples/gluon/07-tmem-candidate-head.py`. The example covers compact
  selected-candidate MXFP8 logits for `C=32/64`, `K=128/256`, padded `C=128`
  baseline, candidate-order validation, TTGIR checks, and inline benchmark
  transcript. Focused validation passed `6/6`. Next concrete slice: Example 4
  layout-as-epilogue store ordering.

- 2026-04-20 22:18 UTC: implemented Phase H layout-as-epilogue example in
  `python/examples/gluon/08-tmem-layout-as-epilogue.py`. The example covers
  direct consumer-order stores for `N=64/128/256`, a canonical-store plus
  PyTorch reorder baseline, and inline benchmark transcript. Focused
  validation passed `5/5`. Next concrete slice: Example 5 fused MLP
  side-projection.

- 2026-04-20 22:31 UTC: implemented Phase H MLP side-projection example in
  `python/examples/gluon/09-tmem-mlp-side-projection.py`. The example covers
  compact side projections `S=32/64`, `K=128/256`, a padded `S=128` baseline,
  a broad-projection/gate wrapper, TTGIR checks, and inline benchmark
  transcript. Focused validation passed `8/8`. Next concrete slice: windowed
  attention score reductions.

- 2026-04-20 22:45 UTC: implemented Phase H windowed attention score/reduction
  example in `python/examples/gluon/10-tmem-windowed-attention-score.py`. The
  example covers TMEM row-max reductions for noncausal and causal score tiles,
  a PyTorch mask/max baseline, Blackwell Ultra gating, and inline benchmark
  transcript. Focused validation passed `5/5`. Next concrete slice: ragged
  expert views.

- 2026-04-20 23:01 UTC: implemented Phase H ragged expert panel example in
  `python/examples/gluon/11-tmem-ragged-expert-views.py`. The example covers a
  Python-scheduled ragged expert list with one empty expert, compact per-expert
  panels, a precomputed padded `N=128` per-expert baseline, correctness checks,
  and inline benchmark transcript. Focused validation passed `2/2`. All seven
  planned Phase H example files now exist; next concrete slice is broad
  examples validation and final plan refresh.

- 2026-04-20 23:14 UTC: broad Phase H validation passed. Py-compile passed for
  `python/examples/gluon/05-tmem-moe-router.py` through
  `python/examples/gluon/11-tmem-ragged-expert-views.py`; combined focused
  pytest over the seven files passed `42 passed in 26.83s`. Phase H is now
  complete for the requested example suite, with caveats recorded in the
  individual example files and plan.

- 2026-04-20 23:25 UTC: Phase H timed comparisons were rewritten to remove
  PyTorch post-processing from benchmark paths. Router top-k, LoRA update,
  layout reorder, MLP side gate, and attention mask-plus-row-max now use plain
  Triton kernels without the new TMEM features. PyTorch remains only for
  correctness references and data setup. Py-compile passed for all seven files,
  combined pytest passed `42 passed in 6.53s`, and benchmark transcripts were
  refreshed from local script runs.

- 2026-04-21 00:05 UTC: removed
  `python/examples/gluon/10-tmem-windowed-attention-score.py`. Rebenchmarking
  against the optimized plain Triton mask-plus-row-max baseline showed the
  standalone TMEM `load_max` example remained slower. Attempts to improve
  parallelism by row-blocking the TMEM tile at `BLOCK_M=32` and `BLOCK_M=64`
  failed at compile time on real TMEM descriptor-view support boundaries, so
  the example did not meet the performance bar for a user-facing example.
  Required `make -j8` was no-op, py-compile passed for the retained six files,
  and combined pytest passed `37 passed in 23.52s`.

- 2026-04-21 00:30 UTC: started three-example consolidation requested by the
  user. Merged candidate-head and ragged expert implementations into
  `05-tmem-moe-router.py`, merged MLP side projection into
  `06-tmem-lora-fusion.py`, and deleted the standalone `07`, `09`, and `11`
  files. Required `make -j8` was no-op, py-compile passed for the three retained
  files, focused pytest passed `37 passed in 23.47s`, and the three script
  benchmark transcripts were refreshed from local runs.

- 2026-04-20 23:34 UTC: active phase is upstream-main merge recovery. Merged
  `upstream/main` at `2c7ce4925d37802dd84dfde1f6458cae19485617` into
  `codex/tmem` from pre-merge `c9166449eeaad20516a3f831c0899c56d91f9fc8`.
  Completed build and conflicted lit recovery: `make -j8` passed with the host
  C++ include path, and the six conflicted lit files passed. Open validation
  blockers are runtime-only: two-CTA scale-copy tests now hit the clean
  unsupported broadcast `warpx4` copy boundary, and scaled-MMA FPSAN payload
  checks fail for both legacy and linear accumulators. Next concrete
  implementation slice is to make the scale-copy tests respect the hardware
  boundary or extend the planner/API, then root-cause the FPSAN payload drift.

- 2026-04-20 23:55 UTC: post-merge runtime recovery completed for the focused
  scaled-MMA/copy surface. The FPSAN payload drift was an oracle merge bug and
  is fixed by carrying separate A/B element types through
  `_mm_scaled_payload_u32`. Two-CTA scaled-copy and 2x2 direct multicast
  accumulator readback cases are now explicit xfails for documented
  hardware/API boundaries. Validation passed for the focused runtime slices,
  retained examples/tutorial, and a split 4-GPU selector over the affected
  Gluon tests. Next concrete slice is final `make`/lit sanity, commit the
  post-merge runtime follow-up, push, and then decide whether a broader
  non-selector Gluon sweep is warranted.

- 2026-04-21 01:13 UTC: superseded the previous scaled-copy/direct-multicast
  xfail classification per user direction. Runtime tests green on main are not
  branch-xfail candidates. Implemented the backend fixes for larger-CGA
  scaled `tcgen05.copy` and 2x2 direct multicast accumulator load selection,
  removed the xfails, restored upstream lit coverage, and audited MLIR test
  symbols against `origin/main` merge-base (`TOTAL_MISSING_FUNCS 0`). Current
  active slice is ready to commit/push after final status review; next broad
  work can resume GB200/Gluon sweeps from a coverage-preserving branch.

- 2026-04-21 02:30 UTC: completed a static multi-CTA CGA backend audit and
  recorded it in
  `.codex/initiatives/tmem_linear_generalization/multi_cta_cga_backend_audit_20260421.md`.
  Active follow-up checklist:
  1. Add mixed-kernel lit coverage for TMA load/gather so unrelated TMA ops do
     not inherit `cta_group::2` merely because the module has two-CTA TMEM.
  2. Add return-side cluster-barrier coverage for `num_ctas > 2` kernels that
     mix two-CTA TMEM with other distributed shared-memory/TMA dependencies.
  3. Add PTX coverage proving 4-CTA TMEM copy/commit emits from each even CTA
     pair leader, not only CTA 0.
  4. Add instrumentation/barrier-recipient coverage for TMEM copy with
     outer-CGA barrier ownership.
  5. Keep module-wide mixed single-CTA/two-CTA TMEM as an explicit design
     restriction until all consumers derive instruction group from per-op or
     per-type state.

- 2026-04-21 02:35 UTC: user clarified module-wide two-CTA mode is a hardware
  restriction for all two-CTA-capable instructions, so TMA load/gather
  `getModuleTwoCTAs` use is not a gap. Implemented the real gap exposed by the
  follow-up 4-CTA probe: `tcgen05.copy.warpx2` shared-source validation now
  accepts canonical outer CTA-pair block bases for `128 * num_ctas` by 4
  source tiles. Added conversion and membar lit coverage for 4-CTA pair-leader
  lowering and pre-copy cluster-barrier insertion. Validation passed: required
  `make -j8`, then focused lit for
  `test/Conversion/tritongpu_to_llvm_blackwell.mlir` and
  `test/TritonNvidiaGPU/membar-cluster.mlir`.

- 2026-04-21 02:51 UTC: extended the multi-CTA coverage to the maximum
  practical CTA-per-CGA point used by this backend path: 16 CTAs. Conversion
  lit now covers 4/8/16 CTA `tcgen05.copy.warpx2` canonical outer block bases,
  and membar lit covers 16-CTA pre-copy cluster-barrier insertion. Validation
  passed: required `make -j8` no-op and focused lit for the two touched files
  passed `2/2`.

- 2026-04-21 04:16 UTC: completed correctness-first recovery for the requested
  Gluon example benchmark targets. Fixed the attention non-causal joined-N row
  max layout mismatch, the `OptimizePartitionWarps` memdesc-capture AxisInfo
  assertion, and the single-fragment narrow-N scaled-MMAv5 B-scale legality
  false negative hit by the example-5 reference persistent matmul. Validation
  passed: required `make -j8`; saved `triton-opt --run-reproducer` for the
  original partition-warps crash; exact attention and MoE repro nodeids; and
  four-GPU split sweep over examples 01 and 05 with `112/112` selected tests
  passing. Next concrete step is performance comparison/benchmarking for the
  now-green examples, unless the user asks for broader correctness sweeps first.

- 2026-04-21 06:46 UTC: completed another adversarial scaled-MMAv5 audit slice.
  Fixed the B-scale rematerialization gap where verifier accepted an
  unpadded/repeated-N32 scaled MMA because padded rematerialization was
  possible, but the rewrite refused any original B-scale descriptor with an
  additional live user. The rewrite now rematerializes a separate padded
  descriptor for the MMA and preserves the original descriptor for other users.
  Validation: required `make -j8`; exact new runtime nodeid passed; neighboring
  scaled selector passed `22 passed, 1593 deselected`; py-compile and
  `git diff --check` passed. Concurrent `TensorMemoryUtils.cpp` edits were not
  staged.

- 2026-04-21 06:47 UTC: completed the requested copy/ld/st/ld.red-focused
  adversarial audit slice. Found and fixed a real multicast-copy ownership
  false negative: 128-row two-CTA `TensorMemoryScalesLayout` destinations
  expose `block=1 -> (64, 0)`, but the copy support checker required
  `[[128, 0]]` for every multicast family. The checker now uses
  `min(row_extent / 2, 128)` for the pair-local selector. Validation passed:
  required `make -j8`; exact scales two-CTA copy nodeid; paired single/two-CTA
  scales copy nodeids; copy adversarial selector `31/31`; descriptor-chain
  ld/st selector `128 passed, 61 skipped`; ld.red selector `52/52`; lit
  `3/3`; `git diff --check`.

- 2026-04-21 08:33 UTC: completed TMEM structural fuzzing round 2, Lane
  Expansion, in report-only discovery mode. Required `make -j8` was no-op.
  Temporary `/tmp` Python/Gluon harnesses expanded around first-round failures
  and found stable adjacent variants in existing buckets: runtime
  `memdesc_index` chain2/chain3 crashes, helper chain0 false-branch and
  `16x64b` miscompiles, ld/st chain2 col-reverse `16x64b` miscompile, broader
  ld.red descriptor-chain plain-load fallback, and resource-valid 2CTA ld.red
  plain-load fallback. Current next discovery slice: promote/minimize the new
  adjacent repros as strict xfails where useful, and compare the 2CTA ld.red
  row against the checked-in two-CTA direct higher-rank positive before any
  repair work.

- 2026-04-21 08:43 UTC: completed TMEM structural fuzzing round 3 and promoted
  the highest-value new sentinels while preserving discovery-only mode. Four
  subagent lanes produced reports for dynamic `memdesc_index`, ld.red opcode
  loss, MMA/copy clean surfaces, and ld/st read-only descriptor-view
  minimization. New checked-in strict xfails:
  `generic-pass-dynamic-index-load-only-128x32`,
  `ldst-fz20260421-0003-chain2-col-reverse-64x32-16x64b`, and
  `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min`. Required
  `make -j8` was no-op; py-compile passed; collect-only found `21` structural
  fuzzer nodeids; exact new sentinels each xfailed; full structural fuzzer
  passed as `9 passed, 12 xfailed`. Existing clean-surface sanity sweeps passed
  for ld.red descriptor matrix (`50/50`), copy warpx2/scales (`103/103`), and
  scaled-MMA descriptor/narrow/tile selectors (`139/139`). Current active
  phase remains continuous fuzzing/cataloging with no backend fixes until
  findings stop or the user pivots.

- 2026-04-21 08:50 UTC: completed TMEM structural fuzzing round 4 sentinel
  promotion. Added eight strict xfails for non-overlapping round-four findings:
  helper/control-flow false branch `16x128b`, inline chain0, tuple-like mixed
  capture, layout-pressure `16x128b`, f16 ld/st subword chain2 identity, and
  2CTA indexed ld.red max/abs/NaN opcode-loss rows. Validation passed:
  py-compile, collect-only `29` nodeids, exact new sentinels `8 xfailed`, and
  full structural fuzzer `9 passed, 20 xfailed`. Report-only follow-ups:
  dynamic `memdesc_index` lit candidate policy, R4-D row/col chain1 optimizer
  crash with a crash-safe harness, and continued clean-negative vs opcode-loss
  separation.

- 2026-04-21 11:03 UTC: completed Round 14 Lane AE `ld/st` descriptor-algebra
  fuzzing in discovery-only mode. Report:
  `agents/fuzz_ldst_descriptor_algebra_round14.md`. Required `make -j8` was a
  no-op. Temporary rank-6/runtime-selector probe classified `12` rows as `1`
  pass, `2` `FZ-20260421-0001`, `2` `FZ-20260421-0003`, `5` clean
  direct-`ld/st` row-anchor diagnostics, and `2` clean rank-view diagnostics.
  Checked-in `ld/st` descriptor/rank/subword selector passed split-4 as
  `134 passed, 98 skipped`; adjacent copy/MMAv5 controls passed as
  `243 passed`. No new independent bucket was assigned. Active phase remains
  continuous discovery/cataloging with backend repairs deferred.
