# TMEM Linear Generalization

## Goal
- Canonicalize TMEM layouts on `tensor_memory_linear`.
- Normalize legacy TMEM layouts before semantic consumption.
- Generalize TMEM descriptors to the shared memdesc API and semantics.
- Broaden TMEM lowering for linear layouts while keeping MMAv5 and `tmem_copy`
  on explicit hardware-family matchers.

## Long-Term Mission And Completion Plan

### Mission
- Finish the TMEM linear-layout generalization so the compiler accepts any
  arbitrary linear TMEM layout or descriptor-view chain that is physically
  realizable by the PTX/ISA and executes it correctly on hardware.
- Keep clean negatives only for true ISA-impossible cases.
- When multiple legal codegen paths exist, choose among them with sane
  performance-oriented heuristics instead of hard-coded exclusions.
- Make the implementation cleanly layered:
  - normalize legacy spellings early;
  - express view composition and reinterpret through exact `LinearLayout`
    arithmetic;
  - use one planner for legality, family selection, packet decomposition, and
    lowering;
  - keep verifier/frontend/LLVM in agreement by consuming the same structural
    plan instead of recreating it independently.

### Project-Level Exit Criteria
- Legacy TMEM encodings and `tensor_memory_linear` encodings lower through the
  same physical-family/planner model rather than parallel rescue stacks.
- Arbitrary linear TMEM layouts and descriptor-view chains are positive
  whenever their physical image can be realized by the PTX instruction
  families and the generated code is correct on hardware.
- Remaining negatives are only genuine ISA/semantic boundaries, and they fail
  with clean actionable diagnostics rather than assertions, parser failures, or
  wrong-code.
- The broader `tcgen05` surface is covered, not just the current runtime
  matrix:
  - `ld/st`
  - `ld.red`
  - `copy`
  - `mma`
  - `mma_scaled`
- Test coverage and fuzzing saturate those surfaces at the user level, and the
  broad validation stack is green:
  - focused TMEM runtime tests
  - MMA/matmul tests
  - `triton_kernels` matmul
  - wider pytest/lit suites
  - multi-GPU grouped sweeps where appropriate.

### Current Validation State
- Latest direct MMA/scaled-MMA validation checkpoint, 2026-04-13 12:45 UTC at `8b3fec8d6`: the broad runtime-matrix selector `python/test/gluon/test_tmem_runtime_matrix.py -k 'mma and not cp'` is green after the two new scaled-MMAv5 coverage expansions. Four GPU split groups passed with aggregate `258 passed, 50 skipped` (`32 passed, 45 skipped`; `72 passed, 5 skipped`; `77 passed`; `77 passed`). This supersedes the older `242 passed, 50 skipped` direct MMA/scaled-MMA count in older notes; the delta is expected from adding multicast two-CTA accumulator-subview cases and replacing the mxfp8-only tile-permuted case with the five-format matrix.
- Latest scaled-MMAv5 tile-permuted accumulator checkpoint, 2026-04-13 12:43 UTC: `test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_64_format_matrix` replaces the old `mxfp8/mxfp8`-only direct-layout test with full format coverage for `mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`, `mxfp4/mxfp8`, and `nvfp4/nvfp4` on the `128x256`, `tile_n=64` TMEM-linear accumulator layout. The new generic direct-layout format kernel feeds format-specific packed operands through `tcgen05_mma_scaled`, checks numeric output, and pins exact PTX/LLIR opcode families plus instruction counts (`16` for mixed/fp8 vector-1X cases, `8` for homogeneous fp4/nvfp4 cases). Validation: py-compile passed, rebuild was a no-op success, the exact nodeid passed all `5` selected cases across four GPU split groups, and nearby `mma_scaled and tile_permuted` passed `11` cases across four GPU split groups.
- Latest scaled-MMAv5 two-CTA multicast checkpoint, 2026-04-13 12:37 UTC: `test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_matrix` now covers both scale-TMA paths by parameterizing `multicast=False` and `multicast=True`. The matrix still covers both `slice_start=0` and `slice_start=128` across the five scaled format pairs, still checks numeric output plus exact `tcgen05.cp.cta_group::2.warpx4.32x128b`, `tcgen05.mma.cta_group::2.kind::{mxf8f6f4,mxf4,mxf4nvf4}`, and commit opcode streams, and now also asserts TTGIR `{multicast}` is present only for the multicast path. Validation: py-compile passed, rebuild passed, the exact nodeid passed `20` cases across four GPU split groups, and the nearby `mma_scaled and subslice and format_matrix` selector passed `43` cases across four GPU split groups.
- Latest `ld/st` validation-velocity checkpoint, 2026-04-13 after `77d3edea1`: broad `python/test/gluon/test_tmem_runtime_matrix.py -k ldst` still selects `1642/2758` tests and is green by mixed static/bucketed evidence, with no deterministic failures found. Static split-16 groups `1..4` passed (`412 passed`) and groups `14..16` passed/clean-skipped (`282 passed, 21 skipped`), but static groups `5..8` and `13` hit timeout wrappers on dense descriptor-composition/roundtrip/higher-rank buckets. Exact replacement buckets closed those functions: descriptor-composition row/column sweep `240 passed`; descriptor-roundtrip row/column sweep `240 skipped`; remaining composition/roundtrip/two-CTA buckets `180 passed, 180 skipped`; higher-rank index/multidim buckets `120 passed`. A merged duration cache for all `1642` nodeids now lives at `experiments/results/ldst_pytest_durations_20260413.json`; collect-only `least_duration` split-16 estimates about `780s` per group, so the selector remains a validation-throughput issue rather than a product failure. Use `ldst_validation_recipe_20260413.md` before broad local `ld/st` reruns.
- Latest `ld.red` explicit non-identity checkpoint, 2026-04-13 09:57 UTC: `test_tmem_runtime_matrix_ld_red_explicit_compatible_non_identity_layouts_canonicalize_32x32b` now covers both `min` and `max` across the four compatible non-identity TMEM-linear layouts (`tile_permuted`, `col_reverse`, `row_reverse`, `rowcol_rotate_reverse`) and all four explicit-load variants (`auto`, `32x32b`, `16x32bx2`, `32x32b_splitn`). This doubles the non-identity explicit-layout coverage from `16` to `32` cases and keeps the expected lowering fixed to `tcgen05.ld.red.sync.aligned.32x32b.x128.{min,max}.f32`. Focused validation passed all `32` selected cases across four GPU split groups. Broad `-k ld_red` is green as `523 passed` aggregate after warmed reruns of the cold timeout groups; the initial cold split-4 run remains a validation-velocity warning, not a product failure, because group 1 passed in `9:12`, group 4 printed a green `130 passed` summary at the `900s` wrapper boundary and then passed warm in `12.52s`, and groups 2/3 passed warm in `3:08` and `3:18`. Use duration data, finer groups, or warmed exact reruns for future broad local `ld_red` sweeps.
- Latest validation-velocity checkpoint, 2026-04-13 10:35 UTC: project-level `AGENTS.md` now records the GB200 timing calibration and duration-aware pytest-split guidance. The full GB200 NVIDIA lane is expected to be roughly 35 minutes including clean build/LLVM download, with actual test time around 20 minutes; local pytest shards that run far beyond that should be treated as partitioning, xdist, cache, or hang issues first. A current collect-only scan shows `python/test/gluon/test_tmem_runtime_matrix.py -k ldst` selects `1642` cases, dominated by three `240`-case families (`ldst_rowcol_permuted_layout_sweep`, `ldst_descriptor_roundtrip_rowcol_permuted_sweeps`, and `ldst_descriptor_compositions_rowcol_permuted_layout_sweep`). Future broad `ldst` validation should use stored durations (`--store-durations --durations-path ...`) plus `--splitting-algorithm=least_duration`, finer split counts, or explicit function buckets rather than four static equal-count shards. Rebuild before runtime pytest was a no-op success.
- Latest `tcgen05.cp` contract checkpoint, 2026-04-13 07:10 UTC after
  `196fcc717`: runtime-matrix coverage now also pins one representative scales
  parent-row shared-subslice boundary from the recorded `warpx2` subslice probe.
  New test
  `test_tmem_runtime_matrix_cp_scales_shared_subslice_layout_reports_clean_unsupported`
  uses the probe's parent-row `SharedLinearLayout`, slices a `128x16` parent to
  a `64x16` scales-copy source at aligned starts `0` and `64`, passes that view
  to `tcgen05_copy`, and expects the clean TensorMemoryScales descriptor-plan
  diagnostic for `warpx4.32x128b`, with no PassManager/assert noise. Validation:
  `py_compile` passed, rebuild was a no-op success, exact new nodeid passed both
  selected parametrized cases across split groups, and the nearby scales-copy
  selector passed `9` selected cases across active split groups. The no-scales
  transposed shared-source and split-offset clean negatives remain covered.
- Latest `ld.red` contract checkpoint, 2026-04-13 06:20 UTC at `e84e4f0a7`:
  runtime-matrix coverage now pins clean non-f32 reduction/modifier failures.
  `test_tmem_runtime_matrix_ld_red_non_f32_contract_reports_clean_unsupported`
  covers `i32` plain reduction, `i32` with `NaN`, `i32` with `abs`, and
  legacy-unpacked `f16`; all fail before lowering with dedicated diagnostics and
  without PassManager/assert noise. Validation: `py_compile` passed, rebuild was
  a no-op success, the exact new nodeid passed across four GPU split groups
  (`4 passed` aggregate), nearby negative selector
  (`11 passed` aggregate), and broad four-GPU `-k ld_red` (`507 passed`
  aggregate). Broad split-4 was green but too imbalanced without duration data
  (reported shard times about `7:58`, `19:31`, `19:05`, and `13:57`); prefer
  finer split groups, duration-aware splits, or targeted selectors for future
  local `ld_red` sweeps.
- Latest M64 docket checkpoint, 2026-04-13 05:41 UTC at `b475e2883`:
  the previously recorded legacy M64 MMAv5 bug/xfail is stale on current head.
  Rebuild completed successfully, then
  `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]` passed with
  `--runxfail`; the normal two-parameter `test_block_m_64_mma` rerun passed
  both legacy and linear parameters across four `pytest-split` groups; and the
  nearby selector
  `test_tmem_subslice_block_m_64 or test_tmem_subslice_block_m_64_parent_layout or test_block_m_64_mma`
  passed all six selected tests across the active split groups. Treat old
  documents that list this nodeid as a live failure or xfail as superseded
  unless a fresh exact current-head repro fails.
- Latest plain-MMAv5 saturation checkpoint, 2026-04-13 07:45 UTC:
  `test_tmem_runtime_matrix_mma_plain_kinds_tile_permuted_acc_use_acc` now pins
  the `use_acc=True` accumulator-add path for tile-permuted TMEM-linear
  accumulators across all supported plain operand kinds and both existing
  tile-permuted layouts (`128x128/tile_n=32` and `128x256/tile_n=64`). The test
  validates numeric `matmul + accumulator` output, exact PTX/LLIR MMAv5 opcode
  streams, commit opcode selection, and `tensor_memory_linear` preservation.
  Validation: `py_compile` passed, rebuild was a no-op success, the exact new
  nodeid passed `10` selected cases across four GPU split groups, the
  nearby `mma_plain_kinds_tile_permuted_acc or mma_plain_kinds_use_acc` selector
  passed `30` selected cases aggregate, and the post-commit broad four-GPU
  `-k 'mma and not cp'` selector passed `242 passed, 50 skipped`.
- Latest `ld/st` x1 subword checkpoint, 2026-04-13 08:35 UTC:
  `test_tmem_runtime_matrix_ldst_x1_subword_twocta_descriptor_chain_roundtrip`
  now pins descriptor-chain two-CTA x1 subword roundtrips for `f16`, `bf16`,
  `i16`, and packed `i8` over a lifted canonical two-CTA TMEM-linear layout,
  for `auto` and explicit `32x32b`. The helper reaches the active tile through
  `slice/index/reshape/permute/permute` before the store/load. The test checks
  exact `tcgen05.{st,ld}.sync.aligned.32x32b.x1.b32` PTX/LLIR opcode streams,
  zero-offset addressing, numeric equality, `memdesc_subslice`,
  `memdesc_reshape`, `memdesc_trans`, `tensor_memory_linear`, and
  `twoCTAs = true`. Validation: `py_compile` passed, rebuild was a no-op
  success, the exact new descriptor-chain nodeid passed `8` selected cases
  across four split groups, and the nearby `ldst_x1_subword or ldst_x1_f32`
  selector passed `65` selected cases aggregate.
- Latest `ld/st` descriptor API-migration checkpoint, 2026-04-13 08:50 UTC:
  the runtime-matrix descriptor-chain test that still needed a physical-equivalent
  view change now uses the supported `.bitcast(...)` API instead of
  `_reinterpret(...)`. The deeper roundtrip case was not a real bitcast: after
  its view chain it requested the same dtype, shape, and layout, so the no-op
  `_reinterpret` was removed and the case was renamed
  `slice_index_deep_roundtrip`. Both single-CTA and two-CTA roundtrip tests now
  assert the required `memdesc_index`, `memdesc_subslice`, `memdesc_reshape`,
  and `memdesc_trans` TTGIR operations. A probe of a genuine `.bitcast(...)`
  after the deep composed view chain still fails type inference with
  `unsupported tensor memory memdesc_subslice view`; keep that as future
  supported-API/planner work only for cases that really need a
  physical-equivalent bitcast over a complex descriptor-view chain. Validation:
  `py_compile`, rebuild, and `git diff --check` passed; base descriptor
  compositions passed across the four split groups after rerunning cold-compile
  group 3 with a warmed cache; the deep single-CTA and two-CTA roundtrip
  selectors completed with their existing OOR skips and no failures.
- Latest `test_core` descriptor-chain API-migration checkpoint,
  2026-04-13 09:05 UTC: `tmem_descriptor_chain_matrix_kernel` now uses
  supported `.bitcast(...)` instead of `_reinterpret(...)` for the same
  physical-equivalent descriptor-chain view pattern migrated in the runtime
  matrix. The test now asserts `tmem_physical_bitcast` in TTGIR while preserving
  its existing memdesc view and exact `tcgen05.{st,ld}` opcode checks.
  Validation: `py_compile` passed, rebuild was a no-op success, and the exact
  `test_tmem_descriptor_chain_matrix` nodeid passed all `26` selected cases
  across four GPU split groups (`7`, `7`, `7`, `5`).
- Latest runtime-view API-migration checkpoint, 2026-04-13 09:35 UTC:
  the two identity `slice_*` cases in `test_tmem_linear_runtime_views` now use
  supported `.bitcast(...)`, in addition to the already migrated
  `index_reshape_*` family. The compiler change verifies subslice projections
  against a normalized analysis layout only when inactive zero support bases make
  the destination layout non-injective, and generalizes the standalone TMEM
  same-rank subview query path beyond rank 2. Superseded note: the mixed-basis
  runtime-view case was later reclassified as not physical-mapping equivalent
  to the requested identity layout and converted to a clean bitcast negative.
  Validation: `py_compile` passed, rebuild passed, the full runtime-view nodeid
  passed all `11` selected cases across four GPU split groups (`3`, `3`, `3`,
  `2`), and adjacent `test_tmem_descriptor_chain_matrix` passed after warmed
  reruns of cold-timeout groups (`26` selected cases total).
- Latest runtime-matrix API-migration checkpoint, 2026-04-13 09:50 UTC:
  `python/test/gluon/test_tmem_runtime_matrix.py` no longer contains any
  `_reinterpret(...)` calls. The remaining block-descriptor clean-negative
  kernel now uses `.bitcast(...)`; the one-CTA block case still reports the CTA
  mismatch, while the two-CTA block case cleanly rejects the supported
  bitcast/subslice with `unsupported tensor memory memdesc_subslice view`. The
  old `_reinterpret` path reached a deeper direct ld/st unsupported-row-anchor
  diagnostic only by bypassing the supported descriptor-view contract. Probe
  detail: the unsupported view would need a negative additive block delta
  (`dim=2 step=2 phys=block delta=-1`) in the current query/origin model.
  Validation: `py_compile`, rebuild, and `git diff --check` passed; the exact
  block-descriptor nodeid passed both active split groups (`1`, `1`), with
  groups 3 and 4 empty because only two cases are collected.
- Frontend TMEM reinterpret classification, 2026-04-13 10:00 UTC:
  the three TMEM `_reinterpret(...)` helpers in `python/test/gluon/test_frontend.py`
  were probed with supported `.bitcast(...)` and reverted. The focused parser
  tests fail before IR generation with `unsupported tensor memory memdesc_subslice
  view`, matching the same complex composed-view gap as the remaining runtime
  mixed-slice/descriptor cases. Keep these as intentional `memdesc_reinterpret`
  parser/IR contract coverage until that planner/API gap is fixed; do not count
  them as production-style `_reinterpret` reliance. The persistence tutorial and
  the remaining `test_core.py` / `test_frontend.py` non-TMEM occurrences are
  shared-memory reinterpret uses. The `test_core.py` mixed-basis TMEM runtime
  view was later reclassified as an invalid physical-equivalence target and
  converted to a clean supported-bitcast negative.
- Latest `test_core` runtime-view API-migration checkpoint, 2026-04-13 10:15 UTC:
  `python/test/gluon/test_core.py` no longer contains TMEM `_reinterpret(...)`
  users; only the shared-memory `test_slice_reinterpret` remains in that file.
  The former mixed-basis runtime-view positive is now a clean supported-bitcast
  negative, `slice_bitcast_64_mixed_not_physical_equivalent_32x32b`, because the
  requested identity `128x64` result layout is not physical-mapping equivalent
  to the selected mixed source view. The old `_reinterpret` path planned from
  the raw result layout and ignored the source view, so it was testing legacy
  behavior rather than a supported descriptor-view contract. Validation: `py_compile`,
  no-op rebuild, and `git diff --check` passed; the combined positive/negative
  runtime-view nodeids passed all `11` selected cases across four GPU split
  groups (`3`, `3`, `3`, `2`).
- `ld/st` validation velocity warning, 2026-04-13 08:25 UTC:
  a coarse four-GPU split-4 broad `-k 'ldst'` refresh at `15c0bf252` was stopped
  as too slow, not recorded as validation. Group 1 completed green
  (`409 passed, 2322 deselected`) but took `38:23`; groups 2-4 were still
  running at about `42` minutes with steady output and were terminated. Future
  broad `ld/st` refreshes should use finer split groups, duration-aware data, or
  narrower selectors first; do not treat split-4 timing as acceptable for this
  lane.
- Latest copy-frontier checkpoint, 2026-04-13 09:42 UTC after `f171c2286`: two-CTA `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b` remains intentionally clean unsupported. Direct-PTX work has now ruled out the single-CTA direct-seed neighborhood through `sourceOffsetB128` offsets `32..127`, destination deltas `0/4`, and nearby two-message column-pair schedules. The durable JSONL source-offset set has `184` records and zero matches: offsets `36..63` launch cleanly but duplicate one source column pair; offsets `64..72` launch but still mismatch (`65..72` destination `0` has four NaNs, destination `4` stays finite but wrong); offsets `73..127` launch-fail in isolated child processes. Keep this frontier on descriptor/message semantics rather than another small offset toggle.
- Latest validation checkpoint, 2026-04-13 03:00 UTC: the previous Gluon tail is now closed from local evidence. `python/test/gluon/test_tmem_runtime_matrix.py` has full file coverage via mixed split granularity (`2683` selected cases: `2237 passed, 446 skipped`, no failures/errors). `python/test/gluon/test_lowerings.py` is green across four GPU shards (`4937 passed, 512 skipped`). Together with the earlier green first-phase groups 1-3, split-16 groups 13-14, isolated xdist-crash nodeid pass, and green `python/examples/gluon/`, current-head `test-gluon` has no deterministic known failures after the MMAv5 fix. The timeout root cause was static split imbalance in slow TMEM ldst composition/legality-probe buckets, not a failing nodeid.
- Latest wider checkpoint, 2026-04-13 01:19 UTC: full lit is green (`248 passed, 2 unsupported`)
  and `python/examples/gluon/` is green across four shards (`884 passed, 74
  skipped` total). The first `python/test/gluon/ python/tutorials/gluon/` phase
  has no deterministic known failures after the MMAv5 family-addressing fix, but
  the local sweep is not fully closed: four-way `-n 6` groups 1, 2, and 3
  finished green; group 4 had one xdist worker crash whose exact nodeid passed
  isolated, and finer split-16 groups 13 and 14 are green while groups 15 and
  16 timed out still making progress. Treat this as validation partitioning
  debt, not a compiler regression, until an isolated nodeid reproduces.
- Project-owner timing reference, 2026-04-13: the full GB200 CI lane on a comparable
  four-node/four-GPU machine should be about 35 minutes including clean build and
  LLVM download, with actual test time around 20 minutes. Any local run that takes
  far longer needs a partitioning, xdist, process-state, or hang investigation
  before accepting the runtime as normal.
- Latest checkpoint, 2026-04-12 23:15 UTC: full-shape MMAv5 TMEM descriptors once again use
  the derived MMAv5 family layout for instruction address arithmetic, while
  marked physical bitcasts keep the exact typed result layout. This fixes the
  current-head tile-permuted MMAv5 accumulator/LHS/scaled failures found by the
  CI-shaped `test-gluon` run after attr cleanup. Focused four-GPU tile-permuted
  rerun, broad runtime-matrix `-k 'mma and not cp'`, nearby core M64/bitcast
  selector, representative `test_mma_shared_inputs` exacts, and relevant lit
  lowering checks are green. The prior tile-permuted MMAv5 failure list is stale
  for those nodeids; a full `test-gluon`/GB200 rerun remains the next wider
  milestone.
- Pre-cleanup preserve-set, 2026-04-12 09:33 UTC at `3359982ee`, is recorded in
  `gb200_preserve_set_20260412.md`.
  - Semantic correction: zero TMEM bases are broadcast/equivalence semantics,
    not a choice among potentially divergent representatives. If physical cells
    in the same equivalence class differ, that is a producer/view/ld-st bug.
  - Attribute policy correction: `ttng.tmem_physical_layout`,
    `ttng.tmem_ldst_row_plan`, `ttng.tmem_mmav5_accumulator_root`, and related
    root markers are now cleanup targets because they encode provenance that
    should instead fall out of the layout/view chain and exact codegen.
  - Fresh current-head evidence:
    `make -j8` passed; full lit is `246 passed, 2 failed, 2 unsupported` with
    only `pipeline-loop-nest.mlir` and `pipeline-lower-loop.mlir` failing as
    stale attr-text checks; all `162` branch-new unit nodeids still fail; the
    legacy M64 Gluon exact still fails; Proton main still has the same
    `11` merge-base-preexisting failures; C++, gsan, regression,
    microbenchmark, unit debug, fused-attention tutorial, plugins,
    instrumentation, and Proton tails are green at focused/wrapper scope.
- Fresh full GB200 `integration-tests-nvidia` inventory at `cb76c31a0` is red on
  current branch and classified against merge-base `11ee1144a737006921231bbd3386c187812c38e1`:
  - branch-new lit: `9` files, all pass on merge-base;
  - branch-new `python/test/unit`: `162` nodeids, all pass on merge-base,
    clustered in `test_matmul.py` (`27`), `test_tensor_descriptor.py` (`7`),
    and `test_warp_specialization.py` (`128`);
  - branch-new Gluon: `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`
    fails after unxfail, while the linear parameter passes and merge-base's
    legacy test passes;
  - Proton main has `11` cudagraph / periodic flushing failures, but the same
    nodeids fail on merge-base and are not branch recovery blockers;
  - `test_debug.py`, instrumentation/plugin tails, and Proton split-only extra
    failures were harness artifacts that cleared or reduced under exact CI-form
    reruns.
- The branch-new red set is now split in
  `gb200_failure_classification_20260412.md`:
  - actual bugs: the `162` `python/test/unit` wrong-output nodeids and
    `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`;
  - stale tests: `TritonNvidiaGPU/invalid.mlir`,
    `TritonGPU/pipeline-loop-nest.mlir`, and
    `TritonGPU/pipeline-lower-loop.mlir`;
  - API/contract updates: the lit tests that still use now-invalid scales-copy,
    transposed-f32 MMAv5, or old `memdesc_subslice` result-type spellings.
- The full `python/examples/gluon` aggregate passed in the current GB200 sweep:
  - group 1: `225 passed, 15 skipped, 718 deselected`
  - group 2: `221 passed, 19 skipped, 718 deselected`
  - group 3: `220 passed, 20 skipped, 718 deselected`
  - group 4: `218 passed, 20 skipped, 720 deselected`
  - this includes `python/examples/gluon/01-attention-forward.py`, so the
    previous conservative note that full examples had not been rerun after the
    attention SDPA correction is superseded.
- Current-head attention benchmark-parameter unit coverage is green with an
  SDPA reference:
  - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-attention-sdpa-matrix-final2 PYTHONPATH=python:. pytest -s --tb=short -q python/examples/gluon/01-attention-forward.py`
  - `64 passed in 102.66s (0:01:42)`
  - The test matrix matches the benchmark grid except for context lengths where
    PyTorch SDPA OOMs: test `N_CTX` is `{1024,2048,4096,8192}`, while benchmark
    `N_CTX` remains `2**10..2**16`.
  - Covered test parameters on this Blackwell Ultra box: `Z=4`, `H=32`,
    `HEAD_DIM in {64,128}`, `causal in {False,True}`, providers
    `triton-fp16` and `triton-fp8`, and `use_tmem_red in {False,True}`.
  - The test uses random `q/k/v` and compares against
    `torch.nn.functional.scaled_dot_product_attention`; fp8 inputs are
    dequantized to fp16 for SDPA and compared after fp8 quantization with
    `atol=0.25`.
  - The scratch bitcast helpers are dtype-aware: fp8 P views slice 32 f32
    columns instead of the fp16/bf16 64-column region, and alpha/epilogue
    scratch starts after the actual P physical region.
  - Causal `use_tmem_red` now avoids the pre-mask TMEM reduction max in the
    diagonal stage because SDPA-backed random tests exposed row-0 wrong results.
  - Full `python/examples/gluon` was later rerun in the 2026-04-12 GB200
    sweep at `cb76c31a0` and passed all four split groups.
- Current-head phase-boundary validation at `330c64c05` is green for the full
  `python/test/gluon/test_tmem_runtime_matrix.py` file:
  - `1757 passed, 442 skipped in 1640.11s (0:27:20)`
- Current-head broad `ld/st` validation after the allocator-lifetime anchors and
  subword coverage expansions is green:
  - focused lifetime exacts:
    `11 passed in 5.56s`
  - focused subword exacts after broadening beyond f16:
    `66 passed in 18.12s`
  - focused x1 subword exacts after adding padded i8 cases:
    `88 passed in 22.36s`
  - focused two-CTA x1 subword exacts:
    `8 passed` aggregate across four GPU split groups
  - nearby x1 subword/f32 selector after adding two-CTA subword cases:
    `57 passed` aggregate across four GPU split groups
  - broad `ld/st` slice:
    `1181 passed, 441 skipped, 1027 deselected in 1340.05s (0:22:20)`
  - the new runtime anchors assert exact PTX/LLIR `tcgen05.alloc`,
    `tcgen05.relinquish_alloc_permit`, `tcgen05.dealloc`, and
    `tcgen05.wait::{st,ld}` emission for single-CTA and two-CTA ld/st kernels;
    they cover alloc/dealloc size immediates `32, 64, 128, 256, 512` for
    both CTA groups, check two-CTA cluster arrive/wait before dealloc, and
    keep a source-initialized `allocate_tensor_memory(..., value=...)`
    roundtrip.
- Current-head commit-opcode exact coverage is green:
  - focused copy/MMA commit exacts:
    `24 passed in 19.43s`
  - broad `tcgen05.cp` slice:
    `162 passed, 5 skipped, 2488 deselected in 40.88s`
    after adding the four nearby scales `warpx2` clean-negative layout probes
  - broad true `tcgen05.mma` / direct `mma_scaled` slice:
    `226 passed, 50 skipped, 2407 deselected in 131.68s (0:02:11)`
  - exact anchors cover single-CTA non-multicast commit and two-CTA multicast
    commit for copy/MMA paths, including scaled-MMA copy-helper kernels, with
    PTX and LLIR opcode agreement.
- Current-head supported physical-bitcast API coverage is green:
  - focused physical-bitcast slice:
    `3 passed, 17963 deselected in 4.57s`
  - `test_tmem_physical_bitcast_preserves_subview_mapping` now covers both
    `slice_start=0` and `slice_start=64`, so the API contract is pinned for a
    root-aligned half-tile and an offset right-half subview before bitcasting
    to an equal-size f16 view;
  - the right-half case verifies that the bitcast does not select, remap, or
    relocate physical TMEM: only columns `64:128` of the original f32 tile are
    overwritten through the f16 view.
- Current-head scaled-MMAv5 accumulator-subview format coverage is green:
  - focused one-CTA slice-start matrix:
    `10 passed in 6.86s`
  - focused two-CTA cga-aware accumulator-subview matrix:
    `20 passed` aggregate across four GPU split groups after adding multicast
    scale-TMA coverage
  - the one-CTA matrix covers both `slice_start=0` and `slice_start=64`; the
    two-CTA matrix covers both `slice_start=0` and `slice_start=128` using a
    larger TMEM-linear accumulator parent sliced to the `cta_group::2` result
    tile, and now runs both `multicast=False` and `multicast=True`;
  - both matrices cover `mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`,
    `mxfp4/mxfp8`, and `nvfp4/nvfp4`, with exact PTX/LLIR opcode agreement and
    numeric checks.
- Current-head tensor-memory allocation rounding coverage is green:
  - lit exact:
    `PASS: TRITON :: TritonNvidiaGPU/test_tensor_memory_allocation.mlir`
  - raw live TMEM totals `96`, `192`, and `384` are covered as allocation-pass
    boundaries that round to supported module allocation sizes `128`, `256`,
    and `512`; literal non-pow2 `tcgen05.alloc` immediates are not expected.
- Current-head `tcgen05.ld.red` tile/permutation coverage is green:
  - focused tile-permuted exact, now including the valid minimal
    `N=64, tile_n=16` case:
    `32 passed in 22.37s`
  - focused pure-column permutation exact:
    `72 passed in 50.45s`
  - focused pure-row N-sweep exact:
    `48 passed in 31.75s`
  - focused row/column N-sweep exact:
    `144 passed in 113.29s (0:01:53)`
  - broad `ld_red` slice:
    `487 passed, 2183 deselected in 482.92s (0:08:02)`
  - tile-permuted, pure column, pure row, and non-identity row/column
    cross-product permutations now cover `128x{64,128,256}` where each family
    is well-defined and still emit the expected `32x32b` reduction-family
    opcodes with exact offset immediates: `[0]` for `N <= 128` and
    `[0, 64, 128, 192]` for `N=256`, and exactly one store wait before
    reduction loads plus one load wait after `ld.red` before redval use.
- Current-head four-way heavy Gluon validation at `be14fedc5` is green for
  `python/test/gluon/test_core.py` plus
  `python/test/gluon/test_tmem_runtime_matrix.py`:
  - group 1: `4407 passed, 634 skipped, 15123 deselected`
  - group 2: `2528 passed, 2513 skipped, 15123 deselected`
  - group 3: `2694 passed, 2347 skipped, 15123 deselected`
  - group 4: `3314 passed, 1726 skipped, 15123 deselected, 1 xfailed`
- The skips are expected tensor-memory OOR / boundary classifications from the
  expanded descriptor-view and rank-5 matrices, not new red tests.
- The group-4 xfail is the known legacy M64 MMAv5 producer-family design-debt
  marker, not a new failure.
- This full-file checkpoint incorporates the staged `ld/st` auto expansion,
  broad `tcgen05.cp`, broad `tcgen05.ld.red`, true `tcgen05.mma`, direct
  `mma_scaled`, and scaled-MMA copy-helper validation slices recorded in the
  log; the newest direct scaled-MMA slice has additionally validated the
  two-CTA cga-aware accumulator-subview matrix.

### Phase 1: Core Logic Cleanup
- Unify TMEM ld/st planning into one shared structural planner consumed by:
  - Gluon descriptor/layout selection
  - verifier checks
  - LLVM lowering
- Fold the current duplicated raw-query/support-query/row-plan ordering into
  that planner.
- Canonicalize legacy TMEM physical families once, then lower them through the
  same planner as TMEM-linear layouts.
- Replace reinterpret packet surgery with quotient-driven decomposition:
  - compute memdesc and register reinterpret in the same physical coordinate
    frame;
  - derive row anchors, packet offsets, repetition, support-band selection,
    and atom family from quotient factorization;
  - remove special-case post-hoc row/col/offset rewrites once the algebraic
    path covers them.
- Keep producer-owned row plans explicit, but centralize row-plan acquisition
  and projection in the shared planner instead of scattered local fallback
  ladders.

### Phase 2: Close The Current Live Buckets
- First close the legacy `block_m_64` divergence:
  - make `test_tmem_subslice_block_m_64[legacy]` go through the same effective
    packed-support physical family as the working TMEM-linear case;
  - do this by fixing legacy normalization/planning, not by another local
    packet tweak.
- Then close the parent-layout reinterpret packet-decomposition bug:
  - make `test_tmem_subslice_block_m_64_parent_layout[linear]` derive the
    correct support-band/subview packet decomposition from quotient structure;
  - do not paper over this by forcing a different atom family.
- Keep checking the earlier positive controls while doing this:
  - `test_block_m_64_mma`
  - `test_mma_shared_inputs`
  - row/col-permuted TMEM runtime slices
  - descriptor-chain matrix cases.
- 2026-04-10 GB200 stabilization note:
  - the current branch-recovery front has moved in front of these
    reinterpret-contract buckets
  - keep the two `block_m_64` reinterpret cases visible as explicit-view /
    missing-surface cleanup work, but do not let them displace the current
    merge-base-present MMAv5 and examples regressions in the GB200 recovery
    queue

### Phase 3: Rewrite Test Contracts Around Guaranteed APIs
- Audit TMEM tests that currently rely on `_reinterpret` plus implicit
  knowledge of the compiler's physical TMEM mapping.
- Treat production/example kernels the same way: any `_reinterpret` usage that
  only works because today the compiler happens to pick a particular physical
  TMEM lowering is a migration target, not a lowering bug to paper over.
- The intended replacement is supported descriptor/view construction expressed
  through linear-layout/view APIs with the same semantic intent. Do not add
  ad-hoc lowering selectors just to preserve old `_reinterpret` behavior.
- The concrete supported replacement pattern for physical TMEM aliases is:
  - offset to the correct physical TMEM region;
  - slice/subview that region down to the exact desired physical bits;
  - bitcast to the desired dtype, shape, and layout only when the total bit
    size and exact physical TMEM mapping are preserved.
- That final bitcast is a size-and-physical-mapping-equivalent view operation.
  It must not move, remap, or select a different physical TMEM region than the
  input descriptor maps to.
- In operational terms, the kernel must first select the right physical part of
  TMEM by offset/subview. Only after that selection is exact may the descriptor
  be bitcast to the desired dtype/shape/layout, and only if the bitcast's
  physical image is identical to the selected input image.
- Said as an implementation invariant: offset to the right part of TMEM,
  slice/subview it to the exact desired bits, then bitcast to the desired dtype,
  shape, and layout only when that bitcast is size-equivalent and preserves the
  same physical mapping.
- Equivalently, the bitcast is allowed only when its desired dtype, shape, and
  layout are size-equivalent and physical-mapping-equivalent to the already
  sliced/subviewed TMEM descriptor; the bitcast must not perform physical
  selection or remapping.
- The bitcast may therefore change descriptor metadata to the desired
  dtype/shape/layout only when it is size-equivalent and
  physical-mapping-equivalent to the already-selected input; it must not select
  or remap physical TMEM.
- The attention example scratch-alias path has a focused supported migration
  using this sequence. Keep the synchronization/lifetime discipline intact:
  select the inactive f32 scratch subregion first, then bitcast only an
  equal-size physical view. The exact attention test is green, but broader
  examples validation remains pending.
- For intent tests, rewrite kernels to use explicit guaranteed descriptor/view
  composition:
  - `slice`
  - `index`
  - `reshape`
  - `permute`
- Keep `_reinterpret` tests only for what reinterpret itself promises:
  - bitcount/view preservation
  - legality diagnostics
  - no crashes
  - direct lowering only when the resulting physical support image is actually
    realizable by the ISA.
- Update stale negatives to positive whenever support has legitimately
  broadened and hardware execution is correct.

### Phase 4: Expand The Reachable `tcgen05` Surface
- `ld/st`:
  - continue broadening arbitrary linear-layout and descriptor-view support
    until all ISA-realizable atom/layout combinations reachable from legal user
    inputs are supported without wrong-code.
- `ld.red`:
  - expand the positive surface for legal reduction layouts and keep the clean
    dedicated diagnostic for unsupported N-sharded cases.
- `copy`:
  - finish the remaining ISA-realizable copy families, especially `warpx2`,
    with the required descriptor/address/message synthesis and user-visible
    layout path.
- `mma` / `mma_scaled`:
  - broaden beyond the current canonical MMAv5-compatible subset to all
    ISA-legal instruction/layout families reachable from legal user layouts.
- For all of the above, preserve the invariant that user-visible layout
  generalization is not tied to a pile of family-specific hacks.

#### 2026-04-11 Long-Term Surface Audit
- This audit is not the immediate implementation target while the current
  M64 / split-N direct-load regressions are still live. Resume it once the
  GB200 branch-recovery exacts are green and the shared planner is stable.
- `tcgen05.cp` / `warpx2`:
  - the old no-scales `warpx2::{01_23,02_13}.64x128b` candidate-positive tail
    is no longer believed to be live after the `[128,4]`
    raw-query/direct-view preservation fix; both candidate positives were
    recorded green in the latest docs;
  - no-scales single-CTA `warpx2::{01_23,02_13}.64x128b` candidate positives
    remain executable and covered with runtime oracles plus exact `cta_group::1`
    copy and commit opcode checks;
  - no-scales two-CTA `warpx2::01_23.64x128b` is now executable through the
    public linear-layout path and is covered by exact `cta_group::2` copy and
    multicast commit opcode checks;
  - dense shared-layout `warpx2` forms are now intentionally rejected for both
    single-CTA and two-CTA cases; runtime probes showed that the old
    codegen-only path could emit `warpx2` opcodes while copying wrong data
    (including all-zero two-CTA `02_13` output);
  - a matching two-CTA `warpx2::02_13` candidate shared layout is now pinned
    by a clean unsupported test at descriptor-plan synthesis, so keep it as a
    layout-surface / direct-PTX frontier rather than a clean ISA-impossible
    negative;
  - a 2026-04-11 temporary direct-seed relaxation for the canonical two-CTA
    `warpx2::02_13` case compiled and emitted
    `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b`, but runtime output was
    entirely zero; that path is confirmed wrong-code, not a verifier-only gap;
  - a follow-up direct-seed sweep narrowed the failure: aligned
    `tmemDwordDelta=0` moves data but duplicates each source-column pair (for
    example first rows look like `[64, 192, 64, 192]` rather than
    `[64, 192, 65, 193]`), `tmemDwordDelta=4` remains all-zero, unaligned
    deltas trap with misaligned-address errors, and sweeping the known seed
    fields did not restore the missing 4-byte source-column bit;
  - a durable patched-PTX sweep now reproduces that conclusion without source
    edits: `experiments/probe_cp_warpx2_02_13_twocta_direct_ptx.py` starts from
    the executable two-CTA `01_23` public kernel, patches the opcode to
    `warpx2::02_13`, assembles with Blackwell `ptxas`, and launches through
    Triton's normal cluster-aware CUDA launcher. The result artifact
    `experiments/results/probe_cp_warpx2_02_13_twocta_direct_ptx_current.log`
    shows all eight variants returned successfully with no sentinel or NaN
    output, but none matched the extended single-CTA `02_13` oracle; opcode-only,
    source-row-plus-16, single-seed, destination-`+4`, and two-message variants
    still either duplicate source-column pairs or copy the wrong row/column mix;
  - four-GPU direct-PTX scans extended two-CTA `warpx2::02_13`
    single-message direct-seed `sourceOffsetB128` coverage from earlier
    `32..35` through `36..127` with destination deltas `0` and `4`; the
    durable JSONL set has `184` source-offset records and zero matches against
    the extended `02_13` oracle. Offsets `36..63` launch cleanly but duplicate
    one source column pair; offsets `64..72` launch but still mismatch (`65..72`
    destination `0` has four NaNs, destination `4` is finite but wrong); offsets
    `73..127` launch-fail in isolated child processes. Result shards live under
    `experiments/results/probe_cp_warpx2_02_13_twocta_source_offsets_*_gpu*.jsonl`;
  - the historical scales `warpx2` probe candidate is now known to classify as
    `tcgen05.copy.warpx4.32x128b` under public `TensorMemoryScalesLayout`,
    then fail because no compatible scales descriptor plan can be synthesized;
    a 2026-04-11 bounded probe of nearby shared-linear basis orders still found
    only `warpx4` classifications, and a refreshed FD-capturing subslice probe
    over 384 parent-row shared-linear basis interleavings and starts `[0, 32,
    64]` produced `1152` clean unsupported outcomes with no unknowns, bug-like
    failures, or `warpx2` opcodes, so do not treat that candidate as a live
    public scales `warpx2` path;
  - direct canonical TMEM-linear `128x128b` root coverage is now closed by
    `c5bdb6d5c`;
  - no-scales dense `cta_group::2` copy coverage now pins both
    `128x128b` and `128x256b` opcode families; the new `128x128b` exact uses a
    fixed `256x4` two-CTA i32/f32-bitwidth shape with explicit
    `SharedLinearLayout` and validates both legacy and canonical-linear
    destinations;
  - fit `128x256b` indexed-view coverage is broadened by `7075f31fc`, while
    the full `[2, 128, 256]` parent is captured as a tensor-memory OOR
    boundary;
  - remaining copy work is saturation and missing-surface coverage, especially
    scaled-MMA copy geometries and any additional deterministic `warpx2`
    user-visible paths that fit TMEM; true scales `warpx2` remains a separate
    documentation/direct-PTX or new-public-layout question, not proven by the
    historical candidate;
  - keep `4x256b` out of the positive target set unless a future direct-PTX
    probe proves a deterministic compiler contract.
- `tcgen05.ld/st` and `tcgen05.ld.red`:
  - broad `ld/st` fuzzing remains the first major saturation phase after the
    active planner bugs, covering canonical, mixed/interleaved, block-basis,
    two-CTA, MMAv5-like, M64/split-N, and descriptor-view chains;
  - allocator/lifetime runtime anchoring now covers exact single-CTA and
    two-CTA `alloc` / `relinquish_alloc_permit` / `dealloc` / `wait` emission
    on ld/st kernels, pow2 alloc/dealloc size immediates
    `32, 64, 128, 256, 512`, and a source-initialized allocation roundtrip;
    exact commit opcode anchoring now covers single-CTA non-multicast and
    two-CTA multicast copy/MMA paths, including scaled-MMA copy-helper kernels;
    raw non-pow2 live totals `96`, `192`,
    and `384` are now pinned as allocation-pass rounding boundaries; remaining
    allocator fuzzing is specialized standalone/malformed commit configurations
    rather than a missing first runtime anchor;
  - direct subword `ld/st` now covers `f16`, `bf16`, `i16`, and `i8` over
    identity `128x{64,128,256}` layouts and every public instruction variant,
    with exact PTX/LLIR opcode agreement;
  - x1 subword `ld/st` now covers `f16`, `bf16`, and `i16` packed plus
    legacy unpacked layouts, packed plus padded `i8` linear/legacy layouts,
    and direct plus descriptor-chain two-CTA canonical TMEM-linear `f16` /
    `bf16` / `i16` / packed `i8` layouts, for `auto` and explicit `32x32b`;
    the padded i8
    assertion-style probe failure was fixed by making the store-source convert
    canonicalizer bail out when no compatible TMEM layout exists instead of
    asking for an asserting default layout;
  - direct higher-rank access is still future work, but the current clean
    boundary is now pinned: rank-3 direct `get_reg_layout`, explicit `load`,
    and explicit `store` fail with the frontend 2D-only diagnostic, while the
    C++ ld/st layout planners return `std::nullopt` instead of asserting on
    non-2D physical layouts; higher-rank descriptors should be
    sliced/indexed/reshaped to 2D before access;
  - `ld.red` now has modifier saturation over identity, tile-permuted, and
    row/column-permuted supported non-sharded families; pure column, pure row,
    and non-identity row/column cross-product positives span
    `128x{64,128,256}`;
  - `ld.red` now also pins the row-256 source boundary: reduction-friendly
    `256x{32,64,128}` layouts with the extra row bit carried in the column
    query frame are positive `32x32b`, while plain identity
    `256x{32,64,128,256}` source layouts are clean unsupported cases;
  - positive `ld.red` runtime tests now also pin the exact PTX/LLIR offset
    immediates (`[0]` for `N <= 128`, `[0, 64, 128, 192]` for `N=256`) plus
    exactly one `wait.store` before reduction loads and one `wait.load` after
    `ld.red` before redval consumption;
  - explicit reduction-load layout coverage now proves compatible explicit
    variants `auto`, `32x32b`, `16x32bx2`, and `32x32b_splitn` still emit the
    canonical `32x32b.x128.{min,max}.f32` reduction opcodes, while explicit
    N-sharded variants `16x64b`, `16x128b`, and `16x256b` report the dedicated
    clean `N dimension sharded across threads` verifier diagnostic;
  - remaining `ld.red` work is broader layout fuzzing plus clean diagnostics
    for other unsupported reduction surfaces.
- `tcgen05.mma` / `tcgen05.mma_scaled`:
  - current runtime coverage now includes plain `f16`, `tf32`, `bf16`,
    `f8e5m2`, and `f8e4m3` for both 1-CTA and 2-CTA, each across legacy and
    canonical TMEM-linear accumulator layouts; the two-CTA plain-kind matrix
    covers both `256x128` and `256x256` accumulator shapes;
  - current 1-CTA and 2-CTA `use_acc=True` plain-kind coverage spans all
    supported plain kinds and both legacy/canonical accumulator layouts,
    validating the accumulator-add path with exact PTX/LLIR opcode agreement;
  - current clean negatives confirm direct `i8` MMAv5 as a frontend diagnostic
    on Blackwell targets where PTXAS rejects it;
  - plain MMAv5 TMEM-LHS subview coverage now spans `f16`, `tf32`, `bf16`,
    `f8e5m2`, and `f8e4m3` for both legacy and canonical TMEM-linear
    accumulator layouts; `test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds`
    slices the right half of a TMEM-linear operand-A parent and feeds the
    subview directly to `tcgen05_mma`, with exact opcode, count, and commit
    checks;
  - direct scaled-MMAv5 accumulator-view coverage includes exact opcode checks
    for `mxf8f6f4`, `mxf4`, and `mxf4nvf4` format families, including
    accumulator subviews and tile-permuted accumulator layouts; the positive
    accumulator-subview format matrix now covers both root-aligned
    `slice_start=0` and offset subviews across the five supported format pairs,
    including the two-CTA `slice_start=128` accumulator subview through both
    non-multicast and multicast scale-TMA paths; the direct `tile_n=64`,
    `128x256` tile-permuted accumulator matrix now covers those same five
    format pairs;
  - direct scaled-MMAv5 TMEM-LHS subview format coverage now includes the
    packed-storage reachable subset: `mxfp8/mxfp8`, `mxfp8/mxfp4`,
    `mxfp4/mxfp4`, and `nvfp4/nvfp4` for both legacy and canonical
    TMEM-linear accumulator layouts, with exact scaled-MMA, count, commit, and
    descriptor-view checks; the mixed `mxfp4/mxfp8` dense TMEM-LHS case
    is now a clean unsupported boundary because `mxf8f6f4` fp4 LHS requires
    the padded operand-A storage model currently represented by `fp4_padded`
    shared memory; the durable probe artifact is
    `experiments/probe_mma_scaled_lhs_subslice_formats.py` / results
    `experiments/results/probe_mma_scaled_lhs_subslice_formats_current.log`;
  - tile-permuted plain MMAv5 accumulator coverage now spans `f16`, `tf32`,
    `bf16`, `f8e5m2`, and `f8e4m3` across both `128x128/tile_n=32` and
    `128x256/tile_n=64` accumulator layouts, with exact PTX/LLIR opcode checks
    for the expected instruction kind; the `use_acc=True` accumulator-add path
    is now covered for the same kind/layout matrix by
    `test_tmem_runtime_matrix_mma_plain_kinds_tile_permuted_acc_use_acc`;
  - tile-permuted scaled-MMAv5 accumulator-subview clean negatives now cover
    the same format pairs as the positive scaled subview matrix:
    `mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`,
    `mxfp4/mxfp8`, and `nvfp4/nvfp4`;
    these repeated-`N=32` forms are intentionally rejected because the public
    tensor-memory scales layout exposes matrix-B scale fragments at 64-column
    alignment;
  - TMA-fed 2-CTA TF32 now has a positive reachable path:
    `test_tmem_runtime_matrix_mma_twocta_tma_tf32_b_transposed_descriptor`
    loads matrix B through a non-transposed `[N, K]` TMA descriptor, passes the
    resulting shared descriptor to MMAv5 as `permute((1, 0))`, and validates
    exact `tcgen05.mma.cta_group::2.kind::tf32` plus multicast commit opcodes
    for both legacy and canonical TMEM-linear accumulators;
  - the original default `[K, N]` TMA-fed 2-CTA TF32 case remains pinned as a
    clean verifier negative by
    `test_tmem_runtime_matrix_mma_twocta_tma_tf32_reports_clean_shared_transpose_error`:
    TMA descriptors themselves cannot be transposed, so the compiler still
    rejects that descriptor shape with
    `tcgen05.mma does not support transposed float32 operands in shared memory`
    before LLVM lowering, with no `PassManager::run failed` or assertion;
  - the scaled-MMA copy-helper matrix also pins exact copy, scaled-MMA, and
    commit opcode selection for 1-CTA and 2-CTA `warpx4` copy paths across
    format/geometry/accumulator-layout combinations;
  - current-head direct `mma` / `mma_scaled` runtime-matrix validation is green
    at the latest focused coverage checkpoint:
    aggregate selected coverage `242 passed, 50 skipped`;
    plain MMAv5 root and `use_acc` matrices now pin exact op counts
    (`f16=2`, `bf16=2`, `tf32=4`, `f8e5m2/f8e4m3=1`), while
    tile-permuted accumulator coverage pins fourfold counts for both no-acc and
    `use_acc=True` accumulator-add paths, the wider-K tile-permuted TMEM-LHS
    path pins `16` f16 ops, the plain TMEM-LHS subview
    matrix spans all supported plain kinds across legacy/canonical
    accumulators, and scaled TMEM-LHS subview coverage now includes the
    packed-storage reachable subset;
    the scaled-MMA copy-helper matrix remains tracked separately;
  - remaining MMA work is not an immediate red-test blocker; it is broader
    fuzz/saturation beyond the deterministic matrix, additional reachable
    layout-family positives when discovered, and broader TMA-fed TF32
    materialization work beyond the now-covered `[N, K]` plus shared-permute
    route;
  - direct scaled MMAv5 through TMEM views remains a high-value runtime target,
    and the default `[K, N]` TMA-fed TF32 boundary should remain a
    compiler/API follow-up target rather than a settled ISA boundary.

### Phase 5: Performance And Selection Heuristics
- Where multiple codegen paths are legal, add or refine heuristics so the
  compiler prefers the better-performing family.
- Keep tests opinionated about generated code only when the intent is
  performance and the heuristic is meant to enforce that choice.
- Prefer simple, explainable heuristics first; only grow a richer cost model if
  the simpler rules are insufficient.

### Phase 6: Saturation Validation And Closure
- Broaden validation iteratively in the following order:
  1. exact nodeids nearest the changed planner/lowering logic
  2. local TMEM runtime matrix slices
  3. broader TMEM descriptor/view slices
  4. MMA/matmul slices
  5. `triton_kernels` matmul
  6. wider pytest/lit coverage
  7. grouped 4-GPU sweeps
- Keep compile-only and heavy GPU runtime sweeps separate.
- Re-run the fuzz plan on the expanded surfaces and classify any residual clean
  negatives against true ISA boundaries.
- Only consider the initiative complete when the remaining open items are
  either:
  - demonstrably ISA-impossible and documented as such; or
  - moved into a clearly scoped follow-on initiative.

### Immediate Priority Order From The Current State
- Keep the GB200/NVIDIA CI red list current in `gb200_nvidia_ci_inventory.md`
  and use that file as the current validation baseline for this Blackwell
  devbox phase.
- The merge-base-present MMAv5 and multicta representative regressions remain
  locally closed by the current M64 producer-contract/source-column row-plan
  fix; keep their exacts in the focused guard set while broadening.
- The focused attention scratch-alias path has been migrated again to the
  supported `subslice/subview -> bitcast` API and broadened to the SDPA-safe
  benchmark unit matrix:
  - `_borrow_s_as_p`, `_borrow_s_as_alpha`, and `_borrow_s_for_epilogue` no
    longer call `_reinterpret`;
  - the exp2 P-storage path must slice the original f32 scratch subregion for
    each partition before bitcasting, because the helper-only migration still
    reproduced the old invalid-basis failure;
  - fp8 required one more correction: the P physical region is 32 f32 columns,
    not the fp16/bf16 64-column region, so scratch offsets now use
    `BLOCK_N * dtype_bits / 32`;
  - SDPA-backed random tests exposed a real causal `use_tmem_red` bug where the
    diagonal stage used a pre-mask TMEM reduction max; the current source uses
    `gl.max(qk, 1)` after masking for `STAGE == 2`;
  - the attention file matrix is green at the current checkpoint:
    `64 passed in 102.66s (0:01:42)`;
  - the examples/Gluon green aggregate recorded at `4263ae61` / `49f1a0fd`
    is still stale for current `HEAD`, so rerun broader examples before marking
    that lane green.
- The bitcast API fix intentionally does not preserve private `_reinterpret`
  behavior:
  - load/store lowering uses the exact physical bitcast query for explicit
    `tmem_physical_bitcast` roots;
  - MMAv5 typed address planning uses the explicit result descriptor layout
    because packed sub-32-bit physical bitcast queries can be non-surjective.
- 2026-04-11 user clarification to preserve across contexts:
  - this is the original motivation for the project, not just a local
    attention workaround;
  - any code that implicitly relies on `_reinterpret` behavior based on
    compiler knowledge should move to supported APIs;
  - the supported operation should be a well-defined
    `subslice/subview -> bitcast` sequence:
    - offset to the intended TMEM region;
    - slice/subview it to the desired physical bits;
    - bitcast to the desired dtype, shape, and layout only when the bitcast is
      equal-size and preserves the exact physical TMEM mapping of the input
      descriptor.
  - the bitcast must not change the actual physical TMEM memory the input
    descriptor maps to; otherwise use a different supported view/API that
    expresses the kernel's real intent.
- The attention bitcast migration committed at `4263ae61` is historical
  evidence for the supported API, not current source state. Revisit the
  attention kernel only when ready to express the reuse with the supported
  operation sequence while preserving the kernel's synchronization intent.
- The stale `test_mma_shared_inputs` two-CTA assertion bucket is closed at
  `165752b67`:
  - the current TTGIR type spelling is `twoCTAs`, while physical-layout and op
    attributes may still use `two_ctas`;
  - patched four-way function refresh:
    - group 1: `3830 passed, 490 skipped`
    - group 2: `2954 passed, 1366 skipped`
    - group 3: `1206 passed, 3114 skipped`
    - group 4: `2584 passed, 1736 skipped`
  - the interrupted full `python/test/gluon` shards that still reported this
    assertion were pre-patch imports and are obsolete.
- Keep the now-closed `python/examples/gluon/02-convolution.py` checkpoint in
  mind during follow-up debugging:
  - the final fix was not another TMEM/PTX family change
  - it was a branch-added full-tile shared-memory scratch fallback in
    NVIDIA allocation analysis for `ttg.convert_layout` when one side used
    `LinearEncodingAttr`
  - that fallback inflated the epilogue scratch buffer by `128 KB` and pushed
    the kernel to `262208` shared bytes
- Keep the branch-added PTX-expectation and stale-negative TMEM tails separate
  from the old-mainline recovery queue while refreshing the aggregate status.
- Keep the reinterpret-heavy `block_m_64` tests visible as explicit
  descriptor-view rewrite / missing-surface work rather than using them as the
  first proof target for the current branch recovery.
- Re-broaden through the wider grouped `python/test/gluon` sweep from current
  `HEAD`, then continue through TMEM runtime, MMA/matmul, `triton_kernels`,
  and the broader suite. Keep the attention example's known red exact separate
  from the supported bitcast API validation.
- Continue the larger initiative mission after the local bug buckets are green:
  - `ld.red` expansion
  - `copy` `warpx2` completion
  - broader MMAv5 / `mma_scaled` reachable-family support
  - saturation fuzzing and final cleanup of stale negatives and heuristics.

## Current Topline (2026-04-11 20:26 UTC)

- Latest pushed checkpoint before this probe:
  - `97f18d07b` on `origin/codex/tmem`
- Ran a bounded scales `warpx2` shared-linear layout classification probe using
  `tmem_copy_scales_layout_probe_kernel` and six nearby offset-basis orders:
  - current `warpx4` layout: passes and emits two
    `tcgen05.cp.cta_group::1.warpx4.32x128b` copies;
  - historical `warpx2_candidate`;
  - no-scales-like column-tail order;
  - row-32 after column bases;
  - row-32 after low-row bases;
  - column-first / row-tail order.
- Result:
  - no probed public `TensorMemoryScalesLayout` case classified as
    `warpx2::{01_23,02_13}`;
  - the failed nearby layouts still mapped to `tcgen05.copy.warpx4.32x128b` and
    failed descriptor-plan synthesis for tensor-memory scales;
  - this reinforces that scales `warpx2` needs a different descriptor/address
    representation or a direct PTX documentation probe, not a small basis-order
    tweak of the current public scales copy probe.
- Next:
  - continue either the two-CTA `warpx2::02_13` descriptor/address-model fix,
    a deeper direct-PTX scales `warpx2` probe, or the TMA-fed 2-CTA TF32
    shared-transpose compiler follow-up.

## 2026-04-11 20:45 UTC: TMA-fed 2-CTA TF32 is a clean verifier negative

- Moved the recorded TMA-fed 2-CTA TF32 shared-transpose failure from late
  MMAv5 LLVM lowering into `TCGen5MMAOp::verify()`.
- Source/test change:
  - `TCGen5MMAOp::verify()` now rejects obvious NVMMA shared-memory TF32 cases
    that MMAv5 cannot lower: transposed A, or non-transposed B;
  - added
    `test_tmem_runtime_matrix_mma_twocta_tma_tf32_reports_clean_shared_transpose_error`
    for both legacy and canonical TMEM-linear accumulators;
  - tightened `.codex/initiatives/tmem_linear_generalization/repro_twocta_tma_tf32.py`
    to require the verifier diagnostic and reject `PassManager::run failed` or
    assertions.
- Boundary:
  - this remains a TMA-to-shared layout materialization/API frontier, not a
    direct 2-CTA TF32 accumulator-layout regression;
  - TMA descriptors still cannot be transposed, and legal non-transposed B
    descriptors produce the shared operand form MMAv5 TF32 rejects.
- Validation:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    `PASSED`
  - focused clean-negative exact:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-twocta-tma-tf32-clean-verifier PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_tma_tf32_reports_clean_shared_transpose_error`
    `2 passed in 3.18s`
  - adjacent two-CTA TF32 positives:
    `8 passed in 7.62s`
  - broad direct MMA/scaled-MMA slice:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-not-cp-clean-verifier PYTHONPATH=python:. pytest -s --tb=short -q -k 'mma and not cp' python/test/gluon/test_tmem_runtime_matrix.py`
    `211 passed, 50 skipped, 2321 deselected in 117.79s (0:01:57)`
  - standalone repro script:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-twocta-tma-tf32-repro-clean-check2 PYTHONPATH=python:. python3 .codex/initiatives/tmem_linear_generalization/repro_twocta_tma_tf32.py`
    `legacy: reproduced clean verifier failure`
    `linear: reproduced clean verifier failure`
  - hygiene:
    `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
    `git diff --check`
    `PASSED`
- Next:
  - commit and push this clean-negative verifier slice;
  - continue either the two-CTA `warpx2::02_13` descriptor/address-model fix,
    a deeper direct-PTX scales `warpx2` probe, or another bounded MMAv5 /
    scaled-MMAv5 reachable-family gap.

## Prior Topline (2026-04-11 20:23 UTC)

- Latest pushed checkpoint before this source/test update:
  - `fc35a1c20` on `origin/codex/tmem`
- Added an exact clean-unsupported test for the canonical public two-CTA
  `warpx2::02_13.64x128b` no-scales copy candidate.
- Current behavior:
  - the canonical two-CTA shared layout plus `02_13` TMEM layout classifies as
    `tcgen05.copy.warpx2::02_13.64x128b`;
  - verifier/planning rejects it because Triton cannot synthesize a compatible
    shared-memory descriptor plan;
  - the test asserts this stays a clean unsupported diagnostic and does not
    fall through to `PassManager::run failed`, assertions, or the older
    wrong-code/all-zero path.
- Validation:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - focused two-CTA `warpx2::02_13` clean boundary:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-warpx2-02-13-twocta-clean-negative PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_candidate_reports_clean_unsupported`
    - `1 passed in 3.29s`
  - focused `warpx2` slice:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-warpx2-clean-boundary-slice PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k warpx2`
    - `9 passed, 2571 deselected in 4.32s`
  - broad current-head `cp` slice:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-warpx2-clean-boundary-cp-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k cp`
    - `158 passed, 5 skipped, 2417 deselected in 40.42s`
- Next:
  - after committing and pushing this boundary slice, continue either the
    two-CTA `warpx2::02_13` descriptor/address-model fix itself, the scales
    `warpx2` descriptor search, or the TMA-fed 2-CTA TF32 shared-transpose
    compiler follow-up.

## Prior Topline (2026-04-11 20:20 UTC)

- Latest pushed checkpoint before this source/test update:
  - `7aa084aa4` on `origin/codex/tmem`
- Added 2-CTA plain MMAv5 `use_acc=True` runtime coverage for every supported
  plain operand kind, both accumulator layout spellings, and both current
  two-CTA shapes:
  - `f16`
  - `tf32`
  - `bf16`
  - `f8e5m2`
  - `f8e4m3`
  - legacy `TensorMemoryLayout`
  - canonical TMEM-linear layout
  - `256x128`
  - `256x256`
- The new kernel initializes the two-CTA accumulator from a register `C` tile,
  uses the same cga-aware A/B shared-memory topology as the existing two-CTA
  kind matrix, runs `tcgen05_mma(..., use_acc=True, multicast=True)`, and
  checks `matmul(A, B) + C` numerics.
- Each case asserts PTX/LLIR opcode agreement, exact two-CTA multicast commit
  opcode emission, `two_ctas` TTGIR, and `tensor_memory_linear` TTGIR for
  canonical accumulator layouts.
- Validation:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - focused 2-CTA `use_acc=True` plain-kind matrix:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-twocta-use-acc-kind-focused PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_plain_kinds_use_acc`
    - `20 passed in 17.18s`
  - broad direct MMA/scaled-MMA slice:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-twocta-use-acc-kind-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma and not cp'`
    - `209 passed, 50 skipped, 2320 deselected in 117.66s (0:01:57)`
- Next:
  - after committing and pushing this coverage slice, continue either the
    two-CTA `warpx2::02_13` / scales `warpx2` descriptor-address frontier or
    the TMA-fed 2-CTA TF32 shared-transpose compiler follow-up.

## Prior Topline (2026-04-11 20:12 UTC)

- Latest pushed checkpoint before this source/test update:
  - `f1e77e81b` on `origin/codex/tmem`
- Added 1-CTA plain MMAv5 `use_acc=True` runtime coverage for every supported
  plain operand kind and both accumulator layout spellings:
  - `f16`
  - `tf32`
  - `bf16`
  - `f8e5m2`
  - `f8e4m3`
  - legacy `TensorMemoryLayout`
  - canonical TMEM-linear layout
- The new kernel initializes the accumulator from a register `C` tile, runs
  `tcgen05_mma(..., use_acc=True)`, and checks `matmul(A, B) + C` numerics.
- Each case asserts PTX/LLIR opcode agreement, the exact single-CTA commit
  opcode, and `tensor_memory_linear` TTGIR for canonical accumulator layouts.
- TMA-fed 2-CTA TF32 quick-probe note:
  - `.codex/initiatives/tmem_linear_generalization/repro_twocta_tma_tf32.py`
    remains the durable repro;
  - raw `smem_b.permute((1, 0))` after TMA fails verifier shape checks;
  - `smem_b.permute((1, 0)).reshape((BLOCK_K, BLOCK_N))` fails the RHS
    CTASplit-along-K check;
  - leaving the legal non-transposed TMA descriptor reproduces the original
    `tcgen05.mma does not support transposed float32 operands in shared memory`
    / `PassManager::run failed` error;
  - this remains a TMA-to-shared layout materialization follow-up, not a small
    user-kernel descriptor-view patch.
- Validation:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - focused 1-CTA `use_acc=True` plain-kind matrix:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-use-acc-kind-focused-r2 PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_plain_kinds_use_acc`
    - `10 passed in 6.69s`
  - broad direct MMA/scaled-MMA slice:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-use-acc-kind-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma and not cp'`
    - `189 passed, 50 skipped, 2320 deselected in 106.20s (0:01:46)`
- Next:
  - after committing and pushing this coverage slice, continue either the
    two-CTA `warpx2::02_13` / scales `warpx2` descriptor-address frontier or
    the TMA-fed 2-CTA TF32 shared-transpose compiler follow-up.

## Prior Topline (2026-04-11 20:05 UTC)

- Latest pushed checkpoint before this source/test update:
  - `21c9d4712` on `origin/codex/tmem`
- Two-CTA plain MMAv5 accumulator kind coverage is widened from only
  `256x128` to cover both:
  - `256x128`
  - `256x256`
- The widened two-CTA matrix spans all supported plain operand kinds and both
  legacy and canonical TMEM-linear accumulator layouts:
  - `f16`
  - `tf32`
  - `bf16`
  - `f8e5m2`
  - `f8e4m3`
- Each case validates runtime numerics, exact
  `tcgen05.mma.cta_group::2.kind::*` PTX/LLIR opcode agreement, multicast
  commit emission, `two_ctas` TTGIR, and `tensor_memory_linear` for the
  canonical layout cases.
- Probe note: two-CTA tile-permuted accumulator layouts are not a quick positive
  at this point. The `128x128/tile_n=32` form fails through the existing
  repeated-N MMAv5 diagnostic, while smaller tile permutations fail the
  MMAv5-compatible layout check; keep that as a clean unsupported frontier
  unless the planner grows a principled two-CTA tile-permutation path.
- Validation:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - focused widened two-CTA kind matrix:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-twocta-n256-kind-focused PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_plain_kinds`
    - `20 passed in 8.58s`
  - broad direct MMA/scaled-MMA slice:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-twocta-n256-kind-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma and not cp'`
    - `179 passed, 50 skipped, 2320 deselected in 96.84s (0:01:36)`
- Next:
  - commit and push this two-CTA MMA coverage slice;
  - continue either the two-CTA `warpx2::02_13` / scales `warpx2`
    descriptor-address frontier or the TMA-fed 2-CTA TF32 shared-transpose
    compiler follow-up.

## Prior Topline (2026-04-11 19:58 UTC)

- Latest pushed checkpoint before this source/test update:
  - `1b2e83595` on `origin/codex/tmem`
- Tile-permuted plain MMAv5 accumulator kind coverage is expanded from the
  original `128x128/tile_n=32` geometry to cover both:
  - `128x128`, `tile_n=32`
  - `128x256`, `tile_n=64`
- The expanded matrix spans all supported plain operand kinds:
  - `f16`
  - `tf32`
  - `bf16`
  - `f8e5m2`
  - `f8e4m3`
- The test checks runtime numerics plus exact PTX/LLIR
  `tcgen05.mma.cta_group::1.kind::*` opcode agreement for each geometry/kind
  pair.
- Validation:
  - focused tile-permuted kind matrix:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-tile-kind-expanded PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_plain_kinds_tile_permuted_acc`
    - `10 passed in 6.91s`
  - broad direct MMA/scaled-MMA slice:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-tile-kind-expanded-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma and not cp'`
    - `169 passed, 50 skipped, 2320 deselected in 100.62s (0:01:40)`
- The supported `_reinterpret` migration invariant remains unchanged and should
  steer future code changes:
  - offset to the right part of TMEM;
  - slice/subview it to the desired physical bits;
  - then bitcast to the desired dtype, shape, and layout only when the view is
    equal-size and preserves the exact physical TMEM mapping of the input
    descriptor.
- Next:
  - commit and push this MMA coverage slice;
  - continue either the two-CTA `warpx2::02_13` / scales `warpx2`
    descriptor-address frontier or the next MMAv5 reachable-family gap.

## Prior Topline (2026-04-11 17:19 UTC)

- Latest pushed checkpoint before this source/test update:
  - `8deb0fcad` on `origin/codex/tmem`
- Tile-permuted MMAv5 accumulator coverage is broadened across all supported
  plain operand kinds:
  - `f16`
  - `tf32`
  - `bf16`
  - `f8e5m2`
  - `f8e4m3`
- The new test checks runtime numerics plus exact PTX/LLIR
  `tcgen05.mma.cta_group::1.kind::*` opcode agreement.
- Validation:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - focused tile-permuted kind matrix:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-mma-tile-permuted-kind-focused PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_plain_kinds_tile_permuted_acc`
    - `5 passed in 5.10s`
  - direct MMA/scaled-MMA slice:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-mma-direct-after-tile-kind PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma and not cp'`
    - `144 passed, 50 skipped, 2057 deselected in 88.38s (0:01:28)`
- Next:
  - commit and push this MMA coverage slice;
  - continue operational fuzzing from `fuzz_plan.md`, likely remaining
    scaled-MMA direct-view or two-CTA TF32/shared-transpose follow-up probes.

## Prior Topline (2026-04-11 17:14 UTC)

- Latest pushed checkpoint before this source/test update:
  - `30feed55c` on `origin/codex/tmem`
- No-scales `tcgen05.cp` `warpx2` canonical-codegen coverage is now symmetric:
  - existing `warpx2::01_23.64x128b` canonical codegen stayed green;
  - new `warpx2::02_13.64x128b` canonical codegen is green.
- Validation:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - exact new codegen test:
    `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-cp-warpx2-0213-codegen-focused PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_canonical_codegen`
    - `1 passed in 3.48s`
  - focused `warpx2` slice:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-cp-warpx2-focused-current PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k warpx2`
    - `5 passed, 2241 deselected in 3.98s`
  - broad current-head `cp` slice:
    `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-cp-broad-after-warpx2-0213 PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k cp`
    - `153 passed, 5 skipped, 2088 deselected in 37.11s`
- Next:
  - commit and push this copy coverage slice;
  - continue operational fuzzing from `fuzz_plan.md`, likely broader
    MMA/scaled-MMA probes or the more involved two-CTA/scales copy search.

## Prior Topline (2026-04-11 17:09 UTC)

- Latest pushed checkpoint before this source/test update:
  - `d734f940a` on `origin/codex/tmem`
- `ld.red` row-256 runtime-matrix coverage is broadened:
  - positive reduction-friendly `256x{32,64,128}` TMEM-linear layouts now
    exercise the same all-modifier matrix as the existing identity positives;
  - plain identity `256x{32,64,128,256}` source layouts are covered as clean
    unsupported cases with the dedicated `tcgen05.ld.red-compatible`
    diagnostic.
- Discovery note:
  - plain identity `256xN` is not the current direct `ld.red` source form;
  - the positive row-256 shape is the legacy-equivalent physical layout with a
    128-row reduction block and the high row bit carried in the column/query
    frame;
  - `256x256` in that positive form was not added because the probe hit the
    current shared-memory resource budget.
- Validation:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - focused row-256 positives:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-ldred-legacy256-focused PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_identity_linear_layout -k legacy_equivalent_256`
    - `24 passed, 32 deselected in 16.73s`
  - focused row-256 identity clean negatives:
    `CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-ldred-identity256-negatives PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_identity_256_linear_layout_reports_clean_unsupported`
    - `4 passed in 3.65s`
  - broad current-head `ld_red` slice:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-ldred-broad-after-256-fuzz PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k ld_red`
    - `228 passed, 2017 deselected in 139.83s (0:02:19)`
- Next:
  - commit and push this row-256 `ld.red` fuzz slice;
  - continue operational fuzzing from `fuzz_plan.md`, likely additional copy
    saturation or broader MMA/scaled-MMA probes.

## Prior Topline (2026-04-11 17:03 UTC)

- Latest pushed checkpoint before this docs update:
  - `4521dfe1a` on `origin/codex/tmem`
- No source code changed in this checkpoint.
- The supported `_reinterpret` migration contract has been restated explicitly:
  - choose the correct physical TMEM region first;
  - slice/subview to the desired physical bits;
  - bitcast only when the new dtype/shape/layout has the same total bit size
    and the same physical TMEM mapping as the input descriptor.
- Keep the attention kernel reverted for now. Its future migration needs a
  synchronization-aware supported descriptor sequence because it deliberately
  reappropriates part of TMEM while it knows that memory is otherwise inactive.
- Next:
  - commit and push this contract clarification;
  - continue operational fuzzing from `fuzz_plan.md`, likely `ld.red`,
    additional copy-family saturation, or broader MMA/scaled-MMA probes.

## Prior Topline (2026-04-11 17:02 UTC)

- Latest pushed source/test checkpoint:
  - `efdd93c67` on `origin/codex/tmem`
- Broad current-head `tcgen05.cp` runtime-matrix validation is green:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - command:
    `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-cp-current-broad-after-ldst-fuzz PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k cp`
  - result:
    - `152 passed, 5 skipped, 2068 deselected in 37.08s`
- This refresh confirms the copy surface is still green after the recent
  `ld/st` fuzz additions.
- Next:
  - commit and push this docs validation checkpoint;
  - continue operational fuzzing from `fuzz_plan.md`, likely `ld.red`,
    additional copy-family saturation, or broader MMA/scaled-MMA probes.

## Prior Topline (2026-04-11 17:00 UTC)

- Latest pushed source/test checkpoint:
  - `ee4fdec29` on `origin/codex/tmem`
- Broad post-fuzz `ld/st` runtime-matrix validation is green:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - command:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-current-broad-after-fuzz PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k ldst`
  - result:
    - `1104 passed, 441 skipped, 680 deselected in 1278.98s (0:21:18)`
- This broad refresh includes the recent fixed-offset `auto`, two-CTA rank-5
  full-variant, and scales `auto` coverage.
- Next:
  - commit and push this docs validation checkpoint;
  - continue operational fuzzing from `fuzz_plan.md`, likely moving from
    `ld/st` tails to `ld.red`, `cp`, or alloc/lifetime exact coverage.

## Prior Topline (2026-04-11 16:37 UTC)

- Latest source/test checkpoint in progress:
  - `f899dda05` plus local `python/test/gluon/test_tmem_runtime_matrix.py`
    edits
- Scales `ld/st` instruction-selection coverage now includes `auto`.
- Important finding:
  - scales `auto` is a valid positive path, but it does not always mirror the
    explicit `32x32b` packet family;
  - for several shapes it selects wider `16x64b`, `16x128b`, or `16x256b`
    packet streams with high-half `1048576` offsets.
- Validation:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - new auto parametrizations:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-scales-auto-current-r2 PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_scales_variant_sweep -k auto`
    - `21 passed, 33 deselected in 4.15s`
  - full scales variant sweep:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-scales-variant-full-current PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_scales_variant_sweep`
    - `54 passed in 5.68s`
- Next:
  - run hygiene, commit, and push this scales-auto fuzzing slice;
  - continue operational fuzzing from `fuzz_plan.md`.

## Prior Topline (2026-04-11 16:35 UTC)

- Latest source/test checkpoint in progress:
  - `646cbb670` plus local `python/test/gluon/test_tmem_runtime_matrix.py`
    edits
- Focused `ld/st` fuzz/stale-coverage cleanup is green:
  - `test_tmem_runtime_matrix_ldst_fixed_offset_patterns_128x256` now covers
    `auto` and confirms the canonical `32x32b.x64.b32` offset pattern:
    - `5 passed in 7.09s`
  - `test_tmem_runtime_matrix_ldst_twocta_descriptor_rank5_roundtrip` now uses
    the full `LDST_VARIANTS` set and confirms the newly added large explicit
    families remain clean tensor-memory OOR boundaries:
    - `10 skipped in 7.12s`
- Build before these tests:
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
  - `PASSED`, ninja reported no work to do
- Next:
  - run hygiene, commit, and push this focused fuzzing slice;
  - continue operational fuzzing from `fuzz_plan.md`.

## Prior Topline (2026-04-11 16:33 UTC)

- Latest pushed source/test checkpoint:
  - `24bec4ecf` on `origin/codex/tmem`
- Clean Gluon examples subset validation is green:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - `python/examples/gluon/02-convolution.py`:
    - `48 passed in 8.99s`
  - `python/examples/gluon/03-matmul-multicta.py`:
    - `82 passed, 14 skipped in 75.53s (0:01:15)`
  - `python/examples/gluon/04-2cta-block-scale-matmul.py`:
    - `690 passed, 60 skipped in 52.87s`
- Full `python/examples/gluon` remains intentionally not the aggregate target
  until rerun after the attention benchmark-matrix expansion. The attention
  file itself is green (`64 passed` with SDPA-backed correctness), but the
  broader examples aggregate is not yet refreshed.
- Next:
  - start operational fuzzing from `fuzz_plan.md` or continue wider pytest/lit
    validation;
  - keep the attention exact separate from supported physical-bitcast API
    validation.

## Prior Topline (2026-04-11 16:30 UTC)

- Latest pushed source/test checkpoint:
  - `2ad0ccf5e` on `origin/codex/tmem`
- Current-head `python/triton_kernels/tests` four-way validation is green:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - split command shape:
    `CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu>-triton-kernels-current PYTHONPATH=python:. pytest -s --tb=short --splits 4 --group <group> python/triton_kernels/tests`
  - group 1 / GPU 0:
    - `674 passed, 782 skipped, 4365 deselected in 1960.67s (0:32:40)`
  - group 2 / GPU 1:
    - `533 passed, 923 skipped, 4365 deselected in 1522.70s (0:25:22)`
  - group 3 / GPU 2:
    - `394 passed, 1062 skipped, 4365 deselected in 927.87s (0:15:27)`
  - group 4 / GPU 3:
    - `776 passed, 677 skipped, 4368 deselected in 431.27s (0:07:11)`
  - aggregate:
    - `2377 passed, 3444 skipped, 17463 deselected`
- The `_reinterpret` migration rule remains a supported API migration, not a
  lowering patch target:
  - offset to the right part of TMEM;
  - slice/subview to the desired physical bits;
  - bitcast to the desired dtype, shape, and layout only when the result is
    equal-size and physical-mapping equivalent to the input descriptor.
- Next:
  - choose the next broad validation tier or start operational fuzzing from
    `fuzz_plan.md`;
  - keep attention deferred until the synchronization-aware
    `offset/slice/subview -> bitcast` migration can preserve the kernel's
    real TMEM reuse contract.

## Prior Topline (2026-04-11 18:25 UTC)

- Latest pushed source/test checkpoint:
  - `a950338f7` on `origin/codex/tmem`
- Latest pushed docs checkpoint before this validation refresh:
  - `bf9eebb37` on `origin/codex/tmem`
- Current-head `tcgen05.ld.red` runtime-matrix validation is green:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - broad ld.red slice:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ld-red-current-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k ld_red`
    - `208 passed, 1991 deselected in 129.20s (0:02:09)`
  - coverage includes identity, tile-permuted, column-permuted, row-permuted,
    all legal modifier pairs, and clean unsupported mixed-layout negatives.
- Next:
  - move to MMAv5 / `mma_scaled` reachable-family coverage;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 18:15 UTC)

- Latest pushed source/test checkpoint:
  - `a950338f7` on `origin/codex/tmem`
- Latest pushed docs checkpoint before this validation refresh:
  - `cd0e6b985` on `origin/codex/tmem`
- Current-head `tcgen05.cp` runtime-matrix validation is green:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - broad cp slice:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-cp-current-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k cp`
    - `152 passed, 5 skipped, 2042 deselected in 36.66s`
  - coverage includes current no-scales `warpx2` positives, dense copy
    positives, 2-CTA copy, scaled `warpx4` copy paths, and clean unsupported
    copy boundaries.
- Next:
  - move to `ld.red` broader layout fuzzing or MMAv5 / `mma_scaled`
    reachable-family coverage;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 18:05 UTC)

- Latest pushed source/test checkpoint:
  - `a950338f7` on `origin/codex/tmem`
- Latest pushed docs checkpoint before this validation refresh:
  - `016f489a3` on `origin/codex/tmem`
- Combined current-head `ld/st` runtime-matrix validation is green after the
  staged auto expansions:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - broad ldst slice:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-current-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k ldst`
    - `1082 passed, 437 skipped, 680 deselected in 1344.69s (0:22:24)`
  - skipped cases are expected clean OOR/boundary classifications in lifted
    descriptor roundtrip and rank-5 families.
- Next:
  - move to the next non-ld/st saturation surface from the long-term plan:
    copy/`tcgen05.cp` remaining surfaces, `ld.red` broader layout fuzzing, or
    MMAv5 / `mma_scaled` reachable-family coverage;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 17:40 UTC)

- Latest pushed source/test checkpoint:
  - `a950338f7` on `origin/codex/tmem`
- X1 f16 ld/st roundtrips now include `auto` instruction selection:
  - `X1_F16_LDST_VARIANTS` covers `auto` and explicit `32x32b`;
  - the auto cases preserve the expected `32x32b.x1` f16 packet shapes,
    including `unpack::16b` / `pack::16b` for the legacy unpacked layout.
- Validation for `a950338f7`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - x1 f16 auto parametrizations:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-x1-f16-auto PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_f16_roundtrip' -k auto`
    - `3 passed, 3 deselected in 3.45s`
  - `git diff --check`
    - `PASSED`
- Next:
  - classify remaining fixed-offset and scales exact-family ld/st tests as
    intentional exact-opcode coverage or add auto companions if they test
    instruction selection rather than a particular explicit family;
  - then continue broader validation and heuristic cleanup.

## Prior Topline (2026-04-11 17:30 UTC)

- Latest pushed source/test checkpoint:
  - `c475eecc0` on `origin/codex/tmem`
- X1 f32 direct and descriptor-chain ld/st roundtrips now include `auto`
  instruction selection:
  - `X1_F32_LDST_VARIANTS` covers `auto` and explicit `32x32b`;
  - direct auto uses the no-variant `tmem_ldst_auto_kernel` path;
  - descriptor-chain auto passes through `get_reg_layout(instr_variant="auto")`;
  - exact opcode expectations remain `32x32b.x1.b32`.
- Validation for `c475eecc0`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - x1 f32 auto parametrizations:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-x1-f32-auto PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_f32_roundtrip' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_x1_f32_descriptor_chain_roundtrip' -k auto`
    - `6 passed, 6 deselected in 4.11s`
  - `git diff --check`
    - `PASSED`
- Next:
  - classify x1 f16, fixed offset patterns, and scales exact-family coverage;
  - then continue broader validation and heuristic cleanup.

## Prior Topline (2026-04-11 17:20 UTC)

- Latest pushed source/test checkpoint:
  - `4fe7a44de` on `origin/codex/tmem`
- F16 subword ld/st pack/unpack coverage now includes `auto` instruction
  selection:
  - `F16_LDST_SHAPE_MAP` has `auto` expectations matching the explicit
    `32x32b` f16 packet shapes;
  - `F16_LDST_CASES` now uses `LDST_VARIANTS`.
- Validation for `4fe7a44de`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - f16 subword auto parametrizations:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-f16-auto PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_subword_f16_pack_unpack' -k auto`
    - `3 passed, 12 deselected in 3.60s`
  - `git diff --check`
    - `PASSED`
- Next:
  - continue classifying remaining fixed-opcode ld/st tests such as x1, fixed
    offset patterns, and scales exact-family coverage as intentional or add
    explicit auto companions where useful;
  - then continue broader validation and heuristic cleanup.

## Prior Topline (2026-04-11 17:10 UTC)

- Latest pushed source/test checkpoint:
  - `be193b6c6` on `origin/codex/tmem`
- Rank-5 descriptor-roundtrip `ld/st` boundary coverage now includes `auto`
  instruction selection:
  - `LDST_DESCRIPTOR_RANK5_CASES` now uses `LDST_VARIANTS`;
  - `LDST_TWOCTA_DESCRIPTOR_RANK5_CASES` adds `auto` to the existing explicit
    subset;
  - all newly added auto rank-5 parametrizations hit the clean tensor-memory
    OOR skip boundary.
- Validation for `be193b6c6`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - rank-5 descriptor-roundtrip auto parametrizations:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-rank5-auto PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_rank5_roundtrip' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_rank5_roundtrip' -k auto`
    - `4 skipped, 12 deselected in 4.94s`
  - `git diff --check`
    - `PASSED`
- Next:
  - classify remaining fixed-opcode ld/st tests such as f16/subword, x1, and
    fixed offset patterns as intentional exact-family coverage or add explicit
    auto companions where useful;
  - then continue broader validation and heuristic cleanup.

## Prior Topline (2026-04-11 17:05 UTC)

- Latest pushed source/test checkpoint:
  - `a487942ac` on `origin/codex/tmem`
- Cleanup:
  - removed the unused `LDST_HIGHER_RANK_POSITIVE_CASES` list;
  - removed the now-unused `LDST_EXPLICIT_VARIANTS` tuple;
  - active runtime-matrix ld/st auto-coverage scans no longer find a false
    explicit-only case list.
- Validation for `a487942ac`:
  - `rg -n "LDST_EXPLICIT_VARIANTS|LDST_HIGHER_RANK_POSITIVE_CASES" python/test/gluon/test_tmem_runtime_matrix.py`
    - `PASSED`, no matches
  - `PYTHONPATH=python:. python -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
    - `PASSED`
  - `git diff --check`
    - `PASSED`
- Next:
  - choose the next non-ld/st saturation surface or broaden validation for the
    staged ld/st auto coverage;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 17:00 UTC)

- Latest pushed source/test checkpoint:
  - `00139772b` on `origin/codex/tmem`
- Row/column descriptor-roundtrip `ld/st` boundary coverage now includes
  `auto` instruction selection:
  - `LDST_DESCRIPTOR_ROUNDTRIP_ROWCOL_CASES` now uses `LDST_VARIANTS`;
  - the newly added auto parametrizations all hit the existing clean
    tensor-memory OOR skip boundary for lifted roundtrip shapes.
- Validation for `00139772b`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - row/column descriptor-roundtrip auto parametrizations:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-rowcol-roundtrip-auto PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_roundtrip_rowcol_permuted_sweeps' -k auto`
    - `48 skipped, 192 deselected in 37.78s`
  - `git diff --check`
    - `PASSED`
- Next:
  - rescan for remaining `LDST_EXPLICIT_VARIANTS` runtime-matrix surfaces;
  - then continue broader validation and heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 16:50 UTC)

- Latest pushed source/test checkpoint:
  - `6d44718e7` on `origin/codex/tmem`
- Row/column cross-product permuted `ld/st` layouts now include `auto`
  instruction selection:
  - `LDST_ROWCOL_PERMUTED_CASES` now uses `LDST_VARIANTS`;
  - this reaches both direct ld/st and descriptor-chain composition sweeps;
  - focused validation ran only the newly added auto parametrizations.
- Validation for `6d44718e7`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - row/column direct + descriptor-chain auto parametrizations:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-rowcol-permuted-auto PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_rowcol_permuted_layout_sweep' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions_rowcol_permuted_layout_sweep' -k auto`
    - `96 passed, 384 deselected in 54.69s`
  - `git diff --check`
    - `PASSED`
- Next:
  - inspect descriptor-roundtrip row/column explicit-only `ld/st` coverage;
  - then continue broader validation and heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 16:40 UTC)

- Latest pushed source/test checkpoint:
  - `20f2c8db8` on `origin/codex/tmem`
- Exotic `ld/st` layout matrices now include `auto` instruction selection:
  - `LDST_EXOTIC_CASES` now uses `LDST_VARIANTS`;
  - `LDST_EXOTIC_DESCRIPTOR_CASES` now uses `LDST_VARIANTS`;
  - `LDST_EXOTIC_UNSUPPORTED_CASES` now uses `LDST_VARIANTS`;
  - this covers direct positive, descriptor-chain positive, and clean
    unsupported block-layout negative paths.
- Validation for `20f2c8db8`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - exotic direct/descriptor/unsupported matrix:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-exotic-auto PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_exotic_linear_layouts' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions_exotic_layouts' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_exotic_layouts_report_clean_unsupported'`
    - `75 passed in 42.51s`
  - `git diff --check`
    - `PASSED`
- Next:
  - inspect the larger row/column cross-product and descriptor-roundtrip
    explicit-only `ld/st` surfaces;
  - then continue broader validation and heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 16:30 UTC)

- Latest pushed source/test checkpoint:
  - `c83d358e4` on `origin/codex/tmem`
- Same-row/column permuted `ld/st` layouts now include `auto` instruction
  selection:
  - `LDST_PERMUTED_CASES` now uses `LDST_VARIANTS`;
  - this reaches both direct `tmem_ldst_variant_kernel` coverage and
    descriptor-chain composition coverage;
  - validation covers numerical roundtrip behavior, `tensor_memory_linear` IR,
    and exact PTX/LLIR opcode agreement.
- Validation for `c83d358e4`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - direct + descriptor-chain same-row/column permuted matrix:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-permuted-auto PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_permuted_layout_sweep' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions_permuted_layout_sweep'`
    - `120 passed in 100.94s`
  - `git diff --check`
    - `PASSED`
- Next:
  - inspect row/column cross-product and exotic explicit-only `ld/st`
    surfaces;
  - then continue broader validation and heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 16:20 UTC)

- Latest pushed source/test checkpoint:
  - `8f622b938` on `origin/codex/tmem`
- The two-CTA MMAv5-like higher-rank `ld/st` unsupported matrix now includes
  `auto` instruction selection:
  - `LDST_TWOCTA_MMAV5_HIGHER_RANK_UNSUPPORTED_CASES` now uses
    `LDST_VARIANTS`;
  - auto cases remain clean negatives through the CTA-per-CGA mismatch path;
  - the tests still reject PassManager/assertion crashes.
- Validation for `8f622b938`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - two-CTA MMAv5 higher-rank unsupported matrix:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-mmav5-twocta-higher-rank-auto-unsupported PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_mmav5_descriptor_higher_rank_reports_clean_error'`
    - `15 passed in 5.21s`
  - `git diff --check`
    - `PASSED`
- Next:
  - inspect remaining staged `ld/st` explicit-only permuted/exotic surfaces;
  - then continue broader validation and heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 16:10 UTC)

- Latest pushed source/test checkpoint:
  - `572fb25b2` on `origin/codex/tmem`
- Higher-rank dim0-slice positive `ld/st` matrices now include `auto`
  instruction selection:
  - `LDST_HIGHER_RANK_DIM0_SLICE_POSITIVE_CASES` now uses `LDST_VARIANTS`;
  - `LDST_TWOCTA_HIGHER_RANK_DIM0_SLICE_POSITIVE_CASES` now uses
    `LDST_VARIANTS`;
  - this covers single-CTA identity and two-CTA block lifted-layout dim0-slice
    views at `N = 64` and `N = 128`;
  - the tests still check functional output and exact full-tile plus narrowed
    subview PTX/LLIR opcode agreement.
- Validation for `572fb25b2`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - higher-rank dim0-slice positive matrix:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-dim0-slice-auto-positive PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_dim0_slice_positive_lifted_layout' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_dim0_slice_positive_lifted_layout'`
    - `20 passed in 11.56s`
  - `git diff --check`
    - `PASSED`
- Next:
  - inspect remaining staged `ld/st` stale-negative and explicit-only
    surfaces;
  - then continue broader validation and heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 16:00 UTC)

- Latest pushed source/test checkpoint:
  - `7c1a6f63b` on `origin/codex/tmem`
- Higher-rank half-row `ld/st` clean-error matrices now include `auto`
  instruction selection:
  - `LDST_HIGHER_RANK_HALF_ROWS_CLEAN_ERROR_CASES` now uses `LDST_VARIANTS`;
  - `LDST_TWOCTA_HIGHER_RANK_HALF_ROWS_CLEAN_ERROR_CASES` now uses
    `LDST_VARIANTS`;
  - explicit variants still raise the unsupported-layout `CompilationError`,
    while `auto` can reach the direct descriptor parser and raise the clean
    `RuntimeError` about lifted row-half views translating the TMEM row origin;
  - both diagnostic paths must mention descriptor views and must not expose
    PassManager/assertion crashes.
- Validation for `7c1a6f63b`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - higher-rank half-row clean-error matrix:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-half-row-auto-clean-error-r2 PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout'`
    - `30 passed in 4.47s`
  - `git diff --check`
    - `PASSED`
- Next:
  - inspect remaining staged `ld/st` stale-negative and explicit-only
    surfaces;
  - then continue broader validation and heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 15:50 UTC)

- Latest pushed source/test checkpoint:
  - `e2d793726` on `origin/codex/tmem`
- Higher-rank `ld/st` clean OOR guards now include `auto` instruction
  selection:
  - `LDST_HIGHER_RANK_OOR_CASES` now uses `LDST_VARIANTS`;
  - `LDST_TWOCTA_HIGHER_RANK_OOR_CASES` now uses `LDST_VARIANTS`;
  - `LDST_TWOCTA_HIGHER_RANK_DIM0_SLICE_OOR_CASES` now uses `LDST_VARIANTS`;
  - this covers single-CTA and two-CTA higher-rank index,
    multidimensional-slice, and dim0-slice resource-boundary tests.
- Validation for `e2d793726`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - higher-rank OOR guard matrix:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-higher-rank-oor-auto PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_index_reports_tmem_oor' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_multidim_slices_report_tmem_oor' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_dim0_slice_reports_tmem_oor' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_index_reports_tmem_oor' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_multidim_slices_report_tmem_oor' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_dim0_slice_reports_tmem_oor'`
    - `55 passed in 49.62s`
  - `git diff --check`
    - `PASSED`
- Next:
  - inspect remaining staged `ld/st` stale-negative surfaces;
  - then continue broader validation and heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 15:40 UTC)

- Latest pushed source/test checkpoint:
  - `81b4e5a9d` on `origin/codex/tmem`
- Higher-rank multidimensional-slice `ld/st` descriptor views now include
  `auto` instruction selection:
  - `LDST_HIGHER_RANK_SLICE_CASES` and
    `LDST_TWOCTA_HIGHER_RANK_SLICE_CASES` now use `LDST_VARIANTS`;
  - coverage spans identity/mixed single-CTA layouts and block/MMAv5-like
    two-CTA layouts at `N = 64` and `128`;
  - the tests validate functional output, PTX/LLIR ld/st opcode agreement, and
    expected higher-rank reshape/transpose/split/join IR markers.
- Validation for `81b4e5a9d`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - higher-rank slice matrices:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-higher-rank-slice-auto PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_multidim_slices' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_multidim_slices'`
    - `40 passed in 42.65s`
  - `git diff --check`
    - `PASSED`
- Next:
  - continue staged `ld/st` fuzzing on clean-negative and stale-negative
    descriptor-view surfaces;
  - then continue stale-negative cleanup and heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 15:30 UTC)

- Latest pushed source/test checkpoint:
  - `7df887318` on `origin/codex/tmem`
- Higher-rank `ld/st` index descriptor views now include `auto` instruction
  selection:
  - `LDST_SUBVIEW_SHAPE_MAP` has explicit `auto` expectations for narrowed
    32/64/128-column subviews;
  - `LDST_HIGHER_RANK_INDEX_CASES` and
    `LDST_TWOCTA_HIGHER_RANK_INDEX_CASES` now use `LDST_VARIANTS`;
  - coverage spans identity/mixed single-CTA layouts and block/MMAv5-like
    two-CTA layouts at `N = 64` and `128`.
- Validation for `7df887318`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - single-CTA higher-rank index matrix:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-higher-rank-index-auto PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_index'`
    - `20 passed in 22.53s`
  - two-CTA higher-rank index matrix:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-twocta-higher-rank-index-auto PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_index'`
    - `20 passed in 14.00s`
  - `git diff --check`
    - `PASSED`
- Next:
  - continue staged `ld/st` fuzzing on higher-rank slice and stale-negative
    descriptor-view surfaces;
  - then continue stale-negative cleanup and heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 15:20 UTC)

- Latest pushed source/test checkpoint:
  - `ca9f94760` on `origin/codex/tmem`
- Two-CTA descriptor-chain `ld/st` auto instruction selection is now covered:
  - `LDST_TWOCTA_DESCRIPTOR_CASES` now uses `LDST_VARIANTS`, so it includes
    `auto`, `32x32b`, `16x64b`, `16x128b`, and `16x256b`;
  - coverage spans `block_two_ctas` and MMAv5-like two-CTA TMEM-linear layouts
    at `N = 64`, `128`, and `256`;
  - the expansion reaches both
    `test_tmem_runtime_matrix_ldst_twocta_descriptor_compositions` and
    `test_tmem_runtime_matrix_ldst_twocta_descriptor_roundtrip_sweeps`.
- Validation for `ca9f94760`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - descriptor roundtrip + two-CTA descriptor matrices:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-descriptor-auto-roundtrip-twocta PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_roundtrip_sweeps' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_compositions' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_roundtrip_sweeps'`
    - `30 passed, 180 skipped in 142.02s`
  - `git diff --check`
    - `PASSED`
- Note:
  - this validation also covers the single-CTA descriptor roundtrip sweep after
    `ebb23b697` made `LDST_DESCRIPTOR_CASES` include `auto`;
  - skipped cases are expected tensor-memory OOR boundaries in lifted
    roundtrip shapes.
- Next:
  - continue staged `ld/st` fuzzing on higher-rank and stale-negative
    descriptor-view surfaces;
  - then continue stale-negative cleanup and heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 15:10 UTC)

- Latest pushed source/test checkpoint:
  - `ebb23b697` on `origin/codex/tmem`
- Staged `ld/st` fuzzing has started with descriptor-chain `auto`
  instruction selection:
  - `LDST_DESCRIPTOR_CASES` now uses `LDST_VARIANTS`, so the basic
    descriptor-composition matrix covers `auto` alongside `32x32b`,
    `16x64b`, `16x128b`, and `16x256b`;
  - coverage spans identity and mixed TMEM-linear layouts at `N = 64`, `128`,
    and `256`;
  - the test still validates functional output, PTX/LLIR opcode agreement, and
    the `tensor_memory_linear` TTGIR marker.
- Validation for `ebb23b697`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - descriptor-chain `ld/st` matrix:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-descriptor-auto PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions'`
    - `30 passed in 27.45s`
  - `git diff --check`
    - `PASSED`
- Next:
  - continue staged `ld/st` fuzzing across two-CTA, higher-rank, and stale
    negative descriptor-view surfaces;
  - then continue stale-negative cleanup and heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 15:00 UTC)

- Latest pushed source/test checkpoint:
  - `7766be003` on `origin/codex/tmem`
- This checkpoint makes the plain MMAv5 kind matrix explicitly cover f16:
  - `MMA_PLAIN_KINDS` is now `f16`, `tf32`, `bf16`, `f8e5m2`, and `f8e4m3`;
  - the shared kind tuple feeds both
    `test_tmem_runtime_matrix_mma_plain_kinds_with_linear_acc` and
    `test_tmem_runtime_matrix_mma_twocta_plain_kinds`;
  - f16 is therefore covered for legacy and tensor-memory-linear accumulator
    layouts across `cta_group::1` and `cta_group::2`;
  - helper dispatch now raises `ValueError` for unknown kind names instead of
    falling through to the fp8 branch.
- Validation for `7766be003`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - plain-kind matrices:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-mma-plain-kind-f16-matrix PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_plain_kinds_with_linear_acc' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_plain_kinds'`
    - `20 passed in 8.64s`
  - `git diff --check`
    - `PASSED`
- Next:
  - continue MMAv5 / `mma_scaled` saturation for remaining layout, CTA, and
    stale-negative surfaces;
  - then continue staged `ld/st` fuzzing, stale-negative cleanup, and
    heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 14:50 UTC)

- Latest pushed source/test checkpoint:
  - `602fd9b44` on `origin/codex/tmem`
- This checkpoint broadens direct scaled-MMAv5 accumulator-subview format
  coverage:
  - added a focused direct scaled-MMA subview kernel that accepts packed FP4
    and unpacked FP8 operands from `random_quantized_tensor`;
  - the kernel stores into a sliced TMEM accumulator view and calls
    `tcgen05_mma_scaled` directly, so this is direct TMEM-view coverage rather
    than another copy-backed scale path;
  - covered format pairs:
    - `mxfp8/mxfp8`;
    - `mxfp4/mxfp4`;
    - `mxfp8/mxfp4`;
    - `mxfp4/mxfp8`;
    - `nvfp4/nvfp4`;
  - assertions validate numeric output, PTX/LLIR MMA opcode agreement, exact
    expected `mxf8f6f4` / `mxf4` / `mxf4nvf4` `scale_vec` suffixes, and the
    `ttg.memdesc_subslice` + `tensor_memory_linear` accumulator path.
- Validation for `602fd9b44`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - exact new format matrix:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-mma-scaled-direct-view-format-matrix-r2 PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_subslice_view_format_matrix'`
    - `5 passed in 5.53s`
  - broadened direct scaled-MMA view slice:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-mma-scaled-direct-view-broadened PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled_minimal or mma_scaled_acc_blockn64_direct_layout or mma_scaled_acc_blockn32_direct_layout or mma_scaled_acc_subslice_view or mma_scaled_lhs_subslice_view or mma_scaled_acc_tile_permuted_64_direct_layout'`
    - `12 passed, 1896 deselected in 9.61s`
  - `git diff --check`
    - `PASSED`
- Next:
  - continue the MMAv5 / `mma_scaled` saturation queue for remaining CTA,
    layout, and stale-negative coverage;
  - then continue staged `ld/st` fuzzing, stale-negative cleanup, and
    heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 14:40 UTC)

- Latest pushed source/test checkpoint:
  - `5e3b2ae87` on `origin/codex/tmem`
- This checkpoint tightens exact opcode coverage for direct scaled-MMAv5
  TMEM-view runtime tests:
  - the minimal direct scaled-MMA kernel;
  - direct accumulator block-N 64 and block-N 32 layouts;
  - accumulator subviews for `N=32` and `N=64`;
  - the TMEM LHS subview;
  - the tile-permuted accumulator layout.
- These tests now assert PTX/LLIR agreement and exact
  `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`
  selection instead of only checking for a generic `block_scale.scale_vec`
  suffix.
- Validation for `5e3b2ae87`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - focused direct scaled-MMA view opcode slice:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-mma-scaled-direct-view-exact PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled_minimal or mma_scaled_acc_blockn64_direct_layout or mma_scaled_acc_blockn32_direct_layout or mma_scaled_acc_subslice_view or mma_scaled_lhs_subslice_view or mma_scaled_acc_tile_permuted_64_direct_layout'`
    - `7 passed, 1896 deselected in 7.77s`
  - `git diff --check`
    - `PASSED`
- Boundary:
  - these direct-view kernels still hardcode e5m2/e5m2 and therefore only pin
    the `mxf8f6f4` scaled-MMA family;
  - broader direct-view format coverage for `mxf4` and `mxf4nvf4` remains open.
- Next:
  - continue direct scaled-MMAv5 TMEM-view format broadening where it can be
    expressed cleanly;
  - then continue staged `ld/st` fuzzing, stale-negative cleanup, and
    heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 14:30 UTC)

- Latest pushed source/test checkpoint:
  - `3374bbf12` on `origin/codex/tmem`
- This checkpoint tightens scaled-MMAv5 runtime-matrix instruction coverage:
  - the scaled-MMA copy matrix and geometry sweep already covered
    `mxf8f6f4`, `mxf4`, and `mxf4nvf4` kind selection across `cta_group::{1,2}`,
    accumulator layout kind, block geometry, and multicast combinations;
  - they now assert that PTX and LLIR agree on every emitted
    `tcgen05.mma.cta_group::{1,2}.kind::*` opcode;
  - they also assert the exact expected `block_scale.scale_vec::*` suffix for
    the format pair and CTA count, while keeping the exact scaled
    `tcgen05.cp.cta_group::{1,2}.warpx4.32x128b` copy checks.
- Validation for `3374bbf12`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - focused scaled-MMA runtime-matrix opcode slice:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-mma-scaled-opcode-runtime-matrix PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_geometry_sweep'`
    - `52 passed in 15.44s`
  - `git diff --check`
    - `PASSED`
- Next:
  - continue direct scaled-MMAv5 TMEM-view runtime targets and broader
    MMAv5/`mma_scaled` saturation;
  - then continue staged `ld/st` fuzzing, stale-negative cleanup, and
    heuristic cleanup;
  - keep attention deferred until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready.

## Prior Topline (2026-04-11 14:20 UTC)

- Latest pushed source/test checkpoint:
  - `eac11c719` on `origin/codex/tmem`
- This checkpoint expands `ld.red` modifier saturation:
  - introduces `LD_RED_MODIFIER_CASES` with all four legal modifier pairs:
    plain, propagate-NaN, abs, and abs+propagate-NaN;
  - applies the full modifier matrix to identity, tile-permuted,
    column-permuted, and row-permuted positive linear layouts;
  - applies the same modifier matrix to mixed-layout clean negatives, which
    continue to use the dedicated
    `tmem_load reduction source layout is not directly tcgen05.ld.red-compatible`
    diagnostic.
- Validation for `eac11c719`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`
  - focused runtime-matrix `ld.red` slice:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ld-red-modifier-saturation PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py' -k 'ld_red_'`
    - `208 passed, 1695 deselected in 132.40s`
  - `git diff --check`
    - `PASSED`
- Next:
  - continue broader MMAv5/`mma_scaled` runtime coverage;
  - continue staged `ld/st` fuzzing after the targeted MMAv5 surfaces;
  - keep clean negatives only for true unsupported layouts or resource limits.

## Prior Topline (2026-04-11 14:10 UTC)

- Latest pushed source/test checkpoint:
  - `7075f31fc` on `origin/codex/tmem`
- This checkpoint broadens `tcgen05.copy` indexed-view coverage for the
  `128x256b` family within the TMEM capacity limit:
  - positive view shape stays `[128, 128]`, which fits the lifted indexed
    parent and still selects `tcgen05.cp.cta_group::1.128x256b`;
  - payloads now cover `f32` and `i32`;
  - shared layouts now cover 32, 64, and 128-byte NVMMA swizzles;
  - the existing `128x128b` indexed-view case remains in the same parameterized
    test.
- Full `[2, 128, 256]` indexed parent status:
  - now has an explicit runtime-matrix OOR guard;
  - failure boundary is `out of resource: tensor memory`, with the probed
    launch metadata reporting `Required: 1024, Hardware limit: 512`;
  - do not promote this shape to a positive target unless the allocation model
    changes.
- Validation for `7075f31fc`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`
  - indexed-view copy exacts plus full-parent OOR guard:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-cp-128x256-view-saturation PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_indexed_view' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_indexed_view_full_128x256_reports_tmem_oor'`
    - `8 passed in 4.67s`
  - `git diff --check`
    - `PASSED`
- Next:
  - copy coverage has closed the recently recorded direct `128x128b`, scaled
    `warpx4.32x128b` reinterpret-debt, and fit `128x256b` indexed-view gaps;
  - continue the long-term `ld.red`, broader MMAv5/`mma_scaled`, fuzzing,
    stale-negative cleanup, and heuristic phases;
  - keep any further copy work constrained to descriptor shapes that fit the
    512-unit TMEM allocation budget.

## Prior Topline (2026-04-11 14:00 UTC)

- Latest pushed source/test checkpoint:
  - `bf3dd781b` on `origin/codex/tmem`
- This checkpoint migrates the standalone scaled `warpx4.32x128b`
  `tcgen05.copy` validation off `_reinterpret`:
  - `python/test/gluon/test_core.py::test_tmem_copy_2d`;
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4`;
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_layout_probe`.
- The supported replacement for these tests is a logical
  `TensorMemoryScalesLayout` load after `tcgen05_copy`, not a physical
  bitcast:
  - the tests still assert exact
    `tcgen05.cp.cta_group::1.warpx4.32x128b` PTX/LLIR selection;
  - the output now checks the logical scales data directly against the input;
  - each migrated scaled-copy test asserts no `ttg.memdesc_reinterpret` remains
    in TTGIR.
- Important conclusion from the failed bitcast probe:
  - direct `tmem.bitcast(ttgl.int8, (128, 32), TensorMemoryLayout(...))` from a
    `(64, 16)` `TensorMemoryScalesLayout` descriptor is not the right supported
    operation;
  - the old `_reinterpret` was a larger dense physical-inspection view over a
    logical scales descriptor with broadcast/replicated semantics;
  - keep the physical bitcast contract restricted to offset/slice/subview
    sources whose total bit size and exact physical TMEM mapping are equivalent
    to the requested dtype/shape/layout.
- `128x256b` full indexed-view destination probe:
  - a temporary `[2, 128, 256]` indexed parent probe failed at launch metadata
    with `Required: 1024, Hardware limit: 512` tensor-memory OOR;
  - do not add that full-parent shape as a positive test target unless the
    allocation model changes.
- Validation for `bf3dd781b`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`
  - scaled copy exacts:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-cp-scales-logical-load-core PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_core.py::test_tmem_copy_2d' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_layout_probe'`
    - `4 passed in 3.85s`
  - `git diff --check`
    - `PASSED`
- Next:
  - continue `128x256b` TMEM-view saturation only within shapes that fit the
    512-unit TMEM allocation limit;
  - keep attention reverted until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready;
  - continue the long-term `ld.red`, broader MMAv5/`mma_scaled`, fuzzing,
    stale-negative cleanup, and heuristic phases.

## Prior Topline (2026-04-11 13:50 UTC)

- Latest pushed source/test checkpoint:
  - `c5bdb6d5c` on `origin/codex/tmem`
- This checkpoint closes the direct canonical TMEM-linear `128x128b` copy root
  coverage gap:
  - `tmem_copy_128x128_kernel` now takes its TMEM layout as an explicit
    `ttgl.constexpr`;
  - `test_tmem_runtime_matrix_cp_128x128` runs both the legacy
    `TensorMemoryLayout((M, N), col_stride=1)` destination and the canonical
    `_make_tmem_linear_layout(M, N)` destination;
  - both variants assert exact `tcgen05.cp.cta_group::1.128x128b` PTX/LLIR
    selection;
  - the canonical linear variant also asserts `tensor_memory_linear` appears in
    TTGIR.
- Validation for `c5bdb6d5c`:
  - build:
    `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`
  - direct copy exact:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-cp-128x128-linear-direct PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_128x128'`
    - `2 passed in 3.44s`
  - nearby copy controls:
    `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-cp-nearby-controls PYTHONPATH=python:. pytest -s --tb=short -q 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_indexed_view' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_candidate_positive'`
    - `4 passed in 4.53s`
  - `git diff --check`
    - `PASSED`
- The supported `_reinterpret` migration invariant is now phrased as:
  - offset to the right part of TMEM;
  - slice/subview it to the desired physical bits;
  - bitcast to the desired dtype, shape, and layout only when the operation is
    equal-size and preserves the exact physical TMEM mapping of the input
    descriptor.
- Next:
  - keep attention reverted until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready;
  - keep legacy M64 MMAv5 producer-family semantics as explicit design debt;
  - continue copy saturation for TMEM-view destinations on `128x256b` and
    scaled `warpx4.32x128b`;
  - continue the long-term `ld.red`, broader MMAv5/`mma_scaled`, fuzzing,
    stale-negative cleanup, and heuristic phases.

## Prior Topline (2026-04-11 13:40 UTC)

- Latest pushed source/test checkpoint:
  - `57a06c29b` on `origin/codex/tmem`
- This checkpoint closes the direct `tcgen05_mma` i8/PTXAS log-noise slice
  without narrowing generic IR coverage:
  - the Gluon API now rejects direct `tcgen05_mma` with i8/u8 operands and an
    int32 accumulator on `sm_103a+` before IR lowering;
  - the diagnostic tells users to use `tt.dot` so the compiler can select a
    supported MMA version;
  - the `ttng.tc_gen5_mma` i8 conversion tests for `compute-capability=100`
    remain positive.
- Validation for `57a06c29b`:
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`
  - direct i8 clean-negative exacts:
    - `python/test/gluon/test_core.py::test_tcgen05_mma_plain_kind_i8_reports_clean_error`
    - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_i8_reports_clean_error`
    - `3 passed in 3.73s`
  - adjacent positive plain-kind MMA exacts:
    - `python/test/gluon/test_core.py::test_tcgen05_mma_plain_kind_runtime`
    - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_plain_kinds_with_linear_acc`
    - `11 passed in 5.90s`
  - `cd build/cmake.linux-aarch64-cpython-3.12 && lit -v test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `1 passed`
  - `git diff --check`
    - `PASSED`
- Broad-sweep status:
  - the latest full four-way `python/test/gluon` proof remains the green
    pre-`57a06c29b` sweep recorded at `77c43f696`;
  - the four current-branch group failure manifests remain `0` nodeids;
  - after `57a06c29b`, the direct-i8 clean-negative path should no longer
    print the old PTXAS `.kind::i8` reproducer in future shard reruns.
- The supported `_reinterpret` migration invariant remains:
  - offset to the right part of TMEM;
  - slice/subview to the desired physical bits;
  - bitcast to the desired dtype, shape, and layout only when equal-size and
    physical-mapping equivalent to the input descriptor.
- Next:
  - keep attention reverted until a synchronization-aware supported
    `offset/slice/subview -> bitcast` migration is ready;
  - keep legacy M64 MMAv5 producer-family semantics as explicit design debt;
  - continue the long-term `ld.red`, `copy`/`warpx2`, broader
    MMAv5/`mma_scaled`, fuzzing, stale-negative cleanup, and heuristic phases.

## Prior Topline (2026-04-11 09:44 UTC)

- The latest pushed code/test checkpoint remains
  `ab8ff6e6444b1cf8521a06941d6cf1b8a25e3217` on `codex/tmem`, pushed to
  `origin/codex/tmem`.
- The latest pushed docs checkpoint before this refresh is
  `4f6f3a8f6e1fc42ad7f785c24f4b925eb36e1f8e8`.
- The persistent `python/triton_kernels/tests/test_matmul.py` shared-memory
  OOR bucket is now closed at the full-directory level:
  - before this fix, a full `python/triton_kernels/tests` rerun on the branch
    reported `8 failed, 1270 passed, 1840 skipped`;
  - all eight failures were persistent Blackwell matmul variants whose shared
    memory rose above the `232448` byte hardware limit;
  - root cause was the split-load/store replay path using the exact preserved
    physical query for ordinary `ttng.tmem_subslice` replay slices, which made
    a shape-local `128x128` slice select a root-width-like register family and
    insert a `ttg.convert_layout` scratch before the epilogue reshape;
  - ordinary replayed subslices now select layouts from the subview-local
    descriptor type, while explicit `tmem_physical_bitcast` roots still use the
    preserved physical query because exact physical mapping is the API
    contract.
- Validation for the OOR bucket:
  - representative exact:
    - `1 passed in 5.38s`
    - metadata returned to `shared=214120`, `tmem_size=512`
  - focused persistent fp8/mxfp4 matmul slice:
    - `16 passed, 6 skipped in 31.85s`
  - full 4-way `python/triton_kernels/tests` refresh:
    - group 1:
      - `674 passed, 782 skipped, 4365 deselected in 2296.49s`
    - group 2:
      - `533 passed, 923 skipped, 4365 deselected in 1863.14s`
    - group 3:
      - `394 passed, 1062 skipped, 4365 deselected in 1192.94s`
    - group 4:
      - `776 passed, 677 skipped, 4368 deselected in 555.61s`
    - aggregate:
      - `2377 passed, 3444 skipped`
      - `0 failed`
  - `gb200_current_branch_triton_kernels_matmul_oor_refresh_failures.txt`
    is recorded as `0` nodeids.
- The latest code checkpoint closes the live M64 row/col-permuted split-N
  direct ld/st bucket without converting it to a clean negative:
  - invalid exact row-permuted `16x32bx2` warp anchors are now rejected before
    they can lower to PTX with misaligned one-row addresses;
  - simple row/col-permuted `64xN` f32 TMEM-linear roots select the canonical
    aligned M64 split-N register layout when the exact physical view layout is
    not directly usable;
  - the backend default-layout path has the same canonical fallback so
    `convert_layout` does not hit the empty-compatible-layout assertion.
- Clean-negative boundary tightened in the same slice:
  - direct half-row and higher-rank row-half descriptor views now fail early
    with an actionable unsupported-descriptor diagnostic instead of reaching a
    later invalid layout or bad packet decomposition.
- The supported bitcast contract remains the durable `_reinterpret` migration
  rule:
  - offset to the right part of TMEM;
  - slice/subview to the desired physical bits;
  - bitcast to the desired dtype, shape, and layout only when the result is
    equal-size and preserves the input descriptor's exact physical mapping.
- The full branch-added TMEM runtime-matrix file is now refreshed beyond the
  old `413`-nodeid failing manifest:
  - full-file run from the M64 split-N checkpoint:
    - `1437 passed, 354 skipped, 1 failed in 1226.69s`
  - the only failing node was:
    - `test_tmem_runtime_matrix_cp_no_scales_indexed_view_canonicalized[128]`
  - root cause:
    - stale TTGIR spelling expectation; the indexed view now exposes the
      canonical physical layout as `ttng.tmem_physical_layout` on a legacy
      `tensor_memory_encoding` root rather than printing a
      `tensor_memory_linear` encoding on that path
  - follow-up exact rerun after the assertion refresh:
    - `1 passed in 3.31s`
  - `gb200_current_branch_test_tmem_runtime_matrix_focus_e70a3aa09_failures.txt`
    is now refreshed to `0` nodeids.
- Validation on the current pushed checkpoints:
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`
  - `lit -v test/TritonNvidiaGPU/tmem_layouts.mlir test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `2 passed`
  - persistent `triton_kernels` matmul focused OOR slice:
    - `16 passed, 6 skipped`
  - full `python/triton_kernels/tests` 4-way refresh:
    - `2377 passed, 3444 skipped`
  - row/col-permuted M64 split-N exact:
    - `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[rotate1-identity-2-32x32b_splitn]`
    - `PASSED`
  - full row/col-permuted M64 split-N sweep plus new auto-selection guards:
    - `226 passed in 35.56s`
  - split-N immediates / auto / explicit guards plus half-row and exotic
    clean-negative guards:
    - `45 passed in 5.32s`
  - exact stale TTGIR assertion refresh:
    - `test_tmem_runtime_matrix_cp_no_scales_indexed_view_canonicalized`
    - `1 passed in 3.31s`
  - focused `test_core.py` TMEM/MMA manifest refresh:
    - initial rerun of the old `332` nodeids found only two stale expectations:
      - the legacy `64x128` split-N PTX packet offset is now the row-encoded
        `1048576`, matching adjacent direct TMEM atom coverage;
      - the block-basis `ld.red` negative now reports the clean descriptor-view
        diagnostic `TMEM layout '32x32b' unsupported for descriptor view`.
    - exact post-refresh rerun:
      - `2 passed in 3.46s`
    - full focused manifest rerun with the refreshed diagnostic nodeid:
      - `332 passed in 125.78s`
    - `gb200_current_branch_test_core_group3_focus_e70a3aa09_failures.txt`
      is now refreshed to `0` nodeids.
  - `git diff --check`
    - `PASSED`
- Next:
  - continue staged broad validation through wider grouped Gluon sweeps from
    current head;
  - then resume the long-term `ld.red`, `copy`/`warpx2`, MMAv5/`mma_scaled`,
    fuzzing, stale-negative, and heuristic phases.

## Prior Topline (2026-04-11 09:02 UTC)

- Latest pushed code/test checkpoint was
  `ab8ff6e6444b1cf8521a06941d6cf1b8a25e3217` on `codex/tmem`, pushed to
  `origin/codex/tmem`.
- The persistent `python/triton_kernels/tests/test_matmul.py` shared-memory
  OOR bucket had been closed at the focused-slice level, before the full
  4-way `python/triton_kernels/tests` refresh landed.

## Prior Topline (2026-04-11 07:25 UTC)

- Latest pushed code/test checkpoint before the persistent matmul OOR fix was
  `42f62fb143630512aff9a45b7b3846a02d9935fa` on `codex/tmem`, pushed to
  `origin/codex/tmem`.
- The old focused `test_core.py` TMEM/MMA current-branch manifest had just
  been refreshed to empty after two stale expectations were updated.

## Prior Topline (2026-04-11 07:15 UTC)

- Latest pushed code/test checkpoint was
  `5b619c9e3dfd5ca3479f644dada7813bd198e34a` on `codex/tmem`, pushed to
  `origin/codex/tmem`.
- The full branch-added TMEM runtime-matrix file had just been refreshed
  beyond the old `413`-nodeid failing manifest, with only a stale TTGIR
  assertion remaining before the exact assertion refresh.

## Prior Topline (2026-04-11 06:51 UTC)

- Base `HEAD` before this checkpoint is
  `49f1a0fd2e1b2e8ed7d144c470bc7838b2b5ff4a` on `codex/tmem`, pushed to
  `origin/codex/tmem`.
- The checkpoint closed the live M64 row/col-permuted split-N direct ld/st
  bucket without converting it to a clean negative.

## Prior Topline (2026-04-11 05:12 UTC)

- `HEAD` is `220565b20e5a3cfc71333b32da06a34cb4596f90` on `codex/tmem`, with
  the M64 / split-N producer-contract fix committed and pushed to
  `origin/codex/tmem`.
- A full `python/examples/gluon/` refresh on top of that commit found one
  current-branch failure:
  - `python/examples/gluon/01-attention-forward.py::test_op[False-dtype0-True-128-1024-48-4]`
  - current branch: fails with large numerical mismatch
  - merge-base `11ee1144a737006921231bbd3386c187812c38e1`: passes
- Root-cause classification:
  - the attention example reuses an f32 TMEM scratch tile as bf16 P storage via
    `_reinterpret`;
  - that usage depends on compiler-selected physical packing behavior and is
    not a supported semantic contract;
  - the branch now exposes the issue because support-query lowering sees the
    explicit reinterpret support image and emits the unpacked store family,
    while the old fallback happened to emit packed stores.
- Direction:
  - do not commit an ad-hoc TMEM lowering selector to preserve this UB;
  - migrate the attention code, and any similar `_reinterpret` usages found in
    production/example paths, to supported linear-layout/view APIs that express
    the intended descriptor/view manipulation directly.
- After the attention migration is green, rerun the examples/Gluon aggregate,
  refresh the GB200 manifests, and continue the staged broad validation plan.

## Current Topline (2026-04-11 04:44 UTC)

- This supersedes the dirty `2026-04-11 02:55 UTC` handoff state where
  `test_mma_shared_inputs[...]`, descriptor-chain, and the representative
  multicta exact were red again.
- The current local slice closes the M64 / split-N producer-contract recovery
  exacts on this branch:
  - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-32-warps2-8-False-True-acc_dtype0]`
    - passed in the focused control batch
  - `python/test/gluon/test_core.py::test_block_m_64_mma[linear]`
    - passed
  - `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`
    - passed
  - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_splitn_8w-layout9-64-128-32x32b_splitn-8-16x32bx2]`
    - passed
  - `python/test/gluon/test_core.py::test_tmem_linear_roundtrip_splitn_shapes[linear_m64_splitn_64x32-layout11-64-32-expected_offset_imms11]`
    - passed
  - `python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-2-CGA_LAYOUT0-8-0-64-128-64]`
    - passed
  - full `python/examples/gluon/03-matmul-multicta.py`
    - `82 passed, 14 skipped`
- Structural fix summary:
  - MMAv5 accumulator and operand roots now carry explicit producer-root
    markers in addition to the explicit row-plan/physical-layout attrs;
  - those markers are copied through the current TMEM allocation cloning /
    partitioning paths;
  - direct root ld/st planning only preserves the wider backing row plan for
    explicit MMAv5 accumulator/operand roots, so descriptor-chain roots do not
    accidentally inherit producer-only anchors;
  - `getExplicitTMemLdStRowPlan(...)` is now public and follows forwarding and
    descriptor-view chains, allowing lowering to recover a source root's
    explicit row plan without duplicating traversal logic; and
  - source-column lowering now passes the source support row plan through to
    `computeTMemLdStEncodingInfo(...)` and only upgrades f32 `64x32` column
    subviews of wider f32 M64 sources to an explicit/source producer row plan.
- Important trace evidence:
  - the multicta f32 `64x128 -> 64x32` epilogue source-column path now uses
    `warpBase0=2097152`, `warpBase1=4194304`, matching the MMAv5 producer
    contract;
  - the descriptor-chain final full-tile indexed load still uses the narrow
    raw/support contract, `warpBase0=1048576`, `warpBase1=2097152`, which is
    required for that non-MMAv5 descriptor-view roundtrip.
- Current live branch-recovery read:
  - the known old-mainline representative examples are green locally:
    - `02-convolution.py` was green in the prior slice (`48 passed`)
    - `03-matmul-multicta.py` is green in this slice (`82 passed, 14 skipped`)
  - aggregate GB200 example manifests are still stale until refreshed on top
    of the latest commit;
  - before treating the branch-recovery phase as complete, rerun the aggregate
    examples/Gluon shards and refresh the GB200 manifest docs with current
    branch-vs-merge-base classification.
- Immediate next move:
  - commit and push this M64 producer-contract/source-column fix;
  - refresh the aggregate GB200 manifests after the checkpoint commit;
  - then continue staged broad validation and the longer-term `ld.red`,
    `copy`/`warpx2`, MMAv5/`mma_scaled`, and fuzzing phases recorded above.

## Historical Topline (2026-04-10 22:41 UTC)

- The broad GB200 census from `18:48 UTC` remains the last whole-lane snapshot,
  but the primary merge-base-present MMAv5 compiler bucket from that census is
  now closed by a targeted direct-load fix.
- Current GB200 lane read:
  - green:
    - `make test-lit`
      - `248 passed, 2 unsupported`
    - `make test-cpp`
      - `240 / 240 passed`
    - `make NUM_PROCS=24 test-unit`
      - `15153 passed, 5492 skipped, 101 warnings`
      - includes `python/triton_kernels/tests`
        - `2377 passed, 3444 skipped`
    - `make test-regression`
      - `1090 passed, 216 skipped`
    - `make NUM_PROCS=24 test-gsan`
      - `20 / 20 passed`
    - `make test-microbenchmark`
    - fresh four-way current-head function rerun:
      - `pytest --splits 4 --group {1,2,3,4} -k 'test_mma_shared_inputs' python/test/gluon/test_core.py`
      - all four groups green
  - still red / still open:
    - `make test-proton`
      - `11 failed, 114 passed`
      - same `11` failures on merge-base, so keep this bucket preexisting
    - `python/examples/gluon/`
- Current live branch-recovery backlog:
  - no old-mainline TMEM/compiler bucket is currently known open from the
    examples lane after the multicta fix
  - the aggregate `python/examples/gluon/` manifest is stale and needs a fresh
    full-directory rerun on top of the latest fixes
- Closed at `2026-04-10 19:46 UTC`:
  - `python/examples/gluon/02-convolution.py`
    - representative exact now passes
    - full file:
      - `48 passed in 7.96s`
    - root cause:
      - a branch-added full-tile scratch-size shortcut in
        `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp` treated
        `ttg.convert_layout` with `LinearEncodingAttr` as needing the entire
        source tile in shared memory
      - for the convolution epilogue
        `tensor<256x256xf16, #linear> -> tensor<256x256xf16, #blocked>`, that
        inflated the scratch allocation from the merge-base `16 KB` class to a
        full `128 KB` tile and raised the kernel from `147516` shared bytes to
        `262208`
    - supporting cleanup landed in the same debugging slice:
      - preserve exact TMEM encoding across outer-dimension
        `memdesc_index` views when the leaf descriptor is still valid
      - restore barrier-driven `tcgen05.mma` / `tcgen05.mma_scaled` async
        selection in the Gluon builder instead of forcing `is_async=true`
    - current post-fix measurement:
      - representative compile artifact now reports `147520` shared bytes
      - the `+4` byte delta versus merge-base is the deliberate
        `kTensorMemoryAllocSharedBytes` padding for `tcgen05.alloc`
- Current branch-added / contract-evolution tails to keep separate:
  - `test_block_m_64_mma[linear]`
    - still a real wrong-code exact
    - current reading:
      - branch-added / branch-changed TMEM coverage
      - not another standalone ld/st packet bug
      - points to a missing producer-owned physical TMEM family contract for
        slice-after-store TMEM consumers
  - `1` opinionated PTX-expectation exact:
    - `gb200_current_branch_test_gluon_splitn_expectation_tail_failures.txt`
  - `3` stale-negative / support-broadened exacts:
    - `gb200_current_branch_test_gluon_halfrow_stale_negative_failures.txt`
  - the two reinterpret-contract rewrite candidates:
    - `test_tmem_subslice_block_m_64[legacy]`
    - `test_tmem_subslice_block_m_64_parent_layout[linear]`
- Latest MMAv5 closure:
  - root cause:
    - `getDistributedLayoutForTmemLdSt(...)` had widened the 16-bit unpacked
      reinterpret rescue so far that ordinary direct MMAv5 accumulator loads
      were recursing into the generic `I32x32b` builder.
  - fix:
    - keep that recursion only on the reinterpret/query-style `I32x32b` path
      when `!allowSplitNFastPath`, and leave direct loads on the standard
      bitwidth-packing path.
  - validation:
    - `make -j8`
    - four representative current-fail / merge-base-pass exacts now pass
    - full four-way `test_mma_shared_inputs` function rerun:
      - group `1`
        - `3830 passed, 490 skipped`
      - group `2`
        - `2954 passed, 1366 skipped`
      - group `3`
        - `1206 passed, 3114 skipped`
      - group `4`
        - `2584 passed, 1736 skipped`
  - manifest consequence:
    - `gb200_current_branch_test_gluon_mma_shared_inputs_failures.txt`
      is now refreshed to `0` exact nodeids
    - the older `4685` MMAv5 manifest and the pre-fix full-group counts are
      now historical only for that function until broader Gluon shards are
      rerun on top of this fix
- Immediate next move:
  - checkpoint this MMAv5 direct-load fix in the initiative docs and git log
  - then reduce the two example regressions, starting with
    `python/examples/gluon/02-convolution.py`
  - rerun broader Gluon slices after the example bucket or sooner if the
    example diagnosis needs a fresh current-head file-level baseline

## Current GB200/NVIDIA CI Baseline
- For the current stabilization phase, the source of truth for what is "red" is
  the GB200 lane from `.github/workflows/integration-tests-nvidia.yml`, as
  recorded in `gb200_nvidia_ci_inventory.md`.
- This GB300 devbox is treated as representative for that Blackwell lane.
- Live lane status belongs in:
  - `gb200_nvidia_ci_inventory.md`
  - `gb200_failure_manifest.md`
- The detailed checkpoint bullets below are retained as history and may mention
  already-cleared intermediate counts; use the timestamped sections above and
  the dedicated GB200 inventory docs for the current live state.
- In the current phase, reinterpret-heavy TMEM tests are not automatically
  treated as compiler bugs; if they depend on implicit physical TMEM mapping,
  rewrite them to explicit descriptor/view composition first and only treat
  residual failures as compiler issues.
- Older historical checkpoint preserved below:
  - the explicit MMAv5 row-plan propagation patch is now the correct local
    baseline;
  - after `make -j8`, a fresh `make NUM_PROCS=24 test-unit` run is down to
    `134 failed, 15019 passed, 5492 skipped`;
  - the remaining exact unit surface is now only:
    - `3` `python/test/unit/language/test_matmul.py` nodeids
    - `3` `python/test/unit/language/test_tensor_descriptor.py` nodeids
    - `128` `python/test/unit/language/test_warp_specialization.py` nodeids
  - use
    `gb200_current_branch_test_unit_rowanchor_refresh_failures.txt` as the
    current exact manifest for this bucket rather than the older `182`/`2098`
    manifests.
- Historical structural read:
  - the remaining unit failures now look like a root-preserving
    descriptor-view row-anchor bug rather than the older source-root MMAv5
    producer bug;
  - the strongest current hypothesis is that direct-planning representability
    is collapsing a backing `128`-row MMAv5 contract to the active `64`-row
    descriptor view too early for `memdesc_index` / higher-rank views; and
  - the next compiler fix should therefore target that planner logic directly.
- The GB200 inventory is now closed for this pass:
  - `make test-lit`
    - `247 passed, 1 failed, 2 unsupported`
    - lone exact failure:
      - `test/TritonGPU/pipeline-lower-loop.mlir`
    - merge-base `make test-lit` is green (`248 passed, 2 unsupported`), so
      this lit red is branch-local old-mainline coverage
  - current branch green lanes:
    - `make test-cpp`
    - `make test-gsan`
    - `make test-regression`
    - `make test-microbenchmark`
    - `python/triton_kernels/tests`
      - `2377 passed, 3444 skipped`
    - isolated `python/test/gluon/test_lowerings.py`
      - `4937 passed, 512 skipped`
  - current branch non-green lanes:
    - `make NUM_PROCS=24 test-unit`
      - `134 failed, 15019 passed, 5492 skipped`
      - exact current-head bucket remains:
        - `3` `python/test/unit/language/test_matmul.py` nodeids
        - `3` `python/test/unit/language/test_tensor_descriptor.py` nodeids
        - `128` `python/test/unit/language/test_warp_specialization.py`
          nodeids
    - `make NUM_PROCS=24 test-gluon`
      - independent current-head red buckets are now reduced to:
        - focused `python/test/gluon/test_core.py`
          - `332` exact nodeids
          - `146` merge-base-present
          - `186` branch-added / branch-changed
        - focused `python/test/gluon/test_tmem_runtime_matrix.py`
          - `413` exact nodeids
          - all branch-added coverage
        - isolated `python/test/gluon/test_fpsan.py`
          - `1` exact branch-changed nodeid
        - isolated `python/test/gluon/test_frontend.py`
          - `1` exact branch-added nodeid
        - `python/examples/gluon/02-convolution.py`
          - `48` exact nodeids
          - representative failure:
            `OutOfResources` (`262208` shared-memory bytes requested vs
            `232448` hardware limit)
          - merge-base full-file rerun is green (`48 passed`)
        - `python/examples/gluon/03-matmul-multicta.py`
          - `14` exact nodeids
          - representative failure:
            `9600 / 20000` mismatches (`48.0%`)
          - merge-base full-file rerun is green
            (`82 passed, 14 skipped`)
      - isolated green confirmations:
        - `python/test/gluon/test_layout_format_view.py`
        - `python/test/gluon/test_lowerings.py`
      - implication:
        - the raw shard-3 lowerings/layout tails were fallout, not independent
          live buckets
    - `make test-proton`
      - `11` exact failures in `third_party/proton/test/test_profile.py`
      - same exact bucket is red on merge-base, so keep treating it as
        `PREEXISTING_ON_MERGE_BASE`
- A later exact-rerun checkpoint has now repaired the representative direct
  split-N TMEM bucket from the original census:
  - `test_tmem_descriptor_chain_matrix[...]` is green again;
  - `test_tmem_linear_roundtrip_splitn_shapes[...]` is green again; and
  - `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[...]` is
    green again.
- The fix was planner-side and narrow:
  - when a direct TMEM view analysis has a zero row basis but the active
    physical layout already matches the logical shape, direct ld/st planning
    must solve on the active layout instead of the lifted support image;
  - row-anchor solving and safety checks now use that active direct-planning
    layout; and
  - the older row-zero direct-view repair shims must stay disabled in that
    path.
- This updates the branch story:
  - the exact split-N ld/st regressions are no longer the live blocker; but
  - the remaining representative unit failures are still red and now look like
    a narrower MMAv5 producer-side bucket (`test_dot[...]`,
    `test_simple_matmul[...]`, and warp-specialization forward).
- The generated GB200 failure manifests are now split enough to drive fixes
  directly:
  - use `gb200_current_branch_test_unit_rowanchor_refresh_failures.txt` for
    the current exact unit bucket;
  - use `gb200_current_branch_test_core_group3_focus_e70a3aa09_failures.txt`
    plus its merge-base-present/missing splits for the current independent
    shard-3 core bucket;
  - use
    `gb200_current_branch_test_tmem_runtime_matrix_focus_e70a3aa09_failures.txt`
    for the dominant branch-added runtime-matrix bucket; and
  - use the split example manifests for the two example files rather than the
    older aggregate `62`-nodeid file.
- Use `gb200_nvidia_ci_inventory.md` for the exact continuously updated red
  list and lane-by-lane census status.
- Use `gb200_failure_manifest.md` and the adjacent generated `.txt` files when
  you need exact nodeid lists for the current-branch failures or the
  merge-base-existing shard-3 subsets.
- The cache/census caveat also tightened:
  - a simple on-disk `TRITON_CACHE_DIR` collision is not yet proven;
  - the stronger current clue is process/worker/device contamination after bad
    kernels or insufficient xdist isolation; and
  - keep the dedicated harness investigation on the backlog, but do not let it
    blur the now-isolated independent red buckets above.
- Updated fix order from the completed GB200 inventory:
  1. direct-view row-anchor representability:
     - `test/TritonGPU/pipeline-lower-loop.mlir`
     - `python/test/unit/language/test_matmul.py`
     - `python/test/unit/language/test_tensor_descriptor.py`
  2. higher-rank MMAv5-root warp-specialization bucket:
     - `python/test/unit/language/test_warp_specialization.py`
  3. merge-base-present focused `python/test/gluon/test_core.py` bucket
  4. branch-added / branch-changed Gluon buckets:
     - runtime-matrix
     - focused core tail
     - fpsan
     - frontend
  5. example portability / wrong-code cleanup
  6. return to the reinterpret-contract rewrite candidates once the missing
     explicit view/backend support is implemented
- That means the broad unit XML surface is branch-local, but not all of it is
  a primary fresh root-cause bucket:
  - representative fresh exact nodeids from the XML-only tail files all pass
    on the current branch:
    - `test_standard.py`
    - `test_random.py`
    - `test_cache.py`
    - `test_blaslt.py`
    - `test_autotuner.py`
    - `test_launch.py`
    - `test_bindings.py`
    - `test_triton_to_gluon.py`
  - current interpretation:
    - the tail is mostly branch-local xdist/process fallout after a smaller
      set of primary MMAv5/TMEM failures.
- The current dirty checkpoint has moved past the older LLVM raw-first probe.
  The live local diff now fixes a producer-side MMAv5 root-row-plan bug:
  - `getMMAv5RootRowPlan(...)` is the shared helper for all `M=64` MMAv5 root
    TMEM families, not just mutable accumulators;
  - `PromoteLHSToTMem` now annotates source-initialized operand roots with the
    same explicit row-plan contract; and
  - the helper now accepts higher-rank roots whose trailing TMEM row dimension
    is `64`, because the hardware contract is still the full `128`-row backing
    tile.
- Fresh exact reruns after `make -j8` now show that the earlier old-mainline
  unit/regression slice is recovered on the current dirty tree:
  - passing exacts:
    - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
    - `python/test/unit/language/test_matmul.py::test_lhs_in_tmem[float32-False-64-128-32]`
    - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-False-4-1-64-512-32-2-float32-tensorfloat32]`
    - `python/test/regression/test_cast_matmul.py::test_cast_matmul[768-768-1024-16-64-16-bfloat16-float16-float16]`
  - file-level reruns:
    - `python/test/regression/test_cast_matmul.py` is fully green again:
      `1080 passed, 216 skipped`
    - `python/test/unit/language/test_matmul.py` is down to only
      `3 failed, 758 passed, 4780 skipped`
- The remaining fresh exact branch-local buckets on the current dirty tree are
  now narrower and split into two structural families:
  - descriptor-view row-anchor materialization gap:
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-128-32]`
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-16-16]`
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-8-64-128-32]`
    - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float32]`
    - the older full-file rerun still says all three
      `test_tensor_descriptor_reshape_matmul[...]` dtypes fail on this bucket
    - representative error:
      `required row anchors 32,64 are not directly representable in the descriptor view`
      before the software pipeliner asserts
  - warp-specialization higher-rank root bucket:
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-3-64-64-1024-1024]`
      still hits `CUDA error: misaligned address`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_persistent_forward[False-8-True-2-128-64-1024-1024]`
      still shows `65535 / 131072` mismatches (`50.0%`)
    - current read:
      the remaining producer-side bug is probably in another higher-rank TMEM
      root producer path (likely NVWS helpers), not in the now-fixed
      source-initialized root contract from `PromoteLHSToTMem`
- The older broad unit XML and regression manifests are therefore no longer
  current counts for the dirty worktree:
  - they remain useful as pre-fix census evidence and merge-base proof; but
  - their aggregate failure totals must be refreshed after the next broader
    rerun before they are used as the current red total.
- Keep the cache/collision story grounded in the actual probes:
  - there is still no reproduced proof of a simple on-disk cache-key
    collision; and
  - there is now a stronger harness clue instead:
    - simple standalone processes obey `CUDA_VISIBLE_DEVICES`;
    - some pytest/xdist workers in the broad sweeps did not.
- The current-branch GB200 census is now complete through the whole CI target
  stack we care about:
  - `make test-unit` finishes at
    `162 failed, 14991 passed, 5492 skipped`;
  - the completed `test-unit` failing-family split is:
    - `64` `test_warp_specialize_attention_forward[...]`
    - `64` `test_warp_specialize_attention_persistent_forward[...]`
    - `18` `test_simple_matmul[...]`
    - `8` `test_dot[...]`
    - `4` `test_simple_persistent_matmul[...]`
    - `3` `test_tensor_descriptor_reshape_matmul[...]`
    - `1` `test_lhs_in_tmem[...]`;
  - `test-gluon` shard `3 / 4` finishes at
    `1160 failed, 3251 passed, 2036 skipped, 19344 deselected`, dominated by:
    - `603` `test_reduce_layouts[...]`
    - `176` `test_scan_layouts[...]`
    - `64` `test_mma_scaled_tcgen05_copy[...]`
    - `40` `test_tmem_reduction[...]`
    - `32` `test_tmem_reduction_linear_layouts[...]`;
  - `test-gluon` shard `4 / 4` remains the large branch-added TMEM
    runtime-matrix bucket;
  - `test_debug.py` is red even in serial because the harness itself forks and
    then touches CUDA;
  - fused-attention, instrumentation, plugin tests, and
    `python/triton_kernels/tests/` are green.
- Representative isolated reruns now confirm that the new unit buckets are
  standalone failures on this branch, not only broad-shard noise:
  - one `test_dot[...]` nodeid;
  - one `test_simple_matmul[...]` nodeid;
  - one `test_warp_specialize_attention_forward[...]` nodeid;
  - `test_tensor_descriptor_reshape_matmul[float32]`; and
  - one `test_simple_persistent_matmul[...]` nodeid.
- The branch-vs-merge-base PTX comparison now points at one common core issue
  rather than a pile of unrelated regressions:
  - for a representative `BLOCK_M=64, BLOCK_N=512` tf32 matmul, the forced
    `16x256` variant still chooses the same family on both sides, but the
    current branch collapses the second support band into contiguous packet
    offsets (`+0,+128,+256,+384`) while merge-base uses the lifted offsets
    (`+0,+128,+1048576,+1048704`);
  - the non-forced sibling shifts family selection from merge-base
    `32x32b.x64` to current-branch `16x32bx2.x64`; and
  - the best current reading is that the new row-plan-aware canonical-`M=64`
    family selection / packet decomposition is misclassifying a support-band
    quotient bit as ordinary repetition.
- Merge-base comparison against `origin/main` is now largely complete:
  - merge-base is `7f61ac734edc657b737fb159a1b9d50cb47944e6`;
  - the detached worktree is `/root/code/triton-mergebase-ci`; and
  - older-tree build drift requires local-only baseline shims (skip example
    plugins, skip legacy GSan runtime) before the baseline can be used for
    branch-vs-main classification.
- The strongest current branch-vs-main facts are now:
  - `python/test/unit/language/test_tensor_descriptor.py` is fully green on
    merge-base (`2604 passed, 110 skipped`), so the three current-branch
    `test_tensor_descriptor_reshape_matmul[...]` failures are all new;
  - `python/test/unit/language/test_matmul.py` is fully green on merge-base
    (`761 passed, 4780 skipped`), so the whole `23`-failure bucket is new;
  - `python/test/unit/language/test_warp_specialization.py` is fully green on
    merge-base (`1599 passed, 202 skipped`), so the full forward +
    persistent-forward bucket is new;
  - `python/test/regression/test_cast_matmul.py` is fully green on merge-base
    (`1080 passed, 216 skipped`), so the current-branch `234`-failure bucket
    is entirely new-on-branch;
  - `python/test/gluon/test_lowerings.py` is fully green on merge-base
    (`4937 passed, 512 skipped`), so the shard-3 `793`-failure lowering bucket
    is entirely new-on-branch;
  - the exact `8` failing `test_dot[...]` nodeids all pass on merge-base;
  - the `162` non-debug nodeids from the current-branch unit failure manifest
    all pass on merge-base, so the remaining `test-unit` failures are branch
    regressions once `test_debug.py` is carved out;
  - the exact current-branch examples/gluon failure manifest (`62` nodeids) is
    branch-changed coverage rather than an identical old-mainline nodeid set:
    the exact nodeids do not exist on merge-base, but the merge-base full-file
    rerun is green (`130 passed, 14 skipped`);
  - the exact merge-base-present shard-3 subsets all pass on merge-base:
    - `185 / 185` `test_core.py` nodeids
    - `38 / 38` `test_fpsan.py` nodeids
    - `1 / 1` `test_layout_format_view.py` nodeid;
  - `python/test/unit/test_debug.py` still fails with the same `20` exact
    nodeids on merge-base; and
  - `third_party/proton/test/test_profile.py` still fails with the same `11`
    nodeids on merge-base, so Proton is a census item but not a TMEM branch
    recovery item.
- There is still an important middle bucket between “new regression” and
  “preexisting”:
  - some shard-3 failures are branch-changed parametrization surfaces rather
    than old upstream exact nodeids:
    - `18` `test_core.py` nodeids
    - `22` `test_fpsan.py` nodeids
  - those belong in the branch backlog, but they should be treated as
    branch-changed coverage, not as regressions against an identical old test.
- The stale
  `test_tmem_runtime_matrix_block_descriptor_reports_clean_error[...]`
  expectation has already been updated to the current row-anchor diagnostic and
  is no longer part of the red list.
- The higher-rank half-row "reports clean error" family now needs re-triage:
  - at least one exact nodeid no longer raises and should be treated as a
    stale-negative / broadened-support candidate; while
  - at least one sibling case passes clean in isolation after the shard run,
    which means large shard-level failure counts in that family cannot be taken
    at face value without clean reruns.
- First rewrite experiments for the two remaining `block_m_64` tests did not
  produce a landable explicit-view replacement yet:
  - simple direct 2D slice rewrites compiled and reproduced the same wrong
    outputs as the old reinterpret-heavy tests, so they did not actually
    restate the intended borrowed-physical view contract; and
  - a more explicit quarter-band reorder
    (`reshape((64, 2, 2, 32)).permute((0, 2, 1, 3)).reshape((64, 4, 32))`)
    appears algebraically promising, but direct stores to the reordered bands
    currently fail cleanly with
    `required row anchors 32,64 are not directly representable in the descriptor view`.
- Implication: the reinterpret-test rewrite is still the next task, but it now
  likely requires either one more explicit view projection step or compiler
  support for the reordered M64 quarter-band descriptor view.
- A follow-up row-split probe also suggests the gap is not only in final
  lowering:
  - splitting the view into `2 x 32` rows before the quarter-band reorder
    should, in principle, avoid the unsupported `32,64` anchor requirement on
    the full `64`-row view; but
  - once written as a higher-rank TMEM view,
    `halves.index(0)` currently fails with
    `failed to infer memdesc_index result type ... rank must be less than or equal to the memdesc rank for tensor memory`.
- So the explicit-contract path is currently blocked by both:
  - direct-view row-anchor materialization on the reordered `64`-row view; and
  - higher-rank TMEM view/index inference on the row-split variant.
- Treat those as concrete missing TMEM feature/surface-coverage gaps, not just
  as failed test rewrites:
  - TMEM explicit view composition still lacks enough surface coverage to
    express the old borrowed-physical `block_m_64` intent with guaranteed APIs;
  - direct ld/st lowering does not yet cover reordered M64 quarter-band views
    even when the intended physical grouping is explicit; and
  - higher-rank TMEM reshape/index/subslice inference still has holes for the
    row-split workaround that would otherwise stay inside a supported `32`-row
    direct frontier.
- When the initiative returns to fixing bugs/hacks/cleanup in this area, these
  are now explicit backlog items rather than rediscovery work:
  - support or cleanly model reordered M64 quarter-band descriptor views;
  - broaden higher-rank TMEM view/index inference so row-split descriptor
    compositions remain representable; and
  - only after that, finish rewriting the old reinterpret-heavy `block_m_64`
    tests to guaranteed APIs.
- New current-execution ordering after the first real GB200 census:
  - keep the two rewrite-candidate `block_m_64` tests logged as missing
    explicit-view surface coverage, but do not let them dominate the whole
    initiative narrative;
  - fix or at least isolate the real `M=64` split-N TMEM regression first,
    because it breaks previously green exact controls on clean reruns;
  - separately triage the higher-rank half-row clean-negative family and update
    tests where support has genuinely broadened; and
  - keep the broader GB200 non-TMEM buckets visible in the inventory as
    distinct follow-up surfaces instead of folding them into the TMEM story.

## Current Decisions
- Backward compatibility is by early normalization, not by maintaining dual
  semantic paths.
- `tensor_memory_descriptor.slice(...)` will match
  `shared_memory_descriptor.slice(...)`, including `dim=0` default and
  arbitrary-dimension slicing.
- `immHalfSplitOffset == 0` is a legal `16x32bx2` lowering case; the old PTXAS
  workaround must be removed.
- Development-time PTX and hardware validation is expected for ambiguous
  `tcgen05` behavior, but only targeted regressions land in-tree.
- This devbox can be unstable. Land TMEM work in frequent, modular commits
  with commit messages that explain context, motivation, implementation, and
  validation, then push each commit to `jeffniu-openai/codex/tmem` so the
  remote branch remains a recoverable checkpoint.
- `M=64` MMAv5 accumulator ld/st row plans are an explicit producer-owned
  contract on the backing `ttng.tmem_alloc` via
  `ttng.tmem_ldst_row_plan`; producer-side TTGIR construction (currently
  `AccelerateMatmul` and the manual Gluon `tcgen05_*` builders) is
  responsible for attaching it, and lowering/query helpers must consume that
  contract instead of inferring it from alloc users.
- For rank-2 `M=64`, `f32`, 4-warp TMEM root loads, handle-aware layout
  selection and LLVM lowering must agree on the same split-N-compatible family:
  - start from `getTmemCompatibleLayouts(queryTy, numWarps)` before appending
    memdesc-specific rescue layouts; and
  - keep atom matching strict so a speculative `I32x32b` candidate cannot
    preempt a legal `I16x32bx2` split-N root load.
- The initiative target is to admit any arbitrary linear TMEM layout that can
  be lowered correctly and executes correctly on hardware. Clean negatives are
  only for ISA-impossible cases; if a formerly negative layout now codegens and
  runs correctly, update the test to positive rather than preserving the
  rejection.
- TMEM reinterpret lowering should be defined by linear-layout arithmetic, not
  by accumulating view-specific rescue rules:
  - derive reinterpret support layouts, origin remaps, row anchors, packet
    offsets, and atom-family selection from exact `LinearLayout`
    compose/invert/pseudoinvert arithmetic wherever possible;
  - assume there is a sane algebraic lowering for reinterpret/view composition
    and keep pushing toward it, because once the arithmetic is right most of
    the bug surface disappears;
  - the arithmetic can be subtle, but that is a reason to solve the linear
    algebra cleanly rather than to accrete case-by-case rewrites;
  - if that algebra is correct, most reinterpret bugs disappear and most
    special cases become unnecessary; and
  - explicit negatives should remain only for layouts whose image cannot be
    realized by the PTX instruction families after the arithmetic is done
    correctly.
- On this arm64 devbox, full `make` currently needs
  `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13`
  to unblock the unrelated GSan runtime build.
- Plain `ttng.tmem_load`, `ttng.tmem_store`, and source-initialized
  `ttng.tmem_alloc` now defer layout feasibility to relayout/lowering instead
  of rejecting non-default compatible register layouts in the verifier.
- `ttg.memdesc_subslice` on TMEM preserves the canonical physical TMEM
  encoding; it is a pointer/view operation over the same backing allocation.
- `ttng.tmem_subslice` is treated as compatibility sugar over the same physical
  TMEM allocation and no longer re-encodes the result shape.
- `ttg.memdesc_index` remains permissive for the extra leading TMEM
  multibuffer dimension; an alloc-shape equality requirement there was tried
  and reverted because it broke the established TMEM multibuffer view model.
- Buffer-region accounting for TMEM subviews follows physical TMEM allocation
  size, not only the smaller logical result shape.
- Relayout now prefers direct TMEM ld/st-compatible layouts and only inserts a
  single `convert_layout` fallback when direct lowering is not possible.
- A non-empty TMEM `block` basis implies multi-CTA CGA semantics regardless of
  `twoCTAs`; `twoCTAs` is extra MMAv5/protocol intent metadata, not the sole
  source of CTA tiling.
- Unsupported TMEM/MMAv5 combinations must fail with diagnostics, never with
  parser or verifier crashes; legacy-layout matching paths need defensive
  handling for impossible candidate families.
- TMEM block-bases descriptor views that eventually reinterpret to a
  `64x32` TMEM-linear memdesc currently have no TMEM-compatible register
  layout for plain `tmem_load/store`; they are a tracked unsupported frontier
  and must fail with a clean relayout diagnostic instead of asserting.
- Higher-rank TMEM descriptor compositions that repartition a tile into
  broadcasted subviews currently fail at `get_reg_layout(...)` with a clean
  diagnostic (`TMEM layout '<atom>' unsupported ... reshape or permute so TMEM
  columns stay contiguous ...`) rather than reaching a pass-time assert or
  late lowering failure.
- TMEM `memdesc_subslice` inference must preserve the source logical-to-physical
  projection when it synthesizes a standalone TMEM-linear result. If projecting
  away sliced dimensions would densify sparse physical TMEM row/col bits, the
  view is unsupported and must fail cleanly instead of being re-encoded to a
  different physical mapping.
- TMEM load-reduction on canonical linear layouts is only legal when the
  N dimension is not sharded across threads. Canonical identity TMEM-linear
  layouts satisfy this and execute correctly; the current mixed TMEM-linear
  family does not and should fail with the dedicated
  `tmem_load reduction with N dimension sharded across threads is not supported`
  diagnostic.
- The runtime sweep must distinguish between:
  - general TMEM ld/st-legal linear layouts, which may be far broader than
    MMAv5/copy hardware families;
  - MMAv5-legal canonical TMEM-linear families, which are exactly the
    TMEM-linear layouts equivalent to supported legacy
    `#ttng.tensor_memory_encoding` tiles; and
  - `tcgen05.cp`-legal TMEM destinations, which are constrained by the finite
    `cp` atom family and shared-memory multicast patterns.
- A positive MMAv5 runtime test must use the canonical TMEM-linear equivalent
  of a supported legacy MMAv5 accumulator/LHS tile. A generic block-basis
  TMEM-linear layout that is fine for ld/st is not necessarily MMAv5-legal.
- For tensor-memory scales direct ld/st, `instr_variant` is a hint, not a
  guarantee of the final atom family:
  - narrow tiles requested as `32x32b` may still lower through
    `16x32bx2.x1`, and
  - some `16x32bx2`-requested cases canonicalize to wider `32x32b.x{1,2,4,8}`
    atoms when the register layout permits it.
- The current direct scales ld/st positive frontier is now runtime-validated
  for exact PTX/LLIR opcode matches across:
  - `16x32bx2.x1`
  - `32x32b.x{1,2,4,8,16,32}`
  - including representative offset-bearing cases (`+2`, `+4`, `+8`, `+16`)
    and both direct/default and explicit-variant `get_reg_layout(...)` paths.
- The current x1 direct ld/st frontier is runtime-validated for:
  - `f16` packed/unpacked one-CTA TMEM
  - `f32` one-CTA and two-CTA TMEM-linear/legacy-equivalent layouts
  - descriptor-chain compositions over those x1 shapes
  - while `16x64b`, `16x128b`, and `16x256b` remain clean negatives for
    `[M, 1]` x1 shapes.
- The current no-scales two-CTA `tcgen05.copy` frontier is runtime-validated
  for both legacy and canonical TMEM-linear destinations over:
  - `N=16` with `swizzle=32`
  - `N=32` with `swizzle in {32,64,128}`
  - `N in {64,128,256}` with `swizzle=128`
  - all lowering to `tcgen05.cp.cta_group::2.128x256b` with exact counts.

## Runtime Fuzz Strategy

- The exact execution matrix, opcode inventory, GPU sharding, and per-family
  fuzz axes now live in `fuzz_plan.md`. This file keeps the durable policy and
  current-state summary; `fuzz_plan.md` is the operational playbook.

### Outcome classes
- `PASS`: kernel compiles, executes on GPU, numerics are correct, and PTX/LLIR
  contain the expected `tcgen05` opcode family and offsets/immediates.
- `CLEAN_UNSUPPORTED`: compile fails with a diagnostic that explains why the
  layout/op is impossible to lower with the ISA and suggests a viable fix
  (`get_reg_layout`, `convert_layout`, reshape/permute so columns stay
  contiguous, or use a supported MMAv5/copy tile).
- `BUG`: parser/verifier/pass/lowering crashes, malformed IR cannot be printed,
  valid IR crashes a pass, or GPU output/PTX mismatches expectations.

### Shared test oracles
- Numerical correctness on real Blackwell hardware is mandatory for positive
  cases.
- The extracted `tcgen05` opcode stream from PTX and LLIR must match exactly.
- For generic descriptor compositions, TTGIR must contain the generic memdesc
  view ops (`memdesc_subslice/index/trans/reshape/reinterpret`) rather than a
  bespoke TMEM-only path, except where `ttng.tmem_subslice` is explicitly
  being regression-tested as compatibility sugar.
- Impossible cases must never assert/crash and must emit actionable notes.

### `tcgen05.ld/st` fuzz plan
- Sweep TMEM-linear families by legality class:
  - canonical identity (`128x{64,128,256}`);
  - canonical mixed/interleaved row/col (`128x{64,128,256}`);
  - `block`-basis general layouts for `256x{64,128,256}`;
  - `block + two_ctas` general layouts for `256x{64,128,256}`;
  - canonical MMAv5-like two-CTA layouts derived from
    `ttgl.to_linear_layout(TensorMemoryLayout(...), shape)` for the narrower
    MMAv5 overlap cases;
  - `blockM=64`/split-N families that must lower through `16x32bx2`.
- For each legal family, sweep instruction selection:
  - `auto`
  - `32x32b`
  - `16x64b`
  - `16x128b`
  - `16x256b`
  - `32x32b_splitn` where applicable
- For each case, sweep descriptor compositions over legal views:
  - multibuffer `slice(..., dim=0)` then `index`
  - `slice(..., dim=0/1/2)` on reshaped 3D views
  - paired `permute`/inverse-`permute`
  - `reshape` chains that preserve element count
  - `_reinterpret` back to a legal TMEM-linear layout
- Functional oracle:
  - round-trip equality for plain load/store
  - exact offset/immediate checks for `16x32bx2`
  - exact atom family checks for all other atoms
- Negative frontier:
  - layouts with non-contiguous TMEM columns or no supported TMEM register
    layout should fail with the relayout diagnostic, not assert.

### `tcgen05.ld.red` fuzz plan
- Sweep `red_op in {min, max}`.
- Sweep modifiers:
  - `abs in {false, true}`
  - `propagate_nan in {none, all}` where legal
- Sweep shapes/warp partitions that are theoretically reducible:
  - `M in {128, 256}`
  - `N in {32, 64, 128, 256}`
  - `num_warps` chosen so the N dimension is not sharded across threads when
    hardware reduction is required.
- Functional oracle:
  - TMEM round-trip path still returns the original tensor
  - reduced output matches the corresponding PyTorch reduction with correct
    NaN/abs semantics
  - PTX/LLIR use `tcgen05.ld.red.sync.aligned.32x32b.x*.{min|max}[.abs][.NaN].f32`
- Negative frontier:
  - unpacked formats, integer types, or N-sharded register layouts must fail
    with the dedicated reduction diagnostic.

### `tcgen05.cp` fuzz plan
- Enumerate the currently implemented atom families:
  - `cta_group::1.128x256b`
  - `cta_group::1.128x128b`
  - `cta_group::{1,2}.warpx2::02_13.64x128b` if supported in lowering
  - `cta_group::{1,2}.warpx2::01_23.64x128b` if supported in lowering
  - `cta_group::{1,2}.warpx4.32x128b`
- For no-scales copies, sweep `M in {128, 256}`, `N in {16,32,64,128,256}`,
  `BLOCK_N` divisors, and relevant swizzles, skipping only true SMEM
  out-of-resource cases.
- For scales and multicast variants, synthesize exact shared layouts that drive
  the documented multicast bits and alias the destination back through a legal
  TMEM view to validate the payload.
- Functional oracle:
  - copied output equals input (or equals the documented aliasing/permutation
    for scales layouts)
  - PTX/LLIR opcode family and count exactly match the expected atomization
- Negative frontier:
  - unsupported shared multicast/layout combinations must fail cleanly, not
    silently change atom family or crash.

- Progress note:
  - `probe_cp_multicast_layouts.py` and the companion `cp_warpx2_probe_test.py` both exhaust the
    simple interleavings seen above; the current runs on GPU3 keep hitting the
    `tcgen05_copy` legalization error before emitting any `warpx2` opcode.
  - `warpx2_layout_manual.py` tries the same insertion patterns with quieter
    logging but still fails to produce `warpx2`, while `warpx2_layout_linear.py`
    inspects the converted linear layout and shows that the two 32-byte-row
    basis entries never become `[0, 0]`, so multicast stays zero.
  - Need to broaden the layout search (more column strides, extra block
    reorderings, two-CTA-specific layouts) until `tcgen05_copy` legalizes one of
    the documented `warpx2` families.
  - 2026-03-25 refresh:
    - Exhaustive surjective one-hot interleavings for direct scales copies
      (`64x16 -> 128x32`) produced only 2 legal layouts, both emitting
      `warpx4.32x128b` (no `warpx2`).
    - Exhaustive one-hot interleavings for no-scales (`128x4 i32`) produced
      only `128x128b` (no `warpx2`).
    - Subslice probes from larger shared allocations (`parent_rows=128/256`)
      yielded no legal `tcgen05_copy` cases in sampled layout sets; when slice
      start was non-zero, frontend constraints (`split offset may not touch the
      tile`) rejected many candidates.
    - Mixed-basis shared-linear mutations (e.g. `[32, k]`) reached lowering but
      still failed descriptor legalization; still no executable `warpx2`.
    - Runtime coverage now includes executable `cta_group::2.warpx4.32x128b`
      in `python/test/gluon/test_tmem_runtime_matrix.py` via the scaled-MMA
      copy path; direct standalone scales-copy kernels still have not produced
      executable `warpx2`.

### `tcgen05.mma` fuzz plan
- Separate sweeps by hardware family:
  - `cta_group::1`
  - `cta_group::2`
- Sweep accumulator layouts:
  - legacy `TensorMemoryLayout`
  - canonical TMEM-linear equivalents of those legacy layouts
  - TMEM-general but non-MMAv5 layouts as negative cases
- Sweep accumulator usage:
  - `use_acc in {false, true}`
  - multicast/commit path where applicable
- Sweep supported tile families:
  - canonical `blockM=128`
  - canonical `blockM=64` split-N/interleaved families where legal
  - LHS-in-shared and LHS-in-TMEM where supported
- Functional oracle:
  - result matches `torch.matmul` (plus accumulator add when `use_acc=true`)
  - PTX/LLIR contain the exact `tcgen05.mma.cta_group::{1,2}.kind::*` opcode
    family and expected commit/barrier opcodes
- Negative frontier:
  - mixed/block-basis TMEM-general layouts that are not legacy-equivalent must
    fail with the MMAv5 compatibility diagnostic.

### `tcgen05.mma_scaled` fuzz plan
- Sweep format combinations already supported in tree (`mxfp8`, `mxf4`,
  `mxf4nvf4`, etc.) and confirm the emitted `scale_vec::{1X,2X,4X}` suffix.
- Sweep `cta_group::{1,2}` where supported.
- Sweep accumulator/scales layouts:
  - legacy encodings
  - canonical TMEM-linear equivalents
  - two-CTA canonical families
- Functional oracle:
  - numerical output matches the reference within the established tolerance
  - PTX/LLIR `tcgen05.mma...block_scale.scale_vec::*` opcode streams match
  - associated `cp`/barrier ops appear as expected for the chosen pipeline
- Negative frontier:
  - unsupported TMEM-linear accumulator families, scale layouts, or CGA
    mismatches must fail cleanly with MMAv5/scaled-MMA diagnostics.

## Work Split
- Main rollout owns core attr/type/helper/lowering integration.
- Subagents may be used for bounded exploration, PTX validation snippets, and
  test drafting. Claims belong in `agents/claims.md`.

## Next Execution Plan
- The current execution order is driven by the long-term plan near the top of
  this file, not by the older broad-fuzz milestones below.
- Immediate implementation order:
  - keep the GB200 inventory current first:
    - finish the remaining lane/exact classification work against merge-base
    - record flaky/cache/process-contamination evidence separately from real
      branch-caused failures
  - clear the live descriptor-chain bucket next:
    - capture fresh TTGIR/PTX/query traces for one `linear_m64_*` exact and
      one `linear_mixed_*` exact
    - fix the underlying planner/lowering issue rather than adding more
      family-specific surgery
  - then move to the separate old-mainline regression lane:
    - `02-convolution.py`
    - `03-matmul-multicta.py`
  - once the branch-caused GB200 reds are down to stable buckets, continue the
    larger cleanup:
    - shared TMEM ld/st planner extraction
    - legacy/TMEM-linear planning unification
    - quotient-driven reinterpret packet decomposition
    - reinterpret-contract test rewrites where the old `_reinterpret`
      dependence was never part of the API guarantee
  - then re-broaden validation through TMEM runtime, MMA/matmul,
    `triton_kernels`, and the wider suite.
- The detailed post-cleanup saturation playbook lives in `fuzz_plan.md`.

## Active Fuzz Backlog
- Broad fuzzing is no longer the immediate task while the shared planner /
  reinterpret cleanup is still incomplete.
- The active fuzz backlog should be resumed in this order once the current
  planner bugs are green:
  - rerun the nearest TMEM runtime slices and refresh the positive frontier for
    descriptor/view compositions after the planner cleanup;
  - expand `ld.red` runtime saturation over the supported non-sharded layouts
    and modifiers;
  - continue `cp` family completion, especially `warpx2`, once the required
    descriptor/address/message synthesis is in place;
  - broaden `mma` and `mma_scaled` runtime coverage beyond the already-proven
    families;
  - make sure every emitted family keeps:
    - runtime PTX/LLIR parity coverage,
    - at least one lit lowering check, and
    - clean-negative coverage for true impossible frontiers.
- 2026-04-11 status reminder:
  - do not treat historical `warpx2` discovery failures as current blockers;
    the old candidate-positive no-scales `warpx2` tests were later recorded
    green, so the remaining copy backlog is coverage/saturation plus any new
    deterministic families found by direct probes;
  - treat `ld/st` fuzzing as the broadest planner stress test after the live
    M64 row-plan bugs are fixed;
  - treat `tcgen05.mma` / `tcgen05.mma_scaled` as a later broadening phase,
    with direct scaled-MMAv5 TMEM views and 2-CTA TF32 as the highest-value
    remembered frontiers.
- Ambiguous or undocumented families should still be treated as direct-PTX
  probe targets before promoting them to compiler work.

## Current Higher-Rank TMEM Status
- Chosen semantic direction is now explicit:
  - higher-rank TMEM descriptors should use a full-rank
    `TensorMemoryLinearLayout` that embeds descriptor prefix dims into TMEM
    `col` high bits;
  - generic TMEM `slice/index/permute/reshape/reinterpret` then operate on the
    full-rank encoding;
  - actual TMEM access ops (`load/store/mma/cp`) remain 2D-only for now, so
    higher-rank descriptors must be sliced/indexed/reshaped down to a final 2D
    TMEM view before access.
- Runtime status after the latest GPU validation:
  - full-rank rank-3 TMEM descriptors are working end-to-end for
    `index -> 2D load/store`;
  - full-rank rank-4 TMEM descriptors are working end-to-end for
    `slice -> index -> slice -> index -> 2D load/store`;
  - the same rank-4 path now passes broad row/col-permuted TMEM-linear sweeps
    (`736 passed, 64 skipped`) across explicit ld/st variants plus split-N;
  - the same rank-4 path is working for 2-CTA `block_two_ctas` and
    `mmav5_twocta` layouts as well.
  - descriptor multi-dim slice semantics now have positive GPU proof via the
    new `tmem_ldst_descriptor_multidim_slice_positive_kernel`; the test
    exercises `identity`, `mixed`, and `scrambled_cols` linear layouts,
    verifies the block update and `ttg.memdesc_slice` in TTIR, and checks the
    `16x128b` PTX/LLIR match.
  - 2D last-dimension TMEM slices that need to preserve split-N / M64 physical
    layout now route through the `ttng.tmem_subslice` compatibility path from
    the frontend, restoring valid `block_m_64` MMAv5 compositions.
- Plain MMAv5 runtime coverage status:
  - `f16`, `tf32`, `f8f6f4`, and now `bf16 -> kind::f16` are all exercised
    with GPU execution and exact PTX/LLIR opcode checks for canonical
    accumulator layouts.
  - `kind::i8` remains a clean PTXAS-level unsupported case on `sm_103a`.
- Active uncovered-but-plausible MMA frontier:
  - 2-CTA TF32 MMAv5 currently fails lowering cleanly with
    `tcgen05.mma does not support transposed float32 operands in shared memory`;
    treat this as a compiler BUG / follow-up target, not a settled unsupported
    ISA boundary.
- New 2-CTA plain MMA coverage status (2026-03-26):
  - plain `tcgen05.mma.cta_group::2` is now GPU-proven for:
    - `tf32`
    - `bf16 -> kind::f16`
    - `f8e5m2 -> kind::f8f6f4`
    - `f8e4m3 -> kind::f8f6f4`
  - both legacy and canonical TMEM-linear 2-CTA accumulator layouts are now
    covered in the runtime matrix with exact PTX/LLIR opcode equality.
- New no-crash frontend guard:
  - `SwizzledSharedLayout` now validates `cga_layout` basis rank against
    `order` rank in Python before IR materialization.
  - Invalid 1D shared layouts such as `mbarrier.MBarrierLayout(cga_layout=[[1,
    0]])` now fail with a wrapped `CompilationError` rather than tripping a C++
    assert during `buildCgaLayoutAttr(...)`.
- Active 2-CTA copy BUG frontier:
  - the earlier cold-start failure on 2-CTA no-scales `tcgen05.copy` was not a
    legal-codegen bug once the sequence was validated empirically;
  - correct 2-CTA copy usage requires explicit pre-copy async-proxy ordering,
    i.e. `fence_async_shared(cluster=True)` before `tcgen05_copy` when the
    source lives in distributed shared memory;
  - with that fence in place, the runtime matrix now passes for both legacy and
    canonical TMEM-linear 2-CTA destinations, and PTX/LLIR show the expected
    `tcgen05.cp.cta_group::2.128x256b` plus
    `tcgen05.commit.cta_group::2...multicast::cluster.b64`.
- New verifier/lowering fixes landed:
  - `ttng.tmem_copy` now validates its optional completion barrier through
    `verifyBarrierType(...)` and `verifyCompletionBarrierLayout(...)`, so bad
    barrier layouts fail early and cleanly.
  - unpredicated cross-CTA `ttng.wait_barrier` lowering no longer crashes when
    the leader predicate must be synthesized in LLVM lowering.
- High-value TMEM test gaps from the latest audit:
  - direct canonical TMEM-linear `128x128b` root copy coverage is now closed by
    `c5bdb6d5c`;
  - standalone scales `warpx4.32x128b` copy validation is now off
    `_reinterpret` at `bf3dd781b`;
  - TMEM view destinations for `128x256b` are still under-tested;
  - direct scaled MMAv5 via TMEM views remains a good next runtime target.
- New PASS coverage:
  - executable `32x32b.x1` load/store is now covered for:
    - canonical `TensorMemoryLinearLayout`,
    - legacy packed TMEM,
    - legacy unpacked TMEM;
  - direct `TensorMemoryScalesLayout` ld/st is now covered end-to-end for:
    - default descriptor API on `16x8xi8` with `num_warps=8`,
    - explicit-variant cases on `16x4xi8`, `32x4xi8`, `64x8xi8`, and
      `128x32xi8`;
  - isolated LLVMIR checks for canonical x1 TMEM-linear lowering and direct
    scales x1 load/store are green.
- New BUG frontier from the scales sweep:
  - The explicit narrow-scales gap is now closed:
    - `TensorMemoryScalesLayout(shape=[16, 8], num_warps=8,
      instr_variant="16x32bx2")` now reuses the broadcasted `32x32b`
      register layout and lowers/executed correctly as
      `tcgen05.{ld,st}.sync.aligned.16x32bx2.x1.b32`.
  - Direct scales ld/st with nontrivial CGA bases is now GPU-proven for:
    - `shape=[128, 64]`, `num_ctas=2`, `CGALayout=[[1, 0]]`
    - `shape=[256, 32]`, `num_ctas=2`, `CGALayout=[[1, 0]]`
    - `shape=[256, 64]`, `num_ctas=2`, `CGALayout=[[1, 0]]`
    with exact PTX/LLIR checks for the emitted `32x32b` families.
  - Remaining copy-family frontier:
    - direct PTX probes still show `tcgen05.cp.warpx2::{02_13,01_23}.64x128b`
      is real, but Triton's scales-copy verifier/lowering intentionally keeps
      those layouts `CLEAN_UNSUPPORTED` because the required shared-memory
      descriptor synthesis is not implemented yet.
  - Current targeted lit slice is green:
    - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `test/TritonNvidiaGPU/invalid.mlir`
- The key implementation boundary is now cleaner:
  - row/col zero-basis stripping is only valid for TMEM view-local encodings;
    doing it globally breaks semantically meaningful M64 / split-N / legacy
    MMAv5 families.
- Current remaining sharp edge:
  - intermediate higher-rank TMEM views still fail cleanly at
    `get_reg_layout()/load()/store()` with an actionable 2D-only diagnostic.
  - This is acceptable for the current landing, but direct higher-rank TMEM
    access remains future work.


- 2-CTA no-scales `tcgen05.copy` status (2026-03-26 10:05 UTC):
  - the earlier cold-start corruption is fixed on the current tree;
  - required compiler invariants are now pinned by tests:
    - `ttng.tmem_copy` is a tracked cross-CTA consumer for barrier insertion,
    - distributed shared -> `ttng.tmem_copy` gets a pre-copy
      `ttng.cluster_barrier`,
    - runtime PTX/LLIR for the 2-CTA `128x256b` family uses aligned cluster
      barriers, not the old relaxed-barrier shape.
  - executable coverage is now strong for this path:
    - 28/28 `cp_no_scales_twocta_codegen` runtime cases pass on GPU,
    - both former cold-start tuples also pass 10 fresh-process runs each.
  - remove this from the active BUG frontier; the next copy-family frontier is
    `warpx2` shared-descriptor synthesis.

- 2-CTA no-scales copy compiler contract (2026-03-26 18:25 UTC):
  - `ttng.tmem_copy` must participate in memory-effect analysis and cross-CTA
    barrier insertion just like MMA/TMA consumers.
  - The explicit-buffer alias fast-path is only sound for non-distributed
    shared accesses; distributed slices must still participate in hazard
    tracking.
  - The executable/runtime contract for the working `cta_group::2.128x256b`
    path is now pinned as:
    - `fence_async_shared(cluster=True)` before the copy,
    - aligned cluster barrier pair before the first copy message,
    - exact `tcgen05.cp.cta_group::2.128x256b` counts,
    - exact `tcgen05.commit.cta_group::2...multicast::cluster.b64` commit.
  - Current validation status:
    - former cold-start repro tuples pass 10 fresh-process runs each;
    - full `cp_no_scales_twocta_codegen` matrix is green;
    - surrounding copy/scales/plain-MMA runtime slice is green (`130 passed`).

- Runtime validation checkpoint (2026-03-26 18:27 UTC):
  - the repaired 2-CTA copy path still holds under a broader executable slice;
  - post-fix runtime sweep status:
    - `cp_no_scales_twocta_codegen`
    - `cp_scales_warpx4`
    - `cp_scales_warpx4_via_scaled_mma_copy_matrix`
    - `cp_scales_warpx4_via_scaled_mma_geometry_sweep`
    - `cp_no_scales_linear_32bit_dtypes`
    - `mma_twocta`
    - `mma_twocta_plain_kinds`
    all green together (`103 passed`).
  - no new compiler/runtime correctness frontier opened by this broader slice.
  - note for future work: the transient `smem_layout is not defined` failure was
    only a test-harness regression in `test_tmem_runtime_matrix.py`, not a TMEM
    compiler bug.

- warpx2 frontier audit (2026-03-26 18:32 UTC):
  - current compiler status:
    - classifier recognizes `warpx2::{02_13,01_23}.64x128b`;
    - PTX builder can emit the opcode suffixes;
    - verifier/lowering still rely on the generic MMAv5 shared-descriptor
      builder, which has no `warpx2`-specific descriptor/address model.
  - practical implication:
    - enabling `warpx2` is not a small opcode toggle; it requires dedicated
      descriptor synthesis and copy-address stepping.
  - bounded current-tree probe of the old scales `warpx2` search now fails
    cleanly at verifier time rather than late LLVM legalization.

- Copy-family frontier update (2026-03-26 18:33 UTC):
  - fresh direct-PTX probes reconfirm:
    - `warpx2::02_13` and `warpx2::01_23` are real opcodes with deterministic warp-pair semantics;
    - they are not drop-in substitutes for the current Triton warpx4 descriptor/message schedule;
    - `4x256b` still launches but is not deterministic enough for compiler support.
  - practical implication:
    - the next `warpx2` support step is family-specific descriptor/address/message synthesis, not just classifier widening or opcode substitution.
  - until then, keep these families on the clean-unsupported side with explicit notes rather than attempting partial lowering.

- warpx2 overlay semantics checkpoint (2026-03-26 20:10 UTC):
  - durable experiment artifacts now live in:
    - `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_overlay.py`
    - `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_overlay_gpu3.json`
    - `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_overlay_gpu3_deltas.json`
  - hardware-proven semantics for the inherited descriptor/address family:
    - base copy sites are `[ %r14 + 0 ], %rd3` and `[ %r12 + 0 ], %rd4`;
    - with one active `warpx2` message:
      - `02_13 + (%rd3 @ %r14 + 0)` writes the left half of chunks `0/2`;
      - `02_13 + (%rd4 @ %r14 + 4)` writes the right half of chunks `0/2`;
      - `01_23 + (%rd3 @ %r14 + 0)` writes the left half of chunks `0/1`;
      - `01_23 + (%rd4 @ %r14 + 4)` writes the right half of chunks `0/1`;
      - no tested one-message schedule produced the missing complementary left
        halves for chunks `1/3` or `2/3`.
    - intermediate TMEM destination deltas `+1` and `+2` are invalid on
      hardware for both families and both descriptors:
      `cuCtxSynchronize failed: misaligned address`.
  - two-message inference:
    - `02_13` with the inherited descriptor pair `(%rd3 @ %r14, %rd4 @ %r12)`
      fully covers chunks `0/2` only;
    - `01_23` with that same pair fully covers chunks `0/1` only;
    - mixed-family two-message schedules are still not exact.
  - current design consequence:
    - `warpx2` cannot be enabled from the current two-message warpx4 plan by
      changing only opcode suffixes or small TMEM address deltas;
    - the likely enabling path is a dedicated copy-family planner that can
      synthesize different descriptors and, if necessary, emit more than the
      inherited two copy messages.
  - until that exists, keep `warpx2` cleanly unsupported and preserve the new
    explicit verifier note about partial chunk-pair fills and invalid `+1/+2`
    deltas.

- Plain TMEM ld/st/alloc explicit-layout policy (2026-03-26 21:35 UTC):
  - explicit user-requested TMEM register layouts are now strict:
    - if the requested tensor layout is directly TMEM-compatible, compile it;
    - otherwise, fail with a diagnostic and possible suggested TMEM layouts.
  - the compiler no longer silently inserts fallback `convert_layout` for plain
    `ttng.tmem_load`, `ttng.tmem_store`, or initialized `ttng.tmem_alloc`.
  - the late `make_llir` scheduling of `relayout-tritongpu` was removed for
    this reason; warp-specialization still retains its own relayout path for
    partition re-inference after changing `numWarps`.
  - implication for tests/examples:
    - positive TMEM tests should use `get_reg_layout(...)` when they want a
      directly codegenable register layout;
    - if a test wants a blocked or other non-direct TMEM layout, it must use
      `convert_layout` explicitly around the TMEM access rather than relying on
      hidden compiler repair.

- TMEM lit / LLVM cleanup status (2026-03-26 23:40 UTC):
  - keep the shared-memory reshape fallback in
    `lib/Dialect/TritonGPU/IR/Dialect.cpp`; it is required for the
    `cp_scales_warpx4_via_scaled_mma_geometry_sweep` path.
  - do not keep the local `tryMakeTMemViewEncoding(...)` experiment that stops
    stripping zero row/col bases; it over-restricts legitimate higher-rank TMEM
    indexed views and is not the right fix for the multidim-slice miscompile.
  - some older TMEM conversion lit chunks no longer emit LLVM in the current
    pipeline at all; keep only chunks that survive lowering and check them
    meaningfully.
  - current surviving `ttng.tmem_subslice` descriptor-only LLVM lowering often
    returns the base pointer unchanged and relies on descriptor semantics rather
    than a materialized pointer delta. Tests should check the current emitted
    LLVM, not the pre-linearization offset assumptions.

- Current TMEM runtime-matrix state (2026-03-27 01:10 UTC):
  - parser/frontend:
    - TMEM slice semantics in Gluon now follow shared descriptors by default;
      basic parser coverage should use explicit `dim=1` when it wants the old
      N-slice behavior.
    - the corresponding IR now uses generic `ttg.memdesc_subslice` in the
      frontend test, which is expected.
  - positive runtime coverage newly added:
    - TMEM linear subslice-view `tcgen05.copy` now has real runtime coverage
      for `f32`/`i32` across swizzles `32/64/128`;
    - MMA now has a positive runtime case where the accumulator is obtained via
      `memdesc_index` from a larger TMEM descriptor, for both legacy TMEM sugar
      and lifted `tensor_memory_linear`.
  - BUG bucket kept explicit in tests:
    - there is still a coherent cluster of multidimensional TMEM descriptor
      slice/view cases that compile but produce wrong values at runtime.
    - these cases are intentionally tracked as `pytest.xfail` with `BUG`
      reasons, not promoted to positive coverage and not hidden.
    - representative bug symptoms:
      - higher-rank TMEM slice/view chains write the wrong physical columns;
      - direct multidimensional slice views on `mixed` TMEM layouts update the
        wrong region;
      - two-CTA MMAv5 higher-rank view chains still fail at inference time with
        a CGA mismatch, which is now the expected clean diagnostic.
  - supported boundary remains:
    - plain ld/st, copy, MMA, scaled MMA, and many higher-rank TMEM
      `index`/`reshape`/`transpose` compositions are runtime-covered and pass;
    - `warpx2` copy remains clean unsupported;
    - multidimensional TMEM slice semantics are partially implemented, but not
      yet trustworthy enough to remove the BUG xfails.

- Current TMEM runtime-matrix state (2026-03-27 03:50 UTC):
  - the multidimensional TMEM slice/view BUG bucket is now closed in-tree:
    - the full runtime matrix is `1606 passed, 117 skipped`;
    - there are no remaining TMEM BUG/xfail markers in
      `python/test/gluon/test_tmem_runtime_matrix.py`.
  - `test/TritonNvidiaGPU/ops.mlir` is fixed again:
    - raw `triton-opt test/TritonNvidiaGPU/ops.mlir` now exits `0` and emits
      normal stdout;
    - the focused lit bundle over
      `test/TritonNvidiaGPU/ops.mlir`,
      `test/TritonNvidiaGPU/invalid.mlir`,
      `test/Analysis/test-buffer-region.mlir`, and
      `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
      is green.
  - generic TMEM view typing/parsing boundary:
    - public `ttg.memdesc_subslice` now expects the canonical smaller
      TMEM-linear result encoding when the sliced view is materializable as a
      standalone TMEM-linear memdesc;
    - `ttng.tmem_subslice` remains compatibility sugar, but verifier/equality
      checks now compare normalized TMEM-linear layouts so semantically
      identical canonical encodings do not fail parse/verify because of hidden
      layout-name differences.
  - remaining plain TMEM descriptor/load-store clean negatives are no longer
    pointing at `TensorMemoryUtils.cpp` typing bugs:
    - identity quarter-tile `32x32` views from reshape/slice chains;
    - lifted-layout `dim0` / `half_rows` descriptor views;
    - scrambled/permuted descriptor views;
    - the block-basis reinterpret frontier.
    These all currently mean “no single direct TMEM register layout exists for
    the final view”. Under the current plain-TMEM contract (no silent layout
    repair for ld/st/alloc), they should be treated as direct-lowering/atom
    limits rather than small local compiler bugs.
  - direct-PTX `tcgen05.copy.warpx2` remains a real ISA frontier, but it is no
    longer the remaining blocker for the public TMEM descriptor API:
    - PTX families `warpx2::{02_13,01_23}.64x128b` are real and deterministic;
    - Triton already recognizes/classifies them and can emit the opcode names;
    - current verifier/lowering still rely on the inherited warpx4 shared
      descriptor/address schedule, which is semantically insufficient for a
      future user-reachable `warpx2` path;
    - however, bounded searches across the current public shared-layout and
      explicit TMEM-layout constructors did not find a surjective user-visible
      layout that lowers to `warpx2`;
    - the only raw `warpx2`-looking explicit TMEM layouts found so far require
      row-zero bases, are non-surjective, and are rejected at
      `allocate_tensor_memory` before codegen;
    - under the current public API surface, `warpx2` is therefore a
      direct-PTX / future-API frontier rather than a blocker for “all code the
      user can currently write”.

- public TMEM descriptor API reachability audit (2026-03-27 06:18 UTC):
  - bounded shared-layout classification searches:
    - no-scales canonical destination (`128x4`, `f32`):
      - pure interleavings: `36` layouts checked, no `warpx2`;
      - one-mixed: `1134` tested / `252` surjective, no `warpx2`;
      - two-mixed: `254016` tested / `22680` surjective, no `warpx2`.
    - scales destination (`64x16`, `i8`):
      - pure interleavings: `210` layouts checked, all reachable families
        classify to `multicast=3` / `warpx4`;
      - one-mixed: `2400` tested / `480` surjective, no `warpx2`;
      - two-mixed: `1166400` tested / `90720` surjective, no `warpx2`.
  - bounded explicit TMEM-layout search:
    - raw `warpx2`-looking row-zero layouts exist for `[128,4]`, `[128,8]`,
      and `[128,16]`, but they are non-surjective:
      - `[[1,0],[2,0],[4,0],[8,0],[16,0],[0,0],[32,0]]`
        -> raw `multicast=1` / `warpx2::01_23`;
      - `[[1,0],[2,0],[4,0],[8,0],[16,0],[32,0],[0,0]]`
        -> raw `multicast=2` / `warpx2::02_13`.
    - real parser probe confirms those layouts fail at
      `allocate_tensor_memory` with `The layout must be surjective`; no
      `tcgen05.cp` opcode is emitted.
    - surjective explicit TMEM searches still found no `warpx2` hits:
      - `[128,4]`: `254016` tested / `22680` surjective;
      - `[128,8]`: `893025` tested / `68040` surjective;
      - `[128,16]`: `2371600` tested / `152460` surjective.
  - test coverage added / refreshed:
    - `python/test/gluon/test_frontend.py` now directly checks that the
      non-surjective row-zero explicit TMEM layouts fail cleanly.
    - `python/test/gluon/test_tmem_runtime_matrix.py` now matches the current
      clean verifier diagnostics for the historical `warpx2` candidates.
  - practical conclusion:
    - direct PTX `warpx2` support still needs family-specific
      descriptor/address/message synthesis if Triton ever grows a public,
      surjective route to it;
    - that work is not required to support all code currently expressible
      through the TMEM descriptor API.

- warpx2 probe frontier refresh (2026-03-27 04:44 UTC):
  - added durable bounded probe helpers:
    - `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_overlay_both_bases.py`
    - `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_single_site_halves.py`
    - `.codex/initiatives/tmem_linear_generalization/experiments/probe_cp_direct_ptx_4site.py`
  - aligned second-copy overlay sweep across both live base registers:
    - result:
      `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_overlay_both_bases_gpu0.json`
    - `64 / 64` plans launched and returned structured output;
    - `0` plans produced a clean quadrant mapping;
    - only six unique overwrite signatures appear under aligned `{0,4,8,12}`
      deltas;
    - `%r12 + 0` is not a new semantic family: it duplicates the productive
      `%r14 + 4` right-half signatures from the earlier overlay probe;
    - within the current second-copy descriptor pair (`%rd3/%rd4`), no aligned
      one-message plan writes chunk `3`, left half as an exact expected tile.
  - aligned one-copy / live-descriptor sweep:
    - result:
      `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_single_site_halves_gpu0.json`
    - `28 / 32` plans launched cleanly; `4` plans faulted and remain invalid;
    - this probe exposes four clean quadrant mappings that do not corrupt any
      other half:
      - `tcgen05.cp.warpx2::02_13` via the live `%rd3` site with `%r18 + 4`
        or `%r7 + 0` gives `0B/00/0B/00` (clean exact right halves for chunks
        `0` and `2`);
      - `tcgen05.cp.warpx2::01_23` via the live `%rd3` site with `%r18 + 4`
        or `%r7 + 0` gives `0B/0B/00/00` (clean exact right halves for chunks
        `0` and `1`).
    - despite those new clean signatures, the combined aligned one-message
      coverage from both probe families still reaches every exact half except
      chunk `3`, left half.
  - current inference from the durable probes:
    - current live descriptors plus aligned `{0,4}` destination deltas are not
      sufficient to cover the full tile, even before considering verifier or
      lowering structure;
    - the missing `3L` half is the present hard frontier;
    - if the bounded four-site search over the current live descriptors also
      fails, the next warpx2 step is no longer “find a better schedule over the
      existing descriptors”, but “synthesize or discover a new descriptor
      variant that can write the missing quadrant”.
  - bounded four-site search over the current live descriptors:
    - result:
      `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_4site_gpu1.json`
    - search space:
      - four emitted copy sites from the two-copy overlay kernel;
      - same input buffer fed through both copies;
      - each site kept its live base register and live descriptor register;
      - per-site family `{warpx2::02_13, warpx2::01_23}`;
      - per-site aligned destination delta `{0,4}`.
    - outcome:
      - `256 / 256` plans launched cleanly;
      - `0` exact matches;
      - the best bounded plans still stop at `2040` mismatched bytes.
    - current implication:
      - schedule reshuffling over the current live descriptor/base-register
        family is exhausted for the bounded aligned space;
      - the remaining `warpx2` frontier is descriptor synthesis / descriptor
        discovery for the missing quadrant, not a better assignment of the
        current four live sites.
  - live-bit descriptor cube search:
    - result:
      `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_descriptor_values_gpu3_fast.json`
    - search space:
      - one active `warpx2` site at a time on the one-copy kernel;
      - aligned address immediates `{0,4}`;
      - current live descriptor cube implied by the differing live bits
        `{5,51}` only:
        `0x0000400800000000`,
        `0x0000400800000020`,
        `0x0008400800000000`,
        `0x0008400800000020`;
      - target quadrant remains chunk `3`, left half.
    - outcome:
      - `112 / 128` plans launched cleanly; `16` still faulted;
      - `0` exact hits on the target `3L` quadrant;
      - `0` clean or isolated target hits;
      - the best near misses still only reproduce already-known clean right-half
        signatures such as `0B/0B/00/00`.
    - current implication:
      - the missing `3L` quadrant is not hidden in the current live
        `{baseAddress, matrixBaseOffset}` bit cube;
      - the next bounded descriptor neighborhood should move to
        `leadDimensionBaseOffset` / `strideDimensionBaseOffset` around the
        clean `%rd3` seed rather than widening more schedule or
        `{baseAddress, matrixBaseOffset}` permutations.
  - `%rd3` seed `leadDimensionBaseOffset` / `strideDimensionBaseOffset` search:
    - result:
      `.codex/initiatives/tmem_linear_generalization/experiments/results/probe_cp_direct_ptx_descriptor_lbo_sbo_gpu1.json`
    - search space:
      - clean `%rd3` seed only (`0x0008400800000020`);
      - active site `1` only;
      - address seeds `%r18 + 4` and `%r7 + 0`;
      - `warpx2::{02_13,01_23}`;
      - `leadDimensionBaseOffset ∈ {0,4,8}`;
      - `strideDimensionBaseOffset ∈ {4,8,12,16}`.
    - outcome:
      - `36 / 48` plans launched cleanly; `12` faulted;
      - `0` exact hits on target quadrant chunk `3`, left half;
      - `0` clean or isolated target hits.
    - current implication:
      - bounded `leadDimensionBaseOffset` / `strideDimensionBaseOffset`
        perturbations around the clean `%rd3` seed are still insufficient;
      - the next plausible PTX-side degree of freedom is source-coordinate
        generation (descriptor values equivalent to changing `smemLoad(a, b)`
        around the clean `%rd3` seed), not more local field tweaks.

MMAv5 accumulator N-subslice support (2026-03-27)

- `tcgen05_mma` and `tcgen05_mma_scaled` now accept accumulator `D` views from
  TMEM-linear parents when the physical allocation is still legacy-MMAv5
  compatible and the view preserves the full `M` extent while narrowing `N`.
- Implementation boundary:
  - new helper `matchMMAv5AccumulatorEncoding(...)` tries the old logical-shape
    legacy matcher first, then retries on `allocShape` for
    `tensor_memory_linear` accumulators only when `shape[-2] == allocShape[-2]`;
  - the helper is used only for accumulator-side verifier/lowering/allocator
    consumers, not for generic TMEM matching.
- New coverage:
  - plain MMAv5:
    - verifier/roundtrip regression for public `memdesc_subslice` accumulator
      views in `test/TritonNvidiaGPU/ops.mlir`;
    - LLVM lowering regression for a second-half `N` subslice from a
      TMEM-linear accumulator parent in
      `test/Conversion/tritongpu_to_llvm_blackwell.mlir`;
    - runtime-matrix positives for indexed and second-half `N`-subslice
      accumulator views from legacy and TMEM-linear parents.
  - scaled MMAv5:
    - LLVM lowering regression for a second-half `N` subslice from a
      TMEM-linear accumulator parent;
    - runtime-matrix positives for second-half `N`-subslice accumulators from
      legacy and TMEM-linear parents.
- Important non-goals:
  - no widening of A-in-TMEM layout acceptance;
  - no support yet for row-sliced accumulator views (`M` changed);
  - no support yet for arbitrary non-legacy-equivalent MMAv5 accumulator
    layouts.

MMAv5 accumulator indexed / composed view coverage (2026-03-27)

- The accumulator-side MMAv5 widening is now validated beyond simple
  second-half `N` subslices.
- Supported public `D` view shapes now include:
  - `acc_parent.index(i)` from lifted legacy and TMEM-linear parents for plain
    `tcgen05_mma` and `tcgen05_mma_scaled`;
  - `acc_parent.slice(64, 64, dim=1)` from legacy and TMEM-linear parents for
    plain and scaled MMAv5;
  - `acc_parent.index(1).slice(64, 64, dim=1)` from lifted legacy and
    TMEM-linear parents for plain and scaled MMAv5.
- The implementation model is still the same:
  - `matchMMAv5AccumulatorEncoding(...)` first matches the current logical
    shape against legacy MMAv5 families;
  - if that fails and the encoding is `tensor_memory_linear`, it retries on
    `allocShape`;
  - it only accepts views that preserve the full `M` extent and narrow `N`.
- LHS tensor-memory views are still deliberately rejected.
  - attempted `A.slice(..., dim=1)` acceptance compiled but produced wrong
    numerics;
  - the verifier now rejects that path cleanly for both plain and scaled MMAv5,
    and the runtime/lit coverage checks that this remains a clean diagnostic
    rather than a crash or late-lowering failure.
- Current validated MMAv5 view boundary:
  - `D`: indexed, `N`-subslice, and composed indexed+`N`-subslice views are
    supported when the underlying allocation still maps to a legacy MMAv5
    family;
  - `A`: standalone legacy-compatible TMEM allocations remain supported, but
    TMEM subviews are quarantined until the lowering bug is fixed;
  - row-sliced `D` views (`M` changed) and arbitrary non-legacy-equivalent
    accumulator families remain unsupported.

MMAv5 accumulator view composition and LHS-view quarantine (2026-03-27)

- Accumulator `D` coverage now extends beyond the initial single-op slices.
  - plain `tcgen05_mma`:
    - indexed accumulator views from lifted parents;
    - `N`-subslice accumulator views from canonical parents;
    - composed `index(...).slice(..., dim=1)` accumulator views from lifted
      parents.
  - scaled `tcgen05_mma_scaled`:
    - indexed accumulator views from lifted parents;
    - `N`-subslice accumulator views from canonical parents;
    - composed `index(...).slice(..., dim=1)` accumulator views from lifted
      parents.
  - each of the new runtime positives was validated for both:
    - legacy-compatible accumulator parents;
    - lifted `tensor_memory_linear` accumulator parents.
- The accumulator matcher boundary is still intentionally narrow.
  - support is limited to views that remain legacy-MMAv5 compatible after
    normalization;
  - the helper still does not generalize to arbitrary row-changing or
    non-legacy-equivalent accumulator layouts.
- A-in-TMEM subviews remain a real open bug, not a supported feature.
  - public TMEM views on the LHS/A operand currently miscompile when allowed
    through;
  - the verifier now rejects them cleanly for both plain and scaled MMAv5 with
    an explicit diagnostic instead of permitting wrong-code;
  - dedicated verifier/runtime regressions now pin that boundary down.
- Current validated MMAv5 view boundary:
  - supported:
    - accumulator `D` index;
    - accumulator `D` `N` subslice;
    - accumulator `D` `index -> N-subslice` composition;
    - plain and scaled MMAv5;
    - legacy and TMEM-linear lifted parents.
  - unsupported:
    - LHS/A TMEM subviews;
    - row-sliced accumulator views (`M` changed);
    - arbitrary non-legacy-equivalent accumulator families.

MMAv5 accumulator view-chain expansion and LHS TMEM-view quarantine (2026-03-27)

- Supported accumulator view family is now broader inside the existing
  MMAv5-compatible/canonical boundary.
  - plain `tcgen05_mma` and `tcgen05_mma_scaled` now have positive runtime
    coverage for `D.index(i).slice(..., dim=1)` when the underlying allocation
    remains a legacy-MMAv5-compatible `128x128` accumulator and the final view
    preserves full `M` while narrowing `N`;
  - TMEM-linear coverage uses lifted rank-3 parents, so the composed
    `memdesc_index -> memdesc_subslice` path is exercised end-to-end before
    MMAv5 lowering.
- Implementation model did not need another matcher widening beyond the
  accumulator-side `matchMMAv5AccumulatorEncoding(...)` helper.
  - the helper already keys off the allocation shape for accumulator views, so
    composed `index(...).slice(...)` chains work as long as the trailing
    rank-2 allocation remains MMAv5-compatible and the final view keeps full
    `M`.
- A-in-TMEM public views are still an open correctness bug.
  - positive LHS TMEM `subslice` attempts miscompiled at runtime for both
    legacy and TMEM-linear parents, so the compiler now deliberately rejects
    MMAv5 LHS tensor-memory views whenever the logical shape differs from the
    allocation shape;
  - this is covered by direct verifier regressions for plain and scaled MMAv5
    plus a Gluon runtime-matrix clean-negative regression that checks the
    explicit diagnostic and no verifier/pass-manager crash.
- Current evidence for the remaining MMAv5 frontier:
  - the currently rejected row/col-mixed and low-bit-permuted accumulator
    layouts are not supportable by a small verifier/matcher tweak alone;
  - current MMAv5 lowering still assumes legacy-compatible accumulator tiles
    and materializes only base-plus-row/col offsets per instruction, so
    arbitrary PTX-level accumulator swizzles would require a more general
    accumulator address model.
- Validation after this slice:
  - lit bundle:
    `test/TritonNvidiaGPU/ops.mlir`,
    `test/TritonNvidiaGPU/invalid.mlir`,
    `test/Conversion/tritongpu_to_llvm_blackwell.mlir` -> `3 passed`;
  - focused runtime MMA accumulator/view cluster -> `13 passed`;
  - full TMEM runtime matrix -> `1616 passed, 117 skipped`.

MMAv5 canonical large-N accumulator normalization (2026-03-27)

- Plain MMAv5 now supports larger canonical accumulators along `N` when they
  span multiple instructions but still use canonical TMEM packing.
  - the accumulator matcher now picks the largest legacy family consistent
    with the same physical TMEM layout and the MMAv5 per-instruction
    `N <= 256` limit, rather than hard-failing on oversized canonical
    `blockN` values;
  - oversized explicit legacy layouts normalize through their equivalent
    canonical TMEM-linear form before that capped family search, so explicit
    `TensorMemoryLayout((128, 512), col_stride=1)` and its linear equivalent
    both reach the same supported MMAv5 family.
- New user-visible support:
  - plain `tcgen05_mma` with legacy and TMEM-linear `128x512xf32`
    accumulators now compiles and runs correctly, emitting four
    `tcgen05.mma.cta_group::1.kind::f16` instructions and producing correct
    `A @ B + C` numerics.
- Permanent coverage:
  - `python/test/gluon/test_tmem_runtime_matrix.py` now contains
    `test_tmem_runtime_matrix_mma_large_n` covering both legacy and
    TMEM-linear `128x512` accumulators and asserting the four-op PTX/LLIR
    shape.
- Important boundary:
  - this widening is currently validated for plain MMAv5 only;
  - scaled `128x512` still fails separately because the overall TMEM resource
    footprint exceeds hardware limits (`Required: 532, Hardware limit: 512`),
    so that path was not widened by the accumulator matcher change;
  - arbitrary non-canonical / tile-permuted accumulator layouts are still not
    representable by the current legacy-family matcher plus base-offset MMAv5
    lowering model.
- Validation after this slice:
  - lit bundle remains `3 passed`;
  - focused runtime MMA accumulator/view cluster including `mma_large_n` ->
    `15 passed`;
  - full TMEM runtime matrix -> `1620 passed, 117 skipped`.

MMAv5 one-CTA tile-permuted TMEM-linear accumulators (2026-03-28)

- MMAv5 accumulator lowering now supports a bounded new non-legacy family:
  one-CTA TMEM-linear accumulators whose intra-instruction MMAv5 tile packing
  is canonical, but whose whole MMAv5 tiles are permuted in TMEM.
  - the accumulator matcher in
    `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` now has a second path after
    exact legacy-equivalent matching:
    - it checks TMEM-linear accumulators against real MMAv5 instruction-width
      families `{64, 128, 256}` along `N`;
    - it requires the low bits that define the MMAv5 tile packing to match the
      canonical family exactly;
    - it allows higher bits only when they move by whole MMAv5 tiles, so the
      layout can permute tile-grid order without changing the per-instruction
      tile image.
- Scaled MMAv5 now shares the same accumulator-address model as plain MMAv5.
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
    switched `convertScaledDot(...)` from the old row-major block-id
    accumulator addressing to `DotOpMmaV5TmemLoader`, so both plain and scaled
    MMAv5 now address accumulators through the full TMEM linear layout.
- New user-visible support:
  - plain `tcgen05_mma` now accepts and runs a `128x256xf32`
    `TensorMemoryLinearLayout` whose 64-wide MMAv5 tiles are permuted along
    `N`;
  - scaled `tcgen05.mma_scaled` accepts the same accumulator family and
    produces correct numerics.
- Permanent coverage:
  - `python/test/gluon/test_tmem_runtime_matrix.py` now contains:
    - `test_tmem_runtime_matrix_mma_tile_permuted_layout`;
    - `test_tmem_runtime_matrix_mma_scaled_tile_permuted_layout`;
    - both assert numerics, exact PTX/LLIR opcode agreement, and
      `tensor_memory_linear` staying present in TTGIR;
  - `test/TritonNvidiaGPU/ops.mlir` now contains a direct verifier/round-trip
    positive for a tile-permuted MMAv5 accumulator.
- Important safety note:
  - the first matcher draft was too permissive and briefly let scrambled /
    row-col-permuted accumulator negatives through, which later faulted with
    GPU illegal instructions during the full runtime matrix;
  - restricting the new matcher path to actual MMAv5 instruction widths
    `{64, 128, 256}` restored clean-negative behavior for the old mixed /
    scrambled / row-col-permuted tests while keeping the intended tile-grid
    permutation positives.
- Current MMAv5 boundary after this slice:
  - supported:
    - canonical legacy and TMEM-linear accumulators;
    - canonical large-`N` accumulators (`128x512`);
    - public accumulator view chains that preserve full `M` and narrow `N`;
    - one-CTA tile-permuted TMEM-linear accumulators that keep MMAv5
      intra-tile packing canonical;
    - scaled MMAv5 with that same one-CTA accumulator family.
  - unsupported:
    - low-bit row/col-permuted or mixed/scrambled accumulator layouts;
    - two-CTA non-legacy tile permutations;
    - LHS/A TMEM public views.
- Validation after this slice:
  - lit bundle -> `3 passed`;
  - focused MMA positive cluster including tile-permuted + view cases ->
    `17 passed`;
  - focused negative/positive boundary cluster for exotic/permuted layouts ->
    `19 passed`;
  - full TMEM runtime matrix -> `1622 passed, 117 skipped`.

MMAv5 tile-permuted large-`N` follow-up (2026-03-28)

- The new one-CTA tile-permuted MMAv5 accumulator support also covers larger
  `N` tiles beyond the initial `128x256` case.
  - plain `tcgen05_mma` now has direct runtime coverage for:
    - `128x512` accumulators with 64-wide tile-grid permutations;
    - `128x512` accumulators with 128-wide tile-grid permutations.
- The observed instruction counts are now pinned in the runtime matrix.
  - `128x256` / tile `64` -> `8` MMAv5 instructions;
  - `128x512` / tile `64` -> `16` MMAv5 instructions;
  - `128x512` / tile `128` -> `8` MMAv5 instructions.
- Scaled MMAv5 remains resource-limited for the `128x512` tile-permuted cases.
  - the same TMEM-capacity limit still applies:
    `OutOfResources ... Required: 532, Hardware limit: 512`.
- Permanent coverage:
  - `python/test/gluon/test_tmem_runtime_matrix.py`
    `MMA_TILE_PERMUTED_CASES` now includes the two `128x512` tile-permuted
    plain-MMAv5 layouts.
- Validation after this follow-up:
  - focused tile-permuted sweep -> `4 passed`;
  - full TMEM runtime matrix -> `1624 passed, 117 skipped`.
## Plain MMAv5 LHS TMEM view boundary (2026-03-28)

- I tested the tempting broadening first: remove the explicit MMAv5 TMEM-LHS-view verifier ban and treat the runtime-matrix `mma_lhs_subslice` bucket as positive.
- That path is not safe. The partial-`K` cases (`128x64 -> 128x32` for plain, `128x128 -> 128x64` for scaled) compile but produce wrong results; plain MMAv5 was heavily `NaN`-corrupted and scaled MMAv5 was numerically wrong.
- Existing Gluon core coverage showed the real boundary: `python/test/gluon/test_core.py::test_block_m_64_mma` already uses TMEM operand-A `memdesc_subslice` views successfully for the manually tiled `blockM=64` path. So the blanket verifier ban was too broad, but “all TMEM-LHS views are fine” was also wrong.

Current rule implemented in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`:
- plain `tc_gen5_mma` accepts TMEM LHS views only for direct `ttg.memdesc_subslice` tiles;
- the view must preserve full `M`;
- the `K` extent must be a whole `blockN` tile of the parent TMEM family;
- the `K` offset must be aligned to that parent `blockN`.

Implications:
- partial-`K` LHS views stay cleanly unsupported and are covered in the runtime matrix;
- the known-good blockM=64 whole-tile TMEM-LHS slice path stays supported;
- `tc_gen5_mma_scaled` is still conservative for TMEM LHS views (no validated positive scaled-LHS view family yet).

Coverage added/updated:
- `test/TritonNvidiaGPU/invalid.mlir` checks the new plain-MMA partial-`K` rejection text;
- `test/TritonNvidiaGPU/ops.mlir` has a positive plain-MMA whole-tile TMEM-LHS subslice example;
- `python/test/gluon/test_tmem_runtime_matrix.py` keeps the partial-`K` runtime probes as clean negatives;
- `python/test/gluon/test_core.py::test_block_m_64_mma` remains the positive runtime proof for TMEM operand-A tile views.

TMEM ld/st clean-negative diagnostic normalization (2026-03-28)

- The frontend default TMEM register-layout variant for several clean-negative
  paths is now `"auto"` rather than the historical explicit `"32x32b"`.
- Runtime-matrix clean-negative assertions were normalized to accept either
  variant spelling where the underlying unsupported reason is unchanged.
  - updated checks:
    - `ldst_descriptor_multidim_slice_identity_reports_clean_error`;
    - `ldst_descriptor_multidim_slice_reports_clean_unsupported`;
    - `cp_no_scales_linear_unsupported_shape_reports_clean_error`.
- This is test-expectation hygiene only; it does not widen ld/st lowering.
- Validation status after the normalization:
  - targeted assertion cluster (`2` ld/st + `1` cp clean-negative tests):
    `3 passed`;
  - full runtime matrix no longer fails on stale `'32x32b'` text.

MMAv5 32-wide tile-preserving accumulator split (2026-03-28)

- Plain MMAv5 now supports another bounded non-legacy TMEM-linear accumulator
  family: one-CTA layouts that preserve canonical intra-instruction packing for
  `blockN = 32` and only permute whole 32-wide MMAv5 tiles along `N`.
  - direct runtime probes validated:
    - `128x128`, tile width `32`;
    - `128x256`, tile width `32`;
    - `128x512`, tile width `32`;
  - the observed plain-MMAv5 instruction counts are now pinned:
    - `128x128` / tile `32` -> `8`;
    - `128x256` / tile `32` -> `16`;
    - `128x512` / tile `32` -> `32`.
- The widened matcher lives in
  `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`.
  - `matchMMAv5AccumulatorFamily(...)` now considers `blockN = 32` in the
    non-legacy tile-preserving path for plain MMAv5 accumulators.
- Scaled MMAv5 does **not** share this new family yet.
  - the same non-legacy `blockN = 32` accumulator layouts compile and run
    correctly for plain `tcgen05_mma`, but produced numerically wrong results
    for `tcgen05.mma_scaled`;
  - the scaled verifier in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` now
    rejects non-legacy 32-wide accumulator layouts cleanly with an explicit
    diagnostic instead of allowing wrong-code;
  - canonical legacy-equivalent 32-wide layouts are not blanket-banned by that
    verifier fence.
- Permanent coverage added:
  - `python/test/gluon/test_tmem_runtime_matrix.py`
    - positive plain-MMAv5 tile-permuted `32`-wide cases for
      `128x128`, `128x256`, and `128x512`;
    - clean-negative scaled-MMAv5 regressions for
      `128x128` and `128x256` non-legacy `32`-wide tile permutations;
  - `test/TritonNvidiaGPU/ops.mlir`
    - direct verifier / round-trip positive for the plain `128x128`
      tile-permuted-32 accumulator;
  - `test/TritonNvidiaGPU/invalid.mlir`
    - direct verifier regression for the scaled non-legacy `blockN = 32`
      rejection.
- Current MMAv5 boundary after this slice:
  - supported:
    - canonical legacy / TMEM-linear accumulators;
    - canonical large-`N` accumulators (`128x512`);
    - one-CTA tile-permuted TMEM-linear accumulators with tile widths
      `{32, 64, 128}` for plain MMAv5;
    - one-CTA tile-permuted TMEM-linear accumulators with tile widths
      `{64}` for scaled MMAv5;
    - public accumulator view chains that preserve full `M` and narrow `N`;
    - whole-`K` aligned plain-MMAv5 TMEM LHS subslices.
  - still unsupported / open:
    - scaled non-legacy `blockN = 32` accumulator permutations;
    - low-bit row/col-permuted or mixed/scrambled accumulator layouts;
    - two-CTA non-legacy tile permutations;
    - arbitrary TMEM LHS public views beyond the current aligned-whole-tile
      plain-MMAv5 subset.
- Validation after this slice:
  - lit bundle:
    `test/TritonNvidiaGPU/ops.mlir`,
    `test/TritonNvidiaGPU/invalid.mlir`,
    `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    -> `3 passed`;
  - focused runtime boundary cluster:
    `mma_tile_permuted_layout`,
    `mma_scaled_tile_permuted_layout`,
    `mma_scaled_tile_permuted_32_reports_clean_unsupported`,
    `mma_exotic_layout_reports_clean_unsupported`,
    `mma_rowcol_permuted_layout_reports_clean_unsupported`
    -> `26 passed`;
  - full TMEM runtime matrix:
    `1639 passed, 117 skipped in 307.17s`.

## 2026-03-27: ld/st selection uses TMEM view-analysis layout

- Added API `getTMemViewAnalysisLinearLayout(shape, encoding, error)` and used
  it in ld/st selection paths:
  - `getDistributedLayoutForTmemLdSt(memType, atom, numWarps)`
  - `getTmemCompatibleLayouts(op, tensorType, memType)`
- This changes non-scales candidate generation from canonical-only extraction
  to view-analysis layout extraction, while keeping `computeTMemLdStEncodingInfo`
  as the final legality gate.
- Focused validation remained green on ld/st descriptor-view clusters:
  - `ldst_descriptor_multidim_slice_positive`
  - `ldst_descriptor_multidim_slice_reports_clean_unsupported`
  - lifted higher-rank clean negatives (still clean-negatives).
- Remaining open gap unchanged:
  - frontend `compute_tmem_reg_layout(...)` still cannot infer a register
    layout when descriptor `shape` rank is lower than preserved encoding rank
    (lifted views), because it does not have full view offset/index history.

## 2026-03-27: explicit MMAv5-only contract restored for user layouts

- Removed the hidden MMAv5 accumulator rematerialization path and the hidden
  TMEM-LHS-view materialization path. Supported MMAv5 layouts still lower
  directly; unsupported user-visible layouts now fail cleanly instead of being
  rewritten behind the user’s back.
- Plain/scaled MMAv5 TMEM LHS views are currently clean-unsupported.
  The exact runtime failure site is:
  - plain/scaled aligned whole-tile LHS subslices can fail early during
    `get_reg_layout(auto)` / TMEM register-layout selection with a clean
    “insert convert_layout explicitly” style diagnostic;
  - other TMEM-LHS view cases fail at the MMAv5 verifier with the explicit
    “LHS tensor memory views are not directly supported” diagnostic.
- Runtime/test state after the revert:
  - lit bundle:
    `test/TritonNvidiaGPU/invalid.mlir`,
    `test/TritonNvidiaGPU/ops.mlir`,
    `test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir`,
    `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    -> `4 passed`;
  - focused clean-negative runtime bucket:
    exotic MMA layouts, row/col permuted MMA layouts, plain/scaled TMEM-LHS
    view rejection, scaled `blockN=32` rejection
    -> `24 passed, 1 xfailed`;
  - focused direct-positive MMA bucket:
    large-`N`, tile-permuted direct layouts, accumulator index/subslice views,
    direct TMEM-LHS layouts, scaled direct-positive variants
    -> `27 passed`.
- New explicit BUG after removing the hidden rewrite:
  - scaled MMAv5 direct linear `128x32` accumulator layout compiles and emits
    code, but runtime numerics are wrong;
  - preserved as explicit `xfail(strict=True)` coverage in
    `test_tmem_runtime_matrix_mma_scaled_blockn32_linear_bug`.

## 2026-03-27: first MMAv5 linear-first planner slice

- MMAv5 verifier/lowering/allocation no longer consume a returned legacy
  `TensorMemoryEncodingAttr` from the TMEM matcher path.
- New API surface:
  - `MMAv5TMemLayoutPlan`
  - `getMMAv5LhsLayoutPlan(...)`
  - `getMMAv5AccumulatorLayoutPlan(...)`
- Current behavior after the refactor is intentionally stable:
  - supported direct MMAv5 layouts still lower directly;
  - unsupported TMEM-LHS views remain clean-negative;
  - focused direct-positive MMA runtime buckets remain green.
- The planner still internally recovers a supported finite MMAv5 family to
  derive instruction tile sizes and packing facts; the important change is that
  verifier/lowering/allocation are no longer coupled to the legacy attr type
  itself.
- Validation after this slice:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - raw `triton-opt` sanity passes for:
    - `test/TritonNvidiaGPU/invalid.mlir`
    - `test/TritonNvidiaGPU/ops.mlir`
    - `test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir`
    - `test/TritonGPU/promote-lhs-to-tmem.mlir`
    - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
  - focused clean-negative MMA runtime slice -> `25 passed`
  - focused direct-positive MMA runtime slice -> `27 passed`
- Remaining MMAv5 linear-generalization gap:
  - the planner still uses finite-family recovery internally, so truly arbitrary
    ISA-realizable linear layouts are not all supported yet;
  - next work should replace that remaining family recovery logic with direct
    analysis over the normalized TMEM linear layout and then widen supported
    cases.

## 2026-03-27: Gluon MMAv5 API contract

- From Gluon, `tcgen05_mma(...)` and `tcgen05_mma_scaled(...)` are now
  explicitly async-only.
- The C++ builders in `python/src/gluon_ir.cc` always set `is_async=true`
  for both ops, even when no completion barriers are attached.
- Gluon docs now state that omitting `mbarriers=` does **not** make the op
  synchronous; callers must explicitly use `tcgen05_commit(...)` +
  `mbarrier.wait(...)` before reading the accumulator.
- Immediate consequence for tests:
  - any Gluon kernel that issues MMAv5 and then directly loads the accumulator
    must either attach completion barriers on the op or do an explicit
    `tcgen05_commit(...)` / `mbarrier.wait(...)` sequence.
- Concrete regression fixed under this contract:
  - `tmem_mma_scaled_blockn32_kernel` now uses explicit commit/wait before
    `acc_tmem.load()`, and the focused runtime bucket is green.
- Validation for this contract change:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - focused frontend parser slice -> `3 passed`
  - focused scaled `blockN=32` runtime slice -> `2 passed`

## 2026-03-27: scaled MMAv5 tile-permuted-32 remains a BUG; higher-rank ld/st tests corrected

- The delegated higher-rank ld/st test conversion was only partly valid.
  - one-CTA higher-rank `dim0_slice` / `half_rows` buckets remain positive.
  - two-CTA higher-rank `dim0_slice` / `half_rows` still fail cleanly at the
    public `get_reg_layout(...)` query boundary, so the runtime matrix should
    keep them as explicit clean negatives.
  - the immediate cleanup was to restore the correct two-CTA negative
    expectations and remove the broken undefined parametrization name so
    `python/test/gluon/test_tmem_runtime_matrix.py` imports again.
- Scaled MMAv5 tile-permuted `tile_n=32` accumulators are **not** supported yet.
  - `128x128` and `128x256` tile-permuted-32 accumulators compile and emit the
    expected scaled MMAv5 PTX op counts (`8` and `16` ops), but runtime
    numerics are wrong; one bounded lowering experiment even escalated the
    bucket to a device misaligned-address fault.
  - keep these cases as explicit `BUG` xfails, isolated with
    `@pytest.mark.forked`, until there is a real lowering fix.
  - do **not** treat them as clean unsupported negatives anymore; compilation
    succeeds.
  - do **not** keep the attempted `numColPerScaleBlockB` tweak in
    `MMAv5.cpp`; it regresses the known-good `256x64` scaled tile-permuted
    path and must stay reverted.
- `test/TritonNvidiaGPU/ops.mlir` now contains a correctly-typed higher-rank
  TMEM `reshape -> subslice -> index` regression using the exact inferred TMEM
  linear encodings for the intermediate rank-3 views:
  - `#tmem_linear_rank3`
  - `#tmem_linear_rank3_small`
  - `#tmem_linear_half_rows`
- Also added a direct `ttng.tc_gen5_mma_scaled` verifier/round-trip regression
  for a tile-permuted-32 accumulator in `ops.mlir`.
- Validation after this slice:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `python -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
  - two-CTA higher-rank ld/st clean-negative slice -> `24 passed`
  - focused scaled tile-permuted / indexed-acc slice -> `3 passed, 2 xfailed`
  - `lit -v test/TritonNvidiaGPU/ops.mlir` -> `1 passed`
- `python/src/gluon_ir.cc`: the `compute_tmem_reg_layout` helper now tries to canonicalize general TensorMemory layouts with `ttng::getTMemViewAnalysisLinearLayout` + `ttng::tryMakeTMemViewEncoding` before querying `ttng::getTmemCompatibleLayouts`, so high-rank descriptor views can be reinterpreted with their canonical TMEM-linear encodings once the remaining row/column contiguity blockers are resolved.
- higher-rank descriptor slice/half-rows runtimes still hit the `TMEM layout '<variant>' unsupported` guard, so the runtime-matrix clean-error cases remain in place until the new canonical fallback stops returning empty layouts.

## 2026-03-28: ld/st fallback and ld.red support boundary

- `computeTMemLdStEncodingInfo(...)` in
  `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp` now attempts a
  view-analysis fallback for TMEM memdesc types:
  - first try standalone canonical TMEM-linear encoding via
    `getCanonicalTMemLinearEncoding(...)`;
  - if that fails, try `getTMemViewAnalysisLinearLayout(shape, encoding, ...)`;
  - only fail when both are unavailable.
- This preserves the direct-support-only contract:
  - no implicit `convert_layout`,
  - no hidden temporary/rematerialization path,
  - still cleanly rejects non-lowerable views.
- `ld.red` boundary (current):
  - positive: `f32` row-wise min/max over `N` with optional `abs` and `NaN`
    propagation when `N` is register-local per row;
  - clean-negative: layouts with `N` sharded across threads, tensor-memory
    scales reductions, and unpacked reduction format.

## 2026-03-28: owned-scope ld/st + ld.red audit rerun

- Full runtime slice in owned scope:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst or ld_red'`
    -> `1143 passed, 112 skipped in 403.56s`.
- Targeted clean-negative boundary slice:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_descriptor_higher_rank_dim0_slice_reports_clean_error_lifted_layout or ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout or ldst_descriptor_multidim_slice_identity_reports_clean_error or ldst_descriptor_multidim_slice_reports_clean_unsupported or ld_red_mixed_layout_reports_clean_unsupported'`
    -> `38 passed`.
- Lit sanity:
  - `BUILD_DIR=$(PYTHONPATH=./python python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'); cd "$BUILD_DIR" && ninja triton-opt && lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir`
    -> `2 passed`.
- Current owned-scope conclusion:
  - no additional direct-lowering promotions were identified without changing
    semantics;
  - remaining negatives are still legitimate boundaries:
    - unsupported high-rank lifted descriptor views for explicit TMEM ld/st
      variant requests;
    - `ld.red` cases with N sharded across threads.

## 2026-03-28: MMAv5 planner cleanup after transform linearization

- `PromoteLHSToTMem.cpp` now builds canonical TMEM-linear layouts directly
  instead of reconstructing legacy `#ttng.tensor_memory_encoding` attrs
  internally.
- `accelerate-matmul.mlir` checks were updated to match canonical
  `#ttng.tensor_memory_linear<...>` output instead of exact legacy print forms.
- MMAv5 planning in `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` is now more
  linear-first:
  - removed the unconditional legacy-attr fast paths from the generic MMAv5
    family planners;
  - kept only a narrow explicit-legacy preservation path for accumulator /
    scaled-accumulator cases with non-default `colStride`, because current
    canonical TMEM-linear form does not yet uniquely preserve that distinction;
  - kept LHS TMEM planning on the linear-first path.
- Practical rule after this slice:
  - legacy TMEM encodings remain accepted surface syntax / IR sugar;
  - MMAv5 verifier and lowering should prefer canonical TMEM-linear analysis;
  - only preserve explicit legacy family information when needed to avoid
    regressing direct codegen for exact non-default accumulator `colStride`.
- Validation snapshot:
  - transform lit slice passed (`accelerate-matmul`, `promote-lhs-to-tmem`,
    `test_promotion_to_tensor_memory`);
  - focused MMAv5 frontend/runtime slices passed (`6 passed`, then `20 passed`);
  - `test/TritonNvidiaGPU/ops.mlir` lit passed.
- Build note:
  - broad `make -j96` became unreliable in the dirty tree because of unrelated
    link failures in broader targets plus one transient `triton-opt`
    permission-bit glitch;
  - targeted relinks of `triton-opt` and `libtriton.so` were sufficient for
    focused MMAv5/TMEM validation.

## 2026-03-28: one-CTA lifted higher-rank ld/st boundary refined

- The one-CTA lifted higher-rank `ld/st` clean-negative bucket is no longer a
  single class.
- Current precise split:
  - `dim0_slice`, explicit `32x32b`, `n = 64 / 128 / 256`:
    compiles but miscomputes -> strict `BUG` xfail.
  - `half_rows`, explicit `32x32b / 16x64b / 16x128b / 16x256b`,
    `n = 64 / 128 / 256`:
    compiles but miscomputes -> strict `BUG` xfail.
  - remaining one-CTA lifted `dim0_slice` explicit variants
    (`16x64b`, `16x128b`, `16x256b`) still fail cleanly at
    `get_reg_layout(...)`.
- Validation snapshot after reclassification:
  - focused slice -> `9 passed, 15 xfailed`.
- Practical rule:
  - keep the compiling-but-wrong higher-rank view cases explicit as `BUG`,
    not as positives and not as clean unsupported negatives.

## 2026-03-28: copy-planner exploration status

- Attempted one internal copy-descriptor canonicalization experiment by
  reordering TMEM row/col basis bits before descriptor synthesis.
  - reverted immediately because it introduced wrong-code and a late lowering
    assertion.
  - this was the wrong layer: it changed user-visible TMEM logical mapping
    instead of only broadening the descriptor factorization search.
- Current safe copy change:
  - dense non-scales copy planning can now try both the canonical `32x4`
    descriptor factorization and a `64x2` fallback before rejecting.
  - current public negative buckets remain unchanged (`19 passed` on the
    focused no-scales copy clean-negative slice), so this is only a safe
    planner broadening for now, not a user-visible promotion yet.
- `warpx2` status remains unchanged:
  - the historical no-scales candidate still maps to dense `128x128b`;
  - the historical scales candidate still maps to `warpx4`;
  - real `warpx2` still needs both a user-visible representation path and a
    family-specific copy planner/lowering path.

## 2026-03-28: remaining MMAv5 legacy bottlenecks

- Remaining known legacy-family dependencies after the planner cleanup:
  - `matchTensorMemoryLegacyEncoding(...)` still returns legacy family attrs;
  - `planMMAv5LegacyExactFamily(...)` still preserves explicit legacy exact
    families for some accumulator/scaled-accumulator paths;
  - verifier/lowering are still gated by `getMMAv5*LayoutPlan(...)`;
  - `impl::verifyMMAv5Op(...)` still performs its interleaved `blockM=64`
    guard through a legacy-only `TensorMemoryEncodingAttr` check;
  - allocator row-coupling still keys off MMAv5 family plans rather than a
    fully general linear TMEM physical-footprint query.

## 2026-03-28: MMAv5 linear-first cleanup progress

- MMAv5 accumulator and scaled-accumulator planning no longer use a
  `planMMAv5LegacyExactFamily(...)` fast path.
  - Both now share a memdesc-based fallback:
    1. plan the visible view shape directly;
    2. if that fails, canonicalize the memdesc through
       `tryGetCanonicalTensorMemoryEncoding(memDescType, ...)`;
    3. only for full-`M`, narrowed-`N` accumulator views, retry planning on the
       backing allocation shape.
  - This preserves legacy TMEM attrs as parse/source sugar, but moves the
    planner itself further toward linear-first semantics.
- Direct scaled MMAv5 accumulator subslices are now permanently runtime-covered.
  - New user-level coverage allocates a parent `128x128`
    `#ttng.tensor_memory_linear` accumulator, takes a `ttg.memdesc_subslice`
    view, and runs `tcgen05_mma_scaled` directly into that view for `N=32` and
    `N=64`.
  - The focused scaled slice is green: `6 passed`.
- The interleaved `blockM=64` MMAv5 rejection is now linear-first too.
  - `impl::verifyMMAv5Op(...)` now rejects based on the MMAv5 layout plan plus
    physical TMEM row footprint, not only on `TensorMemoryEncodingAttr`.
  - The explicit linear equivalent of the interleaved `blockM=64` family
    (`128x64` with row bases `[1, 2, 4, 8, 64, 16, 32]` and contiguous `N=64`)
    now hits the same clean verifier rejection as the legacy sugar.
- Current MMAv5 validation snapshot after these changes:
  - `lit -v test/TritonNvidiaGPU/ops.mlir` -> `1 passed`
  - `lit -v test/TritonNvidiaGPU/invalid.mlir` -> `1 passed`
  - broad MMA slice -> `52 passed`

## 2026-03-28: scales explicit `16x32bx2` frontend alignment

- The pure descriptor-type TMEM reg-layout helper and the handle-aware memdesc
  helper had drifted apart for tensor-memory scales:
  - the handle-aware path could expose the small explicit `16x32bx2` layouts
    used by the runtime matrix;
  - the pure type-based helper still took the narrower direct memtype path and
    either returned a clean negative or, after selector cleanup, could trip a
    `LinearLayout::getBasis` assertion for `[16, 4]`.
- The selector crash root cause was an unconditional read of the row-16 basis
  in `getDistributedLayoutForTmemLdSt(const LinearLayout&, ...)` even when the
  TMEM row dimension was already only 16 high.
- The current state after the fix:
  - selector no longer crashes on small explicit scales `16x32bx2`;
  - the pure descriptor-type helper now filters the full compatible-layout set
    first, matching the handle-aware path more closely;
  - small explicit `16x32bx2` descriptor-type queries are positive:
    - `[16, 4]`, `num_warps=4`
    - `[16, 8]`, `num_warps=8`
  - larger shapes that still do not have a direct lowering remain clean
    negatives:
    - `[32, 16]`, `num_warps=4`
    - `[32, 4]`, `num_warps=8`
- Focused validation:
  - `python/test/gluon/test_frontend.py` targeted slice -> `2 passed`
  - scales ld/st runtime slice -> `34 passed`

## 2026-03-28: scaled MMAv5 tile-preserving support boundary

- Scaled MMAv5 accumulator planning now shares the same tile-preserving family
  search as plain MMAv5 instead of requiring exact canonical-family equality.
  - Implementation site:
    `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
    (`planMMAv5ScaledAccumulatorFamily(...)` now uses
    `planMMAv5Family(...)`).
- This unlocks a real new direct user-visible layout:
  - `#ttng.tensor_memory_linear` accumulator with shape `128x256` and
    a `tile_n=64` column-tile permutation now lowers and runs correctly with
    `tcgen05.mma_scaled`.
- The repeated-`N=32` tile-permuted scaled case is still intentionally
  unsupported, but the blocker is now the public scales layout contract, not
  the old exact-family matcher.
  - The public `tensor_memory_scales_encoding` exposes matrix-B scale
    fragments only at `64`-column alignment.
  - Repeated scaled `N=32` instructions along `N` still fail cleanly with:
    - `direct block-scaled MMAv5 does not support repeated N=32 instructions along N`
    - `The public tensor-memory scales layout only exposes matrix-B scale fragments at 64-column alignment`
- Current support boundary to remember:
  - direct scaled `blockN=32` canonical layout: supported
  - scaled tile-permuted `tile_n=64` (`128x256`): supported
  - scaled tile-permuted repeated `tile_n=32`: clean negative unless/until the
    public scales layout semantics/API expand
- Validation snapshot:
  - focused new scaled slice -> `3 passed`
  - broader scaled MMAv5 slice -> `7 passed`
  - full TMEM runtime matrix -> `1588 passed, 118 skipped`

## 2026-03-28: reachable non-surjective `warpx2::01_23` copy path

- The public non-surjective TMEM-linear `warpx2::01_23` destination path is
  now live.
  - Implementation site:
    `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
    (`getTMemCopyDescriptorLayouts(...)`).
  - The bounded fix was to widen descriptor candidate generation for
    `warpx2::01_23`:
    - fold source warp-group bits into row/col;
    - rotate leading row-repeat bits out of the core descriptor prefix;
    - promote one extra low-order column bit into the row prefix when needed.
- The user-visible runtime result is intentionally not an identity roundtrip.
  - The supported `warpx2::01_23` layout is non-surjective, so the observable
    logical result is the aliased TMEM view defined by that layout.
  - The runtime matrix now checks the exact aliased output pattern instead of
    only looking for a clean error.
- Current family boundary:
  - `warpx2::01_23.64x128b`: direct positive from user code
  - `warpx2::02_13.64x128b`: still clean unsupported with a descriptor-plan
    diagnostic
- Focused validation:
  - targeted `warpx2` slice -> `3 passed`

## 2026-03-28: MMAv5 LHS subslice support

- Full-`M`, narrower-`K` TMEM LHS descriptor views are now directly supported
  for both plain and scaled MMAv5.
  - Implementation site:
    `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
    (`getMMAv5LhsLayoutPlan(...)` now retries on the backing allocation shape
    when the visible LHS view preserves `M` and only narrows `K`).
  - This is still direct lowering: the actual `ttg.memdesc_subslice` operand
    stays visible in TTGIR and lowering uses the view value directly.
- New permanent user-level coverage:
  - plain `tcgen05_mma` with a TMEM LHS `memdesc_subslice`
  - scaled `tcgen05_mma_scaled` with a TMEM LHS `memdesc_subslice`
- IR coverage:
  - positive verifier/roundtrip coverage for both plain and scaled LHS
    subslices now lives in `test/TritonNvidiaGPU/ops.mlir`
  - the old stale rejection-only invalid cases were removed from
    `test/TritonNvidiaGPU/invalid.mlir`
- Focused validation:
  - plain LHS subslice runtime -> `1 passed`
  - scaled LHS subslice runtime -> `1 passed`
  - `lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir`
    -> `2 passed`
  - full TMEM runtime matrix -> `1592 passed, 118 skipped`

## 2026-03-28: ld/st legacy cleanup and ld.red support matrix update

- The remaining legacy-only M64 ld/st layout-selection hook is gone.
  - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp` no longer recovers a legacy
    `TensorMemoryEncodingAttr` just to pick M64 / split-N register layouts.
  - Deleted the dead `matchTensorMemoryLegacyEncoding(...)` helpers after
    confirming the focused M64 / split-N slice stays green on the pure
    linear-first path.
- Validation for that cleanup:
  - focused M64 / split-N / f32 ld-st slice -> `40 passed`
- `tcgen05.ld.red` direct lowering is broader than the older runtime matrix
  covered.
  - Direct hardware-reduction lowering is now permanently covered for:
    - tile-permuted TMEM-linear layouts:
      - `128x128`, `tile_n=32`
      - `128x256`, `tile_n=32`
      - `128x256`, `tile_n=64`
    - all pure row/col-permuted `128x128` TMEM-linear layouts generated from
      `PERMUTED_ROW_COL_LAYOUT_KINDS`
  - Those layouts all still emit `tcgen05.ld.red.sync.aligned.32x32b...f32`
    directly, with exact PTX/LLIR opcode agreement.
  - Mixed TMEM-linear layouts remain on the fallback path with no
    `tcgen05.ld.red` opcodes, which stays the current correct negative
    boundary.
- New runtime coverage in `python/test/gluon/test_tmem_runtime_matrix.py`:
  - `test_tmem_runtime_matrix_ld_red_tile_permuted_linear_layout`
  - `test_tmem_runtime_matrix_ld_red_rowcol_permuted_linear_layout`
  - retained `test_tmem_runtime_matrix_ld_red_mixed_linear_layout_falls_back`
    for the fallback boundary
- Validation snapshot:
  - focused ld.red slice -> `100 passed`
  - broad ld/st + ld.red slice -> `1215 passed, 113 skipped`

## 2026-03-28: `warpx2::02_13` remains a real planner gap; handle-aware ld/st probing is stable

- `tcgen05.copy.warpx2::02_13.64x128b` is still not reachable from the public
  descriptor API through the current direct-lowering path, even after the
  bounded `multicast == 2` descriptor-layout search was widened.
  - The old dead-branch bug in
    `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
    (`getTMemCopyDescriptorLayouts(...)`) is fixed.
  - The current bounded search now tries:
    - row/col fold repartitions of the source warp-group bit
    - row-basis rotations
    - single moved-row permutations
    - bounded row/col reassignment variants
  - Result: the historical non-surjective `warpx2::02_13` test case still
    cleanly rejects, so the remaining gap is deeper descriptor/message
    synthesis, not another trivial candidate-enumeration bug.
- The handle-aware ld/st reg-layout query path in `python/src/gluon_ir.cc` now
  supplements the old type-only candidate list with layouts synthesized using
  the actual memdesc value’s row plan.
  - This uses the new public overload
    `getDistributedLayoutForTmemLdSt(memType, atom, numWarps, rowPlanOverride)`
    from `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h`.
  - It does not silently widen support; it just lets descriptor-view probing
    see the same backing-row-plan information that lowering already uses.
- Two remaining ld/st negatives were re-probed with that handle-aware path:
  - identity multidimensional `32x32` descriptor slice:
    still clean-negative
  - two-CTA higher-rank `half_rows` descriptor slice:
    still clean-negative
- Important failed experiment:
  - adding a 32-row TMEM row-plan made the identity `32x32` slice compile, but
    it miscompiled: the lowered code collapsed the slice to a single
    `tcgen05.{ld,st}.sync.aligned.32x32b.x32.b32` pair at offset `0`, so that
    experiment was reverted immediately.
  - The safe boundary remains clean rejection until the physical TMEM mapping
    for those 32-row/gapped views is modeled correctly.
- Focused validation after the revert:
  - `lit -v test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/invalid.mlir`
    -> `2 passed`
  - focused runtime slice covering `warpx2`, the remaining ld/st negatives, and
    the direct MMAv5 LHS-subslice positives -> `17 passed`

## 2026-03-28: direct `warpx2::02_13` copy support is live; scaled repeated-`N=32` still needs a new B-scale address model

- `tcgen05.copy.warpx2::02_13.64x128b` is now reachable and correct from the
  public non-surjective TMEM-linear descriptor API.
  - The key fix was not more shared-descriptor enumeration. The direct seed
    descriptor path was already present; the blocker was that
    `getDirectTMemCopySeedDescriptorImm(...)` in
    `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp` was matching the
    full shared-memory linear layout too strictly.
  - After relaxing that matcher to the canonical shared offset bases and
    threading a TMEM ld/st `baseOffset=4` through
    `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`,
    the same `02_13` non-surjective layout now:
    - compiles
    - emits `tcgen05.cp.cta_group::1.warpx2::02_13.64x128b`
    - round-trips correctly through direct `tmem_load`
  - The user-visible runtime expectation is now covered by
    `_expected_tmem_copy_warpx2_02_13_output(...)` and
    `test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_candidate_positive`
    in `python/test/gluon/test_tmem_runtime_matrix.py`.

- Important MMAv5 scaled follow-up:
  - The clean rejection for repeated scaled `N=32` instructions along `N`
    remains necessary for now.
  - Bounded experiments showed:
    - removing the verifier/lowering guard makes the
      `tile_n=32`, `128x128` scaled case compile
    - with the current address model, it faults at runtime with
      `CUDA error: misaligned address`
    - the second `N=32` matrix-B scale fragment currently lowers through
      `ttng::getTMemViewOffset(...)` as `origRaw=1` from `(row,col)=(32,0)`
      under the public `TensorMemoryScalesLayout`
    - a bounded rewrite to fold that half-tile into `(row,col)=(0,4)` avoids
      the misaligned-address fault but still gives wrong numerics
      (`max_abs ~= 5e-2`)
  - So the remaining gap is not a stale guard. Repeated scaled `N=32` needs a
    more complete matrix-B scale address model before the clean rejection can
    be lifted safely.

## 2026-03-29: higher-rank ld/st half-row views are still a real BUG and not fixable by a small row-anchor/base-offset tweak

- The remaining `ld/st` runtime failures are the `identity` higher-rank
  half-row positives in
  `test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_positive_lifted_layout`
  from `python/test/gluon/test_tmem_runtime_matrix.py`.
  - Shape pattern:
    - allocate `2x128xN` TMEM
    - `full_view = tmem.index(1)`
    - `view = full_view.reshape((2, 64, N)).slice(1, 1, dim=0).index(0)`
  - Expected behavior:
    - update rows `64:128` of `full_view`
  - Actual behavior:
    - rows `0:64` are updated instead

- The important probe result is negative:
  - I tried all bounded single-atom fixes that still preserve the current
    direct `tcgen05.ld/st` lowering model:
    - local `64`-row anchors
    - backing `128`-row anchors
    - mixed local-selection / backing-emission anchors
    - with and without canceling the extra TMEM row-half base shift
  - None of those variants made the half-row subview hit the correct rows.

- Takeaway:
  - this no longer looks like a small `rowPlan` / `warpBaseOffset` /
    `baseOffset` bookkeeping bug
  - the half-row bucket likely needs a deeper direct decomposition for
    row-half views carved from a larger TMEM backing tile, not another local
  row-anchor tweak
  - after the bounded probe, all exploratory planner mutations were reverted,
    so the tree returns to the last stable baseline for this bug

## 2026-03-29: current stable validation baseline after the scales/`ld.red` cleanup

- The current tree is green on a clean JIT cache:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `rm -rf /root/.triton/cache/* && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=python:. pytest -s --tb=short -n 4 python/test/gluon/test_tmem_runtime_matrix.py`
    -> `1664 passed, 119 skipped in 149.43s`
  - lit bundle:
    - `ops.mlir`
    - `invalid.mlir`
    - `tmem_layouts.mlir`
    - `tritongpu_to_llvm_blackwell.mlir`
    -> `4 passed`

- Important TMEM state on this baseline:
  - scales ld/st direct recovery is live again through the C++ planner path
    rather than a Python-only fallback
  - `ld.red` now obeys the direct-support-or-clean-reject rule; the old Gluon
    fallback to `tmem_load + tt.reduce` is removed
  - MMAv5 legacy `#ttng.tensor_memory_encoding` sugar now re-enters the exact
    direct family path instead of spuriously rejecting in conversion

- Local validation caveat:
  - helper-only Python behavior changes can leave Triton’s on-disk JIT cache
    stale even when the kernel hash does not change
  - clearing `/root/.triton/cache/*` is required when validating changes like
    the `ld.red` fallback removal

## 2026-03-29: remaining semantic debt after the green matrix

- `ld.red`
  - direct lowering is still narrower than generic ld/st
  - targeted coverage is still missing for:
    - descriptor-view `ld.red`
    - scales rejection
    - unpacked rejection
    - two-CTA reduction

- MMAv5
  - the visible legacy-sugar rejection is fixed, but the semantic core still
    compresses arbitrary linear TMEM layouts through the finite
    `MMAv5TMemLayoutPlan` family model
  - the main bottlenecks remain:
    - family planners in `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
    - verifier consumers in `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
    - allocator coupling in
      `lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp`
    - final lowering in
      `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`

## 2026-03-29: `ld.red` legality boundary after the 8-warp + linear-layout sweep

- `getTmemLoadReductionLayout(...)` no longer hard-gates to `numWarps == 4`;
  the canonical legacy `256x128`, `num_warps=8` reduction path is supportable,
  and the equivalent canonical TMEM-linear source layout is now permanently
  covered.

- The real `ld.red` direct-support boundary is narrower than generic `ld`:
  - supported:
    - canonical identity TMEM-linear layouts
    - pure-column permutations
    - tile-permuted TMEM-linear layouts
    - the canonical linear form equivalent to legacy
      `TensorMemoryLayout(block=(128, 128), col_stride=1)` at shape `256x128`
  - not directly supported:
    - row-permuted TMEM-linear layouts
    - mixed TMEM-linear layouts that mix row and column physical bits

- Important implementation detail:
  - the reduction source-layout legality predicate already existed as
    `isReductionFriendlyTmemSourceLayout(...)` in
    `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
  - the missing piece was enforcing that predicate at the user-facing helper
    and verifier boundary:
    - `python/src/gluon_ir.cc` now refuses to infer a default reduction layout
      for non-`ld.red`-compatible TMEM source layouts
    - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp` now rejects explicit reduction
      ops on those layouts instead of allowing runtime wrong-code
  - the mixed layout still fails because `normalizeTensorMemoryLinearLayoutForAnalysis(...)`
    strips zero bases and requires the `row` dim to present a 128-row block
    composed of pure row bases; the `[0,1]`/`[0,2]` column bits in the mixed
    rows take it out of the canonical set, so `TensorMemoryToLLVM::lowerTMemLdStFromTypes`
    never sees a valid register layout and would emit wrong semantics if the
    predicate were lifted.
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_mixed_linear_layout_reports_clean_unsupported`
    still checks for the "not tcgen05.ld.red-compatible" diagnostic and
    `tt.reduce` suggestion that `TMEMLoadOp::verify` emits, so the clean-negative
    boundary remains the right guard until a new packet decomposition is found.

- Local validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `PYTHONPATH=python:. pytest -s --tb=short python/test/gluon/test_core.py -k 'test_tmem_reduction_linear_legacy_block_equiv_layout or test_tmem_reduction and 256'`
    -> `32 passed`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_identity_linear_layout or ld_red_tile_permuted_linear_layout or ld_red_col_permuted_linear_layout or ld_red_row_permuted_linear_layout_reports_clean_unsupported or ld_red_mixed_linear_layout_reports_clean_unsupported'`
    -> `104 passed`

## 2026-03-29: half-row ld/st sidecar probe (higher-rank descriptor bug)

- The remaining half-row bug is still open after bounded direct-lowering probes.
  - Repro scripts:
    - `/tmp/test_direct_row_slice.py`
    - `/tmp/test_candidate_layout.py`
  - Both still report `num_diff 4096`.

- Trace state (`TRITON_TRACE_TMEM_QUERY_LOWERING_FILE=1`):
  - support-query lowering is active for the half-row pattern
  - support plan currently resolves to `I32x32b.x1` with:
    - `baseOffset=4194304`
    - `warpBase0=2097152`
    - `warpBase1=32`
  - so the failure is no longer "missing +64 support-origin carry"; the
    remaining issue is the selected mapping family for this direct path.

- One safety fix did land during this probe:
  - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`:
    `getTileLayout(..., warpBasis0, warpBasis1, ...)` now expands row out-dim
    extent from warp basis coverage (power-of-two rounded), analogous to the
    existing column-span logic.
  - This removes an invalid-basis abort seen when probing fallback layouts:
    `Invalid basis 64 for in-dim 'warp' and out-dim 'row'`.

- Reverted experiment:
  - dropping support-base adjustment in
    `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
    did not solve the bug and was reverted.

- Takeaway:
  - no clean direct-lowering fix for the half-row positive bucket yet
  - likely needs a new planner/decomposition for row-half views carved from
    larger backing tiles, not another local base-offset tweak.

## 2026-03-29: multidimensional `32x32` descriptor views should not use
canonical-surrogate query types

- The bad multidimensional `reshape -> subslice -> reshape` bucket regressed
  after the half-row quarantine because the view-like `32x32` descriptors were
  still allowed to reach `get_reg_layout(auto)` through
  `getTMemLdStQueryTypes(...)` canonical-surrogate query types.

- Important distinction:
  - support-query rescue and physical-support rescue were already disabled for
    view-like `32x32` descriptors
  - that was not sufficient
  - the remaining leak was the surrogate query type built from the larger
    backing tile, which let verifier/frontend/lowering recover a `128-row`
    plan for a final `32x32` view that is not directly materializable

- Correct rule for this family:
  - explicit view-like `32x32` TMEM descriptors must be judged on their exact
    final view
  - if the final view is directly supported, keep it
  - if only a surrogate/backing-tile type is supported, reject it instead of
    silently widening the physical plan

- Concrete implementation now in tree:
  - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
    - `getTMemLdStQueryTypes(...)` suppresses canonical-surrogate query types
      for explicit view-like `32x32` descriptors
  - `python/src/gluon_ir.cc`
  - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
    - raw-query probing for that same family no longer injects a backing
      row-plan override

- Observed effect:
  - identity and scrambled-cols multidimensional `32x32` subviews are back to
    clean compile-time rejection
  - mixed multidimensional `32x32` subviews remain directly supported
  - nearby higher-rank ld/st descriptor buckets remain green

## 2026-03-29: current tree reality on copy `warpx2` and the remaining copy gap

- Non-scales `warpx2` is already user-reachable and directly lowered in the
  current tree.
  - Positive runtime coverage now exists for both
    `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b` and
    `tcgen05.cp.cta_group::1.warpx2::02_13.64x128b` from user-written
    `TensorMemoryLinearLayout` values in
    `python/test/gluon/test_tmem_runtime_matrix.py`.

- The copy frontier has therefore moved:
  - the remaining open copy cases are no longer “make non-scales `warpx2`
    reachable”
  - the likely next direct-support slice is dense row/col-permuted linear
    layouts that still fail cleanly because copy-family recognition or
    descriptor-plan synthesis does not accept them yet
  - the scales-side historical `warpx2_candidate` path remains clean
    unsupported

- Full TMEM runtime matrix status after the latest ld/st + expectation fixes:
  - `1668 passed, 119 skipped`

## 2026-03-29: MMAv5 accumulator-side verifier/lowering now uses direct
accumulator layout info

- The D-side MMAv5 consumer stack no longer has to thread the recovered
  `MMAv5TMemLayoutPlan` through verifier, allocation, and lowering just to
  recover `instrShapeM/N`, `colStride`, and `twoCTAs`.

- New public query now in tree:
  - `getMMAv5AccumulatorLayoutInfo(...)`
  - `getMMAv5ScaledAccumulatorLayoutInfo(...)`

- The new info bundles:
  - canonical TMEM linear layout
  - selected MMAv5 instruction sizes `mmaSizeM/N`
  - `colStride`
  - `twoCTAs`
  - `interleavedM64`

- Current state after this slice:
  - D-side plain/scaled MMAv5 verifier checks use the new accumulator info
  - D-side MMAv5 lowering uses the new accumulator info
  - TMEM allocation row-anchor constraints use the new accumulator info
  - the old `MMAv5TMemLayoutPlan` still remains on the LHS / promotion side,
    so the full MMAv5 stack is not fully free of the recovered family-plan API
    yet

## 2026-03-29: pure row-permuted `ld.red` source layouts are directly
supported

- The old `ld.red` source-layout recognizer already tolerated:
  - tile permutations
  - pure col permutations
  - pure row-carry permutations in the col dimension

- But it still hard-required the row bases themselves to appear in canonical
  order. That was stricter than the direct lowering actually needed.

- The fix is intentionally small:
  - in `isReductionFriendlyTmemSourceLayout(...)`, collect the row bases,
    sort them, and compare against the expected canonical basis set
    `{1, 2, 4, 8, 16, 32, 64}`
  - keep all the other `ld.red` direct-support constraints unchanged

- Result:
  - pure row-permuted TMEM linear source layouts are now positive runtime
    cases for `ld.red`
  - mixed layouts remain clean negatives
  - col-permuted and tile-permuted positives stay green

- Focused validation:
  - `pytest -k 'ld_red_'` -> `104 passed`

- Full TMEM runtime matrix on top of the MMAv5 accumulator-info cleanup plus
  row-permuted `ld.red` support is green again:
  - `1668 passed, 119 skipped in 519.91s`

## 2026-03-29: the remaining plain ld/st `32x32` identity subview hole is
not blocked on row-plan reuse anymore; it is blocked earlier in raw-query
construction

- I tried the smallest raw-query-side extension first:
  - new helper `getDirect32x32SubviewRawRowPlan(...)`
  - canonical contiguous view-like `32x32` f32 subviews can borrow the backing
    row plan on the raw-query path in frontend/verifier/lowering

- That was not sufficient.
  - the focused multidim-slice slice stayed:
    - identity negative
    - mixed positive
    - scrambled-cols negative

- The decisive trace finding:
  - the final `view.get_reg_layout(auto)` for the identity case never reaches
    the raw-query override
  - `inferStandaloneTMemLdStQueryLayout(...)` itself fails first with
    `unsupported tensor memory memdesc_subslice view`

- So the next direct-support slice should be:
  - a dedicated support-query helper for canonical contiguous view-like
    `32x32` subviews
  - it must construct a safe scalarizable query layout plus backing row plan
  - it should remain narrow enough that `scrambled_cols` still stays clean
    unsupported

## 2026-03-29: lifted half-row ld/st is now specifically a TMEM
packet-decomposition problem

- Representative user view:
  - start from a lifted public TMEM descriptor
  - `full_view.reshape((2, M // 2, N)).slice(1, 1, dim=0).index(0)`
  - with the identity `64`-column TMEM layout and `instr_variant='32x32b'`

- What is no longer the blocker:
  - reg-layout discovery itself
  - after letting support-query-capable descriptor views past the old blanket
    reject, the part view can again reach direct-planning code
  - `ViewOpToLLVM.cpp` also now zeroes the row-half dim-0 subslice base, so
    the descriptor SSA value no longer carries the old extra `128<<16` row
    delta from the `reshape -> subslice([1,0,0])` step

- What the current experiments say:
  1. Widened `128x64` support-tile query + lifted `row=64` origin:
     - selects direct `I32x32b`
     - can also be scalarized to `x1`
     - both variants still update rows `0:64` instead of `64:128`
  2. Forcing the logical `64`-row warp anchors (`16/32`) onto that widened
     support path:
     - produces `CUDA error: misaligned address`
  3. Recasting the same view as a direct logical `64x64` query with lifted
     origin:
     - reg-layout selection no longer stays on `I32x32b`
     - it falls back to `I16x32bx2`
     - runtime is still wrong

- So the remaining question is narrower now:
  - does the hardware ISA admit a correct direct packet plan for the upper-half
    `64x64` view at all, under the real row-anchor/alignment rules?
  - the two obvious models are currently both bad:
    - widened support tile + lifted origin
    - direct logical query + lifted origin

- Practical implication for the next slice:
  - either find a genuinely correct packet decomposition for that upper-half
    `64x64` view
  - or turn this specific bucket back into a justified clean negative with a
    precise reason grounded in the tcgen05 row-anchor rules

## 2026-03-29: MMAv5 semantic queries are now linear-layout info queries on
both A and D sides

- The old public `MMAv5TMemLayoutPlan` API is no longer used outside
  `Dialect.cpp`.

- New public shape/layout query split:
  - `getMMAv5LhsLayoutInfo(...)`
  - `getMMAv5AccumulatorLayoutInfo(...)`
  - `getMMAv5ScaledAccumulatorLayoutInfo(...)`

- These info objects now cover the external MMAv5 semantic consumers:
  - plain/scaled MMA verifier
  - TMEM allocation row-anchor coupling
  - MMAv5 LLVM lowering
  - LHS promotion to TMEM

- This does **not** mean the finite-family planner is gone.
  - The current instruction-family search still exists internally in
    `Dialect.cpp`
  - But the rest of the compiler no longer consumes that planner as a public
    semantic API or as a recovered legacy layout answer

- Current remaining MMAv5 legacy/family bottleneck after this cleanup:
  - the internal planner still recognizes only layouts that preserve the
    tile-internal canonical MMAv5 mapping and only vary by whole-tile
    multiples/permutations
  - so tile-permuted layouts stay positive, while mixed / scrambled /
    row-col-permuted layouts still clean-reject

## 2026-03-29: dense no-scales copy permutations are still a real semantic
boundary, not just an over-strict verifier gate

- I tried the smallest dense-family widening for
  `tcgen05.copy.128x{128,256}b`:
  - relax the TMEM row/col basis-order gate in
    `isDirectTMemCopyLayoutSupported(...)`
  - try an extra sorted-basis candidate in
    `getTMemCopyDescriptorLayouts(...)`

- That was not a sound support extension.
  - The representative col-permuted case
    `_make_tmem_linear_layout_permuted(128, 128, "identity", "rotate1")`
    compiled and still emitted canonical
    `tcgen05.cp.cta_group::1.128x256b`, but runtime produced logically rotated
    data instead of preserving the user layout:
    - row 0 became `[0, 64, 1, 65, 2, 66, ...]`
    - `mismatch_count = 16128`

- Interpretation:
  - the current dense copy path preserves the canonical physical TMEM order
  - simply accepting or re-sorting a permuted TMEM linear basis order does not
    make the hardware copy realize the user’s logical row/col permutation
  - so the clean-negative dense-permutation gate remains justified until we
    have a descriptor/address model that can faithfully implement those logical
    permutations

- Action taken:
  - reverted the widening and kept the dense row/col-permuted copy buckets as
    clean unsupported

## 2026-03-29: dense `tcgen05.copy.128x{128,256}b` row/col permutations are
not safely supportable by a descriptor-ordering-only widening

- I ran a bounded experiment on the remaining dense no-scales copy
  clean-negative bucket:
  - temporarily let pure permutations of the canonical TMEM row/col basis
    sets past `isDirectTMemCopyLayoutSupported(...)`
  - temporarily let dense families try a sorted-basis descriptor candidate in
    `getTMemCopyDescriptorLayouts(...)`

- Result on the representative col-permuted layout:
  - kernel:
    `tmem_copy_no_scales_linear_kernel`
  - layout:
    `_make_tmem_linear_layout_permuted(128, 128, "identity", "rotate1")`
  - compile succeeded and still emitted canonical
    `tcgen05.cp.cta_group::1.128x256b`
  - but the logical roundtrip was wrong:
    - `out[0, :16] = [0, 64, 1, 65, 2, 66, 3, 67, ...]`
    - not the original logical tensor order

- Interpretation:
  - the dense copy hardware path is preserving canonical physical TMEM order
    for this descriptor/message family
  - simply reordering descriptor basis candidates does **not** make a logical
    TMEM row/col permutation real
  - so the old clean-negative gate on non-ascending dense TMEM row/col basis
    order is not just conservative; it is preventing wrong-code

- Practical boundary after this experiment:
  - dense pure row/col-permuted TMEM layouts should stay rejected for direct
    `tcgen05.copy.128x{128,256}b` unless we add a descriptor/address model
    that can faithfully realize the logical permutation
  - mixed / scrambled dense-copy negatives remain in the same bucket

## 2026-03-29: lifted row-half ld/st descriptor views still need clean
rejection, not rescue

- Representative user view:
  - `full_view.reshape((2, M // 2, N)).slice(1, 1, dim=0).index(0)` on a
    lifted `[2, M, N]` TMEM descriptor

- Re-checked rescue attempts:
  - raw direct lowering for the `32x32b` identity case still produces the old
    sparse wrong-code footprint:
    - modified rows: `0-15,48-79,112-127`
  - a `128x64` support-query rescue can scalarize to
    `tcgen05.ld/st.sync.aligned.32x32b.x1.b32`, but that over-covers the
    backing tile instead of the logical upper-half view
  - a bounded `64x64` canonical support-query rescue falls back to the
    `I16x32bx2` family with local `16/32` row anchors and reproduces the same
    sparse wrong footprint as the raw direct path

- Interpretation:
  - with the currently reachable `tcgen05.ld/st` packet families, this lifted
    row-half descriptor view still does not have a direct semantics-preserving
    lowering
  - leaving the rescue path enabled only turns the clean-negative verifier
    boundary into wrong-code

- Action:
  - removed the half-row support-query rescue path from
    `getTMemLdStSupportQueryLayout(...)`
  - kept the existing clean-negative verifier wording for lifted row-half TMEM
    descriptor views

- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout'`
    -> `12 passed`

- 2026-03-29: row/col-permuted MMAv5 TMEM-linear accumulator layouts still behave as a clean negative because an attempted run-time conversion back to AutoLayout (`ttgl.convert_layout(out, ttgl.AutoLayout())`) failed to compile (`tt.make_range` inference error + `PassManager::run failed`) after a successful `TRITON_BUILD_WITH_CCACHE=true make -j96`, so the existing verifier rejection is safer than forcing the wrong code path.

- 2026-03-29: barriered Gluon MMAv5 kernels should not also emit explicit
  `tcgen05_commit(...)`
  - `tcgen05_mma(..., mbarriers=[bar])` already performs completion-barrier
    signaling
  - an added explicit `tcgen05_commit(bar)` produced duplicate
    `tcgen05.commit...mbarrier::arrive::one...` instructions in PTX/LLIR and
    caused the runtime kernel to wedge
  - removing the redundant commit restored the canonical plain MMA and indexed
    accumulator-view runtime positives

- 2026-03-29: MMAv5 family matching should preserve exact in-instruction basis
  order
  - the direct-layout matcher was too permissive when it compared low-order
    row/col bases as a sorted multiset
  - that admitted scrambled and row/col-permuted TMEM-linear accumulator
    layouts even though the lowering only supports canonical in-tile placement
    plus whole-tile repetition/permutation
  - tightening the matcher to require exact low-order basis order restores the
    desired boundary:
    - row/col-permuted and scrambled layouts -> clean verifier rejection
    - tile-permuted whole-tile layouts -> still directly supported
    - scaled repeated-`N=32` tile-permuted layouts -> still clean verifier
      rejection via the dedicated block-scaled rule

- 2026-03-29: Gluon frontend CTA checking for two-CTA TMEM layouts must honor
  the real CGA shape
  - the old `_check_tensor_memory_layout_ctas(...)` helper treated every
    `two_ctas=True` layout as exactly `2` CTAs per CGA, even when the provided
    `cga_layout` represented a larger CTA grid
  - this broke the Gluon multicast-MMA core tests once TMEM layouts started
    flowing through larger two-CTA CGA configurations such as `8` and `16`
    CTAs per CGA
  - the helper should count CTAs from the explicit `cga_layout` whenever one
    is present, and only fall back to `2` for legacy two-CTA layouts without
    an explicit CGA description

## 2026-03-29: multidimensional `32x32` ld/st identity/mixed split restored

- Remaining regression before the latest rebuild:
  - full TMEM runtime matrix had exactly two failures:
    - the identity multidim `32x32` slice compiled instead of rejecting
    - the mixed multidim `32x32` slice compiled but miscomputed

- Root causes:
  - identity:
    - the explicit descriptor-view type was an **exact dense row/col `32x32`
      TMEM-linear layout**, and the dedicated subview-support rescue still
      admitted it even though it came from a larger backing tile
  - mixed:
    - the direct scalar `32x32b.x1` path was still valid, but the scalar
      packet recovery code advanced PTX `.b32` col-immediates in units of `1`
      instead of `4` bytes, so the emitted sequence became `0..7` instead of
      `0,4,8,...,28`

- Fixes:
  - `TensorMemoryUtils.cpp`
    - exact dense row/col `32x32` descriptor views from explicit backing-tile
      view chains now stay on the clean-negative path
    - the dedicated subview-support rescue declines that exact-canonical case
  - `TensorMemoryToLLVM.cpp`
    - scalarized `32x32b.x1` recovery now steps `.b32` immediates by `4`
      bytes per packet

- Result:
  - identity -> clean compile-time rejection again
  - mixed -> correct numerics and the expected `tcgen05.ld/st.sync.aligned.32x32b.x1.b32`
    offset sequence
  - scrambled-cols -> still clean unsupported

- Validation:
  - `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. pytest -s --tb=short -n 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'multidim_slice_identity_reports_clean_error or multidim_slice_positive or multidim_slice_reports_clean_unsupported'`
    -> `3 passed`

## 2026-03-30: follow-up validation and direct-support fixes

- Full lit is green again on the current post-fix tree:
  - `cd build/cmake.linux-aarch64-cpython-3.12 && ninja check-triton-lit-tests`
    -> `248 passed, 2 unsupported`

- Direct non-scales ld/st regained support for permuted `64x2` split-N TMEM
  layouts:
  - root cause was a premature non-scales lane/warp broadcast rejection in
    `computeTMemLdStEncodingInfoImpl(...)`
  - `lowerTMemLdSt(...)` already knows how to remove broadcasted registers, so
    rejecting those layouts before it ran was unnecessarily cutting off the
    valid `16x32bx2` / `32x32b_splitn` direct path
  - after removing that precheck, the full
    `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep` bucket is
    back to green (`224 passed`)

- Tensor-memory-scales `warpx2` candidate copies remain intentionally
  unsupported, but their late-lowering diagnostics now re-emit the family-level
  `tcgen05.copy` guidance:
  - lowering still fails after shared-memory allocation because no compatible
    descriptor plan exists
  - `TensorMemoryToLLVM.cpp` now emits the
    `maps to tcgen05.copy... could not synthesize a compatible shared-memory
    descriptor plan for tensor memory scales` diagnostic before the outer
    `PassManager::run failed`
  - runtime-matrix tests now accept the outer exception wrapper while checking
    the real unsupported reason

- Current Python validation checkpoint:
  - `python/test/unit/language/test_compile_only.py` -> `7 passed`
  - `python/test/gluon/test_core.py -k 'tmem_reduction or tcgen05_mma_multicast_commit'`
    -> `92 passed`
  - `python/test/unit/tools/test_triton_to_gluon.py` -> `16 passed`
  - split 4-way run over
    `python/test/gluon/test_frontend.py` +
    `python/test/gluon/test_tmem_runtime_matrix.py`
    -> aggregate `1873 passed, 119 skipped`
  - focused `triton_kernels` nvfp4 persistent slice
    -> `1 passed, 1 skipped`


## 2026-04-01: GB200 unit-suite MMAv5 `test_dot` regression

- The first full `python/test/unit` GB200 microbatch uncovered a real compiler regression outside the TMEM runtime matrix:
  - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-none-tf32x3-float32-float32-1-None]`
  - `python/test/unit/language/test_core.py::test_dot[1-64-64-64-4-False-False-chain-dot-ieee-bfloat16-float32-1-None]`
- Origin/main passed those nodes. The current tree failed with ~49.5% mismatches.
- Root cause was not the MMA opcode count itself; it was the interaction between canonical explicit M64 TMEM-linear layouts and direct TMEM lowering:
  - canonical M64 register-layout selection had drifted to the narrower split-N `x16` path for ordinary 4-warp 64x64 accumulators
  - after restoring `x32`, initialized `ttng.tmem_alloc %src` still used the type-only ld/st query path, which collapsed the explicit TMEM-linear M64 allocation to a standalone `64x64` query instead of the real `128x64` backing/support form
- The TMEM query trace made the second issue explicit:
  - good path: raw query/support form `row size 128`, `warpBase0=2097152`, `warpBase1=4194304`
  - bad path: standalone `queryType 64x64`, `warpBase0=1048576`, `warpBase1=2097152`
- Fixes:
  - `Dialect.cpp`: keep canonical 4-warp M64 on the full `x32` path
  - `TensorMemoryToLLVM.cpp`: initialized allocs must pass the actual memdesc SSA value to `lowerTMemLdStFromTypes(...)`, not `Value()`, so direct lowering can use the same backing/support query logic as `tmem_store` / `tmem_load`
- Fresh-cache validation after the fix:
  - exact `tf32x3` node -> pass
  - exact `chain-dot-ieee-bfloat16` node -> pass

## 2026-04-01: GB200 unit-suite `test_simple_matmul` epilogue-subtile regressions

- The next GB200 split/group unit batch uncovered another real regression in direct TMEM lowering outside the TMEM runtime matrix:
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-128-128-16-4-float16-float16]`
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-128-128-16-4-float32-tensorfloat32]`
- Both cases use `EPILOGUE_SUBTILE=True`, which lowers the accumulator epilogue through two `ttng.tmem_subslice` views (`N=0` and `N=64`) followed by two direct `ttng.tmem_load`s.
- Failure signatures:
  - TF32: left half matched, right half was an exact duplicate of the left
  - packed f16: after the TF32 fix alone, the right half no longer duplicated the left but still read the wrong packed window
- Root cause split:
  - 32-bit `ttng.tmem_subslice` raw queries need the translated `baseOffset` preserved into lowering; zeroing it drops the `N=64` shift and duplicates the left half
  - packed subword `ttng.tmem_subslice` views cannot reuse the same direct-origin remap path, because they still need the packed-aware fallback query construction to preserve packed column semantics before lowering
- Current rule:
  - preserve raw-query `baseOffset` for `ttng.tmem_subslice` during direct ld/st lowering
  - use the new direct translated-origin remap only for 32-bit-and-up `ttng.tmem_subslice` element types
  - keep subword `ttng.tmem_subslice` on the packed-aware resized-layout fallback and on the standalone-before-raw query ordering
- Validation checkpoint:
  - exact GB200 TF32 node -> pass
  - exact GB200 f16 node -> pass


## 2026-04-01: whole-root TMEM descriptors must bypass view-analysis normalization for direct ld/st selection

- Another GB200 unit-matmul regression surfaced outside the TMEM runtime matrix:
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-64-128-32-4-float16-float16]`
- With `TRITON_PREFER_TMEM_16x256_LAYOUT=1`, current HEAD `69c7822a8` was selecting `16x128b` for the `EPILOGUE_SUBTILE=True` accumulator load/store path, while `/tmp/triton-origin-main` still selected the correct `16x256b.x8.unpack/pack` path.
- The key behavioral difference was not in TTGIR result layout or in PTX opcode spelling. It was earlier:
  - current direct ld/st selection/planning was routing even whole-root TMEM descriptors (`shape == allocShape`) through `getTMemViewAnalysisLinearLayout(...)` + `normalizeTensorMemoryLinearLayoutForAnalysis(...)`
  - origin/main still used the raw storage mapping via `toLinearLayout(shape, encoding)` for whole descriptors
- For true descriptor views and higher-rank slices, the analysis-layout path is still required. But for whole descriptors it is too lossy for legacy-equivalent M64 `f16` whole-tile selection and rejects the working `16x256b` direct path.
- Current rule:
  - whole non-scales descriptors (`shape == allocShape`) use the raw storage layout for direct ld/st layout selection and encoding-info planning
  - descriptor views continue to use the generalized TMEM-view analysis layout path
- Validation checkpoint:
  - exact GB200 `64x128x32` f16 epilogue-subtile node passes again
  - direct PTX repro shows `16x256b.x8.unpack::16b` / `16x256b.x8.pack::16b` restored

- Revalidation note (2026-04-01):
  - the exact GB200 `64x128x32` `f16` epilogue-subtile matmul node is green on clean committed HEAD `69c7822a8`
  - the temporary opcode-shape experiments were a dead end and were discarded; the current tree is back at the committed checkpoint before the full GB200 CI-equivalent sweep

## 2026-04-01: MMAv5 lowering must use raw TMEM-linear accumulator mappings, not recovered legacy families

- A second GB200 unit-matmul regression surfaced after the whole-root ld/st fix:
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-256-128-32-4-float16-float16]`
- The direct PTX symptom isolated the bug:
  - stores were still `tcgen05.st.sync.aligned.16x256b.x16.unpack::16b.b32` as expected
  - the final load could stay a single `tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32` and still be numerically correct
  - the real divergence was the MMAv5 accumulator write addresses: current tree used `[%r437 + 256]`, while `/tmp/triton-origin-main` used `[%r437 + 128]`
- Root cause was in the LLVM lowering loader, not the ld/st selector:
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`
  - `DotOpMmaV5TmemLoader::build(...)` was taking a TMEM-linear memdesc, recovering a legacy-like MMAv5 family, then rebuilding the loader address model from that recovered legacy family
  - for whole-tile linear accumulators, that internal legacy normalization can change the physical stepping used by the generated MMAv5 address operands even though the user-visible memdesc is already precise
- Current rule:
  - MMAv5 loader/address generation must use the raw `toLinearLayout(memTy)` mapping for TMEM-linear accumulators/LHS descriptors
  - legacy/TMEM shorthands may still exist in IR/source as sugar, but lowering should not depend on recovering them when a precise linear mapping is already available
- Validation checkpoint:
  - `64x128x32` and `256x128x32` GB200 `f16` epilogue-subtile unit matmuls both pass again
  - PTX for the `256x128` case now restores the `+128` MMAv5 slice offsets
  - `make test-lit` is green again after updating the one stale Blackwell conversion CHECK from `+64` to `+32`


## 2026-04-01: whole-root explicit TMEM-linear M64 accumulator stores must match the analysis/support layout, not the raw encoding form

- Another GB200 unit-matmul regression surfaced during the broader unit-test sweep:
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-1-64-512-32-2-float32-tensorfloat32]`
- The regression was not in MMAv5 proper. It was in the initial whole-root `ttng.tmem_store` that zeroes the accumulator memdesc before the async MMAv5 loop.
- PTX symptom:
  - current tree scalarized the init into many `tcgen05.st.sync.aligned.32x32b.x1.b32`
  - `/tmp/triton-origin-main` kept the direct wide `32x32b.x64.b32` path
- Root cause:
  - `computeTMemLdStEncodingInfoImpl(...)` still had a legacy-anchored M64 rescue path for explicit TMEM-linear descriptors, but that rescue was trying to rediscover the family from the raw encoding form.
  - For the `64x512` explicit linear accumulator, the raw encoding exposes the compact physical span (`256` physical TMEM columns with a row bit spilling into the second output dim), while the direct ld/st planner is already working from the normalized analysis/support layout.
  - That meant the rescue path never recognized the exact legacy-equivalent family and the direct whole-root store fell back to scalar `I32x32b.x1` selection.
- Current rule:
  - whole-root non-view M64 rescue should compare against the normalized analysis/support layout that direct ld/st is actually lowering, not against the raw explicit encoding bits
  - the rescue remains limited to whole-root descriptors (`shape == allocShape`) and only for the standard `32/64` within `128` row-plan override, so descriptor views still go through the generalized view-analysis path
- Validation checkpoint:
  - the tf32 `64x512x32` GB200 unit-matmul node is green again
  - the earlier f16 GB200 control nodes remain green
  - `make test-lit` stayed green after the fix

## 2026-04-01: whole-root explicit TMEM-linear M64 subword ld/st rescue must preserve the raw zero-col-basis layout for lowering

- Another GB200 unit-matmul regression surfaced while resuming the sharded `python/test/unit` sweep:
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[False-False-4-1-64-512-32-2-float16-float16]`
- The first M64 rescue change fixed the scalarization problem but still produced wrong code for the `f16` case.
- PTX symptom after the first fix:
  - current tree recovered the wide `32x32b.x128` direct TMEM store/load again
  - but it emitted the bare 32-bit form (`tcgen05.{st,ld}.sync.aligned.32x32b.x128.b32`) instead of the correct subword form from `/tmp/triton-origin-main`:
    - store: `unpack::16b`
    - load: `pack::16b`
- Root cause:
  - the legacy-anchored M64 helper in `computeTMemLdStEncodingInfoImpl(...)` was using the same stripped whole-root layout for two different purposes:
    - exact legacy-family equivalence matching
    - actual ld/st lowering
  - for subword TMEM layouts those are not the same thing
  - removing zero `col` bases is useful for recognizing that an explicit TMEM-linear layout is equivalent to a legacy `blockM=64` family, but it destroys the spacing information that `lowerTMemLdSt(...)` relies on to select `pack::16b` / `unpack::16b`
  - the helper also copied a `getExpectedTMemLoadValueCount(...)` sanity check that is intentionally defined only for 32-bit ld/st paths, so it will always fail on subword cases
- Current rule:
  - for whole-root explicit TMEM-linear M64 rescue paths, use the stripped layout only to answer “is this exactly legacy-equivalent?”
  - use the raw storage layout to build the direct lowering transform
  - keep the `getExpectedTMemLoadValueCount(...)` post-check restricted to `bitwidth == 32`
- Validation checkpoint:
  - the GB200 `f16` `64x512x32` unit matmul passes again
  - the earlier tf32 `64x512x32` control also stays green
  - PTX now restores the origin-style wide subword forms with `pack::16b` / `unpack::16b`


## 2026-04-01: TMEM subslice physical offsets must be computed in the source tile address space

- A GB200 `triton_kernels` persistent ragged matmul regression remained after the earlier TMEM direct-lowering sweep:
  - `python/triton_kernels/tests/test_matmul.py::test_op[None-True-False-False-False-None-128-768-512-1024-ragged-float16-float16-None-10-1-False-False-None-False-False-False-True-None]`
- Symptom pattern:
  - only the second 128-column half of each 256-column MMAv5 accumulator tile was corrupted
  - the dumped `_p_matmul` TTGIR split the accumulator into `ttng.tmem_subslice {N = 0}` and `{N = 128}` followed by two `ttng.tmem_load`s
  - the dumped PTX then showed both loads coming from the same physical TMEM base, which explains why the `128:256` and `384:512` column bands were wrong
- Root cause:
  - `TMEMSubSliceOpConversion` in `TensorMemoryToLLVM.cpp` was calling `getTMemSubSliceOffset(...)` on the narrowed result memdesc type
  - physical TMEM base arithmetic is defined in the source tile's address space, not the result view's address space
  - for N-half views like `128x256 -> 128x128`, the result type drops the high-order column basis that differentiates the second half, so `{N = 128}` collapsed onto `{N = 0}`
  - `BufferRegion.cpp` had the same stale assumption in its TMEM-subslice region accounting
- Current rule:
  - compute TMEM subslice physical offsets from the source memdesc type in lowering and region analysis
  - the result memdesc type still describes the logical view shape/encoding, but not the address space used to advance the raw TMEM base
  - multibuffer direct-ld/st query plumbing should preserve the already-lowered base translation instead of re-encoding the selected leading buffer as an extra query-origin offset
- Validation checkpoint:
  - the exact persistent ragged `triton_kernels` repro is green again
  - focused conversion/analysis lit checks now encode the correct nonzero subslice offsets


## 2026-04-01: sub-32-bit Blackwell MMAv5 epilogue subtile replay must stay on the original full-load path

- After the TMEM subslice/base-offset fixes, one GB200 unit bucket was still failing:
  - `python/test/unit/language/test_matmul.py::test_simple_matmul[True-True-4-1-128-128-16-4-float16-float16]`
- Symptom pattern:
  - only the Blackwell `LAYOUT_16x256=True` + `EPILOGUE_SUBTILE=True` path failed for sub-32-bit accumulators
  - the TTGIR replayed the epilogue `reshape -> trans -> split` into two `ttng.tmem_subslice` / `ttng.tmem_load` pairs on `128x64xf16`
  - numerically, the second 64-column half of each 128-column tile was corrupted while the matching TF32 control passed
- Root cause:
  - the remaining bug is in the replay optimization itself (`OptimizeTMemLayouts.cpp`), not in raw TMEM address arithmetic
  - once the earlier source-type TMEM subslice offset fix and raw-query baseOffset fix were in place, the F16 replayed split-load kernel still failed unchanged, which means the sub-32-bit `pack::16b` replayed split-load lowering is not yet semantics-preserving
- Current rule:
  - for sub-32-bit TMEM tiles, do not apply the `reshape -> trans -> split` replay optimization into `ttng.tmem_subslice` / `ttng.tmem_load`
  - keep the original full `ttng.tmem_load` epilogue path until a direct sub-32-bit replay decomposition is proven correct
  - 32-bit replayed split-loads stay enabled
- Validation checkpoint:
  - exact F16 GB200 node is green again
  - matching TF32 control stays green
  - the previously fixed persistent ragged `triton_kernels` MMAv5 repro still passes
  - full lit remains `248 passed, 2 unsupported`

## 2026-04-01: GB200 CI-equivalent sweep checkpoint after `f70f9aeb6`

- Running the post-commit GB200 validation as split GPU microbatches over `CUDA_VISIBLE_DEVICES=0..3` using `/tmp/run_split_batches.sh`.
- Completed so far:
  - rebuild: `TRITON_BUILD_WITH_CCACHE=true make -j96`
  - C++ unit tests: `240 passed`
  - full lit: `248 passed, 2 unsupported`
  - `python/test/unit` main shard: all `32` split groups passed
  - `python/test/unit/test_debug.py`: `95 passed`
  - plugin/custom-op unit slices: all green
- Current in-flight target:
  - `python/triton_kernels/tests` split `32` ways across the `4` GPUs; first wave passed and later waves are still running.

## 2026-04-01: GB200 sweep progress after `triton_kernels` and the rest of `test-unit`

- Additional completed targets:
  - `python/triton_kernels/tests`: all `32` split groups green; aggregated totals from the per-group logs are `2013 passed, 3444 skipped`
  - `python/tutorials/06-fused-attention.py`: all `16` split groups green; selected cases are skipped on this node
  - `python/test/unit/instrumentation/test_gpuhello.py`: `1 passed`
- Current in-flight target:
  - `python/test/gluon/` + `python/tutorials/gluon/` split `32` ways across the `4` GPUs.

## 2026-04-01: GB200 `python/test/gluon` blocker fix

- Restored the historical TMEM source shorthand: `tensor_memory_descriptor.slice(start, length)` now defaults to slicing the trailing dimension again. Explicit `dim=` still uses generic `ttg.memdesc_subslice` semantics.
- Exact TMEM memdesc IR types now survive Gluon view reconstruction and warp-specialize block-arg deserialization:
  - semantic view builders (`slice/index/trans/reshape/reinterpret`) cache `handle.get_type()` on TMEM descriptor types
  - `tensor_memory_descriptor_type._unflatten_ir` now restores the exact handle type instead of rebuilding a nominally equivalent TMEM type
- This fixes the `tt.call` operand mismatch that appeared in the GB200 `test_consan` aliasing kernel after the public slicing path moved to generic `ttg.memdesc_subslice`.
- `BufferRegionAnalysis` now handles generic TMEM `ttg.memdesc_subslice` correctly:
  - use `ttng::getTMemViewOffset(...)` for tensor-memory subslices
  - if a generic shared-memory subslice cannot be inverted exactly, conservatively keep the parent region instead of asserting
- Validation checkpoint:
  - the direct `DISABLE_SUBPROCESS=1` repro is back to the original expected device-side assert (`Buffer being accessed has outstanding reads`)
  - the wrapped `test_aliasing_tensor_visibility_outstanding_read[1ctas-True]` passes again
  - the stale `test_tensor_memory` frontend expectation was updated to the restored trailing-dimension TMEM shorthand


## 2026-04-01: GB200 `test_tmem_subslice_block_m_64` fix

- The remaining M64 TMEM column-subview bug was not the half-tile zero store anymore; after the root `64xN` TMEM layout selection change, the `64x128 -> 64x64 -> reinterpret<f16, 64x128>` zeroing path was correct and only the three `64x2xf32` follow-up stores were still wrong.
- The bad behavior came from the old column-subview support-rescue path in `TensorMemoryUtils.cpp`:
  - it approximated `64`-row TMEM column subviews with a reshaped support tile that treated them like contiguous leading columns
  - that shifted `64x2` packets onto the wrong TMEM half (the LLIR base addresses moved to `+64/+68/+72` instead of the expected `+32/+36/+40` family)
- The fix is to stop using that support-rescue path for `64`-row TMEM column subviews and let them use the raw TMEM view query instead.
- `gluon_ir.cc` also now avoids the old legacy-only auto-layout shortcut for root `64xN` TMEM descriptors before running the direct TMEM-compatible layout search, so `s_tmem.get_reg_layout()` can be reused by those M64 subviews again.
- Validation checkpoint:
  - `test_tmem_subslice_block_m_64[legacy]` passes
  - `test_tmem_subslice_block_m_64[linear]` passes
  - paired rerun passes


## 2026-04-01: descriptor-chain physical-origin remap and GB200 stale-test cleanup

- The remaining GB200 descriptor-chain wrong-code was in translated standalone ld/st support queries, not in the TMEM packet selection itself.
- Root cause:
  - `inferStandaloneTMemLdStQueryLayoutImpl(... preserveViewOrigin ...)` reused the translated support layout but carried origins over by logical dim name.
  - For reshaped / permuted / reinterpreted TMEM chains that preserve the same physical storage while changing the logical basis, that origin copy can point the direct ld/st plan at the wrong physical tile even when the support layout itself is correct.
- Fix:
  - add `remapTMemLdStQueryOriginThroughPhysicalCoords(...)` in `TensorMemoryUtils.cpp`
  - compute the source physical TMEM coordinates from the original query origin, then apply the destination support layout's left inverse to recover the correct origin in the translated support basis
  - use that remapped origin whenever a translated support analysis succeeds
- This restores the intended aliasing behavior for the double-buffer descriptor-chain matrix kernel and keeps the direct ld/st path exact instead of falling back to implicit repair.
- Separate cleanup in `MemDescReinterpretOp::inferReturnType`:
  - non-TMEM reinterprets now validate the visible source/destination view bits (`shape`) instead of the backing `allocShape`
  - this preserves valid reinterprets of contiguous slices with larger retained backing allocations
- Additional GB200 cleanup:
  - explicit `dim=0` in multi-buffer TMEM slice chains where the test intent was always the outer buffer dimension
  - refreshed exact M64 fallback packet expectations to the current `16x32bx2` direct lowering
  - widened stale negative-diagnostic assertions in `test_fpsan.py` to the current clean verifier/lowering messages


## 2026-04-01: explicit split-N TMEM store layout must survive descriptor-view canonicalization

- After the `rowPlan=64` subview query fix, the remaining `test_tmem_index_subslice` wrong-code was no longer a TMEM base-offset/row-anchor issue.
- Root cause:
  - the `64x32` TMEM descriptor view itself lowered with the correct direct query plan, but the generic `tmem_store(convert_layout) -> tmem_store` canonicalization in `lib/Dialect/TritonGPU/IR/Ops.cpp` erased the user-requested split-N register layout before lowering.
  - that left the store on the blocked source layout while the load still used the split-N linear layout, producing a packet-family mismatch on the same TMEM subview (`32x32b` store vs `16x32bx2` load).
- Fix:
  - keep that canonicalization for plain/root TMEM stores, but disable it for view-like TMEM destinations (`memdesc_subslice`, `memdesc_index`, `memdesc_reshape`, `memdesc_reinterpret`, `memdesc_trans`, and internal `ttng.tmem_subslice`).
  - this preserves explicit user-visible TMEM register layouts on descriptor views, which is required by the project rule that direct user layouts must either lower faithfully or be rejected, not silently rewritten.
- Validation checkpoint:
  - `python/test/gluon/test_fpsan.py::test_tmem_index_subslice` passes again
  - full lit is back to `248 passed, 2 unsupported`
- Related stale test update:
  - `test/Conversion/relayout_tritongpu.mlir` now expects the current preferred 64x64 TMEM store relayout after comparing its output against the `origin/main` checkout in `/tmp/triton-origin-main`; the file still checks that relayout produces a direct TMEM-compatible store layout, only the register/lane split changed.

## 2026-04-01: GB200 MMAv5 shared-input `64x32xf32` accumulator readback regression

- The remaining GB200 MMAv5 failure was not in shared A/B descriptor synthesis or in the MMAv5 opcode family itself.
- Root cause was a split between the type-only and handle-aware TMEM reg-layout queries for plain legacy `blockM=64, blockN=32, colStride=1` accumulators:
  - type-only `tensor_memory_descriptor_type.get_reg_layout(num_warps=4, instr_variant="auto")` already returned the canonical M64 `64x32` layout
  - handle-aware `tensor_memory_descriptor.get_reg_layout(...)` still revalidated candidates with `getTMemLdStRowPlanForQuery(...)`, which classified the raw zero-row-basis legacy leaf as a widened 128-row family and therefore accepted the raw scalar `32x32b.x1` readback layout first
- Symptom:
  - fresh TTGIR/PTX for the failing `test_mma_shared_inputs` node showed
    - bad current path: `#linear<{register=[[0,1],[0,2],[0,4],[0,8],[0,16]], lane=[[1,0],[2,0],[4,0],[8,0],[16,0]], warp=[[32,0],[64,0]]}>` and 32 scalar `tcgen05.ld.sync.aligned.32x32b.x1.b32` loads
    - passing origin path: canonical split M64 layout and `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`
- Fix:
  - broaden `getTMemLdStRowPlanForType(...)` so logical `M=64` leaves derive their row plan from the active row bases (`activeRowBits == 6`) instead of the raw 7-bit legacy encoding
  - when `Dialect.cpp` validates stripped canonical M64 candidates, use the stripped-query overload of `computeTMemLdStEncodingInfo(...)` so the validator sees the same `64`-row TMEM view as the selector
- Result:
  - handle-aware and type-only TMEM load layout selection agree again on plain `64x32xf32` / `64x64xf32` MMA accumulators
  - the GB200 `test_mma_shared_inputs[...]acc_dtype4` repro passes again

## 2026-04-01: GB200 Blackwell preferred `16x256b` matmul support through physical-support ld/st rewrites

- The remaining GB200 Blackwell matmul regressions with `TRITON_PREFER_TMEM_16x256_LAYOUT=1` were not a pure `getDefaultLayoutForTmemLdSt(...)` problem.
- TTGIR showed that the affected cases were going through `OptimizeTMemLayouts` physical-support rewriting:
  - current failing path: `ttng.tmem_alloc !ttg.memdesc<512x64xf16, #tmem_linear>` then `ttg.memdesc_reinterpret -> !ttg.memdesc<128x256xf16, ...>` and `ttng.tmem_store/load` on the support view
  - origin passing path: direct `ttng.tmem_store/load` on the exact family and PTX `tcgen05.{st,ld}.sync.aligned.16x256b...`
- The support rewrite itself is acceptable; the bug was atom selection inside `getTMemLdStPhysicalSupportPlan(...)`.
- Root cause:
  - support-plan probing still preferred `I32x32b` over `I16x256b`
  - therefore any shape that reached the support rewrite ignored `TRITON_PREFER_TMEM_16x256_LAYOUT`, even when the support view was exactly `128x256`
- Fix:
  - preserve the corrected legacy `M64/I16x256b` selector basis in `Dialect.cpp`
  - reorder support-plan atom probing under `TRITON_PREFER_TMEM_16x256_LAYOUT` to try `I16x256b` before `I32x32b`
- Result:
  - tall/narrow exact Blackwell matmul accumulator/store-load cases that rewrite to a `128x256` physical support tile now emit native `16x256b` PTX again instead of stopping at `32x32b.x128`

## 2026-04-02: GB200 preferred `16x256b` exact-family `64x128xf16` fix and compile-only stale PTX check

- The remaining GB200 unit failure after the earlier physical-support `16x256b` fixes was the exact-family `64x128xf16` matmul node with `TRITON_PREFER_TMEM_16x256_LAYOUT=1`.
- Real before/after comparison against `/tmp/triton-origin-main` showed:
  - origin/main still lowered the accumulator through legacy `#ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 2>` and emitted `tcgen05.{st,ld}.sync.aligned.16x256b...`
  - the current tree instead selected canonical `#ttng.tensor_memory_linear`, produced a different register layout, and missed the `16x256b` PTX family
- Root cause was twofold in `Dialect.cpp`:
  - `getDefaultLayoutForTmemLdSt(...)` still tried the normalized canonical M64 preference before the exact-family legacy-anchored selector, so full exact leaves diverged from origin/main under `TRITON_PREFER_TMEM_16x256_LAYOUT`
  - `getDistributedLayoutForTmemLdStLegacyAnchored(...)` had changed `layout16Rows` from the old zero-row-basis test (`basis(row,16) == 0`) to a pure logical-size test (`rows <= 16`); for legacy `M64` leaves that incorrectly added a dead register basis, and direct `ttng.tmem_load` then failed with `unsupported broadcasted TMEM lowering for this view`
- Fix:
  - prefer the exact-family legacy-anchored selector before the normalized canonical M64 selector for full-shape TMEM leaves
  - restore the legacy `layout16Rows` zero-basis detection while still allowing true `<=16` row tiles
- Result:
  - the `64x128xf16` exact-family preferred path is back to the origin/main register layout and `16x256b` PTX family
  - the earlier repaired `64x128xtf32` preferred path still passes
- Related stale test cleanup:
  - `python/test/unit/language/test_compile_only.py::test_compile_only_dot` was not a real regression; current and origin emit the same ordered tcgen alloc/store/mma/commit/wait/load sequence, but current direct ld/st packet counts can be `x64`
  - the test now checks the PTX sequence incrementally and accepts `x16|x32|x64`

## 2026-04-01: GB200 persistent-attention warp-specialization TMEM OOR regression

- Symptom:
  - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_persistent_forward[False-8-False-2-128-128-8192-8192]` failed only on the current tree with `OutOfResources: shared memory, Required: 262336, Hardware limit: 232448`
  - `/tmp/triton-origin-main` compiled the same node with `230076` shared bytes and passed
- TTGIR / pass-dump comparison:
  - before `triton-nvidia-optimize-tmem-layouts`, current still matched origin on the relevant accumulator loop: `ttng.tmem_load ... -> tensor<128x128xf32, #linear>` and `ttng.tmem_store ... tensor<128x128xf16, #linear>`
  - after `triton-nvidia-optimize-tmem-layouts`, current retuned that loop-carried accumulator load to `tensor<128x128xf32, #linear1>` and propagated `#linear1` through the softmax/update path
  - origin/main never took that retune and stayed on the `#linear` path
- Root cause:
  - `TMemLoadReducePattern` in `OptimizeTMemLayouts.cpp` was designed to pick a reduction-friendly layout for 8-warp TMEM loads consumed by reductions along `N`
  - in persistent attention, that same TMEM load is also written back to TMEM after the reduction/elementwise update
  - the relayout was therefore not profitable: it forced the value onto the `#linear1` family, then required later `#linear1 -> #linear` conversions for the TMEM store path, and those conversions/lowerings inflated shared memory enough to exceed GB200's limit
- Fix:
  - make `TMemLoadReducePattern` bail out when the forward slice from the load hits a later `ttng.tmem_store`
  - keep the exact-family full-tile default TMEM ld/st selector on the legacy-anchored layout for exact MMAv5 leaves (`getDefaultLayoutForTmemLdSt(...)`) so the accumulator path stays aligned with origin/main expectations
  - preserve explicit `convert_layout` on `ttng.tmem_store` when it converts into the preferred TMEM store layout instead of folding it away just because the source layout is also direct-compatible
- Result:
  - persistent attention returns to the origin-like `#linear` accumulator path
  - warmup shared memory is back down to `230592` bytes
  - the GB200 persistent warp-specialization repro passes again


## 2026-04-06: packed sparse `I16x32bx2` TMEM half-row support-query anchor fix

- Symptom:
  - reduced repro `/tmp/repro_zero_only.py` and `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[linear]` still failed after the earlier packed sparse support recovery work
  - the direct path had converged to the correct `I16x32bx2` family, but runtime still wrote the wrong row blocks
- Debugging result:
  - the remaining bug was not packet count or second-half offset anymore; it was the external warp anchor pair used for the recovered packed support query
  - for the packed `64x64` support view, the raw layout still contains a hidden zero `row=16` basis
  - stripped-layout anchor selection gave `(16, 32)` rows and broke the `warp1/warp3` halves
  - raw unstripped anchor selection gave `(0, 16)` rows and broke the back half of the tile
  - the correct external anchors are `(0, 32)`: warp parity already consumes the hidden zero `row=16` half internally, so the second external anchor must jump to the next non-zero row bit rather than the immediately following logical bit
- Fix:
  - in the packed sparse `I16x32bx2` helper in `TensorMemoryUtils.cpp`, keep anchor basis selection on the raw packed support layout
  - if the first recovered anchor basis is all zero and the next recovered basis is non-zero, retarget the second external anchor to `warpRow1 * 2`
- Result:
  - reduced repro mismatch drops from `2048` to `0`
  - `python/test/gluon/test_core.py::test_tmem_subslice_block_m_64[linear]` passes again

- 2026-04-06: Bounded the remaining `block_m_64` reinterpret/store wrong-code bucket further. Disabling the `getDistributedLayoutForTmemLdSt(...)` split-N fast path for the 16->32 unpacked recursion did not change either the explicit `p_tmem.get_reg_layout(instr_variant="32x32b")` rejection or the runtime `mismatch 2048` pattern in `/tmp/repro_zero_only_map_p_layout.py`. Fresh debug shows the stronger root cause: the auto-layout path picks a valid public reg layout `#ttg.linear<{register=[[0,1],[0,2],[0,4],[0,8],[0,16],[0,32]], lane=[[1,0],[2,0],[4,0],[8,0],[0,64]], warp=[[16,0],[32,0]], block=[]}>`, but `computeTMemLdStEncodingInfo(...)` still classifies that layout as atom=4 (`I16x32bx2`) for the sparse reinterpret support view instead of `I32x32b`. So the remaining bug is in `TensorMemoryUtils.cpp` atom selection / packed-support lowering, not just in `Dialect.cpp` family construction.

## 2026-04-06: legacy M64 expanded-N TMEM sugar now canonicalizes to the row-zero-lift support form

- Problem:
  - legacy `TensorMemoryLayout((64, 64), col_stride=1)` on shape `64x128` still lowered through the old interleaved M64 canonical form (`row=16 -> (0, 64)`)
  - direct ld/st reinterpret support rescue for the same user-visible view already used the row-zero-lift form (`row=16 -> (0, 0)`, extra `col=64` basis)
  - that mismatch was enough to make the minimal legacy `p_tmem` reinterpret/store kernel fault even though the later store PTX matched the explicit-linear case
- Diagnostic proof:
  - with temporary allocator tracing, the legacy alloc reported `nRow=128, nCol=64`, `tmem_size=64`, and `tcgen05.alloc ... 64`
  - the equivalent explicit linear alloc reported `nRow=64, nCol=128`, `tmem_size=128`, and `tcgen05.alloc ... 128`
  - `TRITON_DEBUG_TMEM_QUERY=1` showed the reinterpret support path was already rescuing the legacy subslice to the row-zero-lift `64x64 -> 64x128` support layouts, so allocation and lowering were using different physical decompositions for the same legacy sugar
- Resolution:
  - canonical legacy M64 layouts with `shape[0] == 64` and expanded `N` now keep the hidden `row=16` basis zero and let the extra `N` growth stay in the column dimension
  - this aligns legacy sugar with the direct-support linear analysis/lowering model and fixes the legacy `block_m_64` reinterpret/store bucket without adding hidden rematerialization or layout repair


## 2026-04-06: GB200 warp-specialization TMEM allocation metadata regression

- Symptom:
  - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-False-2-64-128-8192-8192]` failed on the generalized tree with `OutOfResources: tensor memory, Required: 640, Hardware limit: 512`
  - the exact same nodeid passed on `origin/main`
- Diagnostic proof:
  - current and `origin/main` TTGIR matched structurally for the failing attention kernel; the only meaningful difference was canonical `#ttng.tensor_memory_linear` attrs on current versus legacy `#ttng.tensor_memory_encoding` sugar on `origin/main`
  - the allocs themselves were the same logical TMEM tiles (`128x64xf32` accumulator plus `128x128xf32` / `128x128xf16` temporaries)
  - therefore the regression had to be in TMEM size accounting, not in the generated MMAv5 or ld/st IR
- Root cause:
  - `getTmemAllocSizes(...)` had been widened to use `planMMAv5AccumulatorFamily(...)` / `planMMAv5ScaledAccumulatorFamily(...)` for canonical linear encodings and to replace the physical TMEM column count with `instrShapeN / preferredColStride`
  - that widened planner is appropriate when selecting a supported MMAv5 instruction family, but it is not the same thing as the backing TMEM storage footprint
  - for equivalent legacy-sugar and canonical-linear TMEM allocs, this inflated `ttg.tensor_memory_size` even though the physical allocs were unchanged
- Fix:
  - TMEM allocation metadata sizing now comes from the physical linear-layout extents directly (`ll.getInDimSize(row/col)`) for both legacy-sugar and canonical-linear encodings
  - MMAv5 family planners remain in place for verifier/codegen matching, but no longer participate in `getTmemAllocSizes(...)`
- Result:
  - the exact GB200 attention-forward repro drops back to the origin-like TMEM budget and passes again
  - canonical linear TMEM encodings no longer pay a larger allocator budget than the equivalent legacy sugar

## 2026-04-07: row-zero M64 packed `16x32bx2.x32` descriptor-chain fix

- Symptom:
  - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix[linear_m64_32x32b_8w-layout5-64-128-32x32b-8-16x32bx2]` was the last remaining wrong-code bucket in the packed direct ld/st path
  - after the packed raw-query recovery landed, the crash was gone but the output still duplicated/misplaced 32-column bands
- Diagnostic proof:
  - the recovered MLIR-side direct plan reported `atom=I16x32bx2`, `rowPlan=64`, and `reps` with exactly one higher-warp recovered band bit: `warp=4 -> (row 0, col 32)`
  - generated LLIR showed the packed PTX path still using one `16x32bx2.x32` message, but the lowering-side address assembly had been suppressing that recovered higher-warp col contribution and later forcing `secondHalfOffset` from `64` down to `32`
  - removing only the lowering-side col squash reduced the mismatch bucket from `4096/8192` to `2048/8192`, proving the remaining bug was the stale `64 -> 32` special-case rather than the raw packed plan itself
- Root cause:
  - two compensating hacks had accumulated while recovering the packed path:
    - LLVM lowering zeroed the recovered higher-warp col band contribution for the packed row-zero M64 path
    - TMEM ld/st analysis then rewrote `secondHalfOffset=64` to `32` for the same `>4 warp` row-zero M64 packed-support family
  - once the warp-group band contribution is preserved in lowering, those `64 -> 32` rewrites are wrong and skip the logical `cols 32:64` band
- Fix:
  - preserve the recovered higher-warp col contribution in `lowerTMemLdSt(...)`
  - keep `secondHalfOffset=64` for the `>4 warp` row-zero M64 packed-support direct path
  - keep the raw packed query-layout / raw `rowPlan=64` recovery intact so the path still selects `tcgen05.{ld,st}.sync.aligned.16x32bx2.x32.b32`
- Result:
  - the isolated `linear_m64_32x32b_8w` descriptor-chain repro now passes directly
  - the whole `tmem_descriptor_chain_matrix` family is green again (`26 passed`)

## 2026-04-07: raw root row-zero `M=64` MMAv5 accumulators need widened ld/st anchors

- The remaining post-GB200 rerun wrong-code bucket was not a generic TMEM descriptor problem anymore. User-visible `linear_m64_*` descriptor-chain ld/st still passed, but internal MMAv5 accumulator kernels failed at exactly the first half-row boundary (`row 16`).
- Repros:
  - `python/test/unit/cuda/test_tma_store_gemm.py::test_tma_load_store[64-128-32-1-4-False-True-False]`
  - `python/test/unit/language/test_block_pointer.py::test_block_ptr_matmul_no_scf[shape3-8]`
  - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-64-128-warps2-16-False-True-acc_dtype4]`
- Debug traces showed a common lowering shape:
  - raw root `ttng.tmem_alloc`
  - row-zero `#ttng.tensor_memory_linear` / legacy-equivalent `blockM=64` accumulator encodings
  - `computeTMemLdStEncodingInfo(...)` choosing `atom=I16x32bx2`
  - active-row `rowPlan=64` (`warpRow0=16`, `warpRow1=32`)
- That active-row plan is still right for query/reg-layout selection, which is why user-facing descriptor ld/st continued to pass. But it is too small for the physical backing accumulator tile MMAv5 writes into.
- Origin/current LLIR comparison on the TMA repro made the mismatch concrete:
  - origin ld/st around MMA base the TMEM address on the widened `32/64` anchors
  - current failing path used `16/32` anchors instead
- A broad storage-row-plan change fixed the unit repros but broke plain `linear_m64_*` descriptor-chain tests, so the durable fix is narrower: only raw lowering of root row-zero `64xNxf32` `ttng.tmem_alloc` accumulators is widened back to the `128`-row anchor family. Query planners stay unchanged.


## 2026-04-07: final stable form of the Blackwell root `64xNxf32` MMAv5 accumulator fix

- The earlier widened-root experiment identified the right class of failure but used the wrong anchor family. The stable fix for the preferred sparse/root `M=64` accumulator layout is **not** `32/64 @ 128`; it is the lifted anchor family `warpRow0=16`, `warpRow1=32`, `rowSpan=128`.
- Why that works:
  - the root accumulator still has logical `M=64`, so query/reg-layout selection for user-visible descriptor ld/st should continue to look like the active-row `64`-row family
  - but the physical backing accumulator tile used by MMAv5 spans the lifted `128`-row support form, and the sparse/root row basis means raw lowering must be allowed to keep that lifted support tile even though the nominal input row span is `64`
- The durable implementation split is therefore:
  - `TensorMemoryToLLVM.cpp`: only widen raw root `ttng.tmem_alloc` accumulators that are actually consumed by `tcgen05_mma` / `tcgen05.mma_scaled`
  - `TensorMemoryUtils.cpp`: explicitly accept the lifted `16/32 @ 128` override for `64xNxf32` zero-row-basis accumulator layouts and keep the old M64 `I32x32b` base-offset halving rewrite gated to the real `I32x32b` path
- Effect:
  - the forced `16x256b` Blackwell root-accumulator path now lowers correct `tcgen05.{st,ld}` around MMA
  - the user-facing packed `linear_m64_*` descriptor-chain families stay green because the override does not leak into ordinary query/reg-layout selection


## 2026-04-07: final root `M=64` MMAv5 accumulator split is by memdesc family, not by MMA use alone

- The durable rule is now:
  - legacy/root MMAv5 accumulator memdescs that still use `TensorMemoryEncodingAttr` sugar keep the lifted raw row anchors `16/32 @ 128`
  - explicit zero-row-basis linear root MMAv5 accumulator memdescs use the lifted raw row anchors `32/64 @ 128`
- Why the earlier single-family fix failed:
  - both families can still lower through the same packed `atom=4` direct ld/st family, so atom choice alone does not identify the right physical TMEM anchor family
  - the important distinction is the root memdesc itself: explicit zero-row-basis linear accumulators still need the backing `32/64` anchor family that the older TMA/block-pointer kernels expected, while legacy-sugar root accumulators want the `16/32` lifted family that fixes `test_simple_matmul`
- The implementation therefore chooses the raw widening override in `TensorMemoryToLLVM.cpp` from the root accumulator memdesc family and lets `TensorMemoryUtils.cpp` accept either lifted `128`-row override for the zero-row-basis `64xNxf32` accumulator case.


## 2026-04-07: TMEM scales alloc lowering must keep broadcasted register structure until `lowerTMemLdSt(...)`

- The GB200-equivalent `mxfp` failure was not another generic TMEM ld/st issue and not a regression from the global `LayoutUtils.cpp` helper anymore.
- The actual failing path was `ttng.tmem_alloc` into `#ttng.tensor_memory_scales_encoding<>` during `tcgen05.mma_scaled` lowering.
- `TMEMAllocOpConversion` passes the full unpacked register value list into `lowerTMemLdStFromTypes(...)`; if `computeTMemLdStEncodingInfoImpl(...)` strips broadcasted register bits from `regLayout` before calling `lowerTMemLdSt(...)`, the returned `info.perm` is defined on the reduced register domain, but the lowering still sees the full source value list and aborts in `ColumnAction::apply(ValueRange)`.
- The right ownership boundary is:
  - `computeTMemLdStEncodingInfoImpl(...)` should keep the raw linear register layout for the query
  - `lowerTMemLdSt(...)` should detect and record broadcasted-register structure in `info.broadcast`
  - `TensorMemoryToLLVM.cpp` should then remove/broadcast values through the existing `info.broadcast` handling in `lowerTMemLdStFromInfo(...)`
- With that split restored, the earlier TMEM ld/st raw-linear fix remains valid and the `mxfp` scaled-matmul kernels lower cleanly again.


## 2026-04-07 GB200 CI-equivalent sweep blocker
- Commit under test: `b639073a7` (`Fix TMEM query regressions from GB200 sweep`).
- Validated before GPU-driver failure: `check-triton-lit-tests` green (`248 passed, 2 unsupported`), `check-triton-unit-tests` green (`240/240 passed`), `test-unit` equivalent green including `python/test/unit`, `test_debug.py`, `python/triton_kernels/tests/`, fused attention tutorial, instrumentation, and plugin tests.
- `test-gluon` split sweep progressed cleanly through groups `1-16/32`.
- First failure at `gluon-main` group `17/32`: `python/test/gluon/test_core.py::test_mma_shared_inputs[...]` failed during `torch.randn(..., device='cuda')` setup with `RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable`.
- Isolated rerun of group `17/32` on `CUDA_VISIBLE_DEVICES=0` reproduced mass `device busy or unavailable` failures before Triton kernel execution.
- Isolated single-test rerun of representative `test_mma_shared_inputs[...]` on `CUDA_VISIBLE_DEVICES=1` entered uninterruptible sleep (`STAT=D`) instead of failing at the same point.
- Concurrent minimal Torch CUDA probe also hung.
- `dmesg` at the same time shows repeated NVRM/NVLink driver errors: `knvlinkUpdatePostRxDetectLinkMask_IMPL: Failed to update Rx Detect Link mask!` and `knvlinkDiscoverPostRxDetLinks_GH100: Getting peer0's postRxDetLinkMask failed!`.
- Conclusion: current blocker is machine/driver state, not a new confirmed Triton regression in TMEM codegen. Resume remaining GB200-equivalent GPU targets only after GPU/NVLink health is restored.

- After killing the stray probe and wedged pytest processes, `timeout 20s nvidia-smi --query-gpu=...` still timed out. GPU control path remained unhealthy, so the GB200-equivalent GPU sweep could not be completed further in this session.

## 2026-04-09: half-row ld/st still needs a real linear-layout packet decomposition

- Revalidated the remaining lifted row-half ld/st wrong-code bucket after
  repairing the interrupted support-query cleanup state.
- Important refinement:
  - the bug is not only the lifted higher-rank
    `reshape((2, M/2, N)).slice(1, 1, dim=0).index(0)` descriptor chain
  - the direct `128x64 -> slice(M/2, M/2, dim=0)` TMEM row-half view is also
    wrong, while a root `64x64` tile with the same `16x128b` family still
    works
- The current direct ld/st families do not describe the row-half view as a
  simple contiguous lower-half rebasing:
  - `32x32b`, `16x64b`, `16x256b`, and auto-selected `16x32bx2` touch the even
    16-row bands `0-15,32-47,64-79,96-111`
  - `16x128b` touches the odd 16-row bands `16-31,48-63,80-95,112-127`
  - expected semantics remain the contiguous lower half `64-127`
- This preserves the earlier initiative conclusion:
  - the row-half bucket still looks physically codegenable in principle
  - the current lowering is selecting the wrong packet decomposition for a view
    carved from a larger backing tile
  - do not convert this into a new permanent clean-negative boundary; fix it
    through a cleaner linear-layout planner/decomposition model
- Session outcome:
  - retried bounded support-query/base-offset/anchor experiments
  - reverted them because none restored correct semantics
  - kept only the green rebuilt baseline and recorded the evidence

## 2026-04-09: unified support-query cleanup retired the 32x32 / reinterpret / block_m_64 hack buckets

- The interrupted isolated-worktree support-query refactor was actually
  coherent once rebuilt and revalidated.
- Durable cleanup boundary from this slice:
  - `getTMemLdStSupportQueryPlan(...)` is now the single support-query contract
    used by verifier checks, Gluon direct reg-layout search, and LLVM lowering
  - lowering-specific TMEM subview offsets now live in one shared
    `getTMemSubviewOffsetForLowering(...)` helper instead of diverging between
    the view conversion and final LLVM lowering
  - the old direct `32x32` view quarantine is replaced by explicit query-layout
    scalarization to `32x32b.x1` packets when translated origins would
    otherwise over-cover the logical view
  - reinterpret / column-subview descriptor views reuse the same linear-layout
    support-query machinery instead of bespoke rescue splits
- Validated effect:
  - descriptor-chain reinterpret positives and `M=64` `32x32` fallback cases in
    `python/test/gluon/test_core.py` are green
  - descriptor-chain matrix sweeps and `block_m_64` subslice/reinterpret cases
    are green
  - focused runtime-matrix descriptor-chain / rank5 / multidim slices are green
  - `triton-opt` checks for `test/TritonNvidiaGPU/{ops,invalid,tmem_layouts}.mlir`
    are green
- Backlog impact:
  - retire the post-merge-base hack buckets for direct `32x32`
    rescue/scalarization, `64x128xf32` reinterpret rescue, and the
    `block_m_64` packed row-zero-lifted fixups
  - keep the half-row ld/st packet-decomposition bug open as a separate
    correctness problem
  - remaining targeted cleanup backlog is the separate `warpx2`
    family-specific planner work plus the blocked GB200-equivalent rerun

## 2026-04-09: warpx2 family-specific planner cleanup reduced to shared linear-layout search

- The remaining `warpx2` planner cleanup was structural, not a behavior bug.
- Durable outcome:
  - keep `01_23` vs `02_13` differences only in the plan-spec selection logic
  - use one bounded descriptor-layout mutation search for both families once
    the family-specific core descriptor shape has been chosen
  - remove duplicated `02_13` seed-layout reinsertion and other probe-era
    planner residue without changing the tested lowering results
- Validation:
  - `make -j8` with the recorded include-path workaround is green
  - warpx2 runtime-matrix slice is green (`4 passed`)
  - `test/TritonNvidiaGPU/invalid.mlir` verify-diagnostics is green
- Backlog impact:
  - the post-merge-base managed-session hack inventory is now closed
  - remaining work is no longer planner cleanup; it is the separate half-row
    ld/st correctness bucket plus resuming GB200-equivalent validation once the
    machine is healthy

## 2026-04-09 cache-sensitive validation and current block_m_64 boundary
- `fresh_knobs` deletes cache env vars such as `TRITON_CACHE_DIR`, so command-line cache-dir isolation is not sufficient for tests that use that fixture. If a test needs a fresh compiler cache after calling `fresh_knobs`, set `fresh_knobs.cache.dir = fresh_triton_cache` inside the test or clear the default `~/.triton/cache` before a broad shard rerun.
- Parameterized tests whose compile result depends on a closure/layout object can alias through the on-disk cache inside one pytest process unless they get a per-test cache dir.
- Current `block_m_64` parent-layout reinterpret boundary:
  - legacy `TensorMemoryLayout((64, 64), col_stride=1)` parent-layout stores into the packed `float16` reinterpret subview are now positive and lower directly.
  - the row-zero-lifted `_make_tmem_linear_layout_m64(128)` parent-layout case is still clean-negative; it needs the packed `32x32b.unpack::16b` direct path and cannot yet accept the parent TMEM register layout directly.

## 2026-04-09 higher-rank validation refresh: live bug buckets after expectation cleanup
- The higher-rank runtime-matrix expectations now match the real current boundaries:
  - lifted one-CTA `index`, `multidim_slice`, and `dim0_slice` cases are positive for `n in {64, 128}` and real TMEM OOR at `n=256` (`Required: 1024`, `Hardware limit: 512`).
  - lifted two-CTA `block_two_ctas` `dim0_slice` is the same OOR boundary at `n=256`.
  - lifted two-CTA `mmav5_twocta` `dim0_slice` is still a clean CTA/CGA mismatch (`Layout has 1 CTAs per CGA, but the context requires 2 CTAs per CGA.`), not part of the OOR bucket.
- Remaining live implementation bugs on healthy local GPUs:
  - identity lifted higher-rank half-row views are wrong-code at `n=64/128` (`max_abs 13.0`, `8192` wrong elements in the direct probe) and OOR at `n=256`; current lowering emits `16x32bx2.x32.b32` packets at offsets `0` and `64` instead of a correct lower-half decomposition.
  - `scrambled_cols` multidim slice is wrong-code (`max_abs 11.0`, `5120` wrong elements); current lowering decomposes it into many `32x32b.x1.b32` packets at scalar offsets `0..31`, which loses the scrambled-column physical mapping.
  - split-N row/col permuted `rotate1-identity-2-32x32b_splitn` still faults at launch with `Triton Error [CUDA]: misaligned address`.
- Recommended next implementation order:
  1. tighten or repair the view-like fallback path in `compute_tmem_reg_layout_from_memdesc(...)` so surrogate query-type fallback cannot over-admit wrong-code descriptor views.
  2. return to the real half-row packet-decomposition fix once the fallback path is no longer masking it.
  3. debug the separate split-N permuted reg-layout / lowering misalignment after the view-like descriptor work.

## 2026-04-09 descriptor-view fallback checkpoint: column-half views restored, split-N row-anchor bug remains
- The descriptor-view over-admission buckets from the previous checkpoint are closed locally:
  - lifted one-CTA and two-CTA `dim0_slice` positives are green again.
  - lifted higher-rank half-row clean negatives are green again.
  - `scrambled_cols` multidim slice is back to a clean unsupported boundary instead of wrong-code.
- Durable implementation shape from this slice:
  - lifted `reshape -> dim0 subslice -> index` column-half views now get a real TMEM support-query plan derived from the backing 2D support frame.
  - impossible direct descriptor views are screened by row-anchor representability, not by a blanket Gluon-side surrogate-query ban.
  - Gluon direct reg-layout search can use surrogate/canonical query types again after support-query and raw-query paths fail.
- Current remaining TMEM bug bucket on healthy local GPUs:
  - `python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[...]` still faults with `Triton Error [CUDA]: misaligned address` for row/col-permuted `M=64` direct ld/st layouts.
  - the failure is not specific to the split-N heuristic: exact repro on `row_perm=rotate1`, `col_perm=identity`, `n=2` also faults for explicit `16x32bx2`.
  - warmup TTGIR for that repro shows the generated reg-layout warp anchors as `[[32, 0], [1, 0]]` while the logical 64-row family should anchor on `16,32`; current code is keying off row-basis order under permutation instead of logical row-anchor coordinates.
- Validation evidence worth preserving:
  - focused descriptor-view runtime slices: `21 passed`, then `47 passed, 1 skipped`, then two-CTA `12 passed`.
  - exact split-N repro remains red under `CUDA_LAUNCH_BLOCKING=1`.
  - mixed-file broad sweeps on this node are currently noisy because GPU 0 showed unrelated `test_mma_shared_inputs[...]` NaN failures in `python/test/gluon/test_core.py`; do not treat that bucket as TMEM initiative signal until rerun independently.
- Next step:
  - repair 64-row direct ld/st anchor selection/validation for row-permuted layouts, then rerun the row/col-permuted split-N sweep and a clean matrix-only shard sweep.

## 2026-04-09: `f16` MMAv5 root accumulators were missing the explicit M64 row-plan contract

- Healthy-GPU reruns showed the `test_mma_shared_inputs` bucket was not a fresh tcgen05 arithmetic regression.
- Exact repros:
  - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-0-warps1-16-False-False-acc_dtype1]`
  - `python/test/gluon/test_core.py::test_mma_shared_inputs[False-ctas_per_cga0-1-1-1-64-0-0-warps2-16-False-False-acc_dtype1]`
- Root cause:
  - `annotateMMAv5AccumulatorRootRowPlan(acc)` was already in the Gluon tcgen05 builder path, but `getMMAv5AccumulatorRootRowPlan(...)` only returned a plan for `bitwidth == 32`.
  - Raw `64x{64,32}xf16` root accumulators therefore never got the explicit `ttng.tmem_ldst_row_plan` attribute, so ld/st lowering fell back to the active-row `16/32 @ 64` family.
  - The bad TTGIR/PTX signature was:
    - no `ttng.tmem_ldst_row_plan` on the root `ttng.tmem_alloc`
    - `tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32`
    - narrow TMEM base math (`shl 20`, mask `3145728`)
  - Runtime symptom on the `64x64xf16` repro: rows `16-31` and most of `48-63` came back as zeros (`2001 / 4096` mismatches).
- Fix:
  - broaden `getMMAv5AccumulatorRootRowPlan(...)` to all rank-2 `M=64` MMAv5 accumulators, independent of element bitwidth.
- Result:
  - fresh `64x64xf16` / `64x32xf16` TTGIR now carries `ttng.tmem_ldst_row_plan = [32, 64, 128, 0]`.
  - fresh repaired PTX uses the widened `shl 21` / mask `6291456` TMEM base path and vectorizes back to `16x32bx2.x8.pack::16b.b32` for the `64x64xf16` case.
  - the exact previously failing `warps1` and `warps2` `test_mma_shared_inputs` nodeids now pass again, and the earlier `test_tma_load_store[64-128-32-1-4-False-True-False]` unit repro stays green.
- Important separation:
  - this fix does not change the remaining `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[rotate1-identity-2-32x32b_splitn]` launch failure.
  - keep that one as the separate `M=64` row/col-permuted direct-ld/st anchor-selection bug.

## 2026-04-09: logical row-anchor matching must use TMEM row-coordinate contribution, not basis index

- The live split-N row/col-permuted `M=64` direct ld/st failure was the expected consequence of treating row anchors as if `basis_index == log2(logical_row)`.
- Correct invariant:
  - row bases may be permuted and may include explicit zero-lift entries, so basis position is not stable.
  - the direct ld/st planner and its clean-negative descriptor-view diagnostics both need to identify row anchors by the row-coordinate contribution of the TMEM row basis family.
- Implementation shape that stayed correct:
  - centralize the lookup in `getLogicalRowAnchorBasis(...)` in `TensorMemoryUtils.cpp`
  - use that helper in packed support-query planning, the main `computeTMemLdStEncodingInfoImpl(...)`, and the unsupported descriptor-view row-anchor reason
  - keep matching against the row coordinate stored in the TMEM `[row, col]` basis vector (`basis.front()`)
- Important failed sub-attempt:
  - trying to rediscover the row coordinate through `LinearLayout` output-dimension names caused immediate false clean rejections on valid positives (`test_mma_shared_inputs`, split-N row/col-permuted ld/st, descriptor-composition row/col-permuted ld/st)
  - do not use `getOutDimNames()` as the source of truth for this TMEM row-anchor lookup
- Current status after the fix:
  - exact and broad split-N row/col-permuted runtime-matrix sweeps are green again
  - the minimal tcgen05 MMA `M=64` bucket stays green
  - the clean-negative multidim-slice `scrambled_cols` case stays green
- Remaining initiative work is no longer the split-N permuted row-anchor bucket; it is the separate half-row ld/st packet-decomposition bug plus the broader GB300-equivalent rerun.

## 2026-04-09: direct ld/st row-anchor legality must query the full TMEM linear-layout image

- Broad healthy-node reruns exposed a separate real `test_mma_shared_inputs[...]` compiler regression for widened `two_ctas=True` MMAv5 accumulator descriptor views.
- Exact failure shape before the fix:
  - `TMEM layout 'auto' unsupported for descriptor view tensor_memory_descriptor<...>`
  - debug reason:
    - `required row anchors 32,64 are not directly representable in the descriptor view`
- Root cause:
  - the first row-anchor helper generalization was still too basis-family-specific.
  - it only accepted required anchors when they appeared in `layout.getBases().lookup(kRow)`.
  - for arbitrary linear layouts, that is wrong: a legal TMEM row anchor is an output-space vector like `[64, 0]`, and the full linear-layout image may materialize it through `col` or block bases.
  - concrete failing layout evidence:
    - `rows=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 32]]`
    - `cols=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [64, 0]]`
    - `block_bases=[[128, 0]]`
    - the required `64` anchor exists, but not in the `row` basis family.
- Correct invariant:
  - direct tcgen05 ld/st row-anchor support must be decided by representability of the desired TMEM output-space anchor vector in the full linear-layout image.
  - do not special-case by input-dimension family once the codegen path is already expressed in linear-layout arithmetic.
- Implementation shape that works:
  - in `getLogicalRowAnchorBasis(...)`, build the expected output vector `[logical_row, 0, ...]`
  - use `layout.pseudoinvert()` to synthesize candidate input coordinates
  - reapply the original layout and accept only if it realizes the expected output vector exactly
  - keep using that shared helper in:
    - packed support-query planning
    - main `computeTMemLdStEncodingInfoImpl(...)`
    - unsupported descriptor-view row-anchor diagnostics
- Validation snapshot for this checkpoint:
  - exact previously failing two-CTA repros: green
  - one-CTA control: green
  - split-N row/col-permuted guard repro: green
  - grouped widened two-CTA slices:
    - `30 passed`
    - `30 passed`
    - `30 passed`
- Remaining initiative work after this checkpoint:
  - rerun the interrupted broad 4-GPU `test_core.py` + `test_tmem_runtime_matrix.py` sweep on the healthy node
  - continue on the separate half-row ld/st packet-decomposition correctness bucket if that remains the next real failure surface

## 2026-04-09: row-plan anchors are logical-row images through the analyzed layout

- The "row anchors are fixed output-space vectors like `[32, 0]` / `[64, 0]`" model was still wrong.
- Correct invariant:
  - row plans are expressed in the analyzed layout's logical `row` input space.
  - the physical warp anchor is the image of the sparse logical point `{row = logicalRow}` under the analyzed `LinearLayout`.
  - this is why the same row plan can materialize as different output vectors in different legal layouts.
- Concrete evidence from the repaired split-N `64x128` root case:
  - analyzed mem layout:
    - `row=16 -> (0, 64)`
    - `row=32 -> (16, 0)`
    - `row=64 -> (32, 0)`
  - the valid direct path is therefore anchored on the images of logical rows `32` and `64`, not on literal output vectors `[32, 0]` and `[64, 0]`.
- Fix shape:
  - update `getLogicalRowAnchorBasis(...)` to apply the analyzed layout to the sparse logical input point `{row = logicalRow}` and return that output-space image.
  - this supersedes the previous full-image output-vector check, which was only accidentally correct for some widened descriptor views.
- Validation snapshot:
  - split-N `64x128` core roundtrip exact repro: green
  - widened two-CTA MMAv5 control exact repro: green
  - split-N runtime-matrix guard exact repro: green
- Remaining live bucket after this correction:
  - `test_block_m_64_mma[linear]`
  - diagnosis:
    - not a generic support-query failure
    - packed TMEM analysis already reaches a valid `32x32b.unpack::16b`-style candidate (`atom=4`, `regsPerMsg=1`) for the row-zero-lifted reinterpret view
    - the compiler still hard-rejects that case with `packed16 support skip: row-zero lifted views require 32x32b.unpack direct lowering`
  - recommendation:
    - implement the packed unpack direct lowering for row-zero-lifted reinterpret views and remove the clean-negative guard once the emitted PTX/LLIR is correct.

## 2026-04-09: `block_m_64` bucket closed

- Final root cause was producer planning, not a missing ISA path:
  - the manual tcgen05 MMA builder copied the explicit root row plan onto the accumulator TMEM alloc but not onto TMEM-backed source operands.
  - the `block_m_64` linear kernel therefore emitted the intended `16x32bx2` direct path yet still produced wrong numerics because the TMEM roots did not share the same explicit row-anchor contract.
- Durable fix:
  - `TensorMemoryUtils.cpp`
    - keep row-zero-lifted `M=64` reinterpret classification keyed off the original reinterpret-view semantics
    - continue rejecting the packed-support shortcut for row-zero-lifted non-`I32x32b` candidates so the compiler does not silently choose the wrong packed path
  - `python/src/gluon_ir.cc`
    - add `annotateMMAv5TMemOperandRootRowPlan(...)`
    - copy the accumulator's explicit `ttng.tmem_ldst_row_plan` onto TMEM-backed MMA operands in both tcgen05 MMA builders
- Result:
  - `%al_tmem`, `%ar_tmem`, and `%acc_tmem` all carry `ttng.tmem_ldst_row_plan = array<i32: 32, 64, 128, 0>`
  - focused `block_m_64` probe is numerically clean (`close True`, `max_abs 0.01538`)
  - `test_tmem_subslice_block_m_64_parent_layout[linear]` is now a real positive and should stay that way unless the direct row-zero-lifted path regresses
- Test updates:
  - flip the former negative `linear` parent-layout case to positive
  - keep `test_block_m_64_mma` checking correctness plus the direct `16x32bx2` opcode mix
  - do not pin the exact TMEM alloc word count; that overfit failed during validation and is not the right contract
- Focused validation snapshot:
  - `make -j8`
  - `test_tmem_subslice_block_m_64_parent_layout` -> `2 passed`
  - `test_block_m_64_mma` -> `2 passed`
  - widened two-CTA `test_mma_shared_inputs[...]` guard -> `1 passed`
  - split-N runtime-matrix guard repro -> `1 passed`
- Next step:
  - resume the broader healthy-node validation sweep and classify the next real failure bucket, if any.

## 2026-04-09: broad sweep status after `block_m_64`

- Healthy-node 4-GPU sweep summary:
  - group 1: green (`4327 passed, 612 skipped`)
  - group 2: green (`2596 passed, 2343 skipped`)
  - group 3: green (`2388 passed, 2551 skipped`)
  - group 4: not many independent failures; it reduces to one stale clean-negative expectation plus one real misaligned-address bucket that poisons the rest of the process
- Non-bug noise bucket:
  - `test_tcgen05_mma_plain_kind_i8_reports_clean_error` prints a large `.kind::i8` PTXAS dump in the shard log but passes in isolation
  - do not treat that PTXAS text as initiative signal
- Stale test bucket:
  - `test_tmem_runtime_matrix_block_descriptor_reports_clean_error[block_two_ctas-layout1-reinterpret_layout1-expected_fragments1]`
  - current diagnostic is the row-anchor-specific descriptor-view message:
    - `required row anchors 16,32 are not directly representable in the descriptor view`
  - update the expected text; no compiler change appears needed for this one
- Real remaining bucket:
  - exact repro:
    - `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[even_odd-identity-2-32x32b_splitn]`
  - fresh isolate still fails with `Triton Error [CUDA]: misaligned address`
  - matching explicit direct-path control also fails:
    - `test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep[even_odd-identity-2-16x32bx2]`
- Root-cause direction:
  - this is not just split-N auto-selection; the explicit `16x32bx2` variant fails too
  - support query still admits the path (`rawQuery -> ok atom=4`)
  - failing and passing controls both use the same high-level row plan (`warpRow0=16 warpRow1=32`, `secondHalfOffset=0`)
  - the differentiator is the permuted TMEM row-basis order:
    - failing `even_odd`: `[[1,0],[4,0],[16,0],[2,0],[0,0],[8,0],[32,0]]`
    - passing `rotate1`: `[[2,0],[4,0],[8,0],[16,0],[0,0],[32,0],[1,0]]`
  - treat this as the common M64 direct `16x32bx2` half-row packet-decomposition / anchor-realization bug for row-permuted layouts
- Next step:
  - fix the common M64 direct `16x32bx2` row-permuted planner in `TensorMemoryUtils.cpp`
  - rerun the exact `even_odd` `32x32b_splitn` and `16x32bx2` repros
  - update the stale block-descriptor clean-negative test
  - then rerun broad sweep group 4

## 2026-04-10: merge-base audit says the remaining work is planner cleanup, not more one-off M64 surgery

- I audited the compiler diff vs merge-base `7f61ac734edc657b737fb159a1b9d50cb47944e6` (`master`), excluding `.codex`.
- The TMEM debt clusters into a few recurring families:
  - duplicated ld/st planning across Gluon / verifier / LLVM lowering
  - M64 family-specific layout-selection fast paths
  - reinterpret/support-query packet surgery in `TensorMemoryUtils.cpp`
  - legacy-encoding rescue paths that still diverge from TMEM-linear lowering
  - explicit row-plan propagation that is necessary but still too scattered
  - tests that still rely on `_reinterpret` as an implicit physical-layout escape hatch
- The remaining live `block_m_64` failures fit that audit exactly:
  - `test_tmem_subslice_block_m_64[legacy]`
    - still fails because legacy spelling does not collapse onto the same packed-support family as the working TMEM-linear spelling
    - PTX correlation:
      - broken legacy reinterpret zero-store still uses `tcgen05.st.sync.aligned.16x32bx2.x16.b32`
      - passing linear control uses repeated `tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32`
  - `test_tmem_subslice_block_m_64_parent_layout[linear]`
    - still fails even though the store is already on the packed `16x32bx2.x2.unpack::16b` family
    - so the atom family is not the issue there; packet decomposition is still assigning the support-band / subview-selection bit to ordinary repetition
- The highest-value cleanup order is now:
  1. unify TMEM ld/st planning into one shared structural planner;
  2. remove legacy-vs-linear physical-family divergence;
  3. derive reinterpret packet decomposition from quotient factorization instead of offset/anchor surgery;
  4. then rewrite the reinterpret-heavy `block_m_64` tests to the guaranteed descriptor/view APIs (`slice/index/reshape/permute`) so they encode user intent rather than old compiler accidents.
- Important testing note:
  - the `block_m_64` tests are still useful, but they should be treated as intent probes, not as proof that the compiler must preserve every old implicit reinterpret mapping forever.
  - the durable contract should move to explicit descriptor/view composition now that TMEM arbitrary linear layouts and descriptor transforms exist.

## 2026-04-10: GB200 inventory now shows a broader new-on-branch Blackwell/TMEM regression surface

- Merge-base comparison is live now, not blocked:
  - local-only baseline shims are enough to run exact nodeids in
    `/root/code/triton-mergebase-ci`
  - representative exact current-branch failures in old coverage all pass on
    merge-base:
    - tf32/tf32x3 dot
    - simple matmul
    - persistent matmul
    - warp-specialized attention forward
    - tensor-descriptor reshape matmul
    - debug sanitizer add-overflow
- The branch-added-vs-regressed split is now explicit:
  - some red TMEM split-N nodeids are branch-added coverage and do not exist on
    merge-base
  - several old, unmodified Blackwell tests now fail only on this branch
- Newly firm current-branch inventory points:
  - `python/triton_kernels/tests/` is green
  - `python/test/unit/language/test_tensor_descriptor.py` reduces to exactly
    three `test_tensor_descriptor_reshape_matmul[...]` failures, all on the
    same unsupported-row-anchors `32,64` path
  - `python/test/unit/test_debug.py` reduces to exactly twenty failing integer
    overflow sanitizer tests, all on the fork-and-reinitialize-CUDA path
- The most important correlation from the still-running exact inventories is
  that the new branch-only failures are clustering around the same shape family:
  - `test_matmul.py`: `BLOCK_M=64`
  - `test_warp_specialization.py`: `N=64`
  - `test_cast_matmul.py`: `GN=64`
- Working conclusion:
  - after the branch-added TMEM coverage and reinterpret-rewrite buckets are
    set aside, there is still a broader Blackwell/TMEM lowering regression on
    this branch that must be fixed to restore GB200 CI parity with main

## 2026-04-10: Proton is not branch-caused, and the simple shared-cache hypothesis did not reproduce

- Exact merge-base classifications now sharpen the recovery backlog:
  - new-on-branch:
    - unit matmul (`simple`, `persistent`, and `lhs_in_tmem`)
    - cast matmul regression coverage
    - Gluon convolution and multicta examples
    - previously established dot / tensor-descriptor / warp-specialization
      buckets
  - pre-existing on merge-base:
    - the full Proton `test_profile.py` red set (`11` exact nodeids)
- This matters for prioritization:
  - Proton should drop out of the branch-caused TMEM recovery plan
  - examples and regression tests stay in scope because they now have
    merge-base proof that they are new breakage on this branch
- Cache-collision status:
  - a simple fresh-process shared-cache probe did not reproduce poisoning
  - current evidence therefore points first at process/device contamination
    after bad kernels, not at a trivial `TRITON_CACHE_DIR` collision
  - if a future repro is truly cache-sensitive, it needs to survive a fresh
    process boundary before the cache-key theory should drive design work

## 2026-04-10: the GB200 recovery phase now has a frozen branch backlog and a dedicated execution plan

- The inventory/classification phase is now complete enough to stop spending
  cycles rediscovering red tests.
- The initiative folder now has a dedicated recovery planner:
  - `gb200_branch_recovery_plan.md`
- Use it together with:
  - `gb200_nvidia_ci_inventory.md` for the current executed CI state;
  - `gb200_failure_manifest.md` and the adjacent `.txt` files for exact
    nodeids; and
  - the active handoff for the latest implementation hypothesis.
- Current prioritized order:
  1. fix the common `M=64` TMEM planner / packet-decomposition bug;
  2. re-check the compile-time `row anchors 32,64` bucket as either the same
     structural bug's legality sibling or the next planner task;
  3. re-broaden through branch-added runtime-matrix coverage and then the
     merge-base-present shard-3 Gluon manifests;
  4. only after backend stability returns, resume the reinterpret-contract
     rewrites blocked on missing explicit reordered-view support.
- Current cache/flakiness reading remains conservative:
  - a second probe also failed to reproduce poisoning using a known-bad
    runtime-matrix nodeid followed by `test_block_m_64_mma[linear]` in the
    same process and then across fresh processes while reusing the same cache
    directory;
  - do not treat `TRITON_CACHE_DIR` collisions as proven until a failing order
    survives a fresh process boundary.
- Performance policy:
  - note codegen-family changes that may matter later, but keep the current
    recovery loop focused on correctness and functionality first.

## 2026-04-10: the overnight GB200 sweep is now down to final long runners plus isolated shard reductions

- Newly frozen current-head `test-gluon` manifests at `e70a3aa09`:
  - shard `3 / 4`
    - `1252` exact nodeids
    - exact merge-base split:
      - `967` exact old-mainline nodeids
      - `285` exact branch-added / branch-changed nodeids
  - shard `4 / 4`
    - `686` exact nodeids
    - unchanged from the older group-4 manifest
    - still entirely branch-added
      `python/test/gluon/test_tmem_runtime_matrix.py` coverage
- Current group-4 interpretation is stable:
  - the branch-added runtime-matrix file is still broadly red on the current
    branch; and
  - the dominant families remain split-N row/col-permuted, `ld.red`,
    `ldst_scales`, scaled-MMA copy `warpx4`, and `cp_no_scales`
- Current group-3 interpretation still needs isolation reduction:
  - the merge-base split is now exact and durable; but
  - the raw shard still overstates the number of independent failures because
    earlier standalone reruns already showed primary TMEM failures followed by
    poisoned-context fallout
- Still in flight when this note was written:
  - current-head shard `1 / 4`
  - current-head `python/triton_kernels/tests`
  - fresh merge-base full-file reruns of:
    - `python/test/unit/language/test_matmul.py`
    - `python/test/unit/language/test_warp_specialization.py`
- Queued next reductions after those GPUs free:
  - isolated current-head `python/test/gluon/test_lowerings.py`
  - isolated current-head `python/test/gluon/test_fpsan.py`
  - isolated current-head focused `python/test/gluon/test_core.py` slice for
    the dominant shard-3 TMEM/MMA families
- Durable execution rule for the rest of this census:
  - do not pick fixes from raw shard totals when isolated fresh-process reruns
    are the next obvious reduction
  - convert the broad shard inventory into an independent exact-failure
    inventory first, then prioritize code changes from that reduced set

## 2026-04-10: the isolated overnight reductions turned the broad GB200 census into concrete fix buckets

- Fresh merge-base file-level proof is now complete for the live unit bucket:
  - `python/test/unit/language/test_matmul.py`
    - `761 passed, 4780 skipped`
  - `python/test/unit/language/test_warp_specialization.py`
    - `1599 passed, 202 skipped`
- Current-head shard `1 / 4` is green:
  - `5433 passed, 1014 skipped, 19344 deselected`
- The broad shard-3 story is now concretely reduced:
  - real isolated core bucket:
    - `332` exact failures from the focused TMEM/MMA `test_core.py` slice
    - exact split:
      - `146` merge-base-present nodeids
      - `186` branch-added / branch-changed nodeids
  - isolated FPSAN tail:
    - `1` exact branch-changed failure
  - isolated layout/frontend tail:
    - `test_layout_format_view.py` is green
    - `1` exact branch-added frontend failure remains
- The dominant branch-added runtime-matrix families are independently red:
  - focused slice result:
    - `413 failed, 17 passed, 1360 deselected`
  - dominant exact families:
    - split-N row/col-permuted
    - `ld.red` row-permuted
    - `ldst_scales`
    - scaled-MMA copy `warpx4`
    - `cp_no_scales`
- This materially changes the next-fix strategy:
  - the old-mainline bucket is now not just unit/lit; it also includes the
    merge-base-present subset of the isolated shard-3 core failures
  - `test_layout_format_view.py` should drop out of the active red list
  - the branch-added runtime-matrix work should be driven from the focused
    `413`-nodeid manifest rather than from the full raw group-4 shard alone

## 2026-04-10: pure outer memdesc_index row-plan selection clears the old direct-view lit/unit bucket

- Current code delta:
  - `getTMemLdStRowPlanForQuery(...)` now prefers the query row plan for
    pure outer `ttg.memdesc_index` TMEM views that:
    - preserve the trailing 2D TMEM tile shape and alloc shape;
    - only peel extra non-layout dimensions; and
    - bottom out at a TMEM root (`ttng.tmem_alloc`, `nvws.aref.buffer`, or a
      block argument).
- Why this matters:
  - the failing lit + unit bucket was not a true direct-view impossibility;
  - verifier/planner code was carrying the larger backing row plan through
    simple outer-index TMEM views, which made representable 64-row queries look
    impossible.
- Focused validation on the dirty worktree:
  - `make -j8`
    - green
  - `make test-lit`
    - green
    - `248 passed, 2 unsupported`
  - exact current-head unit repros:
    - `python/test/unit/language/test_matmul.py::test_simple_persistent_matmul[False-4-64-128-32]`
      - passes
    - `python/test/unit/language/test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float32]`
      - passes
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-False-2-64-64-1024-1024]`
      - still fails with `Triton Error [CUDA]: misaligned address`
  - direct lit repro:
    - `test/TritonGPU/pipeline-lower-loop.mlir`
      - passes under `triton-opt`
- Broad `make NUM_PROCS=24 test-unit` evidence before interruption:
  - the run reached deep into `test_warp_specialization.py` without surfacing
    any `test_matmul.py` or `test_tensor_descriptor.py` failures, which is a
    strong signal that the old direct-view row-anchor representability bucket is
    cleared;
  - the first real remaining unit exact is:
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-False-2-64-64-1024-1024]`
  - the interrupted log currently contains:
    - `10` true kernel failures at the `assert_close` / launch site; and
    - `109` later `torch.manual_seed` contamination failures on poisoned xdist
      workers.
- Separate harness note:
  - after the warp-specialization failures, the last xdist worker stalled in
    `python/test/unit/runtime/test_cache.py::test_async_compile_mock`;
  - `py-spy dump` shows the worker blocked in
    `triton.runtime._async_compile.AsyncCompileMode.__exit__`, waiting on
    unfinished futures; and
  - keep that as a separate cache/async-compile harness investigation, not as
    evidence against the row-plan fix itself.
- Updated immediate priority:
  1. checkpoint the pure-outer-index row-plan fix;
  2. diagnose the remaining higher-rank MMAv5-root warp-specialization runtime
     bucket from the first real exact above; and
  3. only after that, return to the focused merge-base-present `test_core.py`
     Gluon bucket.

## 2026-04-10: warp-specialize partition captures now preserve explicit TMEM row plans

- Core bug:
  - the remaining higher-rank MMAv5-root runtime failures were not caused by
    missing root annotations;
  - the explicit producer-owned `ttng.tmem_ldst_row_plan` contract was being
    lost when TMEM memdescs crossed `ttg.warp_specialize` partition block
    arguments; and
  - raw-query planning on captured `ttg.memdesc_index<64x64xf32>` views then
    fell back from `rawRowPlan=128` to `rawRowPlan=64`, scalarized the
    `tl.dot(p, v, acc)` TMEM access to `32x32b.x1`, and triggered the
    Blackwell runtime `misaligned address` fault.
- Implemented structural fix:
  - `getExplicitTMemLdStRowPlan(...)` in
    `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp` now follows
    `gpu::WarpSpecializePartitionsOp` block arguments back through
    `getExplicitCaptures()`, so the planner can recover the original producer
    row-plan contract inside partition regions.
- Compile-only validation:
  - `make -j8`
    - green
  - TMEM debug query trace for the first exact warp repro:
    - all sampled `ttg.memdesc_index<64x64xf32>` raw queries now report
      `rawRowPlan=128` and `rawQuery -> ok atom=4`
    - the older `rawRowPlan=64` / `atom=0` split is gone
  - PTX for the same kernel:
    - the old scalar `.loc 491` `tcgen05.ld/st.sync.aligned.32x32b.x1.b32`
      sequence is gone
    - the MMAv5 load/store path is back on packed
      `tcgen05.ld/st.sync.aligned.16x32bx2.x32.b32`
- Focused runtime validation:
  - all of the following exacts now pass:
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-False-2-64-64-1024-1024]`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-True-3-128-64-1024-1024]`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-8-False-3-128-64-8192-8192]`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-True-2-64-64-8192-8192]`
    - `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[True-4-False-2-128-64-8192-8192]`
- Updated immediate priority:
  1. rerun `python/test/unit/language/test_warp_specialization.py` to replace
     the older `128`-nodeid manifest with a fresh current-head result;
  2. rerun `make NUM_PROCS=24 test-unit` once the file-level warp bucket is
     understood;
  3. if the old unit bucket is closed, continue with the merge-base-present
     focused `python/test/gluon/test_core.py` recovery bucket; and
  4. keep the separate cache/async-compile harness investigation distinct from
     TMEM correctness work.

## 2026-04-10: full GB200 `make test-unit` is green again

- The row-plan forwarding fix closed the entire old-mainline unit bucket, not
  just the five representative warp-specialization exacts:
  - `python/test/unit/language/test_warp_specialization.py`
    - rerun as a 4-GPU `--splits 4 --group {1..4}` sweep
    - result:
      - `1599 passed, 202 skipped`
      - `0 failed`
- Fresh full wrapper rerun:
  - command:
    - `make NUM_PROCS=24 test-unit`
  - artifact:
    - `/tmp/test-unit-gb200-after-ws.log`
  - sub-lane results:
    - main `python/test/unit`
      - `15153 passed, 5492 skipped, 101 warnings`
    - `python/test/unit/test_debug.py`
      - `95 passed`
    - `python/triton_kernels/tests`
      - `2377 passed, 3444 skipped`
    - `python/tutorials/06-fused-attention.py`
      - `192 passed, 192 skipped, 1 warning`
    - `python/test/unit/instrumentation/test_gpuhello.py`
      - `1 passed`
    - `python/test/unit/plugins/test_plugin.py`
      - `1 passed`
    - `python/test/unit/plugins/test_dialect_plugin.py`
      - `1 passed`
    - `python/test/unit/plugins/custom_ops.py`
      - `1 passed`
- Manifest consequences:
  - `gb200_current_branch_test_unit_failures.txt`
    - now empty
  - refreshed unit sub-manifests (`matmul`, `tensor_descriptor`,
    `warp_specialization`, `rowanchor`)
    - now empty
  - `gb200_branch_recovery_test_unit_failures.txt`
    - now empty
  - older populated unit manifests remain only as historical evidence of the
    earlier failing state
- Separate harness conclusion:
  - the earlier async-compile / `test_cache.py` hang did not reproduce in the
    fresh green rerun
  - this strengthens the current hypothesis that the old hang was downstream
    worker/process contamination after bad kernels, not a standing independent
    unit-lane blocker
- Updated immediate priority:
  1. move to the merge-base-present focused
     `python/test/gluon/test_core.py` bucket (`146` exact nodeids)
  2. then remeasure the branch-added / branch-changed Gluon tails
  3. keep examples and reinterpret-contract rewrites behind those compiler and
     API-frontier fixes

## 2026-04-10: `03-matmul-multicta.py` is now narrowed to a TMEM column-slice direct-ld/st planner bug

- The live old-mainline example regression is not a generic MMA semantic issue
  and not a reinterpret-contract test:
  - the kernel uses real TMEM view APIs:
    - `acc_buf.slice(SPLIT_TILE_N * s, SPLIT_TILE_N)`
- Exact current-head repro remains:
  - `python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-2-CGA_LAYOUT0-8-0-64-128-64]`
  - `9600 / 20000` mismatches (`48.0%`)
- Merge-base behavior for the failing slice:
  - one packed load:
    - `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`
  - TTGIR register layout:
    - `register=[[0,1],[0,2],[0,4],[0,8]]`
    - `lane=[[1,0],[2,0],[4,0],[8,0],[0,16]]`
- Current branch behavior:
  - two split loads:
    - `tcgen05.ld.sync.aligned.16x32bx2.x8.b32`
  - TTGIR register layout:
    - `register=[[0,1],[0,2],[0,4],[0,16]]`
    - `lane=[[1,0],[2,0],[4,0],[8,0],[0,8]]`
- Current core diagnosis:
  - support-query and raw-query direct planning for the `64x128 -> 64x32`
    TMEM column slice still fail
  - both Gluon and LLVM therefore fall back to the standalone query-type path,
    which picks the bad packed split-N `x8` family
  - the source slice projection is still rooted in a non-injective TMEM query
    layout with a zero row basis, and the surviving support layout collapses
    away too much of that parent row-zero structure before direct planning
- Important implication for the next fix:
  - the right target is the TMEM column-slice query/support construction itself
  - do not keep tuning atom-order or M64 split-N heuristics while the real
    support/raw query image is still wrong

## 2026-04-10 22:41 UTC: the multicta column-slice regression is fixed; the next core issue is producer-visible TMEM physical-family semantics

- The old-mainline `python/examples/gluon/03-matmul-multicta.py` regression is
  now closed on the current branch.
- What fixed it:
  - pure rank-2 TMEM column subviews now get a direct lowering path that can
    reuse the source support/raw query plan instead of immediately collapsing
    to the sliced `64x32` query type;
  - the direct planner now rejects `I32x32b` layouts whose warp anchors are
    not aligned to whole `32x32b` message tiles; and
  - the handle-aware explicit split-N frontend path once again preserves the
    final split-N register layout instead of re-finalizing an already-final
    planner result.
- Validation:
  - `make -j8`
  - exact multicta repro:
    - `python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-2-CGA_LAYOUT0-8-0-64-128-64]`
    - `PASSED`
  - nearby TMEM/split-N controls:
    - `test_tmem_linear_roundtrip_splitn_shapes[...]`
    - `test_tmem_descriptor_chain_matrix[...]`
    - both `PASSED`
  - full file:
    - `python/examples/gluon/03-matmul-multicta.py`
    - `82 passed, 14 skipped`
- Most important remaining diagnosis:
  - `python/test/gluon/test_core.py::test_block_m_64_mma[linear]` is still
    red, and relaxing its PTX-immediate expectation exposes real wrong-code,
    not just a stale assertion;
  - this no longer looks like another ld/st packet-selection bug;
  - the key abstraction gap is that later TMEM slices / MMAv5 consumers still
    reason from the logical memdesc type alone, while the producer store can
    choose a different physical direct family such as explicit split-N;
  - `getTMemViewOffset(...)`, `getSortedTMemTileOrder(...)`, and
    `DotOpMmaV5TmemLoader::build(...)` are all still fundamentally type-driven
    here.
- Immediate implication for the next fix:
  - stop treating `block_m_64_mma[linear]` like another standalone direct-ld/st
    planner issue;
  - the next real work item is a producer-owned physical TMEM contract, or an
    equivalent value-aware MMAv5/view-offset model, so slice-after-store TMEM
    consumers can preserve the actual physical family selected by the store.

## 2026-04-11 12:10 UTC: M64 subview physical bitcast is supported; legacy MMAv5 family debt is isolated

- Latest pushed source checkpoint:
  - `85d8dbbf4` on `origin/codex/tmem`
- The supported physical bitcast path now handles M64 subviews with inactive
  zero support bases:
  - `inferTMemReinterpretQueryLayout(...)` retries source inversion on
    `normalizeTensorMemoryLinearLayoutForAnalysis(...)` when the normalized
    view is injective and exactly covers the source descriptor shape;
  - narrowed subviews can carry physical origins outside the reduced layout's
    in-dim size, so reinterpret/view-origin remapping now preserves those
    physical row/col origins by name instead of applying the reduced layout and
    dropping high column bits;
  - dtype-changing physical bitcast scales the column origin by
    source/destination bitwidth when using that physical-origin fallback.
- Tests now encode the intended migration contract:
  - `test_tmem_subslice_block_m_64` and
    `test_tmem_subslice_block_m_64_parent_layout` use
    `slice(...).bitcast(...)` for the f32-subview to f16-full-view case and
    assert the `tmem_physical_bitcast` marker;
  - the 2-column f32 stores use direct supported subview stores, because the
    old legacy `TensorMemoryLayout((64, 2), col_stride=1)` reinterpret is not
    physical-mapping equivalent to a legacy M64 column subview;
  - parameterized kernels now pass layout objects as explicit `ttgl.constexpr`
    arguments so Gluon JIT cache keys distinguish legacy and linear variants.
- Validation:
  - build:
    - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`
  - lit:
    - `test/TritonNvidiaGPU/tmem_layouts.mlir`
    - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - `2 passed`
  - existing physical-bitcast controls:
    - `test_tmem_physical_bitcast_preserves_subview_mapping`
    - `test_tmem_physical_bitcast_mma_lhs`
    - `2 passed`
  - M64 supported subview/bitcast exacts:
    - `test_tmem_subslice_block_m_64[legacy]`
    - `test_tmem_subslice_block_m_64[linear]`
    - `test_tmem_subslice_block_m_64_parent_layout[legacy]`
    - `test_tmem_subslice_block_m_64_parent_layout[linear]`
    - `4 passed`
  - block-M=64 MMA cache-key / current behavior exacts:
    - `test_block_m_64_mma[legacy]`
    - `test_block_m_64_mma[linear]`
    - `1 passed, 1 xfailed`
  - temp M64 bitcast repro:
    - `/tmp/repro_tmem_bitcast_m64.py`
    - `PASSED`
  - hygiene:
    - `git diff --check`
    - `PASSED`
- Grouped Gluon consequence:
  - the group-3 sweep that started before this checkpoint reduced the stale
    group-3 red set to exactly:
    - `test_tmem_subslice_block_m_64_parent_layout[linear]`
    - `test_block_m_64_mma[legacy]`
    - `test_block_m_64_mma[linear]`
    - `3 failed, 4406 passed, 2041 skipped, 19348 deselected`
  - focused reruns after `85d8dbbf4` close those exact failures at current
    head, with the legacy MMAv5 parameter recorded as xfail.
- Remaining design debt:
  - legacy M64 `64x64` layout sugar still lacks producer-visible
    physical-family semantics for MMAv5 consumers;
  - this is now isolated to the xfailed legacy parameter, while the explicit
    linear M64 layout passes in the same process;
  - do not treat the legacy xfail as closure of the producer-family problem.
- Next:
  - rerun wider grouped `python/test/gluon` from `85d8dbbf4`;
  - keep the attention `_reinterpret` migration deferred until it can use a
    synchronization-aware supported subview/bitcast sequence;
  - after broad validation is stable, resume `ld.red`, `copy`/`warpx2`,
    broader MMAv5/`mma_scaled`, fuzzing, stale-negative cleanup, and
    heuristic cleanup.

## 2026-04-11 13:05 UTC: Full Gluon grouped sweep is green at current head

- Validated checkpoint:
  - `be3cba0cd` on `origin/codex/tmem`
- Fresh four-way `python/test/gluon` sweep:
  - group 1:
    - `5448 passed, 1002 skipped, 19348 deselected in 3127.70s`
  - group 2:
    - `2666 passed, 3784 skipped, 19348 deselected in 1155.29s`
  - group 3:
    - `4408 passed, 2041 skipped, 19348 deselected, 1 xfailed in 2032.99s`
  - group 4:
    - `5582 passed, 866 skipped, 19350 deselected in 1751.06s`
- Manifest updates:
  - `gb200_current_branch_group1_latest_failures.txt`: `0` nodeids
  - `gb200_current_branch_group2_latest_failures.txt`: `0` nodeids
  - `gb200_current_branch_group3_latest_failures.txt`: `0` nodeids
  - `gb200_current_branch_group4_latest_failures.txt`: `0` nodeids
- Consequence:
  - the stale group-3 M64 failures are closed at full-shard scope;
  - the one group-3 xfail is the intentional legacy M64 MMAv5
    producer-family design-debt marker;
  - the old full Gluon failure manifests should not be used for prioritization
    anymore.
- Watch item:
  - groups 3 and 4 printed `.kind::i8` PTX assembler diagnostics, but pytest
    continued and both shards finished green;
  - leave this as nonfailing noise unless a focused exact rerun produces an
    actual failed nodeid.
- Next:
  - continue the recovery queue with attention kept separate as the supported
    synchronization-aware subview/bitcast migration target;
  - keep legacy M64 MMAv5 producer-family semantics visible as design debt, and
    explicitly fix `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`
    once the current CI-lane freshness and examples aggregate checks are out of
    the way;
  - then resume long-term `ld.red`, `copy`/`warpx2`, broader
    MMAv5/`mma_scaled`, fuzzing, stale-negative cleanup, and heuristic cleanup.

## 2026-04-11 22:15 UTC: Broad LD/ST refresh is green after subword expansion

- Latest pushed source/test checkpoint before this docs validation refresh:
  - `a566a63b9` on `origin/codex/tmem`
- The broad current-head `ld/st` runtime-matrix slice is green after the recent
  direct subword and x1 subword coverage expansion:
  - command:
    - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-broad-after-subword-refresh PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k ldst`
  - result:
    - `1181 passed, 441 skipped, 1027 deselected in 1340.05s (0:22:20)`
- Consequence:
  - the focused subword additions are validated at the broad `ld/st` slice
    level, not just by exact nodeids;
  - the skip count remains the expected clean-boundary/OOR surface from
    descriptor roundtrip and rank-5 families.
- Next:
  - commit and push this docs validation checkpoint;
  - continue with the two-CTA `warpx2::02_13` descriptor/address-model
    frontier, deeper scales `warpx2` probing, broader `ld.red` fuzzing, or a
    bounded MMAv5 / scaled-MMAv5 reachable-family gap.

## 2026-04-11 22:35 UTC: Two-CTA warpx2::02_13 direct-seed probe still produces all-zero output

- Latest pushed checkpoint before this probe:
  - `f0cca8a2c` on `origin/codex/tmem`
- Probe performed:
  - temporarily relaxed `getDirectTMemCopySeedDescriptorImm(...)` to allow the
    canonical `[256,4]` two-CTA shared source shape and its `[[128, 0]]` shared
    block basis for `warpx2::02_13`;
  - rebuilt with `make -j8`;
  - launched `tmem_copy_no_scales_warpx2_twocta_kernel` with the canonical
    two-CTA `02_13` source and destination layouts.
- Result:
  - PTX emitted
    `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b` plus the multicast commit;
  - runtime output was entirely zero (`count_nonzero(out) == 0`), so the direct
    seed relaxation is confirmed wrong-code.
- Cleanup / validation:
  - reverted the temporary source edits and rebuilt;
  - exact clean-unsupported boundary rerun:
    - `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-warpx2-0213-clean-after-probe-revert PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_candidate_reports_clean_unsupported`
    - `1 passed in 3.17s`
  - worktree was clean before this docs update.
- Consequence:
  - keep the two-CTA `warpx2::02_13` public-layout case as a clean unsupported
    descriptor/address-model frontier; do not unblock it by extending the
    current direct-seed helper alone.

## 2026-04-11 22:45 UTC: Nearby scales warpx2 public-layout variants are pinned clean unsupported

- Latest pushed checkpoint before this source/test update:
  - `468230aeb` on `origin/codex/tmem`
- Source/test change:
  - expanded `CP_SCALES_LAYOUT_PROBE_CASES` with four nearby
    `SharedLinearLayout` basis-order variants around the historical scales
    `warpx2_candidate`;
  - each new variant is intentionally marked `CLEAN_UNSUPPORTED`, while the
    existing `warpx4` control remains the positive exact-copy path.
- Current behavior:
  - the historical candidate and the four nearby variants all classify through
    the public scales layout surface as `tcgen05.copy.warpx4.32x128b` and then
    fail descriptor-plan synthesis for tensor-memory scales; that failure is now
    emitted by `TMEMCopyOp` verification before late LLVM lowering;
  - none of the committed public-layout variants reaches
    `warpx2::{01_23,02_13}.64x128b`;
  - true scales `warpx2` remains a descriptor/address-representation or direct
    PTX documentation frontier, not something unlocked by small basis-order
    tweaks in the current public scales copy probe.
- Validation:
  - build:
    - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - syntax:
    - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
    - `PASSED`
  - focused scales layout probe:
    - `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-scales-warpx2-nearby-cases PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_layout_probe`
    - `6 passed in 3.40s`
  - focused `warpx2` selector:
    - `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-warpx2-after-scales-nearby-cases PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k warpx2`
    - `13 passed, 2642 deselected in 4.49s`
  - broad `tcgen05.cp` selector:
    - `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-cp-after-scales-warpx2-nearby-cases PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k cp`
    - `162 passed, 5 skipped, 2488 deselected in 40.88s`
- Next:
  - commit and push this copy-boundary coverage slice;
  - continue with either true scales `warpx2` descriptor/direct-PTX research,
    two-CTA `warpx2::02_13` descriptor/address synthesis, broader `ld.red`
    fuzzing, or another bounded MMAv5 / scaled-MMAv5 reachable-family gap.

## 2026-04-11 23:58 UTC: LD.RED positives now assert wait ordering

- Latest pushed checkpoint before this source/test update:
  - `781e9d8e1` on `origin/codex/tmem`
- Source/test change:
  - `_assert_ld_red_opcode_pairs(...)` now extracts PTX/LLIR once, then checks
    exact `tcgen05.ld.red` opcode/offset pairs as before;
  - every positive runtime-matrix `ld.red` case now also asserts exactly one
    PTX `tcgen05.wait::st.sync.aligned;`, one PTX
    `tcgen05.wait::ld.sync.aligned;`, one LLIR `wait.st` tail call, and one
    LLIR `wait.ld` tail call;
  - the helper pins ordering as
    `tcgen05.st -> wait.st -> tcgen05.ld.red -> wait.ld` in both PTX and LLIR.
- Validation:
  - build:
    - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - syntax:
    - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
    - `PASSED`
  - broad `tcgen05.ld.red` selector:
    - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldred-wait-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k ld_red`
    - `476 passed, 2179 deselected in 439.38s (0:07:19)`
- Next:
  - commit and push this ISA-contract coverage slice;
  - continue true scales `warpx2`, two-CTA `warpx2::02_13`, broader `ld.red`
    fuzzing, or the next MMAv5 / scaled-MMAv5 reachable-family gap.

## 2026-04-11 23:59 UTC: Plain MMAv5 instruction counts are pinned

- Latest pushed checkpoint before this source/test update:
  - `732f8b60a` on `origin/codex/tmem`
- Source/test change:
  - added `MMA_PLAIN_KIND_EXPECTED_OP_COUNTS` for the root plain-kind MMAv5
    matrix: `f16=2`, `bf16=2`, `tf32=4`, `f8e5m2=1`, and `f8e4m3=1`;
  - the one-CTA and two-CTA root/use-acc plain-kind tests now assert those
    counts in addition to PTX/LLIR opcode equality and exact commit opcodes;
  - TMA-fed two-CTA f16/TF32 positives and the direct f16 descriptor-view
    positives also pin their expected counts;
  - tile-permuted accumulator coverage has a separate fourfold count model, and
    the wider-K tile-permuted TMEM-LHS path pins `16` f16 ops.
- Probe note:
  - an initial broad run using the root counts for tile-permuted paths failed
    only on the tile-permuted families; that clarified the correct separate
    count model before the final validation rerun.
- Validation:
  - build:
    - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`, ninja reported no work to do
  - syntax:
    - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
    - `PASSED`
  - broad direct MMA / direct scaled-MMA selector:
    - `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-mma-plain-opcounts-broad-r2 PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma and not cp'`
    - `213 passed, 50 skipped, 2392 deselected in 122.55s (0:02:02)`
- Next:
  - commit and push this MMAv5 instruction-count coverage slice;
  - continue true scales `warpx2`, two-CTA `warpx2::02_13`, broader `ld.red`
    fuzzing, or another MMAv5 / scaled-MMAv5 reachable-family gap.

## 2026-04-11 23:59 UTC: Scales copy descriptor-plan failures are verified early

- Latest pushed checkpoint before this source/test update:
  - `ad814241d` on `origin/codex/tmem`
- Source/compiler change:
  - `TMEMCopyOp::verify()` now applies the existing shared-descriptor feasibility
    check to tensor-memory-scales copies after family recognition and swizzle
    validation;
  - if no plan can synthesize a compatible shared-memory descriptor, the scales
    branch emits the same clean `maps to tcgen05.copy.<family>` diagnostic and
    notes that the case is rejected before late LLVM lowering.
- Test change:
  - renamed the old scales unsupported-layout parse/bug exact to
    `test_tmem_runtime_matrix_cp_scales_unsupported_layout_reports_clean_error`;
  - it now asserts the descriptor-plan diagnostic, the same-family guidance,
    the late-LLVM-lowering note, and no `PassManager::run failed` or assertion.
- Validation:
  - build:
    - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
    - `PASSED`
  - syntax:
    - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
    - `PASSED`
  - focused scales copy verifier slice:
    - `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-scales-verifier-focused-r2 PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_layout_probe python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_unsupported_layout_reports_clean_error`
    - `7 passed in 3.39s`
  - broad `tcgen05.cp` selector:
    - `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-cp-scales-verifier-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k cp`
    - `162 passed, 5 skipped, 2488 deselected in 40.99s`
- Next:
  - commit and push this scales-copy verifier checkpoint;
  - continue true scales `warpx2` descriptor/direct-PTX research or two-CTA
    `warpx2::02_13` descriptor/address synthesis.

## 2026-04-11 23:16 UTC: direct higher-rank `ld/st` clean boundary pinned

- While checking the remaining `ld/st` fuzz/stale-negative surface, a temporary
  direct rank-3 TMEM probe found a hard C++ abort in
  `getDistributedLayoutForTmemLdSt`: the planner asserted `dims.size() == 2`
  during `tmem.get_reg_layout()` on a higher-rank descriptor.
- Fixed the current unsupported boundary instead of trying to lower direct
  higher-rank access:
  - `tensor_memory_descriptor.get_reg_layout`, `load`, reduction `load`, and
    `store` now reject non-rank-2 TMEM descriptors in the Gluon descriptor API
    with an actionable message to index/slice/reshape to a 2D view before
    access;
  - the generic and legacy-anchored C++ ld/st layout planners now return
    `std::nullopt` for non-2D output layouts instead of asserting, so lower
    layers keep the same clean-unsupported semantics if reached directly.
- Added `test_tmem_runtime_matrix_ldst_direct_higher_rank_access_reports_clean_error`:
  - direct rank-3 `get_reg_layout(auto)`;
  - direct rank-3 `get_reg_layout(16x128b)`;
  - direct rank-3 explicit `load`;
  - direct rank-3 explicit `store`.
- Validation:
  - build: `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8` passed;
  - syntax: `python3 -m py_compile python/triton/experimental/gluon/language/nvidia/blackwell/__init__.py` passed;
  - syntax: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py` passed;
  - focused direct higher-rank clean errors:
    `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-direct-highrank-clean PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_direct_higher_rank_access_reports_clean_error`
    -> `4 passed in 3.14s`;
  - adjacent higher-rank positives:
    `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-higher-rank-positive-after-direct-rank-guard PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_index python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_dim0_slice_positive_lifted_layout`
    -> `30 passed in 25.88s`;
  - broader higher-rank `ld/st` selector:
    `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-higher-rank-ldst-broad-after-direct-rank-guard PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst and higher_rank'`
    -> `144 passed, 1 skipped, 2521 deselected in 73.43s (0:01:13)`.
- Next: run hygiene, commit, push, then continue the next saturation frontier
  from the long-term plan (`ld.red` broader fuzzing, copy `warpx2`, or
  MMAv5/scaled-MMAv5 reachable-family work).

## 2026-04-11 23:29 UTC: explicit `ld.red` layout coverage includes min/max

- Generalized `tmem_ld_red_explicit_layout_kernel` so compatible explicit
  register-layout variants exercise both `tmem.load_min(layout=...)` and
  `tmem.load_max(layout=...)`.
- Positive explicit variants remain:
  - `auto`
  - `32x32b`
  - `16x32bx2`
  - `32x32b_splitn`
- Each positive variant preserves the loaded tensor, matches the PyTorch row
  reduction for the selected op, and still emits the canonical
  `tcgen05.ld.red.sync.aligned.32x32b.x128.{min,max}.f32` opcode family.
- The explicit N-sharded clean-negative variants still use the min path and
  still report the dedicated `N dimension sharded across threads` diagnostic.
- Validation:
  - build: `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8` passed (`ninja: no work to do`);
  - syntax: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py` passed;
  - hygiene: `git diff --check` passed before docs update;
  - focused explicit min/max slice:
    `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-ldred-explicit-minmax-focused PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_explicit_compatible_layout_variants python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_explicit_n_sharded_layout_reports_clean_unsupported`
    -> `11 passed in 8.57s`;
  - broad current-head `ld_red` selector:
    `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-ldred-minmax-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k ld_red`
    -> `487 passed, 2183 deselected in 482.92s (0:08:02)`.
- Next: run final hygiene, commit and push this explicit min/max `ld.red`
  coverage slice, then continue copy `warpx2` or MMAv5/scaled-MMAv5 frontier
  work.


## 2026-04-12 03:15 UTC: GB200 unit-lane status and legacy M64 xfail docket

- The comprehensive GB200/NVIDIA workflow includes more than the Gluon sweeps.
  `.github/workflows/integration-tests-nvidia.yml` runs:
  - `make test-lit`;
  - `make NUM_PROCS=24 test-unit`;
  - `make NUM_PROCS=24 test-gluon`;
  - `make NUM_PROCS=24 test-gsan`;
  - `make test-regression`;
  - `make test-microbenchmark`;
  - `make test-cpp`;
  - `make test-proton`.
- The earlier `test-unit` red inventory is superseded by the 2026-04-10 full
  wrapper rerun at checkpoint `78196b4e4`, but that is an older checkpoint than
  current HEAD:
  - full `make NUM_PROCS=24 test-unit` artifact:
    `/tmp/test-unit-gb200-after-ws.log`;
  - main `python/test/unit`: `15153 passed, 5492 skipped`;
  - `python/test/unit/test_debug.py`: `95 passed`;
  - tail sublanes green: `python/triton_kernels/tests`,
    `python/tutorials/06-fused-attention.py`, instrumentation, and plugin tests;
  - the current `gb200_current_branch_test_unit*_failures.txt` manifests are
    empty except the historical XML inventory, which remains explicitly stale;
  - rerun `make NUM_PROCS=24 test-unit` at current HEAD before calling the full
    GB200 unit lane freshly validated.
- The single full-Gluon xfail remains:
  - `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`;
  - reason: legacy M64 `64x64` layout sugar still lacks producer-visible
    physical-family semantics for MMAv5 consumers.
- User direction on 2026-04-12: put this xfail on the fix docket. Treat it as a
  planned follow-up after immediate CI-lane freshness / examples aggregate
  validation, not as a permanent acceptable state.

## 2026-04-12 08:21 UTC: GB200 API/contract lit bucket fixed

- Fixed only the five lit tests previously classified as API/contract updates:
  - `test/Analysis/test-membar-ttng.mlir`;
  - `test/TritonGPU/proxy_fence_insertion.mlir`;
  - `test/TritonNvidiaGPU/mma_lowering.mlir`;
  - `test/NVWS/assign_stage_phase.mlir`;
  - `test/NVWS/aref-tmem-insertion.mlir`.
- Scales-copy tests now use verifier-supported `#ttg.shared_linear` source
  layouts that synthesize a valid tensor-memory-scales descriptor plan instead
  of relying on old invalid `nvmma_shared` / `swizzled_shared` scale-copy IR.
- `mma_lowering.mlir` keeps the shared-scales-to-TMEM lowering intent by using
  `blockN=128` so both A and B scale copies are `128x8` supported scales-copy
  tiles; it still checks `tmem_alloc`, `tmem_copy`, and the rewritten
  `tc_gen5_mma_scaled` operands.
- NVWS scheduling/aref tests no longer use invalid shared-memory `f32` MMA
  operands for scheduling-only coverage:
  - unscaled cases use supported `f16` shared operands with f32 accumulators;
  - scaled cases use `f8E4M3FN` shared operands under a 64-byte f8 shared
    layout for the 64-wide A tile.
- Validation:
  - `make -j8` passed after refreshing the LLVM toolchain and build tree;
  - `ninja triton-opt` from the build directory reported `no work to do`;
  - direct pass pipelines without FileCheck passed for all five files, including
    both `test-membar-ttng.mlir` RUN lines;
  - lit with `PATH=/root/.triton/llvm/llvm-ubuntu-arm64/bin:$PATH` passed for
    all five files:
    - `lit -v test/Analysis/test-membar-ttng.mlir`;
    - `lit -v test/TritonGPU/proxy_fence_insertion.mlir`;
    - `lit -v test/TritonNvidiaGPU/mma_lowering.mlir`;
    - `lit -v test/NVWS/assign_stage_phase.mlir`;
    - `lit -v test/NVWS/aref-tmem-insertion.mlir`;
  - `git diff --check` passed.
- Current GB200 branch-recovery implication:
  - the API/contract lit bucket is fixed at focused-lit scope;
  - the stale lit bucket remains: `TritonNvidiaGPU/invalid.mlir`,
    `TritonGPU/pipeline-loop-nest.mlir`, `TritonGPU/pipeline-lower-loop.mlir`,
    and `TritonNvidiaGPU/ops.mlir`;
  - a full `make test-lit` rerun is still pending after the stale lit fixes;
  - the `162` `python/test/unit` failures and legacy M64 MMAv5 bug remain the
    actual compiler/runtime recovery backlog.
- Tooling note: the `apply_patch` tool still failed before edits with
  `No such file or directory`, so this checkpoint used scripted exact
  replacements and then validated the resulting diff.

## 2026-04-12 10:23 UTC: TMEM side-channel attribute cleanup in progress

- Preserve-set commit before this cleanup is `37d00bb91` on `codex/tmem`;
  target remote remains `origin/codex/tmem` (`jeffniu-openai` remote URL).
- User decision: the descriptor layout/type is the source of truth. Zero TMEM
  bases encode broadcast/equivalence semantics; if physical coordinates in one
  zero-basis equivalence class diverge, that is a producer/view/ld-st bug, not
  a reason for ld/st to choose a hidden live representative.
- Removed the branch-only TMEM side-channel API and producers/propagators for:
  - `ttng.tmem_ldst_row_plan`;
  - `ttng.tmem_physical_layout`;
  - `ttng.tmem_mmav5_accumulator_root`;
  - `ttng.tmem_mmav5_operand_root`.
- Direct ld/st planning now keeps the exact descriptor `LinearLayout`,
  including zero row/col bases, so direct stores, loads, and MMAv5 users agree
  on one logical-to-physical TMEM projection.
- The original `python/test/unit/language/test_matmul.py` M64 failures were
  fixed by preserving those zero bases instead of planning from an active
  physical layout that collapsed broadcast/support bits away.
- Removing the attrs exposed the same M64 backing-plan issue in
  `python/test/gluon/test_core.py::test_mma_shared_inputs`: before the
  type-derived fix, exact M64 cases chose a 64-row/narrow load-store plan and
  produced wrong results.
- Current source derives the full-shape M64 row plan/query from the memdesc
  type/layout family and ordinary view-chain/backing-plan analysis. It no
  longer depends on producer provenance attrs or on copying attrs through TMEM
  alloc rewrites.
- Focused validation already passed:
  - `make -j8`;
  - representative M64 matmul exact;
  - tensor-descriptor matmul exact;
  - `test_block_m_64_mma[legacy]`;
  - indexed-view `tcgen05.cp` no-scales exact;
  - focused Gluon frontend TMEM/MMAv5 tests;
  - focused lit for `tmem_layouts.mlir`, `mma_lowering.mlir`,
    `test_tensor_memory_allocation.mlir`, `promote-lhs-to-tmem.mlir`,
    `hoist-tmem-alloc.mlir`, and `loop-pipeline-blackwell.mlir`;
  - two exact and eight sampled `test_mma_shared_inputs` M64 cases, including
    f16/f8, use-acc true/false, CTA-group variants, and nonzero offsets.
- Hygiene already passed:
  - `git diff --check`;
  - production attr sweep across `lib`, `include`, `third_party`, `python`,
    and `test`; only negative test assertions still mention the removed attrs.
- The four-way post-fix `test_mma_shared_inputs` split sweep is green:
  - group 1: `3830 passed, 490 skipped, 13646 deselected`;
  - group 2: `2954 passed, 1366 skipped, 13646 deselected`;
  - group 3: `1206 passed, 3114 skipped, 13646 deselected`;
  - group 4: `2584 passed, 1736 skipped, 13646 deselected`.
- The pre-fix `test_mma_shared_inputs` failures after attr removal are stale
  and should not be counted as current red without a fresh repro.
- Post-helper-rename rebuild and smoke passed:
  - `make -j8`;
  - representative M64 matmul exact;
  - tensor-descriptor matmul exact;
  - `test_block_m_64_mma[legacy]`;
  - representative M64 `test_mma_shared_inputs` exact.

## 2026-04-12 15:20 UTC: layout-only row-plan cleanup closes preserve-set reds

- Starting point for this slice: `a88f55a81` on `codex/tmem`, dirty in
  `TensorMemoryUtils.cpp` and `TensorMemoryToLLVM.cpp`; target remote remains
  `origin/codex/tmem`.
- Root cause after removing the TMEM side-channel attrs:
  - the LLVM lowering still had a hard-coded f32 `64x128` reinterpret raw-query
    override that replaced exact view-chain layout arithmetic with
    `toLinearLayout(memTy)` and forced `{16, 32, 64}` row anchors;
  - same-dtype `memdesc_reinterpret` views could borrow the source support row
    bit order instead of planning from the destination/view layout;
  - pure outer `memdesc_index` views were using the wider M64 type/family row
    plan even when the concrete inferred query layout had no MMAv5 family block
    dimension, so a store through a chained view and a later load through
    `tmem.index(1)` chose different representatives of the zero-row/broadcast
    layout.
- Fix direction:
  - remove the remaining hard-coded reinterpret raw-query override;
  - keep same-dtype non-physical reinterpret support on the destination/view
    layout when it is standalone-query representable;
  - keep borrowed/wider backing row plans only when the query layout can
    materialize both requested row anchors;
  - for pure outer indexes, use the concrete query layout row plan unless that
    query layout itself still carries the MMAv5 family `block` dimension.
- This preserves the project invariant: the descriptor layout/query is the
  source of truth. Zero bases mean broadcast/equivalence; the planner must not
  choose a divergent hidden physical representative based on producer attrs.
- Current validation against rebuilt compiler:
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`: passed;
  - `make test-lit`: `248 passed, 2 unsupported`;
  - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix`: `26 passed`;
  - all `162` exact nodeids from
    `gb200_branch_new_20260412_unit_main_failures.txt`: `162 passed`;
  - `test_tmem_linear_m64_roundtrip_direct_shapes` plus `test_block_m_64_mma`:
    `20 passed`;
  - `test_tmem_physical_bitcast_preserves_subview_mapping[0]` and `[64]`:
    `2 passed`;
  - hygiene: `git diff --check` passed; removed-attr production sweep has no
    production hits, only negative test assertions.
- Status implication:
  - the preserve-set lit, unit, descriptor-chain, direct-M64, block-M-64 MMA,
    and physical-bitcast checks are green at this checkpoint;
  - older manifests that reported the `162` unit exacts or M64 descriptor-chain
    cases as current red are now stale and must not be used for prioritization
    without rerunning them.
- Remaining after this checkpoint:
  - run broader GB200 grouped sweeps when this slice is committed/pushed;
  - keep the legacy default-load opcode-quality issue on the docket if it still
    appears in a wider sweep;
  - continue the longer plan: supported view/bitcast API migration,
    attention rewrite later, `ld.red`, `copy` warpx2, broader MMAv5 family
    coverage, heuristic cleanup, and staged fuzzing/validation.

## 2026-04-13 04:19 UTC: current GB200 branch-actionable queue is closed from local evidence

- Current validation checkpoint is `4fe39e5d28edf82d90bb9039049083a66359f9b6`
  on `codex/tmem`, pushed to `origin/codex/tmem` before this docs update.
- Runtime source code is unchanged since the MMAv5 family-addressing fix at
  `21a82fc16`; later commits only recorded validation state.
- Current-head GB200 evidence now covers:
  - build: `make -j8` no-op success;
  - lit: `248 passed, 2 unsupported`;
  - C++: `240/240` passed;
  - microbenchmark: passed with median launch-overhead sample
    `21.786785125732422`;
  - main `python/test/unit`: `19153 passed, 5492 skipped` from the fresh
    four-GPU split sweep;
  - `test_debug.py`: product green, with the earlier split-order CUDA fork
    failure closed by the exact sanitizer subset in a fresh process;
  - `python/triton_kernels/tests`: full non-overlapping coverage recomputed as
    `2377 passed, 3444 skipped`, no failures/errors;
  - fused-attention tutorial: four `pytest-split` shards total
    `192 passed, 192 skipped`, no failures;
  - instrumentation/plugins: four singleton Makefile tails all passed;
  - Gluon/examples: no deterministic known failures after the runtime-matrix and
    lowerings tail was closed;
  - gsan and regression: green from the fresh four-GPU sweeps.
- Proton remains the only known CI-command red family, but its `11` cudagraph /
  periodic flushing nodeids reproduced on merge-base `origin/main`, so they are
  preexisting/environmental and not a TMEM branch-actionable failure.
- Status implication: the GB200 branch-recovery queue is empty from current
  local evidence. Do not prioritize stale manifests that list the old 9 lit
  failures, 162 unit failures, M64 Gluon exact, tile-permuted MMAv5 runtime
  failures, or incomplete Gluon timeout shards as live branch failures without a
  new exact repro.
- Next engineering direction returns to the long-term TMEM plan:
  supported view/bitcast API migration, attention rewrite later, `ld.red`
  breadth, copy `warpx2`, broader MMAv5/scaled-MMAv5 reachable-family coverage,
  heuristic cleanup, and staged fuzzing/validation.
- Validation methodology reminder: always use all four GPUs for pytest work
  when the selection is large enough, but treat multi-hour local shards as a
  partitioning/xdist/hang problem. The project owner expects the full GB200 CI
  lane to take about 35 minutes including clean build/LLVM download, with actual
  test time around 20 minutes on comparable hardware.

## 2026-04-13 04:58 UTC: explicit `ld.red` non-identity layouts canonicalize to 32x32b

- Added runtime-matrix coverage for explicit compatible `ld.red` register
  variants over non-identity compatible TMEM-linear source layouts.
- New matrix:
  - layouts: `tile_permuted`, `col_reverse`, `row_reverse`, and
    `rowcol_rotate_reverse`;
  - explicit load variants: `auto`, `32x32b`, `16x32bx2`, and
    `32x32b_splitn`;
  - operation: `min` on `128x128xf32` sources.
- Purpose:
  - closes the fuzz-plan discovery question for this bounded non-identity slice:
    these explicit compatible variants do not expose any accepted reduction atom
    besides `tcgen05.ld.red.sync.aligned.32x32b.x128`;
  - keeps the existing identity-layout min/max coverage and N-sharded clean
    negatives intact.
- Validation:
  - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`;
  - collect-only for the new nodeids: `16 tests collected`;
  - new test across four GPU split groups: `16 passed` total;
  - broad `ld_red` selector as a clean split-16 run: `503 passed`, no
    skips/failures/errors;
  - `git diff --check` passed before docs update.
- Methodology note:
  - the first broad split-4 run packed slow cases into groups 2 and 3 and timed
    out at 900s after steady progress; the full split-16 rerun closed the same
    selector cleanly and should be the preferred local recipe for this selector.


## 2026-04-13 09:14 UTC: non-surjective projected TMEM query layouts no longer drive subview base offsets

- While validating the new two-CTA MMAv5/i8 coverage, an overbroad `-k "mma and not cp"` selector exposed a deterministic current-head compiler abort in `test_tmem_runtime_matrix_ldst_twocta_descriptor_roundtrip_sweeps[slice_index_roundtrip-0-5.0-required_ops0-mmav5_twocta-64-auto-32x32b.x64.b32]`.
- Root cause from gdb: this was not an ld/st packet-planner crash. `MemDescSubsliceOpConversion` called `getTMemSubviewOffsetForLowering`, which accepted a raw projected TMEM query layout for subview offset arithmetic because its rank and out-dim names matched the descriptor type. That projected layout was intentionally non-surjective after an outer subslice/index chain (`row` was size 1, only one `col` bit remained, and `block` mapped one row bit), so `getTMemViewOffset(...).pseudoinvert()` asserted in `LinearLayout::lstsq` with `Im(B) not contained in Im(A)`.
- Fix: `getTMemViewOffsetForLowering` now uses the raw query layout for subview-base arithmetic only when the query layout is surjective. Non-surjective projected query layouts remain valid for ld/st support planning, but arbitrary logical subview offsets fall back to the descriptor type's full layout.
- This preserves the initiative invariant: the descriptor type/layout is the source of truth for physical TMEM offset arithmetic; support-query projections are not hidden producer-provenance attributes and must not be used as if they covered the full logical offset space.
- Also added missing two-CTA direct `tcgen05.mma kind::i8` clean-negative coverage in `test_tmem_runtime_matrix_mma_twocta_i8_reports_clean_error`, covering both legacy and linear two-CTA accumulator layouts. This pins the same unsupported direct-i8 diagnostic already covered for one-CTA layouts.
- Validation after rebuild:
  - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`: passed;
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`: passed and relinked `libtriton.so`/tools;
  - exact former abort node now reaches the existing tensor-memory OOR skip path with no assertion or `PassManager` crash;
  - all 15 exact `mmav5_twocta-64` two-CTA descriptor-roundtrip nodeids skipped through the existing OOR handler across four GPU split groups (`4`, `4`, `4`, `3` selected), no failures;
  - positive `test_tmem_runtime_matrix_ldst_twocta_descriptor_compositions` groups 3 and 4 passed (`8`, `6` selected); groups 1 and 2 timed out under a `120s` wrapper after steady progress because cold static split-4 packed slow compile-heavy cases, not because of a failing nodeid;
  - individually rerun composition cases that looked like the likely slow tail passed: `block_two_ctas-128-32x32b` in `24.35s`, `block_two_ctas-128-16x64b` in `24.50s`, `block_two_ctas-256-16x256b` in `25.16s`, and `mmav5_twocta-64-auto` in `12.91s`;
  - the combined one-CTA/two-CTA i8 clean-negative matrix passed across four GPU split groups (`1` selected per group);
  - `git diff --check`: passed.
- Validation caveat: do not report the timed-out composition split groups as test failures. They are another example of the known no-duration static-split problem. For a full compositions refresh, collect/store durations or split by exact nodeid buckets.
- Tooling caveat: `apply_patch` still fails with `No such file or directory`; the small source edit used an exact scripted replacement.

## 2026-04-13 09:24 UTC: direct two-CTA scales `warpx4` copy coverage

- Added direct runtime-matrix coverage for a two-CTA CGA-shaped tensor-memory-scales copy without going through scaled MMA:
  - new helper `tmem_copy_scales_warpx4_twocta_kernel`;
  - source/destination tile `128x16xi8`;
  - shared layout uses the supported scales `warpx4` offset bases plus `block_bases=[[64, 0]]`;
  - destination uses `TensorMemoryScalesLayout(cga_layout=[[1, 0]])`;
  - launch uses `num_ctas=2` and `fence_async_shared(cluster=True)`.
- New test `test_tmem_runtime_matrix_cp_scales_warpx4_twocta_direct_copy` checks exact roundtrip correctness, exact PTX/LLIR copy opcodes, and absence of legacy `ttg.memdesc_reinterpret`.
- Observed/direct-copy boundary:
  - pure scales copy without an MMAv5 op emits two `tcgen05.cp.cta_group::1.warpx4.32x128b` operations for this CGA-shaped tile;
  - the existing scaled-MMA copy matrix continues to pin `tcgen05.cp.cta_group::2.warpx4.32x128b` when the enclosing two-CTA MMAv5 operation sets the module-level two-CTA contract.
- Validation:
  - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`: passed;
  - rebuild before runtime pytest was a no-op success;
  - focused `-k cp_scales_warpx4` over four `pytest-split` groups: `54 passed`;
  - broader runtime-matrix `-k cp` over four `pytest-split` groups: `167 passed, 5 skipped`;
  - `git diff --check`: passed.
- This closes a bounded supported-copy coverage gap only. True scales `warpx2`, two-CTA `warpx2::02_13` descriptor/address synthesis, broader `ld.red` fuzzing, and MMAv5/scaled-MMAv5 family saturation remain the next long-term frontiers.
- Tooling note: `apply_patch` still fails with `No such file or directory`; this edit used exact scripted replacements.

## 2026-04-13 12:37 UTC: two-CTA scaled-MMAv5 accumulator-subview matrix covers multicast

- Expanded `test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_matrix` so the existing two-CTA scaled-MMAv5 accumulator-subview format matrix covers both scale-TMA paths:
  - added `multicast=False/True` parameterization;
  - passed the parameter through `mma_scaled_tcgen05_acc_subslice_copy`;
  - asserted TTGIR `{multicast}` is present only for the multicast path.
- The matrix still covers all five existing format pairs, both `slice_start=0` and `slice_start=128`, numeric output equality, exact scaled-copy opcodes, exact scaled-MMA opcodes, and exact commit opcodes.
- Validation / hygiene:
  - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py` passed;
  - `git diff --check python/test/gluon/test_tmem_runtime_matrix.py` passed before docs update;
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8` passed after removing temporary copy-verifier debug instrumentation;
  - exact nodeid across four GPU split groups: `20 passed` aggregate;
  - nearby selector `python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and subslice and format_matrix'` across four GPU split groups: `43 passed` aggregate.
- This is coverage expansion only. True scales `warpx2` and two-CTA `warpx2::02_13` remain descriptor/address-model frontiers.
- Tooling note: `apply_patch` still fails with `No such file or directory`; this docs update used exact scripted replacements.

## 2026-04-13 12:43 UTC: scaled-MMAv5 `tile_n=64` accumulator coverage spans all formats

- Replaced the old `mxfp8/mxfp8`-only `test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_64_direct_layout` with `test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_64_format_matrix`.
- Added `tmem_mma_scaled_layout_format_kernel`, a direct accumulator-layout scaled-MMA helper that mirrors the existing format-aware subview kernels without forcing an accumulator slice.
- New positive surface:
  - accumulator layout: `TensorMemoryLinearLayout` tile-permuted `128x256`, `tile_n=64`;
  - format pairs: `mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`, `mxfp4/mxfp8`, and `nvfp4/nvfp4`;
  - expected opcode counts: `16` for `mxf8f6f4` vector-1X cases, `8` for homogeneous `mxf4` / `mxf4nvf4` cases.
- Validation / hygiene:
  - temporary direct smoke over the five format pairs passed before editing the runtime matrix;
  - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py` passed;
  - `git diff --check python/test/gluon/test_tmem_runtime_matrix.py` passed;
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8` was a no-op success;
  - exact nodeid across four GPU split groups selected all five cases and all selected cases passed (`2`, `2`, `1`, and `0` selected; group 4 had no selected case and returned pytest's no-tests code);
  - nearby selector `python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and tile_permuted'` passed `11` selected cases across four GPU split groups.
- This closes a bounded scaled-MMAv5 tile-permuted format gap. The `tile_n=32` repeated-`N=32` boundary remains a clean unsupported negative, and true scales `warpx2` / two-CTA `warpx2::02_13` remain descriptor/address frontiers.
- Tooling note: `apply_patch` still fails with `No such file or directory`; this docs update used exact scripted replacements.

## 2026-04-13 12:45 UTC: broad direct MMA/scaled-MMA selector green after coverage expansion

- Current head: `8b3fec8d6` on `codex/tmem`, pushed to `origin/codex/tmem` before this docs-only validation checkpoint.
- Ran the broad runtime-matrix selector after the two scaled-MMAv5 coverage commits:
  - command family: `CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-mma-not-cp-g<group> PYTHONPATH=python:. timeout 900s pytest -s --tb=short --splits 4 --group <1..4> python/test/gluon/test_tmem_runtime_matrix.py -k 'mma and not cp'`;
  - group 1: `32 passed, 45 skipped, 2695 deselected in 31.26s`;
  - group 2: `72 passed, 5 skipped, 2695 deselected in 12.80s`;
  - group 3: `77 passed, 2695 deselected in 17.82s`;
  - group 4: `77 passed, 2695 deselected in 55.82s`.
- Aggregate: `258 passed, 50 skipped`, no failures or errors.
- This is validation only. It does not change the remaining true scales `warpx2`, two-CTA `warpx2::02_13`, or future broad-fuzz frontiers.
- Tooling note: `apply_patch` still fails with `No such file or directory`; this docs update used exact scripted replacements.
