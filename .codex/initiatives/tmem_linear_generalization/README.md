# TMEM Linear Generalization Initiative

This directory is the durable coordination space for the TMEM linear-layout
generalization project.

Keep this README up to date when the role of any document changes, when a new
current-state handoff supersedes an older one, or when the source-of-truth
entry points change.

## How To Use This Folder

When resuming the initiative:
- start here;
- then read `memory.md` as the initiative's durable memory/source of truth;
- use `gb200_nvidia_ci_inventory.md` for the current GB200/NVIDIA CI baseline
  and confirmed red list;
- use `gb200_failure_manifest.md` plus the adjacent generated `.txt` files
  when you need the exact nodeid inventories for the current branch or the
  merge-base-existing shard-3 subsets;
- use `gb200_failure_classification_20260412.md` for the current split between
  actual compiler/runtime bugs, stale tests, tests requiring API/contract
  updates, and merge-base-preexisting noise;
- use `gb200_preserve_set_20260412.md` for the current pre-cleanup preserve-set
  before removing branch-only TMEM physical-layout / row-plan / MMAv5-root
  attributes and the hacks that depend on them;
- use `gb200_branch_recovery_plan.md` when you need the prioritized
  branch-caused recovery backlog and the exact representative repros to drive
  fixes;
- then read the tail of the active handoff for the latest live buckets and next
  concrete steps;
- use `log.md` for chronological provenance and validation history; and
- use `fuzz_plan.md` once the current planner/cleanup blockers are green and
  broad saturation work is the next task.
- use `ldst_validation_recipe_20260413.md` for the current duration-cache
  and bucketed recipe for broad `ld/st` runtime-matrix validation.
- use `tmem_runtime_matrix_validation_recipe_20260413.md` and
  `run_tmem_runtime_matrix_sweep.py` for the full current runtime-matrix
  sweep; this is the coverage-preserving replacement for raw static split-4
  full-file runs that time out while still making progress.
- use `backend_completion_plan.md` for the post-coverage TMEM backend
  completion roadmap: exact physical TMEM query model, atomized copy planning,
  scales unification, load/store/reduction cleanup, MMA/scaled-MMA unification,
  broad fuzzing, and compatibility deletion.
- use `completion_execution_tracker.md` for the active progress board against
  the completion plan, including the current clean-negative inventory,
  remaining buckets, completed checkpoints, and next concrete implementation
  slice.

## Current Backend Checkpoint

- 2026-04-17 21:26 UTC: reran the corrected full runtime-matrix sweep after
  the leading-slice replay policy cleanup and MMAv5 family-address helper
  cleanup. The matrix remains green with complete bucket coverage:
  `cp` `312 passed, 4 skipped`; `mma` `601 passed`; `splitn` `35 passed`;
  `ld_red` `247 passed`; `ldst` `295 passed, 98 skipped`; aggregate `1490
  passed, 102 skipped` across all `1592` cases. Logs are under
  `experiments/results/tmem_runtime_matrix_sweep_20260417_211829/`. Next:
  commit/push this validation checkpoint and return to the support-bearing
  Phase C/Phase E frontier plus remaining backend-policy cleanup.
- 2026-04-17 21:08 UTC: finished the current Phase F
  `OptimizeTMemLayouts` shim audit by exposing the backend half-slice replay
  predicate as `getTMemLdStReplayableHalfSliceDim`. The transform now reuses
  that backend predicate and only computes the rewrite-local selected-half
  value. Validation: `make -j8`; focused split-4 half-slice replay selector
  passed `5/5`, `5/5`, `5/5`, and `3/3`; `git diff --check`. No remaining
  transform-local replay predicate currently duplicates a backend helper with
  enough information to delete safely; next work returns to the Phase C/Phase E
  support frontier.
- 2026-04-17 21:05 UTC: continued Phase F by moving the direct `ld/st`
  replay/support register-layout search out of `OptimizeTMemLayouts` and into
  backend `TensorMemoryUtils` as `getTMemLdStDirectSupportTensorType`. The
  transform now only asks the backend helper for the support tensor type used
  by leading-slice and replay rewrites. Validation: `make -j8`; focused
  split-4 runtime selector for multidim replay, direct/half-row replay, and
  direct higher-rank get-reg-layout paths passed `6/6`, `6/6`, `6/6`, and
  `4/4`; `git diff --check`.
- 2026-04-17 21:01 UTC: started the Phase F compatibility-shim audit after
  the full runtime-matrix baseline. `TMEMLoadOp`/`TMEMStoreOp` verification no
  longer carries its own row-zero lifted reinterpret predicate; it delegates to
  the backend-owned `disallowTMemLdStQueryTypeRescue` helper in
  `TensorMemoryUtils`. This is intended as behavior-preserving cleanup that
  leaves the support/boundary surface unchanged while consolidating layout
  policy in the backend helper layer. Validation: `make -j8`; focused
  runtime-matrix clean-error rows for blocked layouts (`2 passed`) and block
  descriptor views (`2 passed`); `git diff --check`. `lit` is not installed in
  this shell as either `lit` or `python3 -m lit`, so no lit command was
  included in this checkpoint.
- 2026-04-17 20:57 UTC: the corrected full runtime-matrix runner passed with
  complete `1592/1592` bucket coverage. Aggregate evidence: `cp`
  `312 passed, 4 skipped`; `mma` `601 passed`; `splitn` `35 passed`;
  `ld_red` `247 passed`; `ldst` `295 passed, 98 skipped`; total `1490
  passed, 102 skipped`. Logs are under
  `experiments/results/tmem_runtime_matrix_sweep_20260417_205257/`.
- 2026-04-17 20:52 UTC: completed the copy-planner boundary audit for the
  remaining representative clean negatives. The focused copy selector covering
  scales descriptor-view masks, ordinary-view `4x256b`, two-CTA
  `warpx2::02_13`, row/column permutations, sub-instruction tile
  permutations, and noncanonical CTA ownership passed `44/44` after `make -j8`.
  No rows were promoted: the current diagnostics still point to true
  packed-lane, high source-column, row/column mask, refresh-remap/readback,
  scales-mask, or CTA-ownership requirements rather than stale generic
  linear-layout arithmetic.
- 2026-04-17 20:48 UTC: integer `ld.red` requests with
  `propagate_nan=ALL` are now supported through the existing software
  reduction path. The frontend treats NaN propagation as `NONE` for
  non-floating element types before choosing the software combiner; floating
  non-f32 semantics are unchanged. This promotes direct and descriptor-chain
  i32 NaN rows from clean unsupported to positives. Validation: `make -j8`,
  exact promoted rows (`4 passed`), full `ld_red_non_f32` selector
  (`44 passed`), runner `ld_red` bucket (`247 passed`), clean-negative
  rebaseline (`111/1592`, combined `161/1592`), Python byte-compile, and
  `git diff --check`.
- 2026-04-17 20:41 UTC: the runtime-matrix runner is corrected and reaudited
  after the M64 physical-subview work. Shards now force checkout-local
  `PYTHONPATH`, the `splitn` bucket includes the two 16-bit M64 parity rows
  that were previously outside the bucket union, and a collection audit proves
  the buckets cover all `1592/1592` collected nodeids. Corrected bucket
  evidence: `cp` `312 passed, 4 skipped`; `mma` `601 passed`; `splitn`
  `35 passed`; `ld_red` `247 passed`; `ldst` `295 passed, 98 skipped`;
  aggregate `1490 passed, 102 skipped`. The only `ld.red` N=256 descriptor
  chain override left is the verified `tile_permuted` split-offset order.
  Validation also includes Python byte-compile for the runner and runtime
  matrix plus `git diff --check`.
- 2026-04-17 20:24 UTC: pure TMEM column subview base lowering now prefers the
  source query's exact surjective `LinearLayout` physical offset before using
  origin-delta fallback. This fixes the legacy M64 split-N MMAv5 path where a
  high-N `memdesc_subslice` had been lowered as a plain column offset even
  though direct root `ld/st` stored the same logical band at the folded
  physical row offset. Validation: `make -j8`, `test_block_m_64_mma`
  (`2 passed`), focused legacy M64 runtime rows (`4 passed`), ld.red
  descriptor-chain tile-permuted N=256 (`1 passed`), Python byte-compile, and
  `git diff --check`.
- 2026-04-17 19:27 UTC: the remaining no-scales `warpx2` subword bucket was
  probed after the source-rematerialization work. Representative f16
  single-CTA `01_23`, single-CTA `02_13`, and two-CTA `01_23` rows all fail
  through the structured destination-column footprint requirement: N=4 exposes
  two column basis bits, while the public 128-bit f16 `warpx2` instruction
  needs three logical column bits and writes four physical dword columns with
  two packed lanes per word. Treat this as a packed-lane storage/ISA boundary
  until a lane-aware source/destination model exists.
- 2026-04-17 19:25 UTC: no-scales `tcgen05.copy.warpx2::01_23`
  now supports source rematerialization for two-CTA dense/noncanonical 256x4
  shared-linear sources. The rematerialized canonical source uses shared block
  basis `[[128, 0]]` and a cluster async-shared fence. Dense two-CTA
  `warpx2::02_13` now rematerializes the source far enough to report the real
  high source-column preservation boundary, and remains clean unsupported.
  Validation: `make -j8`, focused dense two-CTA selector (`4 passed`), full
  `cp_no_scales_warpx2` (`79 passed`), split-4 `cp_no_scales_warpx2`
  (`20/20/20/19` passed), Python byte-compile, and collect-only rebaseline
  (`115/1592`, combined `165/1592`).
- 2026-04-17 19:21 UTC: no-scales `tcgen05.copy.warpx2` now supports
  source rematerialization for single-CTA dense/noncanonical 128x4
  shared-linear sources. The Gluon builtin recognizes TensorMemoryLinear
  destinations that map to public `warpx2::01_23` or `warpx2::02_13`,
  reloads the noncanonical source through logical coordinates, stores it into
  the canonical 128x4 `warpx2` shared source, fences the new shared writes,
  and emits the existing backend copy. The former single-CTA dense-source
  clean negatives are now positive runtime/opcode coverage. Two-CTA dense
  sources and `02_13` high source-column preservation remain the next copy
  frontier/boundary. Validation: `make -j8`, exact promoted selector
  (`4 passed`), neighbor/scales selector (`41 passed`), full
  `cp_no_scales_warpx2` (`79 passed`), split-4 `cp_no_scales_warpx2`
  (`20/20/20/19` passed), Python byte-compile, and collect-only rebaseline
  (`117/1592`, combined `167/1592`).
- 2026-04-17 19:14 UTC: TensorMemoryScales `tcgen05.copy` now supports
  source rematerialization for noncanonical 64x16 shared-linear sources. The
  Gluon builtin copies those source views into the canonical warpx4 shared
  layout before emitting `ttng.tmem_copy`, which promotes the historical
  warpx2-shaped source-layout probes and shared-subslice source probes to
  positive coverage. The scaled-MMA scale descriptor path is preserved, and
  the remaining scales descriptor-view negative is still a destination mask/
  split boundary. Validation: `make -j8`, focused 10-test selector, broader
  `-k 'cp_scales'` (`34 passed`), split-4 `cp_scales` (`9/9/9/7` passed),
  Python byte-compile, and `git diff --check`.
- 2026-04-17 19:07 UTC: classification checkpoint after the scales
  descriptor-view promotion. Direct probes classify sub-instruction
  `tcgen05.copy` tile permutations and row/column permutations as public
  full-footprint atom boundaries needing row/column masks, smaller atoms,
  row/column partitioning, or a different source format. A temporary direct
  MMAv5 i8 guard lift emitted `.kind::i8` PTX, but `ptxas-blackwell` rejects
  that feature on `sm_103a`, so the existing frontend guard remains and the i8
  bucket is an external PTXAS/ISA boundary. `AGENTS.md` now explicitly makes
  the initiative docs the active checklist for no-stop execution; the tracker
  points the next implementation probe at no-scales `tcgen05.copy.4x256b`
  ordinary-view/remap support.
- 2026-04-17 18:59 UTC: the two-CTA scales descriptor-view
  `16x32bx2` direct `ld/st` case is now supported. The backend lifts the
  physical packet layout through the exact descriptor-view query, keeps the
  half-tile split on lane=16, and gives the resulting `TensorMemoryLinear`
  view the split-N preference needed to preserve the requested atom instead
  of selecting `32x32b`. The runtime matrix now covers the former clean
  negative as a positive CGA descriptor-view row emitting
  `tcgen05.ld/st.16x32bx2.x32.b32`. Validation: `make -j8`, exact promoted
  row (`1 passed`), focused CGA selector (`9 passed, 1583 deselected`),
  Python byte-compile, and `git diff --check`.
- 2026-04-17 18:41 UTC: explicit n-sharded scales `ld/st` requests that are
  smaller than the public packet footprint now report a structured
  tensor-memory-scales atom-footprint requirement. The diagnostic records the
  requested atom, required scale-element footprint, and exposed view elements
  instead of stopping at a generic descriptor-view unsupported message.
  Validation: `make -j8`, focused selector (`5 passed`) plus split-4 selected
  groups (`2/2/1`, group 4 empty), built `triton-opt` verify-diagnostics,
  Python byte-compile, and `git diff --check`.
- 2026-04-17 18:36 UTC: subword `warpx2` copy destinations now report a
  structured destination-column footprint requirement. The diagnostic records
  exposed versus required column basis bits, logical instruction columns,
  physical 32-bit dword columns, and packed lanes per word, making the
  under-wide `N=4` subword rows an explicit packed-lane storage/schedule
  boundary. Validation: `make -j8`, split-4 focused subword selector
  (`4/4/4/2` passed), built `triton-opt` verify-diagnostics, Python
  byte-compile, and `git diff --check`.
- 2026-04-17 18:32 UTC: dense/noncanonical `warpx2` shared-source layouts now
  report the first shared offset-basis mismatch through
  `TMemCopyWarpx2SharedSourceRequirement`. Support is unchanged: these rows
  still require source rematerialization, a different source format, or a
  proved message schedule because descriptor representability alone does not
  prove the fixed `warpx2` shared source footprint. Validation: `make -j8`,
  split-4 focused dense shared-source selector (`2/2/2/2` passed), built
  `triton-opt` verify-diagnostics, Python byte-compile, and
  `git diff --check`.
- 2026-04-17 18:10 UTC: no-scales two-CTA `warpx2::02_13` source-column
  preservation now has a typed requirement formatter derived from the
  source-row split requirement. The bucket remains unsupported because public
  `cta_group::2` schedules either duplicate the low source-column pair into the
  high columns or read zeros when trying to complete the single-CTA schedule.
  Validation: `make -j8`, split-4 focused `warpx2::02_13`/neighbor selector
  (`8/8/8/6` passed), built `triton-opt` verify-diagnostics, Python
  byte-compile, and `git diff --check`.
- 2026-04-17 18:04 UTC: scales `tcgen05.copy` descriptor-view failures now
  have a structured exact-view schedule note. The note records that the first
  differing physical-query field is the physical layout for the active
  `128x32` view before reporting the required destination-row/source-message
  schedule. Probes confirm source column bit 2 selects descriptor row `+32`
  for 4-column runs every 8 columns while the public `warpx4.32x128b` atom
  writes the full 16-column destination footprint. Validation: `make -j8`,
  split-4 focused scales-copy selector (`3/3/3/2` passed), built `triton-opt`
  verify-diagnostics, Python byte-compile, and `git diff --check`.
- 2026-04-17 17:59 UTC: direct `ld/st` diagnostics for `4x256b`
  refresh-shaped layouts now share the structured
  `TMemCopy4x256RefreshImageRequirement` facts already used by copy support.
  The refresh-layout and raw-physical-bitcast clean negatives now report the
  logical 4x8 image, low-column row anchors, high-column dword offset, and
  source-column split from data. Validation: `make -j8`, focused
  `cp_no_scales_4x256b or ldst_4x256b_refresh` runtime selector (`5 passed`),
  built `triton-opt` verify-diagnostics for `test/TritonNvidiaGPU/invalid.mlir`,
  Python byte-compile, and `git diff --check`.
- 2026-04-17 17:55 UTC: strengthened the initiative execution contract. The
  active `completion_execution_tracker.md` now carries a phase-by-phase
  remaining-work board from rebaseline/classification through final validation,
  and `AGENTS.md` now requires writing/refreshing the plan in initiative docs,
  tracking progress steadily against it, and continuing execution until the
  tracked plan is complete unless interrupted or concretely blocked.
- 2026-04-17 17:53 UTC: scaled-MMAv5 narrow-N accumulator rejection now records
  `minimumAddressableBScaleFragmentN` in
  `MMAv5ScaledNarrowNScaleFragmentRequirement`, making the 64-column B-scale
  fragment boundary structured rather than string-only. Validation:
  `make -j8`, split-4 scaled narrow clean-negative selector (`5/5/5/5`
  passed), and `git diff --check`.
- 2026-04-17 17:50 UTC: direct copy mixed row/column basis failures now use a
  typed `TMemCopyMixedBasisRequirement`, and the `mixed` no-scales runtime row
  reports the offending basis bit plus physical TMEM delta. Debug probes also
  classified the adjacent sub-instruction column permutations and scales
  descriptor-row split copies as full-footprint/mask gaps. Validation:
  `make -j8`, split-4 copy exotic/rowcol/warpx2-row clean-negative selector
  (`5/5/5/4` passed), and `git diff --check`.
- 2026-04-17 17:45 UTC: the copy planner now represents row-permuted
  destination rejection with a shared typed
  `TMemCopyDestinationRowOrderRequirement`. Dense and multicast paths both
  preserve the old diagnostics while also recording the derived
  destination-row mask/schedule gap that future row-projection support must
  solve. Validation: `make -j8`, split-4 focused row-order clean-negative
  selector (`4/4/4/4` passed), and `git diff --check`.
- 2026-04-17 17:36 UTC: created the active completion execution tracker and
  rebaselined the current clean-negative inventory after `make -j8`.
  `pytest -q --collect-only python/test/gluon/test_tmem_runtime_matrix.py -k
  'reports_clean_unsupported'` collected `124/1592` tests in `3.02s`. The
  tracker records the current buckets: scales `ld/st` atom-footprint
  boundaries, non-f32 NaN reduction semantics, scales copy split/mask
  schedules, `4x256b` refresh readback boundaries, no-scales two-CTA
  `warpx2::02_13`, dense/noncanonical `warpx2` source rematerialization,
  copy row/column permutation schedules, plain MMAv5 instruction-tile
  permutations, mixed fp4A TMEM-LHS storage, and scaled narrow-N accumulator
  fragments.
- 2026-04-17 14:34 UTC: fixed a core `LinearLayout` division contract bug
  exposed by the no-scales two-CTA `warpx2::02_13` copy clean-negative rows.
  `divideLeft` and `divideRight` now use `LinearLayout::tryCreate` for the
  quotient candidate and return `std::nullopt` when the candidate is not a
  well-formed layout or does not reproduce the dividend, instead of letting an
  optional helper abort the process through the checked constructor/assertion.
  Added a focused C++ unit test for the malformed-candidate case. Validation:
  `make -j8`, `unittest/Tools/LinearLayout` (`71` tests), the exact
  `cp_no_scales_warpx2_02_13_twocta` clean-negative repro (`1 passed`), the
  split-4 selector for that bucket (`4/4/4/2` passed), and
  `git diff --check`.
- 2026-04-17 14:27 UTC: added the matching 256-row split store-join lit
  coverage. The existing planner-gated store rewrite now lowers the joined
  `256x64 + 256x64 -> 256x128` pattern into two `ttng.tmem_subslice` plus
  two `ttng.tmem_store` operations when the pre-rewrite full store uses a
  legal full-tile source layout. Validation: `make -j8`, direct `triton-opt`
  run for `test/TritonNvidiaGPU/tmem_layouts.mlir`, and `git diff --check`.
  A synthetic direct LLVM-lowering probe of the optimized store body hit the
  existing shared-memory `allocation.offset` precondition for a generated
  `convert_layout`, so this checkpoint only claims optimizer/lit coverage.
- 2026-04-17 14:26 UTC: promoted the existing 256-row split-load replay case.
  Row-preserving column subviews now complete the missing high row basis for
  exact analysis and validate the 8-warp `I32x32b` direct register-layout
  candidate against the folded physical query used by actual
  `ttng.tmem_subslice` lowering. The optimizer now rewrites
  `@subtile_tmem_load_256` into two `ttng.tmem_subslice` plus two
  `ttng.tmem_load` operations instead of leaving the full load/split in place.
  Validation: `make -j8`, direct `triton-opt` run for
  `test/TritonNvidiaGPU/tmem_layouts.mlir`, minimal LLVM lowering of the two
  256x64 subslice loads with two `tcgen05.ld.sync.aligned.32x32b.x64.b32`
  packets, and `git diff --check`. Lit/FileCheck are unavailable in this
  shell.
- 2026-04-17 13:56 UTC: generalized the TMEM split-load/store-join replay
  optimizer guard. The rewrite no longer hard-codes `M == 128`; it infers the
  future `ttng.tmem_subslice` type, asks the TMEM compatible-layout planner
  whether that slice has a legal register layout, and declines without
  mutating IR when it does not. Existing 256-row lit coverage remains
  unchanged because no compatible slice layout is available yet, but the pass
  is now gated by planner capability rather than a shape-family constant.
  Validation: `make -j8`, direct `triton-opt` run for
  `test/TritonNvidiaGPU/tmem_layouts.mlir`, and `git diff --check`.
- 2026-04-17 13:47 UTC: factored TMEM load/store compatible-layout
  validation in `TritonNvidiaGPU` dialect code. Atom candidates, unique
  special candidates, and the 8-warp split-long-M reduction layout now all
  flow through one append-and-validate helper, removing the old split-long-M
  side path marked as an `isDistributedLayoutTMemCompatible` generalization
  hack. Support is intended to be unchanged. Validation: `make -j8`, collect
  for `ld_red_m64 or ldst_direct_higher_rank_load_red` (`40` cases), split-4
  runtime selector (`10/10/10/10` passed), and `git diff --check`.
- 2026-04-17 13:43 UTC: aligned type-only
  `tensor_memory_descriptor_type.get_reg_layout()` with the handle-aware
  higher-rank replay path. Type-only TMEM-linear higher-rank layouts now
  flatten leading dimensions before calling the rank-2 backend planner and
  unflatten the returned `DistributedLinearLayout` bases. Validation:
  `make -j8`, exact frontend type-only test (`1 passed`), Python compile, and
  `git diff --check`.
- 2026-04-17 13:40 UTC: direct higher-rank TMEM reduction load now replays
  through the flattened descriptor view as well. The implementation reshapes
  the full load result back to the descriptor shape and the reduction result
  back to the leading dimensions, while still letting the rank-2 backend pick
  the hardware/software reduction layout. Validation: `make -j8`, exact
  higher-rank `load_min` replay row (`1 passed`), focused
  `ldst_direct_higher_rank` selector (`4 passed`), Python compile, and
  `git diff --check`.
- 2026-04-17 13:37 UTC: promoted direct higher-rank TMEM
  `get_reg_layout()` for layouts whose flattened rank-2 register layout can be
  exactly unflattened over the original logical dimensions. The frontend now
  maps `DistributedLinearLayout` bases between `[prod(leading), N]` and the
  higher-rank descriptor shape, and explicit higher-rank `load(layout=...)`
  maps such layouts back to the flattened direct-load view. Validation:
  `make -j8`, exact promoted `get_reg_layout` runtime rows (`2 passed`),
  focused `ldst_direct_higher_rank` selector (`3 passed`), Python compile,
  and `git diff --check`.
- 2026-04-17 13:32 UTC: promoted direct higher-rank TMEM `load()` and
  `store()` by replaying them through an explicit 2D descriptor view. The
  Gluon descriptor methods now flatten leading logical dimensions for direct
  load/store only, use the existing rank-2 backend register-layout and
  `ttng.tmem_load`/`ttng.tmem_store` path, then reshape the register value
  back to the original descriptor shape. Direct `get_reg_layout()` and
  reduction load remain rank-2-only because they expose a concrete register
  layout contract rather than a value replay. Validation: `make -j8`,
  exact direct higher-rank clean-negative plus replay positive (`3 passed`),
  focused `ldst_direct_higher_rank` selector (`3 passed`), Python compile,
  and `git diff --check`.
- 2026-04-17 13:24 UTC: tightened a descriptor-view clean negative for
  non-affine tensor-memory subslices. `inferTMemSubsliceEncoding` and
  `inferTMemSubsliceQueryLayout` now report when a subview window crosses a
  physical basis boundary and would require a carry-dependent descriptor view,
  instead of surfacing only the generic memdesc_subslice failure. The
  block-backed two-CTA bitcast/subslice runtime-matrix row now asserts this
  boundary explicitly. Validation: `make -j8`, exact block descriptor
  clean-negative (`2 passed`), neighboring `test_core.py` unsupported bitcast
  row (`1 passed`), Python compile, and `git diff --check`.
- 2026-04-17 13:13 UTC: dense direct `tcgen05.copy` destination support now
  carries the actual two-CTA mode into the layout proof and checks canonical
  `[[128, 0]]` CTA block ownership through the same helper used by multicast
  destinations. Added a focused clean-negative for a noncanonical dense
  two-CTA block basis before descriptor planning. Validation: `make -j8`,
  exact new negative (`1 passed`), split-4 neighboring
  `cp_no_scales_twocta or cp_no_scales_warpx2` selector
  (`31/31/31/28` passed), Python compile, and `git diff --check`.
- 2026-04-17 12:59 UTC: pruned a stale plain-MMAv5 fpsan negative. The
  block-backed `128x128` accumulator row now compiles at current head and was
  removed from `MMA_UNSUPPORTED_LAYOUT_CASES`; the remaining plain-MMAv5
  unsupported fpsan selector still passes (`4 passed`). The broad positive
  fpsan matrix was not expanded because it also exercises an invalid
  block-backed descriptor-view parent shape.
- 2026-04-17 12:58 UTC: scaled-MMAv5 accumulator layout fallback now shares
  `MMAv5TMemInstructionTileRequirement` via a `ScaledAccumulator` role, while
  keeping the narrow-N scale-fragment requirement as the more specific
  diagnostic. Validation also found and fixed stale fpsan coverage: the
  block-backed `128x128` scaled accumulator is supported and now lives in the
  positive scaled-MMA matrix. Validation: `make -j8`, split-4 scaled
  clean-negative runtime-matrix selector (`11/11/11/11` passed), fpsan
  scaled block/unsupported selector (`6 passed`), Python compile, and
  `git diff --check`.
- 2026-04-17 12:53 UTC: plain MMAv5 tensor-memory layout
  incompatibility is now represented by `MMAv5TMemInstructionTileRequirement`
  in the backend dialect API. Plain `tcgen05.mma` LHS/accumulator diagnostics
  and scaled-MMAv5 TMEM-LHS layout diagnostics now share this requirement path,
  while preserving the current clean unsupported boundary for arbitrary
  row/column permutations inside one MMAv5 instruction tile. Validation:
  `make -j8`, focused split-4 MMA/scaled-LHS clean-negative selector
  (`11/11/11/8` passed), and `git diff --check`. Lit was attempted but is not
  installed/on PATH in this shell.
- 2026-04-17 12:25 UTC: completed the current MMAv5 higher-rank
  dim0-slice `ld/st` bucket. The distributed-layout planner now restores CTA
  ownership in query layouts before atom selection, covering the case where an
  indexed descriptor view keeps a size-1 `block` dimension while the ownership
  basis has moved into row/column bases. It also rejects register-layout
  candidates whose CTA ownership disagrees with the memdesc. This promotes
  `mmav5_twocta, N=64, 16x128b` and removes the now-empty MMAv5 clean-negative
  test bucket. Validation: `make -j8`, manual runtime/opcode probe (`maxdiff
  0.0`), Python compile, `git diff --check`, focused MMAv5 higher-rank
  selector (`13 passed`), and split-4 `ldst_twocta_descriptor` selector
  (`9/7/20/9` passed, expected skips only).
- 2026-04-17 12:13 UTC: restored two-CTA ownership while inferring MMAv5
  descriptor-view encodings. A leading-unit slice over a lifted two-CTA MMAv5
  view can move the CTA ownership basis out of `block` and into trailing
  row/column bases; `tryMakeTMemViewEncoding` now tries to re-home that
  full-extent trailing basis back into `block` before giving up, and no longer
  falls back to a one-CTA encoding for a two-CTA source view. This promotes the
  `mmav5_twocta, N=128, auto` higher-rank dim0 slice row, moves the
  `N=256, 32x32b` row to the real TMEM-capacity boundary, and leaves the
  `N=64, 16x128b` row as the remaining clean register-layout mismatch.
  Validation: `make -j8`, Python compile, `git diff --check`, MMAv5
  higher-rank descriptor selector (`13 passed`), and the direct row-half
  replay selector (`6 passed`). A broader split-4
  `ldst_twocta_descriptor` selector also passed across all shards (`9/5/19/11`
  passed, expected skips only).
- 2026-04-17 12:00 UTC: promoted pure rank-2 two-CTA row-half `ld/st`
  descriptor views for block-backed and MMAv5 TMEM roots. The replay recognizer now
  admits exactly one direct row-half slice over a two-CTA root, and
  `OptimizeTMemLayouts` lowers it through the same leading-dimension replay
  shape that already handled lifted row-half views. This selects the CTA
  block half explicitly instead of splitting the loaded tensor on the block
  base as an ordinary row bit. Validation: `make -j8`, exact direct row-half
  selector (`6 passed`), neighboring lifted row-half selector (`5 passed`),
  combined direct/lifted row-half selector (`17 passed`), Python compile, and
  `git diff --check`.
- 2026-04-17 11:50 UTC: promoted the M64 two-CTA
  tensor-memory-scales descriptor-view `ld/st` row by replaying full-tile
  `memdesc_reshape`/`memdesc_trans` views through the directly supported root
  allocation. The shared direct-unsupported classifier now marks the M64
  scales row-anchor boundary before support-query fallback, and the optimizer
  lowers replayable full views to root `ttng.tmem_load`/`ttng.tmem_store`
  plus tensor reshape/transforms. Validation: `make -j8`, the focused
  `ldst_scales_descriptor_view_cga` selector (`9 passed`), neighboring scales
  descriptor-view/variant selector (`20 passed`), Python compile, and
  `git diff --check`.
- 2026-04-17 11:23 UTC: aligned repeated-N32 scaled-MMAv5 LLVM lowering with
  the backend B-scale storage-through-view contract. Lowering now plans and
  validates B-scale storage via `getMMAv5ScaledBScaleStorageTypeThroughViews(...)`,
  the same helper used by `TCGen5MMAScaledOp` verification and
  `RematerializeRepeatedN32BScale`. Added a padded B-scale descriptor-view
  runtime row that is already storage-supported, so tensor-memory allocation
  does not rematerialize it; this proves LLVM lowering can consume the view
  directly. Validation: `make -j8`, exact compact plus padded B-scale
  descriptor-view tests (`2 passed`), repeated-N32 tile-permuted selector
  (`12 passed`), Python compile, and `git diff --check`.
- 2026-04-17 11:16 UTC: tightened shared-scale materialization to preserve the
  rank-2 shared scale view shape rather than redistributing by the MMA
  logical row count. This fixes two-CTA direct shared B-scale views for
  `blockN < 128`, where the source scale view is padded to the public
  128-row scale fragment shape (`128x4`) even though the MMA logical N is
  64. The focused runtime row now covers both single-CTA and two-CTA direct
  shared scale views. Validation: incremental `make -j8`, exact direct
  shared-scale descriptor-view test (`2 passed`), neighboring two-CTA manual
  scale-copy row (`1 passed`), repeated-N32 tile-permuted scaled-MMA selector
  (`11 passed`), Python compile, `triton-opt ... --triton-nvidia-mma-lowering
  | FileCheck test/TritonNvidiaGPU/mma_lowering.mlir`, and `git diff --check`.
- 2026-04-17 11:12 UTC: direct Gluon scaled-MMAv5 now supports shared-memory
  scale descriptor views by materializing shared scale operands to
  `TensorMemoryScalesLayout` allocations during `triton-tensor-memory-allocation`.
  The conversion uses the same backend dialect helper from both the
  normal MMA-lowering pass and the Gluon allocation-stage path, and the old
  broad `subviews NYI` rejection in MMA lowering is gone. Added a runtime row
  where `tcgen05_mma_scaled` consumes unswizzled shared scale
  `memdesc_reshape`/`memdesc_trans` views directly and the backend emits the
  expected scale `tcgen05.cp` messages before the scaled MMA. Validation:
  `make -j8`, exact new shared-scale descriptor-view runtime test (`1
  passed`), neighboring two-CTA manual scale-copy runtime row (`1 passed`),
  repeated-N32 tile-permuted scaled-MMA selector (`11 passed`), Python compile,
  `triton-opt ... --triton-nvidia-mma-lowering | FileCheck
  test/TritonNvidiaGPU/mma_lowering.mlir`, and `git diff --check`.
- 2026-04-17 11:05 UTC: moved repeated-N32 B-scale descriptor-view
  storage-root recovery into the TritonNvidiaGPU dialect API so
  `TCGen5MMAScaledOp` verification and `RematerializeRepeatedN32BScale` use
  one shared helper. Support is unchanged from the 10:57 checkpoint, but the
  scale-view contract is now layered in the backend dialect instead of being
  verifier-local. Validation: full rebuild from the touched header (`make
  -j8`), exact B-scale descriptor-view repeated-N32 test (`1 passed`), the
  repeated-N32 tile-permuted selector (`11 passed`), and `git diff --check`.
- 2026-04-17 10:57 UTC: extended repeated-`N=32` scaled-MMAv5 B-scale
  rematerialization through descriptor-view chains backed by
  `TensorMemoryScalesLayout`. The verifier now recognizes compact B-scale
  operands whose immediate type is a memdesc view but whose storage root is a
  scales allocation, and the tensor-memory allocation pass rematerializes that
  view-backed compact store into a fresh padded scales allocation before LLVM
  lowering. Added a focused runtime-matrix case that stores compact B scales
  through a descriptor view and consumes the same view in the tile-permuted
  repeated-N32 MMA. Validation: `make -j8`, exact B-scale subslice/view test
  (`1 passed`), repeated-N32 tile-permuted selector (`11 passed`), N64
  tile-permuted neighbor (`20 passed`), narrow-N clean-negative selector (`20
  passed`), Python compile, and `git diff --check`.
- 2026-04-17 10:46 UTC: promoted repeated-`N=32` scaled-MMAv5
  tile-permuted accumulator layouts to positive support by rematerializing
  compact matrix-B scale storage into 64-row-aligned public scale-fragment
  slots. `TCGen5MMAScaledOp` verification/lowering now accept already padded
  B-scale storage or compact storage that the tensor-memory allocation pass can
  rewrite. The allocation pass performs exact reshape/broadcast arithmetic and
  creates a padded B-scale allocation before offsets are assigned. Tensor-memory
  allocation now runs before shared-memory allocation so any layout-conversion
  scratch introduced by rematerialization is allocated before LLVM lowering.
  Runtime-matrix repeated-N32 root and acc-subslice tile-permuted rows are now
  positive for all current scale format pairs, including `nvfp4`; narrow-N
  tile-permuted rows remain clean unsupported. Validation: `make -j8`, direct
  three-format repeated-N32 probe, tile-permuted selector (`30 passed`),
  acc-subslice tile-permuted selector (`10 passed`), neighboring scaled
  root/subslice selector (`55 passed`), narrow-N negative selector (`20
  passed`), core `test_core.py -k tmem_linear_m64` (`21 passed`), Python
  compile, and `git diff --check`.
- 2026-04-17 10:07 UTC: moved raw/support query row-plan selection for direct
  `ld/st` register-layout checking into backend helpers. The Gluon pybind
  bridge and generic TMEM load/store verifier now use
  `getTMemLdStRowPlanForRawQuery(...)` and
  `getTMemLdStRowPlanForSupportQuery(...)`, keeping raw-query override
  suppression and support-query fallback order in `TensorMemoryUtils` rather
  than bridge/verifier-local code. Support is unchanged. Validation:
  `make -j8`, direct row-half/x1 fallback selector (`13 passed`), lifted
  row-half selector (`6 passed`), core `test_core.py -k tmem_linear_m64` (`21
  passed`), Python compile, and `git diff --check`.
- 2026-04-17 10:19 UTC: reprobed pure rank-2 two-CTA row-half replay and
  reverted the probe edits. Normalizing the loaded full support image before
  the existing tensor `reshape -> trans -> split` replay is not sufficient:
  the replay load still selected the CTA-local odd/even partition
  (`1,3,5,...,255`), and candidate full-tile layout conversions either did not
  change the generated split path or corrupted more rows. Keep this frontier as
  requiring explicit CTA-block selection/reconstruction, not another layout
  conversion around the current split. Validation after revert: `make -j8` and
  the current clean-negative selector (`3 passed`).
- 2026-04-17 10:03 UTC: moved the half-row descriptor-view type-only fallback
  refusal out of the Gluon pybind bridge and into the backend-owned
  `disallowTMemLdStTypeOnlyFallback(...)` policy. Support is unchanged, but
  register-layout selection now has one shared backend reason path for
  exact-lowering-required views instead of a half-row-specific bridge guard.
  Validation: `make -j8`, direct half-row selectors (`7 passed`), lifted
  half-row selectors (`6 passed`), x1 fallback-boundary selector (`6 passed`),
  and `git diff --check`.
- 2026-04-17 10:00 UTC: reprobed the pure rank-2 two-CTA row-half frontier
  after the single-CTA direct replay checkpoint and reverted the probe edits.
  The full support load for `block_two_ctas` carries row `128` in the CTA
  `block` basis, so ordinary tensor `reshape`/`split` works inside each CTA
  rather than selecting the logical high half globally. Moving the split factor
  only changed which intra-CTA row bit was selected (`1,3,...` then
  `2,3,6,7,...`), never rows `128..255`. Keep this as a real block-aware
  replay/planner gap: it needs CTA-block selection/predication or an exact
  block-aware reconstruction, not a different split shape.
- 2026-04-17 09:55 UTC: promoted pure rank-2 single-CTA row-half `ld/st`
  descriptor views to replay/RMW support. The replay recognizer now permits a
  pure half-slice over a rank-2 TMEM base when the base layout is single-CTA,
  while continuing to require a shape-transformed chain for two-CTA layouts.
  A probe of the corresponding two-CTA block layout selected odd rows
  (`1,3,...,255`) instead of the logical high half (`128..255`), proving that
  naive tensor split/join is not block-base aware enough for that case. The
  runtime matrix now locks single-CTA N=64/128/256 positives and records the
  two-CTA pure rank-2 half-row as a clean unsupported frontier with a
  block-base diagnostic. Validation: `make -j8`, Python compile, `git diff
  --check`, direct half-row positives (`4 passed`), two-CTA clean negatives
  (`3 passed`), neighboring replay selectors (`7 passed` single CTA, `5
  passed` two CTA), and `test_core.py -k tmem_linear_m64` (`21 passed`).
- 2026-04-17 09:42 UTC: promoted lifted row-half `ld/st` descriptor views
  from clean negatives to replayed RMW support. The replay recognizer now
  treats the `slice(...).index(0)` unit-leading-dimension form as an exact
  half-slice over a rank-2 TMEM base, keeping the temporary unit dimension for
  `join` and reshaping back to the rank-2 view. Single-CTA and two-CTA N=64/128
  row-half cases now pass through full backing TMEM load + tensor split/join +
  full backing store; N=256 rows now reach the true TMEM capacity boundary and
  report clean OOR (`Required: 1024`, hardware limit `512`). Validation:
  `make -j8`, focused row-half selectors (`5 passed` single CTA, `5 passed`
  two CTA), replay/direct-neighbor selector (`12 passed`), two-CTA descriptor
  neighbor selector (`17 passed, 1 skipped`), `test_core.py -k
  tmem_linear_m64` (`21 passed`), Python compile, and `git diff --check`.
- 2026-04-17 09:29 UTC: reprobed two hard frontiers and reverted all probe
  edits. The two-CTA no-scales `warpx2::02_13` row-bit-5 bypass moves the
  failure to descriptor synthesis with no representable 64x4 or bounded 32x4
  MMAv5 shared descriptor. The scaled-MMAv5 repeated-`N=32` guard bypass emits
  MMAs but computes wrong results under the old two-column B-scale fragment
  mapping, and a one-column B-scale address probe faults as misaligned. These
  remain real descriptor/address and B-scale storage-rematerialization
  frontiers, not guard-lift fixes.
- 2026-04-17 09:19 UTC: promoted the single-CTA high-quadrant
  multidim-slice `ld/st` descriptor views from clean negatives to replayed
  RMW support. `TensorMemoryUtils` now recognizes replayable reshape/transpose
  plus exact half-slice descriptor chains, Gluon layout inference admits them
  through a backend-owned blocked fallback, TMEM load/store verification lets
  only plain replayable loads/stores through, and `OptimizeTMemLayouts`
  rewrites them to full backing TMEM load + tensor split/join + full backing
  store before direct lowering. Identity and scrambled-column 32x32 rows now
  pass with full backing `32x32b.x128` RMW packets. Validation: `make -j8`,
  replay/direct-neighbor multidim slice selector (`12 passed`), nearby two-CTA
  descriptor slice/index selector (`13 passed, 1 skipped`), and
  `test_core.py -k tmem_linear_m64` (`21 passed`).
- 2026-04-17 09:07 UTC: moved the remaining direct `ld/st` blocked-layout
  fallback policy out of the Gluon pybind bridge and into
  `TensorMemoryUtils` as `getTMemLdStBlockedFallbackLayouts(...)`. Support is
  unchanged; this keeps register-layout selection policy backend-owned as the
  packet/schedule frontiers are generalized. Validation: `make -j8`,
  `git diff --check`, high-quadrant descriptor-view clean-negative
  representative (`1 passed`), runtime-matrix direct-layout representative
  (`1 passed`), and `test_core.py -k tmem_linear_m64` (`21 passed`).
- 2026-04-17 09:01 UTC: moved non-f32 `load_min/load_max` software fallback
  ahead of the non-scales guard, so `TensorMemoryScalesLayout` int8 reductions
  now use ordinary scales `tcgen05.ld` plus layout-aware `ttgl.reduce(axis=1)`
  instead of attempting hardware `ld.red`. Added runtime-matrix scales
  reduction coverage for min/max with and without `abs`; integer NaN
  propagation remains a clean error. Validation: `make -j8`, Python compile,
  focused scales reduction selector (`4 passed`), neighboring scales `ld/st`
  selector (`19 passed`), broader reduction selector (`113 passed`), and
  `git diff --check`.
- 2026-04-17 08:57 UTC: extended the non-f32 `load_min/load_max` software
  fallback to supported modifiers. Software reduction now applies `abs`
  before reducing for non-f32 dtypes and supports NaN propagation for f16/bf16
  through the same layout-aware reduction path; integer NaN propagation remains
  a clean semantic error. The runtime matrix now treats i32 `abs`, f16/bf16
  `abs`, and f16/bf16 NaN propagation as positive software-reduction coverage
  for both direct contract and descriptor-chain rows. Validation: `make -j8`,
  Python compile for touched files, `git diff --check`, focused non-f32
  contract positive (`22 passed`) and negative (`2 passed`) selectors,
  descriptor-chain positive (`18 passed`) and negative (`2 passed`) selectors,
  core i32 representative (`1 passed`), and broader reduction selector
  (`109 passed`).
- 2026-04-17 08:52 UTC: promoted no-modifier non-f32 TMEM
  `load_min/load_max` rows from clean negatives to positive software-reduction
  coverage for non-scales layouts. The API now lowers i32, i16, i8, f16, and
  bf16 no-modifier reductions through ordinary `tcgen05.ld` plus
  layout-aware `ttgl.reduce(axis=1)`; bf16 uses a dedicated combine function
  that compares via the existing f32 min/max path and casts the selected value
  back to bf16 to satisfy reduction-region type checking. `abs` and NaN
  propagation modifiers remain clean unsupported for non-f32. Validation:
  `make -j8`, Python compile for touched files, focused contract positive
  (`12 passed`) and negative (`12 passed`) selectors, descriptor-chain
  positive (`10 passed`) and negative (`8 passed`) selectors, core i32
  representative (`1 passed`), broader reduction selector (`107 passed`), and
  `git diff --check`.
- 2026-04-17 08:45 UTC: converted the stale runtime-matrix mixed-layout
  `ld.red` clean negative into positive software-reduction coverage. The
  backend already lowers this source layout as normal `tcgen05.ld` plus
  layout-aware `ttgl.reduce(axis=1)`; the runtime matrix now asserts no
  hardware `tcgen05.ld.red` is emitted for the mixed row/column layout.
  Validation: `make -j8`, Python compile for the runtime matrix, exact mixed
  runtime selector (`12 passed`), matching core representative (`1 passed`),
  broader reduction selector (`86 passed`), and `git diff --check`.
- 2026-04-17 08:25 UTC: centralized lowered-subview base-offset subtraction
  in `TensorMemoryUtils` as `getTMemSubviewRelativeBaseOffset(...)`. Direct
  `ld/st` raw-query, support-query, source-column raw-query, and `tcgen05.copy`
  lowering now use the shared helper instead of reimplementing the same
  `baseOffset - alreadyAdjustedBase` clamp locally. Support is unchanged.
  Validation: `make -j8`, direct `invalid.mlir` verifier, Python compile for
  affected Gluon tests, expanded descriptor/copy runtime selector (`96 passed,
  1 skipped`), `test_core.py -k tmem_linear_m64` (`21 passed`), and
  `git diff --check`.
- 2026-04-17 08:20 UTC: moved query-type direct `ld/st` scalarization
  refinement from LLVM lowering into `TensorMemoryUtils` as
  `refineTMemLdStQueryTypeEncodingInfo(...)`. Query-type lowering now
  computes the encoding info and asks the backend to refine the 32x32
  descriptor-view scalar-message case, instead of mutating
  `numRegsPerMessage` in the emission layer. Support is unchanged. Validation:
  `make -j8`, direct `invalid.mlir` verifier, Python compile for affected
  Gluon tests, focused M64/direct-layout runtime selector (`66 passed, 1
  skipped`), `test_core.py -k tmem_linear_m64` (`21 passed`), and
  `git diff --check`.
- 2026-04-17 08:16 UTC: moved the direct `ld/st` source-column subview
  detection and borrowed-source row-plan selection from LLVM lowering into
  `TensorMemoryUtils`. Lowering now asks
  `getTMemLdStPure2DColumnSubview(...)`,
  `getTMemLdStSourceColumnSubviewSupportQueryPlan(...)`, and
  `getTMemLdStSourceColumnSubviewRawQueryRowPlan(...)` instead of carrying
  the 2D TMEM column-subview and 64x32 f32 backing-row-plan predicates
  locally. Support is unchanged. Validation: `make -j8`, direct
  `invalid.mlir` verifier, Python compile for affected Gluon tests, focused
  M64/direct-layout runtime selector (`66 passed, 1 skipped`),
  `test_core.py -k tmem_linear_m64` (`21 passed`), and `git diff --check`.
- 2026-04-17 08:11 UTC: moved the direct-root backing row-plan preference
  from LLVM lowering into `TensorMemoryUtils` as
  `preferBackingTMemLdStRowPlanForDirectRoot(...)`. Raw-query,
  support-query, and query-type lowering now call the backend helper instead
  of carrying a lowering-local row-anchor/range predicate for root M64 TMEM
  allocs. Support is unchanged. Validation: `make -j8`, direct `invalid.mlir`
  verifier, Python compile for affected Gluon tests, focused
  M64/direct-layout runtime selector (`64 passed, 1 skipped`),
  `test_core.py -k tmem_linear_m64` (`21 passed`), and `git diff --check`.
- 2026-04-17 08:07 UTC: moved direct `ld/st` generic compatible-layout
  fallback ordering into `TensorMemoryUtils` as
  `getTMemLdStGenericCompatibleLayouts(...)`. The backend now owns the
  canonical M64 split-N compatible-layout deferral for generic fallback
  layouts, while the Gluon bridge only appends backend-provided attributes.
  Support is unchanged. Validation: `make -j8`, direct `invalid.mlir`
  verifier, Python compile for affected Gluon tests, focused
  M64/direct-layout runtime selector (`64 passed, 1 skipped`),
  `test_core.py -k tmem_linear_m64` (`21 passed`), and `git diff --check`.
- 2026-04-17 08:04 UTC: moved direct `ld/st` row-plan candidate layout
  enumeration from the Gluon bridge into `TensorMemoryUtils` as
  `TMemLdStCandidateLayout` plus
  `getTMemLdStCandidateLayoutsForQuery(...)`. The backend now owns the
  row-plan, exact-view M64 direct-layout predicate, atom order, and
  split-N-fastpath refusal used to propose direct register layouts; Gluon only
  normalizes and converts those layouts to Python objects. Support is
  unchanged. Validation: `make -j8`, direct `invalid.mlir` verifier, Python
  compile for affected Gluon tests, focused M64/direct-layout runtime selector
  (`64 passed, 1 skipped`), `test_core.py -k tmem_linear_m64` (`21 passed`),
  and `git diff --check`.
- 2026-04-17 08:01 UTC: moved the direct `ld/st` atom search order out of
  the Gluon pybind bridge and into `TensorMemoryUtils` as
  `getTMemLdStAtomSearchOrder(...)`. The bridge still tries the same atom
  sequence, including the desired-atom priority, but the ordering is now a
  backend-owned policy shared by raw-query and support-query layout selection.
  Support is unchanged. Validation: `make -j8`, direct `invalid.mlir`
  verifier, Python compile for the affected Gluon tests, focused
  M64/direct-layout runtime selector (`64 passed, 1 skipped`),
  `test_core.py -k tmem_linear_m64` (`21 passed`), and `git diff --check`.
- 2026-04-17 07:56 UTC: sharpened the `tcgen05.ld.red` unsupported-layout
  diagnostic. The primary verifier error now says the reduction register
  layout is not directly supported by `tcgen05.ld.red` lowering instead of
  always blaming N sharding; the detailed support note now carries the exact
  failed proof, including M sharding for explicit `16x64b`/`16x128b`/`16x256b`
  layouts. Support is unchanged. Validation: `make -j8`, direct
  `invalid.mlir` verifier, Python compile for the runtime matrix, focused
  reduction selector (`59 passed`), and `git diff --check`.
- 2026-04-17 07:52 UTC: consolidated scaled-MMAv5 kind-derived instruction
  facts into `MMAv5ScaledInstructionInfo`. Lowering now asks one backend
  helper for the selected MXFP kind, MXFP4 classification, K instruction size,
  operand element bit sizes, and scale-factor column grouping before calling
  the shared scale-fragment planner. Support is unchanged. Validation:
  `make -j8`, direct `invalid.mlir` verifier, Python compile for affected
  Gluon tests, focused scaled-MMAv5 runtime selector (`51 passed`), and
  `git diff --check`.
- 2026-04-17 07:49 UTC: moved scaled-MMAv5 operation-kind and element-format
  metadata out of LLVM lowering and into the TritonNvidiaGPU dialect backend.
  `MMAv5ScaledMxfpKind`, `getMMAv5ScaledMxfpKind(...)`,
  `isMMAv5ScaledMxfp4(...)`, `getMMAv5ScaledFormatBitSize(...)`, and
  `getMMAv5ScaleFactorColsPerSet(...)` now live beside the scaled accumulator
  and scale-fragment planners. Support is unchanged; opcode string emission
  remains in lowering. Validation: `make -j8`, direct `invalid.mlir`
  verifier, Python compile for affected Gluon test files, and focused
  scaled-MMAv5 runtime selector (`51 passed`).
- 2026-04-17 07:44 UTC: moved two more Gluon-local direct `ld/st`
  selection policies into `TensorMemoryUtils`: the non-value M64 direct-atom
  canonical-layout retry and the legacy `TensorMemoryLayout` auto path that
  tries `32x32b` first outside M64. Support is unchanged. Validation:
  `make -j8`, direct `invalid.mlir` verifier, Python compile for the affected
  test files, focused M64/direct-layout runtime selector (`64 passed, 1
  skipped`), focused `test_core.py` selector (`21 passed`), and
  `git diff --check`.
- 2026-04-17 07:41 UTC: moved the remaining Gluon-local M64 direct-view
  register-layout selection predicates into `TensorMemoryUtils`. The backend
  now owns when to defer the canonical M64 split-N compatible layout, when to
  use the exact descriptor-view linear planner for direct M64 views, and when
  raw-query row-plan overrides are disallowed for `32x32` descriptor views.
  Support is unchanged. Validation: `make -j8`, direct `invalid.mlir`
  verifier, focused M64/descriptor-view runtime selector (`64 passed, 1
  skipped`), `test_core.py -k tmem_linear_m64` (`21 passed`), Python
  compile for the affected test files, and `git diff --check`.
- 2026-04-17 07:33 UTC: repeated-N32 scaled-MMAv5 support was re-probed from
  the B-scale fragment side. A one-column fragment stride still faults with a
  misaligned scale address; exact scale-layout algebra identifies logical
  N32/N64/N96 rows as physical scale columns 4/8/12, but using the corresponding
  aligned offsets still leaves only the first N32 instruction correct and the
  remaining N instructions numerically wrong. The clean negative is preserved,
  and `MMAv5ScaledRepeatedN32ScaleFragmentRequirement` now records the public
  matrix-B scale fragment's minimum addressable N span so the diagnostic points
  at the missing scale-fragment storage/rematerialization contract. Validation:
  `make -j8`, direct `invalid.mlir` verifier, focused scaled-MMAv5 selector
  (`50 passed`).
- 2026-04-17 07:18 UTC: `tcgen05.ld.red` reduction-layout support now returns
  a structured support object carrying the supported lane-split mask or an
  exact unsupported sharding reason. This preserves current functionality but
  makes the remaining non-M64 explicit n-sharded reduction gap concrete: some
  layouts split N through unsupported thread bases, while the explicit
  `16x64b`/`16x128b`/`16x256b` variants also carry M rows in the register
  value stream and would need a software reduction/writeback schedule, not a
  simple lane-16 shuffle extension. Validation: `make -j8`, direct
  `invalid.mlir` verifier, focused reduction selector (`45 passed`), and
  `git diff --check`.
- 2026-04-17 07:10 UTC: re-probed the single-CTA identity high-quadrant
  `32x32` descriptor-view `ld/st` boundary. Moving the already-lowered row
  origin from the TMEM base to packet offsets, scaling it into packet-row
  units, and borrowing the root support query did not produce correct code:
  the variants updated the top-right footprint, row-16 footprint, or faulted
  with misaligned-address packets. All temporary edits were removed. This
  remains a real warp-row-anchor / packet-footprint planner boundary rather
  than a stale row-origin base-offset bug.
- 2026-04-17 06:51 UTC: moved the row-zero lifted reinterpret query-type
  rescue guard from LLVM lowering into `TensorMemoryUtils` as
  `disallowTMemLdStQueryTypeRescue(...)`. Lowering no longer inspects zero
  row/column bases directly before falling back from raw-query lowering to
  query-type lowering. Support is unchanged. Validation: `make -j8`, direct
  `invalid.mlir` verifier, focused M64/multidim/refresh selector
  (`43 passed`), `test_core.py -k tmem_linear_m64` (`21 passed`), and
  `git diff --check`.
- 2026-04-17 06:43 UTC: aligned LLVM lowering with the backend-owned M64
  query-ordering predicate. `lowerTMemLdStFromTypes(...)` no longer keeps the
  raw-query-first ordering hard-coded for all descriptor values; it asks
  `shouldPreferTMemLdStQueryTypeLayoutsBeforeRawQuery(...)`, using the warp
  count from the selected register layout. Support is unchanged, but
  register-layout selection and lowering now consume the same backend decision
  for non-projected M64 split-N views. Validation: `make -j8`, direct
  `invalid.mlir` verifier, focused M64/multidim selector (`41 passed`),
  `test_core.py -k tmem_linear_m64` (`21 passed`), and `git diff --check`.
- 2026-04-17 06:40 UTC: moved the remaining Gluon register-layout fallback
  predicates for M64 query ordering, half-row descriptor views, and two-CTA
  int8 descriptor-view type-only fallback refusal into `TensorMemoryUtils`.
  The pybind bridge now delegates those support/raw-query decisions to the
  backend instead of reinterpreting raw `LinearLayout` rows, zero bases, and
  view chains locally. Support is unchanged. Validation: `make -j8`, direct
  `invalid.mlir` verifier, focused M64/half-row/two-CTA selector (`62 passed,
  1 skipped`), `test_core.py -k tmem_linear_m64` (`21 passed`), and
  `git diff --check`.
- 2026-04-17 06:35 UTC: moved TMEM descriptor-view base-adjustment semantics
  from LLVM lowering into `TensorMemoryUtils`. The backend now owns
  `getAlreadyAdjustedTMemSubviewBaseOffset(...)` and
  `preserveTMemLdStSupportQueryBaseOffset(...)`; `ld/st` and copy lowering
  consume the shared helpers instead of keeping local packet-base arithmetic.
  Support is unchanged, but the packet-base/per-message-offset work now has a
  backend-owned seam for subtracting already-lowered subview bases. Validation:
  `make -j8`, direct `invalid.mlir` verifier, focused descriptor-view/copy
  subview selector (`38 passed, 1 skipped`), and `git diff --check`.
- 2026-04-17 06:30 UTC: consolidated the canonical warpx2 shared-source
  layout contract inside `TensorMemoryUtils`. The runtime support preflight
  and the `warpx2::02_13` direct-seed descriptor path now share the same
  `hasOnlyWarpx2SharedSourceDims(...)` and
  `hasCanonicalWarpx2SharedSourceOffsetBases(...)` checks instead of carrying
  duplicate expected offset-basis tables. Support is unchanged; this keeps the
  current source-rematerialization boundary in one backend location before any
  future `warpx2` source-message schedule work. Validation: `make -j8`,
  direct `invalid.mlir` verifier, focused warpx2 copy selector (`30 passed`),
  and `git diff --check`.
- 2026-04-17 06:27 UTC: moved scaled-MMAv5 scale-factor fragment arithmetic
  out of LLVM lowering and into the TritonNvidiaGPU dialect backend as
  `getMMAv5ScaleFactorFragment(...)`. Lowering still computes the scale
  operation kind locally, but the TMEM column offset, scale sub-column ID,
  columns-per-scale-block, and K-word index now come from a shared backend
  planner object. Support is unchanged: repeated-N32 and narrow-N scaled
  rows remain clean scale-fragment/storage negatives until the B-scale
  fragment representation changes. Validation: `make -j8`, direct
  `invalid.mlir` verifier, focused scaled-MMAv5 selector (`52 passed`), and
  `git diff --check`.
- 2026-04-17 06:22 UTC: moved the M64 split-N raw-query fallback request
  filter into `TensorMemoryUtils` as
  `getCanonicalM64SplitNLayoutForRawQueryRequest(...)`. The pybind bridge
  still normalizes and returns the layout, but the backend now owns when the
  `auto`/`32x32b_splitn`/`16x32bx2` fallback is allowed and uses the shared
  M64 descriptor-type predicate. Validation: `make -j8`, py-compile for
  Python runtime files, `m64_splitn or ld_red_m64` (`39 passed`), and
  `test_core.py -k tmem_linear_m64` (`21 passed`).
- 2026-04-17 06:19 UTC: moved the M64 reduction-load result-type
  canonicalization helper from `python/src/gluon_ir.cc` into
  `TensorMemoryUtils` as `canonicalizeTMemLoadReductionType(...)`. The helper
  still only rewrites noncanonical rank-2 M64 f32 reduction loads whose exact
  raw query proves the canonical split-N layout and the result remains
  reduction-friendly. Validation: `make -j8`, py-compile for the Python
  runtime files, and the focused reduction selector (`47 passed`).
- 2026-04-17 06:16 UTC: moved TMEM `ld/st` access-atom parsing,
  descriptor-handle split-N request mapping, M64 split-N descriptor-type
  classification, and requested-vs-realized atom compatibility from
  `python/src/gluon_ir.cc` into `TensorMemoryUtils`. This is behavior
  preserving but removes another pybind-local interpretation of the backend
  planner contract. Validation: `make -j8`, py-compile for the Python runtime
  files, focused M64/split-N plus scales-variant selector (`44 passed`),
  `test_core.py -k tmem_linear_m64` (`21 passed`), and x1/scales
  descriptor-view variant diagnostics (`13 passed`).
- 2026-04-17 06:10 UTC: re-probed the single-CTA identity `32x32`
  multidim-slice descriptor-view `ld/st` boundary with a base-relative support
  query. Two variants were rejected: the first selected `16x32bx2.x32` and
  updated the top-right quadrant, while the row-origin-adjusted version
  selected scalar `32x32b.x1` packets but emitted offsets `0..31` and still
  updated the wrong footprint (`3072` mismatches). The temporary source edit
  was removed, `make -j8` rebuilt the restored tree, and the exact clean
  negative remains green (`1 passed`). This row is still a real
  packet-footprint/rematerialization gap, not a support-layout lookup gap.
- 2026-04-17 06:01 UTC: moved Gluon's value-aware reduction register-layout
  selection into `TensorMemoryUtils`. The new
  `getTMemLoadReductionLayoutForMemDesc(...)` backend API owns raw-query
  reduction compatibility checks, descriptor-view exact-query refusal,
  row-plan selection, and the M64 split-N fallback; `python/src/gluon_ir.cc`
  now acts as a thin bridge from backend encoding to Gluon layout. Support is
  unchanged. A pre-edit probe confirmed explicit `32x32b_splitn` already
  works for representative noncanonical `N=256` `ld.red` layouts; explicit
  N-sharded `ld.red` variants remain true reduction-contract boundaries rather
  than stale pybind logic. Validation: `make -j8`, focused `ld.red`
  descriptor-chain/explicit/M64 selector (`55 passed, 1523 deselected`),
  py-compile, and `git diff --check`.
- 2026-04-17 05:53 UTC: moved the simple M64 split-N raw-query proof out of
  `python/src/gluon_ir.cc` and into `TensorMemoryUtils`. The backend now owns
  `hasCanonicalM64SplitNRows(...)` and
  `getCanonicalM64SplitNLayoutForRawQuery(...)`, and the pybind
  `get_reg_layout()` / `ld.red` reduction paths call those helpers instead of
  duplicating row/column-basis arithmetic. Support is unchanged; this is a
  layering cleanup around the M64 split-N paths that were recently promoted.
  Validation: `make -j8`, py-compile for the touched Python files,
  `test_tmem_runtime_matrix.py -k "ld_red_m64 or m64_splitn"`
  (`39 passed, 1539 deselected`), `test_core.py -k tmem_linear_m64`
  (`21 passed, 17945 deselected`), and `git diff --check`.
- 2026-04-17 05:46 UTC: moved the explicit `16x32bx2` two-CTA
  tensor-memory-scales descriptor-view `ld/st` diagnostic out of the Python
  binding and into `TensorMemoryUtils`. The new
  `getUnsupportedDirectTMemLdStVariantReason(...)` first preserves atom
  footprint diagnostics, then explains the scales-CGA half-tile boundary: this
  view is directly realizable by `32x32b` or wider n-sharded scale atoms, but
  not by `16x32bx2` because that atom requires the half-tile split as a
  lane-selected second-half offset. A speculative planner probe that tried to
  treat the view as 16-row-per-CTA did not promote support and was reverted.
  Validation: `make -j8`, exact `16x32bx2` clean negative (`1 passed`), full
  `ldst_scales_descriptor_view_cga` selector (`9 passed, 1569 deselected`),
  py-compile for `test_tmem_runtime_matrix.py`, and `git diff --check`.
- 2026-04-17 05:37 UTC: removed the frontend `tcgen05_mma_scaled`
  `TensorMemoryLayout` blockM=64 assertion and pinned that this unsupported
  ISA shape now reaches the backend verifier. The new runtime-matrix negative
  case uses `M=64,N=128,K=128` so shared-memory B construction is legal, then
  checks for `only supports instruction shape blockM=128` with no Python
  assertion or PassManager crash. Adjacent scaled direct-layout positives still
  pass. Also recorded the latest repeated-N32 probe evidence: guard lifting
  with the current B-scale math compiles but gives wrong output, while forcing
  one-column B scale fragments faults/fails fresh processes, so repeated-N32
  remains a real B-scale fragment alignment/storage boundary. Validation:
  `make -j8`, exact new negative (`1 passed`), adjacent scaled direct-layout
  selector (`4 passed`), and `git diff --check`.
- 2026-04-17 05:27 UTC: promoted the ordinary `tcgen05.copy.4x256b`
  refresh-image boundary into a typed backend requirement. The new
  `TMemCopy4x256RefreshImageRequirement` records the 4x8 refresh image
  contract (logical row bits in TMEM columns, low column bits in rows 32/64,
  high column bit at destination dword +4, and the low/high source-column
  message split), and both direct-destination layout support and descriptor
  plan realization now use one formatter. Support is unchanged: ordinary
  contiguous four-row copies remain clean negatives until an explicit
  refresh-image view/remap plus load/store contract exists. Validation:
  `make -j8`, direct `invalid.mlir` verifier, focused
  `cp_no_scales_4x256b` selector (`3 passed, 1574 deselected`), and
  `git diff --check`.
- 2026-04-17 05:23 UTC: introduced `MMAv5ScaledAccumulatorSupport`, a shared
  scaled-MMAv5 accumulator support carrier containing the selected accumulator
  layout proof plus typed narrow-N and repeated-N32 scale-fragment
  requirements. Verifier and LLVM lowering now consume the same support object
  instead of recomputing separate layout/error probes, while support and
  diagnostics remain unchanged. Validation: `make -j8`,
  `triton-opt --split-input-file test/TritonNvidiaGPU/invalid.mlir
  --verify-diagnostics`, focused scaled-MMAv5 positive/narrow/repeated
  selector (`50 passed, 1527 deselected`), and `git diff --check`.
- 2026-04-17 05:21 UTC: added cheap frontend coverage for the backend-owned
  4x256 refresh direct-`ld/st` type rejection. The new
  `test_frontend.py` cases call `tensor_memory_descriptor_type.get_reg_layout`
  directly for the refresh-shaped `f32` layout and the raw physical-bitcast
  `i8` layout, pinning that the type-only API now reports the C++ backend
  packet-footprint reason. Validation: py-compile for `test_frontend.py`,
  exact two-test frontend selector (`2 passed`), and `git diff --check`.
- 2026-04-17 05:18 UTC: moved the 4x256 refresh direct `ld/st`
  type rejection out of Python layout-pattern helpers and into the backend
  packet-footprint layer. `TensorMemoryUtils` now owns both the refresh-shaped
  `f32` layout diagnostic and the raw physical-bitcast diagnostic, and the
  type-only Gluon register-layout bridge asks that C++ utility before trying
  to promise a load/store register layout. Validation: `make -j8`,
  py-compile for `blackwell/__init__.py` and `test_tmem_runtime_matrix.py`,
  focused `ldst_4x256b_refresh`/`cp_no_scales_4x256b` selector (`5 passed,
  1572 deselected`), adjacent M64 split-N selector (`16 passed, 1561
  deselected`), and `git diff --check`.
- 2026-04-17 05:13 UTC: removed the adjacent Python `store()` equality guard
  for rank-2 M64 f16/bf16 descriptors. Store validation now relies on
  `ttng.tmem_store` / `TMEMStoreOp::verify()` and the backend register-layout
  proof instead of a frontend shape-family check. Validation: `make -j8`,
  py-compile for `blackwell/__init__.py` and `test_tmem_runtime_matrix.py`,
  focused split-N plus blocked-layout selector (`26 passed, 1551
  deselected`), `test_core.py -k tmem_linear_m64` (`21 passed, 17945
  deselected`), and `git diff --check`.
- 2026-04-17 05:11 UTC: removed the Python type-only M64 16-bit split-N
  shortcut from descriptor-handle `get_reg_layout()`. The C++ memdesc
  register-layout bridge now lets the canonical M64 split-N raw-query
  recognizer cover 16-bit f16/bf16 M64 descriptor values as well as the
  existing 32-bit path, and keeps `32x32b_splitn` on the backend-owned M64
  `16x32bx2` route for that family. Added focused runtime coverage comparing
  `auto`, `32x32b_splitn`, and explicit `16x32bx2` for representative 16-bit
  M64 rows. Validation: `make -j8`, py-compile for `blackwell/__init__.py` and
  `test_tmem_runtime_matrix.py`, focused split-N selector (`24 passed, 1553
  deselected`), adjacent M64 reduction/default selector (`34 passed, 1543
  deselected`), and `git diff --check`.
- 2026-04-17 05:03 UTC: removed the remaining Python M64 split-N auto-layout
  shim. The C++ `findDirectLayoutForMemDesc` raw-query recognizer now treats
  `instr_variant="auto"` as eligible for the canonical M64 split-N layout and
  tries that before generic raw-query layout selection for rank-2 M64 f32
  non-scales descriptor values. `load(layout=None)` now routes through
  `get_reg_layout(auto)` directly. Validation: `make -j8`, py-compile for
  `blackwell/__init__.py` and `test_tmem_runtime_matrix.py`, focused M64
  auto/default selector (`32 passed, 1543 deselected`), split-N family
  selector (`28 passed, 1547 deselected`), and `git diff --check`.
- 2026-04-17 05:00 UTC: re-probed ordinary contiguous
  `tcgen05.cp.4x256b` by temporarily removing the direct-support and lowering
  guards. The kernel emitted `tcgen05.cp.cta_group::1.4x256b`, but copied the
  refresh pattern into an ordinary parent view (`out[:4]` began
  `[0, 8, 16, 24, 0, 8, 16, 24]` and the rest of the first four rows were
  zero). The guard was restored and the focused `cp_no_scales_4x256b` selector
  passed (`3 passed, 1572 deselected`). Treat ordinary contiguous 4x256b as a
  real missing representation/remapping layer, not an ISA-enable toggle.
- 2026-04-17 04:53 UTC: moved the remaining explicit-`32x32b` M64
  `ld.red` split-N canonicalization out of Python `_load_red` and into the C++
  Gluon/TTGIR bridge. `create_tmem_load` now receives the active warp count
  and, for rank-2 M64 f32 non-scales reduction loads, recognizes the same
  simple noncanonical split-N raw query image as the backend register-layout
  bridge and selects the canonical split-N reduction layout. The Python
  `_is_simple_m64_splitn_tmem_layout`,
  `_has_canonical_m64_splitn_rows`, and
  `_try_m64_reduction_layout_for_explicit_32x32b` helpers were deleted.
  Validation: `make -j8`, py-compile for `blackwell/__init__.py` and
  `test_tmem_runtime_matrix.py`, focused M64 explicit/default selector
  (`23 passed, 1552 deselected`), adjacent `ld.red` selector including
  explicit compatible positives and n-sharded clean negatives
  (`59 passed, 1516 deselected`), and `git diff --check`.
- 2026-04-17 03:33 UTC: removed the stale scaled-MMAv5 scale-subcolumn
  environment override hooks (`TRITON_MMAV5_SCALE_ID_MAP_A/B`). Scale-fragment
  lowering now uses the deterministic subword index derived from the K
  repetition directly. This deletes probe-only configurability that could
  obscure the real B-scale fragment boundary. Validation: `make -j8`, focused
  scaled-MMAv5 positive/narrow/repeated selector (`41 passed, 1534
  deselected`), and `git diff --check`.
- 2026-04-17 03:27 UTC: typed the scaled-MMAv5 sub-32-N accumulator boundary.
  Layouts that are valid for plain MMAv5 only with an N<32 instruction now
  derive an `MMAv5ScaledNarrowNScaleFragmentRequirement` instead of falling
  through to the generic tile-permuted accumulator error. Verifier and lowering
  both report that public block-scaled MMAv5 has a minimum N=32 tile and
  matrix-B scale fragments at 64-column alignment. Support is unchanged:
  narrow tile-permuted scaled accumulators still need a real scale-fragment
  storage/scheduling model or a larger directly supported tile. Validation:
  `make -j8`, py-compile for `test_tmem_runtime_matrix.py`, direct
  `invalid.mlir` verifier from the build dir, focused scaled narrow/repeated
  selector (`40 passed, 1535 deselected`), and `git diff --check`.
- 2026-04-17 03:19 UTC: moved the no-scales dense copy sub-instruction
  column-permutation boundary into the shared typed requirement layer. Dense
  direct-destination column-tile failures now derive a
  `TMemCopyInstructionColumnPermutationRequirement` and route it through the
  existing destination-mask proof, so tile_n=1/2 layouts report that a public
  copy atom writes the full instruction-column footprint while the permutation
  would need only selected destination-column runs. The same requirement now
  also covers shared-descriptor `NonContiguousOffset` instruction-column
  failures. Support is unchanged: these layouts still need a narrower atom,
  destination-column mask, or a proven non-overwriting multi-message schedule.
  Validation: `make -j8`, py-compile for `test_tmem_runtime_matrix.py`, direct
  `invalid.mlir` verifier from the build dir, focused tile-permuted
  sub-instruction selector (`6 passed`), adjacent copy clean-negative selector
  (`36 passed, 1539 deselected`), and `git diff --check`.
- 2026-04-17 02:55 UTC: re-probed the single-CTA identity 32x32
  multidim-slice `ld/st` boundary after the stale offset-special deletion. No
  source probe was kept. Findings: borrowing the 128x128 root support query
  lets lowering compute an origin-bearing support plan, but `get_reg_layout()`
  cannot return a valid reshaped 32x32 register layout for that full support
  image. A row-local 32x32 register-layout probe emitted the desired
  `32x32b.x1` offsets (`0,4,...,28`) but still updated warp-selected row
  bands instead of the single identity-layout 32-row window. Keeping the full
  support base offset double-applied the already lowered view base and produced
  no visible update. Treat this as a true packet-footprint boundary for direct
  public `tcgen05.ld/st` until a real read/modify/write, row mask, or
  equivalent per-warp row-window decomposition exists; do not retry by only
  forcing root support queries, row-anchor bypasses, or type-only fallback
  layouts.
- 2026-04-17 02:08 UTC: deleted the hard-coded canonical contiguous 32x32
  TMEM subview lowering offset. The generic subview lowering path now handles
  that mixed-layout positive through `inferStandaloneTMemLdStQueryLayout(...)`,
  `tryMakeLeadingUnitSubviewLayout(...)`, and
  `getTMemLdStQueryOriginDeltaBaseOffset(...)`, while the identity 32x32 clean
  negative remains guarded. A temporary origin-translated support-query probe
  that preserved the descriptor type layout still emitted `16x32bx2.x32` for
  the identity subview and produced the known `2048` mismatches, so no support
  promotion was kept. Validation: `make -j8`, py-compile, `git diff --check`,
  focused identity/mixed/x1 selector (`6 passed`), M64/split-N selector
  (`49 passed`), and two-CTA promoted selector (`9 passed, 2 skipped`).
- 2026-04-17 01:32 UTC: added a value-based `ld/st` physical-support plan
  helper and wired handle-aware Gluon `get_reg_layout()` fallback to consult it
  before falling back to standalone type-only support. This is a scaffold for
  the descriptor-view completeness work: future promoted views can now derive
  register layout from the same `getTMemLdStSupportQueryPlan(...)` image that
  lowering tries. Identity 32x32 multidim slice probes remain negative:
  forcing only folded base, or folded base plus a folded query layout, still
  produced wrong output because the register layout selected by
  `get_reg_layout()` was not derived from that exact support image. Validation:
  `make -j8`, py-compile, `git diff --check`, focused identity/mixed/x1
  selector (`6 passed`), M64/split-N selector (`49 passed`), and two-CTA
  promoted selector (`9 passed, 2 skipped`).
- 2026-04-17 00:50 UTC: fixed an analysis-time invalid-IR blocker in nested
  TMEM descriptor-view chains. `ModuleAxisInfoAnalysis` can ask MLIR for fold
  results while building coalescing information; `MemDescSubsliceOp::fold`
  previously mutated nested subslices in place, which is unsafe for tensor
  memory because each subview can carry a different active linear encoding.
  The fold is now disabled for TMEM subviews, AxisInfo skips memdesc SSA values,
  and Gluon layout propagation ignores values outside its requested tensor type
  domain. Added a lit regression in `test/Gluon/infer_coalesced_encoding.mlir`
  for the exact nested active-encoding preservation. Validation: `make -j8`,
  direct `triton-opt --split-input-file ... --gluon-infer-coalesced-encodings`
  regression check (`lit`/`FileCheck` unavailable locally), identity
  multidim `ld/st` boundary selector (`2 passed`), two-CTA promoted/adjacent
  selector (`5 passed, 2 skipped`), and `git diff --check`.
- 2026-04-17 00:12 UTC: promoted the two-CTA higher-rank index
  direct `ld/st` bucket that had been blocked by the generic row-anchor
  diagnostic. The row-anchor check now defers two-CTA linear views with a
  nontrivial block dimension to concrete register-layout selection and
  `TMEMLoad/Store` verification, because those schedules can materialize the
  row anchors through the block dimension rather than as pure row bases. The
  runtime matrix now treats `block_two_ctas` and `mmav5_twocta` index views at
  `(N,variant)=(64,auto)` and `(128,16x128b)` as positive. Validation:
  `make -j8`, py-compile for `test_tmem_runtime_matrix.py`, promoted
  two-CTA index plus adjacent multidim selector (`6 passed, 2 skipped`),
  half-row guard selector (`10 passed`), scrambled/MMAv5/x1 guard selector
  (`8 passed`), and `git diff --check`.
- 2026-04-17 00:05 UTC: extended the shared direct `ld/st`
  packet-footprint requirement to explicit n-sharded atom column footprints.
  Explicit `16x128b` x1 `f32/i32` load/store negatives now report that the
  requested atom owns a 4-dword column footprint while the descriptor view only
  exposes one materializable dword column, and `tensor_memory_descriptor.get_reg_layout`
  asks the backend for variant-specific unsupported reasons before falling back
  to generic view diagnostics. Support is unchanged: these rows still need a
  narrower atom (`32x32b`) or a true column-mask/footprint model. Validation:
  `make -j8`, py-compile for the touched Python files, focused x1 explicit
  negative plus scales negative selector (`9 passed, 1566 deselected`), x1
  positive selector (`8 passed, 1567 deselected`), and `git diff --check`.
- 2026-04-16 23:59 UTC: moved the scaled-MMAv5 repeated-`N=32`
  matrix-B scale-fragment boundary from a string-only helper to a typed
  `MMAv5ScaledRepeatedN32ScaleFragmentRequirement`. Verifier and lowering still
  consume the existing diagnostic wrapper, so support is unchanged, but future
  support work can now reason about accumulator encoding, selected N
  instruction size, CTA N columns, and repeated-N instruction count directly.
  Validation: `make -j8`, direct `invalid.mlir` verifier, focused
  scaled-MMAv5 repeated-N32 plus positive selector
  (`21 passed, 1554 deselected`), and `git diff --check`.
- 2026-04-16 23:56 UTC: started the adjacent Phase 4 `ld/st`
  packet-footprint cleanup. The direct load/store clean negatives for missing
  descriptor-view row anchors, lifted row-half origins, raw 4x256b refresh
  physical bitcasts, and M64 two-CTA scales broadcast row anchors now route
  through a shared `TMemLdStPacketFootprintRequirement` formatter. Support is
  unchanged, but these cases now live under one backend concept: direct
  `tcgen05.ld/st` needs materializable packet row anchors or an explicit
  packet/rematerialization/mask model. Validation: `make -j8`, direct
  `invalid.mlir` verifier, focused `ldst_4x256b_refresh` plus M64 scales
  clean-negative selector (`4 passed, 1571 deselected`), and
  `git diff --check`.
- 2026-04-16 23:52 UTC: made `TMemCopyDestinationMaskRequirement`
  operational in the copy scheduler diagnostics. Descriptor-row column splits
  and source-row selected-offset splits now use one shared helper to prove the
  same no-mask boundary: the public `tcgen05.copy` atom writes the full
  destination row/column footprint while the derived split only owns selected
  rows or columns. This remains a clean unsupported boundary, but the planner
  no longer carries separate family-shaped wording for the shared schedule
  consequence. Validation: `make -j8`, direct `invalid.mlir` verifier,
  focused scales/two-CTA `warpx2` selector (`15 passed, 1560 deselected`), and
  `git diff --check`.
- 2026-04-16 21:58 UTC: factored the common scheduler consequence of the
  scales descriptor-row split and no-scales source-row split into
  `TMemCopyDestinationMaskRequirement`. Descriptor-row column splits derive a
  column-mask requirement; source-row selected-offset splits derive a row-mask
  requirement. This is behavior-preserving but gives the next copy scheduler
  one carrier for "the public atom writes more destination rows/columns than
  this split is allowed to update." Validation: `make -j8`, direct
  `invalid.mlir` verifier, focused source-row/scales split selector
  (`15 passed, 1560 deselected`), and `git diff --check`.
- 2026-04-16 21:52 UTC: resumed Phase 2 copy-planner generalization after the
  runtime-matrix budget work. The two-CTA no-scales `warpx2::02_13` row-bit-5
  failure is now represented as a typed
  `TMemCopySourceRowSplitRequirement`: logical row bit 5 selects shared
  offset `1` for 32-row runs every 64 rows, instead of the affine row-stride
  offset `256` required by the current row projection. The clean negative now
  comes from the instruction-schedule layer with both the derived row-selected
  source-offset requirement and the existing direct-seed probe evidence.
  Validation: `make -j8`, direct `triton-opt --split-input-file
  --verify-diagnostics` for `invalid.mlir`, focused `warpx2` runtime selector
  (`32 passed, 1543 deselected`), and `git diff --check`.
- 2026-04-16 20:04 UTC: tightened the reduced TMEM runtime matrix again to
  make the default full-file 4-GPU split finish under the 10-minute iteration
  target with count-based `pytest-split`. Collection is now `1575` tests
  (down from the `5019` reduced checkpoint and the original `11133`
  exhaustive surface). The kept surface remains representative across all
  TMEM instruction families while sampling independent layout/view axes:
  `ld/st`, descriptor chains, subword/x1, split-N, tensor-memory scales,
  `ld.red`, no-scales copy, scales copy, MMA, and scaled-MMA. Fresh per-GPU
  cache validation: group 1 `293 passed, 101 skipped` in `577s`, group 2
  `385 passed, 9 skipped` in `318s`, group 3 `394 passed` in `266s`, and
  group 4 `393 passed` in `341s`.
- 2026-04-16 18:12 UTC: reduced the TMEM runtime-matrix collection from
  `11133` to `5019` tests to improve iteration speed without dropping any
  instruction family or known hard-frontier diagnostic. The new policy keeps
  full semantic coverage where a test owns modifier/opcode behavior, and uses
  representative matrices for broad layout/view Cartesian products: row/column
  permutations, M64 split-N permutations, MMA descriptor views, and scaled-MMA
  accumulator-add duplication. Validation: `make -j8`, py-compile,
  full-file collect, four-GPU reduced row/column selector (`101` passed),
  exact heavy representatives (`8` passed), and extra MMA representatives
  (`4` passed).
- 2026-04-16 16:32 UTC: investigated `GluonInline` in the slow compile
  profile. IR dumps showed the expensive inliner invocations were running on
  modules with one function and zero call ops, so the cost was generic MLIR
  inliner setup/call-graph overhead rather than useful inlining. The Gluon
  inline pass now returns early when the module has no `CallOpInterface`
  operations. Updated representative cold compiles: `ld/st 3.299s`,
  `ld.red 2.071s`, no-scales copy `0.904s`, scales copy `0.543s`, MMA
  `1.880s`, scaled-MMA+copy `1.087s`. Exact profile representatives passed
  (`6 passed`) and the previous focused guard set passed (`8 passed`).
- 2026-04-16 16:24 UTC: expanded the compile-time profile from the original
  `ld.red`/`ld/st` rows to representative `ld/st`, `ld.red`, no-scales copy,
  scales copy, MMA, and scaled-MMA cases. All cold compiles are now within the
  3-4s target; after a pure-outer-index row-plan cleanup the matrix is:
  `ld/st 3.582s`, `ld.red 2.220s`, no-scales copy `0.924s`, scales copy
  `0.547s`, MMA `1.933s`, and scaled-MMA+copy `1.112s`. The code change
  avoids recomputing standalone raw-query layouts in
  `getTMemLdStRowPlanForQuery(...)` and avoids support-query construction for
  pure outer descriptor indexes that preserve the trailing TMEM tile. Exact
  profile representatives passed (`6 passed`) and the previous nonpreexisting
  guard set passed (`8 passed`).
- 2026-04-16 15:59 UTC: continued compile-time optimization on the TMEM
  verifier paths added during this initiative. A second representative
  hotspot,
  `ldst_x1_subword_twocta_descriptor_chain_roundtrip[auto-i8-torch_dtype3-4]`,
  initially compiled at `4.67s` cold with `ttgir 2.225s` and `llir 1.980s`.
  The verifier now tries the exact memdesc type proof before constructing
  surrogate query-type stacks, and `getTMemLdStRowPlanForQuery(...)` no
  longer recomputes the same query/backing row plans through
  `preferBackingTMemLdStQueryTypes(...)`. Final fresh-cache timing for that
  row is `3.614s` compile (`ir_initialization 0.880s`, `ttgir 1.442s`,
  `llir 1.269s`, `ptx 0.004s`, `cubin 0.019s`) with warm execution still not
  the bottleneck. Focused x1 pytest and the previous 8-case guard set passed.
- 2026-04-16 14:34 UTC: interrupted backend-completeness work to profile the
  slow `ld.red` runtime-matrix tests. The representative
  `rowcol_rotate_reverse_n256_32x32b_splitn` cold compile is now compiler-bound
  but within target: direct compile listener reports total compile `2.252s`
  (`ir_initialization 0.638s`, `ttgir 0.605s`, `llir 0.707s`, `ptx 0.051s`,
  `cubin 0.250s`), cold synchronized call `2.256s`, assertion `0.150s`, and
  warm call `0.0002s`. Pytest exact-node overhead remains visible
  (`1 passed in 5.05s`, process wall `7.09s`). The optimization removes the
  expensive non-block MMAv5-family row-plan probe from arbitrary linear
  `ld.red` views, stops backing-plan walks after a maximal 128-row plan, and
  avoids duplicate generic/reduction verifier proofs. A focused guard set
  excluding one preexisting MMAv5 higher-rank red row passed `8` cases; the
  excluded MMAv5 row fails the same way on clean `origin/codex/tmem`.
- 2026-04-16 12:04 UTC: rehydrated the copy frontiers after the M64 backend
  checkpoint. Two-CTA no-scales `warpx2::02_13` still fails because row bit 32
  is a high source-column selector (`source offset 1`) that current row
  projection cannot carry through a cta-group::2 descriptor schedule. The
  scales descriptor-view copy still fails because source column bit 2 selects
  descriptor row `+32` for 4-column runs inside one 16-column
  `warpx4.32x128b` instruction. These are now documented as
  source-message / destination-footprint scheduler gaps, not frontend fallback
  or shape-gate cleanup.
- 2026-04-16 12:00 UTC: the remaining explicit M64
  requested-variant `16x32bx2` Python canonical fallback has been deleted.
  The same C++ raw-query recognizer now covers both `32x32b_splitn` and
  `16x32bx2`, so row/column-permuted M64 split-N register layout selection is
  backend-owned for auto/default and explicit split-N requests. Validation:
  `make -j8`, py-compile, traced explicit row confirming
  `canonicalM64SplitNRawQuery atomName=16x32bx2 recognized`,
  `splitn_rowcol_permuted` (`518 passed`), full `ld_red_m64` (`73 passed`),
  and `git diff --check`.
- 2026-04-16 11:54 UTC: the handle-aware M64 `32x32b_splitn`
  auto-layout fallback is now backend-owned for simple noncanonical split-N
  physical images. `compute_tmem_reg_layout_from_memdesc(...)` recognizes the
  exact rank-2 M64 32-bit non-scales raw query with one zero row basis plus
  permuted power-of-two row/column bases and returns the canonical split-N
  register layout; `_try_handle_aware_m64_splitn_auto_layout(...)` no longer
  falls back to the Python canonical/type-only stack. Validation: `make -j8`,
  py-compile, the exact old failure set (`15 passed`), full `ld_red_m64`
  (`73 passed`), and `splitn_rowcol_permuted` (`518 passed`). A broader
  combined `splitn or ld_red_m64` sweep was interrupted after a long pass-only
  stretch, so do not record it as passed. A probe deleting the separate
  explicit-`16x32bx2` Python canonical fallback failed many row/column
  permutation rows and was restored; that remains the next M64 split-N cleanup
  target.
- 2026-04-16 11:07 UTC: a deletion probe for the remaining handle-aware M64
  split-N Python fallback was rejected and restored. Returning only
  `compute_tmem_reg_layout_from_memdesc(..., "32x32b_splitn")` broke
  row/column-permuted M64 auto split-N opcode selection and produced wrong
  default/explicit M64 reduction output for row-reverse and
  row-rotate/col-even-odd cases. After restoring the fallback stack,
  `make -j8` and the exact affected selector passed (`15 passed`). The next
  real cleanup needs backend physical-query/packet-equivalence for these
  noncanonical M64 split-N views, not fallback deletion.
- 2026-04-16 10:50 UTC: a temporary env-controlled direct-seed probe for
  two-CTA no-scales `warpx2::02_13` was completed and removed. Source
  offsets `0,16,32,48,64` with `dword_delta=0` all emitted the
  `cta_group::2.warpx2::02_13` opcode but duplicated one source-column pair
  into both destination column pairs; aligned nonzero deltas
  `4,8,12,16` wrote zeros, and source offsets `80+` caused unspecified launch
  failures. After removing the probe hook, `make -j8` and the exact clean
  negative passed. Treat this as confirmation that a direct-seed parameter
  tweak is not the missing two-CTA schedule.
- 2026-04-16 10:43 UTC: descriptor-level direct `ld/st` rejection for the
  `tcgen05.copy.4x256b` refresh image is now backend-owned. The shared
  `isUnsupportedDirectTMemLdStDescriptorView(...)` helper reports the root
  refresh-layout reason, the `TMEMLoadOp`/`TMEMStoreOp` verifier no longer has
  a duplicate branch, and the Python `tensor_memory_descriptor` load/store/
  reduction/get-reg-layout methods no longer prefilter this descriptor shape.
  Type-only layout queries still keep their Python guard because they have no
  memdesc handle. Validation: `make -j8`, full `4x256b` selector (`5 passed`),
  py-compile, and `git diff --check`.
- 2026-04-16 10:38 UTC: the explicit-`32x32b` M64 reduction fallback has been
  moved into the C++ memdesc reduction bridge. When the raw exact query is an
  M64 f32 non-scales view with noncanonical row bases, the bridge now derives
  the canonical M64 split-N register layout, validates it against the exact
  reduction message predicate, and returns it to `_load_red`; the Python
  explicit-layout helper no longer falls back to the frontend split-N planner.
  A canonical-row guard keeps column-only M64 permutations on the existing
  backend path so packet offsets stay stable. Validation: `make -j8`,
  focused row/column-permuted M64 selector (`9 passed`), full `ld_red_m64`
  selector (`73 passed`), py-compile, and `git diff --check`.
- 2026-04-16 10:12 UTC: the explicit-`32x32b` M64 reduction fallback is still
  required. A follow-up removal probe failed six row/column-permuted M64
  default/explicit rows with scalar `.x1` reduction packets, so the fallback
  was restored. The only kept code cleanup is an unused `_load_red` local
  removal. Validation after restore: `make -j8` and full `ld_red_m64` selector
  (`73 passed`).
- 2026-04-16 10:08 UTC: default M64 `load_min/load_max(layout=None)` no
  longer retries the Python split-N fallback immediately after the backend
  reduction-layout query. With the no-override rule owned by
  `getTmemLoadReductionLayout(...)`, the default path first asks the backend
  and then falls through to ordinary layout selection. The explicit-`32x32b`
  last-resort fallback remains live for noncanonical M64 rows. Validation:
  `make -j8`, full `ld_red_m64` selector (`73 passed`), targeted unsafe
  selector (`46 passed`), and `git diff --check`.
- 2026-04-16 09:59 UTC: the reduction-layout helper now owns the safe
  no-override rule for direct-compatible `32x32b` reductions. C++
  `getTmemLoadReductionLayout(...)` returns no override when direct
  `32x32b` is already reduction-compatible, so Python no longer repeats that
  packet-order guard. This preserves the M64 backend rescue while keeping the
  known non-M64 row/column permutation packet-order cases on their direct
  path. Validation: `make -j8`, targeted unsafe selector (`46 passed`), full
  `ld_red_m64` selector (`73 passed`), and `git diff --check`.
- 2026-04-16 09:49 UTC: repeated-`N=32` direct block-scaled MMAv5 was
  re-probed by temporarily removing both verifier and lowering guards. The
  mxfp8/mxfp8 `M=N=128,K=128,tile_n=32` representative compiled and emitted
  16 `tcgen05.mma...block_scale` instructions, but runtime output was wrong
  (`12274 / 16384` mismatched elements, max absolute difference about
  `603.866`). Restoring the guard and rerunning the full repeated-N32
  clean-negative selector passed (`10 passed`). Keep this as a true
  matrix-B scale-fragment addressing boundary unless a future design adds a
  sub-64-column B-scale fragment schedule.
- 2026-04-16 09:43 UTC: two no-code scales-copy support probes were
  completed and reverted. Relaxing only the non-contiguous source-column
  preflight did not promote the scales `warpx2`-like probe family: one case
  immediately became a later descriptor-row split/destination-mask failure and
  another reached descriptor synthesis with no representable MMAv5 descriptor.
  Allowing transposed MMAv5 descriptor orientation for `warpx4.32x128b`,
  both alone and combined with that relaxation, also produced no support path.
  Keep the current preflight strict; support still needs a real source-message
  / destination-mask / source-format model, not descriptor-orientation or
  column-preflight relaxation.
- 2026-04-16 09:37 UTC: the two-CTA tensor-memory-scales descriptor-view
  `M=64` direct `ld/st` boundary now reports a specific row-anchor diagnostic
  instead of the generic exact-projection fallback. A temporary recognizer
  expansion showed that the raw view carries one 32-row warp anchor as
  broadcast/support state; current scales `tcgen05.ld/st` lowering still
  requires materializable row anchors 32 and 64. Validation: `make -j8`,
  exact new clean-negative row (`1 passed`), adjacent CGA scales
  descriptor-view selector (`46 passed`), py-compile, and `git diff --check`.
- 2026-04-16 09:18 UTC: default `load_min/load_max(layout=None)` now asks
  the backend reduction-layout helper first for f32 non-scales TMEM, but the
  helper is conservative: if direct `32x32b` is already reduction-compatible
  it returns `None` and lets the existing direct path preserve exact packet
  order. This moves the M64 scalarization rescue into a globally safe backend
  query without reintroducing the rejected non-M64 row/column-permutation
  broad-route bugs. Validation: py-compile, `make -j8`, targeted unsafe
  selector (`46 passed`), full M64 selector (`73 passed`), and `git diff
  --check`.
- 2026-04-16 09:07 UTC: the reduction message-compatibility predicate is now
  centralized in `TensorMemoryUtils` as
  `isTMemLdStReductionCompatible(...)` / `getTMemLdStReductionRepeats(...)`
  and is shared by the reduction layout helper, verifier, and Python bridge.
  This is behavior-preserving, but it removes duplicated `.ld.red` legality
  arithmetic before the next selector-equivalence work. A forced-helper probe
  confirmed the rejected broad route is specifically a row-permutation
  semantic-equivalence problem: one non-M64 row-permuted case emitted legal
  `32x32b.x128` packets but returned permuted rows. Validation: `make -j8`,
  focused reduction selector (`121 passed`), and `git diff --check`.
- 2026-04-16 08:46 UTC: the backend reduction-layout helper now validates
  actual `tcgen05.ld.red` message legality (`packed`, reduction repeats at
  least `.x2`) instead of treating `I32x32b` atom spelling as sufficient, and
  it may return a `16x32bx2` layout for M64 split-N reductions. The frontend
  uses this helper only for the known M64 direct-`32x32b` scalarization
  boundary. A broad probe that routed all default reductions through the
  helper found wrong output for non-M64 row-permuted roots, so do not make it
  the global default selector until the backend can prove full physical-query
  equivalence, not only reduction-message legality. Validation: py-compile,
  `make -j8`, focused post-fix selector (`121 passed`), and `git diff
  --check`.
- 2026-04-16 07:58 UTC: the M64 explicit-`32x32b` reduction override now
  asks the backend `compute_tmem_reduce_reg_layout_from_memdesc(...)` helper
  before using the frontend split-N fallback, and the positive coverage spans
  the existing row/column-permuted default M64 reduction cases. Validation:
  py-compile, `make -j8`, M64 `ld.red` selector (`73 passed`), and
  `git diff --check`.
- 2026-04-16 07:56 UTC: M64 row-permuted `tcgen05.ld.red` with an explicit
  `instr_variant="32x32b"` is now positive instead of falling into the
  scalar `.x1` reduction guard. The reduction API keeps normal
  `get_reg_layout("32x32b")` behavior unchanged for load/store, but when a
  provided layout exactly matches the direct explicit `32x32b` M64
  row-permuted layout, it asks the existing handle-aware split-N planner for
  the reduction-compatible layout and uses that for `load_min`/`load_max`.
  Validation: py-compile, `make -j8`, promoted row-permuted node (`1
  passed`), full M64 `ld.red` selector (`71 passed`), and `git diff --check`.
- 2026-04-16 07:51 UTC: the explicit two-CTA scales descriptor-view
  `16x32bx2` frontier is now pinned as a precise clean unsupported boundary
  instead of a generic frontend fallback. A temporary support probe showed the
  transposed scales view has an exact physical query that is directly
  realizable by `32x32b` and by the wider n-sharded scale atoms, but the
  native `16x32bx2` half-tile split appears as register/message repetition
  rather than the lane-selected second-half offset required by the ISA atom.
  The frontend now asks the backend for a handle-aware requested-variant
  reason before raising the split-N fallback error, and the runtime matrix
  pins this row as a clean diagnostic. Validation: `make -j8`, py-compile,
  clean-negative node (`1 passed`), expanded positive CGA descriptor-view
  selector (`44 passed`), adjacent non-CGA descriptor-view node (`3 passed`),
  and `git diff --check`.
- 2026-04-16 07:32 UTC: two-CTA tensor-memory-scales descriptor-view
  `tcgen05.ld/st` now supports explicit `16x64b`, `16x128b`, and `16x256b`
  view register layouts. The backend constructs the pre-packed physical
  packet layout for the requested atom and lifts each physical TMEM basis
  through the exact descriptor-view linear query, then keeps the result behind
  the existing `computeTMemLdStEncodingInfo(...)` validation. The expanded CGA
  runtime matrix covers all realizable power-of-two `N=4..128` rows for
  `M in {128,256}` and checks the two-packet offsets (`0`, `1048576`) for the
  explicit atom families. Validation: `make -j8`, py-compile, expanded CGA
  selector (`44 passed`), adjacent non-CGA descriptor-view node (`3 passed`),
  direct `invalid.mlir` verifier, and `git diff --check`.
- 2026-04-16 07:15 UTC: two-CTA tensor-memory-scales descriptor-view
  `tcgen05.ld/st` now uses the exact-query recognizer over the full validated
  power-of-two `N=4..128` range for `M in {128,256}`, instead of the stale
  hard-coded `N in {32,64}` gate. The recognizer is still validated by the
  existing register-layout lowering proof before it is selected. Newly
  promoted runtime rows cover both narrow and wide `N` values and assert the
  expected `16x32bx2` root plus `32x32b` view packet shapes. Validation:
  `make -j8`, py-compile, focused descriptor-view scales runtime selector
  (`16 passed`), direct `invalid.mlir` verifier, and `git diff --check`.
- 2026-04-16 07:05 UTC: a no-code packed-lane `warpx2::01_23` probe was
  completed and reverted. Treating subword copy footprints as physical dword
  columns and allowing the 32-row core descriptor to cover the packed-lane row
  expansion compiled f16/bf16/i16 single-CTA rows, but runtime output copied
  only source columns 0/1 into every destination column pair. The first f16
  row pattern was `[0,128,1,129]`, `[4,132,5,133]`, ... instead of the intended
  mapping that also uses source columns 2/3, leaving `504 / 512` mismatches.
  The next valid `warpx2` subword support step is a lane-aware descriptor /
  instruction schedule, not another footprint-only relaxation.
- 2026-04-16 06:54 UTC: a no-code identity `32x32` descriptor-view
  row-origin probe was completed and reverted. With the row-anchor guard
  lifted, the high-quadrant identity view computes the right raw origin
  (`64<<16 | 64`), but the lowered subview pointer has already advanced by the
  folded support-frame offset `66`. The resulting `16x32bx2.x32` direct path
  updates rows `0..31`, columns `64..95`. Moving the residual `0x3ffffe`
  origin from dynamic base adjustment to the packet immediate changes the
  emitted immediate but not the wrong rows. Treat this as a real
  row-origin/packet-footprint rematerialization gap, not a local base-offset
  bookkeeping issue.
- 2026-04-16 06:45 UTC: a no-code scales source-format probe was completed
  and reverted. Forcing `.b8x16.b6x16_p32` and `.b8x16.b4x16_p64` onto the
  int8 scales `warpx4.32x128b` descriptor-view/shared-subslice frontier did
  not produce a support path: even with the instruction-column preflight
  bypassed for those experimental messages, descriptor synthesis still had no
  representable MMAv5 shared descriptor for the row-split source image. The
  remaining failure is still descriptor row `+32` for 4-column destination
  runs every 8 columns inside one 16-column public copy atom, with no
  destination column mask. Validation after removing the temporary source
  edit: `make -j8` and the scales shared-subslice clean-negative selector
  (`2 passed`).
- 2026-04-16 06:37 UTC: live diagnostics now carry the 06:30 subword
  `warpx2::01_23` finding. Descriptor-synthesis failures append a generic
  source-storage note when a subword copy instruction spans more logical
  element columns than the source linear view exposes, stating that packed
  source-storage or descriptor semantic-equivalence modeling is required and
  descriptor footprint coverage alone is not a correctness proof. Validation:
  `make -j8`, py-compile, subword warpx2 selector (`14 passed`), neighboring
  non-subword warpx2 selector (`64 passed`), and `git diff --check`.
- 2026-04-16 06:30 UTC: a no-code `warpx2::01_23` subword support probe was
  completed and reverted. Making descriptor selection skip candidates whose
  coordinate image could not cover the instruction footprint compiled six
  formerly negative f16/bf16/i16 rows, but the first f16 runtime oracle was
  wrong (`504 / 512` mismatches). The selected representable descriptor used
  row unit offset 8 and produced source rows `0,4,8,...`, while the correct
  subword mapping requires `0,2,4,...`. Conclusion: footprint coverage and
  descriptor representability are not enough; support needs semantic
  equivalence between the selected MMAv5 descriptor and the copy atom's
  internal row/column mapping, or a new source-storage schedule. The source
  edit was removed; validation after cleanup: `make -j8` and subword warpx2
  selector (`14 passed`).
- 2026-04-16 06:24 UTC: no-scales `tcgen05.copy.warpx2` subword rows now
  route through the shared copy planner instead of stopping at the old
  32-bit-shared-element runtime guard. A bounded subword-only
  `warpx2::01_23` 64-row descriptor candidate was added, and the clean
  negatives now assert planner-layer reasons: source-footprint bounds,
  non-representable MMAv5 shared descriptors, direct-seed source-layout
  encoding, or the known `warpx2::02_13` row-bit-5 projection failure. This
  is still unsupported, but the remaining gap is now attached to
  packed/subword source-storage descriptor scheduling rather than a coarse
  family guard. Validation: `make -j8`, subword warpx2 selector (`14
  passed`), neighboring non-subword warpx2 selector (`64 passed`),
  py-compile, and `git diff --check`.
- 2026-04-16 06:15 UTC: descriptor-view `get_reg_layout()` failures now
  surface the backend's direct `ld/st` unsupported reason instead of dropping
  it behind the generic frontend fallback. The row-anchor diagnostic was
  sharpened to name the packet-footprint/rematerialization boundary exposed by
  the 06:11 identity probe: public packets over-cover the available support
  image unless the row origin is decomposed into packet base, row anchors, and
  per-message offsets, or a read/modify/write footprint model is introduced.
  Validation: `make -j8`, py-compile, exact identity clean-negative row, and
  neighboring half-row clean-negative selector (`31 passed`).
- 2026-04-16 06:11 UTC: a no-code exact `32x32` identity descriptor-view
  `ld/st` high-quadrant probe refined the direct-packet boundary. Exact
  value-level view arithmetic preserved the final `origin=(64,64)` and the
  expected `32x32` logical layout after `memdesc_subslice`/`memdesc_reshape`.
  Three temporary support-plan attempts were then removed because they
  compiled but produced wrong runtime output: rebuilding support from the
  final descriptor type/root query selected `16x32bx2.x32.b32` and updated
  row bands outside the logical quadrant, while forcing a logical 32-row row
  plan selected scalar `32x32b.x1.b32` packets and over-updated the whole row
  range. Conclusion: this is not a stale type-inference guard; direct support
  needs a semantics-preserving packet/rematerialization or read/modify/write
  model for the exact physical footprint, or it must remain a clean
  ISA-impossible diagnostic.
- 2026-04-16 05:42 UTC: a no-code lifted row-half `ld/st` support probe
  refined the 05:23 boundary. Temporarily preserving the outer leading-buffer
  column offset while subtracting only the already-applied row component made
  the selected half land in the right outer tile, but the direct packet
  footprint then updated both row halves. Forcing a logical 64-row row plan
  through the support image did not change the public `tcgen05.ld/st`
  footprint enough to prevent the complementary half from being touched. The
  source tree was restored clean; future support needs a real packet/warp
  rematerialization or predicate model, not just base-offset masking or a
  query-row-plan override.
- 2026-04-16 05:30 UTC: dense copy row-order clean negatives now report the
  first row or row-repetition basis that breaks ascending physical row order.
  This keeps the unsupported decision at the instruction-schedule layer while
  exposing the exact source-row rematerialization input a future row-mask /
  row-partition planner would need.
- 2026-04-16 05:27 UTC: the lifted row-half direct `ld/st` clean-negative
  diagnostic is now centralized in the backend and records the actual planner
  boundary from the latest probe. The message states that support-query layout
  selection alone is insufficient; correct direct lowering needs the translated
  row origin decomposed into packet base, row anchors, and per-message offsets.
  The runtime matrix now pins that sharper diagnostic for single-CTA and
  two-CTA lifted half-row views.
- 2026-04-16 05:23 UTC: a no-code lifted row-half `ld/st` probe was
  completed and reverted. Removing the blanket row-half verifier guard lets
  the canonical single-CTA half-row view compile, but runtime output is wrong:
  the selected packets do not update the intended second half. A follow-up
  attempt to suppress the support-query row-origin base offset faults with a
  CUDA misaligned-address error. Conclusion: the existing support-query model
  can find register layouts for the view, but lowering still needs a real
  packet-address / row-origin decomposition for lifted row-half descriptors;
  this is not a safe verifier relaxation or type-fallback issue. The source
  tree and build were restored clean after the probe with `make -j8`.
- 2026-04-16 05:13 UTC: the 4x256b refresh raw physical bitcast readback
  probe is now recorded as an explicit direct `tcgen05.ld/st` unsupported
  contract in both frontend and backend paths. The frontend recognizes the
  original refresh-shaped `4x8xf32` TensorMemoryLinearLayout and its raw
  `32x4xi8` physical bitcast; the backend descriptor-view verifier mirrors
  the raw bitcast signature before generic support-query fallback. The
  diagnostic now states the actual ISA boundary: public `tcgen05.ld/st`
  packets read whole row footprints and expose no lane mask for the sparse
  refresh rows, so the supported access path remains `tcgen05_copy` from
  shared memory or a directly supported 128-row physical layout. Validation:
  `make -j8`, direct `invalid.mlir` verifier, py-compile of touched Python
  files, focused runtime-matrix `4x256b` selector (`5 passed`), and
  `git diff --check`.
- 2026-04-16 05:04 UTC: two no-code probes were completed from the
  `1d07b3bd5` checkpoint and the source tree was restored clean. First,
  temporarily removing the scaled-MMAv5 repeated-N32 guard and trying grouped
  matrix-B scale fragments still produced runtime mismatch (`50%` for the
  mxfp8/mxfp8 `N=128,K=128` row). This confirms the current repeated-N32
  scaled-MMA negative is a real public scale-fragment boundary: the B scale
  layout exposes 64-column fragments and the instruction descriptor has only
  the K sub-column selector, not a low/high N32-half selector. Second, a
  runtime probe copied through `tcgen05.cp.4x256b`, bitcast the refresh image
  to the raw `32x4xi8` physical view, and failed at `get_reg_layout(auto)`.
  The bitcast layout is now inferred correctly, but direct readback still
  needs a sparse physical gather/register-layout plan for the refresh image.
- 2026-04-16 04:53 UTC: physical TMEM bitcast reinterpret inference now
  preserves source physical row/column coordinate units when the source view is
  non-injective only because of inactive zero support bases, and scales TMEM
  column coordinates by source/destination element bitwidth when bitcasting
  across element sizes. The concrete 4x256b refresh probe now infers the
  correct raw `32x4xi8` physical image with row zero bases and byte-lane column
  bases; direct readback remains a separate sparse-lane gather boundary rather
  than a bitcast arithmetic bug. Validation: `make -j8`, focused frontend
  bitcast regression, neighboring frontend bitcast selector (`4 passed`),
  existing GPU physical-bitcast selector (`3 passed`), 4x256b runtime-matrix
  selector (`4 passed`), direct `invalid.mlir` verifier, py-compile, and
  `git diff --check`.
- 2026-04-16 04:38 UTC: broadcasted TMEM load expansion now uses the exact
  inverse of the `ColumnAction` that removed broadcasted register columns,
  instead of inferring the reload count from packed message repetitions. This
  makes subword load/store lowering symmetric with store reduction after f16
  unpacking and promotes the parked no-scales two-CTA subword
  column-slice/subslice copy rows. Validation: `make -j8`,
  two-CTA subslice copy (`36 passed`), dense two-CTA copy (`91 passed`),
  indexed two-CTA copy (`35 passed`), single-CTA subword copy (`8 passed`),
  subword `ld/st` selector (`76 passed`), LinearLayout unit tests (`70
  passed`), direct `invalid.mlir` verifier, py-compile, and
  `git diff --check`.
- 2026-04-16 04:27 UTC: no-scales two-CTA subword dense and leading-indexed
  descriptor-view copy coverage is now enabled for linear layouts using the
  actual shared element bitwidth. Leading TMEM index query inference now has an
  exact projection path for layouts whose explicit outer dimension is consumed
  by `MemDescIndexOpConversion`, and pure outer index recognition accepts the
  corresponding source layout rank delta. Validation: `make -j8`, py-compile,
  two-CTA dense subword copy (`91 passed`), two-CTA indexed subword copy
  (`35 passed`), existing two-CTA subslice copy (`12 passed`), direct
  `invalid.mlir` verifier, and `git diff --check`. Parked next gap: subword
  two-CTA column-slice/subslice copy reaches the copy atom but readback still
  fails in broadcasted `tmem_load` lowering, so the next backend slice is the
  exact load/support-query model for that descriptor view.
- 2026-04-16 04:09 UTC: copy source-row projection failures now have a typed
  `TMemCopySourceRowProjectionFailure` carrier, matching the existing typed
  instruction-column failure path. The two-CTA `warpx2::02_13` known-gap
  diagnostic now consumes the failed row-bit projection facts instead of
  recomputing them from the conversion layout, keeping the boundary attached
  to the planner layer that proved it. Validation: `make -j8`, affected
  `cp_no_scales_warpx2_02_13_twocta` rows (`14 passed`), neighboring
  `cp_no_scales_warpx2` rows excluding that bucket (`64 passed`), and direct
  `invalid.mlir` verifier.
- 2026-04-16 03:45 UTC: `tcgen05.copy` direct-seed message plans now fail
  when the immediate seed descriptor cannot be synthesized instead of falling
  through to descriptor-loader lowering while still carrying direct-seed source
  semantics. A temporary `warpx2` shared-layout preflight lift showed why the
  canonical shared-layout guard remains a real source-rematerialization
  boundary: some noncanonical shared layouts fail descriptor-loader footprint
  bounds, while others compile and copy wrong logical source rows/columns.
  Validation: `make -j8`, affected `warpx2` rows (`6 passed`), and full
  `cp_no_scales_warpx2` selector (`78 passed`).
- 2026-04-16 02:46 UTC: dense `tcgen05.copy` destination scheduling now uses
  exact physical tile coordinates for every dense copy tile instead of a
  128-byte macro-tile heuristic. This fixes a real wrong-code hole for
  no-scales tile-permuted column layouts with 4-column tiles: the planner now
  emits physical tile offsets and uses the narrower `128x128b` fallback where
  needed. Sub-instruction tile permutations (`tile_n` 1/2) remain clean
  unsupported because they break contiguity inside the instruction footprint,
  and row-basis permutations remain a source-row rematerialization gap.
  Validation: `make -j8`, tile-permutation selector (`18 passed`),
  neighboring copy clean-negative/positive selector (`19 passed`), direct
  `invalid.mlir` verifier, py-compile, and `git diff --check`.
- 2026-04-16 02:35 UTC: `MemDescSubsliceOp::fold` now recomputes the inferred
  return type after combining chained subslice offsets and updates the folded
  op result type to match the original source view. This fixes the
  active-view type mismatch exposed by the exact canonical `32x32`
  descriptor-view probe without pretending that identity `32x32` direct
  `tcgen05.ld/st` is supported: the identity row remains a clean row-window /
  rematerialization boundary, while the mixed descriptor-view positive stays
  green. Validation: `make -j8`, focused descriptor-view slice selector
  (`3 passed`), and `git diff --check`.
- 2026-04-16 01:59 UTC: packed-lane copy failures now produce a typed
  `TMemCopyPackedLaneRequirement` and a dedicated instruction-schedule proof.
  The planner records the low lane bits, lanes per 32-bit shared-memory word,
  and contiguous physical dword-column projection, then rejects dword-only
  descriptor/tile schedules because dropping lane bits aliases packed source
  lanes and copies only one lane group. Legacy subword `TensorMemoryLayout`
  copy remains a clean source-storage boundary; dense subword copy through an
  unpacked `TensorMemoryLinearLayout` remains the supported path. Validation:
  `make -j8`, the exact legacy subword clean-negative row (`4 passed`),
  py-compile, and `git diff --check`.
- 2026-04-16 01:41 UTC: moved the two-CTA no-scales
  `warpx2::02_13` clean-negative from a shape-only early schedule-gap guard
  into the source-row projection planner. A bounded bypass confirmed the
  underlying failure: logical row bit 5 maps to a one-dword source offset, not
  an affine multiple of the 8-row source stride. The final unsupported
  diagnostic is unchanged for users, but it now fires only after the planner
  proves that exact row-projection mismatch. Validation: `make -j8`, direct
  `invalid.mlir` verifier, py-compile, `git diff --check`, exact two-CTA
  `02_13` candidate clean-negative (`2 passed`), and the focused
  `cp_no_scales_warpx2` selector (`78 passed`).
- 2026-04-16 01:29 UTC: fixed the scales descriptor-view `ld/st` regression
  caused by the 00:57 reshape cleanup. The trailing zero-basis cardinality
  trim was removed: row/column/block zero bases are semantic TMEM
  broadcast/support axes, not inactive shape padding, and direct support
  planning must decide whether an operation can realize them. Keeping the
  reshaped support bases restores all single-CTA and two-CTA scales
  descriptor-view positives and keeps the exact `32x32` identity descriptor
  row at its later clean direct-ISA boundary. Validation: `make -j8`,
  scales descriptor-view `ld/st` rows (`6 passed`), focused descriptor-view
  sweep (`38 passed, 1 skipped`), scales descriptor-view copy clean-negative
  (`1 passed`), direct `invalid.mlir` verifier, py-compile, and
  `git diff --check`.
- 2026-04-16 01:04 UTC: re-probed the exact canonical `32x32` descriptor-view
  `ld/st` boundary by temporarily adding a 32-row query plan. The probe was
  removed: it got past the previous row-plan gate but exposed an intermediate
  `memdesc_subslice` verifier mismatch between rank-4 active-view inference
  and the rank-3 squeezed type already materialized by IR construction. This
  confirms the current row is still a clean unsupported direct-ISA boundary,
  not a row-plan guard to lift. Future support needs both consistent active
  subview type inference and a proven packet/row-anchor rematerialization
  model. Validation after cleanup: `make -j8`, exact identity descriptor row
  (`1 passed`), and a clean worktree before this docs checkpoint.
- 2026-04-16 00:57 UTC: descriptor-view reshape inference now strips leading
  unit dimensions by layout rank instead of requiring the active view to cover
  the full backing image, and trims only trailing zero bases when the reshaped
  logical input cardinality exceeds the active element count. The identity
  multidim-slice `ld/st` row now reaches the real direct-ISA boundary
  (`32x32` exact subview cannot materialize the required row anchors) instead
  of failing `memdesc_reshape` type inference. Validation: `make -j8`,
  focused descriptor-view sweep (`38 passed, 1 skipped`), identity exact row,
  direct `invalid.mlir` verifier, py-compile, and `git diff --check`.
- 2026-04-16 00:39 UTC: probed the first support path using the packed-lane
  carrier, then removed the temporary source edits. The attempted schedule
  dropped packed lane bases from the descriptor view, used physical source
  widths, shifted source columns, and tried a raw 32-bit MMAShared descriptor.
  It got past the packed-lane preflight but failed descriptor synthesis for
  the actual f16 shared layout: the raw-dword projection is not
  MMAShared-representable. The source tree is restored to the pushed
  `45946a3a2` carrier state. Next packed-lane work needs a descriptor/storage
  model that can preserve the packed source layout, not only a raw dword
  descriptor projection.
- 2026-04-16 00:32 UTC: packed subword copy failures now carry a structured
  physical dword-column projection. Legacy subword `TensorMemoryLayout`
  copies still remain clean unsupported, but the instruction-column planner
  proves when the hidden lane bits are followed by a contiguous physical
  dword-column stream (`physicalColumns=8` for the dense f16/i16/i8 rows).
  This records the exact next scheduler requirement instead of collapsing the
  case to a generic zero-offset column-bit failure. Validation: `make -j8`,
  direct `invalid.mlir` verifier, py-compile, `git diff --check`, the legacy
  subword clean-negative pytest row (`4 passed`), and a debug probe confirming
  `packedLaneBits=1 physicalColumns=8`.
- 2026-04-16 00:25 UTC: direct-seed `tcgen05.copy` messages now get explicit
  source bit-range bounds instead of being an unchecked skip. The planner
  checks rank, direct-seed row representability, 128-bit source-offset
  alignment, and source-tile range. Validation: `make -j8`, direct
  `invalid.mlir` verifier, py-compile, `git diff --check`, and `warpx2::02_13`
  selector (`16 passed`).
- 2026-04-16 00:22 UTC: removed the stale `LogicalSharedTile`
  source-coordinate mode from the copy scheduler. The source-footprint model
  now has only descriptor-loader coordinates for selected MMAShared
  descriptors and direct-seed immediates for the immediate descriptor path.
  Validation: `make -j8`, direct `invalid.mlir` verifier, py-compile,
  `git diff --check`, focused copy selector (`9 passed`). Broad copy
  validation for the preceding descriptor-coordinate fix is complete:
  split groups 1-4 passed with group 1 carrying the repaired 256-row bucket.
- 2026-04-16 00:14 UTC: descriptor-backed copy source footprints now use
  descriptor-loader bounds consistently, including dense no-scales copies. A
  broad copy shard showed that the previous dense/logical-source split falsely
  rejected 256-row positives: exact query selection can encode the high row
  selector in the copy column dimension, so scheduled source columns past the
  logical `N` extent are valid descriptor coordinates. Direct-seed immediates
  still skip source bounds. Validation: `make -j8`, direct `invalid.mlir`
  verifier, py-compile, `git diff --check`, minimal 256x16 repro, and repaired
  group-1 copy shard (`171 passed, 10 skipped, 10807 deselected`).
- 2026-04-16 00:04 UTC: refreshed stale `test/TritonNvidiaGPU/invalid.mlir`
  verifier expectations for the newer copy diagnostics: the public
  `tcgen05.copy` no-mask wording for scales descriptor-row-stride failures and
  the expanded two-CTA `warpx2::02_13` schedule-gap evidence. Direct verifier
  validation with `triton-opt --split-input-file --verify-diagnostics` passes;
  `lit` itself is not installed on PATH in this shell.
- 2026-04-16 00:02 UTC: source-footprint support now proves descriptor-loader
  coordinates against the selected descriptor layout's row/column extent, while
  logical shared-tile coordinates are still checked against the shared tensor
  shape and direct-seed immediates are skipped. This makes scales `warpx4`
  participate in source-footprint legality without misclassifying its second
  descriptor tile as an out-of-bounds logical source column. Validation:
  `make -j8`, py-compile of `test_tmem_runtime_matrix.py`,
  `git diff --check`, and 15 targeted rows covering dense `4x256b`,
  single/two-CTA `warpx2`, scales `warpx4`, scales clean negatives, and legacy
  packed subword diagnostics.
- 2026-04-15 23:59 UTC: scheduled copy source footprints now carry an explicit
  coordinate space: logical shared tile, descriptor-loader coordinates, or
  direct-seed immediate. Dense source-bounds checking now keys off that carrier
  instead of assuming every `instruction.source` row/column is a logical
  tensor coordinate. This is the abstraction needed before descriptor-space
  bounds and scales split scheduling can be made general. Validation:
  `make -j8`, py-compile of `test_tmem_runtime_matrix.py`,
  `git diff --check`, and 15 targeted rows covering dense `4x256b`,
  single/two-CTA `warpx2`, scales `warpx4`, scales clean negatives, and legacy
  packed subword diagnostics.
- 2026-04-15 23:57 UTC: dense copy schedules now get a source-footprint bounds
  proof after instruction scheduling. The check rejects descriptor-loaded dense
  schedules that would read outside the rank-2 shared source tile, while
  direct-seed and non-dense/multicast families are left to a future
  descriptor-space proof because their recorded loader coordinates are not
  always plain logical shared-tile coordinates. This distinction was exposed by
  a `warpx4` scales positive: its second loader coordinate is descriptor-space,
  not an out-of-bounds logical source column. Validation: `make -j8`,
  py-compile of `test_tmem_runtime_matrix.py`, `git diff --check`, and 13
  targeted rows covering dense `4x256b`, two-CTA `warpx2`, scales `warpx4`,
  scales subslice/descriptor-view negatives, and legacy packed subword
  diagnostics.
- 2026-04-15 23:51 UTC: `4x256b` refresh copy now reports the effective
  half-width instruction footprint used by the proven two-message refresh
  schedule. Source and destination footprints for `4x256b` messages cover the
  low or high 4-column half instead of an 8-column rectangle, so the generic
  destination-overlap proof can run for `Dense4x256b` without a family-level
  exemption. Validation: `make -j8`, py-compile of
  `test_tmem_runtime_matrix.py`, `git diff --check`, and 7 targeted copy rows
  covering single/two-CTA `4x256b` refresh, ordinary contiguous `4x256b`
  clean-negative, neighboring two-CTA `warpx2`, scales `warpx4`, and the
  scales descriptor-view clean negative.
- 2026-04-15 23:47 UTC: the copy planner now runs a planner-side
  destination-footprint overlap proof after the emitted instruction stream is
  built. Ordinary copy families are rejected if two scheduled instructions
  write overlapping rectangular physical destination footprints, producing an
  `instruction schedule` diagnostic that asks for a non-overlapping schedule or
  an ISA atom with an explicit destination mask. `Dense4x256b` is exempted
  because its refresh semantics are not modeled as an ordinary rectangular
  logical row/column write footprint. Validation: `make -j8`, py-compile of
  `test_tmem_runtime_matrix.py`, `git diff --check`, and 15 targeted copy rows
  covering direct-seed and two-CTA `warpx2`, `4x256b` refresh, scales
  `warpx4`, the scales descriptor-view clean negative, and a dense tile
  permutation.
- 2026-04-15 23:43 UTC: scheduled copy instructions now also carry the
  message-adjusted `TMemCopyDestinationFootprint`. The instruction scheduler
  applies `tmemRowDelta` and `tmemDwordDelta` once, records the final physical
  row/column and encoded destination offset, and LLVM lowering consumes
  `instruction.destination.offset` directly. This keeps the emitted schedule
  behavior-preserving while making the instruction object the single source of
  truth for both source and destination footprints. Validation: `make -j8`,
  py-compile of `test_tmem_runtime_matrix.py`, `git diff --check`, and 11
  targeted copy rows covering direct-seed `warpx2`, two-CTA `warpx2`, `4x256b`
  dword-delta messages, scales `warpx4`, and the scales descriptor-view clean
  negative.
- 2026-04-15 23:40 UTC: scheduled copy instructions now carry a
  `TMemCopySourceFootprint` paired with the existing scheduled tile. The
  planner computes source row/column and instruction source footprint rows /
  columns once, and LLVM lowering consumes those coordinates for both
  MMAShared descriptor loads and direct-seed descriptor source offsets. This
  is behavior-preserving scaffolding for source/destination footprint
  comparison and non-uniform split schedules. Validation: `make -j8`,
  py-compile of `test_tmem_runtime_matrix.py`, `git diff --check`, and a
  focused copy selector covering direct-seed `warpx2::02_13`, two-CTA
  `warpx2::01_23`, dense tile permutations, `4x256b` refresh copy, scales
  `warpx4`, and the scales descriptor-view clean negative (`366 passed,
  10622 deselected in 610.87s`).
- 2026-04-15 23:27 UTC: probed whether the `warpx2` dense-shared clean
  negatives were only blocked by an overstrict shared-layout preflight. A
  temporary relaxation let `warpx2::01_23` dense shared layouts compile and
  emit the expected opcode for single-CTA and two-CTA rows, but runtime was
  wrong for both `f32` and `i32` (`maxdiff=127` versus the established
  `warpx2::01_23` oracle). The probe was removed and `make -j8` restored a
  source-consistent build. Conclusion: ordinary dense shared-linear source
  order is another schedule/projection gap, not a safe preflight lift.
- 2026-04-15 23:22 UTC: scheduled copy tiles now carry a
  `TMemCopyDestinationFootprint` instead of only a raw destination offset. The
  footprint records logical destination coordinate, physical row/column,
  instruction footprint rows/columns, and encoded TMEM offset, and lowering now
  consumes `tile.destination.offset`. This is behavior-preserving scaffolding
  for planner-owned physical instruction-footprint scheduling and future
  overwrite/mask proofs. Validation: `make -j8`, py-compile of
  `test_tmem_runtime_matrix.py`, focused copy runtime selector
  (`360 passed, 10628 deselected in 577.22s`), and `git diff --check`.
- 2026-04-15 23:08 UTC: verification and lowering now share
  `getTMemCopySourceConversion(...)` for the exact physical TMEM query to
  shared-memory source conversion. This is behavior-preserving, but it removes
  the duplicated `layout.invertAndCompose(shmemLl)` call from `TMEMCopyOp`
  verification and LLVM lowering and gives the next physical-footprint copy
  planner one place to enrich destination-to-source layout algebra. Validation:
  `make -j8`, py-compile of `test_tmem_runtime_matrix.py`, a focused copy
  mini-sweep covering dense tile permutations, `4x256b` refresh copy,
  positive scales `warpx4`, and scales clean negatives (`362` rows), the
  descriptor-view debug probe, and `git diff --check`.
- 2026-04-15 22:56 UTC: descriptor-row-stride copy failures now record
  whether the offending source row delta spans a full instruction row
  footprint. The scales descriptor-view repro reports
  `descriptorRowDelta=32 spansInstructionRows=1`, and the user-facing
  diagnostic now names the public `tcgen05.copy` operand boundary: one TMEM
  address plus one shared descriptor per instruction, with no per-column
  destination mask. This keeps the row as a clean negative unless a different
  atom/source format or a proven multi-message schedule can avoid overwriting
  unrelated destination columns. Validation: `make -j8`, py-compile of
  `test_tmem_runtime_matrix.py`, focused packed/scales copy pytest selector
  (`7` rows), descriptor-view query-debug probe, and `git diff --check`.
- 2026-04-15 22:50 UTC: packed-lane copy was probed one step further with a
  temporary shifted descriptor projection and adjusted physical-column schedule.
  It compiled and emitted the expected dense `tcgen05.cp.128x256b` counts, but
  runtime was wrong: f16/i16 read every second source column and i8 read every
  fourth source column. The probe was removed. This proves that simply dropping
  zero-offset lane bases loses lane selection; real support needs descriptor
  synthesis that preserves lane bits rather than treating physical dword
  columns as logical element columns.
- 2026-04-15 22:45 UTC: the structured copy instruction-column failure now
  preserves the schedule dimensions identified by the latest probes:
  `packedLaneBits` for sub-dword packed-lane state and `descriptorRowDelta` for
  column-selected descriptor-row strides. Normal diagnostics and clean-negative
  behavior are unchanged, but query debug now reports e.g.
  `descriptorRowDelta=32` for scales shared-subslice copy and
  `packedLaneBits=1` for legacy f16 packed copy. Validation: `make -j8`,
  py-compile, focused packed/scales copy pytest selector (`7` rows), debug
  probes for both fields, and `git diff --check`.
- 2026-04-15 22:42 UTC: legacy no-scales subword `tcgen05.copy` was
  re-probed as a packed-lane support candidate. A temporary planner change let
  sub-32-bit lane bits with zero offset pass the instruction-column proof, but
  the next layer still could not synthesize an MMAv5 shared descriptor for the
  packed TMEM projection, and the existing tile scheduler would still count
  the lane-expanded logical column dimension. The probe was removed, source was
  rebuilt, and the existing clean-negative rows passed again. Conclusion:
  legacy packed subword copy needs a first-class lane-aware TMEM/shared
  physical query plus a scheduler whose column extent is physical dwords and
  lane bits, not just a relaxed instruction-column predicate.
- 2026-04-15 22:33 UTC: copy instruction-column projection failures now have
  a typed planner model instead of being only inline diagnostic strings.
  `TMemCopyInstructionColumnProjectionFailure` classifies packed-lane state,
  non-contiguous offsets, non-offset components, and the important
  descriptor-row-stride selection case that blocks scales descriptor-view /
  shared-subslice copy. Query debug now prints the structured failure, e.g.
  `descriptor-row-stride-selection bit=2 actual=1024 expected=4`, giving the
  next source-column/message split scheduler a real requirement object to
  consume. Validation: `make -j8`, py-compile of
  `test_tmem_runtime_matrix.py`, focused copy diagnostic pytest selector
  (`15` rows), query-debug kind probe, and `git diff --check`.
- 2026-04-15 22:29 UTC: two quick guard-lift reprobes were recorded and
  removed without source changes. Mixed-precision fp4A TMEM-LHS scaled-MMAv5
  compiles if the verifier guard is bypassed, but representative tile,
  subslice, legacy-accumulator, and linear-accumulator cases all produce the
  same large numerical error (`max ~= 1084`, `mean ~= 69.8`), confirming this
  is still a padded operand-A storage contract boundary. The scales
  shared-row-subslice copy row fails for the same sub-instruction
  column-selected descriptor-row stride as the descriptor-view row: source
  column bit 2 maps to shared offset 1024, i.e. 32 descriptor-row strides,
  which current `warpx4` copy scheduling cannot split or mask inside one atom.
  The temporary guard lift was removed and `make -j8` rebuilt clean source.
- 2026-04-15 22:23 UTC: two-CTA no-scales
  `tcgen05.copy.warpx2::02_13.64x128b` diagnostics now record the current
  descriptor/address schedule proof instead of the older coarse probe
  summary. The descriptor path fails because logical row bit 5 maps to a
  one-dword source offset rather than an affine 8-row source stride; the
  direct-seed `cta_group::2` path with source offset 32 emits the opcode and
  writes the correct low destination columns, but duplicates that low
  source-column pair into the high destination columns. Non-zero subaligned
  destination dword deltas fault, while aligned deltas that would complete the
  single-CTA schedule read zeros. Validation: `make -j8`, py-compile of
  `test_tmem_runtime_matrix.py`, focused two-CTA `warpx2::02_13` pytest
  selector (`14` rows), and `git diff --check`.
- 2026-04-15 22:10 UTC: plain MMAv5 unsupported linear-layout diagnostics now
  state the current tile-order ISA boundary: public `tcgen05.mma` atoms plan
  tensor-memory operands as physical instruction tiles whose row/column basis
  order must remain canonical inside each tile. Arbitrary in-tile row or
  column permutations therefore remain clean negatives until the backend has a
  supported permutation, masked writeback, or equivalent tile-splitting
  schedule. Validation: `make -j8`, direct invalid verifier RUN, py-compile of
  `test_tmem_runtime_matrix.py`, focused MMA clean-negative pytest selector
  (`17` rows), and `git diff --check`.
- 2026-04-15 22:07 UTC: dense no-scales row-permuted copy diagnostics now
  state the real atomization boundary: current dense `tcgen05.copy` atoms
  write the full physical row footprint in basis order, so row-permuted
  destinations need a destination-row mask, row-partitioned atom, or equivalent
  smaller copy footprint before support can be promoted. The runtime
  clean-negative matrix now asserts this wording. Validation: `make -j8`,
  py-compile of `test_tmem_runtime_matrix.py`, direct invalid verifier RUN,
  row/column permutation clean-negative pytest sweep (`15` rows), and
  `git diff --check`.
- 2026-04-15 22:02 UTC: copy instruction-column projection diagnostics now
  report when an offending sub-instruction source-column bit is actually
  selecting a descriptor-row stride. This preserves the current clean-negative
  behavior, but it makes the scales/warpx4 frontier more precise: the planner
  is missing a schedule dimension that can atomize a column-selected row delta
  with a destination-column mask, source format, or equivalent smaller atom,
  not merely a larger MMAShared descriptor search. Validation: `make -j8`,
  direct invalid verifier RUN, exact `cp_scales_tmem_descriptor_view`
  clean-negative pytest row passed, and `git diff --check`.
- 2026-04-15 21:23 UTC: M64 noncanonical-row `ld.red` default reduction
  layout selection now uses the existing split-N `16x32bx2` path instead of
  asking for a scalarizing `32x32b` layout. Row-permuted M64 linear layouts
  were not ISA-impossible: explicit `auto`, `16x32bx2`, and
  `32x32b_splitn` already produced correct hardware reductions. The frontend
  default now selects that path for noncanonical M64 row layouts while
  preserving the compact canonical-row path. `TMEMLoadOp::verify()` and LLVM
  lowering also reject any selected `tcgen05.ld.red` plan whose message count
  would be `.x1`, with a diagnostic pointing to reduction-compatible layouts
  or explicit software reduction. Validation: `make -j8`, `py_compile`,
  direct invalid verifier RUN, new M64 row/col-permuted focused rows passed
  `7` tests, broad `ld_red_m64` passed `71` rows, neighboring `ld.red`
  positives/negatives passed `160` rows, and `git diff --check`.
- 2026-04-15 21:09 UTC: M64 split-N `tcgen05.ld.red` is now positive for
  the `16x32bx2` lane-split family. The reduction layout predicate now
  distinguishes layouts whose N dimension is entirely in registers from the
  one ISA-realizable cross-lane case where exactly lane bit 4 carries the
  missing N basis. Verifier and lowering share
  `getTmemLoadReductionLaneSplitMask(...)`; lowering combines the two
  per-lane partial reduction values after `tcgen05.wait::ld` with a
  `shfl.xor 16` min/max. Runtime coverage moved M64 `N in {32,64,128,256}`
  out of the clean-negative bucket and covers default `load_min/load_max`
  plus explicit `auto`, `32x32b`, `16x32bx2`, and `32x32b_splitn` variants.
  Validation: `make -j8`, direct invalid verifier RUN, `64` focused M64
  rows passed, neighboring `ld.red` positives/negatives passed `160` rows,
  `py_compile`, and `git diff --check`.
- 2026-04-15 20:51 UTC: the scales descriptor-view copy source-projection
  probe was narrowed further. A temporary projection that moved source offset
  `256` from instruction column bit 2 into the descriptor row dimension and
  moved source offset `8` from row bit 0 into instruction column bit 3 became
  descriptor-representable and preserved source columns, but runtime output
  rotated source rows left by one bit (`dst row r` read `src row rotl7(r)`).
  The row-correct variant with low descriptor row bases restored to
  `8,16,32,64,128` was not representable as an MMAv5 shared descriptor.
  Probe hooks were removed, source rebuilt with `make -j8`, and the original
  clean-negative pytest row passed. The next real path needs an actual
  destination-row/source-message schedule with row-group atomization or a proof
  that the duplicate row/column source basis is ISA-impossible.
- 2026-04-15 20:44 UTC: the C++ TMEM operand verifier now recognizes the
  `tcgen05.copy.4x256b` refresh-shaped tensor-memory layout as the same
  direct `ld/st` row-anchor boundary guarded in Python. Direct MLIR
  `ttng.tmem_load` on that view now reports a clean unsupported diagnostic
  explaining that `tcgen05.ld/st` row anchors must be materializable as warp
  bases; it no longer falls through to generic register-layout enumeration if
  the frontend guard is bypassed. Validation: `make -j8`, direct invalid
  verifier RUN, exact `4x256b` copy-positive and refresh direct-ld/st
  clean-negative pytest nodeids passed, and `git diff --check` passed.
- 2026-04-15 19:42 UTC: scales descriptor-view copy was re-probed with a
  temporary `warpx4` descriptor-variant expansion plus instruction-column
  preflight bypass. The exact view still had logical source column bit 2
  mapping to shared offset `256`, and descriptor synthesis found no
  representable `32x16` MMAShared descriptor across `254` candidates. Probe
  hooks were removed and `make -j8` rebuilt clean source. The next support
  path is an ISA-grounded source-column/message or row-partition split, not
  broader descriptor enumeration.
- 2026-04-15 19:36 UTC: copy tile scheduling now carries both the logical
  destination column and the shared-memory source column. The old
  `TMemCopyDestinationTile` carrier was replaced by `TMemCopyScheduledTile`
  with `logicalCol`, `sourceCol`, and `destinationOffset`, and lowering uses
  `sourceCol` for both descriptor-loader and direct-seed source addressing.
  Current plans keep `sourceCol == logicalCol`, so this is a
  behavior-preserving split-schedule carrier for future non-uniform
  source-column planning. Validation: `make -j8`, direct invalid verifier RUN,
  focused copy selector passed all `375` selected rows, and `git diff --check`
  passed.
- 2026-04-15 19:23 UTC: destination tile planning now lives on the selected
  executable copy plan. `TMemCopyExecutablePlan` carries
  `destinationTiles`, populated in `getTMemCopyPlanRealization(...)`, and
  lowering consumes those tiles instead of recomputing them after plan
  selection. A temporary standalone-query probe for scales descriptor-view copy
  confirmed there is no shortcut through the root query: the standalone view
  fails copy-family classification, so exact-view source-column splitting is
  still required. Validation: `make -j8`, direct invalid verifier RUN, broad
  focused copy selector passed all `409` selected rows, and `git diff --check`
  passed.
- 2026-04-15 19:08 UTC: copy instruction-column projection is now an
  executable-plan carrier, `TMemCopyInstructionColumnProjection`, populated
  during shared copy-plan realization beside the source-row carrier. A bounded
  temporary bypass probe for the scales descriptor-view copy confirmed that the
  next failure is still structural: MMAShared descriptor synthesis has no
  representable 32x16 descriptor for the projection where a low instruction
  column bit maps to shared offset 256. Validation after removing the probe:
  `make -j8`, direct invalid verifier RUN, focused scales copy selector passed
  all `12` selected rows, and `git diff --check` passed.
- 2026-04-15 19:02 UTC: copy source-row projection is now an executable-plan
  carrier, `TMemCopySourceRowProjection`, populated during shared copy-plan
  realization. Lowering no longer reruns the support predicate after selecting
  a plan; it consumes a scheduled message whose row-projection proof has
  already been materialized. Validation: `make -j8`, direct invalid verifier
  RUN, focused row-plan runtime selector passed all `21` selected rows, and
  `git diff --check` passed.
- 2026-04-15 18:35 UTC: dense direct-copy row projection proof is now a named
  helper, `getDenseTMemCopyRowProjectionSupport(...)`. The refactor preserves
  current behavior and diagnostics while creating one planner insertion point
  for future row/source projection schedules. Validation: `make -j8`,
  row/column permutation clean negatives split across four GPUs passed all
  `15` selected rows, positive dense tile-permuted `[128-32-16]` passed, and
  `git diff --check` passed.
- 2026-04-15 18:32 UTC: dense row-permuted copy was re-probed with only the
  row-order physical-query guard lifted. Descriptor search can pick a nominal
  shared descriptor for `reverse/identity`, but lowering then hits the dense
  row-stride invariant because `128x256b` expects affine ascending physical row
  offsets. The probe hooks were removed and `make -j8` rebuilt clean source.
  Keep the current clean negative until a real source-row projection or row
  atomization schedule exists.
- 2026-04-15 18:28 UTC: repeated-`N=32` scaled-MMAv5 got one more targeted
  scale-B probe based on the PTX `scale_vec::1X` B-scale sub-column layout.
  The XOR-derived schedule needs odd B-scale TMEM word columns for some
  N/K fragments and faults with a misaligned address. Probe hooks were removed
  and `make -j8` rebuilt clean source. Treat this as a scales storage/fragment
  contract issue, not an address-remap bug in current lowering.
- 2026-04-15 18:20 UTC: scaled-MMAv5 lowering now factors scale address and
  SFA/SFB sub-column selection through a behavior-preserving
  `MMAv5ScaleFactorFragment` helper. This does not relax the repeated-N32
  guard; it creates the implementation seam needed for a real future B-scale
  fragment planner. Validation after `make -j8`: repeated-N32 clean-negative
  selector split across four GPUs passed all `10` selected rows, the positive
  `mxfp8/mxfp8, M=N=128, K=128, linear` scaled-root row passed, and
  `git diff --check` passed.
- 2026-04-15 18:17 UTC: repeated-`N=32` scaled-MMAv5 was re-probed with an
  env-gated lowering sweep over B-scale address and SFB-ID formulas. The
  default guard-lift path again left only the first 32-column tile correct;
  `wordaddr_nid` and `wordaddr_xor` compiled but stayed numerically wrong,
  while `kaddr_nid`, `kaddr_xor`, and `packed` faulted during execution. All
  probe hooks were removed and `make -j8` rebuilt the clean source. Keep the
  guard: this needs a real B-scale fragment model, not scalar remaps.
- 2026-04-15 18:08 UTC: the dense row-permutation clean negative now reports
  the actual missing layer: an explicit source-row projection schedule for
  row-permuted destinations. Focused four-GPU split validation over the
  row/column permutation clean-negative matrix passed all `15` selected cases.
  This is a diagnostic/contract checkpoint, not a support promotion.
- 2026-04-15 17:59 UTC: the 4x256b refresh direct `tcgen05.ld/st` negative
  was re-probed by temporarily bypassing the Python API guard. The backend
  still found no legal register layout: the raw refresh query uses the
  128-row row plan, but all candidate atoms leave warp bases at zero while
  carrying the required 32/64 row anchors through columns or lane bases. Keep
  this as a true direct-ld/st row-anchor materialization boundary, separate
  from the now-positive `tcgen05.cp.4x256b` refresh copy opcode support.
- 2026-04-15 17:55 UTC: the repeated-`N=32` scaled-MMAv5 guard is now a
  shared physical-layout helper used by both verifier and lowering. Follow-up
  probes confirmed the boundary: fixed B-scale selector remaps do not repair
  the repeated-N32 result, and removing the 64-column B-scale stride reaches a
  misaligned scale address. The next real support path needs a new B-scale
  fragment model, not duplicated guard text or selector-map tuning.
- 2026-04-15 17:36 UTC: dense no-scales copy row permutations were re-probed
  with the physical-query and lowering row-stride guards temporarily bypassed.
  Row-only permutations compile in that mode but produce wrong output:
  `reverse` exposes bit-reversed source-row order, while `rotate1` and
  `even_odd` expose analogous physical-row-basis ordering. Column-only
  permutations still fail cleanly at physical column packet contiguity. Keep
  these as real row/source projection scheduler gaps; guard lifting is wrong.
- 2026-04-15 17:33 UTC: the post-scaled-promotion clean unsupported/error
  selector is still green (`417 passed`, `1 skipped`) after the scaled
  single-CTA and two-CTA `N=32` promotions. The current clean-negative buckets
  remain aligned with the sharpened scaled-N32 and copy-scheduler boundaries.
- 2026-04-15 17:28 UTC: scaled MMAv5 two-CTA accumulator subslice coverage
  now includes `block_n=32`. The two-CTA scaled view rows pass across current
  format pairs, K-depths, multicast modes, slice starts, and accumulator
  modes, further isolating the remaining scaled `N=32` issue to repeated
  tile-permuted scale-fragment schedules.
- 2026-04-15 17:26 UTC: scaled MMAv5 `N=32` root/view coverage is now
  positive for the supported single-CTA families: root accumulators, indexed
  accumulator views, accumulator subslice views, TMEM-LHS subslice views, and
  TMEM-LHS tile-permuted rows. This distinguishes valid single-tile `N=32`
  scaled-MMAv5 layouts from the still-real repeated/tile-permuted `N=32`
  scale-fragment gap.
- 2026-04-15 17:12 UTC: the remaining plain MMAv5 `N=32` coverage debt in
  M64 and TMEM-LHS matrices is now saturated for the supported families. M64
  linear accumulator roots and M64 accumulator subslice views include `N=32`;
  TMEM-LHS tile-permuted and LHS subslice rows include output `N=32`. The
  M64 legacy root keeps its 64-column accumulator tile contract.
- 2026-04-15 16:58 UTC: the plain MMAv5 `N=32` accumulator promotion now
  reaches the adjacent single-CTA and TensorDescriptor-fed two-CTA matrices.
  Positive coverage includes single-CTA root accumulator, indexed accumulator
  view, and accumulator subslice view rows, plus two-CTA TensorDescriptor-fed
  non-TF32 and TF32-transposed-B rows. The direct i8 unsupported diagnostics
  also now cover `N=32`.
- 2026-04-15 16:43 UTC: plain MMAv5 two-CTA accumulator coverage now includes
  `block_n=32` for root layouts, indexed accumulator descriptor views, and
  accumulator subslice views. The backend already handled the narrow
  cta-group::2 physical image after the earlier accumulator-family promotions;
  the stale runtime matrix floor has been removed while preserving the
  `N=256` linear-parent hardware resource boundary.
- 2026-04-15 16:29 UTC: plain MMAv5 TMEM-LHS tile-permuted coverage now
  reaches `K=32` and `K=64`. The LHS family planner admits 8-column storage
  families after probes showed `K=32/tile_n=8` is valid across all plain
  MMAv5 kinds and both accumulator modes; `K=64/tile_n=16` was already valid
  after the earlier 16-column LHS planner promotion.
- 2026-04-15 16:23 UTC: scaled-MMAv5 narrow tile-permuted accumulator layouts
  were re-probed after the plain narrow promotion. Temporarily admitting
  8- and 16-column scaled accumulator families made `N=32/tile_n=8` and
  `N=64/tile_n=16` compile, but every sampled scaled format pair produced
  large numerical error. Keep the scaled narrow rows as scale-fragment
  boundaries, separate from the now-positive plain MMAv5 narrow rows.
- 2026-04-15 16:18 UTC: plain MMAv5 tile-permuted accumulator coverage now
  includes the narrow public instruction families `N=32/tile_n=8` and
  `N=64/tile_n=16`. The plain accumulator planner now admits 8- and
  16-column N families; scaled MMAv5 keeps its separate wider-alignment
  planner. The stale plain narrow clean-negative table/test was removed, while
  scaled narrow negatives remain green.
- 2026-04-15 16:09 UTC: scaled-MMAv5 full-shape TMEM-LHS tile-permuted fp4
  storage `K=128` is now positive. The LHS MMAv5 family planner now includes
  16-column storage families because packed fp4/nvfp4 operands plan over byte
  storage columns, not logical K columns. The stale clean-negative table was
  removed and the existing scaled-LHS tile-permuted positive/use-acc matrices
  now cover `mxfp4/mxfp4` and `nvfp4/nvfp4` at `K=128`.
- 2026-04-15 15:58 UTC: scaled-MMAv5 repeated-`N=32` accumulator layouts were
  re-probed with a temporary verifier/lowering guard lift. Every representative
  scaled format pair compiled, but the runtime oracle failed with large output
  error. This confirms that the existing guard is not stale: the current public
  tensor-memory scales descriptor model only exposes matrix-B scale fragments
  at 64-column alignment, so repeated `N=32` scaled MMA needs a real
  scale-B fragment/addressing plan before promotion.
- 2026-04-15 15:53 UTC: dense no-scales row/column permutation copy guards
  were re-probed with a temporary environment-gated guard lift. Forcing row
  basis permutations past the physical-query guard reached lowering asserts in
  the multicast row-stride checks, while forcing column basis permutations
  either faulted with a misaligned address (`identity/reverse`) or compiled to
  wrong output (`identity/rotate1`, `identity/even_odd`). All probe code was
  removed and the tree rebuilt. Keep these rows as real row/packet schedule
  gaps, not stale clean negatives.
- 2026-04-15 15:47 UTC: a bounded temporary direct-seed sweep for two-CTA
  no-scales `tcgen05.copy.warpx2::02_13` found no correct schedule. Re-enabling
  the single-CTA direct seed under `cta_group::2` and sweeping representative
  B128 source bases plus aligned destination dword deltas either duplicated the
  low source-column pair (`delta=0`) or read zeros (`delta>0`). All probe code
  was removed and the tree rebuilt. This reinforces that support needs a real
  cta-group::2 descriptor/address schedule, not a seed-parameter tweak.
- 2026-04-15 15:31 UTC: no-scales dense linear `tcgen05.copy` coverage was
  corrected for the `M=256,N=16` frontier. A fresh probe showed the current
  exact-query path already folds the high row selector into TMEM columns,
  selects `tcgen05.cp.128x256b`, and passes the runtime oracle for swizzle 32
  and 64 with four copy messages. The stale clean-negative row was removed and
  the two valid swizzles were promoted to positive coverage. Swizzle 128 is a
  shared-memory layout precondition failure for a 16-column f32 tile, not a
  TMEM copy backend gap.
- 2026-04-15 15:24 UTC: M64 split-N `tcgen05.ld.red` was probed and kept as a
  clean reduction-schedule boundary. Temporary local changes could make the
  active 64-row image emit `ld.red.16x32bx2`, but runtime rows still aliased:
  rows 16-31 duplicated rows 0-15 and rows 32-63 were zero. The known-good
  M64 split-N `ld/st` schedule uses two `16x32bx2.x8.b32` messages with an N
  bit in lanes, while `ld.red` currently requires N entirely in registers.
  Future support needs a real cross-lane/thread partial-reduction design, not
  a row-plan or guard-lift tweak.
- 2026-04-15 15:04 UTC: the two-CTA scales descriptor-view direct
  `tcgen05.ld/st` family now covers the full current CGA bucket:
  `128x64`, `256x32`, and `256x64`. The exact recognizer now accepts both
  raw-query forms: the `128x64` view keeps the zero row-tail bases and uses
  row anchors `16,32`, while the `256` views strip the zero row-tail bases,
  use row anchors `0,0`, carry row bits `16,32` in the column stream, and use
  `block=[[64,0]]`. The planner derives the register layout from those exact
  row/column/block facts rather than by enumerating shapes, and validates the
  candidate against the raw descriptor-view query before returning it. Runtime
  coverage promotes all three CGA rows; the old clean-negative table is now
  empty for this kernel.
- 2026-04-15 14:58 UTC: two-CTA scales descriptor-view direct
  `tcgen05.ld/st` now covers the proved `128x64` CGA view. The backend has an
  exact raw-query recognizer for the view produced by the scales
  reshape/permute/reshape chain: logical row bit 6 is carried by lane state,
  logical row bit 4 is the register row selector, CTA ownership is the block
  row selector, and the selected register layout validates against the raw
  descriptor-view query instead of falling through to type-only fallback. The
  old two-CTA int8 guard still blocks unproved views, but allows this exact
  family so the real planner can prove it. Runtime coverage promotes
  `M=128,N=64,num_ctas=2,cga=[[1,0]]`; the `256x32` and `256x64` CGA rows
  remain clean unsupported until their distinct exact layouts are proved.
- 2026-04-15 14:32 UTC: `tcgen05.ld.red` now also covers the expanded-row
  column-permuted f32 subset enabled by the folded direct query. The reduction
  source predicate no longer requires canonical column basis order for
  256-row pure separable layouts; it proves the column bases as a set and
  relies on direct load/store encoding validation to reject any schedule the
  folded query cannot realize. Runtime coverage promotes representative
  `identity/reverse, N=64` and `even_odd/even_odd, N=128` reduction rows over
  min/max, abs, and NaN propagation.
- 2026-04-15 14:26 UTC: expanded-row f32
  `TensorMemoryLinearLayout` direct `tcgen05.ld/st` now supports separable
  256-row layouts with non-canonical column basis order. The temporary
  clean-negative boundary has been replaced by an exact folded physical query:
  low row bases remain in the 128-row TMEM row coordinate, original column
  bases keep their user order in TMEM columns, and the high row selector is
  appended as the next TMEM column bit. Direct load/store no longer reaches a
  raw row-128 schedule or a canonicalized fallback that would hide column
  order. Runtime coverage promotes representative column-permuted direct
  roundtrips, and the nearby expanded-row `ld.red` column-permuted rows now
  fail at the reduction source contract after their source stores succeed.
- Superseded by the 14:26 support checkpoint. 2026-04-15 13:59 UTC:
  expanded-row `TensorMemoryLinearLayout` values with
  non-canonical column packet order now have a shared clean direct `ld/st`
  boundary. A common classifier recognizes the pure expanded-row physical
  image and detects column packet-order permutations. Gluon `load`/`store`,
  source-initialized TMEM allocs, and IR op verifiers now report that direct
  `tcgen05.ld/st` needs an explicit packet-offset schedule for these layouts
  instead of letting them reach allocator assertions or known wrong-code-prone
  schedules. The nearby `ld.red` expanded-row column-permuted negatives now
  point at this first failing store/source packet boundary.
- 2026-04-15 13:43 UTC: `tcgen05.ld.red` now covers 256-row
  `TensorMemoryLinearLayout` sources with row basis permutations and canonical
  column packet order. The allocator now recognizes separable expanded-row
  linear images as a 128-row physical TMEM allocation with the extra row
  selector folded into columns, independent of row-basis order. This fixes the
  allocator assertion exposed by `M=256` row-permuted reductions and keeps
  `reverse`, `rotate1`, and `even_odd` row-basis permutations positive for
  `N in {32,64,128}` representatives. A probe also showed `M=256` column
  permutations miscopy under the current direct load/store packet schedule, so
  the `ld.red` source predicate now rejects expanded-row column permutations
  cleanly until that distinct schedule gap is solved.
- 2026-04-15 13:30 UTC: `tcgen05.ld.red` now accepts ordinary identity
  `TensorMemoryLinearLayout` sources with `M=256` and `N in {32,64,128}`.
  The previous reduction-friendly source predicate only admitted a 128-row
  base tile or the legacy-equivalent 256-row spelling that encoded the high row
  selector as a column carry basis. Runtime probes showed the identity layout
  lowers through the same direct raw-query path and passes the reduction oracle
  with 8 warps, so the predicate now treats both 128-row and 256-row pure row
  bases as reducible. The `256x256` identity case is reclassified as a shared
  memory resource boundary rather than an ISA/layout unsupported case.
- 2026-04-15 13:07 UTC: direct `tmem.load`/`tmem.store` for the 4x256b
  refresh-shaped active layout is now recorded as a proof-level clean
  unsupported contract at the frontend descriptor boundary. Probes showed the
  register-layout planner can only form candidates where the required physical
  row anchors are lane bits or broadcasts; `tcgen05.ld/st` packets require
  those anchors to be materializable as TMEM row warp bases, while the refresh
  view stores logical row bits in TMEM columns and low logical column bits in
  TMEM rows 32/64. Copy support remains positive for the proved refresh
  images, but direct load/store now fails early with an actionable diagnostic
  instead of the generic unsupported-layout message.
- 2026-04-15 13:03 UTC: `tcgen05.cp.4x256b` refresh support now covers the
  two-CTA `cta_group::2` image as well as the single-CTA image. The refresh
  recognizer treats an implicit size-1 block dimension as single-CTA but
  accepts the real two-CTA block lift when `block=1 -> (4,0)` and
  `shape=[8,8]`; the planner then emits the same two-message refresh schedule
  using `tcgen05.cp.cta_group::2.4x256b`. Runtime-matrix coverage now pins the
  two-CTA codegen row, the Blackwell conversion test has the matching
  compiler-only check, and the ordinary contiguous two-CTA `4x256b` case
  remains a clean unsupported negative. Remaining boundary is unchanged:
  direct refresh-layout `tmem.load`/`tmem.store` still lacks a register-layout
  contract, so this checkpoint proves copy scheduling/codegen.
- 2026-04-15 12:52 UTC: `tcgen05.cp.4x256b` now has first-class support for
  the refresh-shaped active `TensorMemoryLinearLayout` image proved in the
  earlier probes. Copy physical-query selection is shared between verifier and
  lowering, exact linear root queries can be selected when no standalone
  canonical spelling exists and the shared source composes with the exact
  image, and the 4x256 planner now emits the two-message refresh schedule with
  per-message descriptor projections. Ordinary contiguous four-row
  `ttng.tmem_copy` remains clean unsupported with a physical-query diagnostic
  because it is not the image the ISA writes. Current boundary: the positive
  row pins copy codegen/opcode emission; direct load/store of the refresh
  layout still needs a real contract before this becomes a full tmem
  round-trip data oracle.
- 2026-04-15 12:35 UTC: `tcgen05.cp.4x256b` now has a stronger physical
  refresh-schedule proof. A temporary re-enable showed one message maps source
  columns 0..3 to physical rows 0/32/64/96 with source rows packed across
  destination dwords; a second message with `smemColOffset=4` and
  `tmemDwordDelta=4` fills source columns 4..7. A refresh-shaped logical view
  would need row bits in TMEM columns, low column bits in TMEM rows 32/64, and
  the high column bit at destination dword +4. Current descriptor-view
  inference/load-store support cannot expose that as a legal logical
  `ttng.tmem_copy` target yet, so ordinary contiguous `4x8` copies remain
  clean unsupported with an updated diagnostic.
- 2026-04-15 12:25 UTC: scales descriptor-view copy was re-probed with
  temporary multi-message and descriptor-skip machinery. A second
  `warpx4.32x128b` message with `smemRow=64` does not change the parent-order
  row mapping; adding `tmemRowDelta=1` faults with a misaligned address, and
  `tmemRowDelta=64` still leaves the same wrong output. Temporarily skipping
  representable descriptor matches found only two representable descriptor
  selections: the first produces the known parent/root row order, and the
  second duplicates row groups. Skips beyond that return to the clean-negative
  diagnostic. All probe code was reverted; this row still needs a real
  row-interleaving schedule, not a broader descriptor search.
- 2026-04-15 12:10 UTC: dense no-scales linear copies now support the
  MMAv5-backed `M=256` row-group cases whose exact physical query is a
  `128x{64,128,256}` family image. Copy planning now accepts exact physical
  queries whenever active shape, element bitwidth, CTA ownership, and scales
  classification match and the exact layout composes with the shared-memory
  source. TMEM allocation sizing also uses the proven MMAv5 family image for
  oversized exact linear layouts, so `256x32`, `256x64`, and `256x128`
  `TensorMemoryLinearLayout` copies are positive runtime rows across the
  32/64/128 swizzle variants. `256x16` remains a clean copy-family negative.
- 2026-04-15 11:56 UTC: dense no-scales copy now reports the concrete
  multi-message row-group gap for single-CTA `M=256,N=128` projections. A
  temporary guard-lift probe classified the shape only by pretending a
  `256`-row conversion was one dense atom, then aborted in descriptor
  realization because the descriptor planner has no first-class row-group
  dimension. The verifier now says the projection has 256 logical source rows
  and that support needs a multi-message row-group schedule preserving the
  extra row selector as descriptor projection plus source and destination row
  offsets.
- 2026-04-15 11:52 UTC: legacy `TensorMemoryLayout` subword dense copy now has
  a precise planner diagnostic instead of a generic descriptor-synthesis miss.
  Canonical `TensorMemoryLinearLayout` subword copies remain positive, but
  legacy packed layouts encode sub-32-bit lane state as zero low column bases
  (`col=1 -> offset 0` for 16-bit, `col=1/2 -> offset 0` for 8-bit). The copy
  descriptor planner cannot synthesize a `tcgen05.copy` descriptor from that
  projection without an explicit packed-lane model, so the clean negative now
  says to use an unpacked `TensorMemoryLinearLayout` or `tmem.store/load` until
  packed-lane copy semantics are modeled.
- 2026-04-15 11:46 UTC: a follow-up `4x256b` view-layout probe tried to make
  the refresh pattern explicit by selecting a four-row view from a
  `128x8` linear parent with rows grouped through a
  reshape/permute/slice/index chain. The exact copy conversion
  for that view loses the source row dimension and has only column bases into
  shared offsets `4` and `8`, so it does not classify as `tcgen05.copy.4x256b`.
  The canonical parent slice still classifies as `4x256b` and is cleanly
  blocked by the refresh-primitive diagnostic. This rules out a simple sparse
  descriptor-view rescue; a real promotion still needs a planner-level refresh
  schedule that explicitly models where the four source rows land.
- 2026-04-15 11:39 UTC: scales descriptor-view copy was probed with a
  temporary descriptor-basis exchange that makes the shared descriptor
  representable by restoring contiguous low source-column bits. Runtime then
  emitted eight `tcgen05.cp.cta_group::1.warpx4.32x128b` messages for the
  128x32 view, but output row `r` read source row
  `(r % 64) * 2 + r / 64`: the parent/root physical row order, not the logical
  descriptor-view row. The probe code was reverted. This confirms the clean
  negative is preventing a real wrong-code path, and support needs an explicit
  row-interleaving destination/source schedule after descriptor projection.
- 2026-04-15 11:30 UTC: two-CTA no-scales `warpx2::02_13`
  diagnostics now preserve the direct probe result in the actual verifier
  message. Reusing the single-CTA direct seed under `cta_group::2` can emit
  `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b`, but it duplicates the low
  source-column pair instead of preserving the high source-column bit. The
  aligned dword deltas that complete the single-CTA schedule read zeros under
  `cta_group::2`, and other tiny deltas are misaligned. This keeps the clean
  negative proof-level while the real work remains deriving a cta-group::2
  descriptor/address schedule rather than retrying the known-bad direct-seed
  path.
- 2026-04-15 11:21 UTC: dense no-scales `tcgen05.cp.128x256b`
  now supports subword element types for canonical linear TMEM copies when the
  source shared-memory descriptor uses the real element bitwidth. The old
  verifier-level `Source element type should be 32-bit` rejection was removed
  so support is decided by the shared copy planner. The test kernels now build
  `NVMMASharedLayout` with the input dtype bitwidth, promoting `f16`, `bf16`,
  `i16`, and `i8` linear dense copy rows from clean negative to positive
  runtime coverage. `warpx2` subword copies remain clean unsupported with a
  planner diagnostic that the current `warpx2` descriptor schedule requires
  32-bit shared elements. Dense atom enumeration is now explicit enough to add
  narrower dense atoms without leaking the dense-only 64x2 descriptor fallback
  into multicast/scales families.
- 2026-04-15 11:10 UTC: `tcgen05.cp.4x256b` diagnostics now reflect the
  ISA-grounded probe result instead of the earlier coarse "single destination
  row" summary. CUTLASS models this instruction as a TMEM refresh primitive,
  and direct runtime probes agree: one message maps source-column vectors onto
  tensor-memory lanes separated by 32, with four source rows packed into
  destination dwords; destination dword offsets must be 128-bit aligned. The
  family remains clean unsupported for ordinary contiguous four-row
  `ttng.tmem_copy` until the linear-layout planner has an explicit refresh
  schedule and a row-coded runtime oracle.
- 2026-04-15 10:57 UTC: no-scales `warpx2` indexed and
  slice-index descriptor views no longer assert while computing already
  adjusted TMEM copy base offsets. The root cause was `getTMemViewOffsetImpl`
  pseudoinverting a non-surjective 2-D view layout even when all trailing
  logical offsets were zero; for `memdesc_index` and related prefix views the
  zero trailing coordinate maps to physical row/column zero, and the prefix
  contribution is added separately. Copy verifier/lowering now also reject
  non-composable exact physical queries before lowering tries to
  `invertAndCompose`, and the legacy load/store support probe now guards the
  same operation. Validation passed: `make -j8`, the exact original failing
  warpx2 indexed-view row (`1 passed`), the 24-case indexed/slice-index
  warpx2 view bucket (`24 passed`), the full 4-GPU `cp_no_scales` sweep
  (`305 passed, 10 skipped`), and `git diff --check`.
- 2026-04-15 10:41 UTC: the dense macro-tile destination schedule now also
  handles non-ascending high macro-selector bases. The first macro-tile
  promotion covered a high-low crossing (`N=128`, `tile_n=32`); a follow-up
  `N=256`, `tile_n=64` probe exposed the high-high reorder case (`128` before
  `64`). `needsDenseTMemCopyPhysicalColumnTileOffsets(...)` now switches to
  physical per-tile destination offsets for both patterns, and the runtime
  matrix covers `tile_n=64` at `N=256` as a positive.
- 2026-04-15 10:38 UTC: dense no-scales `tcgen05.copy.128x256b`
  now separates source-column iteration from destination TMEM tile addressing
  when the destination `LinearLayout` permutes whole 128-byte column
  macro-tiles. The direct-copy support proof still requires every logical
  instruction-width tile to map to one aligned, contiguous, unique physical
  TMEM tile, but higher column macro-selector bases no longer need to stay
  globally ascending. Lowering uses exact layout algebra to compute per-tile
  destination offsets only for macro-crossing permutations; low descriptor
  macro permutations keep the existing address path. Runtime coverage promotes
  `tile_n=32` from clean negative to positive beside `tile_n=8` and
  `tile_n=16`; bit-scrambled and mixed layouts remain clean unsupported.
- 2026-04-15 10:29 UTC: the earlier positive `tcgen05.cp.4x256b`
  conversion-only coverage has been retracted into a clean unsupported
  contract. Runtime probing showed the naïve four-row descriptor schedule
  silently placed source row values into one destination row, so plan
  realization now rejects `4x256b` with an explicit "recognized by the ISA but
  no validated descriptor/address schedule" diagnostic. The family remains a
  real missing ISA-coverage item for the atomized copy planner; it must not be
  re-enabled until a row-coded runtime oracle proves the schedule.
- 2026-04-15 10:15 UTC: descriptor-synthesis failures now add a planner
  note when the source projection moves a source column bit inside one
  `tcgen05.copy` instruction away from contiguous shared offsets. The exact
  scales descriptor-view row is now diagnosed as: source column bit 2 maps to
  shared offset 256 instead of contiguous offset 4, and current scheduling
  cannot split sub-instruction source columns. This records the concrete
  schedule invariant that a future support promotion must solve, not just the
  fact that MMAv5 descriptor synthesis failed.
- 2026-04-15 10:13 UTC: descriptor-candidate generation now consistently
  uses `TMemCopyMessagePlan::descriptorCvt` for descriptor column extents when
  a future atomized schedule supplies a per-message source projection. This is
  behavior-preserving today because no current plan sets `descriptorCvt`, but
  it closes a layering hole where folded `warpx2` descriptor variants and the
  direct shared-seed candidate would have mixed the full copy conversion back
  into descriptor synthesis.
- 2026-04-15 10:06 UTC: exact scales descriptor-view copy diagnostics now
  tell developers to set `TRITON_DEBUG_TMEM_QUERY=1` to print the selected
  copy conversion and descriptor candidates. This keeps the clean negative
  actionable while the schedule remains unsupported.
- 2026-04-15 10:05 UTC: the shared copy planner now also traces
  per-message descriptor synthesis candidates under `TRITON_DEBUG_TMEM_QUERY=1`.
  The scales descriptor-view row shows one candidate with row bases
  `[8,16,32,64,128]` and column bases `[1,2,256,4,512,1024,2048]`, confirming
  the current failure is the exact source descriptor projection and not an
  invisible verifier/lowering drift.
- 2026-04-15 10:02 UTC: `TRITON_DEBUG_TMEM_QUERY=1` now prints the
  source-to-destination copy conversion (`cvt`) after exact-query selection.
  For scales descriptor-view copy debugging this exposes the actual row/column
  basis mapping that the copy atom sees, instead of requiring temporary source
  edits to recover it.
- 2026-04-15 09:59 UTC: copy message plans now carry an optional
  descriptor-projection conversion separate from the full copy conversion.
  Existing plans leave it unset, so behavior is unchanged, but the planner can
  now represent future multi-message schedules where one message's
  shared-memory descriptor covers a legal source projection while the
  destination row/source offset schedule covers the full TMEM view. `TMemCopyPlan`
  and message vectors now use explicit inline counts to keep the larger plan
  objects build-safe.
- 2026-04-15 09:46 UTC: no-scales two-CTA `warpx2::02_13` was
  re-probed with direct-seed and forced-descriptor schedules and remains a
  real source/address-schedule gap, not a verifier over-restriction. Reusing
  the single-CTA direct seed for `cta_group::2` emitted the opcode but read
  zeros; forcing a canonical representable descriptor reached the opcode only
  after bypassing the multicast-row assertion, and then duplicated the
  low source-column pair (`[64, 192, 64, 192, ...]` at the right row offset)
  instead of preserving the high source-column bit. Keep the current clean
  negative until a multi-message or different source descriptor schedule is
  derived from ISA evidence.
- 2026-04-15 09:31 UTC: dense no-scales direct copy now admits the
  tile-permuted TMEM column layouts that the current descriptor schedule can
  prove, instead of rejecting all non-ascending column basis order. The support
  check is still proof-based: instruction-width logical column tiles must be
  aligned and contiguous in physical TMEM columns. Superseded by the
  2026-04-15 10:38 checkpoint for higher column macro-selector permutations:
  `tile_n=32` is now positive, while bit-scrambled column permutations remain
  clean unsupported.
- 2026-04-15 09:08 UTC: copy executable schedules can now carry TMEM
  destination row deltas and optional `tcgen05.cp` source-format suffixes
  through lowering. Defaults preserve existing support; the next task is to
  derive those schedule dimensions from exact `LinearLayout` facts for scales
  descriptor views and two-CTA `warpx2::02_13`.
- 2026-04-15 09:05 UTC: exact scales descriptor-view copy failures now attach
  a schedule-level note when the exact descriptor view changes the physical
  layout relative to the root scales layout. The diagnostic states that
  descriptor selection alone cannot realize the logical row permutation and
  that support requires a destination-row / source-message schedule.
- 2026-04-15 09:02 UTC: copy plan selection now returns a realized executable
  schedule. `TMemCopyPlanSelection` carries the selected family plus per-message
  descriptor-layout selection or direct-seed descriptor immediate, and LLVM
  lowering consumes that schedule instead of recomputing descriptor selection.
  This is intended behavior-preserving, but it removes another verifier/lowering
  drift point before adding destination-row/source-format schedule dimensions.
- 2026-04-15 08:54 UTC: exact scales descriptor-view copy probe showed that
  offset-major repacking of the failing `warpx4.32x128b` descriptor is not a
  correct support promotion. It synthesizes a valid MMAv5 shared descriptor,
  but runtime output is the parent/root physical row order:
  output row `R` reads source row `(R % 64) * 2 + R / 64`. The `.b8x16`
  source-format variants (`b6x16_p32`, `b4x16_p64`) compiled but preserved the
  same wrong logical permutation. Next implementation should model the missing
  logical-view-to-physical-destination schedule, not just make the source
  descriptor representable.
- 2026-04-15 08:35 UTC: scales-root semantics now survive descriptor-view
  chains in the shared physical-query model. Copy verification/lowering uses
  `TMemPhysicalQuery::isScales`, not just the surface destination encoding, so
  scale-backed linear descriptor views no longer fall into non-scales copy
  diagnostics.
- Copy lowering now has an origin-base-offset hook for exact physical queries.
  The immediate remaining scales-copy gap is atomizing the exact permuted
  descriptor-view projection; the current row is pinned as a clean
  tensor-memory-scales copy-family miss.
- 2026-04-15 08:42 UTC: copy planning now consumes the exact scales
  descriptor-view query when the semantic facts match. The same row now reaches
  `warpx4.32x128b` classification and is pinned at descriptor synthesis, so the
  next work is deriving a representable MMAv5 shared-memory descriptor for the
  exact permuted projection.

## Current Project Invariant

- `_reinterpret` is not an escape hatch for relying on whichever physical TMEM
  lowering the compiler happens to choose.
- Production kernels, examples, and tests that currently depend on private
  `_reinterpret` behavior or compiler-specific physical-layout knowledge are
  migration targets.
- The intended replacement is a supported descriptor operation sequence:
  offset to the right TMEM region, slice/subview to the desired physical bits,
  then bitcast to the desired dtype/shape/layout only when total bit size and
  the exact physical TMEM mapping are preserved.
- Operationally, the offset/subview step selects the actual physical TMEM bits
  first; the bitcast only changes the descriptor's dtype/shape/layout view over
  that already-selected physical image.
- This means the bitcast may express the desired dtype/shape/layout only after
  the input descriptor already denotes exactly the same physical TMEM image; it
  must not perform physical selection itself.
- In the concrete attention-style pattern, the kernel first offsets to the
  right part of TMEM, then takes the subview that denotes the exact desired
  bits, then performs a size-and-physical-mapping-equivalent bitcast to the
  desired dtype, shape, and layout.
- Equivalently: offset to the right part of TMEM, slice/subview to the desired
  bit image, then bitcast only to a dtype/shape/layout view whose total size
  and physical mapping are equivalent to that sliced descriptor.
- The bitcast step must not change which physical TMEM memory the input
  descriptor maps to; if the desired view is not equal-size and
  physical-mapping equivalent, use a different supported API that matches the
  real kernel intent.
- Said another way, this is an offset/subview over the existing physical TMEM
  allocation followed by a size-and-physical-mapping-equivalent type/view
  reinterpretation, not a remapping or relocation operation.
- The attention example's scratch-borrow path has a focused supported
  migration: helpers now offset/slice the f32 scratch TMEM before bitcasting,
  and the exp2 partition stores slice the exact physical subregion before
  bitcasting each bf16 view. Preserve this synchronization-aware pattern if
  broader attention coverage exposes more aliasing cases.
- Lowering-side fixes are still appropriate for supported APIs that miscompile,
  but do not add ad-hoc selectors just to preserve old `_reinterpret`
  accidents.
- TMEM layouts are the source of truth. A zero TMEM basis means
  broadcast/equivalence semantics: physical coordinates that map to the same
  logical tensor element must agree, and codegen must preserve that invariant.
- Do not justify ld/st behavior as choosing a `live` representative among
  divergent physical cells. Divergence inside a zero-basis equivalence class is
  itself a bug in producer, view, or load/store lowering.
- The branch-only TMEM attributes (`ttng.tmem_physical_layout`,
  `ttng.tmem_ldst_row_plan`, `ttng.tmem_mmav5_accumulator_root`, and related
  root markers) are now treated as papering over underlying layout/codegen bugs.
  The next cleanup target is to remove them and make lowering respect the
  layout's broadcast and physical mapping directly.

## Current Checkpoint

- Current copy failure-note helper checkpoint, 2026-04-15 08:19 UTC:
  `attachTMemCopyPlanFailureNotes(...)` now lives in `TensorMemoryUtils` and is
  used by both `ttng.tmem_copy` verification and LLVM lowering. This keeps
  all-plan copy failure reporting owned by the shared planner utilities instead
  of duplicated in verifier/lowering call sites. Validation passed: `make -j8`,
  direct invalid/conversion RUN lines via local `triton-opt` and
  `python/triton/FileCheck`, focused `cp_scales and clean` pytest slice
  (`8 passed`), and `git diff --check`.

- Current 4x256b cta-group::2 coverage checkpoint, 2026-04-15 07:39 UTC:
  local implementation already handled `tcgen05.cp.cta_group::2.4x256b` through
  the dense 4-row copy atom, and conversion coverage now exercises that ISA
  variant with a two-CTA shared/tmem linear layout. Validation passed:
  `make -j8` from the preceding code checkpoint, direct Blackwell conversion
  RUN via local `triton-opt` and `python/triton/FileCheck`, direct invalid RUN,
  and `git diff --check`.

- Current all-plan failure evidence checkpoint, 2026-04-15 07:36 UTC:
  `TMemCopyPlanSelection` now retains all structured failed plan-support
  results, and verifier/lowering diagnostics attach each non-empty failure
  note. Dense fallback and two-CTA `warpx2::02_13` clean-negatives now report
  both attempted descriptor schedules. Two support-promotion probes were
  intentionally not kept: applying warpx2 descriptor variants to scales
  `warpx4` only expanded the failed search to 188 candidates, and reusing the
  single-CTA `02_13` direct seed for cta-group::2 compiled but produced zeroed
  output. Validation passed: `make -j8`, direct invalid/conversion RUN lines
  via local `triton-opt` and `python/triton/FileCheck`, focused
  `cp_scales and clean` pytest slice (`8 passed`), two representative
  no-scales copy rows (`2 passed`), and `git diff --check`.

- Current descriptor-layout selection checkpoint, 2026-04-15 07:28 UTC:
  descriptor synthesis now exposes `selectTMemCopyDescriptorLayout(...)`,
  returning the exact shared descriptor layout and MN orientation selected for
  each copy message. The support checker and LLVM lowering consume the same
  selection helper, reducing drift between "verifier says representable" and
  "lowering builds a loader". Validation passed: `make -j8`, direct
  invalid/conversion RUN lines via local `triton-opt` and
  `python/triton/FileCheck`, focused `cp_scales and clean` pytest slice
  (`8 passed`), two representative no-scales copy rows (`2 passed`), and
  `git diff --check`.

- Current unified copy-plan selector checkpoint, 2026-04-15 07:25 UTC:
  copy-plan support now has an explicit `TMemCopyPlanSupportKind` for
  no-scales tensor memory versus tensor-memory scales. Verification and
  lowering both call `selectTMemCopyPlan(...)`; no-scales mode preserves the
  existing destination-layout and shared-runtime preconditions, while scales
  mode uses the shared descriptor-synthesis checks without inheriting
  no-scales-only source-layout restrictions. Validation passed: `make -j8`,
  direct invalid/conversion RUN lines via local `triton-opt` and
  `python/triton/FileCheck`, focused `cp_scales and clean` pytest slice
  (`8 passed`), two representative no-scales copy rows (`2 passed`), and
  `git diff --check`.

- Current scales descriptor-failure diagnostics checkpoint, 2026-04-15
  07:22 UTC: the copy descriptor-synthesis support helper is now public to the
  verifier, and tensor-memory-scales copy clean-negatives surface the same
  per-plan MMAv5 descriptor evidence as no-scales copy. The representative
  scales `warpx4.32x128b` clean-negative now reports that message 0 tried one
  32x16 descriptor/instruction candidate with no representable MMAv5 shared
  descriptor before the existing user guidance notes. Validation passed:
  `make -j8`, direct invalid/conversion RUN lines via local `triton-opt` and
  `python/triton/FileCheck`, focused `cp_scales and clean` pytest slice
  (`8 passed`), and `git diff --check`.

- Current copy descriptor-failure diagnostics checkpoint, 2026-04-15 07:18 UTC:
  descriptor-synthesis support now returns a structured message identifying
  the failing copy family, message index, number of candidate descriptor
  layouts tried, descriptor shape, and instruction shape. The no-scales copy
  verifier/lowering selector surfaces that first failure in diagnostics, so
  hard gaps such as two-CTA `warpx2::02_13` now report planner evidence before
  the higher-level clean-unsupported notes. Validation passed: `make -j8`,
  direct invalid/conversion lit RUN lines via local `triton-opt` and
  `FileCheck`, one targeted runtime clean-negative `warpx2::02_13` root row,
  and `git diff --check`.

- Current copy plan selection checkpoint, 2026-04-15 07:15 UTC:
  no-scales `ttng.tmem_copy` verification and lowering now share
  `selectTMemCopyPlan(...)`, which returns the first supported plan plus the
  first structured failure. This keeps support/failure ordering in one utility
  and gives the next planner slice a concrete result object to extend with
  query-origin, descriptor-message, and schedule-failure details. Validation
  passed: `make -j8`, direct invalid/conversion lit RUN lines via local
  `triton-opt` and `FileCheck`, targeted positive
  `warpx2::01_23` two-CTA slice-index runtime row, targeted clean-negative
  `warpx2::02_13` two-CTA root row, and `git diff --check`.

- Current copy exact-query projection checkpoint, 2026-04-15 07:12 UTC:
  copy verification and lowering now use a copy-specific physical-query
  comparator when deciding whether an exact descriptor-view query may replace
  the standalone query. The comparator requires active shape, element width,
  physical layout, CTA ownership, and scales classification to match, but
  intentionally ignores origin and allocation-shape differences because copy
  lowering carries those through the descriptor base. This preserves existing
  support decisions while allowing origin-divergent descriptor views to flow
  through the exact query carrier when their copy-relevant projection matches.
  Validation passed: `make -j8`, direct invalid/conversion lit RUN lines via
  local `triton-opt` and `FileCheck`, one targeted runtime clean-negative
  `warpx2::02_13` slice-index row, and `git diff --check`.

- Current `tcgen05.cp.4x256b` ISA coverage checkpoint, 2026-04-15 07:08 UTC:
  no-scales `ttng.tmem_copy` now recognizes the dense 4-row, 256-bit copy
  atom, plans it as a single 4x8 shared descriptor for 32-bit elements, and
  lowers it to `tcgen05.cp.cta_group::1.4x256b`. Shared-descriptor support and
  lowering now agree on descriptor orientation: the 4x256b family may use the
  natural transposed 4x8 MMAv5 shared descriptor, while existing copy families
  retain the prior non-transposed descriptor path. Validation passed:
  `make -j8`, direct invalid/conversion lit RUN lines via local `triton-opt`
  and `FileCheck`, and `git diff --check`.

- Current guarded exact-query copy checkpoint, 2026-04-15 06:57 UTC:
  copy verification and no-scales lowering now compute the exact destination
  physical query and use it only when it is equal to the standalone projection.
  Divergent or failed exact queries keep the current standalone behavior, and
  the existing `TRITON_DEBUG_TMEM_QUERY` divergence reporting remains available.
  Validation passed: `make -j8`, direct invalid/conversion lit RUN lines via
  local `triton-opt` and `FileCheck`, and `git diff --check`.

- Current copy lowering alignment checkpoint, 2026-04-15 06:55 UTC:
  `TensorMemoryToLLVM.cpp` now uses `inferStandaloneTMemPhysicalQuery(...)` and
  `getTMemCopyPlanSupport(...)` for no-scales `tcgen05.copy` lowering, matching
  the verifier's physical-query/per-plan support path. This is intended to be
  behavior-preserving and keeps the scales lowering path on its current
  descriptor-synthesis flow. Validation passed: `make -j8`, direct
  invalid/conversion lit RUN lines via local `triton-opt` and `FileCheck`, and
  `git diff --check`.

- Current copy per-plan support checkpoint, 2026-04-15 06:53 UTC:
  `getTMemCopyPlanSupport(...)` now returns one structured support result for a
  copy plan by checking destination physical-query support,
  shared-layout/runtime support, and shared-descriptor synthesis in planner
  order. `ttng.tmem_copy` verification now asks that helper for no-scales plan
  support. Existing diagnostics remain stable; descriptor-synthesis failures
  are classified structurally but do not add a new note yet. Validation passed:
  `make -j8`, direct invalid/conversion lit RUN lines via local `triton-opt`
  and `FileCheck`, and `git diff --check`.

- Current copy support-result checkpoint, 2026-04-15 06:51 UTC:
  copy support checks now have `TMemCopySupportResult` plus
  `TMemCopySupportFailureLayer` so destination physical-query failures and
  shared-layout/runtime failures can be represented as structured planner
  results instead of bool/string side channels. `ttng.tmem_copy` verification
  consumes these results while preserving existing diagnostics and behavior.
  Direct destination support now evaluates the `TMemPhysicalQuery` layout at
  the verifier call site. Validation passed: `make -j8`, direct
  invalid/conversion lit RUN lines via local `triton-opt` and `FileCheck`, and
  `git diff --check`.

- Current copy query debug-consumer checkpoint, 2026-04-15 06:48 UTC:
  `ttng.tmem_copy` verification now computes the exact destination physical
  query under `TRITON_DEBUG_TMEM_QUERY` and reports exact-vs-standalone
  divergence with the first differing query field plus standalone/exact layouts
  and origins. Normal verifier behavior and diagnostics remain unchanged.
  Validation passed: `make -j8`, direct invalid/conversion lit RUN lines via
  local `triton-opt` and `FileCheck`, `TRITON_DEBUG_TMEM_QUERY=1` invalid-file
  smoke, and `git diff --check`.

- Current physical-query comparison checkpoint, 2026-04-15 06:44 UTC:
  `TMemPhysicalQueryDifference`, `getFirstTMemPhysicalQueryDifference(...)`,
  `haveSameTMemPhysicalQueryProjection(...)`, and
  `stringifyTMemPhysicalQueryDifference(...)` now provide a shared vocabulary
  for comparing exact and standalone physical queries. This is side-effect-free
  scaffolding; no verifier diagnostics or support decisions consume it yet.
  Validation passed: `make -j8`, direct invalid/conversion lit RUN lines via
  local `triton-opt` and `FileCheck`, and `git diff --check`.

- Current copy-query routing checkpoint, 2026-04-15 06:42 UTC:
  direct TMEM copy layout support now has a `TMemPhysicalQuery` overload, and
  `ttng.tmem_copy` verification calls that overload for destination layout
  support. The overload deliberately delegates to the existing memdesc-type
  implementation, so support decisions and diagnostics remain unchanged while
  another copy-verifier decision moves onto the physical-query carrier.
  Validation passed: `make -j8`, direct invalid/conversion lit RUN lines via
  local `triton-opt` and `FileCheck`, and `git diff --check`.

- Current exact physical-query API checkpoint, 2026-04-15 06:39 UTC:
  `inferExactTMemPhysicalQuery(...)` now exists beside the standalone physical
  query. The exact path validates the descriptor, reuses the existing
  descriptor-view `inferStandaloneTMemLdStQueryLayout(...)` algebra, and returns
  a `TMemPhysicalQuery` carrying non-zero origins when the view chain has them.
  Copy verification intentionally remains on the standalone query path for this
  checkpoint; the new exact API is the next seam for replacing type-only
  descriptor-view fallback with explicit physical-query decisions. Validation
  passed: `make -j8`, direct invalid/conversion lit RUN lines via local
  `triton-opt` and `FileCheck`, and `git diff --check`.

- Current physical-query shape-facts checkpoint, 2026-04-15 06:37 UTC:
  `TMemPhysicalQuery` now carries explicit active shape, allocation shape, and
  element bitwidth in addition to the standalone memdesc type, physical layout,
  `twoCTAs`, origin, and scales classification. This keeps common planner facts
  available without repeatedly unpacking `MemDescType` and prepares the next
  step: distinguishing exact descriptor-view origin from standalone type
  fallback. Validation passed: `make -j8`, direct invalid/conversion lit RUN
  lines via local `triton-opt` and `FileCheck`, and `git diff --check`.

- Current physical-query API checkpoint, 2026-04-15 06:34 UTC:
  `TMemPhysicalQuery` now exists as the first shared carrier for standalone
  physical TMEM view facts: standalone memdesc type, physical `LinearLayout`,
  `twoCTAs`, zero origin for the current standalone path, and scales-vs-linear
  classification. `inferStandaloneTMemPhysicalQuery(...)` wraps the existing
  standalone view-type inference and `ttng.tmem_copy` now consumes that query
  instead of separately asking for a standalone type and then reconstructing
  `toLinearLayout`. This is intentionally behavior-preserving; it creates the
  Phase 1 API seam for exact physical queries before changing support
  decisions. Validation passed: `make -j8`, direct invalid/conversion lit RUN
  lines via local `triton-opt` and `FileCheck`, and `git diff --check`.

- Current backend-completion planning checkpoint, 2026-04-15 06:15 UTC: the
  expanded runtime matrix is now the correctness anchor for backend work rather
  than the end goal. Preserve the conclusion that the remaining hard gaps
  expose an incomplete linear-layout backend: scales and no-scales two-CTA
  `warpx2::02_13` still lower through under-specified physical query/planner
  machinery, while `tcgen05.cp ... 4x256b` is a separate missing ISA-coverage
  item that should be easier to add once copy lowering is atomized through the
  shared planner. The multi-phase plan is recorded in
  `backend_completion_plan.md`; the first implementation slice centralizes copy
  descriptor-plan representability in `TensorMemoryUtils` so
  `ttng.tmem_copy` verifier logic can evolve toward a shared planner instead
  of carrying duplicate inline descriptor synthesis checks. Validation passed:
  `make -j8`, direct invalid/conversion lit RUN lines via local `triton-opt`
  and `FileCheck`, and `git diff --check`.

- Current `ld.red` explicit pure-row permutation checkpoint, 2026-04-15 05:08 UTC: descriptor-chain and direct explicit `ld.red` N-sweep variants now cover pure row `row_rotate1` and `row_even_odd` layouts at `N in {32,64,256}` for `32x32b`, `16x32bx2`, and `32x32b_splitn`, crossed with min/max and all legal `abs`/`NaN` modifier modes. These rows use the canonical split offsets; the special `N=256` non-`32x32b` split offset table remains limited to `col_reverse`, `col_rotate1`, and `rowcol_rotate_reverse`. Runtime-matrix collection is now `9834` tests: `cp=677`, `mma=2783`, splitn/misc `=571`, `ld_red=2818`, and `ldst=2985`; bucketed evidence aggregates to `9383 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `9834`; `ld_red` collect `2818`; affected pure-row explicit selector collect `288/9834`; four-GPU split execution passed all `288` affected cases (`72` per group; slowest `1601.45s`). Tooling note: pytest in this shell needed `PYTHONPATH=./python` after rebuild because a no-PYTHONPATH collect imported a stale installed Triton.

- Current `ld.red` explicit pure-column permutation checkpoint, 2026-04-14 17:45 UTC: descriptor-chain and direct explicit `ld.red` N-sweep variants now cover pure column `col_rotate1` and `col_even_odd` layouts at `N in {32,64,256}` for `32x32b`, `16x32bx2`, and `32x32b_splitn`, crossed with min/max and all legal `abs`/`NaN` modifier modes. The `col_rotate1`, `N=256`, split explicit variants use offset order `[0,128,64,192]`, while `col_even_odd` keeps canonical `[0,64,128,192]`; `32x32b` remains canonical. Runtime-matrix collection is now `9546` tests: `cp=677`, `mma=2783`, splitn/misc `=571`, `ld_red=2530`, and `ldst=2985`; bucketed evidence aggregates to `9095 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `9546`; `ld_red` collect `2530`; affected pure-column explicit selector collect `288/9546`; four-GPU split execution passed all `288` affected cases (`72` per group; slowest `1542.86s`).

- Current scaled two-CTA MMAv5 `block_n=64` accumulator-subview checkpoint, 2026-04-14 17:14 UTC: the previously parked two-CTA scaled accumulator-subview `block_n=64`, `parent_n=128` surface is now positive. `mma_scaled_tcgen05_acc_subslice_copy_kernel` now derives scale tile extents from the packed scale descriptors, allocates/copies padded 128-row scale TMEM images when an active scale tile is narrower than the descriptor minimum, and the test matrix now covers `block_n in {64,128}` with `parent_n=2*block_n`. Coverage spans every current scaled format pair, `block_k in {128,256}`, both slice starts, multicast on/off, and both zero-initialized plus nonzero accumulator-add paths. Runtime-matrix collection is now `9258` tests: `cp=677`, `mma=2783`, splitn/misc `=571`, `ld_red=2242`, and `ldst=2985`; bucketed evidence aggregates to `8807 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `9258`; affected scaled two-CTA accumulator-subview selector collect `160/9258`; four-GPU split execution passed all `160` affected cases (`40` per group; slowest `38.45s`).

- Current `ld.red` descriptor-chain full mixed auto N-sweep checkpoint, 2026-04-14 17:09 UTC: auto-selected descriptor-chain reductions now cover the remaining mixed row/column permutation families `rowcol_rotate_rotate1`, `rowcol_rotate_even_odd`, `rowcol_even_odd_rotate1`, and `rowcol_even_odd_even_odd`. Together with the prior two `ld.red` commits, the descriptor-chain N-width `auto` sweep now covers identity, tile-permuted, and the full non-identity row/column permutation cross-product at `N in {32,64,256}`. This is test-only ISA saturation for `auto`; explicit compatible variants remain the bounded identity/tile/reverse-family matrix. Runtime-matrix collection is now `9178` tests: `cp=677`, `mma=2703`, splitn/misc `=571`, `ld_red=2242`, and `ldst=2985`; bucketed evidence aggregates to `8727 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `9178`; final mixed auto selector collect `96/9178`; four-GPU split execution passed all `96` cases (`24` per group; slowest `429.64s`).

- Current `ld.red` descriptor-chain mixed reverse-row N-sweep checkpoint, 2026-04-14 17:01 UTC: auto-selected descriptor-chain reductions now cover the three `row_reverse` mixed row/column families: `rowcol_reverse_rotate1`, `rowcol_reverse_even_odd`, and `rowcol_reverse_reverse`. Each runs through the existing generic `slice`/`index`/`reshape` descriptor-chain helper at `N in {32,64,256}`, crossed with `min`/`max` and all legal `abs`/`NaN` modifier modes. This is test-only ISA saturation for `auto` reduction-layout selection; explicit variants for these mixed reverse-row layouts remain unclaimed. Runtime-matrix collection is now `9082` tests: `cp=677`, `mma=2703`, splitn/misc `=571`, `ld_red=2146`, and `ldst=2985`; bucketed evidence aggregates to `8631 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `9082`; new mixed reverse-row selector collect `72/9082`; four-GPU split execution passed all `72` cases (`18` per group; slowest `346.47s`).

- Current `ld.red` descriptor-chain permutation N-sweep checkpoint, 2026-04-14 16:52 UTC: auto-selected descriptor-chain reductions now cover the remaining row/column permutation primitives beside the earlier identity, tile-permuted, reverse, and rotate/reverse rows. `LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_CASES` now includes `col_rotate1`, `col_even_odd`, `row_rotate1`, `row_even_odd`, and `rowcol_even_odd_reverse` at `N in {32,64,256}`, crossed with `min`/`max` and all legal `abs`/`NaN` modifier modes through the generic `slice`/`index`/`reshape` descriptor chain. This is test-only ISA saturation for the `auto` reduction-layout path; explicit variant coverage remains the existing bounded identity/tile/reverse-family matrix. Runtime-matrix collection is now `9010` tests: `cp=677`, `mma=2703`, splitn/misc `=571`, `ld_red=2074`, and `ldst=2985`; bucketed evidence aggregates to `8559 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `9010`; non-explicit descriptor-chain N-sweep collect `240/9010`; new-layout selector collect `120/9010`; four-GPU split execution passed all `120` new-layout cases (`30` per group; slowest `572.00s`).

- Current plain MMAv5 TMEM-LHS subview K-depth checkpoint, 2026-04-14 16:39 UTC: plain TMEM-LHS `memdesc_subslice` descriptor-view coverage now reaches `K=128`. `MMA_LHS_SUBSLICE_NK_CASES` now covers every supported plain kind, legacy/canonical accumulator layouts, `N in {64,128,256}`, and `K in {32,64,128}` for both zero-accumulator and `use_acc=True` paths through a `[128,2*K]` LHS parent. Runtime-matrix collection is now `8890` tests: `cp=677`, `mma=2703`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2985`; bucketed evidence aggregates to `8439 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `8890`; focused `mma_lhs_subslice_view_plain_kinds` collect `180/8890`; four-GPU split execution passed all `180` cases (`45` per group; slowest `86.02s`).

- Current one-CTA plain MMAv5 indexed-accumulator K-depth checkpoint, 2026-04-14 16:36 UTC: one-CTA accumulator `memdesc_index` descriptor-view coverage now reaches `K=128`, matching the two-CTA indexed slice. `MMA_INDEXED_ACC_CASES` now covers every supported plain kind, `N in {64,128,256}`, `K in {32,64,128}`, and both accumulator modes through legacy parents, canonical linear parents where the live `[2,128,N]` image fits, and canonical linear unit parents for `N=256`. Runtime-matrix collection is now `8830` tests: `cp=677`, `mma=2643`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2985`; bucketed evidence aggregates to `8379 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `8830`; focused one-CTA `mma_indexed_acc_view` collect `180/8830`; four-GPU split execution passed all `180` cases (`45` per group; slowest `96.47s`).

- Current TMA-fed two-CTA plain MMAv5 K-depth checkpoint, 2026-04-14 16:32 UTC: descriptor-fed two-CTA `tcgen05.mma` coverage now reaches `K=128`. `MMA_TWOCTA_TMA_NON_TF32_CASES` covers f16, bf16, f8e5m2, and f8e4m3 at `N in {64,128,256}`, `K in {32,64,128}`, legacy/canonical two-CTA accumulators, and both accumulator modes. `MMA_TWOCTA_TMA_TF32_CASES` now also covers `K=128` for the default clean-negative transposed-shared route and the positive B-transposed descriptor route, including `use_acc=True`. Runtime-matrix collection is now `8770` tests: `cp=677`, `mma=2583`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2985`; bucketed evidence aggregates to `8319 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `8770`; exact TMA-fed node-id collect `198`; four-GPU split execution passed all `198` TMA-fed cases (`50`, `50`, `50`, and `48`; slowest `60.96s`).

- Current two-CTA plain MMAv5 accumulator-subview K-depth checkpoint, 2026-04-14 16:29 UTC: two-CTA accumulator `memdesc_subslice` descriptor-view coverage now reaches `K=128`. `MMA_TWOCTA_ACC_SUBSLICE_CASES` now covers every supported plain kind, `N in {64,128,256}`, `K in {32,64,128}`, both slice starts, and both accumulator modes through linear two-CTA `[256,2*N]` parents. This is test-only descriptor-view saturation and closes the direct two-CTA plain root/index/subslice K128 trio; TMA-fed matrices and scaled-MMAv5 remain separate future K-depth slices. Runtime-matrix collection is now `8704` tests: `cp=677`, `mma=2517`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2985`; bucketed evidence aggregates to `8253 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `8704`; focused `mma_twocta_acc_subslice_view_plain_kinds` collect `180/8704`; four-GPU split execution passed all `180` cases (`45` per group; slowest `57.05s`).

- Current two-CTA plain MMAv5 indexed-accumulator K-depth checkpoint, 2026-04-14 16:27 UTC: direct two-CTA accumulator `memdesc_index` descriptor-view coverage now reaches `K=128`. `MMA_TWOCTA_INDEXED_ACC_CASES` now covers every supported plain kind, `N in {64,128,256}`, `K in {32,64,128}`, and both accumulator modes through legacy parents, canonical linear parents where the live `[2,256,N]` image fits, and canonical linear unit parents for `N=256`. This is a test-only descriptor-view saturation slice; two-CTA accumulator subviews, TMA-fed matrices, and scaled-MMAv5 remain separate future K-depth slices. Runtime-matrix collection is now `8644` tests: `cp=677`, `mma=2457`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2985`; bucketed evidence aggregates to `8193 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `8644`; focused `mma_twocta_indexed_acc_view` collect `180/8644`; four-GPU split execution passed all `180` cases (`45` per group; slowest `111.01s`).

- Current two-CTA plain MMAv5 K-depth checkpoint, 2026-04-14 16:22 UTC: direct two-CTA plain `tcgen05.mma` root coverage now reaches `K=128` beside the one-CTA root and accumulator-subview K-depth expansions. `MMA_TWOCTA_PLAIN_KIND_CASES` now covers every supported plain kind, legacy/canonical two-CTA TMEM-linear accumulators, `N in {64,128,256}`, and `K in {32,64,128}`; both no-accumulator and `use_acc=True` tests consume the same expanded matrix. This is test-only ISA saturation over already-supported lowering and stays scoped to direct two-CTA root accumulators; two-CTA indexed accumulator views, two-CTA accumulator subviews, TMA-fed paths, and scaled-MMAv5 remain separate future slices. Runtime-matrix collection is now `8584` tests: `cp=677`, `mma=2397`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2985`; bucketed evidence aggregates to `8133 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `8584`; focused `mma_twocta_plain_kinds` collect `180/8584`; broader `-k 'mma and not cp'` collect selected `2632/8584` because it also matches MMAv5-flavored `ld/st` parameter ids; four-GPU split execution of `mma_twocta_plain_kinds` passed all `180` cases (`45` per group; slowest `102.76s`); `git diff --check` passed before docs.

- Current copy `warpx2` slice-plus-index checkpoint, 2026-04-14 16:14 UTC: no-scales `tcgen05.cp` now supports leading-unit `ttg.memdesc_subslice` views followed by `ttg.memdesc_index` for the reachable public `warpx2` layouts. The compiler fix recognizes leading logical dimensions sliced to unit length, converts the selected logical bits into the TMEM query origin/base offset, squeezes the consumed dimension, and removes only the exact inactive physical bases for that consumed dimension so unrelated zero bases keep their broadcast semantics. Runtime coverage adds single-CTA `warpx2::{01_23,02_13}.64x128b` through `[2,128,4].slice(parent_index,1,dim=0).index(0)`, two-CTA `warpx2::01_23.64x128b` through `[2,256,4].slice(parent_index,1,dim=0).index(0)`, and clean-negative two-CTA `warpx2::02_13.64x128b`, all for `parent_index in {0,1}` and `f32`/`i32`. `test/TritonNvidiaGPU/ops.mlir` was updated to reflect the new squeezed encoding for a leading-unit subslice. Runtime-matrix collection is now `8524` tests: `cp=677`, `mma=2337`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2985`. Validation: `make -j8`; py-compile; full-file collect `8524`; `slice_index_view` collect `16/8524`; `warpx2` collect `83/8524`; `cp_no_scales` collect `310/8524`; four-GPU `slice_index_view` selector passed all `16`; four-GPU `warpx2` selector passed all `83`; four-GPU `cp_no_scales` selector passed/skipped `300 passed, 10 skipped`; lit `invalid.mlir`, `ops.mlir`, and `tritongpu_to_llvm_blackwell.mlir` passed; `git diff --check` passed. Remaining hard copy frontiers are unchanged: true tensor-memory-scales `warpx2` and no-scales two-CTA `warpx2::02_13` still need a real descriptor/address/staging model before positive support.

- Current copy `warpx2` indexed-view checkpoint, 2026-04-14 15:54 UTC: no-scales `tcgen05.cp` now has explicit generic `ttg.memdesc_index` descriptor-view coverage for the reachable public warp2x layouts. Single-CTA `warpx2::{01_23,02_13}.64x128b` copies run through `[2,128,4].index(parent_index)` with `parent_index in {0,1}`, `f32`/`i32` payloads, exact copy and commit opcodes, and runtime oracles. Two-CTA `warpx2::01_23.64x128b` now has the analogous `[2,256,4].index(parent_index)` positive coverage with exact multicast commit checks and no `cta_group::1` leakage. Two-CTA `warpx2::02_13.64x128b` indexed views are pinned as clean unsupported with the same high-source-column-bit diagnostic as the direct and subview forms. This replaces the older scratch note that direct lifted `warpx2` indexing was not a current positive target; direct outer `memdesc_index` is now covered; the slice-plus-index frontier is superseded by the 16:14 checkpoint above, while true tensor-memory-scales `warpx2` remains a separate frontier. Runtime-matrix collection is now `8508` tests: `cp=661`, `mma=2337`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2985`; bucketed evidence aggregates to `8057 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `8508`; `warpx2` collect `67/8508`; CP collect `661/8508`; four-GPU `warpx2` selector passed all `67` cases (`17`, `17`, `17`, and `16`; slowest `67.26s`); four-GPU CP selector passed/skipped `651 passed, 10 skipped` (`156 passed/10 skipped`, `166 passed`, `166 passed`, and `163 passed`; slowest `225.83s`); `git diff --check` passed.

- Current plain MMAv5 accumulator-subview K-depth checkpoint, 2026-04-14 15:46 UTC: one-CTA plain MMAv5 accumulator `memdesc_subslice` descriptor-view coverage now reaches `K=128` beside the direct-root K-depth expansion. `MMA_ACC_SUBSLICE_CASES` covers every supported plain kind, `N in {64,128,256}`, `K in {32,64,128}`, both slice starts, and both accumulator modes through linear `[128,2*N]` parents. `MMA_M64_ACC_SUBSLICE_CASES` now covers the same K axis for M64 `[64,2*N]` parent subviews. This remains scoped to one-CTA descriptor subviews and M64 descriptor subviews; indexed views, two-CTA descriptor views, TMA-fed paths, and scaled-MMAv5 stay at their prior K widths. Runtime-matrix collection is now `8492` tests: `cp=645`, `mma=2337`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2985`; bucketed evidence aggregates to `8041 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `8492`; adjacent accumulator-subslice collect `360/8492`; four-GPU split execution passed all `360` cases (`90` per group; slowest `89.53s`).

- Current plain MMAv5 K-depth checkpoint, 2026-04-14 15:39 UTC: one-CTA plain `tcgen05.mma` instruction coverage now reaches `K=128` for the direct root surfaces that previously stopped at `K in {32,64}`. `MMA_PLAIN_KIND_ACC_CASES` now covers every supported plain kind, legacy/canonical TMEM-linear accumulators, `N in {64,128,256}`, and `K in {32,64,128}` for both no-accumulator and `use_acc=True` tests. `MMA_M64_PLAIN_KIND_CASES` now covers the same K-depth expansion for `blockM=64` root accumulators. Tile-permuted accumulator coverage now includes `K=128` at `128x128/tile_n=32` and `128x256/tile_n=64`, including all plain kinds and the accumulator-add path. This is test-only ISA saturation over already-supported lowering; descriptor-view indexed/subslice and two-CTA/TMA matrices remain at their existing K widths. Runtime-matrix collection is now `8372` tests: `cp=645`, `mma=2217`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2985`; bucketed evidence aggregates to `7921 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `8372`; adjacent MMA selector collect `426/8372`; four-GPU split execution of that selector passed all `426` cases (`107`, `107`, `107`, and `105`; slowest `154.90s`).

- Current scales `ld/st` descriptor-view checkpoint, 2026-04-14 15:32 UTC: default/no-CGA `TensorMemoryScalesLayout()` allocations still reshape through the TMEM descriptor-view algebra into canonical `tensor_memory_linear` views for `ld/st`, with runtime positives at `(M,N) in {(128,32),(128,64),(256,64)}` and exact root-scales plus linear-view opcode assertions. The CGA follow-up exposed an unsound two-CTA int8 type-only fallback for descriptor views with support/broadcast bases: two `256x*` probes compiled but loaded/stored the wrong physical columns before this checkpoint. The branch now derives scales `twoCTAs` from CGA layout in query/view analysis and rejects/avoids the type-only fallback for those exact-query-required descriptor views in both verifier and frontend layout selection. New clean-negative coverage pins `(128,64)`, `(256,32)`, and `(256,64)` non-default-CGA scales descriptor views as unsupported for explicit `32x32b` until a real physical-support projection exists. Runtime-matrix collection is now `8230` tests: `cp=645`, `mma=2075`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2985`; bucketed evidence aggregates to `7779 passed, 451 skipped`. Validation: `make -j8`; py-compile; affected lit (`ops.mlir`, `invalid.mlir`, and `tritongpu_to_llvm_blackwell.mlir`) passed; focused `ldst_scales_descriptor_view` split-4 selector passed all six non-empty cases; broader `ldst_scales` split-4 selector passed all `101`; scaled-MMA smoke passed all nine selected non-empty cases; `git diff --check` passed after docs. Boundaries unchanged: scales multibuffering, true scales `warpx2`, no-scales two-CTA `warpx2::02_13`, low-shape row-anchor cases, and positive two-CTA/CGA scales descriptor views remain outside the supported set.

- Current copy `warpx2` two-CTA subview checkpoint, 2026-04-14 14:57 UTC: no-scales `tcgen05.cp.cta_group::2.warpx2::01_23.64x128b` now has executable generic `ttg.memdesc_subslice` descriptor-view coverage from a wider two-CTA `[256,8]` TMEM-linear parent sliced to the active `[256,4]` view at column starts `0` and `4`, for `f32`/`i32`. The matching `warpx2::02_13` two-CTA parent subviews are pinned as clean unsupported with the same high-source-column-bit diagnostic as the direct candidate, proving subviews do not bypass the descriptor/address frontier. Runtime-matrix collection is now `8224` tests: `cp=645`, `mma=2075`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2979`; bucketed evidence aggregates to `7773 passed, 451 skipped`. Validation: `make -j8`; py-compile; focused collect `8/8224`; full `warpx2` collect `51/8224`; CP collect `645/8224`; full-file collect `8224`; focused split-4 selector passed all `8`; full `warpx2` split-4 selector passed all `51`; full CP split-4 selector passed/skipped `635 passed, 10 skipped`; `git diff --check` passed. Scratch finding: lifted `warpx2` indexed views currently fail in generic view inference before copy lowering, so that is a separate view-algebra gap rather than a support claim for this slice.

- Current two-CTA no-scales copy indexed-view checkpoint, 2026-04-14 14:49 UTC: dense `tcgen05.cp.cta_group::2.128x256b` descriptor-view coverage now also exercises generic `ttg.memdesc_index` over a resource-safe `[2,256,N].index(1)` parent for `N in {64,128}`. `CP_TWOCTA_LINEAR_INDEXED_VIEW_CASES` covers `f32`/`i32`, swizzle byte widths `{32,64,128}`, the two-CTA shared-layout/CGA path, cluster fence, multicast mbarrier commit, exact copy opcode counts (`8` for `N=64`, `16` for `N=128`), and absence of `cta_group::1` or legacy tensor-memory encodings. The `N=256` depth-two parent is not a positive target because it requires 1024 TMEM columns against the 512-column hardware limit; `test_tmem_runtime_matrix_cp_no_scales_twocta_linear_indexed_view_full_256x256_reports_tmem_oor` pins that boundary. Runtime-matrix collection is now `8216` tests: `cp=637`, `mma=2075`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2979`; bucketed evidence aggregates to `7765 passed, 451 skipped`. Validation: `make -j8`; py-compile; focused collect `13/8216`; CP collect `637/8216`; full-file collect `8216`; focused split-4 selector passed all `13` cases; full CP split-4 selector passed/skipped `627 passed, 10 skipped` across four GPUs; `git diff --check` passed.

- Current two-CTA no-scales copy subview checkpoint, 2026-04-14 14:44 UTC: dense `tcgen05.cp.cta_group::2.128x256b` descriptor-view coverage now mirrors the one-CTA column-subslice surface. `CP_TWOCTA_LINEAR_SUBSLICE_VIEW_CASES` covers `f32`/`i32`, `M=256`, `N in {128,256}`, swizzle byte widths `{32,64,128}`, and a `[M,2*N]` canonical TMEM-linear two-CTA parent sliced to the active column half. The new helper uses the two-CTA shared-layout/CGA path, cluster fence, multicast mbarrier commit, exact copy opcode counts (`16` for `N=128`, `32` for `N=256`), and asserts the generic `ttg.memdesc_subslice` chain survives instead of falling back to legacy `ttng.tmem_subslice`. Runtime-matrix collection is now `8203` tests: `cp=624`, `mma=2075`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2979`; bucketed evidence aggregates to `7752 passed, 451 skipped`. Validation: `make -j8`; py-compile; focused collect `12/8203`; CP collect `624/8203`; full-file collect `8203`; focused split-4 selector passed all `12` cases (`3` per group); full CP split-4 selector passed/skipped `614 passed, 10 skipped` across four GPUs; `git diff --check` passed.

- Current `ld.red` descriptor-chain non-f32 checkpoint, 2026-04-14 14:37 UTC: the f32-only reduction contract now covers generic memdesc descriptor-chain sources in addition to direct TMEM sources. `LD_RED_NON_F32_DESCRIPTOR_CHAIN_CASES` adds i32/bf16/f16/i16/i8 plain descriptor-view reductions plus bf16/f16 `NaN` and `abs` modifier diagnostics, crossed with `min` and `max`, all asserting the same public diagnostics without PassManager/assertion noise. The descriptor-chain kernel intentionally receives the base 2D TMEM-linear layout, matching existing positive `ld.red` descriptor-chain tests; an initial lifted-layout probe aborted during `get_reg_layout`, confirming that manual lifting is the wrong construction for this helper. Runtime-matrix collection is now `8191` tests: `cp=612`, `mma=2075`, splitn/misc `=571`, `ld_red=1954`, and `ldst=2979`; bucketed evidence aggregates to `7740 passed, 451 skipped`. Validation: `make -j8`; py-compile; exact descriptor-chain non-f32 collect `18/8191`; `ld_red` collect `1954/8191`; full-file collect `8191`; corrected descriptor-chain non-f32 split-4 selector passed all `18` cases (`5`, `5`, `5`, and `3`).

- Current rank-5 descriptor `ld/st` minimal-N checkpoint, 2026-04-14 14:34 UTC: executable rank-5 descriptor-chain roundtrips now include the minimal `N=32` opcode family where the underlying layout is defined. The expanded rank-5 small matrix adds `N=32` for `single_identity`, `twocta_block`, and `twocta_mmav5` layouts across `f32`/`i32` and every public `ld/st` variant; `single_mixed,N=32` is intentionally omitted because the existing mixed TMEM-linear helper is only meaningful for `N >= 64`. `LDST_SHAPE_MAP` now contains the full-tile `N=32` opcode spellings used by these assertions. Runtime-matrix collection is now `8173` tests: `cp=612`, `mma=2075`, splitn/misc `=571`, `ld_red=1936`, and `ldst=2979`; bucketed evidence aggregates to `7722 passed, 451 skipped`. Validation: `make -j8`; py-compile; rank-5 collect `110/8173`; `ldst` collect `2979/8173`; full-file collect `8173`; corrected rank-5 split-4 selector passed all `110` cases (`28`, `28`, `28`, and `26`). An initial overbroad table attempt selected `single_mixed,N=32` and failed at the layout helper precondition, confirming that row is not a valid coverage target.

- Current `ld.red` non-f32 dtype-boundary checkpoint, 2026-04-14 14:25 UTC: the clean unsupported reduction contract now covers direct bf16/f16/i16/i8 source layouts beside the existing i32 and legacy f16-unpacked rows. Plain bf16/f16/i16/i8 reductions assert `tmem_load reduction currently requires f32 element type`, and bf16/f16 modifier probes assert the public `NaN` / `abs` f32 diagnostics, all without PassManager/assertion noise. Runtime-matrix collection is now `8143` tests: `cp=612`, `mma=2075`, splitn/misc `=571`, `ld_red=1936`, and `ldst=2949`; bucketed evidence aggregates to `7692 passed, 451 skipped`. Validation: `make -j8`; py-compile; exact non-f32 collect `24/8143`; exact non-f32 split-4 selector passed all `24` cases; `ld_red` collect `1936/8143`; full-file collect `8143`; `git diff --check` passed. Side probe: row-256 `ld.red` `N=256` remains omitted; the full reduction-only helper still exceeds shared memory (`262148 > 232448`), and a split-store helper hit an unsupported second-half descriptor-view layout.

- Current scaled MMAv5 indexed-accumulator unit-parent checkpoint, 2026-04-14 14:20 UTC: resource-safe scaled accumulator `memdesc_index` coverage now reaches canonical TMEM-linear `N in {128,256}` and legacy `N=256` by indexing `[1,M,N]` parents and reshaping the result back to the active 2D MMA descriptor. Coverage spans every current scaled format pair, `K in {128,256}`, and both zero-initialized plus nonzero accumulator-add paths. The older `[2,M,N]` linear `N>=128` and legacy `N=256` indexed rows remain resource-bound when scale descriptors are live; the unit-parent rows are the executable positive construction, not evidence that those live-parent shapes fit in TMEM. Runtime-matrix collection is now `8127` tests: `cp=612`, `mma=2075`, splitn/misc `=571`, `ld_red=1920`, and `ldst=2949`; bucketed evidence aggregates to `7676 passed, 451 skipped`. Validation: py-compile; exact N128 unit-parent probe passed; direct N256 linear/legacy unit-parent probes passed; unit-parent collect `60/8127`; unit-parent split-4 selector passed all `60` cases; full scaled-indexed collect `120/8127`; full scaled-indexed split-4 selector passed all `120` cases; tight MMA collect `2075/8127`; full-file collect `8127`; `git diff --check` passed.

- Current plain MMAv5 indexed-accumulator unit-parent checkpoint, 2026-04-14 14:10 UTC: resource-safe `memdesc_index` coverage now fills the canonical TMEM-linear `N=256` gap for one-CTA and two-CTA plain MMAv5 accumulator views. The helper kernels now parameterize parent depth/index and reshape the indexed view back to the active 2D MMA descriptor, so `[1,M,N].index(0)` and `[1,256,N].index(0)` cover `N=256` without keeping the older `[2,M,N]` live parent image. Coverage spans every supported plain kind, `K in {32,64}`, and `use_acc in {False,True}` for both CTA groups; the existing `[2,M,N]` canonical-linear `N=256` rows remain a hardware resource boundary because they require 1024 TMEM columns. Runtime-matrix collection is now `8067` tests: `cp=612`, `mma=2015`, splitn/misc `=571`, `ld_red=1920`, and `ldst=2949`; bucketed evidence aggregates to `7616 passed, 451 skipped`. Validation: py-compile; exact one-case probe passed; exact unit-parent collect `40/8067`; exact unit-parent split-4 selector passed all `40` cases; adjacent non-scaled indexed-accumulator collect `240/8067`; adjacent split-4 selector passed all `240` cases; tight MMA collect `2015/8067`; full-file collect `8067`; `git diff --check` passed.

- Current copy `warpx2` subslice-view checkpoint, 2026-04-14 14:01 UTC: standalone TMEM view reconstruction now preserves semantic row/column zero bases and pure 2D column `memdesc_subslice` views over non-surjective layouts. Pure column subview pointer lowering uses ld/st query-origin deltas rather than pseudoinverting the non-surjective parent layout. Runtime coverage now includes single-CTA no-scales `warpx2::{01_23,02_13}` copied through a `128x8` parent sliced to `128x4` at column starts `0` and `4`, for `f32` and `i32`, with exact `tcgen05.cp.cta_group::1.warpx2::*` opcodes and surviving `ttg.memdesc_subslice`. Runtime-matrix collection is now `8027` tests: `cp=612`, `mma=1975`, splitn/misc `=571`, `ld_red=1920`, and `ldst=2949`; bucketed evidence aggregates to `7576 passed, 451 skipped`. Validation: `make -j8`; scratch `01_23`/`02_13` subslice probe; py-compile; exact collect `8/8027`; warpx2 collect `38/8027`; CP collect `612/8027`; exact, warpx2, and full CP split-4 selectors passed (`602 passed, 10 skipped` for CP); frontend zero-basis parse selector passed; affected invalid/conversion lit passed; `git diff --check` passed.

- Current copy `warpx2` integer subword-boundary checkpoint, 2026-04-14 13:44 UTC: the no-scales `warpx2` clean-unsupported source-width matrix now shares the broader subword dtype table with the linear no-scales copy source-width matrix. `CP_NO_SCALES_WARPX2_SUBWORD_UNSUPPORTED_CASES` covers f16/bf16/i16/i8 for single-CTA `01_23` and `02_13`, plus f16/bf16/i16 for two-CTA `01_23` and `02_13`, asserting the public `Source element type should be 32-bit.` verifier path. Two-CTA i8 is intentionally not in this source-width test because it fails earlier at the descriptor/register-layout boundary (`TMEM layout 'auto' unsupported` for int8 two-CTA views), not at the `ttng.tmem_copy` source-element verifier. Runtime-matrix collection is now `8019` tests: `cp=604`, `mma=1975`, splitn/misc `=571`, `ld_red=1920`, and `ldst=2949`; bucketed evidence aggregates to `7568 passed, 451 skipped`. Validation: `make -j8`; py-compile; exact collect `14/8019`; CP collect `604/8019`; full-file collect `8019`; exact selector passed all `14` cases across split-4 (`4`, `4`, `4`, and `2`; group times `4.56s`, `4.54s`, `4.69s`, and `4.26s`); `git diff --check` passed.

- Current rank-5 descriptor `ld/st` N=256 checkpoint, 2026-04-14 13:41 UTC: rank-5 descriptor positive coverage now reaches `N=256` through a lower-resource `[1,1,1,M,N]` parent helper. The new `test_tmem_runtime_matrix_ldst_descriptor_rank5_unit_parent_n256_roundtrip` covers `f32`/`i32`, single-CTA identity/mixed layouts, two-CTA block/MMAv5-like layouts, every public `ld/st` variant, exact eight-message store/load counts at `N=256`, and the same `index`/`subslice`/`reshape`/`trans` descriptor-chain markers. Runtime-matrix collection is now `8013` tests: `cp=598`, `mma=1975`, splitn/misc `=571`, `ld_red=1920`, and `ldst=2949`; bucketed evidence aggregates to `7562 passed, 451 skipped`. Validation: rank-5 unit-parent probes over all layouts/variants; `make -j8`; py-compile; focused collect `40/8013`; rank-5 collect `130/8013`; `ldst` collect `2949/8013`; full-file collect `8013`; exact selector passed all `40` cases across split-4 (`10` per group; group times `305.64s`, `215.09s`, `252.79s`, and `203.92s`).

- Current linear no-scales copy subword dtype-boundary checkpoint, 2026-04-14 13:27 UTC: the clean unsupported `tcgen05.cp` no-scales linear subword matrix now covers integer subword payloads in addition to f16/bf16. `CP_LINEAR_NO_SCALES_SUBWORD_UNSUPPORTED_CASES` spans `dtype in {f16,bf16,i16,i8}`, `M=128`, `N in {128,256}`, and `swizzle=32`, asserting the public `Source element type should be 32-bit.` verifier path without PassManager/assertion noise. Runtime-matrix collection is now `7973` tests: `cp=598`, `mma=1975`, splitn/misc `=571`, `ld_red=1920`, and `ldst=2909`; bucketed evidence aggregates to `7522 passed, 451 skipped`. Validation: i16/i8 scratch probes hit the intended clean diagnostic; `make -j8`; py-compile; exact collect `8/7973`; CP collect `598/7973`; full-file collect `7973`; exact selector passed all `8` cases across split-4 (`2` per group; slowest `4.52s`); `git diff --check` passed.

- Current linear no-scales copy subslice N=256 checkpoint, 2026-04-14 13:24 UTC: the supported `tcgen05.cp` linear `memdesc_subslice` copy view now covers `N=256` beside the existing `N=128` rows. The table spans `f32`/`i32`, `M=128`, `N in {128,256}`, swizzle byte widths `{32,64,128}`, a `128x(2*N)` parent TMEM-linear allocation, and exact `tcgen05.cp.cta_group::1.128x256b` opcode counts (`16` for `N=128`, `32` for `N=256`). Runtime-matrix collection is now `7969` tests: `cp=594`, `mma=1975`, splitn/misc `=571`, `ld_red=1920`, and `ldst=2909`; bucketed evidence aggregates to `7518 passed, 451 skipped`. Validation: representative N=256 probes for f32/i32; `make -j8`; py-compile; focused collect `12/7969`; CP collect `594/7969`; full-file collect `7969`; focused selector passed all `12` cases across split-4 (`3` per group; slowest `5.46s`); `git diff --check` passed.

- Current M64 split-N `ld/st` N=256 checkpoint, 2026-04-14 13:19 UTC: the M64 split-N runtime matrix now reaches `N=256` for identity split-N immediate checks, auto-selected `16x32bx2`, explicit `16x32bx2` equivalence, the full row/column-permuted M64 sweep, and a representative row/column auto-selection case. Coverage spans `f32`/`i32`, `64xN`, `N in {2,4,8,16,32,64,128,256}`, explicit `32x32b_splitn` and `16x32bx2` variants for row/column layouts, plus the new `identity/reverse,N=256` auto-selection anchor. Runtime-matrix collection is now `7963` tests: `cp=588`, `mma=1975`, splitn/misc `=571` (including the M64 split-N `ld/st` family), `ld_red=1920`, and `ldst=2909`; bucketed evidence aggregates to `7512 passed, 451 skipped`. Validation: `make -j8`; py-compile; full-file collect `7963`; focused split-N collect `566/7963`; focused split-N selector passed all `566` cases across split-4 (`142`, `142`, `142`, and `140`; slowest `30.90s`); `git diff --check` passed.

- Current copy `warpx2` subword-boundary checkpoint, 2026-04-14 13:11 UTC: the no-scales `warpx2` runtime matrix now pins f16/bf16 as clean unsupported for single-CTA `01_23`, single-CTA `02_13`, two-CTA `01_23`, and the already-unsupported two-CTA `02_13` family. These rows preserve the public 32-bit source-element contract (`Source element type should be 32-bit.`) without PassManager/assertion noise. Runtime-matrix collection is now `7891` tests: `cp=588`, `mma=1975`, splitn/misc `=499`, `ld_red=1920`, and `ldst=2909`; bucketed evidence aggregates to `7440 passed, 451 skipped`. Validation: f16/bf16 probes for all four family/CTA combinations; `make -j8`; py-compile; subword focused collect `8/7891`; full `warpx2` collect `29/7891`; CP collect `588/7891`; full-file collect `7891`; full `warpx2` selector passed all `29` cases across split-4 (`8`, `8`, `8`, `5`; slowest `8.57s`); `git diff --check` passed.

- Current TMA-fed two-CTA TF32 K-width checkpoint, 2026-04-14 13:10 UTC: the separate TF32 descriptor-fed `cta_group::2` TMA route now covers `blockK in {32,64}` over `blockN in {64,128,256}`, legacy/canonical two-CTA TMEM-linear accumulator layouts, and both non-accumulator plus `use_acc=True` modes. The default `[K,N]` B descriptor path remains a clean unsupported transposed-float32 shared-operand diagnostic at both K widths; the supported `[N,K]` descriptor plus shared `permute((1,0))` path emits exact `tcgen05.mma.cta_group::2.kind::tf32` PTX/LLIR streams. Runtime-matrix collection is now `7883` tests: `cp=580`, `mma=1975`, splitn/misc `=499`, `ld_red=1920`, and `ldst=2909`; bucketed evidence aggregates to `7432 passed, 451 skipped`. Validation: K64 TF32 positive and clean-negative probes; `make -j8`; py-compile; focused collect `36/7883`; full-file collect `7883`; focused split-4 selector passed all `36` cases (`9` per group; slowest `17.77s`); `git diff --check` passed.

- Current TMA-fed two-CTA non-TF32 K-width checkpoint, 2026-04-14 13:05 UTC: the descriptor-fed `cta_group::2` TMA matrix now covers `dtype in {f16,bf16,f8e5m2,f8e4m3}`, `blockN in {64,128,256}`, `blockK in {32,64}`, legacy/canonical two-CTA TMEM-linear accumulator layouts, and both no-accumulator plus `use_acc=True` modes. Opcode-count assertions now use `_expected_plain_mma_op_count(dtype, blockK)` so f16/bf16 rows double from two to four messages at `K=64`, and f8 rows double from one to two. Runtime-matrix collection is now `7865` tests: `cp=580`, `mma=1957`, splitn/misc `=499`, `ld_red=1920`, and `ldst=2909`; bucketed evidence aggregates to `7414 passed, 451 skipped`. Validation: K64 descriptor-fed probes passed; `make -j8`; py-compile; focused collect `96/7865`; full-file collect `7865`; focused split-4 selector passed all `96` cases (`24` per group; slowest `16.89s`); `git diff --check` passed.

- Current TMA-fed two-CTA non-TF32 MMAv5 descriptor checkpoint, 2026-04-14 13:03 UTC: the descriptor-fed `cta_group::2` TMA path now covers `dtype in {f16,bf16,f8e5m2,f8e4m3}` over `blockN in {64,128,256}`, legacy/canonical two-CTA TMEM-linear accumulator layouts, and both no-accumulator plus `use_acc=True` modes. F8 rows emit exact `tcgen05.mma.cta_group::2.kind::f8f6f4` PTX/LLIR opcodes; f16/bf16 rows emit exact `kind::f16` opcodes, and all rows keep the multicast commit check. Runtime-matrix collection is now `7817` tests: `cp=580`, `mma=1909`, splitn/misc `=499`, `ld_red=1920`, and `ldst=2909`; bucketed evidence aggregates to `7366 passed, 451 skipped`. Validation: f8 descriptor-fed probes passed; `make -j8`; py-compile; focused collect `48/7817`; full-file collect `7817`; focused split-4 selector passed all `48` cases (`12` per group; slowest `16.81s`); `git diff --check` passed.

- Current TMA-fed two-CTA BF16 parity checkpoint, 2026-04-14 13:00 UTC: the descriptor-fed `cta_group::2` f16-like path now covers both `f16` and `bf16` over `blockN in {64,128,256}`, legacy/canonical two-CTA TMEM-linear accumulator layouts, and both no-accumulator plus `use_acc=True` modes. BF16 emits the same exact `tcgen05.mma.cta_group::2.kind::f16` PTX/LLIR opcode stream as the direct BF16 contract while checking `a @ b` / `a @ b + c` with BF16 tolerance. Runtime-matrix collection is now `7793` tests: `cp=580`, `mma=1885`, splitn/misc `=499`, `ld_red=1920`, and `ldst=2909`; bucketed evidence aggregates to `7342 passed, 451 skipped`. Validation: BF16 probe passed; `make -j8`; py-compile; focused collect `24/7793`; full-file collect `7793`; focused split-4 selector passed all `24` cases (`6` per group; slowest `10.77s`); `git diff --check` passed. A rank-5 `N=256` ldst probe was not promoted because the current `[1,1,2,M,N]` helper requires `1024` TMEM columns against the `512` hardware limit.

- Current TMA-fed two-CTA f16 MMAv5 N-width/use-acc checkpoint, 2026-04-14 12:55 UTC: the descriptor-fed `cta_group::2` f16 path now covers `blockN in {64,128,256}` for both legacy and canonical two-CTA TMEM-linear accumulator layouts, and both no-accumulator and `use_acc=True` modes. A new `tmem_mma_twocta_use_acc_kernel` seeds the accumulator TMEM tile from a global `c` tensor before multicast `tcgen05_mma(..., use_acc=True)`. The adjacent direct two-CTA plain-kind test now defines its local `cga_layout_c_arg`, fixing a latent test typo in the same section. Runtime-matrix collection is now `7781` tests: `cp=580`, `mma=1873`, splitn/misc `=499`, `ld_red=1920`, and `ldst=2909`; bucketed evidence aggregates to `7330 passed, 451 skipped`. Validation: `make -j8`; py-compile; focused collect `12/7781`; full-file collect `7781`; focused split-4 selector passed all `12` cases (`3` per group; slowest `8.31s`); adjacent direct two-CTA node `test_tmem_runtime_matrix_mma_twocta_plain_kinds[f16-legacy-64-32]` passed; `git diff --check` passed.

- Current copy `warpx2` lit-boundary checkpoint, 2026-04-14 12:49 UTC: `test/TritonNvidiaGPU/invalid.mlir` now pins the no-scales two-CTA `warpx2::02_13.64x128b` clean-unsupported diagnostic with the canonical shared-linear two-CTA source layout and matching two-CTA TMEM-linear destination. This does not claim new support; it records the current hard requirement that a future positive path needs a `cta_group::2` descriptor/address schedule preserving the high source-column bit. Runtime-matrix collection remains `7771` tests with bucketed evidence `7320 passed, 451 skipped`; the refreshed `-k warpx2` runtime selector collected `21/7771` and passed across split-4 (`6`, `6`, `6`, `3`). Validation: `make -j8`; `lit -v test/TritonNvidiaGPU/invalid.mlir`.

- Current rank-5 descriptor `ld/st` N-width checkpoint, 2026-04-14 12:45 UTC: the executable `[1,1,2,M,N]` rank-5 descriptor roundtrip now covers `N in {64,128}` for both `f32` and `i32` payloads, preserving the single-CTA identity/mixed and two-CTA block/MMAv5-like layout surface across every public `ld/st` variant. Runtime-matrix collection is now `7771` tests: `cp=580`, `mma=1863`, splitn/misc `=499`, `ld_red=1920`, and `ldst=2909`; bucketed evidence aggregates to `7320 passed, 451 skipped`. Validation: `make -j8`; py-compile; focused collect `80/7771`; full `ldst` collect `2909/7771`; focused rank-5 small selector passed all `80` cases across split-4 (`20` per group; slowest `260.54s`). `N=256` remains a separate follow-up because the current opcode-count assertion needs a tiling-aware update before that width is added.

- Current barrier/commit contract checkpoint, 2026-04-14 12:39 UTC: the shared barrier verifier now rejects non-shared-memory mbarrier memdescs, and `ttng.tc_gen5_commit` rejects non-shared-memory descriptor operands in `descs`. `test/TritonNvidiaGPU/invalid.mlir` pins both clean diagnostics. This is compiler-only allocation/commit contract coverage; runtime-matrix collection remains `7731` tests with bucketed evidence `7280 passed, 451 skipped`. Validation: `make -j8`; focused invalid lit; adjacent conversion/NVWS/TritonGPU barrier lit.

- Current M64 MMAv5 accumulator-subview checkpoint, 2026-04-14 12:26 UTC: MMAv5 accumulator view planning now accepts legal `blockM=64` linear descriptor subviews when the backing allocation is wider than the active tile. The verifier restricts the canonical allocation layout to the active view and drops only inactive column selector bases, preserving pre-existing zero bases as layout semantics. Added a fast `ops.mlir` positive for `64x512 -> 64x256` and a 120-case runtime matrix over all supported plain kinds, `N in {64,128,256}`, `K in {32,64}`, both slice starts, and both accumulator modes. Runtime-matrix collection is now `7731` tests with bucketed evidence `7280 passed, 451 skipped`. Validation: `make -j8`; py-compile; focused collect `120/7731`; adjacent collect `360/7731`; split-4 focused and adjacent GPU sweeps passed; `lit -v test/TritonNvidiaGPU/ops.mlir` passed.

- Current malformed `tmem_alloc` verifier checkpoint, 2026-04-14 12:09 UTC: `test/TritonNvidiaGPU/invalid.mlir` now pins clean diagnostics for `ttng.tmem_alloc` result/source contract mismatches: result shape vs alloc shape, source shape vs destination memdesc shape, and source element type vs destination element type. This is compiler-only lit coverage; runtime-matrix counts remain `7611` tests with bucketed evidence `7160 passed, 451 skipped`. Validation: affected invalid lit passed from the CMake build dir.

- Current standalone commit lit checkpoint, 2026-04-14 12:07 UTC: conversion lit now pins standalone two-CTA `ttng.tc_gen5_commit` lowering both without descriptor operands and with two descriptor operands, and both forms lower to `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`. `test/TritonNvidiaGPU/invalid.mlir` now also pins a clean verifier diagnostic for malformed three-descriptor commits. This is compiler-only coverage, so the runtime-matrix collection remains `7611` tests with bucketed evidence `7160 passed, 451 skipped`. Validation: `make -j8`; affected lit files passed from the CMake build dir.

- Current `ld.red` minimal-N descriptor/direct explicit-variant checkpoint, 2026-04-14 12:50 UTC: descriptor-chain `ld.red` N-width coverage now includes `N=32` for identity, tile-permuted, pure column reverse, pure row reverse, and mixed row/column reverse layouts. The same `N=32` layouts also run through the shared explicit-variant matrix for both descriptor-chain and direct source paths over `32x32b`, `16x32bx2`, and `32x32b_splitn`, both reductions, and all legal modifier modes, pinning the minimal `tcgen05.ld.red.sync.aligned.32x32b.x32` opcode family. Runtime-matrix collection is now `7611` tests: `cp=580`, `mma=1743`, splitn/misc `=499`, `ld_red=1920`, and `ldst=2869`; bucketed evidence aggregates to `7160 passed, 451 skipped`. Validation: `make -j8`; py-compile; `git diff --check`; focused `N=32` descriptor/direct selector collect `280/7611`; full `ld_red` collect `1920/7611`; full-file collect `7611`; focused selector passed all `280` cases across split-4 (`70` per group; times `774.99s`, `1402.19s`, `599.46s`, and `547.65s`). The group-2 timing is a scheduling imbalance, not a hang; it printed progress throughout.

- Current scaled `warpx4` copy-helper use-acc checkpoint, 2026-04-14 12:35 UTC: the shared scaled-copy helper now accepts `ACC_INIT` and can seed the accumulator TMEM tile before the first scaled MMA chunk. `test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix_use_acc` mirrors the existing scaled-copy helper matrix over every current format pair, `blockN in {128,256}`, `blockK in {128,256}`, CTA group, multicast, and accumulator-layout combination, checking `matmul + 1.0` while still pinning exact scaled-copy and scaled-MMAv5 opcode streams. Runtime-matrix collection is now `7331` tests: `cp=580`, `mma=1743`, splitn/misc `=499`, `ld_red=1640`, and `ldst=2869`; bucketed evidence aggregates to `6880 passed, 451 skipped`. Validation: `make -j8`; py-compile; `git diff --check`; focused use-acc collect `160/7331`; `cp_scales_warpx4` collect `354/7331`; full-file collect `7331`; focused use-acc selector passed all `160` cases across split-4 (`40` per group; slowest `52.83s`); adjacent `cp_scales_warpx4` selector passed all `354` cases across split-4 (`89`, `89`, `89`, `87`; slowest `89.01s`).

- Current direct `ld.red` explicit N-width checkpoint, 2026-04-14 12:20 UTC: direct hardware reductions now cover the same N-width explicit-variant matrix that descriptor-chain reductions already covered. `tmem_ld_red_explicit_layout_kernel` takes `N` as a constexpr, and the new direct matrix spans identity, tile-permuted, pure column reverse, pure row reverse, and mixed row/column reverse layouts at `N in {64,256}` for explicit `32x32b`, `16x32bx2`, and `32x32b_splitn` requests, both reductions, and all legal modifier modes. Runtime-matrix collection is now `7171` tests: `cp=420`, `mma=1743`, splitn/misc `=499`, `ld_red=1640`, and `ldst=2869`; bucketed evidence aggregates to `6720 passed, 451 skipped`. Validation: `make -j8`; py-compile; `git diff --check`; new direct explicit N-sweep collect `240/7171`; full `ld_red` collect `1640/7171`; full-file collect `7171`; new direct selector passed all `240` cases across split-4 on four GPUs (`60` per group; slowest `573.50s`); adjacent N=128 explicit/unsupported selectors passed all `184` cases across split-4 (`46` per group; slowest `308.59s`).

- Current legacy single-CTA no-scales copy dtype-parity checkpoint, 2026-04-14 12:05 UTC: the original single-CTA no-scales CP root and swizzle matrices now run over the shared `CP_NO_SCALES_128X128_DTYPES` table instead of only `i32`, adding `f32` parity while preserving the existing shape, `BLOCK_N`, and swizzle coverage. Runtime-matrix collection is now `6931` tests: `cp=420`, `mma=1743`, splitn/misc `=499`, `ld_red=1400`, and `ldst=2869`; bucketed evidence aggregates to `6480 passed, 451 skipped`. Validation: `make -j8`; py-compile for `python/test/gluon/test_tmem_runtime_matrix.py`; `git diff --check`; exact root+swizzle collect `78/6931`; full CP collect `420/6931`; full-file collect `6931`; focused root+swizzle split-4 execution passed/skipped as `68 passed, 10 skipped`; full CP bucket split-4 execution passed/skipped as `410 passed, 10 skipped` with groups `95 passed, 10 skipped`, `105 passed`, `105 passed`, and `105 passed`.

- Current two-CTA TMA-fed TF32 MMAv5 use-acc checkpoint, 2026-04-14 10:22 UTC: the reachable TMA-fed `cta_group::2` TF32 path where matrix B is supplied as a non-transposed `[N,K]` descriptor and shared-permuted into MMAv5 now has explicit nonzero accumulator-add coverage. A new helper initializes the two-CTA accumulator TMEM tile from a global `c` tensor, then runs multicast `tcgen05.mma(..., use_acc=True)` against the TMA-loaded operands. Coverage spans `blockN in {64,128,256}` and both legacy/canonical TMEM-linear accumulator layouts, matching the existing no-accumulator TMA-positive route. Runtime-matrix collection is now `6602` tests: `cp=381`, `mma=1693`, splitn/misc `=499`, `ld_red=1160`, and `ldst=2869`; bucketed evidence aggregates to `6156 passed, 446 skipped`. Validation: py-compile; `make -j8`; `git diff --check`; focused collect `6/6602`; tight MMA collect `1693/6602`; split-4 execution passed the three non-empty groups (`2` cases each; group 4 empty), and an aggregate exact selector run passed all `6` cases.

- Current `ld.red` descriptor-chain pure row/column layout checkpoint, 2026-04-14 10:12 UTC: descriptor-view reductions now include pure column-reverse and pure row-reverse TMEM-linear source layouts. At `N=128`, both layouts are covered through the same descriptor chain and every compatible explicit reduction register-layout request (`auto`, `32x32b`, `16x32bx2`, and `32x32b_splitn`); the descriptor-chain N-width sweep also covers both pure layouts at `N=64` and `N=256` with `auto`. Runtime-matrix collection is now `6596` tests: `cp=381`, `mma=1687`, splitn/misc `=499`, `ld_red=1160`, and `ldst=2869`; bucketed evidence aggregates to `6150 passed, 446 skipped`. Validation: py-compile; `make -j8`; `git diff --check`; full-file collect `6596`; descriptor-chain collect `240/6596`; full `ld_red` collect `1160/6596`; focused new pure row/column descriptor-chain selector passed all `96` cases across split-4 on four GPUs (`24` per group; slowest `487.70s`).

- Current copy `warpx2` conversion-lit / linear-layout-asm checkpoint, 2026-04-14 10:03 UTC: generic `LinearLayout` assembly now preserves explicit output codomain sizes for non-surjective layouts by printing/parsing `out = [...]`. This is needed because supported no-scales `warpx2` TMEM-linear layouts contain zero row bases; without the explicit codomain, textual MLIR round-trips infer a smaller surjective layout and the verifier rejects the original `128x4` / `256x4` memdesc shape. Conversion lit coverage now pins the already-supported no-scales copy lowerings for single-CTA `warpx2::01_23`, single-CTA `warpx2::02_13`, and two-CTA `warpx2::01_23`. No production lowering was added for the parked hard frontiers: no-scales two-CTA `warpx2::02_13` and true tensor-memory-scales `warpx2` still need a real descriptor/address/staging model. Validation: `make -j8`; `lit -v test/Conversion/tritongpu_to_llvm_blackwell.mlir`; nearby lit for `test/TritonNvidiaGPU/tmem_layouts.mlir`, `ops.mlir`, and `invalid.mlir`; py-compile for the relevant Gluon tests; `git diff --check`; frontend parser selector passed `3` selected cases in a single process; runtime `-k warpx2` collect selected `21/6500` and split-4 GPU execution passed all selected cases (`6`, `6`, `6`, and `3`).

- Current `ld.red` descriptor-chain N-width checkpoint, 2026-04-14 10:02 UTC: `tmem_ld_red_descriptor_chain_kernel` now takes `N` as a constexpr and the descriptor-chain reduction matrix has an additional N-width sweep. The new positive slice covers `N=64` and `N=256` for identity, tile-permuted, and row/column-permuted descriptor-view layouts with `auto` register-layout selection, across `min`/`max` and all legal `abs`/`NaN` modifier modes. Runtime-matrix collection is now `6500` tests: `cp=381`, `mma=1687`, splitn/misc `=499`, `ld_red=1064`, and `ldst=2869`; bucketed evidence aggregates to `6054 passed, 446 skipped`. Validation included py-compile, `git diff --check`, `make -j8`, focused new-sweep collect `48/6500`, adjacent descriptor-chain collect `144/6500`, full `ld_red` collect `1064/6500`, full-file collect `6500`, four-GPU new-sweep execution (`12` per shard; slowest `235.70s`), and four-GPU adjacent descriptor-chain execution (`36` per shard; slowest `501.35s`).

- Parked frontier, 2026-04-14 09:35 UTC: a scratch extension of scaled two-CTA MMAv5 accumulator-subview coverage to `block_n=64` / `parent_n=128` fails during B-scale shared descriptor construction with a zero descriptor dimension (`shape must have power-of-2 and non-zero dimensions; got 1, 0, ...`). The source was reverted to the green `6363`-case matrix. Do not treat this as a positive or ISA-impossible clean negative until the scale-descriptor helper path is understood.

- Current `ld.red` descriptor-chain explicit-variant checkpoint, 2026-04-14 09:32 UTC: descriptor-view reductions now request `auto`, `32x32b`, `16x32bx2`, and `32x32b_splitn` for identity, tile-permuted, and row/column-permuted descriptor-chain layouts. Current runtime-matrix collection is `6363` tests: `cp=381`, `mma=1598`, splitn/misc `=499`, `ld_red=1016`, and `ldst=2869`; bucketed evidence now aggregates to `5917 passed, 446 skipped`. Validation: `make -j8` no-op success; py-compile passed; `git diff --check` passed before docs; no-PYTHONPATH focused `ld_red_descriptor_chain` collect selected `96/6363`; no-PYTHONPATH `ld_red` collect selected `1016/6363`; full-file collect reported `6363`; focused selector passed all `96` cases across split-4 on four GPUs (`24` per group; times `49.01s`, `159.29s`, `276.27s`, and `458.74s`).

- Current two-CTA plain-MMAv5 accumulator-subslice descriptor-view checkpoint, 2026-04-14 09:23 UTC: added a 120-case positive matrix for `cta_group::2` `tcgen05.mma` accumulator views formed with `ttg.memdesc_subslice` from a linear two-CTA `[256, 2*N]` parent. Coverage spans every supported plain operand kind, `N in {64,128,256}`, `K in {32,64}`, both `slice_start=0` and `slice_start=N`, and both no-accumulator plus `use_acc=True` paths; the `N=256` rows are executable with the 512-column parent. Current runtime-matrix collection is `6315` tests: `cp=381`, `mma=1598`, splitn/misc `=499`, `ld_red=968`, and `ldst=2869`; bucketed evidence now aggregates to `5869 passed, 446 skipped`. Validation: `make -j8` no-op success; py-compile passed; `git diff --check` passed before docs; no-PYTHONPATH focused `mma_twocta_acc_subslice_view_plain_kinds` collect selected `120/6315`; no-PYTHONPATH tight MMA collect selected `1598/6315`; full-file collect reported `6315`; focused selector passed all `120` cases across split-4 on four GPUs (`30` per group; times `33.98s`, `36.59s`, `35.12s`, and `36.07s`).

- Current plain-MMAv5 accumulator-subslice descriptor-view checkpoint, 2026-04-14 09:20 UTC: added a 120-case positive matrix for `tcgen05.mma` accumulator views formed with `ttg.memdesc_subslice` from a linear `[128, 2*N]` parent. Coverage spans every supported plain operand kind, `N in {64,128,256}`, `K in {32,64}`, both `slice_start=0` and `slice_start=N`, and both no-accumulator plus `use_acc=True` paths. Current runtime-matrix collection is `6195` tests: `cp=381`, `mma=1478`, splitn/misc `=499`, `ld_red=968`, and `ldst=2869`; bucketed evidence now aggregates to `5749 passed, 446 skipped`. Validation: `make -j8` no-op success; py-compile passed; `git diff --check` passed before docs; no-PYTHONPATH focused `mma_acc_subslice_view_plain_kinds` collect selected `120/6195`; no-PYTHONPATH tight MMA collect selected `1478/6195`; full-file collect reported `6195`; focused selector passed all `120` cases across split-4 on four GPUs (`30` per group; times `27.11s`, `28.64s`, `26.84s`, and `27.04s`).

- Current `ld.red` descriptor-view explicit-variant checkpoint, 2026-04-14 09:16 UTC: extended the TMEM descriptor-view reduction slice so identity views cover every compatible explicit reduction register-layout request (`auto`, `32x32b`, `16x32bx2`, `32x32b_splitn`) while tile-permuted and row/column-permuted descriptor views continue to cover `auto`. Current runtime-matrix collection is `6075` tests: `cp=381`, `mma=1358`, splitn/misc `=499`, `ld_red=968`, and `ldst=2869`; bucketed evidence now aggregates to `5629 passed, 446 skipped`. Validation: `make -j8` no-op success; py-compile passed; `git diff --check` passed; no-PYTHONPATH focused `ld_red_descriptor_chain` collect selected `48/6075`; no-PYTHONPATH `ld_red` collect selected `968/6075`; focused `ld_red_descriptor_chain` selector passed all `48` cases across split-4 on four GPUs (`12` per group; times `44.84s`, `48.10s`, `58.35s`, and `190.20s`).

- Current `ld.red` descriptor-view checkpoint, 2026-04-14 09:11 UTC: added positive hardware-reduction coverage through a TMEM descriptor chain in `python/test/gluon/test_tmem_runtime_matrix.py`. The new kernel stores through a `[2,128,128]` parent, reduces through a `slice`/`index`/`reshape` descriptor view, and covers identity, tile-permuted, and row/column-permuted source layouts across `min`/`max` plus all legal `abs`/`NaN` modifier modes. Current runtime-matrix collection is `6051` tests: `cp=381`, `mma=1358`, splitn/misc `=499`, `ld_red=944`, and `ldst=2869`; bucketed evidence now aggregates to `5605 passed, 446 skipped`. Validation: `make -j8` no-op success; py-compile passed; `git diff --check` passed; no-PYTHONPATH focused `ld_red_descriptor_chain` collect selected `24/6051`; no-PYTHONPATH `ld_red` collect selected `944/6051`; focused `ld_red_descriptor_chain` selector passed all `24` cases across split-4 on four GPUs (`6` per group; times `25.60s`, `39.63s`, `72.78s`, and `117.55s`). Note: an initial lifted-layout version of this probe aborted in `get_reg_layout` with a descriptor-dimension mismatch; the committed test follows the existing multibuffer `ld/st` pattern by passing the base 2D layout into the `[2,M,N]` allocation.

- Current rank-5 descriptor `ld/st` dtype-parity checkpoint, 2026-04-14 09:02 UTC: the executable `[1,1,2,M,N]` rank-5 descriptor roundtrip now derives its TMEM element type from the input pointer and covers both `f32` and `i32` payloads. This expands the rank-5 positive slice from `20` to `40` cases while preserving the same single-CTA identity/mixed and two-CTA block/MMAv5-like layout coverage across every public `ld/st` variant. Current runtime-matrix collection is `6027` tests: `cp=381`, `mma=1358`, splitn/misc `=499`, `ld_red=920`, and `ldst=2869`; bucketed evidence now aggregates to `5581 passed, 446 skipped`. Validation: `make -j8` no-op success; py-compile passed; `git diff --check` passed; no-PYTHONPATH focused `rank5_small` collect selected `40/6027`; no-PYTHONPATH `ldst` collect selected `2869/6027`; focused `rank5_small` selector passed all `40` cases across split-4 on four GPUs (`10` per group; times `127.35s`, `92.63s`, `127.20s`, and `92.18s`).

- Current rank-5 descriptor `ld/st` positive checkpoint, 2026-04-14 08:57 UTC: added an executable smaller rank-5 descriptor roundtrip matrix in `python/test/gluon/test_tmem_runtime_matrix.py` using a `[1,1,2,M,N]` allocation and lifted `[1,1,2]` TMEM-linear layouts. This preserves the old `[2,2,2,M,N]` rank-5 OOR skip boundary (`Required: 4096`, hardware limit `512`) while adding positive rank-5 `memdesc_index`/`memdesc_subslice`/`memdesc_reshape`/`memdesc_trans` coverage for single-CTA identity and mixed layouts plus two-CTA block and MMAv5-like layouts across every public `ld/st` variant. Current runtime-matrix collection is `6007` tests: `cp=381`, `mma=1358`, splitn/misc `=499`, `ld_red=920`, and `ldst=2849`; bucketed evidence now aggregates to `5561 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused `rank5_small` collect selected `20/6007`; no-PYTHONPATH full-file collect reported `6007`; no-PYTHONPATH `ldst` collect selected `2849/6007`; focused `rank5_small` selector passed all `20` cases across split-4 on four GPUs (`5` per group; times `4.45s`, `4.45s`, `4.66s`, and `4.40s`).

- Current scaled-MMAv5 narrow tile-permuted accumulator clean-negative checkpoint, 2026-04-14 08:48 UTC: direct block-scaled MMAv5 tile-permuted accumulator clean-negative coverage now separately pins the narrow direct-layout boundary, `128x32/tile_n=8` and `128x64/tile_n=16`, across every current scaled format pair and `K in {128,256}`. These cases report the directly-supported block-scaled tensor-memory-layout diagnostic because narrow tile-permuted accumulators are not directly representable for the current direct block-scaled MMAv5 path. This is distinct from the existing repeated-`N=32` `128x128/tile_n=32` clean negative, which remains the scale-fragment 64-column-alignment boundary. Current runtime-matrix collection is `5987` tests: `cp=381`, `mma=1358`, splitn/misc `=499`, `ld_red=920`, and `ldst=2829`; bucketed evidence now aggregates to `5541 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed after docs; `make -j8` no-op success; no-PYTHONPATH focused `scaled_acc_tile_permuted_narrow` collect selected `20/5987`; no-PYTHONPATH tight MMA collect selected `1358/5987`; no-PYTHONPATH full-file collect reported `5987`; focused `scaled_acc_tile_permuted_narrow` selector passed all `20` cases across split-4 on four GPUs (`5` per group; times `5.82s`, `5.90s`, `5.69s`, and `6.08s`).

- Current plain-MMAv5 narrow tile-permuted accumulator clean-negative checkpoint, 2026-04-14 08:44 UTC: plain MMAv5 tile-permuted accumulator clean-negative coverage now pins both narrow unsupported layouts, `128x32/tile_n=8` and `128x64/tile_n=16`, across every supported plain operand kind, `K in {32,64}`, and no-accumulator plus `use_acc=True` paths. The existing `N=64` clean-negative function is now parameterized as the narrow tile-permuted matrix, preserving the clean MMAv5-compatible tensor-memory-layout verifier diagnostic and no PassManager/assertion noise. Current runtime-matrix collection is `5967` tests: `cp=381`, `mma=1338`, splitn/misc `=499`, `ld_red=920`, and `ldst=2829`; bucketed evidence now aggregates to `5521 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed after docs; `make -j8` no-op success; no-PYTHONPATH focused `tile_permuted_narrow` collect selected `40/5967`; no-PYTHONPATH tight MMA collect selected `1338/5967`; no-PYTHONPATH full-file collect reported `5967`; focused `tile_permuted_narrow` selector passed all `40` cases across split-4 on four GPUs (`10` per group; times `5.76s`, `5.98s`, `5.90s`, and `5.76s`).

- Current scales `ld/st` explicit variant checkpoint, 2026-04-14 08:40 UTC: tensor-memory-scales `ld/st` variant coverage now includes explicit N-sharded `16x64b`, `16x128b`, and `16x256b` paths in addition to `auto`, `32x32b`, and `16x32bx2`. The positive matrix covers `M in {64,128,256}`, `N in {4,8,16,32}`, and `dtype_bits=4` whenever `M*N` meets the instruction atom threshold (`256`, `512`, or `1024` elements for `16x64b`, `16x128b`, or `16x256b` respectively), with exact `tcgen05.st/ld.sync.aligned.<variant>.x<count>.b32` opcode checks and offsets `[0, 1048576]`. The below-threshold rows are pinned as clean unsupported descriptor-view diagnostics. Current runtime-matrix collection is `5947` tests: `cp=381`, `mma=1318`, splitn/misc `=499`, `ld_red=920`, and `ldst=2829`; bucketed evidence now aggregates to `5501 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed after docs; `make -j8` no-op success; no-PYTHONPATH focused `ldst_scales_variant` collect selected `91/5947`; no-PYTHONPATH `ldst` collect selected `2829/5947`; full-file collect reported `5947`; focused `ldst_scales_variant` selector passed all `91` cases across split-4 on four GPUs (`23`, `23`, `23`, and `22` selected; times `5.25s`, `5.62s`, `5.23s`, and `5.49s`).

- Current scaled-MMAv5 TMEM-LHS nonzero use-acc checkpoint, 2026-04-14 08:34 UTC: scaled-MMAv5 TMEM-LHS descriptor-view and full-shape tile-permuted operand-A coverage now validates nonzero accumulator-add semantics. `tmem_mma_scaled_lhs_subslice_format_kernel` and `tmem_mma_scaled_lhs_tile_permuted_format_kernel` take `ACC_INIT`; existing zero paths and nearby clean-negative callers pass `0.0`, while the new use-acc matrices initialize accumulators to `1.0`. Coverage spans the existing positive LHS subview surface (`N in {64,128,256}`, `K in {128,256}` across current packed-storage reachable format pairs and legacy/canonical accumulator layouts) and the existing positive full-shape tile-permuted surface (all reachable packed-storage pairs at `K=256` plus mxfp8-storage pairs at `K=128`). The new tests check `a_ref @ b_ref.T + acc_init`, exact K-scaled scaled-MMAv5 opcode counts, exact commit opcodes, and the expected `ttg.memdesc_subslice` or full-shape `tensor_memory_linear` TTGIR markers. Current runtime-matrix collection is `5911` tests: `cp=381`, `mma=1318`, splitn/misc `=499`, `ld_red=920`, and `ldst=2793`; bucketed evidence now aggregates to `5465 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed before docs; `make -j8` no-op success; no-PYTHONPATH focused use-acc collect selected `84/5911`; no-PYTHONPATH tight MMA collect selected `1318/5911`; no-PYTHONPATH adjacent changed-callsite collect selected `198/5911`; focused use-acc selector passed all `84` cases across split-4 on four GPUs (`21` per group; times `44.69s`, `41.20s`, `59.75s`, and `67.78s`); adjacent selector passed all `198` cases across split-4 (`50`, `50`, `50`, and `48` selected; times `100.77s`, `146.15s`, `59.66s`, and `9.47s`).

- Current scaled-MMAv5 indexed-accumulator use-acc checkpoint, 2026-04-14 08:28 UTC: scaled-MMAv5 accumulator `memdesc_index` coverage now validates nonzero accumulator-add semantics. `tmem_mma_scaled_indexed_acc_format_kernel` takes `ACC_INIT`; the existing zero path passes `0.0`, and the new use-acc matrix initializes the indexed accumulator view to `1.0`. Coverage mirrors the resource-safe indexed matrix: every current scaled format pair, `K in {128,256}`, legacy parents at `N in {64,128}`, and canonical TMEM-linear parents at `N=64`; wider indexed parents remain hardware-resource-limited once scale descriptors are live. The new test checks `a_ref @ b_ref.T + acc_init`, exact K-scaled scaled-MMAv5 opcode counts, exact commit opcodes, and `ttg.memdesc_index`/layout markers. Current runtime-matrix collection is `5827` tests: `cp=381`, `mma=1234`, splitn/misc `=499`, `ld_red=920`, and `ldst=2793`; bucketed evidence now aggregates to `5381 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused indexed collect selected `60/5827`; no-PYTHONPATH tight MMA collect selected `1234/5827`; adjacent indexed selector passed all `60` cases across split-4 on four GPUs (`15` per group; times `27.19s`, `27.75s`, `27.17s`, and `27.42s`).

- Current scaled-MMAv5 use-acc K-depth checkpoint, 2026-04-14 08:26 UTC: scaled-MMAv5 nonzero accumulator-add coverage now has K-depth parity with the zero-accumulator matrices. Both the direct root and descriptor-view accumulator-subview `use_acc=True` matrices now cover `K in {128,256}` instead of only `K=128`; root coverage spans every current scaled format pair, `N in {64,128,256}`, and legacy/canonical accumulator layouts, while accumulator-subview coverage spans every format pair, `N in {64,128}`, and `slice_start in {0,N}`. The expected scaled-MMAv5 opcode counts now scale by `K // 128`, so the new `K=256` cases prove doubled instruction depth while still checking `a_ref @ b_ref.T + acc_init` and exact commit opcode counts. Current runtime-matrix collection is `5797` tests: `cp=381`, `mma=1204`, splitn/misc `=499`, `ld_red=920`, and `ldst=2793`; bucketed evidence now aggregates to `5351 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `100/5797`; no-PYTHONPATH tight MMA collect selected `1204/5797`; focused root+subview use-acc selector passed all `100` cases across split-4 on four GPUs (`25` per group; times `32.70s`, `41.00s`, `28.67s`, and `27.19s`).

- Current scaled-MMAv5 accumulator-subview nonzero-accumulator checkpoint, 2026-04-14 08:23 UTC: descriptor-view scaled-MMAv5 accumulator-subview coverage now explicitly validates `use_acc=True` semantics with a nonzero initialized sliced accumulator (`acc_init=1.0`) for every current scaled format pair (`mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`, `mxfp4/mxfp8`, `nvfp4/nvfp4`), `N in {64,128}`, `K=128`, and `slice_start in {0,N}`. Existing zero-accumulator accumulator-subview and tile-permuted clean-negative callers now pass `ACC_INIT=0.0`, while the new matrix checks `a_ref @ b_ref.T + acc_init` and still pins exact scaled-MMAv5 and commit opcode counts. Current runtime-matrix collection is `5747` tests: `cp=381`, `mma=1154`, splitn/misc `=499`, `ld_red=920`, and `ldst=2793`; bucketed evidence now aggregates to `5301 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `20/5747`; no-PYTHONPATH tight MMA collect selected `1154/5747`; no-PYTHONPATH adjacent changed-callsite collect selected `70/5747`; focused subview use-acc selector passed all `20` cases across split-4 on four GPUs (`5` per group; times `9.90s`, `9.09s`, `9.72s`, and `8.97s`); adjacent scaled accumulator-subview selectors passed all `70` cases across split-4 (`18`, `18`, `18`, and `16` selected; times `22.26s`, `25.62s`, `19.00s`, and `6.71s`).

- Current scaled-MMAv5 root nonzero-accumulator checkpoint, 2026-04-14 08:19 UTC: direct root scaled-MMAv5 coverage now explicitly validates `use_acc=True` semantics with a nonzero initialized accumulator (`acc_init=1.0`) for every current scaled format pair (`mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`, `mxfp4/mxfp8`, `nvfp4/nvfp4`), `N in {64,128,256}`, `K=128`, and both legacy plus canonical TMEM-linear accumulator layouts. Existing zero-accumulator root-format callers now pass `ACC_INIT=0.0`, while the new matrix checks `a_ref @ b_ref.T + acc_init` and still pins exact scaled-MMAv5 and commit opcode counts. Current runtime-matrix collection is `5727` tests: `cp=381`, `mma=1134`, splitn/misc `=499`, `ld_red=920`, and `ldst=2793`; bucketed evidence now aggregates to `5281 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `30/5727`; no-PYTHONPATH tight MMA collect selected `1134/5727`; no-PYTHONPATH adjacent changed-callsite collect selected `80/5727`; focused use-acc selector passed all `30` cases across split-4 on four GPUs (`8`, `8`, `8`, and `6` selected; times `15.75s`, `14.97s`, `15.54s`, and `11.66s`); adjacent scaled root/tile selectors passed all `80` cases across split-4 (`20` per group; times `34.83s`, `35.73s`, `35.99s`, and `32.12s`).

- Current direct-i8 MMAv5 clean-negative checkpoint, 2026-04-14 08:14 UTC: direct `tcgen05.mma.kind::i8` clean-negative coverage now has K-depth parity (`K in {32,64}`) for one-CTA and two-CTA legacy/canonical accumulator layouts, and now covers the M64 accumulator family across `N in {64,128,256}`, `K in {32,64}`, and legacy/canonical M64 layouts. These remain clean Blackwell unsupported boundaries, not positive MMAv5 targets. Current runtime-matrix collection is `5697` tests: `cp=381`, `mma=1104`, splitn/misc `=499`, `ld_red=920`, and `ldst=2793`; bucketed evidence now aggregates to `5251 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `36/5697`; no-PYTHONPATH tight MMA collect selected `1104/5697`; focused direct-i8 clean-negative selector passed all `36` cases across split-4 on four GPUs (`9` per group; times `4.43s`, `4.21s`, `4.34s`, and `4.29s`).

- Current x1 i32 `ld/st` descriptor coverage checkpoint, 2026-04-14 08:13 UTC: the one-column 32-bit `ld/st` surface now has f32+i32 parity across direct and descriptor-chain roundtrips for canonical single-CTA, legacy single-CTA, and canonical two-CTA layouts. The positive matrix covers `auto` and explicit `32x32b` variants and pins exact `tcgen05.st/ld.sync.aligned.32x32b.x1.b32` emission; the clean-negative matrix covers explicit `16x64b`, `16x128b`, and `16x256b` one-column variants for i32 with the same unsupported-descriptor diagnostic contract as f32. Current runtime-matrix collection is `5673` tests: `cp=381`, `mma=1080`, splitn/misc `=499`, `ld_red=920`, and `ldst=2793`; bucketed evidence now aggregates to `5227 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `21/5673`; focused x1 i32 selector passed all `21` cases across split-4 on four GPUs (`6`, `6`, `6`, and `3` selected; times `7.75s`, `41.66s`, `4.65s`, and `4.64s`).

- Current two-CTA plain-MMAv5 indexed-accumulator descriptor-view checkpoint, 2026-04-14 08:12 UTC: two-CTA accumulator `memdesc_index` coverage now mirrors the one-CTA plain indexed surface for every supported plain operand kind (`f16`, `tf32`, `bf16`, `f8e5m2`, `f8e4m3`), `K in {32,64}`, `use_acc` false/true, legacy parent layouts at `N in {64,128,256}`, and canonical TMEM-linear parent layouts at `N in {64,128}`. The canonical linear `N=256` two-CTA parent view is omitted as the same live-parent resource boundary as one-CTA linear `N=256`: `[2,256,256]` needs 1024 TMEM columns. Current runtime-matrix collection is `5652` tests: `cp=381`, `mma=1080`, splitn/misc `=499`, `ld_red=920`, and `ldst=2772`; bucketed evidence now aggregates to `5206 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH full-file collect reported `5652`; no-PYTHONPATH tight MMA collect selected `1080/5652`; focused two-CTA indexed-accumulator selector passed all `100` cases across split-4 on four GPUs (`25` per group; times `4.69s`, `4.69s`, `4.69s`, and `4.63s`).

- Current scaled-MMAv5 indexed-accumulator descriptor-view checkpoint, 2026-04-14 08:08 UTC: scaled accumulator `memdesc_index` coverage now spans every current scaled format pair (`mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`, `mxfp4/mxfp8`, `nvfp4/nvfp4`), `K in {128,256}`, legacy parent layouts at `N in {64,128}`, and canonical TMEM-linear parent layouts at `N=64`. The probed omitted shapes are resource boundaries: scaled-indexed linear `N=128` needs at least 520 TMEM columns once scale descriptors are live, and scaled-indexed legacy `N=256` needs at least 524 columns, both above the 512-column hardware limit. Current runtime-matrix collection is `5552` tests: `cp=381`, `mma=980`, splitn/misc `=499`, `ld_red=920`, and `ldst=2772`; bucketed evidence now aggregates to `5106 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH full-file collect reported `5552`; no-PYTHONPATH tight MMA collect selected `980/5552`; focused scaled indexed-accumulator selector passed all `30` cases across split-4 on four GPUs (`8`, `8`, `8`, and `6` selected; times `4.59s`, `6.07s`, `6.52s`, and `4.58s`).

- Current plain-MMAv5 indexed-accumulator descriptor-view checkpoint, 2026-04-14 07:55 UTC: the accumulator `memdesc_index` positive matrix now covers every supported plain operand kind (`f16`, `tf32`, `bf16`, `f8e5m2`, `f8e4m3`), `K in {32,64}`, `use_acc` false/true, legacy parent layouts at `N in {64,128,256}`, and canonical TMEM-linear parent layouts at `N in {64,128}`. The canonical linear `N=256` parent view is intentionally omitted because the live parent image is `[2,128,256]` and requires 1024 TMEM columns, exceeding the 512-column hardware limit before execution. Current runtime-matrix collection is `5522` tests: `cp=381`, `mma=950`, splitn/misc `=499`, `ld_red=920`, and `ldst=2772`; bucketed evidence now aggregates to `5076 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH full-file collect reported `5522`; no-PYTHONPATH tight MMA collect selected `950/5522`; focused indexed-accumulator selector passed all `100` cases across split-4 on four GPUs (`25` per group; times `4.32s`, `5.77s`, `6.60s`, and `7.57s`).

- Current `ld.red` additional unsupported-layout shape sweep checkpoint, 2026-04-14 07:49 UTC: the additional unsupported-layout clean-negative matrix now covers M64 `64xN`, `N in {32,64,128,256}`, and block-basis `128xN`, `N in {64,128,256}`, across `min`/`max`, `abs` false/true, and `PropagateNan.NONE/ALL`. These remain clean unsupported direct hardware `tcgen05.ld.red` source layouts: M64 rows must report the dedicated software-reduction diagnostic, and block-basis rows must report the descriptor-view register-layout diagnostic without PassManager/assertion noise. Current runtime-matrix collection is `5424` tests: `cp=381`, `mma=852`, splitn/misc `=499`, `ld_red=920`, and `ldst=2772`; bucketed evidence now aggregates to `4978 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH full-file collect reported `5424`; no-PYTHONPATH `ld_red` collect selected `920/5424`; focused unsupported-layout selector passed all `56` cases across split-4 on four GPUs (`14` per group; times `4.42s`, `4.42s`, `6.79s`, and `8.01s`).

- Current subword descriptor-chain `ld/st` checkpoint, 2026-04-14 07:47 UTC: broad subword descriptor-chain coverage now includes `f16`, `bf16`, `i16`, and `i8` identity TMEM-linear layouts at `N in {64,128,256}` across every public `ld/st` variant. This complements the existing direct subword matrix and the x1 subword/two-CTA descriptor special cases by proving the generic supported descriptor chain (`subslice`, `index`, `reshape`, `trans`, bitcast) for non-32-bit payloads. Current runtime-matrix collection is `5384` tests: `cp=381`, `mma=852`, splitn/misc `=499`, `ld_red=880`, and `ldst=2772`; bucketed evidence now aggregates to `4938 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `60/5384`; no-PYTHONPATH full-file collect reported `5384`; focused subword descriptor-chain selector passed all `60` cases across split-4 on four GPUs (`15` per group; times `151.24s`, `153.02s`, `150.40s`, and `152.90s`).

- Current higher-rank descriptor `ld/st` i32-parity checkpoint, 2026-04-14 07:41 UTC: the higher-rank descriptor-view positive matrix now covers f32+i32 for indexed views, multidimensional slices, and dim-0 slice/view chains in both single-CTA and two-CTA layouts. The affected kernels now derive the TMEM element type from the input pointer, so the same descriptor-chain lowering is validated for `.b32` integer payloads without changing compiler/lowering source. Current runtime-matrix collection is `5324` tests: `cp=381`, `mma=852`, splitn/misc `=499`, `ld_red=880`, and `ldst=2712`; bucketed evidence now aggregates to `4878 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `200/5324`; no-PYTHONPATH full-file collect reported `5324`; focused higher-rank descriptor selector passed all `200` cases across split-8 on four GPUs (`25` cases per group; times `384.32s`, `339.11s`, `402.75s`, `430.09s`, `418.65s`, `382.93s`, `131.40s`, and `278.68s`).

- Current M64 row/column split-N `ld/st` i32-parity checkpoint, 2026-04-14 07:24 UTC: the row/column-permuted M64 split-N sweep now covers f32+i32 for `64xN`, `N in {2,4,8,16,32,64,128}`, across explicit `32x32b_splitn` and `16x32bx2`, and the representative auto-selection checks also cover both dtypes. This is test-only coverage over already-supported `ld/st` lowering. Current runtime-matrix collection is `5224` tests: `cp=381`, `mma=852`, splitn/misc `=499`, `ld_red=880`, and `ldst=2612`; bucketed evidence now aggregates to `4778 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `452/5224`; no-PYTHONPATH full-file collect reported `5224`; focused M64 row/column split-N selector passed all `452` cases across split-8 on four GPUs (`57`, `57`, `57`, `57`, `57`, `57`, `57`, and `53` cases; slowest `14.91s`).

- Current M64 split-N `ld/st` i32-parity checkpoint, 2026-04-14 07:22 UTC: the identity M64 split-N immediate, auto-selection, and explicit `16x32bx2` equivalence checks now cover f32+i32 for `64xN`, `N in {2,4,8,16,32,64,128}`, preserving exact immediate and opcode equality checks for explicit `32x32b_splitn` and auto-selected `16x32bx2`. This is test-only coverage over already-supported `ld/st` lowering. Current runtime-matrix collection is `4998` tests: `cp=381`, `mma=852`, splitn/misc `=273`, `ld_red=880`, and `ldst=2612`; bucketed evidence now aggregates to `4552 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `42/4998`; no-PYTHONPATH full-file collect reported `4998`; focused M64 split-N selector passed all `42` cases across split-4 on four GPUs (`11`, `11`, `11`, and `9` cases; slowest `5.78s`).

- Current scaled-MMAv5 mixed fp4A clean-negative shape-parity checkpoint, 2026-04-14 07:19 UTC: the full-shape tile-permuted TMEM-LHS mixed `mxfp4/mxfp8` clean negative now covers `N in {64,128,256}`, `K=256`, and legacy plus canonical TMEM-linear accumulator layouts; the companion TMEM-LHS subview clean negative now covers `N in {64,128,256}`, `K in {128,256}`, and the same accumulator-layout pair. These remain clean unsupported because the current scaled-MMAv5 path requires `fp4_padded` shared-memory operand-A storage for mixed fp4A, not dense tensor-memory LHS storage. Current runtime-matrix collection is `4977` tests: `cp=381`, `mma=852`, splitn/misc `=252`, `ld_red=880`, and `ldst=2612`; bucketed evidence now aggregates to `4531 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `18/4977`; no-PYTHONPATH tight MMA collect selected `852/4977`; no-PYTHONPATH full-file collect reported `4977`; focused mixed-fp4A clean-negative selector passed all `18` cases across split-4 on four GPUs (`5`, `5`, `5`, and `3` cases; slowest `5.21s`).

- Current scaled-MMAv5 repeated-N32 clean-negative checkpoint, 2026-04-14 07:14 UTC: the direct `128x128/tile_n=32` repeated-N32 accumulator negative now covers every current scaled format pair and `K in {128,256}`, and the related accumulator-subview tile-permuted clean-negative now covers the same K depths for all format pairs. These remain clean unsupported because public tensor-memory scale descriptors expose matrix-B scale fragments at 64-column alignment. Current runtime-matrix collection is `4963` tests: `cp=381`, `mma=838`, splitn/misc `=252`, `ld_red=880`, and `ldst=2612`; bucketed evidence now aggregates to `4517 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `20/4963`; no-PYTHONPATH tight MMA collect selected `838/4963`; no-PYTHONPATH full-file collect reported `4963`; focused repeated-N32 clean-negative selector passed all `20` cases across split-4 on four GPUs (`5` per group; slowest `5.46s`). Discarded probe: one-CTA accumulator-subview `N=256,parentN=512` positives hit tensor-memory OOR (`Required: 524/536/560`, limit `512`) and remain omitted until a lower-TMEM parent/staging model exists.

- Current `ld.red` non-f32 min/max contract checkpoint, 2026-04-14 07:08 UTC: `tmem_ld_red_non_f32_contract_kernel` now exercises both `load_min` and `load_max`, so the clean non-f32 reduction diagnostics cover `min`/`max` for i32 plain reductions, i32 NaN/abs modifier rejections, and the legacy-unpacked f16 attempt. Current runtime-matrix collection is `4949` tests: `cp=381`, `mma=824`, splitn/misc `=252`, `ld_red=880`, and `ldst=2612`; bucketed evidence now aggregates to `4503 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `8/4949`; no-PYTHONPATH `ld_red` collect selected `880/4949`; focused non-f32 contract selector passed all `8` cases across split-4 on four GPUs (`2` per group; slowest `4.43s`).

- Current `ld.red` explicit N-sharded modifier-matrix checkpoint, 2026-04-14 07:06 UTC: the clean-negative test for explicit N-sharded `tcgen05.ld.red` register layouts (`16x64b`, `16x128b`, and `16x256b`) now covers `min`/`max`, `abs` false/true, and `PropagateNan.NONE/ALL` instead of only the default `min`/no-abs/no-NaN case. These layouts remain clean unsupported because reduction requires the N dimension in registers and M unsharded. Current runtime-matrix collection is `4945` tests: `cp=381`, `mma=824`, splitn/misc `=252`, `ld_red=876`, and `ldst=2612`; bucketed evidence now aggregates to `4499 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `24/4945`; no-PYTHONPATH `ld_red` collect selected `876/4945`; focused N-sharded clean-negative selector passed all `24` cases across split-4 on four GPUs (`6` per group; slowest `4.70s`). Discarded probe: adding the row-256 reduction-friendly `N=256` positive shape hits helper shared-memory OOR (`Required: 262148`, limit `232448`), so it remains omitted rather than reclassified as an ISA/compiler failure.

- Current `ld.red` identity-256 unsupported modifier-matrix checkpoint, 2026-04-14 07:02 UTC: the identity `256xN` TMEM-linear source-layout clean-negative test now covers `min`/`max`, `abs` false/true, and `PropagateNan.NONE/ALL` instead of only the default `min`/no-abs/no-NaN case. These layouts remain clean unsupported direct `tcgen05.ld.red` sources and must report the software-reduction diagnostic without PassManager/assertion noise. Current runtime-matrix collection is `4924` tests: `cp=381`, `mma=824`, splitn/misc `=252`, `ld_red=855`, and `ldst=2612`; bucketed evidence now aggregates to `4478 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `32/4924`; no-PYTHONPATH `ld_red` collect selected `855/4924`; focused identity-256 clean-negative selector passed all `32` cases across split-4 on four GPUs (`8` per group; slowest `5.15s`). A broad `ld_red` rerun remains deferred because this is a small negative-only test expansion over existing diagnostics.

- Current `warpx2` dense-shared clean-negative dtype-parity checkpoint, 2026-04-14 06:59 UTC: the single-CTA and two-CTA dense-shared clean-negative tests for no-scales `tcgen05.copy.warpx2::{01_23,02_13}.64x128b` now cover f32+i32 payloads instead of f32 only. This does not change the supported-positive or parked-frontier status: executable `warpx2` positives remain f32+i32 for single-CTA `01_23`, single-CTA `02_13`, and two-CTA `01_23`; two-CTA `02_13` and true scales `warpx2` still require a real descriptor/address/staging model. Current runtime-matrix collection is `4896` tests: `cp=381`, `mma=824`, splitn/misc `=252`, `ld_red=827`, and `ldst=2612`; bucketed evidence now aggregates to `4450 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `8/4896`; no-PYTHONPATH `cp` collect selected `381/4896`; focused dense-shared `warpx2` clean-negative selector passed all `8` cases across split-4 on four GPUs (`2` per group; slowest `4.35s`); full `cp` bucket passed `376` and skipped `5` across split-4 on four GPUs (`96`, `96`, `96`, and `93` selected; slowest `7.20s` on warm caches).

- Current broad two-CTA no-scales copy dtype-parity checkpoint, 2026-04-14 06:56 UTC: `CP_NO_SCALES_TWOCTA_CASES` is now a visible pytest matrix over legacy/canonical two-CTA destination layouts, f32+i32 payloads, `N`/swizzle shapes, exact `tcgen05.cp.cta_group::2.128x256b` counts, and multicast commit/barrier ordering. This replaces the previous hidden in-test f32-only loop, so pytest-split can schedule the 56 cases directly while the test also proves i32 parity for the broad two-CTA no-scales copy surface. Current runtime-matrix collection is `4892` tests: `cp=377`, `mma=824`, splitn/misc `=252`, `ld_red=827`, and `ldst=2612`; bucketed evidence now aggregates to `4446 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `56/4892`; no-PYTHONPATH `cp` collect selected `377/4892` through split-4; focused two-CTA no-scales selector passed all `56` cases across split-4 on four GPUs (`14` per group; slowest `11.36s`); full `cp` bucket passed `372` and skipped `5` across split-4 on four GPUs (`95`, `95`, `95`, and `92` selected; slowest `81.12s`).

- Current scaled-MMAv5 two-CTA accumulator-subview K-depth checkpoint, 2026-04-14 06:52 UTC: `SCALED_MMA_TWOCTA_ACC_SUBSLICE_K_CASES` now covers every current scaled format pair, `slice_start in {0,128}`, `blockK in {128,256}`, and multicast false/true for the supported two-CTA `blockN=128`, `parentN=256` accumulator-subview path. The test now pins exact scaled-copy and scaled-MMAv5 opcode counts as functions of `blockK // 128`, matching the adjacent one-CTA accumulator-subview K-depth contract. Current runtime-matrix collection is `4837` tests: `cp=322`, `mma=824`, splitn/misc `=252`, `ld_red=827`, and `ldst=2612`; bucketed evidence now aggregates to `4391 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused scaled two-CTA accumulator-subview collect selected `40` cases; no-PYTHONPATH tight MMA collect selected `824/4837`; no-PYTHONPATH full-file collect reported `4837`; focused scaled two-CTA accumulator-subview selector passed all `40` cases across split-4 on four GPUs (`10` per group; slowest `9.78s`); tight MMA selector passed all `824` cases across split-4 on four GPUs (`206` per group; slowest `21.26s`).

- Current plain-MMAv5 tile-permuted accumulator N64 clean-negative checkpoint, 2026-04-14 06:48 UTC: a direct `128x64/tile_n=16` tile-permuted accumulator layout is now pinned as a clean MMAv5-compatible-layout verifier boundary for every supported plain operand kind, `K in {32,64}`, and both no-accumulator plus `use_acc=True` paths. The existing positive tile-permuted accumulator surface remains `128x128/tile_n=32` and `128x256/tile_n=64`; N64 is not promoted as a positive because the verifier rejects its swapped 16/32 column-basis layout before lowering. Current runtime-matrix collection is `4817` tests: `cp=322`, `mma=804`, splitn/misc `=252`, `ld_red=827`, and `ldst=2612`; bucketed evidence now aggregates to `4371 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused tile-permuted accumulator collect selected `64` cases; no-PYTHONPATH tight MMA collect selected `804/4817`; no-PYTHONPATH full-file collect reported `4817`; focused tile-permuted accumulator selector passed all `64` cases across split-4 on four GPUs (`16` per group; slowest `25.16s`); tight MMA selector passed all `804` cases across split-4 on four GPUs (`201` per group; slowest `12.66s`).

- Current scaled-MMAv5 accumulator-subview K-depth parity checkpoint, 2026-04-14 06:42 UTC: `SCALED_MMA_ACC_SUBSLICE_N_CASES` now covers `K in {128, 256}` for every current scaled format pair, `N in {64, 128}`, and both root-aligned plus offset accumulator subviews. The exact PTX/LLIR opcode-count assertion now scales by `K // 128`, matching the existing root and TMEM-LHS scaled-MMAv5 K-depth contract. Current runtime-matrix collection is `4797` tests: `cp=322`, `mma=784`, splitn/misc `=252`, `ld_red=827`, and `ldst=2612`; bucketed evidence now aggregates to `4351 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused scaled accumulator-subview collect selected `40` cases; no-PYTHONPATH tight MMA collect selected `784/4797`; no-PYTHONPATH full-file collect reported `4797`; focused scaled accumulator-subview function passed all `40` cases across split-4 on four GPUs (`10` per group; slowest `17.98s`); tight MMA selector passed all `784` cases across split-4 on four GPUs (`196` per group; slowest `23.12s`).

- Current plain-MMAv5 full-shape TMEM-LHS `N=64` parity checkpoint, 2026-04-14 06:40 UTC: `MMA_LHS_TILE_PERMUTED_NK_CASES` now covers `N in {64, 128, 256}` for every supported plain operand kind and `K in {128,256}` through the full-shape tile-permuted TMEM-LHS path. The existing `tf32,N=256,K=256` omission remains because that direct shared-B helper exceeds shared memory; N64 is a positive target and passed numerically with exact PTX/LLIR opcode-count checks. Current runtime-matrix collection is `4777` tests: `cp=322`, `mma=764`, splitn/misc `=252`, `ld_red=827`, and `ldst=2612`; bucketed evidence now aggregates to `4331 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH full LHS tile-permuted collect selected `29` cases; no-PYTHONPATH tight MMA collect selected `764/4777`; no-PYTHONPATH full-file collect reported `4777`; focused LHS tile-permuted function passed all `29` cases across split-4 on four GPUs (`8`, `8`, `8`, `5`; slowest `14.25s`); tight MMA selector passed all `764` cases across split-4 on four GPUs (`191` each; slowest `22.81s`).

- Current plain-MMAv5 TMEM-LHS subview `N=64` parity checkpoint, 2026-04-14 06:37 UTC: `MMA_LHS_SUBSLICE_NK_CASES` now covers `N in {64, 128, 256}` for every supported plain operand kind, `K in {32,64}`, and both legacy plus canonical TMEM-linear accumulator layouts through the supported `ttg.memdesc_subslice` TMEM-LHS path. This closes the descriptor-view companion to the plain root/two-CTA/TMA-fed `N=64` MMAv5 coverage; exact opcode counts remain `_expected_plain_mma_op_count(kind, K)`. Current runtime-matrix collection is `4767` tests: `cp=322`, `mma=754`, splitn/misc `=252`, `ld_red=827`, and `ldst=2612`; bucketed evidence now aggregates to `4321 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `40/4767`; no-PYTHONPATH tight MMA collect selected `754/4767`; the full LHS-subview function passed all `60` cases across split-4 on four GPUs (`15` per group; slowest `19.74s`); tight MMA selector passed all `754` cases across split-4 on four GPUs (`189`, `189`, `189`, `187`; slowest `23.23s`). A discarded scaled-copy `blockN=64` probe showed the current public scales descriptor helper creates a zero B-scale descriptor dimension (`REP_MN = BLOCK_N // 128`), so do not add those as positives without a real sub-128 scale-descriptor/staging model.

- Current TMA-fed two-CTA TF32 `N=64` parity checkpoint, 2026-04-14 06:33 UTC: `test_tmem_runtime_matrix_mma_twocta_tma_tf32_reports_clean_shared_transpose_error` and `test_tmem_runtime_matrix_mma_twocta_tma_tf32_b_transposed_descriptor` now cover `blockN in {64, 128, 256}` for both legacy and canonical two-CTA TMEM-linear accumulator layouts. This closes the small-N companion to the existing `N=128/256` TMA-fed TF32 surface: the default `[K,N]` B TMA descriptor remains a clean transposed-float32 shared-operand negative, while the supported `[N,K]` descriptor plus shared `permute((1,0))` view is positive. Current runtime-matrix collection is `4747` tests: `cp=322`, `mma=734`, splitn/misc `=252`, `ld_red=827`, and `ldst=2612`; bucketed evidence now aggregates to `4301 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused `tma_tf32` collect selected `12/4747`; focused `tma_tf32` selector passed all `12` cases across split-4 on four GPUs (`3` per group; slowest `6.67s`); no-PYTHONPATH tight MMA collect selected `734/4747`; tight MMA selector passed all `734` cases across split-4 on four GPUs (`184`, `184`, `184`, `182`; slowest `5:26`).

- Current `ld.red` additional unsupported-layout checkpoint, 2026-04-14 06:23 UTC: the runtime matrix now pins clean diagnostics for two more unsupported reduction source layouts: M64 `64x64` TMEM-linear layout and block-basis `128x64` TMEM-linear layout. Each is covered across `min`/`max`, `abs` false/true, and `PropagateNan.NONE/ALL`; the M64 case must report the dedicated `tcgen05.ld.red-compatible` software-reduction diagnostic, and the block-basis case must report the descriptor-view register-layout diagnostic without PassManager/assertion noise. Current runtime-matrix collection is `4743` tests: `cp=322`, `mma=730`, splitn/misc `=252`, `ld_red=827`, and `ldst=2612`; bucketed evidence now aggregates to `4297 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `16/4743`; no-PYTHONPATH `ld_red` collect selected `827/4743`; focused unsupported-layout selector passed all `16` cases across split-4 on four GPUs (`4` per group; slowest `5.23s`). A broad `ld_red` rerun remains deferred because this is a small negative-only test expansion over existing diagnostics.

- Current `ld/st` broad mixed row/column i32 parity checkpoint, 2026-04-14 06:17 UTC: `LDST_ROWCOL_PERMUTED_CASES` now keeps the full f32 row/column cross-product and adds i32 for every non-identity broad row/column `128x{64,128,256}` layout, direct plus descriptor-chain, over every public `ld/st` variant. This completes staged broad row/column i32 parity after the earlier pure row/column slice; identity/identity i32 remains covered by the broad identity/mixed i32 tests rather than duplicated here. Current runtime-matrix collection is `4727` tests: `cp=322`, `mma=730`, splitn/misc `=252`, `ld_red=811`, and `ldst=2612`; bucketed evidence now aggregates to `4281 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused i32 mixed collect selected `270/4727`; no-PYTHONPATH `ldst` collect selected `2612/4727`; focused mixed row/column i32 selector passed all `270` cases across split-16 on four GPUs (groups 1-15 selected `17` cases each and group 16 selected `15`; slowest `7:15`). A full `ldst` bucket rerun remains deferred because no shared lowering changed.

- Current `ld/st` broad pure row/column i32 parity checkpoint, 2026-04-14 05:57 UTC: `LDST_ROWCOL_PERMUTED_CASES` now keeps the full f32 row/column cross-product and adds i32 for the broad pure-row and pure-column `128x{64,128,256}` subset, direct plus descriptor-chain, over every public `ld/st` variant. This deliberately decomposes the larger row/column parity problem: pure row/column i32 is now covered, while mixed non-identity row+column i32 remains the next duration-aware slice. Current runtime-matrix collection is `4457` tests: `cp=322`, `mma=730`, splitn/misc `=252`, `ld_red=811`, and `ldst=2342`; bucketed evidence now aggregates to `4011 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused i32 collect selected `180/4457`; no-PYTHONPATH `ldst` collect selected `2342/4457`; focused pure row/column i32 selector passed all `180` cases across split-8 on four GPUs (group sizes `23`, `23`, `23`, `23`, `23`, `23`, `23`, `19`; slowest `9:27`). A full `ldst` bucket rerun remains deferred because no shared lowering changed.

- Current `ld/st` broad scrambled/exotic i32 parity checkpoint, 2026-04-14 05:42 UTC: `LDST_EXOTIC_CASES` and `LDST_EXOTIC_DESCRIPTOR_CASES` now cover both f32 and i32 for `scrambled_cols` and `scrambled_rows_cols` layouts at `128x{64,128,256}`, direct plus descriptor-chain, over every public `ld/st` variant. This closes the smaller exotic-layout companion to the broad diagonal permutation i32 slice while keeping the exact expected opcode-shape checks. Current runtime-matrix collection is `4277` tests: `cp=322`, `mma=730`, splitn/misc `=252`, `ld_red=811`, and `ldst=2162`; bucketed evidence now aggregates to `3831 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `120/4277`; no-PYTHONPATH `ldst` collect selected `2162/4277`; focused exotic selector passed all `120` cases across split-8 on four GPUs (`15` per group; slowest `6:18`). A full `ldst` bucket rerun remains deferred because no shared lowering changed.

- Current `ld/st` broad diagonal permutation i32 parity checkpoint, 2026-04-14 05:32 UTC: `LDST_PERMUTED_CASES` now covers both f32 and i32 for the broad `128x{64,128,256}` diagonal row/column permutation matrix, including direct access and descriptor-chain compositions over every public `ld/st` variant. This adds dtype parity to identity plus `rotate1`, `even_odd`, and `reverse` diagonal layouts while preserving the exact expected opcode-shape checks. Current runtime-matrix collection is `4217` tests: `cp=322`, `mma=730`, splitn/misc `=252`, `ld_red=811`, and `ldst=2102`; bucketed evidence now aggregates to `3771 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `240/4217`; no-PYTHONPATH `ldst` collect selected `2102/4217`; the first split-4 focused run passed groups 1 and 2 (`60` each) but groups 3 and 4 timed out in the descriptor-heavy tail while making progress; the coverage-preserving split-8 rerun passed all `240` cases (`30` per group, slowest `12:34`). A full `ldst` bucket rerun remains deferred because no shared lowering changed.

- Current `ld/st` N=32 non-identity i32 parity checkpoint, 2026-04-14 05:00 UTC: `LDST_PERMUTED_N32_CASES`, `LDST_ROWCOL_N32_CASES`, and `LDST_EXOTIC_N32_CASES` now cover both f32 and i32 for direct and descriptor-chain `128x32` roundtrips over every public `ld/st` variant. This extends the earlier f32-only N32 diagonal, row/column, and scrambled layout slices to match the existing identity and two-CTA f32+i32 N32 coverage while keeping the exact minimal `x32/x16/x8/x4` opcode checks. Current runtime-matrix collection is `4097` tests: `cp=322`, `mma=730`, splitn/misc `=252`, `ld_red=811`, and `ldst=1982`; bucketed evidence now aggregates to `3651 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `160/4097`; no-PYTHONPATH `ldst` collect selected `1982/4097`; focused N32 non-identity selector passed all `160` cases across four GPUs (`40` each; group times about `7:19`, `8:26`, `9:47`, and `8:53`). A full `ldst` bucket rerun was intentionally deferred because this is a test-only parity expansion and the changed cases were covered directly.

- Current scaled-MMAv5 full-shape tile-permuted TMEM-LHS `N=64` parity checkpoint, 2026-04-14 04:46 UTC: `SCALED_MMA_LHS_TILE_PERMUTED_NK_CASES` now spans `N in {64, 128, 256}` for every currently reachable full-shape tile-permuted TMEM-LHS scaled format pair. Positive coverage includes all packed-storage reachable pairs at `K=256`, the mxfp8-storage LHS pairs at `K=128`, and both legacy plus canonical TMEM-linear accumulator layouts; the fp4-storage `K=128` tile-permuted LHS clean-negative matrix now also spans `N in {64, 128, 256}`. This closes the N64 companion to the scaled-LHS subview parity slice while keeping the fp4-storage K128 boundary as a clean verifier rejection instead of a backend crash. Current runtime-matrix collection is `4017` tests: `cp=322`, `mma=730`, splitn/misc `=252`, `ld_red=811`, and `ldst=1902`; bucketed evidence now aggregates to `3571 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `48/4017`; no-PYTHONPATH tight `mma` collect selected `730/4017`; focused scaled-LHS tile-permuted selector passed all `48` cases across four GPUs (`12` each); tight MMA runner passed all `730` cases across four groups (`183`, `183`, `183`, `181`).

- Current scaled-MMAv5 TMEM-LHS subview `N=64` parity checkpoint, 2026-04-14 04:43 UTC: `SCALED_MMA_LHS_SUBSLICE_NK_CASES` now spans `N in {64, 128, 256}` for the packed-storage reachable TMEM-LHS subview format pairs (`mxfp8/mxfp8`, `mxfp8/mxfp4`, `mxfp4/mxfp4`, and `nvfp4/nvfp4`), `K in {128, 256}`, and both legacy plus canonical TMEM-linear accumulator layouts. This closes the LHS-subview N64 parity gap after direct scaled-root N64 was added. Current runtime-matrix collection is `4001` tests: `cp=322`, `mma=714`, splitn/misc `=252`, `ld_red=811`, and `ldst=1902`; bucketed evidence now aggregates to `3555 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `48/4001`; no-PYTHONPATH tight `mma` collect selected `714/4001`; focused scaled-LHS subview selector passed all `48` cases across four GPUs (`12` each); tight MMA runner passed all `714` cases across four groups (`179`, `179`, `179`, `177`).

- Current direct-i8 MMAv5 clean-negative blockN parity checkpoint, 2026-04-14 04:40 UTC: the one-CTA and two-CTA direct-i8 clean-negative tests now span `blockN in {64, 128, 256}` for both legacy and canonical TMEM-linear accumulator layouts. The tests continue to assert the dedicated `direct tcgen05_mma kind::i8 is not supported on sm_... current Blackwell lowering` diagnostic and absence of PassManager/assertion noise. Current runtime-matrix collection is `3985` tests: `cp=322`, `mma=698`, splitn/misc `=252`, `ld_red=811`, and `ldst=1902`; bucketed evidence now aggregates to `3539 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH i8 collect selected `12/3985`; no-PYTHONPATH tight `mma` collect selected `698/3985`; focused i8 selector passed all `12` cases across four GPUs (`3` each); tight MMA runner passed all `698` cases across four groups (`175`, `175`, `175`, `173`).

- Current plain-MMAv5 two-CTA root/use-acc `N=64` checkpoint, 2026-04-14 04:39 UTC: `MMA_TWOCTA_PLAIN_KIND_CASES` now spans `blockN in {64, 128, 256}` for the standard two-CTA plain-MMAv5 root matrix. Coverage includes every supported plain operand kind, `blockK in {32,64}`, both no-accumulator and `use_acc=True` paths, and both legacy plus canonical two-CTA TMEM-linear accumulator layouts. This closes the positive two-CTA plain-MMAv5 N64 parity gap after the one-CTA, M64, and scaled-root N64 slices. Current runtime-matrix collection is `3979` tests: `cp=322`, `mma=692`, splitn/misc `=252`, `ld_red=811`, and `ldst=1902`; bucketed evidence now aggregates to `3533 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `120/3979`; no-PYTHONPATH tight `mma` collect selected `692/3979`; focused two-CTA plain/use-acc selector passed all `120` cases across four GPUs (`30` each); tight MMA runner passed all `692` cases across four groups (`173` each).

- Current plain-MMAv5 one-CTA root/use-acc `N=64` checkpoint, 2026-04-14 04:35 UTC: `MMA_PLAIN_KIND_ACC_CASES` now spans `N in {64, 128, 256}` for the standard `M=128` one-CTA root accumulator matrix. Coverage includes every supported plain operand kind (`f16`, `tf32`, `bf16`, `f8e5m2`, and `f8e4m3`), `K in {32,64}`, both no-accumulator and `use_acc=True` paths, and both legacy plus canonical TMEM-linear accumulator layouts. This closes the standard root plain-MMAv5 N64 parity gap after the M64 and scaled-root N64 slices. Current runtime-matrix collection is `3939` tests: `cp=322`, `mma=652`, splitn/misc `=252`, `ld_red=811`, and `ldst=1902`; bucketed evidence now aggregates to `3493 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `120/3939`; no-PYTHONPATH tight `mma` collect selected `652/3939`; focused plain root/use-acc selector passed all `120` cases across four GPUs (`30` each); tight MMA runner passed all `652` cases across four groups (`163` each).

- Current direct scaled-MMAv5 root-format `N=64` parity checkpoint, 2026-04-14 04:31 UTC: `SCALED_MMA_ROOT_FORMAT_CASES` now spans `N in {64, 128, 256}` for every current direct scaled-MMAv5 format pair (`mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`, `mxfp4/mxfp8`, and `nvfp4/nvfp4`), `K in {128, 256}`, and both legacy plus canonical TMEM-linear accumulator layouts. This closes the direct root accumulator `N=64` parity gap; earlier `N=64` coverage existed only through subview/specialized paths, not the root format matrix. Current runtime-matrix collection is `3899` tests: `cp=322`, `mma=612`, splitn/misc `=252`, `ld_red=811`, and `ldst=1902`; bucketed evidence now aggregates to `3453 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `60/3899`; no-PYTHONPATH tight `mma` collect selected `612/3899`; focused scaled-root selector passed all `60` cases across four GPUs (`15` each); tight MMA runner passed all `612` cases across four groups (`153` each).

- Current plain-MMAv5 `blockM=64, N=64` checkpoint, 2026-04-14 04:28 UTC: `test_tmem_runtime_matrix_mma_plain_kinds_m64` now covers `N in {64, 128, 256}` for one-CTA root `M=64` MMAv5, every supported plain operand kind, `K in {32,64}`, both no-accumulator and `use_acc=True`, and both legacy M64 sugar plus canonical M64 TMEM-linear accumulator layouts. The existing `_expected_m64_plain_mma_op_count(kind, K, acc_layout_kind, N)` logic naturally handles the new `N=64` surface: legacy M64 sugar emits one 64-column chunk, while canonical M64 linear keeps the root count. Current runtime-matrix collection is `3879` tests: `cp=322`, `mma=592`, splitn/misc `=252`, `ld_red=811`, and `ldst=1902`; bucketed evidence now aggregates to `3433 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH M64 collect selected `120/3879`; no-PYTHONPATH tight `mma` collect selected `592/3879`; focused M64 selector passed all `120` cases across four GPUs (`30` each); tight MMA runner passed all `592` cases across four groups (`148` each); `git diff --check` passed.

- Current plain-MMAv5 full-shape tile-permuted TMEM-LHS K-depth checkpoint, 2026-04-14 04:24 UTC: `tmem_mma_lhs_kernel` now takes `K` as a constexpr and `test_tmem_runtime_matrix_mma_lhs_tile_permuted` covers `K in {128, 256}` for the existing legal full-shape tile-permuted TMEM-LHS plain-MMAv5 surface. The new `K=128` cases use `tile_n=32`; the existing `K=256` cases keep `tile_n=64`. Coverage spans every supported plain operand kind at `N=128`, plus the non-OOR `N=256` shapes including `tf32,K=128`; only `tf32,N=256,K=256` remains omitted because this direct shared-B helper exceeds shared memory at that tile. Exact opcode counts now use `_expected_lhs_tile_permuted_mma_op_count(kind, K)`, so the matrix pins both half-depth and full-depth instruction counts. Current runtime-matrix collection is `3839` tests: `cp=322`, `mma=552`, splitn/misc `=252`, `ld_red=811`, and `ldst=1902`; bucketed evidence now aggregates to `3393 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `19/3839`; no-PYTHONPATH tight `mma` collect selected `552/3839`; focused selector passed all `19` cases across four GPUs (`5`, `5`, `5`, `4`); tight MMA runner passed all `552` cases across four groups (`138`, `138`, `138`, `138`); `git diff --check` passed. A provisional `N=128,tile_n=64` scaled-accumulator probe was discarded before commit because it is not a meaningful nontrivial linear tile permutation under the current helper: there is no second tile-selector bit to swap at exactly two 64-column tiles.

- Current plain-MMAv5 `blockM=64, N=256` checkpoint, 2026-04-14 04:16 UTC: `test_tmem_runtime_matrix_mma_plain_kinds_m64` now covers `N in {128, 256}` for one-CTA root `M=64` MMAv5, every supported plain operand kind, `K in {32,64}`, both no-accumulator and `use_acc=True`, and both legacy M64 sugar plus canonical M64 TMEM-linear accumulator layouts. The exact opcode-count helper now takes `N`; legacy M64 sugar emits one 64-column chunk per half (`N // 64` times the plain-kind K-depth count), while the canonical M64 linear layout keeps the root count. Current runtime-matrix collection is `3829` tests: `cp=322`, `mma=542`, splitn/misc `=252`, `ld_red=811`, and `ldst=1902`; bucketed evidence now aggregates to `3383 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH M64 collect selected `80/3829`; no-PYTHONPATH tight `mma` collect selected `542/3829`; representative f16 `N=256` M64 smoke passed `16` cases across four GPUs; full M64 selector passed all `80` cases (`20` per split group); tight MMA runner passed all `542` cases across four groups (`136`, `136`, `136`, `134`); `git diff --check` passed.

- Current `ld/st` broad two-CTA i32 checkpoint, 2026-04-14 04:08 UTC: `test_tmem_runtime_matrix_ldst_twocta_i32_broad_linear_layouts` adds integer `.b32` payload parity for the broad two-CTA `256x{64,128,256}` root surfaces. It covers `block_two_ctas` and MMAv5-like two-CTA TMEM-linear layouts, direct and supported descriptor-chain modes, and every public `ld/st` variant; existing broad f32 two-CTA coverage remains in the original direct/descriptor tests. Current runtime-matrix collection is `3789` tests: `cp=322`, `mma=502`, splitn/misc `=252`, `ld_red=811`, and `ldst=1902`; bucketed evidence now aggregates to `3343 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `60/3789`; no-PYTHONPATH `ldst` collect selected `1902/3789`; focused two-CTA i32 broad selector passed all `60` cases across four GPUs (`15` each; slowest cold descriptor-heavy group took about `5:04`); `git diff --check` passed.

- Current `ld/st` broad single-CTA i32 checkpoint, 2026-04-14 04:01 UTC: `test_tmem_runtime_matrix_ldst_i32_broad_linear_layouts` adds integer `.b32` payload parity for the existing broad single-CTA `128x{64,128,256}` root surfaces. It covers identity and mixed TMEM-linear layouts, direct and supported descriptor-chain modes, and every public `ld/st` variant; existing f32 coverage remains in the original direct/descriptor tests. Current runtime-matrix collection is `3729` tests: `cp=322`, `mma=502`, splitn/misc `=252`, `ld_red=811`, and `ldst=1842`; bucketed evidence now aggregates to `3283 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `60/3729`; no-PYTHONPATH `ldst` collect selected `1842/3729`; focused i32 broad selector passed all `60` cases across four GPUs (`15` each; cold descriptor-heavy group 4 took about `6:17`); `git diff --check` passed.

- Current `ld/st` two-CTA N=32 32-bit dtype checkpoint, 2026-04-14 03:52 UTC: `test_tmem_runtime_matrix_ldst_twocta_n32_linear_layout` now covers both f32 and i32 for `256x32` two-CTA TMEM-linear roundtrips, across direct access and the supported descriptor-chain path for both `block_two_ctas` and MMAv5-like two-CTA layouts, over every public `ld/st` variant. This reuses the dtype-generic direct/descriptor helper kernels from the prior single-CTA N=32 slice and pins the minimal `x32/x16/x8/x4` opcode families for the two-CTA layouts. Current runtime-matrix collection is `3669` tests: `cp=322`, `mma=502`, splitn/misc `=252`, `ld_red=811`, and `ldst=1782`; bucketed evidence now aggregates to `3223 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `40/3669`; no-PYTHONPATH `ldst` collect selected `1782/3669`; focused two-CTA N=32 selector passed all `40` cases across four GPUs (`10` each), with cold descriptor groups taking about `2:18`; `git diff --check` passed.

- Current copy `warpx2` diagnostic checkpoint, 2026-04-14 03:47 UTC: the remaining no-scales two-CTA `warpx2::02_13` clean-negative diagnostic now records the concrete missing requirement: support needs a `cta_group::2` descriptor/address schedule that preserves the high source-column bit. A scratch direct-PTX decomposition check confirmed that replacing the two-CTA copy with `cta_group::1` is not a viable workaround because the same function still contains `cta_group::2` TMEM alloc/dealloc and ptxas rejects mixed single-CTA and CTA-pair granularity (`uses single CTA(.cta_group::1) and CTA pair granularity(.cta_group::2) and that is not allowed`). Existing direct source-offset/destination-delta scans and true-scales source-offset scans remain negative, so the production boundary is still clean unsupported rather than an alias or software split. Validation: `make -j8` rebuilt `Ops.cpp` and relinked; py-compile passed; focused two-CTA `02_13` clean-negative selector passed the two non-empty split groups; nearby no-PYTHONPATH `-k warpx2` selector passed all `17` selected cases across four GPUs (`5`, `5`, `5`, `2`).

- Current `ld/st` identity N=32 32-bit dtype checkpoint, 2026-04-14 03:45 UTC: `test_tmem_runtime_matrix_ldst_identity_n32_linear_layout` now covers both f32 and i32 for canonical root `128x32` identity TMEM-linear roundtrips, across direct access and the supported descriptor-chain path for every public `ld/st` variant. The shared `tmem_ldst_auto_kernel`, `tmem_ldst_variant_kernel`, and `tmem_ldst_descriptor_chain_kernel` now allocate/bitcast TMEM with `in_ptr.dtype.element_ty`; this preserves existing f32 coverage while allowing the `.b32` ISA path to be exercised by integer payloads. Current runtime-matrix collection is `3649` tests: `cp=322`, `mma=502`, splitn/misc `=252`, `ld_red=811`, and `ldst=1762`; bucketed evidence now aggregates to `3203 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `20/3649`; focused identity N=32 selector passed all `20` cases across four GPUs (`5` each); four representative existing f32 direct/descriptor nodeids passed across four split groups; `git diff --check` passed.

- Current scales `warpx2` direct-source-offset checkpoint, 2026-04-14 03:31 UTC: completed the missing `sourceOffsetB128=73..119` high-half scan for tensor-memory-scales `tcgen05.copy` `warpx2::{01_23,02_13}` with `dst_delta=4`, filling the tracked evidence range to every offset `0..136`. The combined tracked result is now `274` unprimed single-message records with zero matches: offsets `0..8` execute but produce wrong data (`18` records), and offsets `9..136` all fault (`256` records). New shards are `probe_cp_scales_warpx2_offsets_73_84_dst4_gpu0.jsonl`, `..._85_96_dst4_gpu1.jsonl`, `..._97_108_dst4_gpu2.jsonl`, and `..._109_119_dst4_gpu3.jsonl`; compact summary is `probe_cp_scales_warpx2_offsets_dst4_current_summary.json`. No production lowering change is justified by this sweep; true scales `warpx2` remains a descriptor/address/view/staging frontier rather than a source-offset alias of the canonical `warpx4` scales copy.

- Current plain-MMAv5 `blockM=64` checkpoint, 2026-04-14 03:24 UTC: `test_tmem_runtime_matrix_mma_plain_kinds_m64` adds one-CTA root `M=64, N=128` MMAv5 coverage for every supported plain operand kind, both legacy M64 sugar and canonical M64 TMEM-linear accumulator layouts, `K in {32, 64}`, and both no-accumulator and `use_acc=True` paths. Exact opcode counts are pinned with `_expected_m64_plain_mma_op_count(kind, k, acc_layout_kind)`, including the legacy M64 sugar's doubled opcode stream relative to the canonical M64 linear layout. Current runtime-matrix collection is `3639` tests: `cp=322`, `mma=502`, splitn/misc `=252`, `ld_red=811`, and `ldst=1752`; bucketed evidence now aggregates to `3193 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `40/3639`; focused M64 selector passed all `40` cases across four GPUs (`10` each); no-PYTHONPATH tight `mma` collect selected `502/3639`; no-PYTHONPATH full-file collect selected all `3639`; tight `mma` runner passed all `502` cases across four groups (`126`, `126`, `126`, `124`); `git diff --check` passed. Remaining long-term coverage work: continue staged ISA saturation in another bounded family; the old legacy M64 xfail docket remains stale on current head, and this matrix now makes M64 MMAv5 coverage explicit in the runtime matrix.

- Prior `ld/st` exotic N=32 direct/descriptor checkpoint, 2026-04-14 03:20 UTC: `test_tmem_runtime_matrix_ldst_exotic_n32_linear_layout` adds root `128x32` f32 roundtrips for the existing scrambled-column and scrambled-row+column TMEM-linear layout families, covering both direct TMEM access and supported descriptor-chain views across every public `ld/st` variant. The slice pins the minimal `32x32b.x32`, `16x64b.x16`, `16x128b.x8`, and `16x256b.x4` opcode families in PTX/LLIR while keeping the descriptor path on `tensor_memory_linear`. Current runtime-matrix collection is `3599` tests: `cp=322`, `mma=462`, splitn/misc `=252`, `ld_red=811`, and `ldst=1752`; bucketed evidence now aggregates to `3153 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `20/3599`; no-PYTHONPATH `ldst` collect selected `1752/3599`; focused exotic N=32 selector passed all `20` cases across four GPUs (`5` each), with slow cold descriptor splits at about `92s` and `113s`; `git diff --check` passed. Remaining long-term coverage work: continue staged ISA saturation in another bounded family; reserve broad `ldst` runner refresh for shared-lowering changes or a larger accumulated `ld/st` checkpoint.

- Prior scaled-MMAv5 TMEM-LHS tile-permuted K=128 checkpoint, 2026-04-14 03:16 UTC: full-shape tile-permuted TMEM-LHS coverage now includes the reachable `K=128` positive surface for mxfp8-storage operand-A formats (`mxfp8/mxfp8` and `mxfp8/mxfp4`) at `N in {128, 256}` and both legacy/canonical accumulator layouts, while retaining the existing `K=256` coverage for every packed-storage reachable format pair. The fp4-storage `K=128` tile-permuted LHS boundary (`mxfp4/mxfp4` and `nvfp4/nvfp4`) is now pinned as a clean verifier negative because the resulting storage-width-64 tile-permuted descriptor is not MMAv5-compatible. Current runtime-matrix collection is `3579` tests: `cp=322`, `mma=462`, splitn/misc `=252`, `ld_red=811`, and `ldst=1732`; bucketed evidence now aggregates to `3133 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `32/3579`; no-PYTHONPATH tight `mma` collect selected `462/3579`; no-PYTHONPATH full-file collect selected all `3579`; focused scaled-LHS tile-permuted selector passed all `32` cases across four GPUs (`8` each); tight `mma` runner passed all `462` cases across four groups (`116`, `116`, `116`, `114`). Remaining long-term coverage work: continue staged ISA saturation in another bounded family, likely descriptor-view `ld/st` breadth or another scaled-MMAv5 clean-negative/positive parity gap.

- Prior plain-MMAv5 tile-permuted accumulator K-depth checkpoint, 2026-04-14 03:10 UTC: the tile-permuted accumulator matrices now cover `K in {32, 64}` for the simple f16 anchor and for every supported plain operand kind across no-accumulator and `use_acc=True` paths, using `128x128/tile_n=32` and `128x256/tile_n=64` canonical TMEM-linear accumulator layouts. `_expected_tile_permuted_mma_op_count(kind, k)` pins exact `tcgen05.mma` opcode counts as the existing tile-permuted base count times `K // 32`, so the new `K=64` cases prove doubled instruction depth. Current runtime-matrix collection is `3563` tests: `cp=322`, `mma=446`, splitn/misc `=252`, `ld_red=811`, and `ldst=1732`; bucketed evidence now aggregates to `3117 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `44/3563`; no-PYTHONPATH tight `mma` collect selected `446/3563`; no-PYTHONPATH full-file collect selected all `3563`; focused tile-permuted selector passed all `44` cases across four GPUs (`11` each); tight `mma` runner passed all `446` cases across four groups (`112`, `112`, `112`, `110`). Remaining long-term coverage work: continue staged ISA saturation in another bounded family, likely scaled-MMAv5 parity or descriptor-view `ld/st` breadth.

- Prior plain-MMAv5 TMEM-LHS subview K-depth checkpoint, 2026-04-14 03:07 UTC: `tmem_mma_lhs_subslice_kernel` now takes `K` as a constexpr and `test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds` covers `K in {32, 64}` for every supported plain operand kind, both legacy/canonical accumulator layouts, and `N in {128, 256}`. The exact `tcgen05.mma` opcode count uses `_expected_plain_mma_op_count(kind, k)`, so the new `K=64` subview cases prove doubled instruction depth through the supported `ttg.memdesc_subslice` TMEM-LHS path. Current runtime-matrix collection is `3541` tests: `cp=322`, `mma=424`, splitn/misc `=252`, `ld_red=811`, and `ldst=1732`; bucketed evidence now aggregates to `3095 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `40/3541`; no-PYTHONPATH tight `mma` collect selected `424/3541`; focused LHS-subview selector passed all `40` cases across four GPUs (`10` each); tight `mma` runner passed all `424` cases across four groups (`106` each). Remaining long-term coverage work: continue staged ISA saturation in another bounded family, likely scaled-MMAv5 parity or descriptor-view `ld/st` breadth.

- Prior `ld.red` tile-permuted checkpoint, 2026-04-14 03:04 UTC: `LD_RED_TILE_PERMUTED_CASES` now covers every legal helper tile width currently proven for tile-permuted `128xN` f32 reductions: `N=32/tile_n=8`, `N=64/tile_n in {8,16}`, `N=128/tile_n in {8,16,32}`, and `N=256/tile_n in {8,16,32,64}`. This adds five tile-permuted layouts and `40` runtime cases across min/max and all legal abs/NaN modifiers, while continuing to pin exact `32x32b` reduction opcodes, offsets, and wait ordering. A probe of the apparent 256-row `N=256` reduction-friendly layout hit shared-memory OOR (`262148` required vs `232448` limit), so that remains a non-executable boundary rather than a positive target. Current runtime-matrix collection is `3521` tests: `cp=322`, `mma=404`, splitn/misc `=252`, `ld_red=811`, and `ldst=1732`; bucketed evidence now aggregates to `3075 passed, 446 skipped`. Validation: scratch tile-permutation probe passed all `10` legal `(N,tile_n)` pairs for the baseline reduction; py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `80/3521`; no-PYTHONPATH `ld_red` collect selected `811/3521`; focused tile-permuted selector passed all `80` cases across four GPUs (`20` each); full `ld_red` runner passed all `811` cases across split-16/xdist groups. Remaining long-term coverage work: continue staged ISA saturation in another bounded family, likely broader MMAv5/scaled-MMAv5 parity or another descriptor-view `ld/st` slice, while leaving blocked copy `warpx2` frontiers parked until a real descriptor/address model exists.

- Prior plain-MMAv5 K-depth checkpoint, 2026-04-14 02:51 UTC: plain one-CTA and two-CTA root/use-acc matrices now cover `K in {32, 64}` for every supported operand kind, both legacy/canonical accumulator layouts, and `N in {128, 256}`. The expected `tcgen05.mma` opcode count is now pinned as the per-`K=32` base count times `K // 32`, so the new `K=64` cases prove the doubled instruction-depth surface for f16/bf16, tf32, and f8/f6/f4 families. Current runtime-matrix collection is `3481` tests: `cp=322`, `mma=404`, splitn/misc `=252`, `ld_red=771`, and `ldst=1732`; bucketed evidence now aggregates to `3035 passed, 446 skipped`. Validation: one-CTA and two-CTA scratch probes passed `K=64` for all plain kinds with doubled opcode counts; py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `160/3481`; focused K-depth selector passed all `160` cases across four GPUs (`40` each), slowest split about `1:27`; tight `mma` runner passed all `404` cases across four groups (`101` each). Remaining long-term coverage work: continue staged ISA saturation in another family or move to the next bounded MMA/scaled-MMA gap; no broad `ld/st` rerun is implied by this test-only MMA change.

- Prior `ld/st` N=32 row/column checkpoint, 2026-04-14 03:53 UTC: `test_tmem_runtime_matrix_ldst_rowcol_n32_linear_layout` adds representative non-diagonal row/column permutation coverage at root `128x32`: pure row permutation (`rotate1,identity`), pure column permutation (`identity,reverse`), and mixed row+column permutation (`even_odd,reverse`). Each case covers direct TMEM access and supported descriptor-chain views across every public `ld/st` variant, with exact PTX/LLIR minimal `x32/x16/x8/x4` opcode checks. Current runtime-matrix collection is `3401` tests: `cp=322`, `mma=324`, splitn/misc `=252`, `ld_red=771`, and `ldst=1732`; bucketed evidence now aggregates to `2955 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH collect for the focused selector selected `30/3401`; focused row/column N=32 selector passed all `30` cases across four GPUs (`8`, `8`, `8`, `6`), slowest split about `2:39`. Remaining long-term coverage work: continue bounded descriptor-view breadth and only run a broad `ldst` bucket after accumulated slices or shared-lowering source changes justify it.

- Prior two-CTA `ld/st` N=32 checkpoint, 2026-04-14 03:34 UTC: `test_tmem_runtime_matrix_ldst_twocta_n32_linear_layout` pins root `256x32` f32 roundtrips for both supported two-CTA TMEM-linear layout families (`block_two_ctas` and `mmav5_twocta`), both direct TMEM access and supported descriptor-chain views, and every public `ld/st` variant. The slice validates exact PTX/LLIR minimal `x32/x16/x8/x4` opcodes plus the `twoCTAs` TTGIR marker. Current runtime-matrix collection is `3371` tests: `cp=322`, `mma=324`, splitn/misc `=252`, `ld_red=771`, and `ldst=1702`; bucketed evidence now aggregates to `2925 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH collect for the focused selector selected `20/3371`; focused two-CTA N=32 selector passed all `20` cases across four GPUs (`5` each), slowest split about `1:35`. Remaining long-term coverage work: continue bounded `ld/st` row/column cross-product and view-chain slices before a broad `ldst` runner refresh.

- Prior `ld/st` N=32 permuted-layout checkpoint, 2026-04-14 03:20 UTC: `test_tmem_runtime_matrix_ldst_permuted_n32_linear_layout` extends the new root `128x32` f32 coverage from identity to diagonal row/column permutations (`rotate1`, `even_odd`, `reverse`) for both direct TMEM access and supported descriptor-chain views, across every public `ld/st` variant. The slice pins the same minimal `x32/x16/x8/x4` atom families and keeps exact PTX/LLIR equality. Current runtime-matrix collection is `3351` tests: `cp=322`, `mma=324`, splitn/misc `=252`, `ld_red=771`, and `ldst=1682`; bucketed evidence now aggregates to `2905 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH collect for the focused selector selected `30/3351`; focused permuted N=32 selector passed all `30` cases across four GPUs (`8`, `8`, `8`, `6`), with the slowest cold descriptor split taking about `2:58`. Remaining long-term coverage work: continue bounded `ld/st` layout/view breadth, especially two-CTA/minimal-N and row/column cross-product slices, before any broad `ldst` runner refresh.

- Prior `ld/st` N=32 identity checkpoint, 2026-04-14 03:03 UTC: `test_tmem_runtime_matrix_ldst_identity_n32_linear_layout` originally pinned the missing canonical identity `128x32` f32 roundtrip surface for both direct TMEM access and the supported descriptor-chain path, across `auto`, `32x32b`, `16x64b`, `16x128b`, and `16x256b`. The expected opcodes are the minimal `x32/x16/x8/x4` atom families already used by subview expectations, now executed as root identity `ld/st` cases. Current runtime-matrix collection is `3321` tests: `cp=322`, `mma=324`, splitn/misc `=252`, `ld_red=771`, and `ldst=1652`; bucketed evidence now aggregates to `2875 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH collect for the focused selector selected `10/3321`; focused N=32 selector passed all `10` cases across four GPUs. Remaining long-term coverage work: broaden `ld/st` layout/view fuzzing beyond this bounded identity slice, then continue staged `ld.red`, copy, and MMAv5/scaled-MMAv5 saturation.

- Prior scaled-MMAv5 TMEM-LHS tile-permuted checkpoint, 2026-04-14 02:42 UTC: `test_tmem_runtime_matrix_mma_scaled_lhs_tile_permuted_format_matrix` now covers `N=128` and `N=256` at `K=256` for every reachable full-shape tile-permuted LHS format pair and both legacy/canonical accumulator layouts. A provisional wider `K=128` probe found the fp4-storage tile-permuted LHS layouts are cleanly rejected as not MMAv5-compatible, so this slice keeps the positive matrix on the legal `K=256` surface rather than mixing partial format support. Current runtime-matrix collection is `3311` tests: `cp=322`, `mma=324`, splitn/misc `=252`, `ld_red=771`, and `ldst=1642`; bucketed evidence now aggregates to `2865 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH collect for the focused selector selected `16/3311`; focused LHS tile-permuted selector passed all `16` cases across four GPUs; tight `mma` runner bucket passed `324` cases across four groups (`81` each). Remaining long-term coverage work: keep mining bounded MMAv5/scaled-MMAv5 parity gaps, then continue `ld/st` fuzz/layout breadth and the descriptor/address/staging research frontiers for copy `warpx2`.

- Prior scaled-MMAv5 accumulator tile-permuted checkpoint, 2026-04-14 02:24 UTC: `test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_64_format_matrix` now covers `K=128` and `K=256` for every current scaled format pair while keeping the `128x256` accumulator tile-permuted layout and exact PTX/LLIR opcode checks. The expected scaled-MMAv5 opcode count is now `4 * (K // 128) * base_count`, so the `K=256` cases pin the doubled K-depth instruction surface. Current runtime-matrix collection is `3303` tests: `cp=322`, `mma=316`, splitn/misc `=252`, `ld_red=771`, and `ldst=1642`; bucketed evidence now aggregates to `2857 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH collect for the focused selector selected `10/3303`; focused tile-permuted scaled-accumulator selector passed all `10` cases across four GPUs; tight `mma` runner bucket passed `316` cases across four groups (`79` each); `git diff --check` passed. Remaining long-term coverage work: continue broader MMAv5/scaled-MMAv5 saturation, `ld/st` fuzz/layout breadth, and the descriptor/address/staging research frontiers for true scales `warpx2` and no-scales two-CTA `warpx2::02_13`.

- Prior copy `warpx2` frontier checkpoint, 2026-04-14: supported no-scales `warpx2` copy positives cover f32+i32 on single-CTA `01_23`, single-CTA `02_13`, and two-CTA `01_23`; the no-scales two-CTA `02_13` clean-unsupported boundary now also covers f32+i32. A widened direct-PTX destination-delta scan found zero valid two-CTA `02_13` direct-offset schedules across `1278` records (`292` executing duplicate-column variants, `986` failures), recorded compactly in `experiments/results/probe_cp_warpx2_02_13_twocta_dst_deltas_current_summary.json`. Current runtime-matrix collection is `3298` tests: `cp=322`, `mma=311`, splitn/misc `=252`, `ld_red=771`, and `ldst=1642`; bucketed evidence now aggregates to `2852 passed, 446 skipped`. Validation: py-compile passed; `make -j8` rebuilt cleanly; no-PYTHONPATH collect passed; exact two-CTA `02_13` boundary split passed both selected dtype cases; nearby no-scales `warpx2` selector passed `12`; full `cp` runner bucket passed `317`, skipped `5`; `git diff --check` passed. True scales `warpx2` and no-scales two-CTA `02_13` remain descriptor/address/staging research frontiers, not direct-offset fixes.

- Prior scaled-MMAv5 coverage checkpoint, 2026-04-14 01:24 UTC: single-CTA scaled-MMAv5 accumulator-subslice format coverage now includes `N=128` subviews at starts `0` and `128`, in addition to the existing `N=64` start-0/start-64 cases, across all five scaled format pairs. The format helper now sizes the parent accumulator descriptor as `2 * N`, preserving the existing `N=64` behavior while enabling the larger subview. Current runtime-matrix collection is `3294` tests: `cp=318`, `mma=311`, splitn/misc `=252`, `ld_red=771`, and `ldst=1642`; bucketed evidence now aggregates to `2848 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; focused scaled accumulator-subslice format slice passed `20` cases across four GPUs; tight `mma` runner bucket passed `311` cases across four GPUs; `git diff --check` passed.

- Prior `ld.red` coverage checkpoint, 2026-04-14 01:18 UTC: non-identity compatible reduction-source layouts now include the missing `N=32` shape for tile-permuted, pure column-permuted, pure row-permuted, and row+column-permuted families. This adds `128` positive runtime cases that all pin the `tcgen05.ld.red.sync.aligned.32x32b.x32` opcode family across min/max and all `abs`/NaN modifier combinations. Current runtime-matrix collection is `3284` tests: `cp=318`, `mma=301`, splitn/misc `=252`, `ld_red=771`, and `ldst=1642`; bucketed evidence now aggregates to `2838 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; focused new `N=32` slice passed `128` cases across four GPUs; full `ld_red` runner bucket passed `771` cases across 16 split groups; `git diff --check` passed.

- Prior speed/coverage checkpoint, 2026-04-14 01:03 UTC: the runtime-matrix runner now supports exact selected-shard reruns with `--groups` while preserving canonical group-to-GPU cache mapping, plus experimental per-bucket `--xdist-override`. Selected warm `ldst` groups 5 and 8 pass in about `11s` wall each, a 10x+ exact-rerun path versus the prior full-cold `ldst` group range of `208.7s..263.8s`, without reducing selected nodeids. At that point full collection was `3156` tests after the first post-speed ISA coverage slice: `tcgen05.cp.128x128b` covers f32 and i32 across legacy/linear single-CTA, linear indexed-view, and two-CTA paths. Validation: py-compile and `git diff --check` passed; `make -j8` no-op success; focused CP slice passed `17` cases across four GPUs; full `cp` runner bucket passed `313`, skipped `5`.

- Latest GB200 status checkpoint, 2026-04-14 00:46 UTC: current branch `codex/tmem`
  is clean and synced at `cfef1b94f`. Fresh current-HEAD checks after the
  runtime-matrix runner/duration-only commits: `make -j8` success,
  `make test-cpp` `240/240` passed, `make test-lit` `248 passed, 2 unsupported`,
  four-GPU split `python/test/gsan python/test/regression` aggregate
  `1110 passed, 216 skipped`, and `test-microbenchmark` passed with median
  launch overhead `22.5695`. A broad `python/test/unit` four-GPU refresh was
  stopped after three shards were green and the fourth was still making progress
  because no compiler/runtime source changed since the full GB200-green
  inventory; use the existing inventory plus targeted reruns for unchanged
  surfaces instead of spending dev time on blanket retests. Current conclusion:
  no known branch-caused GB200 CI failures; Proton still has the known
  merge-base-reproducing cudagraph / periodic-flushing failures and remains
  non-actionable for this TMEM branch.

- Prior runtime-matrix speed checkpoint, 2026-04-13 23:01 UTC: the five lifted `ld/st` descriptor roundtrip matrices that previously compiled to `OutOfResources` and then skipped are now marked as known pre-execution skips. This preserves instruction/op coverage because those `440` cases never emitted or executed `tcgen05` op coverage; they only consumed compile time before `pytest.skip`. Validation: py-compile passed; `make -j8` no-op success; `env -u PYTHONPATH pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k ldst` still collects `1642/3150`; the exact five skip-only functions now report `440 skipped in 2.40s`; full runner `ldst` passed with the same `1201 passed, 441 skipped`, while shard times improved from the prior `271.68s..350.27s` range to `194.71s..273.96s`.

- Superseded runtime-matrix validation-velocity checkpoint, 2026-04-13 22:15 UTC: the full `python/test/gluon/test_tmem_runtime_matrix.py` local sweep now has a coverage-preserving runner, `.codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py`, plus `tmem_runtime_matrix_validation_recipe_20260413.md`. Full collection remains `3150` tests. The runner partitions the matrix into `cp` (`312`), tight `mma` (`301`), exact-nodeid splitn/misc (`252`), `ld_red` (`643`), and `ldst` (`1642`) buckets, summing to the full matrix. The timeout diagnosis is cold compilation plus bad static partitioning, not a deadlock: representative `ldst` cold/warm timing was about `31s`/`3s`, and representative `ld_red` cold/warm timing was about `10s`/`3s`. Validation: `make -j8` no-op success; runner py-compile passed; runner `--dry-run` emitted deterministic commands; exact-nodeid `splitn` runner smoke passed `252` tests; full runner `ld_red` passed `643` tests with shard times from about `14s` to `96s`; full runner `ldst` passed `1201` and skipped `441` with shard times from about `4:32` to `5:50`. Current per-bucket evidence aggregates to full-matrix coverage: `2704 passed, 446 skipped` across all `3150` collected cases, without reducing the matrix.

- Latest TMEM runtime-matrix test/cache hygiene checkpoint, 2026-04-13 20:43 UTC: `test_tmem_runtime_matrix.py` no longer imports reusable helpers from `python.test.gluon.test_core`; shared TMEM helper kernels and descriptor builders now live in `python/test/gluon/tmem_test_utils.py` and are imported as a sibling helper. The two `tcgen05.cp` two-CTA codegen tests that previously launched `python -c`, injected `PYTHONPATH`, and forced a fresh temporary `TRITON_CACHE_DIR` now run in-process and reuse the shard's stable per-GPU cache. Current cache diagnosis: the active misbehavior was validation self-sabotage from cold temporary caches and child-process imports, not a proven on-disk cache-key collision. Validation: no-PYTHONPATH collect of the full runtime-matrix file reports `3150` tests; `python3 -m py_compile` for `tmem_test_utils.py`, `test_tmem_runtime_matrix.py`, and `test_core.py` passed; `make -j8` no-op success; exact affected/import-helper nodeids passed across four split groups with stable `/tmp/triton-cache-gpu<N>` caches (`1` selected case per group in the final four-nodeid split); `git diff --check` passed.

- Latest plain-MMAv5 TMEM-LHS N-shape checkpoint, 2026-04-13 after the scaled LHS-subview expansion: `test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds` now spans `N in {128, 256}` for every supported plain kind and both legacy/canonical accumulator layouts, while `test_tmem_runtime_matrix_mma_lhs_tile_permuted` spans `N in {128, 256}` for all non-OOR direct shared-B shapes. The only omitted full-shape tile entry is `tf32, N=256, K=256`, which exceeds shared memory for this helper (`Required: 262156`, limit `232448`). Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `git diff --check`; `make -j8` no-op success; exact two-nodeid matrix passed `29` selected cases across four GPU groups (`8`, `8`, `8`, `5`); tight `-k 'test_tmem_runtime_matrix_mma'` passed `301` selected cases across four GPU groups (`76`, `76`, `76`, `73`) in `59s`, `139s`, `123s`, and `131s`.

- Latest scaled-MMAv5 TMEM-LHS subview N/K checkpoint, 2026-04-13 after the scaled-copy multicast expansion: `test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view_format_matrix` now spans the packed-storage reachable format pairs (`mxfp8/mxfp8`, `mxfp8/mxfp4`, `mxfp4/mxfp4`, and `nvfp4/nvfp4`) at `N in {128, 256}` and `K in {128, 256}`, over both legacy and canonical TMEM-linear accumulator layouts. A scratch four-GPU probe confirmed the twenty-four new `N/K` combinations match the dequantized reference; the scaled-MMAv5 opcode count is now pinned as `(K // 128) * base_count`, doubling for `K=256`. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `git diff --check`; `make -j8` no-op success; exact expanded nodeid passed all `32` cases across four GPU split groups (`8` each); nearby `-k 'mma_scaled and lhs and subslice'` passed `35` selected cases (`9`, `9`, `9`, `8`); tight `-k 'test_tmem_runtime_matrix_mma'` passed `287` selected cases across four GPU groups (`72`, `72`, `72`, `71`) in `56s`, `120s`, `118s`, and `132s`.

- Latest scaled-MMAv5 copy-helper multicast checkpoint, 2026-04-13 after the `block_k=256` expansion: `test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix` now spans every current scaled format pair, `block_n in {128, 256}`, `block_k in {128, 256}`, `num_ctas in {1, 2}`, `multicast in {False, True}`, and both legacy/canonical accumulator layouts. A scratch four-GPU probe confirmed the eighty new `multicast=True` combinations match the dequantized reference; one-CTA cases do not emit TTGIR multicast while two-CTA multicast cases do. Exact scale-copy count remains `(1 + block_n // 128) * (block_k // 128) * (32 // vec_size)`, and scaled-MMA count remains `(block_k // 128) * base_count`. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `git diff --check`; `make -j8` no-op success; exact expanded nodeid passed all `160` cases across four GPU split groups (`40` each); nearby `-k 'cp_scales_warpx4_via_scaled_mma'` passed `192` cases (`48` each); broad `-k 'cp'` passed `307` with `5` skips (`73 passed, 5 skipped`; `78`; `78`; `78`).

- Latest direct scaled-MMAv5 root-format checkpoint, 2026-04-13 after the `K=256` expansion: one-CTA root accumulator coverage now spans every current scaled format pair (`mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`, `mxfp4/mxfp8`, and `nvfp4/nvfp4`) at `N in {128, 256}` and `K in {128, 256}`, over both legacy and canonical TMEM-linear accumulator layouts. A scratch four-GPU probe confirmed the twenty new `K=256` combinations match the dequantized reference; the scaled-MMA opcode count is `base_count * (K // 128)`, so `K=256` doubles the existing direct-root counts (`8` ops for fp8/mixed vector-1X cases and `4` ops for homogeneous fp4/nvfp4 cases). The runtime matrix now pins this through `SCALED_MMA_ROOT_FORMAT_CASES`. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `git diff --check`; `make -j8` no-op success; exact expanded nodeid passed all `40` parameters across four GPU split groups (`10` each); tight `-k 'test_tmem_runtime_matrix_mma'` passed `263` selected cases across four GPU groups (`66`, `66`, `66`, `65`) in `54s`, `92s`, `131s`, and `86s`.

- Latest plain-MMAv5 root-accumulator shape checkpoint, 2026-04-13 15:02 UTC: one-CTA root accumulator coverage now spans `N=128` and `N=256` for every supported plain operand kind (`f16`, `tf32`, `bf16`, `f8e5m2`, and `f8e4m3`), both legacy and canonical TMEM-linear accumulator layouts, and both no-accumulator and `use_acc=True` paths. A scratch probe confirmed the wider `128x256` root layouts use the same root opcode counts as `128x128`; the runtime matrix pins this root-only surface through `MMA_PLAIN_KIND_ACC_CASES`. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `make -j8` no-op success; focused `-k 'mma_plain_kinds_with_linear_acc or mma_plain_kinds_use_acc'` passed `40` selected cases across four GPU groups; tight `-k 'test_tmem_runtime_matrix_mma'` passed `223` selected cases across four GPU groups (`56`, `56`, `56`, `55`); `git diff --check` passed.

- Latest scaled-MMAv5 TMEM-LHS clean-negative checkpoint, 2026-04-13 14:57 UTC: full-shape tile-permuted TMEM-LHS now has the same mixed fp4-A contract coverage as the subview path. New `test_tmem_runtime_matrix_mma_scaled_lhs_tile_permuted_mixed_fp4a_reports_clean_unsupported` covers `mxfp4/mxfp8` over legacy and canonical TMEM-linear accumulator layouts at logical `K=256`, confirms the verifier reports the padded-storage fp4-A diagnostic, and checks that the failure is clean rather than a PassManager/assertion crash. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `make -j8` no-op success; exact new nodeid passed its two selected parameters across four GPU split groups; nearby `-k 'mma_scaled and lhs and tile_permuted'` passed `10` selected cases across four groups; tight `-k 'test_tmem_runtime_matrix_mma'` passed `203` selected cases across four GPU split groups (`51`, `51`, `51`, `50`).

- Latest copy-planner cleanup checkpoint, 2026-04-13 14:55 UTC: `getTMemCopyPlans` no longer offers the direct-seed `warpx2::02_13` plan for multi-CTA/block layouts. The direct seed remains available for the executable single-CTA `128x4` `02_13` path, but the two-CTA `256x4` case stays on descriptor-plan search only because repeated actual-layout probes show the direct-seed extension either duplicates the source-column pair, writes all zeros, or traps for unaligned TMEM deltas. The stale comment implying a representable `01_23`-style descriptor shape might be enough was replaced with the current invariant: descriptor representability alone is not proof; a future `02_13` fix must preserve the missing 4-byte source-column bit through a real descriptor/address schedule. Validation: `make -j8` rebuilt cleanly; the affected exact selector (`warpx2_02_13_candidate_positive or warpx2_02_13_twocta_candidate_reports_clean_unsupported or warpx2_01_23_twocta_positive`) passed on the three non-empty four-GPU split groups; broader `-k 'cp_no_scales_warpx2'` passed `8` cases across four GPU split groups; `lit -v test/TritonNvidiaGPU/invalid.mlir` passed; `git diff --check` passed.

- Latest scaled-MMAv5 two-CTA accumulator-subview `block_n=256` probe, 2026-04-13 after the tight MMA validation: do not expand `test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_matrix` to the obvious `parent_n=512` / `slice_start in {0,256}` shape. Representative `mxfp8/mxfp8` probes for both slice starts and multicast modes fail launch metadata with tensor-memory OOR (`Required: 524`, hardware limit `512`). A smaller `parent_n=384` would fit the desired offset range but is invalid for `_make_tmem_linear_layout_mmav5_twocta`, which currently requires power-of-two `N`. Keep current `block_n=128`, `parent_n=256`, `slice_start in {0,128}` coverage as the live offset-subview matrix unless a different legal parent layout is designed.

- Latest broad direct MMA/scaled-MMA runtime-matrix validation checkpoint, 2026-04-13 after the plain LHS `N` expansion: use `-k 'test_tmem_runtime_matrix_mma'` for the pure MMA/scaled-MMA family, not the historical `-k 'mma and not cp'` selector. Collect-only showed `-k 'mma and not cp'` also selects `ldst_*mmav5*` cases because `mmav5` contains `mma`, which explains earlier static-split imbalance. The tight selector now collects `301` cases and passed across four GPU split groups: `76`, `76`, `76`, and `73` passed.

- Superseded broad `tcgen05.cp` runtime-matrix validation checkpoint, 2026-04-13 after the scaled-copy multicast expansion: current head is green for `python/test/gluon/test_tmem_runtime_matrix.py -k 'cp'` across four GPU `pytest-split` groups. Aggregate result: `307 passed, 5 skipped` (`73 passed, 5 skipped`; `78 passed`; `78 passed`; `78 passed`). This supersedes older `227 passed, 5 skipped` broad-copy counts; the delta is expected from adding the all-format multicast scaled-copy matrix.

- Superseded scaled-MMAv5 copy-helper format/shape checkpoint, 2026-04-13 after the TMA TF32 negative expansion: the first copy-helper expansion covered `block_n=128` and `256` for every existing scaled format pair, both `num_ctas=1/2`, and both legacy/canonical accumulator layouts at `block_k=128`. The current latest checkpoint above extends this matrix to `block_k=256` and supersedes the old expected count formula.

- Latest TMA-fed two-CTA TF32 clean-negative checkpoint, 2026-04-13 after the direct-i8 clean-negative expansion: `test_tmem_runtime_matrix_mma_twocta_tma_tf32_reports_clean_shared_transpose_error` now covers `block_n=128` and `block_n=256` for both legacy and canonical TMEM-linear accumulator layouts. This matches the positive `[N, K]` descriptor plus shared-permute route and keeps the default `[K, N]` B TMA descriptor shape pinned as a clean transposed-float32 shared-operand diagnostic. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `make` no-op success; exact expanded negative nodeid across four GPU split groups (`4 passed` aggregate); nearby `-k 'tma_tf32'` selector passed all eight selected positive plus negative cases across the four groups.

- Latest direct-i8 MMAv5 clean-negative checkpoint, 2026-04-13 after the TMA TF32 descriptor expansion: `test_tmem_runtime_matrix_mma_twocta_i8_reports_clean_error` now covers `block_n=128` and `block_n=256` for both legacy and canonical TMEM-linear two-CTA accumulator layouts, matching the supported two-CTA plain-kind `blockN` surface. The test still asserts the dedicated `direct tcgen05_mma kind::i8 is not supported on sm_... current Blackwell lowering` diagnostic and absence of PassManager/assertion noise. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `make` no-op success; exact expanded two-CTA nodeid across four GPU split groups (`4 passed` aggregate); nearby `-k 'i8_reports_clean_error'` selector passed all six selected one-CTA plus two-CTA cases across groups 1-3, with group 4 selecting no tests.

- Latest TMA-fed two-CTA TF32 B-transposed descriptor checkpoint, 2026-04-13 after the scales `warpx2` scan: `test_tmem_runtime_matrix_mma_twocta_tma_tf32_b_transposed_descriptor` now covers both `block_n=128` and `block_n=256` for legacy and canonical TMEM-linear accumulator layouts. Scratch probes first confirmed that the `block_n=256` cases pass for both layouts and emit four matching PTX/LLIR `tcgen05.mma.cta_group::2.kind::tf32` ops. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `make` no-op success; exact expanded nodeid across four GPU `pytest-split` groups (`4 passed` aggregate); nearby `-k 'tma_tf32'` selector passed all six selected cases across groups 1-3 while group 4 selected no cases because the split had no remaining tests. This fills the immediate TMA-fed TF32 `blockN=256` positive gap; the default `[K, N]` TMA descriptor path remains the clean transposed-float32 shared-operand negative.

- Latest two-CTA `warpx2::02_13` corrected direct-PTX checkpoint, 2026-04-13 13:20 UTC: the durable probe `experiments/probe_cp_warpx2_02_13_twocta_direct_ptx.py` now uses compile-only warmup by default, with `--prime-canonical` retained only to reproduce historical runs that launched the canonical `warpx2::01_23` kernel first. Regenerated artifacts show zero matches against the extended two-CTA `02_13` oracle. All `22` named opcode/descriptor/two-message variants execute with finite output but duplicate a source-column pair (`duplicates_col_pair=true`). The four-GPU source-offset JSONL sweep covers `sourceOffsetB128` `36..127` with destination deltas `0/4`: `74` records for offsets `36..72` execute but all duplicate a source-column pair, while `110` records for offsets `73..127` launch-fail in isolated child processes. The older 2026-04-13 09:42 NaN/partial-duplicate breakdown is superseded because that probe launched the canonical kernel before patched cubins. Conclusion: keep public two-CTA `warpx2::02_13` clean unsupported; next useful work is descriptor/address-message synthesis from the layout model, not another direct-seed offset toggle.

- Latest two-CTA `warpx2::02_13` descriptor experiment, 2026-04-13 14:09 UTC: a
  temporary verifier probe compared the known-good `01_23` descriptor plan
  against `02_13`, then injected the `01_23`-style representable `32x4`
  descriptor candidate into the `02_13` fallback and temporarily bypassed the
  lowering row-stride assertion. This proved that descriptor representability
  alone is not sufficient: lowering emitted
  `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b`, but runtime output differed
  in all `1024` elements and duplicated source-column pairs (`[0,128,0,128]`
  style rows instead of the extended `02_13` oracle). All temporary source
  changes were removed and clean source was rebuilt; the restored two-copy
  controls passed (`01_23` positive and `02_13` clean unsupported selected on
  groups 1-2; groups 3-4 had no selected cases). Conclusion: the missing piece
  is not just adding a representable descriptor candidate or skipping the
  stride guard. Future support needs a true `02_13` descriptor/address-message
  schedule that changes the source mapping, or it should remain clean
  unsupported.

- Latest copy-planner debug checkpoint, 2026-04-13 13:43 UTC: temporary `TRITON_TMEM_COPY_DEBUG` instrumentation confirmed that two-CTA `warpx2::02_13` and the tensor-memory-scales `warpx2_candidate` are rejected in `TMEMCopyOp::verify`, before LLVM lowering. Two-CTA `02_13` classifies into three `warpx2::02_13` plans: direct-seed `64x4`, descriptor `64x4`, and fallback descriptor `32x4`; direct seed is unavailable for the `256x4` shared tile, and none of the `95/95/161` descriptor candidates is MMASMEM-representable. The known-good two-CTA `01_23` path succeeds through the `32x4` plan with a descriptor candidate equivalent to row bases `4,8,16,32,64,128` and col bases `1,2`; the analogous two-CTA `02_13` fallback has a zero low col basis and still fails descriptor representation. Single-CTA `02_13` succeeds only through the `128x4` direct seed. Combined with the corrected direct-PTX probes, this rules out blindly extending direct seed to `256x4`; the remaining work is a real descriptor/address-message model or continued clean unsupported.

- Latest scales `warpx2` direct-PTX correction checkpoint, 2026-04-13 13:18 UTC: the durable probe `experiments/probe_cp_scales_warpx2_direct_ptx.py` now uses compile-only warmup by default so patched variants are not primed by a prior canonical `warpx4` launch. The corrected results in `experiments/results/probe_cp_scales_warpx2_direct_ptx_current.jsonl` show the unpatched `warpx4_control` still roundtrips, but unprimed `first_01_23_only` and `first_02_13_only` only fill the low half of each logical row (`diff_count=510` for random, `512` for arange). Patching both original messages or using only the second message still launch-fails with illegal memory access. The old apparent positive was a primed-state artifact: `first_01_23_only --prime-canonical` can reproduce the historical success after launching the canonical copy first, but that is not valid support evidence. Conclusion: true scales `warpx2` remains unsupported/unproven; do not land an alias or opcode-suffix swap based on the earlier primed direct-PTX result.

- Latest subview-offset checkpoint, 2026-04-13 09:14 UTC: fixed a compiler assertion where
  TMEM subview base lowering used a non-surjective projected ld/st query layout
  as if it could invert arbitrary logical offsets. `getTMemViewOffsetForLowering`
  now uses raw query layouts for offset arithmetic only when they are
  surjective, otherwise it falls back to the descriptor type's full layout. The
  former `mmav5_twocta-64-auto` descriptor-roundtrip abort now reaches the
  existing OOR skip path, the exact `mmav5_twocta-64` roundtrip matrix has no
  failures, and new two-CTA direct-i8 MMA clean-negative coverage is in the
  runtime matrix. Validation details and split-timeout caveats are in
  `memory.md`, `log.md`, and the handoff tail.

- Latest `tcgen05.cp` contract checkpoint, 2026-04-13 07:10 UTC:
  runtime-matrix clean-negative coverage now also pins a representative scales
  copy from a larger shared parent subview. The new
  `test_tmem_runtime_matrix_cp_scales_shared_subslice_layout_reports_clean_unsupported`
  covers aligned `start_row` values `0` and `64` with the parent-row layout from
  the 384-layout `warpx2` subslice probe, and confirms the existing clean
  TensorMemoryScales descriptor-plan diagnostic is reported before lowering.
  Validation: `py_compile` passed, rebuild was a no-op success, the exact new
  nodeid passed both selected parametrized cases across split groups, and the
  nearby scales-copy selector passed (`9 passed` aggregate across active split
  groups). The previous no-scales transposed shared-source and invalid
  split-offset clean negatives remain covered.

- Latest `ld.red` contract checkpoint, 2026-04-13 06:20 UTC:
  runtime-matrix clean-negative coverage now pins the non-f32
  reduction/modifier boundary. New coverage rejects `i32` plain reductions with
  the f32-required diagnostic, rejects `i32` reductions with `NaN` / `abs`
  modifiers with the dedicated modifier-type diagnostics, and rejects a
  legacy-unpacked `f16` reduction attempt before lowering. Validation:
  `py_compile` passed, rebuild was a no-op success, the new nodeid passed
  across four GPU split groups (`4 passed` aggregate), the nearby negative
  selector passed (`11 passed` aggregate), and the broad four-GPU `-k ld_red`
  selector passed (`507 passed` aggregate). The broad split-4 run was badly
  imbalanced without duration data (reported shard times about `7:58`,
  `19:31`, `19:05`, and `13:57`), so future broad `ld_red` local validation
  should use finer/duration-aware groups or narrower selectors instead of
  treating a slow split-4 shard as normal.

- Latest M64 docket checkpoint, 2026-04-13 05:41 UTC: the historical
  `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]` bug/xfail item
  is stale at current head. After rebuild, the legacy parameter passed with
  `--runxfail`, both `test_block_m_64_mma` parameters passed normally across
  four `pytest-split` groups, and the nearby M64 subview/MMA selector
  (`test_tmem_subslice_block_m_64`,
  `test_tmem_subslice_block_m_64_parent_layout`, and `test_block_m_64_mma`)
  passed all six selected cases. Do not carry the old M64 xfail as a live
  branch-actionable failure unless a fresh exact repro fails.

- Latest plain-MMAv5 TMEM-LHS checkpoint, 2026-04-13 13:48 UTC: full-shape
  tile-permuted TMEM-LHS coverage now spans every supported plain operand kind
  (`f16`, `tf32`, `bf16`, `f8e5m2`, and `f8e4m3`) instead of only f16.
  `tmem_mma_lhs_kernel` now derives the operand dtype from the A pointer and
  accepts the kind-specific B shared-memory layout, so the widened test feeds
  a `128x256` tile-permuted TMEM-linear operand-A tile directly to
  `tcgen05_mma` and checks numeric output plus exact PTX/LLIR opcode counts
  (`f16/bf16=16`, `tf32=32`, `f8=8`). Validation: py-compile passed,
  rebuild was a no-op success, the exact widened nodeid passed all `5` cases
  across four GPU split groups (group 4 selected no cases because the nodeid
  has only five params), the nearby LHS selector passed `15` cases, and the
  nearby accumulator/LHS tile-permuted selector passed `35` cases across four
  GPUs. A broad `-k 'mma and not cp'` split-4 refresh had groups 3 and 4 green
  (`78` passed each in about `1:40`/`1:46`) but groups 1 and 2 hit the 300s
  guard while still emitting progress; treat that as the known no-duration
  partitioning issue, not as a product failure.

- Latest plain-MMAv5 saturation checkpoint, 2026-04-13 07:45 UTC:
  tile-permuted TMEM-linear accumulator coverage now also pins the `use_acc=True`
  accumulator-add path for all supported plain operand kinds (`f16`, `tf32`,
  `bf16`, `f8e5m2`, and `f8e4m3`) across the existing `128x128/tile_n=32` and
  `128x256/tile_n=64` layouts. The new runtime-matrix test checks numeric
  `matmul + accumulator` results, exact PTX/LLIR `tcgen05.mma` opcode streams,
  commit opcodes, and preservation of `tensor_memory_linear`. Validation:
  `py_compile` passed, rebuild was a no-op success, the exact new nodeid passed
  all `10` selected cases across four GPU split groups, the nearby
  `mma_plain_kinds_tile_permuted_acc or mma_plain_kinds_use_acc` selector passed
  `30` selected cases aggregate, and the post-commit broad four-GPU
  `-k 'mma and not cp'` selector passed `242 passed, 50 skipped`.

- Latest `ld/st` x1 subword checkpoint, 2026-04-13 08:35 UTC:
  descriptor-chain two-CTA x1 subword roundtrips are now pinned in addition to
  the direct two-CTA case. The new runtime-matrix test allocates a lifted
  `[2, M, N]` canonical two-CTA TMEM-linear layout, reaches the active tile
  through `slice/index/reshape/permute/permute`, and covers `f16`, `bf16`,
  `i16`, and packed `i8` for `auto` and explicit `32x32b`. It checks exact
  `tcgen05.{st,ld}.sync.aligned.32x32b.x1.b32` PTX/LLIR opcode streams,
  zero-offset addressing, numeric roundtrip equality, `memdesc_subslice`,
  `memdesc_reshape`, `memdesc_trans`, `tensor_memory_linear`, and
  `twoCTAs = true` TTGIR spelling. Validation: `py_compile` passed, rebuild
  was a no-op success, the exact new descriptor-chain nodeid passed `8`
  selected cases across four GPU split groups, and the nearby
  `ldst_x1_subword or ldst_x1_f32` selector passed `65` selected cases
  aggregate.

- Latest scaled-MMAv5 TMEM-LHS checkpoint, 2026-04-13 05:35 UTC: scaled
  MMAv5 TMEM-LHS subviews now address packed fp4 operand-A descriptors in
  storage-column coordinates for the K tile step, fixing homogeneous A-side
  fp4 positives for `mxfp4/mxfp4` and `nvfp4/nvfp4`. Positive coverage now
  includes `mxfp8/mxfp8`, `mxfp8/mxfp4`, `mxfp4/mxfp4`, and `nvfp4/nvfp4`
  across legacy and canonical TMEM-linear accumulators. The mixed
  `mxfp4/mxfp8` dense TMEM-LHS path no longer wrong-codes: it is a clean
  verifier negative because `mxf8f6f4` fp4 LHS requires the padded operand-A
  storage model currently represented by `fp4_padded` shared memory.
  Validation: rebuild passed, durable probe refreshed, focused positive plus
  negative matrix `10 passed`, nearby scaled-MMA selector `27 passed`, and
  broad four-GPU `-k 'mma and not cp'` selector `242 passed, 50 skipped`.

- Superseded copy-frontier checkpoint, 2026-04-13 09:42 UTC (corrected by the 13:20 compile-only warmup rerun): two-CTA `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b` remained intentionally clean unsupported and the direct-PTX search found zero oracle matches, but the exact NaN/duplicate breakdown is stale because the probe launched the canonical `01_23` kernel before patched cubins. Use the 13:20 corrected artifact summary above for current source-offset counts.

- Latest code checkpoint, 2026-04-13 09:24 UTC: direct two-CTA tensor-memory-scales `tcgen05.copy` now has pure-copy runtime-matrix coverage. The new `tmem_copy_scales_warpx4_twocta_kernel` roundtrips a `128x16xi8` CGA-shaped `TensorMemoryScalesLayout(cga_layout=[[1, 0]])` tile from a shared-linear `warpx4` layout with `block_bases=[[64, 0]]`, and asserts exact PTX/LLIR `tcgen05.cp.cta_group::1.warpx4.32x128b` opcodes plus no legacy `ttg.memdesc_reinterpret`. This is bounded supported-copy coverage only; true scales `warpx2` and two-CTA `warpx2::02_13` remain descriptor/address-model frontiers. Validation: py-compile passed, rebuild was a no-op success, focused `-k cp_scales_warpx4` passed `54` tests across four split groups, broad `-k cp` passed `167` with `5` skips across four split groups, and `git diff --check` passed.

- Latest `ld.red` code checkpoint, 2026-04-13 13:35 UTC: explicit compatible reduction-load layouts now cover the full legal modifier matrix (`abs` false/true and `PropagateNan.NONE/ALL`) for both `min` and `max`. The explicit identity and non-identity tests pass `abs`/`propagate_nan` through `tmem.load_{min,max}`, inject NaNs for the NaN-propagating cases, check PyTorch reduction semantics, and still pin `tcgen05.ld.red.sync.aligned.32x32b.x128` opcode canonicalization for `auto`, `32x32b`, `16x32bx2`, and `32x32b_splitn`. Validation: `py_compile` passed, rebuild was a no-op success, and the focused explicit `ld.red` selector (`ld_red_explicit_compatible or ld_red_explicit_n_sharded`) passed `163` selected cases across four GPU split groups (`41`, `41`, `41`, `40` passed) in `1:50` to `4:42` per group.

- Latest scaled-MMAv5 two-CTA multicast checkpoint, 2026-04-13 12:37 UTC: the existing two-CTA accumulator-subview format matrix now covers both scale-TMA paths by running `multicast=False` and `multicast=True`. The test passes the parameter through `mma_scaled_tcgen05_acc_subslice_copy`, preserves numeric output plus exact copy/MMA/commit opcode checks, and asserts TTGIR `{multicast}` appears only for the multicast path. Validation: py-compile passed, rebuild passed, the exact nodeid passed `20` cases across four GPU split groups, and the nearby `mma_scaled and subslice and format_matrix` selector passed `43` cases across four GPU split groups.

- Latest scaled-MMAv5 full-shape TMEM-LHS checkpoint, 2026-04-13 13:59 UTC: direct
  tile-permuted TMEM-LHS coverage now includes the packed-storage positive
  subset for scaled MMA. New `tmem_mma_scaled_lhs_tile_permuted_format_kernel`
  allocates operand A directly in a tile-permuted TMEM-linear descriptor, uses
  logical `K=256` so fp4-A storage has a reachable `128`-column tile, and feeds
  the descriptor directly to `tcgen05_mma_scaled` without a subview. The new
  matrix covers `mxfp8/mxfp8`, `mxfp8/mxfp4`, `mxfp4/mxfp4`, and
  `nvfp4/nvfp4` across legacy and canonical accumulator layouts, checking
  numeric output, exact PTX/LLIR scaled-MMA opcode/count agreement, commit
  opcodes, no `ttg.memdesc_subslice`, and preserved `tensor_memory_linear`.
  Validation: scratch probe showed `K=128` is positive only for mxfp8-A while
  `K=256` covers the packed-storage subset; py-compile passed, rebuild was a
  no-op success, exact nodeid passed `8` cases across four GPU split groups,
  and nearby `mma_scaled and (lhs or tile_permuted)` passed `30` cases across
  four GPUs.

- Latest scaled-MMAv5 tile-permuted accumulator checkpoint, 2026-04-13 12:43 UTC: the `tile_n=64` direct accumulator layout now has full scaled format coverage instead of the previous `mxfp8/mxfp8`-only test. `test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_64_format_matrix` covers `mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`, `mxfp4/mxfp8`, and `nvfp4/nvfp4` on the `128x256` tile-permuted accumulator layout, with numeric checks plus exact PTX/LLIR scaled-MMA opcode/count checks. Validation: py-compile passed, rebuild was a no-op success, the exact nodeid passed all `5` selected cases across four GPU split groups, and nearby `mma_scaled and tile_permuted` passed `11` cases across four GPU split groups.

- Latest direct MMA/scaled-MMA validation checkpoint, 2026-04-13 12:45 UTC: after the two scaled-MMAv5 coverage commits (`a18607fd9` and `8b3fec8d6`), the broad runtime-matrix selector `python/test/gluon/test_tmem_runtime_matrix.py -k 'mma and not cp'` is green across four GPU split groups: `258 passed, 50 skipped` aggregate (`32/45 skipped`, `72/5 skipped`, `77`, and `77` by group).

- Latest validation checkpoint, 2026-04-13 04:19 UTC: the current GB200 branch-actionable queue is closed from local evidence at `4fe39e5d28edf82d90bb9039049083a66359f9b6`. Build is a no-op success, full lit is green (`248 passed, 2 unsupported`), C++ is green (`240/240`), microbenchmark passes, the fused-attention tutorial is clean across four GPU split groups (`192 passed, 192 skipped`), the singleton instrumentation/plugin unit tails all pass, and the full `python/triton_kernels/tests` aggregate is recomputed as `2377 passed, 3444 skipped` with no failures/errors. Together with the earlier green unit, gsan, regression, examples, and closed Gluon runtime-matrix/lowerings evidence, there are no deterministic known branch-actionable GB200 failures. Proton's `11` cudagraph / periodic flushing failures remain classified as merge-base-preexisting environmental noise.

- Latest validation checkpoint, 2026-04-13 03:00 UTC: the previously incomplete Gluon tail is now closed from local evidence. `python/test/gluon/test_tmem_runtime_matrix.py` was covered by a mixed split plan totaling `2683` selected cases (`2237 passed, 446 skipped`, no failures/errors), and `python/test/gluon/test_lowerings.py` is green across four GPU shards (`4937 passed, 512 skipped`). Combined with the prior green `python/test/gluon/ python/tutorials/gluon/` groups 1-3, split-16 groups 13-14, the isolated xdist-crash nodeid pass, and green `python/examples/gluon/`, there are no deterministic `test-gluon` failures known after the MMAv5 family-addressing fix. The old long-running shards were static split imbalance around slow TMEM ldst legality/probe cases, not a reproduced compiler/runtime failure.

- Latest wider checkpoint, 2026-04-13 01:19 UTC: full lit is green and the examples half of
  `make test-gluon` is green across four GPU shards. The first Gluon phase has
  no deterministic known failures after the MMAv5 family-addressing fix, but
  the local validation is incomplete because the heavy runtime-matrix tail still
  exceeds the current split timeout: four-way `-n 6` groups 1-3 finished green,
  the group-4 worker-crash node passed isolated, split-16 groups 13-14 finished
  green, and split-16 groups 15-16 timed out while still making progress. Next
  action is a better duration-aware/runtime-tail split, not raising timeouts or
  classifying those timeouts as product failures.

- CI timing reference from the project owner, 2026-04-13: the full GB200 CI lane
  on a comparable four-node/four-GPU machine is expected to take about 35 minutes
  including clean build and LLVM download, with actual test time around 20
  minutes. Local validation that runs much longer should be treated as a
  partitioning, xdist, or hang investigation first, not normalized as acceptable
  sweep time.

- Latest checkpoint, 2026-04-12 23:15 UTC: a focused MMAv5 loader fix is ready after the
  post-attr-cleanup `test-gluon` run exposed tile-permuted MMAv5 runtime
  failures. Full-shape MMAv5 descriptors now use the derived family layout for
  instruction address arithmetic; marked physical bitcasts still use the exact
  typed result layout. The focused tile-permuted cluster, broad runtime-matrix
  `-k 'mma and not cp'`, nearby core M64/bitcast selector, representative
  `test_mma_shared_inputs` exacts, and relevant lit lowering checks are green.
  The previous tile-permuted MMAv5 current-head failure list is stale for those
  nodeids; run the next wider GB200/checkpoint sweep before using aggregate
  counts for new prioritization.

- The TMEM side-channel attribute cleanup slice is in progress after the
  preserve-set commit `37d00bb91`:
  - producer/provenance attributes are being removed instead of copied through
    alloc rewrites (`ttng.tmem_physical_layout`, `ttng.tmem_ldst_row_plan`,
    `ttng.tmem_mmav5_accumulator_root`, and
    `ttng.tmem_mmav5_operand_root`);
  - direct ld/st planning now preserves zero bases from the descriptor layout,
    treating them as broadcast/equivalence semantics rather than disposable
    physical storage;
  - the M64 MMAv5 full-shape backing row plan is recovered from the memdesc
    type/layout family and ordinary view-chain analysis, not from producer
    attrs;
  - focused lit, frontend, matmul, tensor-descriptor, Gluon M64, copy, and
    sampled `test_mma_shared_inputs` probes pass at current working tree;
  - the post-fix four-way `test_mma_shared_inputs` split sweep is green,
    so the earlier pre-fix failure count is now stale.
- Current pre-cleanup preserve-set at `3359982ee` is recorded in
  `gb200_preserve_set_20260412.md`:
  - build passed;
  - full lit is down to two stale pipeline FileCheck failures
    (`pipeline-loop-nest.mlir` and `pipeline-lower-loop.mlir`);
  - the `162` branch-new `python/test/unit` exact nodeids still all fail;
  - `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]` still fails;
  - Proton's `11` cudagraph / periodic flushing failures still reproduce and
    remain merge-base-preexisting noise;
  - fresh current-head wrapper/tail checks for C++, gsan, regression,
    microbenchmark, unit debug, fused-attention tutorial, plugins,
    instrumentation, and Proton tails are green.
- Full GB200 `integration-tests-nvidia` inventory at `cb76c31a0` is not green:
  - branch-new actionable failures were `9` lit files, `162` `python/test/unit`
    nodeids, and `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`;
  - those exacts pass on merge-base `11ee1144a737006921231bbd3386c187812c38e1`;
  - Proton's `11` cudagraph / periodic flushing failures reproduce on
    merge-base and are ignored as preexisting GB200 lane noise;
  - as of 2026-04-12 08:21 UTC, the five API/contract lit files from that
    `9`-file lit bucket pass focused lit after supported-spelling rewrites,
    leaving the four stale lit tests as the next lit-only recovery slice before
    a full `make test-lit` rerun.
- The latest current-head `python/examples/gluon` aggregate passed all four
  split groups, so `python/examples/gluon/01-attention-forward.py` is now
  covered by the examples aggregate as well as its focused SDPA-backed matrix.

- The supported `slice/subview -> bitcast` contract now has an explicit
  root-aligned and right-half regression in
  `test_tmem_physical_bitcast_preserves_subview_mapping`: the right-half case
  first selects columns `64:128` of the original f32 TMEM tile, then bitcasts
  that already-selected physical image to an f16 view, and verifies only the
  selected physical half is overwritten.
- Direct subword `ld/st` instruction-selection coverage now spans `f16`,
  `bf16`, `i16`, and `i8` over identity `128x{64,128,256}` layouts and every
  public variant in `{auto,32x32b,16x64b,16x128b,16x256b}`.
- Narrow x1 subword `ld/st` coverage now spans `f16`, `bf16`, and `i16`
  packed/unpacked cases plus packed and padded `i8` x1 cases for `auto` and
  `32x32b`; direct and descriptor-chain two-CTA x1 subword roundtrips now also
  cover `f16`, `bf16`, `i16`, and packed `i8` over canonical two-CTA
  TMEM-linear layouts.
- As of the latest focused coverage checkpoint, the current source/test slices
  add runtime-matrix allocator-lifetime anchors for `tcgen05.alloc`,
  `relinquish_alloc_permit`, `dealloc`, and `wait`, including pow2 alloc-size
  immediates for both CTA groups, plus exact `tcgen05.commit` opcode anchors.
- The latest full `python/test/gluon/test_tmem_runtime_matrix.py` file
  validation checkpoint remains green:
  - `1757 passed, 442 skipped in 1640.11s (0:27:20)`
- The preferred four-way heavy Gluon sweep over
  `python/test/gluon/test_core.py` and
  `python/test/gluon/test_tmem_runtime_matrix.py` is green at `be14fedc5`:
  - group 1:
    `4407 passed, 634 skipped, 15123 deselected`
  - group 2:
    `2528 passed, 2513 skipped, 15123 deselected`
  - group 3:
    `2694 passed, 2347 skipped, 15123 deselected`
  - group 4:
    `3314 passed, 1726 skipped, 15123 deselected, 1 xfailed`
- The preferred four-way `python/triton_kernels/tests` sweep is green at
  `2ad0ccf5e`:
  - group 1:
    `674 passed, 782 skipped, 4365 deselected`
  - group 2:
    `533 passed, 923 skipped, 4365 deselected`
  - group 3:
    `394 passed, 1062 skipped, 4365 deselected`
  - group 4:
    `776 passed, 677 skipped, 4368 deselected`
  - aggregate:
    `2377 passed, 3444 skipped, 17463 deselected`
- The clean Gluon examples subset remains green at `24bec4ecf`, and the
  attention file now has SDPA-backed benchmark-parameter unit coverage at
  current head:
  - `python/examples/gluon/01-attention-forward.py`:
    `64 passed in 102.66s (0:01:42)` over the benchmark grid except the
    SDPA-OOM context lengths (`N_CTX in {1024,2048,4096,8192}` for tests;
    benchmark remains `2**10..2**16`);
  - covered test parameters: `Z=4`, `H=32`, `HEAD_DIM in {64,128}`,
    `causal in {False,True}`, providers `triton-fp16` and `triton-fp8`, and
    `use_tmem_red in {False,True}` on this Blackwell Ultra box;
  - `python/examples/gluon/02-convolution.py`:
    `48 passed`
  - `python/examples/gluon/03-matmul-multicta.py`:
    `82 passed, 14 skipped`
  - `python/examples/gluon/04-2cta-block-scale-matmul.py`:
    `690 passed, 60 skipped`
  - full `python/examples/gluon` was rerun in the 2026-04-12 GB200 sweep at
    `cb76c31a0` and passed all four split groups.
- The current-head runtime-matrix saturation slices are green:
  - broad `ld/st`:
    `1181 passed, 441 skipped, 1027 deselected`
  - broad `tcgen05.cp`:
    `162 passed, 5 skipped, 2488 deselected`
  - broad `tcgen05.ld.red`:
    `487 passed, 2183 deselected`
  - true `tcgen05.mma` / direct `mma_scaled`:
    `226 passed, 50 skipped, 2407 deselected`
  - scaled-MMA copy-helper matrix:
    `52 passed, 2147 deselected` with exact copy, MMA, and commit opcode checks
- Allocator/lifetime coverage now has explicit runtime anchors:
  - single-CTA and two-CTA ld/st kernels assert exact PTX/LLIR
    `tcgen05.alloc`, `tcgen05.relinquish_alloc_permit`, `tcgen05.dealloc`,
    and `tcgen05.wait::{st,ld}` emission;
  - both CTA groups cover alloc/dealloc size immediates
    `32, 64, 128, 256, 512`;
  - the two-CTA case asserts cluster arrive/wait before dealloc;
  - a source-initialized `allocate_tensor_memory(..., value=...)` kernel
    round-trips the initialized values through a TMEM load.
- Tensor-memory allocation rounding coverage now pins raw live TMEM totals
  `96`, `192`, and `384` as rounded boundaries to supported module allocation
  sizes `128`, `256`, and `512`; literal non-pow2 `tcgen05.alloc` immediates
  are not the compiler contract.
- Commit-opcode coverage now has exact PTX/LLIR anchors:
  - single-CTA MMA pins
    `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
  - two-CTA copy and MMA pin
    `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`.
- The latest full four-way `python/test/gluon` sweep remains the green sweep
  recorded at `77c43f696` / source `be3cba0cd`; the newer `57a06c29b` and
  copy / `ld.red` / scaled-MMA coverage slices were validated with focused
  direct-i8 MMA tests, adjacent positive `tcgen05_mma` kind tests, the
  Blackwell conversion lit test, focused copy and `ld.red` runtime exacts, and
  the focused scaled-MMA opcode matrix.
- A fresh four-way `python/test/gluon` sweep from `be3cba0cd` is green:
  - group 1:
    `5448 passed, 1002 skipped, 19348 deselected`
  - group 2:
    `2666 passed, 3784 skipped, 19348 deselected`
  - group 3:
    `4408 passed, 2041 skipped, 19348 deselected, 1 xfailed`
  - group 4:
    `5582 passed, 866 skipped, 19350 deselected`
  - the current `gb200_current_branch_group{1,2,3,4}_latest_failures.txt`
    manifests are refreshed to `0` nodeids.
- The broad-sweep `.kind::i8` PTX assembler diagnostics are now addressed for
  the direct Gluon API path by `57a06c29b`: on `sm_103a+`,
  direct `tcgen05_mma` i8/u8 inputs with an int32 accumulator fail before IR
  lowering with a clean frontend diagnostic, while the generic
  `compute-capability=100` `ttng.tc_gen5_mma` i8 conversion coverage remains
  intact.
- The direct canonical TMEM-linear `128x128b` copy root coverage gap is closed
  by `c5bdb6d5c`:
  - `test_tmem_runtime_matrix_cp_128x128` now runs both the legacy
    `TensorMemoryLayout` destination and canonical `_make_tmem_linear_layout`
    destination;
  - both variants assert the exact `tcgen05.cp.cta_group::1.128x128b` PTX/LLIR
    family;
  - the linear variant also checks the TTGIR contains `tensor_memory_linear`.
- The standalone scaled `warpx4.32x128b` copy validation no longer depends on
  `_reinterpret` after `bf3dd781b`:
  - `test_tmem_copy_2d`, `test_tmem_runtime_matrix_cp_scales_warpx4`, and the
    scales layout probe validate the copied data through supported logical
    `TensorMemoryScalesLayout` loads;
  - the tests still assert exact `tcgen05.cp.cta_group::1.warpx4.32x128b`
    PTX/LLIR selection;
  - they now assert no `ttg.memdesc_reinterpret` appears in TTGIR for those
    kernels.
- Do not treat scales physical-inspection aliases as ordinary equal-size
  bitcasts: a direct `bitcast(...)` replacement exposed that the old dense
  inspection view was larger than the logical scales descriptor. Use logical
  scales loads for this test intent, and reserve physical bitcasts for the
  explicit offset/slice/subview cases where total size and physical mapping are
  equivalent.
- `128x256b` indexed-view copy coverage is broadened by `7075f31fc`:
  - the fit positive view shape `[128, 128]` now covers f32 and i32 payloads
    across 32, 64, and 128-byte shared swizzles;
  - the full `[2, 128, 256]` indexed parent is recorded as a tensor-memory OOR
    boundary (`Required: 1024, Hardware limit: 512`) instead of a positive
    target.
- No-scales `warpx2` copy exact coverage now includes:
  - single-CTA `warpx2::{01_23,02_13}.64x128b` candidate positives with exact
    `cta_group::1` copy and commit opcode checks plus runtime oracles;
  - two-CTA `warpx2::01_23.64x128b` executable coverage through public
    `SharedLinearLayout` + `TensorMemoryLinearLayout` objects, with exact
    `tcgen05.cp.cta_group::2.warpx2::01_23.64x128b` and multicast commit
    opcode checks;
  - dense shared-layout `warpx2` forms are now clean unsupported for both
    single-CTA and two-CTA cases because runtime probes showed the old
    codegen-only path emitted `warpx2` opcodes while copying the wrong data;
  - the two-CTA `warpx2::02_13` candidate shared layout is now pinned as a
    clean unsupported descriptor-synthesis frontier under the canonical public
    shared layout; it remains a layout-surface / direct-PTX frontier, not a
    proven ISA-impossible negative;
  - a 2026-04-11 temporary direct-seed relaxation for the canonical two-CTA
    `warpx2::02_13` source emitted
    `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b` but produced all-zero
    output, so the current clean unsupported boundary is protecting a real
    wrong-code path and should not be removed without a better descriptor/address
    model.
  - a follow-up direct-PTX patch sweep is recorded in
    `experiments/probe_cp_warpx2_02_13_twocta_direct_ptx.py` with results in
    `experiments/results/probe_cp_warpx2_02_13_twocta_direct_ptx_current.log`:
    opcode-only, source-row-plus-16, single-CTA seed, destination `+4`, and two
    message variants all assemble and launch through the cluster-aware Triton
    launcher, but none match the extended single-CTA `02_13` oracle; the
    descriptor/address mutations either duplicate source-column pairs or copy
    the wrong row/column mix, so they still do not recover the missing 4-byte
    source-column bit.
  - the corrected compile-only direct-seed source-offset JSONL scan covers
    offsets `36..127` with destination deltas `0` and `4`: all executing
    offsets `36..72` duplicate one source-column pair, offsets `73..127`
    launch-fail in isolated child processes, and no source-offset record matches
    the extended two-CTA `02_13` oracle. The earlier NaN-specific breakdown is
    superseded by the compile-only warmup rerun.
- The historical scales `warpx2` probe candidate is now pinned more precisely:
  under public `TensorMemoryScalesLayout` it classifies as
  `tcgen05.copy.warpx4.32x128b` and then hits the tensor-memory-scales
  descriptor-plan diagnostic, now from `TMEMCopyOp` verification rather than
  falling through to late LLVM lowering. `CP_SCALES_LAYOUT_PROBE_CASES` now also
  pins four nearby shared-linear basis-order variants as clean unsupported; all
  of them still map to `warpx4.32x128b` descriptor-plan failures rather than
  public `warpx2` scales lowering. A refreshed FD-capturing subslice probe over
  384 parent-row shared-linear basis interleavings and starts `[0, 32, 64]`
  likewise found `1152` clean unsupported outcomes, zero unknown or bug-like
  failures, and no `warpx2` opcode. The `warpx4` control remains the positive
  copy path for this public scales surface.
- No-scales `cta_group::2` dense copy coverage now includes both dense
  families:
  - `tcgen05.cp.cta_group::2.128x128b` for the fixed `256x4` two-CTA
    i32/f32-bitwidth shape through explicit `SharedLinearLayout`;
  - `tcgen05.cp.cta_group::2.128x256b` for the existing wider N matrix.
- `ld.red` modifier saturation is expanded by `eac11c719`:
  - positive identity, tile-permuted, column-permuted, and row-permuted linear
    layouts now cover all four legal modifier pairs:
    plain, propagate-NaN, abs, and abs+propagate-NaN;
  - mixed unsupported layouts check the same modifier matrix and continue to
    report the dedicated `tmem_load reduction source layout is not directly
    tcgen05.ld.red-compatible` diagnostic.
- `ld.red` row-256 source-boundary coverage is expanded:
  - reduction-friendly `256x{32,64,128}` TMEM-linear layouts with the extra
    row bit carried in the column/query frame are positive and still emit
    `tcgen05.ld.red.sync.aligned.32x32b`;
  - plain identity `256x{32,64,128,256}` TMEM-linear source layouts are pinned
    as clean unsupported cases with the same actionable software-reduction
    diagnostic.
- `ld.red` explicit reduction-load layout coverage is expanded:
  - compatible explicit variants `auto`, `32x32b`, `16x32bx2`, and
    `32x32b_splitn` all execute correctly and still canonicalize to the exact
    `tcgen05.ld.red.sync.aligned.32x32b.x128.{min,max}.f32` opcodes;
  - explicit N-sharded variants `16x64b`, `16x128b`, and `16x256b` are pinned
    as clean unsupported cases with the dedicated `N dimension sharded across
    threads` diagnostic and register-layout note;
  - focused min/max explicit-layout validation: `11 passed in 8.57s`;
  - broad `ld_red` validation: `487 passed, 2183 deselected in 482.92s
    (0:08:02)`.
- Scaled-MMAv5 copy-matrix instruction coverage is tightened by `3374bbf12`:
  - the runtime-matrix scaled copy tests still assert exact
    `tcgen05.cp.cta_group::{1,2}.warpx4.32x128b` PTX/LLIR streams;
  - the same format/CTA/geometry/accumulator-layout matrix now also asserts
    exact `tcgen05.mma.cta_group::{1,2}.kind::{mxf8f6f4,mxf4,mxf4nvf4}.block_scale.scale_vec::*`
    opcode selection for PTX and LLIR.
- Tile-permuted MMAv5 accumulator coverage now spans all plain supported
  operand kinds:
  - `f16`, `tf32`, `bf16`, `f8e5m2`, and `f8e4m3`;
  - each kind is covered for `128x128` / `tile_n=32` and `128x256` /
    `tile_n=64` accumulator layouts;
  - both no-accumulator and `use_acc=True` accumulator-add paths are covered;
  - each path checks PTX/LLIR opcode equality and the expected
    `tcgen05.mma` kind, with commit opcode checks on the `use_acc=True` slice.
- Direct scaled-MMAv5 TMEM-view tests now pin exact `mxf8f6f4` scaled-MMA
  opcodes after `5e3b2ae87`:
  - this covers the existing minimal, block-N direct-layout, accumulator
    subview, LHS subview, and tile-permuted accumulator runtime tests;
  - the original e5m2/e5m2 direct-view kernels remain in place.
- Direct scaled-MMAv5 accumulator-subview format coverage is broadened by
  `602fd9b44`:
  - the new direct subview matrix covers `mxfp8/mxfp8`, `mxfp4/mxfp4`,
    `mxfp8/mxfp4`, `mxfp4/mxfp8`, and `nvfp4/nvfp4`;
  - it validates numeric output, exact PTX/LLIR
    `mxf8f6f4` / `mxf4` / `mxf4nvf4` scaled-MMA opcodes, and the
    `ttg.memdesc_subslice` + `tensor_memory_linear` accumulator path.
- That direct scaled-MMAv5 accumulator-subview format matrix now covers both
  `slice_start=0` and `slice_start=64`, so the same format/opcode assertions
  exercise root-aligned and offset accumulator subviews.
- Direct scaled-MMAv5 TMEM-LHS format coverage now includes both subview and
  full-shape tile-permuted packed-storage reachable subsets:
  - `test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view_format_matrix`
    covers `mxfp8/mxfp8`, `mxfp8/mxfp4`, `mxfp4/mxfp4`, and `nvfp4/nvfp4`
    for both legacy and canonical TMEM-linear accumulator layouts;
  - `test_tmem_runtime_matrix_mma_scaled_lhs_tile_permuted_format_matrix`
    covers the same format/layout subset for a full-shape tile-permuted
    operand-A descriptor at logical `K=256`, with no `ttg.memdesc_subslice`;
  - both tests feed TMEM operand-A descriptors directly to `tcgen05_mma_scaled`
    and pin exact PTX/LLIR scaled-MMA and commit opcodes;
  - mixed `mxfp4/mxfp8` dense TMEM-LHS subviews and full-shape
    tile-permuted TMEM-LHS descriptors are pinned as clean unsupported cases
    because `mxf8f6f4` fp4 LHS requires padded operand-A storage currently
    represented by `fp4_padded` shared memory; the durable subview probe is
    `experiments/probe_mma_scaled_lhs_subslice_formats.py` with current results
    in `experiments/results/probe_mma_scaled_lhs_subslice_formats_current.log`.
- Two-CTA direct scaled-MMAv5 accumulator-subview coverage is now present:
  - `test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_matrix`
    reuses the cga-aware TMA/scales-copy topology, allocates a larger
    TMEM-linear accumulator parent, slices it to the `cta_group::2` MMA result
    tile, and covers both `slice_start=0` and `slice_start=128`;
  - it validates numeric output, exact PTX/LLIR scaled-MMA opcodes for the same
    five format pairs, exact `tcgen05.cp.cta_group::2.warpx4.32x128b` scale
    copies, exact multicast `tcgen05.commit.cta_group::2`, and
    `ttg.memdesc_subslice` + `tensor_memory_linear` + `two_ctas` TTGIR.
- Plain MMAv5 kind saturation now includes f16 in the explicit 1-CTA and
  2-CTA kind matrices after `7766be003`, the one-CTA root accumulator matrix
  now spans `N=128` and `N=256`, and the current two-CTA matrix has been
  widened to `N=128` and `N=256`:
  - `MMA_PLAIN_KINDS` covers `f16`, `tf32`, `bf16`, `f8e5m2`, and `f8e4m3`;
  - both legacy and canonical linear accumulator layouts are covered for
    `cta_group::1` and `cta_group::2`;
  - the `cta_group::1` root kind matrix covers both `128x128` and `128x256`
    accumulator shapes, and the `cta_group::2` kind matrix covers both
    `256x128` and `256x256` accumulator shapes;
  - the tests continue to assert exact PTX/LLIR opcode agreement and pin exact
    root op counts: `f16=2`, `bf16=2`, `tf32=4`, and `f8e5m2/f8e4m3=1`.
- Plain MMAv5 `use_acc=True` coverage now spans all `MMA_PLAIN_KINDS`
  across both 1-CTA and 2-CTA paths, for both legacy and canonical
  TMEM-linear accumulator layouts, validating runtime accumulator addition
  plus exact PTX/LLIR opcode agreement and the same exact root op counts.
- Plain MMAv5 tile-permuted accumulator coverage now also pins its expected
  expanded instruction counts: four times the root kind count for accumulator
  tile permutations. Full-shape tile-permuted TMEM-LHS coverage now spans all
  `MMA_PLAIN_KINDS` for a `128x256` operand-A tile and pins eight times the root
  kind count (`f16/bf16=16`, `tf32=32`, `f8e5m2/f8e4m3=8`).
- Plain MMAv5 TMEM-LHS subview coverage now spans all `MMA_PLAIN_KINDS` for
  `N in {128, 256}` and both legacy/canonical TMEM-linear accumulator layouts:
  - `test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds` slices the
    right half of a TMEM-linear operand-A parent and feeds that subview directly
    to `tcgen05_mma`;
  - the matrix pins exact PTX/LLIR opcodes, expected instruction counts
    (`f16=2`, `bf16=2`, `tf32=4`, `f8e5m2/f8e4m3=1`), and the single-CTA
    commit opcode for every supported plain kind.
- Plain full-shape tile-permuted TMEM-LHS coverage now includes `N=256` for all
  non-OOR direct shared-B shapes. The `tf32, N=256, K=256` direct helper shape
  is omitted because it exceeds shared memory before launch.
- Staged `ld/st` fuzzing has started with descriptor-chain `auto` coverage at
  `ebb23b697`:
  - the basic descriptor-composition matrix now covers `auto` instruction
    selection in addition to explicit `32x32b`, `16x64b`, `16x128b`, and
    `16x256b`;
  - coverage spans identity and mixed TMEM-linear layouts at `N = 64, 128,
    256`.
- Two-CTA descriptor-chain `auto` coverage is added by `ca9f94760`:
  - `LDST_TWOCTA_DESCRIPTOR_CASES` now uses the same `LDST_VARIANTS` matrix;
  - coverage reaches both `block_two_ctas` and MMAv5-like two-CTA layouts;
  - the validation also reran the single-CTA descriptor roundtrip sweep now
    reached by `LDST_DESCRIPTOR_CASES`.
- Higher-rank `ld/st` index descriptor views now include `auto` instruction
  selection after `7df887318`:
  - auto subview shapes are pinned to the expected `32x32b` family;
  - single-CTA identity/mixed and two-CTA block/MMAv5-like layouts are covered
    for `N = 64, 128`.
- Higher-rank multidimensional-slice `ld/st` descriptor views now include
  `auto` instruction selection after `81b4e5a9d`:
  - single-CTA identity/mixed and two-CTA block/MMAv5-like layouts are covered
    for `N = 64, 128`;
  - the tests continue to check functional output, PTX/LLIR agreement, and the
    expected reshape/transpose/split/join IR markers.
- Higher-rank clean OOR guards now include `auto` after `e2d793726`:
  - single-CTA and two-CTA higher-rank index, multidimensional-slice, and
    dim0-slice resource-boundary tests all use `LDST_VARIANTS`;
  - validation confirms these remain clean tensor-memory OOR boundaries.
- Direct higher-rank `ld/st` access is now pinned as a clean unsupported
  boundary:
  - direct rank-3 `get_reg_layout`, explicit `load`, and explicit `store`
    attempts fail in the Gluon descriptor API with a 2D-only diagnostic telling
    users to index/slice/reshape higher-rank TMEM descriptors to a 2D view;
  - the C++ `getDistributedLayoutForTmemLdSt` planners return no layout for
    non-2D physical TMEM layouts instead of reaching the old `dims.size() == 2`
    assertion;
  - focused validation: `4 passed in 3.14s`, plus adjacent higher-rank indexed
    and dim0-slice positives `30 passed in 25.88s`;
  - broader `ldst and higher_rank` validation: `144 passed, 1 skipped,
    2521 deselected in 73.43s (0:01:13)`.
- Higher-rank half-row clean errors now include `auto` after `7c1a6f63b`:
  - single-CTA and two-CTA lifted half-row clean-negative matrices now use
    `LDST_VARIANTS`;
  - explicit variants keep the existing unsupported-layout `CompilationError`
    contract, while `auto` may report the direct descriptor parser
    `RuntimeError` about translated row-half TMEM origins;
  - the tests still require a descriptor-view diagnostic and guard against
    PassManager/assertion crashes.
- Higher-rank dim0-slice positive `ld/st` views now include `auto` after
  `572fb25b2`:
  - single-CTA identity and two-CTA block lifted-layout dim0-slice matrices now
    use `LDST_VARIANTS`;
  - the tests validate functional output plus full-tile and narrowed-subview
    PTX/LLIR opcode agreement.
- The two-CTA MMAv5-like higher-rank `ld/st` clean-negative matrix now includes
  `auto` after `8f622b938`:
  - the unsupported matrix uses `LDST_VARIANTS`;
  - validation confirms auto still fails through the clean CTA-per-CGA mismatch
    path rather than PassManager/assertion crashes.
- Same-row/column permuted `ld/st` layouts now include `auto` after
  `c83d358e4`:
  - `LDST_PERMUTED_CASES` uses `LDST_VARIANTS`;
  - this reaches both direct ld/st and descriptor-chain composition runtime
    tests;
  - validation covers numerical behavior, `tensor_memory_linear` IR, and exact
    PTX/LLIR opcode agreement.
- Exotic `ld/st` layout matrices now include `auto` after `20f2c8db8`:
  - direct exotic, descriptor-chain exotic, and clean unsupported block-layout
    matrices use `LDST_VARIANTS`;
  - validation covers positive opcode agreement and clean negative diagnostics.
- Row/column cross-product permuted `ld/st` layouts now include `auto` after
  `6d44718e7`:
  - `LDST_ROWCOL_PERMUTED_CASES` uses `LDST_VARIANTS`;
  - this reaches direct ld/st and descriptor-chain composition sweeps;
  - focused validation ran the newly added auto parametrizations across the
    row/column permutation cross-product.
- Row/column descriptor-roundtrip `ld/st` boundary coverage now includes
  `auto` after `00139772b`:
  - `LDST_DESCRIPTOR_ROUNDTRIP_ROWCOL_CASES` uses `LDST_VARIANTS`;
  - focused validation shows the new auto parametrizations all hit the existing
    clean tensor-memory OOR skip boundary for lifted roundtrip shapes.
- The stale unused `LDST_EXPLICIT_VARIANTS` / higher-rank positive case list is
  removed after `a487942ac`, so scans no longer report a false explicit-only
  active `ld/st` matrix.
- Rank-5 descriptor-roundtrip `ld/st` boundary coverage now includes `auto`
  after `be193b6c6`:
  - one-CTA rank-5 cases use `LDST_VARIANTS`;
  - two-CTA rank-5 cases add `auto` to the existing explicit subset;
  - focused validation shows all new auto cases hit the existing clean
    tensor-memory OOR skip boundary.
- F16 subword ld/st pack/unpack coverage now includes `auto` after
  `4fe7a44de`:
  - `F16_LDST_CASES` uses `LDST_VARIANTS`;
  - the `auto` shape expectations match the explicit `32x32b` f16 packet
    family.
- X1 f32 direct and descriptor-chain ld/st roundtrips now include `auto` after
  `c475eecc0`:
  - `X1_F32_LDST_VARIANTS` covers `auto` and explicit `32x32b`;
  - validation covers one-CTA linear, one-CTA legacy, and two-CTA linear
    layouts with exact `32x32b.x1.b32` opcode expectations.
- X1 f16 ld/st roundtrips now include `auto` after `a950338f7`:
  - `X1_F16_LDST_VARIANTS` covers `auto` and explicit `32x32b`;
  - validation covers linear packed, legacy packed, and legacy unpacked
    layouts with the existing pack/unpack opcode expectations.
- Focused `ld/st` fuzz/stale-coverage cleanup now includes the remaining
  fixed-offset and rank-5 variant tails:
  - `test_tmem_runtime_matrix_ldst_fixed_offset_patterns_128x256` includes an
    `auto` companion and confirms it selects the canonical
    `32x32b.x64.b32` offset pattern;
  - two-CTA rank-5 descriptor roundtrip coverage uses the full
    `LDST_VARIANTS` set and confirms the newly added large explicit families
    hit the clean tensor-memory OOR boundary.
- Scales `ld/st` instruction-selection coverage now includes `auto`:
  - auto does not simply mirror explicit `32x32b`; for several shapes it
    selects wider `16x64b`, `16x128b`, or `16x256b` packet streams with the
    expected high-half `1048576` offsets;
  - `test_tmem_runtime_matrix_ldst_scales_variant_sweep` pins those exact
    PTX/LLIR streams across the existing scales variant matrix.
- Combined current-head `ld/st` runtime-matrix validation is green after the
  staged auto expansions, allocator-lifetime anchors, and subword coverage
  expansions:
  - command:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-broad-after-subword-refresh PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k ldst`;
  - result:
    `1181 passed, 441 skipped, 1027 deselected in 1340.05s (0:22:20)`;
  - skips are expected tensor-memory OOR / clean-boundary cases in lifted
    descriptor roundtrip and rank-5 families.
- Current-head `tcgen05.cp` runtime-matrix validation is green:
  - command:
    `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-cp-scales-verifier-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k cp`;
  - result:
    `162 passed, 5 skipped, 2488 deselected in 40.99s`;
  - this covers the current no-scales `warpx2` positives with exact commit
    opcodes, including the two-CTA `warpx2::01_23` path, the new two-CTA
    `warpx2::02_13` canonical-shared clean unsupported boundary, dense-shared
    `warpx2` clean negatives, dense copy positives, 2-CTA `128x128b` /
    `128x256b` copy, scaled `warpx4` copy paths, exact scaled-copy commit
    opcodes, and clean unsupported copy boundaries; a follow-up direct-seed
    probe found that aligned `tmemDwordDelta=0` moves data for two-CTA
    `02_13` but duplicates each source-column pair, `tmemDwordDelta=4` is the
    known all-zero path, unaligned deltas trap, and the known seed fields do not
    recover the missing source-column bit; it also covers the historical scales
    `warpx2` candidate and four nearby basis-order variants as clean unsupported
    verifier-side `warpx4.32x128b` descriptor-plan failures.
- Current-head `tcgen05.ld.red` runtime-matrix validation is green:
  - command:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldred-wait-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k ld_red`;
  - result:
    `487 passed, 2183 deselected in 482.92s (0:08:02)`;
  - this covers identity, tile-permuted `128x{64,128,256}` including the
    minimal `N=64, tile_n=16` case, pure column-permuted
    `128x{64,128,256}`, pure row-permuted `128x{64,128,256}`, row/column
    cross-product permutations at `128x{64,128,256}`, all legal modifier pairs,
    reduction-friendly row-256 positives, clean unsupported identity row-256
    layouts, exact offset immediates (`[0]` for `N <= 128` and
    `[0, 64, 128, 192]` for `N=256`), exact `tcgen05.wait::{st,ld}`
    emission and ordering around store/reduction-load consumption, and clean
    unsupported mixed-layout negatives.
- Current-head direct `tcgen05.mma` / `mma_scaled` runtime-matrix validation is
  green:
  - command:
    four-GPU `pytest-split` groups over
    `python/test/gluon/test_tmem_runtime_matrix.py -k 'mma and not cp'`;
  - result:
    aggregate selected coverage `242 passed, 50 skipped`;
  - this covers canonical, indexed, subview, tile-permuted, 1-CTA and 2-CTA
    direct MMA surfaces plus direct scaled-MMA view cases, including root-aligned
    and offset one-CTA accumulator subview format-matrix cases, the two-CTA
    cga-aware accumulator-subview matrix, tile-permuted plain-kind accumulators
    at `128x128/tile_n=32` and `128x256/tile_n=64`, the all-plain-kind
    TMEM-LHS subview matrix for both legacy/canonical accumulator layouts, the
    scaled TMEM-LHS subview format matrix, and two-CTA plain-kind accumulators at
    `256x128` and `256x256`; it also covers both 1-CTA and 2-CTA `use_acc=True`
    plain-kind accumulator addition for all supported plain kinds and both
    legacy/canonical accumulator layouts, plus TMA-fed two-CTA TF32 when matrix
    B is loaded through a non-transposed `[N, K]` TMA
    descriptor and passed to MMAv5 as a shared-memory `permute((1, 0))` view.
    Scaled-MMA copy-helper coverage remains tracked separately.
- Tile-permuted scaled-MMAv5 accumulator-subview clean-negative coverage now
  spans the same format pairs as the positive direct subview matrix:
  - `mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`,
    `mxfp4/mxfp8`, and `nvfp4/nvfp4`;
  - each case confirms the repeated `N=32` tile-permuted layout fails with the
    dedicated `matrix-B scale fragments at 64-column alignment` diagnostic and
    does not fall through to a PassManager/assertion crash.
- The recorded 2-CTA TF32 TMA-fed shared-transpose issue is now pinned as a
  clean verifier negative instead of a late LLVM pass failure:
  - `test_tmem_runtime_matrix_mma_twocta_tma_tf32_reports_clean_shared_transpose_error`
    covers both legacy and canonical TMEM-linear accumulators;
  - legal non-transposed TMA descriptors still cannot feed this TF32 MMAv5
    shape because TMA descriptors cannot be transposed and MMAv5 rejects the
    resulting transposed shared-memory operand;
  - the verifier now reports
    `tcgen05.mma does not support transposed float32 operands in shared memory`
    without `PassManager::run failed` or assertions;
  - direct non-TMA 2-CTA TF32 remains covered by the green direct-MMA slice, so
    this remains a TMA-to-shared layout materialization/API frontier rather
    than a direct accumulator-layout regression.
- The supported M64 subview/physical-bitcast slice is now checkpointed:
  - `47a07a37d` added normalized source-query inversion for physical bitcast
    views whose source subview keeps inactive zero support bases;
  - `85d8dbbf4` preserved out-of-range physical row/col origins for narrowed
    subviews, migrated the M64 `_reinterpret` tests to supported
    `slice/subview -> bitcast` where the mapping is equivalent, and made
    parameterized kernels pass layouts as explicit constexpr arguments so
    Gluon cache keys distinguish legacy and linear variants.
- The wider grouped `python/test/gluon` group-3 rerun that started before
  these two checkpoints is now superseded by the green group-3 rerun above. It
  had reduced the stale current-head red set to:
  - `test_tmem_subslice_block_m_64_parent_layout[linear]`;
  - `test_block_m_64_mma[legacy]`;
  - `test_block_m_64_mma[linear]`;
  - result: `3 failed, 4406 passed, 2041 skipped, 19348 deselected`.
- Focused current-head reruns after `85d8dbbf4` close that exact M64 set:
  - subview/physical-bitcast exacts:
    `4 passed`;
  - `test_block_m_64_mma[legacy]` / `[linear]`:
    `1 passed, 1 xfailed`;
  - the xfail is deliberate design debt for legacy M64 `64x64` layout sugar
    lacking producer-visible physical-family semantics for MMAv5 consumers;
    the linear supported layout passes in the same process. This is now an
    explicit follow-up fix on the docket, not just a status note, but it should
    stay behind current CI-lane freshness and aggregate examples validation.
- The supported descriptor bitcast API remains on the branch, and the attention
  scratch-alias path is migrated across the SDPA-safe benchmark-shaped unit
  matrix:
  - `python/examples/gluon/01-attention-forward.py` no longer uses
    `_reinterpret` for the scratch-borrow helpers;
  - P scratch slicing is dtype-aware: fp16/bf16 use 64 f32 columns for the P
    image, while fp8 uses 32, and alpha/epilogue scratch is placed after the
    actual P physical region;
  - exp2 partitions still slice the original f32 scratch subregion before
    bitcasting each exact physical image;
  - causal `use_tmem_red` now computes the diagonal-stage row max after causal
    masking instead of using a pre-mask `ld.red` max;
  - the SDPA-backed attention matrix is green:
    `64 passed in 102.66s (0:01:42)`;
  - full `python/examples/gluon` still needs a fresh aggregate rerun before
    marking the broader examples lane current.
- The stale `test_mma_shared_inputs` two-CTA TTGIR spelling bucket is closed:
  - current TMEM encoding text may spell the type field as `twoCTAs`, while
    physical-layout and op attributes still use `two_ctas`;
  - the assertion now accepts either spelling;
  - patched four-way `test_mma_shared_inputs` refresh:
    `3830 passed, 490 skipped`, `2954 passed, 1366 skipped`,
    `1206 passed, 3114 skipped`, and `2584 passed, 1736 skipped`.
- The focused persistent `python/triton_kernels/tests/test_matmul.py`
  shared-memory OOR bucket is fixed, and the full directory now has a green
  current-head rerun:
  - root cause was ordinary replayed `ttng.tmem_subslice` loads/stores using
    the exact preserved physical query and therefore selecting a root-width
    register family for a shape-local replay slice;
  - replayed subslices now use the subview-local descriptor type for register
    layout selection;
  - only explicit `tmem_physical_bitcast` roots use the preserved physical
    query, keeping the supported physical bitcast contract intact.
- Validation for the latest code checkpoint:
  - build: `PASSED`
  - lit `tmem_layouts.mlir` and `tritongpu_to_llvm_blackwell.mlir`:
    `2 passed`
  - representative persistent matmul OOR repro:
    `1 passed`, metadata back to `shared=214120`
  - focused persistent fp8/mxfp4 matmul slice:
    `16 passed, 6 skipped`
  - full 4-way `python/triton_kernels/tests` refresh:
    `2377 passed, 3444 skipped`
- The M64 row/col-permuted split-N direct ld/st bucket remains closed as a
  positive hardware surface:
  - exact row-permuted `16x32bx2` warp anchors that would lower to misaligned
    one-row PTX addresses are rejected;
  - simple row/col-permuted `64xN` f32 TMEM-linear roots select the canonical
    aligned M64 split-N register layout;
  - backend default-layout selection has the same fallback, so conversions do
    not hit the empty-compatible-layout assertion.
- The full `python/test/gluon/test_tmem_runtime_matrix.py` file was refreshed
  from this branch:
  - full-file rerun before the final assertion refresh:
    `1437 passed, 354 skipped, 1 failed in 1226.69s`;
  - the only failure was a stale TTGIR text assertion in
    `test_tmem_runtime_matrix_cp_no_scales_indexed_view_canonicalized[128]`;
  - that test now checks the stable `ttng.tmem_physical_layout` marker on the
    legacy `tensor_memory_encoding` view path, and the exact nodeid rerun
    passes.
- The old `413`-nodeid branch-added runtime-matrix manifest is now refreshed
  to empty:
  - `gb200_current_branch_test_tmem_runtime_matrix_focus_e70a3aa09_failures.txt`
- The old `332`-nodeid focused `test_core.py` TMEM/MMA manifest is also now
  refreshed to empty:
  - the full focused manifest rerun passed:
    - `332 passed in 125.78s`
  - the only intermediate issues were stale expectations for:
    - the row-encoded `1048576` PTX offset in the legacy `64x128` split-N
      roundtrip;
    - the current clean unsupported descriptor-view diagnostic for a block
      basis `ld.red` negative.
  - `gb200_current_branch_test_core_group3_focus_e70a3aa09_failures.txt`
- The latest recorded full GB200 `make NUM_PROCS=24 test-unit` lane was
  recovered at an older checkpoint and must be rerun at current HEAD before a
  full fresh CI stamp:
  - checkpoint `78196b4e4` reran the full wrapper green;
  - main `python/test/unit`: `15153 passed, 5492 skipped`;
  - `python/test/unit/test_debug.py`: `95 passed`;
  - unit-tail sublanes were green: `python/triton_kernels/tests`,
    `python/tutorials/06-fused-attention.py`, instrumentation, and plugin tests;
  - current unit manifests are empty:
    `gb200_current_branch_test_unit_failures.txt`,
    `gb200_current_branch_test_unit_matmul_refresh_failures.txt`,
    `gb200_current_branch_test_unit_tensor_descriptor_refresh_failures.txt`,
    `gb200_current_branch_test_unit_warp_specialization_refresh_failures.txt`,
    and `gb200_current_branch_test_unit_rowanchor_refresh_failures.txt`;
  - this unit green point predates later focused compiler/test coverage commits,
    so do not treat it as a current-HEAD full-wrapper rerun without refreshing.
- The supported `_reinterpret` migration invariant remains:
  - offset to the right part of TMEM;
  - slice/subview it to the desired physical bits;
  - bitcast to the desired dtype/shape/layout only when equal-size and
    physical-mapping equivalent to the input descriptor.
- Next required durable step: continue the remaining recovery queue with the
  intentionally reverted attention example tracked separately from supported
  bitcast API validation. Keep the legacy M64 MMAv5 xfail as an explicit
  follow-up fix item, then continue broader MMAv5/`mma_scaled`, staged `ld/st`
  fuzzing, stale-negative cleanup, and heuristic phases, with any remaining copy
  work limited to shapes that fit the TMEM allocation budget.

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

## 2026-04-13 08:50 UTC: ld/st descriptor reinterpret test usage is reduced

- Runtime-matrix descriptor-chain coverage is now closer to the supported TMEM
  view API contract:
  - `tmem_ldst_descriptor_chain_kernel` uses `.bitcast(...)` instead of
    `_reinterpret(...)` for the physical-equivalent descriptor view it actually
    needs;
  - the deep descriptor roundtrip chain no longer performs a no-op
    `_reinterpret` to the same dtype, shape, and layout after the view
    roundtrip;
  - the roundtrip parametrization was renamed from `slice_index_reinterpret` to
    `slice_index_deep_roundtrip`, and the required view ops are now asserted in
    TTGIR for both single-CTA and two-CTA roundtrip tests.
- Important boundary:
  - a genuine `.bitcast(...)` after the composed
    `reshape/permute/permute/reshape` view chain still fails type inference with
    `unsupported tensor memory memdesc_subslice view`;
  - that is a future API/planner gap only for real physical-equivalent bitcasts
    over complex descriptor-view chains, not needed for this no-op same-layout
    test case.
- Validation:
  - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
    passed;
  - rebuild passed;
  - `git diff --check` passed before docs update;
  - base descriptor-composition split groups passed, with group 3 requiring a
    warmed-cache rerun after a cold-compile timeout (`8 passed`, `8 passed`,
    `8 passed`, `6 passed` across the four groups);
  - the deep roundtrip selectors completed without failures but all selected
    cases took the existing OOR skip path (`30` single-CTA skips and `30`
    two-CTA skips).
- Velocity note:
  - an initial combined `-k` selector was overbroad because
    `ldst_descriptor_compositions` also matched permuted/rowcol/exotic sweeps;
    use exact nodeids for the base function when validating this slice.

## 2026-04-13 09:05 UTC: core descriptor-chain matrix uses supported bitcast

- Migrated `python/test/gluon/test_core.py::tmem_descriptor_chain_matrix_kernel`
  from `_reinterpret(...)` to `.bitcast(...)` for the same descriptor-chain
  physical-equivalent view pattern already migrated in the runtime matrix.
- `test_tmem_descriptor_chain_matrix` now asserts `tmem_physical_bitcast` in
  TTGIR in addition to the existing descriptor-view operations and exact
  `tcgen05.{st,ld}` opcode checks.
- Validation:
  - `python3 -m py_compile python/test/gluon/test_core.py` passed;
  - rebuild was a no-op success;
  - exact nodeid across four GPU split groups passed all `26` selected cases
    (`7`, `7`, `7`, `5`).

## 2026-04-13 09:35 UTC: runtime-view identity slices use supported bitcast

- Extended the `test_core.py` runtime-view API migration beyond the
  `index_reshape_*` family: the two identity `slice_*` cases now use supported
  `.bitcast(...)` through `tmem_linear_runtime_view_kernel_a`.
- The enabling planner/query change is deliberately narrow:
  - `verifyTMemSubsliceProjection` verifies against the normalized analysis
    layout when the destination layout is non-injective only because of
    inactive zero support bases; this preserves broadcast/equivalence semantics
    instead of rejecting an otherwise physical-equivalent view;
  - the standalone same-rank TMEM subview query path now handles same-rank TMEM
    subslices beyond the previous rank-2 special case.
- Superseded by the 2026-04-13 10:15 UTC checkpoint: the mixed-basis case was
  reclassified as not physical-mapping equivalent to the requested identity
  layout, moved out of the runtime-positive matrix, and converted to a clean
  supported-bitcast negative.
- Validation:
  - `python3 -m py_compile python/test/gluon/test_core.py` passed;
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8` passed;
  - `python/test/gluon/test_core.py::test_tmem_linear_runtime_views` passed all
    `11` selected cases across four GPU split groups (`3`, `3`, `3`, `2`);
  - adjacent `test_tmem_descriptor_chain_matrix` passed all `26` selected cases
    after rerunning cold-timeout groups 3 and 4 against warmed caches (`7`,
    `7`, `7`, `5`).

## 2026-04-13 09:50 UTC: runtime-matrix block descriptor negative uses supported bitcast

- Removed the last `_reinterpret(...)` use from
  `python/test/gluon/test_tmem_runtime_matrix.py`: the block-descriptor
  clean-negative kernel now spells its view change as supported `.bitcast(...)`.
- The one-CTA block case still reports the expected CTA mismatch cleanly. The
  two-CTA block case now cleanly rejects earlier at supported bitcast/subslice
  formation with `unsupported tensor memory memdesc_subslice view`, which better
  matches the supported API contract than using `_reinterpret(...)` to force the
  descriptor into a deeper direct ld/st unsupported-row-anchor diagnostic.
- Probe note: the blocked two-CTA bitcast would require a negative additive
  block-origin delta (`dim=2 step=2 phys=block delta=-1`) in the current query
  model, so it remains a real complex-view boundary rather than a direct ld/st
  negative.
- Validation: `py_compile`, `make -j8`, and `git diff --check` passed. The
  exact block-descriptor nodeid passed both selected cases across the active
  four-GPU split groups (`1`, `1`; groups 3 and 4 were empty because only two
  cases are collected).

## 2026-04-13 10:00 UTC: frontend TMEM reinterpret users classified

- Probed the three TMEM `_reinterpret(...)` helpers in
  `python/test/gluon/test_frontend.py` with supported `.bitcast(...)`. The
  probe was reverted: the focused parser tests fail before IR generation with
  `unsupported tensor memory memdesc_subslice view` on the same composed
  slice/reshape/transpose pattern.
- Classification: keep those frontend uses as intentional `memdesc_reinterpret`
  parser/IR contract coverage until the complex physical-bitcast-over-view-chain
  planner/API gap is fixed. They should not be treated as production-style
  reliance on private `_reinterpret` lowering.
- Remaining `_reinterpret(...)` users after this classification are
  intentional raw frontend `memdesc_reinterpret` parser/IR coverage,
  shared-memory reinterpret tests in `test_core.py` and `test_frontend.py`, and
  the persistence tutorial scratch-buffer borrow path, which borrows shared
  memory B buffers for the epilogue rather than tensor memory.

## 2026-04-13 10:15 UTC: test_core mixed runtime view is a clean bitcast negative

- Removed the last TMEM `_reinterpret(...)` use from
  `python/test/gluon/test_core.py`. The mixed-basis runtime-view case is no
  longer a runtime positive through legacy `_reinterpret(...)`; it is now a
  supported `.bitcast(...)` clean-negative case named
  `slice_bitcast_64_mixed_not_physical_equivalent_32x32b`.
- Classification: this was not a valid physical-equivalent bitcast to the
  requested identity `128x64` layout. The legacy `_reinterpret` lowering planned
  from the raw result layout and ignored the selected source descriptor view, so
  store/load through the same descriptor could pass without proving that the
  physical TMEM image was preserved.
- Validation: `py_compile`, no-op rebuild, and `git diff --check` passed. The
  combined positive/negative runtime-view nodeids passed all `11` selected cases
  across four GPU split groups (`3`, `3`, `3`, `2`).


## 2026-04-13 13:35 UTC: explicit `ld.red` variants cover modifiers

- Extended `tmem_ld_red_explicit_layout_kernel` to forward `abs` and
  `propagate_nan` into explicit-layout `tmem.load_min` / `tmem.load_max`.
- The identity and compatible non-identity explicit-variant tests now cover all
  `LD_RED_MODIFIER_CASES` for both `min` and `max` while keeping the same four
  explicit load variants: `auto`, `32x32b`, `16x32bx2`, and `32x32b_splitn`.
- NaN-propagating cases inject NaNs into representative rows and validate both
  the full tensor output and the reduced result against PyTorch with
  `equal_nan=True`.
- The bounded discovery conclusion is unchanged but stronger: these explicit
  variants still canonicalize to `tcgen05.ld.red.sync.aligned.32x32b.x128`
  rather than revealing another accepted reduction atom, including when the
  `.abs` and `.NaN` opcode modifiers are present.
- Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`,
  no-op `make`, and four-GPU split pytest for
  `ld_red_explicit_compatible or ld_red_explicit_n_sharded` passed `163` selected
  cases aggregate (`41`, `41`, `41`, `40` by group).

## Latest: 2026-04-13 scales `warpx2` direct-source-offset scan

- The direct-PTX scales-copy probe now supports arbitrary single-message opcode,
  source-offset, and TMEM-destination-delta variants with one child process per
  run.
- Updated 2026-04-14 evidence: the canonical `warpx4` high-half message uses
  source offset 32 to copy logical columns `8..15`, but the same source window
  faults for both `warpx2::01_23` and `warpx2::02_13`. The tracked unprimed
  single-message scan now covers every `sourceOffsetB128` value `0..136` at
  `dst_delta=4`, with `274` records and zero matches. Offsets `0..8` execute
  but copy row-shifted low physical columns / otherwise wrong data; offsets
  `9..136` fault. Compact result:
  `experiments/results/probe_cp_scales_warpx2_offsets_dst4_current_summary.json`.
- Current conclusion: do not implement scales `warpx2` by swapping the canonical
  `warpx4` messages. Support needs a different shared descriptor/view/staging
  model, or this remains a clean unsupported boundary.

## Latest: 2026-04-13 ldst duration-cache refresh

- The `ldst` duration cache now reflects the lifted roundtrip pre-skip change: the `440` known pre-execution skips have near-zero scheduling weight instead of stale compile-until-OOR durations.
- Full retained validation: `ldst` runner passed `1201` and skipped `441`, with group times `208.7s..263.8s`.
- Runtime `variant_id` selector kernels were tried and reverted. They are functionally possible with `do_not_specialize`, but they make the real xdist matrix slower by compiling larger branchy kernels per layout.

## Latest: 2026-04-14 selected-shard speed helper and CP 128x128b dtype coverage

- Runner speed workflow: `run_tmem_runtime_matrix_sweep.py` now supports `--groups` for exact pytest-split shard reruns and `--xdist-override BUCKET=N` for controlled local experiments. Selected groups keep the canonical group-to-GPU mapping, so group 5 still uses GPU 0, group 8 still uses GPU 3, etc.; this preserves the stable per-GPU cache behavior that dominates warm rerun speed.
- Measured speed result: selected warm `ldst` groups 5 and 8 passed through the runner in about `11s` wall time each (`77 passed, 27 skipped in 9.57s`; `74 passed, 28 skipped in 9.39s`), versus the documented full-cold duration-cache `ldst` group range of about `208s..264s`. This is a targeted rerun speedup and does not reduce coverage.
- `pytest-xdist -n 8` was tested for warm `ldst` shards and was slower than the retained `-n 4` default, so default inner parallelism is unchanged.
- First post-speed ISA coverage slice: `tcgen05.cp.cta_group::* .128x128b` runtime-matrix coverage now includes both `torch.float32` and `torch.int32` for legacy/linear single-CTA, linear indexed-view, and two-CTA paths while retaining exact PTX/LLIR opcode checks.
- Current validation: `make -j8` no-op success; py-compile and `git diff --check` passed; focused CP slice passed all `17` selected cases across four GPUs; full `cp` runner bucket passed `313` and skipped `5` across four GPUs.


## Latest: 2026-04-14 ld.red descriptor-chain N-width explicit variants

- Runtime-matrix collection is now `6698` tests with bucket totals `cp=381`, `mma=1693`, splitn/misc `=499`, `ld_red=1256`, and `ldst=2869`; current bucketed evidence aggregates to `6252 passed, 446 skipped`.
- Latest coverage slice adds descriptor-chain `ld.red` explicit variants for identity and `rowcol_rotate_reverse` layouts at `N in {64,256}` over `32x32b`, `16x32bx2`, and `32x32b_splitn`, crossed with `min`/`max` and all legal modifier modes.
- Validation for the latest slice: py-compile, `make -j8`, `git diff --check`, focused collect `96/6698`, `ld_red` collect `1256/6698`, full-file collect `6698`, and four-GPU split execution of the focused selector passed all `96` cases.
- The mixed layout at `N=256` uses a legal explicit split offset order `[0, 128, 64, 192]` for `16x32bx2` and `32x32b_splitn`; this is recorded in the test assertion instead of assuming canonical offset order for every layout.

## Latest: 2026-04-14 ld.red descriptor-chain N-width all-layout explicit variants

- Runtime-matrix collection is now `6842` tests with bucket totals `cp=381`, `mma=1693`, splitn/misc `=499`, `ld_red=1400`, and `ldst=2869`; current bucketed evidence aggregates to `6396 passed, 446 skipped`.
- Latest coverage slice completes descriptor-chain `ld.red` explicit variants at `N in {64,256}` for identity, tile-permuted, pure column reverse, pure row reverse, and mixed row/column reverse layouts over `32x32b`, `16x32bx2`, and `32x32b_splitn`, crossed with `min`/`max` and all legal modifier modes.
- Validation for the latest slice: `make -j8`, py-compile, `git diff --check`, focused collect `240/6842`, `ld_red` collect `1400/6842`, full-file collect `6842`, and final four-GPU split execution of the focused selector passed all `240` cases.
- At `N=256`, both `col_reverse` and `rowcol_rotate_reverse` use legal explicit split offset order `[0, 128, 64, 192]` for `16x32bx2` and `32x32b_splitn`; identity, tile-permuted, and row-reverse keep the canonical order.

## Latest: 2026-04-14 scaled-MMAv5 tile-permuted accumulator use-acc coverage

- Runtime-matrix collection is now `6852` tests with bucket totals `cp=381`, `mma=1703`, splitn/misc `=499`, `ld_red=1400`, and `ldst=2869`; current bucketed evidence aggregates to `6406 passed, 446 skipped`.
- Latest coverage slice adds nonzero-accumulator coverage for the scaled-MMAv5 64-column tile-permuted accumulator positive matrix across all scaled format pairs and `K in {128,256}`.
- Validation for the latest slice: `make -j8`, py-compile, `git diff --check`, focused collect `10/6852`, tight MMA collect `1703/6852`, full-file collect `6852`, and four-GPU split execution of the focused selector passed all `10` cases.

## Latest: 2026-04-14 two-CTA scaled-MMAv5 accumulator-subslice use-acc coverage

- Runtime-matrix collection is now `6892` tests with bucket totals `cp=381`, `mma=1743`, splitn/misc `=499`, `ld_red=1400`, and `ldst=2869`; current bucketed evidence aggregates to `6446 passed, 446 skipped`.
- Latest coverage slice adds nonzero accumulator semantics for the supported two-CTA scaled-MMAv5 accumulator-subslice matrix across all scaled format pairs, `slice_start in {0,128}`, `block_k in {128,256}`, and multicast on/off.
- The test helper now has an `ACC_INIT` path for accumulator subviews; zero-init callers preserve old behavior, and the parked two-CTA `block_n=64` scale-descriptor frontier remains unchanged.
- Validation for the latest slice: `make -j8`, py-compile, `git diff --check`, focused collect `40/6892`, tight MMA collect `1743/6892`, full-file collect `6892`, and four-GPU split execution of the focused selector passed all `40` cases.

## Document Roles

- `memory.md`
  - durable memory for the initiative
  - source of truth for mission, current decisions, long-term plan, and current
    undated priorities
  - future sessions should update this when the stable plan or current
    priorities change

- `log.md`
  - append-only chronology of checkpoints, bug classification, validation, and
    conclusions
  - do not rewrite old entries except for obvious factual corrections

- `handoff_2026-04-09.md`
  - current active handoff
  - read its tail first for the latest live buckets, disproved experiments, and
    next implementation order
  - keep it updated at meaningful checkpoints and before likely context
    rollover

- `gb200_nvidia_ci_inventory.md`
  - current GB200/NVIDIA CI baseline for this Blackwell devbox phase
  - records the workflow coverage, confirmed red list, and classification of
    each current failure as stale expectation, rewrite-candidate, or real bug
  - update it before broadening beyond exact nodeids or changing the current
    CI-grounded failure story

- `gb200_failure_manifest.md`
  - index for the generated exact current-branch failure lists and the
    merge-base-existing shard-3 subsets
  - use the adjacent `.txt` files when you need exact nodeids rather than
    prose summaries
  - refresh the manifests whenever the current-branch failure surface changes
    materially

- `gb200_branch_recovery_plan.md`
  - prioritized recovery backlog for restoring GB200 CI parity with main
  - separates real branch regressions from preexisting failures, stale
    negatives, and reinterpret-contract rewrite candidates
  - use it to choose the next fix slice once the GB200 census is current

- `fuzz_plan.md`
  - operational saturation/fuzzing playbook for the broader `tcgen05` surface
  - not the immediate source of truth for current bug-fix priority while the
    shared-planner / legacy-unification / reinterpret-decomposition work is
    still active

- `handoff_2026-03-26.md`
  - historical handoff from an earlier phase of the initiative
  - archival provenance, not current status

- `next_agent_prompt_2026-03-26.md`
  - archived takeover prompt from an earlier managed-session workflow
  - archival only

## Current Session Discipline

- Keep `memory.md`, the active handoff, and this README aligned.
- Treat `memory.md` as memory: if the stable understanding of the project or
  the current undated execution priorities changes, update `memory.md` rather
  than leaving that knowledge only in chat context.
- Keep dated historical analyses in `log.md` and the dated handoff files
  instead of rewriting old reports in place.

## Latest Checkpoint

- 2026-04-17 08:35 UTC: explicit non-M64 f32 non-scales `ld.red` layouts now
  have a software reduction fallback. The Gluon TMEM API asks the backend
  whether the selected register layout is directly `tcgen05.ld.red`-supported;
  unsupported direct-reduction layouts use a normal TMEM load plus
  layout-aware `ttgl.reduce(axis=1)`, preserving `abs` and NaN propagation.
  Hardware-supported layouts still emit `tcgen05.ld.red`. Validation:
  `make -j8`, direct `invalid.mlir` verifier, py-compile, explicit
  n-sharded software selector (`12 passed`), broader reduction selector
  (`77 passed`), core reduction selector (`43 passed`), and
  `git diff --check`.
- 2026-04-17 08:27 UTC: `ld.red` reduction-layout support now carries typed
  unsupported reason kinds. `TMemLoadReductionLayoutSupport` still preserves
  the existing diagnostic text, but future software-reduction planning can
  branch on semantic failure classes such as M-sharded registers, unsupported
  N thread bases, missing lane splits, partial-N registers, and non-contiguous
  N bases. Validation: `make -j8`, direct `invalid.mlir` verifier,
  py-compile, focused reduction selector (`45 passed`), and
  `git diff --check`.
- 2026-04-16 03:34 UTC: dense no-scales copy tile-permutation coverage now
  includes non-adjacent pure tile-selector permutations (`reverse` and
  `even_odd`) where low instruction columns remain contiguous. This pins the
  current planner as instruction-tile general rather than adjacent-swap-only:
  some layouts use `128x128b`, while selector permutations that preserve the
  8-column footprint stay on `128x256b`. Validation: `make -j8`,
  py-compile, and the combined tile-permutation selector (`22 passed`).
- 2026-04-16 03:18 UTC: dense row-permutation reprobes confirm the remaining
  source-row copy frontier is not another destination column offset bug.
  `rotate1` and high-row-tile-only row permutations still need a
  destination-row mask, row-partitioned atom, or smaller footprint schedule.
  The row/column copy clean-negative assertion no longer accepts the stale
  `128-byte descriptor macro-tile` wording.
- 2026-04-16 03:05 UTC: refined dense copy destination scheduling after broad
  `cp_` validation caught a regression in M256 folded-row copies. Physical
  tile offsets are now used only when pure column tile selectors are permuted
  at or above the selected copy instruction width; folded high-row selectors
  stored in the column address space keep the logical destination schedule.
  Validation: `make -j8`, representative M256 copy rows, tile-permutation
  selector (`18` rows), and the split-4 `cp_` runtime sweep
  (`174 passed / 10 skipped`, `184 passed`, `184 passed`, `183 passed`).
- 2026-04-16 02:46 UTC: no-scales dense copy tile-permuted `tile_n=4` rows
  were promoted via exact physical tile-coordinate support for pure column
  tile permutations. This was refined at 03:05 UTC to preserve folded-row M256
  logical scheduling.
- 2026-04-16 01:55 UTC: descriptor-row split requirements now feed a
  schedule-proof failure. For scales descriptor-view/subslice copy, the
  planner proves that separate descriptor-row messages would overwrite the
  complementary destination columns because public `tcgen05.copy` writes the
  full instruction footprint. This keeps the remaining row as an ISA
  mask/narrow-atom boundary, not a descriptor-search gap.
- 2026-04-16 01:52 UTC: the descriptor-row split requirement derivation is
  now a shared TMEM utility API, and copy query debug prints the derived
  selected destination-column run and selection period. This keeps the next
  scheduler slice from reparsing diagnostic fields.
- 2026-04-16 01:49 UTC: descriptor-row-stride copy failures now derive an
  explicit `TMemCopyDescriptorRowSplitRequirement`. The scales
  descriptor-view row still remains clean unsupported, but the planner now
  records the exact split/mask requirement: source column bit 2 selects
  descriptor row `+32` for 4-column destination runs every 8 columns inside a
  16-column `warpx4` instruction, spanning the full 32-row source footprint.
  Validation: `make -j8`, direct invalid verifier, focused scales copy
  selector (`9` rows), py-compile, descriptor-view debug repro, and
  `git diff --check`.
- 2026-04-15 21:45 UTC: single-CTA canonical block-row
  `TensorMemoryLinearLayout` forms are now folded into equivalent row bases
  before TMEM `ld/st` and `ld.red` planning. This promotes the former
  `block_single_cta` clean negatives to positive direct and descriptor
  `ld/st` coverage plus M128 `ld.red` reductions for `N in {64,128,256}`.
  Validation: `make -j8`, py-compile for the touched Python files,
  `git diff --check`, invalid verifier, collect-only for the new selectors,
  and the focused block-row runtime selector passed `74` cases.
- 2026-04-15 20:39 UTC: pure row-order no-scales copy permutation failures now
  report through the `instruction schedule` layer instead of the physical
  query layer. Mixed row/column bases and column-contiguity failures remain
  separate physical-query/atomization boundaries. Validation: `make -j8`,
  invalid verifier, `git diff --check`, and the
  `cp_no_scales_linear_rowcol_permuted_reports_clean_unsupported` sweep passed
  `15` cases.
- 2026-04-15 20:36 UTC: copy scheduled tiles now carry explicit
  `logicalRow` and `sourceRow` coordinates. Current schedules still emit zero
  row deltas, so behavior is unchanged, but lowering now consumes the selected
  schedule's source row instead of assuming all copy tiles start at shared row
  zero. Validation: `make -j8`, invalid verifier, `git diff --check`, and the
  focused copy selector
  `cp_no_scales_warpx2 or cp_scales_tmem_descriptor_view_reports_clean_unsupported or cp_scales_warpx4`
  passed `433` cases.
- 2026-04-15 20:18 UTC: source-format suffix probes on the already-supported
  int8 scales `warpx4` copy compiled but produced wrong output for both
  `.b8x16.b6x16_p32` and `.b8x16.b4x16_p64`. Source-format suffixes are
  therefore not a drop-in descriptor-view copy fix; continue with row
  partition/sub-instruction scheduling.
- 2026-04-17 13:06 UTC: no-scales multicast `tcgen05.copy` destinations now
  have a backend physical-layout proof. `warpx2`/`warpx4` plans no longer
  accept the destination layout merely because source conversion classified a
  copy family; the planner now checks expected multicast zero row bases,
  ascending non-broadcast row bases, contiguous low instruction columns, and
  canonical two-CTA block ownership. This keeps row-permuted multicast
  destinations and subword destinations with too few TMEM column bases as clean
  physical-query negatives. Validation: `make -j8`; exact row-permuted
  `warpx2` negative (`1 passed`); split-4 `cp_no_scales_warpx2`
  (`20/20/20/19` passed); Python compile; `git diff --check`.
- 2026-04-17 18:16 UTC: plain MMAv5 exotic and row/column-permuted
  accumulator negatives now report through a structured instruction-tile order
  requirement. The backend note records logical/CTA shape, element bitwidth,
  the public `64x8`-or-larger tile requirement, and the first noncanonical
  in-tile basis. Behavior is unchanged: whole-tile accumulator permutations
  remain positive, while in-tile permutations still need a real split or
  masked writeback schedule. Validation: `make -j8`, invalid verifier,
  split-4 focused MMAv5 negative selector `5/5/5/2`, Python compile, and
  `git diff --check`.
- 2026-04-17 18:23 UTC: scaled-MMAv5 mixed fp4A TMEM-LHS negatives now report
  through a structured padded operand-A storage requirement. A temporary
  guard-lift probe emitted `mxf8f6f4` but produced wrong output
  (`16376/16384` mismatches), proving raw tensor-memory packed columns do not
  model the shared-memory `fp4_padded` contract. Validation: `make -j8`,
  invalid verifier, split-4 focused mixed-fp4A selector `6/6/6/6`, Python
  compile, and `git diff --check`.
- 2026-04-17 18:27 UTC: scaled-MMAv5 narrow-N accumulator negatives now report
  through a structured scale-fragment requirement with logical/CTA shape, plain
  instruction shape, N-fragment count, and B-scale padding/rematerialization
  factor. Behavior is unchanged: `N=8/16` support still needs both accumulator
  permutation and B-scale fragment rematerialization. Validation: `make -j8`,
  invalid verifier, split-4 focused narrow-N selector `5/5/5/5`, Python
  compile, and `git diff --check`.
- 2026-04-17 21:12 UTC: direct `ld/st` leading-slice replay preservation now
  uses backend utility `shouldPreserveDirectTMemLdStLeadingSliceView` instead
  of a transform-local gapped-column query check. Behavior is unchanged:
  gapped-column leading-slice views still replay to avoid direct physical
  support aliasing logical halves. Validation: `make -j8` and split-4 focused
  replay/direct-support selector `6/6`, `6/6`, `6/6`, `4/4`.
- 2026-04-17 21:17 UTC: MMAv5 family address-layout selection now uses
  backend API `getMMAv5TMemFamilyAddressLayout` instead of a local LLVM
  lowering lambda. Behavior is unchanged for accumulator, scaled-accumulator,
  and TMEM-LHS address paths. Validation: `make -j8` and split-4 focused
  MMAv5 tile-permutation selector `26/26` on each group.
- 2026-04-15 20:16 UTC: copy source-format legality is now an explicit
  planner check via `getTMemCopySourceFormatSupport(...)`. Current schedules
  still use `None`, so behavior is unchanged; the next source-format support
  attempt now has a dedicated ISA-legality layer.
- 2026-04-15 20:14 UTC: copy support failures now have an explicit
  `instruction schedule` layer. Source-row projection failures,
  instruction-column projection failures, known cta-group::2 schedule gaps,
  and invalid emitted-instruction streams no longer masquerade as descriptor
  synthesis failures. Diagnostics stayed text-compatible.
- 2026-04-15 20:11 UTC: bounded frontier probes left no source changes but
  sharpened next-step selection. The scales descriptor-view copy still needs a
  sub-instruction source split or smaller valid atom than the current 16-column
  `warpx4` message. Temporarily bypassing the 4x256b refresh ld/st frontend
  guard let the C++ layout generator answer directly; all candidates rejected
  with `compInput cannot compose`, so that path remains a row-anchor/register
  layout contract gap.
- 2026-04-15 20:07 UTC: instruction schedule construction is now factored
  behind `getTMemCopyInstructionSchedule(...)`. The helper still emits the
  current Cartesian tile/message stream, but future split schedules can now be
  implemented in one planner utility instead of touching lowering.
- 2026-04-15 19:55 UTC: copy executable plans now carry an explicit
  `TMemCopyScheduledInstruction { messageIndex, tile }` stream. Lowering
  consumes that selected stream instead of nesting tiles and messages itself;
  behavior is preserved, but future non-Cartesian source/message split
  schedules now have one planner-owned insertion point.
- 2026-04-15 18:59 UTC: copy source-row projection checks now live in the
  shared planner and lowering uses the same helper, replacing lowering-only
  row projection assertions with clean diagnostics.
- 2026-04-15 18:55 UTC: copy lowering now consumes
  `getTMemCopyDestinationTilePlan(...)`, making destination tile offsets an
  explicit planner artifact for future non-uniform schedules.
- 2026-04-15 18:43 UTC: copy instruction-column projection is now a planner
  preflight before descriptor enumeration. This keeps descriptor search from
  masking sub-instruction source-column permutations that the selected copy
  atom cannot consume.
- 2026-04-15 18:39 UTC: two-CTA no-scales
  `tcgen05.copy.warpx2::02_13.64x128b` is now rejected by the shared copy
  planner as a known cta-group::2 descriptor/address schedule gap, rather than
  by a verifier-only note after descriptor enumeration. See `memory.md`,
  `log.md`, and `handoff_2026-04-09.md` for the exact validation and next
  planner targets.
