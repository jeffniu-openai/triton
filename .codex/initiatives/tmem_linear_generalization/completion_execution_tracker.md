# TMEM Completion Execution Tracker

Last updated: 2026-04-17 21:08 UTC

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

- Phase A, rebaseline and classify: in progress.
- Phase B, complete shared physical-query model: partially complete; continue
  deleting type-only/frontend fallback policy as each family moves to backend
  support objects.
- Phase C, finish `tcgen05.copy` atomized planner: active support frontier.
- Phase D, finish `ld/st` and `ld.red` packet/replay planning: partially
  complete; remaining work is packet-footprint/rematerialization and true
  atom-footprint boundaries.
- Phase E, finish MMAv5 and scaled-MMAv5 descriptor/storage semantics:
  partially complete; remaining work is mainly scaled storage fragments and
  true MMAv5 tile-boundary proofs.
- Phase F, cleanup/redesign deletion: active between support slices.
- Phase G, saturation/performance/final validation: active for the current
  M64/runtime-matrix checkpoint; continue staged broad validation after each
  new support-bearing slice.

## Current Clean-Negative Inventory

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
- Scaled-MMAv5 mixed fp4A TMEM-LHS: now reported through a typed padded
  operand-A storage requirement. A guard-lift probe emitted `mxf8f6f4` but
  produced wrong output, so support still needs real fp4_padded TMEM storage
  semantics rather than a verifier lift.
- Scaled-MMAv5 narrow accumulator `N=8/16`: now reported through a structured
  narrow-N scale-fragment requirement that records the plain accumulator
  instruction shape, fragment count, and B-scale padding/rematerialization
  factor. Support still needs a real scale-fragment and accumulator
  permutation schedule.

## Current Runtime Matrix Validation

Checkpointed at 2026-04-17 20:41 UTC after the M64 physical-subview fix:

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

## Immediate Execution Order

1. Return to the support-bearing frontier: start with the highest-value
   remaining Phase C/Phase E bucket whose probes suggest a real implementation
   path rather than a true ISA/storage/API boundary.
2. Classify each clean-negative bucket in code comments/tests/docs as stale,
   missing planner schedule, missing storage representation, or true boundary.
3. Take the next support-bearing slice from `tcgen05.copy` because it has the
   largest remaining inventory and most directly reflects linear-layout
   incompleteness.
4. Between support slices, delete redundant frontend/lowering compatibility
   policy that is now represented by backend query/support objects.
5. After each meaningful slice, update this file plus `memory.md`, `log.md`,
   and `handoff_2026-04-09.md`; commit with a detailed message and push to
   `origin/codex/tmem`.

## Remaining Work Plan

Status legend: `done`, `active`, `pending`, `blocked`, `boundary`.

### Phase A: Rebaseline And Classify

- `done`: collect the current clean-negative surface after a green build and
  record the bucket inventory.
- `active`: for each bucket, classify it as stale guard, missing planner
  schedule, missing storage representation, missing ISA-family coverage, or
  true hardware/API boundary.
- `active`: keep compile/runtime-duration expectations attached to validation
  commands so broad matrix work remains practical on four GPUs.

### Phase B: Shared Physical Query And Requirement Model

- `active`: move remaining frontend/lowering-only TMEM decisions into backend
  physical-query helpers or typed requirement structs.
- `active`: make diagnostics consume structured requirement data rather than
  ad hoc strings, especially for clean negatives that are likely to remain
  hardware boundaries.
- `pending`: remove compatibility shims once copy, load/store, reduction, MMA,
  and scaled-MMA all consume the shared query/support APIs directly.

### Phase C: `tcgen05.copy` Completion

- `active`: finish direct no-scales dense/multicast planner boundaries:
  row-order, mixed-basis, sub-instruction column masks, and destination/source
  ownership.
- `active`: decide and encode the correct `4x256b` contract: refresh-image
  positive support, ordinary-view remapping, direct readback limitations, and
  clean typed diagnostics for nonmaterializable refresh images.
- `pending`: complete scales copy support or prove boundaries for
  descriptor-row split/mask schedules, source-message formats, and
  destination-column partitions.
- `pending`: complete or prove the no-scales two-CTA `warpx2::02_13`
  schedule, including preservation of the high source-column bit.
- `pending`: handle dense/noncanonical `warpx2` shared-source layouts and
  subword/packed-lane copies through source rematerialization or typed
  non-support proofs.
- `pending`: delete copy-specific rescue stacks that duplicate the shared
  planner once their behavior is represented by query/support objects.

### Phase D: Direct `ld/st` And `ld.red`

- `active`: keep packet-footprint limitations represented as structured
  atom-footprint requirements rather than layout-name failures.
- `active`: continue scales direct `ld/st` descriptor-view support where the
  ISA can realize the requested atom. The two-CTA `16x32bx2` row is now
  positive; remaining clean negatives should be only true packet-footprint
  boundaries or future missing schedules proved by exact layout arithmetic.
- `pending`: decide whether `4x256b` refresh images can be read back through a
  rematerialized public load/store view; otherwise keep a precise row-anchor
  diagnostic.
- `pending`: finish `ld.red` non-f32/NaN semantics either with a correct
  software fallback or a typed semantic-boundary diagnostic.

### Phase E: Plain And Scaled MMAv5

- `done`: keep plain MMAv5 accumulator layout failures tied to a structured
  instruction-tile row/column ordering requirement while preserving existing
  whole-tile permutation positives.
- `done`: prove and structure the scaled-MMAv5 mixed fp4A TMEM-LHS storage
  representation boundary.
- `done`: prove and structure the scaled-MMAv5 narrow-N B-scale fragment
  schedule and accumulator permutation boundary.
- `pending`: expand opcode/runtime positives only when a new backend schedule
  is real, not when a frontend spelling happens to compile.

### Phase F: Cleanup And Redesign Deletion

- `active`: delete stale frontend guards and lowering-local type-only fallback
  policy as backend helpers subsume them.
- `pending`: consolidate repeated layout arithmetic into shared helpers with
  exact `LinearLayout` compose/invert/pseudoinvert proofs.
- `pending`: keep only durable abstractions; quarantine or remove temporary
  row/column rewrite smells after each family is represented in the planner.

### Phase G: Saturation, Performance, And Final Validation

- `pending`: run staged validation: `make`, targeted lit/compiler checks,
  focused four-GPU runtime selectors, then duration-aware broad sweeps.
- `pending`: profile any representative compile above the 3-4 second target or
  execution above the 1-2 second target before accepting the regression.
- `pending`: refresh GB200/NVIDIA manifests after broad validation and mark
  stale pre-fix counts explicitly.
- `pending`: final state requires no unexplained clean negatives, no stale
  family-specific policy, practical split-4 runtime-matrix iteration, and an
  updated handoff summarizing every residual true ISA boundary.

## Next Concrete Slice

The copy-planner boundary audit is complete as of 2026-04-17 20:52 UTC. The
remaining representative copy negatives are clean and correspond to real
schedule/storage/API boundaries: `warpx2::02_13` two-CTA source-column
preservation, subword packed lanes, row/column masks or smaller atom
footprints, ordinary-view `4x256b` refresh-remap/readback, scales
descriptor-view masks, and noncanonical CTA ownership.

The corrected full runtime-matrix runner passed at 2026-04-17 20:57 UTC.
Direct `ld/st` replay/support shim cleanup now has backend helpers for support
type selection, replayable half-slice detection, and leading-slice
direct-vs-replay preservation. Return to the support-bearing Phase C/Phase E
frontier unless another clearly duplicated frontend/lowering policy is found.

## Progress

- 2026-04-17 21:12 UTC: moved the leading-slice direct-vs-replay preservation
  policy into backend utility `shouldPreserveDirectTMemLdStLeadingSliceView`.
  `OptimizeTMemLayouts` now consumes the helper instead of owning the
  gapped-column query check locally. Validation: `make -j8`; split-4 focused
  replay/direct-support runtime selector passed `6/6`, `6/6`, `6/6`, and
  `4/4`.

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
