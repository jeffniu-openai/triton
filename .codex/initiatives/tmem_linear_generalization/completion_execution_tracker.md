# TMEM Completion Execution Tracker

Last updated: 2026-04-17 17:36 UTC

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
- Phase G, saturation/performance/final validation: pending after major
  support frontiers close.

## Current Clean-Negative Inventory

Collected at 2026-04-17 17:36 UTC after `make -j8`:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q --collect-only python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'reports_clean_unsupported'
```

Result: `124/1592 tests collected (1468 deselected) in 3.02s`.

Current buckets:
- `ld/st` scales descriptor-view and variant atom-footprint boundaries:
  explicit two-CTA `16x32bx2` half-tile semantics and too-narrow n-sharded
  scale atoms.
- `ld.red` non-f32 NaN-propagating cases: software fallback exists for many
  non-f32 reductions, but these rows remain true semantic boundaries unless a
  correct fallback can preserve the requested NaN contract.
- `tcgen05.copy` scales shared-subslice/descriptor-view rows: source column bit
  2 selects descriptor row `+32` inside a `warpx4` instruction, requiring a
  source-message/destination-column split, narrower atom, valid source format,
  or destination mask.
- `tcgen05.copy` no-scales ordinary contiguous `4x256b`: copy support is
  positive for refresh-shaped layouts only; ordinary view exposure needs a
  first-class refresh remap/readback contract or stays negative.
- Direct `ld/st` of `4x256b` refresh images: row anchors are not materializable
  as public load/store warp bases without a row-anchor rematerialization model.
- No-scales two-CTA `warpx2::02_13`: current public `cta_group::2`
  direct-seed schedules either duplicate low source columns or read zeros; a
  valid schedule must preserve the high source-column bit.
- `warpx2` dense/noncanonical shared-source layouts and subword copies:
  descriptor representability is not sufficient; support needs source
  rematerialization, packed-lane storage, and descriptor semantic-equivalence
  proofs.
- Copy row/column permutation and sub-instruction tile permutation rows:
  require row/column partitioning, smaller footprints, masks, or explicit
  proof that the full-footprint public atom cannot realize the projection.
- Plain MMAv5 exotic/row-column-permuted accumulators: public atoms require
  canonical row/column basis order within an instruction tile unless a
  tile-splitting or masked writeback schedule is designed.
- Scaled-MMAv5 mixed fp4A TMEM-LHS: needs padded operand-A storage semantics
  matching shared memory before the guard can lift.
- Scaled-MMAv5 narrow accumulator `N=8/16`: guard-lift probes compile but
  produce wrong output; support needs a real scale-fragment and accumulator
  permutation schedule.

## Immediate Execution Order

1. Classify each clean-negative bucket in code comments/tests/docs as stale,
   missing planner schedule, missing storage representation, or true boundary.
2. Take the next support-bearing slice from `tcgen05.copy` because it has the
   largest remaining inventory and most directly reflects linear-layout
   incompleteness.
3. Between support slices, delete redundant frontend/lowering compatibility
   policy that is now represented by backend query/support objects.
4. After each meaningful slice, update this file plus `memory.md`, `log.md`,
   and `handoff_2026-04-09.md`; commit with a detailed message and push to
   `origin/codex/tmem`.

## Next Concrete Slice

Investigate `tcgen05.copy` no-scales row/column permutation and `warpx2`
schedule boundaries from the shared planner side. The first implementation
target is not to widen descriptor enumeration; it is to determine whether the
existing scheduled-instruction carrier can express a legal non-overwriting
row/source projection. If not, promote the proof into a clearer typed
requirement and move to the next reachable support slice.
