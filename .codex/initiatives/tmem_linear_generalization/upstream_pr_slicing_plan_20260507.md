# TMEM Upstream PR Slicing Plan

Date: 2026-05-07

## Goal

Turn the current `codex/tmem` worktree into reviewable upstream changes without
shipping the full branch history or coupling unrelated validation artifacts,
experiments, and local initiative notes to the first code PRs.

The branch has two kinds of content:

- core implementation work that belongs upstream in dependency order; and
- branch-local coordination, fuzz-campaign evidence, experiments, and
  performance notes that should stay out of the first upstream series unless a
  reviewer asks for them.

## Recommended Stacked PR Order

1. **TMEM linear-layout and descriptor-view foundations**
   - Scope:
     - linear-layout parsing / arithmetic needed by TMEM views;
     - TMEM memdesc/view inference;
     - explicit physical-bitcast contract;
     - runtime `taddr` update model for origin-changing views.
   - Why first:
     - every later lowering slice depends on the same descriptor semantics.
   - Validation target:
     - focused lit for view inference / invalid diagnostics;
     - focused Gluon frontend physical-bitcast tests.

2. **TMEM ld/st lowering on current descriptor facts**
   - Scope:
     - type-local ld/st query selection;
     - row-plan and scalar-packet selection;
     - aligned and subword descriptor-view handling;
     - software-reduction fallback boundary where relevant.
   - Why second:
     - this is the widest consumer surface and proves the new memdesc model
       outside MMA-only paths.
   - Validation target:
     - focused ld/st runtime matrix;
     - affected conversion lit;
     - representative unit matmul/dot exacts.

3. **`tcgen05.copy` exact-layout planning**
   - Scope:
     - type-local destination planning;
     - exact destination offsets;
     - direct-root / `warpx2` / tile-permuted families;
     - clean diagnostics for true unsupported cases.
   - Why third:
     - it depends on the same descriptor semantics but is reviewable apart from
       normal ld/st.
   - Validation target:
     - focused copy runtime matrix;
     - copy-specific structural-fuzzer sentinels.

4. **`ld.red` explicit reduction path**
   - Scope:
     - explicit Gluon red-load contract;
     - exact payload ordering;
     - alignment policy split between compiler diagnostics and iisan.
   - Why fourth:
     - it is behaviorally distinct, and the explicit API change deserves its
       own review surface.
   - Validation target:
     - `ld_red` runtime selector;
     - focused lit for diagnostics and control flow.

5. **MMAv5 / scaled-MMAv5 descriptor-family support**
   - Scope:
     - current-type family address layouts;
     - scale descriptor classification and rematerialization;
     - direct TMEM-LHS legality and clean negatives.
   - Why fifth:
     - it depends on the descriptor model and benefits from the earlier ld/st
       review settling the shared invariants.
   - Validation target:
     - plain/scaled MMAv5 runtime selectors;
     - affected allocation and lowering lit.

6. **Coverage and fuzzing promotion**
   - Scope:
     - runtime matrix rows that lock reviewed behavior;
     - reusable test helpers;
     - only the minimized structural-fuzzer rows that become durable regression
       tests.
   - Why sixth:
     - reviewers can evaluate production changes before absorbing the very large
       branch-local fuzz campaign footprint.
   - Validation target:
     - grouped Gluon runtime matrix;
     - structural-fuzzer exact positives / negatives promoted by the earlier
       PRs.

7. **Cleanup follow-up**
   - Scope:
     - low-risk helper compression;
     - stale test expectation cleanup;
     - any post-review API polish.
   - Why last:
     - it should not obscure the semantic changes in the earlier PRs.

## Hold Back From The First Upstream Series

- `.codex/initiatives/**` and local `AGENTS.md` guidance.
- Large fuzz-campaign journals and per-round exploratory reports.
- The D64 FP8 attention tuning commit (`1aebab7dc`) unless it is sent as a
  separate performance PR with benchmark context.
- The fp4 padded-shared-memory exploration note (`4a51f1596`), which is a
  report rather than product code.
- The packed shuffled MX4 layout experiment (`3d1d89f72`) until it has an
  independent product justification and validation story.

## Extraction Strategy

- Build the series from fresh branches rooted at current `upstream/main`, not by
  opening one giant PR from `codex/tmem`.
- Prefer `git cherry-pick -n` plus manual squashing by logical slice over
  replaying the checkpoint-heavy branch history verbatim.
- Keep tests in the same PR as the public behavior they validate.
- Keep branch-local notes out of upstream commits, but preserve the exact local
  validation trail in this initiative directory.
- Before cutting each PR, rerun the narrow focused selector for that slice and
  one broader smoke set that crosses the slice boundary.

## Current Risk To Resolve Before Cutting PR 1

- The deterministic focused recovery set is now green at `51534acf0`:
  - unit exact manifest `115 passed`;
  - regression exact manifest `93 passed`;
  - `triton_kernels` exact manifest `69 passed`;
  - Gluon frontend exact manifest `7 passed`;
  - all four focused lit files from the merge-base comparison passed.
- The remaining release gate is broad validation, not another known focused
  product bucket:
  - rerun the repaired head through the GB200-equivalent lane;
  - refresh aggregate counts before using the pre-fix failure inventory for any
    prioritization;
  - keep the partition-sensitive Gluon tail, `test_debug.py`, and Proton
    classified separately unless a fresh branch-only exact repro appears.
- Treat broad-lane freshness as the gate for starting PR extraction, but keep
  the stacked series structure above rather than collapsing back into one PR.
