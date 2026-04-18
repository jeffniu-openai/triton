# Gap #2/#3 Execution Plan - 2026-04-18

Status: closed at 2026-04-18 23:03 UTC.

Branch: `codex/tmem`

## Scope

Fix the two remaining active TMEM scaled-MMAv5 coverage gaps:

- Gap #2: narrow scaled-MMAv5 accumulator layouts that require logical `N=8`
  or `N=16` fragments, especially tile-permuted accumulator views.
- Gap #3: mixed fp4A TMEM-LHS scaled-MMAv5 where operand A is fp4 and operand B
  is not fp4 under `kind::mxf8f6f4`.

## Constraints

- Preserve the existing positive runtime matrix.
- Keep clean negatives only for cases that remain impossible after a real
  support attempt.
- Do not remove coverage to make the matrix green.
- Use exact linear-layout/address arithmetic and typed requirements; avoid
  ad-hoc format or layout-specific bypasses.
- Make checkpoint commits and push `HEAD` to `origin/codex/tmem` after each
  meaningful support slice.

## Phase 0: Reproduce And Trace

- Done. The initial Gap #3 support attempt compiled one former clean-negative
  case but produced wrong numerics; it was removed before checkpointing.
- Done. Gap #2 direct narrow `N=8/16` scaled instructions are viable when
  B-scale storage is rematerialized to the 64-column fragment granularity.

## Phase 1: Gap #2 Support Slice

- Done. Scaled accumulator planning now admits `N=8/16` family layouts.
- Done. `RematerializeScaledMmaBScaleFragments` pads/rematerializes matrix-B
  scale storage for narrow fragments, reusing the existing allocation-pass
  model for scale-fragment storage rewrites.
- Done. The former narrow clean-negative matrix is now positive runtime
  coverage across all five scaled format pairs, both narrow tile sizes, and
  `K=128/256`.

## Phase 2: Gap #3 Support Slice

- Closed as a typed direct-TMEM unsupported boundary. A padding-only direct
  TMEM representation was proven wrong by runtime output. The missing
  semantics are the shared-memory `fp4_padded` row-dependent 128-byte swizzle
  and padding aliases, not a layout-family planner omission.
- Future support belongs to a new explicit fp4-padded TMEM storage/API or
  staged shared-memory materialization path. The current backend keeps clean
  negatives for raw mixed fp4A TMEM LHS and does not generate incorrect PTX.

## Phase 3: Validation And Docs

- Done. Validation:
  - `make -j8`;
  - focused first slice: `3 passed`;
  - representative narrow-family slice: `4 passed`;
  - full narrow-N matrix: `20 passed`;
  - full mixed-fp4A TMEM-LHS clean-negative matrix: `24 passed`;
  - broader non-twoCTA scaled-accumulator tile-permuted selector:
    `52 passed, 1542 deselected`;
  - clean-negative/error collect-only rebaseline:
    `141/1594` collected.
- Final pre-commit verification at 2026-04-18 23:03 UTC:
  - incremental `make -j8`: no work;
  - full narrow-N matrix: `20 passed`;
  - full mixed-fp4A TMEM-LHS clean-negative matrix: `24 passed`.
- Done. Initiative docs refreshed with the final Gap #2 and Gap #3
  classifications.
