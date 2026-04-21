# Round 53 Local Structural Guardrail

Date: 2026-04-21
Branch: `codex/tmem`
Checkpoint base: `060d6c136 Integrate Round 52 duration grouping evidence`

This local lane re-ran the checked-in structural TMEM fuzzer as a strict
guardrail while Round 53 subagents explored non-overlapping runtime and lit
surfaces. The objective was drift detection, not repair: detect unexpected
passes, changed diagnostics, or contamination between known xfail compiler
crashes/miscompiles and adjacent passing rows.

## Preflight

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Command

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python:./python/test/gluon \
pytest -q -s --tb=short -rx \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

## Result

```text
9 passed, 24 xfailed in 14.26s
```

## Xfail Inventory Observed

The strict xfails remained in the expected buckets:

- `FZ-20260421-0001`: runtime TMEM `memdesc_index` reaches LLVM conversion as
  an illegal op, including the direct load-only row.
- `FZ-20260421-0002`: dynamic/control-flow-carried descriptor views miscompile
  across helper, inline, tuple-capture, and layout-pressure variants.
- `FZ-20260421-0003`: ld/st descriptor-view chain wrong-result rows.
- `FZ-20260421-0004`: `ld.red` descriptor-view and 2CTA indexed rows emit
  plain load/software-reduce forms instead of the expected hardware `ld.red`.
- `FZ-20260421-0005`: 256-row lifted-parent allocation assertion.
- `FZ-20260421-0006`: `ld.red` transpose/slice descriptor view false
  unsupported candidate.
- `FZ-20260421-0007`: scaled-MMAv5 dynamic selected accumulator-view/subslice
  miscompile.
- `FZ-20260421-0008`: 2CTA indexed `ld.red` row/col chain optimizer abort.
- `FZ-20260421-0009`: 1CTA direct indexed `ld.red` allocator assertion.
- `R5-C`: loop-carried TMEM view auto-layout inference failure.

## Classification

No new compiler crash, verifier drift, unexpected unsupported diagnostic,
strict-xfail XPASS, runtime miscompile, or independent `FZ-*` bucket was found
in this local guardrail.

No backend fixes were attempted.
