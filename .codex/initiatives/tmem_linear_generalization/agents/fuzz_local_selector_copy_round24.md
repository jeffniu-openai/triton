# Round 24 Local: Selector-Permuted Copy Rows

Date: 2026-04-21 13:50 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

After the Round 24 structural-generator prototype identified descriptor SSA
selection as a high-value next generator axis, this local slice sanity-checked
the currently checked-in selector-permuted no-scales copy rows.

Selector:

```text
dynamic or generic or branch or selector
```

With the current runtime matrix names this collected the four
`cp_no_scales_linear_tile_selector_permuted` rows.

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'dynamic or generic or branch or selector'
```

Result:

```text
4/1615 tests collected
```

Runtime:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'dynamic or generic or branch or selector' \
  2>&1 | tee /tmp/tmem_r24_selector_rows.log
```

Result:

```text
4 passed, 1611 deselected
```

## Classification

No runtime miscompile, compiler crash, unexpected unsupported diagnostic, or
new independent `FZ-*` bucket was found.

This is only a narrow green guardrail.  It does not cover the dynamic generic
descriptor SSA failures in `FZ-20260421-0001`/`0002`; the next generator
promotion should add explicit branch/helper/loop-carried descriptor values
instead of relying on test-name selection.
