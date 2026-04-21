# Round 14 local twoCTA/high-CGA runtime sweep

Date: 2026-04-21 11:24 UTC
Branch: `codex/tmem`
HEAD before report: `3130f54df`
Mode: discovery/cataloging only. No backend or compiler code was changed.

## Summary

This local sweep sanity-checked the checked-in runtime-matrix surface around
two-CTA TMEM layouts, high-CGA clean boundaries, and CTA ownership roundtrips
after the copy/mbarrier composition checkpoint.

No new `FZ-*` bucket was found. The selected checked-in rows stayed green:
legal two-CTA rows executed, and rows marked as clean high-CGA boundaries
remained skipped or diagnostic-only according to test expectations.

## Required rebuild

Command:

```bash
make -j8
```

Result: no work to do.

## Selector

Collect-only command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(twocta or cga_roundtrip or layout_in_4cta_context) and not reports'
```

Result: `331/1615` selected.

Runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_local_r14_twocta_highcga_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(twocta or cga_roundtrip or layout_in_4cta_context) and not reports'
```

## Results

- GPU 0 / group 1: `56 passed, 27 skipped, 1532 deselected in 4.15s`.
- GPU 1 / group 2: `73 passed, 10 skipped, 1532 deselected in 17.17s`.
- GPU 2 / group 3: `83 passed, 1532 deselected in 12.41s`.
- GPU 3 / group 4: `82 passed, 1533 deselected in 4.87s`.
- Aggregate: `294 passed, 37 skipped`.

Adjacent exact high-CGA core controls:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[False-ctas_per_cga1]' \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[True-ctas_per_cga2]' \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit_twocta_linear_acc[linear]'
```

Result: `3 passed in 4.34s`.

## Classification

- New independent `FZ-*`: none.
- Runtime miscompile: none observed.
- Compiler crash: none observed.
- False unsupported diagnostic: none observed in this checked-in selector.
- Existing boundary coverage: high-CGA/twoCTA boundary rows remain classified
  by their checked-in expectations, next to `FZ-20260421-0010` report-only
  probes for legal 1CTA/2CTA layouts rejected in larger-CGA launch contexts.

Recommended next discovery slice: continue with temporary adversarial probes
that mix high-CGA launch context with multiple TMEM families in a single kernel
and with generic control-flow selected descriptor views, because those are not
fully represented by this green checked-in selector.
