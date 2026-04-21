# Round 25 Local: Descriptor Load/Store Non-Roundtrip Baseline

Date: 2026-04-21 14:24 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

This local slice covered checked-in descriptor load/store rows that actually
execute in the current environment. The first attempted selector,
`ldst_descriptor_roundtrip and not reports`, collected `51/1615` rows but all
selected rows were pre-skipped because the lifted descriptor roundtrip
matrices exceed the current Blackwell TMEM allocation limit. That run is not
counted as runtime evidence.

The executing selector was:

```text
ldst_descriptor and not reports and not roundtrip
```

It covers descriptor compositions, layout permutations, row/column
permutations, exotic layouts, multidimensional slices, higher-rank dim0
slices, multidimensional replay/positive rows, higher-rank half-row positives,
and direct half-row positives.

## Commands

Required build gate had already completed in this Round 25 session:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst_descriptor and not reports and not roundtrip'
```

Result:

```text
53/1615 tests collected
```

Split-4 runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <1..4> \
  --store-durations \
  --durations-path /tmp/tmem_r25_ldst_descriptor_nonroundtrip_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst_descriptor and not reports and not roundtrip'
```

## Results

Aggregate result: `53 passed`.

Per-shard results:

- GPU 0 / group 1: `14 passed, 1601 deselected`.
- GPU 1 / group 2: `14 passed, 1601 deselected`.
- GPU 2 / group 3: `14 passed, 1601 deselected`.
- GPU 3 / group 4: `11 passed, 1604 deselected`.

Durations:

```text
/tmp/tmem_r25_ldst_descriptor_nonroundtrip_durations.json
```

Skipped contrast:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s -rs --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_roundtrip_sweeps[slice_index_roundtrip-0-5.0-required_ops0-identity-128-auto-32x32b.x128.b32]'
```

Result:

```text
SKIPPED: lifted descriptor roundtrip matrices exceed the current Blackwell TMEM allocation limit
```

## Classification

No runtime miscompile, compiler crash, unexpected unsupported diagnostic, or
new independent `FZ-*` bucket was found in the executing non-roundtrip
descriptor load/store slice.

The skipped roundtrip rows remain excluded from runtime evidence for this
slice because the test file intentionally pre-skips those matrices at the
current allocation limit.
