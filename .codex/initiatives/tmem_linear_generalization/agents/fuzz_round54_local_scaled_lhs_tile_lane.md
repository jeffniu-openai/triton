# Round 54 Local Scaled-MMAv5 LHS/Tile/Narrow Lane

Date: 2026-04-21
Branch: `codex/tmem`
Checkpoint base: `e85cf74b7 Integrate Round 54 opcode consistency evidence`

This local lane exercised scaled-MMAv5 LHS subslice, LHS tile-permuted,
accumulator tile-permuted, and narrow accumulator identity/tile rows in
`python/test/gluon/test_tmem_runtime_matrix.py`. The goal was to stress
scale-layout legality, legacy-vs-linear scaled operands, accumulator layout
matching, `use_acc`, and clean unsupported diagnostics without overlapping the
active resource/shape-extreme lane.

No backend fixes were attempted.

## Collection

```bash
PYTHONPATH=.:./python \
pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (mma_scaled_lhs_subslice_view or mma_scaled_lhs_tile_permuted or mma_scaled_acc_tile_permuted_64 or mma_scaled_acc_tile_permuted_narrow or mma_scaled_acc_identity_narrow)'
```

Result:

```text
120/1615 tests collected (1495 deselected) in 3.16s
```

## Runtime Split

Command shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
PYTHONPATH=.:./python \
pytest -q -s --tb=short \
  --splits 4 --group <group> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (mma_scaled_lhs_subslice_view or mma_scaled_lhs_tile_permuted or mma_scaled_acc_tile_permuted_64 or mma_scaled_acc_tile_permuted_narrow or mma_scaled_acc_identity_narrow)'
```

Results:

```text
group 1: 30 passed, 1585 deselected in 10.67s
group 2: 30 passed, 1585 deselected in 8.45s
group 3: 30 passed, 1585 deselected in 6.60s
group 4: 30 passed, 1585 deselected in 4.54s
aggregate: 120 passed, 0 failed
```

## Coverage Notes

The selector covered:

- LHS subslice and LHS tile-permuted scaled-MMAv5 rows;
- legacy and linear scale operand variants;
- `mxfp8`, `mxfp4`, mixed `mxfp8/mxfp4`, and `nvfp4` formats;
- `use_acc` and no-`use_acc` variants;
- mixed-fp4A clean unsupported diagnostics;
- accumulator tile-permuted 64-wide rows;
- narrow accumulator tile-permuted and identity rows.

## Classification

No compiler crash, verifier drift, false unsupported diagnostic,
clean-boundary drift, runtime miscompile, hang, or new independent `FZ-*`
bucket was found in this lane.
