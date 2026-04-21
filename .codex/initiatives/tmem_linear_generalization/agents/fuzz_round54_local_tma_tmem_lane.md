# Round 54 Local TMA/TMEM Runtime Lane

Date: 2026-04-21
Branch: `codex/tmem`
Checkpoint base: `2912b49bb Integrate Round 53 fuzz evidence`

This local lane exercised the two-CTA TMA + TMEM + MMAv5 runtime rows in
`python/test/gluon/test_tmem_runtime_matrix.py`. The goal was to keep async
TMA/MMA descriptor interactions green while other Round 54 lanes fuzz dynamic
2CTA descriptor SSA, resource extremes, and opcode consistency.

No backend fixes were attempted.

## Preflight

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Collection

```bash
PYTHONPATH=.:./python \
pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix_mma_twocta_tma_tf32'
```

Result:

```text
15/1615 tests collected (1600 deselected) in 3.13s
```

## Runtime

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix_mma_twocta_tma_tf32'
```

Result:

```text
15 passed, 1600 deselected in 2.98s
```

## Coverage Notes

The selector covered:

- two-CTA TF32 TMA shared-transpose clean diagnostics;
- B-transposed descriptor positive variants;
- `use_acc` B-transposed descriptor positive variants;
- linear and legacy accumulator layouts;
- representative `(BLOCK_N, BLOCK_K)` pairs including narrow and large
  shapes.

## Classification

No compiler crash, verifier drift, false unsupported diagnostic, clean-boundary
drift, runtime miscompile, hang, or new independent `FZ-*` bucket was found in
this lane.
