# Round 36 Tile-Permuted And Structural Fuzzer Guardrail

Date: 2026-04-21 13:45 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Re-run the checked-in structural fuzzer and cover a broad tile-permuted runtime
slice across TMEM `ld.red`, `tcgen05.copy`, plain MMAv5, and scaled MMAv5. The
goal was to check whether any existing xfail bucket unexpectedly changed and
whether tile-permuted layouts exposed new compiler crashes, verifier
over-strictness, opcode drift, false unsupported diagnostics, or runtime
miscompiles.

## Commands

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_structural_fuzzer.py

PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'tile_permuted and not reports and not resource and not m64'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'tile_permuted and not reports and not resource and not m64'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'tile_permuted and not reports and not resource and not m64'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'tile_permuted and not reports and not resource and not m64'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'tile_permuted and not reports and not resource and not m64'
```

## Results

Structural fuzzer:

- group 1: `5 passed, 4 xfailed`
- group 2: `2 passed, 7 xfailed`
- group 3: `2 passed, 7 xfailed`
- group 4: `6 xfailed`
- aggregate: `9 passed, 24 xfailed`

Tile-permuted runtime matrix:

- collection: `151/1615`
- group 1: `38 passed`
- group 2: `38 passed`
- group 3: `38 passed`
- group 4: `37 passed`
- aggregate: `151 passed`

## Classification

No new independent `FZ-*` bucket was found.

The structural fuzzer remained stable. The expected xfail rows still reproduce
cataloged failures such as `FZ-20260421-0001` late illegal
`ttg.memdesc_index`; their MLIR reproducer output is expected noise for this
lane, not a new failure.

The tile-permuted runtime matrix passed across `ld.red`, copy, plain MMAv5, and
scaled MMAv5 rows. There was no compiler crash, false unsupported diagnostic,
opcode absence, runtime miscompile, xfail drift, or clean-boundary regression.

Backend repair remains deferred while new discovery lanes continue.
