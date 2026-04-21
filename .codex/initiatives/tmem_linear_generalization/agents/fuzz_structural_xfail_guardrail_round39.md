# Round 39 Structural Xfail Guardrail

Date: 2026-04-21 15:08 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend code was modified.

## Objective

Re-run the checked-in structural fuzzer rows most likely to detect drift in
existing TMEM bug sentinels: descriptor-view wrong-result rows, `ld.red`
opcode-loss/unsupported rows, dynamic descriptor generic-pass rows, and the
scaled-MMAv5 dynamic subslice sentinel. The goal was to catch unexpected XPASS,
new unexpected failure, or changed failure shape while backend repair remains
deferred.

## Commands

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'generic_pass or descriptor_view or ldred or scaled_mma'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'generic_pass or descriptor_view or ldred or scaled_mma'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'generic_pass or descriptor_view or ldred or scaled_mma'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'generic_pass or descriptor_view or ldred or scaled_mma'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'generic_pass or descriptor_view or ldred or scaled_mma'
```

## Results

- Collection: `26/33`.
- Split-4 result:
  - group 1: `3 passed, 4 xfailed`
  - group 2: `7 xfailed`
  - group 3: `7 xfailed`
  - group 4: `5 xfailed`
- Aggregate: `3 passed, 23 xfailed`.

## Classification

No new independent `FZ-*` bucket was found.

The expected xfail rows reproduced known cataloged buckets, including noisy
MLIR reproducer output from `FZ-20260421-0001` late illegal
`ttg.memdesc_index` rows. There was no XPASS, unexpected failure, changed
clean-boundary behavior, or new crash outside the expected xfail set.

Backend repair remains deferred while discovery lanes continue.
