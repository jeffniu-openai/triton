# Round 27 Lane: Checked-In Structural Fuzzer Rerun

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler fixes.
- Repo edit scope: this report only.

## Summary

The checked-in TMEM structural fuzzer was rerun across four GPUs after the
Round 26 verifier probes. The result was stable against the current expected
catalog: no unexpected failures and no XPASS.

Aggregate:

```text
9 passed
24 xfailed
0 failed
0 xpassed
```

The xfail output continued to include the known dynamic `ttg.memdesc_index`
LLVM-conversion repros for `FZ-20260421-0001`. No new independent bucket was
found in this rerun.

## Commands

All commands used stable per-GPU caches.

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  2>&1 | tee /tmp/tmem_round27_structural_g1.log

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  2>&1 | tee /tmp/tmem_round27_structural_g2.log

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  2>&1 | tee /tmp/tmem_round27_structural_g3.log

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  2>&1 | tee /tmp/tmem_round27_structural_g4.log
```

## Results

```text
group 1: 5 passed, 24 deselected, 4 xfailed
group 2: 2 passed, 24 deselected, 7 xfailed
group 3: 2 passed, 24 deselected, 7 xfailed
group 4: 27 deselected, 6 xfailed
```

Representative known xfail diagnostic:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

Classification: existing `FZ-20260421-0001`.

## Notes

`pytest-split` reported no duration file for this run and used even splitting.
The file is small enough that this did not affect the result quality.

No backend/compiler code was changed.
