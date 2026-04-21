# Round 31 Local: Checked-In Structural Fuzzer Rerun

Date: 2026-04-21

## Scope

Reran the checked-in structural TMEM fuzzer after the Round 30 and early Round
31 catalog updates. This validates that the current strict xfail catalog still
matches current compiler/runtime behavior and that no expected failure has
silently changed class.

## Commands

Required rebuild:

```bash
make -j8
```

Runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py
```

## Result

The checked-in structural fuzzer completed as:

```text
9 passed, 24 xfailed
```

The emitted xfail diagnostics still include the known illegal
`ttg.memdesc_index` LLVM-conversion repros for `FZ-20260421-0001`. No XPASS,
unexpected failure, new compiler crash, or new runtime miscompile was observed.

## Classification

No new `FZ-*` bucket is needed. This is a stability rerun of the current
checked-in sentinel catalog.

