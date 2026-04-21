# Round 32 Local: Checked-In Structural Fuzzer Rerun

Date: 2026-04-21

## Scope

Reran the checked-in structural TMEM fuzzer after the Round 32 narrow-shape,
plain-MMAv5, lit, and clean-boundary guardrails. This checks that the current
strict xfail catalog still matches current compiler/runtime behavior before
starting the next fuzzing lane.

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
9 passed, 24 xfailed in 9.42s
```

The xfail set remained stable. No XPASS, unexpected failure, new compiler
crash, false unsupported diagnostic, opcode mismatch, or runtime miscompile was
observed.

## Classification

No new `FZ-*` bucket is needed. This is a stability rerun of the checked-in
structural sentinel catalog before continuing adversarial fuzzing.
