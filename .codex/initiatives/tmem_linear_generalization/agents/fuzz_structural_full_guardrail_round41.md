# Round 41 Structural Fuzzer Full Guardrail

Date: 2026-04-21 14:12 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Command

Required rebuild:

```bash
make -j8
```

Result: build-tree ninja reported `no work to do`.

Full checked-in structural fuzzer:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

## Result

```text
9 passed, 24 xfailed in 8.20s
```

The visible xfail output included the expected `FZ-20260421-0001` late illegal
`ttg.memdesc_index` reproducer diagnostics for dynamic descriptor-index rows.

## Classification

No XPASS, unexpected failure, compiler crash outside expected xfail rows,
runtime miscompile, or new independent `FZ-*` bucket was observed. The
checked-in structural sentinel catalog remains stable before the next Round 41
temporary-probe lanes complete.
