# Round 14 Local Structural Fuzzer Smoke Gate

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler fixes were attempted.

## Required rebuild

```bash
make -j8
```

Result: `ninja: no work to do`.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result:

```text
9 passed, 24 xfailed in 10.00s
```

## Classification

No new bucket. The checked-in structural-fuzzer sentinels stayed stable after
the latest Round 14 report checkpoints. The noisy diagnostics are expected
strict xfails, primarily `FZ-20260421-0001` dynamic `ttg.memdesc_index`
illegal lowering and `FZ-20260421-0002` chain0 generic wrong-result rows.
