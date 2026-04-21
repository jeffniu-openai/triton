# Round 35 Local: Structural Descriptor/`ld.red` Fuzzer Slice

Date: 2026-04-21 14:12 UTC
Branch: `codex/tmem`
Scope: discovery-only guardrail run. No backend or checked-in test source was
modified.

## Objective

Rerun the checked-in structural fuzzer slice that directly targets descriptor
views, `ld/st` view roundtrips, and `ld.red` cases. This is lower-level than
the broad runtime matrix selectors and includes expected failures for known
open buckets.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'descriptor_view or ldst_view or ldred or ld_red'
```

Result: `18/33` tests collected.

## Runtime

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'descriptor_view or ldst_view or ldred or ld_red'
```

Result:

```text
7 passed, 15 deselected, 11 xfailed in 6.25s
```

## Classification

No unexpected failure, XPASS, compiler crash, false unsupported diagnostic,
opcode mismatch, runtime miscompile, or new independent `FZ-*` bucket was
found. The selected expected failures remain tracked by existing structural
fuzzer buckets.
