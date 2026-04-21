# Round 52 Local MoE Example Harness Path Check

Date: 2026-04-21 15:04 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Independently confirm the example harness path issue for
`python/examples/gluon/05-moe-bmm1-fused-gather.py`.

## Commands

Bare collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/examples/gluon/05-moe-bmm1-fused-gather.py
```

Corrected collection:

```bash
PYTHONPATH=.:./python:./python/triton_kernels pytest --collect-only -q python/examples/gluon/05-moe-bmm1-fused-gather.py
```

## Results

- Bare collection: exit code `2`, `ModuleNotFoundError: No module named 'triton_kernels.distributed'`.
- Corrected collection: exit code `0`, `48 tests collected`.

## Classification

This is a harness/import-path issue, not a TMEM backend failure. No compiler
crash, verifier issue, runtime miscompile, or new independent `FZ-*` bucket was
observed.

Backend repair remains deferred per the discovery-only campaign.
