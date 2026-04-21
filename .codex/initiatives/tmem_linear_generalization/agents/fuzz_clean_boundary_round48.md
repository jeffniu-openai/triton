# Round 48 Clean Boundary Runtime Guardrail

Date: 2026-04-21 14:44 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Run a broad checked-in runtime guardrail over clean hardware/API boundaries
while excluding M64 rows. The goal was to confirm that expected limitations
continue to report clean diagnostics or pass their explicit boundary assertions,
rather than drifting into crashes, generic pass failures, or runtime
miscompares.

## Required Build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Runtime Command

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short \
  -k '(reports_clean or clean_unsupported or clean_error or tmem_oor or resource) and not m64' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result:

```text
176 passed, 1439 deselected
```

## Classification

No diagnostic drift, assertion crash, generic `PassManager::run failed`,
runtime miscompile, unexpected failure, or new independent `FZ-*` bucket was
observed. The excluded M64 destination-layout failures remain tracked
separately as `FZ-20260421-0012`.

Backend repair remains deferred per the discovery-only campaign.
