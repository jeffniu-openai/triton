# Round 46 Subword and X1 Runtime Guardrail

Date: 2026-04-21 14:38 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Stress subword, `.x1`, non-f32, and descriptor-chain roundtrip coverage in one
runtime selector. This lane intentionally avoided scaled-MMAv5 and multi-CTA
ownership surfaces that were delegated to parallel Round 46 subagents.

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
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short \
  -k '(x1 or subword or i64 or f64 or non_f32 or descriptor_chain_roundtrip) and not reports and not resource and not clean' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result:

```text
301 passed, 70 skipped, 1244 deselected
```

## Classification

No compiler crash, false unsupported diagnostic, runtime miscompile, unexpected
skip/pass drift, or new independent `FZ-*` bucket was observed. The existing
non-f32/subword boundaries remained isolated to their explicit diagnostic
coverage and did not leak into this positive guardrail.

Backend repair remains deferred per the discovery-only campaign.
