# Round 47 Non-M64 LD.Red Positive Guardrail

Date: 2026-04-21 14:41 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Run a focused positive runtime guardrail over non-M64 TMEM reduction rows. This
lane intentionally excluded M64, clean diagnostics, resource boundaries, and
descriptor-chain rows so it could isolate direct positive `tcgen05.ld.red`
lowering while parallel agents covered copy, proxy/mbarrier, and structural IR
surfaces.

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
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short \
  -k '(ld_red or load_red) and not m64 and not reports and not resource and not clean and not descriptor_chain' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result:

```text
156 passed, 1459 deselected
```

## Classification

No compiler crash, false unsupported diagnostic, runtime miscompile, opcode
absence, or new independent `FZ-*` bucket was observed. This lane did not
exercise the known M64 destination-layout gap (`FZ-20260421-0012`) or
descriptor-chain `ld.red` opcode-loss bucket (`FZ-20260421-0004`).

Backend repair remains deferred per the discovery-only campaign.
