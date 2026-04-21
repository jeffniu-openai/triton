# Round 48 TMA/Indexed/Subslice Positive Mix

Date: 2026-04-21 14:45 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Run a compact positive mix over TMA-fed MMAv5 descriptor rows and accumulator
indexed/subslice descriptor-view rows. The selector excludes clean diagnostics,
resource boundaries, and M64 rows so it can act as a green guardrail for
ordinary descriptor-view consumption by MMA paths.

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
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short \
  -k '(tma_b_transposed_descriptor or indexed_acc_view or acc_subslice_view_plain_kinds or indexed_acc_identity_narrow_view_format_use_acc or acc_subslice_view_format_use_acc) and not reports and not clean and not resource and not m64' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result:

```text
142 passed, 1473 deselected
```

## Classification

No compiler crash, false unsupported diagnostic, runtime miscompile, opcode
drift, or new independent `FZ-*` bucket was observed.

Backend repair remains deferred per the discovery-only campaign.
