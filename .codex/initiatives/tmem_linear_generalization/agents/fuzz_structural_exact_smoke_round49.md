# Round 49 Structural Exact Smoke

Date: 2026-04-21 14:49 UTC

Branch: `codex/tmem`

Scope: discovery/cataloging only. No backend/compiler code or checked-in tests
were modified.

## Objective

Run a compact exact structural-fuzzer smoke mixing known positives and expected
xfails. This catches XPASS drift, unexpected failure-signature drift, and
structural fuzzer harness regressions without running the full file again.

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

## Structural Command

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_dynamic_index_load_only[generic-pass-dynamic-index-load-only-128x32]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-2cta]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-chain1-64x32-min]'
```

Result:

```text
2 passed, 2 xfailed
```

The dynamic memdesc xfail emitted the expected `FZ-20260421-0001` illegal
`ttg.memdesc_index` signature at `ConvertTritonGPUToLLVM`; the `ld.red`
descriptor-chain xfail remained expected as `FZ-20260421-0004`.

## Classification

No XPASS drift, unexpected compiler crash, false unsupported diagnostic,
positive-row failure, or new independent `FZ-*` bucket was observed.

Backend repair remains deferred per the discovery-only campaign.
