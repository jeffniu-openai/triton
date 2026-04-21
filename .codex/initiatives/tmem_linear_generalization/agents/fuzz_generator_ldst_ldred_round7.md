# Round 7 Lane B: generator-backed ld/st and ld.red structural fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.

## Scope

Used the Round 6 descriptor generator prototype in `/tmp` as the row selector
for compact `ld/st` and `ld.red` probes that were not exact checked-in
structural-fuzzer ids. The probe focused on:

- read-only descriptor-view loads versus same-view roundtrip masking;
- direct/indexed `ld.red` opcode selection;
- dtype/subword `ld/st` rows;
- clean unsupported descriptor-view boundaries.

The temporary runnable probe is:

`/tmp/tmem_generator_ldst_ldred_round7_probe.py`

The generated inventory refresh is:

`/tmp/tmem_structural_case_generator_round7.json`

The minimized compiler-crash MLIR is:

`/tmp/tmem_generator_ldst_ldred_round7_ldred256x32_alloc_assert.mlir`

## Commands

Required rebuild before tests:

```bash
CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8
```

Result: `ninja: no work to do`.

Generator refresh:

```bash
python /tmp/tmem_structural_case_generator_round6.py \
  --repo /root/code/triton \
  --json-out /tmp/tmem_structural_case_generator_round7.json \
  --limit 48
```

Result:

- `25245` generated descriptors;
- `31` checked-in structural-fuzzer case ids;
- `0` exact normalized-id matches.

Probe syntax and collection:

```bash
PYTHONPATH=.:./python:python/test/gluon python -m py_compile /tmp/tmem_generator_ldst_ldred_round7_probe.py
PYTHONPATH=.:./python:python/test/gluon pytest --collect-only -q /tmp/tmem_generator_ldst_ldred_round7_probe.py
```

Result: `8 tests collected`.

Four-GPU split classification:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r7b-gpu0 PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short --splits 4 --group 1 /tmp/tmem_generator_ldst_ldred_round7_probe.py 2>&1 | tee /tmp/tmem_generator_ldst_ldred_round7_g1.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r7b-gpu1 PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short --splits 4 --group 2 /tmp/tmem_generator_ldst_ldred_round7_probe.py 2>&1 | tee /tmp/tmem_generator_ldst_ldred_round7_g2.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r7b-gpu2 PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short --splits 4 --group 3 /tmp/tmem_generator_ldst_ldred_round7_probe.py 2>&1 | tee /tmp/tmem_generator_ldst_ldred_round7_g3.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r7b-gpu3 PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short --splits 4 --group 4 /tmp/tmem_generator_ldst_ldred_round7_probe.py 2>&1 | tee /tmp/tmem_generator_ldst_ldred_round7_g4.log
```

The split run intentionally let unexpected outcomes fail so the logs preserve
the exact compiler/runtime assertion context.

Exact fresh confirmations:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r7b-confirm-ldred256x32 PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short '/tmp/tmem_generator_ldst_ldred_round7_probe.py::test_generator_ldred_round7[ldred-f32-1cta-ncta1-256x32-index-identity-identity-32x32b-min]' 2>&1 | tee /tmp/tmem_generator_ldst_ldred_round7_confirm_ldred256x32.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r7b-confirm-f16-read PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short '/tmp/tmem_generator_ldst_ldred_round7_probe.py::test_generator_ldst_round7[ldst-f16-1cta-ncta1-64x32-reshape_permute-identity-identity-16x64b-readonly]' 2>&1 | tee /tmp/tmem_generator_ldst_ldred_round7_confirm_f16_readonly.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r7b-confirm-ldred128x64 PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short '/tmp/tmem_generator_ldst_ldred_round7_probe.py::test_generator_ldred_round7[ldred-f32-1cta-ncta1-128x64-index-even_odd-identity-32x32b-min]' 2>&1 | tee /tmp/tmem_generator_ldst_ldred_round7_confirm_ldred128x64.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r7b-confirm-2cta256x64 PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short '/tmp/tmem_generator_ldst_ldred_round7_probe.py::test_generator_ldred_round7[ldred-f32-2cta-ncta2-256x64-index-identity-identity-32x32b-min]' 2>&1 | tee /tmp/tmem_generator_ldst_ldred_round7_confirm_2cta256x64.log
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r7b-confirm-f16-roundtrip PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short '/tmp/tmem_generator_ldst_ldred_round7_probe.py::test_generator_ldst_round7[ldst-f16-1cta-ncta1-64x32-reshape_permute-identity-identity-16x64b-roundtrip]' 2>&1 | tee /tmp/tmem_generator_ldst_ldred_round7_confirm_f16_roundtrip.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r7b-confirm-i32-c3 PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short '/tmp/tmem_generator_ldst_ldred_round7_probe.py::test_generator_ldst_round7[ldst-i32-1cta-ncta1-64x32-double_transpose_slice-identity-identity-32x32b-readonly]' 2>&1 | tee /tmp/tmem_generator_ldst_ldred_round7_confirm_i32_c3.log
```

MLIR replay for the new crash:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
./bin/triton-opt /tmp/tmem_generator_ldst_ldred_round7_ldred256x32_alloc_assert.mlir --run-reproducer > /tmp/tmem_generator_ldst_ldred_round7_ldred256x32_triton_opt.log 2>&1
```

Result: exit `134`, aborting in `TritonTensorMemoryAllocationPass`.

## Classification Matrix

| Case | Generator descriptor | Observed | Classification |
| --- | --- | --- | --- |
| `ldst-f16-1cta-ncta1-64x32-reshape_permute-identity-identity-16x64b-readonly` | `ldst`, f16, `[64,32]`, `reshape_permute`, `16x64b` | Runtime mismatch `1024/2048`, greatest absolute difference `128.0` | Already-covered `FZ-20260421-0003` subword read-only miscompile; checked-in strict xfail exists for this exact semantic row under legacy id `ldst-fz20260421-0003-f16-chain2-identity-64x32-16x64b`. |
| `ldst-f16-1cta-ncta1-64x32-reshape_permute-identity-identity-16x64b-roundtrip` | same row, same-view roundtrip | Clean unsupported diagnostic from `get_reg_layout`: row anchors `32,64` are not directly representable | Clean boundary; not a runtime masking positive for this subword row. |
| `ldst-i32-1cta-ncta1-64x32-double_transpose_slice-identity-identity-32x32b-readonly` | `ldst`, i32, `[64,32]`, `double_transpose_slice`, `32x32b` | Clean unsupported diagnostic from `get_reg_layout`: row anchors `32,64` are not directly representable | Clean unsupported boundary; no crash/miscompile. |
| `ldst-f32-2cta-ncta2-128x32-reshape_permute-identity-identity-32x32b-clean-unsupported` | `ldst`, f32, `[128,32]`, two-CTA, chained view | Clean unsupported diagnostic | Existing clean two-CTA chained-view boundary; no new bug. |
| `ldred-f32-1cta-ncta1-128x64-index-even_odd-identity-32x32b-min` | `ldred`, f32, `[128,64]`, direct indexed parent, row `even_odd` | Passes; emits hardware `.ld.red.` | Positive uncovered generator row. |
| `ldred-f32-1cta-ncta1-256x32-index-identity-identity-32x32b-min` | `ldred`, f32, parent `[2,256,32]`, direct `index(1)` | Compile fails in `TritonTensorMemoryAllocationPass` with `TensorMemoryAllocation.cpp:65` assertion `kNumRows - numRows >= 0` | New stable compiler crash: `FZ-20260421-0009`. |
| `ldred-f32-2cta-ncta2-256x64-index-identity-identity-32x32b-min` | `ldred`, f32, two-CTA `[256,64]`, direct indexed parent | Runtime tensors match, but PTX contains `tcgen05.ld.sync.aligned.16x32bx2.x32.b32`, not `.ld.red.` | Already-covered `FZ-20260421-0004` family; broader shape evidence only. |
| `ldred-f32-2cta-ncta2-128x32-reshape_permute-even_odd-identity-32x32b-clean-unsupported` | Intended clean two-CTA chained-view boundary | The temporary adapter mapped this to the existing column-chain helper and it compiled; not used as a durable classification row | Harness mapping limitation; not cataloged as backend evidence. |

## New Finding

### FZ-20260421-0009: 1CTA indexed ld.red 256-row parent asserts in TensorMemoryAllocation

- Failure class: `compiler_crash`.
- Family: `ldred`.
- Seed/case id:
  `ldred-f32-1cta-ncta1-256x32-index-identity-identity-32x32b-min`,
  seed `0x46708516`.
- Shape: parent `[2,256,32]`, selected view `[256,32]`.
- Layout: 1CTA `TensorMemoryLinearLayout`, row/col `identity`, lifted through
  prefix `[2]`.
- View chain: direct `parent.index(1)`.
- Operation: store full tile, then `view.load_min()`.
- Observed: compiler emits a reproducer and fails in
  `TritonTensorMemoryAllocationPass`; `triton-opt --run-reproducer` aborts
  with exit `134`.
- Assertion:
  `TensorMemoryAllocation.cpp:65: MemoryBitMap::findFirstFit(...): Assertion 'kNumRows - numRows >= 0' failed.`
- Likely owner surface:
  `TritonTensorMemoryAllocationPass` allocation sizing for a 256-row
  descriptor-selected `ld.red` source.
- Relation to existing buckets:
  distinct from `FZ-20260421-0004` because no runtime/opcode stage is reached;
  adjacent to `FZ-20260421-0005` because both hit the allocator assertion on
  256-row lifted parents, but this lane confirms the direct 1CTA indexed
  `ld.red` source path and provides a minimized MLIR replay.

Backend repair remains deferred during the discovery campaign.

## Follow-Up

- If promoted, use a subprocess-isolated strict xfail. The in-process pytest
  row keeps the parent process alive in this run, but `triton-opt` replay aborts
  with exit `134`, so crash-safe promotion should follow the existing allocator
  sentinel pattern.
- Keep `FZ-20260421-0009` separate from `FZ-20260421-0004` opcode-loss rows
  and from clean unsupported descriptor-view diagnostics.
- The Round 6 generator should still stay in `/tmp` until a normalized
  descriptor/legacy-id bridge is accepted; this lane used it only as a
  deterministic selector.
