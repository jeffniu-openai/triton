# TMEM Structural Fuzzing Lane E: Generic Pass Round 1

- Time: 2026-04-21 08:25 UTC
- Branch: `codex/tmem`
- HEAD: `57c00629b`
- Lane: E, generic pass and layout-analysis adversarial testing around TMEM memdescs
- Mode: discovery only; no backend fixes attempted

## Setup

- Read `AGENTS.md`.
- Read `.codex/initiatives/tmem_linear_generalization/tmem_structural_fuzzing_20260421.md`.
- Ran required build first:
  - `make`
  - Result: `ninja: no work to do.`

Initial pytest runs without `PYTHONPATH=./python` imported stale `/tmp/triton-upstream-main-check/python/triton`; all meaningful runs below pin `PYTHONPATH=./python`.

## Probe Harness

Temporary runtime probe harness:

- `/tmp/lane_e_tmem_generic_pass_probe.py`

Harness shape/common parameters:

- `family`: `generic_pass`
- `shape`: `[128, 64]`
- `dtype`: `float32`
- `layout_kind`: identity `TensorMemoryLinearLayout`, lifted to parent shape `[2, 128, 64]`
- `num_warps`: `4`
- `num_ctas`: `1`
- GPU target observed in repros: `cuda:103` on GB300

The harness stresses:

- runtime `memdesc_index` from a loaded selector;
- dynamic `if` values returning TMEM memdescs;
- mixed tensor + memdesc captures from `if`;
- view chains crossing Gluon helper functions;
- `convert_layout` pressure around `RemoveLayoutConversions`/coalescing/thread-locality style pipelines.

## Commands Run

Baseline structural fuzzer, 4-GPU split:

```bash
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_structural_fuzzer.py
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_structural_fuzzer.py
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_structural_fuzzer.py
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_structural_fuzzer.py
```

Lane E harness, 4-GPU split:

```bash
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short --splits 4 --group 1 /tmp/lane_e_tmem_generic_pass_probe.py
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 pytest -s --tb=short --splits 4 --group 2 /tmp/lane_e_tmem_generic_pass_probe.py
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 pytest -s --tb=short --splits 4 --group 3 /tmp/lane_e_tmem_generic_pass_probe.py
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 pytest -s --tb=short --splits 4 --group 4 /tmp/lane_e_tmem_generic_pass_probe.py
```

Exact confirmation commands:

```bash
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short '/tmp/lane_e_tmem_generic_pass_probe.py::test_generic_pass_memdesc_cf_runtime[generic-pass-dynamic-index-chain0]' 2>&1 | tee /tmp/lane_e_dynamic_index_chain0.log
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short '/tmp/lane_e_tmem_generic_pass_probe.py::test_generic_pass_memdesc_cf_runtime[generic-pass-dynamic-index-chain1]' 2>&1 | tee /tmp/lane_e_dynamic_index_chain1.log
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 pytest -s --tb=short '/tmp/lane_e_tmem_generic_pass_probe.py::test_generic_pass_memdesc_cf_runtime[generic-pass-dynamic-if-chain0-true]' 2>&1 | tee /tmp/lane_e_dynamic_if_chain0_true.log
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 pytest -s --tb=short '/tmp/lane_e_tmem_generic_pass_probe.py::test_generic_pass_memdesc_cf_runtime[generic-pass-mixed-captures-chain0]' 2>&1 | tee /tmp/lane_e_mixed_captures_chain0.log
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 pytest -s --tb=short '/tmp/lane_e_tmem_generic_pass_probe.py::test_generic_pass_layout_conversion_pressure[chain0]' 2>&1 | tee /tmp/lane_e_layout_conversion_chain0.log
```

Control confirmations:

```bash
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]'
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-2cta]'
```

## Findings

### LANE-E-GP-001: runtime TMEM `memdesc_index` reaches LLVM lowering as an illegal op

- `case_id`: `generic-pass-dynamic-index-chain0`
- `seed`: `0xE001`
- `view_chain`: `chain0`, `parent.index(runtime_i32).reshape((64,2,64)).permute([1,0,2]).reshape((128,64))`
- `expected`: pass, or at minimum a clean unsupported diagnostic if runtime TMEM indexing is intentionally unsupported
- `observed`: compiler crash
- `failure_class`: `compiler_crash`
- `likely_owner_surface`: dynamic memdesc indexing legalization / `ConvertTritonGPUToLLVM`
- Exact command:
  - `PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short '/tmp/lane_e_tmem_generic_pass_probe.py::test_generic_pass_memdesc_cf_runtime[generic-pass-dynamic-index-chain0]'`
- Repro log:
  - `/tmp/lane_e_dynamic_index_chain0.log`
- Key diagnostic:
  - `failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal`
  - Pipeline failed while executing `ConvertTritonGPUToLLVM`.
- Repro/minimization status:
  - Isolated to one pytest nodeid.
  - Compiler emitted full MLIR plus `mlir_reproducer` metadata in the log.
  - Further minimization could drop the initial two stores and retain only `ttg.memdesc_index %result[%runtime_i32]` plus a TMEM use.

### LANE-E-GP-002: runtime TMEM `memdesc_index` also crashes through a reshape-only view chain

- `case_id`: `generic-pass-dynamic-index-chain1`
- `seed`: `0xE011`
- `view_chain`: `chain1`, `parent.index(runtime_i32).reshape((64,2,32,2)).permute([1,0,3,2]).permute([1,0,3,2]).reshape((128,64))`
- `expected`: pass, or clean unsupported diagnostic if runtime TMEM indexing is intentionally unsupported
- `observed`: compiler crash
- `failure_class`: `compiler_crash`
- `likely_owner_surface`: dynamic memdesc indexing legalization / `ConvertTritonGPUToLLVM`
- Exact command:
  - `PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short '/tmp/lane_e_tmem_generic_pass_probe.py::test_generic_pass_memdesc_cf_runtime[generic-pass-dynamic-index-chain1]'`
- Repro log:
  - `/tmp/lane_e_dynamic_index_chain1.log`
- Key diagnostic:
  - `failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal`
  - Pipeline failed while executing `ConvertTritonGPUToLLVM`.
- Repro/minimization status:
  - Isolated to one pytest nodeid.
  - Same root likely as LANE-E-GP-001, but confirmed not specific to the chain0 permute shape.

### LANE-E-GP-003: helper-returned chain0 TMEM view miscompiles under dynamic `if` memdesc value

- `case_id`: `generic-pass-dynamic-if-chain0-true`
- `seed`: `0xE001`
- `view_chain`: `chain0` returned through `_identity_view0(...)`
- `selector`: runtime scalar `1`, so the true branch selects `parent.index(0)`
- `expected`: `out == input + 30.0`
- `observed`: miscompare after successful compilation/execution
- `failure_class`: `miscompile`
- `likely_owner_surface`: `scf.if` memdesc result + Gluon inline/view-chain layout analysis; possibly coalescing or layout-conversion assumptions after helper inlining
- Exact command:
  - `PYTHONPATH=./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 pytest -s --tb=short '/tmp/lane_e_tmem_generic_pass_probe.py::test_generic_pass_memdesc_cf_runtime[generic-pass-dynamic-if-chain0-true]'`
- Repro log:
  - `/tmp/lane_e_dynamic_if_chain0_true.log`
- Failure detail:
  - `Mismatched elements: 8064 / 8192 (98.4%)`
  - greatest absolute difference `5.677539825439453`
- Repro/minimization status:
  - Isolated to one pytest nodeid.
  - Control `test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]` passes, so the plain inline chain0 ldst path is healthy; the adversarial surface is helper/control-flow interaction.

### LANE-E-GP-004: helper-returned chain0 TMEM view miscompiles with mixed tensor + memdesc captures

- `case_id`: `generic-pass-mixed-captures-chain0`
- `seed`: `0xE001`
- `view_chain`: `chain0` returned through `_identity_view0(...)`
- `selector`: runtime scalar `1`, true branch returns both a TMEM view and tensor bias
- `expected`: `out == input + input + 3.0`
- `observed`: miscompare after successful compilation/execution
- `failure_class`: `miscompile`
- `likely_owner_surface`: mixed tensor/memdesc `scf.if` result handling after Gluon inline and generic layout passes
- Exact command:
  - `PYTHONPATH=./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 pytest -s --tb=short '/tmp/lane_e_tmem_generic_pass_probe.py::test_generic_pass_memdesc_cf_runtime[generic-pass-mixed-captures-chain0]'`
- Repro log:
  - `/tmp/lane_e_mixed_captures_chain0.log`
- Failure detail:
  - `Mismatched elements: 8064 / 8192 (98.4%)`
  - greatest absolute difference `5.677539825439453`
- Repro/minimization status:
  - Isolated to one pytest nodeid.
  - Likely same descriptor-view component as LANE-E-GP-003, but keeps the mixed tensor+memdesc capture requested for this lane.

### LANE-E-GP-005: helper-returned chain0 TMEM view miscompiles under layout-conversion pressure

- `case_id`: `generic-pass-layout-conversion-pressure-chain0`
- `seed`: `0xE100`
- `view_chain`: `chain0` returned through `_identity_view0(...)`
- `stress`: store through `32x32b`, load through `16x64b`, then convert `16x64b -> 32x32b -> 16x64b` before global store
- `expected`: `out == input`
- `observed`: miscompare after successful compilation/execution
- `failure_class`: `miscompile`
- `likely_owner_surface`: helper-inlined TMEM view chain plus `RemoveLayoutConversions` / coalescing / layout-analysis interaction
- Exact command:
  - `PYTHONPATH=./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 pytest -s --tb=short '/tmp/lane_e_tmem_generic_pass_probe.py::test_generic_pass_layout_conversion_pressure[chain0]'`
- Repro log:
  - `/tmp/lane_e_layout_conversion_chain0.log`
- Failure detail:
  - `Mismatched elements: 8064 / 8192 (98.4%)`
  - greatest absolute difference `5.439816474914551`
- Repro/minimization status:
  - Isolated to one pytest nodeid.
  - Chain1 and chain2 variants in the same harness passed, so this currently appears chain0-specific.

## Passing / Non-Finding Cases

- `generic-pass-dynamic-if-chain1-false`: passed in split run.
- `generic-pass-mixed-captures-chain1`: passed in split run with float32 tolerance.
- `test_generic_pass_layout_conversion_pressure[chain1]`: passed in split run.
- `test_generic_pass_layout_conversion_pressure[chain2]`: passed in split run.
- Inline baseline `test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]`: passed exact rerun.

Baseline note:

- In the 4-GPU split baseline, group 3 initially reported `test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-2cta]` observing `cta_group::1` opcodes where the test expected `cta_group::2`.
- Exact fresh rerun of that nodeid passed, so this is not recorded as a lane E finding yet. It may be process/cache/order-sensitive and should be rechecked by a copy/scales lane.

## Suggested Next Cases

- Reduce LANE-E-GP-001 to a lit `ttg.memdesc_index` with runtime index and one use to decide whether the right behavior is support or a clean unsupported diagnostic.
- Compare helper-returned chain0 view vs identical inline chain0 view in one harness to pinpoint whether `gluon-inline` changes the TMEM layout metadata.
- Add dynamic `if` memdesc result cases where both branches use chain1/chain2 but differ only by parent index, to see whether the miscompile is chain0-only or branch-result-origin sensitive.
- Add `scf.for` memdesc `iter_args` lit probes with TMEM memdescs after this runtime batch; Python/Gluon runtime syntax did not directly expose a clean dynamic loop construct during this round.
