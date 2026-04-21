# TMEM Structural Fuzzing Round 8 Lane D: Generic Compiler Pass / Analysis Interactions

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Scope: temporary Python/Gluon runtime probes around TMEM memdesc values
  through helper returns, tuple-like returns, direct controls, sibling views,
  static/constexpr indexes, `ttgl.static_range`, dynamic `scf.if`, runtime
  `memdesc_index`, layout conversions, and non-TMEM tensor side inputs.
- Temporary harness: `/tmp/tmem_generic_pass_round8.py`
- Intent: avoid spending the lane on rows already masked by the known R5-C
  `GluonResolveAutoEncodingsPass` / `tt.make_range` auto-layout crash, and
  prefer cases that reach runtime comparison or known illegal-op lowering.

## Commands

Required rebuild before tests:

```bash
CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8
```

Result: `ninja: no work to do`.

Initial collection:

```bash
PYTHONPATH=.:./python python -m py_compile /tmp/tmem_generic_pass_round8.py
PYTHONPATH=.:./python pytest --collect-only -q /tmp/tmem_generic_pass_round8.py
```

Result: `11 tests collected`.

After adding direct base/direct view controls and using `atol=rtol=1e-5` to
avoid classifying f32 addition-order noise as a TMEM failure:

```bash
PYTHONPATH=.:./python python -m py_compile /tmp/tmem_generic_pass_round8.py
PYTHONPATH=.:./python pytest --collect-only -q /tmp/tmem_generic_pass_round8.py
```

Result: `14 tests collected`.

Four-GPU shard sweep:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r8d2-gpu0 pytest -q -s --tb=short --splits 4 --group 1 /tmp/tmem_generic_pass_round8.py 2>&1 | tee /tmp/tmem_generic_pass_round8_v2_g1.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r8d2-gpu1 pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_generic_pass_round8.py 2>&1 | tee /tmp/tmem_generic_pass_round8_v2_g2.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r8d2-gpu2 pytest -q -s --tb=short --splits 4 --group 3 /tmp/tmem_generic_pass_round8.py 2>&1 | tee /tmp/tmem_generic_pass_round8_v2_g3.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r8d2-gpu3 pytest -q -s --tb=short --splits 4 --group 4 /tmp/tmem_generic_pass_round8.py 2>&1 | tee /tmp/tmem_generic_pass_round8_v2_g4.log
```

Results:

- group 1: `2 failed, 2 passed, 10 deselected`;
- group 2: `2 failed, 2 passed, 10 deselected`;
- group 3: `1 failed, 3 passed, 10 deselected`;
- group 4: `2 failed, 12 deselected`;
- aggregate: `7 failed, 7 passed`.

Fresh representative confirmations:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r8d-confirm-direct pytest -q -s --tb=short '/tmp/tmem_generic_pass_round8.py::test_generic_pass_round8[r8-static-direct-view-chain0-control]' 2>&1 | tee /tmp/tmem_generic_pass_round8_confirm_direct.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r8d-confirm-tuple pytest -q -s --tb=short '/tmp/tmem_generic_pass_round8.py::test_generic_pass_round8[r8-static-helper-tuple-chain0-index1]' 2>&1 | tee /tmp/tmem_generic_pass_round8_confirm_tuple.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r8d-confirm-dynsame pytest -q -s --tb=short '/tmp/tmem_generic_pass_round8.py::test_generic_pass_round8[r8-dynamic-if-same-base-chain0]' 2>&1 | tee /tmp/tmem_generic_pass_round8_confirm_dynsame.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r8d-control-base pytest -q -s --tb=short '/tmp/tmem_generic_pass_round8.py::test_generic_pass_round8[r8-static-direct-base-control]' 2>&1 | tee /tmp/tmem_generic_pass_round8_control_base.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r8d-confirm-chain1 pytest -q -s --tb=short '/tmp/tmem_generic_pass_round8.py::test_generic_pass_round8[r8-static-direct-view-chain1-control]' 2>&1 | tee /tmp/tmem_generic_pass_round8_control_chain1.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r8d-confirm-dyn1 pytest -q -s --tb=short '/tmp/tmem_generic_pass_round8.py::test_generic_pass_round8[r8-dynamic-if-chain1-selector1]' 2>&1 | tee /tmp/tmem_generic_pass_round8_confirm_dyn1.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r8d-confirm-dyn2 pytest -q -s --tb=short '/tmp/tmem_generic_pass_round8.py::test_generic_pass_round8[r8-dynamic-if-chain2-selector0-16x128b]' 2>&1 | tee /tmp/tmem_generic_pass_round8_confirm_dyn2.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r8d-confirm-index pytest -q -s --tb=short '/tmp/tmem_generic_pass_round8.py::test_generic_pass_round8[r8-dynamic-index-direct-base]' 2>&1 | tee /tmp/tmem_generic_pass_round8_confirm_index.log
```

Results:

- direct chain0 view control: `1 failed`, `8061 / 8192` mismatched;
- static helper tuple over chain0: `1 failed`, same mismatch scale;
- dynamic `if` choosing the same chain0 base on both arms: `1 failed`,
  `8064 / 8192` mismatched;
- direct base control: `1 passed`;
- direct chain1 view control: `1 passed`;
- dynamic `if` chain1 with side tensor: `1 passed`;
- dynamic `if` chain2 with `16x128b` and side tensor: `1 passed`;
- dynamic runtime index direct base: `1 failed` in `ConvertTritonGPUToLLVM`
  with illegal `ttg.memdesc_index`, matching `FZ-20260421-0001`.

## Pass/Fail Matrix

| Case | Result | Classification |
| --- | --- | --- |
| `r8-static-direct-base-control` | pass | Direct parent index/base load control is green. |
| `r8-static-direct-view-chain0-control` | fail | Runtime miscompile, `8061 / 8192` mismatched; overlaps the descriptor-view load/store packet-order bucket (`FZ-20260421-0003`), not a generic-pass-specific root. |
| `r8-static-direct-view-chain1-control` | pass | Chain1 direct view control is green. |
| `r8-static-helper-tuple-chain0-index1` | fail | Same chain0 packet-order mismatch as the direct chain0 control; tuple/helper structure is not required. |
| `r8-static-helper-tuple-chain1-index0` | pass | Static helper plus tuple-like `(memdesc, tensor)` return is green for chain1. |
| `r8-static-helper-tuple-chain0-16x128b-revcol` | fail | Same direct chain0 descriptor-view mismatch under `16x128b` and reversed columns; overlaps `FZ-20260421-0003`. |
| `r8-static-sibling-chain0` | fail | Same chain0 descriptor-view mismatch, amplified because two sibling view loads are summed. |
| `r8-static-sibling-chain1-evenrow` | pass | Multiple sibling views are green for the chain1/even-row row. |
| `r8-static-for-chain0-select1` | fail | Same chain0 descriptor-view mismatch after `ttgl.static_range`; static loop is not the root. |
| `r8-static-for-chain1-select0` | pass | Static loop plus chain1 view is green. |
| `r8-dynamic-if-chain1-selector1` | pass | Dynamic `scf.if`, side tensor, and chain1 view reach runtime and pass under normal f32 tolerance. |
| `r8-dynamic-if-chain2-selector0-16x128b` | pass | Dynamic `scf.if`, side tensor, chain2 slice/slice identity, and `16x128b` reach runtime and pass. |
| `r8-dynamic-if-same-base-chain0` | fail | Same chain0 descriptor-view mismatch even when both `scf.if` arms choose the same base; this argues against the branch merge as the root for this row. |
| `r8-dynamic-index-direct-base` | fail | `FZ-20260421-0001`; runtime `ttg.memdesc_index` survives to LLVM conversion as an illegal op. |

## Findings

### R8D-001: chain0 descriptor-view load mismatch is independent of generic control flow

- Extends: `FZ-20260421-0003`.
- Failure class: `miscompile`.
- Shape: parent `[2,128,64]`, selected view `[128,64]`, dtype `f32`.
- View chain: `reshape((64,2,64)).permute([1,0,2]).reshape((128,64))`.
- Observed: direct chain0 load, static helper tuple, static sibling views,
  static loop selection, and dynamic `if` choosing the same chain0 base all
  produce large packet-order mismatches around `8061-8064 / 8192` elements.
- Boundary: direct base load passes, direct chain1 view passes, static tuple
  chain1 passes, sibling chain1 passes, static loop chain1 passes, and dynamic
  `if` chain1/chain2 pass.
- Diagnosis: this is useful generic-pass classification evidence, but not a
  new generic-pass bug. The direct chain0 control fails before tuple returns,
  helper returns, non-TMEM side tensors, `scf.if`, or `ttgl.static_range`
  matter. The generic wrappers only expose the same underlying descriptor-view
  packet mapping problem already tracked by `FZ-20260421-0003`.

### R8D-002: runtime direct `memdesc_index` still reproduces the known illegal-op lowering

- Extends: `FZ-20260421-0001`.
- Failure class: `compiler_failure`.
- Shape: parent `[2,128,64]`, selected view `[128,64]`, dtype `f32`.
- Observed: direct `parent.index(ttgl.load(selector_ptr))` compiles far enough
  to emit a reproducer, then fails in `ConvertTritonGPUToLLVM` because
  `ttg.memdesc_index` remains illegal.
- Boundary: static/constexpr parent indexes and direct base controls pass.

## R5-C Avoidance / Classification

This lane intentionally avoided nested memdesc-valued `scf.if`/dynamic
`scf.for` rows that are already known to stop in
`GluonResolveAutoEncodingsPass` with a `tt.make_range` auto-layout inference
failure. No new R5-C reproducer was needed.

The dynamic `if` rows that used chain1 or chain2 reached runtime and passed,
which gives a useful contrast against Round 7: generic control flow over a
memdesc is not uniformly broken once the row avoids the current chain0
descriptor-view packet mismatch and the R5-C auto-layout owner surface.

The initial exact-equality sweep reported small `3.8e-6` mismatches for the
dynamic `if` chain1/chain2 rows. After rerunning with `atol=rtol=1e-5`, those
rows passed and were classified as normal f32 arithmetic/layout-conversion
rounding, not backend miscompiles.

## Promotion Recommendations

- Do not assign a new `FZ-*` id from this lane.
- Do not promote a new generic-pass strict xfail. The only generic-pass-shaped
  failure that reaches runtime is dominated by the direct chain0 descriptor
  view failure, so it belongs with `FZ-20260421-0003` repair validation.
- Keep the following rows as repair-validation inventory after `FZ-0003` is
  fixed: `r8-static-helper-tuple-chain0-index1`,
  `r8-static-sibling-chain0`, `r8-static-for-chain0-select1`, and
  `r8-dynamic-if-same-base-chain0`. If they still fail after the direct
  chain0 control is green, they should be reconsidered as generic-pass
  findings.
- Keep `r8-dynamic-index-direct-base` report-only under `FZ-20260421-0001`;
  the checked-in dynamic-index sentinels already cover the owner.
