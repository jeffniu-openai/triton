# TMEM Structural Fuzzing Lane R5-C: Generic-Pass Descriptor-View Propagation

- Date: 2026-04-21
- Branch: `codex/tmem`
- HEAD: `729dde3d1`
- Mode: discovery only; no backend/compiler repairs attempted.
- Scope: generic-pass descriptor-view propagation beyond R4-A simple
  `if`/mixed/layout-pressure rows: `scf.for` loop-carried memdesc views,
  nested conditionals, multiple live views from one parent, helper-returned
  view chains, and select-like values where Gluon can express them.
- Temporary harness: `/tmp/tmem_generic_pass_round5.py`

## Commands

Required rebuild before pytest:

```bash
CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8
```

Result: `ninja: no work to do`.

Collection:

```bash
PYTHONPATH=/root/code/triton/python pytest --collect-only -q /tmp/tmem_generic_pass_round5.py
```

Result: `10 tests collected`.

Four-GPU shard sweep:

```bash
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r5c2-gpu0 pytest -q -s --tb=short --splits 4 --group 1 /tmp/tmem_generic_pass_round5.py 2>&1 | tee /tmp/tmem_generic_pass_round5_v2_g1.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r5c2-gpu1 pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_generic_pass_round5.py 2>&1 | tee /tmp/tmem_generic_pass_round5_v2_g2.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r5c2-gpu2 pytest -q -s --tb=short --splits 4 --group 3 /tmp/tmem_generic_pass_round5.py 2>&1 | tee /tmp/tmem_generic_pass_round5_v2_g3.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r5c2-gpu3 pytest -q -s --tb=short --splits 4 --group 4 /tmp/tmem_generic_pass_round5.py 2>&1 | tee /tmp/tmem_generic_pass_round5_v2_g4.log
```

Results:

- group 1: `3 failed, 7 deselected`
- group 2: `3 failed, 7 deselected`
- group 3: `3 failed, 7 deselected`
- group 4: `1 failed, 9 deselected`
- aggregate: `10 failed`

Fresh representative confirmations:

```bash
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r5c-confirm-loop pytest -q -s --tb=short '/tmp/tmem_generic_pass_round5.py::test_generic_pass_round5[loop-carried-chain0-selector1]' 2>&1 | tee /tmp/tmem_generic_pass_round5_confirm_loop.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r5c-confirm-nested pytest -q -s --tb=short '/tmp/tmem_generic_pass_round5.py::test_generic_pass_round5[nested-if-chain0-else-else]' 2>&1 | tee /tmp/tmem_generic_pass_round5_confirm_nested.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r5c-confirm-multilive pytest -q -s --tb=short '/tmp/tmem_generic_pass_round5.py::test_generic_pass_round5[multi-live-chain1-sibling-sum]' 2>&1 | tee /tmp/tmem_generic_pass_round5_confirm_multilive.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r5c-control pytest -q -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]' 2>&1 | tee /tmp/tmem_generic_pass_round5_control_ldst.log
```

Results:

- loop-carried confirmation: `1 failed`
- nested-if confirmation: `1 failed`
- multi-live confirmation: `1 failed`
- known ldst structural-fuzzer control: `1 passed`

## Pass/Fail Matrix

| Case | Result | Failure mode |
| --- | --- | --- |
| `loop-carried-chain0-selector1` | fail | `GluonResolveAutoEncodingsPass` compile crash; emitted IR contains `scf.for ... iter_args(%arg5 = %31) -> !ttg.memdesc<128x64xf32, ...>` plus nested `scf.if` yielding memdesc views. |
| `loop-carried-chain1-selector1` | fail | Same auto-layout compile crash; loop-carried memdesc has chain1 identity view type. |
| `loop-carried-chain0-selector0` | fail | Same auto-layout compile crash; selector changes carried result but not root symptom. |
| `nested-if-chain0-else-else` | fail | `GluonResolveAutoEncodingsPass` compile crash; emitted IR contains nested `scf.if` memdesc results. |
| `nested-if-chain1-else-else` | fail | Same auto-layout compile crash; chain1 variant also fails. |
| `multi-live-chain0-sibling-sum` | fail | Same auto-layout compile crash with two live sibling views from the same parent. |
| `multi-live-chain1-sibling-sum` | fail | Same auto-layout compile crash; fresh rerun stable. |
| `helper-chain-return-chain0` | fail | Same auto-layout compile crash with view chain returned through two helper calls after an `scf.if` base selection. |
| `helper-chain-return-chain2` | fail | Same auto-layout compile crash for helper-returned slice/slice chain. |
| `nested-if-chain0-16x128b` | fail | Same auto-layout compile crash under 16x128b packet layout. |

Common diagnostic:

```text
error: 'tt.make_range' op Failed to infer return type
note: Pipeline failed while executing [`GluonResolveAutoEncodingsPass` on 'builtin.module' operation]
RuntimeError: PassManager::run failed
```

## Finding

### R5C-001: memdesc-valued generic control flow can break auto-layout inference before runtime

- Extends: generic-pass structural fuzzing, adjacent to but distinct from
  `FZ-20260421-0002`.
- Failure class: `compiler_crash`.
- Likely owner surface: `GluonResolveAutoEncodingsPass` / auto-layout
  inference when TMEM descriptor-view values flow through `scf.for` iter args,
  nested `scf.if` results, or multiple live view users.
- Shape: parent `[2, 128, 64]`, selected view `[128, 64]`, dtype `f32`.
- View chains:
  - chain0: `reshape((64,2,64)).permute([1,0,2]).reshape((128,64))`
  - chain1: double transpose identity view
  - chain2: slice/slice identity view
- Observed: compilation stops before runtime execution. The failing IR still
  contains the intended memdesc control-flow constructs, and the auto-layout
  failure is reported on the first `tt.make_range` used for the global
  load/store offsets.
- Control: checked-in
  `test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]`
  passes in the same environment.

This is not the same runtime symptom as R4-A's chain0 `8064/8192` mismatch.
R4-A reached runtime and showed chain1/chain2 green boundaries for simple
`if`/mixed/layout-pressure rows. R5-C instead crashes in the compiler before
output comparison, and it affects chain0, chain1, and chain2 variants once the
memdesc is carried through loop/nested/multi-live/helper control-flow forms.

## Select-Like Surface

A true `arith.select` over memdesc values was not directly expressible through
the Gluon Python surface used here. The expressible scalar-select alternative
is `parent.index(ttgl.where(...))` or `parent.index(tt.load(...))`, which
overlaps the existing `FZ-20260421-0001` dynamic `memdesc_index` illegal-op
bucket, so this lane did not recommend another duplicate xfail for it.

## Recommended Strict Xfails

Recommend one new checked-in xfail sentinel only:

- `generic-pass-loop-carried-memdesc-view-chain0`: covers the non-overlapping
  `scf.for` loop-carried memdesc crash root and pins the `iter_args` surface.

Do not promote every failing R5-C row yet. The nested-if, multi-live, and
helper-chain rows are useful repair-validation cases, but their current
diagnostic is the same `GluonResolveAutoEncodingsPass` auto-layout crash. They
should stay report-only until the loop-carried sentinel is fixed or proves to
mask multiple independent roots.
