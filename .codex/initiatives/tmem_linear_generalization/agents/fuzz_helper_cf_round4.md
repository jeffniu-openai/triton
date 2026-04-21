# TMEM Structural Fuzzing Lane R4-A: Helper/Control-Flow Expansion

- Date: 2026-04-21
- Branch: `codex/tmem`
- HEAD: `d06d769d8`
- Mode: discovery only; no backend/compiler repairs attempted.
- Scope: expand/minimize `FZ-20260421-0002` helper-returned TMEM view
  miscompiles through control flow, mixed tensor+memdesc captures,
  tuple-like captures, layout-conversion pressure, 16x64b/16x128b packet
  layouts, chain1/chain2 boundaries, nested helper returns, and no-helper
  inline comparisons.
- Temporary harness: `/tmp/tmem_helper_cf_round4.py`

## Commands

Required rebuild before runtime tests:

```bash
make -j8
```

Result: `ninja: no work to do`.

Collection:

```bash
PYTHONPATH=/root/code/triton/python pytest --collect-only -q /tmp/tmem_helper_cf_round4.py
```

Result: `24 tests collected`.

Four-GPU shard sweep:

```bash
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r4a2-gpu0 pytest -q -s --tb=short --splits 4 --group 1 /tmp/tmem_helper_cf_round4.py 2>&1 | tee /tmp/tmem_helper_cf_round4_v2_g1.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r4a2-gpu1 pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_helper_cf_round4.py 2>&1 | tee /tmp/tmem_helper_cf_round4_v2_g2.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r4a2-gpu2 pytest -q -s --tb=short --splits 4 --group 3 /tmp/tmem_helper_cf_round4.py 2>&1 | tee /tmp/tmem_helper_cf_round4_v2_g3.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r4a2-gpu3 pytest -q -s --tb=short --splits 4 --group 4 /tmp/tmem_helper_cf_round4.py 2>&1 | tee /tmp/tmem_helper_cf_round4_v2_g4.log
```

Results:

- group 1: `2 failed, 4 passed, 18 deselected`
- group 2: `5 failed, 1 passed, 18 deselected`
- group 3: `2 failed, 4 passed, 18 deselected`
- group 4: `2 failed, 4 passed, 18 deselected`
- aggregate: `11 failed, 13 passed`

Fresh representative confirmations:

```bash
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r4a-confirm-if16x128 pytest -q -s --tb=short '/tmp/tmem_helper_cf_round4.py::test_helper_cf_round4[if-false-chain0-128x64-16x128b-helper]' 2>&1 | tee /tmp/tmem_helper_cf_round4_confirm_if16x128.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r4a-confirm-inline pytest -q -s --tb=short '/tmp/tmem_helper_cf_round4.py::test_helper_cf_round4[if-false-chain0-128x64-32x32b-inline]' 2>&1 | tee /tmp/tmem_helper_cf_round4_confirm_inline.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r4a-confirm-tuple pytest -q -s --tb=short '/tmp/tmem_helper_cf_round4.py::test_helper_cf_round4[tuple-mixed-false-chain0-128x64-32x32b-helper]' 2>&1 | tee /tmp/tmem_helper_cf_round4_confirm_tuple.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r4a-confirm-pressure16x128 pytest -q -s --tb=short '/tmp/tmem_helper_cf_round4.py::test_helper_cf_round4[pressure-chain0-128x64-16x128b-helper]' 2>&1 | tee /tmp/tmem_helper_cf_round4_confirm_pressure16x128.log
```

All four representative confirmations failed with the same stable output
mismatch pattern as the shard sweep.

Positive boundary confirmations:

```bash
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r4a-confirm-chain1pass pytest -q -s --tb=short '/tmp/tmem_helper_cf_round4.py::test_helper_cf_round4[if-false-chain1-128x64-16x128b-helper]' 2>&1 | tee /tmp/tmem_helper_cf_round4_confirm_chain1pass.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r4a-confirm-chain2pass pytest -q -s --tb=short '/tmp/tmem_helper_cf_round4.py::test_helper_cf_round4[pressure-chain2-128x64-16x128b-helper]' 2>&1 | tee /tmp/tmem_helper_cf_round4_confirm_chain2pass.log
```

Both passed with `mismatches=0/8192`.

## Pass/Fail Matrix

| Case | Result | Notes |
| --- | --- | --- |
| `if-false-chain0-128x64-32x32b-helper` | fail | Known `FZ-20260421-0002` false-branch adjacency; `8064/8192` mismatches. |
| `if-false-chain0-128x64-16x128b-helper` | fail | New 16x128b adjacency; `8064/8192` mismatches. |
| `if-false-chain1-128x64-32x32b-helper` | pass | Chain1 boundary stays green. |
| `if-false-chain1-128x64-16x128b-helper` | pass | Chain1 remains green under 16x128b. |
| `if-true-chain1-128x64-16x64b-helper` | pass | Chain1 remains green under true branch and 16x64b. |
| `if-false-chain2-128x64-32x32b-helper` | pass | Chain2 boundary stays green. |
| `if-false-chain2-128x64-16x128b-helper` | pass | Chain2 remains green under 16x128b. |
| `if-false-chain0-128x128-16x128b-helper` | fail | New wider-column 16x128b adjacency; `16128/16384` mismatches. |
| `if-false-chain0-128x64-32x32b-direct` | fail | Chain body inside dispatcher still fails; `8064/8192` mismatches. |
| `if-false-chain0-128x64-32x32b-nested` | fail | Nested helper return still fails; `8064/8192` mismatches. |
| `if-false-chain0-128x64-32x32b-inline` | fail | True no-helper inline chain0 still fails; `8064/8192` mismatches. |
| `mixed-false-chain0-128x64-32x32b-helper` | fail | Known mixed tensor+memdesc capture variant; `8064/8192` mismatches. |
| `mixed-false-chain1-128x64-32x32b-helper` | pass | Chain1 mixed capture stays green. |
| `mixed-true-chain2-128x64-16x64b-helper` | pass | Chain2 mixed capture stays green. |
| `tuple-mixed-false-chain0-128x64-32x32b-helper` | fail | New tuple-like memdesc+tensor capture adjacency; `8064/8192` mismatches. |
| `tuple-mixed-true-chain1-128x64-16x64b-helper` | pass | Tuple-like capture is not independently broken for chain1 true branch. |
| `tuple-mixed-false-chain1-128x64-16x128b-helper` | pass | Tuple-like capture plus 16x128b stays green for chain1. |
| `pressure-chain0-128x64-16x128b-helper` | fail | New layout-conversion pressure 16x128b adjacency; `8064/8192` mismatches. |
| `pressure-chain1-128x64-16x64b-helper` | pass | Chain1 pressure stays green. |
| `pressure-chain1-128x64-16x128b-helper` | pass | Chain1 pressure stays green under 16x128b. |
| `pressure-chain2-128x64-32x32b-helper` | pass | Chain2 pressure stays green. |
| `pressure-chain2-128x64-16x128b-helper` | pass | Chain2 pressure stays green under 16x128b. |
| `pressure-chain0-128x64-16x64b-direct` | fail | Layout pressure fails when chain0 is in dispatcher body; `8064/8192` mismatches. |
| `pressure-chain0-128x64-16x64b-nested` | fail | Layout pressure fails through nested helper return; `8064/8192` mismatches. |

## Minimized Findings

### R4A-001: chain0 false-branch miscompile reaches 16x128b

- Extends: `FZ-20260421-0002`.
- Failure class: `miscompile`.
- Shape: parent `[2, 128, 64]`, selected view `[128, 64]`, dtype `f32`.
- View chain: `reshape((64, 2, 64)).permute([1, 0, 2]).reshape((128, 64))`.
- Control flow: runtime false branch selects `parent.index(1)`.
- Instruction layout: `16x128b`.
- Fresh result: `8064/8192` mismatches.
- Opcode evidence: PTX and LLIR both contain four
  `tcgen05.st.sync.aligned.16x128b.x16.b32` stores and two
  `tcgen05.ld.sync.aligned.16x128b.x16.b32` loads.
- Representative command:

```bash
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r4a-confirm-if16x128 pytest -q -s --tb=short '/tmp/tmem_helper_cf_round4.py::test_helper_cf_round4[if-false-chain0-128x64-16x128b-helper]'
```

### R4A-002: helper boundary is not required

- Extends: `FZ-20260421-0002`.
- Failure class: `miscompile`.
- Shape/layout: same as R4A-001 with `32x32b`.
- Comparison:
  - helper return: fails;
  - nested helper return: fails;
  - dispatcher body: fails;
  - true no-helper inline chain0: fails.
- Fresh inline result: `8064/8192` mismatches.
- Opcode evidence: PTX and LLIR both contain two
  `tcgen05.st.sync.aligned.32x32b.x64.b32` stores and one
  `tcgen05.ld.sync.aligned.32x32b.x64.b32` load.
- Interpretation: the current bug label "helper-returned TMEM view" is too
  narrow. The failing core appears to be chain0 descriptor-view layout
  arithmetic through selected TMEM provenance/control-flow, not Python helper
  boundaries or inlining.

### R4A-003: tuple-like mixed memdesc+tensor capture reproduces

- Extends: `FZ-20260421-0002`.
- Failure class: `miscompile`.
- Shape: parent `[2, 128, 64]`, selected view `[128, 64]`, dtype `f32`.
- View chain: chain0.
- Capture pattern: helper returns `(memdesc_view, tensor_bias)`, then the
  kernel indexes the pair and computes `view.load(...) + bias`.
- Fresh result: `8064/8192` mismatches.
- Opcode evidence: PTX and LLIR both contain two
  `tcgen05.st.sync.aligned.32x32b.x64.b32` stores and one
  `tcgen05.ld.sync.aligned.32x32b.x64.b32` load.
- Boundary: tuple-like capture is not independently sufficient; chain1 tuple
  rows passed under both 16x64b and 16x128b.

### R4A-004: layout-conversion pressure reaches 16x128b

- Extends: `FZ-20260421-0002`.
- Failure class: `miscompile`.
- Shape: parent `[2, 128, 64]`, selected view `[128, 64]`, dtype `f32`.
- View chain: chain0 over `parent.index(1)`.
- Pressure pattern: store base in `32x32b`, load view in `16x128b`, convert
  `16x128b -> 32x32b -> 16x128b`, then store to global.
- Fresh result: `8064/8192` mismatches.
- Opcode evidence: PTX and LLIR both contain
  `tcgen05.st.sync.aligned.32x32b.x64.b32` plus two
  `tcgen05.ld.sync.aligned.16x128b.x16.b32` loads.

## Boundaries and Non-Findings

- Chain1 and chain2 are clean positives in this harness across dynamic true
  and false branches, mixed tensor+memdesc captures, tuple-like captures,
  layout-conversion pressure, `16x64b`, and `16x128b`.
- `16x128b` itself is not generally broken: chain1 and chain2 16x128b rows
  passed with matching PTX/LLIR opcodes and exact output.
- Python helper boundaries are not necessary for the chain0 failure. The
  no-helper inline kernel fails identically to helper and nested-helper forms.
- No compiler crash, false unsupported diagnostic, or opcode mismatch was
  observed in this lane; all failures compiled and launched, then mismatched
  the torch reference.

## Recommended Strict Xfails

Add only a small number of sentinel xfails rather than promoting every
adjacent failing row:

- `generic-pass-dynamic-if-chain0-false-16x128b`: covers false-branch
  control flow plus 16x128b packet layout.
- `generic-pass-dynamic-if-chain0-inline`: documents that helper boundaries
  are not required and prevents overfitting the bug to helper returns.
- `generic-pass-tuple-mixed-captures-chain0`: covers tuple-like
  memdesc+tensor capture without relying on a frontend error.
- `generic-pass-layout-conversion-pressure-chain0-16x128b`: covers the
  16x128b layout-pressure extension.

The existing checked-in `generic-pass-dynamic-if-chain0-true`,
`generic-pass-mixed-captures-chain0`, and
`generic-pass-layout-conversion-pressure-chain0` xfails still cover the
original R2 surface. The new recommended sentinels close the currently missing
false-branch, 16x128b, tuple-capture, and inlining/boundary dimensions without
duplicating the green chain1/chain2 boundary rows.
