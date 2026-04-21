# TMEM Structural Fuzzing Round 6 Lane A: Generic-Pass Memdesc Values

- Date: 2026-04-21
- Branch: `codex/tmem`
- HEAD: `00238cd5a`
- Mode: discovery only; no backend/compiler repairs attempted.
- Scope: generic-pass TMEM memdesc values through loop-carried state,
  nested `if`, multi-live sibling views, helper returns, tuple captures, and
  descriptor-view chains.

## Summary

Round 6 did not find a new non-overlapping generic-pass bucket. The temporary
R5-C harness still reproduces the same compiler crash for memdesc-valued
generic control-flow and multi-live view forms:

- failure class: `compiler_crash`;
- likely owner surface: `GluonResolveAutoEncodingsPass` / auto-layout
  inference;
- common diagnostic: `'tt.make_range' op Failed to infer return type`, with
  `RuntimeError: PassManager::run failed`;
- classification: already-covered by R5-C, with checked-in strict xfail
  sentinel
  `generic-pass-loop-carried-memdesc-view-chain0`.

The checked-in tuple and inline/control-flow rows still xfail under existing
`FZ-20260421-0002` coverage, so no new central catalog entry was added.

## Commands

Required rebuild before pytest:

```bash
CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8
```

Result: `ninja: no work to do`.

Temporary harness used:

- `/tmp/tmem_generic_pass_round5.py`

Collection:

```bash
PYTHONPATH=.:./python python -m py_compile /tmp/tmem_generic_pass_round5.py
PYTHONPATH=.:./python pytest -q --collect-only /tmp/tmem_generic_pass_round5.py
PYTHONPATH=.:./python pytest -q --collect-only python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'
```

Results:

- temporary harness py-compile passed;
- temporary harness collected `10 tests`;
- checked-in generic-pass selector collected `11/30` nodeids.

Four-GPU temporary harness sweep:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r6-generic-gpu0 pytest -q -s --tb=short --splits 4 --group 1 /tmp/tmem_generic_pass_round5.py 2>&1 | tee /tmp/tmem_generic_pass_round6_g1.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r6-generic-gpu1 pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_generic_pass_round5.py 2>&1 | tee /tmp/tmem_generic_pass_round6_g2.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r6-generic-gpu2 pytest -q -s --tb=short --splits 4 --group 3 /tmp/tmem_generic_pass_round5.py 2>&1 | tee /tmp/tmem_generic_pass_round6_g3.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r6-generic-gpu3 pytest -q -s --tb=short --splits 4 --group 4 /tmp/tmem_generic_pass_round5.py 2>&1 | tee /tmp/tmem_generic_pass_round6_g4.log
```

Results:

- group 1: `3 failed, 7 deselected`;
- group 2: `3 failed, 7 deselected`;
- group 3: `3 failed, 7 deselected`;
- group 4: `1 failed, 9 deselected`;
- aggregate: `10 failed`.

Checked-in sentinel confirmations:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r6-tuple-gpu0 pytest -q -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-tuple-mixed-captures-chain0]' 2>&1 | tee /tmp/tmem_generic_pass_round6_tuple.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r6-inline-gpu1 pytest -q -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-if-chain0-inline]' 2>&1 | tee /tmp/tmem_generic_pass_round6_inline.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r6-loop-sentinel-gpu2 pytest -q -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_loop_carried[generic-pass-loop-carried-memdesc-view-chain0]' 2>&1 | tee /tmp/tmem_generic_pass_round6_loop_sentinel.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r6-helper-gpu3 pytest -q -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-if-chain0-true]' 2>&1 | tee /tmp/tmem_generic_pass_round6_dynamic_if_true.log
```

Results: all four reported `1 xfailed`.

## Case Classification

| Case | Seed | Result | Classification |
| --- | --- | --- | --- |
| `loop-carried-chain0-selector1` | `0x5C01` | `GluonResolveAutoEncodingsPass` crash | already-covered compiler crash, R5-C |
| `loop-carried-chain1-selector1` | `0x5C02` | same crash | already-covered compiler crash, R5-C |
| `loop-carried-chain0-selector0` | `0x5C03` | same crash | already-covered compiler crash, R5-C |
| `nested-if-chain0-else-else` | `0x5C04` | same crash | already-covered compiler crash, R5-C report-only adjacent row |
| `nested-if-chain1-else-else` | `0x5C05` | same crash | already-covered compiler crash, R5-C report-only adjacent row |
| `multi-live-chain0-sibling-sum` | `0x5C06` | same crash | already-covered compiler crash, R5-C report-only adjacent row |
| `multi-live-chain1-sibling-sum` | `0x5C07` | same crash | already-covered compiler crash, R5-C report-only adjacent row |
| `helper-chain-return-chain0` | `0x5C08` | same crash | already-covered compiler crash, R5-C report-only adjacent row |
| `helper-chain-return-chain2` | `0x5C09` | same crash | already-covered compiler crash, R5-C report-only adjacent row |
| `nested-if-chain0-16x128b` | `0x5C0A` | same crash | already-covered compiler crash, R5-C report-only adjacent row |
| `generic-pass-tuple-mixed-captures-chain0` | `0xE024` | strict xfail | already-covered runtime miscompile, `FZ-20260421-0002` |
| `generic-pass-dynamic-if-chain0-inline` | `0xE023` | strict xfail | already-covered runtime miscompile, `FZ-20260421-0002` |
| `generic-pass-loop-carried-memdesc-view-chain0` | `0x5C01` | strict xfail | checked-in R5-C compiler-crash sentinel |
| `generic-pass-dynamic-if-chain0-true` | `0xE001` | strict xfail | already-covered runtime miscompile, `FZ-20260421-0002` |

## Catalog Decision

No new central `tmem_structural_fuzzing_20260421.md` entry was added. Round 6
reconfirmed the existing generic-pass buckets and did not identify a stable
new finding outside:

- `FZ-20260421-0002`: helper/control-flow/tuple/layout-pressure runtime
  miscompiles;
- R5-C: memdesc-valued `scf.for`, nested `scf.if`, multi-live sibling views,
  and helper-returned descriptor chains crashing auto-layout inference.

## Next Probe Suggestions

- After the R5-C auto-layout crash is fixed, rerun the same temporary harness
  because the current crash masks whether nested-if, multi-live, and helper
  chain rows have downstream runtime address-arithmetic bugs.
- Keep tuple-capture coverage as `FZ-20260421-0002`; do not promote another
  tuple sentinel unless a chain1/chain2 tuple row reaches a different failure
  mode.
- For future discovery, prefer probes that avoid memdesc-valued `scf.if` or
  `scf.for` when the goal is to expose downstream ld/st planner bugs, because
  the current generic-pass crash dominates those forms.
