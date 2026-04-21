# TMEM Structural Fuzzing Lane E2: Generic Pass Promotion Round 2

- Time: 2026-04-21 08:30 UTC
- Branch: `codex/tmem`
- Mode: discovery only; no backend fixes attempted
- Scope: promote/minimize Lane E generic-pass findings FZ-20260421-0001 and
  FZ-20260421-0002 into checked-in Python runtime xfail tests.

## Changes

Promoted the temporary Lane E probe harness into
`python/test/gluon/test_tmem_structural_fuzzer.py` as self-contained runtime
cases. The new coverage keeps the reproducer local to the structural fuzzer
file and does not import helpers from other tests.

New strict xfail nodeids:

- `test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-index-chain0]`
- `test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-index-chain1]`
- `test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-if-chain0-true]`
- `test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-mixed-captures-chain0]`
- `test_tmem_structural_fuzzer_generic_pass_layout_conversion_pressure[generic-pass-layout-conversion-pressure-chain0]`

## Finding Status

### FZ-20260421-0001

- Case ids:
  - `generic-pass-dynamic-index-chain0`
  - `generic-pass-dynamic-index-chain1`
- Failure class: `compiler_crash`
- Current checked-in status: strict xfail runtime tests.
- Observed during validation: both fresh-process nodeids still xfail with
  `failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal`
  in `ConvertTritonGPUToLLVM`.
- Expected future behavior: either compile and pass the runtime/output/opcode
  assertions, or be reclassified into a clean typed unsupported diagnostic with
  the test contract updated intentionally.

### FZ-20260421-0002

- Case ids:
  - `generic-pass-dynamic-if-chain0-true`
  - `generic-pass-mixed-captures-chain0`
  - `generic-pass-layout-conversion-pressure-chain0`
- Failure class: `miscompile`
- Current checked-in status: strict xfail runtime tests.
- Observed during validation: each fresh-process nodeid still xfails under
  the runtime output comparison.
- Expected future behavior: pass the runtime comparison and keep matching PTX
  and LLIR `tcgen05.ld/st` opcode extraction.

## Validation

Required build before tests:

```bash
make
```

Result: `ninja: no work to do.`

Collection and syntax checks:

```bash
PYTHONPATH=.:./python python -m py_compile python/test/gluon/test_tmem_structural_fuzzer.py
PYTHONPATH=.:./python pytest -q --collect-only python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'
```

Result: `5/18` selected generic-pass nodeids collected.

Fresh-process exact nodeid validation:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-index-chain0]'
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-index-chain1]'
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-if-chain0-true]'
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-mixed-captures-chain0]'
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_layout_conversion_pressure[generic-pass-layout-conversion-pressure-chain0]'
```

Result: all five commands exited successfully as `1 xfailed`.

## Notes

- Backend/compiler code was not changed.
- The promoted tests intentionally assert the desired positive runtime
  behavior, with strict xfail markers documenting the current failure mode.
- During this edit, other uncommitted structural-fuzzer changes for
  FZ-20260421-0003/0004/0006 were present in the same file and were preserved.
