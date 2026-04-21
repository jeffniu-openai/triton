# TMEM Structural Fuzzing Lane R3-A: Dynamic `memdesc_index` Round 3

- Time: 2026-04-21 08:39 UTC
- Branch: `codex/tmem`
- HEAD: `78f2845a0`
- Scope: minimize and diagnose `FZ-20260421-0001`, dynamic TMEM
  `memdesc_index` reaching LLVM conversion as illegal `ttg.memdesc_index`.
- Mode: discovery/minimization only; no backend/compiler repairs attempted.

## Setup

Read first, per lane instruction:

- `.codex/initiatives/tmem_linear_generalization/tmem_structural_fuzzing_20260421.md`
- `python/test/gluon/test_tmem_structural_fuzzer.py`

Required build before pytest:

```bash
make -j8
```

Result: `ninja: no work to do.`

Temporary probes:

- `/tmp/tmem_memdesc_index_r3_probe.py`
- `/tmp/tmem_memdesc_index_r3_compile_only.py`

## Commands and Results

Collection/syntax:

```bash
PYTHONPATH=.:./python python -m py_compile /tmp/tmem_memdesc_index_r3_probe.py
PYTHONPATH=.:./python pytest -q --collect-only /tmp/tmem_memdesc_index_r3_probe.py
```

Result: `11 tests collected`.

Dynamic direct and view-chain probes:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r3-memdesc-gpu0 pytest -s --tb=short '/tmp/tmem_memdesc_index_r3_probe.py::test_dynamic_index[16-16-1-0]' 2>&1 | tee /tmp/tmem_memdesc_index_r3_dynamic_16x16_direct.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r3-memdesc-gpu1 pytest -s --tb=short '/tmp/tmem_memdesc_index_r3_probe.py::test_dynamic_index[64-32-1-0]' 2>&1 | tee /tmp/tmem_memdesc_index_r3_dynamic_64x32_direct.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r3-memdesc-gpu2 pytest -s --tb=short '/tmp/tmem_memdesc_index_r3_probe.py::test_dynamic_index[128-64-1-0]' 2>&1 | tee /tmp/tmem_memdesc_index_r3_dynamic_128x64_direct.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r3-memdesc-gpu0 pytest -s --tb=short '/tmp/tmem_memdesc_index_r3_probe.py::test_dynamic_index[128-64-0-2]' 2>&1 | tee /tmp/tmem_memdesc_index_r3_dynamic_128x64_chain2.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r3-memdesc-gpu2 python - <<'PY' 2>&1 | tee /tmp/tmem_memdesc_index_r3_dynamic_128x32_direct.log
import importlib.util
spec = importlib.util.spec_from_file_location('r3', '/tmp/tmem_memdesc_index_r3_probe.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod._run_dynamic(128, 32, 1, 0)
PY
```

Results:

- `[16,16]` fails before the target bucket with clean frontend
  `TMEM layout '32x32b' unsupported` for the descriptor view.
- `[64,32]` fails before the target bucket with the clean direct ld/st row
  anchor diagnostic: required row anchors `32,64` are not directly
  representable.
- `[128,64]` direct dynamic indexing fails in `ConvertTritonGPUToLLVM` with
  `failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal`.
- `[128,64]` chain2 fails with the same illegal dynamic `ttg.memdesc_index`;
  the descriptor reshape chain is after the failing op and is not required.
- `[128,32]` direct dynamic indexing also fails in `ConvertTritonGPUToLLVM`
  with the same illegal dynamic `ttg.memdesc_index`.

Resource boundary probe:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r3-memdesc-gpu3 python - <<'PY' 2>&1 | tee /tmp/tmem_memdesc_index_r3_dynamic_256x32_direct.log
import importlib.util
spec = importlib.util.spec_from_file_location('r3', '/tmp/tmem_memdesc_index_r3_probe.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod._run_dynamic(256, 32, 1, 0)
PY
```

Result: different known allocator boundary, assertion
`kNumRows - numRows >= 0`, before this lane's LLVM legalization failure.

Static/constexpr controls:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r3-memdesc-gpu1 pytest -s --tb=short '/tmp/tmem_memdesc_index_r3_probe.py::test_constexpr_index[128-64-1-0]' 2>&1 | tee /tmp/tmem_memdesc_index_r3_constexpr_128x64_direct_idx1.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r3-memdesc-gpu0 pytest -s --tb=short '/tmp/tmem_memdesc_index_r3_probe.py::test_constexpr_index[128-64-0-0]' 2>&1 | tee /tmp/tmem_memdesc_index_r3_constexpr_128x64_direct_idx0.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r3-memdesc-gpu1 pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]' 2>&1 | tee /tmp/tmem_memdesc_index_r3_static_ldst_view_identity.log
```

Results:

- constexpr direct index `0`: `1 passed`
- constexpr direct index `1`: `1 passed`
- checked-in static descriptor-view ldst identity chain: `1 passed`

Control note: `/tmp/tmem_memdesc_index_r3_probe.py::test_constexpr_index[128-64-1-1]`
compiled but failed runtime comparison with the already cataloged
helper-returned descriptor-view miscompile shape (`8064 / 8192` mismatched).
That is not this bucket; it confirms direct constexpr `memdesc_index` is not
the failing operation, while helper/view-chain arithmetic remains covered by
`FZ-20260421-0002`.

Load-only compile probe:

```bash
PYTHONPATH=.:./python python -m py_compile /tmp/tmem_memdesc_index_r3_compile_only.py
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r3-memdesc-loadonly-gpu0 python /tmp/tmem_memdesc_index_r3_compile_only.py 2>&1 | tee /tmp/tmem_memdesc_index_r3_dynamic_128x32_load_only.log
```

Result: same `ConvertTritonGPUToLLVM` illegal dynamic `ttg.memdesc_index`
with no TMEM stores and no descriptor view chain.

Smallest structural shape found:

```mlir
%result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x32xf32, ...>
%0 = tt.load %arg1 : !tt.ptr<i32>
%1 = ttg.memdesc_index %result[%0] : ... -> !ttg.memdesc<128x32xf32, ...>
%result_0 = ttng.tmem_load %1 : !ttg.memdesc<128x32xf32, ...> -> tensor<128x32xf32, ...>
tt.store ...
```

No control-flow merge, helper function, reshape, permute, or pre-store is
needed. The only dynamic ingredient is the scalar loaded index operand.

## Lit / MLIR Repro Status

The compiler-emitted MLIR from the load-only probe reproduces directly with
`triton-opt --run-reproducer`:

```bash
sed -n '/^#linear = /,/^#-}/p' /tmp/tmem_memdesc_index_r3_dynamic_128x32_load_only.log | build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt --run-reproducer - 2>&1 | tee /tmp/tmem_memdesc_index_r3_triton_opt_load_only_stdin.log
```

Result:

- Fails at the dynamic `ttg.memdesc_index` line.
- The pass manager reports failure while executing `ConvertTritonGPUToLLVM`.
- `verify_each: true` in the reproducer metadata did not reject the IR before
  conversion.

This can be expressed as a lit/MLIR repro. The best checked-in lit form should
be a negative `triton-opt --run-reproducer` or pass-pipeline test that asserts
the current diagnostic if dynamic TMEM indexing is intentionally unsupported.
If dynamic indexing is meant to be legal, keep the lit repro as a temporary
crash reproducer and replace it with positive lowering/runtime coverage after
the backend is repaired.

## Diagnosis

This is a generic pass legalization gap, not a descriptor-view-chain-specific
failure:

- direct `[2,128,32] -> [128,32]` runtime index reaches LLVM conversion;
- chain0/chain1/chain2/chain3 only add downstream descriptor-view operations
  after the same dynamic index;
- static/constexpr direct `parent.index(0)` and `parent.index(1)` lower and
  run;
- the compiler accepts the dynamic-index IR through earlier pipeline stages
  and only fails when `ttg.memdesc_index` is marked illegal for LLVM
  conversion.

It is also a verifier/diagnostic gap if runtime TMEM indexing is unsupported:
the user-facing failure should be a clean unsupported diagnostic before
`ConvertTritonGPUToLLVM`, not an illegal-op pass-manager failure with a full
MLIR dump.

## Recommended Checked-In Test Additions

- Add one strict xfail Python/Gluon runtime or compile-only case for the
  minimized direct dynamic index shape `[2,128,32] -> [128,32]`, no view chain.
  This is narrower than the existing promoted `[128,64]` chain0/chain1 xfails
  and isolates `FZ-20260421-0001` from descriptor-view arithmetic.
- Add one lit/MLIR repro from the load-only emitted module to pin the current
  failure surface at `ConvertTritonGPUToLLVM`. Use this primarily as a
  diagnostic/negative test if the intended contract is "dynamic TMEM
  memdesc_index unsupported".
- Keep the existing chain0/chain1 structural-fuzzer xfails for coverage of
  runtime indexing composed with descriptor views, but do not treat the view
  chain as the root cause for `FZ-20260421-0001`.
