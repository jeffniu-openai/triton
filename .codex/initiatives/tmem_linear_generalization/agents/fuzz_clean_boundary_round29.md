# Round 29 Clean-Boundary and Verifier Adversarial Probes

Date: 2026-04-21

Scope: discovery/cataloging only. No backend or compiler code was modified.

Required first step:

```bash
make -j8
```

Result: no-op rebuild through `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`.

## Summary

This lane stressed TMEM verifier and lowering clean diagnostics around edge
shapes, memory-space mismatches, memdesc/tensor shape mismatches, CTA-count
requirements, malformed linear layout attributes, and unsupported dtype /
bitwidth rows. Most probes produced typed diagnostics and did not overlap with
the unencoded-tensor verifier crash tracked as `FZ-20260421-0016`.

New candidate:

- `FZ-20260421-0017`: 64-bit TMEM load/store operands (`i64` and `f64`) are
  accepted far enough to reach `-triton-tensor-memory-allocation`, then abort in
  `lowerTMemLdSt` on `Assertion 'bitwidth == 32' failed`. This should become a
  clean verifier/planner diagnostic or a supported lowering, but should not
  crash.

No other new bucket was found in this lane.

## Checked-In Runtime Clean Boundary Sweep

Command:

```bash
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'reports_clean or clean_unsupported or clean_error or tmem_oor'
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'reports_clean or clean_unsupported or clean_error or tmem_oor'
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'reports_clean or clean_unsupported or clean_error or tmem_oor'
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'reports_clean or clean_unsupported or clean_error or tmem_oor'
```

Result:

- collection: `184/1615`
- group 1: `46 passed, 1569 deselected`
- group 2: `46 passed, 1569 deselected`
- group 3: `46 passed, 1569 deselected`
- group 4: `46 passed, 1569 deselected`

This covers the existing runtime clean-negative matrix for OOR reporting,
CTA-count mismatch reporting, unsupported copy/layout families, shared/TMEM
layout incompatibilities, unsupported subword copy forms, and scaled/plain MMA
clean unsupported rows.

## Lit Verifier Baseline

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
lit -v test/TritonNvidiaGPU/invalid.mlir
```

Result:

- `PASS: TRITON :: TritonNvidiaGPU/invalid.mlir`

The checked-in verifier baseline already includes clean diagnostics for source
shape mismatch, result alloc-shape mismatch, source element-type mismatch,
1CTA/2CTA layout mismatch, malformed linear layout attributes, immutable TMEM
stores, unsupported copy families, MMAv5-incompatible layouts, and related
TMEM verifier edges.

## Temporary MLIR Boundary Probes

Artifacts:

- `/tmp/tmem_clean_boundary_round29/summary.json`
- `/tmp/tmem_clean_boundary_round29/*.mlir`
- `/tmp/tmem_clean_boundary_round29/bitwidth/summary.json`
- `/tmp/tmem_clean_boundary_round29/bitwidth/*.mlir`
- `/tmp/tmem_clean_boundary_round29/python_frontend_probe.py`
- `/tmp/tmem_clean_boundary_round29/python_float64.err`
- `/tmp/tmem_clean_boundary_round29/python_int64.err`
- `/tmp/tmem_clean_boundary_round29/fz0017_narrow/summary.json`
- `/tmp/tmem_clean_boundary_round29/fz0017_narrow/*.mlir`
- `/tmp/tmem_clean_boundary_round29/bitwidth_copy_red/summary.json`
- `/tmp/tmem_clean_boundary_round29/bitwidth_copy_red/*.mlir`

Tool:

```bash
/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt
```

Clean diagnostics observed:

- `0x128` TMEM allocation: `shape has 0 dimension`.
- `128x0` TMEM allocation: `shape must have power-of-2 and non-zero dimensions`.
- `127x128` legacy TMEM allocation: legacy sugar cannot canonicalize because
  shape per CTA is smaller than the `128x128` tile.
- `128x127` legacy TMEM allocation: power-of-two/non-zero shape diagnostic.
- `ttng.tmem_store` into shared memory: `should use tensor memory encoding`.
- `ttng.tmem_load` from shared memory: `source must be a tensor memory buffer`.
- store source `64x128` into destination `128x128`: source/destination shape
  mismatch diagnostic.
- load source `128x128` into result `64x128`: source/destination shape
  mismatch diagnostic.
- 1CTA layout in 4CTA module: `Layout has 1 CTAs per CGA, but the context
  requires 4 CTAs per CGA`.
- 2CTA linear layout in 4CTA module: `Layout has 2 CTAs per CGA, but the
  context requires 4 CTAs per CGA`.
- negative linear basis: `Expected all basis values to be non-negative`.
- explicit `out` rank mismatch: `Explicit out rank and rank deduced from LL
  need to match`.
- non-power-of-two explicit `out` shape: `Out-dim sizes must be powers of 2`.

Supported/control bitwidth rows through `-triton-tensor-memory-allocation`:

- `i8`, `i16`, `f16`, `bf16`, `i32`, and `f32` store/load all returned `0`.

## New Candidate: FZ-20260421-0017

Minimal shape:

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f64_store_load(%arg: tensor<128x128xf64, #blocked>) {
    %true = arith.constant true
    %0 = ttng.tmem_alloc %arg : (tensor<128x128xf64, #blocked>) -> !ttg.memdesc<128x128xf64, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %arg, %0, %true : tensor<128x128xf64, #blocked> -> !ttg.memdesc<128x128xf64, #tmem, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf64, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf64, #blocked>
    "use"(%1) : (tensor<128x128xf64, #blocked>) -> ()
    tt.return
  }
}
```

Repro:

```bash
/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_clean_boundary_round29/bitwidth/f64_store_load.mlir \
  -allow-unregistered-dialect \
  -triton-tensor-memory-allocation
```

Observed result:

```text
triton-opt: /root/code/triton/lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp:7713:
... lowerTMemLdSt(...): Assertion `bitwidth == 32' failed.
```

The same abort reproduces for `i64`:

```bash
/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_clean_boundary_round29/bitwidth/i64_store_load.mlir \
  -allow-unregistered-dialect \
  -triton-tensor-memory-allocation
```

Controls in the same generated bitwidth matrix show `i8`, `i16`, `f16`,
`bf16`, `i32`, and `f32` returning successfully from
`-triton-tensor-memory-allocation`, so the crash is isolated to 64-bit
load/store planning rather than a generic malformed-IR setup issue.

This is not `FZ-20260421-0016`: the tensors are encoded, the memdesc is encoded,
and the crash is an explicit 64-bit `lowerTMemLdSt` assertion rather than the
unencoded tensor verifier path.

Python frontend reachability:

```bash
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  python3 /tmp/tmem_clean_boundary_round29/python_frontend_probe.py float64

PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  python3 /tmp/tmem_clean_boundary_round29/python_frontend_probe.py int64
```

Both frontend probes abort with exit `134` during JIT compilation. The stack
reaches `TensorMemory.get_reg_layout()` from
`python/triton/experimental/gluon/language/nvidia/blackwell/__init__.py:590`,
then trips the same `lowerTMemLdSt` `bitwidth == 32` assertion. This means
`FZ-20260421-0017` is not just an IR-only verifier gap; ordinary Gluon kernels
that try to round-trip `torch.float64` or `torch.int64` through TMEM can crash
the Python process during compilation.

Narrowing:

- uninitialized `i64`/`f64` `ttng.tmem_alloc` with no consumer returns `0`
  because allocation is dead-eliminated by the allocation pass.
- initialized `i64`/`f64` `ttng.tmem_alloc` crashes, because the initializer
  path plans a TMEM store.
- standalone `i64`/`f64` `ttng.tmem_store` crashes.
- standalone `i64`/`f64` `ttng.tmem_load` crashes.

That places the boundary at 64-bit TMEM access-layout planning, not at memdesc
type parsing or allocation object construction.

Neighboring 64-bit operation checks:

- `ttng.tmem_load` with `redOp = min` on `i64`/`f64` rejects cleanly with
  `tmem_load reduction currently requires f32 element type`.
- `ttng.tmem_copy` with 64-bit shared/TMEM memdescs survived
  `-triton-tensor-memory-allocation`; this pass does not prove final copy
  lowering support, but it did not hit the `lowerTMemLdSt` crash path.

## Next Suggested Fuzzing

- Add a checked-in strict xfail/lit reproducer for `FZ-20260421-0017` once the
  fuzzing-only phase ends or when the team starts preserving discovered
  failures.
- Extend the 64-bit repro to conversion-to-LLVM once the allocation-pass crash
  is converted into a clean diagnostic, to verify the later lowering path also
  rejects or supports it cleanly.
- Extend Python runtime coverage for `FZ-20260421-0017` once the fuzzing-only
  phase ends: a subprocess-based strict xfail is needed because the current
  failure aborts the interpreter.
