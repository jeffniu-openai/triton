# Round 51: Reproducer Minimization for High-Priority FZ Buckets

Date: 2026-04-21 14:57 UTC

Branch/HEAD: `codex/tmem` at `c3fdbd12bebe`

Scope: cataloging/minimization only. No backend/compiler code and no checked-in
tests were modified. Disposable minimized MLIR probes were written under `/tmp`.

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

## Ranked Minimization Table

| Rank | Bucket | Smallest current repro command | Result/signature | Promotion suitability |
| --- | --- | --- | --- | --- |
| 1 | `FZ-20260421-0014` sequential mbarrier proxy-fence insertion | `bin/triton-opt /tmp/tmem_round51_fz0014_no_store.mlir --run-reproducer` from the build dir | Fails cleanly: `'tt.func' op could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses` | Best immediate lit/MLIR promotion candidate. The Round 51 27-line repro removes the pointer argument and trailing store from the older 29-line saved repro while preserving the same diagnostic. |
| 2 | `FZ-20260421-0016` unencoded TMEM operand verifier crash | `bin/triton-opt /tmp/tmem_round51_fz0016_min.mlir -split-input-file` from the build dir | Aborts in verifier path: `dyn_cast on a non-existent value`; stack includes `toLinearLayout`, `computeTMemLdStEncodingInfoImpl`, `verifyTMEMOperand`, `TMEMStoreOp::verify` | Good future clean-negative promotion after expected behavior is changed from assertion to typed diagnostic. Current test should stay xfail/crash-only if promoted before the fix. |
| 3 | `FZ-20260421-0017` encoded 64-bit TMEM load/store | `bin/triton-opt /tmp/tmem_round51_fz0017_min.mlir -split-input-file --convert-triton-gpu-to-llvm='target=sm_100'` from the build dir | Aborts in `lowerTMemLdSt`: `Assertion 'bitwidth == 32' failed` | Good targeted compiler-only promotion candidate. The 10-line repro is encoded and distinct from `FZ-0016`; it requires only a parent memdesc index and `ttng.tmem_load` returning `i64`. |
| 4 | `FZ-20260421-0001` dynamic copy parent-index | `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor_128x128b[0]'` | Runtime compile fails at `ConvertTritonGPUToLLVM`: live `ttg.memdesc_index` remains illegal while feeding `ttng.tmem_copy` and later `ttng.tmem_load` | High-value checked-in runtime or structural xfail, but less minimized than the MLIR cases because the current smallest stable repro is still a Python `/tmp` harness. Pair with a same-process `warpx2` positive when promoted. |
| 5 | `FZ-20260421-0021` unit-rank half-column descriptor view | `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python ROUND39_CASE='{"case_id":"ldst-unit_rank_half_col_upper-identity_identity-auto","family":"ldst","m":128,"n":128,"row_kind":"identity","col_kind":"identity","view_kind":"unit_rank_half_col_upper","variant":"auto","red_op":"min","seed":121}' python /tmp/tmem_high_rank_chain_shapes_round39/worker.py` | Process aborts with `TMEM layout shape must be bounded by the memdesc shape and allocShape`; shape `128, 1, 128`, allocShape `128, 1, 128`, layoutShape `128, 128` | Useful but not yet promotion-ready as-is. Needs conversion from the generic worker JSON case into a dedicated Python test or compact compiler-only reproducer. |

## Exact Rechecks

### `FZ-20260421-0001`: Dynamic Linear `ttng.tmem_copy`

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor_128x128b[0]'
```

Result: `1 failed in 4.19s`.

Failure signature:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
Pipeline failed while executing [`ConvertTritonGPUToLLVM` on 'builtin.module' operation]
RuntimeError: PassManager::run failed
```

The emitted IR shows `%2`, a branch-yielded child memdesc selected by
`ttg.memdesc_index`, feeding both `ttng.tmem_copy` and `ttng.tmem_load`.

### `FZ-20260421-0014`: Sequential Mbarrier Proxy-Fence Insertion

Baseline saved repro:

```bash
bin/triton-opt /tmp/tmem_fz0014_plain_seq_round19_min_probe_branch_repro.mlir.make_llir.repro.mlir --run-reproducer
```

Result: failed with:

```text
'tt.func' op could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
```

Round 51 minimization:

```bash
bin/triton-opt /tmp/tmem_round51_fz0014_no_store.mlir --run-reproducer
```

Result: failed with the same diagnostic. The minimized file is 27 lines and
removes the unused function argument, ordinary `tt.store`, and `%c1_i32`
constant from the earlier 29-line saved repro.

### `FZ-20260421-0016`: Unencoded TMEM Operand Verifier Crash

Existing checked-in lit guardrail:

```bash
lit -v test/Conversion/relayout_tritongpu.mlir
```

Result: `1 failed`. Failure signature:

```text
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
mlir::triton::gpu::TritonGPUDialect::toLinearLayout(...)
mlir::triton::nvidia_gpu::computeTMemLdStEncodingInfoImpl(...)
mlir::triton::nvidia_gpu::verifyTMEMOperand(...)
mlir::triton::nvidia_gpu::TMEMAllocOp::verify()
FileCheck error: '<stdin>' is empty.
```

Round 51 minimized compiler-only repro:

```bash
bin/triton-opt /tmp/tmem_round51_fz0016_min.mlir -split-input-file
```

Result: process abort. The minimized file is 8 lines and keeps only an
unencoded `tensor<128x64xf32>` source, a TMEM memdesc, and one
`ttng.tmem_store`. It preserves the same `dyn_cast on a non-existent value`
signature, with stack frames through `toLinearLayout`, `verifyTMEMOperand`, and
`TMEMStoreOp::verify`.

### `FZ-20260421-0017`: Encoded 64-bit TMEM Load/Store

Baseline saved repro:

```bash
bin/triton-opt /tmp/tmem_compiler_boundaries_round38/i64_chain_load_fz0017.mlir \
  -split-input-file \
  --convert-triton-gpu-to-llvm='target=sm_100'
```

Result: process abort:

```text
TensorMemoryUtils.cpp:7713:
Assertion `bitwidth == 32' failed.
```

Round 51 minimized compiler-only repro:

```bash
bin/triton-opt /tmp/tmem_round51_fz0017_min.mlir \
  -split-input-file \
  --convert-triton-gpu-to-llvm='target=sm_100'
```

Result: process abort with the same `bitwidth == 32` assertion. The minimized
file is 10 lines and removes unused aliases from the 25-line saved repro while
keeping the encoded `i64` child memdesc load that reaches `lowerTMemLdSt`.

### `FZ-20260421-0021`: Half-Column Descriptor View

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
ROUND39_CASE='{"case_id":"ldst-unit_rank_half_col_upper-identity_identity-auto","family":"ldst","m":128,"n":128,"row_kind":"identity","col_kind":"identity","view_kind":"unit_rank_half_col_upper","variant":"auto","red_op":"min","seed":121}' \
  python /tmp/tmem_high_rank_chain_shapes_round39/worker.py
```

Result: process abort:

```text
TMEM layout shape must be bounded by the memdesc shape and allocShape.
shape = 128, 1, 128, allocShape = 128, 1, 128, layoutShape = 128, 128
Assertion `succeeded( ConcreteT::verifyInvariants(...))' failed.
Fatal Python error: Aborted
```

This remains the least minimized of the five requested buckets because the
smallest stable repro is still embedded in the generic high-rank chain worker.
The next catalog-only sharpening step would be to extract this single case into
a dedicated `/tmp` Python file or MLIR reproducer, without changing backend code.

## Notes

- No new `FZ-*` bucket is needed.
- No backend fix was attempted.
- The `/tmp/tmem_round51_fz0014_no_store.mlir`,
  `/tmp/tmem_round51_fz0016_min.mlir`, and
  `/tmp/tmem_round51_fz0017_min.mlir` files are disposable minimization
  artifacts, not checked-in tests.
