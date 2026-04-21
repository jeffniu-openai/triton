# Round 30 FZ-0017 64-bit TMEM Access Expansion

Date: 2026-04-21

Mode: discovery/cataloging only. No backend or compiler code was modified.

Required first step:

```bash
make -j8
```

Result: no-op rebuild through
`/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`.

## Summary

This lane expanded `FZ-20260421-0017`, the encoded 64-bit TMEM load/store
crash. The failure is broader than the original Round 29 roundtrip:

- `i64` and `f64` direct TMEM load/store crash for legacy and linear TMEM
  layouts.
- initialized allocation, store-only, load-only, and store-then-load all reach
  the same `lowerTMemLdSt` assertion.
- 1CTA and valid 2CTA linear layouts both reproduce the assertion when the
  register layout is CGA-aware enough to pass earlier verification.
- valid descriptor-view `i64`/`f64` load/store can pass
  `-triton-tensor-memory-allocation`, then crash in the later
  TritonGPU-to-LLVM path when the load/store is actually lowered.
- Python/Gluon frontend kernels using `torch.float64` or `torch.int64` reach
  the same assertion for roundtrip, store-only, load-only, legacy layout,
  linear layout, descriptor-sliced view, and 2CTA linear rows.

No new independent bucket was found. These are all classified as expansions of
`FZ-20260421-0017`.

## Artifacts

Temporary artifacts:

- `/tmp/tmem_fz0017_round30/summary.json`
- `/tmp/tmem_fz0017_round30/*.mlir`
- `/tmp/tmem_fz0017_round30/focused/summary.json`
- `/tmp/tmem_fz0017_round30/focused/*.mlir`
- `/tmp/tmem_fz0017_round30/descriptor_valid/summary.json`
- `/tmp/tmem_fz0017_round30/descriptor_valid/llvm_summary.json`
- `/tmp/tmem_fz0017_round30/descriptor_valid/*.mlir`
- `/tmp/tmem_fz0017_round30/copy_summary.json`
- `/tmp/tmem_fz0017_round30/python_file_summary.json`
- `/tmp/tmem_fz0017_round30/python_file_probe.log`
- `/tmp/tmem_fz0017_round30_python_child.py`
- `/tmp/tmem_fz0017_round30_python_controller.py`

Main tool:

```bash
/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt
```

## MLIR Matrix

Primary allocation-pass matrix:

```bash
triton-opt <case>.mlir -allow-unregistered-dialect -triton-tensor-memory-allocation
```

Result:

| Class | Count |
| --- | ---: |
| `ASSERT_BITWIDTH_32` | 60 |
| `CLEAN_DIAGNOSTIC` | 115 |
| `PASS` | 25 |

Confirmed assertion rows include:

- `i64` and `f64`;
- shapes `128x64`, `128x128`, `128x256`, and linear `256x32`;
- legacy and linear TMEM layouts;
- initialized alloc, store-only, load-only, and store/load roundtrip;
- permuted linear register/TMEM layout rows.

The assertion is:

```text
TensorMemoryUtils.cpp:7713: lowerTMemLdSt(...): Assertion `bitwidth == 32' failed.
```

Clean diagnostics in the same matrix came from unrelated structural boundaries:

- `64x64` rows with invalid register layouts for this lowering path;
- legacy `blockM = 256`, which is rejected as `blockM must be 64 or 128`;
- `ld.red` on non-`f32`, which cleanly reports
  `tmem_load reduction currently requires f32 element type`;
- malformed descriptor-view result layouts from the first rough view probe.

Neighboring bitwidth controls:

- `i1`, `i4`, `i8`, `i16`, `i32`, `f16`, `bf16`, and `f32` direct
  load/store rows passed allocation.
- `f32` `ld.red` passed.
- non-`f32` `ld.red` rows rejected cleanly.

## Focused 2CTA and High-CGA Probe

Focused command:

```bash
triton-opt /tmp/tmem_fz0017_round30/focused/<case>.mlir \
  -allow-unregistered-dialect \
  -triton-tensor-memory-allocation
```

Result:

| Class | Count |
| --- | ---: |
| `ASSERT_BITWIDTH_32` | 6 |
| `CLEAN_CTA_DIAGNOSTIC` | 12 |
| `CLEAN_DIAGNOSTIC` | 4 |

The valid 2CTA rows reproduced `FZ-0017` for both `i64` and `f64`:

- `i64_twocta_load`
- `i64_twocta_store`
- `i64_twocta_roundtrip`
- `f64_twocta_load`
- `f64_twocta_store`
- `f64_twocta_roundtrip`

High-CGA memdesc-only rows for 4, 8, and 16 CTA modules rejected before any
bitwidth-specific lowering. These are clean `FZ-0010`-style CTA-count
diagnostics, not new 64-bit crashes. The diagnostics include the expected
context-requires-CTA-count text after the invalid-layout header.

## Descriptor Views

Valid descriptor-view rows were generated with a parent linear TMEM layout and
a `ttg.memdesc_subslice` result that preserves the parent layout attribute and
changes only the logical shape/alloc shape. Allocation alone passes for both
32-bit and 64-bit descriptor-view load/store rows:

```bash
triton-opt /tmp/tmem_fz0017_round30/descriptor_valid/<case>.mlir \
  -allow-unregistered-dialect \
  -triton-tensor-memory-allocation
```

Later lowering exposes `FZ-0017`:

```bash
triton-opt /tmp/tmem_fz0017_round30/descriptor_valid/<case>.mlir \
  -allow-unregistered-dialect \
  --triton-tensor-memory-allocation \
  --allocate-shared-memory-nv=compute-capability=100 \
  --convert-triton-gpu-to-llvm=compute-capability=100
```

Result:

| Class | Count |
| --- | ---: |
| `ASSERT_BITWIDTH_32` | 4 |
| `PASS` | 4 |

`i64`/`f64` descriptor-view load and store crash at the same
`lowerTMemLdSt` assertion. `i32`/`f32` descriptor-view load and store pass the
same pipeline.

## Copy and Reduction Adjacent Paths

64-bit `ttng.tmem_copy` shared-to-TMEM probes did not reproduce the assertion:

```bash
triton-opt /tmp/tmem_clean_boundary_round29/bitwidth_copy_red/i64_copy_shared_to_tmem.mlir \
  -allow-unregistered-dialect \
  --triton-tensor-memory-allocation \
  --allocate-shared-memory-nv=compute-capability=100 \
  --convert-triton-gpu-to-llvm=compute-capability=100
```

Result:

| Case | Allocation | LLVM conversion |
| --- | --- | --- |
| `i64_copy_shared_to_tmem` | pass | pass |
| `f64_copy_shared_to_tmem` | pass | pass |

64-bit reduction loads remain clean diagnostics:

```text
'ttng.tmem_load' op tmem_load reduction currently requires f32 element type
```

That keeps `FZ-0017` scoped to non-reduction TMEM load/store planning, not
`tcgen05.copy` or `ld.red` verifier policy.

## Python Frontend Reachability

The corrected frontend harness uses a real temporary Python module because
Gluon JIT rejects `python -c` kernels:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_fz0017_round30_python_controller.py
```

Result:

| Class | Count |
| --- | ---: |
| `ASSERT_BITWIDTH_32` | 30 |
| `PASS` | 14 |
| `PYTHON_OR_COMPILE_ERROR` | 16 |

By dtype:

| Dtype | Result |
| --- | --- |
| `torch.float64` | 15 `ASSERT_BITWIDTH_32` |
| `torch.int64` | 15 `ASSERT_BITWIDTH_32` |
| `torch.float32` | 7 pass, 8 harness/frontend compile errors |
| `torch.int32` | 7 pass, 8 harness/frontend compile errors |

The `float64` and `int64` rows reproduced the crash for:

- roundtrip, store-only, and load-only;
- linear and legacy TMEM layouts;
- descriptor-sliced views;
- 2CTA linear layout rows.

The `float32`/`int32` frontend compile errors are harness limitations, not new
TMEM backend buckets:

- the rough sliced-view harness attempted unsupported tensor slicing syntax on
  a register tensor (`vals[:, :N // 2]`);
- the 2CTA load/roundtrip controls hit a Python lowering/harness issue
  (`'list' object has no attribute 'type'`) while 2CTA store-only passed.

Those controls are useful as evidence that the 64-bit rows reach the bitwidth
assertion earlier and more consistently than comparable 32-bit rows, but they
should not be cataloged as new backend bugs from this lane.

## Classification

No new `FZ-*` bucket was created.

Expanded existing bucket:

- `FZ-20260421-0017`: encoded 64-bit `i64`/`f64` TMEM load/store lowering
  reaches `lowerTMemLdSt` and aborts on `bitwidth == 32`. The expanded surface
  now includes legacy and linear layouts, 1CTA and 2CTA, initialized allocation,
  store-only, load-only, roundtrip, descriptor-view load/store at LLVM
  conversion, and Python/Gluon frontend kernels.

Clean boundaries:

- 64-bit `ld.red` rejects cleanly because reduction load currently requires
  `f32`.
- 64-bit shared-to-TMEM `ttng.tmem_copy` passes allocation and LLVM conversion
  in the generated probes.
- high-CGA local 1CTA/2CTA rows reject via existing CTA-count diagnostics,
  consistent with `FZ-20260421-0010` and not a new 64-bit failure mode.

## Next Fuzz Hooks

- Build a valid Python descriptor-view harness that avoids unsupported register
  tensor slicing, then rerun 32-bit controls and 64-bit descriptor-view rows
  side by side.
- After the fuzzing-only phase ends, preserve `FZ-0017` with checked-in
  subprocess tests because the current failure aborts the compiler process.
- Once the backend has a clean 64-bit policy, rerun the descriptor-view LLVM
  conversion rows and the Python 2CTA rows to see whether a later PTX/SASS
  boundary appears.
