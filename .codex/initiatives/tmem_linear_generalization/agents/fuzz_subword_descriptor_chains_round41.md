# Round 41 Subword Descriptor-Chain Boundary Fuzzing

Date: 2026-04-21

Mode: discovery/cataloging only. No backend, compiler, or checked-in test code
was modified.

Owned write path for this lane:
`.codex/initiatives/tmem_linear_generalization/agents/fuzz_subword_descriptor_chains_round41.md`.

## Scope

This lane targeted TMEM descriptor-view chains and adjacent non-`f32` data
paths:

- subword `f16`/`bf16`/`i16`/`i8` load/store and copy rows;
- `i32` x1/narrow load/store clean boundaries;
- non-`f32` `ld.red` software/diagnostic contracts for `i32`, `bf16`, `f16`,
  `i16`, and `i8`;
- f16/f8/i8 MMAv5 paths and scaled f8/fp4 narrow accumulator views;
- core descriptor-chain matrix and f16 packed roundtrip rows; and
- compiler-only `i64`/`f64` descriptor-view, copy, and `ld.red` contrasts.

Classification was explicitly checked against known `FZ-20260421-0016`,
`FZ-20260421-0017`, `FZ-20260421-0020`, `FZ-20260421-0021`, and
`FZ-20260421-0022`.

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

## Checked-In Selector

Collection command:

```bash
PYTHONPATH=.:./python:./python/test/gluon pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  python/test/gluon/test_core.py \
  -k '(ldst_subword or x1_subword or cp_128x128_subword_exact_width or cp_no_scales_linear_subword_dtypes or cp_no_scales_twocta_linear_indexed_view or cp_no_scales_twocta_linear_subslice_view or cp_no_scales_warpx2_subword_dtypes_report_clean_error or cp_no_scales_legacy_subword_dtypes_report_clean_error or cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported or ld_red_non_f32 or subword or descriptor_chain_matrix or tmem_linear_f16_roundtrip or tmem_packed_f16_roundtrip or root_format or plain_kind or bscale_descriptor_view or narrow_format or identity_narrow_view or acc_identity_narrow or clean_unsupported or clean_error) and not resource and not reports'
```

Result: `388/19762` tests collected.

Four-GPU runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group <group> \
  --store-durations --durations-path /tmp/tmem_round41_checked_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  python/test/gluon/test_core.py \
  -k '(ldst_subword or x1_subword or cp_128x128_subword_exact_width or cp_no_scales_linear_subword_dtypes or cp_no_scales_twocta_linear_indexed_view or cp_no_scales_twocta_linear_subslice_view or cp_no_scales_warpx2_subword_dtypes_report_clean_error or cp_no_scales_legacy_subword_dtypes_report_clean_error or cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported or ld_red_non_f32 or subword or descriptor_chain_matrix or tmem_linear_f16_roundtrip or tmem_packed_f16_roundtrip or root_format or plain_kind or bscale_descriptor_view or narrow_format or identity_narrow_view or acc_identity_narrow or clean_unsupported or clean_error) and not resource and not reports'
```

Results:

| Group | GPU | Log | Result |
| --- | ---: | --- | --- |
| 1 | 0 | `/tmp/tmem_round41_checked_g1.log` | `97 passed, 19665 deselected` |
| 2 | 1 | `/tmp/tmem_round41_checked_g2.log` | `97 passed, 19665 deselected` |
| 3 | 2 | `/tmp/tmem_round41_checked_g3.log` | `97 passed, 19665 deselected` |
| 4 | 3 | `/tmp/tmem_round41_checked_g4.log` | `94 passed, 3 skipped, 19665 deselected` |

Aggregate: `385 passed, 3 skipped`.

The three skips were stable selected environment/resource-gated rows from the
existing checked-in suite. There were no unexpected failures, XPASS transitions,
compiler crashes, false unsupported diagnostics, opcode absence signals, or
runtime mismatches in the checked-in sweep.

## Compiler-Only 64-Bit And Boundary Probe

Final corrected summary artifact:
`/tmp/tmem_round41_compiler_summary.txt`.

Per-case logs:
`/tmp/tmem_round41_compiler_cases/*.log`.

Command pattern:

```bash
BUILD_DIR=$(PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
TRITON_OPT="$BUILD_DIR/bin/triton-opt"

timeout 60 "$TRITON_OPT" <case>.mlir \
  -allow-unregistered-dialect \
  --triton-tensor-memory-allocation \
  --allocate-shared-memory-nv=compute-capability=100 \
  --convert-triton-gpu-to-llvm=compute-capability=100
```

Results:

| Case | Result | Classification |
| --- | --- | --- |
| `i64_descriptor_valid_load.mlir` | `rc=134`, `bitwidth == 32` assertion | Existing `FZ-20260421-0017` |
| `i64_descriptor_valid_store.mlir` | `rc=134`, `bitwidth == 32` assertion | Existing `FZ-20260421-0017` |
| `f64_descriptor_valid_load.mlir` | `rc=134`, `bitwidth == 32` assertion | Existing `FZ-20260421-0017` |
| `f64_descriptor_valid_store.mlir` | `rc=134`, `bitwidth == 32` assertion | Existing `FZ-20260421-0017` |
| `i32_descriptor_valid_load.mlir` | `rc=0` | Green control |
| `i32_descriptor_valid_store.mlir` | `rc=0` | Green control |
| `f32_descriptor_valid_load.mlir` | `rc=0` | Green control |
| `f32_descriptor_valid_store.mlir` | `rc=0` | Green control |
| `i64_ld_red_min.mlir` | `rc=1`, clean `f32`-only reduction diagnostic | Clean non-`f32` `ld.red` boundary |
| `f64_ld_red_min.mlir` | `rc=1`, clean `f32`-only reduction diagnostic | Clean non-`f32` `ld.red` boundary |
| `i64_copy_shared_to_tmem.mlir` | `rc=0` | Green 64-bit copy contrast |
| `f64_copy_shared_to_tmem.mlir` | `rc=0` | Green 64-bit copy contrast |

This revalidates the current owner split: encoded `i64`/`f64`
non-reduction TMEM load/store descriptor views still map to existing
`FZ-20260421-0017`, while non-`f32` `ld.red` reports a clean unsupported
diagnostic and pure 64-bit `ttng.tmem_copy` lowers successfully.

## Temporary Python Subprocess Contrast

Artifact: `/tmp/tmem_round41_python_probe_summary.txt`.

I also reran a small subset of the existing Round 30 Python child harness in
subprocesses with stable per-GPU caches:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python:./python/test/gluon \
  timeout 180 python /tmp/tmem_fz0017_round30_python_child.py \
  --mode <mode> --dtype <dtype> --layout linear --m <M> --n <N> [--view] [--two-ctas]
```

Results:

| Dtype / shape | Mode | Result | Classification |
| --- | --- | --- | --- |
| `int64 128x128` descriptor view | roundtrip | `rc=134`, `bitwidth == 32` assertion | Existing `FZ-20260421-0017` |
| `float64 128x128` descriptor view | roundtrip | `rc=134`, `bitwidth == 32` assertion | Existing `FZ-20260421-0017` |
| `int64 256x64` 2CTA direct | roundtrip | `rc=134`, `bitwidth == 32` assertion | Existing `FZ-20260421-0017` |
| `float64 256x64` 2CTA direct | load-only | `rc=134`, `bitwidth == 32` assertion | Existing `FZ-20260421-0017` |
| `int32`/`float32` controls | direct/view | Python harness frontend/layout errors | Harness limitation; not classified as backend signal |

The useful part of this subprocess contrast is only the 64-bit reproduction:
it matches existing `FZ-20260421-0017`. I do not treat the `i32`/`f32` control
failures as fuzz findings because the same control surfaces are covered by the
checked-in selector and by the compiler-only controls above.

## Classification

No new independent `FZ-*` bucket is proposed.

Observed classifications:

| Class | Count / evidence |
| --- | ---: |
| Checked-in pass | `385` |
| Checked-in stable skip | `3` |
| Existing `FZ-20260421-0017` | `4` compiler-only 64-bit descriptor-view rows, plus `4` Python subprocess 64-bit rows |
| Clean non-`f32` `ld.red` diagnostic | `2` compiler-only rows, plus checked-in non-`f32` `ld.red` software/diagnostic rows passed |
| Green 64-bit copy contrast | `2` compiler-only rows |
| Green 32-bit/f32 descriptor-view controls | `4` compiler-only rows |
| New compiler crash | `0` |
| New false unsupported diagnostic | `0` |
| New clean-boundary drift | `0` |
| New opcode absence | `0` |
| New runtime miscompile | `0` |

Known-bucket overlap:

- `FZ-20260421-0017`: reproduced for encoded `i64`/`f64` descriptor-view
  non-reduction load/store. This is not new; it is the same `bitwidth == 32`
  lowering assertion already cataloged for 64-bit TMEM load/store.
- `FZ-20260421-0016`: no new unencoded-tensor verifier crash was exercised or
  reproduced by this lane.
- `FZ-20260421-0020`, `FZ-20260421-0021`, `FZ-20260421-0022`: no half-view
  optimizer/dimension/row-reversed `ld.red` signal appeared in the selected
  checked-in rows or compiler contrasts.

Final result: no new candidate bucket. Backend repair remains out of scope for
this discovery-only lane.
