# Round 9 Lane F: clean-diagnostic adversarial fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only. No backend/compiler repairs were attempted.
- Temporary probe: `/tmp/tmem_clean_diagnostics_round9_probe.py`
- Main run log: `/tmp/tmem_clean_diagnostics_round9_run_rerun.log`
- Copy-boundary log: `/tmp/tmem_clean_diagnostics_round9_copy_boundaries.log`
- GPU/cache: `CUDA_VISIBLE_DEVICES=3`,
  `TRITON_CACHE_DIR=/tmp/triton-cache-round9-lane-f-rerun` for the main
  probe and `/tmp/triton-cache-round9-lane-f-copy` for exact copy-boundary
  checks.

## Scope

This lane intentionally targeted TMEM boundaries that should fail cleanly, or
should be positive if the backend can legally realize them:

- `tcgen05.ld.red` `.x1` reductions;
- dynamic `memdesc_index`;
- two-CTA TMEM layouts launched from 4-, 8-, and 16-CTA contexts;
- scales descriptor views;
- 128/256/512-row resource boundaries for direct indexed `ld/st` and
  `ld.red`; and
- no-scales `warpx2` packed/subword and impossible schedule boundaries using
  exact runtime-matrix nodeids.

The probe isolates each generated row in a child Python process so allocator
assertions and LLVM/pass aborts do not terminate the sweep.

## Commands

Required rebuild before tests:

```bash
make -j8
```

Result: `ninja: no work to do`.

Temporary probe syntax check:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_clean_diagnostics_round9_probe.py
```

Result: passed.

Case inventory:

```bash
PYTHONPATH=.:./python:./python/test/gluon python /tmp/tmem_clean_diagnostics_round9_probe.py --list
```

Result: 14 cases.

Main subprocess-isolated sweep:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-round9-lane-f-rerun \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_clean_diagnostics_round9_probe.py \
  2>&1 | tee /tmp/tmem_clean_diagnostics_round9_run_rerun.log
```

Result summary:

```text
SUMMARY {"allocator assertion (FZ-20260421-0005/0009 family)": 4, "clean diagnostic": 8, "late illegal op (FZ-20260421-0001 family)": 1, "pass": 1}
```

Packed/subword and impossible copy schedule checks:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-round9-lane-f-copy \
  PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_subword_dtypes_report_clean_error[single_01_23-1-tmem_layout0-f16-torch_dtype0]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_subword_dtypes_report_clean_error[twocta_02_13-2-tmem_layout11-f16-torch_dtype11]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_candidate_reports_clean_unsupported[f32-torch_dtype0]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_row_permuted_destination_reports_clean_unsupported' \
  2>&1 | tee /tmp/tmem_clean_diagnostics_round9_copy_boundaries.log
```

Result: `4 passed in 3.25s`.

## Classification

### Clean diagnostics

Eight generated rows failed cleanly:

- `ldred-x1-1cta-clean` and `ldred-x1-2cta-clean` both reported that the
  reduction selected a scalar `tcgen05.ld.red` message and that
  `tcgen05.ld.red` requires at least an `.x2` message shape. The diagnostic
  includes the source location at `view.load_min()` and suggests a
  reduction-compatible register layout or explicit software reduction.
- `twocta-layout-in-{4,8,16}cta-context` all rejected the mismatch at
  frontend layout conversion with `Layout has 2 CTAs per CGA, but the context
  requires {4,8,16} CTAs per CGA.`
- `scale-copy-view-1cta` and `scale-copy-view-2cta` rejected
  `TensorMemoryScalesLayout` multibuffering with the explicit diagnostic
  `Scales don't currently support multibuffering`.
- `ldst-resource-128x256` reported a clean tensor-memory
  `OutOfResources`, with required TMEM size `1024` and hardware limit `512`.

The exact copy-boundary checks were also clean:

- single-CTA `warpx2::01_23` f16 packed/subword copy;
- two-CTA `warpx2::02_13` f16 packed/subword copy;
- two-CTA `warpx2::02_13` no-scales candidate; and
- row-permuted `warpx2` destination.

Those rows passed their existing assertions that diagnostics are precise and
do not fall through to `PassManager::run failed` or assertions.

### Positive control

`scale-copy-direct-control` compiled and executed successfully. This confirms
the scales-copy harness itself is viable; the scales-view rows are failing at
the intended descriptor-view/multibuffering boundary rather than because the
direct copy setup is broken.

### Known late illegal op

`dynamic-memdesc-index` still reaches LLVM conversion as illegal
`ttg.memdesc_index`:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

This is an overlap with `FZ-20260421-0001`, not a new non-overlapping finding.
The reproduced pipeline failure is `ConvertTritonGPUToLLVM`.

### Allocator assertion family

Four generated rows assert in `TritonTensorMemoryAllocationPass`:

- `ldst-resource-256x32`
- `ldst-resource-512x32`
- `ldred-resource-256x32`
- `ldred-resource-512x32`

All four report:

```text
TensorMemoryAllocation.cpp:65: ... MemoryBitMap::findFirstFit(...):
Assertion `kNumRows - numRows >= 0' failed.
```

These extend the existing allocator bug family rather than introducing a new
independent bucket. The `256x32` direct indexed `ld.red` row is the same shape
as `FZ-20260421-0009`; the `ld/st` and `512x32` rows reinforce that the clean
resource boundary is not consistently reached for direct indexed parents.

## New FZ Decision

No new non-overlapping `FZ-*` id is recommended from this lane.

The actionable coverage is:

- keep `FZ-20260421-0001` for runtime `memdesc_index` reaching LLVM as an
  illegal op;
- keep `FZ-20260421-0005` / `FZ-20260421-0009` for allocator assertions on
  256-row and larger direct indexed parent views; and
- treat the `.x1 ld.red`, >2-CTA context, scales multibuffering, packed
  subword copy, impossible `warpx2::02_13` two-CTA schedule, row-permuted
  copy, and 128x256 TMEM OOR rows as clean boundaries unless future fuzzing
  finds inconsistent behavior.

Backend repair remains deferred until the fuzzing campaign stops finding new
bugs or the user explicitly pivots from discovery to repair.
