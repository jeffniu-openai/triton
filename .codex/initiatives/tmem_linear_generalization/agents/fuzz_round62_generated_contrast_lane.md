# Round 62 Lane B: Generated Structural/Runtime Contrast Probes

Date: 2026-04-21

Branch/HEAD: `codex/tmem` at `eb7ab3a07c5a`.

Scope: discovery/cataloging only. No backend, compiler, or checked-in test
source was modified by this lane. Existing unrelated worktree changes were
left untouched.

Required first step:

```bash
make -j8
```

Result: no-op rebuild through
`/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`.

## Temporary Artifacts

- `/tmp/tmem_round62_generated_contrast_lane_struct_dynamic_ldst.log`
- `/tmp/tmem_round62_generated_contrast_lane_green_ldst_ldred.log`
- `/tmp/tmem_round62_generated_contrast_lane_ldred_m64.log`
- `/tmp/tmem_round62_generated_contrast_lane_scaled.log`
- `/tmp/tmem_round62_generated_contrast_lane_scaled_dynamic.log`
- `/tmp/tmem_round62_generated_contrast_lane_twocta_copy.log`
- `/tmp/tmem_round62_ldred_half_views/worker.py`
- `/tmp/tmem_round62_ldred_half_views/summary.json`

## Commands and Results

Structural dynamic-index and ld/st descriptor-view sentinels:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_dynamic_index_load_only[generic-pass-dynamic-index-load-only-128x32]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_descriptor_view_read[ldst-fz20260421-0003-chain1-64x32-32x32b]'
```

Result: `2 xfailed`. The first row reproduced existing
`FZ-20260421-0001` with a late illegal `ttg.memdesc_index` in
`ConvertTritonGPUToLLVM`. The second row reproduced existing
`FZ-20260421-0003` descriptor-view ld/st wrong-result coverage.

Green ld/st and `ld.red` descriptor-view controls:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions[identity-128-auto-32x32b.x128.b32]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions_rowcol_permuted_layout_sweep[f32-torch_dtype1-identity-reverse-128-auto-32x32b.x128.b32]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_descriptor_chain_n_sweep_explicit_variants[row_reverse_n64_32x32b-False-propagate_nan0-min]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_descriptor_chain_n_sweep_explicit_variants[rowcol_rotate_reverse_n64_32x32b-False-propagate_nan0-min]'
```

Result: `4 passed`.

M64 row/column `ld.red` contrast:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]'
```

Result: `2 failed`, both existing `FZ-20260421-0012`-class M64 f32
`ld.red` destination-layout lowering failures. Signature:
`ttng.tmem_load failed to compute TMEM encoding info for reduction` with
`unsupported dst layout`. No new signature.

Scaled-MMAv5 checked-in controls:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_scaled_mma_acc_subslice_control_flow[mma-scaled-fz20260421-0007-subslice-if-n64-selector0]' \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_bscale_descriptor_view \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_n16_bscale_descriptor_view_reports_clean_error[False]'
```

Result: `2 passed, 1 xfailed`. The xfail is existing
`FZ-20260421-0007` for dynamic selected scaled-MMAv5 accumulator subslice.
The B-scale descriptor-view row passed, and the N16 B-scale descriptor-view
row stayed a clean unsupported diagnostic.

Scaled-MMAv5 dynamic scale descriptor-view contrast:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  '/tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_bscale_descriptor_selection[0]' \
  '/tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_ascale_descriptor_selection[0]' \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_scale_descriptor_view_chains
```

Result: `4 failed, 2 passed`. The dynamic B-scale row reproduced existing
`FZ-20260421-0015` with `16381 / 16384` mismatches and NaN greatest
difference. The three failing scale descriptor-view chain rows reproduce the
known scaled descriptor-view wrong-result surface tracked with
`FZ-20260421-0013`; mismatch counts were `16107/16384`, `16116/16384`, and
`16373/16384`. The dynamic A-scale control and the all-false static
descriptor-view chain row passed.

Two-CTA copy/indexed descriptor-view contrast:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_linear_indexed_view[f32-torch_dtype0-256-64-32-8-tcgen05.cp.cta_group::2.128x256b]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_linear_subslice_view[f32-torch_dtype0-256-128-32-16-tcgen05.cp.cta_group::2.128x256b]' \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_noncanonical_block_reports_clean_unsupported
```

Result: `3 passed`. The indexed and subslice two-CTA copy rows emitted the
expected `tcgen05.cp.cta_group::2.128x256b` path; the noncanonical two-CTA
block row stayed a clean unsupported diagnostic.

Half-row / half-column `ld.red` descriptor-view replay:

```bash
PYTHONPATH=.:./python:./python/test/gluon python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("probe", "/root/tmp/tmem_ldred_descriptor_views_round35_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.OUT_DIR = probe.Path("/tmp/tmem_round62_ldred_half_views")
probe.WORKER = probe.OUT_DIR / "worker.py"
probe.SUMMARY = probe.OUT_DIR / "summary.json"
probe.CASES = [c for c in probe.CASES if c["shape_kind"] in ("half_row", "half_col")]
probe.main()
PY
```

Raw summary: `32` cases, with `8 pass`, `8 harness_parse_failure`,
`4 FZ-20260421-0020`, and `12 runtime_or_compile_exception_unclassified`.
Manual signature classification matches the prior Round 57/Round 35
half-view inventory:

- `8 pass`: half-column identity and row-reversed rows.
- `8 clean unsupported`: half-row identity and column-reversed rows reported
  `TMEM layout 'auto' unsupported for descriptor view`.
- `8 existing FZ-20260421-0022`: half-row row-reversed or row+column-reversed
  rows reported `ttng.tmem_load failed to compute TMEM encoding info for
  reduction`.
- `4 existing FZ-20260421-0020`: half-column column-reversed rows failed at
  late `ttng.tmem_load` legalization / LLVM conversion.
- `4 clean scalar .x1`: half-column row+column-reversed rows reported
  `tcgen05.ld.red requires at least an .x2 message shape`.

## Classification

No new independent `FZ-*` candidate was found in this lane.

Observed existing buckets:

- `FZ-20260421-0001`: dynamic runtime TMEM `memdesc_index` survives to LLVM
  conversion as an illegal op.
- `FZ-20260421-0003`: ld/st descriptor-view packet mapping wrong-result
  sentinel.
- `FZ-20260421-0007`: scaled-MMAv5 dynamically selected accumulator subslice.
- `FZ-20260421-0012`: M64 f32 row-basis `ld.red` destination-layout lowering.
- `FZ-20260421-0013`: scaled-MMAv5 scale descriptor-view chain wrong results.
- `FZ-20260421-0015`: dynamic B-scale descriptor selection wrong results.
- `FZ-20260421-0020`: half-column column-reversed `ld.red` descriptor-view
  late illegal `ttng.tmem_load`.
- `FZ-20260421-0022`: row-reversed half-row `ld.red` descriptor-view
  parser/lowering failure.

Clean unsupported boundaries stayed clean for N16 scaled B-scale descriptor
views, two-CTA noncanonical copy, half-row `auto` descriptor views, and scalar
`.x1` `ld.red` message shapes. No repair was attempted.
