# TMEM Structural Fuzzing Lane R5-D: Clean-Negative Descriptor-View Boundaries

- Date: 2026-04-21
- Branch: `codex/tmem`
- HEAD: `729dde3d1`
- Mode: discovery/classification only; no backend/compiler repairs attempted.
- Scope: adversarial clean-negative boundaries for TMEM descriptor views:
  small rows, row-anchor/resource-frontier behavior, `M=512`/TMEM-resource
  boundaries, unsupported 2CTA chained `warpx2::02_13` views, and
  subword/f8/f16 boundaries.
- Temporary probe: `/tmp/tmem_clean_negative_boundaries_r5d.py`

## Commands

Required rebuild before pytest:

```bash
make -j8
```

Result: `ninja: no work to do`.

Collection:

```bash
PYTHONPATH=.:./python:python/test/gluon pytest --collect-only -q /tmp/tmem_clean_negative_boundaries_r5d.py
```

Result: `13 tests collected`.

Four-GPU clean-negative probe:

```bash
PYTHONPATH=.:./python:python/test/gluon CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r5d-final-gpu0 pytest -q -s --tb=short --splits 4 --group 1 /tmp/tmem_clean_negative_boundaries_r5d.py 2>&1 | tee /tmp/tmem_clean_negative_boundaries_r5d_final_g1.log
PYTHONPATH=.:./python:python/test/gluon CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r5d-final-gpu1 pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_clean_negative_boundaries_r5d.py 2>&1 | tee /tmp/tmem_clean_negative_boundaries_r5d_final_g2.log
PYTHONPATH=.:./python:python/test/gluon CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r5d-final-gpu2 pytest -q -s --tb=short --splits 4 --group 3 /tmp/tmem_clean_negative_boundaries_r5d.py 2>&1 | tee /tmp/tmem_clean_negative_boundaries_r5d_final_g3.log
PYTHONPATH=.:./python:python/test/gluon CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r5d-final-gpu3 pytest -q -s --tb=short --splits 4 --group 4 /tmp/tmem_clean_negative_boundaries_r5d.py 2>&1 | tee /tmp/tmem_clean_negative_boundaries_r5d_final_g4.log
```

Results:

- group 1: `4 passed, 9 deselected`
- group 2: `4 passed, 9 deselected`
- group 3: `4 passed, 9 deselected`
- group 4: `1 passed, 12 deselected`
- aggregate: `13 passed`

F8 positive control:

```bash
PYTHONPATH=.:./python:python/test/gluon CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r5d-f8 pytest -q -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_indexed_acc_view[f8e5m2-linear-128-64-False]' 2>&1 | tee /tmp/tmem_clean_negative_boundaries_r5d_f8_control.log
```

Result: `1 passed`.

## Classification Matrix

| Case | Seed / selector | Command log | Classification | Notes |
| --- | --- | --- | --- | --- |
| `small-row-16x32-chain0` | `0x501d` | `final_g1` | clean unsupported | Direct indexed descriptor view rejects at `get_reg_layout("32x32b")` with `TMEM layout 'constexpr[32x32b]' unsupported for descriptor view`; no assertion, pass-manager crash, or illegal-op leak. |
| `small-row-32x32-chain0` | `0x502d` | `final_g1` | clean unsupported | Same short clean diagnostic as `16x32`; current diagnostic does not include the longer row-anchor explanatory text. |
| `small-row-64x32-chain0` | `0x504d` | `final_g1` | pass boundary control | Direct chain0 now compiles and round-trips with matching `tcgen05.ld/st` PTX/LLIR. This is not a negative on current HEAD; smaller rows remain the clean boundary. |
| `twocta-indexed-copy-256x256-parent-depth2` | fixed matrix geometry | `final_g1` | clean resource boundary | Raises `OutOfResources` for tensor memory with hardware limit `512`; no compiler crash. This is the effective `M=512` live-TMEM resource frontier for lifted 2CTA copy views. |
| `warpx2::02_13-2cta-indexed-view-{f32,i32}` | parent index `1` | `final_g2` | clean unsupported | Both dtypes reject with shared descriptor-plan diagnostic, `cleanly unsupported`, and no crash-shaped text. |
| `warpx2::02_13-2cta-subslice-view-{f32,i32}` | slice start `4` | `final_g2/final_g3` | clean unsupported | Chained subslice view preserves the known source-column requirement diagnostic. |
| `warpx2::02_13-2cta-slice-index-view-{f32,i32}` | parent index `1` | `final_g3` | clean unsupported | Chained slice+index view rejects cleanly; no unexpected pass. |
| `warpx2::01_23-single-f16` | matrix unsupported case | `final_g3` | clean unsupported | Subword f16 rejects through descriptor-plan diagnostic; no stale generic “requires 32-bit shared elements” message. |
| `warpx2::01_23-single-i8` | matrix unsupported case | `final_g4` | clean unsupported | Subword i8 rejects cleanly with descriptor-plan text and no backend crash. |
| `warpx2::02_13-2cta-f16` | matrix unsupported case | `final_g4` | clean unsupported | 2CTA f16 subword boundary rejects cleanly. |
| `mma-twocta-indexed-acc-f8e5m2` | checked-in runtime matrix nodeid | `f8_control` | pass control | F8 two-CTA indexed accumulator view remains a valid positive boundary and passes. |

## Findings

- No new compiler crash, pass-manager failure, assertion, illegal-op leak, or
  unexpected pass was found in this lane.
- The only classification nuance is diagnostic quality: `16x32` and `32x32`
  small-row descriptor views reject cleanly, but with a short unsupported
  message rather than the more explanatory row-anchor diagnostic seen in some
  adjacent probes. I classify this as clean, not too-strict or crash-shaped.
- `64x32` direct chain0 is a positive boundary on current HEAD. It should not
  be counted as a clean negative; chained/read-only variants remain covered by
  existing `FZ-20260421-0003` sentinels.
- The unsupported 2CTA `warpx2::02_13` chained-view diagnostics remain clean
  across indexed, subslice, and slice+index forms for both `f32` and `i32`.
- Subword copy boundaries remain clean for representative `f16` and `i8`
  rows. The f8 MMA indexed-accumulator control still passes, so no f8
  clean-negative regression was observed.
