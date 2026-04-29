# Handoff: SPUD Gluon MM2 Fused Scatter

Date: 2026-04-29 UTC
Repository: `/root/code/triton-pr10114`
Branch: `codex/spud-gluon-bmm1`
Remote: `jeffniu-openai` (`https://github.com/jeffniu-openai/triton`)

## Current HEAD

- Latest implementation commit before this handoff: `59979e4b1d` (`Optimize low-batch MM2 lock and store epilogue`).
- This handoff file should be committed on top of that commit and pushed to `jeffniu-openai/codex/spud-gluon-bmm1`.

## User Constraints / Invariants

- Do not change the `matmul` API or the information passed from `prepare_case` into the actual kernel call.
- Do not change fused-comm semantics.
- Do not change how fused-comm closures are called; the Gluon kernel should call `map_dst_coord.fn(..., *map_dst_coord.captured)` and `all_writes_issued.fn(*all_writes_issued.captured)`.
- Gluon comparison target for SPUD uses `BLOCK_N=512`; `triton_kernels.matmul` comparison uses `BLOCK_N=256`.
- Optimize all 29 SPUD batch sizes equally and judge by geomean, while tracking low-batch regressions.
- Core numerics should remain unchanged; MMA accumulation order / FP reduction ordering may change if needed.

## Implemented State

File of interest: `python/examples/gluon/06-fused-scatter-bmm2.py`.

Current committed kernel includes:

- Ragged async TMA activation load for MM2 activations.
- Fused comm closure routing matching `_p_matmul` structure.
- KI/SPO-compatible lock/map-destination helper semantics.
- Low-batch cleanup from `59979e4b1d`:
  - Vectorized Gluon peer-lock notification in `gluon_set_locks` using a non-splitting `cga_layout=((0,),)` vector layout.
  - Fully-invalid epilogue row fragments are skipped before packing/storing.

## Validation Already Run

Most recent correctness/build gate before handoff:

```text
make -j128
pytest -s --tb=short python/examples/gluon/06-fused-scatter-bmm2.py::test_gluon_matmul_fused_scatter
```

Result: `4 passed`.

## Performance State

Main all-29 benchmark after `59979e4b1d`:

- Command source: `/tmp/mm2_all29_perf_report.py`
- CSV: `/tmp/spud_mm2_all29_rep300_tflops_tbps.csv`
- Corrected launch-metadata-style CSV: `/tmp/spud_mm2_all29_rep300_metadata_tflops_tbps.csv`
- Setup: Gluon `BLOCK_N=512` vs `triton_kernels.matmul` `BLOCK_N=256`, `rep=300`, all 29 SPUD batch sizes.
- Geomean speedup: `1.171061x`
- Min speedup: `1.019863x`

Low-batch `rep=1000` after final cleanup:

- CSV: `/tmp/spud_mm2_lowbatch_rep1000_final.csv`
- `1024`: `1.0196x`
- `2048`: `1.0302x`
- `3072`: `1.0258x`
- `4096`: `1.0721x`
- Geomean over these four: about `1.0367x`

Corrected metadata-style TB/s note:

- Use `python/triton_kernels/triton_kernels/matmul_details/_common.py::matmul_launch_metadata` semantics for bytes.
- Bytes are `X_bytes + Y_bytes + W_bytes`; do not count tiled/reloaded traffic when reporting `tbps` in the matmul metadata style.

## Recent Tuning Attempts / Dead Ends

Do not blindly repeat these without a new hypothesis:

- Broad 4-GPU selector sweep over `BAND_N`, `STORE_HELPER_REGS`, `LOAD_ACTIVATION_WARPS`, buffer counts, and multicast flags did not find a stable all-29 win.
- Rep=300 finalist rerun CSV: `/tmp/mm2_all29_finalist_rerun_rep300.csv`.
- Best finalists vs current kernel were still below or near parity:
  - `all_store36`: geo `0.999884`, min `0.986600`, max `1.004722`
  - `all_store16`: geo `0.999634`, min `0.985380`, max `1.005824`
  - `all_band16`: geo `0.999599`, min `0.986829`, max `1.005210`
- K=256 variants required descriptor swizzle changes for fp8 activations and broadly regressed after measurement.
- Lower launch-grid occupancy was a large regression.
- Extra weight buffering by reducing activation buffers regressed due shared-memory pressure.
- Inline MMA input release was mixed and was not kept because it would add internal kernel constexpr plumbing under the no-kernel-call-information-change constraint.
- EP1 epilogue path did not materially help 1024 and created OOR risks for some small cases.

## Useful Commands

```bash
# Build before tests per repo instructions
make -j128

# Correctness gate
pytest -s --tb=short python/examples/gluon/06-fused-scatter-bmm2.py::test_gluon_matmul_fused_scatter

# Verify current remote branch
 git ls-remote jeffniu-openai refs/heads/codex/spud-gluon-bmm1
```

## Suggested Next Step

The current selector appears locally optimized for simple knob changes. To improve further, focus on a structural bottleneck rather than another broad config sweep:

1. Profile `bs=1024` and `bs=2048` Gluon vs Triton matmul with NCU.
2. Inspect SASS for barrier/long-scoreboard differences in the TMA/MMA pipeline.
3. Look for a way to reduce low-batch pipeline/barrier overhead without changing the `matmul` API or fused-comm call contract.
4. Keep using all-29 geomean and low-batch minimum speedup as promotion gates.
