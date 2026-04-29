# SPUD D64 Gluon BMM1 Handoff — 2026-04-29

## Current Checkpoint

- Worktree used for the tuning run: `/root/code/triton-spud-opt`.
- Clean push worktree used for this commit: `/tmp/triton-spud-push`.
- Remote target: `jeffniu-openai/jeffniu/mm2`.
- Base remote commit before this checkpoint: `aacaab9195c0aeff368194973179f211d5ba9573`.
- Source checkpoint before this commit was detached at `6f9f2ca0c86fd68343b6b48023b9b48cf048469b` with local untracked profiling dumps left out of the commit.

## What Changed

- `python/examples/gluon/05-moe-bmm1-fused-gather.py`
  - Preserves the SPUD D64 tuned Gluon BMM1 kernel state, including prior exact-numerics inline PTX / `f32x2` SWIGLU epilogue work.
  - Adds the BM256/BN256 large-slice override for SPUD slice `1040`: `BLOCK_M=256`, `BLOCK_N=256`, `NUM_CTAS=2`, `BAND_N=19`, `X_NUM_BUFS=6`, `W_NUM_BUFS=6`, `INLINE_MMA_RELEASE=True`, `FORCE_EPILOGUE_WARPS_N1=True`.
- `python/triton_kernels/triton_kernels/tensor_details/ragged_tensor.py`
  - Extends ragged tensor metadata block sizes through `2**8`, enabling `BLOCK_M=256`.
- `python/triton_kernels/triton_kernels/matmul_details/opt_flags.py`
  - Keeps explicit `block_n` constraints from being swapped away by the MXFP4 block swap heuristic, preserving forced `BLOCK_N=256` reference comparisons.

## BM256/BN256 Experiment Summary

Three SPUD D64 batch sizes were tested with `BLOCK_M=256`, `BLOCK_N=256`, `BLOCK_K=128`, `NUM_CTAS=2`, using `do_bench_cudagraph` and same-GPU finalist reruns:

| Batch | Slice | Best BM256/BN256 Result | Decision |
|---:|---:|---:|---|
| 1024 | 16 | `0.7599x` vs current BM128/BN512 | Do not use |
| 8192 | 128 | `0.8631x` vs current BM128/BN512 | Do not use |
| 66560 | 1040 | `1.0092x` vs current BM128/BN512 | Use for slice `1040` |

The large-slice refined finalist on GPU0 measured `6.852002 ms` / `4097.0 TFLOPS` versus baseline `6.914856 ms` / `4059.7 TFLOPS` with `rep=500`, `repeats=7`.

## Validation

Run in `/root/code/triton-spud-opt` before this handoff commit:

```bash
make -j128
PYTHONPATH=$PWD/python:$PWD/python/triton_kernels CUDA_VISIBLE_DEVICES=0 \
  python3 -m pytest -s --tb=short python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op
```

Result: `77 passed in 11.33s`.

## Local Artifacts Not Committed

The detailed tuner, CSVs, logs, and full experiment summary are in the rescheduled machine's local path:

```text
/root/code/.codex/initiatives/artifacts/bm256-bn256-2026-04-28/
```

Important files there:

- `summary.md`
- `bm256_bn256_tune.py`
- `screen*_gpu*.csv`
- `final*_gpu0.csv`
- `refine_large_gpu3.csv`
- `pytest_05_after_bm256_bn256_1040_gpu0.log`

## Next Steps

- Rehydrate from this commit and rerun the targeted 05 pytest before further tuning.
- If continuing BM256/BN256 work, only pursue large-slice neighborhoods; small and medium were consistently slower.
- Keep core SWIGLU numerics equivalent to `triton_kernels.matmul`; only MMA shape/accumulation/reduction-order changes are acceptable numerical drift sources.
- Avoid committing temporary `tmp_spud_*_dump/` directories from `/root/code/triton-spud-opt`.
