# Example 01/05 SASS Performance Diagnosis - 2026-04-24

## Scope

User requested diagnosis of the `01-attention-forward.py` and `05-moe-bmm1-fused-gather.py` performance regressions by looking at SASS/codegen differences. This note follows the 2026-04-24 benchmark report in `example_benchmark_20260424.md`.

Comparison points:

- Current branch source + branch compiler: `/root/code/triton` at documentation HEAD `b2512dc92` (source benchmarked at `9bee5eb94`; `b2512dc92` is doc-only).
- Upstream main source + main compiler: `/tmp/triton-main-bench` at `27c402843`.
- Upstream main source + branch compiler: same upstream example source imported from `/tmp/triton-main-bench`, but with `PYTHONPATH=/root/code/triton/python`.

Artifacts:

- Capture helper: `/tmp/capture_gluon_artifacts.py`.
- Direct attention timing helper: `/tmp/bench_attention_direct.py`.
- Focused MoE timing helper: `/tmp/bench_moe_one.py`.
- TTGIR/PTX/SASS/metadata outputs: `/tmp/tmem_codegen_diag_20260424/{branch,main,branch_main_source}/`.

## 01 Attention Forward

Representative regressed row: D64 causal fp8 `N_CTX=1024`. Full benchmark ratios from `example_benchmark_20260424.md` were `0.563x` for no-red and `0.582x` for red.

### Benchmark Harness Difference

The branch copy of the example is not equivalent to upstream main:

- Upstream main preallocates `o`/`M` for the benchmark and uses `do_bench_cudagraph`.
- The branch copy allocates `o`/`M` inside `attention_forward` and uses `do_bench`.

A focused direct-launch CUDAGraph timing with preallocated outputs reduces the apparent small-N loss substantially:

| source/compiler | red arg | direct TFLOPS | ratio vs main-source/main-compiler |
|---|---:|---:|---:|
| branch source / branch compiler | false | 437.34 | 0.871x |
| branch source / branch compiler | true | 448.98 | 0.894x |
| main source / main compiler | false | 502.00 | 1.000x |
| main source / main compiler | true arg, selector disables red | 502.03 | 1.000x |
| main source / branch compiler | false | 431.44 | 0.859x |

Conclusion: the extreme `~0.56x` full-benchmark loss is mostly benchmark/source-harness drift, but an actual branch compiler/codegen loss remains for the upstream main source.

### Source/Configuration Difference

Upstream main has a `KernelConfig` selector that the branch copy no longer has. For this row main selects:

```text
KernelConfig(BLOCK_M=256, BLOCK_N=128, GROUP_SIZE_N=8, SPLIT_EXP_FACTOR=4,
             NUM_WARPS=4, MAXNREG=128, OCCUPANCY=1, USE_TMEM_RED=False,
             NUM_KV_BUFFERS=2, USE_EXP2_TURNSTILE=True)
```

The branch copy hardcodes `GROUP_SIZE_N=4` for causal, derives `NUM_KV_BUFFERS=8` for fp8 D64, and passes through the user `use_tmem_red` flag. Metadata for the current branch source shows `shared=99173`, while upstream main source shows `shared=49921/49925`. The branch red benchmark is also not equivalent to main: upstream main deliberately disables the red path for causal fp8, while the branch source emits explicit red loads.

### SASS: Current Branch Source vs Main Source

Current branch source / branch compiler versus upstream main source / main compiler for no-red D64 causal fp8 `N_CTX=1024`:

| metric | branch source/branch compiler | main source/main compiler | delta |
|---|---:|---:|---:|
| SASS instructions | 4680 | 4648 | +32 |
| `UTCQMMA*` | 24 | 36 | -12 |
| `BAR*` | 106 | 94 | +12 |
| `SYNCS*` | 239 | 221 | +18 |
| `LDL*` | 55 | 78 | -23 |
| `STL*` | 43 | 66 | -23 |
| reported spills | 14 | 28 | -14 |

The faster main kernel has more reported spills, so the dominant loss for the current branch-source comparison is not spill count. The larger branch shared footprint, more barriers/syncs, different grouping/buffering, and different QK/O MMA schedule are the source-level explanation.

### SASS: Same Upstream Main Source, Branch Compiler vs Main Compiler

Using the exact upstream main attention source with the branch compiler exposes a real compiler/codegen regression:

| metric | main source/branch compiler | main source/main compiler | delta |
|---|---:|---:|---:|
| direct CUDAGraph TFLOPS | 431.44 | 502.00 | 0.859x |
| SASS instructions | 5304 | 4648 | +656 |
| `LDTM*` | 54 | 22 | +32 |
| `STTM*` | 44 | 28 | +16 |
| PTX `tcgen05.ld.sync...x1` | 38 | 6 | +32 |
| PTX `tcgen05.st.sync...x1` | 24 | 8 | +16 |
| `LOP3.LUT` | 412 | 268 | +144 |
| `SHF.L.U32` | 195 | 67 | +128 |
| `SHF.R.U32.HI` | 129 | 1 | +128 |
| `SEL` | 128 | 0 | +128 |

The extra instructions occur at upstream main `01-attention-forward.py` line 555, inside `_compute_and_store_exp2`, where fp8 probabilities are stored to TMEM. Main compiler emits a direct vector TMEM store:

```ptx
@%p200 tcgen05.st.sync.aligned.32x32b.x8.b32 [%r1674 + 0], {%r1935, ...};
```

Branch compiler emits read/modify/write packing around the same store:

```ptx
tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r2269}, [%r3063 + 0];
tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r2270}, [%r3063 + 8];
... and/shl/or/selp chain ...
@%p201 tcgen05.st.sync.aligned.32x32b.x8.b32 [%r3063 + 0], {%r2271, ...};
@%p205 tcgen05.st.sync.aligned.32x32b.x1.b32 [%r3063 + 8], {%r2279};
```

Interpretation: the branch compiler is over-pessimistic for packed fp8 TMEM stores through the upstream main `_reinterpret` view shape. It treats stores as potentially subword-shifted and emits scalar TMEM RMW plus bit packing, even though the concrete offsets are aligned enough for the direct `x8` store. The branch's current source was rewritten to explicit `bitcast`/physical-column slicing and avoids this particular RMW pattern, but it also lost upstream main's tuning and benchmark harness.

## 05 MoE BMM1 Fused Gather

Representative regressed row: batch `4096`, where the full benchmark showed branch `315.29` TFLOPS versus main `406.83` TFLOPS (`0.775x`).

### Source/Algorithm Difference

This regression is not a same-source compiler regression. The branch copy of example 05 is older/different than upstream main:

- Branch source: `BLOCK_N=256`, `NUM_CTAS=1`, no TMA multicast, helper-store epilogue path.
- Upstream main source: `BLOCK_N=512`, `NUM_CTAS=2`, 2-CTA barriers, multicast TMA for activation gather / scales where enabled, and a different direct/CGA-aware store path.

Current branch source / branch compiler metadata for batch 4096:

```text
KernelConfig(BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, X_NUM_BUFS=5, W_NUM_BUFS=4,
             ACC_NUM_BUFS=1, NUM_WARPS=8, ... EPILOGUE_BUFFER_DEPTH=2, BAND_N=18)
num_ctas=1, shared=225508, n_spills=0
```

Upstream main source / main compiler metadata for the same batch:

```text
KernelConfig(BLOCK_M=128, BLOCK_N=512, BLOCK_K=128, NUM_CTAS=2, X_NUM_BUFS=5, W_NUM_BUFS=5,
             ACC_NUM_BUFS=1, NUM_WARPS=8, ... X_GATHER_MULTICAST=True, W_SCALE_MULTICAST=True)
num_ctas=2, shared=214240, n_spills=0
```

### SASS Evidence

| metric | branch source/branch compiler | main source/main compiler | delta |
|---|---:|---:|---:|
| SASS instructions | 3976 | 2904 | +1072 |
| `UTCQMMA*` | 16 | 8 | +8 |
| `UTCQMMA.2CTA*` | 0 | 8 | -8 |
| `UTMALDG*` | 10 | 6 | +4 |
| `UTMALDG*.2CTA` | 0 | 6 | -6 |
| `STG*` | 64 | 16 | +48 |
| `SYNCS*` | 91 | 49 | +42 |
| `LDG*` | 44 | 27 | +17 |
| reported spills | 0 | 0 | 0 |

PTX count summary:

| PTX op family | branch source/branch compiler | main source/main compiler |
|---|---:|---:|
| `tcgen05.mma.cta_group::1` | 16 | 0 |
| `tcgen05.mma.cta_group::2` | 0 | 8 |
| `cp.async.bulk.tensor` | 10 | 6 |
| `multicast` operands | 0 | 3 |
| `tcgen05.cp` | 4 | 2 |

Same upstream main source compiled with the branch compiler is effectively equivalent to main for this row:

| metric | main source/branch compiler | main source/main compiler |
|---|---:|---:|
| focused example TFLOPS | 411.73 | 406.83 full-run row |
| SASS instructions | 2920 | 2904 |
| `UTCQMMA.2CTA*` | 8 | 8 |
| `UTMALDG*.2CTA` | 6 | 6 |
| `STG*` | 16 | 16 |
| `SYNCS*` | 49 | 49 |

Conclusion: example 05's performance regression is source drift, not a TMEM backend codegen regression. Restoring/porting upstream main's 2-CTA/multicast implementation onto the branch should recover the performance shape.

## Diagnosis Summary

- 01 has two overlapping causes:
  - the branch example dropped upstream main's benchmark harness and tuning selector, creating the very large full-benchmark loss;
  - for upstream main's source, the branch compiler still emits unnecessary packed-fp8 TMEM scalar RMW around aligned stores, costing about `0.86x` on the focused direct kernel row.
- 05 is almost entirely source drift: the branch example is a different 1-CTA, narrower-N kernel. The upstream main 2-CTA source compiled by the branch compiler produces matching SASS and comparable or slightly better focused performance.

## Recommended Next Steps

1. Restore/port upstream main's example 05 implementation to the branch, preserving branch API changes only where required. This should recover the `BLOCK_N=512`, `NUM_CTAS=2`, multicast, and direct/CGA-aware store SASS shape.
2. Restore/port upstream main's attention benchmark harness and `KernelConfig` selector, then adapt it to the branch's current TMEM view/bitcast API without reintroducing illegal view-chain dependencies.
3. Independently fix the branch compiler's same-source attention codegen regression: for packed fp8 TMEM stores where the current type/layout/taddr prove hardware-word alignment, emit the direct vector `tcgen05.st` and DCE the subword RMW path. The repro is upstream main `01-attention-forward.py`, D64 causal fp8 `N_CTX=1024`, compiled under branch PYTHONPATH.
