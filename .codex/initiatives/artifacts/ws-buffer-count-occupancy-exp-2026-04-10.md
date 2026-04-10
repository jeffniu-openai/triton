## WS Buffer Count / Virtual NUM_SMS Experiment

Date: 2026-04-10
Branch: `jeffniu/kernels`
Base example: [05-moe-bmm1-fused-gather.py](/root/code/triton-ws-opt/python/examples/gluon/05-moe-bmm1-fused-gather.py)

### Question

Can low-batch GPT-OSS MM1 performance improve by lowering load-buffer counts enough to bring per-CTA SMEM and TMEM under half of an SM, then doubling the persistent CTA count (`NUM_SMS`) to target 2 CTAs per SM?

### Method

1. Compiled the live kernel with `ws_matmul_kernel.warmup(...)` and read `kernel.metadata.shared` / `kernel.metadata.tmem_size`.
2. Checked the three active low-batch regimes:
   - `BLOCK_M=16` (`batch <= 287`)
   - `BLOCK_M=32` (`288 <= batch <= 543`)
   - `BLOCK_M=64` (`544 <= batch <= 1887`)
3. Swept reduced `X_NUM_BUFS` / `W_NUM_BUFS` pairs chosen to be close to the current default while still crossing the half-SMEM threshold when possible.
4. For every candidate:
   - validated output against `reference_matmul` with the live example tolerances
   - measured median runtime with `do_bench_cudagraph(..., rep=1000)`
5. For candidates with `shared <= 116224` and `tmem <= 256`, also reran with `launch_grid = min(2 * sms, expected_grid_m * grid_n)` and `NUM_SMS=launch_grid` to emulate a 2x persistent-CTA schedule.

Half-SM thresholds used:

- shared memory: `232448 / 2 = 116224 B`
- tensor memory: `512 / 2 = 256`

### Resource Findings

Representative compiled metadata for the live defaults:

| Batch | Block M | X bufs | W bufs | Shared (B) | TMEM |
|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 5 | 4 | 151776 | 64 |
| 384 | 32 | 5 | 4 | 162016 | 128 |
| 1024 | 64 | 5 | 4 | 184544 | 256 |

Minimal near-default under-half candidates:

| Block M | Candidate | Shared (B) | TMEM | Under half? |
|---:|---|---:|---:|---|
| 16 | `x=4, w=3` | 115904 | 64 | yes |
| 32 | `x=2, w=3` | 115872 | 128 | yes |
| 64 | `x=4, w=2` | 108704 | 256 | yes |

### Performance Findings

The reduced-buffer candidates were all slower than the live default, and the virtual 2x-CTA schedule made them slower again.

Representative results:

| Batch | Config | Virtual CTA factor | Median ms | Delta vs default |
|---:|---|---:|---:|---:|
| 128 | default `x=5,w=4` | `1x` | 0.03432 | baseline |
| 128 | `x=4,w=3` | `1x` | 0.03671 | +6.9% |
| 128 | `x=4,w=3` | `2x` | 0.03955 | +15.2% |
| 256 | default `x=5,w=4` | `1x` | 0.03485 | baseline |
| 256 | `x=4,w=3` | `1x` | 0.03742 | +7.4% |
| 256 | `x=4,w=3` | `2x` | 0.04020 | +15.4% |
| 384 | default `x=5,w=4` | `1x` | 0.03586 | baseline |
| 384 | `x=2,w=3` | `1x` | 0.04326 | +20.7% |
| 384 | `x=2,w=3` | `2x` | 0.04628 | +29.1% |
| 512 | default `x=5,w=4` | `1x` | 0.03665 | baseline |
| 512 | `x=2,w=3` | `1x` | 0.04462 | +21.8% |
| 512 | `x=2,w=3` | `2x` | 0.04850 | +32.3% |
| 1024 | default `x=5,w=4` | `1x` | 0.04041 | baseline |
| 1024 | `x=4,w=2` | `1x` | 0.05507 | +36.3% |
| 1024 | `x=4,w=2` | `2x` | 0.05952 | +47.3% |
| 1536 | default `x=5,w=4` | `1x` | 0.04285 | baseline |
| 1536 | `x=4,w=2` | `1x` | 0.05742 | +34.0% |
| 1536 | `x=4,w=2` | `2x` | 0.06151 | +43.6% |

Additional candidate notes:

- `BLOCK_M=16`, `x=5,w=2` was much worse than the milder `x=4,w=3` variant: `0.04807 ms` at `batch=128`.
- `BLOCK_M=32`, weight-buffer reductions to `w=2` were all materially worse than `x=2,w=3`.
- `BLOCK_M=64`, even the not-under-half `x=2,w=3` candidate still regressed badly (`0.05006 ms` at `batch=1024` vs `0.04041 ms` default).

### Interpretation

The current low-batch WS kernel is not overbuffered enough for this trade to work.

- The default pipelines need the extra activation and weight/scale buffering more than they need additional persistent CTAs.
- Crossing the half-SMEM line by itself is not enough; the producer-side latency hiding lost from reducing buffers is larger than any gain from extra CTA residency.
- The failed `2x` virtual-CTA reruns show that even when the resource thresholds make 2 CTAs per SM possible on paper, this kernel gets slower once the buffering is stripped down enough to reach that point.

### Conclusion

Do not lower load-buffer counts in the live example to chase 2x persistent CTA residency.

For this kernel, the better policy remains:

- keep the current `X_NUM_BUFS=5`, `W_NUM_BUFS=4`
- keep the existing low-batch `BLOCK_M` ladder
- look elsewhere for more low-batch speedup

The next credible resource-side idea would need to cut SMEM without starving the producer pipeline, which likely means changing staging topology rather than simply deleting load buffers.
