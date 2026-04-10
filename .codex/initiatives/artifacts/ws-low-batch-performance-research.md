# WS Low-Batch Performance Research

## Goal

Understand why the standalone WS example in
[05-moe-bmm1-fused-gather.py](/root/code/triton-ws-opt/python/examples/gluon/05-moe-bmm1-fused-gather.py)
falls behind the production reference at low GPT-OSS 120B MM1 batch sizes, and determine whether
split-K is actually the first implementation priority.

## Workload And Method

- Model geometry: GPT-OSS 120B MoE MM1
  - `num_experts = 128`
  - `experts_per_token = 4`
  - `hidden_size = 2880`
  - `intermediate_size = 2880`
  - MM1 matmul shape before SwiGLU: `2880 x 5760`
- Kernel under study:
  - example: `python/examples/gluon/05-moe-bmm1-fused-gather.py`
  - reference: `triton_kernels.matmul`
- Device: `cuda:0` on this GB300 host
- Prepared inputs:
  - `seed = 0`
  - same routing pattern and scales for example/reference per batch
- Timing method:
  - `triton.testing.do_bench_cudagraph`
  - main sweep: `rep = 1000`
  - confirmation compare: `rep = 3000`
- Profiling:
  - `ncu --set full` on `batch = 128`, `512`, and `1024`
- Reference heuristic inspection:
  - `triton_kernels.matmul_details.opt_flags.make_opt_flags(...)`

Canonical external references used for context:

- GPT-OSS 120B config:
  - `https://huggingface.co/openai/gpt-oss-120b/blob/main/original/config.json`
- CUTLASS split-K / efficient GEMM:
  - `https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html`
- PyTorch persistent grouped GEMM blog:
  - `https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/`
- TensorRT-LLM expert-parallel scheduling notes:
  - `https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog08_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.html`

Local download attempts with `curl -L` were blocked on this host by outbound TLS failures
(`SSL_ERROR_SYSCALL`), so this note keeps canonical URLs instead of vendored page snapshots.

## Baseline Gap

Current live example vs reference, same prepared case, `rep = 1000`:

| Batch | Example ms | Reference ms | Example TFLOP/s | Reference TFLOP/s | Example / Reference |
|---|---:|---:|---:|---:|---:|
| 128 | 0.04953 | 0.03477 | 44.21 | 62.98 | 0.702x |
| 256 | 0.04578 | 0.03562 | 97.12 | 124.81 | 0.778x |
| 512 | 0.04926 | 0.03731 | 171.08 | 225.85 | 0.757x |
| 1024 | 0.04696 | 0.04090 | 365.93 | 420.18 | 0.871x |
| 2048 | 0.04982 | 0.05116 | 691.31 | 673.21 | 1.027x |
| 4096 | 0.06594 | 0.06748 | 1056.66 | 1032.44 | 1.023x |
| 8192 | 0.10138 | 0.10365 | 1377.73 | 1347.60 | 1.022x |
| 16384 | 0.16051 | 0.16717 | 1732.96 | 1663.94 | 1.041x |

So the current example is only materially behind at `128..1024`.

## What The Reference Is Actually Doing

The production reference does **not** use split-K on this GPT-OSS MM1 workload. The chosen low-batch
configs are:

| Batch | Reference `block_m` | `block_n` | `block_k` | `split_k` | `num_warps` | `num_stages` |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 256 | 128 | 1 | 4 | 4 |
| 256 | 16 | 256 | 128 | 1 | 4 | 4 |
| 512 | 32 | 256 | 128 | 1 | 4 | 4 |
| 1024 | 64 | 256 | 128 | 1 | 4 | 4 |
| 2048 | 64 | 256 | 128 | 1 | 4 | 4 |
| 4096+ | 128 | 256 | 128 | 1 | 8 | 4 |

This changes the priority order materially: the reference wins low batch without split-K, so
split-K is not the first missing feature for GPT-OSS MM1.

## Corrected `block_m` Ablation

The first quick sweep accidentally failed to override the dataclass field in `KernelConfig`; the
corrected sweep replaces `KernelConfig()` with `dataclasses.replace(orig(), block_m=...)`.

Corrected same-process example-only sweep, `rep = 1000`:

| Batch | `block_m` | ms | TFLOP/s | `grid_m` | Avg Valid Rows / Block |
|---|---:|---:|---:|---:|---:|
| 128 | 32 | 0.03532 | 61.99 | 31 | 16.52 |
| 128 | 64 | 0.03955 | 55.37 | 23 | 22.26 |
| 128 | 128 | 0.04952 | 44.22 | 19 | 26.95 |
| 256 | 32 | 0.03590 | 123.83 | 47 | 21.79 |
| 256 | 64 | 0.03768 | 117.98 | 31 | 33.03 |
| 256 | 128 | 0.04579 | 97.09 | 23 | 44.52 |
| 512 | 32 | 0.03705 | 227.45 | 79 | 25.92 |
| 512 | 64 | 0.03860 | 218.33 | 47 | 43.57 |
| 512 | 128 | 0.04911 | 171.59 | 31 | 66.06 |
| 1024 | 32 | 0.04436 | 387.46 | 143 | 28.64 |
| 1024 | 64 | 0.04121 | 417.03 | 79 | 51.85 |
| 1024 | 128 | 0.04832 | 355.68 | 47 | 87.15 |

The best ladder is:

- `block_m = 32` for `128`, `256`, `512`
- `block_m = 64` for `1024`
- `block_m = 128` only after that

This is almost exactly the same shape family the reference chooses.

## Long-Run Confirmation

Same prepared case, `rep = 3000`:

| Batch | Best Example `block_m` | Example ms | Reference ms | Example / Reference |
|---|---:|---:|---:|---:|
| 128 | 32 | 0.03534 | 0.03471 | 0.982x |
| 1024 | 64 | 0.04103 | 0.04102 | 1.000x |

So a low-batch `block_m` ladder alone nearly erases the measured GPT-OSS MM1 gap.

## NCU Read

Current default example vs reference:

| Case | Kernel | Time us | SM % | Tensor % | DRAM % | L2 % | Issue Active | Eligible Warps | Barrier Stall | Long Scoreboard | MIO Throttle | Regs | SMEM KB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 example | `ws_matmul_kernel` | 60.77 | 40.04 | 35.85 | 30.20 | 22.84 | 0.30 | 0.443 | 2.556 | 7.307 | 0.234 | 128 | 210.05 |
| 128 reference | `_p_matmul...16x256x128...` | 43.65 | 34.82 | 6.09 | 42.07 | 31.74 | 0.14 | 0.148 | 3.087 | 7.167 | 0.243 | 255 | 162.92 |
| 512 example | `ws_matmul_kernel` | 58.78 | 44.06 | 39.90 | 33.47 | 25.19 | 0.33 | 0.498 | 2.370 | 6.272 | 0.251 | 128 | 210.05 |
| 512 reference | `_p_matmul...32x256x128...` | 45.44 | 38.07 | 12.89 | 43.16 | 32.44 | 0.16 | 0.176 | 2.595 | 6.167 | 0.190 | 255 | 173.16 |
| 1024 example | `ws_matmul_kernel` | 59.14 | 43.12 | 39.04 | 33.33 | 25.18 | 0.32 | 0.475 | 2.396 | 6.563 | 0.262 | 128 | 210.05 |
| 1024 reference | `_p_matmul...64x256x128...` | 50.72 | 40.65 | 23.33 | 38.79 | 29.27 | 0.19 | 0.211 | 2.219 | 4.078 | 0.959 | 255 | 193.89 |

Interpretation:

- The default example is not “idle” at low batch. It is actively doing more work with a larger
  `block_m`, larger SMEM footprint, and lower useful-byte locality.
- The reference wins on low batch with higher DRAM/L2 throughput and much smaller tiles, not with
  split-K.
- Single-launch NCU on the runtime-overridden `block_m=32` example did not mirror the graph-timed
  speedup cleanly enough to use as a promotion signal, so the timing decision is based on the
  corrected `do_bench_cudagraph` measurements above.

## Epilogue Constraint And Fix

The original exact helper-wavefront example could not compile `block_m = 16` with the old
`epilogue_row_subtile_factor = 8`. The packed epilogue split path failed with:

`Fp4ToFpOp/SplitOp requires at least 2 elements per thread in the axis/last dimension`

The important follow-up is that this was not a fundamental `block_m = 16` limitation. It was a
fragment-layout limitation. The same exact helper-store path becomes legal again if the row
subtiling is relaxed:

- `block_m = 16`, `epilogue_row_subtile_factor = 4` compiles and validates
- `block_m = 16`, `epilogue_row_subtile_factor = 2` compiles and validates

So the real low-batch implementation problem is:

- choose a smaller `block_m`
- choose a matching smaller row-subtile factor for the exact helper-wavefront epilogue

and not “rewrite the epilogue before `block_m = 16` can ever work.”

## Landed Low-Batch Policy

The live example now has a dynamic low-batch selector keyed off `slice_size`:

| `slice_size` | `block_m` | `epilogue_row_subtile_factor` |
|---:|---:|---:|
| `<= 8` | `16` | `2` |
| `<= 16` | `32` | `4` |
| `<= 58` | `64` | `4` |
| `> 58` | `128` | `8` |

This preserves the current high-batch kernel shape while making the exact helper-store path legal
for the smallest GPT-OSS MM1 buckets.

The first landed selector used `<= 32` for the `64/4` region, but a full post-landing sweep showed
that transition was too early and created an avoidable pocket at `batch=1280..1536`. Targeted
same-process A/Bs showed the real crossover is much later:

| Batch | `slice_size` | `64/4` ms | `128/8` ms | Better |
|---|---:|---:|---:|---|
| 1280 | 40 | 0.03582 | 0.04269 | `64/4` |
| 1536 | 48 | 0.03604 | 0.04345 | `64/4` |
| 1792 | 56 | 0.04137 | 0.04153 | `64/4` |
| 1856 | 58 | 0.04095 | 0.04167 | `64/4` |
| 1920 | 60 | 0.04168 | 0.04104 | `128/8` |
| 1984 | 62 | 0.04202 | 0.04136 | `128/8` |

So the stable selector boundary is `slice_size <= 58` for `64/4`, then `128/8` after that.

## Post-Landing Results

Same prepared case, same GPU, `rep = 1000`:

| Batch | Example ms | Reference ms | Example / Reference |
|---|---:|---:|---:|
| 128 | 0.03265 | 0.03304 | 1.012x |
| 256 | 0.03267 | 0.03351 | 1.026x |
| 512 | 0.03379 | 0.03415 | 1.011x |
| 1024 | 0.03583 | 0.03633 | 1.014x |
| 2048 | 0.04062 | 0.04253 | 1.047x |
| 4096 | 0.04855 | 0.04963 | 1.022x |
| 8192 | 0.07159 | 0.07720 | 1.078x |
| 16384 | 0.11184 | 0.12187 | 1.090x |

Long-run confirmation, `rep = 3000`:

| Batch | Example Config | Example ms | Reference ms | Example / Reference |
|---|---|---:|---:|---:|
| 128 | `block_m=16`, `row_subtile=2` | 0.03262 | 0.03306 | 1.013x |
| 256 | `block_m=16`, `row_subtile=2` | 0.03265 | 0.03344 | 1.024x |
| 512 | `block_m=32`, `row_subtile=4` | 0.03377 | 0.03411 | 1.010x |
| 1024 | `block_m=64`, `row_subtile=4` | 0.03583 | 0.03626 | 1.012x |

Raw data for the landed policy is recorded in:

- `.codex/initiatives/artifacts/ws-low-batch-performance-post-landing.csv`
- `.codex/initiatives/artifacts/ws-low-batch-performance-post-landing-rep3000.csv`
- `.codex/initiatives/artifacts/ws-low-batch-performance-refresh-after-boundary-fix.csv`

## Conclusions

1. The biggest missing optimization for GPT-OSS MM1 low batch was **not split-K**. It was a
   low-batch `block_m` ladder plus smaller exact helper-store row fragments.
2. That policy is now landed in the standalone example and flips the GPT-OSS sweep from a
   low-batch loss into a low-batch win.
3. Split-K remains a valid design for future workloads, but it is no longer justified as the next
   step for GPT-OSS 120B MM1 on this example.
4. The real exact-epilogue lesson is that `block_m = 16` only required a fragment-layout change,
   not a major epilogue rewrite.
