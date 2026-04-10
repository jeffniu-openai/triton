# WS Example Split-K Design

## Current Status

This remains a design-only artifact.

After landing the low-batch `block_m` ladder and matching exact helper-store row-subtile policy in
the standalone example, the GPT-OSS 120B MM1 sweep no longer shows a residual low-batch gap that
justifies split-K as the immediate next step. The current live policy is:

- `slice_size <= 8`: `block_m = 16`, `epilogue_row_subtile_factor = 2`
- `slice_size <= 16`: `block_m = 32`, `epilogue_row_subtile_factor = 4`
- `slice_size <= 32`: `block_m = 64`, `epilogue_row_subtile_factor = 4`
- otherwise: `block_m = 128`, `epilogue_row_subtile_factor = 8`

On the current GPT-OSS MM1 sweep, that policy beats the production reference at every measured
batch from `128` through `16384`. So split-K should stay deferred unless a different low-batch
workload still shows a material gap after the landed tile-policy fixes.

## Goal

Design an example-only split-K path for
[05-moe-bmm1-fused-gather.py](/root/code/triton-ws-opt/python/examples/gluon/05-moe-bmm1-fused-gather.py)
that can be implemented later without guessing if a future workload still needs it. This design
assumes the low-batch tile-policy work lands first, because the research pass showed that
low-batch tiling is the primary GPT-OSS MM1 gap and the production reference does not need split-K
to win the low-batch cases.

## Constraints From The Current Example

- The example is a persistent ragged-M grouped GEMM with:
  - one merged load partition
  - one MMA partition
  - one exact helper-wavefront epilogue partition
  - one helper store partition
- Current scheduling space is only `(pid_m, pid_n)`.
- Current epilogue assumes the full K reduction is complete before:
  - bias add
  - exact SwiGLU
  - FP8 requantization
- Output is FP8, but exact SwiGLU correctness requires reduction in pre-activation FP32 space.
- The production `triton_kernels.matmul` path already has split-K infrastructure, but its host policy
  disables split-K when MX scales are present and therefore cannot simply be copied over for this
  FP8 x MXFP4 example.

## Recommended Design

Use a **two-kernel split-K design**.

Kernel 1:

- split the K dimension across `split_k` shards
- compute partial FP32 accumulators only
- write those partials to global scratch

Kernel 2:

- reduce the scratch across `split_k`
- add bias
- run the current exact helper-wavefront SwiGLU epilogue
- write final FP8 output

This is the simplest design that is:

- mathematically correct for exact SwiGLU
- compatible with the current helper-store epilogue
- easy to validate
- aligned with the usual CUTLASS split-K model of partial GEMM followed by a separate reduction

## Rejected Alternatives

- Atomic accumulation directly into the final FP8 output:
  - rejected because output is quantized and activation-reduced
- Atomic accumulation into a final FP32 tensor followed by in-place fused activation:
  - rejected for v1 because it complicates ownership and validation more than a clean two-kernel path
- Applying SwiGLU inside each split-K shard:
  - invalid, because activation must happen after summing all K shards
- Reusing the current helper ring as a split-K reduction path:
  - rejected for v1 because the ring is a CTA-local epilogue pipeline, not a global reduction mechanism

## Kernel 1: Split-K Partial Matmul

### New Config Surface

Add to the example-side config policy:

- `split_k: int = 1`

Keep the initial search space narrow:

- `split_k in {1, 2, 4}`

Only allow `split_k > 1` for low batch, after the low-batch tile policy is already in place.

### Block Mapping

Current example:

- `num_blocks = grid_m * grid_n`
- `block_id -> (pid_m, pid_n)`

Split-K version:

- `num_blocks = grid_m * grid_n * split_k`
- `pid_mnk = block_id % (grid_m * grid_n * split_k)`
- `pid_k = pid_mnk % split_k`
- `pid_mn = pid_mnk // split_k`
- apply the existing ragged-M block schedule to `pid_mn`

So the current `apply_block_schedule(...)` concept stays intact, but it operates on `pid_mn` instead
of the raw `block_id`.

### K Traversal

Replace the current contiguous K loop with a shard-strided loop:

- `base_ki = pid_k`
- `ki = base_ki + iter * split_k`
- `off_k_x = ki * BLOCK_K`
- `off_k_w = ki * PACKED_BLOCK_K`

Equivalent formulation:

- `for ki in range(pid_k, K_TILES, split_k): ...`

This matches the production persistent split-K pattern conceptually.

### Scratch Output

Recommended scratch layout:

- shape: `[split_k, m_local, n_preact]`
- dtype: `float32`
- contiguous in `n_preact`

Where:

- `m_local = gather_indx.shape[0]`
- `n_preact = N = 5760` for GPT-OSS MM1 before SwiGLU reduction

Why this layout:

- simplest correctness story
- easiest reduction kernel
- matches the logical routed-row order already used by the example

### What Kernel 1 Should Not Do

Kernel 1 should not:

- load bias
- run SwiGLU
- requantize to FP8
- use the helper store ring

It should stop at FP32 partial writeback.

## Kernel 2: Split-K Reduction Plus Exact Epilogue

### Work Mapping

Use the same `(pid_m, pid_n)` schedule as the current example, without `pid_k`.

For each logical output tile:

- load the `split_k` FP32 partial tiles from scratch
- sum them in registers
- then reuse the current bias + exact SwiGLU + helper-store output flow

### Reuse Boundary

Do not rewrite the full epilogue if it can be avoided.

Recommended factoring boundary:

- keep the current exact helper-wavefront epilogue helpers as the canonical path from
  “pre-activation FP32 tile” to final output
- add a new reduction-side loader that produces the same accumulator-like tile interface from global
  scratch instead of from TMEM

In other words, reuse:

- bias load
- exact SwiGLU fragment path
- helper enqueue
- helper store partition

and swap only the accumulator source.

### Scratch Memory Cost

For GPT-OSS MM1, low batch only:

- `batch = 128`, `m_local = 512`, `n = 5760`
  - `split_k = 4` scratch = about `45 MB`
- `batch = 1024`, `m_local = 4096`, `n = 5760`
  - `split_k = 4` scratch = about `360 MB`

So split-K must stay low-batch-only in v1. It is not acceptable as a general always-on path.

## Host-Side Policy

Implement split-K only after confirming that the landed low-batch tile policy still leaves a
material gap on the target workload.

Recommended first policy if split-K is revived:

- only try `split_k > 1` on the low-batch region that still lags after the ladder lands
- initial candidate policy:
  - `split_k = 1` by default
  - try `2`, then `4`

Do not enable split-K for large batch or by default across the whole sweep.

## Validation Plan For The Future Implementation

Correctness:

- exact-reference tolerance on:
  - `batch = 128`
  - `256`
  - `512`
  - `1024`
- compare:
  - current example
  - low-batch `block_m` ladder only
  - low-batch ladder + split-K

Performance:

- main GPT-OSS MM1 sweep:
  - `128`, `256`, `512`, `1024`, `2048`, `4096`, `16384`
- use same prepared inputs and `do_bench_cudagraph`
- measure:
  - current example
  - low-batch ladder only
  - low-batch ladder + split-K
  - reference

Acceptance:

- no regression larger than `2%` at `2048`
- no regression larger than `1%` at `16384`
- improve the post-ladder low-batch buckets if split-K is kept

## Recommended Implementation Order

1. Reconfirm that the current landed low-batch policy still leaves a meaningful gap on the target
   workload.
2. Only if low-batch still lags materially, add split-K kernel 1 with FP32 scratch.
3. Add split-K reduction + exact epilogue kernel 2.
4. Benchmark `split_k = 1/2/4` on low batch only.

## Summary

The design is straightforward, but the research result changes the priority:

- low-batch tile policy was first-order and is now landed
- split-K is a deferred second-stage extension
- the recommended split-K path is a clean two-kernel FP32 pre-activation reduction design, not
  atomics and not per-shard SwiGLU
