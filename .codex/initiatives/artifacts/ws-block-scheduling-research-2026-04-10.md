## WS Block Scheduling Research

Updated: 2026-04-10T01:38:45Z

### Scope

This note narrows the next experiment wave to block-scheduling changes only for the live WS example:

- `python/examples/gluon/05-moe-bmm1-fused-gather.py`

The goal is to vary only the mapping from persistent `block_id` to logical `(pid_m, pid_n)` work tiles while leaving the rest of the load/MMA/epilogue pipeline intact.

### Current Kernel Hook

- The current WS example packs `(slice_idx, pid_m_in_slice)` into `x_block_schedule` and decodes it inside `apply_block_schedule(...)`.
- Today the live mapping is a simple row-major / M-stationary traversal over `(grid_m, GRID_N)`:
  - `pid_m = pid_mn // GRID_N`
  - `pid_n = pid_mn % GRID_N`
- The schedule metadata itself is still the canonical ragged-tensor schedule produced by `make_ragged_tensor_metadata(...)`:
  - low 16 bits = `slice_idx`
  - high 16 bits = `block_idx_within_slice`

Relevant local references:

- `python/examples/gluon/05-moe-bmm1-fused-gather.py`
- `python/perf/matmul_gluon.py`
- `python/perf/matmul_gluon_optimized.py`
- `python/examples/gluon/03-matmul-multicta.py`
- `python/examples/gluon/04-2cta-block-scale-matmul.py`
- `python/triton_kernels/triton_kernels/tensor_details/ragged_tensor.py`

### Target Geometry To Tune Against

Primary target bucket:

- case: `bs16384_E256_es8_B5120x10240`
- current WS tile shape: `BLOCK_M=128`, `BLOCK_N=256`
- derived `grid_m=543`, `grid_n=40`, `total_tiles=21720`
- current persistent launch: `NUM_SMS=152`

For the current row-major mapping, a persistent CTA steps by:

- `+3` in the logical `pid_m` dimension
- `+32` in the logical `pid_n` dimension

because `152 = 3 * 40 + 32`.

This matters: the scheduling problem is not just “which order is good in isolation,” but “which order stays locality-friendly under a persistent stride of 152 CTAs.”

### Local Strategy Inventory

#### 1. Current WS row-major / M-stationary traversal

The live WS kernel holds one logical M tile while it walks N. This maximizes gathered-activation reuse because a CTA keeps the same gathered rows across adjacent N panels.

Reference:

- `python/examples/gluon/05-moe-bmm1-fused-gather.py`

#### 2. `matmul_gluon*` M-major / N-major toggles

`python/perf/matmul_gluon.py` and `python/perf/matmul_gluon_optimized.py` already carry a scheduling helper with:

- `N_MAJOR`
- `XCD_SWIZZLE`

Those files prove that this codebase already treats block traversal as a first-order tuning knob, especially for persistent kernels.

References:

- `python/perf/matmul_gluon.py`
- `python/perf/matmul_gluon_optimized.py`

#### 3. Planar snake traversal from the multicta/block-scale examples

`03-matmul-multicta.py` and `04-2cta-block-scale-matmul.py` use `_planar_snake(...)`, borrowed from Pallas / CUTLASS-style traversal logic, to rasterize tiles in bands of configurable width and reverse the major direction every band.

This is the cleanest already-local candidate for:

- “snake over M bands”
- “snake over N bands”
- varying band width without changing the rest of the kernel

References:

- `python/examples/gluon/03-matmul-multicta.py`
- `python/examples/gluon/04-2cta-block-scale-matmul.py`

### External Strategy Inventory

#### CUTLASS grouped persistent scheduler

CUTLASS documents grouped kernels as persistent threadblocks that repeatedly query the next tile and advance by `gridDim.x`. The grouped GEMM scheduler uses round-robin assignment by default. CUTLASS also exposes:

- `kDeviceOnly` scheduling
- `kHostPrecompute` scheduling
- sorting grouped problems by descending `K` to reduce imbalance

Key references:

- CUTLASS grouped scheduler doc:
  - `https://docs.nvidia.com/cutlass/4.3.4/media/docs/cpp/grouped_scheduler.html`
- CUTLASS discussion noting host-precompute and sorting support:
  - `https://github.com/NVIDIA/cutlass/discussions/627`

Most relevant ideas for WS:

- persistent round-robin assignment
- host-precomputed per-block playlists
- sorting tiles/problems by cost

Only the first is directly in-scope for this pass. Host-precompute and sorting would require changing the ragged schedule metadata, not just the WS block-schedule function.

#### CUTLASS 3.x persistent and Stream-K schedulers

The CUTLASS 3.x technical blog calls out two additional tile schedulers for Hopper:

- persistent scheduler: one CTA per SM, each CTA computes multiple output tiles
- Stream-K scheduler: also persistent, but divides some work along K for better load balance

On Blackwell, CUTLASS points users toward cluster-launch-control schedulers instead.

Reference:

- `https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/`

Most relevant ideas for WS:

- persistent CTA ownership matters as much as tile order
- schedule choice should be problem-shape dependent

Stream-K itself is out of scope for this pass because it changes work decomposition along K, not just tile ordering.

#### PyTorch locality-aware MoE GEMM scheduling

PyTorch’s locality-aware MoE post is the most directly relevant public source for fused / routed MoE GEMMs. It explicitly compares:

- Split-K
- row-major traversal
- column-major traversal
- grouped launch ordering

and argues that for skinny MoE GEMMs, column-major ordering can maximize reuse of weight-matrix columns. It reports up to 4x speedups and substantially better cache hit rates versus the baseline traversal in that setting.

Reference:

- `https://pytorch.org/blog/accelerating-moe-model/`

Most relevant ideas for WS:

- row-major vs column-major is worth testing directly
- grouped launch bands are worth testing directly
- the right answer depends on whether activation reuse or weight reuse dominates

#### PyTorch persistent cache-aware grouped GEMM

The later PyTorch blog on a persistent cache-aware grouped GEMM kernel shows a grouped launch schedule that improves cache performance for both A and B, explicitly by causing consecutive CTAs to reuse the same B tile while keeping a band of A rows in cache. It also combines that schedule with persistent execution and device-side TMA descriptor updates for expert weights.

Reference:

- `https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/`

Most relevant ideas for WS:

- grouped launch can balance A and B locality better than pure row-major or pure column-major
- schedule experiments should be interpreted through L2 hit rate and memory-workload counters, not just end-to-end time

#### TensorRT-LLM MoE optimization notes

The public TensorRT-LLM material is high-level on scheduling, but it does confirm two useful themes:

- MoE grouped GEMM performance is sensitive enough that NVIDIA reports CUTLASS Group GEMM optimizations separately
- overlap/runtime scheduling is a first-class optimization dimension in production MoE inference

References:

- `https://nvidia.github.io/TensorRT-LLM/1.1.0rc2.post1/blogs/tech_blog/blog3_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.html`
- `https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog08_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.html`

These are directional references only. They do not expose a concrete per-tile block scheduler detailed enough to transplant into the WS kernel.

#### Stream-K / Stream-K++

Stream-K is the canonical “persistent but not tile-only” scheduler family: instead of assigning full output tiles statically, it partitions work more evenly across processing elements by slicing the aggregate K work. Stream-K++ extends that idea with multiple scheduling policies and a lightweight selector.

References:

- GPU Load Balancing dissertation / Stream-K summary:
  - `https://arxiv.org/abs/2212.08964`
- Stream-K++:
  - `https://arxiv.org/abs/2408.11417`

These are useful for future work, but out of scope for this pass because they require a different decomposition than “change only the WS block scheduling function.”

#### EPS-MoE

EPS-MoE is included mainly as a reminder that MoE scheduling is often load-driven, not just locality-driven. It dynamically chooses dense vs grouped kernels and overlaps communication with expert compute.

Reference:

- `https://arxiv.org/abs/2410.12247`

This is out of scope for the current kernel-local schedule pass, but relevant as future context if the tuning bottleneck becomes load imbalance across slices.

### Strategy Shortlist For This Pass

In-scope:

- row-major / M-stationary
- N-major / B-stationary
- grouped-M bands
- grouped-N bands
- planar snake over M bands
- planar snake over N bands

Likely widths to try first:

- `4`
- `8`
- `16`

Out of scope for now:

- host-precomputed per-block playlists
- sorting slice/problem metadata on the host
- Stream-K or Split-K
- CLC / dynamic tile cancellation
- launch-grid retuning

### Working Hypotheses

1. Current row-major likely favors gathered-activation reuse, because a CTA keeps the same M tile while draining N.
2. A pure N-major schedule may improve weight/scale reuse, but may scatter gathered rows too aggressively under the persistent stride.
3. Grouped or snake schedules are the most plausible middle ground: they can improve B-panel reuse without fully destroying A reuse.
4. Because the persistent stride is `152`, schedules should be judged by the modulo pattern they induce across both `grid_m=543` and `grid_n=40`, not by their non-persistent intuition alone.

### Mirror Status

Direct source mirroring from this host is still partially blocked by outbound TLS failures (`curl: (35) OpenSSL SSL_connect: SSL_ERROR_SYSCALL`) against `pytorch.org`, `docs.nvidia.com`, `developer.nvidia.com`, and `nvidia.github.io`. The durable fallback for this session is:

- keep the canonical references and titles in this note
- capture the extracted conclusions in-repo
- leave raw page/PDF mirroring for a host where those domains can be fetched normally

### Results

#### Strategy set implemented in-kernel

The live WS example now supports a scheduling strategy flag with the following variants:

- `row_major`
- `n_major`
- `grouped_m_{4,8,16}`
- `grouped_n_{4,8,16}`
- `snake_m_{4,8,16}`
- `snake_n_{4,8,16}`
- `row_major_swizzle_{2,4}`
- `n_major_swizzle_{2,4}`

The strategy switch is intentionally narrow: only `apply_block_schedule(...)` changes the logical mapping from persistent `block_id` to `(pid_m, pid_n)`.

#### First 4-GPU screen (`rep=200`, exact validation)

These runs were used only to prune the strategy set. They are not cross-GPU promotion decisions.

Top observations:

- `row_major` stayed strong immediately (`0.3400 ms` on GPU 0)
- `n_major` was clearly worse (`0.3512 ms` on GPU 0)
- grouped and snake variants clustered close to the baseline, but none showed an obvious win
- best screen-side alternates were `snake_n_16`, `snake_n_8`, `snake_n_4`, `grouped_n_16`, and `grouped_m_4`

Clear losers:

- `row_major_swizzle_2 = 0.3551 ms`
- `row_major_swizzle_4 = 0.3894 ms`
- `n_major_swizzle_2 = 0.3430 ms`
- `n_major_swizzle_4 = 0.3453 ms`

The swizzle family was motivated by the target persistent stride (`152`) and the old `matmul_gluon` helper, but on this bucket it over-scattered the walk enough to regress.

#### Same-GPU confirmation (`GPU 0`)

`rep=500`, exact validation:

- `row_major = 0.3405 ms`
- `snake_n_16 = 0.3410 ms`
- `grouped_n_16 = 0.3423 ms`
- `snake_n_8 = 0.3422 ms`
- `grouped_m_4 = 0.3434 ms`

`rep=1000`, exact validation:

- `row_major = 0.3400 ms`
- `snake_n_16 = 0.3406 ms`

So the closest alternate (`snake_n_16`) is real and legal, but still slightly slower than the current row-major default.

#### Same prepared-case check versus `gluon_optimized`

On the aligned `local_rank=0` target case with `rep=1000`, exact validation:

- `ws_optimized (row_major) = 0.3406 ms`
- `gluon_optimized = 0.3408 ms`

This scheduling pass did not find a new win over the current WS default, but it also did not regress it.

#### NCU comparison: `row_major` vs `snake_n_16`

Average across the profiled `ws_matmul_kernel` instances in the two NCU reports:

- `row_major`
  - time `480.0 us`
  - L2 throughput `36.04%`
  - tensor pipe active `73.76%`
  - SM throughput `82.63%`
  - barrier stall `0.579`
  - long scoreboard `7.543`
  - eligible warps `0.527`
- `snake_n_16`
  - time `482.8 us`
  - L2 throughput `32.82%`
  - tensor pipe active `73.45%`
  - SM throughput `82.29%`
  - barrier stall `0.569`
  - long scoreboard `7.230`
  - eligible warps `0.543`

Interpretation:

- the nearest snake variant does reduce some scheduling-pressure counters (`barrier`, `long_scoreboard`)
- but it also reduces L2 throughput and slightly lowers tensor / SM activity
- on this bucket, the current row-major walk still wins because its locality balance is better overall

### Next Experiment Order

1. Implement a strategy flag in `apply_block_schedule(...)`.
2. Add row-major, N-major, grouped-M/N, planar-snake, and stride-aware swizzle variants.
3. Screen all variants on `bs16384_E256_es8_B5120x10240`.
4. Profile the top few variants with NCU to distinguish:
   - better A locality
   - better W/scale locality
   - better persistent-stride behavior
5. Keep `row_major` as the default unless a same-GPU rerun shows a clear win from an alternate strategy.
