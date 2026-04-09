# WS Epilogue Store Overlap SASS Notes

## Goal

Check whether ptxas is interleaving epilogue arithmetic with the store-side work for the two current WS variants:

- helper-store `ws_optimized`
- epilogue-owned ragged async-TMA-store `ws_optimized_tma_store`

Also test whether the helper-store ring slot can be released immediately after loading the fragment out of shared memory, instead of after the helper partition finishes `gl.store`.

## Methodology

Use the target `bs=16384`, `E256`, `es8`, `non-parrot` case on one GPU and force each kernel into its own Triton cache so the generated cubin can be inspected in isolation.

1. Compile the kernel with exact validation and an isolated cache:
   - `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/<cache> PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel <kernel> --limit 1 --validate-only --validation-reference exact`
2. Find the generated cubin under that cache:
   - `find /tmp/<cache> -name '*.cubin'`
3. Dump SASS with source line mappings:
   - `nvdisasm --print-line-info <cubin> > /tmp/<kernel>.sass`
4. Inspect the source-mapped windows for:
   - `_enqueue_packed_fp8_fragment(...)` lines `595-597`
   - `_issue_async_packed_fp8_fragment(...)` lines `614-618`
   - `epilogue_store_partition_optimized(...)` lines `777-787`
5. Measure any code change on the same GPU with a long run:
   - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python/triton_kernels python -m python.perf.bench_matmul_parrot_gather --batch-size 16384 --case-family non-parrot --kernel ws_optimized,gluon_optimized --limit 1 --warmup 30 --rep 1000 --validation-reference exact`

## Findings

### 1. The source is already fragment-pipelined

The row-fragment epilogue source already overlaps `prepare(i)`, `finish(i-1)`, and `enqueue/issue(i-2)` across fragments. The SASS question is not whether ptxas invents software pipelining from nothing, but whether it further mixes the store-side instructions into that pipelined arithmetic stream.

### 2. ptxas does interleave producer-side SMEM writes with arithmetic

For the helper-store producer path, the source just does:

- `mbarrier.wait(empty_bar, store_phase)`
- `store_buf.store(payload)`
- `mbarrier.arrive(ready_bar)`

In SASS, those `STS.U16` payload stores are mixed with ongoing SwiGLU arithmetic and packed conversion instructions rather than being left as one isolated block. So ptxas is already doing useful instruction-level overlap on the producer-side ring writes.

### 3. ptxas also interleaves the async-TMA-store issue path with arithmetic

For the TMA-store variant, the source does:

- `tma.store_wait(depth - 1)`
- `store_buf.store(payload)`
- `tma.async_copy_shared_to_global(...)`

In SASS, the `DEPBAR.LE`, `STS.U16`, `UTMASTG.4D`, and `UTMACMDFLUSH` instructions are also mixed with the fragment arithmetic. So the TMA-store variant is not losing because ptxas failed to overlap the issue path at all.

### 4. The helper consumer partition is mostly a wait/load/store loop

The helper-side global-store partition is different. Its SASS is mostly:

- wait on `ready`
- `LDSM` the packed fragment from shared memory
- address/mask setup
- clustered `STG.E`
- `arrive(empty)`

There is very little arithmetic to hide there. The helper path wins primarily because the final global-store work is on separate warps, not because ptxas found unusual overlap inside that consumer partition.

### 5. Releasing the helper ring slot after the SMEM load is legal and helps a little

Changing the helper consumer from:

- wait `ready`
- load fragment from SMEM
- `gl.store`
- `arrive(empty)`

to:

- wait `ready`
- load fragment from SMEM
- `arrive(empty)`
- `gl.store`

is legal because the fragment is already in registers after the `load(store_layout)`.

This changed the helper-store SASS in the expected way: the `SYNCS.ARRIVE` for the empty barrier moved ahead of the `STG.E` cluster, so the slot is genuinely released early instead of just being moved in source without codegen effect.

Measured on GPU 0 with `warmup=30`, `rep=1000`, exact validation:

- pre-change `HEAD`: `ws_optimized = 0.3481 ms`
- early-release patch: `ws_optimized = 0.3474 ms`
- same run `gluon_optimized = 0.3482 ms`

So the earlier slot release is a small but real improvement, about `0.0007 ms` or `~0.2%` on the target bucket.

## Interpretation

The current helper-vs-TMA-store difference is not explained by a failure of ptxas to overlap store-side instructions. ptxas already interleaves both the helper producer path and the epilogue-owned async-TMA-store path with arithmetic.

The remaining difference is architectural:

- helper-store keeps the final global-store work on separate warps
- helper-store can free the SMEM slot once the fragment is in registers
- epilogue-owned async-TMA-store keeps the store-wait dependency on the compute partition and ties slot reuse to store completion

That matches the earlier NCU result: async TMA store did not lose on write volume, but it did lose on barrier-style stall and issue efficiency.

## Next Experiments Suggested By These Notes

- Keep the early empty-barrier release in the helper consumer.
- If TMA store is revisited, test a separate store partition that issues `tma.async_copy_shared_to_global(...)` from the ring, so the compute partition keeps the helper-style decoupling.
