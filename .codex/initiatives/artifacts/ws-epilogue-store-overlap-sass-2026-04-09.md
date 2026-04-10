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

### 6. The small helper win is consistent with STG scoreboard semantics

The early-empty-barrier speedup is not evidence that the old helper consumer was waiting for the global stores to reach memory before executing `SYNCS.ARRIVE`. The more plausible model is:

- `STG.E` issues quickly once its source registers are ready
- the scoreboard dependency is tied to those source registers and any later register consumers, not to the barrier itself
- `SYNCS.ARRIVE` can therefore issue immediately after the `STG.E` bundle if there is no dependency between them

That means the old helper consumer was already effectively decoupled from global-store completion. Moving `arrive(empty)` earlier only shortened the handoff window by the local issue distance between the barrier and the `STG.E` cluster, which matches the observed tiny-but-real win.

### 7. Helper buffering is not the same as keeping more data in epilogue registers

The helper path is not just "more buffering in registers." It is buffering in a different partition's registers:

- the epilogue partition computes and writes the packed fragment into the SMEM ring
- the helper partition loads that fragment into its own registers
- the helper partition then spends its own issue slots on pointer math, predication, repacking, and `gl.store`

If the fragment stayed in the epilogue registers instead, the epilogue warps would still have to issue all of that final-store work themselves. Raising the epilogue register budget alone cannot create the same decoupling because it does not add separate warp issue bandwidth; it only lets the same warps hold more live data while they still own the address-generation and store instructions.

### 8. Manual source-level interleaving is probably low-upside unless it changes granularity

The current helper producer SASS already shows ptxas doing the obvious useful thing:

- `TRYWAIT`
- early `STS.U16` payload writes
- lots of later SwiGLU / packed-float2 arithmetic
- later `SYNCS.ARRIVE(ready)`

So a source rewrite that merely restates the current full-fragment schedule as "do some arithmetic, then some `STS`, then more arithmetic" is unlikely to buy much. ptxas is already placing the SMEM stores early and overlapping them with the remaining fragment math.

The source-level interleave that still looks plausible is a larger change:

- split the fragment into smaller independently-ready pieces
- perform exact SwiGLU + packing + `STS` per smaller piece
- let the helper consumer observe those smaller ready units

That would change the ownership and readiness granularity, not just the textual order of the current full-fragment code.

### 9. A separate-partition TMA-store consumer needs `fence_async_shared()` and still loses

The follow-up experiment from these notes was to replace the epilogue-owned async-TMA path with a separate store partition that:

- waits on the same ready barrier used by the `gl.store` helper path
- issues `tma.async_copy_shared_to_global(...)` from the shared-memory ring
- uses `tma.store_wait(depth - 1)` before releasing the slot for reuse

The first version of that experiment was not exact. Sparse fragment corruption disappeared only after adding `fence_async_shared()` on the producer side immediately after writing the ring slot and before `mbarrier.arrive(ready_bar)`.

That is a durable correctness finding: a cross-partition TMA-store consumer needs a shared-memory visibility fence before the ready barrier, even though the ordinary helper consumer that loads the fragment into registers does not.

With that fence in place, the kernel validated exactly, but it was still slower on the target GPU 0 run (`warmup=30`, `rep=1000`, exact validation):

- `ws = 0.3549 ms`
- helper-store `ws_optimized = 0.3485 ms`
- separate-partition `ws_optimized_tma_store = 0.3564 ms`
- `gluon_optimized = 0.3486 ms`

NCU on the exact-validation path shows why the helper still wins:

- helper `ws_matmul_kernel_optimized`: `479.712 us`
- separate-partition `ws_matmul_kernel_optimized_tma_store`: `505.312 us`
- registers/thread: both `128`
- shared memory/block: both `226.528 KB`
- tensor throughput: `74.71%` helper vs `70.81%` TMA
- SM throughput: `83.68%` helper vs `79.89%` TMA
- issue active: `0.33` helper vs `0.30` TMA
- eligible warps/scheduler: `0.53` helper vs `0.45` TMA
- barrier stall/issued inst: `0.57` helper vs `0.62` TMA
- long scoreboard/issued inst: `7.54` helper vs `8.87` TMA

So moving TMA store onto a separate partition removes the old "compute partition owns `store_wait`" problem, but it does not create a better kernel. The remaining cost is now in the TMA-store path itself plus the extra shared-memory fence and ready/empty protocol.

## Interpretation

The current helper-vs-TMA-store difference is not explained by a failure of ptxas to overlap store-side instructions. ptxas already interleaves both the helper producer path and the epilogue-owned async-TMA-store path with arithmetic.

The remaining difference is architectural:

- helper-store keeps the final global-store work on separate warps
- helper-store can free the SMEM slot once the fragment is in registers
- epilogue-owned async-TMA-store keeps the store-wait dependency on the compute partition and ties slot reuse to store completion
- even a separate-partition TMA-store consumer still needs `fence_async_shared()` before the ready barrier and remains slower than the helper-store consumer on the target bucket
- direct epilogue store with a higher register budget is still not equivalent to the helper path, because the epilogue warps would continue to own the final address-generation and `STG.E` issue stream

That matches the earlier NCU result: async TMA store did not lose on write volume, but it did lose on barrier-style stall and issue efficiency.

## Next Experiments Suggested By These Notes

- Keep the early empty-barrier release in the helper consumer.
- If TMA store is revisited, remember that a cross-partition TMA-store consumer needs `fence_async_shared()` before `arrive(ready)`. The separate-partition version is exact with that fence, but it still lost to the helper-store consumer on the target bucket, so TMA store should stay deprioritized unless a new design changes more than just the store ownership.
- If direct `gl.store` is revisited, do not expect "more epilogue regs" alone to reproduce the helper result. The more credible direct-store experiment is a finer-grained ready/store protocol that changes how much of the fragment becomes independently storable.
