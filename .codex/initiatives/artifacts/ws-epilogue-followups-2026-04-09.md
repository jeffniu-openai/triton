---
created: 2026-04-09T??:??:??Z
permalink: ws-epilogue-followups-2026-04-09
---

# WS Epilogue Followups

Documenting runnable ideas for a larger exact-math rewrite of `python/perf/matmul_ws_optimized.py`'s epilogue path. These ideas build on the helper + row-fragment infrastructure already in place and deliberately avoid approximations because the target workload requires exact SwiGLU.

## 1. Helper-owned nonlinear + store

- **What to touch:** `_epilogue_from_acc_packed`, `_enqueue_packed_fp8_fragment`, `_prepare_swiglu_fragment_from_packed`, `epilogue_store_partition_optimized`, and the helper buffers (`store_bufs`, `store_empty_bars`, `store_ready_bars`).
- **Idea:** Main partition accumulates + scales bias, packs the raw `float2` fragment, and simply enqueues it (maybe via `_enqueue_acc_packed_fragment`). The helper partition dequeues and runs `_prepare_swiglu_fragment_from_packed`, `_finish_swiglu_fragment_packed`, `_pack_fp8_out_fragment`, and `_store_packed_out_fragment` (with `_pack_packed_fp8x4`/`_pack_packed_fp8x8` if wide store is enabled). This shifts the SwiGLU math and store off the MMA warps, uses spare helper warps, and keeps the main partition linear until after enqueue.
- **Upside:** clears shared-memory relayout pressure, increases apparent pipeline depth, and keeps helper registers dedicated to nonlinear/store work.
- **Blockers:** existing helper barrier dance only recognizes fragments already resized for `_store_packed_out`; the helper currently assumes the payload is already in packed FP8 or `_pack_fp8_out_fragment` is streaming it. We'll need to ensure `store_idx/store_phase` remains consistent when helper takes over nonlinear work.

## 2. Linear/TMEM-friendly helper fragments

- **What to touch:** `ws_matmul_kernel_optimized`'s accumulator layout, `_epilogue_from_acc_packed`, `_store_helper_fragment_layout`, and helper store code.
- **Idea:** Keep accumulator fragments in the TMEM/linear layout until after enqueue; helper warps reinterpret them via `float2` and run the SWiGLU steps plus `_store_packed_out_fragment`. This eliminates the `gl.convert_layout(acc_regs, split_layout)` call and may reduce the 16 KiB shared scratch consumed by `convert_layout`.
- **Upside:** Frees shared memory for more activation/weight buffers, and helper warps deliberately stage the layout change where they already have register-based space.
- **Blockers:** The helper store path and `_pack_*` helpers expect a blocked `[?,2]` layout per thread, so we must ensure the linear fragment is reshaped to that layout before `_prepare_swiglu_fragment_from_packed` runs (likely by adjusting `_store_helper_fragment_layout` or adding a helper convert).

## 3. Register-fragment scheduling with helper prefetch/store loops

- **What to touch:** `_epilogue_from_acc_packed`, helper enqueue helpers, `epilogue_store_partition_optimized`, and `_store_packed_out_fragment` scheduling loops.
- **Idea:** Treat the helper buffers as a ping-pong queue of row fragments. The main partition writes fragment `i+2` while helper warps process fragment `i`. Each helper buffer slot carries metadata (frag index, `off_m`, `out_off_n`) so the helper can issue `_prepare_swiglu_fragment_from_packed`/`_finish`/store without recomputing offsets. This is a CUTLASS-style epilogue pipeline and may reduce register live ranges while keeping exact math.
- **Upside:** Tighter overlap, possible better scoreboard behavior, and the helper can use wide stores once fragments commit.
- **Blockers:** `store_idx/store_phase` already drive barrier ownership; adding metadata/fragment IDs will require rewiring barrier semantics to remain deterministic and preserve the observed 5-element mismatches. The existing store springs may need to become depth-2 for row16.

## 4. Helper-wide store with `USE_WIDE_PACKED_STORE32/64`

- **What to touch:** `_store_packed_out_fragment`, helper buffer loads in `epilogue_store_partition_optimized`, and `ws_matmul_kernel_optimized`'s helper layout flags.
- **Idea:** Ensure the helper path feeds `_store_packed_out_fragment` with `USE_WIDE_PACKED_STORE32` enabled (the helper layout already satisfies the `[?,2]` requirement). Combine this with the helper owning the nonlinear math so the helper issues `_pack_packed_fp8x4` and writes 4 FP8 values per thread. Potentially use `_pack_packed_fp8x8` if the helper can convert to the `gl.BlockedLayout` expected by that helper.
- **Upside:** Reduces global-store latency by widening each store, which can be especially valuable if helper warps now focus on store completion.
- **Blockers:** `_pack_packed_fp8x8` requires a specific layout with `gl.num_warps()` lanes; the helper layout must match or we must convert explicitly. The threading might limit `USE_WIDE_PACKED_STORE64` to full tile rewriting.

-
## 5. Helper-owned pre-SwiGLU ring with metadata-aware buffering

- **What to touch:** `_enqueue_packed_fp8_fragment`, `_enqueue_acc_packed_fragment` (or a new helper enqueue), `store_empty_bars`/`store_ready_bars`, and `epilogue_store_partition_optimized` loops.
- **Idea:** Extend the helper buffers so each slot carries the raw fragment plus `off_m`/`out_off_n` metadata and an explicit fragment ID. The main partition would write fragment `i+2` while the helper processes fragment `i`, keeping a depth-2 helper ring to overlap packing/work. This is the “dedicated helper partition using spare warps” concept, with the helper owning the SwiGLU math + store once the metadata is received.
- **Upside:** Provides CUTLASS-style pipeline depth without approximating SwiGLU, lets helper warps run scheduled fragments with smaller live ranges, and may expose more headroom for wide stores.
- **Blockers:** The barrier semantics in `_enqueue_*` currently only track `store_idx`/`store_phase`; adding fragment IDs requires careful rework to prevent the deterministic 5-element mismatch from earlier helper experiments. We'll also need to keep the metadata small enough to fit the shared helper layout.

These notes can guide the next coding pass once we commit to a helper-heavy epilogue redesign.

## 6. TMEM/Bias metadata followups

- **Linear-friendly pack:** `epilogue_partition_optimized` currently converts `acc_regs` into `split_layout` at ~line 1640 before packing, but `_pack_last_dim_adjacent_linear` (lines 293‑421) already reshapes/permutes distributed-linear layouts into the `[128, 2, N/2]` view the helper path expects. A redo that stays in TMEM layout until `_pack_last_dim_adjacent_linear` is invoked would skip the large `gl.convert_layout`, keep more shared memory free for helper buffers, and feed helper warps raw adjacent pairs. The helper path already supports `_enqueue_packed_float2_fragment` with a `float2.Float2Tensor` layout (lines 1100‑1205), so we can queue the TMEM data directly. Key risk: `_pack_last_dim_adjacent_linear` still calls `gl.convert_layout` under the hood; we must ensure the convert target layout matches the minimal basis so `_split_packed_last_dim_in_half` later succeeds.
- **Bias prepack reuse:** The bias path at lines 1648‑1658 currently `gl.convert_layout`s the bias into `split_layout` each launch. We could precompute bias fragments once per subtile in host code or a small shared buffer, storing `float2` pairs so the helper only runs `_fma` with `acc_packed`. This saves the repeated convert each CTA and reduces register live range when the helper later reads the bias stream. We would need to ensure the bias prepack uses the same layout as `acc_packed` to avoid further conversions.
- **Fragment metadata for helper stores:** In `epilogue_store_partition_optimized` (lines 1485‑1575) the store helper currently reuses the same `out_off_n` for each `store_idx`, which causes alternating 64-column overwrites when `STORE_BUF_N < EPILOGUE_SUBTILE_N/p.REDUCTION_N`. Passing the fragment index (via `store_idx`/`store_phase`) or storing a tiny `frag_off_n` metadata array indexed by `store_idx` would let helper warps compute column offsets per buffer slot and keep the writes contiguous without extra shared memory.
- **Next step to try first:** implement the metadata ring plus helper-owned SwiGLU (already tracked via `USE_HELPER_SWIGLU`), because it both addresses the column-alt issue and pushes the heavy math/stores into the spare helper warps without adding new buffers. If that glacial restructure fails, revisit the TMEM-aligned `float2` pack path.

## 7. Depth-1 ring and subtile followups

- **Depth-1 helper ring is legal now:** On the restored exact baseline, relaxing `EPILOGUE_STORE_HELPER_DEPTH` to `1` does not deadlock. It makes `USE_HELPER_PACKED_OUT_BUFFER=1` legal on the target bucket and also makes one `USE_HELPER_SWIGLU` row-16 config legal.
- **But it still loses:** The best depth-1 helper-pack path (`row8`, helper-owned packed-out buffer) runs around `0.3406 ms` on the target bucket, versus the current default helper-store path at about `0.3377-0.3391 ms`. The legal helper-owned exact SwiGLU path is much worse (`row16`, `depth1` about `0.4388 ms`).
- **Smaller N subtiles still need a real fragment-layout rewrite:** `EPILOGUE_SUBTILE_N < BLOCK_N` can be made to split the full-width packed accumulator with adjacent-linear scaffolding, but once row fragments are introduced the current Gluon `float2.pack/unpack` helpers still fail because the fragment layouts do not preserve the required last-dimension pair basis. So the smaller-subtile idea is still blocked on dedicated row-fragment adjacent-linear pack/unpack support, not on helper-ring shared-memory sizing alone.
