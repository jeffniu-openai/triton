"""
Optimizing a Fused-Gather MoE BMM1 Kernel
=========================================

This tutorial is a **standalone** Gluon optimization example built from the
current tuned MoE BMM1 fused-gather kernel. It intentionally duplicates the
live example implementation instead of importing it, so the tutorial keeps
working even if the example file changes shape later.

Use it as a practical guide for how this project was actually driven:

- define a workload and a correctness contract
- keep baseline and candidate on the same prepared inputs
- benchmark broadly before trusting a win
- use Nsight Compute, PTX, and SASS to answer a concrete question
- keep durable notes for both promoted changes and dead ends

The current kernel structure is:

.. code-block:: text

   activation loader ----.
                         |
   weight+scale loader --+--> MMA --> accumulator --> exact SwiGLU epilogue
                                                       |
                                                       v
                                              helper-store ring in SMEM
                                                       |
                                                       v
                                               store-helper partition
                                                       |
                                                       v
                                                  FP8 global output

What produced durable wins in this project:

- packed ``f32x2`` epilogue math
- packed FP8 conversion / store
- helper-store epilogue ownership
- low-batch ``BLOCK_M`` policy
- measurement discipline

What did not:

- broad schedule sweeps after the promoted banded schedule
- deleting buffers to chase occupancy
- assuming split-K was the low-batch fix
- async TMA store as a drop-in helper replacement
- source-level scalar cleanups without PTX/SASS or NCU evidence

The bottom of this file contains:

- a pytest correctness check against the production reference
- a ``triton.testing.perf_report`` batch sweep

For the full optimization history, see:

- ``.codex/initiatives/artifacts/ws-matmul-performance-report.md``
- ``.codex/initiatives/artifacts/fp8-mxfp4-fused-gather-matmul.md``
"""

from dataclasses import dataclass, replace
from itertools import chain

import pytest
import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
import triton.experimental.gluon.language.nvidia.blackwell as blackwell
import triton.experimental.gluon.language.nvidia.blackwell.tma as tma
from triton.experimental.gluon.language.nvidia.blackwell import float2
import triton.experimental.gluon.language.nvidia.hopper.mbarrier as mbarrier
import triton.language.core as tl_core
import triton.language.extra.libdevice as libdevice
from triton.language.core import _aggregate as aggregate
from triton.testing import do_bench_cudagraph

from triton_kernels.distributed import make_expt_dict_uniform
from triton_kernels.matmul import (
    FlexCtx,
    FnSpecs,
    FusedActivation,
    PrecisionConfig,
    matmul as reference_matmul,
)
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.numerics_details.mxfp import MXFP_BLOCK_SIZE, downcast_to_mxfp
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.tensor import (
    FP4,
    RaggedTensorMetadata,
    Tensor,
    convert_layout,
    make_ragged_tensor_metadata,
    wrap_torch_tensor,
)
from triton_kernels.tensor_details.dtype import UINT8
from triton_kernels.tensor_details.layout import (
    make_default_matmul_mxfp4_w_layout,
    make_default_matmul_mxfp4_w_scale_layout,
)
from triton_kernels.testing import alloc_rand, assert_close
from triton_kernels.topk import topk


# ===-----------------------------------------------------------------------===#
# Device Code
# ===-----------------------------------------------------------------------===#


@gluon.jit
def advance(idx: gl.tensor, phase: gl.tensor, num_bufs: gl.constexpr) -> tuple[gl.tensor, gl.tensor]:
    next_idx = idx + 1
    wrap = next_idx == num_bufs
    return gl.where(wrap, 0, next_idx), gl.where(wrap, phase ^ 1, phase)


@gluon.jit
def unpack_block_schedule(schedule: gl.tensor) -> tuple[gl.tensor, gl.tensor]:
    return schedule & 0xFFFF, schedule >> 16


@gluon.jit
def banded_row_major(lin_idx, m_tiles, n_tiles, BAND_N: gl.constexpr):
    full_band_tiles = m_tiles * BAND_N
    n_full_bands = n_tiles // BAND_N
    full_band_work = n_full_bands * full_band_tiles

    if lin_idx < full_band_work:
        band_id = lin_idx // full_band_tiles
        within_band = lin_idx % full_band_tiles
        return within_band // BAND_N, band_id * BAND_N + (within_band % BAND_N)

    tail_n = n_tiles - n_full_bands * BAND_N
    tail_idx = lin_idx - full_band_work
    return tail_idx // tail_n, n_full_bands * BAND_N + (tail_idx % tail_n)


@gluon.jit
def apply_block_schedule(
    block_id: gl.tensor,
    grid_m: gl.tensor,
    GRID_N: gl.constexpr,
    slice_offsets: gl.tensor,
    block_schedule: gl.tensor,
) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
    pid_mn = block_id % (grid_m * GRID_N)
    schedule_pid_m, pid_n = banded_row_major(pid_mn, grid_m, GRID_N, BAND_N=20)

    slice_idx, pid_m = unpack_block_schedule(gl.load(block_schedule + schedule_pid_m))
    slice_offset = gl.load(slice_offsets + slice_idx)

    return pid_m, pid_n, slice_idx, slice_offset


@gluon.jit
def unswizzle_mx_scale(
    smem,
    SIZE_OUTER: gl.constexpr,
    SIZE_INNER: gl.constexpr,
    MXFP_BLOCK_SIZE: gl.constexpr,
):
    rows: gl.constexpr = smem.shape[1]
    cols: gl.constexpr = smem.shape[2] * smem.shape[3] * smem.shape[4]
    tiles: gl.constexpr = cols // (SIZE_OUTER * SIZE_INNER)
    smem = smem.reshape((rows, tiles, MXFP_BLOCK_SIZE, SIZE_OUTER // MXFP_BLOCK_SIZE, SIZE_INNER))
    smem = smem.permute((0, 3, 2, 1, 4))
    return smem.reshape((rows * SIZE_OUTER, cols // SIZE_OUTER))


@gluon.jit
def alloc_barrier_ring(num_bufs: gl.constexpr):
    bars = gl.allocate_shared_memory(gl.int64, [num_bufs, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_bufs):
        mbarrier.init(bars.index(i), count=1)
    return bars


@gluon.jit
def alloc_empty_ready_barriers(num_bufs: gl.constexpr):
    return alloc_barrier_ring(num_bufs), alloc_barrier_ring(num_bufs)


@gluon.jit
def invalidate_barrier_ring(bars, num_bufs: gl.constexpr):
    for i in gl.static_range(num_bufs):
        mbarrier.invalidate(bars.index(i))


@gluon.jit
def pack_e4m3x2(values):
    return tl_core.inline_asm_elementwise(
        """
        {
            .reg .f32 lane<2>;
            mov.b64 {lane0, lane1}, $1;
            cvt.rn.satfinite.e4m3x2.f32 $0, lane1, lane0;
        }
        """,
        "=h,l",
        [values.value],
        dtype=tl_core.int16,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def pack_u16x2(x0, x1):
    return tl_core.inline_asm_elementwise(
        """
        mov.b32 $0, { $1, $2 };
        """,
        "=r,h,h",
        [x0, x1],
        dtype=tl_core.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def pack_fp8x4(values):
    lhs, rhs = gl.split(values.reshape((values.shape[0], values.shape[1] // 2, 2)))
    return pack_u16x2(lhs, rhs)


@gluon.jit
def _split_m(values):
    return gl.split(values.reshape((2, values.shape[0] // 2, values.shape[1])).permute((1, 2, 0)))


@gluon.jit
def _split_m_float2(values):
    lhs, rhs = _split_m(values.value)
    return float2.Float2Tensor(lhs), float2.Float2Tensor(rhs)


@gluon.jit
def split_m_subtiles(values, subtile_factor: gl.constexpr):
    # For epilogue subtiling.
    subtiles = (values,)
    for split_level in gl.static_range(5):
        if (1 << split_level) < subtile_factor:
            next_subtiles = ()
            for subtile_idx in gl.static_range(1 << split_level):
                lhs, rhs = _split_m_float2(subtiles[subtile_idx])
                next_subtiles += (lhs, rhs)
            subtiles = next_subtiles
    return subtiles


@aggregate
class PartitionArgs:
    x_desc: tma.tensor_descriptor
    w_desc: tma.tensor_descriptor
    scale_desc: tma.tensor_descriptor
    out_desc: tma.tensor_descriptor
    x_scale_ptr: gl.tensor | gl.constexpr
    w_scale_ptr: gl.tensor | gl.constexpr
    out_scale_ptr: gl.tensor

    out_ptr: gl.tensor
    bias_ptr: gl.tensor
    bias_stride: gl.tensor
    gather_indx_ptr: gl.tensor
    x_slice_sizes: gl.tensor
    x_slice_offs: gl.tensor
    x_block_offs: gl.tensor
    x_block_schedule: gl.tensor

    x_bufs: gl.shared_memory_descriptor
    x_empty_bars: gl.shared_memory_descriptor
    x_ready_bars: gl.shared_memory_descriptor
    x_num_bufs: gl.constexpr

    w_bufs: gl.shared_memory_descriptor
    w_scale_bufs: gl.shared_memory_descriptor
    w_empty_bars: gl.shared_memory_descriptor
    w_ready_bars: gl.shared_memory_descriptor
    w_num_bufs: gl.constexpr

    x_scale_tmem: blackwell.tensor_memory_descriptor
    w_scale_tmem: blackwell.tensor_memory_descriptor
    acc_bufs: blackwell.tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    acc_num_bufs: gl.constexpr

    store_bufs: gl.shared_memory_descriptor
    store_empty_bars: gl.shared_memory_descriptor
    store_ready_bars: gl.shared_memory_descriptor

    grid_m: gl.tensor
    GRID_N: gl.constexpr
    K_TILES: gl.constexpr
    SCALE_FLAT_N: gl.constexpr
    SCALE_BLOCK_N_DIV: gl.constexpr
    num_blocks: gl.tensor

    NUM_SMS: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    SCALE_SIZE_OUTER: gl.constexpr
    SCALE_SIZE_INNER: gl.constexpr
    MXFP_BLOCK_SIZE: gl.constexpr
    PACKED_BLOCK_K: gl.constexpr

    SWIGLU_ALPHA: gl.constexpr
    SWIGLU_LIMIT: gl.constexpr
    REDUCTION_N: gl.constexpr
    FLEXPOINT_SATURATE_INF: gl.constexpr

    EPILOGUE_ROW_SUBTILE_FACTOR: gl.constexpr
    EPILOGUE_STORE_HELPER_DEPTH: gl.constexpr

    @gluon.jit
    def apply_block_schedule(self, block_id: gl.tensor) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
        return apply_block_schedule(
            block_id=block_id,
            grid_m=self.grid_m,
            GRID_N=self.GRID_N,
            slice_offsets=self.x_slice_offs,
            block_schedule=self.x_block_schedule,
        )


@gluon.jit
def load_activations(p: PartitionArgs):
    offs_layout: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]),
    )
    tile_x_bytes: gl.constexpr = p.x_desc.block_type.nbytes * p.BLOCK_M

    idx = 0
    phase = 1

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, _, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        shape_m = gl.load(p.x_slice_sizes + slice_idx)

        offs_m = off_m + gl.arange(0, p.BLOCK_M, layout=offs_layout)
        mask_m = offs_m < shape_m
        offs_x_m = gl.load(
            p.gather_indx_ptr + slice_offset + offs_m,
            mask=mask_m,
            other=p.x_desc.shape[0],
        )

        for ki in range(p.K_TILES):
            off_k_x = ki * p.BLOCK_K

            empty_bar = p.x_empty_bars.index(idx)
            ready_bar = p.x_ready_bars.index(idx)
            x_buf = p.x_bufs.index(idx)

            mbarrier.wait(empty_bar, phase)
            mbarrier.expect(ready_bar, tile_x_bytes)
            tma.async_gather(p.x_desc, offs_x_m, off_k_x, ready_bar, x_buf)

            idx, phase = advance(idx, phase, p.x_num_bufs)


@gluon.jit
def load_weights(p: PartitionArgs):
    tile_w_bytes: gl.constexpr = p.w_desc.block_type.nbytes
    tile_scale_bytes: gl.constexpr = p.scale_desc.block_type.nbytes
    bytes_per_stage: gl.constexpr = tile_w_bytes + tile_scale_bytes

    idx = 0
    phase = 1

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        _, pid_n, slice_idx, _ = p.apply_block_schedule(block_id)
        off_n = pid_n * p.BLOCK_N

        scale_idx = slice_idx * p.SCALE_FLAT_N + pid_n * p.SCALE_BLOCK_N_DIV
        for ki in range(p.K_TILES):
            off_k_w = ki * p.PACKED_BLOCK_K
            off_k_scale = off_k_w // 64

            w_empty_bar = p.w_empty_bars.index(idx)
            w_ready_bar = p.w_ready_bars.index(idx)
            w_buf = p.w_bufs.index(idx)
            scale_buf = p.w_scale_bufs.index(idx)

            mbarrier.wait(w_empty_bar, phase)
            mbarrier.expect(w_ready_bar, bytes_per_stage)
            tma.async_copy_global_to_shared(p.w_desc, [slice_idx, off_n, off_k_w], w_ready_bar, w_buf)
            tma.async_copy_global_to_shared(p.scale_desc, [0, scale_idx, off_k_scale, 0, 0], w_ready_bar, scale_buf)

            idx, phase = advance(idx, phase, p.w_num_bufs)


@gluon.jit
def mma_partition(p: PartitionArgs):
    x_idx = 0
    x_phase = 0
    w_idx = 0
    w_phase = 0
    mma_idx = 0
    mma_phase = 1

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        acc_empty_bar = p.acc_empty_bars.index(mma_idx)
        acc_ready_bar = p.acc_ready_bars.index(mma_idx)
        acc_buf = p.acc_bufs.index(mma_idx)
        mbarrier.wait(acc_empty_bar, mma_phase)

        use_acc = False
        for _ in range(p.K_TILES):
            w_ready_bar = p.w_ready_bars.index(w_idx)
            w_empty_bar = p.w_empty_bars.index(w_idx)
            w_buf = p.w_bufs.index(w_idx)
            scale_buf = p.w_scale_bufs.index(w_idx)
            mbarrier.wait(w_ready_bar, w_phase)

            blackwell.tcgen05_copy(
                unswizzle_mx_scale(scale_buf, p.SCALE_SIZE_OUTER, p.SCALE_SIZE_INNER, p.MXFP_BLOCK_SIZE),
                p.w_scale_tmem,
            )

            x_ready_bar = p.x_ready_bars.index(x_idx)
            x_empty_bar = p.x_empty_bars.index(x_idx)
            x_buf = p.x_bufs.index(x_idx)
            mbarrier.wait(x_ready_bar, x_phase)

            blackwell.tcgen05_mma_scaled(
                w_buf.reshape((w_buf.shape[1], w_buf.shape[2])),
                x_buf.permute((1, 0)),
                acc_buf,
                p.w_scale_tmem,
                p.x_scale_tmem,
                a_type="e2m1",
                b_type="e4m3",
                use_acc=use_acc,
            )
            blackwell.tcgen05_commit(x_empty_bar)
            blackwell.tcgen05_commit(w_empty_bar)

            x_idx, x_phase = advance(x_idx, x_phase, p.x_num_bufs)
            w_idx, w_phase = advance(w_idx, w_phase, p.w_num_bufs)
            use_acc = True

        blackwell.tcgen05_commit(acc_ready_bar)
        mma_idx, mma_phase = advance(mma_idx, mma_phase, p.acc_num_bufs)


@gluon.jit
def store_packed_out(
    p: PartitionArgs,
    packed_out,
    off_m,
    out_off_n,
    shape_m,
    slice_offset,
):
    values = pack_fp8x4(packed_out)
    layout: gl.constexpr = values.type.layout
    offs_m = off_m + gl.arange(0, values.shape[0], layout=gl.SliceLayout(1, layout))
    offs_n = out_off_n // 4 + gl.arange(0, values.shape[1], layout=gl.SliceLayout(0, layout))
    mask_m = gl.expand_dims(offs_m < shape_m, 1)
    mask_n = gl.expand_dims(offs_n < (p.out_desc.shape[1] + 3) // 4, 0)
    mask = mask_m & mask_n
    ptrs = p.out_ptr.cast(gl.pointer_type(gl.int32), bitcast=True)
    ptrs = ptrs + gl.expand_dims(slice_offset + offs_m, 1) * (p.out_desc.strides[0] // 4)
    ptrs = ptrs + gl.expand_dims(offs_n, 0) * p.out_desc.strides[1]
    gl.store(ptrs, values, mask=mask)


@gluon.jit
def _swiglu_step1(acc_packed, limit):
    gelu, linear = float2.unpack2(acc_packed)
    gelu = gl.minimum(gelu.to(gl.float32), limit)
    linear = tl_core.clamp(linear.to(gl.float32), -limit, limit)
    return gelu, linear


@gluon.jit
def _swiglu_step2(gelu, linear, alpha):
    den = 1.0 + libdevice.exp(-alpha * gelu)
    activated = gelu / den
    activated_packed = float2.pack(activated, axis=1)
    linear_packed = float2.pack(linear, axis=1)
    return float2.fma(activated_packed, linear_packed, activated_packed)


@gluon.jit
def pack_fp8_out_fragment(out_packed, out_recip):
    scaled_out_packed = out_packed * float2.full_like(out_packed, out_recip)
    return pack_e4m3x2(scaled_out_packed)


@gluon.jit
def _store_out_subtile(
    p: PartitionArgs,
    out_packed,
    out_recip,
    store_idx,
    store_phase,
):
    payload = pack_fp8_out_fragment(out_packed, out_recip)
    empty_bar = p.store_empty_bars.index(store_idx)
    ready_bar = p.store_ready_bars.index(store_idx)
    mbarrier.wait(empty_bar, store_phase)
    p.store_bufs.index(store_idx).store(payload)
    mbarrier.arrive(ready_bar)
    return advance(store_idx, store_phase, p.EPILOGUE_STORE_HELPER_DEPTH)


@gluon.jit
def epilogue_overlapped_store(
    p: PartitionArgs,
    acc_packed,
    out_recip,
    store_idx,
    store_phase,
):
    gl.static_assert(p.EPILOGUE_ROW_SUBTILE_FACTOR > 1, "store helper requires row fragments")
    acc_packed_subtiles = split_m_subtiles(acc_packed, p.EPILOGUE_ROW_SUBTILE_FACTOR)

    # Software pipelined and overlapped SwiGLU with transfer to store partition.
    prepared_gelu, prepared_linear = _swiglu_step1(
        acc_packed_subtiles[0],
        p.SWIGLU_LIMIT,
    )
    ready_out_packed = acc_packed_subtiles[0]
    for frag_idx in gl.static_range(1, p.EPILOGUE_ROW_SUBTILE_FACTOR):
        cur_gelu, cur_linear = _swiglu_step1(
            acc_packed_subtiles[frag_idx],
            p.SWIGLU_LIMIT,
        )
        next_ready_out_packed = _swiglu_step2(
            prepared_gelu,
            prepared_linear,
            p.SWIGLU_ALPHA,
        )
        if frag_idx > 1:
            store_idx, store_phase = _store_out_subtile(
                p,
                ready_out_packed,
                out_recip,
                store_idx,
                store_phase,
            )
        ready_out_packed = next_ready_out_packed
        prepared_gelu = cur_gelu
        prepared_linear = cur_linear

    store_idx, store_phase = _store_out_subtile(
        p,
        ready_out_packed,
        out_recip,
        store_idx,
        store_phase,
    )
    last_out_packed = _swiglu_step2(
        prepared_gelu,
        prepared_linear,
        p.SWIGLU_ALPHA,
    )
    store_idx, store_phase = _store_out_subtile(
        p,
        last_out_packed,
        out_recip,
        store_idx,
        store_phase,
    )
    return store_idx, store_phase


@gluon.jit
def load_bias(
    p: PartitionArgs,
    idx,
    phase,
    pid_n,
    slice_idx,
    split_layout: gl.constexpr,
    bias_layout: gl.constexpr,
    acc_scale,
):
    off_n = pid_n * p.BLOCK_N
    acc_empty_bar = p.acc_empty_bars.index(idx)
    acc_ready_bar = p.acc_ready_bars.index(idx)
    acc_buf = p.acc_bufs.index(idx)
    mbarrier.wait(acc_ready_bar, phase)
    mbarrier.arrive(acc_empty_bar)
    idx, phase = advance(idx, phase, p.acc_num_bufs)

    offs_bias_n = off_n + gl.arange(0, p.BLOCK_N, layout=bias_layout)
    bias = gl.convert_layout(
        gl.expand_dims(gl.load(p.bias_ptr + slice_idx * p.bias_stride + offs_bias_n), axis=0),
        split_layout,
    )
    acc_regs = acc_buf.load().permute((1, 0))
    acc = gl.convert_layout(acc_regs, split_layout)
    acc_packed = float2.pack(acc, axis=1)
    bias_packed = float2.pack(bias, axis=1)
    bias_packed = float2.Float2Tensor(gl.convert_layout(bias_packed.value, acc_packed.value.type.layout))
    acc_packed = float2.fma(acc_packed, float2.full_like(acc_packed, acc_scale), bias_packed)
    return idx, phase, acc_packed


@gluon.jit
def epilogue_store_partition(p: PartitionArgs):
    gl.static_assert(p.EPILOGUE_ROW_SUBTILE_FACTOR > 1, "store helper requires row fragments")
    frag_rows: gl.constexpr = p.BLOCK_M // p.EPILOGUE_ROW_SUBTILE_FACTOR
    store_layout: gl.constexpr = gl.BlockedLayout(
        [frag_rows // gl.num_warps(), 2],
        [1, 32],
        [gl.num_warps(), 1],
        [1, 0],
    )
    gl.static_assert(p.EPILOGUE_STORE_HELPER_DEPTH >= 2, "store helper depth must be at least 2")

    store_idx = 0
    store_phase = 0
    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        out_off_n = (pid_n * p.BLOCK_N) // p.REDUCTION_N
        for frag_idx in gl.static_range(p.EPILOGUE_ROW_SUBTILE_FACTOR):
            frag_off_m = off_m + frag_idx * frag_rows
            ready_bar = p.store_ready_bars.index(store_idx)
            empty_bar = p.store_empty_bars.index(store_idx)
            mbarrier.wait(ready_bar, store_phase)
            packed_fp8 = p.store_bufs.index(store_idx).load(store_layout)
            mbarrier.arrive(empty_bar)
            store_packed_out(
                p,
                packed_fp8,
                frag_off_m,
                out_off_n,
                shape_m,
                slice_offset,
            )
            store_idx, store_phase = advance(store_idx, store_phase, p.EPILOGUE_STORE_HELPER_DEPTH)


@gluon.jit
def epilogue_partition(p: PartitionArgs):
    idx = 0
    phase = 0
    store_idx = 0
    store_phase = 1

    x_scale = 1.0 if p.x_scale_ptr is None else gl.load(p.x_scale_ptr)
    w_scale = 1.0 if p.w_scale_ptr is None else gl.load(p.w_scale_ptr)
    acc_scale = x_scale * w_scale
    out_recip = 1.0 / gl.load(p.out_scale_ptr)

    num_warps: gl.constexpr = gl.num_warps()
    warps_n: gl.constexpr = 2 if num_warps >= 8 and p.BLOCK_N >= 256 else 1
    split_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [1, 32],
        [num_warps // warps_n, warps_n],
        [1, 0],
    )
    bias_layout: gl.constexpr = gl.SliceLayout(0, split_layout)

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, _ = p.apply_block_schedule(block_id)
        idx, phase, acc_packed = load_bias(
            p,
            idx,
            phase,
            pid_n,
            slice_idx,
            split_layout,
            bias_layout,
            acc_scale,
        )

        store_idx, store_phase = epilogue_overlapped_store(
            p,
            acc_packed,
            out_recip,
            store_idx,
            store_phase,
        )


@gluon.jit
def ws_matmul_kernel(
    x_desc: tma.tensor_descriptor,
    w_desc: tma.tensor_descriptor,
    scale_desc: tma.tensor_descriptor,
    out_desc: tma.tensor_descriptor,
    out_ptr: gl.tensor,
    #
    bias_ptr: gl.tensor,
    bias_stride: gl.tensor,
    #
    gather_indx_ptr: gl.tensor,
    #
    x_slice_sizes: gl.tensor,
    x_slice_offs: gl.tensor,
    x_block_offs: gl.tensor,
    x_block_schedule: gl.tensor,
    #
    x_scale_ptr: gl.tensor,
    w_scale_ptr: gl.tensor,
    out_scale_ptr: gl.tensor,
    #
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    NUM_SLICES: gl.constexpr,
    #
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    REDUCTION_N: gl.constexpr,
    #
    FLEXPOINT_SATURATE_INF: gl.constexpr,
    #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_SMS: gl.constexpr,
    X_NUM_BUFS: gl.constexpr,
    W_NUM_BUFS: gl.constexpr,
    LOAD_ACTIVATION_WARPS: gl.constexpr,
    LOAD_WEIGHT_WARPS: gl.constexpr,
    MMA_WARPS: gl.constexpr,
    STORE_HELPER_WARPS: gl.constexpr,
    LOAD_ACTIVATION_REGS: gl.constexpr,
    LOAD_WEIGHT_REGS: gl.constexpr,
    MMA_REGS: gl.constexpr,
    STORE_HELPER_REGS: gl.constexpr,
    EPILOGUE_ROW_SUBTILE_FACTOR: gl.constexpr,
    EPILOGUE_STORE_HELPER_DEPTH: gl.constexpr,
    SCALE_SIZE_OUTER: gl.constexpr,
    SCALE_SIZE_INNER: gl.constexpr,
    MXFP_BLOCK_SIZE: gl.constexpr,
):
    packed_block_k: gl.constexpr = w_desc.block_type.shape[2]

    grid_m = gl.load(x_block_offs + NUM_SLICES)
    grid_n: gl.constexpr = triton.cdiv(N, BLOCK_N)
    k_tiles: gl.constexpr = triton.cdiv(K, BLOCK_K)
    scale_flat_n: gl.constexpr = N // SCALE_SIZE_OUTER
    scale_block_n_div: gl.constexpr = BLOCK_N // SCALE_SIZE_OUTER
    num_blocks = grid_m * grid_n

    scale_k: gl.constexpr = BLOCK_K // MXFP_BLOCK_SIZE
    scale_layout: gl.constexpr = blackwell.TensorMemoryScalesLayout()
    acc_layout: gl.constexpr = blackwell.TensorMemoryLayout([128, BLOCK_M], col_stride=1)

    x_num_bufs: gl.constexpr = X_NUM_BUFS
    x_bufs = gl.allocate_shared_memory(
        x_desc.dtype,
        [x_num_bufs, BLOCK_M, x_desc.block_type.shape[1]],
        x_desc.layout,
    )
    x_empty_bars, x_ready_bars = alloc_empty_ready_barriers(x_num_bufs)

    w_num_bufs: gl.constexpr = W_NUM_BUFS
    w_bufs = gl.allocate_shared_memory(
        w_desc.dtype,
        [w_num_bufs] + w_desc.block_type.shape,
        w_desc.layout,
    )
    w_scale_bufs = gl.allocate_shared_memory(
        scale_desc.dtype,
        [w_num_bufs] + scale_desc.block_type.shape,
        scale_desc.layout,
    )
    w_empty_bars, w_ready_bars = alloc_empty_ready_barriers(w_num_bufs)

    x_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_M, scale_k], scale_layout)
    w_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_N, scale_k], scale_layout)

    acc_num_bufs: gl.constexpr = 1
    acc_tmem = blackwell.allocate_tensor_memory(gl.float32, [acc_num_bufs, BLOCK_N, BLOCK_M], acc_layout)
    acc_empty_bars, acc_ready_bars = alloc_empty_ready_barriers(acc_num_bufs)

    gl.static_assert(EPILOGUE_ROW_SUBTILE_FACTOR > 1, "store helper requires row fragments")
    gl.static_assert(EPILOGUE_STORE_HELPER_DEPTH >= 2, "store helper depth must be at least 2")
    frag_rows: gl.constexpr = BLOCK_M // EPILOGUE_ROW_SUBTILE_FACTOR
    out_packed_n: gl.constexpr = BLOCK_N // REDUCTION_N // 2
    store_bufs = gl.allocate_shared_memory(
        gl.int16,
        [EPILOGUE_STORE_HELPER_DEPTH, frag_rows, out_packed_n],
        gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
    )
    store_empty_bars, store_ready_bars = alloc_empty_ready_barriers(EPILOGUE_STORE_HELPER_DEPTH)

    x_scale_tmem.store(gl.full((BLOCK_M, scale_k), 127, dtype=gl.uint8, layout=x_scale_tmem.get_reg_layout()))

    p = PartitionArgs(
        x_desc=x_desc,
        w_desc=w_desc,
        scale_desc=scale_desc,
        out_desc=out_desc,
        x_scale_ptr=x_scale_ptr,
        w_scale_ptr=w_scale_ptr,
        out_scale_ptr=out_scale_ptr,
        #
        out_ptr=out_ptr,
        bias_ptr=bias_ptr,
        bias_stride=bias_stride,
        gather_indx_ptr=gather_indx_ptr,
        x_slice_sizes=x_slice_sizes,
        x_slice_offs=x_slice_offs,
        x_block_offs=x_block_offs,
        x_block_schedule=x_block_schedule,
        #
        x_bufs=x_bufs,
        x_empty_bars=x_empty_bars,
        x_ready_bars=x_ready_bars,
        x_num_bufs=x_num_bufs,
        #
        w_bufs=w_bufs,
        w_scale_bufs=w_scale_bufs,
        w_empty_bars=w_empty_bars,
        w_ready_bars=w_ready_bars,
        w_num_bufs=w_num_bufs,
        #
        x_scale_tmem=x_scale_tmem,
        w_scale_tmem=w_scale_tmem,
        acc_bufs=acc_tmem,
        acc_empty_bars=acc_empty_bars,
        acc_ready_bars=acc_ready_bars,
        acc_num_bufs=acc_num_bufs,
        #
        store_bufs=store_bufs,
        store_empty_bars=store_empty_bars,
        store_ready_bars=store_ready_bars,
        #
        grid_m=grid_m,
        GRID_N=grid_n,
        K_TILES=k_tiles,
        SCALE_FLAT_N=scale_flat_n,
        SCALE_BLOCK_N_DIV=scale_block_n_div,
        num_blocks=num_blocks,
        #
        NUM_SMS=NUM_SMS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SCALE_SIZE_OUTER=SCALE_SIZE_OUTER,
        SCALE_SIZE_INNER=SCALE_SIZE_INNER,
        MXFP_BLOCK_SIZE=MXFP_BLOCK_SIZE,
        PACKED_BLOCK_K=packed_block_k,
        #
        SWIGLU_ALPHA=SWIGLU_ALPHA,
        SWIGLU_LIMIT=SWIGLU_LIMIT,
        REDUCTION_N=REDUCTION_N,
        FLEXPOINT_SATURATE_INF=FLEXPOINT_SATURATE_INF,
        #
        EPILOGUE_ROW_SUBTILE_FACTOR=EPILOGUE_ROW_SUBTILE_FACTOR,
        EPILOGUE_STORE_HELPER_DEPTH=EPILOGUE_STORE_HELPER_DEPTH,
    )

    gl.warp_specialize(
        [
            (epilogue_partition, (p,)),
            (epilogue_store_partition, (p,)),
            (load_activations, (p,)),
            (load_weights, (p,)),
            (mma_partition, (p,)),
        ],
        [STORE_HELPER_WARPS, LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
        [STORE_HELPER_REGS, LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
    )

    invalidate_barrier_ring(x_empty_bars, x_num_bufs)
    invalidate_barrier_ring(x_ready_bars, x_num_bufs)
    invalidate_barrier_ring(w_empty_bars, w_num_bufs)
    invalidate_barrier_ring(w_ready_bars, w_num_bufs)
    invalidate_barrier_ring(acc_empty_bars, acc_num_bufs)
    invalidate_barrier_ring(acc_ready_bars, acc_num_bufs)
    invalidate_barrier_ring(store_empty_bars, EPILOGUE_STORE_HELPER_DEPTH)
    invalidate_barrier_ring(store_ready_bars, EPILOGUE_STORE_HELPER_DEPTH)


# ===-----------------------------------------------------------------------===#
# Host Code
# ===-----------------------------------------------------------------------===#


def get_operand_layout(t: Tensor, block_shape: list[int]):
    rank = len(block_shape)
    if t.dtype == FP4:
        assert rank == 3
        return gl.NVMMASharedLayout(
            swizzle_byte_width=128,
            element_bitwidth=8,
            rank=rank,
            fp4_padded=True,
        )
    if t.dtype == UINT8:
        assert rank == 5
        return gl.NVMMASharedLayout(
            swizzle_byte_width=0,
            element_bitwidth=8,
            rank=rank,
        )
    if t.dtype == torch.float32:
        assert rank == 2
        return gl.NVMMASharedLayout.get_default_for(
            block_shape,
            torch.float32,
        )

    assert t.dtype == torch.float8_e4m3fn
    return gl.NVMMASharedLayout(
        swizzle_byte_width=128,
        element_bitwidth=8,
        rank=rank,
    )


def make_operand_descriptor(t: torch.Tensor | Tensor, block_shape: tuple[int, ...], transposed: bool = False):
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    ptr = t if isinstance(t, torch.Tensor) else t.storage.data
    shape = list(ptr.shape)
    strides = list(ptr.stride())

    if transposed:
        shape[-1], shape[-2] = shape[-2], shape[-1]
        strides[-1], strides[-2] = strides[-2], strides[-1]

    layout = get_operand_layout(t, list(block_shape))
    return TensorDescriptor(ptr, shape, strides, block_shape, layout)


@dataclass(frozen=True, slots=True)
class KernelConfig:
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 128
    X_NUM_BUFS: int = 5
    W_NUM_BUFS: int = 4
    LOAD_ACTIVATION_WARPS: int = 4
    LOAD_WEIGHT_WARPS: int = 1
    MMA_WARPS: int = 1
    STORE_HELPER_WARPS: int = 2
    EPILOGUE_ROW_SUBTILE_FACTOR: int = 8
    EPILOGUE_STORE_HELPER_DEPTH: int = 2
    LOAD_ACTIVATION_REGS: int = 112
    LOAD_WEIGHT_REGS: int = 48
    MMA_REGS: int = 24
    STORE_HELPER_REGS: int = 16


def estimated_slice_size(ragged_metadata: RaggedTensorMetadata, m: int) -> int:
    if ragged_metadata.expected_slice_size is not None:
        return ragged_metadata.expected_slice_size
    return max(1, m // ragged_metadata.n_slices)


def select_kernel_config(ragged_metadata: RaggedTensorMetadata, m: int) -> KernelConfig:
    slice_size = estimated_slice_size(ragged_metadata, m)
    config = KernelConfig()
    if slice_size <= 8:
        return replace(config, BLOCK_M=16, EPILOGUE_ROW_SUBTILE_FACTOR=2)
    if slice_size <= 16:
        return replace(config, BLOCK_M=32, EPILOGUE_ROW_SUBTILE_FACTOR=4)
    if slice_size <= 58:
        return replace(config, BLOCK_M=64, EPILOGUE_ROW_SUBTILE_FACTOR=4)
    return config


def matmul(
    a: torch.Tensor,
    b: torch.Tensor | Tensor,
    bias: torch.Tensor,
    a_ragged_metadata: RaggedTensorMetadata,
    gather_indx: torch.Tensor,
    precision_config: PrecisionConfig,
    c: torch.Tensor,
    fused_activation: FusedActivation,
):
    specs = fused_activation.specs
    assert specs.name == "swiglu"
    reduction_n = specs.reduction_n
    swiglu_alpha, swiglu_limit = fused_activation.fn_args

    b_mx_scales = precision_config.b_mx_scale

    out_dtype = precision_config.out_dtype
    assert out_dtype is not None

    assert c.ndim == 2

    flex_ctx = precision_config.flex_ctx

    assert a.ndim == 2
    _, k = a.shape
    _, _, n = b.shape
    m = gather_indx.shape[0]

    config = select_kernel_config(a_ragged_metadata, m)

    mxfp_block_size = 32
    scale_size_outer = 128
    scale_size_inner = 4
    x_block_idx = config.BLOCK_M.bit_length() - 5

    expected_grid_m = a_ragged_metadata.n_blocks(a_ragged_metadata.n_slices, m, config.BLOCK_M)
    grid_n = triton.cdiv(n, config.BLOCK_N)
    sms = torch.cuda.get_device_properties(bias.device).multi_processor_count
    launch_grid = max(1, min(sms, expected_grid_m * grid_n))
    grid = (launch_grid,)

    x_desc = make_operand_descriptor(a, (1, config.BLOCK_K))
    w_desc = make_operand_descriptor(b, (1, config.BLOCK_N, config.BLOCK_K // 2), transposed=True)
    scale_desc = make_operand_descriptor(
        b_mx_scales,
        (
            1,
            config.BLOCK_N // scale_size_outer,
            config.BLOCK_K // mxfp_block_size // scale_size_inner,
            2,
            256,
        ),
    )
    out_desc = make_operand_descriptor(c, (config.BLOCK_M, config.BLOCK_N // reduction_n))

    ws_matmul_kernel[grid](
        x_desc=x_desc,
        w_desc=w_desc,
        scale_desc=scale_desc,
        out_desc=out_desc,
        out_ptr=c,
        #
        bias_ptr=bias,
        bias_stride=bias.stride(0),
        #
        gather_indx_ptr=gather_indx,
        #
        x_slice_sizes=a_ragged_metadata.slice_sizes,
        x_slice_offs=a_ragged_metadata.slice_offs,
        x_block_offs=a_ragged_metadata.block_offs_data[x_block_idx],
        x_block_schedule=a_ragged_metadata.block_schedule_data[x_block_idx],
        #
        x_scale_ptr=flex_ctx.lhs_data.scale,
        w_scale_ptr=flex_ctx.rhs_data.scale,
        out_scale_ptr=flex_ctx.out_data.expected_scale,
        #
        M=m,
        N=n,
        K=k,
        NUM_SLICES=a_ragged_metadata.n_slices,
        #
        SWIGLU_ALPHA=swiglu_alpha,
        SWIGLU_LIMIT=swiglu_limit,
        REDUCTION_N=reduction_n,
        #
        FLEXPOINT_SATURATE_INF=precision_config.flexpoint_saturate_inf,
        #
        BLOCK_M=config.BLOCK_M,
        BLOCK_N=config.BLOCK_N,
        BLOCK_K=config.BLOCK_K,
        NUM_SMS=launch_grid,
        X_NUM_BUFS=config.X_NUM_BUFS,
        W_NUM_BUFS=config.W_NUM_BUFS,
        LOAD_ACTIVATION_WARPS=config.LOAD_ACTIVATION_WARPS,
        LOAD_WEIGHT_WARPS=config.LOAD_WEIGHT_WARPS,
        MMA_WARPS=config.MMA_WARPS,
        STORE_HELPER_WARPS=config.STORE_HELPER_WARPS,
        LOAD_ACTIVATION_REGS=config.LOAD_ACTIVATION_REGS,
        LOAD_WEIGHT_REGS=config.LOAD_WEIGHT_REGS,
        MMA_REGS=config.MMA_REGS,
        STORE_HELPER_REGS=config.STORE_HELPER_REGS,
        EPILOGUE_ROW_SUBTILE_FACTOR=config.EPILOGUE_ROW_SUBTILE_FACTOR,
        EPILOGUE_STORE_HELPER_DEPTH=config.EPILOGUE_STORE_HELPER_DEPTH,
        SCALE_SIZE_OUTER=scale_size_outer,
        SCALE_SIZE_INNER=scale_size_inner,
        MXFP_BLOCK_SIZE=mxfp_block_size,
        #
        num_warps=8,
    )

    return c


# ===-----------------------------------------------------------------------===#
# Benchmark and Testing Helpers
# ===-----------------------------------------------------------------------===#

GPT_OSS_120B_NUM_EXPERTS = 128
GPT_OSS_120B_EXPERTS_PER_TOKEN = 4
GPT_OSS_120B_NUM_EXPERT_SHARDS = 8
GPT_OSS_120B_LOCAL_RANK = 0
GPT_OSS_120B_HIDDEN_SIZE = 2880
GPT_OSS_120B_INTERMEDIATE_SIZE = 2880
GPT_OSS_120B_MM1_SHAPE = (GPT_OSS_120B_HIDDEN_SIZE, 2 * GPT_OSS_120B_INTERMEDIATE_SIZE)
GPT_OSS_120B_BATCH_PER_EXPERT = tuple(
    chain.from_iterable(range(2 ** (2 + k), 2 ** (3 + k), min(2**k, 32)) for k in range(8))
)
GPT_OSS_120B_BATCH_SIZES = tuple(
    batch_per_expert * GPT_OSS_120B_NUM_EXPERTS // GPT_OSS_120B_EXPERTS_PER_TOKEN
    for batch_per_expert in GPT_OSS_120B_BATCH_PER_EXPERT
)

OUTPUT_MAXTOL = 0.126
OUTPUT_RMSTOL = 1e-4
SCALE_TOL = 1e-10


@dataclass(frozen=True, slots=True)
class PreparedCase:
    batch_size: int
    local_rank: int
    x: torch.Tensor
    w: Tensor
    w_scale: Tensor
    bias: torch.Tensor
    ragged_metadata: RaggedTensorMetadata
    gather_indx: torch.Tensor
    fused_activation: FusedActivation
    x_scale: torch.Tensor
    y_scale: torch.Tensor
    out_shape: tuple[int, int]
    out_dtype: torch.dtype


def alloc_randn(shape: tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
    if dtype.itemsize == 1:
        return alloc_rand(shape, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def alloc_randn_fp4(shape: tuple[int, ...], device: str) -> tuple[Tensor, Tensor]:
    data = alloc_randn(shape, torch.bfloat16, device)
    data, scale = downcast_to_mxfp(data, FP4, axis=1)  # type: ignore[arg-type]
    data_layout = make_default_matmul_mxfp4_w_layout(mx_axis=1)
    scale_layout = make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=8)
    data = convert_layout(wrap_torch_tensor(data, dtype=FP4), data_layout)
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout)
    return data, scale


def init_routing_data(batch_size: int, local_rank: int, device: str) -> tuple[RaggedTensorMetadata, torch.Tensor]:
    expt_dist = make_expt_dict_uniform(GPT_OSS_120B_NUM_EXPERT_SHARDS, GPT_OSS_120B_NUM_EXPERTS)
    logits = torch.randn((batch_size, GPT_OSS_120B_NUM_EXPERTS), dtype=torch.float16, device=device)
    sparse_logits = topk(logits, GPT_OSS_120B_EXPERTS_PER_TOKEN, apply_softmax=True)
    expt_hist = sparse_logits.mask_metadata.col_sum

    local_expts = expt_dist[local_rank]
    local_expts_hist = expt_hist[local_expts]
    if local_expts_hist.sum() == 0:
        local_expts_hist[torch.randint(0, len(local_expts_hist), size=())] = 1

    ragged_metadata = make_ragged_tensor_metadata(local_expts_hist, batch_size * GPT_OSS_120B_EXPERTS_PER_TOKEN)
    ragged_metadata.expected_slice_size = batch_size * GPT_OSS_120B_EXPERTS_PER_TOKEN // GPT_OSS_120B_NUM_EXPERTS
    combine_indx = sparse_logits.mask_metadata.col_sorted_indx
    gather_indx = torch.div(combine_indx, GPT_OSS_120B_EXPERTS_PER_TOKEN, rounding_mode="trunc")
    return ragged_metadata, gather_indx


def prepare_case(batch_size: int, device: str, seed: int = 0) -> PreparedCase:
    torch.manual_seed(seed)

    local_rank = GPT_OSS_120B_LOCAL_RANK
    k, n = GPT_OSS_120B_MM1_SHAPE
    n_expts_local = GPT_OSS_120B_NUM_EXPERTS // GPT_OSS_120B_NUM_EXPERT_SHARDS
    ragged_metadata, gather_indx = init_routing_data(batch_size, local_rank, device)
    x = alloc_randn((batch_size, k), dtype=torch.float8_e4m3fn, device=device)
    w, w_scale = alloc_randn_fp4((n_expts_local, k, n), device=device)
    bias = alloc_randn((n_expts_local, n), dtype=torch.float32, device=device)

    swiglu_alpha = float(torch.rand((), device=device).item()) / 5 + 1.0
    swiglu_limit = float(torch.rand((), device=device).item()) / 5 + 1.3
    fused_activation = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (swiglu_alpha, swiglu_limit),
    )

    x_scale = (torch.rand((), device=device) + 0.5).reshape(1)
    y_scale = (torch.rand((), device=device) + 3.5).reshape(1)
    return PreparedCase(
        batch_size=batch_size,
        local_rank=local_rank,
        x=x,
        w=w,
        w_scale=w_scale,
        bias=bias,
        ragged_metadata=ragged_metadata,
        gather_indx=gather_indx,
        fused_activation=fused_activation,
        x_scale=x_scale,
        y_scale=y_scale,
        out_shape=(batch_size * GPT_OSS_120B_EXPERTS_PER_TOKEN, n // fused_activation.specs.reduction_n),
        out_dtype=torch.float8_e4m3fn,
    )


def make_precision_config(prepared: PreparedCase) -> PrecisionConfig:
    return PrecisionConfig(
        flexpoint_saturate_inf=True,
        b_mx_scale=prepared.w_scale,
        b_microblock_size=MXFP_BLOCK_SIZE.value,
        out_dtype=prepared.out_dtype,
        flex_ctx=FlexCtx(
            lhs_data=InFlexData(dtype=prepared.out_dtype, scale=prepared.x_scale),
            rhs_data=InFlexData(),
            out_data=OutFlexData(dtype=prepared.out_dtype, expected_scale=prepared.y_scale),
        ),
    )


def make_output_buffer(prepared: PreparedCase) -> torch.Tensor:
    return torch.zeros(prepared.out_shape, dtype=prepared.out_dtype, device=prepared.x.device)


def run_kernel(prepared: PreparedCase, kernel, precision_config: PrecisionConfig, out: torch.Tensor) -> torch.Tensor:
    return kernel(
        a=prepared.x,
        b=prepared.w,
        bias=prepared.bias,
        a_ragged_metadata=prepared.ragged_metadata,
        gather_indx=prepared.gather_indx,
        precision_config=precision_config,
        c=out,
        fused_activation=prepared.fused_activation,
    )

def run_provider(prepared: PreparedCase, provider: str) -> tuple[torch.Tensor, PrecisionConfig]:
    precision_config = make_precision_config(prepared)
    kernel = matmul if provider == "example" else reference_matmul
    y = run_kernel(prepared, kernel, precision_config, make_output_buffer(prepared))
    return y, precision_config


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


def is_blackwell():
    return triton.runtime.driver.active.get_current_target().backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.parametrize("batch_size", [128, 1536, 2048])
@pytest.mark.skipif(not is_blackwell(), reason="Gluon MoE BMM1 fused-gather is only supported on Blackwell GPUs")
def test_op(batch_size):
    prepared = prepare_case(batch_size, device=f"cuda:{torch.cuda.current_device()}", seed=0)
    ref_y, ref_precision = run_provider(prepared, "reference")
    cand_y, cand_precision = run_provider(prepared, "example")
    description = f"gpt-oss-120b-mm1-bs{prepared.batch_size}:example"
    assert_close(
        ref_y.to(torch.float32),
        cand_y.to(torch.float32),
        maxtol=OUTPUT_MAXTOL,
        rmstol=OUTPUT_RMSTOL,
        description=f"{description}:out",
        verbose=False,
    )
    ref_scale = ref_precision.flex_ctx.out_data.actual_scale
    cand_scale = cand_precision.flex_ctx.out_data.actual_scale
    if ref_scale is not None or cand_scale is not None:
        assert ref_scale is not None and cand_scale is not None
        assert_close(
            ref_scale.to(torch.float32),
            cand_scale.to(torch.float32),
            maxtol=SCALE_TOL,
            rmstol=SCALE_TOL,
            description=f"{description}:out_scale",
            verbose=False,
        )


# ===-----------------------------------------------------------------------===#
# Benchmarking
# ===-----------------------------------------------------------------------===#

providers = ["example", "reference"]
bench_configs = [
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=GPT_OSS_120B_BATCH_SIZES,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[("red", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name=(
            "GPT-OSS-120B MoE MM1 "
            f"E={GPT_OSS_120B_NUM_EXPERTS} "
            f"EP={GPT_OSS_120B_NUM_EXPERT_SHARDS} "
            f"B={GPT_OSS_120B_MM1_SHAPE[0]}x{GPT_OSS_120B_MM1_SHAPE[1]}"
        ),
        args={},
    )
]


@triton.testing.perf_report(bench_configs)
def bench(batch_size, provider):
    prepared = prepare_case(batch_size, device=f"cuda:{torch.cuda.current_device()}", seed=0)
    precision_config = make_precision_config(prepared)
    kernel = matmul if provider == "example" else reference_matmul
    out = make_output_buffer(prepared)

    def run() -> torch.Tensor:
        return run_kernel(prepared, kernel, precision_config, out)

    ms = do_bench_cudagraph(run)
    n_tokens = int(prepared.ragged_metadata.slice_sizes.sum().item())
    k, n = GPT_OSS_120B_MM1_SHAPE
    flops = 2 * n_tokens * k * n
    return flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench.run(save_path=".", print_data=True)
