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
import triton.language as tl
from triton.language.core import _aggregate as aggregate
from triton.testing import do_bench_cudagraph

from triton_kernels.distributed import make_expt_dict_uniform
from triton_kernels.matmul import FlexCtx, FusedComm, PrecisionConfig
from triton_kernels.matmul import matmul as reference_matmul
from triton_kernels.meta import Closure
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.numerics_details.mxfp import MXFP_BLOCK_SIZE, downcast_to_mxfp
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
    BlackwellMX4ValueShuffledLayout,
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
    BAND_N: gl.constexpr,
) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
    pid_mn = block_id % (grid_m * GRID_N)
    schedule_pid_m, pid_n = banded_row_major(pid_mn, grid_m, GRID_N, BAND_N=BAND_N)

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
def alloc_barrier_ring(num_bufs: gl.constexpr, two_ctas: gl.constexpr = False):
    bars = mbarrier.allocate_mbarrier(batch=num_bufs, two_ctas=two_ctas)
    for i in gl.static_range(num_bufs):
        mbarrier.init(bars.index(i), count=1)
    return bars


@gluon.jit
def alloc_ring_barriers(
    num_bufs: gl.constexpr,
    producer_two_ctas: gl.constexpr = False,
    consumer_two_ctas: gl.constexpr = False,
):
    return (
        alloc_barrier_ring(num_bufs, two_ctas=producer_two_ctas),
        alloc_barrier_ring(num_bufs, two_ctas=consumer_two_ctas),
    )


@gluon.jit
def invalidate_barrier_ring(bars, num_bufs: gl.constexpr):
    for i in gl.static_range(num_bufs):
        mbarrier.invalidate(bars.index(i))


@gluon.jit
def fma_f32(a, b, c):
    return gl.inline_asm_elementwise(
        """
        fma.rn.f32 $0, $1, $2, $3;
        """,
        "=f,f,f,f",
        [a, b, c],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def pack_e4m3x2(values):
    return gl.inline_asm_elementwise(
        """
        {
            .reg .f32 lane<2>;
            mov.b64 {lane0, lane1}, $1;
            cvt.rn.satfinite.e4m3x2.f32 $0, lane1, lane0;
        }
        """,
        "=h,l",
        [values.value],
        dtype=gl.int16,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def pack_u16x2(x0, x1):
    return gl.inline_asm_elementwise(
        """
        mov.b32 $0, { $1, $2 };
        """,
        "=r,h,h",
        [x0, x1],
        dtype=gl.int32,
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
def split_m_subtiles(values, subtile_factor: gl.constexpr):
    subtiles = (values,)
    for split_level in gl.static_range(5):
        if (1 << split_level) < subtile_factor:
            next_subtiles = ()
            for subtile_idx in gl.static_range(1 << split_level):
                lhs, rhs = _split_m(subtiles[subtile_idx])
                next_subtiles += (lhs, rhs)
            subtiles = next_subtiles
    return subtiles


@gluon.jit
def _split_m_float2(values):
    lhs, rhs = _split_m(values.value)
    return float2.Float2Tensor(lhs), float2.Float2Tensor(rhs)


@gluon.jit
def split_m_subtiles_float2(values, subtile_factor: gl.constexpr):
    subtiles = (values,)
    for split_level in gl.static_range(5):
        if (1 << split_level) < subtile_factor:
            next_subtiles = ()
            for subtile_idx in gl.static_range(1 << split_level):
                lhs, rhs = _split_m_float2(subtiles[subtile_idx])
                next_subtiles += (lhs, rhs)
            subtiles = next_subtiles
    return subtiles


@gluon.jit
def gluon_threadfence_system():
    return gl.inline_asm_elementwise(
        "mov.u32 $0, 0x0; fence.sc.sys;",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def gluon_set_locks(p_locks, n_shards: gl.constexpr):
    my_lock = gl.load(p_locks).cast(gl.pointer_type(gl.uint32), bitcast=True)
    my_grid_ptr = my_lock + 2

    gl.barrier()
    grid_done = gl.atomic_add(my_grid_ptr, val=1, sem="relaxed", scope="gpu")

    if grid_done + 1 == gl.num_programs(0) * gl.num_programs(1) * gl.num_programs(2):
        gluon_threadfence_system()

        shard_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0], cga_layout=((0,),))
        shard_id = gl.arange(0, n_shards, layout=shard_layout)
        peer_begin_ptr = gl.load(p_locks + shard_id).cast(gl.pointer_type(gl.uint32), bitcast=True)
        gl.atomic_add(peer_begin_ptr, val=1, mask=shard_id != 0, sem="release", scope="sys")

        gl.store(my_grid_ptr, 0, cache_modifier=".cg")


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
    out_handles: gl.tensor
    bias_ptr: gl.tensor
    bias_stride: gl.tensor
    scatter_indx_ptr: gl.tensor
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

    grid_m: gl.tensor
    GRID_N: gl.constexpr
    K_TILES: gl.constexpr
    SCALE_FLAT_N: gl.constexpr
    SCALE_BLOCK_N_DIV: gl.constexpr
    num_blocks: gl.tensor

    NUM_SMS: gl.constexpr
    USE_2CTA: gl.constexpr
    BLOCK_M_PER_CTA: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    SCALE_SIZE_OUTER: gl.constexpr
    SCALE_SIZE_INNER: gl.constexpr
    MXFP_BLOCK_SIZE: gl.constexpr
    PACKED_BLOCK_K: gl.constexpr

    REDUCE_RANK: gl.constexpr
    N_REDUCE_SHARDS: gl.constexpr
    FLEXPOINT_SATURATE_INF: gl.constexpr

    EPILOGUE_ROW_SUBTILE_FACTOR: gl.constexpr
    EPILOGUE_STORE_HELPER_DEPTH: gl.constexpr
    BAND_N: gl.constexpr
    X_GATHER_MULTICAST: gl.constexpr
    W_SCALE_MULTICAST: gl.constexpr

    @gluon.jit
    def apply_block_schedule(self, block_id: gl.tensor) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
        return apply_block_schedule(
            block_id=block_id,
            grid_m=self.grid_m,
            GRID_N=self.GRID_N,
            slice_offsets=self.x_slice_offs,
            block_schedule=self.x_block_schedule,
            BAND_N=self.BAND_N,
        )


@gluon.jit
def load_activations(p: PartitionArgs):
    tile_x_bytes: gl.constexpr = p.x_desc.nbytes_per_cta

    idx = 0
    phase = 1
    issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, _, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        ragged_billion: gl.constexpr = 0x40000000
        ragged_c0: gl.constexpr = ragged_billion
        ragged_c1 = slice_offset + shape_m
        ragged_c2 = ragged_billion - shape_m + off_m

        for ki in range(p.K_TILES):
            off_k_x = ki * p.BLOCK_K

            empty_bar = p.x_empty_bars.index(idx)
            ready_bar = p.x_ready_bars.index(idx)
            x_buf = p.x_bufs.index(idx)

            mbarrier.wait(empty_bar, phase, pred=issued >= p.x_num_bufs)
            mbarrier.expect(ready_bar, tile_x_bytes)
            tma.async_copy_global_to_shared(
                p.x_desc,
                [ragged_c0, ragged_c1, ragged_c2, off_k_x],
                ready_bar,
                x_buf,
                multicast=p.USE_2CTA and p.X_GATHER_MULTICAST,
            )

            idx, phase = advance(idx, phase, p.x_num_bufs)
            issued += 1


@gluon.jit
def load_weights(p: PartitionArgs):
    tile_w_bytes: gl.constexpr = p.w_desc.nbytes_per_cta
    tile_scale_bytes: gl.constexpr = p.scale_desc.nbytes_per_cta
    bytes_per_stage: gl.constexpr = tile_w_bytes + tile_scale_bytes

    idx = 0
    phase = 1
    issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        _, pid_n, slice_idx, _ = p.apply_block_schedule(block_id)

        scale_idx = slice_idx * p.SCALE_FLAT_N + pid_n * p.SCALE_BLOCK_N_DIV
        for ki in range(p.K_TILES):
            off_k_scale = ki * p.BLOCK_K // (p.MXFP_BLOCK_SIZE * p.SCALE_SIZE_INNER)

            w_empty_bar = p.w_empty_bars.index(idx)
            w_ready_bar = p.w_ready_bars.index(idx)
            w_buf = p.w_bufs.index(idx)
            scale_buf = p.w_scale_bufs.index(idx)

            mbarrier.wait(w_empty_bar, phase, pred=issued >= p.w_num_bufs)
            mbarrier.expect(w_ready_bar, bytes_per_stage)
            tma.async_copy_global_to_shared(p.w_desc, [slice_idx, ki, pid_n, 0, 0], w_ready_bar, w_buf)
            tma.async_copy_global_to_shared(
                p.scale_desc,
                [0, scale_idx, off_k_scale, 0, 0],
                w_ready_bar,
                scale_buf,
                multicast=p.USE_2CTA and p.W_SCALE_MULTICAST,
            )

            idx, phase = advance(idx, phase, p.w_num_bufs)
            issued += 1


@gluon.jit
def mma_partition(p: PartitionArgs):
    x_idx = 0
    x_phase = 0
    w_idx = 0
    w_phase = 0
    mma_idx = 0
    mma_phase = 1

    for _ in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
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
            x_buf = p.x_bufs.index(x_idx).reshape((p.BLOCK_M, p.BLOCK_K))
            mbarrier.wait(x_ready_bar, x_phase)

            blackwell.tcgen05_mma_scaled(
                w_buf.reshape((p.BLOCK_N, p.BLOCK_K // 2)),
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
def pack_fp8_out_fragment(out_packed, out_recip):
    scaled_out_packed = out_packed * float2.full_like(out_packed, out_recip)
    return pack_e4m3x2(scaled_out_packed)


@gluon.jit
def get_store_layout(p: PartitionArgs):
    frag_rows: gl.constexpr = p.BLOCK_M // p.EPILOGUE_ROW_SUBTILE_FACTOR
    local_cga_layout: gl.constexpr = ((0, 1), ) if p.USE_2CTA else ()
    return gl.BlockedLayout(
        [frag_rows // gl.num_warps(), 2],
        [1, 32],
        [gl.num_warps(), 1],
        [1, 0],
        cga_layout=local_cga_layout,
    )


@gluon.jit
def epilogue_direct_store(
    p: PartitionArgs,
    map_dst_coord,
    out_packed,
    out_recip,
    off_m,
    out_off_n,
    shape_m,
    slice_offset,
    store_layout: gl.constexpr,
):
    gl.static_assert(p.EPILOGUE_ROW_SUBTILE_FACTOR > 1, "direct store uses row fragments")
    frag_rows: gl.constexpr = p.BLOCK_M // p.EPILOGUE_ROW_SUBTILE_FACTOR
    out_packed_subtiles = split_m_subtiles_float2(out_packed, p.EPILOGUE_ROW_SUBTILE_FACTOR)
    for frag_idx in gl.static_range(p.EPILOGUE_ROW_SUBTILE_FACTOR):
        frag_off_m = off_m + frag_idx * frag_rows
        if frag_off_m < shape_m:
            packed_fp8 = gl.convert_layout(pack_fp8_out_fragment(out_packed_subtiles[frag_idx], out_recip), store_layout)
            store_packed_out(
                p,
                map_dst_coord,
                packed_fp8,
                frag_off_m,
                out_off_n,
                shape_m,
                slice_offset,
            )


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
    offs_bias_n = off_n + gl.arange(0, p.BLOCK_N, layout=bias_layout)
    bias = gl.convert_layout(
        gl.expand_dims(gl.load(p.bias_ptr + slice_idx * p.bias_stride + offs_bias_n), axis=0),
        split_layout,
    )
    mbarrier.wait(acc_ready_bar, phase)
    acc_regs = acc_buf.load().permute((1, 0))
    mbarrier.arrive(acc_empty_bar)
    idx, phase = advance(idx, phase, p.acc_num_bufs)
    acc = gl.convert_layout(acc_regs, split_layout)
    acc_packed = float2.pack(acc, axis=1)
    bias_packed = float2.pack(bias, axis=1)
    bias_packed = float2.Float2Tensor(gl.convert_layout(bias_packed.value, acc_packed.value.type.layout))
    acc_packed = float2.fma(acc_packed, float2.full_like(acc_packed, acc_scale), bias_packed)
    return idx, phase, acc_packed


@gluon.jit
def store_packed_out(
    p: PartitionArgs,
    map_dst_coord,
    packed_out,
    off_m,
    out_off_n,
    shape_m,
    slice_offset,
):
    values = pack_fp8x4(packed_out)
    layout: gl.constexpr = values.type.layout
    offs_m = off_m + gl.arange(0, values.shape[0], layout=gl.SliceLayout(1, layout))
    offs_n_i32 = out_off_n // 4 + gl.arange(0, values.shape[1], layout=gl.SliceLayout(0, layout))
    offs_n = out_off_n + 4 * gl.arange(0, values.shape[1], layout=gl.SliceLayout(0, layout))

    scatter_idx = gl.load(
        p.scatter_indx_ptr + slice_offset + offs_m,
        mask=offs_m < shape_m,
        other=-1,
    )
    mask_m = (offs_m < shape_m) & (scatter_idx != -1)
    dst_shard_idx, dst_m, dst_n = map_dst_coord.fn(
        None,
        scatter_idx,
        out_off_n,
        offs_n,
        *map_dst_coord.captured,
    )

    if p.out_desc.shape[2] % p.BLOCK_N == 0:
        mask_n = offs_n_i32 >= 0
    else:
        mask_n = offs_n_i32 < (p.out_desc.shape[2] + 3) // 4
    mask = gl.expand_dims(mask_m, 1) & gl.expand_dims(mask_n, 0)

    for i in gl.static_range(p.N_REDUCE_SHARDS):
        if dst_shard_idx is not None:
            peer = dst_shard_idx * p.N_REDUCE_SHARDS + (p.REDUCE_RANK + i) % p.N_REDUCE_SHARDS
        else:
            peer = (p.REDUCE_RANK + i) % p.N_REDUCE_SHARDS
        peer_handles = gl.load(p.out_handles + peer, mask=gl.expand_dims(mask_m, 1), other=0)
        ptrs = peer_handles.cast(gl.pointer_type(gl.int32), bitcast=True)
        row_stride_i32: gl.constexpr = p.out_desc.shape[2] * p.N_REDUCE_SHARDS // 4
        reduce_offset_i32: gl.constexpr = ((p.REDUCE_RANK + i) % p.N_REDUCE_SHARDS) * p.out_desc.shape[2] // 4
        ptrs = ptrs + gl.expand_dims(dst_m, 1) * row_stride_i32
        ptrs = ptrs + reduce_offset_i32
        ptrs = ptrs + gl.expand_dims(dst_n // 4, 0)
        gl.store(ptrs, values, mask=mask)


@gluon.jit
def epilogue_partition(p: PartitionArgs, map_dst_coord):
    idx = 0
    phase = 0
    x_scale = 1.0 if p.x_scale_ptr is None else gl.load(p.x_scale_ptr)
    w_scale = 1.0 if p.w_scale_ptr is None else gl.load(p.w_scale_ptr)
    acc_scale = x_scale * w_scale
    out_recip = 1.0 / gl.load(p.out_scale_ptr)

    num_warps: gl.constexpr = gl.num_warps()
    warps_n: gl.constexpr = 2 if num_warps >= 8 and p.BLOCK_N >= 256 else 1
    split_cga_layout: gl.constexpr = ((0, 1), ) if p.USE_2CTA else ()
    split_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [1, 32],
        [num_warps // warps_n, warps_n],
        [1, 0],
        cga_layout=split_cga_layout,
    )
    bias_layout: gl.constexpr = gl.SliceLayout(0, split_layout)
    store_layout: gl.constexpr = get_store_layout(p)

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        out_off_n = pid_n * p.BLOCK_N
        idx, phase, out = load_bias(
            p,
            idx,
            phase,
            pid_n,
            slice_idx,
            split_layout,
            bias_layout,
            acc_scale,
        )

        epilogue_direct_store(
            p,
            map_dst_coord,
            out,
            out_recip,
            off_m,
            out_off_n,
            shape_m,
            slice_offset,
            store_layout,
        )


@gluon.jit
def ws_matmul_kernel(
    x_desc: tma.tensor_descriptor,
    w_desc: tma.tensor_descriptor,
    scale_desc: tma.tensor_descriptor,
    out_desc: tma.tensor_descriptor,
    out_ptr: gl.tensor,
    out_handles: gl.tensor,
    map_dst_coord,
    all_writes_issued,
    #
    bias_ptr: gl.tensor,
    bias_stride: gl.tensor,
    #
    scatter_indx_ptr: gl.tensor,
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
    REDUCE_RANK: gl.constexpr,
    N_REDUCE_SHARDS: gl.constexpr,
    FLEXPOINT_SATURATE_INF: gl.constexpr,
    #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_SMS: gl.constexpr,
    X_NUM_BUFS: gl.constexpr,
    W_NUM_BUFS: gl.constexpr,
    ACC_NUM_BUFS: gl.constexpr,
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
    BAND_N: gl.constexpr,
    X_GATHER_MULTICAST: gl.constexpr,
    W_SCALE_MULTICAST: gl.constexpr,
    SCALE_SIZE_OUTER: gl.constexpr,
    SCALE_SIZE_INNER: gl.constexpr,
    MXFP_BLOCK_SIZE: gl.constexpr,
):
    use_2cta: gl.constexpr = gl.num_ctas() > 1
    gl.static_assert(gl.num_ctas() == 2, "MM2 kernel is specialized for 2 CTAs")
    gl.static_assert(BLOCK_N >= 256, "2CTA path requires at least 128 columns per CTA")

    grid_m = gl.load(x_block_offs + NUM_SLICES)
    grid_n: gl.constexpr = triton.cdiv(N, BLOCK_N)
    k_tiles: gl.constexpr = triton.cdiv(K, BLOCK_K)
    scale_flat_n: gl.constexpr = N // SCALE_SIZE_OUTER
    scale_block_n_div: gl.constexpr = BLOCK_N // SCALE_SIZE_OUTER
    num_blocks = grid_m * grid_n

    scale_k: gl.constexpr = BLOCK_K // MXFP_BLOCK_SIZE
    block_m_per_cta: gl.constexpr = BLOCK_M // gl.num_ctas()
    x_scale_layout: gl.constexpr = blackwell.TensorMemoryScalesLayout(cga_layout=((0, 0), ) if use_2cta else ())
    w_scale_layout: gl.constexpr = blackwell.TensorMemoryScalesLayout(cga_layout=((1, 0), ) if use_2cta else ())
    mma_block_col: gl.constexpr = min(128, BLOCK_N // gl.num_ctas())
    acc_layout: gl.constexpr = blackwell.TensorMemoryLayout(
        [mma_block_col, BLOCK_M],
        col_stride=1,
        cga_layout=((1, 0), ) if use_2cta else (),
        two_ctas=use_2cta,
    )

    x_num_bufs: gl.constexpr = X_NUM_BUFS
    x_bufs = gl.allocate_shared_memory(
        x_desc.dtype,
        [x_num_bufs] + x_desc.block_type.shape,
        x_desc.layout,
    )
    x_empty_bars, x_ready_bars = alloc_ring_barriers(x_num_bufs, consumer_two_ctas=use_2cta)

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
    w_empty_bars, w_ready_bars = alloc_ring_barriers(w_num_bufs, consumer_two_ctas=use_2cta)

    x_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_M, scale_k], x_scale_layout)
    w_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_N, scale_k], w_scale_layout)

    acc_num_bufs: gl.constexpr = ACC_NUM_BUFS
    acc_tmem = blackwell.allocate_tensor_memory(gl.float32, [acc_num_bufs, BLOCK_N, BLOCK_M], acc_layout)
    acc_empty_bars, acc_ready_bars = alloc_ring_barriers(acc_num_bufs, producer_two_ctas=use_2cta)

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
        out_handles=out_handles,
        bias_ptr=bias_ptr,
        bias_stride=bias_stride,
        scatter_indx_ptr=scatter_indx_ptr,
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
        grid_m=grid_m,
        GRID_N=grid_n,
        K_TILES=k_tiles,
        SCALE_FLAT_N=scale_flat_n,
        SCALE_BLOCK_N_DIV=scale_block_n_div,
        num_blocks=num_blocks,
        #
        NUM_SMS=NUM_SMS,
        USE_2CTA=use_2cta,
        BLOCK_M_PER_CTA=block_m_per_cta,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SCALE_SIZE_OUTER=SCALE_SIZE_OUTER,
        SCALE_SIZE_INNER=SCALE_SIZE_INNER,
        MXFP_BLOCK_SIZE=MXFP_BLOCK_SIZE,
        PACKED_BLOCK_K=BLOCK_K // 2,
        #
        REDUCE_RANK=REDUCE_RANK,
        N_REDUCE_SHARDS=N_REDUCE_SHARDS,
        FLEXPOINT_SATURATE_INF=FLEXPOINT_SATURATE_INF,
        #
        EPILOGUE_ROW_SUBTILE_FACTOR=EPILOGUE_ROW_SUBTILE_FACTOR,
        EPILOGUE_STORE_HELPER_DEPTH=EPILOGUE_STORE_HELPER_DEPTH,
        BAND_N=BAND_N,
        X_GATHER_MULTICAST=X_GATHER_MULTICAST,
        W_SCALE_MULTICAST=W_SCALE_MULTICAST,
    )

    gl.warp_specialize(
        [
            (epilogue_partition, (p, map_dst_coord)),
            (load_activations, (p,)),
            (load_weights, (p,)),
            (mma_partition, (p,)),
        ],
        [LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
        [LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
    )

    all_writes_issued.fn(*all_writes_issued.captured)

    invalidate_barrier_ring(x_empty_bars, x_num_bufs)
    invalidate_barrier_ring(x_ready_bars, x_num_bufs)
    invalidate_barrier_ring(w_empty_bars, w_num_bufs)
    invalidate_barrier_ring(w_ready_bars, w_num_bufs)
    invalidate_barrier_ring(acc_empty_bars, acc_num_bufs)
    invalidate_barrier_ring(acc_ready_bars, acc_num_bufs)


# ===-----------------------------------------------------------------------===#
# Host Code
# ===-----------------------------------------------------------------------===#


def make_tensor_descriptor(
        t: torch.Tensor | Tensor,
        block_shape: tuple[int, ...],
        *,
        layout_block_shape: tuple[int, ...] | None = None,
        cga_layout: tuple[tuple[int, ...], ...] = (),
):
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    ptr = t if isinstance(t, torch.Tensor) else t.storage.data
    shape = list(ptr.shape)
    strides = list(ptr.stride())
    desc_block_shape = list(block_shape)
    layout_shape = list(layout_block_shape or block_shape)

    if isinstance(t, Tensor) and t.dtype == FP4:
        assert isinstance(t.storage.layout, BlackwellMX4ValueShuffledLayout)
        assert layout_block_shape is None
        desc_block_shape = t.storage.layout.swizzle_block_shape(desc_block_shape)
        desc_block_shape[strides.index(1)] //= 2
        layout_shape = desc_block_shape

    rank = len(layout_shape)
    if t.dtype == FP4:
        assert rank == 5
        layout = gl.NVMMASharedLayout(
            swizzle_byte_width=128,
            element_bitwidth=8,
            rank=rank,
            fp4_padded=True,
            cga_layout=cga_layout,
        )
    elif t.dtype == UINT8:
        assert rank == 5
        layout = gl.NVMMASharedLayout(
            swizzle_byte_width=0,
            element_bitwidth=8,
            rank=rank,
            cga_layout=cga_layout,
        )
    elif t.dtype == torch.float32:
        assert rank == 2
        layout = gl.NVMMASharedLayout.get_default_for(
            layout_shape,
            torch.float32,
            cga_layout=cga_layout,
        )
    else:
        assert t.dtype == torch.float8_e4m3fn
        layout = gl.NVMMASharedLayout(
            swizzle_byte_width=layout_shape[-1],
            element_bitwidth=8,
            rank=rank,
            cga_layout=cga_layout,
        )
    return TensorDescriptor(ptr, shape, strides, desc_block_shape, layout)


def make_ragged_tensor_descriptor(
        t: torch.Tensor,
        block_shape: tuple[int, ...],
        *,
        layout_block_shape: tuple[int, ...] | None = None,
        cga_layout: tuple[tuple[int, ...], ...] = (),
):
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

    assert t.ndim == 2
    ragged_dim = 0
    ragged_billion = 0x40000000
    ragged_max_int = 0x7FFF0000

    shape = list(t.shape)
    shape[ragged_dim] = ragged_billion
    shape = [ragged_max_int, ragged_max_int] + shape

    ragged_stride = t.stride(ragged_dim)
    strides = [2**34 - ragged_stride, ragged_stride] + list(t.stride())

    desc_block_shape = [1, 1] + list(block_shape)
    layout_shape = [1, 1] + list(layout_block_shape or block_shape)
    rank = len(layout_shape)
    assert t.dtype == torch.float8_e4m3fn
    layout = gl.NVMMASharedLayout(
        swizzle_byte_width=layout_shape[-1],
        element_bitwidth=8,
        rank=rank,
        cga_layout=cga_layout,
    )
    return TensorDescriptor(t, shape, strides, desc_block_shape, layout)


@dataclass(frozen=True, slots=True)
class KernelConfig:
    BLOCK_M: int = 128
    BLOCK_N: int = 512
    BLOCK_K: int = 128
    NUM_CTAS: int = 2
    X_NUM_BUFS: int = 5
    W_NUM_BUFS: int = 5
    ACC_NUM_BUFS: int = 1
    NUM_WARPS: int = 8
    LOAD_ACTIVATION_WARPS: int = 4
    LOAD_WEIGHT_WARPS: int = 1
    MMA_WARPS: int = 1
    STORE_HELPER_WARPS: int = 2
    EPILOGUE_ROW_SUBTILE_FACTOR: int = 8
    EPILOGUE_STORE_HELPER_DEPTH: int = 2
    BAND_N: int = 26
    X_GATHER_MULTICAST: bool = True
    W_SCALE_MULTICAST: bool = True
    LOAD_ACTIVATION_REGS: int = 112
    LOAD_WEIGHT_REGS: int = 48
    MMA_REGS: int = 48
    STORE_HELPER_REGS: int = 24
    MAXNREG: int | None = None
    OCCUPANCY: int = 1


def estimated_slice_size(ragged_metadata: RaggedTensorMetadata, m: int) -> int:
    if ragged_metadata.expected_slice_size is not None:
        return ragged_metadata.expected_slice_size
    return max(1, m // ragged_metadata.n_slices)


def select_kernel_config(ragged_metadata: RaggedTensorMetadata, m: int) -> KernelConfig:
    slice_size = estimated_slice_size(ragged_metadata, m)
    config = KernelConfig()
    if slice_size <= 16:
        return replace(
            config,
            BLOCK_M=32,
            BLOCK_N=512,
            EPILOGUE_ROW_SUBTILE_FACTOR=2,
            X_NUM_BUFS=5,
            W_NUM_BUFS=6,
            BAND_N=16,
            LOAD_ACTIVATION_REGS=80,
            MAXNREG=80,
        )
    if slice_size <= 48:
        return replace(
            config,
            BLOCK_M=64,
            BLOCK_N=512,
            EPILOGUE_ROW_SUBTILE_FACTOR=2,
            X_NUM_BUFS=5,
            W_NUM_BUFS=6,
            BAND_N=26,
            LOAD_ACTIVATION_REGS=80,
            MAXNREG=80,
            OCCUPANCY=1,
        )
    if slice_size <= 72:
        return replace(
            config,
            BLOCK_M=64,
            BLOCK_N=512,
            EPILOGUE_ROW_SUBTILE_FACTOR=4,
            X_NUM_BUFS=6,
            W_NUM_BUFS=6,
            BAND_N=8,
            LOAD_ACTIVATION_REGS=80,
            MAXNREG=80,
            OCCUPANCY=1,
        )
    if slice_size <= 128:
        return replace(config, BAND_N=18, LOAD_ACTIVATION_WARPS=2)
    if slice_size <= 256:
        return replace(
            config,
            BAND_N=18,
            LOAD_ACTIVATION_WARPS=2,
            LOAD_WEIGHT_REGS=64,
        )
    if slice_size < 416:
        return replace(config, BAND_N=18)
    if slice_size >= 960:
        return replace(config, BAND_N=16, STORE_HELPER_REGS=36)
    return replace(config, STORE_HELPER_REGS=36)


def matmul(
    a: torch.Tensor,
    b: torch.Tensor | Tensor,
    bias: torch.Tensor,
    a_ragged_metadata: RaggedTensorMetadata,
    scatter_indx: torch.Tensor,
    precision_config: PrecisionConfig,
    c: torch.Tensor,
    fused_comm: FusedComm,
):
    b_mx_scales = precision_config.b_mx_scale
    out_dtype = precision_config.out_dtype
    assert out_dtype == torch.float8_e4m3fn
    assert b_mx_scales is not None
    assert fused_comm is not None
    assert c.ndim == 3

    flex_ctx = precision_config.flex_ctx
    assert a.ndim == 2
    _, k = a.shape
    _, _, n = b.shape
    m = scatter_indx.shape[0]

    config = select_kernel_config(a_ragged_metadata, m)
    assert isinstance(b, Tensor)
    assert isinstance(b.storage.layout, BlackwellMX4ValueShuffledLayout)
    assert b.storage.layout.block_k == config.BLOCK_K
    assert b.storage.layout.block_n == config.BLOCK_N

    mxfp_block_size = 32
    scale_size_outer = 128
    scale_size_inner = 4
    expected_grid_m = a_ragged_metadata.n_blocks(a_ragged_metadata.n_slices, m, config.BLOCK_M)
    grid_n = triton.cdiv(n, config.BLOCK_N)
    sms = torch.cuda.get_device_properties(bias.device).multi_processor_count
    sms *= config.OCCUPANCY
    launch_grid = max(1, min(max(1, sms // config.NUM_CTAS), expected_grid_m * grid_n))
    grid = (launch_grid,)
    acc_cga_layout = ((1, 0),)

    x_desc = make_ragged_tensor_descriptor(
        a,
        (config.BLOCK_M, config.BLOCK_K),
        cga_layout=tuple((0, 0, basis[0], 0) for basis in acc_cga_layout),
    )
    w_desc = make_tensor_descriptor(
        b,
        (1, config.BLOCK_K, config.BLOCK_N),
        cga_layout=tuple((0, 0, 0, basis[0], 0) for basis in acc_cga_layout),
    )
    scale_desc = make_tensor_descriptor(
        b_mx_scales,
        (
            1,
            config.BLOCK_N // scale_size_outer,
            config.BLOCK_K // mxfp_block_size // scale_size_inner,
            2,
            256,
        ),
        cga_layout=tuple((0, basis[0], 0, 0, 0) for basis in acc_cga_layout),
    )
    out_desc = make_tensor_descriptor(c, (1, config.BLOCK_M, min(config.BLOCK_N, 128)))

    ws_matmul_kernel[grid](
        x_desc=x_desc,
        w_desc=w_desc,
        scale_desc=scale_desc,
        out_desc=out_desc,
        out_ptr=c,
        out_handles=fused_comm.out_handles,
        map_dst_coord=fused_comm.map_dst_coord,
        all_writes_issued=Closure(gluon_set_locks, fused_comm.all_writes_issued.captured),
        #
        bias_ptr=bias,
        bias_stride=bias.stride(0),
        #
        scatter_indx_ptr=scatter_indx,
        #
        x_slice_sizes=a_ragged_metadata.slice_sizes,
        x_slice_offs=a_ragged_metadata.slice_offs,
        x_block_offs=a_ragged_metadata.block_offs(config.BLOCK_M),
        x_block_schedule=a_ragged_metadata.block_schedule(config.BLOCK_M),
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
        REDUCE_RANK=fused_comm.reduce_rank,
        N_REDUCE_SHARDS=fused_comm.n_reduce_shards,
        FLEXPOINT_SATURATE_INF=precision_config.flexpoint_saturate_inf,
        #
        BLOCK_M=config.BLOCK_M,
        BLOCK_N=config.BLOCK_N,
        BLOCK_K=config.BLOCK_K,
        NUM_SMS=launch_grid,
        X_NUM_BUFS=config.X_NUM_BUFS,
        W_NUM_BUFS=config.W_NUM_BUFS,
        ACC_NUM_BUFS=config.ACC_NUM_BUFS,
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
        BAND_N=config.BAND_N,
        X_GATHER_MULTICAST=config.X_GATHER_MULTICAST,
        W_SCALE_MULTICAST=config.W_SCALE_MULTICAST,
        SCALE_SIZE_OUTER=scale_size_outer,
        SCALE_SIZE_INNER=scale_size_inner,
        MXFP_BLOCK_SIZE=mxfp_block_size,
        #
        num_warps=config.NUM_WARPS,
        num_ctas=config.NUM_CTAS,
        maxnreg=config.MAXNREG,
    )

    return c[0]


@dataclass(frozen=True, slots=True)
class MM2ModelConfig:
    name: str
    num_experts: int
    experts_per_token: int
    num_expert_shards: int
    hidden_size: int
    intermediate_size: int
    expected_slice_sizes: tuple[int, ...]
    local_rank: int = 0

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.num_expert_shards

    @property
    def bmm2_shape(self) -> tuple[int, int]:
        return (self.intermediate_size, self.hidden_size)

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        return tuple(
            expected_slice_size * self.num_experts // self.experts_per_token
            for expected_slice_size in self.expected_slice_sizes
        )

    def routed_rows(self, batch_size: int) -> int:
        return batch_size * self.experts_per_token

    def expected_slice_size(self, batch_size: int) -> int:
        return batch_size * self.experts_per_token // self.num_experts

    @property
    def plot_name(self) -> str:
        k, n = self.bmm2_shape
        return (
            f"{self.name} MoE MM2 fused scatter "
            f"E={self.num_experts} "
            f"EP={self.num_expert_shards} "
            f"B={k}x{n}"
        )


SPUD_D64_EXPECTED_SLICE_SIZES = tuple(
    chain(range(16, 257, 16), range(320, 1025, 64), (1040,))
)
SPUD_D64_CONFIG = MM2ModelConfig(
    name="SPUD_D64",
    num_experts=256,
    experts_per_token=4,
    num_expert_shards=8,
    hidden_size=8192,
    intermediate_size=8192,
    expected_slice_sizes=SPUD_D64_EXPECTED_SLICE_SIZES,
)

OUTPUT_MAXTOL = 0.2
OUTPUT_RMSTOL = None


@dataclass(frozen=True, slots=True)
class PreparedCase:
    config: MM2ModelConfig
    batch_size: int
    local_rank: int
    a: torch.Tensor
    b: Tensor
    b_scale: Tensor
    bias: torch.Tensor
    ragged_metadata: RaggedTensorMetadata
    scatter_indx: torch.Tensor
    precision_config: PrecisionConfig
    c: torch.Tensor
    out_buffers: list[torch.Tensor]
    fused_comm: FusedComm
    lock: torch.Tensor

    @property
    def routed_rows(self) -> int:
        return self.config.routed_rows(self.batch_size)

    @property
    def dp_batch_size(self) -> int:
        return (
            self.batch_size + self.config.num_expert_shards - 1
        ) // self.config.num_expert_shards


def trace_config_shapes(config: MM2ModelConfig) -> tuple[dict[str, object], ...]:
    k, n = config.bmm2_shape
    return tuple(
        {
            "config": config.name,
            "expected_slice_size": config.expected_slice_size(batch_size),
            "batch_size": batch_size,
            "routed_rows": config.routed_rows(batch_size),
            "a_shape": (config.routed_rows(batch_size), k),
            "b_shape": (config.local_experts, k, n),
            "bias_shape": (config.local_experts, n),
            "scatter_shape": (config.routed_rows(batch_size),),
            "c_shape": (1, config.routed_rows(batch_size), n),
            "slice_sizes_shape": (config.local_experts,),
            "block_schedule_shape": (
                len(RaggedTensorMetadata.block_sizes()),
                RaggedTensorMetadata.max_n_blocks(
                    config.local_experts, config.routed_rows(batch_size)
                ),
            ),
            "fused_comm_out_handles_shape": (config.num_expert_shards,),
            "fused_comm_reduce_rank": 0,
            "fused_comm_n_reduce_shards": 1,
        }
        for batch_size in config.batch_sizes
    )


def alloc_randn(
    shape: tuple[int, ...], dtype: torch.dtype, device: str
) -> torch.Tensor:
    if dtype.itemsize == 1:
        return alloc_rand(shape, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def alloc_randn_fp4(shape: tuple[int, ...], device: str, p: KernelConfig | None = None) -> tuple[Tensor, Tensor]:
    data = alloc_randn(shape, torch.bfloat16, device)
    data, scale = downcast_to_mxfp(data, FP4, axis=1)  # type: ignore[arg-type]
    block_k = KernelConfig.BLOCK_K if p is None else p.BLOCK_K
    block_n = KernelConfig.BLOCK_N if p is None else p.BLOCK_N
    num_warps = KernelConfig.NUM_WARPS if p is None else p.NUM_WARPS
    data_layout = BlackwellMX4ValueShuffledLayout(block_k=block_k, block_n=block_n)
    scale_layout = make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=num_warps)
    data = convert_layout(wrap_torch_tensor(data, dtype=FP4), data_layout)
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout)
    return data, scale


def init_routing_data(
    config: MM2ModelConfig,
    batch_size: int,
    local_rank: int,
    device: str,
) -> tuple[RaggedTensorMetadata, torch.Tensor]:
    expt_dist = make_expt_dict_uniform(config.num_expert_shards, config.num_experts)
    logits = torch.randn(
        (batch_size, config.num_experts), dtype=torch.float16, device=device
    )
    sparse_logits = topk(logits, config.experts_per_token, apply_softmax=True)
    expt_hist = sparse_logits.mask_metadata.col_sum

    local_expts = expt_dist[local_rank]
    local_expts_hist = expt_hist[local_expts]
    if local_expts_hist.sum() == 0:
        local_expts_hist[0] = 1
    ragged_metadata = make_ragged_tensor_metadata(
        local_expts_hist, config.routed_rows(batch_size)
    )
    ragged_metadata.expected_slice_size = config.expected_slice_size(batch_size)

    scatter_indx = sparse_logits.mask_metadata.col_sorted_indx.to(torch.int32)
    return ragged_metadata, scatter_indx


@triton.jit
def threadfence_system():
    tl.inline_asm_elementwise(
        "mov.u32 $0, 0x0; fence.sc.sys;",
        args=(),
        dtype=(tl.int32,),
        is_pure=False,
        pack=1,
        constraints="=r",
    )


@triton.jit
def set_locks(p_locks, n_shards: tl.constexpr):
    my_lock = tl.load(p_locks).item().to(tl.pointer_type(tl.uint32))
    my_grid_ptr = my_lock + 2

    tl.debug_barrier()
    grid_done = tl.atomic_add(my_grid_ptr, val=1, sem="relaxed", scope="gpu")

    if grid_done + 1 == tl.num_programs(0) * tl.num_programs(1) * tl.num_programs(2):
        threadfence_system()

        shard_id = tl.arange(0, n_shards)
        peers = tl.load(p_locks + shard_id).to(tl.pointer_type(tl.uint32))

        peer_begin_ptr = peers
        tl.atomic_add(peer_begin_ptr, val=1, mask=shard_id != 0, sem="release", scope="sys")

        tl.store(my_grid_ptr, 0, cache_modifier=".cg")


@triton.jit
def _map_dst_coord(
    _off_m,
    offs_m,
    _off_n,
    offs_n,
    rank: tl.constexpr,
    n_peers: tl.constexpr,
    dp_batch_size: tl.constexpr,
    topk: tl.constexpr,
):
    dst_shard_idx = offs_m // (dp_batch_size * topk)
    # Handles are rotated such that bfs.matmul_comms_args.handles[0] is the local rank.
    # Apply the inverse rotation.
    dst_shard_idx = (dst_shard_idx - rank) % n_peers
    # wrap negative indices to positive indices
    dst_shard_idx += tl.where(dst_shard_idx < 0, n_peers, 0)
    return dst_shard_idx[:, None], offs_m % (dp_batch_size * topk), offs_n


def make_precision_config(prepared: PreparedCase) -> PrecisionConfig:
    return PrecisionConfig(
        flexpoint_saturate_inf=True,
        b_mx_scale=prepared.b_scale,
        b_microblock_size=int(MXFP_BLOCK_SIZE),
        out_dtype=torch.float8_e4m3fn,
        flex_ctx=FlexCtx(
            lhs_data=InFlexData(
                scale=torch.ones((1,), device=prepared.a.device, dtype=torch.float32)
            ),
            rhs_data=InFlexData(),
            out_data=OutFlexData(
                dtype=torch.float8_e4m3fn,
                expected_scale=torch.full(
                    (1,), 4.0, device=prepared.a.device, dtype=torch.float32
                ),
            ),
        ),
    )


def make_fused_comm(
    config: MM2ModelConfig,
    batch_size: int,
    local_rank: int,
    out_buffers: list[torch.Tensor],
    lock: torch.Tensor,
) -> FusedComm:
    out_handles = torch.tensor(
        [buf.data_ptr() for buf in out_buffers],
        dtype=torch.uint64,
        device=out_buffers[0].device,
    )
    out_handles.__indirect_tensors__ = out_buffers

    lock_handles = torch.tensor(
        [lock.data_ptr()] * config.num_expert_shards,
        dtype=torch.uint64,
        device=lock.device,
    )
    lock_handles.__indirect_tensors__ = [lock] * config.num_expert_shards

    return FusedComm(
        out_handles=out_handles,
        map_dst_coord=Closure(
            _map_dst_coord,
            (
                local_rank,
                config.num_expert_shards,
                (batch_size + config.num_expert_shards - 1) // config.num_expert_shards,
                config.experts_per_token,
            ),
        ),
        all_writes_issued=Closure(
            set_locks,
            (lock_handles, tl.constexpr(config.num_expert_shards)),
        ),
        reduce_rank=0,
        n_reduce_shards=1,
    )


def prepare_case(
    config: MM2ModelConfig,
    batch_size: int,
    device: str,
    seed: int = 0,
    zero_output: bool = True,
) -> PreparedCase:
    torch.manual_seed(seed)

    local_rank = config.local_rank
    k, n = config.bmm2_shape
    routed_rows = config.routed_rows(batch_size)
    ragged_metadata, scatter_indx = init_routing_data(
        config, batch_size, local_rank, device
    )
    a = alloc_randn((routed_rows, k), dtype=torch.float8_e4m3fn, device=device)
    kernel_config = select_kernel_config(ragged_metadata, routed_rows)
    b, b_scale = alloc_randn_fp4((config.local_experts, k, n), device=device, p=kernel_config)
    bias = alloc_randn((config.local_experts, n), dtype=torch.float32, device=device)

    make_out = torch.zeros if zero_output else torch.empty
    out_buffers = [
        make_out((1, routed_rows, n), dtype=torch.float8_e4m3fn, device=device)
        for _ in range(config.num_expert_shards)
    ]
    lock = torch.zeros((4,), dtype=torch.int32, device=device)
    fused_comm = make_fused_comm(config, batch_size, local_rank, out_buffers, lock)

    prepared = PreparedCase(
        config=config,
        batch_size=batch_size,
        local_rank=local_rank,
        a=a,
        b=b,
        b_scale=b_scale,
        bias=bias,
        ragged_metadata=ragged_metadata,
        scatter_indx=scatter_indx,
        precision_config=PrecisionConfig(),
        c=out_buffers[0],
        out_buffers=out_buffers,
        fused_comm=fused_comm,
        lock=lock,
    )
    object.__setattr__(prepared, "precision_config", make_precision_config(prepared))
    return prepared


def run_reference_matmul(
    prepared: PreparedCase,
    c: torch.Tensor | None = None,
    fused_comm: FusedComm | None = None,
):
    return reference_matmul(
        a=prepared.a,
        b=prepared.b,
        bias=prepared.bias,
        a_ragged_metadata=prepared.ragged_metadata,
        scatter_indx=prepared.scatter_indx,
        precision_config=prepared.precision_config,
        c=c,
        fused_comm=fused_comm,
    )


def run_example_matmul(prepared: PreparedCase, c: torch.Tensor | None = None):
    return matmul(
        a=prepared.a,
        b=prepared.b,
        bias=prepared.bias,
        a_ragged_metadata=prepared.ragged_metadata,
        scatter_indx=prepared.scatter_indx,
        precision_config=prepared.precision_config,
        c=prepared.c if c is None else c,
        fused_comm=prepared.fused_comm,
    )


def assert_fused_comm_outputs(prepared: PreparedCase, ref_y: torch.Tensor) -> None:
    torch.cuda.synchronize()

    lock_cpu = prepared.lock.cpu()
    assert lock_cpu[0].item() == prepared.config.num_expert_shards - 1
    assert lock_cpu[1].item() == 0
    assert lock_cpu[2].item() == 0
    assert lock_cpu[3].item() == 0

    shard_rows = prepared.dp_batch_size * prepared.config.experts_per_token
    total_rows = prepared.routed_rows
    for shard, buf in enumerate(prepared.out_buffers):
        src_shard = (shard + prepared.local_rank) % prepared.config.num_expert_shards
        src_lo = src_shard * shard_rows
        src_hi = min((src_shard + 1) * shard_rows, total_rows)
        src_nrows = max(0, src_hi - src_lo)

        expected = torch.zeros_like(buf[0, :shard_rows, :])
        if src_nrows > 0:
            expected[:src_nrows] = ref_y[src_lo:src_hi]
        assert_close(
            expected.to(torch.float32),
            buf[0, :shard_rows, :].to(torch.float32),
            maxtol=OUTPUT_MAXTOL,
            rmstol=OUTPUT_RMSTOL,
            description=f"scatter shard {shard}",
            verbose=False,
        )
        assert (
            torch.count_nonzero(buf[0, shard_rows:, :]).item() == 0
        ), f"unexpected rows in buffer {shard}"


def is_blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


TEST_CASES = tuple(
    pytest.param(batch_size, id=f"SPUD_D64-ess{expected_slice_size}-bs{batch_size}")
    for expected_slice_size, batch_size in zip(
        SPUD_D64_CONFIG.expected_slice_sizes, SPUD_D64_CONFIG.batch_sizes
    )
)


@pytest.mark.parametrize("batch_size", TEST_CASES)
@pytest.mark.skipif(
    not is_blackwell(),
    reason="SPUD D64 fused-scatter MM2 shape tests require Blackwell",
)
def test_triton_kernels_matmul_fused_scatter(batch_size: int):
    prepared = prepare_case(
        SPUD_D64_CONFIG,
        batch_size,
        device=f"cuda:{torch.cuda.current_device()}",
        seed=0,
    )
    ref_y = run_reference_matmul(prepared, c=torch.zeros_like(prepared.c))
    fused_y = run_reference_matmul(
        prepared, c=prepared.c, fused_comm=prepared.fused_comm
    )
    assert tuple(fused_y.shape) == (prepared.routed_rows, SPUD_D64_CONFIG.hidden_size)
    assert fused_y.dtype == torch.float8_e4m3fn
    assert_fused_comm_outputs(prepared, ref_y)


EXAMPLE_TEST_CASES = (
    SPUD_D64_CONFIG.batch_sizes[0],
    SPUD_D64_CONFIG.batch_sizes[1],
    SPUD_D64_CONFIG.batch_sizes[7],
    SPUD_D64_CONFIG.batch_sizes[-1],
)


@pytest.mark.parametrize(
    "batch_size",
    EXAMPLE_TEST_CASES,
    ids=[f"SPUD_D64-example-bs{batch_size}" for batch_size in EXAMPLE_TEST_CASES],
)
@pytest.mark.skipif(
    not is_blackwell(),
    reason="Gluon SPUD D64 fused-scatter MM2 requires Blackwell",
)
def test_gluon_matmul_fused_scatter(batch_size: int):
    prepared = prepare_case(
        SPUD_D64_CONFIG,
        batch_size,
        device=f"cuda:{torch.cuda.current_device()}",
        seed=0,
    )
    ref_y = run_reference_matmul(prepared, c=torch.zeros_like(prepared.c))
    fused_y = run_example_matmul(prepared)
    assert tuple(fused_y.shape) == (prepared.routed_rows, SPUD_D64_CONFIG.hidden_size)
    assert fused_y.dtype == torch.float8_e4m3fn
    assert_fused_comm_outputs(prepared, ref_y)


providers = ["example", "triton_kernels.matmul"]
bench_configs = [
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=SPUD_D64_CONFIG.batch_sizes,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[("red", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name=SPUD_D64_CONFIG.plot_name,
        args={},
    )
]


@triton.testing.perf_report(bench_configs)
def bench(batch_size, provider):
    prepared = prepare_case(
        SPUD_D64_CONFIG,
        batch_size,
        device=f"cuda:{torch.cuda.current_device()}",
        seed=0,
        zero_output=False,
    )
    if provider == "example":
        ms = do_bench_cudagraph(lambda: run_example_matmul(prepared, c=prepared.c))
    else:
        ms = do_bench_cudagraph(
            lambda: run_reference_matmul(
                prepared, c=prepared.c, fused_comm=prepared.fused_comm
            )
        )
    n_tokens = int(prepared.ragged_metadata.slice_sizes.sum().item())
    k, n = SPUD_D64_CONFIG.bmm2_shape
    flops = 2 * n_tokens * k * n
    return flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench.run(save_path=".", print_data=True)
