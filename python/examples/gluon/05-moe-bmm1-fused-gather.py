from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
import triton.experimental.gluon.language.nvidia.blackwell as blackwell
import triton.experimental.gluon.language.nvidia.blackwell.tma as tma
from triton.experimental.gluon.language.nvidia.blackwell import float2
import triton.experimental.gluon.language.nvidia.hopper.mbarrier as mbarrier
import triton.language.core as tl_core
import triton.language.extra.cuda.libdevice as libdevice
from triton.language.core import _aggregate as aggregate

from triton_kernels.matmul import FusedActivation, PrecisionConfig
from triton_kernels.tensor import RaggedTensorMetadata, Tensor
from triton_kernels.tensor_details.dtype import FP4, UINT8


@gluon.jit
def advance(idx: gl.tensor, phase: gl.tensor, num_bufs: gl.constexpr) -> tuple[gl.tensor, gl.tensor]:
    next_idx = idx + 1
    wrap = next_idx == num_bufs
    return gl.where(wrap, 0, next_idx), gl.where(wrap, phase ^ 1, phase)


@gluon.jit
def unpack_block_schedule(schedule):
    return schedule & 0xFFFF, schedule >> 16


@gluon.jit
def xcd_swizzle(block_id, grid_m, GRID_N: gl.constexpr, XCD_SWIZZLE: gl.constexpr):
    if XCD_SWIZZLE == 1:
        return block_id

    num_blocks = grid_m * GRID_N
    per_group = num_blocks // XCD_SWIZZLE
    extra = num_blocks % XCD_SWIZZLE
    group = block_id % XCD_SWIZZLE
    return group * per_group + min(group, extra) + block_id // XCD_SWIZZLE


@gluon.jit
def apply_block_schedule(
    block_id: gl.tensor,
    grid_m: gl.tensor,
    GRID_N: gl.constexpr,
    slice_offsets: gl.tensor,
    block_schedule: gl.tensor,
    XCD_SWIZZLE: gl.constexpr,
    N_MAJOR: gl.constexpr,
) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
    block_id = xcd_swizzle(block_id, grid_m, GRID_N, XCD_SWIZZLE)

    pid_mn = block_id % (grid_m * GRID_N)
    pid_m = pid_mn % grid_m if N_MAJOR else pid_mn // GRID_N
    pid_n = pid_mn // grid_m if N_MAJOR else pid_mn % GRID_N

    slice_idx, pid_m = unpack_block_schedule(gl.load(block_schedule + pid_m))
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
    block_m: int = 128
    block_n: int = 256
    block_k: int = 128
    x_num_bufs: int = 5
    w_num_bufs: int = 4
    num_warps: int = 8
    epilogue_n_elems: int = 4
    load_activation_warps: int = 4
    load_weight_warps: int = 1
    mma_warps: int = 1
    load_activation_regs: int = 96
    load_weight_regs: int = 64
    mma_regs: int = 24


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

    grid_m: gl.tensor
    GRID_N: gl.constexpr
    K_TILES: gl.constexpr
    SCALE_FLAT_N: gl.constexpr
    SCALE_BLOCK_N_DIV: gl.constexpr
    num_blocks: gl.tensor

    NUM_SMS: gl.constexpr
    XCD_SWIZZLE: gl.constexpr
    N_MAJOR: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    EPILOGUE_N_ELEMS: gl.constexpr
    SCALE_SIZE_OUTER: gl.constexpr
    SCALE_SIZE_INNER: gl.constexpr
    MXFP_BLOCK_SIZE: gl.constexpr
    PACKED_BLOCK_K: gl.constexpr

    SWIGLU_ALPHA: gl.constexpr
    SWIGLU_LIMIT: gl.constexpr
    REDUCTION_N: gl.constexpr

    @gluon.jit
    def apply_block_schedule(self, block_id: gl.tensor) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
        return apply_block_schedule(
            block_id=block_id,
            grid_m=self.grid_m,
            GRID_N=self.GRID_N,
            slice_offsets=self.x_slice_offs,
            block_schedule=self.x_block_schedule,
            XCD_SWIZZLE=self.XCD_SWIZZLE,
            N_MAJOR=self.N_MAJOR,
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
def _pack_e4m3x2(values):
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
def _store_packed_out(
    p: PartitionArgs,
    packed_out,
    off_m,
    out_off_n,
    shape_m,
    slice_offset,
):
    out_ptr = p.out_ptr.cast(gl.pointer_type(gl.int16), bitcast=True)
    packed_layout: gl.constexpr = packed_out.type.layout
    block_n_div_act: gl.constexpr = p.BLOCK_N // p.REDUCTION_N

    offs_m = off_m + gl.arange(0, p.BLOCK_M, layout=gl.SliceLayout(1, packed_layout))
    offs_n = out_off_n // 2 + gl.arange(0, block_n_div_act // 2, layout=gl.SliceLayout(0, packed_layout))
    mask = gl.expand_dims(offs_m < shape_m, 1)
    ptrs = out_ptr + gl.expand_dims(slice_offset + offs_m, 1) * (p.out_desc.strides[0] // 2)
    ptrs = ptrs + gl.expand_dims(offs_n, 0) * p.out_desc.strides[1]
    gl.store(ptrs, packed_out, mask=mask)


@gluon.jit
def epilogue_partition_optimized(p: PartitionArgs):
    idx = 0
    phase = 0

    x_scale = 1.0 if p.x_scale_ptr is None else gl.load(p.x_scale_ptr)
    w_scale = 1.0 if p.w_scale_ptr is None else gl.load(p.w_scale_ptr)
    acc_scale = x_scale * w_scale
    out_recip = 1.0 / gl.load(p.out_scale_ptr)

    acc_reg_layout: gl.constexpr = p.acc_bufs.index(0).get_reg_layout()
    num_warps: gl.constexpr = gl.num_warps()
    warps_n: gl.constexpr = 2 if num_warps >= 4 and p.BLOCK_N >= 256 else 1
    split_layout: gl.constexpr = gl.BlockedLayout(
        [1, p.EPILOGUE_N_ELEMS],
        [1, 32],
        [num_warps // warps_n, warps_n],
        [1, 0],
    )
    bias_layout: gl.constexpr = gl.SliceLayout(0, split_layout)

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        off_n = pid_n * p.BLOCK_N
        shape_m = gl.load(p.x_slice_sizes + slice_idx)

        offs_bias_n = off_n + gl.arange(0, p.BLOCK_N, layout=bias_layout)
        bias = gl.convert_layout(
            gl.expand_dims(gl.load(p.bias_ptr + slice_idx * p.bias_stride + offs_bias_n), axis=0),
            split_layout,
        )

        acc_empty_bar = p.acc_empty_bars.index(idx)
        acc_ready_bar = p.acc_ready_bars.index(idx)
        acc_buf = p.acc_bufs.index(idx)
        mbarrier.wait(acc_ready_bar, phase)
        acc = acc_buf.load(acc_reg_layout)
        mbarrier.arrive(acc_empty_bar)
        idx, phase = advance(idx, phase, p.acc_num_bufs)

        out_off_n = pid_n * p.BLOCK_N // p.REDUCTION_N
        acc = gl.convert_layout(acc.permute((1, 0)), split_layout)
        acc_packed = float2.pack(acc, axis=1)
        bias_packed = float2.pack(bias, axis=1)
        acc_packed = float2.fma(acc_packed, float2.full_like(acc_packed, acc_scale), bias_packed)

        gelu, linear = float2.unpack2(acc_packed)
        gelu = gl.minimum(gelu.to(gl.float32), p.SWIGLU_LIMIT)
        linear = tl_core.clamp(linear.to(gl.float32), -p.SWIGLU_LIMIT, p.SWIGLU_LIMIT)
        den = 1.0 + libdevice.exp(-p.SWIGLU_ALPHA * gelu)
        activated = gelu / den
        activated_packed = float2.pack(activated, axis=1)
        linear_packed = float2.pack(linear, axis=1)
        out_packed = float2.fma(activated_packed, linear_packed, activated_packed)
        out_packed = out_packed * float2.full_like(out_packed, out_recip)
        packed_out = _pack_e4m3x2(out_packed)
        _store_packed_out(
            p,
            packed_out,
            off_m,
            out_off_n,
            shape_m,
            slice_offset,
        )


@gluon.jit
def ws_matmul_kernel_optimized(
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
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_SMS: gl.constexpr,
    X_NUM_BUFS: gl.constexpr,
    W_NUM_BUFS: gl.constexpr,
    EPILOGUE_N_ELEMS: gl.constexpr,
    LOAD_ACTIVATION_WARPS: gl.constexpr,
    LOAD_WEIGHT_WARPS: gl.constexpr,
    MMA_WARPS: gl.constexpr,
    LOAD_ACTIVATION_REGS: gl.constexpr,
    LOAD_WEIGHT_REGS: gl.constexpr,
    MMA_REGS: gl.constexpr,
    SCALE_SIZE_OUTER: gl.constexpr,
    SCALE_SIZE_INNER: gl.constexpr,
    MXFP_BLOCK_SIZE: gl.constexpr,
):
    packed_block_k: gl.constexpr = w_desc.block_type.shape[2]
    xcd_swizzle: gl.constexpr = 1
    n_major: gl.constexpr = False

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
    x_empty_bars = gl.allocate_shared_memory(gl.int64, [x_num_bufs, 1], mbarrier.MBarrierLayout())
    x_ready_bars = gl.allocate_shared_memory(gl.int64, [x_num_bufs, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(x_num_bufs):
        mbarrier.init(x_empty_bars.index(i), count=1)
        mbarrier.init(x_ready_bars.index(i), count=1)

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
    w_empty_bars = gl.allocate_shared_memory(gl.int64, [w_num_bufs, 1], mbarrier.MBarrierLayout())
    w_ready_bars = gl.allocate_shared_memory(gl.int64, [w_num_bufs, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(w_num_bufs):
        mbarrier.init(w_empty_bars.index(i), count=1)
        mbarrier.init(w_ready_bars.index(i), count=1)

    x_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_M, scale_k], scale_layout)
    w_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_N, scale_k], scale_layout)

    acc_num_bufs: gl.constexpr = 1
    acc_tmem = blackwell.allocate_tensor_memory(gl.float32, [acc_num_bufs, BLOCK_N, BLOCK_M], acc_layout)
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [acc_num_bufs, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [acc_num_bufs, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(acc_num_bufs):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

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
        grid_m=grid_m,
        GRID_N=grid_n,
        K_TILES=k_tiles,
        SCALE_FLAT_N=scale_flat_n,
        SCALE_BLOCK_N_DIV=scale_block_n_div,
        num_blocks=num_blocks,
        #
        NUM_SMS=NUM_SMS,
        XCD_SWIZZLE=xcd_swizzle,
        N_MAJOR=n_major,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        EPILOGUE_N_ELEMS=EPILOGUE_N_ELEMS,
        SCALE_SIZE_OUTER=SCALE_SIZE_OUTER,
        SCALE_SIZE_INNER=SCALE_SIZE_INNER,
        MXFP_BLOCK_SIZE=MXFP_BLOCK_SIZE,
        PACKED_BLOCK_K=packed_block_k,
        #
        SWIGLU_ALPHA=SWIGLU_ALPHA,
        SWIGLU_LIMIT=SWIGLU_LIMIT,
        REDUCTION_N=REDUCTION_N,
    )

    gl.warp_specialize(
        [
            (epilogue_partition_optimized, (p,)),
            (load_activations, (p,)),
            (load_weights, (p,)),
            (mma_partition, (p,)),
        ],
        [LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
        [LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
    )

    for i in gl.static_range(x_num_bufs):
        mbarrier.invalidate(x_empty_bars.index(i))
        mbarrier.invalidate(x_ready_bars.index(i))
    for i in gl.static_range(w_num_bufs):
        mbarrier.invalidate(w_empty_bars.index(i))
        mbarrier.invalidate(w_ready_bars.index(i))
    for i in gl.static_range(acc_num_bufs):
        mbarrier.invalidate(acc_empty_bars.index(i))
        mbarrier.invalidate(acc_ready_bars.index(i))


def _select_launch_grid(
    grid_n: int,
    expected_grid_m: int,
    sms: int,
) -> int:
    num_tiles = expected_grid_m * grid_n
    return max(1, min(sms, num_tiles))


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

    assert precision_config.flexpoint_saturate_inf

    b_mx_scales = precision_config.b_mx_scale
    out_dtype = precision_config.out_dtype
    assert out_dtype is not None

    assert c.ndim == 3
    assert c.shape[0] == 1
    c = c.squeeze(0)

    flex_ctx = precision_config.flex_ctx

    assert a.ndim == 2
    _, k = a.shape
    _, _, n = b.shape
    m = gather_indx.shape[0]

    config = KernelConfig()
    mxfp_block_size = 32
    scale_size_outer = 128
    scale_size_inner = 4
    x_block_idx = config.block_m.bit_length() - 5

    expected_grid_m = a_ragged_metadata.n_blocks(a_ragged_metadata.n_slices, m, config.block_m)
    grid_n = triton.cdiv(n, config.block_n)
    sms = torch.cuda.get_device_properties(bias.device).multi_processor_count
    launch_grid = _select_launch_grid(
        grid_n=grid_n,
        expected_grid_m=expected_grid_m,
        sms=sms,
    )
    grid = (launch_grid,)

    x_desc = make_operand_descriptor(a, (1, config.block_k))
    w_desc = make_operand_descriptor(b, (1, config.block_n, config.block_k // 2), transposed=True)
    scale_desc = make_operand_descriptor(
        b_mx_scales,
        (
            1,
            config.block_n // scale_size_outer,
            config.block_k // mxfp_block_size // scale_size_inner,
            2,
            256,
        ),
    )
    out_desc = make_operand_descriptor(c, (config.block_m, config.block_n // reduction_n))

    ws_matmul_kernel_optimized[grid](
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
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        NUM_SMS=launch_grid,
        X_NUM_BUFS=config.x_num_bufs,
        W_NUM_BUFS=config.w_num_bufs,
        EPILOGUE_N_ELEMS=config.epilogue_n_elems,
        LOAD_ACTIVATION_WARPS=config.load_activation_warps,
        LOAD_WEIGHT_WARPS=config.load_weight_warps,
        MMA_WARPS=config.mma_warps,
        LOAD_ACTIVATION_REGS=config.load_activation_regs,
        LOAD_WEIGHT_REGS=config.load_weight_regs,
        MMA_REGS=config.mma_regs,
        SCALE_SIZE_OUTER=scale_size_outer,
        SCALE_SIZE_INNER=scale_size_inner,
        MXFP_BLOCK_SIZE=mxfp_block_size,
        #
        num_warps=config.num_warps,
    )

    return c.unsqueeze(0)
