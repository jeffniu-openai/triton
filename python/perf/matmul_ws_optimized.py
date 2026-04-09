from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
import triton.experimental.gluon.language.nvidia.blackwell as blackwell
import triton.experimental.gluon.language.nvidia.hopper.mbarrier as mbarrier

from triton_kernels.matmul import Epilogue, FusedActivation, FusedComm, PrecisionConfig
from triton_kernels.tensor import RaggedTensorMetadata, Tensor

from . import matmul_ws as ws_base


@dataclass(frozen=True, slots=True)
class KernelConfig:
    block_m: int = 128
    block_n: int = 256
    block_k: int = 128
    xcd_swizzle: int = 1
    n_major: bool = False
    num_warps: int = 8


@gluon.jit
def epilogue_partition_optimized(p: ws_base.PartitionArgs):
    idx = 0
    phase = 0

    x_scale = 1.0 if p.x_scale_ptr is None else gl.load(p.x_scale_ptr)
    w_scale = 1.0 if p.w_scale_ptr is None else gl.load(p.w_scale_ptr)
    acc_scale = x_scale * w_scale
    out_recip = 1.0 / gl.load(p.out_scale_ptr)

    acc_reg_layout: gl.constexpr = p.acc_bufs.index(0).get_reg_layout()
    num_warps: gl.constexpr = gl.num_warps()
    warps_n: gl.constexpr = 2 if num_warps >= 4 and p.BLOCK_N >= 256 else 1
    split_layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 32], [num_warps // warps_n, warps_n], [1, 0])
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
        idx, phase = ws_base.advance(idx, phase, p.acc_num_bufs)

        acc = gl.fma(gl.convert_layout(acc.permute((1, 0)), split_layout), acc_scale, bias)

        out = ws_base.swiglu(acc, p.SWIGLU_ALPHA, p.SWIGLU_LIMIT)
        out_off_n = pid_n * p.BLOCK_N // p.REDUCTION_N
        out = out * out_recip
        if p.FLEXPOINT_SATURATE_INF:
            out = gl.clamp(out, -448.0, 448.0)
        out = out.to(p.out_desc.dtype)

        layout: gl.constexpr = out.type.layout
        offs_m = off_m + gl.arange(0, p.BLOCK_M, layout=gl.SliceLayout(1, layout))
        offs_n = out_off_n + gl.arange(0, p.BLOCK_N // p.REDUCTION_N, layout=gl.SliceLayout(0, layout))
        mask = gl.expand_dims(offs_m < shape_m, 1)
        ptrs = p.out_ptr + gl.expand_dims(slice_offset + offs_m, 1) * p.out_desc.strides[0]
        ptrs = ptrs + gl.expand_dims(offs_n, 0) * p.out_desc.strides[1]
        gl.store(ptrs, out, mask=mask)


@gluon.jit
def ws_matmul_kernel_optimized(
    x_desc,
    w_desc,
    scale_desc,
    out_desc,
    out_ptr,
    #
    bias_ptr,
    bias_stride,
    #
    gather_indx_ptr,
    #
    x_slice_sizes,
    x_slice_offs,
    x_block_offs,
    x_block_schedule,
    #
    x_scale_ptr,
    w_scale_ptr,
    out_scale_ptr,
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
    XCD_SWIZZLE: gl.constexpr,
    N_MAJOR: gl.constexpr,
    NUM_SMS: gl.constexpr,
    SCALE_SIZE_OUTER: gl.constexpr,
    SCALE_SIZE_INNER: gl.constexpr,
    MXFP_BLOCK_SIZE: gl.constexpr,
):
    PACKED_BLOCK_K: gl.constexpr = w_desc.block_type.shape[2]

    grid_m = gl.load(x_block_offs + NUM_SLICES)
    GRID_N: gl.constexpr = triton.cdiv(N, BLOCK_N)
    K_TILES: gl.constexpr = triton.cdiv(K, BLOCK_K)
    SCALE_FLAT_N: gl.constexpr = N // SCALE_SIZE_OUTER
    SCALE_BLOCK_N_DIV: gl.constexpr = BLOCK_N // SCALE_SIZE_OUTER
    num_blocks = grid_m * GRID_N

    SCALE_K: gl.constexpr = BLOCK_K // MXFP_BLOCK_SIZE
    scale_layout: gl.constexpr = blackwell.TensorMemoryScalesLayout()
    acc_layout: gl.constexpr = blackwell.TensorMemoryLayout([128, BLOCK_M], col_stride=1)

    x_num_bufs: gl.constexpr = 4
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

    w_num_bufs: gl.constexpr = 4
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

    x_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_M, SCALE_K], scale_layout)
    w_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_N, SCALE_K], scale_layout)

    acc_num_bufs: gl.constexpr = 1
    acc_tmem = blackwell.allocate_tensor_memory(gl.float32, [acc_num_bufs, BLOCK_N, BLOCK_M], acc_layout)
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [acc_num_bufs, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [acc_num_bufs, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(acc_num_bufs):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    x_scale_tmem.store(gl.full((BLOCK_M, SCALE_K), 127, dtype=gl.uint8, layout=x_scale_tmem.get_reg_layout()))

    p = ws_base.PartitionArgs(
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
        GRID_N=GRID_N,
        K_TILES=K_TILES,
        SCALE_FLAT_N=SCALE_FLAT_N,
        SCALE_BLOCK_N_DIV=SCALE_BLOCK_N_DIV,
        num_blocks=num_blocks,
        #
        NUM_SMS=NUM_SMS,
        XCD_SWIZZLE=XCD_SWIZZLE,
        N_MAJOR=N_MAJOR,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SCALE_SIZE_OUTER=SCALE_SIZE_OUTER,
        SCALE_SIZE_INNER=SCALE_SIZE_INNER,
        MXFP_BLOCK_SIZE=MXFP_BLOCK_SIZE,
        PACKED_BLOCK_K=PACKED_BLOCK_K,
        #
        SWIGLU_ALPHA=SWIGLU_ALPHA,
        SWIGLU_LIMIT=SWIGLU_LIMIT,
        REDUCTION_N=REDUCTION_N,
        FLEXPOINT_SATURATE_INF=FLEXPOINT_SATURATE_INF,
    )

    # Keep the baseline partitioning, but loosen the weight-loader register
    # budget. This is the only small tactical change that improved the target
    # E256/es8 bucket on this branch.
    gl.warp_specialize(
        [
            (epilogue_partition_optimized, (p,)),
            (ws_base.load_activations, (p,)),
            (ws_base.load_weights, (p,)),
            (ws_base.mma_partition, (p,)),
        ],
        [4, 1, 1],
        [96, 64, 24],
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


def _row_count(m_rows: int, expected_slice_size: int | None, n_slices: int) -> int:
    exp = -1 if expected_slice_size is None else expected_slice_size
    if exp < 0:
        return m_rows
    if exp == 0:
        return max(1, m_rows // 8)
    return exp * n_slices


def _select_kernel_config(m_rows: int, expected_slice_size: int | None, n_slices: int) -> tuple[KernelConfig, int]:
    cfg = KernelConfig()
    row_count = _row_count(m_rows, expected_slice_size, n_slices)
    return cfg, row_count


def _select_launch_grid(
    m_rows: int,
    grid_n: int,
    block_m: int,
    expected_grid_m: int,
    row_count: int,
    sms: int,
) -> int:
    num_tiles = expected_grid_m * grid_n
    return max(1, min(sms, num_tiles))


def matmul(
    a: torch.Tensor | Tensor,
    b: torch.Tensor | Tensor,
    bias: torch.Tensor | None,
    a_ragged_metadata: RaggedTensorMetadata | None = None,
    b_ragged_metadata: RaggedTensorMetadata | None = None,
    gather_indx: torch.Tensor | None = None,
    scatter_indx: torch.Tensor | None = None,
    precision_config: PrecisionConfig | None = None,
    betas: torch.Tensor | None = None,
    gammas: torch.Tensor | None = None,
    out_alpha: float | None = None,
    c: torch.Tensor | None = None,
    fused_comm: FusedComm | None = None,
    fused_activation: FusedActivation | None = None,
    epilogue: Epilogue | None = None,
    c_acc_in: torch.Tensor | None = None,
):
    assert a_ragged_metadata is not None
    assert b_ragged_metadata is None
    assert bias is not None
    assert gather_indx is not None
    assert scatter_indx is None
    assert precision_config is not None
    assert betas is None
    assert gammas is None
    assert out_alpha is None
    assert c is not None
    assert fused_comm is None
    assert fused_activation is not None
    assert epilogue is None
    assert c_acc_in is None

    specs = fused_activation.specs
    assert specs.name == "swiglu"
    reduction_n = specs.reduction_n
    swiglu_alpha, swiglu_limit = fused_activation.fn_args

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

    config, row_count = _select_kernel_config(
        m_rows=m,
        expected_slice_size=a_ragged_metadata.expected_slice_size,
        n_slices=a_ragged_metadata.n_slices,
    )
    mxfp_block_size = 32
    scale_size_outer = 128
    scale_size_inner = 4
    x_block_idx = config.block_m.bit_length() - 5

    expected_grid_m = a_ragged_metadata.n_blocks(a_ragged_metadata.n_slices, m, config.block_m)
    grid_n = triton.cdiv(n, config.block_n)
    sms = torch.cuda.get_device_properties(bias.device).multi_processor_count
    launch_grid = _select_launch_grid(
        m_rows=m,
        grid_n=grid_n,
        block_m=config.block_m,
        expected_grid_m=expected_grid_m,
        row_count=row_count,
        sms=sms,
    )
    grid = (launch_grid,)

    x_desc = ws_base.make_operand_descriptor(a, [1, config.block_k])
    w_desc = ws_base.make_operand_descriptor(b, [1, config.block_n, config.block_k // 2], transposed=True)
    scale_desc = ws_base.make_operand_descriptor(
        b_mx_scales,
        [
            1,
            config.block_n // scale_size_outer,
            config.block_k // mxfp_block_size // scale_size_inner,
            2,
            256,
        ],
    )
    out_desc = ws_base.make_operand_descriptor(c, [config.block_m, config.block_n])

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
        FLEXPOINT_SATURATE_INF=precision_config.flexpoint_saturate_inf,
        #
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        XCD_SWIZZLE=config.xcd_swizzle,
        N_MAJOR=config.n_major,
        NUM_SMS=launch_grid,
        SCALE_SIZE_OUTER=scale_size_outer,
        SCALE_SIZE_INNER=scale_size_inner,
        MXFP_BLOCK_SIZE=mxfp_block_size,
        #
        num_warps=config.num_warps,
    )

    return c.unsqueeze(0)
