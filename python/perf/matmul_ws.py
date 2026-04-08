import torch

from triton_kernels.tensor import RaggedTensorMetadata, Tensor
from triton_kernels.tensor_details.dtype import FP4, UINT8
from triton_kernels.matmul import PrecisionConfig, FusedComm, FusedActivation, Epilogue

import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
import triton.experimental.gluon.language.nvidia.blackwell as blackwell
import triton.experimental.gluon.language.nvidia.blackwell.tma as tma
import triton.experimental.gluon.language.nvidia.hopper.mbarrier as mbarrier
import triton.language.extra.cuda.libdevice as libdevice


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
    block_id,
    grid_m,
    GRID_N: gl.constexpr,
    slice_offsets,
    block_schedule,
    XCD_SWIZZLE: gl.constexpr,
    N_MAJOR: gl.constexpr,
):
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
    ROWS: gl.constexpr = smem.shape[1]
    COLS: gl.constexpr = smem.shape[2] * smem.shape[3] * smem.shape[4]
    TILES: gl.constexpr = COLS // (SIZE_OUTER * SIZE_INNER)
    smem = smem.reshape((ROWS, TILES, MXFP_BLOCK_SIZE, SIZE_OUTER // MXFP_BLOCK_SIZE, SIZE_INNER))
    smem = smem.permute((0, 3, 2, 1, 4))
    return smem.reshape((ROWS * SIZE_OUTER, COLS // SIZE_OUTER))


@gluon.jit
def swiglu(input, ALPHA: gl.constexpr, LIMIT: gl.constexpr):
    gelu, linear = gl.split(gl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    gelu = gelu.to(gl.float32)
    linear = linear.to(gl.float32)
    gelu = gl.minimum(gelu, LIMIT)
    linear = gl.clamp(linear, -LIMIT, LIMIT)
    s = gelu / (1 + libdevice.fast_expf(-ALPHA * gelu))
    return gl.fma(s, linear, s)


@gluon.jit
def ws_matmul_kernel(
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
    NUM_SMS: gl.constexpr,
    SCALE_SIZE_OUTER: gl.constexpr,
    SCALE_SIZE_INNER: gl.constexpr,
    MXFP_BLOCK_SIZE: gl.constexpr,
):
    PACKED_BLOCK_K: gl.constexpr = w_desc.block_type.shape[2]

    x_scale = 1.0 if x_scale_ptr is None else gl.load(x_scale_ptr)
    w_scale = 1.0 if w_scale_ptr is None else gl.load(w_scale_ptr)
    acc_scale = x_scale * w_scale

    grid_m = gl.load(x_block_offs + NUM_SLICES)
    GRID_N: gl.constexpr = triton.cdiv(N, BLOCK_N)
    K_TILES: gl.constexpr = triton.cdiv(K, BLOCK_K)
    SCALE_FLAT_N: gl.constexpr = N // SCALE_SIZE_OUTER
    SCALE_BLOCK_N_DIV: gl.constexpr = BLOCK_N // SCALE_SIZE_OUTER
    num_blocks = grid_m * GRID_N

    x_offsets_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0]))

    x_buf = gl.allocate_shared_memory(x_desc.dtype, [BLOCK_M, x_desc.block_type.shape[1]], x_desc.layout)
    w_buf = gl.allocate_shared_memory(w_desc.dtype, w_desc.block_type.shape, w_desc.layout)
    scale_buf = gl.allocate_shared_memory(scale_desc.dtype, scale_desc.block_type.shape, scale_desc.layout)

    SCALE_K: gl.constexpr = BLOCK_K // MXFP_BLOCK_SIZE
    scale_layout: gl.constexpr = blackwell.TensorMemoryScalesLayout()
    # X and W are swapped.
    acc_layout: gl.constexpr = blackwell.TensorMemoryLayout([BLOCK_N, BLOCK_M], col_stride=1)
    x_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_M, SCALE_K], scale_layout)
    w_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_N, SCALE_K], scale_layout)
    acc_tmem = blackwell.allocate_tensor_memory(gl.float32, [BLOCK_N, BLOCK_M], acc_layout)

    x_scale_tmem.store(gl.full((BLOCK_M, SCALE_K), 127, dtype=gl.uint8, layout=x_scale_tmem.get_reg_layout()))

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0
    use_acc = False

    out_recip = 1.0 / gl.load(out_scale_ptr)

    start_block_id = gl.program_id(0)
    for block_id in range(start_block_id, num_blocks, NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = apply_block_schedule(
            block_id=block_id,
            grid_m=grid_m,
            GRID_N=GRID_N,
            slice_offsets=x_slice_offs,
            block_schedule=x_block_schedule,
            XCD_SWIZZLE=XCD_SWIZZLE,
            N_MAJOR=False,
        )

        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        shape_m = gl.load(x_slice_sizes + slice_idx)

        offs_m = off_m + gl.arange(0, BLOCK_M, layout=x_offsets_layout)
        mask_m = offs_m < shape_m
        offs_x_m = gl.load(
            gather_indx_ptr + slice_offset + offs_m,
            mask=mask_m,
            other=0,
        )

        scale_idx = slice_idx * SCALE_FLAT_N + pid_n * SCALE_BLOCK_N_DIV

        for ki in range(K_TILES):
            off_k_x = ki * BLOCK_K
            off_k_w = ki * PACKED_BLOCK_K
            off_k_scale = off_k_w // 64

            BYTES_PER_STAGE: gl.constexpr = (
                x_desc.block_type.nbytes * BLOCK_M + w_desc.block_type.nbytes + scale_desc.block_type.nbytes
            )
            mbarrier.expect(bar, BYTES_PER_STAGE)
            tma.async_gather(x_desc, offs_x_m, off_k_x, bar, x_buf)
            tma.async_copy_global_to_shared(w_desc, [slice_idx, off_n, off_k_w], bar, w_buf)
            tma.async_copy_global_to_shared(scale_desc, [0, scale_idx, off_k_scale, 0, 0], bar, scale_buf)
            mbarrier.wait(bar, phase)
            phase ^= 1

            blackwell.tcgen05_copy(
                unswizzle_mx_scale(scale_buf, SCALE_SIZE_OUTER, SCALE_SIZE_INNER, MXFP_BLOCK_SIZE), w_scale_tmem
            )
            blackwell.tcgen05_mma_scaled(
                w_buf.reshape((w_buf.shape[1], w_buf.shape[2])),
                x_buf.permute((1, 0)),
                acc_tmem,
                w_scale_tmem,
                x_scale_tmem,
                a_type="e2m1",
                b_type="e4m3",
                use_acc=use_acc,
            )
            blackwell.tcgen05_commit(bar)

            mbarrier.wait(bar, phase)
            phase ^= 1
            use_acc = True

        offs_bias_n = off_n + gl.arange(0, BLOCK_N)
        bias = gl.load(bias_ptr + slice_idx * bias_stride + offs_bias_n)

        acc = acc_tmem.load()
        acc = gl.fma(acc.permute((1, 0)), acc_scale, gl.expand_dims(bias, axis=0))

        split_layout: gl.constexpr = gl.BlockedLayout(
            [1, 2], [1, 32], [gl.num_warps(), 1], [1, 0]
        )
        acc = gl.convert_layout(acc, split_layout)

        out = swiglu(acc, SWIGLU_ALPHA, SWIGLU_LIMIT)
        out_off_n = pid_n * BLOCK_N // REDUCTION_N
        out = out * out_recip
        if FLEXPOINT_SATURATE_INF:
            INF: gl.constexpr = 448.0
            out = gl.clamp(out, -INF, INF)
        out = out.to(out_desc.dtype)

        offs_m = off_m + gl.arange(0, BLOCK_M)
        offs_n = out_off_n + gl.arange(0, BLOCK_N // REDUCTION_N)
        mask = gl.expand_dims(offs_m < shape_m, 1)
        ptrs = out_ptr + gl.expand_dims(slice_offset + offs_m, 1) * out_desc.strides[0]
        ptrs = ptrs + gl.expand_dims(offs_n, 0) * out_desc.strides[1]
        gl.store(ptrs, out, mask=mask)


def get_operand_layout(t: Tensor, block_shape: list[int]):
    rank = len(block_shape)
    if t.dtype == FP4:
        # Weight tensor.
        assert rank == 3
        return gl.NVMMASharedLayout(
            swizzle_byte_width=128,
            element_bitwidth=8,
            rank=rank,
            fp4_padded=True,
        )
    elif t.dtype == UINT8:
        # Weight mx scales tensor.
        assert rank == 5
        return gl.NVMMASharedLayout(
            swizzle_byte_width=0,
            element_bitwidth=8,
            rank=rank,
        )
    elif t.dtype == torch.float32:
        # Bias tensor.
        assert rank == 2
        return gl.NVMMASharedLayout.get_default_for(
            block_shape,
            torch.float32,
        )
    else:
        # Activations or output tensor.
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

    layout = get_operand_layout(t, block_shape)
    return TensorDescriptor(ptr, shape, strides, block_shape, layout)


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
    # Input invariants.
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

    # Extract input fields.
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

    # d_model = 5120
    # d_out = 10240
    #
    # a: [batch_size, d_model]
    # b: [n_expts_tot / n_expt_shards, d_model, d_out)
    assert a.ndim == 2
    T, K = a.shape
    E_local, K, N = b.shape
    M = gather_indx.shape[0]

    # Heuristics.
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    MXFP_BLOCK_SIZE = 32
    SCALE_SIZE_OUTER = 128
    SCALE_SIZE_INNER = 4
    x_block_idx = BLOCK_M.bit_length() - 5

    # Program grid.
    expected_grid_m = a_ragged_metadata.n_blocks(a_ragged_metadata.n_slices, M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    expected_num_tiles = expected_grid_m * grid_n
    # TODO: Gather emperical data on per-expert M tile fragmentation.
    num_sms = torch.cuda.get_device_properties(bias.device).multi_processor_count
    grid = (min(num_sms, expected_num_tiles),)

    # Operands.
    x_desc = make_operand_descriptor(a, [1, BLOCK_K])
    w_desc = make_operand_descriptor(b, [1, BLOCK_N, BLOCK_K // 2], transposed=True)
    scale_desc = make_operand_descriptor(
        b_mx_scales,
        [
            1,
            BLOCK_N // SCALE_SIZE_OUTER,
            BLOCK_K // MXFP_BLOCK_SIZE // SCALE_SIZE_INNER,
            2,
            256,
        ],
    )
    out_desc = make_operand_descriptor(c, [BLOCK_M, BLOCK_N])

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
        M=M,
        N=N,
        K=K,
        NUM_SLICES=a_ragged_metadata.n_slices,
        #
        SWIGLU_ALPHA=swiglu_alpha,
        SWIGLU_LIMIT=swiglu_limit,
        REDUCTION_N=reduction_n,
        #
        FLEXPOINT_SATURATE_INF=precision_config.flexpoint_saturate_inf,
        #
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        XCD_SWIZZLE=1,
        NUM_SMS=num_sms,
        SCALE_SIZE_OUTER=SCALE_SIZE_OUTER,
        SCALE_SIZE_INNER=SCALE_SIZE_INNER,
        MXFP_BLOCK_SIZE=MXFP_BLOCK_SIZE,
        #
        num_warps=4,
    )

    torch.cuda.synchronize()

    return c.unsqueeze(0)
