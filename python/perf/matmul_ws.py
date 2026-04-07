import torch

from triton_kernels.tensor import RaggedTensorMetadata, Tensor
from triton_kernels.tensor_details.dtype import FP4, UINT8
from triton_kernels.matmul import PrecisionConfig, FusedComm, FusedActivation, Epilogue

import triton
from triton.language.core import _aggregate as aggregate
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl


@aggregate
class WSArgs:
    pass


@gluon.jit
def ws_matmul_load_partition(p: WSArgs):
    pass


@gluon.jit
def ws_matmul_mma_partition(p: WSArgs):
    pass


@gluon.jit
def ws_matmul_epilogue_partition(p: WSArgs):
    pass


@gluon.jit
def ws_matmul_kernel(
    x_desc,
    w_desc,
    scale_desc,
    out_desc,
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
    #
    swiglu_alpha: gl.constexpr,
    swiglu_limit: gl.constexpr,
    reduction_n: gl.constexpr,
    #
    FLEXPOINT_SATURATE_INF: gl.constexpr,
    #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    NUM_SMS: gl.constexpr,
    #
    num_warps=4,
):
    x_scale = 1.0 if x_scale_ptr is None else gl.load(x_scale_ptr)
    w_scale = 1.0 if w_scale_ptr is None else gl.load(w_scale_ptr)
    acc_scale = x_scale * w_scale

    pass


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
    shape = list(t.shape)
    strides = list(t.stride())

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

    # d_model = 5120
    # d_out = 10240
    #
    # a: [batch_size, d_model]
    # b: [n_expts_tot / n_expt_shards, d_model, d_out)
    assert a.ndim == 2
    M, K = a.shape
    E, K, N = b.shape

    out_dtype = precision_config.out_dtype
    assert out_dtype is not None

    assert c.ndim == 3
    assert c.shape[0] == 1
    c = c.unsqueeze(0)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    n_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    num_sms = torch.cuda.get_device_properties(bias.device).multi_processor_count
    grid = (min(num_sms, n_tiles),)

    print(b_mx_scales.shape)
    print(b_mx_scales.layout)

    x_desc = make_operand_descriptor(a, [1, BLOCK_K])
    w_desc = make_operand_descriptor(b, [1, BLOCK_N, BLOCK_K // 2], transposed=True)
    scale_desc = make_operand_descriptor(b_mx_scales, [1, BLOCK_N // 128, BLOCK_K // 32 // 4, 2, 256])
    out_desc = make_operand_descriptor(c, [BLOCK_M, BLOCK_N])

    flex_ctx = precision_config.flex_ctx
    x_block_idx = BLOCK_M.bit_length() - 5

    print(a_ragged_metadata.slice_sizes.shape)
    print(a_ragged_metadata.slice_offs.shape)
    print(a_ragged_metadata.block_offs_data[x_block_idx].shape)
    print(a_ragged_metadata.block_schedule_data[x_block_idx].shape)
    print(a_ragged_metadata.slice_sizes)
    print(a_ragged_metadata.slice_offs)
    print(a_ragged_metadata.block_offs_data[x_block_idx])
    print(a_ragged_metadata.block_schedule_data[x_block_idx])

    ws_matmul_kernel[grid](
        x_desc,
        w_desc,
        scale_desc,
        out_desc,
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
        #
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        reduction_n=reduction_n,
        #
        FLEXPOINT_SATURATE_INF=precision_config.flexpoint_saturate_inf,
        #
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        XCD_SWIZZLE=True,
        NUM_SMS=num_sms,
        #
        num_warps=4,
    )

    torch.cuda.synchronize()

    return c.squeeze(0)
