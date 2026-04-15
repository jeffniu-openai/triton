"""Shared Gluon TMEM test helpers.

Keep reusable kernels and tensor descriptor builders here instead of importing
from one test file into another.  Pytest adds this directory to the import path
when collecting the Gluon tests, so sibling tests can import this module
without importing from another test file.
"""

import math
import re

import torch
import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryLinearLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma,
    tcgen05_mma_scaled,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton._C.libtriton.gluon_ir import make_cga_layout

def _make_tmem_linear_layout(m, n):
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(int(math.log2(m)))],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))],
        shape=[m, n],
    )


def _make_tmem_linear_layout_mmav5_twocta(m, n):
    assert m >= 128 and (m & (m - 1)) == 0
    assert n >= 1 and (n & (n - 1)) == 0
    tile_m = m // 2
    assert tile_m in (64, 128)
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(int(math.log2(tile_m)))],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))],
        block_bases=[[tile_m, 0]],
        shape=[m, n],
        two_ctas=True,
    )


def _extract_tcgen05_opcode_offsets(asm: str, opcodes=("ld", "st")):
    pattern = re.compile(
        rf"(tcgen05\.(?:{'|'.join(opcodes)})(?:\.red)?\.sync\.aligned\.[^\s;\"]+)"
        r"[^\n\[]*?\[\s*[^\]\n]*?\+\s*(\d+)\s*\]"
    )
    return [(opcode, int(offset)) for opcode, offset in pattern.findall(asm)]


def _extract_tcgen05_cp_opcodes(asm: str):
    pattern = re.compile(
        r"(tcgen05\.cp(?:\.cta_group::\d+)?(?:\.warpx[24](?:::[^\s.;]+)*)?\.\d+x\d+b)"
    )
    return pattern.findall(asm)


def _extract_tcgen05_mma_opcodes(asm: str):
    pattern = re.compile(r"(tcgen05\.mma\.cta_group::\d+\.kind::[^\s;\"]+)")
    return pattern.findall(asm)


def _expected_scaled_mma_opcode(a_format, b_format, num_ctas):
    prefix = f"tcgen05.mma.cta_group::{2 if num_ctas == 2 else 1}.kind::"
    if a_format == "nvfp4" and b_format == "nvfp4":
        return prefix + "mxf4nvf4.block_scale.scale_vec::4X"
    if a_format == "mxfp4" and b_format == "mxfp4":
        return prefix + "mxf4.block_scale.scale_vec::2X"
    return prefix + "mxf8f6f4.block_scale.scale_vec::1X"


def _expected_scaled_cp_opcode(num_ctas):
    return f"tcgen05.cp.cta_group::{2 if num_ctas == 2 else 1}.warpx4.32x128b"


@gluon.jit
def mma_kernel(a, b, out, M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr, block_layout_a: ttgl.constexpr,
               block_layout_b: ttgl.constexpr, cga_layout_c: ttgl.constexpr, acc_layout: ttgl.constexpr,
               shared_layout_a: ttgl.constexpr, shared_layout_b: ttgl.constexpr, acc_dtype: ttgl.constexpr,
               ASYNC: ttgl.constexpr, USE_TCGEN05: ttgl.constexpr):
    a_offs_m = ttgl.arange(0, M)[:, None]
    a_offs_k = ttgl.arange(0, K)[None, :]
    b_offs_k = ttgl.arange(0, K)[:, None]
    b_offs_n = ttgl.arange(0, N)[None, :]

    operand_dtype = a.dtype.element_ty
    a_ptrs = a + a_offs_m * K + a_offs_k
    b_ptrs = b + b_offs_k * N + b_offs_n
    a_tile = ttgl.load(ttgl.set_auto_layout(a_ptrs, block_layout_a))
    b_tile = ttgl.load(ttgl.set_auto_layout(b_ptrs, block_layout_b))

    smem_a = ttgl.allocate_shared_memory(operand_dtype, [M, K], shared_layout_a, a_tile)
    smem_b = ttgl.allocate_shared_memory(operand_dtype, [K, N], shared_layout_b, b_tile)

    if USE_TCGEN05:
        two_ctas: ttgl.constexpr = acc_layout.two_ctas
        fence_async_shared(cluster=two_ctas)
        mma_barrier = mbarrier.allocate_mbarrier()
        mbarrier.init(mma_barrier, count=1)

        acc_tmem = allocate_tensor_memory(acc_dtype, [M, N], acc_layout)

        tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=False, mbarriers=[mma_barrier])
        mbarrier.wait(mma_barrier, phase=0, deps=[smem_a, smem_b])
        mbarrier.invalidate(mma_barrier)

        acc = acc_tmem.load()
    else:
        acc = ttgl.zeros([M, N], dtype=acc_dtype, layout=acc_layout)
        acc = hopper.warpgroup_mma(smem_a, smem_b, acc, is_async=ASYNC)

        if ASYNC:
            acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])

    out_offs_m = ttgl.arange(0, M)[:, None]
    out_offs_n = ttgl.arange(0, N)[None, :]
    out_ptrs = out + out_offs_m * N + out_offs_n
    ttgl.store(out_ptrs, acc)


def make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, two_cta_dim):
    ctas_per_cga = list(ctas_per_cga)
    cta_split = list(cta_split)
    assert cta_split[two_cta_dim] > 1
    cta_split[two_cta_dim] //= 2
    ctas_per_cga[two_cta_dim] //= 2
    aux_cga_layout = make_cga_layout(ctas_per_cga, cta_split, cta_order)
    assert two_cta_dim in (0, 1)
    basis = [0, 0]
    basis[two_cta_dim] = 1
    for b in aux_cga_layout:
        b[two_cta_dim] *= 2
    cga_layout = [basis] + aux_cga_layout
    return cga_layout


@gluon.jit
def tmem_reduction_kernel(
    in_ptr,
    out_ptr,
    red_ptr,
    layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    RED_OP: ttgl.constexpr,
    USE_ABS: ttgl.constexpr,
    PROPAGATE_NAN: ttgl.constexpr,
    num_warps: ttgl.constexpr,
):
    """Kernel to test TMEM load with hardware reduction."""
    global_memory_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    global_memory_layout_1d: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [num_warps], [0])

    # Offsets for 2D tensor
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, global_memory_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, global_memory_layout))
    offs_2d = offs_m[:, None] * N + offs_n[None, :]

    # Load input from global memory
    input_data = ttgl.load(in_ptr + offs_2d)

    # Allocate TMEM
    tmem = allocate_tensor_memory(
        element_ty=in_ptr.dtype.element_ty,
        shape=[M, N],
        layout=layout,
    )

    # Get register layout for TMEM access
    tmem_reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    # Store input to TMEM
    input_data = ttgl.convert_layout(input_data, tmem_reg_layout)
    tmem.store(input_data)

    # Load from TMEM with reduction
    if RED_OP == "min":
        output, reduced = tmem.load_min(abs=USE_ABS, propagate_nan=PROPAGATE_NAN)
    elif RED_OP == "max":
        output, reduced = tmem.load_max(abs=USE_ABS, propagate_nan=PROPAGATE_NAN)

    # Store full output
    output = ttgl.convert_layout(output, global_memory_layout)
    ttgl.store(out_ptr + offs_2d, output)

    # Store reduced output (1D tensor of shape [M])
    offs_1d = ttgl.arange(0, M, global_memory_layout_1d)
    reduced = ttgl.convert_layout(reduced, global_memory_layout_1d)
    ttgl.store(red_ptr + offs_1d, reduced)


def _run_tmem_reduction_case(
    layout,
    M,
    N,
    red_op,
    use_abs,
    propagate_nan,
    num_warps,
    expect_hw_reduce=True,
    expected_red_opcode_prefix="tcgen05.ld.red.sync.aligned.32x32b.x",
):
    input_tensor = torch.randn(M, N, dtype=torch.float32, device="cuda")

    use_nan = propagate_nan == tl.PropagateNan.ALL
    if use_nan:
        input_tensor[10, 5] = float("nan")
        input_tensor[50, 15] = float("nan")

    output = torch.empty_like(input_tensor)
    red_output = torch.empty(M, dtype=torch.float32, device="cuda")

    compiled = tmem_reduction_kernel[(1, )](
        input_tensor,
        output,
        red_output,
        layout,
        M,
        N,
        red_op,
        use_abs,
        propagate_nan,
        num_warps=num_warps,
    )

    torch.testing.assert_close(input_tensor, output, atol=0, rtol=0, equal_nan=use_nan)

    ref_input = torch.abs(input_tensor) if use_abs else input_tensor
    torch_red = torch.min if red_op == "min" else torch.max
    expected_red = torch_red(ref_input, dim=1).values
    torch.testing.assert_close(expected_red, red_output, atol=1e-5, rtol=1e-5, equal_nan=use_nan)

    ptx_red_ops = [
        op for op, _ in _extract_tcgen05_opcode_offsets(compiled.asm["ptx"], opcodes=("ld", )) if ".ld.red." in op
    ]
    llir_red_ops = [
        op for op, _ in _extract_tcgen05_opcode_offsets(compiled.asm["llir"], opcodes=("ld", )) if ".ld.red." in op
    ]
    assert ptx_red_ops == llir_red_ops
    if expect_hw_reduce:
        assert ptx_red_ops
        expected_modifier = f".{red_op}"
        if use_abs:
            expected_modifier += ".abs"
        if propagate_nan == tl.PropagateNan.ALL:
            expected_modifier += ".NaN"
        expected_modifier += ".f32"
        if expected_red_opcode_prefix is not None:
            assert all(op.startswith(expected_red_opcode_prefix) for op in ptx_red_ops)
        assert all(expected_modifier in op for op in ptx_red_ops)
    else:
        assert not ptx_red_ops
        ptx_ld_ops = [
            op for op, _ in _extract_tcgen05_opcode_offsets(compiled.asm["ptx"], opcodes=("ld", ))
            if op.startswith("tcgen05.ld.sync.aligned.")
        ]
        assert ptx_ld_ops
    return compiled


def align_to(a, b):
    return triton.cdiv(a, b) * b


def make_operand_descriptor(value, BLOCK_MN, BLOCK_K, MIXED_PREC, cga_layout=None):
    IS_FP4 = value.dtype == torch.uint8
    ELEM_PER_BYTE = 2 if IS_FP4 else 1
    IS_MIXED_PREC_FP4 = MIXED_PREC and IS_FP4
    layout = ttgl.NVMMASharedLayout.get_default_for(
        [BLOCK_MN, BLOCK_K // ELEM_PER_BYTE],
        ttgl.uint8 if IS_FP4 else ttgl.float8e4nv,
        fp4_padded=IS_MIXED_PREC_FP4,
        cga_layout=cga_layout,
    )
    return TensorDescriptor.from_tensor(value, [BLOCK_MN, BLOCK_K // ELEM_PER_BYTE], layout)


def make_output_descriptor(M, N, dtype, BLOCK_M, BLOCK_N, cga_layout=None):
    C = torch.empty(M, N, device="cuda", dtype=dtype)
    C_dtype = getattr(ttgl, str(dtype).split('.')[1])
    C_desc_layout = ttgl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], C_dtype, cga_layout=cga_layout)
    return TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], C_desc_layout)


def random_quantized_tensor(MN, K, format):
    assert format in ["mxfp4", "mxfp8", "nvfp4"]
    VEC_SIZE = 16 if format == "nvfp4" else 32
    base = MXFP4Tensor(size=(MN, K), device="cuda").random()
    scale = MXScaleTensor(size=(MN, K // VEC_SIZE), device="cuda").random(low=1 / 128, high=2.0)
    ref = base.to(torch.float32)
    scale_ref = scale.to(torch.float32)
    value = ref * scale_ref.repeat_interleave(VEC_SIZE, dim=1)
    if format == "mxfp8":
        return ref.to(torch.float8_e4m3fn), scale.data, value
    elif format == "mxfp4":
        return base.to_packed_tensor(dim=1), scale.data, value
    else:
        return base.to_packed_tensor(dim=1), scale_ref.to(torch.float8_e4m3fn), value


def swizzle_scales_packed_block(scales, VEC_SIZE):
    PAD_MN = align_to(scales.shape[0], 128) - scales.shape[0]
    PAD_K = align_to(scales.shape[1], 4) - scales.shape[1]
    scales = torch.nn.functional.pad(scales, (0, PAD_K, 0, PAD_MN))
    MN, SCALE_K = scales.shape[0], scales.shape[1]
    REP_MN = MN // 128
    REP_K = SCALE_K // 4
    scales = scales.reshape(REP_MN, 4, 32, REP_K, 4)
    scales = scales.permute(0, 3, 2, 1, 4)
    return scales.contiguous()


def make_scales_descriptor(scales, BLOCK_MN, BLOCK_K, VEC_SIZE, cga_layout=None):
    REP_MN = BLOCK_MN // 128
    REP_K = BLOCK_K // (VEC_SIZE * 4)
    block_shape = [1, REP_MN, REP_K, 2, 256]
    scales = scales.reshape(1, scales.shape[0], scales.shape[1], 2, 256)
    IS_NVFP4 = scales.dtype == torch.float8_e4m3fn
    layout = ttgl.NVMMASharedLayout.get_default_for(block_shape, ttgl.float8e4nv if IS_NVFP4 else ttgl.uint8,
                                                    cga_layout=cga_layout)
    return TensorDescriptor.from_tensor(scales, block_shape, layout)


@gluon.jit
def unswizzle_scales_shared_memory(smem, BLOCK_MN: ttgl.constexpr, BLOCK_K: ttgl.constexpr, VEC_SIZE: ttgl.constexpr):
    smem = smem.reshape((smem.shape[1], smem.shape[2], 32, 4, 4))
    smem = smem.permute((0, 3, 2, 1, 4))
    return smem.reshape((BLOCK_MN, BLOCK_K // VEC_SIZE))


@gluon.jit
def mma_scaled_tcgen05_copy_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, VEC_SIZE: ttgl.constexpr,
                                   block_layout_c: ttgl.constexpr, acc_tmem_layout: ttgl.constexpr,
                                   ACC_INIT: ttgl.constexpr, multicast: ttgl.constexpr):
    A_IS_FP4: ttgl.constexpr = a_desc.dtype == ttgl.uint8
    B_IS_FP4: ttgl.constexpr = b_desc.dtype == ttgl.uint8
    A_ELEM_PER_BYTE: ttgl.constexpr = 2 if A_IS_FP4 else 1
    B_ELEM_PER_BYTE: ttgl.constexpr = 2 if B_IS_FP4 else 1
    BLOCK_M: ttgl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: ttgl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: ttgl.constexpr = a_desc.block_type.shape[1] * A_ELEM_PER_BYTE
    K = a_desc.shape[1] * A_ELEM_PER_BYTE

    a_smem = ttgl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = ttgl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)

    num_ctas: ttgl.constexpr = ttgl.num_ctas()
    two_ctas: ttgl.constexpr = num_ctas > 1
    scale_layout_a: ttgl.constexpr = TensorMemoryScalesLayout(cga_layout=[[1, 0]] if two_ctas else [])
    scale_layout_b: ttgl.constexpr = TensorMemoryScalesLayout(cga_layout=[[0, 0]] if two_ctas else [])
    a_scale_tmem = allocate_tensor_memory(a_scale_desc.dtype, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout_a)
    b_scale_tmem = allocate_tensor_memory(b_scale_desc.dtype, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout_b)
    acc_tmem = allocate_tensor_memory(ttgl.float32, [BLOCK_M, BLOCK_N], acc_tmem_layout)
    if ACC_INIT != 0.0:
        acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
        acc_tmem.store(
            ttgl.full([BLOCK_M, BLOCK_N], ACC_INIT, ttgl.float32, layout=acc_reg_layout)
        )

    tma_bar = mbarrier.allocate_mbarrier(two_ctas=two_ctas)
    mma_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(tma_bar, count=1)
    mbarrier.init(mma_bar, count=1)

    phase_tma = 0
    phase_mma = 0
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_scale_smem = ttgl.allocate_shared_memory(a_scale_desc.dtype, a_scale_desc.block_type.shape, a_scale_desc.layout)
    b_scale_smem = ttgl.allocate_shared_memory(b_scale_desc.dtype, b_scale_desc.block_type.shape, b_scale_desc.layout)
    REP_M: ttgl.constexpr = a_scale_desc.block_type.shape[1]
    REP_N: ttgl.constexpr = b_scale_desc.block_type.shape[1]
    A_REP_K: ttgl.constexpr = a_scale_desc.block_type.shape[2]
    B_REP_K: ttgl.constexpr = b_scale_desc.block_type.shape[2]
    off_m_a_scale = pid_m * REP_M
    off_n_b_scale = pid_n * REP_N

    for k in range(0, K, BLOCK_K):
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE
        off_k_a_scale = (k // BLOCK_K) * A_REP_K
        off_k_b_scale = (k // BLOCK_K) * B_REP_K

        EXPECTED_BYTES: ttgl.constexpr = (a_desc.nbytes_per_cta + b_desc.nbytes_per_cta + a_scale_desc.nbytes_per_cta +
                                          b_scale_desc.nbytes_per_cta)
        mbarrier.expect(tma_bar, EXPECTED_BYTES)
        tma.async_copy_global_to_shared(a_desc, [off_m, off_k_a], tma_bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [off_n, off_k_b], tma_bar, b_smem)
        tma.async_copy_global_to_shared(a_scale_desc, [0, off_m_a_scale, off_k_a_scale, 0, 0], tma_bar, a_scale_smem)
        tma.async_copy_global_to_shared(b_scale_desc, [0, off_n_b_scale, off_k_b_scale, 0, 0], tma_bar, b_scale_smem,
                                        multicast=multicast)
        mbarrier.wait(tma_bar, phase_tma, deps=[a_smem, b_smem, a_scale_smem, b_scale_smem])
        phase_tma ^= 1

        a_scale = unswizzle_scales_shared_memory(a_scale_smem, BLOCK_M, BLOCK_K, VEC_SIZE)
        b_scale = unswizzle_scales_shared_memory(b_scale_smem, BLOCK_N, BLOCK_K, VEC_SIZE)
        tcgen05_copy(a_scale, a_scale_tmem)
        tcgen05_copy(b_scale, b_scale_tmem)

        a_format: ttgl.constexpr = "e2m1" if A_IS_FP4 else "e4m3"
        b_format: ttgl.constexpr = "e2m1" if B_IS_FP4 else "e4m3"
        tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, a_format, b_format,
                           use_acc=(ACC_INIT != 0.0 or k != 0))
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase_mma)
        phase_mma ^= 1

    mbarrier.invalidate(tma_bar)
    mbarrier.invalidate(mma_bar)
    acc = acc_tmem.load()
    if two_ctas:
        acc = ttgl.convert_layout(acc, block_layout_c)
    acc = acc.to(c_desc.dtype)
    acc_smem = ttgl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    acc_smem.store(acc)
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)


def mma_scaled_tcgen05_copy(A, B, A_scale, B_scale, VEC_SIZE, BLOCK_M, BLOCK_N, BLOCK_K, num_ctas, multicast,
                            out_dtype=torch.float16, acc_layout_kind="legacy", acc_init=0.0):
    from dataclasses import replace
    M, N = A.shape[0], B.shape[0]
    MIXED_PREC = A.dtype != B.dtype
    two_ctas = num_ctas > 1
    warps = [4, 1]
    num_warps = warps[0] * warps[1]

    ctas_per_cga = (2, 1) if two_ctas else None
    cta_order = (1, 0) if two_ctas else None
    cta_split = (2, 1) if two_ctas else None
    cga_layout_a = make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, 0) if two_ctas else None
    cga_layout_b = make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, 0) if two_ctas else None
    cga_layout_c = make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0) if two_ctas else None

    # Scale tensors have shape [1, REP_MN, REP_K, 2, 256] (5 dimensions)
    # CGA layout basis vectors describe how indices change between CTAs
    # A_scale: [0, 1, 0, 0, 0] split A scales along REP_M axis across CTAs
    # B_scale: [0, 0, 0, 0, 0] duplicate (no split) B scales to all CTAs
    cga_layout_a_scale = [[0, 1, 0, 0, 0]] if two_ctas else None
    cga_layout_b_scale = [[0, 0, 0, 0, 0]] if two_ctas else None

    A_desc = make_operand_descriptor(A, BLOCK_M, BLOCK_K, MIXED_PREC, cga_layout=cga_layout_a)
    B_desc = make_operand_descriptor(B, BLOCK_N, BLOCK_K, MIXED_PREC, cga_layout=cga_layout_b)
    C_desc = make_output_descriptor(M, N, out_dtype, BLOCK_M, BLOCK_N, cga_layout=cga_layout_c)
    A_scale_desc = make_scales_descriptor(A_scale, BLOCK_M, BLOCK_K, VEC_SIZE, cga_layout=cga_layout_a_scale)
    B_scale_desc = make_scales_descriptor(B_scale, BLOCK_N, BLOCK_K, VEC_SIZE, cga_layout=cga_layout_b_scale)

    a_scale_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5,
                                            cga_layout=cga_layout_a_scale)
    b_scale_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5,
                                            cga_layout=cga_layout_b_scale)
    A_scale_desc = replace(A_scale_desc, layout=a_scale_layout)
    B_scale_desc = replace(B_scale_desc, layout=b_scale_layout)

    block_layout_c = ttgl.BlockedLayout([1, 8], [1, 32], warps_per_cta=warps, order=[1, 0],
                                        cga_layout=cga_layout_c) if two_ctas else None
    if acc_layout_kind == "legacy":
        acc_tmem_layout = TensorMemoryLayout(
            [BLOCK_M // num_ctas, BLOCK_N],
            col_stride=1,
            cga_layout=[[1, 0]] if two_ctas else [],
            two_ctas=two_ctas,
        )
    elif acc_layout_kind == "linear":
        acc_tmem_layout = (_make_tmem_linear_layout_mmav5_twocta(BLOCK_M, BLOCK_N)
                           if two_ctas else _make_tmem_linear_layout(BLOCK_M, BLOCK_N))
    else:
        raise ValueError(f"unsupported acc_layout_kind: {acc_layout_kind}")

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    compiled = mma_scaled_tcgen05_copy_kernel[grid](
        A_desc,
        B_desc,
        C_desc,
        A_scale_desc,
        B_scale_desc,
        VEC_SIZE,
        block_layout_c,
        acc_tmem_layout,
        acc_init,
        num_warps=num_warps,
        num_ctas=num_ctas,
        multicast=multicast,
    )
    return C_desc.base, compiled
