from __future__ import annotations

import copy
from dataclasses import dataclass
import os

import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
import triton.experimental.gluon.language.nvidia.blackwell as blackwell
from triton.experimental.gluon.language.nvidia.blackwell import float2
import triton.experimental.gluon.language.nvidia.hopper.mbarrier as mbarrier
import triton.language.core as tl_core
import triton.language.extra.cuda.libdevice as libdevice

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
    x_num_bufs: int = 5
    w_num_bufs: int = 4
    load_activation_warps: int = 4
    load_weight_warps: int = 1
    mma_warps: int = 1
    store_helper_warps: int = 2
    epilogue_n_elems: int = 4
    epilogue_subtile_n: int = 256
    epilogue_row_subtile_factor: int = 8
    epilogue_n_fragment_factor: int = 1
    epilogue_schedule: int = 1
    epilogue_store_helper: bool = True
    epilogue_store_helper_depth: int = 2
    load_activation_regs: int = 112
    load_weight_regs: int = 48
    mma_regs: int = 24
    store_helper_regs: int = 16
    acc_instr_variant: str = "32x32b"
    use_exp2_sigmoid: bool = False
    use_packed_final_fma: bool = True
    use_packed_fp8_store: bool = True
    use_blocked_packed_store: bool = False
    use_wide_packed_store32: bool = True
    use_wide_packed_store64: bool = False
    use_packed_out_scale: bool = True
    use_linear_acc_epilogue: bool = False
    use_helper_packed_out_buffer: bool = False
    use_helper_swiglu: bool = False


EPILOGUE_SCHEDULE_DIRECT = 0
EPILOGUE_SCHEDULE_WAVEFRONT = 1
EPILOGUE_STORE_HELPER_DISABLED = 0
EPILOGUE_STORE_HELPER_ENABLED = 1


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
def _pack_u16x2(x0, x1):
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
def _pack_u32x2(x0, x1):
    return tl_core.inline_asm_elementwise(
        """
        mov.b64 $0, { $1, $2 };
        """,
        "=l,r,r",
        [x0, x1],
        dtype=tl_core.int64,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _pack_packed_fp8x4(values):
    lhs, rhs = gl.split(values.reshape((values.shape[0], values.shape[1] // 2, 2)))
    return _pack_u16x2(lhs, rhs)


@gluon.jit
def _pack_packed_fp8x8(values):
    packed_out32 = _pack_packed_fp8x4(values)
    packed_out32 = gl.convert_layout(
        packed_out32,
        gl.BlockedLayout([1, 2], [1, 32], [gl.num_warps(), 1], [1, 0]),
    )
    lhs, rhs = gl.split(packed_out32.reshape((packed_out32.shape[0], packed_out32.shape[1] // 2, 2)))
    return _pack_u32x2(lhs, rhs)


@gluon.jit
def _pack_packed_fp8x4_linear64(values):
    pairs = gl.convert_layout(
        values.reshape((values.shape[0], values.shape[1] // 2, 2)),
        _adjacent_linear_pack32_split_layout(),
    )
    lhs, rhs = gl.split(pairs)
    pack_layout: gl.constexpr = _adjacent_linear_pack32_layout()
    lhs = gl.convert_layout(lhs, pack_layout)
    rhs = gl.convert_layout(rhs, pack_layout)
    return _pack_u16x2(lhs, rhs)


@gluon.jit
def _pack_packed_fp8x4_linear32(values):
    pairs = gl.convert_layout(
        values.reshape((values.shape[0], values.shape[1] // 2, 2)),
        _adjacent_linear_pack16_split_layout(),
    )
    lhs, rhs = gl.split(pairs)
    pack_layout: gl.constexpr = _adjacent_linear_pack16_layout()
    lhs = gl.convert_layout(lhs, pack_layout)
    rhs = gl.convert_layout(rhs, pack_layout)
    return _pack_u16x2(lhs, rhs)


@gluon.jit
def _store_packed_out(
    p,
    packed_out,
    off_m,
    out_off_n,
    shape_m,
    slice_offset,
    USE_BLOCKED_PACKED_STORE: gl.constexpr,
    USE_WIDE_PACKED_STORE32: gl.constexpr,
    USE_WIDE_PACKED_STORE64: gl.constexpr,
    ):
    if USE_BLOCKED_PACKED_STORE:
        # Default-off store relayout probe. It still lowers to b16 stores on
        # this branch, but keeping it behind a flag preserves the experiment.
        packed_out = gl.convert_layout(
            packed_out,
            gl.BlockedLayout([1, 2], [1, 32], [gl.num_warps(), 1], [1, 0]),
        )
    if USE_WIDE_PACKED_STORE64:
        packed_out64 = _pack_packed_fp8x8(packed_out)
        out_ptr = p.out_ptr.cast(gl.pointer_type(gl.int64), bitcast=True)
        packed64_layout: gl.constexpr = packed_out64.type.layout
        offs_m = off_m + gl.arange(0, packed_out64.shape[0], layout=gl.SliceLayout(1, packed64_layout))
        offs_n = out_off_n // 8 + gl.arange(0, packed_out64.shape[1], layout=gl.SliceLayout(0, packed64_layout))
        mask = gl.expand_dims(offs_m < shape_m, 1)
        ptrs = out_ptr + gl.expand_dims(slice_offset + offs_m, 1) * (p.out_desc.strides[0] // 8)
        ptrs = ptrs + gl.expand_dims(offs_n, 0) * p.out_desc.strides[1]
        gl.store(ptrs, packed_out64, mask=mask)
        return
    if USE_WIDE_PACKED_STORE32:
        packed_out_layout: gl.constexpr = packed_out.type.layout
        if isinstance(packed_out_layout, gl.DistributedLinearLayout) or (
            isinstance(packed_out_layout, gl.SliceLayout)
            and isinstance(packed_out_layout.parent, gl.DistributedLinearLayout)
        ):
            if packed_out.shape[1] == 64:
                packed_out32 = _pack_packed_fp8x4_linear64(packed_out)
            elif packed_out.shape[1] == 32:
                packed_out32 = _pack_packed_fp8x4_linear32(packed_out)
            else:
                packed_out = gl.convert_layout(
                    packed_out,
                    gl.BlockedLayout([1, 2], [1, 32], [gl.num_warps(), 1], [1, 0]),
                )
                packed_out32 = _pack_packed_fp8x4(packed_out)
        else:
            packed_out32 = _pack_packed_fp8x4(packed_out)
        out_ptr = p.out_ptr.cast(gl.pointer_type(gl.int32), bitcast=True)
        packed32_layout: gl.constexpr = packed_out32.type.layout
        offs_m = off_m + gl.arange(0, packed_out32.shape[0], layout=gl.SliceLayout(1, packed32_layout))
        offs_n = out_off_n // 4 + gl.arange(0, packed_out32.shape[1], layout=gl.SliceLayout(0, packed32_layout))
        mask = gl.expand_dims(offs_m < shape_m, 1)
        ptrs = out_ptr + gl.expand_dims(slice_offset + offs_m, 1) * (p.out_desc.strides[0] // 4)
        ptrs = ptrs + gl.expand_dims(offs_n, 0) * p.out_desc.strides[1]
        gl.store(ptrs, packed_out32, mask=mask)
        return

    out_ptr = p.out_ptr.cast(gl.pointer_type(gl.int16), bitcast=True)
    packed_layout: gl.constexpr = packed_out.type.layout
    offs_m = off_m + gl.arange(0, packed_out.shape[0], layout=gl.SliceLayout(1, packed_layout))
    offs_n = out_off_n // 2 + gl.arange(0, packed_out.shape[1], layout=gl.SliceLayout(0, packed_layout))
    mask = gl.expand_dims(offs_m < shape_m, 1)
    ptrs = out_ptr + gl.expand_dims(slice_offset + offs_m, 1) * (p.out_desc.strides[0] // 2)
    ptrs = ptrs + gl.expand_dims(offs_n, 0) * p.out_desc.strides[1]
    gl.store(ptrs, packed_out, mask=mask)


@gluon.jit
def _store_out(p, out, off_m, out_off_n, shape_m, slice_offset):
    offs_m = off_m + gl.arange(0, out.shape[0], layout=gl.SliceLayout(1, out.type.layout))
    offs_n = out_off_n + gl.arange(0, out.shape[1], layout=gl.SliceLayout(0, out.type.layout))
    mask = gl.expand_dims(offs_m < shape_m, 1)
    ptrs = p.out_ptr + gl.expand_dims(slice_offset + offs_m, 1) * p.out_desc.strides[0]
    ptrs = ptrs + gl.expand_dims(offs_n, 0) * p.out_desc.strides[1]
    gl.store(ptrs, out, mask=mask)


@gluon.jit
def _split_last_dim_in_half(values):
    layout: gl.constexpr = values.type.layout
    if isinstance(layout, gl.DistributedLinearLayout) or (
        isinstance(layout, gl.SliceLayout) and isinstance(layout.parent, gl.DistributedLinearLayout)
    ):
        lhs, rhs = values.reshape((values.shape[0], 2, values.shape[1] // 2)).permute((0, 2, 1)).split()
        split_layout: gl.constexpr = _get_split_last_dim_layout(lhs.type.layout)
        lhs = gl.convert_layout(lhs, split_layout, assert_trivial=True)
        rhs = gl.convert_layout(rhs, split_layout, assert_trivial=True)
        return lhs, rhs
    return gl.split(values.reshape((values.shape[0], 2, values.shape[1] // 2)).permute((0, 2, 1)))


@gluon.constexpr_function
def _get_split_last_dim_layout(layout: gl.constexpr):
    layout = _to_distributed_linear_layout(layout)
    target = [0, layout.shape[1] // 2]
    last_reg_idx = len(layout.reg_bases) - 1
    reg_last = layout.reg_bases[last_reg_idx]

    if reg_last == target:
        return layout

    ret = copy.deepcopy(layout)
    for basis_list in (ret.reg_bases, ret.lane_bases, ret.warp_bases, ret.block_bases):
        for i, basis in enumerate(basis_list):
            if basis == target:
                basis_list[i], ret.reg_bases[last_reg_idx] = reg_last, target
                return ret
    assert False, f"split requires having a basis {target}. Got\n{layout}"


@gluon.constexpr_function
def _to_distributed_linear_layout(layout: gl.constexpr):
    if isinstance(layout, gl.DistributedLinearLayout):
        return layout

    assert isinstance(layout, gl.SliceLayout), "split requires a linear or slice-of-linear layout"
    assert isinstance(layout.parent, gl.DistributedLinearLayout), "split requires a slice of a linear layout"
    dim = layout.dim
    parent = layout.parent
    shape = parent.shape[:dim] + parent.shape[dim + 1 :]
    return gl.DistributedLinearLayout(
        [basis[:dim] + basis[dim + 1 :] for basis in parent.reg_bases],
        [basis[:dim] + basis[dim + 1 :] for basis in parent.lane_bases],
        [basis[:dim] + basis[dim + 1 :] for basis in parent.warp_bases],
        [basis[:dim] + basis[dim + 1 :] for basis in parent.block_bases],
        shape,
    )


@gluon.jit
def _split_packed_last_dim_in_half(values):
    lhs, rhs = _split_last_dim_in_half(values.value)
    return float2.Float2Tensor(lhs), float2.Float2Tensor(rhs)


@gluon.jit
def _split_packed_last_dim_in_half_adjacent(values):
    gl.static_assert(values.value.shape[0] == 128, "adjacent packed split expects full 128-row tiles")
    gl.static_assert(
        values.value.shape[1] == 32 or values.value.shape[1] == 64 or values.value.shape[1] == 128,
        "unsupported adjacent packed split shape",
    )
    if values.value.shape[1] == 128:
        linear_layout: gl.constexpr = _adjacent_linear_pack128_layout()
    elif values.value.shape[1] == 64:
        linear_layout: gl.constexpr = _adjacent_linear_pack64_layout()
    else:
        linear_layout: gl.constexpr = _adjacent_linear_pack32_layout()
    linear_values = gl.convert_layout(values.value, linear_layout)
    lhs, rhs = linear_values.reshape(
        (linear_values.shape[0], 2, linear_values.shape[1] // 2)
    ).permute((0, 2, 1)).split()
    return float2.Float2Tensor(lhs), float2.Float2Tensor(rhs)


@gluon.jit
def _pack_last_dim_in_half(values):
    lhs, rhs = values.reshape((values.shape[0], values.shape[1] // 2, 2)).split()
    return float2.pack2(lhs, rhs)


@gluon.constexpr_function
def _adjacent_linear_pack128_layout():
    return gl.DistributedLinearLayout(
        [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 64]],
        [[0, 8], [1, 0], [0, 1], [0, 2], [0, 4]],
        [[0, 32], [0, 0], [0, 16]],
        [],
        [128, 128],
    )


@gluon.constexpr_function
def _adjacent_linear_pack128_split_layout():
    return gl.DistributedLinearLayout(
        [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0], [0, 0, 1]],
        [[0, 8, 0], [1, 0, 0], [0, 1, 0], [0, 2, 0], [0, 4, 0]],
        [[0, 32, 0], [0, 64, 0], [0, 16, 0]],
        [],
        [128, 128, 2],
    )


@gluon.constexpr_function
def _adjacent_linear_pack64_layout():
    return gl.DistributedLinearLayout(
        [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 32]],
        [[0, 8], [1, 0], [0, 1], [0, 2], [0, 4]],
        [[0, 0], [0, 0], [0, 16]],
        [],
        [128, 64],
    )


@gluon.constexpr_function
def _adjacent_linear_pack64_split_layout():
    return gl.DistributedLinearLayout(
        [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0], [0, 0, 1]],
        [[0, 8, 0], [1, 0, 0], [0, 1, 0], [0, 2, 0], [0, 4, 0]],
        [[0, 32, 0], [0, 0, 0], [0, 16, 0]],
        [],
        [128, 64, 2],
    )


@gluon.constexpr_function
def _adjacent_linear_pack32_layout():
    return gl.DistributedLinearLayout(
        [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 16]],
        [[0, 8], [1, 0], [0, 1], [0, 2], [0, 4]],
        [[0, 0], [0, 0], [0, 0]],
        [],
        [128, 32],
    )


@gluon.constexpr_function
def _adjacent_linear_pack32_split_layout():
    return gl.DistributedLinearLayout(
        [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0], [0, 0, 1]],
        [[0, 8, 0], [1, 0, 0], [0, 1, 0], [0, 2, 0], [0, 4, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 16, 0]],
        [],
        [128, 32, 2],
    )


@gluon.constexpr_function
def _adjacent_linear_pack16_layout():
    return gl.DistributedLinearLayout(
        [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]],
        [[0, 8], [1, 0], [0, 1], [0, 2], [0, 4]],
        [[0, 0], [0, 0], [0, 0]],
        [],
        [128, 16],
    )


@gluon.constexpr_function
def _adjacent_linear_pack16_split_layout():
    return gl.DistributedLinearLayout(
        [[2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0], [0, 0, 1]],
        [[0, 8, 0], [1, 0, 0], [0, 1, 0], [0, 2, 0], [0, 4, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [],
        [128, 16, 2],
    )


@gluon.jit
def _pack_last_dim_adjacent_linear(values):
    gl.static_assert(values.shape[0] == 128, "adjacent linear pack expects full 128-row tiles")
    gl.static_assert(
        values.shape[1] == 32 or values.shape[1] == 64 or values.shape[1] == 128 or values.shape[1] == 256,
        "unsupported adjacent linear pack shape",
    )
    if values.shape[1] == 256:
        pairs = gl.convert_layout(
            values.reshape((values.shape[0], values.shape[1] // 2, 2)),
            _adjacent_linear_pack128_split_layout(),
        )
        lhs, rhs = gl.split(pairs)
        pack_layout: gl.constexpr = _adjacent_linear_pack128_layout()
    elif values.shape[1] == 128:
        pairs = gl.convert_layout(
            values.reshape((values.shape[0], values.shape[1] // 2, 2)),
            _adjacent_linear_pack64_split_layout(),
        )
        lhs, rhs = gl.split(pairs)
        pack_layout: gl.constexpr = _adjacent_linear_pack64_layout()
    elif values.shape[1] == 64:
        pairs = gl.convert_layout(
            values.reshape((values.shape[0], values.shape[1] // 2, 2)),
            _adjacent_linear_pack32_split_layout(),
        )
        lhs, rhs = gl.split(pairs)
        pack_layout: gl.constexpr = _adjacent_linear_pack32_layout()
    else:
        pairs = gl.convert_layout(
            values.reshape((values.shape[0], values.shape[1] // 2, 2)),
            _adjacent_linear_pack16_split_layout(),
        )
        lhs, rhs = gl.split(pairs)
        pack_layout: gl.constexpr = _adjacent_linear_pack16_layout()
    lhs = gl.convert_layout(lhs, pack_layout)
    rhs = gl.convert_layout(rhs, pack_layout)
    return float2.pack2(lhs, rhs)


@gluon.jit
def _split_packed_last_dim_in_half_fragment(values):
    lhs, rhs = gl.split(values.value.reshape((values.value.shape[0], 2, values.value.shape[1] // 2)).permute((0, 2, 1)))
    return float2.Float2Tensor(lhs), float2.Float2Tensor(rhs)


@gluon.jit
def _split_first_dim_in_half(values):
    return gl.split(values.reshape((2, values.shape[0] // 2, values.shape[1])).permute((1, 2, 0)))


@gluon.jit
def _split_first_dim_in_half_packed(values):
    lhs, rhs = _split_first_dim_in_half(values.value)
    return float2.Float2Tensor(lhs), float2.Float2Tensor(rhs)


@gluon.jit
def _prepare_swiglu_fragment_from_packed(acc_packed, limit):
    gelu, linear = float2.unpack2(acc_packed)
    gelu = gl.minimum(gelu.to(gl.float32), limit)
    linear = tl_core.clamp(linear.to(gl.float32), -limit, limit)
    return gelu, linear


@gluon.jit
def _finish_swiglu_fragment_packed(gelu, linear, alpha, USE_PACKED_FINAL_FMA: gl.constexpr):
    den = 1.0 + libdevice.exp(-alpha * gelu)
    activated = gelu / den
    activated_packed = float2.pack(activated, axis=1)
    if USE_PACKED_FINAL_FMA:
        linear_packed = float2.pack(linear, axis=1)
        return float2.fma(activated_packed, linear_packed, activated_packed)
    return float2.pack(gl.fma(activated, linear, activated), axis=1)


@gluon.jit
def _finish_swiglu_fragment_packed_linear(gelu, linear, alpha, USE_PACKED_FINAL_FMA: gl.constexpr):
    den = 1.0 + libdevice.exp(-alpha * gelu)
    activated = gelu / den
    activated_packed = _pack_last_dim_adjacent_linear(activated)
    if USE_PACKED_FINAL_FMA:
        linear_packed = _pack_last_dim_adjacent_linear(linear)
        return float2.fma(activated_packed, linear_packed, activated_packed)
    return _pack_last_dim_adjacent_linear(gl.fma(activated, linear, activated))


@gluon.jit
def _pack_fp8_out_fragment(out_packed, out_recip):
    scaled_out_packed = out_packed * float2.full_like(out_packed, out_recip)
    return _pack_e4m3x2(scaled_out_packed)


@gluon.jit
def _split_acc_packed_rows(acc_packed, row_subtile_factor: gl.constexpr):
    if row_subtile_factor == 32:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        eighth000, eighth001 = _split_first_dim_in_half_packed(quarter00)
        eighth010, eighth011 = _split_first_dim_in_half_packed(quarter01)
        eighth100, eighth101 = _split_first_dim_in_half_packed(quarter10)
        eighth110, eighth111 = _split_first_dim_in_half_packed(quarter11)
        sixteenth0000, sixteenth0001 = _split_first_dim_in_half_packed(eighth000)
        sixteenth0010, sixteenth0011 = _split_first_dim_in_half_packed(eighth001)
        sixteenth0100, sixteenth0101 = _split_first_dim_in_half_packed(eighth010)
        sixteenth0110, sixteenth0111 = _split_first_dim_in_half_packed(eighth011)
        sixteenth1000, sixteenth1001 = _split_first_dim_in_half_packed(eighth100)
        sixteenth1010, sixteenth1011 = _split_first_dim_in_half_packed(eighth101)
        sixteenth1100, sixteenth1101 = _split_first_dim_in_half_packed(eighth110)
        sixteenth1110, sixteenth1111 = _split_first_dim_in_half_packed(eighth111)
        thirtysecond00000, thirtysecond00001 = _split_first_dim_in_half_packed(sixteenth0000)
        thirtysecond00010, thirtysecond00011 = _split_first_dim_in_half_packed(sixteenth0001)
        thirtysecond00100, thirtysecond00101 = _split_first_dim_in_half_packed(sixteenth0010)
        thirtysecond00110, thirtysecond00111 = _split_first_dim_in_half_packed(sixteenth0011)
        thirtysecond01000, thirtysecond01001 = _split_first_dim_in_half_packed(sixteenth0100)
        thirtysecond01010, thirtysecond01011 = _split_first_dim_in_half_packed(sixteenth0101)
        thirtysecond01100, thirtysecond01101 = _split_first_dim_in_half_packed(sixteenth0110)
        thirtysecond01110, thirtysecond01111 = _split_first_dim_in_half_packed(sixteenth0111)
        thirtysecond10000, thirtysecond10001 = _split_first_dim_in_half_packed(sixteenth1000)
        thirtysecond10010, thirtysecond10011 = _split_first_dim_in_half_packed(sixteenth1001)
        thirtysecond10100, thirtysecond10101 = _split_first_dim_in_half_packed(sixteenth1010)
        thirtysecond10110, thirtysecond10111 = _split_first_dim_in_half_packed(sixteenth1011)
        thirtysecond11000, thirtysecond11001 = _split_first_dim_in_half_packed(sixteenth1100)
        thirtysecond11010, thirtysecond11011 = _split_first_dim_in_half_packed(sixteenth1101)
        thirtysecond11100, thirtysecond11101 = _split_first_dim_in_half_packed(sixteenth1110)
        thirtysecond11110, thirtysecond11111 = _split_first_dim_in_half_packed(sixteenth1111)
        return (
            thirtysecond00000,
            thirtysecond00001,
            thirtysecond00010,
            thirtysecond00011,
            thirtysecond00100,
            thirtysecond00101,
            thirtysecond00110,
            thirtysecond00111,
            thirtysecond01000,
            thirtysecond01001,
            thirtysecond01010,
            thirtysecond01011,
            thirtysecond01100,
            thirtysecond01101,
            thirtysecond01110,
            thirtysecond01111,
            thirtysecond10000,
            thirtysecond10001,
            thirtysecond10010,
            thirtysecond10011,
            thirtysecond10100,
            thirtysecond10101,
            thirtysecond10110,
            thirtysecond10111,
            thirtysecond11000,
            thirtysecond11001,
            thirtysecond11010,
            thirtysecond11011,
            thirtysecond11100,
            thirtysecond11101,
            thirtysecond11110,
            thirtysecond11111,
        )
    if row_subtile_factor == 16:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        eighth000, eighth001 = _split_first_dim_in_half_packed(quarter00)
        eighth010, eighth011 = _split_first_dim_in_half_packed(quarter01)
        eighth100, eighth101 = _split_first_dim_in_half_packed(quarter10)
        eighth110, eighth111 = _split_first_dim_in_half_packed(quarter11)
        sixteenth0000, sixteenth0001 = _split_first_dim_in_half_packed(eighth000)
        sixteenth0010, sixteenth0011 = _split_first_dim_in_half_packed(eighth001)
        sixteenth0100, sixteenth0101 = _split_first_dim_in_half_packed(eighth010)
        sixteenth0110, sixteenth0111 = _split_first_dim_in_half_packed(eighth011)
        sixteenth1000, sixteenth1001 = _split_first_dim_in_half_packed(eighth100)
        sixteenth1010, sixteenth1011 = _split_first_dim_in_half_packed(eighth101)
        sixteenth1100, sixteenth1101 = _split_first_dim_in_half_packed(eighth110)
        sixteenth1110, sixteenth1111 = _split_first_dim_in_half_packed(eighth111)
        return (
            sixteenth0000,
            sixteenth0001,
            sixteenth0010,
            sixteenth0011,
            sixteenth0100,
            sixteenth0101,
            sixteenth0110,
            sixteenth0111,
            sixteenth1000,
            sixteenth1001,
            sixteenth1010,
            sixteenth1011,
            sixteenth1100,
            sixteenth1101,
            sixteenth1110,
            sixteenth1111,
        )
    if row_subtile_factor == 8:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        eighth000, eighth001 = _split_first_dim_in_half_packed(quarter00)
        eighth010, eighth011 = _split_first_dim_in_half_packed(quarter01)
        eighth100, eighth101 = _split_first_dim_in_half_packed(quarter10)
        eighth110, eighth111 = _split_first_dim_in_half_packed(quarter11)
        return (
            eighth000,
            eighth001,
            eighth010,
            eighth011,
            eighth100,
            eighth101,
            eighth110,
            eighth111,
        )
    if row_subtile_factor == 4:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        return (quarter00, quarter01, quarter10, quarter11)
    if row_subtile_factor == 2:
        return _split_first_dim_in_half_packed(acc_packed)
    return (acc_packed,)


@gluon.constexpr_function
def _store_helper_fragment_layout(frag_rows: gl.constexpr, helper_num_warps: gl.constexpr):
    return gl.BlockedLayout(
        [frag_rows // helper_num_warps, 2],
        [1, 32],
        [helper_num_warps, 1],
        [1, 0],
    )


@gluon.constexpr_function
def _store_helper_acc_row32_layout():
    parent = gl.DistributedLinearLayout(
        [[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]],
        [[0, 64, 0]],
        [],
        [4, 128, 2],
    )
    return gl.SliceLayout(2, parent)


@gluon.jit
def _enqueue_store_helper_out_fragment(
    out_packed,
    out_recip,
    store_bufs,
    store_empty_bars,
    store_ready_bars,
    store_idx,
    store_phase,
    USE_HELPER_PACKED_OUT_BUFFER: gl.constexpr,
    STORE_HELPER_DEPTH: gl.constexpr,
):
    gl.static_assert(STORE_HELPER_DEPTH >= 1, "store helper depth must be at least 1")
    empty_bar = store_empty_bars.index(store_idx)
    ready_bar = store_ready_bars.index(store_idx)
    mbarrier.wait(empty_bar, store_phase)
    payload = out_packed.value if USE_HELPER_PACKED_OUT_BUFFER else _pack_fp8_out_fragment(out_packed, out_recip)
    store_bufs.index(store_idx).store(payload)
    mbarrier.arrive(ready_bar)
    return ws_base.advance(store_idx, store_phase, STORE_HELPER_DEPTH)


@gluon.jit
def _enqueue_store_helper_acc_fragment(
    acc_packed,
    store_bufs,
    store_empty_bars,
    store_ready_bars,
    store_idx,
    store_phase,
    STORE_HELPER_DEPTH: gl.constexpr,
):
    gl.static_assert(STORE_HELPER_DEPTH >= 1, "store helper depth must be at least 1")
    empty_bar = store_empty_bars.index(store_idx)
    ready_bar = store_ready_bars.index(store_idx)
    mbarrier.wait(empty_bar, store_phase)
    payload = acc_packed.value
    store_bufs.index(store_idx).store(payload)
    mbarrier.arrive(ready_bar)
    return ws_base.advance(store_idx, store_phase, STORE_HELPER_DEPTH)


@gluon.jit
def _store_packed_out_fragments(
    p: ws_base.PartitionArgs,
    out_packed,
    out_recip,
    off_m,
    out_off_n,
    shape_m,
    slice_offset,
    USE_PACKED_FP8_STORE: gl.constexpr,
    USE_BLOCKED_PACKED_STORE: gl.constexpr,
    USE_WIDE_PACKED_STORE32: gl.constexpr,
    USE_WIDE_PACKED_STORE64: gl.constexpr,
    USE_PACKED_OUT_SCALE: gl.constexpr,
):
    scaled_out_packed = out_packed
    if USE_PACKED_OUT_SCALE or USE_PACKED_FP8_STORE:
        scaled_out_packed = out_packed * float2.full_like(out_packed, out_recip)

    if USE_PACKED_FP8_STORE:
        packed_fp8 = _pack_e4m3x2(scaled_out_packed)
        _store_packed_out(
            p,
            packed_fp8,
            off_m,
            out_off_n,
            shape_m,
            slice_offset,
            USE_BLOCKED_PACKED_STORE,
            USE_WIDE_PACKED_STORE32,
            USE_WIDE_PACKED_STORE64,
        )
        return

    out_frag = float2.unpack(scaled_out_packed if USE_PACKED_OUT_SCALE else out_packed, axis=1)
    if not USE_PACKED_OUT_SCALE:
        out_frag = out_frag * out_recip
    if p.FLEXPOINT_SATURATE_INF:
        out_frag = tl_core.clamp(out_frag, -448.0, 448.0)
    _store_out(p, out_frag.to(p.out_desc.dtype), off_m, out_off_n, shape_m, slice_offset)


@gluon.jit
def _store_scalar_out_fragment(
    p: ws_base.PartitionArgs,
    out_frag,
    out_recip,
    off_m,
    out_off_n,
    shape_m,
    slice_offset,
    USE_PACKED_FP8_STORE: gl.constexpr,
    USE_BLOCKED_PACKED_STORE: gl.constexpr,
    USE_WIDE_PACKED_STORE32: gl.constexpr,
    USE_WIDE_PACKED_STORE64: gl.constexpr,
    USE_PACKED_OUT_SCALE: gl.constexpr,
):
    if USE_PACKED_OUT_SCALE or USE_PACKED_FP8_STORE:
        _store_packed_out_fragments(
            p,
            float2.pack(out_frag, axis=1),
            out_recip,
            off_m,
            out_off_n,
            shape_m,
            slice_offset,
            USE_PACKED_FP8_STORE,
            USE_BLOCKED_PACKED_STORE,
            USE_WIDE_PACKED_STORE32,
            USE_WIDE_PACKED_STORE64,
            USE_PACKED_OUT_SCALE,
        )
        return

    out_frag = out_frag * out_recip
    if p.FLEXPOINT_SATURATE_INF:
        out_frag = tl_core.clamp(out_frag, -448.0, 448.0)
    _store_out(p, out_frag.to(p.out_desc.dtype), off_m, out_off_n, shape_m, slice_offset)


@gluon.jit
def _store_packed_out_fragment(
    p: ws_base.PartitionArgs,
    out_packed,
    out_recip,
    off_m,
    out_off_n,
    shape_m,
    slice_offset,
    USE_PACKED_FP8_STORE: gl.constexpr,
    USE_BLOCKED_PACKED_STORE: gl.constexpr,
    USE_WIDE_PACKED_STORE32: gl.constexpr,
    USE_WIDE_PACKED_STORE64: gl.constexpr,
    USE_PACKED_OUT_SCALE: gl.constexpr,
):
    if USE_PACKED_OUT_SCALE or USE_PACKED_FP8_STORE:
        _store_packed_out_fragments(
            p,
            out_packed,
            out_recip,
            off_m,
            out_off_n,
            shape_m,
            slice_offset,
            USE_PACKED_FP8_STORE,
            USE_BLOCKED_PACKED_STORE,
            USE_WIDE_PACKED_STORE32,
            USE_WIDE_PACKED_STORE64,
            USE_PACKED_OUT_SCALE,
        )
        return

    out_frag = float2.unpack(out_packed, axis=1) * out_recip
    if p.FLEXPOINT_SATURATE_INF:
        out_frag = tl_core.clamp(out_frag, -448.0, 448.0)
    _store_out(p, out_frag.to(p.out_desc.dtype), off_m, out_off_n, shape_m, slice_offset)


@gluon.jit
def _epilogue_from_acc_packed(
    p: ws_base.PartitionArgs,
    acc_packed,
    out_recip,
    off_m,
    out_off_n,
    shape_m,
    slice_offset,
    EPILOGUE_ROW_SUBTILE_FACTOR: gl.constexpr,
    EPILOGUE_N_FRAGMENT_FACTOR: gl.constexpr,
    EPILOGUE_SCHEDULE: gl.constexpr,
    USE_EXP2_SIGMOID: gl.constexpr,
    USE_PACKED_FINAL_FMA: gl.constexpr,
    USE_PACKED_FP8_STORE: gl.constexpr,
    USE_BLOCKED_PACKED_STORE: gl.constexpr,
    USE_WIDE_PACKED_STORE32: gl.constexpr,
    USE_WIDE_PACKED_STORE64: gl.constexpr,
    USE_PACKED_OUT_SCALE: gl.constexpr,
    USE_LINEAR_ACC_EPILOGUE: gl.constexpr,
    USE_LINEAR_FRAGMENT_PACK: gl.constexpr,
):
    gl.static_assert(not USE_EXP2_SIGMOID, "ws_optimized epilogue requires exact SwiGLU math")
    gl.static_assert(
        EPILOGUE_N_FRAGMENT_FACTOR == 1 or EPILOGUE_N_FRAGMENT_FACTOR == 2,
        "unsupported N fragment factor",
    )
    FRAG_ROWS: gl.constexpr = p.BLOCK_M // EPILOGUE_ROW_SUBTILE_FACTOR
    HALF_OUT_N: gl.constexpr = p.BLOCK_N // p.REDUCTION_N // 2
    if EPILOGUE_ROW_SUBTILE_FACTOR == 32:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        eighth000, eighth001 = _split_first_dim_in_half_packed(quarter00)
        eighth010, eighth011 = _split_first_dim_in_half_packed(quarter01)
        eighth100, eighth101 = _split_first_dim_in_half_packed(quarter10)
        eighth110, eighth111 = _split_first_dim_in_half_packed(quarter11)
        sixteenth0000, sixteenth0001 = _split_first_dim_in_half_packed(eighth000)
        sixteenth0010, sixteenth0011 = _split_first_dim_in_half_packed(eighth001)
        sixteenth0100, sixteenth0101 = _split_first_dim_in_half_packed(eighth010)
        sixteenth0110, sixteenth0111 = _split_first_dim_in_half_packed(eighth011)
        sixteenth1000, sixteenth1001 = _split_first_dim_in_half_packed(eighth100)
        sixteenth1010, sixteenth1011 = _split_first_dim_in_half_packed(eighth101)
        sixteenth1100, sixteenth1101 = _split_first_dim_in_half_packed(eighth110)
        sixteenth1110, sixteenth1111 = _split_first_dim_in_half_packed(eighth111)
        thirtysecond00000, thirtysecond00001 = _split_first_dim_in_half_packed(sixteenth0000)
        thirtysecond00010, thirtysecond00011 = _split_first_dim_in_half_packed(sixteenth0001)
        thirtysecond00100, thirtysecond00101 = _split_first_dim_in_half_packed(sixteenth0010)
        thirtysecond00110, thirtysecond00111 = _split_first_dim_in_half_packed(sixteenth0011)
        thirtysecond01000, thirtysecond01001 = _split_first_dim_in_half_packed(sixteenth0100)
        thirtysecond01010, thirtysecond01011 = _split_first_dim_in_half_packed(sixteenth0101)
        thirtysecond01100, thirtysecond01101 = _split_first_dim_in_half_packed(sixteenth0110)
        thirtysecond01110, thirtysecond01111 = _split_first_dim_in_half_packed(sixteenth0111)
        thirtysecond10000, thirtysecond10001 = _split_first_dim_in_half_packed(sixteenth1000)
        thirtysecond10010, thirtysecond10011 = _split_first_dim_in_half_packed(sixteenth1001)
        thirtysecond10100, thirtysecond10101 = _split_first_dim_in_half_packed(sixteenth1010)
        thirtysecond10110, thirtysecond10111 = _split_first_dim_in_half_packed(sixteenth1011)
        thirtysecond11000, thirtysecond11001 = _split_first_dim_in_half_packed(sixteenth1100)
        thirtysecond11010, thirtysecond11011 = _split_first_dim_in_half_packed(sixteenth1101)
        thirtysecond11100, thirtysecond11101 = _split_first_dim_in_half_packed(sixteenth1110)
        thirtysecond11110, thirtysecond11111 = _split_first_dim_in_half_packed(sixteenth1111)
        acc_packed_subtiles = (
            thirtysecond00000,
            thirtysecond00001,
            thirtysecond00010,
            thirtysecond00011,
            thirtysecond00100,
            thirtysecond00101,
            thirtysecond00110,
            thirtysecond00111,
            thirtysecond01000,
            thirtysecond01001,
            thirtysecond01010,
            thirtysecond01011,
            thirtysecond01100,
            thirtysecond01101,
            thirtysecond01110,
            thirtysecond01111,
            thirtysecond10000,
            thirtysecond10001,
            thirtysecond10010,
            thirtysecond10011,
            thirtysecond10100,
            thirtysecond10101,
            thirtysecond10110,
            thirtysecond10111,
            thirtysecond11000,
            thirtysecond11001,
            thirtysecond11010,
            thirtysecond11011,
            thirtysecond11100,
            thirtysecond11101,
            thirtysecond11110,
            thirtysecond11111,
        )
    elif EPILOGUE_ROW_SUBTILE_FACTOR == 16:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        eighth000, eighth001 = _split_first_dim_in_half_packed(quarter00)
        eighth010, eighth011 = _split_first_dim_in_half_packed(quarter01)
        eighth100, eighth101 = _split_first_dim_in_half_packed(quarter10)
        eighth110, eighth111 = _split_first_dim_in_half_packed(quarter11)
        sixteenth0000, sixteenth0001 = _split_first_dim_in_half_packed(eighth000)
        sixteenth0010, sixteenth0011 = _split_first_dim_in_half_packed(eighth001)
        sixteenth0100, sixteenth0101 = _split_first_dim_in_half_packed(eighth010)
        sixteenth0110, sixteenth0111 = _split_first_dim_in_half_packed(eighth011)
        sixteenth1000, sixteenth1001 = _split_first_dim_in_half_packed(eighth100)
        sixteenth1010, sixteenth1011 = _split_first_dim_in_half_packed(eighth101)
        sixteenth1100, sixteenth1101 = _split_first_dim_in_half_packed(eighth110)
        sixteenth1110, sixteenth1111 = _split_first_dim_in_half_packed(eighth111)
        acc_packed_subtiles = (
            sixteenth0000,
            sixteenth0001,
            sixteenth0010,
            sixteenth0011,
            sixteenth0100,
            sixteenth0101,
            sixteenth0110,
            sixteenth0111,
            sixteenth1000,
            sixteenth1001,
            sixteenth1010,
            sixteenth1011,
            sixteenth1100,
            sixteenth1101,
            sixteenth1110,
            sixteenth1111,
        )
    elif EPILOGUE_ROW_SUBTILE_FACTOR == 8:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        eighth000, eighth001 = _split_first_dim_in_half_packed(quarter00)
        eighth010, eighth011 = _split_first_dim_in_half_packed(quarter01)
        eighth100, eighth101 = _split_first_dim_in_half_packed(quarter10)
        eighth110, eighth111 = _split_first_dim_in_half_packed(quarter11)
        acc_packed_subtiles = (
            eighth000,
            eighth001,
            eighth010,
            eighth011,
            eighth100,
            eighth101,
            eighth110,
            eighth111,
        )
    elif EPILOGUE_ROW_SUBTILE_FACTOR == 4:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        acc_packed_subtiles = (quarter00, quarter01, quarter10, quarter11)
    elif EPILOGUE_ROW_SUBTILE_FACTOR == 2:
        acc_packed_subtiles = _split_first_dim_in_half_packed(acc_packed)
    else:
        acc_packed_subtiles = (acc_packed,)

    if EPILOGUE_ROW_SUBTILE_FACTOR == 1 or EPILOGUE_N_FRAGMENT_FACTOR != 1 or EPILOGUE_SCHEDULE != 1:
        for frag_idx in gl.static_range(EPILOGUE_ROW_SUBTILE_FACTOR):
            frag_off_m = off_m + frag_idx * FRAG_ROWS
            if EPILOGUE_N_FRAGMENT_FACTOR == 1:
                gelu_frag, linear_frag = _prepare_swiglu_fragment_from_packed(
                    acc_packed_subtiles[frag_idx],
                    p.SWIGLU_LIMIT,
                )
                out_packed_frag = _finish_swiglu_fragment_packed(
                    gelu_frag,
                    linear_frag,
                    p.SWIGLU_ALPHA,
                    USE_PACKED_FINAL_FMA,
                ) if not USE_LINEAR_ACC_EPILOGUE else _finish_swiglu_fragment_packed_linear(
                    gelu_frag,
                    linear_frag,
                    p.SWIGLU_ALPHA,
                    USE_PACKED_FINAL_FMA,
                )
                _store_packed_out_fragment(
                    p,
                    out_packed_frag,
                    out_recip,
                    frag_off_m,
                    out_off_n,
                    shape_m,
                    slice_offset,
                    USE_PACKED_FP8_STORE,
                    USE_BLOCKED_PACKED_STORE,
                    USE_WIDE_PACKED_STORE32,
                    USE_WIDE_PACKED_STORE64,
                    USE_PACKED_OUT_SCALE,
                )
            else:
                acc_n0, acc_n1 = _split_packed_last_dim_in_half_fragment(acc_packed_subtiles[frag_idx])
                gelu_frag0, linear_frag0 = _prepare_swiglu_fragment_from_packed(
                    acc_n0,
                    p.SWIGLU_LIMIT,
                )
                out_packed_frag0 = _finish_swiglu_fragment_packed(
                    gelu_frag0,
                    linear_frag0,
                    p.SWIGLU_ALPHA,
                    USE_PACKED_FINAL_FMA,
                ) if not USE_LINEAR_ACC_EPILOGUE else _finish_swiglu_fragment_packed_linear(
                    gelu_frag0,
                    linear_frag0,
                    p.SWIGLU_ALPHA,
                    USE_PACKED_FINAL_FMA,
                )
                _store_packed_out_fragment(
                    p,
                    out_packed_frag0,
                    out_recip,
                    frag_off_m,
                    out_off_n,
                    shape_m,
                    slice_offset,
                    USE_PACKED_FP8_STORE,
                    USE_BLOCKED_PACKED_STORE,
                    USE_WIDE_PACKED_STORE32,
                    USE_WIDE_PACKED_STORE64,
                    USE_PACKED_OUT_SCALE,
                )
                gelu_frag1, linear_frag1 = _prepare_swiglu_fragment_from_packed(
                    acc_n1,
                    p.SWIGLU_LIMIT,
                )
                out_packed_frag1 = _finish_swiglu_fragment_packed(
                    gelu_frag1,
                    linear_frag1,
                    p.SWIGLU_ALPHA,
                    USE_PACKED_FINAL_FMA,
                ) if not USE_LINEAR_FRAGMENT_PACK else _finish_swiglu_fragment_packed_linear(
                    gelu_frag1,
                    linear_frag1,
                    p.SWIGLU_ALPHA,
                    USE_PACKED_FINAL_FMA,
                )
                _store_packed_out_fragment(
                    p,
                    out_packed_frag1,
                    out_recip,
                    frag_off_m,
                    out_off_n + HALF_OUT_N,
                    shape_m,
                    slice_offset,
                    USE_PACKED_FP8_STORE,
                    USE_BLOCKED_PACKED_STORE,
                    USE_WIDE_PACKED_STORE32,
                    USE_WIDE_PACKED_STORE64,
                    USE_PACKED_OUT_SCALE,
                )
        return

    prepared_gelu, prepared_linear = _prepare_swiglu_fragment_from_packed(
        acc_packed_subtiles[0], p.SWIGLU_LIMIT
    )
    ready_out_packed = acc_packed_subtiles[0]
    for frag_idx in gl.static_range(1, EPILOGUE_ROW_SUBTILE_FACTOR):
        cur_gelu, cur_linear = _prepare_swiglu_fragment_from_packed(
            acc_packed_subtiles[frag_idx],
            p.SWIGLU_LIMIT,
        )
        next_ready_out_packed = _finish_swiglu_fragment_packed(
            prepared_gelu,
            prepared_linear,
            p.SWIGLU_ALPHA,
            USE_PACKED_FINAL_FMA,
        ) if not USE_LINEAR_FRAGMENT_PACK else _finish_swiglu_fragment_packed_linear(
            prepared_gelu,
            prepared_linear,
            p.SWIGLU_ALPHA,
            USE_PACKED_FINAL_FMA,
        )
        if frag_idx > 1:
            prev_off_m = off_m + (frag_idx - 2) * FRAG_ROWS
            _store_packed_out_fragment(
                p,
                ready_out_packed,
                out_recip,
                prev_off_m,
                out_off_n,
                shape_m,
                slice_offset,
                USE_PACKED_FP8_STORE,
                USE_BLOCKED_PACKED_STORE,
                USE_WIDE_PACKED_STORE32,
                USE_WIDE_PACKED_STORE64,
                USE_PACKED_OUT_SCALE,
            )
        ready_out_packed = next_ready_out_packed
        prepared_gelu = cur_gelu
        prepared_linear = cur_linear

    penultimate_off_m = off_m + (EPILOGUE_ROW_SUBTILE_FACTOR - 2) * FRAG_ROWS
    _store_packed_out_fragment(
        p,
        ready_out_packed,
        out_recip,
        penultimate_off_m,
        out_off_n,
        shape_m,
        slice_offset,
        USE_PACKED_FP8_STORE,
        USE_BLOCKED_PACKED_STORE,
        USE_WIDE_PACKED_STORE32,
        USE_WIDE_PACKED_STORE64,
        USE_PACKED_OUT_SCALE,
    )
    last_out_packed = _finish_swiglu_fragment_packed(
        prepared_gelu,
        prepared_linear,
        p.SWIGLU_ALPHA,
        USE_PACKED_FINAL_FMA,
    ) if not USE_LINEAR_FRAGMENT_PACK else _finish_swiglu_fragment_packed_linear(
        prepared_gelu,
        prepared_linear,
        p.SWIGLU_ALPHA,
        USE_PACKED_FINAL_FMA,
    )
    last_off_m = off_m + (EPILOGUE_ROW_SUBTILE_FACTOR - 1) * FRAG_ROWS
    _store_packed_out_fragment(
        p,
        last_out_packed,
        out_recip,
        last_off_m,
        out_off_n,
        shape_m,
        slice_offset,
        USE_PACKED_FP8_STORE,
        USE_BLOCKED_PACKED_STORE,
        USE_WIDE_PACKED_STORE32,
        USE_WIDE_PACKED_STORE64,
        USE_PACKED_OUT_SCALE,
    )


@gluon.jit
def _epilogue_enqueue_raw_acc_packed_for_helper(
    p: ws_base.PartitionArgs,
    acc_packed,
    off_m,
    EPILOGUE_ROW_SUBTILE_FACTOR: gl.constexpr,
    store_bufs,
    store_empty_bars,
    store_ready_bars,
    store_idx,
    store_phase,
    STORE_HELPER_DEPTH: gl.constexpr,
):
    gl.static_assert(EPILOGUE_ROW_SUBTILE_FACTOR > 1, "store helper requires row fragments")
    if EPILOGUE_ROW_SUBTILE_FACTOR == 32:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        eighth000, eighth001 = _split_first_dim_in_half_packed(quarter00)
        eighth010, eighth011 = _split_first_dim_in_half_packed(quarter01)
        eighth100, eighth101 = _split_first_dim_in_half_packed(quarter10)
        eighth110, eighth111 = _split_first_dim_in_half_packed(quarter11)
        sixteenth0000, sixteenth0001 = _split_first_dim_in_half_packed(eighth000)
        sixteenth0010, sixteenth0011 = _split_first_dim_in_half_packed(eighth001)
        sixteenth0100, sixteenth0101 = _split_first_dim_in_half_packed(eighth010)
        sixteenth0110, sixteenth0111 = _split_first_dim_in_half_packed(eighth011)
        sixteenth1000, sixteenth1001 = _split_first_dim_in_half_packed(eighth100)
        sixteenth1010, sixteenth1011 = _split_first_dim_in_half_packed(eighth101)
        sixteenth1100, sixteenth1101 = _split_first_dim_in_half_packed(eighth110)
        sixteenth1110, sixteenth1111 = _split_first_dim_in_half_packed(eighth111)
        thirtysecond00000, thirtysecond00001 = _split_first_dim_in_half_packed(sixteenth0000)
        thirtysecond00010, thirtysecond00011 = _split_first_dim_in_half_packed(sixteenth0001)
        thirtysecond00100, thirtysecond00101 = _split_first_dim_in_half_packed(sixteenth0010)
        thirtysecond00110, thirtysecond00111 = _split_first_dim_in_half_packed(sixteenth0011)
        thirtysecond01000, thirtysecond01001 = _split_first_dim_in_half_packed(sixteenth0100)
        thirtysecond01010, thirtysecond01011 = _split_first_dim_in_half_packed(sixteenth0101)
        thirtysecond01100, thirtysecond01101 = _split_first_dim_in_half_packed(sixteenth0110)
        thirtysecond01110, thirtysecond01111 = _split_first_dim_in_half_packed(sixteenth0111)
        thirtysecond10000, thirtysecond10001 = _split_first_dim_in_half_packed(sixteenth1000)
        thirtysecond10010, thirtysecond10011 = _split_first_dim_in_half_packed(sixteenth1001)
        thirtysecond10100, thirtysecond10101 = _split_first_dim_in_half_packed(sixteenth1010)
        thirtysecond10110, thirtysecond10111 = _split_first_dim_in_half_packed(sixteenth1011)
        thirtysecond11000, thirtysecond11001 = _split_first_dim_in_half_packed(sixteenth1100)
        thirtysecond11010, thirtysecond11011 = _split_first_dim_in_half_packed(sixteenth1101)
        thirtysecond11100, thirtysecond11101 = _split_first_dim_in_half_packed(sixteenth1110)
        thirtysecond11110, thirtysecond11111 = _split_first_dim_in_half_packed(sixteenth1111)
        acc_packed_subtiles = (
            thirtysecond00000,
            thirtysecond00001,
            thirtysecond00010,
            thirtysecond00011,
            thirtysecond00100,
            thirtysecond00101,
            thirtysecond00110,
            thirtysecond00111,
            thirtysecond01000,
            thirtysecond01001,
            thirtysecond01010,
            thirtysecond01011,
            thirtysecond01100,
            thirtysecond01101,
            thirtysecond01110,
            thirtysecond01111,
            thirtysecond10000,
            thirtysecond10001,
            thirtysecond10010,
            thirtysecond10011,
            thirtysecond10100,
            thirtysecond10101,
            thirtysecond10110,
            thirtysecond10111,
            thirtysecond11000,
            thirtysecond11001,
            thirtysecond11010,
            thirtysecond11011,
            thirtysecond11100,
            thirtysecond11101,
            thirtysecond11110,
            thirtysecond11111,
        )
    elif EPILOGUE_ROW_SUBTILE_FACTOR == 16:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        eighth000, eighth001 = _split_first_dim_in_half_packed(quarter00)
        eighth010, eighth011 = _split_first_dim_in_half_packed(quarter01)
        eighth100, eighth101 = _split_first_dim_in_half_packed(quarter10)
        eighth110, eighth111 = _split_first_dim_in_half_packed(quarter11)
        sixteenth0000, sixteenth0001 = _split_first_dim_in_half_packed(eighth000)
        sixteenth0010, sixteenth0011 = _split_first_dim_in_half_packed(eighth001)
        sixteenth0100, sixteenth0101 = _split_first_dim_in_half_packed(eighth010)
        sixteenth0110, sixteenth0111 = _split_first_dim_in_half_packed(eighth011)
        sixteenth1000, sixteenth1001 = _split_first_dim_in_half_packed(eighth100)
        sixteenth1010, sixteenth1011 = _split_first_dim_in_half_packed(eighth101)
        sixteenth1100, sixteenth1101 = _split_first_dim_in_half_packed(eighth110)
        sixteenth1110, sixteenth1111 = _split_first_dim_in_half_packed(eighth111)
        acc_packed_subtiles = (
            sixteenth0000,
            sixteenth0001,
            sixteenth0010,
            sixteenth0011,
            sixteenth0100,
            sixteenth0101,
            sixteenth0110,
            sixteenth0111,
            sixteenth1000,
            sixteenth1001,
            sixteenth1010,
            sixteenth1011,
            sixteenth1100,
            sixteenth1101,
            sixteenth1110,
            sixteenth1111,
        )
    elif EPILOGUE_ROW_SUBTILE_FACTOR == 8:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        eighth000, eighth001 = _split_first_dim_in_half_packed(quarter00)
        eighth010, eighth011 = _split_first_dim_in_half_packed(quarter01)
        eighth100, eighth101 = _split_first_dim_in_half_packed(quarter10)
        eighth110, eighth111 = _split_first_dim_in_half_packed(quarter11)
        acc_packed_subtiles = (
            eighth000,
            eighth001,
            eighth010,
            eighth011,
            eighth100,
            eighth101,
            eighth110,
            eighth111,
        )
    elif EPILOGUE_ROW_SUBTILE_FACTOR == 4:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        acc_packed_subtiles = (quarter00, quarter01, quarter10, quarter11)
    else:
        acc_packed_subtiles = _split_first_dim_in_half_packed(acc_packed)
    for frag_idx in gl.static_range(EPILOGUE_ROW_SUBTILE_FACTOR):
        store_idx, store_phase = _enqueue_store_helper_acc_fragment(
            acc_packed_subtiles[frag_idx],
            store_bufs,
            store_empty_bars,
            store_ready_bars,
            store_idx,
            store_phase,
            STORE_HELPER_DEPTH,
        )
    return store_idx, store_phase


@gluon.jit
def _epilogue_enqueue_from_acc_packed(
    p: ws_base.PartitionArgs,
    acc_packed,
    out_recip,
    off_m,
    EPILOGUE_ROW_SUBTILE_FACTOR: gl.constexpr,
    EPILOGUE_N_FRAGMENT_FACTOR: gl.constexpr,
    EPILOGUE_SCHEDULE: gl.constexpr,
    USE_PACKED_FINAL_FMA: gl.constexpr,
    store_bufs,
    store_empty_bars,
    store_ready_bars,
    store_idx,
    store_phase,
    USE_HELPER_PACKED_OUT_BUFFER: gl.constexpr,
    STORE_HELPER_DEPTH: gl.constexpr,
    USE_LINEAR_FRAGMENT_PACK: gl.constexpr,
):
    gl.static_assert(EPILOGUE_ROW_SUBTILE_FACTOR > 1, "store helper requires row fragments")
    gl.static_assert(EPILOGUE_N_FRAGMENT_FACTOR == 1, "store helper does not support N fragmenting")
    if EPILOGUE_ROW_SUBTILE_FACTOR == 32:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        eighth000, eighth001 = _split_first_dim_in_half_packed(quarter00)
        eighth010, eighth011 = _split_first_dim_in_half_packed(quarter01)
        eighth100, eighth101 = _split_first_dim_in_half_packed(quarter10)
        eighth110, eighth111 = _split_first_dim_in_half_packed(quarter11)
        sixteenth0000, sixteenth0001 = _split_first_dim_in_half_packed(eighth000)
        sixteenth0010, sixteenth0011 = _split_first_dim_in_half_packed(eighth001)
        sixteenth0100, sixteenth0101 = _split_first_dim_in_half_packed(eighth010)
        sixteenth0110, sixteenth0111 = _split_first_dim_in_half_packed(eighth011)
        sixteenth1000, sixteenth1001 = _split_first_dim_in_half_packed(eighth100)
        sixteenth1010, sixteenth1011 = _split_first_dim_in_half_packed(eighth101)
        sixteenth1100, sixteenth1101 = _split_first_dim_in_half_packed(eighth110)
        sixteenth1110, sixteenth1111 = _split_first_dim_in_half_packed(eighth111)
        thirtysecond00000, thirtysecond00001 = _split_first_dim_in_half_packed(sixteenth0000)
        thirtysecond00010, thirtysecond00011 = _split_first_dim_in_half_packed(sixteenth0001)
        thirtysecond00100, thirtysecond00101 = _split_first_dim_in_half_packed(sixteenth0010)
        thirtysecond00110, thirtysecond00111 = _split_first_dim_in_half_packed(sixteenth0011)
        thirtysecond01000, thirtysecond01001 = _split_first_dim_in_half_packed(sixteenth0100)
        thirtysecond01010, thirtysecond01011 = _split_first_dim_in_half_packed(sixteenth0101)
        thirtysecond01100, thirtysecond01101 = _split_first_dim_in_half_packed(sixteenth0110)
        thirtysecond01110, thirtysecond01111 = _split_first_dim_in_half_packed(sixteenth0111)
        thirtysecond10000, thirtysecond10001 = _split_first_dim_in_half_packed(sixteenth1000)
        thirtysecond10010, thirtysecond10011 = _split_first_dim_in_half_packed(sixteenth1001)
        thirtysecond10100, thirtysecond10101 = _split_first_dim_in_half_packed(sixteenth1010)
        thirtysecond10110, thirtysecond10111 = _split_first_dim_in_half_packed(sixteenth1011)
        thirtysecond11000, thirtysecond11001 = _split_first_dim_in_half_packed(sixteenth1100)
        thirtysecond11010, thirtysecond11011 = _split_first_dim_in_half_packed(sixteenth1101)
        thirtysecond11100, thirtysecond11101 = _split_first_dim_in_half_packed(sixteenth1110)
        thirtysecond11110, thirtysecond11111 = _split_first_dim_in_half_packed(sixteenth1111)
        acc_packed_subtiles = (
            thirtysecond00000,
            thirtysecond00001,
            thirtysecond00010,
            thirtysecond00011,
            thirtysecond00100,
            thirtysecond00101,
            thirtysecond00110,
            thirtysecond00111,
            thirtysecond01000,
            thirtysecond01001,
            thirtysecond01010,
            thirtysecond01011,
            thirtysecond01100,
            thirtysecond01101,
            thirtysecond01110,
            thirtysecond01111,
            thirtysecond10000,
            thirtysecond10001,
            thirtysecond10010,
            thirtysecond10011,
            thirtysecond10100,
            thirtysecond10101,
            thirtysecond10110,
            thirtysecond10111,
            thirtysecond11000,
            thirtysecond11001,
            thirtysecond11010,
            thirtysecond11011,
            thirtysecond11100,
            thirtysecond11101,
            thirtysecond11110,
            thirtysecond11111,
        )
    elif EPILOGUE_ROW_SUBTILE_FACTOR == 16:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        eighth000, eighth001 = _split_first_dim_in_half_packed(quarter00)
        eighth010, eighth011 = _split_first_dim_in_half_packed(quarter01)
        eighth100, eighth101 = _split_first_dim_in_half_packed(quarter10)
        eighth110, eighth111 = _split_first_dim_in_half_packed(quarter11)
        sixteenth0000, sixteenth0001 = _split_first_dim_in_half_packed(eighth000)
        sixteenth0010, sixteenth0011 = _split_first_dim_in_half_packed(eighth001)
        sixteenth0100, sixteenth0101 = _split_first_dim_in_half_packed(eighth010)
        sixteenth0110, sixteenth0111 = _split_first_dim_in_half_packed(eighth011)
        sixteenth1000, sixteenth1001 = _split_first_dim_in_half_packed(eighth100)
        sixteenth1010, sixteenth1011 = _split_first_dim_in_half_packed(eighth101)
        sixteenth1100, sixteenth1101 = _split_first_dim_in_half_packed(eighth110)
        sixteenth1110, sixteenth1111 = _split_first_dim_in_half_packed(eighth111)
        acc_packed_subtiles = (
            sixteenth0000,
            sixteenth0001,
            sixteenth0010,
            sixteenth0011,
            sixteenth0100,
            sixteenth0101,
            sixteenth0110,
            sixteenth0111,
            sixteenth1000,
            sixteenth1001,
            sixteenth1010,
            sixteenth1011,
            sixteenth1100,
            sixteenth1101,
            sixteenth1110,
            sixteenth1111,
        )
    elif EPILOGUE_ROW_SUBTILE_FACTOR == 8:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        eighth000, eighth001 = _split_first_dim_in_half_packed(quarter00)
        eighth010, eighth011 = _split_first_dim_in_half_packed(quarter01)
        eighth100, eighth101 = _split_first_dim_in_half_packed(quarter10)
        eighth110, eighth111 = _split_first_dim_in_half_packed(quarter11)
        acc_packed_subtiles = (
            eighth000,
            eighth001,
            eighth010,
            eighth011,
            eighth100,
            eighth101,
            eighth110,
            eighth111,
        )
    elif EPILOGUE_ROW_SUBTILE_FACTOR == 4:
        half0, half1 = _split_first_dim_in_half_packed(acc_packed)
        quarter00, quarter01 = _split_first_dim_in_half_packed(half0)
        quarter10, quarter11 = _split_first_dim_in_half_packed(half1)
        acc_packed_subtiles = (quarter00, quarter01, quarter10, quarter11)
    else:
        acc_packed_subtiles = _split_first_dim_in_half_packed(acc_packed)

    if EPILOGUE_SCHEDULE != 1 or EPILOGUE_ROW_SUBTILE_FACTOR == 1:
        for frag_idx in gl.static_range(EPILOGUE_ROW_SUBTILE_FACTOR):
            gelu_frag, linear_frag = _prepare_swiglu_fragment_from_packed(
                acc_packed_subtiles[frag_idx],
                p.SWIGLU_LIMIT,
            )
            out_packed_frag = _finish_swiglu_fragment_packed(
                gelu_frag,
                linear_frag,
                p.SWIGLU_ALPHA,
                USE_PACKED_FINAL_FMA,
            ) if not USE_LINEAR_FRAGMENT_PACK else _finish_swiglu_fragment_packed_linear(
                gelu_frag,
                linear_frag,
                p.SWIGLU_ALPHA,
                USE_PACKED_FINAL_FMA,
            )
            store_idx, store_phase = _enqueue_store_helper_out_fragment(
                out_packed_frag,
                out_recip,
                store_bufs,
                store_empty_bars,
                store_ready_bars,
                store_idx,
                store_phase,
                USE_HELPER_PACKED_OUT_BUFFER,
                STORE_HELPER_DEPTH,
            )
        return store_idx, store_phase

    prepared_gelu, prepared_linear = _prepare_swiglu_fragment_from_packed(
        acc_packed_subtiles[0],
        p.SWIGLU_LIMIT,
    )
    ready_out_packed = acc_packed_subtiles[0]
    for frag_idx in gl.static_range(1, EPILOGUE_ROW_SUBTILE_FACTOR):
        cur_gelu, cur_linear = _prepare_swiglu_fragment_from_packed(
            acc_packed_subtiles[frag_idx],
            p.SWIGLU_LIMIT,
        )
        next_ready_out_packed = _finish_swiglu_fragment_packed(
            prepared_gelu,
            prepared_linear,
            p.SWIGLU_ALPHA,
            USE_PACKED_FINAL_FMA,
        ) if not USE_LINEAR_FRAGMENT_PACK else _finish_swiglu_fragment_packed_linear(
            prepared_gelu,
            prepared_linear,
            p.SWIGLU_ALPHA,
            USE_PACKED_FINAL_FMA,
        )
        if frag_idx > 1:
            store_idx, store_phase = _enqueue_store_helper_out_fragment(
                ready_out_packed,
                out_recip,
                store_bufs,
                store_empty_bars,
                store_ready_bars,
                store_idx,
                store_phase,
                USE_HELPER_PACKED_OUT_BUFFER,
                STORE_HELPER_DEPTH,
            )
        ready_out_packed = next_ready_out_packed
        prepared_gelu = cur_gelu
        prepared_linear = cur_linear

    store_idx, store_phase = _enqueue_store_helper_out_fragment(
        ready_out_packed,
        out_recip,
        store_bufs,
        store_empty_bars,
        store_ready_bars,
        store_idx,
        store_phase,
        USE_HELPER_PACKED_OUT_BUFFER,
        STORE_HELPER_DEPTH,
    )
    last_out_packed = _finish_swiglu_fragment_packed(
        prepared_gelu,
        prepared_linear,
        p.SWIGLU_ALPHA,
        USE_PACKED_FINAL_FMA,
    ) if not USE_LINEAR_FRAGMENT_PACK else _finish_swiglu_fragment_packed_linear(
        prepared_gelu,
        prepared_linear,
        p.SWIGLU_ALPHA,
        USE_PACKED_FINAL_FMA,
    )
    store_idx, store_phase = _enqueue_store_helper_out_fragment(
        last_out_packed,
        out_recip,
        store_bufs,
        store_empty_bars,
        store_ready_bars,
        store_idx,
        store_phase,
        USE_HELPER_PACKED_OUT_BUFFER,
        STORE_HELPER_DEPTH,
    )
    return store_idx, store_phase


@gluon.jit
def epilogue_store_partition_optimized(
    p: ws_base.PartitionArgs,
    store_bufs,
    store_empty_bars,
    store_ready_bars,
    EPILOGUE_SUBTILE_N: gl.constexpr,
    EPILOGUE_ROW_SUBTILE_FACTOR: gl.constexpr,
    USE_BLOCKED_PACKED_STORE: gl.constexpr,
    USE_WIDE_PACKED_STORE32: gl.constexpr,
    USE_WIDE_PACKED_STORE64: gl.constexpr,
    USE_HELPER_PACKED_OUT_BUFFER: gl.constexpr,
    USE_HELPER_SWIGLU: gl.constexpr,
    USE_PACKED_FINAL_FMA: gl.constexpr,
    STORE_HELPER_DEPTH: gl.constexpr,
):
    gl.static_assert(EPILOGUE_ROW_SUBTILE_FACTOR > 1, "store helper requires row fragments")
    FRAG_ROWS: gl.constexpr = p.BLOCK_M // EPILOGUE_ROW_SUBTILE_FACTOR
    STORE_LAYOUT: gl.constexpr = _store_helper_fragment_layout(FRAG_ROWS, gl.num_warps())
    SUBTILE_COUNT: gl.constexpr = p.BLOCK_N // EPILOGUE_SUBTILE_N
    gl.static_assert(STORE_HELPER_DEPTH >= 1, "store helper depth must be at least 1")

    store_idx = 0
    store_phase = 0
    out_recip = 1.0 / gl.load(p.out_scale_ptr) if USE_HELPER_PACKED_OUT_BUFFER or USE_HELPER_SWIGLU else 0.0
    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        shape_m = gl.load(p.x_slice_sizes + slice_idx)

        for subtile_idx in gl.static_range(SUBTILE_COUNT):
            subtile_off_n = subtile_idx * EPILOGUE_SUBTILE_N
            out_off_n = (pid_n * p.BLOCK_N + subtile_off_n) // p.REDUCTION_N
            for frag_idx in gl.static_range(EPILOGUE_ROW_SUBTILE_FACTOR):
                frag_off_m = off_m + frag_idx * FRAG_ROWS
                ready_bar = store_ready_bars.index(store_idx)
                empty_bar = store_empty_bars.index(store_idx)
                mbarrier.wait(ready_bar, store_phase)
                store_buf = store_bufs.index(store_idx)
                if USE_HELPER_SWIGLU:
                    if FRAG_ROWS == 4 and gl.num_warps() == 2:
                        acc_packed = float2.Float2Tensor(store_buf.load(_store_helper_acc_row32_layout()))
                    else:
                        acc_packed = float2.Float2Tensor(store_buf.load(STORE_LAYOUT))
                    gelu_frag, linear_frag = _prepare_swiglu_fragment_from_packed(acc_packed, p.SWIGLU_LIMIT)
                    out_packed = _finish_swiglu_fragment_packed(
                        gelu_frag,
                        linear_frag,
                        p.SWIGLU_ALPHA,
                        USE_PACKED_FINAL_FMA,
                    )
                    packed_fp8 = _pack_fp8_out_fragment(out_packed, out_recip)
                    _store_packed_out(
                        p,
                        packed_fp8,
                        frag_off_m,
                        out_off_n,
                        shape_m,
                        slice_offset,
                        USE_BLOCKED_PACKED_STORE,
                        USE_WIDE_PACKED_STORE32,
                        USE_WIDE_PACKED_STORE64,
                    )
                elif USE_HELPER_PACKED_OUT_BUFFER:
                    packed_fp8 = _pack_fp8_out_fragment(float2.Float2Tensor(store_buf.load(STORE_LAYOUT)), out_recip)
                    _store_packed_out(
                        p,
                        packed_fp8,
                        frag_off_m,
                        out_off_n,
                        shape_m,
                        slice_offset,
                        USE_BLOCKED_PACKED_STORE,
                        USE_WIDE_PACKED_STORE32,
                        USE_WIDE_PACKED_STORE64,
                    )
                else:
                    packed_fp8 = store_buf.load(STORE_LAYOUT)
                    _store_packed_out(
                        p,
                        packed_fp8,
                        frag_off_m,
                        out_off_n,
                        shape_m,
                        slice_offset,
                        USE_BLOCKED_PACKED_STORE,
                        USE_WIDE_PACKED_STORE32,
                        USE_WIDE_PACKED_STORE64,
                    )
                mbarrier.arrive(empty_bar)
                store_idx, store_phase = ws_base.advance(store_idx, store_phase, STORE_HELPER_DEPTH)


@gluon.jit
def epilogue_partition_optimized(
    p: ws_base.PartitionArgs,
    store_bufs,
    store_empty_bars,
    store_ready_bars,
    STORE_HELPER_WARPS: gl.constexpr,
    ACC_INSTR_VARIANT: gl.constexpr,
    EPILOGUE_N_ELEMS: gl.constexpr,
    EPILOGUE_SUBTILE_N: gl.constexpr,
    EPILOGUE_ROW_SUBTILE_FACTOR: gl.constexpr,
    EPILOGUE_N_FRAGMENT_FACTOR: gl.constexpr,
    EPILOGUE_SCHEDULE: gl.constexpr,
    USE_EXP2_SIGMOID: gl.constexpr,
    USE_PACKED_FINAL_FMA: gl.constexpr,
    USE_PACKED_FP8_STORE: gl.constexpr,
    USE_BLOCKED_PACKED_STORE: gl.constexpr,
    USE_WIDE_PACKED_STORE32: gl.constexpr,
    USE_WIDE_PACKED_STORE64: gl.constexpr,
    USE_PACKED_OUT_SCALE: gl.constexpr,
    USE_LINEAR_ACC_EPILOGUE: gl.constexpr,
    USE_HELPER_PACKED_OUT_BUFFER: gl.constexpr,
    USE_HELPER_SWIGLU: gl.constexpr,
    USE_EPILOGUE_STORE_HELPER: gl.constexpr,
    STORE_HELPER_DEPTH: gl.constexpr,
):
    idx = 0
    phase = 0
    store_idx = 0
    store_phase = 1

    x_scale = 1.0 if p.x_scale_ptr is None else gl.load(p.x_scale_ptr)
    w_scale = 1.0 if p.w_scale_ptr is None else gl.load(p.w_scale_ptr)
    acc_scale = x_scale * w_scale
    out_recip = 1.0 / gl.load(p.out_scale_ptr)

    num_warps: gl.constexpr = gl.num_warps()
    SUBTILE_COUNT: gl.constexpr = p.BLOCK_N // EPILOGUE_SUBTILE_N
    warps_n: gl.constexpr = 2 if num_warps >= 8 and p.BLOCK_N >= 256 else 1
    split_layout: gl.constexpr = gl.BlockedLayout(
        [1, EPILOGUE_N_ELEMS],
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

        acc_empty_bar = p.acc_empty_bars.index(idx)
        acc_ready_bar = p.acc_ready_bars.index(idx)
        acc_buf = p.acc_bufs.index(idx)
        mbarrier.wait(acc_ready_bar, phase)
        mbarrier.arrive(acc_empty_bar)
        idx, phase = ws_base.advance(idx, phase, p.acc_num_bufs)
        offs_bias_n = off_n + gl.arange(0, p.BLOCK_N, layout=bias_layout)
        bias = gl.convert_layout(
            gl.expand_dims(gl.load(p.bias_ptr + slice_idx * p.bias_stride + offs_bias_n), axis=0),
            split_layout,
        )
        acc_regs = acc_buf.load(acc_buf.get_reg_layout(instr_variant=ACC_INSTR_VARIANT)).permute((1, 0))
        if USE_LINEAR_ACC_EPILOGUE:
            acc_packed = _pack_last_dim_adjacent_linear(acc_regs)
        else:
            acc = gl.convert_layout(acc_regs, split_layout)
            acc_packed = float2.pack(acc, axis=1)
        bias_packed = _pack_last_dim_in_half(bias)
        bias_packed = float2.Float2Tensor(gl.convert_layout(bias_packed.value, acc_packed.value.type.layout))
        acc_packed = float2.fma(acc_packed, float2.full_like(acc_packed, acc_scale), bias_packed)
        if SUBTILE_COUNT == 1:
            acc_packed_subtiles = (acc_packed,)
        elif SUBTILE_COUNT == 2:
            acc_packed_subtiles = (
                _split_packed_last_dim_in_half(acc_packed)
                if USE_LINEAR_ACC_EPILOGUE
                else _split_packed_last_dim_in_half_adjacent(acc_packed)
            )
        else:
            if USE_LINEAR_ACC_EPILOGUE:
                half0, half1 = _split_packed_last_dim_in_half(acc_packed)
                half00, half01 = _split_packed_last_dim_in_half(half0)
                half10, half11 = _split_packed_last_dim_in_half(half1)
            else:
                half0, half1 = _split_packed_last_dim_in_half_adjacent(acc_packed)
                half00, half01 = _split_packed_last_dim_in_half_adjacent(half0)
                half10, half11 = _split_packed_last_dim_in_half_adjacent(half1)
            acc_packed_subtiles = (half00, half01, half10, half11)
        use_linear_fragment_pack: gl.constexpr = USE_LINEAR_ACC_EPILOGUE or SUBTILE_COUNT != 1

        for subtile_idx in gl.static_range(SUBTILE_COUNT):
            subtile_off_n = subtile_idx * EPILOGUE_SUBTILE_N
            out_off_n = (pid_n * p.BLOCK_N + subtile_off_n) // p.REDUCTION_N
            if USE_EPILOGUE_STORE_HELPER:
                if USE_HELPER_SWIGLU:
                    store_idx, store_phase = _epilogue_enqueue_raw_acc_packed_for_helper(
                        p,
                        acc_packed_subtiles[subtile_idx],
                        off_m,
                        EPILOGUE_ROW_SUBTILE_FACTOR,
                        store_bufs,
                        store_empty_bars,
                        store_ready_bars,
                        store_idx,
                        store_phase,
                        STORE_HELPER_DEPTH,
                    )
                else:
                    store_idx, store_phase = _epilogue_enqueue_from_acc_packed(
                        p,
                        acc_packed_subtiles[subtile_idx],
                        out_recip,
                        off_m,
                        EPILOGUE_ROW_SUBTILE_FACTOR,
                        EPILOGUE_N_FRAGMENT_FACTOR,
                        EPILOGUE_SCHEDULE,
                        USE_PACKED_FINAL_FMA,
                        store_bufs,
                        store_empty_bars,
                        store_ready_bars,
                        store_idx,
                        store_phase,
                        USE_HELPER_PACKED_OUT_BUFFER,
                        STORE_HELPER_DEPTH,
                        use_linear_fragment_pack,
                    )
            else:
                _epilogue_from_acc_packed(
                    p,
                    acc_packed_subtiles[subtile_idx],
                    out_recip,
                    off_m,
                    out_off_n,
                    shape_m,
                    slice_offset,
                    EPILOGUE_ROW_SUBTILE_FACTOR,
                    EPILOGUE_N_FRAGMENT_FACTOR,
                    EPILOGUE_SCHEDULE,
                    USE_EXP2_SIGMOID,
                    USE_PACKED_FINAL_FMA,
                    USE_PACKED_FP8_STORE,
                    USE_BLOCKED_PACKED_STORE,
                    USE_WIDE_PACKED_STORE32,
                    USE_WIDE_PACKED_STORE64,
                    USE_PACKED_OUT_SCALE,
                    USE_LINEAR_ACC_EPILOGUE,
                    use_linear_fragment_pack,
                )

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
    ACC_INSTR_VARIANT: gl.constexpr,
    EPILOGUE_N_ELEMS: gl.constexpr,
    EPILOGUE_SUBTILE_N: gl.constexpr,
    EPILOGUE_ROW_SUBTILE_FACTOR: gl.constexpr,
    EPILOGUE_N_FRAGMENT_FACTOR: gl.constexpr,
    EPILOGUE_SCHEDULE: gl.constexpr,
    USE_EXP2_SIGMOID: gl.constexpr,
    USE_PACKED_FINAL_FMA: gl.constexpr,
    USE_PACKED_FP8_STORE: gl.constexpr,
    USE_BLOCKED_PACKED_STORE: gl.constexpr,
    USE_WIDE_PACKED_STORE32: gl.constexpr,
    USE_WIDE_PACKED_STORE64: gl.constexpr,
    USE_PACKED_OUT_SCALE: gl.constexpr,
    USE_LINEAR_ACC_EPILOGUE: gl.constexpr,
    USE_HELPER_PACKED_OUT_BUFFER: gl.constexpr,
    USE_HELPER_SWIGLU: gl.constexpr,
    USE_EPILOGUE_STORE_HELPER: gl.constexpr,
    EPILOGUE_STORE_HELPER_DEPTH: gl.constexpr,
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

    x_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_M, SCALE_K], scale_layout)
    w_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_N, SCALE_K], scale_layout)

    acc_num_bufs: gl.constexpr = 1
    acc_tmem = blackwell.allocate_tensor_memory(gl.float32, [acc_num_bufs, BLOCK_N, BLOCK_M], acc_layout)
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [acc_num_bufs, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [acc_num_bufs, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(acc_num_bufs):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    if USE_EPILOGUE_STORE_HELPER:
        gl.static_assert(not USE_EXP2_SIGMOID, "store helper requires exact SwiGLU math")
        gl.static_assert(USE_PACKED_FP8_STORE, "store helper requires packed FP8 stores")
        gl.static_assert(EPILOGUE_ROW_SUBTILE_FACTOR > 1, "store helper requires row fragments")
        gl.static_assert(EPILOGUE_STORE_HELPER_DEPTH >= 1, "store helper depth must be at least 1")
        FRAG_ROWS: gl.constexpr = BLOCK_M // EPILOGUE_ROW_SUBTILE_FACTOR
        # Size the helper ring to the active N subtile so narrower epilogues can
        # trade helper-buffer shared memory for deeper mainloop staging.
        STORE_BUF_N: gl.constexpr = (
            EPILOGUE_SUBTILE_N // 2 if USE_HELPER_SWIGLU else EPILOGUE_SUBTILE_N // REDUCTION_N // 2
        )
        STORE_BUF_DTYPE: gl.constexpr = gl.int64 if USE_HELPER_PACKED_OUT_BUFFER or USE_HELPER_SWIGLU else gl.int16
        store_bufs = gl.allocate_shared_memory(
            STORE_BUF_DTYPE,
            [EPILOGUE_STORE_HELPER_DEPTH, FRAG_ROWS, STORE_BUF_N],
            gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        )
        store_empty_bars = gl.allocate_shared_memory(
            gl.int64,
            [EPILOGUE_STORE_HELPER_DEPTH, 1],
            mbarrier.MBarrierLayout(),
        )
        store_ready_bars = gl.allocate_shared_memory(
            gl.int64,
            [EPILOGUE_STORE_HELPER_DEPTH, 1],
            mbarrier.MBarrierLayout(),
        )
        for i in gl.static_range(EPILOGUE_STORE_HELPER_DEPTH):
            mbarrier.init(store_empty_bars.index(i), count=1)
            mbarrier.init(store_ready_bars.index(i), count=1)
    else:
        STORE_BUF_DTYPE: gl.constexpr = gl.int64 if USE_HELPER_PACKED_OUT_BUFFER or USE_HELPER_SWIGLU else gl.int16
        store_bufs = gl.allocate_shared_memory(STORE_BUF_DTYPE, [1, 1, 1], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        store_empty_bars = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
        store_ready_bars = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
        mbarrier.init(store_empty_bars.index(0), count=1)
        mbarrier.init(store_ready_bars.index(0), count=1)

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
    if USE_EPILOGUE_STORE_HELPER:
        gl.warp_specialize(
            [
                (
                    epilogue_partition_optimized,
                    (
                        p,
                        store_bufs,
                        store_empty_bars,
                        store_ready_bars,
                        STORE_HELPER_WARPS,
                        ACC_INSTR_VARIANT,
                        EPILOGUE_N_ELEMS,
                        EPILOGUE_SUBTILE_N,
                        EPILOGUE_ROW_SUBTILE_FACTOR,
                        EPILOGUE_N_FRAGMENT_FACTOR,
                        EPILOGUE_SCHEDULE,
                        USE_EXP2_SIGMOID,
                        USE_PACKED_FINAL_FMA,
                        USE_PACKED_FP8_STORE,
                        USE_BLOCKED_PACKED_STORE,
                        USE_WIDE_PACKED_STORE32,
                        USE_WIDE_PACKED_STORE64,
                        USE_PACKED_OUT_SCALE,
                        USE_LINEAR_ACC_EPILOGUE,
                        USE_HELPER_PACKED_OUT_BUFFER,
                        USE_HELPER_SWIGLU,
                        USE_EPILOGUE_STORE_HELPER,
                        EPILOGUE_STORE_HELPER_DEPTH,
                    ),
                ),
                (
                    epilogue_store_partition_optimized,
                    (
                        p,
                        store_bufs,
                        store_empty_bars,
                        store_ready_bars,
                        EPILOGUE_SUBTILE_N,
                        EPILOGUE_ROW_SUBTILE_FACTOR,
                        USE_BLOCKED_PACKED_STORE,
                        USE_WIDE_PACKED_STORE32,
                        USE_WIDE_PACKED_STORE64,
                        USE_HELPER_PACKED_OUT_BUFFER,
                        USE_HELPER_SWIGLU,
                        USE_PACKED_FINAL_FMA,
                        EPILOGUE_STORE_HELPER_DEPTH,
                    ),
                ),
                (ws_base.load_activations, (p,)),
                (ws_base.load_weights, (p,)),
                (ws_base.mma_partition, (p,)),
            ],
            [STORE_HELPER_WARPS, LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
            [STORE_HELPER_REGS, LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
        )
    else:
        gl.warp_specialize(
            [
                (
                    epilogue_partition_optimized,
                    (
                        p,
                        store_bufs,
                        store_empty_bars,
                        store_ready_bars,
                        STORE_HELPER_WARPS,
                        ACC_INSTR_VARIANT,
                        EPILOGUE_N_ELEMS,
                        EPILOGUE_SUBTILE_N,
                        EPILOGUE_ROW_SUBTILE_FACTOR,
                        EPILOGUE_N_FRAGMENT_FACTOR,
                        EPILOGUE_SCHEDULE,
                        USE_EXP2_SIGMOID,
                        USE_PACKED_FINAL_FMA,
                        USE_PACKED_FP8_STORE,
                        USE_BLOCKED_PACKED_STORE,
                        USE_WIDE_PACKED_STORE32,
                        USE_WIDE_PACKED_STORE64,
                        USE_PACKED_OUT_SCALE,
                        USE_LINEAR_ACC_EPILOGUE,
                        USE_HELPER_PACKED_OUT_BUFFER,
                        USE_HELPER_SWIGLU,
                        USE_EPILOGUE_STORE_HELPER,
                        EPILOGUE_STORE_HELPER_DEPTH,
                    ),
                ),
                (ws_base.load_activations, (p,)),
                (ws_base.load_weights, (p,)),
                (ws_base.mma_partition, (p,)),
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
    if USE_EPILOGUE_STORE_HELPER:
        for i in gl.static_range(EPILOGUE_STORE_HELPER_DEPTH):
            mbarrier.invalidate(store_empty_bars.index(i))
            mbarrier.invalidate(store_ready_bars.index(i))
    else:
        mbarrier.invalidate(store_empty_bars.index(0))
        mbarrier.invalidate(store_ready_bars.index(0))


def _row_count(m_rows: int, expected_slice_size: int | None, n_slices: int) -> int:
    exp = -1 if expected_slice_size is None else expected_slice_size
    if exp < 0:
        return m_rows
    if exp == 0:
        return max(1, m_rows // 8)
    return exp * n_slices


def _select_kernel_config(m_rows: int, expected_slice_size: int | None, n_slices: int) -> tuple[KernelConfig, int]:
    defaults = KernelConfig()
    cfg = KernelConfig(
        num_warps=int(os.environ.get("TRITON_WS_NUM_WARPS", defaults.num_warps)),
        x_num_bufs=int(os.environ.get("TRITON_WS_X_NUM_BUFS", defaults.x_num_bufs)),
        w_num_bufs=int(os.environ.get("TRITON_WS_W_NUM_BUFS", defaults.w_num_bufs)),
        load_activation_warps=int(
            os.environ.get("TRITON_WS_LOAD_ACTIVATION_WARPS", defaults.load_activation_warps)
        ),
        load_weight_warps=int(
            os.environ.get("TRITON_WS_LOAD_WEIGHT_WARPS", defaults.load_weight_warps)
        ),
        mma_warps=int(os.environ.get("TRITON_WS_MMA_WARPS", defaults.mma_warps)),
        store_helper_warps=int(
            os.environ.get("TRITON_WS_STORE_HELPER_WARPS", defaults.store_helper_warps)
        ),
        epilogue_n_elems=int(os.environ.get("TRITON_WS_EPILOGUE_N_ELEMS", defaults.epilogue_n_elems)),
        epilogue_subtile_n=int(
            os.environ.get("TRITON_WS_EPILOGUE_SUBTILE_N", defaults.epilogue_subtile_n)
        ),
        epilogue_row_subtile_factor=int(
            os.environ.get(
                "TRITON_WS_EPILOGUE_ROW_SUBTILE_FACTOR",
                defaults.epilogue_row_subtile_factor,
            )
        ),
        epilogue_n_fragment_factor=int(
            os.environ.get(
                "TRITON_WS_EPILOGUE_N_FRAGMENT_FACTOR",
                defaults.epilogue_n_fragment_factor,
            )
        ),
        epilogue_schedule=int(
            os.environ.get(
                "TRITON_WS_EPILOGUE_SCHEDULE",
                defaults.epilogue_schedule,
            )
        ),
        epilogue_store_helper=os.environ.get(
            "TRITON_WS_USE_EPILOGUE_STORE_HELPER",
            "1" if defaults.epilogue_store_helper else "0",
        )
        == "1",
        epilogue_store_helper_depth=int(
            os.environ.get(
                "TRITON_WS_EPILOGUE_STORE_HELPER_DEPTH",
                defaults.epilogue_store_helper_depth,
            )
        ),
        load_activation_regs=int(
            os.environ.get("TRITON_WS_LOAD_ACTIVATION_REGS", defaults.load_activation_regs)
        ),
        load_weight_regs=int(os.environ.get("TRITON_WS_LOAD_WEIGHT_REGS", defaults.load_weight_regs)),
        mma_regs=int(os.environ.get("TRITON_WS_MMA_REGS", defaults.mma_regs)),
        store_helper_regs=int(
            os.environ.get("TRITON_WS_STORE_HELPER_REGS", defaults.store_helper_regs)
        ),
        acc_instr_variant=os.environ.get("TRITON_WS_ACC_INSTR_VARIANT", defaults.acc_instr_variant),
        use_packed_final_fma=os.environ.get(
            "TRITON_WS_USE_PACKED_FINAL_FMA", "1" if defaults.use_packed_final_fma else "0"
        )
        == "1",
        use_packed_fp8_store=os.environ.get(
            "TRITON_WS_USE_PACKED_FP8_STORE", "1" if defaults.use_packed_fp8_store else "0"
        )
        == "1",
        use_blocked_packed_store=os.environ.get(
            "TRITON_WS_USE_BLOCKED_PACKED_STORE", "1" if defaults.use_blocked_packed_store else "0"
        )
        == "1",
        use_wide_packed_store32=os.environ.get(
            "TRITON_WS_USE_WIDE_PACKED_STORE32", "1" if defaults.use_wide_packed_store32 else "0"
        )
        == "1",
        use_wide_packed_store64=os.environ.get(
            "TRITON_WS_USE_WIDE_PACKED_STORE64", "1" if defaults.use_wide_packed_store64 else "0"
        )
        == "1",
        use_packed_out_scale=os.environ.get(
            "TRITON_WS_USE_PACKED_OUT_SCALE", "1" if defaults.use_packed_out_scale else "0"
        )
        == "1",
        use_linear_acc_epilogue=os.environ.get(
            "TRITON_WS_USE_LINEAR_ACC_EPILOGUE", "1" if defaults.use_linear_acc_epilogue else "0"
        )
        == "1",
        use_helper_packed_out_buffer=os.environ.get(
            "TRITON_WS_USE_HELPER_PACKED_OUT_BUFFER",
            "1" if defaults.use_helper_packed_out_buffer else "0",
        )
        == "1",
        use_helper_swiglu=os.environ.get(
            "TRITON_WS_USE_HELPER_SWIGLU",
            "1" if defaults.use_helper_swiglu else "0",
        )
        == "1",
    )
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
    # Keep the current full-width-N epilogue invariant. The smaller subtile
    # experiments in this file are still exploratory and are not promoted as a
    # supported config surface yet.
    assert config.block_n % config.epilogue_subtile_n == 0
    assert config.epilogue_subtile_n % reduction_n == 0
    assert config.block_n // config.epilogue_subtile_n == 1
    # In Gluon warp-specialized kernels, num_warps is the default-partition
    # size. Worker partitions add their own warps on top, so we only need the
    # parent region to satisfy the backend's power-of-two and WS multiple-of-4
    # constraints.
    assert config.num_warps >= 4 and (config.num_warps & (config.num_warps - 1)) == 0
    assert config.num_warps % 4 == 0
    assert config.epilogue_row_subtile_factor in (1, 2, 4, 8, 16, 32)
    assert config.block_m % config.epilogue_row_subtile_factor == 0
    assert config.epilogue_n_fragment_factor in (1, 2)
    assert config.epilogue_schedule in (EPILOGUE_SCHEDULE_DIRECT, EPILOGUE_SCHEDULE_WAVEFRONT)
    assert config.load_activation_warps >= 1
    assert config.load_weight_warps >= 1
    assert config.mma_warps >= 1
    assert config.load_activation_regs >= 1
    assert config.load_weight_regs >= 1
    assert config.mma_regs >= 1
    assert not config.use_exp2_sigmoid
    assert config.epilogue_store_helper_depth in (1, 2, 3, 4)
    assert not (config.use_wide_packed_store32 and config.use_wide_packed_store64)
    assert config.acc_instr_variant in ("32x32b", "32x32b_splitn", "16x64b", "16x128b", "16x256b")
    if config.use_linear_acc_epilogue:
        assert not config.epilogue_store_helper
        assert config.epilogue_row_subtile_factor == 1
        assert config.epilogue_n_fragment_factor == 1
    assert not (config.use_helper_packed_out_buffer and config.use_helper_swiglu)
    if config.epilogue_store_helper:
        assert config.store_helper_warps >= 1
        assert config.store_helper_regs >= 1
        assert config.use_packed_fp8_store
        assert config.epilogue_row_subtile_factor in (2, 4, 8, 16, 32)
        assert config.epilogue_n_fragment_factor == 1
        if config.use_helper_packed_out_buffer:
            assert config.use_packed_out_scale
        if config.use_helper_swiglu:
            assert config.use_packed_out_scale
    else:
        assert not config.use_helper_packed_out_buffer
        assert not config.use_helper_swiglu
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
    out_desc = ws_base.make_operand_descriptor(c, [config.block_m, config.block_n // reduction_n])

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
        X_NUM_BUFS=config.x_num_bufs,
        W_NUM_BUFS=config.w_num_bufs,
        LOAD_ACTIVATION_WARPS=config.load_activation_warps,
        LOAD_WEIGHT_WARPS=config.load_weight_warps,
        MMA_WARPS=config.mma_warps,
        STORE_HELPER_WARPS=config.store_helper_warps,
        LOAD_ACTIVATION_REGS=config.load_activation_regs,
        LOAD_WEIGHT_REGS=config.load_weight_regs,
        MMA_REGS=config.mma_regs,
        STORE_HELPER_REGS=config.store_helper_regs,
        ACC_INSTR_VARIANT=config.acc_instr_variant,
        EPILOGUE_N_ELEMS=config.epilogue_n_elems,
        EPILOGUE_SUBTILE_N=config.epilogue_subtile_n,
        EPILOGUE_ROW_SUBTILE_FACTOR=config.epilogue_row_subtile_factor,
        EPILOGUE_N_FRAGMENT_FACTOR=config.epilogue_n_fragment_factor,
        EPILOGUE_SCHEDULE=config.epilogue_schedule,
        USE_EXP2_SIGMOID=config.use_exp2_sigmoid,
        USE_PACKED_FINAL_FMA=config.use_packed_final_fma,
        USE_PACKED_FP8_STORE=config.use_packed_fp8_store,
        USE_BLOCKED_PACKED_STORE=config.use_blocked_packed_store,
        USE_WIDE_PACKED_STORE32=config.use_wide_packed_store32,
        USE_WIDE_PACKED_STORE64=config.use_wide_packed_store64,
        USE_PACKED_OUT_SCALE=config.use_packed_out_scale,
        USE_LINEAR_ACC_EPILOGUE=config.use_linear_acc_epilogue,
        USE_HELPER_PACKED_OUT_BUFFER=config.use_helper_packed_out_buffer,
        USE_HELPER_SWIGLU=config.use_helper_swiglu,
        USE_EPILOGUE_STORE_HELPER=config.epilogue_store_helper,
        EPILOGUE_STORE_HELPER_DEPTH=config.epilogue_store_helper_depth,
        SCALE_SIZE_OUTER=scale_size_outer,
        SCALE_SIZE_INNER=scale_size_inner,
        MXFP_BLOCK_SIZE=mxfp_block_size,
        #
        num_warps=config.num_warps,
    )

    return c.unsqueeze(0)
