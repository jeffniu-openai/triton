import math
import re
from itertools import product

import pytest
import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError

from triton._internal_testing import is_blackwell, is_blackwell_ultra
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryLinearLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma,
    tcgen05_mma_barrier_count,
    tcgen05_mma_scaled,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton._C.libtriton.gluon_ir import make_cga_layout
from python.test.gluon.test_core import (
    _expected_scaled_cp_opcode,
    _run_tmem_reduction_case,
    mma_scaled_tcgen05_copy,
    random_quantized_tensor,
    swizzle_scales_packed_block,
)


def _make_tmem_linear_layout(m, n):
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(int(math.log2(m)))],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))],
        shape=[m, n],
    )


def _permute_pow2_bases(bits):
    even = list(range(0, len(bits), 2))
    odd = list(range(1, len(bits), 2))
    return [bits[i] for i in even + odd]


def _make_tmem_linear_layout_scrambled(m, n, scramble_rows=True, scramble_cols=True):
    row_bits = [1 << i for i in range(int(math.log2(m)))]
    col_bits = [1 << i for i in range(int(math.log2(n)))]
    if scramble_rows:
        row_bits = _permute_pow2_bases(row_bits)
    if scramble_cols:
        col_bits = _permute_pow2_bases(col_bits)
    return TensorMemoryLinearLayout(
        rows=[[b, 0] for b in row_bits],
        cols=[[0, b] for b in col_bits],
        shape=[m, n],
    )


def _make_tmem_linear_layout_mixed(m, n):
    assert m == 128 and n >= 64 and (n & (n - 1)) == 0
    col_bases = [[32, 0], [64, 0]]
    col = 4
    while col < n:
        col_bases.append([0, col])
        col <<= 1
    return TensorMemoryLinearLayout(
        rows=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1], [0, 2]],
        cols=col_bases,
        shape=[m, n],
    )


def _make_tmem_linear_layout_m64(n):
    return TensorMemoryLinearLayout(
        rows=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [16, 0], [32, 0]],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))],
        shape=[64, n],
    )


def _make_tmem_linear_layout_block(m, n, two_ctas=False):
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(1, int(math.log2(m)))],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))],
        block_bases=[[1, 0]],
        shape=[m, n],
        two_ctas=two_ctas,
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


def _make_tmem_linear_layout_64x32_block(two_ctas=False):
    return TensorMemoryLinearLayout(
        rows=[[2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
        cols=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        block_bases=[[1, 0]],
        shape=[64, 32],
        two_ctas=two_ctas,
    )


def _lift_tmem_layout(base_layout, prefix_shape):
    prefix_shape = list(prefix_shape)
    prefix_rank = len(prefix_shape)
    total_rank = prefix_rank + len(base_layout.shape)

    def extend_bases(bases):
        return [[0] * prefix_rank + list(basis) for basis in bases]

    rows = extend_bases(base_layout.rows)
    cols = extend_bases(base_layout.cols)
    block_bases = extend_bases(base_layout.block_bases)

    # Represent extra descriptor dimensions as progressively higher TMEM col
    # bits so that indexing/slicing them peels away full physical TMEM tiles.
    for dim in range(prefix_rank - 1, -1, -1):
        for bit in range(int(math.log2(prefix_shape[dim]))):
            basis = [0] * total_rank
            basis[dim] = 1 << bit
            cols.append(basis)

    return TensorMemoryLinearLayout(
        rows=rows,
        cols=cols,
        block_bases=block_bases,
        shape=prefix_shape + list(base_layout.shape),
        two_ctas=base_layout.two_ctas,
    )


def _default_cga_layout(num_ctas, rank, dim=0):
    if num_ctas == 1:
        return []
    return [[0] * dim + [1 << i] + [0] * (rank - dim - 1) for i in range(num_ctas.bit_length() - 1)]


def _make_tmem_register_layout(num_ctas):
    return ttgl.BlockedLayout(
        [1, 32],
        [32, 1],
        [4, 1],
        [0, 1],
        cga_layout=_default_cga_layout(num_ctas, 2),
    )


def _make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, two_cta_dim):
    ctas_per_cga = list(ctas_per_cga)
    cta_split = list(cta_split)
    assert cta_split[two_cta_dim] > 1
    cta_split[two_cta_dim] //= 2
    ctas_per_cga[two_cta_dim] //= 2
    aux_cga_layout = make_cga_layout(ctas_per_cga, cta_split, cta_order)
    basis = [0, 0]
    basis[two_cta_dim] = 1
    for b in aux_cga_layout:
        b[two_cta_dim] *= 2
    return [basis] + aux_cga_layout


def _make_scales_shared_layout_warpx4():
    return ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [32, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]]
    )


def _make_scales_shared_layout_warpx2_candidate():
    # Candidate family from PTX probes; currently expected to fail cleanly.
    return ttgl.SharedLinearLayout(
        offset_bases=[[32, 0], [0, 1], [1, 0], [0, 2], [0, 4], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]]
    )


def _extract_tcgen05_opcode_offsets(asm: str, opcodes=("ld", "st")):
    pattern = re.compile(
        rf"(tcgen05\.(?:{'|'.join(opcodes)})(?:\.red)?\.sync\.aligned\.[^\s;\"]+)"
        r"[^\n\[]*?\[\s*[^\]\n]*?\+\s*(\d+)\s*\]"
    )
    return [(opcode, int(offset)) for opcode, offset in pattern.findall(asm)]


def _extract_tcgen05_opcode_offset_immediates(asm: str, opcodes=("ld", "st")):
    pattern = re.compile(
        rf"(tcgen05\.(?:{'|'.join(opcodes)})(?:\.red)?\.sync\.aligned\.[^\s;\"]+)"
        r"[^\n\[]*?\[\s*[^\]\n]*?\+\s*(\d+)\s*\]\s*,\s*(\d+)"
    )
    return [(opcode, int(offset), int(imm)) for opcode, offset, imm in pattern.findall(asm)]


def _extract_tcgen05_cp_opcodes(asm: str):
    pattern = re.compile(
        r"(tcgen05\.cp(?:\.cta_group::\d+)?(?:\.warpx[24](?:::[^\s.;]+)*)?\.\d+x\d+b)"
    )
    return pattern.findall(asm)


def _extract_tcgen05_mma_opcodes(asm: str):
    pattern = re.compile(r"(tcgen05\.mma\.cta_group::\d+\.kind::[^\s;\"]+)")
    return pattern.findall(asm)


def _assert_ldst_ptx_llir_match(compiled):
    ptx_ops = _extract_tcgen05_opcode_offsets(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_opcode_offsets(compiled.asm["llir"])
    assert ptx_ops, "No tcgen05 ld/st opcodes in PTX"
    assert ptx_ops == llir_ops

    ptx_imms = _extract_tcgen05_opcode_offset_immediates(compiled.asm["ptx"])
    llir_imms = _extract_tcgen05_opcode_offset_immediates(compiled.asm["llir"])
    assert ptx_imms == llir_imms
    return ptx_ops, ptx_imms


@gluon.jit
def tmem_ldst_variant_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr, N: ttgl.constexpr,
                             instr_variant: ttgl.constexpr):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    tmem = allocate_tensor_memory(ttgl.float32, [M, N], layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant=instr_variant)
    value = ttgl.load(in_ptr + offs)
    tmem.store(ttgl.convert_layout(value, reg_layout))
    value = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(value, reg_layout))


@gluon.jit
def tmem_ldst_auto_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr, N: ttgl.constexpr):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    tmem = allocate_tensor_memory(ttgl.float32, [M, N], layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()
    value = ttgl.load(in_ptr + offs)
    tmem.store(ttgl.convert_layout(value, reg_layout))
    value = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(value, reg_layout))


@gluon.jit
def tmem_ldst_descriptor_chain_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr, N: ttgl.constexpr,
                                      instr_variant: ttgl.constexpr):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)

    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
    view = tmem.slice(1, 1).index(0).reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
    view = view.permute([1, 0]).permute([1, 0])
    view = view.slice(0, M, dim=0).slice(0, N, dim=1)
    view = view._reinterpret(ttgl.float32, [M, N], layout)

    reg_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    view.store(ttgl.convert_layout(value, reg_layout))
    out = view.load(reg_layout)
    out = out + ttgl.full([M, N], 3.0, ttgl.float32, layout=reg_layout)
    view.store(out)
    out = tmem.index(1).load(reg_layout)
    ttgl.store(out_ptr + offs, out)


@gluon.jit
def tmem_ldst_descriptor_roundtrip_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr,
                                          N: ttgl.constexpr, instr_variant: ttgl.constexpr,
                                          chain_id: ttgl.constexpr, delta: ttgl.constexpr):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)

    tmem = allocate_tensor_memory(ttgl.float32, [2, 2, M, N], layout)
    base = tmem.slice(1, 1, dim=0).index(0).slice(1, 1, dim=0).index(0)
    base_reg_layout: ttgl.constexpr = base.get_reg_layout(instr_variant=instr_variant)
    base.store(ttgl.convert_layout(value, base_reg_layout))

    view = base
    if chain_id == 0:
        view = view.reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
        view = view.permute([1, 0]).permute([1, 0])
    elif chain_id == 1:
        view = view.reshape((M // 2, 2, N // 2, 2))
        view = view.permute([1, 0, 3, 2]).permute([1, 0, 3, 2]).reshape((M, N))
        view = view.slice(0, M, dim=0).slice(0, N, dim=1)
    else:
        view = view.reshape((2, M // 4, 2, N // 2, 2))
        view = view.permute([2, 1, 0, 4, 3]).permute([2, 1, 0, 4, 3]).reshape((M, N))
        view = view._reinterpret(ttgl.float32, [M, N], view.layout)

    view_reg_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    out = view.load(view_reg_layout)
    out = out + ttgl.full([M, N], delta, ttgl.float32, layout=view_reg_layout)
    view.store(out)

    out = base.load(base_reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, base_reg_layout))


@gluon.jit
def tmem_ldst_descriptor_rank5_roundtrip_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr,
                                                N: ttgl.constexpr, instr_variant: ttgl.constexpr,
                                                delta: ttgl.constexpr):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)

    tmem = allocate_tensor_memory(ttgl.float32, [2, 2, 2, M, N], layout)
    base = tmem.index(1).index(1).index(0)
    base_reg_layout: ttgl.constexpr = base.get_reg_layout(instr_variant=instr_variant)
    base.store(ttgl.convert_layout(value, base_reg_layout))

    alias = tmem.slice(1, 1, dim=0).index(0).slice(1, 1, dim=0).index(0).slice(0, 1, dim=0).index(0)
    alias = alias.reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
    alias = alias.permute([1, 0]).permute([1, 0])
    alias = alias.slice(0, M, dim=0).slice(0, N, dim=1)

    alias_reg_layout: ttgl.constexpr = alias.get_reg_layout(instr_variant=instr_variant)
    out = alias.load(alias_reg_layout)
    out = out + ttgl.full([M, N], delta, ttgl.float32, layout=alias_reg_layout)
    alias.store(out)

    out = base.load(base_reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, base_reg_layout))


@gluon.jit
def tmem_ldst_descriptor_higher_rank_index_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr,
                                                  N: ttgl.constexpr, instr_variant: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
    full_view = tmem.index(1)
    full_reg_layout: ttgl.constexpr = full_view.get_reg_layout(instr_variant=instr_variant)
    in_offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, full_reg_layout))
    in_offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, full_reg_layout))
    in_offs = in_offs_m[:, None] * N + in_offs_n[None, :]
    value = ttgl.load(in_ptr + in_offs)
    full_view.store(ttgl.convert_layout(value, full_reg_layout))

    view = full_view.reshape((2, M // 2, 2, N // 2)).permute([2, 1, 0, 3]).reshape((2, M, N // 2))

    part0 = view.slice(0, 1, dim=0).index(0)
    part1 = view.slice(1, 1, dim=0).index(0)
    part_layout: ttgl.constexpr = part0.get_reg_layout(instr_variant=instr_variant)

    part0_value = part0.load(part_layout)
    part0.store(part0_value + ttgl.full([M, N // 2], 5.0, ttgl.float32, layout=part_layout))
    part1_value = part1.load(part_layout)
    part1.store(part1_value + ttgl.full([M, N // 2], 5.0, ttgl.float32, layout=part_layout))

    out = full_view.load(full_reg_layout)
    ttgl.store(out_ptr + in_offs, out)


@gluon.jit
def tmem_ldst_descriptor_multidim_slice_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr,
                                               N: ttgl.constexpr, instr_variant: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
    full_view = tmem.index(1)
    full_reg_layout: ttgl.constexpr = full_view.get_reg_layout(instr_variant=instr_variant)
    in_offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, full_reg_layout))
    in_offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, full_reg_layout))
    in_offs = in_offs_m[:, None] * N + in_offs_n[None, :]
    value = ttgl.load(in_ptr + in_offs)
    full_view.store(ttgl.convert_layout(value, full_reg_layout))

    view = full_view.reshape((M, 2, 2, N // 4)).permute([2, 0, 1, 3]).reshape((2, M, N // 2))

    part0 = view.slice(0, 1, dim=0).index(0)
    part1 = view.slice(1, 1, dim=0).index(0)
    part_layout: ttgl.constexpr = part0.get_reg_layout(instr_variant=instr_variant)

    part0_value = part0.load(part_layout)
    part0.store(part0_value + ttgl.full([M, N // 2], 9.0, ttgl.float32, layout=part_layout))
    part1_value = part1.load(part_layout)
    part1.store(part1_value + ttgl.full([M, N // 2], 9.0, ttgl.float32, layout=part_layout))

    out = full_view.load(full_reg_layout)
    ttgl.store(out_ptr + in_offs, out)


@gluon.jit
def tmem_ldst_descriptor_higher_rank_dim0_slice_positive_kernel(in_ptr, out_ptr, layout: ttgl.constexpr,
                                                                M: ttgl.constexpr, N: ttgl.constexpr,
                                                                instr_variant: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
    full_view = tmem.index(1)
    full_reg_layout: ttgl.constexpr = full_view.get_reg_layout(instr_variant=instr_variant)
    in_offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, full_reg_layout))
    in_offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, full_reg_layout))
    in_offs = in_offs_m[:, None] * N + in_offs_n[None, :]
    value = ttgl.load(in_ptr + in_offs)
    full_view.store(ttgl.convert_layout(value, full_reg_layout))

    view = full_view.reshape((2, M, N // 2)).slice(1, 1, dim=0).index(0)
    part_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    part_value = view.load(part_layout)
    view.store(part_value + ttgl.full([M, N // 2], 7.0, ttgl.float32, layout=part_layout))

    out = full_view.load(full_reg_layout)
    ttgl.store(out_ptr + in_offs, out)


@gluon.jit
def tmem_ldst_descriptor_higher_rank_half_rows_positive_kernel(in_ptr, out_ptr, layout: ttgl.constexpr,
                                                               M: ttgl.constexpr, N: ttgl.constexpr,
                                                               instr_variant: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
    full_view = tmem.index(1)
    full_reg_layout: ttgl.constexpr = full_view.get_reg_layout(instr_variant=instr_variant)
    in_offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, full_reg_layout))
    in_offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, full_reg_layout))
    in_offs = in_offs_m[:, None] * N + in_offs_n[None, :]
    value = ttgl.load(in_ptr + in_offs)
    full_view.store(ttgl.convert_layout(value, full_reg_layout))

    view = full_view.reshape((2, M // 2, N)).slice(1, 1, dim=0).index(0)
    part_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    part_value = view.load(part_layout)
    view.store(part_value + ttgl.full([M // 2, N], 13.0, ttgl.float32, layout=part_layout))

    out = full_view.load(full_reg_layout)
    ttgl.store(out_ptr + in_offs, out)


@gluon.jit
def tmem_ldst_blocked_fallback_kernel(in_ptr, out_ptr, layout: ttgl.constexpr):
    M: ttgl.constexpr = 128
    N: ttgl.constexpr = 128
    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    offs = ttgl.arange(0, M, ttgl.SliceLayout(1, blocked))[:, None] * N + ttgl.arange(
        0, N, ttgl.SliceLayout(0, blocked)
    )[None, :]
    value = ttgl.load(in_ptr + offs)
    tmem = allocate_tensor_memory(ttgl.float32, [M, N], layout)
    tmem.store(value)
    value = tmem.load(blocked)
    ttgl.store(out_ptr + offs, value)


@gluon.jit
def tmem_block_descriptor_compile_kernel(layout: ttgl.constexpr, reinterpret_layout: ttgl.constexpr,
                                         reg_layout: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [2, 128, 128], layout)
    view = tmem.slice(1, 1, dim=0).index(0).permute([1, 0]).reshape((64, 2, 128))
    view = view.permute([0, 2, 1]).reshape((64, 32, 8))
    view = view.slice(16, 32, dim=0).slice(8, 16, dim=1).slice(2, 4, dim=2)
    view = view._reinterpret(ttgl.float32, [64, 32], reinterpret_layout)
    value = ttgl.full([64, 32], 0.0, ttgl.float32, layout=reg_layout)
    view.store(ttgl.convert_layout(value, reg_layout))
    _ = view.load(reg_layout)


@gluon.jit
def tmem_ldst_f16_variant_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr, N: ttgl.constexpr,
                                 instr_variant: ttgl.constexpr):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    tmem = allocate_tensor_memory(ttgl.float16, [M, N], layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant=instr_variant)
    value = ttgl.load(in_ptr + offs)
    tmem.store(ttgl.convert_layout(value, reg_layout))
    value = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(value, reg_layout))


@gluon.jit
def tmem_copy_no_scales_kernel(in_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                               swizzle: ttgl.constexpr):
    tmem_layout: ttgl.constexpr = TensorMemoryLayout(
        block=(128, BLOCK_N),
        col_stride=32 // in_ptr.dtype.element_ty.primitive_bitwidth,
    )
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], tmem_layout)
    tmem_reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, tmem_reg_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, tmem_reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=swizzle, element_bitwidth=32, rank=2)
    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)
    smem.store(value)

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)

    out = tmem.load()
    ttgl.store(out_ptr + offs, out)


@gluon.jit
def tmem_copy_no_scales_linear_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr, N: ttgl.constexpr,
                                      swizzle: ttgl.constexpr):
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=swizzle, element_bitwidth=32, rank=2)
    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)
    smem.store(value)

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)

    out = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, out)


@gluon.jit
def tmem_copy_128x128_kernel(in_ptr, out_ptr, M: ttgl.constexpr):
    N: ttgl.constexpr = 4
    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, blocked))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, blocked))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]],
        alignment=16,
    )
    smem = ttgl.allocate_shared_memory(ttgl.int32, [M, N], layout=smem_layout)
    tmem_layout: ttgl.constexpr = TensorMemoryLayout((M, N), col_stride=1)
    tmem = allocate_tensor_memory(ttgl.int32, [M, N], layout=tmem_layout)

    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    smem.store(value)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    output = tmem.load(blocked)
    ttgl.store(out_ptr + offs, output)


@gluon.jit
def tmem_copy_scales_warpx4_kernel(in_ptr, out_ptr):
    SMEM_H: ttgl.constexpr = 64
    SMEM_W: ttgl.constexpr = 16
    NUM_ROWS: ttgl.constexpr = 128
    NUM_COLS: ttgl.constexpr = (SMEM_H * SMEM_W) // 32

    in_ptrs = in_ptr + ttgl.arange(0, SMEM_H)[:, None] * SMEM_W + ttgl.arange(0, SMEM_W)[None, :]
    out_ptrs = out_ptr + ttgl.arange(0, NUM_ROWS)[:, None] * NUM_COLS + ttgl.arange(0, NUM_COLS)[None, :]

    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    value = ttgl.load(ttgl.set_auto_layout(in_ptrs, blocked))

    smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [32, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]]
    )
    tmem = allocate_tensor_memory(ttgl.int8, (SMEM_H, SMEM_W), layout=TensorMemoryScalesLayout())
    smem = ttgl.allocate_shared_memory(ttgl.int8, (SMEM_H, SMEM_W), layout=smem_layout)
    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)

    smem.store(value)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    tmem_alias: ttgl.constexpr = TensorMemoryLayout((NUM_ROWS, NUM_COLS), col_stride=1)
    alias_view = tmem._reinterpret(ttgl.int8, (NUM_ROWS, NUM_COLS), tmem_alias)
    value = alias_view.load(blocked)
    ttgl.store(ttgl.set_auto_layout(out_ptrs, blocked), value)


@gluon.jit
def tmem_copy_scales_layout_probe_kernel(in_ptr, out_ptr, smem_layout: ttgl.constexpr):
    SMEM_H: ttgl.constexpr = 64
    SMEM_W: ttgl.constexpr = 16
    NUM_ROWS: ttgl.constexpr = 128
    NUM_COLS: ttgl.constexpr = (SMEM_H * SMEM_W) // 32

    in_ptrs = in_ptr + ttgl.arange(0, SMEM_H)[:, None] * SMEM_W + ttgl.arange(0, SMEM_W)[None, :]
    out_ptrs = out_ptr + ttgl.arange(0, NUM_ROWS)[:, None] * NUM_COLS + ttgl.arange(0, NUM_COLS)[None, :]

    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    value = ttgl.load(ttgl.set_auto_layout(in_ptrs, blocked))

    tmem = allocate_tensor_memory(ttgl.int8, (SMEM_H, SMEM_W), layout=TensorMemoryScalesLayout())
    smem = ttgl.allocate_shared_memory(ttgl.int8, (SMEM_H, SMEM_W), layout=smem_layout)
    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)

    smem.store(value)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    tmem_alias: ttgl.constexpr = TensorMemoryLayout((NUM_ROWS, NUM_COLS), col_stride=1)
    alias_view = tmem._reinterpret(ttgl.int8, (NUM_ROWS, NUM_COLS), tmem_alias)
    value = alias_view.load(blocked)
    ttgl.store(ttgl.set_auto_layout(out_ptrs, blocked), value)


@gluon.jit
def tmem_mma_twocta_kernel(a_desc, b_desc, out_ptrs, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                           acc_tmem_layout: ttgl.constexpr, blocked_c: ttgl.constexpr):
    smem_a = ttgl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    smem_b = ttgl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)

    tma_bar = mbarrier.allocate_mbarrier(two_ctas=acc_tmem_layout.two_ctas)
    mbarrier.init(tma_bar, count=1)
    mma_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(mma_bar, count=tcgen05_mma_barrier_count([smem_a, smem_b], True))

    mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
    tma.async_copy_global_to_shared(a_desc, [0, 0], tma_bar, smem_a, multicast=True)
    tma.async_copy_global_to_shared(b_desc, [0, 0], tma_bar, smem_b, multicast=True)
    mbarrier.wait(tma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(tma_bar)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [BLOCK_M, BLOCK_N], acc_tmem_layout)
    tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=False, multicast=True, mbarriers=[mma_bar])
    mbarrier.wait(mma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(mma_bar)

    out = acc_tmem.load()
    out = ttgl.convert_layout(out, blocked_c)
    out_offs_m = ttgl.arange(0, BLOCK_M)[:, None]
    out_offs_n = ttgl.arange(0, BLOCK_N)[None, :]
    ttgl.store(out_ptrs + out_offs_m * BLOCK_N + out_offs_n, out)


@gluon.jit
def tmem_mma_kernel(a_ptr, b_ptr, c_ptr, out_ptr, layout: ttgl.constexpr, use_acc: ttgl.constexpr):
    M: ttgl.constexpr = 128
    N: ttgl.constexpr = 128
    K: ttgl.constexpr = 32
    a_offs = ttgl.arange(0, M)[:, None] * K + ttgl.arange(0, K)[None, :]
    b_offs = ttgl.arange(0, K)[:, None] * N + ttgl.arange(0, N)[None, :]
    c_offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]

    blocked_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    blocked_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    a = ttgl.load(ttgl.set_auto_layout(a_ptr + a_offs, blocked_a))
    b = ttgl.load(ttgl.set_auto_layout(b_ptr + b_offs, blocked_b))
    c = ttgl.load(c_ptr + c_offs)

    smem_a_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16, rank=2)
    smem_b_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16, rank=2)
    smem_a = ttgl.allocate_shared_memory(ttgl.float16, [M, K], layout=smem_a_layout)
    smem_b = ttgl.allocate_shared_memory(ttgl.float16, [K, N], layout=smem_b_layout)
    smem_a.store(a)
    smem_b.store(b)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], layout)
    if use_acc:
        acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
        acc_tmem.store(ttgl.convert_layout(c, acc_reg_layout))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=tcgen05_mma_barrier_count([smem_a, smem_b], False))
    tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=use_acc, mbarriers=[bar])
    mbarrier.wait(bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(bar)

    out = acc_tmem.load()
    ttgl.store(out_ptr + c_offs, out)


@gluon.jit
def tmem_mma_scaled_minimal_kernel(out_ptr, M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr, a, b, a_scale,
                                   b_scale):
    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [ttgl.num_warps(), 1], [1, 0])
    nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, transposed=False,
                                                          element_bitwidth=8, rank=2)
    block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])

    a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout))[:, None]
    a_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, block_layout))[None, :]
    b_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(1, block_layout))[:, None]
    b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, block_layout))[None, :]

    a_tile = ttgl.load(a + a_offs_m * K + a_offs_k)
    b_tile = ttgl.load(b + b_offs_k * N + b_offs_n)
    a_smem = ttgl.allocate_shared_memory(ttgl.float8e5, [M, K], nvmma_layout, a_tile)
    b_smem = ttgl.allocate_shared_memory(ttgl.float8e5, [K, N], nvmma_layout, b_tile)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], TensorMemoryLayout([M, N], col_stride=1))
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.zeros([M, N], ttgl.float32, layout=acc_reg_layout))

    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [M, K // 32], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype.element_ty, [N, K // 32], scale_layout)
    scale_reg_layout_m: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_reg_layout_n: ttgl.constexpr = b_scale_tmem.get_reg_layout()

    scale_offs_k = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
    scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
    scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
    a_scale_tmem.store(ttgl.load(a_scale + scale_offs_m * (K // 32) + scale_offs_k))
    b_scale_tmem.store(ttgl.load(b_scale + scale_offs_n * (K // 32) + scale_offs_k))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(a_smem, b_smem, acc_tmem, a_scale_tmem, b_scale_tmem, "e5m2", "e5m2", use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)

    out_reg = acc_tmem.load()
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, reg_layout))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, reg_layout))[None, :]
    offs = offs_m * N + offs_n
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out_reg, reg_layout))


LDST_SHAPE_MAP = {
    "auto": {64: "32x32b.x64.b32", 128: "32x32b.x128.b32", 256: "32x32b.x64.b32"},
    "32x32b": {64: "32x32b.x64.b32", 128: "32x32b.x128.b32", 256: "32x32b.x64.b32"},
    "16x64b": {64: "16x64b.x32.b32", 128: "16x64b.x64.b32", 256: "16x64b.x64.b32"},
    "16x128b": {64: "16x128b.x16.b32", 128: "16x128b.x32.b32", 256: "16x128b.x32.b32"},
    "16x256b": {64: "16x256b.x8.b32", 128: "16x256b.x16.b32", 256: "16x256b.x16.b32"},
}

LDST_SUBVIEW_SHAPE_MAP = {
    "32x32b": {32: "32x32b.x32.b32", 64: "32x32b.x64.b32", 128: "32x32b.x128.b32"},
    "16x64b": {32: "16x64b.x16.b32", 64: "16x64b.x32.b32", 128: "16x64b.x64.b32"},
    "16x128b": {32: "16x128b.x8.b32", 64: "16x128b.x16.b32", 128: "16x128b.x32.b32"},
    "16x256b": {32: "16x256b.x4.b32", 64: "16x256b.x8.b32", 128: "16x256b.x16.b32"},
}

LDST_LAYOUTS = {
    "identity": lambda n: _make_tmem_linear_layout(128, n),
    "mixed": lambda n: _make_tmem_linear_layout_mixed(128, n),
}

LDST_EXOTIC_LAYOUTS = {
    "scrambled_cols": lambda n: _make_tmem_linear_layout_scrambled(128, n, scramble_rows=False, scramble_cols=True),
    "scrambled_rows_cols": lambda n: _make_tmem_linear_layout_scrambled(128, n, scramble_rows=True,
                                                                         scramble_cols=True),
}

LDST_EXOTIC_UNSUPPORTED_LAYOUTS = {
    "block_single_cta": lambda n: _make_tmem_linear_layout_block(128, n, two_ctas=False),
}

LDST_TWOCTA_LAYOUTS = {
    "block_two_ctas": lambda n: _make_tmem_linear_layout_block(256, n, two_ctas=True),
    "mmav5_twocta": lambda n: _make_tmem_linear_layout_mmav5_twocta(256, n),
}

LDST_VARIANTS = ("auto", "32x32b", "16x64b", "16x128b", "16x256b")
LDST_EXPLICIT_VARIANTS = ("32x32b", "16x64b", "16x128b", "16x256b")

LDST_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for layout_name, n, variant in product(LDST_LAYOUTS.keys(), (64, 128, 256), LDST_VARIANTS)
]

LDST_DESCRIPTOR_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for layout_name, n, variant in product(LDST_LAYOUTS.keys(), (64, 128, 256), LDST_EXPLICIT_VARIANTS)
]

LDST_TWOCTA_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for layout_name, n, variant in product(LDST_TWOCTA_LAYOUTS.keys(), (64, 128, 256), LDST_VARIANTS)
]

LDST_TWOCTA_DESCRIPTOR_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for layout_name, n, variant in product(LDST_TWOCTA_LAYOUTS.keys(), (64, 128, 256), LDST_EXPLICIT_VARIANTS)
]

LDST_EXOTIC_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for layout_name, n, variant in product(LDST_EXOTIC_LAYOUTS.keys(), (64, 128, 256), LDST_EXPLICIT_VARIANTS)
]

LDST_EXOTIC_DESCRIPTOR_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for layout_name, n, variant in product(("scrambled_cols", "scrambled_rows_cols"), (64, 128, 256),
                                           LDST_EXPLICIT_VARIANTS)
]

LDST_EXOTIC_UNSUPPORTED_CASES = [
    (layout_name, n, variant)
    for layout_name, n, variant in product(LDST_EXOTIC_UNSUPPORTED_LAYOUTS.keys(), (64, 128, 256),
                                           LDST_EXPLICIT_VARIANTS)
]

LDST_DESCRIPTOR_ROUNDTRIP_CHAINS = [
    ("slice_index_roundtrip", 0, 5.0, ("ttg.memdesc_index", "ttg.memdesc_subslice", "ttg.memdesc_reshape",
                                       "ttg.memdesc_trans")),
    ("slice_index_multidim", 1, 7.0, ("ttg.memdesc_index", "ttg.memdesc_subslice", "ttg.memdesc_reshape",
                                      "ttg.memdesc_trans")),
    ("slice_index_reinterpret", 2, 11.0, ("ttg.memdesc_index", "ttg.memdesc_subslice", "ttg.memdesc_reshape",
                                          "ttg.memdesc_trans", "ttg.memdesc_reinterpret")),
]

LDST_HIGHER_RANK_INDEX_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n], LDST_SUBVIEW_SHAPE_MAP[variant][n // 2])
    for layout_name, n, variant in product(LDST_LAYOUTS.keys(), (64, 128, 256), LDST_EXPLICIT_VARIANTS)
]

LDST_HIGHER_RANK_SLICE_CASES = [
    (layout_name, n, variant) for layout_name, n, variant in product(LDST_LAYOUTS.keys(), (64, 128, 256),
                                                                      LDST_EXPLICIT_VARIANTS)
]

LDST_TWOCTA_HIGHER_RANK_INDEX_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n], LDST_SUBVIEW_SHAPE_MAP[variant][n // 2])
    for layout_name, n, variant in product(LDST_TWOCTA_LAYOUTS.keys(), (64, 128, 256), LDST_EXPLICIT_VARIANTS)
]

LDST_TWOCTA_HIGHER_RANK_SLICE_CASES = [
    (layout_name, n, variant) for layout_name, n, variant in product(LDST_TWOCTA_LAYOUTS.keys(), (64, 128, 256),
                                                                      LDST_EXPLICIT_VARIANTS)
]

LDST_HIGHER_RANK_POSITIVE_CASES = [
    ("identity", n, variant, LDST_SHAPE_MAP[variant][n])
    for n, variant in product((64, 128, 256), LDST_EXPLICIT_VARIANTS)
]

LDST_TWOCTA_HIGHER_RANK_POSITIVE_CASES = [
    ("block_two_ctas", n, variant, LDST_SHAPE_MAP[variant][n])
    for n, variant in product((64, 128, 256), LDST_EXPLICIT_VARIANTS)
]

LDST_TWOCTA_MMAV5_HIGHER_RANK_UNSUPPORTED_CASES = [
    ("mmav5_twocta", n, variant) for n, variant in product((64, 128, 256), LDST_EXPLICIT_VARIANTS)
]

BLOCKED_FALLBACK_CASES = [
    ("identity", _make_tmem_linear_layout(128, 128)),
    ("mixed", _make_tmem_linear_layout_mixed(128, 128)),
]

UNSUPPORTED_BLOCK_DESCRIPTOR_CASES = [
    ("block", _make_tmem_linear_layout_block(128, 128), _make_tmem_linear_layout_64x32_block()),
    ("block_two_ctas", _make_tmem_linear_layout_block(128, 128, two_ctas=True),
     _make_tmem_linear_layout_64x32_block(two_ctas=True)),
]

M64_SPLITN_CASES = []
for n in (2, 4, 8, 16, 32, 64, 128):
    if n == 2:
        M64_SPLITN_CASES.append((n, 2, [(0, 0)]))
    else:
        x = n // 4
        M64_SPLITN_CASES.append((n, x, [(0, x), (2 * x, x)]))

M64_SPLITN_AUTO_CASES = [
    (2, 1, 1),
    (4, 2, 2),
    (8, 4, 4),
    (16, 8, 8),
    (32, 16, 16),
    (64, 32, 32),
    (128, 64, 64),
]

LDST_DESCRIPTOR_RANK5_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for layout_name, n, variant in product(LDST_LAYOUTS.keys(), (64, ), ("32x32b", "16x64b", "16x128b", "16x256b"))
]

LDST_TWOCTA_DESCRIPTOR_RANK5_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for layout_name, n, variant in product(LDST_TWOCTA_LAYOUTS.keys(), (64, ), ("32x32b", "16x64b"))
]

CP_NO_SCALES_CASES = [
    (m, n, block_n, 32)
    for m, n, block_n in product((128, 256), (16, 32, 64, 128, 256), (16, 32, 64, 128, 256))
    if n % block_n == 0
]

CP_LINEAR_NO_SCALES_CASES = [
    (128, 128, 32, 16),
    (128, 128, 64, 16),
    (128, 128, 128, 16),
    (128, 256, 32, 32),
    (128, 256, 64, 32),
    (128, 256, 128, 32),
]

CP_NO_SCALES_SWIZZLE_CASES = [
    (m, n, block_n, swizzle)
    for swizzle in (32, 64, 128)
    for (m, n, block_n) in ((128, 128, 128), (128, 256, 256), (256, 128, 64))
]

F16_LDST_SHAPE_MAP = {
    "32x32b": {64: "32x32b.x32.b32", 128: "32x32b.x64.b32", 256: "32x32b.x128.b32"},
    "16x64b": {64: "16x64b.x16.b32", 128: "16x64b.x32.b32", 256: "16x64b.x64.b32"},
    "16x128b": {64: "16x128b.x8.b32", 128: "16x128b.x16.b32", 256: "16x128b.x32.b32"},
    "16x256b": {64: "16x256b.x4.b32", 128: "16x256b.x8.b32", 256: "16x256b.x16.b32"},
}

F16_LDST_CASES = [
    ("identity", n, variant, F16_LDST_SHAPE_MAP[variant][n])
    for n, variant in product((64, 128, 256), ("32x32b", "16x64b", "16x128b", "16x256b"))
]

LD_RED_LINEAR_CASES = [
    ("identity", 128, 32, 4, "32x32b.x32"),
    ("identity", 128, 64, 4, "32x32b.x64"),
    ("identity", 128, 128, 4, "32x32b.x128"),
    ("identity", 128, 256, 4, "32x32b.x64"),
]

LD_RED_EXPECTED_OP_COUNT = {32: 1, 64: 1, 128: 1, 256: 4}

LD_RED_MIXED_UNSUPPORTED_CASES = [
    ("mixed", 128, 64, 4),
    ("mixed", 128, 128, 4),
    ("mixed", 128, 256, 4),
]

LDST_EXPECTED_OFFSETS_128x256 = {
    "32x32b": [
        ("tcgen05.st.sync.aligned.32x32b.x64.b32", 0),
        ("tcgen05.st.sync.aligned.32x32b.x64.b32", 64),
        ("tcgen05.st.sync.aligned.32x32b.x64.b32", 128),
        ("tcgen05.st.sync.aligned.32x32b.x64.b32", 192),
        ("tcgen05.ld.sync.aligned.32x32b.x64.b32", 0),
        ("tcgen05.ld.sync.aligned.32x32b.x64.b32", 64),
        ("tcgen05.ld.sync.aligned.32x32b.x64.b32", 128),
        ("tcgen05.ld.sync.aligned.32x32b.x64.b32", 192),
    ],
    "16x64b": [
        ("tcgen05.st.sync.aligned.16x64b.x64.b32", 0),
        ("tcgen05.st.sync.aligned.16x64b.x64.b32", 128),
        ("tcgen05.st.sync.aligned.16x64b.x64.b32", 1048576),
        ("tcgen05.st.sync.aligned.16x64b.x64.b32", 1048704),
        ("tcgen05.ld.sync.aligned.16x64b.x64.b32", 0),
        ("tcgen05.ld.sync.aligned.16x64b.x64.b32", 128),
        ("tcgen05.ld.sync.aligned.16x64b.x64.b32", 1048576),
        ("tcgen05.ld.sync.aligned.16x64b.x64.b32", 1048704),
    ],
    "16x128b": [
        ("tcgen05.st.sync.aligned.16x128b.x32.b32", 0),
        ("tcgen05.st.sync.aligned.16x128b.x32.b32", 128),
        ("tcgen05.st.sync.aligned.16x128b.x32.b32", 1048576),
        ("tcgen05.st.sync.aligned.16x128b.x32.b32", 1048704),
        ("tcgen05.ld.sync.aligned.16x128b.x32.b32", 0),
        ("tcgen05.ld.sync.aligned.16x128b.x32.b32", 128),
        ("tcgen05.ld.sync.aligned.16x128b.x32.b32", 1048576),
        ("tcgen05.ld.sync.aligned.16x128b.x32.b32", 1048704),
    ],
    "16x256b": [
        ("tcgen05.st.sync.aligned.16x256b.x16.b32", 0),
        ("tcgen05.st.sync.aligned.16x256b.x16.b32", 128),
        ("tcgen05.st.sync.aligned.16x256b.x16.b32", 1048576),
        ("tcgen05.st.sync.aligned.16x256b.x16.b32", 1048704),
        ("tcgen05.ld.sync.aligned.16x256b.x16.b32", 0),
        ("tcgen05.ld.sync.aligned.16x256b.x16.b32", 128),
        ("tcgen05.ld.sync.aligned.16x256b.x16.b32", 1048576),
        ("tcgen05.ld.sync.aligned.16x256b.x16.b32", 1048704),
    ],
}

CP_SCALES_LAYOUT_PROBE_CASES = [
    ("warpx4", _make_scales_shared_layout_warpx4(), "PASS"),
    ("warpx2_candidate", _make_scales_shared_layout_warpx2_candidate(), "BUG"),
]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_CASES)
def test_tmem_runtime_matrix_ldst(layout_name, n, variant, expected_shape):
    m = 128
    layout = LDST_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    if variant == "auto":
        compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4)
    else:
        compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert all(op in (expected_st, expected_ld) for op in observed_opcodes)
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_EXOTIC_CASES)
def test_tmem_runtime_matrix_ldst_exotic_linear_layouts(layout_name, n, variant, expected_shape):
    m = 128
    layout = LDST_EXOTIC_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant", LDST_EXOTIC_UNSUPPORTED_CASES)
def test_tmem_runtime_matrix_ldst_exotic_layouts_report_clean_unsupported(layout_name, n, variant):
    m = 128
    layout = LDST_EXOTIC_UNSUPPORTED_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(CompilationError) as excinfo:
        tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)

    msg = str(excinfo.value)
    assert f"TMEM layout '{variant}' unsupported" in msg
    assert "reshape or permute so TMEM columns stay contiguous" in msg
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_DESCRIPTOR_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_compositions(layout_name, n, variant, expected_shape):
    m = 128
    layout = LDST_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_chain_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp + 3.0, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes

    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_EXOTIC_DESCRIPTOR_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_compositions_exotic_layouts(layout_name, n, variant, expected_shape):
    m = 128
    layout = LDST_EXOTIC_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_chain_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp + 3.0, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes

    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_TWOCTA_CASES)
def test_tmem_runtime_matrix_ldst_twocta_block_layouts(layout_name, n, variant, expected_shape):
    m = 256
    layout = LDST_TWOCTA_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    if variant == "auto":
        compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4, num_ctas=2)
    else:
        compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4, num_ctas=2)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert all(op in (expected_st, expected_ld) for op in observed_opcodes)
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes
    assert "twoCTAs = true" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_TWOCTA_DESCRIPTOR_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_compositions(layout_name, n, variant, expected_shape):
    m = 256
    layout = LDST_TWOCTA_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_chain_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4, num_ctas=2)
    torch.testing.assert_close(out, inp + 3.0, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes

    ttgir = compiled.asm["ttgir"]
    assert "twoCTAs = true" in ttgir
    assert "tensor_memory_linear" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_DESCRIPTOR_CASES)
@pytest.mark.parametrize("chain_name,chain_id,delta,required_ops", LDST_DESCRIPTOR_ROUNDTRIP_CHAINS)
def test_tmem_runtime_matrix_ldst_descriptor_roundtrip_sweeps(layout_name, n, variant, expected_shape, chain_name,
                                                              chain_id, delta, required_ops):
    m = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2, 2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    try:
        compiled = tmem_ldst_descriptor_roundtrip_kernel[(1, )](
            inp, out, layout, m, n, variant, chain_id, delta, num_warps=4
        )
    except triton.runtime.errors.OutOfResources:
        pytest.skip(f"tensor memory OOR for chain={chain_name}, layout={layout_name}, n={n}, variant={variant}")
    torch.testing.assert_close(out, inp + delta, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes

    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_TWOCTA_DESCRIPTOR_CASES)
@pytest.mark.parametrize("chain_name,chain_id,delta,required_ops", LDST_DESCRIPTOR_ROUNDTRIP_CHAINS)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_roundtrip_sweeps(layout_name, n, variant, expected_shape,
                                                                     chain_name, chain_id, delta, required_ops):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2, 2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    try:
        compiled = tmem_ldst_descriptor_roundtrip_kernel[(1, )](
            inp, out, layout, m, n, variant, chain_id, delta, num_warps=4, num_ctas=2
        )
    except triton.runtime.errors.OutOfResources:
        pytest.skip(f"tensor memory OOR for chain={chain_name}, layout={layout_name}, n={n}, variant={variant}")
    torch.testing.assert_close(out, inp + delta, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes

    ttgir = compiled.asm["ttgir"]
    assert "twoCTAs = true" in ttgir
    assert "tensor_memory_linear" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape,expected_half_shape", LDST_HIGHER_RANK_INDEX_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_higher_rank_index(layout_name, n, variant, expected_shape,
                                                               expected_half_shape):
    m = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_higher_rank_index_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp + 5.0, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    expected_full_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_full_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    expected_half_st = f"tcgen05.st.sync.aligned.{expected_half_shape}"
    expected_half_ld = f"tcgen05.ld.sync.aligned.{expected_half_shape}"
    assert expected_full_st in observed_opcodes
    assert expected_full_ld in observed_opcodes
    assert expected_half_st in observed_opcodes
    assert expected_half_ld in observed_opcodes

    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttg.memdesc_reshape" in ttgir
    assert "ttg.memdesc_trans" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant", LDST_HIGHER_RANK_SLICE_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_multidim_slices_reports_clean_error(layout_name, n, variant):
    m = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(CompilationError) as excinfo:
        tmem_ldst_descriptor_multidim_slice_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)

    msg = str(excinfo.value)
    assert "unsupported tensor memory memdesc_subslice view" in msg
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape,expected_half_shape", LDST_TWOCTA_HIGHER_RANK_INDEX_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_index(layout_name, n, variant, expected_shape,
                                                                      expected_half_shape):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_higher_rank_index_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
    )
    torch.testing.assert_close(out, inp + 5.0, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    expected_full_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_full_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    expected_half_st = f"tcgen05.st.sync.aligned.{expected_half_shape}"
    expected_half_ld = f"tcgen05.ld.sync.aligned.{expected_half_shape}"
    assert expected_full_st in observed_opcodes
    assert expected_full_ld in observed_opcodes
    assert expected_half_st in observed_opcodes
    assert expected_half_ld in observed_opcodes

    ttgir = compiled.asm["ttgir"]
    assert "twoCTAs = true" in ttgir
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttg.memdesc_reshape" in ttgir
    assert "ttg.memdesc_trans" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant", LDST_TWOCTA_HIGHER_RANK_SLICE_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_multidim_slices_reports_clean_error(layout_name, n, variant):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(CompilationError) as excinfo:
        tmem_ldst_descriptor_multidim_slice_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
        )

    msg = str(excinfo.value)
    assert "unsupported tensor memory memdesc_subslice view" in msg
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_HIGHER_RANK_POSITIVE_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_higher_rank_dim0_slice_reports_clean_error_lifted_layout(
    layout_name, n, variant, expected_shape
):
    m = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(CompilationError) as excinfo:
        tmem_ldst_descriptor_higher_rank_dim0_slice_positive_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4
        )

    msg = str(excinfo.value)
    assert f"TMEM layout '{variant}' unsupported" in msg
    assert "reshape or permute so TMEM columns stay contiguous" in msg
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_HIGHER_RANK_POSITIVE_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout(
    layout_name, n, variant, expected_shape
):
    m = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(CompilationError) as excinfo:
        tmem_ldst_descriptor_higher_rank_half_rows_positive_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4
        )

    msg = str(excinfo.value)
    assert f"TMEM layout '{variant}' unsupported" in msg
    assert "reshape or permute so TMEM columns stay contiguous" in msg
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_TWOCTA_HIGHER_RANK_POSITIVE_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_dim0_slice_reports_clean_error_lifted_layout(
    layout_name, n, variant, expected_shape
):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(CompilationError) as excinfo:
        tmem_ldst_descriptor_higher_rank_dim0_slice_positive_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
        )

    msg = str(excinfo.value)
    assert f"TMEM layout '{variant}' unsupported" in msg
    assert "reshape or permute so TMEM columns stay contiguous" in msg
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_TWOCTA_HIGHER_RANK_POSITIVE_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_half_rows_reports_clean_error_lifted_layout(
    layout_name, n, variant, expected_shape
):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(CompilationError) as excinfo:
        tmem_ldst_descriptor_higher_rank_half_rows_positive_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
        )

    msg = str(excinfo.value)
    assert f"TMEM layout '{variant}' unsupported" in msg
    assert "reshape or permute so TMEM columns stay contiguous" in msg
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant", LDST_TWOCTA_MMAV5_HIGHER_RANK_UNSUPPORTED_CASES)
def test_tmem_runtime_matrix_ldst_twocta_mmav5_descriptor_higher_rank_reports_clean_error(layout_name, n, variant,
                                                                                          capfd):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(Exception) as excinfo:
        tmem_ldst_descriptor_higher_rank_dim0_slice_positive_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "Result has an invalid layout" in text
    assert "Layout has 1 CTAs per CGA, but the context requires 2 CTAs per CGA" in text
    assert "twoCTAs requires a non-empty 'block' basis sequence" not in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,layout", BLOCKED_FALLBACK_CASES)
def test_tmem_runtime_matrix_blocked_fallback(layout_name, layout):
    m = n = 128
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_blocked_fallback_kernel[(1, )](inp, out, layout, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    llir = compiled.asm["llir"]
    ptx = compiled.asm["ptx"]
    assert "st.shared.v4.b32" in ptx
    assert "ld.shared.v4.b32" in ptx
    assert "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32" in llir
    assert "i32 16384" in llir

    expected_offsets = [
        ("tcgen05.st.sync.aligned.32x32b.x128.b32", 0),
        ("tcgen05.ld.sync.aligned.32x32b.x128.b32", 0),
    ]
    assert _extract_tcgen05_opcode_offsets(ptx) == expected_offsets
    assert _extract_tcgen05_opcode_offsets(llir) == expected_offsets


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("name,layout,reinterpret_layout", UNSUPPORTED_BLOCK_DESCRIPTOR_CASES)
def test_tmem_runtime_matrix_block_descriptor_reports_clean_error(name, layout, reinterpret_layout, capfd):
    reg_layout = _make_tmem_register_layout(2)
    with pytest.raises(CompilationError) as excinfo:
        tmem_block_descriptor_compile_kernel[(1, )](
            layout, reinterpret_layout, reg_layout, num_warps=4, num_ctas=2
        )
    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "unsupported tensor memory memdesc_subslice view" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n,splitn_x,offset_imm_pairs", M64_SPLITN_CASES)
def test_tmem_runtime_matrix_splitn_immediates(n, splitn_x, offset_imm_pairs):
    m = 64
    layout = _make_tmem_linear_layout_m64(n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, "32x32b_splitn", num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _, imms = _assert_ldst_ptx_llir_match(compiled)
    expected_opcode = f"tcgen05.st.sync.aligned.16x32bx2.x{splitn_x}.b32"
    expected_opcode_ld = f"tcgen05.ld.sync.aligned.16x32bx2.x{splitn_x}.b32"

    expected = []
    for offset, imm in offset_imm_pairs:
        expected.append((expected_opcode, offset, imm))
    for offset, imm in offset_imm_pairs:
        expected.append((expected_opcode_ld, offset, imm))
    assert imms == expected


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n,splitn_x,expected_imm", M64_SPLITN_AUTO_CASES)
def test_tmem_runtime_matrix_splitn_auto_selects_16x32bx2(n, splitn_x, expected_imm):
    m = 64
    layout = _make_tmem_linear_layout_m64(n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, imms = _assert_ldst_ptx_llir_match(compiled)
    expected_opcode = f"tcgen05.st.sync.aligned.16x32bx2.x{splitn_x}.b32"
    expected_opcode_ld = f"tcgen05.ld.sync.aligned.16x32bx2.x{splitn_x}.b32"

    observed_opcodes = [op for op, _ in ops]
    assert all(op in (expected_opcode, expected_opcode_ld) for op in observed_opcodes)
    assert expected_opcode in observed_opcodes
    assert expected_opcode_ld in observed_opcodes

    assert imms == [
        (expected_opcode, 0, expected_imm),
        (expected_opcode_ld, 0, expected_imm),
    ]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("variant", ("32x32b", "16x64b", "16x128b", "16x256b"))
def test_tmem_runtime_matrix_ldst_fixed_offset_patterns_128x256(variant):
    m, n = 128, 256
    layout = _make_tmem_linear_layout(m, n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ptx_offsets = _extract_tcgen05_opcode_offsets(compiled.asm["ptx"])
    llir_offsets = _extract_tcgen05_opcode_offsets(compiled.asm["llir"])
    assert ptx_offsets == llir_offsets
    assert ptx_offsets == LDST_EXPECTED_OFFSETS_128x256[variant]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", F16_LDST_CASES)
def test_tmem_runtime_matrix_ldst_subword_f16_pack_unpack(layout_name, n, variant, expected_shape):
    m = 128
    layout = LDST_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.float16, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_f16_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert all(op in (expected_st, expected_ld) for op in observed_opcodes)
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_DESCRIPTOR_RANK5_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_rank5_roundtrip(layout_name, n, variant, expected_shape):
    m = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2, 2, 2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    try:
        compiled = tmem_ldst_descriptor_rank5_roundtrip_kernel[(1, )](
            inp, out, layout, m, n, variant, 17.0, num_warps=4
        )
    except triton.runtime.errors.OutOfResources:
        pytest.skip(f"tensor memory OOR for rank5 layout={layout_name}, n={n}, variant={variant}")
    torch.testing.assert_close(out, inp + 17.0, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes

    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttg.memdesc_reshape" in ttgir
    assert "ttg.memdesc_trans" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_TWOCTA_DESCRIPTOR_RANK5_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_rank5_roundtrip(layout_name, n, variant, expected_shape):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2, 2, 2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    try:
        compiled = tmem_ldst_descriptor_rank5_roundtrip_kernel[(1, )](
            inp, out, layout, m, n, variant, 19.0, num_warps=4, num_ctas=2
        )
    except triton.runtime.errors.OutOfResources:
        pytest.skip(f"tensor memory OOR for rank5 twocta layout={layout_name}, n={n}, variant={variant}")
    torch.testing.assert_close(out, inp + 19.0, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes

    ttgir = compiled.asm["ttgir"]
    assert "twoCTAs = true" in ttgir
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttg.memdesc_reshape" in ttgir
    assert "ttg.memdesc_trans" in ttgir


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize("use_abs,propagate_nan", [(False, tl.PropagateNan.NONE), (True, tl.PropagateNan.ALL)])
@pytest.mark.parametrize("layout_name,M,N,num_warps,expected_shape", LD_RED_LINEAR_CASES)
def test_tmem_runtime_matrix_ld_red_identity_linear_layout(red_op, use_abs, propagate_nan, layout_name, M, N, num_warps,
                                                           expected_shape):
    if layout_name == "identity":
        layout = _make_tmem_linear_layout(M, N)
    else:
        layout = _make_tmem_linear_layout_mixed(M, N)
    compiled = _run_tmem_reduction_case(
        layout,
        M,
        N,
        red_op,
        use_abs,
        propagate_nan,
        num_warps=num_warps,
    )
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir

    ptx_red_ops = [
        op for op, _ in _extract_tcgen05_opcode_offsets(compiled.asm["ptx"], opcodes=("ld", )) if ".ld.red." in op
    ]
    llir_red_ops = [
        op for op, _ in _extract_tcgen05_opcode_offsets(compiled.asm["llir"], opcodes=("ld", )) if ".ld.red." in op
    ]
    assert ptx_red_ops == llir_red_ops
    assert len(ptx_red_ops) == LD_RED_EXPECTED_OP_COUNT[N]
    expected_prefix = f"tcgen05.ld.red.sync.aligned.{expected_shape}.{red_op}"
    assert all(op.startswith(expected_prefix) for op in ptx_red_ops)
    assert all(op.endswith(".f32") for op in ptx_red_ops)
    if use_abs:
        assert all(".abs." in op for op in ptx_red_ops)
    else:
        assert all(".abs." not in op for op in ptx_red_ops)
    if propagate_nan == tl.PropagateNan.ALL:
        assert all(".NaN." in op for op in ptx_red_ops)
    else:
        assert all(".NaN." not in op for op in ptx_red_ops)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize("use_abs,propagate_nan", [(False, tl.PropagateNan.NONE), (True, tl.PropagateNan.ALL)])
@pytest.mark.parametrize("layout_name,M,N,num_warps", LD_RED_MIXED_UNSUPPORTED_CASES)
def test_tmem_runtime_matrix_ld_red_mixed_layout_reports_clean_unsupported(
    red_op, use_abs, propagate_nan, layout_name, M, N, num_warps, capfd
):
    layout = _make_tmem_linear_layout_mixed(M, N)
    with pytest.raises(Exception) as excinfo:
        _run_tmem_reduction_case(
            layout,
            M,
            N,
            red_op,
            use_abs,
            propagate_nan,
            num_warps=num_warps,
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "tmem_load reduction with N dimension sharded across threads is not supported" in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("name,smem_layout,expected_status", CP_SCALES_LAYOUT_PROBE_CASES)
def test_tmem_runtime_matrix_cp_scales_layout_probe(name, smem_layout, expected_status, capfd):
    smem_h, smem_w = 64, 16
    num_rows = 128
    num_cols = smem_h * smem_w // 32
    inp = torch.randint(size=(smem_h, smem_w), low=-100, high=100, dtype=torch.int8, device="cuda")
    out = torch.zeros(size=(num_rows, num_cols), dtype=torch.int8, device="cuda")

    if expected_status == "PASS":
        compiled = tmem_copy_scales_layout_probe_kernel[(1, )](inp, out, smem_layout)
        expected = inp.reshape(2, 32, 2, 2, 4).permute(1, 2, 3, 0, 4).reshape(num_rows // 4, num_cols)
        for warp in torch.chunk(out, chunks=4, dim=0):
            torch.testing.assert_close(expected, warp, atol=0, rtol=0)
        ptx_ops = _extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
        llir_ops = _extract_tcgen05_cp_opcodes(compiled.asm["llir"])
        assert ptx_ops == llir_ops
        assert ptx_ops == ["tcgen05.cp.cta_group::1.warpx4.32x128b"] * 2
        return

    with pytest.raises(Exception) as excinfo:
        tmem_copy_scales_layout_probe_kernel[(1, )](inp, out, smem_layout)
    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "failed to find valid tcgen05.copy layout" in text

    if expected_status == "CLEAN_UNSUPPORTED":
        assert "PassManager::run failed" not in text
        assert "Assertion" not in text
        return

    assert expected_status == "BUG"
    assert "PassManager::run failed" in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("M,N,BLOCK_N,swizzle", CP_NO_SCALES_CASES)
def test_tmem_runtime_matrix_cp_no_scales(M, N, BLOCK_N, swizzle):
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N)
    out = torch.empty_like(inp)

    try:
        compiled = tmem_copy_no_scales_kernel[(1, )](inp, out, M, N, BLOCK_N, swizzle, num_warps=4)
    except triton.runtime.errors.OutOfResources:
        pytest.skip(f"shared memory OOR for M={M}, N={N}, BLOCK_N={BLOCK_N}, swizzle={swizzle}")
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ptx_ops = _extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_cp_opcodes(compiled.asm["llir"])
    assert ptx_ops
    assert ptx_ops == llir_ops
    assert all(op.endswith(".128x256b") for op in ptx_ops)
    assert len(ptx_ops) == (M * N) // 1024


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("M,N,BLOCK_N,swizzle", CP_NO_SCALES_SWIZZLE_CASES)
def test_tmem_runtime_matrix_cp_no_scales_swizzles(M, N, BLOCK_N, swizzle):
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N)
    out = torch.empty_like(inp)

    try:
        compiled = tmem_copy_no_scales_kernel[(1, )](inp, out, M, N, BLOCK_N, swizzle, num_warps=4)
    except triton.runtime.errors.OutOfResources:
        pytest.skip(f"shared memory OOR for M={M}, N={N}, BLOCK_N={BLOCK_N}, swizzle={swizzle}")
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ptx_ops = _extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_cp_opcodes(compiled.asm["llir"])
    expected_count = (M * N) // 1024
    assert ptx_ops == llir_ops
    assert ptx_ops == ["tcgen05.cp.cta_group::1.128x256b"] * expected_count


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("M,N,swizzle,expected_count", CP_LINEAR_NO_SCALES_CASES)
def test_tmem_runtime_matrix_cp_no_scales_linear(M, N, swizzle, expected_count):
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout(M, N)

    compiled = tmem_copy_no_scales_linear_kernel[(1, )](inp, out, layout, M, N, swizzle, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ptx_ops = _extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_cp_opcodes(compiled.asm["llir"])
    assert ptx_ops == llir_ops
    assert ptx_ops == ["tcgen05.cp.cta_group::1.128x256b"] * expected_count
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_linear_unsupported_shape_reports_clean_error():
    m, n, swizzle = 256, 128, 32
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout(m, n)

    with pytest.raises(CompilationError) as excinfo:
        tmem_copy_no_scales_linear_kernel[(1, )](inp, out, layout, m, n, swizzle, num_warps=4)

    msg = str(excinfo.value)
    assert "TMEM layout '32x32b' unsupported" in msg
    assert "reshape or permute so TMEM columns stay contiguous" in msg
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("M", (128, ))
def test_tmem_runtime_matrix_cp_128x128(M):
    N = 4
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_copy_128x128_kernel[(1, )](inp, out, M, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ptx_ops = _extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_cp_opcodes(compiled.asm["llir"])
    assert ptx_ops == llir_ops
    assert all(op == "tcgen05.cp.cta_group::1.128x128b" for op in ptx_ops)
    assert len(ptx_ops) == M // 128


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_scales_warpx4():
    smem_h, smem_w = 64, 16
    num_rows = 128
    num_cols = smem_h * smem_w // 32
    inp = torch.randint(size=(smem_h, smem_w), low=-100, high=100, dtype=torch.int8, device="cuda")
    out = torch.zeros(size=(num_rows, num_cols), dtype=torch.int8, device="cuda")

    compiled = tmem_copy_scales_warpx4_kernel[(1, )](inp, out)

    expected = inp.reshape(2, 32, 2, 2, 4).permute(1, 2, 3, 0, 4).reshape(num_rows // 4, num_cols)
    for warp in torch.chunk(out, chunks=4, dim=0):
        torch.testing.assert_close(expected, warp, atol=0, rtol=0)

    ptx_ops = _extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_cp_opcodes(compiled.asm["llir"])
    expected_ops = ["tcgen05.cp.cta_group::1.warpx4.32x128b"] * 2
    assert ptx_ops == expected_ops
    assert llir_ops == expected_ops


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_scales_warpx4_twocta_via_scaled_mma_copy():
    m, n, k = 256, 128, 128
    block_m, block_n, block_k = 256, 128, 128
    vec_size = 32

    torch.manual_seed(0)
    a, a_scale, _ = random_quantized_tensor(m, k, "mxfp8")
    b, b_scale, _ = random_quantized_tensor(n, k, "mxfp8")
    a_scale = swizzle_scales_packed_block(a_scale, vec_size)
    b_scale = swizzle_scales_packed_block(b_scale, vec_size)

    _, compiled = mma_scaled_tcgen05_copy(
        a,
        b,
        a_scale,
        b_scale,
        vec_size,
        block_m,
        block_n,
        block_k,
        num_ctas=2,
        multicast=False,
    )

    ptx_ops = _extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_cp_opcodes(compiled.asm["llir"])
    assert ptx_ops
    assert ptx_ops == llir_ops
    expected = _expected_scaled_cp_opcode(2)
    assert all(op == expected for op in ptx_ops)


MMA_CASES = [
    ("legacy_no_acc", TensorMemoryLayout((128, 128), col_stride=1), False),
    ("legacy_use_acc", TensorMemoryLayout((128, 128), col_stride=1), True),
    ("linear_no_acc", _make_tmem_linear_layout(128, 128), False),
    ("linear_use_acc", _make_tmem_linear_layout(128, 128), True),
]

MMA_TWOCTA_CASES = [
    ("legacy", "legacy"),
    ("linear", "linear"),
]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("name,layout,use_acc", MMA_CASES)
def test_tmem_runtime_matrix_mma(name, layout, use_acc):
    m, n, k = 128, 128, 32
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    c = torch.randn((m, n), dtype=torch.float32, device="cuda")
    out = torch.empty_like(c)

    compiled = tmem_mma_kernel[(1, )](a, b, c, out, layout, use_acc, num_warps=4)
    expected = a.to(torch.float32) @ b.to(torch.float32)
    if use_acc:
        expected = expected + c
    torch.testing.assert_close(out, expected, atol=1e-1, rtol=8e-2)

    assert "tcgen05.mma.cta_group::1.kind::f16" in compiled.asm["ptx"]
    assert "tcgen05.mma.cta_group::1.kind::f16" in compiled.asm["llir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("name,layout_kind", MMA_TWOCTA_CASES)
def test_tmem_runtime_matrix_mma_twocta(name, layout_kind):
    ctas_per_cga = [2, 1]
    ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    block_m = 128 * ctas_per_cga[0]
    block_n = 64 * ctas_per_cga_b[1]
    block_k = 32

    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]
    cta_order = [1, 0]
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)

    shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([block_m, block_k], ttgl.float16, cga_layout=cga_layout_a)
    shared_layout_b = ttgl.NVMMASharedLayout.get_default_for([block_k, block_n], ttgl.float16, cga_layout=cga_layout_b)

    a = torch.randn((block_m, block_k), dtype=torch.float16, device="cuda")
    b = torch.randn((block_k, block_n), dtype=torch.float16, device="cuda")
    out = torch.empty((block_m, block_n), dtype=torch.float32, device="cuda")

    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(a, [block_m, block_k], shared_layout_a)
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(b, [block_k, block_n], shared_layout_b)

    if layout_kind == "legacy":
        acc_tmem_layout = TensorMemoryLayout(
            block=(128, block_n // ctas_per_cga[1]),
            col_stride=1,
            two_ctas=True,
            cga_layout=cga_layout_c,
        )
    else:
        acc_tmem_layout = _make_tmem_linear_layout_mmav5_twocta(block_m, block_n)

    blocked_c = ttgl.BlockedLayout([1, 2], [ctas_per_cga[1], 32 // ctas_per_cga[1]], [4, 1], [1, 0],
                                   cga_layout=cga_layout_c)
    compiled = tmem_mma_twocta_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        block_m,
        block_n,
        acc_tmem_layout,
        blocked_c,
        num_warps=4,
        num_ctas=2,
    )

    torch.testing.assert_close(out, torch.matmul(a.to(torch.float32), b.to(torch.float32)), atol=1e-1, rtol=8e-2)

    ptx_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_mma_ops == llir_mma_ops
    assert ptx_mma_ops
    assert all(op == "tcgen05.mma.cta_group::2.kind::f16" for op in ptx_mma_ops)
    assert "tcgen05.commit.cta_group::2" in compiled.asm["ptx"]
    assert "tcgen05.commit.cta_group::2" in compiled.asm["llir"]
    assert ".multicast::cluster" in compiled.asm["ptx"]


def _fp8e8m0_to_float32(scale):
    scale = scale.view(torch.uint8)
    scale = scale.to(torch.int32)
    scale = scale << 23
    return scale.view(torch.float32)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_scaled_minimal():
    m = n = k = 128
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")
    a = torch.randint(20, 40, (m, k), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    b = torch.randint(20, 40, (k, n), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    a_scale = torch.randint(64, 130, (m, k // 32), dtype=torch.uint8, device="cuda")
    b_scale = torch.randint(64, 130, (n, k // 32), dtype=torch.uint8, device="cuda")

    compiled = tmem_mma_scaled_minimal_kernel[(1, )](out, m, n, k, a, b, a_scale, b_scale, num_warps=4)

    a_ref = a.to(torch.float32) * _fp8e8m0_to_float32(a_scale).repeat_interleave(32, dim=1)
    b_ref = b.to(torch.float32) * _fp8e8m0_to_float32(b_scale).repeat_interleave(32, dim=1).T.contiguous()
    expected = torch.matmul(a_ref, b_ref)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    ptx_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_mma_ops == llir_mma_ops
    assert ptx_mma_ops
    assert all(op.startswith("tcgen05.mma.cta_group::1.kind::") for op in ptx_mma_ops)
    assert all("block_scale.scale_vec::" in op for op in ptx_mma_ops)
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]
