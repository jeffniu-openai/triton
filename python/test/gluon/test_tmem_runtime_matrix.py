import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
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
    fence_async_shared,
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
    mma_kernel,
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


def _permute_pow2_bases_by_kind(bits, kind):
    if kind == "identity":
        return list(bits)
    if kind == "rotate1":
        return list(bits[1:]) + [bits[0]]
    if kind == "even_odd":
        return list(bits[::2]) + list(bits[1::2])
    if kind == "reverse":
        return list(reversed(bits))
    raise ValueError(f"unsupported permutation kind: {kind}")


def _make_tmem_linear_layout_permuted(m, n, row_perm_kind, col_perm_kind):
    row_bits = _permute_pow2_bases_by_kind([1 << i for i in range(int(math.log2(m)))], row_perm_kind)
    col_bits = _permute_pow2_bases_by_kind([1 << i for i in range(int(math.log2(n)))], col_perm_kind)
    return TensorMemoryLinearLayout(
        rows=[[b, 0] for b in row_bits],
        cols=[[0, b] for b in col_bits],
        shape=[m, n],
    )


def _make_tmem_linear_layout_tile_permuted(m, n, tile_n):
    assert m >= 1 and (m & (m - 1)) == 0
    assert n >= 1 and (n & (n - 1)) == 0
    assert tile_n >= 1 and (tile_n & (tile_n - 1)) == 0
    assert n >= 2 * tile_n, "tile permutation requires at least two tiles"

    col_bits = [1 << i for i in range(int(math.log2(n)))]
    tile_idx = int(math.log2(tile_n))
    col_bits[tile_idx], col_bits[tile_idx + 1] = col_bits[tile_idx + 1], col_bits[tile_idx]
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(int(math.log2(m)))],
        cols=[[0, bit] for bit in col_bits],
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


def _make_tmem_linear_layout_m64_permuted(n, row_perm_kind, col_perm_kind):
    row_bits = _permute_pow2_bases_by_kind([1, 2, 4, 8, 16, 32], row_perm_kind)
    col_bits = _permute_pow2_bases_by_kind([1 << i for i in range(int(math.log2(n)))], col_perm_kind)
    return TensorMemoryLinearLayout(
        rows=[[row_bits[0], 0], [row_bits[1], 0], [row_bits[2], 0], [row_bits[3], 0], [0, 0], [row_bits[4], 0],
              [row_bits[5], 0]],
        cols=[[0, bit] for bit in col_bits],
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


def _make_tmem_linear_layout_ld_red_legacy_equivalent_256x128():
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(7)],
        cols=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]],
        shape=[256, 128],
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


def _round_to_tf32(x: torch.Tensor) -> torch.Tensor:
    x = x.view(torch.int32)
    x = x & ~((1 << 13) - 1)
    return x.view(torch.float32)


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
    # Historical warpx2 probe candidate. Through the public descriptor API this
    # currently reaches a clean unsupported descriptor-plan path rather than a
    # live tcgen05.copy.warpx2 lowering.
    return ttgl.SharedLinearLayout(
        offset_bases=[[32, 0], [0, 1], [1, 0], [0, 2], [0, 4], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]]
    )


def _make_tmem_copy_warpx2_shared_layout():
    return ttgl.SharedLinearLayout(
        offset_bases=[[32, 0], [0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [64, 0]],
        alignment=16,
    )


def _make_tmem_copy_warpx2_tmem_layout():
    return TensorMemoryLinearLayout(
        rows=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0], [32, 0]],
        cols=[[0, 1], [0, 2]],
        shape=[128, 4],
    )


def _make_tmem_copy_warpx2_tmem_layout_02_13():
    return TensorMemoryLinearLayout(
        rows=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 0]],
        cols=[[0, 1], [0, 2]],
        shape=[128, 4],
    )


def _expected_tmem_copy_warpx2_01_23_output(inp: torch.Tensor) -> torch.Tensor:
    assert tuple(inp.shape) == (128, 4)
    expected = torch.empty_like(inp)
    for row in range(inp.shape[0]):
        src_base = ((row % 32) // 2) + 16 * (row // 64)
        src_cols = (0, 1) if row % 2 == 0 else (2, 3)
        expected[row, 0] = inp[src_base, src_cols[0]]
        expected[row, 1] = inp[src_base + 32, src_cols[0]]
        expected[row, 2] = inp[src_base, src_cols[1]]
        expected[row, 3] = inp[src_base + 32, src_cols[1]]
    return expected


def _expected_tmem_copy_warpx2_02_13_output(inp: torch.Tensor) -> torch.Tensor:
    assert tuple(inp.shape) == (128, 4)
    expected = torch.empty_like(inp)
    for row in range(inp.shape[0]):
        src_base = ((row % 32) // 2) + 48 * ((row % 64) // 32) + 16
        src_cols = (0, 1) if row % 2 == 0 else (2, 3)
        expected[row, 0] = inp[src_base, src_cols[0]]
        expected[row, 1] = inp[src_base + 32, src_cols[0]]
        expected[row, 2] = inp[src_base, src_cols[1]]
        expected[row, 3] = inp[src_base + 32, src_cols[1]]
    return expected


def _make_tmem_copy_128x128_shared_layout():
    return ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]],
        alignment=16,
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


def _assert_exact_cp_ptx_llir_match(compiled, expected_ops=None):
    ptx_ops = _extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_cp_opcodes(compiled.asm["llir"])
    assert ptx_ops == llir_ops
    if expected_ops is not None:
        assert ptx_ops == list(expected_ops)
    return ptx_ops


def _assert_clean_unsupported_descriptor_view(text: str, expected_text: str):
    assert expected_text in text
    assert "reshape or permute so TMEM columns stay contiguous" not in text


def _assert_clean_cta_per_cga_mismatch(text: str, layout_ctas: int, required_ctas: int):
    assert "Result has an invalid layout:" in text
    assert f"Layout has {layout_ctas} CTAs per CGA, but the context requires {required_ctas} CTAs per CGA." in text


def _assert_clean_cta_per_cga_mismatch_or_descriptor_unsupported(text: str, layout_ctas: int, required_ctas: int):
    if "Result has an invalid layout:" in text:
        _assert_clean_cta_per_cga_mismatch(text, layout_ctas, required_ctas)
        return
    assert "TMEM layout" in text
    assert "unsupported for descriptor view" in text


def _assert_clean_tmem_oor(text: str, required: int, hardware_limit: int):
    assert "out of resource: tensor memory" in text
    assert f"Required: {required}" in text
    assert f"Hardware limit: {hardware_limit}" in text


def _extract_tcgen05_mma_opcodes(asm: str):
    pattern = re.compile(r"(tcgen05\.mma\.cta_group::\d+\.kind::[^\s;\"]+)")
    return pattern.findall(asm)


def _make_mma_plain_kind_inputs(kind: str, m: int, n: int, k: int):
    if kind == "tf32":
        a = _round_to_tf32(torch.randn((m, k), device="cuda", dtype=torch.float32))
        b = _round_to_tf32(torch.randn((k, n), device="cuda", dtype=torch.float32))
        shared_layout_a = ttgl.NVMMASharedLayout(swizzle_byte_width=128, transposed=False, element_bitwidth=32, rank=2)
        shared_layout_b = ttgl.NVMMASharedLayout(swizzle_byte_width=128, transposed=True, element_bitwidth=32, rank=2)
        expected_kind = "tcgen05.mma.cta_group::1.kind::tf32"
        atol, rtol = 5e-4, 5e-3
    elif kind == "bf16":
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
        shared_layout_a = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=False, element_bitwidth=16, rank=2)
        shared_layout_b = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=True, element_bitwidth=16, rank=2)
        expected_kind = "tcgen05.mma.cta_group::1.kind::f16"
        atol, rtol = 1e-1, 1e-1
    else:
        fp8_dtype = torch.float8_e5m2 if kind == "f8e5m2" else torch.float8_e4m3fn
        a = torch.randint(20, 40, (m, k), device="cuda", dtype=torch.uint8).view(fp8_dtype)
        b = torch.randint(20, 40, (k, n), device="cuda", dtype=torch.uint8).view(fp8_dtype)
        shared_layout_a = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=False, element_bitwidth=8, rank=2)
        shared_layout_b = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=True, element_bitwidth=8, rank=2)
        expected_kind = "tcgen05.mma.cta_group::1.kind::f8f6f4"
        atol, rtol = 1e-1, 1e-1
    return a, b, shared_layout_a, shared_layout_b, expected_kind, atol, rtol


def _make_mma_twocta_plain_kind_inputs(kind: str, m: int, n: int, k: int, cga_layout_a, cga_layout_b):
    if kind == "tf32":
        a = _round_to_tf32(torch.randn((m, k), device="cuda", dtype=torch.float32))
        b = _round_to_tf32(torch.randn((k, n), device="cuda", dtype=torch.float32))
        shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([m, k], ttgl.float32, cga_layout=cga_layout_a)
        shared_layout_b = ttgl.NVMMASharedLayout.get_default_for(
            [k, n], ttgl.float32, transposed=True, cga_layout=cga_layout_b
        )
        expected_kind = "tcgen05.mma.cta_group::2.kind::tf32"
        atol, rtol = 5e-4, 5e-3
    elif kind == "bf16":
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
        shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([m, k], ttgl.bfloat16, cga_layout=cga_layout_a)
        shared_layout_b = ttgl.NVMMASharedLayout.get_default_for(
            [k, n], ttgl.bfloat16, transposed=True, cga_layout=cga_layout_b
        )
        expected_kind = "tcgen05.mma.cta_group::2.kind::f16"
        atol, rtol = 1e-1, 1e-1
    else:
        fp8_dtype = torch.float8_e5m2 if kind == "f8e5m2" else torch.float8_e4m3fn
        gluon_dtype = ttgl.float8e5 if kind == "f8e5m2" else ttgl.float8e4nv
        a = torch.randint(20, 40, (m, k), device="cuda", dtype=torch.uint8).view(fp8_dtype)
        b = torch.randint(20, 40, (k, n), device="cuda", dtype=torch.uint8).view(fp8_dtype)
        shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([m, k], gluon_dtype, cga_layout=cga_layout_a)
        shared_layout_b = ttgl.NVMMASharedLayout.get_default_for(
            [k, n], gluon_dtype, transposed=True, cga_layout=cga_layout_b
        )
        expected_kind = "tcgen05.mma.cta_group::2.kind::f8f6f4"
        atol, rtol = 1e-1, 1e-1
    return a, b, shared_layout_a, shared_layout_b, expected_kind, atol, rtol


def _assert_ldst_ptx_llir_match(compiled):
    ptx_ops = _extract_tcgen05_opcode_offsets(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_opcode_offsets(compiled.asm["llir"])
    assert ptx_ops, "No tcgen05 ld/st opcodes in PTX"
    assert ptx_ops == llir_ops

    ptx_imms = _extract_tcgen05_opcode_offset_immediates(compiled.asm["ptx"])
    llir_imms = _extract_tcgen05_opcode_offset_immediates(compiled.asm["llir"])
    assert ptx_imms == llir_imms
    return ptx_ops, ptx_imms


def _expected_ldst_ops(op_shape: str, offsets):
    ops = []
    for offset in offsets:
        ops.append((f"tcgen05.st.sync.aligned.{op_shape}", offset))
    for offset in offsets:
        ops.append((f"tcgen05.ld.sync.aligned.{op_shape}", offset))
    return ops


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
    view = tmem.slice(1, 1, dim=0).index(0).reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
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
    part0_layout: ttgl.constexpr = part0.get_reg_layout(instr_variant=instr_variant)
    part1_layout: ttgl.constexpr = part1.get_reg_layout(instr_variant=instr_variant)

    part0_value = part0.load(part0_layout)
    part0.store(part0_value + ttgl.full([M, N // 2], 9.0, ttgl.float32, layout=part0_layout))
    part1_value = part1.load(part1_layout)
    part1.store(part1_value + ttgl.full([M, N // 2], 9.0, ttgl.float32, layout=part1_layout))

    out = full_view.load(full_reg_layout)
    ttgl.store(out_ptr + in_offs, out)


@gluon.jit
def tmem_ldst_descriptor_multidim_slice_positive_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr,
                                                       N: ttgl.constexpr, instr_variant: ttgl.constexpr,
                                                       num_warps: ttgl.constexpr):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    tmem = allocate_tensor_memory(ttgl.float32, [M, N], layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant=instr_variant)
    value = ttgl.load(in_ptr + offs)
    tmem.store(ttgl.convert_layout(value, reg_layout))

    view = tmem.reshape((2, M // 2, 2, N // 2))
    view = view.slice(1, 1, dim=0)
    view = view.slice(0, M // 4, dim=1)
    view = view.slice(1, 1, dim=2)
    view = view.slice(0, N // 4, dim=3)
    view = view.reshape((M // 4, N // 4))

    sub_layout: ttgl.constexpr = view.get_reg_layout()
    sub_value = view.load(sub_layout)
    view.store(sub_value + ttgl.full([M // 4, N // 4], 11.0, ttgl.float32, layout=sub_layout))

    result = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(result, reg_layout))


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
def tmem_scales_ldst_kernel(in_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr, cga_layout: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.int8, [M, N], TensorMemoryScalesLayout(cga_layout=list(cga_layout)))
    reg_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant="32x32b")
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))[:, None]
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))[None, :]
    offs = offs_m * N + offs_n
    value = ttgl.load(in_ptr + offs)
    tmem.store(ttgl.convert_layout(value, reg_layout))
    value = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, value)


@gluon.jit
def tmem_scales_ldst_variant_kernel(in_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr,
                                    instr_variant: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.int8, [M, N], TensorMemoryScalesLayout())
    reg_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant=instr_variant)
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))[:, None]
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))[None, :]
    offs = offs_m * N + offs_n
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
    fence_async_shared()

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
    fence_async_shared(cluster=layout.two_ctas)

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)

    out = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, out)


@gluon.jit
def tmem_copy_no_scales_indexed_view_kernel(in_ptr, out_ptr, M: ttgl.constexpr):
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
    tmem = allocate_tensor_memory(ttgl.int32, [2, M, N], layout=TensorMemoryLayout((M, N), col_stride=1))
    view = tmem.index(1)

    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    smem.store(value)
    fence_async_shared()
    tcgen05_copy(smem, view)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    reg_layout: ttgl.constexpr = view.get_reg_layout()
    output = view.load(reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(output, blocked))


@gluon.jit
def tmem_copy_no_scales_linear_indexed_view_kernel(in_ptr, out_ptr, lifted_layout: ttgl.constexpr,
                                                   smem_layout: ttgl.constexpr, M: ttgl.constexpr,
                                                   N: ttgl.constexpr):
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [2, M, N], layout=lifted_layout)
    view = tmem.index(1)
    reg_layout: ttgl.constexpr = view.get_reg_layout()

    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)

    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    smem.store(value)
    fence_async_shared()
    tcgen05_copy(smem, view)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    output = view.load(reg_layout)
    ttgl.store(out_ptr + offs, output)


@gluon.jit
def tmem_copy_no_scales_linear_subslice_view_kernel(in_ptr, out_ptr, parent_layout: ttgl.constexpr,
                                                    M: ttgl.constexpr, N: ttgl.constexpr,
                                                    swizzle: ttgl.constexpr):
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, 2 * N], layout=parent_layout)
    view = tmem.slice(N, N, dim=1)
    reg_layout: ttgl.constexpr = view.get_reg_layout()

    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
        swizzle_byte_width=swizzle, element_bitwidth=32, rank=2
    )
    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)

    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    smem.store(value)
    fence_async_shared()
    tcgen05_copy(smem, view)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    output = view.load(reg_layout)
    ttgl.store(out_ptr + offs, output)


@gluon.jit
def tmem_mma_indexed_acc_kernel(a_ptr, b_ptr, out_ptr, parent_layout: ttgl.constexpr):
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

    smem_a_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16, rank=2)
    smem_b_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16, rank=2)
    smem_a = ttgl.allocate_shared_memory(ttgl.float16, [M, K], layout=smem_a_layout)
    smem_b = ttgl.allocate_shared_memory(ttgl.float16, [K, N], layout=smem_b_layout)
    smem_a.store(a)
    smem_b.store(b)

    acc_parent = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
    acc_tmem = acc_parent.index(1)

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=tcgen05_mma_barrier_count([smem_a, smem_b], False))
    tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=False, mbarriers=[bar])
    mbarrier.wait(bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(bar)

    out = acc_tmem.load()
    ttgl.store(out_ptr + c_offs, out)


@gluon.jit
def tmem_mma_lhs_kernel(a_ptr, b_ptr, out_ptr, lhs_layout: ttgl.constexpr, acc_layout: ttgl.constexpr):
    M: ttgl.constexpr = 128
    N: ttgl.constexpr = 128
    K: ttgl.constexpr = 256
    a_offs = ttgl.arange(0, M)[:, None] * K + ttgl.arange(0, K)[None, :]
    b_offs = ttgl.arange(0, K)[:, None] * N + ttgl.arange(0, N)[None, :]
    out_offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]

    blocked_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    blocked_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    a = ttgl.load(ttgl.set_auto_layout(a_ptr + a_offs, blocked_a))
    b = ttgl.load(ttgl.set_auto_layout(b_ptr + b_offs, blocked_b))

    lhs_tmem = allocate_tensor_memory(ttgl.float16, [M, K], lhs_layout)
    lhs_reg_layout: ttgl.constexpr = lhs_tmem.get_reg_layout()
    lhs_tmem.store(ttgl.convert_layout(a, lhs_reg_layout))

    smem_b_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=True, element_bitwidth=16,
                                                           rank=2)
    smem_b = ttgl.allocate_shared_memory(ttgl.float16, [K, N], layout=smem_b_layout)
    smem_b.store(b)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma(lhs_tmem, smem_b, acc_tmem, use_acc=False)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out = acc_tmem.load()
    ttgl.store(out_ptr + out_offs, out)


@gluon.jit
def tmem_mma_lhs_subslice_kernel(a_ptr, b_ptr, out_ptr, parent_layout: ttgl.constexpr, acc_layout: ttgl.constexpr):
    M: ttgl.constexpr = 128
    N: ttgl.constexpr = 128
    K: ttgl.constexpr = 32
    PARENT_K: ttgl.constexpr = 2 * K
    a_offs = ttgl.arange(0, M)[:, None] * K + ttgl.arange(0, K)[None, :]
    b_offs = ttgl.arange(0, K)[:, None] * N + ttgl.arange(0, N)[None, :]
    out_offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]

    blocked_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    blocked_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    a = ttgl.load(ttgl.set_auto_layout(a_ptr + a_offs, blocked_a))
    b = ttgl.load(ttgl.set_auto_layout(b_ptr + b_offs, blocked_b))

    lhs_parent = allocate_tensor_memory(ttgl.float16, [M, PARENT_K], parent_layout)
    lhs_tmem = lhs_parent.slice(K, K, dim=1)
    lhs_reg_layout: ttgl.constexpr = lhs_tmem.get_reg_layout()
    lhs_tmem.store(ttgl.convert_layout(a, lhs_reg_layout))

    smem_b_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
        swizzle_byte_width=32, transposed=True, element_bitwidth=16, rank=2
    )
    smem_b = ttgl.allocate_shared_memory(ttgl.float16, [K, N], layout=smem_b_layout)
    smem_b.store(b)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma(lhs_tmem, smem_b, acc_tmem, use_acc=False)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out = acc_tmem.load()
    ttgl.store(out_ptr + out_offs, out)


@gluon.jit
def tmem_copy_no_scales_warpx2_candidate_kernel(in_ptr, out_ptr, shared_layout: ttgl.constexpr, tmem_layout: ttgl.constexpr):
    M: ttgl.constexpr = 128
    N: ttgl.constexpr = 4
    shared_reg_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
        warp_bases=[[32, 0], [64, 0]],
        block_bases=[],
        shape=[M, N],
    )
    in_offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, shared_reg_layout))
    in_offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, shared_reg_layout))
    in_offs = in_offs_m[:, None] * N + in_offs_n[None, :]
    value = ttgl.load(in_ptr + in_offs)
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=tmem_layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=shared_layout, value=value)
    fence_async_shared()

    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    out = tmem.load(reg_layout)
    out_offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))
    out_offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))
    out_offs = out_offs_m[:, None] * N + out_offs_n[None, :]
    ttgl.store(out_ptr + out_offs, out)


@gluon.jit
def tmem_copy_no_scales_warpx2_codegen_kernel(in_ptr, out_ptr, shared_layout: ttgl.constexpr, tmem_layout: ttgl.constexpr):
    M: ttgl.constexpr = 128
    N: ttgl.constexpr = 4
    shared_reg_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
        warp_bases=[[32, 0], [64, 0]],
        block_bases=[],
        shape=[M, N],
    )
    in_offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, shared_reg_layout))
    in_offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, shared_reg_layout))
    in_offs = in_offs_m[:, None] * N + in_offs_n[None, :]
    value = ttgl.load(in_ptr + in_offs)
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=tmem_layout)

    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=shared_layout, value=value)
    fence_async_shared()

    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)
    mbarrier.invalidate(barrier)

    ttgl.store(out_ptr, 0)


@gluon.jit
def tmem_copy_no_scales_twocta_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, cga_layout: ttgl.constexpr,
                                      M: ttgl.constexpr, N: ttgl.constexpr, swizzle: ttgl.constexpr):
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
        swizzle_byte_width=swizzle,
        element_bitwidth=32,
        rank=2,
        cga_layout=cga_layout,
    )
    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)
    smem.store(value)
    fence_async_shared(cluster=layout.two_ctas)

    barrier = mbarrier.allocate_mbarrier()
    mbarrier.init(barrier, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    out = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, out)


@gluon.jit
def tmem_copy_128x128_kernel(in_ptr, out_ptr, M: ttgl.constexpr, tmem_layout: ttgl.constexpr):
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
    tmem = allocate_tensor_memory(ttgl.int32, [M, N], layout=tmem_layout)
    tmem_reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    smem.store(value)
    fence_async_shared()
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    output = tmem.load(tmem_reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(output, blocked))


@gluon.jit
def tmem_copy_scales_warpx4_kernel(in_ptr, out_ptr):
    SMEM_H: ttgl.constexpr = 64
    SMEM_W: ttgl.constexpr = 16

    in_ptrs = in_ptr + ttgl.arange(0, SMEM_H)[:, None] * SMEM_W + ttgl.arange(0, SMEM_W)[None, :]
    out_ptrs = out_ptr + ttgl.arange(0, SMEM_H)[:, None] * SMEM_W + ttgl.arange(0, SMEM_W)[None, :]

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
    fence_async_shared()
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    reg_layout: ttgl.constexpr = tmem.get_reg_layout()
    value = tmem.load(reg_layout)
    ttgl.store(ttgl.set_auto_layout(out_ptrs, blocked),
               ttgl.convert_layout(value, blocked))


@gluon.jit
def tmem_copy_scales_layout_probe_kernel(in_ptr, out_ptr, smem_layout: ttgl.constexpr):
    SMEM_H: ttgl.constexpr = 64
    SMEM_W: ttgl.constexpr = 16

    in_ptrs = in_ptr + ttgl.arange(0, SMEM_H)[:, None] * SMEM_W + ttgl.arange(0, SMEM_W)[None, :]
    out_ptrs = out_ptr + ttgl.arange(0, SMEM_H)[:, None] * SMEM_W + ttgl.arange(0, SMEM_W)[None, :]

    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    value = ttgl.load(ttgl.set_auto_layout(in_ptrs, blocked))

    tmem = allocate_tensor_memory(ttgl.int8, (SMEM_H, SMEM_W), layout=TensorMemoryScalesLayout())
    smem = ttgl.allocate_shared_memory(ttgl.int8, (SMEM_H, SMEM_W), layout=smem_layout)
    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)

    smem.store(value)
    fence_async_shared()
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    reg_layout: ttgl.constexpr = tmem.get_reg_layout()
    value = tmem.load(reg_layout)
    ttgl.store(ttgl.set_auto_layout(out_ptrs, blocked),
               ttgl.convert_layout(value, blocked))


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

    blocked_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    blocked_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    a = ttgl.load(ttgl.set_auto_layout(a_ptr + a_offs, blocked_a))
    b = ttgl.load(ttgl.set_auto_layout(b_ptr + b_offs, blocked_b))
    c_offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
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

    scale_offs_k_m = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
    scale_offs_k_n = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, scale_reg_layout_n))[None, :]
    scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
    scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
    a_scale_tmem.store(ttgl.load(a_scale + scale_offs_m * (K // 32) + scale_offs_k_m))
    b_scale_tmem.store(ttgl.load(b_scale + scale_offs_n * (K // 32) + scale_offs_k_n))

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


@gluon.jit
def tmem_mma_scaled_layout_kernel(out_ptr, M: ttgl.constexpr, N: ttgl.constexpr, K: ttgl.constexpr, a, b, a_scale,
                                  b_scale, acc_layout: ttgl.constexpr):
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

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.zeros([M, N], ttgl.float32, layout=acc_reg_layout))

    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [M, K // 32], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype.element_ty, [N, K // 32], scale_layout)
    scale_reg_layout_m: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_reg_layout_n: ttgl.constexpr = b_scale_tmem.get_reg_layout()

    scale_offs_k_m = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
    scale_offs_k_n = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, scale_reg_layout_n))[None, :]
    scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
    scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
    a_scale_tmem.store(ttgl.load(a_scale + scale_offs_m * (K // 32) + scale_offs_k_m))
    b_scale_tmem.store(ttgl.load(b_scale + scale_offs_n * (K // 32) + scale_offs_k_n))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(a_smem, b_smem, acc_tmem, a_scale_tmem, b_scale_tmem, "e5m2", "e5m2", use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out_reg = acc_tmem.load()
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, reg_layout))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, reg_layout))[None, :]
    offs = offs_m * N + offs_n
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out_reg, reg_layout))


@gluon.jit
def tmem_mma_scaled_lhs_subslice_kernel(
    out_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
    a,
    b,
    a_scale,
    b_scale,
    parent_layout: ttgl.constexpr,
    acc_layout: ttgl.constexpr,
):
    PARENT_K: ttgl.constexpr = 2 * K
    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [ttgl.num_warps(), 1], [1, 0])
    nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
        swizzle_byte_width=128, transposed=False, element_bitwidth=8, rank=2
    )
    block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])

    a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout))[:, None]
    a_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, block_layout))[None, :]
    b_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(1, block_layout))[:, None]
    b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, block_layout))[None, :]

    a_tile = ttgl.load(a + a_offs_m * K + a_offs_k)
    b_tile = ttgl.load(b + b_offs_k * N + b_offs_n)

    lhs_parent = allocate_tensor_memory(ttgl.float8e5, [M, PARENT_K], parent_layout)
    lhs_tmem = lhs_parent.slice(K, K, dim=1)
    lhs_reg_layout: ttgl.constexpr = lhs_tmem.get_reg_layout()
    lhs_tmem.store(ttgl.convert_layout(a_tile, lhs_reg_layout))
    b_smem = ttgl.allocate_shared_memory(ttgl.float8e5, [K, N], nvmma_layout, b_tile)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.zeros([M, N], ttgl.float32, layout=acc_reg_layout))

    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [M, K // 32], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype.element_ty, [N, K // 32], scale_layout)
    scale_reg_layout_m: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_reg_layout_n: ttgl.constexpr = b_scale_tmem.get_reg_layout()

    scale_offs_k_m = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
    scale_offs_k_n = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, scale_reg_layout_n))[None, :]
    scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
    scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
    a_scale_tmem.store(ttgl.load(a_scale + scale_offs_m * (K // 32) + scale_offs_k_m))
    b_scale_tmem.store(ttgl.load(b_scale + scale_offs_n * (K // 32) + scale_offs_k_n))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(lhs_tmem, b_smem, acc_tmem, a_scale_tmem, b_scale_tmem, "e5m2", "e5m2", use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out_reg = acc_tmem.load()
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, reg_layout))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, reg_layout))[None, :]
    offs = offs_m * N + offs_n
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out_reg, reg_layout))


@gluon.jit
def tmem_mma_scaled_acc_subslice_kernel(
    out_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
    a,
    b,
    a_scale,
    b_scale,
    parent_layout: ttgl.constexpr,
    slice_start: ttgl.constexpr,
):
    parent_n: ttgl.constexpr = 128
    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [ttgl.num_warps(), 1], [1, 0])
    a_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
        swizzle_byte_width=128, transposed=False, element_bitwidth=8, rank=2
    )
    b_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
        swizzle_byte_width=32 if N == 32 else 64,
        transposed=False,
        element_bitwidth=8,
        rank=2,
    )
    block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])

    a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout))[:, None]
    a_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, block_layout))[None, :]
    b_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(1, block_layout))[:, None]
    b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, block_layout))[None, :]

    a_tile = ttgl.load(a + a_offs_m * K + a_offs_k)
    b_tile = ttgl.load(b + b_offs_k * N + b_offs_n)
    a_smem = ttgl.allocate_shared_memory(ttgl.float8e5, [M, K], a_nvmma_layout, a_tile)
    b_smem = ttgl.allocate_shared_memory(ttgl.float8e5, [K, N], b_nvmma_layout, b_tile)

    acc_parent = allocate_tensor_memory(ttgl.float32, [M, parent_n], parent_layout)
    acc_tmem = acc_parent.slice(slice_start, N, dim=1)
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.zeros([M, N], ttgl.float32, layout=acc_reg_layout))

    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [M, K // 32], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype.element_ty, [N, K // 32], scale_layout)
    scale_reg_layout_m: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_reg_layout_n: ttgl.constexpr = b_scale_tmem.get_reg_layout()

    scale_offs_k_m = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
    scale_offs_k_n = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, scale_reg_layout_n))[None, :]
    scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
    scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
    a_scale_tmem.store(ttgl.load(a_scale + scale_offs_m * (K // 32) + scale_offs_k_m))
    b_scale_tmem.store(ttgl.load(b_scale + scale_offs_n * (K // 32) + scale_offs_k_n))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(a_smem, b_smem, acc_tmem, a_scale_tmem, b_scale_tmem, "e5m2", "e5m2", use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

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

MULTIDIM_SLICE_UNSUPPORTED_LAYOUTS_IDENTITY = {
    "identity": LDST_LAYOUTS["identity"],
}

MULTIDIM_SLICE_POSITIVE_LAYOUTS = {
    "mixed": LDST_LAYOUTS["mixed"],
}

MULTIDIM_SLICE_UNSUPPORTED_LAYOUTS = {
    "scrambled_cols": LDST_EXOTIC_LAYOUTS["scrambled_cols"],
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

PERMUTED_LAYOUT_KINDS = ("identity", "rotate1", "even_odd", "reverse")
PERMUTED_ROW_COL_LAYOUT_KINDS = list(product(PERMUTED_LAYOUT_KINDS, PERMUTED_LAYOUT_KINDS))

LDST_PERMUTED_CASES = [
    (perm_kind, n, variant, LDST_SHAPE_MAP[variant][n])
    for perm_kind, n, variant in product(PERMUTED_LAYOUT_KINDS, (64, 128, 256), LDST_EXPLICIT_VARIANTS)
]

LDST_ROWCOL_PERMUTED_CASES = [
    (row_perm_kind, col_perm_kind, n, variant, LDST_SHAPE_MAP[variant][n])
    for (row_perm_kind, col_perm_kind), n, variant in product(
        PERMUTED_ROW_COL_LAYOUT_KINDS, (64, 128, 256), LDST_EXPLICIT_VARIANTS
    )
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

LDST_DESCRIPTOR_ROUNDTRIP_ROWCOL_CASES = [
    (row_perm_kind, col_perm_kind, n, variant, LDST_SHAPE_MAP[variant][n])
    for (row_perm_kind, col_perm_kind), n, variant in product(
        PERMUTED_ROW_COL_LAYOUT_KINDS, (64, 128, 256), LDST_EXPLICIT_VARIANTS
    )
]

LDST_HIGHER_RANK_INDEX_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n], LDST_SUBVIEW_SHAPE_MAP[variant][n // 2])
    for layout_name, n, variant in product(LDST_LAYOUTS.keys(), (64, 128), LDST_EXPLICIT_VARIANTS)
]

LDST_HIGHER_RANK_SLICE_CASES = [
    (layout_name, n, variant)
    for layout_name, n, variant in product(LDST_LAYOUTS.keys(), (64, 128), LDST_EXPLICIT_VARIANTS)
]

LDST_TWOCTA_HIGHER_RANK_INDEX_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n], LDST_SUBVIEW_SHAPE_MAP[variant][n // 2])
    for layout_name, n, variant in product(LDST_TWOCTA_LAYOUTS.keys(), (64, 128), LDST_EXPLICIT_VARIANTS)
]

LDST_TWOCTA_HIGHER_RANK_SLICE_CASES = [
    (layout_name, n, variant)
    for layout_name, n, variant in product(LDST_TWOCTA_LAYOUTS.keys(), (64, 128), LDST_EXPLICIT_VARIANTS)
]

LDST_HIGHER_RANK_OOR_CASES = [
    (layout_name, variant) for layout_name, variant in product(LDST_LAYOUTS.keys(), LDST_EXPLICIT_VARIANTS)
]

LDST_TWOCTA_HIGHER_RANK_OOR_CASES = [
    (layout_name, variant) for layout_name, variant in product(LDST_TWOCTA_LAYOUTS.keys(), LDST_EXPLICIT_VARIANTS)
]

LDST_TWOCTA_HIGHER_RANK_DIM0_SLICE_OOR_CASES = [
    ("block_two_ctas", variant) for variant in LDST_EXPLICIT_VARIANTS
]

LDST_HIGHER_RANK_POSITIVE_CASES = [
    ("identity", n, variant, LDST_SHAPE_MAP[variant][n])
    for n, variant in product((64, 128, 256), LDST_EXPLICIT_VARIANTS)
]

LDST_HIGHER_RANK_DIM0_SLICE_POSITIVE_CASES = [
    ("identity", n, variant, LDST_SHAPE_MAP[variant][n], LDST_SUBVIEW_SHAPE_MAP[variant][n // 2])
    for n, variant in product((64, 128), LDST_EXPLICIT_VARIANTS)
]

LDST_HIGHER_RANK_HALF_ROWS_POSITIVE_CASES = []

LDST_HIGHER_RANK_HALF_ROWS_CLEAN_ERROR_CASES = [
    ("identity", n, variant, LDST_SHAPE_MAP[variant][n])
    for n, variant in product((64, 128, 256), LDST_EXPLICIT_VARIANTS)
]

LDST_TWOCTA_HIGHER_RANK_DIM0_SLICE_POSITIVE_CASES = [
    ("block_two_ctas", n, variant, LDST_SHAPE_MAP[variant][n], LDST_SUBVIEW_SHAPE_MAP[variant][n // 2])
    for n, variant in product((64, 128), LDST_EXPLICIT_VARIANTS)
]

LDST_TWOCTA_HIGHER_RANK_HALF_ROWS_CLEAN_ERROR_CASES = [
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
    (
        "block",
        _make_tmem_linear_layout_block(128, 128),
        _make_tmem_linear_layout_64x32_block(),
        (
            "Layout has 1 CTAs per CGA, but the context requires 2 CTAs per CGA.",
        ),
    ),
    (
        "block_two_ctas",
        _make_tmem_linear_layout_block(128, 128, two_ctas=True),
        _make_tmem_linear_layout_64x32_block(two_ctas=True),
        (
            "source has no supported register layout",
            "unsupported tensor memory descriptor view for direct tcgen05.ld/st",
            "required row anchors 16,32 are not directly representable in the descriptor view",
        ),
    ),
]

M64_SPLITN_CASES = []
for n in (2, 4, 8, 16, 32, 64, 128):
    if n == 2:
        M64_SPLITN_CASES.append((n, 2, [(0, 0)]))
    else:
        x = n // 4
        M64_SPLITN_CASES.append((n, x, [(0, x), (2 * x, x)]))

M64_ROWCOL_PERMUTED_CASES = [
    (row_perm_kind, col_perm_kind, n, variant)
    for (row_perm_kind, col_perm_kind), n, variant in product(
        PERMUTED_ROW_COL_LAYOUT_KINDS, (2, 4, 8, 16, 32, 64, 128), ("32x32b_splitn", "16x32bx2")
    )
]

M64_ROWCOL_PERMUTED_AUTO_CASES = [
    ("rotate1", "identity", 2),
    ("reverse", "even_odd", 128),
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

CP_LINEAR_NO_SCALES_32BIT_DTYPE_CASES = [
    (dtype_name, torch_dtype, m, n, swizzle, expected_count)
    for dtype_name, torch_dtype in (("f32", torch.float32), ("i32", torch.int32))
    for m, n, swizzle, expected_count in CP_LINEAR_NO_SCALES_CASES
]

CP_INDEXED_VIEW_128X128_CASES = (128, )

CP_LINEAR_INDEXED_VIEW_CASES = [
    (
        "128x128b",
        torch.int32,
        128,
        4,
        _make_tmem_copy_128x128_shared_layout(),
        1,
        "tcgen05.cp.cta_group::1.128x128b",
    ),
    *[
        (
            f"128x256b_{dtype_name}_swizzle{swizzle}",
            torch_dtype,
            128,
            128,
            ttgl.NVMMASharedLayout(swizzle_byte_width=swizzle, element_bitwidth=32, rank=2),
            16,
            "tcgen05.cp.cta_group::1.128x256b",
        )
        for dtype_name, torch_dtype in (("f32", torch.float32), ("i32", torch.int32))
        for swizzle in (32, 64, 128)
    ],
]

CP_LINEAR_SUBSLICE_VIEW_CASES = [
    (dtype_name, torch_dtype, 128, 128, swizzle, 16, "tcgen05.cp.cta_group::1.128x256b")
    for dtype_name, torch_dtype in (("f32", torch.float32), ("i32", torch.int32))
    for swizzle in (32, 64, 128)
]

MMA_INDEXED_ACC_CASES = [
    ("legacy_parent", TensorMemoryLayout((128, 128), col_stride=1), "tensor_memory_encoding"),
    ("linear_parent", _lift_tmem_layout(_make_tmem_linear_layout(128, 128), [2]), "tensor_memory_linear"),
]

CP_NO_SCALES_TWOCTA_CASES = [
    (layout_kind, n, swizzle, expected_count)
    for layout_kind, (n, swizzle, expected_count) in product(
        ("linear", "legacy"),
        (
            (16, 32, 2),
            (16, 64, 2),
            (32, 32, 4),
            (32, 64, 4),
            (32, 128, 4),
            (64, 32, 8),
            (64, 64, 8),
            (64, 128, 8),
            (128, 32, 16),
            (128, 64, 16),
            (128, 128, 16),
            (256, 32, 32),
            (256, 64, 32),
            (256, 128, 32),
        ),
    )
]

CP_LINEAR_NO_SCALES_SUBWORD_UNSUPPORTED_CASES = [
    (dtype_name, torch_dtype, 128, n, 32)
    for dtype_name, torch_dtype in (("f16", torch.float16), ("bf16", torch.bfloat16))
    for n in (128, 256)
]

CP_NO_SCALES_SWIZZLE_CASES = [
    (m, n, block_n, swizzle)
    for swizzle in (32, 64, 128)
    for (m, n, block_n) in ((128, 128, 128), (128, 256, 256), (256, 128, 64))
]

CP_NO_SCALES_128X128_CASES = [
    ("legacy", 128),
    ("linear", 128),
]

CP_SCALES_WARPX4_FORMAT_PAIRS = [
    ("mxfp8", "mxfp8"),
    ("mxfp4", "mxfp4"),
    ("mxfp8", "mxfp4"),
    ("mxfp4", "mxfp8"),
    ("nvfp4", "nvfp4"),
]

CP_SCALES_WARPX4_SCALED_MMA_CASES = [
    (a_format, b_format, num_ctas, acc_layout_kind)
    for (a_format, b_format), num_ctas, acc_layout_kind in product(
        CP_SCALES_WARPX4_FORMAT_PAIRS, (1, 2), ("legacy", "linear")
    )
]

CP_SCALES_WARPX4_GEOMETRY_CASES = [
    (block_n, block_k, multicast, num_ctas, acc_layout_kind)
    for block_n, block_k, multicast, num_ctas, acc_layout_kind in product(
        (128, 256), (128, 256), (False, True), (1, 2), ("legacy", "linear")
    )
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

X1_F16_LDST_CASES = [
    ("linear_packed", lambda: _make_tmem_linear_layout(128, 2), "32x32b.x1.b32", "32x32b.x1.b32"),
    ("legacy_packed", lambda: TensorMemoryLayout((128, 2), col_stride=1), "32x32b.x1.b32", "32x32b.x1.b32"),
    ("legacy_unpacked", lambda: TensorMemoryLayout((128, 2), col_stride=2), "32x32b.x1.unpack::16b.b32",
     "32x32b.x1.pack::16b.b32"),
]

X1_F32_LDST_CASES = [
    ("linear_onecta", 128, 1, lambda: _make_tmem_linear_layout(128, 1)),
    ("legacy_onecta", 128, 1, lambda: TensorMemoryLayout((128, 1), col_stride=1)),
    ("linear_twocta", 256, 2, lambda: _make_tmem_linear_layout_mmav5_twocta(256, 1)),
]

X1_F32_UNSUPPORTED_VARIANTS = ("16x64b", "16x128b", "16x256b")

SCALES_LDST_CASES = [
    (16, 8, 8, 1, tuple(), _expected_ldst_ops("16x32bx2.x1.b32", [0])),
    (128, 64, 4, 2, ((1, 0),), _expected_ldst_ops("16x32bx2.x32.b32", [0])),
    (256, 32, 4, 2, ((1, 0),), _expected_ldst_ops("16x32bx2.x32.b32", [0])),
    (256, 64, 4, 2, ((1, 0),), _expected_ldst_ops("16x32bx2.x64.b32", [0])),
]

SCALES_LDST_VARIANT_CASES = [
    (16, 4, 4, "32x32b", _expected_ldst_ops("16x32bx2.x1.b32", [0])),
    (16, 8, 8, "32x32b", _expected_ldst_ops("16x32bx2.x1.b32", [0])),
    (32, 4, 8, "32x32b", _expected_ldst_ops("16x32bx2.x1.b32", [0])),
    (64, 4, 4, "32x32b", _expected_ldst_ops("16x32bx2.x2.b32", [0])),
    (64, 8, 4, "32x32b", _expected_ldst_ops("16x32bx2.x4.b32", [0])),
    (64, 16, 4, "32x32b", _expected_ldst_ops("16x32bx2.x8.b32", [0])),
    (64, 16, 8, "32x32b", _expected_ldst_ops("32x32b.x4.b32", [0])),
    (64, 32, 4, "32x32b", _expected_ldst_ops("16x32bx2.x16.b32", [0])),
    (64, 32, 8, "32x32b", _expected_ldst_ops("32x32b.x8.b32", [0])),
    (128, 4, 4, "32x32b", _expected_ldst_ops("16x32bx2.x4.b32", [0])),
    (128, 8, 4, "32x32b", _expected_ldst_ops("16x32bx2.x8.b32", [0])),
    (128, 8, 8, "32x32b", _expected_ldst_ops("32x32b.x4.b32", [0])),
    (128, 16, 4, "32x32b", _expected_ldst_ops("16x32bx2.x16.b32", [0])),
    (128, 16, 8, "32x32b", _expected_ldst_ops("32x32b.x8.b32", [0])),
    (128, 32, 4, "32x32b", _expected_ldst_ops("16x32bx2.x32.b32", [0])),
    (128, 32, 8, "32x32b", _expected_ldst_ops("32x32b.x16.b32", [0])),
    (16, 4, 4, "16x32bx2", _expected_ldst_ops("16x32bx2.x1.b32", [0])),
    (64, 8, 8, "16x32bx2", _expected_ldst_ops("16x32bx2.x1.b32", [0, 2])),
    (64, 16, 8, "16x32bx2", _expected_ldst_ops("16x32bx2.x2.b32", [0, 4])),
    (64, 32, 8, "16x32bx2", _expected_ldst_ops("16x32bx2.x4.b32", [0, 8])),
    (128, 8, 8, "16x32bx2", _expected_ldst_ops("16x32bx2.x2.b32", [0, 4])),
    (128, 16, 8, "16x32bx2", _expected_ldst_ops("16x32bx2.x4.b32", [0, 8])),
    (128, 32, 8, "16x32bx2", _expected_ldst_ops("16x32bx2.x8.b32", [0, 16])),
    (16, 8, 4, "32x32b", _expected_ldst_ops("16x32bx2.x1.b32", [0])),
    (16, 16, 4, "32x32b", _expected_ldst_ops("16x32bx2.x1.b32", [0, 2])),
    (16, 16, 8, "32x32b", _expected_ldst_ops("16x32bx2.x1.b32", [0])),
    (32, 8, 4, "32x32b", _expected_ldst_ops("16x32bx2.x1.b32", [0, 2])),
    (32, 8, 8, "32x32b", _expected_ldst_ops("32x32b.x1.b32", [0])),
    (16, 8, 4, "16x32bx2", _expected_ldst_ops("16x32bx2.x1.b32", [0, 2])),
    (16, 16, 4, "16x32bx2", _expected_ldst_ops("16x32bx2.x1.b32", [0, 4])),
    (16, 16, 8, "16x32bx2", _expected_ldst_ops("16x32bx2.x1.b32", [0, 4])),
    (32, 16, 4, "16x32bx2", _expected_ldst_ops("16x32bx2.x1.b32", [0, 2, 4, 6])),
    (32, 4, 8, "16x32bx2", _expected_ldst_ops("16x32bx2.x1.b32", [0])),
]

SCALES_LDST_VARIANT_CLEAN_UNSUPPORTED_CASES = [
    (
        16,
        8,
        8,
        "16x32bx2",
        "To be able to `tmem.load` into `tl.split` you need to have more than 4 8-bit registers",
        False,
    ),
]

LD_RED_LINEAR_CASES = [
    ("identity", 128, 32, 4, "32x32b.x32"),
    ("identity", 128, 64, 4, "32x32b.x64"),
    ("identity", 128, 128, 4, "32x32b.x128"),
    ("identity", 128, 256, 4, "32x32b.x64"),
    ("legacy_equivalent_256x128", 256, 128, 8, "32x32b.x128"),
]

LD_RED_EXPECTED_OP_COUNT = {32: 1, 64: 1, 128: 1, 256: 4}

LD_RED_MODIFIER_CASES = [
    (False, tl.PropagateNan.NONE),
    (False, tl.PropagateNan.ALL),
    (True, tl.PropagateNan.NONE),
    (True, tl.PropagateNan.ALL),
]

LD_RED_TILE_PERMUTED_CASES = [
    (128, 32, 4, "32x32b.x128"),
    (256, 32, 4, "32x32b.x64"),
    (256, 64, 4, "32x32b.x64"),
]

LD_RED_ROWCOL_PERMUTED_CASES = [
    (row_perm_kind, col_perm_kind, "32x32b.x128")
    for row_perm_kind, col_perm_kind in PERMUTED_ROW_COL_LAYOUT_KINDS
    if not (row_perm_kind == "identity" and col_perm_kind == "identity")
]

LD_RED_COL_PERMUTED_CASES = [
    (col_perm_kind, "32x32b.x128")
    for col_perm_kind in PERMUTED_LAYOUT_KINDS
    if col_perm_kind != "identity"
]

LD_RED_ROW_PERMUTED_CASES = [
    (row_perm_kind, col_perm_kind, expected_shape)
    for row_perm_kind, col_perm_kind, expected_shape in LD_RED_ROWCOL_PERMUTED_CASES
    if row_perm_kind != "identity"
]

LD_RED_MIXED_CASES = [
    (128, 64, 4),
    (128, 128, 4),
    (128, 256, 4),
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
    ("warpx2_candidate", _make_scales_shared_layout_warpx2_candidate(), "CLEAN_UNSUPPORTED"),
]

CP_LINEAR_EXOTIC_UNSUPPORTED_CASES = [
    ("mixed", _make_tmem_linear_layout_mixed(128, 128)),
    ("scrambled_cols", _make_tmem_linear_layout_permuted(128, 128, "identity", "even_odd")),
    ("scrambled_rows_cols", _make_tmem_linear_layout_permuted(128, 128, "even_odd", "even_odd")),
]

CP_LINEAR_PERMUTED_UNSUPPORTED_CASES = [
    (row_perm_kind, col_perm_kind)
    for row_perm_kind, col_perm_kind in PERMUTED_ROW_COL_LAYOUT_KINDS
    if not (row_perm_kind == "identity" and col_perm_kind == "identity")
]

MMA_EXOTIC_UNSUPPORTED_CASES = [
    ("mixed", _make_tmem_linear_layout_mixed(128, 128)),
    ("scrambled_cols", _make_tmem_linear_layout_permuted(128, 128, "identity", "even_odd")),
]

MMA_ROWCOL_PERMUTED_CASES = [
    (row_perm_kind, col_perm_kind)
    for row_perm_kind, col_perm_kind in PERMUTED_ROW_COL_LAYOUT_KINDS
    if not (row_perm_kind == "identity" and col_perm_kind == "identity")
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
@pytest.mark.parametrize("perm_kind,n,variant,expected_shape", LDST_PERMUTED_CASES)
def test_tmem_runtime_matrix_ldst_permuted_layout_sweep(perm_kind, n, variant, expected_shape):
    m = 128
    layout = _make_tmem_linear_layout_permuted(m, n, perm_kind, perm_kind)
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
@pytest.mark.parametrize("row_perm_kind,col_perm_kind,n,variant,expected_shape", LDST_ROWCOL_PERMUTED_CASES)
def test_tmem_runtime_matrix_ldst_rowcol_permuted_layout_sweep(row_perm_kind, col_perm_kind, n, variant, expected_shape):
    m = 128
    layout = _make_tmem_linear_layout_permuted(m, n, row_perm_kind, col_perm_kind)
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
def test_tmem_runtime_matrix_ldst_exotic_layouts_report_clean_unsupported(layout_name, n, variant, capfd):
    m = 128
    layout = LDST_EXOTIC_UNSUPPORTED_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(Exception) as excinfo:
        tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    _assert_clean_cta_per_cga_mismatch_or_descriptor_unsupported(text, layout_ctas=2, required_ctas=1)
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


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
@pytest.mark.parametrize("perm_kind,n,variant,expected_shape", LDST_PERMUTED_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_compositions_permuted_layout_sweep(perm_kind, n, variant, expected_shape):
    m = 128
    layout = _make_tmem_linear_layout_permuted(m, n, perm_kind, perm_kind)
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
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("row_perm_kind,col_perm_kind,n,variant,expected_shape", LDST_ROWCOL_PERMUTED_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_compositions_rowcol_permuted_layout_sweep(
    row_perm_kind, col_perm_kind, n, variant, expected_shape
):
    m = 128
    layout = _make_tmem_linear_layout_permuted(m, n, row_perm_kind, col_perm_kind)
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
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


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
@pytest.mark.parametrize("row_perm_kind,col_perm_kind,n,variant,expected_shape", LDST_DESCRIPTOR_ROUNDTRIP_ROWCOL_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_roundtrip_rowcol_permuted_sweeps(
    row_perm_kind, col_perm_kind, n, variant, expected_shape
):
    m = 128
    layout = _lift_tmem_layout(_make_tmem_linear_layout_permuted(m, n, row_perm_kind, col_perm_kind), [2, 2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    try:
        compiled = tmem_ldst_descriptor_roundtrip_kernel[(1, )](
            inp, out, layout, m, n, variant, 1, 7.0, num_warps=4
        )
    except triton.runtime.errors.OutOfResources:
        pytest.skip(
            f"tensor memory OOR for row={row_perm_kind}, col={col_perm_kind}, n={n}, variant={variant}"
        )
    torch.testing.assert_close(out, inp + 7.0, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes

    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_reinterpret" not in ttgir


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
def test_tmem_runtime_matrix_ldst_descriptor_multidim_slices(layout_name, n, variant):
    m = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_multidim_slice_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4
    )
    torch.testing.assert_close(out, inp + 9.0, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    assert ops
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "tt.reshape" in ttgir
    assert "tt.trans" in ttgir
    assert "tt.split" in ttgir
    assert "tt.join" in ttgir
    assert "ttg.memdesc_subslice" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,variant", LDST_HIGHER_RANK_OOR_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_higher_rank_index_reports_tmem_oor(layout_name, variant):
    m = 128
    n = 256
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(triton.runtime.errors.OutOfResources) as excinfo:
        tmem_ldst_descriptor_higher_rank_index_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)

    text = str(excinfo.value)
    _assert_clean_tmem_oor(text, required=1024, hardware_limit=512)
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,variant", LDST_HIGHER_RANK_OOR_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_multidim_slices_report_tmem_oor(layout_name, variant):
    m = 128
    n = 256
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(triton.runtime.errors.OutOfResources) as excinfo:
        tmem_ldst_descriptor_multidim_slice_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)

    text = str(excinfo.value)
    _assert_clean_tmem_oor(text, required=1024, hardware_limit=512)
    assert "Assertion" not in text


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
def test_tmem_runtime_matrix_ldst_twocta_descriptor_multidim_slices(layout_name, n, variant):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_multidim_slice_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
    )
    torch.testing.assert_close(out, inp + 9.0, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    assert ops
    ttgir = compiled.asm["ttgir"]
    assert "twoCTAs = true" in ttgir
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "tt.reshape" in ttgir
    assert "tt.trans" in ttgir
    assert "tt.split" in ttgir
    assert "tt.join" in ttgir
    assert "ttg.memdesc_subslice" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,variant", LDST_TWOCTA_HIGHER_RANK_OOR_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_index_reports_tmem_oor(layout_name, variant):
    m = 256
    n = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(triton.runtime.errors.OutOfResources) as excinfo:
        tmem_ldst_descriptor_higher_rank_index_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
        )

    text = str(excinfo.value)
    _assert_clean_tmem_oor(text, required=1024, hardware_limit=512)
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,variant", LDST_TWOCTA_HIGHER_RANK_OOR_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_multidim_slices_report_tmem_oor(layout_name, variant):
    m = 256
    n = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(triton.runtime.errors.OutOfResources) as excinfo:
        tmem_ldst_descriptor_multidim_slice_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
        )

    text = str(excinfo.value)
    _assert_clean_tmem_oor(text, required=1024, hardware_limit=512)
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape,expected_half_shape",
                         LDST_HIGHER_RANK_DIM0_SLICE_POSITIVE_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_higher_rank_dim0_slice_positive_lifted_layout(
    layout_name, n, variant, expected_shape, expected_half_shape
):
    m = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_higher_rank_dim0_slice_positive_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4
    )
    ref = inp.clone()
    ref[:, : n // 2] += 7.0
    torch.testing.assert_close(out, ref, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert f"tcgen05.st.sync.aligned.{expected_shape}" in observed_opcodes
    assert f"tcgen05.ld.sync.aligned.{expected_shape}" in observed_opcodes
    assert f"tcgen05.st.sync.aligned.{expected_half_shape}" in observed_opcodes
    assert f"tcgen05.ld.sync.aligned.{expected_half_shape}" in observed_opcodes


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,variant", LDST_HIGHER_RANK_OOR_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_higher_rank_dim0_slice_reports_tmem_oor(layout_name, variant):
    m = 128
    n = 256
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(triton.runtime.errors.OutOfResources) as excinfo:
        tmem_ldst_descriptor_higher_rank_dim0_slice_positive_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4
        )

    text = str(excinfo.value)
    _assert_clean_tmem_oor(text, required=1024, hardware_limit=512)
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,layout_fn", MULTIDIM_SLICE_UNSUPPORTED_LAYOUTS_IDENTITY.items())
def test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_identity_reports_clean_error(
    layout_name, layout_fn, capfd
):
    m = 128
    n = 128
    layout = layout_fn(n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(CompilationError) as excinfo:
        tmem_ldst_descriptor_multidim_slice_positive_kernel[(1, )](
            inp, out, layout, m, n, "16x128b", num_warps=4
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "TMEM layout 'auto' unsupported for descriptor view" in text
    assert "tensor_memory_descriptor<fp32, [32, 32]," in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,layout_fn", MULTIDIM_SLICE_POSITIVE_LAYOUTS.items())
def test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_positive(layout_name, layout_fn):
    m = 128
    n = 128
    layout = layout_fn(n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_multidim_slice_positive_kernel[(1, )](
        inp, out, layout, m, n, "16x128b", num_warps=4
    )
    ref = inp.clone().reshape(2, m // 2, 2, n // 2)
    ref[1, 0:m // 4, 1, 0:n // 4] += 11.0
    ref = ref.reshape(m, n)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    assert ops == [
        ("tcgen05.st.sync.aligned.16x128b.x32.b32", 0),
        ("tcgen05.st.sync.aligned.16x128b.x32.b32", 1048576),
        ("tcgen05.ld.sync.aligned.32x32b.x1.b32", 0),
        ("tcgen05.ld.sync.aligned.32x32b.x1.b32", 4),
        ("tcgen05.ld.sync.aligned.32x32b.x1.b32", 8),
        ("tcgen05.ld.sync.aligned.32x32b.x1.b32", 12),
        ("tcgen05.ld.sync.aligned.32x32b.x1.b32", 16),
        ("tcgen05.ld.sync.aligned.32x32b.x1.b32", 20),
        ("tcgen05.ld.sync.aligned.32x32b.x1.b32", 24),
        ("tcgen05.ld.sync.aligned.32x32b.x1.b32", 28),
        ("tcgen05.st.sync.aligned.32x32b.x1.b32", 0),
        ("tcgen05.st.sync.aligned.32x32b.x1.b32", 4),
        ("tcgen05.st.sync.aligned.32x32b.x1.b32", 8),
        ("tcgen05.st.sync.aligned.32x32b.x1.b32", 12),
        ("tcgen05.st.sync.aligned.32x32b.x1.b32", 16),
        ("tcgen05.st.sync.aligned.32x32b.x1.b32", 20),
        ("tcgen05.st.sync.aligned.32x32b.x1.b32", 24),
        ("tcgen05.st.sync.aligned.32x32b.x1.b32", 28),
        ("tcgen05.ld.sync.aligned.16x128b.x32.b32", 0),
        ("tcgen05.ld.sync.aligned.16x128b.x32.b32", 1048576),
    ]

    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttg.memdesc_reshape" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,layout_fn", MULTIDIM_SLICE_UNSUPPORTED_LAYOUTS.items())
def test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_reports_clean_unsupported(
    layout_name, layout_fn, capfd
):
    m = 128
    n = 128
    layout = layout_fn(n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(CompilationError) as excinfo:
        tmem_ldst_descriptor_multidim_slice_positive_kernel[(1, )](
            inp, out, layout, m, n, "16x128b", num_warps=4
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert (
        "supported TMEM register layout" in text
        or "TMEM layout 'auto' unsupported for descriptor view" in text
    )
    if "supported TMEM register layout" in text:
        assert "reshape or permute so TMEM columns stay contiguous" in text
        assert "insert convert_layout explicitly" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text

@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_HIGHER_RANK_HALF_ROWS_POSITIVE_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_positive_lifted_layout(
    layout_name, n, variant, expected_shape
):
    m = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_higher_rank_half_rows_positive_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4
    )
    ref = inp.clone()
    ref[m // 2 :, :] += 13.0
    torch.testing.assert_close(out, ref, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert f"tcgen05.st.sync.aligned.{expected_shape}" in observed_opcodes
    assert f"tcgen05.ld.sync.aligned.{expected_shape}" in observed_opcodes


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_HIGHER_RANK_HALF_ROWS_CLEAN_ERROR_CASES)
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
    assert f"TMEM layout 'constexpr[{variant}]' unsupported" in msg
    assert "descriptor view" in msg
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape,expected_half_shape",
                         LDST_TWOCTA_HIGHER_RANK_DIM0_SLICE_POSITIVE_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_dim0_slice_positive_lifted_layout(
    layout_name, n, variant, expected_shape, expected_half_shape
):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_higher_rank_dim0_slice_positive_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
    )
    ref = inp.clone()
    ref[:, : n // 2] += 7.0
    torch.testing.assert_close(out, ref, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert f"tcgen05.st.sync.aligned.{expected_shape}" in observed_opcodes
    assert f"tcgen05.ld.sync.aligned.{expected_shape}" in observed_opcodes
    assert f"tcgen05.st.sync.aligned.{expected_half_shape}" in observed_opcodes
    assert f"tcgen05.ld.sync.aligned.{expected_half_shape}" in observed_opcodes


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,variant", LDST_TWOCTA_HIGHER_RANK_DIM0_SLICE_OOR_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_dim0_slice_reports_tmem_oor(layout_name, variant):
    m = 256
    n = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(triton.runtime.errors.OutOfResources) as excinfo:
        tmem_ldst_descriptor_higher_rank_dim0_slice_positive_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
        )

    text = str(excinfo.value)
    _assert_clean_tmem_oor(text, required=1024, hardware_limit=512)
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_TWOCTA_HIGHER_RANK_HALF_ROWS_CLEAN_ERROR_CASES)
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
    assert f"TMEM layout 'constexpr[{variant}]' unsupported" in msg
    assert "descriptor view" in msg
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
    _assert_clean_cta_per_cga_mismatch(text, layout_ctas=1, required_ctas=2)
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,layout", BLOCKED_FALLBACK_CASES)
def test_tmem_runtime_matrix_blocked_layout_reports_clean_error(layout_name, layout, capfd):
    m = n = 128
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises((CompilationError, RuntimeError)) as excinfo:
        tmem_ldst_blocked_fallback_kernel[(1, )](inp, out, layout, num_warps=4)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "source has no supported register layout" in text or "result has no supported register layout" in text
    assert "potential TMEM layout:" in text
    assert "requested layout direct-lowering details:" in text
    assert "insert convert_layout explicitly" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("name,layout,reinterpret_layout,expected_fragments", UNSUPPORTED_BLOCK_DESCRIPTOR_CASES)
def test_tmem_runtime_matrix_block_descriptor_reports_clean_error(
    name, layout, reinterpret_layout, expected_fragments, capfd
):
    reg_layout = _make_tmem_register_layout(2)
    with pytest.raises(Exception) as excinfo:
        tmem_block_descriptor_compile_kernel[(1, )](
            layout, reinterpret_layout, reg_layout, num_warps=4, num_ctas=2
        )
    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    for fragment in expected_fragments:
        assert fragment in text
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
@pytest.mark.parametrize("n,splitn_x,offset_imm_pairs", M64_SPLITN_CASES)
def test_tmem_runtime_matrix_splitn_auto_selects_16x32bx2(n, splitn_x, offset_imm_pairs):
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

    expected = []
    for offset, imm in offset_imm_pairs:
        expected.append((expected_opcode, offset, imm))
    for offset, imm in offset_imm_pairs:
        expected.append((expected_opcode_ld, offset, imm))
    assert imms == expected


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64, 128])
def test_tmem_runtime_matrix_explicit_16x32bx2_matches_splitn(n):
    m = 64
    layout = _make_tmem_linear_layout_m64(n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out_explicit = torch.empty_like(inp)
    out_splitn = torch.empty_like(inp)

    compiled_explicit = tmem_ldst_variant_kernel[(1, )](
        inp, out_explicit, layout, m, n, "16x32bx2", num_warps=4
    )
    compiled_splitn = tmem_ldst_variant_kernel[(1, )](
        inp, out_splitn, layout, m, n, "32x32b_splitn", num_warps=4
    )

    torch.testing.assert_close(out_explicit, inp, atol=0, rtol=0)
    torch.testing.assert_close(out_splitn, inp, atol=0, rtol=0)
    torch.testing.assert_close(out_explicit, out_splitn, atol=0, rtol=0)

    explicit_ops, explicit_imms = _assert_ldst_ptx_llir_match(compiled_explicit)
    splitn_ops, splitn_imms = _assert_ldst_ptx_llir_match(compiled_splitn)
    assert explicit_ops == splitn_ops
    assert explicit_imms == splitn_imms
    assert all("16x32bx2" in op for op, _ in explicit_ops)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("row_perm_kind,col_perm_kind,n,variant", M64_ROWCOL_PERMUTED_CASES)
def test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep(row_perm_kind, col_perm_kind, n, variant):
    m = 64
    layout = _make_tmem_linear_layout_m64_permuted(n, row_perm_kind, col_perm_kind)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert observed_opcodes
    assert all("16x32bx2" in op for op in observed_opcodes)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("row_perm_kind,col_perm_kind,n", M64_ROWCOL_PERMUTED_AUTO_CASES)
def test_tmem_runtime_matrix_splitn_rowcol_permuted_auto_selects_16x32bx2(row_perm_kind, col_perm_kind, n):
    m = 64
    layout = _make_tmem_linear_layout_m64_permuted(n, row_perm_kind, col_perm_kind)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert observed_opcodes
    assert all("16x32bx2" in op for op in observed_opcodes)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


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
@pytest.mark.parametrize("layout_kind,layout_factory,expected_st,expected_ld", X1_F16_LDST_CASES)
def test_tmem_runtime_matrix_ldst_x1_f16_roundtrip(layout_kind, layout_factory, expected_st, expected_ld):
    m, n = 128, 2
    layout = layout_factory()
    inp = torch.arange(m * n, dtype=torch.float16, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_f16_variant_kernel[(1, )](inp, out, layout, m, n, "32x32b", num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    assert ops == [(f"tcgen05.st.sync.aligned.{expected_st}", 0), (f"tcgen05.ld.sync.aligned.{expected_ld}", 0)]

    ttgir = compiled.asm["ttgir"]
    if layout_kind.startswith("linear"):
        assert "tensor_memory_linear" in ttgir
        assert "tensor_memory_encoding" not in ttgir
    else:
        assert "tensor_memory_encoding" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_kind,m,num_ctas,layout_factory", X1_F32_LDST_CASES)
def test_tmem_runtime_matrix_ldst_x1_f32_roundtrip(layout_kind, m, num_ctas, layout_factory):
    n = 1
    layout = layout_factory()
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, "32x32b", num_warps=4, num_ctas=num_ctas)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    assert ops == [
        ("tcgen05.st.sync.aligned.32x32b.x1.b32", 0),
        ("tcgen05.ld.sync.aligned.32x32b.x1.b32", 0),
    ]
    ttgir = compiled.asm["ttgir"]
    if layout_kind.startswith("linear"):
        assert "tensor_memory_linear" in ttgir
    else:
        assert "tensor_memory_encoding" in ttgir
    if num_ctas == 2:
        assert "twoCTAs = true" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_kind,m,num_ctas,layout_factory", X1_F32_LDST_CASES)
def test_tmem_runtime_matrix_ldst_x1_f32_descriptor_chain_roundtrip(layout_kind, m, num_ctas, layout_factory):
    n = 1
    layout = layout_factory()
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_chain_kernel[(1, )](
        inp, out, layout, m, n, "32x32b", num_warps=4, num_ctas=num_ctas
    )
    torch.testing.assert_close(out, inp + 3.0, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    assert ops == [
        ("tcgen05.st.sync.aligned.32x32b.x1.b32", 0),
        ("tcgen05.ld.sync.aligned.32x32b.x1.b32", 0),
        ("tcgen05.st.sync.aligned.32x32b.x1.b32", 0),
        ("tcgen05.ld.sync.aligned.32x32b.x1.b32", 0),
    ]
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "ttg.memdesc_reshape" in ttgir
    assert "ttg.memdesc_trans" in ttgir
    if num_ctas == 2:
        assert "twoCTAs = true" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_kind,m,num_ctas,layout_factory", X1_F32_LDST_CASES)
@pytest.mark.parametrize("variant", X1_F32_UNSUPPORTED_VARIANTS)
def test_tmem_runtime_matrix_ldst_x1_f32_unsupported_variants_report_clean_unsupported(
    layout_kind, m, num_ctas, layout_factory, variant
):
    n = 1
    layout = layout_factory()
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(CompilationError) as excinfo:
        tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4, num_ctas=num_ctas)

    msg = str(excinfo.value)
    _assert_clean_unsupported_descriptor_view(msg, variant)
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("M,N,num_warps,num_ctas,cga_layout,expected_ops", SCALES_LDST_CASES)
def test_tmem_runtime_matrix_ldst_scales_direct_roundtrip(M, N, num_warps, num_ctas, cga_layout, expected_ops):
    inp = torch.arange(M * N, dtype=torch.int8, device="cuda").reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_scales_ldst_kernel[(1, )](inp, out, M, N, cga_layout, num_warps=num_warps, num_ctas=num_ctas)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    assert ops == expected_ops
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_scales_encoding" in ttgir
    if cga_layout:
        assert f"CGALayout = {str([list(basis) for basis in cga_layout])}" in ttgir
    if num_ctas > 1:
        assert '"ttg.num-ctas" = 2' in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("M,N,num_warps,instr_variant,expected_ops", SCALES_LDST_VARIANT_CASES)
def test_tmem_runtime_matrix_ldst_scales_variant_sweep(M, N, num_warps, instr_variant, expected_ops):
    inp = torch.arange(M * N, dtype=torch.int8, device="cuda").reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_scales_ldst_variant_kernel[(1, )](inp, out, M, N, instr_variant, num_warps=num_warps)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    assert ops == expected_ops
    assert "tensor_memory_scales_encoding" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "M,N,num_warps,instr_variant,expected_text,needs_capfd",
    SCALES_LDST_VARIANT_CLEAN_UNSUPPORTED_CASES,
)
def test_tmem_runtime_matrix_ldst_scales_variant_reports_clean_unsupported(
    M, N, num_warps, instr_variant, expected_text, needs_capfd, capfd
):
    inp = torch.arange(M * N, dtype=torch.int8, device="cuda").reshape(M, N)
    out = torch.empty_like(inp)

    with pytest.raises(Exception) as excinfo:
        tmem_scales_ldst_variant_kernel[(1, )](inp, out, M, N, instr_variant, num_warps=num_warps)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out

    _assert_clean_unsupported_descriptor_view(text, expected_text)
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


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
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_MODIFIER_CASES)
@pytest.mark.parametrize("layout_name,M,N,num_warps,expected_shape", LD_RED_LINEAR_CASES)
def test_tmem_runtime_matrix_ld_red_identity_linear_layout(red_op, use_abs, propagate_nan, layout_name, M, N, num_warps,
                                                           expected_shape):
    if layout_name == "identity":
        layout = _make_tmem_linear_layout(M, N)
    elif layout_name == "legacy_equivalent_256x128":
        layout = _make_tmem_linear_layout_ld_red_legacy_equivalent_256x128()
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
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_MODIFIER_CASES)
@pytest.mark.parametrize("N,tile_n,num_warps,expected_shape", LD_RED_TILE_PERMUTED_CASES)
def test_tmem_runtime_matrix_ld_red_tile_permuted_linear_layout(
    red_op, use_abs, propagate_nan, N, tile_n, num_warps, expected_shape
):
    M = 128
    layout = _make_tmem_linear_layout_tile_permuted(M, N, tile_n)
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
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_MODIFIER_CASES)
@pytest.mark.parametrize("col_perm_kind,expected_shape", LD_RED_COL_PERMUTED_CASES)
def test_tmem_runtime_matrix_ld_red_col_permuted_linear_layout(
    red_op, use_abs, propagate_nan, col_perm_kind, expected_shape
):
    M = N = 128
    num_warps = 4
    layout = _make_tmem_linear_layout_permuted(M, N, "identity", col_perm_kind)
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
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_MODIFIER_CASES)
@pytest.mark.parametrize("row_perm_kind,col_perm_kind,expected_shape", LD_RED_ROW_PERMUTED_CASES)
def test_tmem_runtime_matrix_ld_red_row_permuted_linear_layout(
    red_op, use_abs, propagate_nan, row_perm_kind, col_perm_kind, expected_shape
):
    M = N = 128
    num_warps = 4
    layout = _make_tmem_linear_layout_permuted(M, N, row_perm_kind, col_perm_kind)
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
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_MODIFIER_CASES)
@pytest.mark.parametrize("M,N,num_warps", LD_RED_MIXED_CASES)
def test_tmem_runtime_matrix_ld_red_mixed_linear_layout_reports_clean_unsupported(
    red_op, use_abs, propagate_nan, M, N, num_warps, capfd
):
    layout = _make_tmem_linear_layout_mixed(M, N)
    with pytest.raises(Exception) as err:
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
    text = str(err.value) + captured.err + captured.out
    assert "tmem_load reduction source layout is not directly tcgen05.ld.red-compatible" in text
    assert "tmem.load(...)+tt.reduce(...)" in text
    assert "tt.reduce" in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("name,smem_layout,expected_status", CP_SCALES_LAYOUT_PROBE_CASES)
def test_tmem_runtime_matrix_cp_scales_layout_probe(name, smem_layout, expected_status, capfd):
    smem_h, smem_w = 64, 16
    inp = torch.randint(size=(smem_h, smem_w), low=-100, high=100, dtype=torch.int8, device="cuda")
    out = torch.empty_like(inp)

    if expected_status == "PASS":
        compiled = tmem_copy_scales_layout_probe_kernel[(1, )](inp, out, smem_layout)
        torch.testing.assert_close(out, inp, atol=0, rtol=0)
        _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.warpx4.32x128b"] * 2)
        assert "ttg.memdesc_reinterpret" not in compiled.asm["ttgir"]
        return

    with pytest.raises(Exception) as excinfo:
        tmem_copy_scales_layout_probe_kernel[(1, )](inp, out, smem_layout)
    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out

    if expected_status == "CLEAN_UNSUPPORTED":
        assert "maps to tcgen05.copy." in text
        assert "could not synthesize a compatible shared-memory descriptor plan for tensor memory scales" in text
        assert "Use a shared layout that lowers to tcgen05.copy." in text
        assert "same descriptor family" in text
        assert "Assertion" not in text
        return

    assert expected_status == "BUG"
    assert "failed to find valid tcgen05.copy layout" in text
    assert "PassManager::run failed" in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_bug_cp_scales_unsupported_layout_raises_runtimeerror_parse(capfd):
    smem_h, smem_w = 64, 16
    num_rows = 128
    num_cols = smem_h * smem_w // 32
    inp = torch.randint(size=(smem_h, smem_w), low=-100, high=100, dtype=torch.int8, device="cuda")
    out = torch.zeros(size=(num_rows, num_cols), dtype=torch.int8, device="cuda")
    smem_layout = _make_scales_shared_layout_warpx2_candidate()

    with pytest.raises(RuntimeError) as excinfo:
        tmem_copy_scales_layout_probe_kernel[(1, )](inp, out, smem_layout)

    captured = capfd.readouterr()
    text = captured.err + captured.out
    assert "maps to tcgen05.copy." in text
    assert "could not synthesize a compatible shared-memory descriptor plan for tensor memory scales" in text
    assert "Use a shared layout that lowers to tcgen05.copy." in text
    assert "same descriptor family" in text
    assert "Assertion" not in text
    assert (
        "error encountered during parsing" in str(excinfo.value)
        or "PassManager::run failed" in str(excinfo.value)
    )


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

    expected_count = (M * N) // 1024
    expected_ops = ["tcgen05.cp.cta_group::1.128x256b"] * expected_count
    assert _assert_exact_cp_ptx_llir_match(compiled, expected_ops)


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

    expected_count = (M * N) // 1024
    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.128x256b"] * expected_count)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("M,N,swizzle,expected_count", CP_LINEAR_NO_SCALES_CASES)
def test_tmem_runtime_matrix_cp_no_scales_linear(M, N, swizzle, expected_count):
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout(M, N)

    compiled = tmem_copy_no_scales_linear_kernel[(1, )](inp, out, layout, M, N, swizzle, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.128x256b"] * expected_count)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype,M,N,swizzle,expected_count", CP_LINEAR_NO_SCALES_32BIT_DTYPE_CASES)
def test_tmem_runtime_matrix_cp_no_scales_linear_32bit_dtypes(dtype_name, torch_dtype, M, N, swizzle, expected_count):
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N).to(torch_dtype)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout(M, N)

    compiled = tmem_copy_no_scales_linear_kernel[(1, )](inp, out, layout, M, N, swizzle, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.128x256b"] * expected_count)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("M", CP_INDEXED_VIEW_128X128_CASES)
def test_tmem_runtime_matrix_cp_no_scales_indexed_view_canonicalized(M):
    N = 4
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_copy_no_scales_indexed_view_kernel[(1, )](inp, out, M, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.128x128b"] * (M // 128))
    ttgir = compiled.asm["ttgir"]
    assert "ttng.tmem_physical_layout" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "tensor_memory_encoding" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("name,torch_dtype,M,N,smem_layout,expected_count,expected_opcode", CP_LINEAR_INDEXED_VIEW_CASES)
def test_tmem_runtime_matrix_cp_no_scales_linear_indexed_view(name, torch_dtype, M, N, smem_layout, expected_count,
                                                              expected_opcode):
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)

    lifted_layout = _lift_tmem_layout(_make_tmem_linear_layout(M, N), [2])
    compiled = tmem_copy_no_scales_linear_indexed_view_kernel[(1, )](
        inp, out, lifted_layout, smem_layout, M, N, num_warps=4
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_exact_cp_ptx_llir_match(compiled, [expected_opcode] * expected_count)
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "tensor_memory_encoding" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_linear_indexed_view_full_128x256_reports_tmem_oor():
    M = 128
    N = 256
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty_like(inp)
    lifted_layout = _lift_tmem_layout(_make_tmem_linear_layout(M, N), [2])
    smem_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)

    with pytest.raises(triton.runtime.errors.OutOfResources, match="tensor memory"):
        tmem_copy_no_scales_linear_indexed_view_kernel[(1, )](
            inp, out, lifted_layout, smem_layout, M, N, num_warps=4
        )


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,M,N,swizzle,expected_count,expected_opcode",
    CP_LINEAR_SUBSLICE_VIEW_CASES,
)
def test_tmem_runtime_matrix_cp_no_scales_linear_subslice_view(
    dtype_name, torch_dtype, M, N, swizzle, expected_count, expected_opcode
):
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)

    parent_layout = _make_tmem_linear_layout(M, 2 * N)
    compiled = tmem_copy_no_scales_linear_subslice_view_kernel[(1, )](
        inp, out, parent_layout, M, N, swizzle, num_warps=4
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_exact_cp_ptx_llir_match(compiled, [expected_opcode] * expected_count)
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttng.tmem_subslice" not in ttgir




@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive():
    M = 128
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout()
    tmem_layout = _make_tmem_copy_warpx2_tmem_layout()
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_copy_no_scales_warpx2_candidate_kernel[(1, )](
        inp, out, shared_layout, tmem_layout, num_warps=4
    )

    expected = _expected_tmem_copy_warpx2_01_23_output(inp)
    assert not torch.equal(out, inp)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    _assert_exact_cp_ptx_llir_match(
        compiled,
        ["tcgen05.cp.cta_group::1.warpx2::01_23.64x128b"],
    )
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_candidate_positive():
    M = 128
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout()
    tmem_layout = _make_tmem_copy_warpx2_tmem_layout_02_13()
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_copy_no_scales_warpx2_candidate_kernel[(1, )](
        inp, out, shared_layout, tmem_layout, num_warps=4
    )

    expected = _expected_tmem_copy_warpx2_02_13_output(inp)
    assert not torch.equal(out, inp)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    _assert_exact_cp_ptx_llir_match(
        compiled,
        ["tcgen05.cp.cta_group::1.warpx2::02_13.64x128b"],
    )
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_canonical_codegen():
    M = 128
    N = 4
    shared_layout = _make_tmem_copy_128x128_shared_layout()
    tmem_layout = _make_tmem_copy_warpx2_tmem_layout()
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty((1, ), device="cuda", dtype=torch.int32)

    compiled = tmem_copy_no_scales_warpx2_codegen_kernel[(1, )](
        inp, out, shared_layout, tmem_layout, num_warps=4
    )

    assert int(out.item()) == 0
    _assert_exact_cp_ptx_llir_match(
        compiled,
        ["tcgen05.cp.cta_group::1.warpx2::01_23.64x128b"],
    )
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_twocta_codegen():
    child = textwrap.dedent(f"""
        import torch
        from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLayout
        from python.test.gluon.test_tmem_runtime_matrix import (
            _assert_exact_cp_ptx_llir_match,
            _make_2cta_cga_layout,
            _make_tmem_linear_layout_mmav5_twocta,
            tmem_copy_no_scales_twocta_kernel,
        )

        cases = {CP_NO_SCALES_TWOCTA_CASES!r}
        for layout_kind, N, swizzle, expected_count in cases:
            M = 256
            cga_layout = _make_2cta_cga_layout((2, 1), (2, 1), (1, 0), 0)
            if layout_kind == "linear":
                layout = _make_tmem_linear_layout_mmav5_twocta(M, N)
            else:
                layout = TensorMemoryLayout(block=(128, N), col_stride=1, cga_layout=cga_layout, two_ctas=True)

            inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
            out = torch.empty_like(inp)
            compiled = tmem_copy_no_scales_twocta_kernel[(1, )](
                inp,
                out,
                layout,
                tuple(tuple(basis) for basis in cga_layout),
                M,
                N,
                swizzle,
                num_ctas=2,
                num_warps=4,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(out, inp, atol=0, rtol=0)

            _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::2.128x256b"] * expected_count)
            ptx = compiled.asm["ptx"]
            llir = compiled.asm["llir"]
            first_cp_ptx = ptx.index("tcgen05.cp.cta_group::2.128x256b")
            first_cp_llir = llir.index("tcgen05.cp.cta_group::2.128x256b")

            assert ptx.count("tcgen05.commit.cta_group::2") == 1
            assert llir.count("tcgen05.commit.cta_group::2") == 1
            assert "tcgen05.commit.cta_group::1" not in ptx
            assert "tcgen05.commit.cta_group::1" not in llir
            assert "tcgen05.cp.cta_group::1" not in ptx
            assert ptx.count("fence.proxy.async.shared::cluster") == 1
            assert ptx.count("barrier.cluster.arrive.aligned;") == 2
            assert ptx.count("barrier.cluster.wait.aligned;") == 2
            assert ptx.index("barrier.cluster.arrive.aligned") < ptx.index("barrier.cluster.wait.aligned") < first_cp_ptx
            assert "barrier.cluster.arrive.relaxed.aligned" not in ptx
            assert llir.count("llvm.nvvm.fence.proxy.async.shared_cluster") == 2
            assert llir.count("llvm.nvvm.barrier.cluster.arrive.aligned") == 3
            assert llir.count("llvm.nvvm.barrier.cluster.wait.aligned") == 3
            assert llir.index("llvm.nvvm.barrier.cluster.arrive.aligned") < llir.index("llvm.nvvm.barrier.cluster.wait.aligned") < first_cp_llir
            assert "llvm.nvvm.barrier.cluster.arrive.relaxed.aligned" not in llir
            if layout_kind == "linear":
                assert "tensor_memory_linear" in compiled.asm["ttgir"]
    """)

    def run_child():
        with tempfile.TemporaryDirectory(prefix="tmem-twocta-cache-") as cache_dir:
            env = os.environ.copy()
            pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = "python:." if not pythonpath else f"python:.:{pythonpath}"
            env["TRITON_CACHE_DIR"] = cache_dir
            env.setdefault("CUDA_VISIBLE_DEVICES", "0")
            return subprocess.run(
                [sys.executable, "-c", child],
                cwd=os.getcwd(),
                env=env,
                text=True,
                capture_output=True,
            )

    completed = run_child()
    # Work around an intermittent libtriton import corruption race observed
    # when launching many short-lived child processes from the runtime matrix.
    if completed.returncode != 0 and (
        "libtriton.so: file too short" in completed.stderr
        or "libtriton.so: invalid ELF header" in completed.stderr
    ):
        completed = run_child()
    assert completed.returncode == 0, completed.stdout + completed.stderr



@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype,M,N,swizzle", CP_LINEAR_NO_SCALES_SUBWORD_UNSUPPORTED_CASES)
def test_tmem_runtime_matrix_cp_no_scales_linear_subword_dtypes_report_clean_error(dtype_name, torch_dtype, M, N,
                                                                                    swizzle, capfd):
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N).to(torch_dtype)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout(M, N)

    with pytest.raises(RuntimeError) as excinfo:
        tmem_copy_no_scales_linear_kernel[(1, )](inp, out, layout, M, N, swizzle, num_warps=4)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "Source element type should be 32-bit." in text
    assert "error encountered during parsing" in str(excinfo.value)
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("name,layout", CP_LINEAR_EXOTIC_UNSUPPORTED_CASES)
def test_tmem_runtime_matrix_cp_no_scales_linear_exotic_reports_clean_unsupported(name, layout, capfd):
    m = n = 128
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(Exception) as excinfo:
        tmem_copy_no_scales_linear_kernel[(1, )](inp, out, layout, m, n, 32, num_warps=4)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert (
        "does not match any recognized tcgen05.copy family for non-scales tensor memory copies" in text
        or "could not synthesize a compatible shared-memory descriptor plan for it" in text
    )
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("row_perm_kind,col_perm_kind", CP_LINEAR_PERMUTED_UNSUPPORTED_CASES)
def test_tmem_runtime_matrix_cp_no_scales_linear_rowcol_permuted_reports_clean_unsupported(
    row_perm_kind, col_perm_kind, capfd
):
    m = n = 128
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout_permuted(m, n, row_perm_kind, col_perm_kind)

    with pytest.raises(Exception) as excinfo:
        tmem_copy_no_scales_linear_kernel[(1, )](inp, out, layout, m, n, 32, num_warps=4)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert (
        "does not match any recognized tcgen05.copy family for non-scales tensor memory copies" in text
        or "could not synthesize a compatible shared-memory descriptor plan for it" in text
    )
    assert (
        "ascending physical row order" in text
        or "ascending physical column order" in text
    )
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_linear_unsupported_shape_reports_clean_error(capfd):
    m, n, swizzle = 256, 128, 32
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout(m, n)

    with pytest.raises(Exception) as excinfo:
        tmem_copy_no_scales_linear_kernel[(1, )](inp, out, layout, m, n, swizzle, num_warps=4)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert (
        "does not match any recognized tcgen05.copy family for non-scales tensor memory copies" in text
        or "could not synthesize a compatible shared-memory descriptor plan for it" in text
        or "supported TMEM register layout" in text
    )
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_kind,M", CP_NO_SCALES_128X128_CASES)
def test_tmem_runtime_matrix_cp_128x128(layout_kind, M):
    N = 4
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N)
    out = torch.empty_like(inp)
    if layout_kind == "legacy":
        layout = TensorMemoryLayout((M, N), col_stride=1)
    else:
        layout = _make_tmem_linear_layout(M, N)

    compiled = tmem_copy_128x128_kernel[(1, )](inp, out, M, layout, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.128x128b"] * (M // 128))
    if layout_kind == "linear":
        assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_scales_warpx4():
    smem_h, smem_w = 64, 16
    inp = torch.randint(size=(smem_h, smem_w), low=-100, high=100, dtype=torch.int8, device="cuda")
    out = torch.empty_like(inp)

    compiled = tmem_copy_scales_warpx4_kernel[(1, )](inp, out)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.warpx4.32x128b"] * 2)
    assert "ttg.memdesc_reinterpret" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,num_ctas,acc_layout_kind", CP_SCALES_WARPX4_SCALED_MMA_CASES)
def test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix(a_format, b_format, num_ctas, acc_layout_kind):
    block_m = 256 if num_ctas == 2 else 128
    block_n = 128
    block_k = 128
    m, n, k = block_m, block_n, block_k
    vec_size = 16 if a_format == "nvfp4" else 32

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    a_scale = swizzle_scales_packed_block(a_scale, vec_size)
    b_scale = swizzle_scales_packed_block(b_scale, vec_size)

    out, compiled = mma_scaled_tcgen05_copy(
        a,
        b,
        a_scale,
        b_scale,
        vec_size,
        block_m,
        block_n,
        block_k,
        num_ctas=num_ctas,
        multicast=False,
        acc_layout_kind=acc_layout_kind,
    )
    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    expected = _expected_scaled_cp_opcode(num_ctas)
    expected_count = 64 // vec_size
    _assert_exact_cp_ptx_llir_match(compiled, [expected] * expected_count)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("block_n,block_k,multicast,num_ctas,acc_layout_kind", CP_SCALES_WARPX4_GEOMETRY_CASES)
def test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_geometry_sweep(
    block_n, block_k, multicast, num_ctas, acc_layout_kind
):
    a_format = "mxfp8"
    b_format = "mxfp8"
    block_m = 256 if num_ctas == 2 else 128
    m, n, k = block_m, block_n, block_k
    vec_size = 32

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    a_scale = swizzle_scales_packed_block(a_scale, vec_size)
    b_scale = swizzle_scales_packed_block(b_scale, vec_size)

    out, compiled = mma_scaled_tcgen05_copy(
        a,
        b,
        a_scale,
        b_scale,
        vec_size,
        block_m,
        block_n,
        block_k,
        num_ctas=num_ctas,
        multicast=multicast,
        acc_layout_kind=acc_layout_kind,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    expected = _expected_scaled_cp_opcode(num_ctas)
    cp_ops = _assert_exact_cp_ptx_llir_match(compiled)
    assert cp_ops
    assert all(op == expected for op in cp_ops)


MMA_CASES = [
    ("legacy_no_acc", TensorMemoryLayout((128, 128), col_stride=1), False),
    ("legacy_use_acc", TensorMemoryLayout((128, 128), col_stride=1), True),
    ("linear_no_acc", _make_tmem_linear_layout(128, 128), False),
    ("linear_use_acc", _make_tmem_linear_layout(128, 128), True),
]

MMA_PLAIN_KIND_CASES = [
    (kind, acc_layout_kind)
    for kind, acc_layout_kind in product(("tf32", "bf16", "f8e5m2", "f8e4m3"), ("legacy", "linear"))
]

MMA_TWOCTA_CASES = [
    ("legacy", "legacy"),
    ("linear", "linear"),
]

MMA_TWOCTA_PLAIN_KIND_CASES = [
    (kind, acc_layout_kind)
    for kind, acc_layout_kind in product(("tf32", "bf16", "f8e5m2", "f8e4m3"), ("legacy", "linear"))
]

MMA_TILE_PERMUTED_CASES = [
    (128, 32),
    (256, 64),
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
@pytest.mark.parametrize("kind,acc_layout_kind", MMA_PLAIN_KIND_CASES)
def test_tmem_runtime_matrix_mma_plain_kinds_with_linear_acc(kind, acc_layout_kind):
    m = n = 128
    k = 32
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1])
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    acc_layout = TensorMemoryLayout((m, n), col_stride=1) if acc_layout_kind == "legacy" else _make_tmem_linear_layout(m, n)

    a, b, shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_plain_kind_inputs(kind, m, n, k)
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)

    compiled = mma_kernel[(1, )](
        a,
        b,
        out,
        m,
        n,
        k,
        block_layout_a,
        block_layout_b,
        (),
        acc_layout,
        shared_layout_a,
        shared_layout_b,
        ttgl.float32,
        False,
        True,
        num_warps=4,
    )

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)

    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_ops
    assert ptx_ops == llir_ops
    assert all(op == expected_kind for op in ptx_ops)
    assert "tcgen05.commit.cta_group::1" in compiled.asm["ptx"]
    assert "tcgen05.commit.cta_group::1" in compiled.asm["llir"]
    if acc_layout_kind == "linear":
        assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("acc_layout_kind", ("legacy", "linear"))
def test_tmem_runtime_matrix_mma_i8_reports_clean_error(acc_layout_kind, capfd):
    m = n = 128
    k = 32
    a = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (k, n), device="cuda", dtype=torch.int8)
    out = torch.empty((m, n), device="cuda", dtype=torch.int32)

    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1])
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    shared_layout_a = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=False, element_bitwidth=8, rank=2)
    shared_layout_b = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=True, element_bitwidth=8, rank=2)
    acc_layout = TensorMemoryLayout((m, n), col_stride=1) if acc_layout_kind == "legacy" else _make_tmem_linear_layout(m, n)

    with pytest.raises(Exception) as excinfo:
        mma_kernel[(1, )](
            a,
            b,
            out,
            m,
            n,
            k,
            block_layout_a,
            block_layout_b,
            (),
            acc_layout,
            shared_layout_a,
            shared_layout_b,
            ttgl.int32,
            False,
            True,
            num_warps=4,
        )

    captured = capfd.readouterr()
    msg = str(excinfo.value) + captured.err + captured.out
    assert "direct tcgen05_mma kind::i8 is not supported on sm_" in msg
    assert "current Blackwell lowering" in msg
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("name,layout", MMA_EXOTIC_UNSUPPORTED_CASES)
def test_tmem_runtime_matrix_mma_exotic_layout_reports_clean_unsupported(name, layout, capfd):
    m, n, k = 128, 128, 32
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    c = torch.randn((m, n), dtype=torch.float32, device="cuda")
    out = torch.empty_like(c)

    with pytest.raises(Exception) as excinfo:
        tmem_mma_kernel[(1, )](a, b, c, out, layout, False, num_warps=4)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "must have a MMAv5-compatible tensor memory layout" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("row_perm_kind,col_perm_kind", MMA_ROWCOL_PERMUTED_CASES)
def test_tmem_runtime_matrix_mma_rowcol_permuted_layout_reports_clean_unsupported(
    row_perm_kind, col_perm_kind, capfd
):
    m, n, k = 128, 128, 32
    layout = _make_tmem_linear_layout_permuted(m, n, row_perm_kind, col_perm_kind)
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    c = torch.randn((m, n), dtype=torch.float32, device="cuda")
    out = torch.empty_like(c)

    with pytest.raises(Exception) as excinfo:
        tmem_mma_kernel[(1, )](a, b, c, out, layout, False, num_warps=4)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "must have a MMAv5-compatible tensor memory layout" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


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


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,acc_layout_kind", MMA_TWOCTA_PLAIN_KIND_CASES)
def test_tmem_runtime_matrix_mma_twocta_plain_kinds(kind, acc_layout_kind):
    ctas_per_cga = [2, 1]
    ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]
    cta_order = [1, 0]
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)
    cga_layout_c_arg = tuple(tuple(basis) for basis in cga_layout_c)

    block_m, block_n, block_k = 256, 128, 32
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1], cga_layout=cga_layout_a)
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0], cga_layout=cga_layout_b)

    if acc_layout_kind == "legacy":
        acc_layout = TensorMemoryLayout(
            block=(128, block_n // ctas_per_cga[1]),
            col_stride=1,
            two_ctas=True,
            cga_layout=cga_layout_c,
        )
    else:
        acc_layout = _make_tmem_linear_layout_mmav5_twocta(block_m, block_n)

    a, b, shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_twocta_plain_kind_inputs(
        kind, block_m, block_n, block_k, cga_layout_a, cga_layout_b
    )
    out = torch.empty((block_m, block_n), device="cuda", dtype=torch.float32)

    compiled = mma_kernel[(1,)](
        a,
        b,
        out,
        block_m,
        block_n,
        block_k,
        block_layout_a,
        block_layout_b,
        cga_layout_c_arg,
        acc_layout,
        shared_layout_a,
        shared_layout_b,
        ttgl.float32,
        False,
        True,
        num_warps=4,
        num_ctas=2,
    )

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)

    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_ops
    assert ptx_ops == llir_ops
    assert all(op == expected_kind for op in ptx_ops)
    assert "tcgen05.commit.cta_group::2" in compiled.asm["ptx"]
    assert "tcgen05.commit.cta_group::2" in compiled.asm["llir"]
    assert "two_ctas" in compiled.asm["ttgir"]
    if acc_layout_kind == "linear":
        assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("name,parent_layout,layout_token", MMA_INDEXED_ACC_CASES)
def test_tmem_runtime_matrix_mma_indexed_acc_view(name, parent_layout, layout_token):
    m = n = 128
    k = 32
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_indexed_acc_kernel[(1, )](a, b, out, parent_layout, num_warps=4)

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    torch.testing.assert_close(out.to(torch.float32), ref, atol=1e-1, rtol=8e-2)

    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_ops
    assert ptx_ops == llir_ops
    assert all(op == "tcgen05.mma.cta_group::1.kind::f16" for op in ptx_ops)
    assert "ttg.memdesc_index" in compiled.asm["ttgir"]
    assert layout_token in compiled.asm["ttgir"]
    if name == "linear_parent":
        assert "tensor_memory_encoding" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n,tile_n", MMA_TILE_PERMUTED_CASES)
def test_tmem_runtime_matrix_mma_acc_tile_permuted(n, tile_n):
    m, k = 128, 32
    layout = _make_tmem_linear_layout_tile_permuted(m, n, tile_n)
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")
    block_layout_a = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    block_layout_b = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    shared_layout_a = ttgl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16, rank=2)
    shared_layout_b = ttgl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16, rank=2)

    compiled = mma_kernel[(1, )](
        a,
        b,
        out,
        m,
        n,
        k,
        block_layout_a,
        block_layout_b,
        (),
        layout,
        shared_layout_a,
        shared_layout_b,
        ttgl.float32,
        False,
        True,
        num_warps=4,
    )

    expected = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    torch.testing.assert_close(out.to(torch.float32), expected, atol=1e-1, rtol=8e-2)

    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_ops == llir_ops
    assert ptx_ops
    assert all(op == "tcgen05.mma.cta_group::1.kind::f16" for op in ptx_ops)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_lhs_tile_permuted():
    m = n = 128
    k = 256
    lhs_layout = _make_tmem_linear_layout_tile_permuted(m, k, 64)
    acc_layout = _make_tmem_linear_layout(m, n)
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_lhs_kernel[(1, )](
        a, b, out, lhs_layout, acc_layout, num_warps=4
    )

    expected = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    torch.testing.assert_close(out, expected, atol=1e-1, rtol=8e-2)

    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_ops == llir_ops
    assert ptx_ops
    assert all(op == "tcgen05.mma.cta_group::1.kind::f16" for op in ptx_ops)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_lhs_subslice_view():
    m = n = 128
    k = 32
    parent_layout = _make_tmem_linear_layout(m, 2 * k)
    acc_layout = _make_tmem_linear_layout(m, n)
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_lhs_subslice_kernel[(1, )](
        a, b, out, parent_layout, acc_layout, num_warps=4
    )

    expected = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    torch.testing.assert_close(out, expected, atol=1e-1, rtol=8e-2)

    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_ops == llir_ops
    assert ptx_ops
    assert all(op == "tcgen05.mma.cta_group::1.kind::f16" for op in ptx_ops)
    assert "ttg.memdesc_subslice" in compiled.asm["ttgir"]
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


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


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_scaled_acc_blockn64_direct_layout():
    m = n = k = 128
    layout = TensorMemoryLayout((128, 64), col_stride=1)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")
    a = torch.randint(20, 40, (m, k), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    b = torch.randint(20, 40, (k, n), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    a_scale = torch.randint(64, 130, (m, k // 32), dtype=torch.uint8, device="cuda")
    b_scale = torch.randint(64, 130, (n, k // 32), dtype=torch.uint8, device="cuda")

    compiled = tmem_mma_scaled_layout_kernel[(1, )](out, m, n, k, a, b, a_scale, b_scale, layout, num_warps=4)

    a_ref = a.to(torch.float32) * _fp8e8m0_to_float32(a_scale).repeat_interleave(32, dim=1)
    b_ref = b.to(torch.float32) * _fp8e8m0_to_float32(b_scale).repeat_interleave(32, dim=1).T.contiguous()
    expected = torch.matmul(a_ref, b_ref)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    ptx_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_mma_ops == llir_mma_ops
    assert len(ptx_mma_ops) == 4
    assert all(op.startswith("tcgen05.mma.cta_group::1.kind::") for op in ptx_mma_ops)
    assert all("block_scale.scale_vec::" in op for op in ptx_mma_ops)
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_scaled_acc_blockn32_direct_layout():
    m = n = k = 128
    layout = TensorMemoryLayout((128, 32), col_stride=1)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")
    a = torch.randint(20, 40, (m, k), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    b = torch.randint(20, 40, (k, n), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    a_scale = torch.randint(64, 130, (m, k // 32), dtype=torch.uint8, device="cuda")
    b_scale = torch.randint(64, 130, (n, k // 32), dtype=torch.uint8, device="cuda")

    compiled = tmem_mma_scaled_layout_kernel[(1, )](out, m, n, k, a, b, a_scale, b_scale, layout, num_warps=4)

    a_ref = a.to(torch.float32) * _fp8e8m0_to_float32(a_scale).repeat_interleave(32, dim=1)
    b_ref = b.to(torch.float32) * _fp8e8m0_to_float32(b_scale).repeat_interleave(32, dim=1).T.contiguous()
    expected = torch.matmul(a_ref, b_ref)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    ptx_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_mma_ops == llir_mma_ops
    assert len(ptx_mma_ops) == 4
    assert all(op.startswith("tcgen05.mma.cta_group::1.kind::") for op in ptx_mma_ops)
    assert all("block_scale.scale_vec::" in op for op in ptx_mma_ops)
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n", (32, 64))
def test_tmem_runtime_matrix_mma_scaled_acc_subslice_view(n):
    m = k = 128
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")
    a = torch.randint(20, 40, (m, k), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    b = torch.randint(20, 40, (k, n), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    a_scale = torch.randint(64, 130, (m, k // 32), dtype=torch.uint8, device="cuda")
    b_scale = torch.randint(64, 130, (n, k // 32), dtype=torch.uint8, device="cuda")

    compiled = tmem_mma_scaled_acc_subslice_kernel[(1,)](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        _make_tmem_linear_layout(m, 128),
        64,
        num_warps=4,
    )

    a_ref = a.to(torch.float32) * _fp8e8m0_to_float32(a_scale).repeat_interleave(32, dim=1)
    b_ref = b.to(torch.float32) * _fp8e8m0_to_float32(b_scale).repeat_interleave(32, dim=1).T.contiguous()
    expected = torch.matmul(a_ref, b_ref)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    ptx_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_mma_ops == llir_mma_ops
    assert len(ptx_mma_ops) == 4
    assert all(op.startswith("tcgen05.mma.cta_group::1.kind::") for op in ptx_mma_ops)
    assert all("block_scale.scale_vec::" in op for op in ptx_mma_ops)
    assert "ttg.memdesc_subslice" in compiled.asm["ttgir"]
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view():
    m = n = 128
    k = 64
    parent_layout = _make_tmem_linear_layout(m, 2 * k)
    acc_layout = TensorMemoryLayout((m, n), col_stride=1)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")
    a = torch.randint(20, 40, (m, k), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    b = torch.randint(20, 40, (k, n), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    a_scale = torch.randint(64, 130, (m, k // 32), dtype=torch.uint8, device="cuda")
    b_scale = torch.randint(64, 130, (n, k // 32), dtype=torch.uint8, device="cuda")

    compiled = tmem_mma_scaled_lhs_subslice_kernel[(1, )](
        out, m, n, k, a, b, a_scale, b_scale, parent_layout, acc_layout, num_warps=4
    )

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
    assert "ttg.memdesc_subslice" in compiled.asm["ttgir"]
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_64_direct_layout():
    m, n, k = 128, 256, 128
    layout = _make_tmem_linear_layout_tile_permuted(m, n, 64)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")
    a = torch.randint(20, 40, (m, k), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    b = torch.randint(20, 40, (k, n), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    a_scale = torch.randint(64, 130, (m, k // 32), dtype=torch.uint8, device="cuda")
    b_scale = torch.randint(64, 130, (n, k // 32), dtype=torch.uint8, device="cuda")

    compiled = tmem_mma_scaled_layout_kernel[(1, )](out, m, n, k, a, b, a_scale, b_scale, layout, num_warps=4)

    a_ref = a.to(torch.float32) * _fp8e8m0_to_float32(a_scale).repeat_interleave(32, dim=1)
    b_ref = b.to(torch.float32) * _fp8e8m0_to_float32(b_scale).repeat_interleave(32, dim=1).T.contiguous()
    expected = torch.matmul(a_ref, b_ref)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    ptx_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_mma_ops == llir_mma_ops
    assert len(ptx_mma_ops) == 16
    assert all(op.startswith("tcgen05.mma.cta_group::1.kind::") for op in ptx_mma_ops)
    assert all("block_scale.scale_vec::" in op for op in ptx_mma_ops)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_repeated_n32_reports_clean_unsupported(capfd):
    m = n = k = 128
    layout = _make_tmem_linear_layout_tile_permuted(m, n, 32)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")
    a = torch.randint(20, 40, (m, k), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    b = torch.randint(20, 40, (k, n), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    a_scale = torch.randint(64, 130, (m, k // 32), dtype=torch.uint8, device="cuda")
    b_scale = torch.randint(64, 130, (n, k // 32), dtype=torch.uint8, device="cuda")

    with pytest.raises(Exception) as excinfo:
        tmem_mma_scaled_layout_kernel[(1, )](out, m, n, k, a, b, a_scale, b_scale, layout, num_warps=4)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "direct block-scaled MMAv5 does not support repeated N=32 instructions along N" in text
    assert "matrix-B scale fragments at 64-column alignment" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text
