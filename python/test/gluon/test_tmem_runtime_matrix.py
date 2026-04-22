import math
import re
from dataclasses import replace
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
from tmem_test_utils import (
    _expected_scaled_cp_opcode,
    _expected_scaled_mma_opcode,
    make_operand_descriptor,
    make_output_descriptor,
    make_scales_descriptor,
    mma_kernel,
    _run_tmem_reduction_case,
    mma_scaled_tcgen05_copy,
    random_quantized_tensor,
    swizzle_scales_packed_block,
    unswizzle_scales_shared_memory,
)

TMEM_ENCODING_LAYOUT_KIND = "legacy"
TMEM_LINEAR_LAYOUT_KIND = "linear"
TMEM_LAYOUT_KIND_CASES = (TMEM_ENCODING_LAYOUT_KIND, TMEM_LINEAR_LAYOUT_KIND)

# Runtime-matrix nodeids keep the historical "legacy" spelling because old
# manifests and logs refer to those cases. In these tests it means the public
# TensorMemoryLayout frontend encoding; backend support must still canonicalize
# it before applying TMEM layout policy.
def _uses_tmem_encoding_layout(layout_kind):
    return layout_kind == TMEM_ENCODING_LAYOUT_KIND


def _make_tmem_linear_layout(m, n):
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(int(math.log2(m)))],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))],
        shape=[m, n],
    )


def _make_tmem_acc_layout(layout_kind, m, n):
    if _uses_tmem_encoding_layout(layout_kind):
        return TensorMemoryLayout((m, n), col_stride=1)
    assert layout_kind == TMEM_LINEAR_LAYOUT_KIND
    return _make_tmem_linear_layout(m, n)


def _make_tmem_copy_4x256b_refresh_layout(two_ctas=False):
    kwargs = dict(
        rows=[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]],
        cols=[[1, 0], [2, 0], [0, 4]],
        shape=[8, 8] if two_ctas else [4, 8],
    )
    if two_ctas:
        kwargs["block_bases"] = [[4, 0]]
        kwargs["two_ctas"] = True
    return TensorMemoryLinearLayout(**kwargs)


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


def _make_tmem_linear_layout_tile_selector_permuted(m, n, tile_n, kind):
    assert m >= 1 and (m & (m - 1)) == 0
    assert n >= 1 and (n & (n - 1)) == 0
    assert tile_n >= 1 and (tile_n & (tile_n - 1)) == 0
    assert n >= 2 * tile_n, "tile-selector permutation requires at least two tiles"

    tile_bits = int(math.log2(tile_n))
    col_bits = [1 << i for i in range(int(math.log2(n)))]
    col_bits = col_bits[:tile_bits] + _permute_pow2_bases_by_kind(col_bits[tile_bits:], kind)
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


def _make_tmem_linear_layout_ld_red_legacy_equivalent_256(n):
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(7)],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))] + [[128, 0]],
        shape=[256, n],
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


def _make_tmem_linear_layout_noncanonical_twocta_block(m, n):
    assert m == 256
    assert n >= 1 and (n & (n - 1)) == 0
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(7)],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))],
        block_bases=[[192, 0]],
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
    # classifies as warpx4 under TensorMemoryScalesLayout because scales expose
    # broadcast row bases, then reaches a clean unsupported descriptor-plan path
    # rather than a live tcgen05.copy.warpx2 lowering.
    return ttgl.SharedLinearLayout(
        offset_bases=[[32, 0], [0, 1], [1, 0], [0, 2], [0, 4], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]]
    )


def _make_scales_shared_layout_warpx2_no_scales_like_column_tail():
    return ttgl.SharedLinearLayout(
        offset_bases=[[32, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1], [0, 2], [0, 4], [0, 8]]
    )


def _make_scales_shared_layout_warpx2_row32_after_columns():
    return ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [32, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]]
    )


def _make_scales_shared_layout_warpx2_row32_after_low_rows():
    return ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 8]]
    )


def _make_scales_shared_layout_warpx2_column_first_row_tail():
    return ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]]
    )


def _make_scales_shared_layout_parent_row_subslice_probe():
    return ttgl.SharedLinearLayout(
        offset_bases=[
            [0, 1], [0, 2], [0, 4], [0, 8], [64, 0], [1, 0],
            [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]
        ],
        alignment=16,
    )


def _make_tmem_copy_warpx2_shared_layout():
    return ttgl.SharedLinearLayout(
        offset_bases=[[32, 0], [0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [64, 0]],
        alignment=16,
    )


def _make_tmem_copy_dense_shared_layout(m, n):
    return ttgl.SharedLinearLayout(
        offset_bases=[
            *[[0, 1 << i] for i in range(int(math.log2(n)))],
            *[[1 << i, 0] for i in range(int(math.log2(m)))],
        ],
        alignment=16,
    )


def _make_tmem_copy_warpx2_tmem_layout():
    return TensorMemoryLinearLayout(
        rows=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0], [32, 0]],
        cols=[[0, 1], [0, 2]],
        shape=[128, 4],
    )


def _make_tmem_copy_warpx2_tmem_layout_row_permuted():
    return TensorMemoryLinearLayout(
        rows=[[2, 0], [1, 0], [4, 0], [8, 0], [16, 0], [0, 0], [32, 0]],
        cols=[[0, 1], [0, 2]],
        shape=[128, 4],
    )


def _make_tmem_copy_warpx2_parent_tmem_layout():
    return TensorMemoryLinearLayout(
        rows=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0], [32, 0]],
        cols=[[0, 1], [0, 2], [0, 4]],
        shape=[128, 8],
    )


def _make_tmem_copy_warpx2_parent_tmem_layout_02_13():
    return TensorMemoryLinearLayout(
        rows=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 0]],
        cols=[[0, 1], [0, 2], [0, 4]],
        shape=[128, 8],
    )


def _make_tmem_copy_warpx2_shared_layout_twocta():
    return ttgl.SharedLinearLayout(
        offset_bases=[[32, 0], [0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [64, 0]],
        block_bases=[[128, 0]],
        alignment=16,
    )


def _make_tmem_copy_warpx2_tmem_layout_twocta():
    return TensorMemoryLinearLayout(
        rows=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0], [32, 0]],
        cols=[[0, 1], [0, 2]],
        block_bases=[[128, 0]],
        shape=[256, 4],
        two_ctas=True,
    )


def _make_tmem_copy_warpx2_parent_tmem_layout_twocta():
    return TensorMemoryLinearLayout(
        rows=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0], [32, 0]],
        cols=[[0, 1], [0, 2], [0, 4]],
        block_bases=[[128, 0]],
        shape=[256, 8],
        two_ctas=True,
    )


def _make_tmem_copy_warpx2_parent_tmem_layout_02_13_twocta():
    return TensorMemoryLinearLayout(
        rows=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 0]],
        cols=[[0, 1], [0, 2], [0, 4]],
        block_bases=[[128, 0]],
        shape=[256, 8],
        two_ctas=True,
    )


def _make_tmem_copy_warpx2_tmem_layout_02_13_twocta():
    return TensorMemoryLinearLayout(
        rows=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 0]],
        cols=[[0, 1], [0, 2]],
        block_bases=[[128, 0]],
        shape=[256, 4],
        two_ctas=True,
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


def _expected_tmem_copy_warpx2_01_23_twocta_output(inp: torch.Tensor) -> torch.Tensor:
    assert tuple(inp.shape) == (256, 4)
    expected = torch.empty_like(inp)
    for row in range(inp.shape[0]):
        cta_base = 128 * (row // 128)
        local_row = row % 128
        src_base = cta_base + ((local_row % 32) // 2)
        src_col = (local_row // 64) + (0 if local_row % 2 == 0 else 2)
        expected[row, 0] = inp[src_base, src_col]
        expected[row, 1] = inp[src_base + 32, src_col]
        expected[row, 2] = inp[src_base + 16, src_col]
        expected[row, 3] = inp[src_base + 48, src_col]
    return expected


def _make_tmem_copy_128x128_shared_layout():
    return ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]],
        alignment=16,
    )


def _make_tmem_copy_128x128_shared_layout_twocta():
    return ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]],
        block_bases=[[128, 0]],
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
        r"(tcgen05\.cp(?:\.cta_group::\d+)?(?:\.warpx[24](?:::[^\s.;]+)*)?"
        r"\.\d+x\d+b(?:\.b8x16\.(?:b6x16_p32|b4x16_p64))?)"
    )
    return pattern.findall(asm)


def _assert_exact_cp_ptx_llir_match(compiled, expected_ops=None):
    ptx_ops = _extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_cp_opcodes(compiled.asm["llir"])
    assert ptx_ops == llir_ops
    if expected_ops is not None:
        assert ptx_ops == list(expected_ops)
    return ptx_ops


def _extract_tcgen05_commit_opcodes(asm: str):
    pattern = re.compile(
        r"(tcgen05\.commit\.cta_group::\d+\.mbarrier::arrive::one\.shared::cluster"
        r"(?:\.multicast::cluster)?\.b64)"
    )
    return pattern.findall(asm)


def _assert_exact_commit_ptx_llir_match(compiled, expected_ops=None):
    ptx_ops = _extract_tcgen05_commit_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_commit_opcodes(compiled.asm["llir"])
    assert ptx_ops == llir_ops
    if expected_ops is not None:
        assert ptx_ops == list(expected_ops)
    return ptx_ops


def _expected_commit_opcode(cta_group: int):
    if cta_group == 2:
        return "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
    assert cta_group == 1
    return "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"


def _assert_exact_tmem_lifetime_ptx_llir_match(compiled, cta_group: int, alloc_size: int,
                                               expect_cluster_sync: bool = False):
    alloc = f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"
    relinquish = f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"
    dealloc = f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"
    ptx = compiled.asm["ptx"]
    llir = compiled.asm["llir"]

    assert re.search(rf"{re.escape(alloc)} \[[^\]]+\], {alloc_size};", ptx)
    assert re.search(rf"{re.escape(alloc)} \[\$1\], {alloc_size};", llir)
    assert ptx.count(alloc) == 1
    assert llir.count(alloc) == 1

    assert ptx.count(relinquish) == 1
    assert llir.count(relinquish) == 1

    assert re.search(rf"{re.escape(dealloc)} [^;]+, {alloc_size};", ptx)
    assert re.search(rf"{re.escape(dealloc)} \$1, {alloc_size};", llir)
    assert ptx.count(dealloc) == 1
    assert llir.count(dealloc) == 1

    assert ptx.count("tcgen05.wait::st.sync.aligned;") == 1
    assert ptx.count("tcgen05.wait::ld.sync.aligned;") == 1
    assert llir.count("tail call void @llvm.nvvm.tcgen05.wait.st()") == 1
    assert llir.count("tail call void @llvm.nvvm.tcgen05.wait.ld()") == 1

    assert ptx.index(alloc) < ptx.index(relinquish)
    assert ptx.index("tcgen05.st.sync.aligned") < ptx.index("tcgen05.wait::st.sync.aligned")
    assert ptx.index("tcgen05.ld.sync.aligned") < ptx.index("tcgen05.wait::ld.sync.aligned")
    assert llir.index(alloc) < llir.index(relinquish)
    assert llir.index("tcgen05.st.sync.aligned") < llir.index("@llvm.nvvm.tcgen05.wait.st()")
    assert llir.index("tcgen05.ld.sync.aligned") < llir.index("@llvm.nvvm.tcgen05.wait.ld()")

    if expect_cluster_sync:
        assert (
            ptx.index("barrier.cluster.arrive.aligned") < ptx.index("barrier.cluster.wait.aligned") < ptx.index(dealloc)
        )
        assert (
            llir.index("@llvm.nvvm.barrier.cluster.arrive.aligned")
            < llir.index("@llvm.nvvm.barrier.cluster.wait.aligned") < llir.index(dealloc)
        )
    else:
        assert "barrier.cluster.arrive.aligned" not in ptx
        assert "@llvm.nvvm.barrier.cluster.arrive.aligned" not in llir



def _assert_exact_mma_ptx_llir_match(compiled, expected_ops=None):
    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_ops == llir_ops
    if expected_ops is not None:
        assert ptx_ops == list(expected_ops)
    return ptx_ops


def _assert_exact_mxf8f6f4_scaled_mma(compiled, expected_count=None):
    ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert ops
    if expected_count is not None:
        assert len(ops) == expected_count
    expected = _expected_scaled_mma_opcode("mxfp8", "mxfp8", 1)
    assert all(op == expected for op in ops)
    return ops


def _scaled_mma_operand_params(format_name):
    if format_name == "mxfp8":
        return 1, "e4m3"
    if format_name in ("mxfp4", "nvfp4"):
        return 2, "e2m1"
    raise ValueError(f"unsupported scaled MMA format: {format_name}")


def _expected_scaled_mma_acc_subslice_count(a_format, b_format):
    return 2 if a_format in ("mxfp4", "nvfp4") and b_format in ("mxfp4", "nvfp4") else 4


def _assert_clean_unsupported_descriptor_view(text: str, expected_text: str):
    assert expected_text in text
    assert "reshape or permute so TMEM columns stay contiguous" not in text


def _assert_clean_cta_per_cga_mismatch(text: str, layout_ctas: int, required_ctas: int):
    assert "Result has an invalid layout:" in text
    assert f"Layout has {layout_ctas} CTAs per CGA, but the context requires {required_ctas} CTAs per CGA." in text


def _assert_clean_tmem_oor(text: str, required: int, hardware_limit: int):
    assert "out of resource: tensor memory" in text
    assert f"Required: {required}" in text
    assert f"Hardware limit: {hardware_limit}" in text


def _extract_tcgen05_mma_opcodes(asm: str):
    pattern = re.compile(r"(tcgen05\.mma\.cta_group::\d+\.kind::[^\s;\"]+)")
    return pattern.findall(asm)


def _make_mma_plain_kind_inputs(kind: str, m: int, n: int, k: int):
    if kind == "f16":
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        shared_layout_a = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=False, element_bitwidth=16, rank=2)
        shared_layout_b = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=True, element_bitwidth=16, rank=2)
        expected_kind = "tcgen05.mma.cta_group::1.kind::f16"
        atol, rtol = 1e-1, 8e-2
    elif kind == "tf32":
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
    elif kind in ("f8e5m2", "f8e4m3"):
        fp8_dtype = torch.float8_e5m2 if kind == "f8e5m2" else torch.float8_e4m3fn
        a = torch.randint(20, 40, (m, k), device="cuda", dtype=torch.uint8).view(fp8_dtype)
        b = torch.randint(20, 40, (k, n), device="cuda", dtype=torch.uint8).view(fp8_dtype)
        shared_layout_a = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=False, element_bitwidth=8, rank=2)
        shared_layout_b = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=True, element_bitwidth=8, rank=2)
        expected_kind = "tcgen05.mma.cta_group::1.kind::f8f6f4"
        atol, rtol = 1e-1, 1e-1
    else:
        raise ValueError(f"unsupported MMA kind: {kind}")
    return a, b, shared_layout_a, shared_layout_b, expected_kind, atol, rtol


def _make_mma_twocta_plain_kind_inputs(kind: str, m: int, n: int, k: int, cga_layout_a, cga_layout_b):
    if kind == "f16":
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([m, k], ttgl.float16, cga_layout=cga_layout_a)
        shared_layout_b = ttgl.NVMMASharedLayout.get_default_for(
            [k, n], ttgl.float16, transposed=True, cga_layout=cga_layout_b
        )
        expected_kind = "tcgen05.mma.cta_group::2.kind::f16"
        atol, rtol = 1e-1, 8e-2
    elif kind == "tf32":
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
    elif kind in ("f8e5m2", "f8e4m3"):
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
    else:
        raise ValueError(f"unsupported MMA kind: {kind}")
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


def _expected_scales_ldst_descriptor_view_ops(root_shape: str, view_shape: str, offsets=(0, ), view_offsets=None):
    if view_offsets is None:
        view_offsets = offsets
    ops = []
    for offset in offsets:
        ops.append((f"tcgen05.st.sync.aligned.{root_shape}", offset))
    for offset in view_offsets:
        ops.append((f"tcgen05.ld.sync.aligned.{view_shape}", offset))
    for offset in view_offsets:
        ops.append((f"tcgen05.st.sync.aligned.{view_shape}", offset))
    for offset in offsets:
        ops.append((f"tcgen05.ld.sync.aligned.{root_shape}", offset))
    return ops


SCALES_LDST_N_SHARDED_VARIANT_WIDTHS = {
    "16x64b": 64,
    "16x128b": 128,
    "16x256b": 256,
}


def _scales_ldst_n_sharded_min_elements(instr_variant: str):
    return 4 * SCALES_LDST_N_SHARDED_VARIANT_WIDTHS[instr_variant]


def _expected_scales_ldst_n_sharded_ops(m: int, n: int, instr_variant: str):
    count = (m * n) // _scales_ldst_n_sharded_min_elements(instr_variant)
    return _expected_ldst_ops(f"{instr_variant}.x{count}.b32", [0, 0])


@gluon.jit
def tmem_ldst_variant_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr, N: ttgl.constexpr,
                             instr_variant: ttgl.constexpr):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    element_ty: ttgl.constexpr = in_ptr.dtype.element_ty
    tmem = allocate_tensor_memory(element_ty, [M, N], layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant=instr_variant)
    value = ttgl.load(in_ptr + offs)
    tmem.store(ttgl.convert_layout(value, reg_layout))
    value = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(value, reg_layout))


@gluon.jit
def tmem_ldst_auto_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr, N: ttgl.constexpr):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    element_ty: ttgl.constexpr = in_ptr.dtype.element_ty
    tmem = allocate_tensor_memory(element_ty, [M, N], layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()
    value = ttgl.load(in_ptr + offs)
    tmem.store(ttgl.convert_layout(value, reg_layout))
    value = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(value, reg_layout))


@gluon.jit
def tmem_alloc_source_init_kernel(in_ptr, out_ptr, layout: ttgl.constexpr):
    M: ttgl.constexpr = 128
    N: ttgl.constexpr = 128
    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 128], [32, 1], [4, 1], [0, 1])
    offs = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))[:, None] * N + ttgl.arange(
        0, N, ttgl.SliceLayout(0, reg_layout)
    )[None, :]
    value = ttgl.load(in_ptr + offs)
    tmem = allocate_tensor_memory(ttgl.float32, [M, N], layout, value=value)
    value = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, value)


@gluon.jit
def tmem_ldst_descriptor_chain_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr, N: ttgl.constexpr,
                                      instr_variant: ttgl.constexpr):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)
    element_ty: ttgl.constexpr = in_ptr.dtype.element_ty

    tmem = allocate_tensor_memory(element_ty, [2, M, N], layout)
    view = tmem.slice(1, 1, dim=0).index(0).reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
    view = view.permute([1, 0]).permute([1, 0])
    view = view.slice(0, M, dim=0).slice(0, N, dim=1)
    view = view.bitcast(element_ty, [M, N], layout)

    reg_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    view.store(ttgl.convert_layout(value, reg_layout))
    out = view.load(reg_layout)
    out = out + ttgl.full([M, N], 3, element_ty, layout=reg_layout)
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
def tmem_ldst_descriptor_rank5_small_roundtrip_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr,
                                                      N: ttgl.constexpr, instr_variant: ttgl.constexpr,
                                                      delta: ttgl.constexpr):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)
    element_ty: ttgl.constexpr = in_ptr.dtype.element_ty

    tmem = allocate_tensor_memory(element_ty, [1, 1, 2, M, N], layout)
    base = tmem.index(0).index(0).index(1)
    base_reg_layout: ttgl.constexpr = base.get_reg_layout(instr_variant=instr_variant)
    base.store(ttgl.convert_layout(value, base_reg_layout))

    alias = tmem.slice(0, 1, dim=0).index(0).slice(0, 1, dim=0).index(0).slice(1, 1, dim=0).index(0)
    alias = alias.reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
    alias = alias.permute([1, 0]).permute([1, 0])
    alias = alias.slice(0, M, dim=0).slice(0, N, dim=1)

    alias_reg_layout: ttgl.constexpr = alias.get_reg_layout(instr_variant=instr_variant)
    out = alias.load(alias_reg_layout)
    out = out + ttgl.full([M, N], delta, element_ty, layout=alias_reg_layout)
    alias.store(out)

    out = base.load(base_reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, base_reg_layout))


@gluon.jit
def tmem_ldst_descriptor_rank5_unit_parent_roundtrip_kernel(in_ptr, out_ptr, layout: ttgl.constexpr,
                                                           M: ttgl.constexpr, N: ttgl.constexpr,
                                                           instr_variant: ttgl.constexpr, delta: ttgl.constexpr):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)
    element_ty: ttgl.constexpr = in_ptr.dtype.element_ty

    tmem = allocate_tensor_memory(element_ty, [1, 1, 1, M, N], layout)
    base = tmem.index(0).index(0).index(0).reshape((M, N))
    base_reg_layout: ttgl.constexpr = base.get_reg_layout(instr_variant=instr_variant)
    base.store(ttgl.convert_layout(value, base_reg_layout))

    alias = tmem.slice(0, 1, dim=0).index(0).slice(0, 1, dim=0).index(0).slice(0, 1, dim=0).index(0)
    alias = alias.reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
    alias = alias.permute([1, 0]).permute([1, 0])
    alias = alias.slice(0, M, dim=0).slice(0, N, dim=1)

    alias_reg_layout: ttgl.constexpr = alias.get_reg_layout(instr_variant=instr_variant)
    out = alias.load(alias_reg_layout)
    out = out + ttgl.full([M, N], delta, element_ty, layout=alias_reg_layout)
    alias.store(out)

    out = base.load(base_reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, base_reg_layout))


@gluon.jit
def tmem_ldst_descriptor_higher_rank_index_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr,
                                                  N: ttgl.constexpr, instr_variant: ttgl.constexpr):
    element_ty: ttgl.constexpr = in_ptr.dtype.element_ty
    tmem = allocate_tensor_memory(element_ty, [2, M, N], layout)
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
    part0.store(part0_value + ttgl.full([M, N // 2], 5, element_ty, layout=part_layout))
    part1_value = part1.load(part_layout)
    part1.store(part1_value + ttgl.full([M, N // 2], 5, element_ty, layout=part_layout))

    out = full_view.load(full_reg_layout)
    ttgl.store(out_ptr + in_offs, out)


@gluon.jit
def tmem_ldst_descriptor_multidim_slice_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr,
                                               N: ttgl.constexpr, instr_variant: ttgl.constexpr):
    element_ty: ttgl.constexpr = in_ptr.dtype.element_ty
    tmem = allocate_tensor_memory(element_ty, [2, M, N], layout)
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
    part0.store(part0_value + ttgl.full([M, N // 2], 9, element_ty, layout=part0_layout))
    part1_value = part1.load(part1_layout)
    part1.store(part1_value + ttgl.full([M, N // 2], 9, element_ty, layout=part1_layout))

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
    element_ty: ttgl.constexpr = in_ptr.dtype.element_ty
    tmem = allocate_tensor_memory(element_ty, [2, M, N], layout)
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
    view.store(part_value + ttgl.full([M, N // 2], 7, element_ty, layout=part_layout))

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
def tmem_ldst_descriptor_direct_half_rows_positive_kernel(in_ptr, out_ptr, layout: ttgl.constexpr,
                                                          M: ttgl.constexpr, N: ttgl.constexpr,
                                                          instr_variant: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [M, N], layout)
    full_reg_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant=instr_variant)
    in_offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, full_reg_layout))
    in_offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, full_reg_layout))
    in_offs = in_offs_m[:, None] * N + in_offs_n[None, :]
    value = ttgl.load(in_ptr + in_offs)
    tmem.store(ttgl.convert_layout(value, full_reg_layout))

    view = tmem.slice(M // 2, M // 2, dim=0)
    part_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    part_value = view.load(part_layout)
    view.store(part_value + ttgl.full([M // 2, N], 17.0, ttgl.float32, layout=part_layout))

    out = tmem.load(full_reg_layout)
    ttgl.store(out_ptr + in_offs, out)


@gluon.jit
def tmem_ldst_direct_higher_rank_get_reg_layout_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr,
                                                       N: ttgl.constexpr, instr_variant: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant=instr_variant)
    b_layout: ttgl.constexpr = ttgl.SliceLayout(1, ttgl.SliceLayout(2, reg_layout))
    m_layout: ttgl.constexpr = ttgl.SliceLayout(0, ttgl.SliceLayout(2, reg_layout))
    n_layout: ttgl.constexpr = ttgl.SliceLayout(0, ttgl.SliceLayout(1, reg_layout))
    offs_b = ttgl.arange(0, 2, b_layout)[:, None, None]
    offs_m = ttgl.arange(0, M, m_layout)[None, :, None]
    offs_n = ttgl.arange(0, N, n_layout)[None, None, :]
    offs = offs_b * M * N + offs_m * N + offs_n
    value = ttgl.load(in_ptr + offs)
    tmem.store(ttgl.convert_layout(value, reg_layout))
    out = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, reg_layout))


@gluon.jit
def tmem_ldst_direct_higher_rank_load_kernel(out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr,
                                             N: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 128], [32, 1], [4, 1], [0, 1])
    _ = tmem.load(reg_layout)


@gluon.jit
def tmem_ldst_direct_higher_rank_store_kernel(out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr,
                                              N: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1, 1], [1, 32, 1], [4, 1, 1], [2, 1, 0])
    value = ttgl.full([2, M, N], 0.0, ttgl.float32, layout=reg_layout)
    tmem.store(value)


@gluon.jit
def tmem_ldst_direct_higher_rank_replay_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr,
                                               N: ttgl.constexpr):
    flat_m: ttgl.constexpr = 2 * M
    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
    flat = tmem.reshape((flat_m, N))
    flat_layout: ttgl.constexpr = flat.get_reg_layout()
    offs_m = ttgl.arange(0, flat_m, ttgl.SliceLayout(1, flat_layout))[:, None]
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, flat_layout))[None, :]
    offs = offs_m * N + offs_n
    value = ttgl.load(in_ptr + offs).reshape((2, M, N))
    tmem.store(value)
    out = tmem.load().reshape((flat_m, N))
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, flat_layout))


@gluon.jit
def tmem_ldst_direct_higher_rank_load_red_kernel(
    in_ptr, out_ptr, red_ptr, layout: ttgl.constexpr, red_layout: ttgl.constexpr, M: ttgl.constexpr,
    N: ttgl.constexpr
):
    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()
    b_layout: ttgl.constexpr = ttgl.SliceLayout(1, ttgl.SliceLayout(2, reg_layout))
    m_layout: ttgl.constexpr = ttgl.SliceLayout(0, ttgl.SliceLayout(2, reg_layout))
    n_layout: ttgl.constexpr = ttgl.SliceLayout(0, ttgl.SliceLayout(1, reg_layout))
    offs_b = ttgl.arange(0, 2, b_layout)[:, None, None]
    offs_m = ttgl.arange(0, M, m_layout)[None, :, None]
    offs_n = ttgl.arange(0, N, n_layout)[None, None, :]
    offs = offs_b * M * N + offs_m * N + offs_n
    value = ttgl.load(in_ptr + offs)
    tmem.store(ttgl.convert_layout(value, reg_layout))

    out, reduced = tmem.load_min()
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, reg_layout))

    red_b = ttgl.arange(0, 2, ttgl.SliceLayout(1, red_layout))[:, None]
    red_m = ttgl.arange(0, M, ttgl.SliceLayout(0, red_layout))[None, :]
    red_offs = red_b * M + red_m
    ttgl.store(red_ptr + red_offs, ttgl.convert_layout(reduced, red_layout))


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
    view = view.bitcast(ttgl.float32, [64, 32], reinterpret_layout)
    value = ttgl.full([64, 32], 0.0, ttgl.float32, layout=reg_layout)
    view.store(ttgl.convert_layout(value, reg_layout))
    _ = view.load(reg_layout)


@gluon.jit
def tmem_ldst_subword_variant_kernel(
    in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr, N: ttgl.constexpr,
    instr_variant: ttgl.constexpr
):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant=instr_variant)
    value = ttgl.load(in_ptr + offs)
    tmem.store(ttgl.convert_layout(value, reg_layout))
    value = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(value, reg_layout))


@gluon.jit
def tmem_ldst_x1_subword_twocta_descriptor_chain_kernel(
    in_ptr, out_ptr, layout: ttgl.constexpr, M: ttgl.constexpr, N: ttgl.constexpr, instr_variant: ttgl.constexpr
):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)

    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [2, M, N], layout)
    view = tmem.slice(1, 1, dim=0).index(0)
    view = view.reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
    view = view.permute([1, 0]).permute([1, 0])

    reg_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    view.store(ttgl.convert_layout(value, reg_layout))
    out = view.load(reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, reg_layout))

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
def tmem_scales_ldst_descriptor_view_kernel(in_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr,
                                            instr_variant: ttgl.constexpr, cga_layout: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.int8, [M, N], TensorMemoryScalesLayout(cga_layout=list(cga_layout)))
    root_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant=instr_variant)
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, root_layout))[:, None]
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, root_layout))[None, :]
    offs = offs_m * N + offs_n
    value = ttgl.load(in_ptr + offs)
    tmem.store(ttgl.convert_layout(value, root_layout))

    view = tmem.reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
    view_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    view_value = view.load(view_layout)
    view.store(view_value + ttgl.full([M, N], 3, ttgl.int8, layout=view_layout))

    out = tmem.load(root_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, root_layout))


@gluon.jit
def tmem_ld_red_explicit_layout_kernel(
    in_ptr, out_ptr, red_ptr, layout: ttgl.constexpr, N: ttgl.constexpr, load_variant: ttgl.constexpr,
    red_op: ttgl.constexpr, use_abs: ttgl.constexpr, propagate_nan: ttgl.constexpr
):
    M: ttgl.constexpr = 128
    num_warps: ttgl.constexpr = 4
    global_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    global_layout_1d: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [num_warps], [0])

    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, global_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, global_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=layout)
    store_layout: ttgl.constexpr = tmem.get_reg_layout()
    value = ttgl.convert_layout(value, store_layout)
    tmem.store(value)

    load_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant=load_variant)
    if red_op == "min":
        output, reduced = tmem.load_min(layout=load_layout, abs=use_abs, propagate_nan=propagate_nan)
    else:
        output, reduced = tmem.load_max(layout=load_layout, abs=use_abs, propagate_nan=propagate_nan)
    output = ttgl.convert_layout(output, global_layout)
    ttgl.store(out_ptr + offs, output)

    red_offs = ttgl.arange(0, M, global_layout_1d)
    reduced = ttgl.convert_layout(reduced, global_layout_1d)
    ttgl.store(red_ptr + red_offs, reduced)


@gluon.jit
def tmem_ld_red_descriptor_chain_kernel(
    in_ptr, out_ptr, red_ptr, layout: ttgl.constexpr, N: ttgl.constexpr, load_variant: ttgl.constexpr,
    red_op: ttgl.constexpr, use_abs: ttgl.constexpr, propagate_nan: ttgl.constexpr
):
    M: ttgl.constexpr = 128
    num_warps: ttgl.constexpr = 4
    global_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    global_layout_1d: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [num_warps], [0])

    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, global_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, global_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [2, M, N], layout=layout)
    base = tmem.slice(1, 1, dim=0).index(0)
    base_layout: ttgl.constexpr = base.get_reg_layout()
    base.store(ttgl.convert_layout(value, base_layout))

    view = base.reshape((M // 2, 2, N)).reshape((M, N))
    view = view.slice(0, M, dim=0).slice(0, N, dim=1)
    load_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=load_variant)
    if red_op == "min":
        output, reduced = view.load_min(layout=load_layout, abs=use_abs, propagate_nan=propagate_nan)
    else:
        output, reduced = view.load_max(layout=load_layout, abs=use_abs, propagate_nan=propagate_nan)
    output = ttgl.convert_layout(output, global_layout)
    ttgl.store(out_ptr + offs, output)

    red_offs = ttgl.arange(0, M, global_layout_1d)
    reduced = ttgl.convert_layout(reduced, global_layout_1d)
    ttgl.store(red_ptr + red_offs, reduced)


@gluon.jit
def tmem_ld_red_m64_explicit_layout_kernel(
    in_ptr, out_ptr, red_ptr, layout: ttgl.constexpr, N: ttgl.constexpr, load_variant: ttgl.constexpr,
    red_op: ttgl.constexpr, use_abs: ttgl.constexpr, propagate_nan: ttgl.constexpr
):
    M: ttgl.constexpr = 64
    num_warps: ttgl.constexpr = 4
    global_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    global_layout_1d: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [num_warps], [0])

    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, global_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, global_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=layout)
    store_layout: ttgl.constexpr = tmem.get_reg_layout()
    tmem.store(ttgl.convert_layout(value, store_layout))

    load_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant=load_variant)
    if red_op == "min":
        output, reduced = tmem.load_min(layout=load_layout, abs=use_abs, propagate_nan=propagate_nan)
    else:
        output, reduced = tmem.load_max(layout=load_layout, abs=use_abs, propagate_nan=propagate_nan)
    output = ttgl.convert_layout(output, global_layout)
    ttgl.store(out_ptr + offs, output)

    red_offs = ttgl.arange(0, M, global_layout_1d)
    reduced = ttgl.convert_layout(reduced, global_layout_1d)
    ttgl.store(red_ptr + red_offs, reduced)


@gluon.jit
def tmem_ld_red_non_f32_contract_kernel(
    in_ptr,
    out_ptr,
    red_ptr,
    layout: ttgl.constexpr,
    load_variant: ttgl.constexpr,
    red_op: ttgl.constexpr,
    use_abs: ttgl.constexpr,
    propagate_nan: ttgl.constexpr,
):
    M: ttgl.constexpr = 128
    N: ttgl.constexpr = 128
    num_warps: ttgl.constexpr = 4
    global_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    global_layout_1d: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [num_warps], [0])

    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, global_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, global_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=layout)
    store_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant=load_variant)
    value = ttgl.convert_layout(value, store_layout)
    tmem.store(value)

    if red_op == "min":
        output, reduced = tmem.load_min(layout=store_layout, abs=use_abs, propagate_nan=propagate_nan)
    else:
        output, reduced = tmem.load_max(layout=store_layout, abs=use_abs, propagate_nan=propagate_nan)
    output = ttgl.convert_layout(output, global_layout)
    ttgl.store(out_ptr + offs, output)

    red_offs = ttgl.arange(0, M, global_layout_1d)
    reduced = ttgl.convert_layout(reduced, global_layout_1d)
    ttgl.store(red_ptr + red_offs, reduced)


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

    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
        swizzle_byte_width=swizzle,
        element_bitwidth=in_ptr.dtype.element_ty.primitive_bitwidth,
        rank=2,
    )
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
def tmem_copy_no_scales_transposed_shared_kernel(
    in_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
    swizzle: ttgl.constexpr
):
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

    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
        swizzle_byte_width=swizzle,
        transposed=True,
        element_bitwidth=32,
        rank=2,
    )
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

    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
        swizzle_byte_width=swizzle,
        element_bitwidth=in_ptr.dtype.element_ty.primitive_bitwidth,
        rank=2,
    )
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
def tmem_copy_no_scales_4x256b_view_kernel(in_ptr, out_ptr, parent_layout: ttgl.constexpr):
    M: ttgl.constexpr = 4
    N: ttgl.constexpr = 8
    PARENT_M: ttgl.constexpr = 128

    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    in_m = ttgl.arange(0, M, ttgl.SliceLayout(1, blocked))
    in_n = ttgl.arange(0, N, ttgl.SliceLayout(0, blocked))
    in_offs = in_m[:, None] * N + in_n[None, :]
    value = ttgl.load(in_ptr + in_offs)

    parent = allocate_tensor_memory(in_ptr.dtype.element_ty, [PARENT_M, N], layout=parent_layout)
    tmem = parent.slice(0, M, dim=0)
    smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
        offset_bases=[[1, 0], [2, 0], [0, 1], [0, 2], [0, 4]],
        alignment=16,
    )
    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)
    smem.store(value)
    fence_async_shared()

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)

    reg_layout: ttgl.constexpr = parent.get_reg_layout()
    out_m = ttgl.arange(0, PARENT_M, ttgl.SliceLayout(1, reg_layout))
    out_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))
    out_offs = out_m[:, None] * N + out_n[None, :]
    output = parent.load(reg_layout)
    ttgl.store(out_ptr + out_offs, output)


@gluon.jit
def tmem_copy_no_scales_4x256b_refresh_kernel(in_ptr, out_ptr, layout: ttgl.constexpr):
    M: ttgl.constexpr = 4
    N: ttgl.constexpr = 8

    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    in_m = ttgl.arange(0, M, ttgl.SliceLayout(1, blocked))
    in_n = ttgl.arange(0, N, ttgl.SliceLayout(0, blocked))
    offs = in_m[:, None] * N + in_n[None, :]
    value = ttgl.load(in_ptr + offs)

    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=layout)
    smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
        offset_bases=[[1, 0], [2, 0], [0, 1], [0, 2], [0, 4]],
        alignment=16,
    )
    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)
    smem.store(value)
    fence_async_shared()

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)

    ttgl.store(out_ptr + offs, value)


@gluon.jit
def tmem_copy_no_scales_4x256b_refresh_twocta_kernel(in_ptr, out_ptr, layout: ttgl.constexpr,
                                                     cga_layout: ttgl.constexpr):
    M: ttgl.constexpr = 8
    N: ttgl.constexpr = 8

    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0], cga_layout=cga_layout)
    in_m = ttgl.arange(0, M, ttgl.SliceLayout(1, blocked))
    in_n = ttgl.arange(0, N, ttgl.SliceLayout(0, blocked))
    offs = in_m[:, None] * N + in_n[None, :]
    value = ttgl.load(in_ptr + offs)

    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=layout)
    smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
        offset_bases=[[1, 0], [2, 0], [0, 1], [0, 2], [0, 4]],
        block_bases=[[4, 0]],
        alignment=16,
    )
    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)
    smem.store(value)
    fence_async_shared(cluster=True)

    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)

    ttgl.store(out_ptr + offs, value)


@gluon.jit
def tmem_4x256b_refresh_get_reg_layout_unsupported_kernel(out_ptr, layout: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [4, 8], layout=layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()
    value = ttgl.full((), 0.0, ttgl.float32)
    ttgl.store(out_ptr, ttgl.convert_layout(value, reg_layout))


@gluon.jit
def tmem_4x256b_refresh_raw_bitcast_get_reg_layout_unsupported_kernel(out_ptr, layout: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [4, 8], layout=layout)
    raw = tmem.bitcast(ttgl.int8, [32, 4])
    reg_layout: ttgl.constexpr = raw.get_reg_layout()
    value = ttgl.full((), 0, ttgl.int8)
    ttgl.store(out_ptr, ttgl.convert_layout(value, reg_layout))


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
def tmem_copy_no_scales_twocta_linear_indexed_view_kernel(
    in_ptr,
    out_ptr,
    parent_layout: ttgl.constexpr,
    cga_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    swizzle: ttgl.constexpr,
    parent_depth: ttgl.constexpr,
    parent_index: ttgl.constexpr,
):
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [parent_depth, M, N], layout=parent_layout)
    view = tmem.index(parent_index)
    reg_layout: ttgl.constexpr = view.get_reg_layout()

    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
        swizzle_byte_width=swizzle,
        element_bitwidth=in_ptr.dtype.element_ty.primitive_bitwidth,
        rank=2,
        cga_layout=cga_layout,
    )
    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)
    smem.store(value)
    fence_async_shared(cluster=True)

    barrier = mbarrier.allocate_mbarrier()
    mbarrier.init(barrier, count=1)
    tcgen05_copy(smem, view)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    output = view.load(reg_layout)
    ttgl.store(out_ptr + offs, output)


@gluon.jit
def tmem_copy_no_scales_shared_subslice_bad_offset_kernel(in_ptr, out_ptr, slice_start: ttgl.constexpr):
    M: ttgl.constexpr = 128
    N: ttgl.constexpr = 128
    PARENT_N: ttgl.constexpr = 2 * N
    shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=32, rank=2)
    tmem_layout: ttgl.constexpr = TensorMemoryLayout(block=(128, N), col_stride=1)

    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, PARENT_N], layout=shared_layout)
    view = smem.slice(slice_start, N, dim=1)

    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], tmem_layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    fence_async_shared()
    tcgen05_copy(view, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)

    tmem_reg_layout: ttgl.constexpr = tmem.get_reg_layout()
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, tmem_reg_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, tmem_reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    out = tmem.load(tmem_reg_layout)
    ttgl.store(out_ptr + offs, out)


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
        swizzle_byte_width=swizzle,
        element_bitwidth=in_ptr.dtype.element_ty.primitive_bitwidth,
        rank=2,
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
def tmem_copy_no_scales_twocta_linear_subslice_view_kernel(in_ptr, out_ptr, parent_layout: ttgl.constexpr,
                                                           cga_layout: ttgl.constexpr, M: ttgl.constexpr,
                                                           N: ttgl.constexpr, swizzle: ttgl.constexpr):
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, 2 * N], layout=parent_layout)
    view = tmem.slice(N, N, dim=1)
    reg_layout: ttgl.constexpr = view.get_reg_layout()

    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
        swizzle_byte_width=swizzle,
        element_bitwidth=in_ptr.dtype.element_ty.primitive_bitwidth,
        rank=2,
        cga_layout=cga_layout,
    )
    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)
    smem.store(value)
    fence_async_shared(cluster=True)

    barrier = mbarrier.allocate_mbarrier()
    mbarrier.init(barrier, count=1)
    tcgen05_copy(smem, view)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    output = view.load(reg_layout)
    ttgl.store(out_ptr + offs, output)


@gluon.jit
def tmem_mma_indexed_acc_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
    parent_layout: ttgl.constexpr,
    block_layout_a: ttgl.constexpr,
    block_layout_b: ttgl.constexpr,
    shared_layout_a: ttgl.constexpr,
    shared_layout_b: ttgl.constexpr,
    use_acc: ttgl.constexpr,
    parent_depth: ttgl.constexpr,
    parent_index: ttgl.constexpr,
):
    a_offs = ttgl.arange(0, M)[:, None] * K + ttgl.arange(0, K)[None, :]
    b_offs = ttgl.arange(0, K)[:, None] * N + ttgl.arange(0, N)[None, :]
    c_offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]

    a = ttgl.load(ttgl.set_auto_layout(a_ptr + a_offs, block_layout_a))
    b = ttgl.load(ttgl.set_auto_layout(b_ptr + b_offs, block_layout_b))
    c = ttgl.load(c_ptr + c_offs)
    operand_dtype: ttgl.constexpr = a.dtype

    smem_a = ttgl.allocate_shared_memory(operand_dtype, [M, K], layout=shared_layout_a)
    smem_b = ttgl.allocate_shared_memory(operand_dtype, [K, N], layout=shared_layout_b)
    smem_a.store(a)
    smem_b.store(b)

    acc_parent = allocate_tensor_memory(ttgl.float32, [parent_depth, M, N], parent_layout)
    acc_tmem = acc_parent.index(parent_index).reshape((M, N))
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
def tmem_mma_acc_subslice_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
    parent_layout: ttgl.constexpr,
    slice_start: ttgl.constexpr,
    block_layout_a: ttgl.constexpr,
    block_layout_b: ttgl.constexpr,
    shared_layout_a: ttgl.constexpr,
    shared_layout_b: ttgl.constexpr,
    use_acc: ttgl.constexpr,
):
    a_offs = ttgl.arange(0, M)[:, None] * K + ttgl.arange(0, K)[None, :]
    b_offs = ttgl.arange(0, K)[:, None] * N + ttgl.arange(0, N)[None, :]
    c_offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]

    a = ttgl.load(ttgl.set_auto_layout(a_ptr + a_offs, block_layout_a))
    b = ttgl.load(ttgl.set_auto_layout(b_ptr + b_offs, block_layout_b))
    operand_dtype: ttgl.constexpr = a.dtype

    smem_a = ttgl.allocate_shared_memory(operand_dtype, [M, K], layout=shared_layout_a)
    smem_b = ttgl.allocate_shared_memory(operand_dtype, [K, N], layout=shared_layout_b)
    smem_a.store(a)
    smem_b.store(b)

    acc_parent = allocate_tensor_memory(ttgl.float32, [M, 2 * N], parent_layout)
    acc_tmem = acc_parent.slice(slice_start, N, dim=1)
    if use_acc:
        c = ttgl.load(c_ptr + c_offs)
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
def tmem_mma_lhs_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    lhs_layout: ttgl.constexpr,
    acc_layout: ttgl.constexpr,
    smem_b_layout: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
    ACC_INIT: ttgl.constexpr,
    USE_ACC: ttgl.constexpr,
):
    M: ttgl.constexpr = 128
    a_offs = ttgl.arange(0, M)[:, None] * K + ttgl.arange(0, K)[None, :]
    b_offs = ttgl.arange(0, K)[:, None] * N + ttgl.arange(0, N)[None, :]
    out_offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]

    blocked_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    blocked_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    a = ttgl.load(ttgl.set_auto_layout(a_ptr + a_offs, blocked_a))
    b = ttgl.load(ttgl.set_auto_layout(b_ptr + b_offs, blocked_b))

    operand_dtype: ttgl.constexpr = a_ptr.dtype.element_ty
    lhs_tmem = allocate_tensor_memory(operand_dtype, [M, K], lhs_layout)
    lhs_reg_layout: ttgl.constexpr = lhs_tmem.get_reg_layout()
    lhs_tmem.store(ttgl.convert_layout(a, lhs_reg_layout))

    smem_b = ttgl.allocate_shared_memory(operand_dtype, [K, N], layout=smem_b_layout)
    smem_b.store(b)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)
    if USE_ACC:
        acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
        acc_tmem.store(ttgl.full([M, N], ACC_INIT, ttgl.float32, layout=acc_reg_layout))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma(lhs_tmem, smem_b, acc_tmem, use_acc=USE_ACC)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out = acc_tmem.load()
    ttgl.store(out_ptr + out_offs, out)


@gluon.jit
def tmem_mma_lhs_subslice_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    parent_layout: ttgl.constexpr,
    acc_layout: ttgl.constexpr,
    smem_b_layout: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
    ACC_INIT: ttgl.constexpr,
    USE_ACC: ttgl.constexpr,
):
    M: ttgl.constexpr = 128
    PARENT_K: ttgl.constexpr = 2 * K
    a_offs = ttgl.arange(0, M)[:, None] * K + ttgl.arange(0, K)[None, :]
    b_offs = ttgl.arange(0, K)[:, None] * N + ttgl.arange(0, N)[None, :]
    out_offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]

    blocked_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    blocked_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    a = ttgl.load(ttgl.set_auto_layout(a_ptr + a_offs, blocked_a))
    b = ttgl.load(ttgl.set_auto_layout(b_ptr + b_offs, blocked_b))

    operand_dtype: ttgl.constexpr = a_ptr.dtype.element_ty
    lhs_parent = allocate_tensor_memory(operand_dtype, [M, PARENT_K], parent_layout)
    lhs_tmem = lhs_parent.slice(K, K, dim=1)
    lhs_reg_layout: ttgl.constexpr = lhs_tmem.get_reg_layout()
    lhs_tmem.store(ttgl.convert_layout(a, lhs_reg_layout))

    smem_b = ttgl.allocate_shared_memory(operand_dtype, [K, N], layout=smem_b_layout)
    smem_b.store(b)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)
    if USE_ACC:
        acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
        acc_tmem.store(ttgl.full([M, N], ACC_INIT, ttgl.float32, layout=acc_reg_layout))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma(lhs_tmem, smem_b, acc_tmem, use_acc=USE_ACC)
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
def tmem_copy_no_scales_warpx2_subslice_view_kernel(
    in_ptr,
    out_ptr,
    shared_layout: ttgl.constexpr,
    parent_layout: ttgl.constexpr,
    slice_start: ttgl.constexpr,
):
    M: ttgl.constexpr = 128
    N: ttgl.constexpr = 4
    PARENT_N: ttgl.constexpr = 8
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
    parent = allocate_tensor_memory(
        in_ptr.dtype.element_ty, [M, PARENT_N], layout=parent_layout
    )
    tmem = parent.slice(slice_start, N, dim=1)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    smem = ttgl.allocate_shared_memory(
        in_ptr.dtype.element_ty, [M, N], layout=shared_layout, value=value
    )
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
def tmem_copy_no_scales_warpx2_indexed_view_kernel(
    in_ptr,
    out_ptr,
    shared_layout: ttgl.constexpr,
    tmem_layout: ttgl.constexpr,
    parent_depth: ttgl.constexpr,
    parent_index: ttgl.constexpr,
):
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
    parent = allocate_tensor_memory(
        in_ptr.dtype.element_ty, [parent_depth, M, N], layout=tmem_layout
    )
    tmem = parent.index(parent_index)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    smem = ttgl.allocate_shared_memory(
        in_ptr.dtype.element_ty, [M, N], layout=shared_layout, value=value
    )
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
def tmem_copy_no_scales_warpx2_slice_index_view_kernel(
    in_ptr,
    out_ptr,
    shared_layout: ttgl.constexpr,
    parent_layout: ttgl.constexpr,
    parent_index: ttgl.constexpr,
):
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
    parent = allocate_tensor_memory(
        in_ptr.dtype.element_ty, [2, M, N], layout=parent_layout
    )
    tmem = parent.slice(parent_index, 1, dim=0).index(0)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    smem = ttgl.allocate_shared_memory(
        in_ptr.dtype.element_ty, [M, N], layout=shared_layout, value=value
    )
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
def tmem_copy_no_scales_warpx2_twocta_kernel(in_ptr, out_ptr, shared_layout: ttgl.constexpr, tmem_layout: ttgl.constexpr):
    M: ttgl.constexpr = 256
    N: ttgl.constexpr = 4
    shared_reg_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
        warp_bases=[[32, 0], [64, 0]],
        block_bases=[[128, 0]],
        shape=[M, N],
    )
    in_offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, shared_reg_layout))
    in_offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, shared_reg_layout))
    in_offs = in_offs_m[:, None] * N + in_offs_n[None, :]
    value = ttgl.load(in_ptr + in_offs)
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=tmem_layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=shared_layout, value=value)
    fence_async_shared(cluster=True)

    barrier = mbarrier.allocate_mbarrier()
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
def tmem_copy_no_scales_warpx2_twocta_subslice_view_kernel(
    in_ptr,
    out_ptr,
    shared_layout: ttgl.constexpr,
    parent_layout: ttgl.constexpr,
    slice_start: ttgl.constexpr,
):
    M: ttgl.constexpr = 256
    N: ttgl.constexpr = 4
    PARENT_N: ttgl.constexpr = 8
    shared_reg_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
        warp_bases=[[32, 0], [64, 0]],
        block_bases=[[128, 0]],
        shape=[M, N],
    )
    in_offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, shared_reg_layout))
    in_offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, shared_reg_layout))
    in_offs = in_offs_m[:, None] * N + in_offs_n[None, :]
    value = ttgl.load(in_ptr + in_offs)
    parent = allocate_tensor_memory(
        in_ptr.dtype.element_ty, [M, PARENT_N], layout=parent_layout
    )
    tmem = parent.slice(slice_start, N, dim=1)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    smem = ttgl.allocate_shared_memory(
        in_ptr.dtype.element_ty, [M, N], layout=shared_layout, value=value
    )
    fence_async_shared(cluster=True)

    barrier = mbarrier.allocate_mbarrier()
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
def tmem_copy_no_scales_warpx2_twocta_indexed_view_kernel(
    in_ptr,
    out_ptr,
    shared_layout: ttgl.constexpr,
    tmem_layout: ttgl.constexpr,
    parent_depth: ttgl.constexpr,
    parent_index: ttgl.constexpr,
):
    M: ttgl.constexpr = 256
    N: ttgl.constexpr = 4
    shared_reg_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
        warp_bases=[[32, 0], [64, 0]],
        block_bases=[[128, 0]],
        shape=[M, N],
    )
    in_offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, shared_reg_layout))
    in_offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, shared_reg_layout))
    in_offs = in_offs_m[:, None] * N + in_offs_n[None, :]
    value = ttgl.load(in_ptr + in_offs)
    parent = allocate_tensor_memory(
        in_ptr.dtype.element_ty, [parent_depth, M, N], layout=tmem_layout
    )
    tmem = parent.index(parent_index)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    smem = ttgl.allocate_shared_memory(
        in_ptr.dtype.element_ty, [M, N], layout=shared_layout, value=value
    )
    fence_async_shared(cluster=True)

    barrier = mbarrier.allocate_mbarrier()
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
def tmem_copy_no_scales_warpx2_twocta_slice_index_view_kernel(
    in_ptr,
    out_ptr,
    shared_layout: ttgl.constexpr,
    parent_layout: ttgl.constexpr,
    parent_index: ttgl.constexpr,
):
    M: ttgl.constexpr = 256
    N: ttgl.constexpr = 4
    shared_reg_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
        warp_bases=[[32, 0], [64, 0]],
        block_bases=[[128, 0]],
        shape=[M, N],
    )
    in_offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, shared_reg_layout))
    in_offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, shared_reg_layout))
    in_offs = in_offs_m[:, None] * N + in_offs_n[None, :]
    value = ttgl.load(in_ptr + in_offs)
    parent = allocate_tensor_memory(
        in_ptr.dtype.element_ty, [2, M, N], layout=parent_layout
    )
    tmem = parent.slice(parent_index, 1, dim=0).index(0)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    smem = ttgl.allocate_shared_memory(
        in_ptr.dtype.element_ty, [M, N], layout=shared_layout, value=value
    )
    fence_async_shared(cluster=True)

    barrier = mbarrier.allocate_mbarrier()
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
        element_bitwidth=in_ptr.dtype.element_ty.primitive_bitwidth,
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
def tmem_copy_no_scales_twocta_noncanonical_block_kernel(in_ptr, out_ptr, layout: ttgl.constexpr):
    M: ttgl.constexpr = 256
    N: ttgl.constexpr = 4
    reg_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
        warp_bases=[[32, 0], [64, 0]],
        block_bases=[[128, 0]],
        shape=[M, N],
    )
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]],
        block_bases=[[192, 0]],
        alignment=16,
    )
    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=layout)
    smem.store(value)
    fence_async_shared(cluster=True)

    barrier = mbarrier.allocate_mbarrier()
    mbarrier.init(barrier, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    ttgl.store(out_ptr + offs, value)


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
    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=tmem_layout)
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
def tmem_copy_128x128_subword_exact_kernel(
    in_ptr, out_ptr, N: ttgl.constexpr, tmem_layout: ttgl.constexpr, shared_layout: ttgl.constexpr
):
    M: ttgl.constexpr = 128
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=tmem_layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=shared_layout, value=value)
    fence_async_shared()

    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    out = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, out)


@gluon.jit
def tmem_copy_128x128_twocta_kernel(in_ptr, out_ptr, tmem_layout: ttgl.constexpr):
    M: ttgl.constexpr = 256
    N: ttgl.constexpr = 4
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=tmem_layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]],
        block_bases=[[128, 0]],
        alignment=16,
    )
    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)
    smem.store(value)
    fence_async_shared(cluster=True)

    barrier = mbarrier.allocate_mbarrier()
    mbarrier.init(barrier, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    out = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, out)


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
def tmem_copy_scales_warpx4_twocta_kernel(in_ptr, out_ptr):
    SMEM_H: ttgl.constexpr = 128
    SMEM_W: ttgl.constexpr = 16

    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0], cga_layout=[[1, 0]])
    offs_m = ttgl.arange(0, SMEM_H, layout=ttgl.SliceLayout(1, blocked))
    offs_n = ttgl.arange(0, SMEM_W, layout=ttgl.SliceLayout(0, blocked))
    offs = offs_m[:, None] * SMEM_W + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [32, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]],
        block_bases=[[64, 0]],
    )
    tmem = allocate_tensor_memory(
        ttgl.int8, (SMEM_H, SMEM_W), layout=TensorMemoryScalesLayout(cga_layout=[[1, 0]])
    )
    smem = ttgl.allocate_shared_memory(ttgl.int8, (SMEM_H, SMEM_W), layout=smem_layout)
    smem.store(value)
    fence_async_shared(cluster=True)

    barrier = mbarrier.allocate_mbarrier()
    mbarrier.init(barrier, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    reg_layout: ttgl.constexpr = tmem.get_reg_layout()
    output = tmem.load(reg_layout)
    out_offs_m = ttgl.arange(0, SMEM_H, layout=ttgl.SliceLayout(1, reg_layout))
    out_offs_n = ttgl.arange(0, SMEM_W, layout=ttgl.SliceLayout(0, reg_layout))
    out_offs = out_offs_m[:, None] * SMEM_W + out_offs_n[None, :]
    ttgl.store(out_ptr + out_offs, output)


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
def tmem_copy_scales_shared_subslice_layout_probe_kernel(
    out_ptr,
    parent_layout: ttgl.constexpr,
    start_row: ttgl.constexpr,
):
    PARENT_ROWS: ttgl.constexpr = 128
    SMEM_H: ttgl.constexpr = 64
    SMEM_W: ttgl.constexpr = 16

    smem_parent = ttgl.allocate_shared_memory(ttgl.int8, (PARENT_ROWS, SMEM_W), layout=parent_layout)
    smem = smem_parent.slice(start_row, SMEM_H, dim=0)
    tmem = allocate_tensor_memory(ttgl.int8, (SMEM_H, SMEM_W), layout=TensorMemoryScalesLayout())

    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    ttgl.store(out_ptr, 0)


@gluon.jit
def tmem_copy_scales_tmem_descriptor_view_kernel(in_ptr, out_ptr):
    SMEM_H: ttgl.constexpr = 128
    SMEM_W: ttgl.constexpr = 32

    in_ptrs = in_ptr + ttgl.arange(0, SMEM_H)[:, None] * SMEM_W + ttgl.arange(0, SMEM_W)[None, :]
    out_ptrs = out_ptr + ttgl.arange(0, SMEM_H)[:, None] * SMEM_W + ttgl.arange(0, SMEM_W)[None, :]

    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    value = ttgl.load(ttgl.set_auto_layout(in_ptrs, blocked))

    smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
        offset_bases=[
            [0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0],
            [4, 0], [8, 0], [16, 0], [0, 4], [0, 8], [0, 16]
        ]
    )
    smem = ttgl.allocate_shared_memory(ttgl.int8, (SMEM_H, SMEM_W), layout=smem_layout)
    tmem_parent = allocate_tensor_memory(ttgl.int8, (SMEM_H, SMEM_W), layout=TensorMemoryScalesLayout())
    tmem = tmem_parent.reshape((SMEM_H // 2, 2, SMEM_W)).permute([1, 0, 2]).reshape((SMEM_H, SMEM_W))

    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    smem.store(value)
    fence_async_shared()
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    reg_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant="32x32b")
    output = tmem.load(reg_layout)
    ttgl.store(ttgl.set_auto_layout(out_ptrs, blocked),
               ttgl.convert_layout(output, blocked))


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
def tmem_mma_twocta_use_acc_kernel(a_desc, b_desc, c_ptr, out_ptrs, BLOCK_M: ttgl.constexpr,
                                   BLOCK_N: ttgl.constexpr, acc_tmem_layout: ttgl.constexpr,
                                   blocked_c: ttgl.constexpr):
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
    c_offs_m = ttgl.arange(0, BLOCK_M)[:, None]
    c_offs_n = ttgl.arange(0, BLOCK_N)[None, :]
    c = ttgl.load(ttgl.set_auto_layout(c_ptr + c_offs_m * BLOCK_N + c_offs_n, blocked_c))
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.convert_layout(c, acc_reg_layout))

    tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=True, multicast=True, mbarriers=[mma_bar])
    mbarrier.wait(mma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(mma_bar)

    out = acc_tmem.load()
    out = ttgl.convert_layout(out, blocked_c)
    ttgl.store(out_ptrs + c_offs_m * BLOCK_N + c_offs_n, out)


@gluon.jit
def tmem_mma_twocta_tma_b_transposed_kernel(a_desc, b_desc, out_ptrs, BLOCK_M: ttgl.constexpr,
                                            BLOCK_N: ttgl.constexpr, acc_tmem_layout: ttgl.constexpr,
                                            blocked_c: ttgl.constexpr):
    smem_a = ttgl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    smem_b = ttgl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)
    mma_b = smem_b.permute((1, 0))

    tma_bar = mbarrier.allocate_mbarrier(two_ctas=acc_tmem_layout.two_ctas)
    mbarrier.init(tma_bar, count=1)
    mma_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(mma_bar, count=tcgen05_mma_barrier_count([smem_a, mma_b], True))

    mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
    tma.async_copy_global_to_shared(a_desc, [0, 0], tma_bar, smem_a, multicast=True)
    tma.async_copy_global_to_shared(b_desc, [0, 0], tma_bar, smem_b, multicast=True)
    mbarrier.wait(tma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(tma_bar)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [BLOCK_M, BLOCK_N], acc_tmem_layout)
    tcgen05_mma(smem_a, mma_b, acc_tmem, use_acc=False, multicast=True, mbarriers=[mma_bar])
    mbarrier.wait(mma_bar, phase=0, deps=[smem_a, mma_b])
    mbarrier.invalidate(mma_bar)

    out = acc_tmem.load()
    out = ttgl.convert_layout(out, blocked_c)
    out_offs_m = ttgl.arange(0, BLOCK_M)[:, None]
    out_offs_n = ttgl.arange(0, BLOCK_N)[None, :]
    ttgl.store(out_ptrs + out_offs_m * BLOCK_N + out_offs_n, out)


@gluon.jit
def tmem_mma_twocta_tma_b_transposed_use_acc_kernel(a_desc, b_desc, c_ptr, out_ptrs, BLOCK_M: ttgl.constexpr,
                                                    BLOCK_N: ttgl.constexpr, acc_tmem_layout: ttgl.constexpr,
                                                    blocked_c: ttgl.constexpr):
    smem_a = ttgl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    smem_b = ttgl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)
    mma_b = smem_b.permute((1, 0))

    tma_bar = mbarrier.allocate_mbarrier(two_ctas=acc_tmem_layout.two_ctas)
    mbarrier.init(tma_bar, count=1)
    mma_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(mma_bar, count=tcgen05_mma_barrier_count([smem_a, mma_b], True))

    mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
    tma.async_copy_global_to_shared(a_desc, [0, 0], tma_bar, smem_a, multicast=True)
    tma.async_copy_global_to_shared(b_desc, [0, 0], tma_bar, smem_b, multicast=True)
    mbarrier.wait(tma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(tma_bar)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [BLOCK_M, BLOCK_N], acc_tmem_layout)
    c_offs_m = ttgl.arange(0, BLOCK_M)[:, None]
    c_offs_n = ttgl.arange(0, BLOCK_N)[None, :]
    c = ttgl.load(ttgl.set_auto_layout(c_ptr + c_offs_m * BLOCK_N + c_offs_n, blocked_c))
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.convert_layout(c, acc_reg_layout))

    tcgen05_mma(smem_a, mma_b, acc_tmem, use_acc=True, multicast=True, mbarriers=[mma_bar])
    mbarrier.wait(mma_bar, phase=0, deps=[smem_a, mma_b])
    mbarrier.invalidate(mma_bar)

    out = acc_tmem.load()
    out = ttgl.convert_layout(out, blocked_c)
    ttgl.store(out_ptrs + c_offs_m * BLOCK_N + c_offs_n, out)


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
def tmem_mma_plain_kind_use_acc_kernel(a_ptr, b_ptr, c_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr,
                                       K: ttgl.constexpr, block_layout_a: ttgl.constexpr,
                                       block_layout_b: ttgl.constexpr, acc_layout: ttgl.constexpr,
                                       shared_layout_a: ttgl.constexpr, shared_layout_b: ttgl.constexpr):
    a_offs_m = ttgl.arange(0, M)[:, None]
    a_offs_k = ttgl.arange(0, K)[None, :]
    b_offs_k = ttgl.arange(0, K)[:, None]
    b_offs_n = ttgl.arange(0, N)[None, :]

    a = ttgl.load(ttgl.set_auto_layout(a_ptr + a_offs_m * K + a_offs_k, block_layout_a))
    b = ttgl.load(ttgl.set_auto_layout(b_ptr + b_offs_k * N + b_offs_n, block_layout_b))
    operand_dtype: ttgl.constexpr = a.dtype

    smem_a = ttgl.allocate_shared_memory(operand_dtype, [M, K], shared_layout_a)
    smem_b = ttgl.allocate_shared_memory(operand_dtype, [K, N], shared_layout_b)
    smem_a.store(a)
    smem_b.store(b)

    c_offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    c = ttgl.load(c_ptr + c_offs)
    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.convert_layout(c, acc_reg_layout))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=tcgen05_mma_barrier_count([smem_a, smem_b], False))
    tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=True, mbarriers=[bar])
    mbarrier.wait(bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(bar)

    out = acc_tmem.load()
    ttgl.store(out_ptr + c_offs, out)


@gluon.jit
def tmem_mma_twocta_plain_kind_use_acc_kernel(a_ptr, b_ptr, c_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr,
                                              K: ttgl.constexpr, block_layout_a: ttgl.constexpr,
                                              block_layout_b: ttgl.constexpr, block_layout_c: ttgl.constexpr,
                                              acc_layout: ttgl.constexpr, shared_layout_a: ttgl.constexpr,
                                              shared_layout_b: ttgl.constexpr):
    a_offs_m = ttgl.arange(0, M)[:, None]
    a_offs_k = ttgl.arange(0, K)[None, :]
    b_offs_k = ttgl.arange(0, K)[:, None]
    b_offs_n = ttgl.arange(0, N)[None, :]
    c_offs_m = ttgl.arange(0, M)[:, None]
    c_offs_n = ttgl.arange(0, N)[None, :]

    a = ttgl.load(ttgl.set_auto_layout(a_ptr + a_offs_m * K + a_offs_k, block_layout_a))
    b = ttgl.load(ttgl.set_auto_layout(b_ptr + b_offs_k * N + b_offs_n, block_layout_b))
    c = ttgl.load(ttgl.set_auto_layout(c_ptr + c_offs_m * N + c_offs_n, block_layout_c))
    operand_dtype: ttgl.constexpr = a.dtype

    smem_a = ttgl.allocate_shared_memory(operand_dtype, [M, K], shared_layout_a)
    smem_b = ttgl.allocate_shared_memory(operand_dtype, [K, N], shared_layout_b)
    smem_a.store(a)
    smem_b.store(b)
    fence_async_shared(cluster=True)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.convert_layout(c, acc_reg_layout))

    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=tcgen05_mma_barrier_count([smem_a, smem_b], True))
    tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=True, multicast=True, mbarriers=[bar])
    mbarrier.wait(bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(bar)

    out = acc_tmem.load()
    out = ttgl.convert_layout(out, block_layout_c)
    ttgl.store(out_ptr + c_offs_m * N + c_offs_n, out)


@gluon.jit
def tmem_mma_twocta_indexed_acc_kernel(a_ptr, b_ptr, c_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr,
                                       K: ttgl.constexpr, parent_layout: ttgl.constexpr,
                                       block_layout_a: ttgl.constexpr, block_layout_b: ttgl.constexpr,
                                       block_layout_c: ttgl.constexpr, shared_layout_a: ttgl.constexpr,
                                       shared_layout_b: ttgl.constexpr, use_acc: ttgl.constexpr,
                                       parent_depth: ttgl.constexpr, parent_index: ttgl.constexpr):
    a_offs_m = ttgl.arange(0, M)[:, None]
    a_offs_k = ttgl.arange(0, K)[None, :]
    b_offs_k = ttgl.arange(0, K)[:, None]
    b_offs_n = ttgl.arange(0, N)[None, :]
    c_offs_m = ttgl.arange(0, M)[:, None]
    c_offs_n = ttgl.arange(0, N)[None, :]

    a = ttgl.load(ttgl.set_auto_layout(a_ptr + a_offs_m * K + a_offs_k, block_layout_a))
    b = ttgl.load(ttgl.set_auto_layout(b_ptr + b_offs_k * N + b_offs_n, block_layout_b))
    operand_dtype: ttgl.constexpr = a.dtype

    smem_a = ttgl.allocate_shared_memory(operand_dtype, [M, K], shared_layout_a)
    smem_b = ttgl.allocate_shared_memory(operand_dtype, [K, N], shared_layout_b)
    smem_a.store(a)
    smem_b.store(b)
    fence_async_shared(cluster=True)

    acc_parent = allocate_tensor_memory(ttgl.float32, [parent_depth, M, N], parent_layout)
    acc_tmem = acc_parent.index(parent_index).reshape((M, N))
    if use_acc:
        c = ttgl.load(ttgl.set_auto_layout(c_ptr + c_offs_m * N + c_offs_n, block_layout_c))
        acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
        acc_tmem.store(ttgl.convert_layout(c, acc_reg_layout))

    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=tcgen05_mma_barrier_count([smem_a, smem_b], True))
    tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=use_acc, multicast=True, mbarriers=[bar])
    mbarrier.wait(bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(bar)

    out = acc_tmem.load()
    out = ttgl.convert_layout(out, block_layout_c)
    ttgl.store(out_ptr + c_offs_m * N + c_offs_n, out)


@gluon.jit
def tmem_mma_twocta_acc_subslice_kernel(a_ptr, b_ptr, c_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr,
                                         K: ttgl.constexpr, parent_layout: ttgl.constexpr,
                                         slice_start: ttgl.constexpr, block_layout_a: ttgl.constexpr,
                                         block_layout_b: ttgl.constexpr, block_layout_c: ttgl.constexpr,
                                         shared_layout_a: ttgl.constexpr, shared_layout_b: ttgl.constexpr,
                                         use_acc: ttgl.constexpr):
    a_offs_m = ttgl.arange(0, M)[:, None]
    a_offs_k = ttgl.arange(0, K)[None, :]
    b_offs_k = ttgl.arange(0, K)[:, None]
    b_offs_n = ttgl.arange(0, N)[None, :]
    c_offs_m = ttgl.arange(0, M)[:, None]
    c_offs_n = ttgl.arange(0, N)[None, :]

    a = ttgl.load(ttgl.set_auto_layout(a_ptr + a_offs_m * K + a_offs_k, block_layout_a))
    b = ttgl.load(ttgl.set_auto_layout(b_ptr + b_offs_k * N + b_offs_n, block_layout_b))
    operand_dtype: ttgl.constexpr = a.dtype

    smem_a = ttgl.allocate_shared_memory(operand_dtype, [M, K], shared_layout_a)
    smem_b = ttgl.allocate_shared_memory(operand_dtype, [K, N], shared_layout_b)
    smem_a.store(a)
    smem_b.store(b)
    fence_async_shared(cluster=True)

    acc_parent = allocate_tensor_memory(ttgl.float32, [M, 2 * N], parent_layout)
    acc_tmem = acc_parent.slice(slice_start, N, dim=1)
    if use_acc:
        c = ttgl.load(ttgl.set_auto_layout(c_ptr + c_offs_m * N + c_offs_n, block_layout_c))
        acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
        acc_tmem.store(ttgl.convert_layout(c, acc_reg_layout))

    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=tcgen05_mma_barrier_count([smem_a, smem_b], True))
    tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=use_acc, multicast=True, mbarriers=[bar])
    mbarrier.wait(bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(bar)

    out = acc_tmem.load()
    out = ttgl.convert_layout(out, block_layout_c)
    ttgl.store(out_ptr + c_offs_m * N + c_offs_n, out)


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
def tmem_mma_scaled_layout_format_kernel(
    out_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
    a,
    b,
    a_scale,
    b_scale,
    acc_layout: ttgl.constexpr,
    VEC_SIZE: ttgl.constexpr,
    A_ELEM_PER_BYTE: ttgl.constexpr,
    B_ELEM_PER_BYTE: ttgl.constexpr,
    A_FORMAT: ttgl.constexpr,
    B_FORMAT: ttgl.constexpr,
    ACC_INIT: ttgl.constexpr,
):
    A_STORAGE_K: ttgl.constexpr = K // A_ELEM_PER_BYTE
    B_STORAGE_K: ttgl.constexpr = K // B_ELEM_PER_BYTE
    A_IS_FP4: ttgl.constexpr = A_ELEM_PER_BYTE == 2
    B_IS_FP4: ttgl.constexpr = B_ELEM_PER_BYTE == 2
    MIXED_PREC: ttgl.constexpr = A_ELEM_PER_BYTE != B_ELEM_PER_BYTE

    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [ttgl.num_warps(), 1], [1, 0])
    a_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(
        [M, A_STORAGE_K],
        a.dtype.element_ty,
        fp4_padded=A_IS_FP4 and MIXED_PREC,
    )
    b_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(
        [N, B_STORAGE_K],
        b.dtype.element_ty,
        fp4_padded=B_IS_FP4 and MIXED_PREC,
    )
    block_layout_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])
    block_layout_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])

    a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout_a))[:, None]
    a_offs_k = ttgl.arange(0, A_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_a))[None, :]
    b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, block_layout_b))[:, None]
    b_offs_k = ttgl.arange(0, B_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_b))[None, :]

    a_tile = ttgl.load(a + a_offs_m * A_STORAGE_K + a_offs_k)
    b_tile = ttgl.load(b + b_offs_n * B_STORAGE_K + b_offs_k)
    a_smem = ttgl.allocate_shared_memory(a.dtype.element_ty, [M, A_STORAGE_K], a_nvmma_layout, a_tile)
    b_smem = ttgl.allocate_shared_memory(b.dtype.element_ty, [N, B_STORAGE_K], b_nvmma_layout, b_tile)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.full([M, N], ACC_INIT, ttgl.float32, layout=acc_reg_layout))

    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [M, K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype.element_ty, [N, K // VEC_SIZE], scale_layout)
    scale_reg_layout_m: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_reg_layout_n: ttgl.constexpr = b_scale_tmem.get_reg_layout()

    scale_offs_k_m = ttgl.arange(0, K // VEC_SIZE, layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
    scale_offs_k_n = ttgl.arange(0, K // VEC_SIZE, layout=ttgl.SliceLayout(0, scale_reg_layout_n))[None, :]
    scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
    scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
    a_scale_tmem.store(ttgl.load(a_scale + scale_offs_m * (K // VEC_SIZE) + scale_offs_k_m))
    b_scale_tmem.store(ttgl.load(b_scale + scale_offs_n * (K // VEC_SIZE) + scale_offs_k_n))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(
        a_smem,
        b_smem.permute((1, 0)),
        acc_tmem,
        a_scale_tmem,
        b_scale_tmem,
        A_FORMAT,
        B_FORMAT,
        use_acc=True,
    )
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out_reg = acc_tmem.load()
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, reg_layout))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, reg_layout))[None, :]
    offs = offs_m * N + offs_n
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out_reg, reg_layout))


@gluon.jit
def tmem_mma_scaled_bscale_descriptor_view_format_kernel(
    out_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
    a,
    b,
    a_scale,
    b_scale,
    acc_layout: ttgl.constexpr,
    VEC_SIZE: ttgl.constexpr,
    A_ELEM_PER_BYTE: ttgl.constexpr,
    B_ELEM_PER_BYTE: ttgl.constexpr,
    A_FORMAT: ttgl.constexpr,
    B_FORMAT: ttgl.constexpr,
    PAD_B_SCALE_STORAGE: ttgl.constexpr,
    EXTRA_B_SCALE_USER: ttgl.constexpr = False,
):
    A_STORAGE_K: ttgl.constexpr = K // A_ELEM_PER_BYTE
    B_STORAGE_K: ttgl.constexpr = K // B_ELEM_PER_BYTE
    A_IS_FP4: ttgl.constexpr = A_ELEM_PER_BYTE == 2
    B_IS_FP4: ttgl.constexpr = B_ELEM_PER_BYTE == 2
    MIXED_PREC: ttgl.constexpr = A_ELEM_PER_BYTE != B_ELEM_PER_BYTE
    B_SCALE_ROWS: ttgl.constexpr = 2 * N if PAD_B_SCALE_STORAGE else N
    B_SCALE_INSTR_N: ttgl.constexpr = 32
    B_SCALE_FRAGMENT_N: ttgl.constexpr = 64

    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [ttgl.num_warps(), 1], [1, 0])
    a_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(
        [M, A_STORAGE_K],
        a.dtype.element_ty,
        fp4_padded=A_IS_FP4 and MIXED_PREC,
    )
    b_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(
        [N, B_STORAGE_K],
        b.dtype.element_ty,
        fp4_padded=B_IS_FP4 and MIXED_PREC,
    )
    block_layout_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])
    block_layout_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])

    a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout_a))[:, None]
    a_offs_k = ttgl.arange(0, A_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_a))[None, :]
    b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, block_layout_b))[:, None]
    b_offs_k = ttgl.arange(0, B_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_b))[None, :]

    a_tile = ttgl.load(a + a_offs_m * A_STORAGE_K + a_offs_k)
    b_tile = ttgl.load(b + b_offs_n * B_STORAGE_K + b_offs_k)
    a_smem = ttgl.allocate_shared_memory(a.dtype.element_ty, [M, A_STORAGE_K], a_nvmma_layout, a_tile)
    b_smem = ttgl.allocate_shared_memory(b.dtype.element_ty, [N, B_STORAGE_K], b_nvmma_layout, b_tile)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.full([M, N], 0.0, ttgl.float32, layout=acc_reg_layout))

    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [M, K // VEC_SIZE], scale_layout)
    b_scale_parent = allocate_tensor_memory(b_scale.dtype.element_ty, [B_SCALE_ROWS, K // VEC_SIZE], scale_layout)
    if PAD_B_SCALE_STORAGE:
        b_scale_tmem = b_scale_parent.reshape((1, B_SCALE_ROWS, K // VEC_SIZE)).slice(0, 1, dim=0).index(0)
    else:
        b_scale_tmem = b_scale_parent.reshape((N // 2, 2, K // VEC_SIZE)).permute([1, 0, 2]).reshape(
            (N, K // VEC_SIZE)
        )
    scale_reg_layout_m: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_reg_layout_n: ttgl.constexpr = b_scale_tmem.get_reg_layout()

    scale_offs_k_m = ttgl.arange(0, K // VEC_SIZE, layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
    scale_offs_k_n = ttgl.arange(0, K // VEC_SIZE, layout=ttgl.SliceLayout(0, scale_reg_layout_n))[None, :]
    scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
    scale_offs_n = ttgl.arange(0, B_SCALE_ROWS, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
    if PAD_B_SCALE_STORAGE:
        source_scale_n = (scale_offs_n // B_SCALE_FRAGMENT_N) * B_SCALE_INSTR_N + (scale_offs_n % B_SCALE_INSTR_N)
    else:
        source_scale_n = scale_offs_n
    a_scale_tmem.store(ttgl.load(a_scale + scale_offs_m * (K // VEC_SIZE) + scale_offs_k_m))
    b_scale_tmem.store(ttgl.load(b_scale + source_scale_n * (K // VEC_SIZE) + scale_offs_k_n))
    if EXTRA_B_SCALE_USER:
        probe = b_scale_tmem.load()
        ttgl.store(out_ptr, ttgl.sum(probe.to(ttgl.float32)))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(
        a_smem,
        b_smem.permute((1, 0)),
        acc_tmem,
        a_scale_tmem,
        b_scale_tmem,
        A_FORMAT,
        B_FORMAT,
        use_acc=True,
    )
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out_reg = acc_tmem.load()
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, reg_layout))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, reg_layout))[None, :]
    offs = offs_m * N + offs_n
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out_reg, reg_layout))


@gluon.jit
def tmem_mma_scaled_scale_descriptor_view_format_kernel(
    out_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
    a,
    b,
    a_scale,
    b_scale,
    acc_layout: ttgl.constexpr,
    VEC_SIZE: ttgl.constexpr,
    A_ELEM_PER_BYTE: ttgl.constexpr,
    B_ELEM_PER_BYTE: ttgl.constexpr,
    A_FORMAT: ttgl.constexpr,
    B_FORMAT: ttgl.constexpr,
    SCALE_SIDE: ttgl.constexpr,
):
    A_STORAGE_K: ttgl.constexpr = K // A_ELEM_PER_BYTE
    B_STORAGE_K: ttgl.constexpr = K // B_ELEM_PER_BYTE
    A_IS_FP4: ttgl.constexpr = A_ELEM_PER_BYTE == 2
    B_IS_FP4: ttgl.constexpr = B_ELEM_PER_BYTE == 2
    MIXED_PREC: ttgl.constexpr = A_ELEM_PER_BYTE != B_ELEM_PER_BYTE
    SCALE_K: ttgl.constexpr = K // VEC_SIZE

    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [ttgl.num_warps(), 1], [1, 0])
    a_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(
        [M, A_STORAGE_K],
        a.dtype.element_ty,
        fp4_padded=A_IS_FP4 and MIXED_PREC,
    )
    b_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(
        [N, B_STORAGE_K],
        b.dtype.element_ty,
        fp4_padded=B_IS_FP4 and MIXED_PREC,
    )
    block_layout_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])
    block_layout_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])

    a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout_a))[:, None]
    a_offs_k = ttgl.arange(0, A_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_a))[None, :]
    b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, block_layout_b))[:, None]
    b_offs_k = ttgl.arange(0, B_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_b))[None, :]

    a_tile = ttgl.load(a + a_offs_m * A_STORAGE_K + a_offs_k)
    b_tile = ttgl.load(b + b_offs_n * B_STORAGE_K + b_offs_k)
    a_smem = ttgl.allocate_shared_memory(a.dtype.element_ty, [M, A_STORAGE_K], a_nvmma_layout, a_tile)
    b_smem = ttgl.allocate_shared_memory(b.dtype.element_ty, [N, B_STORAGE_K], b_nvmma_layout, b_tile)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.full([M, N], 0.0, ttgl.float32, layout=acc_reg_layout))

    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    if SCALE_SIDE == "a":
        a_parent = allocate_tensor_memory(a_scale.dtype.element_ty, [M, SCALE_K], scale_layout)
        a_scale_tmem = a_parent.reshape((M // 2, 2, SCALE_K)).permute([1, 0, 2]).reshape((M, SCALE_K))
        b_scale_tmem = allocate_tensor_memory(b_scale.dtype.element_ty, [N, SCALE_K], scale_layout)
    else:
        a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [M, SCALE_K], scale_layout)
        b_parent = allocate_tensor_memory(b_scale.dtype.element_ty, [N, SCALE_K], scale_layout)
        b_scale_tmem = b_parent.reshape((N // 2, 2, SCALE_K)).permute([1, 0, 2]).reshape((N, SCALE_K))

    scale_reg_layout_m: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_reg_layout_n: ttgl.constexpr = b_scale_tmem.get_reg_layout()
    scale_offs_k_m = ttgl.arange(0, SCALE_K, layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
    scale_offs_k_n = ttgl.arange(0, SCALE_K, layout=ttgl.SliceLayout(0, scale_reg_layout_n))[None, :]
    scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
    scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
    a_scale_tmem.store(ttgl.load(a_scale + scale_offs_m * SCALE_K + scale_offs_k_m))
    b_scale_tmem.store(ttgl.load(b_scale + scale_offs_n * SCALE_K + scale_offs_k_n))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(
        a_smem,
        b_smem.permute((1, 0)),
        acc_tmem,
        a_scale_tmem,
        b_scale_tmem,
        A_FORMAT,
        B_FORMAT,
        use_acc=True,
    )
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out_reg = acc_tmem.load()
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, reg_layout))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, reg_layout))[None, :]
    ttgl.store(out_ptr + offs_m * N + offs_n, ttgl.convert_layout(out_reg, reg_layout))


@gluon.jit
def tmem_mma_scaled_dynamic_bscale_direct_kernel(
    out_ptr,
    selector_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
    a,
    b,
    a_scale,
    b_scale0,
    b_scale1,
    acc_layout: ttgl.constexpr,
    SELECT_MODE: ttgl.constexpr,
):
    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [ttgl.num_warps(), 1], [1, 0])
    block_layout_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])
    block_layout_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])
    smem_a_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([M, K], a.dtype.element_ty)
    smem_b_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([N, K], b.dtype.element_ty)

    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout_a))[:, None]
    offs_ak = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, block_layout_a))[None, :]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, block_layout_b))[:, None]
    offs_bk = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, block_layout_b))[None, :]
    a_smem = ttgl.allocate_shared_memory(a.dtype.element_ty, [M, K], smem_a_layout, ttgl.load(a + offs_m * K + offs_ak))
    b_smem = ttgl.allocate_shared_memory(b.dtype.element_ty, [N, K], smem_b_layout, ttgl.load(b + offs_n * K + offs_bk))

    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [M, K // 32], scale_layout)
    b_scale_tmem0 = allocate_tensor_memory(b_scale0.dtype.element_ty, [N, K // 32], scale_layout)
    b_scale_tmem1 = allocate_tensor_memory(b_scale1.dtype.element_ty, [N, K // 32], scale_layout)
    scale_reg_layout_m: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_reg_layout_n: ttgl.constexpr = b_scale_tmem0.get_reg_layout()

    scale_offs_k_m = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
    scale_offs_k_n = ttgl.arange(0, K // 32, layout=ttgl.SliceLayout(0, scale_reg_layout_n))[None, :]
    scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
    scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
    a_scale_tmem.store(ttgl.load(a_scale + scale_offs_m * (K // 32) + scale_offs_k_m))
    b_scale_tmem0.store(ttgl.load(b_scale0 + scale_offs_n * (K // 32) + scale_offs_k_n))
    b_scale_tmem1.store(ttgl.load(b_scale1 + scale_offs_n * (K // 32) + scale_offs_k_n))

    b_scale_selected = b_scale_tmem0
    if SELECT_MODE == "branch":
        if ttgl.load(selector_ptr) != 0:
            b_scale_selected = b_scale_tmem1
        else:
            b_scale_selected = b_scale_tmem0
    else:
        for i in range(0, 2, 1):
            if i == ttgl.load(selector_ptr):
                b_scale_selected = b_scale_tmem1
            else:
                b_scale_selected = b_scale_selected

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.zeros([M, N], ttgl.float32, layout=acc_reg_layout))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(
        a_smem,
        b_smem.permute((1, 0)),
        acc_tmem,
        a_scale_tmem,
        b_scale_selected,
        "e4m3",
        "e4m3",
        use_acc=True,
    )
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out_reg = acc_tmem.load()
    out_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, reg_layout))[:, None]
    out_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, reg_layout))[None, :]
    ttgl.store(out_ptr + out_m * N + out_n, ttgl.convert_layout(out_reg, reg_layout))


@gluon.jit
def tmem_mma_scaled_indexed_acc_format_kernel(
    out_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
    a,
    b,
    a_scale,
    b_scale,
    parent_layout: ttgl.constexpr,
    parent_depth: ttgl.constexpr,
    parent_index: ttgl.constexpr,
    VEC_SIZE: ttgl.constexpr,
    A_ELEM_PER_BYTE: ttgl.constexpr,
    B_ELEM_PER_BYTE: ttgl.constexpr,
    A_FORMAT: ttgl.constexpr,
    B_FORMAT: ttgl.constexpr,
    ACC_INIT: ttgl.constexpr,
):
    A_STORAGE_K: ttgl.constexpr = K // A_ELEM_PER_BYTE
    B_STORAGE_K: ttgl.constexpr = K // B_ELEM_PER_BYTE
    A_IS_FP4: ttgl.constexpr = A_ELEM_PER_BYTE == 2
    B_IS_FP4: ttgl.constexpr = B_ELEM_PER_BYTE == 2
    MIXED_PREC: ttgl.constexpr = A_ELEM_PER_BYTE != B_ELEM_PER_BYTE

    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [ttgl.num_warps(), 1], [1, 0])
    a_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(
        [M, A_STORAGE_K],
        a.dtype.element_ty,
        fp4_padded=A_IS_FP4 and MIXED_PREC,
    )
    b_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(
        [N, B_STORAGE_K],
        b.dtype.element_ty,
        fp4_padded=B_IS_FP4 and MIXED_PREC,
    )
    block_layout_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])
    block_layout_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])

    a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout_a))[:, None]
    a_offs_k = ttgl.arange(0, A_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_a))[None, :]
    b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, block_layout_b))[:, None]
    b_offs_k = ttgl.arange(0, B_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_b))[None, :]

    a_tile = ttgl.load(a + a_offs_m * A_STORAGE_K + a_offs_k)
    b_tile = ttgl.load(b + b_offs_n * B_STORAGE_K + b_offs_k)
    a_smem = ttgl.allocate_shared_memory(a.dtype.element_ty, [M, A_STORAGE_K], a_nvmma_layout, a_tile)
    b_smem = ttgl.allocate_shared_memory(b.dtype.element_ty, [N, B_STORAGE_K], b_nvmma_layout, b_tile)

    acc_parent = allocate_tensor_memory(ttgl.float32, [parent_depth, M, N], parent_layout)
    acc_tmem = acc_parent.index(parent_index).reshape((M, N))
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.full([M, N], ACC_INIT, ttgl.float32, layout=acc_reg_layout))

    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [M, K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype.element_ty, [N, K // VEC_SIZE], scale_layout)
    scale_reg_layout_m: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_reg_layout_n: ttgl.constexpr = b_scale_tmem.get_reg_layout()

    scale_offs_k_m = ttgl.arange(0, K // VEC_SIZE, layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
    scale_offs_k_n = ttgl.arange(0, K // VEC_SIZE, layout=ttgl.SliceLayout(0, scale_reg_layout_n))[None, :]
    scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
    scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
    a_scale_tmem.store(ttgl.load(a_scale + scale_offs_m * (K // VEC_SIZE) + scale_offs_k_m))
    b_scale_tmem.store(ttgl.load(b_scale + scale_offs_n * (K // VEC_SIZE) + scale_offs_k_n))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(
        a_smem,
        b_smem.permute((1, 0)),
        acc_tmem,
        a_scale_tmem,
        b_scale_tmem,
        A_FORMAT,
        B_FORMAT,
        use_acc=True,
    )
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
def tmem_mma_scaled_lhs_subslice_format_kernel(
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
    VEC_SIZE: ttgl.constexpr,
    A_ELEM_PER_BYTE: ttgl.constexpr,
    B_ELEM_PER_BYTE: ttgl.constexpr,
    A_FORMAT: ttgl.constexpr,
    B_FORMAT: ttgl.constexpr,
    ACC_INIT: ttgl.constexpr,
):
    A_STORAGE_K: ttgl.constexpr = K // A_ELEM_PER_BYTE
    B_STORAGE_K: ttgl.constexpr = K // B_ELEM_PER_BYTE
    PARENT_STORAGE_K: ttgl.constexpr = 2 * A_STORAGE_K
    B_IS_FP4: ttgl.constexpr = B_ELEM_PER_BYTE == 2
    MIXED_PREC: ttgl.constexpr = A_ELEM_PER_BYTE != B_ELEM_PER_BYTE

    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [ttgl.num_warps(), 1], [1, 0])
    b_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(
        [N, B_STORAGE_K],
        b.dtype.element_ty,
        fp4_padded=B_IS_FP4 and MIXED_PREC,
    )
    block_layout_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])
    block_layout_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])

    a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout_a))[:, None]
    a_offs_k = ttgl.arange(0, A_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_a))[None, :]
    b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, block_layout_b))[:, None]
    b_offs_k = ttgl.arange(0, B_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_b))[None, :]

    a_tile = ttgl.load(a + a_offs_m * A_STORAGE_K + a_offs_k)
    b_tile = ttgl.load(b + b_offs_n * B_STORAGE_K + b_offs_k)

    lhs_parent = allocate_tensor_memory(a.dtype.element_ty, [M, PARENT_STORAGE_K], parent_layout)
    lhs_tmem = lhs_parent.slice(A_STORAGE_K, A_STORAGE_K, dim=1)
    lhs_reg_layout: ttgl.constexpr = lhs_tmem.get_reg_layout()
    lhs_tmem.store(ttgl.convert_layout(a_tile, lhs_reg_layout))
    b_smem = ttgl.allocate_shared_memory(b.dtype.element_ty, [N, B_STORAGE_K], b_nvmma_layout, b_tile)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.full([M, N], ACC_INIT, ttgl.float32, layout=acc_reg_layout))

    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [M, K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype.element_ty, [N, K // VEC_SIZE], scale_layout)
    scale_reg_layout_m: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_reg_layout_n: ttgl.constexpr = b_scale_tmem.get_reg_layout()

    scale_offs_k_m = ttgl.arange(0, K // VEC_SIZE, layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
    scale_offs_k_n = ttgl.arange(0, K // VEC_SIZE, layout=ttgl.SliceLayout(0, scale_reg_layout_n))[None, :]
    scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
    scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
    a_scale_tmem.store(ttgl.load(a_scale + scale_offs_m * (K // VEC_SIZE) + scale_offs_k_m))
    b_scale_tmem.store(ttgl.load(b_scale + scale_offs_n * (K // VEC_SIZE) + scale_offs_k_n))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(
        lhs_tmem,
        b_smem.permute((1, 0)),
        acc_tmem,
        a_scale_tmem,
        b_scale_tmem,
        A_FORMAT,
        B_FORMAT,
        use_acc=True,
    )
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out_reg = acc_tmem.load()
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, reg_layout))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, reg_layout))[None, :]
    offs = offs_m * N + offs_n
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out_reg, reg_layout))


@gluon.jit
def tmem_mma_scaled_lhs_tile_permuted_format_kernel(
    out_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
    a,
    b,
    a_scale,
    b_scale,
    lhs_layout: ttgl.constexpr,
    acc_layout: ttgl.constexpr,
    VEC_SIZE: ttgl.constexpr,
    A_ELEM_PER_BYTE: ttgl.constexpr,
    B_ELEM_PER_BYTE: ttgl.constexpr,
    A_FORMAT: ttgl.constexpr,
    B_FORMAT: ttgl.constexpr,
    ACC_INIT: ttgl.constexpr,
):
    A_STORAGE_K: ttgl.constexpr = K // A_ELEM_PER_BYTE
    B_STORAGE_K: ttgl.constexpr = K // B_ELEM_PER_BYTE
    B_IS_FP4: ttgl.constexpr = B_ELEM_PER_BYTE == 2
    MIXED_PREC: ttgl.constexpr = A_ELEM_PER_BYTE != B_ELEM_PER_BYTE

    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [ttgl.num_warps(), 1], [1, 0])
    b_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(
        [N, B_STORAGE_K],
        b.dtype.element_ty,
        fp4_padded=B_IS_FP4 and MIXED_PREC,
    )
    block_layout_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])
    block_layout_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])

    a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout_a))[:, None]
    a_offs_k = ttgl.arange(0, A_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_a))[None, :]
    b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, block_layout_b))[:, None]
    b_offs_k = ttgl.arange(0, B_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_b))[None, :]

    a_tile = ttgl.load(a + a_offs_m * A_STORAGE_K + a_offs_k)
    b_tile = ttgl.load(b + b_offs_n * B_STORAGE_K + b_offs_k)

    lhs_tmem = allocate_tensor_memory(a.dtype.element_ty, [M, A_STORAGE_K], lhs_layout)
    lhs_reg_layout: ttgl.constexpr = lhs_tmem.get_reg_layout()
    lhs_tmem.store(ttgl.convert_layout(a_tile, lhs_reg_layout))
    b_smem = ttgl.allocate_shared_memory(b.dtype.element_ty, [N, B_STORAGE_K], b_nvmma_layout, b_tile)

    acc_tmem = allocate_tensor_memory(ttgl.float32, [M, N], acc_layout)
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.full([M, N], ACC_INIT, ttgl.float32, layout=acc_reg_layout))

    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [M, K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype.element_ty, [N, K // VEC_SIZE], scale_layout)
    scale_reg_layout_m: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_reg_layout_n: ttgl.constexpr = b_scale_tmem.get_reg_layout()

    scale_offs_k_m = ttgl.arange(0, K // VEC_SIZE, layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
    scale_offs_k_n = ttgl.arange(0, K // VEC_SIZE, layout=ttgl.SliceLayout(0, scale_reg_layout_n))[None, :]
    scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
    scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
    a_scale_tmem.store(ttgl.load(a_scale + scale_offs_m * (K // VEC_SIZE) + scale_offs_k_m))
    b_scale_tmem.store(ttgl.load(b_scale + scale_offs_n * (K // VEC_SIZE) + scale_offs_k_n))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(
        lhs_tmem,
        b_smem.permute((1, 0)),
        acc_tmem,
        a_scale_tmem,
        b_scale_tmem,
        A_FORMAT,
        B_FORMAT,
        use_acc=True,
    )
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


@gluon.jit
def tmem_mma_scaled_acc_subslice_format_kernel(
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
    VEC_SIZE: ttgl.constexpr,
    A_ELEM_PER_BYTE: ttgl.constexpr,
    B_ELEM_PER_BYTE: ttgl.constexpr,
    A_FORMAT: ttgl.constexpr,
    B_FORMAT: ttgl.constexpr,
    ACC_INIT: ttgl.constexpr,
):
    parent_n: ttgl.constexpr = 2 * N
    A_STORAGE_K: ttgl.constexpr = K // A_ELEM_PER_BYTE
    B_STORAGE_K: ttgl.constexpr = K // B_ELEM_PER_BYTE
    A_IS_FP4: ttgl.constexpr = A_ELEM_PER_BYTE == 2
    B_IS_FP4: ttgl.constexpr = B_ELEM_PER_BYTE == 2
    MIXED_PREC: ttgl.constexpr = A_ELEM_PER_BYTE != B_ELEM_PER_BYTE

    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [ttgl.num_warps(), 1], [1, 0])
    a_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(
        [M, A_STORAGE_K],
        a.dtype.element_ty,
        fp4_padded=A_IS_FP4 and MIXED_PREC,
    )
    b_nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(
        [N, B_STORAGE_K],
        b.dtype.element_ty,
        fp4_padded=B_IS_FP4 and MIXED_PREC,
    )
    block_layout_a: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])
    block_layout_b: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])

    a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout_a))[:, None]
    a_offs_k = ttgl.arange(0, A_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_a))[None, :]
    b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, block_layout_b))[:, None]
    b_offs_k = ttgl.arange(0, B_STORAGE_K, layout=ttgl.SliceLayout(0, block_layout_b))[None, :]

    a_tile = ttgl.load(a + a_offs_m * A_STORAGE_K + a_offs_k)
    b_tile = ttgl.load(b + b_offs_n * B_STORAGE_K + b_offs_k)
    a_smem = ttgl.allocate_shared_memory(a.dtype.element_ty, [M, A_STORAGE_K], a_nvmma_layout, a_tile)
    b_smem = ttgl.allocate_shared_memory(b.dtype.element_ty, [N, B_STORAGE_K], b_nvmma_layout, b_tile)

    acc_parent = allocate_tensor_memory(ttgl.float32, [M, parent_n], parent_layout)
    acc_tmem = acc_parent.slice(slice_start, N, dim=1)
    acc_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(ttgl.full([M, N], ACC_INIT, ttgl.float32, layout=acc_reg_layout))

    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [M, K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype.element_ty, [N, K // VEC_SIZE], scale_layout)
    scale_reg_layout_m: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_reg_layout_n: ttgl.constexpr = b_scale_tmem.get_reg_layout()

    scale_offs_k_m = ttgl.arange(0, K // VEC_SIZE, layout=ttgl.SliceLayout(0, scale_reg_layout_m))[None, :]
    scale_offs_k_n = ttgl.arange(0, K // VEC_SIZE, layout=ttgl.SliceLayout(0, scale_reg_layout_n))[None, :]
    scale_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, scale_reg_layout_m))[:, None]
    scale_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, scale_reg_layout_n))[:, None]
    a_scale_tmem.store(ttgl.load(a_scale + scale_offs_m * (K // VEC_SIZE) + scale_offs_k_m))
    b_scale_tmem.store(ttgl.load(b_scale + scale_offs_n * (K // VEC_SIZE) + scale_offs_k_n))

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(
        a_smem,
        b_smem.permute((1, 0)),
        acc_tmem,
        a_scale_tmem,
        b_scale_tmem,
        A_FORMAT,
        B_FORMAT,
        use_acc=True,
    )
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out_reg = acc_tmem.load()
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, reg_layout))[:, None]
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, reg_layout))[None, :]
    offs = offs_m * N + offs_n
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out_reg, reg_layout))


@gluon.jit
def mma_scaled_tcgen05_acc_subslice_copy_kernel(
    a_desc,
    b_desc,
    c_desc,
    a_scale_desc,
    b_scale_desc,
    VEC_SIZE: ttgl.constexpr,
    block_layout_c: ttgl.constexpr,
    acc_parent_layout: ttgl.constexpr,
    PARENT_N: ttgl.constexpr,
    slice_start: ttgl.constexpr,
    ACC_INIT: ttgl.constexpr,
    multicast: ttgl.constexpr,
    DIRECT_SHARED_SCALES: ttgl.constexpr,
):
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
    A_SCALE_MN: ttgl.constexpr = a_scale_desc.block_type.shape[1] * 128
    B_SCALE_MN: ttgl.constexpr = b_scale_desc.block_type.shape[1] * 128
    if not DIRECT_SHARED_SCALES:
        scale_layout_a: ttgl.constexpr = TensorMemoryScalesLayout(cga_layout=[[1, 0]] if two_ctas else [])
        scale_layout_b: ttgl.constexpr = TensorMemoryScalesLayout(cga_layout=[[0, 0]] if two_ctas else [])
        a_scale_tmem = allocate_tensor_memory(a_scale_desc.dtype, [A_SCALE_MN, BLOCK_K // VEC_SIZE], scale_layout_a)
        b_scale_tmem = allocate_tensor_memory(b_scale_desc.dtype, [B_SCALE_MN, BLOCK_K // VEC_SIZE], scale_layout_b)
    acc_parent = allocate_tensor_memory(ttgl.float32, [BLOCK_M, PARENT_N], acc_parent_layout)
    acc_tmem = acc_parent.slice(slice_start, BLOCK_N, dim=1)
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
    off_m = ttgl.program_id(0) * BLOCK_M
    off_n = ttgl.program_id(1) * BLOCK_N

    a_scale_smem = ttgl.allocate_shared_memory(a_scale_desc.dtype, a_scale_desc.block_type.shape, a_scale_desc.layout)
    b_scale_smem = ttgl.allocate_shared_memory(b_scale_desc.dtype, b_scale_desc.block_type.shape, b_scale_desc.layout)
    REP_M: ttgl.constexpr = a_scale_desc.block_type.shape[1]
    REP_N: ttgl.constexpr = b_scale_desc.block_type.shape[1]
    A_REP_K: ttgl.constexpr = a_scale_desc.block_type.shape[2]
    B_REP_K: ttgl.constexpr = b_scale_desc.block_type.shape[2]
    off_m_a_scale = ttgl.program_id(0) * REP_M
    off_n_b_scale = ttgl.program_id(1) * REP_N

    for k in range(0, K, BLOCK_K):
        off_k_a = k // A_ELEM_PER_BYTE
        off_k_b = k // B_ELEM_PER_BYTE
        off_k_a_scale = (k // BLOCK_K) * A_REP_K
        off_k_b_scale = (k // BLOCK_K) * B_REP_K

        EXPECTED_BYTES: ttgl.constexpr = (
            a_desc.nbytes_per_cta
            + b_desc.nbytes_per_cta
            + a_scale_desc.nbytes_per_cta
            + b_scale_desc.nbytes_per_cta
        )
        mbarrier.expect(tma_bar, EXPECTED_BYTES)
        tma.async_copy_global_to_shared(a_desc, [off_m, off_k_a], tma_bar, a_smem)
        tma.async_copy_global_to_shared(b_desc, [off_n, off_k_b], tma_bar, b_smem)
        tma.async_copy_global_to_shared(a_scale_desc, [0, off_m_a_scale, off_k_a_scale, 0, 0], tma_bar, a_scale_smem)
        tma.async_copy_global_to_shared(
            b_scale_desc,
            [0, off_n_b_scale, off_k_b_scale, 0, 0],
            tma_bar,
            b_scale_smem,
            multicast=multicast,
        )
        mbarrier.wait(tma_bar, phase_tma, deps=[a_smem, b_smem, a_scale_smem, b_scale_smem])
        phase_tma ^= 1

        a_scale = unswizzle_scales_shared_memory(a_scale_smem, A_SCALE_MN, BLOCK_K, VEC_SIZE)
        b_scale = unswizzle_scales_shared_memory(b_scale_smem, B_SCALE_MN, BLOCK_K, VEC_SIZE)

        a_format: ttgl.constexpr = "e2m1" if A_IS_FP4 else "e4m3"
        b_format: ttgl.constexpr = "e2m1" if B_IS_FP4 else "e4m3"
        if DIRECT_SHARED_SCALES:
            tcgen05_mma_scaled(
                a_smem,
                b_smem.permute((1, 0)),
                acc_tmem,
                a_scale,
                b_scale,
                a_format,
                b_format,
                use_acc=(ACC_INIT != 0.0 or k != 0),
            )
        else:
            tcgen05_copy(a_scale, a_scale_tmem)
            tcgen05_copy(b_scale, b_scale_tmem)
            tcgen05_mma_scaled(
                a_smem,
                b_smem.permute((1, 0)),
                acc_tmem,
                a_scale_tmem,
                b_scale_tmem,
                a_format,
                b_format,
                use_acc=(ACC_INIT != 0.0 or k != 0),
            )
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


def mma_scaled_tcgen05_acc_subslice_copy(
    A,
    B,
    A_scale,
    B_scale,
    VEC_SIZE,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    PARENT_N,
    slice_start,
    num_ctas,
    multicast,
    acc_init=0.0,
    direct_shared_scales=False,
):
    M, N = A.shape[0], B.shape[0]
    mixed_prec = A.dtype != B.dtype
    two_ctas = num_ctas > 1
    warps = [4, 1]
    num_warps = warps[0] * warps[1]

    ctas_per_cga = (2, 1) if two_ctas else None
    cta_order = (1, 0) if two_ctas else None
    cta_split = (2, 1) if two_ctas else None
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, 0) if two_ctas else None
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, 0) if two_ctas else None
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0) if two_ctas else None

    cga_layout_a_scale = [[0, 1, 0, 0, 0]] if two_ctas else None
    cga_layout_b_scale = [[0, 0, 0, 0, 0]] if two_ctas else None

    a_desc = make_operand_descriptor(A, BLOCK_M, BLOCK_K, mixed_prec, cga_layout=cga_layout_a)
    b_desc = make_operand_descriptor(B, BLOCK_N, BLOCK_K, mixed_prec, cga_layout=cga_layout_b)
    c_desc = make_output_descriptor(M, N, torch.float16, BLOCK_M, BLOCK_N, cga_layout=cga_layout_c)
    a_scale_desc = make_scales_descriptor(
        A_scale, max(BLOCK_M, 128), BLOCK_K, VEC_SIZE, cga_layout=cga_layout_a_scale
    )
    b_scale_desc = make_scales_descriptor(
        B_scale, max(BLOCK_N, 128), BLOCK_K, VEC_SIZE, cga_layout=cga_layout_b_scale
    )

    a_scale_layout = ttgl.NVMMASharedLayout(
        swizzle_byte_width=0,
        element_bitwidth=8,
        rank=5,
        cga_layout=cga_layout_a_scale,
    )
    b_scale_layout = ttgl.NVMMASharedLayout(
        swizzle_byte_width=0,
        element_bitwidth=8,
        rank=5,
        cga_layout=cga_layout_b_scale,
    )
    a_scale_desc = replace(a_scale_desc, layout=a_scale_layout)
    b_scale_desc = replace(b_scale_desc, layout=b_scale_layout)

    block_layout_c = (
        ttgl.BlockedLayout([1, 8], [1, 32], warps_per_cta=warps, order=[1, 0], cga_layout=cga_layout_c)
        if two_ctas
        else None
    )
    acc_parent_layout = (
        _make_tmem_linear_layout_mmav5_twocta(BLOCK_M, PARENT_N)
        if two_ctas
        else _make_tmem_linear_layout(BLOCK_M, PARENT_N)
    )

    compiled = mma_scaled_tcgen05_acc_subslice_copy_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        a_desc,
        b_desc,
        c_desc,
        a_scale_desc,
        b_scale_desc,
        VEC_SIZE,
        block_layout_c,
        acc_parent_layout,
        PARENT_N,
        slice_start,
        acc_init,
        num_warps=num_warps,
        num_ctas=num_ctas,
        multicast=multicast,
        DIRECT_SHARED_SCALES=direct_shared_scales,
    )
    return c_desc.base, compiled


LDST_SHAPE_MAP = {
    "auto": {32: "32x32b.x32.b32", 64: "32x32b.x64.b32", 128: "32x32b.x128.b32",
             256: "32x32b.x64.b32"},
    "32x32b": {32: "32x32b.x32.b32", 64: "32x32b.x64.b32", 128: "32x32b.x128.b32",
               256: "32x32b.x64.b32"},
    "16x64b": {32: "16x64b.x16.b32", 64: "16x64b.x32.b32", 128: "16x64b.x64.b32",
               256: "16x64b.x64.b32"},
    "16x128b": {32: "16x128b.x8.b32", 64: "16x128b.x16.b32", 128: "16x128b.x32.b32",
                256: "16x128b.x32.b32"},
    "16x256b": {32: "16x256b.x4.b32", 64: "16x256b.x8.b32", 128: "16x256b.x16.b32",
                256: "16x256b.x16.b32"},
}

LDST_SUBVIEW_SHAPE_MAP = {
    "auto": {32: "32x32b.x32.b32", 64: "32x32b.x64.b32", 128: "32x32b.x128.b32"},
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
    "block_single_cta": lambda n: _make_tmem_linear_layout_block(128, n, two_ctas=False),
    "scrambled_cols": lambda n: _make_tmem_linear_layout_scrambled(128, n, scramble_rows=False, scramble_cols=True),
    "scrambled_rows_cols": lambda n: _make_tmem_linear_layout_scrambled(128, n, scramble_rows=True,
                                                                         scramble_cols=True),
}

MULTIDIM_SLICE_REPLAY_LAYOUTS = {
    "identity": LDST_LAYOUTS["identity"],
    "scrambled_cols": LDST_EXOTIC_LAYOUTS["scrambled_cols"],
}

MULTIDIM_SLICE_POSITIVE_LAYOUTS = {
    "mixed": LDST_LAYOUTS["mixed"],
}

LDST_TWOCTA_LAYOUTS = {
    "block_two_ctas": lambda n: _make_tmem_linear_layout_block(256, n, two_ctas=True),
    "mmav5_twocta": lambda n: _make_tmem_linear_layout_mmav5_twocta(256, n),
}

LDST_VARIANTS = ("auto", "32x32b", "16x64b", "16x128b", "16x256b")
LDST_32BIT_DTYPES = (("f32", torch.float32), ("i32", torch.int32))

LDST_CASE_SPECS = list(
    dict.fromkeys(
        [
            (layout_name, 128, "auto") for layout_name in LDST_LAYOUTS.keys()
        ] + [
            ("identity", 128, variant) for variant in LDST_VARIANTS
        ] + [
            ("mixed", n, "16x128b") for n in (64, 256)
        ] + [
            ("identity", 256, "32x32b"),
        ]
    )
)

LDST_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for layout_name, n, variant in LDST_CASE_SPECS
]

LDST_IDENTITY_N32_CASE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, "direct", "auto"),
            ("f32", torch.float32, "descriptor", "32x32b"),
        ] + [
            ("i32", torch.int32, "direct", "32x32b"),
            ("i32", torch.int32, "descriptor", "16x128b"),
        ]
    )
)

LDST_IDENTITY_N32_CASES = [
    (dtype_name, torch_dtype, mode, variant, LDST_SUBVIEW_SHAPE_MAP[variant][32])
    for dtype_name, torch_dtype, mode, variant in LDST_IDENTITY_N32_CASE_SPECS
]

LDST_DESCRIPTOR_CASES = LDST_CASES

LDST_I32_BROAD_CASE_SPECS = list(
    dict.fromkeys(
        [
            ("direct", "identity", 128, "auto"),
            ("descriptor", "mixed", 128, "auto"),
        ] + [
            ("direct", "identity", 128, "16x128b"),
            ("descriptor", "mixed", 128, "32x32b"),
            ("descriptor", "mixed", 128, "16x256b"),
        ] + [
            ("direct", "identity", 64, "16x128b"),
            ("descriptor", "identity", 256, "16x128b"),
        ]
    )
)

LDST_I32_BROAD_CASES = [
    (mode, layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for mode, layout_name, n, variant in LDST_I32_BROAD_CASE_SPECS
]

LDST_TWOCTA_CASE_SPECS = list(
    dict.fromkeys(
        [
            (layout_name, 128, "auto") for layout_name in LDST_TWOCTA_LAYOUTS.keys()
        ] + [
            ("block_two_ctas", 128, variant) for variant in LDST_VARIANTS
        ] + [
            ("mmav5_twocta", n, "16x128b") for n in (64, 256)
        ] + [
            ("block_two_ctas", 256, "32x32b"),
        ]
    )
)

LDST_TWOCTA_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for layout_name, n, variant in LDST_TWOCTA_CASE_SPECS
]

LDST_TWOCTA_DESCRIPTOR_CASES = LDST_TWOCTA_CASES

LDST_TWOCTA_I32_BROAD_CASE_SPECS = list(
    dict.fromkeys(
        [
            ("direct", "block_two_ctas", 128, "auto"),
            ("descriptor", "mmav5_twocta", 128, "auto"),
        ] + [
            ("direct", "block_two_ctas", 128, "16x128b"),
            ("descriptor", "block_two_ctas", 128, "32x32b"),
            ("descriptor", "block_two_ctas", 128, "16x256b"),
        ] + [
            ("direct", "block_two_ctas", 64, "16x128b"),
            ("descriptor", "block_two_ctas", 256, "16x128b"),
        ]
    )
)

LDST_TWOCTA_I32_BROAD_CASES = [
    (mode, layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for mode, layout_name, n, variant in LDST_TWOCTA_I32_BROAD_CASE_SPECS
]

LDST_TWOCTA_N32_CASE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, mode, layout_name, "auto")
            for mode, layout_name in (("direct", "block_two_ctas"), ("descriptor", "mmav5_twocta"))
        ] + [
            ("f32", torch.float32, "direct", "block_two_ctas", "16x128b"),
        ] + [
            ("i32", torch.int32, "direct", "block_two_ctas", "32x32b"),
            ("i32", torch.int32, "descriptor", "block_two_ctas", "16x128b"),
        ]
    )
)

LDST_TWOCTA_N32_CASES = [
    (dtype_name, torch_dtype, mode, layout_name, variant, LDST_SUBVIEW_SHAPE_MAP[variant][32])
    for dtype_name, torch_dtype, mode, layout_name, variant in LDST_TWOCTA_N32_CASE_SPECS
]

ALLOC_LIFETIME_LDST_CASES = [
    ("identity", 1, 1, 128, n, n) for n in (32, 128, 512)
] + [
    ("block_two_ctas", 2, 2, 256, n, n) for n in (32, 128, 512)
]

PERMUTED_LAYOUT_KINDS = ("identity", "rotate1", "even_odd", "reverse")
PERMUTED_ROW_COL_LAYOUT_KINDS = list(product(PERMUTED_LAYOUT_KINDS, PERMUTED_LAYOUT_KINDS))
PERMUTED_ROW_COL_EDGE_LAYOUT_KINDS = (
    ("identity", "reverse"),
    ("reverse", "identity"),
    ("rotate1", "even_odd"),
    ("even_odd", "reverse"),
)

LDST_PERMUTED_N32_CASE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, mode, perm_kind, "auto")
            for mode, perm_kind in (("direct", "rotate1"), ("descriptor", "reverse"))
        ] + [
            ("f32", torch.float32, "direct", "rotate1", "16x128b"),
        ] + [
            ("i32", torch.int32, "direct", "even_odd", "32x32b"),
            ("i32", torch.int32, "descriptor", "reverse", "16x128b"),
        ]
    )
)

LDST_PERMUTED_N32_CASES = [
    (dtype_name, torch_dtype, mode, perm_kind, variant, LDST_SUBVIEW_SHAPE_MAP[variant][32])
    for dtype_name, torch_dtype, mode, perm_kind, variant in LDST_PERMUTED_N32_CASE_SPECS
]

LDST_ROWCOL_N32_LAYOUT_CASES = (
    ("rotate1", "identity"),
    ("identity", "reverse"),
    ("even_odd", "reverse"),
)

LDST_ROWCOL_N32_CASE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, mode, row_perm_kind, col_perm_kind, "auto")
            for mode, row_perm_kind, col_perm_kind in (
                ("direct", "rotate1", "identity"),
                ("descriptor", "identity", "reverse"),
            )
        ] + [
            ("f32", torch.float32, "direct", "rotate1", "identity", "16x128b"),
        ] + [
            ("i32", torch.int32, "direct", "identity", "reverse", "32x32b"),
            ("i32", torch.int32, "descriptor", "even_odd", "reverse", "16x128b"),
        ]
    )
)

LDST_ROWCOL_N32_CASES = [
    (dtype_name, torch_dtype, mode, row_perm_kind, col_perm_kind, variant, LDST_SUBVIEW_SHAPE_MAP[variant][32])
    for dtype_name, torch_dtype, mode, row_perm_kind, col_perm_kind, variant in LDST_ROWCOL_N32_CASE_SPECS
]

# Keep permutation coverage representative: every permutation class is covered
# at a canonical shape, while variant/width/dtype edges are sampled instead of
# fully crossing the axes. Row/column permutation tests below cover mixed row
# and column permutations separately.
LDST_PERMUTED_CASE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, perm_kind, 128, "auto") for perm_kind in PERMUTED_LAYOUT_KINDS
        ] + [
            ("f32", torch.float32, "rotate1", 128, "16x128b"),
            ("f32", torch.float32, "reverse", 256, "16x128b"),
        ] + [
            ("i32", torch.int32, "reverse", 128, "32x32b"),
        ]
    )
)

LDST_PERMUTED_CASES = [
    (dtype_name, torch_dtype, perm_kind, n, variant, LDST_SHAPE_MAP[variant][n])
    for dtype_name, torch_dtype, perm_kind, n, variant in LDST_PERMUTED_CASE_SPECS
]

# Keep all row/column permutation classes covered at one canonical geometry,
# then sample edge geometries where N width or explicit atom variant changes
# the packet schedule. The one-dimensional permutation matrices above already
# cover the full dtype/N/variant Cartesian product.
LDST_ROWCOL_PERMUTED_CASE_SPECS = [
    ("f32", torch.float32, row_perm_kind, col_perm_kind, 128, "auto")
    for row_perm_kind, col_perm_kind in (
        ("identity", "identity"),
        ("identity", "reverse"),
        ("rotate1", "even_odd"),
    )
] + [
    ("f32", torch.float32, row_perm_kind, col_perm_kind, n, variant)
    for row_perm_kind, col_perm_kind, n, variant in (
        ("identity", "reverse", 64, "32x32b"),
        ("rotate1", "even_odd", 256, "16x128b"),
    )
] + [
    ("i32", torch.int32, "rotate1", "reverse", 128, "32x32b")
]

LDST_ROWCOL_PERMUTED_CASES = [
    (dtype_name, torch_dtype, row_perm_kind, col_perm_kind, n, variant, LDST_SHAPE_MAP[variant][n])
    for dtype_name, torch_dtype, row_perm_kind, col_perm_kind, n, variant in LDST_ROWCOL_PERMUTED_CASE_SPECS
]

LDST_EXOTIC_CASE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, layout_name, 128, "auto")
            for layout_name in ("block_single_cta", "scrambled_cols")
        ] + [
            ("f32", torch.float32, "scrambled_rows_cols", 128, "16x128b"),
            ("f32", torch.float32, "scrambled_cols", 256, "16x128b"),
        ] + [
            ("i32", torch.int32, "scrambled_cols", 128, "32x32b"),
        ]
    )
)

LDST_EXOTIC_CASES = [
    (dtype_name, torch_dtype, layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for dtype_name, torch_dtype, layout_name, n, variant in LDST_EXOTIC_CASE_SPECS
]

LDST_EXOTIC_N32_CASE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, "direct", "block_single_cta", "auto"),
            ("f32", torch.float32, "descriptor", "scrambled_cols", "auto"),
        ] + [
            ("f32", torch.float32, "direct", "scrambled_rows_cols", "16x128b"),
        ] + [
            ("i32", torch.int32, "direct", "block_single_cta", "32x32b"),
            ("i32", torch.int32, "descriptor", "scrambled_cols", "16x128b"),
        ]
    )
)

LDST_EXOTIC_N32_CASES = [
    (dtype_name, torch_dtype, mode, layout_name, variant, LDST_SUBVIEW_SHAPE_MAP[variant][32])
    for dtype_name, torch_dtype, mode, layout_name, variant in LDST_EXOTIC_N32_CASE_SPECS
]

LDST_EXOTIC_DESCRIPTOR_CASE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, layout_name, 128, "auto")
            for layout_name in ("scrambled_cols", "scrambled_rows_cols")
        ] + [
            ("f32", torch.float32, "scrambled_rows_cols", 128, "32x32b"),
            ("f32", torch.float32, "scrambled_rows_cols", 128, "16x128b"),
            ("f32", torch.float32, "scrambled_rows_cols", 128, "16x256b"),
        ] + [
            ("f32", torch.float32, "scrambled_cols", 256, "16x128b"),
        ] + [
            ("i32", torch.int32, "scrambled_cols", 128, "32x32b"),
        ]
    )
)

LDST_EXOTIC_DESCRIPTOR_CASES = [
    (dtype_name, torch_dtype, layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for dtype_name, torch_dtype, layout_name, n, variant in LDST_EXOTIC_DESCRIPTOR_CASE_SPECS
]

LDST_DESCRIPTOR_ROUNDTRIP_CHAINS = [
    ("slice_index_roundtrip", 0, 5.0, ("ttg.memdesc_index", "ttg.memdesc_subslice", "ttg.memdesc_reshape",
                                       "ttg.memdesc_trans")),
    ("slice_index_multidim", 1, 7.0, ("ttg.memdesc_index", "ttg.memdesc_subslice", "ttg.memdesc_reshape",
                                      "ttg.memdesc_trans")),
    ("slice_index_deep_roundtrip", 2, 11.0, ("ttg.memdesc_index", "ttg.memdesc_subslice",
                                             "ttg.memdesc_reshape", "ttg.memdesc_trans")),
]

LDST_LIFTED_ROUNDTRIP_OOR_SKIP_REASON = (
    "lifted descriptor roundtrip matrices exceed the current Blackwell TMEM allocation limit; "
    "these cases previously compiled until OutOfResources and then skipped, so pre-skipping "
    "preserves instruction/op coverage while avoiding known non-executable compiles"
)

# Descriptor roundtrip coverage follows the same representative policy as the
# direct row/column sweep; descriptor-chain operation coverage is asserted by
# the dedicated lifted-view tests.
LDST_DESCRIPTOR_ROUNDTRIP_ROWCOL_CASE_SPECS = [
    (row_perm_kind, col_perm_kind, 128, "auto")
    for row_perm_kind, col_perm_kind in PERMUTED_ROW_COL_LAYOUT_KINDS
] + [
    (row_perm_kind, col_perm_kind, n, "16x128b")
    for row_perm_kind, col_perm_kind in PERMUTED_ROW_COL_EDGE_LAYOUT_KINDS
    for n in (64, 256)
]

LDST_DESCRIPTOR_ROUNDTRIP_ROWCOL_CASES = [
    (row_perm_kind, col_perm_kind, n, variant, LDST_SHAPE_MAP[variant][n])
    for row_perm_kind, col_perm_kind, n, variant in LDST_DESCRIPTOR_ROUNDTRIP_ROWCOL_CASE_SPECS
]

LDST_HIGHER_RANK_CASE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, layout_name, 128, "auto") for layout_name in LDST_LAYOUTS.keys()
        ] + [
            ("f32", torch.float32, "mixed", 128, variant) for variant in LDST_VARIANTS
        ] + [
            ("f32", torch.float32, "identity", n, "16x128b") for n in (64, 128)
        ] + [
            ("i32", torch.int32, "mixed", 128, "32x32b"),
        ]
    )
)

LDST_HIGHER_RANK_SLICE_CASES = [
    (dtype_name, torch_dtype, layout_name, n, variant)
    for dtype_name, torch_dtype, layout_name, n, variant in LDST_HIGHER_RANK_CASE_SPECS
]

LDST_TWOCTA_HIGHER_RANK_SLICE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, layout_name, 128, "auto") for layout_name in LDST_TWOCTA_LAYOUTS.keys()
        ] + [
            ("f32", torch.float32, "block_two_ctas", 128, variant) for variant in LDST_VARIANTS
        ] + [
            ("f32", torch.float32, "block_two_ctas", n, "16x128b") for n in (64, 128)
        ] + [
            ("i32", torch.int32, "block_two_ctas", 128, "32x32b"),
        ]
    )
)

LDST_TWOCTA_HIGHER_RANK_INDEX_SPECS = [
    ("block_two_ctas", 64, "auto"),
    ("block_two_ctas", 128, "16x128b"),
    ("mmav5_twocta", 64, "auto"),
    ("mmav5_twocta", 128, "16x128b"),
]

LDST_TWOCTA_HIGHER_RANK_INDEX_CASES = [
    (
        "f32",
        torch.float32,
        layout_name,
        n,
        variant,
        LDST_SHAPE_MAP[variant][n],
        LDST_SUBVIEW_SHAPE_MAP[variant][n // 2],
    )
    for layout_name, n, variant in LDST_TWOCTA_HIGHER_RANK_INDEX_SPECS
]

LDST_TWOCTA_HIGHER_RANK_SLICE_CASES = [
    (dtype_name, torch_dtype, layout_name, n, variant)
    for dtype_name, torch_dtype, layout_name, n, variant in LDST_TWOCTA_HIGHER_RANK_SLICE_SPECS
]

LDST_TWOCTA_HIGHER_RANK_DIM0_SLICE_OOR_CASES = [
    ("block_two_ctas", variant) for variant in LDST_VARIANTS
] + [("mmav5_twocta", "32x32b")]

LDST_HIGHER_RANK_DIM0_SLICE_POSITIVE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, "identity", 128, "auto"),
            ("f32", torch.float32, "identity", 128, "16x128b"),
            ("f32", torch.float32, "identity", 128, "16x256b"),
        ] + [
            ("f32", torch.float32, "identity", 64, "16x128b"),
            ("i32", torch.int32, "identity", 128, "32x32b"),
        ]
    )
)

LDST_HIGHER_RANK_DIM0_SLICE_POSITIVE_CASES = [
    (dtype_name, torch_dtype, layout_name, n, variant, LDST_SHAPE_MAP[variant][n],
     LDST_SUBVIEW_SHAPE_MAP[variant][n // 2])
    for dtype_name, torch_dtype, layout_name, n, variant in LDST_HIGHER_RANK_DIM0_SLICE_POSITIVE_SPECS
]

LDST_HIGHER_RANK_HALF_ROWS_POSITIVE_CASES = [
    ("identity", n, variant, LDST_SHAPE_MAP[variant][n])
    for n, variant in ((64, "16x128b"), (128, "auto"), (128, "16x256b"))
]
LDST_HIGHER_RANK_HALF_ROWS_OOR_CASES = [
    ("identity", n, variant)
    for n, variant in ((256, "32x32b"), (256, "16x128b"))
]

LDST_DIRECT_HALF_ROWS_POSITIVE_CASES = [
    ("identity", n, variant, LDST_SHAPE_MAP[variant][n])
    for n, variant in ((64, "16x128b"), (128, "auto"), (128, "16x256b"), (256, "32x32b"))
]

LDST_TWOCTA_HIGHER_RANK_DIM0_SLICE_POSITIVE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, "block_two_ctas", 128, "auto"),
            ("f32", torch.float32, "block_two_ctas", 128, "16x128b"),
            ("f32", torch.float32, "block_two_ctas", 128, "16x256b"),
            ("f32", torch.float32, "mmav5_twocta", 128, "auto"),
        ] + [
            ("f32", torch.float32, "block_two_ctas", 64, "16x128b"),
            ("f32", torch.float32, "mmav5_twocta", 64, "16x128b"),
            ("i32", torch.int32, "block_two_ctas", 128, "32x32b"),
        ]
    )
)

LDST_TWOCTA_HIGHER_RANK_DIM0_SLICE_POSITIVE_CASES = [
    (dtype_name, torch_dtype, layout_name, n, variant, LDST_SHAPE_MAP[variant][n],
     LDST_SUBVIEW_SHAPE_MAP[variant][n // 2])
    for dtype_name, torch_dtype, layout_name, n, variant in LDST_TWOCTA_HIGHER_RANK_DIM0_SLICE_POSITIVE_SPECS
]

LDST_TWOCTA_HIGHER_RANK_HALF_ROWS_POSITIVE_CASES = [
    ("block_two_ctas", n, variant, LDST_SHAPE_MAP[variant][n])
    for n, variant in ((64, "16x128b"), (128, "auto"), (128, "16x256b"))
]
LDST_TWOCTA_HIGHER_RANK_HALF_ROWS_OOR_CASES = [
    ("block_two_ctas", n, variant)
    for n, variant in ((256, "32x32b"), (256, "16x128b"))
]

LDST_TWOCTA_DIRECT_HALF_ROWS_POSITIVE_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for layout_name, (n, variant) in product(
        ("block_two_ctas", "mmav5_twocta"),
        ((64, "16x128b"), (128, "auto"), (128, "16x256b")),
    )
]

LDST_DIRECT_HIGHER_RANK_GET_REG_LAYOUT_POSITIVE_CASES = [
    ("auto", "32x32b.x64.b32"),
    ("16x128b", LDST_SHAPE_MAP["16x128b"][128]),
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
        "block_two_ctas_bitcast_subslice",
        _make_tmem_linear_layout_block(128, 128, two_ctas=True),
        _make_tmem_linear_layout_64x32_block(two_ctas=True),
        (
            "unsupported tensor memory memdesc_subslice view",
            "is not affine in the selected tensor-memory linear layout",
            "basis step 2 crosses a physical layout boundary",
            "carry-dependent descriptor view",
        ),
    ),
]

M64_SPLITN_DTYPES = (("f32", torch.float32), ("i32", torch.int32))

M64_SPLITN_BASE_CASES = []
for n in (2, 64, 256):
    if n == 2:
        M64_SPLITN_BASE_CASES.append((n, 2, [(0, 0)]))
    else:
        x = n // 4
        M64_SPLITN_BASE_CASES.append((n, x, [(0, x), (2 * x, x)]))

M64_SPLITN_CASES = [
    (dtype_name, torch_dtype, n, splitn_x, offset_imm_pairs)
    for dtype_name, torch_dtype in M64_SPLITN_DTYPES
    for n, splitn_x, offset_imm_pairs in M64_SPLITN_BASE_CASES
]

# M64 split-N keeps every row/column permutation class at one representative
# N/variant and samples the edge widths/explicit variants on the layouts that
# previously exposed backend fallback bugs.
M64_ROWCOL_PERMUTED_CASE_SPECS = [
    ("f32", torch.float32, row_perm_kind, col_perm_kind, 128, "32x32b_splitn")
    for row_perm_kind, col_perm_kind in (
        ("identity", "identity"),
        ("rotate1", "identity"),
        ("identity", "reverse"),
    )
] + [
    ("f32", torch.float32, row_perm_kind, col_perm_kind, n, variant)
    for row_perm_kind, col_perm_kind, n, variant in (
        ("rotate1", "identity", 2, "32x32b_splitn"),
        ("identity", "reverse", 256, "16x32bx2"),
    )
] + [
    ("i32", torch.int32, "reverse", "even_odd", 128, "16x32bx2")
]

M64_ROWCOL_PERMUTED_CASES = [
    (dtype_name, torch_dtype, row_perm_kind, col_perm_kind, n, variant)
    for dtype_name, torch_dtype, row_perm_kind, col_perm_kind, n, variant in M64_ROWCOL_PERMUTED_CASE_SPECS
]

M64_ROWCOL_PERMUTED_AUTO_CASES = [
    (dtype_name, torch_dtype, row_perm_kind, col_perm_kind, n)
    for dtype_name, torch_dtype in M64_SPLITN_DTYPES
    for row_perm_kind, col_perm_kind, n in (
        ("rotate1", "identity", 2),
        ("identity", "reverse", 256),
    )
]

M64_16BIT_AUTO_SPLITN_CASES = (
    ("f16", torch.float16, 2),
    ("bf16", torch.bfloat16, 64),
)

LDST_DESCRIPTOR_RANK5_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for layout_name, n, variant in product(LDST_LAYOUTS.keys(), (64, ), LDST_VARIANTS)
]

LDST_TWOCTA_DESCRIPTOR_RANK5_CASES = [
    (layout_name, n, variant, LDST_SHAPE_MAP[variant][n])
    for layout_name, n, variant in product(LDST_TWOCTA_LAYOUTS.keys(), (64, ), LDST_VARIANTS)
]

LDST_DESCRIPTOR_RANK5_SMALL_LAYOUT_CASES = (
    ("single_identity", "single", "identity", 128, 1),
    ("single_mixed", "single", "mixed", 128, 1),
    ("twocta_block", "twocta", "block_two_ctas", 256, 2),
    ("twocta_mmav5", "twocta", "mmav5_twocta", 256, 2),
)

LDST_DESCRIPTOR_RANK5_SMALL_CASE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, case_name, layout_group, layout_name, m, num_ctas, 64, "auto")
            for case_name, layout_group, layout_name, m, num_ctas in LDST_DESCRIPTOR_RANK5_SMALL_LAYOUT_CASES
        ] + [
            ("f32", torch.float32, "single_identity", "single", "identity", 128, 1, 64, variant)
            for variant in ("32x32b", "16x128b")
        ] + [
            ("f32", torch.float32, "twocta_block", "twocta", "block_two_ctas", 256, 2, 64, variant)
            for variant in ("32x32b", "16x256b")
        ] + [
            ("f32", torch.float32, case_name, layout_group, layout_name, m, num_ctas, n, "16x128b")
            for case_name, layout_group, layout_name, m, num_ctas, n in (
                ("single_identity", "single", "identity", 128, 1, 32),
                ("single_mixed", "single", "mixed", 128, 1, 128),
                ("twocta_block", "twocta", "block_two_ctas", 256, 2, 32),
                ("twocta_mmav5", "twocta", "mmav5_twocta", 256, 2, 128),
            )
        ] + [
            ("i32", torch.int32, "single_identity", "single", "identity", 128, 1, 64, "32x32b"),
            ("i32", torch.int32, "twocta_block", "twocta", "block_two_ctas", 256, 2, 64, "32x32b"),
        ]
    )
)

LDST_DESCRIPTOR_RANK5_SMALL_CASES = [
    (dtype_name, torch_dtype, case_name, layout_group, layout_name, m, num_ctas, n, variant,
     LDST_SHAPE_MAP[variant][n])
    for dtype_name, torch_dtype, case_name, layout_group, layout_name, m, num_ctas, n, variant in
    LDST_DESCRIPTOR_RANK5_SMALL_CASE_SPECS
]

LDST_DESCRIPTOR_RANK5_N256_CASE_SPECS = list(
    dict.fromkeys(
        [
            ("f32", torch.float32, case_name, layout_group, layout_name, m, num_ctas, "auto")
            for case_name, layout_group, layout_name, m, num_ctas in LDST_DESCRIPTOR_RANK5_SMALL_LAYOUT_CASES
        ] + [
            ("f32", torch.float32, "single_identity", "single", "identity", 128, 1, variant)
            for variant in ("32x32b", "16x128b")
        ] + [
            ("f32", torch.float32, "twocta_block", "twocta", "block_two_ctas", 256, 2, variant)
            for variant in ("16x256b",)
        ] + [
            ("i32", torch.int32, "single_identity", "single", "identity", 128, 1, "32x32b"),
            ("i32", torch.int32, "twocta_block", "twocta", "block_two_ctas", 256, 2, "32x32b"),
        ]
    )
)

LDST_DESCRIPTOR_RANK5_N256_CASES = [
    (dtype_name, torch_dtype, case_name, layout_group, layout_name, m, num_ctas, variant, LDST_SHAPE_MAP[variant][256])
    for dtype_name, torch_dtype, case_name, layout_group, layout_name, m, num_ctas, variant in
    LDST_DESCRIPTOR_RANK5_N256_CASE_SPECS
]

CP_NO_SCALES_CASES = [
    (128, 16, 16, 32),
    (128, 32, 32, 32),
    (128, 64, 32, 32),
    (128, 64, 64, 32),
    (128, 128, 32, 32),
    (128, 128, 128, 32),
    (128, 256, 64, 32),
    (128, 256, 256, 32),
    (256, 16, 16, 32),
    (256, 32, 32, 32),
    (256, 64, 64, 32),
    (256, 128, 64, 32),
    (256, 128, 128, 32),
    (256, 256, 128, 32),
    (256, 256, 256, 32),
]

CP_LINEAR_NO_SCALES_CASES = [
    (128, 128, 32, 16),
    (128, 128, 128, 16),
    (128, 256, 32, 32),
    (128, 256, 128, 32),
    (256, 16, 32, 4),
    (256, 32, 32, 8),
    (256, 32, 128, 8),
    (256, 64, 64, 16),
    (256, 128, 32, 32),
    (256, 128, 128, 32),
]

CP_LINEAR_NO_SCALES_32BIT_DTYPE_CASES = [
    (dtype_name, torch_dtype, m, n, swizzle, expected_count)
    for dtype_name, torch_dtype in (("f32", torch.float32), ("i32", torch.int32))
    for m, n, swizzle, expected_count in CP_LINEAR_NO_SCALES_CASES
]

CP_INDEXED_VIEW_128X128_CASES = (128, )

CP_LINEAR_INDEXED_VIEW_CASES = [
    *[
        (
            f"128x128b_{dtype_name}",
            torch_dtype,
            128,
            4,
            _make_tmem_copy_128x128_shared_layout(),
            1,
            "tcgen05.cp.cta_group::1.128x128b",
        )
        for dtype_name, torch_dtype in (("f32", torch.float32), ("i32", torch.int32))
    ],
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

CP_NO_SCALES_SUBWORD_DTYPES = (
    ("f16", torch.float16),
    ("bf16", torch.bfloat16),
    ("i16", torch.int16),
    ("i8", torch.int8),
)

CP_NO_SCALES_SUBWORD_BITWIDTHS = {
    "f16": 16,
    "bf16": 16,
    "i16": 16,
    "i8": 8,
}

CP_TWOCTA_LINEAR_INDEXED_VIEW_CASES = [
    (dtype_name, torch_dtype, 256, n, swizzle, expected_count, "tcgen05.cp.cta_group::2.128x256b")
    for dtype_name, torch_dtype in (("f32", torch.float32), ("i32", torch.int32))
    for n, swizzle, expected_count in ((64, 32, 8), (128, 128, 16))
] + [
    (
        dtype_name,
        torch_dtype,
        256,
        n,
        swizzle,
        n * CP_NO_SCALES_SUBWORD_BITWIDTHS[dtype_name] // 256,
        "tcgen05.cp.cta_group::2.128x256b",
    )
    for dtype_name, torch_dtype, n, swizzle in (
        ("f16", torch.float16, 128, 32),
        ("bf16", torch.bfloat16, 128, 64),
        ("i16", torch.int16, 128, 128),
        ("i8", torch.int8, 128, 32),
    )
]

CP_LINEAR_SUBSLICE_VIEW_CASES = [
    (dtype_name, torch_dtype, 128, n, swizzle, expected_count, "tcgen05.cp.cta_group::1.128x256b")
    for dtype_name, torch_dtype in (("f32", torch.float32), ("i32", torch.int32))
    for n, expected_count in ((128, 16), (256, 32))
    for swizzle in (32, 64, 128)
]

CP_TWOCTA_LINEAR_SUBSLICE_VIEW_CASES = [
    (dtype_name, torch_dtype, 256, n, swizzle, expected_count, "tcgen05.cp.cta_group::2.128x256b")
    for dtype_name, torch_dtype in (("f32", torch.float32), ("i32", torch.int32))
    for n, swizzle, expected_count in ((128, 32, 16), (256, 128, 32))
] + [
    (
        dtype_name,
        torch_dtype,
        256,
        n,
        swizzle,
        n * CP_NO_SCALES_SUBWORD_BITWIDTHS[dtype_name] // 256,
        "tcgen05.cp.cta_group::2.128x256b",
    )
    for dtype_name, torch_dtype, n, swizzle in (
        ("f16", torch.float16, 128, 32),
        ("bf16", torch.bfloat16, 256, 64),
        ("i16", torch.int16, 256, 128),
        ("i8", torch.int8, 256, 32),
    )
]

CP_NO_SCALES_128X128_DTYPES = (("f32", torch.float32), ("i32", torch.int32))

CP_NO_SCALES_TWOCTA_32BIT_GEOMETRY_CASES = (
    (16, 32, 2),
    (32, 128, 4),
    (128, 64, 16),
    (256, 128, 32),
)

CP_NO_SCALES_TWOCTA_CASES = [
    (layout_kind, dtype_name, torch_dtype, n, swizzle, expected_count)
    for layout_kind, (dtype_name, torch_dtype), (n, swizzle, expected_count) in product(
        ("linear", "legacy"), CP_NO_SCALES_128X128_DTYPES, CP_NO_SCALES_TWOCTA_32BIT_GEOMETRY_CASES
    )
] + [
    (
        "linear",
        dtype_name,
        torch_dtype,
        n,
        swizzle,
        n * CP_NO_SCALES_SUBWORD_BITWIDTHS[dtype_name] // 256,
    )
    for dtype_name, torch_dtype, n, swizzle in (
        ("f16", torch.float16, 128, 32),
        ("bf16", torch.bfloat16, 128, 64),
        ("i16", torch.int16, 256, 128),
        ("i8", torch.int8, 256, 32),
    )
]

CP_LINEAR_NO_SCALES_SUBWORD_CASES = [
    (
        dtype_name,
        torch_dtype,
        128,
        n,
        32,
        n * CP_NO_SCALES_SUBWORD_BITWIDTHS[dtype_name] // 256,
    )
    for dtype_name, torch_dtype in CP_NO_SCALES_SUBWORD_DTYPES
    for n in (128, 256)
]

CP_NO_SCALES_SWIZZLE_CASES = [
    (m, n, block_n, swizzle)
    for swizzle in (32, 64, 128)
    for (m, n, block_n) in ((128, 128, 128), (128, 256, 256), (256, 128, 64))
]

CP_NO_SCALES_128X128_CASES = [
    (layout_kind, dtype_name, torch_dtype, 128)
    for layout_kind, (dtype_name, torch_dtype) in product(TMEM_LAYOUT_KIND_CASES, CP_NO_SCALES_128X128_DTYPES)
]

CP_NO_SCALES_128X128_SUBWORD_EXACT_CASES = (
    ("f16", torch.float16, 8),
    ("i8", torch.int8, 16),
)

CP_NO_SCALES_TWOCTA_128X128_CASES = [
    (layout_kind, dtype_name, torch_dtype)
    for layout_kind, (dtype_name, torch_dtype) in product(
        (TMEM_LINEAR_LAYOUT_KIND, TMEM_ENCODING_LAYOUT_KIND), CP_NO_SCALES_128X128_DTYPES
    )
]

CP_NO_SCALES_WARPX2_DTYPES = (("f32", torch.float32), ("i32", torch.int32))

CP_NO_SCALES_WARPX2_SUBWORD_UNSUPPORTED_CASES = [
    (case_name, cta_group, tmem_layout, dtype_name, torch_dtype)
    for case_name, cta_group, tmem_layout in (
        ("single_01_23", 1, _make_tmem_copy_warpx2_tmem_layout()),
        ("single_02_13", 1, _make_tmem_copy_warpx2_tmem_layout_02_13()),
        ("twocta_01_23", 2, _make_tmem_copy_warpx2_tmem_layout_twocta()),
        ("twocta_02_13", 2, _make_tmem_copy_warpx2_tmem_layout_02_13_twocta()),
    )
    for dtype_name, torch_dtype in CP_NO_SCALES_SUBWORD_DTYPES
    if cta_group == 1 or dtype_name != "i8"
]

CP_SCALES_WARPX4_FORMAT_PAIRS = [
    ("mxfp8", "mxfp8"),
    ("mxfp4", "mxfp4"),
    ("mxfp8", "mxfp4"),
    ("mxfp4", "mxfp8"),
    ("nvfp4", "nvfp4"),
]

SCALED_MMA_ROOT_FORMAT_CASES = list(
    dict.fromkeys(
        [
            (a_format, b_format, 128, 128, "linear") for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS
        ] + [
            ("mxfp8", "mxfp8", n, k, acc_layout_kind)
            for n, k, acc_layout_kind in (
                (32, 128, "linear"),
                (64, 256, "legacy"),
                (128, 256, "linear"),
                (256, 128, "legacy"),
                (256, 256, "linear"),
            )
        ] + [
            ("nvfp4", "nvfp4", 256, 256, "legacy"),
        ]
    )
)

SCALED_MMA_ROOT_USE_ACC_CASES = [
    (a_format, b_format, 128, 128, "linear")
    for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS
] + [
    ("mxfp8", "mxfp8", n, k, acc_layout_kind)
    for n, k, acc_layout_kind in (
        (32, 128, "linear"),
        (64, 256, "legacy"),
        (256, 256, "linear"),
    )
]

SCALED_MMA_INDEXED_ACC_FORMAT_CASES = list(
    dict.fromkeys(
        [
            (a_format, b_format, 64, 128, "linear") for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS
        ] + [
            ("mxfp8", "mxfp8", n, k, parent_layout_kind)
            for n, k, parent_layout_kind in (
                (32, 128, "linear"),
                (64, 256, "legacy"),
                (128, 128, "legacy"),
                (128, 256, "linear_unit_parent"),
                (256, 128, "linear_unit_parent"),
                (256, 256, "legacy_unit_parent"),
            )
        ] + [
            ("nvfp4", "nvfp4", 256, 256, "linear_unit_parent"),
        ]
    )
)


SCALED_MMA_LHS_SUBSLICE_FORMAT_CASES = [
    (a_format, b_format, acc_layout_kind)
    for a_format, b_format in (
        ("mxfp8", "mxfp8"),
        ("mxfp8", "mxfp4"),
        ("mxfp4", "mxfp4"),
        ("nvfp4", "nvfp4"),
    )
    for acc_layout_kind in TMEM_LAYOUT_KIND_CASES
]

SCALED_MMA_LHS_SUBSLICE_NK_CASES = list(
    dict.fromkeys(
        [
            (a_format, b_format, 128, 128, acc_layout_kind)
            for a_format, b_format, acc_layout_kind in SCALED_MMA_LHS_SUBSLICE_FORMAT_CASES
        ] + [
            ("mxfp8", "mxfp8", n, k, acc_layout_kind)
            for n, k, acc_layout_kind in (
                (32, 128, "linear"),
                (64, 256, "legacy"),
                (256, 256, "linear"),
            )
        ]
    )
)

SCALED_MMA_LHS_TILE_PERMUTED_NK_CASES = list(
    dict.fromkeys(
        [
            (a_format, b_format, 128, 128, acc_layout_kind)
            for a_format, b_format, acc_layout_kind in SCALED_MMA_LHS_SUBSLICE_FORMAT_CASES
        ] + [
            ("mxfp8", "mxfp8", n, k, acc_layout_kind)
            for n, k, acc_layout_kind in (
                (32, 128, "linear"),
                (64, 256, "legacy"),
                (256, 256, "linear"),
            )
        ]
    )
)

SCALED_MMA_LHS_TILE_PERMUTED_MIXED_FP4A_UNSUPPORTED_CASES = [
    (n, acc_layout_kind) for n, acc_layout_kind in product((32, 64, 128, 256), TMEM_LAYOUT_KIND_CASES)
]

SCALED_MMA_LHS_SUBSLICE_MIXED_FP4A_UNSUPPORTED_CASES = [
    (n, k, acc_layout_kind)
    for n, k, acc_layout_kind in product((32, 64, 128, 256), (128, 256), TMEM_LAYOUT_KIND_CASES)
]

SCALED_MMA_ACC_SUBSLICE_N_CASES = [
    (32, 0, 128),
    (32, 32, 256),
    (64, 0, 128),
    (64, 64, 256),
    (128, 0, 128),
    (128, 128, 256),
]

SCALED_MMA_ACC_SUBSLICE_USE_ACC_CASES = list(
    dict.fromkeys(
        [
            (a_format, b_format, 64, 0, 128) for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS
        ] + [
            (a_format, b_format, 64, 64, 128) for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS
        ] + [
            ("mxfp8", "mxfp8", 32, 0, 128),
            ("mxfp8", "mxfp8", 32, 32, 256),
            ("mxfp8", "mxfp8", 128, 0, 128),
            ("mxfp8", "mxfp8", 128, 128, 256),
        ]
    )
)

SCALED_MMA_ACC_TILE_PERMUTED_K_CASES = [
    (a_format, b_format, k)
    for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS
    for k in (128, 256)
]

SCALED_MMA_ACC_TILE_PERMUTED_N32_CASES = [
    (a_format, b_format, k)
    for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS
    for k in (128, 256)
]

SCALED_MMA_ACC_TILE_PERMUTED_NARROW_CASES = [
    (a_format, b_format, n, tile_n, k)
    for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS
    for n, tile_n in ((32, 8), (64, 16))
    for k in (128, 256)
]

SCALED_MMA_ACC_IDENTITY_NARROW_CASES = [
    (a_format, b_format, 16, k)
    for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS
    for k in (128, 256)
]

SCALED_MMA_ACC_IDENTITY_NARROW_VIEW_CASES = [
    ("mxfp8", "mxfp8", 16, 128, "linear"),
    ("mxfp8", "mxfp8", 16, 128, "linear_unit_parent"),
]

SCALED_MMA_TWOCTA_ACC_SUBSLICE_K_CASES = list(
    dict.fromkeys(
        [
            (a_format, b_format, 64, 0, 128, False) for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS
        ] + [
            (a_format, b_format, 64, 64, 128, True) for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS
        ] + [
            ("mxfp8", "mxfp8", block_n, slice_start, block_k, multicast)
            for block_n, slice_start, block_k, multicast in (
                (32, 0, 128, False),
                (32, 32, 256, True),
                (128, 0, 128, True),
                (128, 128, 256, False),
            )
        ]
    )
)

CP_SCALES_WARPX4_SCALED_MMA_CASES = list(
    dict.fromkeys(
        [
            (a_format, b_format, 128, 128, 1, False, "linear")
            for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS
        ] + [
            ("mxfp8", "mxfp8", block_n, block_k, num_ctas, multicast, acc_layout_kind)
            for block_n, block_k, num_ctas, multicast, acc_layout_kind in (
                (128, 256, 1, True, "legacy"),
                (256, 128, 2, False, "linear"),
                (256, 256, 2, True, "legacy"),
            )
        ] + [
            ("nvfp4", "nvfp4", 256, 256, 2, True, "linear"),
        ]
    )
)
# Accumulator-add semantics do not need to duplicate the full scaled-MMA copy
# geometry matrix; keep format coverage plus representative CTA/multicast and
# large-N/K rows.
CP_SCALES_WARPX4_SCALED_MMA_USE_ACC_CASES = [
    (a_format, b_format, 128, 128, 1, False, "linear")
    for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS
] + [
    ("mxfp8", "mxfp8", 256, 256, 2, True, "legacy"),
    ("nvfp4", "nvfp4", 128, 256, 2, False, "linear"),
]

CP_SCALES_WARPX4_GEOMETRY_CASES = [
    (128, 128, False, 1, "linear"),
    (128, 256, True, 1, "legacy"),
    (256, 128, False, 2, "linear"),
    (256, 256, True, 2, "legacy"),
    (128, 128, True, 2, "linear"),
    (256, 256, False, 1, "legacy"),
]

SUBWORD_LDST_SHAPE_MAP_BY_BITS = {
    16: {
        "auto": {64: "32x32b.x32.b32", 128: "32x32b.x64.b32", 256: "32x32b.x128.b32"},
        "32x32b": {64: "32x32b.x32.b32", 128: "32x32b.x64.b32", 256: "32x32b.x128.b32"},
        "16x64b": {64: "16x64b.x16.b32", 128: "16x64b.x32.b32", 256: "16x64b.x64.b32"},
        "16x128b": {64: "16x128b.x8.b32", 128: "16x128b.x16.b32", 256: "16x128b.x32.b32"},
        "16x256b": {64: "16x256b.x4.b32", 128: "16x256b.x8.b32", 256: "16x256b.x16.b32"},
    },
    8: {
        "auto": {64: "32x32b.x16.b32", 128: "32x32b.x32.b32", 256: "32x32b.x64.b32"},
        "32x32b": {64: "32x32b.x16.b32", 128: "32x32b.x32.b32", 256: "32x32b.x64.b32"},
        "16x64b": {64: "16x64b.x8.b32", 128: "16x64b.x16.b32", 256: "16x64b.x32.b32"},
        "16x128b": {64: "16x128b.x4.b32", 128: "16x128b.x8.b32", 256: "16x128b.x16.b32"},
        "16x256b": {64: "16x256b.x2.b32", 128: "16x256b.x4.b32", 256: "16x256b.x8.b32"},
    },
}

SUBWORD_LDST_DTYPES = (
    ("f16", torch.float16, 16),
    ("bf16", torch.bfloat16, 16),
    ("i16", torch.int16, 16),
    ("i8", torch.int8, 8),
)

SUBWORD_LDST_CASE_SPECS = list(
    dict.fromkeys(
        [
            (dtype_name, torch_dtype, bitwidth, 128, "auto")
            for dtype_name, torch_dtype, bitwidth in SUBWORD_LDST_DTYPES
        ] + [
            ("f16", torch.float16, 16, 128, "16x128b"),
            ("i8", torch.int8, 8, 128, "16x256b"),
        ]
    )
)

SUBWORD_LDST_CASES = [
    (
        dtype_name,
        torch_dtype,
        "identity",
        n,
        variant,
        SUBWORD_LDST_SHAPE_MAP_BY_BITS[bitwidth][variant][n],
    )
    for dtype_name, torch_dtype, bitwidth, n, variant in SUBWORD_LDST_CASE_SPECS
]

SUBWORD_LDST_DESCRIPTOR_CASES = SUBWORD_LDST_CASES

X1_SUBWORD_LDST_16BIT_DTYPES = (
    ("f16", torch.float16),
    ("bf16", torch.bfloat16),
    ("i16", torch.int16),
)

X1_SUBWORD_LDST_16BIT_LAYOUTS = (
    ("linear_packed", 2, lambda: _make_tmem_linear_layout(128, 2), "32x32b.x1.b32", "32x32b.x1.b32"),
    ("legacy_packed", 2, lambda: TensorMemoryLayout((128, 2), col_stride=1), "32x32b.x1.b32", "32x32b.x1.b32"),
    (
        "legacy_unpacked",
        2,
        lambda: TensorMemoryLayout((128, 2), col_stride=2),
        "32x32b.x1.unpack::16b.b32",
        "32x32b.x1.pack::16b.b32",
    ),
)

X1_SUBWORD_LDST_CASES = [
    (
        "f16",
        torch.float16,
        "linear_packed",
        *X1_SUBWORD_LDST_16BIT_LAYOUTS[0][1:],
    ),
    (
        "f16",
        torch.float16,
        "legacy_unpacked",
        *X1_SUBWORD_LDST_16BIT_LAYOUTS[2][1:],
    ),
] + [
    (
        "i8",
        torch.int8,
        "linear_packed",
        4,
        lambda: _make_tmem_linear_layout(128, 4),
        "32x32b.x1.b32",
        "32x32b.x1.b32",
    ),
    (
        "i8",
        torch.int8,
        "legacy_padded_stride4_n4",
        4,
        lambda: TensorMemoryLayout((128, 4), col_stride=4),
        "32x32b.x4.b32",
        "32x32b.x4.b32",
    ),
]

X1_SUBWORD_LDST_VARIANTS = ("auto", "32x32b")

X1_SUBWORD_LDST_TWOCTA_CASES = (
    ("f16", torch.float16, 2),
    ("i8", torch.int8, 4),
)

X1_F32_LDST_CASES = [
    ("linear_onecta", 128, 1, lambda: _make_tmem_linear_layout(128, 1)),
    ("linear_twocta", 256, 2, lambda: _make_tmem_linear_layout_mmav5_twocta(256, 1)),
]

X1_F32_LDST_VARIANTS = ("auto", "32x32b")

X1_F32_UNSUPPORTED_VARIANTS = ("16x128b",)

SCALES_LDST_CASES = [
    (16, 8, 8, 1, tuple(), _expected_ldst_ops("16x32bx2.x1.b32", [0])),
    (128, 64, 4, 2, ((1, 0),), _expected_ldst_ops("16x32bx2.x32.b32", [0])),
    (256, 32, 4, 2, ((1, 0),), _expected_ldst_ops("16x32bx2.x32.b32", [0])),
    (256, 64, 4, 2, ((1, 0),), _expected_ldst_ops("16x32bx2.x64.b32", [0])),
]

SCALES_LDST_EXPLICIT_VARIANT_CASES = [
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
    (16, 8, 8, "16x32bx2", _expected_ldst_ops("16x32bx2.x1.b32", [0])),
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

SCALES_LDST_AUTO_VARIANT_CASES = [
    (16, 4, 4, "auto", _expected_ldst_ops("16x32bx2.x1.b32", [0])),
    (16, 8, 8, "auto", _expected_ldst_ops("16x32bx2.x1.b32", [0])),
    (32, 4, 8, "auto", _expected_ldst_ops("16x32bx2.x1.b32", [0])),
    (64, 4, 4, "auto", _expected_ldst_ops("16x64b.x1.b32", [0, 0])),
    (64, 8, 4, "auto", _expected_ldst_ops("16x128b.x1.b32", [0, 0])),
    (64, 16, 4, "auto", _expected_ldst_ops("16x256b.x1.b32", [0, 0])),
    (64, 16, 8, "auto", _expected_ldst_ops("32x32b.x4.b32", [0])),
    (64, 32, 4, "auto", _expected_ldst_ops("16x256b.x2.b32", [0, 0])),
    (64, 32, 8, "auto", _expected_ldst_ops("32x32b.x8.b32", [0])),
    (128, 4, 4, "auto", _expected_ldst_ops("16x128b.x1.b32", [0, 0])),
    (128, 8, 4, "auto", _expected_ldst_ops("16x256b.x1.b32", [0, 0])),
    (128, 8, 8, "auto", _expected_ldst_ops("32x32b.x4.b32", [0])),
    (128, 16, 4, "auto", _expected_ldst_ops("16x256b.x2.b32", [0, 0])),
    (128, 16, 8, "auto", _expected_ldst_ops("32x32b.x8.b32", [0])),
    (128, 32, 4, "auto", _expected_ldst_ops("16x256b.x4.b32", [0, 0])),
    (128, 32, 8, "auto", _expected_ldst_ops("32x32b.x16.b32", [0])),
    (16, 8, 4, "auto", _expected_ldst_ops("16x32bx2.x1.b32", [0])),
    (16, 16, 4, "auto", _expected_ldst_ops("16x32bx2.x1.b32", [0, 2])),
    (16, 16, 8, "auto", _expected_ldst_ops("16x32bx2.x1.b32", [0])),
    (32, 8, 4, "auto", _expected_ldst_ops("16x32bx2.x1.b32", [0, 2])),
    (32, 8, 8, "auto", _expected_ldst_ops("32x32b.x1.b32", [0])),
]

SCALES_LDST_N_SHARDED_VARIANT_CASES = [
    (m, n, 4, instr_variant, _expected_scales_ldst_n_sharded_ops(m, n, instr_variant))
    for m, n, instr_variant in product(
        (64, 128, 256),
        (4, 8, 16, 32),
        SCALES_LDST_N_SHARDED_VARIANT_WIDTHS,
    )
    if m * n >= _scales_ldst_n_sharded_min_elements(instr_variant)
]

SCALES_LDST_N_SHARDED_VARIANT_CLEAN_UNSUPPORTED_CASES = [
    (
        m,
        n,
        4,
        instr_variant,
        f"TMEM layout 'constexpr[{instr_variant}]' unsupported for descriptor view",
        False,
    )
    for m, n, instr_variant in product(
        (64, 128),
        (4, 8),
        ("16x128b", "16x256b"),
    )
    if m * n < _scales_ldst_n_sharded_min_elements(instr_variant)
]

SCALES_LDST_AUTO_VARIANT_REPRESENTATIVE_KEYS = {
    (16, 4, 4, "auto"),
    (64, 4, 4, "auto"),
    (64, 16, 4, "auto"),
    (128, 32, 8, "auto"),
}

SCALES_LDST_EXPLICIT_VARIANT_REPRESENTATIVE_KEYS = {
    (16, 4, 4, "32x32b"),
    (64, 16, 8, "32x32b"),
    (16, 4, 4, "16x32bx2"),
    (16, 8, 8, "16x32bx2"),
    (32, 16, 4, "16x32bx2"),
}

SCALES_LDST_N_SHARDED_VARIANT_REPRESENTATIVE_KEYS = {
    (64, 16, 4, "16x64b"),
    (64, 32, 4, "16x128b"),
    (128, 32, 4, "16x256b"),
    (256, 16, 4, "16x256b"),
}

SCALES_LDST_VARIANT_CASES = (
    [case for case in SCALES_LDST_AUTO_VARIANT_CASES if case[:4] in SCALES_LDST_AUTO_VARIANT_REPRESENTATIVE_KEYS]
    + [
        case
        for case in SCALES_LDST_EXPLICIT_VARIANT_CASES
        if case[:4] in SCALES_LDST_EXPLICIT_VARIANT_REPRESENTATIVE_KEYS
    ] + [
        case
        for case in SCALES_LDST_N_SHARDED_VARIANT_CASES
        if case[:4] in SCALES_LDST_N_SHARDED_VARIANT_REPRESENTATIVE_KEYS
    ]
)

SCALES_LDST_VARIANT_CLEAN_UNSUPPORTED_CASES = SCALES_LDST_N_SHARDED_VARIANT_CLEAN_UNSUPPORTED_CASES

SCALES_LDST_DESCRIPTOR_VIEW_CASES = [
    (
        128,
        32,
        4,
        tuple(),
        "32x32b",
        _expected_scales_ldst_descriptor_view_ops(
            "16x32bx2.x32.b32", "16x64b.x16.b32", view_offsets=(0, 0)
        ),
    ),
    (
        128,
        64,
        4,
        tuple(),
        "32x32b",
        _expected_scales_ldst_descriptor_view_ops(
            "16x32bx2.x64.b32", "16x64b.x32.b32", view_offsets=(0, 0)
        ),
    ),
    (
        256,
        64,
        4,
        tuple(),
        "32x32b",
        _expected_scales_ldst_descriptor_view_ops(
            "16x32bx2.x128.b32", "16x64b.x64.b32", view_offsets=(0, 0)
        ),
    ),
]

SCALES_LDST_DESCRIPTOR_VIEW_CGA_32X32B_CASES = [
    (
        M,
        N,
        4,
        2,
        ((1, 0),),
        "32x32b",
        _expected_scales_ldst_descriptor_view_ops(
            f"16x32bx2.x{M * N // 256}.b32",
            f"16x64b.x{M * N // 512}.b32",
            view_offsets=(0, 0),
        ),
    )
    for M, N in product((128, 256), (4, 8, 16, 32, 64, 128))
]

SCALES_LDST_DESCRIPTOR_VIEW_CGA_N_SHARDED_CASES = [
    (
        M,
        N,
        4,
        2,
        ((1, 0),),
        instr_variant,
        _expected_scales_ldst_descriptor_view_ops(
            f"{instr_variant}.x{M * N // (8 * width)}.b32",
            f"{instr_variant}.x{M * N // (8 * width)}.b32",
            (0, 0),
            view_offsets=(0, 0),
        ),
    )
    for M, N, (instr_variant, width) in product(
        (128, 256),
        (4, 8, 16, 32, 64, 128),
        SCALES_LDST_N_SHARDED_VARIANT_WIDTHS.items(),
    )
    if M * N >= 8 * width
]

SCALES_LDST_DESCRIPTOR_VIEW_CGA_32X32B_REPRESENTATIVE_KEYS = {
    (128, 4, 4, 2, ((1, 0),), "32x32b"),
    (128, 32, 4, 2, ((1, 0),), "32x32b"),
    (256, 64, 4, 2, ((1, 0),), "32x32b"),
}

SCALES_LDST_DESCRIPTOR_VIEW_CGA_N_SHARDED_REPRESENTATIVE_KEYS = {
    (128, 8, 4, 2, ((1, 0),), "16x64b"),
    (128, 32, 4, 2, ((1, 0),), "16x256b"),
    (256, 16, 4, 2, ((1, 0),), "16x128b"),
    (256, 64, 4, 2, ((1, 0),), "16x256b"),
}

SCALES_LDST_DESCRIPTOR_VIEW_CGA_CASES = (
    [
        (
            64,
            64,
            4,
            2,
            ((1, 0),),
            "32x32b",
            _expected_scales_ldst_descriptor_view_ops(
                "16x32bx2.x1.b32", "16x32bx2.x1.b32", tuple(range(0, 32, 2))
            ),
        ),
        (
            128,
            64,
            4,
            2,
            ((1, 0),),
            "16x32bx2",
            _expected_scales_ldst_descriptor_view_ops(
                "16x32bx2.x32.b32", "16x64b.x16.b32", (0,), view_offsets=(0, 0)
            ),
        )
    ] + [
        case
        for case in SCALES_LDST_DESCRIPTOR_VIEW_CGA_32X32B_CASES
        if case[:6] in SCALES_LDST_DESCRIPTOR_VIEW_CGA_32X32B_REPRESENTATIVE_KEYS
    ] + [
        case
        for case in SCALES_LDST_DESCRIPTOR_VIEW_CGA_N_SHARDED_CASES
        if case[:6] in SCALES_LDST_DESCRIPTOR_VIEW_CGA_N_SHARDED_REPRESENTATIVE_KEYS
    ]
)

LD_RED_LINEAR_CASES = [
    ("identity", 128, 32, 4, "32x32b.x32"),
    ("identity", 128, 128, 4, "32x32b.x128"),
    ("identity", 128, 256, 4, "32x32b.x64"),
    ("identity", 256, 64, 8, "32x32b.x64"),
    ("legacy_equivalent_256", 256, 128, 8, "32x32b.x128"),
]

LD_RED_EXPECTED_OP_COUNT = {32: 1, 64: 1, 128: 1, 256: 4}
LD_RED_EXPECTED_OFFSETS = {32: [0], 64: [0], 128: [0], 256: [0, 64, 128, 192]}


def _assert_ld_red_opcode_pairs(
    compiled,
    N,
    expected_shape,
    red_op,
    use_abs,
    propagate_nan,
    expected_offsets=None,
):
    ptx = compiled.asm["ptx"]
    llir = compiled.asm["llir"]
    ptx_red_pairs = [
        pair
        for pair in _extract_tcgen05_opcode_offsets(ptx, opcodes=("ld", ))
        if ".ld.red." in pair[0]
    ]
    llir_red_pairs = [
        pair
        for pair in _extract_tcgen05_opcode_offsets(llir, opcodes=("ld", ))
        if ".ld.red." in pair[0]
    ]
    assert ptx_red_pairs == llir_red_pairs
    if expected_offsets is None:
        expected_offsets = LD_RED_EXPECTED_OFFSETS[N]
    assert len(ptx_red_pairs) == len(expected_offsets)
    assert [offset for _, offset in ptx_red_pairs] == list(expected_offsets)
    assert ptx.count("tcgen05.wait::st.sync.aligned;") == 1
    assert ptx.count("tcgen05.wait::ld.sync.aligned;") == 1
    assert llir.count("tail call void @llvm.nvvm.tcgen05.wait.st()") == 1
    assert llir.count("tail call void @llvm.nvvm.tcgen05.wait.ld()") == 1
    assert ptx.index("tcgen05.st.sync.aligned") < ptx.index("tcgen05.wait::st.sync.aligned")
    assert ptx.index("tcgen05.wait::st.sync.aligned") < ptx.index("tcgen05.ld.red.sync.aligned")
    assert ptx.index("tcgen05.ld.red.sync.aligned") < ptx.index("tcgen05.wait::ld.sync.aligned")
    assert llir.index("tcgen05.st.sync.aligned") < llir.index("@llvm.nvvm.tcgen05.wait.st()")
    assert llir.index("@llvm.nvvm.tcgen05.wait.st()") < llir.index("tcgen05.ld.red.sync.aligned")
    assert llir.index("tcgen05.ld.red.sync.aligned") < llir.index("@llvm.nvvm.tcgen05.wait.ld()")
    ptx_red_ops = [op for op, _ in ptx_red_pairs]
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


LD_RED_MODIFIER_CASES = [
    (False, tl.PropagateNan.NONE),
    (True, tl.PropagateNan.ALL),
]
LD_RED_REPRESENTATIVE_MODIFIER_CASES = [(False, tl.PropagateNan.NONE)]
LD_RED_REPRESENTATIVE_RED_OPS = ("min",)

LD_RED_TILE_PERMUTED_CASES = [
    (32, 8, 4, "32x32b.x32"),
    (64, 16, 4, "32x32b.x64"),
    (128, 8, 4, "32x32b.x128"),
    (256, 16, 4, "32x32b.x64"),
    (256, 32, 4, "32x32b.x64"),
    (256, 64, 4, "32x32b.x64"),
]

LD_RED_ROWCOL_PERMUTED_CASES = [
    (row_perm_kind, col_perm_kind, "32x32b.x128")
    for row_perm_kind, col_perm_kind in PERMUTED_ROW_COL_LAYOUT_KINDS
    if not (row_perm_kind == "identity" and col_perm_kind == "identity")
]

LD_RED_COL_PERMUTED_SHAPES = [
    (32, "32x32b.x32"),
    (64, "32x32b.x64"),
    (128, "32x32b.x128"),
    (256, "32x32b.x64"),
]

LD_RED_COL_PERMUTED_CASES = [
    (col_perm_kind, n, expected_shape)
    for col_perm_kind in PERMUTED_LAYOUT_KINDS
    if col_perm_kind != "identity"
    for n, expected_shape in LD_RED_COL_PERMUTED_SHAPES
]

LD_RED_ROW_PERMUTED_CASES = [
    (row_perm_kind, col_perm_kind, expected_shape)
    for row_perm_kind, col_perm_kind, expected_shape in LD_RED_ROWCOL_PERMUTED_CASES
    if row_perm_kind != "identity"
    and (
        col_perm_kind == "identity"
        or (row_perm_kind, col_perm_kind) in (("rotate1", "reverse"), ("even_odd", "even_odd"))
    )
]

LD_RED_PURE_ROW_PERMUTED_N_SWEEP_CASES = [
    (row_perm_kind, n, expected_shape)
    for row_perm_kind in PERMUTED_LAYOUT_KINDS
    if row_perm_kind != "identity"
    for n, expected_shape in ((32, "32x32b.x32"), (64, "32x32b.x64"), (256, "32x32b.x64"))
]

LD_RED_EXPANDED_ROW_PERMUTED_CASES = [
    ("reverse", 32, "32x32b.x32"),
    ("rotate1", 64, "32x32b.x64"),
    ("even_odd", 128, "32x32b.x128"),
]

LD_RED_EXPANDED_ROWCOL_PERMUTED_CASES = [
    ("identity", "reverse", 64, "32x32b.x64"),
    ("even_odd", "even_odd", 128, "32x32b.x128"),
]

LD_RED_ROWCOL_PERMUTED_N_SWEEP_CASES = [
    ("reverse", "reverse", 32, "32x32b.x32"),
    ("reverse", "reverse", 256, "32x32b.x64"),
    ("rotate1", "even_odd", 32, "32x32b.x32"),
    ("rotate1", "even_odd", 64, "32x32b.x64"),
    ("rotate1", "even_odd", 256, "32x32b.x64"),
    ("even_odd", "rotate1", 64, "32x32b.x64"),
    ("even_odd", "rotate1", 256, "32x32b.x64"),
    ("reverse", "even_odd", 64, "32x32b.x64"),
    ("reverse", "even_odd", 256, "32x32b.x64"),
]

LD_RED_EXPLICIT_COMPATIBLE_NON_IDENTITY_LAYOUT_CASES = [
    pytest.param("tile_permuted", lambda: _make_tmem_linear_layout_tile_permuted(128, 128, 32), id="tile_permuted"),
    pytest.param("col_reverse", lambda: _make_tmem_linear_layout_permuted(128, 128, "identity", "reverse"),
                 id="col_reverse"),
    pytest.param("row_reverse", lambda: _make_tmem_linear_layout_permuted(128, 128, "reverse", "identity"),
                 id="row_reverse"),
    pytest.param("rowcol_rotate_reverse", lambda: _make_tmem_linear_layout_permuted(128, 128, "rotate1", "reverse"),
                 id="rowcol_rotate_reverse"),
]

LD_RED_DESCRIPTOR_CHAIN_LAYOUT_CASES = [
    ("identity", lambda: _make_tmem_linear_layout(128, 128)),
    ("tile_permuted", lambda: _make_tmem_linear_layout_tile_permuted(128, 128, 32)),
    ("col_reverse", lambda: _make_tmem_linear_layout_permuted(128, 128, "identity", "reverse")),
    ("row_reverse", lambda: _make_tmem_linear_layout_permuted(128, 128, "reverse", "identity")),
    ("rowcol_rotate_reverse", lambda: _make_tmem_linear_layout_permuted(128, 128, "rotate1", "reverse")),
]

LD_RED_DESCRIPTOR_CHAIN_VARIANTS = ("auto", "32x32b")

LD_RED_DESCRIPTOR_CHAIN_CASES = [
    pytest.param(layout_name, layout_factory, load_variant, id=f"{layout_name}_{load_variant}")
    for layout_name, layout_factory in LD_RED_DESCRIPTOR_CHAIN_LAYOUT_CASES
    for load_variant in LD_RED_DESCRIPTOR_CHAIN_VARIANTS
]

LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_CASE_SPECS = (
    ("identity", "identity", "identity", 32, "32x32b.x32"),
    ("identity", "identity", "identity", 64, "32x32b.x64"),
    ("identity", "identity", "identity", 256, "32x32b.x64"),
    ("tile_permuted", "tile", "tile", 32, "32x32b.x32"),
    ("tile_permuted", "tile", "tile", 256, "32x32b.x64"),
    ("col_reverse", "identity", "reverse", 64, "32x32b.x64"),
    ("col_reverse", "identity", "reverse", 256, "32x32b.x64"),
    ("row_reverse", "reverse", "identity", 64, "32x32b.x64"),
    ("row_reverse", "reverse", "identity", 256, "32x32b.x64"),
    ("row_even_odd", "even_odd", "identity", 64, "32x32b.x64"),
    ("rowcol_rotate_reverse", "rotate1", "reverse", 64, "32x32b.x64"),
    ("rowcol_rotate_reverse", "rotate1", "reverse", 256, "32x32b.x64"),
)

LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_CASES = [
    pytest.param("identity", lambda n=n: _make_tmem_linear_layout(128, n), n, expected_shape, id=f"identity_n{n}")
    if layout_name == "identity" else pytest.param(
        "tile_permuted",
        lambda n=n: _make_tmem_linear_layout_tile_permuted(128, n, 8 if n == 32 else 64),
        n,
        expected_shape,
        id=f"tile_permuted_n{n}",
    )
    if layout_name == "tile_permuted" else pytest.param(
        layout_name,
        lambda n=n, row_perm_kind=row_perm_kind, col_perm_kind=col_perm_kind: _make_tmem_linear_layout_permuted(
            128, n, row_perm_kind, col_perm_kind
        ),
        n,
        expected_shape,
        id=f"{layout_name}_n{n}",
    )
    for layout_name, row_perm_kind, col_perm_kind, n, expected_shape in LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_CASE_SPECS
]

LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_EXPLICIT_VARIANT_SPECS = (
    ("identity", 32, "32x32b.x32", "32x32b"),
    ("identity", 64, "32x32b.x64", "32x32b"),
    ("identity", 256, "32x32b.x64", "32x32b_splitn"),
    ("tile_permuted", 32, "32x32b.x32", "32x32b"),
    ("col_reverse", 64, "32x32b.x64", "32x32b"),
    ("row_reverse", 64, "32x32b.x64", "32x32b"),
    ("row_even_odd", 64, "32x32b.x64", "32x32b"),
    ("rowcol_rotate_reverse", 64, "32x32b.x64", "32x32b"),
)

LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_PERMUTED_SPLIT_OFFSETS = {
    "tile_permuted": (0, 128, 64, 192),
}

LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_EXPLICIT_VARIANT_CASES = [
    pytest.param(
        layout_name,
        n,
        expected_shape,
        load_variant,
        LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_PERMUTED_SPLIT_OFFSETS[layout_name]
        if (
            layout_name in LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_PERMUTED_SPLIT_OFFSETS
            and n == 256
            and load_variant != "32x32b"
        )
        else None,
        id=f"{layout_name}_n{n}_{load_variant}",
    )
    for layout_name, n, expected_shape, load_variant in LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_EXPLICIT_VARIANT_SPECS
]

LD_RED_EXPLICIT_N_SWEEP_VARIANT_CASES = LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_EXPLICIT_VARIANT_CASES

LD_RED_M64_SPLITN_CASES = [
    pytest.param(n, f"16x32bx2.x{n // 2}", (0,), id=f"m64_64x{n}")
    for n in (32, 64, 128, 256)
]

LD_RED_M64_EXPLICIT_VARIANT_CASES = [
    pytest.param(n, "32x32b", f"16x32bx2.x{n // 2}", (0,), id=f"m64_64x{n}_32x32b")
    for n in (32, 256)
] + [
    pytest.param(n, load_variant, f"16x32bx2.x{n // 4}", (0, n // 2),
                 id=f"m64_64x{n}_{load_variant}")
    for n, load_variant in (
        (32, "auto"),
        (64, "16x32bx2"),
        (128, "32x32b_splitn"),
        (256, "auto"),
        (256, "32x32b_splitn"),
    )
]

LD_RED_M64_ROWCOL_PERMUTED_DEFAULT_CASES = [
    pytest.param("reverse", "identity", 32, "16x32bx2.x8", (0, 16), id="row_reverse_n32"),
    pytest.param("rotate1", "even_odd", 128, "16x32bx2.x32", (0, 64), id="row_rotate_col_even_odd_n128"),
    pytest.param("identity", "reverse", 32, "16x32bx2.x8", (0, 16), id="col_reverse_n32"),
]


def _make_ld_red_descriptor_chain_n_sweep_explicit_layout(layout_name, n):
    if layout_name == "identity":
        return _make_tmem_linear_layout(128, n)
    if layout_name == "tile_permuted":
        tile_n = {32: 8, 64: 16, 256: 64}[n]
        return _make_tmem_linear_layout_tile_permuted(128, n, tile_n)
    if layout_name == "col_reverse":
        return _make_tmem_linear_layout_permuted(128, n, "identity", "reverse")
    if layout_name == "col_rotate1":
        return _make_tmem_linear_layout_permuted(128, n, "identity", "rotate1")
    if layout_name == "col_even_odd":
        return _make_tmem_linear_layout_permuted(128, n, "identity", "even_odd")
    if layout_name == "row_reverse":
        return _make_tmem_linear_layout_permuted(128, n, "reverse", "identity")
    if layout_name == "row_rotate1":
        return _make_tmem_linear_layout_permuted(128, n, "rotate1", "identity")
    if layout_name == "row_even_odd":
        return _make_tmem_linear_layout_permuted(128, n, "even_odd", "identity")
    if layout_name == "rowcol_rotate_reverse":
        return _make_tmem_linear_layout_permuted(128, n, "rotate1", "reverse")
    raise AssertionError(f"unexpected ld.red descriptor-chain layout {layout_name}")


LD_RED_MIXED_CASES = [
    (128, 64, 4),
    (128, 128, 4),
    (128, 256, 4),
]

LD_RED_SINGLE_CTA_BLOCK_LAYOUT_CASES = [
    (128, n, 4) for n in (64, 128, 256)
]

LD_RED_RESOURCE_BOUNDARY_CASES = [
    ("identity_256x256", 256, 256, 8),
]



def _assert_ld_red_runtime_outputs(inp, out, red, red_op, use_abs, propagate_nan):
    use_nan = propagate_nan == tl.PropagateNan.ALL
    torch.testing.assert_close(inp, out, atol=0, rtol=0, equal_nan=use_nan)
    ref_input = torch.abs(inp) if use_abs else inp
    expected_red = getattr(torch, red_op)(ref_input, dim=1).values
    torch.testing.assert_close(expected_red, red, atol=1e-5, rtol=1e-5, equal_nan=use_nan)


def _seed_ld_red_nan_rows(inp, propagate_nan):
    if propagate_nan == tl.PropagateNan.ALL and inp.is_floating_point():
        inp[10, 5] = float("nan")
        inp[50, 15] = float("nan")


def _make_ld_red_non_f32_input(shape, dtype):
    if dtype in (torch.int8, torch.int16, torch.int32):
        return torch.randint(-50, 50, shape, dtype=dtype, device="cuda")
    return torch.randn(shape, dtype=dtype, device="cuda")


def _assert_ld_red_uses_software_reduce(compiled):
    ptx_red_pairs = [
        pair
        for pair in _extract_tcgen05_opcode_offsets(compiled.asm["ptx"], opcodes=("ld", ))
        if ".ld.red." in pair[0]
    ]
    llir_red_pairs = [
        pair
        for pair in _extract_tcgen05_opcode_offsets(compiled.asm["llir"], opcodes=("ld", ))
        if ".ld.red." in pair[0]
    ]
    assert ptx_red_pairs == llir_red_pairs == []
    ptx_ld_ops = [
        op
        for op, _ in _extract_tcgen05_opcode_offsets(compiled.asm["ptx"], opcodes=("ld", ))
        if op.startswith("tcgen05.ld.sync.aligned.")
    ]
    assert ptx_ld_ops

LD_RED_NON_F32_SOFTWARE_CASES = [
    pytest.param(
        "i32_plain",
        torch.int32,
        _make_tmem_linear_layout(128, 128),
        "auto",
        False,
        tl.PropagateNan.NONE,
        id="i32_plain",
    ),
    pytest.param(
        "i32_abs",
        torch.int32,
        _make_tmem_linear_layout(128, 128),
        "auto",
        True,
        tl.PropagateNan.NONE,
        id="i32_abs",
    ),
    pytest.param(
        "i32_nan",
        torch.int32,
        _make_tmem_linear_layout(128, 128),
        "auto",
        False,
        tl.PropagateNan.ALL,
        id="i32_nan",
    ),
    pytest.param(
        "bf16_plain",
        torch.bfloat16,
        _make_tmem_linear_layout(128, 128),
        "auto",
        False,
        tl.PropagateNan.NONE,
        id="bf16_plain",
    ),
    pytest.param(
        "bf16_nan",
        torch.bfloat16,
        _make_tmem_linear_layout(128, 128),
        "auto",
        False,
        tl.PropagateNan.ALL,
        id="bf16_nan",
    ),
    pytest.param(
        "bf16_abs",
        torch.bfloat16,
        _make_tmem_linear_layout(128, 128),
        "auto",
        True,
        tl.PropagateNan.NONE,
        id="bf16_abs",
    ),
    pytest.param(
        "f16_plain",
        torch.float16,
        _make_tmem_linear_layout(128, 128),
        "auto",
        False,
        tl.PropagateNan.NONE,
        id="f16_plain",
    ),
    pytest.param(
        "f16_nan",
        torch.float16,
        _make_tmem_linear_layout(128, 128),
        "auto",
        False,
        tl.PropagateNan.ALL,
        id="f16_nan",
    ),
    pytest.param(
        "f16_abs",
        torch.float16,
        _make_tmem_linear_layout(128, 128),
        "auto",
        True,
        tl.PropagateNan.NONE,
        id="f16_abs",
    ),
    pytest.param(
        "i16_plain",
        torch.int16,
        _make_tmem_linear_layout(128, 128),
        "auto",
        False,
        tl.PropagateNan.NONE,
        id="i16_plain",
    ),
    pytest.param(
        "i8_plain",
        torch.int8,
        _make_tmem_linear_layout(128, 128),
        "auto",
        False,
        tl.PropagateNan.NONE,
        id="i8_plain",
    ),
    pytest.param(
        "f16_legacy_unpacked",
        torch.float16,
        TensorMemoryLayout(block=(128, 2), col_stride=2),
        "auto",
        False,
        tl.PropagateNan.NONE,
        id="f16_legacy_unpacked",
    ),
]

LD_RED_NON_F32_DESCRIPTOR_CHAIN_SOFTWARE_CASES = [
    pytest.param(
        "i32_plain_descriptor",
        torch.int32,
        "auto",
        False,
        tl.PropagateNan.NONE,
        id="i32_plain_descriptor",
    ),
    pytest.param(
        "i32_nan_descriptor",
        torch.int32,
        "auto",
        False,
        tl.PropagateNan.ALL,
        id="i32_nan_descriptor",
    ),
    pytest.param(
        "bf16_plain_descriptor",
        torch.bfloat16,
        "auto",
        False,
        tl.PropagateNan.NONE,
        id="bf16_plain_descriptor",
    ),
    pytest.param(
        "bf16_nan_descriptor",
        torch.bfloat16,
        "auto",
        False,
        tl.PropagateNan.ALL,
        id="bf16_nan_descriptor",
    ),
    pytest.param(
        "bf16_abs_descriptor",
        torch.bfloat16,
        "auto",
        True,
        tl.PropagateNan.NONE,
        id="bf16_abs_descriptor",
    ),
    pytest.param(
        "f16_plain_descriptor",
        torch.float16,
        "auto",
        False,
        tl.PropagateNan.NONE,
        id="f16_plain_descriptor",
    ),
    pytest.param(
        "f16_nan_descriptor",
        torch.float16,
        "auto",
        False,
        tl.PropagateNan.ALL,
        id="f16_nan_descriptor",
    ),
    pytest.param(
        "f16_abs_descriptor",
        torch.float16,
        "auto",
        True,
        tl.PropagateNan.NONE,
        id="f16_abs_descriptor",
    ),
    pytest.param(
        "i16_plain_descriptor",
        torch.int16,
        "auto",
        False,
        tl.PropagateNan.NONE,
        id="i16_plain_descriptor",
    ),
    pytest.param(
        "i8_plain_descriptor",
        torch.int8,
        "auto",
        False,
        tl.PropagateNan.NONE,
        id="i8_plain_descriptor",
    ),
]

LDST_EXPECTED_OFFSETS_128x256 = {
    "auto": [
        ("tcgen05.st.sync.aligned.32x32b.x64.b32", 0),
        ("tcgen05.st.sync.aligned.32x32b.x64.b32", 64),
        ("tcgen05.st.sync.aligned.32x32b.x64.b32", 128),
        ("tcgen05.st.sync.aligned.32x32b.x64.b32", 192),
        ("tcgen05.ld.sync.aligned.32x32b.x64.b32", 0),
        ("tcgen05.ld.sync.aligned.32x32b.x64.b32", 64),
        ("tcgen05.ld.sync.aligned.32x32b.x64.b32", 128),
        ("tcgen05.ld.sync.aligned.32x32b.x64.b32", 192),
    ],
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
        ("tcgen05.st.sync.aligned.16x64b.x64.b32", 0),
        ("tcgen05.st.sync.aligned.16x64b.x64.b32", 128),
        ("tcgen05.ld.sync.aligned.16x64b.x64.b32", 0),
        ("tcgen05.ld.sync.aligned.16x64b.x64.b32", 128),
        ("tcgen05.ld.sync.aligned.16x64b.x64.b32", 0),
        ("tcgen05.ld.sync.aligned.16x64b.x64.b32", 128),
    ],
    "16x128b": [
        ("tcgen05.st.sync.aligned.16x128b.x32.b32", 0),
        ("tcgen05.st.sync.aligned.16x128b.x32.b32", 128),
        ("tcgen05.st.sync.aligned.16x128b.x32.b32", 0),
        ("tcgen05.st.sync.aligned.16x128b.x32.b32", 128),
        ("tcgen05.ld.sync.aligned.16x128b.x32.b32", 0),
        ("tcgen05.ld.sync.aligned.16x128b.x32.b32", 128),
        ("tcgen05.ld.sync.aligned.16x128b.x32.b32", 0),
        ("tcgen05.ld.sync.aligned.16x128b.x32.b32", 128),
    ],
    "16x256b": [
        ("tcgen05.st.sync.aligned.16x256b.x16.b32", 0),
        ("tcgen05.st.sync.aligned.16x256b.x16.b32", 128),
        ("tcgen05.st.sync.aligned.16x256b.x16.b32", 0),
        ("tcgen05.st.sync.aligned.16x256b.x16.b32", 128),
        ("tcgen05.ld.sync.aligned.16x256b.x16.b32", 0),
        ("tcgen05.ld.sync.aligned.16x256b.x16.b32", 128),
        ("tcgen05.ld.sync.aligned.16x256b.x16.b32", 0),
        ("tcgen05.ld.sync.aligned.16x256b.x16.b32", 128),
    ],
}

CP_SCALES_LAYOUT_PROBE_CASES = [
    ("warpx4", _make_scales_shared_layout_warpx4(), "PASS"),
    ("warpx2_candidate", _make_scales_shared_layout_warpx2_candidate(), "PASS"),
    (
        "warpx2_no_scales_like_column_tail",
        _make_scales_shared_layout_warpx2_no_scales_like_column_tail(),
        "PASS",
    ),
    (
        "warpx2_row32_after_columns",
        _make_scales_shared_layout_warpx2_row32_after_columns(),
        "PASS",
    ),
    (
        "warpx2_row32_after_low_rows",
        _make_scales_shared_layout_warpx2_row32_after_low_rows(),
        "PASS",
    ),
    (
        "warpx2_column_first_row_tail",
        _make_scales_shared_layout_warpx2_column_first_row_tail(),
        "PASS",
    ),
]

CP_LINEAR_EXOTIC_UNSUPPORTED_CASES = [
    ("mixed", _make_tmem_linear_layout_mixed(128, 128)),
    ("scrambled_cols", _make_tmem_linear_layout_permuted(128, 128, "identity", "even_odd")),
    ("scrambled_rows_cols", _make_tmem_linear_layout_permuted(128, 128, "even_odd", "even_odd")),
]

CP_LINEAR_TILE_PERMUTED_CASES = (
    (64, 4, 16, "tcgen05.cp.cta_group::1.128x128b"),
    (64, 8, 8, "tcgen05.cp.cta_group::1.128x256b"),
    (64, 16, 8, "tcgen05.cp.cta_group::1.128x256b"),
    (128, 4, 32, "tcgen05.cp.cta_group::1.128x128b"),
    (128, 8, 16, "tcgen05.cp.cta_group::1.128x256b"),
    (128, 16, 16, "tcgen05.cp.cta_group::1.128x256b"),
    (128, 32, 16, "tcgen05.cp.cta_group::1.128x256b"),
    (256, 4, 64, "tcgen05.cp.cta_group::1.128x128b"),
    (256, 8, 32, "tcgen05.cp.cta_group::1.128x256b"),
    (256, 16, 32, "tcgen05.cp.cta_group::1.128x256b"),
    (256, 32, 32, "tcgen05.cp.cta_group::1.128x256b"),
    (256, 64, 32, "tcgen05.cp.cta_group::1.128x256b"),
)

CP_LINEAR_TILE_PERMUTED_UNSUPPORTED_CASES = (
    (64, 1),
    (64, 2),
    (128, 1),
    (128, 2),
    (256, 1),
    (256, 2),
)

CP_LINEAR_TILE_SELECTOR_PERMUTED_CASES = (
    (64, 4, "reverse", 16, "tcgen05.cp.cta_group::1.128x128b"),
    (128, 4, "even_odd", 16, "tcgen05.cp.cta_group::1.128x256b"),
    (128, 8, "reverse", 16, "tcgen05.cp.cta_group::1.128x256b"),
    (256, 8, "reverse", 32, "tcgen05.cp.cta_group::1.128x256b"),
)

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
@pytest.mark.parametrize("dtype_name,torch_dtype,mode,variant,expected_shape", LDST_IDENTITY_N32_CASES)
def test_tmem_runtime_matrix_ldst_identity_n32_linear_layout(dtype_name, torch_dtype, mode, variant, expected_shape):
    m = 128
    n = 32
    layout = _make_tmem_linear_layout(m, n)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    if mode == "direct":
        if variant == "auto":
            compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4)
        else:
            compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
        torch.testing.assert_close(out, inp, atol=0, rtol=0)
    else:
        compiled = tmem_ldst_descriptor_chain_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
        torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)
        assert "tensor_memory_linear" in compiled.asm["ttgir"]

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype,mode,perm_kind,variant,expected_shape", LDST_PERMUTED_N32_CASES)
def test_tmem_runtime_matrix_ldst_permuted_n32_linear_layout(
    dtype_name, torch_dtype, mode, perm_kind, variant, expected_shape
):
    m = 128
    n = 32
    layout = _make_tmem_linear_layout_permuted(m, n, perm_kind, perm_kind)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    if mode == "direct":
        if variant == "auto":
            compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4)
        else:
            compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
        torch.testing.assert_close(out, inp, atol=0, rtol=0)
    else:
        compiled = tmem_ldst_descriptor_chain_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
        torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)
        assert "tensor_memory_linear" in compiled.asm["ttgir"]

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype,mode,row_perm_kind,col_perm_kind,variant,expected_shape", LDST_ROWCOL_N32_CASES)
def test_tmem_runtime_matrix_ldst_rowcol_n32_linear_layout(
    dtype_name, torch_dtype, mode, row_perm_kind, col_perm_kind, variant, expected_shape
):
    m = 128
    n = 32
    layout = _make_tmem_linear_layout_permuted(m, n, row_perm_kind, col_perm_kind)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    if mode == "direct":
        if variant == "auto":
            compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4)
        else:
            compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
        torch.testing.assert_close(out, inp, atol=0, rtol=0)
    else:
        compiled = tmem_ldst_descriptor_chain_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
        torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)
        assert "tensor_memory_linear" in compiled.asm["ttgir"]

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype,mode,layout_name,variant,expected_shape", LDST_EXOTIC_N32_CASES)
def test_tmem_runtime_matrix_ldst_exotic_n32_linear_layout(
    dtype_name, torch_dtype, mode, layout_name, variant, expected_shape
):
    m = 128
    n = 32
    layout = LDST_EXOTIC_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    if mode == "direct":
        if variant == "auto":
            compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4)
        else:
            compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
        torch.testing.assert_close(out, inp, atol=0, rtol=0)
    else:
        compiled = tmem_ldst_descriptor_chain_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
        torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)
        assert "tensor_memory_linear" in compiled.asm["ttgir"]

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype,mode,layout_name,variant,expected_shape", LDST_TWOCTA_N32_CASES)
def test_tmem_runtime_matrix_ldst_twocta_n32_linear_layout(
    dtype_name, torch_dtype, mode, layout_name, variant, expected_shape
):
    m = 256
    n = 32
    layout = LDST_TWOCTA_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    if mode == "direct":
        if variant == "auto":
            compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4, num_ctas=2)
        else:
            compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4, num_ctas=2)
        torch.testing.assert_close(out, inp, atol=0, rtol=0)
    else:
        compiled = tmem_ldst_descriptor_chain_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
        )
        torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)
        assert "tensor_memory_linear" in compiled.asm["ttgir"]

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes
    assert "twoCTAs = true" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "layout_name,cta_group,num_ctas,m,n,alloc_size",
    ALLOC_LIFETIME_LDST_CASES,
)
def test_tmem_runtime_matrix_alloc_lifetime_ldst(layout_name, cta_group, num_ctas, m, n, alloc_size):
    if cta_group == 1:
        layout = LDST_LAYOUTS[layout_name](n)
    else:
        layout = LDST_TWOCTA_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_variant_kernel[(1, )](
        inp, out, layout, m, n, "32x32b", num_warps=4, num_ctas=num_ctas
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_ldst_ptx_llir_match(compiled)
    _assert_exact_tmem_lifetime_ptx_llir_match(
        compiled,
        cta_group=cta_group,
        alloc_size=alloc_size,
        expect_cluster_sync=cta_group == 2,
    )


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_alloc_source_initialization_lifetime():
    m = 128
    n = 128
    layout = _make_tmem_linear_layout(m, n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_alloc_source_init_kernel[(1, )](inp, out, layout, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    assert "ttng.tmem_alloc %" in compiled.asm["ttgir"]
    _assert_ldst_ptx_llir_match(compiled)
    _assert_exact_tmem_lifetime_ptx_llir_match(compiled, cta_group=1, alloc_size=128)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype,perm_kind,n,variant,expected_shape", LDST_PERMUTED_CASES)
def test_tmem_runtime_matrix_ldst_permuted_layout_sweep(
    dtype_name, torch_dtype, perm_kind, n, variant, expected_shape
):
    m = 128
    layout = _make_tmem_linear_layout_permuted(m, n, perm_kind, perm_kind)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
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
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,row_perm_kind,col_perm_kind,n,variant,expected_shape", LDST_ROWCOL_PERMUTED_CASES
)
def test_tmem_runtime_matrix_ldst_rowcol_permuted_layout_sweep(
    dtype_name, torch_dtype, row_perm_kind, col_perm_kind, n, variant, expected_shape
):
    m = 128
    layout = _make_tmem_linear_layout_permuted(m, n, row_perm_kind, col_perm_kind)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
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
@pytest.mark.parametrize("dtype_name,torch_dtype,layout_name,n,variant,expected_shape", LDST_EXOTIC_CASES)
def test_tmem_runtime_matrix_ldst_exotic_linear_layouts(
    dtype_name, torch_dtype, layout_name, n, variant, expected_shape
):
    m = 128
    layout = LDST_EXOTIC_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
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
@pytest.mark.parametrize(
    "row_perm_kind,col_perm_kind,n",
    [
        ("identity", "reverse", 64),
        ("even_odd", "even_odd", 128),
    ],
)
def test_tmem_runtime_matrix_ldst_expanded_rowcol_permuted_linear_layout(row_perm_kind, col_perm_kind, n):
    m = 256
    layout = _make_tmem_linear_layout_permuted(m, n, row_perm_kind, col_perm_kind)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch.float32)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, "auto", num_warps=8)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_ldst_ptx_llir_match(compiled)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


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
@pytest.mark.parametrize("mode,layout_name,n,variant,expected_shape", LDST_I32_BROAD_CASES)
def test_tmem_runtime_matrix_ldst_i32_broad_linear_layouts(mode, layout_name, n, variant, expected_shape):
    m = 128
    layout = LDST_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    if mode == "direct":
        if variant == "auto":
            compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4)
        else:
            compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
        torch.testing.assert_close(out, inp, atol=0, rtol=0)
    else:
        compiled = tmem_ldst_descriptor_chain_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
        torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)
        assert "tensor_memory_linear" in compiled.asm["ttgir"]

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype,perm_kind,n,variant,expected_shape", LDST_PERMUTED_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_compositions_permuted_layout_sweep(
    dtype_name, torch_dtype, perm_kind, n, variant, expected_shape
):
    m = 128
    layout = _make_tmem_linear_layout_permuted(m, n, perm_kind, perm_kind)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_chain_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,row_perm_kind,col_perm_kind,n,variant,expected_shape", LDST_ROWCOL_PERMUTED_CASES
)
def test_tmem_runtime_matrix_ldst_descriptor_compositions_rowcol_permuted_layout_sweep(
    dtype_name, torch_dtype, row_perm_kind, col_perm_kind, n, variant, expected_shape
):
    m = 128
    layout = _make_tmem_linear_layout_permuted(m, n, row_perm_kind, col_perm_kind)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_chain_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype,layout_name,n,variant,expected_shape", LDST_EXOTIC_DESCRIPTOR_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_compositions_exotic_layouts(
    dtype_name, torch_dtype, layout_name, n, variant, expected_shape
):
    m = 128
    layout = LDST_EXOTIC_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_chain_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)

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
@pytest.mark.parametrize("mode,layout_name,n,variant,expected_shape", LDST_TWOCTA_I32_BROAD_CASES)
def test_tmem_runtime_matrix_ldst_twocta_i32_broad_linear_layouts(mode, layout_name, n, variant, expected_shape):
    m = 256
    layout = LDST_TWOCTA_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    if mode == "direct":
        if variant == "auto":
            compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4, num_ctas=2)
        else:
            compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4, num_ctas=2)
        torch.testing.assert_close(out, inp, atol=0, rtol=0)
    else:
        compiled = tmem_ldst_descriptor_chain_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
        )
        torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)
        assert "tensor_memory_linear" in compiled.asm["ttgir"]

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes
    assert "twoCTAs = true" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.skip(reason=LDST_LIFTED_ROUNDTRIP_OOR_SKIP_REASON)
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
    for required_op in required_ops:
        assert required_op in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.skip(reason=LDST_LIFTED_ROUNDTRIP_OOR_SKIP_REASON)
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
    for required_op in required_ops:
        assert required_op in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.skip(reason=LDST_LIFTED_ROUNDTRIP_OOR_SKIP_REASON)
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
@pytest.mark.parametrize("dtype_name,torch_dtype,layout_name,n,variant", LDST_HIGHER_RANK_SLICE_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_multidim_slices(dtype_name, torch_dtype, layout_name, n, variant):
    m = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_multidim_slice_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4
    )
    torch.testing.assert_close(out, inp + 9, atol=0, rtol=0)

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
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,layout_name,n,variant,expected_shape,expected_half_shape",
    LDST_TWOCTA_HIGHER_RANK_INDEX_CASES,
)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_index(
    dtype_name, torch_dtype, layout_name, n, variant, expected_shape, expected_half_shape
):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_higher_rank_index_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
    )
    torch.testing.assert_close(out, inp + 5, atol=0, rtol=0)

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
@pytest.mark.parametrize("dtype_name,torch_dtype,layout_name,n,variant", LDST_TWOCTA_HIGHER_RANK_SLICE_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_multidim_slices(dtype_name, torch_dtype, layout_name, n, variant):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_multidim_slice_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
    )
    torch.testing.assert_close(out, inp + 9, atol=0, rtol=0)

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
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,layout_name,n,variant,expected_shape,expected_half_shape",
    LDST_HIGHER_RANK_DIM0_SLICE_POSITIVE_CASES,
)
def test_tmem_runtime_matrix_ldst_descriptor_higher_rank_dim0_slice_positive_lifted_layout(
    dtype_name, torch_dtype, layout_name, n, variant, expected_shape, expected_half_shape
):
    m = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_higher_rank_dim0_slice_positive_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4
    )
    ref = inp.clone()
    ref[:, : n // 2] += 7
    torch.testing.assert_close(out, ref, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert f"tcgen05.st.sync.aligned.{expected_shape}" in observed_opcodes
    assert f"tcgen05.ld.sync.aligned.{expected_shape}" in observed_opcodes
    assert f"tcgen05.st.sync.aligned.{expected_half_shape}" in observed_opcodes
    assert f"tcgen05.ld.sync.aligned.{expected_half_shape}" in observed_opcodes

@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,layout_fn", MULTIDIM_SLICE_REPLAY_LAYOUTS.items())
def test_tmem_runtime_matrix_ldst_descriptor_multidim_slice_replays(layout_name, layout_fn):
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
    observed_opcodes = [op for op, _ in ops]
    assert "tcgen05.st.sync.aligned.16x128b.x32.b32" in observed_opcodes
    assert "tcgen05.ld.sync.aligned.16x128b.x32.b32" in observed_opcodes
    assert "tcgen05.ld.sync.aligned.32x32b.x128.b32" in observed_opcodes
    assert "tcgen05.st.sync.aligned.32x32b.x128.b32" in observed_opcodes

    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_subslice" not in ttgir
    assert "tt.split" in ttgir
    assert "tt.join" in ttgir


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
        ("tcgen05.st.sync.aligned.16x128b.x32.b32", 0),
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
        ("tcgen05.ld.sync.aligned.16x128b.x32.b32", 0),
    ]

    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttg.memdesc_reshape" in ttgir


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
    ttgir = compiled.asm["ttgir"]
    assert "tt.split" in ttgir
    assert "tt.join" in ttgir
    assert "ttg.memdesc_subslice" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant", LDST_HIGHER_RANK_HALF_ROWS_OOR_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_reports_tmem_oor(layout_name, n, variant):
    m = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(triton.runtime.errors.OutOfResources) as excinfo:
        tmem_ldst_descriptor_higher_rank_half_rows_positive_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4
        )

    text = str(excinfo.value)
    _assert_clean_tmem_oor(text, required=1024, hardware_limit=512)
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_DIRECT_HALF_ROWS_POSITIVE_CASES)
def test_tmem_runtime_matrix_ldst_descriptor_direct_half_rows_positive(
    layout_name, n, variant, expected_shape
):
    m = 128
    layout = LDST_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_direct_half_rows_positive_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4
    )
    ref = inp.clone()
    ref[m // 2 :, :] += 17.0
    torch.testing.assert_close(out, ref, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert f"tcgen05.st.sync.aligned.{expected_shape}" in observed_opcodes
    assert f"tcgen05.ld.sync.aligned.{expected_shape}" in observed_opcodes
    ttgir = compiled.asm["ttgir"]
    assert "tt.split" in ttgir
    assert "tt.join" in ttgir
    assert "ttg.memdesc_subslice" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,layout_name,n,variant,expected_shape,expected_half_shape",
    LDST_TWOCTA_HIGHER_RANK_DIM0_SLICE_POSITIVE_CASES,
)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_dim0_slice_positive_lifted_layout(
    dtype_name, torch_dtype, layout_name, n, variant, expected_shape, expected_half_shape
):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_higher_rank_dim0_slice_positive_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
    )
    ref = inp.clone()
    ref[:, : n // 2] += 7
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
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_TWOCTA_HIGHER_RANK_HALF_ROWS_POSITIVE_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_half_rows_positive_lifted_layout(
    layout_name, n, variant, expected_shape
):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_higher_rank_half_rows_positive_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
    )
    ref = inp.clone()
    ref[m // 2 :, :] += 13.0
    torch.testing.assert_close(out, ref, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert f"tcgen05.st.sync.aligned.{expected_shape}" in observed_opcodes
    assert f"tcgen05.ld.sync.aligned.{expected_shape}" in observed_opcodes
    ttgir = compiled.asm["ttgir"]
    assert "twoCTAs = true" in ttgir
    assert "tt.split" in ttgir
    assert "tt.join" in ttgir
    assert "ttg.memdesc_subslice" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant", LDST_TWOCTA_HIGHER_RANK_HALF_ROWS_OOR_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_half_rows_reports_tmem_oor(
    layout_name, n, variant
):
    m = 256
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(triton.runtime.errors.OutOfResources) as excinfo:
        tmem_ldst_descriptor_higher_rank_half_rows_positive_kernel[(1, )](
            inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
        )

    text = str(excinfo.value)
    _assert_clean_tmem_oor(text, required=1024, hardware_limit=512)
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,n,variant,expected_shape", LDST_TWOCTA_DIRECT_HALF_ROWS_POSITIVE_CASES)
def test_tmem_runtime_matrix_ldst_twocta_descriptor_direct_half_rows_positive(
    layout_name, n, variant, expected_shape
):
    m = 256
    layout = LDST_TWOCTA_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_direct_half_rows_positive_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
    )
    ref = inp.clone()
    ref[m // 2 :, :] += 17.0
    torch.testing.assert_close(out, ref, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert f"tcgen05.st.sync.aligned.{expected_shape}" in observed_opcodes
    assert f"tcgen05.ld.sync.aligned.{expected_shape}" in observed_opcodes
    ttgir = compiled.asm["ttgir"]
    assert "twoCTAs = true" in ttgir
    assert "tt.split" in ttgir
    assert "tt.join" in ttgir
    assert "ttg.memdesc_subslice" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("variant,expected_shape", LDST_DIRECT_HIGHER_RANK_GET_REG_LAYOUT_POSITIVE_CASES)
def test_tmem_runtime_matrix_ldst_direct_higher_rank_get_reg_layout_positive(variant, expected_shape):
    m = 128
    n = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS["identity"](n), [2])
    inp = torch.arange(2 * m * n, dtype=torch.float32, device="cuda").reshape(2, m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_direct_higher_rank_get_reg_layout_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert f"tcgen05.st.sync.aligned.{expected_shape}" in observed_opcodes
    assert f"tcgen05.ld.sync.aligned.{expected_shape}" in observed_opcodes
    assert "ttg.memdesc_reshape" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_ldst_direct_higher_rank_load_store_replay_positive():
    m = 128
    n = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS["identity"](n), [2])
    inp = torch.arange(2 * m * n, dtype=torch.float32, device="cuda").reshape(2, m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_direct_higher_rank_replay_kernel[(1, )](
        inp, out, layout, m, n, num_warps=4
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert any(op.startswith("tcgen05.st.sync.aligned.32x32b") for op in observed_opcodes)
    assert any(op.startswith("tcgen05.ld.sync.aligned.32x32b") for op in observed_opcodes)
    assert "ttg.memdesc_reshape" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name", ("block_two_ctas", "mmav5_twocta"))
def test_tmem_runtime_matrix_ldst_twocta_direct_higher_rank_load_store_replay_positive(layout_name):
    m = 256
    n = 64
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS[layout_name](n), [2])
    inp = torch.arange(2 * m * n, dtype=torch.float32, device="cuda").reshape(2, m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_direct_higher_rank_replay_kernel[(1, )](
        inp, out, layout, m, n, num_warps=4, num_ctas=2
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert "tcgen05.st.sync.aligned.32x32b.x128.b32" in observed_opcodes
    assert "tcgen05.ld.sync.aligned.32x32b.x128.b32" in observed_opcodes
    ttgir = compiled.asm["ttgir"]
    assert "twoCTAs = true" in ttgir
    assert "ttg.memdesc_reshape" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_ldst_direct_higher_rank_load_red_replay_positive():
    m = 128
    n = 128
    layout = _lift_tmem_layout(LDST_LAYOUTS["identity"](n), [2])
    inp = torch.arange(2 * m * n, dtype=torch.float32, device="cuda").reshape(2, m, n)
    out = torch.empty_like(inp)
    red = torch.empty((2, m), dtype=torch.float32, device="cuda")

    red_layout = ttgl.BlockedLayout([1, 1], [1, 32], [1, 8], [1, 0])
    compiled = tmem_ldst_direct_higher_rank_load_red_kernel[(1, )](
        inp, out, red, layout, red_layout, m, n, num_warps=8
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    torch.testing.assert_close(red, torch.min(inp, dim=2).values, atol=0, rtol=0)

    ptx_red_ops = [
        op
        for op, _ in _extract_tcgen05_opcode_offsets(compiled.asm["ptx"], opcodes=("ld", ))
        if ".ld.red." in op
    ]
    llir_red_ops = [
        op
        for op, _ in _extract_tcgen05_opcode_offsets(compiled.asm["llir"], opcodes=("ld", ))
        if ".ld.red." in op
    ]
    assert ptx_red_ops == llir_red_ops
    assert ptx_red_ops
    assert all(op.startswith("tcgen05.ld.red.sync.aligned.32x32b") for op in ptx_red_ops)
    assert all(".min" in op for op in ptx_red_ops)
    assert "ttg.memdesc_reshape" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_ldst_twocta_direct_higher_rank_load_red_replay_positive():
    m = 256
    n = 64
    layout = _lift_tmem_layout(LDST_TWOCTA_LAYOUTS["block_two_ctas"](n), [2])
    red_layout = ttgl.BlockedLayout([1, 1], [1, 32], [1, 8], [1, 0], cga_layout=[[1, 0]])
    inp = torch.arange(2 * m * n, dtype=torch.float32, device="cuda").reshape(2, m, n)
    out = torch.empty_like(inp)
    red = torch.empty((2, m), dtype=torch.float32, device="cuda")

    compiled = tmem_ldst_direct_higher_rank_load_red_kernel[(1, )](
        inp, out, red, layout, red_layout, m, n, num_warps=8, num_ctas=2
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    torch.testing.assert_close(red, torch.min(inp, dim=2).values, atol=0, rtol=0)

    red_ops = [
        op
        for op, _ in _extract_tcgen05_opcode_offsets(compiled.asm["ptx"], opcodes=("ld", ))
        if ".ld.red." in op
    ]
    assert red_ops == ["tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32"]
    assert "ttg.memdesc_reshape" in compiled.asm["ttgir"]


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
@pytest.mark.parametrize("dtype_name,torch_dtype,n,splitn_x,offset_imm_pairs", M64_SPLITN_CASES)
def test_tmem_runtime_matrix_splitn_immediates(dtype_name, torch_dtype, n, splitn_x, offset_imm_pairs):
    m = 64
    layout = _make_tmem_linear_layout_m64(n)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
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
@pytest.mark.parametrize("dtype_name,torch_dtype,n,splitn_x,offset_imm_pairs", M64_SPLITN_CASES)
def test_tmem_runtime_matrix_splitn_auto_selects_16x32bx2(dtype_name, torch_dtype, n, splitn_x, offset_imm_pairs):
    m = 64
    layout = _make_tmem_linear_layout_m64(n)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
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
@pytest.mark.parametrize("dtype_name,torch_dtype,n", M64_16BIT_AUTO_SPLITN_CASES)
def test_tmem_runtime_matrix_splitn_16bit_m64_auto_matches_explicit(dtype_name, torch_dtype, n):
    m = 64
    layout = _make_tmem_linear_layout_m64(n)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out_auto = torch.empty_like(inp)
    out_splitn = torch.empty_like(inp)
    out_explicit = torch.empty_like(inp)

    compiled_auto = tmem_ldst_auto_kernel[(1, )](inp, out_auto, layout, m, n, num_warps=4)
    compiled_splitn = tmem_ldst_variant_kernel[(1, )](
        inp, out_splitn, layout, m, n, "32x32b_splitn", num_warps=4
    )
    compiled_explicit = tmem_ldst_variant_kernel[(1, )](
        inp, out_explicit, layout, m, n, "16x32bx2", num_warps=4
    )

    torch.testing.assert_close(out_auto, inp, atol=0, rtol=0)
    torch.testing.assert_close(out_splitn, inp, atol=0, rtol=0)
    torch.testing.assert_close(out_explicit, inp, atol=0, rtol=0)
    torch.testing.assert_close(out_auto, out_splitn, atol=0, rtol=0)
    torch.testing.assert_close(out_auto, out_explicit, atol=0, rtol=0)

    auto_ops, auto_imms = _assert_ldst_ptx_llir_match(compiled_auto)
    splitn_ops, splitn_imms = _assert_ldst_ptx_llir_match(compiled_splitn)
    explicit_ops, explicit_imms = _assert_ldst_ptx_llir_match(compiled_explicit)
    assert auto_ops == splitn_ops
    assert auto_ops == explicit_ops
    assert auto_imms == splitn_imms
    assert auto_imms == explicit_imms
    assert all("16x32bx2" in op for op, _ in auto_ops)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", M64_SPLITN_DTYPES)
@pytest.mark.parametrize("n", [2, 64, 256])
def test_tmem_runtime_matrix_explicit_16x32bx2_matches_splitn(n, dtype_name, torch_dtype):
    m = 64
    layout = _make_tmem_linear_layout_m64(n)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
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
@pytest.mark.parametrize("dtype_name,torch_dtype,row_perm_kind,col_perm_kind,n,variant", M64_ROWCOL_PERMUTED_CASES)
def test_tmem_runtime_matrix_splitn_rowcol_permuted_layout_sweep(
    dtype_name, torch_dtype, row_perm_kind, col_perm_kind, n, variant
):
    m = 64
    layout = _make_tmem_linear_layout_m64_permuted(n, row_perm_kind, col_perm_kind)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert observed_opcodes
    assert all("16x32bx2" in op for op in observed_opcodes)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype,row_perm_kind,col_perm_kind,n", M64_ROWCOL_PERMUTED_AUTO_CASES)
def test_tmem_runtime_matrix_splitn_rowcol_permuted_auto_selects_16x32bx2(
    dtype_name, torch_dtype, row_perm_kind, col_perm_kind, n
):
    m = 64
    layout = _make_tmem_linear_layout_m64_permuted(n, row_perm_kind, col_perm_kind)
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    observed_opcodes = [op for op, _ in ops]
    assert observed_opcodes
    assert all("16x32bx2" in op for op in observed_opcodes)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("variant", ("auto", "16x128b", "16x256b"))
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
@pytest.mark.parametrize("dtype_name,torch_dtype,layout_name,n,variant,expected_shape", SUBWORD_LDST_CASES)
def test_tmem_runtime_matrix_ldst_subword_pack_unpack(
    dtype_name, torch_dtype, layout_name, n, variant, expected_shape
):
    m = 128
    layout = LDST_LAYOUTS[layout_name](n)
    inp = torch.arange(m * n, dtype=torch_dtype, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_subword_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert all(op in (expected_st, expected_ld) for op in observed_opcodes)
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype,layout_name,n,variant,expected_shape", SUBWORD_LDST_DESCRIPTOR_CASES)
def test_tmem_runtime_matrix_ldst_subword_descriptor_chain_roundtrip(
    dtype_name, torch_dtype, layout_name, n, variant, expected_shape
):
    m = 128
    layout = LDST_LAYOUTS[layout_name](n)
    base = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n) % 64
    inp = base.to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_chain_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
    torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    assert expected_st in observed_opcodes
    assert expected_ld in observed_opcodes

    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "ttg.memdesc_reshape" in ttgir
    assert "ttg.memdesc_trans" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,layout_kind,n,layout_factory,expected_st,expected_ld", X1_SUBWORD_LDST_CASES
)
@pytest.mark.parametrize("variant", X1_SUBWORD_LDST_VARIANTS)
def test_tmem_runtime_matrix_ldst_x1_subword_roundtrip(
    dtype_name,
    torch_dtype,
    layout_kind,
    n,
    layout_factory,
    expected_st,
    expected_ld,
    variant,
):
    m = 128
    layout = layout_factory()
    inp = torch.arange(m * n, dtype=torch_dtype, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_subword_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4)
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
@pytest.mark.parametrize("dtype_name,torch_dtype,n", X1_SUBWORD_LDST_TWOCTA_CASES)
@pytest.mark.parametrize("variant", X1_SUBWORD_LDST_VARIANTS)
def test_tmem_runtime_matrix_ldst_x1_subword_twocta_roundtrip(dtype_name, torch_dtype, n, variant):
    m = 256
    layout = _make_tmem_linear_layout_mmav5_twocta(m, n)
    inp = torch.arange(m * n, dtype=torch_dtype, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_subword_variant_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    assert ops == [
        ("tcgen05.st.sync.aligned.32x32b.x1.b32", 0),
        ("tcgen05.ld.sync.aligned.32x32b.x1.b32", 0),
    ]
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "twoCTAs = true" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype,n", X1_SUBWORD_LDST_TWOCTA_CASES)
@pytest.mark.parametrize("variant", X1_SUBWORD_LDST_VARIANTS)
def test_tmem_runtime_matrix_ldst_x1_subword_twocta_descriptor_chain_roundtrip(
    dtype_name, torch_dtype, n, variant
):
    m = 256
    layout = _lift_tmem_layout(_make_tmem_linear_layout_mmav5_twocta(m, n), [2])
    inp = torch.arange(m * n, dtype=torch_dtype, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_x1_subword_twocta_descriptor_chain_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4, num_ctas=2
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    assert ops == [
        ("tcgen05.st.sync.aligned.32x32b.x1.b32", 0),
        ("tcgen05.ld.sync.aligned.32x32b.x1.b32", 0),
    ]
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "twoCTAs = true" in ttgir

@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_kind,m,num_ctas,layout_factory", X1_F32_LDST_CASES)
@pytest.mark.parametrize("variant", X1_F32_LDST_VARIANTS)
def test_tmem_runtime_matrix_ldst_x1_f32_roundtrip(layout_kind, m, num_ctas, layout_factory, variant):
    n = 1
    layout = layout_factory()
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    if variant == "auto":
        compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4, num_ctas=num_ctas)
    else:
        compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4, num_ctas=num_ctas)
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
@pytest.mark.parametrize("variant", X1_F32_LDST_VARIANTS)
def test_tmem_runtime_matrix_ldst_x1_f32_descriptor_chain_roundtrip(layout_kind, m, num_ctas, layout_factory, variant):
    n = 1
    layout = layout_factory()
    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_chain_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4, num_ctas=num_ctas
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
@pytest.mark.parametrize("variant", X1_F32_LDST_VARIANTS)
def test_tmem_runtime_matrix_ldst_x1_i32_roundtrip(layout_kind, m, num_ctas, layout_factory, variant):
    n = 1
    layout = layout_factory()
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    if variant == "auto":
        compiled = tmem_ldst_auto_kernel[(1, )](inp, out, layout, m, n, num_warps=4, num_ctas=num_ctas)
    else:
        compiled = tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4, num_ctas=num_ctas)
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
@pytest.mark.parametrize("variant", X1_F32_LDST_VARIANTS)
def test_tmem_runtime_matrix_ldst_x1_i32_descriptor_chain_roundtrip(
    layout_kind, m, num_ctas, layout_factory, variant
):
    n = 1
    layout = layout_factory()
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    compiled = tmem_ldst_descriptor_chain_kernel[(1, )](
        inp, out, layout, m, n, variant, num_warps=4, num_ctas=num_ctas
    )
    torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)

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
def test_tmem_runtime_matrix_ldst_x1_i32_unsupported_variants_report_clean_unsupported(
    layout_kind, m, num_ctas, layout_factory, variant
):
    n = 1
    layout = layout_factory()
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)

    with pytest.raises(CompilationError) as excinfo:
        tmem_ldst_variant_kernel[(1, )](inp, out, layout, m, n, variant, num_warps=4, num_ctas=num_ctas)

    msg = str(excinfo.value)
    _assert_clean_unsupported_descriptor_view(msg, variant)
    assert "requested tcgen05.ld/st atom 16x128b has a 4-dword column footprint" in msg
    assert "descriptor view exposes only 1 materializable dword column" in msg
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


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
    assert "requested tcgen05.ld/st atom 16x128b has a 4-dword column footprint" in msg
    assert "descriptor view exposes only 1 materializable dword column" in msg
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
@pytest.mark.parametrize("M,N,num_warps,cga_layout,instr_variant,expected_ops", SCALES_LDST_DESCRIPTOR_VIEW_CASES)
def test_tmem_runtime_matrix_ldst_scales_descriptor_view_roundtrip(
    M, N, num_warps, cga_layout, instr_variant, expected_ops
):
    inp = torch.arange(M * N, dtype=torch.int8, device="cuda").reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_scales_ldst_descriptor_view_kernel[(1, )](
        inp, out, M, N, instr_variant, cga_layout, num_warps=num_warps
    )
    torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    assert ops == expected_ops
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_scales_encoding" in ttgir
    assert "ttng.tmem_load" in ttgir
    assert "ttng.tmem_store" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "M,N,num_warps,num_ctas,cga_layout,instr_variant,expected_ops",
    SCALES_LDST_DESCRIPTOR_VIEW_CGA_CASES,
)
def test_tmem_runtime_matrix_ldst_scales_descriptor_view_cga_roundtrip(
    M, N, num_warps, num_ctas, cga_layout, instr_variant, expected_ops
):
    inp = torch.arange(M * N, dtype=torch.int8, device="cuda").reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_scales_ldst_descriptor_view_kernel[(1, )](
        inp, out, M, N, instr_variant, cga_layout, num_warps=num_warps, num_ctas=num_ctas
    )
    torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    assert ops == expected_ops
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_scales_encoding" in ttgir
    assert "ttng.tmem_load" in ttgir
    assert "ttng.tmem_store" in ttgir


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
    if instr_variant in SCALES_LDST_N_SHARDED_VARIANT_WIDTHS:
        required_elements = _scales_ldst_n_sharded_min_elements(instr_variant)
        exposed_elements = M * N
        assert f"{required_elements}-element tensor-memory-scales footprint" in text
        assert f"exposes only {exposed_elements} scale elements" in text
        assert "n-sharded scales footprint requirement" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.skip(reason=LDST_LIFTED_ROUNDTRIP_OOR_SKIP_REASON)
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
@pytest.mark.skip(reason=LDST_LIFTED_ROUNDTRIP_OOR_SKIP_REASON)
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


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,case_name,layout_group,layout_name,m,num_ctas,n,variant,expected_shape",
    LDST_DESCRIPTOR_RANK5_SMALL_CASES,
)
def test_tmem_runtime_matrix_ldst_descriptor_rank5_small_roundtrip(
    dtype_name, torch_dtype, case_name, layout_group, layout_name, m, num_ctas, n, variant, expected_shape
):
    base_layout = LDST_LAYOUTS[layout_name](n) if layout_group == "single" else LDST_TWOCTA_LAYOUTS[layout_name](n)
    layout = _lift_tmem_layout(base_layout, [1, 1, 2])
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)
    delta = 23 if dtype_name == "i32" else 23.0

    compiled = tmem_ldst_descriptor_rank5_small_roundtrip_kernel[(1, )](
        inp, out, layout, m, n, variant, delta, num_warps=4, num_ctas=num_ctas
    )
    torch.testing.assert_close(out, inp + delta, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    observed_opcodes = [op for op, _ in ops]
    expected_message_count = 2 if expected_shape.startswith("32x32b") else 4
    assert observed_opcodes.count(expected_st) == expected_message_count
    assert observed_opcodes.count(expected_ld) == expected_message_count

    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir
    # Full-view replay may canonicalize a leading unit subslice plus index into
    # the equivalent parent index before direct tcgen05 ld/st. Runtime
    # correctness and exact opcode counts above are the semantic checks.
    if num_ctas == 2:
        assert "twoCTAs = true" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,case_name,layout_group,layout_name,m,num_ctas,variant,expected_shape",
    LDST_DESCRIPTOR_RANK5_N256_CASES,
)
def test_tmem_runtime_matrix_ldst_descriptor_rank5_unit_parent_n256_roundtrip(
    dtype_name, torch_dtype, case_name, layout_group, layout_name, m, num_ctas, variant, expected_shape
):
    n = 256
    base_layout = LDST_LAYOUTS[layout_name](n) if layout_group == "single" else LDST_TWOCTA_LAYOUTS[layout_name](n)
    layout = _lift_tmem_layout(base_layout, [1, 1, 1])
    inp = torch.arange(m * n, dtype=torch.int32, device="cuda").reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)
    delta = 29 if dtype_name == "i32" else 29.0

    compiled = tmem_ldst_descriptor_rank5_unit_parent_roundtrip_kernel[(1, )](
        inp, out, layout, m, n, variant, delta, num_warps=4, num_ctas=num_ctas
    )
    torch.testing.assert_close(out, inp + delta, atol=0, rtol=0)

    ops, _ = _assert_ldst_ptx_llir_match(compiled)
    expected_st = f"tcgen05.st.sync.aligned.{expected_shape}"
    expected_ld = f"tcgen05.ld.sync.aligned.{expected_shape}"
    support_st = "tcgen05.st.sync.aligned.16x64b.x64.b32"
    support_ld = "tcgen05.ld.sync.aligned.16x64b.x64.b32"
    observed_opcodes = [op for op, _ in ops]
    assert observed_opcodes.count(expected_st) + observed_opcodes.count(support_st) == 8
    assert observed_opcodes.count(expected_ld) + observed_opcodes.count(support_ld) == 8

    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir
    # The unit-parent rank-5 prefix and shape view chain are semantic no-ops
    # after indexing, so the frontend is allowed to fold subslice/reshape/trans
    # descriptors before TTGIR emission.
    if num_ctas == 2:
        assert "twoCTAs = true" in ttgir


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_MODIFIER_CASES)
@pytest.mark.parametrize("layout_name,M,N,num_warps,expected_shape", LD_RED_LINEAR_CASES)
def test_tmem_runtime_matrix_ld_red_identity_linear_layout(red_op, use_abs, propagate_nan, layout_name, M, N, num_warps,
                                                           expected_shape):
    if layout_name == "identity":
        layout = _make_tmem_linear_layout(M, N)
    elif layout_name == "legacy_equivalent_256":
        layout = _make_tmem_linear_layout_ld_red_legacy_equivalent_256(N)
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

    _assert_ld_red_opcode_pairs(compiled, N, expected_shape, red_op, use_abs, propagate_nan)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_MODIFIER_CASES)
@pytest.mark.parametrize("N,expected_shape,expected_offsets", LD_RED_M64_SPLITN_CASES)
def test_tmem_runtime_matrix_ld_red_m64_splitn_linear_layout(
    red_op, use_abs, propagate_nan, N, expected_shape, expected_offsets
):
    M = 64
    layout = _make_tmem_linear_layout_m64(N)
    compiled = _run_tmem_reduction_case(
        layout,
        M,
        N,
        red_op,
        use_abs,
        propagate_nan,
        num_warps=4,
        expected_red_opcode_prefix="tcgen05.ld.red.sync.aligned.16x32bx2.x",
    )
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    _assert_ld_red_opcode_pairs(
        compiled,
        N,
        expected_shape,
        red_op,
        use_abs,
        propagate_nan,
        expected_offsets=expected_offsets,
    )


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize("row_perm_kind,col_perm_kind,N,expected_shape,expected_offsets",
                         LD_RED_M64_ROWCOL_PERMUTED_DEFAULT_CASES)
def test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout(
    red_op, row_perm_kind, col_perm_kind, N, expected_shape, expected_offsets
):
    M = 64
    layout = _make_tmem_linear_layout_m64_permuted(N, row_perm_kind, col_perm_kind)
    compiled = _run_tmem_reduction_case(
        layout,
        M,
        N,
        red_op,
        False,
        tl.PropagateNan.NONE,
        num_warps=4,
        expected_red_opcode_prefix="tcgen05.ld.red.sync.aligned.16x32bx2.x",
    )
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    _assert_ld_red_opcode_pairs(
        compiled,
        N,
        expected_shape,
        red_op,
        False,
        tl.PropagateNan.NONE,
        expected_offsets=expected_offsets,
    )


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize("N,load_variant,expected_shape,expected_offsets", LD_RED_M64_EXPLICIT_VARIANT_CASES)
def test_tmem_runtime_matrix_ld_red_m64_explicit_splitn_variants(
    red_op, N, load_variant, expected_shape, expected_offsets
):
    M = 64
    layout = _make_tmem_linear_layout_m64(N)
    inp = torch.randn(M, N, dtype=torch.float32, device="cuda")
    out = torch.empty_like(inp)
    red = torch.empty(M, dtype=torch.float32, device="cuda")

    compiled = tmem_ld_red_m64_explicit_layout_kernel[(1, )](
        inp, out, red, layout, N, load_variant, red_op, False, tl.PropagateNan.NONE, num_warps=4
    )

    _assert_ld_red_runtime_outputs(inp, out, red, red_op, False, tl.PropagateNan.NONE)
    _assert_ld_red_opcode_pairs(
        compiled,
        N,
        expected_shape,
        red_op,
        False,
        tl.PropagateNan.NONE,
        expected_offsets=expected_offsets,
    )


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize(
    "row_perm_kind,col_perm_kind,N,expected_shape,expected_offsets",
    LD_RED_M64_ROWCOL_PERMUTED_DEFAULT_CASES,
)
def test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn(
    row_perm_kind, col_perm_kind, N, expected_shape, expected_offsets
):
    M = 64
    layout = _make_tmem_linear_layout_m64_permuted(N, row_perm_kind, col_perm_kind)
    inp = torch.randn(M, N, dtype=torch.float32, device="cuda")
    out = torch.empty_like(inp)
    red = torch.empty(M, dtype=torch.float32, device="cuda")

    compiled = tmem_ld_red_m64_explicit_layout_kernel[(1, )](
        inp, out, red, layout, N, "32x32b", "min", False, tl.PropagateNan.NONE, num_warps=4
    )

    _assert_ld_red_runtime_outputs(inp, out, red, "min", False, tl.PropagateNan.NONE)
    _assert_ld_red_opcode_pairs(
        compiled,
        N,
        expected_shape,
        "min",
        False,
        tl.PropagateNan.NONE,
        expected_offsets=expected_offsets,
    )


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_MODIFIER_CASES)
@pytest.mark.parametrize("load_variant", ["auto", "32x32b"])
def test_tmem_runtime_matrix_ld_red_explicit_compatible_layout_variants(
    load_variant, use_abs, propagate_nan, red_op
):
    M = N = 128
    layout = _make_tmem_linear_layout(M, N)
    inp = torch.randn(M, N, dtype=torch.float32, device="cuda")
    _seed_ld_red_nan_rows(inp, propagate_nan)
    out = torch.empty_like(inp)
    red = torch.empty(M, dtype=torch.float32, device="cuda")

    compiled = tmem_ld_red_explicit_layout_kernel[(1, )](
        inp, out, red, layout, N, load_variant, red_op, use_abs, propagate_nan, num_warps=4
    )

    _assert_ld_red_runtime_outputs(inp, out, red, red_op, use_abs, propagate_nan)
    _assert_ld_red_opcode_pairs(compiled, N, "32x32b.x128", red_op, use_abs, propagate_nan)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", LD_RED_REPRESENTATIVE_RED_OPS)
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_REPRESENTATIVE_MODIFIER_CASES)
@pytest.mark.parametrize("layout_name,layout_factory,load_variant", LD_RED_DESCRIPTOR_CHAIN_CASES)
def test_tmem_runtime_matrix_ld_red_descriptor_chain(
    layout_name, layout_factory, load_variant, use_abs, propagate_nan, red_op
):
    M = N = 128
    layout = layout_factory()
    inp = torch.randn(M, N, dtype=torch.float32, device="cuda")
    _seed_ld_red_nan_rows(inp, propagate_nan)
    out = torch.empty_like(inp)
    red = torch.empty(M, dtype=torch.float32, device="cuda")

    compiled = tmem_ld_red_descriptor_chain_kernel[(1, )](
        inp,
        out,
        red,
        layout,
        N,
        load_variant,
        red_op,
        use_abs,
        propagate_nan,
        num_warps=4,
    )

    _assert_ld_red_runtime_outputs(inp, out, red, red_op, use_abs, propagate_nan)
    _assert_ld_red_opcode_pairs(compiled, N, "32x32b.x128", red_op, use_abs, propagate_nan)
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", LD_RED_REPRESENTATIVE_RED_OPS)
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_REPRESENTATIVE_MODIFIER_CASES)
@pytest.mark.parametrize("layout_name,layout_factory,N,expected_shape", LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_CASES)
def test_tmem_runtime_matrix_ld_red_descriptor_chain_n_sweep(
    layout_name, layout_factory, N, expected_shape, use_abs, propagate_nan, red_op
):
    M = 128
    layout = layout_factory()
    inp = torch.randn(M, N, dtype=torch.float32, device="cuda")
    _seed_ld_red_nan_rows(inp, propagate_nan)
    out = torch.empty_like(inp)
    red = torch.empty(M, dtype=torch.float32, device="cuda")

    compiled = tmem_ld_red_descriptor_chain_kernel[(1, )](
        inp, out, red, layout, N, "auto", red_op, use_abs, propagate_nan, num_warps=4
    )

    _assert_ld_red_runtime_outputs(inp, out, red, red_op, use_abs, propagate_nan)
    expected_offsets = (
        LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_PERMUTED_SPLIT_OFFSETS[layout_name]
        if layout_name in LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_PERMUTED_SPLIT_OFFSETS
        and N == 256
        else None
    )
    _assert_ld_red_opcode_pairs(
        compiled,
        N,
        expected_shape,
        red_op,
        use_abs,
        propagate_nan,
        expected_offsets=expected_offsets,
    )
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", LD_RED_REPRESENTATIVE_RED_OPS)
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_REPRESENTATIVE_MODIFIER_CASES)
@pytest.mark.parametrize(
    "layout_name,N,expected_shape,load_variant,expected_offsets",
    LD_RED_DESCRIPTOR_CHAIN_N_SWEEP_EXPLICIT_VARIANT_CASES,
)
def test_tmem_runtime_matrix_ld_red_descriptor_chain_n_sweep_explicit_variants(
    layout_name,
    N,
    expected_shape,
    load_variant,
    expected_offsets,
    use_abs,
    propagate_nan,
    red_op,
):
    M = 128
    layout = _make_ld_red_descriptor_chain_n_sweep_explicit_layout(layout_name, N)
    inp = torch.randn(M, N, dtype=torch.float32, device="cuda")
    _seed_ld_red_nan_rows(inp, propagate_nan)
    out = torch.empty_like(inp)
    red = torch.empty(M, dtype=torch.float32, device="cuda")

    compiled = tmem_ld_red_descriptor_chain_kernel[(1, )](
        inp,
        out,
        red,
        layout,
        N,
        load_variant,
        red_op,
        use_abs,
        propagate_nan,
        num_warps=4,
    )

    _assert_ld_red_runtime_outputs(inp, out, red, red_op, use_abs, propagate_nan)
    _assert_ld_red_opcode_pairs(
        compiled,
        N,
        expected_shape,
        red_op,
        use_abs,
        propagate_nan,
        expected_offsets=expected_offsets,
    )
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttg.memdesc_reshape" in ttgir


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", LD_RED_REPRESENTATIVE_RED_OPS)
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_REPRESENTATIVE_MODIFIER_CASES)
@pytest.mark.parametrize(
    "layout_name,N,expected_shape,load_variant,expected_offsets",
    LD_RED_EXPLICIT_N_SWEEP_VARIANT_CASES,
)
def test_tmem_runtime_matrix_ld_red_explicit_n_sweep_variants(
    layout_name,
    N,
    expected_shape,
    load_variant,
    expected_offsets,
    use_abs,
    propagate_nan,
    red_op,
):
    M = 128
    layout = _make_ld_red_descriptor_chain_n_sweep_explicit_layout(layout_name, N)
    inp = torch.randn(M, N, dtype=torch.float32, device="cuda")
    _seed_ld_red_nan_rows(inp, propagate_nan)
    out = torch.empty_like(inp)
    red = torch.empty(M, dtype=torch.float32, device="cuda")

    compiled = tmem_ld_red_explicit_layout_kernel[(1, )](
        inp,
        out,
        red,
        layout,
        N,
        load_variant,
        red_op,
        use_abs,
        propagate_nan,
        num_warps=4,
    )

    _assert_ld_red_runtime_outputs(inp, out, red, red_op, use_abs, propagate_nan)
    _assert_ld_red_opcode_pairs(
        compiled,
        N,
        expected_shape,
        red_op,
        use_abs,
        propagate_nan,
        expected_offsets=expected_offsets,
    )
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", LD_RED_REPRESENTATIVE_RED_OPS)
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_REPRESENTATIVE_MODIFIER_CASES)
@pytest.mark.parametrize("layout_name,layout_factory", LD_RED_EXPLICIT_COMPATIBLE_NON_IDENTITY_LAYOUT_CASES)
@pytest.mark.parametrize("load_variant", ["auto", "32x32b"])
def test_tmem_runtime_matrix_ld_red_explicit_compatible_non_identity_layouts_canonicalize_32x32b(
    load_variant, layout_name, layout_factory, use_abs, propagate_nan, red_op
):
    M = N = 128
    layout = layout_factory()
    inp = torch.randn(M, N, dtype=torch.float32, device="cuda")
    _seed_ld_red_nan_rows(inp, propagate_nan)
    out = torch.empty_like(inp)
    red = torch.empty(M, dtype=torch.float32, device="cuda")

    compiled = tmem_ld_red_explicit_layout_kernel[(1, )](
        inp, out, red, layout, 128, load_variant, red_op, use_abs, propagate_nan, num_warps=4
    )

    _assert_ld_red_runtime_outputs(inp, out, red, red_op, use_abs, propagate_nan)
    _assert_ld_red_opcode_pairs(compiled, N, "32x32b.x128", red_op, use_abs, propagate_nan)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_MODIFIER_CASES)
@pytest.mark.parametrize("load_variant", ["16x64b", "16x128b", "16x256b"])
def test_tmem_runtime_matrix_ld_red_explicit_n_sharded_layout_uses_software_reduce(
    load_variant, use_abs, propagate_nan, red_op
):
    M = N = 128
    layout = _make_tmem_linear_layout(M, N)
    inp = torch.randn(M, N, dtype=torch.float32, device="cuda")
    _seed_ld_red_nan_rows(inp, propagate_nan)
    out = torch.empty_like(inp)
    red = torch.empty(M, dtype=torch.float32, device="cuda")

    compiled = tmem_ld_red_explicit_layout_kernel[(1, )](
        inp, out, red, layout, 128, load_variant, red_op, use_abs, propagate_nan, num_warps=4
    )

    _assert_ld_red_runtime_outputs(inp, out, red, red_op, use_abs, propagate_nan)
    ptx_red_pairs = [
        pair
        for pair in _extract_tcgen05_opcode_offsets(compiled.asm["ptx"], opcodes=("ld", ))
        if ".ld.red." in pair[0]
    ]
    llir_red_pairs = [
        pair
        for pair in _extract_tcgen05_opcode_offsets(compiled.asm["llir"], opcodes=("ld", ))
        if ".ld.red." in pair[0]
    ]
    assert ptx_red_pairs == llir_red_pairs == []
    ptx_ld_ops = [
        op
        for op, _ in _extract_tcgen05_opcode_offsets(compiled.asm["ptx"], opcodes=("ld", ))
        if op.startswith("tcgen05.ld.sync.aligned.")
    ]
    assert ptx_ld_ops


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize(
    "name,dtype,layout,load_variant,use_abs,propagate_nan", LD_RED_NON_F32_SOFTWARE_CASES
)
def test_tmem_runtime_matrix_ld_red_non_f32_contract_uses_software_reduce(
    name, dtype, layout, load_variant, use_abs, propagate_nan, red_op
):
    inp = _make_ld_red_non_f32_input((128, 128), dtype)
    _seed_ld_red_nan_rows(inp, propagate_nan)
    out = torch.empty_like(inp)
    red = torch.empty((128,), dtype=dtype, device="cuda")

    compiled = tmem_ld_red_non_f32_contract_kernel[(1, )](
        inp, out, red, layout, load_variant, red_op, use_abs, propagate_nan, num_warps=4
    )

    _assert_ld_red_runtime_outputs(inp, out, red, red_op, use_abs, propagate_nan)
    _assert_ld_red_uses_software_reduce(compiled)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize(
    "name,dtype,load_variant,use_abs,propagate_nan", LD_RED_NON_F32_DESCRIPTOR_CHAIN_SOFTWARE_CASES
)
def test_tmem_runtime_matrix_ld_red_non_f32_descriptor_chain_uses_software_reduce(
    name, dtype, load_variant, use_abs, propagate_nan, red_op
):
    layout = _make_tmem_linear_layout(128, 128)
    inp = _make_ld_red_non_f32_input((128, 128), dtype)
    _seed_ld_red_nan_rows(inp, propagate_nan)
    out = torch.empty_like(inp)
    red = torch.empty((128,), dtype=dtype, device="cuda")

    compiled = tmem_ld_red_descriptor_chain_kernel[(1, )](
        inp, out, red, layout, 128, load_variant, red_op, use_abs, propagate_nan, num_warps=4
    )

    _assert_ld_red_runtime_outputs(inp, out, red, red_op, use_abs, propagate_nan)
    _assert_ld_red_uses_software_reduce(compiled)
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize("use_abs", [False, True])
def test_tmem_runtime_matrix_ld_red_scales_uses_software_reduce(red_op, use_abs):
    M = 128
    N = 32
    layout = TensorMemoryScalesLayout()
    inp = torch.randint(-50, 50, (M, N), dtype=torch.int8, device="cuda")
    out = torch.empty_like(inp)
    red = torch.empty((M,), dtype=torch.int8, device="cuda")

    compiled = tmem_ld_red_explicit_layout_kernel[(1, )](
        inp, out, red, layout, N, "32x32b", red_op, use_abs, tl.PropagateNan.NONE, num_warps=4
    )

    _assert_ld_red_runtime_outputs(inp, out, red, red_op, use_abs, tl.PropagateNan.NONE)
    _assert_ld_red_uses_software_reduce(compiled)
    assert "tensor_memory_scales_encoding" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", LD_RED_REPRESENTATIVE_RED_OPS)
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_REPRESENTATIVE_MODIFIER_CASES)
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

    _assert_ld_red_opcode_pairs(compiled, N, expected_shape, red_op, use_abs, propagate_nan)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", LD_RED_REPRESENTATIVE_RED_OPS)
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_REPRESENTATIVE_MODIFIER_CASES)
@pytest.mark.parametrize("col_perm_kind,N,expected_shape", LD_RED_COL_PERMUTED_CASES)
def test_tmem_runtime_matrix_ld_red_col_permuted_linear_layout(
    red_op, use_abs, propagate_nan, col_perm_kind, N, expected_shape
):
    M = 128
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

    _assert_ld_red_opcode_pairs(compiled, N, expected_shape, red_op, use_abs, propagate_nan)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", LD_RED_REPRESENTATIVE_RED_OPS)
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_REPRESENTATIVE_MODIFIER_CASES)
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

    _assert_ld_red_opcode_pairs(compiled, N, expected_shape, red_op, use_abs, propagate_nan)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", LD_RED_REPRESENTATIVE_RED_OPS)
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_REPRESENTATIVE_MODIFIER_CASES)
@pytest.mark.parametrize("row_perm_kind,N,expected_shape", LD_RED_PURE_ROW_PERMUTED_N_SWEEP_CASES)
def test_tmem_runtime_matrix_ld_red_pure_row_permuted_n_sweep(
    red_op, use_abs, propagate_nan, row_perm_kind, N, expected_shape
):
    M = 128
    num_warps = 4
    layout = _make_tmem_linear_layout_permuted(M, N, row_perm_kind, "identity")
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

    _assert_ld_red_opcode_pairs(compiled, N, expected_shape, red_op, use_abs, propagate_nan)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", LD_RED_REPRESENTATIVE_RED_OPS)
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_REPRESENTATIVE_MODIFIER_CASES)
@pytest.mark.parametrize("row_perm_kind,N,expected_shape", LD_RED_EXPANDED_ROW_PERMUTED_CASES)
def test_tmem_runtime_matrix_ld_red_expanded_row_permuted_linear_layout(
    red_op, use_abs, propagate_nan, row_perm_kind, N, expected_shape
):
    M = 256
    num_warps = 8
    layout = _make_tmem_linear_layout_permuted(M, N, row_perm_kind, "identity")
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

    _assert_ld_red_opcode_pairs(compiled, N, expected_shape, red_op, use_abs, propagate_nan)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", LD_RED_REPRESENTATIVE_RED_OPS)
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_REPRESENTATIVE_MODIFIER_CASES)
@pytest.mark.parametrize("row_perm_kind,col_perm_kind,N,expected_shape", LD_RED_EXPANDED_ROWCOL_PERMUTED_CASES)
def test_tmem_runtime_matrix_ld_red_expanded_rowcol_permuted_linear_layout(
    red_op, use_abs, propagate_nan, row_perm_kind, col_perm_kind, N, expected_shape
):
    M = 256
    num_warps = 8
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

    _assert_ld_red_opcode_pairs(compiled, N, expected_shape, red_op, use_abs, propagate_nan)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", LD_RED_REPRESENTATIVE_RED_OPS)
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_REPRESENTATIVE_MODIFIER_CASES)
@pytest.mark.parametrize("row_perm_kind,col_perm_kind,N,expected_shape", LD_RED_ROWCOL_PERMUTED_N_SWEEP_CASES)
def test_tmem_runtime_matrix_ld_red_rowcol_permuted_n_sweep(
    red_op, use_abs, propagate_nan, row_perm_kind, col_perm_kind, N, expected_shape
):
    M = 128
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

    _assert_ld_red_opcode_pairs(compiled, N, expected_shape, red_op, use_abs, propagate_nan)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_MODIFIER_CASES)
@pytest.mark.parametrize("M,N,num_warps", LD_RED_MIXED_CASES)
def test_tmem_runtime_matrix_ld_red_mixed_linear_layout_uses_software_reduce(
    red_op, use_abs, propagate_nan, M, N, num_warps
):
    layout = _make_tmem_linear_layout_mixed(M, N)
    compiled = _run_tmem_reduction_case(
        layout,
        M,
        N,
        red_op,
        use_abs,
        propagate_nan,
        num_warps=num_warps,
        expect_hw_reduce=False,
    )
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_MODIFIER_CASES)
@pytest.mark.parametrize("M,N,num_warps", LD_RED_SINGLE_CTA_BLOCK_LAYOUT_CASES)
def test_tmem_runtime_matrix_ld_red_single_cta_block_linear_layout(
    red_op, use_abs, propagate_nan, M, N, num_warps
):
    layout = _make_tmem_linear_layout_block(M, N)
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


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
@pytest.mark.parametrize("red_op", ["min", "max"])
@pytest.mark.parametrize("use_abs,propagate_nan", LD_RED_MODIFIER_CASES)
@pytest.mark.parametrize("name,M,N,num_warps", LD_RED_RESOURCE_BOUNDARY_CASES)
def test_tmem_runtime_matrix_ld_red_identity_256_linear_layout_reports_resource_boundary(
    red_op, use_abs, propagate_nan, name, M, N, num_warps, capfd
):
    layout = _make_tmem_linear_layout(M, N)
    with pytest.raises(triton.runtime.errors.OutOfResources) as err:
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
    assert "out of resource: shared memory" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


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
        assert "maps to tcgen05.copy.warpx4.32x128b" in text
        assert "could not synthesize a compatible shared-memory descriptor plan for it" in text
        assert "Use the canonical shared layout for tcgen05.copy." in text
        assert "same descriptor family" in text
        assert "Assertion" not in text
        return

    assert expected_status == "BUG"
    assert "failed to find valid tcgen05.copy layout" in text
    assert "PassManager::run failed" in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_scales_noncanonical_layout_rematerializes():
    smem_h, smem_w = 64, 16
    inp = torch.randint(size=(smem_h, smem_w), low=-100, high=100, dtype=torch.int8, device="cuda")
    out = torch.empty_like(inp)
    smem_layout = _make_scales_shared_layout_warpx2_candidate()

    compiled = tmem_copy_scales_layout_probe_kernel[(1, )](inp, out, smem_layout)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.warpx4.32x128b"] * 2)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("start_row", (0, 64))
def test_tmem_runtime_matrix_cp_scales_shared_subslice_layout_rematerializes(start_row):
    out = torch.empty((1, ), dtype=torch.int32, device="cuda")
    parent_layout = _make_scales_shared_layout_parent_row_subslice_probe()

    compiled = tmem_copy_scales_shared_subslice_layout_probe_kernel[(1, )](
        out,
        parent_layout,
        start_row,
        num_warps=4,
    )

    torch.cuda.synchronize()
    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.warpx4.32x128b"] * 2)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_scales_tmem_descriptor_view_reports_clean_unsupported(capfd):
    smem_h, smem_w = 128, 32
    inp = torch.randint(size=(smem_h, smem_w), low=-100, high=100, dtype=torch.int8, device="cuda")
    out = torch.empty_like(inp)

    with pytest.raises(Exception) as excinfo:
        tmem_copy_scales_tmem_descriptor_view_kernel[(1, )](inp, out, num_warps=4)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "maps to tcgen05.copy.warpx4.32x128b" in text
    assert "could not synthesize a compatible shared-memory descriptor plan for it" in text
    assert "requires non-broadcast TMEM row bases to stay in ascending physical row order" in text
    assert "destination-row order requirement" in text
    assert "copy atom writes the full 32-row destination footprint" in text
    assert "first differing physical-query field is physical layout for active view shape 128x32" in text
    assert "needs a destination-row / source-message schedule" in text
    assert "Source element type should be 32-bit" not in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_128X128_DTYPES)
@pytest.mark.parametrize("M,N,BLOCK_N,swizzle", CP_NO_SCALES_CASES)
def test_tmem_runtime_matrix_cp_no_scales(dtype_name, torch_dtype, M, N, BLOCK_N, swizzle):
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N).to(torch_dtype)
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
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_128X128_DTYPES)
@pytest.mark.parametrize("M,N,BLOCK_N,swizzle", CP_NO_SCALES_SWIZZLE_CASES)
def test_tmem_runtime_matrix_cp_no_scales_swizzles(dtype_name, torch_dtype, M, N, BLOCK_N, swizzle):
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N).to(torch_dtype)
    out = torch.empty_like(inp)

    try:
        compiled = tmem_copy_no_scales_kernel[(1, )](inp, out, M, N, BLOCK_N, swizzle, num_warps=4)
    except triton.runtime.errors.OutOfResources:
        pytest.skip(f"shared memory OOR for M={M}, N={N}, BLOCK_N={BLOCK_N}, swizzle={swizzle}")
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    expected_count = (M * N) // 1024
    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.128x256b"] * expected_count)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_transposed_shared_reports_clean_error(capfd):
    M = N = 128
    BLOCK_N = 128
    swizzle = 32
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty_like(inp)

    with pytest.raises(Exception) as excinfo:
        tmem_copy_no_scales_transposed_shared_kernel[(1, )](
            inp, out, M, N, BLOCK_N, swizzle, num_warps=4
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "'ttng.tmem_copy' op" in text
    assert "cleanly unsupported" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


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
def test_tmem_runtime_matrix_cp_no_scales_shared_subslice_bad_offset_reports_clean_error(capfd):
    M = 128
    parent_n = 256
    slice_start = 64
    inp = torch.arange(M * parent_n, device="cuda", dtype=torch.float32).reshape(M, parent_n)
    out = torch.empty((M, parent_n // 2), device="cuda", dtype=torch.float32)

    with pytest.raises(Exception) as excinfo:
        tmem_copy_no_scales_shared_subslice_bad_offset_kernel[(1, )](
            inp, out, slice_start, num_warps=4
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "The split offset may not touch the tile" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_4x256b_reports_clean_unsupported(capfd):
    m = 4
    n = 8
    parent_m = 128
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty((parent_m, n), device="cuda", dtype=torch.float32)
    parent_layout = _make_tmem_linear_layout(parent_m, n)

    with pytest.raises(Exception) as excinfo:
        tmem_copy_no_scales_4x256b_view_kernel[(1, )](
            inp,
            out,
            parent_layout,
            num_warps=4,
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "maps to tcgen05.copy.4x256b" in text
    assert "cannot expose it as an ordinary contiguous four-row ttng.tmem_copy lowering" in text
    assert "refresh-shaped destination view" in text
    assert "logical row bits are stored in TMEM columns" in text
    assert "low logical column bits are stored in TMEM rows 32/64" in text
    assert "high logical column bit is stored at destination dword +4" in text
    assert "cleanly unsupported" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_4x256b_refresh_layout_codegen():
    m = 4
    n = 8
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty_like(inp)
    layout = _make_tmem_copy_4x256b_refresh_layout()

    compiled = tmem_copy_no_scales_4x256b_refresh_kernel[(1, )](
        inp,
        out,
        layout,
        num_warps=4,
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    expected_op = "tcgen05.cp.cta_group::1.4x256b"
    _assert_exact_cp_ptx_llir_match(compiled, [expected_op, expected_op])
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_4x256b_refresh_twocta_layout_codegen():
    m = 8
    n = 8
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty_like(inp)
    layout = _make_tmem_copy_4x256b_refresh_layout(two_ctas=True)
    cga_layout = ((1, 0), )

    compiled = tmem_copy_no_scales_4x256b_refresh_twocta_kernel[(1, )](
        inp,
        out,
        layout,
        cga_layout,
        num_warps=4,
        num_ctas=2,
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    expected_op = "tcgen05.cp.cta_group::2.4x256b"
    _assert_exact_cp_ptx_llir_match(compiled, [expected_op, expected_op])
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_ldst_4x256b_refresh_layout_reports_clean_unsupported(capfd):
    out = torch.empty((), device="cuda", dtype=torch.float32)
    layout = _make_tmem_copy_4x256b_refresh_layout()

    with pytest.raises(Exception) as excinfo:
        tmem_4x256b_refresh_get_reg_layout_unsupported_kernel[(1, )](
            out,
            layout,
            num_warps=4,
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "TMEM layout 'auto' unsupported for descriptor view" in text
    assert "tcgen05.copy.4x256b refresh-shaped tensor memory layout" in text
    assert "4x8 refresh view" in text
    assert "row anchors to be materializable as warp bases" in text
    assert "logical row bits in TMEM columns" in text
    assert "low logical column bits in TMEM rows 32/64" in text
    assert "high logical column bit at destination dword +4" in text
    assert "Source columns [0, 4) and [4, 8)" in text
    assert "directly supported 128-row physical layout" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_ldst_4x256b_refresh_raw_bitcast_reports_clean_unsupported(capfd):
    out = torch.empty((), device="cuda", dtype=torch.int8)
    layout = _make_tmem_copy_4x256b_refresh_layout()

    with pytest.raises(Exception) as excinfo:
        tmem_4x256b_refresh_raw_bitcast_get_reg_layout_unsupported_kernel[(1, )](
            out,
            layout,
            num_warps=4,
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "TMEM layout 'auto' unsupported for descriptor view" in text
    assert "raw physical bitcast of a tcgen05.copy.4x256b refresh image" in text
    assert "read whole row footprints" in text
    assert "do not provide a lane mask for this refresh image" in text
    assert "underlying 4x8 refresh image" in text
    assert "high logical column bit at destination dword +4" in text
    assert "directly supported 128-row physical layout" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


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
    assert "ttng.tmem_physical_layout" not in ttgir
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
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,M,N,swizzle,expected_count,expected_opcode",
    CP_TWOCTA_LINEAR_INDEXED_VIEW_CASES,
)
def test_tmem_runtime_matrix_cp_no_scales_twocta_linear_indexed_view(
    dtype_name, torch_dtype, M, N, swizzle, expected_count, expected_opcode
):
    cga_layout = _make_2cta_cga_layout((2, 1), (2, 1), (1, 0), 0)
    parent_layout = _lift_tmem_layout(_make_tmem_linear_layout_mmav5_twocta(M, N), [2])
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N).to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_copy_no_scales_twocta_linear_indexed_view_kernel[(1, )](
        inp,
        out,
        parent_layout,
        tuple(tuple(basis) for basis in cga_layout),
        M,
        N,
        swizzle,
        2,
        1,
        num_ctas=2,
        num_warps=4,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_exact_cp_ptx_llir_match(compiled, [expected_opcode] * expected_count)
    _assert_exact_commit_ptx_llir_match(
        compiled,
        ["tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"],
    )
    ptx = compiled.asm["ptx"]
    assert "tcgen05.cp.cta_group::1" not in ptx
    assert ptx.count("fence.proxy.async.shared::cluster") == 1
    assert ptx.count("barrier.cluster.arrive.aligned;") == 2
    assert ptx.count("barrier.cluster.wait.aligned;") == 2
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "tensor_memory_encoding" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_twocta_linear_indexed_view_full_256x256_reports_tmem_oor():
    M = 256
    N = 256
    swizzle = 128
    cga_layout = _make_2cta_cga_layout((2, 1), (2, 1), (1, 0), 0)
    parent_layout = _lift_tmem_layout(_make_tmem_linear_layout_mmav5_twocta(M, N), [2])
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty_like(inp)

    with pytest.raises(triton.runtime.errors.OutOfResources, match="tensor memory"):
        tmem_copy_no_scales_twocta_linear_indexed_view_kernel[(1, )](
            inp,
            out,
            parent_layout,
            tuple(tuple(basis) for basis in cga_layout),
            M,
            N,
            swizzle,
            2,
            1,
            num_ctas=2,
            num_warps=4,
        )


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
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,M,N,swizzle,expected_count,expected_opcode",
    CP_TWOCTA_LINEAR_SUBSLICE_VIEW_CASES,
)
def test_tmem_runtime_matrix_cp_no_scales_twocta_linear_subslice_view(
    dtype_name, torch_dtype, M, N, swizzle, expected_count, expected_opcode
):
    cga_layout = _make_2cta_cga_layout((2, 1), (2, 1), (1, 0), 0)
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N).to(torch_dtype)
    out = torch.empty_like(inp)

    parent_layout = _make_tmem_linear_layout_mmav5_twocta(M, 2 * N)
    compiled = tmem_copy_no_scales_twocta_linear_subslice_view_kernel[(1, )](
        inp,
        out,
        parent_layout,
        tuple(tuple(basis) for basis in cga_layout),
        M,
        N,
        swizzle,
        num_ctas=2,
        num_warps=4,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_exact_cp_ptx_llir_match(compiled, [expected_opcode] * expected_count)
    _assert_exact_commit_ptx_llir_match(
        compiled,
        ["tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"],
    )
    ptx = compiled.asm["ptx"]
    assert "tcgen05.cp.cta_group::1" not in ptx
    assert ptx.count("fence.proxy.async.shared::cluster") == 1
    assert ptx.count("barrier.cluster.arrive.aligned;") == 2
    assert ptx.count("barrier.cluster.wait.aligned;") == 2
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttng.tmem_subslice" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
def test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive(dtype_name, torch_dtype):
    M = 128
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout()
    tmem_layout = _make_tmem_copy_warpx2_tmem_layout()
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
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
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
def test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_candidate_positive(dtype_name, torch_dtype):
    M = 128
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout()
    tmem_layout = _make_tmem_copy_warpx2_tmem_layout_02_13()
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
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
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
@pytest.mark.parametrize("slice_start", [0, 4])
@pytest.mark.parametrize(
    "family,parent_layout_fn,expected_fn,expected_opcode",
    [
        (
            "01_23",
            _make_tmem_copy_warpx2_parent_tmem_layout,
            _expected_tmem_copy_warpx2_01_23_output,
            "tcgen05.cp.cta_group::1.warpx2::01_23.64x128b",
        ),
        (
            "02_13",
            _make_tmem_copy_warpx2_parent_tmem_layout_02_13,
            _expected_tmem_copy_warpx2_02_13_output,
            "tcgen05.cp.cta_group::1.warpx2::02_13.64x128b",
        ),
    ],
)
def test_tmem_runtime_matrix_cp_no_scales_warpx2_subslice_view_positive(
    dtype_name,
    torch_dtype,
    slice_start,
    family,
    parent_layout_fn,
    expected_fn,
    expected_opcode,
):
    M = 128
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout()
    parent_layout = parent_layout_fn()
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_copy_no_scales_warpx2_subslice_view_kernel[(1, )](
        inp, out, shared_layout, parent_layout, slice_start, num_warps=4
    )

    expected = expected_fn(inp)
    assert not torch.equal(out, inp)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert family in expected_opcode
    _assert_exact_cp_ptx_llir_match(compiled, [expected_opcode])
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttng.tmem_subslice" not in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
@pytest.mark.parametrize("parent_index", [0, 1])
@pytest.mark.parametrize(
    "family,tmem_layout_fn,expected_fn,expected_opcode",
    [
        (
            "01_23",
            _make_tmem_copy_warpx2_tmem_layout,
            _expected_tmem_copy_warpx2_01_23_output,
            "tcgen05.cp.cta_group::1.warpx2::01_23.64x128b",
        ),
        (
            "02_13",
            _make_tmem_copy_warpx2_tmem_layout_02_13,
            _expected_tmem_copy_warpx2_02_13_output,
            "tcgen05.cp.cta_group::1.warpx2::02_13.64x128b",
        ),
    ],
)
def test_tmem_runtime_matrix_cp_no_scales_warpx2_indexed_view_positive(
    dtype_name,
    torch_dtype,
    parent_index,
    family,
    tmem_layout_fn,
    expected_fn,
    expected_opcode,
):
    M = 128
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout()
    tmem_layout = tmem_layout_fn()
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_copy_no_scales_warpx2_indexed_view_kernel[(1, )](
        inp, out, shared_layout, tmem_layout, 2, parent_index, num_warps=4
    )

    expected = expected_fn(inp)
    assert not torch.equal(out, inp)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert family in expected_opcode
    _assert_exact_cp_ptx_llir_match(compiled, [expected_opcode])
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
@pytest.mark.parametrize("parent_index", [0, 1])
@pytest.mark.parametrize(
    "family,tmem_layout_fn,expected_fn,expected_opcode",
    [
        (
            "01_23",
            _make_tmem_copy_warpx2_tmem_layout,
            _expected_tmem_copy_warpx2_01_23_output,
            "tcgen05.cp.cta_group::1.warpx2::01_23.64x128b",
        ),
        (
            "02_13",
            _make_tmem_copy_warpx2_tmem_layout_02_13,
            _expected_tmem_copy_warpx2_02_13_output,
            "tcgen05.cp.cta_group::1.warpx2::02_13.64x128b",
        ),
    ],
)
def test_tmem_runtime_matrix_cp_no_scales_warpx2_slice_index_view_positive(
    dtype_name,
    torch_dtype,
    parent_index,
    family,
    tmem_layout_fn,
    expected_fn,
    expected_opcode,
):
    M = 128
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout()
    parent_layout = _lift_tmem_layout(tmem_layout_fn(), [2])
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_copy_no_scales_warpx2_slice_index_view_kernel[(1, )](
        inp, out, shared_layout, parent_layout, parent_index, num_warps=4
    )

    expected = expected_fn(inp)
    assert not torch.equal(out, inp)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert family in expected_opcode
    _assert_exact_cp_ptx_llir_match(compiled, [expected_opcode])
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
def test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_positive(dtype_name, torch_dtype):
    M = 256
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout_twocta()
    tmem_layout = _make_tmem_copy_warpx2_tmem_layout_twocta()
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_copy_no_scales_warpx2_twocta_kernel[(1, )](
        inp, out, shared_layout, tmem_layout, num_warps=4, num_ctas=2
    )

    expected = _expected_tmem_copy_warpx2_01_23_twocta_output(inp)
    assert not torch.equal(out, inp)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    _assert_exact_cp_ptx_llir_match(
        compiled,
        ["tcgen05.cp.cta_group::2.warpx2::01_23.64x128b"],
    )
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(2)])
    ptx = compiled.asm["ptx"]
    assert "tcgen05.cp.cta_group::1" not in ptx
    assert "tcgen05.cp.cta_group::2.warpx2::02_13" not in ptx
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
@pytest.mark.parametrize("slice_start", [0, 4])
def test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_subslice_view_positive(
    dtype_name, torch_dtype, slice_start
):
    M = 256
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout_twocta()
    parent_layout = _make_tmem_copy_warpx2_parent_tmem_layout_twocta()
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_copy_no_scales_warpx2_twocta_subslice_view_kernel[(1, )](
        inp, out, shared_layout, parent_layout, slice_start, num_warps=4, num_ctas=2
    )

    expected = _expected_tmem_copy_warpx2_01_23_twocta_output(inp)
    assert not torch.equal(out, inp)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    _assert_exact_cp_ptx_llir_match(
        compiled,
        ["tcgen05.cp.cta_group::2.warpx2::01_23.64x128b"],
    )
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(2)])
    ptx = compiled.asm["ptx"]
    assert "tcgen05.cp.cta_group::1" not in ptx
    assert "tcgen05.cp.cta_group::2.warpx2::02_13" not in ptx
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttng.tmem_subslice" not in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
@pytest.mark.parametrize("parent_index", [0, 1])
def test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_indexed_view_positive(
    dtype_name, torch_dtype, parent_index
):
    M = 256
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout_twocta()
    tmem_layout = _make_tmem_copy_warpx2_tmem_layout_twocta()
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_copy_no_scales_warpx2_twocta_indexed_view_kernel[(1, )](
        inp, out, shared_layout, tmem_layout, 2, parent_index, num_warps=4, num_ctas=2
    )

    expected = _expected_tmem_copy_warpx2_01_23_twocta_output(inp)
    assert not torch.equal(out, inp)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    _assert_exact_cp_ptx_llir_match(
        compiled,
        ["tcgen05.cp.cta_group::2.warpx2::01_23.64x128b"],
    )
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(2)])
    ptx = compiled.asm["ptx"]
    assert "tcgen05.cp.cta_group::1" not in ptx
    assert "tcgen05.cp.cta_group::2.warpx2::02_13" not in ptx
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
@pytest.mark.parametrize("parent_index", [0, 1])
def test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_slice_index_view_positive(
    dtype_name, torch_dtype, parent_index
):
    M = 256
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout_twocta()
    parent_layout = _lift_tmem_layout(
        _make_tmem_copy_warpx2_tmem_layout_twocta(), [2]
    )
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)

    compiled = tmem_copy_no_scales_warpx2_twocta_slice_index_view_kernel[(1, )](
        inp,
        out,
        shared_layout,
        parent_layout,
        parent_index,
        num_warps=4,
        num_ctas=2,
    )

    expected = _expected_tmem_copy_warpx2_01_23_twocta_output(inp)
    assert not torch.equal(out, inp)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    _assert_exact_cp_ptx_llir_match(
        compiled,
        ["tcgen05.cp.cta_group::2.warpx2::01_23.64x128b"],
    )
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(2)])
    ptx = compiled.asm["ptx"]
    assert "tcgen05.cp.cta_group::1" not in ptx
    assert "tcgen05.cp.cta_group::2.warpx2::02_13" not in ptx
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttg.memdesc_subslice" in ttgir
    assert "ttg.memdesc_index" in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
@pytest.mark.parametrize("parent_index", [0, 1])
def test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_slice_index_view_reports_clean_unsupported(
    dtype_name, torch_dtype, parent_index, capfd
):
    M = 256
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout_twocta()
    parent_layout = _lift_tmem_layout(
        _make_tmem_copy_warpx2_tmem_layout_02_13_twocta(), [2]
    )
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)

    with pytest.raises(RuntimeError) as excinfo:
        tmem_copy_no_scales_warpx2_twocta_slice_index_view_kernel[(1, )](
            inp, out, shared_layout, parent_layout, parent_index, num_warps=4, num_ctas=2
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "maps to tcgen05.copy.warpx2::02_13.64x128b" in text
    assert "could not synthesize a compatible shared-memory descriptor plan" in text
    assert "preserves the high source-column bit" in text
    assert "source-column preservation requirement" in text
    assert "logical row bit 5 maps to a one-dword source offset" in text
    assert "source offset 32 emits the opcode and writes the correct low destination columns" in text
    assert "duplicates that low source-column pair into the high destination columns" in text
    assert "Non-zero subaligned destination dword deltas fault" in text
    assert "aligned deltas that complete the single-CTA schedule read zeros under cta_group::2" in text
    assert "cta_group::1 copies is not valid" in text
    assert "cleanly unsupported" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
@pytest.mark.parametrize("parent_index", [0, 1])
def test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_indexed_view_reports_clean_unsupported(
    dtype_name, torch_dtype, parent_index, capfd
):
    M = 256
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout_twocta()
    tmem_layout = _make_tmem_copy_warpx2_tmem_layout_02_13_twocta()
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)

    with pytest.raises(RuntimeError) as excinfo:
        tmem_copy_no_scales_warpx2_twocta_indexed_view_kernel[(1, )](
            inp, out, shared_layout, tmem_layout, 2, parent_index, num_warps=4, num_ctas=2
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "maps to tcgen05.copy.warpx2::02_13.64x128b" in text
    assert "could not synthesize a compatible shared-memory descriptor plan" in text
    assert "preserves the high source-column bit" in text
    assert "source-column preservation requirement" in text
    assert "logical row bit 5 maps to a one-dword source offset" in text
    assert "source offset 32 emits the opcode and writes the correct low destination columns" in text
    assert "duplicates that low source-column pair into the high destination columns" in text
    assert "Non-zero subaligned destination dword deltas fault" in text
    assert "aligned deltas that complete the single-CTA schedule read zeros under cta_group::2" in text
    assert "cta_group::1 copies is not valid" in text
    assert "cleanly unsupported" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
@pytest.mark.parametrize("slice_start", [0, 4])
def test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_subslice_view_reports_clean_unsupported(
    dtype_name, torch_dtype, slice_start, capfd
):
    M = 256
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout_twocta()
    parent_layout = _make_tmem_copy_warpx2_parent_tmem_layout_02_13_twocta()
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)

    with pytest.raises(RuntimeError) as excinfo:
        tmem_copy_no_scales_warpx2_twocta_subslice_view_kernel[(1, )](
            inp, out, shared_layout, parent_layout, slice_start, num_warps=4, num_ctas=2
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "maps to tcgen05.copy.warpx2::02_13.64x128b" in text
    assert "could not synthesize a compatible shared-memory descriptor plan" in text
    assert "preserves the high source-column bit" in text
    assert "source-column preservation requirement" in text
    assert "logical row bit 5 maps to a one-dword source offset" in text
    assert "source offset 32 emits the opcode and writes the correct low destination columns" in text
    assert "duplicates that low source-column pair into the high destination columns" in text
    assert "Non-zero subaligned destination dword deltas fault" in text
    assert "aligned deltas that complete the single-CTA schedule read zeros under cta_group::2" in text
    assert "cta_group::1 copies is not valid" in text
    assert "cleanly unsupported" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
def test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_candidate_reports_clean_unsupported(
    dtype_name, torch_dtype, capfd
):
    M = 256
    N = 4
    shared_layout = _make_tmem_copy_warpx2_shared_layout_twocta()
    tmem_layout = _make_tmem_copy_warpx2_tmem_layout_02_13_twocta()
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)

    with pytest.raises(RuntimeError) as excinfo:
        tmem_copy_no_scales_warpx2_twocta_kernel[(1, )](
            inp, out, shared_layout, tmem_layout, num_warps=4, num_ctas=2
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "maps to tcgen05.copy.warpx2::02_13.64x128b" in text
    assert "could not synthesize a compatible shared-memory descriptor plan" in text
    assert "preserves the high source-column bit" in text
    assert "source-column preservation requirement" in text
    assert "logical row bit 5 maps to a one-dword source offset" in text
    assert "source offset 32 emits the opcode and writes the correct low destination columns" in text
    assert "duplicates that low source-column pair into the high destination columns" in text
    assert "Non-zero subaligned destination dword deltas fault" in text
    assert "aligned deltas that complete the single-CTA schedule read zeros under cta_group::2" in text
    assert "cta_group::1 copies is not valid" in text
    assert "cleanly unsupported" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "case_name,cta_group,tmem_layout,dtype_name,torch_dtype",
    CP_NO_SCALES_WARPX2_SUBWORD_UNSUPPORTED_CASES,
)
def test_tmem_runtime_matrix_cp_no_scales_warpx2_subword_dtypes_report_clean_error(
    case_name, cta_group, tmem_layout, dtype_name, torch_dtype, capfd
):
    M = 128 if cta_group == 1 else 256
    N = 4
    shared_layout = (
        _make_tmem_copy_warpx2_shared_layout()
        if cta_group == 1
        else _make_tmem_copy_warpx2_shared_layout_twocta()
    )
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N).to(torch_dtype)
    out = torch.empty_like(inp)

    with pytest.raises(RuntimeError) as excinfo:
        if cta_group == 1:
            tmem_copy_no_scales_warpx2_candidate_kernel[(1, )](
                inp, out, shared_layout, tmem_layout, num_warps=4
            )
        else:
            tmem_copy_no_scales_warpx2_twocta_kernel[(1, )](
                inp, out, shared_layout, tmem_layout, num_warps=4, num_ctas=2
            )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "could not synthesize a compatible shared-memory descriptor plan" in text
    assert (
        "instruction schedule reads descriptor-loader source footprint" in text
        or "has no representable MMAv5 shared-memory descriptor" in text
        or "direct-seed descriptor plan requires a shared-memory source layout" in text
        or "source row projection requires logical row bit 5" in text
        or "requires enough TMEM column bases to cover the copy instruction width" in text
    )
    if "has no representable MMAv5 shared-memory descriptor" in text:
        assert "descriptor shape" in text
        assert "instruction shape" in text
    if "requires enough TMEM column bases to cover the copy instruction width" in text:
        bitwidth = CP_NO_SCALES_SUBWORD_BITWIDTHS[dtype_name]
        instruction_columns = 128 // bitwidth
        required_bits = int(math.log2(instruction_columns))
        lanes_per_dword = 32 // bitwidth
        physical_dword_columns = instruction_columns // lanes_per_dword
        assert "destination-column footprint requirement" in text
        assert f"exposes 2 column basis bits" in text
        assert f"{instruction_columns}-column copy instruction requires {required_bits} logical column basis bit" in text
        assert f"{physical_dword_columns} physical 32-bit dword columns" in text
        assert f"{lanes_per_dword} packed lanes per word" in text
        assert "packed-lane source/destination storage model" in text
        assert "descriptor footprint coverage alone is not a correctness proof" in text
    if "01_23" in case_name and "requires enough TMEM column bases" not in text:
        assert "subword tcgen05.copy instruction source footprint spans" in text
        assert "descriptor semantic-equivalence model" in text
        assert "descriptor footprint coverage alone is not a correctness proof" in text
    assert "warpx2 tcgen05.copy currently requires 32-bit shared elements" not in text
    assert "cleanly unsupported" in text
    assert "error encountered during parsing" in str(excinfo.value)
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
@pytest.mark.parametrize(
    "family,tmem_layout,expected_fn,expected_opcode",
    [
        (
            "warpx2::01_23.64x128b",
            _make_tmem_copy_warpx2_tmem_layout(),
            _expected_tmem_copy_warpx2_01_23_output,
            "tcgen05.cp.cta_group::1.warpx2::01_23.64x128b",
        ),
        (
            "warpx2::02_13.64x128b",
            _make_tmem_copy_warpx2_tmem_layout_02_13(),
            _expected_tmem_copy_warpx2_02_13_output,
            "tcgen05.cp.cta_group::1.warpx2::02_13.64x128b",
        ),
    ],
)
def test_tmem_runtime_matrix_cp_no_scales_warpx2_dense_shared_rematerializes(
    family, tmem_layout, expected_fn, expected_opcode, dtype_name, torch_dtype
):
    M = 128
    N = 4
    shared_layout = _make_tmem_copy_128x128_shared_layout()
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N).to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_copy_no_scales_warpx2_candidate_kernel[(1, )](
        inp, out, shared_layout, tmem_layout, num_warps=4
    )

    expected = expected_fn(inp)
    assert not torch.equal(out, inp)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert family in expected_opcode
    _assert_exact_cp_ptx_llir_match(compiled, [expected_opcode])
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_warpx2_row_permuted_destination_reports_clean_unsupported(capfd):
    M = 128
    N = 4
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty_like(inp)
    shared_layout = _make_tmem_copy_warpx2_shared_layout()
    tmem_layout = _make_tmem_copy_warpx2_tmem_layout_row_permuted()

    with pytest.raises(Exception) as excinfo:
        tmem_copy_no_scales_warpx2_candidate_kernel[(1, )](
            inp, out, shared_layout, tmem_layout, num_warps=4
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "maps to tcgen05.copy.warpx2::01_23.64x128b" in text
    assert "non-broadcast TMEM row bases" in text
    assert "source-row projection schedule" in text
    assert "destination-row order requirement" in text
    assert "destination rows" in text
    assert "cleanly unsupported" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
def test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_dense_shared_rematerializes(dtype_name, torch_dtype):
    M = 256
    N = 4
    shared_layout = _make_tmem_copy_128x128_shared_layout_twocta()
    tmem_layout = _make_tmem_copy_warpx2_tmem_layout_twocta()
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N).to(torch_dtype)
    out = torch.empty_like(inp)

    compiled = tmem_copy_no_scales_warpx2_twocta_kernel[(1, )](
        inp, out, shared_layout, tmem_layout, num_warps=4, num_ctas=2
    )

    expected = _expected_tmem_copy_warpx2_01_23_twocta_output(inp)
    assert not torch.equal(out, inp)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::2.warpx2::01_23.64x128b"])
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(2)])
    ptx = compiled.asm["ptx"]
    assert "tcgen05.cp.cta_group::1" not in ptx
    assert "tcgen05.cp.cta_group::2.warpx2::02_13" not in ptx
    assert ptx.count("fence.proxy.async.shared::cluster") == 2
    ttgir = compiled.asm["ttgir"]
    assert "tensor_memory_linear" in ttgir
    assert "ttng.tmem_copy" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_WARPX2_DTYPES)
def test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_twocta_dense_shared_reports_clean_unsupported(
    dtype_name, torch_dtype, capfd
):
    M = 256
    N = 4
    shared_layout = _make_tmem_copy_128x128_shared_layout_twocta()
    tmem_layout = _make_tmem_copy_warpx2_tmem_layout_02_13_twocta()
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N).to(torch_dtype)
    out = torch.empty_like(inp)

    with pytest.raises(RuntimeError) as excinfo:
        tmem_copy_no_scales_warpx2_twocta_kernel[(1, )](
            inp, out, shared_layout, tmem_layout, num_warps=4, num_ctas=2
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "maps to tcgen05.copy.warpx2::02_13.64x128b" in text
    assert "could not synthesize a compatible shared-memory descriptor plan" in text
    assert "preserves the high source-column bit" in text
    assert "source-column preservation requirement" in text
    assert "logical row bit 5 maps to a one-dword source offset" in text
    assert "source offset 32 emits the opcode and writes the correct low destination columns" in text
    assert "duplicates that low source-column pair into the high destination columns" in text
    assert "Non-zero subaligned destination dword deltas fault" in text
    assert "aligned deltas that complete the single-CTA schedule read zeros under cta_group::2" in text
    assert "cta_group::1 copies is not valid" in text
    assert "cleanly unsupported" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_kind,dtype_name,torch_dtype,N,swizzle,expected_count", CP_NO_SCALES_TWOCTA_CASES)
def test_tmem_runtime_matrix_cp_no_scales_twocta_codegen(
    layout_kind, dtype_name, torch_dtype, N, swizzle, expected_count
):
    M = 256
    cga_layout = _make_2cta_cga_layout((2, 1), (2, 1), (1, 0), 0)
    if layout_kind == "linear":
        layout = _make_tmem_linear_layout_mmav5_twocta(M, N)
    else:
        layout = TensorMemoryLayout(block=(128, N), col_stride=1, cga_layout=cga_layout, two_ctas=True)

    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N).to(torch_dtype)
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

    _assert_exact_commit_ptx_llir_match(
        compiled,
        ["tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"],
    )
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


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error():
    M = 256
    N = 64
    swizzle = 32
    cga_layout = _make_2cta_cga_layout((4, 1), (4, 1), (1, 0), 0)
    layout = _make_tmem_linear_layout_mmav5_twocta(M, N)
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty_like(inp)

    with pytest.raises(CompilationError) as excinfo:
        tmem_copy_no_scales_twocta_kernel[(1, )](
            inp,
            out,
            layout,
            tuple(tuple(basis) for basis in cga_layout),
            M,
            N,
            swizzle,
            num_ctas=4,
            num_warps=4,
        )

    text = str(excinfo.value)
    assert "Layout has 2 CTAs per CGA, but the context requires 4 CTAs per CGA." in text
    assert "Assertion" not in text
    assert "PassManager::run failed" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_kind,dtype_name,torch_dtype", CP_NO_SCALES_TWOCTA_128X128_CASES)
def test_tmem_runtime_matrix_cp_no_scales_twocta_128x128b_codegen(layout_kind, dtype_name, torch_dtype):
    M = 256
    N = 4
    cga_layout = _make_2cta_cga_layout((2, 1), (2, 1), (1, 0), 0)
    if layout_kind == "linear":
        layout = _make_tmem_linear_layout_mmav5_twocta(M, N)
    else:
        layout = TensorMemoryLayout(block=(128, N), col_stride=1, cga_layout=cga_layout, two_ctas=True)

    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
    out = torch.empty_like(inp)
    compiled = tmem_copy_128x128_twocta_kernel[(1, )](
        inp,
        out,
        layout,
        num_ctas=2,
        num_warps=4,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::2.128x128b"])
    ptx = compiled.asm["ptx"]
    llir = compiled.asm["llir"]
    first_cp_ptx = ptx.index("tcgen05.cp.cta_group::2.128x128b")
    first_cp_llir = llir.index("tcgen05.cp.cta_group::2.128x128b")

    _assert_exact_commit_ptx_llir_match(
        compiled,
        ["tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"],
    )
    assert "tcgen05.cp.cta_group::1" not in ptx
    assert ptx.count("fence.proxy.async.shared::cluster") == 1
    assert ptx.index("barrier.cluster.arrive.aligned") < ptx.index("barrier.cluster.wait.aligned") < first_cp_ptx
    assert llir.index("llvm.nvvm.barrier.cluster.arrive.aligned") < llir.index("llvm.nvvm.barrier.cluster.wait.aligned") < first_cp_llir
    if layout_kind == "linear":
        assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_no_scales_twocta_noncanonical_block_reports_clean_unsupported(capfd):
    M = 256
    N = 4
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout_noncanonical_twocta_block(M, N)

    with pytest.raises(Exception) as excinfo:
        tmem_copy_no_scales_twocta_noncanonical_block_kernel[(1, )](
            inp,
            out,
            layout,
            num_ctas=2,
            num_warps=4,
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "maps to tcgen05.copy.128x128b" in text
    assert "two-CTA destination layouts require the canonical TMEM block basis [[128, 0]]" in text
    assert "CTA ownership is part of the instruction schedule" in text
    assert "cleanly unsupported" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text



@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype,M,N,swizzle,expected_count", CP_LINEAR_NO_SCALES_SUBWORD_CASES)
def test_tmem_runtime_matrix_cp_no_scales_linear_subword_dtypes(
    dtype_name, torch_dtype, M, N, swizzle, expected_count
):
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N).to(torch_dtype)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout(M, N)

    compiled = tmem_copy_no_scales_linear_kernel[(1, )](inp, out, layout, M, N, swizzle, num_warps=4)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.128x256b"] * expected_count)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,torch_dtype", CP_NO_SCALES_SUBWORD_DTYPES)
def test_tmem_runtime_matrix_cp_no_scales_legacy_subword_dtypes_report_clean_error(
    dtype_name, torch_dtype, capfd
):
    m = n = 128
    inp = torch.arange(m * n, device="cuda", dtype=torch.int32).reshape(m, n).to(torch_dtype)
    out = torch.empty_like(inp)

    with pytest.raises(RuntimeError) as excinfo:
        tmem_copy_no_scales_kernel[(1, )](inp, out, m, n, n, 32, num_warps=4)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "maps to tcgen05.copy.128x256b" in text
    assert "source column bit 0 maps to no shared offset" in text
    assert "sub-32-bit packed lane state" in text
    assert "physical dword-column projection is contiguous over 8 columns" in text
    assert "packed-lane source-storage requirement" in text
    assert "logical source columns per 32-bit shared-memory word" in text
    assert "unpacked TensorMemoryLinearLayout" in text
    assert "cleanly unsupported" in text
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
        "does not match any recognized tcgen05.copy family" in text
        or "could not synthesize a compatible shared-memory descriptor plan for it" in text
    )
    if name == "mixed":
        assert "does not support TMEM row bases that mix row and column contributions" in text
        assert "physical TMEM delta" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n,tile_n,expected_count,expected_opcode", CP_LINEAR_TILE_PERMUTED_CASES)
def test_tmem_runtime_matrix_cp_no_scales_linear_tile_permuted(n, tile_n, expected_count, expected_opcode):
    m = 128
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout_tile_permuted(m, n, tile_n)

    compiled = tmem_copy_no_scales_linear_kernel[(1, )](inp, out, layout, m, n, 32, num_warps=4)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    _assert_exact_cp_ptx_llir_match(compiled, [expected_opcode] * expected_count)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n,tile_n,perm_kind,expected_count,expected_opcode", CP_LINEAR_TILE_SELECTOR_PERMUTED_CASES)
def test_tmem_runtime_matrix_cp_no_scales_linear_tile_selector_permuted(
    n, tile_n, perm_kind, expected_count, expected_opcode
):
    m = 128
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout_tile_selector_permuted(m, n, tile_n, perm_kind)

    compiled = tmem_copy_no_scales_linear_kernel[(1, )](inp, out, layout, m, n, 32, num_warps=4)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    _assert_exact_cp_ptx_llir_match(compiled, [expected_opcode] * expected_count)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n,tile_n", CP_LINEAR_TILE_PERMUTED_UNSUPPORTED_CASES)
def test_tmem_runtime_matrix_cp_no_scales_linear_tile_permuted_subinstruction_reports_clean_unsupported(
    n, tile_n, capfd
):
    m = 128
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout_tile_permuted(m, n, tile_n)

    with pytest.raises(Exception) as excinfo:
        tmem_copy_no_scales_linear_kernel[(1, )](inp, out, layout, m, n, 32, num_warps=4)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "maps to tcgen05.copy.128x256b" in text
    assert "contiguous in physical TMEM column order" in text
    assert "destination-column permutation requirement" in text
    assert "copy atom writes the full" in text
    assert "cleanly unsupported" in text
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
        "does not match any recognized tcgen05.copy family" in text
        or "could not synthesize a compatible shared-memory descriptor plan for it" in text
    )
    assert (
        "ascending physical row order" in text
        or "ascending physical column order" in text
        or "aligned to the copy instruction width" in text
        or "contiguous in physical TMEM column order" in text
    )
    if row_perm_kind != "identity":
        assert "source-row projection schedule" in text
        assert "first non-ascending basis" in text
        assert "full physical row footprint" in text
        assert "destination-row order requirement" in text
        assert "destination rows" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_kind,dtype_name,torch_dtype,M", CP_NO_SCALES_128X128_CASES)
def test_tmem_runtime_matrix_cp_128x128(layout_kind, dtype_name, torch_dtype, M):
    N = 4
    inp = torch.arange(M * N, device="cuda", dtype=torch_dtype).reshape(M, N)
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
@pytest.mark.parametrize("dtype_name,torch_dtype,N", CP_NO_SCALES_128X128_SUBWORD_EXACT_CASES)
def test_tmem_runtime_matrix_cp_128x128_subword_exact_width(dtype_name, torch_dtype, N):
    M = 128
    inp = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N).to(torch_dtype)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout(M, N)
    shared_layout = _make_tmem_copy_dense_shared_layout(M, N)

    compiled = tmem_copy_128x128_subword_exact_kernel[(1, )](
        inp, out, N, layout, shared_layout, num_warps=4
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.128x128b"])
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("N", (1, 2))
def test_tmem_runtime_matrix_cp_no_scales_too_small_destination_reports_clean_error(N, capfd):
    M = 128
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout(M, N)
    shared_layout = _make_tmem_copy_dense_shared_layout(M, N)

    with pytest.raises((CompilationError, RuntimeError)) as excinfo:
        tmem_copy_128x128_subword_exact_kernel[(1, )](
            inp, out, N, layout, shared_layout, num_warps=4
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "does not match any recognized tcgen05.copy family" in text
    assert "requires at least 128 bits of logical columns" in text
    assert f"got {N * 32} bits from {N} columns x 32-bit elements" in text
    assert "cannot borrow hidden parent columns" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_scales_warpx4():
    smem_h, smem_w = 64, 16
    inp = torch.randint(size=(smem_h, smem_w), low=-100, high=100, dtype=torch.int8, device="cuda")
    out = torch.empty_like(inp)

    compiled = tmem_copy_scales_warpx4_kernel[(1, )](inp, out)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)

    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.warpx4.32x128b"] * 2)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    assert "ttg.memdesc_reinterpret" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_cp_scales_warpx4_twocta_direct_copy():
    smem_h, smem_w = 128, 16
    inp = torch.randint(size=(smem_h, smem_w), low=-100, high=100, dtype=torch.int8, device="cuda")
    out = torch.empty_like(inp)

    compiled = tmem_copy_scales_warpx4_twocta_kernel[(1, )](inp, out, num_warps=4, num_ctas=2)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    _assert_exact_cp_ptx_llir_match(compiled, ["tcgen05.cp.cta_group::1.warpx4.32x128b"] * 2)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    assert "ttg.memdesc_reinterpret" not in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,block_n,block_k,num_ctas,multicast,acc_layout_kind",
                         CP_SCALES_WARPX4_SCALED_MMA_CASES)
def test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix(
    a_format, b_format, block_n, block_k, num_ctas, multicast, acc_layout_kind
):
    block_m = 256 if num_ctas == 2 else 128
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
        multicast=multicast,
        acc_layout_kind=acc_layout_kind,
    )
    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    expected = _expected_scaled_cp_opcode(num_ctas)
    expected_count = (1 + block_n // 128) * (block_k // 128) * (32 // vec_size)
    _assert_exact_cp_ptx_llir_match(compiled, [expected] * expected_count)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    expected_mma_count = (block_k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert len(mma_ops) == expected_mma_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, num_ctas) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(num_ctas)])
    ttgir = compiled.asm["ttgir"]
    if multicast and num_ctas == 2:
        assert "{multicast}" in ttgir
    else:
        assert "{multicast}" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,block_n,block_k,num_ctas,multicast,acc_layout_kind",
                         CP_SCALES_WARPX4_SCALED_MMA_USE_ACC_CASES)
def test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix_use_acc(
    a_format, b_format, block_n, block_k, num_ctas, multicast, acc_layout_kind
):
    block_m = 256 if num_ctas == 2 else 128
    m, n, k = block_m, block_n, block_k
    acc_init = 1.0
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
        multicast=multicast,
        acc_layout_kind=acc_layout_kind,
        acc_init=acc_init,
    )
    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T + acc_init, atol=1e-3, rtol=1e-3)

    expected = _expected_scaled_cp_opcode(num_ctas)
    expected_count = (1 + block_n // 128) * (block_k // 128) * (32 // vec_size)
    _assert_exact_cp_ptx_llir_match(compiled, [expected] * expected_count)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    expected_mma_count = (block_k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert len(mma_ops) == expected_mma_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, num_ctas) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(num_ctas)])
    ttgir = compiled.asm["ttgir"]
    if multicast and num_ctas == 2:
        assert "{multicast}" in ttgir
    else:
        assert "{multicast}" not in ttgir


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
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, num_ctas) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(num_ctas)])


MMA_CASES = [
    ("legacy_no_acc", TensorMemoryLayout((128, 128), col_stride=1), False),
    ("legacy_use_acc", TensorMemoryLayout((128, 128), col_stride=1), True),
    ("linear_no_acc", _make_tmem_linear_layout(128, 128), False),
    ("linear_use_acc", _make_tmem_linear_layout(128, 128), True),
]

MMA_PLAIN_KINDS = ("f16", "tf32", "bf16", "f8e5m2", "f8e4m3")

MMA_PLAIN_KIND_EXPECTED_OP_COUNTS = {
    "f16": 2,
    "tf32": 4,
    "bf16": 2,
    "f8e5m2": 1,
    "f8e4m3": 1,
}


def _expected_plain_mma_op_count(kind, k):
    assert k % 32 == 0
    return MMA_PLAIN_KIND_EXPECTED_OP_COUNTS[kind] * (k // 32)


def _expected_m64_plain_mma_op_count(kind, k, acc_layout_kind, n):
    encoding_multiplier = n // 64 if _uses_tmem_encoding_layout(acc_layout_kind) else 1
    return encoding_multiplier * _expected_plain_mma_op_count(kind, k)


MMA_TILE_PERMUTED_KIND_EXPECTED_OP_COUNTS = {
    kind: count * 4 for kind, count in MMA_PLAIN_KIND_EXPECTED_OP_COUNTS.items()
}


def _expected_tile_permuted_mma_op_count(kind, k):
    assert k % 32 == 0
    return MMA_TILE_PERMUTED_KIND_EXPECTED_OP_COUNTS[kind] * (k // 32)


def _expected_lhs_tile_permuted_mma_op_count(kind, k):
    assert k % 32 == 0
    return MMA_PLAIN_KIND_EXPECTED_OP_COUNTS[kind] * (k // 32)

MMA_PLAIN_KIND_CASES = [
    (kind, acc_layout_kind)
    for kind, acc_layout_kind in product(MMA_PLAIN_KINDS, TMEM_LAYOUT_KIND_CASES)
]

def _dedupe_matrix_cases(cases):
    deduped = []
    seen = set()
    for case in cases:
        if case in seen:
            continue
        seen.add(case)
        deduped.append(case)
    return deduped


MMA_REPRESENTATIVE_NK_CASES = ((32, 32), (64, 64), (128, 128), (256, 128))
MMA_KIND_REPRESENTATIVE_NK = (128, 64)

# The MMA runtime surface is intentionally representative rather than
# exhaustive: every data kind is checked on a stable geometry, and the broad
# N/K/view/use-acc axes are checked on f16 where those axes affect descriptor
# and packet planning but not the opcode-family dispatch.
MMA_PLAIN_KIND_ACC_CASES = _dedupe_matrix_cases(
    [(kind, "linear", *MMA_KIND_REPRESENTATIVE_NK) for kind in MMA_PLAIN_KINDS] + [
        ("f16", acc_layout_kind, n, k)
        for acc_layout_kind in TMEM_LAYOUT_KIND_CASES
        for n, k in MMA_REPRESENTATIVE_NK_CASES
    ]
)

MMA_INDEXED_ACC_CASES = _dedupe_matrix_cases(
    [(kind, "linear", *MMA_KIND_REPRESENTATIVE_NK, False) for kind in MMA_PLAIN_KINDS] + [
        ("f16", parent_layout_kind, n, k, use_acc)
        for parent_layout_kind in TMEM_LAYOUT_KIND_CASES
        for n, k in MMA_REPRESENTATIVE_NK_CASES
        for use_acc in (False, True)
        # Linear parent views keep the whole [2, M, N] physical image live; N=256
        # needs 1024 TMEM columns and is therefore a hardware resource boundary.
        if not (parent_layout_kind == "linear" and n == 256)
    ] + [
        ("f16", "linear_unit_parent", 256, k, use_acc)
        for k, use_acc in product((32, 128), (False, True))
    ]
)

MMA_ACC_SUBSLICE_CASES = _dedupe_matrix_cases(
    [(kind, *MMA_KIND_REPRESENTATIVE_NK, 0, False) for kind in MMA_PLAIN_KINDS] + [
        ("f16", n, k, slice_start, use_acc)
        for n, k in MMA_REPRESENTATIVE_NK_CASES
        for slice_start, use_acc in product((0, n), (False, True))
    ]
)

MMA_TWOCTA_TMA_NON_TF32_DTYPES = {
    "f16": (torch.float16, ttgl.float16, "tcgen05.mma.cta_group::2.kind::f16", 1e-1, 8e-2),
    "bf16": (torch.bfloat16, ttgl.bfloat16, "tcgen05.mma.cta_group::2.kind::f16", 1e-1, 1e-1),
    "f8e5m2": (torch.float8_e5m2, ttgl.float8e5, "tcgen05.mma.cta_group::2.kind::f8f6f4", 1e-1, 1e-1),
    "f8e4m3": (torch.float8_e4m3fn, ttgl.float8e4nv, "tcgen05.mma.cta_group::2.kind::f8f6f4", 1e-1, 1e-1),
}

MMA_TWOCTA_TMA_NON_TF32_CASES = _dedupe_matrix_cases(
    [(dtype_name, "linear", 128, 64, False) for dtype_name in MMA_TWOCTA_TMA_NON_TF32_DTYPES] + [
        ("f16", acc_layout_kind, block_n, block_k, use_acc)
        for acc_layout_kind in TMEM_LAYOUT_KIND_CASES
        for block_n, block_k in MMA_REPRESENTATIVE_NK_CASES
        for use_acc in (False, True)
    ] + [
        ("bf16", "legacy", 256, 128, True),
        ("f8e5m2", "linear", 32, 32, True),
        ("f8e4m3", "legacy", 64, 128, False),
    ]
)

MMA_TWOCTA_TMA_TF32_CASES = _dedupe_matrix_cases(
    [("linear", 128, 64), ("legacy", 128, 64)] + [
        (acc_layout_kind, block_n, block_k)
        for acc_layout_kind, block_n, block_k in (
            ("linear", 32, 32),
            ("legacy", 64, 128),
            ("linear", 256, 128),
        )
    ]
)

MMA_TWOCTA_PLAIN_KIND_CASES = MMA_PLAIN_KIND_ACC_CASES

MMA_TWOCTA_INDEXED_ACC_CASES = MMA_INDEXED_ACC_CASES

MMA_TWOCTA_ACC_SUBSLICE_CASES = MMA_ACC_SUBSLICE_CASES

MMA_M64_REPRESENTATIVE_NK_CASES = ((32, 32), (64, 64), (128, 128), (256, 128))

MMA_M64_PLAIN_KIND_CASES = _dedupe_matrix_cases(
    [(kind, "linear", *MMA_KIND_REPRESENTATIVE_NK, False) for kind in MMA_PLAIN_KINDS] + [
        ("f16", acc_layout_kind, n, k, use_acc)
        for acc_layout_kind in TMEM_LAYOUT_KIND_CASES
        for n, k in MMA_M64_REPRESENTATIVE_NK_CASES
        for use_acc in (False, True)
        if acc_layout_kind == "linear" or n in (64, 128)
    ]
)

MMA_M64_ACC_SUBSLICE_CASES = MMA_ACC_SUBSLICE_CASES

MMA_TILE_PERMUTED_CASES = [
    (32, 8, 32),
    (64, 16, 64),
    (128, 32, 128),
    (256, 64, 128),
]

MMA_TILE_PERMUTED_KIND_CASES = _dedupe_matrix_cases(
    [(kind, 128, 32, 64) for kind in MMA_PLAIN_KINDS] + [
        ("f16", n, tile_n, k)
        for n, tile_n, k in (
            (32, 8, 32),
            (64, 16, 64),
            (128, 32, 128),
            (256, 64, 128),
        )
    ]
)

MMA_LHS_TILE_PERMUTED_NK_CASES = _dedupe_matrix_cases(
    [(kind, *MMA_KIND_REPRESENTATIVE_NK, MMA_KIND_REPRESENTATIVE_NK[1] // 4) for kind in MMA_PLAIN_KINDS] + [
        ("f16", n, k, k // 4)
        for n, k in ((32, 32), (64, 64), (128, 128), (256, 256))
    ]
)

MMA_LHS_SUBSLICE_NK_CASES = MMA_PLAIN_KIND_ACC_CASES


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
@pytest.mark.parametrize("kind,acc_layout_kind,n,k", MMA_PLAIN_KIND_ACC_CASES)
def test_tmem_runtime_matrix_mma_plain_kinds_with_linear_acc(kind, acc_layout_kind, n, k):
    m = 128
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1])
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    acc_layout = _make_tmem_acc_layout(acc_layout_kind, m, n)

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
    assert len(ptx_ops) == _expected_plain_mma_op_count(kind, k)
    assert all(op == expected_kind for op in ptx_ops)
    _assert_exact_commit_ptx_llir_match(
        compiled,
        ["tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"],
    )
    if acc_layout_kind == "linear":
        assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,acc_layout_kind,n,k", MMA_PLAIN_KIND_ACC_CASES)
def test_tmem_runtime_matrix_mma_plain_kinds_use_acc(kind, acc_layout_kind, n, k):
    m = 128
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1])
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    acc_layout = _make_tmem_acc_layout(acc_layout_kind, m, n)

    a, b, shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_plain_kind_inputs(kind, m, n, k)
    c = torch.randn((m, n), device="cuda", dtype=torch.float32)
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)

    compiled = tmem_mma_plain_kind_use_acc_kernel[(1, )](
        a,
        b,
        c,
        out,
        m,
        n,
        k,
        block_layout_a,
        block_layout_b,
        acc_layout,
        shared_layout_a,
        shared_layout_b,
        num_warps=4,
    )

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32)) + c
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)

    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_ops
    assert ptx_ops == llir_ops
    assert len(ptx_ops) == _expected_plain_mma_op_count(kind, k)
    assert all(op == expected_kind for op in ptx_ops)
    _assert_exact_commit_ptx_llir_match(
        compiled,
        ["tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"],
    )
    if acc_layout_kind == "linear":
        assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,n,k,slice_start,use_acc", MMA_M64_ACC_SUBSLICE_CASES)
def test_tmem_runtime_matrix_mma_m64_acc_subslice_view_plain_kinds(kind, n, k, slice_start, use_acc):
    m = 64
    parent_layout = _make_tmem_linear_layout_m64(2 * n)
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1])
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])

    a, b, shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_plain_kind_inputs(kind, m, n, k)
    c = torch.randn((m, n), device="cuda", dtype=torch.float32)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_acc_subslice_kernel[(1, )](
        a,
        b,
        c,
        out,
        m,
        n,
        k,
        parent_layout,
        slice_start,
        block_layout_a,
        block_layout_b,
        shared_layout_a,
        shared_layout_b,
        use_acc,
        num_warps=4,
    )

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    if use_acc:
        ref = ref + c
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert len(mma_ops) == _expected_m64_plain_mma_op_count(kind, k, "linear", n)
    assert all(op == expected_kind for op in mma_ops)
    if use_acc:
        _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_subslice" in ttgir
    assert "tensor_memory_linear" in ttgir
    assert "tensor_memory_encoding" not in ttgir


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra/sm103")
@pytest.mark.parametrize("acc_layout_kind", TMEM_LAYOUT_KIND_CASES)
@pytest.mark.parametrize("n", (32, 64, 128, 256))
@pytest.mark.parametrize("k", (32, 64))
def test_tmem_runtime_matrix_mma_i8_reports_clean_error(acc_layout_kind, n, k, capfd):
    m = 128
    a = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (k, n), device="cuda", dtype=torch.int8)
    out = torch.empty((m, n), device="cuda", dtype=torch.int32)

    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1])
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    shared_layout_a = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=False, element_bitwidth=8, rank=2)
    shared_layout_b = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=True, element_bitwidth=8, rank=2)
    acc_layout = _make_tmem_acc_layout(acc_layout_kind, m, n)

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


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra/sm103")
@pytest.mark.parametrize("acc_layout_kind", ("legacy", "linear"))
@pytest.mark.parametrize("n", (64, 128, 256))
@pytest.mark.parametrize("k", (32, 64))
def test_tmem_runtime_matrix_mma_m64_i8_reports_clean_error(acc_layout_kind, n, k, capfd):
    m = 64
    a = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (k, n), device="cuda", dtype=torch.int8)
    out = torch.empty((m, n), device="cuda", dtype=torch.int32)

    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1])
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    shared_layout_a = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=False, element_bitwidth=8, rank=2)
    shared_layout_b = ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=True, element_bitwidth=8, rank=2)
    acc_layout = (
        TensorMemoryLayout((64, 64), col_stride=1)
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout_m64(n)
    )

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
@pytest.mark.parametrize("kind,acc_layout_kind,n,k,use_acc", MMA_M64_PLAIN_KIND_CASES)
def test_tmem_runtime_matrix_mma_plain_kinds_m64(kind, acc_layout_kind, n, k, use_acc):
    m = 64
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1])
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    acc_layout = (
        TensorMemoryLayout((64, 64), col_stride=1)
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout_m64(n)
    )

    a, b, shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_plain_kind_inputs(kind, m, n, k)
    c = torch.randn((m, n), device="cuda", dtype=torch.float32)
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)

    if use_acc:
        compiled = tmem_mma_plain_kind_use_acc_kernel[(1, )](
            a,
            b,
            c,
            out,
            m,
            n,
            k,
            block_layout_a,
            block_layout_b,
            acc_layout,
            shared_layout_a,
            shared_layout_b,
            num_warps=4,
        )
    else:
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
    if use_acc:
        ref = ref + c
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert len(mma_ops) == _expected_m64_plain_mma_op_count(kind, k, acc_layout_kind, n)
    assert all(op == expected_kind for op in mma_ops)
    if use_acc:
        _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    if acc_layout_kind == "linear":
        assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra/sm103")
@pytest.mark.parametrize("acc_layout_kind", ("legacy", "linear"))
@pytest.mark.parametrize("block_n", (32, 64, 128, 256))
@pytest.mark.parametrize("block_k", (32, 64))
def test_tmem_runtime_matrix_mma_twocta_i8_reports_clean_error(acc_layout_kind, block_n, block_k, capfd):
    ctas_per_cga = [2, 1]
    ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]
    cta_order = [1, 0]
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)
    cga_layout_c_arg = tuple(tuple(basis) for basis in cga_layout_c)

    block_m = 256
    a = torch.randint(-8, 8, (block_m, block_k), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (block_k, block_n), device="cuda", dtype=torch.int8)
    out = torch.empty((block_m, block_n), device="cuda", dtype=torch.int32)

    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1], cga_layout=cga_layout_a)
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0], cga_layout=cga_layout_b)
    shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([block_m, block_k], ttgl.int8, cga_layout=cga_layout_a)
    shared_layout_b = ttgl.NVMMASharedLayout.get_default_for(
        [block_k, block_n], ttgl.int8, transposed=True, cga_layout=cga_layout_b
    )
    acc_layout = (
        TensorMemoryLayout(
            block=(128, block_n // ctas_per_cga[1]),
            col_stride=1,
            two_ctas=True,
            cga_layout=cga_layout_c,
        )
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout_mmav5_twocta(block_m, block_n)
    )

    with pytest.raises(Exception) as excinfo:
        mma_kernel[(1,)](
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
            ttgl.int32,
            False,
            True,
            num_warps=4,
            num_ctas=2,
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
    assert "MMAv5 instruction-tile order requirement" in text
    assert "logical shape 128x128" in text
    assert "Public tcgen05.mma atoms require each 64x8 or larger instruction tile" in text
    assert "first noncanonical in-tile basis" in text
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
    assert "MMAv5 instruction-tile order requirement" in text
    assert "logical shape 128x128" in text
    assert "Public tcgen05.mma atoms require each 64x8 or larger instruction tile" in text
    assert "first noncanonical in-tile basis" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype_name,acc_layout_kind,block_n,block_k,use_acc", MMA_TWOCTA_TMA_NON_TF32_CASES)
def test_tmem_runtime_matrix_mma_twocta(dtype_name, acc_layout_kind, block_n, block_k, use_acc):
    torch_dtype, gluon_dtype, expected_kind, atol, rtol = MMA_TWOCTA_TMA_NON_TF32_DTYPES[dtype_name]
    ctas_per_cga = [2, 1]
    ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    block_m = 128 * ctas_per_cga[0]

    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]
    cta_order = [1, 0]
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)

    shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([block_m, block_k], gluon_dtype, cga_layout=cga_layout_a)
    shared_layout_b = ttgl.NVMMASharedLayout.get_default_for([block_k, block_n], gluon_dtype, cga_layout=cga_layout_b)

    if dtype_name in ("f8e5m2", "f8e4m3"):
        a = torch.randint(20, 40, (block_m, block_k), device="cuda", dtype=torch.uint8).view(torch_dtype)
        b = torch.randint(20, 40, (block_k, block_n), device="cuda", dtype=torch.uint8).view(torch_dtype)
    else:
        a = torch.randn((block_m, block_k), dtype=torch_dtype, device="cuda")
        b = torch.randn((block_k, block_n), dtype=torch_dtype, device="cuda")
    c = torch.randn((block_m, block_n), dtype=torch.float32, device="cuda")
    out = torch.empty((block_m, block_n), dtype=torch.float32, device="cuda")

    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(a, [block_m, block_k], shared_layout_a)
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(b, [block_k, block_n], shared_layout_b)

    if acc_layout_kind == "legacy":
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
    if use_acc:
        compiled = tmem_mma_twocta_use_acc_kernel[(1, )](
            a_desc,
            b_desc,
            c,
            out,
            block_m,
            block_n,
            acc_tmem_layout,
            blocked_c,
            num_warps=4,
            num_ctas=2,
        )
    else:
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

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    if use_acc:
        ref = ref + c
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    ptx_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_mma_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_mma_ops == llir_mma_ops
    assert ptx_mma_ops
    assert len(ptx_mma_ops) == _expected_plain_mma_op_count(dtype_name, block_k)
    assert all(op == expected_kind for op in ptx_mma_ops)
    _assert_exact_commit_ptx_llir_match(
        compiled,
        ["tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"],
    )
    ttgir = compiled.asm["ttgir"]
    assert "two_ctas" in ttgir
    if acc_layout_kind == "linear":
        assert "tensor_memory_linear" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,acc_layout_kind,block_n,block_k", MMA_TWOCTA_PLAIN_KIND_CASES)
def test_tmem_runtime_matrix_mma_twocta_plain_kinds(kind, acc_layout_kind, block_n, block_k):
    ctas_per_cga = [2, 1]
    ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]
    cta_order = [1, 0]
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)
    cga_layout_c_arg = tuple(tuple(basis) for basis in cga_layout_c)
    block_m = 256
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
    assert len(ptx_ops) == _expected_plain_mma_op_count(kind, block_k)
    assert all(op == expected_kind for op in ptx_ops)
    _assert_exact_commit_ptx_llir_match(
        compiled,
        ["tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"],
    )
    assert "two_ctas" in compiled.asm["ttgir"]
    if acc_layout_kind == "linear":
        assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,acc_layout_kind,block_n,block_k", MMA_TWOCTA_PLAIN_KIND_CASES)
def test_tmem_runtime_matrix_mma_twocta_plain_kinds_use_acc(kind, acc_layout_kind, block_n, block_k):
    ctas_per_cga = [2, 1]
    ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]
    cta_order = [1, 0]
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)

    block_m = 256
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1], cga_layout=cga_layout_a)
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0], cga_layout=cga_layout_b)
    block_layout_c = ttgl.BlockedLayout([1, 2], [ctas_per_cga[1], 32 // ctas_per_cga[1]], [4, 1], [1, 0],
                                        cga_layout=cga_layout_c)

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
    c = torch.randn((block_m, block_n), device="cuda", dtype=torch.float32)
    out = torch.empty((block_m, block_n), device="cuda", dtype=torch.float32)

    compiled = tmem_mma_twocta_plain_kind_use_acc_kernel[(1, )](
        a,
        b,
        c,
        out,
        block_m,
        block_n,
        block_k,
        block_layout_a,
        block_layout_b,
        block_layout_c,
        acc_layout,
        shared_layout_a,
        shared_layout_b,
        num_warps=4,
        num_ctas=2,
    )

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32)) + c
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)

    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_ops
    assert ptx_ops == llir_ops
    assert len(ptx_ops) == _expected_plain_mma_op_count(kind, block_k)
    assert all(op == expected_kind for op in ptx_ops)
    _assert_exact_commit_ptx_llir_match(
        compiled,
        ["tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"],
    )
    assert "two_ctas" in compiled.asm["ttgir"]
    if acc_layout_kind == "linear":
        assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,parent_layout_kind,block_n,block_k,use_acc", MMA_TWOCTA_INDEXED_ACC_CASES)
def test_tmem_runtime_matrix_mma_twocta_indexed_acc_view(kind, parent_layout_kind, block_n, block_k, use_acc):
    ctas_per_cga = [2, 1]
    ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]
    cta_order = [1, 0]
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)
    cga_layout_c_arg = tuple(tuple(basis) for basis in cga_layout_c)

    block_m = 256
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1], cga_layout=cga_layout_a)
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0], cga_layout=cga_layout_b)
    block_layout_c = ttgl.BlockedLayout([1, 2], [ctas_per_cga[1], 32 // ctas_per_cga[1]], [4, 1], [1, 0],
                                        cga_layout=cga_layout_c)

    if parent_layout_kind == "legacy":
        parent_layout = TensorMemoryLayout(
            block=(128, block_n // ctas_per_cga[1]),
            col_stride=1,
            two_ctas=True,
            cga_layout=cga_layout_c,
        )
        parent_depth = 2
        parent_index = 1
        layout_token = "tensor_memory_encoding"
    elif parent_layout_kind == "linear":
        parent_layout = _lift_tmem_layout(_make_tmem_linear_layout_mmav5_twocta(block_m, block_n), [2])
        parent_depth = 2
        parent_index = 1
        layout_token = "tensor_memory_linear"
    else:
        assert parent_layout_kind == "linear_unit_parent"
        parent_layout = _lift_tmem_layout(_make_tmem_linear_layout_mmav5_twocta(block_m, block_n), [1])
        parent_depth = 1
        parent_index = 0
        layout_token = "tensor_memory_linear"

    a, b, shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_twocta_plain_kind_inputs(
        kind, block_m, block_n, block_k, cga_layout_a, cga_layout_b
    )
    c = torch.randn((block_m, block_n), device="cuda", dtype=torch.float32)
    out = torch.empty((block_m, block_n), device="cuda", dtype=torch.float32)

    compiled = tmem_mma_twocta_indexed_acc_kernel[(1,)](
        a,
        b,
        c,
        out,
        block_m,
        block_n,
        block_k,
        parent_layout,
        block_layout_a,
        block_layout_b,
        block_layout_c,
        shared_layout_a,
        shared_layout_b,
        use_acc,
        parent_depth,
        parent_index,
        num_warps=4,
        num_ctas=2,
    )

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    if use_acc:
        ref = ref + c
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert len(mma_ops) == _expected_plain_mma_op_count(kind, block_k)
    assert all(op == expected_kind for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(2)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_index" in ttgir
    assert "two_ctas" in ttgir
    assert layout_token in ttgir
    if parent_layout_kind.startswith("linear"):
        assert "tensor_memory_encoding" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,block_n,block_k,slice_start,use_acc", MMA_TWOCTA_ACC_SUBSLICE_CASES)
def test_tmem_runtime_matrix_mma_twocta_acc_subslice_view_plain_kinds(
    kind, block_n, block_k, slice_start, use_acc
):
    ctas_per_cga = [2, 1]
    ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]
    cta_order = [1, 0]
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)

    block_m = 256
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1], cga_layout=cga_layout_a)
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0], cga_layout=cga_layout_b)
    block_layout_c = ttgl.BlockedLayout([1, 2], [ctas_per_cga[1], 32 // ctas_per_cga[1]], [4, 1], [1, 0],
                                        cga_layout=cga_layout_c)
    parent_layout = _make_tmem_linear_layout_mmav5_twocta(block_m, 2 * block_n)

    a, b, shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_twocta_plain_kind_inputs(
        kind, block_m, block_n, block_k, cga_layout_a, cga_layout_b
    )
    c = torch.randn((block_m, block_n), device="cuda", dtype=torch.float32)
    out = torch.empty((block_m, block_n), device="cuda", dtype=torch.float32)

    compiled = tmem_mma_twocta_acc_subslice_kernel[(1,)](
        a,
        b,
        c,
        out,
        block_m,
        block_n,
        block_k,
        parent_layout,
        slice_start,
        block_layout_a,
        block_layout_b,
        block_layout_c,
        shared_layout_a,
        shared_layout_b,
        use_acc,
        num_warps=4,
        num_ctas=2,
    )

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    if use_acc:
        ref = ref + c
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert len(mma_ops) == _expected_plain_mma_op_count(kind, block_k)
    assert all(op == expected_kind for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(2)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_subslice" in ttgir
    assert "tensor_memory_linear" in ttgir
    assert "two_ctas" in ttgir
    assert "tensor_memory_encoding" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("acc_layout_kind,block_n,block_k", MMA_TWOCTA_TMA_TF32_CASES)
def test_tmem_runtime_matrix_mma_twocta_tma_tf32_reports_clean_shared_transpose_error(acc_layout_kind, block_n, block_k, capfd):
    ctas_per_cga = [2, 1]
    ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    block_m = 128 * ctas_per_cga[0]

    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]
    cta_order = [1, 0]
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)

    shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([block_m, block_k], ttgl.float32, cga_layout=cga_layout_a)
    shared_layout_b = ttgl.NVMMASharedLayout.get_default_for([block_k, block_n], ttgl.float32, cga_layout=cga_layout_b)

    a = _round_to_tf32(torch.randn((block_m, block_k), dtype=torch.float32, device="cuda"))
    b = _round_to_tf32(torch.randn((block_k, block_n), dtype=torch.float32, device="cuda"))
    out = torch.empty((block_m, block_n), dtype=torch.float32, device="cuda")

    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(a, [block_m, block_k], shared_layout_a)
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(b, [block_k, block_n], shared_layout_b)

    if acc_layout_kind == "legacy":
        acc_layout = TensorMemoryLayout(
            block=(128, block_n // ctas_per_cga[1]),
            col_stride=1,
            two_ctas=True,
            cga_layout=cga_layout_c,
        )
    else:
        acc_layout = _make_tmem_linear_layout_mmav5_twocta(block_m, block_n)

    blocked_c = ttgl.BlockedLayout([1, 2], [ctas_per_cga[1], 32 // ctas_per_cga[1]], [4, 1], [1, 0],
                                   cga_layout=cga_layout_c)

    with pytest.raises(RuntimeError) as excinfo:
        tmem_mma_twocta_kernel[(1, )](
            a_desc,
            b_desc,
            out,
            block_m,
            block_n,
            acc_layout,
            blocked_c,
            num_warps=4,
            num_ctas=2,
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "tcgen05.mma does not support transposed float32 operands in shared memory" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("acc_layout_kind,block_n,block_k", MMA_TWOCTA_TMA_TF32_CASES)
def test_tmem_runtime_matrix_mma_twocta_tma_tf32_b_transposed_descriptor(acc_layout_kind, block_n, block_k):
    ctas_per_cga = [2, 1]
    block_m = 128 * ctas_per_cga[0]

    cta_split = [ctas_per_cga[0], ctas_per_cga[1]]
    cta_order = [1, 0]
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, 0)
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, 0)
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)

    shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([block_m, block_k], ttgl.float32, cga_layout=cga_layout_a)
    shared_layout_b = ttgl.NVMMASharedLayout.get_default_for([block_n, block_k], ttgl.float32, cga_layout=cga_layout_b)

    a = _round_to_tf32(torch.randn((block_m, block_k), dtype=torch.float32, device="cuda"))
    b = _round_to_tf32(torch.randn((block_n, block_k), dtype=torch.float32, device="cuda"))
    out = torch.empty((block_m, block_n), dtype=torch.float32, device="cuda")

    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(a, [block_m, block_k], shared_layout_a)
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(b, [block_n, block_k], shared_layout_b)

    if acc_layout_kind == "legacy":
        acc_layout = TensorMemoryLayout(
            block=(128, block_n // ctas_per_cga[1]),
            col_stride=1,
            two_ctas=True,
            cga_layout=cga_layout_c,
        )
    else:
        acc_layout = _make_tmem_linear_layout_mmav5_twocta(block_m, block_n)

    blocked_c = ttgl.BlockedLayout([1, 2], [ctas_per_cga[1], 32 // ctas_per_cga[1]], [4, 1], [1, 0],
                                   cga_layout=cga_layout_c)

    compiled = tmem_mma_twocta_tma_b_transposed_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        block_m,
        block_n,
        acc_layout,
        blocked_c,
        num_warps=4,
        num_ctas=2,
    )

    torch.testing.assert_close(out, torch.matmul(a.to(torch.float32), b.to(torch.float32).T), atol=5e-4, rtol=5e-3)

    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_ops
    assert ptx_ops == llir_ops
    assert len(ptx_ops) == _expected_plain_mma_op_count("tf32", block_k)
    assert all(op == "tcgen05.mma.cta_group::2.kind::tf32" for op in ptx_ops)
    _assert_exact_commit_ptx_llir_match(
        compiled,
        ["tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"],
    )
    assert "ttg.memdesc_trans" in compiled.asm["ttgir"]
    if acc_layout_kind == "linear":
        assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("acc_layout_kind,block_n,block_k", MMA_TWOCTA_TMA_TF32_CASES)
def test_tmem_runtime_matrix_mma_twocta_tma_tf32_b_transposed_descriptor_use_acc(acc_layout_kind, block_n, block_k):
    ctas_per_cga = [2, 1]
    block_m = 128 * ctas_per_cga[0]

    cta_split = [ctas_per_cga[0], ctas_per_cga[1]]
    cta_order = [1, 0]
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, 0)
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, 0)
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)

    shared_layout_a = ttgl.NVMMASharedLayout.get_default_for([block_m, block_k], ttgl.float32, cga_layout=cga_layout_a)
    shared_layout_b = ttgl.NVMMASharedLayout.get_default_for([block_n, block_k], ttgl.float32, cga_layout=cga_layout_b)

    a = _round_to_tf32(torch.randn((block_m, block_k), dtype=torch.float32, device="cuda"))
    b = _round_to_tf32(torch.randn((block_n, block_k), dtype=torch.float32, device="cuda"))
    c = torch.randn((block_m, block_n), dtype=torch.float32, device="cuda")
    out = torch.empty((block_m, block_n), dtype=torch.float32, device="cuda")

    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(a, [block_m, block_k], shared_layout_a)
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(b, [block_n, block_k], shared_layout_b)

    if acc_layout_kind == "legacy":
        acc_layout = TensorMemoryLayout(
            block=(128, block_n // ctas_per_cga[1]),
            col_stride=1,
            two_ctas=True,
            cga_layout=cga_layout_c,
        )
    else:
        acc_layout = _make_tmem_linear_layout_mmav5_twocta(block_m, block_n)

    blocked_c = ttgl.BlockedLayout([1, 2], [ctas_per_cga[1], 32 // ctas_per_cga[1]], [4, 1], [1, 0],
                                   cga_layout=cga_layout_c)

    compiled = tmem_mma_twocta_tma_b_transposed_use_acc_kernel[(1, )](
        a_desc,
        b_desc,
        c,
        out,
        block_m,
        block_n,
        acc_layout,
        blocked_c,
        num_warps=4,
        num_ctas=2,
    )

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32).T) + c
    torch.testing.assert_close(out, ref, atol=5e-4, rtol=5e-3)

    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_ops
    assert ptx_ops == llir_ops
    assert len(ptx_ops) == _expected_plain_mma_op_count("tf32", block_k)
    assert all(op == "tcgen05.mma.cta_group::2.kind::tf32" for op in ptx_ops)
    _assert_exact_commit_ptx_llir_match(
        compiled,
        ["tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"],
    )
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_trans" in ttgir
    if acc_layout_kind == "linear":
        assert "tensor_memory_linear" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,parent_layout_kind,n,k,use_acc", MMA_INDEXED_ACC_CASES)
def test_tmem_runtime_matrix_mma_indexed_acc_view(kind, parent_layout_kind, n, k, use_acc):
    m = 128
    if parent_layout_kind == "legacy":
        parent_layout = TensorMemoryLayout((m, n), col_stride=1)
        parent_depth = 2
        parent_index = 1
        layout_token = "tensor_memory_encoding"
    elif parent_layout_kind == "linear":
        parent_layout = _lift_tmem_layout(_make_tmem_linear_layout(m, n), [2])
        parent_depth = 2
        parent_index = 1
        layout_token = "tensor_memory_linear"
    else:
        assert parent_layout_kind == "linear_unit_parent"
        parent_layout = _lift_tmem_layout(_make_tmem_linear_layout(m, n), [1])
        parent_depth = 1
        parent_index = 0
        layout_token = "tensor_memory_linear"
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1])
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])

    a, b, shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_plain_kind_inputs(kind, m, n, k)
    c = torch.randn((m, n), device="cuda", dtype=torch.float32)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_indexed_acc_kernel[(1, )](
        a,
        b,
        c,
        out,
        m,
        n,
        k,
        parent_layout,
        block_layout_a,
        block_layout_b,
        shared_layout_a,
        shared_layout_b,
        use_acc,
        parent_depth,
        parent_index,
        num_warps=4,
    )

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    if use_acc:
        ref = ref + c
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)

    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_ops
    assert ptx_ops == llir_ops
    assert len(ptx_ops) == _expected_plain_mma_op_count(kind, k)
    assert all(op == expected_kind for op in ptx_ops)
    if use_acc:
        _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_index" in ttgir
    assert layout_token in ttgir
    if parent_layout_kind.startswith("linear"):
        assert "tensor_memory_encoding" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,n,k,slice_start,use_acc", MMA_ACC_SUBSLICE_CASES)
def test_tmem_runtime_matrix_mma_acc_subslice_view_plain_kinds(kind, n, k, slice_start, use_acc):
    m = 128
    parent_layout = _make_tmem_linear_layout(m, 2 * n)
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1])
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])

    a, b, shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_plain_kind_inputs(kind, m, n, k)
    c = torch.randn((m, n), device="cuda", dtype=torch.float32)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_acc_subslice_kernel[(1, )](
        a,
        b,
        c,
        out,
        m,
        n,
        k,
        parent_layout,
        slice_start,
        block_layout_a,
        block_layout_b,
        shared_layout_a,
        shared_layout_b,
        use_acc,
        num_warps=4,
    )

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    if use_acc:
        ref = ref + c
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert len(mma_ops) == _expected_plain_mma_op_count(kind, k)
    assert all(op == expected_kind for op in mma_ops)
    if use_acc:
        _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_subslice" in ttgir
    assert "tensor_memory_linear" in ttgir
    assert "tensor_memory_encoding" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n,tile_n,k", MMA_TILE_PERMUTED_CASES)
def test_tmem_runtime_matrix_mma_acc_tile_permuted(n, tile_n, k):
    m = 128
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
    assert len(ptx_ops) == _expected_tile_permuted_mma_op_count("f16", k)
    assert all(op == "tcgen05.mma.cta_group::1.kind::f16" for op in ptx_ops)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,n,tile_n,k", MMA_TILE_PERMUTED_KIND_CASES)
def test_tmem_runtime_matrix_mma_plain_kinds_tile_permuted_acc(kind, n, tile_n, k):
    m = 128
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1])
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    acc_layout = _make_tmem_linear_layout_tile_permuted(m, n, tile_n)

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

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert len(mma_ops) == _expected_tile_permuted_mma_op_count(kind, k)
    assert all(op == expected_kind for op in mma_ops)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,n,tile_n,k", MMA_TILE_PERMUTED_KIND_CASES)
def test_tmem_runtime_matrix_mma_plain_kinds_tile_permuted_acc_use_acc(kind, n, tile_n, k):
    m = 128
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1])
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    acc_layout = _make_tmem_linear_layout_tile_permuted(m, n, tile_n)

    a, b, shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_plain_kind_inputs(kind, m, n, k)
    c = torch.randn((m, n), device="cuda", dtype=torch.float32)
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)

    compiled = tmem_mma_plain_kind_use_acc_kernel[(1, )](
        a,
        b,
        c,
        out,
        m,
        n,
        k,
        block_layout_a,
        block_layout_b,
        acc_layout,
        shared_layout_a,
        shared_layout_b,
        num_warps=4,
    )

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32)) + c
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert len(mma_ops) == _expected_tile_permuted_mma_op_count(kind, k)
    assert all(op == expected_kind for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,n,k,tile_n", MMA_LHS_TILE_PERMUTED_NK_CASES)
def test_tmem_runtime_matrix_mma_lhs_tile_permuted(kind, n, k, tile_n):
    m = 128
    lhs_layout = _make_tmem_linear_layout_tile_permuted(m, k, tile_n)
    acc_layout = _make_tmem_linear_layout(m, n)
    a, b, _shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_plain_kind_inputs(
        kind, m, n, k
    )
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_lhs_kernel[(1, )](
        a, b, out, lhs_layout, acc_layout, shared_layout_b, n, k, 0.0, False, num_warps=4
    )

    expected = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    torch.testing.assert_close(out.to(torch.float32), expected.to(torch.float32), atol=atol, rtol=rtol)

    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    assert ptx_ops == llir_ops
    assert ptx_ops
    assert len(ptx_ops) == _expected_lhs_tile_permuted_mma_op_count(kind, k)
    assert all(op == expected_kind for op in ptx_ops)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,n,k,tile_n", MMA_LHS_TILE_PERMUTED_NK_CASES)
def test_tmem_runtime_matrix_mma_lhs_tile_permuted_use_acc(kind, n, k, tile_n):
    m = 128
    acc_init = 1.0
    lhs_layout = _make_tmem_linear_layout_tile_permuted(m, k, tile_n)
    acc_layout = _make_tmem_linear_layout(m, n)
    a, b, _shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_plain_kind_inputs(
        kind, m, n, k
    )
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_lhs_kernel[(1, )](
        a, b, out, lhs_layout, acc_layout, shared_layout_b, n, k, acc_init, True, num_warps=4
    )

    expected = torch.matmul(a.to(torch.float32), b.to(torch.float32)) + acc_init
    torch.testing.assert_close(out.to(torch.float32), expected.to(torch.float32), atol=atol, rtol=rtol)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert len(mma_ops) == _expected_lhs_tile_permuted_mma_op_count(kind, k)
    assert all(op == expected_kind for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,acc_layout_kind,n,k", MMA_LHS_SUBSLICE_NK_CASES)
def test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds(kind, acc_layout_kind, n, k):
    m = 128
    parent_layout = _make_tmem_linear_layout(m, 2 * k)
    acc_layout = (
        TensorMemoryLayout((m, n), col_stride=1)
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout(m, n)
    )

    a, b, _shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_plain_kind_inputs(
        kind, m, n, k
    )
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_lhs_subslice_kernel[(1, )](
        a,
        b,
        out,
        parent_layout,
        acc_layout,
        shared_layout_b,
        n,
        k,
        0.0,
        False,
        num_warps=4,
    )

    expected = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    torch.testing.assert_close(out.to(torch.float32), expected.to(torch.float32), atol=atol, rtol=rtol)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert len(mma_ops) == _expected_plain_mma_op_count(kind, k)
    assert all(op == expected_kind for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_subslice" in ttgir
    assert "tensor_memory_linear" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("kind,acc_layout_kind,n,k", MMA_LHS_SUBSLICE_NK_CASES)
def test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds_use_acc(kind, acc_layout_kind, n, k):
    m = 128
    acc_init = 1.0
    parent_layout = _make_tmem_linear_layout(m, 2 * k)
    acc_layout = (
        TensorMemoryLayout((m, n), col_stride=1)
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout(m, n)
    )

    a, b, _shared_layout_a, shared_layout_b, expected_kind, atol, rtol = _make_mma_plain_kind_inputs(
        kind, m, n, k
    )
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_lhs_subslice_kernel[(1, )](
        a,
        b,
        out,
        parent_layout,
        acc_layout,
        shared_layout_b,
        n,
        k,
        acc_init,
        True,
        num_warps=4,
    )

    expected = torch.matmul(a.to(torch.float32), b.to(torch.float32)) + acc_init
    torch.testing.assert_close(out.to(torch.float32), expected.to(torch.float32), atol=atol, rtol=rtol)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert len(mma_ops) == _expected_plain_mma_op_count(kind, k)
    assert all(op == expected_kind for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_subslice" in ttgir
    assert "tensor_memory_linear" in ttgir


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

    _assert_exact_mxf8f6f4_scaled_mma(compiled)
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

    _assert_exact_mxf8f6f4_scaled_mma(compiled, expected_count=4)
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

    _assert_exact_mxf8f6f4_scaled_mma(compiled, expected_count=4)
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_scaled_acc_blockm64_reports_backend_error(capfd):
    m = 64
    n = 128
    k = 128
    layout = TensorMemoryLayout((64, 64), col_stride=1)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")
    a = torch.randint(20, 40, (m, k), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    b = torch.randint(20, 40, (k, n), dtype=torch.uint8, device="cuda").view(torch.float8_e5m2)
    a_scale = torch.randint(64, 130, (m, k // 32), dtype=torch.uint8, device="cuda")
    b_scale = torch.randint(64, 130, (n, k // 32), dtype=torch.uint8, device="cuda")

    with pytest.raises(Exception) as excinfo:
        tmem_mma_scaled_layout_kernel[(1, )](out, m, n, k, a, b, a_scale, b_scale, layout, num_warps=4)

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "only supports instruction shape blockM=128" in text
    assert "tcgen05_mma_scaled does not support blockM=64" not in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,n,k,acc_layout_kind", SCALED_MMA_ROOT_FORMAT_CASES)
def test_tmem_runtime_matrix_mma_scaled_root_format_matrix(a_format, b_format, n, k, acc_layout_kind):
    m = 128
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    acc_layout = (
        TensorMemoryLayout((m, n), col_stride=1)
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout(m, n)
    )

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_layout_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        acc_layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        0.0,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    expected_count = (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]
    if acc_layout_kind == "linear":
        assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,n,k,acc_layout_kind", SCALED_MMA_ROOT_USE_ACC_CASES)
def test_tmem_runtime_matrix_mma_scaled_root_format_use_acc(a_format, b_format, n, k, acc_layout_kind):
    m = 128
    acc_init = 1.0
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    acc_layout = (
        TensorMemoryLayout((m, n), col_stride=1)
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout(m, n)
    )

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_layout_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        acc_layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        acc_init,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T + acc_init, atol=1e-3, rtol=1e-3)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    expected_count = (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]
    if acc_layout_kind == "linear":
        assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,n,k,parent_layout_kind", SCALED_MMA_INDEXED_ACC_FORMAT_CASES)
def test_tmem_runtime_matrix_mma_scaled_indexed_acc_view_format_matrix(
    a_format, b_format, n, k, parent_layout_kind
):
    m = 128
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    if parent_layout_kind == "legacy":
        parent_layout = TensorMemoryLayout((m, n), col_stride=1)
        parent_depth = 2
        parent_index = 1
        layout_token = "tensor_memory_encoding"
    elif parent_layout_kind == "legacy_unit_parent":
        parent_layout = TensorMemoryLayout((m, n), col_stride=1)
        parent_depth = 1
        parent_index = 0
        layout_token = "tensor_memory_encoding"
    elif parent_layout_kind == "linear":
        parent_layout = _lift_tmem_layout(_make_tmem_linear_layout(m, n), [2])
        parent_depth = 2
        parent_index = 1
        layout_token = "tensor_memory_linear"
    else:
        assert parent_layout_kind == "linear_unit_parent"
        parent_layout = _lift_tmem_layout(_make_tmem_linear_layout(m, n), [1])
        parent_depth = 1
        parent_index = 0
        layout_token = "tensor_memory_linear"

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_indexed_acc_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        parent_layout,
        parent_depth,
        parent_index,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        0.0,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    expected_count = (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_index" in ttgir
    assert layout_token in ttgir
    if parent_layout_kind.startswith("linear"):
        assert "tensor_memory_encoding" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,n,k,parent_layout_kind", SCALED_MMA_INDEXED_ACC_FORMAT_CASES)
def test_tmem_runtime_matrix_mma_scaled_indexed_acc_view_format_use_acc(
    a_format, b_format, n, k, parent_layout_kind
):
    m = 128
    acc_init = 1.0
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    if parent_layout_kind == "legacy":
        parent_layout = TensorMemoryLayout((m, n), col_stride=1)
        parent_depth = 2
        parent_index = 1
        layout_token = "tensor_memory_encoding"
    elif parent_layout_kind == "legacy_unit_parent":
        parent_layout = TensorMemoryLayout((m, n), col_stride=1)
        parent_depth = 1
        parent_index = 0
        layout_token = "tensor_memory_encoding"
    elif parent_layout_kind == "linear":
        parent_layout = _lift_tmem_layout(_make_tmem_linear_layout(m, n), [2])
        parent_depth = 2
        parent_index = 1
        layout_token = "tensor_memory_linear"
    else:
        assert parent_layout_kind == "linear_unit_parent"
        parent_layout = _lift_tmem_layout(_make_tmem_linear_layout(m, n), [1])
        parent_depth = 1
        parent_index = 0
        layout_token = "tensor_memory_linear"

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_indexed_acc_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        parent_layout,
        parent_depth,
        parent_index,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        acc_init,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T + acc_init, atol=1e-3, rtol=1e-3)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    expected_count = (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_index" in ttgir
    assert layout_token in ttgir
    if parent_layout_kind.startswith("linear"):
        assert "tensor_memory_encoding" not in ttgir


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

    _assert_exact_mxf8f6f4_scaled_mma(compiled, expected_count=4)
    assert "ttg.memdesc_subslice" in compiled.asm["ttgir"]
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format", CP_SCALES_WARPX4_FORMAT_PAIRS)
@pytest.mark.parametrize("n,slice_start,k", SCALED_MMA_ACC_SUBSLICE_N_CASES)
def test_tmem_runtime_matrix_mma_scaled_acc_subslice_view_format_matrix(a_format, b_format, n, slice_start, k):
    m = 128
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_acc_subslice_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        _make_tmem_linear_layout(m, 2 * n),
        slice_start,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        0.0,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    assert "ttg.memdesc_subslice" in compiled.asm["ttgir"]
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,n,slice_start,k", SCALED_MMA_ACC_SUBSLICE_USE_ACC_CASES)
def test_tmem_runtime_matrix_mma_scaled_acc_subslice_view_format_use_acc(a_format, b_format, n, slice_start, k):
    m = 128
    acc_init = 1.0
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_acc_subslice_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        _make_tmem_linear_layout(m, 2 * n),
        slice_start,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        acc_init,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T + acc_init, atol=1e-3, rtol=1e-3)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    expected_count = (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    assert "ttg.memdesc_subslice" in compiled.asm["ttgir"]
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,block_n,slice_start,block_k,multicast", SCALED_MMA_TWOCTA_ACC_SUBSLICE_K_CASES)
def test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_matrix(
    a_format, b_format, block_n, slice_start, block_k, multicast
):
    block_m = 256
    parent_n = 2 * block_n
    vec_size = 16 if a_format == "nvfp4" else 32

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(block_m, block_k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(block_n, block_k, b_format)
    a_scale = swizzle_scales_packed_block(a_scale, vec_size)
    b_scale = swizzle_scales_packed_block(b_scale, vec_size)

    out, compiled = mma_scaled_tcgen05_acc_subslice_copy(
        a,
        b,
        a_scale,
        b_scale,
        vec_size,
        block_m,
        block_n,
        block_k,
        parent_n,
        slice_start,
        num_ctas=2,
        multicast=multicast,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    cp_ops = _assert_exact_cp_ptx_llir_match(compiled)
    assert cp_ops
    assert len(cp_ops) == (1 + max(block_n, 128) // 128) * (block_k // 128) * (32 // vec_size)
    assert all(op == _expected_scaled_cp_opcode(2) for op in cp_ops)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert len(mma_ops) == (block_k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 2) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(2)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_subslice" in ttgir
    assert "tensor_memory_linear" in ttgir
    assert "two_ctas" in ttgir
    if multicast:
        assert "{multicast}" in ttgir
    else:
        assert "{multicast}" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,block_n,slice_start,block_k,multicast", SCALED_MMA_TWOCTA_ACC_SUBSLICE_K_CASES)
def test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_use_acc(
    a_format, b_format, block_n, slice_start, block_k, multicast
):
    block_m = 256
    parent_n = 2 * block_n
    acc_init = 1.0
    vec_size = 16 if a_format == "nvfp4" else 32

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(block_m, block_k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(block_n, block_k, b_format)
    a_scale = swizzle_scales_packed_block(a_scale, vec_size)
    b_scale = swizzle_scales_packed_block(b_scale, vec_size)

    out, compiled = mma_scaled_tcgen05_acc_subslice_copy(
        a,
        b,
        a_scale,
        b_scale,
        vec_size,
        block_m,
        block_n,
        block_k,
        parent_n,
        slice_start,
        num_ctas=2,
        multicast=multicast,
        acc_init=acc_init,
    )

    torch.testing.assert_close(
        out.to(torch.float32),
        a_ref @ b_ref.T + acc_init,
        atol=1e-3,
        rtol=1e-3,
    )

    cp_ops = _assert_exact_cp_ptx_llir_match(compiled)
    assert cp_ops
    assert len(cp_ops) == (1 + max(block_n, 128) // 128) * (block_k // 128) * (32 // vec_size)
    assert all(op == _expected_scaled_cp_opcode(2) for op in cp_ops)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert len(mma_ops) == (block_k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 2) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(2)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_subslice" in ttgir
    assert "tensor_memory_linear" in ttgir
    assert "two_ctas" in ttgir
    if multicast:
        assert "{multicast}" in ttgir
    else:
        assert "{multicast}" not in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("num_ctas,block_m,block_n,parent_n,slice_start,multicast", [
    (1, 128, 128, 256, 0, False),
    (2, 256, 64, 128, 0, False),
])
def test_tmem_runtime_matrix_mma_scaled_shared_scale_descriptor_view_auto_tmem_copy(
    num_ctas, block_m, block_n, parent_n, slice_start, multicast
):
    a_format = "mxfp8"
    b_format = "mxfp8"
    block_k = 128
    vec_size = 32

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(block_m, block_k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(block_n, block_k, b_format)
    a_scale = swizzle_scales_packed_block(a_scale, vec_size)
    b_scale = swizzle_scales_packed_block(b_scale, vec_size)

    out, compiled = mma_scaled_tcgen05_acc_subslice_copy(
        a,
        b,
        a_scale,
        b_scale,
        vec_size,
        block_m,
        block_n,
        block_k,
        parent_n,
        slice_start,
        num_ctas=num_ctas,
        multicast=multicast,
        direct_shared_scales=True,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    cp_ops = _assert_exact_cp_ptx_llir_match(compiled)
    assert cp_ops
    assert len(cp_ops) == 2
    assert all(op == _expected_scaled_cp_opcode(num_ctas) for op in cp_ops)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert mma_ops
    assert len(mma_ops) == _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, num_ctas) for op in mma_ops)
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_reshape" in ttgir
    assert "ttg.memdesc_trans" in ttgir
    if num_ctas == 2:
        assert "two_ctas" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format", CP_SCALES_WARPX4_FORMAT_PAIRS)
@pytest.mark.parametrize("k", (128, 256))
def test_tmem_runtime_matrix_mma_scaled_acc_subslice_tile_permuted_format_matrix(a_format, b_format, k):
    m = 128
    n = 64
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_acc_subslice_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        _make_tmem_linear_layout_tile_permuted(m, 128, 32),
        64,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        0.0,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    expected_count = 2 * (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
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

    _assert_exact_mxf8f6f4_scaled_mma(compiled)
    assert "ttg.memdesc_subslice" in compiled.asm["ttgir"]
    assert "tensor_memory_linear" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,n,k,acc_layout_kind", SCALED_MMA_LHS_SUBSLICE_NK_CASES)
def test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view_format_matrix(
    a_format, b_format, n, k, acc_layout_kind
):
    m = 128
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    parent_layout = _make_tmem_linear_layout(m, 2 * (k // a_elem_per_byte))
    acc_layout = (
        TensorMemoryLayout((m, n), col_stride=1)
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout(m, n)
    )

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_lhs_subslice_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        parent_layout,
        acc_layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        0.0,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    expected_count = (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_subslice" in ttgir
    assert "tensor_memory_linear" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,n,k,acc_layout_kind", SCALED_MMA_LHS_TILE_PERMUTED_NK_CASES)
def test_tmem_runtime_matrix_mma_scaled_lhs_tile_permuted_format_matrix(
    a_format, b_format, n, k, acc_layout_kind
):
    m = 128
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    lhs_storage_k = k // a_elem_per_byte
    lhs_layout = _make_tmem_linear_layout_tile_permuted(m, lhs_storage_k, lhs_storage_k // 4)
    acc_layout = (
        TensorMemoryLayout((m, n), col_stride=1)
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout(m, n)
    )

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_lhs_tile_permuted_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        lhs_layout,
        acc_layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        0.0,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_subslice" not in ttgir
    assert "tensor_memory_linear" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,n,k,acc_layout_kind", SCALED_MMA_LHS_SUBSLICE_NK_CASES)
def test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view_format_use_acc(
    a_format, b_format, n, k, acc_layout_kind
):
    m = 128
    acc_init = 1.0
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    parent_layout = _make_tmem_linear_layout(m, 2 * (k // a_elem_per_byte))
    acc_layout = (
        TensorMemoryLayout((m, n), col_stride=1)
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout(m, n)
    )

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_lhs_subslice_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        parent_layout,
        acc_layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        acc_init,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T + acc_init, atol=1e-3, rtol=1e-3)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    expected_count = (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_subslice" in ttgir
    assert "tensor_memory_linear" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,n,k,acc_layout_kind", SCALED_MMA_LHS_TILE_PERMUTED_NK_CASES)
def test_tmem_runtime_matrix_mma_scaled_lhs_tile_permuted_format_use_acc(
    a_format, b_format, n, k, acc_layout_kind
):
    m = 128
    acc_init = 1.0
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    lhs_storage_k = k // a_elem_per_byte
    lhs_layout = _make_tmem_linear_layout_tile_permuted(m, lhs_storage_k, lhs_storage_k // 4)
    acc_layout = (
        TensorMemoryLayout((m, n), col_stride=1)
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout(m, n)
    )

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_lhs_tile_permuted_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        lhs_layout,
        acc_layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        acc_init,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T + acc_init, atol=1e-3, rtol=1e-3)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_subslice" not in ttgir
    assert "tensor_memory_linear" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n,acc_layout_kind", SCALED_MMA_LHS_TILE_PERMUTED_MIXED_FP4A_UNSUPPORTED_CASES)
def test_tmem_runtime_matrix_mma_scaled_lhs_tile_permuted_mixed_fp4a_reports_clean_unsupported(
    n, acc_layout_kind, capfd
):
    m = 128
    k = 256
    a_format, b_format = "mxfp4", "mxfp8"
    vec_size = 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    lhs_storage_k = k // a_elem_per_byte
    lhs_layout = _make_tmem_linear_layout_tile_permuted(m, lhs_storage_k, lhs_storage_k // 4)
    acc_layout = (
        TensorMemoryLayout((m, n), col_stride=1)
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout(m, n)
    )

    torch.manual_seed(0)
    a, a_scale, _ = random_quantized_tensor(m, k, a_format)
    b, b_scale, _ = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    with pytest.raises(Exception) as excinfo:
        tmem_mma_scaled_lhs_tile_permuted_format_kernel[(1, )](
            out,
            m,
            n,
            k,
            a,
            b,
            a_scale,
            b_scale,
            lhs_layout,
            acc_layout,
            vec_size,
            a_elem_per_byte,
            b_elem_per_byte,
            a_tcgen_format,
            b_tcgen_format,
            0.0,
            num_warps=4,
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "mixed-precision fp4 LHS operands in tensor memory" in text
    assert "raw storage K columns 128" in text
    assert "logical K 256" in text
    assert "each 16-offset group contains only 8 real packed fp4 values" in text
    assert "cannot model that fp4_padded operand-A contract" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n,k,acc_layout_kind", SCALED_MMA_LHS_SUBSLICE_MIXED_FP4A_UNSUPPORTED_CASES)
def test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view_mixed_fp4a_reports_clean_unsupported(
    n, k, acc_layout_kind, capfd
):
    m = 128
    a_format, b_format = "mxfp4", "mxfp8"
    vec_size = 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    parent_layout = _make_tmem_linear_layout(m, 2 * (k // a_elem_per_byte))
    acc_layout = (
        TensorMemoryLayout((m, n), col_stride=1)
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout(m, n)
    )

    torch.manual_seed(0)
    a, a_scale, _ = random_quantized_tensor(m, k, a_format)
    b, b_scale, _ = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    with pytest.raises(Exception) as excinfo:
        tmem_mma_scaled_lhs_subslice_format_kernel[(1, )](
            out,
            m,
            n,
            k,
            a,
            b,
            a_scale,
            b_scale,
            parent_layout,
            acc_layout,
            vec_size,
            a_elem_per_byte,
            b_elem_per_byte,
            a_tcgen_format,
            b_tcgen_format,
            0.0,
            num_warps=4,
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "mixed-precision fp4 LHS operands in tensor memory" in text
    assert f"raw storage K columns {k // a_elem_per_byte}" in text
    assert f"logical K {k}" in text
    assert "each 16-offset group contains only 8 real packed fp4 values" in text
    assert "cannot model that fp4_padded operand-A contract" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,k", SCALED_MMA_ACC_TILE_PERMUTED_K_CASES)
def test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_64_format_matrix(a_format, b_format, k):
    m, n = 128, 256
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    layout = _make_tmem_linear_layout_tile_permuted(m, n, 64)

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_layout_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        0.0,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    expected_count = 4 * (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("scale_side", ["a", "b"])
def test_tmem_runtime_matrix_mma_scaled_linear_scale_descriptor_view(scale_side):
    m = n = k = 128
    a_format = b_format = "mxfp8"
    layout = _make_tmem_linear_layout(m, n)
    vec_size = 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_scale_descriptor_view_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        scale_side,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    expected_count = (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_reshape" in ttgir
    assert "ttg.memdesc_trans" in ttgir
    assert "ttng.tc_gen5_mma_scaled" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "select_mode,selector,n,k",
    [
        ("branch", 0, 128, 128),
        ("branch", 1, 128, 128),
        ("loop", 0, 128, 128),
        ("loop", 1, 128, 128),
        ("loop", 1, 64, 128),
        ("loop", 1, 128, 256),
    ],
)
def test_tmem_runtime_matrix_mma_scaled_dynamic_bscale_direct(select_mode, selector, n, k):
    m = 128
    torch.manual_seed(1515 + selector + n + k + len(select_mode))
    a, a_scale, a_ref = random_quantized_tensor(m, k, "mxfp8")
    b, b_scale0, b_ref0 = random_quantized_tensor(n, k, "mxfp8")
    b_scale1 = torch.randint(64, 130, (n, k // 32), dtype=torch.uint8, device="cuda")
    b_ref1 = b.to(torch.float32) * _fp8e8m0_to_float32(b_scale1).repeat_interleave(32, dim=1)
    b_ref = b_ref1 if select_mode == "loop" or selector else b_ref0
    selector_tensor = torch.tensor([selector], dtype=torch.int32, device="cuda")
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_dynamic_bscale_direct_kernel[(1, )](
        out,
        selector_tensor,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale0,
        b_scale1,
        _make_tmem_linear_layout(m, n),
        select_mode,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == (k // 128) * _expected_scaled_mma_acc_subslice_count("mxfp8", "mxfp8")
    assert all(op == _expected_scaled_mma_opcode("mxfp8", "mxfp8", 1) for op in mma_ops)
    ttgir = compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in ttgir
    assert "arith.select" in ttgir
    if select_mode == "loop":
        assert "scf.for" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_bscale_descriptor_view():
    m = n = k = 128
    a_format = b_format = "mxfp8"
    layout = _make_tmem_linear_layout_tile_permuted(m, n, 32)
    vec_size = 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_bscale_descriptor_view_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        False,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    expected_count = 4 * (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_bscale_view_extra_user_rematerializes():
    m = n = k = 128
    a_format = b_format = "mxfp8"
    layout = _make_tmem_linear_layout_tile_permuted(m, n, 32)
    vec_size = 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_bscale_descriptor_view_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        False,
        True,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    expected_count = 4 * (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    ttgir = compiled.asm["ttgir"]
    assert "ttng.tmem_load" in ttgir
    assert "ttng.tc_gen5_mma_scaled" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_padded_bscale_descriptor_view():
    m = n = k = 128
    a_format = b_format = "mxfp8"
    layout = _make_tmem_linear_layout_tile_permuted(m, n, 32)
    vec_size = 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_bscale_descriptor_view_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        True,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    expected_count = 4 * (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    assert "ttg.memdesc_reshape" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("pad_b_scale_storage", [False, True])
def test_tmem_runtime_matrix_mma_scaled_acc_n16_bscale_descriptor_view_reports_clean_error(
    pad_b_scale_storage, capfd
):
    m, n, k = 128, 16, 128
    a_format = b_format = "mxfp8"
    layout = _make_tmem_linear_layout(m, n)
    vec_size = 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, _ = random_quantized_tensor(m, k, a_format)
    b, b_scale, _ = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    with pytest.raises(CompilationError) as excinfo:
        tmem_mma_scaled_bscale_descriptor_view_format_kernel[(1, )](
            out,
            m,
            n,
            k,
            a,
            b,
            a_scale,
            b_scale,
            layout,
            vec_size,
            a_elem_per_byte,
            b_elem_per_byte,
            a_tcgen_format,
            b_tcgen_format,
            pad_b_scale_storage,
            num_warps=4,
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "TMEM layout 'auto' unsupported for descriptor view" in text
    assert "tensor_memory_descriptor<uint8" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,k", SCALED_MMA_ACC_TILE_PERMUTED_K_CASES)
def test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_64_format_use_acc(a_format, b_format, k):
    m, n = 128, 256
    acc_init = 1.0
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    layout = _make_tmem_linear_layout_tile_permuted(m, n, 64)

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_layout_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        acc_init,
        num_warps=4,
    )

    torch.testing.assert_close(
        out.to(torch.float32),
        a_ref @ b_ref.T + acc_init,
        atol=1e-3,
        rtol=1e-3,
    )

    expected_count = 4 * (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,n,tile_n,k", SCALED_MMA_ACC_TILE_PERMUTED_NARROW_CASES)
def test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_narrow_format_matrix(a_format, b_format, n, tile_n, k):
    m = 128
    layout = _make_tmem_linear_layout_tile_permuted(m, n, tile_n)
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_layout_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        0.0,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    expected_count = 4 * (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,n,k", SCALED_MMA_ACC_IDENTITY_NARROW_CASES)
def test_tmem_runtime_matrix_mma_scaled_acc_identity_narrow_format_matrix(a_format, b_format, n, k):
    m = 128
    layout = _make_tmem_linear_layout(m, n)
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_layout_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        0.0,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    expected_count = (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_scaled_acc_identity_narrow_format_use_acc():
    m, n, k = 128, 16, 128
    a_format = b_format = "mxfp8"
    acc_init = 1.0
    layout = _make_tmem_linear_layout(m, n)
    vec_size = 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_layout_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        acc_init,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T + acc_init, atol=1e-3, rtol=1e-3)

    expected_count = (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,n,k,parent_layout_kind", SCALED_MMA_ACC_IDENTITY_NARROW_VIEW_CASES)
def test_tmem_runtime_matrix_mma_scaled_indexed_acc_identity_narrow_view_format_use_acc(
    a_format, b_format, n, k, parent_layout_kind
):
    m = 128
    acc_init = 1.0
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    if parent_layout_kind == "linear":
        parent_layout = _lift_tmem_layout(_make_tmem_linear_layout(m, n), [2])
        parent_depth = 2
        parent_index = 1
    else:
        assert parent_layout_kind == "linear_unit_parent"
        parent_layout = _lift_tmem_layout(_make_tmem_linear_layout(m, n), [1])
        parent_depth = 1
        parent_index = 0

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_indexed_acc_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        parent_layout,
        parent_depth,
        parent_index,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        acc_init,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T + acc_init, atol=1e-3, rtol=1e-3)

    expected_count = (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    _assert_exact_commit_ptx_llir_match(compiled, [_expected_commit_opcode(1)])
    ttgir = compiled.asm["ttgir"]
    assert "ttg.memdesc_index" in ttgir
    assert "tensor_memory_linear" in ttgir
    assert "tensor_memory_encoding" not in ttgir
    assert "ttng.tc_gen5_mma_scaled" in ttgir


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_runtime_matrix_mma_scaled_acc_n16_tile_permuted_reports_clean_unsupported(capfd):
    m, n, k = 128, 16, 128
    a_format = b_format = "mxfp8"
    layout = _make_tmem_linear_layout_tile_permuted(m, n, 4)
    vec_size = 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, _ = random_quantized_tensor(m, k, a_format)
    b, b_scale, _ = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError) as excinfo:
        tmem_mma_scaled_layout_format_kernel[(1, )](
            out,
            m,
            n,
            k,
            a,
            b,
            a_scale,
            b_scale,
            layout,
            vec_size,
            a_elem_per_byte,
            b_elem_per_byte,
            a_tcgen_format,
            b_tcgen_format,
            0.0,
            num_warps=4,
        )

    captured = capfd.readouterr()
    text = str(excinfo.value) + captured.err + captured.out
    assert "expected accumulator layout to be directly supported MMAv5 block-scaled tensor memory" in text
    assert "logical shape 128x16" in text
    assert "Public tcgen05.mma atoms require each 64x8 or larger instruction tile" in text
    assert "first noncanonical in-tile basis" in text
    assert "PassManager::run failed" not in text
    assert "Assertion" not in text


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("a_format,b_format,k", SCALED_MMA_ACC_TILE_PERMUTED_N32_CASES)
def test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_format_matrix(a_format, b_format, k):
    m = n = 128
    layout = _make_tmem_linear_layout_tile_permuted(m, n, 32)
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")

    compiled = tmem_mma_scaled_layout_format_kernel[(1, )](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        0.0,
        num_warps=4,
    )

    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)

    expected_count = 4 * (k // 128) * _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    mma_ops = _assert_exact_mma_ptx_llir_match(compiled)
    assert len(mma_ops) == expected_count
    assert all(op == _expected_scaled_mma_opcode(a_format, b_format, 1) for op in mma_ops)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]
