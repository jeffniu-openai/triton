import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass

import pytest
import torch
import triton.language as tl

from triton._internal_testing import is_blackwell
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLinearLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    fence_async_shared,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma_scaled,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier
from tmem_test_utils import random_quantized_tensor


def _permute_bits(bits, kind):
    if kind == "identity":
        return list(bits)
    if kind == "reverse":
        return list(reversed(bits))
    if kind == "even_odd":
        return list(bits[::2]) + list(bits[1::2])
    if kind == "rotate1":
        return list(bits[1:]) + [bits[0]]
    raise ValueError(f"unknown permutation kind {kind}")


def _make_linear_layout(m, n, row_kind="identity", col_kind="identity", two_ctas=False):
    row_bits = _permute_bits([1 << i for i in range(int(math.log2(m)))], row_kind)
    col_bits = _permute_bits([1 << i for i in range(int(math.log2(n)))], col_kind)
    kwargs = {
        "rows": [[bit, 0] for bit in row_bits],
        "cols": [[0, bit] for bit in col_bits],
        "shape": [m, n],
    }
    if two_ctas:
        kwargs["rows"] = [[1 << i, 0] for i in range(1, int(math.log2(m)))]
        kwargs["block_bases"] = [[1, 0]]
        kwargs["two_ctas"] = True
    return TensorMemoryLinearLayout(**kwargs)


def _make_permuted_twocta_layout(m, n, row_kind="identity", col_kind="identity"):
    row_bits = _permute_bits([1 << i for i in range(int(math.log2(m)))], row_kind)
    col_bits = _permute_bits([1 << i for i in range(int(math.log2(n)))], col_kind)
    return TensorMemoryLinearLayout(
        rows=[[bit, 0] for bit in row_bits if bit != 1],
        cols=[[0, bit] for bit in col_bits],
        block_bases=[[1, 0]],
        shape=[m, n],
        two_ctas=True,
    )


def _lift_layout(base_layout, prefix_shape):
    prefix_shape = list(prefix_shape)
    prefix_rank = len(prefix_shape)
    total_rank = prefix_rank + len(base_layout.shape)

    def extend(bases):
        return [[0] * prefix_rank + list(basis) for basis in bases]

    rows = extend(base_layout.rows)
    cols = extend(base_layout.cols)
    block_bases = extend(base_layout.block_bases)
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


def _extract_tcgen05_ops(asm, opcodes):
    pattern = re.compile(
        rf"(tcgen05\.(?:{'|'.join(opcodes)})(?:\.red)?\.sync\.aligned\.[^\s;\"]+)"
    )
    return pattern.findall(asm)


def _extract_tcgen05_copy_ops(asm):
    pattern = re.compile(
        r"(tcgen05\.cp(?:\.cta_group::\d+)?(?:\.warpx[24](?:::[^\s.;]+)*)?"
        r"\.\d+x\d+b(?:\.b8x16\.(?:b6x16_p32|b4x16_p64))?)"
    )
    return pattern.findall(asm)


def _extract_tcgen05_mma_ops(asm):
    pattern = re.compile(r"(tcgen05\.mma\.cta_group::\d+\.kind::[^\s;\"]+)")
    return pattern.findall(asm)


def _run_structural_child(case_id):
    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--structural-child", case_id],
        cwd=os.getcwd(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


@dataclass(frozen=True)
class LdStCase:
    case_id: str
    seed: int
    m: int
    n: int
    row_kind: str
    col_kind: str
    instr_variant: str
    chain_id: int
    dtype_name: str = "f32"


@dataclass(frozen=True)
class LdRedCase:
    case_id: str
    seed: int
    m: int
    n: int
    two_ctas: bool
    chain_id: int
    row_kind: str = "identity"
    col_kind: str = "identity"
    op: str = "min"
    expect_hardware_red: bool = True


@dataclass(frozen=True)
class CopyScalesCase:
    case_id: str
    seed: int
    two_ctas: bool


@dataclass(frozen=True)
class DynamicCopyCase:
    case_id: str
    seed: int
    selector: int
    mode: str


@dataclass(frozen=True)
class DynamicLdRedCase:
    case_id: str
    seed: int
    selector: int


@dataclass(frozen=True)
class GenericPassMemdescCase:
    case_id: str
    seed: int
    chain_id: int
    selector: int
    kind: str
    instr_variant: str = "32x32b"
    tuple_capture: bool = False


@dataclass(frozen=True)
class GenericPassLayoutPressureCase:
    case_id: str
    seed: int
    chain_id: int
    instr_variant: str = "16x64b"


@dataclass(frozen=True)
class GenericPassLoopCarriedCase:
    case_id: str
    seed: int
    chain_id: int
    selector: int
    loops: int
    instr_variant: str = "32x32b"


@dataclass(frozen=True)
class ScaledMmaControlFlowCase:
    case_id: str
    seed: int
    n: int
    selector: int
    loop_count: int


@dataclass(frozen=True)
class DynamicIndexLoadOnlyCase:
    case_id: str
    seed: int
    m: int
    n: int
    selector: int


LDST_CASES = [
    LdStCase("ldst-view-identity-32x32b", 0x101, 128, 64, "identity", "identity", "32x32b", 0),
    LdStCase("ldst-view-col-reverse-32x32b", 0x102, 128, 64, "identity", "reverse", "32x32b", 1),
    LdStCase("ldst-view-row-even-16x64b", 0x103, 128, 64, "even_odd", "identity", "16x64b", 2),
    LdStCase("ldst-view-col-rotate-16x128b", 0x104, 128, 128, "identity", "rotate1", "16x128b", 1),
]

LDST_DESCRIPTOR_VIEW_CASES = [
    LdStCase(
        "ldst-fz20260421-0003-chain1-64x32-32x32b",
        0xA003,
        64,
        32,
        "identity",
        "identity",
        "32x32b",
        1,
    ),
    LdStCase(
        "ldst-fz20260421-0003-chain2-col-reverse-64x32-16x64b",
        0xA013,
        64,
        32,
        "identity",
        "reverse",
        "16x64b",
        2,
    ),
    LdStCase(
        "ldst-fz20260421-0003-f16-chain2-identity-64x32-16x64b",
        0xA023,
        64,
        32,
        "identity",
        "identity",
        "16x64b",
        2,
        "f16",
    ),
]

LDRED_CASES = [
    LdRedCase("ldred-direct-128x64", 0x201, 128, 64, False, 0),
    LdRedCase("ldred-view-128x64", 0x202, 128, 64, False, 1),
    LdRedCase("ldred-twocta-lifted-256x64", 0x203, 256, 64, True, 0),
    LdRedCase("ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min", 0xA014, 256, 32, True, 1),
    LdRedCase("ldred-fz20260421-0004-twocta-indexed-256x32-chain0-max", 0xA024, 256, 32, True, 1, op="max"),
    LdRedCase("ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min-abs", 0xA025, 256, 32, True, 1, op="min_abs"),
    LdRedCase("ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min-nan", 0xA026, 256, 32, True, 1, op="min_nan"),
    LdRedCase("ldred-fz20260421-0004-chain1-64x32-min", 0xA004, 64, 32, False, 2),
    LdRedCase(
        "ldred-fz20260421-0006-rotate1-transpose-slice-max",
        0xA006,
        64,
        128,
        False,
        3,
        "rotate1",
        "identity",
        op="max",
        expect_hardware_red=False,
    ),
]

COPY_CASES = [
    CopyScalesCase("copy-scales-warpx4-1cta", 0x301, False),
    CopyScalesCase("copy-scales-warpx4-2cta", 0x302, True),
]

DYNAMIC_COPY_CASES = [
    DynamicCopyCase("copy-dynamic-branch-index-128x4", 0x560002, 1, "branch_index"),
    DynamicCopyCase("copy-dynamic-runtime-index-128x4", 0x560006, 1, "runtime_index"),
]

DYNAMIC_LDRED_CASES = [
    DynamicLdRedCase("ldred-dynamic-runtime-index-128x32", 0x260001, 1),
]

GENERIC_PASS_MEMDESC_CASES = [
    GenericPassMemdescCase("generic-pass-dynamic-index-chain0", 0xE001, 0, 1, "dynamic_index"),
    GenericPassMemdescCase("generic-pass-dynamic-index-chain1", 0xE011, 1, 0, "dynamic_index"),
    GenericPassMemdescCase("generic-pass-dynamic-if-chain0-true", 0xE001, 0, 1, "dynamic_if"),
    GenericPassMemdescCase("generic-pass-dynamic-if-chain0-false-16x128b", 0xE022, 0, 0, "dynamic_if", "16x128b"),
    GenericPassMemdescCase("generic-pass-dynamic-if-chain0-inline", 0xE023, 0, 0, "dynamic_if_inline"),
    GenericPassMemdescCase("generic-pass-mixed-captures-chain0", 0xE001, 0, 1, "mixed_captures"),
    GenericPassMemdescCase(
        "generic-pass-tuple-mixed-captures-chain0", 0xE024, 0, 0, "mixed_captures", tuple_capture=True
    ),
]

GENERIC_PASS_LAYOUT_PRESSURE_CASES = [
    GenericPassLayoutPressureCase("generic-pass-layout-conversion-pressure-chain0", 0xE100, 0),
    GenericPassLayoutPressureCase("generic-pass-layout-conversion-pressure-chain0-16x128b", 0xE128, 0, "16x128b"),
]

GENERIC_PASS_LOOP_CARRIED_CASES = [
    GenericPassLoopCarriedCase("generic-pass-loop-carried-memdesc-view-chain0", 0x5C01, 0, 1, 2),
]

SCALED_MMA_CONTROL_FLOW_CASES = [
    ScaledMmaControlFlowCase("mma-scaled-fz20260421-0007-subslice-if-n64-selector0", 0x6B00, 64, 0, 2),
]

DYNAMIC_INDEX_LOAD_ONLY_CASES = [
    DynamicIndexLoadOnlyCase("generic-pass-dynamic-index-load-only-128x32", 0xE021, 128, 32, 1),
]


@gluon.jit
def _generic_pass_identity_view0(desc, M: ttgl.constexpr, N: ttgl.constexpr):
    return desc.reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))


@gluon.jit
def _generic_pass_identity_view1(desc, M: ttgl.constexpr, N: ttgl.constexpr):
    view = desc.reshape((M // 2, 2, N // 2, 2))
    return view.permute([1, 0, 3, 2]).permute([1, 0, 3, 2]).reshape((M, N))


@gluon.jit
def _generic_pass_view(desc, M: ttgl.constexpr, N: ttgl.constexpr, chain_id: ttgl.constexpr):
    if chain_id == 0:
        return _generic_pass_identity_view0(desc, M, N)
    elif chain_id == 1:
        return _generic_pass_identity_view1(desc, M, N)
    else:
        return desc.slice(0, M, dim=0).slice(0, N, dim=1)


def _generic_pass_expected_view(value, chain_id):
    m, n = value.shape
    if chain_id == 0:
        return value.reshape(m // 2, 2, n).permute(1, 0, 2).reshape(m, n)
    if chain_id == 1:
        return (
            value.reshape(m // 2, 2, n // 2, 2)
            .permute(1, 0, 3, 2)
            .permute(1, 0, 3, 2)
            .reshape(m, n)
        )
    return value


@gluon.jit
def _fuzz_generic_pass_dynamic_index_kernel(
    in_ptr,
    out_ptr,
    selector_ptr,
    parent_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    chain_id: ttgl.constexpr,
):
    parent = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
    layout: ttgl.constexpr = parent.index(0).get_reg_layout(instr_variant="32x32b")
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)
    parent.index(0).store(ttgl.convert_layout(value + 10.0, layout))
    parent.index(1).store(ttgl.convert_layout(value + 20.0, layout))

    index = ttgl.load(selector_ptr)
    view = _generic_pass_view(parent.index(index), M, N, chain_id)
    view_layout: ttgl.constexpr = view.get_reg_layout(instr_variant="32x32b")
    out = view.load(view_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, view_layout))


@gluon.jit
def _fuzz_generic_pass_dynamic_index_load_only_kernel(
    out_ptr,
    selector_ptr,
    parent_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
):
    parent = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    index = ttgl.load(selector_ptr)
    view = parent.index(index)
    view_layout: ttgl.constexpr = view.get_reg_layout(instr_variant="32x32b")
    out = view.load(view_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, view_layout))


@gluon.jit
def _fuzz_generic_pass_dynamic_if_kernel(
    in_ptr,
    out_ptr,
    selector_ptr,
    parent_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    chain_id: ttgl.constexpr,
    instr_variant: ttgl.constexpr,
):
    parent = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
    layout: ttgl.constexpr = parent.index(0).get_reg_layout(instr_variant=instr_variant)
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)
    parent.index(0).store(ttgl.convert_layout(value + 30.0, layout))
    parent.index(1).store(ttgl.convert_layout(value + 40.0, layout))

    if ttgl.load(selector_ptr) != 0:
        view = _generic_pass_view(parent.index(0), M, N, chain_id)
    else:
        view = _generic_pass_view(parent.index(1), M, N, chain_id)
    view_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    out = view.load(view_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, view_layout))


@gluon.jit
def _fuzz_generic_pass_dynamic_if_inline_chain0_kernel(
    in_ptr,
    out_ptr,
    selector_ptr,
    parent_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    instr_variant: ttgl.constexpr,
):
    parent = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
    layout: ttgl.constexpr = parent.index(0).get_reg_layout(instr_variant=instr_variant)
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)
    parent.index(0).store(ttgl.convert_layout(value + 30.0, layout))
    parent.index(1).store(ttgl.convert_layout(value + 40.0, layout))

    if ttgl.load(selector_ptr) != 0:
        view = parent.index(0).reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
    else:
        view = parent.index(1).reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
    view_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    out = view.load(view_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, view_layout))


@gluon.jit
def _generic_pass_tuple_pair(desc, bias):
    return desc, bias


@gluon.jit
def _fuzz_generic_pass_mixed_captures_kernel(
    in_ptr,
    out_ptr,
    selector_ptr,
    parent_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    chain_id: ttgl.constexpr,
    instr_variant: ttgl.constexpr,
    tuple_capture: ttgl.constexpr,
):
    parent = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
    layout: ttgl.constexpr = parent.index(0).get_reg_layout(instr_variant=instr_variant)
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)
    parent.index(0).store(ttgl.convert_layout(value, layout))
    parent.index(1).store(ttgl.convert_layout(value * 2.0, layout))

    if ttgl.load(selector_ptr) != 0:
        view = _generic_pass_view(parent.index(0), M, N, chain_id)
        bias = value + 3.0
    else:
        view = _generic_pass_view(parent.index(1), M, N, chain_id)
        bias = value + 5.0
    if tuple_capture:
        pair = _generic_pass_tuple_pair(view, bias)
        view = pair[0]
        bias = pair[1]
    view_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    out = view.load(view_layout) + ttgl.convert_layout(bias, view_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, view_layout))


@gluon.jit
def _fuzz_generic_pass_layout_pressure_kernel(
    in_ptr,
    out_ptr,
    parent_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    chain_id: ttgl.constexpr,
    instr_variant: ttgl.constexpr,
):
    parent = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
    base = parent.index(1)
    view = _generic_pass_view(base, M, N, chain_id)
    layout_a: ttgl.constexpr = base.get_reg_layout(instr_variant="32x32b")
    layout_b: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    layout_c: ttgl.constexpr = view.get_reg_layout(instr_variant="32x32b")
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)
    base.store(ttgl.convert_layout(value, layout_a))
    loaded = view.load(layout_b)
    out = ttgl.convert_layout(ttgl.convert_layout(loaded, layout_c), layout_b)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, layout_b))


@gluon.jit
def _fuzz_generic_pass_loop_carried_kernel(
    in_ptr,
    out_ptr,
    selector_ptr,
    loop_count_ptr,
    parent_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    chain_id: ttgl.constexpr,
    instr_variant: ttgl.constexpr,
):
    parent = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
    layout: ttgl.constexpr = parent.index(0).get_reg_layout(instr_variant=instr_variant)
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)
    parent.index(0).store(ttgl.convert_layout(value + 10.0, layout))
    parent.index(1).store(ttgl.convert_layout(value + 20.0, layout))

    view = _generic_pass_view(parent.index(0), M, N, chain_id)
    selector = ttgl.load(selector_ptr)
    for i in range(0, ttgl.load(loop_count_ptr), 1):
        if i == selector:
            view = _generic_pass_view(parent.index(1), M, N, chain_id)
        else:
            view = view

    view_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    out = view.load(view_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, view_layout))


@gluon.jit
def _fuzz_scaled_mma_acc_subslice_if_kernel(
    out_ptr,
    selector_ptr,
    loop_count_ptr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
    a,
    b,
    a_scale,
    b_scale,
    acc_layout: ttgl.constexpr,
):
    VEC_SIZE: ttgl.constexpr = 32
    A_FORMAT: ttgl.constexpr = "e4m3"
    B_FORMAT: ttgl.constexpr = "e4m3"

    reg_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [32, 1], [ttgl.num_warps(), 1], [1, 0])
    smem_layout_a: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([M, K], a.dtype.element_ty)
    smem_layout_b: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([N, K], b.dtype.element_ty)
    block_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [ttgl.num_warps(), 1], [1, 0])

    a_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, block_layout))[:, None]
    a_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, block_layout))[None, :]
    b_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, block_layout))[:, None]
    b_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, block_layout))[None, :]
    a_tile = ttgl.load(a + a_offs_m * K + a_offs_k)
    b_tile = ttgl.load(b + b_offs_n * K + b_offs_k)
    a_smem = ttgl.allocate_shared_memory(a.dtype.element_ty, [M, K], smem_layout_a, a_tile)
    b_smem = ttgl.allocate_shared_memory(b.dtype.element_ty, [N, K], smem_layout_b, b_tile)

    acc_parent = allocate_tensor_memory(ttgl.float32, [M, 2 * N], acc_layout)
    view0 = acc_parent.slice(0, N, dim=1)
    view1 = acc_parent.slice(N, N, dim=1)
    layout0: ttgl.constexpr = view0.get_reg_layout()
    layout1: ttgl.constexpr = view1.get_reg_layout()
    view0.store(ttgl.full([M, N], 3.0, ttgl.float32, layout=layout0))
    view1.store(ttgl.full([M, N], 7.0, ttgl.float32, layout=layout1))

    if ttgl.load(selector_ptr) != 0:
        acc_tmem = view1
    else:
        acc_tmem = view0

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
    ttgl.store(out_ptr + offs_m * N + offs_n, ttgl.convert_layout(out_reg, reg_layout))


@gluon.jit
def _fuzz_ldst_view_kernel(
    in_ptr,
    out_ptr,
    parent_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    instr_variant: ttgl.constexpr,
    chain_id: ttgl.constexpr,
):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [2, M, N], parent_layout)
    base = tmem.index(1)
    base_layout: ttgl.constexpr = base.get_reg_layout(instr_variant=instr_variant)
    value = ttgl.load(in_ptr + offs)
    base.store(ttgl.convert_layout(value, base_layout))

    view = base
    if chain_id == 0:
        view = view.reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
    elif chain_id == 1:
        view = view.reshape((M // 2, 2, N // 2, 2))
        view = view.permute([1, 0, 3, 2]).permute([1, 0, 3, 2]).reshape((M, N))
    elif chain_id == 2:
        view = view.permute([1, 0]).permute([1, 0]).slice(0, M, dim=0).slice(0, N, dim=1)
    else:
        view = base

    view_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    out = view.load(view_layout)
    view.store(out)
    result = base.load(base_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(result, base_layout))


@gluon.jit
def _fuzz_ldst_descriptor_view_read_kernel(
    in_ptr,
    out_ptr,
    parent_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    instr_variant: ttgl.constexpr,
    chain_id: ttgl.constexpr,
):
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [2, M, N], parent_layout)
    view = tmem.index(1)
    reg_layout: ttgl.constexpr = view.get_reg_layout(instr_variant=instr_variant)
    value = ttgl.load(in_ptr + offs)
    view.store(ttgl.convert_layout(value, reg_layout))

    if chain_id == 1:
        reread = view.reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
    elif chain_id == 2:
        reread = view.reshape((M, N // 2, 2)).permute([0, 2, 1]).permute([0, 2, 1]).reshape((M, N))
    elif chain_id == 3:
        reread = view.permute([1, 0]).slice(0, N, dim=0).permute([1, 0]).slice(0, M, dim=0)
    else:
        reread = view

    out_layout: ttgl.constexpr = reread.get_reg_layout(instr_variant=instr_variant)
    out = reread.load(out_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, out_layout))


@gluon.jit
def _fuzz_ldred_kernel(
    in_ptr,
    out_ptr,
    red_ptr,
    layout: ttgl.constexpr,
    parent_layout: ttgl.constexpr,
    red_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    chain_id: ttgl.constexpr,
    use_max: ttgl.constexpr,
):
    if chain_id == 0:
        tmem = allocate_tensor_memory(ttgl.float32, [M, N], layout)
        view = tmem
    else:
        parent = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
        view = parent.index(1)
        if chain_id == 2:
            view = view.reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
        elif chain_id == 3:
            view = view.permute([1, 0]).permute([1, 0]).slice(0, M, dim=0).slice(0, N, dim=1)

    reg_layout: ttgl.constexpr = view.get_reg_layout()
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))[:, None]
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))[None, :]
    offs = offs_m * N + offs_n
    value = ttgl.load(in_ptr + offs)
    view.store(ttgl.convert_layout(value, reg_layout))

    if use_max:
        out, reduced = view.load_max()
    else:
        out, reduced = view.load_min()
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, reg_layout))
    red_m = ttgl.arange(0, M, red_layout)
    ttgl.store(red_ptr + red_m, ttgl.convert_layout(reduced, red_layout))


@gluon.jit
def _fuzz_ldred_twocta_kernel(in_ptr, out_ptr, red_ptr, parent_layout: ttgl.constexpr, red_layout: ttgl.constexpr):
    M: ttgl.constexpr = 256
    N: ttgl.constexpr = 64
    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
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
def _fuzz_ldred_twocta_indexed_kernel(
    in_ptr,
    out_ptr,
    red_ptr,
    parent_layout: ttgl.constexpr,
    red_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    chain_id: ttgl.constexpr,
    red_op: ttgl.constexpr,
    use_abs: ttgl.constexpr,
    propagate_nan: ttgl.constexpr,
):
    parent = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
    view = parent.index(1)
    if chain_id == 2:
        view = view.reshape((M, N // 2, 2)).permute([0, 2, 1]).permute([0, 2, 1]).reshape((M, N))
    reg_layout: ttgl.constexpr = view.get_reg_layout()
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))[:, None]
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))[None, :]
    offs = offs_m * N + offs_n
    value = ttgl.load(in_ptr + offs)
    view.store(ttgl.convert_layout(value, reg_layout))
    if red_op == "max":
        out, reduced = view.load_max(abs=use_abs, propagate_nan=propagate_nan)
    else:
        out, reduced = view.load_min(abs=use_abs, propagate_nan=propagate_nan)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, reg_layout))
    red_m = ttgl.arange(0, M, red_layout)
    ttgl.store(red_ptr + red_m, ttgl.convert_layout(reduced, red_layout))


@gluon.jit
def _fuzz_ldred_twocta_indexed_row_chain_kernel(
    in_ptr,
    out_ptr,
    red_ptr,
    parent_layout: ttgl.constexpr,
    red_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
):
    parent = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
    view = parent.index(1)
    view = view.reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
    reg_layout: ttgl.constexpr = view.get_reg_layout()
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))[:, None]
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))[None, :]
    offs = offs_m * N + offs_n
    value = ttgl.load(in_ptr + offs)
    view.store(ttgl.convert_layout(value, reg_layout))
    out, reduced = view.load_min()
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, reg_layout))
    red_m = ttgl.arange(0, M, red_layout)
    ttgl.store(red_ptr + red_m, ttgl.convert_layout(reduced, red_layout))


@gluon.jit
def _fuzz_copy_scales_kernel(in_ptr, out_ptr, TWO_CTAS: ttgl.constexpr):
    M: ttgl.constexpr = 128 if TWO_CTAS else 64
    N: ttgl.constexpr = 16
    if TWO_CTAS:
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0], cga_layout=[[1, 0]])
    else:
        blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, blocked))
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, blocked))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    if TWO_CTAS:
        smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
            offset_bases=[[0, 1], [0, 2], [32, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]],
            block_bases=[[64, 0]],
        )
        tmem = allocate_tensor_memory(ttgl.int8, (M, N), layout=TensorMemoryScalesLayout(cga_layout=[[1, 0]]))
    else:
        smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
            offset_bases=[[0, 1], [0, 2], [32, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]]
        )
        tmem = allocate_tensor_memory(ttgl.int8, (M, N), layout=TensorMemoryScalesLayout())
    smem = ttgl.allocate_shared_memory(ttgl.int8, (M, N), layout=smem_layout)
    smem.store(value)
    fence_async_shared(cluster=TWO_CTAS)

    barrier = mbarrier.allocate_mbarrier()
    mbarrier.init(barrier, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    reg_layout: ttgl.constexpr = tmem.get_reg_layout()
    output = tmem.load(reg_layout)
    out_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, reg_layout))
    out_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, reg_layout))
    out_offs = out_m[:, None] * N + out_n[None, :]
    ttgl.store(out_ptr + out_offs, output)


@gluon.jit
def _fuzz_copy_dynamic_index_kernel(
    in_ptr,
    out_ptr,
    selector_ptr,
    parent_layout: ttgl.constexpr,
    mode: ttgl.constexpr,
):
    M: ttgl.constexpr = 128
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
    smem.store(value)
    fence_async_shared()
    parent = allocate_tensor_memory(in_ptr.dtype.element_ty, [2, M, N], layout=parent_layout)
    if mode == "runtime_index":
        view = parent.index(ttgl.load(selector_ptr))
    elif ttgl.load(selector_ptr) != 0:
        view = parent.index(0)
    else:
        view = parent.index(1)
    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    tcgen05_copy(smem, view)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)
    reg_layout: ttgl.constexpr = view.get_reg_layout()
    output = view.load(reg_layout)
    ttgl.store(out_ptr + offs, ttgl.convert_layout(output, blocked))


@gluon.jit
def _fuzz_ldred_dynamic_index_kernel(
    in_ptr,
    out_ptr,
    red_ptr,
    selector_ptr,
    parent_layout: ttgl.constexpr,
    red_layout: ttgl.constexpr,
):
    M: ttgl.constexpr = 128
    N: ttgl.constexpr = 32
    parent = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
    reg_layout: ttgl.constexpr = parent.index(0).get_reg_layout(instr_variant="32x32b")
    offs = ttgl.arange(0, M)[:, None] * N + ttgl.arange(0, N)[None, :]
    value = ttgl.load(in_ptr + offs)
    parent.index(0).store(ttgl.convert_layout(value + 1.0, reg_layout))
    parent.index(1).store(ttgl.convert_layout(value + 2.0, reg_layout))
    view = parent.index(ttgl.load(selector_ptr))
    out, reduced = view.load_min()
    ttgl.store(out_ptr + offs, ttgl.convert_layout(out, reg_layout))
    red_m = ttgl.arange(0, M, red_layout)
    ttgl.store(red_ptr + red_m, ttgl.convert_layout(reduced, red_layout))


def _run_ldst_case(case):
    torch.manual_seed(case.seed)
    layout = _make_linear_layout(case.m, case.n, case.row_kind, case.col_kind)
    parent_layout = _lift_layout(layout, [2])
    inp = torch.randn((case.m, case.n), dtype=torch.float32, device="cuda")
    out = torch.empty_like(inp)
    compiled = _fuzz_ldst_view_kernel[(1, )](
        inp, out, parent_layout, case.m, case.n, case.instr_variant, case.chain_id, num_warps=4
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    ptx_ops = _extract_tcgen05_ops(compiled.asm["ptx"], ("ld", "st"))
    llir_ops = _extract_tcgen05_ops(compiled.asm["llir"], ("ld", "st"))
    assert ptx_ops == llir_ops
    assert any(".ld." in op for op in ptx_ops)
    assert any(".st." in op for op in ptx_ops)
    return compiled


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("case", LDST_CASES, ids=lambda case: case.case_id)
def test_tmem_structural_fuzzer_ldst_view_roundtrip(case):
    _run_ldst_case(case)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("case", LDST_DESCRIPTOR_VIEW_CASES, ids=lambda case: case.case_id)
def test_tmem_structural_fuzzer_ldst_descriptor_view_read(case):
    torch.manual_seed(case.seed)
    layout = _make_linear_layout(case.m, case.n, case.row_kind, case.col_kind)
    parent_layout = _lift_layout(layout, [2])
    torch_dtype = {"f32": torch.float32, "f16": torch.float16}[case.dtype_name]
    inp = torch.randn((case.m, case.n), dtype=torch_dtype, device="cuda")
    out = torch.empty_like(inp)
    compiled = _fuzz_ldst_descriptor_view_read_kernel[(1, )](
        inp, out, parent_layout, case.m, case.n, case.instr_variant, case.chain_id, num_warps=4
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    ptx_ops = _extract_tcgen05_ops(compiled.asm["ptx"], ("ld", "st"))
    llir_ops = _extract_tcgen05_ops(compiled.asm["llir"], ("ld", "st"))
    assert ptx_ops == llir_ops
    assert any(".ld." in op for op in ptx_ops)
    assert any(".st." in op for op in ptx_ops)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_structural_fuzzer_ldst_256row_lifted_parent_allocator_crash():
    _run_structural_child("ldst_256row_lifted_parent_allocator_crash")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("case", LDRED_CASES, ids=lambda case: case.case_id)
def test_tmem_structural_fuzzer_ldred(case):
    torch.manual_seed(case.seed)
    red_op = "max" if case.op == "max" else "min"
    use_abs = case.op.endswith("_abs")
    propagate_nan = tl.PropagateNan.ALL if case.op.endswith("_nan") else tl.PropagateNan.NONE
    if case.two_ctas:
        layout = _make_linear_layout(case.m, case.n, two_ctas=True)
        parent_layout = _lift_layout(layout, [2])
        red_layout = ttgl.BlockedLayout([1, 1], [1, 32], [1, 8], [1, 0], cga_layout=[[1, 0]])
        if case.chain_id == 0:
            inp = torch.randn((2, case.m, case.n), dtype=torch.float32, device="cuda")
            out = torch.empty_like(inp)
            red = torch.empty((2, case.m), dtype=torch.float32, device="cuda")
            compiled = _fuzz_ldred_twocta_kernel[(1, )](
                inp, out, red, parent_layout, red_layout, num_warps=8, num_ctas=2
            )
            torch.testing.assert_close(out, inp, atol=0, rtol=0)
            torch.testing.assert_close(red, torch.min(inp, dim=2).values, atol=0, rtol=0)
        else:
            inp = torch.randn((case.m, case.n), dtype=torch.float32, device="cuda")
            if propagate_nan == tl.PropagateNan.ALL:
                inp[10, min(5, case.n - 1)] = float("nan")
                inp[50, min(15, case.n - 1)] = float("nan")
            out = torch.empty_like(inp)
            red = torch.empty((case.m, ), dtype=torch.float32, device="cuda")
            indexed_red_layout = ttgl.BlockedLayout([1], [32], [8], [0], cga_layout=[[1]])
            compiled = _fuzz_ldred_twocta_indexed_kernel[(1, )](
                inp,
                out,
                red,
                parent_layout,
                indexed_red_layout,
                case.m,
                case.n,
                case.chain_id - 1,
                red_op,
                use_abs,
                propagate_nan,
                num_warps=8,
                num_ctas=2,
            )
            torch.testing.assert_close(out, inp, atol=0, rtol=0, equal_nan=propagate_nan == tl.PropagateNan.ALL)
            ref_input = torch.abs(inp) if use_abs else inp
            expected_red = getattr(torch, red_op)(ref_input, dim=1).values
            torch.testing.assert_close(red, expected_red, atol=1e-5, rtol=1e-5, equal_nan=propagate_nan == tl.PropagateNan.ALL)
    else:
        layout = _make_linear_layout(case.m, case.n, case.row_kind, case.col_kind)
        parent_layout = _lift_layout(layout, [2])
        red_layout = ttgl.BlockedLayout([1], [32], [4], [0])
        inp = torch.randn((case.m, case.n), dtype=torch.float32, device="cuda")
        out = torch.empty_like(inp)
        red = torch.empty((case.m, ), dtype=torch.float32, device="cuda")
        compiled = _fuzz_ldred_kernel[(1, )](
            inp,
            out,
            red,
            layout,
            parent_layout,
            red_layout,
            case.m,
            case.n,
            case.chain_id,
            case.op == "max",
            num_warps=4,
        )
        torch.testing.assert_close(out, inp, atol=0, rtol=0)
        if case.op == "max":
            expected_red = torch.max(inp, dim=1).values
        else:
            expected_red = torch.min(inp, dim=1).values
        torch.testing.assert_close(red, expected_red, atol=0, rtol=0)
    ptx_ops = _extract_tcgen05_ops(compiled.asm["ptx"], ("ld", ))
    llir_ops = _extract_tcgen05_ops(compiled.asm["llir"], ("ld", ))
    assert ptx_ops == llir_ops
    if case.expect_hardware_red:
        assert any(".ld.red." in op for op in ptx_ops)
    else:
        assert any(".ld." in op for op in ptx_ops)
        assert not any(".ld.red." in op for op in ptx_ops)
        assert "tt.reduce" in compiled.asm["ttgir"]


def _run_ldred_twocta_rowcol_optimizer_crash_case():
    m = 256
    n = 2
    torch.manual_seed(0x6C00 + m + n + 1)
    layout = _make_permuted_twocta_layout(m, n, "even_odd", "identity")
    parent_layout = _lift_layout(layout, [2])
    red_layout = ttgl.BlockedLayout([1], [32], [8], [0], cga_layout=[[1]])
    inp = torch.randn((m, n), dtype=torch.float32, device="cuda")
    out = torch.empty_like(inp)
    red = torch.empty((m, ), dtype=torch.float32, device="cuda")
    compiled = _fuzz_ldred_twocta_indexed_row_chain_kernel[(1, )](
        inp, out, red, parent_layout, red_layout, m, n, num_warps=8, num_ctas=2
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    torch.testing.assert_close(red, torch.min(inp, dim=1).values, atol=1e-5, rtol=1e-5)
    ptx_ops = _extract_tcgen05_ops(compiled.asm["ptx"], ("ld", ))
    llir_ops = _extract_tcgen05_ops(compiled.asm["llir"], ("ld", ))
    assert ptx_ops == llir_ops
    assert any(".ld." in op for op in ptx_ops)
    assert not any(".ld.red." in op for op in ptx_ops)
    assert "tt.reduce" in compiled.asm["ttgir"]


def _run_ldred_1cta_direct_index_allocator_crash_case():
    m = 256
    n = 32
    torch.manual_seed(0x46708516)
    layout = _make_linear_layout(m, n, "identity", "identity")
    parent_layout = _lift_layout(layout, [2])
    red_layout = ttgl.BlockedLayout([1], [32], [4], [0])
    inp = torch.randn((m, n), dtype=torch.float32, device="cuda")
    out = torch.empty_like(inp)
    red = torch.empty((m, ), dtype=torch.float32, device="cuda")
    compiled = _fuzz_ldred_kernel[(1, )](
        inp,
        out,
        red,
        layout,
        parent_layout,
        red_layout,
        m,
        n,
        1,
        False,
        num_warps=4,
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    torch.testing.assert_close(red, torch.min(inp, dim=1).values, atol=0, rtol=0)
    ptx_ops = _extract_tcgen05_ops(compiled.asm["ptx"], ("ld", ))
    llir_ops = _extract_tcgen05_ops(compiled.asm["llir"], ("ld", ))
    assert ptx_ops == llir_ops
    assert any(".ld." in op for op in ptx_ops)
    assert not any(".ld.red." in op for op in ptx_ops)
    assert "tt.reduce" in compiled.asm["ttgir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_structural_fuzzer_ldred_twocta_rowcol_optimizer_crash():
    _run_structural_child("ldred_twocta_rowcol_optimizer_crash")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_structural_fuzzer_ldred_1cta_direct_index_allocator_crash():
    _run_structural_child("ldred_1cta_direct_index_allocator_crash")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("case", COPY_CASES, ids=lambda case: case.case_id)
def test_tmem_structural_fuzzer_copy_scales(case):
    torch.manual_seed(case.seed)
    m = 128 if case.two_ctas else 64
    n = 16
    inp = torch.randint(-100, 100, (m, n), dtype=torch.int8, device="cuda")
    out = torch.empty_like(inp)
    compiled = _fuzz_copy_scales_kernel[(1, )](inp, out, case.two_ctas, num_warps=4, num_ctas=2 if case.two_ctas else 1)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    ptx_ops = _extract_tcgen05_copy_ops(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_copy_ops(compiled.asm["llir"])
    assert ptx_ops == llir_ops
    # Direct TensorMemoryScalesLayout warpx4 copies are currently a pair of
    # single-CTA messages even when the kernel runs in a two-CTA CGA.
    expected_group = "cta_group::1"
    assert ptx_ops
    assert all(expected_group in op for op in ptx_ops), ptx_ops


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("case", DYNAMIC_COPY_CASES, ids=lambda case: case.case_id)
def test_tmem_structural_fuzzer_copy_dynamic_index(case):
    torch.manual_seed(case.seed)
    m = 128
    n = 4
    layout = _lift_layout(_make_linear_layout(m, n), [2])
    inp = torch.randint(-100, 100, (m, n), dtype=torch.int32, device="cuda")
    out = torch.empty_like(inp)
    selector = torch.tensor([case.selector], dtype=torch.int32, device="cuda")
    compiled = _fuzz_copy_dynamic_index_kernel[(1, )](
        inp, out, selector, layout, case.mode, num_warps=4
    )
    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    ptx_ops = _extract_tcgen05_copy_ops(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_copy_ops(compiled.asm["llir"])
    assert ptx_ops == llir_ops
    assert "tcgen05.cp.cta_group::1.128x128b" in ptx_ops


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("case", DYNAMIC_LDRED_CASES, ids=lambda case: case.case_id)
def test_tmem_structural_fuzzer_ldred_dynamic_index(case):
    torch.manual_seed(case.seed)
    m = 128
    n = 32
    layout = _lift_layout(_make_linear_layout(m, n), [2])
    red_layout = ttgl.BlockedLayout([1], [32], [4], [0])
    inp = torch.randn((m, n), dtype=torch.float32, device="cuda")
    out = torch.empty_like(inp)
    red = torch.empty((m, ), dtype=torch.float32, device="cuda")
    selector = torch.tensor([case.selector], dtype=torch.int32, device="cuda")
    compiled = _fuzz_ldred_dynamic_index_kernel[(1, )](
        inp, out, red, selector, layout, red_layout, num_warps=4
    )
    expected = inp + (2.0 if case.selector else 1.0)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    torch.testing.assert_close(red, torch.min(expected, dim=1).values, atol=0, rtol=0)
    ptx_ops = _extract_tcgen05_ops(compiled.asm["ptx"], ("ld", ))
    llir_ops = _extract_tcgen05_ops(compiled.asm["llir"], ("ld", ))
    assert ptx_ops == llir_ops
    assert any(".ld.red." in op for op in ptx_ops)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("case", GENERIC_PASS_MEMDESC_CASES, ids=lambda case: case.case_id)
def test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow(case):
    m = 128
    n = 64
    torch.manual_seed(case.seed)
    base_layout = _make_linear_layout(m, n)
    parent_layout = _lift_layout(base_layout, [2])
    inp = torch.randn((m, n), dtype=torch.float32, device="cuda")
    out = torch.empty_like(inp)
    selector = torch.tensor([case.selector], dtype=torch.int32, device="cuda")

    if case.kind == "dynamic_index":
        compiled = _fuzz_generic_pass_dynamic_index_kernel[(1, )](
            inp, out, selector, parent_layout, m, n, case.chain_id, num_warps=4
        )
        expected = _generic_pass_expected_view(
            inp + (20.0 if case.selector else 10.0), case.chain_id
        )
    elif case.kind == "dynamic_if":
        compiled = _fuzz_generic_pass_dynamic_if_kernel[(1, )](
            inp, out, selector, parent_layout, m, n, case.chain_id, case.instr_variant, num_warps=4
        )
        expected = _generic_pass_expected_view(
            inp + (30.0 if case.selector else 40.0), case.chain_id
        )
    elif case.kind == "dynamic_if_inline":
        compiled = _fuzz_generic_pass_dynamic_if_inline_chain0_kernel[(1, )](
            inp, out, selector, parent_layout, m, n, case.instr_variant, num_warps=4
        )
        expected = _generic_pass_expected_view(inp + (30.0 if case.selector else 40.0), 0)
    elif case.kind == "mixed_captures":
        compiled = _fuzz_generic_pass_mixed_captures_kernel[(1, )](
            inp,
            out,
            selector,
            parent_layout,
            m,
            n,
            case.chain_id,
            case.instr_variant,
            case.tuple_capture,
            num_warps=4,
        )
        selected = inp if case.selector else inp * 2.0
        bias = inp + (3.0 if case.selector else 5.0)
        expected = _generic_pass_expected_view(selected, case.chain_id) + bias
    else:
        raise AssertionError(f"unknown generic-pass memdesc case kind: {case.kind}")

    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    ptx_ops = _extract_tcgen05_ops(compiled.asm["ptx"], ("ld", "st"))
    llir_ops = _extract_tcgen05_ops(compiled.asm["llir"], ("ld", "st"))
    assert ptx_ops == llir_ops
    assert any(".ld." in op for op in ptx_ops)
    assert any(".st." in op for op in ptx_ops)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("case", DYNAMIC_INDEX_LOAD_ONLY_CASES, ids=lambda case: case.case_id)
def test_tmem_structural_fuzzer_generic_pass_dynamic_index_load_only(case):
    torch.manual_seed(case.seed)
    base_layout = _make_linear_layout(case.m, case.n)
    parent_layout = _lift_layout(base_layout, [2])
    out = torch.empty((case.m, case.n), dtype=torch.float32, device="cuda")
    selector = torch.tensor([case.selector], dtype=torch.int32, device="cuda")
    compiled = _fuzz_generic_pass_dynamic_index_load_only_kernel[(1, )](
        out, selector, parent_layout, case.m, case.n, num_warps=4
    )
    ptx_ops = _extract_tcgen05_ops(compiled.asm["ptx"], ("ld", ))
    llir_ops = _extract_tcgen05_ops(compiled.asm["llir"], ("ld", ))
    assert ptx_ops == llir_ops
    assert any(".ld." in op for op in ptx_ops)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("case", GENERIC_PASS_LAYOUT_PRESSURE_CASES, ids=lambda case: case.case_id)
def test_tmem_structural_fuzzer_generic_pass_layout_conversion_pressure(case):
    m = 128
    n = 64
    torch.manual_seed(case.seed)
    base_layout = _make_linear_layout(m, n)
    parent_layout = _lift_layout(base_layout, [2])
    inp = torch.randn((m, n), dtype=torch.float32, device="cuda")
    out = torch.empty_like(inp)
    compiled = _fuzz_generic_pass_layout_pressure_kernel[(1, )](
        inp, out, parent_layout, m, n, case.chain_id, case.instr_variant, num_warps=4
    )
    expected = _generic_pass_expected_view(inp, case.chain_id)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    ptx_ops = _extract_tcgen05_ops(compiled.asm["ptx"], ("ld", "st"))
    llir_ops = _extract_tcgen05_ops(compiled.asm["llir"], ("ld", "st"))
    assert ptx_ops == llir_ops
    assert any(".ld." in op for op in ptx_ops)
    assert any(".st." in op for op in ptx_ops)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("case", GENERIC_PASS_LOOP_CARRIED_CASES, ids=lambda case: case.case_id)
def test_tmem_structural_fuzzer_generic_pass_loop_carried(case):
    m = 128
    n = 64
    torch.manual_seed(case.seed)
    base_layout = _make_linear_layout(m, n)
    parent_layout = _lift_layout(base_layout, [2])
    inp = torch.randn((m, n), dtype=torch.float32, device="cuda")
    out = torch.empty_like(inp)
    selector = torch.tensor([case.selector], dtype=torch.int32, device="cuda")
    loop_count = torch.tensor([case.loops], dtype=torch.int32, device="cuda")

    compiled = _fuzz_generic_pass_loop_carried_kernel[(1, )](
        inp,
        out,
        selector,
        loop_count,
        parent_layout,
        m,
        n,
        case.chain_id,
        case.instr_variant,
        num_warps=4,
    )
    expected = _generic_pass_expected_view(
        inp + (20.0 if case.selector < case.loops else 10.0), case.chain_id
    )
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    ptx_ops = _extract_tcgen05_ops(compiled.asm["ptx"], ("ld", "st"))
    llir_ops = _extract_tcgen05_ops(compiled.asm["llir"], ("ld", "st"))
    assert ptx_ops == llir_ops
    assert any(".ld." in op for op in ptx_ops)
    assert any(".st." in op for op in ptx_ops)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("case", SCALED_MMA_CONTROL_FLOW_CASES, ids=lambda case: case.case_id)
def test_tmem_structural_fuzzer_scaled_mma_acc_subslice_control_flow(case):
    m = 128
    k = 128
    torch.manual_seed(case.seed)
    a, a_scale, a_ref = random_quantized_tensor(m, k, "mxfp8")
    b, b_scale, b_ref = random_quantized_tensor(case.n, k, "mxfp8")
    acc_layout = _make_linear_layout(m, 2 * case.n)
    out = torch.empty((m, case.n), dtype=torch.float32, device="cuda")
    selector = torch.tensor(case.selector, dtype=torch.int32, device="cuda")
    loop_count = torch.tensor(case.loop_count, dtype=torch.int32, device="cuda")
    compiled = _fuzz_scaled_mma_acc_subslice_if_kernel[(1, )](
        out,
        selector,
        loop_count,
        m,
        case.n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        acc_layout,
        num_warps=4,
    )

    expected_acc = 7.0 if case.selector else 3.0
    torch.testing.assert_close(out.to(torch.float32), a_ref @ b_ref.T + expected_acc, atol=1e-3, rtol=1e-3)
    ptx_ops = _extract_tcgen05_mma_ops(compiled.asm["ptx"])
    llir_ops = _extract_tcgen05_mma_ops(compiled.asm["llir"])
    assert ptx_ops == llir_ops
    assert ptx_ops
    assert all(op.startswith("tcgen05.mma.cta_group::1.kind::mxf8f6f4") for op in ptx_ops)


def _run_structural_child_case(case_id):
    if case_id == "ldst_256row_lifted_parent_allocator_crash":
        case = LdStCase(
            "ldst-fz20260421-0005-256row-lifted-parent",
            0xA005,
            256,
            32,
            "identity",
            "identity",
            "32x32b",
            3,
        )
        _run_ldst_case(case)
        return
    if case_id == "ldred_twocta_rowcol_optimizer_crash":
        _run_ldred_twocta_rowcol_optimizer_crash_case()
        return
    if case_id == "ldred_1cta_direct_index_allocator_crash":
        _run_ldred_1cta_direct_index_allocator_crash_case()
        return
    raise ValueError(f"unknown structural child case {case_id}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--structural-child":
        _run_structural_child_case(sys.argv[2])
    else:
        raise SystemExit("usage: test_tmem_structural_fuzzer.py --structural-child <case-id>")
