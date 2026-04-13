#!/usr/bin/env python3
"""Probe scaled MMAv5 TMEM-LHS subview format reachability.

This is a durable reproduction for the scaled TMEM-LHS subview format frontier.
After the 2026-04-13 packed-TMEM-A K-stride fix, homogeneous fp4 families are
runtime-correct; mixed `mxfp4/mxfp8` remains a wrong-code frontier.
"""

import math

import torch

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma_scaled,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier

from python.test.gluon.test_core import random_quantized_tensor
from python.test.gluon.test_tmem_runtime_matrix import (
    CP_SCALES_WARPX4_FORMAT_PAIRS,
    _extract_tcgen05_mma_opcodes,
    _expected_scaled_mma_acc_subslice_count,
    _expected_scaled_mma_opcode,
    _scaled_mma_operand_params,
)


def _make_tmem_linear_layout(m, n):
    from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLinearLayout

    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(int(math.log2(m)))],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))],
        shape=[m, n],
    )


@gluon.jit
def probe_mma_scaled_lhs_subslice_format_kernel(
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
    acc_tmem.store(ttgl.zeros([M, N], ttgl.float32, layout=acc_reg_layout))

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
    ttgl.store(out_ptr + offs_m * N + offs_n, ttgl.convert_layout(out_reg, reg_layout))


def run_case(a_format, b_format, acc_layout_kind):
    m = n = k = 128
    vec_size = 16 if a_format == "nvfp4" else 32
    a_elem_per_byte, a_tcgen_format = _scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = _scaled_mma_operand_params(b_format)
    torch.manual_seed(0)
    a, a_scale, a_ref = random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")
    acc_layout = (
        TensorMemoryLayout((m, n), col_stride=1)
        if acc_layout_kind == "legacy"
        else _make_tmem_linear_layout(m, n)
    )

    compiled = probe_mma_scaled_lhs_subslice_format_kernel[(1,)](
        out,
        m,
        n,
        k,
        a,
        b,
        a_scale,
        b_scale,
        _make_tmem_linear_layout(m, 2 * (k // a_elem_per_byte)),
        acc_layout,
        vec_size,
        a_elem_per_byte,
        b_elem_per_byte,
        a_tcgen_format,
        b_tcgen_format,
        num_warps=4,
    )

    expected = a_ref @ b_ref.T
    diff = (out.to(torch.float32) - expected).abs()
    max_abs = diff.max().item()
    denom = expected.abs().clamp_min(1e-30)
    max_rel = (diff / denom).max().item()
    ok = torch.allclose(out.to(torch.float32), expected, atol=1e-3, rtol=1e-3)
    ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    expected_op = _expected_scaled_mma_opcode(a_format, b_format, 1)
    expected_count = _expected_scaled_mma_acc_subslice_count(a_format, b_format)
    op_ok = bool(ops) and all(op == expected_op for op in ops) and len(ops) == expected_count
    status = "PASS" if ok and op_ok else "WRONG" if not ok else "OPCODE_MISMATCH"
    print(
        "RESULT"
        f" a={a_format} b={b_format} acc={acc_layout_kind} status={status}"
        f" max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
        f" op_count={len(ops)} expected_count={expected_count}"
        f" expected_op={expected_op}"
    )


if __name__ == "__main__":
    for a_format, b_format in CP_SCALES_WARPX4_FORMAT_PAIRS:
        for acc_layout_kind in ("legacy", "linear"):
            try:
                run_case(a_format, b_format, acc_layout_kind)
            except Exception as exc:  # Preserve compile/runtime failures in the log.
                print(
                    "RESULT"
                    f" a={a_format} b={b_format} acc={acc_layout_kind}"
                    f" status=ERROR error={type(exc).__name__}: {exc}"
                )
