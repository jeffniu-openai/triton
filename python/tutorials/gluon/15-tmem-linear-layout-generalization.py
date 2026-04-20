"""
TMEM Linear Layout Generalization
=================================

The earlier Blackwell tutorials introduce tensor memory (TMEM), `tcgen05_mma`,
`tcgen05_copy`, and block-scaled MMA.  This tutorial focuses on the features
added by the TMEM linear-layout generalization work on this branch.

The short version is that TMEM operations are no longer limited to a small set
of frontend layout spellings.  You can describe the physical TMEM mapping with
`TensorMemoryLinearLayout`, compose descriptor views with `index`, `slice`,
`reshape`, `permute`, and `bitcast`, and let the backend decide whether the
result can be expressed with public Blackwell TMEM instructions.

The new support covers:

- direct TMEM load/store on explicit linear layouts, including descriptor-view
  chains;
- `tcgen05_copy` into explicit linear layouts, including exact-width subword
  copies that previously lacked a public `TensorMemoryLayout` spelling;
- MMAv5 accumulators stored in linear TMEM layouts, including tile-permuted
  accumulator layouts;
- scaled-MMAv5 accumulators with narrow `N=8` and `N=16` fragments;
- typed clean failures for true boundaries such as masks, packed-lane storage,
  refresh-image readback, and mixed-fp4A direct TMEM storage.

The examples below are deliberately small.  They are intended to show the new
surface area in isolation, not to replace the pipelined kernels in
`10-tcgen05-copy.py` and `11-tcgen05-mma-scaled.py`.
"""

import importlib
import math
import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
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
from triton.experimental.gluon.language.nvidia.hopper import mbarrier


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


def is_blackwell_ultra():
    return is_blackwell() and torch.cuda.get_device_capability()[1] >= 3


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")


# Re-use the quantized tensor generator from the scaled-MMA tutorial so the
# examples here stay focused on new TMEM layout behavior.
t11 = importlib.import_module("11-tcgen05-mma-scaled")


# %%
# Linear TMEM layouts
# -------------------
#
# A `TensorMemoryLinearLayout` maps logical tensor index bits to physical TMEM
# row and column coordinates.  The identity mapping for an `M x N` tile is:


def make_tmem_linear_layout(m, n):
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(int(math.log2(m)))],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))],
        shape=[m, n],
    )


# Since the layout is explicit, we can also permute the row or column basis
# bits.  These are still exact linear layouts; the backend now reasons about
# them algebraically instead of trying to recognize one frontend layout class.


def _permute_pow2_bases(bits, kind):
    if kind == "identity":
        return list(bits)
    if kind == "rotate1":
        return list(bits[1:]) + [bits[0]]
    if kind == "even_odd":
        return list(bits[::2]) + list(bits[1::2])
    if kind == "reverse":
        return list(reversed(bits))
    raise ValueError(f"unknown permutation kind: {kind}")


def make_tmem_permuted_layout(m, n, row_kind="identity", col_kind="identity"):
    row_bits = _permute_pow2_bases([1 << i for i in range(int(math.log2(m)))], row_kind)
    col_bits = _permute_pow2_bases([1 << i for i in range(int(math.log2(n)))], col_kind)
    return TensorMemoryLinearLayout(
        rows=[[b, 0] for b in row_bits],
        cols=[[0, b] for b in col_bits],
        shape=[m, n],
    )


def make_tmem_tile_permuted_layout(m, n, tile_n):
    assert n >= 2 * tile_n
    col_bits = [1 << i for i in range(int(math.log2(n)))]
    tile_bit = int(math.log2(tile_n))
    col_bits[tile_bit], col_bits[tile_bit + 1] = col_bits[tile_bit + 1], col_bits[tile_bit]
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(int(math.log2(m)))],
        cols=[[0, bit] for bit in col_bits],
        shape=[m, n],
    )


def make_dense_shared_layout(m, n):
    return gl.SharedLinearLayout(
        offset_bases=[
            *[[0, 1 << i] for i in range(int(math.log2(n)))],
            *[[1 << i, 0] for i in range(int(math.log2(m)))],
        ],
        alignment=16,
    )


def scaled_mma_operand_params(format_name):
    if format_name == "mxfp8":
        return 1, "e4m3"
    if format_name in ("mxfp4", "nvfp4"):
        return 2, "e2m1"
    raise ValueError(f"unsupported scaled MMA format: {format_name}")


def _extract_tcgen05_opcodes(asm, prefix):
    return re.findall(rf"({prefix}[^\s;\"]+)", asm)


# %%
# Descriptor-view chains on linear TMEM
# -------------------------------------
#
# The first new pattern is that descriptor transformations compose cleanly with
# explicit linear TMEM layouts.  The view below goes through a higher-rank
# allocation, slices and indexes it back to a 2D tile, applies reshape/permute
# pairs, then bitcasts to the original element type and layout.  The compiler
# still derives a valid TMEM load/store plan from the final descriptor.


@gluon.jit
def descriptor_chain_roundtrip_kernel(in_ptr, out_ptr, layout: gl.constexpr, M: gl.constexpr, N: gl.constexpr):
    offs = gl.arange(0, M)[:, None] * N + gl.arange(0, N)[None, :]
    value = gl.load(in_ptr + offs)
    element_ty: gl.constexpr = in_ptr.dtype.element_ty

    tmem = allocate_tensor_memory(element_ty, [2, M, N], layout)
    view = tmem.slice(1, 1, dim=0).index(0)
    view = view.reshape((M // 2, 2, N)).permute([1, 0, 2]).reshape((M, N))
    view = view.permute([1, 0]).permute([1, 0])
    view = view.slice(0, M, dim=0).slice(0, N, dim=1)
    view = view.bitcast(element_ty, [M, N], layout)

    reg_layout: gl.constexpr = view.get_reg_layout()
    view.store(gl.convert_layout(value, reg_layout))
    out = view.load(reg_layout)
    out = out + gl.full([M, N], 3, element_ty, layout=reg_layout)
    view.store(out)
    out = tmem.index(1).load(reg_layout)
    gl.store(out_ptr + offs, out)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_descriptor_chain_roundtrip_on_permuted_linear_layout():
    m, n = 128, 32
    layout = make_tmem_permuted_layout(m, n, row_kind="even_odd", col_kind="rotate1")
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty_like(inp)

    compiled = descriptor_chain_roundtrip_kernel[(1, )](inp, out, layout, m, n, num_warps=4)

    torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttg.memdesc_reshape" in compiled.asm["ttgir"]
    assert "ttg.memdesc_subslice" in compiled.asm["ttgir"]


# %%
# Exact-width subword `tcgen05.copy`
# ----------------------------------
#
# `TensorMemoryLayout` is a compact frontend encoding for common 2D TMEM
# layouts, but it cannot spell every useful exact layout.  With
# `TensorMemoryLinearLayout`, a 128-bit copy atom can now target exact-width
# subword tiles such as f16 `128x8` and i8 `128x16`.


@gluon.jit
def copy_128x128b_exact_width_kernel(
    in_ptr, out_ptr, N: gl.constexpr, tmem_layout: gl.constexpr, shared_layout: gl.constexpr
):
    M: gl.constexpr = 128
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=tmem_layout)
    reg_layout: gl.constexpr = tmem.get_reg_layout()

    offs_m = gl.arange(0, M, gl.SliceLayout(1, reg_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = gl.load(in_ptr + offs)

    smem = gl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=shared_layout, value=value)
    fence_async_shared()

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out = tmem.load(reg_layout)
    gl.store(out_ptr + offs, out)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("dtype,n", [(torch.float16, 8), (torch.int8, 16)])
def test_exact_width_subword_copy(dtype, n):
    m = 128
    inp = torch.arange(m * n, device="cuda", dtype=torch.int32).reshape(m, n).to(dtype)
    out = torch.empty_like(inp)
    layout = make_tmem_linear_layout(m, n)
    shared_layout = make_dense_shared_layout(m, n)

    compiled = copy_128x128b_exact_width_kernel[(1, )](inp, out, n, layout, shared_layout, num_warps=4)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "tcgen05.cp.cta_group::1.128x128b" in compiled.asm["ptx"]


# %%
# MMAv5 accumulators in tile-permuted linear TMEM
# -----------------------------------------------
#
# The MMAv5 examples in `06-tcgen05.py` use the standard accumulator layout.
# This branch lets the accumulator be a compatible explicit linear layout.  In
# the next example, the logical accumulator is still `128x128`, but the
# column tile selector bits are permuted.  The backend lowers the descriptor
# views needed to feed public `tcgen05.mma` instruction tiles.


@gluon.jit
def mma_tile_permuted_acc_kernel(a_ptr, b_ptr, out_ptr, acc_layout: gl.constexpr):
    M: gl.constexpr = 128
    N: gl.constexpr = 128
    K: gl.constexpr = 32

    offs_a = gl.arange(0, M)[:, None] * K + gl.arange(0, K)[None, :]
    offs_b = gl.arange(0, K)[:, None] * N + gl.arange(0, N)[None, :]
    a_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    b_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    a = gl.load(gl.set_auto_layout(a_ptr + offs_a, a_layout))
    b = gl.load(gl.set_auto_layout(b_ptr + offs_b, b_layout))

    smem_a_layout: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16, rank=2)
    smem_b_layout: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16, rank=2)
    smem_a = gl.allocate_shared_memory(gl.float16, [M, K], smem_a_layout, a)
    smem_b = gl.allocate_shared_memory(gl.float16, [K, N], smem_b_layout, b)

    acc_tmem = allocate_tensor_memory(gl.float32, [M, N], acc_layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=tcgen05_mma_barrier_count([smem_a, smem_b], False))
    tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=False, mbarriers=[bar])
    mbarrier.wait(bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(bar)

    c_offs = gl.arange(0, M)[:, None] * N + gl.arange(0, N)[None, :]
    out = acc_tmem.load()
    gl.store(out_ptr + c_offs, out)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_mma_with_tile_permuted_linear_accumulator():
    m, n, k = 128, 128, 32
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)
    acc_layout = make_tmem_tile_permuted_layout(m, n, tile_n=32)

    compiled = mma_tile_permuted_acc_kernel[(1, )](a, b, out, acc_layout, num_warps=4)

    torch.testing.assert_close(out, a.to(torch.float32) @ b.to(torch.float32), atol=1e-2, rtol=1e-2)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "tcgen05.mma.cta_group::1.kind::f16" in compiled.asm["ptx"]


# %%
# Narrow scaled-MMAv5 accumulator fragments
# -----------------------------------------
#
# Scaled-MMAv5 has extra scale-storage constraints.  A previous implementation
# rejected narrow accumulator families such as `N=32, tile_n=8` and
# `N=64, tile_n=16`.  The branch now pads/rematerializes the B-scale storage
# so these narrow linear accumulator layouts work.


@gluon.jit
def scaled_mma_narrow_acc_kernel(
    out_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    a,
    b,
    a_scale,
    b_scale,
    acc_layout: gl.constexpr,
    VEC_SIZE: gl.constexpr,
    A_ELEM_PER_BYTE: gl.constexpr,
    B_ELEM_PER_BYTE: gl.constexpr,
    A_FORMAT: gl.constexpr,
    B_FORMAT: gl.constexpr,
):
    A_STORAGE_K: gl.constexpr = K // A_ELEM_PER_BYTE
    B_STORAGE_K: gl.constexpr = K // B_ELEM_PER_BYTE
    A_IS_FP4: gl.constexpr = A_ELEM_PER_BYTE == 2
    B_IS_FP4: gl.constexpr = B_ELEM_PER_BYTE == 2
    MIXED_PREC: gl.constexpr = A_ELEM_PER_BYTE != B_ELEM_PER_BYTE

    block_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [gl.num_warps(), 1], [1, 0])
    a_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [M, A_STORAGE_K],
        a.dtype.element_ty,
        fp4_padded=A_IS_FP4 and MIXED_PREC,
    )
    b_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [N, B_STORAGE_K],
        b.dtype.element_ty,
        fp4_padded=B_IS_FP4 and MIXED_PREC,
    )

    a_m = gl.arange(0, M, gl.SliceLayout(1, block_layout))[:, None]
    a_k = gl.arange(0, A_STORAGE_K, gl.SliceLayout(0, block_layout))[None, :]
    b_n = gl.arange(0, N, gl.SliceLayout(1, block_layout))[:, None]
    b_k = gl.arange(0, B_STORAGE_K, gl.SliceLayout(0, block_layout))[None, :]
    a_tile = gl.load(a + a_m * A_STORAGE_K + a_k)
    b_tile = gl.load(b + b_n * B_STORAGE_K + b_k)
    a_smem = gl.allocate_shared_memory(a.dtype.element_ty, [M, A_STORAGE_K], a_smem_layout, a_tile)
    b_smem = gl.allocate_shared_memory(b.dtype.element_ty, [N, B_STORAGE_K], b_smem_layout, b_tile)

    acc_tmem = allocate_tensor_memory(gl.float32, [M, N], acc_layout)
    acc_reg_layout: gl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(gl.zeros([M, N], gl.float32, layout=acc_reg_layout))

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [M, K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype.element_ty, [N, K // VEC_SIZE], scale_layout)
    a_scale_reg_layout: gl.constexpr = a_scale_tmem.get_reg_layout()
    b_scale_reg_layout: gl.constexpr = b_scale_tmem.get_reg_layout()

    scale_k_a = gl.arange(0, K // VEC_SIZE, gl.SliceLayout(0, a_scale_reg_layout))[None, :]
    scale_k_b = gl.arange(0, K // VEC_SIZE, gl.SliceLayout(0, b_scale_reg_layout))[None, :]
    scale_m = gl.arange(0, M, gl.SliceLayout(1, a_scale_reg_layout))[:, None]
    scale_n = gl.arange(0, N, gl.SliceLayout(1, b_scale_reg_layout))[:, None]
    a_scale_tmem.store(gl.load(a_scale + scale_m * (K // VEC_SIZE) + scale_k_a))
    b_scale_tmem.store(gl.load(b_scale + scale_n * (K // VEC_SIZE) + scale_k_b))

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
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

    out_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
    out_m = gl.arange(0, M, gl.SliceLayout(1, out_layout))[:, None]
    out_n = gl.arange(0, N, gl.SliceLayout(0, out_layout))[None, :]
    out = acc_tmem.load()
    gl.store(out_ptr + out_m * N + out_n, gl.convert_layout(out, out_layout))


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n,tile_n", [(32, 8), (64, 16)])
def test_scaled_mma_with_narrow_linear_accumulator(n, tile_n):
    m, k = 128, 128
    a_format = b_format = "mxfp8"
    vec_size = 32
    a_elem_per_byte, a_tcgen_format = scaled_mma_operand_params(a_format)
    b_elem_per_byte, b_tcgen_format = scaled_mma_operand_params(b_format)

    torch.manual_seed(0)
    a, a_scale, a_ref = t11.random_quantized_tensor(m, k, a_format)
    b, b_scale, b_ref = t11.random_quantized_tensor(n, k, b_format)
    out = torch.empty((m, n), device="cuda", dtype=torch.float32)
    acc_layout = make_tmem_tile_permuted_layout(m, n, tile_n)

    compiled = scaled_mma_narrow_acc_kernel[(1, )](
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
        num_warps=4,
    )

    torch.testing.assert_close(out, a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert "ttng.tc_gen5_mma_scaled" in compiled.asm["ttgir"]
    assert "tcgen05.mma.cta_group::1.kind::mxf8f6f4" in compiled.asm["ptx"]


# %%
# Reductions on explicit layouts
# ------------------------------
#
# TMEM load-reduce support also consumes explicit linear layouts.  When a public
# `tcgen05.ld.red` packet can realize the layout, the backend emits the hardware
# reduction.  For other exact layouts or dtypes, the same high-level operation
# falls back to a software reduction with a structured reason instead of a
# verifier crash.


@gluon.jit
def tmem_reduce_max_kernel(
    in_ptr, out_ptr, red_ptr, layout: gl.constexpr, M: gl.constexpr, N: gl.constexpr,
    propagate_nan: gl.constexpr
):
    reg_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    reg_layout_1d: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offs_m = gl.arange(0, M, gl.SliceLayout(1, reg_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]

    value = gl.load(in_ptr + offs)
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout)
    tmem_layout: gl.constexpr = tmem.get_reg_layout()
    tmem.store(gl.convert_layout(value, tmem_layout))

    output, reduced = tmem.load_max(abs=False, propagate_nan=propagate_nan)
    output = gl.convert_layout(output, reg_layout)
    gl.store(out_ptr + offs, output)

    red_offs = gl.arange(0, M, reg_layout_1d)
    reduced = gl.convert_layout(reduced, reg_layout_1d)
    gl.store(red_ptr + red_offs, reduced)


@pytest.mark.skipif(not is_blackwell_ultra(), reason="Requires Blackwell Ultra")
def test_tile_permuted_linear_layout_reduction():
    m, n = 128, 128
    inp = torch.randn((m, n), device="cuda", dtype=torch.float32)
    out = torch.empty_like(inp)
    red = torch.empty((m,), device="cuda", dtype=torch.float32)
    layout = make_tmem_tile_permuted_layout(m, n, tile_n=32)

    compiled = tmem_reduce_max_kernel[(1, )](inp, out, red, layout, m, n, tl.PropagateNan.NONE, num_warps=4)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    torch.testing.assert_close(red, torch.max(inp, dim=1).values, atol=0, rtol=0)
    assert "tensor_memory_linear" in compiled.asm["ttgir"]
    assert _extract_tcgen05_opcodes(compiled.asm["ptx"], "tcgen05.ld.red.sync.aligned")


# %%
# What still stays unsupported?
# -----------------------------
#
# The generalization does not mean every bit permutation can be expressed with a
# public Blackwell instruction.  The branch now tries to keep these cases as
# clean typed boundaries:
#
# - copy masks or sub-instruction row/column partitions that public
#   `tcgen05.cp` atoms cannot write;
# - packed-lane copy semantics that need a first-class packed TMEM storage
#   contract;
# - ordinary load/store readback of `4x256b` refresh images without an explicit
#   refresh remap/readback API;
# - direct TMEM LHS for mixed fp4A scaled-MMA, because the shared-memory
#   `fp4_padded` operand-A contract has row-dependent swizzle/padding semantics
#   not represented by raw TMEM storage.
#
# Those boundaries are useful: they separate "the backend can now derive the
# layout algebra" from "the public ISA or frontend storage contract cannot
# perform this operation yet."
