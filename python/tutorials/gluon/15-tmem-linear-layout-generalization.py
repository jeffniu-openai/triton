"""
TMEM Linear Layouts on Blackwell
================================

The earlier Blackwell tutorials introduce tensor memory (TMEM), `tcgen05_mma`,
`tcgen05_copy`, and block-scaled MMA.  This tutorial explains the more general
TMEM layout support available through `TensorMemoryLinearLayout`.

`TensorMemoryLinearLayout` describes the physical TMEM address of each logical
tensor index bit.  Once a TMEM allocation has an exact linear layout, descriptor
views such as `index`, `slice`, `reshape`, `permute`, and `bitcast` can be
composed algebraically.  The compiler then asks a hardware question:

Can this logical operation be covered by the public Blackwell TMEM instruction
atoms without changing the requested data semantics?

That distinction is important.  General linear layouts make many useful
descriptor chains and accumulator layouts expressible, but the hardware still
has fixed instruction footprints.  For example:

- `tcgen05.ld` and `tcgen05.st` move packet shapes such as `32x32b`,
  `16x64b`, and related variants.  A layout is usable when the requested view
  can be partitioned into those packets without aliasing or missing elements.
- `tcgen05.cp` writes complete copy-atom footprints.  If a program asks for
  only half of the destination columns of a public copy atom, the instruction
  has no mask bit that makes that a partial write.
- `tcgen05.mma` and `tcgen05.mma_scaled` consume fixed instruction tiles.  The
  compiler can permute whole tiles and split descriptor views, but the row and
  column bases inside each instruction tile must match what the tensor core
  instruction reads and writes.
- Some storage formats have extra semantics beyond a raw TMEM byte layout.  In
  particular, mixed fp4 operand-A paths use the shared-memory `fp4_padded`
  contract.  A direct TMEM LHS would need to represent that padding and swizzle
  contract explicitly; a raw linear TMEM layout alone is not enough.

The examples below execute kernels and check numerical results.  They avoid
compiler-only tests so the file can also be run as a normal tutorial on a
Blackwell machine.
"""

import importlib
import math

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


# Reuse the quantized tensor generator from `11-tcgen05-mma-scaled.py`.  That
# tutorial already explains MXFP4/MXFP8/NVFP4 packing and scale tensors; this
# file focuses on what the new TMEM layouts make possible.
t11 = importlib.import_module("11-tcgen05-mma-scaled")


# %%
# Building useful linear TMEM layouts
# -----------------------------------
#
# The identity layout maps logical row bits to physical TMEM rows and logical
# column bits to physical TMEM columns.


def make_tmem_linear_layout(m, n):
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(int(math.log2(m)))],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))],
        shape=[m, n],
    )


# Whole-tile permutations are useful for epilogues and skinny projections.  The
# helper below swaps the column bit that selects a tile with the next larger
# column bit.  For `N=32, tile_n=8`, this makes the accumulator physically look
# like four 8-column tiles in an order that the MMA planner can split into
# public instruction fragments.


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


# %%
# Example 1: descriptor chains that still load and store correctly
# ---------------------------------------------------------------
#
# This example writes through a descriptor chain rather than the root TMEM
# descriptor.  The chain indexes a higher-rank allocation, reshapes it, permutes
# it twice, slices it, and then bitcasts it back to the original logical tile.
# The interesting part is that the store and load still execute as real TMEM
# load/store packets; the user-visible result is just `input + 3`.


@gluon.jit
def descriptor_chain_add_kernel(in_ptr, out_ptr, layout: gl.constexpr, M: gl.constexpr, N: gl.constexpr):
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
def test_descriptor_chain_add():
    m, n = 128, 32
    layout = make_tmem_tile_permuted_layout(m, n, tile_n=8)
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty_like(inp)

    descriptor_chain_add_kernel[(1, )](inp, out, layout, m, n, num_warps=4)

    torch.testing.assert_close(out, inp + 3, atol=0, rtol=0)


# %%
# Example 2: exact-width subword `tcgen05.copy`
# --------------------------------------------
#
# A 128-bit copy atom is a natural fit for f16 `128x8` or i8 `128x16` tiles.
# Those are awkward to express with the compact `TensorMemoryLayout` frontend
# encoding, but they are direct to express as linear TMEM layouts.


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

    copy_128x128b_exact_width_kernel[(1, )](inp, out, n, layout, shared_layout, num_warps=4)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)


# %%
# Example 3: a real use case - skinny block-scaled projection
# ----------------------------------------------------------
#
# Block-scaled FP8/FP4 GEMM is one of Blackwell's important fast paths.  Large
# GEMMs usually fill broad accumulator tiles, but many model kernels are skinny:
# router projections, low-rank adapter projections, small vocabulary heads, and
# fused side projections often compute `M x K` by `K x 32` or `K x 64`.
#
# Without narrow accumulator fragments, a kernel often has to compute a padded
# `N=128` tile and throw most of it away.  A tile-permuted linear TMEM layout
# lets the scaled-MMA planner cover `N=32` with 8-column fragments and `N=64`
# with 16-column fragments, so the kernel below computes only the useful
# columns.
#
# The important new capability here is not block-scaled MMA itself; that is
# covered in `11-tcgen05-mma-scaled.py`.  The new part is that the accumulator
# can be a tile-permuted `TensorMemoryLinearLayout` whose `N` dimension is
# smaller than the old broad accumulator tile.  On the pre-generalization path,
# the compact `TensorMemoryLayout([BLOCK_M, BLOCK_N])` spelling could not
# express these narrow physical accumulator fragments, so a practical skinny
# projection had to use a broader accumulator tile or a non-TMEM fallback.


@gluon.jit
def skinny_mxfp8_projection_kernel(
    a,
    b,
    a_scale,
    b_scale,
    out,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    acc_layout: gl.constexpr,
):
    BLOCK_M: gl.constexpr = 128
    VEC_SIZE: gl.constexpr = 32
    pid_m = gl.program_id(0)
    off_m = pid_m * BLOCK_M

    block_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [gl.num_warps(), 1], [1, 0])
    a_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, block_layout))[:, None]
    a_k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, block_layout))[None, :]
    b_n = gl.arange(0, BLOCK_N, gl.SliceLayout(1, block_layout))[:, None]
    b_k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, block_layout))[None, :]

    a_tile = gl.load(a + a_m * BLOCK_K + a_k)
    b_tile = gl.load(b + b_n * BLOCK_K + b_k)
    smem_a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float8e4nv)
    smem_b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_K], gl.float8e4nv)
    smem_a = gl.allocate_shared_memory(gl.float8e4nv, [BLOCK_M, BLOCK_K], smem_a_layout, a_tile)
    smem_b = gl.allocate_shared_memory(gl.float8e4nv, [BLOCK_N, BLOCK_K], smem_b_layout, b_tile)

    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], acc_layout)
    acc_reg_layout: gl.constexpr = acc_tmem.get_reg_layout()
    acc_tmem.store(gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=acc_reg_layout))

    scale_layout: gl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype.element_ty, [BLOCK_M, BLOCK_K // VEC_SIZE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype.element_ty, [BLOCK_N, BLOCK_K // VEC_SIZE], scale_layout)
    a_scale_layout: gl.constexpr = a_scale_tmem.get_reg_layout()
    b_scale_layout: gl.constexpr = b_scale_tmem.get_reg_layout()

    a_scale_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, a_scale_layout))[:, None]
    a_scale_k = gl.arange(0, BLOCK_K // VEC_SIZE, gl.SliceLayout(0, a_scale_layout))[None, :]
    b_scale_n = gl.arange(0, BLOCK_N, gl.SliceLayout(1, b_scale_layout))[:, None]
    b_scale_k = gl.arange(0, BLOCK_K // VEC_SIZE, gl.SliceLayout(0, b_scale_layout))[None, :]
    a_scale_tmem.store(gl.load(a_scale + a_scale_m * (BLOCK_K // VEC_SIZE) + a_scale_k))
    b_scale_tmem.store(gl.load(b_scale + b_scale_n * (BLOCK_K // VEC_SIZE) + b_scale_k))

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(
        smem_a,
        smem_b.permute((1, 0)),
        acc_tmem,
        a_scale_tmem,
        b_scale_tmem,
        "e4m3",
        "e4m3",
        use_acc=True,
    )
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
    out_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, out_layout))[:, None]
    out_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, out_layout))[None, :]
    acc = acc_tmem.load()
    gl.store(out + out_m * BLOCK_N + out_n, gl.convert_layout(acc, out_layout))


def make_mxfp8_inputs(total_m, n, k):
    torch.manual_seed(0)
    a, a_scale, a_ref = t11.random_quantized_tensor(total_m, k, "mxfp8")
    b, b_scale, b_ref = t11.random_quantized_tensor(n, k, "mxfp8")
    return a, b, a_scale, b_scale, a_ref, b_ref


def skinny_projection_into(a, b, a_scale, b_scale, out, n, k):
    total_m = a.shape[0]
    tile_n = 8 if n == 32 else 16
    acc_layout = make_tmem_tile_permuted_layout(128, n, tile_n)
    grid = (triton.cdiv(total_m, 128), )
    skinny_mxfp8_projection_kernel[grid](
        a,
        b,
        a_scale,
        b_scale,
        out,
        n,
        k,
        acc_layout,
        num_warps=4,
    )


def skinny_projection(a, b, a_scale, b_scale, n, k):
    total_m = a.shape[0]
    out = torch.empty((total_m, n), device="cuda", dtype=torch.float32)
    skinny_projection_into(a, b, a_scale, b_scale, out, n, k)
    return out


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("n", [32, 64])
def test_skinny_mxfp8_projection(n):
    total_m, k = 512, 128
    a, b, a_scale, b_scale, a_ref, b_ref = make_mxfp8_inputs(total_m, n, k)

    out = skinny_projection(a, b, a_scale, b_scale, n, k)

    torch.testing.assert_close(out, a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)


def benchmark_skinny_projection(total_m=4096, n=32, k=128):
    assert n in (32, 64)
    a, b, a_scale, b_scale, a_ref, b_ref = make_mxfp8_inputs(total_m, n, k)
    padded_n = 128
    _, b_pad, _, b_scale_pad, _, b_ref_pad = make_mxfp8_inputs(padded_n, padded_n, k)

    # Put the useful skinny projection in the leading columns of the padded
    # problem.  The padded baseline models kernels that must execute a full
    # 128-column accumulator tile because they cannot use narrow TMEM fragments.
    b_pad[:n, :] = b
    b_scale_pad[:n, :] = b_scale
    b_ref_pad[:n, :] = b_ref

    narrow_result = torch.empty((total_m, n), device="cuda", dtype=torch.float32)
    padded_result = torch.empty((total_m, padded_n), device="cuda", dtype=torch.float32)
    narrow = lambda: skinny_projection_into(a, b, a_scale, b_scale, narrow_result, n, k)
    padded = lambda: skinny_projection_into(a, b_pad, a_scale, b_scale_pad, padded_result, padded_n, k)

    narrow()
    padded()
    torch.testing.assert_close(narrow_result, a_ref @ b_ref.T, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(padded_result[:, :n], narrow_result, atol=1e-3, rtol=1e-3)

    narrow_ms = triton.testing.do_bench(narrow)
    padded_ms = triton.testing.do_bench(padded)
    useful_flops = 2 * total_m * n * k
    return {
        "M": total_m,
        "N": n,
        "K": k,
        "narrow_ms": narrow_ms,
        "padded_128_ms": padded_ms,
        "speedup": padded_ms / narrow_ms,
        "useful_tflops": useful_flops * 1e-12 / (narrow_ms * 1e-3),
    }


# %%
# Example 4: load-reduce on an explicit tile layout
# -------------------------------------------------
#
# Reductions show the same hardware rule in a smaller kernel.  If the view can
# be covered by public `tcgen05.ld.red` packets, the hardware performs the
# reduction.  If a different exact layout cannot be covered, a good compiler
# should fall back to scalar/software reduction or emit a precise unsupported
# reason.  The user-facing operation below is simply rowwise max.


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
def test_tmem_rowwise_max():
    m, n = 128, 128
    inp = torch.randn((m, n), device="cuda", dtype=torch.float32)
    out = torch.empty_like(inp)
    red = torch.empty((m,), device="cuda", dtype=torch.float32)
    layout = make_tmem_tile_permuted_layout(m, n, tile_n=32)

    tmem_reduce_max_kernel[(1, )](inp, out, red, layout, m, n, tl.PropagateNan.NONE, num_warps=4)

    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    torch.testing.assert_close(red, torch.max(inp, dim=1).values, atol=0, rtol=0)


# %%
# Hardware limits to keep in mind
# -------------------------------
#
# Linear layouts make the descriptor arithmetic exact; they do not add new
# Blackwell instruction masks or storage formats.  A few concrete examples:
#
# `tcgen05.copy` partial footprints
#   A public copy atom writes its full destination footprint.  If a descriptor
#   view asks for only columns 0, 1, 4, and 5 of an 8-column footprint, the
#   compiler cannot silently write columns 2, 3, 6, and 7.  That needs a smaller
#   copy atom, a destination mask, or a different staging format.
#
# Packed-lane subword copies
#   Some i8/f16 copies want two logical lanes packed into one physical TMEM
#   word and then read back as independent logical columns.  A raw linear layout
#   can describe the addresses, but the public copy/load/store sequence also
#   needs an agreed packed-storage contract.
#
# Refresh-image readback
#   `4x256b` refresh-shaped copies are useful because the copy atom writes that
#   image naturally.  Reading the same bytes back through an ordinary contiguous
#   view is a different hardware operation.  It needs a refresh remap/readback
#   model rather than just a different descriptor spelling.
#
# MMAv5 in-tile basis order
#   Whole instruction tiles can be selected and permuted, as the skinny
#   projection demonstrates.  Inside one public MMAv5 instruction tile, however,
#   the row and column basis order must match what the tensor core instruction
#   consumes.  Arbitrary bit permutations inside the tile need extra splitting,
#   masking, or a different MMA family.


if __name__ == "__main__":
    print("Skinny MXFP8 projection benchmark")
    print("=================================")
    for n in (32, 64):
        result = benchmark_skinny_projection(total_m=4096, n=n, k=128)
        print(
            f"M={result['M']} N={result['N']} K={result['K']} | "
            f"narrow={result['narrow_ms']:.3f} ms | "
            f"padded N=128={result['padded_128_ms']:.3f} ms | "
            f"speedup={result['speedup']:.2f}x | "
            f"useful={result['useful_tflops']:.1f} TFLOP/s"
        )

# %%
# On one GB200-class run, executing this file printed:
#
# ```
# Skinny MXFP8 projection benchmark
# =================================
# M=4096 N=32 K=128 | narrow=0.011 ms | padded N=128=0.013 ms | speedup=1.18x | useful=3.0 TFLOP/s
# M=4096 N=64 K=128 | narrow=0.013 ms | padded N=128=0.013 ms | speedup=1.05x | useful=5.3 TFLOP/s
# ```
#
# These tiny shapes are latency dominated, so the speedup is modest.  The point
# of the example is structural: the narrow kernel performs the same mathematical
# projection without materializing an unused 128-column accumulator tile, and it
# does so with the same `tcgen05_mma_scaled` hardware path.
