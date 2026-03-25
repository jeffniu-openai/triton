import math

import torch

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLinearLayout,
    allocate_tensor_memory,
)


def make_layout(m, n):
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(int(math.log2(m)))],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))],
        shape=[m, n],
    )


@gluon.jit
def probe_reshape_slice_dim0_kernel(in_ptr, out_ptr, layout: ttgl.constexpr,
                                    M: ttgl.constexpr, N: ttgl.constexpr,
                                    variant: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
    full = tmem.index(1)
    reg: ttgl.constexpr = full.get_reg_layout(instr_variant=variant)
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg))
    offs = offs_m[:, None] * N + offs_n[None, :]
    val = ttgl.load(in_ptr + offs)
    full.store(ttgl.convert_layout(val, reg))

    view = full.reshape((2, M, N // 2)).slice(1, 1, dim=0).index(0)
    preg: ttgl.constexpr = view.get_reg_layout(instr_variant=variant)
    tmp = view.load(preg)
    view.store(tmp + ttgl.full([M, N // 2], 7.0, ttgl.float32, layout=preg))

    out = full.load(reg)
    ttgl.store(out_ptr + offs, out)


@gluon.jit
def probe_reshape_slice_dim1_kernel(in_ptr, out_ptr, layout: ttgl.constexpr,
                                    M: ttgl.constexpr, N: ttgl.constexpr,
                                    variant: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
    full = tmem.index(1)
    reg: ttgl.constexpr = full.get_reg_layout(instr_variant=variant)
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg))
    offs = offs_m[:, None] * N + offs_n[None, :]
    val = ttgl.load(in_ptr + offs)
    full.store(ttgl.convert_layout(val, reg))

    view = full.reshape((M, 2, N // 2)).slice(1, 1, dim=1).reshape((M, N // 2))
    preg: ttgl.constexpr = view.get_reg_layout(instr_variant=variant)
    tmp = view.load(preg)
    view.store(tmp + ttgl.full([M, N // 2], 11.0, ttgl.float32, layout=preg))

    out = full.load(reg)
    ttgl.store(out_ptr + offs, out)


@gluon.jit
def probe_reshape_half_rows_kernel(in_ptr, out_ptr, layout: ttgl.constexpr,
                                   M: ttgl.constexpr, N: ttgl.constexpr,
                                   variant: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
    full = tmem.index(1)
    reg: ttgl.constexpr = full.get_reg_layout(instr_variant=variant)
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg))
    offs = offs_m[:, None] * N + offs_n[None, :]
    val = ttgl.load(in_ptr + offs)
    full.store(ttgl.convert_layout(val, reg))

    view = full.reshape((2, M // 2, N)).slice(1, 1, dim=0).index(0)
    preg: ttgl.constexpr = view.get_reg_layout(instr_variant=variant)
    tmp = view.load(preg)
    view.store(tmp + ttgl.full([M // 2, N], 13.0, ttgl.float32, layout=preg))

    out = full.load(reg)
    ttgl.store(out_ptr + offs, out)


@gluon.jit
def probe_direct_highrank_layout_kernel(in_ptr, out_ptr, layout: ttgl.constexpr,
                                        M: ttgl.constexpr, N: ttgl.constexpr,
                                        variant: ttgl.constexpr):
    tmem = allocate_tensor_memory(ttgl.float32, [2, 2, M // 2, N], layout)
    view = tmem.slice(1, 1, dim=0).index(0).slice(1, 1, dim=0).index(0)
    reg: ttgl.constexpr = view.get_reg_layout(instr_variant=variant)
    offs_m = ttgl.arange(0, M // 2, ttgl.SliceLayout(1, reg))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg))
    offs = offs_m[:, None] * N + offs_n[None, :]
    val = ttgl.load(in_ptr + offs)
    view.store(ttgl.convert_layout(val, reg))
    out = view.load(reg)
    ttgl.store(out_ptr + offs, out)


def run_probe(kernel, m, n, variant):
    layout = make_layout(m, n)
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty_like(inp)
    return kernel[(1,)](inp, out, layout, m, n, variant, num_warps=4), inp, out


def main():
    probes = [
        ("reshape_slice_dim0", probe_reshape_slice_dim0_kernel, 128, 128,
         "16x128b"),
        ("reshape_slice_dim1", probe_reshape_slice_dim1_kernel, 128, 128,
         "16x128b"),
        ("reshape_half_rows", probe_reshape_half_rows_kernel, 128, 128,
         "16x128b"),
        ("direct_highrank_layout", probe_direct_highrank_layout_kernel, 128,
         128, "16x128b"),
    ]
    for name, kernel, m, n, variant in probes:
        try:
            compiled, inp, out = run_probe(kernel, m, n, variant)
            print(
                f"{name}: PASS "
                f"ld={compiled.asm['ptx'].count('tcgen05.ld.sync.aligned')} "
                f"st={compiled.asm['ptx'].count('tcgen05.st.sync.aligned')}"
            )
            print(out[:4, :8])
        except Exception as exc:
            print(f"{name}: FAIL {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
