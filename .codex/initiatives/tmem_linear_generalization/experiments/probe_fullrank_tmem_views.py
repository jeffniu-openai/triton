import argparse
import math
import re

import torch

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLinearLayout,
    allocate_tensor_memory,
)


def make_rank3_layout(m, n):
    return TensorMemoryLinearLayout(
        rows=[[0, 1 << i, 0] for i in range(int(math.log2(m)))],
        cols=[[0, 0, 1 << i] for i in range(int(math.log2(n)))] + [[1, 0, 0]],
        shape=[2, m, n],
    )


def make_rank4_layout(m, n):
    return TensorMemoryLinearLayout(
        rows=[[0, 0, 1 << i, 0] for i in range(int(math.log2(m)))],
        cols=[[0, 0, 0, 1 << i] for i in range(int(math.log2(n)))] + [[0, 1, 0, 0], [1, 0, 0, 0]],
        shape=[2, 2, m, n],
    )


def extract_ldst_ops(asm: str):
    pattern = re.compile(r"(tcgen05\.(?:ld|st)\.sync\.aligned\.[^\s;\"]+)")
    return pattern.findall(asm)


@gluon.jit
def rank3_index_kernel(inp, out, layout: ttgl.constexpr, m: ttgl.constexpr, n: ttgl.constexpr):
    offs = ttgl.arange(0, m)[:, None] * n + ttgl.arange(0, n)[None, :]
    tmem = allocate_tensor_memory(ttgl.float32, [2, m, n], layout)
    view = tmem.index(1)
    reg_layout: ttgl.constexpr = view.get_reg_layout(instr_variant="32x32b")
    val = ttgl.load(inp + offs)
    view.store(ttgl.convert_layout(val, reg_layout))
    out_val = view.load(reg_layout)
    ttgl.store(out + offs, ttgl.convert_layout(out_val, reg_layout))


@gluon.jit
def rank4_slice_index_kernel(inp, out, layout: ttgl.constexpr, m: ttgl.constexpr, n: ttgl.constexpr):
    offs = ttgl.arange(0, m)[:, None] * n + ttgl.arange(0, n)[None, :]
    tmem = allocate_tensor_memory(ttgl.float32, [2, 2, m, n], layout)
    view = tmem.slice(1, 1, dim=0).index(0).slice(1, 1, dim=0).index(0)
    reg_layout: ttgl.constexpr = view.get_reg_layout(instr_variant="32x32b")
    val = ttgl.load(inp + offs)
    view.store(ttgl.convert_layout(val, reg_layout))
    out_val = view.load(reg_layout)
    ttgl.store(out + offs, ttgl.convert_layout(out_val, reg_layout))


@gluon.jit
def rank4_first_slice_kernel(inp, out, layout: ttgl.constexpr, m: ttgl.constexpr, n: ttgl.constexpr):
    offs = ttgl.arange(0, m)[:, None] * n + ttgl.arange(0, n)[None, :]
    tmem = allocate_tensor_memory(ttgl.float32, [2, 2, m, n], layout)
    view = tmem.slice(1, 1, dim=0)
    reg_layout: ttgl.constexpr = view.index(0).get_reg_layout(instr_variant="32x32b")
    val = ttgl.load(inp + offs)
    view.index(0).store(ttgl.convert_layout(val, reg_layout))
    out_val = view.index(0).load(reg_layout)
    ttgl.store(out + offs, ttgl.convert_layout(out_val, reg_layout))


@gluon.jit
def rank4_first_slice_index_kernel(inp, out, layout: ttgl.constexpr, m: ttgl.constexpr, n: ttgl.constexpr):
    offs = ttgl.arange(0, m)[:, None] * n + ttgl.arange(0, n)[None, :]
    tmem = allocate_tensor_memory(ttgl.float32, [2, 2, m, n], layout)
    view = tmem.slice(1, 1, dim=0).index(0)
    reg_layout: ttgl.constexpr = view.get_reg_layout(instr_variant="32x32b")
    val = ttgl.load(inp + offs)
    view.store(ttgl.convert_layout(val, reg_layout))
    out_val = view.load(reg_layout)
    ttgl.store(out + offs, ttgl.convert_layout(out_val, reg_layout))


def run_probe(kind: str, m: int, n: int):
    if kind == "rank3":
        kernel = rank3_index_kernel
        layout = make_rank3_layout(m, n)
    elif kind == "rank4":
        kernel = rank4_slice_index_kernel
        layout = make_rank4_layout(m, n)
    elif kind == "rank4_first_slice":
        kernel = rank4_first_slice_kernel
        layout = make_rank4_layout(m, n)
    else:
        kernel = rank4_first_slice_index_kernel
        layout = make_rank4_layout(m, n)

    inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
    out = torch.empty_like(inp)
    compiled = kernel[(1,)](inp, out, layout, m, n, num_warps=4)
    return {
        "equal": bool(torch.equal(inp, out)),
        "ops": extract_ldst_ops(compiled.asm["ptx"]),
        "ttgir_has_linear": "tensor_memory_linear" in compiled.asm["ttgir"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("rank3", "rank4", "rank4_first_slice", "rank4_first_slice_index"), required=True)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=32)
    args = parser.parse_args()
    print(run_probe(args.kind, args.m, args.n))


if __name__ == "__main__":
    main()
