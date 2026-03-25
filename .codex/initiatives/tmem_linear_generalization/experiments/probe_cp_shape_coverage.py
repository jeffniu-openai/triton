#!/usr/bin/env python3
"""Probe tcgen05.cp opcode families for targeted layout shapes."""

from __future__ import annotations

import re

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_copy,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier

CP_OPCODE_RE = re.compile(
    r"(tcgen05\.cp(?:\.cta_group::\d+)?(?:\.warpx[24](?:::[^\s.;]+)*)?\.\d+x\d+b)"
)


def extract_cp_opcodes(asm: str) -> list[str]:
    return CP_OPCODE_RE.findall(asm)


def make_identity_shared_layout(shape: tuple[int, ...]) -> ttgl.SharedLinearLayout:
    rank = len(shape)
    offset_bases: list[list[int]] = []
    for dim, length in enumerate(shape):
        stride = 1
        while stride < length:
            basis = [0] * rank
            basis[dim] = stride
            offset_bases.append(basis)
            stride <<= 1
    return ttgl.SharedLinearLayout(offset_bases=offset_bases)


def _build_cp_shape_probe(rows: int, cols: int):
    rows_const = ttgl.constexpr(rows)
    cols_const = ttgl.constexpr(cols)
    base_layout = make_identity_shared_layout((rows, cols))
    @gluon.jit
    def _cp_shape_probe(
        _in,
        _out,
    ):
        grid = ttgl.arange(0, rows_const)[:, None] * cols_const + ttgl.arange(0, cols_const)[None, :]
        value = ttgl.load(_in + grid)

        smem = ttgl.allocate_shared_memory(ttgl.int32, (rows_const, cols_const), layout=base_layout)
        smem.store(value)

        alias_layout: ttgl.constexpr = TensorMemoryLayout((rows_const, cols_const), col_stride=1)
        tmem = allocate_tensor_memory(ttgl.int32, (rows_const, cols_const), layout=alias_layout)

        barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(barrier, count=1)
        tcgen05_copy(smem, tmem)
        tcgen05_commit(barrier)
        mbarrier.wait(barrier, phase=0)

    return _cp_shape_probe


def main():
    torch.manual_seed(0)
    shapes = [
        ("128x128b", 128, 32),
        ("128x256b", 128, 64),
    ]

    kernel_builders = {
        (rows, cols): _build_cp_shape_probe(rows, cols) for _, rows, cols in shapes
    }

    for label, rows, cols in shapes:
        inp = torch.arange(rows * cols, dtype=torch.int32, device="cuda")
        out = torch.empty_like(inp)
        kernel = kernel_builders[(rows, cols)]
        compiled = kernel[(1,)](
            inp,
            out,
            num_warps=4,
        )
        ptx_ops = extract_cp_opcodes(compiled.asm["ptx"])
        llir_ops = extract_cp_opcodes(compiled.asm["llir"])
        print(f"shape={label} pt= {ptx_ops} llir={llir_ops}")


if __name__ == "__main__":
    main()
