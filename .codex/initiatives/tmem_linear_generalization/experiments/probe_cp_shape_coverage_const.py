#!/usr/bin/env python3
"""Probe tcgen05.cp shape families without the tt.make_range-heavy input path."""

from __future__ import annotations

import re

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
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


@gluon.jit
def cp_128x128b_const_kernel():
    rows: ttgl.constexpr = 128
    cols: ttgl.constexpr = 4
    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    value = ttgl.full([rows, cols], 7, ttgl.int32, layout=blocked)
    smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]],
        alignment=16,
    )
    smem = ttgl.allocate_shared_memory(ttgl.int32, [rows, cols], layout=smem_layout)
    tmem_layout: ttgl.constexpr = TensorMemoryLayout((rows, cols), col_stride=1)
    tmem = allocate_tensor_memory(ttgl.int32, [rows, cols], layout=tmem_layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    smem.store(value)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)


@gluon.jit
def cp_128x256b_const_kernel():
    rows: ttgl.constexpr = 128
    cols: ttgl.constexpr = 8
    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [32, 1], [4, 1], [1, 0])
    value = ttgl.full([rows, cols], 7, ttgl.int32, layout=blocked)
    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=32, rank=2)
    smem = ttgl.allocate_shared_memory(ttgl.int32, [rows, cols], layout=smem_layout)
    tmem_layout: ttgl.constexpr = TensorMemoryLayout((rows, cols), col_stride=1)
    tmem = allocate_tensor_memory(ttgl.int32, [rows, cols], layout=tmem_layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    smem.store(value)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)


@gluon.jit
def cp_warpx4_32x128b_const_kernel():
    rows: ttgl.constexpr = 64
    cols: ttgl.constexpr = 16
    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    value = ttgl.full([rows, cols], 3, ttgl.int8, layout=blocked)
    smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
        offset_bases=[[0, 1], [0, 2], [32, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]]
    )
    smem = ttgl.allocate_shared_memory(ttgl.int8, [rows, cols], layout=smem_layout)
    tmem = allocate_tensor_memory(ttgl.int8, [rows, cols], layout=TensorMemoryScalesLayout())
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    smem.store(value)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)


def main():
    torch.manual_seed(0)
    probes = [
        ("cp_128x128b", lambda: cp_128x128b_const_kernel[(1,)](num_warps=4)),
        ("cp_128x256b", lambda: cp_128x256b_const_kernel[(1,)](num_warps=4)),
        ("cp_warpx4_32x128b", lambda: cp_warpx4_32x128b_const_kernel[(1,)]()),
    ]

    for name, runner in probes:
        compiled = runner()
        ptx_ops = extract_cp_opcodes(compiled.asm["ptx"])
        llir_ops = extract_cp_opcodes(compiled.asm["llir"])
        if ptx_ops != llir_ops:
            raise RuntimeError(f"{name}: PTX/LLIR opcode mismatch: {ptx_ops} vs {llir_ops}")
        if not ptx_ops:
            raise RuntimeError(f"{name}: no tcgen05.cp opcode emitted")
        print(f"{name}: count={len(ptx_ops)} opcodes={ptx_ops}")


if __name__ == "__main__":
    main()
