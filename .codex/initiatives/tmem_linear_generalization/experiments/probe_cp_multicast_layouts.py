#!/usr/bin/env python3
"""Probe tcgen05.cp multicast atom selection for scales-style copies."""

from __future__ import annotations

import os
import sys
from typing import Iterable

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

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


def extract_tcgen05_cp_opcodes(asm: str) -> list[str]:
    import re

    pattern = re.compile(
        r"(tcgen05\.cp(?:\.cta_group::\d+)?(?:\.warpx[24](?:::[^\s.;]+)*)?\.\d+x\d+b)"
    )
    return pattern.findall(asm)


def make_layout(interleaved_row: int, insert_at: int):
    col_bases = [[0, 1], [0, 2], [0, 4], [0, 8]]
    row_bases = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]]
    chosen = [interleaved_row, 0]
    row_bases = [basis for basis in row_bases if basis != chosen]
    offset_bases = col_bases[:insert_at] + [chosen] + col_bases[insert_at:] + row_bases
    return ttgl.SharedLinearLayout(offset_bases=offset_bases)


@gluon.jit
def cp_scales_probe_kernel(in_ptr, out_ptr, smem_layout: ttgl.constexpr):
    smem_h: ttgl.constexpr = 64
    smem_w: ttgl.constexpr = 16
    num_rows: ttgl.constexpr = 128
    num_cols: ttgl.constexpr = (smem_h * smem_w) // 32

    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    in_ptrs = in_ptr + ttgl.arange(0, smem_h)[:, None] * smem_w + ttgl.arange(0, smem_w)[None, :]
    out_ptrs = out_ptr + ttgl.arange(0, num_rows)[:, None] * num_cols + ttgl.arange(0, num_cols)[None, :]

    value = ttgl.load(ttgl.set_auto_layout(in_ptrs, blocked))

    smem = ttgl.allocate_shared_memory(ttgl.int8, (smem_h, smem_w), layout=smem_layout)
    tmem = allocate_tensor_memory(ttgl.int8, (smem_h, smem_w), layout=TensorMemoryScalesLayout())
    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)

    smem.store(value)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    alias_layout: ttgl.constexpr = TensorMemoryLayout((num_rows, num_cols), col_stride=1)
    alias_view = tmem._reinterpret(ttgl.int8, (num_rows, num_cols), alias_layout)
    out = alias_view.load(blocked)
    ttgl.store(ttgl.set_auto_layout(out_ptrs, blocked), out)


def candidate_layouts() -> Iterable[tuple[str, object]]:
    for interleaved_row in (1, 2, 4, 8, 16, 32):
        for insert_at in range(5):
            yield f"row{interleaved_row}_insert{insert_at}", make_layout(interleaved_row, insert_at)


def main():
    torch.manual_seed(0)
    inp = torch.arange(64 * 16, dtype=torch.int8, device="cuda").reshape(64, 16)
    out = torch.empty((128, 32), dtype=torch.int8, device="cuda")
    hits: dict[str, list[str]] = {}

    for name, layout in candidate_layouts():
        try:
            compiled = cp_scales_probe_kernel[(1,)](inp, out, layout)
        except Exception as exc:
            print(f"{name}: FAIL {type(exc).__name__}: {exc}")
            continue
        ops = extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
        if any("warpx2" in op for op in ops):
            hits[name] = ops
            print(f"{name}: {ops}")

    if not hits:
      print("No warpx2 layouts found in candidate set.")


if __name__ == "__main__":
    main()
