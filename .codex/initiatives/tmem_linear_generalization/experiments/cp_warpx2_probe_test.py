import itertools
import re

import pytest
import torch

from triton._internal_testing import is_blackwell
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


def _extract_tcgen05_cp_opcodes(asm: str):
    return CP_OPCODE_RE.findall(asm)


@gluon.jit
def probe_scales_copy_kernel(
    in_ptr,
    out_ptr,
    smem_layout: ttgl.constexpr,
    smem_h: ttgl.constexpr,
    smem_w: ttgl.constexpr,
    num_rows: ttgl.constexpr,
    num_cols: ttgl.constexpr,
):
    in_ptrs = in_ptr + ttgl.arange(0, smem_h)[:, None] * smem_w + ttgl.arange(0, smem_w)[None, :]
    out_ptrs = out_ptr + ttgl.arange(0, num_rows)[:, None] * num_cols + ttgl.arange(0, num_cols)[None, :]

    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    value = ttgl.load(ttgl.set_auto_layout(in_ptrs, blocked))

    smem = ttgl.allocate_shared_memory(ttgl.int8, (smem_h, smem_w), layout=smem_layout)
    tmem = allocate_tensor_memory(ttgl.int8, (smem_h, smem_w), layout=TensorMemoryScalesLayout())

    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    smem.store(value)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    alias_layout: ttgl.constexpr = blocked
    tmem_alias: ttgl.constexpr = TensorMemoryLayout((num_rows, num_cols), col_stride=1)
    alias_view = tmem._reinterpret(ttgl.int8, (num_rows, num_cols), tmem_alias)
    value = alias_view.load(alias_layout)
    ttgl.store(ttgl.set_auto_layout(out_ptrs, alias_layout), value)


@gluon.jit
def probe_no_scales_copy_kernel(
    in_ptr,
    out_ptr,
    smem_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
):
    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, blocked))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, blocked))
    offs = offs_m[:, None] * N + offs_n[None, :]

    value = ttgl.load(in_ptr + offs)
    smem = ttgl.allocate_shared_memory(ttgl.int32, [M, N], layout=smem_layout)
    tmem_layout: ttgl.constexpr = TensorMemoryLayout((M, N), col_stride=1)
    tmem = allocate_tensor_memory(ttgl.int32, [M, N], layout=tmem_layout)
    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    smem.store(value)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)
    output = tmem.load(blocked)
    ttgl.store(out_ptr + offs, output)


def _iter_interleavings(row_bases, col_bases):
    total = len(row_bases) + len(col_bases)
    col_positions = range(total)
    for chosen_cols in itertools.combinations(col_positions, len(col_bases)):
        row_idx = 0
        col_idx = 0
        seq = []
        for pos in range(total):
            if pos in chosen_cols:
                seq.append(col_bases[col_idx])
                col_idx += 1
            else:
                seq.append(row_bases[row_idx])
                row_idx += 1
        yield seq


def _iter_special_row_orders(low_rows, special_row, col_bases):
    for insert_pos in range(len(low_rows) + 1):
        row_order = list(low_rows)
        row_order.insert(insert_pos, special_row)
        yield from _iter_interleavings(row_order, col_bases)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_probe_scales_copy_warpx2_families():
    torch.manual_seed(0)

    search_spaces = [
        {
            "name": "64x16",
            "smem_shape": (64, 16),
            "alias_shape": (128, 32),
            "low_rows": [(1, 0), (2, 0), (4, 0), (8, 0), (16, 0)],
            "special_row": (32, 0),
            "cols": [(0, 1), (0, 2), (0, 4), (0, 8)],
        },
        {
            "name": "128x16",
            "smem_shape": (128, 16),
            "alias_shape": (128, 64),
            "low_rows": [(1, 0), (2, 0), (4, 0), (8, 0), (16, 0), (32, 0)],
            "special_row": (64, 0),
            "cols": [(0, 1), (0, 2), (0, 4), (0, 8)],
        },
    ]

    observed = {}
    for space in search_spaces:
        smem_h, smem_w = space["smem_shape"]
        num_rows, num_cols = space["alias_shape"]
        inp = torch.randint(-100, 100, (smem_h, smem_w), dtype=torch.int8, device="cuda")
        out = torch.empty((num_rows, num_cols), dtype=torch.int8, device="cuda")
        tried = 0
        for bases in _iter_special_row_orders(space["low_rows"], space["special_row"], space["cols"]):
            tried += 1
            layout = ttgl.SharedLinearLayout(offset_bases=[list(b) for b in bases])
            try:
                compiled = probe_scales_copy_kernel[(1,)](
                    inp, out, layout, smem_h, smem_w, num_rows, num_cols, num_warps=4
                )
            except Exception:
                continue
            opcodes = tuple(_extract_tcgen05_cp_opcodes(compiled.asm["ptx"]))
            observed.setdefault(opcodes, (space["name"], bases))
        print(space["name"], "tried", tried)

    print("observed opcode families:")
    for opcodes, info in sorted(observed.items()):
        print(opcodes, "example=", info)

    no_scales_spaces = [
        {
            "name": "64x4",
            "shape": (64, 4),
            "low_rows": [(1, 0), (2, 0), (4, 0), (8, 0), (16, 0)],
            "special_row": (32, 0),
            "cols": [(0, 1), (0, 2)],
        },
        {
            "name": "128x4",
            "shape": (128, 4),
            "low_rows": [(1, 0), (2, 0), (4, 0), (8, 0), (16, 0), (32, 0)],
            "special_row": (64, 0),
            "cols": [(0, 1), (0, 2)],
        },
    ]
    observed = {}
    for space in no_scales_spaces:
        m, n = space["shape"]
        inp = torch.arange(m * n, device="cuda", dtype=torch.int32).reshape(m, n)
        out = torch.empty_like(inp)
        tried = 0
        for bases in _iter_special_row_orders(space["low_rows"], space["special_row"], space["cols"]):
            tried += 1
            layout = ttgl.SharedLinearLayout(offset_bases=[list(b) for b in bases], alignment=16)
            try:
                compiled = probe_no_scales_copy_kernel[(1,)](inp, out, layout, m, n, num_warps=4)
            except Exception:
                continue
            opcodes = tuple(_extract_tcgen05_cp_opcodes(compiled.asm["ptx"]))
            observed.setdefault(opcodes, (space["name"], bases))
        print(space["name"], "tried", tried)

    print("observed no-scales opcode families:")
    for opcodes, info in sorted(observed.items()):
        print(opcodes, "example=", info)
