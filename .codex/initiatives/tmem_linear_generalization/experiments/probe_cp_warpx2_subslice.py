#!/usr/bin/env python3
"""Probe whether tcgen05.cp.warpx2 is reachable via shared-memdesc subslices."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import os
import re
import sys
import tempfile

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
from triton.tools import LinearLayout


CP_OPCODE_RE = re.compile(
    r"(tcgen05\.cp(?:\.cta_group::\d+)?(?:\.warpx[24](?:::[^\s.;]+)*)?\.\d+x\d+b)"
)


def extract_tcgen05_cp_opcodes(asm: str):
    return CP_OPCODE_RE.findall(asm)


@contextlib.contextmanager
def capture_fd_output():
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    capture = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(capture.fileno(), 1)
        os.dup2(capture.fileno(), 2)
        yield capture
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        capture.seek(0)


@gluon.jit
def probe_scales_copy_subslice_kernel(
    in_ptr,
    out_ptr,
    parent_layout: ttgl.constexpr,
    parent_rows: ttgl.constexpr,
    start_row: ttgl.constexpr,
):
    smem_w: ttgl.constexpr = 16
    copy_rows: ttgl.constexpr = 64
    out_rows: ttgl.constexpr = 128
    out_cols: ttgl.constexpr = (copy_rows * smem_w) // 32

    blocked_parent: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    in_ptrs = in_ptr + ttgl.arange(0, parent_rows)[:, None] * smem_w + ttgl.arange(0, smem_w)[None, :]
    parent_value = ttgl.load(ttgl.set_auto_layout(in_ptrs, blocked_parent))

    smem_parent = ttgl.allocate_shared_memory(ttgl.int8, (parent_rows, smem_w), layout=parent_layout)
    smem_parent.store(parent_value)
    smem = smem_parent.slice(start_row, copy_rows, dim=0)

    tmem = allocate_tensor_memory(ttgl.int8, (copy_rows, smem_w), layout=TensorMemoryScalesLayout())
    barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(barrier)
    mbarrier.wait(barrier, phase=0)

    blocked_out: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
    out_ptrs = out_ptr + ttgl.arange(0, out_rows)[:, None] * out_cols + ttgl.arange(0, out_cols)[None, :]
    alias_layout: ttgl.constexpr = TensorMemoryLayout((out_rows, out_cols), col_stride=1)
    alias_view = tmem._reinterpret(ttgl.int8, (out_rows, out_cols), alias_layout)
    out_value = alias_view.load(blocked_out)
    ttgl.store(ttgl.set_auto_layout(out_ptrs, blocked_out), out_value)


def iter_interleavings(row_bases, col_bases):
    total = len(row_bases) + len(col_bases)
    for chosen_cols in itertools.combinations(range(total), len(col_bases)):
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


def is_surjective_layout(bases, shape):
    ll = LinearLayout.from_bases([("offset", bases)], ["dim0", "dim1"], list(shape))
    return ll.is_surjective


def search(parent_rows: int, max_layouts: int, start_rows: list[int]):
    row_pows = [1 << i for i in range(parent_rows.bit_length() - 1)]
    # Probe "special high row" reorderings first because multicast bits are taken
    # from row32/row64 in the converter.
    low_rows = [(r, 0) for r in row_pows[:-1]]
    special_row = (row_pows[-1], 0)
    cols = [(0, 1), (0, 2), (0, 4), (0, 8)]

    layouts = []
    for insert_pos in range(len(low_rows) + 1):
        row_order = list(low_rows)
        row_order.insert(insert_pos, special_row)
        for bases in iter_interleavings(row_order, cols):
            if not is_surjective_layout([list(b) for b in bases], (parent_rows, 16)):
                continue
            layouts.append([list(b) for b in bases])
            if len(layouts) >= max_layouts:
                return layouts
    return layouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-rows", type=int, default=128)
    parser.add_argument("--max-layouts", type=int, default=384)
    parser.add_argument("--start-rows", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--show-failure-snippets", action="store_true")
    args = parser.parse_args()

    if not is_blackwell():
        raise SystemExit("Blackwell GPU required")

    parent_rows = args.parent_rows
    smem_w = 16
    if args.start_rows:
        start_rows = [int(tok) for tok in args.start_rows.split(",") if tok]
    else:
        start_rows = [0]
        if parent_rows >= 96:
            start_rows.append(32)
        if parent_rows >= 128:
            start_rows.append(64)

    layouts = search(parent_rows, args.max_layouts, start_rows)
    print(f"candidate_layouts={len(layouts)} parent_rows={parent_rows} starts={start_rows}")

    inp = torch.randint(-100, 100, (parent_rows, smem_w), dtype=torch.int8, device=args.device)
    out = torch.empty((128, 32), dtype=torch.int8, device=args.device)

    success = 0
    clean_unsupported = 0
    bug_like = 0
    unknown_failures = 0
    failure_examples = []
    warpx2_hits = []
    opcode_hist = {}
    for idx, bases in enumerate(layouts):
        layout = ttgl.SharedLinearLayout(offset_bases=bases, alignment=16)
        for start_row in start_rows:
            compile_exc = None
            with capture_fd_output() as captured_output:
                try:
                    compiled = probe_scales_copy_subslice_kernel[(1,)](
                        inp,
                        out,
                        layout,
                        parent_rows,
                        start_row,
                        num_warps=4,
                    )
                except Exception as exc:
                    compile_exc = exc
            diagnostic_text = captured_output.read().decode("utf-8", errors="replace")
            captured_output.close()
            if compile_exc is not None:
                failure_text = f"{compile_exc}\n{diagnostic_text}"
                clean_tokens = (
                    "failed to find valid tcgen05.copy layout",
                    "could not synthesize a compatible shared-memory descriptor plan",
                    "This is reported as cleanly unsupported",
                    "The split offset may not touch the tile",
                )
                if any(token in failure_text for token in clean_tokens):
                    clean_unsupported += 1
                elif any(
                    token in failure_text.lower()
                    for token in ("assert", "segmentation fault", "stack dump", "aborted")
                ):
                    bug_like += 1
                else:
                    unknown_failures += 1
                if len(failure_examples) < 5:
                    failure_examples.append((idx, start_row, bases, failure_text.splitlines()[:6]))
                continue
            success += 1
            opcodes = tuple(extract_tcgen05_cp_opcodes(compiled.asm["ptx"]))
            opcode_hist[opcodes] = opcode_hist.get(opcodes, 0) + 1
            if any("warpx2" in op for op in opcodes):
                warpx2_hits.append((idx, start_row, bases, opcodes))

    print(f"successful_compiles={success}")
    print(
        "failures "
        f"clean_unsupported={clean_unsupported} bug_like={bug_like} unknown={unknown_failures}"
    )
    for ops, count in sorted(opcode_hist.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{count} {ops}")
    if warpx2_hits:
        print("warpx2 hits:")
        for idx, start_row, bases, ops in warpx2_hits:
            print(f"layout_idx={idx} start_row={start_row} ops={ops} bases={bases}")
    else:
        print("no warpx2 opcodes observed")
    if args.show_failure_snippets and failure_examples:
        print("failure snippets:")
        for idx, start_row, bases, lines in failure_examples:
            print(f"layout_idx={idx} start_row={start_row} bases={bases}")
            for line in lines:
                print(f"  {line}")


if __name__ == "__main__":
    main()
