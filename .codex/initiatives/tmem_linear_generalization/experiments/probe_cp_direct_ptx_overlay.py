#!/usr/bin/env python3
"""Overlay probe for tcgen05.cp.warpx2.

Build a kernel with two sequential scales-style tcgen05.copy operations:
  1. a known-good warpx4 copy from sentinel input A
  2. a second copy from diagnostic input B

The first copy seeds TMEM with a known pattern. The second copy is patched
site-by-site to warpx2 with only one site enabled, so we can see exactly which
parts of the alias view get overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
PTXAS_BLACKWELL = REPO_ROOT / "third_party" / "nvidia" / "backend" / "bin" / "ptxas-blackwell"
WARPX4 = "tcgen05.cp.cta_group::1.warpx4.32x128b"
WARPX2_02_13 = "tcgen05.cp.cta_group::1.warpx2::02_13.64x128b"
WARPX2_01_23 = "tcgen05.cp.cta_group::1.warpx2::01_23.64x128b"
KERNEL_NAME = "overlay_copy_kernel"

CP_RE = re.compile(
    r"^(\s*)(@%p\d+)\s+tcgen05\.cp[^\[]*\[\s*([^\]]+?)\s*\]\s*,\s*(%rd\d+)\s*;\s*$"
)
ADDR_RE = re.compile(r"(%r\d+)\s*\+\s*(\d+)")


@dataclass(frozen=True)
class OverlayPlan:
    opcode: str
    active_site: int
    addr_expr: str
    desc_reg: str


def kernel_source() -> str:
    return textwrap.dedent(
        """
        import torch
        from triton.experimental import gluon
        from triton.experimental.gluon import language as ttgl
        from triton.experimental.gluon.language.nvidia.blackwell import (
            TensorMemoryLayout,
            TensorMemoryScalesLayout,
            allocate_tensor_memory,
            fence_async_shared,
            tcgen05_commit,
            tcgen05_copy,
        )
        from triton.experimental.gluon.language.nvidia.hopper import mbarrier

        @gluon.jit
        def overlay_copy_kernel(in0_ptr, in1_ptr, out_ptr):
            SMEM_H: ttgl.constexpr = 64
            SMEM_W: ttgl.constexpr = 16
            NUM_ROWS: ttgl.constexpr = 128
            NUM_COLS: ttgl.constexpr = (SMEM_H * SMEM_W) // 32

            in0_ptrs = in0_ptr + ttgl.arange(0, SMEM_H)[:, None] * SMEM_W + ttgl.arange(0, SMEM_W)[None, :]
            in1_ptrs = in1_ptr + ttgl.arange(0, SMEM_H)[:, None] * SMEM_W + ttgl.arange(0, SMEM_W)[None, :]
            out_ptrs = out_ptr + ttgl.arange(0, NUM_ROWS)[:, None] * NUM_COLS + ttgl.arange(0, NUM_COLS)[None, :]

            blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [32, 1], [4, 1], [1, 0])
            value0 = ttgl.load(ttgl.set_auto_layout(in0_ptrs, blocked))
            value1 = ttgl.load(ttgl.set_auto_layout(in1_ptrs, blocked))

            smem_layout: ttgl.constexpr = ttgl.SharedLinearLayout(
                offset_bases=[[0, 1], [0, 2], [32, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]]
            )
            tmem = allocate_tensor_memory(ttgl.int8, (SMEM_H, SMEM_W), layout=TensorMemoryScalesLayout())
            smem = ttgl.allocate_shared_memory(ttgl.int8, (SMEM_H, SMEM_W), layout=smem_layout)
            barrier0 = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
            barrier1 = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
            mbarrier.init(barrier0, count=1)
            mbarrier.init(barrier1, count=1)

            smem.store(value0)
            fence_async_shared()
            tcgen05_copy(smem, tmem)
            tcgen05_commit(barrier0)
            mbarrier.wait(barrier0, phase=0)

            smem.store(value1)
            fence_async_shared()
            tcgen05_copy(smem, tmem)
            tcgen05_commit(barrier1)
            mbarrier.wait(barrier1, phase=0)

            tmem_alias: ttgl.constexpr = TensorMemoryLayout((NUM_ROWS, NUM_COLS), col_stride=1)
            alias_view = tmem._reinterpret(ttgl.int8, (NUM_ROWS, NUM_COLS), tmem_alias)
            value = alias_view.load(blocked)
            ttgl.store(ttgl.set_auto_layout(out_ptrs, blocked), value)
        """
    )


def compile_base_ptx() -> Dict[str, object]:
    script = kernel_source() + "\n" + textwrap.dedent(
        """
        import json
        import torch

        a = torch.full((64, 16), -77, dtype=torch.int8, device="cuda")
        b = torch.arange(16, dtype=torch.int8, device="cuda").expand(64, 16).clone()
        out = torch.zeros((128, 32), dtype=torch.int8, device="cuda")
        compiled = overlay_copy_kernel[(1,)](a, b, out)
        print(json.dumps({"ptx": compiled.asm["ptx"]}))
        """
    )
    script_path = Path(tempfile.mkdtemp(prefix="cp_overlay_compile_")) / "compile_overlay.py"
    script_path.write_text(script)
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "python:." if not pythonpath else f"python:.:{pythonpath}"
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"compile child exited {proc.returncode}")
    return json.loads(proc.stdout)


def expected_warpx4_output(inp: np.ndarray) -> np.ndarray:
    tile = inp.reshape(2, 32, 2, 2, 4).transpose(1, 2, 3, 0, 4).reshape(32, 32)
    return np.concatenate([tile, tile, tile, tile], axis=0)


def parse_cp_sites(ptx_text: str) -> List[tuple[int, str, str, str, str]]:
    sites = []
    lines = ptx_text.splitlines()
    for idx, line in enumerate(lines):
        if WARPX4 not in line:
            continue
        m = CP_RE.match(line)
        if not m:
            raise RuntimeError(f"failed to parse cp line: {line}")
        indent, pred, addr, desc = m.groups()
        sites.append((idx, indent, pred, addr, desc))
    if len(sites) != 4:
        raise RuntimeError(f"expected 4 cp sites, found {len(sites)}")
    return sites


def make_cp_line(indent: str, pred: str, opcode: str, addr_expr: str, desc_reg: str) -> str:
    return f"{indent}{pred} {opcode} [ {addr_expr} ], {desc_reg};"


def make_nop_line(indent: str, pred: str) -> str:
    return f"{indent}{pred} mov.u32 %r0, %r0;"


def patch_second_copy(base_ptx: str, plan: OverlayPlan) -> str:
    lines = base_ptx.splitlines()
    sites = parse_cp_sites(base_ptx)
    second_pair = sites[2:4]
    for idx, (line_idx, indent, pred, _addr, _desc) in enumerate(second_pair):
        if idx == plan.active_site:
            lines[line_idx] = make_cp_line(indent, pred, plan.opcode, plan.addr_expr, plan.desc_reg)
        else:
            lines[line_idx] = make_nop_line(indent, pred)
    return "\n".join(lines) + "\n"


def assemble(ptx_text: str, ptxas: Path) -> Path:
    work = Path(tempfile.mkdtemp(prefix="cp_overlay_"))
    ptx_path = work / "kernel.ptx"
    cubin_path = work / "kernel.cubin"
    ptx_path.write_text(ptx_text)
    subprocess.check_call([str(ptxas), str(ptx_path), "-arch=sm_103a", "-o", str(cubin_path)])
    return cubin_path


def run_child(cubin_path: Path) -> Dict[str, object]:
    runner = textwrap.dedent(
        """
        import os
        import json
        from pathlib import Path
        import numpy as np
        import pycuda.driver as cuda

        KERNEL_NAME = "overlay_copy_kernel"

        def launch(cubin_path: Path):
            cuda.init()
            ctx = cuda.Device(0).make_context()
            try:
                mod = cuda.module_from_file(str(cubin_path))
                fn = mod.get_function(KERNEL_NAME)
                a = np.full((64, 16), -77, dtype=np.int8)
                b = np.broadcast_to(np.arange(16, dtype=np.int8), (64, 16)).copy()
                out = np.zeros((128, 32), dtype=np.int8)
                d_a = cuda.mem_alloc(a.nbytes)
                d_b = cuda.mem_alloc(b.nbytes)
                d_out = cuda.mem_alloc(out.nbytes)
                cuda.memcpy_htod(d_a, a)
                cuda.memcpy_htod(d_b, b)
                cuda.memcpy_htod(d_out, out)
                fn(d_a, d_b, d_out, block=(128, 1, 1), grid=(1, 1, 1), shared=2048)
                cuda.Context.synchronize()
                cuda.memcpy_dtoh(out, d_out)
                return out
            finally:
                try:
                    ctx.pop()
                except Exception:
                    pass

        out = launch(Path(os.environ["CUBIN"]))
        print(json.dumps({"out": out.tolist()}))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", runner],
        cwd=str(REPO_ROOT),
        env={**os.environ, "CUBIN": str(cubin_path)},
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"child exited {proc.returncode}")
    return json.loads(proc.stdout)


def analyze(out: np.ndarray, a_expected: np.ndarray, b_expected: np.ndarray) -> Dict[str, object]:
    halves = []
    for chunk in range(4):
        row_slice = slice(chunk * 32, (chunk + 1) * 32)
        for half in range(2):
            col_slice = slice(half * 16, (half + 1) * 16)
            tile = out[row_slice, col_slice]
            a_tile = a_expected[row_slice, col_slice]
            b_tile = b_expected[row_slice, col_slice]
            if np.array_equal(tile, a_tile):
                label = "A"
            elif np.array_equal(tile, b_tile):
                label = "B"
            else:
                label = "other"
            halves.append(
                {
                    "chunk": chunk,
                    "half": half,
                    "label": label,
                    "matches_a": int(np.count_nonzero(tile == a_tile)),
                    "matches_b": int(np.count_nonzero(tile == b_tile)),
                }
            )
    return {
        "chunk_halves": halves,
        "num_b_matches_total": int(np.count_nonzero(out == b_expected)),
        "num_a_matches_total": int(np.count_nonzero(out == a_expected)),
    }


def candidate_plans(base_ptx: str) -> Iterable[OverlayPlan]:
    sites = parse_cp_sites(base_ptx)
    second_pair = sites[2:4]
    addr_match = ADDR_RE.match(second_pair[0][3])
    if not addr_match:
        raise RuntimeError(f"failed to parse address expression: {second_pair[0][3]}")
    base_reg = addr_match.group(1)
    addr_exprs = [f"{base_reg} + {imm}" for imm in (0, 1, 2, 4, 8, 12)]
    desc_regs = [second_pair[0][4], second_pair[1][4]]
    for opcode in (WARPX2_02_13, WARPX2_01_23):
        for active_site in (0, 1):
            for desc_reg in desc_regs:
                for addr_expr in addr_exprs:
                    yield OverlayPlan(opcode, active_site, addr_expr, desc_reg)


def run_parent(args):
    base_info = compile_base_ptx()
    base_ptx = base_info["ptx"]
    a_input = np.full((64, 16), -77, dtype=np.int8)
    b_input = np.broadcast_to(np.arange(16, dtype=np.int8), (64, 16)).copy()
    a_expected = expected_warpx4_output(a_input)
    b_expected = expected_warpx4_output(b_input)
    results = {
        "base_sites": parse_cp_sites(base_ptx),
        "plans": [],
    }
    for plan in candidate_plans(base_ptx):
        patched = patch_second_copy(base_ptx, plan)
        cubin = assemble(patched, args.ptxas)
        try:
            out = np.array(run_child(cubin)["out"], dtype=np.int8)
        except Exception as exc:
            results["plans"].append({"plan": asdict(plan), "status": "FAIL", "error": str(exc)})
            continue
        rec = {"plan": asdict(plan), "status": "PASS"}
        rec.update(analyze(out, a_expected, b_expected))
        results["plans"].append(rec)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptxas", type=Path, default=PTXAS_BLACKWELL)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT
        / ".codex"
        / "initiatives"
        / "tmem_linear_generalization"
        / "experiments"
        / "results"
        / "probe_cp_direct_ptx_overlay_gpu0.json",
    )
    args = parser.parse_args()

    results = run_parent(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
