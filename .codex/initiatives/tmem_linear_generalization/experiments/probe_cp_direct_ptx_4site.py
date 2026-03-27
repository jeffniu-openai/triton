#!/usr/bin/env python3
"""Bounded four-site direct-PTX search for tcgen05.cp.warpx2.

This reuses the existing two-copy overlay kernel shape but feeds the same source
data through both copies, so the four patched tcgen05.cp sites behave like a
single four-message schedule.

Default search space is intentionally small and grounded in the existing
overlay findings:
  - all four original cp sites stay enabled
  - each site keeps its original descriptor register
  - each site keeps its original base-address register
  - each site chooses one warpx2 family
  - each site chooses an aligned destination delta of 0 or 4

Optional flags can widen descriptor or base-register choices later if the
bounded search does not find an exact plan.
"""

from __future__ import annotations

import argparse
import itertools
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
import pycuda.driver as cuda


REPO_ROOT = Path(__file__).resolve().parents[4]
PTXAS_BLACKWELL = (
    REPO_ROOT / "third_party" / "nvidia" / "backend" / "bin" / "ptxas-blackwell"
)
WARPX2_02_13 = "tcgen05.cp.cta_group::1.warpx2::02_13.64x128b"
WARPX2_01_23 = "tcgen05.cp.cta_group::1.warpx2::01_23.64x128b"
KERNEL_NAME = "overlay_copy_kernel"

CP_RE = re.compile(
    r"^(\s*)(@%p\d+)\s+tcgen05\.cp[^\[]*\[\s*([^\]]+?)\s*\]\s*,\s*(%rd\d+)\s*;\s*$"
)
ADDR_RE = re.compile(r"(%r\d+)\s*\+\s*(\d+)")


@dataclass(frozen=True)
class SitePlan:
    opcode: str
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


def compile_base_ptx() -> str:
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
    script_path = Path(tempfile.mkdtemp(prefix="cp_4site_compile_")) / "compile_4site.py"
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
        raise RuntimeError(
            proc.stderr.strip()
            or proc.stdout.strip()
            or f"compile child exited {proc.returncode}"
        )
    return json.loads(proc.stdout)["ptx"]


def parse_cp_sites(ptx_text: str) -> List[tuple[int, str, str, str, str]]:
    sites = []
    for idx, line in enumerate(ptx_text.splitlines()):
        if "tcgen05.cp.cta_group::1.warpx4.32x128b" not in line:
            continue
        m = CP_RE.match(line)
        if not m:
            raise RuntimeError(f"failed to parse cp line: {line}")
        indent, pred, addr, desc = m.groups()
        sites.append((idx, indent, pred, addr, desc))
    if len(sites) != 4:
        raise RuntimeError(f"expected exactly 4 cp sites, found {len(sites)}")
    return sites


def make_cp_line(
    indent: str, pred: str, opcode: str, addr_expr: str, desc_reg: str
) -> str:
    return f"{indent}{pred} {opcode} [ {addr_expr} ], {desc_reg};"


def patched_ptx(base_ptx: str, plans: Sequence[SitePlan]) -> str:
    lines = base_ptx.splitlines()
    sites = parse_cp_sites(base_ptx)
    if len(plans) != len(sites):
        raise RuntimeError(f"expected {len(sites)} site plans, got {len(plans)}")
    for (line_idx, indent, pred, _addr, _desc), plan in zip(sites, plans):
        lines[line_idx] = make_cp_line(
            indent, pred, plan.opcode, plan.addr_expr, plan.desc_reg
        )
    return "\n".join(lines) + "\n"


def assemble(ptx_text: str, ptxas: Path) -> Path:
    work = Path(tempfile.mkdtemp(prefix="cp_4site_"))
    ptx_path = work / "kernel.ptx"
    cubin_path = work / "kernel.cubin"
    ptx_path.write_text(ptx_text)
    subprocess.check_call(
        [str(ptxas), str(ptx_path), "-arch=sm_103a", "-o", str(cubin_path)]
    )
    return cubin_path


def expected_warpx4_output(inp: np.ndarray) -> np.ndarray:
    tile = inp.reshape(2, 32, 2, 2, 4).transpose(1, 2, 3, 0, 4).reshape(32, 32)
    return np.concatenate([tile, tile, tile, tile], axis=0)


def analyze(out: np.ndarray, expected: np.ndarray) -> Dict[str, object]:
    chunk_halves = []
    exact_half_indices = []
    half_idx = 0
    for chunk in range(4):
        row_slice = slice(chunk * 32, (chunk + 1) * 32)
        for half in range(2):
            col_slice = slice(half * 16, (half + 1) * 16)
            tile = out[row_slice, col_slice]
            exp = expected[row_slice, col_slice]
            exact = bool(np.array_equal(tile, exp))
            if exact:
                exact_half_indices.append(half_idx)
            chunk_halves.append(
                {
                    "chunk": chunk,
                    "half": half,
                    "exact_expected": exact,
                    "matches_expected": int(np.count_nonzero(tile == exp)),
                }
            )
            half_idx += 1
    return {
        "exact_expected": bool(np.array_equal(out, expected)),
        "num_mismatch_expected": int(np.count_nonzero(out != expected)),
        "exact_half_indices": exact_half_indices,
        "chunk_halves": chunk_halves,
        "sum": int(out.sum()),
    }


def parse_base_reg(addr_expr: str) -> str:
    m = ADDR_RE.match(addr_expr)
    if not m:
        raise RuntimeError(f"failed to parse address expression: {addr_expr}")
    return m.group(1)


def build_site_option_sets(
    base_sites: Sequence[tuple[int, str, str, str, str]],
    addr_deltas: Sequence[int],
    allow_cross_base: bool,
    allow_cross_desc: bool,
) -> List[List[SitePlan]]:
    opcodes = [WARPX2_02_13, WARPX2_01_23]
    base_regs = sorted({parse_base_reg(site[3]) for site in base_sites})
    desc_regs = sorted({site[4] for site in base_sites})

    site_options: List[List[SitePlan]] = []
    for _idx, _indent, _pred, addr_expr, desc_reg in base_sites:
        allowed_base_regs = base_regs if allow_cross_base else [parse_base_reg(addr_expr)]
        allowed_desc_regs = desc_regs if allow_cross_desc else [desc_reg]
        addr_exprs = [
            f"{base_reg} + {delta}"
            for base_reg in allowed_base_regs
            for delta in addr_deltas
        ]
        options = [
            SitePlan(opcode=opcode, addr_expr=site_addr, desc_reg=site_desc)
            for opcode in opcodes
            for site_desc in allowed_desc_regs
            for site_addr in addr_exprs
        ]
        site_options.append(options)
    return site_options


def candidate_plans(
    base_sites: Sequence[tuple[int, str, str, str, str]],
    addr_deltas: Sequence[int],
    allow_cross_base: bool,
    allow_cross_desc: bool,
) -> Iterable[Sequence[SitePlan]]:
    site_option_sets = build_site_option_sets(
        base_sites, addr_deltas, allow_cross_base, allow_cross_desc
    )
    yield from itertools.product(*site_option_sets)


def summarize_plan(plans: Sequence[SitePlan]) -> Dict[str, object]:
    return {f"site{i}": asdict(plan) for i, plan in enumerate(plans)}


def run_child(cubin_path: Path, inp: np.ndarray, shared_bytes: int) -> np.ndarray:
    mod = cuda.module_from_file(str(cubin_path))
    fn = mod.get_function(KERNEL_NAME)
    out = np.zeros((128, 32), dtype=np.int8)
    d_in = cuda.mem_alloc(inp.nbytes)
    d_out = cuda.mem_alloc(out.nbytes)
    cuda.memcpy_htod(d_in, inp)
    cuda.memcpy_htod(d_out, out)
    fn(d_in, d_in, d_out, block=(128, 1, 1), grid=(1, 1, 1), shared=shared_bytes)
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(out, d_out)
    return out


def run_single_plan(
    plan_payload: str,
    base_ptx_path: Path,
    ptxas: Path,
    shared_bytes: int,
    input_seed: int,
) -> Dict[str, object]:
    plans = tuple(SitePlan(**plan) for plan in json.loads(plan_payload))
    base_ptx = base_ptx_path.read_text()
    rng = np.random.default_rng(input_seed)
    inp = rng.integers(-100, 100, size=(64, 16), dtype=np.int8)
    expected = expected_warpx4_output(inp)

    cuda.init()
    ctx = cuda.Device(0).make_context()
    try:
        ptx = patched_ptx(base_ptx, plans)
        cubin = assemble(ptx, ptxas)
        out0 = run_child(cubin, inp, shared_bytes)
        out1 = run_child(cubin, inp, shared_bytes)
    finally:
        try:
            ctx.pop()
        except Exception:
            pass

    rec = analyze(out0, expected)
    rec["deterministic"] = bool(np.array_equal(out0, out1))
    rec["plan"] = summarize_plan(plans)
    rec["cubin"] = str(cubin)
    return rec


def run_parent(args) -> Dict[str, object]:
    base_ptx = compile_base_ptx()
    base_sites = parse_cp_sites(base_ptx)
    base_ptx_dir = Path(tempfile.mkdtemp(prefix="cp_4site_base_"))
    base_ptx_path = base_ptx_dir / "base.ptx"
    base_ptx_path.write_text(base_ptx)

    results: Dict[str, object] = {
        "base_cp_sites": base_sites,
        "tested": 0,
        "launch_failures": 0,
        "search_space": {
            "addr_deltas": list(args.addr_deltas),
            "allow_cross_base": args.allow_cross_base,
            "allow_cross_desc": args.allow_cross_desc,
        },
        "exact_matches": [],
        "best_by_mismatch": [],
    }
    best_records: List[Dict[str, object]] = []

    pythonpath = os.environ.get("PYTHONPATH")
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = "python:." if not pythonpath else f"python:.:{pythonpath}"

    for idx, plans in enumerate(
        candidate_plans(
            base_sites,
            args.addr_deltas,
            args.allow_cross_base,
            args.allow_cross_desc,
        )
    ):
        if args.limit and idx >= args.limit:
            break
        results["tested"] += 1
        payload = json.dumps([asdict(plan) for plan in plans], sort_keys=True)
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "--child-plan",
                payload,
                "--base-ptx-path",
                str(base_ptx_path),
                "--ptxas",
                str(args.ptxas),
                "--shared-bytes",
                str(args.shared_bytes),
                "--input-seed",
                str(args.input_seed),
            ],
            cwd=str(REPO_ROOT),
            env=child_env,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            results["launch_failures"] += 1
            best_records.append(
                {
                    "status": "FAIL",
                    "error": proc.stderr.strip()
                    or proc.stdout.strip()
                    or f"child exited {proc.returncode}",
                    "plan": summarize_plan(plans),
                }
            )
            continue
        rec = json.loads(proc.stdout)
        best_records.append(rec)
        if rec["exact_expected"]:
            results["exact_matches"].append(rec)

    successes = [r for r in best_records if r.get("status") != "FAIL"]
    successes.sort(key=lambda r: (r["num_mismatch_expected"], not r["deterministic"]))
    results["best_by_mismatch"] = successes[:32]
    return results


def parse_addr_deltas(text: str) -> List[int]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(int(item))
    if not vals:
        raise argparse.ArgumentTypeError("addr deltas must not be empty")
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptxas", type=Path, default=PTXAS_BLACKWELL)
    parser.add_argument("--shared-bytes", type=int, default=2048)
    parser.add_argument("--input-seed", type=int, default=123)
    parser.add_argument(
        "--addr-deltas",
        type=parse_addr_deltas,
        default=[0, 4],
        help="comma-separated aligned destination deltas to try per site",
    )
    parser.add_argument(
        "--allow-cross-base",
        action="store_true",
        help="allow each site to use any live base register from the base PTX",
    )
    parser.add_argument(
        "--allow-cross-desc",
        action="store_true",
        help="allow each site to use any live descriptor register from the base PTX",
    )
    parser.add_argument("--limit", type=int, default=0, help="optional cap on tried plan count")
    parser.add_argument("--child-plan", type=str, default="")
    parser.add_argument("--base-ptx-path", type=Path, default=Path())
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT
        / ".codex"
        / "initiatives"
        / "tmem_linear_generalization"
        / "experiments"
        / "results"
        / "probe_cp_direct_ptx_4site_gpu0.json",
    )
    args = parser.parse_args()

    if args.child_plan:
        if not args.base_ptx_path:
            raise SystemExit("--base-ptx-path is required with --child-plan")
        try:
            rec = run_single_plan(
                args.child_plan,
                args.base_ptx_path,
                args.ptxas,
                args.shared_bytes,
                args.input_seed,
            )
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            raise SystemExit(1)
        print(json.dumps(rec, sort_keys=True))
        return

    results = run_parent(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
