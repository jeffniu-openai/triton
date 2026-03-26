#!/usr/bin/env python3
"""Focused direct-PTX schedule probe for tcgen05.cp.warpx2.

This starts from the known-good two-site warpx4 scales-copy kernel and varies,
per copy site:
  - opcode family (`warpx2::02_13` / `warpx2::01_23`)
  - descriptor register (`%rd2` / `%rd3`)
  - TMEM destination address expression (`%r18 + imm`)
  - whether the site is enabled

Each candidate runs in a child subprocess so illegal schedules that poison the
CUDA context do not abort the whole search.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pycuda.driver as cuda
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
TEST_PATH = REPO_ROOT / "python" / "test" / "gluon" / "test_tmem_runtime_matrix.py"
PTXAS_BLACKWELL = REPO_ROOT / "third_party" / "nvidia" / "backend" / "bin" / "ptxas-blackwell"
KERNEL_NAME = "tmem_copy_scales_warpx4_kernel"
BASE_OPCODE = "tcgen05.cp.cta_group::1.warpx4.32x128b"
WARPX2_02_13 = "tcgen05.cp.cta_group::1.warpx2::02_13.64x128b"
WARPX2_01_23 = "tcgen05.cp.cta_group::1.warpx2::01_23.64x128b"

CP_RE = re.compile(
    r"^(\s*)(@%p\d+)\s+tcgen05\.cp[^\[]*\[\s*([^\]]+?)\s*\]\s*,\s*(%rd\d+)\s*;\s*$"
)


@dataclass(frozen=True)
class SitePlan:
    enabled: bool
    opcode: str
    addr_expr: str
    desc_reg: str


def load_runtime_matrix_module():
    spec = importlib.util.spec_from_file_location("tmem_runtime_matrix", TEST_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_base_ptx() -> str:
    runtime = load_runtime_matrix_module()
    torch.manual_seed(0)
    inp_t = torch.randint(-100, 100, (64, 16), dtype=torch.int8, device="cuda")
    out_t = torch.zeros((128, 32), dtype=torch.int8, device="cuda")
    compiled = runtime.tmem_copy_scales_warpx4_kernel[(1,)](inp_t, out_t)
    return compiled.asm["ptx"]


def assemble(ptx_text: str, ptxas: Path) -> Path:
    work = Path(tempfile.mkdtemp(prefix="cp_direct_schedule_"))
    ptx_path = work / "kernel.ptx"
    cubin_path = work / "kernel.cubin"
    ptx_path.write_text(ptx_text)
    subprocess.check_call([str(ptxas), str(ptx_path), "-arch=sm_103a", "-o", str(cubin_path)])
    return cubin_path


def launch(cubin_path: Path, inp: np.ndarray, shared_bytes: int) -> np.ndarray:
    mod = cuda.module_from_file(str(cubin_path))
    fn = mod.get_function(KERNEL_NAME)
    out = np.zeros((128, 32), dtype=np.int8)
    dummy = np.zeros((1,), dtype=np.int8)
    d_in = cuda.mem_alloc(inp.nbytes)
    d_out = cuda.mem_alloc(out.nbytes)
    d_d0 = cuda.mem_alloc(dummy.nbytes)
    d_d1 = cuda.mem_alloc(dummy.nbytes)
    cuda.memcpy_htod(d_in, inp)
    cuda.memcpy_htod(d_out, out)
    fn(d_in, d_out, d_d0, d_d1, block=(128, 1, 1), grid=(1, 1, 1), shared=shared_bytes)
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(out, d_out)
    return out


def parse_cp_sites(ptx_lines: Sequence[str]) -> List[tuple[int, str, str, str, str]]:
    sites = []
    for idx, line in enumerate(ptx_lines):
        if BASE_OPCODE not in line:
            continue
        m = CP_RE.match(line)
        if not m:
            raise RuntimeError(f"failed to parse cp line: {line}")
        indent, pred, addr, desc = m.groups()
        sites.append((idx, indent, pred, addr, desc))
    if len(sites) != 2:
        raise RuntimeError(f"expected exactly 2 cp sites, found {len(sites)}")
    return sites


def make_cp_line(indent: str, pred: str, opcode: str, addr_expr: str, desc_reg: str) -> str:
    return f"{indent}{pred} {opcode} [ {addr_expr} ], {desc_reg};"


def make_nop_line(indent: str, pred: str) -> str:
    return f"{indent}{pred} mov.u32 %r0, %r0;"


def patched_ptx(base_ptx: str, plans: Sequence[SitePlan]) -> str:
    lines = base_ptx.splitlines()
    sites = parse_cp_sites(lines)
    for (idx, indent, pred, _addr, _desc), plan in zip(sites, plans):
        if plan.enabled:
            lines[idx] = make_cp_line(indent, pred, plan.opcode, plan.addr_expr, plan.desc_reg)
        else:
            lines[idx] = make_nop_line(indent, pred)
    return "\n".join(lines) + "\n"


def expected_warpx4_output(inp: np.ndarray) -> np.ndarray:
    tile = inp.reshape(2, 32, 2, 2, 4).transpose(1, 2, 3, 0, 4).reshape(32, 32)
    return np.concatenate([tile, tile, tile, tile], axis=0)


def make_input() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    inp = rng.integers(-100, 100, size=(64, 16), dtype=np.int8)
    return inp, expected_warpx4_output(inp)


def chunk_output(out: np.ndarray) -> List[np.ndarray]:
    assert out.shape == (128, 32)
    return [out[i * 32:(i + 1) * 32, :] for i in range(4)]


def chunk_signature(out: np.ndarray, expected_full: np.ndarray) -> Dict[str, object]:
    chunks = chunk_output(out)
    expected_chunks = chunk_output(expected_full)
    eq = [[bool(np.array_equal(chunks[i], chunks[j])) for j in range(4)] for i in range(4)]
    return {
        "exact_expected": bool(np.array_equal(out, expected_full)),
        "num_mismatch_expected": int(np.count_nonzero(out != expected_full)),
        "chunk_equals_expected": [bool(np.array_equal(ch, expected_chunks[i])) for i, ch in enumerate(chunks)],
        "chunk_pair_equal": {
            "01": eq[0][1],
            "02": eq[0][2],
            "03": eq[0][3],
            "12": eq[1][2],
            "13": eq[1][3],
            "23": eq[2][3],
        },
        "sum": int(out.sum()),
    }


def candidate_plans() -> Iterable[Sequence[SitePlan]]:
    opcodes = [WARPX2_02_13, WARPX2_01_23]
    addr_exprs = [
        "%r18 + 0",
        "%r18 + 2",
        "%r18 + 4",
        "%r18 + 6",
        "%r7 + 0",
        "%r7 + 2",
    ]
    desc_regs = ["%rd2", "%rd3"]
    enabled_masks = [
        (True, True),
        (True, False),
        (False, True),
    ]
    for enabled0, enabled1 in enabled_masks:
        for opcode0, opcode1 in itertools.product(opcodes, repeat=2):
            for desc0, desc1 in itertools.product(desc_regs, repeat=2):
                for addr0, addr1 in itertools.product(addr_exprs, repeat=2):
                    yield (
                        SitePlan(enabled0, opcode0, addr0, desc0),
                        SitePlan(enabled1, opcode1, addr1, desc1),
                    )


def summarize_plan(plans: Sequence[SitePlan]) -> Dict[str, object]:
    return {"site0": asdict(plans[0]), "site1": asdict(plans[1])}


def run_single_plan(plan_payload: str, ptxas: Path, shared_bytes: int) -> Dict[str, object]:
    plans = tuple(SitePlan(**plan) for plan in json.loads(plan_payload))
    base_ptx = load_base_ptx()
    inp, expected_full = make_input()

    cuda.init()
    ctx = cuda.Device(0).make_context()
    try:
        ptx = patched_ptx(base_ptx, plans)
        cubin = assemble(ptx, ptxas)
        out0 = launch(cubin, inp, shared_bytes)
        out1 = launch(cubin, inp, shared_bytes)
    finally:
        try:
            ctx.pop()
        except Exception:
            pass

    rec = chunk_signature(out0, expected_full)
    rec["deterministic"] = bool(np.array_equal(out0, out1))
    rec["plan"] = summarize_plan(plans)
    rec["cubin"] = str(cubin)
    return rec


def run_parent(args) -> Dict[str, object]:
    base_ptx = load_base_ptx()
    results: Dict[str, object] = {
        "base_cp_sites": parse_cp_sites(base_ptx.splitlines()),
        "tested": 0,
        "launch_failures": 0,
        "exact_matches": [],
        "best_by_mismatch": [],
    }
    best_records: List[Dict[str, object]] = []

    pythonpath = os.environ.get("PYTHONPATH")
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = "python:." if not pythonpath else f"python:.:{pythonpath}"

    for idx, plans in enumerate(candidate_plans()):
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
                "--ptxas",
                str(args.ptxas),
                "--shared-bytes",
                str(args.shared_bytes),
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
                    "error": proc.stderr.strip() or proc.stdout.strip() or f"child exited {proc.returncode}",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptxas", type=Path, default=PTXAS_BLACKWELL)
    parser.add_argument("--shared-bytes", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="optional cap on tried plan count")
    parser.add_argument("--child-plan", type=str, default="")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT
        / ".codex"
        / "initiatives"
        / "tmem_linear_generalization"
        / "experiments"
        / "results"
        / "probe_cp_direct_ptx_schedule_gpu0.json",
    )
    args = parser.parse_args()

    if args.child_plan:
        try:
            rec = run_single_plan(args.child_plan, args.ptxas, args.shared_bytes)
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
