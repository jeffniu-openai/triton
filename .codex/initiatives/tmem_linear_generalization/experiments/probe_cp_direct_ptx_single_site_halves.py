#!/usr/bin/env python3
"""Bounded one-site direct-PTX sweep for tcgen05.cp.warpx2.

This reuses the existing `probe_cp_direct_ptx_schedule.py` machinery, but
focuses on one active site at a time and classifies the resulting output by
32x16 half-chunks against the exact expected payload and the zero-initialized
baseline. The goal is to answer one narrow question: does the current one-copy
descriptor pair expose any clean half-tile writes that the overlay probe on the
second-copy descriptor pair does not?
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEDULE_PATH = Path(__file__).with_name("probe_cp_direct_ptx_schedule.py")
DEFAULT_ADDR_IMMS = (0, 4)
ADDR_RE = re.compile(r"(%r\d+)\s*\+\s*(\d+)")


def _load_schedule_module():
    spec = importlib.util.spec_from_file_location("probe_cp_direct_ptx_schedule", SCHEDULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load schedule helper module from {SCHEDULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SCHEDULE = _load_schedule_module()


@dataclass(frozen=True)
class SingleSitePlan:
    opcode: str
    active_site: int
    desc_reg: str
    base_reg: str
    addr_imm: int
    addr_expr: str


def parse_addr_imms(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item, 0))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one address immediate")
    return values


def parse_cp_sites(base_ptx: str):
    return SCHEDULE.parse_cp_sites(base_ptx.splitlines())


def base_reg_options(base_ptx: str) -> List[str]:
    regs = []
    for _idx, _indent, _pred, addr_expr, _desc in parse_cp_sites(base_ptx):
        match = ADDR_RE.match(addr_expr)
        if not match:
            raise RuntimeError(f"failed to parse address expression: {addr_expr}")
        reg = match.group(1)
        if reg not in regs:
            regs.append(reg)
    return regs


def desc_reg_options(base_ptx: str) -> List[str]:
    regs = []
    for _idx, _indent, _pred, _addr_expr, desc_reg in parse_cp_sites(base_ptx):
        if desc_reg not in regs:
            regs.append(desc_reg)
    return regs


def candidate_plans(base_ptx: str, addr_imms: Sequence[int]) -> Iterable[SingleSitePlan]:
    for opcode in (SCHEDULE.WARPX2_02_13, SCHEDULE.WARPX2_01_23):
        for active_site in (0, 1):
            for desc_reg in desc_reg_options(base_ptx):
                for base_reg in base_reg_options(base_ptx):
                    for addr_imm in addr_imms:
                        yield SingleSitePlan(
                            opcode=opcode,
                            active_site=active_site,
                            desc_reg=desc_reg,
                            base_reg=base_reg,
                            addr_imm=addr_imm,
                            addr_expr=f"{base_reg} + {addr_imm}",
                        )


def disabled_plan_template(base_ptx: str, active_site: int):
    sites = parse_cp_sites(base_ptx)
    other_idx = 1 - active_site
    _idx, _indent, _pred, addr_expr, desc_reg = sites[other_idx]
    return SCHEDULE.SitePlan(False, SCHEDULE.WARPX2_02_13, addr_expr, desc_reg)


def materialize_site_plans(base_ptx: str, plan: SingleSitePlan):
    disabled = disabled_plan_template(base_ptx, plan.active_site)
    active = SCHEDULE.SitePlan(True, plan.opcode, plan.addr_expr, plan.desc_reg)
    if plan.active_site == 0:
        return (active, disabled)
    return (disabled, active)


def run_child(plan_payload: str, shared_bytes: int) -> Dict[str, object]:
    import pycuda.driver as cuda

    plans = tuple(SCHEDULE.SitePlan(**plan) for plan in json.loads(plan_payload))
    base_ptx = SCHEDULE.load_base_ptx()
    inp, expected = SCHEDULE.make_input()
    ptx = SCHEDULE.patched_ptx(base_ptx, plans)
    cubin = SCHEDULE.assemble(ptx, SCHEDULE.PTXAS_BLACKWELL)
    cuda.init()
    ctx = cuda.Device(0).make_context()
    try:
        out = SCHEDULE.launch(cubin, inp, shared_bytes)
    finally:
        try:
            ctx.pop()
        except Exception:
            pass
    return {"out": out.tolist(), "expected": expected.tolist()}


def analyze_halves(out, expected) -> Dict[str, object]:
    import numpy as np

    out = np.array(out, dtype=np.int8)
    expected = np.array(expected, dtype=np.int8)
    zero = np.zeros_like(expected)
    halves = []
    for chunk in range(4):
        row_slice = slice(chunk * 32, (chunk + 1) * 32)
        for half in range(2):
            col_slice = slice(half * 16, (half + 1) * 16)
            tile = out[row_slice, col_slice]
            expected_tile = expected[row_slice, col_slice]
            zero_tile = zero[row_slice, col_slice]
            if (tile == expected_tile).all():
                label = "B"
            elif (tile == zero_tile).all():
                label = "0"
            else:
                label = "other"
            halves.append(
                {
                    "chunk": chunk,
                    "half": half,
                    "label": label,
                    "matches_expected": int((tile == expected_tile).sum()),
                    "matches_zero": int((tile == zero_tile).sum()),
                }
            )
    quadrant_signature = "/".join(
        "".join("O" if half["label"] == "other" else half["label"] for half in halves[i : i + 2])
        for i in range(0, len(halves), 2)
    )
    num_b_halves = sum(half["label"] == "B" for half in halves)
    num_other_halves = sum(half["label"] == "other" for half in halves)
    return {
        "chunk_halves": halves,
        "quadrant_signature": quadrant_signature,
        "num_b_halves": num_b_halves,
        "num_other_halves": num_other_halves,
        "clean_quadrant_mapping": num_other_halves == 0 and num_b_halves > 0,
    }


def summarize_plans(plan_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "total_plans": len(plan_results),
        "pass_plans": 0,
        "fail_plans": 0,
        "clean_quadrant_mapping_plans": 0,
        "interesting_passes": [],
        "unique_quadrant_signatures": [],
    }
    signatures = set()
    for rec in plan_results:
        if rec["status"] != "PASS":
            summary["fail_plans"] += 1
            continue
        summary["pass_plans"] += 1
        if rec["clean_quadrant_mapping"]:
            summary["clean_quadrant_mapping_plans"] += 1
        if rec["num_b_halves"] or rec["num_other_halves"]:
            summary["interesting_passes"].append(
                {
                    "opcode": rec["plan"]["opcode"],
                    "active_site": rec["plan"]["active_site"],
                    "desc_reg": rec["plan"]["desc_reg"],
                    "addr_expr": rec["plan"]["addr_expr"],
                    "quadrant_signature": rec["quadrant_signature"],
                    "num_b_halves": rec["num_b_halves"],
                    "num_other_halves": rec["num_other_halves"],
                }
            )
            signatures.add(rec["quadrant_signature"])
    summary["interesting_passes"].sort(
        key=lambda item: (
            item["opcode"],
            item["active_site"],
            item["desc_reg"],
            item["addr_expr"],
        )
    )
    summary["unique_quadrant_signatures"] = sorted(signatures)
    return summary


def run_parent(args) -> Dict[str, object]:
    base_ptx = SCHEDULE.load_base_ptx()
    results: Dict[str, object] = {
        "addr_imms": list(args.addr_imms),
        "base_sites": parse_cp_sites(base_ptx),
        "plans": [],
    }
    child_env = os.environ.copy()
    pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = "python:." if not pythonpath else f"python:.:{pythonpath}"

    for plan in candidate_plans(base_ptx, args.addr_imms):
        payload = json.dumps([asdict(site) for site in materialize_site_plans(base_ptx, plan)], sort_keys=True)
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "--child-plan",
                payload,
                "--shared-bytes",
                str(args.shared_bytes),
            ],
            cwd=str(REPO_ROOT),
            env=child_env,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            results["plans"].append(
                {
                    "plan": asdict(plan),
                    "status": "FAIL",
                    "error": proc.stderr.strip() or proc.stdout.strip() or f"child exited {proc.returncode}",
                }
            )
            continue

        child = json.loads(proc.stdout)
        rec = {"plan": asdict(plan), "status": "PASS"}
        rec.update(analyze_halves(child["out"], child["expected"]))
        results["plans"].append(rec)

    results["summary"] = summarize_plans(results["plans"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-plan", type=str, default="")
    parser.add_argument("--shared-bytes", type=int, default=2048)
    parser.add_argument(
        "--addr-imms",
        type=parse_addr_imms,
        default=list(DEFAULT_ADDR_IMMS),
        help="comma-separated immediate offsets to try for each base register (default: 0,4)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT
        / ".codex"
        / "initiatives"
        / "tmem_linear_generalization"
        / "experiments"
        / "results"
        / "probe_cp_direct_ptx_single_site_halves_gpu0.json",
    )
    args = parser.parse_args()

    if args.child_plan:
        try:
            result = run_child(args.child_plan, args.shared_bytes)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            raise SystemExit(1)
        print(json.dumps(result, sort_keys=True))
        return

    results = run_parent(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
