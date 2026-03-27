#!/usr/bin/env python3
"""Additive descriptor-offset probe around the clean `%rd3` warpx2 seed.

This approximates alternate `smemLoad(a, b)` source-coordinate origins by
adding small signed deltas to the clean `%rd3` descriptor seed, while keeping
the same opcode family and the clean address seeds that already produce exact
right-half writes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
DESC_VALUES_PATH = Path(__file__).with_name("probe_cp_direct_ptx_descriptor_values.py")


def _load_desc_module():
    spec = importlib.util.spec_from_file_location("probe_cp_direct_ptx_descriptor_values", DESC_VALUES_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load descriptor helper module from {DESC_VALUES_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


DESC = _load_desc_module()

DEFAULT_ADDITIVE_DELTAS = tuple(range(33))
DEFAULT_ADDR_EXPRS = ("%r18 + 4", "%r7 + 0")
DEFAULT_TARGET_CHUNK = 3
DEFAULT_TARGET_HALF = 0
SEED_DESC_REG = "%rd3"
SEED_ACTIVE_SITE = 1


@dataclass(frozen=True)
class AdditivePlan:
    opcode: str
    addr_expr: str
    additive_delta: int


def parse_int_csv(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item, 0))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_str_csv(text: str) -> List[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one string")
    return values


def candidate_plans(addr_exprs: Sequence[str], additive_deltas: Sequence[int]) -> Iterable[AdditivePlan]:
    for opcode in (DESC.SCHEDULE.WARPX2_02_13, DESC.SCHEDULE.WARPX2_01_23):
        for addr_expr in addr_exprs:
            for additive_delta in additive_deltas:
                yield AdditivePlan(opcode=opcode, addr_expr=addr_expr, additive_delta=additive_delta)


def run_child(plan_payload: str, base_ptx_path: Path, shared_bytes: int) -> Dict[str, object]:
    import pycuda.driver as cuda

    plan = AdditivePlan(**json.loads(plan_payload))
    base_ptx = base_ptx_path.read_text()
    desc_defs = DESC.parse_descriptor_defs(base_ptx, [SEED_DESC_REG])
    seed = desc_defs[SEED_DESC_REG]
    new_imm = seed.imm + plan.additive_delta

    descriptor_ptx = DESC.patch_descriptor_defs(base_ptx, {SEED_DESC_REG: new_imm})
    one_site_plan = DESC.DescriptorValuePlan(
        opcode=plan.opcode,
        active_site=SEED_ACTIVE_SITE,
        desc_reg=SEED_DESC_REG,
        base_reg=plan.addr_expr.split("+")[0].strip(),
        addr_imm=int(plan.addr_expr.split("+")[1].strip()),
        addr_expr=plan.addr_expr,
        patched_desc_imms={SEED_DESC_REG: new_imm},
    )
    patched_ptx = DESC.SCHEDULE.patched_ptx(
        descriptor_ptx, DESC.materialize_site_plans(base_ptx, one_site_plan)
    )
    cubin = DESC.SCHEDULE.assemble(patched_ptx, DESC.SCHEDULE.PTXAS_BLACKWELL)
    inp, expected = DESC.SCHEDULE.make_input()

    cuda.init()
    ctx = cuda.Device(0).make_context()
    try:
        out = DESC.SCHEDULE.launch(cubin, inp, shared_bytes)
    finally:
        try:
            ctx.pop()
        except Exception:
            pass

    rec = DESC.analyze_halves(out, expected)
    target = DESC.find_target_half(rec["chunk_halves"], DEFAULT_TARGET_CHUNK, DEFAULT_TARGET_HALF)
    rec["plan"] = asdict(plan)
    rec["descriptor_imm"] = new_imm
    rec["descriptor_imm_hex"] = f"{new_imm:#018x}"
    rec["target_label"] = target["label"]
    rec["target_exact"] = target["label"] == "B"
    rec["target_clean_exact"] = rec["target_exact"] and rec["num_other_halves"] == 0
    rec["target_isolated_exact"] = rec["target_clean_exact"] and rec["num_b_halves"] == 1
    return rec


def summarize(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary = {
        "total_plans": len(results),
        "pass_plans": 0,
        "fail_plans": 0,
        "target_exact_hits": 0,
        "target_clean_exact_hits": 0,
        "target_isolated_exact_hits": 0,
        "target_hit_examples": [],
        "best_target_near_misses": [],
    }
    hits = []
    near = []
    for rec in results:
        if rec["status"] != "PASS":
            summary["fail_plans"] += 1
            continue
        summary["pass_plans"] += 1
        compact = {
            "plan": rec["plan"],
            "descriptor_imm_hex": rec["descriptor_imm_hex"],
            "quadrant_signature": rec["quadrant_signature"],
            "target_label": rec["target_label"],
            "target_exact": rec["target_exact"],
            "target_clean_exact": rec["target_clean_exact"],
            "target_isolated_exact": rec["target_isolated_exact"],
            "num_b_halves": rec["num_b_halves"],
            "num_other_halves": rec["num_other_halves"],
        }
        if rec["target_exact"]:
            summary["target_exact_hits"] += 1
            if rec["target_clean_exact"]:
                summary["target_clean_exact_hits"] += 1
            if rec["target_isolated_exact"]:
                summary["target_isolated_exact_hits"] += 1
            hits.append(compact)
        else:
            near.append(compact)
    hits.sort(
        key=lambda item: (
            not item["target_isolated_exact"],
            not item["target_clean_exact"],
            item["num_other_halves"],
            -item["num_b_halves"],
            item["plan"]["opcode"],
            item["plan"]["addr_expr"],
            item["plan"]["additive_delta"],
        )
    )
    near.sort(
        key=lambda item: (
            item["target_label"] != "other",
            item["num_other_halves"],
            -item["num_b_halves"],
            item["plan"]["opcode"],
            item["plan"]["addr_expr"],
            item["plan"]["additive_delta"],
        )
    )
    summary["target_hit_examples"] = hits[:32]
    summary["best_target_near_misses"] = near[:32]
    return summary


def run_parent(args) -> Dict[str, object]:
    base_ptx = DESC.SCHEDULE.load_base_ptx()
    base_ptx_dir = Path(tempfile.mkdtemp(prefix="cp_desc_additive_base_"))
    base_ptx_path = base_ptx_dir / "base.ptx"
    base_ptx_path.write_text(base_ptx)
    seed_def = DESC.parse_descriptor_defs(base_ptx, [SEED_DESC_REG])[SEED_DESC_REG]

    results = {
        "base_sites": DESC.parse_cp_sites(base_ptx),
        "seed_desc_reg": SEED_DESC_REG,
        "seed_active_site": SEED_ACTIVE_SITE,
        "seed_desc_imm": seed_def.imm,
        "seed_desc_imm_hex": f"{seed_def.imm:#018x}",
        "plans": [],
    }

    child_env = os.environ.copy()
    pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = "python:." if not pythonpath else f"python:.:{pythonpath}"

    for idx, plan in enumerate(candidate_plans(args.addr_exprs, args.additive_deltas)):
        if args.limit and idx >= args.limit:
            break
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "--child-plan",
                json.dumps(asdict(plan), sort_keys=True),
                "--base-ptx-path",
                str(base_ptx_path),
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
        rec = json.loads(proc.stdout)
        rec["status"] = "PASS"
        results["plans"].append(rec)

    results["summary"] = summarize(results["plans"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-plan", type=str, default="")
    parser.add_argument("--base-ptx-path", type=Path, default=Path())
    parser.add_argument("--shared-bytes", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--additive-deltas", type=parse_int_csv, default=list(DEFAULT_ADDITIVE_DELTAS))
    parser.add_argument("--addr-exprs", type=parse_str_csv, default=list(DEFAULT_ADDR_EXPRS))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT
        / ".codex"
        / "initiatives"
        / "tmem_linear_generalization"
        / "experiments"
        / "results"
        / "probe_cp_direct_ptx_descriptor_additive_gpu0.json",
    )
    args = parser.parse_args()

    if args.child_plan:
        if not args.base_ptx_path:
            raise SystemExit("--base-ptx-path is required with --child-plan")
        try:
            rec = run_child(args.child_plan, args.base_ptx_path, args.shared_bytes)
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
