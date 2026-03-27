#!/usr/bin/env python3
"""Bounded direct-PTX overlay sweep across both second-copy base registers.

This reuses the existing direct PTX overlay probe, but extends the candidate
matrix so the patched second copy can use either base `%r` register from the
two original second-copy sites. By default it only sweeps aligned immediates to
keep the experiment bounded and to avoid known misaligned-address traps from
dominating the results.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
OVERLAY_PATH = Path(__file__).with_name("probe_cp_direct_ptx_overlay.py")
DEFAULT_ADDR_IMMS = (0, 4, 8, 12)


def _load_overlay_module():
    spec = importlib.util.spec_from_file_location("probe_cp_direct_ptx_overlay", OVERLAY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load overlay helper module from {OVERLAY_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


OVERLAY = _load_overlay_module()


@dataclass(frozen=True)
class DualBaseOverlayPlan:
    opcode: str
    active_site: int
    desc_reg: str
    base_site: int
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


def extract_second_copy_base_options(base_ptx: str) -> List[Dict[str, object]]:
    second_pair = OVERLAY.parse_cp_sites(base_ptx)[2:4]
    options = []
    seen = set()
    for site_idx, (_line_idx, _indent, _pred, addr_expr, desc_reg) in enumerate(second_pair):
        match = OVERLAY.ADDR_RE.match(addr_expr)
        if not match:
            raise RuntimeError(f"failed to parse address expression: {addr_expr}")
        base_reg = match.group(1)
        base_imm = int(match.group(2))
        if base_reg in seen:
            continue
        seen.add(base_reg)
        options.append(
            {
                "base_site": site_idx,
                "base_reg": base_reg,
                "base_site_addr_expr": addr_expr,
                "base_site_desc_reg": desc_reg,
                "base_site_imm": base_imm,
            }
        )
    if len(options) < 2:
        raise RuntimeError(f"expected at least two base-register options, found {options}")
    return options


def candidate_plans(base_ptx: str, addr_imms: Sequence[int]) -> Iterable[DualBaseOverlayPlan]:
    second_pair = OVERLAY.parse_cp_sites(base_ptx)[2:4]
    base_options = extract_second_copy_base_options(base_ptx)
    desc_regs = []
    for _line_idx, _indent, _pred, _addr_expr, desc_reg in second_pair:
        if desc_reg not in desc_regs:
            desc_regs.append(desc_reg)

    for opcode in (OVERLAY.WARPX2_02_13, OVERLAY.WARPX2_01_23):
        for active_site in (0, 1):
            for desc_reg in desc_regs:
                for base in base_options:
                    for addr_imm in addr_imms:
                        yield DualBaseOverlayPlan(
                            opcode=opcode,
                            active_site=active_site,
                            desc_reg=desc_reg,
                            base_site=int(base["base_site"]),
                            base_reg=str(base["base_reg"]),
                            addr_imm=addr_imm,
                            addr_expr=f"{base['base_reg']} + {addr_imm}",
                        )


def make_overlay_plan(plan: DualBaseOverlayPlan):
    return OVERLAY.OverlayPlan(plan.opcode, plan.active_site, plan.addr_expr, plan.desc_reg)


def quadrant_signature(chunk_halves: Sequence[Dict[str, object]]) -> str:
    labels = []
    for half in chunk_halves:
        label = str(half["label"])
        labels.append("O" if label == "other" else label)
    return "/".join("".join(labels[i : i + 2]) for i in range(0, len(labels), 2))


def summarize_plans(plan_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "total_plans": len(plan_results),
        "pass_plans": 0,
        "fail_plans": 0,
        "overwrite_pass_plans": 0,
        "clean_quadrant_mapping_plans": 0,
        "by_base_reg": {},
    }
    by_base_reg: Dict[str, Dict[str, object]] = {}
    for rec in plan_results:
        plan = rec["plan"]
        base_reg = plan["base_reg"]
        base_summary = by_base_reg.setdefault(
            base_reg,
            {
                "total_plans": 0,
                "pass_plans": 0,
                "fail_plans": 0,
                "overwrite_pass_plans": 0,
                "clean_quadrant_mapping_plans": 0,
                "clean_quadrant_signatures": [],
                "interesting_passes": [],
            },
        )
        base_summary["total_plans"] += 1
        if rec["status"] != "PASS":
            summary["fail_plans"] += 1
            base_summary["fail_plans"] += 1
            continue

        summary["pass_plans"] += 1
        base_summary["pass_plans"] += 1
        if rec["num_b_halves"]:
            summary["overwrite_pass_plans"] += 1
            base_summary["overwrite_pass_plans"] += 1
        if rec["clean_quadrant_mapping"]:
            summary["clean_quadrant_mapping_plans"] += 1
            base_summary["clean_quadrant_mapping_plans"] += 1
            if rec["quadrant_signature"] not in base_summary["clean_quadrant_signatures"]:
                base_summary["clean_quadrant_signatures"].append(rec["quadrant_signature"])

        if rec["num_b_halves"] or rec["num_other_halves"]:
            base_summary["interesting_passes"].append(
                {
                    "opcode": plan["opcode"],
                    "active_site": plan["active_site"],
                    "desc_reg": plan["desc_reg"],
                    "base_site": plan["base_site"],
                    "addr_expr": plan["addr_expr"],
                    "quadrant_signature": rec["quadrant_signature"],
                    "num_b_halves": rec["num_b_halves"],
                    "num_other_halves": rec["num_other_halves"],
                }
            )

    for base_reg, base_summary in by_base_reg.items():
        base_summary["clean_quadrant_signatures"].sort()
        base_summary["interesting_passes"].sort(
            key=lambda item: (
                item["opcode"],
                item["active_site"],
                item["desc_reg"],
                item["base_site"],
                item["addr_expr"],
            )
        )
        summary["by_base_reg"][base_reg] = base_summary
    return summary


def run_parent(args) -> Dict[str, object]:
    base_info = OVERLAY.compile_base_ptx()
    base_ptx = base_info["ptx"]
    base_sites = OVERLAY.parse_cp_sites(base_ptx)
    second_copy_sites = base_sites[2:4]
    base_options = extract_second_copy_base_options(base_ptx)

    a_input = np.full((64, 16), -77, dtype=np.int8)
    b_input = np.broadcast_to(np.arange(16, dtype=np.int8), (64, 16)).copy()
    a_expected = OVERLAY.expected_warpx4_output(a_input)
    b_expected = OVERLAY.expected_warpx4_output(b_input)

    results: Dict[str, object] = {
        "addr_imms": list(args.addr_imms),
        "base_sites": base_sites,
        "second_copy_sites": second_copy_sites,
        "base_reg_options": base_options,
        "plans": [],
    }

    for plan in candidate_plans(base_ptx, args.addr_imms):
        patched = OVERLAY.patch_second_copy(base_ptx, make_overlay_plan(plan))
        cubin = OVERLAY.assemble(patched, args.ptxas)
        try:
            out = np.array(OVERLAY.run_child(cubin)["out"], dtype=np.int8)
        except Exception as exc:
            results["plans"].append({"plan": asdict(plan), "status": "FAIL", "error": str(exc)})
            continue

        rec = {"plan": asdict(plan), "status": "PASS"}
        rec.update(OVERLAY.analyze(out, a_expected, b_expected))
        rec["quadrant_signature"] = quadrant_signature(rec["chunk_halves"])
        rec["num_b_halves"] = sum(half["label"] == "B" for half in rec["chunk_halves"])
        rec["num_other_halves"] = sum(half["label"] == "other" for half in rec["chunk_halves"])
        rec["clean_quadrant_mapping"] = rec["num_other_halves"] == 0 and rec["num_b_halves"] > 0
        results["plans"].append(rec)

    results["summary"] = summarize_plans(results["plans"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptxas", type=Path, default=OVERLAY.PTXAS_BLACKWELL)
    parser.add_argument(
        "--addr-imms",
        type=parse_addr_imms,
        default=list(DEFAULT_ADDR_IMMS),
        help="comma-separated immediate offsets to try for each base register (default: 0,4,8,12)",
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
        / "probe_cp_direct_ptx_overlay_both_bases_gpu0.json",
    )
    args = parser.parse_args()

    results = run_parent(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
