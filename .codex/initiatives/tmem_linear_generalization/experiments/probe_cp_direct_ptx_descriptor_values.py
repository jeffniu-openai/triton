#!/usr/bin/env python3
"""Bounded one-copy descriptor-value probe for tcgen05.cp.warpx2.

This starts from the same one-copy scales kernel used by
`probe_cp_direct_ptx_schedule.py`, but instead of only choosing among the live
descriptor registers, it patches the PTX instructions that materialize the
descriptor values for `%rd2` / `%rd3`.

Default search space:
  - one active copy site at a time
  - `warpx2::{02_13,01_23}`
  - aligned address immediates `{0,4}` over the live base registers
  - descriptor values from the 4-point bit subcube implied by the current live
    descriptor pair's differing bits

The target question is whether any one-message plan can write chunk 3, left
half (`3L`) exactly, and whether that hit is clean or isolated.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
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
DEFAULT_TARGET_CHUNK = 3
DEFAULT_TARGET_HALF = 0

ADDR_RE = re.compile(r"(%r\d+)\s*\+\s*(\d+)")
DESC_DEF_RE = re.compile(
    r"^(\s*)(or\.b64|add\.s64)\s+(%rd\d+)\s*,\s*(%rd\d+)\s*,\s*([+-]?(?:0x[0-9A-Fa-f]+|\d+))\s*;\s*$"
)


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
class DescriptorDef:
    line_idx: int
    indent: str
    op: str
    reg: str
    base_reg: str
    imm: int


@dataclass(frozen=True)
class DescriptorValuePlan:
    opcode: str
    active_site: int
    desc_reg: str
    base_reg: str
    addr_imm: int
    addr_expr: str
    patched_desc_imms: Dict[str, int]


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


def parse_optional_int_csv(text: str) -> List[int]:
    text = text.strip()
    if not text:
        return []
    return parse_int_csv(text)


def parse_cp_sites(base_ptx: str):
    return SCHEDULE.parse_cp_sites(base_ptx.splitlines())


def base_reg_options(base_ptx: str) -> List[str]:
    regs = []
    for _idx, _indent, _pred, addr_expr, _desc_reg in parse_cp_sites(base_ptx):
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


def parse_descriptor_defs(base_ptx: str, desc_regs: Sequence[str]) -> Dict[str, DescriptorDef]:
    wanted = set(desc_regs)
    defs: Dict[str, DescriptorDef] = {}
    for idx, line in enumerate(base_ptx.splitlines()):
        match = DESC_DEF_RE.match(line)
        if not match:
            continue
        indent, op, reg, base_reg, imm_text = match.groups()
        if reg not in wanted:
            continue
        defs[reg] = DescriptorDef(
            line_idx=idx,
            indent=indent,
            op=op,
            reg=reg,
            base_reg=base_reg,
            imm=int(imm_text, 0),
        )
    missing = [reg for reg in desc_regs if reg not in defs]
    if missing:
        raise RuntimeError(f"failed to find descriptor materialization lines for: {missing}")
    return defs


def make_desc_def_line(desc_def: DescriptorDef, imm: int) -> str:
    return f"{desc_def.indent}{desc_def.op} {desc_def.reg}, {desc_def.base_reg}, {imm};"


def patch_descriptor_defs(base_ptx: str, desc_imms: Dict[str, int]) -> str:
    if not desc_imms:
        return base_ptx if base_ptx.endswith("\n") else base_ptx + "\n"
    lines = base_ptx.splitlines()
    defs = parse_descriptor_defs(base_ptx, list(desc_imms.keys()))
    for reg, imm in desc_imms.items():
        desc_def = defs[reg]
        lines[desc_def.line_idx] = make_desc_def_line(desc_def, imm)
    return "\n".join(lines) + "\n"


def format_hex_map(values: Dict[str, int]) -> Dict[str, str]:
    return {key: f"{value:#018x}" for key, value in values.items()}


def format_hex_list(values: Sequence[int]) -> List[str]:
    return [f"{value:#018x}" for value in values]


def bits_from_mask(mask: int) -> List[int]:
    return [bit for bit in range(64) if (mask >> bit) & 1]


def derive_candidate_descriptor_values(
    desc_defs: Dict[str, DescriptorDef],
    extra_flip_bits: Sequence[int],
    additive_deltas: Sequence[int],
) -> Dict[str, object]:
    live_imms = {reg: desc_def.imm for reg, desc_def in desc_defs.items()}
    common_bits = 0
    for idx, imm in enumerate(live_imms.values()):
        common_bits = imm if idx == 0 else (common_bits & imm)
    varying_mask = 0
    for imm in live_imms.values():
        varying_mask |= imm ^ common_bits
    live_diff_bits = bits_from_mask(varying_mask)
    flip_bits = sorted(set(live_diff_bits) | set(extra_flip_bits))

    values = set()
    for subset_mask in range(1 << len(flip_bits)):
        value = common_bits
        for idx, bit in enumerate(flip_bits):
            if (subset_mask >> idx) & 1:
                value |= 1 << bit
        for delta in additive_deltas:
            candidate = value + delta
            if 0 <= candidate < (1 << 64):
                values.add(candidate)

    candidate_values = sorted(values)
    return {
        "live_imms": live_imms,
        "common_bits": common_bits,
        "live_diff_bits": live_diff_bits,
        "flip_bits": flip_bits,
        "additive_deltas": list(additive_deltas),
        "candidate_values": candidate_values,
    }


def disabled_plan_template(base_ptx: str, active_site: int):
    sites = parse_cp_sites(base_ptx)
    other_idx = 1 - active_site
    _idx, _indent, _pred, addr_expr, desc_reg = sites[other_idx]
    return SCHEDULE.SitePlan(False, SCHEDULE.WARPX2_02_13, addr_expr, desc_reg)


def materialize_site_plans(base_ptx: str, plan: DescriptorValuePlan):
    disabled = disabled_plan_template(base_ptx, plan.active_site)
    active = SCHEDULE.SitePlan(True, plan.opcode, plan.addr_expr, plan.desc_reg)
    if plan.active_site == 0:
        return (active, disabled)
    return (disabled, active)


def candidate_plans(
    base_ptx: str,
    addr_imms: Sequence[int],
    candidate_values: Sequence[int],
    patch_all_desc_regs: bool,
) -> Iterable[DescriptorValuePlan]:
    desc_regs = desc_reg_options(base_ptx)
    base_regs = base_reg_options(base_ptx)
    opcodes = (SCHEDULE.WARPX2_02_13, SCHEDULE.WARPX2_01_23)
    for opcode, active_site, desc_reg, base_reg, addr_imm in itertools.product(
        opcodes,
        (0, 1),
        desc_regs,
        base_regs,
        addr_imms,
    ):
        addr_expr = f"{base_reg} + {addr_imm}"
        if patch_all_desc_regs:
            for value_tuple in itertools.product(candidate_values, repeat=len(desc_regs)):
                yield DescriptorValuePlan(
                    opcode=opcode,
                    active_site=active_site,
                    desc_reg=desc_reg,
                    base_reg=base_reg,
                    addr_imm=addr_imm,
                    addr_expr=addr_expr,
                    patched_desc_imms={reg: value for reg, value in zip(desc_regs, value_tuple)},
                )
        else:
            for value in candidate_values:
                yield DescriptorValuePlan(
                    opcode=opcode,
                    active_site=active_site,
                    desc_reg=desc_reg,
                    base_reg=base_reg,
                    addr_imm=addr_imm,
                    addr_expr=addr_expr,
                    patched_desc_imms={desc_reg: value},
                )


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


def find_target_half(chunk_halves: Sequence[Dict[str, object]], target_chunk: int, target_half: int) -> Dict[str, object]:
    for half in chunk_halves:
        if half["chunk"] == target_chunk and half["half"] == target_half:
            return half
    raise RuntimeError(f"missing target half chunk={target_chunk} half={target_half}")


def summarize_record(rec: Dict[str, object], target_chunk: int, target_half: int) -> Dict[str, object]:
    target = find_target_half(rec["chunk_halves"], target_chunk, target_half)
    return {
        "opcode": rec["plan"]["opcode"],
        "active_site": rec["plan"]["active_site"],
        "desc_reg": rec["plan"]["desc_reg"],
        "addr_expr": rec["plan"]["addr_expr"],
        "patched_desc_imms": rec["plan"]["patched_desc_imms"],
        "patched_desc_imms_hex": format_hex_map(rec["plan"]["patched_desc_imms"]),
        "quadrant_signature": rec["quadrant_signature"],
        "target_label": target["label"],
        "target_matches_expected": target["matches_expected"],
        "target_matches_zero": target["matches_zero"],
        "num_b_halves": rec["num_b_halves"],
        "num_other_halves": rec["num_other_halves"],
        "target_exact": rec["target_exact"],
        "target_clean_exact": rec["target_clean_exact"],
        "target_isolated_exact": rec["target_isolated_exact"],
    }


def summarize_plans(
    plan_results: Sequence[Dict[str, object]],
    target_chunk: int,
    target_half: int,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "total_plans": len(plan_results),
        "pass_plans": 0,
        "fail_plans": 0,
        "target_exact_hits": 0,
        "target_clean_exact_hits": 0,
        "target_isolated_exact_hits": 0,
        "target_signatures": [],
        "target_hit_examples": [],
        "best_target_near_misses": [],
    }
    target_signatures = set()
    target_hits = []
    near_misses = []
    for rec in plan_results:
        if rec["status"] != "PASS":
            summary["fail_plans"] += 1
            continue
        summary["pass_plans"] += 1
        compact = summarize_record(rec, target_chunk, target_half)
        if rec["target_exact"]:
            summary["target_exact_hits"] += 1
            if rec["target_clean_exact"]:
                summary["target_clean_exact_hits"] += 1
            if rec["target_isolated_exact"]:
                summary["target_isolated_exact_hits"] += 1
            target_signatures.add(rec["quadrant_signature"])
            target_hits.append(compact)
        else:
            near_misses.append(compact)

    target_hits.sort(
        key=lambda item: (
            not item["target_isolated_exact"],
            not item["target_clean_exact"],
            item["num_other_halves"],
            -item["num_b_halves"],
            item["opcode"],
            item["active_site"],
            item["desc_reg"],
            item["addr_expr"],
            json.dumps(item["patched_desc_imms_hex"], sort_keys=True),
        )
    )
    near_misses.sort(
        key=lambda item: (
            -item["target_matches_expected"],
            item["num_other_halves"],
            -item["num_b_halves"],
            item["opcode"],
            item["active_site"],
            item["desc_reg"],
            item["addr_expr"],
            json.dumps(item["patched_desc_imms_hex"], sort_keys=True),
        )
    )
    summary["target_signatures"] = sorted(target_signatures)
    summary["target_hit_examples"] = target_hits[:32]
    summary["best_target_near_misses"] = near_misses[:32]
    return summary


def run_child(plan_payload: str, base_ptx_path: Path, shared_bytes: int) -> Dict[str, object]:
    import pycuda.driver as cuda

    plan = DescriptorValuePlan(**json.loads(plan_payload))
    base_ptx = base_ptx_path.read_text()
    inp, expected = SCHEDULE.make_input()
    ptx = patch_descriptor_defs(base_ptx, plan.patched_desc_imms)
    ptx = SCHEDULE.patched_ptx(ptx, materialize_site_plans(base_ptx, plan))
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


def run_parent(args) -> Dict[str, object]:
    base_ptx = SCHEDULE.load_base_ptx()
    base_ptx_dir = Path(tempfile.mkdtemp(prefix="cp_desc_value_base_"))
    base_ptx_path = base_ptx_dir / "base.ptx"
    base_ptx_path.write_text(base_ptx)
    desc_regs = desc_reg_options(base_ptx)
    desc_defs = parse_descriptor_defs(base_ptx, desc_regs)
    desc_space = derive_candidate_descriptor_values(desc_defs, args.extra_flip_bits, args.additive_deltas)

    results: Dict[str, object] = {
        "addr_imms": list(args.addr_imms),
        "target_chunk": args.target_chunk,
        "target_half": args.target_half,
        "patch_all_desc_regs": args.patch_all_desc_regs,
        "base_sites": parse_cp_sites(base_ptx),
        "descriptor_defs": {
            reg: {
                "line_idx": desc_def.line_idx,
                "op": desc_def.op,
                "base_reg": desc_def.base_reg,
                "imm": desc_def.imm,
                "imm_hex": f"{desc_def.imm:#018x}",
            }
            for reg, desc_def in desc_defs.items()
        },
        "live_descriptor_imms": desc_space["live_imms"],
        "live_descriptor_imms_hex": format_hex_map(desc_space["live_imms"]),
        "descriptor_common_bits": desc_space["common_bits"],
        "descriptor_common_bits_hex": f"{desc_space['common_bits']:#018x}",
        "descriptor_live_diff_bits": desc_space["live_diff_bits"],
        "descriptor_flip_bits": desc_space["flip_bits"],
        "descriptor_additive_deltas": desc_space["additive_deltas"],
        "candidate_descriptor_values": desc_space["candidate_values"],
        "candidate_descriptor_values_hex": format_hex_list(desc_space["candidate_values"]),
        "plans": [],
    }

    child_env = os.environ.copy()
    pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = "python:." if not pythonpath else f"python:.:{pythonpath}"

    for idx, plan in enumerate(
        candidate_plans(
            base_ptx,
            args.addr_imms,
            desc_space["candidate_values"],
            args.patch_all_desc_regs,
        )
    ):
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

        child = json.loads(proc.stdout)
        rec = {"plan": asdict(plan), "status": "PASS"}
        rec.update(analyze_halves(child["out"], child["expected"]))
        target = find_target_half(rec["chunk_halves"], args.target_chunk, args.target_half)
        rec["target_label"] = target["label"]
        rec["target_exact"] = target["label"] == "B"
        rec["target_clean_exact"] = rec["target_exact"] and rec["num_other_halves"] == 0
        rec["target_isolated_exact"] = rec["target_clean_exact"] and rec["num_b_halves"] == 1
        results["plans"].append(rec)

    results["summary"] = summarize_plans(results["plans"], args.target_chunk, args.target_half)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-plan", type=str, default="")
    parser.add_argument("--base-ptx-path", type=Path, default=Path())
    parser.add_argument("--shared-bytes", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="optional cap on tried plan count")
    parser.add_argument(
        "--addr-imms",
        type=parse_int_csv,
        default=list(DEFAULT_ADDR_IMMS),
        help="comma-separated immediate offsets to try for each live base register (default: 0,4)",
    )
    parser.add_argument(
        "--extra-flip-bits",
        type=parse_optional_int_csv,
        default=[],
        help="optional extra descriptor bit positions to add to the live descriptor bit subcube",
    )
    parser.add_argument(
        "--additive-deltas",
        type=parse_int_csv,
        default=[0],
        help="comma-separated additive deltas applied to each descriptor cube value (default: 0)",
    )
    parser.add_argument(
        "--patch-all-desc-regs",
        action="store_true",
        help="assign candidate descriptor values to all live descriptor regs instead of only the chosen active reg",
    )
    parser.add_argument("--target-chunk", type=int, default=DEFAULT_TARGET_CHUNK)
    parser.add_argument("--target-half", type=int, default=DEFAULT_TARGET_HALF)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT
        / ".codex"
        / "initiatives"
        / "tmem_linear_generalization"
        / "experiments"
        / "results"
        / "probe_cp_direct_ptx_descriptor_values_gpu0.json",
    )
    args = parser.parse_args()

    if args.child_plan:
        if not args.base_ptx_path:
            raise SystemExit("--base-ptx-path is required with --child-plan")
        try:
            result = run_child(args.child_plan, args.base_ptx_path, args.shared_bytes)
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
