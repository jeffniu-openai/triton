#!/usr/bin/env python3
"""Exact source-offset descriptor probe for tcgen05.cp.warpx2.

This probe searches descriptor values that match the actual
`DotOpMmaSmemLoader::smemLoad(a, b)` encoding pattern more closely than the
earlier additive or low-bit perturbation probes:

  - keep the live descriptor template fields (`lbo`, `sbo`, swizzle, version)
  - synthesize `matrixBaseOffset` from the source offset in 128b units
  - add the same source offset directly into the descriptor base-address field

This matches the coupled `matrixBaseOffset + smemByteOffsetb128` update in
`DotOpMmaSmemLoader::smemLoad(...)` for the current swizzle-0 scales-copy
kernel. The target is still the missing chunk-3 left half (`3L`) under
`tcgen05.cp.warpx2`.
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
    spec = importlib.util.spec_from_file_location(
        "probe_cp_direct_ptx_descriptor_values", DESC_VALUES_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load descriptor helper module from {DESC_VALUES_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


DESC = _load_desc_module()

DEFAULT_SOURCE_OFFSETS_B128 = tuple(range(64))
DEFAULT_ADDR_IMMS = (0, 4)
DEFAULT_TARGET_CHUNK = 3
DEFAULT_TARGET_HALF = 0
ZERO_OFFSET_SEED_DESC_REG = "%rd2"

BASE_ADDRESS_MASK = (1 << 14) - 1
MATRIX_BASE_SHIFT = 49
MATRIX_BASE_MASK = 0x7 << MATRIX_BASE_SHIFT


@dataclass(frozen=True)
class SourceOffsetPlan:
    opcode: str
    active_site: int
    desc_reg: str
    base_reg: str
    addr_imm: int
    addr_expr: str
    source_offset_b128: int


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


def make_template_imm(seed_imm: int) -> int:
    return seed_imm & ~BASE_ADDRESS_MASK & ~MATRIX_BASE_MASK


def synthesize_source_offset_imm(seed_imm: int, source_offset_b128: int) -> int:
    if source_offset_b128 < 0:
        raise ValueError("source_offset_b128 must be non-negative")
    if source_offset_b128 >= (1 << 14):
        raise ValueError("source_offset_b128 must fit in the descriptor baseAddress field")
    template = make_template_imm(seed_imm)
    matrix_base = (source_offset_b128 >> 3) & 0x7
    return template | (matrix_base << MATRIX_BASE_SHIFT) | source_offset_b128


def candidate_plans(
    base_ptx: str,
    addr_imms: Sequence[int],
    source_offsets_b128: Sequence[int],
) -> Iterable[SourceOffsetPlan]:
    desc_regs = DESC.desc_reg_options(base_ptx)
    base_regs = DESC.base_reg_options(base_ptx)
    opcodes = (DESC.SCHEDULE.WARPX2_02_13, DESC.SCHEDULE.WARPX2_01_23)
    for opcode in opcodes:
        for active_site in (0, 1):
            for desc_reg in desc_regs:
                for base_reg in base_regs:
                    for addr_imm in addr_imms:
                        addr_expr = f"{base_reg} + {addr_imm}"
                        for source_offset_b128 in source_offsets_b128:
                            yield SourceOffsetPlan(
                                opcode=opcode,
                                active_site=active_site,
                                desc_reg=desc_reg,
                                base_reg=base_reg,
                                addr_imm=addr_imm,
                                addr_expr=addr_expr,
                                source_offset_b128=source_offset_b128,
                            )


def run_child(plan_payload: str, base_ptx_path: Path, shared_bytes: int) -> Dict[str, object]:
    import pycuda.driver as cuda

    plan = SourceOffsetPlan(**json.loads(plan_payload))
    base_ptx = base_ptx_path.read_text()

    seed = DESC.parse_descriptor_defs(base_ptx, [ZERO_OFFSET_SEED_DESC_REG])[ZERO_OFFSET_SEED_DESC_REG]
    synthesized_imm = synthesize_source_offset_imm(seed.imm, plan.source_offset_b128)

    descriptor_ptx = DESC.patch_descriptor_defs(base_ptx, {plan.desc_reg: synthesized_imm})
    one_site_plan = DESC.DescriptorValuePlan(
        opcode=plan.opcode,
        active_site=plan.active_site,
        desc_reg=plan.desc_reg,
        base_reg=plan.base_reg,
        addr_imm=plan.addr_imm,
        addr_expr=plan.addr_expr,
        patched_desc_imms={plan.desc_reg: synthesized_imm},
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
    rec["descriptor_imm"] = synthesized_imm
    rec["descriptor_imm_hex"] = f"{synthesized_imm:#018x}"
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
            item["plan"]["active_site"],
            item["plan"]["addr_expr"],
            item["plan"]["source_offset_b128"],
        )
    )
    near.sort(
        key=lambda item: (
            item["target_label"] != "other",
            item["num_other_halves"],
            -item["num_b_halves"],
            item["plan"]["opcode"],
            item["plan"]["active_site"],
            item["plan"]["addr_expr"],
            item["plan"]["source_offset_b128"],
        )
    )
    summary["target_hit_examples"] = hits[:32]
    summary["best_target_near_misses"] = near[:32]
    return summary


def run_parent(args) -> Dict[str, object]:
    base_ptx = DESC.SCHEDULE.load_base_ptx()
    base_ptx_dir = Path(tempfile.mkdtemp(prefix="cp_desc_source_offsets_base_"))
    base_ptx_path = base_ptx_dir / "base.ptx"
    base_ptx_path.write_text(base_ptx)

    zero_seed = DESC.parse_descriptor_defs(base_ptx, [ZERO_OFFSET_SEED_DESC_REG])[ZERO_OFFSET_SEED_DESC_REG]
    results = {
        "base_sites": DESC.parse_cp_sites(base_ptx),
        "zero_offset_seed_desc_reg": ZERO_OFFSET_SEED_DESC_REG,
        "zero_offset_seed_imm": zero_seed.imm,
        "zero_offset_seed_imm_hex": f"{zero_seed.imm:#018x}",
        "plans": [],
    }

    child_env = os.environ.copy()
    pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = "python:." if not pythonpath else f"python:.:{pythonpath}"

    for idx, plan in enumerate(candidate_plans(base_ptx, args.addr_imms, args.source_offsets_b128)):
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
    parser.add_argument(
        "--source-offsets-b128",
        type=parse_int_csv,
        default=list(DEFAULT_SOURCE_OFFSETS_B128),
    )
    parser.add_argument("--addr-imms", type=parse_int_csv, default=list(DEFAULT_ADDR_IMMS))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT
        / ".codex"
        / "initiatives"
        / "tmem_linear_generalization"
        / "experiments"
        / "results"
        / "probe_cp_direct_ptx_descriptor_source_offsets_gpu0.json",
    )
    args = parser.parse_args()

    if args.child_plan:
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
