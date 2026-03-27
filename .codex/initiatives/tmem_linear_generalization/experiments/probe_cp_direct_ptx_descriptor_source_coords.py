#!/usr/bin/env python3
"""Source-coordinate descriptor probe for tcgen05.cp.warpx2.

This is a more structured follow-up to the additive and lbo/sbo descriptor
probes. Instead of nudging the final descriptor value blindly, it enumerates
bounded source-coordinate hypotheses that approximate alternate `smemLoad(a, b)`
origins and maps them into descriptor constructions.

The current model keeps the non-offset descriptor fields from a chosen live seed
descriptor and varies only the fields that `smemLoad(a, b)` would affect for the
current MMAv5-style descriptor path:
  - baseAddress, in 16-byte units
  - matrixBaseOffset, in 128-byte units

Default search space:
  - one active warpx2 site at a time
  - opcode family `{warpx2::02_13, warpx2::01_23}`
  - seed descriptor register `{%rd2, %rd3}`
  - clean aligned TMEM address seeds `{%r18 + 4, %r7 + 0}`
  - construction mode `{relative_seed, zero_based}`
  - row-stride bytes `{128, 256, 512}`
  - col-stride bytes `{16, 32, 64}`
  - source coordinates `a in {0, 1, 2, 4}`, `b in {0, 1, 2, 4}`

The target question is whether any explicit source-coordinate construction can
recover the missing clean chunk-3 left-half (`3L`) write without widening into
arbitrary descriptor-field brute force.
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

DEFAULT_ACTIVE_SITES = (1,)
DEFAULT_ADDR_EXPRS = ("%r18 + 4", "%r7 + 0")
DEFAULT_SEED_DESC_REGS = ("%rd2", "%rd3")
DEFAULT_CONSTRUCTIONS = ("relative_seed", "zero_based")
DEFAULT_ROW_STRIDE_BYTES = (128, 256, 512)
DEFAULT_COL_STRIDE_BYTES = (16, 32, 64)
DEFAULT_A_VALUES = (0, 1, 2, 4)
DEFAULT_B_VALUES = (0, 1, 2, 4)
DEFAULT_LBO_VALUES = (-1,)
DEFAULT_SBO_VALUES = (-1,)
DEFAULT_TARGET_CHUNK = 3
DEFAULT_TARGET_HALF = 0

BASE_ADDRESS_MASK = (1 << 14) - 1
MATRIX_BASE_OFFSET_MASK = 0x7
BASE_ADDRESS_SHIFT = 0
MATRIX_BASE_OFFSET_SHIFT = 49
LBO_MASK = (1 << 14) - 1
LBO_SHIFT = 16
SBO_MASK = (1 << 14) - 1
SBO_SHIFT = 32


def _load_desc_module():
    spec = importlib.util.spec_from_file_location(
        "probe_cp_direct_ptx_descriptor_values", DESC_VALUES_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"failed to load descriptor helper module from {DESC_VALUES_PATH}"
        )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


DESC = _load_desc_module()


@dataclass(frozen=True)
class SourceCoordPlan:
    opcode: str
    active_site: int
    seed_desc_reg: str
    addr_expr: str
    construction: str
    row_stride_bytes: int
    col_stride_bytes: int
    a: int
    b: int
    lbo: int
    sbo: int


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


def decode_descriptor_offset_fields(imm: int) -> Dict[str, int]:
    return {
        "base_address": (imm >> BASE_ADDRESS_SHIFT) & BASE_ADDRESS_MASK,
        "matrix_base_offset": (imm >> MATRIX_BASE_OFFSET_SHIFT)
        & MATRIX_BASE_OFFSET_MASK,
        "lbo": (imm >> LBO_SHIFT) & LBO_MASK,
        "sbo": (imm >> SBO_SHIFT) & SBO_MASK,
        "swizzling_mode": (imm >> 62) & 0x3,
        "mmav5_bit": (imm >> 46) & 0x1,
    }


def encode_descriptor_fields(
    seed_imm: int,
    base_address: int,
    matrix_base_offset: int,
    lbo: int | None = None,
    sbo: int | None = None,
) -> int:
    if not (0 <= base_address <= BASE_ADDRESS_MASK):
        raise ValueError(f"base address field out of range: {base_address}")
    if not (0 <= matrix_base_offset <= MATRIX_BASE_OFFSET_MASK):
        raise ValueError(
            f"matrix base offset field out of range: {matrix_base_offset}"
        )
    if lbo is not None and not (0 <= lbo <= LBO_MASK):
        raise ValueError(f"lbo field out of range: {lbo}")
    if sbo is not None and not (0 <= sbo <= SBO_MASK):
        raise ValueError(f"sbo field out of range: {sbo}")
    imm = seed_imm
    imm &= ~(BASE_ADDRESS_MASK << BASE_ADDRESS_SHIFT)
    imm &= ~(MATRIX_BASE_OFFSET_MASK << MATRIX_BASE_OFFSET_SHIFT)
    imm |= base_address << BASE_ADDRESS_SHIFT
    imm |= matrix_base_offset << MATRIX_BASE_OFFSET_SHIFT
    if lbo is not None:
        imm &= ~(LBO_MASK << LBO_SHIFT)
        imm |= lbo << LBO_SHIFT
    if sbo is not None:
        imm &= ~(SBO_MASK << SBO_SHIFT)
        imm |= sbo << SBO_SHIFT
    return imm


def infer_live_descriptor_delta_bytes(
    desc_defs: Dict[str, DESC.DescriptorDef],
) -> int | None:
    if "%rd2" not in desc_defs or "%rd3" not in desc_defs:
        return None
    rd2 = decode_descriptor_offset_fields(desc_defs["%rd2"].imm)
    rd3 = decode_descriptor_offset_fields(desc_defs["%rd3"].imm)
    delta_base_bytes = (rd3["base_address"] - rd2["base_address"]) * 16
    delta_matrix_bytes = (
        (rd3["matrix_base_offset"] - rd2["matrix_base_offset"]) & 0x7
    ) * 128
    if delta_base_bytes == delta_matrix_bytes:
        return delta_base_bytes
    return None


def source_delta_bytes(plan: SourceCoordPlan) -> int:
    return plan.a * plan.row_stride_bytes + plan.b * plan.col_stride_bytes


def synthesize_descriptor_from_source_coords(
    seed_imm: int,
    construction: str,
    delta_bytes: int,
    lbo: int | None = None,
    sbo: int | None = None,
) -> int:
    if delta_bytes % 16 != 0:
        raise ValueError(
            f"source delta must be 16-byte aligned for descriptor synthesis, got {delta_bytes}"
        )

    seed_fields = decode_descriptor_offset_fields(seed_imm)
    delta_base_address = delta_bytes // 16
    delta_matrix_base_offset = delta_bytes // 128

    if construction == "relative_seed":
        base_address = seed_fields["base_address"] + delta_base_address
        matrix_base_offset = (
            seed_fields["matrix_base_offset"] + delta_matrix_base_offset
        ) & MATRIX_BASE_OFFSET_MASK
    elif construction == "zero_based":
        base_address = delta_base_address
        matrix_base_offset = delta_matrix_base_offset & MATRIX_BASE_OFFSET_MASK
    else:
        raise ValueError(f"unknown construction mode: {construction}")

    return encode_descriptor_fields(
        seed_imm, base_address, matrix_base_offset, lbo=lbo, sbo=sbo
    )


def candidate_plans(args) -> Iterable[SourceCoordPlan]:
    for opcode in (DESC.SCHEDULE.WARPX2_02_13, DESC.SCHEDULE.WARPX2_01_23):
        for active_site in args.active_sites:
            for seed_desc_reg in args.seed_desc_regs:
                for addr_expr in args.addr_exprs:
                    for construction in args.constructions:
                        for row_stride_bytes in args.row_stride_bytes:
                            for col_stride_bytes in args.col_stride_bytes:
                                    for a in args.a_values:
                                        for b in args.b_values:
                                            for lbo in args.lbo_values:
                                                for sbo in args.sbo_values:
                                                    yield SourceCoordPlan(
                                                        opcode=opcode,
                                                        active_site=active_site,
                                                        seed_desc_reg=seed_desc_reg,
                                                        addr_expr=addr_expr,
                                                        construction=construction,
                                                        row_stride_bytes=row_stride_bytes,
                                                        col_stride_bytes=col_stride_bytes,
                                                        a=a,
                                                        b=b,
                                                        lbo=lbo,
                                                        sbo=sbo,
                                                    )


def compact_record(rec: Dict[str, object]) -> Dict[str, object]:
    return {
        "plan": rec["plan"],
        "source_delta_bytes": rec["source_delta_bytes"],
        "descriptor_imm_hex": rec["descriptor_imm_hex"],
        "descriptor_fields": rec["descriptor_fields"],
        "quadrant_signature": rec["quadrant_signature"],
        "target_label": rec["target_label"],
        "target_exact": rec["target_exact"],
        "target_clean_exact": rec["target_clean_exact"],
        "target_isolated_exact": rec["target_isolated_exact"],
        "num_b_halves": rec["num_b_halves"],
        "num_other_halves": rec["num_other_halves"],
    }


def summarize(
    results: Sequence[Dict[str, object]], target_chunk: int, target_half: int
) -> Dict[str, object]:
    summary = {
        "total_plans": len(results),
        "pass_plans": 0,
        "fail_plans": 0,
        "target_exact_hits": 0,
        "target_clean_exact_hits": 0,
        "target_isolated_exact_hits": 0,
        "clean_quadrant_hits": 0,
        "target_signatures": [],
        "target_hit_examples": [],
        "best_target_near_misses": [],
        "best_clean_examples": [],
    }
    hits = []
    near = []
    clean = []
    target_signatures = set()
    for rec in results:
        if rec["status"] != "PASS":
            summary["fail_plans"] += 1
            continue
        summary["pass_plans"] += 1
        compact = compact_record(rec)
        if rec["clean_quadrant_mapping"]:
            summary["clean_quadrant_hits"] += 1
            clean.append(compact)
        if rec["target_exact"]:
            summary["target_exact_hits"] += 1
            if rec["target_clean_exact"]:
                summary["target_clean_exact_hits"] += 1
            if rec["target_isolated_exact"]:
                summary["target_isolated_exact_hits"] += 1
            hits.append(compact)
            target_signatures.add(rec["quadrant_signature"])
        else:
            near.append(compact)

    hits.sort(
        key=lambda item: (
            not item["target_isolated_exact"],
            not item["target_clean_exact"],
            item["num_other_halves"],
            -item["num_b_halves"],
            item["source_delta_bytes"],
            item["plan"]["construction"],
            item["plan"]["lbo"],
            item["plan"]["sbo"],
            item["plan"]["seed_desc_reg"],
            item["plan"]["opcode"],
        )
    )
    near.sort(
        key=lambda item: (
            item["target_label"] != "other",
            item["num_other_halves"],
            -item["num_b_halves"],
            item["source_delta_bytes"],
            item["plan"]["construction"],
            item["plan"]["lbo"],
            item["plan"]["sbo"],
            item["plan"]["seed_desc_reg"],
            item["plan"]["opcode"],
        )
    )
    clean.sort(
        key=lambda item: (
            item["num_other_halves"],
            -item["num_b_halves"],
            item["source_delta_bytes"],
            item["plan"]["construction"],
            item["plan"]["lbo"],
            item["plan"]["sbo"],
            item["plan"]["seed_desc_reg"],
            item["plan"]["opcode"],
        )
    )

    summary["target_signatures"] = sorted(target_signatures)
    summary["target_hit_examples"] = hits[:32]
    summary["best_target_near_misses"] = near[:32]
    summary["best_clean_examples"] = clean[:32]
    return summary


def run_child(
    plan_payload: str,
    base_ptx_path: Path,
    shared_bytes: int,
    target_chunk: int,
    target_half: int,
) -> Dict[str, object]:
    import pycuda.driver as cuda

    plan = SourceCoordPlan(**json.loads(plan_payload))
    base_ptx = base_ptx_path.read_text()
    desc_defs = DESC.parse_descriptor_defs(base_ptx, [plan.seed_desc_reg])
    seed_def = desc_defs[plan.seed_desc_reg]
    delta_bytes = source_delta_bytes(plan)
    new_imm = synthesize_descriptor_from_source_coords(
        seed_def.imm,
        plan.construction,
        delta_bytes,
        lbo=None if plan.lbo < 0 else plan.lbo,
        sbo=None if plan.sbo < 0 else plan.sbo,
    )

    descriptor_ptx = DESC.patch_descriptor_defs(base_ptx, {plan.seed_desc_reg: new_imm})
    one_site_plan = DESC.DescriptorValuePlan(
        opcode=plan.opcode,
        active_site=plan.active_site,
        desc_reg=plan.seed_desc_reg,
        base_reg=plan.addr_expr.split("+")[0].strip(),
        addr_imm=int(plan.addr_expr.split("+")[1].strip()),
        addr_expr=plan.addr_expr,
        patched_desc_imms={plan.seed_desc_reg: new_imm},
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
    target = DESC.find_target_half(rec["chunk_halves"], target_chunk, target_half)
    rec["plan"] = asdict(plan)
    rec["source_delta_bytes"] = delta_bytes
    rec["descriptor_imm"] = new_imm
    rec["descriptor_imm_hex"] = f"{new_imm:#018x}"
    rec["descriptor_fields"] = decode_descriptor_offset_fields(new_imm)
    rec["seed_descriptor_imm_hex"] = f"{seed_def.imm:#018x}"
    rec["seed_descriptor_fields"] = decode_descriptor_offset_fields(seed_def.imm)
    rec["target_label"] = target["label"]
    rec["target_exact"] = target["label"] == "B"
    rec["target_clean_exact"] = rec["target_exact"] and rec["num_other_halves"] == 0
    rec["target_isolated_exact"] = (
        rec["target_clean_exact"] and rec["num_b_halves"] == 1
    )
    return rec


def run_parent(args) -> Dict[str, object]:
    base_ptx = DESC.SCHEDULE.load_base_ptx()
    base_ptx_dir = Path(tempfile.mkdtemp(prefix="cp_desc_source_coords_base_"))
    base_ptx_path = base_ptx_dir / "base.ptx"
    base_ptx_path.write_text(base_ptx)
    desc_defs = DESC.parse_descriptor_defs(base_ptx, args.seed_desc_regs)

    results = {
        "base_sites": DESC.parse_cp_sites(base_ptx),
        "seed_descriptor_imms": {
            reg: desc_defs[reg].imm for reg in args.seed_desc_regs
        },
        "seed_descriptor_imms_hex": {
            reg: f"{desc_defs[reg].imm:#018x}" for reg in args.seed_desc_regs
        },
        "seed_descriptor_fields": {
            reg: decode_descriptor_offset_fields(desc_defs[reg].imm)
            for reg in args.seed_desc_regs
        },
        "live_descriptor_delta_bytes": infer_live_descriptor_delta_bytes(desc_defs),
        "search_space": {
            "active_sites": list(args.active_sites),
            "addr_exprs": list(args.addr_exprs),
            "seed_desc_regs": list(args.seed_desc_regs),
            "constructions": list(args.constructions),
            "row_stride_bytes": list(args.row_stride_bytes),
            "col_stride_bytes": list(args.col_stride_bytes),
            "a_values": list(args.a_values),
            "b_values": list(args.b_values),
            "lbo_values": list(args.lbo_values),
            "sbo_values": list(args.sbo_values),
            "target_chunk": args.target_chunk,
            "target_half": args.target_half,
        },
        "plans": [],
    }

    child_env = os.environ.copy()
    pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = (
        "python:." if not pythonpath else f"python:.:{pythonpath}"
    )

    for idx, plan in enumerate(candidate_plans(args)):
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
                "--target-chunk",
                str(args.target_chunk),
                "--target-half",
                str(args.target_half),
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
                    "error": proc.stderr.strip()
                    or proc.stdout.strip()
                    or f"child exited {proc.returncode}",
                }
            )
            continue
        rec = json.loads(proc.stdout)
        rec["status"] = "PASS"
        results["plans"].append(rec)

    results["summary"] = summarize(
        results["plans"], args.target_chunk, args.target_half
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-plan", type=str, default="")
    parser.add_argument("--base-ptx-path", type=Path, default=Path())
    parser.add_argument("--shared-bytes", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--active-sites", type=parse_int_csv, default=list(DEFAULT_ACTIVE_SITES)
    )
    parser.add_argument(
        "--addr-exprs", type=parse_str_csv, default=list(DEFAULT_ADDR_EXPRS)
    )
    parser.add_argument(
        "--seed-desc-regs",
        type=parse_str_csv,
        default=list(DEFAULT_SEED_DESC_REGS),
    )
    parser.add_argument(
        "--constructions",
        type=parse_str_csv,
        default=list(DEFAULT_CONSTRUCTIONS),
    )
    parser.add_argument(
        "--row-stride-bytes",
        type=parse_int_csv,
        default=list(DEFAULT_ROW_STRIDE_BYTES),
    )
    parser.add_argument(
        "--col-stride-bytes",
        type=parse_int_csv,
        default=list(DEFAULT_COL_STRIDE_BYTES),
    )
    parser.add_argument("--a-values", type=parse_int_csv, default=list(DEFAULT_A_VALUES))
    parser.add_argument("--b-values", type=parse_int_csv, default=list(DEFAULT_B_VALUES))
    parser.add_argument(
        "--lbo-values",
        type=parse_int_csv,
        default=list(DEFAULT_LBO_VALUES),
    )
    parser.add_argument(
        "--sbo-values",
        type=parse_int_csv,
        default=list(DEFAULT_SBO_VALUES),
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
        / "probe_cp_direct_ptx_descriptor_source_coords_gpu0.json",
    )
    args = parser.parse_args()

    if args.child_plan:
        if not args.base_ptx_path:
            raise SystemExit("--base-ptx-path is required with --child-plan")
        try:
            rec = run_child(
                args.child_plan,
                args.base_ptx_path,
                args.shared_bytes,
                args.target_chunk,
                args.target_half,
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
