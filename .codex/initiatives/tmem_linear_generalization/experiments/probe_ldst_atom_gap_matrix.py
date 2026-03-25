#!/usr/bin/env python3
"""Probe TMEM ld/st atom-family reachability across a broad layout/shape matrix."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
TEST_MATRIX_PATH = REPO_ROOT / "python/test/gluon/test_tmem_runtime_matrix.py"


def _load_test_matrix_module():
    spec = importlib.util.spec_from_file_location("tmem_runtime_matrix", TEST_MATRIX_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _classify_failure(exc: BaseException) -> str:
    text = str(exc)
    clean_needles = (
        "unsupported register broadcast pattern for direct lowering",
        "TMEM layout has no supported register layout",
        "failed to infer tensor memory encoding for memdesc_index",
        "TMEM layout '",
    )
    bug_needles = (
        "Segmentation fault",
        "Assertion",
        "INTERNAL COMPILER ERROR",
        "please share the reproducer",
    )
    if any(needle in text for needle in clean_needles):
        return "CLEAN_UNSUPPORTED"
    if any(needle in text for needle in bug_needles):
        return "BUG"
    return "UNKNOWN_FAIL"


def _expected_opcode_fragment(variant: str) -> str:
    if variant == "32x32b_splitn":
        return ".16x32bx2."
    return f".{variant}."


def main():
    torch.manual_seed(0)
    mod = _load_test_matrix_module()

    identity_shapes = [(64, 64), (128, 64), (128, 128), (128, 256), (256, 128)]
    identity_variants = ["32x32b", "16x64b", "16x128b", "16x256b"]
    splitn_cases = [(64, 2), (64, 64), (64, 128)]

    cases = []
    for m, n in identity_shapes:
        for variant in identity_variants:
            cases.append(
                {
                    "family": "identity",
                    "m": m,
                    "n": n,
                    "variant": variant,
                    "layout": mod._make_tmem_linear_layout(m, n),
                }
            )
    for m, n in splitn_cases:
        cases.append(
            {
                "family": "splitn",
                "m": m,
                "n": n,
                "variant": "32x32b_splitn",
                "layout": mod._make_tmem_linear_layout_m64(n),
            }
        )

    results = []
    family_hits = set()
    counts = {"PASS": 0, "CLEAN_UNSUPPORTED": 0, "BUG": 0, "UNKNOWN_FAIL": 0}

    for case in cases:
        m = case["m"]
        n = case["n"]
        variant = case["variant"]
        layout = case["layout"]
        inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
        out = torch.empty_like(inp)

        record = {
            "family": case["family"],
            "m": m,
            "n": n,
            "variant": variant,
        }
        try:
            compiled = mod.tmem_ldst_variant_kernel[(1,)](inp, out, layout, m, n, variant, num_warps=4)
            torch.testing.assert_close(out, inp, atol=0, rtol=0)
            ptx = mod._extract_tcgen05_opcode_offsets(compiled.asm["ptx"])
            llir = mod._extract_tcgen05_opcode_offsets(compiled.asm["llir"])
            if not ptx or ptx != llir:
                raise RuntimeError("PTX/LLIR opcode mismatch or empty opcode list")
            frag = _expected_opcode_fragment(variant)
            if not all(frag in op for op, _ in ptx):
                raise RuntimeError(f"Unexpected opcode family for {variant}: {ptx}")
            record["status"] = "PASS"
            record["opcode_count"] = len(ptx)
            record["opcode"] = ptx[0][0]
            counts["PASS"] += 1
            family_hits.add(record["opcode"])
        except Exception as exc:  # noqa: BLE001
            status = _classify_failure(exc)
            record["status"] = status
            record["error"] = str(exc).splitlines()[0]
            counts[status] += 1
        results.append(record)

    summary = {
        "counts": counts,
        "distinct_pass_opcodes": sorted(family_hits),
        "case_count": len(cases),
        "cases": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
