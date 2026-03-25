#!/usr/bin/env python3
"""Development-only TMEM probe matrix for ambiguous tcgen05 behavior."""

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


def _run_ldst_family_probes(mod):
    families_found = set()
    cases = [
        ("32x32b", 128, 128, mod._make_tmem_linear_layout(128, 128), "32x32b"),
        ("16x64b", 128, 128, mod._make_tmem_linear_layout(128, 128), "16x64b"),
        ("16x128b", 128, 128, mod._make_tmem_linear_layout(128, 128), "16x128b"),
        ("16x256b", 128, 128, mod._make_tmem_linear_layout(128, 128), "16x256b"),
        ("16x32bx2", 64, 64, mod._make_tmem_linear_layout_m64(64), "32x32b_splitn"),
    ]
    case_results = []
    for family, m, n, layout, variant in cases:
        inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
        out = torch.empty_like(inp)
        compiled = mod.tmem_ldst_variant_kernel[(1,)](inp, out, layout, m, n, variant, num_warps=4)
        torch.testing.assert_close(out, inp, atol=0, rtol=0)

        ptx_ops = mod._extract_tcgen05_opcode_offsets(compiled.asm["ptx"])
        llir_ops = mod._extract_tcgen05_opcode_offsets(compiled.asm["llir"])
        assert ptx_ops == llir_ops and ptx_ops
        opcodes = [op for op, _ in ptx_ops]
        for op in opcodes:
            if ".32x32b." in op:
                families_found.add("32x32b")
            if ".16x64b." in op:
                families_found.add("16x64b")
            if ".16x128b." in op:
                families_found.add("16x128b")
            if ".16x256b." in op:
                families_found.add("16x256b")
            if ".16x32bx2." in op:
                families_found.add("16x32bx2")

        case_results.append(
            {
                "family_target": family,
                "variant": variant,
                "shape": [m, n],
                "opcodes": opcodes,
            }
        )

    return sorted(families_found), case_results


def _run_splitn_immediate_probe(mod):
    probes = []
    observed_immediates = set()
    for n in (2, 64):
        m = 64
        layout = mod._make_tmem_linear_layout_m64(n)
        inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
        out = torch.empty_like(inp)
        compiled = mod.tmem_ldst_variant_kernel[(1,)](inp, out, layout, m, n, "32x32b_splitn", num_warps=4)
        torch.testing.assert_close(out, inp, atol=0, rtol=0)
        ptx_imms = mod._extract_tcgen05_opcode_offset_immediates(compiled.asm["ptx"])
        llir_imms = mod._extract_tcgen05_opcode_offset_immediates(compiled.asm["llir"])
        assert ptx_imms == llir_imms and ptx_imms
        for op, _, imm in ptx_imms:
            assert ".16x32bx2." in op
            observed_immediates.add(imm)
        probes.append({"n": n, "immediates": sorted({imm for _, _, imm in ptx_imms}), "records": ptx_imms})

    assert 0 in observed_immediates
    assert any(imm > 0 for imm in observed_immediates)
    return probes


def _run_cp_probes(mod):
    results = {}

    m, n, block_n, swizzle = 128, 256, 256, 32
    inp = torch.arange(m * n, device="cuda", dtype=torch.int32).reshape(m, n)
    out = torch.empty_like(inp)
    compiled = mod.tmem_copy_no_scales_kernel[(1,)](inp, out, m, n, block_n, swizzle, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    ptx_ops = mod._extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
    llir_ops = mod._extract_tcgen05_cp_opcodes(compiled.asm["llir"])
    assert ptx_ops == llir_ops and ptx_ops
    assert all(op.endswith(".128x256b") for op in ptx_ops)
    results["cp_128x256b"] = {"count": len(ptx_ops), "opcodes": ptx_ops}

    m = 128
    inp = torch.arange(m * 4, device="cuda", dtype=torch.int32).reshape(m, 4)
    out = torch.empty_like(inp)
    compiled = mod.tmem_copy_128x128_kernel[(1,)](inp, out, m, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    ptx_ops = mod._extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
    llir_ops = mod._extract_tcgen05_cp_opcodes(compiled.asm["llir"])
    assert ptx_ops == llir_ops and ptx_ops
    assert all(op == "tcgen05.cp.cta_group::1.128x128b" for op in ptx_ops)
    results["cp_128x128b"] = {"count": len(ptx_ops), "opcodes": ptx_ops}

    smem_h, smem_w = 64, 16
    num_rows = 128
    num_cols = smem_h * smem_w // 32
    inp = torch.randint(size=(smem_h, smem_w), low=-100, high=100, dtype=torch.int8, device="cuda")
    out = torch.zeros(size=(num_rows, num_cols), dtype=torch.int8, device="cuda")
    compiled = mod.tmem_copy_scales_warpx4_kernel[(1,)](inp, out)
    ptx_ops = mod._extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
    llir_ops = mod._extract_tcgen05_cp_opcodes(compiled.asm["llir"])
    assert ptx_ops == llir_ops and ptx_ops
    assert all("warpx4.32x128b" in op for op in ptx_ops)
    results["cp_warpx4_32x128b"] = {"count": len(ptx_ops), "opcodes": ptx_ops}

    return results


def _run_mma_probes(mod):
    mma_cases = [
        ("legacy_use_acc", mod.TensorMemoryLayout((128, 128), col_stride=1), True),
        ("linear_no_acc", mod._make_tmem_linear_layout(128, 128), False),
    ]
    results = {}
    m, n, k = 128, 128, 32
    for name, layout, use_acc in mma_cases:
        a = torch.randn((m, k), dtype=torch.float16, device="cuda")
        b = torch.randn((k, n), dtype=torch.float16, device="cuda")
        c = torch.randn((m, n), dtype=torch.float32, device="cuda")
        out = torch.empty_like(c)
        compiled = mod.tmem_mma_kernel[(1,)](a, b, c, out, layout, use_acc, num_warps=4)
        expected = a.to(torch.float32) @ b.to(torch.float32)
        if use_acc:
            expected = expected + c
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=8e-2)
        assert "tcgen05.mma.cta_group::1.kind::f16" in compiled.asm["ptx"]
        assert "tcgen05.mma.cta_group::1.kind::f16" in compiled.asm["llir"]
        results[name] = {"use_acc": use_acc}
    return results


def main():
    torch.manual_seed(0)
    mod = _load_test_matrix_module()
    families, ldst_cases = _run_ldst_family_probes(mod)
    splitn = _run_splitn_immediate_probe(mod)
    cp = _run_cp_probes(mod)
    mma = _run_mma_probes(mod)
    summary = {
        "ldst_families_found": families,
        "ldst_case_count": len(ldst_cases),
        "splitn_cases": splitn,
        "cp_cases": cp,
        "mma_cases": mma,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
