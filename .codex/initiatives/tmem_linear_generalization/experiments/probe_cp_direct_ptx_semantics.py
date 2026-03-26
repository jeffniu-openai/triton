#!/usr/bin/env python3
"""Direct-PTX semantics probe for tcgen05.cp families.

This goes beyond "assembles and launches" and checks whether patched kernels
produce expected outputs for the known scales-copy reference mapping.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import tempfile
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pycuda.driver as cuda
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
TEST_PATH = REPO_ROOT / "python" / "test" / "gluon" / "test_tmem_runtime_matrix.py"
PTXAS_BLACKWELL = REPO_ROOT / "third_party" / "nvidia" / "backend" / "bin" / "ptxas-blackwell"
KERNEL_NAME = "tmem_copy_scales_warpx4_kernel"
BASE_OPCODE = "tcgen05.cp.cta_group::1.warpx4.32x128b"


def load_runtime_matrix_module():
    spec = importlib.util.spec_from_file_location("tmem_runtime_matrix", TEST_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def assemble(ptx_text: str, ptxas: Path) -> Path:
    work = Path(tempfile.mkdtemp(prefix="cp_direct_semantics_"))
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


CP_RE = re.compile(
    r"^(\s*)(@%p\d+)\s+tcgen05\.cp[^\[]*\[\s*([^\]]+?)\s*\]\s*,\s*(%rd\d+)\s*;\s*$"
)


def parse_cp_sites(ptx_lines: List[str]) -> List[Tuple[int, str, str, str, str]]:
    sites = []
    for idx, line in enumerate(ptx_lines):
        if "tcgen05.cp.cta_group::1.warpx4.32x128b" not in line:
            continue
        m = CP_RE.match(line)
        if not m:
            raise RuntimeError(f"Failed to parse cp line: {line}")
        indent, pred, addr, desc = m.groups()
        sites.append((idx, indent, pred, addr, desc))
    return sites


def make_cp_line(indent: str, pred: str, opcode: str, addr: str, desc: str) -> str:
    return f"{indent}{pred} {opcode} [ {addr} ], {desc};"


def make_nop_line(indent: str, pred: str) -> str:
    return f"{indent}{pred} mov.u32 %r0, %r0;"


def patched_ptx(base_ptx: str, opcode: str, mode: str) -> str:
    lines = base_ptx.splitlines()
    cp_sites = parse_cp_sites(lines)
    if len(cp_sites) != 2:
        raise RuntimeError(f"Expected exactly 2 cp sites, got {len(cp_sites)}")

    (i0, ind0, pred0, addr0, desc0), (i1, ind1, pred1, addr1, desc1) = cp_sites

    if mode == "both_orig":
        lines[i0] = make_cp_line(ind0, pred0, opcode, addr0, desc0)
        lines[i1] = make_cp_line(ind1, pred1, opcode, addr1, desc1)
    elif mode == "first_only":
        lines[i0] = make_cp_line(ind0, pred0, opcode, addr0, desc0)
        lines[i1] = make_nop_line(ind1, pred1)
    elif mode == "second_only":
        lines[i0] = make_nop_line(ind0, pred0)
        lines[i1] = make_cp_line(ind1, pred1, opcode, addr1, desc1)
    elif mode == "both_first_map":
        lines[i0] = make_cp_line(ind0, pred0, opcode, addr0, desc0)
        lines[i1] = make_cp_line(ind1, pred1, opcode, addr0, desc0)
    elif mode == "both_second_map":
        lines[i0] = make_cp_line(ind0, pred0, opcode, addr1, desc1)
        lines[i1] = make_cp_line(ind1, pred1, opcode, addr1, desc1)
    elif mode == "swap_maps":
        lines[i0] = make_cp_line(ind0, pred0, opcode, addr1, desc1)
        lines[i1] = make_cp_line(ind1, pred1, opcode, addr0, desc0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return "\n".join(lines) + "\n"


def expected_warpx4_output(inp: np.ndarray) -> np.ndarray:
    # Matches the runtime reference in test_tmem_runtime_matrix.py.
    tile = inp.reshape(2, 32, 2, 2, 4).transpose(1, 2, 3, 0, 4).reshape(32, 32)
    return np.concatenate([tile, tile, tile, tile], axis=0)


def chunk_output(out: np.ndarray) -> List[np.ndarray]:
    assert out.shape == (128, 32)
    return [out[i * 32:(i + 1) * 32, :] for i in range(4)]


def analyze_output(
    out: np.ndarray,
    expected_full: np.ndarray,
    tile_a: np.ndarray,
    tile_b: np.ndarray,
) -> Dict[str, object]:
    chunks = chunk_output(out)
    expected_chunks = chunk_output(expected_full)

    chunk_equals_expected = [bool(np.array_equal(ch, expected_chunks[i])) for i, ch in enumerate(chunks)]
    eq = [[bool(np.array_equal(chunks[i], chunks[j])) for j in range(4)] for i in range(4)]

    def match_tile(ch: np.ndarray) -> str:
        if np.array_equal(ch, tile_a):
            return "A"
        if np.array_equal(ch, tile_b):
            return "B"
        return "other"

    tile_map = [match_tile(ch) for ch in chunks]

    return {
        "exact_expected": bool(np.array_equal(out, expected_full)),
        "num_mismatch_expected": int(np.count_nonzero(out != expected_full)),
        "sum": int(out.sum()),
        "chunk_equals_expected": chunk_equals_expected,
        "chunk_pair_equal": {
            "01": eq[0][1],
            "02": eq[0][2],
            "03": eq[0][3],
            "12": eq[1][2],
            "13": eq[1][3],
            "23": eq[2][3],
        },
        "tile_map_vs_controls": tile_map,
        "warpx2_02_13_pattern": bool(eq[0][2] and eq[1][3]),
        "warpx2_01_23_pattern": bool(eq[0][1] and eq[2][3]),
    }


def run_variant(
    ctx: cuda.Context,
    base_ptx: str,
    opcode: str,
    mode: str,
    inp: np.ndarray,
    shared_bytes: int,
    ptxas: Path,
) -> Dict[str, object]:
    rec: Dict[str, object] = {
        "opcode": opcode,
        "mode": mode,
    }
    try:
        ptx_text = patched_ptx(base_ptx, opcode, mode)
        cubin = assemble(ptx_text, ptxas)
        out0 = launch(cubin, inp, shared_bytes)
        out1 = launch(cubin, inp, shared_bytes)
        rec["status"] = "PASS"
        rec["deterministic"] = bool(np.array_equal(out0, out1))
        rec["cubin"] = str(cubin)
        rec["out"] = out0
    except Exception as exc:
        rec["status"] = "FAIL"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        # Recover CUDA context after possible launch failure.
        try:
            ctx.pop()
        except Exception:
            pass
        new_ctx = cuda.Device(0).make_context()
        rec["new_ctx"] = new_ctx
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ptxas",
        type=Path,
        default=PTXAS_BLACKWELL,
        help="Path to ptxas-blackwell",
    )
    parser.add_argument("--shared-bytes", type=int, default=2048)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / ".codex" / "initiatives" / "tmem_linear_generalization" / "experiments" / "results" /
        "probe_cp_direct_ptx_semantics_gpu2.json",
    )
    args = parser.parse_args()

    runtime = load_runtime_matrix_module()
    torch.manual_seed(0)
    inp_t = torch.randint(-100, 100, (64, 16), dtype=torch.int8, device="cuda")
    out_t = torch.zeros((128, 32), dtype=torch.int8, device="cuda")
    compiled = runtime.tmem_copy_scales_warpx4_kernel[(1,)](inp_t, out_t)
    base_ptx = compiled.asm["ptx"]

    rng = np.random.default_rng(123)
    inp = rng.integers(-100, 100, size=(64, 16), dtype=np.int8)
    expected_full = expected_warpx4_output(inp)

    modes = [
        "both_orig",
        "first_only",
        "second_only",
        "both_first_map",
        "both_second_map",
        "swap_maps",
    ]
    families = {
        "warpx4_control": BASE_OPCODE,
        "warpx2_02_13": "tcgen05.cp.cta_group::1.warpx2::02_13.64x128b",
        "warpx2_01_23": "tcgen05.cp.cta_group::1.warpx2::01_23.64x128b",
        "cp_4x256b": "tcgen05.cp.cta_group::1.4x256b",
    }

    results: Dict[str, object] = {
        "input_seed": 123,
        "families": {},
    }

    cuda.init()
    ctx = cuda.Device(0).make_context()
    try:
        # First collect control tiles for descriptor-map matching.
        control_first = run_variant(ctx, base_ptx, BASE_OPCODE, "first_only", inp, args.shared_bytes, args.ptxas)
        if "new_ctx" in control_first:
            ctx = control_first.pop("new_ctx")
        control_second = run_variant(ctx, base_ptx, BASE_OPCODE, "second_only", inp, args.shared_bytes, args.ptxas)
        if "new_ctx" in control_second:
            ctx = control_second.pop("new_ctx")

        if control_first.get("status") != "PASS" or control_second.get("status") != "PASS":
            raise RuntimeError(
                "Failed to build control tiles with warpx4 first_only/second_only; "
                f"first={control_first.get('error')} second={control_second.get('error')}"
            )

        tile_a = chunk_output(control_first["out"])[0]
        tile_b = chunk_output(control_second["out"])[0]

        for family, opcode in families.items():
            family_records = []
            for mode in modes:
                rec = run_variant(ctx, base_ptx, opcode, mode, inp, args.shared_bytes, args.ptxas)
                if "new_ctx" in rec:
                    ctx = rec.pop("new_ctx")
                if rec.get("status") == "PASS":
                    analysis = analyze_output(rec["out"], expected_full, tile_a, tile_b)
                    rec.update(analysis)
                    rec.pop("out", None)
                family_records.append(rec)
            results["families"][family] = family_records
    finally:
        try:
            ctx.pop()
        except Exception:
            pass

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
