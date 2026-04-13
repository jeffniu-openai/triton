#!/usr/bin/env python3
"""Direct-PTX probe for tensor-memory-scales tcgen05.copy warpx2 aliases.

This is an experiment, not a supported lowering path. It starts from the
known-good public scales warpx4 copy kernel, patches the generated PTX copy
sequence, assembles the result with ptxas, and launches the patched cubin in a
fresh process. Run one variant per process because illegal copy variants poison
the CUDA context.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

import torch
from triton.backends.nvidia.compiler import get_ptxas, sm_arch_from_capability
from triton.runtime import driver

from python.test.gluon.test_tmem_runtime_matrix import tmem_copy_scales_warpx4_kernel

BASE_DESC_IMM = 70403103916032
SEED_DESC_IMM = 2322202917601312
WARPX4_OPCODE = "tcgen05.cp.cta_group::1.warpx4.32x128b"
WARPX2_01_23_OPCODE = "tcgen05.cp.cta_group::1.warpx2::01_23.64x128b"
WARPX2_02_13_OPCODE = "tcgen05.cp.cta_group::1.warpx2::02_13.64x128b"


@dataclass(frozen=True)
class Variant:
    name: str
    first_opcode: Optional[str]
    second_opcode: Optional[str]


def _desc_line(op: str, dst: str, src: str, imm: int) -> str:
    if op == "or":
        return f"\tor.b64 \t{dst}, {src}, {imm};"
    if op == "add":
        return f"\tadd.s64 \t{dst}, {src}, {imm};"
    raise ValueError(op)


def _cp_line(pred: str, opcode: Optional[str], dst_expr: str, desc: str) -> str:
    if opcode is None:
        return "\t// skipped patched tcgen05.cp"
    return f"\t@{pred} {opcode} [ {dst_expr} ], {desc};"


def patch_ptx(ptx: str, variant: Variant) -> str:
    replacement = "\n".join(
        [
            _desc_line("or", "%rd2", "%rd8", BASE_DESC_IMM),
            "\t// begin inline asm",
            _cp_line("%p3", variant.first_opcode, "%r22 + 0", "%rd2"),
            "\t// end inline asm",
            _desc_line("add", "%rd3", "%rd8", SEED_DESC_IMM),
            "\tadd.s32 \t%r7, %r22, 4;",
            "\t// begin inline asm",
            _cp_line("%p3", variant.second_opcode, "%r7 + 0", "%rd3"),
            "\t// end inline asm",
        ]
    )
    pattern = (
        r"\tor\.b64\s+%rd2, %rd8, 70403103916032;\n"
        r"\t// begin inline asm\n"
        r"\t@%p3 tcgen05\.cp\.cta_group::1\.warpx4\.32x128b \[ %r22 \+ 0 \], %rd2;\n"
        r"\t// end inline asm\n"
        r"\tadd\.s64\s+%rd3, %rd8, 2322202917601312;\n"
        r"\tadd\.s32\s+%r7, %r22, 4;\n"
        r"\t// begin inline asm\n"
        r"\t@%p3 tcgen05\.cp\.cta_group::1\.warpx4\.32x128b \[ %r7 \+ 0 \], %rd3;\n"
        r"\t// end inline asm"
    )
    patched, count = re.subn(pattern, replacement, ptx)
    if count != 1:
        raise RuntimeError(f"expected one canonical copy sequence, found {count}")
    return patched


def assemble_ptx(ptx: str, arch: int) -> bytes:
    ptxas = get_ptxas(arch).path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ptx", mode="w") as src:
        src.write(ptx)
        src.flush()
        ptx_path = src.name
    cubin_path = ptx_path + ".o"
    log_path = ptx_path + ".log"
    try:
        with open(log_path, "w") as log:
            subprocess.run(
                [
                    ptxas,
                    "-lineinfo",
                    "-v",
                    "--regAllocOptLevel=2",
                    f"--gpu-name={sm_arch_from_capability(arch)}",
                    ptx_path,
                    "-o",
                    cubin_path,
                ],
                check=True,
                stderr=log,
            )
        with open(cubin_path, "rb") as f:
            return f.read()
    except subprocess.CalledProcessError:
        with open(log_path) as log:
            print(log.read(), file=sys.stderr)
        raise


def make_input(kind: str) -> torch.Tensor:
    if kind == "arange":
        return torch.arange(64 * 16, dtype=torch.int8, device="cuda").reshape(64, 16)
    torch.manual_seed(0)
    return torch.randint(-100, 100, (64, 16), dtype=torch.int8, device="cuda")


def compile_seed(inp: torch.Tensor):
    out = torch.empty_like(inp)
    compiled = tmem_copy_scales_warpx4_kernel[(1,)](inp, out, num_warps=4)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)
    return compiled, out


def run_variant(variant: Variant, input_kind: str, arch: int) -> dict:
    inp = make_input(input_kind)
    compiled, out = compile_seed(inp)
    patched_ptx = patch_ptx(compiled.asm["ptx"], variant)
    cubin = assemble_ptx(patched_ptx, arch)
    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)
    _, function, _, _, _ = driver.active.utils.load_binary(
        compiled.metadata.name, cubin, compiled.metadata.shared, device
    )

    sentinel = torch.empty((), dtype=torch.int8, device="cuda").fill_(-77).item()
    out.fill_(sentinel)
    compiled.run(1, 1, 1, stream, function, compiled.packed_metadata, None, None, None, inp, out)
    torch.cuda.synchronize()

    equal = torch.equal(out, inp)
    return {
        "variant": variant.name,
        "input": input_kind,
        "matches_input": bool(equal),
        "diff_count": int((out != inp).sum().item()),
        "same_count": int((out == inp).sum().item()),
        "sentinel_count": int((out == sentinel).sum().item()),
        "row0": out[0].detach().cpu().tolist(),
        "row32": out[32].detach().cpu().tolist(),
    }


VARIANTS = {
    "warpx4_control": Variant("warpx4_control", WARPX4_OPCODE, WARPX4_OPCODE),
    "both_01_23_original_descs": Variant(
        "both_01_23_original_descs", WARPX2_01_23_OPCODE, WARPX2_01_23_OPCODE
    ),
    "both_02_13_original_descs": Variant(
        "both_02_13_original_descs", WARPX2_02_13_OPCODE, WARPX2_02_13_OPCODE
    ),
    "first_01_23_only": Variant("first_01_23_only", WARPX2_01_23_OPCODE, None),
    "first_02_13_only": Variant("first_02_13_only", WARPX2_02_13_OPCODE, None),
    "second_01_23_only": Variant("second_01_23_only", None, WARPX2_01_23_OPCODE),
    "second_02_13_only": Variant("second_02_13_only", None, WARPX2_02_13_OPCODE),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--input", choices=("random", "arange"), default="random")
    parser.add_argument("--arch", type=int, default=103)
    args = parser.parse_args()

    try:
        result = run_variant(VARIANTS[args.variant], args.input, args.arch)
    except Exception as exc:
        print(json.dumps({"variant": args.variant, "error": type(exc).__name__, "message": str(exc)[:500]}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
