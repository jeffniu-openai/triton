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
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from triton.backends.nvidia.compiler import get_ptxas, sm_arch_from_capability
from triton.runtime import driver

REPO_ROOT = Path(__file__).resolve().parents[4]
GLUON_TEST_DIR = REPO_ROOT / "python" / "test" / "gluon"
if str(GLUON_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(GLUON_TEST_DIR))

from test_tmem_runtime_matrix import tmem_copy_scales_warpx4_kernel

BASE_DESC_IMM = 70403103916032
SEED_DESC_IMM = 2322202917601312
WARPX4_OPCODE = "tcgen05.cp.cta_group::1.warpx4.32x128b"
WARPX2_01_23_OPCODE = "tcgen05.cp.cta_group::1.warpx2::01_23.64x128b"
WARPX2_02_13_OPCODE = "tcgen05.cp.cta_group::1.warpx2::02_13.64x128b"
SELECTED_ROWS = (0, 1, 2, 15, 16, 31, 32, 33, 48, 63)


def direct_seed_imm(source_offset_b128: int) -> int:
    seed = (1 << 46) | (8 << 32)
    return seed + source_offset_b128 + (((source_offset_b128 >> 3) & 0x7) << 49)


@dataclass(frozen=True)
class CopyMessage:
    opcode: Optional[str]
    desc_op: str
    imm: int
    dst_delta: int


@dataclass(frozen=True)
class Variant:
    name: str
    messages: tuple[CopyMessage, ...]


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
    replacement_lines = []
    for message in variant.messages:
        replacement_lines.append(_desc_line(message.desc_op, "%rd2", "%rd8", message.imm))
        replacement_lines.append("\t// begin inline asm")
        replacement_lines.append(_cp_line("%p3", message.opcode, f"%r22 + {message.dst_delta}", "%rd2"))
        replacement_lines.append("\t// end inline asm")
    replacement = "\n".join(replacement_lines)
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
    finally:
        for temp_path in (ptx_path, cubin_path, log_path):
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass


def make_input(kind: str) -> torch.Tensor:
    if kind == "arange":
        return torch.arange(64 * 16, dtype=torch.int8, device="cuda").reshape(64, 16)
    torch.manual_seed(0)
    return torch.randint(-100, 100, (64, 16), dtype=torch.int8, device="cuda")


def compile_seed(inp: torch.Tensor, prime_canonical: bool):
    out = torch.empty_like(inp)
    compiled = tmem_copy_scales_warpx4_kernel.warmup(inp, out, grid=(1,), num_warps=4)
    if prime_canonical:
        compiled[(1, 1, 1)](inp, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, inp, atol=0, rtol=0)
    return compiled, out


def run_variant(variant: Variant, input_kind: str, arch: int, prime_canonical: bool) -> dict:
    inp = make_input(input_kind)
    compiled, out = compile_seed(inp, prime_canonical)
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
    diff_by_col = (out != inp).sum(dim=0).detach().cpu().tolist()
    same_by_col = (out == inp).sum(dim=0).detach().cpu().tolist()
    return {
        "variant": variant.name,
        "input": input_kind,
        "prime_canonical": bool(prime_canonical),
        "matches_input": bool(equal),
        "diff_count": int((out != inp).sum().item()),
        "same_count": int((out == inp).sum().item()),
        "sentinel_count": int((out == sentinel).sum().item()),
        "diff_by_col": [int(value) for value in diff_by_col],
        "same_by_col": [int(value) for value in same_by_col],
        "row0": out[0].detach().cpu().tolist(),
        "row32": out[32].detach().cpu().tolist(),
        "selected_rows": {str(row): out[row].detach().cpu().tolist() for row in SELECTED_ROWS},
    }


VARIANTS = {
    "warpx4_control": Variant(
        "warpx4_control",
        (
            CopyMessage(WARPX4_OPCODE, "or", BASE_DESC_IMM, 0),
            CopyMessage(WARPX4_OPCODE, "add", SEED_DESC_IMM, 4),
        ),
    ),
    "both_01_23_original_descs": Variant(
        "both_01_23_original_descs",
        (
            CopyMessage(WARPX2_01_23_OPCODE, "or", BASE_DESC_IMM, 0),
            CopyMessage(WARPX2_01_23_OPCODE, "add", SEED_DESC_IMM, 4),
        ),
    ),
    "both_02_13_original_descs": Variant(
        "both_02_13_original_descs",
        (
            CopyMessage(WARPX2_02_13_OPCODE, "or", BASE_DESC_IMM, 0),
            CopyMessage(WARPX2_02_13_OPCODE, "add", SEED_DESC_IMM, 4),
        ),
    ),
    "first_01_23_only": Variant(
        "first_01_23_only", (CopyMessage(WARPX2_01_23_OPCODE, "or", BASE_DESC_IMM, 0),)
    ),
    "first_02_13_only": Variant(
        "first_02_13_only", (CopyMessage(WARPX2_02_13_OPCODE, "or", BASE_DESC_IMM, 0),)
    ),
    "second_01_23_only": Variant(
        "second_01_23_only", (CopyMessage(WARPX2_01_23_OPCODE, "add", SEED_DESC_IMM, 4),)
    ),
    "second_02_13_only": Variant(
        "second_02_13_only", (CopyMessage(WARPX2_02_13_OPCODE, "add", SEED_DESC_IMM, 4),)
    ),
}

OPCODES = {
    "warpx4": WARPX4_OPCODE,
    "01_23": WARPX2_01_23_OPCODE,
    "02_13": WARPX2_02_13_OPCODE,
}


def parse_int_csv(text: str) -> tuple[int, ...]:
    values = []
    for part in text.split(","):
        item = part.strip()
        if item:
            values.append(int(item, 0))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return tuple(values)


def parse_opcode_csv(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one opcode")
    unknown = sorted(set(values) - set(OPCODES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown opcode(s): {unknown}")
    return values


def run_source_offset_child(args: argparse.Namespace) -> int:
    opcode = OPCODES[args.opcode]
    desc_op = "or" if args.source_offset == 0 else "add"
    imm = BASE_DESC_IMM if args.source_offset == 0 else direct_seed_imm(args.source_offset)
    variant = Variant(
        f"{args.opcode}_source_off{args.source_offset}_dst{args.dst_delta}",
        (CopyMessage(opcode, desc_op, imm, args.dst_delta),),
    )
    try:
        result = run_variant(variant, args.input, args.arch, args.prime_canonical)
    except Exception as exc:
        result = {
            "variant": variant.name,
            "opcode": args.opcode,
            "source_offset_b128": args.source_offset,
            "dst_delta": args.dst_delta,
            "input": args.input,
            "prime_canonical": bool(args.prime_canonical),
            "error": type(exc).__name__,
            "message": str(exc)[:500],
        }
        print(json.dumps(result, sort_keys=True))
        return 1
    result["opcode"] = args.opcode
    result["source_offset_b128"] = args.source_offset
    result["dst_delta"] = args.dst_delta
    result["messages"] = [asdict(message) for message in variant.messages]
    print(json.dumps(result, sort_keys=True))
    return 0


def parse_child_record(
    proc: subprocess.CompletedProcess[str],
    opcode: str,
    source_offset_b128: int,
    dst_delta: int,
    input_kind: str,
    prime_canonical: bool,
) -> dict[str, object]:
    if proc.returncode != 0:
        try:
            parsed = json.loads(proc.stdout.strip().splitlines()[-1])
            if isinstance(parsed, dict):
                parsed["returncode"] = proc.returncode
                return parsed
        except Exception:
            pass
        return {
            "variant": f"{opcode}_source_off{source_offset_b128}_dst{dst_delta}",
            "opcode": opcode,
            "source_offset_b128": source_offset_b128,
            "dst_delta": dst_delta,
            "input": input_kind,
            "prime_canonical": bool(prime_canonical),
            "returncode": proc.returncode,
            "output_tail": proc.stdout[-2000:],
        }
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_source_offset_parent(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    output_path = Path(args.jsonl_output) if args.jsonl_output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")
    status = 0
    for source_offset_b128 in range(args.source_offset_start, args.source_offset_end + 1):
        for dst_delta in args.dst_deltas:
            for opcode in args.opcodes:
                env = os.environ.copy()
                env.setdefault(
                    "TRITON_CACHE_DIR",
                    f"/tmp/triton-cache-scales-warpx2-{opcode}-off{source_offset_b128}-dst{dst_delta}",
                )
                cmd = [
                    sys.executable,
                    str(script),
                    "--source-offset",
                    str(source_offset_b128),
                    "--dst-delta",
                    str(dst_delta),
                    "--opcode",
                    opcode,
                    "--input",
                    args.input,
                    "--arch",
                    str(args.arch),
                    *( ["--prime-canonical"] if args.prime_canonical else [] ),
                ]
                try:
                    proc = subprocess.run(
                        cmd,
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=args.child_timeout,
                    )
                except subprocess.TimeoutExpired as exc:
                    output = exc.stdout if isinstance(exc.stdout, str) else ""
                    record = {
                        "variant": f"{opcode}_source_off{source_offset_b128}_dst{dst_delta}",
                        "opcode": opcode,
                        "source_offset_b128": source_offset_b128,
                        "dst_delta": dst_delta,
                        "input": args.input,
                        "prime_canonical": bool(args.prime_canonical),
                        "status": "timeout",
                        "timeout_s": args.child_timeout,
                        "output_tail": output[-2000:],
                    }
                    status = max(status, 124)
                else:
                    record = parse_child_record(
                        proc,
                        opcode,
                        source_offset_b128,
                        dst_delta,
                        args.input,
                        args.prime_canonical,
                    )
                    if proc.returncode != 0:
                        status = max(status, proc.returncode)
                line = json.dumps(record, sort_keys=True)
                print(line, flush=True)
                if output_path:
                    with output_path.open("a") as f:
                        f.write(line + "\n")
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", nargs="?", choices=sorted(VARIANTS))
    parser.add_argument("--input", choices=("random", "arange"), default="random")
    parser.add_argument("--arch", type=int, default=103)
    parser.add_argument("--source-offset", type=int)
    parser.add_argument("--source-offset-start", type=int)
    parser.add_argument("--source-offset-end", type=int)
    parser.add_argument("--dst-delta", type=int)
    parser.add_argument("--dst-deltas", type=parse_int_csv, default=(0, 4))
    parser.add_argument("--opcode", choices=sorted(OPCODES), default="01_23")
    parser.add_argument("--opcodes", type=parse_opcode_csv, default=("01_23", "02_13"))
    parser.add_argument("--jsonl-output")
    parser.add_argument("--child-timeout", type=int, default=90)
    parser.add_argument(
        "--prime-canonical",
        action="store_true",
        help="Launch the unpatched canonical warpx4 kernel before the patched cubin. This reproduces the historical primed probe but is not valid support evidence.",
    )
    args = parser.parse_args()

    if args.source_offset is not None:
        if args.dst_delta is None:
            parser.error("--source-offset requires --dst-delta")
        return run_source_offset_child(args)
    if args.source_offset_start is not None or args.source_offset_end is not None:
        if args.source_offset_start is None or args.source_offset_end is None:
            parser.error("--source-offset-start and --source-offset-end must be provided together")
        if args.source_offset_end < args.source_offset_start:
            parser.error("--source-offset-end must be >= --source-offset-start")
        return run_source_offset_parent(args)
    if args.variant is None:
        parser.error("variant or source-offset mode is required")

    try:
        result = run_variant(VARIANTS[args.variant], args.input, args.arch, args.prime_canonical)
    except Exception as exc:
        print(json.dumps({"variant": args.variant, "error": type(exc).__name__, "message": str(exc)[:500]}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
