#!/usr/bin/env python3
"""Probe two-CTA tcgen05.copy.warpx2::02_13 with patched PTX.

The public lowering keeps the two-CTA 02_13 candidate cleanly unsupported until
Triton can synthesize a correct shared-memory descriptor plan. This experiment
starts from the known-good two-CTA 01_23 kernel's compile-only warmup,
patches only the PTX copy opcode / descriptor immediate / TMEM destination
offset, assembles the result with ptxas, and launches through Triton's normal
cluster-aware CUDA launcher. Use --prime-canonical only to reproduce historical
probes that launched the canonical 01_23 kernel before the patched cubin.
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

import torch

from triton.backends.nvidia.compiler import get_ptxas, sm_arch_from_capability
from triton.runtime import driver

from python.test.gluon.test_tmem_runtime_matrix import (
    _expected_tmem_copy_warpx2_01_23_twocta_output,
    _make_tmem_copy_warpx2_shared_layout_twocta,
    _make_tmem_copy_warpx2_tmem_layout_twocta,
    tmem_copy_no_scales_warpx2_twocta_kernel,
)


BASE_DESCRIPTOR_IMM = 70403103916032  # 0x400800000000
SOURCE_ROW_PLUS_16_IMM = 70403103916064  # 0x400800000020
SINGLE_CTA_DIRECT_SEED_IMM = 2322202917601312  # 0x8400800000020


def direct_seed_imm(source_offset_b128: int) -> int:
    """Descriptor immediate used by the direct-seed lowering path."""
    seed = (1 << 46) | (8 << 32)
    return seed + source_offset_b128 + (((source_offset_b128 >> 3) & 0x7) << 49)


@dataclass(frozen=True)
class CopyMessage:
    op: str
    imm: int
    dst_delta: int


@dataclass(frozen=True)
class Variant:
    name: str
    messages: tuple[CopyMessage, ...]


VARIANTS = (
    Variant("opcode_only", (CopyMessage("or", BASE_DESCRIPTOR_IMM, 0),)),
    Variant("opcode_dst4", (CopyMessage("or", BASE_DESCRIPTOR_IMM, 4),)),
    Variant("source_row_plus16", (CopyMessage("or", SOURCE_ROW_PLUS_16_IMM, 0),)),
    Variant("source_row_plus16_dst4", (CopyMessage("or", SOURCE_ROW_PLUS_16_IMM, 4),)),
    Variant("single_seed", (CopyMessage("add", SINGLE_CTA_DIRECT_SEED_IMM, 0),)),
    Variant("single_seed_dst4", (CopyMessage("add", SINGLE_CTA_DIRECT_SEED_IMM, 4),)),
    Variant("direct_seed_off33", (CopyMessage("add", direct_seed_imm(33), 0),)),
    Variant("direct_seed_off33_dst4", (CopyMessage("add", direct_seed_imm(33), 4),)),
    Variant("direct_seed_off34", (CopyMessage("add", direct_seed_imm(34), 0),)),
    Variant("direct_seed_off34_dst4", (CopyMessage("add", direct_seed_imm(34), 4),)),
    Variant("direct_seed_off35", (CopyMessage("add", direct_seed_imm(35), 0),)),
    Variant("direct_seed_off35_dst4", (CopyMessage("add", direct_seed_imm(35), 4),)),
    Variant(
        "two_msg_direct_off32_0_off32_4",
        (
            CopyMessage("add", direct_seed_imm(32), 0),
            CopyMessage("add", direct_seed_imm(32), 4),
        ),
    ),
    Variant(
        "two_msg_direct_off32_4_off32_0",
        (
            CopyMessage("add", direct_seed_imm(32), 4),
            CopyMessage("add", direct_seed_imm(32), 0),
        ),
    ),
    Variant(
        "two_msg_direct_off32_0_off33_4",
        (
            CopyMessage("add", direct_seed_imm(32), 0),
            CopyMessage("add", direct_seed_imm(33), 4),
        ),
    ),
    Variant(
        "two_msg_direct_off32_4_off33_0",
        (
            CopyMessage("add", direct_seed_imm(32), 4),
            CopyMessage("add", direct_seed_imm(33), 0),
        ),
    ),
    Variant(
        "two_msg_direct_off32_0_off34_4",
        (
            CopyMessage("add", direct_seed_imm(32), 0),
            CopyMessage("add", direct_seed_imm(34), 4),
        ),
    ),
    Variant(
        "two_msg_direct_off32_4_off34_0",
        (
            CopyMessage("add", direct_seed_imm(32), 4),
            CopyMessage("add", direct_seed_imm(34), 0),
        ),
    ),
    Variant(
        "two_msg_direct_off32_0_off35_4",
        (
            CopyMessage("add", direct_seed_imm(32), 0),
            CopyMessage("add", direct_seed_imm(35), 4),
        ),
    ),
    Variant(
        "two_msg_direct_off32_4_off35_0",
        (
            CopyMessage("add", direct_seed_imm(32), 4),
            CopyMessage("add", direct_seed_imm(35), 0),
        ),
    ),
    Variant(
        "two_msg_base0_plus16_4",
        (
            CopyMessage("or", BASE_DESCRIPTOR_IMM, 0),
            CopyMessage("or", SOURCE_ROW_PLUS_16_IMM, 4),
        ),
    ),
    Variant(
        "two_msg_plus16_0_base4",
        (
            CopyMessage("or", SOURCE_ROW_PLUS_16_IMM, 0),
            CopyMessage("or", BASE_DESCRIPTOR_IMM, 4),
        ),
    ),
)

SELECTED_ROWS = (0, 1, 32, 33, 64, 65, 96, 97, 128, 129, 160, 161, 192, 193, 224, 225)


def assemble_ptx(ptx: str, arch: int = 103) -> bytes:
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
        with open(cubin_path, "rb") as cubin:
            return cubin.read()
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


def expected_extended_single_cta_02_13(inp: torch.Tensor) -> torch.Tensor:
    expected = torch.empty_like(inp)
    for row in range(inp.shape[0]):
        cta_base = 128 * (row // 128)
        local_row = row % 128
        src_base = cta_base + ((local_row % 32) // 2) + 48 * ((local_row % 64) // 32) + 16
        src_cols = (0, 1) if local_row % 2 == 0 else (2, 3)
        expected[row, 0] = inp[src_base, src_cols[0]]
        expected[row, 1] = inp[src_base + 32, src_cols[0]]
        expected[row, 2] = inp[src_base, src_cols[1]]
        expected[row, 3] = inp[src_base + 32, src_cols[1]]
    return expected


def compile_seed_kernel(prime_canonical: bool):
    shared_layout = _make_tmem_copy_warpx2_shared_layout_twocta()
    tmem_layout = _make_tmem_copy_warpx2_tmem_layout_twocta()
    inp = torch.arange(256 * 4, device="cuda", dtype=torch.float32).reshape(256, 4)
    out = torch.empty_like(inp)
    compiled = tmem_copy_no_scales_warpx2_twocta_kernel.warmup(
        inp, out, shared_layout, tmem_layout, grid=(1,), num_warps=4, num_ctas=2
    )
    if prime_canonical:
        compiled[(1, 1, 1)](inp, out)
        torch.cuda.synchronize()
        expected = _expected_tmem_copy_warpx2_01_23_twocta_output(inp)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
    return compiled, inp, out


def patch_ptx(ptx: str, variant: Variant) -> str:
    ptx = ptx.replace(
        "tcgen05.cp.cta_group::2.warpx2::01_23.64x128b",
        "tcgen05.cp.cta_group::2.warpx2::02_13.64x128b",
    )
    message_lines: list[str] = []
    for message in variant.messages:
        if message.op == "or":
            message_lines.append(f"\tor.b64 \t%rd2, %rd7, {message.imm};")
        elif message.op == "add":
            message_lines.append(f"\tadd.s64 \t%rd2, %rd7, {message.imm};")
        else:
            raise ValueError(f"unknown descriptor op {message.op}")
        message_lines.extend(
            [
                "\t// begin inline asm",
                (
                    "\t@%p3 tcgen05.cp.cta_group::2.warpx2::02_13.64x128b "
                    f"[ %r13 + {message.dst_delta} ], %rd2;"
                ),
                "\t// end inline asm",
            ]
        )
    replacement = "\n".join(message_lines)
    pattern = (
        r"\tor\.b64\s+%rd2, %rd7, 70403103916032;\n"
        r"\t// begin inline asm\n"
        r"\t@%p3 tcgen05\.cp\.cta_group::2\.warpx2::02_13\.64x128b "
        r"\[ %r13 \+ 0 \], %rd2;\n"
        r"\t// end inline asm"
    )
    ptx, count = re.subn(pattern, replacement, ptx)
    if count != 1:
        raise RuntimeError(f"failed to patch copy sequence; replacement count={count}")
    return ptx


def analyze_variant(variant: Variant, prime_canonical: bool = False) -> dict[str, object]:
    compiled, inp, out = compile_seed_kernel(prime_canonical)
    ptx = patch_ptx(compiled.asm["ptx"], variant)
    cubin = assemble_ptx(ptx)
    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)
    _, function, _, _, _ = driver.active.utils.load_binary(
        compiled.metadata.name, cubin, compiled.metadata.shared, device
    )
    out.fill_(-777)
    compiled.run(
        1,
        1,
        1,
        stream,
        function,
        compiled.packed_metadata,
        None,
        None,
        None,
        inp,
        out,
    )
    torch.cuda.synchronize()

    expected = expected_extended_single_cta_02_13(inp)
    return {
        "variant": variant.name,
        "messages": [asdict(message) for message in variant.messages],
        "prime_canonical": bool(prime_canonical),
        "status": "ok",
        "matches_extended_single_cta_formula": bool(torch.equal(out, expected)),
        "duplicates_col_pair": bool(
            torch.equal(out[:, 0], out[:, 2]) and torch.equal(out[:, 1], out[:, 3])
        ),
        "sentinel_count": int((out == -777).sum().item()),
        "nan_count": int(torch.isnan(out).sum().item()),
        "selected_rows": {str(row): out[row].detach().cpu().tolist() for row in SELECTED_ROWS},
    }


def print_variant_record(record: dict[str, object]) -> None:
    print(f"VARIANT {record['variant']}")
    print(f"messages={record['messages']}")
    print(f"matches_extended_single_cta_formula={record['matches_extended_single_cta_formula']}")
    print(f"duplicates_col_pair={record['duplicates_col_pair']}")
    print(f"sentinel_count={record['sentinel_count']}")
    print(f"nan_count={record['nan_count']}")
    selected_rows = record.get("selected_rows", {})
    for row in SELECTED_ROWS:
        print(f"row {row}: {selected_rows[str(row)]}")


def run_variant(variant: Variant, prime_canonical: bool) -> int:
    print_variant_record(analyze_variant(variant, prime_canonical))
    return 0


def parse_int_csv(text: str) -> tuple[int, ...]:
    values = []
    for part in text.split(","):
        item = part.strip()
        if item:
            values.append(int(item, 0))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return tuple(values)


def run_source_offset_child(
    source_offset_b128: int, dst_delta: int, json_record: bool, prime_canonical: bool
) -> int:
    variant = Variant(
        f"direct_seed_off{source_offset_b128}_dst{dst_delta}",
        (CopyMessage("add", direct_seed_imm(source_offset_b128), dst_delta),),
    )
    record = analyze_variant(variant, prime_canonical)
    record["source_offset_b128"] = source_offset_b128
    record["dst_delta"] = dst_delta
    if json_record:
        record.pop("selected_rows", None)
        print(json.dumps(record, sort_keys=True))
    else:
        print_variant_record(record)
    return 0


def run_source_offset_parent(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    output_path = Path(args.jsonl_output) if args.jsonl_output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")
    status = 0
    for source_offset_b128 in range(args.source_offset_start, args.source_offset_end + 1):
        for dst_delta in args.dst_deltas:
            env = os.environ.copy()
            env.setdefault(
                "TRITON_CACHE_DIR",
                f"/tmp/triton-cache-warpx2-02-13-off{source_offset_b128}-dst{dst_delta}",
            )
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--source-offset",
                        str(source_offset_b128),
                        "--dst-delta",
                        str(dst_delta),
                        "--json-record",
                        *(["--prime-canonical"] if args.prime_canonical else []),
                    ],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=args.child_timeout,
                )
            except subprocess.TimeoutExpired as exc:
                status = 124
                output = exc.stdout if isinstance(exc.stdout, str) else ""
                record = {
                    "variant": f"direct_seed_off{source_offset_b128}_dst{dst_delta}",
                    "source_offset_b128": source_offset_b128,
                    "dst_delta": dst_delta,
                    "status": "timeout",
                    "prime_canonical": bool(args.prime_canonical),
                    "timeout_s": args.child_timeout,
                    "output_tail": output[-2000:],
                }
            else:
                record = parse_child_record(
                    proc, source_offset_b128, dst_delta, args.prime_canonical
                )
                if proc.returncode != 0:
                    status = max(status, proc.returncode)
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            if output_path:
                with output_path.open("a") as f:
                    f.write(line + "\n")
    return status


def parse_child_record(
    proc: subprocess.CompletedProcess[str],
    source_offset_b128: int,
    dst_delta: int,
    prime_canonical: bool,
) -> dict[str, object]:
    if proc.returncode != 0:
        return {
            "variant": f"direct_seed_off{source_offset_b128}_dst{dst_delta}",
            "source_offset_b128": source_offset_b128,
            "dst_delta": dst_delta,
            "status": "failed",
            "prime_canonical": bool(prime_canonical),
            "returncode": proc.returncode,
            "output_tail": proc.stdout[-2000:],
        }
    json_line = proc.stdout.strip().splitlines()[-1]
    return json.loads(json_line)


def run_parent(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    status = 0
    for variant in VARIANTS:
        env = os.environ.copy()
        env.setdefault("TRITON_CACHE_DIR", f"/tmp/triton-cache-warpx2-02-13-{variant.name}")
        print(f"===== RUN {variant.name} =====", flush=True)
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--variant",
                variant.name,
                *(["--prime-canonical"] if args.prime_canonical else []),
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
        print(proc.stdout, end="")
        print(f"RETURN_CODE {proc.returncode}", flush=True)
        status = max(status, proc.returncode)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=[variant.name for variant in VARIANTS])
    parser.add_argument("--source-offset", type=int)
    parser.add_argument("--dst-delta", type=int)
    parser.add_argument("--json-record", action="store_true")
    parser.add_argument(
        "--prime-canonical",
        action="store_true",
        help="Launch the known-good two-CTA 01_23 kernel before the patched cubin. This reproduces historical primed probes but is not support evidence.",
    )
    parser.add_argument("--source-offset-start", type=int)
    parser.add_argument("--source-offset-end", type=int)
    parser.add_argument("--dst-deltas", type=parse_int_csv, default=(0, 4))
    parser.add_argument("--jsonl-output")
    parser.add_argument("--child-timeout", type=int, default=90)
    args = parser.parse_args()
    if args.source_offset is not None:
        if args.dst_delta is None:
            parser.error("--source-offset requires --dst-delta")
        return run_source_offset_child(
            args.source_offset, args.dst_delta, args.json_record, args.prime_canonical
        )
    if args.source_offset_start is not None or args.source_offset_end is not None:
        if args.source_offset_start is None or args.source_offset_end is None:
            parser.error("--source-offset-start and --source-offset-end must be provided together")
        if args.source_offset_end < args.source_offset_start:
            parser.error("--source-offset-end must be >= --source-offset-start")
        return run_source_offset_parent(args)
    if args.variant is None:
        return run_parent(args)
    variant = next(variant for variant in VARIANTS if variant.name == args.variant)
    return run_variant(variant, args.prime_canonical)


if __name__ == "__main__":
    raise SystemExit(main())
