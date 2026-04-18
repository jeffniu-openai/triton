#!/usr/bin/env python3
"""Generate a runnable sm100 PTX artifact for signed i8 tcgen05.mma.

This script intentionally runs in the local Triton checkout.  The generated
PTX can then be copied to a GB200 host and launched with run_i8_ptx_torch.py
without rebuilding Triton in the remote container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
OUT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "python/test/gluon"))

import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.experimental.gluon import language as ttgl  # noqa: E402
from triton.experimental.gluon._runtime import GluonASTSource  # noqa: E402
from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLayout  # noqa: E402
from tmem_test_utils import mma_kernel  # noqa: E402


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def make_source() -> GluonASTSource:
    threads_per_warp = 32
    m = n = 128
    k = 32

    block_layout_a = ttgl.BlockedLayout([1, 8], [1, threads_per_warp], warps_per_cta=[4, 1], order=[0, 1])
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, threads_per_warp], warps_per_cta=[4, 1], order=[1, 0])
    shared_layout_a = ttgl.NVMMASharedLayout(
        swizzle_byte_width=32,
        transposed=False,
        element_bitwidth=8,
        rank=2,
    )
    shared_layout_b = ttgl.NVMMASharedLayout(
        swizzle_byte_width=32,
        transposed=True,
        element_bitwidth=8,
        rank=2,
    )
    acc_layout = TensorMemoryLayout((m, n), col_stride=1)

    signature = {
        "a": "*i8",
        "b": "*i8",
        "out": "*i32",
        "M": "constexpr",
        "N": "constexpr",
        "K": "constexpr",
        "block_layout_a": "constexpr",
        "block_layout_b": "constexpr",
        "cga_layout_c": "constexpr",
        "acc_layout": "constexpr",
        "shared_layout_a": "constexpr",
        "shared_layout_b": "constexpr",
        "acc_dtype": "constexpr",
        "ASYNC": "constexpr",
        "USE_TCGEN05": "constexpr",
    }
    constexprs = {
        "M": m,
        "N": n,
        "K": k,
        "block_layout_a": block_layout_a,
        "block_layout_b": block_layout_b,
        "cga_layout_c": (),
        "acc_layout": acc_layout,
        "shared_layout_a": shared_layout_a,
        "shared_layout_b": shared_layout_b,
        "acc_dtype": ttgl.int32,
        "ASYNC": False,
        "USE_TCGEN05": True,
    }
    return GluonASTSource(fn=mma_kernel, signature=signature, constexprs=constexprs)


def extract_entry_param_count(ptx: str, kernel_name: str) -> int:
    match = re.search(rf"\.visible \.entry {re.escape(kernel_name)}\((.*?)\)\s*\.reqntid", ptx, re.S)
    if not match:
        raise RuntimeError(f"could not find .entry prototype for {kernel_name}")
    return len(re.findall(r"\.param\b", match.group(1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target = GPUTarget("cuda", 100, 32)
    compiled = triton.compile(make_source(), target=target, options={"num_warps": 4})
    ptx = compiled.asm["ptx"]

    kernel_name = compiled.metadata.name
    ptx_file = out_dir / "tcgen05_i8_signed_sm100.ptx"
    cubin_file = out_dir / "tcgen05_i8_signed_sm100.cubin"
    metadata_file = out_dir / "tcgen05_i8_signed_sm100.metadata.json"
    ptx_file.write_text(ptx)
    cubin = compiled.asm.get("cubin", b"")
    if not cubin:
        raise RuntimeError("expected ptxas cubin artifact for sm100 compile")
    cubin_file.write_bytes(cubin)

    mma_ops = re.findall(r"tcgen05\.mma\.cta_group::\d+\.kind::[^\s;]+", ptx)
    if mma_ops != ["tcgen05.mma.cta_group::1.kind::i8"]:
        raise RuntimeError(f"unexpected tcgen05.mma opcodes: {mma_ops}")
    descriptor_immediates = sorted({int(value) for value in re.findall(r"mov\.b32\s+%r\d+,\s*(\d+);", ptx)})
    if 136316064 not in descriptor_immediates:
        raise RuntimeError(f"expected signed i8 descriptor immediate missing: {descriptor_immediates}")

    metadata = {
        "description": "Signed i8 tcgen05.mma frontend/runtime smoke test generated from Gluon mma_kernel.",
        "triton_head": git_head(),
        "target": "sm_100a",
        "ptx_version": re.search(r"\.version\s+([0-9.]+)", ptx).group(1),
        "kernel_name": kernel_name,
        "ptx_file": ptx_file.name,
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        "cubin_file": cubin_file.name,
        "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
        "shape": {"M": 128, "N": 128, "K": 32},
        "dtypes": {"a": "torch.int8", "b": "torch.int8", "out": "torch.int32"},
        "launch": {
            "grid": [1, 1, 1],
            "block": [128, 1, 1],
            "dynamic_shared_memory_bytes": compiled.metadata.shared,
            "num_warps": compiled.metadata.num_warps,
            "entry_param_count": extract_entry_param_count(ptx, kernel_name),
            "live_runtime_params": ["a", "b", "out"],
            "trailing_dummy_u64_params": 2,
        },
        "expected_ptx": {
            "mma_opcodes": mma_ops,
            "descriptor_immediate": 136316064,
            "commit_opcode_substring": "tcgen05.commit.cta_group::1",
        },
        "local_ptxas_cubin_bytes": len(cubin),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {ptx_file}")
    print(f"wrote {cubin_file}")
    print(f"wrote {metadata_file}")


if __name__ == "__main__":
    main()
