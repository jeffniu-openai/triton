#!/usr/bin/env python3
"""Direct-PTX execution probe for tcgen05.cp variant opcodes.

Builds a known-good Gluon warpx4 scales-copy kernel, patches the cp opcode,
assembles with ptxas-blackwell, and launches via CUDA driver.
"""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path
import sys

import numpy as np
import pycuda.driver as cuda
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
TEST_PATH = REPO_ROOT / "python" / "test" / "gluon" / "test_tmem_runtime_matrix.py"
_spec = importlib.util.spec_from_file_location("test_tmem_runtime_matrix", TEST_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
tmem_copy_scales_warpx4_kernel = _mod.tmem_copy_scales_warpx4_kernel


PTXAS_BLACKWELL = "/root/code/triton/third_party/nvidia/backend/bin/ptxas-blackwell"
BASE_OPCODE = "tcgen05.cp.cta_group::1.warpx4.32x128b"
KERNEL_NAME = "tmem_copy_scales_warpx4_kernel"


def assemble(ptx_text: str, ptxas: str) -> Path:
    work = Path(tempfile.mkdtemp(prefix="cp_ptx_variant_"))
    ptx_path = work / "kernel.ptx"
    cubin_path = work / "kernel.cubin"
    ptx_path.write_text(ptx_text)
    import subprocess

    subprocess.check_call([ptxas, str(ptx_path), "-arch=sm_103a", "-o", str(cubin_path)])
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-bytes", type=int, default=2048)
    parser.add_argument("--ptxas", type=str, default=PTXAS_BLACKWELL)
    args = parser.parse_args()

    inp_t = torch.randint(-100, 100, (64, 16), dtype=torch.int8, device="cuda")
    out_t = torch.zeros((128, 32), dtype=torch.int8, device="cuda")
    baseline_ptx = tmem_copy_scales_warpx4_kernel[(1,)](inp_t, out_t).asm["ptx"]

    variants = {
        "warpx4_base": BASE_OPCODE,
        "warpx2_02_13": "tcgen05.cp.cta_group::1.warpx2::02_13.64x128b",
        "warpx2_01_23": "tcgen05.cp.cta_group::1.warpx2::01_23.64x128b",
        "cp_4x256b": "tcgen05.cp.cta_group::1.4x256b",
    }

    rng = np.random.default_rng(0)
    inp = rng.integers(-100, 100, size=(64, 16), dtype=np.int8)

    cuda.init()
    ctx = cuda.Device(0).make_context()
    outputs: dict[str, np.ndarray] = {}
    try:
        for name, opcode in variants.items():
            try:
                patched = baseline_ptx.replace(BASE_OPCODE, opcode)
                cubin = assemble(patched, args.ptxas)
            except Exception as exc:
                print(f"{name}: CLEAN_UNSUPPORTED assemble_failed: {type(exc).__name__}: {exc}")
                continue
            try:
                out = launch(cubin, inp, args.shared_bytes)
            except Exception as exc:
                print(f"{name}: BUG launch_failed: {type(exc).__name__}: {exc}")
                # Illegal memory access poisons the context; recreate it.
                ctx.pop()
                ctx = cuda.Device(0).make_context()
                continue
            outputs[name] = out
            print(f"{name}: PASS launch_ok sum={int(out.sum())} cubin={cubin}")

        if "warpx4_base" in outputs:
            base = outputs["warpx4_base"]
            for name in ("warpx2_02_13", "warpx2_01_23", "cp_4x256b"):
                if name in outputs:
                    diff = int(np.count_nonzero(outputs[name] != base))
                    print(f"{name}: diff_vs_warpx4_base={diff}")
    finally:
        ctx.pop()


if __name__ == "__main__":
    main()
