#!/usr/bin/env python3
"""Run the generated signed-i8 tcgen05.mma PTX with torch-owned buffers."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


def check_cuda(result: int, lib, what: str) -> None:
    if result == 0:
        return
    name = ctypes.c_char_p()
    message = ctypes.c_char_p()
    lib.cuGetErrorName(result, ctypes.byref(name))
    lib.cuGetErrorString(result, ctypes.byref(message))
    decoded_name = name.value.decode() if name.value else f"CUresult({result})"
    decoded_message = message.value.decode() if message.value else ""
    raise RuntimeError(f"{what} failed ({decoded_name}): {decoded_message}")


def launch_with_ctypes(ptx: str, kernel_name: str, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor,
                       shared_bytes: int) -> None:
    lib_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
    lib = ctypes.CDLL(lib_name)

    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    lib.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    lib.cuCtxSynchronize.argtypes = []
    lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
    lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

    check_cuda(lib.cuInit(0), lib, "cuInit")
    context = ctypes.c_void_p()
    check_cuda(lib.cuCtxGetCurrent(ctypes.byref(context)), lib, "cuCtxGetCurrent")
    if not context.value:
        raise RuntimeError("no current CUDA context; torch CUDA allocation should have initialized one")

    module = ctypes.c_void_p()
    check_cuda(lib.cuModuleLoadData(ctypes.byref(module), ptx.encode()), lib, "cuModuleLoadData")
    try:
        function = ctypes.c_void_p()
        check_cuda(
            lib.cuModuleGetFunction(ctypes.byref(function), module, kernel_name.encode()),
            lib,
            "cuModuleGetFunction",
        )

        a_ptr = ctypes.c_void_p(a.data_ptr())
        b_ptr = ctypes.c_void_p(b.data_ptr())
        out_ptr = ctypes.c_void_p(out.data_ptr())
        zero0 = ctypes.c_uint64(0)
        zero1 = ctypes.c_uint64(0)
        params = (ctypes.c_void_p * 5)(
            ctypes.cast(ctypes.byref(a_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(b_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(out_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(zero0), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(zero1), ctypes.c_void_p),
        )
        check_cuda(
            lib.cuLaunchKernel(function, 1, 1, 1, 128, 1, 1, shared_bytes, None, params, None),
            lib,
            "cuLaunchKernel",
        )
        check_cuda(lib.cuCtxSynchronize(), lib, "cuCtxSynchronize")
    finally:
        check_cuda(lib.cuModuleUnload(module), lib, "cuModuleUnload")


def launch_with_cpp(ptx: str, kernel_name: str, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor,
                    shared_bytes: int, verbose: bool) -> None:
    from torch.utils.cpp_extension import load

    module = load(
        name="triton_i8_ptx_driver",
        sources=[str(HERE / "ptx_driver.cpp")],
        extra_cflags=["-O2"],
        extra_ldflags=["-lcuda"],
        with_cuda=False,
        verbose=verbose,
    )
    module.launch_ptx(ptx, kernel_name, a, b, out, shared_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=HERE / "tcgen05_i8_signed_sm100.metadata.json")
    parser.add_argument("--ptx", type=Path, default=None)
    parser.add_argument("--launcher", choices=["cpp", "ctypes"], default="cpp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    ptx_path = args.ptx or args.metadata.with_name(metadata["ptx_file"])
    ptx = ptx_path.read_text()
    actual_hash = hashlib.sha256(ptx.encode()).hexdigest()
    if actual_hash != metadata["ptx_sha256"]:
        raise RuntimeError(f"PTX hash mismatch for {ptx_path}: {actual_hash} != {metadata['ptx_sha256']}")

    if args.dry_run:
        print(f"PTX hash ok: {actual_hash}")
        print(f"kernel: {metadata['kernel_name']}")
        print(f"target: {metadata['target']}")
        print(f"launch: {metadata['launch']}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (10, 0):
        raise RuntimeError(f"expected GB200/sm100 CUDA device, got capability {(major, minor)}")

    shape = metadata["shape"]
    torch.manual_seed(args.seed)
    a = torch.randint(-8, 8, (shape["M"], shape["K"]), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (shape["K"], shape["N"]), device="cuda", dtype=torch.int8)
    out = torch.empty((shape["M"], shape["N"]), device="cuda", dtype=torch.int32)

    shared_bytes = metadata["launch"]["dynamic_shared_memory_bytes"]
    if args.launcher == "cpp":
        launch_with_cpp(ptx, metadata["kernel_name"], a, b, out, shared_bytes, args.verbose_build)
    else:
        launch_with_ctypes(ptx, metadata["kernel_name"], a, b, out, shared_bytes)

    ref = a.cpu().to(torch.int32) @ b.cpu().to(torch.int32)
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
    print("PASS signed i8 tcgen05.mma sm100 PTX matches torch int32 matmul")


if __name__ == "__main__":
    main()
