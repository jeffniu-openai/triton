#!/usr/bin/env python3
"""Catalog executable TMEM opcode families on Blackwell (GPU-local probe)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
import sys
import contextlib
import io

import numpy as np
import pycuda.driver as cuda
import torch
import triton
import triton.language as tl


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
PTXAS_BLACKWELL = REPO_ROOT / "third_party" / "nvidia" / "backend" / "bin" / "ptxas-blackwell"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MATRIX = _load_module(REPO_ROOT / "python" / "test" / "gluon" / "test_tmem_runtime_matrix.py", "tmem_runtime_matrix")
CORE = _load_module(REPO_ROOT / "python" / "test" / "gluon" / "test_core.py", "tmem_core")


def classify_exception(text: str) -> str:
    low = text.lower()
    if "not supported on .target" in low:
        return "CLEAN_UNSUPPORTED"
    if any(tok in low for tok in ("assert", "segmentation fault", "stack dump", "aborted", "illegal memory access")):
        return "BUG"
    if "passmanager::run failed" in text:
        return "BUG"
    return "CLEAN_UNSUPPORTED"


def run_silenced(fn):
    sink_out = io.StringIO()
    sink_err = io.StringIO()
    with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
        return fn()


def run_ldst_cases():
    cases = [
        ("32x32b", 128, 128, MATRIX._make_tmem_linear_layout(128, 128), "32x32b"),
        ("16x64b", 128, 128, MATRIX._make_tmem_linear_layout(128, 128), "16x64b"),
        ("16x128b", 128, 128, MATRIX._make_tmem_linear_layout(128, 128), "16x128b"),
        ("16x256b", 128, 256, MATRIX._make_tmem_linear_layout(128, 256), "16x256b"),
        ("16x32bx2_via_splitn", 64, 64, MATRIX._make_tmem_linear_layout_m64(64), "32x32b_splitn"),
        ("16x32bx2_explicit", 64, 64, MATRIX._make_tmem_linear_layout_m64(64), "16x32bx2"),
    ]
    results = []
    for name, m, n, layout, variant in cases:
        inp = torch.arange(m * n, dtype=torch.float32, device="cuda").reshape(m, n)
        out = torch.empty_like(inp)
        rec = {"case": name, "variant": variant, "shape": [m, n]}
        try:
            compiled = MATRIX.tmem_ldst_variant_kernel[(1,)](inp, out, layout, m, n, variant, num_warps=4)
            torch.testing.assert_close(out, inp, atol=0, rtol=0)
            ops, _ = MATRIX._assert_ldst_ptx_llir_match(compiled)
            rec["status"] = "PASS"
            rec["opcodes"] = sorted({op for op, _ in ops})
        except Exception as exc:
            text = str(exc)
            rec["status"] = classify_exception(text)
            rec["error"] = text.splitlines()[0] if text else type(exc).__name__
        results.append(rec)
    return results


def run_ld_red_cases():
    cases = [
        ("min", False, tl.PropagateNan.NONE, 128, 64),
        ("min", True, tl.PropagateNan.ALL, 128, 128),
        ("max", False, tl.PropagateNan.NONE, 128, 256),
        ("max", True, tl.PropagateNan.ALL, 128, 128),
    ]
    results = []
    for red_op, use_abs, prop_nan, m, n in cases:
        layout = MATRIX._make_tmem_linear_layout(m, n)
        rec = {
            "red_op": red_op,
            "use_abs": bool(use_abs),
            "propagate_nan": "ALL" if prop_nan == tl.PropagateNan.ALL else "NONE",
            "shape": [m, n],
        }
        try:
            compiled = CORE._run_tmem_reduction_case(layout, m, n, red_op, use_abs, prop_nan, num_warps=4)
            ptx_red = [op for op, _ in MATRIX._extract_tcgen05_opcode_offsets(compiled.asm["ptx"], opcodes=("ld",))
                       if ".ld.red." in op]
            llir_red = [op for op, _ in MATRIX._extract_tcgen05_opcode_offsets(compiled.asm["llir"], opcodes=("ld",))
                        if ".ld.red." in op]
            assert ptx_red == llir_red and ptx_red
            rec["status"] = "PASS"
            rec["opcodes"] = sorted(set(ptx_red))
        except Exception as exc:
            text = str(exc)
            rec["status"] = classify_exception(text)
            rec["error"] = text.splitlines()[0] if text else type(exc).__name__
        results.append(rec)

    # One mixed-layout negative to classify behavior.
    rec = {"case": "mixed_layout_negative", "red_op": "min", "use_abs": False, "propagate_nan": "NONE", "shape": [128, 128]}
    try:
        mixed = MATRIX._make_tmem_linear_layout_mixed(128, 128)
        CORE._run_tmem_reduction_case(mixed, 128, 128, "min", False, tl.PropagateNan.NONE, num_warps=4)
        rec["status"] = "PASS"
    except Exception as exc:
        text = str(exc)
        rec["status"] = classify_exception(text)
        rec["error"] = text.splitlines()[0] if text else type(exc).__name__
    results.append(rec)
    return results


def run_cp_gluon_cases():
    results = []

    # 128x256
    inp = torch.arange(128 * 256, device="cuda", dtype=torch.int32).reshape(128, 256)
    out = torch.empty_like(inp)
    rec = {"case": "cp_128x256b"}
    try:
        compiled = MATRIX.tmem_copy_no_scales_kernel[(1,)](inp, out, 128, 256, 256, 32, num_warps=4)
        torch.testing.assert_close(out, inp, atol=0, rtol=0)
        ops = MATRIX._extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
        rec["status"] = "PASS"
        rec["opcodes"] = sorted(set(ops))
    except Exception as exc:
        text = str(exc)
        rec["status"] = classify_exception(text)
        rec["error"] = text.splitlines()[0] if text else type(exc).__name__
    results.append(rec)

    # 128x128
    inp = torch.arange(128 * 4, device="cuda", dtype=torch.int32).reshape(128, 4)
    out = torch.empty_like(inp)
    rec = {"case": "cp_128x128b"}
    try:
        compiled = MATRIX.tmem_copy_128x128_kernel[(1,)](inp, out, 128, num_warps=4)
        torch.testing.assert_close(out, inp, atol=0, rtol=0)
        ops = MATRIX._extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
        rec["status"] = "PASS"
        rec["opcodes"] = sorted(set(ops))
    except Exception as exc:
        text = str(exc)
        rec["status"] = classify_exception(text)
        rec["error"] = text.splitlines()[0] if text else type(exc).__name__
    results.append(rec)

    # scales warpx4
    inp = torch.randint(-100, 100, (64, 16), dtype=torch.int8, device="cuda")
    out = torch.zeros((128, 32), dtype=torch.int8, device="cuda")
    rec = {"case": "cp_warpx4_32x128b_scales"}
    try:
        compiled = MATRIX.tmem_copy_scales_warpx4_kernel[(1,)](inp, out)
        ops = MATRIX._extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
        rec["status"] = "PASS"
        rec["opcodes"] = sorted(set(ops))
    except Exception as exc:
        text = str(exc)
        rec["status"] = classify_exception(text)
        rec["error"] = text.splitlines()[0] if text else type(exc).__name__
    results.append(rec)

    # scales warpx2 candidate negative
    inp = torch.randint(-100, 100, (64, 16), dtype=torch.int8, device="cuda")
    out = torch.zeros((128, 32), dtype=torch.int8, device="cuda")
    rec = {"case": "cp_warpx2_scales_candidate"}
    try:
        layout = MATRIX._make_scales_shared_layout_warpx2_candidate()
        MATRIX.tmem_copy_scales_layout_probe_kernel[(1,)](inp, out, layout)
        rec["status"] = "PASS"
    except Exception as exc:
        text = str(exc)
        rec["status"] = classify_exception(text)
        rec["error"] = text.splitlines()[0] if text else type(exc).__name__
    results.append(rec)
    return results


def _assemble_variant(baseline_ptx: str, opcode: str) -> Path:
    work = Path(tempfile.mkdtemp(prefix="cp_direct_"))
    ptx_path = work / "kernel.ptx"
    cubin_path = work / "kernel.cubin"
    ptx_path.write_text(baseline_ptx.replace("tcgen05.cp.cta_group::1.warpx4.32x128b", opcode))
    subprocess.check_call([str(PTXAS_BLACKWELL), str(ptx_path), "-arch=sm_103a", "-o", str(cubin_path)])
    return cubin_path


def _launch_copy_kernel(cubin: Path, inp: np.ndarray, shared_bytes: int = 2048) -> np.ndarray:
    mod = cuda.module_from_file(str(cubin))
    fn = mod.get_function("tmem_copy_scales_warpx4_kernel")
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


def run_cp_direct_ptx_cases():
    inp_t = torch.randint(-100, 100, (64, 16), dtype=torch.int8, device="cuda")
    out_t = torch.zeros((128, 32), dtype=torch.int8, device="cuda")
    baseline_ptx = MATRIX.tmem_copy_scales_warpx4_kernel[(1,)](inp_t, out_t).asm["ptx"]
    variants = {
        "warpx2_02_13_64x128b": "tcgen05.cp.cta_group::1.warpx2::02_13.64x128b",
        "warpx2_01_23_64x128b": "tcgen05.cp.cta_group::1.warpx2::01_23.64x128b",
        "cp_4x256b": "tcgen05.cp.cta_group::1.4x256b",
    }
    rng = np.random.default_rng(0)
    inp = rng.integers(-100, 100, size=(64, 16), dtype=np.int8)
    results = []

    cuda.init()
    ctx = cuda.Device(0).make_context()
    try:
        for name, opcode in variants.items():
            rec = {"case": name, "opcode": opcode}
            try:
                cubin = _assemble_variant(baseline_ptx, opcode)
                out = _launch_copy_kernel(cubin, inp)
                rec["status"] = "PASS"
                rec["sum"] = int(out.sum())
                rec["cubin"] = str(cubin)
            except Exception as exc:
                text = str(exc)
                rec["status"] = classify_exception(text)
                rec["error"] = text.splitlines()[0] if text else type(exc).__name__
                try:
                    ctx.pop()
                except Exception:
                    pass
                ctx = cuda.Device(0).make_context()
            results.append(rec)
    finally:
        ctx.pop()
    return results


def run_mma_cases():
    results = []
    block_layout_a = MATRIX.ttgl.BlockedLayout([1, 8], [1, CORE.THREADS_PER_WARP], warps_per_cta=[4, 1], order=[0, 1])
    block_layout_b = MATRIX.ttgl.BlockedLayout([1, 8], [1, CORE.THREADS_PER_WARP], warps_per_cta=[4, 1], order=[1, 0])

    # Supported plain MMA kinds.
    for kind in ("tf32", "f8e5m2", "f8e4m3"):
        rec = {"case": f"mma_plain_{kind}"}
        try:
            m = n = 128
            k = 32
            if kind == "tf32":
                a = CORE._round_to_tf32(torch.randn((m, k), device="cuda", dtype=torch.float32))
                b = CORE._round_to_tf32(torch.randn((k, n), device="cuda", dtype=torch.float32))
                out = torch.empty((m, n), device="cuda", dtype=torch.float32)
                shared_layout_a = MATRIX.ttgl.NVMMASharedLayout(swizzle_byte_width=128, transposed=False, element_bitwidth=32, rank=2)
                shared_layout_b = MATRIX.ttgl.NVMMASharedLayout(swizzle_byte_width=128, transposed=True, element_bitwidth=32, rank=2)
                gl_acc_dtype = MATRIX.ttgl.float32
            else:
                fp8 = torch.float8_e5m2 if kind == "f8e5m2" else torch.float8_e4m3fn
                a = torch.randint(20, 40, (m, k), device="cuda", dtype=torch.uint8).view(fp8)
                b = torch.randint(20, 40, (k, n), device="cuda", dtype=torch.uint8).view(fp8)
                out = torch.empty((m, n), device="cuda", dtype=torch.float32)
                shared_layout_a = MATRIX.ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=False, element_bitwidth=8, rank=2)
                shared_layout_b = MATRIX.ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=True, element_bitwidth=8, rank=2)
                gl_acc_dtype = MATRIX.ttgl.float32

            acc_layout = MATRIX.TensorMemoryLayout((m, n), col_stride=1)
            compiled = CORE.mma_kernel[(1,)](
                a, b, out, m, n, k,
                block_layout_a, block_layout_b, (),
                acc_layout, shared_layout_a, shared_layout_b,
                gl_acc_dtype, False, True, num_warps=4
            )
            ops = CORE._extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
            rec["status"] = "PASS"
            rec["opcodes"] = sorted(set(ops))
        except Exception as exc:
            text = str(exc)
            rec["status"] = classify_exception(text)
            rec["error"] = text.splitlines()[0] if text else type(exc).__name__
        results.append(rec)

    # i8 expected to fail at PTXAS on this target.
    rec = {"case": "mma_plain_i8"}
    try:
        m = n = 128
        k = 32
        a = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-8, 8, (k, n), device="cuda", dtype=torch.int8)
        out = torch.empty((m, n), device="cuda", dtype=torch.int32)
        shared_layout_a = MATRIX.ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=False, element_bitwidth=8, rank=2)
        shared_layout_b = MATRIX.ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=True, element_bitwidth=8, rank=2)
        acc_layout = MATRIX.TensorMemoryLayout((m, n), col_stride=1)
        run_silenced(
            lambda: CORE.mma_kernel[(1,)](
                a, b, out, m, n, k,
                block_layout_a, block_layout_b, (),
                acc_layout, shared_layout_a, shared_layout_b,
                MATRIX.ttgl.int32, False, True, num_warps=4
            )
        )
        rec["status"] = "PASS"
    except Exception as exc:
        text = str(exc)
        rec["status"] = classify_exception(text)
        rec["error"] = text.splitlines()[0] if text else type(exc).__name__
    results.append(rec)

    # Scaled MMA: representative known-good combinations.
    for a_format, b_format, num_ctas, acc_kind in [("mxfp8", "mxfp8", 1, "legacy"), ("nvfp4", "nvfp4", 2, "linear")]:
        rec = {
            "case": f"mma_scaled_{a_format}_{b_format}_cta{num_ctas}_{acc_kind}",
            "num_ctas": num_ctas,
            "acc_layout_kind": acc_kind,
        }
        try:
            block_m = 256 if num_ctas == 2 else 128
            block_n = 128
            block_k = 128
            vec_size = 16 if a_format == "nvfp4" else 32
            a, a_scale, _ = CORE.random_quantized_tensor(block_m, block_k, a_format)
            b, b_scale, _ = CORE.random_quantized_tensor(block_n, block_k, b_format)
            a_scale = CORE.swizzle_scales_packed_block(a_scale, vec_size)
            b_scale = CORE.swizzle_scales_packed_block(b_scale, vec_size)
            _, compiled = CORE.mma_scaled_tcgen05_copy(
                a,
                b,
                a_scale,
                b_scale,
                vec_size,
                block_m,
                block_n,
                block_k,
                num_ctas=num_ctas,
                multicast=False,
                acc_layout_kind=acc_kind,
            )
            mma_ops = CORE._extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
            cp_ops = MATRIX._extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
            rec["status"] = "PASS"
            rec["mma_opcodes"] = sorted(set(mma_ops))
            rec["cp_opcodes"] = sorted(set(cp_ops))
        except Exception as exc:
            text = str(exc)
            rec["status"] = classify_exception(text)
            rec["error"] = text.splitlines()[0] if text else type(exc).__name__
        results.append(rec)

    # Unsupported Triton MMA layout family (mixed TMEM linear accumulator).
    rec = {"case": "mma_f16_mixed_linear_acc_layout"}
    try:
        m = n = 128
        k = 32
        a = torch.randn((m, k), dtype=torch.float16, device="cuda")
        b = torch.randn((k, n), dtype=torch.float16, device="cuda")
        c = torch.randn((m, n), dtype=torch.float32, device="cuda")
        out = torch.empty_like(c)
        bad_layout = MATRIX._make_tmem_linear_layout_mixed(m, n)
        MATRIX.tmem_mma_kernel[(1,)](a, b, c, out, bad_layout, False, num_warps=4)
        rec["status"] = "PASS"
    except Exception as exc:
        text = str(exc)
        rec["status"] = classify_exception(text)
        rec["error"] = text.splitlines()[0] if text else type(exc).__name__
    results.append(rec)
    return results


def main():
    torch.manual_seed(0)
    summary = {
        "ldst": run_ldst_cases(),
        "ld_red": run_ld_red_cases(),
        "cp_gluon": run_cp_gluon_cases(),
        "cp_direct_ptx": run_cp_direct_ptx_cases(),
        "mma": run_mma_cases(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
