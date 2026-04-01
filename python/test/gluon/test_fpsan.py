# ruff: noqa: F821
import itertools
import math
import numpy as np
import pytest
import re
import torch

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.errors import CompilationError
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton import language as tl
from triton._internal_testing import is_blackwell, is_cuda, is_hip, is_hip_cdna3, is_hip_cdna4, is_hip_gfx1250, is_interpreter
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryLinearLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    tcgen05_mma_barrier_count,
    tcgen05_mma_scaled,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier as hopper_mbarrier, tma
from triton._filecheck import run_parser

THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size
BLACKWELL_PARSER_TARGET = GPUTarget("cuda", 100, 32)


def _make_tmem_linear_layout(m, n):
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(m.bit_length() - 1)],
        cols=[[0, 1 << i] for i in range(n.bit_length() - 1)],
        shape=[m, n],
    )


def _make_tmem_linear_layout_mixed_128x128():
    return TensorMemoryLinearLayout(
        rows=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1], [0, 2]],
        cols=[[32, 0], [64, 0], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]],
        shape=[128, 128],
    )


def _make_tmem_linear_layout_block(m, n, two_ctas=False):
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(1, int(math.log2(m)))],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))],
        block_bases=[[1, 0]],
        shape=[m, n],
        two_ctas=two_ctas,
    )


def _make_tmem_linear_layout_64x32_block(two_ctas=False):
    return TensorMemoryLinearLayout(
        rows=[[2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
        cols=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
        block_bases=[[1, 0]],
        shape=[64, 32],
        two_ctas=two_ctas,
    )


def _lift_tmem_layout(base_layout, prefix_shape):
    prefix_shape = list(prefix_shape)
    prefix_rank = len(prefix_shape)
    total_rank = prefix_rank + len(base_layout.shape)

    def extend_bases(bases):
        return [[0] * prefix_rank + list(basis) for basis in bases]

    rows = extend_bases(base_layout.rows)
    cols = extend_bases(base_layout.cols)
    block_bases = extend_bases(base_layout.block_bases)

    for dim in range(prefix_rank - 1, -1, -1):
        for bit in range(int(math.log2(prefix_shape[dim]))):
            basis = [0] * total_rank
            basis[dim] = 1 << bit
            cols.append(basis)

    return TensorMemoryLinearLayout(
        rows=rows,
        cols=cols,
        block_bases=block_bases,
        shape=prefix_shape + list(base_layout.shape),
        two_ctas=base_layout.two_ctas,
    )


def _make_tmem_linear_layout_mmav5_twocta(m, n):
    assert m >= 128 and (m & (m - 1)) == 0
    assert n >= 1 and (n & (n - 1)) == 0
    tile_m = m // 2
    assert tile_m in (64, 128)
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(int(math.log2(tile_m)))],
        cols=[[0, 1 << i] for i in range(int(math.log2(n)))],
        block_bases=[[tile_m, 0]],
        shape=[m, n],
        two_ctas=True,
    )


def _default_cga_layout(num_ctas, rank, dim=0):
    if num_ctas == 1:
        return []
    return [[0] * dim + [1 << i] + [0] * (rank - dim - 1) for i in range(num_ctas.bit_length() - 1)]


def _make_tmem_register_layout(num_ctas):
    return gl.BlockedLayout(
        [1, 32],
        [32, 1],
        [2, 1],
        [0, 1],
        cga_layout=_default_cga_layout(num_ctas, 2),
    )


def _make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, two_cta_dim):
    ctas_per_cga = list(ctas_per_cga)
    cta_split = list(cta_split)
    assert cta_split[two_cta_dim] > 1
    cta_split[two_cta_dim] //= 2
    ctas_per_cga[two_cta_dim] //= 2
    aux_cga_layout = make_cga_layout(ctas_per_cga, cta_split, cta_order)
    basis = [0, 0]
    basis[two_cta_dim] = 1
    for b in aux_cga_layout:
        b[two_cta_dim] *= 2
    return [basis] + aux_cga_layout


def _extract_tcgen05_mma_opcodes(asm: str):
    pattern = re.compile(r"(tcgen05\.mma[^\s;\"]*)")
    return pattern.findall(asm)


def _assert_mma_codegen_opcodes(compiled):
    llir_ops = _extract_tcgen05_mma_opcodes(compiled.asm["llir"])
    ptx_ops = _extract_tcgen05_mma_opcodes(compiled.asm["ptx"])
    if llir_ops and ptx_ops:
        assert ptx_ops == llir_ops
    return ptx_ops if ptx_ops else llir_ops


def _assert_tmem_allocator_lifetime(compiled, cta_group: int):
    alloc_opcode = f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"
    relinquish_opcode = f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"

    llir = compiled.asm["llir"]
    ptx = compiled.asm["ptx"]
    has_llir = alloc_opcode in llir or relinquish_opcode in llir
    has_ptx = alloc_opcode in ptx or relinquish_opcode in ptx
    if not has_llir and not has_ptx:
        return
    if has_llir:
        assert alloc_opcode in llir, f"Missing {alloc_opcode} in llir"
        assert relinquish_opcode in llir, f"Missing {relinquish_opcode} in llir"
        assert llir.count(alloc_opcode) == llir.count(relinquish_opcode)

    if has_ptx:
        assert alloc_opcode in ptx, f"Missing {alloc_opcode} in ptx"
        assert relinquish_opcode in ptx, f"Missing {relinquish_opcode} in ptx"
        assert ptx.count(alloc_opcode) == ptx.count(relinquish_opcode)

@gluon.jit
def _tcgen05_mma_twocta_linear_kernel(a_desc, b_desc, out_ptrs, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                      acc_tmem_layout: gl.constexpr, acc_tmem_base_layout: gl.constexpr,
                                      blocked_c: gl.constexpr, ACC_VIA_VIEW: gl.constexpr):
    smem_a = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    smem_b = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)

    tma_bar = hopper_mbarrier.allocate_mbarrier(two_ctas=acc_tmem_layout.two_ctas)
    hopper_mbarrier.init(tma_bar, count=1)
    mma_bar = hopper_mbarrier.allocate_mbarrier()
    hopper_mbarrier.init(mma_bar, count=tcgen05_mma_barrier_count([smem_a, smem_b], True))

    hopper_mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
    tma.async_copy_global_to_shared(a_desc, [0, 0], tma_bar, smem_a, multicast=True)
    tma.async_copy_global_to_shared(b_desc, [0, 0], tma_bar, smem_b, multicast=True)
    hopper_mbarrier.wait(tma_bar, phase=0, deps=[smem_a, smem_b])
    hopper_mbarrier.invalidate(tma_bar)

    if ACC_VIA_VIEW:
        acc_base = allocate_tensor_memory(gl.float32, [2, BLOCK_M, BLOCK_N], acc_tmem_base_layout)
        acc_tmem = acc_base.index(1)
    else:
        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], acc_tmem_layout)
    tcgen05_mma(smem_a, smem_b, acc_tmem, use_acc=False, multicast=True, mbarriers=[mma_bar])
    hopper_mbarrier.wait(mma_bar, phase=0, deps=[smem_a, smem_b])
    hopper_mbarrier.invalidate(mma_bar)

    out = acc_tmem.load()
    out = gl.convert_layout(out, blocked_c)
    out_offs_m = gl.arange(0, BLOCK_M)[:, None]
    out_offs_n = gl.arange(0, BLOCK_N)[None, :]
    gl.store(out_ptrs + out_offs_m * BLOCK_N + out_offs_n, out)

def _hip_device_supports_fpsan():
    return is_hip_cdna3() or is_hip_cdna4() or is_hip_gfx1250()


def _require_cuda_backend(device: str):
    # CUDA and HIP both use torch device 'cuda'. fpsan is plumbed through both CUDAOptions and HIPOptions.
    if device != "cuda":
        pytest.skip("fpsan tests require torch device 'cuda'")
    if is_interpreter():
        pytest.skip("fpsan tests require a real backend (not the interpreter)")
    if not (is_cuda() or is_hip()):
        pytest.skip("fpsan tests require CUDA or HIP")
    if is_hip() and not _hip_device_supports_fpsan():
        pytest.skip("fpsan is not supported on this HIP device")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _as_u32(x_i32: np.ndarray) -> np.ndarray:
    assert x_i32.dtype == np.int32
    return x_i32.view(np.uint32)


def _u32_to_i32(x_u32: np.ndarray) -> np.ndarray:
    assert x_u32.dtype == np.uint32
    return x_u32.view(np.int32)


def _expected_add_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    x_u32 = _as_u32(x_i32).astype(np.uint64)
    y_u32 = _as_u32(y_i32).astype(np.uint64)
    out_u32 = ((x_u32 + y_u32) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return _u32_to_i32(out_u32)


def _expected_sub_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    x_u32 = _as_u32(x_i32).astype(np.uint64)
    y_u32 = _as_u32(y_i32).astype(np.uint64)
    out_u32 = ((x_u32 - y_u32) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return _u32_to_i32(out_u32)


def _expected_mul_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    x_u32 = _as_u32(x_i32).astype(np.uint64)
    y_u32 = _as_u32(y_i32).astype(np.uint64)
    out_u32 = ((x_u32 * y_u32) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return _u32_to_i32(out_u32)


def _expected_srem_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    # Match LLVM srem semantics: remainder after trunc-toward-zero division.
    # NOTE: Python/NumPy '%' uses floor division for negatives, so we implement explicitly.
    #
    # In fpsan mode we force denominator non-zero using `den | 1` in the *payload* domain.
    x = x_i32.astype(np.int64)
    y_safe_u32 = (_as_u32(y_i32) | np.uint32(1)).astype(np.uint32)
    y = _u32_to_i32(y_safe_u32).astype(np.int64)
    q = (np.sign(x) * np.sign(y) * (np.abs(x) // np.abs(y))).astype(np.int64)
    r = (x - q * y).astype(np.int32)
    return r


def murmur64Mixer(h: np.uint64) -> np.uint64:
    with np.errstate(over="ignore"):
        h = np.uint64(h)
        h ^= h >> np.uint64(33)
        h = np.uint64(h * np.uint64(0xff51afd7ed558ccd))
        h ^= h >> np.uint64(33)
        h = np.uint64(h * np.uint64(0xc4ceb9fe1a85ec53))
        h ^= h >> np.uint64(33)
        return np.uint64(h)


OP_TO_ID_U64 = {
    "exp": np.uint64(0),
    "log": np.uint64(1),
    "exp2": np.uint64(2),
    "log2": np.uint64(3),
    "cos": np.uint64(4),
    "sin": np.uint64(5),
    "sqrt": np.uint64(6),
    "rsqrt": np.uint64(7),
    "erf": np.uint64(8),
    "floor": np.uint64(9),
    "ceil": np.uint64(10),
    "sqrt_rn": np.uint64(11),
    "div_inv": np.uint64(12),
}

OP_TO_TAG_U32 = {name: np.uint32(murmur64Mixer(op_id) & np.uint64(0xFFFFFFFF)) for name, op_id in OP_TO_ID_U64.items()}


def _expected_div_payload_i32(x_i32: np.ndarray, y_i32: np.ndarray) -> np.ndarray:
    # fpsan division is defined as: num_bits * (den_bits xor DivInvOpId) mod 2^32.
    # Keep this in sync with UnaryOpId::DivInv in FpSanitizer.cpp.
    div_inv_tag = OP_TO_TAG_U32["div_inv"].astype(np.uint64)
    mask = np.uint64(0xFFFFFFFF)
    num = _as_u32(x_i32).astype(np.uint64)
    den = _as_u32(y_i32).astype(np.uint64)
    tagged = (den ^ div_inv_tag).astype(np.uint64)
    out_u32 = ((num * tagged) & mask).astype(np.uint32)
    return _u32_to_i32(out_u32)


def _expected_unary_tag_i32(x_i32: np.ndarray, op: str) -> np.ndarray:
    # Keep this mapping in sync with UnaryOpId in FpSanitizer.cpp.
    tag = OP_TO_TAG_U32[op]
    out_u32 = _as_u32(x_i32) ^ tag
    return _u32_to_i32(out_u32)


def stable_string_hash_u64(s: str) -> np.uint64:
    with np.errstate(over="ignore"):
        h = np.uint64(14695981039346656037)
        for c in s.encode("utf-8"):
            h ^= np.uint64(c)
            h = np.uint64(h * np.uint64(1099511628211))
        return h


def _expected_extern_unary_tag_i32(x_i32: np.ndarray, symbol: str) -> np.ndarray:
    return _expected_extern_variadic_tag_i32([x_i32], symbol)


def _expected_extern_binary_tag_i32(x_i32: np.ndarray, y_i32: np.ndarray, symbol: str) -> np.ndarray:
    return _expected_extern_variadic_tag_i32([x_i32, y_i32], symbol)


def _rotl_u32(x_u32: np.ndarray, amount: int) -> np.ndarray:
    amount = amount & 31
    if amount == 0:
        return x_u32
    x = x_u32.astype(np.uint64)
    out_u32 = ((x << np.uint64(amount)) | (x >> np.uint64(32 - amount))) & np.uint64(0xFFFFFFFF)
    return out_u32.astype(np.uint32)


def _expected_extern_variadic_tag_i32(args_i32: list[np.ndarray], symbol: str) -> np.ndarray:
    tag = np.uint64(stable_string_hash_u64(symbol) & np.uint64(0xFFFFFFFF))
    total_u64 = np.zeros_like(_as_u32(args_i32[0]), dtype=np.uint64)
    for i, arg in enumerate(args_i32):
        rotated = _rotl_u32(_as_u32(arg), i).astype(np.uint64)
        total_u64 = (total_u64 + rotated) & np.uint64(0xFFFFFFFF)
    out_u32 = (total_u64 ^ tag).astype(np.uint32)
    return _u32_to_i32(out_u32)


UNARY_EXTERN_SYMBOLS = {
    "cuda": [
        ("tan", "__nv_tanf"),
        ("tanh", "__nv_tanhf"),
        ("log1p", "__nv_log1pf"),
        ("cbrt", "__nv_cbrtf"),
        ("round", "__nv_roundf"),
    ],
    "hip": [
        ("tan", "__ocml_tan_f32"),
        ("tanh", "__ocml_tanh_f32"),
        ("log1p", "__ocml_log1p_f32"),
        ("round", "__ocml_round_f32"),
    ],
}

BINARY_EXTERN_SYMBOLS = {
    "cuda": [
        ("atan2", "__nv_atan2f"),
        ("hypot", "__nv_hypotf"),
        ("pow", "__nv_powf"),
    ],
    "hip": [
        ("atan2", "__ocml_atan2_f32"),
        ("hypot", "__ocml_hypot_f32"),
        ("pow", "__ocml_pow_f32"),
    ],
}

TERNARY_EXTERN_SYMBOLS = {
    "cuda": [
        ("fma", "__nv_fmaf"),
    ],
    "hip": [
        ("fma", "__ocml_fma_f32"),
    ],
}

MIXED_EXTERN_SYMBOLS = {
    "cuda": [
        ("ldexp", "__nv_ldexpf"),
    ],
    "hip": [
        ("ldexp", "__ocml_ldexp_f32"),
    ],
}


def _extern_backend_name() -> str:
    if is_hip():
        return "hip"
    return "cuda"


EXTERN_UNARY_CASES = UNARY_EXTERN_SYMBOLS[_extern_backend_name()]
EXTERN_BINARY_CASES = BINARY_EXTERN_SYMBOLS[_extern_backend_name()]
EXTERN_TERNARY_CASES = TERNARY_EXTERN_SYMBOLS[_extern_backend_name()]
EXTERN_MIXED_CASES = MIXED_EXTERN_SYMBOLS[_extern_backend_name()]


def _as_payload_np_i32(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if not isinstance(x, np.ndarray):
        raise TypeError(f"unsupported input type: {type(x)}")
    if x.dtype == np.int32:
        return x.astype(np.int32, copy=False)
    if x.dtype == np.uint32:
        return x.view(np.int32)
    if x.dtype == np.float32:
        return x.view(np.int32)
    raise TypeError(f"unsupported dtype for payload comparison: {x.dtype}")


def _assert_payload_equal(actual, expected) -> None:
    np.testing.assert_array_equal(_as_payload_np_i32(actual), _as_payload_np_i32(expected))


def _payload_equal(a, b) -> bool:
    return np.array_equal(_as_payload_np_i32(a), _as_payload_np_i32(b))


@gluon.jit
def _binop_kernel(x_ptr, y_ptr, out_ptr, n_elements, OP: gl.constexpr, BLOCK: gl.constexpr,
                  THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = gl.load(y_ptr + offs, mask=mask, other=0.0)

    if OP == "add":
        z = x + y
    elif OP == "sub":
        z = x - y
    elif OP == "mul":
        z = x * y
    elif OP == "truediv":
        z = x / y
    elif OP == "fdiv":
        z = gl.fdiv(x, y)
    elif OP == "mod":
        z = x % y
    else:
        gl.static_assert(False, "unsupported OP")

    gl.store(out_ptr + offs, z, mask=mask)


@pytest.mark.parametrize(
    "op,expected_fn",
    [
        ("add", _expected_add_i32),
        ("sub", _expected_sub_i32),
        ("mul", _expected_mul_i32),
        ("truediv", _expected_div_payload_i32),
        ("fdiv", _expected_div_payload_i32),
        ("mod", _expected_srem_i32),
    ],
)
def test_binops_payload_semantics(device, op, expected_fn, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    # Use int32 storage but treat it as float32 via TensorWrapper so fpsan operates on payload bits.
    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    yw = triton.TensorWrapper(y, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _binop_kernel[grid](xw, yw, outw, n_elements, OP=op, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = expected_fn(x.cpu().numpy().astype(np.int32, copy=False), y.cpu().numpy().astype(np.int32, copy=False))
    _assert_payload_equal(out_np, exp_np)


@pytest.mark.parametrize(
    "op,expected_fn",
    [
        ("truediv", _expected_div_payload_i32),
        ("fdiv", _expected_div_payload_i32),
        ("mod", _expected_srem_i32),
    ],
)
def test_binops_payload_semantics_zero_denominator(device, op, expected_fn, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(123)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y[::7] = 0

    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    yw = triton.TensorWrapper(y, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _binop_kernel[grid](xw, yw, outw, n_elements, OP=op, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = expected_fn(x.cpu().numpy().astype(np.int32, copy=False), y.cpu().numpy().astype(np.int32, copy=False))
    _assert_payload_equal(out_np, exp_np)


@gluon.jit
def _unary_math_kernel(x_ptr, out_ptr, n_elements, OP: gl.constexpr, BLOCK: gl.constexpr,
                       THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    z = getattr(gl, OP)(x)
    gl.store(out_ptr + offs, z, mask=mask)


@pytest.mark.parametrize(
    "op",
    [
        "exp",
        "exp2",
        "log",
        "log2",
        "cos",
        "sin",
        "sqrt",
        "sqrt_rn",
        "rsqrt",
        "erf",
        "floor",
        "ceil",
    ],
)
def test_unary_math_identity(device, op, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256
    rs = np.random.RandomState(0)
    # Includes negative values for log/sqrt on purpose; fpsan works on payload bits.
    xf = rs.randn(n_elements).astype(np.float32)
    x_bits = xf.view(np.int32)

    x = torch.tensor(x_bits, dtype=torch.int32, device="cuda")
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(n_elements, BLOCK), )
    _unary_math_kernel[grid](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(out, dtype=torch.float32),
        n_elements,
        OP=op,
        BLOCK=BLOCK,
        THREADS_PER_WARP=THREADS_PER_WARP,
    )

    exp_bits = _expected_unary_tag_i32(x_bits, op)
    _assert_payload_equal(out, exp_bits)


@gluon.jit
def _extern_unary_math_kernel(x_ptr, out_ptr, n_elements, OP: gl.constexpr, BLOCK: gl.constexpr,
                              THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    if OP == "tan":
        z = gl.extra.libdevice.tan(x)
    elif OP == "tanh":
        z = gl.extra.libdevice.tanh(x)
    elif OP == "log1p":
        z = gl.extra.libdevice.log1p(x)
    elif OP == "cbrt":
        z = gl.extra.libdevice.cbrt(x)
    elif OP == "round":
        z = gl.extra.libdevice.round(x)
    else:
        gl.static_assert(False, "unsupported OP")
    gl.store(out_ptr + offs, z, mask=mask)


@pytest.mark.parametrize(
    "op,symbol",
    EXTERN_UNARY_CASES,
)
def test_extern_unary_payload_semantics(device, op, symbol, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256
    rs = np.random.RandomState(11)
    xf = rs.randn(n_elements).astype(np.float32)
    x_bits = xf.view(np.int32)

    x = torch.tensor(x_bits, dtype=torch.int32, device="cuda")
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(n_elements, BLOCK), )
    _extern_unary_math_kernel[grid](
        triton.TensorWrapper(x, dtype=torch.float32),
        triton.TensorWrapper(out, dtype=torch.float32),
        n_elements,
        OP=op,
        BLOCK=BLOCK,
        THREADS_PER_WARP=THREADS_PER_WARP,
    )

    exp_bits = _expected_extern_unary_tag_i32(x_bits, symbol)
    _assert_payload_equal(out, exp_bits)


@gluon.jit
def _extern_binary_math_kernel(x_ptr, y_ptr, out_ptr, n_elements, OP: gl.constexpr, BLOCK: gl.constexpr,
                               THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = gl.load(y_ptr + offs, mask=mask, other=0.0)
    if OP == "atan2":
        z = gl.extra.libdevice.atan2(x, y)
    elif OP == "hypot":
        z = gl.extra.libdevice.hypot(x, y)
    elif OP == "pow":
        z = gl.extra.libdevice.pow(x, y)
    else:
        gl.static_assert(False, "unsupported OP")
    gl.store(out_ptr + offs, z, mask=mask)


@gluon.jit
def _extern_ternary_math_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n_elements, OP: gl.constexpr, BLOCK: gl.constexpr,
                                THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = gl.load(y_ptr + offs, mask=mask, other=0.0)
    z = gl.load(z_ptr + offs, mask=mask, other=0.0)
    if OP == "fma":
        out = gl.extra.libdevice.fma(x, y, z)
    else:
        gl.static_assert(False, "unsupported OP")
    gl.store(out_ptr + offs, out, mask=mask)


@gluon.jit
def _extern_mixed_math_kernel(x_ptr, y_ptr, out_ptr, n_elements, OP: gl.constexpr, BLOCK: gl.constexpr,
                              THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = gl.load(y_ptr + offs, mask=mask, other=0)
    if OP == "ldexp":
        out = gl.extra.libdevice.ldexp(x, y)
    else:
        gl.static_assert(False, "unsupported OP")
    gl.store(out_ptr + offs, out, mask=mask)


@pytest.mark.parametrize(
    "op,symbol",
    EXTERN_BINARY_CASES,
)
def test_extern_binary_payload_semantics(device, op, symbol, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256
    g = torch.Generator(device="cuda")
    g.manual_seed(23)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    yw = triton.TensorWrapper(y, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _extern_binary_math_kernel[grid](xw, yw, outw, n_elements, OP=op, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    exp_bits = _expected_extern_binary_tag_i32(
        x.cpu().numpy().astype(np.int32, copy=False),
        y.cpu().numpy().astype(np.int32, copy=False),
        symbol,
    )
    _assert_payload_equal(out, exp_bits)


@pytest.mark.parametrize(
    "op,symbol",
    EXTERN_TERNARY_CASES,
)
def test_extern_ternary_payload_semantics(device, op, symbol, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256
    g = torch.Generator(device="cuda")
    g.manual_seed(29)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    z = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    yw = triton.TensorWrapper(y, dtype=torch.float32)
    zw = triton.TensorWrapper(z, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _extern_ternary_math_kernel[grid](xw, yw, zw, outw, n_elements, OP=op, BLOCK=BLOCK,
                                      THREADS_PER_WARP=THREADS_PER_WARP)

    exp_bits = _expected_extern_variadic_tag_i32(
        [
            x.cpu().numpy().astype(np.int32, copy=False),
            y.cpu().numpy().astype(np.int32, copy=False),
            z.cpu().numpy().astype(np.int32, copy=False),
        ],
        symbol,
    )
    _assert_payload_equal(out, exp_bits)


@pytest.mark.parametrize(
    "op,symbol",
    EXTERN_MIXED_CASES,
)
def test_extern_mixed_payload_semantics(device, op, symbol, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256
    g = torch.Generator(device="cuda")
    g.manual_seed(31)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _extern_mixed_math_kernel[grid](xw, y, outw, n_elements, OP=op, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    exp_bits = _expected_extern_variadic_tag_i32(
        [
            x.cpu().numpy().astype(np.int32, copy=False),
            y.cpu().numpy().astype(np.int32, copy=False),
        ],
        symbol,
    )
    _assert_payload_equal(out, exp_bits)


def _expected_fma_i32(x_i32: np.ndarray, y_i32: np.ndarray, z_i32: np.ndarray) -> np.ndarray:
    return _expected_add_i32(_expected_mul_i32(x_i32, y_i32), z_i32)


def _expected_trunc_ext_roundtrip_i32(x_i32: np.ndarray) -> np.ndarray:
    x_u32 = _as_u32(x_i32)
    out_u32 = x_u32 & np.uint32(0x0000FFFF)
    return _u32_to_i32(out_u32)


def _expected_ext_f16_to_f32_i32(x_i16: np.ndarray) -> np.ndarray:
    out_u32 = x_i16.view(np.uint16).astype(np.uint32)
    return out_u32.view(np.int32)


@gluon.jit
def _fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n_elements, BLOCK: gl.constexpr, THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = gl.load(y_ptr + offs, mask=mask, other=0.0)
    z = gl.load(z_ptr + offs, mask=mask, other=0.0)
    out = gl.fma(x, y, z)
    gl.store(out_ptr + offs, out, mask=mask)


def test_fma_payload_semantics(device, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    y = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    z = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    yw = triton.TensorWrapper(y, dtype=torch.float32)
    zw = triton.TensorWrapper(z, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _fma_kernel[grid](xw, yw, zw, outw, n_elements, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = _expected_fma_i32(
        x.cpu().numpy().astype(np.int32, copy=False),
        y.cpu().numpy().astype(np.int32, copy=False),
        z.cpu().numpy().astype(np.int32, copy=False),
    )
    _assert_payload_equal(out_np, exp_np)


@gluon.jit
def _cast_trunc_ext_kernel(x_ptr, out_ptr, n_elements, BLOCK: gl.constexpr, THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x.to(gl.float16)
    z = y.to(gl.float32)
    gl.store(out_ptr + offs, z, mask=mask)


def test_cast_trunc_ext_payload_semantics(device, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(17)
    x = torch.randint(-(2**31), 2**31 - 1, (n_elements, ), dtype=torch.int32, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _cast_trunc_ext_kernel[grid](xw, outw, n_elements, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = _expected_trunc_ext_roundtrip_i32(x.cpu().numpy().astype(np.int32, copy=False))
    _assert_payload_equal(out_np, exp_np)


@gluon.jit
def _cast_ext_kernel(x_ptr, out_ptr, n_elements, BLOCK: gl.constexpr, THREADS_PER_WARP: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[2], threads_per_warp=[THREADS_PER_WARP], warps_per_cta=[4],
                                            order=[0])
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n_elements
    x = gl.load(x_ptr + offs, mask=mask, other=0.0)
    z = x.to(gl.float32)
    gl.store(out_ptr + offs, z, mask=mask)


def test_cast_ext_payload_semantics(device, fresh_knobs):
    _require_cuda_backend(device)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    n_elements = 1024
    BLOCK = 256

    g = torch.Generator(device="cuda")
    g.manual_seed(19)
    x = torch.randint(-(2**15), 2**15 - 1, (n_elements, ), dtype=torch.int16, device="cuda", generator=g)
    out = torch.empty((n_elements, ), dtype=torch.int32, device="cuda")

    xw = triton.TensorWrapper(x, dtype=torch.float16)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    grid = (triton.cdiv(n_elements, BLOCK), )
    _cast_ext_kernel[grid](xw, outw, n_elements, BLOCK=BLOCK, THREADS_PER_WARP=THREADS_PER_WARP)

    out_np = out.cpu().numpy().astype(np.int32, copy=False)
    exp_np = _expected_ext_f16_to_f32_i32(x.cpu().numpy().astype(np.int16, copy=False))
    _assert_payload_equal(out_np, exp_np)


def _mm_payload_u32(a_i32: np.ndarray, b_i32: np.ndarray, c_i32: np.ndarray = None) -> np.ndarray:
    # Computes: c + a @ b in Z/(2^32) on raw payload bits.
    a_u = a_i32.view(np.uint32).astype(np.uint64)
    b_u = b_i32.view(np.uint32).astype(np.uint64)
    c_u = c_i32.view(np.uint32).astype(np.uint64) if c_i32 is not None else None
    m, k = a_u.shape
    k2, n = b_u.shape
    assert k == k2
    out = np.empty((m, n), dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFF)
    for i in range(m):
        for j in range(n):
            s = c_u[i, j] if c_u is not None else 0
            for kk in range(k):
                s = (s + (a_u[i, kk] * b_u[kk, j])) & mask
            out[i, j] = s
    return out.astype(np.uint32).view(np.int32)


def _unpack_element(data: np.ndarray, row: int, col: int, pack: int, pack_axis: int = 1) -> np.uint64:
    if pack_axis == 1:
        raw = np.uint64(data[row, col // pack])
        nibble_idx = col
    else:
        raw = np.uint64(data[row // pack, col])
        nibble_idx = row
    if pack == 2:
        return (raw >> np.uint64(4 * (nibble_idx % pack))) & np.uint64(0x0F)
    return raw


def _scale_element(val: np.uint64, scale, idx: int, k: int, mask: np.uint64) -> np.uint64:
    if scale is not None:
        return (val * np.uint64(scale[idx, k // 32])) & mask
    return val


def _dot_scaled_payload_u32(a_data: np.ndarray, b_data: np.ndarray, a_scale, b_scale, a_pack: int,
                            b_pack: int) -> np.ndarray:
    M, N = a_data.shape[0], b_data.shape[1]
    K = a_data.shape[1] * a_pack
    mask = np.uint64(0xFFFFFFFF)
    out = np.zeros((M, N), dtype=np.uint64)
    for i, j in itertools.product(range(M), range(N)):
        s = np.uint64(0)
        for kk in range(K):
            a_val = _unpack_element(a_data, i, kk, a_pack, pack_axis=1)
            b_val = _unpack_element(b_data, kk, j, b_pack, pack_axis=0)
            a_val = _scale_element(a_val, a_scale, i, kk, mask)
            b_val = _scale_element(b_val, b_scale, j, kk, mask)
            s = (s + a_val * b_val) & mask
        out[i, j] = s
    return out.astype(np.uint32).view(np.int32)


def _mm_scaled_payload_u32(a_u8: np.ndarray, b_u8: np.ndarray, a_scale_u8: np.ndarray, b_scale_u8: np.ndarray,
                           c_i32: np.ndarray = None, a_pack: int = 1, b_pack: int = 1) -> np.ndarray:
    a_scale = a_scale_u8.astype(np.uint16)
    b_scale = b_scale_u8.astype(np.uint16)
    c_u = c_i32.view(np.uint32).astype(np.uint64) if c_i32 is not None else None

    m = a_u8.shape[0]
    n = b_u8.shape[1]
    k = a_u8.shape[1] * a_pack
    assert k == b_u8.shape[0] * b_pack
    assert a_scale.shape == (m, k // 32)
    assert b_scale.shape == (n, k // 32)

    def unpack(data: np.ndarray, row: int, col: int, pack: int, pack_axis: int) -> np.uint16:
        if pack == 1:
            return np.uint16(data[row, col])
        return np.uint16(_unpack_element(data, row, col, pack, pack_axis=pack_axis))

    out = np.empty((m, n), dtype=np.uint64)
    mask16 = np.uint32(0xFFFF)
    mask32 = np.uint64(0xFFFFFFFF)
    for i in range(m):
        for j in range(n):
            s = c_u[i, j] if c_u is not None else 0
            for kk in range(k):
                a_val = unpack(a_u8, i, kk, a_pack, pack_axis=1)
                b_val = unpack(b_u8, kk, j, b_pack, pack_axis=0)
                lhs = np.uint16((np.uint32(a_val) * np.uint32(a_scale[i, kk // 32])) & mask16)
                rhs = np.uint16((np.uint32(b_val) * np.uint32(b_scale[j, kk // 32])) & mask16)
                s = (s + ((np.uint64(lhs) * np.uint64(rhs)) & mask32)) & mask32
            out[i, j] = s
    return out.astype(np.uint32).view(np.int32)


def test_dot_fma(device, fresh_knobs):
    _require_cuda_backend(device)

    B = 16
    BLOCK = gl.constexpr(B)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, THREADS_PER_WARP: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [THREADS_PER_WARP, 1], [4, 1], [1, 0])
        lhs_layout: gl.constexpr = gl.DotOperandLayout(parent=layout, operand_index=0, k_width=0)
        rhs_layout: gl.constexpr = gl.DotOperandLayout(parent=layout, operand_index=1, k_width=0)

        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))[None, :]
        # Important: build separate offsets for A and B.
        # dot_fma expects operands to represent A[M,K] and B[K,N]. Using the same
        # linearized (m,n) offsets for both makes B effectively transposed.
        offs_k = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))[None, :]
        a_offs = offs_m * BLOCK + offs_k
        b_offs = offs_n * BLOCK + offs_m  # load B^T so dot_fma produces A @ B
        out_offs = offs_m * BLOCK + offs_n

        a = gl.convert_layout(gl.load(a_ptr + a_offs), lhs_layout)
        b = gl.convert_layout(gl.load(b_ptr + b_offs), rhs_layout)
        c = gl.load(c_ptr + out_offs)
        out = gl.dot_fma(a, b, c)
        gl.store(out_ptr + out_offs, out)

    rs = np.random.RandomState(0)
    a_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    b_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    exp_bits = _mm_payload_u32(a_bits, b_bits.T, c_bits)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.int32)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.int32)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((B, B), device="cuda", dtype=torch.int32)

    # Wrap int storage as fp32 so fpsan operates on payload bits.
    aw = triton.TensorWrapper(a, dtype=torch.float32)
    bw = triton.TensorWrapper(b, dtype=torch.float32)
    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](aw, bw, cw, outw, THREADS_PER_WARP=THREADS_PER_WARP)

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not (is_hip_cdna4() or is_hip_gfx1250()), reason="Requires DotScaledOp support (CDNA4, or GFX1250)")
@pytest.mark.parametrize("type_a", ["e2m1", "e4m3", "e5m2"])
@pytest.mark.parametrize("type_b", ["e2m1", "e4m3", "e5m2", "bf16"])
def test_dot_scaled(device, type_a, type_b, fresh_knobs):
    _require_cuda_backend(device)

    B = 32
    K = 64
    SCALE_K = K // 32

    def allocator(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int32)

    triton.set_allocator(allocator)
    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @triton.jit
    def kernel(a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr, TYPE_A: tl.constexpr, TYPE_B: tl.constexpr):
        DIV_FACTOR_A: tl.constexpr = 2 if TYPE_A == "e2m1" else 1
        DIV_FACTOR_B: tl.constexpr = 2 if TYPE_B == "e2m1" else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
        SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32

        offs_am = tl.arange(0, BLOCK_M)[:, None]
        offs_bn = tl.arange(0, BLOCK_N)[None, :]
        offs_ak = tl.arange(0, PACKED_BLOCK_K_A)[None, :]
        offs_bk = tl.arange(0, PACKED_BLOCK_K_B)[:, None]

        a = tl.load(a_ptr + offs_am * PACKED_BLOCK_K_A + offs_ak)
        b = tl.load(b_ptr + offs_bk * BLOCK_N + offs_bn)

        offs_scale_ak = tl.arange(0, SCALE_BLOCK_K)[None, :]
        offs_scale_bk = tl.arange(0, SCALE_BLOCK_K)[None, :]
        a_scale = tl.load(a_scale_ptr + offs_am * SCALE_BLOCK_K + offs_scale_ak)
        b_scale = tl.load(b_scale_ptr + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + offs_scale_bk)

        c = tl.dot_scaled(a, a_scale, TYPE_A, b, b_scale, TYPE_B)
        tl.store(out_ptr + offs_am * BLOCK_N + offs_bn, c)

    a_pack = 2 if type_a == "e2m1" else 1
    b_pack = 2 if type_b == "e2m1" else 1
    packed_k_a = K // a_pack
    packed_k_b = K // b_pack

    rs = np.random.RandomState(1)
    a_bits = rs.randint(0, 256, size=(B, packed_k_a)).astype(np.uint8)
    b_bits = rs.randint(0, 256, size=(packed_k_b, B)).astype(np.uint8)
    a_scale_bits = rs.randint(0, 255, size=(B, SCALE_K)).astype(np.uint8)
    b_scale_bits = rs.randint(0, 255, size=(B, SCALE_K)).astype(np.uint8)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.uint8)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.uint8)
    a_scale = torch.tensor(a_scale_bits, device="cuda", dtype=torch.uint8)
    b_scale = torch.tensor(b_scale_bits, device="cuda", dtype=torch.uint8)

    if type_b == "bf16":
        b_bits = rs.randint(0, 65536, size=(packed_k_b, B)).astype(np.uint16)
        b = torch.tensor(b_bits, device="cuda", dtype=torch.uint16).view(torch.bfloat16)

    exp_bits = _dot_scaled_payload_u32(a_bits, b_bits, a_scale_bits, None if type_b == "bf16" else b_scale_bits, a_pack,
                                       b_pack)

    out = torch.empty((B, B), device="cuda", dtype=torch.int32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](a, a_scale, b, b_scale, outw, BLOCK_M=B, BLOCK_N=B, BLOCK_K=K, TYPE_A=type_a, TYPE_B=type_b)

    _assert_payload_equal(out, exp_bits)


MMA_ACC_LAYOUT_CASES = [
    ("legacy_64", 64, TensorMemoryLayout((64, 64), col_stride=1)),
    ("legacy_128", 128, TensorMemoryLayout((128, 128), col_stride=1)),
    ("linear_128", 128, _make_tmem_linear_layout(128, 128)),
]

MMA_UNSUPPORTED_LAYOUT_CASES = [
    ("linear_identity_64", _make_tmem_linear_layout(64, 64), 1),
    ("linear_block_64", _make_tmem_linear_layout_block(64, 64), 1),
    ("mixed", _make_tmem_linear_layout_mixed_128x128(), 1),
    ("block", _make_tmem_linear_layout_block(128, 128), 1),
    ("block_two_ctas", _make_tmem_linear_layout_block(128, 128, two_ctas=True), 2),
]

MMA_SCALED_ACC_LAYOUT_CASES = [
    ("legacy", TensorMemoryLayout((128, 128), col_stride=1)),
    ("linear_identity", _make_tmem_linear_layout(128, 128)),
]

MMA_SCALED_TYPE_CASES = [
    ("e2m1", "e2m1"),
    ("e4m3", "e4m3"),
    ("e5m2", "e5m2"),
    ("e4m3", "e2m1"),
    ("e2m1", "e4m3"),
]

MMA_TWOCTA_LAYOUT_CASES = [
    "linear",
    "legacy",
]


def _fp8_type_to_torch_dtype(elem_type: str):
    if elem_type == "e4m3":
        return torch.float8_e4m3fn
    if elem_type == "e5m2":
        return torch.float8_e5m2
    raise ValueError(f"unsupported fp8 type: {elem_type}")


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_name,b,acc_layout", MMA_ACC_LAYOUT_CASES)
@pytest.mark.parametrize("use_acc", [False, True])
@pytest.mark.parametrize("acc_via_view", [False, True])
def test_tcgen05_mma(device, layout_name, b, acc_layout, use_acc, acc_via_view, fresh_knobs):
    _require_cuda_backend(device)

    B = b
    BLOCK = gl.constexpr(B)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def mma_plain_kernel(a_ptr, b_ptr, c_ptr, out_ptr, USE_ACC: gl.constexpr, ACC_LAYOUT: gl.constexpr,
                         ACC_BASE_LAYOUT: gl.constexpr, ACC_VIA_VIEW: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])

        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))[None, :]
        offs_k_row = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))[:, None]
        offs_k_col = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))[None, :]

        a_offs = offs_m * BLOCK + offs_k_col
        b_offs = offs_k_row * BLOCK + offs_n
        out_offs = offs_m * BLOCK + offs_n

        a_tile = gl.load(a_ptr + a_offs)
        b_tile = gl.load(b_ptr + b_offs)

        smem_layout_a: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK, BLOCK], gl.float32)
        smem_layout_b: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK, BLOCK], gl.float32)
        smem_a = gl.allocate_shared_memory(gl.float32, [BLOCK, BLOCK], smem_layout_a)
        smem_b = gl.allocate_shared_memory(gl.float32, [BLOCK, BLOCK], smem_layout_b)
        smem_a.store(a_tile)
        smem_b.store(b_tile)

        if ACC_VIA_VIEW:
            acc_base = allocate_tensor_memory(gl.float32, [2, BLOCK, BLOCK], layout=ACC_BASE_LAYOUT)
            acc_tmem = acc_base.index(1)
        else:
            acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK, BLOCK], layout=ACC_LAYOUT)
        acc_reg_layout: gl.constexpr = acc_tmem.get_reg_layout()
        if USE_ACC:
            c_tile = gl.load(c_ptr + out_offs)
            acc_init = gl.convert_layout(c_tile, acc_reg_layout)
            acc_tmem.store(acc_init)

        bar = gl.allocate_shared_memory(gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(bar, count=1)

        smem_b_T = smem_b.permute((1, 0))
        tcgen05_mma(smem_a, smem_b_T, acc_tmem, use_acc=USE_ACC, pred=True, mbarriers=[bar])

        mbarrier.wait(bar, phase=0, deps=[smem_a, smem_b])
        mbarrier.invalidate(bar)

        out = acc_tmem.load()
        out = gl.convert_layout(out, layout)
        gl.store(out_ptr + out_offs, out)

    rs = np.random.RandomState(0)
    a_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    b_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    exp_bits = _mm_payload_u32(a_bits, b_bits.T, c_bits if use_acc else None)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.int32)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.int32)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((B, B), device="cuda", dtype=torch.int32)

    aw = triton.TensorWrapper(a, dtype=torch.float32)
    bw = triton.TensorWrapper(b, dtype=torch.float32)
    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)
    if acc_via_view and not isinstance(acc_layout, TensorMemoryLinearLayout):
        pytest.skip("lifted MMA accumulator view coverage uses canonical TMEM linear layouts")
    acc_base_layout = _lift_tmem_layout(acc_layout, [2]) if acc_via_view else acc_layout

    if acc_via_view and B == 64:
        with pytest.raises((CompilationError, RuntimeError, ValueError)) as excinfo:
            mma_plain_kernel[(1, )](
                aw,
                bw,
                cw,
                outw,
                USE_ACC=use_acc,
                ACC_LAYOUT=acc_layout,
                ACC_BASE_LAYOUT=acc_base_layout,
                ACC_VIA_VIEW=acc_via_view,
            )
        msg = str(excinfo.value)
        assert "TMEM layout '32x32b' unsupported for shape [64, 64]" in msg
        assert "PassManager::run failed" not in msg
        assert "Assertion" not in msg
        return

    compiled = mma_plain_kernel[(1, )](
        aw,
        bw,
        cw,
        outw,
        USE_ACC=use_acc,
        ACC_LAYOUT=acc_layout,
        ACC_BASE_LAYOUT=acc_base_layout,
        ACC_VIA_VIEW=acc_via_view,
    )

    _assert_payload_equal(out, exp_bits)
    mma_ops = _assert_mma_codegen_opcodes(compiled)
    if mma_ops:
        assert all("tcgen05.mma.cta_group::1.kind::tf32" in op for op in mma_ops)
    _assert_tmem_allocator_lifetime(compiled, cta_group=1)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("layout_kind", MMA_TWOCTA_LAYOUT_CASES)
@pytest.mark.parametrize("acc_via_view", [False, True])
def test_tcgen05_mma_twocta(device, layout_kind, acc_via_view):
    _require_cuda_backend(device)

    ctas_per_cga = [2, 1]
    ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    block_m = 128 * ctas_per_cga[0]
    block_n = 64 * ctas_per_cga_b[1]
    block_k = 32

    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]
    cta_order = [1, 0]
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)

    shared_layout_a = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16, cga_layout=cga_layout_a)
    shared_layout_b = gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16, cga_layout=cga_layout_b)

    a = torch.randn((block_m, block_k), dtype=torch.float16, device="cuda")
    b = torch.randn((block_k, block_n), dtype=torch.float16, device="cuda")
    out = torch.empty((block_m, block_n), dtype=torch.float32, device="cuda")

    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(a, [block_m, block_k], shared_layout_a)
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(b, [block_k, block_n], shared_layout_b)
    if layout_kind == "linear":
        acc_layout = _make_tmem_linear_layout_mmav5_twocta(block_m, block_n)
    else:
        acc_layout = TensorMemoryLayout(block=(128, block_n // ctas_per_cga[1]), col_stride=1, two_ctas=True,
                                        cga_layout=cga_layout_c)
    if acc_via_view and not isinstance(acc_layout, TensorMemoryLinearLayout):
        pytest.skip("lifted two-CTA MMA view coverage uses canonical TMEM linear layouts")
    acc_base_layout = _lift_tmem_layout(acc_layout, [2]) if acc_via_view else acc_layout
    blocked_c = gl.BlockedLayout([1, 2], [ctas_per_cga[1], 32 // ctas_per_cga[1]], [4, 1], [1, 0],
                                 cga_layout=cga_layout_c)

    compiled = _tcgen05_mma_twocta_linear_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        block_m,
        block_n,
        acc_layout,
        acc_base_layout,
        blocked_c,
        acc_via_view,
        num_warps=4,
        num_ctas=2,
    )
    torch.testing.assert_close(out, torch.matmul(a.to(torch.float32), b.to(torch.float32)), atol=1e-1, rtol=8e-2)

    mma_ops = _assert_mma_codegen_opcodes(compiled)
    assert all("tcgen05.mma.cta_group::2.kind::f16" in op for op in mma_ops)
    assert "tcgen05.commit.cta_group::2" in compiled.asm["ptx"]
    assert "tcgen05.commit.cta_group::2" in compiled.asm["llir"]
    assert ".multicast::cluster" in compiled.asm["ptx"]
    _assert_tmem_allocator_lifetime(compiled, cta_group=2)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tcgen05_mma_twocta_asymmetric_shape_reports_clean_error(device, fresh_knobs, capfd):
    _require_cuda_backend(device)
    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    ctas_per_cga = [1, 2]
    ctas_per_cga_b = [2, 1]
    block_m = 128 * ctas_per_cga[0]
    block_n = 64 * ctas_per_cga_b[1]
    block_k = 32

    cta_order = [1, 0]
    cga_layout_a = _make_2cta_cga_layout(ctas_per_cga, [1, 2], cta_order, 1)
    cga_layout_b = _make_2cta_cga_layout(ctas_per_cga_b, [2, 1], cta_order, 0)
    cga_layout_c = _make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 1)

    shared_layout_a = gl.NVMMASharedLayout.get_default_for([block_m, block_k], gl.float16, cga_layout=cga_layout_a)
    shared_layout_b = gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16, cga_layout=cga_layout_b)

    a = torch.randn((block_m, block_k), dtype=torch.float16, device="cuda")
    b = torch.randn((block_k, block_n), dtype=torch.float16, device="cuda")
    out = torch.empty((block_m, block_n), dtype=torch.float32, device="cuda")
    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(a, [block_m, block_k], shared_layout_a)
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(b, [block_k, block_n], shared_layout_b)
    acc_layout = _make_tmem_linear_layout_mmav5_twocta(block_m, block_n)
    blocked_c = gl.BlockedLayout([1, 2], [ctas_per_cga[1], 32 // ctas_per_cga[1]], [4, 1], [1, 0],
                                 cga_layout=cga_layout_c)

    with pytest.raises((CompilationError, RuntimeError, ValueError)) as excinfo:
        _tcgen05_mma_twocta_linear_kernel[(1, )](
            a_desc,
            b_desc,
            out,
            block_m,
            block_n,
            acc_layout,
            acc_layout,
            blocked_c,
            False,
            num_warps=4,
            num_ctas=2,
        )
    captured = capfd.readouterr()
    msg = str(excinfo.value) + captured.err + captured.out
    assert ("supported MMAv5 tile" in msg or "MMAv5-compatible tensor memory layout" in msg)
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("name,acc_layout,num_ctas", MMA_UNSUPPORTED_LAYOUT_CASES)
def test_tcgen05_mma_unsupported_linear_layout_reports_clean_error(device, name, acc_layout, num_ctas, fresh_knobs,
                                                                   capfd):
    _require_cuda_backend(device)
    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    B = 128
    BLOCK = gl.constexpr(B)

    @gluon.jit
    def mma_unsupported_layout_kernel(a_ptr, b_ptr, out_ptr, ACC_LAYOUT: gl.constexpr):
        blocked: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, blocked))[:, None]
        offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, blocked))[None, :]
        offs_k_row = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, blocked))[:, None]
        offs_k_col = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, blocked))[None, :]

        a_tile = gl.load(a_ptr + offs_m * BLOCK + offs_k_col)
        b_tile = gl.load(b_ptr + offs_k_row * BLOCK + offs_n)

        smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK, BLOCK], gl.float32)
        smem_a = gl.allocate_shared_memory(gl.float32, [BLOCK, BLOCK], smem_layout)
        smem_b = gl.allocate_shared_memory(gl.float32, [BLOCK, BLOCK], smem_layout)
        smem_a.store(a_tile)
        smem_b.store(b_tile)

        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK, BLOCK], layout=ACC_LAYOUT)
        bar = gl.allocate_shared_memory(gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(bar, count=1)
        tcgen05_mma(smem_a, smem_b.permute((1, 0)), acc_tmem, use_acc=False, pred=True, mbarriers=[bar])
        mbarrier.wait(bar, phase=0, deps=[smem_a, smem_b])
        mbarrier.invalidate(bar)

        out = gl.convert_layout(acc_tmem.load(), blocked)
        gl.store(out_ptr + offs_m * BLOCK + offs_n, out)

    a = torch.randint(-(2**31), 2**31 - 1, (B, B), dtype=torch.int32, device="cuda")
    b = torch.randint(-(2**31), 2**31 - 1, (B, B), dtype=torch.int32, device="cuda")
    out = torch.empty((B, B), dtype=torch.int32, device="cuda")

    aw = triton.TensorWrapper(a, dtype=torch.float32)
    bw = triton.TensorWrapper(b, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    with pytest.raises((CompilationError, RuntimeError, ValueError)) as excinfo:
        mma_unsupported_layout_kernel[(1, )](aw, bw, outw, ACC_LAYOUT=acc_layout, num_warps=4, num_ctas=num_ctas)
    captured = capfd.readouterr()
    msg = str(excinfo.value) + captured.err + captured.out
    assert (
        "MMAv5-compatible tensor memory" in msg
        or "TMEM layout '32x32b' unsupported" in msg
        or "allocation shape must match the TMEM linear layout" in msg
        or "TMEM layout shape must be bounded by the memdesc shape and allocShape" in msg
        or "invalid tensor memory layout" in msg
        or "directly supported MMAv5" in msg
        or "CTAs per CGA" in msg
        or "Result has an invalid" in msg
    )
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("elem_type_a,elem_type_b", MMA_SCALED_TYPE_CASES)
@pytest.mark.parametrize("layout_name,acc_layout", MMA_SCALED_ACC_LAYOUT_CASES)
def test_tcgen05_mma_scaled(device, elem_type_a, elem_type_b, layout_name, acc_layout, fresh_knobs):
    _require_cuda_backend(device)

    B = 128
    BLOCK = gl.constexpr(B)
    SCALE_K = gl.constexpr(B // 32)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def mma_scaled_kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr, out_ptr, TYPE_A: gl.constexpr,
                          TYPE_B: gl.constexpr, ACC_LAYOUT: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
        IS_A_FP4: gl.constexpr = TYPE_A == "e2m1"
        IS_B_FP4: gl.constexpr = TYPE_B == "e2m1"
        PACK_FACTOR_A: gl.constexpr = 2 if IS_A_FP4 else 1
        PACK_FACTOR_B: gl.constexpr = 2 if IS_B_FP4 else 1
        PACKED_K_A: gl.constexpr = BLOCK // PACK_FACTOR_A
        PACKED_K_B: gl.constexpr = BLOCK // PACK_FACTOR_B
        ELEM_DTYPE_A: gl.constexpr = gl.uint8 if IS_A_FP4 else (gl.float8e4nv if TYPE_A == "e4m3" else gl.float8e5)
        ELEM_DTYPE_B: gl.constexpr = gl.uint8 if IS_B_FP4 else (gl.float8e4nv if TYPE_B == "e4m3" else gl.float8e5)
        a_nvmma_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK, PACKED_K_A], ELEM_DTYPE_A)
        b_nvmma_layout: gl.constexpr = (gl.NVMMASharedLayout.get_default_for([BLOCK, PACKED_K_B], ELEM_DTYPE_B)
                                        if IS_B_FP4 else gl.NVMMASharedLayout(swizzle_byte_width=128, transposed=False,
                                                                              element_bitwidth=8, rank=2))
        scale_layout: gl.constexpr = TensorMemoryScalesLayout()

        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))[None, :]
        offs_k_row_b = gl.arange(0, PACKED_K_B, layout=gl.SliceLayout(1, layout))[:, None]
        offs_k_col_a = gl.arange(0, PACKED_K_A, layout=gl.SliceLayout(0, layout))[None, :]
        offs_k_col_b = gl.arange(0, PACKED_K_B, layout=gl.SliceLayout(0, layout))[None, :]

        a_tile = gl.load(a_ptr + offs_m * PACKED_K_A + offs_k_col_a)
        c_tile = gl.load(c_ptr + offs_m * BLOCK + offs_n)
        a_smem = gl.allocate_shared_memory(ELEM_DTYPE_A, [BLOCK, PACKED_K_A], a_nvmma_layout, a_tile)
        if IS_B_FP4:
            b_tile = gl.load(b_ptr + offs_m * PACKED_K_B + offs_k_col_b)
            b_smem = gl.allocate_shared_memory(ELEM_DTYPE_B, [BLOCK, PACKED_K_B], b_nvmma_layout, b_tile)
            b_mma = b_smem.permute((1, 0))
        else:
            b_tile = gl.load(b_ptr + offs_k_row_b * BLOCK + offs_n)
            b_smem = gl.allocate_shared_memory(ELEM_DTYPE_B, [PACKED_K_B, BLOCK], b_nvmma_layout, b_tile)
            b_mma = b_smem

        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK, BLOCK], layout=ACC_LAYOUT)
        acc_tmem.store(gl.convert_layout(c_tile, acc_tmem.get_reg_layout()))

        a_scale_tmem = allocate_tensor_memory(gl.int8, [BLOCK, SCALE_K], layout=scale_layout)
        b_scale_tmem = allocate_tensor_memory(gl.int8, [BLOCK, SCALE_K], layout=scale_layout)
        a_scale_reg_layout: gl.constexpr = a_scale_tmem.get_reg_layout()
        b_scale_reg_layout: gl.constexpr = b_scale_tmem.get_reg_layout()
        scale_offs_k = gl.arange(0, SCALE_K, layout=gl.SliceLayout(0, a_scale_reg_layout))[None, :]
        scale_offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, a_scale_reg_layout))[:, None]
        scale_offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, b_scale_reg_layout))[:, None]
        a_scale_tmem.store(gl.load(a_scale_ptr + scale_offs_m * SCALE_K + scale_offs_k))
        b_scale_tmem.store(gl.load(b_scale_ptr + scale_offs_n * SCALE_K + scale_offs_k))

        bar = gl.allocate_shared_memory(gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(bar, count=1)
        tcgen05_mma_scaled(a_smem, b_mma, acc_tmem, a_scale_tmem, b_scale_tmem, TYPE_A, TYPE_B, use_acc=True)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)

        out = gl.convert_layout(acc_tmem.load(), layout)
        gl.store(out_ptr + offs_m * BLOCK + offs_n, out)

    rs = np.random.RandomState(0)
    a_pack = 2 if elem_type_a == "e2m1" else 1
    b_pack = 2 if elem_type_b == "e2m1" else 1
    packed_k_a = B // a_pack
    packed_k_b = B // b_pack
    a_bits = rs.randint(0 if elem_type_a == "e2m1" else 20, 256 if elem_type_a == "e2m1" else 40,
                        size=(B, packed_k_a), dtype=np.uint8)
    if elem_type_b == "e2m1":
        b_bits = rs.randint(0, 256, size=(B, packed_k_b), dtype=np.uint8)
        b_ref_bits = b_bits.T
    else:
        b_bits = rs.randint(20, 40, size=(packed_k_b, B), dtype=np.uint8)
        b_ref_bits = b_bits
    a_scale_bits = rs.randint(1, 4, size=(B, B // 32), dtype=np.int8)
    b_scale_bits = rs.randint(1, 4, size=(B, B // 32), dtype=np.int8)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    exp_bits = _mm_scaled_payload_u32(a_bits, b_ref_bits, a_scale_bits.view(np.uint8), b_scale_bits.view(np.uint8),
                                      c_bits, a_pack=a_pack, b_pack=b_pack)

    if elem_type_a == "e2m1":
        a = torch.tensor(a_bits, device="cuda", dtype=torch.uint8)
    else:
        a = torch.tensor(a_bits, device="cuda", dtype=torch.uint8).view(_fp8_type_to_torch_dtype(elem_type_a))
    if elem_type_b == "e2m1":
        b = torch.tensor(b_bits, device="cuda", dtype=torch.uint8)
    else:
        b = torch.tensor(b_bits, device="cuda", dtype=torch.uint8).view(_fp8_type_to_torch_dtype(elem_type_b))
    a_scale = torch.tensor(a_scale_bits, device="cuda", dtype=torch.int8)
    b_scale = torch.tensor(b_scale_bits, device="cuda", dtype=torch.int8)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((B, B), device="cuda", dtype=torch.int32)

    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    compiled = mma_scaled_kernel[(1, )](a, b, a_scale, b_scale, cw, outw, TYPE_A=elem_type_a, TYPE_B=elem_type_b,
                                        ACC_LAYOUT=acc_layout)

    _assert_payload_equal(out, exp_bits)
    mma_ops = _assert_mma_codegen_opcodes(compiled)
    if mma_ops:
        assert all(op.startswith("tcgen05.mma.cta_group::1.kind::") for op in mma_ops)
        assert all("block_scale.scale_vec::" in op for op in mma_ops)
        assert "tcgen05.commit.cta_group::1" in compiled.asm["ptx"]
        assert "tcgen05.commit.cta_group::1" in compiled.asm["llir"]


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize(
    "acc_layout",
    [_make_tmem_linear_layout_mixed_128x128(), _make_tmem_linear_layout_block(128, 128)],
)
def test_tcgen05_mma_scaled_unsupported_linear_layout_reports_clean_error(device, acc_layout, fresh_knobs, capfd):
    _require_cuda_backend(device)
    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    B = 128
    BLOCK = gl.constexpr(B)
    SCALE_K = gl.constexpr(B // 32)

    @gluon.jit
    def mma_scaled_unsupported_layout_kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr, out_ptr,
                                             ACC_LAYOUT: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
        a_nvmma_layout: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, transposed=False,
                                                            element_bitwidth=8, rank=2)

        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))[None, :]
        offs_k = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))[None, :]

        a_tile = gl.load(a_ptr + offs_m * BLOCK + offs_k)
        b_tile = gl.load(b_ptr + offs_m * BLOCK + offs_k).permute((1, 0))
        c_tile = gl.load(c_ptr + offs_m * BLOCK + offs_n)

        a_smem = gl.allocate_shared_memory(gl.float8e5, [BLOCK, BLOCK], a_nvmma_layout, a_tile)
        b_smem = gl.allocate_shared_memory(gl.float8e5, [BLOCK, BLOCK], a_nvmma_layout, b_tile)

        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK, BLOCK], layout=ACC_LAYOUT)
        acc_tmem.store(gl.convert_layout(c_tile, acc_tmem.get_reg_layout()))

        scale_layout: gl.constexpr = TensorMemoryScalesLayout()
        a_scale_tmem = allocate_tensor_memory(gl.int8, [BLOCK, SCALE_K], layout=scale_layout)
        b_scale_tmem = allocate_tensor_memory(gl.int8, [BLOCK, SCALE_K], layout=scale_layout)
        a_scale_reg_layout: gl.constexpr = a_scale_tmem.get_reg_layout()
        b_scale_reg_layout: gl.constexpr = b_scale_tmem.get_reg_layout()
        scale_offs_k = gl.arange(0, SCALE_K, layout=gl.SliceLayout(0, a_scale_reg_layout))[None, :]
        scale_offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, a_scale_reg_layout))[:, None]
        scale_offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, b_scale_reg_layout))[:, None]
        a_scale_tmem.store(gl.load(a_scale_ptr + scale_offs_m * SCALE_K + scale_offs_k))
        b_scale_tmem.store(gl.load(b_scale_ptr + scale_offs_n * SCALE_K + scale_offs_k))

        bar = gl.allocate_shared_memory(gl.int64, [1], gl.constexpr(mbarrier.MBarrierLayout()))
        mbarrier.init(bar, count=1)
        tcgen05_mma_scaled(a_smem, b_smem, acc_tmem, a_scale_tmem, b_scale_tmem, "e5m2", "e5m2", use_acc=True)
        tcgen05_commit(bar)
        mbarrier.wait(bar, phase=0)

        out = gl.convert_layout(acc_tmem.load(), layout)
        gl.store(out_ptr + offs_m * BLOCK + offs_n, out)

    rs = np.random.RandomState(1)
    a = torch.tensor(rs.randint(20, 40, size=(B, B), dtype=np.uint8), device="cuda", dtype=torch.uint8).view(
        torch.float8_e5m2
    )
    b = torch.tensor(rs.randint(20, 40, size=(B, B), dtype=np.uint8), device="cuda", dtype=torch.uint8).view(
        torch.float8_e5m2
    )
    a_scale = torch.tensor(rs.randint(1, 4, size=(B, B // 32), dtype=np.int8), device="cuda", dtype=torch.int8)
    b_scale = torch.tensor(rs.randint(1, 4, size=(B, B // 32), dtype=np.int8), device="cuda", dtype=torch.int8)
    c = torch.tensor(rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32), device="cuda", dtype=torch.int32)
    out = torch.empty((B, B), device="cuda", dtype=torch.int32)

    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    with pytest.raises((CompilationError, RuntimeError, ValueError)) as excinfo:
        mma_scaled_unsupported_layout_kernel[(1, )](a, b, a_scale, b_scale, cw, outw, ACC_LAYOUT=acc_layout,
                                                    num_warps=4)
    captured = capfd.readouterr()
    msg = str(excinfo.value) + captured.err + captured.out
    assert ("MMAv5-compatible tensor memory" in msg or "TMEM layout '32x32b' unsupported" in msg or "directly supported MMAv5" in msg or "block-scaled tensor memory" in msg or "CTAs per CGA" in msg or "Result has an invalid" in msg)
    assert "PassManager::run failed" not in msg
    assert "Assertion" not in msg


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tmem_index_subslice(device, fresh_knobs):
    _require_cuda_backend(device)

    B = 64
    BLOCK = gl.constexpr(B)
    SLICE_N = gl.constexpr(32)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(x_ptr, out_ptr):
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [gl.num_warps(), 1], [1, 0])
        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))[:, None]
        offs_n = gl.arange(0, SLICE_N, layout=gl.SliceLayout(0, layout))[None, :]
        offs = offs_m * SLICE_N + offs_n

        x = gl.load(x_ptr + offs)

        tmem_layout: gl.constexpr = TensorMemoryLayout((BLOCK, BLOCK), col_stride=1)
        tmem = allocate_tensor_memory(gl.float32, [2, BLOCK, BLOCK], layout=tmem_layout)
        view = tmem.index(1)
        sub = view.slice(0, SLICE_N, dim=1)

        sub_reg_layout: gl.constexpr = sub.get_reg_layout(instr_variant="32x32b_splitn")
        x_reg = gl.convert_layout(x, sub_reg_layout)
        sub.store(x_reg)
        out = sub.load()
        out = gl.convert_layout(out, layout)
        gl.store(out_ptr + offs, out)

    rs = np.random.RandomState(0)
    x_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, 32), dtype=np.int32)
    exp_bits = x_bits.copy()

    x = torch.tensor(x_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((B, 32), device="cuda", dtype=torch.int32)

    xw = triton.TensorWrapper(x, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](xw, outw)

    torch.testing.assert_close(out, x)
    np.testing.assert_array_equal(out.cpu().numpy().view(np.int32), exp_bits)


def test_reduction(device, fresh_knobs):
    _require_cuda_backend(device)

    @triton.jit
    def reduce_kernel(a_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr,
                      ORDER: tl.constexpr):
        a_ptrs = a_ptr + (tl.arange(0, M)[:, None] * stride_am + (tl.arange(0, N)[None, :]) * stride_ak)
        a = tl.load(a_ptrs)
        r1 = tl.sum(a, axis=ORDER)
        r2 = tl.sum(r1, axis=ORDER - 1)
        tl.store(c_ptr, r2)

    M, N = 512, 512
    torch.manual_seed(0)
    a = torch.randn((M, N), dtype=torch.float32, device="cuda")
    # Make non-associativity visible and deterministic: large + tiny magnitudes.
    a[:, :64] *= 1e10
    a[:, 64:] *= 1e-10
    c1 = torch.empty((1, ), dtype=torch.float32).to('cuda')
    c2 = torch.empty((1, ), dtype=torch.float32).to('cuda')

    reduce_kernel[(1, )](a, c1, M=M, N=N, stride_am=a.stride(0), stride_ak=a.stride(1), ORDER=0)
    reduce_kernel[(1, )](a, c2, M=M, N=N, stride_am=a.stride(0), stride_ak=a.stride(1), ORDER=1)
    assert not _payload_equal(c1, c2)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    reduce_kernel[(1, )](a, c1, M=M, N=N, stride_am=a.stride(0), stride_ak=a.stride(1), ORDER=0)
    reduce_kernel[(1, )](a, c2, M=M, N=N, stride_am=a.stride(0), stride_ak=a.stride(1), ORDER=1)
    assert _payload_equal(c1, c2)


def test_reduction_matches_loop(device, fresh_knobs):
    _require_cuda_backend(device)

    @triton.jit
    def reduce_sum_kernel(x_ptr, out_ptr, N: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, N))
        tl.store(out_ptr, tl.sum(x, axis=0))

    @triton.jit
    def loop_sum_kernel(x_ptr, out_ptr, N: tl.constexpr):
        acc = tl.full([], 0.0, tl.float32)
        for i in tl.static_range(0, N):
            acc += tl.load(x_ptr + i)
        tl.store(out_ptr, acc)

    N = 256
    pattern = torch.tensor([1e20, 1.0, -1e20, 1.0], dtype=torch.float32, device="cuda")
    x = pattern.repeat(N // pattern.numel())
    reduce_out = torch.empty((1, ), dtype=torch.float32, device="cuda")
    loop_out = torch.empty((1, ), dtype=torch.float32, device="cuda")

    reduce_sum_kernel[(1, )](x, reduce_out, N=N)
    loop_sum_kernel[(1, )](x, loop_out, N=N)
    assert not _payload_equal(reduce_out, loop_out)

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    reduce_sum_kernel[(1, )](x, reduce_out, N=N)
    loop_sum_kernel[(1, )](x, loop_out, N=N)
    _assert_payload_equal(reduce_out, loop_out)


@pytest.mark.skipif(not (is_hip_cdna3() or is_hip_cdna4()), reason="Requires CDNA3 or CDNA4")
def test_mfma_dot(device, fresh_knobs):
    _require_cuda_backend(device)

    M, N, K = 16, 16, 32

    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    cdna_version = 3 if is_hip_cdna3() else 4
    nonkdim = 32
    kdim = 8 if cdna_version == 3 else 16
    k_width_val = 4 if cdna_version == 3 else 8

    blocked = gl.BlockedLayout([4, 4], [4, 16], [4, 1], [1, 0])
    mfma_layout = gl.amd.AMDMFMALayout(cdna_version, [nonkdim, nonkdim, kdim], True, [4, 1])

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
               blocked: gl.constexpr, k_width: gl.constexpr, mfma_layout: gl.constexpr):
        dot_a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=k_width)
        dot_b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=k_width)

        offs_am = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked))
        offs_bn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked))
        offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked))
        offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, blocked))

        a = gl.load(a_ptr + offs_am[:, None] * BLOCK_K + offs_ak[None, :])
        b = gl.load(b_ptr + offs_bk[:, None] * BLOCK_N + offs_bn[None, :])
        c = gl.load(c_ptr + offs_am[:, None] * BLOCK_N + offs_bn[None, :])

        a1 = gl.convert_layout(a, layout=dot_a_layout)
        b1 = gl.convert_layout(b, layout=dot_b_layout)
        c_acc = gl.convert_layout(c, layout=mfma_layout)

        result = gl.amd.cdna3.mfma(a1, b1, c_acc)
        result = gl.convert_layout(result, layout=blocked)
        gl.store(out_ptr + offs_am[:, None] * BLOCK_N + offs_bn[None, :], result)

    rs = np.random.RandomState(0)
    a_bits = rs.randint(-(2**31), 2**31 - 1, size=(M, K), dtype=np.int32)
    b_bits = rs.randint(-(2**31), 2**31 - 1, size=(K, N), dtype=np.int32)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(M, N), dtype=np.int32)
    exp_bits = _mm_payload_u32(a_bits, b_bits, c_bits)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.int32)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.int32)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((M, N), device="cuda", dtype=torch.int32)

    aw = triton.TensorWrapper(a, dtype=torch.float32)
    bw = triton.TensorWrapper(b, dtype=torch.float32)
    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](aw, bw, cw, outw, BLOCK_M=M, BLOCK_N=N, BLOCK_K=K, blocked=blocked, k_width=k_width_val,
                  mfma_layout=mfma_layout)

    _assert_payload_equal(out, exp_bits)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
def test_wmma_dot(device, fresh_knobs):
    _require_cuda_backend(device)

    B = 32
    fresh_knobs.compilation.instrumentation_mode = "fpsan"

    @gluon.jit
    def kernel(a_ptr, b_ptr, c_ptr, out_ptr, BLOCK: gl.constexpr, INSTR_SHAPE_K: gl.constexpr, K_WIDTH: gl.constexpr):
        blocked: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        wmma: gl.constexpr = gl.amd.AMDWMMALayout(3, True, [[0, 1], [1, 0]], [], [16, 16, INSTR_SHAPE_K])

        offs_m = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, blocked))[:, None]
        offs_k = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, blocked))[None, :]
        offs_bk = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, blocked))[:, None]
        offs_n = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, blocked))[None, :]

        a = gl.load(a_ptr + offs_m * BLOCK + offs_k)
        b = gl.load(b_ptr + offs_bk * BLOCK + offs_n)
        c = gl.load(c_ptr + offs_m * BLOCK + offs_n)
        c = gl.convert_layout(c, wmma)

        a = gl.convert_layout(a, gl.DotOperandLayout(0, wmma, K_WIDTH))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, wmma, K_WIDTH))
        acc = gl.amd.gfx1250.wmma(a, b, c)

        out_layout: gl.constexpr = gl.SliceLayout(1, wmma)
        offs_cm = gl.arange(0, BLOCK, layout=out_layout)[:, None]
        offs_cn = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, wmma))[None, :]
        gl.store(out_ptr + offs_cm * BLOCK + offs_cn, acc)

    rs = np.random.RandomState(0)
    a_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    b_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    c_bits = rs.randint(-(2**31), 2**31 - 1, size=(B, B), dtype=np.int32)
    exp_bits = _mm_payload_u32(a_bits, b_bits, c_bits)

    a = torch.tensor(a_bits, device="cuda", dtype=torch.int32)
    b = torch.tensor(b_bits, device="cuda", dtype=torch.int32)
    c = torch.tensor(c_bits, device="cuda", dtype=torch.int32)
    out = torch.empty((B, B), device="cuda", dtype=torch.int32)

    aw = triton.TensorWrapper(a, dtype=torch.float32)
    bw = triton.TensorWrapper(b, dtype=torch.float32)
    cw = triton.TensorWrapper(c, dtype=torch.float32)
    outw = triton.TensorWrapper(out, dtype=torch.float32)

    kernel[(1, )](aw, bw, cw, outw, BLOCK=B, INSTR_SHAPE_K=4, K_WIDTH=2)

    _assert_payload_equal(out, exp_bits)
