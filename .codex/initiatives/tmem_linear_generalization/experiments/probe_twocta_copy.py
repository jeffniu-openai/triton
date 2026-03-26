import os

import torch

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    fence_async_shared,
    tcgen05_commit,
    tcgen05_copy,
)
from triton.experimental.gluon.language.nvidia.hopper import cluster, mbarrier

from python.test.gluon.test_tmem_runtime_matrix import (
    _extract_tcgen05_cp_opcodes,
    _make_2cta_cga_layout,
    _make_tmem_linear_layout_mmav5_twocta,
)


@gluon.jit
def probe_twocta_copy(in_ptr, out_ptr, layout: ttgl.constexpr,
                      cga_layout: ttgl.constexpr, M: ttgl.constexpr,
                      N: ttgl.constexpr, swizzle: ttgl.constexpr,
                      barrier_count: ttgl.constexpr,
                      pre_copy_cluster_fence: ttgl.constexpr,
                      pre_copy_cluster_barrier: ttgl.constexpr,
                      post_wait_cluster_barrier: ttgl.constexpr):
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout)
    reg_layout: ttgl.constexpr = tmem.get_reg_layout()

    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, reg_layout))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]
    value = ttgl.load(in_ptr + offs)

    smem_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
        swizzle_byte_width=swizzle,
        element_bitwidth=32,
        rank=2,
        cga_layout=cga_layout,
    )
    smem = ttgl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N],
                                       layout=smem_layout)
    smem.store(value)
    if pre_copy_cluster_fence:
        fence_async_shared(cluster=True)
    if pre_copy_cluster_barrier:
        cluster.barrier()

    bar = mbarrier.allocate_mbarrier()
    mbarrier.init(bar, count=barrier_count)
    tcgen05_copy(smem, tmem)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    if post_wait_cluster_barrier:
        cluster.barrier()

    out = tmem.load(reg_layout)
    ttgl.store(out_ptr + offs, out)


def main():
    ctas_per_cga = (2, 1)
    cta_split = (2, 1)
    cta_order = (1, 0)
    cga_layout = _make_2cta_cga_layout(ctas_per_cga, cta_split, cta_order, 0)

    M, N, swizzle = 256, 128, 128
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty_like(inp)
    layout = _make_tmem_linear_layout_mmav5_twocta(M, N)
    compiled = probe_twocta_copy[(1, )](
        inp,
        out,
        layout,
        tuple(tuple(b) for b in cga_layout),
        M,
        N,
        swizzle,
        int(os.environ.get("BARRIER_COUNT", "1")),
        os.environ.get("PRE_COPY_CLUSTER_FENCE", "0") == "1",
        os.environ.get("PRE_COPY_CLUSTER_BARRIER", "0") == "1",
        os.environ.get("POST_WAIT_CLUSTER_BARRIER", "0") == "1",
        num_ctas=2,
        num_warps=4,
    )
    print("PRE_COPY_CLUSTER_FENCE", os.environ.get("PRE_COPY_CLUSTER_FENCE", "0"))
    print("PRE_COPY_CLUSTER_BARRIER",
          os.environ.get("PRE_COPY_CLUSTER_BARRIER", "0"))
    print("POST_WAIT_CLUSTER_BARRIER",
          os.environ.get("POST_WAIT_CLUSTER_BARRIER", "0"))
    print("TTGIR_HAS_ATTR", "ttng.two-ctas" in compiled.asm["ttgir"])
    print("TTGIR_HAS_FLAG", "two_ctas" in compiled.asm["ttgir"])
    print("PTX_OPS", _extract_tcgen05_cp_opcodes(compiled.asm["ptx"])[:8])
    print("LLIR_OPS", _extract_tcgen05_cp_opcodes(compiled.asm["llir"])[:8])
    print("HAS_COMMIT2_PTX", "tcgen05.commit.cta_group::2" in compiled.asm["ptx"])
    print("HAS_COMMIT2_LLIR", "tcgen05.commit.cta_group::2" in compiled.asm["llir"])
    print("OUT_OK", bool(torch.equal(out, inp)))


if __name__ == "__main__":
    main()
