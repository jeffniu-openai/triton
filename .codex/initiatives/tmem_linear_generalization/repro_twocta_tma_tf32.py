import torch

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLayout

from python.test.gluon.test_tmem_runtime_matrix import (
    _make_2cta_cga_layout,
    _make_tmem_linear_layout_mmav5_twocta,
    _round_to_tf32,
    tmem_mma_twocta_kernel,
)


EXPECTED = "tcgen05.mma does not support transposed float32 operands in shared memory"


def run(layout_kind: str):
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

    shared_layout_a = ttgl.NVMMASharedLayout.get_default_for(
        [block_m, block_k], ttgl.float32, cga_layout=cga_layout_a
    )
    shared_layout_b = ttgl.NVMMASharedLayout.get_default_for(
        [block_k, block_n], ttgl.float32, cga_layout=cga_layout_b
    )

    a = _round_to_tf32(torch.randn((block_m, block_k), dtype=torch.float32, device="cuda"))
    b = _round_to_tf32(torch.randn((block_k, block_n), dtype=torch.float32, device="cuda"))
    out = torch.empty((block_m, block_n), dtype=torch.float32, device="cuda")

    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(a, [block_m, block_k], shared_layout_a)
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(b, [block_k, block_n], shared_layout_b)

    if layout_kind == "legacy":
        acc_layout = TensorMemoryLayout(
            block=(128, block_n // ctas_per_cga[1]),
            col_stride=1,
            two_ctas=True,
            cga_layout=cga_layout_c,
        )
    else:
        acc_layout = _make_tmem_linear_layout_mmav5_twocta(block_m, block_n)

    blocked_c = ttgl.BlockedLayout(
        [1, 2],
        [ctas_per_cga[1], 32 // ctas_per_cga[1]],
        [4, 1],
        [1, 0],
        cga_layout=cga_layout_c,
    )

    tmem_mma_twocta_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        block_m,
        block_n,
        acc_layout,
        blocked_c,
        num_warps=4,
        num_ctas=2,
    )


def main():
    for layout_kind in ("legacy", "linear"):
        try:
            run(layout_kind)
        except Exception as exc:
            text = str(exc)
            if EXPECTED not in text and "PassManager::run failed" not in text:
                raise
            print(f"{layout_kind}: reproduced expected failure")
            continue
        raise AssertionError(f"{layout_kind}: unexpectedly passed")


if __name__ == "__main__":
    main()
