import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from triton._filecheck import run_parser
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
)

BLACKWELL_TARGET = GPUTarget("cuda", 100, 32)


@gluon.jit
def inspect_acc_tmem_kernel():
    num_ctas: ttgl.constexpr = ttgl.num_ctas()
    two_ctas: ttgl.constexpr = num_ctas > 1
    tmem_layout: ttgl.constexpr = TensorMemoryLayout(
        [256 // num_ctas, 128],
        col_stride=1,
        cga_layout=[[1, 0]] if two_ctas else [],
        two_ctas=two_ctas,
    )
    computed_layout: ttgl.constexpr = ttgl.to_linear_layout(tmem_layout, [256, 128])
    ttgl.static_print(computed_layout)
    acc_tmem = allocate_tensor_memory(ttgl.float32, [256, 128], tmem_layout)
    reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    val = ttgl.full([256, 128], 0.0, ttgl.float32, layout=reg_layout)
    acc_tmem.store(val)


if __name__ == "__main__":
    mod = run_parser(
        inspect_acc_tmem_kernel,
        kwargs={"num_ctas": 2, "num_warps": 4},
        target=BLACKWELL_TARGET,
    )
    print(mod)
