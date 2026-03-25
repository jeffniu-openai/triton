import torch

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLinearLayout,
    allocate_tensor_memory,
)


def make_layout(m, n):
    return TensorMemoryLinearLayout(
        rows=[[1 << i, 0] for i in range(m.bit_length() - 1)],
        cols=[[0, 1 << i] for i in range(n.bit_length() - 1)],
        shape=[m, n],
    )


@gluon.jit
def kernel(in_ptr, out_ptr, layout: ttgl.constexpr):
    m: ttgl.constexpr = 128
    n: ttgl.constexpr = 128
    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    offs = ttgl.arange(0, m, ttgl.SliceLayout(1, blocked))[:, None] * n + ttgl.arange(
        0, n, ttgl.SliceLayout(0, blocked)
    )[None, :]
    value = ttgl.load(in_ptr + offs)
    tmem = allocate_tensor_memory(ttgl.float32, [m, n], layout)
    tmem.store(value)
    value = tmem.load(blocked)
    ttgl.store(out_ptr + offs, value)


def main():
    layout = make_layout(128, 128)
    inp = torch.empty((128, 128), device="cuda", dtype=torch.float32)
    out = torch.empty_like(inp)
    compiled = kernel.warmup(inp, out, layout, grid=(1,), num_warps=4)
    print("===TTGIR===")
    print(compiled.asm["ttgir"])
    print("===LLIR===")
    print(compiled.asm["llir"])
    print("===PTX===")
    print(compiled.asm["ptx"])


if __name__ == "__main__":
    main()
