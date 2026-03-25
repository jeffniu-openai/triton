import torch

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl


@gluon.jit
def probe_shared_reshape_slice_dim0_kernel(in_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr):
    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    shared: ttgl.constexpr = ttgl.SharedLinearLayout(
        offset_bases=[
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
        ],
        alignment=16,
    )
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, blocked))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, blocked))
    offs = offs_m[:, None] * N + offs_n[None, :]

    inp = ttgl.load(in_ptr + offs)
    smem = ttgl.allocate_shared_memory(ttgl.float32, [M, N], shared)
    smem.store(inp)

    view = smem.reshape((2, M, N // 2)).slice(1, 1, dim=0).index(0)
    reg: ttgl.constexpr = blocked
    view.store(view.load(reg) + ttgl.full([M, N // 2], 7.0, ttgl.float32, layout=reg))

    out = smem.load(reg)
    ttgl.store(out_ptr + offs, out)


@gluon.jit
def probe_shared_reshape_half_rows_kernel(in_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr):
    blocked: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
    shared: ttgl.constexpr = ttgl.SharedLinearLayout(
        offset_bases=[
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
        ],
        alignment=16,
    )
    offs_m = ttgl.arange(0, M, ttgl.SliceLayout(1, blocked))
    offs_n = ttgl.arange(0, N, ttgl.SliceLayout(0, blocked))
    offs = offs_m[:, None] * N + offs_n[None, :]

    inp = ttgl.load(in_ptr + offs)
    smem = ttgl.allocate_shared_memory(ttgl.float32, [M, N], shared)
    smem.store(inp)

    view = smem.reshape((2, M // 2, N)).slice(1, 1, dim=0).index(0)
    reg: ttgl.constexpr = blocked
    view.store(view.load(reg) + ttgl.full([M // 2, N], 13.0, ttgl.float32, layout=reg))

    out = smem.load(reg)
    ttgl.store(out_ptr + offs, out)


def run_probe(kernel, m=128, n=128):
    inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
    out = torch.empty_like(inp)
    return kernel[(1,)](inp, out, m, n, num_warps=4), inp, out


def main():
    for name, kernel in [
        ("shared_dim0", probe_shared_reshape_slice_dim0_kernel),
        ("shared_half_rows", probe_shared_reshape_half_rows_kernel),
    ]:
        compiled, inp, out = run_probe(kernel)
        diff = out - inp
        print(name, torch.unique(diff).cpu().tolist(), int((diff != 0).sum().item()))
        print(diff[:4, :8])
        print(compiled.asm["ttgir"])


if __name__ == "__main__":
    main()
