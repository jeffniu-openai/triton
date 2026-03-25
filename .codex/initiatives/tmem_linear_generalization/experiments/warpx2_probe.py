import math
from typing import Sequence

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_copy,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier


M, N = 128, 128
M_SIZE = ttgl.constexpr(M)
N_SIZE = ttgl.constexpr(N)

TMEM_LAYOUTS = {
    1: TensorMemoryLayout(block=(128, 128), col_stride=1, two_ctas=False),
    2: TensorMemoryLayout(block=(128, 128), col_stride=1, two_ctas=True),
}
TMEM_LAYOUT_CONST = {
    idx: ttgl.constexpr(layout)
    for idx, layout in TMEM_LAYOUTS.items()
}


def _build_cp_probe(layout_const: ttgl.constexpr):
    @gluon.jit
    def cp_probe(in_ptr, out_ptr, shared_layout: ttgl.constexpr):
        offs = ttgl.arange(0, M_SIZE)[:, None] * N_SIZE + ttgl.arange(0, N_SIZE)[None, :]
        value = ttgl.load(in_ptr + offs)
        tmem = allocate_tensor_memory(ttgl.int32, [M_SIZE, N_SIZE], layout=layout_const)
        tmem.store(value)
        smem = ttgl.allocate_shared_memory(ttgl.int32, [M_SIZE, N_SIZE], layout=shared_layout)
        smem.store(value)
        barrier = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(barrier, count=1)
        tcgen05_copy(smem, tmem)
        mbarrier.wait(barrier, phase=0)
        out = tmem.load()
        ttgl.store(out_ptr + offs, out)

    return cp_probe


PROBE_KERNELS = {idx: _build_cp_probe(layout) for idx, layout in TMEM_LAYOUT_CONST.items()}


def _make_shared_linear_layout(offset_bases: Sequence[Sequence[int]]):
    return ttgl.SharedLinearLayout(offset_bases=offset_bases, alignment=16)


def _extract_cp_opcodes(asm: str) -> list[str]:
    return [line for line in asm.splitlines() if "tcgen05.cp" in line]


def main():
    inp = torch.arange(M * N, dtype=torch.int32, device="cuda").reshape(M, N)
    out = torch.empty_like(inp)
    candidate_bases = [
        # roster of row-first bases plus column strides
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [64, 0],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
        ],
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [64, 0],
        ],
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [0, 32],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 64],
            [32, 0],
            [64, 0],
        ],
        [
            [32, 0],
            [64, 0],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
        ],
    ]

    def try_variants(num_ctas: int):
        found = {}
        kernel = PROBE_KERNELS[num_ctas]
        for idx, bases in enumerate(candidate_bases):
            shared_layout = _make_shared_linear_layout(bases)
            try:
                compiled = kernel[(1,)](
                    inp,
                    out,
                    shared_layout,
                    num_warps=4,
                    num_ctas=num_ctas,
                )
            except Exception as err:
                print(f"variant {idx} (ctas={num_ctas}) failed: {err}")
                continue
            ops = _extract_cp_opcodes(compiled.asm["ptx"])
            warpx2_ops = [op for op in ops if "warpx2" in op]
            if warpx2_ops:
                found[tuple(map(tuple, bases))] = (warpx2_ops, compiled.asm)
                break
        return found

    results = {
        1: try_variants(1),
        2: try_variants(2),
    }

    for ctas, pairs in results.items():
        for bases_key, (ops, asm) in pairs.items():
            print(f"num_ctas={ctas}, layout={bases_key}")
            for opcode in ops:
                llir_ops = _extract_cp_opcodes(asm["llir"])
                print("  ptx:", opcode)
                match = [ll for ll in llir_ops if opcode in ll]
                print("  llir:", match[0] if match else "n/a")


if __name__ == "__main__":
    main()
