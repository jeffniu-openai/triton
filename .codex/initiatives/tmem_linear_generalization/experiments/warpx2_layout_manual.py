import importlib.util
import itertools
import os

import torch

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLayout
from triton.experimental.gluon.language.nvidia.hopper import mbarrier

spec = importlib.util.spec_from_file_location(
    "cp_warpx2_probe_test",
    os.path.join(
        os.path.dirname(__file__),
        "cp_warpx2_probe_test.py",
    ),
)
cp_warpx2_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp_warpx2_probe)
extract_tcgen05_cp_opcodes = cp_warpx2_probe._extract_tcgen05_cp_opcodes
probe_scales_copy_kernel = cp_warpx2_probe.probe_scales_copy_kernel


def make_layout(interleaved_row: int, insert_at: int) -> ttgl.SharedLinearLayout:
    col_bases = [[0, 1], [0, 2], [0, 4], [0, 8]]
    row_bases = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]]
    chosen = [interleaved_row, 0]
    row_bases = [basis for basis in row_bases if basis != chosen]
    offset_bases = col_bases[:insert_at] + [chosen] + col_bases[insert_at:] + row_bases
    return ttgl.SharedLinearLayout(offset_bases=offset_bases, alignment=16)


def main():
    smem_shape = (64, 16)
    alias_shape = (128, 32)
    inp = torch.arange(smem_shape[0] * smem_shape[1], dtype=torch.int8, device="cuda").reshape(
        smem_shape
    )
    out = torch.empty(alias_shape, dtype=torch.int8, device="cuda")

    found = {}
    for interleaved_row in (32, 64):
        for insert_at in range(5):
            layout = make_layout(interleaved_row, insert_at)
            try:
                compiled = probe_scales_copy_kernel[(1,)](
                    inp,
                    out,
                    layout,
                    smem_shape[0],
                    smem_shape[1],
                    alias_shape[0],
                    alias_shape[1],
                    num_warps=4,
                )
            except Exception:
                continue
            ops = extract_tcgen05_cp_opcodes(compiled.asm["ptx"])
            for opcode in ops:
                if "warpx2::02_13" in opcode or "warpx2::01_23" in opcode:
                    found_opcode = opcode
                    found.setdefault(opcode, []).append((interleaved_row, insert_at, layout))
                    print(f"{opcode} from row={interleaved_row}, insert_at={insert_at}")
    if not found:
        print("no warpx2 cp cases found")


if __name__ == "__main__":
    main()
