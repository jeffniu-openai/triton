import itertools
import math

from triton.tools import LinearLayout


M, N = 128, 32

ROW_BASES = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]]
COL_BASES = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]]


def make_shared_layout(interleaved_row: int, insert_at: int):
    col_bases = COL_BASES.copy()
    row_bases = ROW_BASES.copy()
    chosen = [interleaved_row, 0]
    row_bases = [basis for basis in row_bases if basis != chosen]
    offset_bases = col_bases[:insert_at] + [chosen] + col_bases[insert_at:] + row_bases
    return offset_bases


def make_linear_layout(bases_map, shape):
    base_items = [(name, bases) for name, bases in bases_map.items()]
    return LinearLayout.from_bases(base_items, ["dim0", "dim1"], shape)


def is_zero_basis(basis):
    return basis is not None and basis == [0, 0]


def analyze():
    tmem_layout = make_linear_layout(
        {"row": [[1 << i, 0] for i in range(int(math.log2(M)))], "col": [[0, 1 << i] for i in range(int(math.log2(N)))]},
        [M, N],
    )
    for interleaved_row in (32, 64):
        for insert_at in range(len(COL_BASES) + 1):
            shared_layout = make_linear_layout(
                {"offset": make_shared_layout(interleaved_row, insert_at)},
                [M, N],
            )
            cvt = tmem_layout.invert_and_compose(shared_layout)
            bases_map = {name: bases for name, bases in cvt.bases}
            row_bases_list = bases_map.get("row", [])
            idx5 = int(math.log2(32)) + 0
            idx6 = int(math.log2(32)) + 1
            base5 = row_bases_list[idx5] if idx5 < len(row_bases_list) else None
            base6 = row_bases_list[idx6] if idx6 < len(row_bases_list) else None
            multicast = (base5 == [0, 0]) | ((base6 == [0, 0]) << 1)
            bit5 = is_zero_basis(base5)
            bit6 = is_zero_basis(base6)
            multicast = bit5 | (bit6 << 1)
            print(
                f"row={interleaved_row}, insert={insert_at}, basis5={base5}, basis6={base6}, dispatch={multicast}"
            )


if __name__ == "__main__":
    analyze()
