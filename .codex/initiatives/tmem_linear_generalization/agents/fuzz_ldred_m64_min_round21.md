# Round 21 Lane AZ: M64 `ld.red` Row-Basis Minimization

- Date: 2026-04-21 12:02 UTC
- Branch: `codex/tmem`
- Mode: discovery/catalog only; no backend/compiler repairs attempted.
- Repo edit scope: this report only.
- Target: existing `FZ-20260421-0012`, especially
  `row_reverse_n32` and `row_rotate_col_even_odd_n128`, with
  `col_reverse_n32` as the nearby passing control.

## Summary

`FZ-20260421-0012` minimizes to the M64 row basis, not to a specific
column layout, `N`, reduction op, or user-requested load variant.

The stable split from this lane:

- M64 f32 `ld.red` with a non-identity row basis fails during Gluon
  parsing/lowering of `ttng.tmem_load` with `unsupported dst layout`.
- M64 f32 `ld.red` with an identity row basis and a column-only permutation
  passes and emits `tcgen05.ld.red`.
- Explicit `auto`, `32x32b`, `16x32bx2`, and `32x32b_splitn` requests do not
  rescue the row-permuted M64 cases.
- The failure happens before runtime execution and before a PTX opcode can be
  inspected for the failing f32 rows.

No new independent `FZ-*` is warranted.  This round sharpens the existing
bucket as:

```text
FZ-20260421-0012: M64 f32 `tcgen05.ld.red` destination-layout lowering rejects
non-identity effective row bases as unsupported.  Column-only permutations and
canonical split-N variants are supported, but row-permuted M64 layouts do not
reach a valid hardware-reduction destination layout.
```

## Commands

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Exact failing rows:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]' \
  2>&1 | tee /tmp/tmem_laneaz_row_reverse_n32_min.log

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]' \
  2>&1 | tee /tmp/tmem_laneaz_row_reverse_n32_max.log

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]' \
  2>&1 | tee /tmp/tmem_laneaz_row_rotate_col_even_odd_n128_min.log

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]' \
  2>&1 | tee /tmp/tmem_laneaz_row_rotate_col_even_odd_n128_max.log

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]' \
  2>&1 | tee /tmp/tmem_laneaz_row_reverse_n32_explicit.log

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]' \
  2>&1 | tee /tmp/tmem_laneaz_row_rotate_col_even_odd_n128_explicit.log
```

All six failed with the `unsupported dst layout` diagnostic family.

Passing controls:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[col_reverse_n32-min]' \
  2>&1 | tee /tmp/tmem_laneaz_col_reverse_n32_min.log

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[col_reverse_n32]' \
  2>&1 | tee /tmp/tmem_laneaz_col_reverse_n32_explicit.log
```

Both controls passed.

Explicit variant probe:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon python3 - <<'PY' \
  2>&1 | tee /tmp/tmem_laneaz_variant_probe.log
import torch
import triton.language as tl
from python.test.gluon.test_tmem_runtime_matrix import (
    _make_tmem_linear_layout_m64_permuted,
    _assert_ld_red_runtime_outputs,
    _extract_tcgen05_opcode_offsets,
    tmem_ld_red_m64_explicit_layout_kernel,
)

for row, col, n in [
    ("reverse", "identity", 32),
    ("rotate1", "even_odd", 128),
    ("identity", "reverse", 32),
]:
    for variant in ["auto", "32x32b", "16x32bx2", "32x32b_splitn"]:
        layout = _make_tmem_linear_layout_m64_permuted(n, row, col)
        inp = torch.randn(64, n, dtype=torch.float32, device="cuda")
        out = torch.empty_like(inp)
        red = torch.empty(64, dtype=torch.float32, device="cuda")
        name = f"{row}_{col}_n{n}_{variant}"
        try:
            compiled = tmem_ld_red_m64_explicit_layout_kernel[(1, )](
                inp, out, red, layout, n, variant, "min", False,
                tl.PropagateNan.NONE, num_warps=4)
            _assert_ld_red_runtime_outputs(
                inp, out, red, "min", False, tl.PropagateNan.NONE)
            red_ops = [
                op for op, _ in _extract_tcgen05_opcode_offsets(
                    compiled.asm["ptx"], opcodes=("ld",))
                if ".ld.red." in op
            ]
            print(f"PASS {name} red_ops={red_ops}")
        except Exception as e:
            msg = str(e).replace("\n", " ")[:360]
            print(f"FAIL {name} {type(e).__name__}: {msg}")
PY
```

Non-f32 software-reduce contrast:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon python3 - <<'PY' \
  2>&1 | tee /tmp/tmem_laneaz_int32_detail.log
import torch
import triton.language as tl
from python.test.gluon.test_tmem_runtime_matrix import (
    _make_tmem_linear_layout_m64_permuted,
    _extract_tcgen05_opcode_offsets,
    tmem_ld_red_m64_explicit_layout_kernel,
)

for row, col in [("reverse", "identity"), ("identity", "reverse")]:
    n = 32
    layout = _make_tmem_linear_layout_m64_permuted(n, row, col)
    inp = torch.arange(64 * n, dtype=torch.int32, device="cuda").reshape(64, n) - 1000
    out = torch.empty_like(inp)
    red = torch.empty(64, dtype=torch.int32, device="cuda")
    compiled = tmem_ld_red_m64_explicit_layout_kernel[(1, )](
        inp, out, red, layout, n, "32x32b", "min", False,
        tl.PropagateNan.NONE, num_warps=4)
    out_mismatch = int((out != inp).sum().item())
    red_expected = torch.min(inp, dim=1).values
    red_mismatch = int((red != red_expected).sum().item())
    red_ops = [
        op for op, _ in _extract_tcgen05_opcode_offsets(
            compiled.asm["ptx"], opcodes=("ld",))
        if ".ld.red." in op
    ]
    ld_ops = [
        (op, off) for op, off in _extract_tcgen05_opcode_offsets(
            compiled.asm["ptx"], opcodes=("ld",))
        if ".ld.sync." in op
    ]
    print(row, col, "out_mismatch", out_mismatch,
          "red_mismatch", red_mismatch, "red_ops", red_ops,
          "ld_sample", ld_ops[:4])
PY
```

## Row Table

| Row | Result | Diagnostic / opcode evidence |
| --- | --- | --- |
| `row_reverse_n32-min` | fail | `ttng.tmem_load` parse/lowering error: `Failed to lower TMEM load/store: unsupported dst layout`; out dims `[row (size 128), col (size 32)]`. |
| `row_reverse_n32-max` | fail | Same diagnostic at `tmem.load_max`; out dims `[row (size 128), col (size 32)]`. |
| `row_rotate_col_even_odd_n128-min` | fail | Same diagnostic; out dims `[row (size 128), col (size 128)]`. |
| `row_rotate_col_even_odd_n128-max` | fail | Same diagnostic at `tmem.load_max`; out dims `[row (size 128), col (size 128)]`. |
| `row_reverse_n32` explicit `32x32b` split-N test | fail | Same `ttng.tmem_load` unsupported-destination diagnostic in `tmem_ld_red_m64_explicit_layout_kernel`. |
| `row_rotate_col_even_odd_n128` explicit `32x32b` split-N test | fail | Same `ttng.tmem_load` unsupported-destination diagnostic in the explicit kernel. |
| `col_reverse_n32-min` | pass | Runtime correct. |
| `col_reverse_n32` explicit `32x32b` split-N test | pass | Runtime correct. |

Explicit variant probe:

| Layout | `auto` | `32x32b` | `16x32bx2` | `32x32b_splitn` |
| --- | --- | --- | --- | --- |
| `row=reverse, col=identity, N=32` | fail | fail | fail | fail |
| `row=rotate1, col=even_odd, N=128` | fail | fail | fail | fail |
| `row=identity, col=reverse, N=32` | pass | pass | pass | pass |

The passing column-only controls emitted hardware reductions:

```text
identity_reverse_n32_auto: tcgen05.ld.red.sync.aligned.16x32bx2.x2.min.f32
identity_reverse_n32_32x32b: tcgen05.ld.red.sync.aligned.16x32bx2.x8.min.f32
identity_reverse_n32_16x32bx2: tcgen05.ld.red.sync.aligned.16x32bx2.x2.min.f32
identity_reverse_n32_32x32b_splitn: tcgen05.ld.red.sync.aligned.16x32bx2.x2.min.f32
```

## Diagnostic Classification

The failure arises in frontend/Gluon parsing-time lowering of the
`ttng.tmem_load` reduction operation, when the op asks the TMEM encoding-info
planner for a reduction-compatible destination layout:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
requested layout direct-lowering details:
Failed to lower TMEM load/store: unsupported dst layout
```

This is not a runtime miscompile, not a PTX assembler rejection, and not a
post-TTGIR LLVM-conversion failure for the f32 hardware rows.  The failing rows
do not reach executable code.

The searched backend sites point to the layout-selection path around:

- `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`, where `ttng.tmem_load` computes
  reduction encoding info and emits the high-level diagnostic.
- `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`, where
  `getTMemLdStEncodingInfo` / M64 split-N helpers choose `I16x32bx2` vs
  canonical M64 layouts and eventually emit `unsupported dst layout`.

The observed printed layouts for the failing rows still carry non-identity row
bases in the lane/warp row positions, e.g. `row_reverse_n32` has lane row bases
`64, 32, 8, 4` instead of the canonical M64 split-N row order expected by the
`16x32bx2` hardware-reduction family.  Column-only permutations leave the row
basis canonical enough for the M64 split-N path and pass.

## Fallback / Expected Behavior Notes

- Alternate explicit load layouts are not sufficient: all four requested
  variants failed on the row-permuted M64 rows.
- Canonicalization to split-N is the likely intended repair direction for f32
  hardware `ld.red`, but it must be row-aware.  The current path appears to
  canonicalize or recognize simple M64 split-N only when the effective row
  basis remains canonical/identity.
- A software-reduce fallback is not an acceptable replacement for f32 rows
  that are expected to use `tcgen05.ld.red`; it would also change the opcode
  contract of these tests.  As a contrast, an int32 M64 column-only row compiled
  with no `.ld.red.` opcodes and passed.  The int32 row-reverse contrast
  compiled with no `.ld.red.` opcodes but produced mismatches in this temporary
  probe, so software fallback is not a proven safe rescue for row-permuted M64
  full-output semantics either.

## Hypothesis

`FZ-20260421-0012` is an over-strict or incomplete M64 split-N destination
layout planner for `tcgen05.ld.red`.  The planner can map identity-row M64
queries, including column permutations, onto `16x32bx2` split-N hardware
reduction layouts.  It rejects non-identity effective row bases before it can
derive an equivalent canonical split-N destination layout or a valid
row-permuted packet plan.

The eventual fix should probably live in the shared TMEM layout arithmetic /
encoding-info planner rather than in the tests:

- compute the effective M64 row anchors from the `LinearLayout`;
- canonicalize row-permuted M64 query layouts to a split-N-compatible
  reduction destination when the ISA can realize the same logical rows; and
- keep a clean unsupported diagnostic only when the requested row basis cannot
  be represented by `tcgen05.ld.red` packets plus the required post-reduction
  combine.

No backend/compiler code was changed.
