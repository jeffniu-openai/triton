# Round 13 Lane Z: copy, scales, subword, and packed-lane edge fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Repo edit scope: this report plus initiative documentation.
- Temporary probes: inline Python subprocesses only; no temporary repo code.

## Scope

This lane extended Lane V's copy descriptor/addressing coverage without
rerunning the same exact row set. The focus was clean-boundary precision and
neighboring positive surfaces around:

- no-scales `tcgen05.copy` linear 32-bit dtype shapes;
- tile-permuted and tile-selector-permuted destination layouts;
- row/column-permuted destination clean diagnostics;
- mixed/exotic linear-layout clean diagnostics;
- `4x256b` refresh-shaped destinations;
- exact-width subword `128x128b` positives;
- legacy/unpacked subword clean diagnostics; and
- 2CTA local copy in 4/8/16 CTA launch contexts for high-CGA contrast.

The lane intentionally avoided Lane V's exact `warpx2` indexed/subslice,
scales descriptor-view, and `cp_scales` selector rows except where a high-CGA
contrast was needed to confirm the existing `FZ-20260421-0010` owner.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Checked-in selector collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_linear_tile_permuted or cp_no_scales_linear_tile_selector_permuted or cp_no_scales_linear_rowcol_permuted or cp_no_scales_linear_exotic or cp_no_scales_4x256b or cp_128x128_subword_exact_width or cp_no_scales_legacy_subword or cp_no_scales_linear_32bit_dtypes'
```

Result: `69/1615` tests collected.

Split-4 runtime command:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_lane_z_copy_subword_round13_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_linear_tile_permuted or cp_no_scales_linear_tile_selector_permuted or cp_no_scales_linear_rowcol_permuted or cp_no_scales_linear_exotic or cp_no_scales_4x256b or cp_128x128_subword_exact_width or cp_no_scales_legacy_subword or cp_no_scales_linear_32bit_dtypes'
```

Results:

```text
group 1/GPU 0: 18 passed, 1597 deselected in 9.64s
group 2/GPU 1: 18 passed, 1597 deselected in 8.46s
group 3/GPU 2: 18 passed, 1597 deselected in 11.38s
group 4/GPU 3: 15 passed, 1600 deselected in 4.51s
aggregate:     69 passed
```

Representative opcode probe:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon python - <<'PY'
import torch
from test_tmem_runtime_matrix import (
    _extract_tcgen05_cp_opcodes,
    _make_tmem_copy_4x256b_refresh_layout,
    _make_tmem_copy_dense_shared_layout,
    _make_tmem_linear_layout,
    _make_tmem_linear_layout_tile_permuted,
    tmem_copy_128x128_subword_exact_kernel,
    tmem_copy_no_scales_4x256b_refresh_kernel,
    tmem_copy_no_scales_linear_kernel,
)

def ops(compiled):
    return _extract_tcgen05_cp_opcodes(compiled.asm["ptx"]), _extract_tcgen05_cp_opcodes(compiled.asm["llir"])

m, n = 128, 64
inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
out = torch.empty_like(inp)
compiled = tmem_copy_no_scales_linear_kernel[(1, )](
    inp, out, _make_tmem_linear_layout_tile_permuted(m, n, 4), m, n, 32, num_warps=4
)
torch.testing.assert_close(out, inp, atol=0, rtol=0)
print("tile_permuted_128x64_tile4", ops(compiled))

m, n = 4, 8
inp = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
out = torch.empty_like(inp)
compiled = tmem_copy_no_scales_4x256b_refresh_kernel[(1, )](
    inp, out, _make_tmem_copy_4x256b_refresh_layout(), num_warps=4
)
torch.testing.assert_close(out, inp, atol=0, rtol=0)
print("refresh_4x256b", ops(compiled))

m, n = 128, 8
inp = torch.arange(m * n, device="cuda", dtype=torch.int32).reshape(m, n).to(torch.float16)
out = torch.empty_like(inp)
compiled = tmem_copy_128x128_subword_exact_kernel[(1, )](
    inp, out, n, _make_tmem_linear_layout(m, n),
    _make_tmem_copy_dense_shared_layout(m, n), num_warps=4
)
torch.testing.assert_close(out, inp, atol=0, rtol=0)
print("subword_exact_f16_128x8", ops(compiled))
PY
```

Representative high-CGA contrast:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon python - <<'PY'
import torch
from triton.compiler.errors import CompilationError
from test_tmem_runtime_matrix import (
    _make_2cta_cga_layout,
    _make_tmem_linear_layout_mmav5_twocta,
    tmem_copy_no_scales_twocta_kernel,
)

for ctas in (4, 8, 16):
    M, N, swizzle = 256, 64, 32
    cga_layout = _make_2cta_cga_layout((ctas, 1), (ctas, 1), (1, 0), 0)
    layout = _make_tmem_linear_layout_mmav5_twocta(M, N)
    inp = torch.arange(M * N, device="cuda", dtype=torch.float32).reshape(M, N)
    out = torch.empty_like(inp)
    try:
        tmem_copy_no_scales_twocta_kernel[(1, )](
            inp, out, layout, tuple(tuple(basis) for basis in cga_layout),
            M, N, swizzle, num_ctas=ctas, num_warps=4,
        )
        print(f"high_cga_{ctas}: UNEXPECTED_PASS")
    except CompilationError as e:
        text = str(e)
        line = next((ln for ln in text.splitlines() if "Layout has" in ln), text.splitlines()[-1])
        print(f"high_cga_{ctas}: {line}")
PY
```

Result:

```text
high_cga_4: Layout has 2 CTAs per CGA, but the context requires 4 CTAs per CGA.
high_cga_8: Layout has 2 CTAs per CGA, but the context requires 8 CTAs per CGA.
high_cga_16: Layout has 2 CTAs per CGA, but the context requires 16 CTAs per CGA.
```

## Classification

No new independent `FZ-*` candidate is warranted.

Final classification:

```text
pass / asserted clean boundary                 69
FZ-20260421-0010 high-CGA CTA-count gate        3
runtime miscompile                              0
opcode mismatch                                 0
compiler crash                                  0
false unsupported candidate                     0
```

The three high-CGA rows are not counted in the checked-in 69-row selector; they
are a focused contrast probe and map to the existing `FZ-20260421-0010`
CTA-count gate.

## Positive opcode evidence

Representative positives had matching PTX and LLIR opcode extraction:

```text
tile_permuted_128x64_tile4:
  PTX/LLIR: tcgen05.cp.cta_group::1.128x128b x16

refresh_4x256b:
  PTX/LLIR: tcgen05.cp.cta_group::1.4x256b x2

subword_exact_f16_128x8:
  PTX/LLIR: tcgen05.cp.cta_group::1.128x128b x1
```

The split-4 selector also covered larger tile-permuted `128x256b` positives,
tile-selector permutations, 32-bit `f32`/`i32` shape variants, and exact-width
`i8` subword `128x128b`.

## Clean-boundary precision

The checked-in clean-boundary rows all passed their diagnostic assertions:

- `4x256b` ordinary contiguous view rejects with a refresh-shaped destination
  explanation and no `PassManager::run failed` or assertion text.
- Legacy/unpacked subword copy rejects with the packed-lane source-storage
  requirement and no late lowering crash.
- Mixed/exotic linear layouts reject before LLVM lowering; the mixed row
  asserts the row/column contribution diagnostic.
- Subinstruction tile permutations reject with destination-column permutation
  and full-copy-footprint diagnostics.
- Row/column-permuted destinations reject with row-order, column-order, or
  alignment diagnostics; row-permuted cases assert the source-row projection
  schedule details.

These results support the current copy-family boundary classification: the
positive rows lower to exact `tcgen05.cp` forms, and the unsupported rows fail
with typed planner diagnostics instead of compiler crashes or silent
miscompiles.
