# Lane R4-D round 4: 2CTA indexed ld.red opcode selection

- Time: 2026-04-21 UTC
- Lane: R4-D
- Branch/HEAD: `codex/tmem` at `98fde8a79`
- Scope: fuzz `ld.red` opcode selection around 2CTA indexed-view
  provenance, non-min reductions, abs/NaN modifiers, row/col layout
  permutations, descriptor chain0/1/2, and resource/clean diagnostic
  boundaries.
- Backend repair status: no backend/compiler code changed.

## Artifacts

- Temporary probe: `/tmp/tmem_ldred_2cta_round4_probe.py`
- Split logs:
  - `/tmp/tmem_ldred_2cta_round4_g1.log`
  - `/tmp/tmem_ldred_2cta_round4_g2.log`
  - `/tmp/tmem_ldred_2cta_round4_g3.log`
  - `/tmp/tmem_ldred_2cta_round4_g4.log`
- Fresh confirmation logs:
  - `/tmp/tmem_ldred_2cta_round4_confirm_min.log`
  - `/tmp/tmem_ldred_2cta_round4_confirm_max.log`
  - `/tmp/tmem_ldred_2cta_round4_confirm_abs.log`
  - `/tmp/tmem_ldred_2cta_round4_confirm_nan.log`
  - `/tmp/tmem_ldred_2cta_round4_confirm_chain2_n128.log`
  - `/tmp/tmem_ldred_2cta_round4_confirm_rotate1.log`
  - `/tmp/tmem_ldred_2cta_round4_confirm_evenodd_crash.log`
  - `/tmp/tmem_ldred_2cta_round4_confirm_boundary512.log`
  - `/tmp/tmem_ldred_2cta_round4_existing_twocta_xfail.log`
  - `/tmp/tmem_ldred_2cta_round4_existing_full_parent_positive.log`

## Commands

Required rebuild before pytest:

```bash
make -j8
```

Result: `ninja: no work to do`.

Probe syntax and collection:

```bash
PYTHONPATH=.:./python python -m py_compile /tmp/tmem_ldred_2cta_round4_probe.py
PYTHONPATH=.:./python pytest --collect-only -q /tmp/tmem_ldred_2cta_round4_probe.py
```

Result: `42 tests collected`.

Four-GPU split sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 /tmp/tmem_ldred_2cta_round4_probe.py 2>&1 | tee /tmp/tmem_ldred_2cta_round4_g1.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 /tmp/tmem_ldred_2cta_round4_probe.py 2>&1 | tee /tmp/tmem_ldred_2cta_round4_g2.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 /tmp/tmem_ldred_2cta_round4_probe.py 2>&1 | tee /tmp/tmem_ldred_2cta_round4_g3.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 /tmp/tmem_ldred_2cta_round4_probe.py 2>&1 | tee /tmp/tmem_ldred_2cta_round4_g4.log
```

Results: group 1 `11 failed`; group 2 `11 failed`; group 3 `11 failed`;
group 4 `9 failed`. All failures were expected discovery failures from this
probe's assertion that hardware `ld.red` should be selected after runtime
correctness succeeds, except the explicitly noted compiler-crash and resource
diagnostic rows below.

Fresh single-node confirmations:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-confirm-min PYTHONPATH=.:./python pytest -s --tb=short '/tmp/tmem_ldred_2cta_round4_probe.py::test_ldred_2cta_indexed_opcode_round4[2cta-idx-256x32-c0-identity-identity-min-plain-nonan-ldred]' 2>&1 | tee /tmp/tmem_ldred_2cta_round4_confirm_min.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-confirm-max PYTHONPATH=.:./python pytest -s --tb=short '/tmp/tmem_ldred_2cta_round4_probe.py::test_ldred_2cta_indexed_opcode_round4[2cta-idx-256x32-c0-identity-identity-max-plain-nonan-ldred]' 2>&1 | tee /tmp/tmem_ldred_2cta_round4_confirm_max.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-confirm-abs PYTHONPATH=.:./python pytest -s --tb=short '/tmp/tmem_ldred_2cta_round4_probe.py::test_ldred_2cta_indexed_opcode_round4[2cta-idx-256x32-c0-identity-identity-min-abs-nonan-ldred]' 2>&1 | tee /tmp/tmem_ldred_2cta_round4_confirm_abs.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-confirm-nan PYTHONPATH=.:./python pytest -s --tb=short '/tmp/tmem_ldred_2cta_round4_probe.py::test_ldred_2cta_indexed_opcode_round4[2cta-idx-256x32-c0-identity-identity-min-plain-nan-ldred]' 2>&1 | tee /tmp/tmem_ldred_2cta_round4_confirm_nan.log
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-confirm-chain2 PYTHONPATH=.:./python pytest -s --tb=short '/tmp/tmem_ldred_2cta_round4_probe.py::test_ldred_2cta_indexed_opcode_round4[2cta-idx-256x128-c2-identity-identity-min-plain-nonan-ldred]' 2>&1 | tee /tmp/tmem_ldred_2cta_round4_confirm_chain2_n128.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-confirm-layout PYTHONPATH=.:./python pytest -s --tb=short '/tmp/tmem_ldred_2cta_round4_probe.py::test_ldred_2cta_indexed_opcode_round4[2cta-idx-256x64-c2-rotate1-identity-min-plain-nonan-ldred]' 2>&1 | tee /tmp/tmem_ldred_2cta_round4_confirm_rotate1.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-confirm-crash PYTHONPATH=.:./python pytest -s --tb=short '/tmp/tmem_ldred_2cta_round4_probe.py::test_ldred_2cta_indexed_opcode_round4[2cta-idx-256x32-c1-even_odd-identity-min-plain-nonan-ldred]' 2>&1 | tee /tmp/tmem_ldred_2cta_round4_confirm_evenodd_crash.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-confirm-boundary512 PYTHONPATH=.:./python pytest -s --tb=short '/tmp/tmem_ldred_2cta_round4_probe.py::test_ldred_2cta_indexed_resource_boundary_round4[boundary-2cta-idx-512x32-clean-or-ldred]' 2>&1 | tee /tmp/tmem_ldred_2cta_round4_confirm_boundary512.log
```

Existing structural-fuzzer sentinels:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-existing-xfail PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min]' 2>&1 | tee /tmp/tmem_ldred_2cta_round4_existing_twocta_xfail.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r4d-existing-positive PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-twocta-lifted-256x64]' 2>&1 | tee /tmp/tmem_ldred_2cta_round4_existing_full_parent_positive.log
```

Results: existing 2CTA indexed sentinel reported `1 xfailed`; full-parent
2CTA positive reported `1 passed`.

## Pass/Fail Matrix

All rows that reached PTX/LLIR first passed the runtime oracle: the full output
matched the input, the reduced output matched PyTorch `min` or `max` over the
row dimension, and `equal_nan=True` was used for NaN-propagating cases. PTX and
LLIR opcode extraction matched before classification.

| Case family | Rows | Result | Opcode / diagnostic |
| --- | ---: | --- | --- |
| 2CTA indexed, identity layout, `M=256`, `N=32`, chain0/1/2, min/max | 6 | runtime-correct opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x16.b32` |
| 2CTA indexed, identity layout, `M=256`, `N=64`, chain0/1/2, min/max | 6 | runtime-correct opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x32.b32` |
| 2CTA indexed, identity layout, `M=256`, `N=128`, chain0/1/2, min/max | 6 | runtime-correct opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x64.b32` |
| 2CTA indexed row reverse, `256x32`, chain0, min/max | 2 | runtime-correct opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x16.b32` |
| 2CTA indexed row rotate1, `256x64`, chain2, min/max | 2 | runtime-correct opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x32.b32` |
| 2CTA indexed col rotate1, `256x128`, chain2, min/max | 2 | runtime-correct opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x64.b32` |
| 2CTA indexed identity, abs modifier cases over `N=32/64/128` | 6 | runtime-correct opcode mismatch | plain `tcgen05.ld.sync...`; no `.ld.red.`, so no `.abs` hardware modifier |
| 2CTA indexed identity, NaN-propagating cases over `N=32/64/128` | 6 | runtime-correct opcode mismatch | plain `tcgen05.ld.sync...`; no `.ld.red.`, so no `.NaN` hardware modifier |
| Boundary `M=128,N=32`, chain0 | 1 | runtime-correct opcode mismatch | `tcgen05.ld.sync.aligned.32x32b.x8.b32` |
| Row even_odd `256x32`, chain1, min/max | 2 | compiler crash | `TritonNvidiaGPUOptimizeTMemLayoutsPass`: dims `["row","col"]` vs `["row","col","block"]` |
| Col reverse `256x64`, chain1, min/max | 2 | compiler crash | same optimizer failure class |
| Boundary `M=512,N=32`, chain0 | 1 | clean diagnostic boundary | `ttng.tmem_load` failed to compute TMEM encoding info; CGA layout mismatch |

## Minimized Stable Findings

### R4D-LDRED-0004-C: non-min 2CTA indexed opcode loss

- Catalog bucket: extends `FZ-20260421-0004`.
- Seed formula in probe: `0xD400 + M + N + chain_id * 17`; representative
  seed `0xD520`.
- Shape: parent `[2,256,32]`, selected view `[256,32]`.
- Layout: identity two-CTA `TensorMemoryLinearLayout`, lifted through prefix
  `[2]`.
- Chain: direct `parent.index(1)`.
- Operation: `load_max`, `num_warps=8`, `num_ctas=2`.
- Runtime oracle: passed before opcode assertion.
- Opcode: `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`; no `.ld.red.`.
- Fresh command log: `/tmp/tmem_ldred_2cta_round4_confirm_max.log`.
- Stability: reproduced in split sweep and fresh single-node process.

### R4D-LDRED-0004-D: abs modifier 2CTA indexed opcode loss

- Catalog bucket: extends `FZ-20260421-0004`.
- Shape: parent `[2,256,32]`, selected view `[256,32]`.
- Chain: direct `parent.index(1)`.
- Operation: `load_min(abs=True)`.
- Runtime oracle: passed against `torch.min(torch.abs(inp), dim=1).values`.
- Opcode: `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`; no `.ld.red.`, so the
  hardware `.abs` modifier is absent.
- Fresh command log: `/tmp/tmem_ldred_2cta_round4_confirm_abs.log`.
- Stability: reproduced in split sweep and fresh single-node process.

### R4D-LDRED-0004-E: NaN-propagating 2CTA indexed opcode loss

- Catalog bucket: extends `FZ-20260421-0004`.
- Shape: parent `[2,256,32]`, selected view `[256,32]`.
- Chain: direct `parent.index(1)`.
- Operation: `load_min(propagate_nan=tl.PropagateNan.ALL)`.
- Runtime oracle: passed with seeded NaNs and `equal_nan=True`.
- Opcode: `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`; no `.ld.red.`, so the
  hardware `.NaN` modifier is absent.
- Fresh command log: `/tmp/tmem_ldred_2cta_round4_confirm_nan.log`.
- Stability: reproduced in split sweep and fresh single-node process.

### R4D-LDRED-0004-F: broader packet/layout opcode loss

- Catalog bucket: extends `FZ-20260421-0004`.
- Representative stable rows:
  - `2cta-idx-256x128-c2-identity-identity-min-plain-nonan-ldred` emitted
    `tcgen05.ld.sync.aligned.16x32bx2.x64.b32`;
  - `2cta-idx-256x64-c2-rotate1-identity-min-plain-nonan-ldred` emitted
    `tcgen05.ld.sync.aligned.16x32bx2.x32.b32`.
- Runtime oracle: passed before opcode assertion.
- Fresh command logs:
  - `/tmp/tmem_ldred_2cta_round4_confirm_chain2_n128.log`;
  - `/tmp/tmem_ldred_2cta_round4_confirm_rotate1.log`.
- Stability: reproduced in split sweep and fresh single-node processes.

### R4D-LDRED-CRASH-001: row/col permutation plus chain1 optimizer crash

- Failure class: compiler crash, not an opcode mismatch.
- Representative row: `2cta-idx-256x32-c1-even_odd-identity-min-plain-nonan-ldred`.
- Shape: parent `[2,256,32]`, selected view `[256,32]`.
- Chain: `index(1).reshape((128,2,32)).permute([1,0,2]).reshape((256,32))`.
- Layout: row `even_odd`, col `identity`.
- Failure: `TritonNvidiaGPUOptimizeTMemLayoutsPass` fails with dimensions
  mismatch, reporting `["row","col"]` vs `["row","col","block"]`.
- Fresh command log: `/tmp/tmem_ldred_2cta_round4_confirm_evenodd_crash.log`.
- Related split rows: `identity/reverse` col layout at `256x64` chain1 also
  fails with the same owner surface.

### R4D-LDRED-BOUNDARY-001: larger resource clean diagnostic

- Boundary row: `boundary-2cta-idx-512x32-clean-or-ldred`.
- Failure class: clean diagnostic boundary, not an opcode mismatch.
- Diagnostic: `ttng.tmem_load` failed to compute TMEM encoding info for
  reduction because register and memory CGA layouts differ.
- Fresh command log: `/tmp/tmem_ldred_2cta_round4_confirm_boundary512.log`.

## Diagnosis

Round 4 broadens the R3-B boundary without changing the likely owner surface.
Full-parent 2CTA reduction still passes in the checked-in structural fuzzer,
but indexing a resource-valid 2CTA parent down to a 2D descriptor loses
hardware `ld.red` before any chain composition. The loss is not specific to
`min`: direct 2CTA indexed `max`, `abs`, and NaN-propagating reductions are
runtime-correct but lower as plain TMEM loads plus software reductions. The
same fallback holds across `N=32/64/128`, chain0/1/2, and several row/col
permutations.

The row/col permutation chain1 crashes are adjacent but separate from the
opcode-loss bucket: they fail in TMEM layout optimization before runtime or
PTX/LLIR opcode inspection, with a block-dimension mismatch after view-chain
composition. The `M=512` row reports a direct TMEM encoding/CGA-layout
diagnostic and should be treated as a resource boundary unless a later ISA
audit says the shape should be realizable.

## Recommended Checked-In Strict Xfails

Existing checked-in coverage remains valid and should stay:

- `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min`

Recommended additional strict xfails, in priority order:

- `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-max`
  - Smallest stable non-min sentinel.
  - Separates the bug from min-only opcode selection.
- `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min-abs`
  - Captures missing hardware modifier selection after runtime correctness.
- `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min-nan`
  - Captures missing NaN-propagating hardware modifier selection.

Optional broader sentinels if more coverage is wanted:

- `ldred-fz20260421-0004-twocta-indexed-256x128-chain2-min`
  - Captures the `x64` packet family and a chain2 descriptor view.
- `ldred-fz20260421-0004-twocta-indexed-256x64-chain2-rotate1-min`
  - Captures a non-identity row layout that is runtime-correct but still
    emits plain load.
- `ldred-fz20260421-crash-twocta-indexed-256x32-chain1-even_odd-min`
  - Separate compiler-crash sentinel for the optimizer dimensions mismatch;
    do not merge this with opcode-loss xfails.
