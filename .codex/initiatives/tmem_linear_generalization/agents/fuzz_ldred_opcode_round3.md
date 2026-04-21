# Lane R3-B round 3: ld.red opcode loss expansion

- Time: 2026-04-21 08:39 UTC
- Lane: R3-B
- Branch/HEAD: `codex/tmem` at `78f2845a0`
- Scope: expand and diagnose `FZ-20260421-0004` ld.red descriptor-chain
  plain-load fallback/opcode loss, including resource-valid 2CTA rows and
  nearby descriptor-view/layout shapes.
- Backend repair status: no backend/compiler code changed.

## Artifacts

- Temporary probe: `/tmp/tmem_ldred_opcode_round3_probe.py`
- Split logs:
  - `/tmp/tmem_ldred_opcode_round3_g1.log`
  - `/tmp/tmem_ldred_opcode_round3_g2.log`
  - `/tmp/tmem_ldred_opcode_round3_g3.log`
  - `/tmp/tmem_ldred_opcode_round3_g4.log`
- Fresh confirmation logs:
  - `/tmp/tmem_ldred_opcode_round3_confirm_onecta_chain0_positive.log`
  - `/tmp/tmem_ldred_opcode_round3_confirm_onecta_chain1.log`
  - `/tmp/tmem_ldred_opcode_round3_confirm_onecta_128x64_chain1_positive.log`
  - `/tmp/tmem_ldred_opcode_round3_confirm_twocta_index_chain0.log`
  - `/tmp/tmem_ldred_opcode_round3_confirm_twocta_full_parent_positive.log`
  - `/tmp/tmem_ldred_opcode_round3_checked_fz0004.log`
  - `/tmp/tmem_ldred_opcode_round3_checked_twocta_positive.log`

## Commands

Required rebuild before pytest:

```bash
make -j8
```

Result: `ninja: no work to do`.

Probe syntax and collection:

```bash
PYTHONPATH=.:./python python -m py_compile /tmp/tmem_ldred_opcode_round3_probe.py
PYTHONPATH=.:./python pytest --collect-only -q /tmp/tmem_ldred_opcode_round3_probe.py
```

Result: `19 tests collected`.

Four-GPU split sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r3b-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 /tmp/tmem_ldred_opcode_round3_probe.py 2>&1 | tee /tmp/tmem_ldred_opcode_round3_g1.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r3b-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 /tmp/tmem_ldred_opcode_round3_probe.py 2>&1 | tee /tmp/tmem_ldred_opcode_round3_g2.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r3b-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 /tmp/tmem_ldred_opcode_round3_probe.py 2>&1 | tee /tmp/tmem_ldred_opcode_round3_g3.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r3b-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 /tmp/tmem_ldred_opcode_round3_probe.py 2>&1 | tee /tmp/tmem_ldred_opcode_round3_g4.log
```

Fresh single-node confirmations:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r3b-confirm-c1 PYTHONPATH=.:./python pytest -s --tb=short '/tmp/tmem_ldred_opcode_round3_probe.py::test_onecta_ldred_opcode_round3[parent_index-64-32-identity-identity-1-min-False-PROPAGATE_NAN.NONE]' 2>&1 | tee /tmp/tmem_ldred_opcode_round3_confirm_onecta_chain1.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r3b-confirm-c0 PYTHONPATH=.:./python pytest -s --tb=short '/tmp/tmem_ldred_opcode_round3_probe.py::test_onecta_ldred_opcode_round3[parent_index-64-32-identity-identity-0-min-False-PROPAGATE_NAN.NONE]' 2>&1 | tee /tmp/tmem_ldred_opcode_round3_confirm_onecta_chain0_positive.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r3b-confirm-twocta-index PYTHONPATH=.:./python pytest -s --tb=short '/tmp/tmem_ldred_opcode_round3_probe.py::test_twocta_indexed_ldred_opcode_round3[256-32-0]' 2>&1 | tee /tmp/tmem_ldred_opcode_round3_confirm_twocta_index_chain0.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r3b-confirm-twocta-parent PYTHONPATH=.:./python pytest -s --tb=short '/tmp/tmem_ldred_opcode_round3_probe.py::test_twocta_full_parent_ldred_positive_round3' 2>&1 | tee /tmp/tmem_ldred_opcode_round3_confirm_twocta_full_parent_positive.log
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r3b-checked-fz0004 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-chain1-64x32-min]' 2>&1 | tee /tmp/tmem_ldred_opcode_round3_checked_fz0004.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r3b-checked-twocta PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-twocta-lifted-256x64]' 2>&1 | tee /tmp/tmem_ldred_opcode_round3_checked_twocta_positive.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r3b-confirm-128x64 PYTHONPATH=.:./python pytest -s --tb=short '/tmp/tmem_ldred_opcode_round3_probe.py::test_onecta_ldred_opcode_round3[parent_index-128-64-identity-identity-1-min-False-PROPAGATE_NAN.NONE]' 2>&1 | tee /tmp/tmem_ldred_opcode_round3_confirm_onecta_128x64_chain1_positive.log
```

## Pass/Fail Matrix

| Row | Shape / source | View chain | Result | Opcode evidence |
| --- | --- | --- | --- | --- |
| 1CTA parent index | `[2,64,32] -> index(1)` | chain0 direct indexed view | pass | `tcgen05.ld.red.sync.aligned.16x32bx2.x16.min.f32` |
| 1CTA descriptor chain | `[2,64,32] -> index(1)` | chain1 reshape/permute/reshape | fail, opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x16.b32` |
| 1CTA descriptor chain | `[2,64,32] -> index(1)` | chain2 double col reshape/permute | fail, opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x16.b32` |
| 1CTA descriptor chain | `[2,64,32] -> index(1)` | chain4 nested row reshapes/permutes | fail, opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x16.b32` |
| 1CTA descriptor chain | `[2,64,64]`, col reverse | chain2 | fail, opcode mismatch | repeated `tcgen05.ld.sync.aligned.16x32bx2.x2.b32` |
| 1CTA descriptor chain | `[2,64,64]`, row rotate1 | chain2 | fail, opcode mismatch | repeated `tcgen05.ld.sync.aligned.32x32b.x1.b32` |
| 1CTA descriptor chain | `[2,64,128]` | chain1 max | fail, opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x64.b32` |
| 1CTA descriptor chain | `[2,64,64]` | chain1 min abs | fail, opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x32.b32` |
| 1CTA descriptor chain | `[2,64,64]` | chain1 max NaN-propagating | fail, opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x32.b32` |
| 1CTA descriptor chain | `[2,128,64]` | chain1 | pass | `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32` |
| 2CTA full parent | `[2,256,64]` full parent load_min | no indexed view | pass | `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32` |
| 2CTA indexed parent | `[2,256,32] -> index(1)` | chain0 direct indexed view | fail, opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x16.b32` |
| 2CTA indexed parent | `[2,256,32] -> index(1)` | chain1 / chain2 | fail, opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x16.b32` |
| 2CTA indexed parent | `[2,256,64] -> index(1)` | chain0 / chain1 / chain2 | fail, opcode mismatch | `tcgen05.ld.sync.aligned.16x32bx2.x32.b32` |

All opcode-mismatch rows completed the runtime oracle first: output tensors
matched the input and reduced tensors matched PyTorch min/max with `equal_nan`
where applicable. PTX and LLIR opcode extraction matched in each row. The
failing assertion was only the missing `.ld.red.` opcode.

Two probe rows are not counted as `FZ-20260421-0004` failures:

- direct allocation `[64,32]` with auto layout rejected at
  `view.get_reg_layout()` with the existing row-anchor diagnostic;
- chain3 `[2,64,32] -> index(1).permute(...).slice(...)` rejected with the
  same row-anchor diagnostic.

Those are diagnostic/boundary rows, not plain-load opcode-loss rows.

## Minimized Stable Failures

### R3B-LDRED-0004-A: 1CTA descriptor-chain chain1 opcode loss

- Catalog bucket: extends existing `FZ-20260421-0004`.
- Seed: `0xB300 + 64 * 3 + 32 * 5 + 1 * 11`.
- Shape: parent `[2,64,32]`, selected view `[64,32]`.
- Layout: identity `TensorMemoryLinearLayout`.
- Chain: `index(1).reshape((M//2,2,N)).permute([1,0,2]).reshape((M,N))`.
- Operation: `load_min`, `num_warps=4`, `num_ctas=1`.
- Runtime oracle: output and reduced tensor pass.
- Opcode: `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`; no `.ld.red.`.
- Fresh command: `/tmp/tmem_ldred_opcode_round3_confirm_onecta_chain1.log`.
- Stability: reproduced in the split sweep and fresh single-node process.
- Checked-in coverage: already present as strict xfail
  `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-chain1-64x32-min]`,
  confirmed as `1 xfailed`.

### R3B-LDRED-0004-B: resource-valid 2CTA indexed parent opcode loss

- Catalog bucket: extends existing `FZ-20260421-0004`.
- Seed: `0xB400 + 32 * 7 + 0`.
- Shape: two-CTA parent `[2,256,32]`, selected view `[256,32]`.
- Layout: `TensorMemoryLinearLayout(..., block_bases=[[128,0]], two_ctas=True)`,
  lifted through prefix `[2]`.
- Chain: direct `index(1)` with no reshape/permutation.
- Operation: `load_min`, `num_warps=8`, `num_ctas=2`.
- Runtime oracle: output and reduced tensor pass.
- Opcode: `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`; no `.ld.red.`.
- Fresh command: `/tmp/tmem_ldred_opcode_round3_confirm_twocta_index_chain0.log`.
- Stability: reproduced in the split sweep and fresh single-node process.
- Contrast row: full-parent 2CTA `[2,256,64]` load_min still passes and emits
  `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32`, both in the temporary
  probe and in the checked-in `ldred-twocta-lifted-256x64` positive.

## Diagnosis

The new boundary isolates the bug more tightly than Round 2:

- 1CTA `parent.index(1)` without an additional descriptor-view chain can still
  select hardware `ld.red`.
- 1CTA descriptor-view chains over that indexed view frequently lose the
  reduction opcode and lower as plain TMEM loads plus software reduction.
- 2CTA full-parent reduction still selects `ld.red`, so 2CTA support itself is
  not globally broken.
- 2CTA `parent.index(1)` loses `ld.red` even at chain0, before any reshape or
  permutation. That points at descriptor/view provenance or layout support
  after selecting from a resource-valid 2CTA parent, not only at complex view
  composition.
- The `[128,64]` 1CTA chain1 positive emits `ld.red`, while `[64,32]`,
  `[64,64]`, and `[64,128]` variants fall back. The failure is shape/layout
  packet-family dependent, not a blanket ban on all descriptor chains.

Likely owner surface remains ld.red opcode selection for TMEM descriptor views:
the lowering preserves correct values but loses the hardware reduction form
when the selected view requires certain packet shapes or when a 2CTA parent is
indexed down to a 2D descriptor.

## Recommended Checked-In Strict Xfails

Existing checked-in coverage for `FZ-20260421-0004` is still valid and should
stay:

- `ldred-fz20260421-0004-chain1-64x32-min`

Recommended additional strict xfail:

- `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min`
  - This is the smallest stable 2CTA resource-valid row found in this lane.
  - It separates the failure from full-parent 2CTA support because the
    full-parent positive emits `ld.red`.
  - It does not require adding backend fixes or relying on clean-error
    diagnostics.

Optional broader checked-in coverage, if one more shape-family sentinel is
desired:

- `ldred-fz20260421-0004-chain2-64x64-rotate1-min`
  - This captures the repeated `32x32b.x1` plain-load family and exercises a
    non-identity row layout.
  - It is less minimal than the existing chain1 xfail and should be secondary
    to the 2CTA indexed sentinel.
