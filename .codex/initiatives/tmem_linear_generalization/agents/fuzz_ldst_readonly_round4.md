# Lane R4-C: ld/st read-only descriptor-view fuzzing

- Date: 2026-04-21
- Branch/HEAD: `codex/tmem` / `98fde8a79`
- Mode: discovery/classification only; no backend/compiler repair.
- Temporary probe: `/tmp/tmem_ldst_readonly_round4_probe.py`

## Scope

Adversarially expanded `FZ-20260421-0003` around read-only `tcgen05.ld/st`
descriptor-view loads beyond the existing f32 chain1/chain2 sentinels:

- dtype/subword variants: `f32`, `i32`, `f16`, and legal `f8` smoke rows;
- packet variants: `32x32b`, `16x64b`, `16x128b`, `16x256b`;
- M/N boundaries: `64x32`, `64x64`, `64x128`, `128x32`, `128x64`;
- lifted/indexed two-CTA descriptor views over `[2,M,N]` parents;
- same-view roundtrip controls that can mask read-only address bugs.

Probe chain ids:

- `c0`: direct `parent.index(1)` control.
- `c1`: `reshape((M//2,2,N)).permute([1,0,2]).reshape((M,N))`.
- `c2`: `reshape((M,N//2,2)).permute([0,2,1]).permute([0,2,1]).reshape((M,N))`.
- `c3`: double transpose plus slices.
- `c4`: 4D row/col pair permutation.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Collect:

```bash
PYTHONPATH=.:./python pytest --collect-only -q /tmp/tmem_ldst_readonly_round4_probe.py
```

Result after adding f8 smoke rows: `21 tests collected`.

Four-GPU sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r4c2-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 /tmp/tmem_ldst_readonly_round4_probe.py 2>&1 | tee /tmp/tmem_ldst_readonly_round4_fixed_g1.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r4c2-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_ldst_readonly_round4_probe.py 2>&1 | tee /tmp/tmem_ldst_readonly_round4_fixed_g2.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r4c2-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 /tmp/tmem_ldst_readonly_round4_probe.py 2>&1 | tee /tmp/tmem_ldst_readonly_round4_fixed_g3.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r4c2-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 /tmp/tmem_ldst_readonly_round4_probe.py 2>&1 | tee /tmp/tmem_ldst_readonly_round4_fixed_g4.log
```

Raw split results from the 19-row fixed sweep before f8 smoke rows were added:

- group 1: `4 failed, 1 passed, 14 deselected`
- group 2: `5 failed, 14 deselected`
- group 3: `4 failed, 1 passed, 14 deselected`
- group 4: `2 failed, 2 passed, 15 deselected`

F8 and fresh representative confirmations:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r4c-confirm-f32 PYTHONPATH=.:./python pytest -q -s --tb=short '/tmp/tmem_ldst_readonly_round4_probe.py::test_ldst_readonly_round4[r4-f32-c1-64x32-32x32b]' 2>&1 | tee /tmp/tmem_ldst_readonly_round4_confirm_f32_c1.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r4c-confirm-i32 PYTHONPATH=.:./python pytest -q -s --tb=short '/tmp/tmem_ldst_readonly_round4_probe.py::test_ldst_readonly_round4[r4-i32-c1-64x32-32x32b]' 2>&1 | tee /tmp/tmem_ldst_readonly_round4_confirm_i32_c1.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r4c-confirm-f16c2 PYTHONPATH=.:./python pytest -q -s --tb=short '/tmp/tmem_ldst_readonly_round4_probe.py::test_ldst_readonly_round4[r4-f16-c2-64x32-16x64b]' 2>&1 | tee /tmp/tmem_ldst_readonly_round4_confirm_f16_c2.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r4c-confirm-coli32 PYTHONPATH=.:./python pytest -q -s --tb=short '/tmp/tmem_ldst_readonly_round4_probe.py::test_ldst_readonly_round4[r4-colrev-i32-c2-64x32-16x64b]' 2>&1 | tee /tmp/tmem_ldst_readonly_round4_confirm_i32_colrev_c2.log
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r4c-confirm-f8c1 PYTHONPATH=.:./python pytest -q -s --tb=short '/tmp/tmem_ldst_readonly_round4_probe.py::test_ldst_readonly_round4[r4-f8-c1-64x32-16x64b]' 2>&1 | tee /tmp/tmem_ldst_readonly_round4_confirm_f8_c1.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r4c-confirm-f8c2 PYTHONPATH=.:./python pytest -q -s --tb=short '/tmp/tmem_ldst_readonly_round4_probe.py::test_ldst_readonly_round4[r4-f8-c2-64x32-16x64b]' 2>&1 | tee /tmp/tmem_ldst_readonly_round4_confirm_f8_c2.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r4c-confirm-rtf32 PYTHONPATH=.:./python pytest -q -s --tb=short '/tmp/tmem_ldst_readonly_round4_probe.py::test_ldst_readonly_round4[r4-roundtrip-f32-c1-64x32-32x32b]' 2>&1 | tee /tmp/tmem_ldst_readonly_round4_confirm_roundtrip_f32.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r4c-confirm-twocta PYTHONPATH=.:./python pytest -q -s --tb=short '/tmp/tmem_ldst_readonly_round4_probe.py::test_ldst_readonly_round4[r4-twocta-f32-c1-128x32-32x32b]' 2>&1 | tee /tmp/tmem_ldst_readonly_round4_confirm_twocta_c1.log
```

## Pass/Fail Matrix

| Case | Result | Classification |
| --- | --- | --- |
| `f32 c0 64x32 32x32b` | pass | direct indexed-view control; lifted parent alone is not failing. |
| `f32 c1 64x32 32x32b` | mismatch `1984/2048` | existing minimal `FZ-0003` row, stable fresh confirmation. |
| `i32 c1 64x32 32x32b` | mismatch `1984/2048` | same row-stripe failure as f32; dtype-stability evidence, not new root. |
| `f16 c1 64x32 16x64b` | mismatch `1984/2048` | same chain1 row-stripe class with subword opcodes. |
| `f16 c2 64x32 16x64b` | mismatch `1024/2048` | new subword-only chain2 identity failure; fresh confirmation stable. |
| `f16 c4 64x32 16x64b` | mismatch `1024/2048` | same samples/opcodes as f16 c2; same-root evidence. |
| `f8 c1/c2 64x32 16x64b` | pass | legal f8 smoke rows did not reproduce f16/f32 failures. |
| `f32 c1 64x64 16x128b` | mismatch `3968/4096` | same row-stripe class scaled to N64. |
| `f32 c1 64x128 16x256b` | mismatch `7936/8192` | same row-stripe class scaled to N128. |
| `f32 c1 128x32 32x32b` | mismatch `4032/4096` | same row-stripe class scaled to M128. |
| `f32 c1 128x64 16x128b` | mismatch `8064/8192` | same row-stripe class scaled to M128/N64. |
| `2CTA f32/i32/f16 c1 128x32` | clean unsupported diagnostic | no runtime miscompile; current backend rejects chained two-CTA descriptor view before launch. |
| `2CTA f32 c0 128x32 32x32b` | pass | direct two-CTA indexed-view control. |
| `f32/i32 c2 col-reverse 64x32 16x64b` | mismatch `1792/2048` | same packet-order class as existing chain2 col-reverse sentinel; dtype-stability evidence. |
| `roundtrip f32/i32 c1 64x32` | pass | confirms same-view roundtrip masks the read-only address bug. |
| `roundtrip 2CTA f32 c1 128x32` | clean unsupported diagnostic | same current two-CTA chained-view boundary. |

## Stable Findings

### R4C-001: `FZ-0003` chain1 row-stripe bug is dtype and packet stable

- Representative fresh repro:
  `/tmp/tmem_ldst_readonly_round4_probe.py::test_ldst_readonly_round4[r4-i32-c1-64x32-32x32b]`
- Observed: `1984/2048` mismatches with first bad samples
  `(32 expected 32 got 64)`, `(33 expected 33 got 65)`, ...
- Opcode evidence:
  `tcgen05.st.sync.aligned.16x32bx2.x16.b32`,
  `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`.
- Assessment: same root as the existing f32 chain1 xfail; do not add a second
  i32 xfail.

### R4C-002: f16 `c2/c4` identity `16x64b` exposes a subword-only read bug

- Representative fresh repro:
  `/tmp/tmem_ldst_readonly_round4_probe.py::test_ldst_readonly_round4[r4-f16-c2-64x32-16x64b]`
- Observed: `1024/2048` mismatches with first bad samples
  `(512 expected 64.0 got 192.0)`, `(513 expected 64.125 got 192.125)`, ...
- Opcode evidence:
  `tcgen05.st.sync.aligned.16x64b.x8.b32` stores followed by
  `tcgen05.ld.sync.aligned.16x32bx2.x8.b32`.
- Difference from existing chain2 col-reverse sentinel: f32/i32 chain2
  identity was green in R3-D, while this f16 identity row fails without
  col-reverse. That makes it a non-overlapping subword/root candidate.

### R4C-003: two-CTA chained descriptor-view `ld/st` currently rejects cleanly

- Representative repro:
  `/tmp/tmem_ldst_readonly_round4_probe.py::test_ldst_readonly_round4[r4-twocta-f32-c1-128x32-32x32b]`
- Observed diagnostic:
  `TMEM layout 'constexpr[32x32b]' unsupported for descriptor view tensor_memory_descriptor<fp32, [128, 32], ... two_ctas=True>`.
- Control: direct two-CTA `c0` indexed view passed.
- Assessment: current boundary is clean unsupported, not a crash or
  miscompile. Do not xfail as a runtime bug.

### R4C-004: f8 legal smoke rows pass

- Repros:
  `r4-f8-c1-64x32-16x64b` and `r4-f8-c2-64x32-16x64b`.
- Result: both passed in fresh processes. No f8 finding from this lane.

## Recommendation

Add at most one additional checked-in strict xfail if the campaign wants a
non-overlapping sentinel beyond the existing f32 chain1 and f32 chain2
col-reverse rows:

```python
LdStCase(
    "ldst-fz20260421-0003-f16-chain2-identity-64x32-16x64b",
    0xA023,
    64,
    32,
    "identity",
    "identity",
    "16x64b",
    2,
)
```

This sentinel covers the f16/subword chain2 identity failure that is not covered
by the existing chain1 row-stripe sentinel or the f32/i32 chain2 col-reverse
packet-order sentinel. Keep the i32 chain1, larger-N/M chain1, c4 f16, and
i32 col-reverse rows report-only because their samples/opcodes are same-root
evidence for already promoted failures.

