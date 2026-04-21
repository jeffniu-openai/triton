# Lane R3-D: FZ-20260421-0003 ld/st descriptor-view chain minimization

- Date: 2026-04-21
- Branch/HEAD: `codex/tmem` / `78f2845a0`
- Mode: discovery/minimization only; no backend/compiler repair.
- Temporary probe: `/tmp/tmem_ldst_round3_probe.py`

## Scope

Expanded and minimized `FZ-20260421-0003` around ld/st descriptor-view chain
runtime miscompiles. The probe separates read-only descriptor-view loads from
same-view load/store roundtrips because the roundtrip form can hide bad address
mapping.

Probe chain ids:

- `c0`: direct `index(1)` control.
- `c1`: `reshape((M//2,2,N)).permute([1,0,2]).reshape((M,N))`.
- `c2`: `reshape((M,N//2,2)).permute([0,2,1]).permute([0,2,1]).reshape((M,N))`.
- `c3`: double transpose plus slices.
- `c4`: double 4D row/col pair permutation.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Collect:

```bash
PYTHONPATH=.:./python pytest --collect-only -q /tmp/tmem_ldst_round3_probe.py
```

Result: `332 tests collected`.

Four-GPU sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r3-ldst-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 /tmp/tmem_ldst_round3_probe.py 2>&1 | tee /tmp/tmem_ldst_round3_g1.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r3-ldst-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_ldst_round3_probe.py 2>&1 | tee /tmp/tmem_ldst_round3_g2.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r3-ldst-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 /tmp/tmem_ldst_round3_probe.py 2>&1 | tee /tmp/tmem_ldst_round3_g3.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r3-ldst-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 /tmp/tmem_ldst_round3_probe.py 2>&1 | tee /tmp/tmem_ldst_round3_g4.log
```

Raw split results:

- group 1: `72 failed, 11 passed, 249 deselected`
- group 2: `38 failed, 45 passed, 249 deselected`
- group 3: `50 failed, 33 passed, 249 deselected`
- group 4: `30 failed, 53 passed, 249 deselected`

The raw failures include expected clean unsupported diagnostics for tiny rows
and `c3` row-anchor cases; only successful compile+launch mismatches are
counted as FZ-0003 miscompiles below.

Fresh confirmations:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r3-confirm-c1 PYTHONPATH=.:./python pytest -q -s --tb=short '/tmp/tmem_ldst_round3_probe.py::test_ldst_round3[read-64x32-identity-identity-c1-32x32b-f32]' 2>&1 | tee /tmp/tmem_ldst_round3_confirm_c1.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r3-confirm-c2-colrev PYTHONPATH=.:./python pytest -q -s --tb=short '/tmp/tmem_ldst_round3_probe.py::test_ldst_round3[read-64x32-identity-reverse-c2-16x64b-f32]' 2>&1 | tee /tmp/tmem_ldst_round3_confirm_c2_colrev.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r3-confirm-c4-colrev PYTHONPATH=.:./python pytest -q -s --tb=short '/tmp/tmem_ldst_round3_probe.py::test_ldst_round3[read-64x32-identity-reverse-c4-16x64b-f32]' 2>&1 | tee /tmp/tmem_ldst_round3_confirm_c4_colrev.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r3-confirm-roundtrip-mask PYTHONPATH=.:./python pytest -q -s --tb=short '/tmp/tmem_ldst_round3_probe.py::test_ldst_round3[roundtrip-64x32-identity-identity-c1-32x32b-f32]' 2>&1 | tee /tmp/tmem_ldst_round3_confirm_roundtrip_mask.log
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r3-confirm-direct PYTHONPATH=.:./python pytest -q -s --tb=short '/tmp/tmem_ldst_round3_probe.py::test_ldst_round3[read-64x32-identity-identity-c0-32x32b-f32]' 2>&1 | tee /tmp/tmem_ldst_round3_confirm_direct.log
```

Results:

- `read-64x32-identity-identity-c1-32x32b-f32`: stable mismatch,
  `1984/2048`, failed in `4.86s`.
- `read-64x32-identity-reverse-c2-16x64b-f32`: stable mismatch,
  `1792/2048`, failed in `4.68s`.
- `read-64x32-identity-reverse-c4-16x64b-f32`: stable mismatch,
  `1792/2048`, failed in `4.71s`.
- `roundtrip-64x32-identity-identity-c1-32x32b-f32`: passed in `4.78s`.
- `read-64x32-identity-identity-c0-32x32b-f32`: passed in `4.49s`.

## Pass/Fail Matrix

Read-only `f32` rows:

| Shape | Chain/layout/variant | Result | Notes |
| --- | --- | --- | --- |
| `16x32` | all probed | clean unsupported | `base.get_reg_layout` rejects tiny descriptor view; not a miscompile. |
| `32x32` | all probed | clean unsupported | row-anchor diagnostic; not a miscompile. |
| `64x32` | `c0`, all layouts, `32x32b`/`16x64b` | pass | direct indexed-view control. |
| `64x32` | `c1`, identity/identity, `32x32b` | mismatch `1984/2048` | minimal stable FZ-0003 row. |
| `64x32` | `c1`, identity/identity, `16x64b` | mismatch `1984/2048` | same first bad samples as `32x32b`; load opcode falls back to `16x32bx2`. |
| `64x32` | `c1`, identity/reverse, `32x32b` | mismatch `1984/2048` | same row-stripe shift. |
| `64x32` | `c1`, identity/reverse, `16x64b` | mismatch `2024/2048` | row bug plus col-reverse packet ordering. |
| `64x32` | `c1`, even_odd/identity, `32x32b`/`16x64b` | mismatch `1984/2048` | row-layout permutation does not remove failure. |
| `64x32` | `c2`, identity/identity, both variants | pass | double column-pair permutation alone is green. |
| `64x32` | `c2`, identity/reverse, `32x32b` | pass | col-reverse alone is green for `32x32b`. |
| `64x32` | `c2`, identity/reverse, `16x64b` | mismatch `1792/2048` | round-2 expansion reproduces here. |
| `64x32` | `c2`, even_odd/identity, both variants | pass | not a row-layout-only failure. |
| `64x32` | `c3`, all probed | clean unsupported | row-anchor diagnostic; not a miscompile. |
| `64x32` | `c4`, identity/identity and even_odd/identity | pass | double 4D pair permutation mostly green. |
| `64x32` | `c4`, identity/reverse, `16x64b` | mismatch `1792/2048` | same samples/opcodes as `c2` col-reverse. |
| `64x64` | `c1`, all layouts, both variants | mismatch | same row-stripe shift, `3968/4096` or `4066/4096` with col-reverse `16x64b`. |
| `64x64` | `c2`/`c4`, identity/reverse, `16x64b` | mismatch `3840/4096` | same col-reverse packet class. |
| `128x32` | `c1`, all layouts, both variants | mismatch `4032/4096` | chain1 scales to larger `M`. |
| `128x32` | `c2`/`c3`/`c4`, all probed | pass | larger `M` avoids the col-reverse `16x64b` failure seen at `M=64`. |

Representative dtype rows at `64x32`:

| Dtype | Chain/layout/variant | Result |
| --- | --- | --- |
| `i32` | `c1`, identity/identity, `32x32b` and `16x64b` | mismatch `1984/2048` |
| `i32` | `c1`, identity/reverse, `32x32b` | mismatch `1984/2048` |
| `i32` | `c1`, identity/reverse, `16x64b` | mismatch `2024/2048` |
| `i32` | `c2`/`c4`, identity/reverse, `16x64b` | mismatch `1792/2048` |
| `f16` | `c1`, `16x64b`, identity/identity and identity/reverse | mismatch `1984/2048` |
| `f16` | `c2`/`c4`, identity/identity, `16x64b` | mismatch `1024/2048` |
| `f16` | `c2`/`c4`, identity/reverse, `16x64b` | pass |
| `f16` | many `32x32b` rows | clean unsupported | descriptor view rejects before launch. |

Roundtrip rows:

- `64x32` `c1` roundtrip passes despite read-only `c1` failing. The same
  wrong view arithmetic is applied to the load and store through the alias, then
  the base is reloaded, so the wrong addressing cancels out.
- `128x32` roundtrip rows pass broadly, including `c1`, matching the existing
  runtime-matrix coverage shape.
- Smaller `16x32`/`32x32` and some `c3` roundtrip rows cleanly reject with the
  same row-anchor diagnostic; not counted as miscompiles.

## Opcode Evidence

Minimal chain1:

- `read-64x32-identity-identity-c1-32x32b-f32`
- Mismatch: `1984/2048`; first bad samples read row 2 values where row 1 was
  expected: `(32, expected 32, got 64)`, `(33, expected 33, got 65)`, ...
- PTX ops:
  `tcgen05.st.sync.aligned.16x32bx2.x16.b32`,
  `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`.

Chain2 col-reverse:

- `read-64x32-identity-reverse-c2-16x64b-f32`
- Mismatch: `1792/2048`; first bad samples show column packet permutation:
  `(1, expected 1, got 16)`, `(2, expected 2, got 8)`,
  `(3, expected 3, got 24)`.
- PTX ops:
  stores are `tcgen05.st.sync.aligned.16x64b.x16.b32` while loads are repeated
  `tcgen05.ld.sync.aligned.16x32bx2.x2.b32`.

Chain4 col-reverse:

- `read-64x32-identity-reverse-c4-16x64b-f32`
- Mismatch and sample pattern exactly match chain2 col-reverse:
  `1792/2048`, first bad samples `(1 -> 16)`, `(2 -> 8)`, `(3 -> 24)`.
- PTX ops also match chain2 col-reverse:
  `16x64b` stores followed by repeated `16x32bx2.x2` loads.

## Boundary Assessment

Smallest runtime miscompile found:

- Shape `[2, 64, 32]` parent, indexed view `[64, 32]`.
- `f32`, chain `c1`, identity layout, `32x32b`.
- This is the same minimal shape already recorded for `FZ-20260421-0003`.
  Smaller `[2, 16, 32]` and `[2, 32, 32]` probes reject before launch with
  descriptor-view unsupported diagnostics.

Chain2/col-reverse relationship:

- `c2` identity layout passes for both `32x32b` and `16x64b`.
- `c2` col-reverse passes for `32x32b` but miscompiles for `16x64b`.
- `c4` col-reverse `16x64b` has the same mismatch cardinality, samples, and
  opcode pattern as `c2` col-reverse `16x64b`; treat `c4` as the same likely
  root cause, not a distinct catalog bucket.
- This appears adjacent to but not identical to the `c1` row-stripe failure:
  `c1` breaks across identity, reverse, and row-permuted layouts; `c2/c4`
  require col-reverse plus `16x64b` in the tested `f32/i32` rows.

Existing green matrix explanation:

- Existing descriptor-chain runtime-matrix tests mostly use load/store
  roundtrip patterns through the same descriptor alias, for example
  `tmem_ldst_descriptor_chain_kernel` and
  `tmem_ldst_descriptor_roundtrip_kernel`. The roundtrip control above passes
  for the minimal `c1` row where read-only fails, so those green tests do not
  prove that descriptor-view read address arithmetic is correct.
- Existing direct indexed-view controls are still meaningful: direct `c0`
  read-only passes at `64x32`, so the failure is introduced by descriptor-view
  chain arithmetic, not by the lifted parent layout alone.
- The matrix also explains the clean-negative edge: row-anchor diagnostics for
  tiny shapes and `c3` variants are current clean unsupported boundaries, not
  runtime miscompiles.

## Recommendations

Checked-in strict xfails:

- Keep the existing strict xfail:
  `test_tmem_structural_fuzzer_ldst_descriptor_view_read[ldst-fz20260421-0003-chain1-64x32-32x32b]`.
- Add one strict xfail for the stable adjacent col-reverse packet case:
  `LdStCase("ldst-fz20260421-0003-chain2-col-reverse-64x32-16x64b", ..., 64, 32, "identity", "reverse", "16x64b", 2)`.
  This catches the `c2`/round-2 expansion class without depending on broader
  dtype/subword behavior.

Report-only catalog additions:

- Record `c4` col-reverse `16x64b` as same-root evidence under FZ-0003, not a
  separate strict xfail, because it reproduces the same samples and PTX shape as
  `c2`.
- Record `i32` chain1 and col-reverse rows as dtype stability evidence.
- Keep `f16` rows report-only for now: several compile through and mismatch,
  but `32x32b` coverage often rejects early and the subword opcode selection
  differs enough that it should not be the primary checked-in minimizer.

