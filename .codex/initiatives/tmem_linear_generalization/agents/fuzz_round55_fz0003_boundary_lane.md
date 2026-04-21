# Round 55 FZ-0003 Descriptor-View Boundary Lane B

Date: 2026-04-21
Branch: `codex/tmem`
Scope: discovery/cataloging only; no backend repairs attempted.

This lane sharpened existing `FZ-20260421-0003` around static descriptor-view
`ld/st` wrong results. The target was a minimal boundary for chain-0 versus
chain-1, 1CTA versus 2CTA, row/column layouts, dtype variants, and descriptor
view operation sequence.

## Preflight

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Temporary Probe

Temporary pytest probe:

```text
/tmp/tmem_round55_fz0003_boundary_probe.py
```

The probe imports the checked-in structural fuzzer helpers, then builds a
single static descriptor-view kernel with this schema:

```text
case_id, seed, M, N, two_cta, row_kind, col_kind, dtype_name, chain_id,
op_sequence, instr_variant
```

Descriptor chains:

```text
chain0 = index(1).reshape((M//2, 2, N)).permute([1, 0, 2]).reshape((M, N))
chain1 = index(1).reshape((M//2, 2, N//2, 2))
              .permute([1, 0, 3, 2]).permute([1, 0, 3, 2]).reshape((M, N))
```

Operation sequences:

```text
load_only = base.store(input); view.load(); global.store(view_load)
load_store_base_read = base.store(input); view.load(); view.store(load); base.load()
store_through_view_read_view = base.store(input); view.load(); view.store(load); view.load()
```

Syntax check:

```bash
PYTHONPATH=.:./python python3 -m py_compile /tmp/tmem_round55_fz0003_boundary_probe.py
```

Result: passed.

Focused run:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest -vv -s --tb=short /tmp/tmem_round55_fz0003_boundary_probe.py
```

Result:

```text
23 passed, 1 failed in 6.13s
```

The failure was a clean compile-time unsupported boundary for the minimal
two-CTA chain-0 row:

```text
r55-min-chain0-2cta-identity-f32-load-only:
TMEM layout 'constexpr[32x32b]' unsupported for descriptor view
tensor_memory_descriptor<fp32, [128, 32], ... two_ctas=True>
```

All rows that reached runtime had matching PTX/LLIR opcode lists. Runtime
wrong-result rows therefore remain `FZ-20260421-0003` miscompiles, not opcode
drift or IR/PTX disagreement.

## Results

| Case id | Seed | Axes | Result |
| --- | ---: | --- | --- |
| `r55-min-chain0-1cta-identity-f32-load-only` | `0x550101` | 1CTA, `64x32`, identity, f32, chain0, load-only | wrong result, `1984/2048` mismatches, max abs `5.158187` |
| `r55-min-chain1-1cta-identity-f32-load-only` | `0x550102` | 1CTA, `64x32`, identity, f32, chain1, load-only | pass, `0/2048` mismatches |
| `r55-min-chain0-1cta-row-reverse-f32-load-only` | `0x550103` | 1CTA, `64x32`, row reverse, f32, chain0, load-only | wrong result, `1984/2048` mismatches, max abs `4.374455` |
| `r55-min-chain1-1cta-row-reverse-f32-load-only` | `0x550104` | 1CTA, `64x32`, row reverse, f32, chain1, load-only | pass, `0/2048` mismatches |
| `r55-min-chain0-1cta-col-reverse-f32-load-only` | `0x550105` | 1CTA, `64x32`, col reverse, f32, chain0, load-only | wrong result, `2032/2048` mismatches, max abs `5.106162` |
| `r55-min-chain1-1cta-col-reverse-f32-load-only` | `0x550106` | 1CTA, `64x32`, col reverse, f32, chain1, load-only | wrong result, `1792/2048` mismatches, max abs `5.145791` |
| `r55-min-chain0-2cta-identity-f32-load-only` | `0x550107` | 2CTA, `128x32`, identity, f32, chain0, load-only | clean unsupported compile diagnostic |
| `r55-min-chain1-2cta-identity-f32-load-only` | `0x550108` | 2CTA, `128x32`, identity, f32, chain1, load-only | pass, `0/4096` mismatches |
| `r55-chain0-1cta-identity-f32-load-only` | `0x550001` | 1CTA, `128x64`, identity, f32, chain0, load-only | wrong result, `8064/8192` mismatches, max abs `6.103236` |
| `r55-chain1-1cta-identity-f32-load-only` | `0x550002` | 1CTA, `128x64`, identity, f32, chain1, load-only | pass, `0/8192` mismatches |
| `r55-chain0-2cta-identity-f32-load-only` | `0x550003` | 2CTA, `256x64`, identity, f32, chain0, load-only | wrong result, `16256/16384` mismatches, max abs `5.866594` |
| `r55-chain1-2cta-identity-f32-load-only` | `0x550004` | 2CTA, `256x64`, identity, f32, chain1, load-only | pass, `0/16384` mismatches |
| `r55-chain0-1cta-row-rotate-col-rev-f32-load-only` | `0x550005` | 1CTA, `128x64`, row rotate1/col reverse, f32, chain0, load-only | wrong result, `8064/8192` mismatches, max abs `5.661026` |
| `r55-chain1-1cta-row-rotate-col-rev-f32-load-only` | `0x550006` | 1CTA, `128x64`, row rotate1/col reverse, f32, chain1, load-only | pass, `0/8192` mismatches |
| `r55-chain0-2cta-row-rotate-col-rev-f32-load-only` | `0x550007` | 2CTA, `256x64`, row rotate1/col reverse, f32, chain0, load-only | wrong result, `16256/16384` mismatches, max abs `6.111139` |
| `r55-chain1-2cta-row-rotate-col-rev-f32-load-only` | `0x550008` | 2CTA, `256x64`, row rotate1/col reverse, f32, chain1, load-only | pass, `0/16384` mismatches |
| `r55-chain0-1cta-identity-i32-load-only` | `0x550009` | 1CTA, `128x64`, identity, i32, chain0, load-only | wrong result, `8063/8192` mismatches, max abs `1964.000000` |
| `r55-chain1-1cta-identity-i32-load-only` | `0x55000a` | 1CTA, `128x64`, identity, i32, chain1, load-only | pass, `0/8192` mismatches |
| `r55-chain0-1cta-identity-f16-load-only` | `0x55000b` | 1CTA, `128x64`, identity, f16, chain0, load-only | wrong result, `8064/8192` mismatches, max abs `5.777344` |
| `r55-chain1-1cta-identity-f16-load-only` | `0x55000c` | 1CTA, `128x64`, identity, f16, chain1, load-only | pass, `0/8192` mismatches |
| `r55-chain0-1cta-identity-f32-load-store-base-read` | `0x55000d` | 1CTA, `128x64`, identity, f32, chain0, load/store/base-read | pass, `0/8192` mismatches |
| `r55-chain1-1cta-identity-f32-load-store-base-read` | `0x55000e` | 1CTA, `128x64`, identity, f32, chain1, load/store/base-read | pass, `0/8192` mismatches |
| `r55-chain0-1cta-identity-f32-store-through-view-read-view` | `0x55000f` | 1CTA, `128x64`, identity, f32, chain0, store-through-view/read-view | wrong result, `8064/8192` mismatches, max abs `5.905981` |
| `r55-chain1-1cta-identity-f32-store-through-view-read-view` | `0x550010` | 1CTA, `128x64`, identity, f32, chain1, store-through-view/read-view | pass, `0/8192` mismatches |

## Checked-In Exact Anchors

Command:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest -vv -s --tb=short \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_descriptor_view_read[ldst-fz20260421-0003-chain1-64x32-32x32b]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_descriptor_view_read[ldst-fz20260421-0003-chain2-col-reverse-64x32-16x64b]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_descriptor_view_read[ldst-fz20260421-0003-f16-chain2-identity-64x32-16x64b]'
```

Result:

```text
3 xfailed in 2.80s
```

These checked-in anchors remain stable and keep the older `chain1`, `chain2`
col-reverse, and f16 chain2 variants pinned as expected `FZ-20260421-0003`
rows.

## Boundary Summary

- Smallest new static chain-0 wrong-result control: 1CTA `64x32` identity f32
  load-only, seed `0x550101`, `1984/2048` mismatches.
- Adjacent chain-1 positive control: 1CTA `64x32` identity f32 load-only,
  seed `0x550102`, `0/2048` mismatches with the same `16x32bx2.x16` opcode
  family.
- 2CTA positive/negative split: minimal `128x32` chain-0 is clean unsupported,
  while `128x32` chain-1 passes; larger `256x64` chain-0 wrong-results and
  chain-1 passes.
- Row/column split: identity and row-only permutations preserve the chain-0
  wrong-result / chain-1 pass boundary at `64x32`; pure column reverse is
  harsher and makes both chain-0 and chain-1 wrong-result rows at `64x32`.
  The larger `128x64` row-rotate/col-reverse rows restore the chain-0 fail /
  chain-1 pass split.
- Dtype split: f32, i32, and f16 all reproduce the chain-0 `128x64`
  load-only wrong result, while the matching chain-1 rows pass.
- Operation sequence split: direct `view.load()` and
  `view.load(); view.store(); view.load()` expose chain-0 wrong results.
  `view.load(); view.store(); base.load()` masks the issue because the bad
  load/store mapping is self-consistent when read back through the base view.

## Classification

No new independent `FZ-*` was opened. The wrong-result rows sharpen existing
`FZ-20260421-0003` as a static descriptor-view layout materialization /
packet-mapping bug. The key discriminator is not dynamic SSA/control flow:
static no-control-flow chain-0 rows are enough. Passing rows differ by using a
chain that canonicalizes to a layout the backend maps consistently, or by using
an operation sequence whose load/store mismatch cancels before global readback.
