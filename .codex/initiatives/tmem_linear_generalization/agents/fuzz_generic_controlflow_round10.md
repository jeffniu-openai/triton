# Round 10 Lane L: Generic pass / control-flow TMEM descriptor fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Repo edit scope: report only.
- Temporary exact probe: `/tmp/tmem_round10_exact_probe.py`
- Results:
  - `/tmp/tmem_round10_exact_shard0.jsonl`
  - `/tmp/tmem_round10_exact_shard1.jsonl`
  - `/tmp/tmem_round10_exact_shard2.jsonl`
  - `/tmp/tmem_round10_exact_shard3.jsonl`

## Scope

This lane stress-tested TMEM descriptor plumbing through generic pass/control-flow
paths using the checked-in structural fuzzer module as the source of cases.
The exact sweep covered:

- nested helpers and repeated inline chains;
- tuple/struct-like returns and sibling views from one parent;
- loops with memdesc and tensor iterator arguments;
- dynamic `if` / `select`;
- mixed TMEM and non-TMEM tensors;
- layout conversions and layout-pressure cases;
- runtime scalar indices; and
- `ld/st`, `ld.red` readback, and one scaled-MMA accumulator descriptor row.

I used an exact runner that imported
`python/test/gluon/test_tmem_structural_fuzzer.py` directly and executed the
checked-in case functions. That kept the result set aligned with the repo's
current fuzz inventory instead of an ad hoc probe harness.

## Commands

Required rebuild before probing:

```bash
make -j8
```

Exact probe syntax check:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_round10_exact_probe.py
```

Exact sweep, sharded across four GPUs with separate caches:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r10-exact-gpu0 \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_round10_exact_probe.py --shard-index 0 --num-shards 4 \
  --results /tmp/tmem_round10_exact_shard0.jsonl

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r10-exact-gpu1 \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_round10_exact_probe.py --shard-index 1 --num-shards 4 \
  --results /tmp/tmem_round10_exact_shard1.jsonl

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r10-exact-gpu2 \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_round10_exact_probe.py --shard-index 2 --num-shards 4 \
  --results /tmp/tmem_round10_exact_shard2.jsonl

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r10-exact-gpu3 \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_round10_exact_probe.py --shard-index 3 --num-shards 4 \
  --results /tmp/tmem_round10_exact_shard3.jsonl
```

## Results Summary

The exact rerun produced:

- `7` passes;
- `1` clean unsupported diagnostic;
- `20` failures that all overlap existing known buckets; and
- `0` new independent buckets.

The main failure families were:

- `FZ-20260421-0001` for `ttg.memdesc_index` legalization / pass-manager failure;
- `FZ-20260421-0002` for generic pass/control-flow/layout-pressure miscompiles;
- `FZ-20260421-0003` for descriptor-view packet-order miscompiles in `ld/st`;
- `FZ-20260421-0004` for `ld.red` descriptor-chain fallback or wrong-value rows;
- `FZ-20260421-0006` for a clean unsupported `ld.red` transpose/slice case;
- `FZ-20260421-0007` for the scaled-MMA accumulator-view miscompile; and
- the known `R5-C` auto-layout crash sentinel for the loop-carried view row.

## Case Table

| Case | Representative layout / shape | Outcome | Bucket / note |
| --- | --- | --- | --- |
| `generic-pass-dynamic-index-chain0` | nested helper + runtime scalar index | compiler crash | `FZ-20260421-0001` |
| `generic-pass-dynamic-index-chain1` | nested helper + runtime scalar index | compiler crash | `FZ-20260421-0001` |
| `generic-pass-dynamic-index-load-only-128x32` | load-only, scalar index | compiler crash | `FZ-20260421-0001` |
| `generic-pass-dynamic-if-chain0-true` | dynamic `if`, inline chain | miscompile | `FZ-20260421-0002` |
| `generic-pass-dynamic-if-chain0-false-16x128b` | dynamic `if`, inline chain, `16x128b` | miscompile | `FZ-20260421-0002` |
| `generic-pass-dynamic-if-chain0-inline` | repeated inline chain under `if` | miscompile | `FZ-20260421-0002` |
| `generic-pass-mixed-captures-chain0` | tuple + mixed TMEM/non-TMEM captures | miscompile | `FZ-20260421-0002` |
| `generic-pass-tuple-mixed-captures-chain0` | tuple-like returns + mixed captures | miscompile | `FZ-20260421-0002` |
| `generic-pass-layout-conversion-pressure-chain0` | layout conversions under helper chaining | miscompile | `FZ-20260421-0002` |
| `generic-pass-layout-conversion-pressure-chain0-16x128b` | layout conversions, `16x128b` | miscompile | `FZ-20260421-0002` |
| `generic-pass-loop-carried-memdesc-view-chain0` | loop-carried memdesc/view iter args | compiler crash | `R5-C` auto-layout crash sentinel |
| `ldst-view-col-reverse-32x32b` | sibling view, col reverse | pass | positive control |
| `ldst-view-row-even-16x64b` | sibling view, row even | pass | positive control |
| `ldst-view-col-rotate-16x128b` | sibling view, col rotate | pass | positive control |
| `ldst-view-identity-32x32b` | direct identity view | pass | positive control |
| `ldst-fz20260421-0003-chain1-64x32-32x32b` | descriptor chain | miscompile | `FZ-20260421-0003` |
| `ldst-fz20260421-0003-chain2-col-reverse-64x32-16x64b` | deeper chain, reversed col | miscompile | `FZ-20260421-0003` |
| `ldst-fz20260421-0003-f16-chain2-identity-64x32-16x64b` | deeper chain, `f16` | miscompile | `FZ-20260421-0003` |
| `ldred-twocta-lifted-256x64` | 2CTA lifted parent | pass | positive control |
| `ldred-direct-128x64` | direct readback | pass | positive control |
| `ldred-view-128x64` | direct view readback | pass | positive control |
| `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min` | 2CTA indexed chain | wrong result / assertion | `FZ-20260421-0004` |
| `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-max` | 2CTA indexed chain | wrong result / assertion | `FZ-20260421-0004` |
| `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min-abs` | 2CTA indexed chain | wrong result / assertion | `FZ-20260421-0004` |
| `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min-nan` | 2CTA indexed chain | wrong result / assertion | `FZ-20260421-0004` |
| `ldred-fz20260421-0004-chain1-64x32-min` | descriptor chain | wrong result / assertion | `FZ-20260421-0004` |
| `ldred-fz20260421-0006-rotate1-transpose-slice-max` | transpose + slice | clean diagnostic | unsupported descriptor-view boundary |
| `mma-scaled-fz20260421-0007-subslice-if-n64-selector0` | scaled-MMA accumulator view | miscompile | `FZ-20260421-0007` |

## Representative Error Shapes

The generic pass/control-flow miscompiles were highly repeatable:

- `generic-pass-dynamic-if-chain0-*` rows reported `8064/8192` mismatched
  elements.
- `generic-pass-layout-conversion-pressure-chain0-*` rows reported
  `8064/8192` mismatched elements.
- `generic-pass-mixed-captures-chain0` and
  `generic-pass-tuple-mixed-captures-chain0` showed the same
  `8064/8192` mismatch pattern.

The `ld/st` packet-order failures were smaller but still deterministic:

- `ldst-fz20260421-0003-chain1-64x32-32x32b` reported `1984/2048` mismatched;
- `ldst-fz20260421-0003-chain2-col-reverse-64x32-16x64b` reported `1792/2048`
  mismatched; and
- `ldst-fz20260421-0003-f16-chain2-identity-64x32-16x64b` reported
  `1023/2048` mismatched.

The scaled-MMA accumulator-view row stayed in the known `FZ-20260421-0007`
bucket and produced a large wrong-value set with NaNs present in the result.

For the `ld.red` two-CTA indexed family, the captured failures were assertion
rows with wrong results, but not a new root cause. The checked-in case names
still align with the existing `FZ-20260421-0004` behavior.

## Classification Notes

### `FZ-20260421-0001`

The dynamic index cases failed before normal lowering completed. One row
emitted the familiar `ttg.memdesc_index` legalization failure, and the other
two surfaced the same compiler-failure class through the exact runner.

### `FZ-20260421-0002`

The generic pass/control-flow rows were all wrong-result rows, not clean
unsupported diagnostics. The failures clustered around dynamic `if`, repeated
inline chains, tuple-like returns, mixed captures, and layout-conversion
pressure.

The loop-carried memdesc/view row belongs with the existing `R5-C` auto-layout
crash sentinel rather than a new bucket.

### `FZ-20260421-0003`

The `ld/st` rows remained split between positive control rows and
descriptor-chain packet-order failures. The known packet-order family is still
the right bucket for the failing rows.

### `FZ-20260421-0004`

The `ld.red` two-CTA indexed/chain rows are still the known fallback / wrong
result family. The exact rows in this lane did not expose a distinct new root.

### `FZ-20260421-0006`

`ldred-fz20260421-0006-rotate1-transpose-slice-max` produced a clean
unsupported diagnostic, so it is a negative boundary rather than a bug.

### `FZ-20260421-0007`

The scaled-MMA accumulator-view row reproduced the known miscompile family.

## New FZ Decision

No new independent `FZ-*` bucket is warranted from this lane.

The only non-pass outcomes were:

- known compiler-failure or miscompile families already covered by
  `FZ-20260421-0001`, `FZ-20260421-0002`, `FZ-20260421-0003`,
  `FZ-20260421-0004`, and `FZ-20260421-0007`;
- the existing `R5-C` auto-layout crash sentinel; and
- one clean unsupported `ld.red` descriptor-view boundary.

That means this round expanded coverage and confirmed the current bucket
boundaries, but it did not uncover a fresh root cause.
