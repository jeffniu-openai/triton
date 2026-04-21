# Round 10 Lane I: `ld.red` modifier and NaN edge fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Starting HEAD: `536b66a17 Record TMEM lit sanity check`
- Mode: discovery only; no backend/compiler repairs attempted.
- Repo edit scope: this report only.
- Temporary probe: `/tmp/tmem_ldred_modifiers_round10_probe.py`
- Main results: `/tmp/tmem_ldred_modifiers_round10_results.jsonl`
- Main run log: `/tmp/tmem_ldred_modifiers_round10_run.log`
- Smoke logs:
  - `/tmp/tmem_ldred_modifiers_round10_smoke.log`
  - `/tmp/tmem_ldred_modifiers_round10_smoke2.log`

## Scope

This lane fuzzed `tcgen05.ld.red` min/max modifiers and NaN handling through
runtime Python/Gluon probes. Each row ran in a child Python process so compiler
assertions, LLVM aborts, and pass-manager failures could be classified without
stopping the sweep.

The generated matrix covered:

- `N in {2, 16, 32, 64, 128, 256}`;
- 1CTA direct, 1CTA indexed parent views, 1CTA descriptor chains, 2CTA indexed
  parent views, and 2CTA descriptor chains;
- row/column permutations `identity`, `even_odd`, and `reverse`;
- reductions `min`, `max`, `min(abs=True)`, `max(abs=True)`;
- NaN propagation with first-column, last-column, and multi-payload placement;
  and
- opcode checks for `.min`, `.max`, `.abs`, `.NaN`, and `.f32` modifiers.

The probe used distinct quiet-NaN input payloads to make placement explicit in
the input matrix, while validating the hardware contract with `equal_nan=True`
for propagation rows.

## Commands

Required rebuild before testing:

```bash
make -j8
```

Result: `ninja: no work to do`.

Temporary probe syntax check:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_ldred_modifiers_round10_probe.py
```

Result: passed.

Case inventory:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_modifiers_round10_probe.py --list | tail -5
```

Result: `TOTAL 266`.

Initial smoke run exposed a harness issue: two-CTA cases were launched without
`num_ctas=2` and correctly failed frontend context checks. I fixed the
temporary probe launcher only, then reran the smoke:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-round10-lane-i-smoke2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_modifiers_round10_probe.py --limit 12 \
  2>&1 | tee /tmp/tmem_ldred_modifiers_round10_smoke2.log
```

Result:

```text
SUMMARY {"opcode fallback": 3, "optimizer abort": 1, "pass": 8}
```

Full subprocess-isolated sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-round10-lane-i \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_modifiers_round10_probe.py \
  2>&1 | tee /tmp/tmem_ldred_modifiers_round10_run.log
```

Raw summary:

```text
SUMMARY {"compiler crash/failure": 28, "opcode fallback": 70, "optimizer abort": 28, "pass": 140}
```

The 28 raw `compiler crash/failure` rows were inspected from their captured
tails and all were clean `OutOfResources` reports, not assertions:

```text
triton.runtime.errors.OutOfResources: out of resource: tensor memory,
Required: 1024, Hardware limit: 512.
```

Reclassified summary:

```text
pass                                           140
opcode fallback (FZ-20260421-0004 family)      70
optimizer abort (FZ-20260421-0008 family)      28
clean OutOfResources diagnostic                28
runtime miscompile                              0
allocator assertion                             0
clean .x1 diagnostic                            0
```

Fresh exact confirmations:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-round10-lane-i-confirm-pass \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_modifiers_round10_probe.py \
  --case ldred-r10-1cta-chain_row-128x64-even_odd-identity-min-abs-nan-multi_payload \
  2>&1 | tee /tmp/tmem_ldred_modifiers_round10_confirm_pass.log

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-round10-lane-i-confirm-fallback \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_modifiers_round10_probe.py \
  --case ldred-r10-2cta-indexed-256x32-identity-reverse-min-nan-first \
  2>&1 | tee /tmp/tmem_ldred_modifiers_round10_confirm_fallback.log

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-round10-lane-i-confirm-optimizer \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_modifiers_round10_probe.py \
  --case ldred-r10-2cta-chain_row-256x64-even_odd-reverse-min-nan-first \
  2>&1 | tee /tmp/tmem_ldred_modifiers_round10_confirm_optimizer.log

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-round10-lane-i-confirm-resource \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_modifiers_round10_probe.py \
  --case ldred-r10-1cta-indexed-128x256-identity-identity-min \
  2>&1 | tee /tmp/tmem_ldred_modifiers_round10_confirm_resource.log
```

Results:

- positive row emitted
  `tcgen05.ld.red.sync.aligned.32x32b.x64.min.abs.NaN.f32`;
- 2CTA indexed row emitted no `.ld.red.` opcode and used
  `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`;
- 2CTA row-chain row failed in `TritonNvidiaGPUOptimizeTMemLayoutsPass`; and
- parent-view `N=256` row reported clean TMEM `OutOfResources`.

## Classification

### Positive rows

All 140 positive rows compiled, executed, matched PyTorch references, and
emitted expected `tcgen05.ld.red` opcodes with the requested modifier suffix.

Positive coverage included:

- 1CTA direct reductions for every requested `N`, including `N=256`;
- 1CTA indexed and descriptor-chain reductions for `N <= 128`;
- row-chain `even_odd` layout at `N in {2,16,32,64}`;
- reversed-column rows at `N in {16,32,64,128,256}` where generated; and
- first-column, last-column, and multi-payload NaN placements with
  `propagate_nan=ALL`.

This is useful negative evidence: Lane I did not find a modifier-specific or
NaN-placement-specific runtime miscompile on the 1CTA hardware-`ld.red` path.

### Opcode fallback: `FZ-20260421-0004` family

Seventy rows produced correct runtime values but emitted plain `tcgen05.ld`
instead of `tcgen05.ld.red`. These were all 2CTA indexed or 2CTA chain-col
views for `N in {2,16,32,64,128}`.

The fallback pattern held uniformly across:

- `min` and `max`;
- `abs=True`;
- `propagate_nan=ALL`;
- first/last/multi-payload NaN placement; and
- identity and reversed column layouts.

This broadens `FZ-20260421-0004`, but does not warrant a new FZ id because the
owner, behavior, and emitted-opcode failure mode match the existing 2CTA
indexed `ld.red` fallback bucket.

### Optimizer abort: `FZ-20260421-0008` family

Twenty-eight rows failed in `TritonNvidiaGPUOptimizeTMemLayoutsPass`. They were
all 2CTA row-chain views over an `even_odd` row layout at
`N in {2,16,32,64}`.

The exact confirmation reproduced the existing pass-manager failure shape with
row/col/block layout mismatch. This extends `FZ-20260421-0008` across all
modifier and NaN-placement rows generated by this lane, but does not create a
new independent bucket.

### Clean resource diagnostics

Twenty-eight rows reached clean TMEM `OutOfResources` diagnostics. These were
parent-view shapes at `N=256`:

- 1CTA indexed parent views over `[2,128,256]`;
- 1CTA chain-col parent views over `[2,128,256]`;
- 2CTA indexed parent views over `[2,256,256]`; and
- 2CTA chain-col parent views over `[2,256,256]`.

The direct 1CTA `[128,256]` rows passed, so the resource boundary is specific
to the larger parent allocation, not the reduced logical view. The diagnostic
was clean and reported required TMEM size `1024` against hardware limit `512`.

### Empty categories

Lane I found no runtime miscompile, allocator assertion, or clean `.x1`
diagnostic. The `.x1` count is expected to be zero for this lane because the
requested `N` set starts at `2`; `N=2` rows either emitted `.x2` hardware
reductions on 1CTA or fell into the existing 2CTA opcode/optimizer families.

## New FZ Decision

No new non-overlapping `FZ-*` id is warranted from Round 10 Lane I.

The findings reinforce existing buckets:

- `FZ-20260421-0004`: 2CTA indexed/chain-col `ld.red` falls back to plain
  `ld`, independent of min/max, abs, NaN propagation, NaN placement, or
  reversed columns.
- `FZ-20260421-0008`: 2CTA row-chain `ld.red` aborts in the optimizer,
  independent of modifier and NaN placement.

The 1CTA hardware path handled all generated modifier and NaN-placement rows
correctly. Backend repair remains deferred until the discovery campaign stops
finding new bugs or the user explicitly pivots from discovery to repair.
