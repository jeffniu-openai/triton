# Round 8 Lane A: `ld.red` allocator/opcode structural fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Temporary probe: `/tmp/tmem_ldred_allocator_opcode_round8_probe.py`
- Aggregate results: `/tmp/tmem_ldred_allocator_opcode_round8_results.jsonl`
- Full run log: `/tmp/tmem_ldred_allocator_opcode_round8_run.log`

## Scope

This lane expanded generator-backed runtime probing around the known
`ld.red`/`ld/st` allocation and opcode boundaries:

- `FZ-20260421-0004`: 2CTA indexed `ld.red` falls back to plain `ld`;
- `FZ-20260421-0005`: 256-row lifted-parent allocator assertion;
- `FZ-20260421-0008`: 2CTA indexed row-chain optimizer abort; and
- `FZ-20260421-0009`: 1CTA direct indexed `ld.red` over a 256-row parent
  allocator assertion.

The probe exercised 55 subprocess-isolated Python/Gluon cases across:

- `ld.red` direct indexed parents and row/column descriptor chains;
- 1CTA and 2CTA parent shapes `[2,M,N]`;
- `M in {128,256,512}` and `N in {1,2,16,32,64,128}` where the existing
  helper kernels can express the row;
- row/column permutations `identity`, `even_odd`, and `reverse`;
- reductions `min`, `max`, `min(abs=True)`, and NaN-propagating `min`; and
- `ld/st` read-only dtype variants `f16` and `i32`.

C++ assertions and optimizer aborts were isolated in child Python processes.
Rows were cataloged only; no backend fixes were attempted.

## Commands

Required rebuild before tests:

```bash
CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8
```

Result: `ninja: no work to do`.

Probe syntax check:

```bash
PYTHONPATH=.:./python:python/test/gluon python -m py_compile /tmp/tmem_ldred_allocator_opcode_round8_probe.py
```

Result: passed.

Case inventory:

```bash
PYTHONPATH=.:./python:python/test/gluon python - <<'PY'
import importlib.util
spec=importlib.util.spec_from_file_location('p','/tmp/tmem_ldred_allocator_opcode_round8_probe.py')
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
print(len(p.selected_cases()))
PY
```

Result: `55`.

Full subprocess-isolated sweep:

```bash
PYTHONPATH=.:./python:python/test/gluon python /tmp/tmem_ldred_allocator_opcode_round8_probe.py 2>&1 | tee /tmp/tmem_ldred_allocator_opcode_round8_run.log
```

Raw parent summary:

```text
SUMMARY {"FZ-20260421-0004 ld.red opcode fallback": 18, "pass": 18, "uncategorized failure": 19}
```

The probe intentionally stores only a small structured exception string in the
child result. The 19 raw `uncategorized failure` rows were classified from the
captured `stderr_tail` pass name and diagnostic text:

```text
FZ-20260421-0004 ld.red opcode fallback                  18
allocator failure/assert (FZ-0005/FZ-0009 family)         14
clean x1 ld.red unsupported diagnostic                     1
optimizer row/col/block abort (FZ-0008 family)             4
pass                                                      18
```

Fresh exact confirmations:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r8a-confirm-pos PYTHONPATH=.:./python:python/test/gluon python /tmp/tmem_ldred_allocator_opcode_round8_probe.py --case ldred-1cta-index-128x128-row-evenodd-min 2>&1 | tee /tmp/tmem_ldred_allocator_opcode_round8_confirm_pos.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r8a-confirm-opcode PYTHONPATH=.:./python:python/test/gluon python /tmp/tmem_ldred_allocator_opcode_round8_probe.py --case ldred-2cta-index-256x64-identity-min_nan 2>&1 | tee /tmp/tmem_ldred_allocator_opcode_round8_confirm_opcode.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r8a-confirm-opt PYTHONPATH=.:./python:python/test/gluon python /tmp/tmem_ldred_allocator_opcode_round8_probe.py --case ldred-2cta-rowchain-256x2-evenodd-min 2>&1 | tee /tmp/tmem_ldred_allocator_opcode_round8_confirm_optimizer.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r8a-confirm-alloc PYTHONPATH=.:./python:python/test/gluon python /tmp/tmem_ldred_allocator_opcode_round8_probe.py --case ldst-readonly-i32-256x32-direct-index 2>&1 | tee /tmp/tmem_ldred_allocator_opcode_round8_confirm_allocator.log
```

Results:

- Positive control emitted `tcgen05.ld.red.sync.aligned.32x32b.x128.min.f32`.
- Opcode fallback control emitted `tcgen05.ld.sync.aligned.16x32bx2.x32.b32`
  instead of `.ld.red.`.
- Optimizer control reproduced the `row`/`col` versus `row`/`col`/`block`
  dimension mismatch in `TritonNvidiaGPUOptimizeTMemLayoutsPass`.
- Allocator control reproduced
  `TensorMemoryAllocation.cpp:65: MemoryBitMap::findFirstFit(...)` with
  assertion `kNumRows - numRows >= 0`.

## Classification

### Positive rows

The following surfaces compiled, executed, matched PyTorch references, and
emitted expected TCGEN05 hardware messages:

- 1CTA indexed `ld.red` for `128x{16,32,64,128}`.
- 1CTA row-permuted indexed `ld.red` for `128x{16,32,64,128}`.
- 1CTA column-chain `load_max` for `128x{16,32,64,128}` with reverse column
  layout.
- `ld/st` read-only `f16` descriptor-chain rows for `128x{16,32,64}`.
- `ld/st` read-only `i32` descriptor-chain rows for `128x{16,32,64}`.

These positives are useful boundaries: 128-row direct/indexed and chained
views are not generally broken, and row/column permutations do not by
themselves prevent hardware `ld.red`.

### `FZ-20260421-0004` extension: 2CTA indexed opcode fallback

Eighteen rows produced correct runtime values but emitted plain `tcgen05.ld`
instead of `tcgen05.ld.red`.

Representative rows:

- `ldred-2cta-index-128x2-identity-min`;
- `ldred-2cta-index-128x32-identity-min`;
- `ldred-2cta-index-128x64-identity-min`;
- `ldred-2cta-index-256x{2,32,64}-identity-min`; and
- `ldred-2cta-index-256x{2,16,32,64}-identity-{max,min_abs,min_nan}`.

The `max`, `abs`, and NaN variants follow the same fallback pattern as plain
`min`, so this does not look modifier-specific. It is broader evidence for the
existing 2CTA indexed opcode-selection bug rather than a new non-overlapping
finding.

### `FZ-20260421-0008` extension: row/col/block optimizer abort

Four two-CTA row-chain rows failed in
`TritonNvidiaGPUOptimizeTMemLayoutsPass`:

- `ldred-2cta-rowchain-256x2-evenodd-min`;
- `ldred-2cta-rowchain-256x16-evenodd-min`;
- `ldred-2cta-rowchain-256x32-evenodd-min`; and
- `ldred-2cta-rowchain-256x64-evenodd-min`.

The exact confirmation for `256x2` shows the existing fatal message:

```text
LLVM ERROR: Dimensions must match, ignoring order, but they don't.
Got dims: ["row", "col"] and ["row", "col", "block"]
```

This extends `FZ-20260421-0008` across `N in {2,16,32,64}`. No new FZ id is
warranted because the owner pass, shape of view chain, and failure mode match
the promoted row/col optimizer sentinel.

### Allocator failure/assert family: `FZ-20260421-0005` / `FZ-20260421-0009`

Fourteen rows failed in `TritonTensorMemoryAllocationPass` or reproduced the
underlying `MemoryBitMap::findFirstFit` assertion:

- 1CTA direct indexed `ld.red` for `256x{16,32,64,128}`;
- 1CTA direct indexed `ld.red` for `512x{16,32,64,128}`;
- direct indexed `ld/st` read-only `f16` for `256x{16,32,64}`; and
- direct indexed `ld/st` read-only `i32` for `256x{16,32,64}`.

The exact `ld/st` `i32` confirmation reproduced:

```text
TensorMemoryAllocation.cpp:65: MemoryBitMap::findFirstFit(...):
Assertion `kNumRows - numRows >= 0' failed.
```

This is broader evidence that the allocator boundary is not limited to
`ld.red` or `f32`; direct indexed 256-row parents also fail for read-only
`ld/st` on `f16` and `i32`. I do not assign a new FZ id because the pass,
assertion, and selected 256-row direct-index shape overlap the existing
allocator buckets. If the later repair phase wants finer ownership labels,
the useful split is:

- `FZ-20260421-0009`: positive-sized 256-row 1CTA indexed `ld.red` source
  should compile to hardware `ld.red`; and
- `FZ-20260421-0005`-adjacent: generic 256-row direct-index `ld/st` and
  larger-than-TMEM-capacity 512-row allocation diagnostics should be made
  clean and precise instead of asserting/failing late.

### Clean diagnostic

`ldred-2cta-rowchain-256x1-evenodd-min` emitted a clean verifier diagnostic:

```text
tmem_load reduction selected a scalar tcgen05.ld.red message, but
tcgen05.ld.red requires at least an .x2 message shape.
```

This is a valid hardware/ISA boundary rather than a bug.

## New FZ Decision

No new non-overlapping `FZ-*` id is warranted from this lane.

The sweep found substantial new coverage for already-known failure families:

- wider `FZ-20260421-0004` evidence across 2CTA indexed shapes and reduction
  modifiers;
- wider `FZ-20260421-0008` evidence across row-chain `N`; and
- wider allocator evidence showing direct indexed 256-row parent failures for
  both `ld.red` and `ld/st` dtype variants.

Backend repair remains deferred until the discovery campaign stops finding new
failures or the user explicitly pivots from discovery to repair.
