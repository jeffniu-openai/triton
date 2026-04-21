# Round 12 Lane U: `tcgen05.ld.red` row/column/opcode fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Repo edit scope: this report only.
- Temporary probe: `/tmp/tmem_ldred_rowcol_round12_probe.py`
- Full run log: `/tmp/tmem_ldred_rowcol_round12_run.log`
- Structured results: `/tmp/tmem_ldred_rowcol_round12_results.jsonl`

## Scope

This lane adversarially fuzzed `tcgen05.ld.red` row/column/opcode behavior
after the Round 10 and Round 12 findings. The probe reused the checked-in Gluon
structural-fuzzer kernels and ran every row in a fresh subprocess with a stable
Lane U cache.

Coverage:

- `N in {2,16,32,64,128,256,512}` where the existing kernels could express the
  row;
- 1CTA direct reductions over `M=128`;
- 1CTA indexed parent and descriptor-chain reductions over `M in {128,256}`;
- 2CTA whole-parent direct control over `256x64`;
- 2CTA indexed and column-chain reductions over `M=256`;
- 2CTA row-chain reductions over `M=256`;
- row/column permutations `identity`, `reverse`, `even_odd`, and `rotate1`;
- `min` and `max` direct controls; and
- `min` plus `max(abs=True, propagate_nan=ALL)` on modifier-capable 2CTA
  indexed and column-chain rows.

The temporary probe truncated opcode lists in JSON output, but kept enough
representative opcodes for classification and confirmed PTX/LLIR opcode
agreement on every compiled row.

## Commands

Required rebuild before testing:

```bash
make -j8
```

Result: `ninja: no work to do`.

Temporary probe syntax and inventory:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_ldred_rowcol_round12_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_rowcol_round12_probe.py --list | tail -5
```

Result: `TOTAL 94`.

Full subprocess-isolated sweep:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-round12-lane-u \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_ldred_rowcol_round12_probe.py \
  2>&1 | tee /tmp/tmem_ldred_rowcol_round12_run.log
```

Raw summary:

```text
SUMMARY {"FZ-20260421-0004 opcode fallback": 19, "clean TMEM OutOfResources boundary": 6, "pass": 42, "uncategorized failure": 27}
```

The 27 raw `uncategorized failure` rows were manually classified from fresh
exact reruns and captured MLIR reproducer diagnostics:

```text
pass                                             42
FZ-20260421-0004 opcode fallback                19
FZ-20260421-0008 optimizer abort                14
FZ-20260421-0005/0009 allocator/resource        13
clean TMEM OutOfResources boundary               6
runtime miscompile                               0
new independent candidate                        0
```

Fresh representative confirmations:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-round12-lane-u-confirm-pass \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_ldred_rowcol_round12_probe.py \
  --case ldred-r12-1cta-direct-128x512-identity-min \
  > /tmp/tmem_ldred_rowcol_round12_confirm_pass.log 2>&1

CUDA_VISIBLE_DEVICES=1 \
TRITON_CACHE_DIR=/tmp/triton-cache-round12-lane-u-confirm-alloc \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_ldred_rowcol_round12_probe.py \
  --case ldred-r12-1cta-indexed-256x32-identity-min \
  > /tmp/tmem_ldred_rowcol_round12_confirm_alloc.log 2>&1

CUDA_VISIBLE_DEVICES=2 \
TRITON_CACHE_DIR=/tmp/triton-cache-round12-lane-u-confirm-rowchain \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_ldred_rowcol_round12_probe.py \
  --case ldred-r12-2cta-rowchain-256x32-even_odd-identity-min \
  > /tmp/tmem_ldred_rowcol_round12_confirm_rowchain.log 2>&1

CUDA_VISIBLE_DEVICES=3 \
TRITON_CACHE_DIR=/tmp/triton-cache-round12-lane-u-confirm-fallback \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_ldred_rowcol_round12_probe.py \
  --case ldred-r12-2cta-indexed-256x32-identity-max_abs_nan \
  > /tmp/tmem_ldred_rowcol_round12_confirm_fallback.log 2>&1
```

## Positive controls

Forty-two rows compiled, executed, matched the PyTorch reduction reference, and
emitted hardware `tcgen05.ld.red` opcodes.

Positive coverage included:

- 1CTA direct `M=128` reductions at every requested `N`, including `N=2` and
  `N=512`;
- 1CTA direct row/column permutations at `N in {32,64,128,256}`;
- 1CTA indexed `M=128` rows at `N in {2,16,32,64,128}`;
- 1CTA `M=128` row-chain and column-chain descriptor views at
  `N in {32,64,128}`; and
- the 2CTA whole-parent direct `256x64` control.

Representative positive opcode:

```text
tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32
```

For `ldred-r12-1cta-direct-128x512-identity-min`, the compiler emitted eight
`32x32b.x64.min.f32` packets and runtime output matched the input and row-wise
minimum reference.

## `FZ-20260421-0004`: opcode fallback

Nineteen rows produced correct runtime values but emitted plain `tcgen05.ld`
instead of hardware `tcgen05.ld.red`.

Rows:

- 2CTA indexed `256x{2,16,32,64,128}` for `min`;
- 2CTA indexed `256x{2,16,32,64,128}` for
  `max(abs=True, propagate_nan=ALL)`;
- 2CTA column-chain reversed-column `256x{16,32,64,128}` for both modifier
  variants; and
- one 2CTA row-chain row,
  `ldred-r12-2cta-rowchain-256x2-rotate1-even_odd-min`.

Representative fallback opcode:

```text
tcgen05.ld.sync.aligned.16x32bx2.x16.b32
```

The representative fallback row had matching PTX and LLIR opcode extraction and
no `.ld.red.` opcode. This broadens the existing opcode-selection bucket across
`max.abs.NaN` and one row-chain permutation, but the root still matches
`FZ-20260421-0004`.

## `FZ-20260421-0008`: optimizer abort

Fourteen 2CTA row-chain rows failed in
`TritonNvidiaGPUOptimizeTMemLayoutsPass`.

Rows:

- `N in {2,16,32,64,128}`;
- row/column permutations `even_odd/identity`, `reverse/identity`, and
  `rotate1/even_odd`, except the `N=2 rotate1/even_odd` row which compiled and
  fell into `FZ-20260421-0004`.

Representative diagnostic:

```text
LLVM ERROR: Dimensions must match, ignoring order, but they don't.
Got dims: ["row", "col"] and ["row", "col", "block"]
Pipeline failed while executing [`TritonNvidiaGPUOptimizeTMemLayoutsPass`]
```

This extends the known row/col/block mismatch from `N <= 64` through `N=128`
and across additional row permutations. No new bucket is warranted because the
failure pass and dimension-mismatch signature are identical to
`FZ-20260421-0008`.

## `FZ-20260421-0005/0009`: allocator/resource failure

Thirteen 1CTA `M=256` indexed or descriptor-chain rows failed in
`TritonTensorMemoryAllocationPass`.

Rows:

- 1CTA indexed `256x{2,16,32,64,128,256,512}`;
- 1CTA row-chain `256x{32,64,128}`; and
- 1CTA column-chain `256x{32,64,128}`.

Representative diagnostic:

```text
TensorMemoryAllocation.cpp:65: MemoryBitMap::findFirstFit(...):
Assertion `kNumRows - numRows >= 0' failed.
Pipeline failed while executing [`TritonTensorMemoryAllocationPass`]
```

The exact reproducer for `ldred-r12-1cta-indexed-256x32-identity-min` shows a
256-row indexed parent view reducing through a software `tt.reduce` after
`ttng.tmem_load`, then asserting in the allocator. This remains in the existing
`FZ-20260421-0005/0009` family. A later repair phase should split cleanly
between resource-impossible parent allocations and positive-sized 256-row
indexed `ld.red` rows that should lower or reject cleanly.

## Clean boundaries

Six rows reached clean tensor-memory resource diagnostics:

- 1CTA indexed `128x256` and `128x512`;
- 2CTA indexed `256x256` and `256x512` for `min`; and
- 2CTA indexed `256x256` and `256x512` for
  `max(abs=True, propagate_nan=ALL)`.

Representative diagnostic:

```text
triton.runtime.errors.OutOfResources: out of resource: tensor memory,
Required: 1024, Hardware limit: 512.
```

These are clean hardware/resource boundaries, not new backend bugs.

## New FZ decision

No new independent `FZ-*` bucket is warranted from Lane U.

The sweep produced stronger evidence for existing buckets:

- promote broader `FZ-20260421-0004` coverage for 2CTA indexed and column-chain
  `min`, `max.abs.NaN`, and the `N=2 rotate1/even_odd` row-chain fallback;
- promote broader `FZ-20260421-0008` coverage for 2CTA row-chain
  `N in {2,16,32,64,128}` and multiple row permutations; and
- promote broader `FZ-20260421-0005/0009` coverage for 1CTA `M=256`
  indexed/row-chain/column-chain reductions across narrow and wide `N`.

No runtime miscompile was observed. Backend repairs remain deferred until the
discovery campaign stops finding new failures or the user explicitly pivots
from discovery to repair.
