# Lane C round 6: ld.red optimizer crash isolation

- Time: 2026-04-21 09:03 UTC
- Branch/HEAD: `codex/tmem` at `60475efa3`
- Lane: Round 6 Lane C
- Mode: discovery only; no compiler/backend repairs attempted

## Scope

Focused on the report-only R5-A `ld.red` optimizer crash family for row/col
chain1 two-CTA indexed reduction-load variants. The goal was to keep this
separate from the already cataloged plain `ld` opcode fallback
(`FZ-20260421-0004`) and from clean unsupported row-anchor/resource
boundaries.

Probed:

- `M/N`: `256x2`, `256x32`, `256x64`, and `128x32`;
- row/col kinds: `even_odd/identity`, `identity/identity`,
  `identity/reverse`;
- operations: `load_min`, `load_max`, `load_min(abs=True)`, and
  NaN-propagating `load_min`;
- chain shapes: chain1 row/CTA reshape-transpose-reshape,
  chain2 column identity shuffle, and chain3 transpose/slice identity.

## Artifacts

- Crash-safe parent pytest probe:
  `/tmp/tmem_ldred_crash_round6_probe.py`
- File-backed child program materialized by the probe:
  `/tmp/tmem_ldred_crash_round6_child.py`
- Parent classification log:
  `/tmp/tmem_ldred_crash_round6_probe.log`
- Direct minimized crash log:
  `/tmp/tmem_ldred_crash_round6_min_256x2_evenodd.log`
- Extracted MLIR reproducer:
  `/tmp/tmem_ldred_crash_round6_min_256x2_evenodd.mlir`
- `triton-opt --run-reproducer` log:
  `/tmp/tmem_ldred_crash_round6_triton_opt.log`

## Commands

Required rebuild before pytest:

```bash
CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8
```

Result: `ninja: no work to do`.

Probe syntax, collection, and subprocess-isolated classification:

```bash
PYTHONPATH=.:./python python -m py_compile /tmp/tmem_ldred_crash_round6_probe.py
PYTHONPATH=.:./python pytest --collect-only -q /tmp/tmem_ldred_crash_round6_probe.py
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r6c-gpu0 PYTHONPATH=.:./python pytest -s --tb=short /tmp/tmem_ldred_crash_round6_probe.py 2>&1 | tee /tmp/tmem_ldred_crash_round6_probe.log
```

Results:

- collect-only found `9` nodeids;
- parent pytest passed as `9 passed`;
- every candidate ran in a child Python process, so optimizer failures stayed
  isolated from the parent pytest process.

Direct minimized crash confirmation:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r6c-direct-min CASE_M=256 CASE_N=2 CASE_CHAIN=1 CASE_ROW=even_odd CASE_COL=identity CASE_OP=min CASE_ABS=0 CASE_NAN=0 PYTHONPATH=.:./python python /tmp/tmem_ldred_crash_round6_child.py > /tmp/tmem_ldred_crash_round6_min_256x2_evenodd.log 2>&1
```

Result: exit `1`, with:

```text
LLVM ERROR: Dimensions must match, ignoring order, but they don't.  Got dims: ["row", "col"] and ["row", "col", "block"]
Pipeline failed while executing [`TritonNvidiaGPUOptimizeTMemLayoutsPass` on 'builtin.module' operation]
```

MLIR candidate extraction and replay:

```bash
sed -n '90,176p' /tmp/tmem_ldred_crash_round6_min_256x2_evenodd.log > /tmp/tmem_ldred_crash_round6_min_256x2_evenodd.mlir
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
./bin/triton-opt /tmp/tmem_ldred_crash_round6_min_256x2_evenodd.mlir --run-reproducer > /tmp/tmem_ldred_crash_round6_triton_opt.log 2>&1
```

Result: exit `134` from an abort in
`TritonNvidiaGPUOptimizeTMemLayoutsPass::runOnOperation()` with the same
`["row", "col"]` vs `["row", "col", "block"]` dimension mismatch.

## Classification Matrix

| Case | Classification | Notes |
| --- | --- | --- |
| `256x2`, chain1, row `even_odd`, col `identity`, `min` | optimizer crash | Smallest stable repro on current `HEAD`; no runtime/opcode stage reached. |
| `256x2`, chain1, row `even_odd`, col `identity`, `max` | optimizer crash | Same owner surface; not min-specific. |
| `256x2`, chain1, row `even_odd`, col `identity`, `min(abs=True)` | optimizer crash | Same owner surface; abs modifier is not reached. |
| `256x2`, chain1, row `even_odd`, col `identity`, NaN-propagating `min` | optimizer crash | Same owner surface; NaN-propagating reducer is not reached. |
| `256x64`, chain1, row `identity`, col `reverse`, `min` | optimizer crash | Confirms R4-D/R5-A adjacent col-permuted crash row still reproduces. |
| `256x32`, chain1, row `identity`, col `identity`, `min` | plain `ld` fallback | Runtime-correct but emits `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`; belongs to `FZ-20260421-0004`, not this crash family. |
| `256x2`, chain2, row `even_odd`, col `identity`, `min` | plain `ld` fallback | Runtime-correct opcode fallback; chain1-specific for the optimizer crash. |
| `256x2`, chain3, row `even_odd`, col `identity`, `min` | plain `ld` fallback | Runtime-correct opcode fallback; no optimizer crash. |
| `128x32`, chain1, row `even_odd`, col `identity`, `min` | clean unsupported | Fails early with `TMEM layout 'auto' unsupported for descriptor view ...`; keep as clean boundary pending policy changes. |

## Finding

### R6C-LDRED-CRASH-001: chain1 two-CTA indexed ld.red row/col optimizer crash

- Extends: R4-D `R4D-LDRED-CRASH-001` and R5-A report-only optimizer-crash
  family.
- Failure class: `compiler_crash`.
- Likely owner surface:
  `TritonNvidiaGPUOptimizeTMemLayoutsPass`.
- Seed/case id:
  `ldred-r6c-crash-twocta-indexed-256x2-chain1-even_odd-min`,
  seed formula in the probe is `0x6C00 + M + N + chain`.
- Shape: parent `[2,256,2]`, selected view `[256,2]`.
- Layout: two-CTA `TensorMemoryLinearLayout`, row `even_odd`, col
  `identity`, lifted through prefix `[2]`.
- View chain:
  `parent.index(1).reshape((128,2,2)).permute([1,0,2]).reshape((256,2))`.
- Operation: store full tile, then `view.load_min()`.
- Observed: optimizer abort before runtime or opcode inspection.
- Repro command:
  ```bash
  CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r6c-direct-min CASE_M=256 CASE_N=2 CASE_CHAIN=1 CASE_ROW=even_odd CASE_COL=identity CASE_OP=min CASE_ABS=0 CASE_NAN=0 PYTHONPATH=.:./python python /tmp/tmem_ldred_crash_round6_child.py
  ```
- MLIR replay:
  `/tmp/tmem_ldred_crash_round6_min_256x2_evenodd.mlir` with
  `triton-opt --run-reproducer` aborts with the same dimension mismatch.

## Strict Xfail Assessment

A checked-in strict xfail can be made crash-safe only if it uses the same
subprocess isolation pattern as the existing allocator-assertion sentinel.
An in-process pytest xfail is not safe for this family because the minimized
MLIR replay still aborts `triton-opt` with exit `134`.

Recommended policy:

- keep this separate from `FZ-20260421-0004` plain-load opcode fallback rows;
- if promoted, add exactly one subprocess-isolated strict xfail for
  `ldred-r6c-crash-twocta-indexed-256x2-chain1-even_odd-min`;
- do not promote the `max`, `abs`, or NaN variants yet because they collapse to
  the same optimizer-pass owner surface;
- do not promote the chain2/chain3 `256x2` controls as crash sentinels because
  they are plain-load fallback rows, not optimizer crashes;
- keep the `128x32` `even_odd` row as a clean unsupported boundary unless the
  descriptor-view row-anchor policy changes.

No backend/compiler repairs were attempted.
