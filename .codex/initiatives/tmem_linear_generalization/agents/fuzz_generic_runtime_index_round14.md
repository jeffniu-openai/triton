# Round 14 Lane AC: Generic Runtime-Index and Control-Flow Descriptor Fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only. No backend/compiler fixes were attempted.
- Repo edit scope: this report only.
- Temporary probes:
  - `/tmp/tmem_generic_runtime_index_round14_probe.py`
  - `/tmp/tmem_generic_consumers_round14_probe.py`

## Scope

This lane extended Lane W's generic descriptor-view fuzzing around dynamic
descriptor values and runtime descriptor indices. The sweep focused on:

- runtime `memdesc_index` across chain0, chain1, chain2, `N=32/64/128`, and
  load/store versus load-only consumers;
- dynamic `if` and loop-carried descriptor values;
- tuple-like and mixed tensor plus memdesc captures;
- helper-returned descriptor chains versus inline chain0 expressions;
- chain0 versus chain1/chain2 green boundaries; and
- copy and plain-MMAv5 consumers, not only ld/st-style loads.

The goal was classification. I did not promote tests or start backend repairs.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Generic runtime-index/control-flow probe:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_generic_runtime_index_round14_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_generic_runtime_index_round14_probe.py
```

The probe driver ran one fresh subprocess per row, rotating
`CUDA_VISIBLE_DEVICES={0,1,2,3}` and using stable
`TRITON_CACHE_DIR=/tmp/triton-cache-gpu{0,1,2,3}`.

Copy/MMA consumer probe:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_generic_consumers_round14_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_generic_consumers_round14_probe.py
```

Representative diagnostic reruns:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_generic_consumers_round14_probe.py \
  --worker-case '{"case_id":"r14-copy-dyn-if-direct-sel1","consumer":"copy","mode":"dynamic_if_direct","seed":44082,"chain_id":0,"selector":1,"use_acc":false,"m":128,"n":32,"k":64}' \
  2>&1 | rg -n "memdesc_index|unsupported|error:|RuntimeError|PassManager|failed to legalize|type|cannot|expected"

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_generic_consumers_round14_probe.py \
  --worker-case '{"case_id":"r14-copy-dyn-if-chain0-sel1","consumer":"copy","mode":"dynamic_if_chain","seed":44083,"chain_id":0,"selector":1,"use_acc":false,"m":128,"n":32,"k":64}' \
  2>&1 | rg -n "memdesc_index|unsupported|error:|RuntimeError|PassManager|failed to legalize|type|cannot|expected"

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_generic_consumers_round14_probe.py \
  --worker-case '{"case_id":"r14-mma-dyn-if-direct-sel1-useacc","consumer":"mma","mode":"dynamic_if_direct","seed":44100,"chain_id":0,"selector":1,"use_acc":true,"m":128,"n":64,"k":64}' \
  2>&1 | rg -n "memdesc_index|unsupported|error:|RuntimeError|PassManager|failed to legalize|type|cannot|expected|Mismatched"

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_generic_consumers_round14_probe.py \
  --worker-case '{"case_id":"r14-mma-dyn-if-chain0-sel1","consumer":"mma","mode":"dynamic_if_chain","seed":44098,"chain_id":0,"selector":1,"use_acc":false,"m":128,"n":64,"k":64}' \
  2>&1 | rg -n "memdesc_index|unsupported|error:|RuntimeError|PassManager|failed to legalize|type|cannot|expected|Mismatched"
```

Checked-in generic-pass selector:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'
```

Result: `11/33` collected; split result `11 xfailed` with shard counts
`3`, `3`, `3`, and `2`.

Adjacent runtime-matrix collection around descriptor composition/selector rows:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'runtime_selector or descriptor_roundtrip or descriptor_compositions'
```

Result: `116/1615` collected.

## Results Summary

No new independent `FZ-*` bucket is warranted from this lane.

- Generic ld/st-style probe: `27` rows total:
  - `10` pass;
  - `6` `FZ-20260421-0001`;
  - `11` `FZ-20260421-0002`.
- Copy/MMA consumer probe: `12` rows total:
  - `2` pass direct/static controls;
  - `8` `FZ-20260421-0001`;
  - `2` clean unsupported / hardware-layout diagnostic rows.
- Checked-in generic-pass selector: `11 xfailed`, matching the existing
  strict sentinel set.

## Generic Probe Case Table

| Case family | Rows | Outcome |
| --- | ---: | --- |
| chain0 dynamic `if`, including inline chain0 and `N=32/64/128` | 5 | `FZ-20260421-0002` runtime wrong results. |
| chain1/chain2 dynamic `if` | 4 | pass. |
| chain0 mixed tensor+memdesc and tuple capture | 2 | `FZ-20260421-0002` runtime wrong results. |
| chain1/chain2 mixed capture | 2 | pass. |
| chain0 layout pressure | 2 | `FZ-20260421-0002` runtime wrong results. |
| chain1/chain2 layout pressure | 2 | pass. |
| chain0 loop-carried values | 2 | `FZ-20260421-0002` runtime wrong results. |
| chain1/chain2 loop-carried values | 2 | pass. |
| runtime `parent.index(load(selector))` load/store | 4 | `FZ-20260421-0001`. |
| runtime `parent.index(load(selector))` load-only | 2 | `FZ-20260421-0001`. |

Representative mismatch counts for `FZ-0002` remain stable:

- `r14-dyn-if-chain0-sel0-32x32`: `8064 / 8192` mismatches.
- `r14-dyn-if-chain0-sel0-16x128`: `16128 / 16384` mismatches.
- `r14-inline-chain0-sel0-32x32-n32`: `4031 / 4096` mismatches.
- `r14-layout-pressure-chain0-32x32-n32`: `4032 / 4096` mismatches.
- `r14-loop-chain0-sel0-loops1`: `8063 / 8192` mismatches.

The narrow green boundary remains sharp: chain1/chain2 variants of the same
dynamic `if`, loop, mixed-capture, and layout-pressure families passed.

## Copy/MMA Consumer Table

| Case | Outcome | Classification |
| --- | --- | --- |
| `r14-copy-static-direct-control` | pass | Static direct copy control. |
| `r14-copy-runtime-index-sel0` | compiler failure | `FZ-20260421-0001`. |
| `r14-copy-runtime-index-sel1` | compiler failure | `FZ-20260421-0001`. |
| `r14-copy-dyn-if-direct-sel1` | compiler failure | `FZ-20260421-0001`. |
| `r14-copy-dyn-if-chain0-sel1` | clean diagnostic | Source shared layout cannot synthesize compatible `tcgen05.copy.128x256b` descriptor plan. |
| `r14-copy-dyn-if-chain1-sel0` | compiler failure | `FZ-20260421-0001`. |
| `r14-mma-static-direct-control` | pass | Static direct MMAv5 accumulator control. |
| `r14-mma-runtime-index-sel0` | compiler failure | `FZ-20260421-0001`. |
| `r14-mma-runtime-index-sel1-useacc` | compiler failure | `FZ-20260421-0001`. |
| `r14-mma-dyn-if-chain0-sel1` | clean diagnostic | Noncanonical chain0 row basis is not MMAv5-compatible. |
| `r14-mma-dyn-if-chain1-sel0` | compiler failure | `FZ-20260421-0001`. |
| `r14-mma-dyn-if-direct-sel1-useacc` | compiler failure | `FZ-20260421-0001`. |

The important new classification detail is that copy and MMA consumers broaden
`FZ-0001`: even branch-selected descriptor values with static `parent.index(0)`
and `parent.index(1)` operands are materialized as `ttg.memdesc_index` values
that survive to LLVM conversion.

## Representative Diagnostics

### `FZ-20260421-0001`

Runtime descriptor indexing still reaches LLVM conversion as an illegal op:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
Pipeline failed while executing [`ConvertTritonGPUToLLVM` on 'builtin.module' operation]
RuntimeError: PassManager::run failed
```

The copy dynamic-if direct rerun showed the same failure even though the source
program selected between static branch operands:

```text
%19 = ttg.memdesc_index %result[%c1_i32]
%19 = ttg.memdesc_index %result[%c0_i32]
```

The MMA dynamic-if direct rerun matched this shape for an accumulator memdesc:

```text
%40 = ttg.memdesc_index %result[%c1_i32]
%40 = ttg.memdesc_index %result[%c0_i32]
```

This makes `FZ-0001` broader than explicit `parent.index(load(selector))`: it
also covers generic pass/control-flow paths that merge memdesc values before a
consumer.

### `FZ-20260421-0002`

Chain0 generic descriptor-view rows still compile and then produce large,
deterministic wrong-value sets. The pattern holds across dynamic `if`, inline
chain0, mixed tensor+memdesc captures, tuple-like captures, layout-conversion
pressure, and loop-carried descriptor values. Nearby chain1/chain2 controls
passed.

### Clean Copy Diagnostic

The copy chain0 consumer reported a clean front-end diagnostic:

```text
'ttng.tmem_copy' op The source shared layout maps to tcgen05.copy.128x256b,
but Triton could not synthesize a compatible shared-memory descriptor plan for it.
This is reported as cleanly unsupported instead of falling through to late LLVM lowering.
```

I am not assigning this as a new bucket from this lane. It is a clean
unsupported diagnostic for the specific shared-memory descriptor plan used by
the temporary probe, not a proven false negative.

### Clean MMA Diagnostic

The MMA chain0 consumer reported a clean MMAv5 layout diagnostic:

```text
'ttng.tc_gen5_mma' op return operand must have a MMAv5-compatible tensor memory layout
...
first noncanonical in-tile basis is row input bit 0 for candidate tile 64x8,
got physical delta [64, 0] but the canonical delta is [1, 0]
```

This also is not promoted as a new bucket. It is consistent with the existing
MMAv5 instruction-tile order requirement and distinguishes a hardware/layout
boundary from the chain0 ld/st miscompile family.

## Classification

- `FZ-20260421-0001`: expands to copy and MMAv5 consumers and to dynamic
  branch-selected memdesc values, not only explicit runtime `parent.index(load)`.
- `FZ-20260421-0002`: remains the owner for chain0 generic
  control-flow/helper/mixed-capture/layout-pressure runtime wrong results.
- `FZ-20260421-0011`: not expanded here. The custom plain-MMAv5 runtime-index
  rows did not reach the report-only FPSAN-specific runtime-miscompile surface;
  they failed earlier as `FZ-0001`.
- Clean hardware/layout diagnostics: copy chain0 descriptor-chain consumer and
  MMA chain0 descriptor-chain consumer.
- New roots: none.

## Follow-Up Candidates

1. During the eventual repair phase, treat `FZ-0001` as a general memdesc SSA
   control-flow/index lowering gap, not just a dynamic integer indexing gap.
2. Keep chain1/chain2 generic-pass controls as narrow green boundaries for the
   `FZ-0002` repair.
3. If copy descriptor-chain support is later broadened, rerun the copy chain0
   consumer with a descriptor plan known to be hardware-realizable before
   deciding whether it is a false unsupported diagnostic.
4. Do not assign a new `FZ-*` id from this lane.
