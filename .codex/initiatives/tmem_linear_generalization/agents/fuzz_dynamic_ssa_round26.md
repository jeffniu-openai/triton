# Round 26 Dynamic Descriptor SSA / Control-Flow Fuzzing

Date: 2026-04-21

Scope: temporary Python/Gluon runtime probes for TMEM descriptor SSA values
flowing through nested helpers, branch-yielded descriptors, tuple-like returns,
loop-carried vars, dynamic indices, mixed `ld/st` + `ld.red` consumers,
`tcgen05.copy`, and plain-MMAv5 accumulator consumers. This was discovery and
cataloging only; no backend/compiler fixes were attempted.

## Required Build

Command:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Probe Artifacts

- Probe file: `/tmp/tmem_dynamic_ssa_round26_probe.py`
- Logs:
  - `/tmp/tmem_dynamic_ssa_round26_g1_rerun.log`
  - `/tmp/tmem_dynamic_ssa_round26_g2_rerun.log`
  - `/tmp/tmem_dynamic_ssa_round26_g3_rerun.log`
  - `/tmp/tmem_dynamic_ssa_round26_g4_rerun.log`
  - `/tmp/tmem_dynamic_ssa_round26_copy.log`
  - Earlier setup/rerun logs:
    `/tmp/tmem_dynamic_ssa_round26_g1.log`,
    `/tmp/tmem_dynamic_ssa_round26_g2.log`,
    `/tmp/tmem_dynamic_ssa_round26_g3.log`,
    `/tmp/tmem_dynamic_ssa_round26_g4.log`,
    `/tmp/tmem_dynamic_ssa_round26_mixed_rerun.log`

Probe validation:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_dynamic_ssa_round26_probe.py
PYTHONPATH=.:./python:./python/test/gluon pytest -q --collect-only /tmp/tmem_dynamic_ssa_round26_probe.py
```

Result: `7 tests collected`.

## Commands

Four-GPU split rerun:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 1 /tmp/tmem_dynamic_ssa_round26_probe.py 2>&1 | tee /tmp/tmem_dynamic_ssa_round26_g1_rerun.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 2 /tmp/tmem_dynamic_ssa_round26_probe.py 2>&1 | tee /tmp/tmem_dynamic_ssa_round26_g2_rerun.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 3 /tmp/tmem_dynamic_ssa_round26_probe.py 2>&1 | tee /tmp/tmem_dynamic_ssa_round26_g3_rerun.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 4 /tmp/tmem_dynamic_ssa_round26_probe.py 2>&1 | tee /tmp/tmem_dynamic_ssa_round26_g4_rerun.log
```

Copy row exact rerun after adding the copy consumer:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short /tmp/tmem_dynamic_ssa_round26_probe.py::test_round26_dynamic_ssa_copy_branch_then_load 2>&1 | tee /tmp/tmem_dynamic_ssa_round26_copy.log
```

## Results

| Row | Consumer path | Observed result | Classification |
| --- | --- | --- | --- |
| `nested-helper-tuple-ldst` | branch-selected descriptor through nested helper and tuple-like return, then `tmem_load` | wrong output: `4096/4096` mismatches, max abs diff about `4.96` | Existing `FZ-20260421-0002` |
| `loop-carried-index-ldst` | descriptor view updated through loop-carried SSA, then `tmem_load` | wrong output: `4032/4096` mismatches, max abs diff about `5.11` | Existing `FZ-20260421-0002` / loop-carried descriptor SSA family |
| `runtime-index-ldred` | `parent.index(ttgl.load(selector))` feeding `load_min` | compiler failure in `ConvertTritonGPUToLLVM`: illegal `ttg.memdesc_index` remains | Existing `FZ-20260421-0001` |
| `branch-view-ldred` | branch-yielded descriptor view feeding `load_min` | emits `tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32`, but wrong output: `4032/4096` output mismatches and `126/128` reduction mismatches | Existing `FZ-20260421-0002`, not `FZ-0004` because `.ld.red` is present |
| `mixed-ldst-ldred` | same branch-yielded descriptor feeds `tmem_load` and `load_min` | emits `tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32`, but wrong output: `4032/4096` output mismatches and `128/128` reduction mismatches | Existing `FZ-20260421-0002` |
| `plain-mma-acc-branch` | branch-selected accumulator view feeds plain `tcgen05.mma` with zero A/B and `use_acc=True` | passed | Positive control for plain-MMAv5 branch-selected accumulator descriptor |
| `copy-branch-then-load` | branch-yielded descriptor feeds `tcgen05.copy`, then `tmem_load` | compiler failure in `ConvertTritonGPUToLLVM`: illegal branch-contained `ttg.memdesc_index` remains | Existing `FZ-20260421-0001` |

## Diagnosis

No new independent `FZ-*` bucket is warranted from this lane. The new copy row
is an important broadening of `FZ-20260421-0001`: even branch-yielded constant
indices can survive as illegal `ttg.memdesc_index` when the merged memdesc feeds
`ttng.tmem_copy` and then `ttng.tmem_load`. The `runtime-index-ldred` row is the
same illegal dynamic-index lowering failure for an `ld.red` consumer.

The `ld/st` and `ld.red` wrong-output rows continue to point at the already
cataloged generic memdesc SSA/control-flow miscompile family
`FZ-20260421-0002`. The `ld.red` rows are not opcode-fallback instances of
`FZ-20260421-0004`; the emitted PTX contains the expected hardware reduction
opcode. The failure is in the descriptor/view value semantics under dynamic
branching or loop-carried SSA, not in selecting `.ld.red`.

The plain-MMAv5 accumulator branch control passed, which helps narrow this
round's failures to `ld/st`, `ld.red`, and copy paths rather than all dynamic
TMEM memdesc operands.

## Next Fuzzing Hooks

- Add a copy row that uses a runtime index directly (`parent.index(selector)`)
  and compare its failure text against the branch-yielded copy row.
- Add selector-0 reruns for the wrong-output rows to confirm both branch arms
  fail symmetrically or identify arm-specific descriptor offset loss.
- Add a same-object branch control where both branch arms yield the identical
  descriptor object. That should separate generic branch merge lowering from
  distinct-descriptor address rematerialization.
- Keep discovery mode active; do not repair `FZ-0001` or `FZ-0002` until the
  current fuzz campaign stops finding new variants.

## Follow-up Controls

Additional rows were added to the same temporary probe after the first report
draft. The probe collected `10` tests after these additions.

Commands:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_dynamic_ssa_round26_probe.py
PYTHONPATH=.:./python:./python/test/gluon pytest -q --collect-only /tmp/tmem_dynamic_ssa_round26_probe.py
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short '/tmp/tmem_dynamic_ssa_round26_probe.py::test_round26_dynamic_ssa_ldst[same-object-branch-ldst]' 2>&1 | tee /tmp/tmem_dynamic_ssa_round26_same_ldst.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short /tmp/tmem_dynamic_ssa_round26_probe.py::test_round26_dynamic_ssa_copy_runtime_index_then_load 2>&1 | tee /tmp/tmem_dynamic_ssa_round26_copy_runtime_index.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short /tmp/tmem_dynamic_ssa_round26_probe.py::test_round26_dynamic_ssa_copy_same_object_branch_then_load 2>&1 | tee /tmp/tmem_dynamic_ssa_round26_copy_same_object.log
```

Results:

| Row | Consumer path | Observed result | Classification |
| --- | --- | --- | --- |
| `same-object-branch-ldst` | both branch arms yield the same descriptor object after the chain0 reshape/permute view, then `tmem_load` | wrong output: `4032/4096` mismatches, max abs diff about `6.49` | Existing `FZ-20260421-0002`; the view-chain/control-flow interaction is enough even without distinct descriptor identity |
| `copy-runtime-index-then-load` | `parent.index(ttgl.load(selector))` feeds `tcgen05.copy`, then `tmem_load` | compiler failure in `ConvertTritonGPUToLLVM`: illegal dynamic `ttg.memdesc_index` remains | Existing `FZ-20260421-0001` |
| `copy-same-object-branch-then-load` | both branch arms yield the same direct descriptor object, then `tcgen05.copy` and `tmem_load` | passed, `copy mismatch_count 0` | Positive control; copy does not fail merely because a memdesc crosses an `scf.if` |

Updated diagnosis:

- `FZ-0001` now covers both runtime-index and branch-yielded copy consumers.
- Same-object direct copy passing narrows the copy failure to unresolved
  descriptor identity/indexing, not all memdesc values crossing an `scf.if`.
- Same-object chain0 `ld/st` still miscompiling suggests the `FZ-0002`
  descriptor-view/control-flow issue is not only distinct-object address
  rematerialization. The view-chain semantics are implicated even when both
  control-flow arms name the same base descriptor object.

## Direct Descriptor Controls

The same probe was extended again with direct-no-view controls to separate
plain memdesc control flow from descriptor-view composition.

Commands:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_dynamic_ssa_round26_probe.py
PYTHONPATH=.:./python:./python/test/gluon pytest -q --collect-only /tmp/tmem_dynamic_ssa_round26_probe.py
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short '/tmp/tmem_dynamic_ssa_round26_probe.py::test_round26_dynamic_ssa_ldst[branch-direct-ldst]' 2>&1 | tee /tmp/tmem_dynamic_ssa_round26_branch_direct_ldst.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short '/tmp/tmem_dynamic_ssa_round26_probe.py::test_round26_dynamic_ssa_ldst[same-object-direct-ldst]' 2>&1 | tee /tmp/tmem_dynamic_ssa_round26_same_direct_ldst.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short '/tmp/tmem_dynamic_ssa_round26_probe.py::test_round26_dynamic_ssa_ldred[branch-direct-ldred]' 2>&1 | tee /tmp/tmem_dynamic_ssa_round26_branch_direct_ldred.log
```

Results:

| Row | Consumer path | Observed result | Classification |
| --- | --- | --- | --- |
| `branch-direct-ldst` | branch-selected direct `parent.index(0/1)` descriptor, then `tmem_load` | passed, `ldst mismatch_count 0` | Positive control |
| `same-object-direct-ldst` | both branch arms yield the same direct `parent.index(0)` descriptor, then `tmem_load` | passed, `ldst mismatch_count 0` | Positive control |
| `branch-direct-ldred` | branch-selected direct `parent.index(0/1)` descriptor, then `load_min` | passed with `tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32`; output and reduction mismatch counts both `0` | Positive control |

Updated narrowing:

`FZ-20260421-0002` is not triggered by plain branch-yielded TMEM descriptors
feeding `ld/st` or `ld.red`. The failing rows require descriptor-view
composition, such as the chain0 reshape/permute/reshape identity view, crossing
the control-flow/SSA boundary. This makes the likely repair target the
interaction between memdesc view-chain lowering and control-flow value
materialization, rather than a blanket inability to carry TMEM memdesc values
through `scf.if`.
