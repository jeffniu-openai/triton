# Round 29 Lane: FZ-0002 Descriptor SSA Arm Symmetry

Date: 2026-04-21

Scope: temporary Python/Gluon runtime probes for `FZ-20260421-0002`
descriptor SSA wrong-output arm symmetry and same-object controls. This lane
reran selector-0 and selector-1 forms for the Round 26 nested-helper,
loop-carried, branch `ld/st`, and `ld.red` wrong-output rows, plus direct
same-object and offset-only controls. It also reran copy consumers for
`FZ-20260421-0001` symmetry. Discovery/cataloging only; no backend/compiler
code was modified.

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

- Probe file: `/tmp/tmem_dynamic_arm_symmetry_round29_probe.py`
- Logs:
  - `/tmp/tmem_dynamic_arm_symmetry_round29_exact.log`
  - `/tmp/tmem_dynamic_arm_symmetry_round29_g1.log`
  - `/tmp/tmem_dynamic_arm_symmetry_round29_g2.log`
  - `/tmp/tmem_dynamic_arm_symmetry_round29_g3.log`
  - `/tmp/tmem_dynamic_arm_symmetry_round29_g4.log`

The four split logs used `-k round29`, but because the temporary filename also
contains `round29`, they include some older Round 26 rows. The exact log is the
clean source for the table below.

Probe validation:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_dynamic_arm_symmetry_round29_probe.py
PYTHONPATH=.:./python:./python/test/gluon pytest -q --collect-only /tmp/tmem_dynamic_arm_symmetry_round29_probe.py
```

Result: `38 tests collected`, including `25` new Round 29 selector/control
rows.

Clean exact run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
    /tmp/tmem_dynamic_arm_symmetry_round29_probe.py::test_round29_dynamic_arm_symmetry_ldst \
    /tmp/tmem_dynamic_arm_symmetry_round29_probe.py::test_round29_dynamic_arm_symmetry_ldred \
    /tmp/tmem_dynamic_arm_symmetry_round29_probe.py::test_round29_dynamic_arm_symmetry_copy \
  2>&1 | tee /tmp/tmem_dynamic_arm_symmetry_round29_exact.log
```

Result: `17 failed, 8 passed in 4.65s`.

## Results

| Row | Selector / control | Observed result | Classification |
| --- | --- | --- | --- |
| `nested-helper-tuple-ldst` | selector `0` | wrong output: `4096/4096` mismatches, max abs diff about `3.85` | Existing `FZ-20260421-0002` |
| `nested-helper-tuple-ldst` | selector `1` | wrong output: `4096/4096` mismatches, max abs diff about `4.33` | Existing `FZ-20260421-0002` |
| `loop-carried-index-ldst` | selector `0`, loop count `2` | wrong output: `4032/4096` mismatches, max abs diff about `5.49` | Existing `FZ-20260421-0002` |
| `loop-carried-index-ldst` | selector `1`, loop count `2` | wrong output: `4032/4096` mismatches, max abs diff about `5.91` | Existing `FZ-20260421-0002` |
| `loop-carried-index-ldst-no-switch` | selector `2`, loop count `2` | wrong output: `4032/4096` mismatches, max abs diff about `5.56` | Existing `FZ-20260421-0002`; loop-carried descriptor var itself is enough |
| `same-object-view-ldst` | selector `0`, both arms yield same descriptor-view chain | wrong output: `4032/4096` mismatches, max abs diff about `4.76` | Existing `FZ-20260421-0002` |
| `same-object-view-ldst` | selector `1`, both arms yield same descriptor-view chain | wrong output: `4032/4096` mismatches, max abs diff about `5.00` | Existing `FZ-20260421-0002` |
| `offset-only-direct-ldst` | selector `0`, direct descriptors differ only by parent index | passed, `0` mismatches | Positive control |
| `offset-only-direct-ldst` | selector `1`, direct descriptors differ only by parent index | passed, `0` mismatches | Positive control |
| `same-object-direct-ldst` | selector `0`, both arms yield same direct descriptor | passed, `0` mismatches | Positive control |
| `same-object-direct-ldst` | selector `1`, both arms yield same direct descriptor | passed, `0` mismatches | Positive control |
| `runtime-index-ldred` | selector `0` | compiler failure in `ConvertTritonGPUToLLVM`: illegal `ttg.memdesc_index` remains | Existing `FZ-20260421-0001` |
| `runtime-index-ldred` | selector `1` | same compiler failure in `ConvertTritonGPUToLLVM` | Existing `FZ-20260421-0001` |
| `branch-view-ldred` | selector `0` | emits `tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32`, wrong output `4032/4096`, reduction mismatches `126/128` | Existing `FZ-20260421-0002` |
| `branch-view-ldred` | selector `1` | emits `tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32`, wrong output `4032/4096`, reduction mismatches `126/128` | Existing `FZ-20260421-0002` |
| `mixed-ldst-ldred` | selector `0` | emits `.ld.red`, wrong output `4032/4096`, reduction mismatches `128/128` | Existing `FZ-20260421-0002` |
| `mixed-ldst-ldred` | selector `1` | emits `.ld.red`, wrong output `4032/4096`, reduction mismatches `128/128` | Existing `FZ-20260421-0002` |
| `offset-only-direct-ldred` | selector `0`, direct descriptors differ only by parent index | passed with `.ld.red`, output and reduction mismatch counts `0` | Positive control |
| `offset-only-direct-ldred` | selector `1`, direct descriptors differ only by parent index | passed with `.ld.red`, output and reduction mismatch counts `0` | Positive control |
| `branch-copy` | selector `0` | compiler failure in `ConvertTritonGPUToLLVM`: branch-yielded `ttg.memdesc_index` remains illegal | Existing `FZ-20260421-0001` |
| `branch-copy` | selector `1` | same compiler failure in `ConvertTritonGPUToLLVM` | Existing `FZ-20260421-0001` |
| `runtime-index-copy` | selector `0` | compiler failure in `ConvertTritonGPUToLLVM`: dynamic `ttg.memdesc_index` remains illegal | Existing `FZ-20260421-0001` |
| `runtime-index-copy` | selector `1` | same compiler failure in `ConvertTritonGPUToLLVM` | Existing `FZ-20260421-0001` |
| `same-object-copy` | selector `0`, both arms yield same direct descriptor | passed, `copy mismatch_count 0` | Positive control |
| `same-object-copy` | selector `1`, both arms yield same direct descriptor | passed, `copy mismatch_count 0` | Positive control |

## Diagnosis

No new independent `FZ-*` bucket is warranted. Round 29 sharpens the existing
classification:

- `FZ-20260421-0002` is not arm-specific. Selector `0` and selector `1` both
  fail for nested-helper, loop-carried, branch `ld/st`, and branch `ld.red`
  descriptor-view consumers.
- The issue is not merely distinct-descriptor address rematerialization. The
  same-object descriptor-view branch still miscompiles on both arms even when
  the two branches yield the identical base descriptor and identical view
  chain.
- Direct descriptors crossing the same branch/SSA structures are healthy.
  Offset-only direct `ld/st` and direct `ld.red` controls passed for both
  selectors, and direct same-object `ld/st` passed for both selectors.
- The loop-carried no-switch row also fails. Even when the runtime selector is
  outside the loop update range and the initial descriptor should remain live,
  carrying the descriptor-view value through the loop-carried SSA variable is
  enough to corrupt output.
- `ld.red` wrong-output rows emit the expected `.ld.red` PTX opcode, so these
  are not `FZ-20260421-0004` opcode fallback cases.
- Copy rows continue to belong to `FZ-20260421-0001`: branch-yielded and
  dynamic-index descriptors feeding `tcgen05.copy` leave illegal
  `ttg.memdesc_index` for both selectors. Same-object direct copy passes for
  both selectors, narrowing the copy failure to unresolved descriptor
  indexing/identity rather than all memdesc values crossing `scf.if`.

The likely repair target for `FZ-0002`, after the fuzzing campaign stops
finding new variants, remains descriptor-view chain materialization across
control-flow/SSA boundaries. Direct TMEM memdesc SSA appears sound in these
controls; the miscompile begins when a reshape/permute/reshape descriptor view
chain is merged or carried.

## Next Fuzzing Hooks

- Add explicit TTGIR capture for one same-object view row and one direct
  offset-only row to compare descriptor roots after `combine-tensor-select` and
  before `TensorMemoryAllocation`.
- Keep discovery mode active; do not repair `FZ-0001` or `FZ-0002` until the
  current fuzzing campaign stops finding new variants.

## Follow-up: Same-Object `ld.red` View Controls

The temporary probe was extended with same-object descriptor-view `ld.red`
controls after the first report draft.

Commands:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_dynamic_arm_symmetry_round29_probe.py
PYTHONPATH=.:./python:./python/test/gluon pytest -q --collect-only /tmp/tmem_dynamic_arm_symmetry_round29_probe.py::test_round29_dynamic_arm_symmetry_ldred
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
    /tmp/tmem_dynamic_arm_symmetry_round29_probe.py::test_round29_dynamic_arm_symmetry_ldred \
  2>&1 | tee /tmp/tmem_dynamic_arm_symmetry_round29_ldred_same_object.log
```

Results:

```text
10 tests collected
8 failed, 2 passed in 6.00s
```

Additional rows:

| Row | Selector / control | Observed result | Classification |
| --- | --- | --- | --- |
| `same-object-view-ldred` | selector `0`, both arms yield same descriptor-view chain | emits `tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32`, wrong output `4032/4096`, reduction mismatches `126/128`, max abs diff about `4.99` | Existing `FZ-20260421-0002` |
| `same-object-view-ldred` | selector `1`, both arms yield same descriptor-view chain | emits `tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32`, wrong output `4032/4096`, reduction mismatches `126/128`, max abs diff about `5.95` | Existing `FZ-20260421-0002` |

Updated diagnosis:

Same-object descriptor-view miscompilation is shared by `ld/st` and `ld.red`.
This further weakens an address-rematerialization-only explanation: both
branches yield the same base descriptor and the same view chain, yet the merged
descriptor-view value is corrupted for reduction load as well. Direct
offset-only `ld.red` remains green for both selectors in the same run, so the
failure still points at descriptor-view chain materialization across
control-flow/SSA rather than ordinary direct memdesc selection.

## Follow-up: View-Chain Shape and No-Branch Controls

The temporary probe was extended again with:

- an alternate identity view chain:
  `reshape((M // 2, 2, N // 2, 2)) -> permute([1, 0, 3, 2]) -> permute([1, 0, 3, 2]) -> reshape((M, N))`;
- branch-selected and same-object `ld/st` controls using that alternate chain;
- no-branch `ld/st` controls for the original chain0 helper and the alternate
  chain.

Commands:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_dynamic_arm_symmetry_round29_probe.py
PYTHONPATH=.:./python:./python/test/gluon pytest -q --collect-only /tmp/tmem_dynamic_arm_symmetry_round29_probe.py::test_round29_dynamic_arm_symmetry_ldst
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
    /tmp/tmem_dynamic_arm_symmetry_round29_probe.py::test_round29_dynamic_arm_symmetry_ldst \
  2>&1 | tee /tmp/tmem_dynamic_arm_symmetry_round29_alt_view_ldst.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
    /tmp/tmem_dynamic_arm_symmetry_round29_probe.py::test_round29_dynamic_arm_symmetry_ldst \
  2>&1 | tee /tmp/tmem_dynamic_arm_symmetry_round29_no_branch_view_ldst.log
```

Key results from the second run:

```text
17 tests collected
8 failed, 9 passed in 6.56s
```

Additional rows:

| Row | Selector / control | Observed result | Classification |
| --- | --- | --- | --- |
| `branch-alt-view-ldst` | selector `0` | passed, `0` mismatches | Positive control |
| `branch-alt-view-ldst` | selector `1` | passed, `0` mismatches | Positive control |
| `same-object-alt-view-ldst` | selector `0` | passed, `0` mismatches | Positive control |
| `same-object-alt-view-ldst` | selector `1` | passed, `0` mismatches | Positive control |
| `no-branch-view-ldst` | original chain0, no branch/control-flow merge | wrong output: `4032/4096` mismatches, max abs diff about `5.68` | Existing descriptor-view chain wrong-code family, likely adjacent to `FZ-20260421-0003` |
| `no-branch-alt-view-ldst` | alternate identity chain, no branch/control-flow merge | passed, `0` mismatches | Positive control |

Updated diagnosis:

This follow-up narrows the Round 29 wrong-output rows further. The original
chain0 helper is not merely a control-flow/SSA problem: it is already wrong for
a direct no-branch `ld/st` consumer at `128x32`. Branches, tuple captures, and
loop-carried variables expose the same bad chain through more SSA shapes, but
they are not required for the base corruption. In contrast, the alternate
identity chain is green both without control flow and through branch-selected
or same-object branch values. That points away from a blanket "all descriptor
views through branches are bad" model and toward a chain-shape-specific
descriptor-view packet/layout mapping issue. The direct descriptor controls
remain green.
