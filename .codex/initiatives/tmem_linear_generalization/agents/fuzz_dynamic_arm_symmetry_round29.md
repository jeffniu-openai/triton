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

## Follow-up: Chain Shape Matrix

A separate temporary probe swept the original chain0 helper and the alternate
identity chain over direct-root and indexed-root `ld/st` descriptors for a small
shape matrix.

Artifacts:

- Probe file: `/tmp/tmem_chain_shape_round29_probe.py`
- Log: `/tmp/tmem_chain_shape_round29.log`

Commands:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_chain_shape_round29_probe.py
PYTHONPATH=.:./python:./python/test/gluon pytest -q --collect-only /tmp/tmem_chain_shape_round29_probe.py
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short /tmp/tmem_chain_shape_round29_probe.py \
  2>&1 | tee /tmp/tmem_chain_shape_round29.log
```

Result:

```text
12 tests collected
8 failed, 4 passed in 5.83s
```

Rows:

| Shape | Root | Chain | Observed result | Classification |
| --- | --- | --- | --- | --- |
| `64x32` | direct | chain0 | clean compile-time unsupported row-anchor diagnostic | Existing clean boundary |
| `64x32` | direct | alternate | same clean unsupported diagnostic before the alternate view can execute | Existing clean boundary |
| `64x32` | indexed | chain0 | clean compile-time unsupported row-anchor diagnostic | Existing clean boundary |
| `64x32` | indexed | alternate | same clean unsupported diagnostic | Existing clean boundary |
| `128x32` | direct | chain0 | wrong output: `4032/4096` mismatches, max abs diff about `6.04` | Existing descriptor-view chain wrong-code family, likely adjacent to `FZ-20260421-0003` |
| `128x32` | direct | alternate | passed, `0` mismatches | Positive control |
| `128x32` | indexed | chain0 | wrong output: `4032/4096` mismatches, max abs diff about `5.45` | Existing descriptor-view chain wrong-code family |
| `128x32` | indexed | alternate | passed, `0` mismatches | Positive control |
| `128x64` | direct | chain0 | wrong output: `8064/8192` mismatches, max abs diff about `5.91` | Existing descriptor-view chain wrong-code family |
| `128x64` | direct | alternate | passed, `0` mismatches | Positive control |
| `128x64` | indexed | chain0 | wrong output: `8064/8192` mismatches, max abs diff about `6.23` | Existing descriptor-view chain wrong-code family |
| `128x64` | indexed | alternate | passed, `0` mismatches | Positive control |

Updated diagnosis:

The wrong-code is reproducible without parent indexing, without branches, and
without loop-carried SSA for the chain0 shape at `M=128`. Parent indexing does
not change the mismatch count. The alternate chain remains a strong positive
control for both direct and indexed roots at `128x32` and `128x64`. The `64x32`
cases did not produce runtime evidence because `32x32b` descriptor-view
`ld/st` correctly stops at the existing row-anchor diagnostic.

This makes the Round 29 branch/SSA rows look like amplifications of a
chain-shape-specific descriptor-view mapping problem rather than a purely
control-flow-specific bug. The branch and loop variants still matter because
they show how the same bad view can travel through generic descriptor SSA, but
the minimal wrong-code seed for this slice is now "chain0 view + direct
`ld/st`, M=128".

## Follow-up: IR/PTX Capture for Minimal Bad and Green Control

The minimal bad chain0 direct-root case and the green alternate-chain direct
control were compiled once more without assertions so their generated artifacts
could be saved for later repair work.

Artifacts:

- Capture script: `/tmp/tmem_chain_shape_round29_capture.py`
- Capture log: `/tmp/tmem_chain_shape_round29_capture.log`
- Bad chain0 direct `128x32`:
  - `/tmp/tmem_chain_shape_round29_chain0_direct_128x32.ttgir`
  - `/tmp/tmem_chain_shape_round29_chain0_direct_128x32.llir`
  - `/tmp/tmem_chain_shape_round29_chain0_direct_128x32.ptx`
- Green alternate direct `128x32`:
  - `/tmp/tmem_chain_shape_round29_alt_direct_128x32.ttgir`
  - `/tmp/tmem_chain_shape_round29_alt_direct_128x32.llir`
  - `/tmp/tmem_chain_shape_round29_alt_direct_128x32.ptx`

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_chain_shape_round29_capture.py \
  2>&1 | tee /tmp/tmem_chain_shape_round29_capture.log
```

Result:

```text
chain0_direct_128x32 asm_keys ['cubin', 'llir', 'ptx', 'source', 'ttgir']
chain0_direct_128x32 mismatch_count 4032 max_abs_diff 5.975822448730469
alt_direct_128x32 asm_keys ['cubin', 'llir', 'ptx', 'source', 'ttgir']
alt_direct_128x32 mismatch_count 0 max_abs_diff 0.0
```

PTX observation:

- Both cases emit the same public packet shape:
  `tcgen05.st.sync.aligned.32x32b.x32.b32` followed by
  `tcgen05.ld.sync.aligned.32x32b.x32.b32`.
- The difference is not opcode selection.

TTGIR observation:

- Bad chain0 direct creates a final descriptor view with row basis
  `[[64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]]` before
  `ttng.tmem_load`.
- Green alternate direct canonicalizes back to the original
  `#ttng.tensor_memory_linear` layout before `ttng.tmem_load`; the two
  opposite permutes disappear into a reshape back to the base descriptor
  layout.

Updated diagnosis:

The minimal bad case now has a concrete layout signature: a semantically
identity chain that rotates the row basis so the `64` row anchor becomes the
first basis vector is accepted for `32x32b` `ld/st`, emits the same packet shape
as the green control, and then reads back the wrong rows. This looks closer to
an accepted-but-misplanned descriptor-view layout family than to an SSA
control-flow issue.

## Follow-up: Direct Row-Basis Roundtrip Controls

A direct-layout control then tested whether the rotated row basis is inherently
bad when used as the allocation layout itself, with no descriptor view.

Artifacts:

- Probe file: `/tmp/tmem_row_basis_round29_probe.py`
- Log: `/tmp/tmem_row_basis_round29.log`

Commands:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_row_basis_round29_probe.py
PYTHONPATH=.:./python:./python/test/gluon pytest -q --collect-only /tmp/tmem_row_basis_round29_probe.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short /tmp/tmem_row_basis_round29_probe.py \
  2>&1 | tee /tmp/tmem_row_basis_round29.log
```

Result:

```text
8 passed in 8.39s
```

Rows covered:

- `128x32` and `128x64`;
- row basis orders: identity, `last_first` (`[64, 1, 2, ...]` for `M=128`),
  reverse, and even/odd;
- direct `ttng.tmem_alloc`, store, then load with the same descriptor layout.

Updated diagnosis:

The rotated row basis is not inherently broken as a direct allocation layout.
All direct row-basis roundtrips pass. The wrong-code seed requires a
cross-layout path: store through the base descriptor layout, then load through a
semantically identity descriptor-view layout whose row basis has been rotated.
That points at descriptor-view layout equivalence/materialization, not generic
row-basis support in `tcgen05.ld/st`.
