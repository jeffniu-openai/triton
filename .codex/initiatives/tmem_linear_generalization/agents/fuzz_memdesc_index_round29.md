# Round 29 FZ-0001 Memdesc Index Expansion

Date: 2026-04-21
Lane: runtime-index and branch-yielded TMEM descriptors feeding `tcgen05.copy`,
`tmem_load`, `tmem_store`, `ld.red`, and mixed consumers.
Mode: discovery/cataloging only. No backend/compiler fixes attempted.

## Setup

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Temporary probe:

```bash
/tmp/tmem_memdesc_index_round29_probe.py
```

The probe generated `40` runtime cases:

- consumers: `tmem_load`, `tmem_store`, `tcgen05.copy` followed by
  `tmem_load`, `ld.red`, and a mixed `load -> store -> ld.red` consumer;
- descriptor selection modes: constant index, same-object branch, distinct-arm
  branch, and runtime `parent.index(ttgl.load(selector))`;
- descriptor shape forms: direct indexed descriptor and a
  `reshape -> permute -> reshape` descriptor-view chain;
- selector value: runtime selector `1`, so distinct branches target index `1`
  and same-object branches target index `0`.

Probe validation:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_memdesc_index_round29_probe.py
PYTHONPATH=.:./python:./python/test/gluon pytest -q --collect-only /tmp/tmem_memdesc_index_round29_probe.py
```

Collection result: `40 tests collected`.

Main run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 1 /tmp/tmem_memdesc_index_round29_probe.py 2>&1 | tee /tmp/tmem_memdesc_index_round29_g1_rerun.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 2 /tmp/tmem_memdesc_index_round29_probe.py 2>&1 | tee /tmp/tmem_memdesc_index_round29_g2_rerun.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 3 /tmp/tmem_memdesc_index_round29_probe.py 2>&1 | tee /tmp/tmem_memdesc_index_round29_g3_rerun.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 4 /tmp/tmem_memdesc_index_round29_probe.py 2>&1 | tee /tmp/tmem_memdesc_index_round29_g4_rerun.log
```

Artifacts:

- `/tmp/tmem_memdesc_index_round29_probe.py`
- `/tmp/tmem_memdesc_index_round29_g1_rerun.log`
- `/tmp/tmem_memdesc_index_round29_g2_rerun.log`
- `/tmp/tmem_memdesc_index_round29_g3_rerun.log`
- `/tmp/tmem_memdesc_index_round29_g4_rerun.log`

## Classification Matrix

Legend:

- `PASS`: output matched the reference.
- `ILLEGAL_INDEX`: late `ConvertTritonGPUToLLVM` failure with illegal
  `ttg.memdesc_index`.
- `MISCOMPILE`: kernel compiled and ran but returned wrong data.
- `CLEAN_UNSUPPORTED`: verifier/planner rejected the case before LLVM lowering
  with a specific unsupported-layout diagnostic.

| Consumer | Descriptor form | constant index | same-object branch | distinct branch | runtime index |
| --- | --- | --- | --- | --- | --- |
| `tmem_load` | direct | PASS | PASS | PASS | ILLEGAL_INDEX |
| `tmem_load` | chain | MISCOMPILE | MISCOMPILE | MISCOMPILE | ILLEGAL_INDEX |
| `tmem_store` then readback | direct | PASS | PASS | PASS | ILLEGAL_INDEX |
| `tmem_store` then readback | chain | PASS | PASS | PASS | ILLEGAL_INDEX |
| `tcgen05.copy` then readback | direct | PASS | PASS | ILLEGAL_INDEX | ILLEGAL_INDEX |
| `tcgen05.copy` then readback | chain | CLEAN_UNSUPPORTED | CLEAN_UNSUPPORTED | CLEAN_UNSUPPORTED | CLEAN_UNSUPPORTED |
| `ld.red` | direct | PASS | PASS | PASS | ILLEGAL_INDEX |
| `ld.red` | chain | MISCOMPILE | MISCOMPILE | MISCOMPILE | ILLEGAL_INDEX |
| mixed `load -> store -> ld.red` | direct | PASS | PASS | PASS | ILLEGAL_INDEX |
| mixed `load -> store -> ld.red` | chain | MISCOMPILE | MISCOMPILE | MISCOMPILE | ILLEGAL_INDEX |

## FZ-0001 Coverage Expansion

This lane strengthens `FZ-20260421-0001` in two ways:

1. Runtime `parent.index(ttgl.load(selector))` leaves illegal
   `ttg.memdesc_index` for every direct consumer tested:
   `tmem_load`, `tmem_store`, `ld.red`, and mixed `load -> store -> ld.red`.
   The chain variants also fail with the same late illegal-index signature
   after composing the unresolved runtime index through the descriptor view.
2. `tcgen05.copy` is stricter than the other direct consumers for branch-yielded
   descriptors: the direct distinct-branch copy row fails late with illegal
   `ttg.memdesc_index` even though direct branch-yielded `load`, `store`,
   `ld.red`, and mixed rows all pass.

Representative illegal-index diagnostic:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

The direct distinct-branch copy case is especially useful because both branch
arms use constant indices (`parent.index(1)` and `parent.index(0)`), but the
merged `scf.if` memdesc still reaches `ttng.tmem_copy` and subsequent
`ttng.tmem_load` without being lowered before LLVM conversion.

## Non-FZ-0001 Findings

Descriptor-chain `tmem_load`, `ld.red`, and mixed rows miscompiled with
`4032/4096` full-output mismatches. The `ld.red` and mixed rows also reported
`126/128` reduction mismatches while emitting
`tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32`. These rows overlap the
existing descriptor-view semantic buckets (`FZ-20260421-0002` and
`FZ-20260421-0003`) rather than the illegal-index lowering bucket, because
constant-index and same-object controls fail too.

Descriptor-chain `tmem_store` roundtrips passed for constant-index,
same-object branch, and distinct-branch modes. That narrows the descriptor-view
wrong-result symptom in this probe to read/reduction consumers, not all
descriptor-chain stores.

Descriptor-chain `tcgen05.copy` rows cleanly rejected with the current hardware
planner diagnostic:

```text
The source shared layout maps to tcgen05.copy.128x128b, but Triton could not
synthesize a compatible shared-memory descriptor plan for it.
```

The diagnostic explicitly says the row-permuted destination would require a
destination-row/source-message schedule or narrower atom and is reported before
LLVM lowering. These rows are not `FZ-0001`.

## Positive Controls

The following controls passed:

- direct constant-index, same-object branch, and distinct-branch `tmem_load`;
- direct constant-index, same-object branch, and distinct-branch `tmem_store`
  with readback;
- direct constant-index and same-object branch `tcgen05.copy` with readback;
- direct constant-index, same-object branch, and distinct-branch `ld.red`;
- direct constant-index, same-object branch, and distinct-branch mixed
  `load -> store -> ld.red`.

These controls show the direct TMEM descriptor value can cross a runtime branch
for `load`, `store`, and `ld.red`, and that the illegal-index failures are tied
to unresolved runtime index or to `tcgen05.copy` consuming an `scf.if`-merged
distinct descriptor.

## Next Hooks

- Minimize the direct distinct-branch `tcgen05.copy` illegal-index case into a
  smaller reproducer; it is a clean `FZ-0001` expansion and differs from direct
  branch `load`/`store`/`ld.red` controls.
- Split descriptor-chain read/reduction miscompile cases away from the
  `FZ-0001` tracker so runtime-index lowering and packet-order semantics remain
  separately diagnosable.
- Keep this lane in discovery mode; do not fix the backend until fuzzing stops
  finding new variants.

## Selector-0 Follow-Up

Follow-up probe:

```bash
/tmp/tmem_memdesc_index_round29_selector0.py
```

Commands:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_memdesc_index_round29_selector0.py
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short /tmp/tmem_memdesc_index_round29_selector0.py 2>&1 | tee /tmp/tmem_memdesc_index_round29_selector0.log
```

Results:

| Row | Selector | Result |
| --- | --- | --- |
| direct distinct-branch `tmem_load` | `0` | PASS, mismatch count `0` |
| direct distinct-branch `ld.red` | `0` | PASS, `.ld.red` emitted, output and reduction mismatch counts `0` |
| direct distinct-branch `tcgen05.copy` | `0` | ILLEGAL_INDEX |

This confirms the direct distinct-branch copy failure is arm-independent: both
selector `0` and selector `1` reach late illegal `ttg.memdesc_index`, while
adjacent direct branch `load` and `ld.red` controls pass for both arms.

## Copy-Only Minimization

Follow-up probe:

```bash
/tmp/tmem_memdesc_index_round29_copy_min.py
```

Commands:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_memdesc_index_round29_copy_min.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short /tmp/tmem_memdesc_index_round29_copy_min.py 2>&1 | tee /tmp/tmem_memdesc_index_round29_copy_min.log
```

Results:

| Row | Selectors | Result |
| --- | --- | --- |
| copy-only constant index | `0`, `1` | PASS |
| copy-only same-object branch | `0`, `1` | PASS |
| copy-only distinct branch | `0`, `1` | ILLEGAL_INDEX |

The subsequent `tmem_load` readback is not required. A branch-yielded distinct
TMEM descriptor feeding `ttng.tmem_copy` alone is sufficient to leave illegal
`ttg.memdesc_index` at LLVM conversion. That makes the smallest runtime probe
for this `FZ-0001` expansion:

1. allocate `[2, 128, 4]` TMEM;
2. select `parent.index(1)` versus `parent.index(0)` through `scf.if`;
3. pass the merged descriptor directly to `tcgen05_copy`.
