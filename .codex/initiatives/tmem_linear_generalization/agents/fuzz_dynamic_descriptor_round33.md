# Round 33 Dynamic Descriptor SSA / Control-Flow Runtime Lane

Date: 2026-04-21 13:11 UTC

Mode: discovery/cataloging only. No backend or compiler source was modified.

## Scope

This lane targeted dynamic TMEM descriptor SSA/control-flow combinations next
to, but not identical to, the committed structural fuzzer rows:

- runtime-selected `parent.index(ttgl.load(selector))`;
- same-parent distinct-index branch selection for both selector arms;
- loop-carried selected descriptors;
- direct descriptors and the chain0 reshape/permute/reshape descriptor view;
- mixed consumers across `tmem_load`, `tmem_store`, `tcgen05.copy`, hardware
  `.ld.red`, mixed load/store/`.ld.red`, plus practical plain/scaled MMAv5
  controls.

## Required Build

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

- Temporary probe: `/tmp/tmem_dynamic_descriptor_round33_probe.py`
- Logs:
  - `/tmp/tmem_dynamic_descriptor_round33_g1.log`
  - `/tmp/tmem_dynamic_descriptor_round33_g2.log`
  - `/tmp/tmem_dynamic_descriptor_round33_g3.log`
  - `/tmp/tmem_dynamic_descriptor_round33_g4.log`
  - `/tmp/tmem_dynamic_descriptor_round33_plain_mma.log`
  - `/tmp/tmem_dynamic_descriptor_round33_scaled_mma.log`

Probe validation:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_dynamic_descriptor_round33_probe.py
PYTHONPATH=.:./python:./python/test/gluon pytest -q --collect-only /tmp/tmem_dynamic_descriptor_round33_probe.py
```

Collection result: `100 tests collected`.

## Commands

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 1 /tmp/tmem_dynamic_descriptor_round33_probe.py 2>&1 | tee /tmp/tmem_dynamic_descriptor_round33_g1.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 2 /tmp/tmem_dynamic_descriptor_round33_probe.py 2>&1 | tee /tmp/tmem_dynamic_descriptor_round33_g2.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 3 /tmp/tmem_dynamic_descriptor_round33_probe.py 2>&1 | tee /tmp/tmem_dynamic_descriptor_round33_g3.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group 4 /tmp/tmem_dynamic_descriptor_round33_probe.py 2>&1 | tee /tmp/tmem_dynamic_descriptor_round33_g4.log
```

Practical MMAv5 controls:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_indexed_acc_view[f16-linear-32-32-True]' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_indexed_acc_view[f16-linear-32-32-True]' 2>&1 | tee /tmp/tmem_dynamic_descriptor_round33_plain_mma.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_scaled_mma_acc_subslice_control_flow 2>&1 | tee /tmp/tmem_dynamic_descriptor_round33_scaled_mma.log
```

## Results

Main 100-row split:

| Group | Result |
| --- | --- |
| 1 | `12 failed, 13 passed, 75 deselected` |
| 2 | `8 failed, 17 passed, 75 deselected` |
| 3 | `17 failed, 8 passed, 75 deselected` |
| 4 | `17 failed, 8 passed, 75 deselected` |

Manual classification:

| Classification | Count | Rows |
| --- | ---: | --- |
| Pass | 46 | Direct load/store/`ld.red`/mixed rows for constant, same-branch, distinct-branch, and loop-carried modes; direct copy constant/same-branch/loop-carried rows |
| Existing `FZ-20260421-0001` | 20 | Runtime-index direct/chain load, store, `ld.red`, and mixed rows; direct runtime-index copy; direct same-parent distinct-branch copy |
| Existing `FZ-20260421-0002`/`FZ-20260421-0003` descriptor-view wrong result | 24 | Chain0 load, `ld.red`, and mixed non-runtime-index rows; all report `4032/4096` output mismatches, with `ld.red`/mixed also reporting `126/128` reduction mismatches while emitting hardware `.ld.red` |
| Existing clean unsupported copy boundary | 10 | Chain0 `tcgen05.copy` rows for every selector mode; diagnostic says the shared layout maps to `tcgen05.copy.128x128b` but no compatible shared-memory descriptor plan exists for the row-permuted destination |

MMAv5 controls:

- Plain MMAv5 indexed accumulator controls: `2 passed`.
- Scaled-MMAv5 dynamic accumulator subslice sentinel:
  `1 xfailed`, existing `FZ-20260421-0007`.

## Classification

No new independent `FZ-*` bucket is proposed.

The lane broadens and reconfirms existing buckets:

- `FZ-20260421-0001`: runtime `parent.index(ttgl.load(selector))` still leaves
  illegal `ttg.memdesc_index` for load, store, copy, `ld.red`, and mixed
  consumers. Same-parent distinct-index branch selection remains copy-specific:
  direct branch load/store/`ld.red`/mixed rows pass, while direct copy rows fail
  late with illegal `ttg.memdesc_index`.
- `FZ-20260421-0002` / adjacent `FZ-20260421-0003`: chain0 descriptor-view
  read/reduction semantics are wrong even without runtime index. This lane
  adds loop-carried selected chain0 descriptors to the wrong-result evidence.
- Clean copy boundary: chain0 descriptor-view copy remains an early clean
  unsupported planner diagnostic, not a late illegal-op or runtime miscompile.
- Plain MMAv5 selected/indexed accumulator controls remain green; scaled-MMAv5
  dynamic low-subslice accumulator use remains the known `FZ-0007` xfail.

## Follow-Up Hooks

- Promote loop-carried chain0 mixed-consumer rows only after repair work begins;
  they are redundant with current `FZ-0002`/`FZ-0003` evidence during discovery.
- For `FZ-0001`, the cleanest still-interesting copy discriminator is direct
  same-parent distinct-index selection versus direct loop-carried selection:
  branch-distinct copy fails, loop-carried direct copy passes in this lane.
- Keep repair deferred until the fuzzing campaign explicitly pivots from
  discovery to fixing.
