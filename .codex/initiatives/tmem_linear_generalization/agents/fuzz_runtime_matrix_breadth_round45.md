# Round 45 Runtime-Matrix Breadth Sweep

- Date: 2026-04-21 14:35 UTC
- Branch: `codex/tmem`
- Scope: discovery/cataloging only; no backend/compiler code modified.
- Required rebuild: `make -j8`
  - Result: `ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12`
    reported `ninja: no work to do`.
- Environment correction: initial `pytest --collect-only` attempts without
  `PYTHONPATH=./python` imported an installed/stale Triton package and failed
  before collection (`ModuleNotFoundError` for checkout modules). All recorded
  collections and shard runs below used `PYTHONPATH=./python`.

## Lane A: positive descriptor/scaled/copy breadth

Purpose: combine descriptor/high-rank, scaled/scales-layout, and copy positive
runtime-matrix surfaces without using any one recent exact selector as the
entire lane. The explicit negatives keep known diagnostic/resource rows out of
the positive pass/fail oracle.

File:

`python/test/gluon/test_tmem_runtime_matrix.py`

Selector:

```text
((descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows or descriptor_chain or bscale_descriptor_view or shared_scale_descriptor_view_auto_tmem_copy or bscale_view_extra_user or scales_variant or scales_layout or cp_scales_layout_probe or scales_ldst or cp_no_scales or cp_scales) and not reports and not resource and not clean and not m64 and not non_f32)
```

Collection:

```text
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest --collect-only -q -k '<selector>' python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `452/1615` tests collected.

Split-4 run pattern:

```text
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> pytest -s --tb=short --splits 4 --group <group> -k '<selector>' python/test/gluon/test_tmem_runtime_matrix.py
```

Shard results:

- Group 1 / GPU 0: `113 passed, 1502 deselected` in `11.43s`
- Group 2 / GPU 1: `93 passed, 20 skipped, 1502 deselected` in `5.66s`
- Group 3 / GPU 2: `109 passed, 4 skipped, 1502 deselected` in `29.98s`
- Group 4 / GPU 3: `113 passed, 1502 deselected` in `5.56s`

Aggregate: `428 passed, 24 skipped`.

Classification:

- No compiler crash.
- No runtime wrong-result signal.
- No false unsupported diagnostic surfaced in the positive selector.
- No known-red diagnostic/resource rows leaked into this positive lane.
- Skip behavior was stable but nonuniform: all skips remained within the
  positive lane's explicitly selected rows, concentrated in groups 2 and 3.
- No new independent `FZ-*`.

## Lane B: isolated diagnostic/resource boundary rows

Purpose: run the adjacent known-red or clean-boundary inventory for the same
families separately from positives, checking for diagnostic drift, unexpected
passes, or failure-mode changes.

File:

`python/test/gluon/test_tmem_runtime_matrix.py`

Selector:

```text
((descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows or descriptor_chain or bscale_descriptor_view or shared_scale_descriptor_view_auto_tmem_copy or bscale_view_extra_user or scales_variant or scales_layout or cp_scales_layout_probe or scales_ldst or cp_no_scales or cp_scales) and (reports or resource or clean or m64 or non_f32))
```

Collection:

```text
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest --collect-only -q -k '<selector>' python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `103/1615` tests collected.

Split-4 run pattern:

```text
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> pytest -s --tb=short --splits 4 --group <group> -k '<selector>' python/test/gluon/test_tmem_runtime_matrix.py
```

Shard results:

- Group 1 / GPU 0: `26 passed, 1589 deselected` in `4.21s`
- Group 2 / GPU 1: `26 passed, 1589 deselected` in `8.68s`
- Group 3 / GPU 2: `26 passed, 1589 deselected` in `7.08s`
- Group 4 / GPU 3: `25 passed, 1590 deselected` in `5.48s`

Aggregate: `103 passed`.

Classification:

- Clean diagnostic/resource assertions stayed green.
- No unexpected pass/fail drift in known-red or clean-boundary rows.
- No new compiler crash or different failure signature.
- No new independent `FZ-*`.

## Lane C: compact core TMA/MMAv5/scaled-copy control

Purpose: cover core TMA/multicast, MMAv5 multicast commit, a direct scaled
multicast barrier row, and scaled-copy linear-accumulator rows without running
the overly broad `test_core.py` selector that expands through generic `copy`
and `mma` names.

File:

`python/test/gluon/test_core.py`

Selector:

```text
(test_tma_multicast_copy or test_tcgen05_mma_multicast_commit or test_tcgen05_mma_scaled_direct_multicast_barrier or test_mma_scaled_tcgen05_copy_linear_acc)
```

Collection:

```text
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest --collect-only -q -k '<selector>' python/test/gluon/test_core.py
```

Result: `20/18114` tests collected.

Split-4 run pattern:

```text
PYTHONPATH=./python CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> pytest -s --tb=short --splits 4 --group <group> -k '<selector>' python/test/gluon/test_core.py
```

Shard results:

- Group 1 / GPU 0: `5 passed, 18109 deselected` in `4.49s`
- Group 2 / GPU 1: `5 passed, 18109 deselected` in `4.49s`
- Group 3 / GPU 2: `5 passed, 18109 deselected` in `6.32s`
- Group 4 / GPU 3: `5 passed, 18109 deselected` in `5.37s`

Aggregate: `20 passed`.

Classification:

- Core TMA/multicast and scaled-copy controls stayed green.
- No skip drift.
- No compiler crash, runtime miscompare, opcode assertion drift, or new
  independent `FZ-*`.

## Selector notes and future partitioning

- The broad `test_core.py` selector `(tma and (tcgen05 or mma or copy or multicast or scaled))`
  collected `17742/18114` because `copy` and `mma` match broad helper and
  parametrized test names. Avoid that shape for practical discovery lanes.
- The compact core selector above is a useful low-cost control, but it does
  duplicate some rows covered by prior guardrails. For future breadth rounds,
  keep `test_core.py` TMA controls compact and add targeted exact function
  names rather than family words.
- The runtime-matrix positive and diagnostic/resource selectors should remain
  separate. This run found no leak from known-red/clean-boundary rows into the
  positive lane, and the separate boundary lane gives cleaner drift detection.
- If this breadth union is repeated often, store durations for the 452-row
  positive lane and reuse duration-aware splitting. Group 3 took `29.98s`,
  much longer than the other three groups despite equal selected counts.
