# Round 47: Copy Descriptor Layout Variants

- Date: 2026-04-21 14:41 UTC
- Branch: `codex/tmem`
- HEAD: `f54e7c64c350`
- Scope: discovery/cataloging only. No backend/compiler code and no checked-in
  tests were modified. The only checked-in write from this lane is this report.

## Objective

Adversarially exercise TMEM copy descriptor layout variants and nearby clean
boundaries:

- linear `128x128b` and `128x256b` copy families;
- `warpx2::01_23` and `warpx2::02_13`;
- two-CTA copy;
- indexed, subslice, and slice-index descriptor views;
- scales copy and scaled-MMA setup copies;
- dense shared-layout rematerialization;
- clean unsupported copy/resource boundaries.

Classification was against existing copy-adjacent buckets:

- `FZ-20260421-0001`: dynamic/control-flow-carried TMEM memdesc values can
  leave `ttg.memdesc_index` live into late lowering;
- `FZ-20260421-0020` and `FZ-20260421-0021`: half-row/half-column descriptor
  view boundaries;
- clean hardware/resource boundaries for copy instruction footprint, two-CTA
  ownership, and representable shared-memory descriptor plans.

No temporary reproducers were needed; this lane used checked-in tests only.

## Required build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Lane A: runtime-matrix positive copy variants

File:

```text
python/test/gluon/test_tmem_runtime_matrix.py
```

Selector:

```text
((cp_no_scales or cp_scales) and not reports and not unsupported and not resource and not clean)
```

Collection:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  pytest --collect-only -q \
  -k '((cp_no_scales or cp_scales) and not reports and not unsupported and not resource and not clean)' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result:

```text
246/1615 tests collected (1369 deselected) in 1.93s
```

Split-4 run pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  -k '((cp_no_scales or cp_scales) and not reports and not unsupported and not resource and not clean)' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Shard results:

- Group 1 / GPU 0: `58 passed, 4 skipped, 1553 deselected` in `3.78s`
- Group 2 / GPU 1: `62 passed, 1553 deselected` in `3.45s`
- Group 3 / GPU 2: `62 passed, 1553 deselected` in `4.18s`
- Group 4 / GPU 3: `60 passed, 1555 deselected` in `4.32s`

Aggregate:

```text
242 passed, 4 skipped
```

Coverage notes:

- single-CTA linear copy matrices over `128x128b` and `128x256b`;
- linear indexed-view and subslice-view copies, including full-width indexed
  contrasts;
- two-CTA linear indexed/subslice copies and `128x128b` two-CTA codegen;
- `warpx2::01_23` and `warpx2::02_13` single-CTA positives;
- `warpx2` subslice, indexed, and slice-index descriptor views;
- `warpx2::01_23` two-CTA positives;
- dense shared-layout rematerialization for both `warpx2` families, plus
  two-CTA `01_23`;
- scales copy, direct two-CTA scales copy, and scaled-MMAv5 copy setup rows.

Classification:

- No compiler crash.
- No runtime wrong-result signal.
- No unexpected unsupported diagnostic in the positive lane.
- No new evidence for `FZ-20260421-0001`; the checked-in static descriptor
  view copies stay green. The known dynamic/control-flow-yielded linear copy
  evidence from Round 45 remains the minimal `FZ-0001` copy sentinel.
- No new copy-specific bucket.

## Lane B: runtime-matrix clean copy boundaries

File:

```text
python/test/gluon/test_tmem_runtime_matrix.py
```

Selector:

```text
((cp_no_scales or cp_scales) and (reports or unsupported or resource or clean))
```

Collection:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  pytest --collect-only -q \
  -k '((cp_no_scales or cp_scales) and (reports or unsupported or resource or clean))' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result:

```text
67/1615 tests collected (1548 deselected) in 1.95s
```

Split-4 run pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  -k '((cp_no_scales or cp_scales) and (reports or unsupported or resource or clean))' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Shard results:

- Group 1 / GPU 0: `17 passed, 1598 deselected` in `8.73s`
- Group 2 / GPU 1: `17 passed, 1598 deselected` in `6.70s`
- Group 3 / GPU 2: `17 passed, 1598 deselected` in `5.97s`
- Group 4 / GPU 3: `16 passed, 1599 deselected` in `4.90s`

Aggregate:

```text
67 passed
```

Coverage notes:

- transposed/shared-subslice invalid copy diagnostics;
- `4x256b` unsupported copy boundaries;
- full-footprint `128x256`/`256x256` TMEM resource boundaries;
- two-CTA `warpx2::02_13` candidate, subslice, indexed, and slice-index clean
  unsupported rows;
- `warpx2` subword clean errors;
- row-permuted `warpx2` destination clean unsupported row;
- two-CTA layout in a four-CTA context clean ownership error;
- noncanonical two-CTA block clean unsupported row;
- legacy subword, exotic linear, tile-permuted subinstruction, and row/column
  permuted linear copy clean unsupported rows.

Classification:

- Clean diagnostics stayed stable and did not degrade into assertions,
  `PassManager::run failed`, or runtime miscompares.
- The two-CTA `warpx2::02_13` rows remain clean hardware/descriptor-plan
  boundaries rather than false unsupported cases.
- The full-width copy rows remain resource-boundary diagnostics.
- No evidence for `FZ-20260421-0020` or `FZ-20260421-0021` drift in copy rows;
  this selector did not expose a half-row/half-column copy-specific failure.
- No new copy-specific bucket.

## Lane C: structural copy scales guardrail

File:

```text
python/test/gluon/test_tmem_structural_fuzzer.py
```

Collection:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  pytest --collect-only -q \
  -k 'test_tmem_structural_fuzzer_copy_scales' \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result:

```text
2/33 tests collected (31 deselected) in 1.78s
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  -k 'test_tmem_structural_fuzzer_copy_scales' \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result:

```text
2 passed, 31 deselected in 3.11s
```

Classification:

- The structural scales-copy rows still emit matching PTX/LLIR copy opcodes.
- No new structural fuzzer failure was exposed.

## Lane D: older `test_core.py` copy controls

File:

```text
python/test/gluon/test_core.py
```

Selector:

```text
(test_tmem_copy_2d or test_tmem_copy_no_scales_shared_linear_128x128b or test_tmem_copy_no_scales or test_tmem_copy_no_scales_matrix)
```

Collection:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  pytest --collect-only -q \
  -k '(test_tmem_copy_2d or test_tmem_copy_no_scales_shared_linear_128x128b or test_tmem_copy_no_scales or test_tmem_copy_no_scales_matrix)' \
  python/test/gluon/test_core.py
```

Result:

```text
50/18114 tests collected (18064 deselected) in 2.49s
```

Split-4 run pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  -k '(test_tmem_copy_2d or test_tmem_copy_no_scales_shared_linear_128x128b or test_tmem_copy_no_scales or test_tmem_copy_no_scales_matrix)' \
  python/test/gluon/test_core.py
```

Shard results:

- Group 1 / GPU 0: `13 passed, 18101 deselected` in `9.88s`
- Group 2 / GPU 1: `13 passed, 18101 deselected` in `9.76s`
- Group 3 / GPU 2: `13 passed, 18101 deselected` in `10.87s`
- Group 4 / GPU 3: `6 passed, 5 skipped, 18103 deselected` in `12.47s`

Aggregate:

```text
45 passed, 5 skipped
```

Classification:

- Older copy controls agree with the runtime-matrix lane.
- Skips remained limited to the resource-heavy matrix tail.
- No crash, wrong-result signal, or opcode drift.

## Overall classification

Round 47 did not find a new TMEM-copy backend bug.

The strongest copy-specific active issue remains the already-classified
`FZ-20260421-0001` dynamic/control-flow-yielded linear `ttng.tmem_copy`
sentinel from Round 45. Static linear descriptor views, `warpx2` descriptor
views, dense shared rematerialization, scales copy, and two-CTA copy layouts
all stayed green in checked-in runtime coverage.

The clean unsupported rows still look like hardware/ISA or descriptor-plan
boundaries:

- two-CTA `warpx2::02_13` cannot be synthesized with a correct shared-memory
  source plan under the tested schedules;
- noncanonical two-CTA ownership layouts are rejected cleanly;
- copy atoms that require a full instruction footprint remain resource or
  footprint boundaries rather than late compiler failures;
- exotic row/column permutations report clean incompatibilities instead of
  crashing.

No backend fix was started, per the continuous fuzzing campaign rule.
