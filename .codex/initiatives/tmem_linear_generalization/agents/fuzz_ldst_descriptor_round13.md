# Round 13 Local: LD/ST Descriptor-Selection Runtime Sweep

## Scope

This lane broadened runtime descriptor-selection coverage away from the
FPSAN/MMAv5-specific `FZ-20260421-0011` surface. It targeted checked-in
`ld/st` runtime-matrix rows that exercise descriptor roundtrips,
descriptor-composition chains, row/column-permuted descriptor parents,
tile-selector adjacency, two-CTA descriptor parents, and subword descriptor
chains.

This lane did not edit backend/compiler code.

## Commands

Required rebuild:

```bash
make -j8
```

Result: no-op success.

Collect-only:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst and (descriptor_roundtrip or descriptor_compositions or descriptor_chain or runtime_selector or rowcol_permuted or tile_selector) and not reports'
```

Result: `142/1615` rows collected.

Runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 --store-durations --durations-path /tmp/tmem_local_r13_ldst_descriptor_runtime_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst and (descriptor_roundtrip or descriptor_compositions or descriptor_chain or runtime_selector or rowcol_permuted or tile_selector) and not reports'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 --store-durations --durations-path /tmp/tmem_local_r13_ldst_descriptor_runtime_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst and (descriptor_roundtrip or descriptor_compositions or descriptor_chain or runtime_selector or rowcol_permuted or tile_selector) and not reports'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 --store-durations --durations-path /tmp/tmem_local_r13_ldst_descriptor_runtime_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst and (descriptor_roundtrip or descriptor_compositions or descriptor_chain or runtime_selector or rowcol_permuted or tile_selector) and not reports'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 --store-durations --durations-path /tmp/tmem_local_r13_ldst_descriptor_runtime_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst and (descriptor_roundtrip or descriptor_compositions or descriptor_chain or runtime_selector or rowcol_permuted or tile_selector) and not reports'
```

## Results

- Group 1/GPU 0: `36 passed, 1579 deselected in 9.79s`
- Group 2/GPU 1: `10 passed, 26 skipped, 1579 deselected in 36.03s`
- Group 3/GPU 2: `36 skipped, 1579 deselected in 2.26s`
- Group 4/GPU 3: `18 passed, 16 skipped, 1581 deselected in 3.18s`

Aggregate: `64 passed, 78 skipped`, no failures.

## Classification

No new `FZ-*` candidate.

This is a green-control slice for runtime descriptor selection outside the
FPSAN plain-MMAv5 path:

- descriptor roundtrips and descriptor-composition chains did not reproduce
  the FPSAN runtime-index MMAv5 NaN/mismatch signature;
- row/column-permuted `ld/st` descriptor parents did not reproduce the new
  M64 row-permuted `ld.red` unsupported-destination diagnostic from
  `FZ-20260421-0012`;
- subword descriptor-chain roundtrips did not expose copy/subword opcode or
  data mismatches;
- skipped rows were expected environment/support skips in assigned partitions,
  not compiler crashes, verifier failures, or runtime mismatches.

The FPSAN-specific plain-MMAv5 runtime selector issue remains mapped to
`FZ-20260421-0011`; this sweep does not warrant a new bucket.

## Follow-up Descriptor/X1/Subword Sweep

After the descriptor-selection slice above, I ran a broader checked-in selector
covering descriptor chains, replay rows, x1 variants, and subword ld/st rows:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst and (descriptor_chain or x1 or subword or replay)'
```

Result: `261/1615` rows collected.

Runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 --store-durations --durations-path /tmp/tmem_local_r13_ldst_descriptor_x1_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst and (descriptor_chain or x1 or subword or replay)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 --store-durations --durations-path /tmp/tmem_local_r13_ldst_descriptor_x1_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst and (descriptor_chain or x1 or subword or replay)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 --store-durations --durations-path /tmp/tmem_local_r13_ldst_descriptor_x1_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst and (descriptor_chain or x1 or subword or replay)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 --store-durations --durations-path /tmp/tmem_local_r13_ldst_descriptor_x1_durations.json python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst and (descriptor_chain or x1 or subword or replay)'
```

Results after required `make -j8` per shard:

- group 1/GPU 0: `66 passed, 1549 deselected in 27.90s`;
- group 2/GPU 1: `21 passed, 45 skipped, 1549 deselected in 42.80s`;
- group 3/GPU 2: `45 passed, 21 skipped, 1549 deselected in 56.78s`;
- group 4/GPU 3: `59 passed, 4 skipped, 1552 deselected in 15.76s`.

Aggregate: `191 passed, 70 skipped`, no failures.

The follow-up broadens the negative result across x1 f32/i32/subword rows,
two-CTA x1 descriptor chains, subword pack/unpack, descriptor-chain
roundtrips, replay rows, and rank5/higher-rank descriptor rows. It does not
change bucket classification: no new `FZ-*` candidate was found.
