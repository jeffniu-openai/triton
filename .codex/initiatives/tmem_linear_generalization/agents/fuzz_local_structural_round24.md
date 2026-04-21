# Round 24 Local: Checked-In Structural Fuzzer Guardrail

Date: 2026-04-21 13:55 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

Reran the checked-in TMEM structural fuzzer rows, excluding the explicit
allocator/optimizer crash repro tests, while Round 24 generator work continued.
This validates that the current known-xfail inventory and green structural rows
still behave as expected.

Selector:

```text
not allocator_crash and not optimizer_crash
```

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'not allocator_crash and not optimizer_crash'
```

Result:

```text
30/33 tests collected
```

Split-4 runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <1..4> \
  --store-durations --durations-path /tmp/tmem_r24_structural_durations.json \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'not allocator_crash and not optimizer_crash'
```

Logs:

```text
/tmp/tmem_r24_structural_g1.log
/tmp/tmem_r24_structural_g2.log
/tmp/tmem_r24_structural_g3.log
/tmp/tmem_r24_structural_g4.log
```

## Results

Aggregate result: `9 passed, 21 xfailed`.

Per-shard results:

- GPU 0 / group 1: `5 passed, 25 deselected, 3 xfailed`.
- GPU 1 / group 2: `2 passed, 25 deselected, 6 xfailed`.
- GPU 2 / group 3: `2 passed, 25 deselected, 6 xfailed`.
- GPU 3 / group 4: `27 deselected, 6 xfailed`.

The xfailed rows replayed existing expected buckets, including dynamic
`ttg.memdesc_index` reaching LLVM conversion for `FZ-20260421-0001`; no xfail
unexpectedly passed.

## Classification

No new independent `FZ-*` bucket was found.  The checked-in structural fuzzer
inventory remains stable:

- green rows still pass;
- known `FZ-20260421-0001`/`0002`/`0003`/`0004`/`0006`/`0007` rows remain
  expected xfails;
- no additional compiler crash or runtime miscompile appeared outside the
  current inventory.
