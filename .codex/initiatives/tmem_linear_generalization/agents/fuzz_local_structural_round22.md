# Round 22 Local: Structural Fuzzer Guardrail Sweep

Date: 2026-04-21 12:03 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes.

## Scope

This local guardrail sweep reran the checked-in structural fuzzer positive
controls and expected xfail sentinels, excluding the dedicated allocator-crash
and optimizer-crash reproducer tests. The purpose was to catch unexpected
passes or new failures after the latest branch state while keeping known crash
sentinels out of this runtime shard.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Collect-only selected `30/33` rows:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'structural_fuzzer and not allocator_crash and not optimizer_crash'
```

Split-4 runtime sweep used stable per-GPU caches and stored durations in:

```text
/tmp/tmem_r22_structural_durations.json
```

## Results

Aggregate result: `9 passed, 21 xfailed`.

Per-shard results:

- GPU 0 / group 1: `5 passed, 3 xfailed, 25 deselected`.
- GPU 1 / group 2: `2 passed, 6 xfailed, 25 deselected`.
- GPU 2 / group 3: `2 passed, 6 xfailed, 25 deselected`.
- GPU 3 / group 4: `6 xfailed, 27 deselected`.

## Classification

No unexpected pass, unexpected failure, runtime miscompile outside known
sentinels, compiler crash outside known xfail rows, or new independent `FZ-*`
bucket was found.

The xfail diagnostics observed in the run match known structural buckets,
including dynamic `memdesc_index` illegal-op cases under
`FZ-20260421-0001`. This run therefore confirms the checked-in structural
fuzzer remains stable as a guardrail while the Round 22 subagents explore
non-overlapping fuzz lanes.
