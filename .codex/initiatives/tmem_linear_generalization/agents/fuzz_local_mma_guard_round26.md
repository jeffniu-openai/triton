# Round 26 Local: MMAv5-Adjacent Runtime Guardrail

- Date: 2026-04-21 15:20 UTC
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler code was changed.

## Summary

The focused MMAv5-adjacent runtime selector collected `337` checked-in
runtime-matrix rows and completed split-4 as `323 passed, 14 skipped`.

The selector was intentionally broad enough to include plain MMAv5 rows plus
nearby `mmav5_twocta` descriptor, ld/st, and scale-copy setup rows selected by
the `mma` marker text. It did not include report-only failures, clean
diagnostic rows, resource-boundary rows, or `mma_scaled` rows.

No compiler crash, verifier failure, runtime miscompile, opcode mismatch, or
new independent `FZ-*` bucket was found.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma and not mma_scaled and not reports and not clean and not resource'
```

Result:

```text
337/1615 tests collected (1278 deselected)
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <1..4> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma and not mma_scaled and not reports and not clean and not resource'
```

Observed split results:

```text
group 1: 71 passed, 14 skipped
group 2: 85 passed
group 3: 85 passed
group 4: 82 passed
aggregate: 323 passed, 14 skipped
```

Logs:

```text
/tmp/tmem_r26_mma_guard_g1.log
/tmp/tmem_r26_mma_guard_g2.log
/tmp/tmem_r26_mma_guard_g3.log
/tmp/tmem_r26_mma_guard_g4.log
```

## Classification

No new bucket. This run is a green guardrail for the checked-in plain-MMAv5 and
MMAv5-adjacent runtime matrix while Round 26 subagents fuzz dynamic descriptor,
verifier, and high-CGA mixed-ownership surfaces.
