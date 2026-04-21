# Round 25 Lane BL: High-CGA Mixed Ownership Fuzzing

- Date: 2026-04-21 12:25 UTC
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler code was changed.
- Target: kernels that put a legal high-CGA operation in the same Gluon kernel
  as a local 1CTA/2CTA TMEM operation while launching in 4/8/16 CTA contexts.

## Summary

No new independent `FZ-*` bucket was found.

The useful result is that same-kernel mixed ownership still fails as the known
`FZ-20260421-0010` CTA-count gate. The probe combined:

- full-CGA TMA multicast plus local 1CTA/2CTA no-scales `tcgen05.copy`;
- high-CGA 2CTA MMAv5 multicast/commit plus local 1CTA/2CTA no-scales
  `tcgen05.copy`;
- one attempted scaled-MMAv5 plus local descriptor-view row.

The TMA and MMAv5 mixed rows all fail at the local TMEM allocation with the
same diagnostic family:

```text
Layout has 1 CTAs per CGA, but the context requires 4 CTAs per CGA.
Layout has 2 CTAs per CGA, but the context requires 8 CTAs per CGA.
Layout has 2 CTAs per CGA, but the context requires 16 CTAs per CGA.
```

There were no proxy-fence crashes, PTX assembler failures, runtime wrong
results, or changed failure modes in the rows that reached the intended mixed
ownership check. The scaled-MMAv5/local-view row failed earlier in the temporary
probe setup with an unsupported high-CGA `TensorMemoryScalesLayout` register
layout query, so it is not counted as backend evidence for a new mixed
ownership bucket.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Temporary probe:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_high_cga_mixed_round25_probe.py \
  2>&1 | tee /tmp/tmem_high_cga_mixed_round25_probe_verbose.log
```

Artifacts:

```text
/tmp/tmem_high_cga_mixed_round25_probe.py
/tmp/tmem_high_cga_mixed_round25_probe.log
/tmp/tmem_high_cga_mixed_round25_probe_verbose.log
```

## Case Table

| Case | CTA context | Result | Classification |
| --- | --- | --- | --- |
| TMA multicast plus local 1CTA no-scales copy | 4 | Compile-time local allocation diagnostic: layout has 1 CTA, context requires 4 | Existing `FZ-20260421-0010` |
| TMA multicast plus local 2CTA no-scales copy | 4 | Compile-time local allocation diagnostic: layout has 2 CTAs, context requires 4 | Existing `FZ-20260421-0010` |
| TMA multicast plus local 1CTA no-scales copy | 8 | Compile-time local allocation diagnostic: layout has 1 CTA, context requires 8 | Existing `FZ-20260421-0010` |
| TMA multicast plus local 2CTA no-scales copy | 8 | Compile-time local allocation diagnostic: layout has 2 CTAs, context requires 8 | Existing `FZ-20260421-0010` |
| TMA multicast plus local 1CTA no-scales copy | 16 | Compile-time local allocation diagnostic: layout has 1 CTA, context requires 16 | Existing `FZ-20260421-0010` |
| TMA multicast plus local 2CTA no-scales copy | 16 | Compile-time local allocation diagnostic: layout has 2 CTAs, context requires 16 | Existing `FZ-20260421-0010` |
| 2CTA MMAv5 multicast/commit plus local 1CTA no-scales copy | 4 | Compile-time local allocation diagnostic: layout has 1 CTA, context requires 4 | Existing `FZ-20260421-0010` |
| 2CTA MMAv5 multicast/commit plus local 2CTA no-scales copy | 4 | Compile-time local allocation diagnostic: layout has 2 CTAs, context requires 4 | Existing `FZ-20260421-0010` |
| 2CTA MMAv5 multicast/commit plus local 1CTA no-scales copy | 8 | Compile-time local allocation diagnostic: layout has 1 CTA, context requires 8 | Existing `FZ-20260421-0010` |
| 2CTA MMAv5 multicast/commit plus local 2CTA no-scales copy | 8 | Compile-time local allocation diagnostic: layout has 2 CTAs, context requires 8 | Existing `FZ-20260421-0010` |
| scaled-MMAv5 multicast plus local indexed descriptor view | 4 | Temporary probe failed before the local ownership check: `TMEM layout 'auto' unsupported for descriptor view tensor_memory_descriptor<uint8, ... TensorMemoryScalesLayout(cga_layout=((0, 1), (0, 2)))>` | Harness/setup limitation |

## Diagnostic Notes

The mixed TMA rows prove that a legal high-CGA non-TMEM operation in the same
kernel does not change the local-TMEM failure shape. The local allocation is
still rejected before runtime.

The mixed MMAv5 rows are the stronger contrast: they place a legal high-CGA
2CTA `tcgen05.mma`/commit path before the local copy. Those rows still fail at
the local copy allocation and not in `getModuleTwoCTAs`, proxy-fence insertion,
or MMAv5 lowering. This keeps `FZ-0010` focused on the layout/context ownership
gate rather than a broader mixed-ownership backend crash.

The scaled-MMAv5 row needs a better harness if more evidence is desired. The
temporary direct-scale form queried a high-CGA B-scale TMEM register layout
before reaching the local indexed view. That setup issue should not be promoted
as a new bucket from this lane.

## New Candidates

None.

