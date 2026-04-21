# Round 26 Lane: High-CGA Scaled-MMAv5 Mixed Ownership Fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler code was changed.
- Target: legal high-CGA scaled-MMAv5 kernels mixed with local 1CTA/2CTA TMEM
  descriptor-view `st`, `ld`, `ld.red`, and `tcgen05.copy` operations in
  4/8/16 CTA launch contexts.

## Summary

No new independent `FZ-*` bucket was found.

The Round 25 scaled-MMAv5 harness limitation was repaired. The probe now uses a
direct high-CGA scaled-MMAv5 setup derived from the green `test_core.py`
multicast-barrier pattern, then creates a local descriptor view from a
1CTA/2CTA `TensorMemoryLinearLayout` parent and attempts each local operation.

The scaled-MMAv5 high-CGA controls pass for `num_ctas=4`, `8`, and `16`, emit
the expected scaled-MMAv5 opcode, and match the torch matmul reference. When a
local 1CTA/2CTA TMEM descriptor-view operation is added in the same kernel, all
mixed rows fail at the known layout/context ownership gate:

```text
Layout has 1 CTAs per CGA, but the context requires 4 CTAs per CGA.
Layout has 2 CTAs per CGA, but the context requires 8 CTAs per CGA.
Layout has 2 CTAs per CGA, but the context requires 16 CTAs per CGA.
```

Classification: existing `FZ-20260421-0010`.

There were no proxy-fence crashes, PTX assembler failures, runtime wrong
results, or new scaled-MMAv5-specific miscompiles. The probe did not exercise
the `FZ-20260421-0013` scale descriptor-view operand path or the
`FZ-20260421-0015` runtime-selected direct B-scale path; scale descriptors were
direct and static so the lane stayed focused on high-CGA mixed ownership.

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
  python /tmp/tmem_high_cga_scaled_round26_probe.py \
  2>&1 | tee /tmp/tmem_high_cga_scaled_round26_probe_final.log
```

Artifacts:

```text
/tmp/tmem_high_cga_scaled_round26_probe.py
/tmp/tmem_high_cga_scaled_round26_probe_final.log
```

## Case Table

| Case | Result | Classification |
| --- | --- | --- |
| scaled-MMAv5 high-CGA control, `num_ctas=4` | Passed, scaled opcode present, torch reference matched | Green control |
| scaled-MMAv5 high-CGA control, `num_ctas=8` | Passed, scaled opcode present, torch reference matched | Green control |
| scaled-MMAv5 high-CGA control, `num_ctas=16` | Passed, scaled opcode present, torch reference matched | Green control |
| scaled-MMAv5 high-CGA + local 1CTA descriptor-view `st`, `num_ctas=4/8/16` | Compile-time CTA-count diagnostic | Existing `FZ-20260421-0010` |
| scaled-MMAv5 high-CGA + local 1CTA descriptor-view `ld`, `num_ctas=4/8/16` | Compile-time CTA-count diagnostic | Existing `FZ-20260421-0010` |
| scaled-MMAv5 high-CGA + local 1CTA descriptor-view `ld.red`, `num_ctas=4/8/16` | Compile-time CTA-count diagnostic | Existing `FZ-20260421-0010` |
| scaled-MMAv5 high-CGA + local 1CTA descriptor-view `tcgen05.copy`, `num_ctas=4/8/16` | Compile-time CTA-count diagnostic | Existing `FZ-20260421-0010` |
| scaled-MMAv5 high-CGA + local 2CTA descriptor-view `st`, `num_ctas=4/8/16` | Compile-time CTA-count diagnostic | Existing `FZ-20260421-0010` |
| scaled-MMAv5 high-CGA + local 2CTA descriptor-view `ld`, `num_ctas=4/8/16` | Compile-time CTA-count diagnostic | Existing `FZ-20260421-0010` |
| scaled-MMAv5 high-CGA + local 2CTA descriptor-view `ld.red`, `num_ctas=4/8/16` | Compile-time CTA-count diagnostic | Existing `FZ-20260421-0010` |
| scaled-MMAv5 high-CGA + local 2CTA descriptor-view `tcgen05.copy`, `num_ctas=4/8/16` | Compile-time CTA-count diagnostic | Existing `FZ-20260421-0010` |

## Harness Notes

The failed Round 25 scaled row used high-CGA scale layouts directly for both
scale operands and stopped while querying a high-CGA
`TensorMemoryScalesLayout` register layout. Round 26 avoids that setup failure
by preserving the direct high-CGA scaled-MMAv5 pattern that already works for
4CTA and by using replicated B-scale layouts for the broader 8CTA/16CTA
controls. The control rows prove the scaled-MMAv5 operation itself remains
legal before the local descriptor-view operation is introduced.

The local side uses a parent `TensorMemoryLinearLayout` with shape `[64, 64]`
and a column slice descriptor view of shape `[64, 32]`. That view is then used
as the operand for local store, load, reduction load, or `tcgen05.copy`. These
are the intended local descriptor-view ownership probes, not plain direct
TMEM-only rows.

## New Candidates

None.

