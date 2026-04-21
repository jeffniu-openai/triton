# Round 17 Lane AP: copy/ldst mixed runtime fuzzing

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend/compiler code was changed. No
commit or push was made by this lane.

## Scope

Lane AP fuzzed mixed TMEM no-scales copy plus TMEM `ld/st` readback surfaces,
away from scaled-MMAv5:

- copy into direct TMEM, then immediate TMEM load/store/load mutation;
- copy into descriptor-chain TMEM, then descriptor-chain load/store/load;
- legal single-region 2CTA copy plus load/store/load;
- two independent legal 2CTA copy regions with readbacks as a contrast against
  `FZ-20260421-0014`;
- an int8 subword copy boundary row.

The lane classified against existing `FZ-20260421-0001`,
`FZ-20260421-0003`, `FZ-20260421-0010`, and `FZ-20260421-0014`. No new
independent `FZ-*` candidate was found.

## Required rebuild

Command:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Temporary probe

Probe file:

```text
/tmp/tmem_copy_ldst_mixed_round17_probe.py
```

Per-row logs:

```text
/tmp/tmem_copy_ldst_mixed_round17_ap001.log
/tmp/tmem_copy_ldst_mixed_round17_ap002.log
/tmp/tmem_copy_ldst_mixed_round17_ap003.log
/tmp/tmem_copy_ldst_mixed_round17_ap004.log
/tmp/tmem_copy_ldst_mixed_round17_ap005.log
/tmp/tmem_copy_ldst_mixed_round17_ap006.log
/tmp/tmem_copy_ldst_mixed_round17_ap007.log
/tmp/tmem_copy_ldst_mixed_round17_all.log
/tmp/tmem_copy_ldst_mixed_round17_all2.log
```

Collect-only command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_copy_ldst_mixed_round17_probe.py
```

Initial result: `5 tests collected`. After the `FZ-0014` wait/readback
follow-up rows were added, the same collect command selected `7 tests`.

Aggregate runtime command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_copy_ldst_mixed_round17_probe.py
```

Initial result: `5 passed in 15.34s`. After adding `AP-006` and `AP-007`, the
aggregate rerun passed as `7 passed in 7.88s`; `AP-007` prints the known
proxy-fence reproducer and classifies it as existing `FZ-0014`.

## Temporary case matrix

| Case | Focus | Result | Classification |
| --- | --- | --- | --- |
| `AP-001` | 1CTA `warpx2::01_23` no-scales copy into direct TMEM, then `load -> +5 -> store -> load` | passed, opcode `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b`, output matched expected copy permutation plus `5` | green mixed copy+ld/st control |
| `AP-002` | 1CTA `warpx2::01_23` no-scales copy into a lifted descriptor-chain view, then descriptor-chain `load -> +7 -> store -> load` | passed, opcode `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b`, output matched expected copy permutation plus `7` | green descriptor-chain copy+ld/st control; no `FZ-0003` reproduction |
| `AP-003` | legal 2CTA `warpx2::01_23` no-scales copy into direct TMEM, then `load -> +11 -> store -> load` | passed, opcode `tcgen05.cp.cta_group::2.warpx2::01_23.64x128b`, output matched expected 2CTA copy permutation plus `11` | green 2CTA mixed copy+ld/st control |
| `AP-004` | two independent legal 2CTA `warpx2::01_23` copy regions with separate mbarriers and readbacks | passed, emitted two `tcgen05.cp.cta_group::2.warpx2::01_23.64x128b` ops, both outputs matched expected 2CTA copy permutations | green contrast against `FZ-0014`; this exact readback/ordering shape did not reproduce proxy-fence insertion failure |
| `AP-005` | int8 subword copy boundary with `N=16` and `NVMMASharedLayout(swizzle_byte_width=32)` | clean frontend/shared-layout diagnostic: contiguous dimension too small for swizzle byte size | clean boundary, not a backend failure |
| `AP-006` | two independent legal 2CTA `warpx2::01_23` copy regions with separate mbarriers, waits, and no TMEM readback | passed, emitted two `tcgen05.cp.cta_group::2.warpx2::01_23.64x128b` ops, scalar status store completed | green contrast against `FZ-0014`; region count alone is not sufficient |
| `AP-007` | two independent legal 2CTA `warpx2::01_23` copy regions, but wait the first mbarrier before starting the second region and then read back both regions | compiler failed in proxy-fence insertion with `could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses`; pytest classified it as `AP007_CLASS=existing_fz0014` | existing `FZ-20260421-0014`, narrowed trigger |

## Checked-in adjacent baseline

Collect-only command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and (warpx2 or twocta) and not reports) or (ldst_descriptor and not reports)'
```

Result: `240/1615` selected.

Runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_copy_ldst_mixed_round17_checked_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and (warpx2 or twocta) and not reports) or (ldst_descriptor and not reports)'
```

Results:

- group 1 / GPU 0: `29 passed, 31 skipped, 1555 deselected in 4.42s`;
- group 2 / GPU 1: `30 passed, 30 skipped, 1555 deselected in 25.12s`;
- group 3 / GPU 2: `60 passed, 1555 deselected in 13.76s`;
- group 4 / GPU 3: `60 passed, 1555 deselected in 48.23s`;
- aggregate: `179 passed, 61 skipped`.

This keeps the existing no-scales `warpx2`/2CTA copy and descriptor `ld/st`
surface green adjacent to the temporary mixed probe.

## Classification notes

No row reproduced `FZ-20260421-0001`: the probe did not carry memdesc values
through runtime scalar SSA or generic control-flow paths.

No row reproduced `FZ-20260421-0003`: descriptor-chain copy followed by
descriptor-chain `ld/st` readback (`AP-002`) produced correct runtime output.
This does not clear deeper packet-order descriptor chains already cataloged
under `FZ-0003`; it only adds a green mixed copy/readback contrast.

No row reproduced `FZ-20260421-0010`: the lane did not run 2CTA local TMEM
layouts inside 4/8/16 CTA launch contexts. Existing high-CGA clean diagnostics
remain owned by prior lanes.

`AP-007` reproduced existing `FZ-20260421-0014`: the saved log contains the
same proxy-fence insertion diagnostic, `could not find an insertion point
between cross-CTA mbarrier.init ops and tracked mbarrier uses`. The lane did
not assign a new bucket because the diagnostic and pass location match
`FZ-0014`.

`AP-004` and `AP-006` are useful negative contrasts. Both contain two
independent legal 2CTA copy regions with separate mbarriers. `AP-004` waits
both barriers after both commits and then reads back both regions; `AP-006`
does not read TMEM back at all. Both compile and execute correctly. `AP-007`
waits the first region before allocating/initializing the second mbarrier, then
reads both regions, and that ordering reproduces `FZ-0014`. This narrows the
failure toward proxy-fence interval tracking across sequential cross-CTA
mbarrier regions, rather than simply region count, readback presence, or lack
of readback.

## Final result

- New independent `FZ-*` candidates: `0`.
- Runtime miscompiles: `0`.
- Unexpected unsupported diagnostics: `0`.
- Compiler crashes: `1` reproduced existing `FZ-0014`.
- Clean boundary diagnostics: `1` subword shared-layout row.
- Backend/compiler repairs: `0`.

Recommended next discovery slice: continue around `FZ-0014` by varying wait
placement, invalidate placement, readback presence, and descriptor-chain
destinations in the two-region 2CTA copy reproducer. `AP-004`, `AP-006`, and
`AP-007` suggest the proxy-fence insertion failure depends on the exact tracked
mbarrier use interval across sequential regions rather than simply on the count
of 2CTA copy regions.
