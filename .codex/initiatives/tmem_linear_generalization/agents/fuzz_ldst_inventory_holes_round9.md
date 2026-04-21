# Round 9 Lane F: `ld/st` Inventory-Hole Probe

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery-only. No backend/compiler repairs attempted.
- Temporary probe: `/tmp/tmem_round9_ldst_hole_probe.py`
- Run log: `/tmp/tmem_round9_ldst_hole_probe.log`

## Scope

This follow-up used the Round 9 normalized inventory to select a compact set
of uncovered `ld/st` descriptor-view rows. The goal was to test whether the
large `ldst` inventory holes represented new backend owners or just broader
evidence for existing buckets.

The probe reused the checked-in structural fuzzer's `ld/st` descriptor-view
kernel without modifying repo test code.

## Commands

The required build was already run for Round 9 Lane E before this runnable
probe:

```bash
CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8
```

Result: `ninja: no work to do`.

Probe syntax and collection:

```bash
PYTHONPATH=.:./python:python/test/gluon python -m py_compile \
  /tmp/tmem_round9_ldst_hole_probe.py
PYTHONPATH=.:./python:python/test/gluon pytest --collect-only -q \
  /tmp/tmem_round9_ldst_hole_probe.py
```

Result: `8 tests collected`.

Probe run:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-round9-lane-f-ldst \
PYTHONPATH=.:./python:python/test/gluon pytest -q -s --tb=short \
  /tmp/tmem_round9_ldst_hole_probe.py 2>&1 | tee \
  /tmp/tmem_round9_ldst_hole_probe.log
```

Result:

```text
4 failed, 4 passed in 5.82s
```

## Results

Passed rows:

- `r9f-ldst-direct-128x64-f32-control`;
- `r9f-ldst-chain2-128x64-f32-revcol-16x64b`;
- `r9f-ldst-chain3-128x64-f32-transpose-slice`; and
- `r9f-ldst-chain2-128x32-f16-revcol-16x64b`.

Rows extending `FZ-20260421-0003`:

- `r9f-ldst-chain1-128x64-f32-identity`: `8064 / 8192`
  mismatched;
- `r9f-ldst-chain1-128x64-f32-evenrow`: `8064 / 8192`
  mismatched; and
- `r9f-ldst-chain1-128x32-f16-identity`: `4032 / 4096`
  mismatched.

Clean unsupported row:

- `r9f-ldst-chain3-64x32-f32-transpose-slice` stopped with the expected
  descriptor-view row-anchor diagnostic:
  `required row anchors 32,64 are not directly representable in the
  descriptor view`.

## Classification

No new `FZ-*` id is warranted.

The failing chain1 rows are broader evidence for the existing
`FZ-20260421-0003` descriptor-view packet-order mismatch, not a new owner. The
small transpose/slice row is a clean hardware-layout boundary under the
current public `tcgen05.ld/st` packet model.

The positive chain2 and larger transpose/slice rows are useful future repair
controls: they show that the new probe was not merely generating universally
bad descriptor views.
