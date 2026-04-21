# Round 10 Lane J: TMEM descriptor-view composition-depth fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Temporary probe: `/tmp/tmem_descriptor_depth_round10_probe.py`
- Full run log: `/tmp/tmem_descriptor_depth_round10_run.log`

## Scope

This lane probed deeper `ld/st` and `ld.red` descriptor-view composition
chains with immediate readback, focusing on:

- `reshape / permute / reshape` chains;
- double-transpose and slice/slice variants;
- rank-4 and rank-5 parent descriptors;
- sibling-view coexistence on the same parent;
- `ld/st` and `ld.red` immediate readback after the final view selection.

The probe used subprocess isolation per row so compiler crashes, unsupported
diagnostics, and resource ceilings would not terminate the whole sweep.

## Probe Commands

Required rebuild before tests:

```bash
make -j8
```

Probe syntax check:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_descriptor_depth_round10_probe.py
```

Full sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-round10-lane-j \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_descriptor_depth_round10_probe.py 2>&1 | tee /tmp/tmem_descriptor_depth_round10_run.log
```

Direct reruns for the ambiguous rows:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-round10-lane-j \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_descriptor_depth_round10_probe.py --child r10-ldst-r5-sibling-rpr

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-round10-lane-j \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_descriptor_depth_round10_probe.py --child r10-ldred-r5-sibling-rpr
```

## Results Summary

The final sweep split into:

- `7` passes;
- `3` clean diagnostics / clean resource ceilings; and
- `0` new compiler crashes, verifier failures, or miscompiles.

The probe did not discover a new independent `FZ-*` bucket.

## Case Table

| Case | Shape / chain | Outcome | Notes |
| --- | --- | --- | --- |
| `r10-ldst-r3-direct` | `[2,128,64]`, direct `index(1)` | pass | `tcgen05.st` / `tcgen05.ld` parity held. |
| `r10-ldst-r3-rpr` | `[2,64,32]`, `reshape/permute/reshape` | pass | `tcgen05.st` / `tcgen05.ld` parity held. |
| `r10-ldst-r3-double-transpose` | `[2,64,32]`, double transpose + slice | pass | `tcgen05.st` / `tcgen05.ld` parity held. |
| `r10-ldst-r4-slice-mix` | `[2,2,64,32]`, slice/transpose mix | clean diagnostic | `ttng.tmem_store` rejected the source view as an unsupported register layout; row anchors `32,64` are not directly representable. |
| `r10-ldst-r5-sibling-rpr` | `[2,2,2,64,32]`, sibling views + `reshape/permute/reshape` | clean diagnostic | Same unsupported direct `tcgen05.ld/st` register-layout boundary as the rank-4 slice row. |
| `r10-ldred-r3-direct` | `[2,128,32]`, direct `index(1)` | pass | Hardware `tcgen05.ld.red` path matched PTX/LLIR. |
| `r10-ldred-r3-rpr` | `[2,128,32]`, `reshape/permute/reshape` | pass | Hardware `tcgen05.ld.red` path matched PTX/LLIR. |
| `r10-ldred-r3-double-transpose` | `[2,128,32]`, double transpose + slice | pass | Hardware `tcgen05.ld.red` path matched PTX/LLIR. |
| `r10-ldred-r4-slice-mix` | `[2,2,128,32]`, slice/transpose mix | pass | Hardware `tcgen05.ld.red` path matched PTX/LLIR. |
| `r10-ldred-r5-sibling-rpr` | `[2,2,2,128,32]`, sibling views + `reshape/permute/reshape` | clean resource ceiling | `OutOfResources: tensor memory`, required `2048`, hardware limit `512`. This is a probe-sizing boundary, not a compiler regression. |

## Classification Notes

### `ld/st` deeper chains

The direct and deep-chain `ld/st` rows stayed green:

- rank-3 direct and `reshape/permute/reshape` readback;
- rank-3 double-transpose readback;
- rank-4 and rank-5 parent descriptors with immediate readback on sibling
  or deep descendant views.

I did not recover the known `FZ-20260421-0003` packet-order symptom from this
probe. The deeper rows either passed or stopped at the existing unsupported
direct-layout boundary.

### `ldred` deeper chains

The direct, reshaped, double-transposed, and rank-4 slice-mix `ld.red` rows
all compiled, executed, and matched the expected reduction results with
matching PTX/LLIR `tcgen05.ld.red` opcodes.

The rank-5 sibling row did not reach compiler lowering because the probe shape
exceeded the tensor-memory budget and failed with a clean `OutOfResources`
diagnostic.

## New FZ Decision

No new non-overlapping `FZ-*` id is warranted from this lane.

The only non-pass rows were clean boundaries:

- unsupported direct `ld/st` register-layout materialization for the rank-4
  and rank-5 descriptor views; and
- an `OutOfResources` ceiling on the rank-5 sibling `ld.red` probe.

Those are useful guardrails for the descriptor-depth search space, but they do
not constitute new backend regressions.
