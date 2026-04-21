# Round 54 Dynamic 2CTA Descriptor SSA Lane

Date: 2026-04-21 15:21 UTC
Branch: `codex/tmem`
Checkpoint base: `ac0791c80`

This lane targeted the Round 53 generator gap: directly crossing dynamic
descriptor SSA/control-flow selection with legal two-CTA positive TMEM
load/store layouts. No backend repairs were attempted.

## Preflight

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Probe

Temporary pytest probe:

```text
/tmp/tmem_round54_dynamic_2cta_probe.py
```

Case schema:

```text
case_id, seed, kind, row_kind, col_kind, n, selector, chain_id,
instr_variant, loop_count, tuple_capture
```

The dynamic matrix used `M=256`, `num_ctas=2`, `num_warps=4`, lifted
two-CTA `TensorMemoryLinearLayout` parents with shape `[2, M, N]`, and
runtime-selected or control-flow-carried descriptor views feeding TMEM
`st`/`ld` roundtrips.

Syntax check:

```bash
PYTHONPATH=.:./python python3 -m py_compile /tmp/tmem_round54_dynamic_2cta_probe.py
```

Result: passed.

Initial focused run:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest -s --tb=short /tmp/tmem_round54_dynamic_2cta_probe.py
```

Result:

```text
3 failed, 3 passed in 22.47s
```

Focused rerun with static controls:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest -vv -s --tb=short /tmp/tmem_round54_dynamic_2cta_probe.py
```

Result:

```text
6 failed, 3 passed in 9.51s
```

## Dynamic Cases

| Case | Seed | Parameters | Result | Classification |
| --- | ---: | --- | --- | --- |
| `r54-twocta-if-block-chain0-sel1` | `0x540001` | `dynamic_if`, identity rows/cols, `N=64`, selector `1`, chain `0`, `32x32b` | wrong result, `16255/16384` mismatches, max abs diff `5.680189` | existing `FZ-20260421-0003` |
| `r54-twocta-if-mmav5-chain1-sel0` | `0x540002` | `dynamic_if`, reverse rows, identity cols, `N=128`, selector `0`, chain `1`, `16x128b` | pass | green |
| `r54-twocta-mixed-capture-evenodd-colrev` | `0x540003` | mixed descriptor/tensor capture, even/odd rows, reverse cols, `N=64`, selector `1`, chain `1`, `32x32b`, tuple capture | pass | green |
| `r54-twocta-loop-carried-rotate1-colrev` | `0x540004` | loop-carried descriptor, rotate1 rows, reverse cols, `N=64`, selector `1`, chain `0`, `16x64b`, loop count `3` | wrong result, `16256/16384` mismatches, max abs diff `5.382111` | existing `FZ-20260421-0003` |
| `r54-twocta-loop-no-take-n128` | `0x540005` | loop-carried descriptor, identity rows, reverse cols, `N=128`, selector `4`, chain `1`, `16x128b`, loop count `2` | pass | green |
| `r54-twocta-mixed-capture-sel0-n128` | `0x540006` | mixed descriptor/tensor capture, reverse rows/cols, `N=128`, selector `0`, chain `0`, `16x128b` | wrong result, `32512/32768` mismatches, max abs diff `15.051070` | existing `FZ-20260421-0003` |

## Static Controls

The failing dynamic shapes were rerun without runtime descriptor control-flow.
The static controls also miscompiled, so the Round 54 failures are not a new
dynamic-only bucket.

| Case | Seed | Parameters | Result | Classification |
| --- | ---: | --- | --- | --- |
| `r54-static-control-block-chain0` | `0x540101` | identity rows/cols, `N=64`, parent index `1`, chain `0`, `32x32b` | wrong result, `16256/16384` mismatches, max abs diff `5.574171` | existing `FZ-20260421-0003` |
| `r54-static-control-rotate1-chain0` | `0x540102` | rotate1 rows, reverse cols, `N=64`, parent index `1`, chain `0`, `16x64b` | wrong result, `16256/16384` mismatches, max abs diff `6.368280` | existing `FZ-20260421-0003` |
| `r54-static-control-rev-chain0-n128` | `0x540103` | reverse rows/cols, `N=128`, parent index `0`, chain `0`, `16x128b` | wrong result, `32511/32768` mismatches, max abs diff `5.511019` | existing `FZ-20260421-0003` |

## Classification

Aggregate: `9` rows total, `3` pass, `6` existing
`FZ-20260421-0003`, `0` new independent `FZ-*`.

No compiler crash, verifier strictness issue, false unsupported diagnostic,
opcode mismatch, or process contamination was observed. The lane expands
`FZ-20260421-0003` to legal two-CTA lifted descriptor-view chain-0 layouts and
shows that dynamic descriptor SSA/control-flow can pass for neighboring
chain-1 cases, while chain-0 failures are already present in static controls.
