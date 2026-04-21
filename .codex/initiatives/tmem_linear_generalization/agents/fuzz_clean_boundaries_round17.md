# Round 17 Lane AQ: Clean Boundary Diagnostics

- Date: 2026-04-21
- Branch: `codex/tmem`
- HEAD at start: `bc6d74ca6`
- Mode: discovery/cataloging only. No backend/compiler code was changed. This
  lane did not commit or push.
- Temporary probe: `/tmp/tmem_clean_boundaries_round17_probe.py`
- Temporary log: `/tmp/tmem_clean_boundaries_round17_probe.log`

## Summary

No new independent `FZ-*` bucket was found.

The checked-in clean-negative/error runtime matrix remains green on current
HEAD. Focused probes around descriptor views, high-CGA CTA-count gates,
subword copy boundaries, non-f32 `ld.red` software-reduce paths, unsupported
scale descriptor shapes, and parent-view `memdesc_subslice` limitations all
compiled or failed cleanly through typed diagnostics. No probe produced a late
LLVM illegal-op signature such as unconverted `ttg.memdesc_index` or
`ttg.memdesc_subslice`.

Classification:

- Late generic memdesc SSA/index/control-flow failures are still owned by
  `FZ-20260421-0001`; this lane did not reproduce that illegal-op path.
- High-CGA CTA-count-gate diagnostics remain under `FZ-20260421-0010`; the
  checked-in clean diagnostic row passed.
- Parent-view `memdesc_subslice` limitations remained clean report-only
  boundaries, not crashes.

## Commands

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Clean-boundary inventory:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'reports_clean_unsupported or reports_clean_error or reports_tmem_oor'
```

Result:

```text
157/1615 tests collected
```

Four-GPU split run:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_r17_clean_boundaries_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'reports_clean_unsupported or reports_clean_error or reports_tmem_oor'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `40 passed, 1575 deselected` |
| 2 | 1 | `40 passed, 1575 deselected` |
| 3 | 2 | `40 passed, 1575 deselected` |
| 4 | 3 | `37 passed, 1578 deselected` |

Focused temporary subprocess probe:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python \
  python /tmp/tmem_clean_boundaries_round17_probe.py \
  | tee /tmp/tmem_clean_boundaries_round17_probe.log
```

Result:

| Bucket | Rows | Result | Late illegal op signature |
| --- | ---: | --- | --- |
| Descriptor views | 5 | passed | no |
| High-CGA context gate | 1 | passed | no |
| Subword copy boundaries | 4 | passed | no |
| Non-f32 `ld.red` software reduce | 4 | passed | no |
| Scale descriptor shapes | 3 | passed | no |
| Parent-view `memdesc_subslice` limitations | 2 | passed | no |

## Surface Notes

Descriptor-view clean negatives covered unsupported direct `ld/st` variants,
scales `ld/st` descriptor-view rejection, and the 4x256 refresh layout
diagnostic. These remained typed clean-unsupported/error rows rather than late
LLVM conversion failures.

The high-CGA row covered a two-CTA TMEM copy layout launched in a four-CTA CGA
context. It passed as the existing clean CTA-count mismatch diagnostic, so it
continues to be categorized under `FZ-20260421-0010` only when fuzz probes use
otherwise-legal instruction-local layouts in incompatible launch contexts.

Subword copy boundaries covered `warpx2` f16/i16 paths, the legacy i8 subword
copy path, and a 4x256b no-scales copy shape. All stayed as clean
unsupported/error rows.

The non-f32 `ld.red` checks covered direct and descriptor-chain f16/bf16
software reductions with min/max, abs, and NaN variants. They passed and did
not regress to an accidental hardware `.ld.red` fallback or a compiler
diagnostic.

Scale descriptor shape checks covered both `ld/st` scales descriptor-layout
rejection and scaled-MMAv5 narrow/tile-permuted scale descriptor boundaries.
They stayed clean.

The parent-view `memdesc_subslice` rows covered the checked-in block descriptor
boundary and a `test_core.py` runtime view bitcast row. Both remained clean
diagnostics for unsupported parent-view composition, not compiler crashes.

## Next Fuzzing Pressure

This lane did not find a new bug. The next clean-boundary pass should combine
the same negative surfaces with dynamic memdesc SSA selection and multiple
consumers, because the current green rows mostly use direct/static descriptor
values and therefore do not stress the `FZ-20260421-0001` class.
