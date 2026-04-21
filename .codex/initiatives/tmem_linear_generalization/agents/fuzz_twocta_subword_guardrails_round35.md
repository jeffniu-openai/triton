# Round 35 Local: Two-CTA Proxy And Subword Guardrails

Date: 2026-04-21 13:55 UTC
Branch: `codex/tmem`
Scope: discovery-only guardrail runs. No backend or checked-in test source was
modified.

## Build

Required rebuild before this local batch:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Scale/Proxy Selector

Initial audit selector:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(mbarrier or proxy or cp_scales) and not resource'
```

Result: `34/1615` collected. The selected rows were scale-copy and
scaled-MMAv5 copy rows rather than explicit mbarrier/proxy-named tests.

Split-4 runtime result:

- Group 1: `9 passed, 1606 deselected`
- Group 2: `9 passed, 1606 deselected`
- Group 3: `9 passed, 1606 deselected`
- Group 4: `7 passed, 1608 deselected`

Aggregate: `34 passed`.

## Two-CTA Commit/Proxy-Heavy Selector

The mbarrier/proxy assertions in `test_tmem_runtime_matrix.py` are attached to
two-CTA copy/MMA tests, so this broader selector was used:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(twocta or two_cta) and (indexed_view or subslice_view or dense_shared or codegen or tma or plain_kinds) and not reports and not resource'
```

Result: `325/1615` collected.

Split-4 runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_twocta_proxy_round35_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(twocta or two_cta) and (indexed_view or subslice_view or dense_shared or codegen or tma or plain_kinds) and not reports and not resource'
```

Result:

- Group 1: `55 passed, 27 skipped, 1533 deselected`
- Group 2: `72 passed, 10 skipped, 1533 deselected`
- Group 3: `82 passed, 1533 deselected`
- Group 4: `79 passed, 1536 deselected`

Aggregate: `288 passed`, `37 skipped`.

## Subword And Non-F32 `ld.red`

Selector:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red_non_f32 or subword'
```

Result: `100/1615` collected.

Split-4 runtime result:

- Group 1: `25 passed, 1590 deselected`
- Group 2: `25 passed, 1590 deselected`
- Group 3: `25 passed, 1590 deselected`
- Group 4: `25 passed, 1590 deselected`

Aggregate: `100 passed`.

## Classification

No compiler crash, false unsupported diagnostic, opcode mismatch, runtime
miscompile, clean-boundary drift, unexpected xfail/pass transition, or new
independent `FZ-*` bucket was found.

The two-CTA selector keeps the CGA commit/proxy-fence-heavy checked-in runtime
surface stable. The subword selector keeps sub-32-bit TMEM `ld/st`, copy
boundaries, and non-f32 `ld.red` software-reduce descriptor chains stable.
