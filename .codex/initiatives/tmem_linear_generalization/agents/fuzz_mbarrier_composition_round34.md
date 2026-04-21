# Round 34: mbarrier/proxy-fence composition around TMEM operations

Date: 2026-04-21 13:17 UTC
Branch: `codex/tmem`
HEAD: `5f44cbe65`
Mode: discovery/cataloging only. No backend/compiler code was changed.

## Summary

This lane exercised adversarial Python/Gluon runtime composition around
cross-CTA mbarriers and TMEM operations:

- sequential 2CTA no-scale `tcgen05.copy.warpx2::01_23` followed by 2CTA
  `cp.scales` `tcgen05.copy.warpx4`;
- the reverse `cp.scales` then no-scale copy ordering;
- an init-all contrast that mixes no-scale copy, `cp.scales`, plain mbarrier,
  and direct TMEM load/store;
- a wait-each chain mixing no-scale copy, direct TMEM load/store, a plain
  mbarrier interval, and another no-scale copy.

No new independent `FZ-*` bucket was found. Three sequential wait-each rows
reproduced the existing `FZ-20260421-0014` proxy-fence insertion diagnostic.
The init-all mixed contrast passed and emitted the expected 2CTA no-scale copy
plus 2CTA scale-copy opcodes. The checked-in adjacent
`cp_scales`/mbarrier/proxy/clean-boundary selector stayed green as `55 passed`.

## Required rebuild

```bash
make -j8
```

Result: no work to do.

## Temporary probe

Probe:

```text
/tmp/tmem_mbarrier_composition_round34_probe.py
```

Syntax and collection:

```bash
python3 -m py_compile /tmp/tmem_mbarrier_composition_round34_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_mbarrier_composition_round34_probe.py
```

Collection result: `4 tests collected`.

Runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group <group> \
  /tmp/tmem_mbarrier_composition_round34_probe.py
```

Split results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `1 passed, 3 deselected`; row classified `existing_fz0014` |
| 2 | 1 | `1 passed, 3 deselected`; row classified `existing_fz0014` |
| 3 | 2 | `1 passed, 3 deselected`; row classified `green` |
| 4 | 3 | `1 passed, 3 deselected`; row classified `existing_fz0014` |

## Row table

| Case | Composition | Result | Classification |
| --- | --- | --- | --- |
| `R34-001` | 2CTA no-scale `warpx2::01_23` copy; wait; then 2CTA `cp.scales` `warpx4`; wait; read both TMEM outputs | proxy-fence insertion diagnostic | existing `FZ-20260421-0014` |
| `R34-002` | 2CTA `cp.scales` `warpx4`; wait; then 2CTA no-scale `warpx2::01_23`; wait; status store only | proxy-fence insertion diagnostic | existing `FZ-20260421-0014` |
| `R34-003` | initialize no-scale copy, scale-copy, and plain mbarriers first; run no-scale copy, `cp.scales`, plain arrive; wait all; direct TMEM load/store/readback | passed with exact output | green contrast |
| `R34-004` | no-scale copy; wait; direct TMEM load/store; plain mbarrier arrive/wait; second no-scale copy; wait | proxy-fence insertion diagnostic | existing `FZ-20260421-0014` |

`R34-003` emitted:

```text
tcgen05.cp.cta_group::2.warpx2::01_23.64x128b
tcgen05.cp.cta_group::2.warpx4.32x128b
tcgen05.cp.cta_group::2.warpx4.32x128b
```

## Checked-in adjacent selector

Collect-only:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_scales or mbarrier or proxy or clean_error or clean_unsupported) and not reports and not resource'
```

Result: `55/1615` collected.

Runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_scales or mbarrier or proxy or clean_error or clean_unsupported) and not reports and not resource'
```

Results:

- group 1 / GPU 0: `14 passed, 1601 deselected`;
- group 2 / GPU 1: `14 passed, 1601 deselected`;
- group 3 / GPU 2: `14 passed, 1601 deselected`;
- group 4 / GPU 3: `13 passed, 1602 deselected`;
- aggregate: `55 passed`.

## Classification

No new independent `FZ-*` candidate was assigned.

The lane broadens existing `FZ-20260421-0014`: the same proxy-fence insertion
failure occurs when the second sequential region is a 2CTA scale-copy region,
when the first sequential region is scale-copy, and when a direct TMEM
load/store plus plain mbarrier interval sits between two 2CTA no-scale copy
regions.

`R34-003` keeps the known green contrast intact: initializing all relevant
mbarriers before using any of the regions lets mixed no-scale copy,
`cp.scales`, direct load/store, and a plain mbarrier coexist in one legal
`num_ctas=2` kernel.

No `FZ-20260421-0010` ownership diagnostic was observed in this lane; all
temporary rows used a legal `num_ctas=2` launch context. No runtime wrong
result, opcode mismatch, false unsupported diagnostic, or new compiler crash
shape was observed.
