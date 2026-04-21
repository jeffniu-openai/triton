# Round 13 Lane Y: FPSAN MMAv5 Runtime Descriptor Selection

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Repo edit scope: this report only.
- Temporary reducer reused: `/tmp/tmem_plain_mma_runtime_index_round12_probe.py`
- Result logs:
  - `/tmp/tmem_fpsan_mma_round13_results.jsonl`
  - `/tmp/tmem_fpsan_mma_round13_repeat.jsonl`

## Scope

This lane broadened Lane S's report-only `FZ-20260421-0011` around
FPSAN/instrumentation interactions with plain MMAv5 runtime accumulator
descriptor selection. The goal was to distinguish a general FPSAN/MMAv5 issue
from a narrower instrumentation interaction with runtime `memdesc_index`.

Axes probed:

- FPSAN on/off contrast for the same runtime selector rows;
- lifted rank-3 linear parents, legacy parents, and direct 2D linear parents;
- selector `0` and `1`;
- `use_acc=False` and `use_acc=True`;
- `N/K` variants around the Lane S minimal row;
- constexpr-index and dynamic-slice green controls;
- explicit reshape and preinit perturbations;
- imported-helper contrast through the older Round 10 dynamic probe; and
- checked-in plain MMAv5 and scaled-MMAv5 FPSAN tests.

## Commands

Required rebuild before testing:

```bash
make -j8
```

Result: `ninja: no work to do`.

Temporary reducer syntax check:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_plain_mma_runtime_index_round12_probe.py
```

Checked-in plain-MMAv5 FPSAN collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_fpsan.py -k 'tcgen05_mma and not scaled'
```

Result: `21/104` collected.

Checked-in plain-MMAv5 FPSAN split run:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  python/test/gluon/test_fpsan.py -k 'tcgen05_mma and not scaled'
```

Result by group: `6 passed`, `2 passed, 4 skipped`, `5 passed, 1 skipped`,
and `3 passed`; aggregate `16 passed, 5 skipped`.

Checked-in scaled-MMAv5 FPSAN collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_fpsan.py -k 'tcgen05_mma_scaled and not unsupported'
```

Result: `15/104` collected.

Checked-in scaled-MMAv5 FPSAN split run:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  python/test/gluon/test_fpsan.py \
  -k 'tcgen05_mma_scaled and not unsupported'
```

Result by group: `4 passed`, `4 passed`, `4 passed`, and `3 passed`;
aggregate `15 passed`.

Temporary 16-row contrast grid:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
TRITON_CACHE_DIR=/tmp/triton-cache-round13-fpsan-<case-cache> \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_plain_mma_runtime_index_round12_probe.py --case '<json-case>' \
  | tee -a /tmp/tmem_fpsan_mma_round13_results.jsonl
```

Fresh-process repeat and imported-helper contrast:

```bash
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((i % 4)) \
  TRITON_CACHE_DIR=/tmp/triton-cache-round13-fpsan-repeat-$i \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_plain_mma_runtime_index_round12_probe.py --case \
  '{"case_id":"r13-repeat-fpsan-runtime-lifted-n32-k128-sel1-acc0-'$i'","fpsan":true,"mode":"runtime_index","n":32,"k":128,"selector":1,"use_acc":false,"parent_kind":"lifted_linear"}'
done | tee /tmp/tmem_fpsan_mma_round13_repeat.jsonl

CUDA_VISIBLE_DEVICES=3 \
TRITON_CACHE_DIR=/tmp/triton-cache-round13-fpsan-imported \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_plain_mma_runtime_index_round12_probe.py --case \
'{"case_id":"r13-imported-helper-fpsan-runtime-lifted-n64-k128-sel1-acc0","fpsan":true,"mode":"runtime_index","n":64,"k":128,"selector":1,"use_acc":false,"parent_kind":"linear","import_rtm":true}' \
| tee -a /tmp/tmem_fpsan_mma_round13_repeat.jsonl
```

## Result Counts

Temporary 16-row contrast grid:

```text
8  FZ-20260421-0011 runtime miscompile
6  pass
2  FZ-20260421-0001 compiler failure by manual classification
```

The temporary JSON classifier printed those two compiler failures as
`compiler_failure` because the Python exception string was only
`PassManager::run failed`, but stderr contained the exact illegal
`ttg.memdesc_index` diagnostic.

Fresh-process repeat and imported-helper contrast:

```text
4  FZ-20260421-0011
```

Checked-in control coverage:

```text
plain MMAv5 FPSAN controls:   16 passed, 5 skipped
scaled MMAv5 FPSAN controls:  15 passed
```

## Case Table

| Case | Outcome | Classification |
| --- | --- | --- |
| `r13-default-runtime-lifted-n32-k128-sel1-acc0` | compiler failure | `FZ-20260421-0001` |
| `r13-fpsan-runtime-lifted-n32-k128-sel1-acc0` | `4081/4096` mismatches, `16` NaNs | `FZ-20260421-0011` |
| `r13-fpsan-runtime-lifted-n32-k128-sel0-acc0` | `4081/4096` mismatches, `16` NaNs | `FZ-20260421-0011` |
| `r13-fpsan-runtime-lifted-n32-k128-sel1-acc1` | `4083/4096` mismatches, `18` NaNs | `FZ-20260421-0011` |
| `r13-fpsan-runtime-lifted-n32-k64-sel1-acc0` | `4067/4096` mismatches, `14` NaNs | `FZ-20260421-0011` |
| `r13-fpsan-runtime-lifted-n64-k128-sel1-acc0` | `8141/8192` mismatches, `31` NaNs | `FZ-20260421-0011` |
| `r13-default-runtime-legacy-n32-k128-sel1-acc0` | pass | pass |
| `r13-fpsan-runtime-legacy-n32-k128-sel1-acc0` | `4081/4096` mismatches, `16` NaNs | `FZ-20260421-0011` |
| `r13-default-runtime-direct2d-n32-k128-sel1-acc0` | pass | pass |
| `r13-fpsan-runtime-direct2d-n32-k128-sel1-acc0` | `4081/4096` mismatches, `16` NaNs | `FZ-20260421-0011` |
| `r13-default-constexpr-lifted-n32-k128-sel1-acc0` | pass | pass |
| `r13-fpsan-constexpr-lifted-n32-k128-sel1-acc0` | pass | pass |
| `r13-default-dynamic-slice-n32-k128-sel1-acc0` | pass | pass |
| `r13-fpsan-dynamic-slice-n32-k128-sel1-acc0` | pass | pass |
| `r13-fpsan-runtime-lifted-n32-k128-sel1-acc0-reshape` | compiler failure | `FZ-20260421-0001` |
| `r13-fpsan-runtime-lifted-n32-k128-sel1-acc0-preinit` | `4081/4096` mismatches, `16` NaNs | `FZ-20260421-0011` |

Repeat rows:

- `r13-repeat-fpsan-runtime-lifted-n32-k128-sel1-acc0-{0,1,2}`:
  `3/3` reproduced `4081/4096` mismatches, `16` NaNs, and finite max abs diff
  `3.3029999366871882e+38`.
- `r13-imported-helper-fpsan-runtime-lifted-n64-k128-sel1-acc0`:
  reproduced through the older imported helper with `8153/8192` mismatches.

## Findings

No new independent `FZ-*` candidate is warranted.

The `FZ-20260421-0011` failure is not a broad FPSAN/MMAv5 failure:

- checked-in direct/plain FPSAN MMAv5 controls passed or skipped for expected
  coverage boundaries;
- checked-in scaled-MMAv5 FPSAN controls passed all `15` collected rows;
- constexpr parent selection passes with and without FPSAN; and
- dynamic selection between two static slice views passes with and without
  FPSAN.

The preserving condition is runtime outer descriptor selection by
`parent.index(ttgl.load(selector_ptr))` when FPSAN is enabled. Under FPSAN, this
runtime descriptor selection executes and produces NaN-heavy wrong results
across lifted-linear, legacy, and direct-linear parent shapes. Without FPSAN,
legacy and direct-linear runtime-index rows pass, while the lifted-linear
runtime-index row follows the known `FZ-20260421-0001` illegal
`ttg.memdesc_index` lowering path.

Two boundary refinements:

- `use_acc=True` is affected, but not required.
- An explicit no-op `reshape((M, N))` after runtime `memdesc_index` does not
  preserve the FPSAN runtime miscompile in this reducer; it falls back to the
  known illegal `ttg.memdesc_index` lowering failure.

## Representative Diagnostics

Default lifted-linear runtime index:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
Pipeline failed while executing [`ConvertTritonGPUToLLVM` on 'builtin.module' operation]
```

Minimal FPSAN runtime-index miscompile:

```text
r13-fpsan-runtime-lifted-n32-k128-sel1-acc0:
4081 / 4096 mismatches
16 NaNs
finite max abs diff 3.3029999366871882e+38
```

Imported-helper FPSAN contrast:

```text
r13-imported-helper-fpsan-runtime-lifted-n64-k128-sel1-acc0:
8153 / 8192 mismatches
```

## Classification

- FPSAN runtime parent-index descriptor selection feeding plain MMAv5:
  `FZ-20260421-0011`.
- Lifted-linear runtime parent-index rows that fail before runtime:
  `FZ-20260421-0001`.
- Plain MMAv5 direct/view FPSAN tests, scaled-MMAv5 FPSAN tests, constexpr
  parent-index controls, and dynamic slice controls: green controls.
- New bucket: none.

## Next Discovery Suggestions

- Probe whether FPSAN suppresses or rewrites dynamic `ttg.memdesc_index`
  differently before `ConvertTritonGPUToLLVM`, since the failing FPSAN rows'
  returned TTGIR metadata did not expose the same `memdesc_index` markers as
  the default-pipeline controls.
- If promoting a checked-in sentinel later, pin
  `r13-fpsan-runtime-lifted-n32-k128-sel1-acc0` with FPSAN explicitly and add
  adjacent green controls for constexpr index and dynamic slice selection.
