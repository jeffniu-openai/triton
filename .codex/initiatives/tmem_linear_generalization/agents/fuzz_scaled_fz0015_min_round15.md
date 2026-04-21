# Round 15 Local: FZ-0015 Scaled B-Scale Dynamic Selection Minimization

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler fixes were attempted.
- Temporary probe:
  - `/tmp/tmem_fz0015_min_round15_probe.py`
- Logs:
  - `/tmp/tmem_fz0015_min_round15_g1.log`
  - `/tmp/tmem_fz0015_min_round15_g2.log`
  - `/tmp/tmem_fz0015_min_round15_g3.log`
  - `/tmp/tmem_fz0015_min_round15_g4.log`

## Scope

This slice minimized Lane AM's report-only `FZ-20260421-0015` candidate around
scaled-MMAv5 direct B-scale TMEM descriptors selected by runtime control flow.

The probe tested:

- direct B-scale descriptor control;
- constexpr selection between distinct B-scale descriptors;
- runtime branch selection between distinct B-scale descriptors;
- runtime branch selection where both branches use the same descriptor object;
- loop-carried selection between distinct B-scale descriptors;
- runtime branch selection between distinct A-scale descriptors; and
- an extra live user of the selected B-scale descriptor before scaled MMA.

All scale payloads in each pair were identical clones, so either runtime branch
should compute the same result.

## Commands

Required rebuild was already run for the Round 15 fuzzing slice:

```bash
make -j8
```

Result: `ninja: no work to do`.

Probe syntax and collection:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_fz0015_min_round15_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_fz0015_min_round15_probe.py
```

Result: `10 tests collected`.

Runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group <group> \
  /tmp/tmem_fz0015_min_round15_probe.py
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `2 passed, 1 failed, 7 deselected in 7.16s` |
| 2 | 1 | `1 passed, 2 failed, 7 deselected in 7.09s` |
| 3 | 2 | `2 passed, 1 failed, 7 deselected in 6.29s` |
| 4 | 3 | `1 failed, 9 deselected in 5.57s` |

Aggregate: `5 passed`, `5 failed`.

## Case Table

| Case | Result | Classification |
| --- | --- | --- |
| `direct_control` | passed | Green direct B-scale control. |
| `b_constexpr_distinct` | passed | Green constexpr selection between distinct B-scale descriptors. |
| `b_runtime_distinct`, selector `0` | failed, `16381/16384` mismatches | `FZ-20260421-0015`. |
| `b_runtime_distinct`, selector `1` | failed, `16381/16384` mismatches | `FZ-20260421-0015`. |
| `b_runtime_same_object`, selector `1` | passed | Dynamic branch alone is not sufficient when both branches use the same descriptor object. |
| `b_loop_distinct`, selector `0` | failed, `16383/16384` mismatches | `FZ-20260421-0015`. |
| `b_loop_distinct`, selector `1` | failed, `16381/16384` mismatches | `FZ-20260421-0015`. |
| `a_runtime_distinct`, selector `0` | passed | A-scale dynamic selection control. |
| `a_runtime_distinct`, selector `1` | passed | A-scale dynamic selection control. |
| `b_runtime_distinct` with extra selected-scale user | failed, `16382/16384` mismatches | `FZ-20260421-0015`; extra user does not avoid the issue. |

All failing rows reported NaN or infinity in the greatest-difference summary,
matching Lane AM's NaN-heavy wrong-result shape.

## Classification

This minimization keeps `FZ-20260421-0015` as a distinct report-only candidate.

Observed boundaries:

- Not just "any dynamic branch": the same-object B-scale dynamic branch passed.
- Not constexpr descriptor choice: constexpr selection between distinct B-scale
  descriptors passed.
- Not generic scaled-MMAv5 or scale descriptor reuse: direct controls passed,
  and Lane AM's multi-MMA/reuse controls were green.
- Not A-scale dynamic selection in this shape: A-scale runtime branch selection
  passed for both selectors.
- Not avoided by an extra selected-scale user: the extra-user variant still
  failed.
- Not limited to structured `if`: loop-carried distinct B-scale selection also
  failed.

The current smallest failing trigger is: two distinct direct B-scale
`TensorMemoryScalesLayout` descriptors with identical payloads, selected by
runtime control flow, then used as the B-scale operand of `tcgen05_mma_scaled`.

## Next Minimize

Before backend repair, continue narrowing:

- use a single allocated parent with `index(0/1)` versus two independent B-scale
  allocations;
- test `N=64`, `N=32`, `K=128`, and `K=256`;
- test `mxfp4`/`nvfp4` formats;
- inspect TTGIR around the selected B-scale SSA value and compare direct versus
  dynamic selection lowering;
- check whether the issue is specific to B-scale row-major `[N, K/VEC]`
  orientation or to scaled-MMAv5's B-scale operand position.
