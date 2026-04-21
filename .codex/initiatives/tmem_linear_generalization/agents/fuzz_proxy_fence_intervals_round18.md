# Round 18 Lane AR: proxy-fence interval variants for `FZ-20260421-0014`

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend/compiler code was changed. No
commit or push was made by this lane.

## Scope

Lane AR focused on `FZ-20260421-0014`, the proxy-fence insertion failure around
legal 2CTA no-scales `tcgen05.copy` regions with independent mbarriers:

```text
could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
```

The lane varied:

- wait placement: wait both after all commits vs wait region 0 before region 1
  starts;
- invalidate placement: no invalidate vs invalidate after each wait;
- readback presence and order: no readback, read both after both waits, or read
  the first region before starting the second;
- direct TMEM destinations vs descriptor-chain destinations;
- two sequential regions vs three regions.

All failures below matched existing `FZ-20260421-0014`; no independent new
bucket was found.

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
/tmp/tmem_proxy_fence_intervals_round18_probe.py
```

The probe started from the prior Round 17 copy/ldst mixed temporary probe and
added AR-specific interval variants. Known `FZ-0014` rows catch the compiler
exception and print a row classification so the pytest run remains green while
still documenting the crash.

Syntax/collection:

```bash
python3 -m py_compile /tmp/tmem_proxy_fence_intervals_round18_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_proxy_fence_intervals_round18_probe.py
```

Result:

```text
13 tests collected in 2.95s
```

Focused AR runtime command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_proxy_fence_intervals_round18_probe.py -k 'ar00'
```

Result:

```text
AR001_CLASS=existing_fz0014
AR002_CLASS=existing_fz0014
AR003_CLASS=green
AR004_CLASS=existing_fz0014
AR005_CLASS=existing_fz0014
AR006_CLASS=green
6 passed, 7 deselected in 28.73s
```

Full temporary-probe command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_proxy_fence_intervals_round18_probe.py
```

Result:

```text
AP006_CLASS=green
AP007_CLASS=existing_fz0014
AR001_CLASS=existing_fz0014
AR002_CLASS=existing_fz0014
AR003_CLASS=green
AR004_CLASS=existing_fz0014
AR005_CLASS=existing_fz0014
AR006_CLASS=green
13 passed in 21.17s
```

## Row table

| Case | Interval shape | Result | Classification |
| --- | --- | --- | --- |
| AP-004 inherited | two independent direct 2CTA no-scales copy regions; init both mbarriers before copies; commit both; wait both; read both | passed, exact output for both regions | green contrast |
| AP-006 inherited | two independent direct 2CTA no-scales copy regions; init both before copies; commit both; wait both; no readback | passed, status store completed | green contrast |
| AP-007 inherited | direct 2CTA no-scales region 0; wait region 0; then direct region 1; wait region 1; read both after both waits | proxy-fence insertion failed with the known insertion-point diagnostic | existing `FZ-20260421-0014` |
| AR-001 | direct 2CTA no-scales region 0; wait region 0; then direct region 1; wait region 1; no readback | proxy-fence insertion failed with the known insertion-point diagnostic | existing `FZ-20260421-0014` |
| AR-002 | same as AR-001, but invalidate each mbarrier immediately after its wait | proxy-fence insertion failed with the known insertion-point diagnostic | existing `FZ-20260421-0014` |
| AR-003 | init both mbarriers before copies; commit both; wait both; invalidate both; read both after invalidation | passed, exact output for both regions | green contrast |
| AR-004 | direct region 0; wait region 0; read region 0; then direct region 1; wait/read region 1 | proxy-fence insertion failed with the known insertion-point diagnostic | existing `FZ-20260421-0014` |
| AR-005 | descriptor-chain destinations for both regions; wait region 0 before starting region 1; read both | proxy-fence insertion failed with the known insertion-point diagnostic | existing `FZ-20260421-0014` |
| AR-006 | three independent direct 2CTA no-scales regions; init all mbarriers before copies; commit all; wait all; read all | passed, exact output for all three regions | green contrast |

## Classification

No row produced a new independent `FZ-*` candidate. Every failing AR row reached
the same NVIDIA proxy-fence insertion diagnostic already cataloged as
`FZ-20260421-0014`.

The stronger narrowing from this lane is that readback is not required:
AR-001 has no TMEM readback and still reproduces `FZ-0014`. Invalidation also
does not rescue the failing shape: AR-002 fails after invalidating region 0
before region 1 is initialized.

Region count is not sufficient: AR-006 shows three independent regions can pass
when all cross-CTA mbarriers are initialized before the copy/commit/wait phase,
and AP-006 shows the same for two regions with no readback. Readback after
invalidation is also not inherently bad: AR-003 passes.

Descriptor-chain destination views are not required for the bug, but they are
not a separate failure mode either. AR-005 fails with the same `FZ-0014`
diagnostic when the sequential wait-before-next-region shape is used.

Current hypothesis: proxy-fence insertion's tracked interval model assumes it
can place the cross-CTA proxy fence between a group of relevant `mbarrier.init`
operations and their tracked uses. Sequential cross-CTA no-scales copy regions
that wait the first mbarrier before the next region's `mbarrier.init` split that
interval in a way the pass cannot currently represent. The bug is more about
ordering/interval construction than about readbacks, descriptor-chain layout
arithmetic, invalidation, or the number of regions.

## Checked-in adjacent selector

There is no checked-in proxy-fence interval sentinel for this exact bug yet.
The closest checked-in coverage is the 2CTA no-scales copy runtime surface:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales and twocta and not reports'
```

Result:

```text
63/1615 tests collected (1552 deselected) in 2.64s
```

Runtime commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales and twocta and not reports'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales and twocta and not reports'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales and twocta and not reports'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales and twocta and not reports'
```

Results:

- group 1 / GPU 0: `16 passed, 1599 deselected in 4.64s`;
- group 2 / GPU 1: `16 passed, 1599 deselected in 5.02s`;
- group 3 / GPU 2: `16 passed, 1599 deselected in 5.68s`;
- group 4 / GPU 3: `15 passed, 1600 deselected in 4.87s`;
- aggregate: `63 passed`.

## Suggested next probes

- Add one temporary row with one 2CTA no-scales copy region followed by a
  second independent cross-CTA mbarrier region that has no `tmem_copy`, if a
  non-hanging producer/consumer pattern is available. This separates
  "sequential mbarriers" from "sequential copy-tracked mbarriers".
- Vary the sequential failing shape with a shared mbarrier reused across
  regions after invalidation. Prior work showed a shared mbarrier contrast can
  pass for mixed copy/scales; this would test the no-scales-only interval
  variant.
- Save one minimized MLIR reproducer for AR-001 before repair work begins; it
  is the smallest observed no-readback reproducer and avoids descriptor-chain
  distractions.

## Final result

- New independent `FZ-*` candidates: `0`.
- Runtime miscompiles: `0`.
- Unexpected unsupported diagnostics: `0`.
- Compiler crashes: several reproductions, all classified as existing
  `FZ-20260421-0014`.
- Backend/compiler repairs: `0`.
