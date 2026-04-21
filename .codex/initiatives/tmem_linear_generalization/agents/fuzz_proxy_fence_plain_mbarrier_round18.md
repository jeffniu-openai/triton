# Round 18 follow-up: plain mbarrier interval contrast for `FZ-20260421-0014`

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend/compiler code was changed. No
commit or push was made by this lane.

## Scope

This follow-up tested a question left open by Lane AR: does
`FZ-20260421-0014` require two copy-tracked 2CTA no-scales `tcgen05.copy`
regions, or can one copy-tracked region plus another independent mbarrier
interval reproduce the same proxy-fence insertion failure?

The probe added two rows to the temporary Lane AR file:

```text
/tmp/tmem_proxy_fence_intervals_round18_probe.py
```

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

## Probe validation

Syntax/collection:

```bash
python3 -m py_compile /tmp/tmem_proxy_fence_intervals_round18_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_proxy_fence_intervals_round18_probe.py -k 'as00'
```

Result:

```text
2/15 tests collected (13 deselected) in 2.95s
```

Runtime command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_proxy_fence_intervals_round18_probe.py -k 'as00'
```

Result:

```text
AS001_CLASS=existing_fz0014
AS002_CLASS=existing_fz0014
2 passed, 13 deselected in 3.07s
```

## Row table

| Case | Interval shape | Result | Classification |
| --- | --- | --- | --- |
| AS-001 | one direct 2CTA no-scales copy region; wait its commit mbarrier; then initialize/arrive/wait a separate plain mbarrier; no TMEM readback | proxy-fence insertion failed with the known insertion-point diagnostic | existing `FZ-20260421-0014` |
| AS-002 | initialize/arrive/wait a separate plain mbarrier first; then one direct 2CTA no-scales copy region; wait its commit mbarrier; no TMEM readback | proxy-fence insertion failed with the known insertion-point diagnostic | existing `FZ-20260421-0014` |

## Classification

No new independent `FZ-*` was assigned because both rows fail in the same
`triton-nvidia-gpu-proxy-fence-insertion` pass with the same diagnostic already
cataloged under `FZ-20260421-0014`.

This does sharpen the existing bucket. The failure does not require two
copy-tracked mbarriers. One legal 2CTA no-scales copy-tracked mbarrier plus a
second independent mbarrier interval is enough, and either ordering reproduces
the failure.

One nuance from the emitted TTGIR: the plain mbarrier row lowered to a
single-CTA-shaped `!ttg.memdesc<1xi64, ...>` allocation even when requested via
`allocate_mbarrier(two_ctas=True)` in this simple arrive/wait-only shape. The
failing function still combines that plain interval with a 2CTA copy-commit
mbarrier and reaches the same proxy-fence insertion assertion. That suggests
the pass is sensitive to multiple independent mbarrier intervals around a 2CTA
copy region, not only to multiple copy-tracked regions.

## Next probes

- Minimize AS-001 into a saved MLIR reproducer before the repair phase. It has
  no TMEM readback and only one `ttng.tmem_copy`, making it a cleaner proxy-fence
  insertion reproducer than the earlier two-copy cases.
- Add a contrast with two plain mbarrier intervals and no TMEM copy, if a safe
  non-hanging row is available, to isolate whether `ttng.tmem_copy` contributes
  a special tracked use or only introduces the cross-CTA commit interval.
- Add a contrast where all mbarriers are initialized before either interval is
  used: Lane AR already showed that shape passes for two and three copy regions.

## Final result

- New independent `FZ-*` candidates: `0`.
- Runtime miscompiles: `0`.
- Unexpected unsupported diagnostics: `0`.
- Compiler crashes: `2`, both classified as existing `FZ-20260421-0014`.
- Backend/compiler repairs: `0`.
