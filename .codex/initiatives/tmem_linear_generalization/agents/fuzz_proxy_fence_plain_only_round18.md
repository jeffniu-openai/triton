# Round 18 follow-up: plain-only mbarrier intervals for `FZ-20260421-0014`

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend/compiler code was changed. No
commit or push was made by this lane.

## Scope

This follow-up tested whether `FZ-20260421-0014` needs TMEM copy participation
at all. It adds two plain-only mbarrier rows to:

```text
/tmp/tmem_proxy_fence_intervals_round18_probe.py
```

The rows use no `ttng.tmem_copy`, no `tcgen05_commit`, and no TMEM allocation.

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
  pytest --collect-only -q /tmp/tmem_proxy_fence_intervals_round18_probe.py -k 'at00'
```

Result:

```text
2/17 tests collected (15 deselected) in 3.03s
```

Runtime command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_proxy_fence_intervals_round18_probe.py -k 'at00'
```

Result:

```text
AT001_CLASS=existing_fz0014
AT002_CLASS=green
2 passed, 15 deselected in 2.61s
```

## Row table

| Case | Interval shape | Result | Classification |
| --- | --- | --- | --- |
| AT-001 | plain mbarrier 0: init/arrive/wait; then plain mbarrier 1: init/arrive/wait; no TMEM operations | proxy-fence insertion failed with the known insertion-point diagnostic | existing `FZ-20260421-0014` |
| AT-002 | allocate/init both plain mbarriers first; arrive both; wait both; no TMEM operations | passed, status store completed | green contrast |

## Classification

No new independent `FZ-*` was assigned because the failure is the same
`triton-nvidia-gpu-proxy-fence-insertion` diagnostic already tracked as
`FZ-20260421-0014`.

This does materially revise the diagnosis: the proxy-fence insertion failure is
not TMEM-copy-specific. A pure mbarrier program with two sequential
init/use intervals is sufficient. TMEM copy tests expose the bug because 2CTA
copy lowering naturally creates cross-CTA commit/wait intervals, but the core
gap is in proxy-fence interval construction/order handling for independent
mbarrier lifetimes.

The green AT-002 control matches Lane AR's copy-region controls: initializing
all mbarriers before any arrive/wait use gives the pass a valid insertion
window. Sequentially starting the second mbarrier only after the first has been
waited splits the interval and reproduces the failure.

## Next probes

- Save AT-001 as the minimal non-TMEM MLIR reproducer before repair work begins;
  it should be easier to reason about than any TMEM-bearing reproducer.
- Check whether the same plain-only sequential failure appears on the merge
  base/main branch before attributing it to the TMEM branch. It may be a
  preexisting proxy-fence pass limitation newly exposed by TMEM tests.
- Add a checked-in sentinel only after deciding whether this belongs in TMEM
  runtime coverage or in a lower-level mbarrier/proxy-fence test surface.

## Final result

- New independent `FZ-*` candidates: `0`.
- Runtime miscompiles: `0`.
- Unexpected unsupported diagnostics: `0`.
- Compiler crashes: `1`, classified as existing `FZ-20260421-0014`.
- Backend/compiler repairs: `0`.
