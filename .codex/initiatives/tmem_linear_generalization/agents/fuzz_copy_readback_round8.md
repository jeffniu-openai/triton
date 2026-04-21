# Round 8 Lane C: copy/readback generator adapters

- Time: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery-only structural fuzzing. No backend/compiler repairs were
  attempted.
- Temporary probe: `/tmp/tmem_copy_readback_round8_probe.py`
- GPU/cache: `CUDA_VISIBLE_DEVICES=2`,
  `TRITON_CACHE_DIR=/tmp/triton-cache-round8-lane-c`

## Scope

This lane extended the Round 7 copy/readback coverage with a deterministic
temporary pytest probe that enumerates copy-family adapters explicitly and
executes the kernels through the established runtime-matrix launchers. The
probe is still independent from the checked-in test selection: it owns the
case table, input generation, immediate readback checks, opcode checks, and
clean-diagnostic classification.

Covered surfaces:

- no-scales `warpx2::01_23` and `warpx2::02_13`;
- direct, indexed, subslice, and `slice(...).index(...)` TMEM descriptor
  views;
- two-CTA positive `warpx2::01_23` copy/readback rows;
- two-CTA `warpx2::02_13` clean unsupported rows;
- packed/subword clean diagnostics;
- scales `warpx4` direct and two-CTA copy/readback positives;
- scales `warpx4` descriptor-view clean unsupported boundary;
- `ld.red` descriptor-chain readback using direct helper/kernel invocation;
- two-CTA layout used from `num_ctas` `{4, 8, 16}` contexts.

## Probe Cases

Positive copy/readback rows:

- `r8c-noscale-warpx2-01-23-direct-f32`
- `r8c-noscale-warpx2-02-13-direct-i32`
- `r8c-noscale-warpx2-01-23-sublice-view-f32`
- `r8c-noscale-warpx2-02-13-indexed-view-f32`
- `r8c-noscale-warpx2-01-23-slice-index-view-f32`
- `r8c-noscale-warpx2-01-23-twocta-direct-f32`
- `r8c-noscale-warpx2-01-23-twocta-indexed-view-f32`
- `r8c-noscale-warpx2-01-23-twocta-subslice-view-f32`
- `r8c-scales-warpx4-direct`
- `r8c-scales-warpx4-twocta-direct`
- `test_round8_ldred_readback_descriptor_chain_positive`, direct
  `tmem_ld_red_descriptor_chain_kernel` call for identity `N=64`,
  `32x32b`, `min`, `PropagateNan.NONE`.

Clean diagnostics:

- `r8c-noscale-warpx2-02-13-twocta-direct-clean`
- `r8c-noscale-warpx2-02-13-twocta-indexed-clean`
- `r8c-noscale-warpx2-packed-f16-clean`
- `r8c-twocta-layout-in-4cta-clean`
- `r8c-twocta-layout-in-8cta-clean`
- `r8c-twocta-layout-in-16cta-clean`
- `r8c-scales-warpx4-descriptor-view-clean`

## Classification

No stable new backend/compiler failure was found, and no new `FZ-*` id is
warranted from this lane.

Positive support stayed positive for:

- single-CTA no-scales `warpx2::01_23` and `warpx2::02_13` direct/view
  copy/readback;
- two-CTA no-scales `warpx2::01_23` direct/view copy/readback;
- scales `warpx4` direct and two-CTA direct copy/readback;
- descriptor-chain `ld.red` readback using the hardware
  `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32` path.

Clean diagnostics stayed clean for:

- two-CTA no-scales `warpx2::02_13` direct and indexed view rows;
- f16 packed/subword no-scales `warpx2`;
- two-CTA TMEM layout used in 4-, 8-, and 16-CTA contexts;
- scales `warpx4` descriptor-view copy where the view permutes physical TMEM
  rows in a way the current public copy atom cannot realize without a narrower
  row-masked or row-partitioned schedule.

The first version of the probe incorrectly classified the scales descriptor
view as a positive readback. It failed with a clean diagnostic:

`The source shared layout maps to tcgen05.copy.warpx4.32x128b, but Triton could
not synthesize a compatible shared-memory descriptor plan for it.`

The diagnostic also stated that it was reported as cleanly unsupported instead
of falling through to late LLVM lowering. That row was therefore moved into the
clean-boundary group rather than assigned a new failure id.

The first version also used nested `pytest.main(...)` for the `ld.red` row,
which triggered a pytest plugin internal error about missing
`filtered_exceptions`. That was a harness issue, not a compiler result. The
final probe calls the `tmem_ld_red_descriptor_chain_kernel` directly and
validates runtime output plus `.ld.red.` opcode pairs.

## Validation

- Required build:
  `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8`
  reported `ninja: no work to do`.
- Py-compile:
  `PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_copy_readback_round8_probe.py`
  passed.
- Collect-only:
  `PYTHONPATH=.:./python:./python/test/gluon pytest --collect-only -q /tmp/tmem_copy_readback_round8_probe.py`
  collected 18 nodeids.
- Full probe:
  `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-round8-lane-c PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short /tmp/tmem_copy_readback_round8_probe.py`
  passed `18 passed in 4.55s`.

## Follow-Ups

- Do not promote a new structural-fuzzer sentinel from this lane.
- Future copy/readback fuzzing should continue turning the temporary adapter
  into a generator that emits runnable direct kernels, but this lane did not
  expose an uncovered compiler crash, false unsupported case, or miscompile.
