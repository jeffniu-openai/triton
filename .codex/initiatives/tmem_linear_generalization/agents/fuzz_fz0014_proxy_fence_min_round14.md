# Lane AF Round 14: FZ-20260421-0014 proxy-fence minimization

Scope: discovery/cataloging only. No backend/compiler code was changed. This
lane minimized the new report-only `FZ-20260421-0014` from
`agents/fuzz_high_cga_copy_scales_round14.md`: a legal `num_ctas=2` mixed
2CTA copy/scales kernel aborting in proxy-fence insertion with:

```text
could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
```

## Required rebuild

Command:

```bash
make -j8
```

Result: no work to do.

## Saved compiler-pass reproducer

Command:

```bash
build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_high_cga_copy_scales_round14_mixed_fail.mlir --run-reproducer
```

Result: reproduced the same pass failure at the `tt.func`:

```text
'tt.func' op could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
```

The pass pipeline failed while executing `ConvertTritonGPUToLLVM`, with the
failing sub-pass in the reproducer pipeline:
`triton-nvidia-gpu-proxy-fence-insertion{compute-capability=103}`.

## Temporary minimization probe

Temporary file:

```text
/tmp/tmem_fz0014_min_probe.py
```

Collect-only:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_fz0014_min_probe.py
```

Result: `10 tests collected`.

The probe reuses checked-in runtime-matrix layout helpers and runs each row in
a fresh pytest subprocess with stable per-GPU caches. Case ids are deterministic
and use no random seed except for scales input data in rows that either fail at
compile time or validate with exact equality.

## Minimized case matrix

All commands below were run from `/root/code/triton`.

### AF-001: original shape, descriptor view, both readbacks

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
  '/tmp/tmem_fz0014_min_probe.py::test_fz0014_no_first_matrix[view_both_readbacks]'
```

Result: reproduced `FZ-20260421-0014`.

### AF-002: descriptor view, only scales readback

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
  '/tmp/tmem_fz0014_min_probe.py::test_fz0014_no_first_matrix[view_only_scales_readback]'
```

Result: reproduced `FZ-20260421-0014`.

### AF-003: descriptor view, only no-scales readback

Command:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
  '/tmp/tmem_fz0014_min_probe.py::test_fz0014_no_first_matrix[view_only_no_readback]'
```

Result: reproduced `FZ-20260421-0014`.

### AF-004: descriptor view, no readbacks

Command:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
  '/tmp/tmem_fz0014_min_probe.py::test_fz0014_no_first_matrix[view_no_readbacks]'
```

Result: reproduced `FZ-20260421-0014`.

Conclusion: neither readback is required.

### AF-005: direct no-scales TMEM, both readbacks

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
  '/tmp/tmem_fz0014_min_probe.py::test_fz0014_no_first_matrix[direct_both_readbacks]'
```

Result: reproduced `FZ-20260421-0014`.

### AF-006: direct no-scales TMEM, no readbacks

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
  '/tmp/tmem_fz0014_min_probe.py::test_fz0014_no_first_matrix[direct_no_readbacks]'
```

Result: reproduced `FZ-20260421-0014`.

Conclusion: the no-scales descriptor-chain view is not required.

### AF-007: reversed order, scales first then no-scales view

Command:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
  /tmp/tmem_fz0014_min_probe.py::test_fz0014_scales_first_order
```

Result: reproduced `FZ-20260421-0014`.

Conclusion: the original no-scales-then-scales ordering is not required.

### AF-008: shared single mbarrier contrast

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
  /tmp/tmem_fz0014_min_probe.py::test_fz0014_shared_barrier_contrast
```

Result: `1 passed in 6.30s`. The temporary test also validated no-scales and
scales outputs with exact equality.

Conclusion: the current minimized failure requires independent mbarrier
alloc/init regions; sharing one mbarrier avoids the proxy-fence insertion
abort for this probe.

### AF-009: two independent scales-copy regions

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
  /tmp/tmem_fz0014_min_probe.py::test_fz0014_two_scales_independent_barriers_probe
```

Result: `1 passed in 3.72s`, with exact output equality.

Conclusion: two independent 2CTA scales-copy regions are not sufficient to
trigger `FZ-0014`.

### AF-010: two independent no-scales direct-copy regions

Command:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
  /tmp/tmem_fz0014_min_probe.py::test_fz0014_two_no_direct_independent_barriers_probe
```

Result: the test passed because it expected the compiler failure; stderr
contained:

```text
'tt.func' op could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
```

Conclusion: a scales copy is not required. Two independent direct no-scales
2CTA `warpx2::01_23` copy regions are sufficient.

## Single-region controls

No-scales 2CTA `warpx2` direct copy:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2_01_23_twocta_positive'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_twocta_positive[f32-torch_dtype0]'
```

Results: collect-only selected `2/1615`; exact f32 row passed
`1 passed in 5.13s`.

Scales 2CTA direct copy:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_twocta_direct_copy
```

Result: `1 passed in 2.70s`.

## Classification

`FZ-20260421-0014` remains an independent report-only compiler crash bucket.
It is not `FZ-20260421-0010` because every reproduced row here uses legal
`num_ctas=2` and reaches proxy-fence insertion instead of the high-CGA
CTA-count layout validator.

Minimized ownership:

- Crash type: compiler crash / pass-manager abort.
- Failing pass surface: NVIDIA proxy-fence insertion during
  `ConvertTritonGPUToLLVM`.
- Smallest current positive: two direct no-scales 2CTA `warpx2::01_23`
  `ttng.tmem_copy` regions in one kernel, each with its own cross-CTA
  `mbarrier.init`/`tc_gen5_commit`/`wait_barrier`.
- Not required: either readback, no-scales descriptor-chain/subslice view,
  scales copy, or the original no-scales-before-scales ordering.
- Negative controls: a single no-scales 2CTA copy passes; a single scales 2CTA
  copy passes; two independent scales-copy regions pass; a mixed no-scales
  plus scales probe passes when both regions share one mbarrier.

Current hypothesis: proxy-fence insertion cannot place fences for multiple
independent cross-CTA mbarrier regions once at least one tracked region is a
2CTA no-scales linear TMEM copy. The issue is not descriptor-view arithmetic
or readback lowering; it is closer to region ownership/order planning for
cross-CTA no-scales copy mbarriers.

## Follow-up recommendations

- Keep `FZ-0014` report-only until the campaign pivots from discovery to
  repair.
- If promoting a sentinel, prefer a minimized checked-in structural row for
  "two direct 2CTA no-scales `warpx2::01_23` copies with independent
  mbarriers" rather than the original mixed copy/scales row.
- Before fixing, add one compile-only probe for "one no-scales 2CTA copy plus
  a second independent cross-CTA mbarrier without a second copy" if a safe
  non-hanging form is available; that would separate "two no-scales copy
  regions" from "one no-scales tracked copy plus another independent mbarrier
  region".
- Backend repair should target proxy-fence insertion's cross-CTA mbarrier
  region ordering, not TMEM descriptor-chain lowering or scales descriptor
  planning.
