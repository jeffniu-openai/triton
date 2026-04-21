# Lane AD Round 14: high-CGA copy/scales ownership fuzzing

Scope: discovery-only Python/Gluon runtime fuzzing around high-CGA launch
contexts, local 1CTA/2CTA copy/scales operations, descriptor chains feeding
copy/scales, and module-wide two-CTA ownership constraints. No backend or
compiler code was changed.

## Required rebuild

Command:

```bash
make -j8
```

Result: no work to do.

## Checked-in copy/scales baseline

Collect-only:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_scales and not reports'
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales_warpx2 and not reports'
```

Results:

- `cp_scales and not reports`: `33/1615` selected.
- `cp_no_scales_warpx2 and not reports`: `62/1615` selected.

Runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group <group> \
  --store-durations --durations-path /tmp/tmem_lane_ad_round14_copy_scales_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_scales and not reports) or (cp_no_scales_warpx2 and not reports)'
```

Results:

- GPU 0 / group 1: `24 passed, 1591 deselected in 4.88s`.
- GPU 1 / group 2: `24 passed, 1591 deselected in 4.93s`.
- GPU 2 / group 3: `24 passed, 1591 deselected in 6.11s`.
- GPU 3 / group 4: `23 passed, 1592 deselected in 4.97s`.
- Aggregate: `95 passed`.

Coverage anchored the existing green surface for:

- scales direct/rematerialized/shared-subslice copy;
- scaled-MMA copy setup with 1CTA and 2CTA scale copies;
- no-scales `warpx2::{01_23,02_13}` direct/indexed/subslice/slice-index
  descriptor chains;
- no-scales 2CTA `warpx2::01_23` descriptor-chain positives; and
- known no-scales subword and `warpx2::02_13` clean diagnostics.

High-CGA MMA controls:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -s --tb=short \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[False-ctas_per_cga1]' \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[True-ctas_per_cga2]'
```

Result: `2 passed in 3.19s`.

## Temporary probe

Temporary file:

```text
/tmp/tmem_high_cga_copy_scales_round14_probe.py
```

The probe intentionally stayed outside the repo. It imports the checked-in
runtime-matrix module to reuse existing copy layouts and expected-output
helpers, then defines temporary mixed-module kernels.

Collect-only:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_high_cga_copy_scales_round14_probe.py
```

Result: `5 tests collected`.

### AD-001: mixed 2CTA no-scales descriptor-chain copy plus scales direct copy

Case id: `AD-001`, no random seed. Inputs are deterministic for no-scales
`torch.arange`; the scales tensor is random but the kernel fails before
execution.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
  /tmp/tmem_high_cga_copy_scales_round14_probe.py::test_ad_mixed_twocta_no_scales_descriptor_chain_plus_scales_direct_passes
```

Result: `1 failed`. Failure mode:

```text
'tt.func' op could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
RuntimeError: PassManager::run failed
Pipeline failed while executing ConvertTritonGPUToLLVM
```

Reproducer artifacts:

- full pytest log:
  `/tmp/tmem_high_cga_copy_scales_round14_mixed_fail.log`;
- extracted MLIR reproducer:
  `/tmp/tmem_high_cga_copy_scales_round14_mixed_fail.mlir`;
- repro command:

```bash
build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_high_cga_copy_scales_round14_mixed_fail.mlir --run-reproducer
```

`triton-opt --run-reproducer` reproduced the same error at the temporary
mixed kernel's `tt.func`.

Classification: new report-only candidate `FZ-20260421-0014`.

Why this is not `FZ-20260421-0010`: the failure is in a legal `num_ctas=2`
launch, not a 4/8/16 CTA launch context, and it reaches
`ConvertTritonGPUToLLVM` rather than the high-CGA CTA-count validator.

Current hypothesis: a mixed module with a 2CTA no-scales descriptor-chain
copy followed by a two-CTA scales copy creates multiple cross-CTA mbarrier
regions that the proxy-fence insertion pass cannot order. The checked-in
single-surface copy/scales baselines above pass, so the failure appears to be
module-composition/ownership sensitive.

### AD-002 through AD-004: same mixed module in 4/8/16 CTA launch contexts

Case ids: `AD-002` (`num_ctas=4`), `AD-003` (`num_ctas=8`),
`AD-004` (`num_ctas=16`).

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -s --tb=short \
  '/tmp/tmem_high_cga_copy_scales_round14_probe.py::test_ad_mixed_twocta_no_scales_descriptor_chain_plus_scales_direct_high_cga_fz0010[4]' \
  '/tmp/tmem_high_cga_copy_scales_round14_probe.py::test_ad_mixed_twocta_no_scales_descriptor_chain_plus_scales_direct_high_cga_fz0010[8]' \
  '/tmp/tmem_high_cga_copy_scales_round14_probe.py::test_ad_mixed_twocta_no_scales_descriptor_chain_plus_scales_direct_high_cga_fz0010[16]'
```

Result: `3 passed in 2.79s`, where pass means each row raised the expected
diagnostic:

```text
Layout has 2 CTAs per CGA, but the context requires {4,8,16} CTAs per CGA.
```

Classification: existing `FZ-20260421-0010`.

This broadens `FZ-0010` evidence to a mixed copy/scales module whose
instruction-local no-scales side is 2CTA and descriptor-chain fed.

### AD-005: mixed 2CTA no-scales copy plus scales descriptor-view copy

Case id: `AD-005`, no random seed. This row is not classified.

The first attempt missed a 2CTA CGA layout on the temporary scales input path
and failed with `Layout has 1 CTAs per CGA, but the context requires 2 CTAs
per CGA`. After adding the CGA layout, the temporary shared layout failed
during setup with:

```text
After removing the zero bases the layout must be bijective
```

Classification: harness/setup limitation. It is not counted as a backend
bucket and was not promoted.

## Final classification

- Checked-in baseline: `95 passed`.
- High-CGA MMA controls: `2 passed`.
- Temporary classified rows: `3` existing `FZ-20260421-0010`, `1` new
  candidate `FZ-20260421-0014`.
- Temporary harness/setup limitations: `1`.
- No runtime wrong-result/miscompile was observed in this lane.
- No backend/compiler repair was attempted.

## Candidate bucket

`FZ-20260421-0014`: mixed 2CTA no-scales descriptor-chain copy plus scales
copy aborts in proxy-fence insertion.

Minimal current shape:

- `num_ctas=2`, `num_warps=4`;
- first region: `TensorMemoryLinearLayout(two_ctas=True)` no-scales
  `warpx2::01_23` copy into a subsliced parent descriptor, with cross-CTA
  mbarrier/commit/wait and readback;
- second region: `TensorMemoryScalesLayout(cga_layout=[[1, 0]])` scales
  `warpx4` copy with its own cross-CTA mbarrier/commit/wait and readback;
- failure: `triton-nvidia-gpu-proxy-fence-insertion` cannot find an insertion
  point between cross-CTA `mbarrier.init` operations and tracked mbarrier uses.

Recommended next discovery step: minimize whether `FZ-0014` needs both
readbacks, both independent mbarriers, or the no-scales descriptor-chain view;
do not repair backend code until this is minimized and the campaign pivots to
fixing.
