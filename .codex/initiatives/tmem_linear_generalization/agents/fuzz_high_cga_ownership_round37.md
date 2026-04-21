# Round 37: high-CGA TMEM ownership fuzzing

Scope: discovery-only audit of TMEM-bearing kernels in high-CGA launch
contexts. I did not edit backend code or initiative tracker files. Temp probe:
`/tmp/tmem_high_cga_ownership_round37_probe.py`.

## Baseline

- Branch/HEAD before report commit: `codex/tmem` at `db29e14c6`.
- Required rebuild:
  - `make -j8`
  - Result: `ninja: no work to do`.

## Commands

Checked-in high-CGA/multicast collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest --collect-only -q -k 'layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma' python/test/gluon/test_tmem_structural_fuzzer.py python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py
```

Collected 28 selected tests.

Four-GPU checked-in selector:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 -k 'layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma' python/test/gluon/test_tmem_structural_fuzzer.py python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 -k 'layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma' python/test/gluon/test_tmem_structural_fuzzer.py python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 -k 'layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma' python/test/gluon/test_tmem_structural_fuzzer.py python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 -k 'layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma' python/test/gluon/test_tmem_structural_fuzzer.py python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py
```

Results: groups 1-4 each ran 7 selected tests; all 28 passed.

Lit ownership/mbarrier control:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
lit -v test/TritonNvidiaGPU/membar-cluster.mlir
```

Result: `PASS: TRITON :: TritonNvidiaGPU/membar-cluster.mlir`.

Note: running `lit -v /root/code/triton/test/TritonNvidiaGPU/membar-cluster.mlir`
from the build dir used the source-tree config and failed with
`llvm_config.use_lit_shell` unset; the build-tree-relative lit command above is
the valid repro command.

Additional focused checked-in runtime controls:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_matrix[mxfp8-mxfp8-128-0-128-True]' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_use_acc[mxfp8-mxfp8-128-128-256-False]'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[True-ctas_per_cga2]' 'python/test/gluon/test_core.py::test_tcgen05_mma_scaled_direct_multicast_barrier'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_core.py::test_tma_multicast_copy[ctas_per_cga2]' 'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[False-ctas_per_cga2]'
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix[mxfp8-mxfp8-256-128-2-False-linear]' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix[nvfp4-nvfp4-256-256-2-True-linear]'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_shared_scale_descriptor_view_auto_tmem_copy[1-128-128-256-0-False]' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_shared_scale_descriptor_view_auto_tmem_copy[2-256-64-128-0-False]'
```

Results: all listed focused controls passed. One earlier attempted command used
nonexistent scaled-copy node ids with `2048` in the id and failed collection
with `collected 0 items`; corrected node ids are listed above.

Temp ownership probe:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon python /tmp/tmem_high_cga_ownership_round37_probe.py
```

Result: `SUMMARY expected_diagnostics=15/15`.

## Matrix

| Surface | CTA contexts | Result | Classification |
| --- | ---: | --- | --- |
| Checked-in high-CGA/multicast selector | 2/4/16 as encoded by tests | 28 passed | Green control |
| Lit `membar-cluster.mlir` two-CTA/four-CTA mbarrier ownership | 2/4 | Passed | Green control |
| Plain MMAv5 multicast commit, 1CTA and 2CTA instruction-local rows | 16 | Passed | Green control |
| TMA multicast copy | 16 | Passed | Green control |
| Scaled-MMAv5 direct multicast barrier | 4 | Passed | Green control |
| Scaled-MMAv5 two-CTA accumulator subslice | 2 | Passed | Green control |
| Scaled-MMAv5 scale-copy via MMA | 2 | Passed | Green control |
| Scaled-MMAv5 shared scale descriptor auto-copy | 1/2 | Passed | Green control |
| Temp `ld/st` descriptor-view 1CTA local layout | 4/8/16 | Expected CTA-count diagnostic | Existing `FZ-20260421-0010` |
| Temp `ld.red` direct 1CTA local layout | 4/8/16 | Expected CTA-count diagnostic | Existing `FZ-20260421-0010` |
| Temp `ld.red` lifted 2CTA local layout | 4/8/16 | Expected CTA-count diagnostic | Existing `FZ-20260421-0010` |
| Temp scales copy 1CTA local layout | 4/8/16 | Expected CTA-count diagnostic wrapped as Gluon parse error | Existing `FZ-20260421-0010` |
| Temp scales copy 2CTA local layout | 4/8/16 | Expected CTA-count diagnostic wrapped as Gluon parse error | Existing `FZ-20260421-0010` |

## Repros and classification

`R37-LDST-1CTA-N{4,8,16}`, `R37-LDRED-1CTA-N{4,8,16}`, and
`R37-LDRED-2CTA-N{4,8,16}` fail before lowering with the direct diagnostic:

```text
Layout has 1 CTAs per CGA, but the context requires N CTAs per CGA.
Layout has 2 CTAs per CGA, but the context requires N CTAs per CGA.
```

`R37-CPSCALES-1CTA-N{4,8,16}` and `R37-CPSCALES-2CTA-N{4,8,16}` emit the same
diagnostic on stderr but Gluon wraps it as `RuntimeError('error encountered
during parsing')`. These are still classified under existing
`FZ-20260421-0010` because the diagnostic is the same local-layout versus
kernel-CGA conflation already cataloged for copy, ld/st, and ld.red.

No new independent `FZ-*` candidate is proposed. I did not reproduce a missed
`getModuleTwoCTAs` propagation failure, verifier over-strictness beyond
`FZ-0010`, compiler crash, or runtime miscompile in this lane. The high-CGA
plain-MMAv5, scaled-MMAv5, TMA, and mbarrier controls stayed green, while local
1CTA/2CTA non-MMA TMEM work in 4/8/16 CTA launch contexts remains gated by
`FZ-20260421-0010`.
