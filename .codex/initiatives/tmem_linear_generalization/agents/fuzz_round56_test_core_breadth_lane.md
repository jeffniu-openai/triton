# Round 56 Test Core TMEM Breadth Lane C

Date: 2026-04-21
Branch: `codex/tmem`
Checkpoint base: `57262fc4a Integrate Round 56 scaled and ldred fuzz evidence`

This lane broadened local `python/test/gluon/test_core.py` TMEM runtime
sampling beyond the Round 54 small smoke. It stayed in catalog/discovery mode:
no backend fixes were attempted.

## Required Build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Main Breadth Selector

```bash
PYTHONPATH=.:./python \
pytest --collect-only -q python/test/gluon/test_core.py \
  -k 'test_tcgen05_mma_multicast_commit or test_tcgen05_mma_multicast_commit_twocta_linear_acc or test_tcgen05_mma_scaled_direct_multicast_barrier or test_tma_multicast_copy or test_tma_gather_scatter_multi_cta or test_tmem_copy_no_scales or test_tmem_copy_no_scales_matrix or test_tmem_linear_runtime_views or test_tmem_descriptor_chain_matrix or test_tmem_linear_roundtrip_atom_shapes or test_tmem_linear_roundtrip_variant_sweep or test_tmem_linear_roundtrip_splitn_shapes or test_tmem_linear_m64_roundtrip_direct_shapes or test_tmem_linear_m64_roundtrip_32x32b_fallback or test_tmem_linear_roundtrip_blocked_fallback or test_tmem_packed_f16_roundtrip_atom_shapes or test_tmem_reduction or test_tmem_reduction_linear_layouts or test_tmem_reduction_linear_legacy_block_equiv_layout or test_tmem_reduction_linear_former_clean_errors_are_supported or test_tmem_reduction_linear_mixed_layout_uses_software_reduce or test_tmem_reduction_non_f32_uses_software_reduce or test_tcgen05_mma_plain_kind_runtime or test_tcgen05_mma_plain_kind_i8_runtime_sm100 or test_tcgen05_mma_plain_kind_i8_reports_clean_error or test_tcgen05_mma_scaled_minimal or test_mma_scaled_tcgen05_copy_linear_acc'
```

Collection:

```text
305/18114 tests collected (17809 deselected) in 4.16s
```

Runtime split command shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
PYTHONPATH=.:./python \
pytest -s --tb=short \
  --splits 4 --group <group> \
  python/test/gluon/test_core.py \
  -k '<main breadth selector above>'
```

Results:

```text
group 1: 69 passed, 8 skipped, 18037 deselected in 25.97s
group 2: 77 passed, 18037 deselected in 50.60s
group 3: 77 passed, 18037 deselected in 4.80s
group 4: 74 passed, 18040 deselected in 16.43s
aggregate: 297 passed, 8 skipped, 0 failed
```

## High-CGA Scaled-Copy Selector

```bash
PYTHONPATH=.:./python \
pytest --collect-only -q python/test/gluon/test_core.py \
  -k 'test_mma_scaled_tcgen05_copy and (ctas_per_cga3 or ctas_per_cga4 or ctas_per_cga5)'
```

Collection:

```text
48/18114 tests collected (18066 deselected) in 3.75s
```

Runtime split:

```text
group 1: 12 passed, 18102 deselected in 11.96s
group 2: 12 passed, 18102 deselected in 5.14s
group 3: 12 passed, 18102 deselected in 10.63s
group 4: 12 passed, 18102 deselected in 4.74s
aggregate: 48 passed, 0 failed
```

## TMA-to-TMEM Shared-Input Selector

```bash
PYTHONPATH=.:./python \
pytest --collect-only -q python/test/gluon/test_core.py \
  -k 'test_tma_mma_shared_inputs and ctas_per_cga2 and True'
```

Collection:

```text
63/18114 tests collected (18051 deselected) in 3.76s
```

Runtime split:

```text
group 1: 16 passed, 18098 deselected in 4.27s
group 2: 16 passed, 18098 deselected in 7.11s
group 3: 13 passed, 3 skipped, 18098 deselected in 6.21s
group 4: 12 passed, 3 skipped, 18099 deselected in 4.99s
aggregate: 57 passed, 6 skipped, 0 failed
```

## Coverage Notes

The lane covered:

- allocation/lifetime through TMEM roundtrip, packed f16, atom-shape, split-N,
  M64 direct/fallback, and blocked fallback rows;
- multicast commit through plain MMAv5, two-CTA linear accumulator, and scaled
  direct multicast barrier rows;
- TMA/mbarrier/multicast and TMA-to-TMEM shared-input flows;
- no-scale `tcgen05.copy`, matrix copy, and scaled-copy helper rows;
- linear runtime descriptor views and descriptor-chain matrix rows;
- `ld.red` hardware and software-reduce rows across legacy, linear, former
  clean-error, mixed-layout, and non-f32 cases;
- plain MMAv5 `f16`, `tf32`, `f8f6f4`, and `i8` runtime rows, including the
  clean i8 diagnostic row;
- high-CGA scaled-copy rows over `ctas_per_cga3`, `ctas_per_cga4`, and
  `ctas_per_cga5`.

## Classification

No compiler crash, verifier drift, false unsupported diagnostic, opcode
absence, runtime miscompile, hang, unexpected skip/fail transition, or new
independent `FZ-*` bucket was found in this lane. All observed skips were
existing test-level skips in the selected parametrizations. Classification is
`existing/green coverage`; no backend repair was attempted.
