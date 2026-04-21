# Round 54 Lane C: Opcode Consistency and Lowering Drift

- Time: 2026-04-21 15:21 UTC
- Branch: `codex/tmem`
- Scope: discovery/cataloging only; no backend repair attempted.
- Target: TTGIR/LLIR/PTX opcode consistency for selected TMEM runtime kernels
  covering `ld/st`, `ld.red`, copy, plain MMAv5, scaled MMAv5, wait/commit,
  proxy fence, and mbarrier/cluster synchronization.

## Commands

Required rebuild:

```bash
make -j8
```

Result: no-op rebuild via
`ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12`;
`ninja: no work to do`.

Initial harness check without local `PYTHONPATH`:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k "(ldst and (alloc_lifetime or fixed_offset_patterns or subword_pack_unpack or scales_direct_roundtrip)) or (ld_red and (identity_linear_layout or explicit_n_sharded_layout_uses_software_reduce or non_f32_contract_uses_software_reduce or descriptor_chain_n_sweep_explicit_variants)) or (cp_no_scales and (warpx2_01_23_candidate_positive or twocta_codegen or linear_tile_permuted)) or cp_scales_warpx4 or mma_plain_kinds_with_linear_acc or mma_scaled_minimal or mma_scaled_root_format_matrix"
```

Result: collection failed before tests with
`ModuleNotFoundError: No module named 'triton.compiler.errors'`. This is a
local invocation setup issue; all following commands used `PYTHONPATH=.:./python`.

Focused opcode selector collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k "(ldst and (alloc_lifetime or fixed_offset_patterns or subword_pack_unpack or scales_direct_roundtrip)) or (ld_red and (identity_linear_layout or explicit_n_sharded_layout_uses_software_reduce or non_f32_contract_uses_software_reduce or descriptor_chain_n_sweep_explicit_variants)) or (cp_no_scales and (warpx2_01_23_candidate_positive or twocta_codegen or linear_tile_permuted)) or cp_scales_warpx4 or mma_plain_kinds_with_linear_acc or mma_scaled_minimal or mma_scaled_root_format_matrix"
```

Result: `172/1615` selected.

Four-GPU focused opcode sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k "(ldst and (alloc_lifetime or fixed_offset_patterns or subword_pack_unpack or scales_direct_roundtrip)) or (ld_red and (identity_linear_layout or explicit_n_sharded_layout_uses_software_reduce or non_f32_contract_uses_software_reduce or descriptor_chain_n_sweep_explicit_variants)) or (cp_no_scales and (warpx2_01_23_candidate_positive or twocta_codegen or linear_tile_permuted)) or cp_scales_warpx4 or mma_plain_kinds_with_linear_acc or mma_scaled_minimal or mma_scaled_root_format_matrix"
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k "(ldst and (alloc_lifetime or fixed_offset_patterns or subword_pack_unpack or scales_direct_roundtrip)) or (ld_red and (identity_linear_layout or explicit_n_sharded_layout_uses_software_reduce or non_f32_contract_uses_software_reduce or descriptor_chain_n_sweep_explicit_variants)) or (cp_no_scales and (warpx2_01_23_candidate_positive or twocta_codegen or linear_tile_permuted)) or cp_scales_warpx4 or mma_plain_kinds_with_linear_acc or mma_scaled_minimal or mma_scaled_root_format_matrix"
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k "(ldst and (alloc_lifetime or fixed_offset_patterns or subword_pack_unpack or scales_direct_roundtrip)) or (ld_red and (identity_linear_layout or explicit_n_sharded_layout_uses_software_reduce or non_f32_contract_uses_software_reduce or descriptor_chain_n_sweep_explicit_variants)) or (cp_no_scales and (warpx2_01_23_candidate_positive or twocta_codegen or linear_tile_permuted)) or cp_scales_warpx4 or mma_plain_kinds_with_linear_acc or mma_scaled_minimal or mma_scaled_root_format_matrix"
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k "(ldst and (alloc_lifetime or fixed_offset_patterns or subword_pack_unpack or scales_direct_roundtrip)) or (ld_red and (identity_linear_layout or explicit_n_sharded_layout_uses_software_reduce or non_f32_contract_uses_software_reduce or descriptor_chain_n_sweep_explicit_variants)) or (cp_no_scales and (warpx2_01_23_candidate_positive or twocta_codegen or linear_tile_permuted)) or cp_scales_warpx4 or mma_plain_kinds_with_linear_acc or mma_scaled_minimal or mma_scaled_root_format_matrix"
```

Result: `43 passed` on each shard, aggregate `172 passed`.

Focused proxy/mbarrier/multicast selector:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py -k "(dense_shared_rematerializes or twocta_codegen or twocta_128x128b_codegen or mma_multicast_commit or scaled_direct_multicast_barrier)"
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py -k "(dense_shared_rematerializes or twocta_codegen or twocta_128x128b_codegen or mma_multicast_commit or scaled_direct_multicast_barrier)"
```

Result: `39/19729` selected, `39 passed`.

Representative artifact dump:

```bash
rm -rf /tmp/tmem_round54_opcode_dump && mkdir -p /tmp/tmem_round54_opcode_dump && CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/tmem_round54_opcode_dump pytest -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_alloc_lifetime_ldst[identity-1-1-128-128-128]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_fixed_offset_patterns_128x256[16x256b]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_identity_linear_layout[identity-128-128-4-32x32b.x128-True-propagate_nan1-max]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_non_f32_contract_uses_software_reduce[i32_plain-min]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive[f32-torch_dtype0]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_codegen[linear-f32-torch_dtype0-16-32-2]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_plain_kinds_with_linear_acc[f16-linear-128-64]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_root_format_matrix[mxfp8-mxfp8-128-128-linear]' \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[True-ctas_per_cga0]'
```

Result: `9 passed`; dumped `27` TTGIR/LLIR/PTX files under
`/tmp/tmem_round54_opcode_dump`.

## Case Coverage and Counts

- `ld/st`: allocation lifetime, fixed-offset `128x256`, subword pack/unpack,
  and scale-layout load/store rows. The selected runtime matrix contributes
  `19` rows; representative dumps include `32x32b.x128` and scale
  `16x128b.x1` load/store opcode streams.
- `ld.red`: hardware f32 identity and descriptor-chain rows plus software
  fallback sentinels. The selected matrix contributes `62` rows: `20`
  identity hardware reductions, `8` descriptor-chain explicit variants, `12`
  n-sharded software-reduce rows, and `22` non-f32 software-reduce rows.
- `copy`: no-scale `warpx2`, two-CTA `128x256b`, tile-permuted finite atoms,
  and scaled-copy `warpx4` rows. The selected matrix contributes `56` rows in
  the main sweep plus proxy/mbarrier duplicates in the focused sweep.
- Plain MMAv5: `13` linear/legacy accumulator rows covering f16, bf16, tf32,
  f8e5m2, and f8e4m3.
- Scaled MMAv5: `12` rows including minimal and root-format combinations for
  `mxfp8`, `mxfp4`, mixed `mxfp8/mxfp4`, and `nvfp4`.
- Proxy/mbarrier/commit/multicast: focused selector contributes `39` rows,
  including dense shared rematerialization, two-CTA copy codegen, two-CTA
  `128x128b`, multicast plain MMA, and scaled direct multicast barrier.

## Representative Dump Summary

The representative artifact dump confirmed these TTGIR-to-LLIR/PTX streams:

- `tmem_ldst_variant_kernel`: TTGIR has `ttng.tmem_alloc`,
  `ttng.tmem_store`, and `ttng.tmem_load`; LLIR/PTX both contain one
  `tcgen05.st.sync.aligned.32x32b.x128.b32`, one
  `tcgen05.ld.sync.aligned.32x32b.x128.b32`, one `wait::st`, one `wait::ld`,
  and matching alloc/relinquish/dealloc.
- `tmem_reduction_kernel`: LLIR/PTX both contain one hardware
  `tcgen05.ld.red.sync.aligned.32x32b.x128.max.abs.NaN.f32`, with store wait
  before the reduction and load wait after the reduction.
- `tmem_ld_red_non_f32_contract_kernel`: LLIR/PTX contain ordinary
  `tcgen05.ld.sync.aligned.32x32b.x128.b32` and no hardware `ld.red`, matching
  the expected non-f32 software-reduce classification.
- `tmem_copy_no_scales_warpx2_candidate_kernel`: LLIR/PTX both contain
  `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b`, single-CTA commit, and
  loadback via `tcgen05.ld.sync.aligned.32x32b.x4.b32`.
- `tmem_copy_no_scales_twocta_kernel`: LLIR/PTX both contain two
  `tcgen05.cp.cta_group::2.128x256b` operations, two-CTA multicast commit,
  and loadback via `tcgen05.ld.sync.aligned.32x32b.x16.b32`. PTX has the
  expected `fence.proxy.async.shared::cluster` and cluster barriers; LLIR has
  the matching NVVM proxy/barrier intrinsics.
- `mma_kernel`: LLIR/PTX both contain four
  `tcgen05.mma.cta_group::1.kind::f16` operations, single-CTA commit, and
  loadback via `tcgen05.ld.sync.aligned.32x32b.x128.b32`.
- `tmem_mma_scaled_layout_format_kernel`: TTGIR has
  `ttng.tc_gen5_mma_scaled`; LLIR/PTX both contain four
  `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`
  operations, scale stores via `tcgen05.st.sync.aligned.16x128b.x1.b32`,
  single-CTA commit, and loadback via `32x32b.x128`.
- `tcgen05_mma_multicast_commit_kernel`: LLIR/PTX both contain two
  `tcgen05.mma.cta_group::2.kind::f16` operations, one
  `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`,
  two-CTA alloc/relinquish/dealloc, and cluster barrier ordering before
  deallocation.

## Classification

- `PASS`: `172` focused runtime-matrix rows, `39` proxy/mbarrier/multicast
  rows, and `9` artifact-dump rows.
- `CLEAN_UNSUPPORTED`: none newly encountered in this lane; existing
  software-reduce sentinels intentionally pass and confirm no accidental
  hardware `ld.red` emission for non-f32 reductions.
- `BUG`: none. No PTX-vs-LLIR opcode mismatch, missing hardware opcode,
  unexpected software fallback, verifier over-strictness, compiler crash,
  runtime miscompile, or new independent `FZ-*` candidate was found.

## Notes

- The only harness issue was the initial collection without `PYTHONPATH=.:./python`;
  this did not reach TMEM compilation.
- The dumped artifacts in `/tmp/tmem_round54_opcode_dump` are temporary
  evidence for this lane and are not checked in.
