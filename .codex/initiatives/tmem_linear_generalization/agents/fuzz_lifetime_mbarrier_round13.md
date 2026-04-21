# Round 13 Local: TMEM Lifetime, Commit, and MBarrier Sanity

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Repo edit scope: this report only.

## Scope

This local lane checked existing runtime coverage around TMEM allocation
lifetime bookkeeping, commit opcode selection, multicast mbarrier plumbing,
descriptor-chain subviews, and physical-bitcast view mapping. It was intended
as a non-overlapping sanity pass after the FPSAN runtime-index lane.

## Commands and Results

Required rebuild before each runtime slice:

```bash
make -j8
```

Result each time: `ninja: no work to do`.

Core commit/mbarrier selector:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_core.py \
  -k 'mbarrier or commit or membar or multicast or warp_special'
```

Relevant collected rows for this lane:

```text
test_tma_multicast_copy[*]
test_tcgen05_mma_multicast_commit[*]
test_tcgen05_mma_multicast_commit_twocta_linear_acc[*]
test_tcgen05_mma_scaled_direct_multicast_barrier
test_async_copy_mbarrier
```

Split-4 runtime command:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> python/test/gluon/test_core.py \
  -k 'tma_multicast_copy or tcgen05_mma_multicast_commit or tcgen05_mma_scaled_direct_multicast_barrier or async_copy_mbarrier'
```

Result by group: `4 passed`, `4 passed`, `4 passed`, and `1 passed`;
aggregate `13 passed`.

Runtime-matrix allocation lifetime selector:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'alloc_lifetime or source_initialization_lifetime'
```

Result: `7/1615` collected.

Split-4 runtime command:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'alloc_lifetime or source_initialization_lifetime'
```

Result by group: `2 passed`, `2 passed`, `2 passed`, and `1 passed`;
aggregate `7 passed`.

Descriptor-chain and physical-bitcast view selector:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_core.py \
  -k 'tmem_alloc or tensor_memory or tmem_store or physical_bitcast or descriptor_chain'
```

Relevant selected rows:

```text
26 test_tmem_descriptor_chain_matrix rows
2  test_tmem_physical_bitcast_preserves_subview_mapping rows
1  test_tmem_physical_bitcast_mma_lhs row
```

Split-4 runtime command:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> python/test/gluon/test_core.py \
  -k 'tmem_descriptor_chain_matrix or tmem_physical_bitcast_preserves_subview_mapping or tmem_physical_bitcast_mma_lhs'
```

Result by group: `8 passed`, `8 passed`, `8 passed`, and `5 passed`;
aggregate `29 passed`.

## Classification

No new `FZ-*` candidate was found.

- Commit/mbarrier runtime coverage stayed green for TMA multicast,
  1CTA/2CTA MMAv5 multicast commit, scaled-MMAv5 multicast barrier, and async
  copy mbarrier rows.
- Allocation lifetime opcode coverage stayed green for 1CTA and 2CTA
  load/store allocations, including alloc/relinquish opcode matching.
- Descriptor-chain and physical-bitcast rows stayed green across multiple live
  allocations, subviews, and a TMEM-LHS MMA path.

This lane is a negative result only. It does not reduce or expand any existing
bug bucket.
