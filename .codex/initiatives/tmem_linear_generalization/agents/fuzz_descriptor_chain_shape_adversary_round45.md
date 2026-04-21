# Round 45 Descriptor-Chain Shape Adversary

Date: 2026-04-21
Branch: `codex/tmem`
Scope: TMEM descriptor-view chains with rank/shape/layout perturbations. This lane was cataloging only; no backend or test files were modified.

## Summary

This pass exercised checked-in Gluon runtime and structural-fuzzer coverage around descriptor-view chains:

- rank/stride-preserving identity and mixed descriptor chains
- reshape/slice chains around N64/N128/N256 and M64 reduction cases
- row/column permutation descriptor chains
- unit-dim high-rank and parent-index-style descriptor chains
- existing structural descriptor-chain expected-failure sentinels

No new bug bucket is needed. The only unexpected-red-at-command-level rows were the already-cataloged M64 `ld.red` unsupported destination-layout cases, classified as `FZ-0012`. Structural xfails remained expected and map to existing buckets (`FZ-0003`, `FZ-0004`, and related descriptor-chain sentinels).

## Commands And Results

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Collected broad descriptor-chain runtime surface:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k '(ldst_descriptor_roundtrip_sweeps or ldst_descriptor_roundtrip_rowcol_permuted_sweeps or ldst_descriptor_rank5 or ldst_descriptor_higher_rank or ldst_descriptor_multidim_slices or ld_red_descriptor_chain_n_sweep)'
```

Result: `123/1615 tests collected`.

Four-GPU runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(ldst_descriptor_roundtrip_sweeps or ldst_descriptor_roundtrip_rowcol_permuted_sweeps or ldst_descriptor_rank5 or ldst_descriptor_higher_rank or ldst_descriptor_multidim_slices or ld_red_descriptor_chain_n_sweep)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k '(ldst_descriptor_roundtrip_sweeps or ldst_descriptor_roundtrip_rowcol_permuted_sweeps or ldst_descriptor_rank5 or ldst_descriptor_higher_rank or ldst_descriptor_multidim_slices or ld_red_descriptor_chain_n_sweep)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k '(ldst_descriptor_roundtrip_sweeps or ldst_descriptor_roundtrip_rowcol_permuted_sweeps or ldst_descriptor_rank5 or ldst_descriptor_higher_rank or ldst_descriptor_multidim_slices or ld_red_descriptor_chain_n_sweep)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k '(ldst_descriptor_roundtrip_sweeps or ldst_descriptor_roundtrip_rowcol_permuted_sweeps or ldst_descriptor_rank5 or ldst_descriptor_higher_rank or ldst_descriptor_multidim_slices or ld_red_descriptor_chain_n_sweep)'
```

Results:

```text
group 1: 31 skipped, 1584 deselected
group 2: 11 passed, 20 skipped, 1584 deselected
group 3: 21 passed, 10 skipped, 1584 deselected
group 4: 30 passed, 1585 deselected
aggregate selected: 62 passed, 61 skipped
```

M64/N64/N128 reduction and descriptor-shape pass:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k '(ld_red_m64 or splitn_16bit_m64_auto_matches_explicit)'
```

Result:

```text
6 failed, 35 passed, 1574 deselected
```

Failed nodeids:

- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]`

Failure signature:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
Failed to lower TMEM load/store: unsupported dst layout
```

Classification: existing `FZ-0012` M64 f32 `tcgen05.ld.red` destination-layout planner gap for non-identity row basis. No new bucket.

Structural descriptor-chain sentinels:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py -k '(ldst_view_roundtrip or ldst_descriptor_view_read or ldred)'
```

Result:

```text
7 passed, 15 deselected, 11 xfailed
```

Classification: expected structural xfails only. These continue to cover already-known descriptor-chain buckets, notably `FZ-0003` for ld/st descriptor-view chain mapping/packet-order failures and `FZ-0004` for `ld.red` descriptor-chain opcode loss/software-reduce behavior. No unexpected verifier or compiler crash drift was observed here.

Unit-dim high-rank and parent-index-style descriptor chains:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k '(ldst_descriptor_rank5_unit_parent_n256_roundtrip or ldst_descriptor_rank5_small_roundtrip or ldst_descriptor_multidim_slice_replays or ldst_descriptor_multidim_slice_positive)'
```

Result:

```text
26 passed, 1589 deselected
```

Descriptor-composition perturbation sweep:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py -k '(ldst_descriptor_compositions or ldst_i32_broad_linear_layouts) and not roundtrip'
```

Result:

```text
36 passed, 1579 deselected
```

## Diagnosis

The checked-in runtime matrix is green for the descriptor-view shape perturbations outside known clean skips and known M64 `ld.red` destination-layout failures. Rank-5 unit-parent chains, small/high-rank chains, multidim slice replays, and descriptor-composition variants did not expose fresh miscompiles or new unsupported cases.

The M64 failures are not new and match `FZ-0012`: row-basis permutations in M64 reduction layouts still reach the `ttng.tmem_load` reduction planner with a register/lane/warp basis it cannot lower as a destination layout. The row-reverse N32 and row-rotate/col-even-odd N128 forms both fail during parsing/JIT IR construction with the same unsupported-destination-layout diagnostic.

The structural fuzzer remained stable: expected descriptor-chain xfails stayed xfailed, and no `FZ-0019` dimension-name mismatch or `FZ-0021` half-column boundary manifested as a new unexpected failure in this selected pass.

## New Bucket Decision

No new FZ bucket needed.

