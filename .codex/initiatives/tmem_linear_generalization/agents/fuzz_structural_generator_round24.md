# Round 24 Lane BI: TMEM Descriptor Structural Generator Prototype

- Date: 2026-04-21 13:35 UTC
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler repairs attempted.
- Repo edit scope: this report only.
- Temporary prototype: `/tmp/tmem_structural_generator_round24.py`
- Logs:
  - `/tmp/tmem_structural_generator_round24.log`
  - `/tmp/tmem_structural_generator_round24_rerun.log`

## Scope

This lane prototyped a small structural generator for TMEM descriptor-view
programs.  The goal was not broad random coverage yet; it was to confirm a
repeatable generator shape that can be promoted into checked-in runtime fuzz
tests after a few more seed rounds.

The prototype currently drives existing Gluon kernels from
`python/test/gluon/test_tmem_runtime_matrix.py`, but selects cases through a
generator-style matrix rather than a hand-picked pytest node list.  That keeps
the first seed set executable while still recording the structural dimensions
that should become first-class generator axes.

## Generator Dimensions

The first prototype varies:

- consumer family:
  - direct `ttng.tmem_load`/`ttng.tmem_store`;
  - descriptor-chain `load/store`;
  - lifted rank-4 descriptor roundtrip `load/store`;
  - M64 f32 `ld.red`;
  - no-scales `tcgen05.cp` with indexed and slice+indexed descriptor views.
- view chain:
  - none;
  - `index`;
  - `slice + index`;
  - `index + reshape + permute + reshape + trans roundtrip + slice`;
  - lifted rank-4 `slice/index` chains with reshape/permute variants.
- linear layout basis:
  - identity;
  - mixed row/column bases;
  - row/column-permuted bases;
  - M64 row-permuted and column-permuted `ld.red` contrasts.
- geometry:
  - `M=128, N=32/64/128` for ordinary load/store;
  - `M=64, N=32/128` for M64 `ld.red`;
  - `M=128, N=4` for warpx2 copy probes.
- expected classification:
  - green runtime rows;
  - known `FZ-20260421-0012` row-basis failures;
  - clean unsupported descriptor-view boundaries.

The prototype records TTGIR/PTX feature bits for each passing row:
`memdesc_index`, `memdesc_subslice`, `memdesc_reshape`, `memdesc_trans`,
`ttng.tmem_copy`, MMAv5/scaled-MMAv5 markers, and `.ld.red.` opcodes.

## Commands

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Prototype syntax check:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_structural_generator_round24.py
```

First seed run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_structural_generator_round24.py \
  2>&1 | tee /tmp/tmem_structural_generator_round24.log
```

The first run exposed two prototype-classification issues: the runtime
exception wrapper hid the full M64 `ld.red` diagnostic from Python, and one
invalid mixed-layout seed used `N=32` even though the local helper requires
`N >= 64`.  The prototype was corrected in `/tmp` only.

Rerun:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_structural_generator_round24.py \
  2>&1 | tee /tmp/tmem_structural_generator_round24_rerun.log
```

## Rerun Results

The corrected seed set ran 12 generated cases:

| Case | Consumer / chain | Result | Classification |
| --- | --- | --- | --- |
| `ldst_direct_identity_n32_auto` | direct load/store, no view | pass | green |
| `ldst_chain_mixed_n128_auto` | index + subslice + reshape + trans chain | pass | green |
| `ldst_chain_row_rotate_col_even_odd_n128` | row/column-permuted descriptor chain | pass | green |
| `ldst_roundtrip_identity_n32_chain0` | lifted rank-4 roundtrip chain 0 | error | clean unsupported |
| `ldst_roundtrip_mixed_n64_chain1` | lifted rank-4 roundtrip chain 1 | error | clean unsupported |
| `ldst_roundtrip_identity_n64_chain2` | lifted rank-4 roundtrip chain 2 | error | clean unsupported |
| `ldred_m64_col_reverse_n32` | M64 column-only f32 `ld.red` | pass | green |
| `ldred_m64_row_reverse_n32` | M64 row-reversed f32 `ld.red` | error | `FZ-20260421-0012` |
| `ldred_m64_row_rotate_col_even_odd_n128` | M64 row+column f32 `ld.red` | error | `FZ-20260421-0012` |
| `copy_indexed_m128` | no-scales copy through `index` view | pass | green |
| `copy_warpx2_slice_index_parent0` | warpx2 copy through `slice + index` view | pass | green |
| `copy_warpx2_slice_index_parent1` | warpx2 copy through alternate parent index | pass | green |

Summary:

```text
{"FZ-20260421-0012": 2, "clean-unsupported": 3, "green": 7}
```

No new independent `FZ-*` bucket was found in this seed set.

## Classification Notes

The two M64 row-basis failures are existing `FZ-20260421-0012`.  They print the
same `ttng.tmem_load` reduction diagnostic on stderr:

```text
Failed to lower TMEM load/store: unsupported dst layout
```

The passing M64 column-only control emits hardware `.ld.red.` and remains an
important contrast.

The three lifted rank-4 descriptor roundtrip rows hit the current clean
unsupported direct `tcgen05.ld/st` descriptor-view boundary:

```text
unsupported tensor memory descriptor view for direct tcgen05.ld/st:
required row anchors 32,64 are not directly representable in the descriptor view
```

Those rows are useful generator guardrails but are not new bugs unless a later
hardware-feasibility audit decides the row anchors can be decomposed into legal
packet bases and offsets.  For now they should stay in a clean-boundary bucket.

The green rows covered:

- direct load/store with no descriptor view;
- descriptor chains containing `memdesc_index`, `memdesc_subslice`,
  `memdesc_reshape`, and `memdesc_trans`;
- row/column-permuted descriptor-chain load/store;
- M64 column-permuted hardware `ld.red`;
- no-scales copy through `memdesc_index`;
- warpx2 copy through `memdesc_subslice + memdesc_index`.

## Promotion Plan

The next promoted generator should avoid relying on pytest test functions while
still reusing local kernels and layout builders.  Recommended axes:

- explicit seed object:
  `(consumer, dtype, M, N, rank_lift, view_ops, row_basis, col_basis,
  instr_variant, cta_group, expected_class)`;
- deterministic seed IDs and a compact `pytest.mark.parametrize` table for a
  first checked-in shard;
- stderr-aware classifier for Gluon parsing failures so `FZ-0012` and similar
  frontend diagnostics are recognized without relying only on Python exception
  text;
- separate green, known-xfail, and clean-boundary selectors so broad sweeps do
  not mix executable runtime rows with expected diagnostics;
- a second-generation compiler-only path that emits TTGIR/MLIR repros for
  generated unsupported rows, especially descriptor-view chains that are close
  to the current clean boundary.

High-value next seed families:

- dynamic descriptor SSA selection crossed with the generated view chains,
  targeting existing `FZ-20260421-0001` and `FZ-20260421-0002`;
- M64 `ld.red` row-basis mutations with explicit split-N variants, extending
  `FZ-20260421-0012`;
- scaled-MMAv5 accumulator and scale descriptor views with allocation-order
  permutations, extending `FZ-20260421-0007`, `FZ-20260421-0013`, and
  `FZ-20260421-0015`;
- high-CGA launch contexts with local 1CTA/2CTA generated layouts, extending
  `FZ-20260421-0010`.

No backend/compiler code was changed.
