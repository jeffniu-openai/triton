# Round 22 Lane BB: `tcgen05.ld.red` Layout Fuzzing

Date: 2026-04-21 12:22 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes, no commit, no push.

## Scope

This lane fuzzed `tcgen05.ld.red` and adjacent software-fallback paths around:

- M64 split-N hardware reductions;
- M=128,N=32 and M=128,N=128 linear layouts;
- row-basis and column-basis permutations;
- min/max, `abs`, and NaN-propagating modifiers;
- explicit split-N variants;
- descriptor-chain reductions;
- 1CTA/2CTA indexed structural-fuzzer sentinels; and
- non-f32 and scales-layout software-reduce boundaries.

Existing runtime-matrix helpers were used wherever possible. One temporary
probe filled a cross-product gap for M64 row/column permutations with
`abs`/NaN modifiers and M=128,N=32 neighboring layouts.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Broad runtime-matrix selector, split across all four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  --store-durations --durations-path /tmp/tmem_r22_ldred_layouts_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource' \
  2>&1 | tee /tmp/tmem_r22_ldred_g1.log

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  --store-durations --durations-path /tmp/tmem_r22_ldred_layouts_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource' \
  2>&1 | tee /tmp/tmem_r22_ldred_g2.log

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  --store-durations --durations-path /tmp/tmem_r22_ldred_layouts_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource' \
  2>&1 | tee /tmp/tmem_r22_ldred_g3.log

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  --store-durations --durations-path /tmp/tmem_r22_ldred_layouts_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource' \
  2>&1 | tee /tmp/tmem_r22_ldred_g4.log
```

Focused controls:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'ldred' \
  2>&1 | tee /tmp/tmem_r22_structural_ldred.log

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red_non_f32 or ld_red_scales or ld_red_explicit_n_sharded' \
  2>&1 | tee /tmp/tmem_r22_ldred_fallback_dtype.log

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red_m64_splitn_linear_layout or ld_red_m64_explicit_splitn_variants or ld_red_explicit_compatible_layout_variants' \
  2>&1 | tee /tmp/tmem_r22_ldred_m64_modifiers.log

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red_descriptor_chain and not reports' \
  2>&1 | tee /tmp/tmem_r22_ldred_descriptor_chain.log
```

Temporary cross-product probe:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python3 /tmp/tmem_r22_ldred_layout_probe.py \
  2>&1 | tee /tmp/tmem_r22_ldred_layout_probe.log
```

The probe was run from an inline `/tmp` script. It used
`tmem_ld_red_m64_explicit_layout_kernel`,
`tmem_ld_red_explicit_layout_kernel`, and the existing runtime-matrix layout
helpers.

## Results

| Slice | Result |
| --- | --- |
| Broad runtime-matrix `ld_red and not reports and not resource` | `237 passed, 6 failed` |
| Structural-fuzzer `ldred` sentinels | `3 passed, 8 xfailed` |
| Non-f32/scales/n-sharded fallback boundaries | `60 passed` |
| M64 split-N and modifier controls | `38 passed` |
| Descriptor-chain reductions | `30 passed` |
| Temporary M64/M128 cross-product probe | `8` M64 row-basis failures, `16` adjacent controls passed |

The six broad-selector failures are the same M64 row-basis rows already
classified as `FZ-20260421-0012`:

| Runtime-matrix row | Result | Classification |
| --- | --- | --- |
| `ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]` | compile-time parse failure: `unsupported dst layout` | `FZ-0012` |
| `ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]` | compile-time parse failure: `unsupported dst layout` | `FZ-0012` |
| `ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]` | compile-time parse failure: `unsupported dst layout` | `FZ-0012` |
| `ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]` | compile-time parse failure: `unsupported dst layout` | `FZ-0012` |
| `ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]` | compile-time parse failure: `unsupported dst layout` | `FZ-0012` |
| `ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]` | compile-time parse failure: `unsupported dst layout` | `FZ-0012` |

## Opcode Evidence

Passing M64 column-only controls emit split-N hardware reductions:

```text
M64 row=identity col=reverse n=32 min:
  tcgen05.ld.red.sync.aligned.16x32bx2.x8.min.f32
M64 row=identity col=reverse n=32 max:
  tcgen05.ld.red.sync.aligned.16x32bx2.x8.max.f32
M64 row=identity col=reverse n=32 min abs NaN:
  tcgen05.ld.red.sync.aligned.16x32bx2.x8.min.abs.NaN.f32
M64 row=identity col=reverse n=32 max abs NaN:
  tcgen05.ld.red.sync.aligned.16x32bx2.x8.max.abs.NaN.f32
```

Adjacent M=128,N=32 layouts, including row permutations, emit canonical
`32x32b.x32` hardware reductions and pass runtime checks:

```text
M128_N32_identity min:
  tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32
M128_N32_identity max:
  tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32
M128_N32_identity min abs NaN:
  tcgen05.ld.red.sync.aligned.32x32b.x32.min.abs.NaN.f32
M128_N32_row_reverse min:
  tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32
M128_N32_row_reverse max:
  tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32
M128_N32_row_reverse min abs NaN:
  tcgen05.ld.red.sync.aligned.32x32b.x32.min.abs.NaN.f32
M128_N32_col_reverse min:
  tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32
M128_N32_tile8 min:
  tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32
```

The fallback/dtype selector passed all checked rows and asserted there were no
`.ld.red.` opcodes for non-f32, scales-layout, or explicitly n-sharded
software-reduce rows.

## Classification Against Existing Buckets

- `FZ-20260421-0004`: structural-fuzzer 2CTA indexed `ld.red` rows remain
  strict xfails. Round 22 found no new variant beyond the existing plain-load
  instead of `.ld.red.` opcode-loss bucket.
- `FZ-20260421-0005/0009`: structural-fuzzer allocator/assertion sentinels
  remained strict xfails. No new allocator assertion surfaced in this lane.
- `FZ-20260421-0006`: structural-fuzzer transpose/slice `ld.red` cleanly
  remained a known strict xfail. Descriptor-chain runtime-matrix rows outside
  that bucket passed.
- `FZ-20260421-0008`: structural-fuzzer 2CTA row/column-chain optimizer crash
  remained a known strict xfail. Round 22 did not find a new 2CTA row/column
  crash shape.
- `FZ-20260421-0012`: confirmed and sharpened. The failing condition is M64
  f32 hardware `ld.red` with a non-identity effective row basis. The failure
  persists across `min`, `max`, `abs`, NaN propagation, default layout
  selection, and explicit `32x32b` split-N requests. Column-only M64
  permutations pass and emit `16x32bx2` `.ld.red.` opcodes. Neighboring
  M=128,N=32 row-permuted layouts also pass and emit `32x32b.x32`, so the gap
  is specific to the M64 split-N destination-layout planner rather than
  row-permutation handling in all hardware reductions.

No non-overlapping issue was found, so no new `FZ-*` id is proposed from this
lane.

## Artifacts

```text
/tmp/tmem_r22_ldred_g1.log
/tmp/tmem_r22_ldred_g2.log
/tmp/tmem_r22_ldred_g3.log
/tmp/tmem_r22_ldred_g4.log
/tmp/tmem_r22_structural_ldred.log
/tmp/tmem_r22_ldred_fallback_dtype.log
/tmp/tmem_r22_ldred_m64_modifiers.log
/tmp/tmem_r22_ldred_descriptor_chain.log
/tmp/tmem_r22_ldred_layout_probe.log
/tmp/tmem_r22_ldred_layouts_durations.json
```

## Next Probe Ideas

- Fuzz M64 split-N with row-basis normalization attempts at TTGIR-only compile
  time, to distinguish a missing canonicalization from a true ISA shape limit.
- Add descriptor-chain M64 split-N cases once a compact M64 descriptor-chain
  helper exists; this lane covered M=128 descriptor-chain reductions and direct
  M64 reductions separately.
- Continue discovery on non-`ld.red` TMEM consumers before starting repairs;
  all Round 22 failures stayed inside existing `FZ-*` buckets.
