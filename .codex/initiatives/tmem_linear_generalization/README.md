# TMEM Linear Generalization Initiative

This directory is the durable coordination space for the TMEM linear-layout
generalization project.

Keep this README up to date when the role of any document changes, when a new
current-state handoff supersedes an older one, or when the source-of-truth
entry points change.

## How To Use This Folder

When resuming the initiative:
- start here;
- then read `memory.md` as the initiative's durable memory/source of truth;
- use `gb200_nvidia_ci_inventory.md` for the current GB200/NVIDIA CI baseline
  and confirmed red list;
- use `gb200_failure_manifest.md` plus the adjacent generated `.txt` files
  when you need the exact nodeid inventories for the current branch or the
  merge-base-existing shard-3 subsets;
- use `gb200_failure_classification_20260412.md` for the current split between
  actual compiler/runtime bugs, stale tests, tests requiring API/contract
  updates, and merge-base-preexisting noise;
- use `gb200_preserve_set_20260412.md` for the current pre-cleanup preserve-set
  before removing branch-only TMEM physical-layout / row-plan / MMAv5-root
  attributes and the hacks that depend on them;
- use `gb200_branch_recovery_plan.md` when you need the prioritized
  branch-caused recovery backlog and the exact representative repros to drive
  fixes;
- then read the tail of the active handoff for the latest live buckets and next
  concrete steps;
- use `log.md` for chronological provenance and validation history; and
- use `fuzz_plan.md` once the current planner/cleanup blockers are green and
  broad saturation work is the next task.
- use `ldst_validation_recipe_20260413.md` for the current duration-cache
  and bucketed recipe for broad `ld/st` runtime-matrix validation.
- use `tmem_runtime_matrix_validation_recipe_20260413.md` and
  `run_tmem_runtime_matrix_sweep.py` for the full 3639-case runtime-matrix
  sweep; this is the coverage-preserving replacement for raw static split-4
  full-file runs that time out while still making progress.

## Current Project Invariant

- `_reinterpret` is not an escape hatch for relying on whichever physical TMEM
  lowering the compiler happens to choose.
- Production kernels, examples, and tests that currently depend on private
  `_reinterpret` behavior or compiler-specific physical-layout knowledge are
  migration targets.
- The intended replacement is a supported descriptor operation sequence:
  offset to the right TMEM region, slice/subview to the desired physical bits,
  then bitcast to the desired dtype/shape/layout only when total bit size and
  the exact physical TMEM mapping are preserved.
- Operationally, the offset/subview step selects the actual physical TMEM bits
  first; the bitcast only changes the descriptor's dtype/shape/layout view over
  that already-selected physical image.
- This means the bitcast may express the desired dtype/shape/layout only after
  the input descriptor already denotes exactly the same physical TMEM image; it
  must not perform physical selection itself.
- In the concrete attention-style pattern, the kernel first offsets to the
  right part of TMEM, then takes the subview that denotes the exact desired
  bits, then performs a size-and-physical-mapping-equivalent bitcast to the
  desired dtype, shape, and layout.
- Equivalently: offset to the right part of TMEM, slice/subview to the desired
  bit image, then bitcast only to a dtype/shape/layout view whose total size
  and physical mapping are equivalent to that sliced descriptor.
- The bitcast step must not change which physical TMEM memory the input
  descriptor maps to; if the desired view is not equal-size and
  physical-mapping equivalent, use a different supported API that matches the
  real kernel intent.
- Said another way, this is an offset/subview over the existing physical TMEM
  allocation followed by a size-and-physical-mapping-equivalent type/view
  reinterpretation, not a remapping or relocation operation.
- The attention example's scratch-borrow path has a focused supported
  migration: helpers now offset/slice the f32 scratch TMEM before bitcasting,
  and the exp2 partition stores slice the exact physical subregion before
  bitcasting each bf16 view. Preserve this synchronization-aware pattern if
  broader attention coverage exposes more aliasing cases.
- Lowering-side fixes are still appropriate for supported APIs that miscompile,
  but do not add ad-hoc selectors just to preserve old `_reinterpret`
  accidents.
- TMEM layouts are the source of truth. A zero TMEM basis means
  broadcast/equivalence semantics: physical coordinates that map to the same
  logical tensor element must agree, and codegen must preserve that invariant.
- Do not justify ld/st behavior as choosing a `live` representative among
  divergent physical cells. Divergence inside a zero-basis equivalence class is
  itself a bug in producer, view, or load/store lowering.
- The branch-only TMEM attributes (`ttng.tmem_physical_layout`,
  `ttng.tmem_ldst_row_plan`, `ttng.tmem_mmav5_accumulator_root`, and related
  root markers) are now treated as papering over underlying layout/codegen bugs.
  The next cleanup target is to remove them and make lowering respect the
  layout's broadcast and physical mapping directly.

## Current Checkpoint

- Current `ld/st` broad two-CTA i32 checkpoint, 2026-04-14 04:08 UTC: `test_tmem_runtime_matrix_ldst_twocta_i32_broad_linear_layouts` adds integer `.b32` payload parity for the broad two-CTA `256x{64,128,256}` root surfaces. It covers `block_two_ctas` and MMAv5-like two-CTA TMEM-linear layouts, direct and supported descriptor-chain modes, and every public `ld/st` variant; existing broad f32 two-CTA coverage remains in the original direct/descriptor tests. Current runtime-matrix collection is `3789` tests: `cp=322`, `mma=502`, splitn/misc `=252`, `ld_red=811`, and `ldst=1902`; bucketed evidence now aggregates to `3343 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `60/3789`; no-PYTHONPATH `ldst` collect selected `1902/3789`; focused two-CTA i32 broad selector passed all `60` cases across four GPUs (`15` each; slowest cold descriptor-heavy group took about `5:04`); `git diff --check` passed.

- Current `ld/st` broad single-CTA i32 checkpoint, 2026-04-14 04:01 UTC: `test_tmem_runtime_matrix_ldst_i32_broad_linear_layouts` adds integer `.b32` payload parity for the existing broad single-CTA `128x{64,128,256}` root surfaces. It covers identity and mixed TMEM-linear layouts, direct and supported descriptor-chain modes, and every public `ld/st` variant; existing f32 coverage remains in the original direct/descriptor tests. Current runtime-matrix collection is `3729` tests: `cp=322`, `mma=502`, splitn/misc `=252`, `ld_red=811`, and `ldst=1842`; bucketed evidence now aggregates to `3283 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `60/3729`; no-PYTHONPATH `ldst` collect selected `1842/3729`; focused i32 broad selector passed all `60` cases across four GPUs (`15` each; cold descriptor-heavy group 4 took about `6:17`); `git diff --check` passed.

- Current `ld/st` two-CTA N=32 32-bit dtype checkpoint, 2026-04-14 03:52 UTC: `test_tmem_runtime_matrix_ldst_twocta_n32_linear_layout` now covers both f32 and i32 for `256x32` two-CTA TMEM-linear roundtrips, across direct access and the supported descriptor-chain path for both `block_two_ctas` and MMAv5-like two-CTA layouts, over every public `ld/st` variant. This reuses the dtype-generic direct/descriptor helper kernels from the prior single-CTA N=32 slice and pins the minimal `x32/x16/x8/x4` opcode families for the two-CTA layouts. Current runtime-matrix collection is `3669` tests: `cp=322`, `mma=502`, splitn/misc `=252`, `ld_red=811`, and `ldst=1782`; bucketed evidence now aggregates to `3223 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `40/3669`; no-PYTHONPATH `ldst` collect selected `1782/3669`; focused two-CTA N=32 selector passed all `40` cases across four GPUs (`10` each), with cold descriptor groups taking about `2:18`; `git diff --check` passed.

- Current copy `warpx2` diagnostic checkpoint, 2026-04-14 03:47 UTC: the remaining no-scales two-CTA `warpx2::02_13` clean-negative diagnostic now records the concrete missing requirement: support needs a `cta_group::2` descriptor/address schedule that preserves the high source-column bit. A scratch direct-PTX decomposition check confirmed that replacing the two-CTA copy with `cta_group::1` is not a viable workaround because the same function still contains `cta_group::2` TMEM alloc/dealloc and ptxas rejects mixed single-CTA and CTA-pair granularity (`uses single CTA(.cta_group::1) and CTA pair granularity(.cta_group::2) and that is not allowed`). Existing direct source-offset/destination-delta scans and true-scales source-offset scans remain negative, so the production boundary is still clean unsupported rather than an alias or software split. Validation: `make -j8` rebuilt `Ops.cpp` and relinked; py-compile passed; focused two-CTA `02_13` clean-negative selector passed the two non-empty split groups; nearby no-PYTHONPATH `-k warpx2` selector passed all `17` selected cases across four GPUs (`5`, `5`, `5`, `2`).

- Current `ld/st` identity N=32 32-bit dtype checkpoint, 2026-04-14 03:45 UTC: `test_tmem_runtime_matrix_ldst_identity_n32_linear_layout` now covers both f32 and i32 for canonical root `128x32` identity TMEM-linear roundtrips, across direct access and the supported descriptor-chain path for every public `ld/st` variant. The shared `tmem_ldst_auto_kernel`, `tmem_ldst_variant_kernel`, and `tmem_ldst_descriptor_chain_kernel` now allocate/bitcast TMEM with `in_ptr.dtype.element_ty`; this preserves existing f32 coverage while allowing the `.b32` ISA path to be exercised by integer payloads. Current runtime-matrix collection is `3649` tests: `cp=322`, `mma=502`, splitn/misc `=252`, `ld_red=811`, and `ldst=1762`; bucketed evidence now aggregates to `3203 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `20/3649`; focused identity N=32 selector passed all `20` cases across four GPUs (`5` each); four representative existing f32 direct/descriptor nodeids passed across four split groups; `git diff --check` passed.

- Current scales `warpx2` direct-source-offset checkpoint, 2026-04-14 03:31 UTC: completed the missing `sourceOffsetB128=73..119` high-half scan for tensor-memory-scales `tcgen05.copy` `warpx2::{01_23,02_13}` with `dst_delta=4`, filling the tracked evidence range to every offset `0..136`. The combined tracked result is now `274` unprimed single-message records with zero matches: offsets `0..8` execute but produce wrong data (`18` records), and offsets `9..136` all fault (`256` records). New shards are `probe_cp_scales_warpx2_offsets_73_84_dst4_gpu0.jsonl`, `..._85_96_dst4_gpu1.jsonl`, `..._97_108_dst4_gpu2.jsonl`, and `..._109_119_dst4_gpu3.jsonl`; compact summary is `probe_cp_scales_warpx2_offsets_dst4_current_summary.json`. No production lowering change is justified by this sweep; true scales `warpx2` remains a descriptor/address/view/staging frontier rather than a source-offset alias of the canonical `warpx4` scales copy.

- Current plain-MMAv5 `blockM=64` checkpoint, 2026-04-14 03:24 UTC: `test_tmem_runtime_matrix_mma_plain_kinds_m64` adds one-CTA root `M=64, N=128` MMAv5 coverage for every supported plain operand kind, both legacy M64 sugar and canonical M64 TMEM-linear accumulator layouts, `K in {32, 64}`, and both no-accumulator and `use_acc=True` paths. Exact opcode counts are pinned with `_expected_m64_plain_mma_op_count(kind, k, acc_layout_kind)`, including the legacy M64 sugar's doubled opcode stream relative to the canonical M64 linear layout. Current runtime-matrix collection is `3639` tests: `cp=322`, `mma=502`, splitn/misc `=252`, `ld_red=811`, and `ldst=1752`; bucketed evidence now aggregates to `3193 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `40/3639`; focused M64 selector passed all `40` cases across four GPUs (`10` each); no-PYTHONPATH tight `mma` collect selected `502/3639`; no-PYTHONPATH full-file collect selected all `3639`; tight `mma` runner passed all `502` cases across four groups (`126`, `126`, `126`, `124`); `git diff --check` passed. Remaining long-term coverage work: continue staged ISA saturation in another bounded family; the old legacy M64 xfail docket remains stale on current head, and this matrix now makes M64 MMAv5 coverage explicit in the runtime matrix.

- Prior `ld/st` exotic N=32 direct/descriptor checkpoint, 2026-04-14 03:20 UTC: `test_tmem_runtime_matrix_ldst_exotic_n32_linear_layout` adds root `128x32` f32 roundtrips for the existing scrambled-column and scrambled-row+column TMEM-linear layout families, covering both direct TMEM access and supported descriptor-chain views across every public `ld/st` variant. The slice pins the minimal `32x32b.x32`, `16x64b.x16`, `16x128b.x8`, and `16x256b.x4` opcode families in PTX/LLIR while keeping the descriptor path on `tensor_memory_linear`. Current runtime-matrix collection is `3599` tests: `cp=322`, `mma=462`, splitn/misc `=252`, `ld_red=811`, and `ldst=1752`; bucketed evidence now aggregates to `3153 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `20/3599`; no-PYTHONPATH `ldst` collect selected `1752/3599`; focused exotic N=32 selector passed all `20` cases across four GPUs (`5` each), with slow cold descriptor splits at about `92s` and `113s`; `git diff --check` passed. Remaining long-term coverage work: continue staged ISA saturation in another bounded family; reserve broad `ldst` runner refresh for shared-lowering changes or a larger accumulated `ld/st` checkpoint.

- Prior scaled-MMAv5 TMEM-LHS tile-permuted K=128 checkpoint, 2026-04-14 03:16 UTC: full-shape tile-permuted TMEM-LHS coverage now includes the reachable `K=128` positive surface for mxfp8-storage operand-A formats (`mxfp8/mxfp8` and `mxfp8/mxfp4`) at `N in {128, 256}` and both legacy/canonical accumulator layouts, while retaining the existing `K=256` coverage for every packed-storage reachable format pair. The fp4-storage `K=128` tile-permuted LHS boundary (`mxfp4/mxfp4` and `nvfp4/nvfp4`) is now pinned as a clean verifier negative because the resulting storage-width-64 tile-permuted descriptor is not MMAv5-compatible. Current runtime-matrix collection is `3579` tests: `cp=322`, `mma=462`, splitn/misc `=252`, `ld_red=811`, and `ldst=1732`; bucketed evidence now aggregates to `3133 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `32/3579`; no-PYTHONPATH tight `mma` collect selected `462/3579`; no-PYTHONPATH full-file collect selected all `3579`; focused scaled-LHS tile-permuted selector passed all `32` cases across four GPUs (`8` each); tight `mma` runner passed all `462` cases across four groups (`116`, `116`, `116`, `114`). Remaining long-term coverage work: continue staged ISA saturation in another bounded family, likely descriptor-view `ld/st` breadth or another scaled-MMAv5 clean-negative/positive parity gap.

- Prior plain-MMAv5 tile-permuted accumulator K-depth checkpoint, 2026-04-14 03:10 UTC: the tile-permuted accumulator matrices now cover `K in {32, 64}` for the simple f16 anchor and for every supported plain operand kind across no-accumulator and `use_acc=True` paths, using `128x128/tile_n=32` and `128x256/tile_n=64` canonical TMEM-linear accumulator layouts. `_expected_tile_permuted_mma_op_count(kind, k)` pins exact `tcgen05.mma` opcode counts as the existing tile-permuted base count times `K // 32`, so the new `K=64` cases prove doubled instruction depth. Current runtime-matrix collection is `3563` tests: `cp=322`, `mma=446`, splitn/misc `=252`, `ld_red=811`, and `ldst=1732`; bucketed evidence now aggregates to `3117 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `44/3563`; no-PYTHONPATH tight `mma` collect selected `446/3563`; no-PYTHONPATH full-file collect selected all `3563`; focused tile-permuted selector passed all `44` cases across four GPUs (`11` each); tight `mma` runner passed all `446` cases across four groups (`112`, `112`, `112`, `110`). Remaining long-term coverage work: continue staged ISA saturation in another bounded family, likely scaled-MMAv5 parity or descriptor-view `ld/st` breadth.

- Prior plain-MMAv5 TMEM-LHS subview K-depth checkpoint, 2026-04-14 03:07 UTC: `tmem_mma_lhs_subslice_kernel` now takes `K` as a constexpr and `test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds` covers `K in {32, 64}` for every supported plain operand kind, both legacy/canonical accumulator layouts, and `N in {128, 256}`. The exact `tcgen05.mma` opcode count uses `_expected_plain_mma_op_count(kind, k)`, so the new `K=64` subview cases prove doubled instruction depth through the supported `ttg.memdesc_subslice` TMEM-LHS path. Current runtime-matrix collection is `3541` tests: `cp=322`, `mma=424`, splitn/misc `=252`, `ld_red=811`, and `ldst=1732`; bucketed evidence now aggregates to `3095 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `40/3541`; no-PYTHONPATH tight `mma` collect selected `424/3541`; focused LHS-subview selector passed all `40` cases across four GPUs (`10` each); tight `mma` runner passed all `424` cases across four groups (`106` each). Remaining long-term coverage work: continue staged ISA saturation in another bounded family, likely scaled-MMAv5 parity or descriptor-view `ld/st` breadth.

- Prior `ld.red` tile-permuted checkpoint, 2026-04-14 03:04 UTC: `LD_RED_TILE_PERMUTED_CASES` now covers every legal helper tile width currently proven for tile-permuted `128xN` f32 reductions: `N=32/tile_n=8`, `N=64/tile_n in {8,16}`, `N=128/tile_n in {8,16,32}`, and `N=256/tile_n in {8,16,32,64}`. This adds five tile-permuted layouts and `40` runtime cases across min/max and all legal abs/NaN modifiers, while continuing to pin exact `32x32b` reduction opcodes, offsets, and wait ordering. A probe of the apparent 256-row `N=256` reduction-friendly layout hit shared-memory OOR (`262148` required vs `232448` limit), so that remains a non-executable boundary rather than a positive target. Current runtime-matrix collection is `3521` tests: `cp=322`, `mma=404`, splitn/misc `=252`, `ld_red=811`, and `ldst=1732`; bucketed evidence now aggregates to `3075 passed, 446 skipped`. Validation: scratch tile-permutation probe passed all `10` legal `(N,tile_n)` pairs for the baseline reduction; py-compile passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `80/3521`; no-PYTHONPATH `ld_red` collect selected `811/3521`; focused tile-permuted selector passed all `80` cases across four GPUs (`20` each); full `ld_red` runner passed all `811` cases across split-16/xdist groups. Remaining long-term coverage work: continue staged ISA saturation in another bounded family, likely broader MMAv5/scaled-MMAv5 parity or another descriptor-view `ld/st` slice, while leaving blocked copy `warpx2` frontiers parked until a real descriptor/address model exists.

- Prior plain-MMAv5 K-depth checkpoint, 2026-04-14 02:51 UTC: plain one-CTA and two-CTA root/use-acc matrices now cover `K in {32, 64}` for every supported operand kind, both legacy/canonical accumulator layouts, and `N in {128, 256}`. The expected `tcgen05.mma` opcode count is now pinned as the per-`K=32` base count times `K // 32`, so the new `K=64` cases prove the doubled instruction-depth surface for f16/bf16, tf32, and f8/f6/f4 families. Current runtime-matrix collection is `3481` tests: `cp=322`, `mma=404`, splitn/misc `=252`, `ld_red=771`, and `ldst=1732`; bucketed evidence now aggregates to `3035 passed, 446 skipped`. Validation: one-CTA and two-CTA scratch probes passed `K=64` for all plain kinds with doubled opcode counts; py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH focused collect selected `160/3481`; focused K-depth selector passed all `160` cases across four GPUs (`40` each), slowest split about `1:27`; tight `mma` runner passed all `404` cases across four groups (`101` each). Remaining long-term coverage work: continue staged ISA saturation in another family or move to the next bounded MMA/scaled-MMA gap; no broad `ld/st` rerun is implied by this test-only MMA change.

- Prior `ld/st` N=32 row/column checkpoint, 2026-04-14 03:53 UTC: `test_tmem_runtime_matrix_ldst_rowcol_n32_linear_layout` adds representative non-diagonal row/column permutation coverage at root `128x32`: pure row permutation (`rotate1,identity`), pure column permutation (`identity,reverse`), and mixed row+column permutation (`even_odd,reverse`). Each case covers direct TMEM access and supported descriptor-chain views across every public `ld/st` variant, with exact PTX/LLIR minimal `x32/x16/x8/x4` opcode checks. Current runtime-matrix collection is `3401` tests: `cp=322`, `mma=324`, splitn/misc `=252`, `ld_red=771`, and `ldst=1732`; bucketed evidence now aggregates to `2955 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH collect for the focused selector selected `30/3401`; focused row/column N=32 selector passed all `30` cases across four GPUs (`8`, `8`, `8`, `6`), slowest split about `2:39`. Remaining long-term coverage work: continue bounded descriptor-view breadth and only run a broad `ldst` bucket after accumulated slices or shared-lowering source changes justify it.

- Prior two-CTA `ld/st` N=32 checkpoint, 2026-04-14 03:34 UTC: `test_tmem_runtime_matrix_ldst_twocta_n32_linear_layout` pins root `256x32` f32 roundtrips for both supported two-CTA TMEM-linear layout families (`block_two_ctas` and `mmav5_twocta`), both direct TMEM access and supported descriptor-chain views, and every public `ld/st` variant. The slice validates exact PTX/LLIR minimal `x32/x16/x8/x4` opcodes plus the `twoCTAs` TTGIR marker. Current runtime-matrix collection is `3371` tests: `cp=322`, `mma=324`, splitn/misc `=252`, `ld_red=771`, and `ldst=1702`; bucketed evidence now aggregates to `2925 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH collect for the focused selector selected `20/3371`; focused two-CTA N=32 selector passed all `20` cases across four GPUs (`5` each), slowest split about `1:35`. Remaining long-term coverage work: continue bounded `ld/st` row/column cross-product and view-chain slices before a broad `ldst` runner refresh.

- Prior `ld/st` N=32 permuted-layout checkpoint, 2026-04-14 03:20 UTC: `test_tmem_runtime_matrix_ldst_permuted_n32_linear_layout` extends the new root `128x32` f32 coverage from identity to diagonal row/column permutations (`rotate1`, `even_odd`, `reverse`) for both direct TMEM access and supported descriptor-chain views, across every public `ld/st` variant. The slice pins the same minimal `x32/x16/x8/x4` atom families and keeps exact PTX/LLIR equality. Current runtime-matrix collection is `3351` tests: `cp=322`, `mma=324`, splitn/misc `=252`, `ld_red=771`, and `ldst=1682`; bucketed evidence now aggregates to `2905 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH collect for the focused selector selected `30/3351`; focused permuted N=32 selector passed all `30` cases across four GPUs (`8`, `8`, `8`, `6`), with the slowest cold descriptor split taking about `2:58`. Remaining long-term coverage work: continue bounded `ld/st` layout/view breadth, especially two-CTA/minimal-N and row/column cross-product slices, before any broad `ldst` runner refresh.

- Prior `ld/st` N=32 identity checkpoint, 2026-04-14 03:03 UTC: `test_tmem_runtime_matrix_ldst_identity_n32_linear_layout` originally pinned the missing canonical identity `128x32` f32 roundtrip surface for both direct TMEM access and the supported descriptor-chain path, across `auto`, `32x32b`, `16x64b`, `16x128b`, and `16x256b`. The expected opcodes are the minimal `x32/x16/x8/x4` atom families already used by subview expectations, now executed as root identity `ld/st` cases. Current runtime-matrix collection is `3321` tests: `cp=322`, `mma=324`, splitn/misc `=252`, `ld_red=771`, and `ldst=1652`; bucketed evidence now aggregates to `2875 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH collect for the focused selector selected `10/3321`; focused N=32 selector passed all `10` cases across four GPUs. Remaining long-term coverage work: broaden `ld/st` layout/view fuzzing beyond this bounded identity slice, then continue staged `ld.red`, copy, and MMAv5/scaled-MMAv5 saturation.

- Prior scaled-MMAv5 TMEM-LHS tile-permuted checkpoint, 2026-04-14 02:42 UTC: `test_tmem_runtime_matrix_mma_scaled_lhs_tile_permuted_format_matrix` now covers `N=128` and `N=256` at `K=256` for every reachable full-shape tile-permuted LHS format pair and both legacy/canonical accumulator layouts. A provisional wider `K=128` probe found the fp4-storage tile-permuted LHS layouts are cleanly rejected as not MMAv5-compatible, so this slice keeps the positive matrix on the legal `K=256` surface rather than mixing partial format support. Current runtime-matrix collection is `3311` tests: `cp=322`, `mma=324`, splitn/misc `=252`, `ld_red=771`, and `ldst=1642`; bucketed evidence now aggregates to `2865 passed, 446 skipped`. Validation: py-compile passed; `git diff --check` passed; `make -j8` no-op success; no-PYTHONPATH collect for the focused selector selected `16/3311`; focused LHS tile-permuted selector passed all `16` cases across four GPUs; tight `mma` runner bucket passed `324` cases across four groups (`81` each). Remaining long-term coverage work: keep mining bounded MMAv5/scaled-MMAv5 parity gaps, then continue `ld/st` fuzz/layout breadth and the descriptor/address/staging research frontiers for copy `warpx2`.

- Prior scaled-MMAv5 accumulator tile-permuted checkpoint, 2026-04-14 02:24 UTC: `test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_64_format_matrix` now covers `K=128` and `K=256` for every current scaled format pair while keeping the `128x256` accumulator tile-permuted layout and exact PTX/LLIR opcode checks. The expected scaled-MMAv5 opcode count is now `4 * (K // 128) * base_count`, so the `K=256` cases pin the doubled K-depth instruction surface. Current runtime-matrix collection is `3303` tests: `cp=322`, `mma=316`, splitn/misc `=252`, `ld_red=771`, and `ldst=1642`; bucketed evidence now aggregates to `2857 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; no-PYTHONPATH collect for the focused selector selected `10/3303`; focused tile-permuted scaled-accumulator selector passed all `10` cases across four GPUs; tight `mma` runner bucket passed `316` cases across four groups (`79` each); `git diff --check` passed. Remaining long-term coverage work: continue broader MMAv5/scaled-MMAv5 saturation, `ld/st` fuzz/layout breadth, and the descriptor/address/staging research frontiers for true scales `warpx2` and no-scales two-CTA `warpx2::02_13`.

- Prior copy `warpx2` frontier checkpoint, 2026-04-14: supported no-scales `warpx2` copy positives cover f32+i32 on single-CTA `01_23`, single-CTA `02_13`, and two-CTA `01_23`; the no-scales two-CTA `02_13` clean-unsupported boundary now also covers f32+i32. A widened direct-PTX destination-delta scan found zero valid two-CTA `02_13` direct-offset schedules across `1278` records (`292` executing duplicate-column variants, `986` failures), recorded compactly in `experiments/results/probe_cp_warpx2_02_13_twocta_dst_deltas_current_summary.json`. Current runtime-matrix collection is `3298` tests: `cp=322`, `mma=311`, splitn/misc `=252`, `ld_red=771`, and `ldst=1642`; bucketed evidence now aggregates to `2852 passed, 446 skipped`. Validation: py-compile passed; `make -j8` rebuilt cleanly; no-PYTHONPATH collect passed; exact two-CTA `02_13` boundary split passed both selected dtype cases; nearby no-scales `warpx2` selector passed `12`; full `cp` runner bucket passed `317`, skipped `5`; `git diff --check` passed. True scales `warpx2` and no-scales two-CTA `02_13` remain descriptor/address/staging research frontiers, not direct-offset fixes.

- Prior scaled-MMAv5 coverage checkpoint, 2026-04-14 01:24 UTC: single-CTA scaled-MMAv5 accumulator-subslice format coverage now includes `N=128` subviews at starts `0` and `128`, in addition to the existing `N=64` start-0/start-64 cases, across all five scaled format pairs. The format helper now sizes the parent accumulator descriptor as `2 * N`, preserving the existing `N=64` behavior while enabling the larger subview. Current runtime-matrix collection is `3294` tests: `cp=318`, `mma=311`, splitn/misc `=252`, `ld_red=771`, and `ldst=1642`; bucketed evidence now aggregates to `2848 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; focused scaled accumulator-subslice format slice passed `20` cases across four GPUs; tight `mma` runner bucket passed `311` cases across four GPUs; `git diff --check` passed.

- Prior `ld.red` coverage checkpoint, 2026-04-14 01:18 UTC: non-identity compatible reduction-source layouts now include the missing `N=32` shape for tile-permuted, pure column-permuted, pure row-permuted, and row+column-permuted families. This adds `128` positive runtime cases that all pin the `tcgen05.ld.red.sync.aligned.32x32b.x32` opcode family across min/max and all `abs`/NaN modifier combinations. Current runtime-matrix collection is `3284` tests: `cp=318`, `mma=301`, splitn/misc `=252`, `ld_red=771`, and `ldst=1642`; bucketed evidence now aggregates to `2838 passed, 446 skipped`. Validation: py-compile passed; `make -j8` no-op success; focused new `N=32` slice passed `128` cases across four GPUs; full `ld_red` runner bucket passed `771` cases across 16 split groups; `git diff --check` passed.

- Prior speed/coverage checkpoint, 2026-04-14 01:03 UTC: the runtime-matrix runner now supports exact selected-shard reruns with `--groups` while preserving canonical group-to-GPU cache mapping, plus experimental per-bucket `--xdist-override`. Selected warm `ldst` groups 5 and 8 pass in about `11s` wall each, a 10x+ exact-rerun path versus the prior full-cold `ldst` group range of `208.7s..263.8s`, without reducing selected nodeids. At that point full collection was `3156` tests after the first post-speed ISA coverage slice: `tcgen05.cp.128x128b` covers f32 and i32 across legacy/linear single-CTA, linear indexed-view, and two-CTA paths. Validation: py-compile and `git diff --check` passed; `make -j8` no-op success; focused CP slice passed `17` cases across four GPUs; full `cp` runner bucket passed `313`, skipped `5`.

- Latest GB200 status checkpoint, 2026-04-14 00:46 UTC: current branch `codex/tmem`
  is clean and synced at `cfef1b94f`. Fresh current-HEAD checks after the
  runtime-matrix runner/duration-only commits: `make -j8` success,
  `make test-cpp` `240/240` passed, `make test-lit` `248 passed, 2 unsupported`,
  four-GPU split `python/test/gsan python/test/regression` aggregate
  `1110 passed, 216 skipped`, and `test-microbenchmark` passed with median
  launch overhead `22.5695`. A broad `python/test/unit` four-GPU refresh was
  stopped after three shards were green and the fourth was still making progress
  because no compiler/runtime source changed since the full GB200-green
  inventory; use the existing inventory plus targeted reruns for unchanged
  surfaces instead of spending dev time on blanket retests. Current conclusion:
  no known branch-caused GB200 CI failures; Proton still has the known
  merge-base-reproducing cudagraph / periodic-flushing failures and remains
  non-actionable for this TMEM branch.

- Prior runtime-matrix speed checkpoint, 2026-04-13 23:01 UTC: the five lifted `ld/st` descriptor roundtrip matrices that previously compiled to `OutOfResources` and then skipped are now marked as known pre-execution skips. This preserves instruction/op coverage because those `440` cases never emitted or executed `tcgen05` op coverage; they only consumed compile time before `pytest.skip`. Validation: py-compile passed; `make -j8` no-op success; `env -u PYTHONPATH pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k ldst` still collects `1642/3150`; the exact five skip-only functions now report `440 skipped in 2.40s`; full runner `ldst` passed with the same `1201 passed, 441 skipped`, while shard times improved from the prior `271.68s..350.27s` range to `194.71s..273.96s`.

- Superseded runtime-matrix validation-velocity checkpoint, 2026-04-13 22:15 UTC: the full `python/test/gluon/test_tmem_runtime_matrix.py` local sweep now has a coverage-preserving runner, `.codex/initiatives/tmem_linear_generalization/run_tmem_runtime_matrix_sweep.py`, plus `tmem_runtime_matrix_validation_recipe_20260413.md`. Full collection remains `3150` tests. The runner partitions the matrix into `cp` (`312`), tight `mma` (`301`), exact-nodeid splitn/misc (`252`), `ld_red` (`643`), and `ldst` (`1642`) buckets, summing to the full matrix. The timeout diagnosis is cold compilation plus bad static partitioning, not a deadlock: representative `ldst` cold/warm timing was about `31s`/`3s`, and representative `ld_red` cold/warm timing was about `10s`/`3s`. Validation: `make -j8` no-op success; runner py-compile passed; runner `--dry-run` emitted deterministic commands; exact-nodeid `splitn` runner smoke passed `252` tests; full runner `ld_red` passed `643` tests with shard times from about `14s` to `96s`; full runner `ldst` passed `1201` and skipped `441` with shard times from about `4:32` to `5:50`. Current per-bucket evidence aggregates to full-matrix coverage: `2704 passed, 446 skipped` across all `3150` collected cases, without reducing the matrix.

- Latest TMEM runtime-matrix test/cache hygiene checkpoint, 2026-04-13 20:43 UTC: `test_tmem_runtime_matrix.py` no longer imports reusable helpers from `python.test.gluon.test_core`; shared TMEM helper kernels and descriptor builders now live in `python/test/gluon/tmem_test_utils.py` and are imported as a sibling helper. The two `tcgen05.cp` two-CTA codegen tests that previously launched `python -c`, injected `PYTHONPATH`, and forced a fresh temporary `TRITON_CACHE_DIR` now run in-process and reuse the shard's stable per-GPU cache. Current cache diagnosis: the active misbehavior was validation self-sabotage from cold temporary caches and child-process imports, not a proven on-disk cache-key collision. Validation: no-PYTHONPATH collect of the full runtime-matrix file reports `3150` tests; `python3 -m py_compile` for `tmem_test_utils.py`, `test_tmem_runtime_matrix.py`, and `test_core.py` passed; `make -j8` no-op success; exact affected/import-helper nodeids passed across four split groups with stable `/tmp/triton-cache-gpu<N>` caches (`1` selected case per group in the final four-nodeid split); `git diff --check` passed.

- Latest plain-MMAv5 TMEM-LHS N-shape checkpoint, 2026-04-13 after the scaled LHS-subview expansion: `test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds` now spans `N in {128, 256}` for every supported plain kind and both legacy/canonical accumulator layouts, while `test_tmem_runtime_matrix_mma_lhs_tile_permuted` spans `N in {128, 256}` for all non-OOR direct shared-B shapes. The only omitted full-shape tile entry is `tf32, N=256, K=256`, which exceeds shared memory for this helper (`Required: 262156`, limit `232448`). Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `git diff --check`; `make -j8` no-op success; exact two-nodeid matrix passed `29` selected cases across four GPU groups (`8`, `8`, `8`, `5`); tight `-k 'test_tmem_runtime_matrix_mma'` passed `301` selected cases across four GPU groups (`76`, `76`, `76`, `73`) in `59s`, `139s`, `123s`, and `131s`.

- Latest scaled-MMAv5 TMEM-LHS subview N/K checkpoint, 2026-04-13 after the scaled-copy multicast expansion: `test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view_format_matrix` now spans the packed-storage reachable format pairs (`mxfp8/mxfp8`, `mxfp8/mxfp4`, `mxfp4/mxfp4`, and `nvfp4/nvfp4`) at `N in {128, 256}` and `K in {128, 256}`, over both legacy and canonical TMEM-linear accumulator layouts. A scratch four-GPU probe confirmed the twenty-four new `N/K` combinations match the dequantized reference; the scaled-MMAv5 opcode count is now pinned as `(K // 128) * base_count`, doubling for `K=256`. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `git diff --check`; `make -j8` no-op success; exact expanded nodeid passed all `32` cases across four GPU split groups (`8` each); nearby `-k 'mma_scaled and lhs and subslice'` passed `35` selected cases (`9`, `9`, `9`, `8`); tight `-k 'test_tmem_runtime_matrix_mma'` passed `287` selected cases across four GPU groups (`72`, `72`, `72`, `71`) in `56s`, `120s`, `118s`, and `132s`.

- Latest scaled-MMAv5 copy-helper multicast checkpoint, 2026-04-13 after the `block_k=256` expansion: `test_tmem_runtime_matrix_cp_scales_warpx4_via_scaled_mma_copy_matrix` now spans every current scaled format pair, `block_n in {128, 256}`, `block_k in {128, 256}`, `num_ctas in {1, 2}`, `multicast in {False, True}`, and both legacy/canonical accumulator layouts. A scratch four-GPU probe confirmed the eighty new `multicast=True` combinations match the dequantized reference; one-CTA cases do not emit TTGIR multicast while two-CTA multicast cases do. Exact scale-copy count remains `(1 + block_n // 128) * (block_k // 128) * (32 // vec_size)`, and scaled-MMA count remains `(block_k // 128) * base_count`. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `git diff --check`; `make -j8` no-op success; exact expanded nodeid passed all `160` cases across four GPU split groups (`40` each); nearby `-k 'cp_scales_warpx4_via_scaled_mma'` passed `192` cases (`48` each); broad `-k 'cp'` passed `307` with `5` skips (`73 passed, 5 skipped`; `78`; `78`; `78`).

- Latest direct scaled-MMAv5 root-format checkpoint, 2026-04-13 after the `K=256` expansion: one-CTA root accumulator coverage now spans every current scaled format pair (`mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`, `mxfp4/mxfp8`, and `nvfp4/nvfp4`) at `N in {128, 256}` and `K in {128, 256}`, over both legacy and canonical TMEM-linear accumulator layouts. A scratch four-GPU probe confirmed the twenty new `K=256` combinations match the dequantized reference; the scaled-MMA opcode count is `base_count * (K // 128)`, so `K=256` doubles the existing direct-root counts (`8` ops for fp8/mixed vector-1X cases and `4` ops for homogeneous fp4/nvfp4 cases). The runtime matrix now pins this through `SCALED_MMA_ROOT_FORMAT_CASES`. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `git diff --check`; `make -j8` no-op success; exact expanded nodeid passed all `40` parameters across four GPU split groups (`10` each); tight `-k 'test_tmem_runtime_matrix_mma'` passed `263` selected cases across four GPU groups (`66`, `66`, `66`, `65`) in `54s`, `92s`, `131s`, and `86s`.

- Latest plain-MMAv5 root-accumulator shape checkpoint, 2026-04-13 15:02 UTC: one-CTA root accumulator coverage now spans `N=128` and `N=256` for every supported plain operand kind (`f16`, `tf32`, `bf16`, `f8e5m2`, and `f8e4m3`), both legacy and canonical TMEM-linear accumulator layouts, and both no-accumulator and `use_acc=True` paths. A scratch probe confirmed the wider `128x256` root layouts use the same root opcode counts as `128x128`; the runtime matrix pins this root-only surface through `MMA_PLAIN_KIND_ACC_CASES`. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `make -j8` no-op success; focused `-k 'mma_plain_kinds_with_linear_acc or mma_plain_kinds_use_acc'` passed `40` selected cases across four GPU groups; tight `-k 'test_tmem_runtime_matrix_mma'` passed `223` selected cases across four GPU groups (`56`, `56`, `56`, `55`); `git diff --check` passed.

- Latest scaled-MMAv5 TMEM-LHS clean-negative checkpoint, 2026-04-13 14:57 UTC: full-shape tile-permuted TMEM-LHS now has the same mixed fp4-A contract coverage as the subview path. New `test_tmem_runtime_matrix_mma_scaled_lhs_tile_permuted_mixed_fp4a_reports_clean_unsupported` covers `mxfp4/mxfp8` over legacy and canonical TMEM-linear accumulator layouts at logical `K=256`, confirms the verifier reports the padded-storage fp4-A diagnostic, and checks that the failure is clean rather than a PassManager/assertion crash. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `make -j8` no-op success; exact new nodeid passed its two selected parameters across four GPU split groups; nearby `-k 'mma_scaled and lhs and tile_permuted'` passed `10` selected cases across four groups; tight `-k 'test_tmem_runtime_matrix_mma'` passed `203` selected cases across four GPU split groups (`51`, `51`, `51`, `50`).

- Latest copy-planner cleanup checkpoint, 2026-04-13 14:55 UTC: `getTMemCopyPlans` no longer offers the direct-seed `warpx2::02_13` plan for multi-CTA/block layouts. The direct seed remains available for the executable single-CTA `128x4` `02_13` path, but the two-CTA `256x4` case stays on descriptor-plan search only because repeated actual-layout probes show the direct-seed extension either duplicates the source-column pair, writes all zeros, or traps for unaligned TMEM deltas. The stale comment implying a representable `01_23`-style descriptor shape might be enough was replaced with the current invariant: descriptor representability alone is not proof; a future `02_13` fix must preserve the missing 4-byte source-column bit through a real descriptor/address schedule. Validation: `make -j8` rebuilt cleanly; the affected exact selector (`warpx2_02_13_candidate_positive or warpx2_02_13_twocta_candidate_reports_clean_unsupported or warpx2_01_23_twocta_positive`) passed on the three non-empty four-GPU split groups; broader `-k 'cp_no_scales_warpx2'` passed `8` cases across four GPU split groups; `lit -v test/TritonNvidiaGPU/invalid.mlir` passed; `git diff --check` passed.

- Latest scaled-MMAv5 two-CTA accumulator-subview `block_n=256` probe, 2026-04-13 after the tight MMA validation: do not expand `test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_matrix` to the obvious `parent_n=512` / `slice_start in {0,256}` shape. Representative `mxfp8/mxfp8` probes for both slice starts and multicast modes fail launch metadata with tensor-memory OOR (`Required: 524`, hardware limit `512`). A smaller `parent_n=384` would fit the desired offset range but is invalid for `_make_tmem_linear_layout_mmav5_twocta`, which currently requires power-of-two `N`. Keep current `block_n=128`, `parent_n=256`, `slice_start in {0,128}` coverage as the live offset-subview matrix unless a different legal parent layout is designed.

- Latest broad direct MMA/scaled-MMA runtime-matrix validation checkpoint, 2026-04-13 after the plain LHS `N` expansion: use `-k 'test_tmem_runtime_matrix_mma'` for the pure MMA/scaled-MMA family, not the historical `-k 'mma and not cp'` selector. Collect-only showed `-k 'mma and not cp'` also selects `ldst_*mmav5*` cases because `mmav5` contains `mma`, which explains earlier static-split imbalance. The tight selector now collects `301` cases and passed across four GPU split groups: `76`, `76`, `76`, and `73` passed.

- Superseded broad `tcgen05.cp` runtime-matrix validation checkpoint, 2026-04-13 after the scaled-copy multicast expansion: current head is green for `python/test/gluon/test_tmem_runtime_matrix.py -k 'cp'` across four GPU `pytest-split` groups. Aggregate result: `307 passed, 5 skipped` (`73 passed, 5 skipped`; `78 passed`; `78 passed`; `78 passed`). This supersedes older `227 passed, 5 skipped` broad-copy counts; the delta is expected from adding the all-format multicast scaled-copy matrix.

- Superseded scaled-MMAv5 copy-helper format/shape checkpoint, 2026-04-13 after the TMA TF32 negative expansion: the first copy-helper expansion covered `block_n=128` and `256` for every existing scaled format pair, both `num_ctas=1/2`, and both legacy/canonical accumulator layouts at `block_k=128`. The current latest checkpoint above extends this matrix to `block_k=256` and supersedes the old expected count formula.

- Latest TMA-fed two-CTA TF32 clean-negative checkpoint, 2026-04-13 after the direct-i8 clean-negative expansion: `test_tmem_runtime_matrix_mma_twocta_tma_tf32_reports_clean_shared_transpose_error` now covers `block_n=128` and `block_n=256` for both legacy and canonical TMEM-linear accumulator layouts. This matches the positive `[N, K]` descriptor plus shared-permute route and keeps the default `[K, N]` B TMA descriptor shape pinned as a clean transposed-float32 shared-operand diagnostic. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `make` no-op success; exact expanded negative nodeid across four GPU split groups (`4 passed` aggregate); nearby `-k 'tma_tf32'` selector passed all eight selected positive plus negative cases across the four groups.

- Latest direct-i8 MMAv5 clean-negative checkpoint, 2026-04-13 after the TMA TF32 descriptor expansion: `test_tmem_runtime_matrix_mma_twocta_i8_reports_clean_error` now covers `block_n=128` and `block_n=256` for both legacy and canonical TMEM-linear two-CTA accumulator layouts, matching the supported two-CTA plain-kind `blockN` surface. The test still asserts the dedicated `direct tcgen05_mma kind::i8 is not supported on sm_... current Blackwell lowering` diagnostic and absence of PassManager/assertion noise. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `make` no-op success; exact expanded two-CTA nodeid across four GPU split groups (`4 passed` aggregate); nearby `-k 'i8_reports_clean_error'` selector passed all six selected one-CTA plus two-CTA cases across groups 1-3, with group 4 selecting no tests.

- Latest TMA-fed two-CTA TF32 B-transposed descriptor checkpoint, 2026-04-13 after the scales `warpx2` scan: `test_tmem_runtime_matrix_mma_twocta_tma_tf32_b_transposed_descriptor` now covers both `block_n=128` and `block_n=256` for legacy and canonical TMEM-linear accumulator layouts. Scratch probes first confirmed that the `block_n=256` cases pass for both layouts and emit four matching PTX/LLIR `tcgen05.mma.cta_group::2.kind::tf32` ops. Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`; `make` no-op success; exact expanded nodeid across four GPU `pytest-split` groups (`4 passed` aggregate); nearby `-k 'tma_tf32'` selector passed all six selected cases across groups 1-3 while group 4 selected no cases because the split had no remaining tests. This fills the immediate TMA-fed TF32 `blockN=256` positive gap; the default `[K, N]` TMA descriptor path remains the clean transposed-float32 shared-operand negative.

- Latest two-CTA `warpx2::02_13` corrected direct-PTX checkpoint, 2026-04-13 13:20 UTC: the durable probe `experiments/probe_cp_warpx2_02_13_twocta_direct_ptx.py` now uses compile-only warmup by default, with `--prime-canonical` retained only to reproduce historical runs that launched the canonical `warpx2::01_23` kernel first. Regenerated artifacts show zero matches against the extended two-CTA `02_13` oracle. All `22` named opcode/descriptor/two-message variants execute with finite output but duplicate a source-column pair (`duplicates_col_pair=true`). The four-GPU source-offset JSONL sweep covers `sourceOffsetB128` `36..127` with destination deltas `0/4`: `74` records for offsets `36..72` execute but all duplicate a source-column pair, while `110` records for offsets `73..127` launch-fail in isolated child processes. The older 2026-04-13 09:42 NaN/partial-duplicate breakdown is superseded because that probe launched the canonical kernel before patched cubins. Conclusion: keep public two-CTA `warpx2::02_13` clean unsupported; next useful work is descriptor/address-message synthesis from the layout model, not another direct-seed offset toggle.

- Latest two-CTA `warpx2::02_13` descriptor experiment, 2026-04-13 14:09 UTC: a
  temporary verifier probe compared the known-good `01_23` descriptor plan
  against `02_13`, then injected the `01_23`-style representable `32x4`
  descriptor candidate into the `02_13` fallback and temporarily bypassed the
  lowering row-stride assertion. This proved that descriptor representability
  alone is not sufficient: lowering emitted
  `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b`, but runtime output differed
  in all `1024` elements and duplicated source-column pairs (`[0,128,0,128]`
  style rows instead of the extended `02_13` oracle). All temporary source
  changes were removed and clean source was rebuilt; the restored two-copy
  controls passed (`01_23` positive and `02_13` clean unsupported selected on
  groups 1-2; groups 3-4 had no selected cases). Conclusion: the missing piece
  is not just adding a representable descriptor candidate or skipping the
  stride guard. Future support needs a true `02_13` descriptor/address-message
  schedule that changes the source mapping, or it should remain clean
  unsupported.

- Latest copy-planner debug checkpoint, 2026-04-13 13:43 UTC: temporary `TRITON_TMEM_COPY_DEBUG` instrumentation confirmed that two-CTA `warpx2::02_13` and the tensor-memory-scales `warpx2_candidate` are rejected in `TMEMCopyOp::verify`, before LLVM lowering. Two-CTA `02_13` classifies into three `warpx2::02_13` plans: direct-seed `64x4`, descriptor `64x4`, and fallback descriptor `32x4`; direct seed is unavailable for the `256x4` shared tile, and none of the `95/95/161` descriptor candidates is MMASMEM-representable. The known-good two-CTA `01_23` path succeeds through the `32x4` plan with a descriptor candidate equivalent to row bases `4,8,16,32,64,128` and col bases `1,2`; the analogous two-CTA `02_13` fallback has a zero low col basis and still fails descriptor representation. Single-CTA `02_13` succeeds only through the `128x4` direct seed. Combined with the corrected direct-PTX probes, this rules out blindly extending direct seed to `256x4`; the remaining work is a real descriptor/address-message model or continued clean unsupported.

- Latest scales `warpx2` direct-PTX correction checkpoint, 2026-04-13 13:18 UTC: the durable probe `experiments/probe_cp_scales_warpx2_direct_ptx.py` now uses compile-only warmup by default so patched variants are not primed by a prior canonical `warpx4` launch. The corrected results in `experiments/results/probe_cp_scales_warpx2_direct_ptx_current.jsonl` show the unpatched `warpx4_control` still roundtrips, but unprimed `first_01_23_only` and `first_02_13_only` only fill the low half of each logical row (`diff_count=510` for random, `512` for arange). Patching both original messages or using only the second message still launch-fails with illegal memory access. The old apparent positive was a primed-state artifact: `first_01_23_only --prime-canonical` can reproduce the historical success after launching the canonical copy first, but that is not valid support evidence. Conclusion: true scales `warpx2` remains unsupported/unproven; do not land an alias or opcode-suffix swap based on the earlier primed direct-PTX result.

- Latest subview-offset checkpoint, 2026-04-13 09:14 UTC: fixed a compiler assertion where
  TMEM subview base lowering used a non-surjective projected ld/st query layout
  as if it could invert arbitrary logical offsets. `getTMemViewOffsetForLowering`
  now uses raw query layouts for offset arithmetic only when they are
  surjective, otherwise it falls back to the descriptor type's full layout. The
  former `mmav5_twocta-64-auto` descriptor-roundtrip abort now reaches the
  existing OOR skip path, the exact `mmav5_twocta-64` roundtrip matrix has no
  failures, and new two-CTA direct-i8 MMA clean-negative coverage is in the
  runtime matrix. Validation details and split-timeout caveats are in
  `memory.md`, `log.md`, and the handoff tail.

- Latest `tcgen05.cp` contract checkpoint, 2026-04-13 07:10 UTC:
  runtime-matrix clean-negative coverage now also pins a representative scales
  copy from a larger shared parent subview. The new
  `test_tmem_runtime_matrix_cp_scales_shared_subslice_layout_reports_clean_unsupported`
  covers aligned `start_row` values `0` and `64` with the parent-row layout from
  the 384-layout `warpx2` subslice probe, and confirms the existing clean
  TensorMemoryScales descriptor-plan diagnostic is reported before lowering.
  Validation: `py_compile` passed, rebuild was a no-op success, the exact new
  nodeid passed both selected parametrized cases across split groups, and the
  nearby scales-copy selector passed (`9 passed` aggregate across active split
  groups). The previous no-scales transposed shared-source and invalid
  split-offset clean negatives remain covered.

- Latest `ld.red` contract checkpoint, 2026-04-13 06:20 UTC:
  runtime-matrix clean-negative coverage now pins the non-f32
  reduction/modifier boundary. New coverage rejects `i32` plain reductions with
  the f32-required diagnostic, rejects `i32` reductions with `NaN` / `abs`
  modifiers with the dedicated modifier-type diagnostics, and rejects a
  legacy-unpacked `f16` reduction attempt before lowering. Validation:
  `py_compile` passed, rebuild was a no-op success, the new nodeid passed
  across four GPU split groups (`4 passed` aggregate), the nearby negative
  selector passed (`11 passed` aggregate), and the broad four-GPU `-k ld_red`
  selector passed (`507 passed` aggregate). The broad split-4 run was badly
  imbalanced without duration data (reported shard times about `7:58`,
  `19:31`, `19:05`, and `13:57`), so future broad `ld_red` local validation
  should use finer/duration-aware groups or narrower selectors instead of
  treating a slow split-4 shard as normal.

- Latest M64 docket checkpoint, 2026-04-13 05:41 UTC: the historical
  `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]` bug/xfail item
  is stale at current head. After rebuild, the legacy parameter passed with
  `--runxfail`, both `test_block_m_64_mma` parameters passed normally across
  four `pytest-split` groups, and the nearby M64 subview/MMA selector
  (`test_tmem_subslice_block_m_64`,
  `test_tmem_subslice_block_m_64_parent_layout`, and `test_block_m_64_mma`)
  passed all six selected cases. Do not carry the old M64 xfail as a live
  branch-actionable failure unless a fresh exact repro fails.

- Latest plain-MMAv5 TMEM-LHS checkpoint, 2026-04-13 13:48 UTC: full-shape
  tile-permuted TMEM-LHS coverage now spans every supported plain operand kind
  (`f16`, `tf32`, `bf16`, `f8e5m2`, and `f8e4m3`) instead of only f16.
  `tmem_mma_lhs_kernel` now derives the operand dtype from the A pointer and
  accepts the kind-specific B shared-memory layout, so the widened test feeds
  a `128x256` tile-permuted TMEM-linear operand-A tile directly to
  `tcgen05_mma` and checks numeric output plus exact PTX/LLIR opcode counts
  (`f16/bf16=16`, `tf32=32`, `f8=8`). Validation: py-compile passed,
  rebuild was a no-op success, the exact widened nodeid passed all `5` cases
  across four GPU split groups (group 4 selected no cases because the nodeid
  has only five params), the nearby LHS selector passed `15` cases, and the
  nearby accumulator/LHS tile-permuted selector passed `35` cases across four
  GPUs. A broad `-k 'mma and not cp'` split-4 refresh had groups 3 and 4 green
  (`78` passed each in about `1:40`/`1:46`) but groups 1 and 2 hit the 300s
  guard while still emitting progress; treat that as the known no-duration
  partitioning issue, not as a product failure.

- Latest plain-MMAv5 saturation checkpoint, 2026-04-13 07:45 UTC:
  tile-permuted TMEM-linear accumulator coverage now also pins the `use_acc=True`
  accumulator-add path for all supported plain operand kinds (`f16`, `tf32`,
  `bf16`, `f8e5m2`, and `f8e4m3`) across the existing `128x128/tile_n=32` and
  `128x256/tile_n=64` layouts. The new runtime-matrix test checks numeric
  `matmul + accumulator` results, exact PTX/LLIR `tcgen05.mma` opcode streams,
  commit opcodes, and preservation of `tensor_memory_linear`. Validation:
  `py_compile` passed, rebuild was a no-op success, the exact new nodeid passed
  all `10` selected cases across four GPU split groups, the nearby
  `mma_plain_kinds_tile_permuted_acc or mma_plain_kinds_use_acc` selector passed
  `30` selected cases aggregate, and the post-commit broad four-GPU
  `-k 'mma and not cp'` selector passed `242 passed, 50 skipped`.

- Latest `ld/st` x1 subword checkpoint, 2026-04-13 08:35 UTC:
  descriptor-chain two-CTA x1 subword roundtrips are now pinned in addition to
  the direct two-CTA case. The new runtime-matrix test allocates a lifted
  `[2, M, N]` canonical two-CTA TMEM-linear layout, reaches the active tile
  through `slice/index/reshape/permute/permute`, and covers `f16`, `bf16`,
  `i16`, and packed `i8` for `auto` and explicit `32x32b`. It checks exact
  `tcgen05.{st,ld}.sync.aligned.32x32b.x1.b32` PTX/LLIR opcode streams,
  zero-offset addressing, numeric roundtrip equality, `memdesc_subslice`,
  `memdesc_reshape`, `memdesc_trans`, `tensor_memory_linear`, and
  `twoCTAs = true` TTGIR spelling. Validation: `py_compile` passed, rebuild
  was a no-op success, the exact new descriptor-chain nodeid passed `8`
  selected cases across four GPU split groups, and the nearby
  `ldst_x1_subword or ldst_x1_f32` selector passed `65` selected cases
  aggregate.

- Latest scaled-MMAv5 TMEM-LHS checkpoint, 2026-04-13 05:35 UTC: scaled
  MMAv5 TMEM-LHS subviews now address packed fp4 operand-A descriptors in
  storage-column coordinates for the K tile step, fixing homogeneous A-side
  fp4 positives for `mxfp4/mxfp4` and `nvfp4/nvfp4`. Positive coverage now
  includes `mxfp8/mxfp8`, `mxfp8/mxfp4`, `mxfp4/mxfp4`, and `nvfp4/nvfp4`
  across legacy and canonical TMEM-linear accumulators. The mixed
  `mxfp4/mxfp8` dense TMEM-LHS path no longer wrong-codes: it is a clean
  verifier negative because `mxf8f6f4` fp4 LHS requires the padded operand-A
  storage model currently represented by `fp4_padded` shared memory.
  Validation: rebuild passed, durable probe refreshed, focused positive plus
  negative matrix `10 passed`, nearby scaled-MMA selector `27 passed`, and
  broad four-GPU `-k 'mma and not cp'` selector `242 passed, 50 skipped`.

- Superseded copy-frontier checkpoint, 2026-04-13 09:42 UTC (corrected by the 13:20 compile-only warmup rerun): two-CTA `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b` remained intentionally clean unsupported and the direct-PTX search found zero oracle matches, but the exact NaN/duplicate breakdown is stale because the probe launched the canonical `01_23` kernel before patched cubins. Use the 13:20 corrected artifact summary above for current source-offset counts.

- Latest code checkpoint, 2026-04-13 09:24 UTC: direct two-CTA tensor-memory-scales `tcgen05.copy` now has pure-copy runtime-matrix coverage. The new `tmem_copy_scales_warpx4_twocta_kernel` roundtrips a `128x16xi8` CGA-shaped `TensorMemoryScalesLayout(cga_layout=[[1, 0]])` tile from a shared-linear `warpx4` layout with `block_bases=[[64, 0]]`, and asserts exact PTX/LLIR `tcgen05.cp.cta_group::1.warpx4.32x128b` opcodes plus no legacy `ttg.memdesc_reinterpret`. This is bounded supported-copy coverage only; true scales `warpx2` and two-CTA `warpx2::02_13` remain descriptor/address-model frontiers. Validation: py-compile passed, rebuild was a no-op success, focused `-k cp_scales_warpx4` passed `54` tests across four split groups, broad `-k cp` passed `167` with `5` skips across four split groups, and `git diff --check` passed.

- Latest `ld.red` code checkpoint, 2026-04-13 13:35 UTC: explicit compatible reduction-load layouts now cover the full legal modifier matrix (`abs` false/true and `PropagateNan.NONE/ALL`) for both `min` and `max`. The explicit identity and non-identity tests pass `abs`/`propagate_nan` through `tmem.load_{min,max}`, inject NaNs for the NaN-propagating cases, check PyTorch reduction semantics, and still pin `tcgen05.ld.red.sync.aligned.32x32b.x128` opcode canonicalization for `auto`, `32x32b`, `16x32bx2`, and `32x32b_splitn`. Validation: `py_compile` passed, rebuild was a no-op success, and the focused explicit `ld.red` selector (`ld_red_explicit_compatible or ld_red_explicit_n_sharded`) passed `163` selected cases across four GPU split groups (`41`, `41`, `41`, `40` passed) in `1:50` to `4:42` per group.

- Latest scaled-MMAv5 two-CTA multicast checkpoint, 2026-04-13 12:37 UTC: the existing two-CTA accumulator-subview format matrix now covers both scale-TMA paths by running `multicast=False` and `multicast=True`. The test passes the parameter through `mma_scaled_tcgen05_acc_subslice_copy`, preserves numeric output plus exact copy/MMA/commit opcode checks, and asserts TTGIR `{multicast}` appears only for the multicast path. Validation: py-compile passed, rebuild passed, the exact nodeid passed `20` cases across four GPU split groups, and the nearby `mma_scaled and subslice and format_matrix` selector passed `43` cases across four GPU split groups.

- Latest scaled-MMAv5 full-shape TMEM-LHS checkpoint, 2026-04-13 13:59 UTC: direct
  tile-permuted TMEM-LHS coverage now includes the packed-storage positive
  subset for scaled MMA. New `tmem_mma_scaled_lhs_tile_permuted_format_kernel`
  allocates operand A directly in a tile-permuted TMEM-linear descriptor, uses
  logical `K=256` so fp4-A storage has a reachable `128`-column tile, and feeds
  the descriptor directly to `tcgen05_mma_scaled` without a subview. The new
  matrix covers `mxfp8/mxfp8`, `mxfp8/mxfp4`, `mxfp4/mxfp4`, and
  `nvfp4/nvfp4` across legacy and canonical accumulator layouts, checking
  numeric output, exact PTX/LLIR scaled-MMA opcode/count agreement, commit
  opcodes, no `ttg.memdesc_subslice`, and preserved `tensor_memory_linear`.
  Validation: scratch probe showed `K=128` is positive only for mxfp8-A while
  `K=256` covers the packed-storage subset; py-compile passed, rebuild was a
  no-op success, exact nodeid passed `8` cases across four GPU split groups,
  and nearby `mma_scaled and (lhs or tile_permuted)` passed `30` cases across
  four GPUs.

- Latest scaled-MMAv5 tile-permuted accumulator checkpoint, 2026-04-13 12:43 UTC: the `tile_n=64` direct accumulator layout now has full scaled format coverage instead of the previous `mxfp8/mxfp8`-only test. `test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_64_format_matrix` covers `mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`, `mxfp4/mxfp8`, and `nvfp4/nvfp4` on the `128x256` tile-permuted accumulator layout, with numeric checks plus exact PTX/LLIR scaled-MMA opcode/count checks. Validation: py-compile passed, rebuild was a no-op success, the exact nodeid passed all `5` selected cases across four GPU split groups, and nearby `mma_scaled and tile_permuted` passed `11` cases across four GPU split groups.

- Latest direct MMA/scaled-MMA validation checkpoint, 2026-04-13 12:45 UTC: after the two scaled-MMAv5 coverage commits (`a18607fd9` and `8b3fec8d6`), the broad runtime-matrix selector `python/test/gluon/test_tmem_runtime_matrix.py -k 'mma and not cp'` is green across four GPU split groups: `258 passed, 50 skipped` aggregate (`32/45 skipped`, `72/5 skipped`, `77`, and `77` by group).

- Latest validation checkpoint, 2026-04-13 04:19 UTC: the current GB200 branch-actionable queue is closed from local evidence at `4fe39e5d28edf82d90bb9039049083a66359f9b6`. Build is a no-op success, full lit is green (`248 passed, 2 unsupported`), C++ is green (`240/240`), microbenchmark passes, the fused-attention tutorial is clean across four GPU split groups (`192 passed, 192 skipped`), the singleton instrumentation/plugin unit tails all pass, and the full `python/triton_kernels/tests` aggregate is recomputed as `2377 passed, 3444 skipped` with no failures/errors. Together with the earlier green unit, gsan, regression, examples, and closed Gluon runtime-matrix/lowerings evidence, there are no deterministic known branch-actionable GB200 failures. Proton's `11` cudagraph / periodic flushing failures remain classified as merge-base-preexisting environmental noise.

- Latest validation checkpoint, 2026-04-13 03:00 UTC: the previously incomplete Gluon tail is now closed from local evidence. `python/test/gluon/test_tmem_runtime_matrix.py` was covered by a mixed split plan totaling `2683` selected cases (`2237 passed, 446 skipped`, no failures/errors), and `python/test/gluon/test_lowerings.py` is green across four GPU shards (`4937 passed, 512 skipped`). Combined with the prior green `python/test/gluon/ python/tutorials/gluon/` groups 1-3, split-16 groups 13-14, the isolated xdist-crash nodeid pass, and green `python/examples/gluon/`, there are no deterministic `test-gluon` failures known after the MMAv5 family-addressing fix. The old long-running shards were static split imbalance around slow TMEM ldst legality/probe cases, not a reproduced compiler/runtime failure.

- Latest wider checkpoint, 2026-04-13 01:19 UTC: full lit is green and the examples half of
  `make test-gluon` is green across four GPU shards. The first Gluon phase has
  no deterministic known failures after the MMAv5 family-addressing fix, but
  the local validation is incomplete because the heavy runtime-matrix tail still
  exceeds the current split timeout: four-way `-n 6` groups 1-3 finished green,
  the group-4 worker-crash node passed isolated, split-16 groups 13-14 finished
  green, and split-16 groups 15-16 timed out while still making progress. Next
  action is a better duration-aware/runtime-tail split, not raising timeouts or
  classifying those timeouts as product failures.

- CI timing reference from the project owner, 2026-04-13: the full GB200 CI lane
  on a comparable four-node/four-GPU machine is expected to take about 35 minutes
  including clean build and LLVM download, with actual test time around 20
  minutes. Local validation that runs much longer should be treated as a
  partitioning, xdist, or hang investigation first, not normalized as acceptable
  sweep time.

- Latest checkpoint, 2026-04-12 23:15 UTC: a focused MMAv5 loader fix is ready after the
  post-attr-cleanup `test-gluon` run exposed tile-permuted MMAv5 runtime
  failures. Full-shape MMAv5 descriptors now use the derived family layout for
  instruction address arithmetic; marked physical bitcasts still use the exact
  typed result layout. The focused tile-permuted cluster, broad runtime-matrix
  `-k 'mma and not cp'`, nearby core M64/bitcast selector, representative
  `test_mma_shared_inputs` exacts, and relevant lit lowering checks are green.
  The previous tile-permuted MMAv5 current-head failure list is stale for those
  nodeids; run the next wider GB200/checkpoint sweep before using aggregate
  counts for new prioritization.

- The TMEM side-channel attribute cleanup slice is in progress after the
  preserve-set commit `37d00bb91`:
  - producer/provenance attributes are being removed instead of copied through
    alloc rewrites (`ttng.tmem_physical_layout`, `ttng.tmem_ldst_row_plan`,
    `ttng.tmem_mmav5_accumulator_root`, and
    `ttng.tmem_mmav5_operand_root`);
  - direct ld/st planning now preserves zero bases from the descriptor layout,
    treating them as broadcast/equivalence semantics rather than disposable
    physical storage;
  - the M64 MMAv5 full-shape backing row plan is recovered from the memdesc
    type/layout family and ordinary view-chain analysis, not from producer
    attrs;
  - focused lit, frontend, matmul, tensor-descriptor, Gluon M64, copy, and
    sampled `test_mma_shared_inputs` probes pass at current working tree;
  - the post-fix four-way `test_mma_shared_inputs` split sweep is green,
    so the earlier pre-fix failure count is now stale.
- Current pre-cleanup preserve-set at `3359982ee` is recorded in
  `gb200_preserve_set_20260412.md`:
  - build passed;
  - full lit is down to two stale pipeline FileCheck failures
    (`pipeline-loop-nest.mlir` and `pipeline-lower-loop.mlir`);
  - the `162` branch-new `python/test/unit` exact nodeids still all fail;
  - `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]` still fails;
  - Proton's `11` cudagraph / periodic flushing failures still reproduce and
    remain merge-base-preexisting noise;
  - fresh current-head wrapper/tail checks for C++, gsan, regression,
    microbenchmark, unit debug, fused-attention tutorial, plugins,
    instrumentation, and Proton tails are green.
- Full GB200 `integration-tests-nvidia` inventory at `cb76c31a0` is not green:
  - branch-new actionable failures were `9` lit files, `162` `python/test/unit`
    nodeids, and `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`;
  - those exacts pass on merge-base `11ee1144a737006921231bbd3386c187812c38e1`;
  - Proton's `11` cudagraph / periodic flushing failures reproduce on
    merge-base and are ignored as preexisting GB200 lane noise;
  - as of 2026-04-12 08:21 UTC, the five API/contract lit files from that
    `9`-file lit bucket pass focused lit after supported-spelling rewrites,
    leaving the four stale lit tests as the next lit-only recovery slice before
    a full `make test-lit` rerun.
- The latest current-head `python/examples/gluon` aggregate passed all four
  split groups, so `python/examples/gluon/01-attention-forward.py` is now
  covered by the examples aggregate as well as its focused SDPA-backed matrix.

- The supported `slice/subview -> bitcast` contract now has an explicit
  root-aligned and right-half regression in
  `test_tmem_physical_bitcast_preserves_subview_mapping`: the right-half case
  first selects columns `64:128` of the original f32 TMEM tile, then bitcasts
  that already-selected physical image to an f16 view, and verifies only the
  selected physical half is overwritten.
- Direct subword `ld/st` instruction-selection coverage now spans `f16`,
  `bf16`, `i16`, and `i8` over identity `128x{64,128,256}` layouts and every
  public variant in `{auto,32x32b,16x64b,16x128b,16x256b}`.
- Narrow x1 subword `ld/st` coverage now spans `f16`, `bf16`, and `i16`
  packed/unpacked cases plus packed and padded `i8` x1 cases for `auto` and
  `32x32b`; direct and descriptor-chain two-CTA x1 subword roundtrips now also
  cover `f16`, `bf16`, `i16`, and packed `i8` over canonical two-CTA
  TMEM-linear layouts.
- As of the latest focused coverage checkpoint, the current source/test slices
  add runtime-matrix allocator-lifetime anchors for `tcgen05.alloc`,
  `relinquish_alloc_permit`, `dealloc`, and `wait`, including pow2 alloc-size
  immediates for both CTA groups, plus exact `tcgen05.commit` opcode anchors.
- The latest full `python/test/gluon/test_tmem_runtime_matrix.py` file
  validation checkpoint remains green:
  - `1757 passed, 442 skipped in 1640.11s (0:27:20)`
- The preferred four-way heavy Gluon sweep over
  `python/test/gluon/test_core.py` and
  `python/test/gluon/test_tmem_runtime_matrix.py` is green at `be14fedc5`:
  - group 1:
    `4407 passed, 634 skipped, 15123 deselected`
  - group 2:
    `2528 passed, 2513 skipped, 15123 deselected`
  - group 3:
    `2694 passed, 2347 skipped, 15123 deselected`
  - group 4:
    `3314 passed, 1726 skipped, 15123 deselected, 1 xfailed`
- The preferred four-way `python/triton_kernels/tests` sweep is green at
  `2ad0ccf5e`:
  - group 1:
    `674 passed, 782 skipped, 4365 deselected`
  - group 2:
    `533 passed, 923 skipped, 4365 deselected`
  - group 3:
    `394 passed, 1062 skipped, 4365 deselected`
  - group 4:
    `776 passed, 677 skipped, 4368 deselected`
  - aggregate:
    `2377 passed, 3444 skipped, 17463 deselected`
- The clean Gluon examples subset remains green at `24bec4ecf`, and the
  attention file now has SDPA-backed benchmark-parameter unit coverage at
  current head:
  - `python/examples/gluon/01-attention-forward.py`:
    `64 passed in 102.66s (0:01:42)` over the benchmark grid except the
    SDPA-OOM context lengths (`N_CTX in {1024,2048,4096,8192}` for tests;
    benchmark remains `2**10..2**16`);
  - covered test parameters: `Z=4`, `H=32`, `HEAD_DIM in {64,128}`,
    `causal in {False,True}`, providers `triton-fp16` and `triton-fp8`, and
    `use_tmem_red in {False,True}` on this Blackwell Ultra box;
  - `python/examples/gluon/02-convolution.py`:
    `48 passed`
  - `python/examples/gluon/03-matmul-multicta.py`:
    `82 passed, 14 skipped`
  - `python/examples/gluon/04-2cta-block-scale-matmul.py`:
    `690 passed, 60 skipped`
  - full `python/examples/gluon` was rerun in the 2026-04-12 GB200 sweep at
    `cb76c31a0` and passed all four split groups.
- The current-head runtime-matrix saturation slices are green:
  - broad `ld/st`:
    `1181 passed, 441 skipped, 1027 deselected`
  - broad `tcgen05.cp`:
    `162 passed, 5 skipped, 2488 deselected`
  - broad `tcgen05.ld.red`:
    `487 passed, 2183 deselected`
  - true `tcgen05.mma` / direct `mma_scaled`:
    `226 passed, 50 skipped, 2407 deselected`
  - scaled-MMA copy-helper matrix:
    `52 passed, 2147 deselected` with exact copy, MMA, and commit opcode checks
- Allocator/lifetime coverage now has explicit runtime anchors:
  - single-CTA and two-CTA ld/st kernels assert exact PTX/LLIR
    `tcgen05.alloc`, `tcgen05.relinquish_alloc_permit`, `tcgen05.dealloc`,
    and `tcgen05.wait::{st,ld}` emission;
  - both CTA groups cover alloc/dealloc size immediates
    `32, 64, 128, 256, 512`;
  - the two-CTA case asserts cluster arrive/wait before dealloc;
  - a source-initialized `allocate_tensor_memory(..., value=...)` kernel
    round-trips the initialized values through a TMEM load.
- Tensor-memory allocation rounding coverage now pins raw live TMEM totals
  `96`, `192`, and `384` as rounded boundaries to supported module allocation
  sizes `128`, `256`, and `512`; literal non-pow2 `tcgen05.alloc` immediates
  are not the compiler contract.
- Commit-opcode coverage now has exact PTX/LLIR anchors:
  - single-CTA MMA pins
    `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
  - two-CTA copy and MMA pin
    `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`.
- The latest full four-way `python/test/gluon` sweep remains the green sweep
  recorded at `77c43f696` / source `be3cba0cd`; the newer `57a06c29b` and
  copy / `ld.red` / scaled-MMA coverage slices were validated with focused
  direct-i8 MMA tests, adjacent positive `tcgen05_mma` kind tests, the
  Blackwell conversion lit test, focused copy and `ld.red` runtime exacts, and
  the focused scaled-MMA opcode matrix.
- A fresh four-way `python/test/gluon` sweep from `be3cba0cd` is green:
  - group 1:
    `5448 passed, 1002 skipped, 19348 deselected`
  - group 2:
    `2666 passed, 3784 skipped, 19348 deselected`
  - group 3:
    `4408 passed, 2041 skipped, 19348 deselected, 1 xfailed`
  - group 4:
    `5582 passed, 866 skipped, 19350 deselected`
  - the current `gb200_current_branch_group{1,2,3,4}_latest_failures.txt`
    manifests are refreshed to `0` nodeids.
- The broad-sweep `.kind::i8` PTX assembler diagnostics are now addressed for
  the direct Gluon API path by `57a06c29b`: on `sm_103a+`,
  direct `tcgen05_mma` i8/u8 inputs with an int32 accumulator fail before IR
  lowering with a clean frontend diagnostic, while the generic
  `compute-capability=100` `ttng.tc_gen5_mma` i8 conversion coverage remains
  intact.
- The direct canonical TMEM-linear `128x128b` copy root coverage gap is closed
  by `c5bdb6d5c`:
  - `test_tmem_runtime_matrix_cp_128x128` now runs both the legacy
    `TensorMemoryLayout` destination and canonical `_make_tmem_linear_layout`
    destination;
  - both variants assert the exact `tcgen05.cp.cta_group::1.128x128b` PTX/LLIR
    family;
  - the linear variant also checks the TTGIR contains `tensor_memory_linear`.
- The standalone scaled `warpx4.32x128b` copy validation no longer depends on
  `_reinterpret` after `bf3dd781b`:
  - `test_tmem_copy_2d`, `test_tmem_runtime_matrix_cp_scales_warpx4`, and the
    scales layout probe validate the copied data through supported logical
    `TensorMemoryScalesLayout` loads;
  - the tests still assert exact `tcgen05.cp.cta_group::1.warpx4.32x128b`
    PTX/LLIR selection;
  - they now assert no `ttg.memdesc_reinterpret` appears in TTGIR for those
    kernels.
- Do not treat scales physical-inspection aliases as ordinary equal-size
  bitcasts: a direct `bitcast(...)` replacement exposed that the old dense
  inspection view was larger than the logical scales descriptor. Use logical
  scales loads for this test intent, and reserve physical bitcasts for the
  explicit offset/slice/subview cases where total size and physical mapping are
  equivalent.
- `128x256b` indexed-view copy coverage is broadened by `7075f31fc`:
  - the fit positive view shape `[128, 128]` now covers f32 and i32 payloads
    across 32, 64, and 128-byte shared swizzles;
  - the full `[2, 128, 256]` indexed parent is recorded as a tensor-memory OOR
    boundary (`Required: 1024, Hardware limit: 512`) instead of a positive
    target.
- No-scales `warpx2` copy exact coverage now includes:
  - single-CTA `warpx2::{01_23,02_13}.64x128b` candidate positives with exact
    `cta_group::1` copy and commit opcode checks plus runtime oracles;
  - two-CTA `warpx2::01_23.64x128b` executable coverage through public
    `SharedLinearLayout` + `TensorMemoryLinearLayout` objects, with exact
    `tcgen05.cp.cta_group::2.warpx2::01_23.64x128b` and multicast commit
    opcode checks;
  - dense shared-layout `warpx2` forms are now clean unsupported for both
    single-CTA and two-CTA cases because runtime probes showed the old
    codegen-only path emitted `warpx2` opcodes while copying the wrong data;
  - the two-CTA `warpx2::02_13` candidate shared layout is now pinned as a
    clean unsupported descriptor-synthesis frontier under the canonical public
    shared layout; it remains a layout-surface / direct-PTX frontier, not a
    proven ISA-impossible negative;
  - a 2026-04-11 temporary direct-seed relaxation for the canonical two-CTA
    `warpx2::02_13` source emitted
    `tcgen05.cp.cta_group::2.warpx2::02_13.64x128b` but produced all-zero
    output, so the current clean unsupported boundary is protecting a real
    wrong-code path and should not be removed without a better descriptor/address
    model.
  - a follow-up direct-PTX patch sweep is recorded in
    `experiments/probe_cp_warpx2_02_13_twocta_direct_ptx.py` with results in
    `experiments/results/probe_cp_warpx2_02_13_twocta_direct_ptx_current.log`:
    opcode-only, source-row-plus-16, single-CTA seed, destination `+4`, and two
    message variants all assemble and launch through the cluster-aware Triton
    launcher, but none match the extended single-CTA `02_13` oracle; the
    descriptor/address mutations either duplicate source-column pairs or copy
    the wrong row/column mix, so they still do not recover the missing 4-byte
    source-column bit.
  - the corrected compile-only direct-seed source-offset JSONL scan covers
    offsets `36..127` with destination deltas `0` and `4`: all executing
    offsets `36..72` duplicate one source-column pair, offsets `73..127`
    launch-fail in isolated child processes, and no source-offset record matches
    the extended two-CTA `02_13` oracle. The earlier NaN-specific breakdown is
    superseded by the compile-only warmup rerun.
- The historical scales `warpx2` probe candidate is now pinned more precisely:
  under public `TensorMemoryScalesLayout` it classifies as
  `tcgen05.copy.warpx4.32x128b` and then hits the tensor-memory-scales
  descriptor-plan diagnostic, now from `TMEMCopyOp` verification rather than
  falling through to late LLVM lowering. `CP_SCALES_LAYOUT_PROBE_CASES` now also
  pins four nearby shared-linear basis-order variants as clean unsupported; all
  of them still map to `warpx4.32x128b` descriptor-plan failures rather than
  public `warpx2` scales lowering. A refreshed FD-capturing subslice probe over
  384 parent-row shared-linear basis interleavings and starts `[0, 32, 64]`
  likewise found `1152` clean unsupported outcomes, zero unknown or bug-like
  failures, and no `warpx2` opcode. The `warpx4` control remains the positive
  copy path for this public scales surface.
- No-scales `cta_group::2` dense copy coverage now includes both dense
  families:
  - `tcgen05.cp.cta_group::2.128x128b` for the fixed `256x4` two-CTA
    i32/f32-bitwidth shape through explicit `SharedLinearLayout`;
  - `tcgen05.cp.cta_group::2.128x256b` for the existing wider N matrix.
- `ld.red` modifier saturation is expanded by `eac11c719`:
  - positive identity, tile-permuted, column-permuted, and row-permuted linear
    layouts now cover all four legal modifier pairs:
    plain, propagate-NaN, abs, and abs+propagate-NaN;
  - mixed unsupported layouts check the same modifier matrix and continue to
    report the dedicated `tmem_load reduction source layout is not directly
    tcgen05.ld.red-compatible` diagnostic.
- `ld.red` row-256 source-boundary coverage is expanded:
  - reduction-friendly `256x{32,64,128}` TMEM-linear layouts with the extra
    row bit carried in the column/query frame are positive and still emit
    `tcgen05.ld.red.sync.aligned.32x32b`;
  - plain identity `256x{32,64,128,256}` TMEM-linear source layouts are pinned
    as clean unsupported cases with the same actionable software-reduction
    diagnostic.
- `ld.red` explicit reduction-load layout coverage is expanded:
  - compatible explicit variants `auto`, `32x32b`, `16x32bx2`, and
    `32x32b_splitn` all execute correctly and still canonicalize to the exact
    `tcgen05.ld.red.sync.aligned.32x32b.x128.{min,max}.f32` opcodes;
  - explicit N-sharded variants `16x64b`, `16x128b`, and `16x256b` are pinned
    as clean unsupported cases with the dedicated `N dimension sharded across
    threads` diagnostic and register-layout note;
  - focused min/max explicit-layout validation: `11 passed in 8.57s`;
  - broad `ld_red` validation: `487 passed, 2183 deselected in 482.92s
    (0:08:02)`.
- Scaled-MMAv5 copy-matrix instruction coverage is tightened by `3374bbf12`:
  - the runtime-matrix scaled copy tests still assert exact
    `tcgen05.cp.cta_group::{1,2}.warpx4.32x128b` PTX/LLIR streams;
  - the same format/CTA/geometry/accumulator-layout matrix now also asserts
    exact `tcgen05.mma.cta_group::{1,2}.kind::{mxf8f6f4,mxf4,mxf4nvf4}.block_scale.scale_vec::*`
    opcode selection for PTX and LLIR.
- Tile-permuted MMAv5 accumulator coverage now spans all plain supported
  operand kinds:
  - `f16`, `tf32`, `bf16`, `f8e5m2`, and `f8e4m3`;
  - each kind is covered for `128x128` / `tile_n=32` and `128x256` /
    `tile_n=64` accumulator layouts;
  - both no-accumulator and `use_acc=True` accumulator-add paths are covered;
  - each path checks PTX/LLIR opcode equality and the expected
    `tcgen05.mma` kind, with commit opcode checks on the `use_acc=True` slice.
- Direct scaled-MMAv5 TMEM-view tests now pin exact `mxf8f6f4` scaled-MMA
  opcodes after `5e3b2ae87`:
  - this covers the existing minimal, block-N direct-layout, accumulator
    subview, LHS subview, and tile-permuted accumulator runtime tests;
  - the original e5m2/e5m2 direct-view kernels remain in place.
- Direct scaled-MMAv5 accumulator-subview format coverage is broadened by
  `602fd9b44`:
  - the new direct subview matrix covers `mxfp8/mxfp8`, `mxfp4/mxfp4`,
    `mxfp8/mxfp4`, `mxfp4/mxfp8`, and `nvfp4/nvfp4`;
  - it validates numeric output, exact PTX/LLIR
    `mxf8f6f4` / `mxf4` / `mxf4nvf4` scaled-MMA opcodes, and the
    `ttg.memdesc_subslice` + `tensor_memory_linear` accumulator path.
- That direct scaled-MMAv5 accumulator-subview format matrix now covers both
  `slice_start=0` and `slice_start=64`, so the same format/opcode assertions
  exercise root-aligned and offset accumulator subviews.
- Direct scaled-MMAv5 TMEM-LHS format coverage now includes both subview and
  full-shape tile-permuted packed-storage reachable subsets:
  - `test_tmem_runtime_matrix_mma_scaled_lhs_subslice_view_format_matrix`
    covers `mxfp8/mxfp8`, `mxfp8/mxfp4`, `mxfp4/mxfp4`, and `nvfp4/nvfp4`
    for both legacy and canonical TMEM-linear accumulator layouts;
  - `test_tmem_runtime_matrix_mma_scaled_lhs_tile_permuted_format_matrix`
    covers the same format/layout subset for a full-shape tile-permuted
    operand-A descriptor at logical `K=256`, with no `ttg.memdesc_subslice`;
  - both tests feed TMEM operand-A descriptors directly to `tcgen05_mma_scaled`
    and pin exact PTX/LLIR scaled-MMA and commit opcodes;
  - mixed `mxfp4/mxfp8` dense TMEM-LHS subviews and full-shape
    tile-permuted TMEM-LHS descriptors are pinned as clean unsupported cases
    because `mxf8f6f4` fp4 LHS requires padded operand-A storage currently
    represented by `fp4_padded` shared memory; the durable subview probe is
    `experiments/probe_mma_scaled_lhs_subslice_formats.py` with current results
    in `experiments/results/probe_mma_scaled_lhs_subslice_formats_current.log`.
- Two-CTA direct scaled-MMAv5 accumulator-subview coverage is now present:
  - `test_tmem_runtime_matrix_mma_scaled_twocta_acc_subslice_view_format_matrix`
    reuses the cga-aware TMA/scales-copy topology, allocates a larger
    TMEM-linear accumulator parent, slices it to the `cta_group::2` MMA result
    tile, and covers both `slice_start=0` and `slice_start=128`;
  - it validates numeric output, exact PTX/LLIR scaled-MMA opcodes for the same
    five format pairs, exact `tcgen05.cp.cta_group::2.warpx4.32x128b` scale
    copies, exact multicast `tcgen05.commit.cta_group::2`, and
    `ttg.memdesc_subslice` + `tensor_memory_linear` + `two_ctas` TTGIR.
- Plain MMAv5 kind saturation now includes f16 in the explicit 1-CTA and
  2-CTA kind matrices after `7766be003`, the one-CTA root accumulator matrix
  now spans `N=128` and `N=256`, and the current two-CTA matrix has been
  widened to `N=128` and `N=256`:
  - `MMA_PLAIN_KINDS` covers `f16`, `tf32`, `bf16`, `f8e5m2`, and `f8e4m3`;
  - both legacy and canonical linear accumulator layouts are covered for
    `cta_group::1` and `cta_group::2`;
  - the `cta_group::1` root kind matrix covers both `128x128` and `128x256`
    accumulator shapes, and the `cta_group::2` kind matrix covers both
    `256x128` and `256x256` accumulator shapes;
  - the tests continue to assert exact PTX/LLIR opcode agreement and pin exact
    root op counts: `f16=2`, `bf16=2`, `tf32=4`, and `f8e5m2/f8e4m3=1`.
- Plain MMAv5 `use_acc=True` coverage now spans all `MMA_PLAIN_KINDS`
  across both 1-CTA and 2-CTA paths, for both legacy and canonical
  TMEM-linear accumulator layouts, validating runtime accumulator addition
  plus exact PTX/LLIR opcode agreement and the same exact root op counts.
- Plain MMAv5 tile-permuted accumulator coverage now also pins its expected
  expanded instruction counts: four times the root kind count for accumulator
  tile permutations. Full-shape tile-permuted TMEM-LHS coverage now spans all
  `MMA_PLAIN_KINDS` for a `128x256` operand-A tile and pins eight times the root
  kind count (`f16/bf16=16`, `tf32=32`, `f8e5m2/f8e4m3=8`).
- Plain MMAv5 TMEM-LHS subview coverage now spans all `MMA_PLAIN_KINDS` for
  `N in {128, 256}` and both legacy/canonical TMEM-linear accumulator layouts:
  - `test_tmem_runtime_matrix_mma_lhs_subslice_view_plain_kinds` slices the
    right half of a TMEM-linear operand-A parent and feeds that subview directly
    to `tcgen05_mma`;
  - the matrix pins exact PTX/LLIR opcodes, expected instruction counts
    (`f16=2`, `bf16=2`, `tf32=4`, `f8e5m2/f8e4m3=1`), and the single-CTA
    commit opcode for every supported plain kind.
- Plain full-shape tile-permuted TMEM-LHS coverage now includes `N=256` for all
  non-OOR direct shared-B shapes. The `tf32, N=256, K=256` direct helper shape
  is omitted because it exceeds shared memory before launch.
- Staged `ld/st` fuzzing has started with descriptor-chain `auto` coverage at
  `ebb23b697`:
  - the basic descriptor-composition matrix now covers `auto` instruction
    selection in addition to explicit `32x32b`, `16x64b`, `16x128b`, and
    `16x256b`;
  - coverage spans identity and mixed TMEM-linear layouts at `N = 64, 128,
    256`.
- Two-CTA descriptor-chain `auto` coverage is added by `ca9f94760`:
  - `LDST_TWOCTA_DESCRIPTOR_CASES` now uses the same `LDST_VARIANTS` matrix;
  - coverage reaches both `block_two_ctas` and MMAv5-like two-CTA layouts;
  - the validation also reran the single-CTA descriptor roundtrip sweep now
    reached by `LDST_DESCRIPTOR_CASES`.
- Higher-rank `ld/st` index descriptor views now include `auto` instruction
  selection after `7df887318`:
  - auto subview shapes are pinned to the expected `32x32b` family;
  - single-CTA identity/mixed and two-CTA block/MMAv5-like layouts are covered
    for `N = 64, 128`.
- Higher-rank multidimensional-slice `ld/st` descriptor views now include
  `auto` instruction selection after `81b4e5a9d`:
  - single-CTA identity/mixed and two-CTA block/MMAv5-like layouts are covered
    for `N = 64, 128`;
  - the tests continue to check functional output, PTX/LLIR agreement, and the
    expected reshape/transpose/split/join IR markers.
- Higher-rank clean OOR guards now include `auto` after `e2d793726`:
  - single-CTA and two-CTA higher-rank index, multidimensional-slice, and
    dim0-slice resource-boundary tests all use `LDST_VARIANTS`;
  - validation confirms these remain clean tensor-memory OOR boundaries.
- Direct higher-rank `ld/st` access is now pinned as a clean unsupported
  boundary:
  - direct rank-3 `get_reg_layout`, explicit `load`, and explicit `store`
    attempts fail in the Gluon descriptor API with a 2D-only diagnostic telling
    users to index/slice/reshape higher-rank TMEM descriptors to a 2D view;
  - the C++ `getDistributedLayoutForTmemLdSt` planners return no layout for
    non-2D physical TMEM layouts instead of reaching the old `dims.size() == 2`
    assertion;
  - focused validation: `4 passed in 3.14s`, plus adjacent higher-rank indexed
    and dim0-slice positives `30 passed in 25.88s`;
  - broader `ldst and higher_rank` validation: `144 passed, 1 skipped,
    2521 deselected in 73.43s (0:01:13)`.
- Higher-rank half-row clean errors now include `auto` after `7c1a6f63b`:
  - single-CTA and two-CTA lifted half-row clean-negative matrices now use
    `LDST_VARIANTS`;
  - explicit variants keep the existing unsupported-layout `CompilationError`
    contract, while `auto` may report the direct descriptor parser
    `RuntimeError` about translated row-half TMEM origins;
  - the tests still require a descriptor-view diagnostic and guard against
    PassManager/assertion crashes.
- Higher-rank dim0-slice positive `ld/st` views now include `auto` after
  `572fb25b2`:
  - single-CTA identity and two-CTA block lifted-layout dim0-slice matrices now
    use `LDST_VARIANTS`;
  - the tests validate functional output plus full-tile and narrowed-subview
    PTX/LLIR opcode agreement.
- The two-CTA MMAv5-like higher-rank `ld/st` clean-negative matrix now includes
  `auto` after `8f622b938`:
  - the unsupported matrix uses `LDST_VARIANTS`;
  - validation confirms auto still fails through the clean CTA-per-CGA mismatch
    path rather than PassManager/assertion crashes.
- Same-row/column permuted `ld/st` layouts now include `auto` after
  `c83d358e4`:
  - `LDST_PERMUTED_CASES` uses `LDST_VARIANTS`;
  - this reaches both direct ld/st and descriptor-chain composition runtime
    tests;
  - validation covers numerical behavior, `tensor_memory_linear` IR, and exact
    PTX/LLIR opcode agreement.
- Exotic `ld/st` layout matrices now include `auto` after `20f2c8db8`:
  - direct exotic, descriptor-chain exotic, and clean unsupported block-layout
    matrices use `LDST_VARIANTS`;
  - validation covers positive opcode agreement and clean negative diagnostics.
- Row/column cross-product permuted `ld/st` layouts now include `auto` after
  `6d44718e7`:
  - `LDST_ROWCOL_PERMUTED_CASES` uses `LDST_VARIANTS`;
  - this reaches direct ld/st and descriptor-chain composition sweeps;
  - focused validation ran the newly added auto parametrizations across the
    row/column permutation cross-product.
- Row/column descriptor-roundtrip `ld/st` boundary coverage now includes
  `auto` after `00139772b`:
  - `LDST_DESCRIPTOR_ROUNDTRIP_ROWCOL_CASES` uses `LDST_VARIANTS`;
  - focused validation shows the new auto parametrizations all hit the existing
    clean tensor-memory OOR skip boundary for lifted roundtrip shapes.
- The stale unused `LDST_EXPLICIT_VARIANTS` / higher-rank positive case list is
  removed after `a487942ac`, so scans no longer report a false explicit-only
  active `ld/st` matrix.
- Rank-5 descriptor-roundtrip `ld/st` boundary coverage now includes `auto`
  after `be193b6c6`:
  - one-CTA rank-5 cases use `LDST_VARIANTS`;
  - two-CTA rank-5 cases add `auto` to the existing explicit subset;
  - focused validation shows all new auto cases hit the existing clean
    tensor-memory OOR skip boundary.
- F16 subword ld/st pack/unpack coverage now includes `auto` after
  `4fe7a44de`:
  - `F16_LDST_CASES` uses `LDST_VARIANTS`;
  - the `auto` shape expectations match the explicit `32x32b` f16 packet
    family.
- X1 f32 direct and descriptor-chain ld/st roundtrips now include `auto` after
  `c475eecc0`:
  - `X1_F32_LDST_VARIANTS` covers `auto` and explicit `32x32b`;
  - validation covers one-CTA linear, one-CTA legacy, and two-CTA linear
    layouts with exact `32x32b.x1.b32` opcode expectations.
- X1 f16 ld/st roundtrips now include `auto` after `a950338f7`:
  - `X1_F16_LDST_VARIANTS` covers `auto` and explicit `32x32b`;
  - validation covers linear packed, legacy packed, and legacy unpacked
    layouts with the existing pack/unpack opcode expectations.
- Focused `ld/st` fuzz/stale-coverage cleanup now includes the remaining
  fixed-offset and rank-5 variant tails:
  - `test_tmem_runtime_matrix_ldst_fixed_offset_patterns_128x256` includes an
    `auto` companion and confirms it selects the canonical
    `32x32b.x64.b32` offset pattern;
  - two-CTA rank-5 descriptor roundtrip coverage uses the full
    `LDST_VARIANTS` set and confirms the newly added large explicit families
    hit the clean tensor-memory OOR boundary.
- Scales `ld/st` instruction-selection coverage now includes `auto`:
  - auto does not simply mirror explicit `32x32b`; for several shapes it
    selects wider `16x64b`, `16x128b`, or `16x256b` packet streams with the
    expected high-half `1048576` offsets;
  - `test_tmem_runtime_matrix_ldst_scales_variant_sweep` pins those exact
    PTX/LLIR streams across the existing scales variant matrix.
- Combined current-head `ld/st` runtime-matrix validation is green after the
  staged auto expansions, allocator-lifetime anchors, and subword coverage
  expansions:
  - command:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldst-broad-after-subword-refresh PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k ldst`;
  - result:
    `1181 passed, 441 skipped, 1027 deselected in 1340.05s (0:22:20)`;
  - skips are expected tensor-memory OOR / clean-boundary cases in lifted
    descriptor roundtrip and rank-5 families.
- Current-head `tcgen05.cp` runtime-matrix validation is green:
  - command:
    `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-cp-scales-verifier-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k cp`;
  - result:
    `162 passed, 5 skipped, 2488 deselected in 40.99s`;
  - this covers the current no-scales `warpx2` positives with exact commit
    opcodes, including the two-CTA `warpx2::01_23` path, the new two-CTA
    `warpx2::02_13` canonical-shared clean unsupported boundary, dense-shared
    `warpx2` clean negatives, dense copy positives, 2-CTA `128x128b` /
    `128x256b` copy, scaled `warpx4` copy paths, exact scaled-copy commit
    opcodes, and clean unsupported copy boundaries; a follow-up direct-seed
    probe found that aligned `tmemDwordDelta=0` moves data for two-CTA
    `02_13` but duplicates each source-column pair, `tmemDwordDelta=4` is the
    known all-zero path, unaligned deltas trap, and the known seed fields do not
    recover the missing source-column bit; it also covers the historical scales
    `warpx2` candidate and four nearby basis-order variants as clean unsupported
    verifier-side `warpx4.32x128b` descriptor-plan failures.
- Current-head `tcgen05.ld.red` runtime-matrix validation is green:
  - command:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-ldred-wait-broad PYTHONPATH=python:. pytest -s --tb=short -q python/test/gluon/test_tmem_runtime_matrix.py -k ld_red`;
  - result:
    `487 passed, 2183 deselected in 482.92s (0:08:02)`;
  - this covers identity, tile-permuted `128x{64,128,256}` including the
    minimal `N=64, tile_n=16` case, pure column-permuted
    `128x{64,128,256}`, pure row-permuted `128x{64,128,256}`, row/column
    cross-product permutations at `128x{64,128,256}`, all legal modifier pairs,
    reduction-friendly row-256 positives, clean unsupported identity row-256
    layouts, exact offset immediates (`[0]` for `N <= 128` and
    `[0, 64, 128, 192]` for `N=256`), exact `tcgen05.wait::{st,ld}`
    emission and ordering around store/reduction-load consumption, and clean
    unsupported mixed-layout negatives.
- Current-head direct `tcgen05.mma` / `mma_scaled` runtime-matrix validation is
  green:
  - command:
    four-GPU `pytest-split` groups over
    `python/test/gluon/test_tmem_runtime_matrix.py -k 'mma and not cp'`;
  - result:
    aggregate selected coverage `242 passed, 50 skipped`;
  - this covers canonical, indexed, subview, tile-permuted, 1-CTA and 2-CTA
    direct MMA surfaces plus direct scaled-MMA view cases, including root-aligned
    and offset one-CTA accumulator subview format-matrix cases, the two-CTA
    cga-aware accumulator-subview matrix, tile-permuted plain-kind accumulators
    at `128x128/tile_n=32` and `128x256/tile_n=64`, the all-plain-kind
    TMEM-LHS subview matrix for both legacy/canonical accumulator layouts, the
    scaled TMEM-LHS subview format matrix, and two-CTA plain-kind accumulators at
    `256x128` and `256x256`; it also covers both 1-CTA and 2-CTA `use_acc=True`
    plain-kind accumulator addition for all supported plain kinds and both
    legacy/canonical accumulator layouts, plus TMA-fed two-CTA TF32 when matrix
    B is loaded through a non-transposed `[N, K]` TMA
    descriptor and passed to MMAv5 as a shared-memory `permute((1, 0))` view.
    Scaled-MMA copy-helper coverage remains tracked separately.
- Tile-permuted scaled-MMAv5 accumulator-subview clean-negative coverage now
  spans the same format pairs as the positive direct subview matrix:
  - `mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`,
    `mxfp4/mxfp8`, and `nvfp4/nvfp4`;
  - each case confirms the repeated `N=32` tile-permuted layout fails with the
    dedicated `matrix-B scale fragments at 64-column alignment` diagnostic and
    does not fall through to a PassManager/assertion crash.
- The recorded 2-CTA TF32 TMA-fed shared-transpose issue is now pinned as a
  clean verifier negative instead of a late LLVM pass failure:
  - `test_tmem_runtime_matrix_mma_twocta_tma_tf32_reports_clean_shared_transpose_error`
    covers both legacy and canonical TMEM-linear accumulators;
  - legal non-transposed TMA descriptors still cannot feed this TF32 MMAv5
    shape because TMA descriptors cannot be transposed and MMAv5 rejects the
    resulting transposed shared-memory operand;
  - the verifier now reports
    `tcgen05.mma does not support transposed float32 operands in shared memory`
    without `PassManager::run failed` or assertions;
  - direct non-TMA 2-CTA TF32 remains covered by the green direct-MMA slice, so
    this remains a TMA-to-shared layout materialization/API frontier rather
    than a direct accumulator-layout regression.
- The supported M64 subview/physical-bitcast slice is now checkpointed:
  - `47a07a37d` added normalized source-query inversion for physical bitcast
    views whose source subview keeps inactive zero support bases;
  - `85d8dbbf4` preserved out-of-range physical row/col origins for narrowed
    subviews, migrated the M64 `_reinterpret` tests to supported
    `slice/subview -> bitcast` where the mapping is equivalent, and made
    parameterized kernels pass layouts as explicit constexpr arguments so
    Gluon cache keys distinguish legacy and linear variants.
- The wider grouped `python/test/gluon` group-3 rerun that started before
  these two checkpoints is now superseded by the green group-3 rerun above. It
  had reduced the stale current-head red set to:
  - `test_tmem_subslice_block_m_64_parent_layout[linear]`;
  - `test_block_m_64_mma[legacy]`;
  - `test_block_m_64_mma[linear]`;
  - result: `3 failed, 4406 passed, 2041 skipped, 19348 deselected`.
- Focused current-head reruns after `85d8dbbf4` close that exact M64 set:
  - subview/physical-bitcast exacts:
    `4 passed`;
  - `test_block_m_64_mma[legacy]` / `[linear]`:
    `1 passed, 1 xfailed`;
  - the xfail is deliberate design debt for legacy M64 `64x64` layout sugar
    lacking producer-visible physical-family semantics for MMAv5 consumers;
    the linear supported layout passes in the same process. This is now an
    explicit follow-up fix on the docket, not just a status note, but it should
    stay behind current CI-lane freshness and aggregate examples validation.
- The supported descriptor bitcast API remains on the branch, and the attention
  scratch-alias path is migrated across the SDPA-safe benchmark-shaped unit
  matrix:
  - `python/examples/gluon/01-attention-forward.py` no longer uses
    `_reinterpret` for the scratch-borrow helpers;
  - P scratch slicing is dtype-aware: fp16/bf16 use 64 f32 columns for the P
    image, while fp8 uses 32, and alpha/epilogue scratch is placed after the
    actual P physical region;
  - exp2 partitions still slice the original f32 scratch subregion before
    bitcasting each exact physical image;
  - causal `use_tmem_red` now computes the diagonal-stage row max after causal
    masking instead of using a pre-mask `ld.red` max;
  - the SDPA-backed attention matrix is green:
    `64 passed in 102.66s (0:01:42)`;
  - full `python/examples/gluon` still needs a fresh aggregate rerun before
    marking the broader examples lane current.
- The stale `test_mma_shared_inputs` two-CTA TTGIR spelling bucket is closed:
  - current TMEM encoding text may spell the type field as `twoCTAs`, while
    physical-layout and op attributes still use `two_ctas`;
  - the assertion now accepts either spelling;
  - patched four-way `test_mma_shared_inputs` refresh:
    `3830 passed, 490 skipped`, `2954 passed, 1366 skipped`,
    `1206 passed, 3114 skipped`, and `2584 passed, 1736 skipped`.
- The focused persistent `python/triton_kernels/tests/test_matmul.py`
  shared-memory OOR bucket is fixed, and the full directory now has a green
  current-head rerun:
  - root cause was ordinary replayed `ttng.tmem_subslice` loads/stores using
    the exact preserved physical query and therefore selecting a root-width
    register family for a shape-local replay slice;
  - replayed subslices now use the subview-local descriptor type for register
    layout selection;
  - only explicit `tmem_physical_bitcast` roots use the preserved physical
    query, keeping the supported physical bitcast contract intact.
- Validation for the latest code checkpoint:
  - build: `PASSED`
  - lit `tmem_layouts.mlir` and `tritongpu_to_llvm_blackwell.mlir`:
    `2 passed`
  - representative persistent matmul OOR repro:
    `1 passed`, metadata back to `shared=214120`
  - focused persistent fp8/mxfp4 matmul slice:
    `16 passed, 6 skipped`
  - full 4-way `python/triton_kernels/tests` refresh:
    `2377 passed, 3444 skipped`
- The M64 row/col-permuted split-N direct ld/st bucket remains closed as a
  positive hardware surface:
  - exact row-permuted `16x32bx2` warp anchors that would lower to misaligned
    one-row PTX addresses are rejected;
  - simple row/col-permuted `64xN` f32 TMEM-linear roots select the canonical
    aligned M64 split-N register layout;
  - backend default-layout selection has the same fallback, so conversions do
    not hit the empty-compatible-layout assertion.
- The full `python/test/gluon/test_tmem_runtime_matrix.py` file was refreshed
  from this branch:
  - full-file rerun before the final assertion refresh:
    `1437 passed, 354 skipped, 1 failed in 1226.69s`;
  - the only failure was a stale TTGIR text assertion in
    `test_tmem_runtime_matrix_cp_no_scales_indexed_view_canonicalized[128]`;
  - that test now checks the stable `ttng.tmem_physical_layout` marker on the
    legacy `tensor_memory_encoding` view path, and the exact nodeid rerun
    passes.
- The old `413`-nodeid branch-added runtime-matrix manifest is now refreshed
  to empty:
  - `gb200_current_branch_test_tmem_runtime_matrix_focus_e70a3aa09_failures.txt`
- The old `332`-nodeid focused `test_core.py` TMEM/MMA manifest is also now
  refreshed to empty:
  - the full focused manifest rerun passed:
    - `332 passed in 125.78s`
  - the only intermediate issues were stale expectations for:
    - the row-encoded `1048576` PTX offset in the legacy `64x128` split-N
      roundtrip;
    - the current clean unsupported descriptor-view diagnostic for a block
      basis `ld.red` negative.
  - `gb200_current_branch_test_core_group3_focus_e70a3aa09_failures.txt`
- The latest recorded full GB200 `make NUM_PROCS=24 test-unit` lane was
  recovered at an older checkpoint and must be rerun at current HEAD before a
  full fresh CI stamp:
  - checkpoint `78196b4e4` reran the full wrapper green;
  - main `python/test/unit`: `15153 passed, 5492 skipped`;
  - `python/test/unit/test_debug.py`: `95 passed`;
  - unit-tail sublanes were green: `python/triton_kernels/tests`,
    `python/tutorials/06-fused-attention.py`, instrumentation, and plugin tests;
  - current unit manifests are empty:
    `gb200_current_branch_test_unit_failures.txt`,
    `gb200_current_branch_test_unit_matmul_refresh_failures.txt`,
    `gb200_current_branch_test_unit_tensor_descriptor_refresh_failures.txt`,
    `gb200_current_branch_test_unit_warp_specialization_refresh_failures.txt`,
    and `gb200_current_branch_test_unit_rowanchor_refresh_failures.txt`;
  - this unit green point predates later focused compiler/test coverage commits,
    so do not treat it as a current-HEAD full-wrapper rerun without refreshing.
- The supported `_reinterpret` migration invariant remains:
  - offset to the right part of TMEM;
  - slice/subview it to the desired physical bits;
  - bitcast to the desired dtype/shape/layout only when equal-size and
    physical-mapping equivalent to the input descriptor.
- Next required durable step: continue the remaining recovery queue with the
  intentionally reverted attention example tracked separately from supported
  bitcast API validation. Keep the legacy M64 MMAv5 xfail as an explicit
  follow-up fix item, then continue broader MMAv5/`mma_scaled`, staged `ld/st`
  fuzzing, stale-negative cleanup, and heuristic phases, with any remaining copy
  work limited to shapes that fit the TMEM allocation budget.

## 2026-04-12 15:20 UTC: layout-only row-plan cleanup closes preserve-set reds

- Starting point for this slice: `a88f55a81` on `codex/tmem`, dirty in
  `TensorMemoryUtils.cpp` and `TensorMemoryToLLVM.cpp`; target remote remains
  `origin/codex/tmem`.
- Root cause after removing the TMEM side-channel attrs:
  - the LLVM lowering still had a hard-coded f32 `64x128` reinterpret raw-query
    override that replaced exact view-chain layout arithmetic with
    `toLinearLayout(memTy)` and forced `{16, 32, 64}` row anchors;
  - same-dtype `memdesc_reinterpret` views could borrow the source support row
    bit order instead of planning from the destination/view layout;
  - pure outer `memdesc_index` views were using the wider M64 type/family row
    plan even when the concrete inferred query layout had no MMAv5 family block
    dimension, so a store through a chained view and a later load through
    `tmem.index(1)` chose different representatives of the zero-row/broadcast
    layout.
- Fix direction:
  - remove the remaining hard-coded reinterpret raw-query override;
  - keep same-dtype non-physical reinterpret support on the destination/view
    layout when it is standalone-query representable;
  - keep borrowed/wider backing row plans only when the query layout can
    materialize both requested row anchors;
  - for pure outer indexes, use the concrete query layout row plan unless that
    query layout itself still carries the MMAv5 family `block` dimension.
- This preserves the project invariant: the descriptor layout/query is the
  source of truth. Zero bases mean broadcast/equivalence; the planner must not
  choose a divergent hidden physical representative based on producer attrs.
- Current validation against rebuilt compiler:
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`: passed;
  - `make test-lit`: `248 passed, 2 unsupported`;
  - `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix`: `26 passed`;
  - all `162` exact nodeids from
    `gb200_branch_new_20260412_unit_main_failures.txt`: `162 passed`;
  - `test_tmem_linear_m64_roundtrip_direct_shapes` plus `test_block_m_64_mma`:
    `20 passed`;
  - `test_tmem_physical_bitcast_preserves_subview_mapping[0]` and `[64]`:
    `2 passed`;
  - hygiene: `git diff --check` passed; removed-attr production sweep has no
    production hits, only negative test assertions.
- Status implication:
  - the preserve-set lit, unit, descriptor-chain, direct-M64, block-M-64 MMA,
    and physical-bitcast checks are green at this checkpoint;
  - older manifests that reported the `162` unit exacts or M64 descriptor-chain
    cases as current red are now stale and must not be used for prioritization
    without rerunning them.
- Remaining after this checkpoint:
  - run broader GB200 grouped sweeps when this slice is committed/pushed;
  - keep the legacy default-load opcode-quality issue on the docket if it still
    appears in a wider sweep;
  - continue the longer plan: supported view/bitcast API migration,
    attention rewrite later, `ld.red`, `copy` warpx2, broader MMAv5 family
    coverage, heuristic cleanup, and staged fuzzing/validation.

## 2026-04-13 08:50 UTC: ld/st descriptor reinterpret test usage is reduced

- Runtime-matrix descriptor-chain coverage is now closer to the supported TMEM
  view API contract:
  - `tmem_ldst_descriptor_chain_kernel` uses `.bitcast(...)` instead of
    `_reinterpret(...)` for the physical-equivalent descriptor view it actually
    needs;
  - the deep descriptor roundtrip chain no longer performs a no-op
    `_reinterpret` to the same dtype, shape, and layout after the view
    roundtrip;
  - the roundtrip parametrization was renamed from `slice_index_reinterpret` to
    `slice_index_deep_roundtrip`, and the required view ops are now asserted in
    TTGIR for both single-CTA and two-CTA roundtrip tests.
- Important boundary:
  - a genuine `.bitcast(...)` after the composed
    `reshape/permute/permute/reshape` view chain still fails type inference with
    `unsupported tensor memory memdesc_subslice view`;
  - that is a future API/planner gap only for real physical-equivalent bitcasts
    over complex descriptor-view chains, not needed for this no-op same-layout
    test case.
- Validation:
  - `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`
    passed;
  - rebuild passed;
  - `git diff --check` passed before docs update;
  - base descriptor-composition split groups passed, with group 3 requiring a
    warmed-cache rerun after a cold-compile timeout (`8 passed`, `8 passed`,
    `8 passed`, `6 passed` across the four groups);
  - the deep roundtrip selectors completed without failures but all selected
    cases took the existing OOR skip path (`30` single-CTA skips and `30`
    two-CTA skips).
- Velocity note:
  - an initial combined `-k` selector was overbroad because
    `ldst_descriptor_compositions` also matched permuted/rowcol/exotic sweeps;
    use exact nodeids for the base function when validating this slice.

## 2026-04-13 09:05 UTC: core descriptor-chain matrix uses supported bitcast

- Migrated `python/test/gluon/test_core.py::tmem_descriptor_chain_matrix_kernel`
  from `_reinterpret(...)` to `.bitcast(...)` for the same descriptor-chain
  physical-equivalent view pattern already migrated in the runtime matrix.
- `test_tmem_descriptor_chain_matrix` now asserts `tmem_physical_bitcast` in
  TTGIR in addition to the existing descriptor-view operations and exact
  `tcgen05.{st,ld}` opcode checks.
- Validation:
  - `python3 -m py_compile python/test/gluon/test_core.py` passed;
  - rebuild was a no-op success;
  - exact nodeid across four GPU split groups passed all `26` selected cases
    (`7`, `7`, `7`, `5`).

## 2026-04-13 09:35 UTC: runtime-view identity slices use supported bitcast

- Extended the `test_core.py` runtime-view API migration beyond the
  `index_reshape_*` family: the two identity `slice_*` cases now use supported
  `.bitcast(...)` through `tmem_linear_runtime_view_kernel_a`.
- The enabling planner/query change is deliberately narrow:
  - `verifyTMemSubsliceProjection` verifies against the normalized analysis
    layout when the destination layout is non-injective only because of
    inactive zero support bases; this preserves broadcast/equivalence semantics
    instead of rejecting an otherwise physical-equivalent view;
  - the standalone same-rank TMEM subview query path now handles same-rank TMEM
    subslices beyond the previous rank-2 special case.
- Superseded by the 2026-04-13 10:15 UTC checkpoint: the mixed-basis case was
  reclassified as not physical-mapping equivalent to the requested identity
  layout, moved out of the runtime-positive matrix, and converted to a clean
  supported-bitcast negative.
- Validation:
  - `python3 -m py_compile python/test/gluon/test_core.py` passed;
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8` passed;
  - `python/test/gluon/test_core.py::test_tmem_linear_runtime_views` passed all
    `11` selected cases across four GPU split groups (`3`, `3`, `3`, `2`);
  - adjacent `test_tmem_descriptor_chain_matrix` passed all `26` selected cases
    after rerunning cold-timeout groups 3 and 4 against warmed caches (`7`,
    `7`, `7`, `5`).

## 2026-04-13 09:50 UTC: runtime-matrix block descriptor negative uses supported bitcast

- Removed the last `_reinterpret(...)` use from
  `python/test/gluon/test_tmem_runtime_matrix.py`: the block-descriptor
  clean-negative kernel now spells its view change as supported `.bitcast(...)`.
- The one-CTA block case still reports the expected CTA mismatch cleanly. The
  two-CTA block case now cleanly rejects earlier at supported bitcast/subslice
  formation with `unsupported tensor memory memdesc_subslice view`, which better
  matches the supported API contract than using `_reinterpret(...)` to force the
  descriptor into a deeper direct ld/st unsupported-row-anchor diagnostic.
- Probe note: the blocked two-CTA bitcast would require a negative additive
  block-origin delta (`dim=2 step=2 phys=block delta=-1`) in the current query
  model, so it remains a real complex-view boundary rather than a direct ld/st
  negative.
- Validation: `py_compile`, `make -j8`, and `git diff --check` passed. The
  exact block-descriptor nodeid passed both selected cases across the active
  four-GPU split groups (`1`, `1`; groups 3 and 4 were empty because only two
  cases are collected).

## 2026-04-13 10:00 UTC: frontend TMEM reinterpret users classified

- Probed the three TMEM `_reinterpret(...)` helpers in
  `python/test/gluon/test_frontend.py` with supported `.bitcast(...)`. The
  probe was reverted: the focused parser tests fail before IR generation with
  `unsupported tensor memory memdesc_subslice view` on the same composed
  slice/reshape/transpose pattern.
- Classification: keep those frontend uses as intentional `memdesc_reinterpret`
  parser/IR contract coverage until the complex physical-bitcast-over-view-chain
  planner/API gap is fixed. They should not be treated as production-style
  reliance on private `_reinterpret` lowering.
- Remaining `_reinterpret(...)` users after this classification are
  intentional raw frontend `memdesc_reinterpret` parser/IR coverage,
  shared-memory reinterpret tests in `test_core.py` and `test_frontend.py`, and
  the persistence tutorial scratch-buffer borrow path, which borrows shared
  memory B buffers for the epilogue rather than tensor memory.

## 2026-04-13 10:15 UTC: test_core mixed runtime view is a clean bitcast negative

- Removed the last TMEM `_reinterpret(...)` use from
  `python/test/gluon/test_core.py`. The mixed-basis runtime-view case is no
  longer a runtime positive through legacy `_reinterpret(...)`; it is now a
  supported `.bitcast(...)` clean-negative case named
  `slice_bitcast_64_mixed_not_physical_equivalent_32x32b`.
- Classification: this was not a valid physical-equivalent bitcast to the
  requested identity `128x64` layout. The legacy `_reinterpret` lowering planned
  from the raw result layout and ignored the selected source descriptor view, so
  store/load through the same descriptor could pass without proving that the
  physical TMEM image was preserved.
- Validation: `py_compile`, no-op rebuild, and `git diff --check` passed. The
  combined positive/negative runtime-view nodeids passed all `11` selected cases
  across four GPU split groups (`3`, `3`, `3`, `2`).


## 2026-04-13 13:35 UTC: explicit `ld.red` variants cover modifiers

- Extended `tmem_ld_red_explicit_layout_kernel` to forward `abs` and
  `propagate_nan` into explicit-layout `tmem.load_min` / `tmem.load_max`.
- The identity and compatible non-identity explicit-variant tests now cover all
  `LD_RED_MODIFIER_CASES` for both `min` and `max` while keeping the same four
  explicit load variants: `auto`, `32x32b`, `16x32bx2`, and `32x32b_splitn`.
- NaN-propagating cases inject NaNs into representative rows and validate both
  the full tensor output and the reduced result against PyTorch with
  `equal_nan=True`.
- The bounded discovery conclusion is unchanged but stronger: these explicit
  variants still canonicalize to `tcgen05.ld.red.sync.aligned.32x32b.x128`
  rather than revealing another accepted reduction atom, including when the
  `.abs` and `.NaN` opcode modifiers are present.
- Validation: `python3 -m py_compile python/test/gluon/test_tmem_runtime_matrix.py`,
  no-op `make`, and four-GPU split pytest for
  `ld_red_explicit_compatible or ld_red_explicit_n_sharded` passed `163` selected
  cases aggregate (`41`, `41`, `41`, `40` by group).

## Latest: 2026-04-13 scales `warpx2` direct-source-offset scan

- The direct-PTX scales-copy probe now supports arbitrary single-message opcode,
  source-offset, and TMEM-destination-delta variants with one child process per
  run.
- Updated 2026-04-14 evidence: the canonical `warpx4` high-half message uses
  source offset 32 to copy logical columns `8..15`, but the same source window
  faults for both `warpx2::01_23` and `warpx2::02_13`. The tracked unprimed
  single-message scan now covers every `sourceOffsetB128` value `0..136` at
  `dst_delta=4`, with `274` records and zero matches. Offsets `0..8` execute
  but copy row-shifted low physical columns / otherwise wrong data; offsets
  `9..136` fault. Compact result:
  `experiments/results/probe_cp_scales_warpx2_offsets_dst4_current_summary.json`.
- Current conclusion: do not implement scales `warpx2` by swapping the canonical
  `warpx4` messages. Support needs a different shared descriptor/view/staging
  model, or this remains a clean unsupported boundary.

## Latest: 2026-04-13 ldst duration-cache refresh

- The `ldst` duration cache now reflects the lifted roundtrip pre-skip change: the `440` known pre-execution skips have near-zero scheduling weight instead of stale compile-until-OOR durations.
- Full retained validation: `ldst` runner passed `1201` and skipped `441`, with group times `208.7s..263.8s`.
- Runtime `variant_id` selector kernels were tried and reverted. They are functionally possible with `do_not_specialize`, but they make the real xdist matrix slower by compiling larger branchy kernels per layout.

## Latest: 2026-04-14 selected-shard speed helper and CP 128x128b dtype coverage

- Runner speed workflow: `run_tmem_runtime_matrix_sweep.py` now supports `--groups` for exact pytest-split shard reruns and `--xdist-override BUCKET=N` for controlled local experiments. Selected groups keep the canonical group-to-GPU mapping, so group 5 still uses GPU 0, group 8 still uses GPU 3, etc.; this preserves the stable per-GPU cache behavior that dominates warm rerun speed.
- Measured speed result: selected warm `ldst` groups 5 and 8 passed through the runner in about `11s` wall time each (`77 passed, 27 skipped in 9.57s`; `74 passed, 28 skipped in 9.39s`), versus the documented full-cold duration-cache `ldst` group range of about `208s..264s`. This is a targeted rerun speedup and does not reduce coverage.
- `pytest-xdist -n 8` was tested for warm `ldst` shards and was slower than the retained `-n 4` default, so default inner parallelism is unchanged.
- First post-speed ISA coverage slice: `tcgen05.cp.cta_group::* .128x128b` runtime-matrix coverage now includes both `torch.float32` and `torch.int32` for legacy/linear single-CTA, linear indexed-view, and two-CTA paths while retaining exact PTX/LLIR opcode checks.
- Current validation: `make -j8` no-op success; py-compile and `git diff --check` passed; focused CP slice passed all `17` selected cases across four GPUs; full `cp` runner bucket passed `313` and skipped `5` across four GPUs.


## Document Roles

- `memory.md`
  - durable memory for the initiative
  - source of truth for mission, current decisions, long-term plan, and current
    undated priorities
  - future sessions should update this when the stable plan or current
    priorities change

- `log.md`
  - append-only chronology of checkpoints, bug classification, validation, and
    conclusions
  - do not rewrite old entries except for obvious factual corrections

- `handoff_2026-04-09.md`
  - current active handoff
  - read its tail first for the latest live buckets, disproved experiments, and
    next implementation order
  - keep it updated at meaningful checkpoints and before likely context
    rollover

- `gb200_nvidia_ci_inventory.md`
  - current GB200/NVIDIA CI baseline for this Blackwell devbox phase
  - records the workflow coverage, confirmed red list, and classification of
    each current failure as stale expectation, rewrite-candidate, or real bug
  - update it before broadening beyond exact nodeids or changing the current
    CI-grounded failure story

- `gb200_failure_manifest.md`
  - index for the generated exact current-branch failure lists and the
    merge-base-existing shard-3 subsets
  - use the adjacent `.txt` files when you need exact nodeids rather than
    prose summaries
  - refresh the manifests whenever the current-branch failure surface changes
    materially

- `gb200_branch_recovery_plan.md`
  - prioritized recovery backlog for restoring GB200 CI parity with main
  - separates real branch regressions from preexisting failures, stale
    negatives, and reinterpret-contract rewrite candidates
  - use it to choose the next fix slice once the GB200 census is current

- `fuzz_plan.md`
  - operational saturation/fuzzing playbook for the broader `tcgen05` surface
  - not the immediate source of truth for current bug-fix priority while the
    shared-planner / legacy-unification / reinterpret-decomposition work is
    still active

- `handoff_2026-03-26.md`
  - historical handoff from an earlier phase of the initiative
  - archival provenance, not current status

- `next_agent_prompt_2026-03-26.md`
  - archived takeover prompt from an earlier managed-session workflow
  - archival only

## Current Session Discipline

- Keep `memory.md`, the active handoff, and this README aligned.
- Treat `memory.md` as memory: if the stable understanding of the project or
  the current undated execution priorities changes, update `memory.md` rather
  than leaving that knowledge only in chat context.
- Keep dated historical analyses in `log.md` and the dated handoff files
  instead of rewriting old reports in place.
