# TMEM Codegen Fuzz Plan

## Objective
- Prove that every TMEM composition that is theoretically codegenable by the
  current ISA both compiles and executes correctly on Blackwell.
- Separate three outcomes cleanly:
  - `PASS`: correct runtime result and expected PTX/LLIR opcode stream.
  - `CLEAN_UNSUPPORTED`: impossible-to-lower layout/op rejected with a useful
    diagnostic and actionable notes.
  - `BUG`: parser/verifier/pass crash, malformed IR emission, wrong opcode, or
    wrong runtime result.

## How To Use This File
- `memory.md` is the durable source of truth for mission, current status, and
  immediate implementation priorities.
- `handoff_2026-04-09.md` is the active handoff for the latest live bug
  buckets and debug conclusions.
- This file is the operational saturation/fuzzing playbook for the broader
  `tcgen05` surface once the current planner/cleanup blockers are green.
- Do not treat the broad coverage items here as a reason to skip the current
  higher-priority planner work in `memory.md`:
  - shared ld/st planner cleanup
  - legacy-vs-linear unification
  - quotient-driven reinterpret packet decomposition
  - rewrite of reinterpret-heavy intent tests to guaranteed descriptor/view
    APIs

## Instruction Inventory

### Allocation / lifetime
- `tcgen05.alloc.cta_group::{1,2}.sync.aligned.shared::cta.b32`
- `tcgen05.relinquish_alloc_permit.cta_group::{1,2}.sync.aligned`
- `tcgen05.dealloc.cta_group::{1,2}.sync.aligned.b32`
- `tcgen05.commit.cta_group::{1,2}.mbarrier::arrive::one.shared::cluster.b64`
- `tcgen05.commit.cta_group::{1,2}.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64`
- `tcgen05.wait::{load,store}`

### Load / store
- `tcgen05.st.sync.aligned.32x32b.x<num>.b32`
- `tcgen05.ld.sync.aligned.32x32b.x<num>.b32`
- `tcgen05.st.sync.aligned.16x64b.x<num>.b32`
- `tcgen05.ld.sync.aligned.16x64b.x<num>.b32`
- `tcgen05.st.sync.aligned.16x128b.x<num>.b32`
- `tcgen05.ld.sync.aligned.16x128b.x<num>.b32`
- `tcgen05.st.sync.aligned.16x256b.x<num>.b32`
- `tcgen05.ld.sync.aligned.16x256b.x<num>.b32`
- `tcgen05.st.sync.aligned.16x32bx2.x<num>.b32`
- `tcgen05.ld.sync.aligned.16x32bx2.x<num>.b32`
- Packed / unpacked subword variants are reached through the same shapes with
  `.pack::16b` and `.unpack::16b`.

### Load reduction
- `tcgen05.ld.red.sync.aligned.32x32b.x<num>.min.f32`
- `tcgen05.ld.red.sync.aligned.32x32b.x<num>.max.f32`
- `tcgen05.ld.red.sync.aligned.32x32b.x<num>.min.abs.f32`
- `tcgen05.ld.red.sync.aligned.32x32b.x<num>.max.NaN.f32`
- `tcgen05.ld.red.sync.aligned.32x32b.x<num>.max.abs.NaN.f32`
- Current lowering/tests show reduction is materially exercised only through
  `32x32b`. The fuzz plan treats any other emitted `ld.red` atom as a discovery
  that must be runtime-validated and then locked in with tests.

### Copy
- `tcgen05.cp.cta_group::{1,2}.128x256b`
- `tcgen05.cp.cta_group::{1,2}.128x128b`
- `tcgen05.cp.cta_group::{1,2}.warpx2::02_13.64x128b`
- `tcgen05.cp.cta_group::{1,2}.warpx2::01_23.64x128b`
- `tcgen05.cp.cta_group::{1,2}.warpx4.32x128b`
- `.4x256b` remains explicitly NYI in the backend and is not treated as a pass
  target until lowering exists.

### MMA
- `tcgen05.mma.cta_group::{1,2}.kind::f16`
- `tcgen05.mma.cta_group::{1,2}.kind::tf32`
- `tcgen05.mma.cta_group::{1,2}.kind::f8f6f4`
- `tcgen05.mma.cta_group::{1,2}.kind::i8`

### Scaled MMA
- `tcgen05.mma.cta_group::{1,2}.kind::mxf8f6f4.block_scale.scale_vec::1X`
- `tcgen05.mma.cta_group::{1,2}.kind::mxf4.block_scale.scale_vec::2X`
- `tcgen05.mma.cta_group::{1,2}.kind::mxf4nvf4.block_scale.scale_vec::4X`

## Common Harness Design

### Case record
Every fuzz case records:
- `family`: `alloc`, `ldst`, `ldred`, `cp`, `mma`, `mma_scaled`.
- `shape`: logical tensor shape and alloc shape if different.
- `dtype`: tensor element type plus scale types where applicable.
- `layout`: canonical TMEM-linear bases and any legacy source layout used as a
  compatibility input.
- `view_chain`: sequence of `slice/index/permute/reshape/_reinterpret` applied
  before the final TMEM op.
- `reg_layout`: explicit register layout or `auto`.
- `num_warps`, `num_ctas`.
- `expected_outcome`: `PASS` or `CLEAN_UNSUPPORTED`.
- `expected_opcode_prefixes` and exact offset/immediate expectations when known.

### Compile pipeline
1. Generate the case from Python using the Gluon surface whenever possible.
2. Compile and capture `ttgir`, `llir`, and `ptx`.
3. Parse opcodes from both PTX and LLIR and require exact equality.
4. Only if compile succeeds and the expected outcome is `PASS`, execute on GPU
   and compare numerics.
5. On compile failure, require a known diagnostic substring for
   `CLEAN_UNSUPPORTED`.
6. Any crash, malformed IR, or mismatch between PTX and LLIR is a `BUG`.

### Shared legality filters
- Only generate shapes with power-of-two TMEM row/column bases unless the
  backend already materializes the family through a canonical layout helper.
- For positive `ld/st`, require `computeTMemLdStEncodingInfo(...)` to succeed
  or `get_reg_layout(...)` to return a layout; otherwise the case is negative.
- For positive `mma` / `mma_scaled`, require the TMEM accumulator layout to be
  equivalent to a supported legacy MMAv5 family.
- For positive `cp`, require the shared-to-TMEM map to match one of the finite
  copy atoms. Any generated case that is merely surjective but not atomizable is
  negative.
- For higher-rank descriptor chains, positive cases must preserve contiguous
  TMEM columns at the final access; chains that produce the known broadcasted
  unsupported frontier stay negative until lowering is extended.

### Shared runtime oracles
- Numerics must match PyTorch/reference bit-exactly for integer and exact
  float32 round-trips, or within the existing tolerance for MMA/scaled MMA.
- TTGIR for descriptor tests must contain generic memdesc ops
  (`memdesc_subslice`, `memdesc_index`, `memdesc_trans`, `memdesc_reshape`,
  `memdesc_reinterpret`) rather than bespoke TMEM-only lowering, except when
  explicitly testing `ttng.tmem_subslice` compatibility sugar.
- PTX and LLIR must contain the same opcode multiset in the same order.
- Invalid or unsupported cases must fail without parser/verifier/pass crashes.

## Family-Specific Fuzz Matrices

### 1. `alloc` / `relinquish_alloc_permit` / `dealloc` / `commit` / `wait`

#### Positive matrix
- `cta_group in {1, 2}`
- `size in {32, 64, 96, 128, 192, 256, 384, 512}`
- `alloc_source in {none, initialized_from_registers}`
- `commit_mode in {none, barrier, barrier_multicast}`
- `two_ctas in {false, true}`

#### Checks
- LLIR contains exactly one `alloc`, one `relinquish_alloc_permit`, and one
  `dealloc` for each kernel.
- `alloc` size immediate matches the module TMEM size.
- `ttng.tmem_alloc` with source initialization round-trips the initialized
  values through a load.
- `wait <store>` is present after source-initialized alloc/store paths.
- `wait <load>` is present after `tmem_load` and before any redval use.
- For 2 CTA kernels, cluster sync is present before dealloc and `commit` uses
  `cta_group::2`.
- Current runtime anchors cover single-CTA and two-CTA ld/st allocation
  lifetimes, exact PTX/LLIR alloc/relinquish/dealloc/wait emission, pow2
  alloc/dealloc size immediates `32, 64, 128, 256, 512`, two-CTA cluster sync
  before dealloc, one source-initialized allocation roundtrip, exact
  commit-opcode anchors for single-CTA non-multicast plus two-CTA multicast
  copy/MMA paths including scaled-MMA copy-helper kernels, and allocation-pass
  rounding boundaries for raw live totals
  `96`, `192`, and `384` to supported module sizes `128`, `256`, and `512`.
  Remaining allocator fuzzing is specialized standalone/malformed commit
  configurations beyond these anchors.

#### Negative matrix
- `size > 512`
- 2 CTA kernels with incompatible barrier/commit configuration
- malformed or mismatched alloc result types

### 2. `ld` / `st`

#### Positive TMEM layout families
- canonical identity:
  - `M in {64, 128, 256}`
  - `N in {32, 64, 128, 256}`
  - current root identity `128x32` coverage includes direct and descriptor-chain
    f32 and i32 roundtrips over every public `ld/st` variant, pinning the
    minimal `x32/x16/x8/x4` opcode families.
  - current broad single-CTA `128x{64,128,256}` coverage includes direct and
    descriptor-chain i32 roundtrips for identity and mixed TMEM-linear layouts
    over every public `ld/st` variant, complementing the existing broad f32
    direct/descriptor tests.
- diagonal row/column permutations:
  - current root `128x32` coverage includes `rotate1`, `even_odd`, and
    `reverse` permutations for direct and descriptor-chain f32 and i32
    roundtrips over every public `ld/st` variant.
  - current broad `128x{64,128,256}` diagonal coverage includes identity plus
    `rotate1`, `even_odd`, and `reverse` layouts for direct and
    descriptor-chain f32 and i32 roundtrips over every public `ld/st`
    variant.
- representative non-diagonal row/column permutations:
  - current root `128x32` coverage includes pure row (`rotate1,identity`),
    pure column (`identity,reverse`), and mixed row+column (`even_odd,reverse`)
    layouts for direct and descriptor-chain f32 and i32 roundtrips over every
    public `ld/st` variant.
  - current broad `128x{64,128,256}` coverage includes the full f32 row/column
    cross-product plus i32 for every non-identity row/column layout for direct
    and descriptor-chain roundtrips over every public `ld/st` variant.
    Identity/identity i32 is covered by the broad identity/mixed i32 tests
    rather than duplicated in this matrix.
- scrambled/exotic linear families:
  - current root `128x32` coverage includes `scrambled_cols` and
    `scrambled_rows_cols` layouts for direct and descriptor-chain f32 and i32
    roundtrips over every public `ld/st` variant.
  - current broad `128x{64,128,256}` coverage includes `scrambled_cols` and
    `scrambled_rows_cols` layouts for direct and descriptor-chain f32 and i32
    roundtrips over every public `ld/st` variant.
- canonical mixed/interleaved linear families:
  - current `128 x {64, 128, 256}` mixed family
  - additional mixed bases discovered by compile-only sweep over row/col basis
    permutations that preserve surjectivity and contiguous TMEM columns
- block-basis linear families:
  - `block_bases = [[1, 0]]`
  - `shape = {128, 256} x {64, 128, 256}`
- 2 CTA families:
  - generic block-basis `two_ctas=true`
  - canonical MMAv5-equivalent 2 CTA families from
    `ttgl.to_linear_layout(TensorMemoryLayout(...))`
  - current root `256x32` coverage includes both `block_two_ctas` and
    `mmav5_twocta` layouts for direct and descriptor-chain f32 and i32
    roundtrips over every public `ld/st` variant.
  - current broad two-CTA `256x{64,128,256}` coverage includes direct and
    descriptor-chain i32 roundtrips for `block_two_ctas` and MMAv5-like
    layouts over every public `ld/st` variant, complementing the existing broad
    f32 direct/descriptor tests.
- m64 / split-N families:
  - `64 x {2, 4, 8, 16, 32, 64, 128}`
- subword families:
  - `dtype in {f16, bf16, i16, i8}`
  - packed, unpacked, and padded one-column cases
  - current direct coverage now includes `f16`, `bf16`, `i16`, and `i8` for
    identity `128x{64,128,256}` across all public variants
  - current x1 coverage includes `f16`, `bf16`, and `i16` packed/unpacked
    cases plus packed and padded `i8` x1 cases for `auto` and `32x32b`; direct
    and descriptor-chain two-CTA x1 subword coverage now includes `f16`,
    `bf16`, `i16`, and packed `i8` over canonical two-CTA TMEM-linear layouts

#### Positive view chains
- plain TMEM load/store
- multibuffer:
  - `allocate_tensor_memory(..., [B, M, N], ...)`
  - `slice(start, 1, dim=0).index(0)`
- arbitrary-dimension slicing:
  - reshape to rank 3 or 4
  - slice each dimension independently
  - permute back to a legal final view
- paired `permute` / inverse `permute`
- reshape chains that preserve element count
- `_reinterpret` back into an equivalent TMEM-linear layout
- mixed view chains combining all of the above

#### Instruction selection sweep
- `instr_variant in {auto, 32x32b, 16x64b, 16x128b, 16x256b}`
- explicit split-N path:
  - `instr_variant = 32x32b_splitn`

#### Checks
- Runtime round-trip equality.
- Exact PTX/LLIR opcode equality.
- Exact `secondHalfOffset` immediate checks for `16x32bx2`.
- For descriptor tests, TTGIR must show generic memdesc views.
- For direct-lowering tests, PTX should contain the selected atom family with no
  unexpected atom substitution.
- For explicit fallback tests, TTGIR should show the inserted
  `convert_layout` exactly where expected.

#### Negative frontier
- layouts with no supported TMEM register layout
- descriptor chains that break TMEM column contiguity
- known broadcasted higher-rank TMEM subviews until lowering is extended
- direct higher-rank TMEM `ld/st` access: current contract requires callers to
  index/slice/reshape higher-rank descriptors to a 2D view before direct
  `get_reg_layout`, `load`, `load_red`, or `store`; non-2D direct access must
  stay a clean diagnostic until a true direct higher-rank lowering is designed
- any case where CGA block bases differ between memdesc and register layouts

### 3. `ld.red`

#### Positive matrix
- `redOp in {min, max}`
- `abs in {false, true}`
- `NaN in {false, true}` with legal `f32` cases
- `M in {128, 256}`
- `N in {32, 64, 128, 256}`
- positive layouts where the N dimension is not sharded across threads:
  - canonical identity for the 128-row reduction block
  - row-256 reduction-friendly physical forms where the source layout still
    has a 128-row reduction block and carries the high row bit through the
    column/query frame
  - tile-permuted `128xN` layouts across all currently proven helper widths:
    `N=32/tile_n=8`, `N=64/tile_n in {8,16}`,
    `N=128/tile_n in {8,16,32}`, and
    `N=256/tile_n in {8,16,32,64}`
  - pure column-permuted, pure row-permuted, and non-identity row/column
    cross-product `128x{64,128,256}` layouts, which are now proven to emit
    `32x32b` reduction-family opcodes across legal modifiers
  - any additional TMEM-linear family that compile-only search proves emits
    legal reduction code

#### Discovery sweep
- Force compile-only sweeps across all explicit `ld/st` register variants and
  supported TMEM-linear families to determine whether any `ld.red` path emits
  atoms other than `32x32b`.
- If a non-`32x32b` reduction atom appears and PTXAS accepts it, add runtime
  coverage immediately.
- If compile always canonicalizes to `32x32b`, lock that in as expected
  behavior with tests.

#### Checks
- The tensor result still matches the loaded tensor.
- The reduction result matches the PyTorch reduction with correct `abs` and
  `NaN` semantics.
- PTX and LLIR use the same `tcgen05.ld.red` opcode stream.
- Current positives pin exact offset immediates: `[0]` for `N <= 128` and
  `[0, 64, 128, 192]` for `N=256`.
- Explicit compatible register-layout variants cover both min and max plus all
  legal `abs` / `NaN` modifier combinations for `auto`, `32x32b`,
  `16x32bx2`, and `32x32b_splitn` on the identity layout.
- Explicit compatible register-layout variants also cover non-identity
  compatible TMEM-linear source families (`tile_permuted`, `col_reverse`,
  `row_reverse`, and `rowcol_rotate_reverse`) across the same operation and
  modifier matrix, proving they still canonicalize to
  `tcgen05.ld.red.sync.aligned.32x32b.x128` rather than discovering a new
  reduction atom.
- Positive runtime-matrix `ld.red` tests assert exactly one `wait <store>`
  before reduction loads, exactly one `wait <load>` after `ld.red`, and the
  PTX/LLIR ordering `store -> wait.store -> ld.red -> wait.load`.

#### Negative frontier / covered clean negatives
- Covered by `test_tmem_runtime_matrix_ld_red_non_f32_contract_reports_clean_unsupported`:
  `i32` plain reductions, `i32` reductions with `NaN` or `abs` modifiers, and
  legacy-unpacked `f16` reduction attempts all fail before lowering with clean
  verifier diagnostics.
- Additional N-sharded register layouts beyond the explicit `16x64b`,
  `16x128b`, and `16x256b` clean-negative coverage remain a watch item if new
  explicit variants become reachable.
- M64 and block-basis TMEM-linear source layouts are now covered as clean
  unsupported runtime-matrix cases across the legal modifier matrix; keep them
  on the negative frontier unless a future planner grows a real direct
  reduction lowering.
- Plain identity `256xN` source layouts are already pinned as not the current
  direct `tcgen05.ld.red` source form and should keep the clean
  software-reduction diagnostic unless the planner grows a real positive
  lowering.
- If compile-only discovery reaches a true f32 unpacked/reinterpret descriptor
  view for reduction, it must either fail with the packed-format diagnostic or
  be promoted only after a proven legal lowering exists.

### 4. `cp`

#### No-scales positive matrix
- `M in {128, 256}`
- `N in {16, 32, 64, 128, 256}`
- `BLOCK_N` divisors of `N`
- `swizzle in {32, 64, 128}`
- TMEM destinations:
  - legacy `TensorMemoryLayout`
  - canonical TMEM-linear equivalent
- Dense `cta_group::2` no-scales coverage now includes exact `128x128b` and
  `128x256b` opcode checks; keep future dense two-CTA additions focused on
  new shapes/layouts rather than re-proving those two base families.
- Special no-scales `cta_group::1` `warpx2::{01_23,02_13}.64x128b` paths are
  covered by executable candidate tests with runtime oracles and exact commit
  opcode checks. Dense shared-layout `warpx2` forms are clean negatives until a
  correct descriptor/address model exists; they previously emitted opcodes but
  copied wrong data.
- No-scales `cta_group::2` `warpx2::01_23.64x128b` is covered by an
  executable public-layout test with exact copy and multicast commit opcodes.
  `cta_group::2 warpx2::02_13` remains a layout-surface frontier: the
  canonical candidate shared layout is pinned as a clean descriptor-plan
  unsupported case, and the dense shared-layout form is rejected to avoid
  known wrong-code. A direct-seed sweep found that the aligned address variant
  can move data but loses the source-column bit, while the old `+4` TMEM dword
  delta is all-zero and unaligned deltas trap. A 2026-04-13 direct-seed
  follow-up around `sourceOffsetB128` `32..35`, destination deltas `0/4`, and
  two-message column-pair schedules still duplicated one source column pair or
  overwrote with another duplicate, so this remains a descriptor/message
  semantics frontier rather than a small offset toggle.

#### Scales / multicast positive matrix
- scales payload layouts that are known to alias to legal TMEM scales tiles
- multicast shapes:
  - `.128x256b`
  - `.128x128b`
  - `.warpx4.32x128b`
  - `.warpx2::{02_13,01_23}.64x128b` once legalization finds executable cases
- `cta_group in {1, 2}`
- Scaled-MMA copy-helper coverage through `tcgen05.copy.warpx4.32x128b` now
  spans all current scaled format pairs at `blockN in {128, 256}` and
  `blockK in {128, 256}`, `num_ctas in {1, 2}`, `multicast in {false, true}`,
  and legacy/canonical accumulator layouts; the expected scaled-copy message
  count is `(1 + blockN // 128) * (blockK // 128) * (32 // vec_size)`. The
  companion scaled-MMA instruction count is pinned as
  `base_count * (blockK // 128)`, and only two-CTA multicast cases should carry
  the TTGIR `{multicast}` marker.
- Do not use the historical public-layout "warpx2 candidate" as proof of a
  scales `warpx2` path: it currently classifies as `warpx4.32x128b` under
  `TensorMemoryScalesLayout` because the scales layout carries broadcast row
  bases. `CP_SCALES_LAYOUT_PROBE_CASES` now also pins four nearby basis-order
  variants as clean unsupported, and all of them still classify as `warpx4`
  before descriptor-plan synthesis fails. A refreshed 384-layout parent-row
  subslice probe also found only clean unsupported verifier boundaries and no
  `warpx2` opcodes. Unsupported scales copy layouts now report this boundary
  from `TMEMCopyOp` verification instead of late LLVM lowering. Treat true
  scales `warpx2` as a descriptor/address-representation or direct-PTX
  documentation frontier. The current unprimed direct-PTX high-half source
  offset sweep covers `sourceOffsetB128=0..136` for
  `warpx2::{01_23,02_13}` at `dst_delta=4` with zero matches:
  offsets `0..8` execute with wrong data, and offsets `9..136` fault. Do not
  promote a scales `warpx2` alias unless a different descriptor/view/staging
  model covers the whole scale tile without priming from canonical `warpx4`.

#### Shared-layout search strategy
- Generate shared linear layouts by basis mutation rather than random dense
  matrices:
  - row basis permutations
  - column stride changes
  - one-hot row-to-offset interleavings
  - block-basis reorderings
  - two-CTA CGA variants
- Reject non-surjective or non-invertible shared layouts early.
- Compute the shared-to-TMEM conversion and classify the intended atom using
  the same rules as `getTMemCopyAtom(...)`.
- Only runtime-execute layouts that legalize through the actual copy matcher.

#### Direct PTX fallback for documentation gaps
- When the backend cannot legalize a documented atom family but the layout math
  suggests it should exist, write a direct PTX microkernel to determine whether
  the ISA/toolchain accepts the opcode.
- The no-scales `cta_group::1` `warpx2::{02_13,01_23}.64x128b` candidate
  cases and the no-scales `cta_group::2 warpx2::01_23.64x128b` case are now
  covered through the public backend path with exact commit opcode checks;
  the no-scales `cta_group::2 warpx2::02_13.64x128b` canonical candidate is
  covered as a clean descriptor-plan unsupported boundary. Reserve direct PTX
  probes for remaining scales or two-CTA `warpx2::02_13` documentation/layout
  gaps. Dense shared-layout `warpx2` forms are negative until descriptor
  synthesis can prove correct runtime semantics. For two-CTA `02_13`, do not
  promote a direct-seed path unless it preserves both the row mapping and the
  4-byte source-column bit; the current direct-seed variants either duplicate
  columns, return zeros, or fault. The durable patched-PTX sweep in
  `experiments/probe_cp_warpx2_02_13_twocta_direct_ptx.py` / results file
  `experiments/results/probe_cp_warpx2_02_13_twocta_direct_ptx_current.log`
  extends this to opcode-only, source-row-plus-16, single-CTA seed,
  destination-`+4`, and two-message variants; none match the extended
  single-CTA `02_13` oracle. The current direct-PTX harness uses compile-only
  warmup by default so patched cubins are not primed by a prior canonical
  `01_23` launch. The four-GPU JSONL shard set
  `experiments/results/probe_cp_warpx2_02_13_twocta_source_offsets_*_gpu*.jsonl`
  extends the single-message direct-seed scan through offsets `36..127` with
  destination deltas `0` and `4`; the durable source-offset records have zero
  matches against the extended `02_13` oracle. Offsets `36..72` execute but all
  duplicate one source-column pair, and offsets `73..127` launch-fail in
  isolated child processes. The older column-offset/NaN breakdown is historical
  and superseded by the compile-only rerun.

#### Checks
- Output matches input for no-scales copies.
- Scales paths match the documented aliasing/permutation.
- PTX and LLIR `cp` opcodes match exactly in family and count.
- Barrier/commit emission is present when requested.

#### Negative frontier / covered clean negatives
- Covered by `test_tmem_runtime_matrix_cp_no_scales_transposed_shared_reports_clean_error`:
  no-scales copies from transposed `NVMMASharedLayout` sources fail with the
  clean `The source should not be transposed or padded` diagnostic.
- Covered by `test_tmem_runtime_matrix_cp_no_scales_shared_subslice_bad_offset_reports_clean_error`:
  no-scales copies from shared subviews whose non-zero split start intersects a
  tile fail with the clean `The split offset may not touch the tile` diagnostic.
- Covered by `test_tmem_runtime_matrix_cp_scales_shared_subslice_layout_reports_clean_unsupported`:
  a representative parent-row shared subview from the `warpx2` subslice probe
  maps to `tcgen05.copy.warpx4.32x128b` but fails cleanly when no compatible
  TensorMemoryScales shared descriptor plan can be synthesized.
- Additional multicast/layout combinations that cannot be described by the copy
  atom set remain a watch item.

### 5. `mma`

#### Positive matrix
- `kind in {f16, tf32, bf16, f8e5m2, f8e4m3}` for current positive plain MMA
  runtime coverage; direct `i8` remains a clean frontend/ptxas-negative
  boundary on the current Blackwell target.
- `cta_group in {1, 2}`
- current one-CTA root plain-kind matrix covers `blockN in {64, 128, 256}` and
  `blockK in {32, 64}` for legacy and canonical TMEM-linear accumulator layouts,
  with exact opcode counts scaling by `blockK // 32`.
- current two-CTA plain-kind matrix covers `blockN in {64, 128, 256}` and
  `blockK in {32, 64}` for legacy and canonical TMEM-linear accumulator layouts,
  with exact opcode counts scaling by `blockK // 32`.
- `use_acc in {false, true}`; current 1-CTA and 2-CTA `use_acc=true`
  coverage spans all supported plain kinds, both legacy/canonical accumulator
  layouts, `blockN in {64, 128, 256}`, and `blockK in {32, 64}`.
- `multicast in {false, true}` where supported
- TMA-fed two-CTA TF32 has a positive reachable route when matrix B is supplied
  as a non-transposed `[N, K]` descriptor and passed to MMAv5 through a shared
  `permute((1, 0))` view; current coverage spans `blockN in {64, 128, 256}` for
  both legacy and canonical TMEM-linear accumulator layouts. Default `[K, N]`
  B descriptors remain a clean negative because TMA descriptors cannot be
  transposed.
- accumulator layout family:
  - legacy TMEM layout
  - canonical TMEM-linear equivalent
  - tile-permuted canonical TMEM-linear accumulators for all supported plain
    operand kinds, currently covered at `128x128/tile_n=32` and
    `128x256/tile_n=64`, `blockK in {32, 64}`, including the `use_acc=True`
    accumulator-add path; exact opcode counts scale by `blockK // 32`
- A operand:
  - shared-memory path
  - TMEM path where the op and backend support it
  - full-shape tile-permuted TMEM-LHS coverage now spans `f16`, `bf16`,
    `f8e5m2`, and `f8e4m3` at `N in {128, 256}`, and `tf32` at `N=128`, for
    a `128x256` operand-A tile, with exact PTX/LLIR opcode checks and
    eight-times-root instruction counts. The direct shared-B `tf32, N=256,
    K=256` helper shape exceeds shared memory and is not a positive target.
  - current TMEM-LHS subview coverage spans `f16`, `tf32`, `bf16`, `f8e5m2`,
    and `f8e4m3` at `N in {128, 256}` and `K in {32, 64}` for both
    legacy and canonical TMEM-linear accumulator layouts, with exact PTX/LLIR
    opcode and instruction-count checks
- supported MMAv5 tile families:
  - `blockM in {64, 128}`; current root `blockM=64` runtime coverage spans
    every supported plain operand kind at `N in {64,128,256}`, `K in {32,64}`,
    both legacy/canonical M64 accumulator layouts, and both no-accumulator and
    `use_acc=True` paths.
  - `blockN` and `K` values accepted by the in-tree verifier/matcher

#### Checks
- Result matches the reference matmul, including accumulator add when
  `use_acc=true`.
- PTX and LLIR use the same `tcgen05.mma` opcode family, with exact op counts
  pinned for the plain-kind root matrices, tile-permuted accumulator coverage,
  full-shape tile-permuted TMEM-LHS coverage, and TMEM-LHS subview coverage.
- `commit` opcode family and multicast suffix are correct.
- TMEM accumulator layout must remain canonical TMEM-linear in TTGIR when the
  source came from `TensorMemoryLinearLayout`.

#### Negative frontier
- Direct `i8` MMAv5 remains a clean Blackwell unsupported boundary; runtime
  coverage now includes one-CTA and two-CTA legacy/canonical accumulator
  layouts at `blockN in {64, 128, 256}`.
- non-MMAv5-equivalent TMEM-linear accumulator layouts
- transposed float32 shared operands
  - current live nuance: direct non-TMA 2-CTA TF32 coverage is green, and
    TMA-fed 2-CTA TF32 is positive at `blockN in {64, 128, 256}` when B is loaded
    as `[N, K]` and then shared-permuted into MMAv5's required `[K, N]` operand
    view. The default `[K, N]` B TMA descriptor shape is covered as a clean
    unsupported boundary at the same `blockN in {64, 128, 256}` shapes because TMA
    descriptors cannot themselves be transposed.
- CGA mismatches between descriptors and the module CTA topology

### 6. `mma_scaled`

#### Positive matrix
- `kind in {mxf8f6f4.scale_vec::1X, mxf4.scale_vec::2X, mxf4nvf4.scale_vec::4X}`
- `cta_group in {1, 2}`
- scale factor subtype / position combinations allowed by the verifier
- canonical TMEM-linear accumulator and scale layouts equivalent to supported
  legacy layouts
- direct one-CTA root accumulator coverage now spans every current scaled
  format pair (`mxfp8/mxfp8`, `mxfp4/mxfp4`, `mxfp8/mxfp4`, `mxfp4/mxfp8`,
  and `nvfp4/nvfp4`) at `blockN in {64, 128, 256}` and `blockK in {128, 256}`
  for both legacy and canonical TMEM-linear accumulator layouts, with exact
  scaled-MMAv5 opcode counts pinned as `base_count * (blockK // 128)`.
- accumulator subview starts that preserve the same supported physical family,
  currently including one-CTA root-aligned `slice_start=0` and offset
  `slice_start=64` direct subviews plus two-CTA cga-aware `block_n=128`,
  `parent_n=256`, `slice_start in {0,128}` direct subviews across the proven
  format pairs. The obvious two-CTA `block_n=256`, `parent_n=512` offset-subview
  extension exceeds TMEM capacity (`Required: 524`, limit `512`); a
  `parent_n=384` alternative is not expressible by the current power-of-two
  MMAv5 two-CTA layout helper.
- TMEM-LHS format coverage currently includes the packed-storage reachable
  subset (`mxfp8/mxfp8`, `mxfp8/mxfp4`, `mxfp4/mxfp4`, and `nvfp4/nvfp4`)
  for both legacy and canonical accumulator layouts. Subview operand-A
  descriptors now span `blockN in {64, 128, 256}` and `blockK in {128, 256}` with
  exact scaled-MMAv5 opcode counts pinned as `base_count * (blockK // 128)`.
  Full-shape tile-permuted operand-A descriptors now span `blockN in {64,
  128, 256}` for all reachable packed storage pairs at logical `K=256`, and
  the mxfp8-storage LHS pairs (`mxfp8/mxfp8` and `mxfp8/mxfp4`) at logical
  `K=128`; fp4-storage `K=128` tile-permuted LHS descriptors are pinned over
  the same N range as a clean MMAv5-layout-compatible verifier negative.
  Mixed `mxfp4/mxfp8` dense TMEM-LHS subviews and full-shape
  tile-permuted TMEM-LHS descriptors are clean unsupported boundaries, not
  positive targets yet, because `mxf8f6f4` fp4 LHS currently requires
  `fp4_padded` shared-memory operand-A storage. The subview frontier is pinned
  by `experiments/probe_mma_scaled_lhs_subslice_formats.py`, and the runtime
  matrix now also pins the full-shape tile-permuted clean-negative paths.

#### Checks
- Runtime result matches a dequantized reference within established tolerances.
- PTX and LLIR use the same scaled-MMA opcode family and scale-vector suffix.
- Required `cp` / `commit` companion instructions appear when the pipeline
  needs them.

#### Negative frontier
- unsupported TMEM-linear accumulator or scale layouts
- tile-permuted accumulator subviews that would require repeated `N=32`
  block-scaled MMAv5 instructions; these should keep the clean diagnostic
  explaining that public tensor-memory scales expose matrix-B scale fragments
  only at 64-column alignment
- transpose on `mxf4` / `mxf4nvf4`
- unsupported scale-factor subindices

## Execution Order

### Phase 0: compile-only inventory
- Enumerate all candidate cases and classify them as:
  - reachable pass target
  - expected unsupported
  - backend/documentation ambiguity
- Do this without runtime execution first so the matrix stays bounded.

### Phase 1: ld/st saturation
- Finish the full `ld/st` runtime sweep first because it is the broadest TMEM
  surface and is the prerequisite for descriptor generality.
- Promote supported higher-rank descriptor chains from negative to positive as
  backend gaps are closed.

### Phase 2: reduction
- Saturate `ld.red` after plain `ld/st`, because its legality is a strict
  refinement of the same layout math.

### Phase 3: copy
- First saturate no-scales `128x256b` / `128x128b`.
- Then pursue `warpx4`.
- Then dedicate search/probe time to `warpx2`.

### Phase 4: MMA
- Sweep `f16` first, then `tf32`, then `f8f6f4`, then `i8`.
- Expand 2 CTA families only after the 1 CTA family is saturated for that
  `kind`.

### Phase 5: scaled MMA
- Sweep `mxf8f6f4` first, then `mxf4`, then `mxf4nvf4`.

## GPU Sharding
- `CUDA_VISIBLE_DEVICES=0`: `ld/st` and `ld.red` runtime sweeps.
- `CUDA_VISIBLE_DEVICES=1`: `cp` sweeps and direct PTX copy probes.
- `CUDA_VISIBLE_DEVICES=2`: `mma` and `mma_scaled` runtime sweeps.
- `CUDA_VISIBLE_DEVICES=3`: ambiguity probes, crash repro minimization, and
  exotic descriptor-chain searches.

## Result Logging
- Every sweep writes one JSONL artifact under
  `.codex/initiatives/tmem_linear_generalization/experiments/results/`.
- Each line records:
  - case id
  - opcode family
  - layout summary
  - compile result
  - runtime result
  - diagnostic text if any
  - PTX opcode list
  - LLIR opcode list
- `log.md` receives only the high-signal summary: newly reachable atom
  families, newly confirmed unsupported frontiers, and any bug/crash fixes.

## Immediate Missing Coverage To Tackle First
- Before broad saturation, close the current planner/cleanup blockers recorded
  in `memory.md`:
  - shared TMEM ld/st planner cleanup
  - legacy `block_m_64` support-family unification
  - parent-layout reinterpret packet decomposition
  - reinterpret-heavy test-contract rewrites
- Once those are green, the next missing coverage to tackle is:
  - refreshed TMEM descriptor/view runtime slices after the planner cleanup
  - `ld.red` runtime saturation over all supported modifiers and non-sharded
    layouts
  - `cp` `cta_group::2` coverage and any reachable `warpx2` cases
  - `mma` runtime coverage beyond the already-proven anchor cases
  - `mma_scaled` runtime coverage beyond the current minimal and accumulator
    slice-start anchors
  - specialized standalone/malformed commit configurations beyond the current
    runtime and allocation-pass anchors

## Immediate Next Code Changes
- Near-term code changes should focus on the planner/lowering core, not on
  expanding the broad test matrix first:
  - `lib/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.cpp`
    - shared ld/st planner cleanup
    - reinterpret quotient / packet-decomposition cleanup
    - legacy-vs-linear physical-family unification
  - `lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp`
    - family-selection cleanup as planner responsibilities are centralized
  - `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
    - consume the same structural planner decisions as frontend/verifier
  - `python/src/gluon_ir.cc`
    - keep descriptor/layout selection aligned with the shared planner
  - `lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`
    - verifier consumption of the shared planner/results
- Test changes that should happen alongside that code:
  - `python/test/gluon/test_core.py`
    - keep the current `block_m_64` anchors green
    - rewrite reinterpret-heavy intent tests toward guaranteed
      descriptor/view APIs once the lowering is stable
  - `python/test/gluon/test_tmem_runtime_matrix.py`
    - refresh the nearest TMEM runtime slices after the planner cleanup
    - only then resume broader saturation work
  - `test/Conversion/tritongpu_to_llvm_blackwell.mlir`
    - add or refresh exact opcode checks when the planner cleanup changes the
      stable direct family

## 2026-04-13 Descriptor View/API Migration Note

- One runtime-matrix `ld/st` descriptor-chain use has been migrated from
  `_reinterpret` to supported `.bitcast(...)` where a physical-equivalent view
  bitcast is actually intended.
- The deep `slice/index/reshape/permute` roundtrip case no longer uses
  `_reinterpret`; it requested the same dtype, shape, and layout and should stay
  a pure view-chain test.
- Remaining `_reinterpret` migrations should follow this distinction:
  - use supported `.bitcast(...)` only when the test really needs a
    physical-equivalent descriptor view change;
  - delete no-op same-layout reinterprets;
  - keep genuinely unsupported bitcasts over complex descriptor chains on the
    planner/API backlog instead of adding ad hoc lowering patches.
- `python/test/gluon/test_tmem_runtime_matrix.py` now has no remaining
  `_reinterpret(...)` users. The former block-descriptor negative uses
  supported `.bitcast(...)`; its two-CTA block parameter is a clean
  bitcast/subslice negative because the composed view would require a negative
  additive block delta (`dim=2 step=2 phys=block delta=-1`).
- Frontend TMEM parser tests in `python/test/gluon/test_frontend.py` still use
  raw `_reinterpret(...)` intentionally. A `.bitcast(...)` probe fails before
  IR generation with `unsupported tensor memory memdesc_subslice view`, so these
  remain contract coverage until complex physical-bitcast-over-view-chain support
  exists.
- `python/test/gluon/test_core.py` now has no TMEM `_reinterpret(...)` users.
  The former mixed-basis runtime-view positive is a clean supported-bitcast
  negative because the requested identity `128x64` result layout is not
  physical-mapping equivalent to the selected mixed source descriptor view.
