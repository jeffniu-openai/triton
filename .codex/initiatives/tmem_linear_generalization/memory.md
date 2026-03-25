# TMEM Linear Generalization

## Goal
- Canonicalize TMEM layouts on `tensor_memory_linear`.
- Normalize legacy TMEM layouts before semantic consumption.
- Generalize TMEM descriptors to the shared memdesc API and semantics.
- Broaden TMEM lowering for linear layouts while keeping MMAv5 and `tmem_copy`
  on explicit hardware-family matchers.

## Current Decisions
- Backward compatibility is by early normalization, not by maintaining dual
  semantic paths.
- `tensor_memory_descriptor.slice(...)` will match
  `shared_memory_descriptor.slice(...)`, including `dim=0` default and
  arbitrary-dimension slicing.
- `immHalfSplitOffset == 0` is a legal `16x32bx2` lowering case; the old PTXAS
  workaround must be removed.
- Development-time PTX and hardware validation is expected for ambiguous
  `tcgen05` behavior, but only targeted regressions land in-tree.
- Plain `ttng.tmem_load`, `ttng.tmem_store`, and source-initialized
  `ttng.tmem_alloc` now defer layout feasibility to relayout/lowering instead
  of rejecting non-default compatible register layouts in the verifier.
- `ttg.memdesc_subslice` on TMEM preserves the canonical physical TMEM
  encoding; it is a pointer/view operation over the same backing allocation.
- `ttng.tmem_subslice` is treated as compatibility sugar over the same physical
  TMEM allocation and no longer re-encodes the result shape.
- `ttg.memdesc_index` remains permissive for the extra leading TMEM
  multibuffer dimension; an alloc-shape equality requirement there was tried
  and reverted because it broke the established TMEM multibuffer view model.
- Buffer-region accounting for TMEM subviews follows physical TMEM allocation
  size, not only the smaller logical result shape.
- Relayout now prefers direct TMEM ld/st-compatible layouts and only inserts a
  single `convert_layout` fallback when direct lowering is not possible.
- A non-empty TMEM `block` basis implies multi-CTA CGA semantics regardless of
  `twoCTAs`; `twoCTAs` is extra MMAv5/protocol intent metadata, not the sole
  source of CTA tiling.
- Unsupported TMEM/MMAv5 combinations must fail with diagnostics, never with
  parser or verifier crashes; legacy-layout matching paths need defensive
  handling for impossible candidate families.
- TMEM block-bases descriptor views that eventually reinterpret to a
  `64x32` TMEM-linear memdesc currently have no TMEM-compatible register
  layout for plain `tmem_load/store`; they are a tracked unsupported frontier
  and must fail with a clean relayout diagnostic instead of asserting.
- Higher-rank TMEM descriptor compositions that repartition a tile into
  broadcasted subviews currently fail at `get_reg_layout(...)` with a clean
  diagnostic (`TMEM layout '<atom>' unsupported ... reshape or permute so TMEM
  columns stay contiguous ...`) rather than reaching a pass-time assert or
  late lowering failure.
- TMEM load-reduction on canonical linear layouts is only legal when the
  N dimension is not sharded across threads. Canonical identity TMEM-linear
  layouts satisfy this and execute correctly; the current mixed TMEM-linear
  family does not and should fail with the dedicated
  `tmem_load reduction with N dimension sharded across threads is not supported`
  diagnostic.
- The runtime sweep must distinguish between:
  - general TMEM ld/st-legal linear layouts, which may be far broader than
    MMAv5/copy hardware families;
  - MMAv5-legal canonical TMEM-linear families, which are exactly the
    TMEM-linear layouts equivalent to supported legacy
    `#ttng.tensor_memory_encoding` tiles; and
  - `tcgen05.cp`-legal TMEM destinations, which are constrained by the finite
    `cp` atom family and shared-memory multicast patterns.
- A positive MMAv5 runtime test must use the canonical TMEM-linear equivalent
  of a supported legacy MMAv5 accumulator/LHS tile. A generic block-basis
  TMEM-linear layout that is fine for ld/st is not necessarily MMAv5-legal.

## Runtime Fuzz Strategy

- The exact execution matrix, opcode inventory, GPU sharding, and per-family
  fuzz axes now live in `fuzz_plan.md`. This file keeps the durable policy and
  current-state summary; `fuzz_plan.md` is the operational playbook.

### Outcome classes
- `PASS`: kernel compiles, executes on GPU, numerics are correct, and PTX/LLIR
  contain the expected `tcgen05` opcode family and offsets/immediates.
- `CLEAN_UNSUPPORTED`: compile fails with a diagnostic that explains why the
  layout/op is impossible to lower with the ISA and suggests a viable fix
  (`get_reg_layout`, `convert_layout`, reshape/permute so columns stay
  contiguous, or use a supported MMAv5/copy tile).
- `BUG`: parser/verifier/pass/lowering crashes, malformed IR cannot be printed,
  valid IR crashes a pass, or GPU output/PTX mismatches expectations.

### Shared test oracles
- Numerical correctness on real Blackwell hardware is mandatory for positive
  cases.
- The extracted `tcgen05` opcode stream from PTX and LLIR must match exactly.
- For generic descriptor compositions, TTGIR must contain the generic memdesc
  view ops (`memdesc_subslice/index/trans/reshape/reinterpret`) rather than a
  bespoke TMEM-only path, except where `ttng.tmem_subslice` is explicitly
  being regression-tested as compatibility sugar.
- Impossible cases must never assert/crash and must emit actionable notes.

### `tcgen05.ld/st` fuzz plan
- Sweep TMEM-linear families by legality class:
  - canonical identity (`128x{64,128,256}`);
  - canonical mixed/interleaved row/col (`128x{64,128,256}`);
  - `block`-basis general layouts for `256x{64,128,256}`;
  - `block + two_ctas` general layouts for `256x{64,128,256}`;
  - canonical MMAv5-like two-CTA layouts derived from
    `ttgl.to_linear_layout(TensorMemoryLayout(...), shape)` for the narrower
    MMAv5 overlap cases;
  - `blockM=64`/split-N families that must lower through `16x32bx2`.
- For each legal family, sweep instruction selection:
  - `auto`
  - `32x32b`
  - `16x64b`
  - `16x128b`
  - `16x256b`
  - `32x32b_splitn` where applicable
- For each case, sweep descriptor compositions over legal views:
  - multibuffer `slice(..., dim=0)` then `index`
  - `slice(..., dim=0/1/2)` on reshaped 3D views
  - paired `permute`/inverse-`permute`
  - `reshape` chains that preserve element count
  - `_reinterpret` back to a legal TMEM-linear layout
- Functional oracle:
  - round-trip equality for plain load/store
  - exact offset/immediate checks for `16x32bx2`
  - exact atom family checks for all other atoms
- Negative frontier:
  - layouts with non-contiguous TMEM columns or no supported TMEM register
    layout should fail with the relayout diagnostic, not assert.

### `tcgen05.ld.red` fuzz plan
- Sweep `red_op in {min, max}`.
- Sweep modifiers:
  - `abs in {false, true}`
  - `propagate_nan in {none, all}` where legal
- Sweep shapes/warp partitions that are theoretically reducible:
  - `M in {128, 256}`
  - `N in {32, 64, 128, 256}`
  - `num_warps` chosen so the N dimension is not sharded across threads when
    hardware reduction is required.
- Functional oracle:
  - TMEM round-trip path still returns the original tensor
  - reduced output matches the corresponding PyTorch reduction with correct
    NaN/abs semantics
  - PTX/LLIR use `tcgen05.ld.red.sync.aligned.32x32b.x*.{min|max}[.abs][.NaN].f32`
- Negative frontier:
  - unpacked formats, integer types, or N-sharded register layouts must fail
    with the dedicated reduction diagnostic.

### `tcgen05.cp` fuzz plan
- Enumerate the currently implemented atom families:
  - `cta_group::1.128x256b`
  - `cta_group::1.128x128b`
  - `cta_group::{1,2}.warpx2::02_13.64x128b` if supported in lowering
  - `cta_group::{1,2}.warpx2::01_23.64x128b` if supported in lowering
  - `cta_group::{1,2}.warpx4.32x128b`
- For no-scales copies, sweep `M in {128, 256}`, `N in {16,32,64,128,256}`,
  `BLOCK_N` divisors, and relevant swizzles, skipping only true SMEM
  out-of-resource cases.
- For scales and multicast variants, synthesize exact shared layouts that drive
  the documented multicast bits and alias the destination back through a legal
  TMEM view to validate the payload.
- Functional oracle:
  - copied output equals input (or equals the documented aliasing/permutation
    for scales layouts)
  - PTX/LLIR opcode family and count exactly match the expected atomization
- Negative frontier:
  - unsupported shared multicast/layout combinations must fail cleanly, not
    silently change atom family or crash.

- Progress note:
  - `probe_cp_multicast_layouts.py` and the companion `cp_warpx2_probe_test.py` both exhaust the
    simple interleavings seen above; the current runs on GPU3 keep hitting the
    `tcgen05_copy` legalization error before emitting any `warpx2` opcode.
  - `warpx2_layout_manual.py` tries the same insertion patterns with quieter
    logging but still fails to produce `warpx2`, while `warpx2_layout_linear.py`
    inspects the converted linear layout and shows that the two 32-byte-row
    basis entries never become `[0, 0]`, so multicast stays zero.
  - Need to broaden the layout search (more column strides, extra block
    reorderings, two-CTA-specific layouts) until `tcgen05_copy` legalizes one of
    the documented `warpx2` families.
  - 2026-03-25 refresh:
    - Exhaustive surjective one-hot interleavings for direct scales copies
      (`64x16 -> 128x32`) produced only 2 legal layouts, both emitting
      `warpx4.32x128b` (no `warpx2`).
    - Exhaustive one-hot interleavings for no-scales (`128x4 i32`) produced
      only `128x128b` (no `warpx2`).
    - Subslice probes from larger shared allocations (`parent_rows=128/256`)
      yielded no legal `tcgen05_copy` cases in sampled layout sets; when slice
      start was non-zero, frontend constraints (`split offset may not touch the
      tile`) rejected many candidates.
    - Mixed-basis shared-linear mutations (e.g. `[32, k]`) reached lowering but
      still failed descriptor legalization; still no executable `warpx2`.
    - Runtime coverage now includes executable `cta_group::2.warpx4.32x128b`
      in `python/test/gluon/test_tmem_runtime_matrix.py` via the scaled-MMA
      copy path; direct standalone scales-copy kernels still have not produced
      executable `warpx2`.

### `tcgen05.mma` fuzz plan
- Separate sweeps by hardware family:
  - `cta_group::1`
  - `cta_group::2`
- Sweep accumulator layouts:
  - legacy `TensorMemoryLayout`
  - canonical TMEM-linear equivalents of those legacy layouts
  - TMEM-general but non-MMAv5 layouts as negative cases
- Sweep accumulator usage:
  - `use_acc in {false, true}`
  - multicast/commit path where applicable
- Sweep supported tile families:
  - canonical `blockM=128`
  - canonical `blockM=64` split-N/interleaved families where legal
  - LHS-in-shared and LHS-in-TMEM where supported
- Functional oracle:
  - result matches `torch.matmul` (plus accumulator add when `use_acc=true`)
  - PTX/LLIR contain the exact `tcgen05.mma.cta_group::{1,2}.kind::*` opcode
    family and expected commit/barrier opcodes
- Negative frontier:
  - mixed/block-basis TMEM-general layouts that are not legacy-equivalent must
    fail with the MMAv5 compatibility diagnostic.

### `tcgen05.mma_scaled` fuzz plan
- Sweep format combinations already supported in tree (`mxfp8`, `mxf4`,
  `mxf4nvf4`, etc.) and confirm the emitted `scale_vec::{1X,2X,4X}` suffix.
- Sweep `cta_group::{1,2}` where supported.
- Sweep accumulator/scales layouts:
  - legacy encodings
  - canonical TMEM-linear equivalents
  - two-CTA canonical families
- Functional oracle:
  - numerical output matches the reference within the established tolerance
  - PTX/LLIR `tcgen05.mma...block_scale.scale_vec::*` opcode streams match
  - associated `cp`/barrier ops appear as expected for the chosen pipeline
- Negative frontier:
  - unsupported TMEM-linear accumulator families, scale layouts, or CGA
    mismatches must fail cleanly with MMAv5/scaled-MMA diagnostics.

## Work Split
- Main rollout owns core attr/type/helper/lowering integration.
- Subagents may be used for bounded exploration, PTX validation snippets, and
  test drafting. Claims belong in `agents/claims.md`.

## Next Execution Plan
- The detailed per-instruction fuzz plan lives in
  `experiments/tmem_fuzz_plan.md`.
- Immediate priority order:
  - close the clean-diagnostic vs. runtime-pass boundary for higher-rank
    broadcasted TMEM views;
  - broaden positive runtime coverage until every currently documented and
    backend-reachable TMEM opcode family is seen in both PTX and LLIR on real
    hardware;
  - then use the fuzz results to decide which remaining clean failures are true
    ISA limits and which are lowering gaps that should be implemented.

## Active Fuzz Backlog
- Convert the current higher-rank TMEM `ld/st` broadcasted-subview failure into
  a permanent clean-diagnostic regression, then expand positive higher-rank
  runtime coverage only on compositions with proven legal TMEM register layouts.
- Status update (2026-03-25):
  - positive higher-rank descriptor compositions are now covered for
    `identity` (1-CTA) and `block_two_ctas` (2-CTA) over
    `N in {64,128,256}` and variants
    `{32x32b,16x64b,16x128b,16x256b}` with GPU execution and PTX/LLIR checks.
  - MMAv5 two-CTA higher-rank compositions are now locked as clean negatives
    (`failed to infer tensor memory encoding for memdesc_index`).
- Ensure every currently emitted TMEM instruction family has:
  - at least one passing runtime test with exact PTX/LLIR opcode checks;
  - at least one lit lowering test that validates the generated LLVMIR; and
  - at least one clean-negative regression if the family has known impossible
    frontiers.
- Treat undocumented or ambiguous documented families, especially `cp.warpx2`,
  as direct-PTX probe targets until the backend/legalizer boundary is clear.

## Current Higher-Rank TMEM Status
- TMEM view inference is no longer on the crash path for
  `reshape -> slice -> index` compositions.
  - `memdesc_subslice`, `memdesc_index`, and TMEM reshape inference now rebuild
    view-local TMEM-linear encodings rather than reusing the parent encoding.
  - Builder-side TMEM `memdesc_subslice` inference mirrors the IR-side logic so
    Gluon no longer needs to guess the result type for TMEM views.
- Shared-memory descriptor semantics were preserved after the TMEM view work.
  - The temporary regression from narrowing `alloc_shape` for all memdescs has
    been fixed; only TMEM uses view-local `alloc_shape`.
- One-CTA higher-rank TMEM positives remain semantically suspect.
  - The currently executable compositions:
    - `reshape((2, M, N/2)).slice(dim=0).index(0)`
    - `reshape((2, M/2, N)).slice(dim=0).index(0)`
    still update the entire tensor in runtime probes instead of a strict
    subview.
  - Treat these as active probe cases, not fully trusted semantic coverage,
    until the offset/layout semantics are nailed down empirically.
- Two-CTA MMAv5 higher-rank compositions are currently expected clean
  negatives.
  - They diagnose invalid layout / CTA-group mismatch during parsing.
  - The diagnostic quality still needs improvement because the frontend wraps
    it as a generic parse `RuntimeError`.
