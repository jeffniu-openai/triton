# Gap #2 Audit: `tcgen05.cp` Complete Support Umbrella

Last updated: 2026-04-18 06:57 UTC

## Scope

Gap #2 owns all remaining `tcgen05.cp` support and coverage questions. It
started as an audit of supported-layout completeness for the public copy
instruction families, but now also owns the formerly separate copy-specific
gaps: partial footprints, `4x256b` refresh views, packed/subword `warpx2`,
two-CTA `warpx2::02_13`, and `tcgen05_copy` source/frontend contracts.

This audit still treats legacy tensor-memory encodings as frontend syntax only:
compiler support must be judged after conversion to normalized `LinearLayout`.
The gap register was later reconsolidated to remove the old copy-specific
placeholder gap numbers entirely; active `tcgen05.cp` work should use Gap #2
sub-buckets.

## Normalization Audit

The `tcgen05.cp` verifier and planner are already structured around linear
layout arithmetic:

- `TMemCopyOp::verify` obtains the shared source layout with
  `toLinearLayout(srcTy)`.
- Destination legality flows through `selectTMemCopyPhysicalQuery`, which
  compares standalone and exact descriptor-view queries represented as
  `TMemPhysicalQuery { layout, origin, twoCTAs, elementBitWidth, isScales }`.
- The destination-to-source relation is computed with
  `getTMemCopySourceConversion(query, shmemLl)`, and copy-family selection is
  derived from that conversion in `getTMemCopyAtom` / `getTMemCopyPlans`.
- Legacy `#ttng.tensor_memory_encoding` canonicalizes to a
  `TensorMemoryLinearEncodingAttr` / `LinearLayout` through
  `tryGetCanonicalTensorMemoryLinearLayout`; explicit
  `#ttng.tensor_memory_linear` supplies the layout directly.

Conclusion: there is no intended `legacy` versus `linear` support distinction
inside the copy planner. Tests may keep a few legacy rows for frontend
compatibility, but Gap #2 coverage should be grouped by normalized
`LinearLayout` equivalence class.

## Recognized Public Families

The current planner recognizes these public `tcgen05.cp` families:

| Family | Recognition condition after layout conversion | Current support state |
| --- | --- | --- |
| `128x128b` | dense row projection, no multicast, 128-bit column footprint | Supported for canonical dense layouts, one-CTA and two-CTA, with representative legacy/linear and indexed-view rows. |
| `128x256b` | dense row projection, no multicast, >=256-bit column footprint | Supported for canonical dense layouts, one-CTA and two-CTA, linear indexed/subslice views, swizzle variants, 32-bit and representative subword linear rows. |
| `4x256b` | refresh-shaped 32-bit conversion, or four-row 256-bit atom candidate | Supported only for explicit refresh-shaped layouts, one-CTA and two-CTA. Ordinary contiguous four-row views are Gap #2C. |
| `warpx2::01_23.64x128b` | row bit 5 broadcast, row bit 6 non-broadcast | Supported for 32-bit no-scales single-CTA and two-CTA canonical layouts, plus dense-source rematerialization and descriptor views. Subword variants are Gap #2D. |
| `warpx2::02_13.64x128b` | row bit 6 broadcast, row bit 5 non-broadcast | Supported for 32-bit no-scales single-CTA canonical layouts and source rematerialization. Two-CTA remains Gap #2E. Subword variants are Gap #2D. |
| `warpx4.32x128b` | row bits 5 and 6 both broadcast | Supported for tensor-memory scales, direct copy, source rematerialization, one-CTA/two-CTA scaled-MMA use, and format/geometry sweeps. Descriptor-view mask cases are Gap #2B/#2F. |

## Coverage Evidence By Family

### Dense `128x128b`

Positive coverage:
- one-CTA direct runtime rows for legacy and explicit linear spelling;
- two-CTA direct runtime rows for legacy and explicit linear spelling;
- linear indexed-view rows;
- lit/FileCheck rows for core lowering and indexed view lowering.

Clean negatives:
- row/column-permuted layouts that require row/column masks or reordered
  sub-instruction footprints;
- noncanonical two-CTA block ownership;
- mixed row/column TMEM bases.

Audit result: implementation support is present. Legacy rows are compatibility
aliases, not a separate backend capability. Optional evidence improvement: add
a very small explicit-linear subword exact-width row, such as f16 `128x8` or
i8 `128x16`, if we want direct runtime proof that the dense `128x128b`
subword path is covered independently of `128x256b`.

### Dense `128x256b`

Positive coverage:
- one-CTA direct runtime rows across 128/256 row shapes, multiple N widths,
  block-N values, and swizzles;
- explicit linear 32-bit rows for f32/i32;
- explicit linear subword rows for f16/bf16/i16/i8;
- one-CTA indexed and subslice view rows;
- two-CTA direct rows for legacy and explicit linear spelling;
- two-CTA indexed/subslice view rows, including representative subword rows;
- tile-selector permutations where the selector is outside the instruction
  footprint and can be represented by physical tile offsets.

Clean negatives:
- sub-instruction tile permutations that would update only part of the public
  atom footprint;
- row/column permutations needing destination masks;
- mixed row/column bases;
- TMEM allocation OOR cases for intentionally too-large lifted views.

Audit result: implementation support is present for the realizable dense
linear layouts currently in scope. Remaining negatives are Gap #2B
full-footprint/mask boundaries, not generic layout incompleteness.

### Refresh `4x256b`

Positive coverage:
- explicit refresh-shaped 32-bit layout copy, one-CTA;
- explicit refresh-shaped 32-bit layout copy, two-CTA;
- lowering emits two `4x256b` messages for low/high source-column halves.

Clean negatives:
- ordinary contiguous `4x8` one-CTA and two-CTA views;
- direct `ld/st` readback of refresh images;
- raw physical bitcast readback of the sparse refresh image.

Audit result: the public copy instruction itself is covered for the only
layout contract currently exposed. Broader ordinary-view/readback behavior is
not a supported-layout-baseline miss; it is Gap #2C because it needs a
first-class refresh image view/remap/load-store contract.

### No-Scales `warpx2::01_23`

Positive coverage:
- 32-bit single-CTA canonical copy;
- 32-bit single-CTA indexed, subslice, and slice-index view rows;
- 32-bit single-CTA dense-source rematerialization;
- 32-bit two-CTA canonical copy;
- 32-bit two-CTA indexed, subslice, and slice-index view rows;
- 32-bit two-CTA dense-source rematerialization.

Clean negatives:
- subword variants that expose too few logical column bases for the 128-bit
  public atom and require packed-lane source/destination storage.

Audit result: support is complete for the 32-bit realizable family. Subword
support is Gap #2D, not a 32-bit layout-baseline issue.

### No-Scales `warpx2::02_13`

Positive coverage:
- 32-bit single-CTA canonical copy;
- 32-bit single-CTA indexed, subslice, and slice-index view rows;
- 32-bit single-CTA dense-source rematerialization.

Clean negatives:
- every current two-CTA row reaches the high source-column preservation
  diagnostic;
- subword variants require packed-lane storage.

Audit result: single-CTA support is covered. The two-CTA hole is the dedicated
Gap #2E research question; subword support is Gap #2D.

### Scales `warpx4.32x128b`

Positive coverage:
- direct single-CTA scales copy;
- direct two-CTA scales copy path currently exercised by explicit scales
  layouts;
- multiple noncanonical shared source layouts that rematerialize into the
  canonical `warpx4` copy source;
- shared-source subslice rematerialization;
- scaled-MMA copy paths for mxfp8/mxfp4/nvfp4 format pairs, use-acc variants,
  geometry sweeps, one-CTA/two-CTA, and multicast-sensitive cases.

Clean negatives:
- descriptor-view cases whose exact active view would require column/row masks
  or a copy-source contract the frontend does not expose yet.

Audit result: core `warpx4` scales copy coverage is strong. Remaining failures
belong to Gap #2B and Gap #2F.

## Cross-Cutting Negative Categories

The clean-negative set is now structured enough to avoid treating all failures
as "not layout-general":

- Source image not contained in destination view image: query selection rejects
  descriptors whose linear images cannot compose.
- Row-source projection split: a source-row bit wants to change source offset
  for only part of the instruction row footprint.
- Instruction-column projection split/permutation: a source/destination column
  bit wants a noncontiguous or descriptor-row-selected update inside one full
  atom.
- Packed-lane state: sub-32-bit copies need lane selection carried through
  descriptor synthesis and scheduling.
- Refresh image: `4x256b` writes a sparse refresh-shaped physical image.
- CTA ownership: two-CTA layouts must expose the canonical block basis used by
  the copy schedule.

These categories explain the current negatives without falling back to a vague
"linear layouts unsupported" bucket.

## Audit Findings

1. Gap #2 should no longer distinguish legacy and linear layouts as separate
   backend capabilities. Legacy coverage should be retained only as frontend
   compatibility evidence.
2. No new broad `tcgen05.cp` implementation gap was found. The major remaining
   unsupported rows all map cleanly to Gap #2B through Gap #2F.
3. The only evidence-level hole found by this audit is small: dense subword
   exact-width `128x128b` rows are not explicitly represented. Existing dense
   subword `128x256b` rows exercise the same packed dense path at wider N, so
   this is a test-evidence improvement rather than a known implementation
   issue.
4. Descriptor-view coverage is strong for index/subslice views, including
   `warpx2`, but broader reshape/transpose destination-view semantics should
   be discussed under Gap #2F if users need explicit frontend contracts for
   them.

## Active Sub-Buckets

- `2A`: optional dense subword exact-width `128x128b` evidence row.
- `2B`: partial copy footprints and mask/full-atom decomposition.
- `2C`: `4x256b` refresh image ordinary-view and readback semantics.
- `2D`: packed-lane `tcgen05.cp` and subword no-scales `warpx2`.
- `2E`: two-CTA no-scales `warpx2::02_13` source-column preservation.
- `2F`: frontend descriptor and copy-source contracts for `tcgen05_copy`.

## Recommended Closure Criteria

The supported-layout baseline inside Gap #2 can be closed when one of these
happens:
- add one or two tiny dense subword exact-width rows for `128x128b`, then mark
  Gap #2A supported; or
- explicitly waive that optional evidence as redundant with the existing dense
  subword `128x256b` rows, and mark Gap #2A as audit-complete rather than an
  implementation gap.

Full Gap #2 closure requires each sub-bucket `2A` through `2F` to be classified
as supported, impossible, or deferred. Future `tcgen05.cp` work should be
driven by these sub-buckets, not by legacy-versus-linear distinctions.
