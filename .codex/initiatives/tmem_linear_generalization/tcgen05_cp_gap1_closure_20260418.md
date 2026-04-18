# Gap #1 `tcgen05.cp` Closure Report

Date: 2026-04-18 19:50 UTC

## Result

Gap #1 is closed for the TMEM linear-layout generalization project. The
remaining `tcgen05.cp` rejected rows are no longer unexplained
legacy-vs-linear, scales-vs-no-scales, or generic linear-layout backend gaps.
They are typed boundaries with stable diagnostics and focused coverage.

This does not mean every possible copy spelling is positive. It means the
current backend is complete over the public copy families and normalized
`LinearLayout` classes that the ISA/storage/frontend contracts can represent
today, and the remaining non-positive rows need new masks, new storage/API
contracts, or new ISA behavior.

## Sub-Bucket Disposition

| Sub-bucket | Disposition | Evidence |
| --- | --- | --- |
| `1A` dense subword exact-width `128x128b` | Supported and closed. | Added f16 `128x8` and i8 `128x16` explicit-linear positives checking exact `tcgen05.cp.cta_group::1.128x128b`. |
| `1B` partial/full-atom footprint rows | Closed as classified. Full public-copy atoms and tile selectors outside the instruction footprint are supported; sub-instruction permutations, row/column masks, noncanonical CTA ownership, and scales descriptor-view row masks remain clean unsupported because public `tcgen05.cp` exposes no destination mask or narrower atom for those footprints. | Focused selector covering positive tile permutations and clean-negative sub-instruction/row-column/CTA/scales-view rows passed `39/39`. |
| `1C` `4x256b` refresh image views | Closed as classified. Explicit refresh-shaped copy layouts are supported for one-CTA and two-CTA. Ordinary contiguous `4x8` exposure and `ld/st` readback are deferred behind a first-class refresh-image view/remap/load-store API contract. | Focused refresh selector passed `5/5`, including positive one-CTA/two-CTA copies and clean-negative ordinary/readback/raw-bitcast rows. |
| `1D` packed-lane/subword `warpx2` | Closed as classified. Direct public `tcgen05.cp.warpx2` lacks pack/unpack modifiers and cannot model sub-dword lane selection through the current descriptor and destination footprints. Any future support should be a separate packed-storage/staged-copy design, not a hidden linear-layout fallback. | Packed-lane clean-error rows passed in the refresh/packed selector; diagnostics name the packed-lane source/destination storage requirement. |
| `1E` two-CTA `warpx2::02_13` | Closed as an ISA/schedule boundary. Current descriptor plans require row bit 5 to select a one-dword source offset inside a full 64-row atom; public copy has no row mask. The prior direct-seed sweep emitted `cta_group::2.warpx2::02_13` but duplicated the selected low source-column pair into the high destination columns or read zeros/faulted for offset variants. | Current direct/index/subslice/slice-index/dense-source clean-negative rows passed `16/16`; historical direct-seed sweep is recorded in the handoff. |
| `1F` frontend/source contracts | Closed as classified. Automatic rematerialization is supported where semantics are safe and bounded. Remaining rows require explicit descriptor-view/copy-source API contracts or stay clean unsupported when implicit rematerialization would hide expensive/ambiguous movement. | Focused source-contract selector passed `17/17`, covering scales shared rematerialization, no-scales `warpx2` dense-source rematerialization, and the remaining clean-negative descriptor-view/source-contract rows. |

## Final Classification

- Supported today: dense `128x128b`, dense `128x256b`, explicit refresh-shaped
  `4x256b`, no-scales 32-bit `warpx2::01_23`, no-scales 32-bit single-CTA
  `warpx2::02_13`, and scales `warpx4.32x128b`, including the descriptor-view
  and source-rematerialization rows already covered by the runtime matrix.
- Clean final boundaries under current public ISA/storage contracts:
  sub-instruction destination masks, noncanonical two-CTA block ownership,
  mixed row/column physical bases, and two-CTA `warpx2::02_13` high
  source-column preservation.
- Deferred non-linear-layout projects: explicit `4x256b` refresh-image
  load/store views, packed-lane/staged-copy storage for subword `warpx2`, and
  explicit frontend copy-source/descriptor-view contracts.

## Validation

- `make -j8`:
  `ninja: no work to do`.
- `1E` direct/dense-source focused rows:
  `4 passed in 3.28s`.
- `1E` direct/index/subslice/slice-index/dense-source focused rows:
  `16 passed in 6.93s`.
- `1B` partial-footprint and mask/full-atom selector:
  `39 passed in 13.85s`.
- `1C` plus `1D` refresh/packed selector:
  `23 passed in 5.16s`.
- `1F` source-contract/rematerialization selector:
  `17 passed in 10.17s`.

## Reopen Conditions

Reopen `tcgen05.cp` as a linear-layout gap only if a supported public
instruction/storage contract and normalized layout class fails for reasons not
covered by the classifications above. New work on masks, packed lanes,
refresh-image readback, or frontend copy-source APIs should start as a separate
ISA/API/storage initiative or follow-up, not as a resurrection of the generic
linear-layout gap.
