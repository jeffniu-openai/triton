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

## Work Split
- Main rollout owns core attr/type/helper/lowering integration.
- Subagents may be used for bounded exploration, PTX validation snippets, and
  test drafting. Claims belong in `agents/claims.md`.
