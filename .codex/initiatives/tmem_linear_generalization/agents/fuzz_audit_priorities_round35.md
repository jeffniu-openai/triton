# Round 35 Read-Only Audit Priorities

Date: 2026-04-21 13:35 UTC
Branch: `codex/tmem`
Scope: read-only audit. No files or backend code were edited by the auditor.

## Summary

The audit reviewed TMEM/Gluon backend surfaces and the checked-in runtime
matrix for under-fuzzed structural combinations. Highest expected yield:

1. Dynamic descriptor view selection across mbarrier/proxy-fence regions.
2. Shared-scale auto-materialization with multiple B-scale users and dynamic
   views.
3. High-rank `ld.red` through unit-prefix and half-slice descriptor chains.
4. Allocator row constraints with co-live LHS TMEM, accumulator TMEM, scale
   TMEM, and unrelated `ld/st` allocations.
5. `ld.red` resource-boundary fuzzing around `FZ-20260421-0018`, including
   descriptor views and software-reduce fallback boundaries.

## Concrete Probe Queue

- Dynamic descriptor view crossing mbarrier/proxy-fence regions.
  Start selector:
  `pytest -s --tb=short -k "(mbarrier or proxy or cp_scales) and not resource" python/test/gluon/test_tmem_runtime_matrix.py`.
- Shared-scale auto-materialization with multiple B-scale users and dynamic
  views. Start selector:
  `pytest -s --tb=short -k "bscale_descriptor_view or shared_scale_descriptor_view_auto_tmem_copy" python/test/gluon/test_tmem_runtime_matrix.py`.
- High-rank `ld.red` through unit-prefix and half-slice chains. Start selector:
  `pytest -s --tb=short -k "ld_red and (higher_rank or rank5 or multidim)" python/test/gluon/test_tmem_runtime_matrix.py`.
- `tcgen05.copy` no-scales over rank-4/rank-5 descriptor views. Start selector:
  `pytest -s --tb=short -k "cp_no_scales and (higher_rank or rank5 or multidim or warpx2 or warpx4)" python/test/gluon/test_tmem_runtime_matrix.py`.
- Allocator row constraints with co-live LHS TMEM, accumulator, scale TMEM, and
  unrelated live allocations. Start selector:
  `pytest -s --tb=short -k "alloc_lifetime or mma_scaled or lhs_subslice" python/test/gluon/test_tmem_runtime_matrix.py`.
- Legal 2CTA ops inside higher `num_ctas` launch contexts. Start selector:
  `pytest -s --tb=short -k "(twocta or cga_roundtrip or layout_in_4cta_context) and not reports" python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py`.
- `ld.red` resource boundary around `FZ-20260421-0018` with descriptor views
  and software-reduce fallbacks. Start selector:
  `pytest -s --tb=short -k "ld_red and (resource or m64 or descriptor_chain or rowcol)" python/test/gluon/test_tmem_runtime_matrix.py`.
- Sub-32-bit descriptor views with `ld.red` software reduce. Start selector:
  `pytest -s --tb=short -k "ld_red_non_f32 or subword" python/test/gluon/test_tmem_runtime_matrix.py`.
- Explicit unsupported diagnostics for x1/narrow shapes across copy, `ld/st`,
  and scaled MMA. Start selector:
  `pytest -s --tb=short -k "(x1 or n16 or n32 or clean_unsupported) and not reports" python/test/gluon/test_tmem_runtime_matrix.py`.
- Warp-specialization and loop-carried memdesc with TMEM alloc liveness. Start
  selector:
  `pytest -s --tb=short -k "generic_pass or loop_carried or dynamic_index" python/test/gluon/test_tmem_structural_fuzzer.py`.
- `tcgen05.copy` `warpx2`/`warpx4` opcode family with odd shared-layout
  rematerialization. Start selector:
  `pytest -s --tb=short -k "(copy and warpx) or cp_scales_layout_probe or shared_subslice_layout" python/test/gluon/test_tmem_runtime_matrix.py`.
- Descriptor-view algebra equivalence oracle: generate pairs of logically
  equivalent view chains, then write through chain A and read/consume through
  chain B for `ld/st`, `ld.red`, copy readback, and scaled-MMA accumulators.

## Current Follow-Through

The copy/scales descriptor-view guardrail and broad `ld.red`/descriptor
positive sweep were executed immediately after this audit. The former stayed
green; the latter expanded `FZ-20260421-0012` with M64 row-basis `ld.red`
planner failures.
