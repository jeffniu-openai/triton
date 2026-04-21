# TMEM Structural Fuzzing Lane P: Clean Boundary Adversarial Sweep

Scope: discovery-only boundary fuzzing. I used the temporary subprocess-isolated runner
`/tmp/fuzz_clean_boundary_round10_probe.py` and did not modify backend code or tests.

Repro commands:

```bash
CUDA_VISIBLE_DEVICES=0 python3 /tmp/fuzz_clean_boundary_round10_probe.py --suite gpu0 > /tmp/fuzz_round10_gpu0.log 2>&1
CUDA_VISIBLE_DEVICES=1 python3 /tmp/fuzz_clean_boundary_round10_probe.py --suite gpu1 > /tmp/fuzz_round10_gpu1.log 2>&1
CUDA_VISIBLE_DEVICES=2 python3 /tmp/fuzz_clean_boundary_round10_probe.py --suite gpu2 > /tmp/fuzz_round10_gpu2.log 2>&1
CUDA_VISIBLE_DEVICES=3 python3 /tmp/fuzz_clean_boundary_round10_probe.py --suite gpu3 > /tmp/fuzz_round10_gpu3.log 2>&1
CUDA_VISIBLE_DEVICES=1 python3 /tmp/fuzz_clean_boundary_round10_probe.py --case x1_subword_f16_linear_packed
CUDA_VISIBLE_DEVICES=1 python3 /tmp/fuzz_clean_boundary_round10_probe.py --case x1_subword_f16_legacy_unpacked
```

Findings:

- CP / TMEM OOR boundary rows
  - `cp_linear_indexed_128x64`: pass.
  - `cp_linear_indexed_128x128`: pass.
  - `cp_linear_indexed_128x256`: clean OOR, `Required: 1024, Hardware limit: 512`.
  - `cp_linear_indexed_128x512`: clean OOR, `Required: 262156, Hardware limit: 232448`.
  - `cp_twocta_indexed_256x64`: pass.
  - `cp_twocta_indexed_256x128`: pass.
  - `cp_twocta_indexed_256x256`: clean OOR, `Required: 1024, Hardware limit: 512`.
  - `cp_twocta_indexed_256x512`: clean OOR, `Required: 262156, Hardware limit: 232448`.
  - Overlap: these stay on the existing copy resource boundaries already tracked in the lane notes, including the 128x256 and 256x256 OOR families. No new bucket.

- Explicit subword lane rows
  - `x1_subword_f16_linear_packed`: pass, exact `tcgen05.st.sync.aligned.32x32b.x1.b32` / `ld.sync.aligned.32x32b.x1.b32`.
  - `x1_subword_f16_legacy_unpacked`: pass, exact `tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32` / `ld.sync.aligned.32x32b.x1.pack::16b.b32`.
  - `subword_legacy`: false unsupported / verifier failure. The compiler emitted `error encountered during parsing` for `f16`, `bf16`, `i16`, and `i8`; this is a generic parse-time failure, not a backend crash.
  - Overlap: explicit x1 rows are already-supported and the broad legacy-subword parse failures match the existing subword boundary family, not a new bucket.

- warpx2 schedule / descriptor boundaries
  - `warpx2_single_01_23`: pass.
  - `warpx2_single_02_13`: pass.
  - `warpx2_twocta_01_23`: pass.
  - `warpx2_twocta_02_13`: false unsupported / verifier failure. The note says the source-row projection cannot satisfy the `02_13` row-bit schedule.
  - `warpx2_twocta_slice_index_02_13`: subprocess-level MLIR shape mismatch abort before the row could lower.
  - Overlap: this stays on the known `warpx2::02_13` clean-unsupported frontier, not a new crash bucket.

- Scales descriptor-view and scaled-MMA boundaries
  - `scales_layout_warpx2_candidate`: pass.
  - `scales_layout_warpx2_tail`: pass.
  - `scales_tmem_descriptor_view`: false unsupported / verifier failure. The diagnostic says the row order must stay ascending for direct `warpx4.32x128b` lowering.
  - `mma_scaled_acc_tile_permuted_n16`: false unsupported / verifier failure. The diagnostic says the accumulator layout must be directly supported MMAv5 block-scaled TMEM; tile-permuted accumulators are not directly representable.
  - `mma_scaled_acc_tile_permuted_n32`: pass.
  - `cta_mismatch_3`: false unsupported / verifier failure, `Layout has 2 CTAs per CGA, but the context requires 3 CTAs per CGA.`
  - `cta_mismatch_4`: false unsupported / verifier failure, `Layout has 2 CTAs per CGA, but the context requires 4 CTAs per CGA.`
  - Overlap: the scales-view row is the existing clean descriptor-plan boundary; the tile-permuted scaled-MMA row stays on the known `FZ-20260421-0007`-family clean negative.

- `ld.red` transpose / slice boundary rows
  - `ld_red_identity_n128`: pass.
  - `ld_red_identity_n256`: pass.
  - `ld_red_identity_n512`: compiler crash / assertion boundary. PTXAS fails register allocation with 255 registers.
  - `ld_red_m64_splitn_n128`: pass.
  - `ld_red_m64_splitn_n256`: pass.
  - `ld_red_m64_splitn_n512`: pass.
  - Overlap: the `N=512` crash sits with the existing `FZ-20260421-0005/0009` `ld.red` resource/crash family already noted in `agents/fuzz_allocator_resource_round10.md`; it is not a new bucket.

Bottom line: no new independent `FZ-*` bucket from this lane. The only compiler failure row is an overlap with the existing `ld.red` crash family; the rest are either passes or clean boundary diagnostics.
