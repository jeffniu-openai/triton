# TMEM ISA/API Follow-Ups Outside Linear Layout

Last updated: 2026-04-18 19:50 UTC

This file records TMEM-related ISA/API context that is useful but not part of
the linear-layout generalization gap register. Items here should not consume
`remaining_coverage_gaps.md` gap numbers unless they become layout-dependent.
Some items are explicitly out of scope for this project and should not be
treated as active blockers.

## i8 MMAv5 Signedness and Saturation

- Area: plain MMAv5 direct `tcgen05.mma.kind::i8`.
- Project disposition: closed for the TMEM linear-layout generalization
  project.
- Classification: ISA/API exposure context, not a linear-layout gap.
- Delivered baseline: signed i8 direct MMAv5 is supported on GB200/sm100 for the
  current frontend/lowering path. Runtime pytest coverage includes
  `64x128x32`, `128x128x32`, and `128x256x64`, with exact int32 matmul checks
  and PTX/LLIR opcode checks.
- Out-of-scope idesc attribute work:
  - expose independent A/B signedness controls instead of the current single
    `is_unsigned` bit;
  - decide whether and how to expose integer saturation descriptor controls;
  - thread those controls through Triton IR, Gluon/frontends, descriptor
    encoding, and targeted FileCheck/PTXAS/runtime validation.
- Reopen condition: only reopen under a separate ISA/API project or an
  explicit user request to expose those instruction-descriptor attributes.
- Evidence/runbook: `gb200_i8_validation_artifacts.md`.
- Key lowering site:
  `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`.

## `tcgen05.cp` Follow-Ups Outside Linear Layout

- Area: public copy masks/smaller atoms, `4x256b` refresh-image load/store
  views, packed-lane/staged-copy storage, and explicit frontend copy-source or
  descriptor-view contracts.
- Project disposition: Gap #1 `tcgen05.cp` is closed for the TMEM
  linear-layout generalization project as of 2026-04-18 19:50 UTC.
- Classification: these are ISA/API/storage-model follow-ups, not residual
  generic linear-layout backend gaps.
- Current direct-support boundary:
  - public `tcgen05.cp` exposes full destination atom footprints and no
    per-row/per-column destination mask;
  - public `tcgen05.cp.warpx2` has no `ld/st`-style pack/unpack modifier, so
    sub-dword lane selection needs a first-class packed source/destination
    storage model or staged movement plan;
  - `tcgen05.cp.4x256b` writes a sparse refresh-shaped TMEM image, so ordinary
    contiguous `4x8` views and direct `ld/st` readback need an explicit
    refresh view/remap/load-store API;
  - cta-group::2 `warpx2::02_13` direct-seed probes emit the opcode but do not
    preserve the high source-column pair under the tested legal source/dest
    offset schedules.
- Reopen condition: only reopen under a separate request to design one of
  these contracts, or if a new public PTX/ISA capability exposes the missing
  mask/packing/source-selection semantics.
- Evidence/runbook: `tcgen05_cp_gap1_closure_20260418.md`.
