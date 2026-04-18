# TMEM ISA/API Follow-Ups Outside Linear Layout

Last updated: 2026-04-18 07:16 UTC

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
