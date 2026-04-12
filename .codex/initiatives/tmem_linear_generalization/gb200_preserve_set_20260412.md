# GB200 Preserve Set - 2026-04-12 09:33 UTC

This is the preserve-set checkpoint before removing the branch-only TMEM
physical-layout / row-plan / MMAv5-root attributes and the lowering-side
special cases that used them as provenance.

## Semantic Decision

- The layout is the source of truth for TMEM descriptor semantics.
- A zero basis in a TMEM layout is broadcast/equivalence semantics, not a
  license to choose between divergent physical representatives.
- If two TMEM coordinates map to the same logical tensor element, codegen must
  preserve that equivalence. A store must update the physical image required by
  the layout, and a load may use any legal packetization only if the layout
  invariant is true.
- Therefore the branch-only attributes are not a principled way to pick a
  "live" representative. They are evidence that some producer, view, or ld/st
  lowering path is not respecting the layout's physical/broadcast semantics.
- The cleanup target is to remove these attributes and replace their effects
  with exact layout/view arithmetic plus codegen that implements broadcast
  layouts correctly.

## Checkpoint

- Branch / HEAD: `codex/tmem` at `3359982ee`.
- Artifact directory: `/tmp/gb200-ci-preserve-20260412-092408`.
- Build before tests:
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13 make -j8`
  - result: passed, ninja reported no work to do.
- Code-diff note:
  - since the full GB200 sweep at `cb76c31a0`, runtime compiler/source code is
    unchanged;
  - intervening changes are initiative docs plus lit/MLIR test updates only.

## Current Red Set

- `make test-lit`:
  - `246 passed, 2 failed, 2 unsupported`;
  - failures:
    - `TRITON :: TritonGPU/pipeline-loop-nest.mlir`;
    - `TRITON :: TritonGPU/pipeline-lower-loop.mlir`;
  - classification: stale FileCheck checks expecting bare `ttng.tmem_alloc`
    text while current output carries branch-only alloc attributes.
- `python/test/unit` branch-new exact manifest:
  - command reran all `162` nodeids from
    `gb200_branch_new_20260412_unit_main_failures.txt`;
  - result: `162 failed in 70.43s`;
  - breakdown remains:
    - `27` matmul / persistent matmul nodeids;
    - `7` tensor descriptor matmul nodeids;
    - `128` warp-specialization attention nodeids;
  - symptom: roughly half the output is wrong across the cluster.
- Gluon exact:
  - `python/test/gluon/test_core.py::test_block_m_64_mma[legacy]`;
  - result: failed in `4.52s`;
  - symptom: `8085 / 8192` elements mismatched.
- Proton main:
  - `python3 -m pytest -n 8 third_party/proton/test --ignore=third_party/proton/test/test_override.py -k "not test_overhead and not test_hw_trace"`;
  - result: `11 failed, 114 passed`;
  - classification: preexisting on merge-base; not part of TMEM branch
    recovery.

## Fresh Current-Head Green Evidence

- `make test-cpp`: `240/240` passed.
- `make NUM_PROCS=24 test-gsan`: `20 passed`.
- `make test-regression`: `1090 passed, 216 skipped`.
- `make test-microbenchmark`: completed with rc `0`.
- `python/test/unit/test_debug.py`: `95 passed`.
- `python/tutorials/06-fused-attention.py`: `192 passed, 192 skipped`.
- `python/test/unit/instrumentation/test_gpuhello.py`: `1 passed`.
- Unit plugin tail:
  - `python/test/unit/plugins/test_plugin.py`;
  - `python/test/unit/plugins/test_dialect_plugin.py`;
  - `python/test/unit/plugins/custom_ops.py`;
  - result: `3 passed`.
- Proton tails:
  - `test_profile.py::test_hw_trace`: `1 passed`;
  - `test_override.py`: `1 passed`;
  - `test_instrumentation.py::test_overhead`: `1 passed`.

## Carried-Forward Current Classification

These surfaces were covered by the full GB200 sweep at `cb76c31a0` and remain
valid for preserve-set purposes because runtime compiler/source code has not
changed since then:

- `python/triton_kernels/tests`: green split sweep.
- `python/examples/gluon`: green split sweep, including attention.
- `python/test/gluon` plus `python/tutorials/gluon`: only the legacy M64 MMA
  exact remains red.
- Merge-base comparison:
  - the `162` unit nodeids pass on merge-base;
  - the legacy M64 Gluon nodeid passes on merge-base;
  - the Proton `11` fail on merge-base too.

## Preserve Rule For The Cleanup

During attribute removal, do not introduce new failures outside this preserve
set. Narrow reruns should use the exact red nodeids above plus nearby TMEM
runtime/Gluon controls; broader reruns should refresh lit and the relevant
split pytest lanes at milestones.

## 2026-04-12 15:20 UTC Post-Cleanup Status

The red set recorded above is the preserve-set baseline, not the current head
status after the layout-only cleanup. At current head after the row-plan fixes:

- `make test-lit`: `248 passed, 2 unsupported`.
- The `162` exact `python/test/unit` nodeids from
  `gb200_branch_new_20260412_unit_main_failures.txt`: `162 passed`.
- `python/test/gluon/test_core.py::test_tmem_descriptor_chain_matrix`:
  `26 passed`.
- `test_tmem_linear_m64_roundtrip_direct_shapes` plus `test_block_m_64_mma`:
  `20 passed`.
- Physical bitcast subview mapping exacts `[0]` and `[64]`: `2 passed`.

Use the original preserve-set section to understand what had to be preserved
while removing attrs. Use this post-cleanup section as the current-head reading
for these exact buckets.
