# Latest-Main Merge Resolution Audit - 2026-04-24

## Merge Context

- Branch: `codex/tmem`
- Current checkpoint before this audit: `5aaf5b6fa`
- Merge commit under review: `79d67152c0ace61850da0e260ca0d5854ff16112`
- Branch parent before merge: `29a061b538713146d0e269b7cb30d0e755f101b5`
- Incoming upstream main parent: `27c402843d9201afc22bde4df223934373721709`
- Merge base: `37c9a4b569a0f1719bdd77ca5f666096e10b7227`

## Incoming Main Intent

The incoming main changes were not just API churn. The important intent to preserve on this branch is:

- `GenericLinearEncodingAttr` introduces a less restrictive distributed linear encoding for layouts that are not representable by the older injective/permutation-only path. It also adds default-deny verifier coverage so only audited operations accept generic linear layouts.
- Generic-linear local load/store lowering removes an older tile-bases workaround and teaches lowering about `regsPerInst` in `actionAdditiveStrides`. The branch should keep this as an upstream backend correctness improvement, not paper over it with TMEM-specific special cases.
- Gluon attention moved to a `KernelConfig` selector and preallocated/CUDAGraph benchmark harness. This is source/harness intent from main and should remain the baseline when comparing example performance.
- Gluon MoE fused gather moved to the 2-CTA/multicast/direct-store implementation on main. The merged branch now matches main for `python/examples/gluon/05-moe-bmm1-fused-gather.py`; future regressions there should be treated as compiler/runtime deltas, not stale example source.
- FPSAN gained Hopper/WGMMA-facing imports and coverage. The conflict resolution preserved those imports while keeping branch TMEM helper coverage.

## Conflict Resolutions Reworked

### MMAv5 TMEM Address Layout

The branch had been using the MMAv5 family layout both to select an instruction family and to compute the TMEM tile address. That is wrong for layouts that are MMAv5-compatible only because they preserve the instruction tile interior while permuting whole instruction tiles.

Concrete failing shape: a `128 x 32` accumulator with an 8-column tile-selector permutation. The old lowering selected the correct narrow MMAv5 family, but addressed columns through the canonical family layout. Runtime output was correct up to swapping columns `8..15` with `16..23`, proving the result was being written in family order rather than the memdesc's exact physical order.

Fix: `getMMAv5TMemAddressLayout` and MMAv5 tile-order offset computation now prefer the exact current `MemDescType`/layout address image. The family layout remains only as a fallback for descriptors that still verify as MMAv5-compatible but are not yet exactly representable as a type-local view layout. The family layout remains the planner input for choosing legal MMA atom shapes. This preserves the project rule that codegen facts come from the current memdesc type/layout and immediate SSA value, not from producer/view-chain recovery.

### TMEM Allocation Versus Shared Allocation Ordering

After the merge, scaled-MMAv5 tile-permuted tests exposed a pass-order incompatibility: `allocate_shared_memory_nv` ran before `allocate_tensor_memory`, but tensor-memory allocation can rematerialize scale fragments and insert `ttg.convert_layout` operations that require shared scratch. Those newly inserted conversions did not receive `allocation.offset`, and LLVM lowering crashed in `getSharedMemoryBase`.

Fix: the NVIDIA lowering pipeline now runs `add_allocate_tensor_memory` before `add_allocate_shared_memory_nv`, so any conversions/rematerialized fragments introduced by tensor-memory allocation are visible to shared-memory allocation.

## Incoming Main Change Intentionally Not Adopted

Incoming main added a stricter `TMEMLoadOp::verify()` rule for `ld.red` that required the reduction dimension to be entirely register-local and rejected N-sharded layouts. That is too narrow for the TMEM project goal: this branch has generalized direct `ld.red` support for layouts where the N dimension is directly reducible, including lane-bit-4 cases with explicit software combination after the hardware reduction.

Resolution: keep the branch verifier/lowering semantics. The upstream stricter rule is recorded here as an intentional semantic divergence from main, not an accidental dropped conflict hunk. Future `ld.red` work should improve direct verifier/lowering support or produce clean unsupported diagnostics for true ISA/codegen impossibility; it should not restore the old software fallback or the upstream overly strict N-local-only rule.

## Validation After Rework

- `make` rebuilt the edited C++ and Python pipeline code.
- Exact prior crash row now passes: `test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_narrow_format_matrix[mxfp8-mxfp8-32-8-128]`.
- Exact prior tile-swap row now passes: `test_tmem_runtime_matrix_mma_acc_tile_permuted[32-8-32]`.
- 4-GPU split selector `mma_acc_tile_permuted or mma_plain_kinds_tile_permuted_acc or mma_scaled_acc_tile_permuted_narrow_format_matrix`: `42 passed`.
- 4-GPU split selector `mma_scaled_acc_tile_permuted`: `53 passed`.
- Lit conflict/regression files: `test/TritonGPU/invalid.mlir`, `test/TritonNvidiaGPU/invalid.mlir`, `test/TritonNvidiaGPU/tmem_layouts.mlir`: `3 passed`.
- `python/test/gluon/test_fpsan.py`: `85 passed, 22 skipped`.

## Remaining Boundaries

The family-layout fallback in MMAv5 address selection is retained as a compatibility boundary, not as a desired endpoint. The long-term model should reject or fix any descriptor that verifies as MMAv5-compatible but cannot provide an exact current type-local address image. No producer/view-chain walking was added in this rework.

The broad post-merge runtime-matrix inventory from `5aaf5b6fa` remains stale for MMAv5 tile-permuted/scaled-MMAv5 crash rows fixed by this slice. Other buckets from that inventory, including subword ld/st rows, descriptor-view ld/st expectations, copy-planner gaps, and structural-fuzzer rows that unexpectedly compile, still need independent refresh before prioritization.
