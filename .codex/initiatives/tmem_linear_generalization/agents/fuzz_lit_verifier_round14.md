# Round 14 Lane AH: Lit and Verifier TMEM Backend Fuzzing

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend/compiler code was changed.

## Summary

Lane AH focused on compiler-only/lit/verifier coverage for surfaces that are
easy to miss with Python runtime probes: TMEM layout verifiers,
Blackwell `tritongpu-to-llvm` conversion, proxy-fence insertion,
`OptimizeTMemLayouts`, and matmul two-CTA module consistency.

No new independent `FZ-*` bucket was found and no checked-in lit sentinel was
warranted. The temporary MLIR probes either passed, produced clean expected
diagnostics, or reproduced existing report-only buckets:

- `FZ-20260421-0001`: dynamic TMEM `ttg.memdesc_index` still reaches LLVM
  conversion as an illegal op. Static index control passes.
- `FZ-20260421-0012`: a 12-line compiler-only M64 row-rotated `ld.red` probe
  reproduces unsupported destination-layout lowering. M64 row-identity and
  M128 row-rotated controls pass.
- `FZ-20260421-0014`: the saved mixed copy/scales reproducer still fails in
  the full reproducer pipeline at proxy-fence insertion. Running the raw file
  through only the proxy-fence pass passes, so the failure still depends on
  the preceding allocation/membar pipeline state.
- 4/8/16 CTA layout-context verifier probes produced clean expected
  diagnostics, mapping to the known high-CGA over-strict gate family rather
  than a new bucket.
- Matmul two-CTA consistency probes for 4/8/16 CTA contexts passed or produced
  clean verifier diagnostics; the pass correctly propagated `"ttng.two-ctas" =
  true` for consistent high-CGA two-CTA MMA modules.

## Required Setup

```bash
make -j8
```

Result: no work to do.

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
```

Result: no work to do.

## Anchor Lit Sweep

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
lit -v \
  test/TritonNvidiaGPU/tmem_layouts.mlir \
  test/TritonNvidiaGPU/invalid.mlir \
  test/TritonNvidiaGPU/membar-cluster.mlir \
  test/TritonNvidiaGPU/membar.mlir \
  test/Conversion/tritongpu_to_llvm_blackwell.mlir
```

Result:

```text
Total Discovered Tests: 5
  Passed: 5 (100.00%)
```

## Temporary MLIR Probes

Temporary files created under `/tmp`:

- `/tmp/tmem_laneah_cta_context_verifier.mlir`
- `/tmp/tmem_laneah_ldred_m64_rotate.mlir`
- `/tmp/tmem_laneah_ldred_m64_identity.mlir`
- `/tmp/tmem_laneah_ldred_m128_rotate.mlir`
- `/tmp/tmem_laneah_mma_twocta_consistency.mlir`
- `/tmp/tmem_laneah_mma_twocta_high_cga_consistent.mlir`

Existing temporary repros reused:

- `/tmp/tmem_memdesc_index_r4_lit_candidate.mlir`
- `/tmp/tmem_memdesc_index_r4_static_c1.mlir`
- `/tmp/tmem_high_cga_copy_scales_round14_mixed_fail.mlir`
- `/tmp/tmem_laneaa_ldred_fz0012_child.py`

## TMEM Layout Verifier, 4/8/16 CTA Contexts

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt --split-input-file --verify-diagnostics \
  /tmp/tmem_laneah_cta_context_verifier.mlir
```

Result: passed. The three split chunks produced expected clean diagnostics:

```text
Layout has 1 CTAs per CGA, but the context requires 4 CTAs per CGA.
Layout has 1 CTAs per CGA, but the context requires 8 CTAs per CGA.
Layout has 1 CTAs per CGA, but the context requires 16 CTAs per CGA.
```

Classification: existing high-CGA CTA-count verifier boundary (`FZ-20260421-0010`
family), not a new candidate.

## `FZ-20260421-0001` Compiler-Only Recheck

Dynamic memdesc-index candidate:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt --run-reproducer /tmp/tmem_memdesc_index_r4_lit_candidate.mlir
```

Result: failed as existing `FZ-20260421-0001`:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

Static-index control:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt --run-reproducer /tmp/tmem_memdesc_index_r4_static_c1.mlir
```

Result: passed through the reproducer pipeline. It emitted only an unrelated
vectorization remark at the final `tt.store`.

Diagnostic cleanliness check:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt --verify-diagnostics --run-reproducer \
  /tmp/tmem_memdesc_index_r4_lit_candidate.mlir
```

Result: failed with `unexpected error`, confirming this remains a diagnostic
gap if runtime TMEM indexing is intended to be unsupported.

Classification: existing `FZ-20260421-0001`; no new bucket.

## `FZ-20260421-0012` Compiler-Only M64 `ld.red`

Smallest compiler-only M64 row-rotated probe:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt /tmp/tmem_laneah_ldred_m64_rotate.mlir \
  --convert-triton-gpu-to-llvm=compute-capability=103 -cse
```

Result: failed as existing `FZ-20260421-0012`:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
Failed to lower TMEM load/store: unsupported dst layout
where out dims are: [row (size 128), col (size 32)]
```

M64 row-identity control:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt /tmp/tmem_laneah_ldred_m64_identity.mlir \
  --convert-triton-gpu-to-llvm=compute-capability=103 -cse
```

Result: passed.

M128 row-rotated control:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt /tmp/tmem_laneah_ldred_m128_rotate.mlir \
  --convert-triton-gpu-to-llvm=compute-capability=103 -cse
```

Result: passed.

Python child command used only to confirm the same diagnostic source as Lane AA:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python3 /tmp/tmem_laneaa_ldred_fz0012_child.py \
  --mode default --layout m64 --n 32 --row rotate1 --col identity --op min
```

Result: failed during compilation with the same unsupported destination-layout
diagnostic. No kernel execution or repair was attempted.

Classification: existing `FZ-20260421-0012`; the new `/tmp` MLIR is a smaller
compiler-only repro candidate than the Python child.

## `FZ-20260421-0014` Proxy-Fence Recheck

Full saved reproducer:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt --run-reproducer \
  /tmp/tmem_high_cga_copy_scales_round14_mixed_fail.mlir
```

Result: reproduced existing `FZ-20260421-0014`:

```text
'tt.func' op could not find an insertion point between cross-CTA
mbarrier.init ops and tracked mbarrier uses
```

Raw proxy-fence-only contrast:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt /tmp/tmem_high_cga_copy_scales_round14_mixed_fail.mlir \
  --triton-nvidia-gpu-proxy-fence-insertion=compute-capability=103
```

Result: passed. This keeps the classification on the full
`ConvertTritonGPUToLLVM` reproducer pipeline after tensor-memory allocation,
shared-memory allocation, and matmul two-CTA consistency checking.

Classification: existing `FZ-20260421-0014`; no smaller raw-pass-only repro
was found in this lane.

## Matmul Two-CTA Module Consistency

4CTA consistency and negative diagnostic:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt --split-input-file --verify-diagnostics \
  /tmp/tmem_laneah_mma_twocta_consistency.mlir \
  --triton-nvidia-check-matmul-two-cta
```

Result: passed. The positive chunk gained `"ttng.two-ctas" = true`; the
negative chunk produced a clean verifier diagnostic:

```text
The returned value's encoding must have twoCTA=0 to be used in a non-twoCTA kernel
```

8CTA and 16CTA consistent high-CGA controls:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt --split-input-file \
  /tmp/tmem_laneah_mma_twocta_high_cga_consistent.mlir \
  --triton-nvidia-check-matmul-two-cta
```

Result: passed. Both modules gained `"ttng.two-ctas" = true`.

Classification: no new matmul consistency candidate. The pass handles
consistent 4/8/16 CTA two-CTA MMA modules, and the inconsistent 4CTA case is
rejected by an existing clean verifier diagnostic before a pass-manager crash.

## Counts

Anchor lit files: `5 passed / 0 failed`.

Temporary `triton-opt` probes:

```text
7 passed:
  - 4/8/16 CTA verifier expected diagnostics
  - M64 row-identity ld.red control
  - M128 row-rotated ld.red control
  - FZ-0001 static-index control
  - FZ-0014 raw proxy-fence-only contrast
  - 4CTA matmul consistency/expected diagnostic file
  - 8/16CTA matmul consistency controls

4 failed as known buckets:
  - FZ-0001 dynamic memdesc_index run-reproducer
  - FZ-0001 dynamic memdesc_index verify-diagnostics cleanliness check
  - FZ-0012 M64 row-rotated ld.red conversion
  - FZ-0014 full run-reproducer pipeline
```

Additional Python compile-only confirmation: `1` known `FZ-0012` compile
failure, with no runtime execution.

No new report-only candidate was assigned.
