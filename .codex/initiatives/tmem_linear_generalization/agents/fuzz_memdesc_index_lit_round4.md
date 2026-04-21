# TMEM Structural Fuzzing Lane R4-B: Dynamic `memdesc_index` Lit Candidate

- Time: 2026-04-21 08:47 UTC
- Branch: `codex/tmem`
- HEAD: `98fde8a79`
- Scope: produce a compact lit/MLIR minimization candidate for direct runtime
  TMEM `memdesc_index`, and test nearby static/dynamic diagnostic boundaries
  for `FZ-20260421-0001`.
- Mode: discovery/minimization only. No backend/compiler repairs attempted.

## Setup

Read first:

- `.codex/initiatives/tmem_linear_generalization/tmem_structural_fuzzing_20260421.md`
- `.codex/initiatives/tmem_linear_generalization/agents/fuzz_memdesc_index_round3.md`
- `python/test/gluon/test_tmem_structural_fuzzer.py`

Required build before pytest:

```bash
make -j8
```

Result: `ninja: no work to do.`

## Checked-In Python Repro Confirmation

Command:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r4-memdesc-gpu0 pytest -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_dynamic_index_load_only[generic-pass-dynamic-index-load-only-128x32]' 2>&1 | tee /tmp/tmem_memdesc_index_r4_checked_in_xfail.log
```

Result: `1 xfailed`.

Current diagnostic is still:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

The emitted reproducer still points at direct
`view = parent.index(index)` in
`_fuzz_generic_pass_dynamic_index_load_only_kernel`, with no descriptor-view
reshape/permute chain.

## Compact MLIR Candidates

Temporary MLIR files created under `/tmp`:

- `/tmp/tmem_memdesc_index_r4_min_direct.mlir`:
  24-line direct runtime selector-load candidate. Keeps only TMEM allocation,
  `tt.load` scalar selector, dynamic `ttg.memdesc_index`, `ttng.tmem_load`,
  and a store to make the view live.
- `/tmp/tmem_memdesc_index_r4_dynamic_arg.mlir`:
  23-line smaller dynamic scalar-argument candidate. This removes the selector
  pointer load and proves the issue is the dynamic SSA index operand itself.
- `/tmp/tmem_memdesc_index_r4_static_c1.mlir`:
  24-line static `%c1` control.
- `/tmp/tmem_memdesc_index_r4_lit_candidate.mlir`:
  26-line lit-shaped candidate using the dynamic scalar argument and a
  `// RUN: not triton-opt --run-reproducer %s 2>&1 | FileCheck %s` line.

Important minimization note: an even smaller IR that computed
`ttg.memdesc_index` but did not use the resulting view was DCE'd and passed.
The `ttng.tmem_load` plus output store is the smallest live-use form tested.

## `triton-opt --run-reproducer` Results

Direct runtime selector-load candidate:

```bash
set -o pipefail; build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt --run-reproducer /tmp/tmem_memdesc_index_r4_min_direct.mlir >/tmp/tmem_memdesc_index_r4_min_direct_triton_opt_nopipe.log 2>&1; echo exit:$?; sed -n '1,80p' /tmp/tmem_memdesc_index_r4_min_direct_triton_opt_nopipe.log
```

Result: `exit:1`, failing at the dynamic `ttg.memdesc_index`.

Dynamic scalar-argument candidate:

```bash
set -o pipefail; build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt --run-reproducer /tmp/tmem_memdesc_index_r4_dynamic_arg.mlir >/tmp/tmem_memdesc_index_r4_dynamic_arg_triton_opt.log 2>&1; echo dynamic_arg_exit:$?; sed -n '1,80p' /tmp/tmem_memdesc_index_r4_dynamic_arg_triton_opt.log
```

Result: `dynamic_arg_exit:1`, failing at the dynamic `ttg.memdesc_index`.

Static `%c1` control:

```bash
set -o pipefail; build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt --run-reproducer /tmp/tmem_memdesc_index_r4_static_c1.mlir >/tmp/tmem_memdesc_index_r4_static_c1_triton_opt.log 2>&1; echo static_c1_exit:$?; sed -n '1,40p' /tmp/tmem_memdesc_index_r4_static_c1_triton_opt.log
```

Result: `static_c1_exit:0`. The static index lowers through the full
reproducer pipeline. It emits an unrelated vectorization remark at the store,
but no `memdesc_index` legalization failure.

`--verify-diagnostics` boundary:

```bash
set -o pipefail; build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt --verify-diagnostics --run-reproducer /tmp/tmem_memdesc_index_r4_min_direct.mlir >/tmp/tmem_memdesc_index_r4_min_direct_verify_diag.log 2>&1; echo verify_diag_exit:$?; sed -n '1,80p' /tmp/tmem_memdesc_index_r4_min_direct_verify_diag.log
```

Result: `verify_diag_exit:1`, with `unexpected error: failed to legalize
operation 'ttg.memdesc_index'...`. This supports the round-3 classification:
if runtime TMEM indexing is unsupported, the current failure is still a
diagnostic gap rather than a clean unsupported diagnostic.

Lit-shaped candidate validation:

```bash
build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt --run-reproducer /tmp/tmem_memdesc_index_r4_lit_candidate.mlir > /tmp/tmem_memdesc_index_r4_lit_candidate_check.log 2>&1; status=$?; rg -q "failed to legalize operation 'ttg.memdesc_index'" /tmp/tmem_memdesc_index_r4_lit_candidate_check.log; check=$?; echo triton_opt_exit:$status text_check_exit:$check; sed -n '1,40p' /tmp/tmem_memdesc_index_r4_lit_candidate_check.log
```

Result: `triton_opt_exit:1 text_check_exit:0`.

Standalone shell validation of the exact `RUN` line was blocked because
`FileCheck` is not on PATH in this environment, and `not` is a lit helper, not
a shell command. The candidate is still mechanically lit-shaped; a normal lit
environment with `FileCheck` available should be able to run it.

## Nearby Python/Gluon Boundaries

Small direct dynamic shape:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r4-memdesc-gpu1 pytest -s --tb=short '/tmp/tmem_memdesc_index_r3_probe.py::test_dynamic_index[64-32-1-0]' 2>&1 | tee /tmp/tmem_memdesc_index_r4_dynamic_64x32_direct.log
```

Result: `1 failed` before LLVM conversion with a clean frontend diagnostic:

```text
TMEM layout 'constexpr[32x32b]' unsupported ... required row anchors 32,64 are not directly representable
```

Static/constexpr direct control:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r4-memdesc-gpu2 pytest -s --tb=short '/tmp/tmem_memdesc_index_r3_probe.py::test_constexpr_index[128-64-1-0]' 2>&1 | tee /tmp/tmem_memdesc_index_r4_constexpr_128x64_direct_idx1.log
```

Result: `1 passed`.

These current-head reruns preserve the round-3 boundary:

- small direct shapes can fail early with clean packet/row-anchor diagnostics;
- static/constexpr direct outer `memdesc_index` passes;
- dynamic direct outer `memdesc_index` reaches `ConvertTritonGPUToLLVM` as an
  illegal op when the shape is otherwise resource-valid.

## Lit Readiness

Best current lit/minimization candidate:

- `/tmp/tmem_memdesc_index_r4_lit_candidate.mlir`

This is compact enough for a future compiler-only test. It isolates the
failure to a dynamic scalar index into a rank-3 TMEM descriptor and avoids
helper-returned views, control flow, reshape, permute, pre-store, and runtime
data comparison.

Recommended checked-in policy for now:

- Keep the current checked-in test as Python strict xfail coverage for the
  public Gluon/runtime path.
- Do not add a normal passing lit test that expects the current illegal-op
  crash diagnostic; that would pin the bug-shaped failure text as desired
  behavior.
- Add lit later in one of two forms:
  - as a clean negative diagnostic test if runtime TMEM `memdesc_index` is
    intentionally unsupported; or
  - as an `XFAIL` compiler-only reproducer while backend repair is underway,
    then flip it to positive lowering once dynamic indexing is supported.

If a GPU-free repair target is needed before that policy decision, the
26-line `/tmp/tmem_memdesc_index_r4_lit_candidate.mlir` is ready to copy into
the lit tree with the appropriate `XFAIL` or expected-diagnostic contract.
