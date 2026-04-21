# Round 12 Lane T: Cache and Process Stability

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery-only; no backend fixes; no commits.

## Scope

Lane T stress-tested cache, process, and replay stability for a compact set of
TMEM sentinels and recent green controls. The goal was to find cache-key
omissions, stale runtime metadata, process-local contamination after bad
compiler/runtime rows, or failures that disappear only when changing process or
cache boundaries.

The matrix covered:

- `FZ-20260421-0001`: runtime TMEM `memdesc_index` illegal-op lowering.
- `FZ-20260421-0003`: ld/st descriptor-view packet-order miscompile.
- `FZ-20260421-0004`: `ld.red` descriptor/indexed rows emitting plain `ld`.
- `FZ-20260421-0007`: scaled-MMAv5 dynamic accumulator-view miscompile.
- `FZ-20260421-0009`: 1CTA direct indexed `ld.red` allocator assertion.
- Green structural controls for direct ld/st, direct `ld.red`, and scales copy.
- Runtime-matrix controls for clean copy diagnostics and two-CTA scales copy.

## Build

Required build command:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Sentinel Set

The ordered sentinel list was:

```text
python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]
python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_descriptor_view_read[ldst-fz20260421-0003-chain1-64x32-32x32b]
python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-direct-128x64]
python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min]
python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-1cta]
python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-2cta]
python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-index-chain0]
python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_scaled_mma_acc_subslice_control_flow[mma-scaled-fz20260421-0007-subslice-if-n64-selector0]
python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred_1cta_direct_index_allocator_crash
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_4x256b_reports_clean_unsupported
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_twocta_direct_copy
```

## Same Process, Stable Cache

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short \
  python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b] \
  python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_descriptor_view_read[ldst-fz20260421-0003-chain1-64x32-32x32b] \
  python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-direct-128x64] \
  python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min] \
  python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-1cta] \
  python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-2cta] \
  python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-index-chain0] \
  python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_scaled_mma_acc_subslice_control_flow[mma-scaled-fz20260421-0007-subslice-if-n64-selector0] \
  python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred_1cta_direct_index_allocator_crash \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_4x256b_reports_clean_unsupported \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error \
  python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_warpx4_twocta_direct_copy
```

Results:

- First run: `7 passed, 5 xfailed in 5.71s`.
- Immediate repeat with the same stable cache: `7 passed, 5 xfailed in 5.19s`.

The known bad rows still failed as their strict xfail buckets expect. The green
rows after the bad rows stayed green, including clean diagnostic checks and
copy controls. No process-local contamination was observed.

## Fresh Subprocesses, Same Stable Cache

Command shape:

```bash
for nodeid in "${NODEIDS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
    pytest -q -s --tb=short "$nodeid"
done
```

Results by row:

```text
ldst-view-identity-32x32b: 1 passed
ldst-fz20260421-0003-chain1-64x32-32x32b: 1 xfailed
ldred-direct-128x64: 1 passed
ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min: 1 xfailed
copy-scales-warpx4-1cta: 1 passed
copy-scales-warpx4-2cta: 1 passed
generic-pass-dynamic-index-chain0: 1 xfailed
mma-scaled-fz20260421-0007-subslice-if-n64-selector0: 1 xfailed
ldred_1cta_direct_index_allocator_crash: 1 xfailed
cp_no_scales_4x256b_reports_clean_unsupported: 1 passed
cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error: 1 passed
cp_scales_warpx4_twocta_direct_copy: 1 passed
```

All subprocess rows returned pytest status `0`. The same-cache fresh-process
classification matched the combined-process classification exactly.

## Fresh Diagnostic Cache

This was a diagnostic contrast only, not a workaround:

```bash
rm -rf /tmp/triton-cache-round12-fresh-gpu0
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-round12-fresh-gpu0 PYTHONPATH=.:./python pytest -s --tb=short <same sentinel list>
```

Result:

```text
7 passed, 5 xfailed in 8.30s
```

The fresh cache did not change any pass/xfail classification. No failure
disappeared or appeared because of a cache reset.

## Environment Hygiene Note

One exploratory `pytest --collect-only` command was intentionally discarded
because it was run without the required repository-local `PYTHONPATH=.:./python`
pinning and imported a stale `/tmp/triton-upstream-main-check` package:

```text
ModuleNotFoundError: No module named 'triton.runtime.jit'
```

That was an ambient environment/import-path issue, not a TMEM cache/process
signal. All recorded validation commands above used `PYTHONPATH=.:./python`.

## Classification

No new cache/process/replay instability candidate was found.

- Stable same-process replay on `/tmp/triton-cache-gpu0`.
- Stable immediate repeat with the same cache.
- Stable fresh subprocess replay using the same cache.
- Stable fresh diagnostic cache contrast.
- Bad sentinels remained in their existing known buckets.
- Green and clean-diagnostic controls remained green after bad sentinels.

Recommendation: do not create a new `FZ-*` bucket from Lane T. Continue using
the stable per-GPU cache directories for broad sweeps. Keep `PYTHONPATH=.:./python`
explicit in ad hoc probe commands because the workstation has stale Triton
checkouts under `/tmp` that can pollute collection if the environment is not
pinned.
