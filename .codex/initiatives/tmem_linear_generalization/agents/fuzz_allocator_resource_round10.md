# Round 10 Lane K: TMEM allocator/resource-boundary adversarial fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only. No backend/compiler repairs were attempted.
- Temporary probes: inline `/tmp`-style child-process wrappers run from stdin
  on the four local GPUs.

## Scope

This lane probed allocator and resource boundaries across:

- live parent vs selected child size;
- direct index, slice, and subslice chains;
- `M` values at the practical power-of-two boundaries around
  `64, 128, 256, 512`;
- `N` values at `32, 64, 128, 256` for the kernels that accept them;
- `ld/st`, `ld.red`, copy, and scaled-MMAv5 accumulator descriptor paths;
- subword and dtype variants where the runtime API already exposes them; and
- `num_ctas`-sensitive cases where the runtime matrix supports them.

The probe isolated each crashy row in a child Python process so a compiler
assertion or MLIR/LLVM abort could not kill the whole batch.

## Commands

Required rebuild:

```bash
make -j8
```

Structural-fuzzer boundary sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-round10-laneK-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon python - <<'PY'
# child-process wrapper that called:
#   sf.test_tmem_structural_fuzzer_ldst_view_roundtrip(...)
#   sf.test_tmem_structural_fuzzer_ldst_descriptor_view_read(...)
#   sf.test_tmem_structural_fuzzer_ldred(...)
# on a small set of 64/128/256/512-row cases
PY
```

Copy boundary sweep:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-round10-laneK-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon python - <<'PY'
# child-process wrapper that called runtime-matrix copy tests for:
#   CP_LINEAR_INDEXED_VIEW_CASES[0:2]
#   CP_LINEAR_SUBSLICE_VIEW_CASES[0] and [5]
#   CP_TWOCTA_LINEAR_INDEXED_VIEW_CASES[0]
#   CP_TWOCTA_LINEAR_SUBSLICE_VIEW_CASES[-1]
PY
```

Scaled-MMA accumulator-descriptor sweep:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-round10-laneK-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon python - <<'PY'
# child-process wrapper that called runtime-matrix scaled-MMA tests for:
#   SCALED_MMA_INDEXED_ACC_FORMAT_CASES[0] and [-1]
#   SCALED_MMA_ACC_SUBSLICE_USE_ACC_CASES[0] and [-1]
#   SCALED_MMA_TWOCTA_ACC_SUBSLICE_K_CASES[0] and [-1]
PY
```

Explicit boundary reproductions:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-round10-laneK-gpu3-resource \
  PYTHONPATH=.:./python:./python/test/gluon python - <<'PY'
rm.tmem_copy_no_scales_twocta_linear_indexed_view_kernel[(1,)](
    inp,
    out,
    parent_layout,
    tuple(tuple(basis) for basis in cga_layout),
    M,
    N,
    swizzle,
    2,
    1,
    num_ctas=2,
    num_warps=4,
)
PY

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-round10-laneK-gpu3-unsupported \
  PYTHONPATH=.:./python:./python/test/gluon python - <<'PY'
rm.tmem_mma_scaled_layout_format_kernel[(1,)](
    out,
    m,
    n,
    k,
    a,
    b,
    a_scale,
    b_scale,
    layout,
    vec_size,
    a_elem_per_byte,
    b_elem_per_byte,
    a_tcgen_format,
    b_tcgen_format,
    0.0,
    num_warps=4,
)
PY
```

## Observed Results

### Passes

- `ldst-64x32-chain0`
- `ldst-128x64-chain1`
- `ldred-128x64-direct`
- `ldred-128x64-chain1`
- `ldred-256x64-twocta-direct`
- all six copy rows in the copy slice
- all six scaled-MMA rows in the scaled-MMA slice

### Clean resource / hardware diagnostic

The direct 2CTA copy OOR boundary reproduced cleanly:

```text
OutOfResources
out of resource: tensor memory, Required: 1024, Hardware limit: 512.
Reducing block sizes or `num_stages` may help.
```

This is a normal resource diagnostic, not a backend crash.

### Clean unsupported boundary

The scaled-MMA tile-permuted accumulator boundary reproduced cleanly:

```text
RuntimeError
error encountered during parsing
```

with the leading diagnostic:

```text
ttng.tc_gen5_mma_scaled op expected accumulator layout to be directly supported MMAv5
block-scaled tensor memory ... tile-permuted accumulator layouts are not directly representable.
```

This remains a verifier-style unsupported boundary, not a crash.

### Compiler crashes / assertions

- `ldst-256x32-chain0`
- `ldst-512x32-chain0`
- `ldred-256x32-direct`
- `ldred-512x32-direct`

All four still die during compilation with `RuntimeError: PassManager::run failed`.
They do not degrade into a clean resource diagnostic.

### Miscompile / opcode regression

- `ldst-desc-f16-64x32-chain2` failed its `torch.testing.assert_close` check
  with 50% mismatched elements. This overlaps the existing `FZ-20260421-0003`
  ld/st descriptor-view chain miscompile bucket.
- `ldred-256x32-twocta-indexed` failed because the expected `.ld.red.` opcode
  was missing and the lowering fell back to plain `ld`. This overlaps
  `FZ-20260421-0004`.

## Classification

No new non-overlapping `FZ-*` bucket is recommended from this lane.

What the fresh probes show:

- `FZ-20260421-0005` and `FZ-20260421-0009` still behave like compiler
  assertion/crash boundaries, not clean diagnostics, for 256-row and larger
  direct-index `ld/st` and `ld.red` parents.
- `FZ-20260421-0004` still owns the 2CTA indexed `ld.red` plain-load fallback.
- `FZ-20260421-0003` still owns the descriptor-view `ld/st` chain miscompile.
- The scaled-MMA accumulator tile-permuted `N=16` case is a clean unsupported
  boundary and should stay classified as such.
- The 2CTA copy `256x256` boundary is a clean `OutOfResources` diagnostic.

Backend repair remains deferred until the discovery campaign stops finding new
bugs or the user explicitly pivots from discovery to repair.
