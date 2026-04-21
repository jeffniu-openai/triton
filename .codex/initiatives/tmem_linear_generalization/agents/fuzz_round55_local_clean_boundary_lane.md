# Round 55 Local Clean-Boundary Runtime Lane

Date: 2026-04-21
Branch: `codex/tmem`
Checkpoint base: `904fad1e5 Integrate Round 54 scaled lhs tile evidence`

This local lane re-ran a compact clean-boundary selector over TMEM runtime rows
that should reject unsupported forms with stable diagnostics rather than crash,
miscompile, or silently become unsupported in a different way. It was chosen to
avoid overlapping the active Round 55 subagent lanes for dynamic two-CTA
descriptor consumers, `FZ-20260421-0003` sharpening, and lit/compiler-only
negative-boundary drift.

No backend fixes were attempted.

## Preflight

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Collection

```bash
PYTHONPATH=.:./python \
pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (reports_clean or reports_clean_unsupported or reports_clean_error or report_clean or clean_unsupported or clean_error) and (ldst_x1 or cp_no_scales or mma_scaled_lhs or ld_red or block_descriptor or blocked_layout)'
```

Result:

```text
96/1615 tests collected (1519 deselected) in 2.94s
```

## Runtime Split

Command shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
PYTHONPATH=.:./python \
pytest -q -s --tb=short \
  --splits 4 --group <group> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (reports_clean or reports_clean_unsupported or reports_clean_error or report_clean or clean_unsupported or clean_error) and (ldst_x1 or cp_no_scales or mma_scaled_lhs or ld_red or block_descriptor or blocked_layout)'
```

Results:

```text
group 1: 24 passed, 1591 deselected in 9.21s
group 2: 24 passed, 1591 deselected in 7.04s
group 3: 24 passed, 1591 deselected in 4.85s
group 4: 24 passed, 1591 deselected in 10.55s
aggregate: 96 passed, 0 failed
```

## Coverage Notes

The selector covered:

- blocked-layout and block-descriptor clean errors;
- `.x1` unsupported variants for `f32` and `i32`, 1CTA and 2CTA;
- no-scale copy clean errors and clean unsupported diagnostics, including
  transposed shared inputs, bad shared subslice offsets, `4x256b`, `warpx2`
  `02_13`, row-permuted destinations, noncanonical two-CTA blocks, legacy
  subword dtype boundaries, exotic linear layouts, tile-permuted
  subinstructions, and row/column-permuted linear layouts;
- scaled-MMAv5 LHS tile/subslice mixed-fp4A clean unsupported diagnostics.

## Classification

No compiler crash, verifier drift, diagnostic regression, false unsupported
diagnostic, runtime miscompile, hang, or new independent `FZ-*` bucket was
found in this lane.
