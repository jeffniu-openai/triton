# Round 56 Local `ld.red` Mixed Hardware/Software Lane

Date: 2026-04-21
Branch: `codex/tmem`
Checkpoint base: `e0c1f9aba Integrate Round 55 local runtime evidence`

This local lane exercised a focused `ld.red` selector over hardware f32
reductions, descriptor-chain reductions, compatible non-identity layouts that
canonicalize to `32x32b`, explicit software-reduce layouts, and non-f32
software-reduce contracts. It complements the Round 56 subagent lanes for
dynamic copy descriptors, scaled-MMAv5 dynamic operands, and broad
`test_core.py` sampling.

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
  -k 'test_tmem_runtime_matrix and ld_red and (non_f32_contract_uses_software_reduce or explicit_n_sharded_layout_uses_software_reduce or descriptor_chain_n_sweep_explicit_variants or explicit_compatible_layout_variants or explicit_compatible_non_identity_layouts_canonicalize_32x32b) and not reports and not resource'
```

Result:

```text
60/1615 tests collected (1555 deselected) in 3.09s
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
  -k 'test_tmem_runtime_matrix and ld_red and (non_f32_contract_uses_software_reduce or explicit_n_sharded_layout_uses_software_reduce or descriptor_chain_n_sweep_explicit_variants or explicit_compatible_layout_variants or explicit_compatible_non_identity_layouts_canonicalize_32x32b) and not reports and not resource'
```

Results:

```text
group 1: 15 passed, 1600 deselected in 4.55s
group 2: 15 passed, 1600 deselected in 4.24s
group 3: 15 passed, 1600 deselected in 4.19s
group 4: 15 passed, 1600 deselected in 4.20s
aggregate: 60 passed, 0 failed
```

## Coverage Notes

The selector covered:

- explicit compatible hardware f32 `ld.red` variants for `min`, `max`, abs,
  and NaN propagation;
- descriptor-chain N-sweep explicit variants, including identity,
  tile-permuted, column-reversed, row-reversed, row-even/odd, and mixed
  row/column layouts;
- compatible non-identity layouts that canonicalize to `32x32b`;
- explicit N-sharded layouts that intentionally use software reduce;
- non-f32 `ld.red` software-reduce contracts for `i32`, `bf16`, `f16`,
  `i16`, `i8`, and legacy unpacked f16 forms.

## Classification

No compiler crash, verifier drift, false unsupported diagnostic, opcode
absence, hardware/software classification drift, runtime miscompile, hang, or
new independent `FZ-*` bucket was found in this lane.
