# Round 55 Local Copy/MMA Positive Runtime Lane

Date: 2026-04-21
Branch: `codex/tmem`
Checkpoint base: `20c481e5a Record Round 55 local clean boundary lane`

This local lane exercised a compact positive runtime mix over TMEM copy,
scaled-copy, plain MMAv5, and scaled-MMAv5 root rows in
`python/test/gluon/test_tmem_runtime_matrix.py`. It was intended to keep a
green positive signal running while the Round 55 subagents focused on dynamic
two-CTA descriptor consumers, `FZ-20260421-0003` boundaries, and compiler-only
negative-boundary drift.

No backend fixes were attempted.

## Collection

```bash
PYTHONPATH=.:./python \
pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and ((cp_no_scales and (warpx2_01_23_candidate_positive or dense_shared_rematerializes or linear_tile_permuted or shared_subslice or twocta_codegen)) or cp_scales_warpx4 or mma_plain_kinds_with_linear_acc or mma_scaled_root_format_matrix) and not reports and not resource'
```

Result:

```text
88/1615 tests collected (1527 deselected) in 2.98s
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
  -k 'test_tmem_runtime_matrix and ((cp_no_scales and (warpx2_01_23_candidate_positive or dense_shared_rematerializes or linear_tile_permuted or shared_subslice or twocta_codegen)) or cp_scales_warpx4 or mma_plain_kinds_with_linear_acc or mma_scaled_root_format_matrix) and not reports and not resource'
```

Results:

```text
group 1: 22 passed, 1593 deselected in 4.39s
group 2: 22 passed, 1593 deselected in 7.39s
group 3: 22 passed, 1593 deselected in 5.77s
group 4: 22 passed, 1593 deselected in 4.82s
aggregate: 88 passed, 0 failed
```

## Coverage Notes

The selector covered:

- no-scale copy `warpx2::01_23` candidate positives;
- dense-shared rematerialized `warpx2::01_23` and `warpx2::02_13` rows;
- two-CTA copy codegen for linear and legacy layouts across f32/i32 and
  subword payload types;
- linear tile-permuted copy rows over `128x128b` and `128x256b` atoms;
- `tcgen05.cp` scales `warpx4` direct and scaled-MMAv5-derived rows;
- plain MMAv5 root kinds with linear accumulators;
- scaled-MMAv5 root-format rows across `mxfp8`, `mxfp4`, mixed formats, and
  `nvfp4`.

## Classification

No compiler crash, verifier drift, false unsupported diagnostic, opcode
absence, runtime miscompile, hang, or new independent `FZ-*` bucket was found
in this lane.
