# Round 54 Local Split-N / x1 / Subword Runtime Lane

Date: 2026-04-21
Branch: `codex/tmem`
Checkpoint base: `105498b06 Record Round 54 local TMA TMEM lane`

This local lane exercised split-N, explicit `16x32bx2`, x1, and subword
TMEM runtime rows in `python/test/gluon/test_tmem_runtime_matrix.py`. The goal
was to stress narrow packet choices, packed/unpacked subword forms, descriptor
chains, and 1CTA/2CTA x1 rows without mixing in known-red M64 row/column
`ld.red` cases.

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

Initial broad collection:

```bash
PYTHONPATH=.:./python \
pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (ldst_x1_i32 or ldst_x1_f32 or ldst_x1_subword or splitn or explicit_16x32bx2)'
```

Result:

```text
101/1615 tests collected (1514 deselected) in 2.22s
```

The runtime lane excluded `rowcol_permuted_explicit` so the positive sweep did
not include the existing M64 row/column `ld.red` known-red rows.

## Runtime Split

Command shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
PYTHONPATH=.:./python \
pytest -q -s --tb=short \
  --splits 4 --group <group> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and (ldst_x1_i32 or ldst_x1_f32 or ldst_x1_subword or splitn or explicit_16x32bx2) and not rowcol_permuted_explicit'
```

Results:

```text
group 1: 25 passed, 1590 deselected in 5.72s
group 2: 25 passed, 1590 deselected in 4.48s
group 3: 25 passed, 1590 deselected in 4.85s
group 4: 23 passed, 1592 deselected in 8.77s
aggregate: 98 passed, 0 failed
```

## Coverage Notes

The selector covered:

- split-N immediate and auto-selection rows for `f32` and `i32`;
- explicit `16x32bx2` rows matched against split-N lowering;
- row/column-permuted split-N positive rows excluding known-red M64
  row/column `ld.red` cases;
- f16 and i8 x1 subword roundtrips, including packed/unpacked forms;
- 1CTA and 2CTA x1 `f32`/`i32` direct and descriptor-chain rows;
- clean unsupported diagnostics for x1 unsupported variants.

## Classification

No compiler crash, verifier drift, false unsupported diagnostic, clean-boundary
drift, runtime miscompile, hang, or new independent `FZ-*` bucket was found in
this lane.
