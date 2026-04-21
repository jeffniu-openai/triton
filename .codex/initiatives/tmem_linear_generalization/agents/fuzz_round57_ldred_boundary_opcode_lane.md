# Round 57 Lane E: `ld.red` Clean-Boundary and Opcode Selection

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code
was modified.

This lane adversarially tested `ld.red` clean-boundary and opcode-selection
behavior without repeating the completed Round 56 local mixed `ld.red` selector.
The focus was M64/resource-ish rows, non-identity row bases, descriptor-view
half-row/half-column chains, and software-vs-hardware reduction classification.

No new independent `FZ-*` bucket is proposed.

## Preflight

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Existing Classifications Recorded

- `FZ-20260421-0012`: M64 f32 hardware `ld.red` with an effective
  non-identity row basis fails during Gluon parse/lowering with
  `ttng.tmem_load` unsupported destination layout. Column-only M64
  permutations and identity-row split-N variants remain green.
- `FZ-20260421-0017`: encoded `i64`/`f64` non-reduction TMEM load/store
  lowering asserts on `bitwidth == 32`; non-f32 `ld.red` is not part of this
  bucket because reduction loads stay cleanly limited to f32 or use software
  reduction where the Python contract requests that path.
- `FZ-20260421-0018`: large 4-warp hardware `ld.red` resource/ptxas boundary,
  including direct or same-footprint descriptor-view large tiles such as
  `M128xN512` and `M256xN256`, while neighboring large legal rows can still
  assemble and run.
- `FZ-20260421-0020`: valid half-column column-reversed `ld.red`
  descriptor-view chains can pass parser acceptance and fail later in the
  optimize/LLVM pipeline.
- `FZ-20260421-0021`: unit-rank half-column non-reduction `ld/st`
  descriptor-view aborts with a memdesc-shape / `MemDescType` invariant
  failure. This lane did not reproduce it because the tested half-view consumer
  was `ld.red`, not non-reduction `ld/st`.
- `FZ-20260421-0022`: row-reversed or row+column-reversed half-row actual
  `ld.red` descriptor-view chains fail at parser/lowering time while computing
  reduction TMEM encoding info.

## M64 / Resource-ish Boundary Slice

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and ld_red and (m64 or resource) and not reports'
```

Result:

```text
39/1615 tests collected (1576 deselected) in 2.49s
```

Runtime commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix and ld_red and (m64 or resource) and not reports'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix and ld_red and (m64 or resource) and not reports'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix and ld_red and (m64 or resource) and not reports'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'test_tmem_runtime_matrix and ld_red and (m64 or resource) and not reports'
```

Results:

```text
group 1: 10 passed, 1605 deselected in 4.19s
group 2: 6 passed, 4 failed, 1605 deselected in 4.93s
group 3: 10 passed, 1605 deselected in 4.13s
group 4: 7 passed, 2 failed, 1606 deselected in 4.53s
aggregate: 33 passed, 6 failed
```

The six failures are the existing `FZ-20260421-0012` rows:

```text
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]
test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]
```

All six emitted the known unsupported-destination diagnostic:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
requested layout direct-lowering details:
Failed to lower TMEM load/store: unsupported dst layout
where out dims are: [row (size 128), col (size 32 or 128)]
```

No `FZ-20260421-0018` ptxas/resource failure appeared in this checked-in M64
slice.

## Non-M64 Non-Identity Hardware `ld.red` Slice

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and ld_red and (tile_permuted or col_permuted or row_permuted or rowcol_permuted_n_sweep or expanded_row or single_cta_block) and not m64 and not reports and not resource'
```

Result:

```text
66/1615 tests collected (1549 deselected) in 2.17s
```

Runtime commands used the same four-GPU split shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and ld_red and (tile_permuted or col_permuted or row_permuted or rowcol_permuted_n_sweep or expanded_row or single_cta_block) and not m64 and not reports and not resource'
```

Results:

```text
group 1: 17 passed, 1598 deselected in 8.56s
group 2: 17 passed, 1598 deselected in 30.56s
group 3: 17 passed, 1598 deselected in 4.33s
group 4: 15 passed, 1600 deselected in 4.35s
aggregate: 66 passed, 0 failed
```

This covered tile-permuted, column-permuted, row-permuted, row+column
permuted N sweeps, expanded row/row+column layouts, and single-CTA block
layouts outside M64. No hardware `ld.red` opcode absence, software fallback,
runtime miscompile, clean-boundary drift, or verifier crash appeared.

## Software-vs-Hardware Classification Slice

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and ld_red and (non_f32_descriptor_chain_uses_software_reduce or scales_uses_software_reduce or mixed_linear_layout_uses_software_reduce or single_cta_block_linear_layout) and not reports and not resource and not m64'
```

Result:

```text
48/1615 tests collected (1567 deselected) in 2.37s
```

Runtime commands used the same four-GPU split shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'test_tmem_runtime_matrix and ld_red and (non_f32_descriptor_chain_uses_software_reduce or scales_uses_software_reduce or mixed_linear_layout_uses_software_reduce or single_cta_block_linear_layout) and not reports and not resource and not m64'
```

Results:

```text
group 1: 12 passed, 1603 deselected in 4.37s
group 2: 12 passed, 1603 deselected in 4.36s
group 3: 12 passed, 1603 deselected in 4.29s
group 4: 12 passed, 1603 deselected in 4.24s
aggregate: 48 passed, 0 failed
```

This slice stayed outside the Round 56 local mixed selector. It rechecked:

- non-f32 descriptor-chain software reductions;
- int8 scale-layout software reductions;
- mixed linear layouts that intentionally use software reduction; and
- single-CTA block f32 hardware reductions.

No software-vs-hardware classification drift appeared. Non-f32 descriptor and
scale rows continued to use software reduction without `.ld.red`; the f32
single-CTA block rows continued to compile and execute correctly.

## Descriptor-View Half-Row / Half-Column Replay

Temporary replay command:

```bash
PYTHONPATH=.:./python:./python/test/gluon python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location('probe', '/root/tmp/tmem_ldred_descriptor_views_round35_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.OUT_DIR = probe.Path('/tmp/tmem_round57_ldred_half_views')
probe.WORKER = probe.OUT_DIR / 'worker.py'
probe.SUMMARY = probe.OUT_DIR / 'summary.json'
probe.CASES = [c for c in probe.CASES if c['shape_kind'] in ('half_row', 'half_col')]
probe.main()
PY
```

Result:

```text
count: 32
pass: 8
FZ-20260421-0020: 4
harness_parse_failure: 8
runtime_or_compile_exception_unclassified: 12
summary: /tmp/tmem_round57_ldred_half_views/summary.json
```

Manual inspection of the 12 rows labeled unclassified by the old temporary
harness showed no new bucket:

- `8` half-row row-reversed or row+column-reversed rows are existing
  `FZ-20260421-0022`, failing at parser/lowering time with
  `ttng.tmem_load failed to compute TMEM encoding info for reduction`.
- `4` half-column row+column-reversed rows are clean scalar `.x1`
  diagnostics:
  `tcgen05.ld.red requires at least an .x2 message shape`.

Effective classification for the 32-row half-view replay:

```text
8 pass
8 clean unsupported descriptor-view diagnostics / harness parse failures
4 clean scalar .x1 diagnostics
8 existing FZ-20260421-0022
4 existing FZ-20260421-0020
0 new independent FZ-*
```

## Classification

This lane found no compiler crash, verifier drift, false unsupported
diagnostic, opcode-selection regression, software/hardware reduction
classification drift, runtime miscompile, or new independent `FZ-*` bucket.

The new evidence is catalog-only:

- `FZ-20260421-0012` remains the sole failing checked-in M64 boundary in this
  lane, still isolated to f32 M64 `ld.red` with non-identity row basis.
- Non-M64 non-identity row/column hardware `ld.red` rows stayed green.
- Software-reduce contract rows stayed software-only where intended, and
  adjacent f32 block rows stayed hardware-capable.
- Half-view descriptor chains remained within existing `FZ-20260421-0020`,
  existing `FZ-20260421-0022`, or clean diagnostics.
- `FZ-20260421-0017` and `FZ-20260421-0021` were recorded as adjacent
  non-reduction `ld/st` boundaries, not reproduced as `ld.red` failures here.

Backend repairs remain deferred per the active fuzzing campaign instructions.
