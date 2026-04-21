# Round 35 `ld.red` Descriptor-View Chain Fuzzing

Date: 2026-04-21 13:35 UTC

Scope: discovery/cataloging only. No backend or compiler code was modified.

Required first step:

```bash
make -j8
```

Result: no-op rebuild through
`/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`.

## Summary

This lane targeted `tcgen05.ld.red` descriptor-view chains with row/column
reversals, rank-4/rank-5 identity reshapes, half-row/half-column views, and
`min`/`max` reductions. Cases ran one Python subprocess per row with
`CUDA_VISIBLE_DEVICES = case_index % 4` and stable cache directories
`/tmp/triton-cache-gpu{0..3}`.

Artifacts:

- temporary driver: `/root/tmp/tmem_ldred_descriptor_views_round35_probe.py`
- full summary: `/tmp/tmem_ldred_descriptor_views_round35/summary.json`
- corrected half-view summary:
  `/tmp/tmem_ldred_descriptor_views_round35_half/summary.json`

Aggregate classification:

- direct/rank identity matrix: `40 pass`;
- corrected half-view rerun: `8 pass`, `8` clean unsupported descriptor-view
  diagnostics, `4` clean scalar `.x1` reduction diagnostics, `8` existing
  `FZ-20260421-0022`, and `4` existing `FZ-20260421-0020`;
- checked-in descriptor-chain guardrail:
  `30 passed` split across four GPUs as `8/8/8/6`.

No runtime wrong-result miscompile, opcode mismatch, process abort, or new
independent `FZ-*` bucket was found.

## Commands

Temporary probe:

```bash
PYTHONPATH=.:./python:./python/test/gluon python /root/tmp/tmem_ldred_descriptor_views_round35_probe.py
```

Corrected half-view rerun:

```bash
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location('probe', '/root/tmp/tmem_ldred_descriptor_views_round35_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.OUT_DIR = probe.Path('/tmp/tmem_ldred_descriptor_views_round35_half')
probe.WORKER = probe.OUT_DIR / 'worker.py'
probe.SUMMARY = probe.OUT_DIR / 'summary.json'
probe.CASES = [c for c in probe.CASES if c['shape_kind'] in ('half_row', 'half_col')]
probe.main()
PY
```

Checked-in guardrail split-4:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red and descriptor_chain and not non_f32'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red and descriptor_chain and not non_f32'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red and descriptor_chain and not non_f32'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red and descriptor_chain and not non_f32'
```

## Probe Matrix

Direct and semantically identity descriptor-view chains:

- shapes: `[128, N]`, `N in {64, 128}`;
- layouts: identity, row-reversed rows, column-reversed columns, row+column
  reverse;
- consumers: `view.load_min(...)` and `view.load_max(...)`;
- chains: direct TMEM, rank-4 double permute identity, rank-5 double permute
  identity.

All `40` direct/rank-identity rows passed runtime comparison. Every passing
row emitted one `tcgen05.ld.red` PTX opcode. Rank-4 and rank-5 rows preserved
`ttg.memdesc_subslice` and `ttg.memdesc_reshape` in TTGIR before lowering.

Corrected half-view matrix:

- half-row view: `base.slice(0, M // 2, dim=0)`;
- half-column view: `base.slice(0, N // 2, dim=1)`;
- same layout, `N`, and reduction-op cross product as above.

Result:

- half-column identity and row-reversed rows passed: `8 pass`;
- half-row identity and column-reversed rows reported the existing clean
  unsupported descriptor-view diagnostic: `8 clean unsupported`;
- row+column-reversed half-column rows reported the clean scalar `.x1`
  `tcgen05.ld.red` diagnostic: `4 clean diagnostics`;
- row-reversed and row+column-reversed half-row rows reproduced existing
  `FZ-20260421-0022`: `8` parser/lowering failures with
  `ttng.tmem_load failed to compute TMEM encoding info for reduction`;
- column-reversed half-column rows reproduced existing `FZ-20260421-0020`:
  `4` failures in the later optimize/LLVM pipeline with an MLIR reproducer.

## Classification Notes

The lane did not justify a new `FZ-*` id:

- `FZ-20260421-0022` already covers row-reversed half-row actual `ld.red`
  parse/lowering failures. Round 35 expands it from `load_max` to both `min`
  and `max`, and across `N=64/128`.
- `FZ-20260421-0020` already covers valid half-view descriptor chains that
  fail after parser acceptance. Round 35 adds column-reversed half-column
  `ld.red` rows that reach the allocation/LLVM pipeline before failing.
- The scalar `.x1` and unsupported descriptor-view diagnostics are clean
  boundaries, not false negatives in this lane.

Backend repair remains deferred per the discovery-only fuzzing mandate.
