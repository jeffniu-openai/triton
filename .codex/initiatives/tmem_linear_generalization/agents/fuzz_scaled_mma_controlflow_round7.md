# Round 7 Lane A: scaled-MMAv5 accumulator control-flow expansion

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery-only; no backend/compiler repair attempted.
- Focus: expand `FZ-20260421-0007` around scaled-MMAv5 `use_acc`
  accumulator subslices selected through dynamic `if`, loop, and
  helper-returned paths.

## Harness

- Temporary pytest harness:
  `/tmp/tmem_scaled_mma_controlflow_round7_probe.py`
- Result log:
  `/tmp/tmem_scaled_mma_controlflow_round7_results.jsonl`
- Coverage:
  - modes: `direct_low`, `direct_high`, `subslice_if`, `subslice_loop`,
    `subslice_helper`, `indexed_if`, `indexed_loop`, `indexed_helper`;
  - `N in {32,64,128}`, `K in {128,256}`;
  - formats: `mxfp8xmxfp8`, `mxfp8xmxfp4`, `mxfp4xmxfp8`,
    `mxfp4xmxfp4`, `nvfp4xnvfp4`;
  - selector values `0` and `1` for dynamic modes.
- The pytest cases classify outcomes and write JSONL records. A pytest
  `passed` result means the classifier completed; backend failures are in the
  JSONL as `miscompile` or `exception`.

## Validation

- Required rebuild before tests:
  `make -j8`
  - result: `ninja: no work to do.`
- Harness syntax:
  `PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_scaled_mma_controlflow_round7_probe.py`
  - result: passed.
- Collection:
  `PYTHONPATH=.:./python:./python/test/gluon pytest --collect-only -q /tmp/tmem_scaled_mma_controlflow_round7_probe.py`
  - result: `410` nodeids.
- Four-GPU split sweep:
  - GPU 0:
    `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 1 /tmp/tmem_scaled_mma_controlflow_round7_probe.py`
    -> `103 passed, 307 deselected in 40.78s`;
  - GPU 1:
    `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_scaled_mma_controlflow_round7_probe.py`
    -> `103 passed, 307 deselected in 40.97s`;
  - GPU 2:
    `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 3 /tmp/tmem_scaled_mma_controlflow_round7_probe.py`
    -> `103 passed, 307 deselected in 52.59s`;
  - GPU 3:
    `CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 4 /tmp/tmem_scaled_mma_controlflow_round7_probe.py`
    -> `101 passed, 309 deselected in 53.49s`.

## Sweep Summary

Corrected full-sweep JSONL rows: `410`.

- Status counts:
  - `pass`: `140`;
  - `miscompile`: `210`;
  - `exception`: `60`.
- Passing controls:
  - `direct_low`: `30`;
  - `direct_high`: `20`;
  - `subslice_if` selector `1`: `30`;
  - `subslice_helper` selector `1`: `30`;
  - `indexed_if` selector `1`: `30`.
- Miscompiles:
  - `subslice_if` selector `0`: `30`;
  - `subslice_helper` selector `0`: `30`;
  - `subslice_loop` selectors `0` and `1`: `60`;
  - `indexed_if` selector `0`: `30`;
  - `indexed_loop` selectors `0` and `1`: `60`.
- Exceptions:
  - `indexed_helper`: `60`, all compiler pass failures from dynamic
    `ttg.memdesc_index` illegal lowering in the helper-returned indexed view.
    These overlap the existing `FZ-20260421-0001` dynamic-index bucket and do
    not receive a new id here.

## Fresh Exact Confirmations

- `subslice_if`, `mxfp8xmxfp8`, `N=32`, `K=128`, selector `0`:
  `miscompile`, `1520 / 4096` mismatches, `132` NaNs,
  `4` scaled-MMA ops, opcode match true.
- Adjacent `subslice_if` selector `1` for the same shape:
  `pass`, `0 / 4096` mismatches.
- Direct low and direct high controls for the same shape:
  both `pass`, `0 / 4096` mismatches.
- `subslice_if`, `mxfp8xmxfp8`, `N=64`, `K=256`, selector `0`:
  `miscompile`, `4301 / 8192` mismatches, `420` NaNs,
  `8` scaled-MMA ops, opcode match true.
- `subslice_if`, `mxfp8xmxfp8`, `N=128`, `K=256`, selector `0`:
  `miscompile`, `11669 / 16384` mismatches, `2357` NaNs,
  `8` scaled-MMA ops, opcode match true.
- `indexed_if`, `mxfp8xmxfp8`, `N=64`, `K=128`, selector `0`:
  `miscompile`, `2441 / 8192` mismatches, `258` NaNs,
  `4` scaled-MMA ops, opcode match true.
- Adjacent `indexed_if` selector `1` for the same shape:
  `pass`, `0 / 8192` mismatches.
- `subslice_helper`, `mxfp8xmxfp4`, `N=128`, `K=128`, selector `0`:
  `miscompile`, `4547 / 16384` mismatches, `264` NaNs,
  `4` scaled-MMA ops, opcode match true.
- `subslice_loop`, `mxfp8xmxfp4`, `N=128`, `K=128`, selector `0`:
  `miscompile`, `5363 / 16384` mismatches, `801` NaNs,
  `4` scaled-MMA ops, opcode match true.

## Classification

This is a broad expansion of `FZ-20260421-0007`, not a new independent
finding. Direct low/high accumulator subslices pass, and high-selector
dynamic `if` / helper / indexed rows pass. Low-selector dynamic paths
miscompile across every probed `N`, `K`, and feasible scaled format that
reached runtime. The loop forms also miscompile because the dynamic loop
construction lands on the low accumulator view in the corrected harness.

The `indexed_helper` compiler failures are stable but overlap the existing
dynamic `memdesc_index` illegal-lowering bucket (`FZ-20260421-0001`), so the
central catalog should record them as overlap evidence only.

No new stable non-overlapping `FZ-*` id was assigned in this lane.
