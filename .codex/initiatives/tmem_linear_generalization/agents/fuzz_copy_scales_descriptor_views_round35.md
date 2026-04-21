# Round 35: `tcgen05.copy` Scales + Descriptor-View Runtime Guardrail

Date: 2026-04-21 13:24 UTC
Branch: `codex/tmem`
HEAD: `6623c94d9c77ab2aa1f9fd95936ae23c182c3455`
Scope: discovery only; no backend repairs attempted.

## Objective

Adversarially rerun the checked-in Python/Gluon runtime surface for
`tcgen05.copy` with scale layouts and descriptor-view/subslice composition.
The targeted failure modes were compiler crashes, false unsupported
diagnostics, opcode mismatches, and runtime miscompiles.

## Build

Required rebuild before tests:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Commands and Results

Collected the scale-copy descriptor-view/layout-probe selector:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest --collect-only -q \
  -k 'cp_scales and (descriptor_view or tmem_descriptor or layout_probe or warpx2 or warpx4)' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `31/1615` tests collected.

Ran that selector split across all four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 \
  -k 'cp_scales and (descriptor_view or tmem_descriptor or layout_probe or warpx2 or warpx4)' \
  python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 \
  -k 'cp_scales and (descriptor_view or tmem_descriptor or layout_probe or warpx2 or warpx4)' \
  python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 \
  -k 'cp_scales and (descriptor_view or tmem_descriptor or layout_probe or warpx2 or warpx4)' \
  python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 \
  -k 'cp_scales and (descriptor_view or tmem_descriptor or layout_probe or warpx2 or warpx4)' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `31 passed` (`8/8/8/7` by group).

Collected the scale-specific descriptor-view/subslice selector:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest --collect-only -q \
  -k 'cp_scales and (descriptor_view or chain or indexed or subslice)' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `3/1615` tests collected:

- `test_tmem_runtime_matrix_cp_scales_shared_subslice_layout_rematerializes[0]`
- `test_tmem_runtime_matrix_cp_scales_shared_subslice_layout_rematerializes[64]`
- `test_tmem_runtime_matrix_cp_scales_tmem_descriptor_view_reports_clean_unsupported`

Ran the exact scale-specific selector:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short \
  -k 'cp_scales and (descriptor_view or chain or indexed or subslice)' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `3 passed`.

Collected the adjacent descriptor-view copy slice:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest --collect-only -q \
  -k '(cp_scales or copy) and (descriptor_view or descriptor_chain or chain or indexed_view or subslice_view)' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `19/1615` tests collected.

Ran the adjacent descriptor-view copy slice split across all four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 \
  -k '(cp_scales or copy) and (descriptor_view or descriptor_chain or chain or indexed_view or subslice_view)' \
  python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 \
  -k '(cp_scales or copy) and (descriptor_view or descriptor_chain or chain or indexed_view or subslice_view)' \
  python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 \
  -k '(cp_scales or copy) and (descriptor_view or descriptor_chain or chain or indexed_view or subslice_view)' \
  python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 \
  -k '(cp_scales or copy) and (descriptor_view or descriptor_chain or chain or indexed_view or subslice_view)' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `19 passed` (`5/5/5/4` by group).

## Classification

Aggregate runtime result: `53` checked-in row executions passed across the
three runtime sweeps (`31 + 3 + 19`). The `3` scale-specific descriptor-view
rows are included in the broader `31` row sweep, so unique checked-in coverage
is `31` scale-copy rows plus `19` adjacent descriptor-view copy rows.

No compiler crash, false unsupported diagnostic, opcode mismatch, runtime
miscompile, clean-boundary drift, unexpected xfail/pass transition, or new
independent `FZ-*` bucket was found.

Existing bucket relationship:

- The TMEM descriptor-view scale-copy row remains a clean unsupported boundary,
  not a new false negative.
- The scaled-MMAv5 shared-scale descriptor-view auto-copy rows stayed green and
  continued to check `tcgen05.cp` opcode selection alongside MMA correctness.
- Adjacent no-scale descriptor-view/subslice/indexed `warpx2` copy rows stayed
  green, providing contrast against the scale-copy descriptor-view boundary.

Backend repair remains deferred.
