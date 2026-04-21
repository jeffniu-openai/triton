# Round 39 MMAv5/TMA Runtime Guardrail

Date: 2026-04-21 14:05 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Surface

This lane targeted checked-in runtime rows that combine two-CTA MMAv5 with
descriptor views, indexed/subslice accumulator views, and TMA-fed TF32
transposed-B descriptors:

- `test_tmem_runtime_matrix_mma_twocta_indexed_acc_view`
- `test_tmem_runtime_matrix_mma_twocta_acc_subslice_view_plain_kinds`
- `test_tmem_runtime_matrix_mma_twocta_tma_tf32_b_transposed_descriptor`
- `test_tmem_runtime_matrix_mma_twocta_tma_tf32_b_transposed_descriptor_use_acc`

The selector intentionally avoided broad `plain_kind`/`tma` keyword matching,
which over-selected most of the runtime matrix.

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_twocta_tma_tf32_b_transposed_descriptor or mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds'
```

Result: `54/1615` tests collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_twocta_tma_tf32_b_transposed_descriptor or mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_twocta_tma_tf32_b_transposed_descriptor or mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_twocta_tma_tf32_b_transposed_descriptor or mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_twocta_tma_tf32_b_transposed_descriptor or mma_twocta_indexed_acc_view or mma_twocta_acc_subslice_view_plain_kinds'
```

## Results

- Group 1: `14 passed, 1601 deselected`
- Group 2: `14 passed, 1601 deselected`
- Group 3: `14 passed, 1601 deselected`
- Group 4: `12 passed, 1603 deselected`

Aggregate: `54 passed`.

## Classification

No compiler crash, false unsupported diagnostic, opcode absence, runtime
miscompile, unexpected pass/fail transition, or new independent `FZ-*` bucket
was observed. This is a positive guardrail for two-CTA MMAv5 descriptor-view
ownership and TMA descriptor lowering after the Round 39 dynamic/scaled lanes.
