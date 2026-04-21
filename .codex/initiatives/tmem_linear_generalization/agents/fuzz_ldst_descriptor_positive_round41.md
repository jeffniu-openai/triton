# Round 41 ld/st Descriptor Positive Guardrail

Date: 2026-04-21 14:13 UTC

Scope: discovery/cataloging only. No backend code was modified.

## Surface

This lane targeted checked-in positive `ld/st` descriptor roundtrip rows for
descriptor compositions, multidim slices, rank-5 roots, and two-CTA descriptor
roots. It intentionally excluded report/resource rows and subword/x1/scales
rows, which are covered by separate clean-boundary and subword descriptor-chain
lanes.

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ldst_descriptor_roundtrip or ldst_descriptor_compositions or ldst_twocta_descriptor_compositions or ldst_descriptor_multidim_slice_positive or ldst_descriptor_rank5_roundtrip or ldst_twocta_descriptor_rank5_roundtrip) and not reports and not resource and not x1 and not subword and not scales'
```

Result: `28/1615` tests collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(ldst_descriptor_roundtrip or ldst_descriptor_compositions or ldst_twocta_descriptor_compositions or ldst_descriptor_multidim_slice_positive or ldst_descriptor_rank5_roundtrip or ldst_twocta_descriptor_rank5_roundtrip) and not reports and not resource and not x1 and not subword and not scales'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k '(ldst_descriptor_roundtrip or ldst_descriptor_compositions or ldst_twocta_descriptor_compositions or ldst_descriptor_multidim_slice_positive or ldst_descriptor_rank5_roundtrip or ldst_twocta_descriptor_rank5_roundtrip) and not reports and not resource and not x1 and not subword and not scales'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k '(ldst_descriptor_roundtrip or ldst_descriptor_compositions or ldst_twocta_descriptor_compositions or ldst_descriptor_multidim_slice_positive or ldst_descriptor_rank5_roundtrip or ldst_twocta_descriptor_rank5_roundtrip) and not reports and not resource and not x1 and not subword and not scales'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k '(ldst_descriptor_roundtrip or ldst_descriptor_compositions or ldst_twocta_descriptor_compositions or ldst_descriptor_multidim_slice_positive or ldst_descriptor_rank5_roundtrip or ldst_twocta_descriptor_rank5_roundtrip) and not reports and not resource and not x1 and not subword and not scales'
```

## Results

- Group 1: `5 passed, 2 skipped, 1608 deselected`
- Group 2: `1 passed, 6 skipped, 1608 deselected`
- Group 3: `7 skipped, 1608 deselected`
- Group 4: `7 skipped, 1608 deselected`

Aggregate: `6 passed, 22 skipped`.

## Classification

No compiler crash, false unsupported diagnostic, clean-boundary drift,
unexpected pass/fail transition, runtime miscompile, or new independent
`FZ-*` bucket was observed. The selected positive descriptor-roundtrip surface
remains stable under the current runtime skip conditions.
