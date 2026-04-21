# Round 59 Local: no-scale copy codegen and minimal scaled-MMA opcode lane

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `8ee469bd0`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code was edited.

## Scope

This local lane sampled compact opcode/codegen sentinels without rerunning the
Round 58 scales-copy lane. Coverage came from
`python/test/gluon/test_tmem_runtime_matrix.py` and included:

- no-scale `4x256b` refresh codegen;
- two-CTA no-scale copy codegen across linear/legacy layouts and
  `f32`/`i32`/subword dtypes;
- two-CTA `128x128b` no-scale copy codegen;
- minimal scaled-MMAv5 runtime row.

## Required build

```bash
make -j8
```

Result: `ninja: no work to do`.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and codegen) or mma_scaled_minimal'
```

Result: `27/1615` collected.

## Runtime split

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(cp_no_scales and codegen) or mma_scaled_minimal'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k '(cp_no_scales and codegen) or mma_scaled_minimal'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k '(cp_no_scales and codegen) or mma_scaled_minimal'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k '(cp_no_scales and codegen) or mma_scaled_minimal'
```

Result:

- group 1: `7 passed, 1608 deselected`
- group 2: `7 passed, 1608 deselected`
- group 3: `7 passed, 1608 deselected`
- group 4: `6 passed, 1609 deselected`
- aggregate: `27 passed`

## Classification

No compiler crash, verifier drift, opcode/codegen assertion drift, false
unsupported diagnostic, runtime miscompile, hang, or new independent `FZ-*`
bucket was found. No-scale copy codegen and minimal scaled-MMAv5 opcode
sentinels stayed green.
