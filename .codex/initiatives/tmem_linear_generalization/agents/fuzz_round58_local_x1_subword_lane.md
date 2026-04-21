# Round 58 Local: x1 and subword load/store lane

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `7c499fb4d`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code was edited.

## Scope

This local lane covered x1 and subword TMEM load/store boundaries in
`python/test/gluon/test_tmem_runtime_matrix.py`, separate from the active
Round 58 scaled, high-CGA, and plain-MMAv5 worker lanes. Coverage included:

- subword pack/unpack for `f16`, `bf16`, `i16`, and `i8`;
- subword descriptor-chain roundtrips;
- x1 subword roundtrips for linear and legacy pack/unpack layouts;
- two-CTA x1 subword roundtrips and descriptor-chain roundtrips;
- x1 `f32` and `i32` roundtrips and descriptor-chain roundtrips;
- clean unsupported x1 variants for `16x128b`.

## Required build

```bash
make -j8
```

Result: `ninja: no work to do`.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst_x1 or subword_descriptor_chain or ldst_subword'
```

Result: `48/1615` collected.

## Runtime split

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_x1 or subword_descriptor_chain or ldst_subword'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_x1 or subword_descriptor_chain or ldst_subword'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_x1 or subword_descriptor_chain or ldst_subword'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_x1 or subword_descriptor_chain or ldst_subword'
```

Result:

- group 1: `12 passed, 1603 deselected`
- group 2: `12 passed, 1603 deselected`
- group 3: `12 passed, 1603 deselected`
- group 4: `12 passed, 1603 deselected`
- aggregate: `48 passed`

## Classification

No compiler crash, verifier drift, false unsupported diagnostic, clean-boundary
drift, runtime miscompile, hang, or new independent `FZ-*` bucket was found.
x1 and subword load/store, descriptor-chain, two-CTA, and clean unsupported
boundaries stayed green.
