# Round 57 Local: direct higher-rank TMEM load/store lane

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `63924ddb3`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code was edited.

## Scope

This local lane sampled direct higher-rank TMEM load/store and load-reduce
positive rows in `python/test/gluon/test_tmem_runtime_matrix.py`, separate from
the descriptor-chain and structural-fuzzer bridge lanes. It covered:

- direct higher-rank `get_reg_layout` positives;
- direct higher-rank load/store replay;
- direct higher-rank `load_red` replay.

The selector excluded `reports` rows and did not overlap the Round 57 copy
view-boundary worker lane.

## Required build

```bash
make -j8
```

Result: `ninja: no work to do`.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst_direct_higher_rank and not reports'
```

Result: `4/1615` collected.

## Runtime split

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_direct_higher_rank and not reports'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_direct_higher_rank and not reports'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_direct_higher_rank and not reports'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'ldst_direct_higher_rank and not reports'
```

Result:

- group 1: `1 passed, 1614 deselected`
- group 2: `1 passed, 1614 deselected`
- group 3: `1 passed, 1614 deselected`
- group 4: `1 passed, 1614 deselected`
- aggregate: `4 passed`

## Classification

No compiler crash, verifier drift, false unsupported diagnostic, runtime
miscompile, hang, or new independent `FZ-*` bucket was found. Direct
higher-rank load/store and load-reduce positives remain green.
