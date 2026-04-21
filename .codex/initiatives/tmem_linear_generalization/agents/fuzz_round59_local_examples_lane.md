# Round 59 Local: focused examples runtime lane

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `4cd98c293`
Mode: discovery/cataloging only. No backend, compiler, example, or checked-in test code was edited.

## Scope

This local lane used focused `python/examples/gluon` runtime tests as
application-level TMEM coverage while avoiding the heaviest full attention and
large matmul sweeps. It targeted:

- `08-tmem-layout-as-epilogue.py` direct consumer ordering;
- layout-as-epilogue baseline comparison;
- `05-tmem-moe-router.py` candidate projection ordering;
- ragged expert output correctness.

## Required build

```bash
make -j8
```

Result: `ninja: no work to do`.

## Collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/examples/gluon/08-tmem-layout-as-epilogue.py \
  python/examples/gluon/05-tmem-moe-router.py \
  -k 'layout_epilogue or direct_consumer_order or candidate_projection_matches_selected_vocab_order or ragged_expert_outputs_match_torch'
```

Result: `10/21` collected.

## Runtime split

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/examples/gluon/08-tmem-layout-as-epilogue.py python/examples/gluon/05-tmem-moe-router.py -k 'layout_epilogue or direct_consumer_order or candidate_projection_matches_selected_vocab_order or ragged_expert_outputs_match_torch'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/examples/gluon/08-tmem-layout-as-epilogue.py python/examples/gluon/05-tmem-moe-router.py -k 'layout_epilogue or direct_consumer_order or candidate_projection_matches_selected_vocab_order or ragged_expert_outputs_match_torch'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/examples/gluon/08-tmem-layout-as-epilogue.py python/examples/gluon/05-tmem-moe-router.py -k 'layout_epilogue or direct_consumer_order or candidate_projection_matches_selected_vocab_order or ragged_expert_outputs_match_torch'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/examples/gluon/08-tmem-layout-as-epilogue.py python/examples/gluon/05-tmem-moe-router.py -k 'layout_epilogue or direct_consumer_order or candidate_projection_matches_selected_vocab_order or ragged_expert_outputs_match_torch'
```

Result:

- group 1: `3 passed, 18 deselected`
- group 2: `3 passed, 18 deselected`
- group 3: `3 passed, 18 deselected`
- group 4: `1 passed, 20 deselected`
- aggregate: `10 passed`

## Classification

No compiler crash, verifier drift, false unsupported diagnostic, runtime
miscompile, hang, example-level correctness regression, or new independent
`FZ-*` bucket was found. The selected layout-as-epilogue and MoE/router example
paths remain green.
