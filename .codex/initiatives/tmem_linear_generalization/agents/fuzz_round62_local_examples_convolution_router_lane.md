# Round 62 Local: convolution and router examples lane

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `0fc0fa73f`
Mode: discovery/cataloging only. No backend, compiler, example, or checked-in
test code was edited.

## Scope

This lane covered the remaining `python/examples/gluon` surfaces that were not
part of the Round 60 examples breadth exact set or the Round 61 local examples
edge exact set:

- convolution example rows across filter sizes, spatial sizes, and split-K;
- TMEM MoE router projection, top-k, padded-baseline, candidate projection,
  and ragged expert output checks.

## Required build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Broad collection

```bash
PYTHONPATH=.:./python:./python/triton_kernels pytest --collect-only -q \
  python/examples/gluon/02-convolution.py \
  python/examples/gluon/05-tmem-moe-router.py
```

Result: `64 tests collected`.

## Focused exact selector

Selected exact nodeids:

```text
python/examples/gluon/02-convolution.py::test_op[1-2-3-3-384-384-64-64-1]
python/examples/gluon/02-convolution.py::test_op[1-2-5-5-416-416-64-64-128]
python/examples/gluon/02-convolution.py::test_op[1-2-4-4-384-384-64-64-128]
python/examples/gluon/05-tmem-moe-router.py::test_router_projection_matches_torch[128-32]
python/examples/gluon/05-tmem-moe-router.py::test_router_projection_matches_torch[256-64]
python/examples/gluon/05-tmem-moe-router.py::test_router_topk_matches_torch[64]
python/examples/gluon/05-tmem-moe-router.py::test_router_padded_baseline_matches_narrow[32]
python/examples/gluon/05-tmem-moe-router.py::test_candidate_projection_matches_selected_vocab_order[128-64]
python/examples/gluon/05-tmem-moe-router.py::test_candidate_projection_matches_selected_vocab_order[256-32]
python/examples/gluon/05-tmem-moe-router.py::test_candidate_padded_baseline_matches_compact[64]
python/examples/gluon/05-tmem-moe-router.py::test_ragged_expert_outputs_match_torch
python/examples/gluon/05-tmem-moe-router.py::test_ragged_expert_padded_baseline_matches_compact
```

Collection result: `12 tests collected`.

## Runtime split

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/triton_kernels xargs -a /tmp/round62_examples_exact.txt pytest -q -s --tb=short --splits 4 --group 1
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/triton_kernels xargs -a /tmp/round62_examples_exact.txt pytest -q -s --tb=short --splits 4 --group 2
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/triton_kernels xargs -a /tmp/round62_examples_exact.txt pytest -q -s --tb=short --splits 4 --group 3
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/triton_kernels xargs -a /tmp/round62_examples_exact.txt pytest -q -s --tb=short --splits 4 --group 4
```

Results:

- group 1: `3 passed, 9 deselected`
- group 2: `3 passed, 9 deselected`
- group 3: `3 passed, 9 deselected`
- group 4: `3 passed, 9 deselected`
- aggregate: `12 passed, 0 failed, 0 skipped`

## Classification

- Convolution rows: `3 passed`; no compiler crash, codegen failure, runtime
  wrong result, or split-K shape regression.
- MoE router rows: `9 passed`; projection, top-k, padded-baseline,
  candidate-projection, and ragged-output paths stayed green.

No compiler crash, verifier drift, false unsupported diagnostic, runtime
miscompile, hang, example-level correctness regression, or new independent
`FZ-*` bucket was found. No repairs were attempted.
