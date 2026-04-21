# Round 61 Local: examples edge-parameter lane

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `a35ca5bc6`
Mode: discovery/cataloging only. No backend, compiler, example, or checked-in
test code was edited.

## Scope

This lane sampled `python/examples/gluon` edge parameters that did not overlap
with the Round 60 examples breadth exact set. It focused on:

- long-sequence attention with causal/non-causal and fp8/fp16 combinations;
- multicta matmul with alternate CGA layouts and resource-heavy tiles;
- 2CTA block-scale matmul with fp4/fp8 dtype combinations and larger `K`;
- MoE fused gather at the smallest and largest collected token counts;
- LoRA padded-baseline comparisons;
- layout-as-epilogue direct/baseline edge rows.

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
  python/examples/gluon/01-attention-forward.py \
  python/examples/gluon/03-matmul-multicta.py \
  python/examples/gluon/04-2cta-block-scale-matmul.py \
  python/examples/gluon/05-moe-bmm1-fused-gather.py \
  python/examples/gluon/06-tmem-lora-fusion.py \
  python/examples/gluon/08-tmem-layout-as-epilogue.py
```

Result: `979 tests collected`.

## Focused exact selector

The selected exact nodeids were:

```text
python/examples/gluon/01-attention-forward.py::test_op[True-triton-fp8-False-128-8192-32-4]
python/examples/gluon/01-attention-forward.py::test_op[False-triton-fp16-False-64-4096-32-4]
python/examples/gluon/01-attention-forward.py::test_op[True-triton-fp16-True-64-8192-32-4]
python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-2-CGA_LAYOUT0-8-0-128-256-128]
python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-4-CGA_LAYOUT2-8-0-128-256-128]
python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp4-mxfp4-128-128-128]
python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp4-mxfp8-500-600-1152]
python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp8-mxfp8-256-256-4096]
python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[128-c0]
python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[31744-c0]
python/examples/gluon/06-tmem-lora-fusion.py::test_lora_padded_baseline_matches_compact[32]
python/examples/gluon/06-tmem-lora-fusion.py::test_lora_padded_baseline_matches_compact[64]
python/examples/gluon/06-tmem-lora-fusion.py::test_side_projection_padded_baseline_matches_compact[32]
python/examples/gluon/06-tmem-lora-fusion.py::test_side_projection_padded_baseline_matches_compact[64]
python/examples/gluon/08-tmem-layout-as-epilogue.py::test_direct_consumer_order_matches_pytorch[256-64]
python/examples/gluon/08-tmem-layout-as-epilogue.py::test_layout_epilogue_baseline_matches_direct[128-32]
```

Collection result: `16 tests collected`.

## Runtime split

Each split used the same exact nodeid list with one pytest process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/triton_kernels xargs -a /tmp/round61_examples_exact.txt pytest -q -s --tb=short --splits 4 --group 1
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/triton_kernels xargs -a /tmp/round61_examples_exact.txt pytest -q -s --tb=short --splits 4 --group 2
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/triton_kernels xargs -a /tmp/round61_examples_exact.txt pytest -q -s --tb=short --splits 4 --group 3
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/triton_kernels xargs -a /tmp/round61_examples_exact.txt pytest -q -s --tb=short --splits 4 --group 4
```

Results:

- group 1: `4 passed, 12 deselected`
- group 2: `3 passed, 1 skipped, 12 deselected`
- group 3: `4 passed, 12 deselected`
- group 4: `4 passed, 12 deselected`
- aggregate: `15 passed, 1 skipped`

Skip reason rerun:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/triton_kernels xargs -a /tmp/round61_examples_exact.txt pytest -q -s -rs --tb=short --splits 4 --group 2
```

Result:

```text
SKIPPED [1] python/examples/gluon/03-matmul-multicta.py:722: Out of resources
3 passed, 1 skipped, 12 deselected
```

## Classification

- Attention edge rows: `3 passed`; no recurrence of prior attention failures
  or new runtime miscompile.
- Multicta matmul edge rows: `1 passed, 1 skipped`; the skip is an existing
  example resource guard, not a compiler crash or verifier failure.
- 2CTA block-scale matmul edge rows: `3 passed`; no scaled-MMAv5 operand
  miscompile or false unsupported diagnostic surfaced.
- MoE fused gather extremes: `2 passed`; the smallest and largest collected
  token-count rows stayed correct.
- LoRA padded-baseline rows: `4 passed`; compact-vs-padded comparisons stayed
  green.
- Layout-as-epilogue rows: `2 passed`; no direct/baseline ordering mismatch.

No compiler crash, verifier drift, false unsupported diagnostic, runtime
miscompile, hang, example-level correctness regression, or new independent
`FZ-*` bucket was found. No repairs were attempted.
