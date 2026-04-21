# Round 60 Lane A: examples breadth beyond Round 59 focus

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `b5d0cd781`
Mode: discovery/cataloging only. No backend, compiler, example, or checked-in
test code was edited.

## Scope

This lane covered bounded Python examples in `python/examples/gluon` that were
not part of the Round 59 focused examples lane. Round 59 covered
`05-tmem-moe-router.py` and `08-tmem-layout-as-epilogue.py`; this lane covered
attention, LoRA fusion, 2CTA block-scale matmul, matmul-multicta, and MoE
fused gather.

The fused-gather example uses the separate `triton_kernels` package under the
repo, so all collection/runtime commands used:

```bash
PYTHONPATH=.:./python:./python/triton_kernels
```

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

## Collection

Broad collection command used to inspect candidate nodeids:

```bash
PYTHONPATH=.:./python:./python/triton_kernels pytest --collect-only -q \
  python/examples/gluon/01-attention-forward.py \
  python/examples/gluon/03-matmul-multicta.py \
  python/examples/gluon/04-2cta-block-scale-matmul.py \
  python/examples/gluon/05-moe-bmm1-fused-gather.py \
  python/examples/gluon/06-tmem-lora-fusion.py
```

Result: `974 tests collected`.

Focused exact selector:

```bash
PYTHONPATH=.:./python:./python/triton_kernels pytest --collect-only -q \
  'python/examples/gluon/01-attention-forward.py::test_op[True-triton-fp8-True-128-1024-32-4]' \
  'python/examples/gluon/01-attention-forward.py::test_op[False-triton-fp16-True-128-2048-32-4]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-4-CGA_LAYOUT1-8-1-64-128-64]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-4-CGA_LAYOUT2-8-1-64-128-64]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp8-mxfp4-500-600-640]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp4-mxfp8-256-256-704]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[160-c0]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[512-c0]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_lora_down_projection_matches_torch[256-64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_lora_update_matches_torch[64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_side_projection_matches_torch[256-64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_side_gated_output_matches_torch[64]'
```

Result: `12 tests collected`.

## Runtime split

Each split used the same exact 12 nodeids with one process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/triton_kernels pytest -q -s --tb=short --splits 4 --group 1 \
  'python/examples/gluon/01-attention-forward.py::test_op[True-triton-fp8-True-128-1024-32-4]' \
  'python/examples/gluon/01-attention-forward.py::test_op[False-triton-fp16-True-128-2048-32-4]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-4-CGA_LAYOUT1-8-1-64-128-64]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-4-CGA_LAYOUT2-8-1-64-128-64]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp8-mxfp4-500-600-640]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp4-mxfp8-256-256-704]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[160-c0]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[512-c0]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_lora_down_projection_matches_torch[256-64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_lora_update_matches_torch[64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_side_projection_matches_torch[256-64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_side_gated_output_matches_torch[64]'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/triton_kernels pytest -q -s --tb=short --splits 4 --group 2 \
  'python/examples/gluon/01-attention-forward.py::test_op[True-triton-fp8-True-128-1024-32-4]' \
  'python/examples/gluon/01-attention-forward.py::test_op[False-triton-fp16-True-128-2048-32-4]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-4-CGA_LAYOUT1-8-1-64-128-64]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-4-CGA_LAYOUT2-8-1-64-128-64]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp8-mxfp4-500-600-640]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp4-mxfp8-256-256-704]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[160-c0]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[512-c0]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_lora_down_projection_matches_torch[256-64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_lora_update_matches_torch[64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_side_projection_matches_torch[256-64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_side_gated_output_matches_torch[64]'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/triton_kernels pytest -q -s --tb=short --splits 4 --group 3 \
  'python/examples/gluon/01-attention-forward.py::test_op[True-triton-fp8-True-128-1024-32-4]' \
  'python/examples/gluon/01-attention-forward.py::test_op[False-triton-fp16-True-128-2048-32-4]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-4-CGA_LAYOUT1-8-1-64-128-64]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-4-CGA_LAYOUT2-8-1-64-128-64]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp8-mxfp4-500-600-640]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp4-mxfp8-256-256-704]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[160-c0]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[512-c0]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_lora_down_projection_matches_torch[256-64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_lora_update_matches_torch[64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_side_projection_matches_torch[256-64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_side_gated_output_matches_torch[64]'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/triton_kernels pytest -q -s --tb=short --splits 4 --group 4 \
  'python/examples/gluon/01-attention-forward.py::test_op[True-triton-fp8-True-128-1024-32-4]' \
  'python/examples/gluon/01-attention-forward.py::test_op[False-triton-fp16-True-128-2048-32-4]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-4-CGA_LAYOUT1-8-1-64-128-64]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-4-CGA_LAYOUT2-8-1-64-128-64]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp8-mxfp4-500-600-640]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp4-mxfp8-256-256-704]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[160-c0]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[512-c0]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_lora_down_projection_matches_torch[256-64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_lora_update_matches_torch[64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_side_projection_matches_torch[256-64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_side_gated_output_matches_torch[64]'
```

Results:

- group 1: `3 passed, 9 deselected in 20.22s`
- group 2: `2 passed, 1 skipped, 9 deselected in 7.34s`
- group 3: `3 passed, 9 deselected in 14.64s`
- group 4: `3 passed, 9 deselected in 6.76s`
- aggregate: `11 passed, 1 skipped`

The group-2 skip was rerun with `-rs`:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/triton_kernels pytest -q -s -rs --tb=short --splits 4 --group 2 \
  'python/examples/gluon/01-attention-forward.py::test_op[True-triton-fp8-True-128-1024-32-4]' \
  'python/examples/gluon/01-attention-forward.py::test_op[False-triton-fp16-True-128-2048-32-4]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-4-CGA_LAYOUT1-8-1-64-128-64]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-4-CGA_LAYOUT2-8-1-64-128-64]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp8-mxfp4-500-600-640]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp4-mxfp8-256-256-704]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[160-c0]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[512-c0]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_lora_down_projection_matches_torch[256-64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_lora_update_matches_torch[64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_side_projection_matches_torch[256-64]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_side_gated_output_matches_torch[64]'
```

Skip reason:

```text
SKIPPED [1] python/examples/gluon/04-2cta-block-scale-matmul.py:808: fp4 packed tensor descriptor requires K to be a multiple of 128
```

## Classification

- Attention examples: `2 passed`; no known Round 51 attention exact failure
  reproduced and no new correctness issue appeared.
- Matmul-multicta examples: `2 passed`; no recurrence of historical
  examples-lane multicta reds.
- 2CTA block-scale matmul: `1 passed, 1 skipped`; the skip is a pre-existing
  example guard for fp4 packed descriptors with non-128-multiple `K`, not a
  compiler/backend failure.
- MoE fused gather: `2 passed`; path-sensitive `triton_kernels` harness
  requirement remained satisfied.
- LoRA fusion: `4 passed`; compact LoRA and side-projection checks stayed
  green.

No compiler crash, verifier drift, false unsupported diagnostic, runtime
miscompile, hang, example-level correctness regression, or new independent
`FZ-*` bucket was found. No repairs were attempted.
