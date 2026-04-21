# Round 51: TMEM Example Runtime Sanity

Date: 2026-04-21
Branch: `codex/tmem`
Scope: sanity-check TMEM-bearing example/runtime surfaces outside the core TMEM runtime matrix. This round is cataloging only; no backend fixes, checked-in tests, or example code were changed.

## Rebuild

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Discovery

Command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/examples/gluon/*.py
```

Result:

```text
995 tests collected, 1 error in 2.39s
```

The collection error was isolated to `python/examples/gluon/05-moe-bmm1-fused-gather.py`:

```text
ModuleNotFoundError: No module named 'triton_kernels.distributed'
```

Classification: example invocation/path issue, not a TMEM backend failure. The checkout contains the package under `python/triton_kernels/triton_kernels`, and the example collects and runs once `./python/triton_kernels` is added to `PYTHONPATH`.

Follow-up collect command:

```bash
PYTHONPATH=.:./python:./python/triton_kernels pytest --collect-only -q python/examples/gluon/05-moe-bmm1-fused-gather.py
```

Result:

```text
48 tests collected in 1.70s
```

## Correctness Smokes

### Compact TMEM Capability Examples

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/triton_kernels \
  pytest -q -s --tb=short \
  python/examples/gluon/05-tmem-moe-router.py \
  python/examples/gluon/06-tmem-lora-fusion.py \
  python/examples/gluon/08-tmem-layout-as-epilogue.py
```

Result:

```text
37 passed in 23.15s
```

Coverage:
- `05-tmem-moe-router.py`: router projection, top-k wrapper, candidate-head projection, ragged expert panels, and padded-baseline comparisons.
- `06-tmem-lora-fusion.py`: compact LoRA down-projection, fused LoRA update, side projection, gated side output, and padded-baseline comparisons.
- `08-tmem-layout-as-epilogue.py`: direct consumer-order layout stores and canonical-store-plus-reorder comparisons.

Classification: green. No new backend `FZ-*`.

### Fused-Gather MoE Example

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/triton_kernels \
  pytest -q -s --tb=short \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[128-c0]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[1024-c0]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[4096-c0]'
```

Result:

```text
3 passed in 14.81s
```

Classification: green with corrected example `PYTHONPATH`. The bare collect/import failure is a harness packaging issue, not a compiler crash, unsupported TMEM case, verifier issue, or miscompile.

### Older TMEM-Bearing Example Surfaces

Command:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/triton_kernels \
  pytest -q -s --tb=short \
  'python/examples/gluon/01-attention-forward.py::test_op[False-triton-fp16-False-64-1024-32-4]' \
  'python/examples/gluon/01-attention-forward.py::test_op[False-triton-fp16-True-64-1024-32-4]' \
  'python/examples/gluon/02-convolution.py::test_op[0-1-3-3-384-384-64-64-1]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-2-CGA_LAYOUT0-8-0-64-128-64]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-2-CGA_LAYOUT1-8-0-64-128-64]' \
  'python/examples/gluon/03-matmul-multicta.py::test_matmul_matches_torch[100-200-200-4-32-2-2-CGA_LAYOUT2-8-0-64-128-64]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[2-256-256-4-mxfp8-mxfp8-128-128-128]' \
  'python/examples/gluon/04-2cta-block-scale-matmul.py::test_mma_scaled_warp_specialized[1-128-64-5-mxfp8-mxfp8-128-128-128]'
```

Result:

```text
8 passed in 22.94s
```

Coverage:
- attention forward, non-causal and causal fp16 smoke.
- convolution TMA/MMA smoke.
- multi-CTA matmul for 1CTA, 2CTA, and 4CTA CGA layouts.
- 2CTA block-scaled matmul plus a 1CTA scaled-matmul contrast.

Classification: green. No new backend `FZ-*`.

## Lightweight Benchmark Entrypoints

Benchmarks were run only for examples with direct, bounded script entrypoints and small problem sizes. The full fused-gather benchmark was intentionally not launched because its script iterates the complete batch-size sweep.

### Sparse Router / Candidate / Ragged Panels

Command:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/triton_kernels \
  python python/examples/gluon/05-tmem-moe-router.py \
  --mode all --m 512 --k 128 --experts 32 64 --candidates 32 64
```

Result:

```text
TMEM sparse router benchmark
============================
M=512 E=32 K=128 | narrow=0.011 ms | padded E=128=0.013 ms | speedup=1.16x | useful=0.4 TFLOP/s
M=512 E=64 K=128 | narrow=0.013 ms | padded E=128=0.013 ms | speedup=0.99x | useful=0.6 TFLOP/s
M=512 E=32 K=128 + topk | narrow=0.015 ms | padded E=128=0.024 ms | speedup=1.56x | useful=0.3 TFLOP/s
M=512 E=64 K=128 + topk | narrow=0.017 ms | padded E=128=0.022 ms | speedup=1.27x | useful=0.5 TFLOP/s
TMEM candidate-head benchmark
=============================
M=512 C=32 K=128 | compact=0.011 ms | padded C=128=0.013 ms | speedup=1.17x | useful=0.4 TFLOP/s | first_candidate=63
M=512 C=64 K=128 | compact=0.013 ms | padded C=128=0.013 ms | speedup=0.99x | useful=0.7 TFLOP/s | first_candidate=63
TMEM ragged expert panel benchmark
==================================
experts=4 active=3 tokens=512 K=128 | compact=0.087 ms | padded N=128=0.123 ms | speedup=1.43x | useful=0.1 TFLOP/s
```

Classification: benchmark path executes and correctness checks inside the benchmark pass. No new backend `FZ-*`.

### LoRA / Side Projection

Command:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/triton_kernels \
  python python/examples/gluon/06-tmem-lora-fusion.py \
  --m 512 --k 128 --n 128 --mode all --ranks 32 64 --side 32 64
```

Result:

```text
TMEM LoRA adapter benchmark
===========================
M=512 R=32 K=128 N=128 | compact=0.016 ms | padded R=128=0.021 ms | speedup=1.33x | useful=0.5 TFLOP/s
M=512 R=64 K=128 N=128 | compact=0.020 ms | padded R=128=0.020 ms | speedup=1.04x | useful=0.9 TFLOP/s
TMEM MLP side-projection benchmark
==================================
M=512 S=32 K=128 N=128 | compact=0.017 ms | padded S=128=0.020 ms | speedup=1.14x | useful=1.2 TFLOP/s
M=512 S=64 K=128 N=128 | compact=0.019 ms | padded S=128=0.021 ms | speedup=1.10x | useful=1.3 TFLOP/s
```

Classification: benchmark path executes and correctness checks inside the benchmark pass. No new backend `FZ-*`.

### Layout-As-Epilogue

Command:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/triton_kernels \
  python python/examples/gluon/08-tmem-layout-as-epilogue.py --m 128 --n-max 256
```

Result:

```text
TMEM layout-as-epilogue benchmark
=================================
M=128 N=64 tile_n=16 | direct=0.012 ms | canonical+reorder=0.013 ms | speedup=1.10x | bytes=32768
M=128 N=128 tile_n=32 | direct=0.017 ms | canonical+reorder=0.015 ms | speedup=0.85x | bytes=65536
M=128 N=256 tile_n=64 | direct=0.027 ms | canonical+reorder=0.019 ms | speedup=0.70x | bytes=131072
```

Classification: benchmark path executes and correctness checks inside the benchmark pass. No new backend `FZ-*`. Performance remains shape-dependent as already documented by the example.

## Classification Summary

- New backend `FZ-*` needed: no.
- Compiler crashes: none observed.
- Unsupported/verifier failures that should have compiled: none observed.
- Runtime miscompiles: none observed.
- Known FZ bucket reproductions: none in this bounded examples sanity round.
- Example/harness issue: `05-moe-bmm1-fused-gather.py` does not collect with only `PYTHONPATH=.:./python` because `triton_kernels` lives under `python/triton_kernels`. Adding `./python/triton_kernels` makes collection and sampled runtime tests pass.

## Notes

- Stale `__pycache__` entries exist for removed example filenames (`07`, `09`, `10`, `11`), but there are no corresponding source files in `python/examples/gluon`; they were not considered runtime surfaces.
- The full example collection is very large, especially attention, convolution, multi-CTA matmul, and 2CTA scaled matmul. This round used exact nodeids for older examples to keep the sanity check practical while covering representative TMEM-bearing paths.
