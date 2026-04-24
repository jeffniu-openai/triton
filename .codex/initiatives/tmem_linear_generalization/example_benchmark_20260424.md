# Example 01/05 Branch-vs-Main Benchmark - 2026-04-24

## Environment

- GPU: NVIDIA GB300/Blackwell Ultra, `CUDA_VISIBLE_DEVICES=0`.
- Branch: `codex/tmem` at `9bee5eb94`.
- Upstream main worktree: `/tmp/triton-main-bench` at `27c402843`.
- Both trees were rebuilt with `make` before benchmarking.
- Separate caches were used: `/tmp/triton-cache-bench-branch-01`, `/tmp/triton-cache-bench-main-01`, `/tmp/triton-cache-bench-branch-05`, `/tmp/triton-cache-bench-main-05`.
- Raw logs and parsed CSVs are in `/tmp/tmem_bench_20260424/`: `branch_01.log`, `main_01.log`, `branch_05.log`, `main_05.log`, `attention_01_branch_vs_main.csv`, `moe_05_branch_vs_main.csv`.

## 01 Attention Forward

- Full common sweep: 112 cells, geomean `0.906x` (-9.4%), wins `7`, losses `105`, worst `0.563x`, best `1.052x`.
- `use_tmem_red=False`: 56 cells, geomean `0.879x` (-12.1%), wins `0`, losses `56`, worst `0.563x`, best `0.961x`.
- `use_tmem_red=True`: 56 cells, geomean `0.934x` (-6.6%), wins `7`, losses `49`, worst `0.582x`, best `1.052x`.

| D | causal | red | dtype | geomean branch/main | min | max |
|---:|:---:|:---:|:---:|---:|---:|---:|
| 64 | False | False | fp16 | 0.884x | 0.642x | 0.942x |
| 64 | False | False | fp8 | 0.878x | 0.731x | 0.913x |
| 64 | False | True | fp16 | 0.917x | 0.673x | 0.976x |
| 64 | False | True | fp8 | 0.948x | 0.820x | 0.983x |
| 64 | True | False | fp16 | 0.842x | 0.584x | 0.928x |
| 64 | True | False | fp8 | 0.867x | 0.563x | 0.949x |
| 64 | True | True | fp16 | 0.916x | 0.663x | 1.026x |
| 64 | True | True | fp8 | 0.899x | 0.582x | 0.988x |
| 128 | False | False | fp16 | 0.906x | 0.833x | 0.955x |
| 128 | False | False | fp8 | 0.870x | 0.778x | 0.899x |
| 128 | False | True | fp16 | 0.943x | 0.879x | 0.992x |
| 128 | False | True | fp8 | 0.949x | 0.861x | 0.967x |
| 128 | True | False | fp16 | 0.888x | 0.718x | 0.950x |
| 128 | True | False | fp8 | 0.897x | 0.638x | 0.961x |
| 128 | True | True | fp16 | 0.932x | 0.761x | 1.008x |
| 128 | True | True | fp8 | 0.968x | 0.675x | 1.052x |

Worst 01 cells:

| ratio | D | causal | red | N_CTX | dtype | branch TFLOPS | main TFLOPS |
|---:|---:|:---:|:---:|---:|:---:|---:|---:|
| 0.563x | 64 | True | False | 1024 | fp8 | 280.93 | 499.28 |
| 0.582x | 64 | True | True | 1024 | fp8 | 291.00 | 500.04 |
| 0.584x | 64 | True | False | 1024 | fp16 | 274.77 | 470.82 |
| 0.638x | 128 | True | False | 1024 | fp8 | 565.00 | 885.82 |
| 0.642x | 64 | False | False | 1024 | fp16 | 506.09 | 788.09 |
| 0.663x | 64 | True | True | 1024 | fp16 | 312.18 | 470.98 |
| 0.673x | 64 | False | True | 1024 | fp16 | 535.02 | 794.61 |
| 0.675x | 128 | True | True | 1024 | fp8 | 598.72 | 886.96 |

Best 01 cells:

| ratio | D | causal | red | N_CTX | dtype | branch TFLOPS | main TFLOPS |
|---:|---:|:---:|:---:|---:|:---:|---:|---:|
| 1.052x | 128 | True | True | 32768 | fp8 | 2128.18 | 2022.10 |
| 1.051x | 128 | True | True | 16384 | fp8 | 2031.70 | 1933.43 |
| 1.042x | 128 | True | True | 65536 | fp8 | 2134.61 | 2048.22 |
| 1.037x | 128 | True | True | 4096 | fp8 | 1581.79 | 1524.69 |
| 1.037x | 128 | True | True | 8192 | fp8 | 1889.45 | 1821.99 |
| 1.026x | 64 | True | True | 65536 | fp16 | 1075.17 | 1047.89 |
| 1.008x | 128 | True | True | 8192 | fp16 | 1445.98 | 1434.86 |
| 0.996x | 128 | True | True | 4096 | fp16 | 1227.64 | 1232.49 |

## 05 MoE BMM1 Fused Gather

- example TFLOPS: 48 batch sizes, geomean `0.885x` (-11.5%), wins `0`, losses `48`, worst `0.775x`, best `0.998x`.
- reference TFLOPS: 48 batch sizes, geomean `0.960x` (-4.0%), wins `21`, losses `27`, worst `0.752x`, best `1.078x`.
- Branch example/reference geomean: `1.166x`; main example/reference geomean: `1.264x`.

Worst 05 example branch/main cells:

| ratio | batch size | branch example TFLOPS | main example TFLOPS | branch ref TFLOPS | main ref TFLOPS |
|---:|---:|---:|---:|---:|---:|
| 0.775x | 4096 | 315.29 | 406.83 | 302.31 | 333.30 |
| 0.802x | 8192 | 654.34 | 816.10 | 641.87 | 595.34 |
| 0.804x | 6144 | 502.85 | 625.24 | 469.11 | 492.28 |
| 0.810x | 2560 | 205.66 | 253.91 | 204.75 | 195.03 |
| 0.817x | 5120 | 419.09 | 513.20 | 409.02 | 414.73 |
| 0.819x | 2048 | 179.16 | 218.85 | 160.28 | 201.48 |
| 0.826x | 3072 | 256.82 | 311.02 | 237.95 | 238.97 |
| 0.839x | 224 | 46.34 | 55.21 | 37.41 | 40.44 |
| 0.840x | 23552 | 1443.96 | 1718.41 | 1344.13 | 1329.03 |
| 0.850x | 11264 | 797.48 | 938.72 | 755.92 | 757.85 |

Best 05 example branch/main cells:

| ratio | batch size | branch example TFLOPS | main example TFLOPS | branch ref TFLOPS | main ref TFLOPS |
|---:|---:|---:|---:|---:|---:|
| 0.998x | 31744 | 1980.76 | 1984.87 | 1784.51 | 1795.21 |
| 0.979x | 28672 | 1753.91 | 1790.81 | 1608.44 | 1616.18 |
| 0.979x | 30720 | 1873.76 | 1913.29 | 1651.81 | 1666.57 |
| 0.969x | 29696 | 1801.85 | 1860.23 | 1616.63 | 1588.44 |
| 0.966x | 1024 | 125.19 | 129.54 | 89.18 | 97.41 |
| 0.964x | 640 | 96.92 | 100.53 | 63.12 | 73.21 |
| 0.960x | 896 | 101.25 | 105.52 | 70.59 | 78.51 |
| 0.947x | 448 | 86.82 | 91.70 | 54.94 | 65.07 |
| 0.941x | 21504 | 1433.74 | 1523.74 | 1268.65 | 1259.61 |
| 0.939x | 512 | 75.92 | 80.89 | 61.22 | 74.70 |

## Commands

```bash
git fetch upstream main
git -C /tmp/triton-main-bench checkout --detach upstream/main
make
cd /tmp/triton-main-bench && make
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-bench-branch-01 PYTHONPATH=/root/code/triton/python python3 /root/code/triton/python/examples/gluon/01-attention-forward.py
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-bench-main-01 PYTHONPATH=/tmp/triton-main-bench/python python3 /tmp/triton-main-bench/python/examples/gluon/01-attention-forward.py
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-bench-branch-05 PYTHONPATH=/root/code/triton/python:/root/code/triton/python/triton_kernels python3 /root/code/triton/python/examples/gluon/05-moe-bmm1-fused-gather.py
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-bench-main-05 PYTHONPATH=/tmp/triton-main-bench/python:/tmp/triton-main-bench/python/triton_kernels python3 /tmp/triton-main-bench/python/examples/gluon/05-moe-bmm1-fused-gather.py
```
