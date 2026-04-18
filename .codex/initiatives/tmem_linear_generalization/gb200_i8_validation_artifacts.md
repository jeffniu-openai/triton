# GB200 Signed i8 MMAv5 Validation Artifacts

Last updated: 2026-04-18 01:36 UTC

This is the reusable artifact/runbook for Gap #1 signed i8 direct MMAv5
validation. It records the exact generated module, remote environment, and
commands used to prove that the current signed i8 `tcgen05.mma.kind::i8`
frontend/lowering path runs correctly on GB200.

Do not store CaaS API keys in this file. Supply `CAAS_API_KEY` in the shell
environment when rerunning the validation.

## Local Artifact Directory

Directory:

```bash
.codex/initiatives/tmem_linear_generalization/experiments/mmav5_i8_remote
```

Files:

- `generate_i8_sm100_ptx.py`: local generator; compiles the Gluon
  `mma_kernel` for `GPUTarget("cuda", 100, 32)`.
- `tcgen05_i8_signed_sm100.ptx`: human/codegen inspection artifact.
- `tcgen05_i8_signed_sm100.cubin`: local ptxas sm100 cubin loaded by default
  on the remote host.
- `tcgen05_i8_signed_sm100.metadata.json`: launch/shape/hash metadata.
- `run_i8_ptx_torch.py`: remote torch runner.
- `ptx_driver.cpp`: tiny torch C++ extension using `dlopen("libcuda.so.1")`
  and CUDA Driver API symbols.
- `run_on_caas_gb200.py`: reusable CaaS launcher for `caas-gpu10` /
  `cudaberry-arm`.
- `README.md`: compact local/remote usage notes.

Hashes and sizes from the validated artifact:

```text
PTX sha256    012c73541aaaf373d8c346cf6a4763c337ed8c714dc1742b6f68b5c96e5aed7f
CUBIN sha256  5523626a5be4d52afe4f816a91ebfdb5d15359dcb9b44c0aae84fb0a23d61770
CUBIN bytes   97072
PTX target    sm_100a
PTX version   9.1
kernel        mma_kernel
opcode        tcgen05.mma.cta_group::1.kind::i8
descriptor    136316064
shape         M=128, N=128, K=32
block         128 threads
shared bytes  8204
```

## Local Regeneration

From `/root/code/triton`:

```bash
python3 .codex/initiatives/tmem_linear_generalization/experiments/mmav5_i8_remote/generate_i8_sm100_ptx.py
python3 .codex/initiatives/tmem_linear_generalization/experiments/mmav5_i8_remote/run_i8_ptx_torch.py --dry-run
```

The generator writes both PTX and cubin. Keep the cubin because the validated
remote driver rejected PTX JIT for `.version 9.1` with
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`; cubin loading is the repeatable execution
path.

## Remote Environment

Validated remote:

```text
cluster: caas-gpu10
endpoint: https://caas-gpu10.ace-research.openai.org
image: cudaberry-arm
gpu type: GB200
container TTL: 1200 seconds (CaaS rejected 3600)
file write timeout cap: 60 seconds
```

Remote hardware evidence from the check:

```text
NVIDIA GB200, 10.0, 00000008:01:00.0
NVIDIA GB200, 10.0, 00000009:01:00.0
NVIDIA GB200, 10.0, 00000018:01:00.0
NVIDIA GB200, 10.0, 00000019:01:00.0
```

Execution probe:

```text
torch 2.6.0+openai.040da295524.branchsuffix.oai.cuda.128.os.noble.builderversion.5
cuda_available True
device_count 1
0 NVIDIA GB200 (10, 0)
```

## One-Command Rerun

Install the CaaS package if needed:

```bash
cd ~/code/openai
oaipkg install caas
```

Then from `/root/code/triton`, with `CAAS_API_KEY` already set:

```bash
python3 .codex/initiatives/tmem_linear_generalization/experiments/mmav5_i8_remote/run_on_caas_gb200.py
```

Useful variants:

```bash
# Probe four GB200s and then run the same one-kernel validation.
python3 .codex/initiatives/tmem_linear_generalization/experiments/mmav5_i8_remote/run_on_caas_gb200.py --gpus 4

# Hardware/torch probe only.
python3 .codex/initiatives/tmem_linear_generalization/experiments/mmav5_i8_remote/run_on_caas_gb200.py --skip-artifact-run

# Force PTX JIT for diagnostic purposes. This failed in the validated remote
# environment with CUDA_ERROR_UNSUPPORTED_PTX_VERSION.
python3 .codex/initiatives/tmem_linear_generalization/experiments/mmav5_i8_remote/run_on_caas_gb200.py --module ptx
```

Expected passing output:

```text
PASS signed i8 tcgen05.mma sm100 PTX matches torch int32 matmul
```

## Classification

Gap #1 signed i8 direct MMAv5 is supported on GB200 for the current
frontend/lowering path. Remaining i8 work is separate: unsigned/per-operand
signedness and integer saturation need explicit IR/frontend exposure decisions.
