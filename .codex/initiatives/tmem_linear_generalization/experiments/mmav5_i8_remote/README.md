# GB200 signed i8 MMAv5 PTX smoke test

This directory packages a no-Triton-runtime smoke test for Gap #1.  Generate
PTX locally, copy this directory to a GB200 host, and run the PTX against
torch-owned tensors in the remote container.

Local generation from the Triton checkout:

```bash
python3 .codex/initiatives/tmem_linear_generalization/experiments/mmav5_i8_remote/generate_i8_sm100_ptx.py
```

Remote run on GB200:

```bash
python3 run_i8_ptx_torch.py --launcher cpp
```

Artifact-only validation, which does not launch the kernel:

```bash
python3 run_i8_ptx_torch.py --dry-run
```

If the remote container lacks CUDA headers for the tiny C++ extension, the
same launch can be done without compiling the shim:

```bash
python3 run_i8_ptx_torch.py --launcher ctypes
```

Expected result:

```text
PASS signed i8 tcgen05.mma sm100 PTX matches torch int32 matmul
```
