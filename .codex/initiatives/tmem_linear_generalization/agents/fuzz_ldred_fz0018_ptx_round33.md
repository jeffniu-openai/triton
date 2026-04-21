# Round 33: FZ-0018 PTX Artifact Capture

Date: 2026-04-21 13:13 UTC
Branch: `codex/tmem`

## Scope

Discovery-only artifact capture for `FZ-20260421-0018`. This follow-up did not
edit backend or test source. It checked whether the failing `ld.red` ptxas
failure can be preserved as standalone dumped PTX rather than only as a Python
runtime subprocess failure.

## Reproducer Path Check

Command:

```bash
rm -f /tmp/tmem_fz0018_reproducer.mlir
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  TRITON_REPRODUCER_PATH=/tmp/tmem_fz0018_reproducer.mlir \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_fz0018_min_round32/ldred_child.py \
  '{"root":"direct","layout":"identity","M":128,"N":512,"num_warps":4,"variant":"auto","red_op":"min"}' \
  > /tmp/tmem_fz0018_reproducer_probe.stdout \
  2> /tmp/tmem_fz0018_reproducer_probe.stderr
```

Result: no `/tmp/tmem_fz0018_reproducer.mlir` was produced. This is expected
because the failure occurs at the ptxas/cubin stage, not during an MLIR pass
crash.

## Kernel Dump

Command:

```bash
rm -rf /tmp/tmem_fz0018_dump
mkdir -p /tmp/tmem_fz0018_dump
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/tmem_fz0018_dump \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_fz0018_min_round32/ldred_child.py \
  '{"root":"direct","layout":"identity","M":128,"N":512,"num_warps":4,"variant":"auto","red_op":"min"}' \
  > /tmp/tmem_fz0018_dump_probe.stdout \
  2> /tmp/tmem_fz0018_dump_probe.stderr
```

Dumped artifacts:

```text
/tmp/tmem_fz0018_dump/USMVLYV4TKIBORO3LKD4YLECH3BNDLCWPOXSGX4OK5FB3F4TA42Q/direct_kernel.ttgir
/tmp/tmem_fz0018_dump/USMVLYV4TKIBORO3LKD4YLECH3BNDLCWPOXSGX4OK5FB3F4TA42Q/direct_kernel.llir
/tmp/tmem_fz0018_dump/USMVLYV4TKIBORO3LKD4YLECH3BNDLCWPOXSGX4OK5FB3F4TA42Q/direct_kernel.ptx
```

Sizes:

```text
65 lines   direct_kernel.ttgir
6963 lines direct_kernel.llir
5544 lines direct_kernel.ptx
```

Standalone ptxas replay:

```bash
/root/code/triton/python/triton/backends/nvidia/bin/ptxas-blackwell \
  -lineinfo -v --regAllocOptLevel=2 --gpu-name=sm_103a \
  /tmp/tmem_fz0018_dump/USMVLYV4TKIBORO3LKD4YLECH3BNDLCWPOXSGX4OK5FB3F4TA42Q/direct_kernel.ptx \
  -o /tmp/tmem_fz0018_dump/direct_kernel.ptx.o
```

Result:

```text
exit: 255
ptxas-blackwell fatal: (C7600) Register allocation failed with register count of '255'.
ptxas-blackwell fatal: Ptx assembly aborted due to errors
```

## Passing Dump Controls

Commands used `TRITON_ALWAYS_COMPILE=1`, `TRITON_KERNEL_DUMP=1`, and fresh
per-case cache/dump directories under `/tmp/tmem_fz0018_dump_compare_pass`.

Controls:

```text
pass_direct_col_reverse M128 N512 warps4 auto min
  PTX: /tmp/tmem_fz0018_dump_compare_pass/pass_direct_col_reverse/VZNA3EAXB36VW7WNVHLGQ7TZX2MO6KZUKFS6C2T5EAH3C3VJM6EQ/direct_kernel.ptx
  ptxas exit: 0
  .ld.red occurrences in PTX: 8
  ptxas summary: Used 96 registers, 6200 byte stack frame, 9720 spill stores, 9740 spill loads

pass_direct_m64 M64 N512 warps4 auto min
  PTX: /tmp/tmem_fz0018_dump_compare_pass/pass_direct_m64/LEQ7OMHY447T7CQ22IGH7TXOCZ257MM56YBAJYM2GVO4F3IJMOYA/direct_kernel.ptx
  ptxas exit: 0
  .ld.red occurrences in PTX: 4
  ptxas summary: Used 255 registers, 1160 byte stack frame, 1172 spill stores, 1184 spill loads
```

## Classification

`FZ-0018` has a standalone failing PTX artifact. The failure is not just a
Python runtime harness property: replaying `ptxas-blackwell` on the dumped PTX
reproduces the register-allocation failure.

The passing `M64xN512` control is useful because it can use `255` registers and
still assemble successfully, so the FZ-0018 failure should not be described as
only "any 255-register kernel fails." The more precise diagnosis remains a
large 4-warp hardware-`ld.red` resource-planning/ptxas boundary where some
legal neighboring resource-heavy hardware-`ld.red` PTX assembles and runs.
