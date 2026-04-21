# Round 34 Local: Copy warpx2 Compiler-Probe Attempt

Date: 2026-04-21
Branch: `codex/tmem`

## Scope

Follow-up to the Round 33 compiler-boundary note that future copy probes should
use CGA-compatible shared layouts so they reach the copy planner instead of
stopping at context-layout verification. This lane stayed discovery-only and
did not edit backend or checked-in tests.

## Runtime-Dumped Positive Control

Command:

```bash
rm -rf /tmp/tmem_copy_warpx2_dump_round34
mkdir -p /tmp/tmem_copy_warpx2_dump_round34
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-copy-warpx2-dump \
  TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 \
  TRITON_DUMP_DIR=/tmp/tmem_copy_warpx2_dump_round34 \
  PYTHONPATH=.:./python \
  pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2_01_23_twocta_positive and f32'
```

Result:

```text
1 passed, 1614 deselected
```

Dumped artifacts:

```text
/tmp/tmem_copy_warpx2_dump_round34/M5FHXFTOSIBMHCRC7EAGTJRZPGI72LN5SCBJ5SFVGW7GIEFRTRQQ/tmem_copy_no_scales_warpx2_twocta_kernel.ttgir
/tmp/tmem_copy_warpx2_dump_round34/M5FHXFTOSIBMHCRC7EAGTJRZPGI72LN5SCBJ5SFVGW7GIEFRTRQQ/tmem_copy_no_scales_warpx2_twocta_kernel.llir
/tmp/tmem_copy_warpx2_dump_round34/M5FHXFTOSIBMHCRC7EAGTJRZPGI72LN5SCBJ5SFVGW7GIEFRTRQQ/tmem_copy_no_scales_warpx2_twocta_kernel.ptx
/tmp/tmem_copy_warpx2_dump_round34/M5FHXFTOSIBMHCRC7EAGTJRZPGI72LN5SCBJ5SFVGW7GIEFRTRQQ/tmem_copy_no_scales_warpx2_twocta_kernel.cubin
/tmp/tmem_copy_warpx2_dump_round34/M5FHXFTOSIBMHCRC7EAGTJRZPGI72LN5SCBJ5SFVGW7GIEFRTRQQ/tmem_copy_no_scales_warpx2_twocta_kernel.sass
```

The dumped TTGIR contains a CGA-compatible two-CTA module:

```text
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32}
```

The dumped PTX contains the expected copy instruction:

```text
tcgen05.cp.cta_group::2.warpx2::01_23.64x128b
```

## Clean-Unsupported Control

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-copy-warpx2-dump2 \
  TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 \
  TRITON_DUMP_DIR=/tmp/tmem_copy_warpx2_dump_round34 \
  PYTHONPATH=.:./python \
  pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2_02_13_twocta_candidate_reports_clean_unsupported and f32'
```

Result:

```text
1 passed, 1614 deselected
```

This row catches the expected clean unsupported diagnostic internally, so no
additional kernel dump was produced. It remains covered by checked-in runtime
clean-boundary assertions.

## Standalone triton-opt Attempt

Commands:

```bash
BUILD=/root/code/triton/build/cmake.linux-aarch64-cpython-3.12
OPT=$BUILD/bin/triton-opt
TTGIR=/tmp/tmem_copy_warpx2_dump_round34/M5FHXFTOSIBMHCRC7EAGTJRZPGI72LN5SCBJ5SFVGW7GIEFRTRQQ/tmem_copy_no_scales_warpx2_twocta_kernel.ttgir

$OPT "$TTGIR" --triton-nvidia-optimize-tmem-layouts --allow-unregistered-dialect \
  > /tmp/tmem_copy_warpx2_compiler_round34/positive.optimize.mlir \
  2> /tmp/tmem_copy_warpx2_compiler_round34/positive.optimize.log

$OPT "$TTGIR" \
  --triton-nvidia-optimize-tmem-layouts \
  --allocate-shared-memory-nv='compute-capability=100 ptx-version=87' \
  --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' \
  --convert-nv-gpu-to-llvm \
  --allow-unregistered-dialect \
  > /tmp/tmem_copy_warpx2_compiler_round34/positive.lower.mlir \
  2> /tmp/tmem_copy_warpx2_compiler_round34/positive.lower.log
```

Results:

```text
optimize exit: 0
lower exit: 139
```

The lowering attempt also crashes with `compute-capability=103`. The stack is
in `TensorMemoryAllocOpConversion::matchAndRewrite` while converting
`ttng.tmem_alloc`.

## Classification

No new independent `FZ-*` bucket is assigned from this lane.

The runtime/JIT path for the same two-CTA `warpx2::01_23` row compiles,
executes, dumps LLIR/PTX/cubin/SASS, and emits the expected `tcgen05.cp`
instruction. The standalone `triton-opt` lowering crash is therefore cataloged
as a compiler-probe harness limitation until a faithful reproduction of the JIT
pipeline proves otherwise.

Useful artifact for future probe work: the dumped positive TTGIR under
`/tmp/tmem_copy_warpx2_dump_round34/.../tmem_copy_no_scales_warpx2_twocta_kernel.ttgir`
is a concrete CGA-compatible copy source that reaches the copy planner in the
JIT path.
