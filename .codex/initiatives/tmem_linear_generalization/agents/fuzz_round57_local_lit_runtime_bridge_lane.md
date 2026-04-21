# Round 57 Local: structural-fuzzer and lit/runtime bridge lane

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `1ae7c6130`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code was edited.

## Scope

This local lane bridged checked-in Python structural-fuzzer sentinels with
nearby lit verifier/conversion coverage. It targeted:

- dynamic TMEM `memdesc_index` control-flow and runtime-index boundaries;
- 256-row lifted parent allocator crash sentinels;
- `ld.red` row/column optimizer and direct-index allocator sentinels;
- scaled-MMAv5 accumulator subslice dynamic control-flow sentinel;
- TMEM invalid/ops/layout lit verifier and conversion coverage.

The lane was intentionally catalog-only; backend repairs remain deferred while
fuzzing continues to find or sharpen buckets.

## Required build

```bash
make -j8
```

Result: `ninja: no work to do`.

## Python structural-fuzzer selector

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'generic_pass or ldst_256row or ldred_twocta_rowcol_optimizer_crash or ldred_1cta_direct_index_allocator_crash or scaled_mma_acc_subslice_control_flow'
```

Result: `15/33` collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass or ldst_256row or ldred_twocta_rowcol_optimizer_crash or ldred_1cta_direct_index_allocator_crash or scaled_mma_acc_subslice_control_flow'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass or ldst_256row or ldred_twocta_rowcol_optimizer_crash or ldred_1cta_direct_index_allocator_crash or scaled_mma_acc_subslice_control_flow'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass or ldst_256row or ldred_twocta_rowcol_optimizer_crash or ldred_1cta_direct_index_allocator_crash or scaled_mma_acc_subslice_control_flow'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass or ldst_256row or ldred_twocta_rowcol_optimizer_crash or ldred_1cta_direct_index_allocator_crash or scaled_mma_acc_subslice_control_flow'
```

Result:

- group 1: `4 xfailed, 29 deselected`
- group 2: `4 xfailed, 29 deselected`
- group 3: `4 xfailed, 29 deselected`
- group 4: `3 xfailed, 30 deselected`
- aggregate: `15 xfailed`

The xfails revalidated existing buckets:

- `FZ-20260421-0001`: runtime/dynamic TMEM `memdesc_index` reaches LLVM
  conversion as illegal op;
- `FZ-20260421-0002`: helper-returned/layout-conversion-pressure descriptor
  view wrong-result sentinels;
- `FZ-20260421-0007`: scaled-MMAv5 `use_acc` accumulator subslice selected
  through dynamic control flow;
- `FZ-20260421-0008`: 2CTA indexed `ld.red` row/column optimizer crash;
- `FZ-20260421-0009`: 1CTA direct-index `ld.red` allocator assertion.

No xfail unexpectedly passed or failed differently.

## Lit verifier/conversion bridge

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
lit -v test/TritonNvidiaGPU/invalid.mlir test/TritonNvidiaGPU/ops.mlir test/TritonNvidiaGPU/tmem_layouts.mlir
```

Result:

```text
Total Discovered Tests: 3
  Passed: 3 (100.00%)
```

The selected lit files cover nearby TMEM encoding verifier negatives,
high-rank view-chain ops, and TMEM load/reduce layout lowering.

## Classification

No new independent `FZ-*` bucket was found. This lane revalidated existing
structural-fuzzer sentinels and green lit verifier/conversion guardrails.
Backend repair remains deferred.
