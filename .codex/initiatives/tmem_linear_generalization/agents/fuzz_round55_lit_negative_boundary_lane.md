# Round 55 TMEM lit negative-boundary lane C

Date: 2026-04-21
Branch: `codex/tmem`
HEAD at start: `20c481e5a`
Scope: compiler-only/lit negative-boundary drift for TMEM verifier diagnostics,
clean unsupported cases, unencoded and non-distributed operands, 64-bit
bitwidth, proxy/mbarrier, relayout, allocation, and tensor-memory conversion.
Backend repairs were intentionally not attempted.

## Build

```bash
cd /root/code/triton
make -j8
```

Result: pass/no-op.

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: no work to do.
```

Build dir:

```bash
PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'
```

Result: `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`.

## Checked-in lit sweep

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
lit -v \
  test/Conversion/lower_tensor_memory_to_llvm.mlir \
  test/Conversion/tritongpu_to_llvm_blackwell.mlir \
  test/Conversion/tritonnvidiagpu_to_llvm.mlir \
  test/Conversion/relayout_tritongpu.mlir \
  test/TritonGPU/proxy_fence_insertion.mlir \
  test/TritonGPU/memdesc-subview-split.mlir \
  test/TritonGPU/hoist-tmem-alloc.mlir \
  test/TritonGPU/promote-lhs-to-tmem.mlir \
  test/TritonGPU/invalid.mlir \
  test/TritonNvidiaGPU/tmem_layouts.mlir \
  test/TritonNvidiaGPU/interleave_tmem.mlir \
  test/TritonNvidiaGPU/membar.mlir \
  test/TritonNvidiaGPU/membar-cluster.mlir \
  test/TritonNvidiaGPU/invalid.mlir \
  test/TritonNvidiaGPU/test_tensor_memory_allocation.mlir \
  test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir
```

Result:

```text
Total Discovered Tests: 16
  Passed: 15
  Failed: 1
```

Failure: `TRITON :: Conversion/relayout_tritongpu.mlir`, assertion abort in
`TritonGPUDialect::toLinearLayout` / `computeTMemLdStEncodingInfoImpl` from
`dyn_cast on a non-existent value`.

Classification: existing `FZ-20260421-0016` unencoded/non-distributed TMEM
operand verifier crash. No new independent candidate.

## Proxy, mbarrier, TMA, and invalid diagnostic lit sweep

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
lit -v \
  test/TritonNvidiaGPU/mma_lowering.mlir \
  test/TritonNvidiaGPU/ops.mlir \
  test/TritonNvidiaGPU/tma_lowering.mlir \
  test/TritonGPU/fence-inserstion.mlir \
  test/Analysis/test-membar-ttng.mlir \
  test/Conversion/tma_to_llvm.mlir \
  test/Conversion/tma_multicast_to_llvm.mlir \
  test/Conversion/tensor_memory_to_llvm.mlir \
  2>&1 | tee /tmp/tmem_round55_lane_c_lit_extra.log
```

Result:

```text
lit warning: input 'test/Conversion/tensor_memory_to_llvm.mlir' contained no tests
Total Discovered Tests: 7
  Passed: 7
```

Classification: proxy/mbarrier insertion, TMA conversion, NvidiaGPU operation
lowering, and invalid diagnostic guardrails stayed green. The no-tests warning
is a path inventory issue, not a backend result.

## Direct Blackwell conversion probe

Command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt /root/code/triton/test/Conversion/tritongpu_to_llvm_blackwell.mlir \
  -split-input-file --convert-triton-gpu-to-llvm=compute-capability=100 -cse \
  >/tmp/tmem_round55_lane_c_blackwell.out \
  2>/tmp/tmem_round55_lane_c_blackwell.err
wc -l /tmp/tmem_round55_lane_c_blackwell.out /tmp/tmem_round55_lane_c_blackwell.err
```

Result:

```text
exit=0
30805 /tmp/tmem_round55_lane_c_blackwell.out
0     /tmp/tmem_round55_lane_c_blackwell.err
```

Classification: pass. Checked-in Blackwell tensor-memory conversion remains
green for the representative TMEM/MMAv5/copy/wait/commit surface.

## Round 38 minimized compiler-boundary corpus replay

Corpus: `/tmp/tmem_compiler_boundaries_round38/*.mlir`

Corrected command pattern:

```bash
OPT=/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt
ROOT=/tmp/tmem_compiler_boundaries_round38

"$OPT" "$f" -split-input-file --mlir-disable-threading
"$OPT" "$f" -split-input-file --mlir-disable-threading \
  --triton-nvidia-optimize-tmem-layouts --allow-unregistered-dialect
"$OPT" "$f" -split-input-file --mlir-disable-threading \
  --triton-tensor-memory-allocation --allow-unregistered-dialect
"$OPT" "$f" -split-input-file --mlir-disable-threading \
  --triton-nvidia-optimize-tmem-layouts \
  --allocate-shared-memory-nv='compute-capability=100 ptx-version=87' \
  --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' \
  --convert-nv-gpu-to-llvm --allow-unregistered-dialect
```

Artifacts:

- `/tmp/tmem_round55_lane_c_boundary_results.tsv` for verify/optimize/
  tmem-allocation modes plus an initial invalid lower run.
- `/tmp/tmem_round55_lane_c_boundary_lower_corrected.tsv` for corrected lower
  mode.
- `/tmp/tmem_round55_lane_c_*.log` for per-row stderr/stdout.

Note: the first lower-mode loop split `ptx-version=87` as a positional
argument, producing invalid `Too many positional arguments specified!` rows.
Those rows are not used in the counts below; the corrected lower command above
is the source of truth.

Corrected aggregate across 14 cases x 4 modes:

```text
PASS:            18
CLEAN_DIAG:      24
ILLEGAL_OP:       5
ABORT_OR_CRASH:   9
```

Case classification:

```text
bad_linear_dup_clean                   clean diag in all 4 modes
bad_linear_zero_clean                  clean diag in all 4 modes
copy_dynamic_dst_warpx2_fz0001         pass verify/optimize/tmem_alloc; illegal op in lower
copy_tmem_src_wrong_space_clean        clean diag in all 4 modes
dynamic_idx_acc_mma_fz0001             pass verify/optimize/tmem_alloc; illegal op in lower
dynamic_idx_load_live_fz0001           pass verify/optimize/tmem_alloc; illegal op in lower
dynamic_idx_scales_mma_scaled_fz0001   pass verify/optimize/tmem_alloc; illegal op in lower
dynamic_idx_store_live_fz0001          pass verify/optimize/tmem_alloc; illegal op in lower
high_rank_const_index_chain_pass       clean unsupported diagnostic in all 4 modes
high_rank_dynamic_second_index_fz0001  clean unsupported diagnostic in all 4 modes
i64_chain_load_fz0017                  abort/crash in all 4 modes
ldred_non_f32_clean                    clean diagnostic in all 4 modes
ldred_reduction_unencoded_fz0016       pass verify/optimize/tmem_alloc; abort/crash in lower
unencoded_store_fz0016                 abort/crash in all 4 modes
```

Existing bucket classification:

- `FZ-20260421-0001`: five dynamic descriptor-index rows still pass earlier
  modes and stop as late illegal `ttg.memdesc_index` operations in lower mode.
- `FZ-20260421-0016`: unencoded store remains verifier/parse-time abort, and
  unencoded `ld.red` reduction-result remains a late lower-mode crash.
- `FZ-20260421-0017`: encoded `i64` chain load remains a 64-bit TMEM
  bitwidth assertion in every replayed mode.
- Clean boundaries stayed clean for malformed linear encodings, wrong memory
  space, high-rank unsupported view chains, and non-f32 `ld.red`.

No new independent `FZ-*` candidate was found.

## Round 24 FZ-0016 parse-only corpus replay

Command:

```bash
OPT=/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt
ROOT=/tmp/tmem_fz0016_round24
for f in "$ROOT"/*.mlir; do
  "$OPT" "$f" -split-input-file --mlir-disable-threading \
    >/tmp/tmem_round55_lane_c_fz0016_round24_$(basename "$f" .mlir).log 2>&1
done
```

Result over 55 available MLIR files:

```text
ABORT_OR_CRASH: 15
CLEAN_DIAG:     40
```

Classification:

- Existing `FZ-20260421-0016`: all abort rows are the known unencoded
  `ttng.tmem_alloc` operand verifier/encoding-info crash family.
- Clean diagnostics stayed clean for encoded operand controls and invalid
  scale-bitwidth rows.
- No new independent `FZ-*`.

## Minimized known repro controls

`FZ-20260421-0016` command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt /tmp/tmem_round51_fz0016_min.mlir \
  -split-input-file \
  -convert-triton-to-tritongpu='target=cuda:100 num-warps=4 enable-source-remat=true' \
  -relayout-tritongpu \
  >/tmp/tmem_round55_lane_c_fz0016.out \
  2>/tmp/tmem_round55_lane_c_fz0016.err
```

Result: `exit=134`,
`Assertion detail::isPresent(Val) && "dyn_cast on a non-existent value" failed`.

`FZ-20260421-0017` command:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt /tmp/tmem_round51_fz0017_min.mlir \
  --convert-triton-gpu-to-llvm=compute-capability=100 -cse \
  >/tmp/tmem_round55_lane_c_fz0017.out \
  2>/tmp/tmem_round55_lane_c_fz0017.err
```

Result: `exit=134`, `Assertion bitwidth == 32 failed`.

f64 bitwidth contrast:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
sed 's/xi64/xf64/g; s/@i64_chain_load_fz0017/@f64_chain_load_round55_control/g' \
  /tmp/tmem_round51_fz0017_min.mlir \
  > /tmp/tmem_round55_lane_c_f64_chain_load.mlir
bin/triton-opt /tmp/tmem_round55_lane_c_f64_chain_load.mlir \
  --convert-triton-gpu-to-llvm=compute-capability=100 -cse \
  >/tmp/tmem_round55_lane_c_f64.out \
  2>/tmp/tmem_round55_lane_c_f64.err
```

Result: `exit=134`, `Assertion bitwidth == 32 failed`.

Classification: same owner surface as existing `FZ-20260421-0017`; f64 is not
a new independent bucket.

f32 same-shape control:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
sed 's/xi64/xf32/g; s/@i64_chain_load_fz0017/@f32_chain_load_round55_control/g' \
  /tmp/tmem_round51_fz0017_min.mlir \
  > /tmp/tmem_round55_lane_c_f32_chain_load_control.mlir
bin/triton-opt /tmp/tmem_round55_lane_c_f32_chain_load_control.mlir \
  --convert-triton-gpu-to-llvm=compute-capability=100 -cse \
  >/tmp/tmem_round55_lane_c_f32.out \
  2>/tmp/tmem_round55_lane_c_f32.err
```

Result: `exit=1`,
`failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal`.

Classification: clean late illegal-op diagnostic for the same indexed TMEM
view-chain shape, consistent with the existing `FZ-20260421-0001` family. It
does not hit the bitwidth assertion.

## Summary

- Required `make -j8`: pass/no-op.
- `ninja triton-opt`: pass/no-op.
- Checked-in TMEM/proxy/mbarrier/relayout/allocation/conversion lit sweep:
  `15 passed, 1 failed`; the failure is existing `FZ-20260421-0016`.
- Extra proxy/mbarrier/TMA/invalid lit sweep: `7 passed`; one no-tests warning
  for an inventory path.
- Direct Blackwell conversion probe: exit `0`.
- Round 38 compiler-boundary corpus replay: `18 pass`, `24 clean diagnostic`,
  `5` existing late illegal-op rows, and `9` existing abort/crash rows.
- Round 24 parse-only FZ-0016 corpus replay: `15` existing abort/crash rows
  and `40` clean diagnostics.
- Minimized repros revalidated existing `FZ-20260421-0016` and
  `FZ-20260421-0017`; f64 broadens the same bitwidth surface, while f32 stays
  on the clean indexed-view-chain boundary.

No backend code was changed. No new independent `FZ-*` candidate was found in
Round 55 lane C.
