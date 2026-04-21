# Round 47: proxy/mbarrier ordering fuzzing

Date: 2026-04-21
Branch: `codex/tmem`

This lane adversarially tested TMEM proxy, mbarrier, and fence ordering around
TMA, `ttng.tmem_copy`, scaled copy, and MMAv5 sequences. The goal was discovery
and classification only; no backend fixes or checked-in test changes were made.

## Summary

No new `FZ-*` bucket is needed.

Checked-in runtime and lit coverage stayed green across the proxy-adjacent
surfaces tested here:

- TMA multicast, TMA gather/scatter, MMAv5 multicast commit, scaled-MMAv5
  direct multicast barrier, and TMA-fed MMAv5 shared-input sequencing:
  `184 passed, 48 skipped`.
- Runtime-matrix TMEM copy/scaled-copy/MMAv5 proxy-adjacent selector:
  `213 passed`.
- Core scaled-copy linear-accumulator controls:
  `9 passed`.
- Checked-in proxy fence lit test:
  `1 passed`.

The only failures came from disposable saved probes/reproducers and are existing
`FZ-20260421-0014`: the proxy-fence insertion pass cannot place a fence for
sequential cross-CTA mbarrier intervals when an earlier interval is waited
before a later cross-CTA mbarrier init/use interval. This replay sharpens the
known classification but does not create a new bucket.

## Commands And Results

Rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Core proxy/TMA/MMAv5/mbarrier selector collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q --collect-only python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tcgen05_mma_scaled_direct_multicast_barrier or tma_gather_scatter_multi_cta or tma_multicast_copy'
```

Result: `232/18114` selected for the execution selector below.

Core proxy/TMA/MMAv5/mbarrier execution, split across four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tcgen05_mma_scaled_direct_multicast_barrier or tma_gather_scatter_multi_cta or tma_multicast_copy'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tcgen05_mma_scaled_direct_multicast_barrier or tma_gather_scatter_multi_cta or tma_multicast_copy'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tcgen05_mma_scaled_direct_multicast_barrier or tma_gather_scatter_multi_cta or tma_multicast_copy'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or tcgen05_mma_multicast_commit or tcgen05_mma_scaled_direct_multicast_barrier or tma_gather_scatter_multi_cta or tma_multicast_copy'
```

Results:

```text
group 1: 49 passed, 9 skipped, 18056 deselected
group 2: 49 passed, 9 skipped, 18056 deselected
group 3: 44 passed, 14 skipped, 18056 deselected
group 4: 42 passed, 16 skipped, 18056 deselected
total:   184 passed, 48 skipped
```

Core scaled-copy linear-accumulator controls:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short python/test/gluon/test_core.py \
  -k 'mma_scaled_tcgen05_copy_linear_acc or tcgen05_mma_scaled_direct_multicast_barrier'
```

Result:

```text
9 passed, 18105 deselected
```

Runtime-matrix proxy-adjacent selector collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q --collect-only python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2 or cp_no_scales_linear or shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_twocta_acc_subslice_view_format or mma_scaled_acc_tile_permuted_32_bscale or mma_scaled_acc_identity_narrow_format_use_acc or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc'
```

Result:

```text
213/1615 tests collected
```

Runtime-matrix proxy-adjacent execution, split across four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2 or cp_no_scales_linear or shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_twocta_acc_subslice_view_format or mma_scaled_acc_tile_permuted_32_bscale or mma_scaled_acc_identity_narrow_format_use_acc or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2 or cp_no_scales_linear or shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_twocta_acc_subslice_view_format or mma_scaled_acc_tile_permuted_32_bscale or mma_scaled_acc_identity_narrow_format_use_acc or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2 or cp_no_scales_linear or shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_twocta_acc_subslice_view_format or mma_scaled_acc_tile_permuted_32_bscale or mma_scaled_acc_identity_narrow_format_use_acc or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2 or cp_no_scales_linear or shared_scale_descriptor_view_auto_tmem_copy or mma_scaled_twocta_acc_subslice_view_format or mma_scaled_acc_tile_permuted_32_bscale or mma_scaled_acc_identity_narrow_format_use_acc or mma_scaled_indexed_acc_identity_narrow_view_format_use_acc'
```

Results:

```text
group 1: 54 passed, 1561 deselected
group 2: 54 passed, 1561 deselected
group 3: 54 passed, 1561 deselected
group 4: 51 passed, 1564 deselected
total:   213 passed
```

Checked-in proxy-fence lit guardrail:

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
lit -v test/TritonGPU/proxy_fence_insertion.mlir
```

Result:

```text
PASS: TRITON :: TritonGPU/proxy_fence_insertion.mlir
Total Discovered Tests: 1
  Passed: 1
```

Disposable Round 14 copy/mbarrier composition replay:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_copy_mbarrier_composition_round14_probe.py
```

Result:

```text
2 failed, 6 passed
```

Failed rows:

```text
test_ai_001_two_independent_twocta_no_scales_direct_regions
test_ai_002_two_independent_twocta_no_scales_descriptor_chain_regions
```

Both failures hit the existing proxy-fence insertion diagnostic:

```text
could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
```

Classification: existing `FZ-20260421-0014`. The passing rows in the same
probe keep the contrast sharp: mixed one-CTA/two-CTA copy regions, copy before
or after an ordinary TMEM store, copy plus one-CTA MMA with independent
mbarriers, and adjacent checked-in two-CTA MMA control all passed.

Saved plain sequential-mbarrier reproducer:

```bash
/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_fz0014_plain_seq_round19_min_probe_branch_repro.mlir.make_llir.repro.mlir \
  --run-reproducer
```

Result: failed with the same existing `FZ-20260421-0014` diagnostic:

```text
could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
```

Raw proxy-fence-only contrast:

```bash
/root/code/triton/build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt \
  /tmp/tmem_fz0014_plain_seq_round19_min_probe_branch_repro.mlir.make_llir.repro.mlir \
  --triton-nvidia-gpu-proxy-fence-insertion=compute-capability=103 \
  --allow-unregistered-dialect
```

Result: exit code `0`; the raw pass alone leaves the module unchanged. This
matches prior classification that `FZ-0014` depends on the full lowering
pipeline state reaching proxy-fence insertion, not on the raw saved IR plus the
single pass alone.

## Classification

`FZ-20260421-0014` remains the only observed failure class in this lane.

The checked-in positive coverage says ordinary proxy/TMA/MMAv5/TMEM-copy
ordering is currently stable for:

- TMA load/store plus explicit `mbarrier.expect/wait/invalidate`.
- TMA gather/scatter in multi-CTA layouts.
- `tcgen05.mma` and `tcgen05.mma_scaled` with completion mbarriers.
- One-CTA and two-CTA commit opcode selection.
- Warpx2 no-scales `ttng.tmem_copy`, including descriptor-view and two-CTA
  rows.
- Scaled-MMAv5 rows that combine shared-scale descriptor-view auto-copy,
  two-CTA accumulator views, and B-scale descriptor-view rematerialization.

The existing failure remains specifically about sequential independent
cross-CTA mbarrier intervals. The key trigger is not TMA, scaled copy, or MMAv5
itself; the disposable replays show plain mbarrier-only and no-scales copy
forms can both reproduce the same insertion-point failure. Green contrasts show
that initializing all cross-CTA mbarriers before the tracked copy/commit/wait
phase avoids the issue, and that independent region count alone is not enough.

## Follow-Up

When the fuzzing campaign moves from cataloging to fixing, `FZ-0014` should be
treated as a proxy-fence interval planning bug in the full lowering pipeline.
The fix should preserve the green checked-in behavior above and add an explicit
checked-in sentinel for at least the minimal sequential plain-mbarrier
reproducer plus one TMEM-copy variant.
