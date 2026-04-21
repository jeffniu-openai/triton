# Round 58 High-CGA Ownership Lane B

Date: 2026-04-21
Branch: `codex/tmem`
Base HEAD before report commit: `8521122cf`

Scope: catalog/discovery only. I did not edit backend/compiler code. This lane
focused on high-CGA / CTA ownership boundaries beyond the recent high-CGA
scaled-copy checks, especially mixed local TMEM ownership inside global CGA
contexts and adjacent mbarrier/proxy/TMA flows.

## Worktree Note

Before this report was written, the worktree already had modified initiative
files and two untracked Round 58 reports from other workers:

```text
 M .codex/initiatives/tmem_linear_generalization/README.md
 M .codex/initiatives/tmem_linear_generalization/completion_execution_tracker.md
 M .codex/initiatives/tmem_linear_generalization/handoff_2026-04-09.md
 M .codex/initiatives/tmem_linear_generalization/log.md
 M .codex/initiatives/tmem_linear_generalization/memory.md
?? .codex/initiatives/tmem_linear_generalization/agents/fuzz_round58_plain_mma_acc_views_lane.md
?? .codex/initiatives/tmem_linear_generalization/agents/fuzz_round58_scaled_tile_boundary_lane.md
```

I did not modify or stage those files.

## Required Build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Checked-In High-CGA Sweep

Collection command:

```bash
PYTHONPATH=.:./python \
pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_core.py \
  -k '(layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma or (test_tma_mma_shared_inputs and ctas_per_cga2 and True) or (test_mma_scaled_tcgen05_copy and ctas_per_cga5 and False)) and not reports and not resource'
```

Collection result:

```text
93/19729 tests collected (19636 deselected)
```

Runtime split command shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
PYTHONPATH=.:./python \
pytest -s --tb=short --splits 4 --group <group> \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_core.py \
  -k '(layout_in_4cta_context or cta8 or cta16 or high_cga or multicast or twocta_tma or (test_tma_mma_shared_inputs and ctas_per_cga2 and True) or (test_mma_scaled_tcgen05_copy and ctas_per_cga5 and False)) and not reports and not resource'
```

Results:

```text
group 1: 24 passed, 19705 deselected
group 2: 24 passed, 19705 deselected
group 3: 21 passed, 3 skipped, 19705 deselected
group 4: 18 passed, 3 skipped, 19708 deselected
aggregate: 87 passed, 6 skipped, 0 failed
```

Coverage included:

- checked-in `FZ-20260421-0010` 2CTA-layout-in-4CTA clean diagnostic;
- high-CGA / multicast / twoCTA-TMA rows from the runtime matrix and
  `test_core.py`;
- 16-CTA `test_tcgen05_mma_multicast_commit` controls for both 1CTA and 2CTA
  instruction-local ownership;
- 16-CTA `test_tma_mma_shared_inputs` rows with TMA/mbarrier, multicast, and
  gather/scatter variants selected by `ctas_per_cga2 and True`;
- 16-CTA scaled-copy controls at `ctas_per_cga5 and False`, retained only as
  max-CGA green controls while the lane emphasized non-overlapping ownership
  contexts.

## Lit Mbarrier / Cluster Control

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
lit -v test/TritonNvidiaGPU/membar-cluster.mlir
```

Result:

```text
ninja: no work to do.
PASS: TRITON :: TritonNvidiaGPU/membar-cluster.mlir
Total Discovered Tests: 1
Passed: 1
```

## Temporary Ownership Probe

Temporary file:

```text
/tmp/tmem_round58_high_cga_ownership_lane_probe.py
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_round58_high_cga_ownership_lane_probe.py
```

Rows:

- `cp_no_scales_twocta_in_{4,8,16}cta`: 2CTA local linear TMEM copy in
  4/8/16 CTA kernel contexts.
- `copy_scales_1cta_in_{4,8,16}cta`: 1CTA local
  `TensorMemoryScalesLayout` plus proxy fence, mbarrier, `tcgen05.copy`, and
  commit in 4/8/16 CTA contexts.
- `copy_scales_2cta_in_{4,8,16}cta`: 2CTA local
  `TensorMemoryScalesLayout(cga_layout=[[1, 0]])` plus the same proxy/mbarrier
  copy sequence in 4/8/16 CTA contexts.
- `ldred_twocta_in_{4,8,16}cta`: 2CTA local linear TMEM `ld.red` in 4/8/16
  CTA contexts.

Result:

```text
cp_no_scales_twocta_in_4cta: FZ-20260421-0010 2cta-in-4cta
copy_scales_1cta_in_4cta: FZ-20260421-0010 parser-wrapper
copy_scales_2cta_in_4cta: FZ-20260421-0010 parser-wrapper
ldred_twocta_in_4cta: FZ-20260421-0010 2cta-in-4cta
cp_no_scales_twocta_in_8cta: FZ-20260421-0010 2cta-in-8cta
copy_scales_1cta_in_8cta: FZ-20260421-0010 parser-wrapper
copy_scales_2cta_in_8cta: FZ-20260421-0010 parser-wrapper
ldred_twocta_in_8cta: FZ-20260421-0010 2cta-in-8cta
cp_no_scales_twocta_in_16cta: FZ-20260421-0010 2cta-in-16cta
copy_scales_1cta_in_16cta: FZ-20260421-0010 parser-wrapper
copy_scales_2cta_in_16cta: FZ-20260421-0010 parser-wrapper
ldred_twocta_in_16cta: FZ-20260421-0010 2cta-in-16cta
SUMMARY {'fz0010': 6, 'fz0010_parser_wrapper': 6, 'unexpected_pass': 0, 'unclassified': 0}
```

The scales-copy rows print the same CTA-count diagnostic to stderr before
Gluon wraps the Python exception as `RuntimeError("error encountered during
parsing")`. Example stderr diagnostic:

```text
Layout has 1 CTAs per CGA, but the context requires 16 CTAs per CGA.
```

The 2CTA scales-copy rows likewise print:

```text
Layout has 2 CTAs per CGA, but the context requires 16 CTAs per CGA.
```

## Classification

No new independent `FZ-*` bucket.

Matrix:

| Surface | CTA context | Result | Classification |
| --- | ---: | --- | --- |
| Checked-in high-CGA runtime selector | 4/8/16 as encoded by tests | 87 passed, 6 skipped | Green / existing skips |
| Lit `membar-cluster.mlir` | cluster mbarrier | 1 passed | Green control |
| TMA/mbarrier shared-input rows | 16 | Passed or existing skips | Green control |
| Plain MMAv5 multicast commit | 16 | Passed | Green control |
| Scaled-copy max-CGA controls | 16 | Passed | Green control |
| No-scales 2CTA local copy in larger CGA | 4/8/16 | CTA-count diagnostic | Existing `FZ-20260421-0010` |
| Scales copy, 1CTA local layout in larger CGA | 4/8/16 | CTA-count diagnostic, parser wrapped | Existing `FZ-20260421-0010` |
| Scales copy, 2CTA local layout in larger CGA | 4/8/16 | CTA-count diagnostic, parser wrapped | Existing `FZ-20260421-0010` |
| 2CTA local `ld.red` in larger CGA | 4/8/16 | CTA-count diagnostic | Existing `FZ-20260421-0010` |

The lane did not find a missed `getModuleTwoCTAs` propagation bug, compiler
crash, verifier drift, false unsupported diagnostic outside the known
CTA-count gate, runtime miscompile, hang, or proxy/TMA/mbarrier ownership
regression. High-CGA green controls remain stable when the layouts carry full
CGA metadata. Local 1CTA/2CTA TMEM work inside 4/8/16 CTA kernel contexts still
stops at the existing `FZ-20260421-0010` equality gate.
