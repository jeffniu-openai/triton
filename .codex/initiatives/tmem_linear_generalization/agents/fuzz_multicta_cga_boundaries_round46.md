# Round 46: Multi-CTA and CGA Boundary Fuzzing

- Date: 2026-04-21 14:38 UTC
- Branch: `codex/tmem`
- HEAD: `76758d1cc04c`
- Scope: discovery/classification only. No backend/compiler code and no
  checked-in tests were modified. The only checked-in write from this lane is
  this report.
- Required rebuild:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Objective

Adversarially exercise TMEM backend behavior around more than one CTA per CGA,
with emphasis on:

- 2CTA instruction consistency across TMEM consumers;
- 4CTA and larger CGA context boundaries;
- clean high-CGA diagnostics, especially the known 2CTA-in-4CTA boundary;
- two-CTA copy and TMA descriptor paths;
- multicast barriers and TMA/mbarrier sequences;
- maximum-CTA-per-CGA surfaces already represented in checked-in tests.

No temporary reproducers were needed. All coverage came from checked-in
runtime tests in `python/test/gluon/test_tmem_runtime_matrix.py` and
`python/test/gluon/test_core.py`.

## Lane A: runtime-matrix multi-CTA/two-CTA breadth

Purpose: run the broad checked-in runtime matrix rows that mention `twocta`,
`two_cta`, `multicast`, `cga`, or `mbarrier`, while excluding the explicitly
named `reports_clean` and `resource` selectors from the primary positive lane.

File:

```text
python/test/gluon/test_tmem_runtime_matrix.py
```

Selector:

```text
(twocta or two_cta or multicast or cga or mbarrier) and not reports_clean and not resource
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k '(twocta or two_cta or multicast or cga or mbarrier) and not reports_clean and not resource' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result:

```text
350/1615 tests collected (1265 deselected) in 2.52s
```

Split-4 run pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  -k '(twocta or two_cta or multicast or cga or mbarrier) and not reports_clean and not resource' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Shard results:

- Group 1 / GPU 0: `61 passed, 27 skipped, 1527 deselected` in `4.29s`
- Group 2 / GPU 1: `78 passed, 10 skipped, 1527 deselected` in `5.16s`
- Group 3 / GPU 2: `88 passed, 1527 deselected` in `6.35s`
- Group 4 / GPU 3: `86 passed, 1529 deselected` in `5.69s`

Aggregate:

```text
313 passed, 37 skipped
```

Coverage notes:

- two-CTA TMEM load/store linear layouts;
- two-CTA descriptor compositions, higher-rank descriptors, multidim slices,
  half-row views, rank-5 views, and direct descriptor-chain roundtrips;
- subword two-CTA load/store and descriptor-chain roundtrips;
- two-CTA `ttng.tmem_copy` linear, subslice, indexed, `warpx2::01_23`, and
  scales-copy paths;
- two-CTA MMAv5 plain, use-acc, indexed-acc, accumulator-subslice, TMA
  transposed-B, and scaled accumulator-subslice rows;
- CGA-aware scales load/store descriptor-view rows.

Classification:

- No compiler crash.
- No runtime wrong-result signal.
- No unexpected unsupported diagnostic in this selected lane.
- No new evidence for `FZ-0010`, `FZ-0012`, or `FZ-0014`.
- No new independent `FZ-*`.

## Lane B: high-CGA and clean-boundary diagnostics

Purpose: isolate the adjacent clean-diagnostic and CGA boundary rows from the
positive breadth lane, including 4CTA/two-CTA incompatibility checks and
clean unsupported copy/MMA diagnostics.

File:

```text
python/test/gluon/test_tmem_runtime_matrix.py
```

Selector:

```text
(reports_clean or resource or high_cga or cga) and (twocta or two_cta or cga or multicast)
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k '(reports_clean or resource or high_cga or cga) and (twocta or two_cta or cga or multicast)' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result:

```text
56/1615 tests collected (1559 deselected) in 2.52s
```

Split-4 run pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  -k '(reports_clean or resource or high_cga or cga) and (twocta or two_cta or cga or multicast)' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Shard results:

- Group 1 / GPU 0: `14 passed, 1601 deselected` in `3.44s`
- Group 2 / GPU 1: `14 passed, 1601 deselected` in `7.13s`
- Group 3 / GPU 2: `14 passed, 1601 deselected` in `4.42s`
- Group 4 / GPU 3: `14 passed, 1601 deselected` in `3.46s`

Aggregate:

```text
56 passed
```

Coverage notes:

- `block_two_ctas` reinterpret/subslice clean-error reporting;
- CGA-aware scales descriptor-view positives;
- `warpx2::02_13` two-CTA clean unsupported copy rows;
- `test_tmem_runtime_matrix_cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error`;
- noncanonical two-CTA block clean unsupported row;
- two-CTA i8 MMAv5 clean unsupported matrix;
- two-CTA TMA TF32 shared-transpose clean error rows.

Classification:

- Clean hardware/API boundary diagnostics stayed stable.
- The 4CTA-context/two-CTA-layout boundary remained a clean diagnostic, not a
  crash or silent miscompile.
- No failure-mode drift in the known high-CGA boundary area associated with
  `FZ-0010`.
- No new independent `FZ-*`.

## Lane C: focused core TMA, multicast, and MMAv5 controls

Purpose: exercise checked-in `test_core.py` surfaces that directly cover
TMA multicast, gather/scatter multi-CTA layouts, MMAv5 multicast commit, the
two-CTA linear accumulator path, and the scaled direct multicast barrier row.
This lane includes the existing maximum-CTA-per-CGA TMA multicast surface:
`test_tma_multicast_copy[ctas_per_cga2]` uses `[4, 4]`, i.e. 16 CTAs per CGA.

File:

```text
python/test/gluon/test_core.py
```

Selector:

```text
(test_tma_multicast_copy or test_tma_gather_scatter_multi_cta or test_tcgen05_mma_multicast_commit or test_tcgen05_mma_multicast_commit_twocta_linear_acc or test_tcgen05_mma_scaled_direct_multicast_barrier or test_tma_mma_uses_tma_shared_inputs_multicast or test_tma_mma_uses_tma_gather_shared_inputs_multicast)
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k '(test_tma_multicast_copy or test_tma_gather_scatter_multi_cta or test_tcgen05_mma_multicast_commit or test_tcgen05_mma_multicast_commit_twocta_linear_acc or test_tcgen05_mma_scaled_direct_multicast_barrier or test_tma_mma_uses_tma_shared_inputs_multicast or test_tma_mma_uses_tma_gather_shared_inputs_multicast)' \
  python/test/gluon/test_core.py
```

Result:

```text
16/18114 tests collected (18098 deselected) in 3.09s
```

Collected nodeids:

```text
python/test/gluon/test_core.py::test_tma_multicast_copy[ctas_per_cga0]
python/test/gluon/test_core.py::test_tma_multicast_copy[ctas_per_cga1]
python/test/gluon/test_core.py::test_tma_multicast_copy[ctas_per_cga2]
python/test/gluon/test_core.py::test_tma_gather_scatter_multi_cta[cga_layout0]
python/test/gluon/test_core.py::test_tma_gather_scatter_multi_cta[cga_layout1]
python/test/gluon/test_core.py::test_tma_gather_scatter_multi_cta[cga_layout2]
python/test/gluon/test_core.py::test_tma_gather_scatter_multi_cta[cga_layout3]
python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[True-ctas_per_cga0]
python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[True-ctas_per_cga1]
python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[True-ctas_per_cga2]
python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[False-ctas_per_cga0]
python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[False-ctas_per_cga1]
python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[False-ctas_per_cga2]
python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit_twocta_linear_acc[legacy]
python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit_twocta_linear_acc[linear]
python/test/gluon/test_core.py::test_tcgen05_mma_scaled_direct_multicast_barrier
```

Split-4 run pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  -k '(test_tma_multicast_copy or test_tma_gather_scatter_multi_cta or test_tcgen05_mma_multicast_commit or test_tcgen05_mma_multicast_commit_twocta_linear_acc or test_tcgen05_mma_scaled_direct_multicast_barrier or test_tma_mma_uses_tma_shared_inputs_multicast or test_tma_mma_uses_tma_gather_shared_inputs_multicast)' \
  python/test/gluon/test_core.py
```

Shard results:

- Group 1 / GPU 0: `4 passed, 18110 deselected` in `4.43s`
- Group 2 / GPU 1: `4 passed, 18110 deselected` in `4.34s`
- Group 3 / GPU 2: `4 passed, 18110 deselected` in `3.81s`
- Group 4 / GPU 3: `4 passed, 18110 deselected` in `4.81s`

Aggregate:

```text
16 passed
```

Classification:

- TMA multicast for `[2, 1]`, `[1, 4]`, and `[4, 4]` stayed green.
- Multi-CTA TMA gather/scatter stayed green.
- Two-CTA MMAv5 multicast commit and the two-CTA linear accumulator variant
  stayed green.
- The scaled direct multicast barrier row stayed green.
- No `FZ-0014` proxy/mbarrier evidence surfaced here.
- No new independent `FZ-*`.

## Failed nodeids

None.

## Overall classification

Round 46 found no new compiler crash, false unsupported case, or runtime
miscompile in the checked-in multi-CTA/CGA boundary surfaces exercised here.

The run strengthens the current classification that:

- `FZ-0010` remains a focused high-CGA/CTA-count ownership gap; the clean
  checked-in boundary rows did not drift and the 4CTA context diagnostic
  stayed controlled.
- `FZ-0012` did not appear in this multi-CTA run; no M64 `ld.red` destination
  planner failures were selected by these lanes.
- `FZ-0014` did not appear in the multicast/TMA/mbarrier controls; no new
  proxy-fence or mbarrier sequencing failure was observed.

No new `FZ-*` bucket is needed from this round.
