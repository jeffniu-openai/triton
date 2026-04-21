# Round 18 Lane AS: Dynamic Memdesc SSA Selection and Clean Boundaries

Date: 2026-04-21
Branch: `codex/tmem`
Scope: report-only discovery lane. No backend/compiler code changed, no commits,
no pushes.

## Objective

Stress dynamic/generic TMEM memdesc SSA values at clean-boundary surfaces:
descriptor views, parent-view subslices, scale descriptor shapes, high-CGA
clean gates, and multiple consumers. The lane looked for late illegal
`ttg.memdesc_index`/`ttg.memdesc_subslice`, false clean unsupported diagnostics,
over-strict verifiers, and wrong results. Fixes remain deferred until fuzzing
stops finding new bugs.

## Commands

Required build gate:

```bash
make -j8
```

Result: no work to do.

Temporary focused probe:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_dynamic_clean_boundaries_round18_probe.py \
  | tee /tmp/tmem_dynamic_clean_boundaries_round18_probe_rerun.log
```

Checked-in coverage collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(generic_pass or dynamic_index or ldst_scales_descriptor_view or scales_descriptor_view or scale_descriptor or bscale_descriptor_view or layout_in_4cta_context or reports_clean_unsupported or reports_clean_error) and not block_descriptor'
```

Result: `170/1648` selected.

Checked-in 4-GPU split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_structural_fuzzer.py python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(generic_pass or dynamic_index or ldst_scales_descriptor_view or scales_descriptor_view or scale_descriptor or bscale_descriptor_view or layout_in_4cta_context or reports_clean_unsupported or reports_clean_error) and not block_descriptor'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_structural_fuzzer.py python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(generic_pass or dynamic_index or ldst_scales_descriptor_view or scales_descriptor_view or scale_descriptor or bscale_descriptor_view or layout_in_4cta_context or reports_clean_unsupported or reports_clean_error) and not block_descriptor'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_structural_fuzzer.py python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(generic_pass or dynamic_index or ldst_scales_descriptor_view or scales_descriptor_view or scale_descriptor or bscale_descriptor_view or layout_in_4cta_context or reports_clean_unsupported or reports_clean_error) and not block_descriptor'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_structural_fuzzer.py python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(generic_pass or dynamic_index or ldst_scales_descriptor_view or scales_descriptor_view or scale_descriptor or bscale_descriptor_view or layout_in_4cta_context or reports_clean_unsupported or reports_clean_error) and not block_descriptor'
```

Results:

- group 1: `32 passed, 11 xfailed`
- group 2: `43 passed`
- group 3: `43 passed`
- group 4: `41 passed`
- aggregate: `159 passed, 11 xfailed`

## Temporary Probe Rows

| Row | Surface | Result | Classification |
| --- | --- | --- | --- |
| `dynamic_index_load_only_parent_view` | runtime `parent.index(index)` feeding only `tmem_load` | late illegal `ttg.memdesc_index` in `ConvertTritonGPUToLLVM` | `FZ-20260421-0001` |
| `dynamic_if_chain0_multi_consumer` | runtime branch-selected descriptor view through store/load path | `4032/4096` mismatches, `max_abs_diff ~= 5.33`, TTGIR contains `ttg.memdesc_index` | `FZ-20260421-0002` |
| `tuple_mixed_capture_chain0` | branch-selected descriptor plus tensor bias through tuple-like helper | `4061/4096` mismatches, `max_abs_diff ~= 10.32`, TTGIR contains `ttg.memdesc_index` | `FZ-20260421-0002` |
| `high_cga_2cta_layout_in_4cta_context` | 2CTA TMEM layout in 4CTA launch context | clean frontend diagnostic: layout has 2 CTAs per CGA, context requires 4 | `FZ-20260421-0010` |
| `scale_descriptor_clean_boundary` | scales TMEM descriptor-view copy | checked-in clean unsupported test passed | existing clean boundary |
| `scale_ldst_2cta_cga_positive` | 2CTA scales descriptor-view `ld/st` positive | checked-in runtime row passed | green control |
| `fz0015_runtime_selected_bscale` | runtime-selected distinct B-scale descriptor consumed by scaled MMAv5 | minimizer reports `16379/16384` mismatches, `119` infs, 4 scaled-MMA ops | `FZ-20260421-0015` |

## Classification

No new independent `FZ-*` bucket was found in this lane.

This round strengthens existing buckets:

- `FZ-20260421-0001`: the smallest dynamic-index load-only row still reaches
  LLVM conversion with an illegal `ttg.memdesc_index`. Descriptor-view chains
  are not required.
- `FZ-20260421-0002`: multiple-consumer generic pass rows remain wrong-result
  miscompiles, including tuple-like descriptor/tensor capture. The failures are
  not clean unsupported diagnostics and are not late verifier failures.
- `FZ-20260421-0010`: high-CGA 2CTA layout in a 4CTA context still reports a
  clean CTA-count diagnostic. This is a known policy/gating bucket, not a new
  crash.
- `FZ-20260421-0015`: runtime-selected distinct direct B-scale descriptors
  still miscompile only at scaled-MMAv5 consumption; the probe preserved the
  direct selected-B-scale signature without descriptor-view chains.

The checked-in clean-boundary sweep did not expose a new false positive or
false negative. Scale descriptor-view clean negatives and 2CTA scale `ld/st`
positives stayed green.

## Next Probes

- Extend dynamic memdesc SSA selection to `memdesc_subslice` specifically:
  runtime branch between two parent subslices, then one `ld/st` consumer and one
  scaled-MMA accumulator consumer.
- Add a raw discriminator where dynamic descriptor SSA has both a `tmem_load`
  and a `tcgen05_mma` consumer in the same kernel, to test whether the load
  rematerializes while MMA still consumes a stale physical address.
- Sweep `FZ-20260421-0010` with `num_ctas=8` and `16` for copy, `ld/st`, and
  plain/scaled MMAv5, recording which paths diagnose in frontend construction
  versus backend passes.
- For `FZ-20260421-0015`, test selected B-scale descriptor with a preceding
  selected-scale `load` forced live into an output buffer, to confirm again that
  the descriptor value is readable and only the scaled-MMA operand path is
  wrong.

## Follow-Up: Dynamic `memdesc_subslice` `ld/st`

Temporary probe:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_dynamic_subslice_round18_probe.py \
  | tee /tmp/tmem_dynamic_subslice_round18_probe.log
```

Rows:

| Row | Selector | Result | Notes |
| --- | --- | --- | --- |
| `if_view1_on_true` | `0` | pass, `0/4096` mismatches | TTGIR has `ttg.memdesc_subslice`, no `ttg.memdesc_index` |
| `if_view1_on_true` | `1` | pass, `0/4096` mismatches | dynamic branch-selected subslice readback is correct |
| `if_view0_on_true` | `1` | pass, `0/4096` mismatches | reversed branch polarity is correct |
| `helper_after_if` | `1` | pass, `0/4096` mismatches | identity helper after branch does not break subslice descriptor |
| `extra_store_user` | `1` | pass, `0/4096` mismatches | selected subslice with both load and store user remains correct |

Classification: green discriminator, no new bucket. Unlike dynamic
`memdesc_index`, dynamic branch selection between two same-shaped column
subslice views can lower and execute correctly for this `ld/st` surface. The
next useful discriminator is to feed the same selected subslice value to MMAv5
or scaled-MMAv5, where accumulator operand handling may differ from `ld/st`.
