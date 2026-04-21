# Round 14 Lane AI: copy/mbarrier composition fuzzing

Date: 2026-04-21 11:21 UTC
Branch: `codex/tmem`
HEAD: `ec42ca421`
Mode: discovery/cataloging only. No backend or compiler code was changed.

## Summary

Lane AI focused on adversarial runtime/Gluon copy and mbarrier composition
beyond the initial `FZ-20260421-0014` mixed no-scales/scales case:

- multiple independent 2CTA no-scales copy regions;
- descriptor-chain versus direct TMEM copy;
- mixed 1CTA/2CTA copy regions;
- copy before/after direct TMEM store/load consumers;
- copy plus MMA controls with independent mbarriers;
- `warpx2` and high-CGA contrasts.

No new independent `FZ-*` bucket was assigned and no backend repair was
attempted. Two temporary rows broaden `FZ-20260421-0014`: a legal
`num_ctas=2` kernel containing two independent 2CTA no-scales copy regions
aborts in proxy-fence insertion even without scales copy. This reproduces for
both direct TMEM destinations and descriptor-chain destinations.

No runtime wrong-result/miscompile was observed. Green rows covered copy
before/after TMEM stores, 1CTA copy plus 1CTA MMA with independent barriers,
and checked-in 2CTA MMA controls. Clean context diagnostics remained mapped to
`FZ-20260421-0010`.

## Required rebuild

Command:

```bash
make -j8
```

Result: no work to do.

## Checked-in baseline selector

Collect-only:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales and (warpx2 or twocta) and not reports'
```

Result: `103/1615` selected.

Runtime command pattern:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group <group> \
  --store-durations --durations-path /tmp/tmem_lane_ai_round14_copy_baseline_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales and (warpx2 or twocta) and not reports'
```

Results:

- GPU 0 / group 1: `26 passed, 1589 deselected in 4.30s`.
- GPU 1 / group 2: `26 passed, 1589 deselected in 4.39s`.
- GPU 2 / group 3: `26 passed, 1589 deselected in 14.21s`.
- GPU 3 / group 4: `25 passed, 1590 deselected in 11.61s`.
- Aggregate: `103 passed`.

This confirms the checked-in single-copy `warpx2`, two-CTA no-scales copy,
descriptor-chain copy, and clean subword diagnostic surface remains green.

## Temporary subprocess probe

Temporary file:

```text
/tmp/tmem_copy_mbarrier_composition_round14_probe.py
```

The probe stayed outside the repo and imported the checked-in runtime matrix
helpers for layouts and expected-copy results. Each row below was run as an
exact pytest node in a fresh process with a stable per-GPU cache.

Collect-only:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_copy_mbarrier_composition_round14_probe.py
```

Result: `8 tests collected`.

## Case matrix

| Case | Focus | Command summary | Result | Classification |
| --- | --- | --- | --- | --- |
| `AI-001` | two independent direct 2CTA no-scales `warpx2::01_23` copies with independent mbarriers | exact node `test_ai_001_two_independent_twocta_no_scales_direct_regions` on GPU 0 | compile crash in proxy-fence insertion | existing `FZ-20260421-0014`, broadened |
| `AI-002` | two independent descriptor-chain 2CTA no-scales `warpx2::01_23` copies with independent mbarriers | exact node `test_ai_002_two_independent_twocta_no_scales_descriptor_chain_regions` on GPU 1 | compile crash in proxy-fence insertion | existing `FZ-20260421-0014`, broadened |
| `AI-003` | mixed 1CTA plus 2CTA copy regions in `num_ctas=2` context | exact node `test_ai_003_mixed_onecta_twocta_copy_regions_clean_context_boundary` on GPU 2 | expected clean context diagnostic | existing `FZ-20260421-0010` |
| `AI-004a` | 2CTA copy before direct TMEM store/load consumer | exact node `test_ai_004_copy_tmem_store_ordering[copy_before_store]` on GPU 3 | passed | green control |
| `AI-004b` | direct TMEM store/load before 2CTA copy | exact node `test_ai_004_copy_tmem_store_ordering[store_before_copy]` on GPU 0 | passed | green control |
| `AI-005` | direct 2CTA no-scales copy pair in `num_ctas=4` context | exact node `test_ai_005_direct_twocta_no_scales_copy_in_high_cga_context_fz0010` on GPU 1 | expected clean context diagnostic | existing `FZ-20260421-0010` |
| `AI-006` | 1CTA no-scales copy plus 1CTA MMA with independent mbarriers | exact node `test_ai_006_copy_plus_onecta_mma_independent_mbarriers` on GPU 2 | passed | green control |
| `AI-007` | checked-in style 2CTA MMA control adjacent to copy/mbarrier surface | exact node `test_ai_007_checked_in_twocta_mma_control_adjacent_to_copy_surface` on GPU 3 | passed | green control |

Temporary probe aggregate:

- Runtime/compile green controls: `4` passed (`AI-004a`, `AI-004b`,
  `AI-006`, `AI-007`).
- Clean expected diagnostics: `2` (`AI-003`, `AI-005`), classified as
  existing `FZ-20260421-0010`.
- Known proxy-fence compiler crashes: `2` (`AI-001`, `AI-002`), classified as
  existing `FZ-20260421-0014` broadening.
- Runtime miscompiles: `0`.
- New independent compiler crashes: `0`.

## `FZ-20260421-0014` broadening

`AI-001` and `AI-002` both fail at the same proxy-fence insertion diagnostic:

```text
'tt.func' op could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
RuntimeError: PassManager::run failed
Pipeline failed while executing ConvertTritonGPUToLLVM
```

Reproducer artifacts:

- direct/direct two-copy pytest log:
  `/tmp/tmem_copy_mbarrier_round14_ai001.log`;
- direct/direct extracted MLIR:
  `/tmp/tmem_copy_mbarrier_round14_ai001_direct_two_copy_fail.mlir`;
- descriptor-chain/descriptor-chain pytest log:
  `/tmp/tmem_copy_mbarrier_round14_ai002.log`;
- descriptor-chain/descriptor-chain extracted MLIR:
  `/tmp/tmem_copy_mbarrier_round14_ai002_chain_two_copy_fail.mlir`.

Repro commands:

```bash
build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt --run-reproducer \
  /tmp/tmem_copy_mbarrier_round14_ai001_direct_two_copy_fail.mlir
```

```bash
build/cmake.linux-aarch64-cpython-3.12/bin/triton-opt --run-reproducer \
  /tmp/tmem_copy_mbarrier_round14_ai002_chain_two_copy_fail.mlir
```

Both reproduce the same proxy-fence insertion error.

Classification rationale:

- This is not `FZ-20260421-0010`: the failing kernels use legal
  `num_ctas=2` contexts rather than 4/8/16 CTA contexts.
- This is not a new independent bucket: the diagnostic and pass location match
  `FZ-20260421-0014`.
- It broadens `FZ-0014` by showing scales copy is not required. Multiple
  independent cross-CTA no-scales copy/mbarrier regions are enough, for both
  direct and descriptor-chain destinations.

## Final classification

- Checked-in baseline: `103 passed`.
- Temporary rows: `4` passed, `2` clean `FZ-0010` diagnostics, `2`
  `FZ-0014` proxy-fence crashes.
- No runtime wrong-result/miscompile was found.
- No new independent `FZ-*` candidate was assigned.
- No backend/compiler repair was attempted.

Recommended next discovery slice: minimize `FZ-0014` across region count,
readback presence, mbarrier object count, direct-vs-chain destinations, and
copy/MMA mixed-region variants before any backend repair work.
