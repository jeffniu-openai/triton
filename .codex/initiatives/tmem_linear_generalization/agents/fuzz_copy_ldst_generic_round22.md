# Round 22 Lane BC: Copy/Ld-St Generic Descriptor Fuzzing

- Date: 2026-04-21 12:55 UTC
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler fixes.
- Repo edit scope: this report only.

## Scope

This lane stressed TMEM copy, `tmem_load`, and `tmem_store` descriptor-view
composition under generic memdesc SSA:

- branch-selected TMEM descriptors feeding `ttng.tmem_copy` plus readback;
- loop-carried descriptors feeding copy;
- branch-selected descriptors with multiple ld/st consumers;
- 1CTA and 2CTA `warpx2::01_23` copy descriptor branches;
- checked-in copy/ld-st descriptor coverage, including nested
  index/subslice/reshape chains, high-CGA diagnostics, and scale-copy controls.

No backend/compiler code was changed.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Temporary structural probe:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python3 /tmp/tmem_round22_copy_generic_probe.py \
  2>&1 | tee /tmp/tmem_round22_copy_generic_probe3.log
```

Checked-in selector collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k '(cp_no_scales and (indexed_view or subslice_view or warpx2 or twocta or linear) and not reports) or (generic_pass and not mma) or (ldst_descriptor and not reports)'
```

Result:

```text
329/1648 tests collected (1319 deselected)
```

Split-4 runtime selector:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <1..4> \
  --store-durations --durations-path /tmp/tmem_r22_copy_ldst_generic_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k '(cp_no_scales and (indexed_view or subslice_view or warpx2 or twocta or linear) and not reports) or (generic_pass and not mma) or (ldst_descriptor and not reports)'
```

Results:

```text
group 1: 32 passed, 51 skipped
group 2: 73 passed, 10 skipped
group 3: 83 passed
group 4: 66 passed, 14 xfailed
total:   254 passed, 61 skipped, 14 xfailed
```

Scale-copy and high-CGA diagnostic selector:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_scales and not reports) or cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error or cp_no_scales_warpx2_02_13_twocta_candidate_reports_clean_unsupported'
```

Result:

```text
36 passed, 1579 deselected
```

## Temporary Probe Results

The temporary probe lives at:

```text
/tmp/tmem_round22_copy_generic_probe.py
/tmp/tmem_round22_copy_generic_probe3.log
```

Rows:

| Row | Result | Classification |
| --- | --- | --- |
| `copy_branch_linear`, selector `0` | LLVM conversion failure: illegal `ttg.memdesc_index` left after a branch-selected `scf.if` memdesc feeds `ttng.tmem_copy` and `ttng.tmem_load`. | Existing `FZ-20260421-0001`. |
| `copy_branch_linear`, selector `1` | Same failure. | Existing `FZ-20260421-0001`. |
| `copy_loop_linear`, selector `0` | Pass, `0` mismatches; TTGIR still contains two static `ttg.memdesc_index` ops and one `ttng.tmem_copy`. | Passing contrast for `FZ-0001`. |
| `copy_loop_linear`, selector `1` | Pass, `0` mismatches. | Passing contrast for `FZ-0001`. |
| `ldst_branch_multi_consumer`, selector `0` | Compiles but miscompiles: `16128/16384` mismatches after branch-selected descriptor load/store/load. | Existing `FZ-20260421-0002`. |
| `ldst_branch_multi_consumer`, selector `1` | Same mismatch count. | Existing `FZ-20260421-0002`. |
| `copy_warpx2_branch_1cta_01_23`, selector `0/1` | Pass, `0` mismatches; one `ttng.tmem_copy`. | Passing contrast. |
| `copy_warpx2_branch_2cta_01_23`, selector `0/1` | Pass, `0` mismatches; one `ttng.tmem_copy`. | Passing 2CTA contrast. |

The failing `copy_branch_linear` row is a useful new reproducer shape for the
already known generic descriptor problem. It is not a copy-planner false
unsupported diagnostic: the copy op is present in TTGIR, but the selected
descriptor returned by `scf.if` still contains branch-local `ttg.memdesc_index`
operations when the pipeline reaches `ConvertTritonGPUToLLVM`.

Minimized repro command:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python3 /tmp/tmem_round22_copy_generic_probe.py
```

The `copy_loop_linear` contrast is important: not every control-flow-looking
descriptor variable is currently broken. In this probe, the loop-carried form
specializes/canonicalizes enough to execute correctly, while the explicit
branch-yielded memdesc does not.

The `warpx2::01_23` copy rows further narrow the copy side: static 1CTA and
2CTA indexed descriptor branches for the hardware-copy family execute
correctly. The current Round 22 failure is generic descriptor SSA lowering, not
the `warpx2` copy schedule itself.

## Checked-In Coverage Classification

The broad checked-in selector stayed at the expected state:

- all non-report copy and ld/st descriptor rows passed or skipped as expected;
- the 14 xfails are existing generic descriptor buckets, mostly
  `FZ-20260421-0001` illegal dynamic memdesc index and
  `FZ-20260421-0002` branch/helper/layout-pressure wrong-result rows;
- no unexpected pass/fail, compiler crash, or new miscompile appeared.

The separate scale-copy/high-CGA selector also stayed clean:

- scale-copy controls passed;
- the 4CTA context diagnostic for 2CTA copy layouts stayed a clean
  `FZ-20260421-0010` diagnostic shape;
- the known 2CTA `warpx2::02_13` clean unsupported row stayed diagnostic-only;
- no `FZ-20260421-0014` proxy-fence failure was reproduced in this lane.

## FZ Classification

No new independent `FZ-*` bucket is proposed.

Sharpened existing buckets:

- `FZ-20260421-0001`: now includes branch-yielded generic TMEM descriptors
  feeding `ttng.tmem_copy` plus readback. The failure is the same illegal
  `ttg.memdesc_index` left for LLVM conversion, not a copy-family-specific
  planner rejection.
- `FZ-20260421-0002`: branch-selected ld/st descriptors with multiple
  consumers still compile and miscompile. The new probe reproduces the issue
  with an inline branch and two post-selection consumers.

Negative/clean contrasts:

- loop-carried copy descriptor probe passed for both selectors;
- 1CTA and 2CTA `warpx2::01_23` branch-selected copy descriptor probes passed;
- checked-in copy descriptor, scale-copy, and clean high-CGA diagnostic rows
  remained green.

## Next Fuzzing Slice

The next useful structural direction is to combine the failing
branch-yielded-copy shape with:

- nested `memdesc_subslice`/`reshape` chains before the branch;
- two independent copy consumers of the same selected descriptor;
- `warpx4` scale-copy source layouts where the destination is selected by
  control flow;
- 4/8/16 CTA kernel contexts containing only instruction-local 1CTA/2CTA copy
  descriptors, to further separate `FZ-0010` from copy lowering.
