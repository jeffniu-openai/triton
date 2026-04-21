# Round 39: MMAv5 Dynamic Descriptor/View Fuzzing

Date: 2026-04-21 13:58 UTC
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend/compiler code or checked-in tests
were edited.
Write set: this report only.

## Objective

Adversarially exercise MMAv5 and scaled-MMAv5 descriptor/control-flow surfaces:

- dynamic or selected accumulator descriptors;
- A-scale and B-scale descriptor views;
- loop-carried, helper-returned, nested-branch, and branch-selected scale
  descriptors;
- extra scale descriptor users and rematerialization-adjacent rows;
- `use_acc`, accumulator subslices, two-CTA rows, and multicast/high-CGA
  controls where available.

The run classifies against existing `FZ-20260421-0001`,
`FZ-20260421-0007`, `FZ-20260421-0013`, `FZ-20260421-0015`, and
`FZ-20260421-0010`. No new independent `FZ-*` bucket is proposed.

## Environment Note

An initial collect attempt without explicit `PYTHONPATH` imported a stale
Triton checkout from `/tmp/triton-upstream-main-check` and failed collection
with missing `triton.compiler.errors` / `triton.runtime.jit`. All real
validation below used `PYTHONPATH=.:./python` or
`PYTHONPATH=.:./python:./python/test/gluon`.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Checked-in runtime selector collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q --collect-only -s --tb=short \
  -k '(mma_scaled and (descriptor_view or bscale_view_extra_user or use_acc or indexed_acc or lhs_subslice or subslice_view or twocta or multicast)) or (use_acc and subslice)' \
  python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py
```

Result: `190/19729` collected.

Checked-in runtime selector split-4:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 -k '(mma_scaled and (descriptor_view or bscale_view_extra_user or use_acc or indexed_acc or lhs_subslice or subslice_view or twocta or multicast)) or (use_acc and subslice)' python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py 2>&1 | tee /tmp/fuzz_mmav5_dynamic_views_round39_checked_g0.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 -k '(mma_scaled and (descriptor_view or bscale_view_extra_user or use_acc or indexed_acc or lhs_subslice or subslice_view or twocta or multicast)) or (use_acc and subslice)' python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py 2>&1 | tee /tmp/fuzz_mmav5_dynamic_views_round39_checked_g1.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 -k '(mma_scaled and (descriptor_view or bscale_view_extra_user or use_acc or indexed_acc or lhs_subslice or subslice_view or twocta or multicast)) or (use_acc and subslice)' python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py 2>&1 | tee /tmp/fuzz_mmav5_dynamic_views_round39_checked_g2.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 -k '(mma_scaled and (descriptor_view or bscale_view_extra_user or use_acc or indexed_acc or lhs_subslice or subslice_view or twocta or multicast)) or (use_acc and subslice)' python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_core.py 2>&1 | tee /tmp/fuzz_mmav5_dynamic_views_round39_checked_g3.log
```

Result: `190 passed` split as `48/48/48/46`.

Checked-in structural selector collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q --collect-only -s --tb=short \
  -k 'scaled_mma_acc_subslice_control_flow or dynamic_index_load_only or generic_pass_dynamic_index or generic_pass_loop_carried or copy_scales' \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `5/33` collected.

Checked-in structural split-4:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 -k 'scaled_mma_acc_subslice_control_flow or dynamic_index_load_only or generic_pass_dynamic_index or generic_pass_loop_carried or copy_scales' python/test/gluon/test_tmem_structural_fuzzer.py 2>&1 | tee /tmp/fuzz_mmav5_dynamic_views_round39_struct_g0.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 -k 'scaled_mma_acc_subslice_control_flow or dynamic_index_load_only or generic_pass_dynamic_index or generic_pass_loop_carried or copy_scales' python/test/gluon/test_tmem_structural_fuzzer.py 2>&1 | tee /tmp/fuzz_mmav5_dynamic_views_round39_struct_g1.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 -k 'scaled_mma_acc_subslice_control_flow or dynamic_index_load_only or generic_pass_dynamic_index or generic_pass_loop_carried or copy_scales' python/test/gluon/test_tmem_structural_fuzzer.py 2>&1 | tee /tmp/fuzz_mmav5_dynamic_views_round39_struct_g2.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 -k 'scaled_mma_acc_subslice_control_flow or dynamic_index_load_only or generic_pass_dynamic_index or generic_pass_loop_carried or copy_scales' python/test/gluon/test_tmem_structural_fuzzer.py 2>&1 | tee /tmp/fuzz_mmav5_dynamic_views_round39_struct_g3.log
```

Result: `2 passed, 3 xfailed` split as `2 pass`, `2 xfail`, `1 xfail`, and
one empty shard. The `generic-pass-dynamic-index-load-only-128x32` xfail
printed the known late illegal `ttg.memdesc_index` MLIR reproducer.

Temporary probe commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_scaled_dynamic_scales_round22_probe.py \
  2>&1 | tee /tmp/fuzz_mmav5_dynamic_views_round39_dynamic_scales.log

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_bscale_dynamic_views_round36_probe.py \
  2>&1 | tee /tmp/fuzz_mmav5_dynamic_views_round39_bscale_views.log

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_bscale_descriptor_selection \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_ascale_descriptor_selection \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_dynamic_bscale_same_object_selection \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_branch_contained_bscale_mma \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_scale_descriptor_view_chains \
  /tmp/tmem_scaled_multi_mma_round15_probe.py::test_am_acc_subslice_multi_mma \
  2>&1 | tee /tmp/fuzz_mmav5_dynamic_views_round39_scaled_multi.log

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_high_cga_scaled_round26_probe.py \
  2>&1 | tee /tmp/fuzz_mmav5_dynamic_views_round39_high_cga.log
```

## Matrix

| Surface | Coverage | Result | Classification |
| --- | --- | --- | --- |
| Checked-in scaled/MMAv5 runtime selector | 190 rows: descriptor views, `bscale_view_extra_user`, `use_acc`, indexed acc, lhs/acc subslices, two-CTA scaled rows, multicast scaled barrier | `190 passed` | Green controls; no new bucket |
| Checked-in structural sentinels | 5 rows: copy scales, dynamic-index load-only, loop-carried generic view, scaled acc-subslice control flow | `2 passed, 3 xfailed` | Existing `FZ-20260421-0001` and `FZ-20260421-0007` sentinels remain expected |
| Dynamic scale selection probe | 16 rows: direct A/B/both branch selection, helper return, nested branch, loop carried, two-MMA, and reshape/permute/reshape scale views | 10 direct rows passed; 6 view rows failed as expected; `unexpected=0` | View rows are existing `FZ-20260421-0013`; direct selected scale controls remain green |
| B-scale dynamic-view/remat probe | 5 rows: direct, branch-distinct, branch chain with extra users, loop-carried chain with extra users, alternate selector | `5 passed` | Green contrast; no new `FZ-0015` or `FZ-0013` reproduction in this valid probe |
| Scaled multi-MMA discriminator | 14 selected rows | `9 passed, 5 failed` | `2` existing `FZ-20260421-0015`, `3` existing `FZ-20260421-0013`; passing controls cover A-scale selection, same-object B-scale selection, branch-contained B-scale MMA, and acc subslice multi-MMA |
| High-CGA scaled probe | 3 high-CGA scaled controls plus 24 local 1CTA/2CTA mixed TMEM rows under `num_ctas=4/8/16` | 3 controls passed; 24 expected diagnostics | Existing `FZ-20260421-0010` for local layout CTA-count mismatch |

## Repro/Classifications

- `FZ-20260421-0001`: reproduced only through the checked-in structural xfail
  `test_tmem_structural_fuzzer_generic_pass_dynamic_index_load_only[generic-pass-dynamic-index-load-only-128x32]`.
  The symptom is the known late illegal `ttg.memdesc_index` during
  `ConvertTritonGPUToLLVM`. The dynamic scaled scale/accumulator runtime
  probes in this lane did not add a new `FZ-0001` shape.
- `FZ-20260421-0007`: the checked-in structural xfail
  `test_tmem_structural_fuzzer_scaled_mma_acc_subslice_control_flow[mma-scaled-fz20260421-0007-subslice-if-n64-selector0]`
  remained expected. Adjacent checked-in `use_acc` and accumulator-subslice
  runtime rows passed, so this lane does not broaden `FZ-0007`.
- `FZ-20260421-0013`: reproduced by scale descriptor-view operands consumed by
  scaled-MMAv5. In `/tmp/tmem_scaled_dynamic_scales_round22_probe.py`, the
  six `rtr` view rows had `16368` to `16374` mismatches out of `16384`, while
  side-channel A/B scale descriptor loads had `0` mismatches and PTX/LLIR both
  emitted the expected scaled-MMAv5 ops. In
  `/tmp/tmem_scaled_multi_mma_round15_probe.py`, the three
  `test_am_scale_descriptor_view_chains` rows failed with `16107/16384`,
  `16116/16384`, and `16373/16384` mismatches.
- `FZ-20260421-0015`: reproduced by direct distinct B-scale descriptors merged
  through runtime selection before scaled-MMAv5. The two
  `test_am_dynamic_bscale_descriptor_selection` rows failed with
  `16381/16384` and `16380/16384` mismatches and NaN greatest differences.
  The same run kept A-scale dynamic selection, same-object B-scale selection,
  branch-contained B-scale MMA, and static accumulator subslice multi-MMA rows
  passing, preserving the existing bucket boundary.
- `FZ-20260421-0010`: reproduced by
  `/tmp/tmem_high_cga_scaled_round26_probe.py` for local 1CTA/2CTA TMEM
  `st`, `ld`, `ldred`, and `copy` rows in `num_ctas=4/8/16` contexts. The
  diagnostics remain the expected CTA-count ownership messages, while the
  high-CGA scaled-MMAv5 controls passed.

## Temporary Paths

Probe files used:

- `/tmp/tmem_scaled_dynamic_scales_round22_probe.py`
- `/tmp/tmem_bscale_dynamic_views_round36_probe.py`
- `/tmp/tmem_scaled_multi_mma_round15_probe.py`
- `/tmp/tmem_high_cga_scaled_round26_probe.py`

Round-39 logs:

- `/tmp/fuzz_mmav5_dynamic_views_round39_checked_g0.log`
- `/tmp/fuzz_mmav5_dynamic_views_round39_checked_g1.log`
- `/tmp/fuzz_mmav5_dynamic_views_round39_checked_g2.log`
- `/tmp/fuzz_mmav5_dynamic_views_round39_checked_g3.log`
- `/tmp/fuzz_mmav5_dynamic_views_round39_struct_g0.log`
- `/tmp/fuzz_mmav5_dynamic_views_round39_struct_g1.log`
- `/tmp/fuzz_mmav5_dynamic_views_round39_struct_g2.log`
- `/tmp/fuzz_mmav5_dynamic_views_round39_struct_g3.log`
- `/tmp/fuzz_mmav5_dynamic_views_round39_dynamic_scales.log`
- `/tmp/fuzz_mmav5_dynamic_views_round39_bscale_views.log`
- `/tmp/fuzz_mmav5_dynamic_views_round39_scaled_multi.log`
- `/tmp/fuzz_mmav5_dynamic_views_round39_high_cga.log`

## Conclusion

No new independent `FZ-*` bucket was found. The adversarial combinations
preserve the existing boundaries:

- dynamic `memdesc_index` compiler failure remains `FZ-0001`;
- selected accumulator subslice scaled-MMAv5 remains `FZ-0007`;
- scale descriptor-view operands consumed by scaled-MMAv5 remain `FZ-0013`;
- direct distinct runtime-selected B-scale descriptors remain `FZ-0015`;
- high-CGA mixed local TMEM ownership diagnostics remain `FZ-0010`.

Backend repair remains deferred.
