# Round 50: Dynamic Descriptor Control-Flow Adversary

Date: 2026-04-21

Branch: `codex/tmem`

Mode: discovery/cataloging only. No backend fixes and no checked-in tests were
added. The only repo write from this lane is this report. Disposable probes and
logs live under `/tmp`.

## Objective

Adversarially retest dynamic/control-flow-carried TMEM memdesc values beyond
the minimized linear-copy sentinel. The tested axes were:

- runtime `parent.index(ttgl.load(selector))`;
- branch-selected, same-object, and distinct-index descriptors;
- loop-carried descriptors;
- helper-returned descriptors;
- separate TMEM object selection;
- load/store/copy/`ld.red`/mixed consumers, plus nearby MMAv5 controls.

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

## Broad Dynamic Descriptor Rerun

Probe:

```text
/tmp/tmem_dynamic_descriptor_round33_probe.py
```

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 1 /tmp/tmem_dynamic_descriptor_round33_probe.py 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_g1.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_dynamic_descriptor_round33_probe.py 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_g2.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 3 /tmp/tmem_dynamic_descriptor_round33_probe.py 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_g3.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 4 /tmp/tmem_dynamic_descriptor_round33_probe.py 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_g4.log
```

Result:

| Group | Result |
| --- | --- |
| 1 | `12 failed, 13 passed, 75 deselected` |
| 2 | `8 failed, 17 passed, 75 deselected` |
| 3 | `17 failed, 8 passed, 75 deselected` |
| 4 | `17 failed, 8 passed, 75 deselected` |

Manual classification remains unchanged from Round 33:

| Classification | Count | Details |
| --- | ---: | --- |
| Pass | 46 | Direct load/store/`ld.red`/mixed rows for constant, same-branch, distinct-branch, and loop-carried modes; direct copy constant/same-branch/loop-carried rows. |
| Existing `FZ-20260421-0001` | 20 | Runtime-index direct/chain load, store, `ld.red`, and mixed rows; direct runtime-index copy; direct same-parent distinct-branch copy. |
| Existing `FZ-20260421-0002`/`FZ-20260421-0003` | 24 | Chain0 load, `ld.red`, and mixed non-runtime-index rows. Full output mismatches were `4032/4096`; `ld.red`/mixed also reported `126/128` reduction mismatches while still emitting hardware `.ld.red`. |
| Clean unsupported copy boundary | 10 | Chain0 `ttng.tmem_copy` rows for every selector mode. |

Representative `FZ-0001` signature:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
Pipeline failed while executing [`ConvertTritonGPUToLLVM` on 'builtin.module' operation]
```

Representative `FZ-0002`/`FZ-0003` signature:

```text
load ... mismatch_count 4032
ldred ... out_mismatch_count 4032
ldred ... red_mismatch_count 126
```

Representative clean copy diagnostic:

```text
The source shared layout maps to tcgen05.copy.128x128b, but Triton could not
synthesize a compatible shared-memory descriptor plan for it.
```

## Distinct TMEM Object Probe

New disposable probe:

```text
/tmp/tmem_dynamic_distinct_objects_round50_probe.py
```

This probe selects between two separately allocated TMEM objects with the same
shape/layout, rather than selecting indexed children of one parent descriptor.
It crosses:

- `branch-same-object`;
- `branch-distinct-object`;
- `helper-distinct`;
- `loop-distinct`;
- consumers: `tmem_load`, hardware `ld.red`, and `ttng.tmem_copy`.

Setup and collection:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_dynamic_distinct_objects_round50_probe.py
PYTHONPATH=.:./python:./python/test/gluon pytest --collect-only -q /tmp/tmem_dynamic_distinct_objects_round50_probe.py
```

Result: `24 tests collected`.

Run commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 1 /tmp/tmem_dynamic_distinct_objects_round50_probe.py 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_distinct_g1.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_dynamic_distinct_objects_round50_probe.py 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_distinct_g2.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 3 /tmp/tmem_dynamic_distinct_objects_round50_probe.py 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_distinct_g3.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 4 /tmp/tmem_dynamic_distinct_objects_round50_probe.py 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_distinct_g4.log
```

Result:

```text
group 1: 6 passed, 18 deselected
group 2: 6 passed, 18 deselected
group 3: 6 passed, 18 deselected
group 4: 6 passed, 18 deselected
```

All `24` rows passed. Every printed mismatch count was `0`, and every
`ld.red` row emitted:

```text
tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32
```

This is an important green dynamic case: memdesc SSA values can cross branch,
helper, and loop selection when the alternatives are separate concrete TMEM
objects with the same descriptor type. The failing surface is narrower than
"any dynamic memdesc"; it is tied to unresolved parent-index descriptor values
and descriptor-view arithmetic.

## Linear Copy Sentinel and Warpx2 Contrast

Linear copy sentinel:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor_128x128b[0]' '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor_128x128b[1]' 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_copy_linear.log
```

Result: `2 failed`, both existing `FZ-20260421-0001`.

Signature:

```text
scf.if -> !ttg.memdesc<128x4xf32, ...>
ttng.tmem_copy %shared, %if_result
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

Warpx2 dynamic-copy controls:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[01_23-0-index]' '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[01_23-1-index]' '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[02_13-0-index]' '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[02_13-1-slice_index]' 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_copy_warpx2.log
```

Result: `4 passed`.

These controls reconfirm that dynamic copy is not universally broken:
branch-selected `warpx2::{01_23,02_13}` descriptors still lower and execute
correctly, including the `02_13` `slice(...).index(0)` path.

## Structural and MMA Controls

Checked-in structural dynamic rows:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass_dynamic or generic_pass_loop_carried or scaled_mma_acc_subslice_control_flow' 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_structural.log
```

Result:

```text
30 deselected, 3 xfailed
```

The visible diagnostic was the expected `FZ-20260421-0001` illegal
`ttg.memdesc_index` for the dynamic-index load-only row.

MMAv5 controls:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_indexed_acc_view[f16-linear-32-32-True]' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_twocta_indexed_acc_view[f16-linear-32-32-True]' 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_scaled_mma_acc_subslice_control_flow' 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_mma.log
```

Result:

```text
2 passed, 1 xfailed
```

Classification:

- plain MMAv5 indexed accumulator controls are green;
- scaled-MMAv5 dynamic accumulator subslice remains existing
  `FZ-20260421-0007`.

## Helper Harness Notes

Older disposable helper probe:

```text
/tmp/tmem_generic_memdesc_control_round15_probe.py
```

Command:

```bash
PYTHONPATH=.:./python:./python/test/gluon python /tmp/tmem_generic_memdesc_control_round15_probe.py --results /tmp/tmem_dynamic_descriptor_controlflow_round50_helper_results.jsonl 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_helper.log
```

Result file buckets:

```text
25 total rows
4 pass
2 FZ-20260421-0002
15 compiler_failure_unclassified
4 failure_unclassified
```

The harness is stale and its controller records only wrapper exceptions for
many compiler failures. I isolated a representative helper-returned row:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon python /tmp/tmem_generic_memdesc_control_round15_probe.py --worker-case '{"case_id":"r15-ldst_branch_helper-chain0-sel0","kind":"ldst_branch_helper","chain":0,"selector":0,"seed":15000,"M":128,"N":64,"variant":"32x32b"}' 2>&1 | tee /tmp/tmem_dynamic_descriptor_controlflow_round50_helper_detail.log
```

That isolated row is existing `FZ-20260421-0001`:

```text
return view_chain(parent.index(0), M, N, chain)
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

The four green helper rows are loop-carried `ld/st` helper controls for chain1
and direct slice-only chain2, both selectors. The two explicit
`FZ-20260421-0002` rows are loop-carried chain0 `ld/st` rows with
`8064/8192` mismatches. The stale helper harness should not be promoted as-is;
future checked-in coverage should keep the useful kernels but capture stderr
directly and drop harness-invalid scaled-cache-key rows.

## Classification

No new `FZ-*` bucket is needed.

- `FZ-20260421-0001` remains the dynamic parent-index lowering failure:
  runtime `parent.index(ttgl.load(...))`, helper-returned parent-index views,
  and branch-selected same-parent distinct-index descriptors feeding
  `ttng.tmem_copy` can leave illegal `ttg.memdesc_index` live into
  `ConvertTritonGPUToLLVM`.
- `FZ-20260421-0002`/`FZ-20260421-0003` remain descriptor-view semantic
  wrong-result buckets. Chain0 read/reduction/mixed consumers still miscompile
  even when the selector is static or loop-carried, so these are not pure
  dynamic-index failures.
- `FZ-20260421-0007` remains the scaled-MMAv5 dynamic accumulator subslice
  xfail.
- `FZ-20260421-0015` was not reproduced by this lane. The direct B-scale
  dynamic descriptor rows were not the primary target here; recent Round 46/47
  scaled-MMAv5 descriptor lanes remain the current evidence for that bucket.

Green dynamic cases worth preserving:

- branch/helper/loop selection between separate TMEM objects: `24 passed`;
- branch-selected direct load/store/`ld.red`/mixed descriptors from the same
  parent in the broad probe: green except runtime-index and chain-view cases;
- branch-selected `warpx2` copy descriptors: `4 passed`;
- plain MMAv5 indexed accumulator descriptors: `2 passed`.

Working hypothesis after this round: generic memdesc SSA itself is not the
problem. The backend gap is specifically unresolved parent-index/view-chain
lowering when the selected descriptor must still be interpreted as a concrete
TMEM address/layout by copy/load/reduction/MMA consumers.
