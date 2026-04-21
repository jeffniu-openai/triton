# Round 45: Dynamic copy sentinel sharpening

Date: 2026-04-21 14:34 UTC

Branch: `codex/tmem`

HEAD: `9255fbd9992a`

Scope: discovery/classification only. No backend/compiler code or checked-in
tests were modified. The only checked-in write from this lane is this report.

## Objective

Sharpen the Round 44 structural-audit candidate for dynamic descriptor values
feeding `ttng.tmem_copy` as expected `FZ-20260421-0001` evidence. The specific
question was whether the future structural sentinel should encode a
branch-selected linear-copy descriptor failure, a branch-selected `warpx2`
copy positive, or both.

## Required build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Temporary probe

Base probe:

```text
/tmp/tmem_copy_dynamic_descriptor_round40_probe.py
```

I used it as the executable Round 40/Round 44 evidence source, then added a
temporary-only `test_branch_selected_linear_copy_descriptor_128x128b` variant
inside `/tmp` to minimize the failing linear shape. No checked-in test file was
changed.

Syntax and collection after the temporary-only addition:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_copy_dynamic_descriptor_round40_probe.py
PYTHONPATH=.:./python:./python/test/gluon pytest --collect-only -q /tmp/tmem_copy_dynamic_descriptor_round40_probe.py
```

Result: `8` rows collected:

- `4` branch-selected `warpx2` rows from the original probe;
- `2` original broad branch-selected linear rows at `M=128, N=128`;
- `2` temporary minimized branch-selected linear rows at `M=128, N=4`.

## Exact dynamic-copy results

Original broad linear expected-failure controls:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor[0]'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor[1]'
```

Result: both failed during LLVM conversion with the expected
`FZ-20260421-0001` signature:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
Pipeline failed while executing [`ConvertTritonGPUToLLVM` on 'builtin.module' operation]
RuntimeError: PassManager::run failed
```

These rows have `M=128, N=128`, use a lifted parent linear layout
`_lift_tmem_layout(_make_tmem_linear_layout(128, 128), [2])`, branch-select
`parent.index(0)` vs `parent.index(1)` through `scf.if`, then feed the yielded
view to `ttng.tmem_copy` and `ttng.tmem_load`.

Minimized linear expected-failure controls:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor_128x128b[0]'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_linear_copy_descriptor_128x128b[1]'
```

Result: both failed with the same expected `FZ-20260421-0001` signature.

The minimized shape is enough:

- dtype: `f32`;
- tensor shape: `M=128, N=4`;
- TMEM layout: `_lift_tmem_layout(_make_tmem_linear_layout(128, 4), [2])`;
- shared layout: `_make_tmem_copy_128x128_shared_layout()`;
- dynamic producer: `scf.if` over `selector_ptr` yielding either
  `parent.index(0)` or `parent.index(1)`;
- dynamic consumers: `ttng.tmem_copy` and post-copy `ttng.tmem_load`;
- expected opcode if fixed: one `tcgen05.cp.cta_group::1.128x128b`.

This is a tighter sentinel than the original `128x128` row because it removes
the broad `128x256b` copy geometry and still reproduces the dynamic memdesc
lowering failure.

Branch-selected `warpx2` dynamic-copy positives:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[01_23-0-index]' '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[01_23-1-index]'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[02_13-0-index]' '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[02_13-1-slice_index]'
```

Result: `4 passed`.

Positive shapes:

- dtype: `f32`;
- tensor shape: `M=128, N=4`;
- branch-selected descriptor:
  - `warpx2::01_23`, selectors `0` and `1`, direct `parent.index`;
  - `warpx2::02_13`, selector `0`, direct `parent.index`;
  - `warpx2::02_13`, selector `1`, `slice(...).index(0)` chain;
- expected opcodes:
  - `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b`;
  - `tcgen05.cp.cta_group::1.warpx2::02_13.64x128b`;
- runtime oracle: output matches the checked-in `warpx2` expected transform,
  not the input identity.

## Static positive contrasts

Static linear indexed-view positive matching the minimized dynamic-failure
geometry:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales_linear_indexed_view and 128x128b_f32'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_indexed_view[128x128b_f32-torch_dtype0-128-4-smem_layout0-1-tcgen05.cp.cta_group::1.128x128b]'
```

Result: collection `1/1615`; exact row `1 passed`.

Static `warpx2` f32 positives:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_01_23_candidate_positive[f32-torch_dtype0]' 'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_02_13_candidate_positive[f32-torch_dtype0]'
```

Result: `2 passed`.

These contrasts show the failure is not ordinary linear copy support, ordinary
`ttg.memdesc_index`, or `warpx2` copy layout selection. The failure requires a
control-flow-yielded TMEM memdesc value reaching the copy/load consumers.

## Classification

No new independent `FZ-*` bucket is proposed.

The minimized branch-selected linear copy row is expected
`FZ-20260421-0001`: dynamic or control-flow-carried TMEM memdesc values can
leave `ttg.memdesc_index` live into late lowering. This lane sharpens the copy
evidence to a minimal `128x4`/`128x128b` linear-copy shape.

The branch-selected `warpx2` rows are positive controls. They should not be
classified as expected failures: both families execute correctly under branch
selection, including a `slice(...).index(0)` chain for `02_13`.

## Future checked-in sentinel recommendation

Promote one expected-failure structural sentinel:

- name/intent: branch-selected linear `ttng.tmem_copy` dynamic descriptor
  expected `FZ-20260421-0001`;
- shape: `f32`, `M=128`, `N=4`, lifted parent
  `_make_tmem_linear_layout(128, 4)` with two indexed children;
- shared layout: the existing `128x128b` copy shared layout;
- consumer sequence: `ttng.tmem_copy`, commit/wait, then `ttng.tmem_load`;
- expected diagnostic: late LLVM-conversion illegal `ttg.memdesc_index`.

Optionally add one positive structural guardrail later:

- branch-selected `warpx2::01_23` or `warpx2::02_13` copy descriptor at
  `M=128, N=4`;
- assert the exact `tcgen05.cp.cta_group::1.warpx2::* .64x128b` opcode and
  runtime transform.

The expected-failure linear sentinel has higher value because the structural
fuzzer currently lacks a copy-specific dynamic descriptor xfail. The `warpx2`
positive is lower priority because checked-in runtime-matrix positives already
cover static `warpx2` families, while the temporary dynamic branch-selected
`warpx2` evidence stayed green.

Backend repair remains deferred per the discovery-only campaign.
