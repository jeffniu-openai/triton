# Round 40: copy/scales dynamic descriptor selection fuzzing

- Date: 2026-04-21 14:09 UTC
- Branch: `codex/tmem`
- HEAD: `ed546cbef`
- Mode: discovery/cataloging only; no backend/compiler code modified.
- Owned report path:
  `.codex/initiatives/tmem_linear_generalization/agents/fuzz_copy_dynamic_descriptor_round40.md`
- Temporary probe:
  `/tmp/tmem_copy_dynamic_descriptor_round40_probe.py`

## Scope

This lane fuzzed TMEM copy/scales runtime behavior around:

- checked-in `tcgen05.copy` no-scales positives, indexed views, subslice
  views, dense-source rematerialization rows, source-initialization lifetime,
  and scales-copy/scale-layout rows;
- checked-in structural dynamic descriptor rows that overlap existing
  `FZ-20260421-0001`, `FZ-20260421-0002`, and `FZ-20260421-0003`;
- temporary branch-selected destination descriptors feeding `ttng.tmem_copy`;
  and
- temporary `warpx2::{01_23,02_13}` descriptor-view chains with dynamic branch
  selection before copy.

The temporary probe imports checked-in test helpers only for temporary
classification. No repo source or backend code was changed.

## Commands and Results

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Checked-in runtime collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales or cp_scales or shared_scale_descriptor_view_auto_tmem_copy or bscale_descriptor_view or bscale_view_extra_user or source_initialization or rematerializes) and not reports and not resource'
```

Result: `270/1615` rows collected.

Checked-in runtime split run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales or cp_scales or shared_scale_descriptor_view_auto_tmem_copy or bscale_descriptor_view or bscale_view_extra_user or source_initialization or rematerializes) and not reports and not resource'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales or cp_scales or shared_scale_descriptor_view_auto_tmem_copy or bscale_descriptor_view or bscale_view_extra_user or source_initialization or rematerializes) and not reports and not resource'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales or cp_scales or shared_scale_descriptor_view_auto_tmem_copy or bscale_descriptor_view or bscale_view_extra_user or source_initialization or rematerializes) and not reports and not resource'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales or cp_scales or shared_scale_descriptor_view_auto_tmem_copy or bscale_descriptor_view or bscale_view_extra_user or source_initialization or rematerializes) and not reports and not resource'
```

Result: `266 passed, 4 skipped` across `270` selected rows.
Shard breakdown: group 1 `64 passed, 4 skipped`; group 2 `68 passed`;
group 3 `68 passed`; group 4 `66 passed`.

Checked-in structural collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'copy_scales or generic_pass or dynamic_index or dynamic_if or descriptor_view'
```

Result: `16/33` rows collected.

Checked-in structural split run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'copy_scales or generic_pass or dynamic_index or dynamic_if or descriptor_view'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'copy_scales or generic_pass or dynamic_index or dynamic_if or descriptor_view'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'copy_scales or generic_pass or dynamic_index or dynamic_if or descriptor_view'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'copy_scales or generic_pass or dynamic_index or dynamic_if or descriptor_view'
```

Result: `2 passed, 14 xfailed`. The xfail output remained stable:
dynamic `ttg.memdesc_index` rows still fail during LLVM conversion with
`failed to legalize operation 'ttg.memdesc_index' that was explicitly marked
illegal`, and descriptor-view read/control-flow rows remain under the existing
known-bucket xfails.

Temporary probe syntax and collection:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_copy_dynamic_descriptor_round40_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_copy_dynamic_descriptor_round40_probe.py
```

Result: syntax check passed; `6` rows collected.

Temporary probe split run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  --splits 4 --group 1 /tmp/tmem_copy_dynamic_descriptor_round40_probe.py

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  --splits 4 --group 2 /tmp/tmem_copy_dynamic_descriptor_round40_probe.py

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  --splits 4 --group 3 /tmp/tmem_copy_dynamic_descriptor_round40_probe.py

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  --splits 4 --group 4 /tmp/tmem_copy_dynamic_descriptor_round40_probe.py
```

Result after correcting an initial temporary harness layout issue:

- group 1: `2 passed`;
- group 2: `2 passed`;
- group 3: `2 failed`;
- group 4: no selected tests, pytest exit `5`.

Exact temporary `warpx2` reruns:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[01_23-0-index]'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[01_23-1-index]'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[02_13-0-index]'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  '/tmp/tmem_copy_dynamic_descriptor_round40_probe.py::test_branch_selected_warpx2_copy_descriptor[02_13-1-slice_index]'
```

Result: all four exact `warpx2` branch-selected rows passed.

## Temporary Probe Case Table

| Case | Result | Classification |
| --- | --- | --- |
| `branch_selected_warpx2_copy_descriptor[01_23-0-index]` | pass | Branch-selected descriptor feeds `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b`; no wrong result or opcode absence. |
| `branch_selected_warpx2_copy_descriptor[01_23-1-index]` | pass | Same as above with opposite selector; no selector-dependent drift. |
| `branch_selected_warpx2_copy_descriptor[02_13-0-index]` | pass | Branch-selected descriptor feeds `tcgen05.cp.cta_group::1.warpx2::02_13.64x128b`; no wrong result or opcode absence. |
| `branch_selected_warpx2_copy_descriptor[02_13-1-slice_index]` | pass | Branch selection plus slice/index descriptor-view chain remains positive. |
| `branch_selected_linear_copy_descriptor[0]` | compiler failure | Existing `FZ-20260421-0001`: branch-selected memdesc value feeding `ttng.tmem_copy` leaves `ttg.memdesc_index` illegal at LLVM conversion. |
| `branch_selected_linear_copy_descriptor[1]` | compiler failure | Same existing `FZ-20260421-0001` signature with alternate selector. |

## Representative Diagnostics

The temporary linear branch-selected copy rows lower to an `scf.if` yielding a
TMEM memdesc, then feed that yielded memdesc to `ttng.tmem_copy` and
`ttng.tmem_load`. LLVM conversion fails with:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
Pipeline failed while executing [`ConvertTritonGPUToLLVM` on 'builtin.module' operation]
RuntimeError: PassManager::run failed
```

This is the same owner surface as existing `FZ-20260421-0001`: dynamic or
control-flow-carried descriptor values can leave `ttg.memdesc_index` live into
late lowering. The temporary rows broaden the copy-specific evidence because
the dynamic descriptor value is consumed by `ttng.tmem_copy`, not only by
load/store or MMAv5 consumers.

An initial temporary `warpx2` harness variant used a lifted layout where the
checked-in helper pattern expects the base `warpx2` layout for direct index
allocation. That produced a frontend `failed to infer memdesc_index result
type` diagnostic. I discarded it as a probe-construction error after changing
only the temporary probe and rerunning all four exact `warpx2` rows
successfully.

## Classification

No new independent `FZ-*` bucket was found.

Positive coverage:

- checked-in copy/scales/source-rematerialization rows remain stable:
  `266 passed, 4 skipped`;
- checked-in structural dynamic descriptor rows remain stable:
  `2 passed, 14 xfailed`;
- branch-selected `warpx2::01_23` and `warpx2::02_13` copy descriptors,
  including one slice/index descriptor-view chain, produce the expected
  `tcgen05.cp` opcodes and correct runtime output.

Known-bucket overlap:

- `FZ-20260421-0001`: reproduced and copy-sharpened by the temporary
  `branch_selected_linear_copy_descriptor[0/1]` rows. These are compiler
  failures, not runtime miscompiles.
- `FZ-20260421-0002` / `FZ-20260421-0003`: checked-in structural xfails remain
  stable; this lane did not find a copy-specific new wrong-result signature.
- `FZ-20260421-0014`: no new proxy-fence-specific or barrier-lifetime signal;
  copy/scales rows that execute continue to pass.
- `FZ-20260421-0017`: no new 64-bit TMEM load/store/copy assertion signal in
  this lane.

No compiler crash outside expected known-bucket failures, false unsupported
diagnostic, clean-boundary drift, opcode absence, or runtime miscompile was
observed. Backend repair remains deferred.
