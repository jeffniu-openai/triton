# Round 36: Dynamic Descriptor Views Crossing Proxy/Mbarrier Regions

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend/compiler code was changed.

## Scope

Temporary Python/Gluon runtime probe for dynamic TMEM descriptor-view values
feeding sequential async/proxy regions and mbarrier waits. The probe targeted:

- runtime-selected indexed descriptors feeding `tcgen05.copy` + commit/wait;
- runtime-selected slice descriptors feeding `tcgen05.copy` + commit/wait;
- runtime-selected indexed descriptors feeding direct TMEM load/store controls;
- two sequential dynamic-index copy/proxy regions with separate mbarriers;
- adjacent checked-in static `warpx2`, direct ld/st, direct `ld.red`, and
  dynamic-index `FZ-20260421-0001` controls.

Temporary files:

- Probe: `/tmp/tmem_dynamic_proxy_views_round36_probe.py`
- Result JSONL: `/tmp/tmem_dynamic_proxy_views_round36_results.jsonl`

## Commands

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

Probe syntax and collection:

```bash
python3 -m py_compile /tmp/tmem_dynamic_proxy_views_round36_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_dynamic_proxy_views_round36_probe.py
```

Collection result: `8 tests collected`.

Four-GPU split run:

```bash
rm -f /tmp/tmem_dynamic_proxy_views_round36_results.jsonl

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
TMEM_ROUND36_RESULTS=/tmp/tmem_dynamic_proxy_views_round36_results.jsonl \
PYTHONPATH=.:./python:./python/test/gluon \
pytest -q -s --tb=short --splits 4 --group 1 /tmp/tmem_dynamic_proxy_views_round36_probe.py

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
TMEM_ROUND36_RESULTS=/tmp/tmem_dynamic_proxy_views_round36_results.jsonl \
PYTHONPATH=.:./python:./python/test/gluon \
pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_dynamic_proxy_views_round36_probe.py

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
TMEM_ROUND36_RESULTS=/tmp/tmem_dynamic_proxy_views_round36_results.jsonl \
PYTHONPATH=.:./python:./python/test/gluon \
pytest -q -s --tb=short --splits 4 --group 3 /tmp/tmem_dynamic_proxy_views_round36_probe.py

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
TMEM_ROUND36_RESULTS=/tmp/tmem_dynamic_proxy_views_round36_results.jsonl \
PYTHONPATH=.:./python:./python/test/gluon \
pytest -q -s --tb=short --splits 4 --group 4 /tmp/tmem_dynamic_proxy_views_round36_probe.py
```

Split results:

| Group | Result |
| --- | --- |
| 1 | `2 passed, 6 deselected in 4.12s` |
| 2 | `2 passed, 6 deselected in 4.22s` |
| 3 | `2 passed, 6 deselected in 3.99s` |
| 4 | `2 passed, 6 deselected in 4.10s` |

The pytest rows pass when the failure is classified and recorded in JSONL.

Adjacent controls:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python:./python/test/gluon \
pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_indexed_view_positive[01_23-_make_tmem_copy_warpx2_tmem_layout-_expected_tmem_copy_warpx2_01_23_output-tcgen05.cp.cta_group::1.warpx2::01_23.64x128b-1-i32-torch_dtype1]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_subslice_view_positive[01_23-_make_tmem_copy_warpx2_parent_tmem_layout-_expected_tmem_copy_warpx2_01_23_output-tcgen05.cp.cta_group::1.warpx2::01_23.64x128b-4-i32-torch_dtype1]'
```

Result: `2 passed in 3.37s`.

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
PYTHONPATH=.:./python:./python/test/gluon \
pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-direct-128x64]'
```

Result: `2 passed in 4.23s`.

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
PYTHONPATH=.:./python:./python/test/gluon \
pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_dynamic_index_load_only[generic-pass-dynamic-index-load-only-128x32]'
```

Result: `1 xfailed in 3.49s`, existing `FZ-20260421-0001`.

## Matrix

| Probe nodeid suffix | Shape | Result | Classification |
| --- | --- | --- | --- |
| `r36_dynamic_index_copy_proxy_sel0` | `parent.index(dynamic selector)` -> `tcgen05.copy` -> commit/wait -> load | compiler exception at `_choose_index_view(parent, selector)` | Existing `FZ-20260421-0001` |
| `r36_dynamic_index_copy_proxy_sel1` | same, selector `1` | same compiler exception | Existing `FZ-20260421-0001` |
| `r36_dynamic_slice_copy_proxy_sel0` | branch-selected `parent.slice(0/4, 4, dim=1)` -> `tcgen05.copy` -> commit/wait -> load | clean `ttng.tmem_copy` diagnostic: source shared-memory image not contained in selected descriptor view | Existing clean copy descriptor-view boundary |
| `r36_dynamic_slice_copy_proxy_sel1` | same, selector `1` | same clean diagnostic | Existing clean copy descriptor-view boundary |
| `r36_dynamic_index_ldst_control_sel0` | `parent.index(dynamic selector)` -> direct TMEM store/load | compiler exception at `_choose_index_view(parent, selector)` | Existing `FZ-20260421-0001` |
| `r36_dynamic_index_ldst_control_sel1` | same, selector `1` | same compiler exception | Existing `FZ-20260421-0001` |
| `r36_sequential_dynamic_index_copy_proxy_0_1` | dynamic-index copy region 0 wait, then dynamic-index copy region 1 wait | compiler exception before proxy-fence pass | Existing `FZ-20260421-0001` |
| `r36_sequential_dynamic_index_copy_proxy_1_0` | same, reversed selectors | same compiler exception | Existing `FZ-20260421-0001` |

Exact repro pattern for any probe row:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
TMEM_ROUND36_RESULTS=/tmp/tmem_dynamic_proxy_views_round36_results.jsonl \
PYTHONPATH=.:./python:./python/test/gluon \
pytest -q -s --tb=short \
  '/tmp/tmem_dynamic_proxy_views_round36_probe.py::test_round36_dynamic_proxy_view_case[<suffix>]'
```

Example failing nodeid:

```text
/tmp/tmem_dynamic_proxy_views_round36_probe.py::test_round36_dynamic_proxy_view_case[r36_dynamic_index_copy_proxy_sel0]
```

The underlying exception contains:

```text
failed to infer memdesc_index result type
unsupported tensor memory memdesc_subslice view; preserved tensor memory view encoding also failed
```

Example clean-boundary nodeid:

```text
/tmp/tmem_dynamic_proxy_views_round36_probe.py::test_round36_dynamic_proxy_view_case[r36_dynamic_slice_copy_proxy_sel0]
```

The diagnostic is:

```text
'ttng.tmem_copy' op unsupported tensor memory descriptor view for tcgen05.copy:
the source shared-memory layout image is not contained in the selected
tensor-memory descriptor view image
```

## Classification

No new independent `FZ-*` is proposed.

- `FZ-20260421-0001` covers the six dynamic-index rows. The important proxy
  observation is negative: dynamic indexed descriptors fail before reaching
  mbarrier/proxy-fence insertion, so this lane does not add a distinct
  proxy-region bug on top of the known illegal/dynamic `memdesc_index` bucket.
- The two dynamic-slice copy rows are existing clean copy descriptor-view
  boundaries. Static `warpx2` indexed/subslice copy controls passed, so this is
  not a general `warpx2::01_23` opcode absence or runtime miscompile.
- Direct ld/st and direct `ld.red` controls passed in checked-in structural
  fuzzer rows. The dynamic direct ld/st row in this probe collapses to
  `FZ-20260421-0001`, matching the checked-in dynamic-index load-only xfail.
- No runtime miscompile was observed in this lane.
- No `FZ-20260421-0014` proxy-fence insertion failure was observed, because the
  sequential dynamic-index copy rows fail earlier in descriptor formation.

## Final Counts

- New independent `FZ-*` candidates: `0`
- Existing `FZ-20260421-0001`: `6`
- Existing clean copy descriptor-view boundary: `2`
- Runtime miscompiles: `0`
- Opcode absence: `0`
- Proxy-fence `FZ-20260421-0014` reproductions: `0`
- Backend/compiler repairs: `0`
