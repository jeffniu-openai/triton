# Round 40 Proxy/TMA Sequencing Fuzz

Date: 2026-04-21 14:09 UTC

Branch: `codex/tmem`

Mode: discovery/cataloging only. No backend/compiler code was modified.

Owned write path for this lane:
`.codex/initiatives/tmem_linear_generalization/agents/fuzz_proxy_tma_sequences_round40.md`.

## Objective

Adversarially catalog Python/Gluon runtime behavior around:

- proxy fences and cross-CTA `mbarrier` sequencing;
- TMA multicast load/store paths and `tcgen05.commit` sequencing;
- TMA-fed 1CTA/2CTA MMAv5 operations;
- local 1CTA/2CTA TMEM load/store/copy/MMA interactions near proxy and
  mbarrier surfaces;
- overlap with known fuzz buckets, especially `FZ-20260421-0014` and
  `FZ-20260421-0001`.

## Required Rebuild

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Checked-In Selectors

### TMA multicast, commit, and plain mbarrier controls

Collect:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_core.py \
  -k 'tma_multicast or multicast_commit or mbarrier or proxy or tcgen05_commit'
```

Result: `12/18114` collected.

Runtime:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_core.py \
  -k 'tma_multicast or multicast_commit or mbarrier or proxy or tcgen05_commit'
```

Result: `12 passed, 18102 deselected`.

Classification: green guardrail. No compiler crash, false unsupported
diagnostic, opcode absence, runtime miscompile, clean-boundary drift, or new
`FZ-*`.

### Broad 1CTA/2CTA runtime-matrix proxy/mbarrier/multicast surface

Collect:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(proxy or mbarrier or multicast or twocta or layout_in_4cta_context) and not reports and not resource'
```

Result: `322/1615` collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(proxy or mbarrier or multicast or twocta or layout_in_4cta_context) and not reports and not resource'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(proxy or mbarrier or multicast or twocta or layout_in_4cta_context) and not reports and not resource'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(proxy or mbarrier or multicast or twocta or layout_in_4cta_context) and not reports and not resource'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(proxy or mbarrier or multicast or twocta or layout_in_4cta_context) and not reports and not resource'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `54 passed, 27 skipped, 1534 deselected` |
| 2 | 1 | `71 passed, 10 skipped, 1534 deselected` |
| 3 | 2 | `81 passed, 1534 deselected` |
| 4 | 3 | `79 passed, 1536 deselected` |

Aggregate: `285 passed, 37 skipped` selected rows.

Classification: green/stably skipped checked-in surface. No compiler crash,
false unsupported diagnostic, opcode absence, runtime miscompile,
clean-boundary drift, unexpected xfail/pass transition, or new `FZ-*`.

### Structural known-bucket drift check

Collect:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'mbarrier or proxy or multicast or twocta or high_cga or generic_pass'
```

Result: `17/33` collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'mbarrier or proxy or multicast or twocta or high_cga or generic_pass'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'mbarrier or proxy or multicast or twocta or high_cga or generic_pass'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'mbarrier or proxy or multicast or twocta or high_cga or generic_pass'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'mbarrier or proxy or multicast or twocta or high_cga or generic_pass'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `1 passed, 28 deselected, 4 xfailed` |
| 2 | 1 | `28 deselected, 5 xfailed` |
| 3 | 2 | `28 deselected, 5 xfailed` |
| 4 | 3 | `31 deselected, 2 xfailed` |

Aggregate: `1 passed, 16 xfailed` selected rows.

Classification: expected known-bucket xfails only. The visible diagnostics are
the known illegal dynamic `ttg.memdesc_index` family already tracked under
`FZ-20260421-0001` and related structural sentinels. No XPASS, unexpected
failure, or new bucket.

### TMA-fed MMAv5 shared-input sequencing matrix

Collect:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or mma_scaled_direct_multicast_barrier'
```

Result: `217/18114` collected.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or mma_scaled_direct_multicast_barrier'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or mma_scaled_direct_multicast_barrier'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or mma_scaled_direct_multicast_barrier'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_core.py \
  -k 'tma_mma_shared_inputs or mma_scaled_direct_multicast_barrier'
```

Results:

| Group | GPU | Result |
| --- | --- | --- |
| 1 | 0 | `46 passed, 9 skipped, 18059 deselected` |
| 2 | 1 | `46 passed, 9 skipped, 18059 deselected` |
| 3 | 2 | `40 passed, 15 skipped, 18059 deselected` |
| 4 | 3 | `37 passed, 15 skipped, 18062 deselected` |

Aggregate: `169 passed, 48 skipped` selected rows.

Classification: green/stably skipped TMA-fed MMAv5 sequencing matrix. This
covers TMA async load, TMA gather/scatter variants, multicast toggles, 1CTA and
2CTA MMAv5 commit paths, repeated K-loop barrier phases, and the direct scaled
multicast barrier row. No runtime wrong result, opcode absence, compiler
crash, or new `FZ-*`.

## Temporary Probe Replays

### Round 34 mbarrier/proxy composition probe

Probe:

```text
/tmp/tmem_mbarrier_composition_round34_probe.py
```

Syntax/collection:

```bash
python3 -m py_compile /tmp/tmem_mbarrier_composition_round34_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_mbarrier_composition_round34_probe.py
```

Collection: `4 tests collected`.

Runtime split:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 1 \
  /tmp/tmem_mbarrier_composition_round34_probe.py

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 2 \
  /tmp/tmem_mbarrier_composition_round34_probe.py

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 3 \
  /tmp/tmem_mbarrier_composition_round34_probe.py

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 4 \
  /tmp/tmem_mbarrier_composition_round34_probe.py
```

Results:

| Row | Result | Classification |
| --- | --- | --- |
| `R34_001` copy then `cp.scales`, wait each | pytest row passed after classifying proxy-fence insertion diagnostic | existing `FZ-20260421-0014` |
| `R34_002` `cp.scales` then copy, wait each | pytest row passed after classifying proxy-fence insertion diagnostic | existing `FZ-20260421-0014` |
| `R34_003` init all, copy + `cp.scales` + direct ld/st + plain mbarrier | passed with expected opcodes | green contrast |
| `R34_004` copy, direct ld/st, plain mbarrier, second copy | pytest row passed after classifying proxy-fence insertion diagnostic | existing `FZ-20260421-0014` |

The repeated diagnostic was:

```text
'tt.func' op could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
```

Green contrast opcode evidence from `R34_003`:

```text
tcgen05.cp.cta_group::2.warpx2::01_23.64x128b
tcgen05.cp.cta_group::2.warpx4.32x128b
tcgen05.cp.cta_group::2.warpx4.32x128b
```

Classification: no new bucket. The failures are known
`FZ-20260421-0014`; the mixed init-all contrast still passes, which supports
the existing split-interval proxy-fence hypothesis rather than a general
TMA/TMEM/mbarrier opcode absence.

### Round 36 dynamic proxy-view classifier

Probe:

```text
/tmp/tmem_dynamic_proxy_views_round36_probe.py
```

Result artifact for this replay:

```text
/tmp/tmem_dynamic_proxy_views_round36_results_round40.jsonl
```

Syntax/collection:

```bash
python3 -m py_compile /tmp/tmem_dynamic_proxy_views_round36_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_dynamic_proxy_views_round36_probe.py
```

Collection: `8 tests collected`.

Runtime split:

```bash
rm -f /tmp/tmem_dynamic_proxy_views_round36_results_round40.jsonl

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  TMEM_ROUND36_RESULTS=/tmp/tmem_dynamic_proxy_views_round36_results_round40.jsonl \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 1 \
  /tmp/tmem_dynamic_proxy_views_round36_probe.py

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  TMEM_ROUND36_RESULTS=/tmp/tmem_dynamic_proxy_views_round36_results_round40.jsonl \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 2 \
  /tmp/tmem_dynamic_proxy_views_round36_probe.py

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  TMEM_ROUND36_RESULTS=/tmp/tmem_dynamic_proxy_views_round36_results_round40.jsonl \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 3 \
  /tmp/tmem_dynamic_proxy_views_round36_probe.py

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  TMEM_ROUND36_RESULTS=/tmp/tmem_dynamic_proxy_views_round36_results_round40.jsonl \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group 4 \
  /tmp/tmem_dynamic_proxy_views_round36_probe.py
```

Split result: each group ran `2` rows and passed after classification.
Aggregate: `8 passed` classified rows.

Sorted result JSONL:

```text
{"case": "r36_dynamic_index_copy_proxy_sel0", "classification": "existing_fz0001_memdesc_index", "detail": "           ^", "exception_type": "CompilationError", "status": "exception"}
{"case": "r36_dynamic_index_copy_proxy_sel1", "classification": "existing_fz0001_memdesc_index", "detail": "           ^", "exception_type": "CompilationError", "status": "exception"}
{"case": "r36_dynamic_index_ldst_control_sel0", "classification": "existing_fz0001_memdesc_index", "detail": "           ^", "exception_type": "CompilationError", "status": "exception"}
{"case": "r36_dynamic_index_ldst_control_sel1", "classification": "existing_fz0001_memdesc_index", "detail": "           ^", "exception_type": "CompilationError", "status": "exception"}
{"case": "r36_dynamic_slice_copy_proxy_sel0", "classification": "existing_clean_copy_descriptor_view_boundary", "detail": "error encountered during parsing", "exception_type": "RuntimeError", "status": "exception"}
{"case": "r36_dynamic_slice_copy_proxy_sel1", "classification": "existing_clean_copy_descriptor_view_boundary", "detail": "error encountered during parsing", "exception_type": "RuntimeError", "status": "exception"}
{"case": "r36_sequential_dynamic_index_copy_proxy_0_1", "classification": "existing_fz0001_memdesc_index", "detail": "            ^", "exception_type": "CompilationError", "status": "exception"}
{"case": "r36_sequential_dynamic_index_copy_proxy_1_0", "classification": "existing_fz0001_memdesc_index", "detail": "            ^", "exception_type": "CompilationError", "status": "exception"}
```

Classification: no new proxy/mbarrier-specific bucket. Dynamic-index rows
still fail before proxy-fence reasoning as existing `FZ-20260421-0001`; dynamic
slice copy rows remain existing clean descriptor-view boundaries.

## Overall Classification

No new independent `FZ-*` bucket is proposed for Round 40.

Observed known overlaps:

- Existing `FZ-20260421-0014`: reproduced by three temporary sequential
  cross-CTA proxy/mbarrier rows in the Round 34 probe. The diagnostic and
  row shapes match the prior split-interval proxy-fence insertion bucket.
- Existing `FZ-20260421-0001`: reproduced by structural xfails and Round 36
  dynamic proxy-view rows involving runtime dynamic `memdesc_index`.
- Existing clean descriptor-view boundary: dynamic slice `tcgen05.copy` rows
  still produce the clean unsupported diagnostic that the selected
  shared-memory layout image is not contained in the descriptor-view image.

Not observed:

- new compiler crash outside expected xfail/classified temporary-probe rows;
- false unsupported diagnostic;
- clean-boundary drift;
- TMEM/TMA opcode absence;
- runtime wrong-result miscompile;
- unexpected XPASS or new checked-in test failure;
- new proxy-fence failure beyond existing `FZ-20260421-0014`.

Backend repair remains intentionally deferred.
