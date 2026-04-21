# Round 57 Copy View Boundaries Lane D

Date: 2026-04-21 15:53 UTC
Branch: `codex/tmem`
Checkpoint base: `1ae7c6130`

Scope: catalog/discovery only for no-scale `tcgen05.copy` descriptor/view
boundaries adjacent to the Round 56 copy/dynamic descriptor lane. No
backend/compiler repairs were attempted. After this lane started, an
untracked Round 56 report appeared in the worktree covering the direct dynamic
copy descriptor/control-flow probe rows; this lane therefore treats the
non-overlap checked-in selector below as primary and records the broader copy
run only as guardrail evidence.

## Worktree Safety

Preexisting/unowned dirty files observed before writing this report:

```text
 M .codex/initiatives/tmem_linear_generalization/memory.md
 M .codex/initiatives/tmem_linear_generalization/tmem_structural_fuzzing_20260421.md
?? .codex/initiatives/tmem_linear_generalization/agents/fuzz_round56_copy_dynamic_descriptor_lane.md
```

This lane edited only this report file.

## Required Build

Command:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Primary Non-Overlap Selector

Purpose: cover checked-in no-scale copy descriptor/view boundaries not covered
by the Round 56 dynamic copy selector: `4x256b` refresh/unsupported,
canonical indexed view, shared-source subslice bad-offset diagnostic, and
TMEM out-of-resource boundaries.

Collection:

```bash
PYTHONPATH=./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales and (4x256b or indexed_view_canonicalized or shared_subslice_bad_offset or full_128x256_reports_tmem_oor or full_256x256_reports_tmem_oor)'
```

Result:

```text
7/1615 tests collected (1608 deselected)
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales and (4x256b or indexed_view_canonicalized or shared_subslice_bad_offset or full_128x256_reports_tmem_oor or full_256x256_reports_tmem_oor)'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=./python pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales and (4x256b or indexed_view_canonicalized or shared_subslice_bad_offset or full_128x256_reports_tmem_oor or full_256x256_reports_tmem_oor)'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=./python pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales and (4x256b or indexed_view_canonicalized or shared_subslice_bad_offset or full_128x256_reports_tmem_oor or full_256x256_reports_tmem_oor)'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=./python pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales and (4x256b or indexed_view_canonicalized or shared_subslice_bad_offset or full_128x256_reports_tmem_oor or full_256x256_reports_tmem_oor)'
```

Result:

```text
group 1: 2 passed, 1613 deselected
group 2: 2 passed, 1613 deselected
group 3: 2 passed, 1613 deselected
group 4: 1 passed, 1614 deselected
aggregate: 7 passed, 0 failed, 0 skipped, 0 xfailed
```

Classification:

- `4x256b` refresh and two-CTA refresh remain green or cleanly unsupported as
  expected by the checked-in tests.
- Canonical no-scale indexed copy view remains green.
- Shared-source bad-offset and full-parent TMEM OOR cases remain clean
  diagnostics, not crashes.
- No new independent `FZ-*`.

## Broad Copy Guardrail

This was launched before the Round 56 dynamic copy report appeared locally.
It overlaps Round 56 checked-in copy coverage, so it is retained only as
guardrail evidence and not counted as the primary non-overlap lane.

Collection:

```bash
PYTHONPATH=./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales and (warpx2 or 4x256b or indexed_view or subslice_view or slice_index_view)'
```

Result:

```text
121/1615 tests collected (1494 deselected)
```

Execution result:

```text
group 1: 31 passed, 1584 deselected
group 2: 31 passed, 1584 deselected
group 3: 31 passed, 1584 deselected
group 4: 28 passed, 1587 deselected
aggregate: 121 passed, 0 failed, 0 skipped, 0 xfailed
```

Coverage included no-scale `warpx2::01_23`, `warpx2::02_13`, 1CTA/2CTA,
indexed views, subslice views, slice+index chains, subword clean diagnostics,
dense shared rematerialization rows, and `4x256b` rows. No runtime
miscompile, compiler crash, false unsupported diagnostic, or opcode assertion
failure appeared.

## Structural Control-Flow Contrast

Purpose: classify control-flow/dynamic memdesc behavior fairly against
existing buckets without adding another dynamic copy probe that would overlap
the Round 56 lane.

Collection:

```bash
PYTHONPATH=./python pytest --collect-only -q python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'copy or generic_pass_memdesc_control_flow or dynamic_index_load_only'
```

Result:

```text
10/33 tests collected (23 deselected)
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_structural_fuzzer.py -k 'copy or generic_pass_memdesc_control_flow or dynamic_index_load_only'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=./python pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_structural_fuzzer.py -k 'copy or generic_pass_memdesc_control_flow or dynamic_index_load_only'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=./python pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_structural_fuzzer.py -k 'copy or generic_pass_memdesc_control_flow or dynamic_index_load_only'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=./python pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_structural_fuzzer.py -k 'copy or generic_pass_memdesc_control_flow or dynamic_index_load_only'
```

Result:

```text
group 1: 2 passed, 1 xfailed, 30 deselected
group 2: 3 xfailed, 30 deselected
group 3: 3 xfailed, 30 deselected
group 4: 1 xfailed, 32 deselected
aggregate: 2 passed, 0 failed, 0 skipped, 8 xfailed
```

Classification:

- Copy-scales structural controls passed; no no-scale copy structural row
  failed.
- Dynamic memdesc index/control-flow rows reproduced existing expected xfails:
  `FZ-20260421-0001` for runtime `ttg.memdesc_index` reaching LLVM conversion
  and `FZ-20260421-0002` for helper/branch-carried TMEM view miscompile
  sentinels.
- No XPASS, new crash, or new independent `FZ-*`.

## Notes

- Initial collection without `PYTHONPATH=./python` failed during import by
  resolving a stale installed Triton package. All counted collections and
  executions above used `PYTHONPATH=./python`.
- The Round 56 dynamic copy report classifies direct dynamic copy descriptor
  rows as existing `FZ-20260421-0001` plus clean unsupported
  `warpx2::02_13` 2CTA boundaries. This lane did not create or run additional
  temporary dynamic copy probes after that report appeared.

## Final Classification

Primary non-overlap checked-in copy boundary result: `7 passed`, `0 failed`,
`0 skipped`, `0 xfailed`.

Supporting guardrails: broad checked-in no-scale copy selector `121 passed`;
structural control-flow contrast `2 passed, 8 xfailed`.

No new independent `FZ-*` was found. Backend repair remains deferred.
