# Lane A2 round 2: ld/st and ld.red promotion

- Time: 2026-04-21 09:10 UTC
- Branch/HEAD before commit: `codex/tmem`
- Lane: ld/st and ld.red structural fuzzing promotion
- Mode: discovery only; no compiler/backend repair

## Scope

Promoted lane A round-1 findings `FZ-20260421-0003`,
`FZ-20260421-0004`, `FZ-20260421-0005`, and `FZ-20260421-0006` into
checked-in Python runtime xfail coverage in
`python/test/gluon/test_tmem_structural_fuzzer.py`.

The file already had concurrent generic-pass xfail additions for
`FZ-20260421-0001` and `FZ-20260421-0002`; those edits were preserved and this
round layered only the lane A ld/st and ld.red cases on top.

## Promoted Cases

### FZ-20260421-0003: ld/st descriptor-view chain1 miscompile

- Checked-in nodeid:
  `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_descriptor_view_read[ldst-fz20260421-0003-chain1-64x32-32x32b]`
- Shape: parent `[2, 64, 32]`, indexed view `[64, 32]`.
- Layout: identity `TensorMemoryLinearLayout`.
- View chain:
  `index(1).reshape((M//2,2,N)).permute([1,0,2]).reshape((M,N))`.
- Expected fixed behavior: runtime output matches PyTorch/reference and emits
  direct `tcgen05.ld/st` instructions for the descriptor view.
- Current observed behavior: strict xfail due runtime output mismatch.
- Minimization note: the existing roundtrip fuzzer kernel did not expose this
  case because a load/store back through the same view can hide the bad read.
  The promoted repro reads from the descriptor chain and stores directly to
  global output, matching the round-1 probe.

### FZ-20260421-0004: ld.red descriptor chain emits plain ld plus software reduce

- Checked-in nodeid:
  `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-chain1-64x32-min]`
- Shape: parent `[2, 64, 32]`, indexed view `[64, 32]`.
- Layout: identity `TensorMemoryLinearLayout`.
- View chain:
  `index(1).reshape((M//2,2,N)).permute([1,0,2]).reshape((M,N))`.
- Expected fixed behavior: output and reduction match PyTorch and PTX/LLIR
  contain `tcgen05.ld.red.sync.aligned...`.
- Current observed behavior: output and reduction match, but the strict opcode
  assertion xfails because the generated code contains plain `tcgen05.ld...`.

### FZ-20260421-0005: 256-row lifted parent allocator assertion

- Checked-in nodeid:
  `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_256row_lifted_parent_allocator_crash`
- Shape: parent `[2, 256, 32]`, indexed view `[256, 32]`.
- Layout: identity `TensorMemoryLinearLayout`.
- View chain: direct `index(1)`.
- Expected fixed behavior: supported lowering or a clean Python-visible
  resource/unsupported diagnostic.
- Current observed behavior: `TensorMemoryAllocation.cpp:65`
  `MemoryBitMap::findFirstFit(...): Assertion 'kNumRows - numRows >= 0' failed.`
- Minimization note: the checked-in test runs the compile in a subprocess and
  asserts success under a strict xfail marker so the main pytest process
  survives the current C++ assertion.

### FZ-20260421-0006: ld.red transpose/slice false-unsupported candidate

- Checked-in nodeid:
  `python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0006-rotate1-transpose-slice-max]`
- Shape: parent `[2, 64, 128]`, indexed view `[64, 128]`.
- Layout: row `rotate1`, col `identity`.
- View chain:
  `index(1).permute([1,0]).permute([1,0]).slice(0,M,dim=0).slice(0,N,dim=1)`.
- Expected fixed behavior: if ISA-realizable, compile and execute `load_max`
  with hardware `tcgen05.ld.red...`; otherwise produce a clean, intentional
  boundary that the planner can classify.
- Current observed behavior: strict xfail on `view.get_reg_layout()` row-anchor
  diagnostic requiring row anchors `32,64`.

## Validation

- Required rebuild:
  `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8`
  - Result: ninja reported no work to do.
- Syntax check:
  `PYTHONPATH=.:./python python -m py_compile python/test/gluon/test_tmem_structural_fuzzer.py`
  - Result: passed.
- Exact nodeid validation:
  ```bash
  CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short \
    'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_descriptor_view_read[ldst-fz20260421-0003-chain1-64x32-32x32b]' \
    'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-chain1-64x32-min]' \
    'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0006-rotate1-transpose-slice-max]' \
    python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_256row_lifted_parent_allocator_crash
  ```
  - Result: `4 xfailed`.
- Full structural fuzzer validation:
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py`
  - Result: `9 passed, 9 xfailed`.

## Remaining Work

- Optional minimization for `FZ-20260421-0005`: capture the MLIR reproducer to
  `/tmp/laneA_256row_tmem_alloc.mlir` and rerun with
  `triton-opt --run-reproducer`.
- Expand `FZ-20260421-0006` over adjacent row/col permutations and packet
  shapes before classifying it as either true ISA boundary or false
  unsupported.
- Do not repair these backend issues until the fuzzing campaign pivots from
  discovery to fixing.
