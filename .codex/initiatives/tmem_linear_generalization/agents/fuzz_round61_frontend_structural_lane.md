# Round 61 Lane B: frontend plus structural adversarial fuzzer

Date: 2026-04-21 16:23 UTC
Branch: `codex/tmem`
HEAD before report commit: `a35ca5bc67e0`
Mode: discovery/cataloging only. No backend, compiler, frontend, or checked-in
test source was edited.

## Scope

This lane exercised frontend tensor-memory, descriptor, and layout diagnostic
coverage in `python/test/gluon/test_frontend.py`, then swept the checked-in
structural TMEM fuzzer in `python/test/gluon/test_tmem_structural_fuzzer.py`.
The objective was to detect compiler crashes, verifier assertions that should
be clean diagnostics, unexpected xpass/xfail transitions, and frontend
expectation drift distinct from `FZ-20260421-0023`.

## Required build

```bash
make -j8
```

Result: build-tree ninja reported `no work to do`.

## Frontend collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_frontend.py \
  -k 'tensor_memory or tmem_'
```

Result: `29/225` collected.

The collected nodeids covered tensor-memory allocation/load/store IR,
linear-view and descriptor-chain IR, clean diagnostic paths for unsupported
view loads, bitcast size mismatch, backend `4x256b` descriptor-layout reasons,
higher-rank descriptor `get_reg_layout`, invalid linear-layout construction,
non-surjective layout parsing, MMAv5 layout diagnostics, scale descriptor
layout diagnostics, and constexpr TMEM index/subslice/reduction layout checks.

## Frontend focused run

```bash
PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_frontend.py \
  -k 'tensor_memory or tmem_'
```

Result: `28 passed, 1 failed, 196 deselected`.

Failing node:

```text
python/test/gluon/test_frontend.py::test_tmem_subslice_reg_layout_constexpr
```

Failure signature:

```text
Expected #linear register basis:
[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]]

Actual #linear register basis:
[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]]
```

Exact rerun:

```bash
PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_frontend.py::test_tmem_subslice_reg_layout_constexpr
```

Result: `1 failed` with the same inline `expecttest` mismatch.

Classification: existing `FZ-20260421-0023`. This is stable frontend
expectation drift in the TMEM subslice register-layout text; it is not a new
compiler crash, verifier assertion, clean-diagnostic regression, runtime
miscompile, or independent frontend drift bucket.

## Structural fuzzer collection

Full structural fuzzer:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `33` tests collected.

Adversarial selector:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'generic_pass or descriptor_view or ldred or scaled_mma or ldst_view or copy_scales'
```

Result: `32/33` collected. The one deselected row was the dedicated
`test_tmem_structural_fuzzer_ldst_256row_lifted_parent_allocator_crash`
sentinel; it remained covered by the full structural sweep below.

## Structural fuzzer split run

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Results:

```text
group 1: 5 passed, 24 deselected, 4 xfailed
group 2: 2 passed, 24 deselected, 7 xfailed
group 3: 2 passed, 24 deselected, 7 xfailed
group 4: 27 deselected, 6 xfailed
aggregate: 9 passed, 24 xfailed
```

The visible MLIR reproducer output came from expected xfail rows with late
illegal `ttg.memdesc_index` signatures already cataloged under
`FZ-20260421-0001`. The remaining xfail inventory matched checked-in reasons:
`FZ-20260421-0002`, `FZ-20260421-0003`, `FZ-20260421-0004`,
`FZ-20260421-0005`, `FZ-20260421-0006`, `FZ-20260421-0007`,
`FZ-20260421-0008`, `FZ-20260421-0009`, and the existing `R5-C` loop-carried
auto-layout inference bucket.

## Classification

No new independent `FZ-*` bucket was found.

- Frontend: reproduced existing `FZ-20260421-0023` only. All other collected
  tensor-memory/descriptor/layout diagnostic nodeids passed.
- Structural fuzzer: no XPASS, unexpected failure, verifier assertion outside
  expected xfails, compiler crash outside expected xfails, runtime miscompile,
  hang, or clean-diagnostic drift.
- No backend repair or expectation update was attempted.
