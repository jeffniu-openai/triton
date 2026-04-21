# Round 60 Local: frontend tensor-memory diagnostic lane

Date: 2026-04-21
Branch: `codex/tmem`
HEAD before report commit: `b5d0cd781`
Mode: discovery/cataloging only. No backend, compiler, frontend, or checked-in test code was edited.

## Scope

This local lane sampled frontend-only tensor-memory construction, IR, and
diagnostic coverage in `python/test/gluon/test_frontend.py`. It was run after
the Round 59 runtime/lit lanes and is intentionally catalog-only.

## Required build

```bash
make -j8
```

Result: `ninja: no work to do`.

## Broad collection

The initial broad selector was too wide for a focused TMEM lane:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_frontend.py \
  -k 'tmem or tensor_memory or descriptor or layout'
```

Result: `97/225` collected, including many non-TMEM layout tests. The lane was
narrowed to explicit tensor-memory names.

## Focused collection

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_frontend.py \
  -k 'tensor_memory or tmem_'
```

Result: `29/225` collected.

## Focused runtime

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

## Classification

Candidate frontend expectation drift: `FZ-20260421-0023`.

The failure is not a runtime miscompile or compiler crash; it is a stable
frontend inline-expectation mismatch for the register basis returned by a TMEM
subslice descriptor's register layout. The actual IR has a wider column
register basis than the checked-in expected text. Backend repair and test
expectation updates remain deferred while the fuzzing campaign continues.
