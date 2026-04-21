# Round 29 Python Runtime Bitwidth Follow-Up

Date: 2026-04-21
Mode: discovery/cataloging only. No backend/compiler fixes attempted.

## Scope

This lane followed up on `FZ-20260421-0017` from the clean-boundary fuzzing
report. The goal was to verify Python/Gluon runtime reachability and split the
64-bit crash by consumer shape:

- full TMEM store/load roundtrip;
- TMEM store only;
- TMEM load only.

The probe runs each compiler-crashing case in a subprocess so an LLVM assertion
does not poison the controller process.

## Commands

Temporary probe:

```bash
/tmp/tmem_bitwidth_runtime_round29_probe.py
```

Validation and run:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile /tmp/tmem_bitwidth_runtime_round29_probe.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon python /tmp/tmem_bitwidth_runtime_round29_probe.py 2>&1 | tee /tmp/tmem_bitwidth_runtime_round29_probe.log
```

## Results

```text
roundtrip,float64,returncode=-6,class=ASSERT_BITWIDTH_32
roundtrip,int64,returncode=-6,class=ASSERT_BITWIDTH_32
store_only,float64,returncode=-6,class=ASSERT_BITWIDTH_32
store_only,int64,returncode=-6,class=ASSERT_BITWIDTH_32
load_only,float64,returncode=-6,class=ASSERT_BITWIDTH_32
load_only,int64,returncode=-6,class=ASSERT_BITWIDTH_32
```

All rows abort with `SIGABRT` (`returncode=-6`) and the
`lowerTMemLdSt` assertion:

```text
bitwidth == 32
```

## Classification

No new independent bucket beyond `FZ-20260421-0017`.

This follow-up broadens `FZ-0017` from a generic 64-bit roundtrip crash to a
consumer-level runtime matrix:

| Consumer shape | `float64` | `int64` |
| --- | --- | --- |
| store then load roundtrip | ASSERT_BITWIDTH_32 | ASSERT_BITWIDTH_32 |
| store only | ASSERT_BITWIDTH_32 | ASSERT_BITWIDTH_32 |
| load only | ASSERT_BITWIDTH_32 | ASSERT_BITWIDTH_32 |

The store-only and load-only rows show both sides of `lowerTMemLdSt` need a
clean 64-bit policy. This is not limited to initialized allocation lowering or
roundtrip composition.

## Next Hooks

- When the fuzzing phase ends, preserve `FZ-0017` with focused checked-in
  compiler/runtime regression tests before fixing the verifier or lowering.
- Add adjacent `i32`/`f32` subprocess controls if this report is promoted into
  a checked-in test, so the test documents the intended 32-bit boundary.
