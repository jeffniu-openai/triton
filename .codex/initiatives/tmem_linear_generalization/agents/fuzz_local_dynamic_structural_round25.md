# Round 25 Local: Dynamic Descriptor Structural Guardrail

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler fixes.
- Logs:
  - `/tmp/tmem_r25_dynamic_structural_g1.log`
  - `/tmp/tmem_r25_dynamic_structural_g2.log`
  - `/tmp/tmem_r25_dynamic_structural_g3.log`
  - `/tmp/tmem_r25_dynamic_structural_g4.log`
- Durations:
  `/tmp/tmem_r25_dynamic_structural_durations.json`

## Scope

This local guardrail reran the checked-in dynamic descriptor structural rows
after the Round 25 dynamic generator prototype.  The goal was to check whether
any maintained xfail changed mode or unexpectedly passed.

Selector:

```text
generic_pass or dynamic_index or loop_carried or scaled_mma_acc_subslice
```

## Commands

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'generic_pass or dynamic_index or loop_carried or scaled_mma_acc_subslice'
```

Result:

```text
12/33 tests collected
```

Split-4 runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <1..4> \
  --store-durations \
  --durations-path /tmp/tmem_r25_dynamic_structural_durations.json \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'generic_pass or dynamic_index or loop_carried or scaled_mma_acc_subslice'
```

## Results

Aggregate result: `12 xfailed`.

Per-shard results:

- GPU 0 / group 1: `3 xfailed, 30 deselected`.
- GPU 1 / group 2: `3 xfailed, 30 deselected`.
- GPU 2 / group 3: `3 xfailed, 30 deselected`.
- GPU 3 / group 4: `3 xfailed, 30 deselected`.

Observed diagnostics match existing buckets:

- dynamic index rows still fail with illegal `ttg.memdesc_index` at LLVM
  conversion, matching `FZ-20260421-0001`;
- generic branch/helper/loop-carried descriptor chain rows still xfail under
  the checked-in `FZ-20260421-0002` expectations;
- the scaled accumulator subslice control-flow row remains an expected
  `FZ-20260421-0007` xfail.

## Classification

No unexpected pass, changed crash mode, runtime miscompile outside the existing
xfail set, or new independent `FZ-*` bucket was found.

No backend/compiler code was changed.
