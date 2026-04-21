# Round 32 Local: scaled-MMAv5 2CTA runtime guardrail

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend/compiler code or checked-in tests
were edited.

## Scope

Reran the checked-in 2CTA scaled-MMAv5 runtime rows after the Round 31 scaled
operand fuzzing. This keeps the legal 2CTA surface green next to the
report-only scaled operand buckets:

- `FZ-20260421-0007`: dynamic accumulator-view selected scaled-MMAv5
  miscompile;
- `FZ-20260421-0013`: scale descriptor-view operand miscompile;
- `FZ-20260421-0015`: runtime-selected distinct direct B-scale operand
  miscompile.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Four-GPU runtime run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and twocta and not reports and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and twocta and not reports and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and twocta and not reports and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and twocta and not reports and not resource'
```

Logs:

- `/tmp/tmem_scaled_operand_round32_twocta_g0.log`
- `/tmp/tmem_scaled_operand_round32_twocta_g1.log`
- `/tmp/tmem_scaled_operand_round32_twocta_g2.log`
- `/tmp/tmem_scaled_operand_round32_twocta_g3.log`

## Result

- Group 1: `7 passed, 1608 deselected`
- Group 2: `7 passed, 1608 deselected`
- Group 3: `7 passed, 1608 deselected`
- Group 4: `7 passed, 1608 deselected`
- Aggregate selected result: `28 passed`

## Classification

No compiler crash, false unsupported diagnostic, opcode mismatch, runtime
miscompile, XPASS, or new independent `FZ-*` bucket was found. This is a green
2CTA scaled-MMAv5 guardrail around the Round 31 scaled operand failures, not a
backend repair.
