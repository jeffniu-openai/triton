# Round 32 Python Sub-32/32-bit TMEM Roundtrip Guardrail

Date: 2026-04-21

Mode: discovery/cataloging only. No backend or compiler code was modified.

Required first step:

```bash
make -j8
```

Result: no-op rebuild through
`/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`.

## Scope

After `FZ-20260421-0017` was expanded to 64-bit TMEM load/store, this lane
checked neighboring Python/Gluon runtime roundtrips for sub-32-bit and 32-bit
types. The goal was to find crashes or miscompiles that would indicate a
broader bitwidth bug rather than a 64-bit-only lowering policy gap.

The probe covered:

- dtypes: `float32`, `float16`, `bfloat16`, `int32`, `int16`, `int8`,
  `uint8`, and `bool`;
- shapes: `128x128`, `128x64`, and linear-only `256x32`;
- layouts: `TensorMemoryLinearLayout` and `TensorMemoryLayout` where valid;
- descriptor views: direct parent roundtrip and N-sliced parent view roundtrip;
- exact correctness checks with `torch.testing.assert_close(..., atol=0, rtol=0)`.

Legacy `256`-row cases were intentionally excluded because the legacy encoding
parser cleanly rejects `blockM = 256`; that boundary is unrelated to this
runtime bitwidth check.

## Artifacts

Temporary artifacts:

- `/tmp/tmem_sub32_round32/child.py`
- `/tmp/tmem_sub32_round32/controller.py`
- `/tmp/tmem_sub32_round32/run_g1.log`
- `/tmp/tmem_sub32_round32/run_g2.log`
- `/tmp/tmem_sub32_round32/run_g3.log`
- `/tmp/tmem_sub32_round32/run_g4.log`
- `/tmp/tmem_sub32_round32/summary_g1.json`
- `/tmp/tmem_sub32_round32/summary_g2.json`
- `/tmp/tmem_sub32_round32/summary_g3.json`
- `/tmp/tmem_sub32_round32/summary_g4.json`
- `/tmp/tmem_sub32_round32/summary_all.json`

Commands:

```bash
PYTHONPATH=.:./python:./python/test/gluon python -m py_compile \
  /tmp/tmem_sub32_round32/child.py \
  /tmp/tmem_sub32_round32/controller.py

PYTHONPATH=.:./python:./python/test/gluon CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  python /tmp/tmem_sub32_round32/controller.py --group 1 --splits 4 --gpu 0
PYTHONPATH=.:./python:./python/test/gluon CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  python /tmp/tmem_sub32_round32/controller.py --group 2 --splits 4 --gpu 1
PYTHONPATH=.:./python:./python/test/gluon CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  python /tmp/tmem_sub32_round32/controller.py --group 3 --splits 4 --gpu 2
PYTHONPATH=.:./python:./python/test/gluon CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  python /tmp/tmem_sub32_round32/controller.py --group 4 --splits 4 --gpu 3
```

## Result

Aggregate result:

```text
80 passed
```

By dtype:

| Dtype | Result |
| --- | ---: |
| `float32` | 10 passed |
| `float16` | 10 passed |
| `bfloat16` | 10 passed |
| `int32` | 10 passed |
| `int16` | 10 passed |
| `int8` | 10 passed |
| `uint8` | 10 passed |
| `bool` | 10 passed |

By shape:

| Shape | Result |
| --- | ---: |
| `128x128` | 32 passed |
| `128x64` | 32 passed |
| `256x32` | 16 passed |

No compiler crash, runtime mismatch, false unsupported diagnostic, or
unexpected assertion was observed.

## Classification

No new `FZ-*` bucket was found.

This is a green guardrail around `FZ-20260421-0017`: the observed bitwidth
crash remains isolated to encoded 64-bit `i64`/`f64` TMEM load/store planning.
Sub-32 and 32-bit Python/Gluon TMEM roundtrips, including descriptor views,
currently execute correctly for the covered shapes and layouts.
