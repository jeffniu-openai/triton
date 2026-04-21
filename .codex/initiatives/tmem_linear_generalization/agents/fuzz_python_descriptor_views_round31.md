# Round 31 Python Descriptor-View TMEM Fuzzing

Date: 2026-04-21

Mode: discovery/cataloging only. No backend or compiler code was modified.

Required first step:

```bash
make -j8
```

Result: no-op rebuild through
`/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`.

## Scope

Round 30 proved Python frontend reachability for `FZ-20260421-0017`, but the
first sliced-view Python harness had rough 32-bit controls because it sliced a
register tensor with unsupported Python syntax. This lane rebuilt the frontend
descriptor-view probe so the input tensor is already shaped like the view:

- allocate a larger TMEM parent;
- slice a descriptor view with `parent.slice(SLICE_START, N, dim=1)`;
- store/load a register tensor with the exact view shape;
- compare 32-bit roundtrips against the input.

The lane primarily checks whether Python descriptor-view kernels reproduce the
64-bit `lowerTMemLdSt` assertion while comparable 32-bit view kernels compile
and execute.

## Artifacts

Temporary artifacts:

- `/tmp/tmem_python_descriptor_round31/child.py`
- `/tmp/tmem_python_descriptor_round31/controller.py`
- `/tmp/tmem_python_descriptor_round31/run.log`
- `/tmp/tmem_python_descriptor_round31/summary.json`

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_python_descriptor_round31/controller.py
```

## Results

Overall:

| Class | Count |
| --- | ---: |
| `ASSERT_BITWIDTH_32` | 48 |
| `PASS` | 40 |
| `PYTHON_OR_COMPILE_ERROR` | 20 |

By dtype and CTA mode:

| Dtype / CTA mode | Result |
| --- | --- |
| `float64`, 1CTA | 18 `ASSERT_BITWIDTH_32` |
| `float64`, 2CTA | 6 `ASSERT_BITWIDTH_32`, 3 Python compile diagnostics |
| `int64`, 1CTA | 18 `ASSERT_BITWIDTH_32` |
| `int64`, 2CTA | 6 `ASSERT_BITWIDTH_32`, 3 Python compile diagnostics |
| `float32`, 1CTA | 18 pass |
| `float32`, 2CTA | 2 pass, 7 Python compile diagnostics |
| `int32`, 1CTA | 18 pass |
| `int32`, 2CTA | 2 pass, 7 Python compile diagnostics |

## Confirmed FZ-0017 Expansion

The valid 1CTA descriptor-view matrix reproduced the `bitwidth == 32`
assertion for every 64-bit row:

- dtypes: `torch.float64`, `torch.int64`;
- parent layouts: `TensorMemoryLinearLayout`, `TensorMemoryLayout`;
- modes: roundtrip, store-only, load-only;
- parent/view shape: parent `128x256`, view `128x128`;
- slice offsets: `0`, `64`, `128`.

The matching `torch.float32` and `torch.int32` rows passed, including roundtrip
correctness checks for the same layout kinds and slice offsets.

The valid 2CTA descriptor-view rows also reproduced `FZ-0017` for `float64`
and `int64` at in-bounds slice offsets `0` and `64`:

- parent/view shape: parent `256x128`, view `256x64`;
- layout: 2CTA `TensorMemoryLinearLayout`;
- modes: roundtrip, store-only, load-only.

That removes the Round 30 harness ambiguity for Python descriptor-sliced
64-bit views: these are ordinary Gluon kernels that reach the same TMEM
load/store lowering assertion.

## Non-New Diagnostics

The `PYTHON_OR_COMPILE_ERROR` rows are not new TMEM backend buckets.

Out-of-bounds 2CTA slice rows used `slice_start = 128` with `N = 64` and
`parent_n = 128`; they reject cleanly:

```text
failed to infer memdesc_subslice result type
subslice must stay within the source shape
```

Some 2CTA `float32`/`int32` load/roundtrip controls still hit the same frontend
or harness limitation seen in Round 30 when storing a CGA-layout tensor back to
global memory:

```text
'list' object has no attribute 'type'
```

The corresponding 2CTA store-only controls pass, and the 64-bit rows hit
`FZ-0017` before that global-store issue for in-bounds offsets. This lane does
not classify the 2CTA global-store diagnostic as a new TMEM backend bug.

## Classification

No new `FZ-*` bucket was created.

Expanded existing bucket:

- `FZ-20260421-0017`: Python/Gluon descriptor-view kernels with 64-bit TMEM
  load/store operands reproduce the `lowerTMemLdSt` `bitwidth == 32`
  assertion across legacy and linear parent layouts, multiple N-slice offsets,
  roundtrip/store-only/load-only consumers, and valid 1CTA/2CTA layouts.

Controls:

- 1CTA `float32`/`int32` descriptor-view roundtrip/store/load rows pass for
  the same parent/view shapes and slice offsets.
- 2CTA `float32`/`int32` store-only rows pass for in-bounds offsets.
