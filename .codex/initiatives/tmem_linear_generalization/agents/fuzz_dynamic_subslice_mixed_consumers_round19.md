# Round 19 Lane AV: Dynamic `memdesc_subslice` SSA and Mixed Selected-Descriptor Consumers

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only; no backend/compiler fixes, no commit, no push.

## Scope

Stress runtime-selected `ttg.memdesc_subslice` SSA values when the same selected
descriptor feeds both:

- a direct `ttng.tmem_load` consumer; and
- a `tcgen05_mma` or `tcgen05_mma_scaled` accumulator consumer in the same
  kernel.

The intent is to distinguish generic descriptor-selection failures from stale
or rematerialized MMA accumulator-address paths. Both tested kernels branch at
runtime between two sibling column subslices of one `[M, 2 * N]` TMEM parent,
initialize the low view with `3.0`, initialize the high view with `7.0`, load
from the selected view before MMA, then use the same selected descriptor as the
MMA accumulator with `use_acc=True`.

## Commands

Build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Temporary probe, branch-only first pass:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py
```

Result:

```text
..F.
FAILED /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py::test_scaled_selected_subslice_load_and_mma[0]
1 failed, 3 passed in 4.61s
```

Extended probe after adding helper-returned and loop-carried accumulator
descriptor variants:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py
```

Result:

```text
..FFF...
FAILED /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py::test_scaled_selected_subslice_load_and_mma[0-branch]
FAILED /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py::test_scaled_selected_subslice_load_and_mma[0-helper]
FAILED /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py::test_scaled_selected_subslice_load_and_mma[0-loop]
3 failed, 5 passed in 5.66s
```

Extended probe after adding selected direct B-scale descriptor mixed load+MMA
variants:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short \
  /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py
```

Result:

```text
..FFF.........
FAILED /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py::test_scaled_selected_subslice_load_and_mma[0-branch]
FAILED /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py::test_scaled_selected_subslice_load_and_mma[0-helper]
FAILED /tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py::test_scaled_selected_subslice_load_and_mma[0-loop]
3 failed, 11 passed in 3.22s
```

Mismatch summarizer:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python - <<'PY'
import importlib.util
import torch

spec = importlib.util.spec_from_file_location(
    'probe', '/tmp/tmem_dynamic_subslice_mixed_consumers_round19_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

rows = []
for family in ('plain', 'scaled'):
    for selector in (0, 1):
        torch.manual_seed((1901 if family == 'plain' else 1902) + selector)
        m = n = k = 128
        layout = probe._make_tmem_linear_layout(m, 2 * n)
        selector_t = torch.tensor(selector, device='cuda', dtype=torch.int32)
        load_out = torch.empty((m, n), device='cuda', dtype=torch.float32)
        mma_out = torch.empty((m, n), device='cuda', dtype=torch.float32)
        init = 3.0 if selector == 0 else 7.0
        if family == 'plain':
            a = torch.randn((m, k), device='cuda', dtype=torch.float16)
            b = torch.randn((k, n), device='cuda', dtype=torch.float16)
            compiled = probe._plain_selected_subslice_load_and_mma_kernel[(1,)](
                a, b, load_out, mma_out, selector_t, m, n, k, layout, num_warps=4)
            expected = torch.matmul(a.to(torch.float32), b.to(torch.float32)) + init
            atol = rtol = 1e-2
        else:
            a, a_scale, a_ref = probe.random_quantized_tensor(m, k, 'mxfp8')
            b, b_scale, b_ref = probe.random_quantized_tensor(n, k, 'mxfp8')
            compiled = probe._scaled_selected_subslice_load_and_mma_kernel[(1,)](
                mma_out, load_out, selector_t, m, n, k, a, b, a_scale, b_scale,
                layout, num_warps=4)
            expected = a_ref @ b_ref.T + init
            atol = rtol = 1e-3
        load_ok = torch.allclose(load_out, torch.full_like(load_out, init), atol=0, rtol=0)
        finite = torch.isfinite(mma_out) & torch.isfinite(expected)
        close = torch.isclose(mma_out, expected, atol=atol, rtol=rtol, equal_nan=False)
        mismatches = int((~close).sum().item())
        max_abs = float(torch.max(torch.abs(torch.where(finite, mma_out - expected,
                                                        torch.zeros_like(mma_out)))).item())
        has_nan = bool(torch.isnan(mma_out).any().item())
        rows.append((family, selector, load_ok, mismatches, max_abs, has_nan,
                     len(probe._extract_mma_ops(compiled.asm['ptx']))))
for row in rows:
    print(row)
PY
```

Initial branch-only result:

```text
('plain', 0, True, 0, 2.288818359375e-05, False, 8)
('plain', 1, True, 0, 3.0517578125e-05, False, 8)
('scaled', 0, True, 4413, 3.3962024765942725e+38, True, 4)
('scaled', 1, True, 0, 3.0517578125e-05, False, 4)
```

Extended result:

```text
('plain', 'branch', 0, True, 0, False, 8)
('plain', 'branch', 1, True, 0, False, 8)
('scaled', 'branch', 0, True, 4413, True, 4)
('scaled', 'branch', 1, True, 0, False, 4)
('scaled', 'helper', 0, True, 4413, True, 4)
('scaled', 'helper', 1, True, 0, False, 4)
('scaled', 'loop', 0, True, 4413, True, 4)
('scaled', 'loop', 1, True, 0, False, 4)
```

Selected B-scale result:

```text
('bscale', 'branch', 0, 0, 0, False, 4)
('bscale', 'branch', 1, 0, 0, False, 4)
('bscale', 'helper', 0, 0, 0, False, 4)
('bscale', 'helper', 1, 0, 0, False, 4)
('bscale', 'loop', 0, 0, 0, False, 4)
('bscale', 'loop', 1, 0, 0, False, 4)
```

## Row Table

| Row | Family | Runtime selector | Pre-MMA selected `tmem_load` | MMA accumulator result | PTX MMA ops | Classification |
| --- | --- | ---: | --- | --- | ---: | --- |
| AV-001 | Plain MMAv5 | 0, low subslice | pass, all `3.0` | pass | 8 | Green control |
| AV-002 | Plain MMAv5 | 1, high subslice | pass, all `7.0` | pass | 8 | Green control |
| AV-003 | Scaled MMAv5 | 0, low subslice | pass, all `3.0` | miscompile, `4413/16384` mismatches, NaNs present | 4 | Existing `FZ-20260421-0007` |
| AV-004 | Scaled MMAv5 | 1, high subslice | pass, all `7.0` | pass | 4 | Green contrast |
| AV-005 | Scaled MMAv5 helper-returned descriptor | 0, low subslice | pass, all `3.0` | miscompile, `4413/16384` mismatches, NaNs present | 4 | Existing `FZ-20260421-0007` |
| AV-006 | Scaled MMAv5 helper-returned descriptor | 1, high subslice | pass, all `7.0` | pass | 4 | Green contrast |
| AV-007 | Scaled MMAv5 loop-carried descriptor | 0, low subslice | pass, all `3.0` | miscompile, `4413/16384` mismatches, NaNs present | 4 | Existing `FZ-20260421-0007` |
| AV-008 | Scaled MMAv5 loop-carried descriptor | 1, high subslice | pass, all `7.0` | pass | 4 | Green contrast |
| AV-009 | Scaled MMAv5 direct B-scale descriptor | 0/1 branch | pass, `0/512` scale byte mismatches | pass | 4 | Green contrast to `FZ-0015` |
| AV-010 | Scaled MMAv5 direct B-scale descriptor | 0/1 helper | pass, `0/512` scale byte mismatches | pass | 4 | Green contrast to `FZ-0015` |
| AV-011 | Scaled MMAv5 direct B-scale descriptor | 0/1 loop | pass, `0/512` scale byte mismatches | pass | 4 | Green contrast to `FZ-0015` |

## Classification

No new independent `FZ-*` bucket is needed.

`AV-003` expands `FZ-20260421-0007`:

- the selected descriptor is a runtime SSA value produced by branching between
  two `memdesc_subslice` siblings;
- the exact same selected descriptor feeds a `ttng.tmem_load` and a
  `ttng.tc_gen5_mma_scaled` accumulator consumer;
- the `tmem_load` observes the correct selected subslice initialization, so the
  dynamic descriptor is not globally wrong;
- plain MMAv5 with the same mixed load+MMA shape passes for both selectors;
- branch, helper-returned, and loop-carried scaled MMAv5 rows all fail only for
  the low-subslice dynamic selector, matching the previously cataloged
  `FZ-20260421-0007` low-subslice accumulator pattern.

This points more specifically at scaled-MMAv5 accumulator lowering/address
materialization for dynamically selected low subslices, not at generic
`memdesc_subslice` SSA selection or all MMA consumers.

The direct B-scale mixed-consumer rows did not reproduce `FZ-20260421-0015`.
They branch/helper/loop select between two independent direct
`TensorMemoryScalesLayout` descriptors, use the selected descriptor for both a
scale `tmem_load` and the scaled-MMA B-scale operand, and pass for both
selectors. This is a useful negative contrast: the `FZ-0015` trigger likely
requires the more specific selected B-scale shape already minimized in the
Round 15 reports, rather than any selected direct B-scale descriptor with an
extra load user.

## Next Probes

- Re-run the exact Round 15 `FZ-0015` selected-B-scale minimal shape with this
  lane's scale-load side channel to see which shape/detail is missing from
  `AV-009` through `AV-011`.
- Dump TTGIR/LLVM for `AV-003` and `AV-004` side by side if repair mode begins,
  focusing on the scaled-MMA accumulator address operand and any low-subslice
  rematerialization before `tcgen05.mma`.
