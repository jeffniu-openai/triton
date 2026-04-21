# Round 53 TMEM structural/runtime fuzz lane A

Date: 2026-04-21 15:12 UTC
Branch: `codex/tmem`
HEAD before report: `2e0211daa`
Scope: discovery/cataloging only; no backend fixes or checked-in test changes.

## Required Rebuild

Command:

```bash
make -j8
```

Result: `ninja: no work to do.`

## Structural Fuzzer

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_structural_fuzzer.py
```

Collection result: `33 tests collected`.

Run commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_structural_fuzzer.py
```

Results:

- Group 1: `5 passed, 24 deselected, 4 xfailed`
- Group 2: `2 passed, 24 deselected, 7 xfailed`
- Group 3: `2 passed, 24 deselected, 7 xfailed`
- Group 4: `27 deselected, 6 xfailed`
- Aggregate selected result: `9 passed, 24 xfailed`

Classification:

- No XPASS drift.
- No unexpected compiler crash, verifier over-strictness, false unsupported
  diagnostic, or runtime miscompile.
- The visible group 3 and group 4 diagnostics match existing
  `FZ-20260421-0001`: dynamic TMEM `ttg.memdesc_index` reaches
  `ConvertTritonGPUToLLVM` as an illegal op.
- The remaining xfails stay within existing checked-in structural buckets:
  `FZ-20260421-0003`, `FZ-20260421-0004`, `FZ-20260421-0005`,
  `FZ-20260421-0006`, `FZ-20260421-0007`, `FZ-20260421-0008`, and
  `FZ-20260421-0009`.

## Higher-Rank Descriptor Runtime Matrix

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q -k "ldst_descriptor_rank5 or ldst_twocta_descriptor_rank5 or ldst_descriptor_higher_rank or ldst_twocta_descriptor_higher_rank or ldst_direct_higher_rank" python/test/gluon/test_tmem_runtime_matrix.py
```

Collection result: `79/1615 tests collected`.

Run commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 -k "ldst_descriptor_rank5 or ldst_twocta_descriptor_rank5 or ldst_descriptor_higher_rank or ldst_twocta_descriptor_higher_rank or ldst_direct_higher_rank" python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 -k "ldst_descriptor_rank5 or ldst_twocta_descriptor_rank5 or ldst_descriptor_higher_rank or ldst_twocta_descriptor_higher_rank or ldst_direct_higher_rank" python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 -k "ldst_descriptor_rank5 or ldst_twocta_descriptor_rank5 or ldst_descriptor_higher_rank or ldst_twocta_descriptor_higher_rank or ldst_direct_higher_rank" python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 -k "ldst_descriptor_rank5 or ldst_twocta_descriptor_rank5 or ldst_descriptor_higher_rank or ldst_twocta_descriptor_higher_rank or ldst_direct_higher_rank" python/test/gluon/test_tmem_runtime_matrix.py
```

Results:

- Group 1: `20 passed, 1595 deselected`
- Group 2: `16 passed, 4 skipped, 1595 deselected`
- Group 3: `4 passed, 16 skipped, 1595 deselected`
- Group 4: `19 passed, 1596 deselected`
- Aggregate selected result: `59 passed, 20 skipped`

Classification: no compiler crash, verifier drift, unsupported diagnostic
regression, clean-boundary drift, or runtime miscompile. Skips are existing
runtime capability/resource guards.

## Scaled/Narrow Accumulator Descriptor Runtime Matrix

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q -k "mma_scaled_acc_n16 or bscale_descriptor_view or bscale_view_extra_user or indexed_acc_identity_narrow or acc_identity_narrow or acc_tile_permuted_32" python/test/gluon/test_tmem_runtime_matrix.py
```

Collection result: `29/1615 tests collected`.

Run commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 -k "mma_scaled_acc_n16 or bscale_descriptor_view or bscale_view_extra_user or indexed_acc_identity_narrow or acc_identity_narrow or acc_tile_permuted_32" python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 -k "mma_scaled_acc_n16 or bscale_descriptor_view or bscale_view_extra_user or indexed_acc_identity_narrow or acc_identity_narrow or acc_tile_permuted_32" python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 -k "mma_scaled_acc_n16 or bscale_descriptor_view or bscale_view_extra_user or indexed_acc_identity_narrow or acc_identity_narrow or acc_tile_permuted_32" python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 -k "mma_scaled_acc_n16 or bscale_descriptor_view or bscale_view_extra_user or indexed_acc_identity_narrow or acc_identity_narrow or acc_tile_permuted_32" python/test/gluon/test_tmem_runtime_matrix.py
```

Results:

- Group 1: `8 passed, 1607 deselected`
- Group 2: `8 passed, 1607 deselected`
- Group 3: `8 passed, 1607 deselected`
- Group 4: `5 passed, 1610 deselected`
- Aggregate selected result: `29 passed`

Classification: no compiler crash, verifier drift, false unsupported
diagnostic, clean diagnostic drift, or runtime miscompile.

## Disposable Structural Permutation Probe

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon python - <<'PY'
import traceback
import torch
import test_tmem_structural_fuzzer as mod

cases = [
    ("ldst-extra-row-reverse-col-even-chain0-128x64-32x32b", lambda: mod._run_ldst_case(mod.LdStCase("round53-ldst-extra-0", 0x530001, 128, 64, "reverse", "even_odd", "32x32b", 0))),
    ("ldst-extra-row-rotate-col-reverse-chain1-128x128-16x128b", lambda: mod._run_ldst_case(mod.LdStCase("round53-ldst-extra-1", 0x530002, 128, 128, "rotate1", "reverse", "16x128b", 1))),
    ("ldst-extra-row-even-col-rotate-chain2-128x64-16x64b", lambda: mod._run_ldst_case(mod.LdStCase("round53-ldst-extra-2", 0x530003, 128, 64, "even_odd", "rotate1", "16x64b", 2))),
    ("ldred-extra-direct-row-reverse-col-even-128x64-max", lambda: mod.test_tmem_structural_fuzzer_ldred(mod.LdRedCase("round53-ldred-extra-0", 0x530101, 128, 64, False, 0, "reverse", "even_odd", "max"))),
    ("ldred-extra-direct-row-rotate-col-reverse-128x64-min", lambda: mod.test_tmem_structural_fuzzer_ldred(mod.LdRedCase("round53-ldred-extra-1", 0x530102, 128, 64, False, 1, "rotate1", "reverse", "min"))),
]

passed = failed = 0
for name, fn in cases:
    try:
        fn()
        torch.cuda.synchronize()
        print(f"PASS {name}")
        passed += 1
    except BaseException as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=4)
        failed += 1
print(f"SUMMARY passed={passed} failed={failed}")
raise SystemExit(1 if failed else 0)
PY
```

Result:

- `PASS ldst-extra-row-reverse-col-even-chain0-128x64-32x32b`
- `PASS ldst-extra-row-rotate-col-reverse-chain1-128x128-16x128b`
- `PASS ldst-extra-row-even-col-rotate-chain2-128x64-16x64b`
- `PASS ldred-extra-direct-row-reverse-col-even-128x64-max`
- `PASS ldred-extra-direct-row-rotate-col-reverse-128x64-min`
- `SUMMARY passed=5 failed=0`

Seeds and case ids:

- `round53-ldst-extra-0`, seed `0x530001`
- `round53-ldst-extra-1`, seed `0x530002`
- `round53-ldst-extra-2`, seed `0x530003`
- `round53-ldred-extra-0`, seed `0x530101`
- `round53-ldred-extra-1`, seed `0x530102`

Classification: no new `FZ-*` candidate.

## Summary

- Checked-in selected runtime rows: `141` total, with `97 passed`,
  `20 skipped`, and `24 expected xfails`.
- Disposable probe rows: `5 passed`.
- New failures: none.
- New `FZ-*` candidates: none.
- Existing failures were observed only through checked-in xfail sentinels,
  primarily `FZ-20260421-0001` dynamic `memdesc_index` diagnostics.
