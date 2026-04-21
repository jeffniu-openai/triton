# Round 43: Copy, Load/Store, and Lifetime Interleaving Fuzzing

Date: 2026-04-21 14:24 UTC

Branch: `codex/tmem`

HEAD before report: `1b564a2a0d85ba3625a3863ea431678540e6c856`

Scope: discovery/cataloging only. No backend or compiler code was modified.

## Objective

Adversarially fuzz mixed TMEM copy and load/store lifetime behavior:

- `tcgen05.copy` followed by TMEM `ld/st` in the same lifetime window;
- TMEM `ld/st` followed by `tcgen05.copy`;
- descriptor views reused across copy, store, and load;
- source rematerialization and dense-shared copy recovery;
- allocation lifetime pressure with multiple live TMEM allocations;
- clean diagnostics after positive copy/ldst rows.

## Build

Required rebuild before tests:

```bash
make -j8
```

Result: build was current (`ninja: no work to do`).

## Checked-In Runtime Selector

Initial collection without `PYTHONPATH=./python` imported an unrelated installed
Triton and failed during pytest collection. No test rows ran. All real
classification commands below used the checkout's rebuilt Python package.

Collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python \
  pytest --collect-only -q \
  -k '((cp_no_scales or ldst or alloc_lifetime or allocation or source_initialization) and (descriptor or view or rematerializes or dense_shared or clean or positive or twocta or ldst)) and not reports and not resource and not m64' \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `516/1648` rows collected.

Four-GPU split run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python pytest -s --tb=short --splits 4 --group 1 -k '((cp_no_scales or ldst or alloc_lifetime or allocation or source_initialization) and (descriptor or view or rematerializes or dense_shared or clean or positive or twocta or ldst)) and not reports and not resource and not m64' python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=./python pytest -s --tb=short --splits 4 --group 2 -k '((cp_no_scales or ldst or alloc_lifetime or allocation or source_initialization) and (descriptor or view or rematerializes or dense_shared or clean or positive or twocta or ldst)) and not reports and not resource and not m64' python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=./python pytest -s --tb=short --splits 4 --group 3 -k '((cp_no_scales or ldst or alloc_lifetime or allocation or source_initialization) and (descriptor or view or rematerializes or dense_shared or clean or positive or twocta or ldst)) and not reports and not resource and not m64' python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_tmem_structural_fuzzer.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=./python pytest -s --tb=short --splits 4 --group 4 -k '((cp_no_scales or ldst or alloc_lifetime or allocation or source_initialization) and (descriptor or view or rematerializes or dense_shared or clean or positive or twocta or ldst)) and not reports and not resource and not m64' python/test/gluon/test_tmem_runtime_matrix.py python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `414 passed, 98 skipped, 4 xfailed`.

Split details:

- group 1: `120 passed, 9 skipped`;
- group 2: `60 passed, 69 skipped`;
- group 3: `109 passed, 20 skipped`;
- group 4: `125 passed, 4 xfailed`.

The xfails are the checked-in structural `ld/st` descriptor-view rows for
existing `FZ-20260421-0003`; there were no XPASS transitions.

## Temporary Interleaving Probe

Temporary path:

```text
/tmp/tmem_copy_ldst_lifetime_round43_probe.py
```

The probe defines standalone Gluon kernels, avoiding imports from checked-in
test helper files, and runs four runtime rows:

| Case | Coverage | Expected output | Result |
| --- | --- | --- | --- |
| `copy_then_ldst` | copy into TMEM, load, store incremented value, reload | `inp + 5` | pass |
| `ldst_then_copy` | store incremented value, then copy original shared source over it | `inp` | pass |
| `descriptor_reuse` | copy into `tmem.index(1)`, load/store same view, reload through fresh `index(1)` | `inp + 7` | pass |
| `lifetime_pressure` | one copy-backed TMEM allocation plus one ld/st-backed allocation live together | `inp + (inp + 13)` | pass |

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=./python \
  python /tmp/tmem_copy_ldst_lifetime_round43_probe.py
```

Result:

```text
SUMMARY [{'case': 'copy_then_ldst', 'ptx_cp': 16, 'ptx_st': 17, 'ptx_ld': 18}, {'case': 'ldst_then_copy', 'ptx_cp': 16, 'ptx_st': 17, 'ptx_ld': 17}, {'case': 'descriptor_reuse', 'ptx_cp': 16, 'ptx_st': 17, 'ptx_ld': 18}, {'case': 'lifetime_pressure', 'ptx_cp': 16, 'ptx_st': 17, 'ptx_ld': 18}]
```

All four rows compiled, executed, and matched the runtime oracle. The PTX/LLIR
contained the expected `tcgen05.cp.cta_group::1.128x256b`, TMEM store, TMEM
load, alloc, relinquish, and dealloc signals.

## Clean-Diagnostic Selector

After the positive and interleaving rows, I ran a report-inclusive clean
diagnostic selector over copy and ld/st boundaries:

Collection:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=./python \
  pytest --collect-only -q \
  -k '(clean_unsupported or reports_clean or clean_error or tmem_oor or row_permuted_destination or subword_dtypes_report_clean_error or linear_exotic_reports_clean_unsupported) and (cp_no_scales or ldst or alloc) and not mma and not scaled and not ld_red' \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_core.py
```

Result: `85/19729` rows collected.

Run:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=./python \
  pytest -s --tb=short \
  -k '(clean_unsupported or reports_clean or clean_error or tmem_oor or row_permuted_destination or subword_dtypes_report_clean_error or linear_exotic_reports_clean_unsupported) and (cp_no_scales or ldst or alloc) and not mma and not scaled and not ld_red' \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_core.py
```

Result: `85 passed`.

These rows covered TMEM OOR diagnostics, descriptor-view unsupported diagnostics,
subword `warpx2` copy diagnostics, row/column-permuted copy diagnostics,
4x256b refresh-layout boundaries, and two-CTA `warpx2::02_13` clean negatives.

## Classification

No new independent `FZ-*` bucket is proposed.

Observed:

- no compiler crash in checked-in positives, clean diagnostics, or the temp
  interleaving probe;
- no runtime miscompile in the copy/ldst/lifetime rows;
- no opcode absence in the temp probe or checked-in opcode assertions;
- no false unsupported diagnostic in the positive rows;
- no clean-boundary drift in report-inclusive copy/ldst diagnostics;
- no unexpected xfail/pass transition.

Overlap analysis against requested buckets:

- `FZ-20260421-0001`: not reproduced. The temp probe used static descriptor
  views and did not leave runtime `ttg.memdesc_index` to LLVM conversion.
- `FZ-20260421-0002`: not reproduced. No helper/control-flow/tuple
  descriptor-view wrong-result signature appeared.
- `FZ-20260421-0003`: the only active overlap was the expected structural
  `ld/st` descriptor-view xfail set (`4 xfailed` in split group 4).
- `FZ-20260421-0014`: not reproduced. The interleaving probe used one 1CTA
  copy-tracked mbarrier at a time, so it did not hit the proxy-fence insertion
  failure family.
- `FZ-20260421-0017`: not reproduced. This lane stayed on `f32`/`i32` copy and
  ld/st rows plus checked-in subword clean diagnostics, not encoded `i64`/`f64`
  descriptor-view load/store assertions.

Backend repair remains deferred while discovery continues.
