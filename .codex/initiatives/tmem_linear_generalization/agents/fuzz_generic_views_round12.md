# Round 12 Lane W: Generic Pass TMEM Descriptor-View Fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Repo edit scope: report only.
- Checked-in source: `python/test/gluon/test_tmem_structural_fuzzer.py`
- Temporary probe: `/tmp/tmem_generic_views_round12_probe.py`

## Scope

This lane adversarially fuzzed generic pass interactions with TMEM descriptor
views after the current bucket set. The sweep focused on:

- helper-returned descriptor views;
- dynamic `if` values carrying memdescs;
- loop-carried memdesc values;
- tuple-like tensor plus memdesc captures;
- layout-conversion pressure around descriptor-view loads/stores;
- runtime `memdesc_index`; and
- direct/static/constexpr green controls.

The goal was classification, not repair. I separated known `FZ-20260421-0001`,
`FZ-20260421-0002`, `FZ-20260421-0003`, historical `R5-C`, and any possible
new root.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Checked-in generic-pass collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'
```

Result: `11/33` tests collected.

Checked-in generic-pass split run, one fresh pytest process per GPU with stable
per-GPU caches:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'
```

Result: `11 xfailed` total. Shards selected `3`, `3`, `3`, and `2` rows.

Temporary probe syntax and collection:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_generic_views_round12_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_generic_views_round12_probe.py
```

Result: `9` tests collected.

Temporary probe split run, also one fresh pytest process per GPU with stable
per-GPU caches:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  --splits 4 --group 1 /tmp/tmem_generic_views_round12_probe.py

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  --splits 4 --group 2 /tmp/tmem_generic_views_round12_probe.py

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  --splits 4 --group 3 /tmp/tmem_generic_views_round12_probe.py

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  --splits 4 --group 4 /tmp/tmem_generic_views_round12_probe.py
```

Effective result after excluding and rerunning one temporary harness-call bug:
`7 passed, 2 failed`. The corrected direct-control row was rerun as:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short \
  '/tmp/tmem_generic_views_round12_probe.py::test_generic_views_round12[r12-static-direct-ldst-identity-control]'
```

Result: `1 passed`.

Representative checked-in rows were rerun with `--runxfail` in fresh pytest
processes to capture concrete diagnostics:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --runxfail \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_loop_carried[generic-pass-loop-carried-memdesc-view-chain0]'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --runxfail \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-if-chain0-true]'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --runxfail \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_dynamic_index_load_only[generic-pass-dynamic-index-load-only-128x32]'
```

## Results Summary

No new independent `FZ-*` bucket is warranted from this lane.

The checked-in generic-pass inventory remains stable as strict xfail coverage:
`11 xfailed`. The temporary probe added executable green controls and two
known-bad contrasts:

- `7` temporary rows passed;
- `1` runtime-index row reproduced `FZ-20260421-0001`;
- `1` chain0 dynamic-if row reproduced the `FZ-20260421-0002` mismatch
  pattern; and
- no row produced a new compiler diagnostic, clean unsupported false negative,
  or distinct runtime mismatch signature.

## Temporary Probe Case Table

| Case | Outcome | Classification |
| --- | --- | --- |
| `r12-static-direct-ldst-identity-control` | pass | Direct/static descriptor-view control. |
| `r12-static-direct-layout-pressure-chain1-control` | pass | Static `parent.index(1)` plus chain1 layout pressure. |
| `r12-static-direct-layout-pressure-chain2-control` | pass | Static `parent.index(1)` plus chain2 layout pressure. |
| `r12-dynamic-if-chain1-selector1-control` | pass | Dynamic `if` over memdesc values is green for chain1. |
| `r12-dynamic-if-chain2-selector0-control` | pass | Dynamic `if` over memdesc values is green for chain2. |
| `r12-mixed-captures-chain1-selector0-control` | pass | Tensor plus memdesc capture is green for chain1. |
| `r12-tuple-mixed-captures-chain1-selector1-control` | pass | Tuple-like `(memdesc, tensor)` helper return is green for chain1. |
| `r12-runtime-index-chain2-loadstore` | compiler failure | `FZ-20260421-0001`: runtime `ttg.memdesc_index` reaches LLVM conversion. |
| `r12-dynamic-if-chain0-selector1-known` | miscompile | `FZ-20260421-0002`: `8063 / 8192` mismatches. |

## Representative Diagnostics

### `FZ-20260421-0001`

The runtime-index rows still fail in `ConvertTritonGPUToLLVM` with illegal
`ttg.memdesc_index`. The load-only checked-in representative reported:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
Pipeline failed while executing [`ConvertTritonGPUToLLVM` on 'builtin.module' operation]
RuntimeError: PassManager::run failed
```

The temporary `r12-runtime-index-chain2-loadstore` row reproduced the same
owner surface with a chain2 descriptor-view use after the runtime index.

### `FZ-20260421-0002`

The chain0 generic-pass runtime rows still produce large deterministic
wrong-value sets:

- checked-in `generic-pass-dynamic-if-chain0-true`: `8064 / 8192`
  mismatches;
- temporary `r12-dynamic-if-chain0-selector1-known`: `8063 / 8192`
  mismatches; and
- checked-in `generic-pass-loop-carried-memdesc-view-chain0`, when forced with
  `--runxfail` on this head, now reaches runtime and reports `8064 / 8192`
  mismatches.

That last point is a bucket-boundary refinement: the checked-in loop-carried
sentinel still xfails, but the current observed failure is not the historical
R5-C auto-layout crash for this row. It now behaves like the same chain0
descriptor-view/control-flow wrong-value family as `FZ-20260421-0002`.

### `R5-C`

No fresh R5-C compiler-crash row was newly exposed by this lane. The historical
R5-C reports remain valid repair-validation inventory for nested helper
returns, loop-carried memdesc values, and tuple-like memdesc/tensor results,
but the checked-in loop-carried row is currently classified more accurately as
an `FZ-20260421-0002` runtime miscompile on this head.

### `FZ-20260421-0003`

This lane did not find a new `ld/st` packet-order root. The direct/static
identity `ld/st` control passed, and the temporary generic-pass controls were
chosen to avoid reclassifying direct chain0 descriptor-view packet-order rows
that are already covered by `FZ-20260421-0003` reports.

## Classification

- `FZ-20260421-0001`: still owns runtime `memdesc_index` values that survive
  to LLVM lowering as illegal operations, including chain2 load/store use.
- `FZ-20260421-0002`: still owns chain0 generic pass/control-flow/layout
  pressure wrong results. On current head, the checked-in loop-carried row also
  presents as this runtime mismatch family.
- `FZ-20260421-0003`: not expanded by this lane; direct/static control passed.
- `R5-C`: not newly reproduced in this lane; keep the historical reports as
  repair-validation inventory, but do not treat the current checked-in
  loop-carried row's observed failure as an auto-layout crash without rerunning
  it.
- New roots: none.

## Promotion Candidates

Recommended promotions after this discovery lane:

1. Keep the checked-in generic-pass strict xfail set, but update the
   `generic-pass-loop-carried-memdesc-view-chain0` expected reason during the
   eventual repair/test-refresh phase if repeated reruns continue to show the
   `8064 / 8192` runtime mismatch instead of the historical R5-C diagnostic.
2. Promote the green controls only if generic-pass repair work needs compact
   regression coverage:
   `r12-dynamic-if-chain1-selector1-control`,
   `r12-dynamic-if-chain2-selector0-control`, and
   `r12-tuple-mixed-captures-chain1-selector1-control`.
3. Do not promote `r12-runtime-index-chain2-loadstore`; checked-in
   `generic-pass-dynamic-index-*` and `generic-pass-dynamic-index-load-only`
   already cover `FZ-20260421-0001`.
4. Do not assign a new `FZ-*` id.
