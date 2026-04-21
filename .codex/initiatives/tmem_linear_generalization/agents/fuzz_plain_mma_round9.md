# Round 9 Lane G: plain MMAv5 descriptor-flow fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler fixes attempted.
- Scope: non-scaled `tcgen05.mma` accumulator TMEM descriptor views, dynamic control flow, helper-returned views, `use_acc` accumulator initialization, 1CTA/2CTA runtime correctness, larger-CGA clean diagnostics, and narrow `N`/`K` boundaries.
- Temporary probe: `/tmp/tmem_plain_mma_round9_probe.py`
- Corrected aggregate results: `/tmp/tmem_plain_mma_round9_results.jsonl`
- Corrected full run log: `/tmp/tmem_plain_mma_round9_run_fixed.log`

## Commands

Required rebuild before tests:

```bash
make -j8
```

Result: `ninja: no work to do`.

Probe creation and syntax/inventory checks:

```bash
PYTHONPATH=.:./python:python/test/gluon python -m py_compile /tmp/tmem_plain_mma_round9_probe.py
PYTHONPATH=.:./python:python/test/gluon python /tmp/tmem_plain_mma_round9_probe.py --list
```

Inventory result: `153` subprocess-isolated cases.

Full corrected sweep:

```bash
CUDA_VISIBLE_DEVICES=3 \
TRITON_CACHE_DIR=/tmp/triton-cache-round9-lane-g \
PYTHONPATH=.:./python:python/test/gluon \
python /tmp/tmem_plain_mma_round9_probe.py 2>&1 | tee /tmp/tmem_plain_mma_round9_run_fixed.log
```

Corrected result:

```text
SUMMARY {"clean CTA-per-CGA mismatch diagnostic": 3, "pass": 150}
RESULTS /tmp/tmem_plain_mma_round9_results.jsonl
```

An initial run reported four `compiler crash/assert` rows for `K=16`, but
those were harness false positives. The kernels had compiled and matched the
reference; the probe then called the existing runtime-matrix opcode-count helper
that asserts `k % 32 == 0`. After changing the temporary probe to skip that
helper assertion for narrow-`K` exploratory rows, the four exact cases reran as
`pass`, and the full corrected sweep produced the summary above.

Exact narrow rerun:

```bash
CUDA_VISIBLE_DEVICES=3 \
TRITON_CACHE_DIR=/tmp/triton-cache-round9-lane-g \
PYTHONPATH=.:./python:python/test/gluon \
python /tmp/tmem_plain_mma_round9_probe.py \
  --case 1cta-narrow-subslice-n32-k16 \
  --case 1cta-narrow-dyn-if-n32-k16 \
  --case 1cta-narrow-subslice-n16-k16 \
  --case 1cta-narrow-dyn-if-n16-k16 \
  2>&1 | tee /tmp/tmem_plain_mma_round9_narrow_rerun.log
```

Result:

```text
SUMMARY {"pass": 4}
```

## Coverage

The 153 corrected rows split as:

- `dynamic_if`: 27 rows.
- `helper`: 24 rows.
- `indexed`: 12 rows.
- `subslice`: 27 rows.
- `twocta_dynamic_if`: 24 rows.
- `twocta_indexed`: 12 rows.
- `twocta_subslice`: 24 rows.
- `larger_cga_clean`: 3 rows.

Runtime-positive rows covered:

- 1CTA accumulator `slice(0,N)` and `slice(N,N)` views.
- 1CTA dynamic `if` selection between low/high accumulator subslices.
- 1CTA helper-returned accumulator subslice descriptors selected by a runtime
  scalar.
- 1CTA indexed parent views with both depth-2 linear parents and unit-depth
  linear parents.
- 2CTA accumulator subslice and indexed parent views.
- 2CTA dynamic `if` selection between accumulator subslices.
- `use_acc=False` and `use_acc=True` for direct, dynamic, helper, indexed, and
  2CTA rows.
- `N in {32,64,128}` and `K in {32,128}` for the main descriptor-flow matrix.
- Narrow exploratory rows for `N=16`, `K=16`, and `N=32,K=16`.
- Larger-CGA clean diagnostics for a two-CTA TMEM layout in `{4,8,16}` CTA
  contexts.

All runtime-positive rows emitted matching PTX/LLIR `tcgen05.mma` opcodes:

- 90 rows emitted `tcgen05.mma.cta_group::1.kind::f16`.
- 60 rows emitted `tcgen05.mma.cta_group::2.kind::f16`.

Representative corrected rows:

- `1cta-narrow-subslice-n16-k32`: pass, 2 one-CTA f16 MMA ops.
- `1cta-narrow-subslice-n32-k16`: pass, 1 one-CTA f16 MMA op.
- `2cta-dyn-if-n128-k128-sel1-acc1`: pass, 8 two-CTA f16 MMA ops and correct
  `use_acc` initialization from the high accumulator view.
- `larger-cga-twocta-layout-in-16cta`: clean CTA-per-CGA mismatch diagnostic.

## Classification

No new backend/compiler failure was found in this lane.

The non-scaled plain-MMAv5 accumulator descriptor-flow surface did not
reproduce the scaled-MMAv5 owner issue tracked as `FZ-20260421-0007`. Dynamic
runtime selection between low/high accumulator subslices passed for plain
`tcgen05.mma` in both 1CTA and 2CTA forms, including `use_acc=True`.

The lane also did not produce a new dynamic `memdesc_index` illegal-lowering
case (`FZ-20260421-0001`). The dynamic rows used slice/subslice descriptor
views, and the indexed parent rows stayed static. This intentionally keeps
runtime dynamic-index failures separated from the plain-MMA descriptor-flow
owner surface.

The larger-CGA rows remained clean negatives. A two-CTA accumulator layout in
4/8/16 CTA contexts reported the expected CTA-per-CGA mismatch diagnostic
without `PassManager::run failed`, assertion text, or process abort.

## Promotion Recommendation

Do not promote a new `FZ-*` from Round 9 Lane G.

Useful follow-up coverage, if another plain-MMA lane is assigned:

- add a generator-backed variant that composes indexed accumulator parents with
  dynamic helper-returned slice views while avoiding runtime `memdesc_index`;
- extend the plain-MMA dynamic-selector probe to bf16/tf32/f8 kinds after
  confirming the operand and shared-memory layout setup stays comparable; and
- keep the narrow `K=16` rows as positive exploratory evidence only unless a
  backend diagnostic or miscompile appears in a future, independently checked
  harness.
