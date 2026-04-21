# Round 12 Lane S: plain-MMAv5 runtime-selector-index reduction

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery-only. No backend/compiler fixes were attempted.
- Temporary reducer: `/tmp/tmem_plain_mma_runtime_index_round12_probe.py`
- Prior seed harness: `/tmp/tmem_mma_dynamic_round10_probe.py`

## Scope

This lane reduced the report-only `FZ-20260421-0011` candidate: plain MMAv5
runtime accumulator selection by `parent.index(ttgl.load(selector_ptr))`.

The goal was to preserve the runtime NaN-heavy miscompile signature and avoid
degrading into known `FZ-20260421-0001` late illegal `ttg.memdesc_index`
lowering. Axes probed:

- selector values `0` and `1`;
- `use_acc` false and true;
- `N in {32,64,128,256}` and `K in {32,64,128,256}` where the temporary
  reducer fit resource limits;
- lifted rank-3 linear parent layouts, legacy parents, and direct 2D linear
  parents passed to a rank-3 allocation;
- explicit reshape after `memdesc_index`;
- preinitializing the selected accumulator view before MMA;
- manual vs `get_default_for` shared layouts;
- manual vs smaller load layouts;
- dynamic slice selection and constexpr `memdesc_index` controls; and
- importing the runtime-matrix helpers vs using a standalone reducer.

## Validation Commands

Required rebuild:

```bash
make -j8
```

Result: no work to do.

Seed repros from the prior harness:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-round12-plain-mma-base \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_mma_dynamic_round10_probe.py --worker-case \
'{"case_id":"r12-plain-runtime-index-n64-k128-sel0-acc0","family":"plain","mode":"runtime_index","n":64,"k":128,"use_acc":false,"selector":0,"loop_count":2,"num_ctas":1,"fmt_a":"mxfp8","fmt_b":"mxfp8","chain_id":0,"parent_kind":"linear","slice_start":0}'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-round12-plain-mma-base-gpu1 \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_mma_dynamic_round10_probe.py --worker-case \
'{"case_id":"r12-plain-runtime-index-n64-k128-sel1-acc0","family":"plain","mode":"runtime_index","n":64,"k":128,"use_acc":false,"selector":1,"loop_count":2,"num_ctas":1,"fmt_a":"mxfp8","fmt_b":"mxfp8","chain_id":0,"parent_kind":"linear","slice_start":0}'
```

Standalone reducer compile check:

```bash
python -m py_compile /tmp/tmem_plain_mma_runtime_index_round12_probe.py
```

Default-pipeline sweep:

```bash
GPU=0 bash /tmp/tmem_plain_mma_runtime_index_round12_cases.sh \
  | tee /tmp/tmem_plain_mma_runtime_index_round12_results.jsonl
```

FPSAN-core sweep:

```bash
bash /tmp/tmem_plain_mma_runtime_index_round12_fpsan_core.sh \
  | tee /tmp/tmem_plain_mma_runtime_index_round12_fpsan_core.jsonl
```

Fresh-process repeat of the smallest stable row:

```bash
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-round12-plain-mma-repeat-$i \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_plain_mma_runtime_index_round12_probe.py --case \
  '{"case_id":"repeat_fpsan_runtime_n32_k128_sel1_acc0_'$i'","fpsan":true,"n":32,"k":128,"selector":1,"use_acc":false}'
done | tee /tmp/tmem_plain_mma_runtime_index_round12_repeat.jsonl
```

## Result Counts

- Default-pipeline sweep: `23` rows
  - `14` known `FZ-20260421-0001` compiler failures. The JSON classifier recorded
    these as `compiler_failure`, but stderr showed the exact
    `failed to legalize operation 'ttg.memdesc_index' that was explicitly marked
    illegal` signature.
  - `2` imported-helper rows also hit the same `FZ-20260421-0001` path when
    FPSAN was not enabled.
  - `6` pass controls.
  - `1` harness issue: intentionally exotic reversed-column parent produced the
    expected MMAv5 clean layout diagnostic, but the temporary classifier labeled
    it as a harness issue.
- FPSAN-core sweep: `8` rows, all `FZ-20260421-0011`.
- Fresh-process repeat: `3/3` rows reproduced `FZ-20260421-0011`.

## Findings

### `FZ-20260421-0011` is stable under FPSAN

The smallest stable runtime-miscompile form is:

- enable `triton.knobs.compilation.instrumentation_mode = "fpsan"`;
- allocate a rank-3 TMEM accumulator parent `[2, 128, N]`;
- select the accumulator with `parent.index(ttgl.load(selector_ptr))`;
- feed the selected view directly to `tcgen05_mma`; and
- read the selected view back to global memory.

Representative stable row:

- `repeat_fpsan_runtime_n32_k128_sel1_acc0_*`
- `3/3` fresh subprocesses reproduced exactly `4081 / 4096` mismatches;
- each run reported `16` NaNs;
- finite maximum absolute difference was approximately `3.303e38`.

Broader FPSAN coverage:

- `core_fpsan_runtime_n64_k128_sel0_acc0`: `8141 / 8192` mismatches, `31` NaNs;
- `core_fpsan_runtime_n64_k128_sel0_acc1`: `8160 / 8192` mismatches, `37` NaNs;
- `core_fpsan_runtime_n64_k128_sel1_acc0`: `8141 / 8192` mismatches, `31` NaNs;
- `core_fpsan_runtime_n64_k128_sel1_acc1`: `8160 / 8192` mismatches, `37` NaNs;
- `core_fpsan_runtime_n64_k32_sel1_acc0`: `8132 / 8192` mismatches, `35` NaNs;
- `core_fpsan_runtime_n64_k64_sel1_acc0`: `8131 / 8192` mismatches, `27` NaNs;
- `core_fpsan_runtime_n128_k128_sel1_acc0`: `16314 / 16384` mismatches, `65` NaNs.

The same failure reproduced in the prior imported-helper harness under FPSAN:

- selector `0`: `8157 / 8192` mismatches;
- selector `1`: `8148 / 8192` mismatches.

### FPSAN is the critical switch

Without FPSAN, the lifted-linear rank-3 runtime-index shape does not reach the
runtime miscompile. It fails in `ConvertTritonGPUToLLVM` as known
`FZ-20260421-0001`:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

This held across selector, `use_acc`, `N/K`, explicit reshape, preinit,
`get_default_for` shared layouts, and smaller load layouts.

Importing runtime-matrix helpers is not the preserving detail. Imported-helper
rows without FPSAN also hit `FZ-20260421-0001`; imported-helper rows with FPSAN
reproduced `FZ-20260421-0011`.

### Controls

These controls passed:

- constexpr index on lifted linear parent, selectors `0` and `1`;
- dynamic slice selection between `[M, 2N]` sibling views, selectors `0` and
  `1`;
- default-pipeline runtime index on a legacy parent;
- default-pipeline runtime index when a direct 2D linear parent layout is passed
  to the rank-3 allocation.

Under FPSAN, the legacy-parent and direct-2D-parent runtime-index rows also
miscompiled. That suggests the FPSAN pipeline is exposing or introducing a
runtime-index descriptor lowering/metadata problem broader than the lifted
rank-3 linear-layout form.

## Classification

- FPSAN runtime-index rows: `FZ-20260421-0011`.
- Non-FPSAN lifted-linear runtime-index rows: `FZ-20260421-0001`.
- Dynamic slice and constexpr-index controls: pass.
- Scaled-MMAv5 was not part of this lane; no new `FZ-20260421-0007` evidence.
- One exotic reversed-column parent row is a clean MMAv5 layout diagnostic, not
  a new bucket.

## Checked-In Test Readiness

A strict xfail is ready only if it explicitly pins FPSAN instrumentation. The
minimal robust sentinel should use the `N=32, K=128, selector=1, use_acc=False`
standalone shape because it is the smallest reproduced case and it repeated
`3/3` times with stable `4081 / 4096` mismatches.

Do not check in the same source without FPSAN as the `FZ-20260421-0011`
sentinel: it changes failure mode to known `FZ-20260421-0001` for lifted linear
parents. Also keep dynamic-slice and constexpr-index controls next to the xfail
when promoting this to a runtime test; both controls are green and distinguish
runtime outer `memdesc_index` from ordinary dynamic view selection.

