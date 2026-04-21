# Round 25 Lane BJ: Dynamic Descriptor Structural Generator Prototype

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only; no backend/compiler repairs attempted.
- Repo edit scope: this report only.
- Temporary prototype: `/tmp/tmem_structural_generator_dynamic_round25.py`
- Logs:
  - `/tmp/tmem_structural_generator_dynamic_round25.log`
  - `/tmp/tmem_structural_generator_dynamic_round25_rerun.log`
- isolated ambiguous-seed logs under `/tmp/tmem_round25_<seed-id>.log`
- reduction detail log: `/tmp/tmem_round25_ldred_chain0_detail.log`

## Scope

This lane promoted the Round 24 Lane BI generator idea into a second
prototype focused on dynamic descriptor SSA axes.  The generated seeds cross:

- descriptor SSA axis:
  dynamic `memdesc_index`, dynamic branch-selected descriptors,
  helper-returned descriptors, mixed memdesc/tensor captures, layout-pressure
  helper use, and loop-carried descriptors;
- view chain:
  reshape/transpose/reshape chain 0, reshape/reshape chain 1, and direct
  slice-only chain 2;
- consumer:
  `ttng.tmem_load`/`ttng.tmem_store`, `ttng.tmem_copy` with readback, and
  f32 `ttng.tmem_load` reductions where feasible.

No backend or compiler code was changed.

## Commands

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Prototype syntax check:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_structural_generator_dynamic_round25.py
```

Initial run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_structural_generator_dynamic_round25.py \
  2>&1 | tee /tmp/tmem_structural_generator_dynamic_round25.log
```

Rerun after adding direct slice-only copy and `ld.red` controls:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_structural_generator_dynamic_round25.py \
  2>&1 | tee /tmp/tmem_structural_generator_dynamic_round25_rerun.log
```

Ambiguous compiler-failure rows were isolated because Python only received
`PassManager::run failed` while the useful diagnostic was printed on stderr:

```bash
for seed in \
  BJ-COPY-BRANCH-CHAIN1-SEL0 \
  BJ-COPY-BRANCH-CHAIN2-SEL1 \
  BJ-COPY-HELPER-CHAIN2-SEL1 \
  BJ-LDRED-HELPER-CHAIN2-SEL1; do
  CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
    PYTHONPATH=.:./python:./python/test/gluon python3 - <<PY \
    2>&1 | tee /tmp/tmem_round25_${seed}.log
import importlib.util, json
spec = importlib.util.spec_from_file_location(
    "bj", "/tmp/tmem_structural_generator_dynamic_round25.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
seed = next(s for s in mod.SEEDS if s.seed_id == "$seed")
print(json.dumps(mod.run_seed(seed), sort_keys=True))
PY
done
```

Reduction detail probe for the two `ld.red` chain 0 wrong-result rows:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon python3 - <<'PY' \
  2>&1 | tee /tmp/tmem_round25_ldred_chain0_detail.log
# Imports the /tmp prototype and runs only BJ-LDRED-{BRANCH,LOOP}-CHAIN0,
# checking full output and reduced vector separately.
PY
```

## Seed Results

The rerun executed 22 generated seeds.

| Seed ID | Consumer | SSA axis | Chain | Result | Classification |
| --- | --- | --- | --- | --- | --- |
| `BJ-LDST-DYNIDX-CHAIN0-SEL1` | ld/st | dynamic index | 0 | compiler failure | Existing `FZ-20260421-0001`: runtime `ttg.memdesc_index` remains illegal at LLVM conversion. |
| `BJ-LDST-DYNIDX-CHAIN1-SEL0` | ld/st | dynamic index | 1 | compiler failure | Existing `FZ-20260421-0001`. |
| `BJ-LDST-DYNIDX-LOADONLY-SEL1` | ld only | dynamic index | direct | compiler failure | Existing `FZ-20260421-0001`; no store/copy consumer is required. |
| `BJ-LDST-BRANCH-CHAIN0-SEL1` | ld/st | branch | 0 | wrong result | Existing `FZ-20260421-0002`; `8063/8192` mismatches. |
| `BJ-LDST-BRANCH-CHAIN1-SEL0` | ld/st | branch | 1 | pass | Green contrast; emits one `tcgen05.ld` and two `tcgen05.st` ops. |
| `BJ-LDST-INLINE-BRANCH-CHAIN0-SEL0` | ld/st | inline branch | 0 | wrong result | Existing `FZ-20260421-0002`; `8064/8192` mismatches. |
| `BJ-LDST-MIXED-CAPTURES-CHAIN0-SEL1` | ld/st | mixed captures | 0 | wrong result | Existing `FZ-20260421-0002`; `8064/8192` mismatches. |
| `BJ-LDST-LAYOUT-PRESSURE-CHAIN0` | ld/st | layout pressure | 0 | wrong result | Existing `FZ-20260421-0002`; `8064/8192` mismatches. |
| `BJ-LDST-LOOP-CARRIED-CHAIN0-SEL1` | ld/st | loop-carried | 0 | wrong result | Existing `FZ-20260421-0002`; this HEAD compiles and miscompiles instead of stopping at the older auto-layout boundary. |
| `BJ-COPY-BRANCH-CHAIN0-SEL1` | copy | branch | 0 | clean diagnostic | Clean copy planner unsupported row-permuted destination boundary. |
| `BJ-COPY-HELPER-CHAIN0-SEL1` | copy | helper | 0 | clean diagnostic | Same clean copy planner boundary. |
| `BJ-COPY-LOOP-CARRIED-CHAIN0-SEL1` | copy | loop-carried | 0 | clean diagnostic | Same clean copy planner boundary. |
| `BJ-COPY-BRANCH-CHAIN1-SEL0` | copy | branch | 1 | compiler failure | Existing `FZ-20260421-0001`; isolated log shows illegal dynamic `ttg.memdesc_index` at LLVM conversion. |
| `BJ-COPY-BRANCH-CHAIN2-SEL1` | copy | branch | 2 | compiler failure | Existing `FZ-20260421-0001`; direct slice-only branch-yielded descriptor still lowers to illegal dynamic `memdesc_index`. |
| `BJ-COPY-HELPER-CHAIN2-SEL1` | copy | helper | 2 | compiler failure | Existing `FZ-20260421-0001`; helper-returned descriptor becomes dynamic `memdesc_index`. |
| `BJ-COPY-LOOP-CARRIED-CHAIN2-SEL1` | copy | loop-carried | 2 | pass | Green contrast; emits eight `tcgen05.cp.cta_group::1.128x256b` ops and readback `tcgen05.ld`. |
| `BJ-LDRED-BRANCH-CHAIN0-SEL1` | ld.red | branch | 0 | wrong reduced rows | Likely existing `FZ-20260421-0002` extended to reduction consumer; `126/128` reduced rows mismatched. |
| `BJ-LDRED-HELPER-CHAIN0-SEL1` | ld.red | helper | 0 | compiler failure | Existing `FZ-20260421-0001`; helper path leaves illegal dynamic `memdesc_index`. |
| `BJ-LDRED-LOOP-CARRIED-CHAIN0-SEL1` | ld.red | loop-carried | 0 | wrong reduced rows | Likely existing `FZ-20260421-0002` extended to reduction consumer; `126/128` reduced rows mismatched. |
| `BJ-LDRED-BRANCH-CHAIN2-SEL1` | ld.red | branch | 2 | pass | Green contrast; emits `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32`. |
| `BJ-LDRED-HELPER-CHAIN2-SEL1` | ld.red | helper | 2 | compiler failure | Existing `FZ-20260421-0001`; isolated log shows illegal dynamic `ttg.memdesc_index`. |
| `BJ-LDRED-LOOP-CARRIED-CHAIN2-SEL1` | ld.red | loop-carried | 2 | pass | Green contrast; emits `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32`. |

Rerun summary before stderr-aware post-classification:

```text
{"FZ-20260421-0002": 5, "error-unclassified": 11, "green": 4, "miscompile-new-or-related": 2}
```

After inspecting stderr for the unclassified compiler failures:

```text
FZ-20260421-0001: 8
FZ-20260421-0002: 5 direct ld/st rows plus 2 likely ld.red reduction-consumer rows
clean copy unsupported boundary: 3
green: 4
```

## Findings

No new independent `FZ-*` is proven by this lane, but the generator sharpens
two existing buckets and produces promotion-ready seeds.

`FZ-20260421-0001` now clearly covers multiple consumers beyond direct
`tmem_load`/`tmem_store`:

- dynamic index into a parent TMEM descriptor feeding ld/st;
- dynamic index feeding load-only;
- branch/helper-selected descriptors feeding `ttng.tmem_copy`;
- helper-selected descriptors feeding f32 `ld.red`.

The repeated backend diagnostic is:

```text
failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
```

The compiler often prints the useful MLIR reproducer on stderr while raising
only `PassManager::run failed` in Python, so a checked-in generator should use
subprocess isolation for expected compiler failures.

`FZ-20260421-0002` is sharpened as a dynamic descriptor-view chain 0
miscompile that is not restricted to helper boundaries:

- branch-selected chain 0 miscompiles;
- inline branch chain 0 miscompiles;
- mixed memdesc/tensor captures miscompile;
- layout-conversion pressure miscompiles;
- loop-carried chain 0 now compiles and miscompiles on this HEAD.

The `ld.red` chain 0 branch and loop-carried seeds also produced wrong results,
but the detail probe shows the full matrix readback is already wrong:

```text
branch       out_mismatch 8064  red_mismatch 126  emits tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32
loop_carried out_mismatch 8064  red_mismatch 126  emits tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32
```

So these rows are not a reduction-only bug. They are best classified as
`FZ-20260421-0002` extended to an `ld.red` consumer: the dynamic descriptor
chain 0 view is wrong before, or as part of, the hardware reduction.

The copy rows also define an important clean-boundary contrast:

- branch/helper/loop-carried chain 0 copy rows hit the existing direct copy
  planner diagnostic for row-permuted destinations;
- branch/helper chain 2 copy rows avoid that row-permuted boundary and expose
  `FZ-0001`;
- loop-carried chain 2 copy passes and emits the expected copy/readback ops.

The direct slice-only `ld.red` contrasts are useful:

- branch and loop-carried chain 2 pass and emit hardware `.ld.red`;
- helper chain 2 fails as `FZ-0001`.

That split suggests the branch and loop-carried direct-slice descriptors can be
resolved before LLVM conversion in some forms, while helper-returned direct
slice descriptors still leave a generic dynamic `memdesc_index`.

## Promotion Plan

Promote this prototype into checked-in Python runtime coverage after the active
discovery burst:

- Add a deterministic seed table with fields
  `(seed_id, consumer, ssa_axis, chain_id, selector, expected_class)`.
- Run green runtime seeds in-process:
  `BJ-LDST-BRANCH-CHAIN1-SEL0`,
  `BJ-COPY-LOOP-CARRIED-CHAIN2-SEL1`,
  `BJ-LDRED-BRANCH-CHAIN2-SEL1`, and
  `BJ-LDRED-LOOP-CARRIED-CHAIN2-SEL1`.
- Run expected compiler-failure seeds in subprocesses so stderr can be matched
  against the real diagnostic instead of the Python wrapper exception.
- Keep copy chain 0 rows as clean-boundary tests unless the copy planner grows
  row-projection support.
- Promote `ld.red` chain 0 branch/loop-carried rows as `FZ-0002` extensions,
  not as a new reduction-specific bucket, because full-output readback is
  wrong in the same rows.
- Add a generator-owned summary assertion that every expected-failure row
  reports one of the known bucket diagnostics, so a future backend fix causes a
  strict unexpected pass rather than silently changing classification.

No backend/compiler code was changed.
