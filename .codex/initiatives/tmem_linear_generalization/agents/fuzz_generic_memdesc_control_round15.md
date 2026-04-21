# Round 15 Lane AK: Generic Memdesc Control-Flow Fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler fixes were attempted.
- Repo edit scope: this report only.
- Temporary probes/logs:
  - `/tmp/tmem_generic_memdesc_control_round15_probe.py`
  - `/tmp/tmem_generic_memdesc_control_round15_results.jsonl`
  - `/tmp/tmem_r15_scaled_probe.py`
  - `/tmp/tmem_r15_scaled_results.jsonl`
  - `/tmp/tmem_r15_raw_branch.txt`
  - `/tmp/tmem_r15_raw_copy_chain0.txt`
  - `/tmp/tmem_r15_raw_copy_chain1.txt`
  - `/tmp/tmem_r15_raw_mma_chain0.txt`
  - `/tmp/tmem_r15_raw_mma_chain1.txt`

## Scope

This lane targeted generic memdesc/control-flow interactions that are adjacent
to, but not identical to, the checked-in structural fuzzer rows:

- helper-returned descriptor views selected through a runtime branch;
- loop-carried helper-returned descriptor views;
- runtime `parent.index(load(selector))` followed by helper-returned view chains;
- generic branch-selected views feeding TMEM `ld/st`, `tcgen05.copy`, plain
  MMAv5, and scaled-MMAv5; and
- selector and chain variants that distinguish existing `FZ-0001`,
  `FZ-0002`, `FZ-0003`, `FZ-0011`, and `FZ-0013` boundaries.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Temporary probe syntax check:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_generic_memdesc_control_round15_probe.py
```

Main temporary probe. The driver ran one fresh subprocess per row, rotated
`CUDA_VISIBLE_DEVICES={0,1,2,3}`, and used stable
`TRITON_CACHE_DIR=/tmp/triton-cache-gpu{0,1,2,3}`:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_generic_memdesc_control_round15_probe.py \
  --results /tmp/tmem_generic_memdesc_control_round15_results.jsonl
```

Raw diagnostic reruns for rows where the JSON wrapper only preserved the Python
traceback:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_generic_memdesc_control_round15_probe.py \
  --worker-case '{"case_id":"r15-ldst_branch_helper-chain1-sel1","kind":"ldst_branch_helper","chain":1,"selector":1,"seed":15003,"M":128,"N":64,"variant":"32x32b"}' \
  > /tmp/tmem_r15_raw_branch.txt 2>&1

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_generic_memdesc_control_round15_probe.py \
  --worker-case '{"case_id":"r15-copy-branch-helper-chain1","kind":"copy_branch_helper","chain":1,"selector":1,"seed":15020}' \
  > /tmp/tmem_r15_raw_copy_chain1.txt 2>&1

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_generic_memdesc_control_round15_probe.py \
  --worker-case '{"case_id":"r15-mma-branch-acc-chain1","kind":"mma_branch_acc","chain":1,"selector":1,"seed":15022}' \
  > /tmp/tmem_r15_raw_mma_chain1.txt 2>&1

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_generic_memdesc_control_round15_probe.py \
  --worker-case '{"case_id":"r15-copy-branch-helper-chain0","kind":"copy_branch_helper","chain":0,"selector":1,"seed":15019}' \
  > /tmp/tmem_r15_raw_copy_chain0.txt 2>&1

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_generic_memdesc_control_round15_probe.py \
  --worker-case '{"case_id":"r15-mma-branch-acc-chain0","kind":"mma_branch_acc","chain":0,"selector":1,"seed":15021}' \
  > /tmp/tmem_r15_raw_mma_chain0.txt 2>&1
```

Corrected scaled-MMAv5 branch-selected scale-descriptor probe:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_r15_scaled_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_r15_scaled_probe.py
```

Checked-in control selector:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_structural_fuzzer.py -k 'generic_pass'
```

Result: `11 xfailed` across the four shards (`3`, `3`, `3`, `2`).

## Results Summary

No new independent `FZ-*` bucket is warranted from this lane.

After raw rerun classification, the temporary probe produced:

- `4` passes: loop-carried helper views for chain1/chain2, both selectors;
- `16` `FZ-20260421-0001` rows: branch-selected helper returns, runtime
  `memdesc_index`, copy chain1, and plain-MMAv5 chain1;
- `2` `FZ-20260421-0002` rows: loop-carried chain0 helper views;
- `2` clean unsupported diagnostics: chain0 copy and chain0 plain-MMAv5
  consumer controls;
- `4` `FZ-20260421-0013` rows from the corrected scaled-MMAv5 scale descriptor
  branch-selection probe; and
- `2` discarded probe-invalid scaled rows from the first scaled probe revision,
  caused by referencing a Python helper inside JIT source. Those rows are not
  used for bucket decisions.

## Case Table

| Case family | Rows | Outcome | Classification |
| --- | ---: | --- | --- |
| `ldst_branch_helper`, chain0/1/2, selectors 0/1 | 6 | Compiler failure in `ConvertTritonGPUToLLVM` | `FZ-20260421-0001` |
| `ldst_branch_helper`, chain0, `N=32`, `16x64b` | 1 | Compiler failure in `ConvertTritonGPUToLLVM` | `FZ-20260421-0001` |
| `ldst_runtime_index`, chain0/1/2, selectors 0/1 | 6 | Compiler failure in `ConvertTritonGPUToLLVM` | `FZ-20260421-0001` |
| `ldst_loop_helper`, chain0, selectors 0/1 | 2 | Runtime wrong results, `8064/8192` mismatches | `FZ-20260421-0002` |
| `ldst_loop_helper`, chain1/2, selectors 0/1 | 4 | Pass | Green boundary |
| `copy_branch_helper`, chain0 | 1 | Clean unsupported shared-memory descriptor-plan diagnostic | True boundary / not a new bug |
| `copy_branch_helper`, chain1 | 1 | Compiler failure in `ConvertTritonGPUToLLVM` | `FZ-20260421-0001` |
| `mma_branch_acc`, chain0 | 1 | Clean unsupported MMAv5 tile-order diagnostic | True boundary / not a new bug |
| `mma_branch_acc`, chain1 | 1 | Compiler failure in `ConvertTritonGPUToLLVM` | `FZ-20260421-0001` |
| corrected `scaled_branch_scale`, chain0/1, selectors 0/1 | 4 | Runtime wrong results, about `16381-16382/16384` mismatches | `FZ-20260421-0013` |

## Representative Diagnostics

### `FZ-20260421-0001`

The branch-selected helper-returned view rows failed with the same illegal
runtime memdesc-index lowering as earlier generic-runtime-index lanes:

```text
error: failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal
Pipeline failed while executing [`ConvertTritonGPUToLLVM` on 'builtin.module' operation]
```

The raw IR for `r15-ldst_branch_helper-chain1-sel1` shows both static branch
indices and the merged runtime index surviving:

```text
%11 = ttg.memdesc_index %result[%c0_i32]
%13 = ttg.memdesc_index %result[%c1_i32]
%18 = ttg.memdesc_index %result[%17]
```

The same failure shape was reproduced for `tcgen05.copy` and plain MMAv5
consumers when the selected descriptor-view chain was otherwise MMAv5/copy
compatible (`chain1`). This broadens `FZ-0001` to branch-selected helper returns
feeding those consumers; it is not a new root cause.

### `FZ-20260421-0002`

Loop-carried helper-returned `ld/st` views keep the previously sharp chain0
wrong-result boundary:

- `r15-ldst_loop_helper-chain0-sel0`: `8064 / 8192` mismatches.
- `r15-ldst_loop_helper-chain0-sel1`: `8064 / 8192` mismatches.
- chain1 and chain2 variants with the same loop-carried structure passed.

This is the same generic pass/control-flow/layout-pressure miscompile bucket,
not a fresh `ld/st` descriptor-chain packet-order bucket.

### Clean Unsupported Rows

Two consumer controls failed cleanly before late lowering:

- `r15-copy-branch-helper-chain0`: `tcgen05.copy.128x256b` shared-memory
  descriptor plan could not be synthesized. The diagnostic explicitly says this
  is reported cleanly instead of falling through to LLVM lowering.
- `r15-mma-branch-acc-chain0`: MMAv5 rejected a noncanonical accumulator tile
  order and explained the first offending in-tile basis.

Both are treated as true unsupported boundaries for this lane.

### `FZ-20260421-0013`

The corrected scaled-MMAv5 probe selected between two scale descriptors through
runtime control flow and then fed the selected descriptor to `tcgen05_mma_scaled`.
All four rows compiled and executed but produced large wrong-result sets:

- `r15-scaled-branch-scale-chain0-sel0`: `16382 / 16384` mismatches.
- `r15-scaled-branch-scale-chain0-sel1`: `16381 / 16384` mismatches.
- `r15-scaled-branch-scale-chain1-sel0`: `16381 / 16384` mismatches.
- `r15-scaled-branch-scale-chain1-sel1`: `16382 / 16384` mismatches.

This overlaps the existing scaled-MMAv5 scale descriptor-view wrong-result
bucket (`FZ-20260421-0013`), now with explicit runtime branch selection between
scale memdesc values.

## New FZ Decision

No new bucket is assigned.

This lane broadens:

- `FZ-20260421-0001`: runtime branch/helper-selected memdesc values can reach
  late LLVM conversion as illegal `ttg.memdesc_index`, including copy and
  plain-MMAv5 consumers.
- `FZ-20260421-0002`: loop-carried helper-returned chain0 views remain wrong at
  runtime, while equivalent chain1/chain2 loop rows pass.
- `FZ-20260421-0013`: scaled-MMAv5 scale descriptor views miscompile even when
  the scale descriptor is selected by runtime branch control flow.

No evidence in this lane points to a new `FZ-0011` FPSAN-specific plain-MMAv5
parent-index bucket or a new `FZ-0003` ld/st packet-order bucket.
