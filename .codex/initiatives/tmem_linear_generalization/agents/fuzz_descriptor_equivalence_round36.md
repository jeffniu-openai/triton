# Round 36 Descriptor-View Algebra Equivalence Oracle

Date: 2026-04-21

Scope: discovery/cataloging only. No backend or compiler source was modified.

## Required Build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Oracle Design

Temporary probe:

- driver: `/root/tmp/tmem_descriptor_equivalence_round36_probe.py`
- generated worker: `/tmp/tmem_descriptor_equivalence_round36/worker.py`
- summary: `/tmp/tmem_descriptor_equivalence_round36/summary.json`
- per-case logs: `/tmp/tmem_descriptor_equivalence_round36/*.log`

The oracle compares semantically equivalent descriptor constructions by writing
through the candidate descriptor and reading/reducing through the canonical
descriptor:

- `direct`: direct `[M,N]` TMEM allocation;
- `parent_index`: `parent.index(1)` from a `[2,M,N]` parent;
- `chain0`: `reshape(M/2,2,N) -> permute([1,0,2]) -> reshape(M,N)`;
- `chain1`: column unit-pair double permutation identity;
- `rank4_unit`: rank-4 unit-dimension reshape/permute/reshape identity;
- `rank5_unit`: rank-5 unit-dimension reshape/permute/reshape identity;
- `permute_identity`: double transpose identity;
- `half_row`: `base.slice(0, M/2, dim=0)`;
- `half_col`: `base.slice(0, N/2, dim=1)`.

Rows used `M=128,N=64`, layouts `{identity,row_reverse,col_reverse,row+col_reverse}`,
and stable per-GPU cache directories `/tmp/triton-cache-gpu{0..3}`. Runtime
checks compare full output tensors against the source slice. `ld.red` rows also
compare reduced outputs against PyTorch `min`/`max` and require a hardware
`.ld.red` opcode when the row is intended to test hardware reduction selection.

## Commands

Main subprocess-isolated four-GPU oracle:

```bash
PYTHONPATH=.:./python:./python/test/gluon python /root/tmp/tmem_descriptor_equivalence_round36_probe.py
```

The driver dispatches each case with:

```bash
CUDA_VISIBLE_DEVICES=<case_index % 4> \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu<case_index % 4> \
PYTHONPATH=.:./python:./python/test/gluon \
TMEM_EQ_CASE='<case-json>' \
python /tmp/tmem_descriptor_equivalence_round36/worker.py
```

Feasible checked-in `tcgen05.copy` descriptor-view guardrail:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales and (indexed_view or subslice_view or slice_index_view) and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales and (indexed_view or subslice_view or slice_index_view) and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales and (indexed_view or subslice_view or slice_index_view) and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales and (indexed_view or subslice_view or slice_index_view) and not resource'
```

Result: `87 passed` split as `22/22/22/21`.

## Results

Main oracle aggregate:

| Classification | Count |
| --- | ---: |
| pass | 84 |
| wrong result | 12 |
| existing `FZ-20260421-0022` | 4 |
| runtime-correct opcode fallback / existing `FZ-20260421-0004` | 4 |
| existing `FZ-20260421-0020` | 2 |
| clean scalar `.x1` diagnostic | 2 |

The raw driver summary labels the last three groups as
`compiler_crash`/`other_failure`; manual log inspection gives the corrected
classification above.

Passing surfaces:

- all direct, `parent.index`, `chain1`, rank-4 unit, rank-5 unit,
  double-permute identity, half-row `ld/st`, and half-column `ld/st` rows;
- all direct, `parent.index`, `chain1`, rank-4 unit, rank-5 unit, and
  double-permute identity `ld.red` rows;
- half-column `ld.red` for identity and row-reversed layouts.

Wrong-result rows:

- `ldst-chain0-{identity_identity,reverse_identity,identity_reverse,reverse_reverse}`;
- `ldred-{min,max}-chain0-{identity_identity,reverse_identity,identity_reverse,reverse_reverse}`.

Representative exact repro:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon TMEM_EQ_CASE='{"case_id":"ldred-min-chain0-identity_identity","family":"ldred","construct":"chain0","m":128,"n":64,"row_kind":"identity","col_kind":"identity","op":"min"}' python /tmp/tmem_descriptor_equivalence_round36/worker.py
```

Observed signature:

```text
Mismatched elements: 8064 / 8192 (98.4%)
```

This expands existing descriptor-view wrong-result coverage
`FZ-20260421-0002` / `FZ-20260421-0003`: writing through the row-chain
identity-equivalent descriptor and reading/reducing through canonical
`parent.index(1)` does not preserve the physical image.

Hardware `ld.red` opcode fallback rows:

- `ldred-{min,max}-half_row-identity_identity`;
- `ldred-{min,max}-half_row-identity_reverse`.

Representative exact repro:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon TMEM_EQ_CASE='{"case_id":"ldred-min-half_row-identity_identity","family":"ldred","construct":"half_row","m":128,"n":64,"row_kind":"identity","col_kind":"identity","op":"min"}' python /tmp/tmem_descriptor_equivalence_round36/worker.py
```

Observed signature:

```text
AssertionError: (['tcgen05.ld.sync.aligned.32x32b.x64.b32', ...],
                 ['tcgen05.ld.sync.aligned.32x32b.x64.b32', ...])
```

Runtime values were correct, but PTX/LLIR contained plain `tcgen05.ld`, not
`.ld.red`. This is existing `FZ-20260421-0004` opcode-loss/software-reduction
behavior, not a new independent bucket.

Existing half-view reduction failures:

- `ldred-{min,max}-half_row-reverse_identity`;
- `ldred-{min,max}-half_row-reverse_reverse`.

These reproduce existing `FZ-20260421-0022` with
`ttng.tmem_load failed to compute TMEM encoding info for reduction`.

- `ldred-{min,max}-half_col-identity_reverse`.

These reproduce existing `FZ-20260421-0020`-adjacent half-view lowering failure:
`failed to legalize operation 'ttng.tmem_load'` during
`ConvertTritonGPUToLLVM`.

- `ldred-{min,max}-half_col-reverse_reverse`.

These are clean scalar `.x1` diagnostics:

```text
tmem_load reduction selected a scalar tcgen05.ld.red message, but tcgen05.ld.red requires at least an .x2 message shape.
```

## Classification

No new independent `FZ-*` bucket is proposed.

This round sharpens existing buckets:

- `FZ-20260421-0002` / `FZ-20260421-0003`: row-chain descriptor-view algebra is
  not equivalent to canonical `parent.index(1)` for load/store or `ld.red`
  runtime semantics.
- `FZ-20260421-0004`: some runtime-correct `ld.red` descriptor-view reductions
  still lower as plain TMEM loads plus software reduction, so hardware opcode
  selection is absent.
- `FZ-20260421-0020`: half-column column-reversed `ld.red` descriptor views can
  reach late legalization failure.
- `FZ-20260421-0022`: row-reversed half-row `ld.red` descriptor views still
  fail during reduction encoding computation.

The checked-in copy descriptor-view guardrail stayed green (`87 passed`) and
did not expose copy-specific opcode absence, crash, false unsupported
diagnostics, or runtime miscompile in this round.
