# Round 44: `tcgen05.ld.red` M64 and Descriptor Boundary Catalog

Date: 2026-04-21
Branch: `codex/tmem`
Scope: discovery/classification only. No backend/compiler source was modified.
Owned write path: this report only.

## Summary

The checked-in `ld_red_m64` selector remains stable:

```text
39 collected
33 passed
6 failed
```

All six failures are the existing `FZ-20260421-0012` M64 row-basis
unsupported-destination diagnostic. The passing controls include direct M64
split-N loads for `N={32,64,128,256}`, row-identity column-reversed M64 rows,
and explicit split-N variant controls.

Clean resource diagnostics stayed clean:

```text
4 passed
```

The structural descriptor/`ld.red` sentinel selector stayed stable:

```text
3 passed
11 xfailed
```

The strict xfails cover existing `FZ-20260421-0003` descriptor-view
wrong-output sentinels, `FZ-20260421-0004` opcode-loss/software-reduction
sentinels, and adjacent known `ld.red` crash/unsupported sentinels. There was
no XPASS, no new unexpected failure, and no diagnostic drift.

I also reran the Round 41 temporary M64 same-footprint descriptor-chain oracle
for the smallest known discriminator. It still classifies as existing
`FZ-20260421-0003` evidence, not `FZ-0012` or `FZ-0004`: same-footprint M64
descriptor chains produce 50% zero-row wrong output, while direct M64
`ld.red` with the same footprint passes and the reduction row emits hardware
`.ld.red`.

No new independent `FZ-*` is warranted.

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

## Checked-In Collection

M64 runtime selector:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red_m64'
```

Result:

```text
39/1615 tests collected (1576 deselected)
```

Descriptor/M64 equivalent selector:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ld_red and descriptor_chain and m64) or (ld_red_m64 and not reports)'
```

Result:

```text
39/1615 tests collected (1576 deselected)
```

Clean/resource `ld.red` selector:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ld_red and (reports or resource or clean)) or (ld_red_m64 and reports)'
```

Result:

```text
4/1615 tests collected (1611 deselected)
```

Structural descriptor/`ld.red` sentinels:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py \
  -k 'ldred or descriptor_view'
```

Result:

```text
14/33 tests collected (19 deselected)
```

## Checked-In Runtime Results

M64 `ld.red` split-4:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_m64'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_m64'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_m64'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red_m64'
```

Results:

```text
group 1/GPU0: 10 passed, 1605 deselected
group 2/GPU1:  6 passed, 4 failed, 1605 deselected
group 3/GPU2: 10 passed, 1605 deselected
group 4/GPU3:  7 passed, 2 failed, 1606 deselected
aggregate:    33 passed, 6 failed
```

Clean/resource diagnostics:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(ld_red and (reports or resource or clean)) or (ld_red_m64 and reports)'
```

Result:

```text
4 passed, 1611 deselected
```

Structural sentinels:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_structural_fuzzer.py -k 'ldred or descriptor_view'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_structural_fuzzer.py -k 'ldred or descriptor_view'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_structural_fuzzer.py -k 'ldred or descriptor_view'
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_structural_fuzzer.py -k 'ldred or descriptor_view'
```

Results:

```text
group 1/GPU1: 1 passed, 3 xfailed
group 2/GPU2: 2 passed, 2 xfailed
group 3/GPU3: 4 xfailed
group 4/GPU0: 2 xfailed
aggregate:    3 passed, 11 xfailed
```

## Exact `FZ-20260421-0012` Rerun

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]'
```

Result:

```text
6 failed
```

Failing nodeids:

```text
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]
python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]
```

Representative diagnostic:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
requested layout direct-lowering details:
Failed to lower TMEM load/store: unsupported dst layout
where out dims are: [row (size 128), col (size 32)]
```

The `row_rotate_col_even_odd_n128` rows report the same unsupported-destination
diagnostic for `[row (size 128), col (size 128)]`.

Classification: existing `FZ-20260421-0012`, not a new bucket. The exact
failures are still limited to M64 row-basis cases; row-identity M64 controls
inside the same selector passed.

## M64 Descriptor-Chain Oracle Refresh

Temporary oracle:

```text
/tmp/tmem_m64_descriptor_oracle_round41.py
```

Smallest plain-load discriminator:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_m64_descriptor_oracle_round41.py \
  '{"mode":"same_load","layout":"m64","n":32,"warps":4}'
```

Result:

```text
status: ran
out mismatches: 1024 / 2048 (50%)
mismatch rows: 16..31 and 48..63
ptx_ld_count: 2
ptx_ldred_count: 0
```

Smallest `ld.red` descriptor-chain discriminator:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_m64_descriptor_oracle_round41.py \
  '{"mode":"same_ldred","layout":"m64","n":32,"op":"min","variant":"auto","warps":4}'
```

Result:

```text
status: ran
out mismatches: 1024 / 2048 (50%)
red mismatches: 32 / 64 (50%)
mismatch rows: 16..31 and 48..63
ptx_ldred_count: 2
llir_ldred_count: 2
```

Direct M64 control:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_m64_descriptor_oracle_round41.py \
  '{"mode":"direct_ldred","layout":"m64","n":32,"op":"min","variant":"auto","warps":4}'
```

Result:

```text
status: ran
out mismatches: 0 / 2048
red mismatches: 0 / 64
ptx_ldred_count: 2
llir_ldred_count: 2
```

Classification: existing `FZ-20260421-0003` M64 descriptor-chain wrong-output
evidence. This is not `FZ-20260421-0012` because there is no unsupported
destination diagnostic and the direct M64 control passes. It is not
`FZ-20260421-0004` because hardware `.ld.red` is present for the reduction row,
and the plain-load discriminator fails without reduction opcode selection.

## Boundary Classification

- `FZ-20260421-0012`: stable. Six checked-in M64 row-basis `ld.red` rows fail
  with unsupported destination-layout diagnostics.
- `FZ-20260421-0003`: stable. The refreshed M64 same-footprint descriptor-chain
  oracle reproduces 50% zero-row wrong output on rows `16..31` and `48..63`;
  direct M64 `ld.red` passes.
- `FZ-20260421-0004`: stable through structural strict xfails. No new opcode
  loss was found in the M64 checked-in runtime selector or Round 41 M64 oracle
  refresh.
- Clean resource/unsupported diagnostics: stable. The checked-in
  `identity_256x256` `ld.red` resource-boundary rows passed with clean
  resource handling and no compiler assertion/PassManager drift.

No new independent `FZ-*` is warranted, and no backend repair was attempted.
