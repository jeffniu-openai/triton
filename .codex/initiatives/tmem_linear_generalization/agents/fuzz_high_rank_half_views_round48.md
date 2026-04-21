# Round 48: high-rank and half-view TMEM fuzzing

Date: 2026-04-21
Branch/HEAD: `codex/tmem` at `ac874ace1`
Scope: report-only fuzzing of high-rank descriptor views and half-row /
half-column boundaries across TMEM `ld/st` and `ld.red`. No backend or test code
was changed.

## Summary

No new independent `FZ-*` bucket is needed.

Checked-in runtime coverage stayed stable:

- High-rank / half-view `ld/st`: `112` selected, `92 passed, 20 skipped`.
- Adjacent `ld.red` descriptor/M64/row-permutation selector: `88` selected,
  `82 passed, 6 failed`. All six failures match existing
  `FZ-20260421-0012` M64 f32 `tcgen05.ld.red` destination-layout planner
  coverage.
- Structural `ld/st` + `ld.red` subset: `15` selected,
  `7 passed, 8 xfailed`.

Disposable `/tmp` repros sharpened the boundary:

- Unit-rank half-column `ld/st` still aborts with the existing
  `FZ-20260421-0021` memdesc-shape / `MemDescType` invariant failure.
- Rank-5 half-row `ld/st` passes.
- Rank-5 half-row `ld.red` through the Round 39 high-rank worker emits a plain
  `tcgen05.ld.sync...` instead of hardware `tcgen05.ld.red...`, matching
  existing `FZ-20260421-0004`.
- Older Round 35 half-view disposable probes were discarded as evidence here:
  on this checkout their helper-side `SliceLayout` rank mismatch fires before
  reaching TMEM lowering, so they do not classify `FZ-20260421-0020` or
  `FZ-20260421-0022` in this round.

## Commands

Rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k '(rank5 or higher_rank or multidim_slice or half_rows) and ldst' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `112/1615 tests collected`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k 'ld_red_descriptor_chain or ldst_direct_higher_rank_load_red or ld_red_m64 or ld_red_rowcol_permuted or ld_red_pure_row_permuted' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `88/1615 tests collected`.

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  -k 'ldst_view or ldred' \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `15/33 tests collected`.

High-rank / half-view `ld/st` runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 -k '(rank5 or higher_rank or multidim_slice or half_rows) and ldst' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 -k '(rank5 or higher_rank or multidim_slice or half_rows) and ldst' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 -k '(rank5 or higher_rank or multidim_slice or half_rows) and ldst' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 -k '(rank5 or higher_rank or multidim_slice or half_rows) and ldst' python/test/gluon/test_tmem_runtime_matrix.py
```

Results:

```text
group 1: 28 passed, 1587 deselected in 3.29s
group 2: 28 passed, 1587 deselected in 3.29s
group 3: 13 passed, 15 skipped, 1587 deselected in 3.24s
group 4: 23 passed, 5 skipped, 1587 deselected in 3.22s
aggregate: 92 passed, 20 skipped
```

The skips are the intentionally skipped rank-5 lifted roundtrip rows that exceed
the current tensor-memory allocation boundary.

Adjacent `ld.red` descriptor/M64/row-permutation runtime sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 -k 'ld_red_descriptor_chain or ldst_direct_higher_rank_load_red or ld_red_m64 or ld_red_rowcol_permuted or ld_red_pure_row_permuted' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 -k 'ld_red_descriptor_chain or ldst_direct_higher_rank_load_red or ld_red_m64 or ld_red_rowcol_permuted or ld_red_pure_row_permuted' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 -k 'ld_red_descriptor_chain or ldst_direct_higher_rank_load_red or ld_red_m64 or ld_red_rowcol_permuted or ld_red_pure_row_permuted' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 -k 'ld_red_descriptor_chain or ldst_direct_higher_rank_load_red or ld_red_m64 or ld_red_rowcol_permuted or ld_red_pure_row_permuted' python/test/gluon/test_tmem_runtime_matrix.py
```

Results:

```text
group 1: 4 failed, 18 passed, 1593 deselected in 4.63s
group 2: 2 failed, 20 passed, 1593 deselected in 4.31s
group 3: 22 passed, 1593 deselected in 3.74s
group 4: 22 passed, 1593 deselected in 3.73s
aggregate: 82 passed, 6 failed
```

Failed nodeids, all existing `FZ-20260421-0012`:

- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-min]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_reverse_n32-max]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-min]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_default_layout[row_rotate_col_even_odd_n128-max]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_reverse_n32]`
- `test_tmem_runtime_matrix_ld_red_m64_rowcol_permuted_explicit_32x32b_uses_splitn[row_rotate_col_even_odd_n128]`

Representative diagnostic:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
Failed to lower TMEM load/store: unsupported dst layout
```

Structural fuzzer subset:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short \
  -k 'ldst_view or ldred' \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result:

```text
7 passed, 18 deselected, 8 xfailed in 6.69s
```

## Disposable Probes

The disposable probes used existing `/tmp` workers from earlier rounds. They
were used only for classification sharpening; no repo code was changed.

Unit-rank half-column `ld/st`, existing `FZ-20260421-0021`:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
ROUND39_CASE='{"case_id":"ldst-unit_rank_half_col_upper-identity_identity-auto","family":"ldst","m":128,"n":128,"row_kind":"identity","col_kind":"identity","view_kind":"unit_rank_half_col_upper","variant":"auto","red_op":"min","seed":121}' \
python /tmp/tmem_high_rank_chain_shapes_round39/worker.py
```

Result: process abort after a clean verifier diagnostic:

```text
TMEM layout shape must be bounded by the memdesc shape and allocShape.
shape = 128, 1, 128, allocShape = 128, 1, 128, layoutShape = 128, 128
Assertion `succeeded( ConcreteT::verifyInvariants(...))' failed.
```

Rank-5 half-row `ld/st` positive:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
ROUND39_CASE='{"case_id":"ldst-rank5_half_row_lower-identity_identity-auto","family":"ldst","m":128,"n":128,"row_kind":"identity","col_kind":"identity","view_kind":"rank5_half_row_lower","variant":"auto","red_op":"min","seed":122}' \
python /tmp/tmem_high_rank_chain_shapes_round39/worker.py
```

Result:

```json
{"status":"pass","ptx_ld":["tcgen05.ld.sync.aligned.32x32b.x128.b32","tcgen05.ld.sync.aligned.32x32b.x128.b32","tcgen05.ld.sync.aligned.32x32b.x128.b32"]}
```

Rank-5 half-row `ld.red`, existing `FZ-20260421-0004`:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
ROUND39_CASE='{"case_id":"ldred-rank5_half_row_lower-reverse_identity-auto-min","family":"ldred","m":128,"n":128,"row_kind":"reverse","col_kind":"identity","view_kind":"rank5_half_row_lower","variant":"auto","red_op":"min","seed":133}' \
python /tmp/tmem_high_rank_chain_shapes_round39/worker.py
```

Result:

```json
{"classification":"FZ-20260421-0004","ptx_ld":["tcgen05.ld.sync.aligned.32x32b.x128.b32"],"status":"opcode_loss"}
```

Identity rank-5 half-row `ld.red` produced the same `FZ-20260421-0004`
classification.

Discarded harness attempts:

```bash
python /tmp/tmem_ldred_descriptor_views_round35/worker.py '{"shape_kind":"half_col","layout_kind":"col_reverse","n":128,"red_op":"min"}'
python /tmp/tmem_ldred_descriptor_views_round35/worker.py '{"shape_kind":"half_row","layout_kind":"row_reverse","n":128,"red_op":"min"}'
```

Both failed before TMEM lowering with a helper-side `SliceLayout` rank mismatch,
so they are not used as evidence for `FZ-20260421-0020` or
`FZ-20260421-0022` in this round.

## Classification

- `FZ-20260421-0019`: no new descriptor dimension-name mismatch appeared in
  checked-in high-rank/rank-5 runtime tests or structural fuzzer subset.
- `FZ-20260421-0020`: not freshly reproduced by the checked-in selectors in
  this round; the older disposable half-view worker did not reach TMEM lowering
  and was discarded as evidence.
- `FZ-20260421-0021`: freshly reproduced by the unit-rank half-column `ld/st`
  disposable probe as the existing memdesc-shape / `MemDescType` invariant
  abort.
- `FZ-20260421-0022`: not freshly reproduced by the checked-in selectors in
  this round; the older disposable worker did not reach TMEM lowering and was
  discarded as evidence.
- `FZ-20260421-0002` / `FZ-20260421-0003`: no runtime wrong-result evidence was
  found in the checked-in high-rank `ld/st` sweep; rank-5 small and N=256 unit
  parent descriptor-chain rows stayed green.
- `FZ-20260421-0004`: structural xfails stayed xfailed, and disposable rank-5
  half-row `ld.red` rows emitted plain `tcgen05.ld.sync...` rather than
  `tcgen05.ld.red...`.
- `FZ-20260421-0012`: the six `ld.red` M64 row-plan failures are unchanged and
  remain the only runtime failures from the adjacent `ld.red` selector.

## Conclusion

Round 48 did not discover a new TMEM backend bug. The strongest fresh evidence
is that checked-in high-rank and half-row `ld/st` positives remain green, while
the high-rank half-column and rank-5 half-row `ld.red` disposable probes still
map cleanly onto existing `FZ-20260421-0021` and `FZ-20260421-0004`.
