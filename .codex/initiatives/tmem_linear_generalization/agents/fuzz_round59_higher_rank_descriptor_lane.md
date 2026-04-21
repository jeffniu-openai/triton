# Round 59 Lane A: higher-rank descriptor-chain boundaries

Date: 2026-04-21 16:09 UTC
Branch: `codex/tmem`
HEAD before report commit: `8ee469bd0`
Mode: discovery/cataloging only. No backend, compiler, or checked-in test code was edited.

## Scope

This lane targeted descriptor-chain rank and higher-rank boundaries not covered
by the Round 56 descriptor positives or the Round 57 direct higher-rank lane:

- rank-5 small and unit-parent descriptor-chain roundtrips;
- higher-rank descriptor dim0 and half-row positives, including two-CTA rows;
- higher-rank descriptor out-of-resource diagnostics;
- direct two-CTA higher-rank load/store replay as a contrast;
- frontend dimension/rank and descriptor-chain CTA-count clean diagnostics.

Adjacent existing buckets were checked explicitly:

- `FZ-20260421-0019`: rank-4/rank-5 unit-prefix dimension-name/get-reg-layout
  aborts;
- `FZ-20260421-0021`: unit-rank half-column descriptor-view memdesc-shape
  invariant aborts.

## Required build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Checked-in runtime selector

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k "(ldst_descriptor and (rank5_small or rank5_unit_parent or higher_rank_dim0_slice_positive or higher_rank_half_rows_positive or higher_rank_dim0_slice_reports_tmem_oor or higher_rank_half_rows_reports_tmem_oor or direct_half_rows_positive)) or ldst_twocta_direct_higher_rank_load_store_replay_positive"
```

Result: `39/1615` collected.

Runtime commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k "(ldst_descriptor and (rank5_small or rank5_unit_parent or higher_rank_dim0_slice_positive or higher_rank_half_rows_positive or higher_rank_dim0_slice_reports_tmem_oor or higher_rank_half_rows_reports_tmem_oor or direct_half_rows_positive)) or ldst_twocta_direct_higher_rank_load_store_replay_positive"
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k "(ldst_descriptor and (rank5_small or rank5_unit_parent or higher_rank_dim0_slice_positive or higher_rank_half_rows_positive or higher_rank_dim0_slice_reports_tmem_oor or higher_rank_half_rows_reports_tmem_oor or direct_half_rows_positive)) or ldst_twocta_direct_higher_rank_load_store_replay_positive"
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k "(ldst_descriptor and (rank5_small or rank5_unit_parent or higher_rank_dim0_slice_positive or higher_rank_half_rows_positive or higher_rank_dim0_slice_reports_tmem_oor or higher_rank_half_rows_reports_tmem_oor or direct_half_rows_positive)) or ldst_twocta_direct_higher_rank_load_store_replay_positive"
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k "(ldst_descriptor and (rank5_small or rank5_unit_parent or higher_rank_dim0_slice_positive or higher_rank_half_rows_positive or higher_rank_dim0_slice_reports_tmem_oor or higher_rank_half_rows_reports_tmem_oor or direct_half_rows_positive)) or ldst_twocta_direct_higher_rank_load_store_replay_positive"
```

Result:

- group 1: `10 passed, 1605 deselected`
- group 2: `10 passed, 1605 deselected`
- group 3: `10 passed, 1605 deselected`
- group 4: `9 passed, 1606 deselected`
- aggregate: `39 passed`

## Frontend clean-diagnostic controls

Collection command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_frontend.py \
  -k "descriptor_chain_reports_two_ctas_mismatch or mismatch_shape_and_layout_rank or higher_rank_descriptor_type_get_reg_layout"
```

Result: `3/225` collected.

Runtime command:

```bash
PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_frontend.py \
  -k "descriptor_chain_reports_two_ctas_mismatch or mismatch_shape_and_layout_rank or higher_rank_descriptor_type_get_reg_layout"
```

Result: `3 passed, 222 deselected`.

## Disposable contrast probes

The existing Round 39 subprocess worker was still present at
`/tmp/tmem_high_rank_chain_shapes_round39/worker.py`, so this lane used two
single-row probes for adjacent bucket classification.

Existing `FZ-20260421-0021` half-column descriptor-view repro:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python ROUND39_CASE='{"case_id":"ldst-unit_rank_half_col_upper-identity_identity-auto","col_kind":"identity","family":"ldst","m":128,"n":128,"red_op":null,"row_kind":"identity","seed":121,"variant":"auto","view_kind":"unit_rank_half_col_upper"}' python /tmp/tmem_high_rank_chain_shapes_round39/worker.py
```

Result: process aborted with the existing `FZ-20260421-0021` signature:

```text
TMEM layout shape must be bounded by the memdesc shape and allocShape. shape = 128, 1, 128, allocShape = 128, 1, 128, layoutShape = 128, 128, layoutRank = 2
```

Rank-5 unit-prefix full-view contrast for `FZ-20260421-0019`:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python ROUND39_CASE='{"case_id":"ldst-rank5_unit_prefix_full-identity_identity-auto","col_kind":"identity","family":"ldst","m":128,"n":128,"red_op":null,"row_kind":"identity","seed":73,"variant":"auto","view_kind":"rank5_unit_prefix_full"}' python /tmp/tmem_high_rank_chain_shapes_round39/worker.py
```

Result: `status: pass`, with `tcgen05.ld.sync.aligned.32x32b.x128.b32`
loads observed in PTX. This lane did not reproduce `FZ-20260421-0019`.

## Classification

No new independent `FZ-*` bucket was found.

- The checked-in descriptor-chain and direct two-CTA higher-rank rows passed:
  no compiler crash, verifier drift, false unsupported diagnostic, runtime
  miscompile, or hang.
- Frontend dimension/rank mismatch and descriptor-chain CTA-count mismatch
  diagnostics stayed clean.
- Existing `FZ-20260421-0021` was freshly reproduced only by the known
  half-column/unit-rank subprocess contrast, outside the checked-in positive
  selector.
- Existing `FZ-20260421-0019` was not reproduced; the adjacent rank-5
  unit-prefix full-view contrast passed.

Backend repair remains deferred.
