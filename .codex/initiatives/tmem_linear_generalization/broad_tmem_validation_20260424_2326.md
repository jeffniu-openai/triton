# Broad TMEM Validation Reconfirmation - 2026-04-24 23:26 UTC

## Scope

Reconfirmed the green/red state after the merge-resolution MMAv5/pass-order fix at `6a50032c3`. Per repository instructions, ran `make` first, then reran the broad Gluon TMEM validation files across all 4 GPUs without `TRITON_ALWAYS_COMPILE`:

```bash
make
CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> PYTHONPATH=./python \
  pytest -s --tb=short --splits 4 --group <1..4> \
  python/test/gluon/test_core.py \
  python/test/gluon/test_tmem_runtime_matrix.py \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

## Result

The suite is **not green**. Three shards were green and one shard failed.

- Group 1 / GPU 0: `4347 passed, 620 skipped, 14901 deselected` in `0:52:05`.
- Group 2 / GPU 1: `2671 passed, 2296 skipped, 14901 deselected` in `0:33:10`.
- Group 3 / GPU 2: `2359 passed, 2608 skipped, 14901 deselected` in `0:29:54`.
- Group 4 / GPU 3: `184 failed, 3394 passed, 1389 skipped, 14901 deselected` in `0:50:17`.

Aggregated selected-test outcome across the split: `184 failed, 12771 passed, 6913 skipped`.

## Failure Buckets Observed

The last targeted MMAv5 accumulator/scaled-MMAv5 tile-permuted rows fixed by `6a50032c3` are not the remaining blocker. The broad run still exposes older/open TMEM backend gaps:

- Subword ld/st dynamic and loop-carried descriptor views: opcode expectations produce scalar `x1` sequences instead of the expected wide atom, and dynamic f16 subword views have real roundtrip miscompares.
- Multidimensional descriptor ld/st views: several tests still expect `tt.reshape` in TTGIR, while other replay/half-row cases have clean unsupported `get_reg_layout` failures or 50% roundtrip miscompares.
- Tensor-memory scales descriptor views: several positive rows now fail in frontend `get_reg_layout`; some clean-unsupported rows report different generic unsupported text than the test expects.
- `ld.red` M64/permuted/tile-permuted layouts: compiler lowering failures for unsupported destination layouts, opcode-count expectation mismatches, and multiple 50% runtime miscompares.
- `tcgen05.copy`: dense 256-row copy positives are now clean unsupported; warpx2 descriptor/subslice/indexed views and tile-permuted/tile-selector-permuted linear layouts show runtime miscompares.
- MMAv5/Scaled-MMAv5 LHS tile-permuted cases: plain LHS tile-permuted f16/f8 rows and scaled LHS tile-permuted rows still miscompare; scaled accumulator subslice tile-permuted rows fail earlier in `get_reg_layout` for descriptor views.
- Structural fuzzer clean-unsupported sentinels for `FZ-20260421-0003` descriptor-view ld/st now compile instead of raising, so those rows need correctness classification rather than assuming the old diagnostic remains valid.

## Interpretation

The branch is **not broadly green**. The latest focused merge-fallout fixes improved the MMAv5 accumulator/scaled-MMAv5 address/pass-order subset, but the broad TMEM matrix still has substantial open correctness and expectation buckets. Some failures are stale expectation checks, but many are real runtime miscompares or compile failures and should be treated as open backend gaps until minimized and fixed or reclassified as true clean unsupported cases.
