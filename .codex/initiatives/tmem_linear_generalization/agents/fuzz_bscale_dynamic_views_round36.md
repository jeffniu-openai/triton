# Round 36 B-scale dynamic descriptor-view fuzzing

Date: 2026-04-21 13:33 UTC
Branch: `codex/tmem`
Scope: discovery-only TMEM fuzzing; no backend repairs.

## Objective

Probe shared-scale/B-scale auto-materialization around multiple B-scale users
and descriptor-view-like selection feeding `tcgen05_mma_scaled`.

Priority inputs were the checked-in selectors
`bscale_descriptor_view` and `shared_scale_descriptor_view_auto_tmem_copy`,
then a temporary Python/Gluon probe that used distinct B-scale TMEM allocations,
branch/loop descriptor selection, optional view chains, optional extra
`b_scale_tmem.load` side users, and torch runtime reference checks.

## Commands

Required rebuild:

```bash
make -j8
```

Checked-in selector collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q -s --tb=short \
  -k 'bscale_descriptor_view or shared_scale_descriptor_view_auto_tmem_copy or bscale_view_extra_user' \
  python/test/gluon/test_tmem_runtime_matrix.py
```

Checked-in selector split-4:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 -k 'bscale_descriptor_view or shared_scale_descriptor_view_auto_tmem_copy or bscale_view_extra_user' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 -k 'bscale_descriptor_view or shared_scale_descriptor_view_auto_tmem_copy or bscale_view_extra_user' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 -k 'bscale_descriptor_view or shared_scale_descriptor_view_auto_tmem_copy or bscale_view_extra_user' python/test/gluon/test_tmem_runtime_matrix.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 -k 'bscale_descriptor_view or shared_scale_descriptor_view_auto_tmem_copy or bscale_view_extra_user' python/test/gluon/test_tmem_runtime_matrix.py
```

Temporary probe collection:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest --collect-only -q -s --tb=short /tmp/tmem_bscale_dynamic_views_round36_probe.py
```

Temporary probe split-4:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 1 /tmp/tmem_bscale_dynamic_views_round36_probe.py
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_bscale_dynamic_views_round36_probe.py
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 3 /tmp/tmem_bscale_dynamic_views_round36_probe.py
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python:./python/test/gluon pytest -q -s --tb=short --splits 4 --group 4 /tmp/tmem_bscale_dynamic_views_round36_probe.py
```

## Results

- `make -j8`: no-op build.
- Checked-in selector: `7/1615` rows collected.
- Checked-in split-4: `7 passed` (`2/2/2/1`).
- Temporary valid probe: `5` rows collected and passed split-4 as
  `5 passed` (`2/2/1/0`).
- Aggregate valid runtime coverage: `12` row executions passed.

Temporary valid rows:

- `direct`: direct B-scale TMEM descriptor feeding scaled MMAv5.
- `branch-distinct`: runtime branch selects one of two distinct B-scale TMEM
  descriptors before scaled MMAv5.
- `branch-distinct-chain-users`: branch-selected B-scale descriptor views are
  reshaped/permuted/reshaped before scaled MMAv5 and are also loaded as side
  users.
- `loop-carried-chain-users`: loop-carried descriptor selection with the same
  view chain and side users.
- `branch-distinct-chain-users-sel0`: branch-selected chain/side-user case
  with the alternate selector value.

## Excluded harness iterations

Two probe drafts were excluded before classification:

- A `[2, N, K//32]` `TensorMemoryScalesLayout` parent failed during Gluon
  parsing with `Scales don't currently support multibuffering`. This is a
  clean frontend/API boundary for scale descriptors, not a backend FZ bucket.
- A tile-permuted accumulator draft initially used an invalid local layout,
  then reached the existing clean diagnostic for repeated N=32 scaled-MMAv5
  instructions with public B-scale fragments. That is a known clean boundary
  adjacent to the rematerialization path, not a new crash or miscompile.

## Classification

No new independent `FZ-*` bucket.

The valid B-scale dynamic/branch/loop descriptor-selection rows passed with
runtime reference checks. The lane did not reproduce existing
`FZ-20260421-0015` under the distinct-allocation branch-selection variant, did
not trigger `FZ-20260421-0001` because no illegal runtime `memdesc_index` was
used in the valid rows, and did not expose a new crash, false unsupported
diagnostic, opcode absence in TTGIR, or runtime miscompile.

Backend repair remains deferred; discovery should continue with broader
resource/shape pressure or a checked-in sentinel only if a stable failing row is
found.
