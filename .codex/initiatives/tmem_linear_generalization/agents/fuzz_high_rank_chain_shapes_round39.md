# Round 39 High-Rank Descriptor-View Chain Shapes

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

## Checked-In Selector

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k '(higher_rank or rank5 or half_rows or ld_red and descriptor_chain) and not reports and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k '(higher_rank or rank5 or half_rows or ld_red and descriptor_chain) and not reports and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k '(higher_rank or rank5 or half_rows or ld_red and descriptor_chain) and not reports and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k '(higher_rank or rank5 or half_rows or ld_red and descriptor_chain) and not reports and not resource'
```

Result: `92 passed, 20 skipped, 6328 deselected` split as
`33 passed`, `13 passed / 20 skipped`, `33 passed`, `33 passed`.

## Temporary Probe

Artifacts:

- driver: `/root/tmp/fuzz_high_rank_chain_shapes_round39_probe.py`
- generated worker: `/tmp/tmem_high_rank_chain_shapes_round39/worker.py`
- summary: `/tmp/tmem_high_rank_chain_shapes_round39/summary.json`

Command:

```bash
PYTHONPATH=.:./python python /root/tmp/fuzz_high_rank_chain_shapes_round39_probe.py
```

The driver dispatches subprocess-isolated rows with four concurrent workers:

```bash
CUDA_VISIBLE_DEVICES=<case_index % 4> \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu<case_index % 4> \
PYTHONPATH=.:./python \
ROUND39_CASE='<case-json>' \
python /tmp/tmem_high_rank_chain_shapes_round39/worker.py
```

Matrix:

- consumers: `ld/st`, `load_min`, `load_max`;
- view shapes: direct full tile, rank-4 unit-prefix full view, rank-5
  unit-prefix full view, inner unit-rank full view, rank-5 lower half-row
  view, unit-rank upper half-column view;
- layouts: identity, row-reverse, column-reverse, and even/odd row plus
  rotated-column bases;
- variants: `auto` and explicit `32x32b`;
- shape: `M=128, N=128`, `f32`.

Aggregate:

| Classification | Count |
| --- | ---: |
| pass | 80 |
| existing opcode loss / `FZ-20260421-0004` | 16 |
| clean TMEM resource boundary | 24 |
| existing half-column/unit-rank abort / `FZ-20260421-0021` | 24 |

By view family:

| View family | Result |
| --- | --- |
| `direct_full` | `24 pass`; direct `ld.red` emits `tcgen05.ld.red.sync.aligned.32x32b.x128.{min,max}.f32`. |
| `rank4_unit_prefix_full` | `24 pass`; rank-4 unit-prefix full views preserve runtime values and hardware `ld.red`. |
| `rank5_unit_prefix_full` | `24 pass`; rank-5 unit-prefix full views preserve runtime values and hardware `ld.red`. |
| `unit_inner_rank_full` | `24` clean resource diagnostics: `out of resource: tensor memory, Required: 16384, Hardware limit: 512`. |
| `rank5_half_row_lower` | `8 pass` for `ld/st`; `16` runtime-correct `ld.red` opcode-loss rows classified as existing `FZ-20260421-0004`. |
| `unit_rank_half_col_upper` | `24` process aborts classified as existing `FZ-20260421-0021`: `TMEM layout shape must be bounded by the memdesc shape and allocShape`, followed by a `MemDescType` invariant assertion. |

Representative repros:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python ROUND39_CASE='{"case_id":"ldred-rank5_half_row_lower-identity_identity-auto-min","col_kind":"identity","family":"ldred","m":128,"n":128,"red_op":"min","row_kind":"identity","seed":98,"variant":"auto","view_kind":"rank5_half_row_lower"}' python /tmp/tmem_high_rank_chain_shapes_round39/worker.py
```

Observed: runtime reduction values match PyTorch, but PTX contains plain
`tcgen05.ld.sync.aligned.32x32b.x128.b32` and no `.ld.red`; existing
`FZ-20260421-0004`.

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python ROUND39_CASE='{"case_id":"ldst-unit_rank_half_col_upper-identity_identity-auto","col_kind":"identity","family":"ldst","m":128,"n":128,"red_op":null,"row_kind":"identity","seed":121,"variant":"auto","view_kind":"unit_rank_half_col_upper"}' python /tmp/tmem_high_rank_chain_shapes_round39/worker.py
```

Observed: subprocess aborts after
`TMEM layout shape must be bounded by the memdesc shape and allocShape` and a
`MemDescType` invariant assertion from `get_reg_layout`; existing
`FZ-20260421-0021` half-column/unit-rank dimension abort family.

## Classification

No new independent `FZ-*` bucket is proposed.

- No row reproduced `FZ-20260421-0002` or `FZ-20260421-0003`: there were no
  runtime wrong-result `ld/st` or `ld.red` mismatches in this matrix.
- No row reproduced `FZ-20260421-0019`: rank-4 and rank-5 unit-prefix full
  views were green; inner unit-rank rows stopped at a clean TMEM resource
  boundary before exercising `get_reg_layout` abort behavior.
- No row reproduced `FZ-20260421-0020` or `FZ-20260421-0022`: the half-row
  reduction rows were runtime-correct software-reduction opcode-loss cases,
  and the half-column/unit-rank rows matched the existing `FZ-20260421-0021`
  process-abort signature.
- The only opcode-loss signal is existing `FZ-20260421-0004`, broadened here
  to rank-5 lower half-row descriptor views across `min`/`max`, `auto` and
  `32x32b`, and four row/column basis layouts.

Backend repair remains deferred per the discovery-only mandate.
