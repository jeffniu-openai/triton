# Round 14 Lane AE: LD/ST Descriptor Algebra Fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler fixes were attempted.
- Repo edit scope: this report and initiative docs only.
- Temporary probe:
  `/tmp/tmem_ldst_descriptor_algebra_round14_probe.py`
- Logs:
  `/tmp/tmem_ldst_descriptor_algebra_round14_probe_final.log`,
  `/tmp/tmem_ldst_descriptor_algebra_round14_{g1,g2,g3,g4}.log`,
  `/tmp/tmem_ldst_descriptor_algebra_round14_controls_{g1,g2,g3,g4}.log`

## Scope

Lane AE targeted direct `ld/st` descriptor-view algebra around rank-6
parents, deeper `reshape/trans/slice/index` chains, row/column permutations,
descriptor roundtrips/compositions, runtime selector edges, subword views, and
copy/MMAv5 consumers as controls.

## Commands

Required rebuild:

```bash
make -j8
```

Result: `ninja: no work to do`.

Custom probe:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_ldst_descriptor_algebra_round14_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldst_descriptor_algebra_round14_probe.py \
  2>&1 | tee /tmp/tmem_ldst_descriptor_algebra_round14_probe_final.log
```

The driver launched one fresh subprocess per row, rotating
`CUDA_VISIBLE_DEVICES={0,1,2,3}` and stable caches
`TRITON_CACHE_DIR=/tmp/triton-cache-gpu{0,1,2,3}`.

Checked-in `ld/st` descriptor/rank/subword selector:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst and (descriptor_chain or descriptor_roundtrip or descriptor_compositions or runtime_selector or rowcol_permuted or rank5 or higher_rank or subword) and not reports'
```

Result: `232/1615` rows collected.

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  --store-durations --durations-path /tmp/tmem_local_r14_lane_ae_ldst_descriptor_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ldst and (descriptor_chain or descriptor_roundtrip or descriptor_compositions or runtime_selector or rowcol_permuted or rank5 or higher_rank or subword) and not reports'
```

Copy/MMAv5 controls:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and (subword or descriptor or twocta) or mma_twocta_indexed_acc_view or acc_subslice_view) and not reports'
```

Result: `243/1615` rows collected.

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group <group> \
  --store-durations --durations-path /tmp/tmem_local_r14_lane_ae_controls_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k '(cp_no_scales and (subword or descriptor or twocta) or mma_twocta_indexed_acc_view or acc_subslice_view) and not reports'
```

## Results

Custom probe summary:

```text
SUMMARY {"FZ-20260421-0001": 2, "FZ-20260421-0003": 2, "clean-direct-ldst-unsupported": 5, "clean-rank-view-diagnostic": 2, "pass": 1}
```

Checked-in `ld/st` selector:

- group 1/GPU 0: `46 passed, 12 skipped`
- group 2/GPU 1: `58 skipped`
- group 3/GPU 2: `50 passed, 8 skipped`
- group 4/GPU 3: `38 passed, 20 skipped`
- aggregate: `134 passed, 98 skipped`, no failures

Copy/MMAv5 controls:

- group 1/GPU 0: `61 passed`
- group 2/GPU 1: `61 passed`
- group 3/GPU 2: `61 passed`
- group 4/GPU 3: `60 passed`
- aggregate: `243 passed`, no failures

## Case Classification

| Case id | Seed | Shape / chain | Outcome |
| --- | ---: | --- | --- |
| `ae-rank6-identity-128x32-32x32b` | `0xAE01` | rank-6 `[1,1,1,2,128,32]`, identity | clean direct-`ld/st` row-anchor diagnostic |
| `ae-rank6-row-even-col-reverse-128x64-16x64b` | `0xAE02` | rank-6, row even/odd, col reverse | clean direct-`ld/st` row-anchor diagnostic |
| `ae-rank6-m64-row-reverse-64x64-32x32b` | `0xAE03` | rank-6 M64 row-reverse | clean rank-view diagnostic |
| `ae-rank6-twocta-even-256x32-32x32b` | `0xAE04` | rank-6 two-CTA, row even/odd | pass; PTX/LLIR `tcgen05.ld/st` opcodes matched |
| `ae-rank6-f16-subword-128x32-16x64b` | `0xAE05` | rank-6 f16 subword | clean direct-`ld/st` row-anchor diagnostic |
| `ae-view-chain1-f32-known-fz0003-64x32` | `0xAE06` | known chain1 f32 view-read | `FZ-20260421-0003`, `1984/2048` mismatches |
| `ae-view-chain2-f16-known-fz0003-64x32` | `0xAE07` | known chain2 f16 view-read | `FZ-20260421-0003`, `1023/2048` mismatches |
| `ae-view-chain3-rowcol-64x64` | `0xAE08` | row rotate, col reverse, transpose/slice | clean direct-`ld/st` row-anchor diagnostic |
| `ae-dynamic-index-direct-sel0` | `0xAE09` | runtime `parent.index(load(selector))` | `FZ-20260421-0001` |
| `ae-dynamic-index-chain1-sel1` | `0xAE0A` | runtime index plus chain1 | `FZ-20260421-0001` |
| `ae-rank6-rotate-subword-i32-128x32-x1` | `0xAE0B` | rank-6 i32, row rotate, col even/odd | clean direct-`ld/st` row-anchor diagnostic |
| `ae-rank6-m64-even-reverse-64x128-16x128b` | `0xAE0C` | rank-6 M64, row even/odd, col reverse | clean rank-view diagnostic |

## Classification

No new independent `FZ-*` bucket is warranted from Lane AE.

- `FZ-20260421-0001`: runtime descriptor indexing still reaches LLVM
  conversion as illegal `ttg.memdesc_index`, including direct and chain1
  `ld/st` consumers.
- `FZ-20260421-0003`: known direct `ld/st` descriptor-view packet-order
  miscompile reproduced for f32 chain1 and f16 chain2 rows.
- `FZ-20260421-0005/0009`: no allocator assertion or resource-crash row was
  found in this lane.
- `FZ-20260421-0012`: not reproduced by direct `ld/st`; row/column-permuted
  `ld/st` runtime-matrix controls stayed green or expected-skip. The M64
  rank-6 probe rows stopped at a compile-time rank-view diagnostic, not the
  `ld.red` M64 destination-layout failure.
- Rank-6 single-CTA rows mostly broaden the already-recorded Round 10
  direct-`ld/st` row-anchor clean diagnostic. A two-CTA rank-6 row did execute
  and compare correctly.
- Copy and MMAv5 descriptor-view controls stayed green, so this lane does not
  broaden `FZ-20260421-0011` or `FZ-20260421-0013`.

Backend repair remains deferred.
