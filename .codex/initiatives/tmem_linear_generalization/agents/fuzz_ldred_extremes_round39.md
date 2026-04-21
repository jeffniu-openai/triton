# Round 39: `ld.red` Extreme Shape and Resource Boundary Fuzzing

Date: 2026-04-21
Branch: `codex/tmem`

Scope: discovery-only lane for `tcgen05.ld.red`. No backend code or checked-in
tests were edited. This report is the only repository file written by this
lane.

## Required Build

```bash
make -j8
```

Result: no-op rebuild in
`/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`.

## Checked-In Guardrail

Collected the existing positive f32 `ld.red` selector first:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource and not m64 and not non_f32'
```

Result: `160/1615 tests collected`.

Then ran the selector split across all four GPUs with stable cache directories:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red and not reports and not resource and not m64 and not non_f32'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red and not reports and not resource and not m64 and not non_f32'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red and not reports and not resource and not m64 and not non_f32'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'ld_red and not reports and not resource and not m64 and not non_f32'
```

Results: groups 1-4 each passed `40`; aggregate `160 passed`.

## Temporary Probes

Large/resource subprocess matrix:

- worker reused from prior minimization:
  `/tmp/tmem_ldred_fz0018_min_round32/ldred_child.py`
- round 39 output directory:
  `/tmp/tmem_ldred_extremes_round39`
- summary:
  `/tmp/tmem_ldred_extremes_round39/round32_child_matrix_summary.json`
- per-case logs:
  `/tmp/tmem_ldred_extremes_round39/<case>.stdout.txt` and `.stderr.txt`

Command shape:

```bash
CUDA_VISIBLE_DEVICES=<case_index % 4> \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu<case_index % 4> \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_ldred_fz0018_min_round32/ldred_child.py '<json case>'
```

The driver launched one subprocess per row. Axes covered direct roots,
indexed roots, parent-expanding chains, same-footprint descriptor chains,
`M64/M128/M256`, `N128/N256/N512`, `num_warps=4/8`, identity, row-reverse,
column-reverse, tile-swap, M64, and legacy M256 layouts.

Half-view descriptor subprocess matrix:

- driver reused from prior descriptor-view lane:
  `/root/tmp/tmem_ldred_descriptor_views_round35_probe.py`
- round 39 output directory:
  `/tmp/tmem_ldred_extremes_round39_half`
- generated worker:
  `/tmp/tmem_ldred_extremes_round39_half/worker.py`
- summary:
  `/tmp/tmem_ldred_extremes_round39_half/summary.json`

Command:

```bash
PYTHONPATH=.:./python:./python/test/gluon python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location('probe', '/root/tmp/tmem_ldred_descriptor_views_round35_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.OUT_DIR = probe.Path('/tmp/tmem_ldred_extremes_round39_half')
probe.WORKER = probe.OUT_DIR / 'worker.py'
probe.SUMMARY = probe.OUT_DIR / 'summary.json'
probe.CASES = [c for c in probe.CASES if c['shape_kind'] in ('half_row', 'half_col') and c['n'] in (64, 128)]
probe.main()
PY
```

## Large/Resource Matrix

| Case | Classification | Hardware `.ld.red` count |
| --- | --- | --- |
| `direct_m128_n512_identity_w4` | `FZ-20260421-0018` | |
| `direct_m128_n512_identity_w8` | clean shared-memory boundary | |
| `direct_m128_n512_col_reverse_w4` | pass | 8 |
| `direct_m128_n512_row_reverse_w4` | `FZ-20260421-0018` | |
| `direct_m128_n512_tile_swap_w4` | `FZ-20260421-0018` | |
| `same_chain_m128_n512_identity_w4` | `FZ-20260421-0018` | |
| `same_chain_m128_n512_col_reverse_w4` | pass | 8 |
| `same_chain_m128_n512_identity_w8` | clean shared-memory boundary | |
| `indexed_m128_n512_identity_w4` | clean tensor-memory boundary | |
| `chain_m128_n512_identity_w4` | clean tensor-memory boundary | |
| `indexed_m128_n256_row_reverse_w4` | pass | 4 |
| `chain_m128_n256_tile_swap_w4` | pass | 4 |
| `direct_m256_n256_identity_w4` | `FZ-20260421-0018` | |
| `direct_m256_n256_identity_w8` | clean shared-memory boundary | |
| `direct_m256_n128_legacy_w8` | pass | 1 |
| `direct_m64_n512_m64_identity_w4` | pass | 4 |
| `direct_m64_n256_m64_row_reverse_w4` | pass | 1 |
| `same_chain_m64_n256_m64_row_reverse_w4` | descriptor-chain wrong-output evidence, not promoted | |

Representative diagnostics:

- `FZ-20260421-0018`: `ptxas-blackwell fatal: (C7600) Register allocation failed with register count of '255'.`
- clean shared-memory boundary: `out of resource: shared memory, Required: 262148, Hardware limit: 232448.`
- clean tensor-memory boundary: `out of resource: tensor memory, Required: 1024, Hardware limit: 512.`

The `same_chain_m64_n256_m64_row_reverse_w4` row returned a 50% replay-output
mismatch from the reused same-footprint chain harness. I did not promote this
to a new `FZ-*`: it is outside the requested resource-boundary signature,
requires a smaller dedicated oracle/minimizer, and is closer to previously
cataloged descriptor-chain wrong-output evidence than to `FZ-0012`,
`FZ-0018`, `FZ-0020`, or `FZ-0022`.

## Half-View Matrix

Counts:

```text
32 total
8 pass
8 clean unsupported descriptor-view diagnostics
4 clean scalar .x1 ld.red diagnostics
4 existing FZ-20260421-0020
8 existing FZ-20260421-0022
```

| Shape | Layouts | N | Ops | Classification |
| --- | --- | --- | --- | --- |
| `half_row` | identity, col_reverse | 64, 128 | min, max | clean unsupported descriptor-view diagnostic |
| `half_row` | row_reverse, rowcol_reverse | 64, 128 | min, max | `FZ-20260421-0022` |
| `half_col` | identity, row_reverse | 64, 128 | min, max | pass |
| `half_col` | col_reverse | 64, 128 | min, max | `FZ-20260421-0020` |
| `half_col` | rowcol_reverse | 64, 128 | min, max | clean scalar `.x1` `ld.red` diagnostic |

Representative diagnostics:

- clean unsupported descriptor-view:
  `TMEM layout 'auto' unsupported for descriptor view ... unsupported tensor memory descriptor view`
- `FZ-20260421-0022`:
  `'ttng.tmem_load' op failed to compute TMEM encoding info for reduction`
- `FZ-20260421-0020`: later optimize/LLVM pipeline failure classified by the
  reused Round 35 classifier.
- clean scalar diagnostic:
  `tmem_load reduction selected a scalar tcgen05.ld.red message, but tcgen05.ld.red requires at least an .x2 message shape.`

## Classification

No new independent `FZ-*` bucket is proposed.

- `FZ-20260421-0012`: not reproduced in this round. The M64 rows selected here
  were identity or M64 row-reverse controls and passed hardware `ld.red`.
- `FZ-20260421-0018`: reproduced for large 4-warp hardware `ld.red` tiles:
  direct `M128xN512`, direct row/tile permutations, same-footprint descriptor
  chain `M128xN512`, and direct `M256xN256`.
- Clean resource diagnostics: reproduced for the expected `num_warps=8`
  shared-memory boundaries and parent-expanding high-N descriptor roots that
  exceed tensor-memory capacity.
- `FZ-20260421-0020`: reproduced for half-column column-reversed descriptor
  views at `N=64/128`, both `min` and `max`.
- `FZ-20260421-0022`: reproduced for half-row row-reversed and row+column
  reversed descriptor views at `N=64/128`, both `min` and `max`.

The strongest resource-boundary signal remains existing `FZ-0018`: it is
specific to large 4-warp hardware `ld.red` planning and has adjacent passing
hardware controls (`M128xN512` column-reversed, `M64xN512`, and
`M256xN128,w8`) plus adjacent clean resource diagnostics.
