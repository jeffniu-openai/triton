# Lane AB Round 14: scaled-MMAv5 descriptor/view and scale-copy fuzzing

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery only; no backend/compiler fixes and no commit.

## Scope

This lane fuzzed scaled-MMAv5 interactions outside the already-green
`mma_scaled and (tile_permuted or narrow or e2m1 or fp4)` control slice. The
focus was:

- shared scale descriptor views that trigger automatic `tcgen05.cp` into TMEM
  scales;
- direct shared-scale operands versus explicitly copied TMEM scales;
- accumulator descriptor views with indexed/subslice/narrow N shapes;
- 1CTA and 2CTA variants, including `use_acc=True`;
- neighboring clean unsupported diagnostics for B-scale descriptor layouts.

Backend/compiler code was not modified.

## Build

Command:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Checked-In Selector

Collected the scaled-MMAv5 runtime-matrix rows outside the excluded green
tile/narrow/fp4/e2m1 slice:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and not (tile_permuted or narrow or e2m1 or fp4)'
```

Result: `69/1615` selected.

Four-GPU split run:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and not (tile_permuted or narrow or e2m1 or fp4)'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and not (tile_permuted or narrow or e2m1 or fp4)'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and not (tile_permuted or narrow or e2m1 or fp4)'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'mma_scaled and not (tile_permuted or narrow or e2m1 or fp4)'
```

Result:

- group 1: `18 passed, 1597 deselected`
- group 2: `18 passed, 1597 deselected`
- group 3: `18 passed, 1597 deselected`
- group 4: `15 passed, 1600 deselected`

This confirms the existing scaled descriptor/control rows stay green on current
HEAD.

## Temporary Probe A: shared scale descriptor auto-copy and explicit scale copy

Probe file: `/tmp/tmem_scaled_descriptor_round14_probe.py`

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:python/test/gluon \
  python /tmp/tmem_scaled_descriptor_round14_probe.py
```

Rows probed: `10`.

Classification:

- `8` pass
- `2` harness/setup limitations
- `0` compiler crashes
- `0` opcode mismatches
- `0` runtime miscompiles

Positive rows:

- `direct-shared-1cta-n128-slice128-useacc`: `cp=2`, `mma=4`,
  `tcgen05.cp.cta_group::1.warpx4.32x128b`,
  `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`
- `direct-shared-1cta-n64-slice64-useacc`: same 1CTA opcodes, `cp=2`,
  `mma=4`
- `direct-shared-2cta-n64-slice64-useacc-mcast`: `cp=2`, `mma=4`,
  `tcgen05.cp.cta_group::2.warpx4.32x128b`,
  `tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X`
- `direct-shared-2cta-n128-slice128-noacc`: same 2CTA opcodes, `cp=2`,
  `mma=4`
- `auto-copy-1cta-n64-slice64-useacc`: same 1CTA opcodes, `cp=2`,
  `mma=4`
- `auto-copy-1cta-n128-slice128-k256-useacc`: same 1CTA opcodes, `cp=4`,
  `mma=8`
- `direct-shared-1cta-n32-slice32`: same 1CTA opcodes, `cp=2`, `mma=4`
- `auto-copy-2cta-n32-slice32`: same 2CTA opcodes, `cp=2`, `mma=4`

The two high-CGA contrast rows failed before backend compilation:

```text
AssertionError: Shape [1, 1, 1, 2, 256] is not divisible by CGA layout [[0, 1, 0, 0, 0]]
```

These are classified as temporary harness/setup limitations. The shared helper
constructs 2CTA scale CGA layouts for `num_ctas > 1`, so it is not a valid
backend probe for 4CTA scale descriptors.

## Temporary Probe B: B-scale descriptor-view chains

Probe files:

- `/tmp/tmem_scaled_bscale_descriptor_round14_probe.py`
- `/tmp/tmem_scaled_bscale_descriptor_round14_probe2.py`
- `/tmp/tmem_scaled_bscale_min_repro_round14.py`
- `/tmp/tmem_scaled_bscale_error_round14.py`

Primary command:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:python/test/gluon \
  python /tmp/tmem_scaled_bscale_descriptor_round14_probe2.py
```

Rows probed in the corrected B-scale sweep: `7`.

Classification:

- `1` pass
- `3` runtime miscompile candidates
- `3` clean unsupported / diagnostic-boundary rows

Passing row:

- `bscale-n128-tile32-k256-pad1`: `mma=32`
  `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`;
  TTGIR retained descriptor reshapes/transposes and a TMEM load.

Clean unsupported / diagnostic-boundary rows:

- `bscale-n64-tile16-pad0`: rejected as unpadded B-scale storage for an
  accumulator schedule requiring N=16 instructions; the diagnostic explains
  the required B-scale padding/rematerialization factor.
- `bscale-n64-tile16-pad1-extra`: same unsupported layout family; isolated
  rerun reports `RuntimeError: error encountered during parsing` after the
  frontend emits the explicit scaled-MMAv5 unsupported diagnostic.
- `bscale-n32-tile8-pad1-boundary`: same narrow-N unsupported family.

The earlier `/tmp/tmem_scaled_bscale_descriptor_round14_probe.py` also included
three invalid tile choices where `_make_tmem_linear_layout_tile_permuted`
raised `IndexError`; those were discarded as probe-shape limitations.

## New candidate: B-scale descriptor-view miscompile

The corrected B-scale probe found three stable wrong-result rows that compiled,
executed, emitted matching LLIR/PTX scaled-MMA opcodes, and retained descriptor
view chains in TTGIR:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:python/test/gluon \
  python /tmp/tmem_scaled_bscale_min_repro_round14.py
```

Result:

```text
bscale-n128-linear-pad0-extra mismatch 16111 / 16384 max_abs 746.5385131835938
bscale-n128-linear-pad0-extra mma_count 4 sample tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
bscale-n128-linear-pad0-extra ttgir reshape True trans True tmem_load True
bscale-n128-linear-pad1 mismatch 12270 / 16384 max_abs 759.0625
bscale-n128-linear-pad1 mma_count 4 sample tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
bscale-n128-linear-pad1 ttgir reshape True trans True tmem_load True
bscale-n256-tile64-k128-pad0 mismatch 32413 / 32768 max_abs 621.6177978515625
bscale-n256-tile64-k128-pad0 mma_count 16 sample tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
bscale-n256-tile64-k128-pad0 ttgir reshape True trans True tmem_load True
```

Minimal current forms:

- `N=128`, linear accumulator layout, B-scale descriptor view via
  `reshape -> trans -> reshape`, unpadded B-scale storage, `EXTRA_B_SCALE_USER=True`.
- `N=128`, linear accumulator layout, padded B-scale storage, no extra user.
- `N=256`, tile-permuted accumulator layout with `tile_n=64`, unpadded B-scale
  storage.

This does not map cleanly to existing scaled-MMAv5 buckets:

- It is not `FZ-20260421-0007`; there is no dynamic accumulator-view selection.
- It is not `FZ-20260421-0010`; all miscompile rows use local 1CTA launch.
- It is not `FZ-20260421-0011`; FPSAN is not involved.

Recommendation: open a new bucket, tentatively
`FZ-20260421-0013: scaled-MMAv5 B-scale descriptor-view miscompile`, unless a
later audit proves it is another manifestation of an already recorded
descriptor-view scale-fragment contract gap.

## Summary

Total rows executed or classified in this lane:

- checked-in selector: `69 passed`
- temporary shared-scale/copy probe: `8 passed`, `2` harness/setup limitations
- temporary B-scale descriptor probe: `1 passed`, `3` miscompile candidates,
  `3` clean unsupported / diagnostic-boundary rows

No backend fixes were attempted. The only new independent finding is the
B-scale descriptor-view wrong-result candidate above.
