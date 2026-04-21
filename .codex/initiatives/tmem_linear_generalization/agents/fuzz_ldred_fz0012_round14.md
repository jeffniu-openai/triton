# Round 14 Lane AA: `FZ-20260421-0012` M64 `ld.red` Expansion

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Repo edit scope: this report only.
- Target: broaden and minimize `FZ-20260421-0012` from
  `agents/fuzz_ldred_m64_permuted_round13.md`.

## Summary

`FZ-20260421-0012` is not isolated to the six Round 13 rows.  It generalizes
to every direct M64 `ld.red` layout probed where the M64 row basis is
non-identity, independent of:

- `N` in `{32, 64, 128, 256}`;
- column layout in `{identity, rotate1, even_odd, reverse}`;
- row permutation in `{rotate1, even_odd, reverse}`;
- reduction op `min` vs `max`;
- `abs` plus `PropagateNan.ALL` modifiers;
- explicit load variants `32x32b`, `auto`, `16x32bx2`, and
  `32x32b_splitn`; and
- descriptor-preserving `slice` and `reshape` view chains.

Nearby controls stayed green:

- all direct M64 row-identity layouts passed across the same `N` and column
  permutation sweep;
- row-identity M64 modifier and explicit-variant controls passed;
- M64 descriptor-preserving `slice`/`reshape` controls passed when the effective
  row basis stayed identity; and
- M128 direct row-permuted controls passed for the same row/column kinds.

No additional independent `FZ-*` candidate is warranted from this lane.  The
candidate should be broadened as:

```text
FZ-20260421-0012: M64 `tcgen05.ld.red` destination-layout lowering rejects
effective row-permuted M64 layouts as unsupported, even when the same column
permutation, modifiers, explicit split-N variant, or M128 row-permuted layout
is otherwise supported.
```

## Commands

Required rebuild:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Checked-in M64 selector collection:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red_m64 and not reports'
```

Result:

```text
39/1615 tests collected (1576 deselected)
```

Checked-in M64 selector split-4:

```bash
CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  --splits 4 --group <group> \
  --store-durations \
  --durations-path /tmp/tmem_laneaa_ldred_m64_durations.json \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red_m64 and not reports'
```

Results:

```text
group 1/GPU 0: 10 passed, 1605 deselected in 4.22s
group 2/GPU 1:  6 passed, 4 failed, 1605 deselected in 7.15s
group 3/GPU 2: 10 passed, 1605 deselected in 6.53s
group 4/GPU 3:  7 passed, 2 failed, 1606 deselected in 6.72s
aggregate:      33 passed, 6 failed
```

Temporary child runner:

```bash
python3 -m py_compile /tmp/tmem_laneaa_ldred_fz0012_child.py
```

Temporary subprocess-isolated grid:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python3 /tmp/tmem_laneaa_ldred_fz0012_driver.py \
  > /tmp/tmem_laneaa_ldred_fz0012_results.json
```

Each row used a fresh child process with:

```bash
CUDA_VISIBLE_DEVICES=<idx % 4>
TRITON_CACHE_DIR=/tmp/triton-cache-gpu<idx % 4>
PYTHONPATH=.:./python:./python/test/gluon
```

Exact representative reruns:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python3 /tmp/tmem_laneaa_ldred_fz0012_child.py \
  --mode default --layout m64 --n 32 --row rotate1 --col identity --op min

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python3 /tmp/tmem_laneaa_ldred_fz0012_child.py \
  --mode default --layout m64 --n 32 --row identity --col reverse --op min

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python3 /tmp/tmem_laneaa_ldred_fz0012_child.py \
  --mode explicit --layout m64 --n 32 --row even_odd --col identity \
  --variant 16x32bx2 --op min

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python3 /tmp/tmem_laneaa_ldred_fz0012_child.py \
  --mode default --layout m128 --m 128 --n 32 --row rotate1 --col identity \
  --op min
```

The first and third commands reproduced `FZ-20260421-0012`; the second and
fourth commands passed.

## Subprocess Grid Classification

Raw count from `/tmp/tmem_laneaa_ldred_fz0012_results.json`:

```text
total rows: 133
pass: 38
FZ-20260421-0012: 86
FZ-20260421-0010: 5
descriptor-permute harness-limited rows: 4
```

The temporary driver originally labeled two descriptor-permute runtime
mismatches as `FZ-20260421-0005/0009` because the generic classifier saw an
`AssertionError`; that was a classifier artifact, not an allocator assertion.
Those rows are counted here as harness-limited because the quick oracle did not
model the logical row reorder introduced by the descriptor `permute` chain.

Breakdown:

```text
direct_m64:   64 rows -> 16 pass, 48 FZ-0012
mod_m64:      16 rows ->  4 pass, 12 FZ-0012
explicit_m64: 24 rows ->  8 pass, 16 FZ-0012
desc_m64:     18 rows ->  4 pass, 10 FZ-0012, 4 descriptor-permute harness-limited
twocta_m64:    5 rows ->  5 FZ-0010
direct_m128:   6 rows ->  6 pass
```

Direct M64 row-basis split:

```text
row identity:  16 pass
row rotate1:  16 FZ-0012
row even_odd: 16 FZ-0012
row reverse:  16 FZ-0012
```

## Minimized Failing Rows

Smallest stable direct row:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python3 /tmp/tmem_laneaa_ldred_fz0012_child.py \
  --mode default --layout m64 --n 32 --row rotate1 --col identity --op min
```

Representative diagnostic:

```text
'ttng.tmem_load' op failed to compute TMEM encoding info for reduction
requested layout direct-lowering details:
Failed to lower TMEM load/store: unsupported dst layout
where out dims are: [row (size 128), col (size 32)]
```

Other minimized direct rows with the same diagnostic family:

```text
--mode default --layout m64 --n 32 --row even_odd --col identity --op min
--mode default --layout m64 --n 32 --row reverse --col identity --op min
--mode default --layout m64 --n 32 --row rotate1 --col reverse --op min
```

Modifier expansion:

```text
--mode default --layout m64 --n 32 --row reverse --col identity --op min --abs --nan all
--mode default --layout m64 --n 32 --row reverse --col identity --op max
--mode default --layout m64 --n 32 --row rotate1 --col identity --op max --abs --nan all
```

Explicit variant expansion:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python3 /tmp/tmem_laneaa_ldred_fz0012_child.py \
  --mode explicit --layout m64 --n 32 --row even_odd --col identity \
  --variant 16x32bx2 --op min
```

This failed with the same unsupported-destination diagnostic at the explicit
`tmem.load_min(layout=load_layout, ...)` site.

Descriptor-preserving chain expansion:

```text
--mode descriptor --layout m64 --n 32 --row reverse --col identity --variant 32x32b --chain slice
--mode descriptor --layout m64 --n 32 --row reverse --col identity --variant 32x32b --chain reshape
--mode descriptor --layout m64 --n 32 --row even_odd --col identity --variant 32x32b --chain slice
--mode descriptor --layout m64 --n 128 --row rotate1 --col even_odd --variant 32x32b --chain reshape
```

All failed with the same `unsupported dst layout` diagnostic.  The
row-reordering `permute` descriptor chains should get a dedicated oracle before
promotion: some fail with a compatible unsupported-destination diagnostic, and
some compile then mismatch under the temporary unpermuted-output oracle.

## Nearby Green Controls

Direct M64 column permutations with row identity passed for all probed
`N`:

```text
row identity, col identity, N={32,64,128,256}
row identity, col rotate1,  N={32,64,128,256}
row identity, col even_odd, N={32,64,128,256}
row identity, col reverse,  N={32,64,128,256}
```

Representative command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python3 /tmp/tmem_laneaa_ldred_fz0012_child.py \
  --mode default --layout m64 --n 32 --row identity --col reverse --op min
```

Result:

```text
PASS
```

Explicit M64 column controls also passed:

```text
row identity, col reverse, N=32,  variant={32x32b,auto,16x32bx2,32x32b_splitn}
row identity, col reverse, N=128, variant={32x32b,auto,16x32bx2,32x32b_splitn}
```

M128 direct row-permuted controls passed:

```text
M=128, N=32, row reverse,  col identity
M=128, N=32, row rotate1,  col identity
M=128, N=32, row even_odd, col identity
M=128, N=128, row rotate1, col even_odd
M=128, N=32, row reverse,  col reverse
```

Representative command:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python3 /tmp/tmem_laneaa_ldred_fz0012_child.py \
  --mode default --layout m128 --m 128 --n 32 --row rotate1 --col identity \
  --op min
```

Result:

```text
PASS
```

## Existing Bucket Mappings

The attempted two-CTA M64-style rows did not give useful `FZ-0012` evidence.
They all mapped to the existing high-CGA CTA-count gate:

```text
twocta_m64_identity_identity_n32 -> FZ-20260421-0010
twocta_m64_reverse_identity_n32  -> FZ-20260421-0010
twocta_m64_rotate1_identity_n32  -> FZ-20260421-0010
twocta_m64_identity_reverse_n32  -> FZ-20260421-0010
twocta_m64_rotate1_even_odd_n128 -> FZ-20260421-0010
```

Diagnostic:

```text
Layout has 1 CTAs per CGA, but the context requires 2 CTAs per CGA.
```

## Conclusion

`FZ-20260421-0012` is a broad M64 row-basis lowering gap, not a narrow
`row_reverse_n32` test artifact.  The minimal direct reproducer is an M64
`64x32` `load_min` with `row=rotate1`, `col=identity`.  Column permutations are
not the deciding factor: row-identity column-permuted controls pass, while
non-identity row bases fail across every probed column basis and `N`.

No backend repair was attempted.
