# Round 31: ld.red Descriptor/Layout Extremes

Date: 2026-04-21
Branch: `codex/tmem`

## Scope

Focused lane for `ttng.tmem_load` reductions and `tcgen05.ld.red` lowering.
The temporary runtime probe stressed M/N extremes, row/column basis
permutations, explicit instruction variants, indexed descriptor roots,
descriptor-view chains, 1CTA vs invalid 2CTA ownership boundaries, clean
resource boundaries, and opcode consistency. No backend or checked-in test
source was edited.

## Required Build

```bash
make -j8
```

Result: no-op rebuild in `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`.

## Temporary Runtime Probe

Artifacts:

- Driver: `/tmp/tmem_ldred_extremes_round31.py`
- Child harness: `/tmp/tmem_ldred_extremes_round31/ldred_child.py`
- Summary: `/tmp/tmem_ldred_extremes_round31/summary.json`
- Per-case stdout/stderr: `/tmp/tmem_ldred_extremes_round31/<case>.stdout.txt` and `.stderr.txt`

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  python /tmp/tmem_ldred_extremes_round31.py
```

The driver launches one subprocess per case so compiler assertions or ptxas
failures do not poison the rest of the matrix. Passing rows execute the kernel,
compare full TMEM replay output and row reductions against torch, and verify
that PTX and LLIR agree on `.ld.red` opcode use when hardware reduction is
expected. Software-reduction rows assert that no `.ld.red` opcode is emitted.

Counts:

```text
23 total
15 pass
8 classified failures/boundaries

FZ-20260421-0010: 1
FZ-20260421-0012: 3
FZ-20260421-0018: 2
clean-resource-boundary: 2
pass: 15
```

## Case Matrix

```text
direct_m128_n16_identity                         PASS
direct_m128_n32_col_reverse_max_abs_nan          PASS
direct_m128_n64_row_rotate_col_evenodd           PASS
direct_m128_n512_identity                        FZ-20260421-0018
direct_m128_n512_identity_w8                     clean-resource-boundary
direct_m128_n512_identity_32x32b_splitn          FZ-20260421-0018
direct_m64_n512_identity_splitn                  PASS
direct_m256_n32_reverse_rows                     PASS
direct_m256_n256_identity_resource               clean-resource-boundary
direct_m64_n16_identity_splitn                   PASS
direct_m64_n32_row_reverse                       FZ-20260421-0012
direct_m64_n64_col_reverse                       PASS
direct_m64_n256_row_rotate_col_evenodd           FZ-20260421-0012
direct_m128_n64_mixed_software                   PASS
direct_m128_n64_twocta_block                     FZ-20260421-0010
indexed_m128_n16_identity_none                   PASS
indexed_m128_n32_slice_both                      PASS
indexed_m128_n64_col_identity_chain              PASS
indexed_m128_n128_row_identity_chain             PASS
indexed_m128_n128_row_permute_chain              PASS
indexed_m128_n256_tilelike_col_reverse           PASS
indexed_m64_n32_parent_m64                       PASS
indexed_m64_n128_row_rotate                      FZ-20260421-0012
```

Representative passing opcode checks:

- `direct_m128_n16_identity`: PTX/LLIR both emit
  `tcgen05.ld.red.sync.aligned.32x32b.x16.min.f32`.
- `direct_m64_n512_identity_splitn`: PTX/LLIR both emit four
  `tcgen05.ld.red.sync.aligned.16x32bx2.x64.min.f32` instructions and match
  torch.
- `indexed_m128_n128_row_permute_chain`: PTX/LLIR both emit
  `tcgen05.ld.red.sync.aligned.32x32b.x128.min.f32`; TTGIR contains
  `ttg.memdesc_index` and `ttg.memdesc_reshape`, so this is a live
  descriptor-view-chain control.

## Classifications

### Existing: FZ-20260421-0012

The three M64 row/permuted failures reproduce the known M64 destination-layout
planner gap:

```text
Failed to lower TMEM load/store: unsupported dst layout
```

Rows:

- `direct_m64_n32_row_reverse`
- `direct_m64_n256_row_rotate_col_evenodd`
- `indexed_m64_n128_row_rotate`

Nearby controls passed: `direct_m64_n16_identity_splitn`,
`direct_m64_n64_col_reverse`, `direct_m64_n512_identity_splitn`, and
`indexed_m64_n32_parent_m64`.

### Existing: FZ-20260421-0010

`direct_m128_n64_twocta_block` intentionally feeds a 2CTA TMEM layout to a
1CTA kernel context and receives the existing clean ownership diagnostic:

```text
Layout has 2 CTAs per CGA, but the context requires 1 CTAs per CGA.
```

This is not a new backend failure.

### Clean Resource Boundaries

Two rows produced clean resource diagnostics:

- `direct_m128_n512_identity_w8`:
  `out of resource: shared memory, Required: 262148, Hardware limit: 232448`
- `direct_m256_n256_identity_resource`:
  `out of resource: shared memory, Required: 262148, Hardware limit: 232448`

### New Candidate: FZ-20260421-0018

`M=128, N=512` direct f32 `ld.red` with the 4-warp layout reaches ptxas
register allocation failure instead of an earlier clean compiler/resource
diagnostic:

```text
ptxas-blackwell fatal: (C7600) Register allocation failed with register count of '255'.
```

Rows:

- `direct_m128_n512_identity`
- `direct_m128_n512_identity_32x32b_splitn`

Why this is independent:

- It is not `FZ-0012`: no `unsupported dst layout`; the kernel reaches ptxas.
- It is not `FZ-0010`: no CTA ownership mismatch.
- It is not a generic N=512 impossibility: `direct_m64_n512_identity_splitn`
  executes correctly with hardware `ld.red`.
- It is a boundary-policy gap: the same `M128x512` shape with `num_warps=8`
  rejects cleanly as shared-memory OOR.

Current classification: independent new candidate for late ptxas failure on a
large direct `ld.red` tile. Backend repair is intentionally deferred while the
fuzzing campaign is still finding/classifying failures.

## Checked-In Guardrail

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python \
  pytest -q -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and not reports and not resource'
```

Results:

```text
group 1: 6 failed, 55 passed, 1554 deselected
group 2: 61 passed, 1554 deselected
group 3: 61 passed, 1554 deselected
group 4: 60 passed, 1555 deselected
aggregate: 237 passed, 6 failed
```

The six checked-in failures are the already-known `FZ-20260421-0012` M64
row/column-permuted destination-layout planner rows. No changed checked-in
failure mode was observed.

## Next Fuzzing Work

- Expand `FZ-0018` with compiler-only minimizers that preserve `.ld.red`
  generation while sweeping `M=128`, high `N`, `num_warps`, and explicit
  variants.
- Add adjacent indexed-root and descriptor-chain high-N rows to determine
  whether the late ptxas failure follows direct layout only or general
  reduction tile size.
- Keep `FZ-0012` separate from high-N resource-boundary behavior; both present
  different failure signatures.
