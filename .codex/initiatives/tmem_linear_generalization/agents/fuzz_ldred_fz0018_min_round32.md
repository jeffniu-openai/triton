# Round 32: FZ-20260421-0018 ld.red Minimization

Date: 2026-04-21
Branch: `codex/tmem`

## Scope

Discovery-only follow-up for `FZ-20260421-0018`, the Round 31 candidate where
direct `M128xN512` f32 hardware `ld.red` reaches `ptxas-blackwell` register
allocation failure instead of a clean resource diagnostic. No backend or
checked-in test source was edited.

This lane swept:

- `M={96,128,160,256}` where shape construction was feasible;
- `N={256,384,512}`;
- `num_warps={4,8}`;
- direct roots, indexed roots, parent-expanding descriptor chains, and
  same-footprint descriptor-view chains;
- `auto` and explicit `32x32b` variants;
- identity, row-reversed, column-reversed, and tile-swapped linear layouts;
- `min` and `max` reduction ops.

## Required Build

```bash
make -j8
```

Result: no-op rebuild in
`/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`.

## Temporary Runtime Probe

Artifacts:

- Driver: `/tmp/tmem_ldred_fz0018_min_round32.py`
- Child harness: `/tmp/tmem_ldred_fz0018_min_round32/ldred_child.py`
- Summary: `/tmp/tmem_ldred_fz0018_min_round32/summary.json`
- Per-case logs:
  `/tmp/tmem_ldred_fz0018_min_round32/<case>.stdout.txt` and `.stderr.txt`

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_fz0018_min_round32.py
```

Each row ran in a fresh subprocess so ptxas failures did not contaminate later
rows. Passing rows executed the kernel, checked replay output and per-row torch
reductions, and counted `.ld.red` instructions in PTX/LLIR.

Counts:

```text
60 total
16 FZ-20260421-0018
12 pass
6 clean shared-memory resource boundaries
6 clean tensor-memory resource boundaries
20 clean power-of-two shape boundaries
```

## FZ-0018 Rows

All FZ-0018 rows fail with:

```text
ptxas-blackwell fatal: (C7600) Register allocation failed with register count of '255'.
```

Rows:

```text
direct identity M128 N512 warps4 auto min
direct identity M256 N256 warps4 auto min
direct identity M128 N512 warps4 32x32b min
direct row_reverse M128 N512 warps4 auto min
direct row_reverse M128 N512 warps4 32x32b min
direct tile_swap M128 N512 warps4 auto min
direct tile_swap M128 N512 warps4 32x32b min
direct legacy256 M256 N256 warps4 auto min
direct identity M128 N512 warps4 auto max
same_chain identity M128 N512 warps4 auto max
same_chain identity M128 N512 warps4 auto min
same_chain identity M128 N512 warps4 32x32b min
same_chain row_reverse M128 N512 warps4 auto min
same_chain row_reverse M128 N512 warps4 32x32b min
same_chain tile_swap M128 N512 warps4 auto min
same_chain tile_swap M128 N512 warps4 32x32b min
```

## Passing Controls

```text
direct identity M128 N256 warps4 auto min              PASS, 4 PTX .ld.red ops
indexed identity M128 N256 warps4 auto min             PASS, 4 PTX .ld.red ops
chain identity M128 N256 warps4 auto min               PASS, 4 PTX .ld.red ops
direct col_reverse M128 N512 warps4 auto min           PASS, 8 PTX .ld.red ops
direct col_reverse M128 N512 warps4 32x32b min         PASS, 8 PTX .ld.red ops
same_chain col_reverse M128 N512 warps4 auto min       PASS, 8 PTX .ld.red ops
same_chain col_reverse M128 N512 warps4 32x32b min     PASS, 8 PTX .ld.red ops
direct m64 M64 N256 warps4 auto min                    PASS, 2 PTX .ld.red ops
direct m64 M64 N512 warps4 auto min                    PASS, 4 PTX .ld.red ops
```

The `M128xN256` rows with `num_warps=8` also executed correctly, but emitted
no hardware `.ld.red` opcodes in this temporary harness, so they are not used
as hardware-reduction controls for FZ-0018.

## Clean Boundaries

Power-of-two frontend/Gluon shape boundaries:

- all `M=96` direct rows;
- all `M=160` direct rows;
- all `N=384` direct/indexed/chain rows;
- `M256xN384` direct rows.

Clean tensor-memory capacity boundaries:

- direct `M256xN512,w4`;
- indexed `M128xN512,w4`;
- parent-expanding chain `M128xN512,w4`;
- direct legacy256 `M256xN512,w4`;
- indexed/parent-expanding-chain `M128xN512,w4,max`.

Clean shared-memory boundaries:

- direct `M128xN512,w8`;
- direct `M256xN256,w8`;
- direct `M256xN512,w8`;
- indexed/parent-expanding-chain/same-footprint-chain `M128xN512,w8`.

## Classification

`FZ-20260421-0018` is not direct-only: it reproduces for direct descriptors
and for same-footprint descriptor-view chains over `[M,N]`. It is, however,
blocked by tensor-memory capacity for parent-expanding indexed/chain variants
that allocate `[2,M,N]`.

It is not a generic high-`N` or high-footprint impossibility:

- `M64xN512` hardware `ld.red` executes correctly;
- `M128xN512` with column-reversed layout executes correctly with eight
  hardware `.ld.red` instructions;
- `M128xN256` direct/indexed/chain hardware `ld.red` executes correctly.

The failing footprint generalizes beyond the original `M128xN512` row:
`M256xN256` direct/legacy-equivalent rows also reach the same ptxas register
allocation failure at `num_warps=4`. The same shapes with `num_warps=8`
produce clean shared-memory OOR diagnostics. This makes the likely owner a
4-warp hardware-`ld.red` resource-policy/planning gap for large direct or
same-footprint descriptor-view reduction tiles, not an ISA impossibility and
not a descriptor-view-chain-only problem.

Backend repair remains intentionally deferred while the fuzzing campaign is
still in discovery mode.
