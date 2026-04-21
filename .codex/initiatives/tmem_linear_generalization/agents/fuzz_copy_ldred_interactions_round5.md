# Lane R5-B: copy / ld.red descriptor-view interactions round 5

- Time: 2026-04-21 UTC
- Lane: R5-B
- Branch/HEAD: `codex/tmem` at `729dde3d16e6`
- Scope: combined TMEM copy plus readback probes not covered by the
  standalone copy and standalone ld/st or ld.red lanes. Cases copy into
  descriptor-view or subslice parents and then force TMEM `ld/st` or
  `ld.red` readback. The lane also covers `warpx2::01_23`,
  `warpx2::02_13`, two-CTA `warpx2` descriptor views, and larger-CGA clean
  diagnostics.
- Backend repair status: no backend/compiler code changed.

## Artifacts

- Temporary probe: `/tmp/tmem_copy_ldred_interactions_r5b_probe.py`
- Split logs:
  - `/tmp/tmem_copy_ldred_interactions_r5b_g1_rerun.log`
  - `/tmp/tmem_copy_ldred_interactions_r5b_g2_rerun.log`
  - `/tmp/tmem_copy_ldred_interactions_r5b_g3_rerun.log`
  - `/tmp/tmem_copy_ldred_interactions_r5b_g4_rerun.log`
- Opcode summary:
  `/tmp/tmem_copy_ldred_interactions_r5b_opcode_summary.log`

An initial probe revision used Python helper calls inside `@gluon.jit` bodies
and over-lifted the two-CTA `warpx2` indexed layout. Those failures were
harness setup errors, not backend findings. The corrected probe passed syntax,
collection, and runtime validation.

## Commands

Required rebuild before pytest:

```bash
make -j8
```

Result: passed; Ninja reported `no work to do`.

Probe syntax and collection:

```bash
PYTHONPATH=.:./python:python/test/gluon python -m py_compile /tmp/tmem_copy_ldred_interactions_r5b_probe.py
PYTHONPATH=.:./python:python/test/gluon pytest --collect-only -q /tmp/tmem_copy_ldred_interactions_r5b_probe.py
```

Result: `9 tests collected`.

Four-GPU split sweep:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r5b-gpu0b PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short --splits 4 --group 1 /tmp/tmem_copy_ldred_interactions_r5b_probe.py 2>&1 | tee /tmp/tmem_copy_ldred_interactions_r5b_g1_rerun.log
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r5b-gpu1b PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short --splits 4 --group 2 /tmp/tmem_copy_ldred_interactions_r5b_probe.py 2>&1 | tee /tmp/tmem_copy_ldred_interactions_r5b_g2_rerun.log
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r5b-gpu2b PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short --splits 4 --group 3 /tmp/tmem_copy_ldred_interactions_r5b_probe.py 2>&1 | tee /tmp/tmem_copy_ldred_interactions_r5b_g3_rerun.log
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r5b-gpu3b PYTHONPATH=.:./python:python/test/gluon pytest -s --tb=short --splits 4 --group 4 /tmp/tmem_copy_ldred_interactions_r5b_probe.py 2>&1 | tee /tmp/tmem_copy_ldred_interactions_r5b_g4_rerun.log
```

Results:

- Group 1: `3 passed, 6 deselected`.
- Group 2: `3 passed, 6 deselected`.
- Group 3: `3 passed, 6 deselected`.
- Group 4: `9 deselected`.

Opcode summary command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r5b-summary PYTHONPATH=.:./python:python/test/gluon python - <<'PY' 2>&1 | tee /tmp/tmem_copy_ldred_interactions_r5b_opcode_summary.log
# imports /tmp/tmem_copy_ldred_interactions_r5b_probe.py, reruns the five
# positive kernels, checks outputs, and prints PTX copy/ld/st opcode lists.
PY
```

Result: passed.

## Case Matrix

| Case | Seed | Shape | View chain | Expected | Observed |
| --- | ---: | --- | --- | --- | --- |
| `r5b-copy-indexed-ldst` | `0x5B01` | parent `[2,128,128]`, view `[128,128]`, f32 | `index(1)` then scratch TMEM `store/load` | pass | pass |
| `r5b-copy-subslice-ldred` | `0x5B02` | parent `[128,256]`, view `[128,128]`, f32 | `slice(128,128,dim=1)` then `load_min` | pass with hardware `ld.red` | pass |
| `r5b-warpx2-01-23-indexed-1cta` | `0x5B03` | parent `[2,128,4]`, i32 | `index(1)` then scratch TMEM `store/load` | pass | pass |
| `r5b-warpx2-02-13-indexed-1cta` | `0x5B04` | parent `[2,128,4]`, i32 | `index(1)` then scratch TMEM `store/load` | pass | pass |
| `r5b-warpx2-01-23-indexed-2cta` | `0x5B05` | parent `[2,256,4]`, i32 | `index(1)` then scratch TMEM `store/load` | pass | pass |
| `r5b-twocta-parent-in-4cta-context` | `0x5B06` | parent `[2,256,128]`, f32 | `index(1)` copy in `num_ctas=4` | clean diagnostic | pass |
| `r5b-twocta-parent-in-8cta-context` | `0x5B07` | parent `[2,256,128]`, f32 | `index(1)` copy in `num_ctas=8` | clean diagnostic | pass |
| `r5b-twocta-parent-in-16cta-context` | `0x5B08` | parent `[2,256,128]`, f32 | `index(1)` copy in `num_ctas=16` | clean diagnostic | pass |
| `r5b-warpx2-02-13-indexed-2cta` | `0x5B09` | parent `[2,256,4]`, i32 | `index(1)` | clean unsupported | pass |

## Opcode Evidence

- `r5b-copy-indexed-ldst` emitted sixteen
  `tcgen05.cp.cta_group::1.128x256b`, then
  `tcgen05.ld.sync.aligned.32x32b.x128.b32`,
  `tcgen05.st.sync.aligned.32x32b.x128.b32`, and a final
  `tcgen05.ld.sync.aligned.32x32b.x128.b32`. Output matched input.
- `r5b-copy-subslice-ldred` emitted sixteen
  `tcgen05.cp.cta_group::1.128x256b` and one
  `tcgen05.ld.red.sync.aligned.32x32b.x128.min.f32`. The full tensor
  readback matched input and the reduced output matched `torch.min`.
- `r5b-warpx2-01-23-indexed-1cta` emitted
  `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b`, two
  `tcgen05.ld.sync.aligned.32x32b.x4.b32`, and one
  `tcgen05.st.sync.aligned.32x32b.x4.b32`. Output matched the existing
  `warpx2::01_23` runtime-matrix oracle.
- `r5b-warpx2-02-13-indexed-1cta` emitted
  `tcgen05.cp.cta_group::1.warpx2::02_13.64x128b`, two
  `tcgen05.ld.sync.aligned.32x32b.x4.b32`, and one
  `tcgen05.st.sync.aligned.32x32b.x4.b32`. Output matched the existing
  `warpx2::02_13` runtime-matrix oracle.
- `r5b-warpx2-01-23-indexed-2cta` emitted
  `tcgen05.cp.cta_group::2.warpx2::01_23.64x128b`, two
  `tcgen05.ld.sync.aligned.32x32b.x4.b32`, and one
  `tcgen05.st.sync.aligned.32x32b.x4.b32`. Output matched the existing
  two-CTA `warpx2::01_23` runtime-matrix oracle.

For all positive rows, PTX and LLIR opcode extraction matched for copy, load,
and store opcode lists.

## Larger-CGA Diagnostics

The corrected probe reused the existing two-CTA linear indexed-view copy kernel
inside `num_ctas=4`, `8`, and `16` contexts. Each row failed cleanly with:

`Layout has 2 CTAs per CGA, but the context requires <num_ctas> CTAs per CGA.`

No row contained `Assertion` or `PassManager::run failed`.

The two-CTA `warpx2::02_13` indexed descriptor-view row failed cleanly with
the established source-column preservation diagnostic, including
`maps to tcgen05.copy.warpx2::02_13.64x128b` and `cleanly unsupported`.

## Findings

No new backend compiler crash, false unsupported diagnostic, opcode mismatch,
or runtime miscompile was found in this lane.

The key positive boundary is that copy into a descriptor-indexed parent can be
followed by independent TMEM `ld/st` readback, and copy into a TMEM subslice
can be followed by hardware `ld.red` readback when the final view remains in a
supported packet family. This is distinct from the existing standalone
`FZ-20260421-0004` indexed `ld.red` opcode-loss bucket: the subslice copy
interaction preserved `tcgen05.ld.red`.

The clean-negative boundary remains unchanged for two-CTA `warpx2::02_13`
descriptor views and larger-CGA contexts.

## Recommended Follow-Up

- Add a checked-in positive sentinel later if the campaign wants durable
  coverage for `copy -> subslice -> ld.red`; it is a useful non-regression
  anchor because it proves descriptor-view copy does not universally erase
  hardware reduction selection.
- Continue fuzzing chained copy parents with deeper descriptor transforms,
  especially `slice(...).index(...).reshape(...).permute(...)` around the
  known `FZ-20260421-0003` and `FZ-20260421-0004` boundaries.
