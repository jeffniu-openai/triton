# Round 7 Lane C: copy and copy-readback generator gaps

- Time: 2026-04-21 09:13 UTC
- Branch: `codex/tmem`
- Mode: discovery-only structural fuzzing. No backend/compiler repairs were
  attempted.
- Temporary probe: `/tmp/tmem_copy_readback_round7_probe.py`

## Scope

This lane focused on `tcgen05.copy` descriptor-view chains plus immediate
readback behavior:

- no-scales `warpx2::01_23` descriptor-view chains with immediate `tmem.load`
  readback;
- no-scales `warpx2::02_13` two-CTA descriptor-view clean unsupported
  diagnostics;
- packed/subword `warpx2` clean diagnostics;
- copy-adjacent `ld.red` readback opcode/runtime rows;
- scales `warpx4` direct and descriptor-view readback rows;
- two-CTA TMEM layout in a larger-CGA context.

## Probe Cases

The final `/tmp` probe is a deterministic Python launcher over the checked-in
runtime-matrix kernels. It keeps the opcode/readback assertions in the owning
tests and records the case id plus seed at launch:

- `copy-r7c-noscale-warpx2-01-23-twocta-view-readback`, seed `0x7c01`:
  two-CTA no-scales `warpx2::01_23` `subslice`, `index`, and
  `slice(index)` view readback rows.
- `copy-r7c-noscale-warpx2-02-13-twocta-clean`, seed `0x7c02`:
  two-CTA no-scales `warpx2::02_13` descriptor-view clean unsupported rows.
- `copy-r7c-noscale-warpx2-packed-subword-clean`, seed `0x7c03`:
  packed/subword `warpx2` clean unsupported rows.
- `copy-r7c-noscale-copy-ldred-readback`, seed `0x7c04`:
  descriptor-chain `ld.red` `N=64` readback row that asserts hardware
  `.ld.red.` and runtime reduction correctness in the runtime matrix.
- `copy-r7c-scales-warpx4-view-readback`, seed `0x7c05`:
  scales `warpx4` direct, two-CTA direct, descriptor-view, and scaled-MMA
  copy-path rows selected by the matrix selector.
- `copy-r7c-twocta-layout-in-4cta-clean`, seed `0x7c06`:
  two-CTA layout in a four-CTA context clean diagnostic.

## Classification

No stable new backend/compiler failure was found in this lane.

Positive support stayed positive for:

- two-CTA no-scales `warpx2::01_23` descriptor-view copy plus immediate
  readback;
- scales `warpx4` direct and descriptor-view copy plus readback;
- descriptor-chain `ld.red` `N=64` readback using the hardware
  `tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32` path.

Clean diagnostics stayed clean for:

- two-CTA no-scales `warpx2::02_13` descriptor-view rows;
- packed/subword `warpx2` storage-model boundaries;
- two-CTA TMEM layout used in a four-CTA context.

The first custom `/tmp` harness draft also tried to synthesize new combined
`warpx2` copy/readback kernels. Those rows were not cataloged as backend
findings because they failed at harness construction: helper calls inside JIT
bodies, hand-copied layout shape mistakes, and overly synthetic view layouts
that diverged from the runtime-matrix kernels. The final probe intentionally
reused established runtime-matrix kernels for classification.

## Validation

- Required build:
  `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8`
  reported `ninja: no work to do`.
- Py-compile:
  `PYTHONPATH=.:./python python -m py_compile /tmp/tmem_copy_readback_round7_probe.py`.
- `/tmp` launcher:
  `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon python /tmp/tmem_copy_readback_round7_probe.py`
  passed all six deterministic launch cases:
  `12 passed`, `14 passed`, `14 passed`, `1 passed`, `24 passed`, and
  `1 passed`.
- Four-GPU selector sweep:
  `CUDA_VISIBLE_DEVICES=<0..3> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<0..3> PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short --splits 4 --group <1..4> python/test/gluon/test_tmem_runtime_matrix.py -k 'cp_no_scales_warpx2 or cp_scales_warpx4 or cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error or ld_red_descriptor_chain_n_sweep_explicit'`
  passed as `28 passed, 1587 deselected` on each of the four groups.

## Follow-Ups

- Keep this lane out of the central `FZ-*` failure list because it found no
  stable new crash, false unsupported diagnostic, opcode mismatch, or
  miscompile.
- A future generator-backed copy lane should build a repo-local
  family-specific copy case adapter instead of constructing ad hoc combined
  kernels in `/tmp`; the matrix kernels are currently the reliable executable
  source for these descriptor-view readback contracts.
