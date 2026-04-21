# TMEM Structural Fuzzing Round 7 Lane D: Generic Pass / Analysis Interactions

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Scope: generic-pass and analysis interaction fuzz after the scaled-MMAv5
  and `ld.red` promotions. Focused on TMEM memdesc values through nested
  `scf.if`, `scf.for`, tuple-like returns, helper chains, layout-conversion
  pressure, and interactions with non-TMEM tensors.
- Temporary harness: `/tmp/tmem_generic_analysis_round7.py`

## Commands

Required rebuild before tests:

```bash
CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8
```

Result: `ninja: no work to do`.

Collection:

```bash
PYTHONPATH=.:./python python -m py_compile /tmp/tmem_generic_analysis_round7.py
PYTHONPATH=.:./python pytest --collect-only -q /tmp/tmem_generic_analysis_round7.py
```

Result: `10 tests collected`.

Four-GPU shard sweep:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r7d-gpu0 pytest -q -s --tb=short --splits 4 --group 1 /tmp/tmem_generic_analysis_round7.py 2>&1 | tee /tmp/tmem_generic_analysis_round7_g1.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r7d-gpu1 pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_generic_analysis_round7.py 2>&1 | tee /tmp/tmem_generic_analysis_round7_g2.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r7d-gpu2 pytest -q -s --tb=short --splits 4 --group 3 /tmp/tmem_generic_analysis_round7.py 2>&1 | tee /tmp/tmem_generic_analysis_round7_g3.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r7d-gpu3 pytest -q -s --tb=short --splits 4 --group 4 /tmp/tmem_generic_analysis_round7.py 2>&1 | tee /tmp/tmem_generic_analysis_round7_g4.log
```

Results:

- group 1: `3 failed, 7 deselected`;
- group 2: `3 failed, 7 deselected`;
- group 3: `3 failed, 7 deselected`;
- group 4: `1 failed, 9 deselected`.

Fresh representative confirmations:

```bash
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r7d-confirm-tuple pytest -q -s --tb=short '/tmp/tmem_generic_analysis_round7.py::test_generic_analysis_round7[r7-tuple-branch-chain0-selector1]' 2>&1 | tee /tmp/tmem_generic_analysis_round7_confirm_tuple.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r7d-confirm-loop pytest -q -s --tb=short '/tmp/tmem_generic_analysis_round7.py::test_generic_analysis_round7[r7-loop-carried-with-tensor-chain0]' 2>&1 | tee /tmp/tmem_generic_analysis_round7_confirm_loop.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r7d-confirm-layout pytest -q -s --tb=short '/tmp/tmem_generic_analysis_round7.py::test_generic_analysis_round7[r7-layout-convert-nontmem-chain0-16x128b]' 2>&1 | tee /tmp/tmem_generic_analysis_round7_confirm_layout.log
PYTHONPATH=.:./python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r7d-control pytest -q -s --tb=short 'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]' 2>&1 | tee /tmp/tmem_generic_analysis_round7_control_ldst.log
```

Results:

- tuple branch confirmation: `1 failed` with the R5-C
  `GluonResolveAutoEncodingsPass` / `tt.make_range` auto-layout diagnostic;
- loop-carried plus tensor confirmation: `1 failed` with the same R5-C
  diagnostic;
- layout-conversion plus non-TMEM tensor confirmation: `1 failed` with
  `8063 / 8192` mismatched elements after successful compile and launch;
- known structural-fuzzer ld/st control: `1 passed`.

## Pass/Fail Matrix

| Case | Result | Classification |
| --- | --- | --- |
| `r7-nested-if-helper-chain0-selector0` | fail | R5-C auto-layout crash in `GluonResolveAutoEncodingsPass`; nested memdesc `scf.if` result through helper chain. |
| `r7-nested-if-helper-chain1-selector5` | fail | R5-C auto-layout crash; chain1 also fails when nested helper-selected memdesc reaches auto-layout. |
| `r7-tuple-branch-chain0-selector1` | fail | R5-C auto-layout crash; `scf.if` yields `(memdesc, tensor)` tuple-like result. |
| `r7-tuple-branch-chain1-selector0` | fail | R5-C auto-layout crash; tuple-like result is not chain0-specific. |
| `r7-helper-tuple-chain-chain0` | fail | R5-C auto-layout crash; helper chain returns `(memdesc, tensor)` after dynamic base selection. |
| `r7-layout-convert-nontmem-chain0-16x128b` | fail | FZ-20260421-0002-style runtime miscompile; successful compile, `8063 / 8192` mismatched. |
| `r7-loop-carried-with-tensor-chain0` | fail | R5-C auto-layout crash; `scf.for` iter args carry both memdesc and non-TMEM tensor. |
| `r7-loop-carried-with-tensor-chain2` | fail | R5-C auto-layout crash; slice/slice identity chain does not avoid the loop-carried failure. |
| `r7-multi-live-nontmem-chain0` | fail | R5-C auto-layout crash; multiple sibling views plus non-TMEM tensor side input. |
| `r7-direct-control-chain0` | fail | Harness-level R5-C auto-layout crash from helper-returned non-TMEM tensor value; not treated as a direct TMEM backend failure. |

## Findings

### R7D-001: R5-C auto-layout crash extends to tuple-like and non-TMEM tensor interactions

- Extends: R5-C generic-pass memdesc-valued control flow.
- Failure class: `compiler_crash`.
- Likely owner surface: `GluonResolveAutoEncodingsPass` / auto-layout
  inference when memdesc SSA values and non-TMEM tensor values are carried
  together through generic control flow.
- Shape: parent `[2,128,64]`, selected view `[128,64]`, dtype `f32`.
- View chains: chain0 row-interleaving identity, chain1 double-transpose
  identity, and chain2 slice/slice identity.
- Observed: compilation stops before runtime with
  `'tt.make_range' op Failed to infer return type` while the IR contains
  memdesc-valued `scf.if` or `scf.for` results. Tuple-like `(memdesc, tensor)`
  returns and helper chains reproduce the same owner surface.
- Relationship to R5-C: same pass, same diagnostic family, and same
  resource-valid TMEM descriptor-view control-flow shape. This round broadens
  the reproducer surface but does not justify a new FZ id while the existing
  R5-C sentinel remains red.

### R7D-002: FZ-0002 miscompile survives additional non-TMEM layout-conversion pressure

- Extends: `FZ-20260421-0002`.
- Failure class: `miscompile`.
- Shape: parent `[2,128,64]`, selected view `[128,64]`, dtype `f32`.
- View chain: chain0 helper-equivalent
  `reshape((64,2,64)).permute([1,0,2]).reshape((128,64))`.
- Additional pressure: load a separate non-TMEM tensor, force layout
  conversions between `32x32b` and `16x128b`, then add it to the TMEM load.
- Observed: successful compile and launch but fresh runtime comparison reports
  `8063 / 8192` mismatched elements, matching the existing FZ-0002 mismatch
  scale. This is not the R5-C auto-layout crash and not the R5-C loop-carried
  compiler failure.

## Classification Notes

- No evidence of the R5-C generic auto-layout crash overlapping the
  `FZ-20260421-0008` `ld.red` optimizer crash; this lane did not exercise
  `ld.red` and no `TritonNvidiaGPUOptimizeTMemLayoutsPass` dimension mismatch
  appeared.
- No new dynamic `memdesc_index` illegal-lowering row was found; runtime
  `parent.index(tt.load(...))` remains covered by `FZ-20260421-0001`.
- The only runtime miscompile in this lane is best classified under
  `FZ-20260421-0002`, not as a new pass failure.
- The direct-control row in this temporary harness is contaminated by
  `_fill_parent` returning a non-TMEM tensor with `#gluon.auto_encoding`; use
  the checked-in `ldst-view-identity-32x32b` pass as the direct TMEM control.

## Recommendations

- Do not promote additional strict xfails yet. The checked-in
  `generic-pass-loop-carried-memdesc-view-chain0` sentinel already covers the
  R5-C pass owner, and the checked-in FZ-0002 sentinels already cover the
  chain0 runtime miscompile family.
- Keep this report as repair-validation inventory. Once the R5-C sentinel is
  fixed, rerun the Round 7 tuple/nested/helper/non-TMEM rows to determine
  whether any hidden FZ-0002-style runtime miscompile remains underneath the
  current auto-layout crash.
