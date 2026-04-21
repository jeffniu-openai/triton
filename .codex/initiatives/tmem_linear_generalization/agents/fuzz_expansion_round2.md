# TMEM Structural Fuzzing Round 2, Lane Expansion

- Date: 2026-04-21
- Branch: `codex/tmem`
- HEAD at start: `916057bc91fb3367f9b022616348741001160b13`
- Mode: discovery only; no backend/compiler repairs.
- Temporary harnesses:
  - `/tmp/tmem_expansion_round2_cf.py`
  - `/tmp/tmem_expansion_round2_ldst_ldred.py`
- Build prerequisite:
  - `CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8`
  - Result: `ninja: no work to do`.

## Sweep Commands

Control-flow/helper-view sweep:

```bash
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-r2-cf2-gpu<gpu> pytest -s --tb=short --splits 4 --group <1-4> /tmp/tmem_expansion_round2_cf.py 2>&1 | tee /tmp/tmem_expansion_round2_cf2_g<group>.log
```

Results:

- group 1: `10 failed, 5 passed, 42 deselected`
- group 2: `8 failed, 7 passed, 42 deselected`
- group 3: `9 failed, 6 passed, 42 deselected`
- group 4: `9 failed, 3 passed, 45 deselected`

ld/st and ld.red adjacency sweep:

```bash
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-r2-lane-gpu<gpu> pytest -q -s --tb=short --splits 4 --group <1-4> /tmp/tmem_expansion_round2_ldst_ldred.py 2>&1 | tee /tmp/tmem_expansion_round2_ldst_ldred_g<group>.log
```

Results:

- group 1: `56 failed, 62 passed, 354 deselected`
- group 2: `102 failed, 16 passed, 354 deselected`
- group 3: `102 failed, 16 passed, 354 deselected`
- group 4: `98 failed, 20 passed, 354 deselected`

The raw failure count is intentionally not a bug count. Many rows were broad
adjacency probes that hit already-known clean row-anchor diagnostics or harness
expectations that were too aggressive for unsupported layouts.

## Findings

### EXP-R2-001: runtime memdesc_index crash extends to chain2 and chain3

- Catalog bucket: extends `FZ-20260421-0001`.
- Failure class: `compiler_crash`.
- Family: `generic_pass`.
- Shape: parent `[2, 128, 64]`, selected view `[128, 64]`, dtype `f32`.
- Stable new adjacent parameters:
  - `test_cf_helper_returned_views[index-128-64-identity-identity-2-0-32x32b]`
  - `test_cf_helper_returned_views[index-128-64-identity-identity-3-0-32x32b]`
- Observed: `ConvertTritonGPUToLLVM` fails because runtime
  `ttg.memdesc_index` remains illegal after descriptor-view chains.
- Representative fresh repro:

```bash
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r2-confirm-cf-index-chain3 pytest -s --tb=short '/tmp/tmem_expansion_round2_cf.py::test_cf_helper_returned_views[index-128-64-identity-identity-3-0-32x32b]' 2>&1 | tee /tmp/tmem_expansion_round2_confirm_cf_index_chain3.log
```

- Fresh result: failed in `5.08s`; diagnostic starts with
  `failed to legalize operation 'ttg.memdesc_index' that was explicitly marked illegal`.

### EXP-R2-002: helper-returned chain0 control-flow miscompile broadens

- Catalog bucket: extends `FZ-20260421-0002`.
- Failure class: `miscompile`.
- Family: `generic_pass`.
- Shape: parent `[2, 128, 64]`, selected view `[128, 64]`, dtype `f32`.
- View chain:
  `reshape((64,2,64)).permute([1,0,2]).reshape((128,64))`.
- Stable new adjacent parameters:
  - false branch of dynamic `if`, selector `0`:
    `test_cf_helper_returned_views[if-128-64-identity-identity-0-0-32x32b]`
  - same case with `16x64b`:
    `test_cf_helper_returned_views[if-128-64-identity-identity-0-0-16x64b]`
  - layout-conversion pressure:
    `test_cf_layout_pressure_views[128-64-0]`
- Observed: successful compile and launch, but output mismatches reference.
- Representative fresh repros:

```bash
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r2-confirm-cf-if-false pytest -s --tb=short '/tmp/tmem_expansion_round2_cf.py::test_cf_helper_returned_views[if-128-64-identity-identity-0-0-32x32b]' 2>&1 | tee /tmp/tmem_expansion_round2_confirm_cf_if_false.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r2-confirm-cf-if-false-16x64 pytest -s --tb=short '/tmp/tmem_expansion_round2_cf.py::test_cf_helper_returned_views[if-128-64-identity-identity-0-0-16x64b]' 2>&1 | tee /tmp/tmem_expansion_round2_confirm_cf_if_false_16x64.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r2-confirm-cf-layout-pressure pytest -s --tb=short '/tmp/tmem_expansion_round2_cf.py::test_cf_layout_pressure_views[128-64-0]' 2>&1 | tee /tmp/tmem_expansion_round2_confirm_cf_layout_pressure.log
```

- Fresh results:
  - both dynamic-if variants mismatch `8064 / 8192` elements;
  - layout-pressure variant mismatches `8064 / 8192` elements.

### EXP-R2-003: ld/st col-reverse chain2 16x64b miscompiles

- Catalog bucket: extends `FZ-20260421-0003`.
- Failure class: `miscompile`.
- Family: `ldst`.
- Shape: parent `[2, 64, 32]`, selected view `[64, 32]`, dtype `f32`.
- Layout parameters: row `identity`, col `reverse`.
- View chain: chain2 double-transpose/slice descriptor chain.
- Instruction variant: `16x64b`.
- Observed: kernel stores `view.store(y + 1.0)` and reloads the base
  descriptor, but only half the base tile matches `input + 1.0`.
- Fresh repro:

```bash
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-r2-confirm-ldst-c2 pytest -s --tb=short '/tmp/tmem_expansion_round2_ldst_ldred.py::test_ldst_adjacent_row_col_permutations[64-32-identity-reverse-2-16x64b]' 2>&1 | tee /tmp/tmem_expansion_round2_confirm_ldst_chain2_colrev.log
```

- Fresh result: mismatched `1024 / 2048` elements.

### EXP-R2-004: ld.red descriptor chains lose hardware reduction in broader layout space

- Catalog bucket: extends `FZ-20260421-0004`.
- Failure class: `opcode_mismatch`.
- Family: `ldred`.
- Shapes: parent `[2, 64, N]`, selected view `[64, N]`, with `N` in
  `{32, 64, 128}`.
- Expanded parameters:
  - chain1 identity/identity min reduction still emits plain load;
  - chain2 `rotate1`/identity min reduction also emits plain load;
  - sweep observed the same pattern across min/max, NaN propagation settings,
    transpose flags, and adjacent row/col permutations when the runtime output
    remained correct.
- Representative fresh repros:

```bash
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r2-confirm-ldred-opcode-c1 pytest -s --tb=short '/tmp/tmem_expansion_round2_ldst_ldred.py::test_ldred_descriptor_chain_opcode_selection[64-32-identity-identity-1-min-False-PROPAGATE_NAN.NONE]' 2>&1 | tee /tmp/tmem_expansion_round2_confirm_ldred_chain1_opcode.log
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-r2-confirm-ldred-opcode-c2 pytest -s --tb=short '/tmp/tmem_expansion_round2_ldst_ldred.py::test_ldred_descriptor_chain_opcode_selection[64-64-rotate1-identity-2-min-False-PROPAGATE_NAN.NONE]' 2>&1 | tee /tmp/tmem_expansion_round2_confirm_ldred_chain2_rot_opcode.log
```

- Fresh results:
  - chain1 identity emits `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`;
  - chain2 rotate1 emits repeated `tcgen05.ld.sync.aligned.32x32b.x1.b32`;
  - neither contains `.ld.red.`.

### EXP-R2-005: resource-valid 2CTA ld.red emits plain ld

- Catalog bucket: extends `FZ-20260421-0004`.
- Failure class: `opcode_mismatch`.
- Family: `ldred`.
- Shape: two-CTA parent `[2, 256, 32]`, selected view `[256, 32]`,
  dtype `f32`.
- View chain: direct indexed parent.
- Observed: output is correct, but PTX contains only
  `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`, not hardware `.ld.red.`.
- Fresh repro:

```bash
PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-r2-confirm-twocta-ldred pytest -s --tb=short '/tmp/tmem_expansion_round2_ldst_ldred.py::test_twocta_ldred_resource_valid_descriptor_chains[256-32-0]' 2>&1 | tee /tmp/tmem_expansion_round2_confirm_twocta_ldred.log
```

- Fresh result: assertion fails on missing `.ld.red.` with PTX ops
  `['tcgen05.ld.sync.aligned.16x32bx2.x16.b32']`.
- Caution: this should be compared against the checked-in two-CTA direct
  higher-rank positive before repair. The failure is still useful discovery
  evidence because it reproduces in a resource-valid 2CTA ld.red harness row.

## Non-Findings and Boundaries

- Initial control-flow logs `/tmp/tmem_expansion_round2_cf_g*.log` are
  discarded: the first harness used a non-constexpr helper dispatcher that
  caused frontend return-type errors.
- Many `[64, 32]` helper-returned view rows cleanly reject with row-anchor
  diagnostics around required anchors `32,64`; they were not counted as new
  bugs here.
- Many broad ld.red row/col permutations are clean unsupported or clean-error
  rows from current descriptor-view support limits; only representative stable
  opcode mismatches are cataloged.
- 2CTA ld/st rows in this temporary harness hit layout/broadcast mismatch
  paths and were not cataloged as backend failures without a smaller oracle.

## Next Minimization Queue

- Add checked-in xfail coverage or lit minimization for runtime
  `memdesc_index` through chain2/chain3.
- Extend the existing helper/control-flow xfail coverage to include false
  branch and `16x64b` if the current checked-in cases are not considered
  enough coverage.
- Promote the chain2 col-reverse ld/st `16x64b` repro, or minimize whether
  chain2 itself or col-reverse address arithmetic is the first broken layer.
- Compare the 2CTA ld.red temporary row against the existing two-CTA direct
  higher-rank positive to separate output red-layout mismatch from opcode
  selection loss.
