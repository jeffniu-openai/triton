# Round 12 Lane V: copy descriptor/addressing fuzzing

Date: 2026-04-21
Lane: V
Scope: discovery-only fuzzing of `tcgen05.copy` descriptor and addressing
behavior. No backend fixes attempted and no commits made.

## Build gate

Command:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Checked-in runtime matrix slice

I first collected the existing copy/scales surface:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python \
  pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2 or cp_scales or cp_128x128 or cp_no_scales_linear_subword or cp_no_scales_twocta_layout_in_4cta_context'
```

Result:

```text
128/1615 tests collected (1487 deselected)
```

Then I ran the slice split across all four GPUs with stable per-GPU caches:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2 or cp_scales or cp_128x128 or cp_no_scales_linear_subword or cp_no_scales_twocta_layout_in_4cta_context'

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2 or cp_scales or cp_128x128 or cp_no_scales_linear_subword or cp_no_scales_twocta_layout_in_4cta_context'

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2 or cp_scales or cp_128x128 or cp_no_scales_linear_subword or cp_no_scales_twocta_layout_in_4cta_context'

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2 or cp_scales or cp_128x128 or cp_no_scales_linear_subword or cp_no_scales_twocta_layout_in_4cta_context'
```

Results:

```text
group 1: 32 passed, 1583 deselected in 4.54s
group 2: 32 passed, 1583 deselected in 8.56s
group 3: 32 passed, 1583 deselected in 6.53s
group 4: 32 passed, 1583 deselected in 5.57s
total:   128 passed
```

Coverage from this slice:

- no-scales copy 1CTA/2CTA;
- warpx2 `01_23` and `02_13`;
- indexed, subslice, and slice-index descriptor views;
- dense-shared rematerialization;
- linear no-scales subword f16, bf16, i16, and i8 positive rows;
- warpx2 subword f16, bf16, i16, and i8 clean-boundary rows;
- `128x128b` and `128x256b` no-scales families;
- scales `warpx4` direct copy, layout rematerialization, source subslice, and
  descriptor-view clean boundary;
- scaled-MMA copy paths that exercise scales copy setup; and
- the checked-in 4-CTA high-CGA no-scales gate row.

## Fresh-process opcode and high-CGA probe

Temporary probe:

```text
/tmp/tmem_copy_descriptor_round12_probe.py
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_copy_descriptor_round12_probe.py | tee /tmp/tmem_copy_descriptor_round12_probe.log
```

The probe ran every row in a fresh child process. Summary:

```text
pass: 6
FZ-20260421-0010: 3
unclassified harness capture limitation: 1
```

The unclassified row was the known scales descriptor-view clean boundary; the
temporary worker captured only the Python traceback tail and not pytest's
`capfd` diagnostic stream. The exact checked-in pytest row below passed and
keeps that row classified as a clean boundary.

Representative pass rows and opcode checks:

```text
noscales_1cta_01_23_f32:
  PTX  tcgen05.cp.cta_group::1.warpx2::01_23.64x128b
  LLIR tcgen05.cp.cta_group::1.warpx2::01_23.64x128b

noscales_1cta_02_13_i32_indexed:
  PTX  tcgen05.cp.cta_group::1.warpx2::02_13.64x128b
  LLIR tcgen05.cp.cta_group::1.warpx2::02_13.64x128b
  TTGIR contains ttg.memdesc_index

noscales_2cta_01_23_slice_index:
  PTX  tcgen05.cp.cta_group::2.warpx2::01_23.64x128b
  LLIR tcgen05.cp.cta_group::2.warpx2::01_23.64x128b
  TTGIR contains ttg.memdesc_subslice and ttg.memdesc_index

noscales_linear_subword_i8_128x128:
  PTX/LLIR both emit tcgen05.cp.cta_group::1.128x256b x4

scales_warpx4_1cta:
  PTX/LLIR both emit tcgen05.cp.cta_group::1.warpx4.32x128b x2

scales_warpx4_2cta:
  PTX/LLIR both emit tcgen05.cp.cta_group::1.warpx4.32x128b x2
```

The scales 2CTA direct-copy opcode remains `cta_group::1`; that matches the
checked-in expectation for `TensorMemoryScalesLayout(cga_layout=[[1, 0]])`
direct copies and did not present a new root-cause signal in this lane.

High-CGA copy rows:

```text
high_cga_noscales_2cta_4:
  FZ-20260421-0010
  Layout has 2 CTAs per CGA, but the context requires 4 CTAs per CGA.

high_cga_noscales_2cta_8:
  FZ-20260421-0010
  Layout has 2 CTAs per CGA, but the context requires 8 CTAs per CGA.

high_cga_noscales_2cta_16:
  FZ-20260421-0010
  Layout has 2 CTAs per CGA, but the context requires 16 CTAs per CGA.
```

Representative diagnostic:

```text
triton.compiler.errors.CompilationError: at 3:11:
def tmem_copy_no_scales_twocta_kernel(in_ptr, out_ptr, layout: ttgl.constexpr, cga_layout: ttgl.constexpr,
                                      M: ttgl.constexpr, N: ttgl.constexpr, swizzle: ttgl.constexpr):
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout)
           ^
Layout has 2 CTAs per CGA, but the context requires 16 CTAs per CGA.
```

## Focused clean-boundary confirmations

Scales descriptor-view clean boundary:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_tmem_descriptor_view_reports_clean_unsupported'
```

Result:

```text
1 passed in 3.43s
```

Representative asserted diagnostic fragments:

```text
maps to tcgen05.copy.warpx4.32x128b
could not synthesize a compatible shared-memory descriptor plan for it
requires non-broadcast TMEM row bases to stay in ascending physical row order
destination-row order requirement
copy atom writes the full 32-row destination footprint
first differing physical-query field is physical layout for active view shape 128x32
PassManager::run failed not present
Assertion not present
```

Warpx2 subword clean-boundary spot check:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_subword_dtypes_report_clean_error[single_01_23-1-tmem_layout0-f16-torch_dtype0]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_subword_dtypes_report_clean_error[single_01_23-1-tmem_layout1-bf16-torch_dtype1]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_subword_dtypes_report_clean_error[single_01_23-1-tmem_layout2-i16-torch_dtype2]' \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_warpx2_subword_dtypes_report_clean_error[single_01_23-1-tmem_layout3-i8-torch_dtype3]'
```

Result:

```text
4 passed in 3.57s
```

Representative asserted diagnostic fragments:

```text
could not synthesize a compatible shared-memory descriptor plan
requires enough TMEM column bases to cover the copy instruction width
destination-column footprint requirement
packed-lane source/destination storage model
descriptor footprint coverage alone is not a correctness proof
cleanly unsupported
PassManager::run failed not present
Assertion not present
```

Checked-in 4-CTA no-scales gate:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -s --tb=short \
  'python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error'
```

Result:

```text
1 passed in 3.30s
```

This confirms the high-CGA behavior is currently checked in as a clean error,
but Lane R and this lane both classify the root as `FZ-20260421-0010`: a likely
over-strict equality gate between layout CTA count and kernel CGA size.

## Classification

No new independent `FZ-*` bucket was found in this lane.

Pass:

- no-scales 1CTA `warpx2::01_23` and `warpx2::02_13`;
- no-scales 1CTA indexed/subslice/slice-index descriptor chains;
- no-scales 2CTA `warpx2::01_23` direct and descriptor-view chains;
- linear subword no-scales `128x256b` copies for f16, bf16, i16, and i8;
- exact-width subword `128x128b` copies for f16 and i8;
- scales `warpx4` direct 1CTA/2CTA copies;
- scales shared-layout rematerialization and shared-source subslice rows; and
- scaled-MMA copy setup rows selected by the copy/scales slice.

Clean boundary:

- scales TMEM descriptor-view copy where the destination view violates the
  row-order/footprint requirements of `tcgen05.copy.warpx4.32x128b`;
- no-scales `warpx2` subword packed-lane candidates where the current layout
  model cannot represent the packed source/destination storage proof;
- no-scales 2CTA `warpx2::02_13` rows that require a source-column schedule not
  representable under `cta_group::2`; and
- row-permuted or exotic no-scales destination layouts that fail the copy atom's
  row/column ordering requirements.

`FZ-20260421-0010`:

- no-scales 2CTA copy in 4/8/16-CTA launch contexts still trips
  `Layout has 2 CTAs per CGA, but the context requires Y CTAs per CGA`;
- this broadens the existing high-CGA gate evidence specifically over
  `tcgen05.copy` no-scales descriptors.

Existing copy/scales resource/API buckets:

- The selected checked-in slice did not surface a new scales resource/API
  failure. Earlier lanes already recorded `make_scales_descriptor` shape
  divisibility assertions for some `num_ctas > 2` scaled-MMA helper paths; I did
  not promote those here because this lane's direct copy rows did not reproduce
  a new independent root.

## Recommended promotion candidates

1. Promote a 4/8/16-CTA no-scales copy matrix for `FZ-20260421-0010`, covering
   both 1CTA and 2CTA `TensorMemoryLinearLayout` local copy instructions inside
   larger CGA launches. The checked-in test currently covers only one 4-CTA
   2CTA row.
2. Add opcode-consistency assertions for representative descriptor-view copy
   positives:
   `warpx2::01_23` direct, `warpx2::02_13` indexed, and 2CTA `warpx2::01_23`
   slice-index. These are compact guards for PTX/LLIR agreement and for
   preserving `ttg.memdesc_index` / `ttg.memdesc_subslice` through lowering.
3. Keep the scales descriptor-view row as a clean-negative test until the copy
   planner learns an ISA-valid destination-row/source-message schedule for that
   view. Its current diagnostic is specific and avoids a late pass-manager
   failure.
4. Keep warpx2 subword rows as clean-negative tests unless a packed-lane
   source/destination storage model is introduced. The current diagnostics are
   hardware-footprint based and do not look like verifier overreach.

Backend repair remains deferred under the campaign rule: continue discovery and
cataloging until fuzzing stops finding new bugs or the user explicitly pivots to
fixes.
