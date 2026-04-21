# Round 11 Lane Q: multi-CTA/CGA TMEM structural fuzzing

Date: 2026-04-21
Lane: Q
Scope: discovery-only fuzzing of TMEM-bearing Gluon code in 4/8/16 CTA-per-CGA contexts. No backend fixes attempted.

## Build gate

Command:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Checked-in high-CGA controls

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python \
  pytest -s --tb=short \
  python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit \
  python/test/gluon/test_core.py::test_tma_mma_shared_inputs \
  -k 'ctas_per_cga2 or ctas_per_cga1'
```

Result:

```text
136 passed, 12 skipped, 74 deselected in 139.31s
```

Coverage notes:

- Includes checked-in TCGEN05 MMA multicast/commit controls at `ctas_per_cga=[2,4]` and `[4,4]`.
- Includes 16-CTA TMA + MMA shared-input controls.
- These controls prove the branch can compile and execute legal high-CGA MMA/TMA-MMA paths when all participating layouts carry full CGA layout metadata.

## Temporary structural probe

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_multicta_cga_round11_probe.py
```

The probe ran each row in a fresh subprocess. It intentionally compared 4/8/16-CTA launch contexts against nearby 1CTA/2CTA TMEM operations and descriptor-view chains:

- `ld/st` descriptor-view chain using a 1CTA `TensorMemoryLinearLayout`, launched with `num_ctas in {4,8,16}`.
- Tensor-memory-scales `tcgen05.copy` with a 1CTA layout, launched with `num_ctas in {4,8,16}`.
- Tensor-memory-scales `tcgen05.copy` with a 2CTA layout, launched with `num_ctas in {4,8,16}`.
- 2CTA indexed `ld.red` layout, launched with `num_ctas in {4,8,16}`.
- Checked-in plain MMAv5 high-CGA controls for `[2,4]` and `[4,4]`.
- Checked-in TMA + MMA shared-input high-CGA controls for `[4,4]`.

First run result summary:

```text
PASS: 4
unclassified failures: 17
```

The first run had two harness mistakes:

- scaled-MMA helper was called with wrong positional arguments, so those rows are not backend evidence;
- direct calls into `test_tma_mma_shared_inputs` passed `reps=1` instead of `[1,1,1]`, so those two rows are not backend evidence.

Corrected rerun for the high-CGA controls:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_multicta_cga_round11_probe.py
```

Corrected rerun result summary:

```text
PASS: 6
unclassified failures: 15
```

The corrected rerun was partially contaminated by concurrent edits to `python/test/gluon/test_tmem_structural_fuzzer.py`; several fresh subprocesses failed while importing that file with:

```text
NameError: name 'PLAIN_MMA_RUNTIME_INDEX_CASES' is not defined
```

Those import failures are not TMEM backend findings and should be ignored for this lane. The clean first-run subprocesses before that concurrent edit are the evidence for the CTA-count finding below.

## Finding: over-strict layout CTA-count gate blocks local 1CTA/2CTA TMEM work inside larger CGAs

Classification: new independent candidate, tentatively `FZ-20260421-0010`.

This does not match existing buckets:

- not `FZ-20260421-0001`: no illegal `memdesc_index` reaches LLVM conversion;
- not `FZ-20260421-0002` or `R5-C`: no generic-pass auto-encoding crash;
- not `FZ-20260421-0003`: no packet-order miscompile reached execution;
- not `FZ-20260421-0004`: no `ld.red` opcode fallback reached codegen;
- not `FZ-20260421-0005`/`0009`: no allocator assertion;
- not `FZ-20260421-0006`/`0008`: no clean-unsupported transpose/slice or OptimizeTMemLayouts abort.

Fresh-process repro signatures:

```text
ValueError: Layout has 1 CTAs per CGA, but the context requires 4 CTAs per CGA.
ValueError: Layout has 1 CTAs per CGA, but the context requires 8 CTAs per CGA.
ValueError: Layout has 1 CTAs per CGA, but the context requires 16 CTAs per CGA.
ValueError: Layout has 2 CTAs per CGA, but the context requires 4 CTAs per CGA.
ValueError: Layout has 2 CTAs per CGA, but the context requires 8 CTAs per CGA.
ValueError: Layout has 2 CTAs per CGA, but the context requires 16 CTAs per CGA.
```

Representative 1CTA linear-layout repro:

```text
case: ldst_chain1_1cta_instr_in_multicta_context
num_ctas: 4
error site:
  tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [2, M, N], parent_layout)
message:
  Layout has 1 CTAs per CGA, but the context requires 4 CTAs per CGA.
```

Representative 1CTA scales-layout repro:

```text
case: copy_scales_1cta_instr_in_multicta_context
num_ctas: 4
error site:
  offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, blocked))
message:
  Result has an invalid layout: #ttg.slice<...>
  Layout has 1 CTAs per CGA, but the context requires 4 CTAs per CGA.
```

Representative 2CTA scales-layout repro:

```text
case: copy_scales_2cta_layout_in_larger_cga_context
num_ctas: 4
error site:
  offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, blocked))
message:
  Result has an invalid layout: #ttg.slice<... CGALayout = [[1, 0]]>
  Layout has 2 CTAs per CGA, but the context requires 4 CTAs per CGA.
```

Representative 2CTA `ld.red` repro:

```text
case: ldred_2cta_layout_in_larger_cga_context
num_ctas: 4
error site:
  parent = allocate_tensor_memory(ttgl.float32, [2, M, N], parent_layout)
message:
  Layout has 2 CTAs per CGA, but the context requires 4 CTAs per CGA.
```

Why this looks like a real gap:

- Checked-in high-CGA controls pass for `tcgen05.mma` at 8 and 16 CTAs when the layouts are full CGA-aware `TensorMemoryLayout`/shared layouts.
- The rejected rows are not asking for a 4/8/16-CTA TCGEN05 TMEM instruction. They are local 1CTA or 2CTA TMEM operations inside larger CGA launch contexts.
- `TensorMemoryLinearLayout` has only a `two_ctas` boolean and no full `cga_layout` field, so the frontend cannot express linear TMEM layouts that participate in a 4/8/16-CTA kernel while still lowering a local 1CTA/2CTA instruction.
- `TensorMemoryScalesLayout` can carry `cga_layout`, but the current validation requires its counted CTA split to equal kernel `num_ctas`; the nearby passing MMA controls suggest the backend should distinguish kernel CGA shape from instruction-local `cta_group`.

This should be audited against the hardware rule recorded earlier: if any instruction that can be 2CTA is emitted as 2CTA, then all such instructions in the kernel must be 2CTA. That rule does not obviously imply that every TMEM layout in an 8/16-CTA kernel must have 8/16 CTA splits, especially when passing controls emit legal local TCGEN05 operations in larger CGAs.

## Non-findings / harness-only rows

Corrected high-CGA controls:

```text
plain_mma_multicast_commit_context [2,4], two_ctas=False: PASS
plain_mma_multicast_commit_context [2,4], two_ctas=True: PASS
plain_mma_multicast_commit_context [4,4], two_ctas=False: PASS
plain_mma_multicast_commit_context [4,4], two_ctas=True: PASS
tma_mma_shared_inputs_context [4,4], two_ctas=False, multicast=True: PASS
tma_mma_shared_inputs_context [4,4], two_ctas=True, multicast=True, gather/scatter=True: PASS
```

Scaled-MMA helper rows with `num_ctas > 2` did not produce backend evidence. The shared test helper assumes two-CTA scale descriptor shapes for all `num_ctas > 1`; at 16 CTAs it failed before compiling the kernel:

```text
RuntimeError: shape '[1, 256, 4, 2, 256]' is invalid for input of size 1024
```

## Recommended promotion candidates

1. Add a checked-in runtime test that attempts a 1CTA `TensorMemoryLinearLayout` `ld/st` descriptor-view chain inside `num_ctas=4`, `8`, and `16` contexts. Today this cleanly reproduces candidate `FZ-20260421-0010`.
2. Add a checked-in runtime test that attempts a 2CTA linear `ld.red` descriptor-view chain inside `num_ctas=4`, `8`, and `16` contexts. This is a focused regression guard for the distinction between kernel CGA size and TCGEN05 instruction-local `cta_group`.
3. Add a structural frontend/API test for `TensorMemoryLinearLayout` with full `cga_layout` support, or document and cleanly reject the missing representational capability. The current `two_ctas` boolean is the likely schema bottleneck for arbitrary high-CGA linear TMEM layouts.
4. Add a dedicated high-CGA scaled-MMAv5 structural probe once the helper can build scale descriptors for `num_ctas > 2`; the current helper failure is not enough to classify backend behavior.

## Report-only guarantee

No backend code was modified by this lane. The only intended repository write is this report file.
