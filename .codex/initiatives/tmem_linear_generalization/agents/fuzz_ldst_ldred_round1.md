# Lane A round 1: ld/st and ld.red structural fuzzing

- Time: 2026-04-21 08:25 UTC
- Branch/HEAD: `codex/tmem` at `57c00629b`
- Lane: ld/st and ld.red Python/Gluon runtime fuzzing
- Mode: discovery only; no backend or shared test file changes

## Setup and baseline

- Read `AGENTS.md` and `tmem_structural_fuzzing_20260421.md`.
- Ran `make` from `/root/code/triton`; ninja reported no work to do.
- Initial pytest collection without an explicit `PYTHONPATH` picked up stale
  `/tmp/triton-upstream-main-check/python` and failed with
  `ModuleNotFoundError: No module named 'triton.runtime.jit'`. All subsequent
  commands used `PYTHONPATH=/root/code/triton/python`.
- Baseline command:
  `PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_view_roundtrip python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred`
  - First run: one transient mismatch in
    `ldst-view-identity-32x32b`; `8064 / 8192` elements mismatched.
  - Fresh isolated rerun on GPU 1 with
    `TRITON_CACHE_DIR=/tmp/triton-cache-laneA-repro1` passed.
  - Fresh full-suite rerun on GPU 1 with
    `TRITON_CACHE_DIR=/tmp/triton-cache-laneA-suite2` passed.
  - Four-way split over GPUs 0-3 passed: 7 selected cases total.
  - Classification: `flake` / process-cache-sensitive symptom, not a stable
    minimized failure yet.

## Ad-hoc driver

Temporary driver used for round 1:
`/tmp/tmem_lane_a_probe.py`.

The driver builds deterministic Gluon kernels with lifted TMEM parents, then
executes and compares GPU outputs against PyTorch. It covered:

- ld/st: f32/i32/f16 where practical, `N=32/64/128/256`,
  rows `64/128/256`, row/col bit permutations, and chains:
  direct, `reshape(M//2,2,N).permute(1,0,2).reshape(M,N)`,
  `reshape(M,N//2,2).permute(...).permute(...).reshape(M,N)`,
  transpose/slice/transpose/slice, and one higher-rank reshape/permute chain.
- ld.red: min/max, direct/indexed parent views, reshape/permute chains,
  transpose/slice chains, `M=64/128/256`, `N=32/64/128`.
- First pass used parent shape `[4, M, N]`; many larger shapes were expected
  TMEM resource boundaries. Second pass used `[2, M, N]` to reduce resource
  noise.

Representative broad command:

`PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-laneA-adHoc-prefix2 python /tmp/tmem_lane_a_probe.py --limit 80`

That run was intentionally interrupted after reaching stable i32 coverage
because 256-row cases were repeatedly hitting the same compiler assertion.

## Findings

### A1: ld/st chain-1 descriptor view miscompile

- Failure class: `miscompile`
- Family: `ldst`
- Stable repro: yes, fresh process/cache reruns failed.
- Minimal case parameters:
  - `seed`: deterministic driver seed `0xA000 + case suffix`
  - `shape`: `[2, 64, 32]` parent, indexed view `[64, 32]`
  - `dtype`: `f32`
  - `layout_kind`: identity `TensorMemoryLinearLayout`
  - `view_chain`: `index(1).reshape((M//2,2,N)).permute([1,0,2]).reshape((M,N))`
  - `num_warps`: 4
  - `num_ctas`: 1
  - `instr_variant`: fails for `32x32b`, `16x64b`, and `16x128b`
- Exact repro command:
  ```bash
  PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-laneA-min-identity-chain1b python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("lane", "/tmp/tmem_lane_a_probe.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
case = mod.Case("laneA-ldst-1000", "ldst", "f32", 64, 32, "identity", "identity", "32x32b", 1)
mod._run_case(case)
PY
  ```
- Observed result:
  - `32x32b`, `16x64b`, `16x128b` all fail with `Tensor-likes are not equal`.
  - Example stable failure: `2032 / 2048` mismatched elements for
    `M=64,N=32,row=identity,col=reverse,instr=16x64b,chain=1`.
- Neighbor controls:
  - Direct chain `chain_id=0` passes.
  - `chain_id=2` identity layout passes for all three instruction variants.
  - Higher-rank chain `chain_id=4` on `M=128,N=32` identity layout passes.
- Extended coverage:
  - i32 shows the same chain-1 mismatch family, for example
    `M=64,N=32,row=reverse,col=identity,instr=16x64b,chain=1`
    mismatched `1984 / 2048`.
  - f16 direct descriptor view with `32x32b` is rejected before execution
    (`TMEM layout 'constexpr[32x32b]' unsupported for descriptor view
    tensor_memory_descriptor<fp16,...>`), but f16 chain-1 cases that compile
    also show mismatches, e.g. `M=64,N=64,instr=16x64b,chain=1`.
- Likely owner surface: descriptor-view linear-layout arithmetic for
  reshape/permute row grouping in direct `tcgen05.ld/st`.

### A2: ld.red view chains produce plain ld plus software reduce

- Failure class: `opcode_mismatch`
- Family: `ldred`
- Stable repro: yes for focused cases; data comparison passed before opcode
  assertion.
- Minimal case parameters:
  - `shape`: `[2, 64, 32]` parent, indexed view `[64, 32]`
  - `dtype`: `f32`
  - `layout_kind`: identity `TensorMemoryLinearLayout`
  - `view_chain`: `index(1).reshape((M//2,2,N)).permute([1,0,2]).reshape((M,N))`
  - `operation`: `load_min`
  - `num_warps`: 4
  - `num_ctas`: 1
- Exact repro command:
  ```bash
  PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-laneA-ldred-focused python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("lane", "/tmp/tmem_lane_a_probe.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
case = mod.Case("laneA-ldred-2001", "ldred", "f32", 64, 32, "identity", "identity", "auto", 1, 0)
mod._run_case(case)
PY
  ```
- Observed result:
  - Output tensor and reduced tensor matched PyTorch.
  - PTX contained `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`, not
    `.ld.red.`.
- Additional opcode mismatch:
  - `laneA-ldred-2002`: `M=64,N=64,row=identity,col=reverse,chain=2,load_min`
    emitted repeated plain `tcgen05.ld.sync.aligned.16x32bx2.x2.b32`.
- Likely owner surface: ld.red lowering loses hardware reduction selection
  after descriptor-view chains, falling back to normal load plus software
  reduction.

### A3: 256-row lifted parent trips tensor memory allocation assertion

- Failure class: `compiler_crash`
- Family: `ldst` and `ldred`
- Stable repro: yes, fresh process/cache reruns failed.
- Minimal ld/st case parameters:
  - `shape`: `[2, 256, 32]` parent, indexed view `[256, 32]`
  - `dtype`: `f32`
  - `layout_kind`: identity `TensorMemoryLinearLayout`
  - `view_chain`: direct `index(1)`
  - `instr_variant`: `32x32b`
  - `num_warps`: 4
  - `num_ctas`: 1
- Exact repro command:
  ```bash
  PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-laneA-crash256-min python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("lane", "/tmp/tmem_lane_a_probe.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
case = mod.Case("laneA-ldst-0036", "ldst", "f32", 256, 32, "identity", "identity", "32x32b", 0)
mod._run_case(case)
PY
  ```
- Observed result:
  - Assertion from `TensorMemoryAllocation.cpp:65`:
    `MemoryBitMap::findFirstFit(...): Assertion 'kNumRows - numRows >= 0' failed.`
  - Pass manager reports failure while executing
    `TritonTensorMemoryAllocationPass`.
- ld.red variant:
  - `laneA-ldred-2007`: `M=256,N=32,row=identity,col=identity,chain=0,load_min`
    hits the same assertion.
- Likely owner surface: TMEM allocation should reject or split row footprints
  cleanly instead of asserting when descriptor view requests 256 rows.

### A4: ld.red transpose/slice descriptor view rejected before reduction

- Failure class: `false_unsupported` candidate
- Family: `ldred`
- Stable repro: yes.
- Case parameters:
  - `case_id`: `laneA-ldred-2003`
  - `shape`: `[2, 64, 128]` parent, indexed view `[64, 128]`
  - `dtype`: `f32`
  - `layout_kind`: row `rotate1`, col `identity`
  - `view_chain`: `index(1).permute([1,0]).permute([1,0]).slice(0,M,dim=0).slice(0,N,dim=1)`
  - `operation`: `load_max`
- Exact repro command:
  ```bash
  PYTHONPATH=/root/code/triton/python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-laneA-ldred2003-min python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("lane", "/tmp/tmem_lane_a_probe.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
case = mod.Case("laneA-ldred-2003", "ldred", "f32", 64, 128, "rotate1", "identity", "auto", 3, 1)
mod._run_case(case)
PY
  ```
- Observed result:
  - Compile-time diagnostic at `view.get_reg_layout()`.
  - Message: `TMEM layout 'auto' unsupported for descriptor view ... required row anchors 32,64 are not directly representable...`
- Classification note:
  - This may be a true current lowering boundary for direct ld/st packet
    row anchors, but the campaign objective is arbitrary linear view chains
    where ISA-realizable. Keep as `false_unsupported` candidate until the
    planner proves it ISA-impossible.

## Expected or lower-priority boundaries

- Resource boundaries:
  - Parent `[2,M,N]` with `N=256` often reports clean
    `OutOfResources: tensor memory, Required: 1024, Hardware limit: 512`.
  - These were not cataloged as backend failures.
- f16 direct descriptor view:
  - `M=64,N=32,instr=32x32b` reports
    `TMEM layout 'constexpr[32x32b]' unsupported for descriptor view
    tensor_memory_descriptor<fp16,...>`.
  - Treat as subword coverage boundary unless a fp16-specific ld/st packet
    variant is expected.

## Suggested next cases

- Convert the stable A1 chain-1 repro into a minimal committed pytest only
  after checking no other lane is editing `test_tmem_structural_fuzzer.py`.
- Capture the A3 MLIR reproducer to `/tmp/laneA_256row_tmem_alloc.mlir` and
  rerun with `triton-opt --run-reproducer` to confirm it is independent of
  Python runtime launch.
- Add two-CTA ld/st and ld.red variants with resource-valid shapes
  (`M=256,N=32` two-CTA layout may avoid the one-CTA 256-row allocator
  assertion and test the intended 2-CTA path).
- Extend ld.red opcode checks across `load_max`, abs/nan flags, and
  `N=32/64/128` after descriptor chains to distinguish plain-load fallback
  from true unsupported cases.
