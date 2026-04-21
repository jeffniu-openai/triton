# Lane R5-A round 5: ld.red row/col chain1 optimizer crash minimization

- Time: 2026-04-21 UTC
- Lane: R5-A
- Scope: minimize the R4-D report-only row/col chain1 optimizer crash in
  `TritonNvidiaGPUOptimizeTMemLayoutsPass`.
- Backend repair status: no backend/compiler code changed.

## Artifacts

- Crash-safe parent pytest probe:
  `/tmp/tmem_ldred_optimizer_crash_round5_probe.py`
- File-backed child program materialized by the probe:
  `/tmp/tmem_ldred_optimizer_crash_round5_child.py`
- Parent classification log:
  `/tmp/tmem_ldred_optimizer_crash_round5_probe.log`
- Direct minimized crash log:
  `/tmp/tmem_ldred_optimizer_crash_round5_min_256x2.log`

## Commands

Required rebuild before pytest:

```bash
make -j8
```

Result: `ninja: no work to do`.

Probe syntax, collection, and subprocess-isolated classification:

```bash
PYTHONPATH=.:./python python -m py_compile /tmp/tmem_ldred_optimizer_crash_round5_probe.py
PYTHONPATH=.:./python pytest --collect-only -q /tmp/tmem_ldred_optimizer_crash_round5_probe.py
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r5a-crash-final PYTHONPATH=.:./python pytest -s --tb=short /tmp/tmem_ldred_optimizer_crash_round5_probe.py 2>&1 | tee /tmp/tmem_ldred_optimizer_crash_round5_probe.log
```

Results: collect-only found four nodeids; parent pytest passed as `4 passed`.
Each row ran in a child Python process, so the optimizer failure and any LLVM
abort/signal remain isolated from the parent pytest process.

Direct minimized crash confirmation:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-r5a-min-final CASE_M=256 CASE_N=2 CASE_CHAIN=1 CASE_ROW=even_odd CASE_COL=identity PYTHONPATH=.:./python python /tmp/tmem_ldred_optimizer_crash_round5_child.py 2>&1 | tee /tmp/tmem_ldred_optimizer_crash_round5_min_256x2.log
```

Result: reproduces the optimizer failure and prints:

```text
LLVM ERROR: Dimensions must match, ignoring order, but they don't.  Got dims: ["row", "col"] and ["row", "col", "block"]
Pipeline failed while executing [`TritonNvidiaGPUOptimizeTMemLayoutsPass` on 'builtin.module' operation]
```

## Minimized Stable Repro

- Case id: `ldred-fz20260421-crash-twocta-indexed-256x2-chain1-even_odd-min`
- Family: `ldred`
- Shape: parent `[2,256,2]`, selected view `[256,2]`
- Layout: 2CTA `TensorMemoryLinearLayout`, row `even_odd`, col `identity`,
  lifted through prefix `[2]`
- View chain:
  `parent.index(1).reshape((128,2,2)).permute([1,0,2]).reshape((256,2))`
- Operation: store full tile, then `view.load_min()`
- Runtime status: no runtime/opcode stage reached
- Failure class: compiler crash/optimizer abort in
  `TritonNvidiaGPUOptimizeTMemLayoutsPass`
- Exact symptom: dimensions mismatch `["row","col"]` vs
  `["row","col","block"]`
- Stability: reproduced through the subprocess pytest harness and direct child
  process command.

## Boundary Checks

- `M=128,N=32,chain1,row=even_odd` and `M=64,N=32,chain1,row=even_odd` do not
  reach this optimizer crash; they stop earlier with clean
  `TMEM layout 'auto' unsupported for descriptor view` diagnostics.
- `M=256,N=1,chain1,row=even_odd` does not hit this optimizer crash; it reports
  the clean `.x1` ld.red minimum-message diagnostic.
- `M=256,N=32,chain1,row=identity,col=identity` compiles and remains an
  ld.red opcode fallback row, not a crash: `CLASS plain_ld` with
  `tcgen05.ld.sync.aligned.16x32bx2.x16.b32`.
- `M=512,N=32,chain0,row=identity,col=identity` remains a clean resource
  diagnostic: register and memory `cga_layout` differ.
- The R4-D adjacent `M=256,N=64,chain1,row=identity,col=reverse` row still
  reproduces the same optimizer-pass failure class.

## Recommendation

Use report-only status for now unless a subprocess-isolated crash harness is
accepted for checked-in structural-fuzzer coverage. This row can trip an LLVM
error/signal while emitting the MLIR reproducer, so a normal strict xfail around
the in-process kernel is not crash-safe enough.

If checked-in coverage is desired, add only a subprocess-isolated strict xfail
candidate for
`ldred-fz20260421-crash-twocta-indexed-256x2-chain1-even_odd-min`, and keep it
separate from `FZ-20260421-0004` opcode-fallback xfails. The expected class
should be `optimizer_crash`, not `plain_ld` and not a clean unsupported
diagnostic.
