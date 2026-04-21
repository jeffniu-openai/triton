# Round 30 Descriptor-View Chain-Shape Fuzzing

Date: 2026-04-21

Scope: discovery/cataloging only. No backend or compiler code was modified.

Required first step:

```bash
make -j8
```

Result: no-op rebuild through
`/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`.

## Summary

This lane extended the descriptor-view chain-shape fuzzing surface around
`reshape`, `permute`, `slice`, `index`, direct versus indexed roots, unit-prefix
rank-4 views, `ld/st`, and actual `ld.red` consumers. The temporary probe runs
each case in a fresh Python subprocess so compiler aborts do not kill the whole
lane.

Artifacts:

- `/tmp/tmem_chain_shapes_round30.py`
- `/tmp/tmem_chain_shapes_round30/worker.py`
- `/tmp/tmem_chain_shapes_round30/summary_ldst.json`
- `/tmp/tmem_chain_shapes_round30/summary_ldst_unit.json`
- `/tmp/tmem_chain_shapes_round30/summary_ldred.json`

New or expanded buckets:

- `FZ-20260421-0019`: expanded from the Round 34 rank-5 unit-dimension
  finding; rank-4 unit-prefix descriptor views also crash in `get_reg_layout`
  with an LLVM dimension-name mismatch before TTGIR lowering.
- `FZ-20260421-0020`: valid half-row `ld/st` descriptor-view chains can fail
  `TritonNvidiaGPUOptimizeTMemLayoutsPass` with a signal-generated MLIR
  reproducer instead of compiling or giving a clean diagnostic.
- `FZ-20260421-0021`: valid half-column `ld/st` descriptor-view chains can
  abort the Python compiler process with an LLVM dimension-name mismatch.
- `FZ-20260421-0022`: actual `tcgen05.ld.red` on a row-reversed half-row
  descriptor view fails during Gluon parsing for the otherwise-supported shape
  family; identity and column-reversed controls pass.

No runtime wrong-result miscompile was confirmed in this lane. An early
software-reduction probe had an invalid row-sum oracle for a non-row-preserving
reshape/permute chain and is not counted as a backend bug. The final `ld.red`
results below use `view.load_max(...)`.

## Temporary Probe Matrix

Command:

```bash
PYTHONPATH=.:./python python3 /tmp/tmem_chain_shapes_round30.py --only ldst
PYTHONPATH=.:./python python3 /tmp/tmem_chain_shapes_round30.py --only ldst_unit
PYTHONPATH=.:./python python3 /tmp/tmem_chain_shapes_round30.py --only ldred
```

Environment:

- one subprocess per case;
- `CUDA_VISIBLE_DEVICES = case_index % 4`;
- stable cache directories `/tmp/triton-cache-gpu0` ...
  `/tmp/triton-cache-gpu3`;
- no per-test cache workaround.

`ld/st` matrix:

- rows: `276`
- result counts: `132 pass`, `48 clean_or_false_unsupported`,
  `42 clean_oor`, `36 exception`, `18 compiler_crash_or_abort`
- shapes: `M=128`, `N in {64, 128, 256}`
- layouts: identity, row-reversed rows, column-reversed columns
- variants: `auto`, `32x32b`, `16x128b`
- roots: `index(1)` and `slice(1, 1).index(0)` from a `[2, M, N]` parent
- chains:
  - chain 0: `reshape(M/2,2,N) -> permute -> reshape(M,N) -> double transpose`
  - chain 1: rank-4 double permutation, semantically identity
  - chain 2: rank-5 double permutation, semantically identity
  - chain 3: half-row view followed by reshape/permute/reshape
  - chain 4: direct half-column view
  - chain 5: reshaped/transposed half-column view

`ld/st` positives:

- full-tile and semantically identity chains 0, 1, and 2 passed across all
  tested `N=64/128` layouts and variants.
- row-reversed direct-root half-row chain 3 passed for `N=64/128`; the same
  chain through a slice/index root failed, which narrows one root-sensitivity
  axis.
- column-reversed half-column chain 5 passed for `N=64/128`; identity and
  row-reversed variants expose `FZ-0021` at `N >= 128`.
- `N=256` full-tile rows mostly hit clean TMEM OOR boundaries, as expected.

Clean boundaries:

- direct half-column views with shape like `[1, 32]` or unsupported row-anchor
  footprints report the existing clean descriptor-view diagnostic:
  `unsupported tensor memory descriptor view for direct tcgen05.ld/st`.
- `N=256` full-tile chains report clean TMEM OOR in the resource-heavy rows.

Actual `ld.red` matrix:

- rows: `48`
- result counts: `32 pass`, `12 clean_or_false_unsupported`, `4 exception`
- shapes: `M=128`, `N in {64, 128}`
- layouts: identity, row-reversed rows, column-reversed columns
- variants: `auto`, `32x32b`
- consumer: `view.load_max(layout=..., abs=False, propagate_nan=tl.PropagateNan.NONE)`

`ld.red` positives:

- identity and column-reversed layouts pass for full-tile, semantically
  identity rank-4 chains, and half-row slice chains.
- row-reversed layouts pass for full-tile and rank-4 identity chains.
- half-column `ld.red` views cleanly reject with the existing descriptor-view
  unsupported diagnostic.

## FZ-20260421-0019: Unit-Prefix Rank Crash

Minimal family:

```python
tmem = allocate_tensor_memory(ttgl.float32, [1, 1, M, N], layout)
base = tmem.index(0).index(0)
base_layout = base.get_reg_layout(instr_variant="auto")
```

Observed:

```text
LLVM ERROR: Dimensions must match, ignoring order, but they don't.
Got dims: ["dim0", "dim1"] and ["dim2", "dim3"]
Fatal Python error: Aborted
```

Coverage:

- `8/8` unit-prefix cases aborted.
- `N in {64, 128}`
- roots: `index(0).index(0)` and
  `slice(0, 1).index(0).slice(0, 1).index(0)`
- variants: `auto`, `32x32b`

This is distinct from `FZ-20260421-0016`: tensors and memdescs are encoded, and
the crash is a dimension-name mismatch in descriptor-view layout arithmetic, not
the unencoded tensor verifier path.

## FZ-20260421-0020: Half-Row Optimizer Signal

Minimal family:

```python
tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
base = tmem.index(1)  # or tmem.slice(1, 1, dim=0).index(0)
view = base.reshape((2, M // 2, N)).slice(0, 1, dim=0).index(0)
view = view.reshape((M // 4, 2, N)).permute([1, 0, 2]).reshape((M // 2, N))
view_layout = view.get_reg_layout(instr_variant=variant)
```

Observed:

```text
RuntimeError: PassManager::run failed
Pipeline failed while executing
`TritonNvidiaGPUOptimizeTMemLayoutsPass`
```

The generated reproducer contains a chain like:

```text
ttg.memdesc_reshape ... !ttg.memdesc<128x64xf32, ...>
  -> !ttg.memdesc<2x64x64xf32, ...>
ttg.memdesc_reshape ... !ttg.memdesc<64x64xf32, ...>
  -> !ttg.memdesc<32x2x64xf32, ...>
ttg.memdesc_reshape ... !ttg.memdesc<2x32x64xf32, ...>
  -> !ttg.memdesc<64x64xf32, ...>
```

Coverage:

- `36` failures in the `ld/st` matrix.
- identity layout: both direct and slice/index roots fail for `N=64/128/256`.
- column-reversed layout: both roots fail for `N=64/128`.
- row-reversed layout: direct root passes, slice/index root fails for
  `N=64/128`.
- all variants tested (`auto`, `32x32b`, `16x128b`) reproduce when the layout
  and root combination is in the failing set.

The chain is semantically valid: it updates a half-row view and compares the
base tile against a reference with only the first half of rows incremented.

## FZ-20260421-0021: Half-Column Dimension Abort

Minimal family:

```python
tmem = allocate_tensor_memory(ttgl.float32, [2, M, N], layout)
base = tmem.index(1)
view = base.reshape((2, M // 2, 2, N // 2))
view = view.permute([2, 1, 0, 3]).reshape((2, M, N // 2))
view = view.slice(1, 1, dim=0).index(0)
view_layout = view.get_reg_layout(instr_variant=variant)
```

Observed:

```text
LLVM ERROR: Dimensions must match, ignoring order, but they don't.
Got dims: ["dim0", "dim1"] and ["dim1", "dim2"]
Fatal Python error: Aborted
```

Coverage:

- `18` process aborts in the `ld/st` matrix.
- identity layout: `N=128` and `N=256`, both roots, all variants.
- row-reversed layout: `N=128`, both roots, all variants.
- column-reversed layout: `N=64/128` controls pass.
- `N=64` identity/row-reversed slice/index roots mostly cleanly reject because
  the selected descriptor view cannot materialize the public row anchors.

This is a process abort, not a typed compiler diagnostic.

## FZ-20260421-0022: Row-Reversed Half-Row `ld.red` Parse Failure

Minimal family:

```python
tmem = allocate_tensor_memory(ttgl.float32, [2, 128, N], layout)
base = tmem.index(1)
view = base.slice(0, 64, dim=0)
view_layout = view.get_reg_layout(instr_variant=variant)
loaded, reduced = view.load_max(
    layout=view_layout, abs=False, propagate_nan=tl.PropagateNan.NONE
)
```

Observed:

```text
RuntimeError: error encountered during parsing
```

Coverage:

- `4` failures: row-reversed layout, `N in {64, 128}`, variants `auto` and
  `32x32b`.
- identity and column-reversed controls for the same half-row `ld.red` shape
  pass.
- half-column `ld.red` views reject cleanly with the existing descriptor-view
  unsupported diagnostic.

This should be minimized further before repair work starts; the failure is
currently a frontend/Gluon parse error surfaced by a TMEM descriptor-view
family, not yet a narrowed backend pass crash.

## Notes For Follow-Up

- Preserve the positive controls when turning this into checked-in tests:
  identity/col-reversed `ld.red` half-row rows and row-reversed direct-root
  `ld/st` half-row rows are important support-matrix evidence.
- The unit-prefix crash is a good adversarial verifier/reg-layout boundary
  because it uses encoded tensors and valid unit dimensions, so it should not
  be folded into the unencoded tensor bucket.
- Before fixing, minimize `FZ-0021` into TTGIR/MLIR and compare its dimension
  names against the passing column-reversed controls; this is likely a clean
  way to expose the underlying descriptor-view composition mismatch.
