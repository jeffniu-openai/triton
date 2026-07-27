# Remove layout conversions handoff

This packet records the current state of `-tritongpu-remove-layout-conversions`
for long `reshape`/`join` chains and the dependency on `allow_reorder`. It was
validated on `origin/main` at
`c8305cccdeb7c2eb07441a5e696e7f9b17cd8e24` on 2026-07-27.

## Isolated environment

The validation used a worktree-local Python 3.12 virtual environment and an
out-of-tree CMake build. The standalone build avoids installing Triton's
Python package or changing another worktree:

```shell
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt \
  -r python/test-requirements.txt cmake ninja

.venv/bin/cmake -S . -B /tmp/triton-rlc-build -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_EXTERNAL_LIT="$PWD/.venv/bin/lit" \
  -DTRITON_BUILD_PYTHON_MODULE=OFF \
  '-DTRITON_CODEGEN_BACKENDS=nvidia;amd' \
  -DTRITON_BUILD_PROTON=OFF
.venv/bin/ninja -C /tmp/triton-rlc-build -j64 triton-opt
```

The LLVM cache or `TRITON_CACHE_PATH` still needs enough space for the normal
Triton LLVM dependency.

## Reproducer

The executable reproducer is
`test/TritonGPU/remove-layout-conversions-handoff.mlir`. It contains two cases
and two runs of the stochastic-rounding case:

- The order-preserving stochastic-rounding chain enters the pass with eight
  conversions and exits with eight. A second pass produces byte-identical IR.
- The same source is transformed by the test's second `RUN` line to mark six
  reshapes as `allow_reorder`. The pass removes all eight conversions.
  Removing any one of those six annotations leaves one conversion.
- The nested join tree from
  [issue #4030](https://github.com/triton-lang/triton/issues/4030) enters with
  fourteen conversions. Current main moves them through the tree and leaves
  one conversion after the final reshape because the function result fixes a
  different blocked layout. A second pass does not change it.

Run the focused test from a configured build directory:

```shell
lit -v test --filter=remove-layout-conversions-handoff
```

To inspect the two stochastic-rounding outputs directly:

```shell
triton-opt ../test/TritonGPU/remove-layout-conversions-handoff.mlir \
  -split-input-file -tritongpu-remove-layout-conversions

sed '/ALLOW-REORDER/s/ : tensor/ allow_reorder : tensor/' \
  ../test/TritonGPU/remove-layout-conversions-handoff.mlir |
  triton-opt - -split-input-file -tritongpu-remove-layout-conversions
```

## What remains unresolved

The first question is semantic: determine which of the eight conversions are
required when every reshape must preserve element order. `allow_reorder`
changes that contract, so the zero-conversion result is a useful upper bound,
not proof that the order-preserving form can reach zero.

If some conversions are avoidable, the current pass cannot find that global
layout assignment. It treats `allow_reorder` reshapes as layout anchors during
forward propagation, then applies backward rematerialization to a fixed point.
Without those anchors, the reduced chain is unchanged after both stages.

The second question is cost. A transformation that removes a conversion can
duplicate integer work or move a larger conversion elsewhere. The result needs
to be evaluated using generated shared-memory traffic and kernel performance,
not conversion count alone.

## Relevant code

- `lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp`
  - `isLayoutAnchor` marks `allow_reorder` reshapes as anchors.
  - `LayoutPropagation` performs forward propagation and conflict resolution.
  - `LayoutRematerialization` builds and prices backward slices.
  - `runOnOperation` repeats backward rematerialization until no conversion is
    removed.
- `lib/Dialect/TritonGPU/Transforms/Utility.cpp`
  - `inferDstEncoding` and `inferSrcEncoding` define propagation through
    `JoinOp`, `SplitOp`, and `ReshapeOp`.
- `lib/Dialect/TritonGPU/IR/Dialect.cpp`
  - `tryJoinOnAxis` implements forward join and backward split layout
    inference.
- `python/triton/language/core.py`
  - `tl.cat(can_reorder=False)` currently expands to
    `join` + `permute` + `reshape`.
  - `tl.cat(can_reorder=True)` uses `CatOp`.

## Suggested investigation order

1. Instrument forward conflict resolution on
   `@stochastic_rounding_join_chain` and record which producer first prevents
   each of the six useful anchor layouts from propagating.
2. For each retained conversion, ask whether the target layout can propagate
   backward through the adjacent join without changing element order. Keep the
   semantic constraint separate from the physical register layout.
3. Compare a global layout assignment across the full join tree with the
   current conversion-at-a-time backward slices.
4. Price shared-memory conversions, duplicated arithmetic, and register
   pressure together. Test both values of
   `disable-remat-splitting`.
5. Recreate the Python-level `tl.cat(can_reorder=False)` path so frontend
   lowering and TTGIR behavior remain covered together.

## Public history

- [#5783](https://github.com/triton-lang/triton/pull/5783) removed an unsafe
  rematerialization condition while explicitly accepting that some layouts
  would remain.
- [#5987](https://github.com/triton-lang/triton/pull/5987),
  [#6012](https://github.com/triton-lang/triton/pull/6012), and
  [#7065](https://github.com/triton-lang/triton/pull/7065) fixed join and
  reshape inference cases.
- [#8776](https://github.com/triton-lang/triton/pull/8776) made backward
  propagation iterate to a fixed point.
- [#9019](https://github.com/triton-lang/triton/pull/9019),
  [#9020](https://github.com/triton-lang/triton/pull/9020), and
  [#9463](https://github.com/triton-lang/triton/pull/9463) strengthened
  rematerialization and refactored forward propagation.
- [#9997](https://github.com/triton-lang/triton/pull/9997) and
  [#9998](https://github.com/triton-lang/triton/pull/9998) tightened
  `allow_reorder` layout inference and verification.
- [#10128](https://github.com/triton-lang/triton/pull/10128) and
  [#10129](https://github.com/triton-lang/triton/pull/10129) improved the cost
  model for register reorders and duplicated elementwise work.
- [#10646](https://github.com/triton-lang/triton/pull/10646) and
  [#10706](https://github.com/triton-lang/triton/pull/10706) fixed stale
  rematerialization reuse and mapping lifetime bugs.
- [#10749](https://github.com/triton-lang/triton/pull/10749) generalized
  split/join layout handling.
- [#11003](https://github.com/triton-lang/triton/pull/11003) added safer
  cast/broadcast hoisting and documents that forward conflict resolution is
  still not shared-memory-cost aware.

## Completion criteria

- The order-preserving reproducer has a documented lower bound for necessary
  conversions, with a correctness argument for each remaining one.
- Any reduction in conversions is stable after a second pass and does not
  introduce invalid dominance or stale rematerialization reuse.
- The `allow_reorder` variant does not regress.
- End-to-end kernels show no correctness change and no performance regression
  on the affected GPU architectures.
