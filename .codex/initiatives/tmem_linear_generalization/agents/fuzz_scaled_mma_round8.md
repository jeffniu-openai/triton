# Round 8 Lane B: scaled-MMAv5 accumulator descriptor-view control-flow fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- HEAD at lane close: `4645caa9f`
- Mode: discovery-only. No backend/compiler fixes were attempted.
- Temporary harness: `/tmp/tmem_scaled_mma_round8_probe.py`
- Result JSONL: `/tmp/tmem_scaled_mma_round8_results.jsonl`
- Dedicated runtime lane: `CUDA_VISIBLE_DEVICES=1`,
  `TRITON_CACHE_DIR=/tmp/triton-cache-round8-laneB-gpu1`

## Scope

This lane expanded scaled-MMAv5 accumulator descriptor-view control-flow
coverage around `FZ-20260421-0007`.

The temporary harness compiled, ran, compared outputs against Torch reference
matmuls, and inspected PTX/LLIR `tcgen05.mma` opcodes for:

- accumulator views: low/high direct slices, dynamic slice `if`, dynamic slice
  loop, helper-returned slices, dynamic indexed `if`, dynamic indexed loop, and
  helper-returned indexed views;
- `use_acc` values: `False` and `True`;
- `N in {16, 32, 64, 128}`;
- `K in {128, 256}`;
- formats: `mxfp8xmxfp8`, `mxfp8xmxfp4`, `mxfp4xmxfp8`,
  `mxfp4xmxfp4`, and `nvfp4xnvfp4`;
- selector values `0` and `1`.

The custom control-flow harness is 1CTA. Larger-CGA sanity was covered with
existing scaled-MMA runtime controls, including 4CTA and 16CTA cases.

## Commands

Rebuild:

```bash
make -j8
```

Harness syntax:

```bash
python -m py_compile /tmp/tmem_scaled_mma_round8_probe.py
```

Collection:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  pytest --collect-only -q /tmp/tmem_scaled_mma_round8_probe.py
```

Result: `1120` nodeids collected.

Full sweep on the lane GPU/cache:

```bash
rm -f /tmp/tmem_scaled_mma_round8_results.jsonl
CUDA_VISIBLE_DEVICES=1 \
TRITON_CACHE_DIR=/tmp/triton-cache-round8-laneB-gpu1 \
TMEM_ROUND8_RESULTS=/tmp/tmem_scaled_mma_round8_results.jsonl \
PYTHONPATH=.:./python:./python/test/gluon \
pytest -q -s --tb=short --splits 4 --group 2 /tmp/tmem_scaled_mma_round8_probe.py

for g in 1 3 4; do
  CUDA_VISIBLE_DEVICES=1 \
  TRITON_CACHE_DIR=/tmp/triton-cache-round8-laneB-gpu1 \
  TMEM_ROUND8_RESULTS=/tmp/tmem_scaled_mma_round8_results.jsonl \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short --splits 4 --group "$g" /tmp/tmem_scaled_mma_round8_probe.py
done
```

Results:

- group 1: `280 passed, 840 deselected in 158.78s`;
- group 2: `280 passed, 840 deselected in 97.18s`;
- group 3: `280 passed, 840 deselected in 111.53s`;
- group 4: `280 passed, 840 deselected in 133.57s`.

The pytest rows pass when the row is classified and recorded. Backend
exceptions and runtime miscompiles are recorded in the JSONL.

Fresh exact contrasts for the narrow `nvfp4` high-selector finding:

```bash
rm -f /tmp/tmem_scaled_mma_round8_exact.jsonl
for pat in \
  'direct_high-nvfp4xnvfp4-n16-k256-sel0-acc1' \
  'direct_low-nvfp4xnvfp4-n16-k256-sel0-acc1' \
  'slice_if-nvfp4xnvfp4-n16-k256-sel1-acc1' \
  'indexed_if-nvfp4xnvfp4-n16-k256-sel1-acc1' \
  'slice_if-nvfp4xnvfp4-n16-k256-sel0-acc1'; do
  CUDA_VISIBLE_DEVICES=1 \
  TRITON_CACHE_DIR=/tmp/triton-cache-round8-laneB-gpu1 \
  TMEM_ROUND8_RESULTS=/tmp/tmem_scaled_mma_round8_exact.jsonl \
  PYTHONPATH=.:./python:./python/test/gluon \
  pytest -q -s --tb=short /tmp/tmem_scaled_mma_round8_probe.py -k "$pat"
done
```

Each exact selector returned `1 passed, 1119 deselected`. JSONL outcomes:

- `direct_high-nvfp4xnvfp4-n16-k256-sel0-acc1`: `pass`,
  opcode match true, `4` MMA ops;
- `direct_low-nvfp4xnvfp4-n16-k256-sel0-acc1`: `pass`,
  opcode match true, `4` MMA ops;
- `slice_if-nvfp4xnvfp4-n16-k256-sel1-acc1`: `miscompile`,
  `2048` mismatches, `224` NaNs, opcode match true, `4` MMA ops;
- `indexed_if-nvfp4xnvfp4-n16-k256-sel1-acc1`: `miscompile`,
  `2048` mismatches, `288` NaNs, opcode match true, `4` MMA ops;
- `slice_if-nvfp4xnvfp4-n16-k256-sel0-acc1`: `miscompile`,
  `2048` mismatches, `160` NaNs, opcode match true, `4` MMA ops.

Larger-CGA sanity controls:

```bash
CUDA_VISIBLE_DEVICES=1 \
TRITON_CACHE_DIR=/tmp/triton-cache-round8-laneB-gpu1 \
PYTHONPATH=.:./python:./python/test/gluon \
pytest -q -s --tb=short \
  'python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy[False-ctas_per_cga2-mxfp8-mxfp8-128-2048-2048-4096]' \
  'python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy[False-ctas_per_cga5-mxfp8-mxfp8-128-2048-2048-4096]'
```

Result: `2 passed in 4.67s`. These cover 4CTA and 16CTA scaled-MMA copy
controls with existing runtime coverage.

## Counts

JSONL rows: `1120`.

Status totals:

- `pass`: `628`;
- `miscompile`: `332`;
- `exception`: `160`.

By mode:

- `direct_low`: `80 pass`;
- `direct_high`: `80 pass`;
- `slice_if`: `78 pass`, `82 miscompile`;
- `slice_loop`: `117 pass`, `43 miscompile`;
- `slice_helper`: `78 pass`, `82 miscompile`;
- `indexed_if`: `78 pass`, `82 miscompile`;
- `indexed_loop`: `117 pass`, `43 miscompile`;
- `indexed_helper`: `160 exception`.

Opcode inspection:

- all `pass` and `miscompile` rows had PTX/LLIR opcode agreement;
- all emitted MMA opcodes matched the expected scaled-MMAv5 opcode family for
  their format and CTA group;
- no opcode mismatch was isolated in this lane.

Exception classification:

- the `indexed_helper` exceptions are dynamic `ttg.memdesc_index` legalization
  failures printed by the compiler while lowering helper-returned indexed
  accumulator views;
- this overlaps existing `FZ-20260421-0001` rather than introducing a separate
  scaled-MMAv5-specific compiler bucket.

## Findings

### FZ-20260421-0007 expansion

The broad dynamic descriptor-view accumulator miscompile remains stable.
Direct low and high controls passed across all probed formats, `N`, `K`, and
`use_acc` values. Dynamic slice/indexed paths miscompiled with correct-looking
`tcgen05.mma` opcodes, which points at descriptor-view/control-flow lowering
or view selection rather than opcode selection.

Most miscompiles follow the known low-selector shape from `FZ-20260421-0007`.
Round 8 also found a narrower high-selector variant:

- `nvfp4xnvfp4`, `N=16`, `K=256`, selector `1`;
- reproduces through `slice_if`, `slice_loop`, `slice_helper`, `indexed_if`,
  and `indexed_loop`;
- reproduces for both `use_acc=False` and `use_acc=True`;
- direct low/high controls pass;
- PTX/LLIR MMA opcode inspection still matches expected opcodes.

This should be treated as an important `FZ-20260421-0007` expansion rather than
a separate id for now. It shares the same dynamic accumulator view selection
surface, but it proves the bug is not limited to low-column view selection or
`use_acc=True`.

### FZ-20260421-0001 overlap

All helper-returned indexed accumulator rows failed to compile via dynamic
`ttg.memdesc_index` illegal lowering. This is useful scaled-MMAv5 evidence for
`FZ-20260421-0001`, but not a new independent bucket.

### Larger-CGA boundary

The existing 4CTA and 16CTA scaled-MMA copy controls passed. This lane did not
find a larger-CGA clean-diagnostic failure or larger-CGA miscompile for the
scaled-MMAv5 copy path. The custom descriptor-view control-flow harness remains
1CTA; extending that harness to real 2CTA and larger-CGA descriptor-view
control flow is still useful follow-up coverage.

## Promotion Recommendations

- Add a strict runtime xfail row for the narrow high-selector expansion:
  `nvfp4xnvfp4`, `N=16`, `K=256`, selector `1`, dynamic slice/indexed `if`,
  with direct high control documented as passing.
- Keep that row under `FZ-20260421-0007` unless a later minimization separates
  the `N=16`/`nvfp4` high-selector path from the existing dynamic accumulator
  view-selection root cause.
- Do not promote the `indexed_helper` failures as a new id; add them only as
  scaled-MMAv5 overlap evidence for `FZ-20260421-0001`.
- Add true 2CTA/larger-CGA descriptor-view control-flow fuzzing in a later lane.
  Existing copy controls are green, but this lane did not exercise dynamic
  accumulator view selection in larger-CGA layouts.
