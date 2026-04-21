# Round 23 Lane BD: high-CGA instruction-local CTA ownership fuzzing

- Date: 2026-04-21 13:20 UTC
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler code was changed.
- Target: kernels launched in 4/8/16 CTA-per-CGA contexts that contain
  instruction-local 1CTA or 2CTA TMEM operations.

## Summary

No new independent `FZ-*` bucket was found.

This round reconfirms `FZ-20260421-0010`: local 1CTA/2CTA TMEM layouts are
still rejected in larger 4/8/16 CTA launch contexts by a layout-context gate
that requires layout CTA count to equal kernel CTA count. The rejected rows are
clean parser/compiler diagnostics, not runtime wrong results or backend
crashes. The same run also reconfirms that full high-CGA MMAv5, scaled-MMAv5
copy+MMA, TMA multicast, and TMA+MMA controls still compile and execute.

The contrast remains important: the hardware `getModuleTwoCTAs` rule is about
consistent instruction-local 2CTA usage for instructions that can be 2CTA. The
red rows below are rejected earlier because their layout metadata has 1 or 2
CTAs while the kernel launch context has 4/8/16 CTAs. That looks like the
existing over-strict context/layout ownership gate, not a new hardware
consistency diagnostic.

## Commands

Required build gate:

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Checked-in runtime-matrix high-CGA/CTA diagnostic selector:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cga or cta_per_cga' --splits 4 --group 1 \
  --store-durations --durations-path /tmp/tmem_r23_high_cga_runtime_durations.json \
  2>&1 | tee /tmp/tmem_r23_high_cga_runtime_g1.log

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cga or cta_per_cga' --splits 4 --group 2 \
  --store-durations --durations-path /tmp/tmem_r23_high_cga_runtime_durations.json \
  2>&1 | tee /tmp/tmem_r23_high_cga_runtime_g2.log

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cga or cta_per_cga' --splits 4 --group 3 \
  --store-durations --durations-path /tmp/tmem_r23_high_cga_runtime_durations.json \
  2>&1 | tee /tmp/tmem_r23_high_cga_runtime_g3.log

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cga or cta_per_cga' --splits 4 --group 4 \
  --store-durations --durations-path /tmp/tmem_r23_high_cga_runtime_durations.json \
  2>&1 | tee /tmp/tmem_r23_high_cga_runtime_g4.log
```

Result:

```text
16 passed total:
- group 1: 4 passed, 1611 deselected
- group 2: 4 passed, 1611 deselected
- group 3: 4 passed, 1611 deselected
- group 4: 4 passed, 1611 deselected
```

High-CGA green controls:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_core.py::test_tma_multicast_copy[ctas_per_cga1]' \
  'python/test/gluon/test_core.py::test_tma_multicast_copy[ctas_per_cga2]' \
  2>&1 | tee /tmp/tmem_r23_core_tma_multicast.log

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[True-ctas_per_cga1]' \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[False-ctas_per_cga1]' \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[True-ctas_per_cga2]' \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[False-ctas_per_cga2]' \
  2>&1 | tee /tmp/tmem_r23_core_mma_multicast.log

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy[False-ctas_per_cga2-mxfp8-mxfp8-128-2048-2048-4096]' \
  'python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy[False-ctas_per_cga4-mxfp8-mxfp8-128-2048-2048-4096]' \
  'python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy[False-ctas_per_cga5-mxfp8-mxfp8-128-2048-2048-4096]' \
  2>&1 | tee /tmp/tmem_r23_core_scaled_mma_copy.log

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python pytest -q -s --tb=short \
  'python/test/gluon/test_core.py::test_tma_mma_shared_inputs[False-False-False-ctas_per_cga2-reps0-warps0]' \
  'python/test/gluon/test_core.py::test_tma_mma_shared_inputs[True-True-False-ctas_per_cga2-reps0-warps0]' \
  2>&1 | tee /tmp/tmem_r23_core_tma_mma.log
```

Result:

```text
- TMA multicast 4/16 CTA controls: 2 passed
- MMAv5 multicast/commit 8/16 CTA controls, 1CTA and 2CTA forms: 4 passed
- scaled-MMAv5 copy+MMA 4/8/16 CTA controls: 3 passed
- TMA+MMA 16 CTA controls, 1CTA and 2CTA forms: 2 passed
```

Temporary local-TMEM-in-high-CGA probe:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_high_cga_gate_round12_probe.py \
  2>&1 | tee /tmp/tmem_r23_high_cga_gate_probe.log
```

Result summary:

```text
SUMMARY {"FZ-20260421-0010": 15, "unclassified": 9}
```

The `9` unclassified rows were probe limitations: six scale-copy rows hid the
parser stderr that contained the CTA-count diagnostic, and three no-scales
2CTA rows used a stale helper signature.

Follow-up probe for the ambiguous rows:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python - <<'PY' 2>&1 | tee /tmp/tmem_r23_high_cga_gate_probe_followup.log
  # Subprocess probe for cp_scales_{1cta,2cta}_warpx4 and corrected
  # cp_noscales_2cta_128x128 in num_ctas={4,8,16}.
PY
```

Result summary:

```text
SUMMARY {"FZ-20260421-0010": 9}
```

## Case Table

| Surface | CTA context | Result | Classification |
| --- | --- | --- | --- |
| `ld/st` descriptor-view chain, 1CTA linear TMEM layout | 4, 8, 16 | compile-time diagnostic: `Layout has 1 CTAs per CGA, but the context requires X CTAs per CGA` | existing `FZ-20260421-0010` |
| `ld/st` descriptor-view chain, 2CTA linear TMEM layout | 4, 8, 16 | compile-time diagnostic: `Layout has 2 CTAs per CGA, but the context requires X CTAs per CGA` | existing `FZ-20260421-0010` |
| `ld.red`, 1CTA direct linear TMEM layout | 4, 8, 16 | compile-time diagnostic at `allocate_tensor_memory`: layout has 1 CTA, context requires X | existing `FZ-20260421-0010` |
| `ld.red`, 2CTA direct linear TMEM layout | 4, 8, 16 | compile-time diagnostic at `allocate_tensor_memory`: layout has 2 CTAs, context requires X | existing `FZ-20260421-0010` |
| no-scales `tcgen05.copy`, 1CTA 128x128 layout | 4, 8, 16 | compile-time diagnostic at TMEM allocation: layout has 1 CTA, context requires X | existing `FZ-20260421-0010` |
| no-scales `tcgen05.copy`, corrected 2CTA 128x128 layout | 4, 8, 16 | compile-time diagnostic at TMEM allocation: layout has 2 CTAs, context requires X | existing `FZ-20260421-0010` |
| scales `tcgen05.copy`, 1CTA `TensorMemoryScalesLayout` | 4, 8, 16 | parser diagnostic on `ttgl.arange`/blocked layout: layout has 1 CTA, context requires X | existing `FZ-20260421-0010` |
| scales `tcgen05.copy`, 2CTA `TensorMemoryScalesLayout(cga_layout=[[1,0]])` | 4, 8, 16 | parser diagnostic on `ttgl.arange`/blocked layout: layout has 2 CTAs, context requires X | existing `FZ-20260421-0010` |
| TMA multicast controls | 4, 16 | passed | green high-CGA control |
| MMAv5 multicast/commit controls | 8, 16 | passed for both 1CTA and 2CTA forms | green high-CGA control |
| scaled-MMAv5 copy+MMA controls | 4, 8, 16 | passed | green high-CGA control |
| TMA+MMA controls | 16 | passed for sampled 1CTA and 2CTA forms | green high-CGA control |

## Classification Notes

The red local-TMEM rows all fail before runtime execution. There were no
runtime wrong results, no PTX assembler rejections, no proxy-fence crashes, and
no new unsupported-diagnostic shape beyond the known CTA-count gate.

Representative diagnostics:

```text
Layout has 1 CTAs per CGA, but the context requires 4 CTAs per CGA.
Layout has 2 CTAs per CGA, but the context requires 8 CTAs per CGA.
Layout has 2 CTAs per CGA, but the context requires 16 CTAs per CGA.
```

For scale-copy rows the diagnostic is emitted while building the blocked
register layout:

```text
Result has an invalid layout: #ttg.slice<...>.
Layout has 2 CTAs per CGA, but the context requires 16 CTAs per CGA.
```

This confirms the earlier `FZ-0010` diagnosis across the requested surfaces:
`ld/st`, `ld.red`, no-scales copy, and scales copy. The green controls show the
backend is not generally unable to compile 4/8/16 CTA CGA kernels. The gap is
specifically around expressing instruction-local 1CTA/2CTA TMEM work inside a
larger kernel CGA context without forcing the TMEM layout CTA count to match
the kernel CTA count.

## New Candidates

None.

Suggested next discovery slice: fuzz mixed high-CGA kernels that pair one legal
full-CGA MMAv5/TMA operation with one local-TMEM operation, so the eventual
repair for `FZ-0010` can be validated against real mixed ownership rather than
single-surface probes only.
