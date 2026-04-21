# Round 14 Lane AJ: high-CGA mixed ownership fuzzing

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery/cataloging only. No backend or compiler code was changed.

## Scope

Lane AJ focused on high-CGA mixed ownership around local TMEM operations,
MMAv5/TMA-MMA controls, no-scales/scales copy, and pass interactions around
`triton-nvidia-check-matmul-two-cta`.

The lane classified against the existing high-CGA local-TMEM CTA-count gate
(`FZ-20260421-0010`) and the mixed two-CTA copy/mbarrier proxy-fence crash
(`FZ-20260421-0014`). No new independent `FZ-*` bucket was found.

## Required rebuild

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

## Case matrix

| Surface | Command/probe | Result | Classification |
| --- | --- | --- | --- |
| Local 1CTA/2CTA `ld/st`, `ld.red`, no-scales copy, scales copy in 4/8/16 CTA launch contexts | `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon python /tmp/tmem_high_cga_gate_round12_probe.py` | Raw rerun: `15` rows with the exact CTA-count diagnostic, `9` harness-limited/unclassified rows | Existing `FZ-20260421-0010` for the `15` direct diagnostic rows; harness rows were not counted as new backend evidence |
| Mixed 2CTA no-scales descriptor-chain copy plus scales direct copy | `CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:./python/test/gluon pytest -s --tb=short /tmp/tmem_high_cga_copy_scales_round14_probe.py` | `2 failed, 3 passed`; the legal `num_ctas=2` mixed row failed in proxy-fence insertion | Existing `FZ-20260421-0014` |
| Same mixed copy/scales module in 4/8/16 CTA launch contexts | Same pytest command above | `3` high-CGA rows passed their expected diagnostic assertions | Existing `FZ-20260421-0010` |
| Scales descriptor-view mixed row | Same pytest command above | Failed during temporary setup with `After removing the zero bases the layout must be bijective` | Harness/setup limitation |
| High-CGA MMAv5 controls | `CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[False-ctas_per_cga1]' 'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[True-ctas_per_cga1]' 'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[False-ctas_per_cga2]' 'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[True-ctas_per_cga2]'` | `4 passed` | Green control |
| High-CGA TMA multicast controls | `CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_core.py::test_tma_multicast_copy[ctas_per_cga1]' 'python/test/gluon/test_core.py::test_tma_multicast_copy[ctas_per_cga2]'` | `2 passed` | Green control |
| Scaled-MMAv5 scale-copy plus MMA controls for 4/8/16 CTA CGA shapes | `CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short 'python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy[False-ctas_per_cga2-mxfp8-mxfp8-128-2048-2048-4096]' 'python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy[False-ctas_per_cga4-mxfp8-mxfp8-128-2048-2048-4096]' 'python/test/gluon/test_core.py::test_mma_scaled_tcgen05_copy[False-ctas_per_cga5-mxfp8-mxfp8-128-2048-2048-4096]'` | `3 passed` | Green mixed copy+MMAv5 control |
| Full saved proxy-fence reproducer | `cd build/cmake.linux-aarch64-cpython-3.12 && bin/triton-opt --run-reproducer /tmp/tmem_high_cga_copy_scales_round14_mixed_fail.mlir` | Failed with `could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses` | Existing `FZ-20260421-0014` |
| Two-CTA matmul consistency, 8/16 CTA contexts | `cd build/cmake.linux-aarch64-cpython-3.12 && bin/triton-opt --split-input-file /tmp/tmem_laneah_mma_twocta_high_cga_consistent.mlir --triton-nvidia-check-matmul-two-cta \| rg 'ttng.two-ctas\|tt.func\|module attributes'` | Passed; both modules gained `"ttng.two-ctas" = true` | Green pass control |
| Saved mixed copy/scales repro through check + raw proxy-fence only | `cd build/cmake.linux-aarch64-cpython-3.12 && bin/triton-opt /tmp/tmem_high_cga_copy_scales_round14_mixed_fail.mlir --triton-nvidia-check-matmul-two-cta --triton-nvidia-gpu-proxy-fence-insertion=compute-capability=103` | Passed and gained `"ttng.two-ctas" = true` | Confirms `FZ-0014` requires the full allocation/proxy-fence reproducer pipeline |

## Counts

- Green Python/Gluon controls: `9` passed.
- Green compiler/pass controls: `2` high-CGA two-CTA matmul modules plus `1`
  raw check+proxy-fence contrast passed.
- Existing `FZ-20260421-0010`: `18` counted rows this lane (`15` direct rows
  from the high-CGA gate probe plus `3` mixed copy/scales high-CGA rows).
- Existing `FZ-20260421-0014`: `2` confirmations (`1` pytest mixed
  copy/scales row plus `1` `triton-opt --run-reproducer` rerun).
- Harness/setup limitations: `10` raw rows (`9` from the inherited gate probe:
  six scales rows with stderr not included in the JSON payload and three
  no-scales 2CTA helper-argument rows; one mixed scales descriptor-view row).
- New candidates: `0`.
- Runtime miscompiles: `0`.
- Backend/compiler repairs: `0`.

## Classification notes

The `FZ-0010` rows continue to fail before lowering with the exact diagnostic:

```text
Layout has X CTAs per CGA, but the context requires Y CTAs per CGA.
```

That held for local `ld/st`, `ld.red`, and no-scales copy rows in 4/8/16 CTA
launch contexts. The temporary scales rows in the inherited gate probe emitted
only the outer parser failure in this rerun's JSON payload, so they are left as
harness-limited here even though earlier Lane R evidence captured the same
underlying CTA-count text on stderr.

The `FZ-0014` row is still a legal `num_ctas=2` mixed module, not a high-CGA
CTA-count failure. The module contains a 2CTA no-scales descriptor-chain copy
and a scales copy with independent mbarrier regions. The full reproducer
pipeline fails in proxy-fence insertion, while the saved raw input passed
through `triton-nvidia-check-matmul-two-cta` plus raw proxy-fence insertion.
This keeps the owner on the full allocation/proxy-fence pipeline state rather
than the matmul two-CTA checker alone.

The MMAv5/TMA-MMA controls stayed green in adjacent 4/8/16 CTA contexts. The
scaled-MMAv5 scale-copy controls also stayed green, so Lane AJ did not find a
new mixed copy+MMAv5 crash or wrong-result case beyond the already cataloged
`FZ-0014` copy/mbarrier composition family.

## New candidates

None.

The next non-overlapping discovery slice should minimize `FZ-0014` against
MMAv5 adjacency directly: start from a legal two-CTA copy/mbarrier reproducer
and add a minimal two-CTA MMAv5 or scaled-MMAv5 op after the copy regions,
checking whether the proxy-fence abort is unchanged, masked, or replaced by a
matmul two-CTA consistency diagnostic.
