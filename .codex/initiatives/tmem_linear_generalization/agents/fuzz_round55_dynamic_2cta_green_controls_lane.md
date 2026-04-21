# Round 55 Dynamic 2CTA Green-Controls Lane A

Date: 2026-04-21 15:35 UTC
Branch: `codex/tmem`
Checkpoint base: `9ae757492`

Scope: follow up Round 54 dynamic two-CTA descriptor SSA without entering the
known chain-0 static-failing shapes. This lane targeted legal two-CTA dynamic
descriptor selection over chain-1 and direct-index controls, plus adjacent
copy, plain MMAv5 accumulator, and scaled/copy checked-in controls. No backend
repairs were attempted.

## Preflight

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

## Temporary probe

Probe path:

```text
/tmp/tmem_round55_dynamic_2cta_green_controls_probe.py
```

Summary artifact:

```text
/tmp/tmem_round55_dynamic_2cta_green_controls_summary.json
```

Case schema:

```text
case_id, seed, family, kind, row_kind, col_kind, n, selector, chain_id,
instr_variant, loop_count
```

Common shape policy:

- `M=256`
- `num_ctas=2`
- `num_warps=4`
- two-CTA lifted `TensorMemoryLinearLayout` parent for dynamic `ld/st`
- two-CTA MMAv5-equivalent lifted parent for dynamic copy rows
- `chain_id=1` means the Round 54 green
  `reshape((M/2,2,N/2,2))->permute->permute->reshape` identity chain
- `chain_id=2` means direct parent index with no descriptor-view chain
- known Round 54 chain-0 static-failing shapes were intentionally excluded

Syntax check:

```bash
PYTHONPATH=.:./python python3 -m py_compile /tmp/tmem_round55_dynamic_2cta_green_controls_probe.py
```

Result: passed.

First run exposed that Python exception text only preserved
`PassManager::run failed` for existing illegal-`ttg.memdesc_index` rows, so the
temporary classifier was adjusted to map runtime-index and copy dynamic-if rows
with that failure mode to existing `FZ-20260421-0001`.

Final run:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest -vv -s --tb=short /tmp/tmem_round55_dynamic_2cta_green_controls_probe.py
```

Result:

```text
1 passed in 4.70s
```

The single pytest item internally ran eight generated cases and wrote the JSON
summary.

## Generated case results

| Case | Seed | Parameters | Result | Classification |
| --- | ---: | --- | --- | --- |
| `r55-ldst-if-chain1-mmav5-sel0` | `0x550001` | `ldst`, dynamic if, reverse rows, identity cols, `N=128`, selector `0`, chain-1, `16x128b` | pass; PTX/LLIR both four `st.16x128b.x32` plus two `ld.16x128b.x32` | green |
| `r55-ldst-if-chain1-block-sel1` | `0x550002` | `ldst`, dynamic if, identity rows/cols, `N=64`, selector `1`, chain-1, `32x32b` | pass; PTX/LLIR both two `st.32x32b.x64` plus one `ld.32x32b.x64` | green |
| `r55-ldst-mixed-chain1-evenodd-colrev` | `0x550003` | `ldst`, mixed tensor/memdesc capture, even/odd rows, reverse cols, `N=64`, selector `1`, chain-1, `32x32b` | pass; PTX/LLIR both two `st.32x32b.x64` plus one `ld.32x32b.x64` | green |
| `r55-ldst-loop-chain1-no-take` | `0x550004` | `ldst`, loop-carried descriptor, identity rows, reverse cols, `N=128`, selector `4`, chain-1, `16x128b`, loop count `2` | pass; PTX/LLIR both four `st.16x128b.x32` plus two `ld.16x128b.x32` | green |
| `r55-ldst-direct-if-block-sel1` | `0x550005` | `ldst`, dynamic if, identity rows/cols, `N=64`, selector `1`, direct parent index, `32x32b` | pass; PTX/LLIR both two `st.32x32b.x64` plus one `ld.32x32b.x64` | green |
| `r55-ldst-direct-runtime-index-sel1` | `0x550006` | `ldst`, runtime `parent.index(selector)`, identity rows/cols, `N=64`, selector `1`, direct parent index, `32x32b` | LLVM conversion failure; stderr shows illegal `ttg.memdesc_index` | existing `FZ-20260421-0001` |
| `r55-copy-direct-if-mmav5-sel1` | `0x550007` | `copy`, dynamic if selecting MMAv5 two-CTA parent index, `N=64`, selector `1` | LLVM conversion failure; stderr shows illegal `ttg.memdesc_index` yielded from branch into `ttng.tmem_copy` | existing `FZ-20260421-0001` |
| `r55-copy-direct-if-mmav5-sel0` | `0x550008` | `copy`, dynamic if selecting MMAv5 two-CTA parent index, `N=128`, selector `0` | LLVM conversion failure; stderr shows illegal `ttg.memdesc_index` yielded from branch into `ttng.tmem_copy` | existing `FZ-20260421-0001` |

Aggregate: `8` generated rows: `5` pass, `3` existing
`FZ-20260421-0001`, `0` new independent `FZ-*`.

## Checked-in adjacent controls

Collection:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest --collect-only -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_twocta_linear_indexed_view or mma_twocta_indexed_acc_view or mma_scaled_twocta_acc_subslice_view_format_use_acc or cp_scales_warpx4_twocta_direct_copy'
```

Result:

```text
47/1615 tests collected (1568 deselected)
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_twocta_linear_indexed_view or mma_twocta_indexed_acc_view or mma_scaled_twocta_acc_subslice_view_format_use_acc or cp_scales_warpx4_twocta_direct_copy'
```

Result:

```text
47 passed, 1568 deselected in 3.46s
```

Covered controls:

- static two-CTA no-scale copy indexed views, including full-size OOR clean
  boundary and the two-CTA scales copy direct control
- static plain-MMAv5 two-CTA indexed accumulator views across legacy, linear,
  and unit-parent linear layouts, including use-acc rows
- static scaled-MMAv5 two-CTA accumulator subslice/copy controls

## Classification

No new independent `FZ-*` was found. Chain-1 and direct-branch `ld/st` dynamic
selection is green for the probed legal two-CTA shapes, which confirms the
Round 54 wrong results were not a generic two-CTA dynamic-selection failure.
The remaining dynamic direct runtime-index and copy branch rows reproduce
existing `FZ-20260421-0001` illegal `ttg.memdesc_index` at LLVM conversion.

Backend repair remains deferred.
