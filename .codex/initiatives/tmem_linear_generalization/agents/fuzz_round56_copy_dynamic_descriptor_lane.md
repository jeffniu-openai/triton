# Round 56 Copy Dynamic Descriptor Lane A

Date: 2026-04-21 15:51 UTC
Branch: `codex/tmem`
Checkpoint base: `7fa0fdae3`

Scope: targeted `tcgen05.copy` descriptor SSA/control-flow boundaries where
static controls are green. This lane covered direct index, runtime index,
branch-selected concrete TMEM views, descriptor slice/index chains, shared
source subslices, `128x128b`, `128x256b`, `warpx2::01_23`, `warpx2::02_13`,
1CTA, and 2CTA. No backend repairs were attempted.

## Preflight

```bash
make -j8
```

Result:

```text
ninja: no work to do.
```

Preexisting dirty worktree note: central docs and two other Round 56 reports
were already modified or untracked before this lane report was written. This
lane did not edit those other reports.

## Temporary Probe

Probe path:

```text
/tmp/tmem_round56_copy_dynamic_descriptor_lane_probe.py
```

Summary artifact:

```text
/tmp/tmem_round56_copy_dynamic_descriptor_lane_summary.json
```

Syntax check:

```bash
PYTHONPATH=.:./python \
python3 -m py_compile /tmp/tmem_round56_copy_dynamic_descriptor_lane_probe.py
```

Result: passed.

Runtime command:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest -vv -s --tb=short /tmp/tmem_round56_copy_dynamic_descriptor_lane_probe.py
```

Result:

```text
1 passed in 16.18s
```

The single pytest item internally ran `23` generated rows and wrote the JSON
summary.

## Temporary Probe Results

Aggregate:

```text
23 generated rows:
  15 green
   7 existing FZ-20260421-0001
   1 clean unsupported
   0 new independent FZ-*
```

By atom:

```text
128x128b:      3 green, 2 existing FZ-20260421-0001
128x256b:      6 green, 5 existing FZ-20260421-0001
warpx2_01_23:  4 green
warpx2_02_13:  2 green, 1 clean unsupported
```

| Case | Seed | Parameters | Result | Classification |
| --- | ---: | --- | --- | --- |
| `r56-128x128-static-index-1cta` | `0x560001` | static TMEM index, `128x128b`, 1CTA, selector 1 | pass; one `tcgen05.cp.cta_group::1.128x128b` | green |
| `r56-128x128-branch-index-1cta-sel1` | `0x560002` | branch-selected `parent.index(0/1)` TMEM dest, `128x128b`, 1CTA, selector 1 | LLVM conversion failure with illegal `ttg.memdesc_index` | existing `FZ-20260421-0001` |
| `r56-128x128-slice-index-1cta` | `0x560003` | static `parent.slice(...).index(0)` TMEM dest, `128x128b`, 1CTA | pass; one `tcgen05.cp.cta_group::1.128x128b` | green |
| `r56-128x256-static-index-1cta` | `0x560004` | static TMEM index, `128x256b`, 1CTA | pass; sixteen `tcgen05.cp.cta_group::1.128x256b` | green |
| `r56-128x256-branch-index-1cta-sel0` | `0x560005` | branch-selected `parent.index(0/1)` TMEM dest, `128x256b`, 1CTA, selector 0 | LLVM conversion failure with illegal `ttg.memdesc_index` | existing `FZ-20260421-0001` |
| `r56-128x256-runtime-index-1cta-sel1` | `0x560006` | runtime `parent.index(ttgl.load(selector))` TMEM dest, `128x256b`, 1CTA, selector 1 | LLVM conversion failure with illegal `ttg.memdesc_index` | existing `FZ-20260421-0001` |
| `r56-128x256-static-subslice-1cta` | `0x560007` | static TMEM column subslice dest, `128x256b`, 1CTA | pass; sixteen `tcgen05.cp.cta_group::1.128x256b` | green |
| `r56-128x256-branch-subslice-1cta-sel1` | `0x560008` | branch-selected TMEM column subslice dest, `128x256b`, 1CTA, selector 1 | parse/LLVM failure on dynamic descriptor view feeding copy | existing `FZ-20260421-0001` |
| `r56-128x256-shared-static-subslice` | `0x560009` | static shared-source subslice feeding static TMEM dest, `128x256b`, 1CTA | pass; sixteen `tcgen05.cp.cta_group::1.128x256b` | green |
| `r56-128x256-shared-branch-subslice-sel1` | `0x56000a` | branch-selected shared-source subslice feeding static TMEM dest, `128x256b`, 1CTA, selector 1 | pass; sixteen `tcgen05.cp.cta_group::1.128x256b` | green |
| `r56-warpx2-01-static-index-1cta` | `0x56000b` | static TMEM index, `warpx2::01_23`, 1CTA | pass; one `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b` | green |
| `r56-warpx2-01-branch-index-1cta-sel1` | `0x56000c` | branch-selected concrete TMEM index, `warpx2::01_23`, 1CTA, selector 1 | pass; one `tcgen05.cp.cta_group::1.warpx2::01_23.64x128b` | green |
| `r56-warpx2-02-static-index-1cta` | `0x56000d` | static TMEM index, `warpx2::02_13`, 1CTA | pass; one `tcgen05.cp.cta_group::1.warpx2::02_13.64x128b` | green |
| `r56-warpx2-02-branch-index-1cta-sel0` | `0x56000e` | branch-selected concrete TMEM index, `warpx2::02_13`, 1CTA, selector 0 | pass; one `tcgen05.cp.cta_group::1.warpx2::02_13.64x128b` | green |
| `r56-128x256-static-index-2cta` | `0x56000f` | static TMEM index, `128x256b`, 2CTA | pass; sixteen `tcgen05.cp.cta_group::2.128x256b` | green |
| `r56-128x256-branch-index-2cta-sel1` | `0x560010` | branch-selected `parent.index(0/1)` TMEM dest, `128x256b`, 2CTA, selector 1 | LLVM conversion failure with illegal `ttg.memdesc_index` | existing `FZ-20260421-0001` |
| `r56-128x256-static-subslice-2cta` | `0x560011` | static TMEM column subslice dest, `128x256b`, 2CTA | pass; sixteen `tcgen05.cp.cta_group::2.128x256b` | green |
| `r56-128x256-branch-subslice-2cta-sel0` | `0x560012` | branch-selected TMEM column subslice dest, `128x256b`, 2CTA, selector 0 | parse/LLVM failure on dynamic descriptor view feeding copy | existing `FZ-20260421-0001` |
| `r56-128x128-static-index-2cta` | `0x560013` | static TMEM index, `128x128b`, 2CTA | pass; one `tcgen05.cp.cta_group::2.128x128b` | green |
| `r56-128x128-branch-index-2cta-sel0` | `0x560014` | branch-selected `parent.index(0/1)` TMEM dest, `128x128b`, 2CTA, selector 0 | LLVM conversion failure with illegal `ttg.memdesc_index` | existing `FZ-20260421-0001` |
| `r56-warpx2-01-static-index-2cta` | `0x560015` | static TMEM index, `warpx2::01_23`, 2CTA | pass; one `tcgen05.cp.cta_group::2.warpx2::01_23.64x128b` | green |
| `r56-warpx2-01-branch-index-2cta-sel1` | `0x560016` | branch-selected concrete TMEM index, `warpx2::01_23`, 2CTA, selector 1 | pass; one `tcgen05.cp.cta_group::2.warpx2::01_23.64x128b` | green |
| `r56-warpx2-02-static-index-2cta` | `0x560017` | static TMEM index, `warpx2::02_13`, 2CTA | clean diagnostic: no compatible shared-memory descriptor plan | clean unsupported |

Interpretation:

- Dynamic TMEM destination selection for dense `tcgen05.copy` remains the
  existing `FZ-20260421-0001` illegal-`ttg.memdesc_index` boundary. This now
  covers `128x128b`, `128x256b`, 1CTA, 2CTA, branch-selected direct indices,
  runtime direct indices, and branch-selected TMEM column subslices.
- Branch-selected shared-source subslices are green when the TMEM destination
  is static; the failure is on dynamic tensor-memory destination descriptors,
  not generic branch-carried shared memdescs.
- `warpx2::01_23` branch-selected concrete TMEM index views are green for
  1CTA and 2CTA. Single-CTA `warpx2::02_13` branch-selected concrete views are
  green. Two-CTA `warpx2::02_13` remains a clean unsupported static boundary.

## Checked-In Selector

Collection:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest --collect-only -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2 or cp_no_scales_linear_indexed_view or cp_no_scales_twocta_linear_indexed_view or cp_no_scales_linear_subslice_view or cp_no_scales_twocta_linear_subslice_view or cp_no_scales_twocta_128x128b_codegen or cp_128x128'
```

Result:

```text
127/1615 tests collected (1488 deselected) in 1.69s
```

Execution:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python \
pytest -q -s --tb=short python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'cp_no_scales_warpx2 or cp_no_scales_linear_indexed_view or cp_no_scales_twocta_linear_indexed_view or cp_no_scales_linear_subslice_view or cp_no_scales_twocta_linear_subslice_view or cp_no_scales_twocta_128x128b_codegen or cp_128x128'
```

Result:

```text
127 passed, 1488 deselected in 9.52s
```

## Classification

No new independent `FZ-*` was found. Existing `FZ-20260421-0001` is broadened
to additional `tcgen05.copy` dynamic TMEM destination cases. Static copy
controls and checked-in copy selectors stayed green, and the only non-green
non-`FZ-0001` row was the existing clean unsupported 2CTA
`warpx2::02_13` boundary.

Backend repair remains deferred.
