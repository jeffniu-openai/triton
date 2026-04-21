# Round 15 Local: FZ-0015 TTGIR and parent-view discriminator

- Date: 2026-04-21 11:34 UTC
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler fixes were attempted.
- Temporary probes/logs:
  - `/tmp/tmem_fz0015_parent_round15_probe.py`
  - `/tmp/tmem_fz0015_parent_round15_g*.log`
  - `/tmp/tmem_fz0015_parent_round15_v2_g*.log`
  - `/tmp/tmem_fz0015_ttgir_inspect_round15.log`

## Summary

This local discriminator followed the first `FZ-20260421-0015` minimization
slice. It tried to separate independent B-scale allocations from distinct
B-scale descriptor values created from one parent, and inspected TTGIR for the
direct, constexpr, branch, and loop forms.

The parent-view experiment is probe-limited, not new bug evidence: both
`slice(dim=0)` and `reshape((2, N, K/VEC)).index(...)` variants fail before
lowering for all rows, including direct controls. The TTGIR inspection
sharpened `FZ-0015`: failing rows feed scaled MMAv5 with a merged B-scale
memdesc SSA value.

## Parent-view probe

Initial slice form:

```text
b_parent = allocate_tensor_memory(..., [2 * N, K / VEC], TensorMemoryScalesLayout())
b0 = b_parent.slice(0, N, dim=0)
b1 = b_parent.slice(N, N, dim=0)
```

Result: `5` collected; `5` parser failures. Earliest diagnostic:

```text
src and dst must both be of shared memory encoding
```

Revised reshape/index form:

```text
b_parent_rank3 = b_parent.reshape((2, N, K / VEC))
b0 = b_parent_rank3.index(0)
b1 = b_parent_rank3.index(1)
```

Result: `5` collected; `5` front-end/view failures. Diagnostic:

```text
failed to infer memdesc_index result type
unsupported tensor memory memdesc_subslice view; preserved tensor memory view
encoding also failed: rank must be less than or equal to the memdesc rank for
tensor memory
```

Classification: probe-limited. This does not broaden `FZ-0015` because direct
and constexpr controls fail before the scaled-MMAv5 operation is reached.

## TTGIR inspection

Rows inspected:

- `direct_control`, selector `0`;
- `b_constexpr_distinct`, selector `1`;
- `b_runtime_distinct`, selectors `0` and `1`;
- `b_loop_distinct`, selector `0`.

Direct control feeds scaled MMAv5 with a concrete B-scale memdesc:

```text
%23 = ttng.tc_gen5_mma_scaled ..., %a0, %b0, ...
```

Constexpr distinct selection also feeds a concrete B-scale memdesc:

```text
%23 = ttng.tc_gen5_mma_scaled ..., %a0, %b1, ...
```

Runtime branch selection lowers to an `arith.select` between two B-scale
memdesc values, and the selected SSA value feeds scaled MMAv5:

```text
%24 = arith.select %23, %b_sel_17, %b_sel
%26 = ttng.tc_gen5_mma_scaled ..., %a0, %24, ...
```

Loop selection lowers to an `scf.for` carrying a B-scale memdesc iter_arg, and
the loop result feeds scaled MMAv5:

```text
%b_sel_20 = scf.for ... iter_args(%b_sel_23 = %b_sel)
%23 = ttng.tc_gen5_mma_scaled ..., %a0, %b_sel_20, ...
```

Observed output diagnostics during inspection remained NaN/Inf-heavy for the
runtime-selected rows, while direct/constexpr controls had no NaNs or Infs.

## Classification

`FZ-20260421-0015` remains distinct and is now sharper:

- direct and constexpr B-scale descriptors feed scaled MMAv5 correctly;
- same-object dynamic B-scale selection passed in the prior minimization;
- branch-contained scaled MMA passed in the prior minimization;
- branch and loop failures both have a merged B-scale memdesc SSA value feeding
  `ttng.tc_gen5_mma_scaled`;
- parent-view discrimination is blocked by current tensor-memory
  `memdesc_subslice` view support, so it is not evidence for or against the
  independent-allocation hypothesis.

Next discovery slice: generate a small TTGIR/MLIR reproducer for the
`arith.select` B-scale memdesc form and inspect the lowering/planner path that
consumes the selected scale memdesc operand. Keep backend repair deferred until
that minimization is complete.
