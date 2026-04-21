# Round 36 Local: High-Rank `ld.red` Compile/Execute Classifier

Date: 2026-04-21 14:25 UTC
Branch: `codex/tmem`
Scope: discovery-only temporary probe. No backend or checked-in test source was
modified.

## Objective

Probe high-rank `tcgen05.ld.red` descriptor-view lowering for compiler
crashes, process aborts, and false unsupported diagnostics. Checked-in coverage
is currently thin:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_runtime_matrix.py \
  -k 'ld_red and (higher_rank or rank5 or multidim)'
```

Result:

```text
no tests collected (1615 deselected)
```

## Probe

Temporary probe:

```text
/tmp/tmem_high_rank_ldred_round35_probe.py
```

The probe ran one subprocess per case with stable per-GPU cache directories.
It allocated rank-5 TMEM parents and consumed rank-reduced descriptor views via
`view.load_min/load_max`.

Axes:

- parent families: `[1, 1, 1, M, N]`, `[1, 2, 1, M, N]`, `[2, 1, 2, M, N]`;
- layout families: identity, row-reversed, column-reversed;
- `M=128`, `N in {64, 128}`;
- `instr_variant in {auto, 32x32b}`;
- `red_op in {min, max}`.

Total matrix: `72` cases.

## Command

```bash
PYTHONPATH=.:./python python -m py_compile /tmp/tmem_high_rank_ldred_round35_probe.py
PYTHONPATH=.:./python python /tmp/tmem_high_rank_ldred_round35_probe.py \
  | tee /tmp/tmem_high_rank_ldred_round35_probe.log
```

Result:

```text
SUMMARY {"pass": 72}
```

## Classification

No compiler crash, process abort, false unsupported diagnostic, or runtime
exception was observed.

Important limitation: this lane is a compile/execute classifier only. An
initial version compared the returned full tile against the original input, but
that oracle was invalid for the chosen reshape/permute chains and was
discarded. This report therefore does not claim runtime-correctness or
miscompile absence for the full-tile output mapping.

No new independent `FZ-*` bucket was found.
