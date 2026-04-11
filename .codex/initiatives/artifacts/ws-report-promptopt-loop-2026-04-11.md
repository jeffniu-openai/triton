# WS Report Prompt-Optimization Loop

## Goal

Use the report at
[ws-matmul-performance-report.md](/root/code/triton-ws-opt/.codex/initiatives/artifacts/ws-matmul-performance-report.md)
as the primary “prompt” for isolated agents with no prior context. Each agent gets a private
workspace copy of the repo snapshot plus only that report artifact. The agent is asked to improve
the standalone Gluon example kernel, and the resulting code is scored independently against the
current baseline.

The loop objective is to improve the report itself:

- if agents find improvements, identify which parts of the report enabled them
- if agents stall or chase bad directions, revise the report so the next round does better

## Initial Scoring Policy

Representative batches for hillclimbing:

- `128`
- `256`
- `512`
- `1024`
- `2048`
- `8192`
- `16384`
- `31744`

Primary score:

- mean relative speedup vs the current mainline example across those batches

Secondary metrics:

- number of winning points
- worst regression
- exact correctness vs reference on the same prepared inputs

Evaluation is done from a central scorer that:

- prepares inputs using the current mainline example
- times the current mainline example and the candidate on the same prepared inputs
- validates the candidate against the production reference

This prevents agents from “winning” by changing local benchmark helpers or data generation.

## Rounds

### Round 1

- Report version: `be3a017ef0`
- Workspaces:
  - `r1a1`
  - `r1a2`
- Scorer:
  - `/root/code/triton-ws-opt/.codex/initiatives/artifacts/ws-report-promptopt-eval.py`
  - representative batches: `128, 256, 512, 1024, 2048, 8192, 16384, 31744`
  - quick ranking rep count: `200`
- Workspace setup:
  - copied repo snapshot under `/root/code/promptopt/`
  - removed all other initiative artifacts from each workspace
  - copied only `ws-matmul-performance-report.md` into each workspace artifact directory
- Baseline scorer sanity:
  - candidate = current mainline example itself
  - result: effectively zero, as expected
  - mean speedup vs self: `-0.0032%`
  - worst point vs self: `-0.0897%`
- Agent results:
  - `r1a1`
    - change type: occupancy-aware selector fallback
    - independent score: `-0.0048%` mean vs baseline, effectively noise-flat
    - note: no measured win; rationale was heuristic and occupancy-driven
  - `r1a2`
    - change type: host-side slice-statistics selector rewrite
    - result: rejected
    - reason: broke cudagraph capture by calling `.cpu()` in the hot path
- Prompt/report lessons from round 1:
  - the report did not yet make cudagraph-safety explicit enough
  - the report still allowed agents to spend effort on unmeasured selector heuristics
  - the next report revision must make “measured wins only” a hard rule and explicitly demote
    occupancy-first or selector-theory-first changes
- Status: completed

### Round 2

- Report version: working tree after the “Hard Rules” / round-1-failures update
- Planned adjustments:
  - emphasize cudagraph-safe hot-path constraints
  - explicitly ban unmeasured selector theory as a first move
  - require scorer-verified improvements as the acceptance contract
- Agent results:
  - `r2a1`
    - change type: none landed
    - result: effectively stalled
    - reason: the agent spent time orienting itself and initially hit workspace import-path issues;
      it never produced a code candidate worth scoring
  - `r2a2`
    - change type: helper-depth policy tied to row-subtile factor
    - agent-local evidence: looked healthy at two large points (`8192`, `16384`)
    - independent score: `-0.0677%` mean vs baseline, `3 / 8` wins, worst regression
      `-0.4463%`
- Prompt/report lessons from round 2:
  - the report still needed to say explicitly that two-point spot checks are not enough
  - selector-only or buffering-only heuristics can look good at large points and still lose on the
    representative sweep
  - the next report revision should promote the central scorer over any narrow hand-picked timing
- Status: completed

### Round 3

- Report version: working tree after the round-2-failures update
- Planned adjustments:
  - add an explicit hard rule against two-point spot checks for selector or buffering policy changes
  - make “broad-sweep scorer first, narrative second” even more explicit
  - respawn fresh isolated workers from a clean workspace snapshot with the revised report only
- Status: pending
