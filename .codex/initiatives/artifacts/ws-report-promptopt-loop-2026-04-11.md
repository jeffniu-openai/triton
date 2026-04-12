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

- geometric speedup
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
- Workspaces:
  - `r3a1`
  - `r3a2`
  - `r3a3`
- Agents:
  - `Sartre`
  - `Faraday`
  - `Cicero`
- Setup notes:
  - each workspace is a fresh copy of the current mainline snapshot
  - all initiative artifacts were removed from the workspace copy except the revised
    `ws-matmul-performance-report.md`
  - each worker was explicitly told to treat the workspace as the entire world and to use
    workspace-local `PYTHONPATH=python/triton_kernels:python` commands
- Early findings:
  - `r3a1` produced a launch-grid tweak that allowed up to two persistent CTAs per SM on the
    `BLOCK_M=128` path
  - one score pass came back with a *positive arithmetic mean* but a *negative geometric mean* and
    a severe worst regression (`-27%` at one point)
  - that is strong evidence that arithmetic mean alone is too easy to fool when a candidate
    redistributes performance unevenly
  - the loop should therefore treat geometric speedup and worst-regression guardrails as more
    important than arithmetic mean when deciding whether a candidate is real
- Final round outcome:
  - `r3a1`
    - change type: launch-grid tweak allowing up to two CTAs/SM on the `BLOCK_M=128` path
    - result: rejected
    - reason: unstable and not promotable; one pass showed a positive arithmetic mean, but the
      geometric mean was negative and the worst regression was severe
  - `r3a2`
    - change type: no usable candidate produced before shutdown
  - `r3a3`
    - change type: no usable candidate produced before shutdown
- Prompt/report lessons from round 3:
  - arithmetic mean speedup alone is too easy to game or misread
  - non-responsive workers should be culled quickly instead of allowing the loop to stall
  - future rounds should use the revised report and judge candidates primarily by geometric speedup
    plus explicit worst-regression guardrails
- Status: completed

### Round 4

- Report version: after the scoring-policy tightening from rounds 2 and 3
- Workspaces:
  - `r4a1`
  - `r4a2`
  - `r4a3`
- Agents:
  - `Descartes`
  - `Kuhn`
  - `Kepler`
- Setup notes:
  - each workspace is a fresh copy of the current mainline snapshot
  - all initiative artifacts were removed from the workspace copy except the revised
    `ws-matmul-performance-report.md`
  - agents were given slightly different search biases to improve exploration diversity:
    kernel-internal dataflow, epilogue/store-side execution, and low-batch tile/config
  - all three were still instructed to treat the report as the only high-level guidance and to use
    broad evidence rather than narrow spot checks
- Final round outcome:
  - no worker produced a usable candidate before shutdown
  - no worker edited the kernel file before the round was culled
- Prompt/report lessons from round 4:
  - the report was still too open-ended for cold-start isolated workers
  - future prompts need an explicit fast-start protocol rather than only principles and hard rules
  - stalled workers should be terminated quickly and replaced with a more scaffolded prompt
- Status: completed

### Round 5

- Report version: after the fast-start-protocol update
- Planned adjustments:
  - keep the same report-only setup, but give workers a more concrete initial execution protocol
  - require an early sanity command, one narrow hypothesis, one small diff, and more than one
    measurement before any success claim
- Workspaces:
  - `r5a1`
  - `r5a2`
- Agents:
  - `Helmholtz`
  - `Feynman`
- Setup notes:
  - same report-only isolation as earlier rounds
  - prompt now includes an explicit fast-start protocol rather than only principles
  - workers were explicitly told to avoid selector/launch-grid heuristics and to get to one small
    kernel diff quickly
- Early findings:
  - the prompt itself improved, but the workspace recipe still had hidden runtime gaps
  - copied workspaces initially lacked `python/triton/language/extra/cuda/libdevice.py`
  - after fixing that, they still lacked `python/triton/backends/nvidia/`, which made Triton see
    `0 active drivers`
  - so round 5 was primarily useful for discovering that a report-only prompt still depends on a
    runtime-complete sandbox recipe
- Final round outcome:
  - `r5a1`
    - change type: small helper-store micro-optimization plus a local runtime import fix
    - useful signal: the worker finally reached a real kernel diff once the sandbox runtime was
      repaired
    - rejection reason: the candidate bundled a runtime workaround, and the central scoring path was
      still too fragile to treat the worker's local benchmark as authoritative
  - `r5a2`
    - change type: no usable kernel candidate before shutdown
- Prompt/report lessons from round 5:
  - the isolated workers should not own end-to-end benchmarking unless the sandbox runtime is known
    to be complete
  - the workspace recipe must explicitly dereference or overlay Triton runtime symlink targets
  - future rounds should keep workers focused on producing a small kernel diff and leave broad
    scoring to the main agent in a known-good environment
- Status: completed

### Round 6

- Report version: after the sandbox-completeness and central-scoring refinement
- Planned adjustments:
  - create private workspaces with dereferenced Triton runtime symlink targets
  - tell workers to propose one or two small kernel diffs only
  - keep all broad ranking in the central scorer instead of delegating benchmarking to workers
- Setup notes:
  - added [ws-report-promptopt-make-workspace.py](/root/code/triton-ws-opt/.codex/initiatives/artifacts/ws-report-promptopt-make-workspace.py)
    to build private workspaces from the current worktree while overlaying the active Triton runtime
    with symlinks dereferenced
  - validated in a fresh sandbox that:
    - `from triton.runtime import driver; driver.active` succeeds
    - `triton.language.extra.cuda.libdevice.exp` is present
    - the standalone example imports and runs a quick benchmark sanity point
  - also removed a local shadow-package trap in the canonical environment:
    `third_party/nvidia/language/cuda/libdevice/` had been shadowing the real tracked
    `libdevice.py` module
- Workspaces:
  - `r6a1`
  - `r6a2`
- Agents:
  - `James`
  - `Fermat`
- Current prompt differences:
  - workers are now asked for one or two small kernel diffs only
  - runtime/import patching is explicitly forbidden
  - local benchmarking is optional sanity only; central scoring remains authoritative
- Final round outcome:
  - `r6a1`
    - change type: hoist helper-store descriptor pointer/stride/chunk math out of the inner store
    - result: rejected at real kernel compilation
    - reason: introduced a tensor/scalar type mismatch in the helper-store mask path
  - `r6a2`
    - change type: reuse a precomputed reciprocal fill for packed FP8 store fragments
    - result: rejected at real kernel compilation
    - reason: introduced an incompatible `Float2Tensor` broadcast shape
- Prompt/report lessons from round 6:
  - asking for small diffs worked; both workers finally proposed interpretable kernel changes
  - `py_compile` is not an adequate sanity check for Triton/Gluon code
  - future workers need one real post-edit kernel invocation before they report a candidate
- Status: completed

### Round 7

- Report version: after the “real kernel invocation, not just py_compile” update
- Planned adjustments:
  - keep the fixed workspace-maker and proposal-only contract
  - require one actual edited-kernel invocation as a worker-side sanity gate
  - continue using central scoring as the authority for promotion
- Workspaces:
  - `r7a1`
  - `r7a2`
- Agents:
  - `Parfit`
  - `Bernoulli`
- Current prompt differences:
  - workers must supply one real post-edit kernel invocation, not just `py_compile`
  - runtime/import patching remains forbidden
  - central scoring remains authoritative for promotion
- Final round outcome:
  - `r7a1`
    - change type: small helper-store inner-loop arithmetic reshaping
    - local status: compiled and passed a real post-edit kernel invocation
    - independent score: `-0.82%` mean, `-0.82%` geometric, `5 / 8` wins, worst regression
      `-11.07%`
    - result: rejected
  - `r7a2`
    - change type: small packed-store-path reshaping
    - local status: passed a one-point real kernel invocation
    - central outcome: rejected on correctness before performance ranking
    - reason: broad scorer found `176797 / 1474560` mismatches on the `batch=128` reference check
- Prompt/report lessons from round 7:
  - one real post-edit kernel invocation is necessary but not sufficient
  - workers also need a local reference-comparison gate before they report success
  - the broad central scorer should continue to be the only promotion authority, but the worker
    contract is now tighter: syntax check, real invocation, and local reference sanity must all
    pass before a candidate is even worth scoring
- Status: completed

### Round 8

- Report version: after the reference-backed worker sanity-gate update
- Planned adjustments:
  - keep the repaired workspace maker
  - require the new sanity script both before and after the edit
  - continue to forbid runtime/import patching and broad selector rewrites
- Workspaces:
  - `r8a1`
  - `r8a2`
- Agents:
  - `Volta`
  - `Galileo`
- Current prompt differences:
  - workers must run the local reference-backed sanity helper before and after editing
  - workers are restricted to one small kernel diff only
  - central scoring remains authoritative for promotion
- Final round outcome:
  - `r8a1`
    - change type: widened the low-batch `BLOCK_M=64` selector cutoff from `slice_size <= 58` to
      `<= 64`
    - local status: passed the worker sanity gate
    - independent score: `-0.40%` mean, `-0.41%` geometric, `3 / 8` wins, worst regression
      `-1.68%`
    - result: rejected
  - `r8a2`
    - change type: replaced the weight-scale `off_k_w // 64` division with a precomputed stride
      multiply
    - local status: passed the worker sanity gate
    - independent score: `-5.84%` mean, `-7.56%` geometric, `3 / 8` wins, worst regression
      `-46.62%`
    - result: rejected
- Prompt/report lessons from round 8:
  - the current low-batch selector boundary is more coupled to the live kernel than it looks from
    one representative point
  - mathematically exact source-level scalar-arithmetic “simplifications” can still damage Triton
    or downstream codegen badly
  - future workers should spend less time on selector-threshold edits or hand-strength-reduced
    scalar math and more time on small structural hypotheses
- Status: completed
