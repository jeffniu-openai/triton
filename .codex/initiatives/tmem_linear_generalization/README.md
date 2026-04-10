# TMEM Linear Generalization Initiative

This directory is the durable coordination space for the TMEM linear-layout
generalization project.

Keep this README up to date when the role of any document changes, when a new
current-state handoff supersedes an older one, or when the source-of-truth
entry points change.

## How To Use This Folder

When resuming the initiative:
- start here;
- then read `memory.md` as the initiative's durable memory/source of truth;
- use `gb200_nvidia_ci_inventory.md` for the current GB200/NVIDIA CI baseline
  and confirmed red list;
- use `gb200_failure_manifest.md` plus the adjacent generated `.txt` files
  when you need the exact nodeid inventories for the current branch or the
  merge-base-existing shard-3 subsets;
- use `gb200_branch_recovery_plan.md` when you need the prioritized
  branch-caused recovery backlog and the exact representative repros to drive
  fixes;
- then read the tail of the active handoff for the latest live buckets and next
  concrete steps;
- use `log.md` for chronological provenance and validation history; and
- use `fuzz_plan.md` once the current planner/cleanup blockers are green and
  broad saturation work is the next task.

## Document Roles

- `memory.md`
  - durable memory for the initiative
  - source of truth for mission, current decisions, long-term plan, and current
    undated priorities
  - future sessions should update this when the stable plan or current
    priorities change

- `log.md`
  - append-only chronology of checkpoints, bug classification, validation, and
    conclusions
  - do not rewrite old entries except for obvious factual corrections

- `handoff_2026-04-09.md`
  - current active handoff
  - read its tail first for the latest live buckets, disproved experiments, and
    next implementation order
  - keep it updated at meaningful checkpoints and before likely context
    rollover

- `gb200_nvidia_ci_inventory.md`
  - current GB200/NVIDIA CI baseline for this Blackwell devbox phase
  - records the workflow coverage, confirmed red list, and classification of
    each current failure as stale expectation, rewrite-candidate, or real bug
  - update it before broadening beyond exact nodeids or changing the current
    CI-grounded failure story

- `gb200_failure_manifest.md`
  - index for the generated exact current-branch failure lists and the
    merge-base-existing shard-3 subsets
  - use the adjacent `.txt` files when you need exact nodeids rather than
    prose summaries
  - refresh the manifests whenever the current-branch failure surface changes
    materially

- `gb200_branch_recovery_plan.md`
  - prioritized recovery backlog for restoring GB200 CI parity with main
  - separates real branch regressions from preexisting failures, stale
    negatives, and reinterpret-contract rewrite candidates
  - use it to choose the next fix slice once the GB200 census is current

- `fuzz_plan.md`
  - operational saturation/fuzzing playbook for the broader `tcgen05` surface
  - not the immediate source of truth for current bug-fix priority while the
    shared-planner / legacy-unification / reinterpret-decomposition work is
    still active

- `handoff_2026-03-26.md`
  - historical handoff from an earlier phase of the initiative
  - archival provenance, not current status

- `next_agent_prompt_2026-03-26.md`
  - archived takeover prompt from an earlier managed-session workflow
  - archival only

## Current Session Discipline

- Keep `memory.md`, the active handoff, and this README aligned.
- Treat `memory.md` as memory: if the stable understanding of the project or
  the current undated execution priorities changes, update `memory.md` rather
  than leaving that knowledge only in chat context.
- Keep dated historical analyses in `log.md` and the dated handoff files
  instead of rewriting old reports in place.
