# TMEM Experiment Results

Store only compact, reusable result artifacts here. Large raw sweep logs,
per-offset JSONL dumps, and one-off probe transcripts should stay local and
untracked after their conclusions have been summarized in the initiative docs.

For new compact JSONL summaries, each record should include:
- `case_id`
- `family`
- `shape`
- `dtype`
- `layout`
- `view_chain`
- `num_warps`
- `num_ctas`
- `expected_outcome`
- `actual_outcome`
- `diagnostic`
- `ptx_opcodes`
- `llir_opcodes`

Keep high-signal summaries in `../../log.md`, `../../memory.md`, or a focused
closure report. This directory is for small artifacts that other agents can
consume directly, plus duration cache files used by validation runners.
