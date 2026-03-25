# TMEM Fuzz Results

Store JSONL sweep outputs here.

Each record should include:
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

Keep high-signal summaries in `../../log.md`. This directory is for detailed
artifacts that other agents can consume directly.
