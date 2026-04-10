# GPT-OSS-120B MM1 Shapes

## Sources

- OpenAI GPT-OSS 120B original config:
  - `https://huggingface.co/openai/gpt-oss-120b/raw/main/original/config.json`
  - downloaded locally as `gpt-oss-120b-original-config.json`
- OpenAI GPT-OSS announcement page:
  - `https://openai.com/index/introducing-gpt-oss/`
  - downloaded locally as `openai-introducing-gpt-oss.html`

## Extracted Geometry

From `gpt-oss-120b-original-config.json`:

- `num_experts = 128`
- `experts_per_token = 4`
- `hidden_size = 2880`
- `intermediate_size = 2880`

## Derived MM1 Shape

This example is the first expert MLP projection (`MM1`) fused with gather and SwiGLU. For SwiGLU, the first projection emits `2 * intermediate_size` channels before the reduction:

- input K = `hidden_size = 2880`
- output N = `2 * intermediate_size = 5760`

So the benchmarked expert-local MM1 weight shape is:

- `B = (2880, 5760)`

## Why This Differs From `bench_mlp.py`

`python/triton_kernels/bench/bench_mlp.py` uses `dim1=5760, dim2=5760` in its `gpt-oss` example invocation. That benchmark covers a broader end-to-end MoE MLP path, not this specific fused-gather MM1 example.

For this standalone fused-gather MM1 example, the official GPT-OSS config is the better source of truth, so the example uses:

- `num_experts = 128`
- `experts_per_token = 4`
- `MM1 B = (2880, 5760)`

## Batch Sweep

The batch sweep reuses the `batch_per_expert` progression from `bench_mlp.py`:

- `batch_per_expert = 4 .. 992`

Mapped back to total token batch size for GPT-OSS 120B MM1:

- `batch_size = batch_per_expert * num_experts / experts_per_token`
- `batch_size = batch_per_expert * 32`
- total sweep range: `128 .. 31744`

## Outputs

The benchmark outputs in this directory are:

- `GPT-OSS-120B MoE MM1 E=128 EP=8 B=2880x5760.csv`
- `GPT-OSS-120B MoE MM1 E=128 EP=8 B=2880x5760.png`
- `results.html`
