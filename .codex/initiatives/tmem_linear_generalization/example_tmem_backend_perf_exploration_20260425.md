# Example 01/05 TMEM Backend Performance Exploration - 2026-04-25

## Scope

Explored branch-local ways to use the generalized TMEM backend for the Gluon
examples:

- `python/examples/gluon/01-attention-forward.py`
- `python/examples/gluon/05-moe-bmm1-fused-gather.py`

Promotion rule for kernel changes in this pass: final output tensors must stay
bitexact relative to the old branch source on the same prepared inputs. The 01
attention `M` buffer is not bit-stable for some existing baseline rows even
when rerunning the same unmodified selector; output tensors are bit-stable and
were used as the practical equivalence gate for selector performance changes.

## Promoted Change

### 01 Attention, Noncausal D64 FP8 on Blackwell Ultra

The old selector used `NUM_KV_BUFFERS=2` at `N_CTX <= 1024` and `8` above that
for noncausal D64 FP8. On this branch, `NUM_KV_BUFFERS=4` is consistently faster
across the benchmark sequence-length range while preserving output bits.

Focused same-input A/B against `HEAD` before the selector change:

| N_CTX | old KV buffers | new KV buffers | old TFLOPS | new TFLOPS | ratio | output bitexact |
|---:|---:|---:|---:|---:|---:|:---:|
| 1024 | 2 | 4 | 858.55 | 864.39 | 1.0068x | yes |
| 2048 | 8 | 4 | 1117.51 | 1123.40 | 1.0053x | yes |
| 4096 | 8 | 4 | 1163.47 | 1171.79 | 1.0072x | yes |
| 8192 | 8 | 4 | 1238.41 | 1246.16 | 1.0063x | yes |
| 16384 | 8 | 4 | 1250.29 | 1258.03 | 1.0062x | yes |
| 32768 | 8 | 4 | 1255.74 | 1263.27 | 1.0060x | yes |
| 65536 | 8 | 4 | 1259.38 | 1267.19 | 1.0062x | yes |

The selector change is scoped to Blackwell Ultra FP8 D64 noncausal. The analogous
FP16 D64 noncausal sweep was mixed: `KV=4` lost at `N_CTX=1024`, was nearly flat
around `2048..8192`, and only showed a small win at `16384`; no FP16 selector
change was promoted.

## 05 Fused Gather Exploration

### Removing the Epilogue Convert Layout

The current 05 epilogue converts each post-quantization `i16` FP8-pair fragment
to the store layout before packing pairs into `i32` stores. Directly removing
that convert is not valid: `pack_fp8x4` then fails because the native fragment
layout does not provide two elements per thread along the last dimension.

Tested alternatives:

- Direct `i16` stores from the native post-quantization layout. This compiled
  and was output-stable, but slowed representative rows: batch 4096 `1796.43`
  TFLOPS versus baseline `1852.52`, batch 8192 `2438.00` versus `2532.35`, and
  batch 31744 `3324.55` versus `3439.58`.
- Store-compatible accumulator layout `[2,2]`. This broke the repeated M-subtile
  split used by the SwiGLU epilogue.
- Wider-N split layout `[1,8]`. This compiled and was bit-stable, but it left the
  same eight packed-output `ttg.convert_layout` ops in TTGIR and was slower on
  representative rows.
- `SWIGLU_SUBTILE_FACTOR` sweep. Legal factors were bitexact, but differences
  were sub-percent and inconsistent across batch sizes. The current large-batch
  factor remains best or tied; no selector change was promoted.
- Forcing the epilogue warp-N split was bitexact but neutral to slightly worse;
  no selector change was promoted.

Conclusion: the visible 05 convert layout is doing necessary final store-packing
layout work. Removing it needs a different packing strategy, likely a custom
cross-lane `i16 -> i32` pack/store path or a compiler/layout enhancement that
materializes the post-quantization fragment directly in the store-pack layout.
The simple source-level variants either failed to compile or lost performance.

## 01 Attention Dead Ends

- `qk_layout=auto` produced tiny output-equivalent speed changes but changed the
  returned `M` values on rows where baseline `M` is already not repeat-stable.
  It was not promoted.
- Explicit QK `32x32b` was neutral to slightly worse than `auto` and not better
  than the original split-N layout.
- O-half `32x32b` was slower; O-half `16x64b` failed direct store lowering in the
  correction path.
- Alternate scalar TMEM state layout `[16,2]` changed output bits and was slower.
- KV-buffer and EX2-turnstile sweeps showed the promoted D64 FP8 `KV=4` win;
  other rows were neutral, slower, OOR, or only output-equivalent without a
  stable performance win.

## Validation Commands

```bash
make -j8
python3 -m py_compile python/examples/gluon/01-attention-forward.py
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-tmem-opt-01-kv4-final PYTHONPATH=/root/code/triton/python python3 <same-input A/B script>
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-tmem-opt-01-pytest PYTHONPATH=/root/code/triton/python pytest -q -s --tb=short 'python/examples/gluon/01-attention-forward.py::test_op[False-triton-fp8-False-64-4096-32-4]' 'python/examples/gluon/01-attention-forward.py::test_op[True-triton-fp8-False-64-4096-32-4]'
```

Results:

- `make -j8`: no work to do, build current.
- Same-input A/B: all D64 FP8 noncausal benchmark sequence lengths output-bitexact; `KV=4` speedup `1.0053x..1.0072x`.
- Focused pytest rows: `2 passed in 6.41s`.
