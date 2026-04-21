# Round 52: Example Harness Path Sharpening

Date: 2026-04-21
Branch: `codex/tmem`
Scope: sharpen the `python/examples/gluon` collection/import behavior found during Round 51. This pass is cataloging only: no backend fixes, checked-in tests, examples, or docs were changed.

## Rebuild

Command:

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Per-Example Collection Without `./python/triton_kernels`

Command form:

```bash
PYTHONPATH=.:./python pytest --collect-only -q <example>
```

Results:

| Example | Result |
| --- | --- |
| `python/examples/gluon/01-attention-forward.py` | `64 tests collected` |
| `python/examples/gluon/02-convolution.py` | `48 tests collected` |
| `python/examples/gluon/03-matmul-multicta.py` | `96 tests collected` |
| `python/examples/gluon/04-2cta-block-scale-matmul.py` | `750 tests collected` |
| `python/examples/gluon/05-moe-bmm1-fused-gather.py` | collection error, `ModuleNotFoundError: No module named 'triton_kernels.distributed'` |
| `python/examples/gluon/05-tmem-moe-router.py` | `16 tests collected` |
| `python/examples/gluon/06-tmem-lora-fusion.py` | `16 tests collected` |
| `python/examples/gluon/08-tmem-layout-as-epilogue.py` | `5 tests collected` |

The import search confirmed that only `05-moe-bmm1-fused-gather.py` imports the separate `triton_kernels` package:

```bash
rg -n "triton_kernels|distributed" python/examples/gluon
```

Relevant output:

```text
python/examples/gluon/05-moe-bmm1-fused-gather.py:17:from triton_kernels.distributed import make_expt_dict_uniform
python/examples/gluon/05-moe-bmm1-fused-gather.py:18:from triton_kernels.matmul import (
python/examples/gluon/05-moe-bmm1-fused-gather.py:25:from triton_kernels.numerics import InFlexData, OutFlexData
python/examples/gluon/05-moe-bmm1-fused-gather.py:26:from triton_kernels.numerics_details.mxfp import MXFP_BLOCK_SIZE, downcast_to_mxfp
python/examples/gluon/05-moe-bmm1-fused-gather.py:27:from triton_kernels.swiglu import swiglu_fn
python/examples/gluon/05-moe-bmm1-fused-gather.py:28:from triton_kernels.tensor import (
python/examples/gluon/05-moe-bmm1-fused-gather.py:36:from triton_kernels.tensor_details.dtype import UINT8
python/examples/gluon/05-moe-bmm1-fused-gather.py:37:from triton_kernels.tensor_details.layout import (
python/examples/gluon/05-moe-bmm1-fused-gather.py:41:from triton_kernels.testing import alloc_rand, assert_close
python/examples/gluon/05-moe-bmm1-fused-gather.py:42:from triton_kernels.topk import topk
```

Classification: this is a harness/package-path issue. It is not a TMEM backend compiler crash, verifier gap, unsupported TMEM case, or runtime miscompile.

## Corrected Aggregate Collection

Command:

```bash
PYTHONPATH=.:./python:./python/triton_kernels pytest --collect-only -q python/examples/gluon/*.py
```

Result:

```text
1043 tests collected in 3.31s
```

This equals the path-clean examples plus the 48 fused-gather tests. The corrected path removes the only collection/import error.

## Corrected Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/triton_kernels \
  pytest -q -s --tb=short \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[128-c0]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[1024-c0]' \
  'python/examples/gluon/05-moe-bmm1-fused-gather.py::test_op[4096-c0]' \
  'python/examples/gluon/05-tmem-moe-router.py::test_router_projection_matches_torch[128-32]' \
  'python/examples/gluon/06-tmem-lora-fusion.py::test_lora_down_projection_matches_torch[128-32]' \
  'python/examples/gluon/08-tmem-layout-as-epilogue.py::test_direct_consumer_order_matches_pytorch[64-16]'
```

Result:

```text
6 passed in 16.10s
```

Coverage:
- three path-sensitive fused-gather MoE sizes: `128`, `1024`, and `4096`;
- one compact TMEM router example;
- one compact TMEM LoRA example;
- one layout-as-epilogue example.

## Recommendation

For any aggregate `python/examples/gluon` collection or smoke that includes `05-moe-bmm1-fused-gather.py`, invoke pytest with:

```bash
PYTHONPATH=.:./python:./python/triton_kernels
```

The other seven Gluon examples collect under the ordinary repo path:

```bash
PYTHONPATH=.:./python
```

No new `FZ-*` bucket is needed. This should be tracked as an example harness/documentation invocation requirement, not as a backend issue.
