# Lane R3-C: MMA / Copy Structural Fuzzing Round 3

- Time: 2026-04-21 UTC
- Branch: `codex/tmem`
- Scope: adversarial probes for copy, plain MMA, scaled MMA, membar, and warp-specialized surfaces, emphasizing descriptor-view accumulators, two-CTA layouts in larger CGA contexts, scales layouts with K extremes, and copy `warpx2`/`warpx4` descriptor views.
- Backend edits: none.

## Required Setup

Command:

```bash
make -j8
```

Result: pass. Ninja reported no work to do in `build/cmake.linux-aarch64-cpython-3.12`.

## Inventory / Collection

Copy `warpx2` descriptor-view and clean-negative inventory:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'warpx2'
```

Result: `84/1615` selected.

Plain MMA two-CTA accumulator descriptor-view inventory:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_twocta and (indexed_acc_view or acc_subslice_view_plain_kinds or plain_kinds_use_acc or tma_tf32_b_transposed_descriptor_use_acc)'
```

Result: `62/1615` selected.

Scaled MMA descriptor-view / two-CTA / K-extreme inventory:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_runtime_matrix.py -k 'mma_scaled and (twocta_acc_subslice_view_format or shared_scale_descriptor_view_auto_tmem_copy or indexed_acc_identity_narrow_view_format_use_acc or acc_subslice_view_format_use_acc or acc_tile_permuted_32_bscale)'
```

Result: `48/1615` selected.

Structural-fuzzer copy / two-CTA ld.red / generic-pass anchors:

```bash
PYTHONPATH=.:./python pytest --collect-only -q python/test/gluon/test_tmem_structural_fuzzer.py -k 'copy or ldred-twocta or generic-pass'
```

Result: `8/18` selected.

## Runtime Matrix Sweep

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 1 python/test/gluon/test_tmem_runtime_matrix.py -k 'warpx2 or (mma_twocta and (indexed_acc_view or acc_subslice_view_plain_kinds or plain_kinds_use_acc or tma_tf32_b_transposed_descriptor_use_acc)) or (mma_scaled and (twocta_acc_subslice_view_format or shared_scale_descriptor_view_auto_tmem_copy or indexed_acc_identity_narrow_view_format_use_acc or acc_subslice_view_format_use_acc or acc_tile_permuted_32_bscale))'
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 2 python/test/gluon/test_tmem_runtime_matrix.py -k 'warpx2 or (mma_twocta and (indexed_acc_view or acc_subslice_view_plain_kinds or plain_kinds_use_acc or tma_tf32_b_transposed_descriptor_use_acc)) or (mma_scaled and (twocta_acc_subslice_view_format or shared_scale_descriptor_view_auto_tmem_copy or indexed_acc_identity_narrow_view_format_use_acc or acc_subslice_view_format_use_acc or acc_tile_permuted_32_bscale))'
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 3 python/test/gluon/test_tmem_runtime_matrix.py -k 'warpx2 or (mma_twocta and (indexed_acc_view or acc_subslice_view_plain_kinds or plain_kinds_use_acc or tma_tf32_b_transposed_descriptor_use_acc)) or (mma_scaled and (twocta_acc_subslice_view_format or shared_scale_descriptor_view_auto_tmem_copy or indexed_acc_identity_narrow_view_format_use_acc or acc_subslice_view_format_use_acc or acc_tile_permuted_32_bscale))'
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 PYTHONPATH=.:./python pytest -s --tb=short --splits 4 --group 4 python/test/gluon/test_tmem_runtime_matrix.py -k 'warpx2 or (mma_twocta and (indexed_acc_view or acc_subslice_view_plain_kinds or plain_kinds_use_acc or tma_tf32_b_transposed_descriptor_use_acc)) or (mma_scaled and (twocta_acc_subslice_view_format or shared_scale_descriptor_view_auto_tmem_copy or indexed_acc_identity_narrow_view_format_use_acc or acc_subslice_view_format_use_acc or acc_tile_permuted_32_bscale))'
```

Results:

- Group 1: `49 passed, 1566 deselected`.
- Group 2: `49 passed, 1566 deselected`.
- Group 3: `49 passed, 1566 deselected`.
- Group 4: `47 passed, 1568 deselected`.

No compiler crashes, opcode mismatches, false unsupported diagnostics, or runtime miscompiles were observed.

## Structural-Fuzzer Anchors

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_tmem_structural_fuzzer.py -k 'copy or ldred-twocta or generic-pass'
```

Result: `3 passed, 10 deselected, 5 xfailed`.

The xfails are the existing `FZ-20260421-0001` / `FZ-20260421-0002` generic-pass dynamic-index/control-flow/layout-pressure failures. No new copy, ld.red two-CTA, or generic-pass variant appeared in this selector.

## Larger-CGA Temporary Probes

Initial setup note: a first inline import probe failed before compilation because importing `python/test/gluon/test_tmem_runtime_matrix.py` directly did not include its adjacent `tmem_test_utils` helper on `PYTHONPATH`. It was rerun with `PYTHONPATH=.:./python:python/test/gluon`; that setup failure is not a backend finding.

Copy two-CTA layout in larger CGA contexts:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:python/test/gluon python - <<'PY'
import importlib.util
import sys
import torch
from triton.compiler.errors import CompilationError

spec = importlib.util.spec_from_file_location('rtm', 'python/test/gluon/test_tmem_runtime_matrix.py')
rtm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rtm
spec.loader.exec_module(rtm)

M = 256
N = 64
swizzle = 32
layout = rtm._make_tmem_linear_layout_mmav5_twocta(M, N)
inp = torch.arange(M * N, device='cuda', dtype=torch.float32).reshape(M, N)
out = torch.empty_like(inp)
for num_ctas in (4, 8, 16):
    cga_layout = rtm._make_2cta_cga_layout((num_ctas, 1), (num_ctas, 1), (1, 0), 0)
    try:
        rtm.tmem_copy_no_scales_twocta_kernel[(1,)](
            inp, out, layout, tuple(tuple(basis) for basis in cga_layout), M, N, swizzle,
            num_ctas=num_ctas, num_warps=4,
        )
    except CompilationError as exc:
        text = str(exc)
        expected = f'Layout has 2 CTAs per CGA, but the context requires {num_ctas} CTAs per CGA.'
        if expected not in text or 'Assertion' in text or 'PassManager::run failed' in text:
            raise AssertionError(f'bad diagnostic for num_ctas={num_ctas}: {text[-1200:]}') from exc
        print(f'copy-twocta-layout-in-{num_ctas}cta-context: clean diagnostic')
    else:
        raise AssertionError(f'num_ctas={num_ctas} unexpectedly compiled')
PY
```

Result: pass. Clean diagnostics for `num_ctas=4`, `8`, and `16`.

Plain MMA two-CTA accumulator layout in larger CGA contexts:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:python/test/gluon python - <<'PY'
import importlib.util
import sys
import torch
from triton.compiler.errors import CompilationError
from triton.experimental.gluon import language as ttgl

spec = importlib.util.spec_from_file_location('rtm', 'python/test/gluon/test_tmem_runtime_matrix.py')
rtm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rtm
spec.loader.exec_module(rtm)

kind = 'f16'
block_m = 256
block_n = 64
block_k = 64
for num_ctas in (4, 8, 16):
    ctas_per_cga = [num_ctas, 1]
    ctas_per_cga_b = [ctas_per_cga[0] // 2, 2 * ctas_per_cga[1]]
    cta_split_a = [ctas_per_cga[0], 1]
    cta_split_b = [1, ctas_per_cga_b[1]]
    cta_order = [1, 0]
    cga_layout_a = rtm._make_2cta_cga_layout(ctas_per_cga, cta_split_a, cta_order, 0)
    cga_layout_b = rtm._make_2cta_cga_layout(ctas_per_cga_b, cta_split_b, cta_order, 1)
    cga_layout_c = rtm._make_2cta_cga_layout(ctas_per_cga, ctas_per_cga, cta_order, 0)
    block_layout_a = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [0, 1], cga_layout=cga_layout_a)
    block_layout_b = ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0], cga_layout=cga_layout_b)
    acc_layout = rtm._make_tmem_linear_layout_mmav5_twocta(block_m, block_n)
    a, b, shared_layout_a, shared_layout_b, _, _, _ = rtm._make_mma_twocta_plain_kind_inputs(
        kind, block_m, block_n, block_k, cga_layout_a, cga_layout_b
    )
    out = torch.empty((block_m, block_n), device='cuda', dtype=torch.float32)
    try:
        rtm.mma_kernel[(1,)](
            a, b, out, block_m, block_n, block_k, block_layout_a, block_layout_b,
            tuple(tuple(basis) for basis in cga_layout_c), acc_layout, shared_layout_a, shared_layout_b,
            ttgl.float32, False, True, num_warps=4, num_ctas=num_ctas,
        )
    except CompilationError as exc:
        text = str(exc)
        expected = f'Layout has 2 CTAs per CGA, but the context requires {num_ctas} CTAs per CGA.'
        if expected not in text or 'Assertion' in text or 'PassManager::run failed' in text:
            raise AssertionError(f'bad diagnostic for plain-mma num_ctas={num_ctas}: {text[-1200:]}') from exc
        print(f'plain-mma-twocta-acc-in-{num_ctas}cta-context: clean diagnostic')
    else:
        raise AssertionError(f'plain-mma num_ctas={num_ctas} unexpectedly compiled')
PY
```

Result: pass. Clean diagnostics for `plain-mma-twocta-acc-in-4cta-context`, `8cta`, and `16cta`.

Scaled MMA two-CTA accumulator-subslice/copy layout in larger CGA contexts:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python:python/test/gluon python - <<'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location('rtm', 'python/test/gluon/test_tmem_runtime_matrix.py')
rtm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rtm
spec.loader.exec_module(rtm)

for num_ctas in (4, 8, 16):
    block_m = 256
    block_n = 64
    block_k = 128
    parent_n = 128
    vec_size = 32
    rtm.torch.manual_seed(0)
    a, a_scale, _ = rtm.random_quantized_tensor(block_m, block_k, 'mxfp8')
    b, b_scale, _ = rtm.random_quantized_tensor(block_n, block_k, 'mxfp8')
    a_scale = rtm.swizzle_scales_packed_block(a_scale, vec_size)
    b_scale = rtm.swizzle_scales_packed_block(b_scale, vec_size)
    try:
        rtm.mma_scaled_tcgen05_acc_subslice_copy(
            a, b, a_scale, b_scale, vec_size, block_m, block_n, block_k, parent_n, 0,
            num_ctas=num_ctas, multicast=False,
        )
    except Exception as exc:
        text = str(exc)
        expected = f'Layout has 2 CTAs per CGA, but the context requires {num_ctas} CTAs per CGA.'
        if expected not in text or 'Assertion' in text or 'PassManager::run failed' in text:
            raise AssertionError(f'bad diagnostic for scaled-mma num_ctas={num_ctas}: {text[-1600:]}') from exc
        print(f'scaled-mma-twocta-acc-in-{num_ctas}cta-context: clean diagnostic')
    else:
        raise AssertionError(f'scaled-mma num_ctas={num_ctas} unexpectedly compiled')
PY
```

Result: pass. Clean diagnostics for `scaled-mma-twocta-acc-in-4cta-context`, `8cta`, and `16cta`.

## Warp-Specialized / Membar Probes

Command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 PYTHONPATH=.:./python pytest -s --tb=short python/test/gluon/test_fpsan.py::test_tmem_copy_scales_in_warp_specialize_partition python/test/gluon/test_fpsan.py::test_tmem_store_in_warp_specialize_partition_visible_to_parent
```

Result: `2 passed`.

## Findings

No new backend findings.

Covered surfaces:

- copy `warpx2::{01_23,02_13}` direct, indexed, subslice, slice-index, dense-shared rematerialization, two-CTA positives, two-CTA clean unsupported boundaries, and subword clean errors;
- direct scales copy `warpx4` structural anchors;
- plain MMA two-CTA descriptor-view accumulators, including indexed and subslice accumulator views, `use_acc`, `tf32` transposed descriptor cases, and K-extreme representatives;
- scaled MMA descriptor-view accumulators, two-CTA accumulator subslices, shared scale descriptor-view auto-copy, narrow `N=16` indexed accumulator view, B-scale descriptor views, and K=128/256 representatives;
- two-CTA layouts rejected cleanly in 4/8/16 CTA contexts for copy, plain MMA, and scaled MMA;
- warp-specialized TMEM copy/store plus mbarrier synchronization.

## Recommended Follow-up Tests

- Promote the temporary 8/16 CTA clean-negative probes into checked-in pytest coverage next to the existing `test_tmem_runtime_matrix_cp_no_scales_twocta_layout_in_4cta_context_reports_clean_error`.
- Add analogous checked-in clean-negative coverage for plain MMA and scaled MMA two-CTA accumulator layouts in larger CGA contexts; the temporary probes show the existing diagnostic path is already clean.
- Keep the runtime-matrix selector above as a round-3 regression slice for copy/MMA/scaled-MMA descriptor-view and opcode stability.
