# Round 12 Lane R: high-CGA CTA-count gate minimization

Date: 2026-04-21
Lane: R
Scope: discovery-only minimization and adversarial expansion of `FZ-20260421-0010`.
No backend fixes attempted.

## Build gate

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

## Fresh-process high-CGA probe

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_high_cga_gate_round12_probe.py | tee /tmp/tmem_high_cga_gate_round12_probe.log
```

The first attempt had a harness-only typo in the temporary script seed literal
(`0xCGA1200`) and produced no backend evidence. I fixed the seed to
`0xC9A1200` and reran the full matrix above.

The probe launched every row in a fresh subprocess and used the stable GPU-1
cache. It covered `num_ctas in {4, 8, 16}` for these local 1CTA/2CTA TMEM
operations:

- 1CTA `TensorMemoryLinearLayout` ld/st descriptor-view chain.
- 2CTA `TensorMemoryLinearLayout` ld/st descriptor-view chain.
- 1CTA direct `ld.red` over `TensorMemoryLinearLayout`.
- 2CTA direct `ld.red` over `TensorMemoryLinearLayout`.
- 1CTA `tcgen05.cp` no-scales copy over `TensorMemoryLinearLayout`.
- 2CTA `tcgen05.cp` no-scales copy over `TensorMemoryLinearLayout`.
- 1CTA and 2CTA `TensorMemoryScalesLayout` warpx4 copy rows.

Summary of the rerun:

```text
FZ-20260421-0010: 15
unclassified/harness rows: 9
```

After the full run, I fixed the 2CTA no-scales helper choice and reran those
three rows with `rt._make_tmem_linear_layout_block(..., two_ctas=True)`. Those
three rows also reproduce `FZ-20260421-0010`, bringing confirmed minimized
evidence to 18 rows.

Confirmed `FZ-20260421-0010` rows:

```text
ldst_linear_1cta_chain1, num_ctas=4  -> Layout has 1 CTAs per CGA, context requires 4
ldst_linear_1cta_chain1, num_ctas=8  -> Layout has 1 CTAs per CGA, context requires 8
ldst_linear_1cta_chain1, num_ctas=16 -> Layout has 1 CTAs per CGA, context requires 16
ldst_linear_2cta_chain1, num_ctas=4  -> Layout has 2 CTAs per CGA, context requires 4
ldst_linear_2cta_chain1, num_ctas=8  -> Layout has 2 CTAs per CGA, context requires 8
ldst_linear_2cta_chain1, num_ctas=16 -> Layout has 2 CTAs per CGA, context requires 16
ldred_linear_1cta_direct, num_ctas=4  -> Layout has 1 CTAs per CGA, context requires 4
ldred_linear_1cta_direct, num_ctas=8  -> Layout has 1 CTAs per CGA, context requires 8
ldred_linear_1cta_direct, num_ctas=16 -> Layout has 1 CTAs per CGA, context requires 16
ldred_linear_2cta_direct, num_ctas=4  -> Layout has 2 CTAs per CGA, context requires 4
ldred_linear_2cta_direct, num_ctas=8  -> Layout has 2 CTAs per CGA, context requires 8
ldred_linear_2cta_direct, num_ctas=16 -> Layout has 2 CTAs per CGA, context requires 16
cp_noscales_1cta_128x128, num_ctas=4  -> Layout has 1 CTAs per CGA, context requires 4
cp_noscales_1cta_128x128, num_ctas=8  -> Layout has 1 CTAs per CGA, context requires 8
cp_noscales_1cta_128x128, num_ctas=16 -> Layout has 1 CTAs per CGA, context requires 16
cp_noscales_2cta_128x128, num_ctas=4  -> Layout has 2 CTAs per CGA, context requires 4
cp_noscales_2cta_128x128, num_ctas=8  -> Layout has 2 CTAs per CGA, context requires 8
cp_noscales_2cta_128x128, num_ctas=16 -> Layout has 2 CTAs per CGA, context requires 16
```

## Representative diagnostics

1CTA ld/st descriptor-view chain in a 4-CTA launch:

```text
triton.compiler.errors.CompilationError: at 11:11:
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [2, M, N], parent_layout)
           ^
Layout has 1 CTAs per CGA, but the context requires 4 CTAs per CGA.
```

2CTA ld/st descriptor-view chain in an 8-CTA launch:

```text
triton.compiler.errors.CompilationError: at 11:11:
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [2, M, N], parent_layout)
           ^
Layout has 2 CTAs per CGA, but the context requires 8 CTAs per CGA.
```

1CTA direct `ld.red` in a 16-CTA launch:

```text
triton.compiler.errors.CompilationError: at 14:15:
    if chain_id == 0:
        tmem = allocate_tensor_memory(ttgl.float32, [M, N], layout)
               ^
Layout has 1 CTAs per CGA, but the context requires 16 CTAs per CGA.
```

2CTA no-scales copy in a 16-CTA launch:

```text
triton.compiler.errors.CompilationError: at 4:11:
def tmem_copy_128x128_twocta_kernel(in_ptr, out_ptr, tmem_layout: ttgl.constexpr):
    M: ttgl.constexpr = 256
    N: ttgl.constexpr = 4
    tmem = allocate_tensor_memory(in_ptr.dtype.element_ty, [M, N], layout=tmem_layout)
           ^
Layout has 2 CTAs per CGA, but the context requires 16 CTAs per CGA.
```

The scales-copy rows hit the same gate earlier, while constructing register
layouts for `TensorMemoryScalesLayout` paths:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python - <<'PY' 2>&1 | tail -120
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location('sf', 'python/test/gluon/test_tmem_structural_fuzzer.py')
sf = importlib.util.module_from_spec(spec); sys.modules[spec.name]=sf; spec.loader.exec_module(sf)
inp=torch.randint(-100,100,(64,16),dtype=torch.int8,device='cuda'); out=torch.empty_like(inp)
sf._fuzz_copy_scales_kernel[(1,)](inp,out,False,num_warps=4,num_ctas=4)
PY
```

Result:

```text
python/test/gluon/test_tmem_structural_fuzzer.py:899:14: error: Result has an invalid layout:
#ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4],
threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>}>.
Layout has 1 CTAs per CGA, but the context requires 4 CTAs per CGA.
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, blocked))
             ^
RuntimeError: error encountered during parsing
```

2CTA scales-copy in a 4-CTA launch:

```text
python/test/gluon/test_tmem_structural_fuzzer.py:899:14: error: Result has an invalid layout:
#ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4],
threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0],
CGALayout = [[1, 0]]}>}>.
Layout has 2 CTAs per CGA, but the context requires 4 CTAs per CGA.
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, blocked))
             ^
RuntimeError: error encountered during parsing
```

I classify the scales rows as `FZ-20260421-0010` manifestations, but their
promotion test should use a direct layout-construction diagnostic or a smaller
kernel so failures are easier to assert than the generic parser error wrapper.

## Positive high-CGA controls

Command:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python \
  pytest -s --tb=short \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[False-ctas_per_cga1]' \
  'python/test/gluon/test_core.py::test_tcgen05_mma_multicast_commit[True-ctas_per_cga2]'
```

Result:

```text
collected 2 items

python/test/gluon/test_core.py ..

2 passed in 3.18s
```

These are checked-in TCGEN05 MMA multicast/commit controls:

- `False-ctas_per_cga1`: 8-CTA context, 1CTA MMA instruction-local group.
- `True-ctas_per_cga2`: 16-CTA context, 2CTA MMA instruction-local group.

The controls still pass when full CGA-aware legacy TMEM/shared layouts are used,
which is the key contrast with the failing linear/scales layout rows.

## Classification

`FZ-20260421-0010` is a real over-strict gate candidate, not a duplicate of the
earlier TMEM buckets:

- Not `FZ-20260421-0001`: no runtime `memdesc_index` illegal op reaches LLVM.
- Not `FZ-20260421-0002` or `R5-C`: no generic-pass auto-layout crash.
- Not `FZ-20260421-0003`: descriptor-view packet order is never reached.
- Not `FZ-20260421-0004`: no `ld.red` opcode fallback reaches codegen.
- Not `FZ-20260421-0005` or `FZ-20260421-0009`: no allocator assertion.
- Not `FZ-20260421-0006` or `FZ-20260421-0008`: no false-unsupported
  transpose/slice or optimizer abort.

The failing rows ask for local 1CTA or 2CTA TCGEN05 TMEM work inside larger
4/8/16-CTA kernel contexts. The current validation instead requires the TMEM
layout's CTA count to equal the kernel CGA size exactly:

```text
Layout has X CTAs per CGA, but the context requires Y CTAs per CGA.
```

That equality requirement does not follow from the hardware rule previously
recorded for 2CTA-capable instructions. The hardware rule says that if any
instruction in a kernel that can be emitted as 2CTA is emitted as 2CTA, then all
such instructions in the kernel must use 2CTA. It does not by itself imply that
a local 1CTA or 2CTA TMEM operation is illegal in an 8- or 16-CTA launch,
especially because checked-in high-CGA MMA controls pass for both 1CTA and 2CTA
instruction-local groups when the layouts have enough CGA metadata.

## Likely abstraction gap

`TensorMemoryLinearLayout` exposes only `two_ctas: bool`, so it can represent
instruction-local 1CTA vs 2CTA participation but not an independent kernel CGA
layout. `TensorMemoryScalesLayout` has `cga_layout`, but its validation path
also treats the number of CTAs encoded in the layout as an exact match for the
kernel context rather than separating:

- kernel-level launch/CGA size;
- layout visibility/multicast mapping across the CGA;
- instruction-local `tcgen05` `cta_group::1` vs `cta_group::2`.

This is why the high-CGA legacy MMA controls pass while linear/scales TMEM rows
with local 1CTA/2CTA intent fail before lowering.

## Recommended promotion candidates

1. Add a runtime xfail that compiles a 1CTA `TensorMemoryLinearLayout` ld/st
   descriptor-view chain with `num_ctas=4`, `8`, and `16`. This is the smallest
   linear-layout allocation repro.

2. Add a runtime xfail that compiles a 2CTA `TensorMemoryLinearLayout` ld/st
   descriptor-view chain with `num_ctas=4`, `8`, and `16`. This covers the
   instruction-local `cta_group::2` case and guards against conflating 2CTA with
   full-CGA width.

3. Add a runtime xfail for 1CTA and 2CTA direct `ld.red` in high-CGA contexts.
   This confirms the gate is operation-family independent and blocks reductions
   before the existing `ld.red` opcode/planner buckets are reached.

4. Add no-scales copy xfails for 1CTA and 2CTA linear layouts in high-CGA
   contexts. These are useful because `tcgen05.cp` is not an MMA-only path and
   reproduces at `allocate_tensor_memory`.

5. Add a small parser/compile diagnostic test for `TensorMemoryScalesLayout`
   register-layout construction in a high-CGA context. Use the exact diagnostic
   substring `Layout has X CTAs per CGA, but the context requires Y CTAs per
   CGA`; avoid asserting only the outer `RuntimeError: error encountered during
   parsing`.

6. Keep the passing high-CGA MMA controls in the same test group as non-xfail
   evidence. The useful pair is:
   `test_tcgen05_mma_multicast_commit[False-ctas_per_cga1]` and
   `test_tcgen05_mma_multicast_commit[True-ctas_per_cga2]`.

## Report-only guarantee

No backend code was modified by this lane. The only intended repository write is
this report file.
