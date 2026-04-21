# Round 13 Lane X: TMEM allocation, lifetime, commit, and barrier fuzzing

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery only; no backend/compiler repairs attempted.
- Repo edit scope: this report only.
- Temporary probe: `/tmp/tmem_alloc_lifetime_round13_probe.py`
- Full run log: `/tmp/tmem_alloc_lifetime_round13_run.log`
- Exact diagnostic logs:
  - `/tmp/tmem_alloc_lifetime_round13_copy32.log`
  - `/tmp/tmem_alloc_lifetime_round13_copy64.log`
  - `/tmp/tmem_alloc_lifetime_round13_mma2cta32.log`
  - `/tmp/tmem_alloc_lifetime_round13_mma2cta64.log`

## Scope

This lane adversarially fuzzed TMEM allocation and lifetime behavior rather
than descriptor row/column mapping. The probe intentionally kept source code in
`/tmp` and launched each row in a fresh subprocess using a stable cache.

Coverage:

- two independent live `allocate_tensor_memory` allocations in one kernel;
- sibling descriptor views from one lifted parent allocation, with overlapping
  live ranges and intervening stores/loads;
- one `tcgen05.copy` plus an unrelated TMEM load/store allocation guarded by
  one `tcgen05.commit` / `mbarrier.wait` sequence;
- one `tcgen05.mma` accumulator view sharing a lifted parent allocation with a
  simultaneously live sibling view;
- launch `num_ctas in {1,2,4}` where local TMEM layouts stayed 1CTA; and
- multiple same-shape allocations around the 512-column TMEM resource limit.

## Commands

Required rebuild before testing:

```bash
make -j8
```

Result: `ninja: no work to do`.

Temporary probe syntax and inventory:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_alloc_lifetime_round13_probe.py

PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_alloc_lifetime_round13_probe.py --list | tail -3
```

Result: `TOTAL 28`.

Full subprocess-isolated sweep:

```bash
make -j8 && \
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-round13-lane-x \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_alloc_lifetime_round13_probe.py \
  2>&1 | tee /tmp/tmem_alloc_lifetime_round13_run.log
```

Raw summary:

```text
SUMMARY {"FZ-20260421-0010 high-CGA CTA-count gate": 4, "clean TMEM OutOfResources boundary": 2, "pass": 18, "uncategorized failure": 4}
```

The four raw `uncategorized failure` rows were exact-rerun diagnostics where
the compiler printed the useful verifier note outside the Python exception.
Manual final classification:

```text
pass                                             18
FZ-20260421-0010 high-CGA CTA-count gate          4
clean TMEM OutOfResources boundary                2
clean tcgen05.copy packed-lane unsupported        2
harness/shared-layout setup limitation            2
runtime miscompile                                0
new independent FZ candidate                      0
```

Exact representative reruns:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-round13-lane-x \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_alloc_lifetime_round13_probe.py \
  --case copy-plus-ldst-one-commit-128x32 --inprocess \
  > /tmp/tmem_alloc_lifetime_round13_copy32.log 2>&1

CUDA_VISIBLE_DEVICES=2 \
TRITON_CACHE_DIR=/tmp/triton-cache-round13-lane-x \
PYTHONPATH=.:./python:./python/test/gluon \
python /tmp/tmem_alloc_lifetime_round13_probe.py \
  --case mma-sibling-2cta-256x32x64 --inprocess \
  > /tmp/tmem_alloc_lifetime_round13_mma2cta32.log 2>&1
```

## Positive controls

Eighteen rows compiled, executed, and matched the PyTorch/reference values.
These are useful positive allocation-lifetime controls:

- five independent two-allocation rows:
  `two-alloc-ldst-128x{16,32,64,128,256}`;
- nine sibling-view rows:
  `sibling-interleave-chain{0,1,2}-128x{32,64,128}`;
- two 1CTA MMA-plus-sibling rows:
  `mma-sibling-1cta-128x{32,64}x64`;
- two allocator pressure rows below the hardware limit:
  `oversubscribe-2x128x128` and `oversubscribe-3x128x128`.

Representative opcode counts:

```text
two-alloc-ldst-128x32            ld=2 st=2
sibling-interleave-chain0-128x32 ld=2 st=3
mma-sibling-1cta-128x32x64       mma=4 ld=2 st=1
oversubscribe-3x128x128          ld=3 st=3
```

These rows demonstrate that overlapping live TMEM allocations, sibling
descriptor views, and one MMA accumulator view sharing a parent allocation with
a sibling load/store view do not expose a new lifetime or commit/barrier bug in
the tested 1CTA cases.

## Existing bucket: `FZ-20260421-0010`

Four high-CGA rows reproduced the existing over-strict CTA-count gate:

- `two-alloc-high-cga-2cta-128x32`;
- `copy-plus-ldst-high-cga-2cta-128x32`;
- `two-alloc-high-cga-4cta-128x32`; and
- `copy-plus-ldst-high-cga-4cta-128x32`.

Representative diagnostic:

```text
Layout has 1 CTAs per CGA, but the context requires 4 CTAs per CGA.
```

This strengthens `FZ-20260421-0010` with multiple live allocations and a
copy-plus-ldst commit sequence, but does not warrant a new bucket.

## Clean resource boundary

Two rows intentionally exceeded the 512-column hardware allocation budget:

- `oversubscribe-2x128x256`;
- `oversubscribe-3x128x256`.

Representative diagnostic:

```text
OutOfResources(768, 512, 'tensor memory')
```

This is the expected clean Python-side resource error, not an allocator assert.

## Clean unsupported copy boundary

Two copy-plus-ldst 1CTA rows reached a clean copy-family diagnostic:

- `copy-plus-ldst-one-commit-128x32`;
- `copy-plus-ldst-one-commit-128x64`.

Representative diagnostic:

```text
The source shared layout maps to tcgen05.copy.warpx2::01_23.64x128b, but Triton could not synthesize a compatible shared-memory descriptor plan for it.
```

The detailed note says this is reported cleanly unsupported instead of falling
through to late LLVM lowering. This overlaps Lane V's clean packed-lane copy
boundary and is not a new TMEM lifetime or barrier issue.

## Harness/shared-layout setup limitation

Two 2CTA MMA sibling-view rows failed before exercising the intended TMEM
lifetime path:

- `mma-sibling-2cta-256x32x64`;
- `mma-sibling-2cta-256x64x64`.

Representative diagnostic:

```text
Result has an invalid layout: #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>.
Layout has 1 CTAs per CGA, but the context requires 2 CTAs per CGA.
```

This row needs a corrected 2CTA shared-memory setup before it can be used as a
backend signal. I am not assigning a new FZ bucket from this harness limitation.

## Conclusion

No new independent `FZ-*` bucket is warranted from Lane X. The allocation and
lifetime positives are valuable coverage candidates for checked-in runtime
tests after the discovery campaign, especially:

- multiple independent live TMEM allocations;
- overlapping sibling descriptor views with intervening stores and loads;
- a 1CTA MMA accumulator view sharing a parent allocation with a live sibling;
- allocator-pressure cases that distinguish legal reuse from clean 512-column
  resource rejection.
