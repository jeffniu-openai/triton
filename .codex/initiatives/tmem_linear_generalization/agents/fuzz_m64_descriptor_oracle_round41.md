# Round 41: M64 Same-Footprint Descriptor-Chain Oracle

Date: 2026-04-21
Branch: `codex/tmem`
Scope: discovery/classification only. No backend/compiler source or checked-in
tests were modified. This report is the only repository file written by this
lane.

## Target

Investigated the unpromoted Round 39 watch item from
`agents/fuzz_ldred_extremes_round39.md`:
`same_chain_m64_n256_m64_row_reverse_w4` in the same-footprint descriptor-chain
harness.

The Round 39 summary JSON shows the actual case was:

```json
{
  "M": 64,
  "N": 256,
  "layout": "m64",
  "name": "same_chain_m64_n256_m64_row_reverse_w4",
  "num_warps": 4,
  "red_op": "max",
  "root": "same_chain",
  "variant": "auto"
}
```

So despite the case name containing `row_reverse`, the repro uses the M64
layout constructor, not a non-identity row permutation.

## Required Build

```bash
make -j8
```

Result:

```text
ninja -C /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja: Entering directory `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12'
ninja: no work to do.
```

## Original Harness Reruns

Suspect row, isolated:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_fz0018_min_round32/ldred_child.py \
  '{"root":"same_chain","M":64,"N":256,"layout":"m64","variant":"auto","red_op":"min","num_warps":4}'
```

Result: reproduced a 50% replay mismatch:

```text
Mismatched elements: 8192 / 16384 (50.0%)
Greatest absolute difference: 4.883898735046387 at index (22, 175)
Greatest relative difference: 1.0 at index (16, 0)
```

Direct M64 control:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_fz0018_min_round32/ldred_child.py \
  '{"root":"direct","M":64,"N":256,"layout":"m64","variant":"auto","red_op":"min","num_warps":4}'
```

Result: pass. PTX/LLIR each contained two
`tcgen05.ld.red.sync.aligned.16x32bx2.x64.min.f32` packets.

Smaller same-chain M64 control:

```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_fz0018_min_round32/ldred_child.py \
  '{"root":"same_chain","M":64,"N":128,"layout":"m64","variant":"auto","red_op":"min","num_warps":4}'
```

Result: reproduced the same 50% replay mismatch at `N=128`.

Identity-layout same-chain contrast:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_ldred_fz0018_min_round32/ldred_child.py \
  '{"root":"same_chain","M":64,"N":256,"layout":"identity","variant":"auto","red_op":"min","num_warps":4}'
```

Result: clean unsupported descriptor-view diagnostic for row anchors `32,64`,
not a runtime wrong result.

## Dedicated Temporary Oracle

Temporary probe:

```text
/tmp/tmem_m64_descriptor_oracle_round41.py
```

Validation:

```bash
PYTHONPATH=.:./python:./python/test/gluon \
  python -m py_compile /tmp/tmem_m64_descriptor_oracle_round41.py
```

The oracle keeps the Round 39 same-footprint chain shape:

```python
view = tmem.reshape((M // 2, 2, N)).reshape((M, N))
view = view.slice(0, M, dim=0).slice(0, N, dim=1)
```

It runs three modes:

- `direct_ldred`: direct M64 `tmem.load_min/load_max`;
- `same_ldred`: same-footprint descriptor chain followed by
  `view.load_min/load_max`;
- `same_load`: same-footprint descriptor chain followed by plain
  `view.load()`.

Representative exact repro for the original Round 39 shape:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_m64_descriptor_oracle_round41.py \
  '{"mode":"same_ldred","layout":"m64","n":256,"op":"max","variant":"auto","warps":4}'
```

Observed output:

```text
out mismatches: 8192 / 16384 (50.0%)
red mismatches: 32 / 64 (50.0%)
mismatch rows: 16..31 and 48..63
sample: actual 0.0, expected 2.163057565689087 at [16, 0]
PTX ld.red count: 2
LLIR ld.red count: 2
```

Plain-load discriminator:

```bash
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_m64_descriptor_oracle_round41.py \
  '{"mode":"same_load","layout":"m64","n":256,"warps":4}'
```

Observed output:

```text
out mismatches: 8192 / 16384 (50.0%)
mismatch rows: 16..31 and 48..63
sample: actual 0.0, expected 2.163057565689087 at [16, 0]
PTX plain ld count: 2
PTX ld.red count: 0
```

Direct M64 discriminator:

```bash
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_m64_descriptor_oracle_round41.py \
  '{"mode":"direct_ldred","layout":"m64","n":256,"op":"max","variant":"auto","warps":4}'
```

Observed output:

```text
out mismatches: 0 / 16384
red mismatches: 0 / 64
PTX ld.red count: 2
```

## Minimized Matrix

Driver command:

```bash
PYTHONPATH=.:./python:./python/test/gluon python - <<'PY'
import json, os, subprocess, sys
cases = []
for n in (32, 64, 128, 256):
    cases.append({"mode": "direct_ldred", "layout": "m64", "n": n, "op": "max", "variant": "auto", "warps": 4})
    cases.append({"mode": "same_load", "layout": "m64", "n": n, "warps": 4})
    for op in ("min", "max"):
        cases.append({"mode": "same_ldred", "layout": "m64", "n": n, "op": op, "variant": "auto", "warps": 4})
for variant in ("32x32b", "16x32bx2", "32x32b_splitn"):
    cases.append({"mode": "same_ldred", "layout": "m64", "n": 256, "op": "max", "variant": variant, "warps": 4})
for i, case in enumerate(cases):
    gpu = i % 4
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "TRITON_CACHE_DIR": f"/tmp/triton-cache-gpu{gpu}",
        "PYTHONPATH": ".:./python:./python/test/gluon",
    })
    proc = subprocess.run(
        [sys.executable, "/tmp/tmem_m64_descriptor_oracle_round41.py", json.dumps(case)],
        cwd="/root/code/triton", env=env, text=True, capture_output=True, timeout=180,
    )
    print(proc.stdout)
PY
```

Condensed results:

| Case | Result |
| --- | --- |
| `direct_ldred`, M64, `N=32/64/128/256`, `max`, `auto` | pass: `0` output mismatches, `0` reduction mismatches, hardware `.ld.red` present |
| `same_load`, M64, `N=32` | wrong output: `1024/2048` mismatches, rows `16..31` and `48..63`, plain `tcgen05.ld` present |
| `same_load`, M64, `N=64` | wrong output: `2048/4096` mismatches, same row set, plain `tcgen05.ld` present |
| `same_load`, M64, `N=128` | wrong output: `4096/8192` mismatches, same row set, plain `tcgen05.ld` present |
| `same_load`, M64, `N=256` | wrong output: `8192/16384` mismatches, same row set, plain `tcgen05.ld` present |
| `same_ldred`, M64, `N=32/64/128/256`, `min` | wrong output and wrong reduction: 50% output mismatch, `32/64` reduction mismatch, hardware `.ld.red` present |
| `same_ldred`, M64, `N=32/64/128/256`, `max` | wrong output and wrong reduction: 50% output mismatch, `32/64` reduction mismatch, hardware `.ld.red` present |
| `same_ldred`, M64, `N=256`, `variant=32x32b` | same wrong rows, `32/64` reduction mismatch, one hardware `.ld.red` packet |
| `same_ldred`, M64, `N=256`, `variant=16x32bx2` | same wrong rows, `32/64` reduction mismatch, two hardware `.ld.red` packets |
| `same_ldred`, M64, `N=256`, `variant=32x32b_splitn` | same wrong rows, `32/64` reduction mismatch, two hardware `.ld.red` packets |

Smallest stable reproducer found:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:./python/test/gluon \
  python /tmp/tmem_m64_descriptor_oracle_round41.py \
  '{"mode":"same_load","layout":"m64","n":32,"warps":4}'
```

This plain-load row is smaller and more diagnostic than the original
`ld.red` watch item because it removes reduction semantics while preserving the
same zero-row descriptor-chain failure.

## Classification

This is real wrong-output evidence, not an oracle artifact:

- the dedicated oracle checks replay and reduction separately;
- `same_ldred` returns zeros for rows `16..31` and `48..63` in both replay and
  reduced results;
- `same_load` has the identical replay failure without any reduction;
- direct M64 `ld.red` passes at every probed `N`; and
- hardware `.ld.red` is present for the `ld.red` rows, so this is not an opcode
  fallback/software-reduction artifact.

Bucket overlap:

- Not `FZ-20260421-0012`: no `unsupported dst layout` diagnostic, and direct
  M64 `ld.red` with the same M64 row-identity layout passes.
- Not `FZ-20260421-0004`: hardware `.ld.red` packets are emitted for the
  reduction rows; the plain-load discriminator also fails before opcode
  selection is relevant.
- Not `FZ-20260421-0002`: this row has no dynamic selector, helper-returned
  descriptor, or control-flow-selected descriptor.
- Best classified as an extension of existing `FZ-20260421-0003`: static
  descriptor-view chain load semantics are wrong. This lane adds a minimized
  M64 zero-row/no-op reshape+slice descriptor-chain form where the bad rows are
  exactly `16..31` and `48..63`.

No new independent `FZ-*` bucket is proposed. The Round 39 watch item should be
promoted only as additional `FZ-20260421-0003` evidence, with the preferred
minimal sentinel being the plain-load `M64,N=32` same-footprint chain above.
