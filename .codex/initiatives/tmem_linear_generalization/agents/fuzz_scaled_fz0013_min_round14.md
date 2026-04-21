# Round 14 Lane AG: FZ-20260421-0013 scaled-MMAv5 scale descriptor minimization

- Date: 2026-04-21 11:11 UTC
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler fixes were attempted.

## Required rebuild

```bash
make -j8
```

Result: `ninja: no work to do`.

## Scope

This lane minimized and broadened `FZ-20260421-0013` from
`agents/fuzz_scaled_descriptor_round14.md`, originally described as a local
1CTA non-FPSAN B-scale descriptor-view wrong-result candidate with matching
scaled-MMA opcodes.

The temporary probe varied:

- `N=128` versus `N=256`;
- padded versus unpadded scale storage;
- extra scale user versus no extra user;
- linear, legacy, tile-permuted, and nearby permuted accumulator layouts;
- `use_acc=True` versus `use_acc=False`;
- `mxfp8/mxfp8`, mixed `mxfp8/mxfp4`, `mxfp4/mxfp8`, `mxfp4/mxfp4`, and
  `nvfp4/nvfp4` scaled formats;
- A-scale versus B-scale descriptor views;
- `reshape -> trans -> reshape`, same-shape `slice -> index`, direct no-view,
  and `index -> slice` descriptor chains;
- nearby narrow-N unsupported boundaries.

## Temporary probe

Probe file:

```text
/tmp/tmem_scaled_fz0013_min_round14_probe.py
```

Py-compile:

```bash
PYTHONPATH=.:./python:python/test/gluon \
  python -m py_compile /tmp/tmem_scaled_fz0013_min_round14_probe.py
```

Result: pass.

One-row smoke:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:python/test/gluon \
  python /tmp/tmem_scaled_fz0013_min_round14_probe.py \
  --case min-n128-b-rtr-pad0-linear
```

Result: runtime miscompile, `16109/16384` mismatches,
`max_abs=593.6329956054688`, `mma_count=4`, opcode sample
`tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`.
TTGIR retained `memdesc_reshape`, `memdesc_trans`, and `ttng.tmem_load`.

Four-GPU fresh-subprocess matrix:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:python/test/gluon \
  python /tmp/tmem_scaled_fz0013_min_round14_probe.py \
  --shard-index 0 --shard-count 4 --timeout 360 | tee /tmp/tmem_scaled_fz0013_round14_shard0.jsonl

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  PYTHONPATH=.:./python:python/test/gluon \
  python /tmp/tmem_scaled_fz0013_min_round14_probe.py \
  --shard-index 1 --shard-count 4 --timeout 360 | tee /tmp/tmem_scaled_fz0013_round14_shard1.jsonl

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  PYTHONPATH=.:./python:python/test/gluon \
  python /tmp/tmem_scaled_fz0013_min_round14_probe.py \
  --shard-index 2 --shard-count 4 --timeout 360 | tee /tmp/tmem_scaled_fz0013_round14_shard2.jsonl

CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  PYTHONPATH=.:./python:python/test/gluon \
  python /tmp/tmem_scaled_fz0013_min_round14_probe.py \
  --shard-index 3 --shard-count 4 --timeout 360 | tee /tmp/tmem_scaled_fz0013_round14_shard3.jsonl
```

Raw class counts over `33` fresh-child rows:

- `19` runtime miscompiles;
- `5` passes;
- `2` clean unsupported diagnostics;
- `1` explicit narrow-N unsupported diagnostic followed by `PassManager::run failed`;
- `6` parser/setup/unsupported-view limitations.

Repeat of the new smallest row:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:python/test/gluon \
  python - <<'PY' | tee /tmp/tmem_scaled_fz0013_round14_min_repeat.jsonl
import json, os, subprocess, sys
cmd=[sys.executable, '/tmp/tmem_scaled_fz0013_min_round14_probe.py',
     '--case', 'min-n128-b-rtr-pad0-linear']
for i in range(3):
    p=subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, env=os.environ.copy(),
                     timeout=240)
    payload=None
    for line in reversed([l for l in p.stdout.splitlines() if l.strip()]):
        if line.startswith('{'):
            payload=json.loads(line); break
    payload['repeat']=i
    payload['returncode']=p.returncode
    print(json.dumps(payload, sort_keys=True), flush=True)
PY
```

Result: `3/3` fresh subprocesses reproduced the same wrong result:
`16109/16384` mismatches, `max_abs=593.6329956054688`, matching PTX/LLIR
opcode list, and descriptor-chain TTGIR retained.

Prior corrected B-scale probe replay:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  PYTHONPATH=.:./python:python/test/gluon \
  python /tmp/tmem_scaled_bscale_descriptor_round14_probe2.py \
  | tee /tmp/tmem_scaled_fz0013_round14_prior_probe2_replay.txt
```

Result matched the earlier lane shape: `3` runtime miscompile candidates,
`1` pass, and nearby narrow-N diagnostic boundaries. The unpadded `N=64`
tile-16 row emits the intended B-scale padding/rematerialization unsupported
diagnostic before the raw process reports `PassManager::run failed`; this was
not treated as a new runtime-miscompile bucket.

## Case matrix

| Case | Result | Notes |
| --- | --- | --- |
| `min-n128-b-rtr-pad0-linear` | `FZ-0013` | Smallest row found here: `N=128,K=128`, B-scale `reshape -> trans -> reshape`, unpadded, no extra user, linear accumulator; `16109/16384` mismatches, `mma=4`. |
| `min-n128-b-rtr-pad1-linear` | `FZ-0013` | Padded storage still miscompiles; `16235/16384`, `mma=4`. |
| `n128-b-rtr-pad0-extra` | `FZ-0013` | Extra B-scale user not required; same mismatch as no-extra row. |
| `n128-b-rtr-pad1-extra` | `FZ-0013` | Padded plus extra user still miscompiles. |
| `n128-b-rtr-pad0-useacc0` | `FZ-0013` | `use_acc=False` still miscompiles. |
| `n128-b-rtr-pad1-useacc0` | `FZ-0013` | Padded `use_acc=False` still miscompiles. |
| `n256-b-rtr-pad0-linear` | `FZ-0013` | `N=256` linear miscompiles; `32420/32768`, `mma=4`. |
| `n256-b-rtr-pad1-linear` | `FZ-0013` | `N=256` padded linear miscompiles; `32471/32768`, `mma=4`. |
| `n256-b-rtr-pad0-tile64` | `FZ-0013` | `N=256` tile-N64 miscompiles; `32420/32768`, `mma=16`. |
| `n256-b-rtr-pad1-tile64` | `FZ-0013` | `N=256` padded tile-N64 miscompiles; `32420/32768`, `mma=16`. |
| `n128-b-rtr-pad0-tile32` | pass | Unpadded tile-N32 accumulator is a positive control; `mma=16`. |
| `n128-b-rtr-pad1-tile32` | `FZ-0013` | Padded tile-N32 miscompiles; `16237/16384`, `mma=16`. |
| `n128-b-rtr-pad0-legacy` | `FZ-0013` | Legacy accumulator layout miscompiles like linear. |
| `n128-b-rtr-pad0-rowrev` | probe limitation | Parser setup failed before backend classification. |
| `n128-b-rtr-pad0-colrev` | probe limitation | Parser setup failed before backend classification. |
| `n128-b-slice-index-pad0` | pass | Same-shape `reshape -> slice -> index` B-scale view passes without padding. |
| `n128-b-slice-index-pad1` | `FZ-0013` | Padded same-shape `slice -> index` B-scale view miscompiles; `12286/16384`. |
| `n128-b-index-slice-pad0` | clean unsupported/probe boundary | `failed to infer memdesc_index result type`; no runtime execution. |
| `n128-b-none-pad0-control` | pass | B-scale direct no-view control passes. |
| `n128-a-rtr-pad0` | `FZ-0013` | A-scale `reshape -> trans -> reshape` also miscompiles; `16110/16384`, `mma=4`. |
| `n128-a-rtr-pad1` | `FZ-0013` | Padded A-scale descriptor view also miscompiles. |
| `n128-a-slice-index-pad0` | pass | Same-shape A-scale `slice -> index` view passes without padding. |
| `n128-a-index-slice-pad0` | clean unsupported/probe boundary | `failed to infer memdesc_index result type`; no runtime execution. |
| `n128-no-scale-view-control` | pass | Direct A/B scale control passes. |
| `fmt-mxfp8-mxfp4-b-rtr` | `FZ-0013` | Mixed B format miscompiles with matching opcode list. |
| `fmt-mxfp4-mxfp8-b-rtr` | `FZ-0013` | Mixed A format miscompiles with matching opcode list. |
| `fmt-mxfp4-mxfp4-b-rtr` | `FZ-0013` | FP4/FP4 miscompiles; opcode sample `tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X`. |
| `fmt-nvfp4-nvfp4-b-rtr` | `FZ-0013` | NVFP4/NVFP4 miscompiles; opcode sample `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X`. |
| `boundary-n64-b-rtr-pad0-tile16` | diagnostic boundary | Emits explicit unpadded B-scale storage unsupported diagnostic, then raw process reports `PassManager::run failed`. |
| `boundary-n64-b-rtr-pad1-tile16` | probe/diagnostic limitation | Parser failed before a reliable runtime classification. |
| `boundary-n32-b-rtr-pad1-tile8` | probe/diagnostic limitation | Parser failed before a reliable runtime classification. |
| `boundary-n16-b-rtr-pad0-linear` | clean unsupported | Clean `TMEM layout 'auto' unsupported for descriptor view` diagnostic. |
| `boundary-n16-b-rtr-pad1-linear` | clean unsupported | Clean `TMEM layout 'auto' unsupported for descriptor view` diagnostic. |

## Classification

`FZ-20260421-0013` is confirmed and broadened. The smallest stable row no
longer needs an extra B-scale user:

```text
N=128, K=128, local 1CTA, non-FPSAN,
linear accumulator layout,
B-scale descriptor view reshape -> trans -> reshape,
unpadded B-scale storage,
use_acc=True or False,
no extra scale consumer required.
```

The bucket is broader than B-scale only: A-scale `reshape -> trans -> reshape`
descriptor views also miscompile under the same local 1CTA non-FPSAN scaled
MMA path. The issue also crosses padded/unpadded storage, `N=128/256`, legacy
and linear accumulator layouts, selected tile-permuted layouts, `use_acc`, and
all probed scale format combinations. PTX and LLIR opcode extraction matched
for every runtime-miscompile row.

Positive controls narrow the trigger: direct no-view scale descriptors pass,
unpadded same-shape `slice -> index` views pass for A and B scales, and the
unpadded `N=128` tile-N32 accumulator row passes despite retaining descriptor
view TTGIR. This points at descriptor-view scale-fragment mapping/rematerialized
scale storage rather than generic scaled-MMAv5 execution or opcode selection.

No new independent `FZ-*` bucket was found. The only new classification change
is that `FZ-20260421-0013` should be described as a scaled-MMAv5 scale
descriptor-view wrong-result bucket, covering both A-scale and B-scale
descriptor views, rather than B-scale only.

## Follow-up

- Keep `FZ-20260421-0013` report-only until a checked-in sentinel is requested
  or the discovery campaign pivots from finding to fixing.
- If promoted later, use the smallest repeated row
  `min-n128-b-rtr-pad0-linear`; it is deterministic across fresh subprocesses
  and does not require FPSAN, dynamic accumulator selection, high-CGA launch,
  or an extra scale user.
- Do not use the `index -> slice` rows as miscompile evidence; they currently
  stop at descriptor-view support diagnostics.
