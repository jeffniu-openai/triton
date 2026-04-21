# Lane AU Round 19: plain-only proxy-fence reproducer for `FZ-20260421-0014`

Date: 2026-04-21
Branch: `codex/tmem`
Head: `a80254d4314af537122f6f6e23a6e62d63de4bc6`
Mode: discovery/cataloging only. No backend/compiler code was changed. No
commit or push was made by this lane.

## Scope

Round 18 showed that `FZ-20260421-0014` no longer needs TMEM copy operations:
two sequential plain cross-CTA mbarrier intervals are enough to reproduce the
proxy-fence insertion failure. This lane saved a minimal MLIR reproducer,
verified it with `triton-opt --run-reproducer`, and compared the failure
against current upstream main and the branch merge-base in an isolated
worktree.

## Required rebuild

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

## Branch Python repro and saved MLIR

Command:

```bash
rm -f /tmp/tmem_fz0014_plain_seq_round19_repro.mlir* \
  /tmp/tmem_fz0014_plain_seq_round19_at001.log

CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
TRITON_ALWAYS_COMPILE=1 \
TRITON_REPRODUCER_PATH=/tmp/tmem_fz0014_plain_seq_round19_repro.mlir \
PYTHONPATH=.:./python:./python/test/gluon \
pytest -q -s --tb=short \
  /tmp/tmem_proxy_fence_intervals_round18_probe.py::test_at001_two_plain_mbarriers_sequential_interval \
  2>&1 | tee /tmp/tmem_fz0014_plain_seq_round19_at001.log
```

Result:

```text
/tmp/tmem_proxy_fence_intervals_round18_probe.py:923:1: error: 'tt.func' op could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
def at_two_plain_mbarriers_sequential_kernel(status):
^
<unknown>:0: error: Failures have been detected while processing an MLIR pass pipeline
<unknown>:0: note: Pipeline failed while executing `ConvertTritonGPUToLLVM` on 'builtin.module' operation: reproducer generated at `/tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir`
AT001_CLASS=existing_fz0014
.
1 passed in 2.79s
```

Saved reproducer:

```text
/tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir
```

The saved reproducer is plain-only:

```text
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8 : i32, ttg.target = "cuda:103", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, "ttng.two-ctas" = false}
```

The body contains two independent `ttg.local_alloc` mbarriers, each with
`ttng.init_barrier`, `ttng.arrive_barrier`, and `ttng.wait_barrier`, followed
by one scalar `tt.store`. It contains no `ttng.tmem*`, `tcgen05*`, or
`ttng.tmem_copy` operations.

## Branch `triton-opt --run-reproducer`

Command:

```bash
cd build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt --run-reproducer \
  /tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir \
  2>&1 | tee /tmp/tmem_fz0014_plain_seq_round19_run_reproducer.log

set -o pipefail
bin/triton-opt --run-reproducer \
  /tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir \
  > /tmp/tmem_fz0014_plain_seq_round19_branch_run_reproducer_pipefail.log 2>&1
echo exit=$?
```

Result:

```text
/tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir:4:3: error: 'tt.func' op could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
exit=1
```

## Upstream main comparison

I used the isolated worktree `/tmp/triton-upstream-main-check`, leaving
`/root/code/triton` on `codex/tmem`.

Commands:

```bash
git fetch upstream main
git -C /tmp/triton-upstream-main-check checkout -f upstream/main
make -j8

cd /tmp/triton-upstream-main-check/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt --run-reproducer \
  /tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir \
  2>&1 | tee /tmp/tmem_fz0014_plain_seq_round19_upstream_main_run_reproducer.log

set -o pipefail
bin/triton-opt --run-reproducer \
  /tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir \
  > /tmp/tmem_fz0014_plain_seq_round19_upstream_main_run_reproducer_pipefail.log 2>&1
echo exit=$?
```

Comparison commit:

```text
dea2e9d7324309fdc9198144669621f57f3704a2
dea2e9d73 [AMD] Add atomic vectorization cap (#10093)
```

Build result: `make -j8` rebuilt the isolated worktree successfully.

Result:

```text
/tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir:4:3: error: 'tt.func' op could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
exit=1
```

I also ran the equivalent standalone Python probe against upstream main:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/triton-main-cache-gpu0 \
TRITON_ALWAYS_COMPILE=1 \
TRITON_REPRODUCER_PATH=/tmp/tmem_fz0014_plain_seq_round19_min_probe_upstream_main_repro.mlir \
PYTHONPATH=.:./python \
python /tmp/tmem_fz0014_plain_seq_round19_min_probe.py \
  2>&1 | tee /tmp/tmem_fz0014_plain_seq_round19_min_probe_upstream_main.log
```

It emitted the same source diagnostic and generated:

```text
/tmp/tmem_fz0014_plain_seq_round19_min_probe_upstream_main_repro.mlir.make_llir.repro.mlir
```

## Merge-base comparison

Commands:

```bash
git -C /tmp/triton-upstream-main-check checkout -f 2c7ce4925d37802dd84dfde1f6458cae19485617
make -j8

cd /tmp/triton-upstream-main-check/build/cmake.linux-aarch64-cpython-3.12
bin/triton-opt --run-reproducer \
  /tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir \
  2>&1 | tee /tmp/tmem_fz0014_plain_seq_round19_mergebase_run_reproducer.log

set -o pipefail
bin/triton-opt --run-reproducer \
  /tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir \
  > /tmp/tmem_fz0014_plain_seq_round19_mergebase_run_reproducer_pipefail.log 2>&1
echo exit=$?
```

Comparison commit:

```text
2c7ce4925d37802dd84dfde1f6458cae19485617
2c7ce4925 [AMD][gfx1250] Add Scaled WMMA 32x16 Shape for FP4 (#10082)
```

Build result: `make -j8` rebuilt the isolated worktree successfully.

Result:

```text
/tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir:4:3: error: 'tt.func' op could not find an insertion point between cross-CTA mbarrier.init ops and tracked mbarrier uses
exit=1
```

## Artifact summary

```text
/tmp/tmem_fz0014_plain_seq_round19_repro.mlir.make_llir.repro.mlir
/tmp/tmem_fz0014_plain_seq_round19_at001.log
/tmp/tmem_fz0014_plain_seq_round19_run_reproducer.log
/tmp/tmem_fz0014_plain_seq_round19_branch_run_reproducer_pipefail.log
/tmp/tmem_fz0014_plain_seq_round19_upstream_main_run_reproducer.log
/tmp/tmem_fz0014_plain_seq_round19_upstream_main_run_reproducer_pipefail.log
/tmp/tmem_fz0014_plain_seq_round19_mergebase_run_reproducer.log
/tmp/tmem_fz0014_plain_seq_round19_mergebase_run_reproducer_pipefail.log
/tmp/tmem_fz0014_plain_seq_round19_min_probe.py
/tmp/tmem_fz0014_plain_seq_round19_min_probe_branch.log
/tmp/tmem_fz0014_plain_seq_round19_min_probe_branch_repro.mlir.make_llir.repro.mlir
/tmp/tmem_fz0014_plain_seq_round19_min_probe_upstream_main.log
/tmp/tmem_fz0014_plain_seq_round19_min_probe_upstream_main_repro.mlir.make_llir.repro.mlir
```

The standalone Python probe reproducer hashes match between branch and upstream
main:

```text
3c78bd16ac7e49c38c8ddf6464875d28639c880f9c9007ec52d747b9e9319a9b  /tmp/tmem_fz0014_plain_seq_round19_min_probe_branch_repro.mlir.make_llir.repro.mlir
3c78bd16ac7e49c38c8ddf6464875d28639c880f9c9007ec52d747b9e9319a9b  /tmp/tmem_fz0014_plain_seq_round19_min_probe_upstream_main_repro.mlir.make_llir.repro.mlir
```

## Classification

`FZ-20260421-0014` plain-only sequential mbarrier failure is **preexisting on
main and merge-base**, not branch-specific.

The TMEM branch did not introduce this minimal proxy-fence pass limitation. TMEM
copy tests still legitimately expose it because 2CTA copy lowering creates the
same kind of cross-CTA mbarrier intervals, but the minimized root surface is a
generic proxy-fence insertion ordering problem for two sequential independent
cross-CTA mbarrier lifetimes.

No new independent `FZ-*` bucket is needed. Keep this under
`FZ-20260421-0014`, with the branch-regression classification refined to:

```text
preexisting upstream proxy-fence insertion limitation, newly exposed by TMEM
copy/mbarrier structural fuzzing
```
