# Round 30 Local: TMEM Lit Baseline

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery/cataloging only. No backend/compiler code was changed.

## Summary

Two compiler-only TMEM lit baselines passed while Round 30 runtime fuzzing
lanes continued:

```text
test/TritonNvidiaGPU/tmem_layouts.mlir: PASS
test/TritonNvidiaGPU/invalid.mlir: PASS
```

## Commands

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
lit -v test/TritonNvidiaGPU/tmem_layouts.mlir
```

```bash
cd /root/code/triton/build/cmake.linux-aarch64-cpython-3.12
ninja triton-opt
lit -v test/TritonNvidiaGPU/invalid.mlir
```

Logs:

```text
/tmp/tmem_r30_lit_tmem_layouts.log
/tmp/tmem_r30_lit_invalid.log
```

## Classification

No new bucket. The compiler-only TMEM layout and invalid verifier baselines
remain green.
