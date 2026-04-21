# Round 42 TMEM Cache/State Reuse Probe

Date: 2026-04-21
Branch: `codex/tmem`
Scope: discovery/cataloging only. No backend/compiler code modified.
Owned report path: `.codex/initiatives/tmem_linear_generalization/agents/fuzz_cache_state_reuse_round42.md`

## Objective

Adversarially exercise TMEM-bearing Python/Gluon runtime coverage for:

- repeated in-process compile/execute of mixed TMEM shapes and layouts;
- dynamic descriptor failures followed by positive runtime rows in the same
  pytest process;
- stable `TRITON_CACHE_DIR` reuse across all four GPUs without fresh temporary
  cache workarounds;
- process-boundary behavior after expected structural failures have populated
  or consulted the same per-GPU cache directories.

## Environment and Cache Policy

- Required rebuild:
  `make -j8`
  - Result: build entered
    `/root/code/triton/build/cmake.linux-aarch64-cpython-3.12`; `ninja: no work to do`.
- All pytest commands used `PYTHONPATH=python`.
- Stable cache directories were reused:
  - GPU 0: `TRITON_CACHE_DIR=/tmp/triton-cache-gpu0`
  - GPU 1: `TRITON_CACHE_DIR=/tmp/triton-cache-gpu1`
  - GPU 2: `TRITON_CACHE_DIR=/tmp/triton-cache-gpu2`
  - GPU 3: `TRITON_CACHE_DIR=/tmp/triton-cache-gpu3`
- No fresh temporary cache directories were introduced.
- Initial collection without `PYTHONPATH=python` hit the wrong installed
  Triton package and failed import collection. That was a harness invocation
  error, not a TMEM/cache signal. All real classifications below use the
  corrected `PYTHONPATH=python` commands.

## Discovery Collections

Collected exact checked-in controls before execution:

1. Runtime matrix descriptor/ldst/clean selector:

   ```bash
   PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
     pytest --collect-only -q -s --tb=short \
       -k "(ldst and not reports and not resource and not roundtrip and not clean) or (clean_unsupported or clean_error or reports_clean or tmem_oor) or (descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows)" \
       python/test/gluon/test_tmem_runtime_matrix.py
   ```

   Result: `428/1615` collected.

2. Structural known-failure selector:

   ```bash
   PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
     pytest --collect-only -q -s --tb=short \
       -k "generic_pass or descriptor_view or ldred or scaled_mma" \
       python/test/gluon/test_tmem_structural_fuzzer.py
   ```

   Result: `26/33` collected.

3. `test_core.py` TMEM/MMAv5 guardrail selector:

   ```bash
   PYTHONPATH=python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
     pytest --collect-only -q -s --tb=short \
       -k "tcgen05 and (tmem or mma or copy or multicast or mbarrier)" \
       python/test/gluon/test_core.py
   ```

   Result: `123/18114` collected.

4. Corrected rank-5 nodeid collection after one truncated-nodeid harness miss:

   ```bash
   PYTHONPATH=python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
     pytest --collect-only -q -s --tb=short \
       -k "rank5_unit_parent_n256_roundtrip" \
       python/test/gluon/test_tmem_runtime_matrix.py
   ```

   Result: `9/1615` collected. The valid first nodeid is
   `test_tmem_runtime_matrix_ldst_descriptor_rank5_unit_parent_n256_roundtrip[f32-torch_dtype0-single_identity-single-identity-128-1-auto-32x32b.x64.b32]`.

## In-Process Adversarial Orderings

These commands intentionally mixed positives, clean diagnostics, and expected
dynamic descriptor failures in one pytest process per GPU, then ran a positive
row after the expected failure. Repeated positive nodeids in the command line
were accepted by pytest collection but did not produce a separate counted rerun
in all cases, so classification relies on the mixed-order process and the later
warm-cache process-boundary sweeps.

### GPU 0 ordering: positives, clean diagnostic, FZ dynamic failures, positive

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  pytest -s --tb=short -rxX \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst[identity-128-auto-32x32b.x128.b32] \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions[mixed-128-auto-32x32b.x128.b32] \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_higher_rank_half_rows_positive_lifted_layout[identity-128-auto-32x32b.x128.b32] \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_scales_tmem_descriptor_view_reports_clean_unsupported \
    python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-index-chain0] \
    python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-if-chain0-true] \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_bscale_descriptor_view \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst[identity-128-auto-32x32b.x128.b32]
```

Result: `5 passed, 2 xfailed`.

Expected xfails:

- `generic-pass-dynamic-index-chain0`: existing `FZ-20260421-0001`, runtime TMEM
  `memdesc_index` reaches LLVM conversion as an illegal op.
- `generic-pass-dynamic-if-chain0-true`: existing `FZ-20260421-0002`, dynamic-if
  helper-returned TMEM view miscompile.

The post-xfail scaled B-scale descriptor-view positive passed in the same
process.

### GPU 1 ordering: FZ dynamic failure first, clean diagnostic, positives, FZ

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/tmp/triton-cache-gpu1 \
  pytest -s --tb=short -rxX \
    python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_memdesc_control_flow[generic-pass-dynamic-index-chain1] \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_n16_bscale_descriptor_view_reports_clean_error[False] \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions_exotic_layouts[f32-torch_dtype1-scrambled_rows_cols-128-auto-32x32b.x128.b32] \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_compositions[mmav5_twocta-128-auto-32x32b.x128.b32] \
    python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_layout_conversion_pressure[generic-pass-layout-conversion-pressure-chain0] \
    python/test/gluon/test_core.py::test_tcgen05_mma_scaled_minimal \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_compositions_exotic_layouts[f32-torch_dtype1-scrambled_rows_cols-128-auto-32x32b.x128.b32]
```

Result: `4 passed, 2 xfailed`.

Expected xfails:

- `generic-pass-dynamic-index-chain1`: existing `FZ-20260421-0001`.
- `generic-pass-layout-conversion-pressure-chain0`: existing
  `FZ-20260421-0002`.

Both the clean diagnostic and later positives passed after the initial expected
dynamic descriptor failure.

### GPU 2 ordering: higher-rank positive, clean diagnostic, ld.red FZ, positive

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton-cache-gpu2 \
  pytest -s --tb=short -rxX \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_index[f32-torch_dtype2-mmav5_twocta-64-auto-32x32b.x64.b32-32x32b.x32.b32] \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_cp_no_scales_linear_rowcol_permuted_reports_clean_unsupported[rotate1-reverse] \
    python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-fz20260421-0004-chain1-64x32-min] \
    python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-direct-128x64] \
    python/test/gluon/test_core.py::test_tmem_reduction_linear_former_clean_errors_are_supported[layout1-128-64-tcgen05.ld.red.sync.aligned.32x32b.x] \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_twocta_descriptor_higher_rank_index[f32-torch_dtype2-mmav5_twocta-64-auto-32x32b.x64.b32-32x32b.x32.b32]
```

Result: `4 passed, 1 xfailed`.

Expected xfail:

- `ldred-fz20260421-0004-chain1-64x32-min`: existing `FZ-20260421-0004`,
  descriptor-view `ld.red` emits plain load plus software reduce.

The direct `ld.red` structural positive and `test_core.py` reduction clean-error
support row passed after the known FZ row.

### GPU 3 ordering: rank-5 positive, clean diagnostic, dynamic failure, positives

The first GPU 3 execution used a bad rank-5 nodeid derived from truncated
collection output and exited with pytest code `4` (`no tests ran`). The same
stable cache directory was then reused with the corrected nodeid:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/triton-cache-gpu3 \
  pytest -s --tb=short -rxX \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_rank5_unit_parent_n256_roundtrip[f32-torch_dtype0-single_identity-single-identity-128-1-auto-32x32b.x64.b32] \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_i8_reports_clean_error[32-64-linear] \
    python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_dynamic_index_load_only[generic-pass-dynamic-index-load-only-128x32] \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_mma_scaled_acc_tile_permuted_32_bscale_view_extra_user_rematerializes \
    python/test/gluon/test_core.py::test_tcgen05_mma_plain_kind_runtime[tf32] \
    python/test/gluon/test_tmem_runtime_matrix.py::test_tmem_runtime_matrix_ldst_descriptor_rank5_unit_parent_n256_roundtrip[f32-torch_dtype0-single_identity-single-identity-128-1-auto-32x32b.x64.b32]
```

Result: `4 passed, 1 xfailed`.

Expected xfail:

- `generic-pass-dynamic-index-load-only-128x32`: existing `FZ-20260421-0001`,
  direct runtime TMEM `memdesc_index` reaches LLVM conversion as an illegal op.

The post-xfail B-scale extra-user positive and `test_core.py` MMAv5 positive
passed in the same process.

## Warm-Cache Process-Boundary Controls

After the mixed in-process orderings, the same per-GPU cache directories were
reused for broader checked-in selectors in fresh pytest processes.

### Descriptor/high-rank positive selector

Commands:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  pytest -s --tb=short --splits 4 --group <1..4> \
    -k "(descriptor_compositions or higher_rank or rank5 or multidim_slice or half_rows) and not reports and not resource and not clean" \
    python/test/gluon/test_tmem_runtime_matrix.py
```

Results:

- Group 1 / GPU 0: `35 passed`.
- Group 2 / GPU 1: `35 passed`.
- Group 3 / GPU 2: `27 passed, 8 skipped`.
- Group 4 / GPU 3: `23 passed, 12 skipped`.

Aggregate: `120 passed, 20 skipped`, matching the recent Round 39 guardrail.

### Clean diagnostic selector

Command:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
  pytest -s --tb=short -rxX \
    -k "(clean_unsupported or clean_error or reports_clean or tmem_oor) and not reports" \
    python/test/gluon/test_tmem_runtime_matrix.py
```

Result: `22 passed`.

### Structural expected-failure selector

Commands:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=<gpu> TRITON_CACHE_DIR=/tmp/triton-cache-gpu<gpu> \
  pytest -s --tb=short -rxX --splits 4 --group <1..4> \
    -k "generic_pass or descriptor_view or ldred or scaled_mma" \
    python/test/gluon/test_tmem_structural_fuzzer.py
```

Results:

- Group 1 / GPU 1: `3 passed, 4 xfailed`.
- Group 2 / GPU 2: `7 xfailed`.
- Group 3 / GPU 3: `7 xfailed`.
- Group 4 / GPU 0: `5 xfailed`.

Aggregate: `3 passed, 23 xfailed`, matching the recent Round 39 structural
xfail guardrail. The dynamic `memdesc_index` rows again emitted the known MLIR
reproducer diagnostics for `FZ-20260421-0001`; no XPASS or unexpected failure
appeared.

## Classification

- Cache/state bug: none observed.
- Process-boundary sensitivity: none observed. Fresh pytest processes after
  mixed dynamic-descriptor failures continued to pass the checked-in positive
  and clean-diagnostic selectors with stable cache reuse.
- In-process state contamination: none observed. Positives following expected
  dynamic descriptor failures in the same pytest process passed on all four GPU
  lanes.
- Stable positives:
  - load/store direct and descriptor-composition rows;
  - descriptor higher-rank and rank-5 rows;
  - scaled/MMAv5 B-scale descriptor-view and extra-user rows;
  - `test_core.py` MMAv5 and reduction support controls.
- Stable clean diagnostics:
  - copy/scales descriptor-view unsupported diagnostic;
  - B-scale N16 clean error;
  - row/column-permuted copy unsupported diagnostic;
  - MMA i8 clean error;
  - broader clean selector `22 passed`.
- Expected known FZ failures:
  - `FZ-20260421-0001`: runtime TMEM `memdesc_index` reaches LLVM conversion as
    illegal op.
  - `FZ-20260421-0002`: helper-returned/dynamic-if TMEM view miscompile family.
  - `FZ-20260421-0004`: `ld.red` descriptor-view chain opcode/software-reduce
    issue.
  - Existing structural selector also revalidated `FZ-20260421-0003`,
    `FZ-20260421-0006`, `FZ-20260421-0007`, `FZ-20260421-0008`, and
    `FZ-20260421-0009` as xfails only.
- Candidate new FZ bucket: none.

## Notes

- No temporary `/tmp` probe file was needed beyond stable cache directories;
  checked-in selectors and explicit nodeid orderings covered the requested
  cache/state surfaces.
- The one `no tests ran` event on GPU 3 was a bad nodeid from truncated output,
  then corrected and rerun against the same stable cache. It is not classified
  as a compiler/runtime/cache symptom.
- The noisy MLIR reproducer output in structural rows is the already cataloged
  `FZ-20260421-0001` behavior and did not prevent subsequent positives in the
  same process or later warm-cache processes.
