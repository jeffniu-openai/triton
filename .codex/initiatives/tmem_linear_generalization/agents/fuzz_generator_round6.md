# Round 6 Lane D: Deterministic TMEM Structural Fuzzer Generator Prototype

Date: 2026-04-21
Branch: `codex/tmem`
Mode: discovery only; no backend/compiler repairs attempted.

## Objective

Build a systematic deterministic generator for compact TMEM structural case
descriptors that can enumerate across:

- family: `ldst`, `ldred`, `copy`, `mma`, `mma_scaled`;
- shape;
- dtype;
- `two_cta` and `num_ctas`;
- descriptor-view chain;
- row/col layout permutation;
- register-layout or instruction variant;
- expected class: `pass`, `clean_unsupported`, `clean_error`, or deferred
  contract class.

The prototype stayed in `/tmp` because the schema is useful, but not ready for
repo promotion: it currently generates descriptors and coverage inventories,
not runnable Gluon kernels for every family.

## Prototype Artifact

- Script: `/tmp/tmem_structural_case_generator_round6.py`
- JSON inventory: `/tmp/tmem_structural_case_generator_round6.json`
- Command:

```bash
python /tmp/tmem_structural_case_generator_round6.py \
  --repo /root/code/triton \
  --json-out /tmp/tmem_structural_case_generator_round6.json \
  --limit 32
```

The prototype uses only Python stdlib. It parses
`python/test/gluon/test_tmem_structural_fuzzer.py` with `ast`, extracts current
case ids, enumerates normalized descriptors, and prints representative
uncovered rows.

## Generator Schema

The prototype descriptor is:

```python
CaseDescriptor(
    case_id: str,
    seed: int,
    family: str,
    shape: tuple[int, ...],
    dtype: str,
    two_cta: bool,
    num_ctas: int,
    view_chain: str,
    row_kind: str,
    col_kind: str,
    reg_variant: str,
    expected_class: str,
)
```

Enumerated values:

- `family`: `ldst`, `ldred`, `copy`, `mma`, `mma_scaled`
- `view_chain`: `direct`, `index`, `reshape_permute`,
  `double_transpose_slice`, `subslice`, `dynamic_index`, `if_result`,
  `loop_carried`
- `row_kind`: `identity`, `reverse`, `even_odd`, `rotate1`
- `col_kind`: `identity`, `reverse`, `rotate1`
- `reg_variant`: `32x32b`, `16x64b`, `16x128b`

The seed is a stable SHA1-derived integer over descriptor fields, so generated
cases are reproducible independent of enumeration order.

## Inventory Output

Prototype output:

```text
generated_total: 25245
checked_in_ids: 30
generated_exact_id_matches: 0

generated_by_family:
  copy: 8415
  ldred: 3315
  ldst: 8160
  mma: 2295
  mma_scaled: 3060

generated_by_expected_class:
  pass: 15168
  clean_unsupported: 2805
  clean_error: 3024
  clean_error_or_pass: 1980
  defer_unchecked: 2268
```

The zero exact-id match is expected and useful: the checked-in fuzzer currently
uses historical repro-oriented ids such as
`ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min`, while the prototype
uses normalized schema ids such as
`ldred-f32-2cta-ncta2-256x32-index-identity-identity-32x32b`.

## Runtime Sample

Required pre-test build:

```bash
CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8
```

Result: no rebuild needed; Ninja reported no work to do.

Collect-only command:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result: `30 tests collected`.

Small runtime sample command:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-gpu0 \
PYTHONPATH=.:./python pytest -s --tb=short \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-direct-128x64]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-1cta]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_generic_pass_loop_carried[generic-pass-loop-carried-memdesc-view-chain0]'
```

Result: `3 passed, 1 xfailed in 4.61s`.

No new backend failure was discovered by this lane. The xfail is the already
cataloged Round 5 loop-carried generic-pass compiler crash sentinel.

## Coverage Gaps in Current Checked-In Structural Fuzzer

The current checked-in fuzzer is effective for known repro promotion, but it is
not yet a systematic structural generator.

Observed gaps:

- No normalized case schema exists in the checked-in fuzzer. Family-specific
  dataclasses do not carry a shared expected class, `num_ctas`, `two_cta`,
  row/col permutation, register variant, or view-chain taxonomy.
- Current ids are repro-history ids, not descriptor ids. This blocks mechanical
  diffing between generated inventory and checked-in coverage.
- There are no checked-in structural-fuzzer cases for `mma` or `mma_scaled`.
  Those families are covered elsewhere in `test_tmem_runtime_matrix.py`, but
  not in the structural fuzzer's compact promotion harness.
- Copy coverage is only `TensorMemoryScalesLayout` direct `warpx4` single-CTA
  and two-CTA. Missing structural dimensions include non-scales copy,
  `warpx2::{01_23,02_13}`, indexed/subslice destination views, larger-CGA
  clean diagnostics, packed/subword copy boundaries, and copy->readback
  combinations.
- `num_ctas > 2` is not represented as a first-class structural-fuzzer axis.
  Larger-CGA clean diagnostics are recorded in reports and runtime matrix
  slices, but not in the compact fuzzer.
- `ldst` coverage has useful positives and strict xfails, but not a systematic
  row/col/dtype/packet grid. Roundtrip positives can hide bad address
  arithmetic, so read-only descriptor-view rows should be generated
  separately from same-view roundtrip rows.
- `ldred` coverage has strong xfails for opcode fallback and the false
  unsupported candidate, but lacks a normalized split between direct/indexed,
  descriptor-chain, row/col permutation, 1CTA/2CTA, and reduction modifier
  axes.
- Dynamic descriptor provenance (`dynamic_index`, `if_result`,
  `loop_carried`) is currently only represented under generic-pass buckets,
  not cross-producted into copy/MMA/scaled-MMA policy as clean diagnostics or
  explicit deferred contracts.

## Recommendations

1. Promote a small `StructuralCaseDescriptor` schema before promoting the full
   generator. Keep it data-only and independent of Gluon imports so collect and
   coverage reporting stay cheap.
2. Add a `legacy_case_id` or `source_case_id` field when migrating existing
   rows. Preserve current pytest ids initially, but make the normalized
   descriptor id available for generated-inventory comparisons.
3. Separate generated descriptors from runnable kernels:
   - descriptor inventory and pruning can be CPU-only;
   - runnable adapters can remain family-specific;
   - unsupported/deferred descriptors should still appear in the inventory with
     an expected class.
4. Add first runnable generator-backed slices in this order:
   - `ldred` direct/indexed 1CTA/2CTA opcode rows, because the current
     fallback bucket is active and well understood;
   - `ldst` read-only descriptor-view rows, because roundtrip controls can
     mask failures;
   - copy descriptor-view and `warpx2` readback rows;
   - compact MMA/scaled-MMA descriptor-view control-flow rows that initially
     assert clean diagnostics or defer explicitly.
5. Keep the generated case count bounded through tiering:
   - tier 0: smoke positives and known strict xfails;
   - tier 1: one representative per family/view/CTA/expected-class bucket;
   - tier 2: duration-aware broad sweeps across row/col permutations and dtype
     variants.

## Promotion Status

Do not promote `/tmp/tmem_structural_case_generator_round6.py` yet. The next
repo-ready slice should add only the schema and inventory helper, plus one
family adapter, after the campaign decides whether structural fuzzer ids should
switch to normalized descriptors or keep legacy pytest ids with descriptor
metadata attached.
