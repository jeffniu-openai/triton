# Round 9 Lane E: Normalized TMEM Structural Generator Inventory

- Date: 2026-04-21
- Branch: `codex/tmem`
- Mode: discovery-only support work. No backend/compiler repairs attempted.
- Repo edits: this report only.
- Temporary prototype: `/tmp/tmem_generator_inventory_round9.py`
- JSON inventory: `/tmp/tmem_generator_inventory_round9.json`

## Objective

Build a temporary normalized structural case generator prototype under `/tmp`,
informed by the Round 6 generator and the current checked-in structural fuzzer.
This lane focused on data-only descriptor coverage inventory plus a small
runnable sample, not repo promotion.

The prototype enumerates normalized descriptors for:

- family: `ldst`, `ldred`, `copy`, `mma`, `mma_scaled`;
- normalized id;
- legacy case linkage;
- expected class;
- shape;
- dtype;
- `two_cta`;
- `num_ctas`;
- view chain;
- row/column layout kind; and
- register/instruction variant.

## Prototype

The Round 9 prototype is still `/tmp`-only:

```bash
/tmp/tmem_generator_inventory_round9.py
/tmp/tmem_generator_inventory_round9.json
```

Unlike the Round 6 prototype, it keeps two separate identities:

- `normalized_id`, a stable descriptor id such as
  `ldred-f32-2cta-ncta2-256x32-index-identity-identity-32x32b`; and
- `legacy_case_ids`, links back to repro-style checked-in pytest ids such as
  `ldred-fz20260421-0004-twocta-indexed-256x32-chain0-min`.

That split is important because the checked-in structural fuzzer is currently
organized around promoted bug history, while generator coverage needs a stable
schema that can be diffed mechanically.

## Commands

Required build before runnable probes:

```bash
CPLUS_INCLUDE_PATH=/usr/include/c++/13:/usr/include/aarch64-linux-gnu/c++/13:/usr/lib/gcc/aarch64-linux-gnu/13/include make -j8
```

Result:

```text
ninja: no work to do.
```

Prototype syntax check:

```bash
PYTHONPATH=.:./python python -m py_compile /tmp/tmem_generator_inventory_round9.py
```

Result: passed.

Inventory generation:

```bash
PYTHONPATH=.:./python /tmp/tmem_generator_inventory_round9.py \
  --repo /root/code/triton \
  --json-out /tmp/tmem_generator_inventory_round9.json \
  --limit 32
```

Structural-fuzzer collect-only comparison:

```bash
PYTHONPATH=.:./python pytest --collect-only -q \
  python/test/gluon/test_tmem_structural_fuzzer.py
```

Result:

```text
33 tests collected in 2.60s
```

Small runnable sample:

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-cache-round9-lane-e \
PYTHONPATH=.:./python pytest -s --tb=short \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldst_view_roundtrip[ldst-view-identity-32x32b]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred[ldred-direct-128x64]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_copy_scales[copy-scales-warpx4-1cta]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_scaled_mma_acc_subslice_control_flow[mma-scaled-fz20260421-0007-subslice-if-n64-selector0]' \
  'python/test/gluon/test_tmem_structural_fuzzer.py::test_tmem_structural_fuzzer_ldred_1cta_direct_index_allocator_crash'
```

Result:

```text
3 passed, 2 xfailed in 6.96s
```

The two xfails are existing strict sentinels for `FZ-20260421-0007` and
`FZ-20260421-0009`; no new backend/compiler failure was discovered.

## Inventory Counts

Prototype summary:

```json
{
  "axis_buckets_total": 46,
  "axis_buckets_with_legacy_link": 18,
  "checked_in_legacy_ids": 33,
  "checked_in_legacy_ids_missing_round9_link": [],
  "checked_in_legacy_ids_with_round9_link": 33,
  "generated_by_expected_class": {
    "clean_error": 3,
    "clean_unsupported": 4856,
    "defer_unchecked": 750,
    "known_failure": 10590,
    "pass": 27844
  },
  "generated_by_family": {
    "copy": 14643,
    "ldred": 9690,
    "ldst": 17595,
    "mma": 675,
    "mma_scaled": 1440
  },
  "generated_total": 44043,
  "generated_with_legacy_link": 460,
  "generated_with_legacy_link_by_family": {
    "copy": 24,
    "ldred": 140,
    "ldst": 280,
    "mma_scaled": 16
  }
}
```

Interpretation:

- all 33 checked-in structural-fuzzer legacy ids have a Round 9 normalized
  linkage;
- 460 generated descriptors map to at least one legacy checked-in case through
  normalized axes;
- the current compact fuzzer covers 18 of 46 generated
  `(family, view_chain, expected_class)` buckets; and
- no checked-in `mma` structural-fuzzer row exists yet, while one
  `mma_scaled` strict xfail maps to the normalized scaled-MMA accumulator
  subslice bucket.

## Largest Coverage Holes

Top uncovered buckets by normalized inventory count:

| Family | View chain | Expected class | Count |
| --- | --- | ---: | ---: |
| `ldst` | `subslice` | `pass` | 2484 |
| `ldst` | `dynamic_index` | `known_failure` | 2464 |
| `ldst` | `loop_carried` | `known_failure` | 2464 |
| `ldst` | `if_result` | `known_failure` | 2444 |
| `ldst` | `index` | `pass` | 1928 |
| `copy` | `dynamic_index` | `pass` | 1920 |
| `copy` | `if_result` | `pass` | 1920 |
| `copy` | `index` | `pass` | 1920 |
| `copy` | `slice_index` | `pass` | 1920 |
| `copy` | `subslice` | `pass` | 1920 |
| `ldst` | `double_transpose_slice` | `pass` | 1584 |
| `ldst` | `reshape_permute` | `pass` | 1560 |
| `ldred` | `dynamic_index` | `pass` | 1368 |
| `ldred` | `if_result` | `pass` | 1368 |
| `ldred` | `loop_carried` | `pass` | 1368 |
| `ldred` | `subslice` | `pass` | 1368 |
| `ldred` | `transpose_slice` | `pass` | 1336 |
| `ldred` | `reshape_permute` | `pass` | 1174 |
| `copy` | `dynamic_index` | `clean_unsupported` | 960 |
| `copy` | `if_result` | `clean_unsupported` | 960 |

The first descriptor-level holes printed by the prototype are dominated by
copy subword/packed clean-diagnostic rows, for example:

```text
copy-f16-1cta-ncta1-128x128-index-even_odd-identity-warpx2::01_23
copy-f16-1cta-ncta1-128x128-index-even_odd-identity-warpx2::02_13
copy-f16-1cta-ncta1-128x128-index-identity-identity-warpx2::01_23
copy-f16-1cta-ncta1-128x128-index-identity-identity-warpx2::02_13
```

Those should not become the first repo promotion target merely because they
sort first. They are useful clean-boundary inventory, but the more valuable
runnable-generator targets are the buckets below.

## Recommendations

1. Keep the generator out of the repo for now. The normalized schema and
   legacy-link table are useful, but the prototype still lacks first-class
   runnable adapters per family.
2. Promote a small data schema before a broad generator: `normalized_id`,
   `legacy_case_ids`, `expected_class`, `shape`, `dtype`, `two_cta`,
   `num_ctas`, `view_chain`, `row_kind`, `col_kind`, and `reg_variant`.
3. Add runnable adapters in this order:
   - `ldred` direct/indexed/chained rows around `FZ-20260421-0004`,
     `FZ-20260421-0008`, and `FZ-20260421-0009`, because existing reports
     already separate opcode fallback, optimizer abort, and allocator abort;
   - `ldst` read-only descriptor-view rows, especially `subslice`, `index`,
     and `reshape_permute`, because same-view roundtrip rows can hide address
     arithmetic bugs;
   - `copy` direct/view readback rows for `warpx2::{01_23,02_13}` plus
     explicit packed/subword clean diagnostics;
   - `mma` structural rows, which are absent from the compact fuzzer even
     though related coverage exists in runtime-matrix tests; and
   - `mma_scaled` accumulator-view rows, preserving the existing
     `FZ-20260421-0007` link while adding positive controls.
4. Use bucket-tiering before broad execution:
   - tier 0: one positive and one known/clean diagnostic per family;
   - tier 1: one row per `(family, view_chain, expected_class)` bucket;
   - tier 2: row/col/dtype/register cross-products selected by duration-aware
     splitting.
5. Keep `legacy_case_ids` during migration. Do not rewrite checked-in pytest
   ids to normalized ids until the adapters can prove equivalent coverage and
   report old-to-new links.

## Classification

No new `FZ-*` id is warranted from this lane. This was discovery-support
inventory work, not a backend/compiler bug-finding lane. The runnable sample
only exercised existing pass rows and existing strict xfail sentinels.
