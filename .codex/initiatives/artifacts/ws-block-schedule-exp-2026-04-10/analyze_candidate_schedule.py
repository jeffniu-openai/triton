from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from triton.testing import do_bench_cudagraph


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TRITON_KERNELS_ROOT = REPO_ROOT / "python" / "triton_kernels"
if str(TRITON_KERNELS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRITON_KERNELS_ROOT))

from python.perf.bench_matmul_parrot_gather import (  # noqa: E402
    EXACT_REFERENCE_NAME,
    build_validation_reference,
    make_cases,
    make_output_buffer,
    make_precision_config,
    normalize_output_tensor,
    prepare_case,
    validate_case_outputs,
)


def load_experiment_module():
    module_name = "ws_block_schedule_experiment_module_analysis"
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = Path(__file__).with_name("moe_bmm1_fused_gather_exp.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_experiment = load_experiment_module()


@dataclass(frozen=True)
class BenchResult:
    round_idx: int
    order: str
    row_major_ms: float
    candidate_ms: float

    @property
    def delta_ms(self) -> float:
        return self.row_major_ms - self.candidate_ms


def resolve_case(case_family: str, batch_size: int, n_expts_tot: int, n_expts_shards: int):
    cases = make_cases(case_family, batch_size, batch_size, None)
    matches = [c for c in cases if c.n_expts_tot == n_expts_tot and c.n_expts_shards == n_expts_shards]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one matching case, found {len(matches)}")
    return matches[0]


def build_runner(prepared, strategy_name: str, strategy_id: int):
    precision_config = make_precision_config(prepared)
    out = make_output_buffer(prepared)

    def run():
        return _experiment.matmul(
            a=prepared.x,
            b=prepared.w,  # type: ignore[arg-type]
            bias=prepared.bias,
            a_ragged_metadata=prepared.ragged_batch_metadata,
            gather_indx=prepared.gather_indx,
            precision_config=precision_config,
            c=out,
            fused_activation=prepared.fused_activation,
            block_schedule_strategy=strategy_id,
        )

    return run, precision_config


def warmup_runner(run, warmup: int) -> None:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()


def validate_runner(prepared, strategy_name: str, run, precision_config) -> None:
    validation_reference = build_validation_reference(prepared, EXACT_REFERENCE_NAME)
    y = normalize_output_tensor(run())
    validate_case_outputs(
        prepared.case,
        strategy_name,
        (y, precision_config),
        validation_reference[0],
        validation_reference[1],
    )


def bootstrap_ci(deltas: list[float], trials: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(trials):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo_idx = int(0.025 * (trials - 1))
    hi_idx = int(0.975 * (trials - 1))
    return means[lo_idx], means[hi_idx]


def sign_test_two_sided(deltas: list[float]) -> float:
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def write_csv(path: Path, rows: list[BenchResult]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round_idx", "order", "row_major_ms", "candidate_ms", "delta_ms"])
        for row in rows:
            writer.writerow(
                [
                    row.round_idx,
                    row.order,
                    f"{row.row_major_ms:.8f}",
                    f"{row.candidate_ms:.8f}",
                    f"{row.delta_ms:.8f}",
                ]
            )


def write_markdown(
    path: Path,
    rows: list[BenchResult],
    candidate_name: str,
    rep: int,
    warmup: int,
    bootstrap_trials: int,
    bootstrap_seed: int,
):
    row_times = [row.row_major_ms for row in rows]
    cand_times = [row.candidate_ms for row in rows]
    deltas = [row.delta_ms for row in rows]
    mean_row = statistics.fmean(row_times)
    mean_cand = statistics.fmean(cand_times)
    mean_delta = statistics.fmean(deltas)
    median_delta = statistics.median(deltas)
    std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    ci_lo, ci_hi = bootstrap_ci(deltas, bootstrap_trials, bootstrap_seed)
    sign_p = sign_test_two_sided(deltas)
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    mean_pct = (mean_delta / mean_row) * 100.0
    significant = ci_lo > 0.0 or ci_hi < 0.0

    path.write_text(
        "\n".join(
            [
                "# Candidate Schedule vs Row-Major",
                "",
                f"- candidate: `{candidate_name}`",
                f"- rounds: `{len(rows)}`",
                f"- rep per timing call: `{rep}`",
                f"- manual warmup calls before each timing: `{warmup}`",
                f"- order: alternated every round",
                "",
                "## Summary",
                "",
                f"- mean row_major: `{mean_row:.8f} ms`",
                f"- mean {candidate_name}: `{mean_cand:.8f} ms`",
                f"- mean delta (`row_major - candidate`): `{mean_delta:.8f} ms`",
                f"- mean relative edge for candidate: `{mean_pct:.4f}%`",
                f"- median delta: `{median_delta:.8f} ms`",
                f"- delta stddev: `{std_delta:.8f} ms`",
                f"- bootstrap 95% CI for mean delta: `[{ci_lo:.8f}, {ci_hi:.8f}] ms`",
                f"- sign test (two-sided) p-value: `{sign_p:.6f}`",
                f"- candidate wins / losses / ties: `{wins} / {losses} / {ties}`",
                f"- CI-excludes-zero: `{'yes' if significant else 'no'}`",
                "",
                "## Interpretation",
                "",
                (
                    f"The repeated long-run comparison {'supports' if significant else 'does not yet support'} "
                    f"a statistically clear difference between `row_major` and `{candidate_name}` under this setup."
                ),
                "",
                f"Bootstrap settings: `{bootstrap_trials}` resamples, seed `{bootstrap_seed}`.",
                "",
                "See the sibling CSV for raw per-round timings.",
            ]
        )
        + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Repeated long-run compare of candidate block schedule vs row-major.")
    parser.add_argument("--case-family", choices=("non-parrot", "parrot"), default="non-parrot")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--n-expts-tot", type=int, default=256)
    parser.add_argument("--n-expts-shards", type=int, default=8)
    parser.add_argument("--candidate", type=str, default="band_n_20_row_major")
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--rep", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--bootstrap-trials", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate not in _experiment.BLOCK_SCHEDULE_STRATEGIES:
        raise ValueError(f"Unknown candidate strategy: {args.candidate}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    case = resolve_case(args.case_family, args.batch_size, args.n_expts_tot, args.n_expts_shards)
    prepared = prepare_case(case, device="cuda", seed=args.seed, local_rank_override=args.local_rank)

    row_run, row_precision = build_runner(prepared, "row_major", _experiment.BLOCK_SCHEDULE_ROW_MAJOR)
    cand_run, cand_precision = build_runner(
        prepared,
        args.candidate,
        _experiment.BLOCK_SCHEDULE_STRATEGIES[args.candidate],
    )

    validate_runner(prepared, "row_major", row_run, row_precision)
    validate_runner(prepared, args.candidate, cand_run, cand_precision)

    rows: list[BenchResult] = []
    for round_idx in range(args.rounds):
        order = [("row_major", row_run), (args.candidate, cand_run)]
        if round_idx % 2 == 1:
            order.reverse()

        times = {}
        order_names = []
        for name, run in order:
            warmup_runner(run, args.warmup)
            times[name] = float(do_bench_cudagraph(run, rep=args.rep))
            order_names.append(name)

        rows.append(
            BenchResult(
                round_idx=round_idx,
                order="->".join(order_names),
                row_major_ms=times["row_major"],
                candidate_ms=times[args.candidate],
            )
        )
        print(
            f"[{round_idx + 1:>2}/{args.rounds:>2}] {rows[-1].order:>29} | "
            f"row_major={rows[-1].row_major_ms:>10.6f} ms | "
            f"{args.candidate}={rows[-1].candidate_ms:>10.6f} ms | "
            f"delta={rows[-1].delta_ms:>+10.6f} ms"
        )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.csv_out, rows)
    write_markdown(
        args.md_out,
        rows,
        candidate_name=args.candidate,
        rep=args.rep,
        warmup=args.warmup,
        bootstrap_trials=args.bootstrap_trials,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.md_out}")


if __name__ == "__main__":
    main()
