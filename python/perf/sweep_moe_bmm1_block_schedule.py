from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from triton.testing import do_bench_cudagraph

from .bench_matmul_parrot_gather import (
    EXACT_REFERENCE_NAME,
    build_validation_reference,
    compute_matmul_proton_metrics,
    format_progress,
    make_cases,
    make_output_buffer,
    make_precision_config,
    normalize_output_tensor,
    prepare_case,
    validate_case_outputs,
)
from .matmul_ws_optimized import BLOCK_SCHEDULE_STRATEGIES, matmul as matmul_ws_optimized


@dataclass(frozen=True)
class StrategyResult:
    strategy_name: str
    strategy_id: int
    runtime_ms: float
    tflops: float
    tbps: float
    validated: bool


def resolve_case(case_family: str, batch_size: int, n_expts_tot: int, n_expts_shards: int):
    cases = make_cases(case_family, batch_size, batch_size, None)
    matches = [c for c in cases if c.n_expts_tot == n_expts_tot and c.n_expts_shards == n_expts_shards]
    if not matches:
        raise ValueError(
            f"No case matched family={case_family}, batch_size={batch_size}, "
            f"n_expts_tot={n_expts_tot}, n_expts_shards={n_expts_shards}"
        )
    if len(matches) != 1:
        raise ValueError(f"Expected a unique case, found {len(matches)}")
    return matches[0]


def parse_strategy_names(strategy_spec: str) -> list[str]:
    if strategy_spec == "all":
        return list(BLOCK_SCHEDULE_STRATEGIES)
    names = [name.strip() for name in strategy_spec.split(",") if name.strip()]
    unknown = [name for name in names if name not in BLOCK_SCHEDULE_STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")
    return names


def benchmark_strategy(
    prepared,
    strategy_name: str,
    strategy_id: int,
    warmup: int,
    rep: int,
    validate: bool,
):
    precision_config = make_precision_config(prepared)
    out = make_output_buffer(prepared)

    def run() -> torch.Tensor:
        return matmul_ws_optimized(
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

    y = normalize_output_tensor(run())
    validated = False
    if validate:
        validation_reference = build_validation_reference(prepared, EXACT_REFERENCE_NAME)
        validate_case_outputs(
            prepared.case,
            strategy_name,
            (y, precision_config),
            validation_reference[0],
            validation_reference[1],
        )
        validated = True

    _ = warmup
    runtime_ms = float(do_bench_cudagraph(run, rep=rep))
    metrics = compute_matmul_proton_metrics(
        prepared.x,
        prepared.w,
        out,
        prepared.ragged_batch_metadata,
        n=prepared.case.n,
        k=prepared.case.k,
    )
    tflops = metrics.flops / runtime_ms / 1e9
    tbps = metrics.bytes / runtime_ms / 1e9
    return StrategyResult(
        strategy_name=strategy_name,
        strategy_id=strategy_id,
        runtime_ms=runtime_ms,
        tflops=tflops,
        tbps=tbps,
        validated=validated,
    )


def write_csv(path: Path, prepared, results: list[StrategyResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "local_rank",
                "strategy_name",
                "strategy_id",
                "runtime_ms",
                "tflops",
                "tbps",
                "validated",
                "expected_slice_size",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    prepared.case.case_id,
                    prepared.local_rank,
                    result.strategy_name,
                    result.strategy_id,
                    f"{result.runtime_ms:.8f}",
                    f"{result.tflops:.8f}",
                    f"{result.tbps:.8f}",
                    str(result.validated).lower(),
                    prepared.ragged_batch_metadata.expected_slice_size,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep block-schedule strategies for the WS MoE BMM1 example.")
    parser.add_argument("--case-family", choices=("non-parrot", "parrot"), default="non-parrot")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--n-expts-tot", type=int, default=256)
    parser.add_argument("--n-expts-shards", type=int, default=8)
    parser.add_argument("--strategies", type=str, default="all")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rep", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise ValueError(f"Only cuda is supported, got {args.device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep")

    case = resolve_case(args.case_family, args.batch_size, args.n_expts_tot, args.n_expts_shards)
    prepared = prepare_case(case, device=args.device, seed=args.seed, local_rank_override=args.local_rank)
    strategy_names = parse_strategy_names(args.strategies)

    print(
        f"case={prepared.case.case_id} local_rank={prepared.local_rank} "
        f"expected_slice_size={prepared.ragged_batch_metadata.expected_slice_size}"
    )
    results: list[StrategyResult] = []
    best_runtime = None
    for idx, strategy_name in enumerate(strategy_names, start=1):
        strategy_id = BLOCK_SCHEDULE_STRATEGIES[strategy_name]
        result = benchmark_strategy(
            prepared,
            strategy_name=strategy_name,
            strategy_id=strategy_id,
            warmup=args.warmup,
            rep=args.rep,
            validate=args.validate,
        )
        results.append(result)
        if best_runtime is None or result.runtime_ms < best_runtime:
            best_runtime = result.runtime_ms
        speedup_vs_best = 1.0 if best_runtime is None else best_runtime / result.runtime_ms
        print(
            f"{format_progress(idx, len(strategy_names))} | {strategy_name:>13} | "
            f"id={strategy_id:>2} | ms={result.runtime_ms:>8.4f} | "
            f"TFLOP/s={result.tflops:>8.2f} | TB/s={result.tbps:>6.2f} | "
            f"vs_best={speedup_vs_best:>6.3f}x | "
            f"{'ok(exact)' if result.validated else ''}"
        )

    if args.csv_out is not None:
        write_csv(args.csv_out, prepared, results)
        print(f"Wrote {len(results)} rows to {args.csv_out}")


if __name__ == "__main__":
    main()
