from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterator

from triton.runtime.errors import OutOfResources

from . import bench_matmul_parrot_gather as bench


X_ENV = "TRITON_WS_SPLIT_SF_X_NUM_BUFS"
W_ENV = "TRITON_WS_SPLIT_SF_W_NUM_BUFS"
SCALE_ENV = "TRITON_WS_SPLIT_SF_SCALE_NUM_BUFS"


@dataclass(frozen=True)
class BufferConfig:
    x_num_bufs: int
    w_num_bufs: int
    scale_num_bufs: int

    @property
    def label(self) -> str:
        return f"x={self.x_num_bufs},w={self.w_num_bufs},s={self.scale_num_bufs}"


@dataclass(frozen=True)
class SweepResult:
    config: BufferConfig
    status: str
    runtime_ms: float | None
    tflops: float | None
    tbps: float | None
    validated: bool
    error: str | None = None
    required_shared: int | None = None
    shared_limit: int | None = None


def parse_range(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            values.append(int(part))
            continue
        fields = [int(field) for field in part.split(":")]
        if len(fields) not in (2, 3):
            raise ValueError(f"Invalid range spec {part!r}")
        start, stop = fields[:2]
        step = fields[2] if len(fields) == 3 else 1
        if step == 0:
            raise ValueError(f"Invalid zero step in range spec {part!r}")
        if step > 0:
            stop_inclusive = stop + 1
        else:
            stop_inclusive = stop - 1
        values.extend(range(start, stop_inclusive, step))
    if not values:
        raise ValueError("Range spec produced no values")
    return sorted(set(values))


@contextmanager
def override_ws_split_sf_env(config: BufferConfig) -> Iterator[None]:
    prev = {
        X_ENV: os.environ.get(X_ENV),
        W_ENV: os.environ.get(W_ENV),
        SCALE_ENV: os.environ.get(SCALE_ENV),
    }
    os.environ[X_ENV] = str(config.x_num_bufs)
    os.environ[W_ENV] = str(config.w_num_bufs)
    os.environ[SCALE_ENV] = str(config.scale_num_bufs)
    try:
        yield
    finally:
        for name, value in prev.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def classify_error(exc: Exception) -> tuple[str, str, int | None, int | None]:
    if isinstance(exc, OutOfResources):
        if exc.name == "shared memory":
            return "oor_shared", str(exc), int(exc.required), int(exc.limit)
        return "oor", str(exc), int(exc.required), int(exc.limit)
    return "error", repr(exc), None, None


def benchmark_config(
    prepared: bench.PreparedCase,
    config: BufferConfig,
    rep: int,
    peak_fp8_tflops: float,
    peak_mem_tbps: float,
    validation_reference: tuple[str, tuple[object, object]] | None,
) -> SweepResult:
    with override_ws_split_sf_env(config):
        try:
            result = bench.benchmark_prepared_case(
                prepared,
                bench.WS_SPLIT_SF_KERNEL_NAME,
                warmup=0,
                rep=rep,
                peak_fp8_tflops=peak_fp8_tflops,
                peak_mem_tbps=peak_mem_tbps,
                validation_reference=validation_reference,
            )
        except Exception as exc:
            status, message, required, limit = classify_error(exc)
            return SweepResult(
                config=config,
                status=status,
                runtime_ms=None,
                tflops=None,
                tbps=None,
                validated=False,
                error=message,
                required_shared=required,
                shared_limit=limit,
            )

    return SweepResult(
        config=config,
        status="ok",
        runtime_ms=result.runtime_ms,
        tflops=result.tflops,
        tbps=result.tbps,
        validated=result.validated,
    )


def format_result(result: SweepResult) -> str:
    cfg = result.config
    if result.status == "ok":
        return (
            f"{cfg.x_num_bufs:>2} | {cfg.w_num_bufs:>2} | {cfg.scale_num_bufs:>2} | "
            f"{result.runtime_ms:>8.4f} | {result.tflops:>8.2f} | {result.tbps:>6.2f} | "
            f"{'yes' if result.validated else 'no ':>3} | ok"
        )
    if result.required_shared is not None and result.shared_limit is not None:
        return (
            f"{cfg.x_num_bufs:>2} | {cfg.w_num_bufs:>2} | {cfg.scale_num_bufs:>2} | "
            f"{'-':>8} | {'-':>8} | {'-':>6} | {'no ':>3} | "
            f"{result.status} ({result.required_shared}>{result.shared_limit})"
        )
    return (
        f"{cfg.x_num_bufs:>2} | {cfg.w_num_bufs:>2} | {cfg.scale_num_bufs:>2} | "
        f"{'-':>8} | {'-':>8} | {'-':>6} | {'no ':>3} | {result.status}"
    )


def write_csv(path: Path, results: list[SweepResult], rerun_results: dict[BufferConfig, SweepResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "x_num_bufs",
                "w_num_bufs",
                "scale_num_bufs",
                "status",
                "runtime_ms",
                "tflops",
                "tbps",
                "validated",
                "required_shared",
                "shared_limit",
                "error",
                "rerun_status",
                "rerun_runtime_ms",
                "rerun_tflops",
                "rerun_tbps",
                "rerun_validated",
                "rerun_required_shared",
                "rerun_shared_limit",
                "rerun_error",
            ]
        )
        for result in results:
            rerun = rerun_results.get(result.config)
            writer.writerow(
                [
                    result.config.x_num_bufs,
                    result.config.w_num_bufs,
                    result.config.scale_num_bufs,
                    result.status,
                    None if result.runtime_ms is None else f"{result.runtime_ms:.8f}",
                    None if result.tflops is None else f"{result.tflops:.8f}",
                    None if result.tbps is None else f"{result.tbps:.8f}",
                    str(result.validated).lower(),
                    result.required_shared,
                    result.shared_limit,
                    result.error or "",
                    "" if rerun is None else rerun.status,
                    "" if rerun is None or rerun.runtime_ms is None else f"{rerun.runtime_ms:.8f}",
                    "" if rerun is None or rerun.tflops is None else f"{rerun.tflops:.8f}",
                    "" if rerun is None or rerun.tbps is None else f"{rerun.tbps:.8f}",
                    "" if rerun is None else str(rerun.validated).lower(),
                    "" if rerun is None else rerun.required_shared,
                    "" if rerun is None else rerun.shared_limit,
                    "" if rerun is None or rerun.error is None else rerun.error,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep ws_split_sf x/w/scale buffer-count combinations on one prepared parrot-gather case."
    )
    parser.add_argument("--case-family", choices=("non-parrot", "parrot"), default="non-parrot")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--peak-fp8-tflops", type=float, default=bench.DEFAULT_PEAK_FP8_TFLOPS)
    parser.add_argument("--peak-mem-tbps", type=float, default=bench.DEFAULT_PEAK_MEM_TBPS)
    parser.add_argument("--rep", type=int, default=15, help="Broad-pass cudagraph reps per config.")
    parser.add_argument("--rerun-rep", type=int, default=60, help="Higher-fidelity reps for top configs.")
    parser.add_argument("--top", type=int, default=10, help="Number of top broad-pass configs to rerun.")
    parser.add_argument("--x-range", type=str, default="2:6")
    parser.add_argument("--w-range", type=str, default="2:5")
    parser.add_argument("--scale-range", type=str, default="1:8")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise ValueError(f"Only cuda is supported, got {args.device}")
    if args.rep <= 0:
        raise ValueError(f"--rep must be > 0, got {args.rep}")
    if args.rerun_rep <= 0:
        raise ValueError(f"--rerun-rep must be > 0, got {args.rerun_rep}")
    if args.top < 0:
        raise ValueError(f"--top must be >= 0, got {args.top}")

    cases = bench.make_cases(args.case_family, args.batch_size, args.batch_size, limit=1)
    if len(cases) != 1:
        raise ValueError(f"Expected exactly one case for batch_size={args.batch_size}, got {len(cases)}")
    case = cases[0]
    prepared = bench.prepare_case(case, device=args.device, seed=args.seed, local_rank_override=args.local_rank)

    validation_reference = None
    if args.validate:
        validation_reference = (bench.ORIGINAL_KERNEL_NAME, bench.run_case_once(prepared, bench.ORIGINAL_KERNEL_NAME))

    print(
        f"case={case.case_id} rank={prepared.local_rank} x_range={args.x_range} "
        f"w_range={args.w_range} scale_range={args.scale_range} rep={args.rep} rerun_rep={args.rerun_rep}"
    )

    baseline_names = (
        bench.WS_KERNEL_NAME,
        bench.WS_OPTIMIZED_KERNEL_NAME,
        bench.GLUON_OPTIMIZED_KERNEL_NAME,
    )
    print("baseline | kernel          |       ms |  TFLOP/s |   TB/s")
    for kernel_name in baseline_names:
        baseline = bench.benchmark_prepared_case(
            prepared,
            kernel_name,
            warmup=0,
            rep=args.rerun_rep,
            peak_fp8_tflops=args.peak_fp8_tflops,
            peak_mem_tbps=args.peak_mem_tbps,
            validation_reference=validation_reference if kernel_name != bench.ORIGINAL_KERNEL_NAME else None,
        )
        print(
            f"baseline | {kernel_name:>14} | {baseline.runtime_ms:>8.4f} | "
            f"{baseline.tflops:>8.2f} | {baseline.tbps:>6.2f}"
        )

    configs = [
        BufferConfig(x_num_bufs=x, w_num_bufs=w, scale_num_bufs=s)
        for x in parse_range(args.x_range)
        for w in parse_range(args.w_range)
        for s in parse_range(args.scale_range)
    ]
    print(f"sweeping {len(configs)} configs")
    print("  x |  w |  s |       ms |  TFLOP/s |   TB/s | val | status")

    results: list[SweepResult] = []
    for idx, config in enumerate(configs, start=1):
        result = benchmark_config(
            prepared,
            config,
            rep=args.rep,
            peak_fp8_tflops=args.peak_fp8_tflops,
            peak_mem_tbps=args.peak_mem_tbps,
            validation_reference=validation_reference,
        )
        results.append(result)
        print(f"{idx:>3}/{len(configs):<3} {format_result(result)}")

    ok_results = [result for result in results if result.status == "ok" and result.runtime_ms is not None]
    ok_results.sort(key=lambda result: result.runtime_ms)

    rerun_results: dict[BufferConfig, SweepResult] = {}
    if args.top > 0 and ok_results:
        top_results = ok_results[: min(args.top, len(ok_results))]
        print(f"rerunning top {len(top_results)} configs with rep={args.rerun_rep}")
        print("rank |  x |  w |  s | broad_ms | rerun_ms | delta_ms")
        for rank, broad in enumerate(top_results, start=1):
            rerun = benchmark_config(
                prepared,
                broad.config,
                rep=args.rerun_rep,
                peak_fp8_tflops=args.peak_fp8_tflops,
                peak_mem_tbps=args.peak_mem_tbps,
                validation_reference=validation_reference,
            )
            rerun_results[broad.config] = rerun
            delta = None if rerun.runtime_ms is None or broad.runtime_ms is None else rerun.runtime_ms - broad.runtime_ms
            delta_str = "-" if delta is None else f"{delta:+8.4f}"
            rerun_ms = "-" if rerun.runtime_ms is None else f"{rerun.runtime_ms:8.4f}"
            print(
                f"{rank:>4} | {broad.config.x_num_bufs:>2} | {broad.config.w_num_bufs:>2} | "
                f"{broad.config.scale_num_bufs:>2} | {broad.runtime_ms:>8.4f} | {rerun_ms} | {delta_str}"
            )

    if ok_results:
        print("top broad-pass configs")
        print("rank |  x |  w |  s |       ms |  TFLOP/s |   TB/s")
        for rank, result in enumerate(ok_results[: min(10, len(ok_results))], start=1):
            print(
                f"{rank:>4} | {result.config.x_num_bufs:>2} | {result.config.w_num_bufs:>2} | "
                f"{result.config.scale_num_bufs:>2} | {result.runtime_ms:>8.4f} | "
                f"{result.tflops:>8.2f} | {result.tbps:>6.2f}"
            )
    else:
        print("no valid configs completed")

    if args.csv_out is not None:
        write_csv(args.csv_out, results, rerun_results)
        print(f"wrote {len(results)} rows to {args.csv_out}")


if __name__ == "__main__":
    main()
