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


ENV_DEFAULTS = {
    "TRITON_WS_NUM_WARPS": "8",
    "TRITON_WS_X_NUM_BUFS": "5",
    "TRITON_WS_W_NUM_BUFS": "4",
    "TRITON_WS_LOAD_WEIGHT_WARPS": "1",
    "TRITON_WS_MMA_WARPS": "1",
    "TRITON_WS_USE_EPILOGUE_STORE_HELPER": "1",
    "TRITON_WS_USE_HELPER_PACKED_OUT_BUFFER": "0",
    "TRITON_WS_USE_WIDE_PACKED_STORE32": "1",
    "TRITON_WS_USE_WIDE_PACKED_STORE64": "0",
}


@dataclass(frozen=True)
class EpiConfig:
    store_helper_warps: int
    store_helper_regs: int
    row_subtile_factor: int
    helper_depth: int
    store_mode: str

    @property
    def load_activation_warps(self) -> int:
        return 6 - self.store_helper_warps

    @property
    def label(self) -> str:
        return (
            f"helper={self.store_helper_warps},load={self.load_activation_warps},"
            f"regs={self.store_helper_regs},rows={self.row_subtile_factor},"
            f"depth={self.helper_depth},store={self.store_mode}"
        )


@dataclass(frozen=True)
class SweepResult:
    config: EpiConfig
    status: str
    runtime_ms: float | None
    tflops: float | None
    tbps: float | None
    validated: bool
    error: str | None = None
    required_shared: int | None = None
    shared_limit: int | None = None


def parse_int_list(spec: str) -> list[int]:
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
        stop_inclusive = stop + 1 if step > 0 else stop - 1
        values.extend(range(start, stop_inclusive, step))
    if not values:
        raise ValueError("Range spec produced no values")
    return sorted(set(values))


def parse_store_modes(spec: str) -> list[str]:
    modes = [part.strip() for part in spec.split(",") if part.strip()]
    if not modes:
        raise ValueError("No store modes specified")
    for mode in modes:
        if mode not in {"narrow", "wide32", "wide64"}:
            raise ValueError(f"Unsupported store mode {mode!r}")
    return modes


def classify_error(exc: Exception) -> tuple[str, str, int | None, int | None]:
    if isinstance(exc, OutOfResources):
        if exc.name == "shared memory":
            return "oor_shared", str(exc), int(exc.required), int(exc.limit)
        return "oor", str(exc), int(exc.required), int(exc.limit)
    return "error", repr(exc), None, None


@contextmanager
def override_env(config: EpiConfig) -> Iterator[None]:
    prev = {name: os.environ.get(name) for name in ENV_DEFAULTS}
    extra_names = [
        "TRITON_WS_STORE_HELPER_WARPS",
        "TRITON_WS_STORE_HELPER_REGS",
        "TRITON_WS_LOAD_ACTIVATION_WARPS",
        "TRITON_WS_EPILOGUE_ROW_SUBTILE_FACTOR",
        "TRITON_WS_EPILOGUE_STORE_HELPER_DEPTH",
    ]
    prev.update({name: os.environ.get(name) for name in extra_names})

    os.environ.update(ENV_DEFAULTS)
    os.environ["TRITON_WS_STORE_HELPER_WARPS"] = str(config.store_helper_warps)
    os.environ["TRITON_WS_STORE_HELPER_REGS"] = str(config.store_helper_regs)
    os.environ["TRITON_WS_LOAD_ACTIVATION_WARPS"] = str(config.load_activation_warps)
    os.environ["TRITON_WS_EPILOGUE_ROW_SUBTILE_FACTOR"] = str(config.row_subtile_factor)
    os.environ["TRITON_WS_EPILOGUE_STORE_HELPER_DEPTH"] = str(config.helper_depth)
    os.environ["TRITON_WS_USE_WIDE_PACKED_STORE32"] = "1" if config.store_mode == "wide32" else "0"
    os.environ["TRITON_WS_USE_WIDE_PACKED_STORE64"] = "1" if config.store_mode == "wide64" else "0"
    try:
        yield
    finally:
        for name, value in prev.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def benchmark_config(
    prepared: bench.PreparedCase,
    config: EpiConfig,
    rep: int,
    peak_fp8_tflops: float,
    peak_mem_tbps: float,
    validation_reference: tuple[str, tuple[object, object]] | None,
) -> SweepResult:
    with override_env(config):
        try:
            result = bench.benchmark_prepared_case(
                prepared,
                bench.WS_OPTIMIZED_KERNEL_NAME,
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


def write_csv(path: Path, results: list[SweepResult], reruns: dict[EpiConfig, SweepResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "store_helper_warps",
                "load_activation_warps",
                "store_helper_regs",
                "row_subtile_factor",
                "helper_depth",
                "store_mode",
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
            rerun = reruns.get(result.config)
            writer.writerow(
                [
                    result.config.store_helper_warps,
                    result.config.load_activation_warps,
                    result.config.store_helper_regs,
                    result.config.row_subtile_factor,
                    result.config.helper_depth,
                    result.config.store_mode,
                    result.status,
                    "" if result.runtime_ms is None else f"{result.runtime_ms:.8f}",
                    "" if result.tflops is None else f"{result.tflops:.8f}",
                    "" if result.tbps is None else f"{result.tbps:.8f}",
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


def format_result(result: SweepResult) -> str:
    cfg = result.config
    if result.status == "ok":
        return (
            f"{cfg.store_helper_warps:>2} | {cfg.load_activation_warps:>2} | {cfg.store_helper_regs:>3} | "
            f"{cfg.row_subtile_factor:>2} | {cfg.helper_depth:>2} | {cfg.store_mode:>6} | "
            f"{result.runtime_ms:>8.4f} | {result.tflops:>8.2f} | {result.tbps:>6.2f} | "
            f"{'yes' if result.validated else 'no ':>3} | ok"
        )
    if result.required_shared is not None and result.shared_limit is not None:
        detail = f"{result.required_shared}>{result.shared_limit}"
    else:
        detail = result.status
    return (
        f"{cfg.store_helper_warps:>2} | {cfg.load_activation_warps:>2} | {cfg.store_helper_regs:>3} | "
        f"{cfg.row_subtile_factor:>2} | {cfg.helper_depth:>2} | {cfg.store_mode:>6} | "
        f"{'-':>8} | {'-':>8} | {'-':>6} | {'no ':>3} | {detail}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep ws_optimized epilogue/helper configurations on one prepared parrot-gather case."
    )
    parser.add_argument("--case-family", choices=("non-parrot", "parrot"), default="non-parrot")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--peak-fp8-tflops", type=float, default=bench.DEFAULT_PEAK_FP8_TFLOPS)
    parser.add_argument("--peak-mem-tbps", type=float, default=bench.DEFAULT_PEAK_MEM_TBPS)
    parser.add_argument("--rep", type=int, default=12)
    parser.add_argument("--rerun-rep", type=int, default=60)
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--helper-warps", type=str, default="1:4")
    parser.add_argument("--helper-regs", type=str, default="16,24,32")
    parser.add_argument("--row-factors", type=str, default="4,8,16")
    parser.add_argument("--depths", type=str, default="2,3,4")
    parser.add_argument("--store-modes", type=str, default="narrow,wide32,wide64")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> list[EpiConfig]:
    configs: list[EpiConfig] = []
    for helper_warps in parse_int_list(args.helper_warps):
        load_activation_warps = 6 - helper_warps
        if helper_warps < 1 or load_activation_warps < 1:
            continue
        for helper_regs in parse_int_list(args.helper_regs):
            for row_factor in parse_int_list(args.row_factors):
                for depth in parse_int_list(args.depths):
                    for store_mode in parse_store_modes(args.store_modes):
                        configs.append(
                            EpiConfig(
                                store_helper_warps=helper_warps,
                                store_helper_regs=helper_regs,
                                row_subtile_factor=row_factor,
                                helper_depth=depth,
                                store_mode=store_mode,
                            )
                        )
    if args.num_shards <= 0:
        raise ValueError(f"--num-shards must be > 0, got {args.num_shards}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")
    return [cfg for idx, cfg in enumerate(configs) if idx % args.num_shards == args.shard_index]


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise ValueError(f"Only cuda is supported, got {args.device}")
    if args.rep <= 0 or args.rerun_rep <= 0:
        raise ValueError("--rep and --rerun-rep must be > 0")
    if args.top < 0:
        raise ValueError("--top must be >= 0")

    cases = bench.make_cases(args.case_family, args.batch_size, args.batch_size, limit=1)
    if len(cases) != 1:
        raise ValueError(f"Expected exactly one case for batch_size={args.batch_size}, got {len(cases)}")
    case = cases[0]
    prepared = bench.prepare_case(case, device=args.device, seed=args.seed, local_rank_override=args.local_rank)
    validation_reference = None
    if args.validate:
        validation_reference = bench.build_validation_reference(prepared, "exact")

    print(
        f"case={case.case_id} rank={prepared.local_rank} helper-warps={args.helper_warps} "
        f"helper-regs={args.helper_regs} rows={args.row_factors} depths={args.depths} "
        f"store-modes={args.store_modes} shard={args.shard_index}/{args.num_shards}"
    )
    baseline = bench.benchmark_prepared_case(
        prepared,
        bench.WS_OPTIMIZED_KERNEL_NAME,
        warmup=0,
        rep=args.rerun_rep,
        peak_fp8_tflops=args.peak_fp8_tflops,
        peak_mem_tbps=args.peak_mem_tbps,
        validation_reference=validation_reference,
    )
    print(
        "baseline | "
        f"runtime_ms={baseline.runtime_ms:.4f} | tflops={baseline.tflops:.2f} | "
        f"tbps={baseline.tbps:.2f}"
    )

    configs = build_configs(args)
    print(
        "cfg_h | cfg_l | regs | rf | dp |  store |       ms |  TFLOP/s |   TB/s | val | status"
    )
    results: list[SweepResult] = []
    for cfg in configs:
        result = benchmark_config(
            prepared,
            cfg,
            rep=args.rep,
            peak_fp8_tflops=args.peak_fp8_tflops,
            peak_mem_tbps=args.peak_mem_tbps,
            validation_reference=validation_reference,
        )
        results.append(result)
        print(format_result(result))

    reruns: dict[EpiConfig, SweepResult] = {}
    ok_results = [result for result in results if result.status == "ok" and result.runtime_ms is not None]
    ok_results.sort(key=lambda result: result.runtime_ms)
    for result in ok_results[: args.top]:
        rerun = benchmark_config(
            prepared,
            result.config,
            rep=args.rerun_rep,
            peak_fp8_tflops=args.peak_fp8_tflops,
            peak_mem_tbps=args.peak_mem_tbps,
            validation_reference=validation_reference,
        )
        reruns[result.config] = rerun
        print(f"rerun | {result.config.label} | ms={rerun.runtime_ms:.4f}" if rerun.runtime_ms is not None else f"rerun | {result.config.label} | {rerun.status}")

    if args.csv_out is not None:
        write_csv(args.csv_out, results, reruns)
        print(f"csv={args.csv_out}")


if __name__ == "__main__":
    main()
