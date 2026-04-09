from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import traceback
from typing import Iterator

import torch
from triton.runtime.errors import OutOfResources

from . import bench_matmul_parrot_gather as bench


HELPER_MODE_DIRECT = "direct"
HELPER_MODE_STORE_ONLY = "store_only"
HELPER_MODE_PACKED_OUT = "packed_out"
HELPER_MODE_HELPER_SWIGLU = "helper_swiglu"
HELPER_MODES = (
    HELPER_MODE_DIRECT,
    HELPER_MODE_STORE_ONLY,
    HELPER_MODE_PACKED_OUT,
    HELPER_MODE_HELPER_SWIGLU,
)


ENV_KEYS = (
    "TRITON_WS_X_NUM_BUFS",
    "TRITON_WS_W_NUM_BUFS",
    "TRITON_WS_EPILOGUE_ROW_SUBTILE_FACTOR",
    "TRITON_WS_EPILOGUE_SCHEDULE",
    "TRITON_WS_EPILOGUE_WARPS_N",
    "TRITON_WS_USE_WIDE_PACKED_STORE32",
    "TRITON_WS_USE_WIDE_PACKED_STORE64",
    "TRITON_WS_USE_FRAGMENT_BIAS_SCALE",
    "TRITON_WS_USE_EPILOGUE_STORE_HELPER",
    "TRITON_WS_EPILOGUE_STORE_HELPER_DEPTH",
    "TRITON_WS_USE_HELPER_PACKED_OUT_BUFFER",
    "TRITON_WS_USE_HELPER_SWIGLU",
    "TRITON_WS_STORE_HELPER_WARPS",
    "TRITON_WS_STORE_HELPER_REGS",
)


@dataclass(frozen=True)
class SweepConfig:
    helper_mode: str
    row_factor: int
    helper_depth: int
    schedule: int
    wide_store32: bool
    fragment_bias: bool
    x_num_bufs: int
    w_num_bufs: int
    helper_warps: int
    helper_regs: int

    @property
    def label(self) -> str:
        flags = [
            f"mode={self.helper_mode}",
            f"row={self.row_factor}",
            f"depth={self.helper_depth}",
            f"schedule={self.schedule}",
            f"wide32={int(self.wide_store32)}",
            f"frag_bias={int(self.fragment_bias)}",
            f"x={self.x_num_bufs}",
            f"w={self.w_num_bufs}",
            f"hwarps={self.helper_warps}",
            f"hregs={self.helper_regs}",
        ]
        return ",".join(flags)

    def env(self) -> dict[str, str]:
        use_helper = self.helper_mode != HELPER_MODE_DIRECT
        return {
            "TRITON_WS_X_NUM_BUFS": str(self.x_num_bufs),
            "TRITON_WS_W_NUM_BUFS": str(self.w_num_bufs),
            "TRITON_WS_EPILOGUE_ROW_SUBTILE_FACTOR": str(self.row_factor),
            "TRITON_WS_EPILOGUE_SCHEDULE": str(self.schedule),
            "TRITON_WS_USE_WIDE_PACKED_STORE32": "1" if self.wide_store32 else "0",
            "TRITON_WS_USE_FRAGMENT_BIAS_SCALE": "1" if self.fragment_bias else "0",
            "TRITON_WS_USE_EPILOGUE_STORE_HELPER": "1" if use_helper else "0",
            "TRITON_WS_EPILOGUE_STORE_HELPER_DEPTH": str(self.helper_depth),
            "TRITON_WS_USE_HELPER_PACKED_OUT_BUFFER": "1" if self.helper_mode == HELPER_MODE_PACKED_OUT else "0",
            "TRITON_WS_USE_HELPER_SWIGLU": "1" if self.helper_mode == HELPER_MODE_HELPER_SWIGLU else "0",
            "TRITON_WS_STORE_HELPER_WARPS": str(self.helper_warps),
            "TRITON_WS_STORE_HELPER_REGS": str(self.helper_regs),
        }


@dataclass(frozen=True)
class SweepResult:
    config: SweepConfig
    device_index: int
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
        pieces = [int(field) for field in part.split(":")]
        if len(pieces) not in (2, 3):
            raise ValueError(f"Invalid range spec {part!r}")
        start, stop = pieces[:2]
        step = pieces[2] if len(pieces) == 3 else 1
        if step == 0:
            raise ValueError(f"Invalid zero step in range spec {part!r}")
        stop_inclusive = stop + 1 if step > 0 else stop - 1
        values.extend(range(start, stop_inclusive, step))
    if not values:
        raise ValueError(f"Range spec {spec!r} produced no values")
    return sorted(set(values))


def parse_devices(spec: str) -> list[int]:
    devices = parse_int_list(spec)
    available = torch.cuda.device_count()
    for device in devices:
        if device < 0 or device >= available:
            raise ValueError(f"Device {device} is out of range for {available} visible GPUs")
    return devices


def parse_bool_list(spec: str) -> list[bool]:
    values: list[bool] = []
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in ("1", "true", "t", "yes", "y", "on"):
            values.append(True)
        elif part in ("0", "false", "f", "no", "n", "off"):
            values.append(False)
        else:
            raise ValueError(f"Invalid boolean value {part!r}")
    if not values:
        raise ValueError(f"Boolean list {spec!r} produced no values")
    return list(dict.fromkeys(values))


def parse_helper_modes(spec: str) -> list[str]:
    modes: list[str] = []
    for part in spec.split(","):
        mode = part.strip()
        if not mode:
            continue
        if mode not in HELPER_MODES:
            raise ValueError(f"Unknown helper mode {mode!r}; expected one of {HELPER_MODES}")
        modes.append(mode)
    if not modes:
        raise ValueError("No helper modes selected")
    return list(dict.fromkeys(modes))


@contextmanager
def override_env(overrides: dict[str, str]) -> Iterator[None]:
    prev = {key: os.environ.get(key) for key in ENV_KEYS}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def classify_error(exc: Exception) -> tuple[str, str, int | None, int | None]:
    if isinstance(exc, OutOfResources):
        if exc.name == "shared memory":
            return "oor_shared", str(exc), int(exc.required), int(exc.limit)
        return "oor", str(exc), int(exc.required), int(exc.limit)
    return "error", repr(exc), None, None


def benchmark_config(
    prepared: bench.PreparedCase,
    config: SweepConfig,
    rep: int,
    peak_fp8_tflops: float,
    peak_mem_tbps: float,
    validation_reference: tuple[str, tuple[torch.Tensor, bench.PrecisionConfig]] | None,
) -> SweepResult:
    with override_env(config.env()):
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
                device_index=torch.cuda.current_device(),
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
        device_index=torch.cuda.current_device(),
        status="ok",
        runtime_ms=result.runtime_ms,
        tflops=result.tflops,
        tbps=result.tbps,
        validated=result.validated,
    )


def generate_configs(args: argparse.Namespace) -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for helper_mode in parse_helper_modes(args.helper_modes):
        for row_factor in parse_int_list(args.row_factors):
            for x_num_bufs in parse_int_list(args.x_range):
                for w_num_bufs in parse_int_list(args.w_range):
                    for schedule in parse_int_list(args.schedules):
                        for wide_store32 in parse_bool_list(args.wide_store32):
                            for fragment_bias in parse_bool_list(args.fragment_bias):
                                if helper_mode == HELPER_MODE_DIRECT:
                                    configs.append(
                                        SweepConfig(
                                            helper_mode=helper_mode,
                                            row_factor=row_factor,
                                            helper_depth=1,
                                            schedule=schedule,
                                            wide_store32=wide_store32,
                                            fragment_bias=fragment_bias,
                                            x_num_bufs=x_num_bufs,
                                            w_num_bufs=w_num_bufs,
                                            helper_warps=2,
                                            helper_regs=16,
                                        )
                                    )
                                    continue
                                for helper_depth in parse_int_list(args.helper_depths):
                                    for helper_warps in parse_int_list(args.helper_warps):
                                        for helper_regs in parse_int_list(args.helper_regs):
                                            configs.append(
                                                SweepConfig(
                                                    helper_mode=helper_mode,
                                                    row_factor=row_factor,
                                                    helper_depth=helper_depth,
                                                    schedule=schedule,
                                                    wide_store32=wide_store32,
                                                    fragment_bias=fragment_bias,
                                                    x_num_bufs=x_num_bufs,
                                                    w_num_bufs=w_num_bufs,
                                                    helper_warps=helper_warps,
                                                    helper_regs=helper_regs,
                                                )
                                            )
    return configs


def format_result(result: SweepResult) -> str:
    cfg = result.config
    if result.status == "ok":
        return (
            f"gpu={result.device_index} | {cfg.helper_mode:>11} | row={cfg.row_factor:>2} | "
            f"depth={cfg.helper_depth:>2} | schedule={cfg.schedule} | wide={int(cfg.wide_store32)} | "
            f"frag_bias={int(cfg.fragment_bias)} | x={cfg.x_num_bufs} | w={cfg.w_num_bufs} | "
            f"hwarps={cfg.helper_warps} | hregs={cfg.helper_regs:>2} | "
            f"{result.runtime_ms:>8.4f} ms | {result.tflops:>8.2f} TF | {result.tbps:>6.2f} TB | "
            f"val={'yes' if result.validated else 'no'}"
        )
    if result.required_shared is not None and result.shared_limit is not None:
        return (
            f"gpu={result.device_index} | {cfg.label} | {result.status} "
            f"({result.required_shared}>{result.shared_limit})"
        )
    return f"gpu={result.device_index} | {cfg.label} | {result.status}"


def prepare_for_device(
    case: bench.Case,
    device_index: int,
    seed: int,
    local_rank: int | None,
    validation_reference_name: str | None,
) -> tuple[
    bench.PreparedCase,
    tuple[str, tuple[torch.Tensor, bench.PrecisionConfig]] | None,
]:
    device = f"cuda:{device_index}"
    torch.cuda.set_device(device_index)
    prepared = bench.prepare_case(case, device=device, seed=seed, local_rank_override=local_rank)
    validation_reference = None
    if validation_reference_name is not None:
        validation_reference = bench.build_validation_reference(prepared, validation_reference_name)
    return prepared, validation_reference


def worker_main(
    device_index: int,
    case: bench.Case,
    configs: list[SweepConfig],
    seed: int,
    local_rank: int | None,
    rep: int,
    peak_fp8_tflops: float,
    peak_mem_tbps: float,
    validation_reference_name: str | None,
    queue: mp.Queue,
) -> None:
    try:
        prepared, validation_reference = prepare_for_device(
            case=case,
            device_index=device_index,
            seed=seed,
            local_rank=local_rank,
            validation_reference_name=validation_reference_name,
        )
        for config in configs:
            result = benchmark_config(
                prepared=prepared,
                config=config,
                rep=rep,
                peak_fp8_tflops=peak_fp8_tflops,
                peak_mem_tbps=peak_mem_tbps,
                validation_reference=validation_reference,
            )
            queue.put(("result", result))
        queue.put(("done", device_index))
    except Exception:
        queue.put(("worker_error", device_index, traceback.format_exc()))


def write_csv(path: Path, results: list[SweepResult], rerun_results: dict[SweepConfig, SweepResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "helper_mode",
                "row_factor",
                "helper_depth",
                "schedule",
                "wide_store32",
                "fragment_bias",
                "x_num_bufs",
                "w_num_bufs",
                "helper_warps",
                "helper_regs",
                "device_index",
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
                    result.config.helper_mode,
                    result.config.row_factor,
                    result.config.helper_depth,
                    result.config.schedule,
                    int(result.config.wide_store32),
                    int(result.config.fragment_bias),
                    result.config.x_num_bufs,
                    result.config.w_num_bufs,
                    result.config.helper_warps,
                    result.config.helper_regs,
                    result.device_index,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep ws_optimized epilogue helper/direct variants across multiple GPUs."
    )
    parser.add_argument("--case-family", choices=("non-parrot", "parrot"), default="non-parrot")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--peak-fp8-tflops", type=float, default=bench.DEFAULT_PEAK_FP8_TFLOPS)
    parser.add_argument("--peak-mem-tbps", type=float, default=bench.DEFAULT_PEAK_MEM_TBPS)
    parser.add_argument("--rep", type=int, default=10, help="Broad-pass cudagraph reps per config.")
    parser.add_argument("--rerun-rep", type=int, default=60, help="Longer rerun reps for top configs.")
    parser.add_argument("--top", type=int, default=12, help="Number of configs to rerun on GPU 0.")
    parser.add_argument("--helper-modes", type=str, default="store_only,packed_out,direct")
    parser.add_argument("--row-factors", type=str, default="8,16")
    parser.add_argument("--helper-depths", type=str, default="1,2")
    parser.add_argument("--schedules", type=str, default="0,1")
    parser.add_argument("--wide-store32", type=str, default="0,1")
    parser.add_argument("--fragment-bias", type=str, default="0,1")
    parser.add_argument("--x-range", type=str, default="4,5")
    parser.add_argument("--w-range", type=str, default="4")
    parser.add_argument("--helper-warps", type=str, default="2,4")
    parser.add_argument("--helper-regs", type=str, default="16,24")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--validation-reference",
        choices=(bench.ORIGINAL_KERNEL_NAME, bench.EXACT_REFERENCE_NAME),
        default=bench.EXACT_REFERENCE_NAME,
    )
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    devices = parse_devices(args.devices)
    configs = generate_configs(args)
    if not configs:
        raise ValueError("No sweep configs generated")

    print(
        f"case={case.case_id} devices={devices} configs={len(configs)} rep={args.rep} "
        f"rerun_rep={args.rerun_rep} validate={args.validate}"
    )

    chunks = [[] for _ in devices]
    for idx, config in enumerate(configs):
        chunks[idx % len(devices)].append(config)

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=worker_main,
            args=(
                device_index,
                case,
                chunk,
                args.seed,
                args.local_rank,
                args.rep,
                args.peak_fp8_tflops,
                args.peak_mem_tbps,
                args.validation_reference if args.validate else None,
                queue,
            ),
        )
        for device_index, chunk in zip(devices, chunks)
        if chunk
    ]

    for proc in procs:
        proc.start()

    remaining = len(procs)
    results: list[SweepResult] = []
    completed = 0
    print("progress | result")
    while remaining > 0:
        item = queue.get()
        kind = item[0]
        if kind == "result":
            result = item[1]
            results.append(result)
            completed += 1
            print(f"[{completed:>4}/{len(configs):>4}] | {format_result(result)}")
        elif kind == "done":
            remaining -= 1
        elif kind == "worker_error":
            _, device_index, tb = item
            for proc in procs:
                proc.join(timeout=0.1)
            raise RuntimeError(f"worker on device {device_index} failed:\n{tb}")
        else:
            raise RuntimeError(f"Unknown queue item kind {kind!r}")

    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"worker exited with code {proc.exitcode}")

    ok_results = [result for result in results if result.status == "ok" and result.runtime_ms is not None]
    ok_results.sort(key=lambda result: result.runtime_ms)

    rerun_results: dict[SweepConfig, SweepResult] = {}
    if args.top > 0 and ok_results:
        top_results = ok_results[: min(args.top, len(ok_results))]
        rerun_device = devices[0]
        prepared, validation_reference = prepare_for_device(
            case=case,
            device_index=rerun_device,
            seed=args.seed,
            local_rank=args.local_rank,
            validation_reference_name=args.validation_reference if args.validate else None,
        )
        print(f"rerunning top {len(top_results)} configs on gpu={rerun_device} with rep={args.rerun_rep}")
        print("baseline | kernel          |       ms |  TFLOP/s |   TB/s")
        for kernel_name in (
            bench.WS_KERNEL_NAME,
            bench.WS_OPTIMIZED_KERNEL_NAME,
            bench.GLUON_OPTIMIZED_KERNEL_NAME,
        ):
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

        print("rank | broad_ms | rerun_ms | config")
        for rank, broad in enumerate(top_results, start=1):
            rerun = benchmark_config(
                prepared=prepared,
                config=broad.config,
                rep=args.rerun_rep,
                peak_fp8_tflops=args.peak_fp8_tflops,
                peak_mem_tbps=args.peak_mem_tbps,
                validation_reference=validation_reference,
            )
            rerun_results[broad.config] = rerun
            rerun_ms = "-" if rerun.runtime_ms is None else f"{rerun.runtime_ms:8.4f}"
            broad_ms = "-" if broad.runtime_ms is None else f"{broad.runtime_ms:8.4f}"
            print(f"{rank:>4} | {broad_ms} | {rerun_ms} | {broad.config.label}")

    if ok_results:
        print("top broad-pass configs")
        print("rank |       ms | config")
        for rank, result in enumerate(ok_results[: min(15, len(ok_results))], start=1):
            print(f"{rank:>4} | {result.runtime_ms:>8.4f} | {result.config.label}")
    else:
        print("no valid configs completed")

    if args.csv_out is not None:
        write_csv(args.csv_out, results, rerun_results)
        print(f"wrote {len(results)} rows to {args.csv_out}")


if __name__ == "__main__":
    main()
