from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Any, Iterator


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    kernels_root_str = str(repo_root / "python" / "triton_kernels")
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    if kernels_root_str not in sys.path:
        sys.path.insert(0, kernels_root_str)

    import triton.language.extra.libdevice as libdevice

    cuda_mod = sys.modules.get("triton.language.extra.cuda")
    if cuda_mod is None:
        cuda_mod = types.ModuleType("triton.language.extra.cuda")
        cuda_mod.__path__ = []
        sys.modules["triton.language.extra.cuda"] = cuda_mod
    cuda_mod.libdevice = libdevice
    sys.modules["triton.language.extra.cuda.libdevice"] = libdevice


_bootstrap_imports()

from triton.runtime.errors import OutOfResources

from python.perf import bench_matmul_parrot_gather as bench


SCHEDULES = {
    "direct": 0,
    "wavefront": 1,
}


@dataclass(frozen=True)
class SweepConfig:
    x_num_bufs: int
    w_num_bufs: int
    row_factor: int
    packed_final_fma: bool
    store_helper: bool
    schedule: str
    load_activation_warps: int
    load_weight_warps: int
    mma_warps: int
    store_helper_warps: int
    load_activation_regs: int
    load_weight_regs: int
    mma_regs: int
    store_helper_regs: int

    @property
    def label(self) -> str:
        fields = [
            f"x={self.x_num_bufs}",
            f"w={self.w_num_bufs}",
            f"row={self.row_factor}",
            f"pfma={int(self.packed_final_fma)}",
            f"helper={int(self.store_helper)}",
            f"sched={self.schedule}",
            f"aw={self.load_activation_warps}",
            f"ww={self.load_weight_warps}",
            f"mw={self.mma_warps}",
            f"srw={self.store_helper_warps}",
            f"ar={self.load_activation_regs}",
            f"wr={self.load_weight_regs}",
            f"mr={self.mma_regs}",
            f"srr={self.store_helper_regs}",
        ]
        return ",".join(fields)

    def env(self) -> dict[str, str]:
        return {
            "TRITON_WS_X_NUM_BUFS": str(self.x_num_bufs),
            "TRITON_WS_W_NUM_BUFS": str(self.w_num_bufs),
            "TRITON_WS_EPILOGUE_ROW_SUBTILE_FACTOR": str(self.row_factor),
            "TRITON_WS_USE_PACKED_FINAL_FMA": "1" if self.packed_final_fma else "0",
            "TRITON_WS_USE_EPILOGUE_STORE_HELPER": "1" if self.store_helper else "0",
            "TRITON_WS_EPILOGUE_SCHEDULE": str(SCHEDULES[self.schedule]),
            "TRITON_WS_LOAD_ACTIVATION_WARPS": str(self.load_activation_warps),
            "TRITON_WS_LOAD_WEIGHT_WARPS": str(self.load_weight_warps),
            "TRITON_WS_MMA_WARPS": str(self.mma_warps),
            "TRITON_WS_STORE_HELPER_WARPS": str(self.store_helper_warps),
            "TRITON_WS_LOAD_ACTIVATION_REGS": str(self.load_activation_regs),
            "TRITON_WS_LOAD_WEIGHT_REGS": str(self.load_weight_regs),
            "TRITON_WS_MMA_REGS": str(self.mma_regs),
            "TRITON_WS_STORE_HELPER_REGS": str(self.store_helper_regs),
        }


@dataclass(frozen=True)
class WorkerBaseline:
    gpu: int
    ws_optimized_ms: float
    gluon_optimized_ms: float


@dataclass(frozen=True)
class SweepResult:
    gpu: int
    config: SweepConfig
    status: str
    runtime_ms: float | None
    tflops: float | None
    tbps: float | None
    error: str | None = None
    required_shared: int | None = None
    shared_limit: int | None = None
    delta_vs_ws_ms: float | None = None
    delta_vs_gluon_ms: float | None = None


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
        stop_inclusive = stop + (1 if step > 0 else -1)
        values.extend(range(start, stop_inclusive, step))
    if not values:
        raise ValueError(f"Range spec produced no values: {spec!r}")
    return sorted(set(values))


def parse_bool_values(spec: str) -> list[bool]:
    values: list[bool] = []
    for value in parse_range(spec):
        if value not in (0, 1):
            raise ValueError(f"Boolean spec must contain only 0 or 1, got {value}")
        values.append(bool(value))
    return values


def parse_schedule_values(spec: str) -> list[str]:
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part not in SCHEDULES:
            raise ValueError(f"Unknown schedule {part!r}; expected one of {sorted(SCHEDULES)}")
        values.append(part)
    if not values:
        raise ValueError("Schedule spec produced no values")
    return values


def parse_gpu_list(spec: str) -> list[int]:
    values = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if not values:
        raise ValueError("GPU list produced no values")
    return values


def classify_error(exc: Exception) -> tuple[str, str, int | None, int | None]:
    if isinstance(exc, OutOfResources):
        if exc.name == "shared memory":
            return "oor_shared", str(exc), int(exc.required), int(exc.limit)
        return "oor", str(exc), int(exc.required), int(exc.limit)
    return "error", repr(exc), None, None


@contextmanager
def override_env(mapping: dict[str, str]) -> Iterator[None]:
    prev = {name: os.environ.get(name) for name in mapping}
    os.environ.update(mapping)
    try:
        yield
    finally:
        for name, value in prev.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def build_configs(args: argparse.Namespace) -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    seen: set[SweepConfig] = set()
    for x_num_bufs in parse_range(args.x_num_bufs):
        for w_num_bufs in parse_range(args.w_num_bufs):
            for row_factor in parse_range(args.row_factors):
                for packed_final_fma in parse_bool_values(args.packed_final_fma):
                    for store_helper in parse_bool_values(args.store_helper):
                        for schedule in parse_schedule_values(args.schedules):
                            if store_helper:
                                store_helper_warp_values = parse_range(args.store_helper_warps)
                                store_helper_reg_values = parse_range(args.store_helper_regs)
                            else:
                                store_helper_warp_values = [2]
                                store_helper_reg_values = [32]
                            for load_activation_warps in parse_range(args.load_activation_warps):
                                for load_weight_warps in parse_range(args.load_weight_warps):
                                    for mma_warps in parse_range(args.mma_warps):
                                        for store_helper_warps in store_helper_warp_values:
                                            if store_helper:
                                                if row_factor == 1:
                                                    continue
                                                if (
                                                    store_helper_warps
                                                    + load_activation_warps
                                                    + load_weight_warps
                                                    + mma_warps
                                                    > 8
                                                ):
                                                    continue
                                            elif load_activation_warps + load_weight_warps + mma_warps > 8:
                                                continue
                                            for load_activation_regs in parse_range(args.load_activation_regs):
                                                for load_weight_regs in parse_range(args.load_weight_regs):
                                                    for mma_regs in parse_range(args.mma_regs):
                                                        for store_helper_regs in store_helper_reg_values:
                                                            config = SweepConfig(
                                                                x_num_bufs=x_num_bufs,
                                                                w_num_bufs=w_num_bufs,
                                                                row_factor=row_factor,
                                                                packed_final_fma=packed_final_fma,
                                                                store_helper=store_helper,
                                                                schedule=schedule,
                                                                load_activation_warps=load_activation_warps,
                                                                load_weight_warps=load_weight_warps,
                                                                mma_warps=mma_warps,
                                                                store_helper_warps=store_helper_warps,
                                                                load_activation_regs=load_activation_regs,
                                                                load_weight_regs=load_weight_regs,
                                                                mma_regs=mma_regs,
                                                                store_helper_regs=store_helper_regs,
                                                            )
                                                            if config in seen:
                                                                continue
                                                            seen.add(config)
                                                            configs.append(config)
    return configs


def select_case(batch_size: int, n_expts_tot: int, n_expts_shards: int) -> bench.Case:
    matches = [
        case
        for case in bench.make_cases("non-parrot", batch_size, batch_size, limit=None)
        if case.n_expts_tot == n_expts_tot and case.n_expts_shards == n_expts_shards
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one case for batch_size={batch_size}, E={n_expts_tot}, es={n_expts_shards}; got {len(matches)}"
        )
    return matches[0]


def benchmark_kernel(
    prepared: bench.PreparedCase,
    kernel_name: str,
    rep: int,
) -> bench.BenchResult:
    return bench.benchmark_prepared_case(
        prepared,
        kernel_name,
        warmup=0,
        rep=rep,
        peak_fp8_tflops=bench.DEFAULT_PEAK_FP8_TFLOPS,
        peak_mem_tbps=bench.DEFAULT_PEAK_MEM_TBPS,
    )


def benchmark_config(
    prepared: bench.PreparedCase,
    gpu: int,
    config: SweepConfig,
    rep: int,
    baseline: WorkerBaseline,
) -> SweepResult:
    with override_env(config.env()):
        try:
            result = benchmark_kernel(prepared, bench.WS_OPTIMIZED_KERNEL_NAME, rep=rep)
        except Exception as exc:
            status, message, required, limit = classify_error(exc)
            return SweepResult(
                gpu=gpu,
                config=config,
                status=status,
                runtime_ms=None,
                tflops=None,
                tbps=None,
                error=message,
                required_shared=required,
                shared_limit=limit,
            )
    return SweepResult(
        gpu=gpu,
        config=config,
        status="ok",
        runtime_ms=result.runtime_ms,
        tflops=result.tflops,
        tbps=result.tbps,
        delta_vs_ws_ms=result.runtime_ms - baseline.ws_optimized_ms,
        delta_vs_gluon_ms=result.runtime_ms - baseline.gluon_optimized_ms,
    )


def run_worker(
    gpu: int,
    configs: list[SweepConfig],
    batch_size: int,
    n_expts_tot: int,
    n_expts_shards: int,
    rep: int,
    seed: int,
    local_rank: int,
) -> dict[str, Any]:
    case = select_case(batch_size=batch_size, n_expts_tot=n_expts_tot, n_expts_shards=n_expts_shards)
    prepared = bench.prepare_case(case, device="cuda", seed=seed, local_rank_override=local_rank)
    ws_baseline = benchmark_kernel(prepared, bench.WS_OPTIMIZED_KERNEL_NAME, rep=rep)
    gluon_baseline = benchmark_kernel(prepared, bench.GLUON_OPTIMIZED_KERNEL_NAME, rep=rep)
    baseline = WorkerBaseline(
        gpu=gpu,
        ws_optimized_ms=ws_baseline.runtime_ms,
        gluon_optimized_ms=gluon_baseline.runtime_ms,
    )
    results = [benchmark_config(prepared, gpu=gpu, config=config, rep=rep, baseline=baseline) for config in configs]
    return {
        "gpu": gpu,
        "case_id": case.case_id,
        "baseline": asdict(baseline),
        "results": [
            {
                **asdict(result),
                "config": asdict(result.config),
            }
            for result in results
        ],
    }


def write_csv(path: Path, baselines: dict[int, WorkerBaseline], results: list[SweepResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "gpu",
                "status",
                "runtime_ms",
                "delta_vs_ws_ms",
                "delta_vs_gluon_ms",
                "tflops",
                "tbps",
                "required_shared",
                "shared_limit",
                "error",
                "baseline_ws_ms",
                "baseline_gluon_ms",
                "x_num_bufs",
                "w_num_bufs",
                "row_factor",
                "packed_final_fma",
                "store_helper",
                "schedule",
                "load_activation_warps",
                "load_weight_warps",
                "mma_warps",
                "store_helper_warps",
                "load_activation_regs",
                "load_weight_regs",
                "mma_regs",
                "store_helper_regs",
            ]
        )
        for result in results:
            baseline = baselines[result.gpu]
            writer.writerow(
                [
                    result.gpu,
                    result.status,
                    "" if result.runtime_ms is None else f"{result.runtime_ms:.8f}",
                    "" if result.delta_vs_ws_ms is None else f"{result.delta_vs_ws_ms:.8f}",
                    "" if result.delta_vs_gluon_ms is None else f"{result.delta_vs_gluon_ms:.8f}",
                    "" if result.tflops is None else f"{result.tflops:.8f}",
                    "" if result.tbps is None else f"{result.tbps:.8f}",
                    result.required_shared or "",
                    result.shared_limit or "",
                    result.error or "",
                    f"{baseline.ws_optimized_ms:.8f}",
                    f"{baseline.gluon_optimized_ms:.8f}",
                    result.config.x_num_bufs,
                    result.config.w_num_bufs,
                    result.config.row_factor,
                    int(result.config.packed_final_fma),
                    int(result.config.store_helper),
                    result.config.schedule,
                    result.config.load_activation_warps,
                    result.config.load_weight_warps,
                    result.config.mma_warps,
                    result.config.store_helper_warps,
                    result.config.load_activation_regs,
                    result.config.load_weight_regs,
                    result.config.mma_regs,
                    result.config.store_helper_regs,
                ]
            )


def print_summary(baselines: dict[int, WorkerBaseline], results: list[SweepResult], top: int) -> None:
    print("same-GPU baselines")
    print("gpu | ws_opt_ms | gluon_opt_ms | delta_ms")
    for gpu in sorted(baselines):
        baseline = baselines[gpu]
        delta = baseline.ws_optimized_ms - baseline.gluon_optimized_ms
        print(
            f"{gpu:>3} | {baseline.ws_optimized_ms:>9.4f} | "
            f"{baseline.gluon_optimized_ms:>12.4f} | {delta:>+8.4f}"
        )

    ok_results = [result for result in results if result.status == "ok" and result.runtime_ms is not None]
    ok_results.sort(key=lambda result: (result.delta_vs_gluon_ms, result.runtime_ms))
    print()
    print(f"top {min(top, len(ok_results))} configs by same-GPU delta vs gluon_optimized")
    print("rank | gpu | delta_gluon | delta_ws | runtime_ms | config")
    for rank, result in enumerate(ok_results[:top], start=1):
        print(
            f"{rank:>4} | {result.gpu:>3} | {result.delta_vs_gluon_ms:>+11.4f} | "
            f"{result.delta_vs_ws_ms:>+8.4f} | {result.runtime_ms:>10.4f} | {result.config.label}"
        )


def run_driver(args: argparse.Namespace) -> None:
    configs = build_configs(args)
    gpus = parse_gpu_list(args.gpus)
    if len(configs) == 0:
        raise ValueError("No configs to sweep")

    chunks: list[list[SweepConfig]] = [[] for _ in gpus]
    for idx, config in enumerate(configs):
        chunks[idx % len(gpus)].append(config)

    worker_payloads: list[Path] = []
    worker_outputs: list[Path] = []
    procs: list[subprocess.Popen[str]] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="wsopt-sweep-"))
    try:
        for gpu, chunk in zip(gpus, chunks, strict=True):
            payload_path = tmpdir / f"gpu{gpu}_configs.json"
            output_path = tmpdir / f"gpu{gpu}_results.json"
            payload_path.write_text(
                json.dumps([asdict(config) for config in chunk], indent=2),
                encoding="utf-8",
            )
            worker_payloads.append(payload_path)
            worker_outputs.append(output_path)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [
                sys.executable,
                __file__,
                "--worker",
                "--gpu",
                str(gpu),
                "--worker-config-json",
                str(payload_path),
                "--worker-output-json",
                str(output_path),
                "--batch-size",
                str(args.batch_size),
                "--n-expts-tot",
                str(args.n_expts_tot),
                "--n-expts-shards",
                str(args.n_expts_shards),
                "--rep",
                str(args.rep),
                "--seed",
                str(args.seed),
                "--local-rank",
                str(args.local_rank),
            ]
            procs.append(subprocess.Popen(cmd, env=env, text=True))

        for proc in procs:
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"Worker exited with status {ret}")

        baselines: dict[int, WorkerBaseline] = {}
        results: list[SweepResult] = []
        for output_path in worker_outputs:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            baseline = WorkerBaseline(**payload["baseline"])
            baselines[baseline.gpu] = baseline
            for row in payload["results"]:
                config = SweepConfig(**row["config"])
                result_data = dict(row)
                result_data["config"] = config
                results.append(SweepResult(**result_data))

        print(f"swept {len(configs)} configs across {len(gpus)} GPUs")
        print_summary(baselines, results, top=args.top)
        if args.csv_out is not None:
            write_csv(args.csv_out, baselines, results)
            print(f"wrote {len(results)} rows to {args.csv_out}")
    finally:
        for path in worker_payloads + worker_outputs:
            if path.exists():
                path.unlink()
        if tmpdir.exists():
            tmpdir.rmdir()


def run_worker_cli(args: argparse.Namespace) -> None:
    configs = [SweepConfig(**row) for row in json.loads(Path(args.worker_config_json).read_text(encoding="utf-8"))]
    payload = run_worker(
        gpu=args.gpu,
        configs=configs,
        batch_size=args.batch_size,
        n_expts_tot=args.n_expts_tot,
        n_expts_shards=args.n_expts_shards,
        rep=args.rep,
        seed=args.seed,
        local_rank=args.local_rank,
    )
    Path(args.worker_output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep ws_optimized epilogue and worker-partition configs on the target non-parrot 16K case."
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-config-json", type=Path, default=None)
    parser.add_argument("--worker-output-json", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--n-expts-tot", type=int, default=256)
    parser.add_argument("--n-expts-shards", type=int, default=8)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rep", type=int, default=12)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--x-num-bufs", type=str, default="5")
    parser.add_argument("--w-num-bufs", type=str, default="4")
    parser.add_argument("--row-factors", type=str, default="8,16")
    parser.add_argument("--packed-final-fma", type=str, default="1")
    parser.add_argument("--store-helper", type=str, default="0,1")
    parser.add_argument("--schedules", type=str, default="direct")
    parser.add_argument("--load-activation-warps", type=str, default="4,5")
    parser.add_argument("--load-weight-warps", type=str, default="1,2")
    parser.add_argument("--mma-warps", type=str, default="1")
    parser.add_argument("--store-helper-warps", type=str, default="2")
    parser.add_argument("--load-activation-regs", type=str, default="96,112")
    parser.add_argument("--load-weight-regs", type=str, default="48,64,80")
    parser.add_argument("--mma-regs", type=str, default="24")
    parser.add_argument("--store-helper-regs", type=str, default="24,32")
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rep <= 0:
        raise ValueError(f"--rep must be > 0, got {args.rep}")
    if args.top <= 0:
        raise ValueError(f"--top must be > 0, got {args.top}")
    if args.worker:
        if args.worker_config_json is None or args.worker_output_json is None:
            raise ValueError("--worker requires --worker-config-json and --worker-output-json")
        run_worker_cli(args)
        return
    run_driver(args)


if __name__ == "__main__":
    main()
