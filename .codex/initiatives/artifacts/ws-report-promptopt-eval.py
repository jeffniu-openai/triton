from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path

import torch
import triton
from triton_kernels.testing import assert_close


BASELINE_EXAMPLE = Path("/root/code/triton-ws-opt/python/examples/gluon/05-moe-bmm1-fused-gather.py")
REPRESENTATIVE_BATCHES = (128, 256, 512, 1024, 2048, 8192, 16384, 31744)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def time_kernel(run_fn, rep: int) -> float:
    for _ in range(10):
        run_fn()
    torch.cuda.synchronize()
    return float(triton.testing.do_bench_cudagraph(run_fn, rep=rep, return_mode="mean"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rep", type=int, default=1000)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_module(BASELINE_EXAMPLE, "baseline_ws_example")
    candidate = load_module(args.candidate, "candidate_ws_example")

    rows: list[dict[str, float | int | str]] = []
    speedups = []
    wins = 0

    for batch_size in REPRESENTATIVE_BATCHES:
        prepared = baseline.prepare_case(batch_size=batch_size, device=args.device, seed=0)
        ref_y, ref_precision = baseline.run_provider(prepared, "reference")

        def run_baseline():
            precision = baseline.make_precision_config(prepared)
            out = baseline.make_output_buffer(prepared)
            return baseline.matmul(
                a=prepared.x,
                b=prepared.w,
                bias=prepared.bias,
                a_ragged_metadata=prepared.ragged_metadata,
                gather_indx=prepared.gather_indx,
                precision_config=precision,
                c=out,
                fused_activation=prepared.fused_activation,
            )

        def run_candidate():
            precision = baseline.make_precision_config(prepared)
            out = baseline.make_output_buffer(prepared)
            return candidate.matmul(
                a=prepared.x,
                b=prepared.w,
                bias=prepared.bias,
                a_ragged_metadata=prepared.ragged_metadata,
                gather_indx=prepared.gather_indx,
                precision_config=precision,
                c=out,
                fused_activation=prepared.fused_activation,
            )

        cand_y = run_candidate()
        assert_close(
            ref_y.to(torch.float32),
            cand_y.to(torch.float32),
            maxtol=baseline.OUTPUT_MAXTOL,
            rmstol=baseline.OUTPUT_RMSTOL,
            description=f"promptopt:bs{batch_size}:out",
            verbose=False,
        )

        ref_scale = ref_precision.flex_ctx.out_data.actual_scale
        cand_precision = baseline.make_precision_config(prepared)
        cand_y = candidate.matmul(
            a=prepared.x,
            b=prepared.w,
            bias=prepared.bias,
            a_ragged_metadata=prepared.ragged_metadata,
            gather_indx=prepared.gather_indx,
            precision_config=cand_precision,
            c=baseline.make_output_buffer(prepared),
            fused_activation=prepared.fused_activation,
        )
        cand_scale = cand_precision.flex_ctx.out_data.actual_scale
        if ref_scale is not None or cand_scale is not None:
            assert ref_scale is not None and cand_scale is not None
            assert_close(
                ref_scale.to(torch.float32),
                cand_scale.to(torch.float32),
                maxtol=baseline.SCALE_TOL,
                rmstol=baseline.SCALE_TOL,
                description=f"promptopt:bs{batch_size}:out_scale",
                verbose=False,
            )

        baseline_ms = time_kernel(run_baseline, args.rep)
        candidate_ms = time_kernel(run_candidate, args.rep)
        speedup_pct = (baseline_ms / candidate_ms - 1.0) * 100.0
        if speedup_pct > 0:
            wins += 1
        speedups.append(speedup_pct)
        rows.append(
            {
                "batch_size": batch_size,
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "speedup_pct": speedup_pct,
            }
        )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_size", "baseline_ms", "candidate_ms", "speedup_pct"])
        writer.writeheader()
        writer.writerows(rows)

    mean_speedup = sum(speedups) / len(speedups)
    geom_speedup = math.exp(sum(math.log((100.0 + s) / 100.0) for s in speedups) / len(speedups)) - 1.0
    worst = min(speedups)
    print(
        {
            "candidate": str(args.candidate),
            "mean_speedup_pct": mean_speedup,
            "geom_speedup_pct": geom_speedup * 100.0,
            "wins": wins,
            "num_points": len(speedups),
            "worst_speedup_pct": worst,
            "csv": str(args.csv_out),
        }
    )


if __name__ == "__main__":
    main()
