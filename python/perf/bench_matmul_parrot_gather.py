from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from triton.testing import do_bench

from triton_kernels.distributed import make_expt_dict_uniform
from triton_kernels.matmul import FlexCtx, FnSpecs, FusedActivation, PrecisionConfig, matmul
from triton_kernels.numerics import InFlexData, OutFlexData
from triton_kernels.numerics_details.mxfp import MXFP_BLOCK_SIZE, downcast_to_mxfp
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.tensor import FP4, RaggedTensorMetadata, Tensor, convert_layout, make_ragged_tensor_metadata, wrap_torch_tensor
from triton_kernels.tensor_details.layout import make_default_matmul_mxfp4_w_layout, make_default_matmul_mxfp4_w_scale_layout
from triton_kernels.topk import topk

# Default roofline constants for a single NVIDIA GB300 GPU.
# FP8 peak is inferred from the official GB300 NVL72 rack spec: 720 PFLOP/s
# FP8/FP6 Tensor Core with sparsity across 72 GPUs, or 5 PFLOP/s dense per GPU.
# HBM peak is aligned with NVIDIA's published "up to 8 TB/s" bandwidth for B300.
DEFAULT_PEAK_FP8_TFLOPS = 5_000.0
DEFAULT_PEAK_MEM_TBPS = 8.0


@dataclass(frozen=True)
class Params:
    n_expts_tot: int
    n_expts_shards: int
    n_expts_act: int
    b_shape: tuple[int, int]
    batch_sizes: tuple[int, ...]
    is_parrot_gather: bool

    def iter_cases(self) -> Iterable["Case"]:
        for batch_size in self.batch_sizes:
            yield Case(
                n_expts_tot=self.n_expts_tot,
                n_expts_shards=self.n_expts_shards,
                n_expts_act=self.n_expts_act,
                b_shape=self.b_shape,
                batch_size=batch_size,
                is_parrot_gather=self.is_parrot_gather,
            )


@dataclass(frozen=True)
class Case:
    n_expts_tot: int
    n_expts_shards: int
    n_expts_act: int
    b_shape: tuple[int, int]
    batch_size: int
    is_parrot_gather: bool

    @property
    def case_id(self) -> str:
        label = f"bs{self.batch_size}_E{self.n_expts_tot}_es{self.n_expts_shards}_B{self.b_shape[0]}x{self.b_shape[1]}"
        return f"{label}_parrot" if self.is_parrot_gather else label

    @property
    def k(self) -> int:
        return self.b_shape[0]

    @property
    def n(self) -> int:
        return self.b_shape[1]


@dataclass(frozen=True)
class BenchMetrics:
    n_tokens: int
    n_nonzero_experts: int
    flops: int
    bytes: int

    @property
    def tflops_per_ms(self) -> float:
        return self.flops / 1e9

    @property
    def tbps_per_ms(self) -> float:
        return self.bytes / 1e9


@dataclass(frozen=True)
class BenchResult:
    case: Case
    local_rank: int
    n_expts_local: int
    runtime_ms: float
    tflops: float
    tbps: float
    pct_peak_fp8_tflops: float
    pct_peak_mem_tbps: float
    peak_fp8_tflops: float
    peak_mem_tbps: float
    metrics: BenchMetrics
    expected_slice_size: int | None


LARGE_BATCH_SIZES = (
    1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2304, 2560,
    2816, 3072, 3328, 3584, 3840, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360,
    16384, 20480, 24576, 28672, 32768, 33792,
)

PARROT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 96, 128)

PARAMS = [
    Params(256, 8, 4, (5120, 10240), LARGE_BATCH_SIZES, False),
    Params(256, 16, 4, (5120, 10240), LARGE_BATCH_SIZES, False),
    Params(256, 32, 4, (5120, 10240), LARGE_BATCH_SIZES, False),
    Params(272, 8, 4, (5120, 10240), LARGE_BATCH_SIZES, False),
    Params(288, 8, 4, (5120, 10240), LARGE_BATCH_SIZES, False),
    Params(64, 1, 4, (1280, 2560), PARROT_BATCH_SIZES, True),
]


def alloc_randn(shape: tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
    if dtype.itemsize == 1:
        tmp = 2 ** -(torch.randint(4, 8, shape, device=device, dtype=torch.float16))
        return tmp.to(dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def alloc_randn_fp4(shape: tuple[int, ...], device: str) -> tuple[Tensor, Tensor]:
    data = alloc_randn(shape, torch.bfloat16, device)
    data, scale = downcast_to_mxfp(data, FP4, axis=1)  # type: ignore[arg-type]
    data_layout = make_default_matmul_mxfp4_w_layout(mx_axis=1)
    scale_layout = make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=8)
    data = convert_layout(wrap_torch_tensor(data, dtype=FP4), data_layout)
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout)
    return data, scale


def init_routing_data(
    batch_size: int,
    n_expts_tot: int,
    n_expts_act: int,
    local_rank: int,
    expt_dist: dict[int, list[int]],
    use_expected_slice_size: bool,
    device: str,
) -> tuple[RaggedTensorMetadata, torch.Tensor]:
    logits = torch.randn((batch_size, n_expts_tot), dtype=torch.float16, device=device)
    sparse_logits = topk(logits, n_expts_act, apply_softmax=True)
    expt_hist = sparse_logits.mask_metadata.col_sum

    local_expts = expt_dist[local_rank]
    local_expts_hist = expt_hist[local_expts]
    if local_expts_hist.sum() == 0:
        local_expts_hist[torch.randint(0, len(local_expts_hist), size=())] = 1

    ragged_batch_metadata = make_ragged_tensor_metadata(local_expts_hist, batch_size * n_expts_act)
    if use_expected_slice_size:
        ragged_batch_metadata.expected_slice_size = batch_size * n_expts_act // n_expts_tot

    combine_indx = sparse_logits.mask_metadata.col_sorted_indx
    gather_indx = torch.div(combine_indx, n_expts_act, rounding_mode="trunc")
    return ragged_batch_metadata, gather_indx


def to_storage_tensor(tensor: Tensor | torch.Tensor) -> torch.Tensor:
    return tensor.storage.data if isinstance(tensor, Tensor) else tensor


def normalize_flex_scale(scale: torch.Tensor) -> torch.Tensor:
    # Triton-kernels flex dataclasses expect len(scale) to be valid, unlike the
    # KI copies used in the original test harness.
    return scale.reshape(1) if scale.ndim == 0 else scale


def compute_matmul_proton_metrics(
    x: torch.Tensor,
    w: Tensor | torch.Tensor,
    y: torch.Tensor,
    ragged_metadata: RaggedTensorMetadata,
    n: int,
    k: int,
) -> BenchMetrics:
    # Match the ragged M-dimension accounting in triton_kernels.matmul_details._common.matmul_launch_metadata.
    slice_sizes = ragged_metadata.slice_sizes
    n_tokens = int(slice_sizes.sum().item())
    n_nonzero_experts = int((slice_sizes > 0).sum().item())

    x_data = to_storage_tensor(x)
    w_data = to_storage_tensor(w)
    y_data = to_storage_tensor(y)

    flops = int(2 * n_tokens * n * k)
    n_x_bytes = n_tokens * x_data.shape[-1] * x_data.element_size()
    n_y_bytes = n_tokens * y_data.shape[-1] * y_data.element_size()
    n_w_bytes = (w_data.numel() * w_data.element_size() // slice_sizes.numel()) * n_nonzero_experts
    return BenchMetrics(n_tokens=n_tokens, n_nonzero_experts=n_nonzero_experts, flops=flops, bytes=int(n_x_bytes + n_y_bytes + n_w_bytes))


def make_cases(case_family: str, min_batch_size: int | None, max_batch_size: int | None, limit: int | None) -> list[Case]:
    cases: list[Case] = []
    for param in PARAMS:
        if case_family == "parrot" and not param.is_parrot_gather:
            continue
        if case_family == "non-parrot" and param.is_parrot_gather:
            continue
        for case in param.iter_cases():
            if min_batch_size is not None and case.batch_size < min_batch_size:
                continue
            if max_batch_size is not None and case.batch_size > max_batch_size:
                continue
            cases.append(case)
    if limit is not None:
        cases = cases[:limit]
    return cases


def benchmark_case(
    case: Case,
    device: str,
    warmup: int,
    rep: int,
    seed: int,
    local_rank_override: int | None,
    peak_fp8_tflops: float,
    peak_mem_tbps: float,
) -> BenchResult:
    torch.manual_seed(seed)

    if local_rank_override is None:
        local_rank = int(torch.randint(0, case.n_expts_shards, (1,)).item())
    else:
        local_rank = local_rank_override
        if not (0 <= local_rank < case.n_expts_shards):
            raise ValueError(f"local_rank={local_rank} is out of range for n_expts_shards={case.n_expts_shards}")

    expt_dist = make_expt_dict_uniform(case.n_expts_shards, case.n_expts_tot)
    ragged_batch_metadata, gather_indx = init_routing_data(
        case.batch_size,
        case.n_expts_tot,
        case.n_expts_act,
        local_rank,
        expt_dist,
        use_expected_slice_size=not case.is_parrot_gather,
        device=device,
    )

    n_expts_local = case.n_expts_tot // case.n_expts_shards
    x = alloc_randn((case.batch_size, case.k), dtype=torch.float8_e4m3fn, device=device)
    w, w_scale = alloc_randn_fp4((n_expts_local, case.k, case.n), device=device)
    bias = alloc_randn((n_expts_local, case.n), dtype=torch.float32, device=device)

    swiglu_alpha = float(torch.rand((), device=device).item()) / 5 + 1.0
    swiglu_limit = float(torch.rand((), device=device).item()) / 5 + 1.3
    fused_activation = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (swiglu_alpha, swiglu_limit),
    )

    rand_scale = (lambda s: torch.rand((1,), device=device) + s) if case.is_parrot_gather else (lambda s: torch.rand((), device=device) + s)
    x_scale = normalize_flex_scale(rand_scale(0.5))
    y_scale = normalize_flex_scale(rand_scale(3.5))

    precision_config = PrecisionConfig(
        flexpoint_saturate_inf=True,
        b_mx_scale=w_scale,
        b_microblock_size=MXFP_BLOCK_SIZE.value,
        out_dtype=torch.float8_e4m3fn,
        flex_ctx=FlexCtx(
            lhs_data=InFlexData(dtype=torch.float8_e4m3fn, scale=x_scale),
            rhs_data=InFlexData(),
            out_data=OutFlexData(dtype=torch.float8_e4m3fn, expected_scale=y_scale),
        ),
    )

    out = torch.zeros(
        (1, case.batch_size * case.n_expts_act, case.n // fused_activation.specs.reduction_n),
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    def run() -> torch.Tensor:
        return matmul(
            a=x,
            b=w,  # type: ignore[arg-type]
            bias=bias,
            a_ragged_metadata=ragged_batch_metadata,
            gather_indx=gather_indx,
            precision_config=precision_config,
            c=out,
            fused_activation=fused_activation,
        )

    y = run()
    if y.dtype != out.dtype:
        raise RuntimeError(f"Expected output dtype {out.dtype}, got {y.dtype}")

    runtime_ms = float(do_bench(run, warmup=warmup, rep=rep))
    metrics = compute_matmul_proton_metrics(x, w, out, ragged_batch_metadata, n=case.n, k=case.k)
    tflops = metrics.flops / runtime_ms / 1e9
    tbps = metrics.bytes / runtime_ms / 1e9
    pct_peak_fp8_tflops = 100.0 * tflops / peak_fp8_tflops
    pct_peak_mem_tbps = 100.0 * tbps / peak_mem_tbps

    return BenchResult(
        case=case,
        local_rank=local_rank,
        n_expts_local=n_expts_local,
        runtime_ms=runtime_ms,
        tflops=tflops,
        tbps=tbps,
        pct_peak_fp8_tflops=pct_peak_fp8_tflops,
        pct_peak_mem_tbps=pct_peak_mem_tbps,
        peak_fp8_tflops=peak_fp8_tflops,
        peak_mem_tbps=peak_mem_tbps,
        metrics=metrics,
        expected_slice_size=ragged_batch_metadata.expected_slice_size,
    )


def format_result(result: BenchResult) -> str:
    case = result.case
    kind = "parrot" if case.is_parrot_gather else "non-parrot"
    return (
        f"{case.case_id:>34} | kind={kind:>11} | rank={result.local_rank:>2} | "
        f"E_local={result.n_expts_local:>3} | tokens={result.metrics.n_tokens:>6} | "
        f"nonzero_expts={result.metrics.n_nonzero_experts:>3} | "
        f"ms={result.runtime_ms:>8.4f} | TFLOP/s={result.tflops:>8.2f} | "
        f"TB/s={result.tbps:>6.2f} | fp8_roof={result.pct_peak_fp8_tflops:>6.2f}% | "
        f"hbm_roof={result.pct_peak_mem_tbps:>6.2f}%"
    )


def write_csv(path: Path, results: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "case_family",
                "batch_size",
                "n_expts_tot",
                "n_expts_shards",
                "n_expts_local",
                "n_expts_act",
                "k",
                "n",
                "local_rank",
                "expected_slice_size",
                "n_tokens",
                "n_nonzero_experts",
                "runtime_ms",
                "tflops",
                "tbps",
                "pct_peak_fp8_tflops",
                "pct_peak_mem_tbps",
                "peak_fp8_tflops",
                "peak_mem_tbps",
                "flops",
                "bytes",
            ]
        )
        for result in results:
            case = result.case
            writer.writerow(
                [
                    case.case_id,
                    "parrot" if case.is_parrot_gather else "non-parrot",
                    case.batch_size,
                    case.n_expts_tot,
                    case.n_expts_shards,
                    result.n_expts_local,
                    case.n_expts_act,
                    case.k,
                    case.n,
                    result.local_rank,
                    result.expected_slice_size,
                    result.metrics.n_tokens,
                    result.metrics.n_nonzero_experts,
                    f"{result.runtime_ms:.8f}",
                    f"{result.tflops:.8f}",
                    f"{result.tbps:.8f}",
                    f"{result.pct_peak_fp8_tflops:.8f}",
                    f"{result.pct_peak_mem_tbps:.8f}",
                    f"{result.peak_fp8_tflops:.8f}",
                    f"{result.peak_mem_tbps:.8f}",
                    result.metrics.flops,
                    result.metrics.bytes,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark triton_kernels.matmul on the parrot-gather workload grid derived from the reference test."
    )
    parser.add_argument("--case-family", choices=("all", "non-parrot", "parrot"), default="all")
    parser.add_argument("--min-batch-size", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--peak-fp8-tflops", type=float, default=DEFAULT_PEAK_FP8_TFLOPS)
    parser.add_argument("--peak-mem-tbps", type=float, default=DEFAULT_PEAK_MEM_TBPS)
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda":
        raise ValueError(f"Only cuda is supported, got {args.device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.peak_fp8_tflops <= 0:
        raise ValueError(f"--peak-fp8-tflops must be > 0, got {args.peak_fp8_tflops}")
    if args.peak_mem_tbps <= 0:
        raise ValueError(f"--peak-mem-tbps must be > 0, got {args.peak_mem_tbps}")

    cases = make_cases(args.case_family, args.min_batch_size, args.max_batch_size, args.limit)
    if not cases:
        raise ValueError("No cases matched the requested filters")

    results: list[BenchResult] = []
    for idx, case in enumerate(cases, start=1):
        result = benchmark_case(
            case,
            device=args.device,
            warmup=args.warmup,
            rep=args.rep,
            seed=args.seed,
            local_rank_override=args.local_rank,
            peak_fp8_tflops=args.peak_fp8_tflops,
            peak_mem_tbps=args.peak_mem_tbps,
        )
        results.append(result)
        print(f"[{idx:>3}/{len(cases):>3}] {format_result(result)}")

    if args.csv_out is not None:
        write_csv(args.csv_out, results)
        print(f"Wrote {len(results)} rows to {args.csv_out}")


if __name__ == "__main__":
    main()
