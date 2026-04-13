#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from triton_kernels.testing import assert_close


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("ws_promptopt_candidate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real kernel sanity check for a promptopt candidate.")
    parser.add_argument("--candidate", type=Path, required=True, help="Path to candidate example file.")
    parser.add_argument("--batch", type=int, default=128, help="Batch size for the sanity run.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device, e.g. cuda:0")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for prepare_case")
    args = parser.parse_args()

    module = load_module(args.candidate.resolve())
    prepared = module.prepare_case(batch_size=args.batch, device=args.device, seed=args.seed)

    example_y, example_precision = module.run_provider(prepared, "example")
    reference_y, reference_precision = module.run_provider(prepared, "reference")

    # Compare decoded float32 outputs using the example's own published tolerances.
    assert_close(
        reference_y.to(torch.float32),
        example_y.to(torch.float32),
        maxtol=module.OUTPUT_MAXTOL,
        rmstol=module.OUTPUT_RMSTOL,
        description=f"promptopt-sanity:bs{args.batch}:out",
        verbose=False,
    )
    example_scale = example_precision.flex_ctx.out_data.actual_scale
    reference_scale = reference_precision.flex_ctx.out_data.actual_scale
    if example_scale is not None or reference_scale is not None:
        assert example_scale is not None and reference_scale is not None
        assert_close(
            reference_scale.to(torch.float32),
            example_scale.to(torch.float32),
            maxtol=module.SCALE_TOL,
            rmstol=module.SCALE_TOL,
            description=f"promptopt-sanity:bs{args.batch}:scale",
            verbose=False,
        )

    print(
        {
            "candidate": str(args.candidate),
            "batch": args.batch,
            "device": args.device,
            "shape": tuple(example_y.shape),
            "dtype": str(example_y.dtype),
        }
    )


if __name__ == "__main__":
    main()
