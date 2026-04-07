from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from triton_kernels.target_info import is_cuda
from triton_kernels.testing import assert_close


@lru_cache(maxsize=1)
def _load_parrot_gather_bench_module():
    module_name = "_test_matmul_parrot_gather_bench"
    module_path = Path(__file__).with_name("bench_matmul_parrot_gather.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_matmul_ogs_cases():
    return _load_parrot_gather_bench_module().make_cases("all", None, None, None)


def _get_test_device() -> str:
    if not is_cuda() or not torch.cuda.is_available():
        pytest.skip("Only supported on CUDA")
    device_index = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device_index)[0] < 10:
        pytest.skip("matmul_ogs benchmark shapes require a Blackwell-class CUDA GPU")
    return f"cuda:{device_index}"


@pytest.mark.parametrize("case", _build_matmul_ogs_cases(), ids=lambda case: case.case_id)
def test_matmul_ogs_matches_matmul(case):
    bench = _load_parrot_gather_bench_module()
    prepared = bench.prepare_case(case, device=_get_test_device(), seed=0, local_rank_override=0)

    ref_y, ref_precision = bench.run_case_once(prepared, bench.ORIGINAL_KERNEL_NAME)
    gluon_y, gluon_precision = bench.run_case_once(prepared, bench.GLUON_KERNEL_NAME)

    assert ref_y.shape == gluon_y.shape
    assert ref_y.dtype == gluon_y.dtype
    assert_close(
        ref_y.to(torch.float32),
        gluon_y.to(torch.float32),
        maxtol=3e-2,
        rmstol=None,
        description=case.case_id,
        verbose=False,
    )

    ref_scale = ref_precision.flex_ctx.out_data.actual_scale
    gluon_scale = gluon_precision.flex_ctx.out_data.actual_scale
    if ref_scale is not None or gluon_scale is not None:
        assert ref_scale is not None and gluon_scale is not None
        assert_close(
            ref_scale.to(torch.float32),
            gluon_scale.to(torch.float32),
            maxtol=1e-10,
            rmstol=1e-10,
            description=f"{case.case_id}:out_scale",
            verbose=False,
        )
