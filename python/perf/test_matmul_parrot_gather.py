import pytest
import torch

from . import bench_matmul_parrot_gather as BENCH
from triton_kernels.target_info import is_cuda
from triton_kernels.testing import assert_close

KERNEL_NAMES = BENCH.iter_kernel_names(BENCH.DEFAULT_KERNEL_MODE)
CASES = BENCH.make_cases("all", None, None, None)


def _get_test_device() -> str:
    if not is_cuda() or not torch.cuda.is_available():
        pytest.skip("Only supported on CUDA")
    device_index = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device_index)[0] < 10:
        pytest.skip("matmul_ogs benchmark shapes require a Blackwell-class CUDA GPU")
    return f"cuda:{device_index}"


def _run_case_outputs(case):
    prepared = BENCH.prepare_case(case, device=_get_test_device(), seed=0, local_rank_override=0)
    return {kernel_name: BENCH.run_case_once(prepared, kernel_name) for kernel_name in KERNEL_NAMES}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_matmul_ogs_matches_matmul(case):
    outputs = _run_case_outputs(case)
    ref_y, ref_precision = outputs[BENCH.ORIGINAL_KERNEL_NAME]
    gluon_y, gluon_precision = outputs[BENCH.GLUON_KERNEL_NAME]
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
